# Architecture — signature-cropping

Code-aligned spec for the signature-cropping service. The long research
brief lives in `temp/implementation_plan.md`; this document is the
contract the code follows.

---

## 1. Goals & constraints

| Item | Value |
|---|---|
| Function | Crop every handwritten signature from a scanned form (PDF / PNG / JPEG / TIFF) |
| Compute | AWS ECS Fargate, x86_64, 2 vCPU / 4 GB, **CPU only** |
| Interfaces | FastAPI REST (port 8080), MCP stdio (process) |
| P50 / P95 / P99 latency (A4, 300 DPI) | 220 / 400 / 650 ms |
| Cold start to `/readyz` | ≤ 8 s |
| Image size | ≤ 700 MB |
| Memory ceiling | ≤ 1.5 GB RSS / task |
| License posture | Apache-2.0 stack only (no AGPL Ultralytics weights) |

## 2. Pipeline

```
bytes ──▶ ingest ──▶ preprocess ──▶ heuristics ──▶ detector ──▶ postprocess ──▶ encode ──▶ response
```

| Stage | Module | Responsibility | Library |
|---|---|---|---|
| ingest | `pipeline/ingest.py` | Sniff MIME; PDF→pages, image→ndarray | PyMuPDF, Pillow, OpenCV |
| preprocess | `pipeline/preprocess.py` | OSD orientation, Hough/projection deskew (±5°), CLAHE, letterbox 640×640 | OpenCV |
| heuristics | `pipeline/heuristics.py` | CCL + stroke-density + form-line removal → candidate ROIs | OpenCV |
| detector | `pipeline/detector.py` | RT-DETRv2-S INT8, ORT + OpenVINO EP, lazy single-session | onnxruntime-openvino |
| postprocess | `pipeline/postprocess.py` | Score threshold, class-aware NMS, padding, mask, coord remap | OpenCV, NumPy |
| encode | `pipeline/encode.py` | PNG (RGBA optional), base64 inline or S3 PUT | Pillow, boto3 |

**Latency budget (P95, 2 vCPU x86, INT8 model, A4 @ 300 DPI):**

| Stage | Budget |
|---|---|
| ingest | 30 ms |
| preprocess | 50 ms |
| heuristics | 25 ms |
| detector | 220 ms |
| postprocess | 15 ms |
| encode | 20 ms |
| reserve | 40 ms |
| **total** | **≤ 400 ms** |

Multi-page PDFs run pages sequentially within one request; the per-page
budget above applies. Concurrency comes from ECS task count, not threading
inside a request — the ONNX session uses `intra_op_num_threads = vCPU`.

## 3. Detector choice

| Option | License | mAP@50 | CPU latency | Decision |
|---|---|---|---|---|
| **Conditional-DETR-R50 (`tech4humans/conditional-detr-50-signature-detector`)** | Apache-2.0 | 0.937 | 400–600 ms | **MVP — off-the-shelf, no training** |
| RT-DETRv2-S (fine-tuned) | Apache-2.0 | ~0.94+ (post fine-tune) | 250–400 ms (INT8) | **Post-MVP** primary, once HSBC samples are labelled |
| YOLOv11s (Ultralytics) | **AGPL-3.0** | 0.94 | 150 ms | Forbidden (license) |
| Heuristics only | n/a | low recall | < 25 ms | Pre-filter only |

The heuristic stage runs **before** the detector and prunes obviously
empty regions (zero candidates → skip the detector for that page). It
never overrides the detector. If the detector is unavailable, the
service returns `MODEL_UNAVAILABLE` (503) rather than falling back to
heuristics blindly.

## 4. REST API

Base URL: `https://signature-cropper.internal/v1`. Auth via mTLS at the
ALB plus a service-account bearer token (HMAC-signed JWT) validated by
the app.

### `POST /v1/crop-signature`

`multipart/form-data`:

| Field | Type | Required | Default |
|---|---|---|---|
| `file` | binary | yes | — |
| `confidence_threshold` | float | no | 0.55 |
| `padding_pct` | float | no | 0.08 |
| `apply_mask` | bool | no | false |
| `return_format` | enum(`inline_b64`, `s3`) | no | `inline_b64` |
| `s3_prefix` | string | when `return_format=s3` | — |
| `request_id` | string | no | server-generated |

**200 response:**

```json
{
  "request_id": "req_01HZ...",
  "model_version": "rtdetrv2-s-int8-2026.05.07",
  "page_count": 3,
  "signatures": [
    {
      "id": "sig_01HZ...",
      "page": 1,
      "bbox": { "x": 1240, "y": 1850, "w": 460, "h": 180 },
      "bbox_normalized": { "x": 0.51, "y": 0.74, "w": 0.19, "h": 0.07 },
      "confidence": 0.974,
      "crop_b64": "iVBORw0KGgo...",
      "crop_s3_uri": null,
      "mask_applied": false
    }
  ],
  "timing_ms": { "preprocess": 42, "inference": 187, "postprocess": 11 }
}
```

**Other routes:**

| Route | Purpose |
|---|---|
| `GET /healthz` | Liveness (always 200 if process up) |
| `GET /readyz` | Returns 200 only after ONNX session warm-up |
| `GET /v1/model` | Active model version, lineage hash, metrics |
| `GET /metrics` | Prometheus exposition |

## 5. MCP server

FastMCP, stdio transport. Three tools:

| Tool | Input | Output |
|---|---|---|
| `crop_signature` | `{file_uri, options?}` | Full response equivalent to REST 200 |
| `list_signature_regions` | `{file_uri}` | `{regions:[{page,bbox,confidence}]}` (coordinates only, no pixels) |
| `get_model_info` | — | `{model_version, training_lineage_hash, metrics, license}` |

`file_uri` accepts `file://`, `s3://`, and `data:application/pdf;base64,...`.
The MCP tools delegate to the same pipeline module as the REST layer; no
duplicated logic.

## 6. Error contract

Both REST and MCP surface this envelope (REST wraps it as the JSON body
of the non-2xx response; MCP returns it as a tool error):

```json
{
  "error_code": "PAGE_UNREADABLE",
  "message": "Page 2 failed PyMuPDF rasterization",
  "request_id": "req_01HZ...",
  "retryable": false
}
```

| Code | HTTP | Retryable | Cause |
|---|---|---|---|
| `INVALID_MIME` | 400 | no | Unsupported file type |
| `CORRUPT_FILE` | 400 | no | Parse failure |
| `PAYLOAD_TOO_LARGE` | 413 | no | > 25 MB sync, > 200 MB async |
| `PAGE_UNREADABLE` | 422 | no | Per-page render/decode failure |
| `NO_SIGNATURE_FOUND` | 200 | n/a | Empty `signatures: []` — not an error |
| `LOW_CONTRAST` | 422 | no | Image below quality floor for detection |
| `MODEL_UNAVAILABLE` | 503 | yes | ONNX session not loaded |
| `RATE_LIMITED` | 429 | yes | Token bucket exhausted |

## 7. Acceptance criteria

| Metric | MVP (Conditional-DETR off-the-shelf) | Production (fine-tuned RT-DETRv2-S INT8) |
|---|---|---|
| Precision, clear scans (≥ 250 DPI) | ≥ 0.95 | ≥ 0.99 |
| Recall, clear scans | ≥ 0.93 | ≥ 0.98 |
| Precision, noisy / overlapping ink | ≥ 0.88 | ≥ 0.95 |
| Recall, noisy | ≥ 0.85 | ≥ 0.93 |
| mAP@50 on held-out HSBC test set | ≥ 0.90 | ≥ 0.94 |
| Latency P50 / P95 / P99 (A4, 2 vCPU) | 380 / 600 / 900 ms | 220 / 400 / 650 ms |
| Cold start to `/readyz` | ≤ 12 s | ≤ 8 s |
| Memory RSS | ≤ 2.0 GB | ≤ 1.5 GB |
| Container image size | ≤ 900 MB | ≤ 700 MB |

Drift gate: nightly canary set; alert if mAP@50 drops > 2 points.

## 8. Deployment

ECS Fargate, x86_64, 2 vCPU / 4 GB, min 2 tasks. Internal ALB only,
ACM TLS, mTLS. Auto-scale on `RequestCountPerTarget` and CPU @ 60%.
S3 (KMS) for request payloads larger than inline base64 caps. Model
weights bake into the image — **no network egress at inference time**.

## 9. License posture

Detector: RT-DETRv2 (Apache-2.0). Datasets: Tobacco-800, tech4humans
signature-detection (both Apache-2.0) plus in-house HSBC samples. The
Ultralytics YOLO stack is AGPL-3.0 and excluded unless an Enterprise
License is procured. See `temp/implementation_plan.md` §3 for the full
analysis.

## 10. Out of scope

- Signature verification / matching (this is a cropper, not a comparator).
- OCR of other form fields.
- Human-in-the-loop review UI.
- Training pipeline beyond the export-and-quantize scripts in `scripts/`.
