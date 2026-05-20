# Signature Cropping from HSBC Account Opening Forms — Research & Implementation Plan

**Target deployment:** AWS ECS, x86_64, CPU-only (no GPU)
**Interface:** REST API + MCP server
**Author:** Technical research brief
**Date:** May 2026

-----

## 1. Executive Summary

**Recommendation:** A fine-tuned **RT-DETRv2 (small)** or **YOLOv11s / YOLOv8s** detector exported to **ONNX with OpenVINO Execution Provider**, wrapped in a **FastAPI** service that is also exposed as an **MCP server via FastMCP**, deployed on **AWS ECS Fargate (x86, c7i-equivalent)** behind an internal ALB.

**Why this stack:**

|Concern                                 |Choice                                                                                                           |Why                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
|Accuracy                                |Fine-tune on tech4humans/Tobacco800 + 500–1,000 in-house HSBC samples                                            |Public baselines reach mAP@50 ≈ 0.94 on similar forms               |
|CPU efficiency                          |ONNX Runtime + OpenVINO EP + INT8 quantization                                                                   |2–4× speedup on Intel Xeon vs vanilla PyTorch                       |
|License compliance (critical for a bank)|**RT-DETR / Conditional-DETR / D-FINE / YOLO-NAS (Apache 2.0)** rather than Ultralytics YOLOv8/v11 (**AGPL-3.0**)|AGPL-3.0 forces source disclosure of the “network-using” application|
|Deployment                              |ECS Fargate behind internal ALB                                                                                  |No GPU needed; auto-scaling; no EC2 patching                        |
|Two interfaces                          |FastAPI + FastMCP mounted on same process                                                                        |Single binary, single model load, both transports                   |

**Expected production numbers** (per A4 page, 300 DPI, single x86 vCPU, INT8 model):

- Latency: **150–350 ms** per page
- Throughput per task (2 vCPU, 4 GB): **6–12 pages/sec sustained**
- Accuracy on held-out HSBC test set (target): **precision ≥ 0.96, recall ≥ 0.95, mAP@50 ≥ 0.94**

> ⚠️ **License warning up front:** Ultralytics YOLOv5/v8/v11 weights and the Ultralytics training/inference code are released under **AGPL-3.0**. Deploying them as an internal network service inside HSBC without an Ultralytics Enterprise License is a real legal risk — AGPL extends GPL’s copyleft to network services. The plan below uses Apache-2.0 architectures as the default and treats YOLOv8/v11 only as an optional “if-Enterprise-license-is-acquired” path.

-----

## 2. Problem Restatement

You need to take a scanned page (or multi-page PDF) of an HSBC account-opening form and return cropped images of every handwritten signature on it. Concretely:

- **Input:** PDF or image (PNG/JPEG/TIFF), typically 200–400 DPI, A4 or Letter, sometimes mildly skewed, occasionally with blue ink / faded ink / overlap with printed lines.
- **Output:** N cropped signature images, with bounding-box coordinates on the source page, a confidence score, and (optionally) the page number.
- **Constraints:** x86 CPU on AWS ECS, no GPU, must be exposed as both a REST API and an MCP server.
- **Sensitivity:** Signatures are PII / biometric-like data. Logging, retention, and network egress controls apply.

-----

## 3. Technology Landscape (with what current research shows)

### 3.1 Classical CV (baseline only)

Connected-component analysis + heuristics (size, ink ratio, stroke continuity) — e.g. `ahmetozlu/signature_extractor` on GitHub. **Verdict:** Useful as a sanity-check fallback and for *post-detection mask cleanup*, but precision/recall on real bank forms is brittle — printed signatures of “John Smith” in script font, stamps, and dense form fields fool it. Don’t use as the primary detector.

### 3.2 Deep object-detection models (the right tier)

Two open benchmarks dominate the public literature for this exact task: **Tobacco-800** and the merged **signature-detection** dataset by tech4humans (2,819 document images, mixed sources, Apache 2.0 dataset).

Published numbers on that test set (tech4humans benchmark, ONNX Runtime on CPU):

|Model               |Architecture   |License                              |mAP@50      |mAP@50:95   |CPU latency (ONNX)|Params|
|--------------------|---------------|-------------------------------------|------------|------------|------------------|------|
|YOLOv8s (fine-tuned)|CNN one-stage  |**AGPL-3.0**                         |**0.945**   |0.674       |~172 ms           |11M   |
|YOLOv11s            |CNN+attn       |**AGPL-3.0**                         |~0.94       |~0.66       |~150 ms           |9M    |
|Conditional-DETR-R50|Transformer    |**Apache-2.0**                       |0.937       |0.653       |~400–600 ms       |~43M  |
|RT-DETR-R18 / v2-S  |Transformer    |**Apache-2.0**                       |~0.93 (est.)|~0.65 (est.)|~250–400 ms       |20M   |
|D-FINE-S / RF-DETR-S|Transformer    |**Apache-2.0**                       |comparable  |comparable  |comparable        |20–30M|
|YOLO-NAS-S          |NAS-derived CNN|Apache 2.0 (code) / weights = special|~0.93 (est.)|—           |~120–180 ms       |12M   |

Source: Open-Source Handwritten Signature Detection paper/blog by Samuel Lima Braz / tech4humans (Hugging Face, March 2025), which fine-tuned 21 architectures on the same dataset.

**Key empirical findings from that work, which directly inform our pick:**

1. **Smaller models matched larger ones** on this task — signatures are mid-sized objects and don’t need a big backbone.
1. **Pure CNN one-stage detectors (YOLO family) ran 2–4× faster on CPU** than transformer-based DETR variants at similar accuracy.
1. **Hyperparameter tuning (Optuna) added ~8% F1** over default training.
1. **OpenVINO Execution Provider** in ONNX Runtime was the single biggest CPU speedup lever.

### 3.3 Foundation-model OCR (PaddleOCR-VL, Donut, etc.)

PaddleOCR-VL 0.9B (Oct 2025) advertises signature/checkbox detection inside a unified VLM. **Verdict:** Overkill for this task on CPU — a 0.9B VLM is 50–100× slower than a 10M-param YOLO on the same hardware, and you’d be paying for capability you don’t need. Useful only if you *also* need OCR of the entire form in the same call.

### 3.4 Managed services (AWS Textract, Google Document AI)

Textract has a `SIGNATURES` feature type. **Verdict:** Pragmatic, but (a) it sends bank-customer signatures to a third-party endpoint — usually a non-starter for HSBC’s data-sovereignty review, (b) it’s a per-page cost that scales with volume, and (c) the bounding boxes are good but coarser than a custom detector. Worth keeping as a **comparison oracle** during model evaluation, not as the production path.

### 3.5 What this means for the recommendation

If license were free, YOLOv11s + OpenVINO is clearly the sweet spot.

For a **bank deploying on its own infrastructure**, the choice narrows to:

- **Primary recommendation: RT-DETRv2-S** (Apache-2.0, transformer, modern, well-supported in HF Transformers, exportable to ONNX). Slightly slower than YOLO on CPU but the gap closes with quantization, and the license is unambiguous.
- **Alternative: train a YOLOv8-architecture model from scratch using a re-implementation under Apache 2.0** (e.g. Keras-CV’s YOLOv8). This is legally cleaner than using Ultralytics weights but requires more engineering effort and a slightly larger dataset for parity.
- **Fastest-path alternative if HSBC has (or will obtain) an Ultralytics Enterprise License:** fine-tune YOLOv11s, export to ONNX+OpenVINO, ship. This is the smoothest engineering path and the one with the most third-party tooling.

The rest of this document assumes the **RT-DETRv2-S** primary path, with notes where YOLOv11s would differ.

-----

## 4. Recommended End-to-End Architecture

```
                   ┌────────────────────────────────────────────────┐
                   │              AWS Account / VPC                 │
                   │                                                │
   PDF/image  ──▶  │  ALB (internal) ──▶  ECS Fargate Service       │
                   │                       (2+ tasks, x86, 2 vCPU)  │
                   │                          │                     │
                   │                          ▼                     │
                   │  ┌───────────────────────────────────────────┐ │
                   │  │  Container: signature-cropper             │ │
                   │  │   ├─ FastAPI  (REST,  port 8080)          │ │
                   │  │   ├─ FastMCP  (HTTP,  port 8081)          │ │
                   │  │   ├─ Pre-proc: pdf→img, deskew, denoise   │ │
                   │  │   ├─ Inference: ORT + OpenVINO EP (INT8)  │ │
                   │  │   ├─ Post-proc: NMS, padding, crop, mask  │ │
                   │  │   └─ Output writer (PNG bytes / S3 PUT)   │ │
                   │  └───────────────────────────────────────────┘ │
                   │       │                            │           │
                   │       ▼                            ▼           │
                   │   S3 (input/output, KMS)     CloudWatch logs   │
                   │       │                            │           │
                   │       ▼                                        │
                   │   ECR (image)   Secrets Manager (config)       │
                   └────────────────────────────────────────────────┘
```

### 4.1 Inference pipeline (inside the container)

```
input bytes
   │
   ▼
[1] file-type detect (magic bytes)
   │
   ├── PDF ── PyMuPDF (fitz) ─▶ render each page @ 300 DPI ─▶ ndarray
   └── img ─ PIL/OpenCV decode ─▶ ndarray
   │
   ▼
[2] preprocess
   • deskew (Hough-line or projection-profile, <±5° correction)
   • orientation check (text-direction OCR snippet → rotate 0/90/180/270)
   • normalize: resize long-edge to 1024px, letterbox to 640×640 for model
   • optional: contrast stretch (CLAHE) if mean intensity high / low
   │
   ▼
[3] detector inference
   • RT-DETRv2-S INT8, ONNX Runtime, OpenVINO EP, intra_op_num_threads = vCPU
   • single batch, dynamic input
   │
   ▼
[4] postprocess
   • score threshold (default 0.55, configurable)
   • class-aware NMS (single class "signature")
   • map letterbox coords → original page coords
   • add padding (default 8% of bbox shorter side, configurable)
   • optional: connected-component mask within crop to remove printed form-line bleed
   │
   ▼
[5] output
   • for each signature: PNG (RGBA, transparent background if mask=true), bbox JSON,
     confidence, page number, source_id
   • upload to S3 (if configured) or return base64 in response
```

-----

## 5. Implementation Plan — Phased

### Phase 0 — Data & Legal (1–2 weeks)

- Confirm with HSBC Legal whether AGPL-3.0 is workable. If not (likely), lock the architecture to Apache-2.0 (RT-DETR or equivalent).
- Stand up a **labelling environment** (CVAT or Label Studio) inside the HSBC VPC. Do **not** use Roboflow’s hosted product — uploading customer-signed account-opening forms to a third party will fail compliance review.
- Collect ~500–1,000 in-house pages (anonymized / a representative sample), 70/15/15 split. Combine with public Tobacco-800 and tech4humans/signature-detection (both Apache-2.0 datasets) for pre-training breadth.
- Define labelling guidance: single class `signature`; label every handwritten signature including initials; **do not** label printed name labels, stamps, or pre-printed flourishes.

### Phase 1 — Model training (2 weeks)

- Base model: `PekingU/rtdetr_v2_r18vd` (or `r34vd`) from Hugging Face, Apache 2.0.
- Framework: HF Transformers + PyTorch (CPU training is OK for fine-tune, but a 1× T4 or A10 spot instance for training will cut wall-clock to a few hours).
- Training recipe:
  - Input size 640×640, letterbox
  - 60 epochs, AdamW, lr 1e-4 backbone / 1e-3 head, cosine schedule
  - Augmentations: random rotation ±5°, brightness/contrast ±15%, Gaussian noise, JPEG compression (50–95 quality), random erasing on text regions (not on signature regions)
  - Optuna 20-trial hyperparameter sweep on dropout, lr0, box-loss weight (as tech4humans demonstrated, this adds ~7–8% F1)
- Acceptance gate for this phase: **mAP@50 ≥ 0.92 on the held-out HSBC test split.**

### Phase 2 — Optimization for CPU (1 week)

- Export to ONNX (opset 17): `torch.onnx.export(...)` or use HF Optimum.
- Optimize with `onnxruntime.tools.optimizer` (constant folding, fusion).
- INT8 post-training quantization with `onnxruntime.quantization` using a 200–500 image calibration set drawn from the train split.
- Validate accuracy degradation is ≤ 1.5 mAP@50 points; if greater, switch to **QDQ + per-channel weight quantization** or fall back to FP32.
- Bench on the target instance (Fargate 2 vCPU, 4 GB) with ORT + OpenVINO EP; target ≤ 350 ms p95 per page.

### Phase 3 — Service implementation (1.5 weeks)

- Single Python service (`app/`):
  - `app/inference.py` — model wrapper, lazy load, thread-safe
  - `app/preprocess.py` — PDF rendering, deskew, normalization
  - `app/postprocess.py` — NMS, padding, masking, cropping
  - `app/api.py` — FastAPI REST routes (§6)
  - `app/mcp_server.py` — FastMCP tools (§7)
  - `app/main.py` — process entry: starts uvicorn on 8080 (FastAPI) and FastMCP on 8081 in the same event loop
  - `app/config.py` — Pydantic BaseSettings (env-driven)
  - `app/logging.py` — structured JSON logs to stdout, no signature pixels in logs

### Phase 4 — AWS deployment (1 week)

- Dockerfile: `python:3.12-slim`, multi-stage, final image target < 700 MB. Install `onnxruntime-openvino`, `pymupdf`, `pillow`, `opencv-python-headless`, `fastapi`, `fastmcp`, `uvicorn[standard]`.
- ECR repository with image scanning enabled.
- ECS Fargate task definition: 2 vCPU, 4 GB, x86_64 platform.
- ECS Service: min 2 tasks (HA), target-tracking auto-scaling on `RequestCountPerTarget` and `CPUUtilization` (target 60%).
- Internal Application Load Balancer (no public exposure), TLS via ACM, mutual TLS or signed-URL auth to ALB.
- VPC: private subnets, no NAT egress to the open internet (model weights bake into the image).
- S3 buckets: server-side encryption with KMS CMK, bucket policy restricts to the task role only, lifecycle to delete inputs/outputs after N hours.
- IAM task role: minimal — `s3:GetObject`/`PutObject` only on the designated bucket prefix, `logs:*` to one log group.

### Phase 5 — Testing, governance & rollout (1.5 weeks)

- Unit tests (preproc, postproc, NMS), integration tests (golden image fixtures), load tests (Locust, target 100 RPS sustained on 4-task cluster).
- Model card with training data lineage, evaluation metrics, known failure modes, version.
- Drift monitor: nightly job that runs the model on a fixed canary set and alerts if mAP drifts > 2 points.
- Shadow rollout: run new service in parallel with whatever process exists today, diff outputs on 1,000 real forms, only cut over when agreement ≥ 99%.

**Total timeline:** ~7–8 weeks for a small team (1 ML engineer + 1 backend engineer + part-time SRE), excluding legal review.

-----

## 6. REST API Specification

Base URL: `https://signature-cropper.internal.hsbc/v1`

Authentication: mTLS at the ALB plus a service-account bearer token validated by the app (HMAC-signed JWT, short TTL).

### 6.1 `POST /v1/signatures:extract`

Synchronous extraction for a single file ≤ 25 MB.

**Request** (`multipart/form-data`):

|Field                 |Type                    |Required                |Description                                      |
|----------------------|------------------------|------------------------|-------------------------------------------------|
|`file`                |binary                  |yes                     |PDF, PNG, JPEG, or TIFF                          |
|`confidence_threshold`|float                   |no, default 0.55        |Detection score cutoff                           |
|`padding_pct`         |float                   |no, default 0.08        |Padding around bbox, fraction of shorter side    |
|`apply_mask`          |bool                    |no, default false       |If true, return RGBA with form-line bleed removed|
|`return_format`       |enum(`inline_b64`, `s3`)|no, default `inline_b64`|Where to put crop bytes                          |
|`s3_prefix`           |string                  |conditional             |Required when `return_format=s3`                 |
|`request_id`          |string                  |no                      |Client-supplied idempotency key                  |

**Response 200** (`application/json`):

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
      "crop_b64": "iVBORw0KGgoAAAANSUhEUg...",   // present iff return_format=inline_b64
      "crop_s3_uri": null,                       // present iff return_format=s3
      "mask_applied": false
    }
  ],
  "timing_ms": { "preprocess": 42, "inference": 187, "postprocess": 11 }
}
```

**Errors:**

|Code|Meaning                                                |
|----|-------------------------------------------------------|
|400 |Bad input (unsupported MIME, corrupt PDF, file > 25 MB)|
|401 |Auth missing/invalid                                   |
|413 |Payload too large                                      |
|422 |Page failed preprocessing (e.g. unreadable)            |
|429 |Rate-limited                                           |
|500 |Internal error (model load failure, etc.)              |
|503 |Backend overloaded (queue-shed)                        |

### 6.2 `POST /v1/signatures:extract_async`

Asynchronous variant for larger PDFs (up to 200 MB). Returns `202` with a `job_id`; client polls `GET /v1/jobs/{job_id}` or subscribes to an SNS topic if pre-configured. Useful for back-office batch ingestion.

### 6.3 `GET /v1/healthz` and `GET /v1/readyz`

Standard liveness/readiness probes for ECS. `readyz` only returns 200 once the ONNX session is loaded and a warm-up inference has completed.

### 6.4 `GET /v1/model`

Returns the active model version, training-data lineage hash, and last-evaluated metrics. Useful for audit.

### 6.5 `GET /v1/metrics`

Prometheus exposition format: request count, latency histogram, detection-count distribution, model load status.

-----

## 7. MCP Server Specification

Built with **FastMCP** (the standard Python MCP framework, now part of the official MCP Python SDK). The MCP server runs in the same process as the FastAPI app and reuses the same loaded model and the same internal `inference.run()` function — there is no second model copy, no second container.

Transport: **Streamable HTTP** (recommended over stdio for a cloud service) on a separate port behind the same ALB, with a distinct path prefix `/mcp/`.

### 7.1 Tools exposed

```python
# app/mcp_server.py  (sketch)

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from .inference import run_extraction

mcp = FastMCP(name="hsbc-signature-cropper")

class ExtractParams(BaseModel):
    file_uri: str = Field(description="s3:// URI of the PDF/image to process")
    confidence_threshold: float = 0.55
    padding_pct: float = 0.08
    apply_mask: bool = False

@mcp.tool(
    annotations={
        "title": "Extract signatures from a document",
        "readOnlyHint": False,   # writes crops to S3
        "openWorldHint": False,  # only touches the configured bucket
    }
)
async def extract_signatures(params: ExtractParams) -> dict:
    """Detect every handwritten signature on a scanned form and return
    cropped PNGs plus bounding boxes. Input must be in the approved S3
    bucket; output crops are written to the same bucket under a
    per-request prefix."""
    return await run_extraction(
        source=params.file_uri,
        threshold=params.confidence_threshold,
        padding=params.padding_pct,
        mask=params.apply_mask,
        sink="s3",
    )

@mcp.tool(annotations={"title": "Model info", "readOnlyHint": True})
def model_info() -> dict:
    """Return active model version, training lineage hash, and last
    evaluation metrics."""
    ...

@mcp.tool(annotations={"title": "Health check", "readOnlyHint": True})
def healthz() -> dict:
    """Liveness probe — returns ok=True if the model session is loaded."""
    ...
```

### 7.2 Why expose this as MCP at all?

The MCP surface lets an agent (Claude, an internal copilot) invoke signature extraction as part of a larger workflow — e.g. “for the 12 PDFs in this case folder, pull out the signatures and put them next to the customer record.” This is precisely the right use of MCP: a small, well-typed, side-effectful tool with bounded permissions. The REST API remains the contract for non-agent callers (the document-processing batch system, RPA bots, etc.).

### 7.3 MCP-specific safety rules

- The MCP tool **only** accepts `s3://` URIs inside the pre-approved bucket prefix. No file-path inputs, no raw bytes (forces all data through the audited storage path).
- The MCP tool **does not** return base64 crop bytes to the agent — only S3 URIs and bounding-box metadata. This keeps signature pixels out of LLM context windows.
- `mask_error_details=True` on the FastMCP instance so internal exception traces don’t surface to the agent.
- Per-tool RBAC via the bearer token at the ALB — only enrolled service identities can call `extract_signatures`.

-----

## 8. AWS ECS Deployment Specification

### 8.1 Compute

- **Task:** Fargate, x86_64, 2 vCPU / 4 GB RAM. (For higher throughput, scale out horizontally rather than up — a single inference is single-threaded-friendly with ORT thread pinning.)
- **Service:** desired count 2, max 20, target-tracking on CPU 60% and `ALBRequestCountPerTarget` 50.
- **Placement:** at least two AZs, no public subnet attachment.

### 8.2 Networking

- Internal ALB, target group `ip` mode, health-check path `/v1/readyz`, healthy threshold 2.
- Security groups: ALB SG allows 443 from the consuming-app SG only; task SG allows 8080/8081 from the ALB SG only.
- VPC endpoints for S3, ECR, CloudWatch Logs, Secrets Manager — **no internet egress** from tasks.

### 8.3 Storage

- ECR private repository for the image.
- S3 bucket `hsbc-sig-cropper-{env}` with:
  - SSE-KMS with a customer-managed key
  - Object Lock not required (we delete after processing), but consider it for an audit copy bucket
  - Lifecycle rule: delete `inputs/` and `outputs/` after 24 hours
  - Block Public Access: all four toggles on
  - Bucket policy: only the ECS task role can `s3:GetObject`/`s3:PutObject` under the relevant prefixes

### 8.4 Observability

- CloudWatch Logs: structured JSON, no payload bytes, request_id propagated.
- CloudWatch Metrics: latency, success rate, detections-per-page, p95/p99 latency. Alarms on p95 > 800 ms and error rate > 1%.
- X-Ray (optional) for distributed tracing if other services in the chain are X-Ray-enabled.

### 8.5 Secrets & config

- All thresholds, S3 bucket names, KMS key IDs, model version pins are environment variables driven by Secrets Manager / SSM Parameter Store, not baked into the image.
- Model artifact (ONNX) **is** baked into the image to avoid runtime download from the internet. Update = new image build = new task definition revision = blue/green deploy via ECS rolling update.

-----

## 9. Acceptance Criteria

These are the criteria by which the system is considered production-ready. They are designed to be objectively measurable.

### 9.1 Accuracy (measured on held-out HSBC test set, ≥ 500 pages, ≥ 700 signatures)

|Metric                                                                                           |Threshold|
|-------------------------------------------------------------------------------------------------|---------|
|Precision @ default threshold (0.55)                                                             |≥ 0.96   |
|Recall @ default threshold                                                                       |≥ 0.95   |
|mAP@50 (single class `signature`)                                                                |≥ 0.94   |
|mAP@50:95                                                                                        |≥ 0.65   |
|False-positive rate on no-signature pages (50 pages of forms with the signature field left blank)|≤ 2%     |
|Bounding-box IoU vs. human label (median, only on TP detections)                                 |≥ 0.85   |

### 9.2 Performance (measured on Fargate 2 vCPU / 4 GB, INT8 model)

|Metric                                     |Threshold    |
|-------------------------------------------|-------------|
|p50 latency, single A4 page @ 300 DPI      |≤ 250 ms     |
|p95 latency, single A4 page @ 300 DPI      |≤ 400 ms     |
|p99 latency, single A4 page @ 300 DPI      |≤ 700 ms     |
|Sustained throughput per task              |≥ 6 pages/sec|
|Cold-start time (task start → `readyz` 200)|≤ 15 s       |
|Memory headroom under sustained load       |≥ 25% free   |

### 9.3 API & MCP contract

- OpenAPI 3.1 spec published and matches the implementation, validated in CI.
- MCP server passes `mcp-inspector` connectivity test and returns valid JSON-RPC for all three tools.
- Idempotent on `request_id` — replaying the same request_id within 24 h returns the same response.

### 9.4 Security & compliance

- No signature pixel data in any log line (verified by a CI regex check on log output during tests).
- TLS 1.2+ only at the ALB; perfect-forward-secrecy ciphers only.
- Container image passes `trivy` scan with **zero CRITICAL** CVEs at deploy time.
- Image scanned monthly; alerts on any new CRITICAL/HIGH.
- IAM least privilege validated by `iam-policy-simulator` against the documented permission list.
- Data retention: inputs and outputs deleted from S3 within 24 hours (verified by an automated S3 inventory check).
- Audit log captures: timestamp, request_id, caller identity, page count, signature count, model version. **Never** signature bytes.

### 9.5 Operational

- Two ECS tasks survive a single-AZ failure (chaos test: stop one task, p95 latency returns to baseline within 60 s).
- Image is reproducible: `docker build` from the tagged source repo produces a byte-identical image (multi-stage with pinned dependencies and `--no-cache-dir`).
- Runbook exists covering: model rollback, threshold tuning, dependency CVE patch, drift-alert response.
- Model card and OpenAPI spec checked into the repo and version-aligned with the image tag.

### 9.6 License & legal

- Final model architecture is **Apache-2.0** end-to-end (training code, inference code, weights file) **or** an executed Ultralytics Enterprise License is on file before any AGPL-licensed weight ever runs on the production cluster.
- Third-party-license inventory generated by `pip-licenses` is reviewed and on file.

-----

## 10. Risks & Mitigations

|Risk                                                                       |Likelihood|Impact  |Mitigation                                                                                                                                                     |
|---------------------------------------------------------------------------|----------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
|Model misses signatures in unusual ink colors (e.g. red)                   |Medium    |Medium  |Include red/blue/black ink variants in training augmentations; collect failure cases from production into a feedback loop.                                     |
|Form layouts change (HSBC redesigns the account opening form)              |Medium    |High    |Drift monitor on a canary set; quarterly re-evaluation; design system to support multiple model versions concurrently (`X-Model-Version` header).              |
|Printed cursive signatures (e.g. “Sample” watermarks) cause false positives|Medium    |Low     |Train with explicit negative examples; consider a secondary binary classifier (“is this stroke handwritten?”) as a confidence booster on borderline detections.|
|Confidential data leakage via logs or LLM-context (MCP)                    |Low       |Critical|Hard rule: pixels never enter logs or LLM context; S3 URIs only via MCP; CI regex check.                                                                       |
|AGPL surprise (someone pulls in a YOLOv8-licensed dep transitively)        |Medium    |Critical|Automated license scan (e.g. `pip-licenses --fail-on AGPL`) gating the CI pipeline.                                                                            |
|ONNX Runtime OpenVINO EP regression on a new release                       |Low       |Medium  |Pin `onnxruntime-openvino` version; validate accuracy + latency on every dependency bump.                                                                      |
|Throughput insufficient at peak volumes                                    |Medium    |Medium  |ECS auto-scaling already in design; if peak is sustained, switch to bare EC2 c7i.large with reserved capacity for ~30% cost reduction at high utilization.     |

-----

## 11. Estimated Cost (steady-state, illustrative)

Assumptions: 4 Fargate tasks running 24/7 in eu-west-2; 50 GB/month S3 ingress; 50 GB/month S3 storage (24-hour retention turns it into a much smaller working set); 100 GB/month CloudWatch logs.

|Item                           |Approx. monthly cost (USD)|
|-------------------------------|--------------------------|
|Fargate 4 × (2 vCPU, 4 GB) 24/7|~$210                     |
|Internal ALB                   |~$25                      |
|S3 storage + requests          |~$10                      |
|KMS requests                   |~$5                       |
|CloudWatch logs + metrics      |~$30                      |
|ECR storage                    |~$1                       |
|**Total**                      |**~$280/month**           |

Prices vary by region and over time — confirm against the current AWS Pricing Calculator before committing to a number internally. Per-page cost at 1 M pages/month is roughly **$0.0003 per page**, which is 30–100× cheaper than Textract’s `SIGNATURES` per-page price at comparable volume.

-----

## 12. What to Build First (week-1 concrete checklist)

1. Stand up the labelling environment inside the VPC; export the tech4humans Apache-2.0 dataset as a starting set.
1. Get 50–100 anonymized HSBC pages from a sympathetic stakeholder for a smoke-test set.
1. Spike: clone `PekingU/rtdetr_v2_r18vd`, fine-tune on the tech4humans dataset for 20 epochs on a T4 spot instance, export to ONNX, benchmark on a `c7i.large` EC2 — sanity-check that we land near published numbers and that latency on the target hardware is acceptable.
1. In parallel: scaffold the FastAPI + FastMCP service skeleton with a stub model that returns canned detections, deploy it to a dev ECS Fargate service, confirm the whole rail (ALB → ECS → S3 → logs → readyz) works end-to-end before the real model lands.
1. Legal sign-off on the Apache-2.0 path so it can’t block the launch later.

Once items 3 and 4 land, the remaining work is fine-tuning, hardening, and rollout — all of which are well-understood, predictable engineering.

-----

## Appendix A — Key references

- “Open-Source Handwritten Signature Detection Model” — S. Lima Braz / tech4humans, Hugging Face, March 2025. Full benchmark of 21 architectures on a 2,819-image signature-detection dataset, with CPU-ONNX timings.
- `tech4humans/yolov8s-signature-detector` and `tech4humans/conditional-detr-50-signature-detector` on Hugging Face — published weights and metrics on the same dataset.
- `PekingU/rtdetr_v2_r18vd` on Hugging Face — Apache-2.0 RT-DETRv2 baseline suitable for fine-tuning.
- “DETRs Beat YOLOs on Real-time Object Detection” (RT-DETR), CVPR 2024.
- AWS Blog, “Accelerate inference with ONNX Runtime on AWS” — guidance on ORT execution providers and CPU tuning.
- FastMCP documentation, `gofastmcp.com` — the canonical way to build MCP servers in Python.
- Ultralytics License page — authoritative source on the AGPL-3.0 vs Enterprise distinction for YOLOv5/v8/v11.