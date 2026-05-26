# Fetching detector weights in a network-restricted environment

Investigation log for obtaining the two detector ONNX weights when the
default source (Hugging Face) is unreachable. Written after a benchmark run
in a sandbox whose egress policy allows **only GitHub and PyPI**.

## TL;DR

- **YOLOv8s (`yolov8`): obtained.** The signature-trained ONNX is committed
  (not LFS, a real 44.6 MB binary) inside the project's own server repo on
  GitHub, which is reachable. Verified its I/O matches the repo backend.
- **Conditional-DETR (`conditional-detr`): NOT obtainable here.** It lives
  only on Hugging Face (blocked). It is not committed to any reachable GitHub
  repo, has no release asset, and reproducing the ONNX export needs `torch` +
  the HF safetensors (also blocked).

## Why the default path fails

`scripts/fetch_pretrained.py` pulls both models from Hugging Face:
- `yolov8` → `hf_hub_download("tech4humans/yolov8s-signature-detector", "yolov8s.onnx")`
- `conditional-detr` → `transformers.from_pretrained(...)` + `torch.onnx.export`

Both require `huggingface.co`, which this environment blocks.

## Reachability probes

Method: `urllib.request.urlopen(url, timeout=…)`, observing status / error.
A `403` means a filtering proxy denied the host; `404`/`400`/`206` on a
reachable host count as **reachable**; DNS errors mean the host does not
resolve.

| Host | Result | Reachable? |
|---|---|---|
| `pypi.org` / `files.pythonhosted.org` | 200 | ✅ |
| `github.com` | 200 | ✅ |
| `raw.githubusercontent.com` | 200 | ✅ |
| `codeload.github.com` | 200 | ✅ |
| GitHub release CDN (real asset, `Range` req) | 206 Partial | ✅ (binaries OK) |
| `api.github.com` | 403 | ❌ (blocked unauth) |
| `huggingface.co` | 403 | ❌ |
| `hf.co` | 403 | ❌ |
| `cas-bridge.xethub.hf.co` | 403 | ❌ |
| `cdn-lfs.huggingface.co` | DNS failure | ❌ |
| `hf-mirror.com` (+ direct ONNX URL) | 403 | ❌ |
| `drive.google.com` | 403 | ❌ |
| `zenodo.org` | 403 | ❌ |
| `gitlab.com` | 403 | ❌ |
| `www.modelscope.cn` | 403 | ❌ |
| `www.kaggle.com` | 403 | ❌ |

**Conclusion:** the only viable download route is GitHub (repo contents via
`codeload`/`raw`, or release assets) and PyPI wheels.

## Failed attempts (chronological)

1. **`pip`/`uv` from PyPI for a packaged model** — no PyPI package bundles the
   tech4humans signature weights. PyPI is only useful for runtime libs
   (`onnxruntime`, `opencv-python-headless`, …), which did install fine.
2. **Direct HF download** (`huggingface.co/.../yolov8s.onnx`) — 403.
3. **HF mirror** (`hf-mirror.com`) — 403, including the direct ONNX path.
4. **Other model hosts** (Kaggle, Google Drive, Zenodo, GitLab, ModelScope) —
   all 403.
5. **GitHub API** (`api.github.com/users/.../repos`) to enumerate repos — 403
   (unauthenticated API is blocked even though `github.com` HTML is not).
6. **Medium article** (project write-up, may link the repo) — 403.
7. **WebFetch of the author's GitHub profile** — listed repos, but no
   `signature-detection` repo under that account.
8. **`codeload` of `kherrick/signature-detection` @ `main`** — 404; the repo's
   default branch is `develop`, not `main`.
9. **Candidate repo names** `tech4humans/signature-detection`,
   `tech4ai/signature-detection`, `samuellimabraz/signature-detection` — all
   404. The real org repo is named `t4ai-signature-detect-server`.
10. **GitHub releases of the canonical repo** — tags `v1.0.0` / `v1.0.1`
    exist but carry **no downloadable assets**.
11. **DETR via GitHub** — no `conditional-detr` ONNX is committed in any
    reachable repo, and (no `.gitattributes` → no LFS) there is nowhere else
    it could hide. Reproducing it locally needs `torch` + HF safetensors →
    blocked. **DETR remains unobtainable in this environment.**

## What worked — YOLOv8s

1. Searched GitHub (`site:github.com`) and identified the canonical project
   server repo: **`tech4ai/t4ai-signature-detect-server`** (default branch
   `develop`), plus the fork **`kherrick/signature-detection`** with the same
   contents.
2. Confirmed **no Git LFS** (no `.gitattributes`), so committed binaries are
   real files, not pointers.
3. Downloaded the repo zipball via `codeload` and scanned for weight
   extensions. Found exactly one:

   ```
   signature-detection/models/yolov8s/1/model.onnx   44,615,140 bytes
   ```

   (Triton `model_repository` layout.)
4. Extracted it and recorded its hash:

   ```
   sha256 = f4c3e51b5aecfda1be1de12cb9b0960495029006a85e2b05334fb4d2a572403c
   ```
5. **Verified it is the right model** with `onnxruntime`:
   - input  `images`  `[batch, 3, height, width]`  float32
   - output `output0` `(1, 5, 8400)` at 640×640 — the single-class `(1,5,N)`
     layout that `decode_yolov8_output` expects
   - metadata: `names={0:'signature'}`, `task=detect`,
     `Ultralytics YOLOv8 v8.3.58`, `license=AGPL-3.0`

   This matches the `yolov8` backend contract, and the registry pins an empty
   `sha256` for YOLO (verification skipped), so it loads as-is.

### Reproduce

```bash
python - <<'PY'
import urllib.request, io, zipfile
url = "https://codeload.github.com/tech4ai/t4ai-signature-detect-server/zip/refs/heads/develop"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=200).read()))
name = "t4ai-signature-detect-server-develop/signature-detection/models/yolov8s/1/model.onnx"
open("yolov8s_signature.onnx", "wb").write(z.read(name))
print("saved", name)
PY
# place where the backend looks (SIGCROP_MODEL_DIR / yolov8s_signature.onnx)
```

## Caveats

- This is the **upstream Ultralytics-derived** weight; provenance matches the
  HF model card but it is **AGPL-3.0** — same production blocker as before
  (`docs/ARCHITECTURE.md` §9). Fine for benchmarking, not for shipping.
- The YOLO ONNX has **dynamic** input dims; the pipeline always feeds
  640×640, which is what the model was exported/trained for.
- DETR could not be benchmarked in this environment; its comparison figures
  come from the model card / `ARCHITECTURE.md`, not a local run.
