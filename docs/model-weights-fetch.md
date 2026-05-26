# Fetching detector weights in a network-restricted environment

How to obtain the two detector ONNX weights when the default source (Hugging
Face) is unreachable — e.g. a sandbox whose egress policy allows only GitHub
and PyPI. `scripts/fetch_pretrained.py` pulls both models from Hugging Face,
so it fails outright when `huggingface.co` is blocked.

## TL;DR

- **YOLOv8s (`yolov8`): obtained.** The signature-trained ONNX is committed
  (not LFS, a real 44.6 MB binary) inside the project's own server repo on
  GitHub, which is reachable.
- **Conditional-DETR (`conditional-detr`): NOT obtainable.** It lives only on
  Hugging Face, is not committed to any reachable GitHub repo, has no release
  asset, and reproducing the ONNX export needs `torch` + the HF safetensors.

## Reachability

`403` = a filtering proxy denied the host; `404`/`206` on a reachable host
still counts as reachable; DNS errors mean the host does not resolve.

| Host | Reachable? |
|---|---|
| `pypi.org` / `files.pythonhosted.org` | ✅ |
| `github.com`, `raw.githubusercontent.com`, `codeload.github.com` | ✅ |
| GitHub release-asset CDN (binaries) | ✅ |
| `api.github.com` | ❌ 403 (unauth) |
| `huggingface.co` / `hf.co` / `cas-bridge.xethub.hf.co` | ❌ 403 |
| `cdn-lfs.huggingface.co` | ❌ DNS failure |
| `hf-mirror.com`, Kaggle, Google Drive, Zenodo, GitLab, ModelScope | ❌ 403 |

**Only GitHub (repo contents + release assets) and PyPI are usable.**

## What worked — YOLOv8s

1. Canonical project repo: **`tech4ai/t4ai-signature-detect-server`** (default
   branch `develop`; the fork `kherrick/signature-detection` has the same
   contents). Note the default branch is `develop`, not `main`.
2. No `.gitattributes` → no Git LFS, so committed binaries are real files.
3. The repo zipball contains exactly one weight file (Triton layout):

   ```
   signature-detection/models/yolov8s/1/model.onnx   44,615,140 bytes
   sha256 = f4c3e51b5aecfda1be1de12cb9b0960495029006a85e2b05334fb4d2a572403c
   ```
4. Verified with `onnxruntime`: input `images [batch,3,h,w]`, output
   `output0 (1,5,8400)` — the single-class `(1,5,N)` layout
   `decode_yolov8_output` expects; metadata `names={0:'signature'}`,
   `Ultralytics YOLOv8 v8.3.58`, `license=AGPL-3.0`. The registry pins an empty
   `sha256` for YOLO (verification skipped), so it loads as-is.

### Reproduce

```bash
python - <<'PY'
import urllib.request, io, zipfile
url = "https://codeload.github.com/tech4ai/t4ai-signature-detect-server/zip/refs/heads/develop"
z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=200).read()))
name = "t4ai-signature-detect-server-develop/signature-detection/models/yolov8s/1/model.onnx"
open("yolov8s_signature.onnx", "wb").write(z.read(name))
PY
# then place it at  $SIGCROP_MODEL_DIR/yolov8s_signature.onnx
```

## Dead ends

- GitHub releases of the canonical repo (`v1.0.0` / `v1.0.1`) carry no
  downloadable assets.
- **Conditional-DETR**: no ONNX committed in any reachable repo, no release
  asset, no LFS pointer to chase. Reproducing it needs `torch` + the HF
  safetensors → blocked. Unobtainable in this environment.

## Caveats

- The YOLO weight is the upstream Ultralytics-derived model — **AGPL-3.0**, the
  same production blocker as before (`docs/ARCHITECTURE.md` §9). Fine for
  benchmarking, not for shipping.
- The ONNX has dynamic input dims; the pipeline always feeds 640×640.
- DETR cannot be benchmarked here; its comparison figures come from the model
  card / `ARCHITECTURE.md`, not a local run.
