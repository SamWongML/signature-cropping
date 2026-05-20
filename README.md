# signature-cropping

CPU-only signature extraction service for scanned bank account-opening forms.
Single Python process exposes a FastAPI REST endpoint and an MCP stdio server
backed by an ONNX Runtime + OpenVINO inference pipeline.

- **Target:** AWS ECS Fargate, x86_64, 2 vCPU / 4 GB
- **MVP latency:** P95 ≤ 600 ms per A4 page (300 DPI), off-the-shelf FP32 model
- **Detector:** `tech4humans/conditional-detr-50-signature-detector` (Apache-2.0, mAP@50 = 0.937)

## Documents

- `docs/ARCHITECTURE.md` — code-aligned spec (pipeline, REST, MCP, acceptance)
- `temp/implementation_plan.md` — research brief / archive
- `CLAUDE.md` — project memory for AI assistants

## Quick start (no training required)

```bash
make install
python scripts/fetch_pretrained.py        # downloads + exports the ONNX (one-time)
make test                                  # unit tests pass; integration tests run if model is present
make run-api                               # http://localhost:8080/docs
make run-mcp                               # MCP stdio (wire to your agent)
```

Or build everything into a Docker image (model weights are baked in):

```bash
make docker
docker run --rm -p 8080:8080 sigcrop:dev
```

Smoke test the running service:

```bash
curl -F file=@tests/fixtures/synthetic_form.png \
     -H 'Authorization: Bearer dev' \
     http://localhost:8080/v1/crop-signature | jq
```

## Status

End-to-end working with an off-the-shelf Apache-2.0 detector. No
fine-tuning needed for the MVP. See `docs/ARCHITECTURE.md` §7 for the
MVP vs Production acceptance thresholds and `temp/implementation_plan.md`
for the full research brief covering the post-MVP fine-tuned path.
