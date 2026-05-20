# CLAUDE.md — signature-cropping

Project memory for an AI signature-extraction service. Read this before changing code.

## What this project is

A CPU-only signature cropper for scanned bank account-opening forms (HSBC-style layouts). Two interfaces over one process: FastAPI REST and an MCP stdio server. Deploys to AWS ECS Fargate, x86_64, 2 vCPU / 4 GB.

Research brief: `temp/implementation_plan.md` (archive, do not edit).
Code-aligned spec: `docs/ARCHITECTURE.md` (canonical).

## Hard constraints

- **CPU only, x86_64.** No CUDA, no MPS, no GPU code paths. Anything that imports `torch.cuda` or assumes a GPU is wrong.
- **License-clean.** Detector weights and runtime code must be Apache-2.0 (or equivalent permissive). Ultralytics YOLOv5/8/11 are **AGPL-3.0** and forbidden unless an Enterprise License is acquired — see `docs/ARCHITECTURE.md` §License.
- **PII is signature pixels.** Never log raw images, base64 crops, or bbox pixel values together with a customer identifier. Logs are JSON to stdout only.
- **Latency budget: P95 ≤ 400 ms** per A4 page on 2 vCPU. New code must not push allocations into the hot path.
- **No network egress at inference time.** Model weights bake into the container image. S3 is the only allowed external dependency.

## Tech stack (locked)

| Layer     | Choice                                           |
| --------- | ------------------------------------------------ |
| Language  | Python 3.11                                      |
| Web       | FastAPI + Uvicorn                                |
| MCP       | FastMCP (stdio transport)                        |
| Vision    | OpenCV (headless), Pillow, PyMuPDF               |
| Inference | ONNX Runtime + OpenVINO Execution Provider, INT8 |
| Detector  | Conditional-DETR-50 (Apache-2.0), off-the-shelf — see docs/ARCHITECTURE.md §3 |
| Config    | Pydantic Settings v2                             |
| Tests     | pytest, pytest-asyncio                           |
| Lint      | ruff, mypy --strict                              |

Do not add a new dependency without updating `docs/ARCHITECTURE.md` and `pyproject.toml` together.

## Code rules

1. **Pipeline stages are pure functions or single-method classes.** No global state except the lazy-loaded ONNX session in `pipeline/detector.py`.
2. **Type everything.** `mypy --strict` is a CI gate. `Any` requires a comment justifying it.
3. **Errors are typed.** Raise from `sigcrop.errors`. Never raise bare `Exception`. Map to HTTP/MCP error codes in the boundary layer only.
4. **No prints, no `logging.getLogger(__name__).info("...")` with PII.** Use `sigcrop.logging.get_logger()` and pass structured fields.
5. **One way to do each thing.** If you find two image-decode paths, two NMS implementations, two config loaders — collapse them.
6. **Don't add a CLI flag, env var, or API parameter "for flexibility."** Add it when there is a caller that needs it.

## Behavioral guidelines (carried from prior CLAUDE.md)

- State assumptions before implementing. If multiple interpretations exist, ask.
- Surgical changes — every changed line should trace to the request.
- No speculative abstractions. 50 lines beats 200 lines.
- Define verifiable success criteria before coding.

## Useful commands

```bash
make install        # uv sync, dev deps included
make lint           # ruff + mypy
make test           # pytest -q
make bench          # latency harness against models/sample.onnx
make docker         # build the release image
make run-api        # FastAPI on :8080
make run-mcp        # MCP stdio server (for agent dev)
```

## Where things live

- Pipeline: `src/sigcrop/pipeline/*` — pure, testable, no FastAPI imports.
- REST: `src/sigcrop/api/*` — thin HTTP layer, delegates to pipeline.
- MCP: `src/sigcrop/mcp/*` — tool definitions, delegates to pipeline.
- Shared schemas: `src/sigcrop/api/schemas.py` — both interfaces import from here.
- Tests mirror source layout under `tests/`.

## Definition of done for a change

1. `make lint test` green.
2. New behavior covered by a test (unit if possible, integration if it crosses a boundary).
3. If touching the pipeline: benchmark before/after, paste P50/P95 in the PR.
4. If touching API/MCP schema: update `docs/ARCHITECTURE.md`.
5. No new dependency without justification in the PR description.

## Git Commit Guidelines

When creating or recommending git commits, strictly adhere to the Conventional Commits 1.0.0 specification:

- **Format**: `<type>[optional scope]: <description>` followed by an optional blank line and `[body]`, then `[footer(s)]`.
- **Allowed Types**:
    - `feat`: New feature (correlates with SemVer MINOR).
    - `fix`: Bug fix (correlates with SemVer PATCH).
    - `build`, `chore`, `ci`, `docs`, `style`, `refactor`, `perf`, `test`.
- **Scope**: Optional, noun describing a section of the codebase in parentheses, e.g., `fix(parser):`.
- **Description**: Imperative mood, lowercase, short summary, no period at the end.
- **Breaking Changes**: Append a `!` right before the colon (e.g., `feat(api)!:`) OR start a footer with uppercase `BREAKING CHANGE:`.
