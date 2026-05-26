"""Latency benchmark — the strict contract every detector backend is measured against.

This harness defines ONE fixed measurement protocol so that results for any
backend (current `conditional-detr` / `yolov8`, or any future model) are
directly comparable. The protocol — corpus, warm-up/iteration counts, metrics,
and the latency acceptance gate (docs/ARCHITECTURE.md §7) — is frozen in
`CONTRACT` below. Bump `CONTRACT.version` whenever the methodology changes;
results are only comparable within the same contract version.

What is measured, per backend (full `run_pipeline` path — the production code):
  - cold start: first `warm_up()` (ONNX session build + one forward)
  - end-to-end latency percentiles (P50/P95/P99) over the whole corpus × iters
  - mean stage split (preprocess / inference / postprocess) from the pipeline's
    own `timing_ms` accounting
  - detection count + confidences per image (no ground truth → reported, not gated)

The gate (PASS/FAIL, drives the exit code) is on end-to-end latency only.

Usage:
    SIGCROP_MODEL_DIR=/path/to/models python bench/latency.py            # all present backends
    python bench/latency.py --backend yolov8                              # one backend
    python bench/latency.py --json bench_result.json                      # archive the artifact
    python bench/latency.py --no-gate                                     # report without failing

Exit codes: 0 = all measured backends pass the gate (or skipped for missing
weights); 1 = a measured backend violated the gate; 2 = a backend named with
--backend has no weights present.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sigcrop.api.schemas import CropOptions
from sigcrop.config import get_settings
from sigcrop.models.registry import REGISTRY
from sigcrop.pipeline.detector import available_backends, get_backend, get_detector
from sigcrop.pipeline.run import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class LatencyGate:
    """End-to-end per-page latency budget (docs/ARCHITECTURE.md §7, MVP tier)."""

    p50_ms: float = 380.0
    p95_ms: float = 600.0
    p99_ms: float = 900.0
    cold_start_ms: float = 12_000.0


@dataclass(frozen=True)
class BenchContract:
    """Frozen measurement protocol. Do not tune per-run — bump `version` instead."""

    version: str = "1.0"
    corpus_dir: Path = REPO_ROOT / "samples"
    corpus_glob: str = "*.png"
    # Excluded from the latency corpus: the 10-signature montage is not a
    # realistic single-form input (matches tests/integration exclusion).
    corpus_exclude: tuple[str, ...] = ("Gemini_Generated_Image_8b41128b41128b41.png",)
    mime: str = "image/png"
    warmup_iters: int = 5
    measured_iters: int = 30
    gate: LatencyGate = field(default_factory=LatencyGate)


CONTRACT = BenchContract()


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile; q in [0, 100]. `sorted_vals` must be sorted asc."""
    if not sorted_vals:
        return float("nan")
    rank = max(1, -(-len(sorted_vals) * int(q) // 100))  # ceil(n*q/100), >=1
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def _corpus(contract: BenchContract, corpus_dir: Path) -> list[Path]:
    paths = sorted(
        p
        for p in corpus_dir.glob(contract.corpus_glob)
        if p.name not in contract.corpus_exclude
    )
    if not paths:
        raise SystemExit(f"no corpus images under {corpus_dir} matching {contract.corpus_glob!r}")
    return paths


def _weights_present(backend_name: str) -> bool:
    record = REGISTRY.get(get_backend(backend_name).model_version)
    if record is None:
        return False
    return (get_settings().model_dir / record.filename).is_file()


@dataclass
class BackendResult:
    model_version: str
    license: str
    default_conf: float
    default_iou: float
    cold_start_ms: float
    latency_ms: dict[str, float]
    stage_ms_mean: dict[str, float]
    detections: list[dict[str, object]]
    gate_pass: bool
    gate_violations: list[str]


def bench_backend(backend_name: str, corpus: list[Path], contract: BenchContract) -> BackendResult:
    det = get_detector(backend_name)
    opts = CropOptions(detector_backend=backend_name)

    t0 = time.perf_counter()
    det.warm_up()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    e2e: list[float] = []
    pre: list[int] = []
    inf: list[int] = []
    post: list[int] = []
    detections: list[dict[str, object]] = []

    for path in corpus:
        data = path.read_bytes()
        # One representative run for detection accounting.
        rep = run_pipeline(data, contract.mime, opts, request_id="bench")
        detections.append(
            {
                "image": path.name,
                "count": len(rep.signatures),
                "confidences": [round(s.confidence, 3) for s in rep.signatures],
            }
        )
        for i in range(contract.warmup_iters + contract.measured_iters):
            start = time.perf_counter()
            resp = run_pipeline(data, contract.mime, opts, request_id="bench")
            ms = (time.perf_counter() - start) * 1000.0
            if i >= contract.warmup_iters:
                e2e.append(ms)
                pre.append(resp.timing_ms.preprocess)
                inf.append(resp.timing_ms.inference)
                post.append(resp.timing_ms.postprocess)

    e2e.sort()
    latency = {
        "n": float(len(e2e)),
        "min": e2e[0],
        "p50": _percentile(e2e, 50),
        "p95": _percentile(e2e, 95),
        "p99": _percentile(e2e, 99),
        "max": e2e[-1],
        "mean": statistics.fmean(e2e),
    }
    stage = {
        "preprocess": statistics.fmean(pre),
        "inference": statistics.fmean(inf),
        "postprocess": statistics.fmean(post),
    }

    gate = contract.gate
    violations: list[str] = []
    if latency["p50"] > gate.p50_ms:
        violations.append(f"P50 {latency['p50']:.0f} > {gate.p50_ms:.0f} ms")
    if latency["p95"] > gate.p95_ms:
        violations.append(f"P95 {latency['p95']:.0f} > {gate.p95_ms:.0f} ms")
    if latency["p99"] > gate.p99_ms:
        violations.append(f"P99 {latency['p99']:.0f} > {gate.p99_ms:.0f} ms")
    if cold_ms > gate.cold_start_ms:
        violations.append(f"cold start {cold_ms:.0f} > {gate.cold_start_ms:.0f} ms")

    return BackendResult(
        model_version=det.model_version,
        license=det.model_license,
        default_conf=det.default_confidence,
        default_iou=det.default_nms_iou,
        cold_start_ms=cold_ms,
        latency_ms=latency,
        stage_ms_mean=stage,
        detections=detections,
        gate_pass=not violations,
        gate_violations=violations,
    )


def _print_report(results: dict[str, BackendResult], corpus: list[Path]) -> None:
    g = CONTRACT.gate
    print(f"\n=== benchmark contract v{CONTRACT.version} ===")
    print(
        f"corpus: {len(corpus)} images from {CONTRACT.corpus_dir} | "
        f"warmup={CONTRACT.warmup_iters} iters={CONTRACT.measured_iters}"
    )
    print(f"gate (e2e ms): P50≤{g.p50_ms:.0f} P95≤{g.p95_ms:.0f} P99≤{g.p99_ms:.0f} "
          f"cold≤{g.cold_start_ms:.0f}\n")

    hdr = f"{'backend':<18}{'license':<12}{'cold':>7}{'P50':>7}{'P95':>7}{'P99':>7}{'gate':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        lat = r.latency_ms
        verdict = "PASS" if r.gate_pass else "FAIL"
        print(
            f"{name:<18}{r.license:<12}{r.cold_start_ms:>7.0f}"
            f"{lat['p50']:>7.0f}{lat['p95']:>7.0f}{lat['p99']:>7.0f}{verdict:>7}"
        )

    for name, r in results.items():
        s = r.stage_ms_mean
        print(f"\n[{name}] {r.model_version}")
        print(f"  stage mean ms: pre={s['preprocess']:.0f} inf={s['inference']:.0f} "
              f"post={s['postprocess']:.0f}  (conf={r.default_conf} iou={r.default_iou})")
        for d in r.detections:
            print(f"  {d['image']}: {d['count']} sig(s) {d['confidences']}")
        if r.gate_violations:
            print(f"  GATE VIOLATIONS: {'; '.join(r.gate_violations)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=available_backends(),
        help="Benchmark one backend (default: every backend whose weights are present).",
    )
    parser.add_argument("--corpus", type=Path, help="Override corpus directory.")
    parser.add_argument("--json", type=Path, help="Write the machine-readable artifact here.")
    parser.add_argument(
        "--no-gate", action="store_true", help="Report only; never fail on gate violations."
    )
    args = parser.parse_args()

    corpus_dir = args.corpus or CONTRACT.corpus_dir
    corpus = _corpus(CONTRACT, corpus_dir)

    if args.backend:
        if not _weights_present(args.backend):
            print(
                f"requested backend {args.backend!r} has no weights in "
                f"{get_settings().model_dir}; nothing to measure.",
                file=sys.stderr,
            )
            return 2
        selected = [args.backend]
    else:
        selected = [b for b in available_backends() if _weights_present(b)]
        for b in available_backends():
            if b not in selected:
                print(f"skip {b}: weights not present in {get_settings().model_dir}",
                      file=sys.stderr)
        if not selected:
            print("no backend weights present; nothing to measure.", file=sys.stderr)
            return 0

    results = {name: bench_backend(name, corpus, CONTRACT) for name in selected}
    _print_report(results, corpus)

    if args.json:
        artifact = {
            "contract_version": CONTRACT.version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "corpus_dir": str(corpus_dir),
                "n_images": len(corpus),
                "warmup_iters": CONTRACT.warmup_iters,
                "measured_iters": CONTRACT.measured_iters,
                "mime": CONTRACT.mime,
                "gate": asdict(CONTRACT.gate),
            },
            "results": {name: asdict(r) for name, r in results.items()},
        }
        args.json.write_text(json.dumps(artifact, indent=2))
        print(f"\nwrote artifact → {args.json}")

    failed = [n for n, r in results.items() if not r.gate_pass]
    if failed and not args.no_gate:
        print(f"\nGATE FAILED for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
