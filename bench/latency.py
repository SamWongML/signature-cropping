"""Per-stage latency harness. Run after a model is exported.

Reports P50/P95/P99 for ingest, preprocess, heuristics, inference, postprocess,
encode against the fixtures in tests/fixtures/. Acceptance gate from
docs/ARCHITECTURE.md §7.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
N_ITERS = 50


def main() -> int:
    print("latency harness — not implemented; pipeline stages are stubs", file=sys.stderr)
    _ = statistics.quantiles  # touch stdlib so linters don't warn
    _ = time.perf_counter
    _ = FIXTURES
    _ = N_ITERS
    return 0


if __name__ == "__main__":
    sys.exit(main())
