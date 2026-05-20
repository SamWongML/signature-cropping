"""Run the detector against a labelled test split; report precision/recall/mAP.

Acceptance gates (docs/ARCHITECTURE.md §7):
- clear scans: precision ≥ 0.99, recall ≥ 0.98
- noisy scans: precision ≥ 0.95, recall ≥ 0.93
- overall:     mAP@50 ≥ 0.94
"""

from __future__ import annotations

import sys


def main() -> int:
    raise SystemExit(
        "eval_dataset is a placeholder; implement after the detector is wired."
    )


if __name__ == "__main__":
    sys.exit(main())
