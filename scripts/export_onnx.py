"""Deprecated — use scripts/fetch_pretrained.py instead.

Kept as a forwarding shim so existing docs / CI calls keep working.
"""

from __future__ import annotations

import runpy
import sys


def main() -> int:
    print("export_onnx.py is deprecated; forwarding to fetch_pretrained.py", file=sys.stderr)
    runpy.run_path(str(__file__).replace("export_onnx.py", "fetch_pretrained.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
