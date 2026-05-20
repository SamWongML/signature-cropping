"""Operator CLI. Not on the API hot path."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sigcrop", description="Signature cropper ops CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    crop = sub.add_parser("crop", help="Crop signatures from a local file")
    crop.add_argument("file", help="Path to PDF/PNG/JPEG/TIFF")
    crop.add_argument("--out", default=".", help="Output directory for PNG crops")
    crop.add_argument("--confidence", type=float, default=0.55)

    args = parser.parse_args(argv)
    if args.cmd == "crop":
        raise NotImplementedError("wire to sigcrop.pipeline once stages are implemented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
