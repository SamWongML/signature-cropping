"""Download the off-the-shelf signature detector and export to ONNX.

Default model: `tech4humans/conditional-detr-50-signature-detector` (Apache-2.0,
mAP@50 = 0.937). Override with --hf-id or SIGCROP_MODEL_HF_ID.

The export is a one-shot, build-time operation. The container runtime never
runs this script — at inference time the ONNX file is already in
$SIGCROP_MODEL_DIR.

Steps:
1. Load the HF model + processor.
2. Wrap so forward(pixel_values) returns (logits, pred_boxes).
3. torch.onnx.export with a fixed (1, 3, 640, 640) dummy input.
4. Smoke-test with onnxruntime.
5. Print SHA-256 so it can be pinned in sigcrop.models.registry.REGISTRY.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-id",
        default=os.environ.get(
            "SIGCROP_MODEL_HF_ID",
            "tech4humans/conditional-detr-50-signature-detector",
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SIGCROP_MODEL_DIR", "models"),
        help="Output directory for the exported ONNX",
    )
    parser.add_argument(
        "--out-file",
        default="conditional_detr_signature.onnx",
        help="Filename for the exported ONNX model",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (17 covers ConditionalDETR ops)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square input edge in pixels (matches preprocess.letterbox)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_file

    # Lazy import — these deps are heavy and only needed at build time.
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    print(f"[fetch_pretrained] loading {args.hf_id}")
    model = AutoModelForObjectDetection.from_pretrained(args.hf_id)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(args.hf_id)
    processor.save_pretrained(out_dir / "processor")

    class _Export(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.m(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    wrapper = _Export(model)
    dummy = torch.zeros(1, 3, args.input_size, args.input_size, dtype=torch.float32)

    print(f"[fetch_pretrained] exporting → {out_path} (opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    digest = sha256_of(out_path)
    print(f"[fetch_pretrained] sha256={digest}")
    print(f"[fetch_pretrained] file_size_bytes={out_path.stat().st_size}")

    # Smoke-test the ONNX with onnxruntime.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    pixel_values = np.zeros((1, 3, args.input_size, args.input_size), dtype=np.float32)
    outputs = sess.run(None, {"pixel_values": pixel_values})
    print(f"[fetch_pretrained] smoke test ok — output shapes: {[o.shape for o in outputs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
