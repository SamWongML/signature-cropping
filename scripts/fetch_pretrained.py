"""Download an off-the-shelf signature detector and place its ONNX in $SIGCROP_MODEL_DIR.

Two backends supported:

  --backend conditional-detr  (default)
      Loads `tech4humans/conditional-detr-50-signature-detector` (Apache-2.0,
      mAP@50 = 0.937) via transformers and exports to ONNX with torch.onnx.

  --backend yolov8
      Downloads the pre-exported ONNX from
      `tech4humans/yolov8s-signature-detector` (AGPL-3.0 weights, accepted
      for MVP A/B evaluation only — see docs/ARCHITECTURE.md §9). No torch
      export step; the model is already an ONNX file on the Hub.

In both modes the script prints the SHA-256 so it can be pinned in
`sigcrop.models.registry.REGISTRY`.
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


def _fetch_conditional_detr(
    hf_id: str, out_path: Path, out_dir: Path, opset: int, input_size: int
) -> None:
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    print(f"[fetch_pretrained] loading {hf_id}")
    model = AutoModelForObjectDetection.from_pretrained(hf_id)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(hf_id)
    processor.save_pretrained(out_dir / "processor")

    class _Export(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.m(pixel_values=pixel_values)
            return out.logits, out.pred_boxes

    wrapper = _Export(model)
    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)

    print(f"[fetch_pretrained] exporting → {out_path} (opset={opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["pixel_values"],
            output_names=["logits", "pred_boxes"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def _fetch_yolov8(hf_id: str, hf_filename: str, out_path: Path) -> None:
    from huggingface_hub import hf_hub_download

    print(f"[fetch_pretrained] downloading {hf_id}:{hf_filename}")
    local = hf_hub_download(repo_id=hf_id, filename=hf_filename)
    out_path.write_bytes(Path(local).read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["conditional-detr", "yolov8"],
        default="conditional-detr",
    )
    parser.add_argument(
        "--hf-id",
        default=os.environ.get("SIGCROP_MODEL_HF_ID"),
        help="Hugging Face repo id (default depends on --backend)",
    )
    parser.add_argument(
        "--hf-filename",
        default=None,
        help="(yolov8 only) Filename inside the HF repo to download",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SIGCROP_MODEL_DIR", "models"),
        help="Output directory for the ONNX file",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Filename for the local ONNX (default depends on --backend)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (ConditionalDETR path only)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square input edge in pixels (matches preprocess.letterbox)",
    )
    args = parser.parse_args()

    backend_defaults = {
        "conditional-detr": {
            "hf_id": "tech4humans/conditional-detr-50-signature-detector",
            "hf_filename": None,
            "out_file": "conditional_detr_signature.onnx",
            "input_name": "pixel_values",
        },
        "yolov8": {
            "hf_id": "tech4humans/yolov8s-signature-detector",
            "hf_filename": "yolov8s.onnx",
            "out_file": "yolov8s_signature.onnx",
            "input_name": "images",
        },
    }
    defaults = backend_defaults[args.backend]

    hf_id = args.hf_id or defaults["hf_id"]
    out_file = args.out_file or defaults["out_file"]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_file

    if args.backend == "conditional-detr":
        _fetch_conditional_detr(hf_id, out_path, out_dir, args.opset, args.input_size)
    else:
        hf_filename = args.hf_filename or defaults["hf_filename"]
        _fetch_yolov8(hf_id, hf_filename, out_path)

    digest = sha256_of(out_path)
    print(f"[fetch_pretrained] sha256={digest}")
    print(f"[fetch_pretrained] file_size_bytes={out_path.stat().st_size}")

    # Smoke-test the ONNX with onnxruntime.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, args.input_size, args.input_size), dtype=np.float32)
    outputs = sess.run(None, {input_name: dummy})
    print(
        f"[fetch_pretrained] smoke test ok — input={input_name!r} "
        f"output shapes: {[o.shape for o in outputs]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
