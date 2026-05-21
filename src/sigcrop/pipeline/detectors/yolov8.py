"""YOLOv8s signature detector backend (tech4humans/yolov8s-signature-detector).

The single-class ONNX export emits one tensor of shape `(1, 5, N)`:
  channels[0:4] — cxcywh box coords in input pixels (640 by default)
  channels[4]   — class probability (already in [0, 1]; no sigmoid needed)

NMS is intentionally NOT applied here — `pipeline/postprocess.py` owns the
single NMS implementation for the project.

Defaults bias toward recall ("max detection performance") for the MVP, since
we have no in-domain HSBC samples to tune against:
  - confidence ≥ 0.25 — Ultralytics' standard default; lower than DETR's 0.55
    because YOLO logits are not directly comparable
  - NMS IoU 0.45 — tighter than DETR's 0.50; signature boxes on a form rarely
    overlap meaningfully

License posture: AGPL-3.0 weights (derived from Ultralytics). Accepted for
MVP A/B evaluation; see `docs/ARCHITECTURE.md` §9 for the productionization
gate.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from sigcrop.config import get_settings
from sigcrop.pipeline.detectors._session import OrtSession
from sigcrop.pipeline.detectors.base import Detection, DetectorBackend, PreprocessSpec

# Pre-filter floor inside the backend. Any box below this is never a real
# signature; keeping it would just slow down the shared NMS for 8400 rows.
_PRE_NMS_FLOOR: float = 0.05


class YOLOv8Backend(DetectorBackend):
    NAME: ClassVar[str] = "yolov8"
    MODEL_VERSION: ClassVar[str] = "yolov8s-signature-2026.05.21"
    LICENSE: ClassVar[str] = "AGPL-3.0"
    LINEAGE: ClassVar[str] = "tech4humans/yolov8s-signature-detector"
    DEFAULT_CONF: ClassVar[float] = 0.25
    DEFAULT_IOU: ClassVar[float] = 0.45

    def __init__(self) -> None:
        self._ort = OrtSession(self.MODEL_VERSION)

    @property
    def model_version(self) -> str:
        return self.MODEL_VERSION

    @property
    def model_license(self) -> str:
        return self.LICENSE

    @property
    def training_lineage_hash(self) -> str:
        return self.LINEAGE

    @property
    def default_confidence(self) -> float:
        return self.DEFAULT_CONF

    @property
    def default_nms_iou(self) -> float:
        return self.DEFAULT_IOU

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(
            input_size=get_settings().detector_input_size,
            letterbox_fill=114,
            normalize="rescale",
        )

    @property
    def is_ready(self) -> bool:
        return self._ort.is_ready

    def warm_up(self) -> None:
        self._ort.warm_up(get_settings().detector_input_size)

    def infer(self, model_input_nchw: np.ndarray) -> list[Detection]:
        outputs = self._ort.run(model_input_nchw)
        return decode_yolov8_output(outputs[0])


def decode_yolov8_output(raw: np.ndarray) -> list[Detection]:
    """Decode a single (1, 5, N) YOLOv8 single-class ONNX tensor.

    Exposed at module scope so it can be unit-tested with a synthetic array
    without spinning up an ONNX session.
    """
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 5:
        raise ValueError(
            f"Expected YOLOv8 single-class output of shape (1, 5, N); got {raw.shape}"
        )

    pred = raw[0].T  # (N, 5)
    scores = pred[:, 4]
    mask = scores >= _PRE_NMS_FLOOR
    if not mask.any():
        return []

    pred = pred[mask]
    cx = pred[:, 0]
    cy = pred[:, 1]
    bw = pred[:, 2]
    bh = pred[:, 3]
    sc = pred[:, 4]
    x = cx - bw / 2.0
    y = cy - bh / 2.0

    return [
        Detection(
            x=float(x[i]),
            y=float(y[i]),
            w=float(bw[i]),
            h=float(bh[i]),
            confidence=float(sc[i]),
        )
        for i in range(sc.shape[0])
    ]
