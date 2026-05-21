"""Conditional-DETR-R50 backend.

Post-processing follows the HF `post_process_object_detection` reference:
- logits: (1, num_queries, num_labels) — per-class sigmoid, no "no object" slot
- pred_boxes: (1, num_queries, 4) in cxcywh, normalized to [0, 1]

Behavior is unchanged from the pre-refactor `pipeline/detector.py`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from sigcrop.config import get_settings
from sigcrop.pipeline.detectors._session import OrtSession
from sigcrop.pipeline.detectors.base import Detection, DetectorBackend, PreprocessSpec


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


class ConditionalDETRBackend(DetectorBackend):
    NAME: ClassVar[str] = "conditional-detr"
    MODEL_VERSION: ClassVar[str] = "conditional-detr-50-fp32-2026.05.20"
    LICENSE: ClassVar[str] = "Apache-2.0"
    LINEAGE: ClassVar[str] = "tech4humans/conditional-detr-50-signature-detector"
    DEFAULT_CONF: ClassVar[float] = 0.55
    DEFAULT_IOU: ClassVar[float] = 0.50

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
            normalize="imagenet",
        )

    @property
    def is_ready(self) -> bool:
        return self._ort.is_ready

    def warm_up(self) -> None:
        self._ort.warm_up(get_settings().detector_input_size)

    def infer(self, model_input_nchw: np.ndarray) -> list[Detection]:
        outputs = self._ort.run(model_input_nchw)
        logits, boxes = self._unpack_outputs(outputs)

        probs = _sigmoid(logits[0])  # (num_queries, num_labels)
        scores = probs.max(axis=-1)

        target = float(get_settings().detector_input_size)
        cx, cy, bw, bh = (boxes[0, :, i] for i in range(4))
        x = (cx - bw / 2.0) * target
        y = (cy - bh / 2.0) * target
        w = bw * target
        h = bh * target

        return [
            Detection(
                x=float(x[i]),
                y=float(y[i]),
                w=float(w[i]),
                h=float(h[i]),
                confidence=float(scores[i]),
            )
            for i in range(scores.shape[0])
        ]

    @staticmethod
    def _unpack_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # The 4-feature output is boxes, the other is logits.
        a, b = outputs[0], outputs[1]
        if a.ndim == 3 and a.shape[-1] == 4:
            return b, a
        return a, b
