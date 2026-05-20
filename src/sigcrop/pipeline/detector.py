"""ONNX detector wrapper.

One process owns one `ort.InferenceSession`. Lazily constructed on first use
(or by `/readyz` warm-up). Build is guarded by a lock; onnxruntime sessions
are themselves reentrant for `session.run`.

Post-processing follows the ConditionalDETR convention used by the HF
`post_process_object_detection` reference impl:
- logits: (1, num_queries, num_labels) — sigmoid per class, no "no object" slot
- pred_boxes: (1, num_queries, 4) in cxcywh, normalized to [0, 1]
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigcrop.config import get_settings
from sigcrop.errors import ModelUnavailable
from sigcrop.models.registry import REGISTRY, verify_weights


@dataclass(slots=True, frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    confidence: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


class Detector:
    """Off-the-shelf signature detector. Single class output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: object | None = None
        self._input_name: str = "pixel_values"
        self._model_version: str = get_settings().model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def _build_session(self) -> object:
        import onnxruntime as ort

        settings = get_settings()
        model_path = Path(settings.model_path)
        if not model_path.is_file():
            raise ModelUnavailable(f"Model file not found at {model_path}")

        record = REGISTRY.get(settings.model_version)
        if record is not None:
            verify_weights(model_path, record.sha256)

        so = ort.SessionOptions()
        so.intra_op_num_threads = settings.intra_op_num_threads
        so.inter_op_num_threads = settings.inter_op_num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = set(ort.get_available_providers())
        providers: list[str | tuple[str, dict[str, str]]] = []
        if "OpenVINOExecutionProvider" in available:
            providers.append(("OpenVINOExecutionProvider", {"device_type": "CPU"}))
        providers.append("CPUExecutionProvider")

        try:
            session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        except Exception as exc:  # noqa: BLE001 — wrap any ORT load failure
            raise ModelUnavailable(f"Failed to build ONNX session: {exc}") from exc

        self._input_name = session.get_inputs()[0].name
        return session

    def warm_up(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            session = self._build_session()
            # Run one synthetic inference so the first real request doesn't pay
            # the JIT / cache-fill cost.
            target = get_settings().detector_input_size
            dummy = np.zeros((1, 3, target, target), dtype=np.float32)
            session.run(None, {self._input_name: dummy})  # type: ignore[attr-defined]
            self._session = session

    def infer(self, model_input_nchw: np.ndarray) -> list[Detection]:
        if self._session is None:
            raise ModelUnavailable("ONNX session not loaded; call warm_up first")

        outputs = self._session.run(None, {self._input_name: model_input_nchw})  # type: ignore[attr-defined]

        # ConditionalDETR exports as (logits, pred_boxes). Be defensive about order:
        logits, boxes = self._unpack_outputs(outputs)

        # Conditional DETR uses per-class sigmoid; no "no object" slot.
        probs = _sigmoid(logits[0])  # (num_queries, num_labels)
        scores = probs.max(axis=-1)

        target = float(get_settings().detector_input_size)
        cx, cy, bw, bh = (boxes[0, :, i] for i in range(4))
        x = (cx - bw / 2.0) * target
        y = (cy - bh / 2.0) * target
        w = bw * target
        h = bh * target

        return [
            Detection(x=float(x[i]), y=float(y[i]), w=float(w[i]), h=float(h[i]),
                      confidence=float(scores[i]))
            for i in range(scores.shape[0])
        ]

    @staticmethod
    def _unpack_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        # Heuristic: the 4-feature output is boxes, the other is logits.
        a, b = outputs[0], outputs[1]
        if a.ndim == 3 and a.shape[-1] == 4:
            return b, a
        return a, b


_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def require_ready_detector() -> Detector:
    det = get_detector()
    if det._session is None:  # noqa: SLF001 — boundary check
        raise ModelUnavailable("ONNX session not loaded; call /readyz first")
    return det
