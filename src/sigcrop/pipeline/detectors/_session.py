"""Shared lazy ONNX Runtime session builder.

Both backends use the same OpenVINO-EP-preferred, CPU-fallback session
configuration; extracted here so each backend body stays small.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from sigcrop.config import get_settings
from sigcrop.errors import ModelUnavailable
from sigcrop.models.registry import REGISTRY, verify_weights


class OrtSession:
    """Holds one lazily-built ORT session + its primary input name."""

    def __init__(self, model_version: str) -> None:
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._input_name: str = ""
        self._model_version = model_version

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    def model_path(self) -> Path:
        settings = get_settings()
        record = REGISTRY.get(self._model_version)
        if record is None:
            raise ModelUnavailable(f"Unknown model_version: {self._model_version}")
        return settings.model_dir / record.filename

    def build(self) -> None:
        """Build the session and verify weights. Idempotent under the lock."""
        import onnxruntime as ort

        with self._lock:
            if self._session is not None:
                return

            settings = get_settings()
            model_path = self.model_path()
            if not model_path.is_file():
                raise ModelUnavailable(f"Model file not found at {model_path}")

            record = REGISTRY.get(self._model_version)
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
                session = ort.InferenceSession(
                    str(model_path), sess_options=so, providers=providers
                )
            except Exception as exc:  # noqa: BLE001 — wrap any ORT load failure
                raise ModelUnavailable(f"Failed to build ONNX session: {exc}") from exc

            self._input_name = session.get_inputs()[0].name
            self._session = session

    def warm_up(self, input_size: int) -> None:
        """Build session and burn a dummy forward pass to pre-fill caches."""
        if self._session is not None:
            return
        self.build()
        dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        self.run(dummy)

    def run(self, model_input_nchw: np.ndarray) -> list[np.ndarray]:
        if self._session is None:
            raise ModelUnavailable("ONNX session not loaded; call warm_up first")
        outputs: list[np.ndarray] = self._session.run(
            None, {self._input_name: model_input_nchw}
        )
        return outputs
