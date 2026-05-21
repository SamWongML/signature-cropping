"""Detector backend registry.

Add a new backend by:
  1. Implementing `DetectorBackend` in a new module.
  2. Appending a `ModelRecord` to `sigcrop.models.registry.REGISTRY`.
  3. Registering the backend class below.
"""

from __future__ import annotations

from collections.abc import Callable

from sigcrop.errors import ModelUnavailable
from sigcrop.pipeline.detectors.base import (
    Detection,
    DetectorBackend,
    NormalizationKind,
    PreprocessSpec,
)
from sigcrop.pipeline.detectors.conditional_detr import ConditionalDETRBackend
from sigcrop.pipeline.detectors.yolov8 import YOLOv8Backend

BACKEND_FACTORIES: dict[str, Callable[[], DetectorBackend]] = {
    ConditionalDETRBackend.NAME: ConditionalDETRBackend,
    YOLOv8Backend.NAME: YOLOv8Backend,
}

_INSTANCES: dict[str, DetectorBackend] = {}


def available_backends() -> list[str]:
    return sorted(BACKEND_FACTORIES.keys())


def get_backend(name: str) -> DetectorBackend:
    """Return the lazy singleton for a backend name."""
    if name not in BACKEND_FACTORIES:
        raise ModelUnavailable(
            f"Unknown detector backend: {name!r}. "
            f"Available: {available_backends()}"
        )
    instance = _INSTANCES.get(name)
    if instance is None:
        instance = BACKEND_FACTORIES[name]()
        _INSTANCES[name] = instance
    return instance


__all__ = [
    "BACKEND_FACTORIES",
    "Detection",
    "DetectorBackend",
    "NormalizationKind",
    "PreprocessSpec",
    "available_backends",
    "get_backend",
]
