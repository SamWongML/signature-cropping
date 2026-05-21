"""Detector facade.

This module is a backward-compatible shim over `pipeline/detectors/`. New
backends should be added to that package. Importers can keep using
`Detection`, `get_detector()`, and `require_ready_detector()` from here.
"""

from __future__ import annotations

from sigcrop.config import get_settings
from sigcrop.errors import ModelUnavailable
from sigcrop.pipeline.detectors import (
    Detection,
    DetectorBackend,
    available_backends,
    get_backend,
)


def get_detector(backend_name: str | None = None) -> DetectorBackend:
    """Return the lazy singleton for the requested (or configured) backend."""
    name = backend_name or get_settings().detector_backend
    return get_backend(name)


def require_ready_detector(backend_name: str | None = None) -> DetectorBackend:
    det = get_detector(backend_name)
    if not det.is_ready:
        raise ModelUnavailable("ONNX session not loaded; call /readyz first")
    return det


__all__ = [
    "Detection",
    "DetectorBackend",
    "available_backends",
    "get_detector",
    "require_ready_detector",
]
