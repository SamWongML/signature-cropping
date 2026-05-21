"""Detector backend contract.

Each backend owns its ONNX session, knows how the input tensor wants to be
normalized, and decodes its model-specific output shape into a flat
`list[Detection]` for the shared postprocessor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

NormalizationKind = Literal["imagenet", "rescale"]


@dataclass(slots=True, frozen=True)
class Detection:
    """Detector output in letterboxed-input pixel space.

    Coordinates are xywh in the model's input frame (typically 640x640 after
    letterbox). The postprocess stage remaps these to source-image pixels.
    """

    x: float
    y: float
    w: float
    h: float
    confidence: float


@dataclass(slots=True, frozen=True)
class PreprocessSpec:
    """How the active backend wants its input prepared."""

    input_size: int = 640
    letterbox_fill: int = 114
    normalize: NormalizationKind = "imagenet"


class DetectorBackend(ABC):
    """ONNX detector. Lazy session; one process owns one session per backend."""

    @property
    @abstractmethod
    def model_version(self) -> str: ...

    @property
    @abstractmethod
    def model_license(self) -> str: ...

    @property
    @abstractmethod
    def training_lineage_hash(self) -> str: ...

    @property
    @abstractmethod
    def default_confidence(self) -> float: ...

    @property
    @abstractmethod
    def default_nms_iou(self) -> float: ...

    @property
    @abstractmethod
    def preprocess_spec(self) -> PreprocessSpec: ...

    @abstractmethod
    def warm_up(self) -> None: ...

    @abstractmethod
    def infer(self, model_input_nchw: np.ndarray) -> list[Detection]: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...
