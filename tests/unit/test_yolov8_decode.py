"""Unit test for YOLOv8 (1, 5, N) → Detection decoding.

Synthesizes the raw ORT output tensor in-memory so the test runs without
needing the actual ONNX file on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigcrop.pipeline.detectors.yolov8 import decode_yolov8_output


def _make_tensor(boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """Build a (1, 5, N) tensor from cxcywh+score rows."""
    arr = np.array(boxes, dtype=np.float32).T  # (5, N)
    return arr[np.newaxis, :, :]


def test_decode_filters_low_confidence_floor() -> None:
    # Pre-filter floor inside the backend is 0.05.
    tensor = _make_tensor([
        (320.0, 240.0, 100.0, 50.0, 0.01),  # below floor
        (160.0, 120.0, 80.0, 40.0, 0.04),  # below floor
    ])
    assert decode_yolov8_output(tensor) == []


def test_decode_converts_cxcywh_to_xywh() -> None:
    tensor = _make_tensor([
        (320.0, 240.0, 100.0, 50.0, 0.8),
    ])
    out = decode_yolov8_output(tensor)
    assert len(out) == 1
    d = out[0]
    # cx=320, w=100 → x=270; cy=240, h=50 → y=215. Floats are float32 round-tripped.
    assert d.x == pytest.approx(270.0)
    assert d.y == pytest.approx(215.0)
    assert d.w == pytest.approx(100.0)
    assert d.h == pytest.approx(50.0)
    assert d.confidence == pytest.approx(0.8, abs=1e-6)


def test_decode_passes_through_above_floor() -> None:
    tensor = _make_tensor([
        (100.0, 100.0, 40.0, 40.0, 0.06),  # just above floor
        (200.0, 200.0, 80.0, 80.0, 0.95),
    ])
    out = decode_yolov8_output(tensor)
    assert len(out) == 2
    confidences = sorted(d.confidence for d in out)
    assert confidences == pytest.approx([0.06, 0.95])


def test_decode_rejects_wrong_shape() -> None:
    bad = np.zeros((1, 84, 10), dtype=np.float32)  # 84-channel COCO export
    with pytest.raises(ValueError, match=r"\(1, 5, N\)"):
        decode_yolov8_output(bad)
