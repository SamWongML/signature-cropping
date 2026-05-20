"""Postprocess tests — NMS, coord remap, padding."""

from __future__ import annotations

import numpy as np

from sigcrop.pipeline.detector import Detection
from sigcrop.pipeline.postprocess import postprocess_detections
from sigcrop.pipeline.preprocess import LetterboxParams


def _make_page() -> np.ndarray:
    page = np.full((1000, 800, 3), 250, dtype=np.uint8)
    page[400:500, 200:400] = 0  # an ink blob to be cropped
    return page


def _identity_letterbox(src_h: int = 1000, src_w: int = 800) -> LetterboxParams:
    return LetterboxParams(scale=1.0, pad_x=0, pad_y=0, src_w=src_w, src_h=src_h)


def test_empty_detections_returns_empty() -> None:
    out = postprocess_detections(
        detections=[],
        page_bgr=_make_page(),
        letterbox=_identity_letterbox(),
        page_index=1,
        confidence_threshold=0.5,
        nms_iou=0.5,
        padding_pct=0.0,
        apply_mask=False,
    )
    assert out == []


def test_below_threshold_filtered_out() -> None:
    det = Detection(x=200, y=400, w=200, h=100, confidence=0.3)
    out = postprocess_detections(
        detections=[det],
        page_bgr=_make_page(),
        letterbox=_identity_letterbox(),
        page_index=1,
        confidence_threshold=0.5,
        nms_iou=0.5,
        padding_pct=0.0,
        apply_mask=False,
    )
    assert out == []


def test_nms_collapses_duplicates() -> None:
    a = Detection(x=200, y=400, w=200, h=100, confidence=0.95)
    b = Detection(x=205, y=405, w=200, h=100, confidence=0.90)  # huge overlap
    out = postprocess_detections(
        detections=[a, b],
        page_bgr=_make_page(),
        letterbox=_identity_letterbox(),
        page_index=1,
        confidence_threshold=0.5,
        nms_iou=0.5,
        padding_pct=0.0,
        apply_mask=False,
    )
    assert len(out) == 1
    assert out[0].confidence == 0.95


def test_padding_and_clamp() -> None:
    det = Detection(x=10, y=10, w=100, h=100, confidence=0.9)
    out = postprocess_detections(
        detections=[det],
        page_bgr=_make_page(),
        letterbox=_identity_letterbox(),
        page_index=1,
        confidence_threshold=0.5,
        nms_iou=0.5,
        padding_pct=0.2,
        apply_mask=False,
    )
    assert len(out) == 1
    sig = out[0]
    # 20% of min(100,100)=20 padding on each side, clamped at 0.
    assert sig.x == 0
    assert sig.y == 0
    assert sig.w == 140  # 100 + 2*20 padding; clamp doesn't bite (src is 800 wide)
    assert sig.h == 140
