"""Preprocess tests against the synthetic fixture."""

from __future__ import annotations

import numpy as np
import pytest

from sigcrop.errors import LowContrast
from sigcrop.pipeline.preprocess import preprocess_page
from tests.fixtures._synthesize import to_bgr_ndarray


def test_preprocess_returns_letterboxed_nchw() -> None:
    pp = preprocess_page(to_bgr_ndarray(), page_index=0)
    assert pp.model_input.shape == (1, 3, 640, 640)
    assert pp.model_input.dtype == np.float32
    assert pp.letterbox.src_w > 0
    assert pp.letterbox.src_h > 0
    assert 0.0 < pp.letterbox.scale <= 1.0


def test_preprocess_rejects_low_contrast() -> None:
    flat = np.full((2200, 1700, 3), 255, dtype=np.uint8)
    with pytest.raises(LowContrast):
        preprocess_page(flat, page_index=0)
