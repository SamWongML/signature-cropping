"""Heuristic candidate-region tests."""

from __future__ import annotations

import numpy as np

from sigcrop.pipeline.heuristics import find_candidate_regions
from tests.fixtures._synthesize import to_bgr_ndarray


def test_blank_page_yields_no_candidates() -> None:
    blank = np.full((2200, 1700, 3), 255, dtype=np.uint8)
    assert find_candidate_regions(blank) == []


def test_synthetic_form_yields_at_least_one_candidate() -> None:
    candidates = find_candidate_regions(to_bgr_ndarray())
    assert candidates, "expected at least one ink blob from the synthetic scribble"
    # Every candidate should have positive dimensions and a sensible density.
    for c in candidates:
        assert c.w > 0
        assert c.h > 0
        assert 0.05 <= c.stroke_density <= 0.50
