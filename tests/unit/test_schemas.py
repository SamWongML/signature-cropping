"""Schema validation is the one piece that's safe to test on the scaffold."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sigcrop.api.schemas import BBoxNorm, CropOptions, ReturnFormat


def test_crop_options_defaults() -> None:
    opts = CropOptions()
    assert opts.confidence_threshold == 0.55
    assert opts.padding_pct == 0.08
    assert opts.apply_mask is False
    assert opts.return_format is ReturnFormat.INLINE_B64


def test_crop_options_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        CropOptions(confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        CropOptions(padding_pct=-0.1)


def test_bbox_normalized_bounds() -> None:
    BBoxNorm(x=0.0, y=0.0, w=1.0, h=1.0)
    with pytest.raises(ValidationError):
        BBoxNorm(x=-0.1, y=0.0, w=0.5, h=0.5)
