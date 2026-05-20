"""Encode tests — PNG round-trip via base64."""

from __future__ import annotations

import base64

import numpy as np

from sigcrop.pipeline.encode import encode_signatures
from sigcrop.pipeline.postprocess import Signature


def _fake_sig() -> Signature:
    crop = np.zeros((40, 60, 3), dtype=np.uint8)
    crop[10:30, 20:40] = (0, 0, 255)
    return Signature(
        page=1, x=0, y=0, w=60, h=40, confidence=0.9,
        crop_bgr=crop, mask_applied=False,
    )


def test_encode_inline_b64_round_trips() -> None:
    out = encode_signatures([_fake_sig()], return_format="inline_b64", s3_prefix=None)
    assert len(out) == 1
    enc = out[0]
    assert enc.crop_s3_uri is None
    assert enc.crop_b64 is not None
    raw = base64.b64decode(enc.crop_b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
