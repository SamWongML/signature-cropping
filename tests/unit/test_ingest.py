"""Ingest tests against the synthetic PNG fixture."""

from __future__ import annotations

import pytest

from sigcrop.errors import CorruptFile, InvalidMime
from sigcrop.pipeline.ingest import ingest
from tests.fixtures._synthesize import to_png_bytes


def test_ingest_png_returns_one_page() -> None:
    doc = ingest(to_png_bytes(), mime_hint="image/png")
    assert doc.source_mime == "image/png"
    assert len(doc.pages) == 1
    page = doc.pages[0]
    assert page.ndim == 3
    assert page.shape[2] == 3


def test_ingest_rejects_empty_payload() -> None:
    with pytest.raises(CorruptFile):
        ingest(b"")


def test_ingest_rejects_unknown_mime() -> None:
    with pytest.raises(InvalidMime):
        ingest(b"not an image", mime_hint="application/octet-stream")
