"""End-to-end pipeline test — needs a real model in models/."""

from __future__ import annotations

import pytest


@pytest.mark.needs_model
@pytest.mark.integration
def test_e2e_single_page_png() -> None:
    from sigcrop.api.schemas import CropOptions
    from sigcrop.pipeline.run import run_pipeline
    from tests.fixtures._synthesize import to_png_bytes

    resp = run_pipeline(
        data=to_png_bytes(),
        mime_hint="image/png",
        options=CropOptions(),
        request_id="req_test",
    )
    assert resp.page_count == 1
    assert resp.model_version
    # Detector may or may not flag the synthetic scribble; just confirm shape.
    assert isinstance(resp.signatures, list)
