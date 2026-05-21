"""Sample-based smoke tests for both detector backends.

Runs the full pipeline against the HSBC-style scanned forms in `samples/`,
parameterized over (sample, backend). The 10-signature montage
`Gemini_Generated_Image_8b41128b41128b41.png` is deliberately excluded
per the task brief — it's not a realistic single-form input.

Each (sample, backend) combo skips automatically if the backend's ONNX
file is not present in `$SIGCROP_MODEL_DIR`, so the test stays useful in
both ops-loaded and CI-without-weights environments.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sigcrop.api.schemas import CropOptions
from sigcrop.config import get_settings
from sigcrop.models.registry import REGISTRY
from sigcrop.pipeline.detector import get_detector
from sigcrop.pipeline.run import run_pipeline

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "samples"

# (filename, minimum expected signatures, description). The "minimum" is one
# below the human-visible count to leave room for the off-the-shelf models'
# known recall on out-of-domain HSBC layouts.
SAMPLE_EXPECTATIONS: list[tuple[str, int, str]] = [
    ("Gemini_Generated_Image_8oxn998oxn998oxn.png", 1, "HSBC AO page 4/6 (2 visible)"),
    ("Gemini_Generated_Image_i1gp79i1gp79i1gp.png", 1, "HSBC personal AO (2 visible)"),
    ("Gemini_Generated_Image_klc1ofklc1ofklc1.png", 1, "HSBC sect 6 (1 visible)"),
    ("Gemini_Generated_Image_lsvugqlsvugqlsvu.png", 1, "HSBC AO form (2 visible)"),
    ("Gemini_Generated_Image_qwgulfqwgulfqwgu.png", 1, "HSBC UK personal AO (3 visible)"),
]

BACKENDS = ["conditional-detr", "yolov8"]


def _backend_weights_present(backend_name: str) -> bool:
    from sigcrop.pipeline.detectors import get_backend

    record = REGISTRY.get(get_backend(backend_name).model_version)
    if record is None:
        return False
    return (get_settings().model_dir / record.filename).is_file()


@pytest.mark.integration
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("sample_name", "min_sigs", "_descr"), SAMPLE_EXPECTATIONS)
def test_detector_on_sample(backend: str, sample_name: str, min_sigs: int, _descr: str) -> None:
    if not _backend_weights_present(backend):
        pytest.skip(f"weights missing for backend {backend}")

    path = SAMPLES_DIR / sample_name
    assert path.is_file(), f"sample missing: {path}"

    resp = run_pipeline(
        data=path.read_bytes(),
        mime_hint="image/png",
        options=CropOptions(detector_backend=backend),
        request_id=f"req_{sample_name}_{backend}",
    )

    assert resp.page_count == 1
    assert resp.model_version == get_detector(backend).model_version
    assert len(resp.signatures) >= min_sigs, (
        f"{backend} on {sample_name}: got {len(resp.signatures)} sigs, "
        f"expected ≥ {min_sigs}"
    )

    default_conf = get_detector(backend).default_confidence
    for sig in resp.signatures:
        # Bbox sanity (not a detection-quality assertion).
        assert sig.bbox.x >= 0
        assert sig.bbox.y >= 0
        assert sig.bbox.w > 0
        assert sig.bbox.h > 0
        # Confidence at/above the backend's tuned default (postprocess filtered).
        assert sig.confidence >= default_conf
        # Normalized bbox stays in [0, 1].
        assert 0.0 <= sig.bbox_normalized.x <= 1.0
        assert 0.0 <= sig.bbox_normalized.y <= 1.0
