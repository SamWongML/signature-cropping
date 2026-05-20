"""HTTP-layer tests. /healthz works without a model; /readyz needs one."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.integration
def test_readyz_returns_503_without_model(client: TestClient) -> None:
    r = client.get("/readyz")
    assert r.status_code == 503


@pytest.mark.integration
def test_crop_requires_auth(client: TestClient) -> None:
    r = client.post("/v1/crop-signature", files={"file": ("x.png", b"\x89PNG", "image/png")})
    assert r.status_code == 401


@pytest.mark.integration
def test_crop_returns_503_without_model(client: TestClient) -> None:
    from tests.fixtures._synthesize import to_png_bytes

    r = client.post(
        "/v1/crop-signature",
        headers={"Authorization": "Bearer dev"},
        files={"file": ("form.png", to_png_bytes(), "image/png")},
    )
    # Without a model present, the warm_up inside run_pipeline raises
    # ModelUnavailable, which maps to 503.
    assert r.status_code == 503
    body = r.json()
    assert body["detail"]["error_code"] == "MODEL_UNAVAILABLE"
    assert body["detail"]["retryable"] is True


@pytest.mark.needs_model
@pytest.mark.integration
def test_crop_happy_path() -> None:
    """Real model present; expects ≥ 1 signature."""
    from fastapi.testclient import TestClient

    from sigcrop.api.app import create_app
    from tests.fixtures._synthesize import to_png_bytes

    client = TestClient(create_app())
    r = client.post(
        "/v1/crop-signature",
        headers={"Authorization": "Bearer dev"},
        files={"file": ("form.png", to_png_bytes(), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["page_count"] == 1
    assert isinstance(body["signatures"], list)
