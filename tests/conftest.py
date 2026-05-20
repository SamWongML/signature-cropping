"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sigcrop.api.app import create_app
from sigcrop.config import get_settings


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Gate model-dependent tests on whether the ONNX file is present locally.

    - `needs_model` tests skip when the file is missing.
    - Tests whose name ends with `_without_model` skip when the file IS present —
      they exercise the negative path and would otherwise return 200.
    """
    has_model = Path(get_settings().model_path).is_file()
    skip_needs = pytest.mark.skip(reason="needs_model: ONNX file not present in models/")
    skip_without = pytest.mark.skip(reason="model is present; without_model paths inverted")
    for item in items:
        if "needs_model" in item.keywords and not has_model:
            item.add_marker(skip_needs)
        if item.name.endswith("_without_model") and has_model:
            item.add_marker(skip_without)
