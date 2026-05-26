"""Shared loader for the (non-package) bench/latency.py module."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def latency() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_latency", _REPO / "bench" / "latency.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod
