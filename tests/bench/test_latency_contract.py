"""Contract tests for bench/latency.py.

These tests ARE the strict contract: they pin the measurement protocol (gate
budget from docs/ARCHITECTURE.md §7, corpus exclusion, percentile method) and
the weight-presence / gate-evaluation logic, so the harness cannot drift or be
silently weakened without a failing test. They run without any ONNX weights —
only the pure, weight-independent surface of the harness is exercised.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_MONTAGE = "Gemini_Generated_Image_8b41128b41128b41.png"


def test_percentile_is_nearest_rank(latency: ModuleType) -> None:
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert latency._percentile(vals, 50) == 30.0
    assert latency._percentile(vals, 95) == 50.0
    assert latency._percentile(vals, 99) == 50.0
    assert latency._percentile([7.0], 50) == 7.0


def test_percentile_empty_is_nan(latency: ModuleType) -> None:
    assert math.isnan(latency._percentile([], 50))


def test_corpus_excludes_montage_and_non_png(latency: ModuleType, tmp_path: Path) -> None:
    for name in ("b.png", "a.png", _MONTAGE, "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    got = [p.name for p in latency._corpus(latency.CONTRACT, tmp_path)]
    assert got == ["a.png", "b.png"]  # sorted, montage + txt dropped


def test_corpus_empty_raises(latency: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no corpus"):
        latency._corpus(latency.CONTRACT, tmp_path)


def test_weights_present_reflects_file(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(latency, "get_settings", lambda: SimpleNamespace(model_dir=tmp_path))
    assert latency._weights_present("yolov8") is False
    record = latency.REGISTRY[latency.get_backend("yolov8").model_version]
    (tmp_path / record.filename).write_bytes(b"fake-onnx")
    assert latency._weights_present("yolov8") is True


def test_gate_passes_under_budget(latency: ModuleType) -> None:
    gate = latency.LatencyGate()
    under = {"p50": 200.0, "p95": 300.0, "p99": 400.0}
    assert latency.evaluate_gate(under, cold_start_ms=1_000.0, gate=gate) == []


@pytest.mark.parametrize(
    ("lat", "cold", "needle"),
    [
        ({"p50": 999.0, "p95": 300.0, "p99": 400.0}, 1_000.0, "P50"),
        ({"p50": 200.0, "p95": 999.0, "p99": 400.0}, 1_000.0, "P95"),
        ({"p50": 200.0, "p95": 300.0, "p99": 9_999.0}, 1_000.0, "P99"),
        ({"p50": 200.0, "p95": 300.0, "p99": 400.0}, 99_999.0, "cold start"),
    ],
)
def test_gate_flags_specific_violation(
    latency: ModuleType, lat: dict[str, float], cold: float, needle: str
) -> None:
    violations = latency.evaluate_gate(lat, cold_start_ms=cold, gate=latency.LatencyGate())
    assert len(violations) == 1
    assert needle in violations[0]


def test_contract_locks_section7_budget(latency: ModuleType) -> None:
    gate = latency.LatencyGate()
    # docs/ARCHITECTURE.md §7, MVP tier — must not be silently loosened.
    assert (gate.p50_ms, gate.p95_ms, gate.p99_ms, gate.cold_start_ms) == (
        380.0,
        600.0,
        900.0,
        12_000.0,
    )
    assert _MONTAGE in latency.CONTRACT.corpus_exclude
    assert latency.CONTRACT.warmup_iters == 5
    assert latency.CONTRACT.measured_iters == 30
