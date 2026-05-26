"""Behavioral tests for bench/latency.py runtime paths.

Exercises the parts that normally need ONNX weights — `bench_backend`'s
aggregation loop and `main`'s backend selection / exit codes / JSON artifact —
by injecting a fake detector and stubbing `run_pipeline`. No weights required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from sigcrop.api.schemas import BBox, BBoxNorm, CropResponse, SignatureResult, TimingMs


def _fake_detector() -> SimpleNamespace:
    return SimpleNamespace(
        warm_up=lambda: None,
        model_version="fake-v1",
        model_license="MIT",
        default_confidence=0.3,
        default_nms_iou=0.5,
    )


def _fake_response(confs: list[float], pre: int = 10, inf: int = 20, post: int = 1) -> CropResponse:
    sigs = [
        SignatureResult(
            id=f"s{i}",
            page=1,
            bbox=BBox(x=0, y=0, w=10, h=10),
            bbox_normalized=BBoxNorm(x=0.0, y=0.0, w=0.1, h=0.1),
            confidence=c,
        )
        for i, c in enumerate(confs)
    ]
    return CropResponse(
        request_id="r",
        model_version="fake-v1",
        page_count=1,
        signatures=sigs,
        timing_ms=TimingMs(preprocess=pre, inference=inf, postprocess=post),
    )


def test_bench_backend_aggregates(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(b"x")
    corpus = sorted(tmp_path.glob("*.png"))
    monkeypatch.setattr(latency, "get_detector", lambda _name: _fake_detector())
    monkeypatch.setattr(latency, "run_pipeline", lambda *_a, **_k: _fake_response([0.9, 0.4]))
    contract = latency.BenchContract(warmup_iters=1, measured_iters=3)

    r = latency.bench_backend("yolov8", corpus, contract)

    assert r.model_version == "fake-v1"
    assert r.license == "MIT"
    assert r.latency_ms["n"] == 3 * 2  # measured_iters x images (warm-up excluded)
    assert r.stage_ms_mean == {"preprocess": 10.0, "inference": 20.0, "postprocess": 1.0}
    assert [d["count"] for d in r.detections] == [2, 2]
    assert r.detections[0]["confidences"] == [0.9, 0.4]
    assert r.gate_pass is True
    assert r.gate_violations == []


def test_bench_backend_flags_gate_failure(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.png").write_bytes(b"x")
    corpus = sorted(tmp_path.glob("*.png"))
    monkeypatch.setattr(latency, "get_detector", lambda _name: _fake_detector())
    monkeypatch.setattr(latency, "run_pipeline", lambda *_a, **_k: _fake_response([0.9]))
    zero_gate = latency.LatencyGate(p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, cold_start_ms=0.0)
    contract = latency.BenchContract(warmup_iters=0, measured_iters=2, gate=zero_gate)

    r = latency.bench_backend("yolov8", corpus, contract)

    assert r.gate_pass is False
    assert r.gate_violations


def _install_fake_bench(
    latency: ModuleType, monkeypatch: pytest.MonkeyPatch, *, gate_pass: bool
) -> None:
    def fake(name: str, corpus: list[Path], contract: object) -> object:
        return latency.BackendResult(
            model_version=f"{name}-v1",
            license="MIT",
            default_conf=0.3,
            default_iou=0.5,
            cold_start_ms=2.0,
            latency_ms={"n": 3.0, "min": 1.0, "p50": 2.0, "p95": 3.0, "p99": 3.0,
                        "max": 3.0, "mean": 2.0},
            stage_ms_mean={"preprocess": 1.0, "inference": 1.0, "postprocess": 0.0},
            detections=[{"image": "a.png", "count": 1, "confidences": [0.9]}],
            gate_pass=gate_pass,
            gate_violations=[] if gate_pass else ["P50 999 > 380 ms"],
        )

    monkeypatch.setattr(latency, "bench_backend", fake)


def _prep_main(
    latency: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    *,
    present: bool,
) -> Path:
    (tmp_path / "a.png").write_bytes(b"x")
    monkeypatch.setattr(sys, "argv", ["latency.py", "--corpus", str(tmp_path), *argv])
    monkeypatch.setattr(latency, "available_backends", lambda: ["m1"])
    monkeypatch.setattr(latency, "_weights_present", lambda _name: present)
    return tmp_path


def test_main_exit_0_on_pass(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_main(latency, monkeypatch, tmp_path, [], present=True)
    _install_fake_bench(latency, monkeypatch, gate_pass=True)
    assert latency.main() == 0


def test_main_exit_1_on_gate_fail(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_main(latency, monkeypatch, tmp_path, [], present=True)
    _install_fake_bench(latency, monkeypatch, gate_pass=False)
    assert latency.main() == 1


def test_main_no_gate_suppresses_failure(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_main(latency, monkeypatch, tmp_path, ["--no-gate"], present=True)
    _install_fake_bench(latency, monkeypatch, gate_pass=False)
    assert latency.main() == 0


def test_main_exit_2_when_requested_backend_missing(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_main(latency, monkeypatch, tmp_path, ["--backend", "m1"], present=False)
    assert latency.main() == 2


def test_main_exit_0_when_no_backend_present(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _prep_main(latency, monkeypatch, tmp_path, [], present=False)
    assert latency.main() == 0


def test_main_writes_json_artifact(
    latency: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "artifact.json"
    _prep_main(latency, monkeypatch, tmp_path, ["--json", str(out)], present=True)
    _install_fake_bench(latency, monkeypatch, gate_pass=True)

    assert latency.main() == 0
    doc = json.loads(out.read_text())

    assert doc["contract_version"] == latency.CONTRACT.version
    assert "generated_at" in doc
    assert doc["config"]["n_images"] == 1
    assert doc["config"]["gate"]["p95_ms"] == 600.0
    assert set(doc["results"]) == {"m1"}
    res = doc["results"]["m1"]
    assert res["model_version"] == "m1-v1"
    assert res["gate_pass"] is True
    assert set(res["latency_ms"]) >= {"p50", "p95", "p99"}
