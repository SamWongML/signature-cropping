"""Pinned model metadata + SHA-256 verification at load time."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class ModelRecord:
    version: str
    filename: str
    sha256: str
    license: str
    training_lineage_hash: str
    metrics: dict[str, float]


# Append-only registry. Bumping a version always means a new entry, never an edit.
# SHA-256 is empty until ops runs scripts/fetch_pretrained.py and pins it.
REGISTRY: dict[str, ModelRecord] = {
    "conditional-detr-50-fp32-2026.05.20": ModelRecord(
        version="conditional-detr-50-fp32-2026.05.20",
        filename="conditional_detr_signature.onnx",
        sha256="071183e076187407d1638f8d37d4a00f30b3c25f89fac2460337430ce300c9ea",
        license="Apache-2.0",
        training_lineage_hash="tech4humans/conditional-detr-50-signature-detector",
        metrics={"map50": 0.937, "map50_95": 0.653},
    ),
}


def verify_weights(model_path: Path, expected_sha256: str) -> None:
    """Raise ValueError if the file at model_path does not match the pinned hash.

    An empty `expected_sha256` skips verification — used for dev/MVP where the
    operator has not yet pinned the hash after the first fetch.
    """
    if not expected_sha256:
        return
    h = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Model weight hash mismatch: expected {expected_sha256}, got {actual}"
        )
