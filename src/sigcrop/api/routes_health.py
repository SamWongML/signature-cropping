"""Liveness, readiness, and model-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sigcrop.api.schemas import ModelInfo
from sigcrop.errors import ModelUnavailable
from sigcrop.models.registry import REGISTRY
from sigcrop.pipeline.detector import get_detector

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz() -> dict[str, str]:
    try:
        get_detector().warm_up()
    except (ModelUnavailable, NotImplementedError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@router.get("/v1/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    detector = get_detector()
    record = REGISTRY.get(detector.model_version)
    return ModelInfo(
        model_version=detector.model_version,
        training_lineage_hash=detector.training_lineage_hash,
        metrics=record.metrics if record is not None else {},
        license=detector.model_license,
    )
