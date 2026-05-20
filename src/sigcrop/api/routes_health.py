"""Liveness, readiness, and model-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from sigcrop.api.schemas import ModelInfo
from sigcrop.config import get_settings
from sigcrop.errors import ModelUnavailable
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
    settings = get_settings()
    return ModelInfo(
        model_version=settings.model_version,
        training_lineage_hash="",
        metrics={},
        license="Apache-2.0",
    )
