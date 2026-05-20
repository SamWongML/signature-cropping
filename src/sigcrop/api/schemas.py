"""Pydantic v2 request/response schemas shared by REST and MCP."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReturnFormat(StrEnum):
    INLINE_B64 = "inline_b64"
    S3 = "s3"


class CropOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    padding_pct: float = Field(default=0.08, ge=0.0, le=0.5)
    apply_mask: bool = False
    return_format: ReturnFormat = ReturnFormat.INLINE_B64
    s3_prefix: str | None = None
    request_id: str | None = None


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class BBoxNorm(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class SignatureResult(BaseModel):
    id: str
    page: int = Field(ge=1)
    bbox: BBox
    bbox_normalized: BBoxNorm
    confidence: float = Field(ge=0.0, le=1.0)
    crop_b64: str | None = None
    crop_s3_uri: str | None = None
    mask_applied: bool = False


class TimingMs(BaseModel):
    preprocess: int
    inference: int
    postprocess: int


class CropResponse(BaseModel):
    request_id: str
    model_version: str
    page_count: int = Field(ge=0)
    signatures: list[SignatureResult]
    timing_ms: TimingMs


class ErrorBody(BaseModel):
    error_code: str
    message: str
    request_id: str
    retryable: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    training_lineage_hash: str
    metrics: dict[str, float]
    license: str
