"""Runtime configuration. Env-driven via `SIGCROP_*` variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_thread_count(cores: int) -> int:
    """Cap at 8 — Conditional-DETR-R50 stops scaling past ~8 threads (HT hurts)."""
    return max(1, min(8, cores))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIGCROP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service
    log_level: str = "INFO"
    request_id_header: str = "x-request-id"
    max_upload_mb: int = 25
    max_async_upload_mb: int = 200

    # Model
    model_dir: Path = Path("/opt/sigcrop/models")
    model_file: str = "conditional_detr_signature.onnx"
    model_version: str = "conditional-detr-50-fp32-2026.05.20"
    model_hf_id: str = "tech4humans/conditional-detr-50-signature-detector"
    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1

    @field_validator("intra_op_num_threads", mode="before")
    @classmethod
    def _expand_auto_threads(cls, v: object) -> object:
        if isinstance(v, str) and v.strip().lower() == "auto":
            return _resolve_thread_count(os.cpu_count() or 2)
        return v
    detector_input_size: int = 640

    # Detection defaults (overridable per-request)
    confidence_threshold: float = 0.55
    nms_iou: float = 0.5
    padding_pct: float = 0.08

    # Preprocessing
    render_dpi: int = 300
    max_skew_deg: float = 5.0
    letterbox_size: int = 640

    # S3 (optional)
    s3_bucket: str = ""
    s3_kms_key_id: str = ""

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_file


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
