"""MCP tool implementations. Both tools delegate to `sigcrop.pipeline.run`."""

from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sigcrop.api.schemas import CropOptions, CropResponse, ModelInfo
from sigcrop.errors import CorruptFile, InvalidMime
from sigcrop.pipeline.run import run_pipeline, run_pipeline_regions_only


def _resolve_file_uri(file_uri: str) -> tuple[bytes, str | None]:
    """Return (bytes, mime_hint). Supports file://, s3://, data:..."""
    if file_uri.startswith("data:"):
        try:
            header, encoded = file_uri.split(",", 1)
        except ValueError as exc:
            raise CorruptFile("Malformed data URI") from exc
        mime = header[5:].split(";")[0] or None
        is_b64 = ";base64" in header
        payload = base64.b64decode(encoded) if is_b64 else unquote(encoded).encode()
        return payload, mime

    parsed = urlparse(file_uri)
    if parsed.scheme in {"", "file"}:
        path = Path(unquote(parsed.path) if parsed.scheme else file_uri)
        if not path.is_file():
            raise CorruptFile(f"File not found: {path}")
        return path.read_bytes(), None

    if parsed.scheme == "s3":
        import boto3

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise CorruptFile(f"Malformed s3 URI: {file_uri}")
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return obj["Body"].read(), obj.get("ContentType")

    raise InvalidMime(f"Unsupported URI scheme: {parsed.scheme}")


async def crop_signature_tool(
    file_uri: str, options: dict[str, Any] | None = None
) -> CropResponse:
    opts = CropOptions.model_validate(options or {})
    data, mime = _resolve_file_uri(file_uri)
    req_id = opts.request_id or f"req_{uuid.uuid4().hex}"
    return run_pipeline(data, mime_hint=mime, options=opts, request_id=req_id)


async def list_signature_regions_tool(file_uri: str) -> dict[str, Any]:
    data, mime = _resolve_file_uri(file_uri)
    opts = CropOptions()
    req_id = f"req_{uuid.uuid4().hex}"
    resp = run_pipeline_regions_only(data, mime_hint=mime, options=opts, request_id=req_id)
    return {
        "regions": [
            {
                "page": s.page,
                "bbox": s.bbox.model_dump(),
                "confidence": s.confidence,
            }
            for s in resp.signatures
        ],
        "page_count": resp.page_count,
        "model_version": resp.model_version,
    }


def get_model_info_tool() -> ModelInfo:
    from sigcrop.models.registry import REGISTRY
    from sigcrop.pipeline.detector import get_detector

    detector = get_detector()
    record = REGISTRY.get(detector.model_version)
    metrics = record.metrics if record is not None else {}
    return ModelInfo(
        model_version=detector.model_version,
        training_lineage_hash=detector.training_lineage_hash,
        metrics=metrics,
        license=detector.model_license,
    )
