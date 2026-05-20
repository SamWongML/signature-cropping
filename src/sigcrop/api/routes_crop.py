"""POST /v1/crop-signature — synchronous extraction for files up to 25 MB."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from sigcrop.api.deps import enforce_size_limit, request_id, require_service_auth
from sigcrop.api.schemas import CropOptions, CropResponse, ReturnFormat
from sigcrop.config import get_settings
from sigcrop.errors import PayloadTooLarge, SigcropError
from sigcrop.pipeline.run import run_pipeline

router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_service_auth), Depends(enforce_size_limit)],
)


@router.post("/crop-signature", response_model=CropResponse)
async def crop_signature(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.55),
    padding_pct: float = Form(0.08),
    apply_mask: bool = Form(False),
    return_format: ReturnFormat = Form(ReturnFormat.INLINE_B64),
    s3_prefix: str | None = Form(None),
    req_id: str = Depends(request_id),
) -> CropResponse:
    options = CropOptions(
        confidence_threshold=confidence_threshold,
        padding_pct=padding_pct,
        apply_mask=apply_mask,
        return_format=return_format,
        s3_prefix=s3_prefix,
        request_id=req_id,
    )
    try:
        return await _run(file, options, req_id)
    except SigcropError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={
                "error_code": exc.code.value,
                "message": exc.message,
                "request_id": req_id,
                "retryable": exc.retryable,
            },
        ) from exc


async def _run(file: UploadFile, options: CropOptions, req_id: str) -> CropResponse:
    settings = get_settings()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise PayloadTooLarge(f"File exceeds {settings.max_upload_mb} MB limit")
    return run_pipeline(data, mime_hint=file.content_type, options=options, request_id=req_id)
