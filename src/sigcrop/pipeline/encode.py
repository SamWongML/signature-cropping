"""Encode cropped signatures to PNG bytes; return inline base64 or S3 URI."""

from __future__ import annotations

import base64
import uuid
from dataclasses import dataclass

import cv2

from sigcrop.config import get_settings
from sigcrop.pipeline.postprocess import Signature


@dataclass(slots=True, frozen=True)
class EncodedSignature:
    page: int
    x: int
    y: int
    w: int
    h: int
    confidence: float
    mask_applied: bool
    crop_b64: str | None
    crop_s3_uri: str | None


def _encode_png(sig: Signature) -> bytes:
    ok, buf = cv2.imencode(".png", sig.crop_bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return bytes(buf)


def _put_s3(png_bytes: bytes, s3_prefix: str) -> str:
    import boto3

    settings = get_settings()
    if not settings.s3_bucket:
        raise RuntimeError("S3 return_format requested but SIGCROP_S3_BUCKET is empty")

    key = f"{s3_prefix.rstrip('/')}/{uuid.uuid4().hex}.png"
    extra: dict[str, str] = {}
    if settings.s3_kms_key_id:
        extra["ServerSideEncryption"] = "aws:kms"
        extra["SSEKMSKeyId"] = settings.s3_kms_key_id
    else:
        extra["ServerSideEncryption"] = "AES256"

    boto3.client("s3").put_object(
        Bucket=settings.s3_bucket, Key=key, Body=png_bytes, **extra
    )
    return f"s3://{settings.s3_bucket}/{key}"


def encode_signatures(
    signatures: list[Signature],
    *,
    return_format: str,
    s3_prefix: str | None,
) -> list[EncodedSignature]:
    out: list[EncodedSignature] = []
    for sig in signatures:
        png = _encode_png(sig)
        b64: str | None = None
        s3_uri: str | None = None
        if return_format == "s3":
            if not s3_prefix:
                raise ValueError("s3_prefix is required when return_format=s3")
            s3_uri = _put_s3(png, s3_prefix)
        else:
            b64 = base64.b64encode(png).decode("ascii")
        out.append(
            EncodedSignature(
                page=sig.page,
                x=sig.x,
                y=sig.y,
                w=sig.w,
                h=sig.h,
                confidence=sig.confidence,
                mask_applied=sig.mask_applied,
                crop_b64=b64,
                crop_s3_uri=s3_uri,
            )
        )
    return out
