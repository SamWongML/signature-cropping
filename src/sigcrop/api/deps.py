"""FastAPI dependencies: request-id propagation, auth, size guards."""

from __future__ import annotations

import uuid

from fastapi import Header, HTTPException, Request

from sigcrop.config import get_settings


def request_id(
    x_request_id: str | None = Header(default=None),
) -> str:
    """Return the client-supplied request-id or a freshly-minted ULID-style id."""
    return x_request_id or f"req_{uuid.uuid4().hex}"


def enforce_size_limit(request: Request) -> None:
    """Reject requests exceeding the configured sync upload limit."""
    cl = request.headers.get("content-length")
    if cl is None:
        return
    settings = get_settings()
    if int(cl) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PAYLOAD_TOO_LARGE")


def require_service_auth(authorization: str | None = Header(default=None)) -> None:
    """Validate the bearer token. Real JWT verification belongs here.

    The scaffold accepts any non-empty token; production must verify an
    HMAC-signed JWT against a short-TTL key from Secrets Manager.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
