"""Typed errors. Mapped to HTTP / MCP codes only at the boundary layer."""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_MIME = "INVALID_MIME"
    CORRUPT_FILE = "CORRUPT_FILE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    PAGE_UNREADABLE = "PAGE_UNREADABLE"
    LOW_CONTRAST = "LOW_CONTRAST"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RATE_LIMITED = "RATE_LIMITED"


class SigcropError(Exception):
    code: ErrorCode
    http_status: int
    retryable: bool

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidMime(SigcropError):
    code = ErrorCode.INVALID_MIME
    http_status = 400
    retryable = False


class CorruptFile(SigcropError):
    code = ErrorCode.CORRUPT_FILE
    http_status = 400
    retryable = False


class PayloadTooLarge(SigcropError):
    code = ErrorCode.PAYLOAD_TOO_LARGE
    http_status = 413
    retryable = False


class PageUnreadable(SigcropError):
    code = ErrorCode.PAGE_UNREADABLE
    http_status = 422
    retryable = False


class LowContrast(SigcropError):
    code = ErrorCode.LOW_CONTRAST
    http_status = 422
    retryable = False


class ModelUnavailable(SigcropError):
    code = ErrorCode.MODEL_UNAVAILABLE
    http_status = 503
    retryable = True


class RateLimited(SigcropError):
    code = ErrorCode.RATE_LIMITED
    http_status = 429
    retryable = True
