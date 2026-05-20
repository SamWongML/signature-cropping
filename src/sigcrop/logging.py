"""Structured JSON logging.

No PII rule: never log raw image bytes, base64 crops, or pixel-level bbox
values together with a customer identifier. Bbox coordinates alone are fine.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from sigcrop.config import get_settings

_configured = False


def configure() -> None:
    global _configured
    if _configured:
        return

    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> Any:
    configure()
    return structlog.get_logger(name)
