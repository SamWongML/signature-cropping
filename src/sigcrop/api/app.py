"""FastAPI factory + uvicorn entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from sigcrop.api.routes_crop import router as crop_router
from sigcrop.api.routes_health import router as health_router
from sigcrop.logging import configure as configure_logging
from sigcrop.logging import get_logger
from sigcrop.pipeline.detector import get_detector

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    try:
        get_detector().warm_up()
    except NotImplementedError:
        log.warning("detector.warm_up not implemented; /readyz will fail")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="sigcrop",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.include_router(health_router)
    app.include_router(crop_router)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()


def run() -> None:
    """Console-script entrypoint: `sigcrop-api`."""
    import uvicorn

    uvicorn.run(
        "sigcrop.api.app:app",
        host="0.0.0.0",  # noqa: S104 — container listens on all interfaces
        port=8080,
        workers=1,
        log_config=None,
    )
