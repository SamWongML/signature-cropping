"""FastMCP stdio server entrypoint.

Run with: `python -m sigcrop.mcp.server` or `sigcrop-mcp`.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from sigcrop.api.schemas import CropResponse, ModelInfo
from sigcrop.logging import configure as configure_logging
from sigcrop.mcp.tools import (
    crop_signature_tool,
    get_model_info_tool,
    list_signature_regions_tool,
)

mcp = FastMCP("sigcrop")


@mcp.tool()
async def crop_signature(file_uri: str, options: dict[str, Any] | None = None) -> CropResponse:
    """Crop every handwritten signature from a scanned form.

    file_uri: file://, s3://, or data:application/pdf;base64,...
    options: see CropOptions schema (confidence_threshold, padding_pct, ...)
    """
    return await crop_signature_tool(file_uri, options)


@mcp.tool()
async def list_signature_regions(file_uri: str) -> dict[str, Any]:
    """Return signature bounding boxes only (no pixel payload).

    Cheaper than crop_signature when the caller just needs coordinates.
    """
    return await list_signature_regions_tool(file_uri)


@mcp.tool()
def get_model_info() -> ModelInfo:
    """Return the active model version, training-lineage hash, and metrics."""
    return get_model_info_tool()


def run() -> None:
    """Console-script entrypoint: `sigcrop-mcp`."""
    configure_logging()
    mcp.run()


if __name__ == "__main__":
    run()
