"""MCP tool surface tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigcrop.errors import CorruptFile, InvalidMime
from sigcrop.mcp.tools import _resolve_file_uri, get_model_info_tool


def test_get_model_info_returns_license() -> None:
    info = get_model_info_tool()
    assert info.license == "Apache-2.0"
    assert info.model_version


def test_resolve_data_uri_base64() -> None:
    payload = b"hello world"
    import base64

    uri = f"data:application/octet-stream;base64,{base64.b64encode(payload).decode()}"
    data, mime = _resolve_file_uri(uri)
    assert data == payload
    assert mime == "application/octet-stream"


def test_resolve_unknown_scheme_rejected() -> None:
    with pytest.raises(InvalidMime):
        _resolve_file_uri("ftp://example.com/x.png")


def test_resolve_missing_file_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.png"
    with pytest.raises(CorruptFile):
        _resolve_file_uri(f"file://{missing}")
