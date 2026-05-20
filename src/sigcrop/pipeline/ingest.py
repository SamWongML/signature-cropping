"""File ingest: bytes → list of page arrays (BGR uint8).

Supported MIME types:
- application/pdf      → PyMuPDF rasterizes each page at `settings.render_dpi`
- image/png, image/jpeg, image/tiff → Pillow/OpenCV decode

PyMuPDF returns RGB; we convert to BGR so downstream OpenCV ops see the
channel order they expect.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from sigcrop.config import get_settings
from sigcrop.errors import CorruptFile, InvalidMime

_SUPPORTED_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/tiff"})
_PDF_MIME = "application/pdf"


@dataclass(slots=True, frozen=True)
class IngestedDocument:
    pages: list[np.ndarray]  # BGR, uint8, original resolution
    dpi: int
    source_mime: str


def _sniff_mime(data: bytes, hint: str | None) -> str:
    if hint:
        h = hint.lower().split(";")[0].strip()
        if h == "image/jpg":
            h = "image/jpeg"
        if h == _PDF_MIME or h in _SUPPORTED_IMAGE_MIMES:
            return h

    # Magic-byte sniff. python-magic is optional; fall back to a tiny hand-rolled
    # check for the four headers we care about so the pipeline runs without it.
    try:
        import magic

        detected = magic.from_buffer(data[:4096], mime=True)
        if detected == "image/jpg":
            detected = "image/jpeg"
        if detected == _PDF_MIME or detected in _SUPPORTED_IMAGE_MIMES:
            return detected
    except Exception:  # noqa: BLE001, S110 — fall through to header sniff
        pass

    if data.startswith(b"%PDF"):
        return _PDF_MIME
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    raise InvalidMime("Unsupported or unrecognised file type")


def _decode_image(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptFile(f"Image decode failed: {exc}") from exc
    if img.mode != "RGB":
        img = img.convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _decode_pdf(data: bytes, dpi: int) -> list[np.ndarray]:
    import fitz

    pages: list[np.ndarray] = []
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — fitz raises a bare RuntimeError
        raise CorruptFile(f"PDF parse failed: {exc}") from exc

    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            pages.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    finally:
        doc.close()

    if not pages:
        raise CorruptFile("PDF contained zero pages")
    return pages


def ingest(data: bytes, mime_hint: str | None = None) -> IngestedDocument:
    """Sniff MIME and decode to a list of page ndarrays."""
    if not data:
        raise CorruptFile("Empty payload")

    mime = _sniff_mime(data, mime_hint)
    settings = get_settings()

    if mime == _PDF_MIME:
        pages = _decode_pdf(data, settings.render_dpi)
        return IngestedDocument(pages=pages, dpi=settings.render_dpi, source_mime=mime)

    page = _decode_image(data)
    return IngestedDocument(pages=[page], dpi=settings.render_dpi, source_mime=mime)
