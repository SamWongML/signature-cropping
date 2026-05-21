"""Deterministic synthetic fixture for the pipeline tests.

Generates a ~2200x1700 PNG that looks vaguely like a scanned form: a few
horizontal printed-line bars, a "Signature" label box, and a hand-drawn ink
scribble inside it. Same seed → same image, every time.
"""

from __future__ import annotations

import io
import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw

PAGE_W = 1700
PAGE_H = 2200
SEED = 42


def _scribble(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
              rng: random.Random) -> None:
    """Bezier-ish scribble inside the given box."""
    cx = x + w // 2
    cy = y + h // 2
    points: list[tuple[float, float]] = []
    for t in range(0, 200):
        u = t / 200.0
        px = cx + (math.sin(u * 8 + rng.random()) * w * 0.35)
        py = cy + (math.cos(u * 11 + rng.random()) * h * 0.25)
        points.append((px, py))
    for i in range(len(points) - 1):
        draw.line([points[i], points[i + 1]], fill=(0, 0, 0), width=4)


def render(page_w: int = PAGE_W, page_h: int = PAGE_H, seed: int = SEED) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # A few printed horizontal lines so heuristics has form-lines to strip.
    for y in (300, 500, 700, 900, 1100, 1300, 1500, 1700):
        draw.line([(120, y), (page_w - 120, y)], fill=(50, 50, 50), width=2)

    # Faux labels (just rectangles, no real text — keep deps minimal).
    for ry in (250, 450, 650, 850, 1050, 1250):
        draw.rectangle(((140, ry), (240, ry + 20)), fill=(80, 80, 80))

    # Signature box near the bottom, with a scribble inside it.
    sig_x, sig_y, sig_w, sig_h = 300, 1820, 700, 220
    draw.rectangle(((sig_x, sig_y), (sig_x + sig_w, sig_y + sig_h)),
                   outline=(50, 50, 50), width=2)
    _scribble(draw, sig_x + 40, sig_y + 30, sig_w - 80, sig_h - 60, rng)

    return img


def to_png_bytes() -> bytes:
    img = render()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def to_bgr_ndarray() -> np.ndarray:
    img = render()
    arr = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
