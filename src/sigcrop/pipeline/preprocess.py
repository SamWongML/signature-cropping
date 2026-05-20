"""Per-page preprocessing: deskew, CLAHE, letterbox.

Operations (in order):
1. Deskew: angle from `cv2.minAreaRect` of the ink mask, capped at
   ±`settings.max_skew_deg`.
2. Contrast: CLAHE on the L channel of LAB.
3. Letterbox to a `letterbox_size` square, padded with 114-gray.
4. Normalize to NCHW float32 with ImageNet mean/std (matches the
   ConditionalDETR image processor configuration).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sigcrop.config import get_settings
from sigcrop.errors import LowContrast

# ImageNet normalization — used by every HF object-detection processor we ship.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(slots=True, frozen=True)
class LetterboxParams:
    scale: float
    pad_x: int
    pad_y: int
    src_w: int
    src_h: int


@dataclass(slots=True)
class PreprocessedPage:
    page_index: int
    src_bgr: np.ndarray
    model_input: np.ndarray
    letterbox: LetterboxParams
    rotation_applied: int
    skew_corrected_deg: float


def _estimate_skew_deg(gray: np.ndarray, max_deg: float) -> float:
    # Invert + threshold to a binary ink mask; minAreaRect on the largest
    # contour cluster gives the dominant text-line angle.
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(ink > 0))
    if coords.size == 0:
        return 0.0

    angle = cv2.minAreaRect(coords[:, ::-1])[-1]
    # minAreaRect returns angle in (-90, 0]; normalize to (-45, 45].
    if angle < -45:
        angle = 90 + angle
    angle = float(angle)
    if angle > max_deg:
        angle = max_deg
    elif angle < -max_deg:
        angle = -max_deg
    return angle


def _apply_skew(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return bgr
    h, w = bgr.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        bgr, matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    return cv2.cvtColor(cv2.merge([l_eq, a_chan, b_chan]), cv2.COLOR_LAB2BGR)


def _letterbox(bgr: np.ndarray, target: int) -> tuple[np.ndarray, LetterboxParams]:
    h, w = bgr.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target - new_w) // 2
    pad_y = (target - new_h) // 2
    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    return canvas, LetterboxParams(
        scale=scale, pad_x=pad_x, pad_y=pad_y, src_w=w, src_h=h
    )


def _to_nchw(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def preprocess_page(page_bgr: np.ndarray, page_index: int) -> PreprocessedPage:
    settings = get_settings()

    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean < 25 or lap_var < 25:
        raise LowContrast(
            f"Page {page_index} below contrast floor "
            f"(mean={mean:.1f}, laplacian_var={lap_var:.1f})"
        )

    skew = _estimate_skew_deg(gray, settings.max_skew_deg)
    deskewed = _apply_skew(page_bgr, -skew)
    contrasted = _apply_clahe(deskewed)
    letterbox_img, lb = _letterbox(contrasted, settings.detector_input_size)
    model_input = _to_nchw(letterbox_img)

    return PreprocessedPage(
        page_index=page_index,
        src_bgr=contrasted,
        model_input=model_input,
        letterbox=lb,
        rotation_applied=0,
        skew_corrected_deg=skew,
    )
