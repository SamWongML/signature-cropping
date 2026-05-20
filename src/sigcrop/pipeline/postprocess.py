"""Postprocess detector output → source-space signature crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from sigcrop.pipeline.detector import Detection
from sigcrop.pipeline.preprocess import LetterboxParams

_MIN_BBOX_SIDE = 20


@dataclass(slots=True, frozen=True)
class Signature:
    page: int
    x: int
    y: int
    w: int
    h: int
    confidence: float
    crop_bgr: np.ndarray
    mask_applied: bool


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]

    return keep


def _remap_to_source(box: tuple[float, float, float, float], lb: LetterboxParams
                     ) -> tuple[int, int, int, int]:
    lx, ly, lw, lh = box
    sx = (lx - lb.pad_x) / lb.scale
    sy = (ly - lb.pad_y) / lb.scale
    sw = lw / lb.scale
    sh = lh / lb.scale
    return int(round(sx)), int(round(sy)), int(round(sw)), int(round(sh))


def _pad_and_clamp(x: int, y: int, w: int, h: int, src_w: int, src_h: int,
                   padding_pct: float) -> tuple[int, int, int, int]:
    pad = int(round(padding_pct * min(w, h)))
    nx = max(0, x - pad)
    ny = max(0, y - pad)
    nw = min(src_w - nx, w + 2 * pad)
    nh = min(src_h - ny, h + 2 * pad)
    return nx, ny, max(0, nw), max(0, nh)


def _apply_central_mask(crop: np.ndarray) -> np.ndarray:
    """Return BGRA with form-line bleed suppressed.

    Keeps connected components that touch the central 60% rectangle of the
    crop — the assumption is the signature is centered after padding.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    num, labels, _stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h, w = crop.shape[:2]
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    central = labels[cy0:cy1, cx0:cx1]
    keep_labels = set(np.unique(central).tolist()) - {0}

    if not keep_labels:
        # Nothing in the center — fall back to the whole ink mask.
        keep_mask = binary
    else:
        keep_mask = np.isin(labels, list(keep_labels)).astype(np.uint8) * 255

    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = keep_mask
    _ = num  # silence unused warning
    return bgra


def postprocess_detections(
    detections: list[Detection],
    page_bgr: np.ndarray,
    letterbox: LetterboxParams,
    page_index: int,
    *,
    confidence_threshold: float,
    nms_iou: float,
    padding_pct: float,
    apply_mask: bool,
) -> list[Signature]:
    if not detections:
        return []

    scored = [d for d in detections if d.confidence >= confidence_threshold]
    if not scored:
        return []

    boxes = np.array([[d.x, d.y, d.w, d.h] for d in scored], dtype=np.float32)
    scores = np.array([d.confidence for d in scored], dtype=np.float32)
    keep = _nms(boxes, scores, nms_iou)

    src_h, src_w = page_bgr.shape[:2]
    out: list[Signature] = []
    for idx in keep:
        sx, sy, sw, sh = _remap_to_source(tuple(boxes[idx].tolist()), letterbox)
        sx, sy, sw, sh = _pad_and_clamp(sx, sy, sw, sh, src_w, src_h, padding_pct)
        if sw < _MIN_BBOX_SIDE or sh < _MIN_BBOX_SIDE:
            continue
        crop = page_bgr[sy : sy + sh, sx : sx + sw].copy()
        if crop.size == 0:
            continue
        if apply_mask:
            crop = _apply_central_mask(crop)
        out.append(
            Signature(
                page=page_index,
                x=sx,
                y=sy,
                w=sw,
                h=sh,
                confidence=scored[idx].confidence,
                crop_bgr=crop,
                mask_applied=apply_mask,
            )
        )
    return out
