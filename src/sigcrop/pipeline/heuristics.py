"""Heuristic candidate-region pre-filter.

Used as a cheap blank-page detector before the ONNX session runs. When no
candidates are returned, the pipeline skips the detector for that page —
saving 400 ms on a fully blank scan.

Algorithm:
1. Adaptive threshold → binary ink mask.
2. Remove long printed form-lines via horizontal/vertical morphological
   opening with structuring elements ~1/30 of page width/height.
3. Connected-component labelling.
4. Keep components whose bbox area is in [median*4, median*50] AND stroke
   density is in [0.05, 0.30] AND aspect ratio is not absurdly thin.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True, frozen=True)
class CandidateROI:
    x: int
    y: int
    w: int
    h: int
    stroke_density: float


def _binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=9,
    )


def _remove_form_lines(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    h_len = max(15, w // 30)
    v_len = max(15, h // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines = cv2.bitwise_or(h_lines, v_lines)
    return cv2.bitwise_and(binary, cv2.bitwise_not(lines))


def find_candidate_regions(page_bgr: np.ndarray) -> list[CandidateROI]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    binary = _binarize(gray)
    clean = _remove_form_lines(binary)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        clean, connectivity=8
    )
    if num_labels <= 1:
        return []

    page_area = page_bgr.shape[0] * page_bgr.shape[1]
    lo = 100
    hi = int(page_area * 0.05)

    candidates: list[CandidateROI] = []
    for i in range(1, num_labels):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < lo or area > hi:
            continue
        if w == 0 or h == 0:
            continue
        if w > h * 30 or h > w * 10:
            continue
        density = area / float(w * h)
        if density < 0.05 or density > 0.50:
            continue
        candidates.append(CandidateROI(x=x, y=y, w=w, h=h, stroke_density=density))

    return candidates
