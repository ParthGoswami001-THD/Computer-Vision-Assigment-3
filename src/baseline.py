"""@file baseline.py
@brief Secondary Canny + OpenCV contour baseline.

This baseline is included only as a practical reference. It is not the main
research contribution of the assignment.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def canny_contour_baseline(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> Tuple[np.ndarray, List[Point]]:
    """@brief Run Canny edge detection and OpenCV contour extraction.

    @param image Input grayscale image.
    @param low_threshold Canny low threshold.
    @param high_threshold Canny high threshold.
    @return Tuple (contour_mask, contour_points) for largest external contour.
    """
    edges = cv2.Canny(image, low_threshold, high_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(image, dtype=np.uint8)
    if not contours:
        return mask, []
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, 255, 1)
    points = [(int(p[0][0]), int(p[0][1])) for p in largest]
    return mask, points
