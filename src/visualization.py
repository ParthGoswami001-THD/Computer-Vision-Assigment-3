"""@file visualization.py
@brief Visualization helpers for IEPS + SCF results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def to_bgr(image: np.ndarray) -> np.ndarray:
    """@brief Convert grayscale or BGR image to BGR for drawing.

    @param image Input image.
    @return BGR image.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def draw_points(image: np.ndarray, points: Iterable[Point], color=(0, 0, 255), radius: int = 3) -> np.ndarray:
    """@brief Draw points on an image.

    @param image Input image.
    @param points Points as (x, y).
    @param color BGR drawing color.
    @param radius Point radius.
    @return Output BGR image.
    """
    out = to_bgr(image)
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def draw_center(image: np.ndarray, center: Sequence[float], color=(0, 255, 255)) -> np.ndarray:
    """@brief Draw center of gravity.

    @param image Input image.
    @param center Center as (x, y).
    @param color BGR color.
    @return Output BGR image.
    """
    out = to_bgr(image)
    cv2.drawMarker(out, (int(round(center[0])), int(round(center[1]))), color, cv2.MARKER_CROSS, 12, 2)
    return out


def draw_scan_lines(
    image: np.ndarray,
    scan_lines: Iterable[Sequence[Point]],
    points: Iterable[Point] | None = None,
    center: Sequence[float] | None = None,
    line_color=(0, 200, 255),
    point_color=(0, 0, 255),
) -> np.ndarray:
    """@brief Draw IEPS initial scan lines with the selected points and center.

    @param image Input image.
    @param scan_lines Sampled scan lines, each a list of (x, y) points.
    @param points Optional points selected on the scan lines.
    @param center Optional scan-line origin as (x, y).
    @param line_color BGR scan-line color.
    @param point_color BGR selected-point color.
    @return Output BGR image.
    """
    out = to_bgr(image)
    for line in scan_lines:
        if len(line) >= 2:
            first, last = line[0], line[-1]
            cv2.line(out, (int(first[0]), int(first[1])), (int(last[0]), int(last[1])), line_color, 1, cv2.LINE_AA)
    if center is not None:
        cv2.drawMarker(out, (int(round(center[0])), int(round(center[1]))), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)
    if points:
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), 3, point_color, -1)
    return out


def draw_contour_points(image: np.ndarray, points: Iterable[Point], color=(255, 0, 0)) -> np.ndarray:
    """@brief Draw a polyline/point contour on an image.

    @param image Input image.
    @param points Contour points.
    @param color BGR color.
    @return Output BGR image.
    """
    out = to_bgr(image)
    pts = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=1)
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), 1, color, -1)
    return out


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    """@brief Overlay a binary mask on an image.

    @param image Input image.
    @param mask Binary mask.
    @param color BGR overlay color.
    @return Output BGR image.
    """
    out = to_bgr(image)
    out[mask > 0] = color
    return out


def save_image(path: Path, image: np.ndarray) -> None:
    """@brief Save image to disk, creating parent folders.

    @param path Output path.
    @param image Image to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def make_panel(images: list[np.ndarray], labels: list[str], tile_width: int = 256) -> np.ndarray:
    """@brief Create a horizontal labeled result panel.

    @param images List of images.
    @param labels List of labels.
    @param tile_width Width of each tile.
    @return BGR panel image.
    """
    tiles = []
    for img, label in zip(images, labels):
        tile = to_bgr(img)
        scale = tile_width / tile.shape[1]
        tile = cv2.resize(tile, (tile_width, int(tile.shape[0] * scale)))
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (255, 255, 255), -1)
        cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    return cv2.hconcat(tiles)
