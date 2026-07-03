"""@file geometry.py
@brief Geometry utilities for IEPS/SCF: moments, scan-line sampling, ordering.

The paper uses mathematical coordinates (x, y), while NumPy/OpenCV images are
indexed as image[y, x]. All public functions in this module accept and return
points as (x, y) to stay close to the paper notation.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]


def compute_center_of_gravity(image: np.ndarray, mode: str = "contrast") -> FloatPoint:
    """@brief Compute image/object center of gravity using spatial moments.

    @param image Input grayscale image.
    @param mode Moment mode:
        - "raw": use image intensities directly, matching Eq. (1)-(3).
        - "contrast": subtract image minimum before moments. This is the
          default because non-zero synthetic backgrounds bias the raw moment.
        - "binary": threshold the image by its mean and use a binary foreground.
    @return Center of gravity as (x_c, y_c).
    """
    img = image.astype(np.float64)
    if mode == "contrast":
        img = img - float(img.min())
    elif mode == "binary":
        threshold = 0.5 * (float(img.min()) + float(img.max()))
        img = (img > threshold).astype(np.float64)
    elif mode != "raw":
        raise ValueError(f"Unknown center-of-gravity mode: {mode}")

    height, width = img.shape[:2]
    ys, xs = np.indices((height, width), dtype=np.float64)
    m00 = img.sum()
    if m00 <= 1e-12:
        return (width / 2.0, height / 2.0)
    return (float((xs * img).sum() / m00), float((ys * img).sum() / m00))


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    """@brief Compute Euclidean distance between two 2D points.

    @param p1 First point as (x, y).
    @param p2 Second point as (x, y).
    @return Euclidean distance.
    """
    return float(math.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


def in_bounds(point: Point, image_shape: Tuple[int, int]) -> bool:
    """@brief Check if a point is inside image bounds.

    @param point Point as (x, y).
    @param image_shape Image shape as (height, width).
    @return True when point is inside image.
    """
    x, y = point
    height, width = image_shape[:2]
    return 0 <= x < width and 0 <= y < height


def unique_preserve_order(points: Iterable[Point], min_distance: float = 0.0) -> List[Point]:
    """@brief Remove duplicate/near-duplicate points without changing order.

    @param points Input points.
    @param min_distance Minimum allowed distance between retained points.
    @return Ordered unique point list.
    """
    out: List[Point] = []
    for x, y in points:
        p = (int(x), int(y))
        if all(euclidean_distance(p, q) > min_distance for q in out):
            out.append(p)
    return out


def sample_ray_from_center(
    center: Sequence[float],
    angle_rad: float,
    image_shape: Tuple[int, int],
    max_distance: float | None = None,
) -> List[Point]:
    """@brief Sample integer pixels on a ray from a center point.

    @param center Ray origin as (x, y).
    @param angle_rad Ray angle in radians. x grows right, y grows downward.
    @param image_shape Image shape as (height, width).
    @param max_distance Optional maximum sampling distance.
    @return Unique sampled points as (x, y).
    """
    height, width = image_shape[:2]
    cx, cy = float(center[0]), float(center[1])
    if max_distance is None:
        max_distance = math.hypot(width, height)
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    points: List[Point] = []
    seen = set()
    for r in np.arange(0.0, max_distance + 1.0, 1.0):
        point = (int(round(cx + dx * r)), int(round(cy + dy * r)))
        if not in_bounds(point, image_shape):
            if r > 0:
                break
            continue
        if point not in seen:
            points.append(point)
            seen.add(point)
    return points


def sample_line_through_point(
    midpoint: Sequence[float],
    direction: Sequence[float],
    image_shape: Tuple[int, int],
    radius: float,
) -> List[Point]:
    """@brief Sample a finite line segment through a point.

    @param midpoint Base point M_i^(p) as (x, y).
    @param direction Direction vector of the scan line.
    @param image_shape Image shape as (height, width).
    @param radius Scan radius on both sides of the midpoint.
    @return Unique sampled points as (x, y).
    """
    mx, my = float(midpoint[0]), float(midpoint[1])
    dx, dy = float(direction[0]), float(direction[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-12:
        return []
    dx, dy = dx / norm, dy / norm
    points: List[Point] = []
    seen = set()
    for t in np.arange(-radius, radius + 1.0, 1.0):
        point = (int(round(mx + dx * t)), int(round(my + dy * t)))
        if in_bounds(point, image_shape) and point not in seen:
            points.append(point)
            seen.add(point)
    return points


def sort_points_by_angle(points: Iterable[Point], center: Sequence[float]) -> List[Point]:
    """@brief Sort points by polar angle around a center.

    @param points Input points as (x, y).
    @param center Center as (x, y).
    @return Angle-sorted unique points.
    """
    cx, cy = float(center[0]), float(center[1])
    unique = unique_preserve_order(points)
    return sorted(unique, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def midpoint(p1: Sequence[float], p2: Sequence[float]) -> FloatPoint:
    """@brief Compute midpoint between two points.

    @param p1 First point.
    @param p2 Second point.
    @return Midpoint as (x, y).
    """
    return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)


def normal_vector(p1: Sequence[float], p2: Sequence[float]) -> FloatPoint:
    """@brief Compute one normal vector to segment p1-p2.

    @param p1 First segment endpoint.
    @param p2 Second segment endpoint.
    @return Normal vector as (nx, ny).
    """
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    return (-dy, dx)
