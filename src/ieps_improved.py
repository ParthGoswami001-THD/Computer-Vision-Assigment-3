"""@file ieps_improved.py
@brief Traditional-CV IEPS extension for concave-shape robustness.

The paper-faithful IEPS implementation is kept in ieps.py. This module adds a
separate improved IEPS variant that stays inside the authors' direction:
automatic center/ray initialization plus geometric refinement, without deep
learning or unrelated preprocessing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .geometry import (
    compute_center_of_gravity,
    euclidean_distance,
    sample_ray_from_center,
    sort_points_by_angle,
    unique_preserve_order,
)
from .ieps import IEPSResult, _select_edge_candidate_on_line, run_ieps


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]


def _as_uint8_gray(image: np.ndarray) -> np.ndarray:
    """@brief Convert a grayscale image to uint8 for OpenCV thresholding.

    @param image Input grayscale image.
    @return uint8 grayscale image.
    """
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    min_val = float(img.min())
    max_val = float(img.max())
    if max_val <= min_val:
        return np.zeros_like(img, dtype=np.uint8)
    return (((img - min_val) / (max_val - min_val)) * 255.0).astype(np.uint8)


def robust_interior_seed(image: np.ndarray, center_mode: str = "contrast") -> Tuple[FloatPoint, Dict[str, object]]:
    """@brief Compute an IEPS seed that lies inside the object silhouette.

    The original paper emits scan lines from the center of gravity. That works
    for centrally filled, star-convex objects, but a concave U-shape can place
    the center of gravity in the background notch. This function keeps the CoG
    when it is already inside an Otsu foreground silhouette; otherwise it moves
    the seed to the distance-transform maximum of that silhouette.

    @param image Input grayscale image.
    @param center_mode Center-of-gravity mode passed to compute_center_of_gravity.
    @return Tuple (seed, debug dictionary).
    """
    center = compute_center_of_gravity(image, mode=center_mode)
    height, width = image.shape[:2]
    cx, cy = int(round(center[0])), int(round(center[1]))

    gray = _as_uint8_gray(image)
    if int(gray.max()) == int(gray.min()):
        return center, {
            "relocated": False,
            "reason": "flat_image",
            "old_center": center,
            "seed": center,
        }

    _threshold, silhouette = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if int(cv2.countNonZero(silhouette)) == 0:
        return center, {
            "relocated": False,
            "reason": "empty_silhouette",
            "old_center": center,
            "seed": center,
        }

    center_inside = 0 <= cx < width and 0 <= cy < height and silhouette[cy, cx] > 0
    if center_inside:
        return center, {
            "relocated": False,
            "reason": "center_inside_silhouette",
            "old_center": center,
            "seed": center,
        }

    distance = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(distance)
    if max_val <= 0:
        return center, {
            "relocated": False,
            "reason": "no_positive_distance",
            "old_center": center,
            "seed": center,
        }

    seed = (float(max_loc[0]), float(max_loc[1]))
    return seed, {
        "relocated": True,
        "reason": "distance_transform_seed",
        "old_center": center,
        "seed": seed,
    }


def order_points_boundary(points: Sequence[Point], seed: Sequence[float]) -> List[Point]:
    """@brief Order IEPS points with a simple boundary-following tour.

    Angular order is kept as the initial condition, then a greedy nearest
    neighbor traversal follows local boundary proximity. This is a compact
    traditional-CV fix for concave objects where angular neighbors can chord
    across a notch.

    @param points Input IEPS points.
    @param seed Interior seed used for initial angular ordering.
    @return Ordered point list.
    """
    ordered = sort_points_by_angle(points, seed)
    if len(ordered) < 3:
        return ordered

    remaining = ordered.copy()
    tour = [remaining.pop(0)]
    while remaining:
        last = tour[-1]
        next_index = min(range(len(remaining)), key=lambda idx: euclidean_distance(last, remaining[idx]))
        tour.append(remaining.pop(next_index))
    return tour


def _fill_coverage_gaps(
    points: Sequence[Point],
    gradient: np.ndarray,
    seed: Sequence[float],
    threshold: float,
    max_gap_degrees: float,
    fallback_mode: str,
) -> Tuple[List[Point], int]:
    """@brief Add extra radial samples in large angular gaps.

    @param points Existing IEPS points.
    @param gradient Sobel gradient magnitude image.
    @param seed Interior seed.
    @param threshold Gradient threshold.
    @param max_gap_degrees Maximum allowed angular gap.
    @param fallback_mode IEPS fallback mode.
    @return Tuple (point list, number of added points).
    """
    ordered = sort_points_by_angle(points, seed)
    if len(ordered) < 3:
        return list(ordered), 0

    angles = [math.atan2(p[1] - seed[1], p[0] - seed[0]) for p in ordered]
    extra: List[Point] = []
    for idx, angle0 in enumerate(angles):
        angle1 = angles[(idx + 1) % len(angles)]
        gap = (angle1 - angle0) % (2.0 * math.pi)
        gap_degrees = math.degrees(gap)
        if gap_degrees <= max_gap_degrees:
            continue
        extra_count = int(gap_degrees // max_gap_degrees)
        for step in range(1, extra_count + 1):
            angle = angle0 + gap * step / float(extra_count + 1)
            ray = sample_ray_from_center(seed, angle, gradient.shape)
            selected = _select_edge_candidate_on_line(
                ray,
                gradient,
                seed,
                threshold,
                selection="farthest_from_center",
                fallback_mode=fallback_mode,
            )
            if selected is not None:
                extra.append(selected)

    merged = unique_preserve_order(list(ordered) + extra, min_distance=1.5)
    return merged, len(merged) - len(ordered)


def run_ieps_improved(
    image: np.ndarray,
    gradient: np.ndarray,
    initial_scan_lines: int = 8,
    iterations: int = 3,
    threshold: float = 64.0,
    center_mode: str = "contrast",
    refinement_selection: str = "closest_to_reference",
    fallback_mode: str = "max_gradient",
    coverage_fill: bool = True,
    max_gap_degrees: float = 25.0,
) -> IEPSResult:
    """@brief Run the improved IEPS initialization extension.

    @param image Input grayscale image.
    @param gradient Sobel gradient magnitude image.
    @param initial_scan_lines Number of initial rays; default is denser than paper.
    @param iterations IEPS refinement iterations.
    @param threshold Gradient threshold.
    @param center_mode Center mode used before possible seed relocation.
    @param refinement_selection Refinement selection rule.
    @param fallback_mode Fallback mode when no thresholded candidate exists.
    @param coverage_fill Add rays in large angular gaps if True.
    @param max_gap_degrees Maximum gap before adding coverage rays.
    @return IEPSResult with improved points and debug information.
    """
    seed, seed_info = robust_interior_seed(image, center_mode=center_mode)
    base = run_ieps(
        image,
        gradient,
        initial_scan_lines=initial_scan_lines,
        iterations=iterations,
        threshold=threshold,
        center=seed,
        center_mode=center_mode,
        order_mode="angle",
        refinement_selection=refinement_selection,
        fallback_mode=fallback_mode,
    )

    points = base.points.copy()
    coverage_added = 0
    if coverage_fill:
        points, coverage_added = _fill_coverage_gaps(
            points,
            gradient,
            seed,
            threshold,
            max_gap_degrees=max_gap_degrees,
            fallback_mode=fallback_mode,
        )

    ordered_points = order_points_boundary(points, seed)
    debug = dict(base.debug)
    debug.update({
        "ieps_mode": "improved",
        "seed_info": seed_info,
        "coverage_fill": coverage_fill,
        "coverage_added": coverage_added,
        "max_gap_degrees": max_gap_degrees,
        "boundary_order": "nearest_neighbor",
        "points_before_boundary_order": points,
    })
    debug.setdefault("points_by_iteration", []).append(ordered_points.copy())
    return IEPSResult(points=ordered_points, center=seed, debug=debug)
