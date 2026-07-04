"""@file ieps.py
@brief Full Initial Edge Point Selection (IEPS) implementation.

This module implements the paper direction:
center of gravity -> equal-angle scan lines -> Sobel threshold -> farthest edge
candidate -> iterative midpoint/normal scan-line refinement. The code also
keeps the ambiguous implementation choices configurable so the project can
report reproducibility bugs and traditional-CV fixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    compute_center_of_gravity,
    euclidean_distance,
    midpoint,
    normal_vector,
    sample_line_through_point,
    sample_ray_from_center,
    sort_points_by_angle,
    unique_preserve_order,
)


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]


@dataclass
class IEPSResult:
    """@brief Result returned by IEPS.

    @param points Final selected IEPS points.
    @param center Center of gravity.
    @param debug Intermediate scan lines, iteration points, and choices.
    """

    points: List[Point]
    center: FloatPoint
    debug: Dict[str, object]


@dataclass
class IEPSConfig:
    """@brief Configuration for IEPS reproduction experiments.

    @param initial_scan_lines Number of initial scan lines N.
    @param iterations Refinement iterations p, giving N * 2^p possible points.
    @param threshold Sobel threshold.
    @param center_mode Center-of-gravity mode: raw, contrast, or binary.
    @param order_mode "topological" preserves iterative insertion order;
        "angle" sorts by polar angle. Topological order follows the paper's
        segment-insertion idea better, while angle sorting is useful for
        diagnostics on convex shapes.
    @param initial_selection Selection on initial rays: normally farthest_from_center.
    @param refinement_selection Selection on refinement lines.
    @param fallback_mode "drop" follows the paper text strictly; "max_gradient"
        keeps point count stable when thresholded points are missing.
    @param min_duplicate_distance Distance used to remove near-duplicate points.
    """

    initial_scan_lines: int = 4
    iterations: int = 3
    threshold: float = 64.0
    center_mode: str = "contrast"
    order_mode: str = "topological"
    initial_selection: str = "farthest_from_center"
    refinement_selection: str = "farthest_from_center"
    fallback_mode: str = "max_gradient"
    min_duplicate_distance: float = 1.5


def _gradient_at(gradient: np.ndarray, point: Point) -> float:
    """@brief Read gradient magnitude at a point.

    @param gradient Sobel magnitude image.
    @param point Point as (x, y).
    @return Gradient value.
    """
    x, y = point
    return float(gradient[y, x])


def _select_edge_candidate_on_line(
    points: List[Point],
    gradient: np.ndarray,
    reference: Sequence[float],
    threshold: float,
    selection: str = "farthest_from_center",
    fallback_mode: str = "max_gradient",
    exclude: Sequence[Point] | None = None,
    exclude_radius: float = 0.0,
) -> Optional[Point]:
    """@brief Select an IEPS edge candidate from a scan line.

    @param points Sampled scan-line points.
    @param gradient Sobel magnitude image.
    @param reference Reference point for distance-based selection.
    @param threshold Minimum gradient threshold.
    @param selection Candidate rule: farthest_from_center, max_gradient,
        closest_to_reference, or farthest_from_reference.
    @param fallback_mode drop or max_gradient when no point passes threshold.
    @param exclude Optional points to exclude, useful to avoid reselecting endpoints.
    @param exclude_radius Exclusion radius around excluded points.
    @return Selected point or None.
    """
    if not points:
        return None

    candidates = [p for p in points if _gradient_at(gradient, p) >= threshold]
    if exclude:
        candidates = [
            p for p in candidates
            if all(euclidean_distance(p, q) > exclude_radius for q in exclude)
        ]

    if not candidates:
        if fallback_mode == "drop":
            return None
        candidates = list(points)
        if exclude:
            filtered = [
                p for p in candidates
                if all(euclidean_distance(p, q) > exclude_radius for q in exclude)
            ]
            candidates = filtered if filtered else candidates

    if selection == "max_gradient":
        return max(candidates, key=lambda p: (_gradient_at(gradient, p), euclidean_distance(p, reference)))
    if selection == "closest_to_reference":
        return min(candidates, key=lambda p: (euclidean_distance(p, reference), -_gradient_at(gradient, p)))
    if selection in {"farthest_from_center", "farthest_from_reference"}:
        return max(candidates, key=lambda p: (euclidean_distance(p, reference), _gradient_at(gradient, p)))
    raise ValueError(f"Unknown IEPS selection rule: {selection}")


def _finalize_order(points: List[Point], center: FloatPoint, mode: str, min_distance: float) -> List[Point]:
    """@brief Deduplicate and order IEPS points.

    @param points Input point list.
    @param center Center of gravity.
    @param mode topological or angle.
    @param min_distance Duplicate distance.
    @return Finalized point list.
    """
    ordered = unique_preserve_order(points, min_distance=min_distance)
    if mode == "angle":
        return sort_points_by_angle(ordered, center)
    if mode == "topological":
        return ordered
    raise ValueError(f"Unknown IEPS order mode: {mode}")


def run_ieps(
    image: np.ndarray,
    gradient: np.ndarray,
    initial_scan_lines: int = 4,
    iterations: int = 3,
    threshold: float = 64.0,
    center: Optional[FloatPoint] = None,
    default_scan_distance: Optional[float] = None,
    center_mode: str = "contrast",
    order_mode: str = "topological",
    initial_selection: str = "farthest_from_center",
    refinement_selection: str = "farthest_from_center",
    fallback_mode: str = "max_gradient",
    exclude_existing_endpoints: bool = False,
) -> IEPSResult:
    """@brief Run IEPS using explicit paper/reproducibility parameters.

    @param image Input grayscale image.
    @param gradient Sobel gradient magnitude normalized to 0..255.
    @param initial_scan_lines Initial scan-line count N.
    @param iterations Iteration count p; expected maximum points = N * 2^p.
    @param threshold Gradient threshold.
    @param center Optional center override.
    @param default_scan_distance Default refinement distance D.
    @param center_mode Moment mode when center is not supplied.
    @param order_mode topological or angle.
    @param initial_selection Selection rule for initial rays.
    @param refinement_selection Selection rule for refinement scan lines.
    @param fallback_mode drop or max_gradient.
    @param exclude_existing_endpoints Avoid selecting p_i or p_i+1 as a new point.
    @return IEPSResult with final points and debug data.
    """
    if center is None:
        center = compute_center_of_gravity(image, mode=center_mode)
    height, width = image.shape[:2]
    if default_scan_distance is None:
        # Paper Eq. (7) leaves D undefined; D = min(h, w) / 4 gives the same
        # radius sequence as the earlier min(h, w) / 2 with exponent p.
        default_scan_distance = min(height, width) / 4.0

    debug: Dict[str, object] = {
        "center_mode": center_mode,
        "order_mode": order_mode,
        "initial_selection": initial_selection,
        "refinement_selection": refinement_selection,
        "fallback_mode": fallback_mode,
        "initial_scan_lines": [],
        "refinement_lines": [],
        "points_by_iteration": [],
    }

    points: List[Point] = []
    for i in range(initial_scan_lines):
        angle = (2.0 * math.pi * i) / float(initial_scan_lines)
        ray = sample_ray_from_center(center, angle, image.shape)
        selected = _select_edge_candidate_on_line(
            ray,
            gradient,
            center,
            threshold,
            selection=initial_selection,
            fallback_mode=fallback_mode,
        )
        debug["initial_scan_lines"].append(ray)
        if selected is not None:
            points.append(selected)

    points = _finalize_order(points, center, order_mode, min_distance=1.5)
    debug["points_by_iteration"].append(points.copy())

    # Paper refinement: each neighboring pair creates a new normal scan line.
    # The new point is inserted between the pair, preserving segment topology.
    for iteration in range(1, iterations + 1):
        if len(points) < 2:
            break
        radius = max(3.0, default_scan_distance / (2.0 ** (iteration - 1)))  # Eq. (7): d = D / 2^(p-1)
        next_points: List[Point] = []
        iteration_lines: List[List[Point]] = []

        for idx in range(len(points)):
            p1 = points[idx]
            p2 = points[(idx + 1) % len(points)]
            mid = midpoint(p1, p2)
            normal = normal_vector(p1, p2)
            scan = sample_line_through_point(mid, normal, image.shape, radius=radius)
            reference = center if "center" in refinement_selection else mid
            exclude = [p1, p2] if exclude_existing_endpoints else None
            selected = _select_edge_candidate_on_line(
                scan,
                gradient,
                reference,
                threshold,
                selection=refinement_selection,
                fallback_mode=fallback_mode,
                exclude=exclude,
                exclude_radius=2.0,
            )
            iteration_lines.append(scan)
            next_points.append(p1)
            if selected is not None:
                next_points.append(selected)

        points = _finalize_order(next_points, center, order_mode, min_distance=1.5)
        debug["refinement_lines"].append(iteration_lines)
        debug["points_by_iteration"].append(points.copy())

    return IEPSResult(points=points, center=center, debug=debug)


def run_ieps_with_config(image: np.ndarray, gradient: np.ndarray, config: IEPSConfig) -> IEPSResult:
    """@brief Convenience wrapper for IEPSConfig.

    @param image Input grayscale image.
    @param gradient Sobel magnitude image.
    @param config IEPS configuration.
    @return IEPSResult.
    """
    return run_ieps(
        image,
        gradient,
        initial_scan_lines=config.initial_scan_lines,
        iterations=config.iterations,
        threshold=config.threshold,
        center_mode=config.center_mode,
        order_mode=config.order_mode,
        initial_selection=config.initial_selection,
        refinement_selection=config.refinement_selection,
        fallback_mode=config.fallback_mode,
    )
