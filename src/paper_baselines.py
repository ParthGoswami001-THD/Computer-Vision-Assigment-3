"""@file paper_baselines.py
@brief Paper-comparison baselines used for fuller experimental reproduction.

The assigned paper compares IEPS/SCF with Yuen's initialization, Snake/Kass
active contours, and Chen's contour tracing. The full implementation details of
those comparison methods are not all present in the paper excerpt, so this file
implements compact, clearly labeled traditional-CV approximations for the
paper-style comparison tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import compute_center_of_gravity, euclidean_distance, sample_ray_from_center
from .ieps import _select_edge_candidate_on_line
from .scf import run_scf


Point = Tuple[int, int]
FloatPoint = Tuple[float, float]


@dataclass
class PointSelectionResult:
    """@brief Result for an initial-point selection baseline.

    @param points Selected points.
    @param center Center used for ray casting.
    @param debug Method-specific debug information.
    """

    points: List[Point]
    center: FloatPoint
    debug: Dict[str, object]


@dataclass
class SnakeResult:
    """@brief Result for the simple active-contour approximation.

    @param points Final snake control points.
    @param debug Iteration diagnostics.
    """

    points: List[Point]
    debug: Dict[str, object]


def run_yuen_style_initialization(
    image: np.ndarray,
    gradient: np.ndarray,
    num_points: int = 32,
    threshold: float = 64.0,
    center_mode: str = "contrast",
    fallback_mode: str = "max_gradient",
) -> PointSelectionResult:
    """@brief Approximate Yuen-style fixed-angle initial point selection.

    The paper describes Yuen's method as searching farthest edge pixels along
    fixed-angle scan lines. This implementation uses 32 fixed radial rays and
    chooses the farthest thresholded edge candidate on each ray.

    @param image Input grayscale image.
    @param gradient Sobel gradient image.
    @param num_points Number of fixed-angle scan lines.
    @param threshold Edge threshold.
    @param center_mode Center-of-gravity mode.
    @param fallback_mode Fallback when a ray has no thresholded edge.
    @return PointSelectionResult.
    """
    center = compute_center_of_gravity(image, mode=center_mode)
    points: List[Point] = []
    rays: List[List[Point]] = []
    for idx in range(num_points):
        angle = 2.0 * math.pi * idx / float(num_points)
        ray = sample_ray_from_center(center, angle, image.shape)
        selected = _select_edge_candidate_on_line(
            ray,
            gradient,
            center,
            threshold,
            selection="farthest_from_center",
            fallback_mode=fallback_mode,
        )
        rays.append(ray)
        if selected is not None:
            points.append(selected)

    return PointSelectionResult(
        points=points,
        center=center,
        debug={
            "method": "yuen_style_fixed_angle",
            "num_points": num_points,
            "threshold": threshold,
            "fallback_mode": fallback_mode,
            "rays": rays,
        },
    )


def _gradient_at(gradient: np.ndarray, point: Point) -> float:
    """@brief Read a gradient value.

    @param gradient Sobel magnitude image.
    @param point Point as (x, y).
    @return Gradient value.
    """
    x, y = point
    return float(gradient[y, x])


def _candidate_window(point: Point, image_shape: Tuple[int, int], radius: int) -> List[Point]:
    """@brief Return candidate pixels around a point.

    @param point Center point.
    @param image_shape Image shape.
    @param radius Search radius.
    @return Candidate points.
    """
    x, y = point
    height, width = image_shape[:2]
    candidates: List[Point] = []
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            candidates.append((xx, yy))
    return candidates


def run_simple_snake(
    gradient: np.ndarray,
    initial_points: Sequence[Point],
    iterations: int = 80,
    search_radius: int = 3,
    alpha: float = 0.15,
    beta: float = 0.10,
    gamma: float = 2.0,
) -> SnakeResult:
    """@brief Run a compact greedy Snake-style active contour.

    This is a lightweight approximation of the Kass Snake idea for assignment
    comparison: points move locally to balance continuity, curvature, and edge
    attraction. It is not a full variational Snake solver.

    @param gradient Sobel gradient magnitude image.
    @param initial_points Starting control points.
    @param iterations Number of optimization sweeps.
    @param search_radius Local candidate radius.
    @param alpha Continuity weight.
    @param beta Curvature weight.
    @param gamma Edge-attraction weight.
    @return SnakeResult.
    """
    points = [(int(x), int(y)) for x, y in initial_points]
    if len(points) < 3:
        return SnakeResult(points=points, debug={"method": "simple_snake", "iterations": 0})

    max_gradient = max(1.0, float(gradient.max()))
    avg_spacing = float(np.mean([
        euclidean_distance(points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
    ]))

    for _iteration in range(iterations):
        next_points = points.copy()
        for idx, point in enumerate(points):
            prev_point = next_points[idx - 1]
            next_point = points[(idx + 1) % len(points)]
            best_point = point
            best_energy = float("inf")
            scale = max(1.0, avg_spacing * avg_spacing)
            for candidate in _candidate_window(point, gradient.shape, search_radius):
                continuity = ((euclidean_distance(candidate, prev_point) - avg_spacing) ** 2) / scale
                mid_x = 0.5 * (prev_point[0] + next_point[0])
                mid_y = 0.5 * (prev_point[1] + next_point[1])
                curvature = ((candidate[0] - mid_x) ** 2 + (candidate[1] - mid_y) ** 2) / scale
                edge = -(_gradient_at(gradient, candidate) / max_gradient)
                energy = alpha * continuity + beta * curvature + gamma * edge
                if energy < best_energy:
                    best_energy = energy
                    best_point = candidate
            next_points[idx] = best_point
        points = next_points

    return SnakeResult(
        points=points,
        debug={
            "method": "simple_snake",
            "iterations": iterations,
            "search_radius": search_radius,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        },
    )


def run_chen_style_tracing(
    gradient: np.ndarray,
    ieps_points: Sequence[Point],
    center: Sequence[float] | None = None,
    stop_tolerance: float = 2.0,
):
    """@brief Approximate Chen-style gradient-only contour tracing.

    The assigned paper describes Chen's comparison method as relying on gradient
    magnitude without considering the next initial contour position in the same
    way as SCF. In this compact comparison, the same SCF segment machinery is
    used with a gradient-only score to isolate the effect of removing the
    gravity/distance term.

    @param gradient Sobel gradient magnitude image.
    @param ieps_points Segment endpoints.
    @param center Optional center for compatibility.
    @param stop_tolerance Segment stop tolerance.
    @return SCFResult from gradient-only tracing.
    """
    return run_scf(
        gradient,
        list(ieps_points),
        center=center,
        stop_tolerance=stop_tolerance,
        score_mode="gradient_only",
        method="greedy",
        sort_before_following=False,
    )
