"""@file scf.py
@brief Segmental Contour Following (SCF) implementation and CV fixes.

The paper describes SCF as a local contour follower using related direction,
3-candidate operating masks, and a gravity-like force based on gradient and
candidate distance to the next initial edge point. This module implements that
paper-faithful greedy SCF and an optional graph-search SCF used only as a
traditional-CV improvement experiment.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import euclidean_distance, in_bounds, sort_points_by_angle


Point = Tuple[int, int]


@dataclass
class SCFResult:
    """@brief Result returned by SCF.

    @param contour_points Full contour point list.
    @param debug Per-segment diagnostics.
    """

    contour_points: List[Point]
    debug: Dict[str, object]


def _sign(value: float) -> int:
    """@brief Return -1, 0, or 1 for a scalar.

    @param value Input scalar.
    @return Sign integer.
    """
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def direction_state(current: Point, target: Point) -> str:
    """@brief Classify related direction into the paper's 8-neighborhood states.

    @param current Current origin point.
    @param target Target IEPS point.
    @return Direction state name using image coordinates.
    """
    sx = _sign(target[0] - current[0])
    sy = _sign(target[1] - current[1])
    if sx == 0 and sy == 0:
        return "middle"
    names = {
        (1, 0): "east",
        (1, 1): "south_east",
        (0, 1): "south",
        (-1, 1): "south_west",
        (-1, 0): "west",
        (-1, -1): "north_west",
        (0, -1): "north",
        (1, -1): "north_east",
    }
    return names[(sx, sy)]


def _directional_candidates(current: Point, target: Point, image_shape: Tuple[int, int]) -> List[Point]:
    """@brief Generate the three operating-mask candidates A/B/C.

    @param current Current contour point as (x, y).
    @param target Segment target point as (x, y).
    @param image_shape Image shape as (height, width).
    @return Candidate points inside image bounds.
    """
    cx, cy = current
    sx = _sign(target[0] - cx)
    sy = _sign(target[1] - cy)
    if sx == 0 and sy == 0:
        offsets: List[Tuple[int, int]] = []
    elif sx != 0 and sy != 0:
        offsets = [(sx, sy), (sx, 0), (0, sy)]
    elif sx != 0:
        offsets = [(sx, 0), (sx, -1), (sx, 1)]
    else:
        offsets = [(0, sy), (-1, sy), (1, sy)]
    candidates = [(cx + dx, cy + dy) for dx, dy in offsets]
    return [p for p in candidates if in_bounds(p, image_shape)]


def _eight_neighbors(point: Point, image_shape: Tuple[int, int]) -> List[Point]:
    """@brief Return all 8-connected neighbors of a point.

    @param point Current point.
    @param image_shape Image shape.
    @return Valid neighboring points.
    """
    x, y = point
    out: List[Point] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            p = (x + dx, y + dy)
            if in_bounds(p, image_shape):
                out.append(p)
    return out


def _point_segment_distance(point: Point, start: Point, target: Point) -> float:
    """@brief Distance from a point to a line segment.

    @param point Query point.
    @param start Segment start.
    @param target Segment end.
    @return Euclidean distance to the finite segment.
    """
    px, py = float(point[0]), float(point[1])
    sx, sy = float(start[0]), float(start[1])
    tx, ty = float(target[0]), float(target[1])
    vx, vy = tx - sx, ty - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return euclidean_distance(point, start)
    t = ((px - sx) * vx + (py - sy) * vy) / denom
    t = max(0.0, min(1.0, t))
    closest = (sx + t * vx, sy + t * vy)
    return euclidean_distance(point, closest)


def _turn_penalty(previous: Point, current: Point, candidate: Point) -> float:
    """@brief Penalize sharp changes in tracing direction.

    @param previous Previous contour point.
    @param current Current contour point.
    @param candidate Candidate next point.
    @return Penalty in [0, 2], where 0 is straight continuation.
    """
    v1x = float(current[0] - previous[0])
    v1y = float(current[1] - previous[1])
    v2x = float(candidate[0] - current[0])
    v2y = float(candidate[1] - current[1])
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cosine = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def _gradient_at(gradient: np.ndarray, point: Point) -> float:
    """@brief Read gradient value at a point.

    @param gradient Sobel magnitude image.
    @param point Point as (x, y).
    @return Gradient value.
    """
    x, y = point
    return float(gradient[y, x])


def _score_candidate(gradient: np.ndarray, candidate: Point, target: Point, mode: str) -> float:
    """@brief Score a candidate using paper-style or diagnostic scoring.

    @param gradient Sobel magnitude image.
    @param candidate Candidate point.
    @param target Segment target point.
    @param mode gradient_distance2 or gradient_only.
    @return Candidate score, larger is better.
    """
    g = _gradient_at(gradient, candidate)
    if mode == "gradient_only":
        return g
    if mode == "gradient_distance2":
        d = euclidean_distance(candidate, target)
        return g / ((d * d) + 1.0)
    raise ValueError(f"Unknown SCF score mode: {mode}")


def trace_segment_greedy(
    gradient: np.ndarray,
    start: Point,
    target: Point,
    stop_tolerance: float = 2.0,
    max_steps: int | None = None,
    score_mode: str = "gradient_distance2",
    weak_gradient_threshold: float = 1.0,
) -> Tuple[List[Point], Dict[str, object]]:
    """@brief Trace one SCF segment using the paper-style greedy mask.

    @param gradient Sobel gradient magnitude image.
    @param start Start IEPS point.
    @param target Target IEPS point.
    @param stop_tolerance Stop when current is within this target distance.
    @param max_steps Maximum segment steps.
    @param score_mode Candidate score mode.
    @param weak_gradient_threshold If all candidates are below this value, use
        closest-to-target fallback.
    @return Tuple (path, debug dictionary).
    """
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))
    if max_steps is None:
        max_steps = max(5, int(math.ceil(3.0 * euclidean_distance(start, target))))

    current = start
    path: List[Point] = [current]
    visited = {current}
    stopping_reason = "max_steps_reached"

    for _ in range(max_steps):
        if euclidean_distance(current, target) <= stop_tolerance:
            stopping_reason = "target_reached"
            break
        candidates = _directional_candidates(current, target, gradient.shape)
        if not candidates:
            stopping_reason = "no_candidates"
            break
        unvisited = [p for p in candidates if p not in visited]
        active = unvisited if unvisited else candidates
        max_g = max(_gradient_at(gradient, p) for p in active)
        if max_g < weak_gradient_threshold:
            selected = min(active, key=lambda p: euclidean_distance(p, target))
        else:
            selected = max(
                active,
                key=lambda p: (
                    _score_candidate(gradient, p, target, score_mode),
                    _gradient_at(gradient, p),
                    -euclidean_distance(p, target),
                ),
            )
        if selected == current:
            stopping_reason = "stalled"
            break
        current = selected
        path.append(current)
        visited.add(current)

    if path[-1] != target and euclidean_distance(path[-1], target) <= stop_tolerance:
        path.append(target)
    return path, {
        "start": start,
        "target": target,
        "method": "greedy_mask",
        "state": direction_state(start, target),
        "steps": len(path),
        "max_steps": max_steps,
        "stopping_reason": stopping_reason,
    }


def trace_segment_graph(
    gradient: np.ndarray,
    start: Point,
    target: Point,
    stop_tolerance: float = 2.0,
    max_expansions: int = 40000,
    edge_weight: float = 4.0,
    heuristic_weight: float = 0.15,
) -> Tuple[List[Point], Dict[str, object]]:
    """@brief Trace one segment using gradient-weighted A* graph search.

    This is not claimed as the original paper algorithm. It is included as a
    traditional-CV improvement/fix when greedy local masks break on concavity or
    noisy edges.

    @param gradient Sobel gradient magnitude image.
    @param start Start point.
    @param target Target point.
    @param stop_tolerance Stop radius around target.
    @param max_expansions Maximum A* node expansions.
    @param edge_weight Penalty for low-gradient pixels.
    @param heuristic_weight Strength of target-distance heuristic.
    @return Tuple (path, debug dictionary).
    """
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))
    pq: List[Tuple[float, Point]] = [(0.0, start)]
    g_score: Dict[Point, float] = {start: 0.0}
    parent: Dict[Point, Point] = {}
    visited = set()
    expansions = 0
    reached = start

    while pq and expansions < max_expansions:
        _, current = heapq.heappop(pq)
        if current in visited:
            continue
        visited.add(current)
        expansions += 1
        reached = current
        if euclidean_distance(current, target) <= stop_tolerance:
            break
        for nb in _eight_neighbors(current, gradient.shape):
            dx = nb[0] - current[0]
            dy = nb[1] - current[1]
            step = math.hypot(dx, dy)
            edge_strength = _gradient_at(gradient, nb) / 255.0
            cost = step * (1.0 + edge_weight * (1.0 - edge_strength))
            new_cost = g_score[current] + cost
            if new_cost < g_score.get(nb, float("inf")):
                g_score[nb] = new_cost
                parent[nb] = current
                priority = new_cost + heuristic_weight * euclidean_distance(nb, target)
                heapq.heappush(pq, (priority, nb))

    path = [reached]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    if path[-1] != target and euclidean_distance(path[-1], target) <= stop_tolerance:
        path.append(target)
    return path, {
        "start": start,
        "target": target,
        "method": "graph_search",
        "steps": len(path),
        "expansions": expansions,
        "stopping_reason": "target_reached" if euclidean_distance(path[-1], target) <= stop_tolerance else "max_expansions_reached",
    }


def trace_segment_band_graph(
    gradient: np.ndarray,
    start: Point,
    target: Point,
    stop_tolerance: float = 2.0,
    max_expansions: int = 30000,
    band_radius: float | None = None,
    edge_weight: float = 4.0,
    band_weight: float = 0.4,
    curvature_weight: float = 1.2,
    heuristic_weight: float = 0.15,
) -> Tuple[List[Point], Dict[str, object]]:
    """@brief Trace one segment using band-limited curvature-aware A*.

    This is a traditional-CV SCF improvement. It keeps the graph-search idea but
    constrains the search near the segment between IEPS points and discourages
    high-curvature zig-zags.

    @param gradient Sobel gradient magnitude image.
    @param start Start point.
    @param target Target point.
    @param stop_tolerance Stop radius around target.
    @param max_expansions Maximum A* state expansions.
    @param band_radius Preferred distance from the start-target segment.
    @param edge_weight Penalty for low-gradient pixels.
    @param band_weight Penalty for leaving the preferred band.
    @param curvature_weight Penalty for sharp turns.
    @param heuristic_weight Strength of target-distance heuristic.
    @return Tuple (path, debug dictionary).
    """
    start = (int(start[0]), int(start[1]))
    target = (int(target[0]), int(target[1]))
    endpoint_distance = euclidean_distance(start, target)
    if band_radius is None:
        band_radius = max(6.0, 0.45 * endpoint_distance)
    hard_band_radius = max(band_radius * 2.0, band_radius + stop_tolerance + 2.0)
    max_gradient = max(1.0, float(gradient.max()))

    start_state = (start, start)
    pq: List[Tuple[float, int, Point, Point]] = [(0.0, 0, start, start)]
    g_score: Dict[Tuple[Point, Point], float] = {start_state: 0.0}
    parent: Dict[Tuple[Point, Point], Tuple[Point, Point]] = {}
    visited = set()
    expansions = 0
    reached_state = start_state
    counter = 1

    while pq and expansions < max_expansions:
        _priority, _count, current, previous = heapq.heappop(pq)
        state = (current, previous)
        if state in visited:
            continue
        visited.add(state)
        expansions += 1
        reached_state = state
        if euclidean_distance(current, target) <= stop_tolerance:
            break

        for nb in _eight_neighbors(current, gradient.shape):
            distance_to_segment = _point_segment_distance(nb, start, target)
            if distance_to_segment > hard_band_radius and euclidean_distance(nb, target) > stop_tolerance:
                continue

            dx = nb[0] - current[0]
            dy = nb[1] - current[1]
            step = math.hypot(dx, dy)
            edge_strength = min(1.0, _gradient_at(gradient, nb) / max_gradient)
            edge_cost = step * (1.0 + edge_weight * (1.0 - edge_strength))
            band_overflow = max(0.0, distance_to_segment - band_radius)
            band_cost = band_weight * band_overflow * band_overflow
            curvature_cost = curvature_weight * _turn_penalty(previous, current, nb)
            new_cost = g_score[state] + edge_cost + band_cost + curvature_cost
            next_state = (nb, current)
            if new_cost < g_score.get(next_state, float("inf")):
                g_score[next_state] = new_cost
                parent[next_state] = state
                priority = new_cost + heuristic_weight * euclidean_distance(nb, target)
                heapq.heappush(pq, (priority, counter, nb, current))
                counter += 1

    path = [reached_state[0]]
    state = reached_state
    while state in parent:
        state = parent[state]
        path.append(state[0])
    path.reverse()
    if path[-1] != target and euclidean_distance(path[-1], target) <= stop_tolerance:
        path.append(target)
    return path, {
        "start": start,
        "target": target,
        "method": "band_limited_curvature_graph",
        "steps": len(path),
        "expansions": expansions,
        "band_radius": band_radius,
        "hard_band_radius": hard_band_radius,
        "edge_weight": edge_weight,
        "band_weight": band_weight,
        "curvature_weight": curvature_weight,
        "stopping_reason": "target_reached" if euclidean_distance(path[-1], target) <= stop_tolerance else "max_expansions_reached",
    }


def run_scf(
    gradient: np.ndarray,
    ieps_points: List[Point],
    center: Sequence[float] | None = None,
    stop_tolerance: float = 2.0,
    max_step_factor: float = 3.0,
    score_mode: str = "gradient_distance2",
    method: str = "greedy",
    sort_before_following: bool = False,
) -> SCFResult:
    """@brief Run SCF over all neighboring IEPS points.

    @param gradient Sobel magnitude image.
    @param ieps_points IEPS points to connect.
    @param center Optional center used only if sort_before_following=True.
    @param stop_tolerance Segment target tolerance.
    @param max_step_factor Greedy max steps as factor of endpoint distance.
    @param score_mode Greedy score: gradient_distance2 or gradient_only.
    @param method greedy for paper-style SCF, graph or band_graph for improved
        traditional CV.
    @param sort_before_following If True, sort points by polar angle before SCF.
    @return SCFResult.
    """
    if len(ieps_points) < 2:
        return SCFResult(contour_points=ieps_points.copy(), debug={"segments": []})
    points = sort_points_by_angle(ieps_points, center) if (center is not None and sort_before_following) else ieps_points.copy()

    contour_points: List[Point] = []
    segment_debug: List[Dict[str, object]] = []
    for i in range(len(points)):
        start = points[i]
        target = points[(i + 1) % len(points)]
        if method == "greedy":
            max_steps = max(5, int(math.ceil(max_step_factor * euclidean_distance(start, target))))
            path, dbg = trace_segment_greedy(
                gradient,
                start,
                target,
                stop_tolerance=stop_tolerance,
                max_steps=max_steps,
                score_mode=score_mode,
            )
        elif method == "graph":
            path, dbg = trace_segment_graph(gradient, start, target, stop_tolerance=stop_tolerance)
        elif method == "band_graph":
            path, dbg = trace_segment_band_graph(gradient, start, target, stop_tolerance=stop_tolerance)
        else:
            raise ValueError(f"Unknown SCF method: {method}")
        contour_points.extend(path[:-1] if i < len(points) - 1 else path)
        segment_debug.append(dbg)

    return SCFResult(contour_points=contour_points, debug={"method": method, "segments": segment_debug})
