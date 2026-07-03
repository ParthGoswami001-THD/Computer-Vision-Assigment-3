"""@file evaluation.py
@brief Evaluation metrics for IEPS + SCF validation.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Tuple, TypeVar

import cv2
import numpy as np

from .geometry import euclidean_distance


Point = Tuple[int, int]
T = TypeVar("T")


def points_to_mask(points: Iterable[Point], shape: Tuple[int, int], thickness: int = 1) -> np.ndarray:
    """@brief Convert contour points to a binary mask.

    @param points Iterable of points as (x, y).
    @param shape Output mask shape as (height, width).
    @param thickness Drawing thickness.
    @return Binary mask with values 0 and 255.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = list(points)
    for p in pts:
        x, y = p
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            cv2.circle(mask, (int(x), int(y)), max(1, thickness), 255, -1)
    return mask


def _distance_to_contour_map(contour_mask: np.ndarray) -> np.ndarray:
    """@brief Compute distance to nearest true contour pixel.

    @param contour_mask Binary contour mask.
    @return Distance transform map.
    """
    binary = (contour_mask > 0).astype(np.uint8)
    inverse = 1 - binary
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3)


def point_accuracy(points: List[Point], ground_truth_contour: np.ndarray, tolerance: float = 2.0) -> float:
    """@brief Compute IEPS point accuracy with pixel tolerance.

    @param points Selected IEPS points.
    @param ground_truth_contour Binary ground-truth contour mask.
    @param tolerance Maximum distance to ground truth for a point to be correct.
    @return Accuracy in [0, 1].
    """
    if not points:
        return 0.0
    dist_map = _distance_to_contour_map(ground_truth_contour)
    correct = 0
    for x, y in points:
        if 0 <= y < dist_map.shape[0] and 0 <= x < dist_map.shape[1]:
            if dist_map[y, x] <= tolerance:
                correct += 1
    return correct / float(len(points))


def contour_metrics(
    predicted_contour: np.ndarray,
    ground_truth_contour: np.ndarray,
    tolerance: int = 2,
) -> Dict[str, float]:
    """@brief Compute tolerance-based contour precision, recall, and F1.

    @param predicted_contour Binary predicted contour mask.
    @param ground_truth_contour Binary ground-truth contour mask.
    @param tolerance Pixel tolerance for matching contour pixels.
    @return Dictionary with precision, recall, and f1.
    """
    pred = (predicted_contour > 0).astype(np.uint8)
    gt = (ground_truth_contour > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1))
    gt_dilated = cv2.dilate(gt, kernel)
    pred_dilated = cv2.dilate(pred, kernel)

    pred_count = int(pred.sum())
    gt_count = int(gt.sum())

    tp_precision = int((pred & gt_dilated).sum())
    tp_recall = int((gt & pred_dilated).sum())

    precision = tp_precision / pred_count if pred_count > 0 else 0.0
    recall = tp_recall / gt_count if gt_count > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def neighbor_distance_stats(points: List[Point]) -> Dict[str, float]:
    """@brief Compute neighbor distance mean and standard deviation.

    @param points Ordered contour points.
    @return Dictionary with mean and standard deviation.
    """
    if len(points) < 2:
        return {"neighbor_mean": 0.0, "neighbor_std": 0.0}
    distances = [euclidean_distance(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    arr = np.asarray(distances, dtype=np.float64)
    return {"neighbor_mean": float(arr.mean()), "neighbor_std": float(arr.std())}


def measure_runtime(func: Callable[..., T], *args, **kwargs) -> Tuple[T, float]:
    """@brief Measure runtime of a function.

    @param func Callable to execute.
    @param args Positional arguments.
    @param kwargs Keyword arguments.
    @return Tuple (function_result, elapsed_ms).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms
