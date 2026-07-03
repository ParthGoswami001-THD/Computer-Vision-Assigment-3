"""@file main.py
@brief Run full IEPS + SCF reproduction, bug checks, and improvement experiments.

Execute from project root:
    python main.py

The script generates author-style synthetic images, runs the paper-aligned IEPS
and SCF pipeline, runs a compact parameter/bug study, and saves images/CSV
reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.baseline import canny_contour_baseline
from src.evaluation import (
    contour_metrics,
    measure_runtime,
    neighbor_distance_stats,
    point_accuracy,
    points_to_mask,
)
from src.gradients import sobel_gradient_magnitude
from src.ieps import run_ieps
from src.image_generation import (
    add_gaussian_noise,
    create_circle_image,
    create_u_shape_image,
    create_vase_like_image,
)
from src.scf import run_scf
from src.visualization import draw_center, draw_contour_points, draw_points, make_panel, overlay_mask, save_image


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"


def _prepare_cases() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """@brief Prepare paper-style validation cases.

    @return Mapping from case name to (image, mask, contour).
    """
    circle_img, circle_mask, circle_contour = create_circle_image()
    u_img, u_mask, u_contour = create_u_shape_image()
    vase_img, vase_mask, vase_contour = create_vase_like_image()
    return {
        "circle_clean": (circle_img, circle_mask, circle_contour),
        "circle_noisy": (add_gaussian_noise(circle_img, sigma=20, seed=7), circle_mask, circle_contour),
        "u_shape_clean": (u_img, u_mask, u_contour),
        "u_shape_noisy": (add_gaussian_noise(u_img, sigma=20, seed=9), u_mask, u_contour),
        "vase_like_clean": (vase_img, vase_mask, vase_contour),
    }


def run_single_case(
    name: str,
    image: np.ndarray,
    gt_contour: np.ndarray,
    initial_scan_lines: int = 4,
    iterations: int = 3,
    threshold: float = 64.0,
    scf_tolerance: float = 2.0,
    center_mode: str = "contrast",
    order_mode: str = "topological",
    refinement_selection: str = "farthest_from_center",
    fallback_mode: str = "max_gradient",
    scf_method: str = "greedy",
    scf_score_mode: str = "gradient_distance2",
    save_outputs: bool = True,
) -> Dict[str, float | str | int]:
    """@brief Run one complete IEPS + SCF experiment.

    @param name Case name.
    @param image Input grayscale image.
    @param gt_contour Ground-truth contour mask.
    @param initial_scan_lines Number of initial IEPS scan lines.
    @param iterations Number of IEPS refinement iterations.
    @param threshold Sobel threshold.
    @param scf_tolerance SCF stop tolerance.
    @param center_mode raw, contrast, or binary center-of-gravity mode.
    @param order_mode IEPS order mode: topological or angle.
    @param refinement_selection IEPS refinement candidate rule.
    @param fallback_mode IEPS fallback: drop or max_gradient.
    @param scf_method greedy paper-style or graph improvement.
    @param scf_score_mode gradient_distance2 or gradient_only.
    @param save_outputs Save result images if True.
    @return Metrics dictionary.
    """
    case_dir = RESULTS_DIR / name
    gradient = sobel_gradient_magnitude(image)

    ieps_result, ieps_ms = measure_runtime(
        run_ieps,
        image,
        gradient,
        initial_scan_lines=initial_scan_lines,
        iterations=iterations,
        threshold=threshold,
        center_mode=center_mode,
        order_mode=order_mode,
        refinement_selection=refinement_selection,
        fallback_mode=fallback_mode,
    )

    scf_result, scf_ms = measure_runtime(
        run_scf,
        gradient,
        ieps_result.points,
        center=ieps_result.center,
        stop_tolerance=scf_tolerance,
        score_mode=scf_score_mode,
        method=scf_method,
        sort_before_following=False,
    )

    pred_mask = points_to_mask(scf_result.contour_points, image.shape, thickness=1)
    metrics = contour_metrics(pred_mask, gt_contour, tolerance=2)
    ieps_acc = point_accuracy(ieps_result.points, gt_contour, tolerance=2)
    neighbor_stats = neighbor_distance_stats(ieps_result.points)

    canny_mask, _ = canny_contour_baseline(image)
    canny_metrics = contour_metrics(canny_mask, gt_contour, tolerance=2)

    if save_outputs:
        save_image(case_dir / "original.png", image)
        save_image(case_dir / "sobel_gradient.png", gradient.astype(np.uint8))
        save_image(case_dir / "ground_truth_contour.png", gt_contour)
        save_image(case_dir / "center.png", draw_center(image, ieps_result.center))
        save_image(case_dir / "ieps_points.png", draw_points(image, ieps_result.points))
        save_image(case_dir / f"scf_{scf_method}.png", draw_contour_points(image, scf_result.contour_points))
        save_image(case_dir / "canny_baseline.png", overlay_mask(image, canny_mask, color=(255, 0, 0)))
        panel = make_panel(
            [
                image,
                gradient.astype(np.uint8),
                draw_center(image, ieps_result.center),
                draw_points(image, ieps_result.points),
                draw_contour_points(image, scf_result.contour_points),
                overlay_mask(image, gt_contour, color=(0, 255, 0)),
            ],
            ["Input", "Sobel", "CoG", "IEPS", f"SCF-{scf_method}", "Ground Truth"],
        )
        save_image(case_dir / f"panel_{scf_method}.png", panel)

    stopping_reasons = [str(s.get("stopping_reason", "")) for s in scf_result.debug.get("segments", [])]
    target_reached_count = sum(r == "target_reached" for r in stopping_reasons)

    return {
        "case": name,
        "scan_lines": initial_scan_lines,
        "iterations": iterations,
        "threshold": threshold,
        "center_mode": center_mode,
        "order_mode": order_mode,
        "refinement_selection": refinement_selection,
        "fallback_mode": fallback_mode,
        "scf_method": scf_method,
        "scf_score_mode": scf_score_mode,
        "scf_tolerance": scf_tolerance,
        "ieps_points": len(ieps_result.points),
        "ieps_accuracy": ieps_acc,
        "neighbor_mean": neighbor_stats["neighbor_mean"],
        "neighbor_std": neighbor_stats["neighbor_std"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "target_reached_segments": target_reached_count,
        "total_segments": len(stopping_reasons),
        "ieps_ms": ieps_ms,
        "scf_ms": scf_ms,
        "total_ms": ieps_ms + scf_ms,
        "canny_precision": canny_metrics["precision"],
        "canny_recall": canny_metrics["recall"],
        "canny_f1": canny_metrics["f1"],
    }


def run_main_experiments() -> pd.DataFrame:
    """@brief Run default paper-style and improved SCF experiments.

    @return Main results DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    for case, (image, _mask, contour) in _prepare_cases().items():
        if case == "vase_like_clean":
            # Optional real-like shape; keep qualitative and compact.
            continue
        rows.append(run_single_case(case, image, contour, scf_method="greedy", save_outputs=True))
        rows.append(run_single_case(case, image, contour, scf_method="graph", save_outputs=True))
    return pd.DataFrame(rows)


def run_bug_fix_study() -> pd.DataFrame:
    """@brief Run compact study for discovered bugs and fixes.

    @return Bug/fix study DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    cases = _prepare_cases()
    image, _mask, contour = cases["u_shape_clean"]

    # Center-of-gravity bug/fix: raw moments can be biased by non-zero background.
    nonzero_bg_img, _, nonzero_bg_contour = create_u_shape_image(background=40, foreground=200)
    for mode in ["raw", "contrast", "binary"]:
        rows.append(run_single_case(
            f"bug_center_{mode}", nonzero_bg_img, nonzero_bg_contour,
            center_mode=mode, scf_method="greedy", save_outputs=False,
        ))

    # IEPS ordering and candidate rule study for concave objects.
    for order in ["topological", "angle"]:
        for selection in ["farthest_from_center", "max_gradient"]:
            rows.append(run_single_case(
                f"bug_order_{order}_{selection}", image, contour,
                order_mode=order, refinement_selection=selection,
                scf_method="greedy", save_outputs=False,
            ))

    # Paper score vs diagnostic score and graph-search traditional CV fix.
    for score in ["gradient_distance2", "gradient_only"]:
        rows.append(run_single_case(
            f"bug_scf_score_{score}", image, contour,
            scf_score_mode=score, scf_method="greedy", save_outputs=False,
        ))
    rows.append(run_single_case(
        "fix_graph_search_scf", image, contour,
        scf_method="graph", save_outputs=False,
    ))

    return pd.DataFrame(rows)


def run_parameter_study() -> pd.DataFrame:
    """@brief Run author-aligned parameter study.

    @return Parameter study DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    image, _mask, contour = _prepare_cases()["u_shape_noisy"]

    for threshold in [40.0, 64.0, 90.0]:
        rows.append(run_single_case(f"param_threshold_{int(threshold)}", image, contour, threshold=threshold, save_outputs=False))
    for scan_lines in [4, 8]:
        rows.append(run_single_case(f"param_scanlines_{scan_lines}", image, contour, initial_scan_lines=scan_lines, save_outputs=False))
    for iterations in [2, 3]:
        rows.append(run_single_case(f"param_iterations_{iterations}", image, contour, iterations=iterations, save_outputs=False))
    for tolerance in [1.0, 2.0, 3.0]:
        rows.append(run_single_case(f"param_scf_tol_{int(tolerance)}", image, contour, scf_tolerance=tolerance, save_outputs=False))
    return pd.DataFrame(rows)


def main() -> None:
    """@brief Entry point."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    main_results = run_main_experiments()
    bug_fix_results = run_bug_fix_study()
    parameter_results = run_parameter_study()

    main_results.to_csv(RESULTS_DIR / "tables" / "main_results.csv", index=False)
    bug_fix_results.to_csv(RESULTS_DIR / "tables" / "bug_fix_study.csv", index=False)
    parameter_results.to_csv(RESULTS_DIR / "tables" / "parameter_study.csv", index=False)

    print("\nMain results:")
    print(main_results[["case", "scf_method", "ieps_points", "ieps_accuracy", "precision", "recall", "f1", "total_ms"]])
    print("\nBug/fix study:")
    print(bug_fix_results[["case", "center_mode", "order_mode", "refinement_selection", "scf_method", "ieps_accuracy", "f1"]])
    print("\nSaved results under:", RESULTS_DIR)


if __name__ == "__main__":
    main()
