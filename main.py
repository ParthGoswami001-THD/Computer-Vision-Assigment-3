"""@file main.py
@brief Run selected IEPS + SCF reproduction experiments.

Execute from project root:
    python main.py
    python main.py --run all

By default the script runs only the main author-style synthetic experiment with
the paper-style greedy SCF. Use --run to generate parameter studies, paper
comparisons, real-vase tests, or the complete result set, and --scf to opt into
the graph/band_graph extension methods.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from src.ieps_improved import run_ieps_improved
from src.image_generation import (
    add_gaussian_noise,
    add_gaussian_noise_snr,
    create_circle_image,
    create_u_shape_image,
    create_vase_like_image,
    load_real_vase_case,
)
from src.scf import run_scf
from src.paper_baselines import (
    run_chen_style_tracing,
    run_simple_snake,
    run_yuen_style_initialization,
)
from src.visualization import (
    draw_center,
    draw_contour_points,
    draw_points,
    draw_scan_lines,
    make_panel,
    overlay_mask,
    save_image,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"

MAIN_CASES = ["circle_clean", "circle_noisy", "u_shape_clean", "u_shape_noisy"]
SCF_METHODS = ["greedy", "graph", "band_graph"]
RUN_ORDER = [
    "main",
    "parameter",
    "improvement",
    "vase",
    "paper-comparison",
]
RUN_DESCRIPTIONS = {
    "main": "Core synthetic circle/U-shape IEPS + SCF results (paper-style greedy SCF by default).",
    "parameter": "Reproducibility parameter study inside IEPS + SCF.",
    "improvement": "Paper IEPS versus improved IEPS extension.",
    "vase": "Real-vase path, or clearly labeled fallback if no vase image exists.",
    "paper-comparison": "Optional paper-context approximation tables for Yuen/Snake/Chen-style comparisons.",
    "all": "Run and write every result table and figure set.",
}


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
    ieps_mode: str = "paper",
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
    @param ieps_mode paper or improved IEPS initialization.
    @param scf_method greedy paper-style, graph, or band_graph improvement.
    @param scf_score_mode gradient_distance2 or gradient_only.
    @param save_outputs Save result images if True.
    @return Metrics dictionary.
    """
    case_dir = RESULTS_DIR / name
    gradient = sobel_gradient_magnitude(image)

    if ieps_mode == "paper":
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
    elif ieps_mode == "improved":
        ieps_result, ieps_ms = measure_runtime(
            run_ieps_improved,
            image,
            gradient,
            initial_scan_lines=initial_scan_lines,
            iterations=iterations,
            threshold=threshold,
            center_mode=center_mode,
            refinement_selection="closest_to_reference",
            fallback_mode=fallback_mode,
        )
    else:
        raise ValueError(f"Unknown IEPS mode: {ieps_mode}")

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

    (canny_mask, _canny_points), canny_ms = measure_runtime(canny_contour_baseline, image)
    canny_metrics = contour_metrics(canny_mask, gt_contour, tolerance=2)

    if save_outputs:
        # Plan-style output names; the paper-style greedy run owns the plain
        # names, extension SCF methods get a method suffix.
        scf_image_name = "scf_contour.png" if scf_method == "greedy" else f"scf_contour_{scf_method}.png"
        panel_name = "panel.png" if scf_method == "greedy" else f"panel_{scf_method}.png"
        initial_lines = ieps_result.debug.get("initial_scan_lines", [])
        points_by_iteration = ieps_result.debug.get("points_by_iteration") or [[]]
        save_image(case_dir / "original.png", image)
        save_image(case_dir / "gradient.png", gradient.astype(np.uint8))
        save_image(case_dir / "ground_truth.png", gt_contour)
        save_image(case_dir / "center.png", draw_center(image, ieps_result.center))
        save_image(
            case_dir / "initial_scan_lines.png",
            draw_scan_lines(image, initial_lines, points=points_by_iteration[0], center=ieps_result.center),
        )
        save_image(case_dir / "ieps_points.png", draw_points(image, ieps_result.points))
        save_image(case_dir / scf_image_name, draw_contour_points(image, scf_result.contour_points))
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
        save_image(case_dir / panel_name, panel)

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
        "ieps_mode": ieps_mode,
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
        "canny_ms": canny_ms,
    }


def run_improvement_comparison() -> pd.DataFrame:
    """@brief Compare paper IEPS with improved IEPS extension.

    @return Improvement comparison DataFrame.
    """
    rows: List[Dict[str, float | str | int | bool]] = []
    for case, (image, _mask, contour) in _prepare_cases().items():
        if case == "vase_like_clean":
            continue
        paper = run_single_case(
            case,
            image,
            contour,
            ieps_mode="paper",
            scf_method="greedy",
            save_outputs=False,
        )
        improved = run_single_case(
            case,
            image,
            contour,
            initial_scan_lines=8,
            ieps_mode="improved",
            scf_method="greedy",
            save_outputs=False,
        )
        rows.append({
            "case": case,
            "paper_ieps_points": paper["ieps_points"],
            "paper_ieps_accuracy": paper["ieps_accuracy"],
            "paper_f1": paper["f1"],
            "improved_ieps_points": improved["ieps_points"],
            "improved_ieps_accuracy": improved["ieps_accuracy"],
            "improved_f1": improved["f1"],
            "f1_delta": improved["f1"] - paper["f1"],
            "paper_total_ms": paper["total_ms"],
            "improved_total_ms": improved["total_ms"],
        })
    return pd.DataFrame(rows)


def run_main_experiments(
    selected_cases: Sequence[str] | None = None,
    selected_scf_methods: Sequence[str] | None = None,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """@brief Run default paper-style and improved SCF experiments.

    @param selected_cases Optional case names to run.
    @param selected_scf_methods Optional SCF methods to run.
    @param save_outputs Save result images if True.
    @return Main results DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    case_filter = set(selected_cases or MAIN_CASES)
    scf_methods = list(selected_scf_methods or SCF_METHODS)
    for case, (image, _mask, contour) in _prepare_cases().items():
        if case not in case_filter:
            continue
        for scf_method in scf_methods:
            rows.append(run_single_case(case, image, contour, scf_method=scf_method, save_outputs=save_outputs))
    return pd.DataFrame(rows)


def run_parameter_study() -> pd.DataFrame:
    """@brief Run author-aligned parameter study.

    @return Parameter study DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    cases = _prepare_cases()
    clean_image, _clean_mask, clean_contour = cases["u_shape_clean"]
    image, _mask, contour = cases["u_shape_noisy"]

    for threshold in [40.0, 64.0, 90.0]:
        rows.append(run_single_case(f"param_threshold_{int(threshold)}", image, contour, threshold=threshold, save_outputs=False))
    for scan_lines in [4, 8]:
        rows.append(run_single_case(f"param_scanlines_{scan_lines}", image, contour, initial_scan_lines=scan_lines, save_outputs=False))
    for iterations in [2, 3]:
        rows.append(run_single_case(f"param_iterations_{iterations}", image, contour, iterations=iterations, save_outputs=False))
    for tolerance in [1.0, 2.0, 3.0]:
        rows.append(run_single_case(f"param_scf_tol_{int(tolerance)}", image, contour, scf_tolerance=tolerance, save_outputs=False))
    for score_mode in ["gradient_only", "gradient_distance2"]:
        rows.append(run_single_case(
            f"param_scf_score_{score_mode}",
            image,
            contour,
            scf_score_mode=score_mode,
            save_outputs=False,
        ))
    # The implementation plan describes closest-to-midpoint refinement and
    # angle-sorted point order; the paper-mode defaults differ, so both
    # under-specified choices are measured here instead of hidden.
    for refinement_selection in ["farthest_from_center", "closest_to_reference", "max_gradient"]:
        rows.append(run_single_case(
            f"param_refinement_{refinement_selection}",
            image,
            contour,
            refinement_selection=refinement_selection,
            save_outputs=False,
        ))
    for order_mode in ["topological", "angle"]:
        rows.append(run_single_case(
            f"param_order_{order_mode}",
            image,
            contour,
            order_mode=order_mode,
            save_outputs=False,
        ))
    for noise_name, noise_image in [
        ("clean", clean_image),
        ("low_noise_sigma10", add_gaussian_noise(clean_image, sigma=10, seed=11)),
        ("paperlike_noise_sigma20", add_gaussian_noise(clean_image, sigma=20, seed=9)),
    ]:
        rows.append(run_single_case(
            f"param_noise_{noise_name}",
            noise_image,
            clean_contour,
            save_outputs=False,
        ))
    return pd.DataFrame(rows)


def _find_real_vase_image_path() -> Path | None:
    """@brief Find a local vase image using common image extensions.

    @return Path to the first available vase image, or None.
    """
    candidates = [
        DATA_DIR / "vase.png",
        DATA_DIR / "vase.jpg",
        DATA_DIR / "vase.jpeg",
        DATA_DIR / "vase.webp",
        DATA_DIR / "vase.png.webp",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in sorted(DATA_DIR.glob("vase.*")):
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            return path
    return None


def _prepare_real_vase_case() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """@brief Prepare a real vase case, or a labeled fallback if no image exists.

    @return Tuple (image, mask, contour, input_source, mask_source).
    """
    image_path = _find_real_vase_image_path()
    mask_path = DATA_DIR / "vase_mask.png"
    if image_path is not None:
        image, mask, contour, mask_source = load_real_vase_case(image_path, mask_path=mask_path)
        return image, mask, contour, str(image_path.relative_to(BASE_DIR)), mask_source

    image, mask, contour = create_vase_like_image()
    return image, mask, contour, "synthetic_fallback_missing_real_vase_image", "generated_synthetic_mask"


def run_real_vase_test(save_outputs: bool = True) -> pd.DataFrame:
    """@brief Run the real-vase-style qualitative IEPS + SCF test.

    The paper evaluates a real vase image, but this repository does not include
    the original paper image. If a local data/vase image exists with a common
    extension, it is used. Otherwise a clearly labeled synthetic fallback keeps
    the run executable without making a false real-image claim.

    @param save_outputs Save result images if True.
    @return Real-vase test DataFrame.
    """
    image, _mask, contour, input_source, mask_source = _prepare_real_vase_case()
    rows: List[Dict[str, float | str | int]] = []

    experiments = [
        ("real_vase_paper", "paper", 4, "greedy"),
        ("real_vase_paper", "paper", 4, "graph"),
        ("real_vase_improved", "improved", 8, "greedy"),
    ]
    for case_name, ieps_mode, scan_lines, scf_method in experiments:
        row = run_single_case(
            case_name,
            image,
            contour,
            initial_scan_lines=scan_lines,
            ieps_mode=ieps_mode,
            scf_method=scf_method,
            save_outputs=save_outputs,
        )
        row["input_source"] = input_source
        row["mask_source"] = mask_source
        row["quantitative_note"] = (
            "proxy_metrics_from_estimated_or_generated_mask"
            if mask_source != "provided_mask"
            else "metrics_from_provided_mask"
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _contour_metric_row(
    result_group: str,
    case: str,
    method: str,
    image: np.ndarray,
    gt_contour: np.ndarray,
    contour_points: List[Tuple[int, int]],
    elapsed_ms: float,
    extra: Dict[str, float | str | int],
) -> Dict[str, float | str | int]:
    """@brief Build a paper-comparison metric row for contour points."""
    pred_mask = points_to_mask(contour_points, image.shape, thickness=1)
    metrics = contour_metrics(pred_mask, gt_contour, tolerance=2)
    row: Dict[str, float | str | int] = {
        "result_group": result_group,
        "case": case,
        "method": method,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "elapsed_ms": elapsed_ms,
    }
    row.update(extra)
    return row


def run_paper_comparison_experiments(save_outputs: bool = True) -> pd.DataFrame:
    """@brief Run fuller paper-style comparison experiments.

    This adds compact Yuen-style, Snake-style, and Chen-style approximations so
    the project can execute the comparison structure described in the paper.
    These are clearly labeled approximations, not exact source-code
    reproductions of the original comparison methods.

    @param save_outputs Save result images if True.
    @return Comparison results DataFrame.
    """
    rows: List[Dict[str, float | str | int]] = []
    out_dir = RESULTS_DIR / "paper_comparisons"
    cases = {
        "circle_noisy": _prepare_cases()["circle_noisy"],
        "u_shape_noisy": _prepare_cases()["u_shape_noisy"],
    }

    for case, (image, _mask, contour) in cases.items():
        gradient = sobel_gradient_magnitude(image)
        yuen_result, yuen_ms = measure_runtime(
            run_yuen_style_initialization,
            image,
            gradient,
            num_points=32,
            threshold=64.0,
        )
        ieps_result, ieps_ms = measure_runtime(
            run_ieps,
            image,
            gradient,
            initial_scan_lines=4,
            iterations=3,
            threshold=64.0,
            center_mode="contrast",
            order_mode="topological",
            refinement_selection="farthest_from_center",
            fallback_mode="max_gradient",
        )

        for method, result, elapsed in [
            ("yuen_style_initial_points", yuen_result, yuen_ms),
            ("ieps_paper_initial_points", ieps_result, ieps_ms),
        ]:
            stats = neighbor_distance_stats(result.points)
            accuracy = point_accuracy(result.points, contour, tolerance=2)
            rows.append({
                "result_group": "initial_point_selection",
                "case": case,
                "method": method,
                "precision": "",
                "recall": "",
                "f1": "",
                "elapsed_ms": elapsed,
                "point_count": len(result.points),
                "point_accuracy": accuracy,
                "true_points": int(round(accuracy * len(result.points))),
                "neighbor_mean": stats["neighbor_mean"],
                "neighbor_std": stats["neighbor_std"],
                "noise_snr_db": "",
                "input_source": "synthetic",
                "note": "Yuen-style is fixed-angle farthest-edge approximation.",
            })
            if save_outputs:
                save_image(out_dir / f"{case}_{method}.png", draw_points(image, result.points))

        for init_method, init_points in [
            ("yuen_style", yuen_result.points),
            ("ieps_paper", ieps_result.points),
        ]:
            snake_result, snake_ms = measure_runtime(run_simple_snake, gradient, init_points)
            rows.append(_contour_metric_row(
                "snake_initialization_comparison",
                case,
                f"simple_snake_from_{init_method}",
                image,
                contour,
                snake_result.points,
                snake_ms,
                {
                    "point_count": len(snake_result.points),
                    "point_accuracy": "",
                    "true_points": "",
                    "neighbor_mean": "",
                    "neighbor_std": "",
                    "noise_snr_db": "",
                    "input_source": "synthetic",
                    "note": "Simple greedy Snake-style approximation; not full Kass solver.",
                },
            ))
            if save_outputs:
                save_image(
                    out_dir / f"{case}_snake_from_{init_method}.png",
                    draw_contour_points(image, snake_result.points),
                )

    circle_img, _circle_mask, circle_contour = create_circle_image()
    for snr_db in [29.9, 23.9, 20.3]:
        image = add_gaussian_noise_snr(circle_img, snr_db=snr_db, seed=int(round(snr_db * 10)))
        gradient = sobel_gradient_magnitude(image)
        ieps_result = run_ieps(
            image,
            gradient,
            initial_scan_lines=4,
            iterations=3,
            threshold=64.0,
            center_mode="contrast",
            order_mode="topological",
            refinement_selection="farthest_from_center",
            fallback_mode="max_gradient",
        )
        chen_result, chen_ms = measure_runtime(
            run_chen_style_tracing,
            gradient,
            ieps_result.points,
            center=ieps_result.center,
            stop_tolerance=2.0,
        )
        proposed_result, proposed_ms = measure_runtime(
            run_scf,
            gradient,
            ieps_result.points,
            center=ieps_result.center,
            stop_tolerance=2.0,
            score_mode="gradient_distance2",
            method="greedy",
            sort_before_following=False,
        )
        for method, result, elapsed in [
            ("chen_style_gradient_only", chen_result, chen_ms),
            ("proposed_scf_gradient_distance", proposed_result, proposed_ms),
        ]:
            rows.append(_contour_metric_row(
                "scf_vs_chen_snr",
                f"circle_snr_{str(snr_db).replace('.', '_')}",
                method,
                image,
                circle_contour,
                result.contour_points,
                elapsed,
                {
                    "point_count": len(ieps_result.points),
                    "point_accuracy": point_accuracy(ieps_result.points, circle_contour, tolerance=2),
                    "true_points": "",
                    "neighbor_mean": "",
                    "neighbor_std": "",
                    "noise_snr_db": snr_db,
                    "input_source": "synthetic_snr",
                    "note": "Chen-style uses gradient-only SCF approximation; SNR noise is approximate.",
                },
            ))
            if save_outputs:
                save_image(
                    out_dir / f"circle_snr_{str(snr_db).replace('.', '_')}_{method}.png",
                    draw_contour_points(image, result.contour_points),
                )

    vase_image, _vase_mask, vase_contour, input_source, mask_source = _prepare_real_vase_case()
    vase_gradient = sobel_gradient_magnitude(vase_image)
    vase_ieps = run_ieps(
        vase_image,
        vase_gradient,
        initial_scan_lines=4,
        iterations=3,
        threshold=64.0,
        center_mode="contrast",
        order_mode="topological",
        refinement_selection="farthest_from_center",
        fallback_mode="max_gradient",
    )
    vase_snake, vase_snake_ms = measure_runtime(run_simple_snake, vase_gradient, vase_ieps.points)
    vase_chen, vase_chen_ms = measure_runtime(
        run_chen_style_tracing,
        vase_gradient,
        vase_ieps.points,
        center=vase_ieps.center,
        stop_tolerance=2.0,
    )
    vase_proposed, vase_proposed_ms = measure_runtime(
        run_scf,
        vase_gradient,
        vase_ieps.points,
        center=vase_ieps.center,
        stop_tolerance=2.0,
        score_mode="gradient_distance2",
        method="greedy",
        sort_before_following=False,
    )
    for method, points, elapsed in [
        ("simple_snake_kass_style", vase_snake.points, vase_snake_ms),
        ("chen_style_gradient_only", vase_chen.contour_points, vase_chen_ms),
        ("proposed_scf_gradient_distance", vase_proposed.contour_points, vase_proposed_ms),
    ]:
        rows.append(_contour_metric_row(
            "vase_comparison",
            "real_vase_or_fallback",
            method,
            vase_image,
            vase_contour,
            points,
            elapsed,
            {
                "point_count": len(vase_ieps.points),
                "point_accuracy": point_accuracy(vase_ieps.points, vase_contour, tolerance=2),
                "true_points": "",
                "neighbor_mean": "",
                "neighbor_std": "",
                "noise_snr_db": "",
                "input_source": input_source,
                "note": f"mask_source={mask_source}; Snake/Chen are compact approximations.",
            },
        ))
        if save_outputs:
            save_image(out_dir / f"vase_{method}.png", draw_contour_points(vase_image, points))

    return pd.DataFrame(rows)


def _friendly_method_name(method: str) -> str:
    """@brief Convert internal method keys to presentation labels."""
    labels = {
        "yuen_style_initial_points": "Yuen-style",
        "ieps_paper_initial_points": "IEPS",
        "simple_snake_from_yuen_style": "Snake from Yuen-style points",
        "simple_snake_from_ieps_paper": "Snake from IEPS points",
        "chen_style_gradient_only": "Chen-style",
        "proposed_scf_gradient_distance": "Proposed SCF",
        "simple_snake_kass_style": "Simple Snake-style",
    }
    return labels.get(method, method.replace("_", " "))


def _single_comparison_row(df: pd.DataFrame, group: str, case: str, method: str) -> pd.Series:
    """@brief Fetch one row from the raw paper-comparison table."""
    matches = df[
        (df["result_group"] == group)
        & (df["case"] == case)
        & (df["method"] == method)
    ]
    if matches.empty:
        raise ValueError(f"Missing comparison row for {group}/{case}/{method}")
    return matches.iloc[0]


def build_paper_comparison_tables(raw_results: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """@brief Build presentation-ready paper comparison tables.

    The raw paper comparison table intentionally keeps every measurement in one
    long-form CSV. These derived tables are cleaner for reports and slides.

    @param raw_results Raw DataFrame from run_paper_comparison_experiments.
    @return Mapping from output filename to formatted DataFrame.
    """
    tables: Dict[str, pd.DataFrame] = {}

    initial_rows: List[Dict[str, float | str | int]] = []
    for case in ["circle_noisy", "u_shape_noisy"]:
        yuen = _single_comparison_row(raw_results, "initial_point_selection", case, "yuen_style_initial_points")
        ieps = _single_comparison_row(raw_results, "initial_point_selection", case, "ieps_paper_initial_points")
        initial_rows.append({
            "case": case,
            "yuen_true_points": int(yuen["true_points"]),
            "yuen_total_points": int(yuen["point_count"]),
            "yuen_accuracy": yuen["point_accuracy"],
            "yuen_neighbor_mean": yuen["neighbor_mean"],
            "yuen_neighbor_std": yuen["neighbor_std"],
            "ieps_true_points": int(ieps["true_points"]),
            "ieps_total_points": int(ieps["point_count"]),
            "ieps_accuracy": ieps["point_accuracy"],
            "ieps_neighbor_mean": ieps["neighbor_mean"],
            "ieps_neighbor_std": ieps["neighbor_std"],
        })
    tables["paper_initial_point_comparison.csv"] = pd.DataFrame(initial_rows)

    snake_rows: List[Dict[str, float | str]] = []
    for case in ["circle_noisy", "u_shape_noisy"]:
        yuen_snake = _single_comparison_row(
            raw_results,
            "snake_initialization_comparison",
            case,
            "simple_snake_from_yuen_style",
        )
        ieps_snake = _single_comparison_row(
            raw_results,
            "snake_initialization_comparison",
            case,
            "simple_snake_from_ieps_paper",
        )
        snake_rows.append({
            "case": case,
            "yuen_snake_precision": yuen_snake["precision"],
            "yuen_snake_recall": yuen_snake["recall"],
            "yuen_snake_f1": yuen_snake["f1"],
            "yuen_snake_ms": yuen_snake["elapsed_ms"],
            "ieps_snake_precision": ieps_snake["precision"],
            "ieps_snake_recall": ieps_snake["recall"],
            "ieps_snake_f1": ieps_snake["f1"],
            "ieps_snake_ms": ieps_snake["elapsed_ms"],
        })
    tables["paper_snake_initialization_comparison.csv"] = pd.DataFrame(snake_rows)

    scf_rows: List[Dict[str, float | str]] = []
    for snr_db in [29.9, 23.9, 20.3]:
        case = f"circle_snr_{str(snr_db).replace('.', '_')}"
        chen = _single_comparison_row(raw_results, "scf_vs_chen_snr", case, "chen_style_gradient_only")
        proposed = _single_comparison_row(raw_results, "scf_vs_chen_snr", case, "proposed_scf_gradient_distance")
        scf_rows.append({
            "snr_db": snr_db,
            "chen_precision": chen["precision"],
            "chen_recall": chen["recall"],
            "chen_f1": chen["f1"],
            "chen_ms": chen["elapsed_ms"],
            "proposed_precision": proposed["precision"],
            "proposed_recall": proposed["recall"],
            "proposed_f1": proposed["f1"],
            "proposed_ms": proposed["elapsed_ms"],
        })
    tables["paper_scf_chen_comparison.csv"] = pd.DataFrame(scf_rows)

    vase_rows: List[Dict[str, float | str]] = []
    vase = raw_results[raw_results["result_group"] == "vase_comparison"]
    for _, row in vase.iterrows():
        vase_rows.append({
            "method": _friendly_method_name(str(row["method"])),
            "input_source": row["input_source"],
            "point_accuracy": row["point_accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "elapsed_ms": row["elapsed_ms"],
            "note": row["note"],
        })
    tables["paper_vase_method_comparison.csv"] = pd.DataFrame(vase_rows)

    return tables


def _paper_cell(value: object, decimals: int = 4) -> str:
    """@brief Format a scalar value for a compact paper-style CSV cell."""
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.{decimals}f}"


def _paper_ms_cell(value: object) -> str:
    """@brief Format elapsed time for a compact paper-style CSV cell."""
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value):.1f} ms"


def _paper_point_ratio(row: pd.Series) -> str:
    """@brief Format true-positive initial points as true/total."""
    return f"{int(float(row['true_points']))}/{int(float(row['point_count']))}"


def _paper_summary_row(
    paper_section: str,
    case_or_condition: str,
    metric: str,
    *,
    yuen_style: str = "",
    ieps: str = "",
    chen_style: str = "",
    proposed_scf: str = "",
    simple_snake_style: str = "",
    note: str = "",
) -> Dict[str, str]:
    """@brief Create one paper-style comparison row with stable columns."""
    return {
        "paper_section": paper_section,
        "case_or_condition": case_or_condition,
        "metric": metric,
        "yuen_style": yuen_style,
        "ieps": ieps,
        "chen_style": chen_style,
        "proposed_scf": proposed_scf,
        "simple_snake_style": simple_snake_style,
        "note": note,
    }


def build_paper_style_comparison_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    """@brief Convert raw paper-comparison runs into one row-column table.

    This table mirrors the paper's style more closely than the raw measurement
    log: each row is a paper figure/table item, and each method has its own
    column. The detailed long-form data remains available separately.

    @param raw_results Raw DataFrame from run_paper_comparison_experiments.
    @return Compact paper-style summary DataFrame.
    """
    rows: List[Dict[str, str]] = []

    for case in ["circle_noisy", "u_shape_noisy"]:
        yuen = _single_comparison_row(raw_results, "initial_point_selection", case, "yuen_style_initial_points")
        ieps = _single_comparison_row(raw_results, "initial_point_selection", case, "ieps_paper_initial_points")
        rows.append(_paper_summary_row(
            "Table II - initial edge points",
            case,
            "true points / total",
            yuen_style=_paper_point_ratio(yuen),
            ieps=_paper_point_ratio(ieps),
            note="Yuen-style is an approximation from the paper description.",
        ))
        rows.append(_paper_summary_row(
            "Table II - initial edge points",
            case,
            "point accuracy",
            yuen_style=_paper_cell(yuen["point_accuracy"]),
            ieps=_paper_cell(ieps["point_accuracy"]),
        ))
        rows.append(_paper_summary_row(
            "Table III - neighboring point spacing",
            case,
            "mean distance",
            yuen_style=_paper_cell(yuen["neighbor_mean"], decimals=3),
            ieps=_paper_cell(ieps["neighbor_mean"], decimals=3),
        ))
        rows.append(_paper_summary_row(
            "Table III - neighboring point spacing",
            case,
            "std distance",
            yuen_style=_paper_cell(yuen["neighbor_std"], decimals=3),
            ieps=_paper_cell(ieps["neighbor_std"], decimals=3),
        ))

    for case in ["circle_noisy", "u_shape_noisy"]:
        yuen_snake = _single_comparison_row(
            raw_results,
            "snake_initialization_comparison",
            case,
            "simple_snake_from_yuen_style",
        )
        ieps_snake = _single_comparison_row(
            raw_results,
            "snake_initialization_comparison",
            case,
            "simple_snake_from_ieps_paper",
        )
        rows.append(_paper_summary_row(
            "Fig. 7 - Snake initialization",
            case,
            "F1",
            yuen_style=_paper_cell(yuen_snake["f1"]),
            ieps=_paper_cell(ieps_snake["f1"]),
            simple_snake_style="approximation",
            note="Simple greedy Snake-style approximation, not the full Kass solver.",
        ))
        rows.append(_paper_summary_row(
            "Fig. 7 - Snake initialization",
            case,
            "runtime",
            yuen_style=_paper_ms_cell(yuen_snake["elapsed_ms"]),
            ieps=_paper_ms_cell(ieps_snake["elapsed_ms"]),
            simple_snake_style="approximation",
        ))

    for snr_db in [29.9, 23.9, 20.3]:
        case = f"circle_snr_{str(snr_db).replace('.', '_')}"
        chen = _single_comparison_row(raw_results, "scf_vs_chen_snr", case, "chen_style_gradient_only")
        proposed = _single_comparison_row(raw_results, "scf_vs_chen_snr", case, "proposed_scf_gradient_distance")
        rows.append(_paper_summary_row(
            "Table IV - SCF tracing",
            f"SNR={snr_db:.1f} dB",
            "F1",
            chen_style=_paper_cell(chen["f1"]),
            proposed_scf=_paper_cell(proposed["f1"]),
            note="Chen-style baseline and SNR noise are documented approximations.",
        ))
        rows.append(_paper_summary_row(
            "Table IV - SCF tracing",
            f"SNR={snr_db:.1f} dB",
            "runtime",
            chen_style=_paper_ms_cell(chen["elapsed_ms"]),
            proposed_scf=_paper_ms_cell(proposed["elapsed_ms"]),
        ))

    vase_case = "real_vase_or_fallback"
    snake = _single_comparison_row(raw_results, "vase_comparison", vase_case, "simple_snake_kass_style")
    chen = _single_comparison_row(raw_results, "vase_comparison", vase_case, "chen_style_gradient_only")
    proposed = _single_comparison_row(raw_results, "vase_comparison", vase_case, "proposed_scf_gradient_distance")
    input_source = str(proposed["input_source"])
    rows.append(_paper_summary_row(
        "Fig. 10 / Table V - real vase",
        input_source,
        "F1",
        chen_style=_paper_cell(chen["f1"]),
        proposed_scf=_paper_cell(proposed["f1"]),
        simple_snake_style=_paper_cell(snake["f1"]),
        note=str(proposed["note"]),
    ))
    rows.append(_paper_summary_row(
        "Fig. 10 / Table V - real vase",
        input_source,
        "runtime",
        chen_style=_paper_ms_cell(chen["elapsed_ms"]),
        proposed_scf=_paper_ms_cell(proposed["elapsed_ms"]),
        simple_snake_style=_paper_ms_cell(snake["elapsed_ms"]),
    ))

    return pd.DataFrame(rows)


def _markdown_value(value: object) -> str:
    """@brief Format a value for generated Markdown tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """@brief Convert a DataFrame to a simple GitHub-flavored Markdown table."""
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_markdown_value(row[column]) for column in df.columns) + " |")
    return "\n".join(lines)


def write_paper_comparison_tables(raw_results: pd.DataFrame) -> Dict[str, Path]:
    """@brief Write presentation-ready paper comparison tables.

    @param raw_results Raw DataFrame from run_paper_comparison_experiments.
    @return Mapping from filename to written path.
    """
    tables = build_paper_comparison_tables(raw_results)
    written: Dict[str, Path] = {}
    for filename, table in tables.items():
        path = TABLES_DIR / filename
        table.to_csv(path, index=False, float_format="%.4f")
        written[filename] = path

    markdown_path = TABLES_DIR / "paper_comparison_tables.md"
    markdown_sections = [
        "# Paper Comparison Tables",
        "",
        "These tables are presentation-ready summaries derived from `paper_comparison_results_raw.csv`.",
    ]
    for filename, table in tables.items():
        title = filename.replace(".csv", "").replace("_", " ").title()
        markdown_sections.extend(["", f"## {title}", "", _dataframe_to_markdown(table)])
    markdown_path.write_text("\n".join(markdown_sections) + "\n", encoding="utf-8")
    written["paper_comparison_tables.md"] = markdown_path
    return written


def _ensure_output_dirs() -> None:
    """@brief Create output directories used by selected experiment runs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def _normalise_runs(requested_runs: Sequence[str]) -> List[str]:
    """@brief Return selected runs in stable execution order."""
    if "all" in requested_runs:
        return RUN_ORDER.copy()
    requested = set(requested_runs)
    return [run_name for run_name in RUN_ORDER if run_name in requested]


def _normalise_scf_methods(requested_methods: Sequence[str]) -> List[str]:
    """@brief Expand the SCF method selector."""
    if "all" in requested_methods:
        return SCF_METHODS.copy()
    return list(dict.fromkeys(requested_methods))


def _selected_cases(case_selector: str) -> List[str]:
    """@brief Expand the main-experiment case selector."""
    if case_selector == "all":
        return MAIN_CASES.copy()
    return [case_selector]


def _main_results_filename(case_selector: str, requested_scf_methods: Sequence[str]) -> str:
    """@brief Choose the main results table name for the selected filters.

    The default paper-method run (all cases, greedy SCF) owns main_results.csv.
    Opting into every SCF method writes the extension table
    main_results_all_scf.csv, and any other filter uses the selected table so
    the canonical files are never overwritten by partial runs.
    """
    methods = set(_normalise_scf_methods(list(requested_scf_methods)))
    if case_selector == "all" and methods == {"greedy"}:
        return "main_results.csv"
    if case_selector == "all" and methods == set(SCF_METHODS):
        return "main_results_all_scf.csv"
    return "main_results_selected.csv"


def _write_and_print_table(
    title: str,
    df: pd.DataFrame,
    filename: str,
    summary_columns: Sequence[str],
) -> None:
    """@brief Save one result table and print a compact terminal summary."""
    path = TABLES_DIR / filename
    df.to_csv(path, index=False, float_format="%.4f")
    relative_path = path.relative_to(BASE_DIR)
    print(f"\n{title}: {len(df)} rows -> {relative_path}")
    available_columns = [column for column in summary_columns if column in df.columns]
    if available_columns and not df.empty:
        print(df[available_columns].to_string(index=False))


def _print_available_runs() -> None:
    """@brief Print run modes for the CLI."""
    print("Available runs:")
    for run_name in [*RUN_ORDER, "all"]:
        print(f"  {run_name:16} {RUN_DESCRIPTIONS[run_name]}")
    print("\nExamples:")
    print("  python main.py")
    print("  python main.py --run main --scf all")
    print("  python main.py --run main --case u_shape_noisy --scf graph")
    print("  python main.py --run parameter")
    print("  python main.py --run paper-comparison")
    print("  python main.py --run all")


def _build_parser() -> argparse.ArgumentParser:
    """@brief Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run selected IEPS + SCF experiments instead of regenerating everything each time.",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        choices=[*RUN_ORDER, "all"],
        default=["main"],
        help="Experiment group(s) to run. Default: main.",
    )
    parser.add_argument(
        "--case",
        choices=[*MAIN_CASES, "all"],
        default="all",
        help="Case filter for --run main. Default: all.",
    )
    parser.add_argument(
        "--scf",
        nargs="+",
        choices=[*SCF_METHODS, "all"],
        default=["greedy"],
        help="SCF method filter for --run main. Default: greedy (paper method); graph/band_graph/all are opt-in extensions.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Write CSV tables only; skip generated PNG figures for selected runs.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show available run modes and exit.",
    )
    return parser


def run_cli(args: argparse.Namespace) -> None:
    """@brief Execute selected CLI run modes."""
    _ensure_output_dirs()
    requested_runs = _normalise_runs(args.run)
    save_outputs = not args.no_images
    selected_cases = _selected_cases(args.case)
    selected_scf_methods = _normalise_scf_methods(args.scf)
    cache: Dict[str, pd.DataFrame] = {}

    def get_main_results(full: bool = False) -> pd.DataFrame:
        key = "main_full" if full else "main_requested"
        if key not in cache:
            cache[key] = run_main_experiments(
                selected_cases=None if full else selected_cases,
                selected_scf_methods=None if full else selected_scf_methods,
                save_outputs=save_outputs if not full else False,
            )
        return cache[key]

    def get_improvement_results() -> pd.DataFrame:
        if "improvement" not in cache:
            cache["improvement"] = run_improvement_comparison()
        return cache["improvement"]

    def get_real_vase_results(write_images: bool) -> pd.DataFrame:
        key = "vase_with_images" if write_images else "vase_no_images"
        if key not in cache:
            cache[key] = run_real_vase_test(save_outputs=write_images)
        return cache[key]

    def get_paper_comparison_results(write_images: bool) -> pd.DataFrame:
        key = "paper_comparison_with_images" if write_images else "paper_comparison_no_images"
        if key not in cache:
            cache[key] = run_paper_comparison_experiments(save_outputs=write_images)
        return cache[key]

    for run_name in requested_runs:
        if run_name == "main":
            main_df = get_main_results(full=False)
            main_filename = _main_results_filename(args.case, args.scf)
            _write_and_print_table(
                "Main synthetic results",
                main_df,
                main_filename,
                ["case", "scf_method", "ieps_points", "ieps_accuracy", "precision", "recall", "f1", "total_ms"],
            )
            runtime_columns = ["case", "scf_method", "ieps_ms", "scf_ms", "total_ms", "canny_ms"]
            _write_and_print_table(
                "Runtime results",
                main_df[[column for column in runtime_columns if column in main_df.columns]],
                main_filename.replace("main_results", "runtime_results"),
                runtime_columns,
            )
        elif run_name == "parameter":
            cache["parameter"] = run_parameter_study()
            _write_and_print_table(
                "Parameter study",
                cache["parameter"],
                "parameter_study.csv",
                ["case", "threshold", "scan_lines", "iterations", "scf_tolerance", "scf_score_mode", "ieps_accuracy", "f1"],
            )
        elif run_name == "improvement":
            _write_and_print_table(
                "Improvement comparison",
                get_improvement_results(),
                "improvement_comparison.csv",
                ["case", "paper_f1", "improved_f1", "f1_delta", "paper_ieps_accuracy", "improved_ieps_accuracy"],
            )
        elif run_name == "vase":
            _write_and_print_table(
                "Real vase test",
                get_real_vase_results(write_images=save_outputs),
                "real_vase_results.csv",
                ["case", "input_source", "mask_source", "ieps_mode", "scf_method", "ieps_points", "f1", "total_ms"],
            )
        elif run_name == "paper-comparison":
            paper_comparison_raw = get_paper_comparison_results(write_images=save_outputs)
            _write_and_print_table(
                "Raw paper comparison log",
                paper_comparison_raw,
                "paper_comparison_results_raw.csv",
                ["result_group", "case", "method", "point_accuracy", "f1", "elapsed_ms"],
            )
            paper_comparison_results = build_paper_style_comparison_results(paper_comparison_raw)
            _write_and_print_table(
                "Paper-style comparison table",
                paper_comparison_results,
                "paper_comparison_results.csv",
                [
                    "paper_section",
                    "case_or_condition",
                    "metric",
                    "yuen_style",
                    "ieps",
                    "chen_style",
                    "proposed_scf",
                    "simple_snake_style",
                ],
            )
            written_tables = write_paper_comparison_tables(paper_comparison_raw)
            print("Presentation comparison tables:")
            for path in written_tables.values():
                print(f"  {path.relative_to(BASE_DIR)}")

    print(f"\nDone. Selected run(s): {', '.join(requested_runs)}")
    print("Results directory:", RESULTS_DIR)


def main(argv: Sequence[str] | None = None) -> None:
    """@brief Entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list:
        _print_available_runs()
        return
    run_cli(args)


if __name__ == "__main__":
    main()
