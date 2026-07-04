"""Build the final IEPS + SCF presentation as an editable PPTX.

The preferred artifact-tool runtime is not available in this workspace, so this
script writes a compact OpenXML PowerPoint file directly and renders matching
PNG previews with Pillow for visual QA.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PPTX_PATH = OUT / "ieps_scf_final_presentation.pptx"
PREVIEW_DIR = OUT / "ieps_scf_final_presentation_preview"
PREVIEW_DIR.mkdir(exist_ok=True)
WORK = Path(os.environ.get("TEMP", str(OUT))) / "ieps_scf_final_pptx_build"
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)

SLIDE_W = 1280
SLIDE_H = 720
EMU = 9525

BG = "F7F7F2"
INK = "111827"
MUTED = "5B6472"
LINE = "D7D7CC"
DARK = "101418"
PAPER = "FFFFFF"
ACCENT = "0F766E"
ACCENT2 = "C2410C"
BLUE = "1D4ED8"
AMBER = "B45309"
RED = "B91C1C"
GREEN = "15803D"

FONT_REG = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.replace("#", "")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def emu(v: float) -> int:
    return int(round(v * EMU))


def xesc(text: Any) -> str:
    return escape(str(text), quote=False)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pct(v: Any, decimals: int = 1) -> str:
    return f"{float(v) * 100:.{decimals}f}%"


def get_rows(rows: List[Dict[str, str]], **criteria: str) -> List[Dict[str, str]]:
    output = []
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            output.append(row)
    return output


main_rows = read_csv(ROOT / "results" / "tables" / "main_results.csv")
param_rows = read_csv(ROOT / "results" / "tables" / "parameter_study.csv")
improve_rows = read_csv(ROOT / "results" / "tables" / "improvement_comparison.csv")
vase_rows = read_csv(ROOT / "results" / "tables" / "real_vase_results.csv")


def main_case(case: str, method: str = "greedy") -> Dict[str, str]:
    return get_rows(main_rows, case=case, scf_method=method)[0]


circle_clean = main_case("circle_clean")
circle_noisy = main_case("circle_noisy")
u_clean = main_case("u_shape_clean")
u_noisy = main_case("u_shape_noisy")
u_graph = main_case("u_shape_noisy", "graph")
u_band = main_case("u_shape_noisy", "band_graph")


@dataclass
class Element:
    kind: str
    x: float
    y: float
    w: float
    h: float
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Slide:
    title: str
    elements: List[Element] = field(default_factory=list)


slides: List[Slide] = []


def add(slide: Slide, kind: str, x: float, y: float, w: float, h: float, **props: Any) -> None:
    slide.elements.append(Element(kind, x, y, w, h, props))


def add_text(
    slide: Slide,
    text: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    size: int = 22,
    color: str = INK,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
    name: str = "text",
    fill: Optional[str] = None,
    line: Optional[str] = None,
) -> None:
    add(
        slide,
        "text",
        x,
        y,
        w,
        h,
        text=text,
        size=size,
        color=color,
        bold=bold,
        align=align,
        valign=valign,
        name=name,
        fill=fill,
        line=line,
    )


def add_rect(slide: Slide, x: float, y: float, w: float, h: float, fill: str = PAPER, line: str = LINE) -> None:
    add(slide, "rect", x, y, w, h, fill=fill, line=line)


def add_image(
    slide: Slide,
    path: Path,
    x: float,
    y: float,
    w: float,
    h: float,
    fit: str = "contain",
    caption: Optional[str] = None,
) -> None:
    add(slide, "image", x, y, w, h, path=path, fit=fit)
    if caption:
        add_text(slide, caption, x, y + h + 8, w, 24, size=12, color=MUTED, align="center")


def add_title(slide: Slide, title: str, kicker: str = "", subtitle: str = "") -> None:
    if kicker:
        add_text(slide, kicker.upper(), 64, 36, 720, 24, size=12, color=ACCENT, bold=True)
    add_text(slide, title, 64, 62, 900, 78, size=34, color=INK, bold=True)
    if subtitle:
        add_text(slide, subtitle, 64, 124, 1030, 42, size=15, color=MUTED)
    add(slide, "line", 64, 154, 1152, 1, fill=LINE)


def add_footer(slide: Slide, n: int) -> None:
    add_text(slide, f"IEPS + SCF reproducibility study | {n:02d}", 64, 676, 420, 20, size=10, color="727B85")
    add_text(
        slide,
        "Generated from project source, CSV tables, and result figures",
        760,
        676,
        456,
        20,
        size=10,
        color="727B85",
        align="right",
    )


def add_card(slide: Slide, x: float, y: float, w: float, h: float, title: str, body: str, accent: str = ACCENT) -> None:
    add_rect(slide, x, y, w, h, fill=PAPER, line=LINE)
    add_rect(slide, x, y, 7, h, fill=accent, line=accent)
    add_text(slide, title, x + 18, y + 16, w - 28, 28, size=17, color=INK, bold=True)
    add_text(slide, body, x + 18, y + 50, w - 32, h - 60, size=14, color=MUTED)


def add_table(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    headers: List[str],
    rows: List[List[str]],
    col_fracs: Optional[List[float]] = None,
    font_size: int = 12,
) -> None:
    ncols = len(headers)
    col_fracs = col_fracs or [1.0 / ncols] * ncols
    col_widths = [w * fraction / sum(col_fracs) for fraction in col_fracs]
    header_h = 36
    row_h = (h - header_h) / max(len(rows), 1)

    cx = x
    for c, header in enumerate(headers):
        cw = col_widths[c]
        add_rect(slide, cx, y, cw, header_h, fill=DARK, line=DARK)
        add_text(slide, header, cx + 8, y + 7, cw - 16, header_h - 8, size=font_size, color="FFFFFF", bold=True, valign="middle")
        cx += cw

    for r, row in enumerate(rows):
        cy = y + header_h + r * row_h
        cx = x
        fill = "FFFFFF" if r % 2 == 0 else "F1F5F2"
        for c, value in enumerate(row):
            cw = col_widths[c]
            add_rect(slide, cx, cy, cw, row_h, fill=fill, line=LINE)
            add_text(slide, value, cx + 8, cy + 6, cw - 16, row_h - 10, size=font_size, color=INK if c == 0 else MUTED, bold=c == 0, valign="middle")
            cx += cw


def add_bar_chart(
    slide: Slide,
    x: float,
    y: float,
    w: float,
    h: float,
    labels: List[str],
    values: List[float],
    max_value: float = 1.0,
    color: str = ACCENT,
    title: str = "",
) -> None:
    if title:
        add_text(slide, title, x, y - 34, w, 28, size=18, color=INK, bold=True)
    left_label = 150
    bar_gap = 14
    bar_h = (h - bar_gap * (len(values) - 1)) / len(values)
    for i, (label, value) in enumerate(zip(labels, values)):
        yy = y + i * (bar_h + bar_gap)
        add_text(slide, label, x, yy + 2, left_label - 12, bar_h, size=13, color=INK, bold=True, valign="middle")
        add_rect(slide, x + left_label, yy + 5, w - left_label - 70, bar_h - 10, fill="E5E7EB", line="E5E7EB")
        bar_w = (w - left_label - 70) * min(max(value / max_value, 0), 1)
        add_rect(slide, x + left_label, yy + 5, bar_w, bar_h - 10, fill=color, line=color)
        add_text(slide, f"{value * 100:.1f}%", x + w - 64, yy + 3, 64, bar_h, size=12, color=INK, bold=True, valign="middle")


def add_pipeline(slide: Slide, items: List[Tuple[str, str]], x: float, y: float, w: float, h: float) -> None:
    gap = 18
    box_w = (w - gap * (len(items) - 1)) / len(items)
    for i, (head, body) in enumerate(items):
        bx = x + i * (box_w + gap)
        add_rect(slide, bx, y, box_w, h, fill=PAPER, line=LINE)
        add_text(slide, head, bx + 14, y + 18, box_w - 28, 32, size=16, color=INK, bold=True, align="center")
        add_text(slide, body, bx + 14, y + 58, box_w - 28, h - 72, size=12, color=MUTED, align="center")
        if i < len(items) - 1:
            add(slide, "arrow", bx + box_w + 4, y + h / 2 - 10, gap - 8, 20, fill=ACCENT, line=ACCENT)


def build_slides() -> None:
    s = Slide("Title")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_text(s, "REIMPLEMENTATION AND VALIDATION", 64, 54, 740, 28, size=13, color=ACCENT, bold=True)
    add_text(s, "IEPS + SCF\nObject Contour Extraction", 64, 112, 650, 130, size=38, color=INK, bold=True)
    add_text(s, "A traditional computer vision reproducibility study of Hsu et al. (ICARCV 2010)", 64, 252, 720, 56, size=18, color=MUTED)
    add_card(s, 64, 382, 342, 150, "Final research question", "Can the method be rebuilt from the paper description and validated on author-style circle and U-shape images, including noise, while identifying missing implementation choices?", ACCENT)
    add_card(s, 430, 382, 300, 150, "Core claim", "The method is reproducible for star-convex shapes; concave U-shapes expose under-specified assumptions in IEPS and SCF.", BLUE)
    add_image(s, ROOT / "results" / "u_shape_noisy" / "panel_greedy.png", 820, 116, 380, 284, caption="Project-generated U-shape noisy IEPS + SCF panel")
    add_rect(s, 820, 430, 380, 86, fill=INK, line=INK)
    add_text(s, "Contribution: implementation + validation + missing-detail audit", 844, 454, 332, 38, size=17, color="FFFFFF", bold=True, align="center", valign="middle")
    slides.append(s)

    s = Slide("Research direction")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Research direction after professor feedback", "Framing", "The project is not a broad Canny comparison. It is a reproducibility study of the paper's own IEPS + SCF pipeline.")
    add_card(s, 64, 194, 346, 170, "Original risk", "A broad edge-detector comparison would drift away from the authors' contribution and become a generic segmentation project.", RED)
    add_card(s, 466, 194, 346, 170, "Correct focus", "Reimplement IEPS and SCF, test the same style of synthetic images, and document the assumptions needed for executable code.", ACCENT)
    add_card(s, 868, 194, 346, 170, "Best extension", "Stay inside the paper: run parameter validation for scan lines, thresholds, IEPS iterations, SCF stopping, scoring, and noise.", BLUE)
    add_text(s, "Presentation stance", 64, 430, 300, 30, size=24, color=INK, bold=True)
    add_text(s, "After discussion with the professor, I focused on the authors' original direction: reimplementing and validating IEPS + SCF. Instead of adding unrelated preprocessing or large external methods, I investigated the parameters and implementation choices necessary to reproduce the paper's contour extraction behavior.", 64, 472, 1120, 88, size=20, color=INK, bold=True)
    slides.append(s)

    s = Slide("Paper method")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "What Hsu et al. propose", "Paper model", "The paper couples automatic initialization with local contour following to avoid manual Snake initialization.")
    add_pipeline(s, [("Input image", "circle, U-shape, noisy versions, and vase"), ("Sobel gradient", "edge strength for IEPS and SCF"), ("IEPS", "CoG, scan lines, 3 refinements, T=64"), ("SCF", "trace between neighboring IEPS points"), ("Closed contour", "combine segmental contours")], 78, 214, 1124, 168)
    add_table(s, 94, 434, 1092, 164, ["Paper component", "Implemented interpretation"], [["IEPS", "4 initial scan lines, 3 iterations, threshold 64, about 32 final edge points."], ["Related direction", "Dx and Dy signs classify the target direction between neighboring IEPS points."], ["Gravity-style SCF", "Candidate score combines Sobel gradient strength with distance-to-target information."], ["Validation", "Selected points and final contour are compared against known synthetic ground truth."]], [0.28, 0.72], 13)
    slides.append(s)

    s = Slide("Source map")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Implementation map from source code", "Read-through result", "The codebase separates image synthesis, IEPS, SCF, evaluation, baselines, and reporting into testable modules.")
    modules = [("image_generation.py", "circle, U-shape, noise, vase path"), ("gradients.py", "Sobel magnitude normalized to 0..255"), ("geometry.py", "CoG, ray sampling, line sampling, ordering"), ("ieps.py", "paper-faithful initial edge selection"), ("scf.py", "greedy, graph, band-graph contour following"), ("evaluation.py", "point accuracy, contour metrics, runtime"), ("paper_baselines.py", "Yuen/Snake/Chen-style approximations"), ("main.py", "CLI run modes and table generation")]
    for i, (head, body) in enumerate(modules):
        x = 64 + (i % 4) * 288
        y = 204 + (i // 4) * 150
        add_card(s, x, y, 252, 110, head, body, [ACCENT, BLUE, AMBER, ACCENT2][i % 4])
    add_text(s, "Important design decision", 78, 534, 318, 30, size=22, color=INK, bold=True)
    add_text(s, "The unavailable Yuen, Kass/Snake, and Chen methods are implemented only as clearly labeled approximations. The core claim rests on the IEPS + SCF implementation and its own validation artifacts.", 78, 572, 1070, 44, size=18, color=INK)
    slides.append(s)

    s = Slide("IEPS")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "IEPS: executable interpretation", "Initial edge point selection", "The paper's geometric description is made concrete through scan-line sampling and candidate selection rules.")
    add_image(s, ROOT / "results" / "circle_noisy" / "ieps_points.png", 82, 210, 250, 250, caption="circle noisy: IEPS points")
    add_image(s, ROOT / "results" / "u_shape_noisy" / "ieps_points.png", 376, 210, 250, 250, caption="U-shape noisy: IEPS points")
    add_table(s, 680, 198, 500, 276, ["Step", "Concrete implementation"], [["Center", "contrast-weighted center of gravity"], ["First rays", "4 fixed scan lines through center"], ["Refinement", "midpoints + normal scan lines"], ["Iterations", "3 refinement passes"], ["Threshold", "Sobel magnitude >= 64"], ["Fallback", "max-gradient candidate if threshold misses"]], [0.34, 0.66], 12)
    add_card(s, 82, 514, 1098, 78, "Finding", "IEPS works cleanly for the circle because the center of gravity sits inside a star-convex object. The U-shape is harder because the center can lie in or near the concavity, changing ray intersections and point order.", ACCENT)
    slides.append(s)

    s = Slide("SCF")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "SCF: from flow chart to code", "Segmental contour following", "Each neighboring IEPS pair defines one segment; the implementation turns the paper's direction mask into deterministic pixel tracing.")
    add_pipeline(s, [("Si -> Si+1", "neighboring IEPS points define target direction"), ("Direction state", "sign(Dx), sign(Dy) maps to N, NE, E, ..."), ("Candidate mask", "local candidates consistent with direction"), ("Score", "gradient / (distance^2 + 1) by default"), ("Stop", "target reached within tolerance or guard rule fires")], 74, 206, 1132, 150)
    add_table(s, 78, 420, 548, 160, ["Paper phrase", "Implemented assumption"], [["closed contour?", "all segments finish and concatenate"], ["force of gravity", "gradient plus inverse squared target distance"], ["mask direction", "three candidate moves by related direction"], ["avoid loops", "visited-pixel and max-step guards"]], [0.38, 0.62], 12)
    add_image(s, ROOT / "results" / "circle_noisy" / "scf_greedy.png", 700, 412, 220, 180, caption="circle SCF")
    add_image(s, ROOT / "results" / "u_shape_noisy" / "scf_greedy.png", 956, 412, 220, 180, caption="U-shape SCF")
    slides.append(s)

    s = Slide("Validation design")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Validation design", "Experiments", "The validation is intentionally close to the paper: author-style synthetic shapes, Gaussian noise, IEPS point accuracy, final contour metrics, and runtime.")
    add_image(s, ROOT / "results" / "circle_noisy" / "panel_greedy.png", 64, 204, 540, 236, caption="circle_noisy panel")
    add_image(s, ROOT / "results" / "u_shape_noisy" / "panel_greedy.png", 676, 204, 540, 236, caption="u_shape_noisy panel")
    add_table(s, 86, 504, 1088, 100, ["Metric", "Purpose", "Current operational definition"], [["IEPS accuracy", "validate initial edge points", "point is correct if within 2 px of ground-truth contour"], ["Precision / recall / F1", "validate final contour", "predicted contour mask compared to ground truth with tolerance"], ["Runtime", "efficiency", "IEPS time + SCF time in milliseconds"]], [0.18, 0.27, 0.55], 12)
    slides.append(s)

    s = Slide("Main results")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Main result: circles reproduce cleanly, U-shapes expose assumptions", "Quantitative finding", "The default paper-mode IEPS + greedy SCF is excellent on circles and weaker on concave U-shapes.")
    labels = ["circle clean", "circle noisy", "U clean", "U noisy"]
    add_bar_chart(s, 80, 220, 520, 250, labels, [float(circle_clean["f1"]), float(circle_noisy["f1"]), float(u_clean["f1"]), float(u_noisy["f1"])], color=ACCENT, title="Final contour F1")
    add_bar_chart(s, 684, 220, 500, 250, labels, [float(circle_clean["ieps_accuracy"]), float(circle_noisy["ieps_accuracy"]), float(u_clean["ieps_accuracy"]), float(u_noisy["ieps_accuracy"])], color=BLUE, title="IEPS point accuracy")
    add_table(s, 102, 530, 1030, 92, ["Case", "IEPS", "F1", "Segments reached", "Runtime"], [["circle_noisy", pct(circle_noisy["ieps_accuracy"]), pct(circle_noisy["f1"]), f"{circle_noisy['target_reached_segments']}/{circle_noisy['total_segments']}", f"{float(circle_noisy['total_ms']):.1f} ms"], ["u_shape_noisy", pct(u_noisy["ieps_accuracy"]), pct(u_noisy["f1"]), f"{u_noisy['target_reached_segments']}/{u_noisy['total_segments']}", f"{float(u_noisy['total_ms']):.1f} ms"]], [0.28, 0.16, 0.16, 0.22, 0.18], 12)
    slides.append(s)

    s = Slide("What authors forgot")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "What the paper forgot to specify", "Reproducibility contribution", "These are not small coding details. They directly change selected IEPS points, segment tracing, and reported accuracy.")
    add_table(s, 64, 190, 1152, 424, ["Missing detail", "Why it matters", "Project assumption"], [["Scan-line discretization", "which pixels lie on a ray or normal line", "implemented explicit row/column sampling"], ["Threshold tuning", "64 works for paper scale, not universal", "tested 40/64/90"], ["Noise generation", "SNR cannot be reproduced exactly", "documented Gaussian sigma and SNR cases"], ["True-positive tolerance", "point accuracy changes with radius", "used 2 px tolerance"], ["SCF stopping", "'closed contour?' is not code", "target tolerance plus guard rules"], ["Tie-breaking", "many candidate pixels can score similarly", "deterministic candidate order"], ["Loop prevention", "contour following can revisit pixels", "visited-set and max-step checks"], ["Coordinate convention", "paper x/y vs OpenCV row/column", "kept public points as (x, y)"]], [0.26, 0.37, 0.37], 11)
    slides.append(s)

    s = Slide("Parameter validation")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Parameter validation inside IEPS + SCF", "Best extension", "The most defensible extension is not an external method; it is testing the under-specified choices inside the proposed algorithm.")
    add_table(s, 64, 196, 740, 360, ["Choice", "Tested values", "Interpretation"], [["Sobel threshold", "40, 64, 90", "U-noisy F1 changes only modestly; 64 is plausible but not uniquely optimal."], ["Initial scan lines", "4 vs 8", "More rays do not automatically improve paper-mode U-shape performance."], ["IEPS iterations", "2 vs 3", "Additional refinement can shift points rather than monotonically improve them."], ["SCF tolerance", "1, 2, 3 px", "Stopping tolerance changes contour completeness and precision trade-off."], ["Score formula", "gradient only vs gradient/distance", "Distance term makes the rule closer to the paper's gravity description."], ["Noise", "clean, low, paper-like", "Noise primarily hurts concave U-shape contour quality."]], [0.24, 0.22, 0.54], 11)
    thr = [(r["threshold"], r["f1"]) for r in param_rows if r["case"].startswith("param_threshold")]
    add_bar_chart(s, 850, 240, 320, 168, [f"T={int(float(t))}" for t, _ in thr], [float(v) for _, v in thr], color=AMBER, title="Threshold study F1")
    add_bar_chart(s, 850, 488, 320, 114, ["clean", "low", "paper-like"], [float(r["f1"]) for r in param_rows if r["case"].startswith("param_noise")], color=BLUE, title="Noise study F1")
    slides.append(s)

    s = Slide("Improved IEPS")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Traditional-CV improvement: interior seed + denser coverage", "Extension", "The improved mode keeps the research inside IEPS: it changes the seed and coverage, not the problem into deep learning or unrelated preprocessing.")
    add_card(s, 70, 204, 330, 150, "Problem in paper IEPS", "The paper assumes the center of gravity lies inside a star-convex object. On concave U-shapes, this can place rays through the notch or create poor angular ordering.", RED)
    add_card(s, 474, 204, 330, 150, "Improved IEPS", "Use an Otsu silhouette only to check the seed. If the CoG is outside foreground, relocate to the distance-transform maximum, then use denser radial coverage.", ACCENT)
    add_card(s, 878, 204, 330, 150, "Research position", "This is an extension, not the main method. The report preserves paper-mode IEPS + greedy SCF as the primary reproduction.", BLUE)
    imp_table = []
    for r in improve_rows:
        if "u_shape" in r["case"] or r["case"] == "circle_noisy":
            imp_table.append([r["case"], pct(r["paper_f1"]), pct(r["improved_f1"]), f"+{float(r['f1_delta']) * 100:.1f} pts", pct(r["paper_ieps_accuracy"]), pct(r["improved_ieps_accuracy"])])
    add_table(s, 94, 442, 1088, 132, ["Case", "Paper F1", "Improved F1", "Delta", "Paper IEPS", "Improved IEPS"], imp_table, [0.24, 0.15, 0.16, 0.13, 0.16, 0.16], 12)
    slides.append(s)

    s = Slide("SCF variants")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "SCF implementation variants", "Method audit", "The paper-faithful greedy SCF is the default. Graph variants test whether the local segment-following assumption is responsible for U-shape errors.")
    add_table(s, 78, 196, 520, 210, ["Method", "Role", "U noisy F1", "Runtime"], [["greedy", "paper-style local tracing", pct(u_noisy["f1"]), f"{float(u_noisy['total_ms']):.1f} ms"], ["graph", "global cost path", pct(u_graph["f1"]), f"{float(u_graph['total_ms']):.1f} ms"], ["band_graph", "corridor + curvature penalty", pct(u_band["f1"]), f"{float(u_band['total_ms']):.1f} ms"]], [0.24, 0.38, 0.18, 0.20], 12)
    add_image(s, ROOT / "results" / "u_shape_noisy" / "panel_greedy.png", 672, 184, 500, 178, caption="paper-style greedy")
    add_image(s, ROOT / "results" / "u_shape_noisy" / "panel_band_graph.png", 672, 430, 500, 178, caption="band graph variant")
    add_card(s, 80, 462, 518, 102, "Interpretation", "Band-graph improves contour quality slightly on the U-shape, but the larger improvement comes from IEPS initialization. That supports the paper's emphasis on initial edge point quality.", ACCENT)
    slides.append(s)

    s = Slide("Paper comparison caveats")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Baseline comparisons are contextual only", "Unavailable baseline source code", "The paper reports Yuen, Snake/Kass, and Chen comparisons, but it does not provide enough implementation detail for exact source-level reproduction.")
    add_table(s, 64, 190, 1152, 214, ["Paper comparison", "What this project implements", "Claim level"], [["Yuen initialization", "fixed-angle farthest-edge approximation", "context-only baseline"], ["Snake/Kass", "simple greedy Snake-style relaxation", "not full variational solver"], ["Chen tracing", "gradient-only SCF approximation", "behavioral context"], ["SNR table", "Gaussian SNR noise generated from image variance", "not pixel-identical noise"]], [0.24, 0.48, 0.28], 12)
    add_table(s, 116, 462, 1048, 104, ["Case", "Yuen-style true points", "IEPS true points", "What to say"], [["circle_noisy", "32/32", "32/32", "both succeed on current synthetic circle"], ["u_shape_noisy", "29/32", "25/32", "approximation differs from paper; report as limitation"]], [0.22, 0.22, 0.18, 0.38], 12)
    slides.append(s)

    s = Slide("Real vase")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Real vase test path", "Qualitative validation", "The project now supports a real image path with optional mask. Without the paper's original vase and exact baselines, vase metrics are proxy evidence.")
    add_image(s, ROOT / "results" / "real_vase_paper" / "panel_greedy.png", 70, 190, 548, 238, caption="real_vase_paper: generated panel")
    add_image(s, ROOT / "results" / "paper_comparisons" / "vase_proposed_scf_gradient_distance.png", 716, 202, 240, 238, caption="proposed SCF contour")
    add_image(s, ROOT / "results" / "paper_comparisons" / "vase_chen_style_gradient_only.png", 982, 202, 206, 238, caption="Chen-style approximation")
    rv = vase_rows[0] if vase_rows else {}
    add_table(s, 96, 506, 1064, 96, ["Input source", "Mask source", "Paper-mode F1", "Quantitative status"], [[rv.get("input_source", "data/vase.png"), rv.get("mask_source", "estimated_or_provided"), pct(rv.get("f1", 0.0)) if rv else "n/a", rv.get("quantitative_note", "proxy metrics")]], [0.25, 0.24, 0.16, 0.35], 12)
    slides.append(s)

    s = Slide("Conclusion")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "Final findings", "Conclusion", "The most important result is a reproducibility story with measured assumptions, not a single leaderboard number.")
    add_card(s, 76, 190, 342, 154, "Finding 1", "IEPS + greedy SCF reproduces the expected behavior on synthetic circles, including noisy cases: 32/32 IEPS points and F1 near 1.0 in the current run.", ACCENT)
    add_card(s, 470, 190, 342, 154, "Finding 2", "Concave U-shapes are the stress test. Paper-mode IEPS reaches 25/32 points and about 59% F1 on the noisy U-shape.", AMBER)
    add_card(s, 864, 190, 342, 154, "Finding 3", "The missing implementation details are a research contribution because they explain why exact reproduction is fragile.", BLUE)
    add_text(s, "Best final answer", 92, 426, 320, 30, size=23, color=INK, bold=True)
    add_text(s, "The authors gave a useful geometric method, but the paper is under-specified at the level required for executable reproduction. This project makes those assumptions explicit, validates them on author-style data, and shows a traditional-CV improvement for the U-shape failure mode.", 92, 468, 1060, 72, size=20, color=INK, bold=True)
    add_text(s, "Future work: recover exact original images, implement full Kass and Chen baselines, run tolerance sweeps, and test more non-star-convex shapes.", 92, 580, 1060, 30, size=15, color=MUTED)
    slides.append(s)

    s = Slide("References")
    add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=BG, line=BG)
    add_title(s, "IEEE-style references", "Sources", "Primary paper and baseline methods cited in the project report.")
    refs = [
        '[1] R. C. Hsu, P.-W. Kao, W.-J. Lai, and C.-T. Liu, "An initial edge point selection and segmental contour following for object contour extraction," in Proc. Int. Conf. Control, Automation, Robotics and Vision (ICARCV), 2010.',
        '[2] P. C. Yuen, G. C. Feng, and J. P. Zhou, "A contour detection method: Initialization and contour model," Pattern Recognition Letters, vol. 20, no. 2, pp. 141-148, 1999.',
        '[3] M. Kass, A. Witkin, and D. Terzopoulos, "Snakes: Active contour models," International Journal of Computer Vision, vol. 1, no. 4, pp. 321-331, 1988.',
        '[4] B. D. Chen and P. Siy, "Forward/backward contour tracing with feedback," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 9, no. 3, pp. 438-446, 1987.',
        '[5] K. S. Fu and J. K. Mui, "A survey on image segmentation," Pattern Recognition, vol. 13, pp. 3-16, 1981.',
    ]
    add_text(s, refs, 78, 190, 1090, 318, size=15, color=INK)
    add_table(s, 110, 530, 1000, 104, ["Project evidence", "Location"], [["Source implementation", "src/ieps.py, src/scf.py, src/ieps_improved.py, main.py"], ["Results used in slides", "results/tables/*.csv and results/*/*.png"]], [0.32, 0.68], 12)
    slides.append(s)

    for idx, slide in enumerate(slides, start=1):
        if idx != 1:
            add_footer(slide, idx)


def fit_rect(img_w: int, img_h: int, x: float, y: float, w: float, h: float, fit: str = "contain") -> Tuple[float, float, float, float]:
    if img_w <= 0 or img_h <= 0:
        return x, y, w, h
    scale = min(w / img_w, h / img_h) if fit == "contain" else max(w / img_w, h / img_h)
    nw, nh = img_w * scale, img_h * scale
    return x + (w - nw) / 2, y + (h - nh) / 2, nw, nh


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    lines: List[str] = []
    for raw in str(text).split("\n"):
        words = raw.split(" ")
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            trial = word if not current else current + " " + word
            if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: Any,
    box: Tuple[float, float, float, float],
    size_pt: int,
    color: str,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
) -> None:
    x, y, w, h = box
    fnt = font(max(8, int(size_pt * 1.25)), bold)
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    lines = wrap_lines(draw, str(text), fnt, max(10, w))
    line_h = int(size_pt * 1.55)
    total_h = len(lines) * line_h
    yy = y
    if valign == "middle":
        yy = y + max(0, (h - total_h) / 2)
    elif valign == "bottom":
        yy = y + max(0, h - total_h)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        tx = x
        if align == "center":
            tx = x + max(0, (w - (bbox[2] - bbox[0])) / 2)
        elif align == "right":
            tx = x + max(0, w - (bbox[2] - bbox[0]))
        draw.text((tx, yy), line, fill=hex_to_rgb(color), font=fnt)
        yy += line_h
        if yy > y + h + line_h:
            break


def render_slide(slide: Slide, idx: int) -> Path:
    image = Image.new("RGB", (SLIDE_W, SLIDE_H), hex_to_rgb(BG))
    draw = ImageDraw.Draw(image)
    for element in slide.elements:
        x, y, w, h = element.x, element.y, element.w, element.h
        props = element.props
        if element.kind == "rect":
            draw.rectangle([x, y, x + w, y + h], fill=hex_to_rgb(props.get("fill", PAPER)), outline=hex_to_rgb(props.get("line", LINE)))
        elif element.kind == "line":
            draw.line([x, y, x + w, y + h], fill=hex_to_rgb(props.get("fill", LINE)), width=max(1, int(h) if h > 1 else 1))
        elif element.kind == "arrow":
            color = hex_to_rgb(props.get("fill", ACCENT))
            draw.rectangle([x, y + h * 0.35, x + w * 0.7, y + h * 0.65], fill=color)
            draw.polygon([(x + w * 0.7, y), (x + w, y + h / 2), (x + w * 0.7, y + h)], fill=color)
        elif element.kind == "text":
            if props.get("fill"):
                draw.rectangle([x, y, x + w, y + h], fill=hex_to_rgb(props["fill"]), outline=hex_to_rgb(props.get("line", props["fill"])))
            draw_text_box(draw, props.get("text", ""), (x, y, w, h), int(props.get("size", 18)), props.get("color", INK), bool(props.get("bold", False)), props.get("align", "left"), props.get("valign", "top"))
        elif element.kind == "image":
            try:
                img = Image.open(props["path"]).convert("RGB")
                ix, iy, iw, ih = fit_rect(img.width, img.height, x, y, w, h, props.get("fit", "contain"))
                img = img.resize((max(1, int(iw)), max(1, int(ih))))
                draw.rectangle([x, y, x + w, y + h], fill=hex_to_rgb("FFFFFF"), outline=hex_to_rgb(LINE))
                image.paste(img, (int(ix), int(iy)))
            except OSError as exc:
                draw.rectangle([x, y, x + w, y + h], fill=hex_to_rgb("FEE2E2"), outline=hex_to_rgb(RED))
                draw_text_box(draw, f"Missing image: {exc}", (x + 8, y + 8, w - 16, h - 16), 12, RED)
    path = PREVIEW_DIR / f"slide_{idx:02d}.png"
    image.save(path)
    return path


def render_previews() -> Path:
    rendered = [render_slide(slide, i) for i, slide in enumerate(slides, start=1)]
    thumb_w, thumb_h = 320, 180
    cols = 4
    rows_n = math.ceil(len(rendered) / cols)
    montage = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + 24)), "white")
    draw = ImageDraw.Draw(montage)
    for i, path in enumerate(rendered):
        img = Image.open(path).resize((thumb_w, thumb_h))
        x = (i % cols) * thumb_w
        y = (i // cols) * (thumb_h + 24)
        montage.paste(img, (x, y))
        draw.text((x + 6, y + thumb_h + 4), f"Slide {i + 1}", fill=hex_to_rgb(INK), font=font(12, True))
    montage_path = PREVIEW_DIR / "montage.png"
    montage.save(montage_path)
    return montage_path


def solid_fill(color: Optional[str]) -> str:
    if not color or color == "none":
        return "<a:noFill/>"
    return f'<a:solidFill><a:srgbClr val="{color.replace("#", "")}"/></a:solidFill>'


def line_xml(color: Optional[str], width_px: float = 1) -> str:
    if not color or color == "none":
        return "<a:ln><a:noFill/></a:ln>"
    return f'<a:ln w="{max(1, int(width_px * EMU))}"><a:solidFill><a:srgbClr val="{color.replace("#", "")}"/></a:solidFill></a:ln>'


def text_body(text: Any, size_pt: int, color: str, bold: bool = False, align: str = "left", valign: str = "top") -> str:
    paragraphs = [str(item) for item in text] if isinstance(text, list) else str(text).split("\n")
    algn = {"left": "l", "center": "ctr", "right": "r"}.get(align, "l")
    anchor = {"top": "t", "middle": "ctr", "bottom": "b"}.get(valign, "t")
    body = [f'<a:bodyPr wrap="square" anchor="{anchor}" lIns="0" tIns="0" rIns="0" bIns="0"><a:spAutoFit/></a:bodyPr><a:lstStyle/>']
    bold_attr = ' b="1"' if bold else ""
    for para in paragraphs:
        body.append(
            f'<a:p><a:pPr algn="{algn}"/><a:r><a:rPr lang="en-US" sz="{int(size_pt * 100)}"{bold_attr} dirty="0"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Aptos"/><a:cs typeface="Aptos"/></a:rPr><a:t>{xesc(para)}</a:t></a:r></a:p>'
        )
    return f'<p:txBody>{"".join(body)}</p:txBody>'


def sp_xml(shape_id: int, element: Element) -> str:
    props = element.props
    tx = element.kind == "text"
    geom = "rightArrow" if element.kind == "arrow" else "rect"
    fill = props.get("fill", "none" if tx and not props.get("fill") else PAPER)
    line = props.get("line", "none" if tx and not props.get("line") else LINE)
    text_xml = ""
    if tx:
        text_xml = text_body(props.get("text", ""), int(props.get("size", 18)), props.get("color", INK), bool(props.get("bold", False)), props.get("align", "left"), props.get("valign", "top"))
    return f"""<p:sp>
<p:nvSpPr><p:cNvPr id="{shape_id}" name="{xesc(props.get("name", element.kind))}"/><p:cNvSpPr{" txBox=\"1\"" if tx else ""}/><p:nvPr/></p:nvSpPr>
<p:spPr><a:xfrm><a:off x="{emu(element.x)}" y="{emu(element.y)}"/><a:ext cx="{emu(element.w)}" cy="{emu(element.h)}"/></a:xfrm><a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>{solid_fill(fill)}{line_xml(line)}</p:spPr>
{text_xml}
</p:sp>"""


def image_xml(shape_id: int, element: Element, rid: str) -> str:
    props = element.props
    with Image.open(props["path"]) as img:
        ix, iy, iw, ih = fit_rect(img.width, img.height, element.x, element.y, element.w, element.h, props.get("fit", "contain"))
    return f"""<p:pic>
<p:nvPicPr><p:cNvPr id="{shape_id}" name="image" descr="{xesc(Path(props["path"]).name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{emu(ix)}" y="{emu(iy)}"/><a:ext cx="{emu(iw)}" cy="{emu(ih)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>"""


media_entries: List[Tuple[str, Path]] = []


def slide_xml(slide: Slide) -> Tuple[str, str]:
    shape_id = 2
    rels: List[str] = []
    sp_tree = [
        '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
    ]
    for element in slide.elements:
        if element.kind in {"rect", "text", "arrow"}:
            sp_tree.append(sp_xml(shape_id, element))
            shape_id += 1
        elif element.kind == "line":
            line_element = Element("rect", element.x, element.y, max(1, element.w), max(1, element.h), {"fill": element.props.get("fill", LINE), "line": element.props.get("fill", LINE)})
            sp_tree.append(sp_xml(shape_id, line_element))
            shape_id += 1
        elif element.kind == "image":
            media_name = f"image{len(media_entries) + 1}.png"
            media_entries.append((media_name, element.props["path"]))
            rid = f"rId{len(rels) + 1}"
            rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_name}"/>')
            sp_tree.append(image_xml(shape_id, element, rid))
            shape_id += 1
    xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree>{"".join(sp_tree)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'''
    rel_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(rels)}</Relationships>'''
    return xml, rel_xml


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_pptx() -> None:
    ppt = WORK / "ppt"
    (ppt / "slides" / "_rels").mkdir(parents=True)
    (ppt / "slideMasters" / "_rels").mkdir(parents=True)
    (ppt / "slideLayouts" / "_rels").mkdir(parents=True)
    (ppt / "theme").mkdir(parents=True)
    (ppt / "media").mkdir(parents=True)
    (WORK / "_rels").mkdir(parents=True)
    (WORK / "docProps").mkdir(parents=True)

    for i, slide in enumerate(slides, start=1):
        sx, sr = slide_xml(slide)
        write_text(ppt / "slides" / f"slide{i}.xml", sx)
        write_text(ppt / "slides" / "_rels" / f"slide{i}.xml.rels", sr)

    for media_name, src in media_entries:
        shutil.copyfile(src, ppt / "media" / media_name)

    slide_overrides = "".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, len(slides) + 1)
    )
    write_text(
        WORK / "[Content_Types].xml",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/><Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/><Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>{slide_overrides}<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>''',
    )
    write_text(
        WORK / "_rels" / ".rels",
        '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>''',
    )

    sld_ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{1 + i}"/>' for i in range(1, len(slides) + 1))
    pres_rels = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    pres_rels.extend(
        f'<Relationship Id="rId{1 + i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, len(slides) + 1)
    )
    write_text(
        ppt / "presentation.xml",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{emu(SLIDE_W)}" cy="{emu(SLIDE_H)}" type="wide"/><p:notesSz cx="6858000" cy="9144000"/><p:defaultTextStyle><a:defPPr><a:defRPr lang="en-US"/></a:defPPr></p:defaultTextStyle></p:presentation>''',
    )
    write_text(
        ppt / "_rels" / "presentation.xml.rels",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(pres_rels)}</Relationships>''',
    )

    write_text(
        ppt / "slideMasters" / "slideMaster1.xml",
        '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>''',
    )
    write_text(
        ppt / "slideMasters" / "_rels" / "slideMaster1.xml.rels",
        '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>''',
    )
    write_text(
        ppt / "slideLayouts" / "slideLayout1.xml",
        '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>''',
    )
    write_text(
        ppt / "slideLayouts" / "_rels" / "slideLayout1.xml.rels",
        '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>''',
    )
    write_text(
        ppt / "theme" / "theme1.xml",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="IEPS SCF Theme"><a:themeElements><a:clrScheme name="Custom"><a:dk1><a:srgbClr val="{INK}"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="{DARK}"/></a:dk2><a:lt2><a:srgbClr val="{BG}"/></a:lt2><a:accent1><a:srgbClr val="{ACCENT}"/></a:accent1><a:accent2><a:srgbClr val="{BLUE}"/></a:accent2><a:accent3><a:srgbClr val="{AMBER}"/></a:accent3><a:accent4><a:srgbClr val="{ACCENT2}"/></a:accent4><a:accent5><a:srgbClr val="{GREEN}"/></a:accent5><a:accent6><a:srgbClr val="{RED}"/></a:accent6><a:hlink><a:srgbClr val="0563C1"/></a:hlink><a:folHlink><a:srgbClr val="954F72"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/><a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/><a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme><a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>''',
    )

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    write_text(
        WORK / "docProps" / "core.xml",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>IEPS + SCF Final Presentation</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>''',
    )
    write_text(
        WORK / "docProps" / "app.xml",
        f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex OpenXML</Application><PresentationFormat>On-screen Show (16:9)</PresentationFormat><Slides>{len(slides)}</Slides><Notes>0</Notes><HiddenSlides>0</HiddenSlides><ScaleCrop>false</ScaleCrop><Company></Company><AppVersion>16.0000</AppVersion></Properties>''',
    )

    if PPTX_PATH.exists():
        PPTX_PATH.unlink()
    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(WORK.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(WORK).as_posix())

    with zipfile.ZipFile(PPTX_PATH, "r") as archive:
        names = set(archive.namelist())
        assert "ppt/presentation.xml" in names
        assert "[Content_Types].xml" in names
        assert len([name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml")]) == len(slides)


def main() -> None:
    build_slides()
    montage = render_previews()
    write_pptx()
    print(f"PPTX={PPTX_PATH}")
    print(f"MONTAGE={montage}")
    print(f"SLIDES={len(slides)}")
    print(f"MEDIA={len(media_entries)}")


if __name__ == "__main__":
    main()
