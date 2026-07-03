# IEPS + SCF Object Contour Extraction

## Computer Vision Assignment 3

This repository contains a traditional computer vision reimplementation of the paper:

**An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction**  
Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu, ICARCV 2010.

The project follows the professor-aligned direction:

> Reimplement and validate the authors' IEPS + SCF method first. Extensions must stay inside the authors' traditional computer vision direction.

---

## 1. Project Goal

The goal is to reproduce the paper's two-stage contour extraction method:

```text
Input grayscale image
-> Sobel gradient magnitude
-> Center of gravity
-> Initial Edge Point Selection (IEPS)
-> Segmental Contour Following (SCF)
-> Closed object contour
-> Validation
```

The project also documents bugs, missing implementation details, and traditional-CV improvements discovered during reimplementation.

---

## 2. What Is Implemented

### Main paper implementation

- Synthetic circle and U-shape images
- Gaussian noisy test images
- Sobel gradient magnitude
- Center of gravity using spatial moments
- IEPS with scan lines and iterative refinement
- SCF with related direction and 3-candidate operating mask
- Gravity-inspired score: `gradient / distance^2`
- Contour validation against ground truth
- Runtime measurement

### Debugging and reproducibility study

- Center-of-gravity modes: `raw`, `contrast`, `binary`
- IEPS order modes: `topological`, `angle`
- IEPS refinement selection modes: `farthest_from_center`, `max_gradient`, `closest_to_reference`
- Optional improved IEPS mode with robust interior seed and denser scan lines
- SCF score modes: `gradient_distance2`, `gradient_only`
- Optional graph-search SCF as a traditional-CV improvement
- Optional band-limited curvature-aware graph SCF

### Secondary baseline

- Canny + OpenCV contour extraction baseline

This baseline is included only for reference. It is not the main contribution.

---

## 3. Repository Structure

```text
cv_assignment3_ieps_scf/
|
+-- main.py
+-- requirements.txt
+-- Doxyfile
+-- README.md
|
+-- src/
|   +-- image_generation.py
|   +-- gradients.py
|   +-- geometry.py
|   +-- ieps.py
|   +-- ieps_improved.py
|   +-- scf.py
|   +-- baseline.py
|   +-- evaluation.py
|   +-- visualization.py
|
+-- docs/
|   +-- FINAL_PLAN.md
|   +-- RESEARCH_REPORT.md
|   +-- FULL_IMPLEMENTATION_BUG_FIX_REPORT.md
|   +-- DOXYGEN_STYLE.md
|
+-- results/
    +-- circle_clean/
    +-- circle_noisy/
    +-- u_shape_clean/
    +-- u_shape_noisy/
    +-- tables/
```

---

## 4. Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## 5. Run the Full Project

From the repository root:

```bash
python main.py
```

The script will:

1. Generate author-style test images.
2. Run IEPS.
3. Run paper-style greedy SCF.
4. Run graph-search SCF as a traditional-CV improvement.
5. Run band-limited curvature-aware graph SCF as a stronger traditional-CV improvement.
6. Run Canny baseline.
7. Run bug/fix study.
8. Run improved-IEPS comparison.
9. Run parameter study.
10. Save images and CSV tables.

---

## 6. Output Files

Main result images:

```text
results/circle_clean/panel_greedy.png
results/circle_noisy/panel_greedy.png
results/u_shape_clean/panel_greedy.png
results/u_shape_noisy/panel_greedy.png
```

Improved graph-SCF panels:

```text
results/<case>/panel_graph.png
results/<case>/panel_band_graph.png
```

CSV tables:

```text
results/tables/main_results.csv
results/tables/bug_fix_study.csv
results/tables/parameter_study.csv
results/tables/improvement_comparison.csv
```

---

## 7. Important Reproducibility Finding

The most important under-specified part of the conference paper is the practical SCF implementation.

The paper explains:

- related direction,
- operating masks,
- gravity-like force,
- closed contour decision,

but it does not fully specify:

- stopping tolerance,
- loop prevention,
- candidate tie-breaking,
- weak-gradient fallback,
- maximum segment steps,
- border handling,
- contour-point ordering for concave shapes.

This implementation makes those rules explicit.

---

## 8. Main Bug Fixes

### Fix 1: Center-of-gravity bias

A non-zero background biases raw image moments. The implementation now uses black/white default synthetic images and supports contrast-based moments.

Recommended:

```text
center_mode = contrast
```

### Fix 2: Ground-truth contour definition

The ground-truth contour is generated using morphological gradient:

```text
dilate(mask) - erode(mask)
```

This better matches gradient-based contour evidence.

### Fix 3: Concave-shape ordering ambiguity

For concave shapes, polar-angle ordering can connect contour points incorrectly. The implementation supports topological insertion order.

Recommended:

```text
order_mode = topological
```

### Fix 4: SCF missing engineering rules

SCF now includes:

```text
stop_tolerance = 2 pixels
visited set for loop prevention
tie-breaking by score, gradient, and distance
weak-gradient fallback
max-step limit
border checks
```

---

## 9. Traditional CV Improvements Included

The project keeps the paper-style greedy SCF and paper IEPS as the main method. It also includes traditional-CV improvements:

```text
improved IEPS with robust interior seed and denser coverage
gradient-weighted graph-search SCF
band-limited curvature-aware graph SCF
```

The improved IEPS mode addresses concave-shape failures by checking whether the center of gravity lies inside an Otsu object silhouette. If it falls in the background notch, it relocates the seed to the distance-transform maximum, then uses denser scan-line coverage and boundary-style point ordering.

The graph-search SCF treats the Sobel gradient image as a cost map and finds a stronger edge path between neighboring IEPS points.

The band-limited curvature-aware graph SCF adds two extra traditional-CV constraints: it prefers paths near the current IEPS segment and penalizes sharp direction changes. This reduces wandering and zig-zagging while still following strong Sobel edges. These extensions are traditional computer vision, not deep learning.

---

## 10. Generate Doxygen Documentation

The code uses Doxygen-style docstrings with:

```text
@file
@brief
@param
@return
```

If Doxygen is installed:

```bash
doxygen Doxyfile
```

HTML documentation will be generated under:

```text
docs/doxygen/html/
```

---

## 11. Recommended Presentation Message

> I implemented the authors' IEPS + SCF method as a traditional computer vision pipeline. During reproduction, the main challenge was not Sobel or center of gravity, but the missing practical details in SCF: stopping tolerance, loop prevention, tie-breaking, weak-gradient fallback, and contour-point ordering. These choices strongly affect contour closure, especially for concave U-shapes. I also tested traditional CV fixes such as contrast-based moments, robust interior IEPS seeding, denser scan-line coverage, graph-search contour following, and band-limited curvature-aware SCF.

---

## 12. Limitations

- The method assumes one dominant object.
- It works best when object and background have clear contrast.
- Concave objects are more difficult than circles.
- Heavy noise can create false gradient responses.
- SCF is sensitive to point ordering and stopping criteria.
- The graph-search and band-graph improvements are more robust but slower.

---

## 13. Future Work

Traditional CV future work:

1. Adaptive Sobel thresholding.
2. Multi-scale Sobel gradient.
3. Hybrid IEPS candidate selection.
4. Curvature-aware SCF scoring.
5. Graph-based contour ordering.
6. Full Snake comparison using IEPS initialization.
7. Full Chen method comparison.

Deep learning such as U-Net should only be mentioned as future work, not as part of this assignment implementation.
