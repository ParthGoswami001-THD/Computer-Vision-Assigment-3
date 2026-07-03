# Research Report
## Reimplementation and Validation of IEPS + SCF for Object Contour Extraction

**Course:** Computer Vision Assignment 3  
**Assigned paper:** Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu, *An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction*, IEEE ICARCV 2010.

---

## 1. Assignment Alignment

The assignment requires an individual research question based on a scientific workshop or conference paper. The main goal is to understand and reimplement the paper's main scientific contribution, validate its results, and potentially propose a focused extension. The implementation should use Python with NumPy and OpenCV, with hand-coded algorithmic parts preferred.

This project therefore focuses on a traditional computer vision reimplementation of the paper's proposed method:

1. **Initial Edge Point Selection (IEPS)**
2. **Segmental Contour Following (SCF)**

The project does not treat deep learning, full Snake implementation, full Chen implementation, or general segmentation as the main contribution.

---

## 2. Original Paper Problem

Contour extraction is a classical image segmentation problem. The objective is to identify the boundary of an object in an image. Edge-based operators such as Sobel are efficient but do not guarantee a closed contour. Active contour models such as Snake can generate closed contours, but they usually require manual initialization and higher computation.

The paper's research question can be stated as:

> How can an object contour be extracted automatically and efficiently without manually selecting initial contour points, while still producing an accurate closed contour?

---

## 3. Main Contribution of the Paper

The paper proposes a two-stage traditional computer vision method.

### 3.1 Initial Edge Point Selection (IEPS)

IEPS automatically selects initial edge points around an object.

The method uses:

- grayscale image intensity,
- image moments / center of gravity,
- radial scan lines,
- Sobel gradient magnitude,
- threshold-based edge candidate selection,
- iterative scan-line refinement between neighboring points.

Author-style experimental parameters reported in the paper:

- 4 initial scan lines,
- 3 refinement iterations,
- approximately 32 initial edge points,
- Sobel threshold around 64.

### 3.2 Segmental Contour Following (SCF)

SCF connects neighboring IEPS points into a closed contour.

For each pair of neighboring initial points, SCF:

1. treats one point as the current origin and the next as the target,
2. computes the related direction,
3. selects a local directional mask,
4. evaluates candidate pixels using gradient magnitude and a gravity-like distance term,
5. moves to the best candidate,
6. repeats until the target is reached,
7. links all segments to form the object contour.

---

## 4. Final Research Question

> Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying the missing implementation details needed to reproduce the contour-following behavior?

---

## 5. Research Direction After Professor Discussion

The final direction follows the authors' particular approach rather than adding unrelated preprocessing or modern deep learning. The priority is:

1. Reimplement IEPS + SCF.
2. Validate author-style results on synthetic clean/noisy images.
3. Investigate missing implementation details inside IEPS/SCF.
4. Run a focused parameter study.
5. Keep Canny/OpenCV as a small secondary baseline only.
6. Mention U-Net only as future work.

---

## 6. Key Reproducibility Finding

The most important under-specified part of the paper is the practical SCF contour-following logic.

The paper describes the general idea of related direction, local operating masks, and gravity-like force, but it does not fully define several details needed for reliable code:

| Under-specified SCF detail | Implementation decision in this project |
|---|---|
| When does a segment stop? | Stop when distance to target is less than or equal to a tolerance. Default: 2 pixels. |
| How to avoid loops? | Maintain a visited-pixel set per segment. |
| What if no candidate has strong gradient? | Use closest-to-target fallback. |
| How are ties handled? | Prefer highest score, then highest gradient, then shortest distance to target. |
| How many steps are allowed? | Maximum steps = 3 times the distance between segment endpoints. |
| How are image borders handled? | Ignore out-of-image candidates. |
| What defines a closed contour? | All neighboring IEPS points are connected, including the last point back to the first. |

This is the main reproducibility contribution of this implementation.

---

## 7. Validation Design

### 7.1 Author-Style Test Images

The project validates the method on:

- clean circle,
- noisy circle,
- clean U-shape,
- noisy U-shape.

Each synthetic image has:

- input grayscale image,
- binary object mask,
- ground-truth contour mask.

### 7.2 Main Metrics

| Metric | Purpose |
|---|---|
| IEPS point accuracy | Measures whether selected initial edge points lie near the true contour. |
| Neighbor mean distance | Measures average spacing between IEPS points. |
| Neighbor distance standard deviation | Measures evenness of IEPS point distribution. |
| Contour precision | Measures how many predicted contour pixels are correct. |
| Contour recall | Measures how much true contour is recovered. |
| F1 score | Balances precision and recall. |
| Runtime | Validates the low-complexity claim. |

### 7.3 Tolerance-Based Evaluation

A predicted point is considered correct if it lies within a small pixel tolerance of the ground-truth contour. Default tolerance: 2 pixels.

This is necessary because the paper reports true-positive ratios but does not fully specify the exact pixel tolerance.

---

## 8. Focused Parameter Study

The parameter study stays inside the authors' method.

| Parameter | Values |
|---|---|
| Sobel threshold | 40, 64, 90 |
| Initial scan lines | 4, 8 |
| IEPS iterations | 2, 3 |
| SCF stopping tolerance | 1, 2, 3 pixels |
| SCF score | gradient only vs gradient / distance^2 |

The aim is not to create a new algorithm, but to show which missing or weakly specified details affect reproducibility.

---

## 9. Comparison Strategy

### Main Method

- IEPS + SCF

### Traditional-CV Extension

- Improved IEPS + SCF
- Band-limited curvature-aware graph SCF

The improved IEPS mode keeps the authors' initialization direction but replaces a fragile center assumption with a robust interior seed. If the center of gravity lies outside the Otsu object silhouette, the seed is relocated to the distance-transform maximum. The method also uses denser scan-line coverage and boundary-style point ordering for concave objects.

The band-limited curvature-aware graph SCF keeps the SCF segment idea but replaces local greedy movement with A* search over an edge-cost map. The cost combines weak-gradient penalty, distance from the IEPS segment corridor, and curvature penalty. This is used as an improved traditional-CV contour-following method, not as the paper-faithful baseline.

### Secondary Baseline

- Canny + OpenCV `findContours`

The baseline is included only as a reference. It is not the main contribution.

### Optional Future Work

- IEPS + Snake comparison,
- full Chen method comparison,
- U-Net-based segmentation comparison,
- multiple-object extension,
- larger real image dataset.

---

## 10. Expected Results

Expected successful behavior:

1. IEPS should select points close to the boundary for centered circle and U-shape objects.
2. 4 scan lines and 3 iterations should create approximately 32 edge points.
3. SCF should connect neighboring IEPS points into a mostly closed contour.
4. Noise should reduce accuracy but not necessarily destroy the method.
5. SCF stopping tolerance and tie-breaking should strongly affect contour closure.

Expected limitations:

1. Heavy noise may create false Sobel gradients.
2. Weak object/background contrast may reduce IEPS reliability.
3. Multiple objects are not handled.
4. Off-center objects may affect center-of-gravity based scan lines.
5. SCF depends on implementation details not fully specified in the paper.

---

## 11. Final Contribution Statement

This project contributes a reproducible traditional computer vision implementation of IEPS + SCF. It follows the authors' direction and documents the practical SCF rules that are necessary to make the algorithm executable: stopping tolerance, loop prevention, candidate tie-breaking, weak-gradient fallback, and max-step handling.

It also identifies a deeper IEPS limitation: the paper's IEPS assumes the center of gravity lies inside a star-convex object. This fails on concave U-shapes. A traditional CV fix using an interior distance-transform seed and denser scan-line coverage improves U-shape performance while preserving circle performance.

For SCF, the project adds a second extension: band-limited curvature-aware graph search. This addresses the paper-style greedy mask's tendency to make local mistakes, wander on noisy gradients, or zig-zag around corners.
