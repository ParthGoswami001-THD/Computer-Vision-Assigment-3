# Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study

**Course:** Computer Vision Assignment 3  
**Paper:** Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu, *An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction*, ICARCV 2010.

## Research Question

> Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying which missing implementation parameters are necessary to reproduce the reported behavior?

This question became the center of the project after the professor discussion. Instead of making Canny, preprocessing, or deep learning the main story, I focused on whether the authors' own method can be reproduced carefully from the paper.

## What The Paper Is Trying To Solve

The paper deals with object contour extraction. A simple edge detector such as Sobel can find strong intensity changes, but it does not automatically give a clean closed contour. Active contour methods can close contours, but they often need manual initialization.

Hsu et al. try to solve this by combining two steps:

1. **IEPS: Initial Edge Point Selection**  
   Choose a small set of useful boundary points automatically.

2. **SCF: Segmental Contour Following**  
   Connect those points segment by segment until a full contour is formed.

The idea is attractive because it keeps the method classical and lightweight. The hard part is that the paper leaves some practical implementation details open.

## Reimplemented Pipeline

The implemented pipeline is:

```text
input image
-> Sobel gradient magnitude
-> center of gravity
-> IEPS scan-line point selection
-> SCF segment tracing
-> predicted contour
-> evaluation against ground truth
```

The default paper-style configuration is:

| Component | Default used here |
|---|---|
| Initial scan lines | 4 |
| IEPS refinement iterations | 3 |
| Sobel threshold | 64 |
| Final IEPS point count | about 32 |
| SCF stop tolerance | 2 pixels |
| SCF score | gradient / distance-to-target squared |

The main implementation is in `src/ieps.py` and `src/scf.py`. The improved IEPS extension is kept separate in `src/ieps_improved.py` so the paper-faithful version is still easy to identify.

## Validation Data

I used synthetic images similar to the author-style examples:

- clean circle,
- noisy circle,
- clean U-shape,
- noisy U-shape.

Each case has an input image, a binary object mask, and a ground-truth contour mask. Gaussian noise is generated with fixed seeds so the results are reproducible.

## Evaluation Metrics

The method is evaluated at two levels:

| Metric | Why it is needed |
|---|---|
| IEPS point accuracy | Checks whether the selected initial points are near the true contour. |
| Neighbor distance mean/std | Shows whether IEPS points are evenly spaced. |
| Contour precision | Measures how much of the predicted contour lies near ground truth. |
| Contour recall | Measures how much of the ground-truth contour is recovered. |
| F1 score | Summarizes precision and recall. |
| Runtime | Checks the practical cost of IEPS and SCF. |

Both IEPS points and contour pixels are evaluated with a small tolerance. This is important because gradient responses and contour masks do not always land on exactly the same pixel.

## Main Reproducibility Finding

The paper's overall idea is clear, but the implementation is not fully specified. During reimplementation, I had to make several choices that directly changed the results.

| Under-specified detail | Why it matters | Decision used in this implementation |
|---|---|---|
| Scan-line sampling | Pixel discretization changes which edge point is selected. | Integer rays from the center and finite normal lines through midpoints. |
| Threshold tuning | The paper mentions 64, but not how to adapt it. | Use 64 as default and test 40, 64, 90. |
| Gaussian noise | Noise results depend on sigma and random seed. | Fixed seeds, with clean, low-noise, and paper-like noisy tests. |
| True-positive tolerance | Pixel-perfect evaluation is too strict for edge bands. | Default tolerance is 2 pixels. |
| SCF stopping rule | "Reach the next point" needs a code-level definition. | Stop when distance to target is within tolerance. |
| Candidate tie-breaking | Several neighboring pixels can have similar scores. | Prefer score, then gradient, then target distance. |
| Loop handling | Greedy following can revisit pixels. | Keep a visited set for each segment. |
| Missing scan-line edge | Some lines have no pixel above threshold. | Use a configurable fallback; default is max-gradient point. |
| Coordinate convention | Paper uses `(x, y)`; OpenCV arrays use `[y, x]`. | Public geometry uses `(x, y)` and converts only when indexing arrays. |
| Normal-line discretization | The paper gives the math, not exact pixels. | Sample a finite integer line with decreasing radius each iteration. |

The SCF part was the most sensitive. Without explicit stopping, loop prevention, tie-breaking, and fallback rules, the contour follower is easy to make unstable.

## Results Summary

The current full run writes detailed values to `results/tables/main_results.csv`. The high-level behavior is:

- Circle cases are handled very well by the paper-style method.
- U-shape cases are harder because concavity breaks some assumptions behind center-based scan lines and local contour following.
- The Canny baseline is useful as a reference, but it is not the research focus.

Representative values from the current run:

| Case | Method | F1 |
|---|---|---|
| circle_clean | paper IEPS + greedy SCF | 1.0000 |
| circle_noisy | paper IEPS + greedy SCF | 1.0000 |
| u_shape_clean | paper IEPS + greedy SCF | 0.6135 |
| u_shape_noisy | paper IEPS + greedy SCF | 0.5908 |
| u_shape_noisy | graph SCF | 0.6114 |
| u_shape_noisy | band-graph SCF | 0.6122 |

The improved IEPS extension gives the clearest practical improvement on concave shapes:

| Case | Paper IEPS F1 | Improved IEPS F1 |
|---|---:|---:|
| circle_clean | 1.0000 | 1.0000 |
| circle_noisy | 1.0000 | 1.0000 |
| u_shape_clean | 0.6135 | 0.7004 |
| u_shape_noisy | 0.5908 | 0.6965 |

This supports the main interpretation: the original IEPS idea is strong on simple star-convex shapes, but concave shapes need more careful seeding and coverage.

## Parameter Study

The parameter study stays inside IEPS and SCF rather than adding unrelated algorithms.

| Parameter | Values tested |
|---|---|
| Sobel threshold | 40, 64, 90 |
| Initial scan lines | 4, 8 |
| IEPS iterations | 2, 3 |
| SCF stopping tolerance | 1, 2, 3 pixels |
| SCF score | gradient only, gradient / distance^2 |
| Noise level | clean, low noise, paper-like noisy |

This is important because it turns the paper's missing details into measurable choices instead of hidden assumptions.

## Extension Kept Inside The Paper Direction

I added two traditional computer vision extensions, but kept them secondary to the paper reproduction.

**Improved IEPS**  
The paper assumes that the center of gravity is a good place to emit scan lines. This can fail on concave U-shapes because the center may fall in the background notch. The improved version checks an Otsu silhouette and, if needed, moves the seed to the maximum of the distance transform. It also uses denser rays and fills large angular gaps.

**Band-limited graph SCF**  
The paper-style greedy SCF is fast but local. The graph version treats the Sobel image as a cost map. The band-limited version adds a corridor around the current IEPS segment and a curvature penalty, which helps reduce wandering and zig-zagging. It is slower, so I would present it as an extension rather than the default method.

## Limitations

The current implementation is intentionally small and controlled. It assumes one dominant object, strong object/background contrast, and simple synthetic shapes. The U-shape already shows that concave objects are much harder than circles. More realistic images would need more robust thresholding, ordering, and stopping checks.

## Final Contribution

The project reimplements the IEPS + SCF method and validates it on clean/noisy circle and U-shape cases. The main finding is that the method is reproducible on simple shapes, but exact behavior depends on implementation details that the paper does not fully specify.

The strongest research point is this:

> During reimplementation, I found that some practical details are under-specified, so I documented my assumptions and tested their effect using parameter runs.

That keeps the project aligned with the authors' method while still adding a clear research contribution.
