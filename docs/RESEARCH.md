# Computer Vision Assignment 3

## Research Direction: Reimplementation and Validation of IEPS + SCF

## 1. Project Title

**Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study**

## 2. Assigned Paper

**Title:** An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction
**Authors:** Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu
**Conference:** IEEE ICARCV 2010
**Topic:** Classical object contour extraction using automatic initial edge point selection and segmental contour following.

## 3. Main Research Direction

This project must follow the authors' traditional computer vision direction.

The goal is **not** to create a new general segmentation system.
The goal is to **reimplement and validate the authors' proposed algorithm** as closely as possible from the paper description.

The project focuses on:

1. Traditional computer vision.
2. Sobel gradient magnitude.
3. Image moments and center of gravity.
4. Scan-line-based initial edge point selection.
5. Segmental contour following.
6. Reproducibility analysis.
7. Parameter sensitivity inside the authors' method.

Deep learning, U-Net, full Snake, and full Chen method are not part of the main implementation.

## 4. Original Paper Research Question

The original paper addresses this question:

**How can an object contour be extracted automatically and efficiently without manually selecting initial contour points, while still producing an accurate closed contour?**

The paper tries to solve the limitations of:

1. Simple edge detectors, which are fast but may not produce closed contours.
2. Snake or active contour models, which can produce closed contours but often require manual initialization.
3. Earlier contour tracing methods, which may be affected by noise or incomplete direction information.

## 5. Main Contribution of the Paper

The paper proposes a two-stage method:

### 5.1 Initial Edge Point Selection — IEPS

IEPS automatically finds initial edge points around the object.

Main idea:

1. Compute the center of gravity of the image/object.
2. Emit scan lines from the center of gravity.
3. Compute Sobel gradient magnitude.
4. Select strong edge pixels along scan lines.
5. Iteratively generate new scan lines between neighboring selected points.
6. Obtain a set of initial edge points around the object contour.

Author-style parameters:

- Initial scan lines: 4
- Refinement iterations: 3
- Expected final edge points: 32
- Sobel threshold: approximately 64

### 5.2 Segmental Contour Following — SCF

SCF connects neighboring IEPS points to form a closed contour.

Main idea:

1. Take two neighboring IEPS points.
2. Use one as origin and the next as target.
3. Compute related direction from origin to target.
4. Use a local operating mask around the current point.
5. Select the next contour point using a gravity-like score based on gradient magnitude and distance to the target.
6. Continue until the target point is reached.
7. Repeat for all neighboring point pairs.
8. Connect all segments to form a closed contour.

## 6. Final Research Question for This Assignment

**Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying the missing implementation details needed to reproduce the contour extraction behavior?**

## 7. Important Research Finding to Investigate

The most important under-specified part of the paper is the **practical SCF contour-following logic**.

The paper explains the concept of:

- related direction,
- local masks,
- gravity-like score,
- closed contour decision,

but does not fully specify all practical implementation rules.

The implementation must explicitly implement and document:

1. Segment stopping condition.
2. Distance tolerance to target point.
3. Candidate tie-breaking.
4. Loop prevention.
5. Handling weak-gradient candidate pixels.
6. Maximum step count per segment.
7. Border handling.
8. Definition of closed contour.
9. Candidate mask offsets in image coordinates.

This is the key reproducibility contribution of the project.

## 8. Research Scope

### 8.1 Must Do

The project must implement and validate:

1. Synthetic circle image.
2. Synthetic U-shape image.
3. Clean and noisy versions.
4. Sobel gradient magnitude.
5. Center of gravity.
6. IEPS initial edge point selection.
7. IEPS iterative refinement.
8. SCF segmental contour following.
9. Ground-truth contour generation.
10. IEPS point accuracy.
11. Final contour accuracy.
12. Runtime measurement.
13. Parameter run inside IEPS + SCF.

### 8.2 Secondary Only

The following may be included only after the main method works:

1. Canny + OpenCV contour baseline.
2. One qualitative real image test.

### 8.3 Out of Scope

Do not implement as main work:

1. Full Snake / active contour model.
2. Full Chen forward/backward contour tracing.
3. Full Yuen method.
4. Deep learning or U-Net.
5. Large real-world datasets.
6. Complex preprocessing pipelines.
7. Multiple-object segmentation.

## 9. Validation Direction

The validation must stay close to the paper.

Use author-style images:

1. Circle.
2. U-shape.
3. Clean version.
4. Gaussian noisy version.

The validation must answer:

1. Are the selected IEPS points close to the true contour?
2. Does SCF connect the IEPS points into a closed contour?
3. How much does the final contour match the ground truth?
4. How sensitive is the method to threshold, number of scan lines, iterations, and SCF stopping tolerance?
5. Which implementation choices were missing from the paper and had to be defined?

## 10. Metrics

Use the following metrics:

### 10.1 IEPS Point Accuracy

A selected IEPS point is correct if it lies within a fixed pixel tolerance from the ground-truth contour.

Recommended tolerance:

- 2 pixels
- optionally test 3 pixels

Metric:

```text
IEPS accuracy = correct IEPS points / total IEPS points
```

### 10.2 Neighbor Distance Statistics

Calculate distance between neighboring IEPS points.

Report:

1. Mean distance.
2. Standard deviation.

This follows the paper's idea of evaluating whether selected points are evenly distributed.

### 10.3 Contour Precision

```text
precision = correct predicted contour pixels / all predicted contour pixels
```

### 10.4 Contour Recall

```text
recall = recovered true contour pixels / all true contour pixels
```

### 10.5 F1 Score

```text
F1 = 2 * precision * recall / (precision + recall)
```

### 10.6 Runtime

Measure runtime using Python timing.

Report:

1. IEPS runtime.
2. SCF runtime.
3. Total IEPS + SCF runtime.
4. Optional Canny + OpenCV baseline runtime.

## 11. Parameter Run

The extension must stay inside the authors' method.

Run a small parameter study:

| Parameter | Values |
|---|---|
| Sobel threshold | 40, 64, 90 |
| Initial scan lines | 4, 8 |
| IEPS iterations | 2, 3 |
| SCF stopping tolerance | 1, 2, 3 pixels |
| SCF score formula | gradient only vs gradient / distance² |

Main purpose:

To understand which missing or weakly specified details affect reproducibility.

## 12. Expected Results

Expected successful behavior:

1. IEPS should select points close to the circle and U-shape boundary.
2. 4 scan lines with 3 iterations should produce approximately 32 initial edge points.
3. SCF should connect neighboring points into a mostly closed contour.
4. Moderate noise should reduce accuracy but should not completely break the method.
5. Threshold and SCF tolerance will strongly affect the result.
6. SCF implementation choices will be critical for reproducing the paper behavior.

Expected limitations:

1. Heavy noise may create false gradients.
2. Weak edges may cause SCF to leave the true contour.
3. Multiple objects are not handled.
4. Off-center objects may affect center-of-gravity-based scan lines.
5. The method is sensitive to parameter choices.
6. The paper does not fully define all implementation details needed for exact reproduction.

## 13. Final Research Contribution

The project contribution is:

**A traditional computer vision reimplementation and validation of IEPS + SCF, with explicit documentation of missing implementation choices in SCF and a focused parameter study showing how those choices affect contour closure and accuracy.**

## 14. Presentation Message

Use this final message:

**The IEPS + SCF method is a classical computer vision algorithm that combines image gradients and geometry. My reimplementation shows that the IEPS idea is understandable and reproducible, but the SCF stage requires additional practical rules such as stopping tolerance, loop prevention, tie-breaking, and fallback handling. These details are under-specified in the paper and significantly affect whether the contour closes correctly.**
