# Final Project Plan

## Title

**Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study**

## Research Question

Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying which missing implementation parameters are necessary to reproduce the reported behavior?

## Scope

The project should stay close to the paper. The goal is not to build a new segmentation system, but to understand and reproduce the authors' pipeline carefully.

Required work:

- generate synthetic circle and U-shape images,
- create clean and Gaussian-noisy versions,
- compute Sobel gradient magnitude,
- compute the center of gravity,
- implement IEPS scan-line selection and refinement,
- implement SCF segment tracing between neighboring IEPS points,
- validate IEPS points and final contours against ground truth,
- record runtime,
- save visual panels and CSV result tables,
- document the missing implementation choices.

Secondary work:

- Canny + OpenCV contour extraction as a small baseline,
- optional real-like synthetic shape,
- optional graph-based SCF and improved IEPS experiments.

Out of scope for the main assignment:

- full Snake implementation,
- full Chen-method reproduction,
- U-Net or other deep learning,
- large real-image datasets,
- heavy preprocessing pipelines.

## Implementation Steps

### 1. Synthetic Data

Create the author-style test cases:

- circle clean,
- circle noisy,
- U-shape clean,
- U-shape noisy.

Each case needs the grayscale image, object mask, and ground-truth contour mask. Noise should use fixed seeds so the results can be repeated.

### 2. Core Geometry And Gradients

Implement:

- Sobel gradient magnitude,
- center of gravity from image moments,
- ray sampling from the center,
- midpoint and normal-line sampling,
- point ordering around the contour.

The code should consistently treat points as `(x, y)`, even though NumPy arrays are indexed as `image[y, x]`.

### 3. IEPS

Implement the paper-style version first:

- 4 initial scan lines,
- farthest edge candidate above threshold,
- 3 refinement iterations,
- threshold near 64,
- about 32 final IEPS points.

Expose ambiguous choices as parameters instead of hiding them. This is useful for the reproducibility discussion.

### 4. SCF

Implement segment-by-segment contour following:

- start from one IEPS point,
- trace toward the next IEPS point,
- use local directional candidates,
- score candidates using gradient and distance to target,
- stop when close enough to the target,
- avoid loops,
- handle weak-gradient cases,
- connect the last point back to the first.

The important part is to make the missing rules explicit: stopping tolerance, tie-breaking, loop prevention, and max-step behavior.

### 5. Evaluation

Measure:

- IEPS point accuracy,
- neighbor distance mean and standard deviation,
- contour precision,
- contour recall,
- contour F1,
- runtime.

The evaluation should use a small pixel tolerance because exact pixel alignment is not realistic for edge-based contours.

### 6. Parameter Study

Run compact tests inside IEPS and SCF:

- threshold: `40`, `64`, `90`,
- scan lines: `4`, `8`,
- IEPS iterations: `2`, `3`,
- SCF tolerance: `1`, `2`, `3`,
- SCF score: `gradient_only` vs `gradient_distance2`,
- noise level: clean, low noise, paper-like noisy.

This is the best extension because it directly answers what the paper leaves under-specified.

## Final Presentation Message

After the discussion with the professor, I focused the work on the authors' original direction: reimplementing and validating IEPS + SCF. Instead of adding unrelated preprocessing or large external methods, I investigated the parameters and implementation choices that are necessary to reproduce the paper's contour extraction behavior.

The key finding is that IEPS + SCF can be reproduced on simple synthetic shapes, but the final behavior depends strongly on practical details that are not fully specified in the paper. This is especially visible on the concave U-shape.
