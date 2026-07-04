# Full Implementation And Bug-Fix Report

Project title: **Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study**

Assigned paper: *An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction*.

The implementation follows the paper's original direction:

```text
image -> Sobel gradient -> center of gravity -> IEPS -> SCF -> closed contour -> validation
```

The point of this work is not to replace the paper with Canny, preprocessing, or deep learning. The point is to reimplement IEPS + SCF closely enough to see which details are missing from the paper description.

Final research question:

> Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying which missing implementation parameters are necessary to reproduce the reported behavior?

## What Was Implemented

### IEPS

The IEPS implementation includes:

- spatial-moment center of gravity,
- equal-angle scan lines from the center,
- Sobel gradient magnitude,
- threshold-based edge candidate selection,
- farthest-edge selection for the first scan lines,
- midpoint generation between neighboring selected points,
- normal scan-line refinement,
- decreasing scan distance over iterations,
- debug storage for selected points and scan lines.

The paper-style default is:

```text
initial scan lines = 4
iterations = 3
Sobel threshold = 64
expected points ~= 32
```

### SCF

The SCF implementation traces the contour segment by segment:

- connect each IEPS point to the next one,
- compute the related direction,
- use the local 3-candidate mask,
- score candidates with a gradient/distance term,
- stop when the target is reached within tolerance,
- prevent loops with a visited set,
- break ties explicitly,
- handle weak-gradient cases,
- close the contour by connecting the last IEPS point back to the first.

### Validation

The full run validates:

- circle clean,
- circle noisy,
- U-shape clean,
- U-shape noisy,
- optional vase-like synthetic object,
- IEPS point accuracy,
- neighbor distance mean and standard deviation,
- contour precision, recall, and F1,
- runtime,
- Canny + OpenCV as a secondary baseline,
- parameter study,
- improved IEPS comparison.

## Main Reproducibility Issue

The largest gap in the paper is not the Sobel operator or the center-of-gravity equation. Those are straightforward. The harder part is SCF, because the paper describes the idea of related directions and gravity-like force, but does not fully define the code-level rules.

In the final version, I treat these as under-specified implementation choices from the paper, not as a separate bug-fix experiment. The evidence belongs in the reproducibility discussion and parameter study.

The rules I had to make explicit were:

| Missing SCF detail | Rule used here |
|---|---|
| Segment stopping | Stop when distance to target is within 2 pixels |
| Loop prevention | Keep visited pixels for each segment |
| Weak-gradient case | Move to the candidate closest to the target |
| Candidate tie-breaking | Score first, gradient second, distance third |
| Step limit | 3 times the endpoint distance |
| Image borders | Ignore out-of-image candidates |
| Closure definition | Connect all neighboring IEPS points, including last-to-first |

This is a major part of the research contribution. SCF is not only a formula; it needs engineering rules. Changing those rules changes whether the contour closes correctly.

## Bugs And Fixes

### 1. Center Of Gravity Bias

The first synthetic version used a non-zero background:

```text
background = 40
foreground = 200
```

Raw image moments include the background intensity, so the center of gravity can drift toward the image center instead of the object. That matters because every IEPS scan line starts from this center.

The current synthetic defaults are black background and white foreground:

```text
background = 0
foreground = 255
```

The code also exposes three center modes:

```text
raw       = direct image moments
contrast  = subtract image minimum before moments
binary    = thresholded foreground moments
```

The default used in experiments is:

```text
center_mode = contrast
```

### 2. Ground-Truth Contour Was Too Narrow

The first ground-truth contour was based on:

```text
mask - eroded(mask)
```

That mostly gives an inner boundary. Sobel and Canny respond around the intensity transition, which can sit on both sides of the object boundary.

The current ground truth uses a morphological gradient:

```text
dilate(mask) - erode(mask)
```

This gives a fairer contour band for tolerance-based evaluation.

### 3. Concave Object Ordering

For a circle, sorting points by angle usually works. For a U-shape, angle sorting can connect points across the open notch, which creates contour segments that do not exist on the real boundary.

SCF assumes that consecutive IEPS points are neighboring contour points. If the order is wrong, the contour follower traces the wrong segments even if the individual IEPS points are reasonable.

The code now supports:

```text
order_mode = topological
order_mode = angle
```

The recommended default is:

```text
order_mode = topological
```

Topological ordering keeps the iterative IEPS insertion order: if a point is created between two old points, it stays between them.

### 4. Refinement Candidate Selection

The paper is clear about farthest-edge selection for the first radial scan lines. The refinement scan lines are less clear.

Different choices behave differently:

```text
farthest_from_center -> good for outer contours
max_gradient         -> can find strong inner edges but may disturb ordering
closest_to_reference -> stable, but may miss a true boundary point
```

Instead of hiding this choice, the implementation exposes it:

```text
refinement_selection = farthest_from_center | max_gradient | closest_to_reference
```

That makes the ambiguity part of the reproducibility study.

### 5. Greedy SCF Can Break On Concavity

The paper-style SCF only checks three local candidates at each step. It is fast, but it can fail when the edge is noisy, the contour turns sharply, the shape is concave, or the next IEPS point is not reachable by a purely local choice.

The paper-style greedy SCF remains the main method. Two optional traditional-CV extensions were added for comparison:

```text
scf_method = graph
scf_method = band_graph
```

The graph mode treats the Sobel image as a cost map and runs a gradient-weighted path search. The band-graph mode adds a soft corridor around the current IEPS segment and penalizes sharp turns. It is more controlled, but also much slower.

### 6. IEPS Assumes A Good Interior Seed

The original IEPS idea works best when the center of gravity is inside the object and the boundary is roughly star-convex from that center. A concave U-shape violates this assumption because the center can fall near the empty notch.

The improved IEPS mode keeps the paper-faithful method separate, but adds a traditional CV fix:

```text
ieps_mode = improved
```

This mode:

- estimates an object silhouette with Otsu thresholding,
- checks whether the center of gravity is inside the silhouette,
- moves the seed to the distance-transform maximum if needed,
- uses denser initial rays,
- fills large angular gaps,
- orders points with a simple boundary-style nearest-neighbor pass.

The main finding is:

> The paper's IEPS assumes the center of gravity lies inside a star-convex object. This works well for circles but is fragile for concave U-shapes. A traditional CV fix using an interior distance-transform seed and denser scan-line coverage improves U-shape performance while preserving circle performance.

## Result Summary

The generated tables are:

```text
results/tables/main_results.csv
results/tables/parameter_study.csv
results/tables/improvement_comparison.csv
```

Observed behavior from the current validated run:

- clean and noisy circles are handled very well,
- U-shapes are much harder because of concavity and ordering ambiguity,
- IEPS is sensitive to center mode and refinement selection,
- SCF is sensitive to stopping tolerance and candidate scoring,
- improved IEPS gives the strongest practical gain on U-shapes,
- graph and band-graph SCF are useful extensions but slower than greedy SCF.

The improved IEPS comparison is especially important:

| Case | Paper IEPS F1 | Improved IEPS F1 |
|---|---:|---:|
| circle_clean | 1.0000 | 1.0000 |
| circle_noisy | 1.0000 | 1.0000 |
| u_shape_clean | 0.6135 | 0.7004 |
| u_shape_noisy | 0.5908 | 0.6965 |

## Improvement Ideas That Stay Within Traditional CV

These are reasonable extensions because they still work inside the authors' IEPS + SCF direction.

### Adaptive Sobel Threshold

The paper uses threshold 64, but does not explain how to tune it. A future version could use a percentile threshold or Otsu threshold on the Sobel magnitude:

```text
threshold = 75th percentile of non-zero Sobel values
```

This would reduce manual tuning across contrast changes.

### Contrast-Based Center Of Gravity

Raw moments are sensitive to non-zero backgrounds. Computing moments on:

```text
I_contrast = I - min(I)
```

keeps the center estimate closer to the object mass.

### Hybrid IEPS Candidate Selection

Farthest-from-center works well for convex shapes. Concave shapes may need a hybrid rule that uses farthest-from-center for initial rays and max-gradient or multi-candidate refinement near difficult regions.

### Robust Interior IEPS Seed

If the center of gravity falls outside the object silhouette:

```text
seed = distance-transform maximum inside the silhouette
```

This keeps the automatic initialization idea but makes it less fragile on concave shapes.

### Edge-Graph SCF

The greedy 3-candidate SCF can make local mistakes. A graph-search version can trace a path through strong Sobel responses:

```text
cost = step_length * (1 + edge_weight * (1 - normalized_gradient))
```

The tradeoff is runtime.

### Band-Limited Curvature-Aware SCF

The band-graph version adds two controls to graph SCF:

- do not wander too far from the IEPS segment corridor,
- penalize sharp direction changes.

This reduces irregular paths, but it introduces more parameters and is slower.

### Multi-Scale Sobel

Single-scale Sobel is sensitive to noise. A future version could compare the current Sobel result with a lightly smoothed version, for example Gaussian sigma 1 before Sobel.

This should stay as future work, not as the central claim.

## Final Interpretation

The method is reproducible on simple high-contrast shapes, especially circles. The difficult part is making the paper's SCF idea precise enough for code.

The final conclusion is:

> The paper's main idea is clear, but exact reproducibility depends on missing implementation choices: SCF stopping rules, candidate selection, loop prevention, contour ordering, center-of-gravity computation, scan-line discretization, and tolerance-based evaluation. Concave shapes also reveal an unstated IEPS assumption: the center of gravity should be a useful interior seed. These choices directly affect contour closure and accuracy.

## Presentation Claim

Use this version in the final presentation:

> After the discussion with the professor, I focused the work on the authors' original direction: reimplementing and validating IEPS + SCF. Instead of adding unrelated preprocessing or large external methods, I investigated the parameters and implementation choices that are necessary to reproduce the paper's contour extraction behavior.

Then follow it with:

> The main lesson is that the algorithm can be reproduced on simple synthetic shapes, but the paper leaves several practical choices open. Those choices become visible on the U-shape, where center placement, point ordering, stopping tolerance, and candidate scoring strongly affect the final contour.
