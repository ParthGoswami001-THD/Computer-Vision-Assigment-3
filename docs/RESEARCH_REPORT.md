# Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study

**Course:** Computer Vision Assignment 3  
**Paper:** Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu, *An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction*, ICARCV 2010 [1].

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

## SCF Validation Against The Paper Description

The SCF implementation follows the flow described in the paper section on segmental contour following.

| Paper step | Validation in this implementation |
|---|---|
| Use initial contour points from IEPS | `run_scf()` receives the final IEPS point list from `run_ieps()`. |
| Trace segmental contours between neighboring initial edge points | `run_scf()` loops through each point `S_i` and target `S_{i+1}`, including the last point back to the first. |
| Calculate related direction | `direction_state()` computes `Dx = x_target - x_current` and `Dy = y_target - y_current`, then classifies the sign pair into one of the eight movement directions or `middle`. |
| Decide mask direction | `_directional_candidates()` chooses the three local candidate pixels in the selected direction, matching the paper's A/B/C operating-mask idea. |
| Calculate gravity-style force | `_score_candidate()` uses the Sobel gradient and distance to the target point. The default score is `gradient / (distance^2 + 1)`, which is the implemented gravity-style candidate score. |
| Repeat until closed contour | `trace_segment_greedy()` stops a segment when it reaches the target within tolerance; `run_scf()` connects all neighboring segments into one contour. |

I also checked the direction states directly in code. With image coordinates `(x, y)`, positive `x` is east and positive `y` is south, so the implementation uses:

| `Dx` sign | `Dy` sign | Direction used |
|---|---|---|
| positive | positive | south-east |
| positive | zero | east |
| positive | negative | north-east |
| zero | positive | south |
| zero | zero | middle / linked |
| zero | negative | north |
| negative | positive | south-west |
| negative | zero | west |
| negative | negative | north-west |

This follows the coordinate convention used by OpenCV images. If the copied paper table appears to list `Dy < 0` as west or `Dx < 0, Dy = 0` as north, that is likely a table/OCR typo; the figure and image-coordinate logic correspond to north for negative `y` and west for negative `x`.

In the current validated run, all main SCF experiments report every segment reaching its target:

```text
circle_clean: 32 / 32 segments reached target
circle_noisy: 32 / 32 segments reached target
u_shape_clean: 32 / 32 segments reached target
u_shape_noisy: 32 / 32 segments reached target
```

## Validation Data

I used synthetic images similar to the author-style examples:

- clean circle,
- noisy circle,
- clean U-shape,
- noisy U-shape.

Each case has an input image, a binary object mask, and a ground-truth contour mask. Gaussian noise is generated with fixed seeds so the results are reproducible.

One deliberate deviation from the written project plan is documented here: the plan specifies synthetic intensities of background 40 and object 200, while this implementation defaults to background 0 and object 255. The reason is that raw center-of-gravity moments are biased by a non-zero background, which shifts every IEPS scan line. The generator functions still accept `background`/`foreground` arguments, so the 40/200 configuration remains reproducible, and the default `contrast` center mode (subtracting the image minimum before computing moments) handles non-zero backgrounds correctly. The bias itself is treated as one of the reproducibility findings rather than silently patched.

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
| Refinement rule and point order | Different refinement candidates and orders change the U-shape contour. | Defaults are `farthest_from_center` and `topological`; `closest_to_reference`, `max_gradient`, and `angle` order are measured in the parameter study. |

The SCF part was the most sensitive. Without explicit stopping, loop prevention, tie-breaking, and fallback rules, the contour follower is easy to make unstable.

I do not treat these as a separate bug-fix experiment in the final results. They are better understood as reproducibility issues caused by missing implementation details in the paper. For example, raw center-of-gravity moments can be biased when the background is not zero, different IEPS point-ordering rules change the U-shape contour, and SCF needs explicit choices for stopping, loop prevention, weak-gradient fallback, and candidate tie-breaking. These choices are documented here and tested through the parameter study rather than reported as a separate bug-fix table.

## Validation Against The Paper Experiments

The experimental section of the paper reports two main results: first, IEPS is compared with Yuen's initialization method [2]; second, SCF is compared with Chen's contour tracing method [4]. The paper also shows Snake results using initial points from Yuen and IEPS [3], plus a real vase image test.

This project validates the paper direction end to end: the author-style synthetic shapes, IEPS with 4 scan lines and 3 iterations, threshold 64, 32 initial edge points, SCF between neighboring IEPS points, true-edge comparison, runtime, SNR-based noisy circle tests, and the real-vase test path. As secondary paper context, I also added compact Yuen-style, Snake-style, and Chen-style comparison baselines. These baselines are approximations from the paper description, not exact source-code reproductions of the cited methods, and they are not the main contribution.

| Paper experiment | Paper reported result | Validation in this project |
|---|---|---|
| Man-made circle and U-shape images, clean and noisy | The paper uses synthetic images where true edge positions are known. | The project generates clean/noisy circle and U-shape images with ground-truth contour masks. |
| IEPS setup | 4 scan lines, 3 iterations, threshold 64, 32 initial edge points. | Implemented as the paper-style default. Main runs produce 32 IEPS points for all four required cases. |
| IEPS true-positive ratio versus Yuen [2] | Paper Table II reports Yuen: circle 17/32, U-shape 23/32; IEPS: circle 22/32, U-shape 28/32. | A Yuen-style fixed-angle farthest-edge initialization baseline is implemented and compared with paper IEPS in `paper_comparison_results.csv`. Our paper-style IEPS gets circle noisy 32/32 and U-shape noisy 25/32 with the current 2-pixel tolerance. Improved IEPS gets U-shape noisy 60/64. **Honest caveat: in the current run the Yuen-style approximation scores 29/32 on the noisy U-shape, i.e., it is not worse than paper-mode IEPS. That is the opposite direction from the paper's Table II and is reported as a limitation of the approximation and synthetic geometry, not as a refutation of the paper.** |
| Neighbor distance and standard deviation | Paper Table III reports IEPS circle mean/std 19.492/3.923 and U-shape mean/std 17.893/4.438. | Current paper-style IEPS gives circle noisy mean/std 13.861/0.333 and U-shape noisy mean/std 19.310/5.573. Absolute means differ because the synthetic image geometry is not identical, but the same spacing idea is measured. |
| Snake results using Yuen and IEPS initial points [3] | Paper Fig. 7 and Fig. 8 show Snake converging better with IEPS initial points. | A simple greedy Snake-style approximation is implemented for both Yuen-style and IEPS initial points. It is useful for validating the experimental structure, but it is not claimed as the exact Kass variational solver. **Honest caveat: the current run does not reproduce the paper's direction. Snake from Yuen-style points scores at least as well as Snake from IEPS points (circle F1 equal at 0.633; U-shape 0.389 versus 0.339), so this comparison is structural context only.** |
| SCF true-positive ratio versus Chen [4] | Paper Table IV reports Chen around 80-81% and proposed SCF around 96-98% for noisy circles at different SNR values. | Chen-style gradient-only tracing and proposed gradient/distance SCF are both run at 29.9, 23.9, and 20.3 dB. **Honest caveat: this test does not reproduce the paper's Chen-versus-SCF gap. In the current run both methods reach F1 = 1.0 at all three SNR levels, so the comparison is non-discriminating: the synthetic circle at these noise levels is too easy to separate gradient-only tracing from the gravity-style score. The SNR generation and Chen baseline are documented approximations, and this table should not be cited as evidence for or against the paper's Table IV numbers.** |
| Real vase image and timing | Paper compares Snake, Chen, and proposed SCF on a vase image. | The project supports a real-vase test through `data/vase.png` or common alternatives such as WebP, with optional `data/vase_mask.png`. It also generates simple Snake-style, Chen-style, and proposed SCF comparison rows. If the image is missing, the run uses a clearly labeled synthetic fallback and does not claim to reproduce the paper's original vase timing table. |

The closest direct reproduction is still the IEPS setup and synthetic-shape validation. The comparison with Yuen, Snake, and Chen is now executable, but should be described as approximation baselines because the paper does not provide enough implementation detail to reproduce those cited methods exactly. In their current form these approximations do not reproduce the paper's comparative claims: Yuen-style initialization is not worse than IEPS on the noisy U-shape, Snake from Yuen-style points is not worse than Snake from IEPS points, and Chen-style tracing ties the proposed SCF at every tested SNR. For the presentation, these tables are structural context that shows the experimental design can be executed; the reproduction evidence for the paper itself comes from the IEPS + SCF pipeline and its own validation.

This validation is a one-time interpretation after implementing the method, not a separate runtime experiment. The report text above explains which paper results are reproduced, which are approximate comparisons, and where exact reproduction is limited by missing paper details. `paper_comparison_results.csv` is kept only as the raw long-form measurement log. For the report and presentation, the same data is exported as clean comparison tables:

```text
results/tables/paper_comparison_results.csv
results/tables/paper_initial_point_comparison.csv
results/tables/paper_snake_initialization_comparison.csv
results/tables/paper_scf_chen_comparison.csv
results/tables/paper_vase_method_comparison.csv
results/tables/paper_comparison_tables.md
```

## Real Vase Test Implementation

The paper also evaluates a real vase image. Since the original vase image is not included in this repository, I implemented the test as an optional real-image path:

```text
data/vase.png
data/vase_mask.png    optional
```

When a local vase image exists as `data/vase.png`, `data/vase.jpg`, `data/vase.jpeg`, `data/vase.webp`, or `data/vase.png.webp`, `main.py` loads it, resizes it to a manageable size, estimates or loads the object mask, orients the grayscale image so the object is brighter than the background, and then runs:

- paper IEPS + greedy SCF,
- paper IEPS + graph SCF,
- improved IEPS + greedy SCF.

The outputs are saved under:

```text
results/real_vase_paper/
results/real_vase_improved/
results/tables/real_vase_results.csv
```

If `data/vase_mask.png` is not provided, the mask is estimated with Otsu thresholding and largest-component selection. Those values are useful for qualitative validation, but they should not be described as true ground-truth measurements. If no local vase image is found, the code uses the synthetic vase-like fallback and marks the row as `synthetic_fallback_missing_real_vase_image`.

One honest caveat on the improved IEPS extension: on the current real vase image the improved mode performs much worse than paper mode (proxy F1 about 0.25 versus 0.94), because the denser rays, gap filling, and nearest-neighbor reordering misfire on the vase silhouette and its estimated mask. The improved mode's gains are therefore claimed only for the synthetic concave U-shape, not as a general improvement.

## Results Summary

The normal command `python main.py` runs the paper-style greedy SCF and writes `results/tables/main_results.csv` plus the dedicated `results/tables/runtime_results.csv`. The graph and band-graph extension methods are opt-in via `python main.py --run main --scf all`, which writes `results/tables/main_results_all_scf.csv` so the extension rows never mix into the paper-method table. The complete regeneration command is `python main.py --run all`. The high-level behavior is:

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
| IEPS refinement rule | farthest_from_center, closest_to_reference, max_gradient |
| IEPS point order | topological, angle |
| Noise level | clean, low noise, paper-like noisy |

The refinement-rule and point-order rows exist because the written implementation plan describes closest-to-midpoint refinement and angle-sorted ordering, while the paper-mode defaults here are farthest-from-center and topological insertion order. Measuring both sides keeps that deviation explicit instead of hidden, and the measurement supports the defaults: on the noisy U-shape, farthest-from-center refinement reaches F1 0.591 versus 0.442 for closest-to-reference and 0.436 for max-gradient, while topological and angle order tie on this case (0.591).

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

## References

[1] R. C. Hsu, P.-W. Kao, W.-J. Lai, and C.-T. Liu, "An initial edge point selection and segmental contour following for object contour extraction," in *Proc. Int. Conf. Control, Automation, Robotics and Vision (ICARCV)*, 2010.

[2] P. C. Yuen, G. C. Feng, and J. P. Zhou, "A contour detection method: Initialization and contour model," *Pattern Recognition Letters*, vol. 20, no. 2, pp. 141-148, 1999.

[3] M. Kass, A. Witkin, and D. Terzopoulos, "Snakes: Active contour models," *International Journal of Computer Vision*, vol. 1, no. 4, pp. 321-331, 1988.

[4] B. D. Chen and P. Siy, "Forward/backward contour tracing with feedback," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 9, no. 3, pp. 438-446, 1987.

[5] K. S. Fu and J. K. Mui, "A survey on image segmentation," *Pattern Recognition*, vol. 13, pp. 3-16, 1981.
