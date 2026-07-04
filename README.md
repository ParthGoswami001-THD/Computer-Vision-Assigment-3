# Reimplementation and Validation of IEPS + SCF for Object Contour Extraction: A Traditional Computer Vision Reproducibility Study

Computer Vision Assignment 3  
Paper: **An Initial Edge Point Selection and Segmental Contour Following for Object Contour Extraction**  
Roy Chaoming Hsu, Ping-Wen Kao, Wei-Jie Lai, Cheng-Ting Liu, ICARCV 2010.

This repository is a reproducibility-focused implementation of the paper's IEPS + SCF contour extraction method. After the discussion with the professor, I kept the work centered on the authors' original idea instead of turning the assignment into a broad Canny, preprocessing, or deep-learning comparison.

The final research question is:

> Can the IEPS + SCF method proposed by Hsu et al. be reimplemented from the paper description and validated on author-style synthetic circle and U-shape images, including noisy cases, while identifying which missing implementation parameters are necessary to reproduce the reported behavior?

## What The Code Does

The main pipeline is:

```text
grayscale image
-> Sobel gradient magnitude
-> center of gravity
-> Initial Edge Point Selection (IEPS)
-> Segmental Contour Following (SCF)
-> contour mask
-> validation against ground truth
```

The paper-faithful setup uses the same direction as the authors:

- synthetic circle and U-shape images,
- clean and Gaussian-noisy versions,
- Sobel gradient magnitude for edge evidence,
- center of gravity from image moments,
- IEPS with 4 initial scan lines,
- 3 IEPS refinement iterations,
- Sobel threshold around 64,
- about 32 selected initial edge points,
- SCF between neighboring IEPS points,
- a gradient/distance-style candidate score,
- contour evaluation against ground-truth masks.

Canny + OpenCV contours are included only as a small reference baseline. They are not the main topic of the project.

### IEPS In Short

IEPS selects the initial edge points automatically:

1. Compute the center of gravity from image moments.
2. Emit 4 equal-angle scan lines from the center.
3. On each line, keep the farthest pixel whose Sobel magnitude passes the threshold (default 64).
4. For 3 refinement iterations, place a normal scan line through the midpoint of each neighboring pair and insert the selected candidate between them.
5. With 4 scan lines and 3 iterations this yields about 32 points (`4 * 2^3`).

If no pixel on a line passes the threshold, the default fallback keeps the maximum-gradient pixel on that line instead of dropping the point.

### SCF In Short

SCF connects each IEPS point to the next one (including last back to first) by walking pixel by pixel:

1. Classify the related direction to the target from the signs of `Dx` and `Dy`.
2. Build the three-candidate operating mask for that direction.
3. Score each candidate with the gravity-style rule `gradient / (distance_to_target^2 + 1)`.
4. Move to the best candidate and repeat until the target is reached.

The paper describes this idea but not the code-level rules, so the following decisions were fixed explicitly:

| Missing SCF detail | Decision used here |
|---|---|
| Segment finished | distance to target <= stop tolerance |
| Default stop tolerance | 2 pixels |
| Loop prevention | visited-pixel set per segment |
| Max steps per segment | 3 x start-target distance |
| Candidate tie-break | score first, gradient second, target distance third |
| Weak-gradient fallback | move to the candidate closest to the target |
| Border handling | ignore candidates outside the image |
| Closed contour | connect all IEPS points, including last to first |

> The main reproducibility challenge was SCF. The paper describes the gravity-force and mask idea, but does not fully specify stopping tolerance, loop prevention, tie-breaking, weak-gradient fallback, and max-step handling. These choices were implemented explicitly and evaluated.

## Why This Became A Reproducibility Study

The paper gives the overall IEPS + SCF idea clearly, but several details are not specific enough to implement directly. Those choices matter more than I expected, especially for the U-shape.

The main missing or under-specified details I had to make explicit were:

- how scan lines are discretized into pixels,
- how the Sobel threshold should be chosen,
- how Gaussian noise is generated,
- what pixel tolerance counts as a true positive,
- when an SCF segment should stop,
- how candidate ties are broken,
- how loops are avoided,
- what to do when no scan-line point passes the threshold,
- how to handle `(x, y)` paper coordinates versus OpenCV `image[y, x]`,
- how midpoint normal scan lines should be sampled.

That is the main research contribution here: not just re-running Sobel, but documenting the practical assumptions needed to make IEPS + SCF executable.

### Documented Deviations From The Written Plan

Three implementation choices deliberately differ from the written project plan, and each is measured instead of hidden:

1. **Synthetic intensities.** The plan specifies background 40 / object 200. The implementation defaults to background 0 / object 255 because raw center-of-gravity moments are biased by a non-zero background: the raw moment centroid drifts toward the image center, which shifts every IEPS scan line. Both generator functions still accept `background`/`foreground` arguments, so the 40/200 setting is reproducible, and the default `contrast` center mode (subtract the image minimum before moments) handles non-zero backgrounds correctly. This bias is itself one of the reproducibility findings.
2. **IEPS point ordering.** The plan says to sort points around the center after each refinement. The default here is `order_mode="topological"`, which keeps each inserted point between its parent pair, because angle sorting can connect points across the U-shape notch. Both orders are compared in the parameter study (`param_order_*` rows).
3. **Refinement candidate rule.** The plan prefers the candidate closest to the midpoint on a strong gradient; the paper-mode default here is `farthest_from_center`, matching the paper's initial-ray rule. Both rules (plus `max_gradient`) are compared in the parameter study (`param_refinement_*` rows).

## Implemented Experiments

The main run generates four required author-style cases:

- `circle_clean`
- `circle_noisy`
- `u_shape_clean`
- `u_shape_noisy`

Other run modes cover:

- the paper-style greedy SCF,
- optional paper-context approximations for Yuen-style initialization, simple Snake-style contour refinement, and Chen-style gradient-only tracing,
- SNR-based circle tracing tests at 29.9, 23.9, and 20.3 dB,
- graph-search SCF as a traditional-CV extension,
- band-limited curvature-aware graph SCF,
- an improved IEPS mode for concave objects,
- a real-vase test when `data/vase.png` is available,
- a focused parameter study,
- the Canny/OpenCV secondary baseline.

The comparison approximations are context only: in the current runs they do not reproduce every direction reported in the paper (see the honesty notes in `docs/RESEARCH_REPORT.md`), which is reported as an approximation limitation rather than hidden.

The parameter study stays inside IEPS and SCF:

- Sobel threshold: `40`, `64`, `90`
- initial scan lines: `4`, `8`
- IEPS iterations: `2`, `3`
- SCF stopping tolerance: `1`, `2`, `3` pixels
- SCF score: `gradient_only` vs `gradient_distance2`
- IEPS refinement rule: `farthest_from_center`, `closest_to_reference`, `max_gradient`
- IEPS point order: `topological` vs `angle`
- noise level: clean, low noise, paper-like noisy

## Results In Short

The circle cases reproduce well. The paper-style IEPS + greedy SCF reaches perfect F1 on the clean and noisy circle images in the current synthetic setup.

The U-shape is the more interesting case. After matching the paper's sloped-shoulder, narrow-notch, rounded-bottom silhouette, the paper-style method performs much better than it did on the earlier rectangular placeholder. It still exposes two problems that are easy to miss from the paper description:

1. The center of gravity can be a weak seed for concave shapes.
2. Neighboring points in an IEPS list are not always neighboring points on the real contour.

The improved IEPS extension handles that by using an interior distance-transform seed and denser scan-line coverage. On the current generated results, it gives a smaller but still visible U-shape gain while preserving circle performance.

The band-limited graph SCF is more robust than the local greedy idea in some noisy cases, but it is much slower. I would present it as an extension, not as the default method.

## Running The Project

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv assignment-3
assignment-3\Scripts\activate
pip install -r requirements.txt
```

Run the main synthetic IEPS + SCF results from the repository root:

```bash
python main.py
```

In this workspace I used:

```bash
.\assignment-3\Scripts\python.exe main.py
```

The default run uses the paper-style greedy SCF only. The graph and band-graph extension methods are opt-in:

```bash
python main.py --list
python main.py --run main --scf all
python main.py --run main --case u_shape_noisy --scf graph
python main.py --run parameter
python main.py --run improvement
python main.py --run vase
python main.py --run paper-comparison
python main.py --run all
```

Use `--no-images` when you only want CSV tables:

```bash
python main.py --run all --no-images
```

Table naming rules: the default paper-method run (all cases, greedy SCF) writes `main_results.csv`; `--scf all` writes `main_results_all_scf.csv`; any other filter writes `main_results_selected.csv`. A matching `runtime_results*.csv` is written next to each main table.

## Output Files

CSV tables are generated by the matching run mode:

```text
python main.py                         -> results/tables/main_results.csv + runtime_results.csv
python main.py --run main --scf all    -> results/tables/main_results_all_scf.csv + runtime_results_all_scf.csv
python main.py --run main --case ...   -> results/tables/main_results_selected.csv + runtime_results_selected.csv
python main.py --run parameter         -> results/tables/parameter_study.csv
python main.py --run improvement       -> results/tables/improvement_comparison.csv
python main.py --run vase              -> results/tables/real_vase_results.csv
python main.py --run paper-comparison  -> results/tables/paper_comparison_results.csv + paper_comparison_results_raw.csv
```

Each case directory contains the plan-style debug and result images:

```text
results/<case>/original.png            input image
results/<case>/gradient.png            Sobel gradient magnitude
results/<case>/center.png              center of gravity marker
results/<case>/initial_scan_lines.png  initial rays + first selected points
results/<case>/ieps_points.png         final refined IEPS points
results/<case>/scf_contour.png         SCF contour (paper-style greedy)
results/<case>/ground_truth.png        ground-truth contour band
results/<case>/canny_baseline.png      secondary Canny baseline overlay
results/<case>/panel.png               combined labeled panel
```

Extension SCF methods write suffixed files instead (`scf_contour_graph.png`, `panel_band_graph.png`, ...).

The optional paper-comparison run also writes presentation-ready tables:

```text
results/tables/paper_initial_point_comparison.csv
results/tables/paper_snake_initialization_comparison.csv
results/tables/paper_scf_chen_comparison.csv
results/tables/paper_vase_method_comparison.csv
results/tables/paper_comparison_tables.md
```

Useful visual panels:

```text
results/circle_clean/panel.png
results/circle_noisy/panel.png
results/u_shape_clean/panel.png
results/u_shape_noisy/panel.png
results/<case>/panel_graph.png
results/<case>/panel_band_graph.png
results/real_vase_paper/panel.png
results/real_vase_improved/panel.png
results/paper_comparisons/*.png
```

## Real Vase Test

To run the real-image vase test from the paper direction, place a vase image at:

```text
data/vase.png
```

The loader also accepts `data/vase.jpg`, `data/vase.jpeg`, `data/vase.webp`, and `data/vase.png.webp`.

An optional hand mask can be added as:

```text
data/vase_mask.png
```

If the mask is missing, the code estimates a proxy mask with Otsu thresholding and the largest object component. If the image itself is missing, the run uses a clearly labeled synthetic vase-like fallback so `main.py` still completes. In that fallback case, the output table marks the input source as `synthetic_fallback_missing_real_vase_image`.

## Repository Layout

```text
main.py                         selectable experiment runner
requirements.txt                Python dependencies
Doxyfile                        Doxygen configuration for the source docs
data/README.md                  real vase input instructions
src/image_generation.py         synthetic shapes and noise
src/gradients.py                Sobel gradient magnitude
src/geometry.py                 center, scan-line, and point geometry
src/ieps.py                     paper-faithful IEPS
src/ieps_improved.py            concave-shape IEPS extension
src/scf.py                      greedy, graph, and band-graph SCF
src/paper_baselines.py          Yuen/Snake/Chen-style comparison approximations
src/evaluation.py               point and contour metrics
src/baseline.py                 secondary Canny baseline
src/visualization.py            saved visual outputs
docs/RESEARCH.md                assigned research direction
docs/IMPLEMENTATION_PLAN.md     assigned implementation plan
docs/RESEARCH_REPORT.md         final research write-up
docs/FULL_IMPLEMENTATION_BUG_FIX_REPORT.md
docs/FINAL_PLAN.md
docs/DOXYGEN_STYLE.md
results/                        generated figures and tables
tools/build_final_presentation.py  presentation builder (PPTX + previews)
outputs/                        generated presentation artifacts
```

## Presentation Summary

Use this as the main presentation line:

> After the discussion with the professor, I focused the work on the authors' original direction: reimplementing and validating IEPS + SCF. Instead of adding unrelated preprocessing or large external methods, I investigated the parameters and implementation choices that are necessary to reproduce the paper's contour extraction behavior.

My strongest finding is that the algorithm is not difficult because of Sobel itself. The difficult part is turning the paper's SCF description into exact code: stopping tolerance, loop prevention, candidate tie-breaking, weak-gradient fallback, max-step handling, and contour ordering all affect whether the contour closes correctly.

## Limitations

- The implementation assumes one dominant object with strong object/background contrast; multiple objects are not handled.
- The concave U-shape is sensitive to the paper's implicit assumptions (interior center of gravity, point ordering, and scan-line coverage). With the corrected author-style silhouette it performs well, but the parameter study still shows that these choices affect the contour.
- The Yuen/Snake/Chen comparison baselines are compact approximations built from the paper's descriptions, not the cited methods' source code. In the current runs they do not reproduce every direction the paper reports (details in `docs/RESEARCH_REPORT.md`), so they are context, not evidence about the original methods.
- Real-vase metrics use an Otsu-estimated proxy mask unless `data/vase_mask.png` is provided, so they are qualitative support rather than true ground-truth measurements.
- Synthetic noise uses fixed seeds and an image-variance SNR definition; the paper's exact noise generator is unknown, so noisy results are comparable within this project but not pixel-identical to the paper.

## Future Work

The next improvements should still stay close to the paper:

- adaptive Sobel thresholding,
- multi-scale Sobel,
- better IEPS candidate selection for concave shapes,
- curvature-aware SCF scoring,
- graph-based contour ordering,
- Snake comparison initialized from IEPS points.

U-Net or other deep-learning methods should only be mentioned as future work, not as part of this assignment's main implementation.
