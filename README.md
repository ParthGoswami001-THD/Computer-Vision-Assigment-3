# Reimplementation and Validation of IEPS + SCF for Object Contour Extraction

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

The parameter study stays inside IEPS and SCF:

- Sobel threshold: `40`, `64`, `90`
- initial scan lines: `4`, `8`
- IEPS iterations: `2`, `3`
- SCF stopping tolerance: `1`, `2`, `3` pixels
- SCF score: `gradient_only` vs `gradient_distance2`
- noise level: clean, low noise, paper-like noisy

## Results In Short

The circle cases reproduce well. The paper-style IEPS + greedy SCF reaches perfect F1 on the clean and noisy circle images in the current synthetic setup.

The U-shape is the more interesting case. It exposes two problems that are easy to miss from the paper description:

1. The center of gravity can be a weak seed for concave shapes.
2. Neighboring points in an IEPS list are not always neighboring points on the real contour.

The improved IEPS extension handles that by using an interior distance-transform seed and denser scan-line coverage. On the current generated results, it improves U-shape performance while preserving circle performance.

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

Useful run modes:

```bash
python main.py --list
python main.py --run main --case u_shape_noisy --scf greedy
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

Filtered main runs write `results/tables/main_results_selected.csv` so they do not overwrite the full `main_results.csv`.

## Output Files

CSV tables are generated by the matching run mode:

```text
python main.py                         -> results/tables/main_results.csv
python main.py --run main --case ...   -> results/tables/main_results_selected.csv
python main.py --run parameter         -> results/tables/parameter_study.csv
python main.py --run improvement       -> results/tables/improvement_comparison.csv
python main.py --run vase              -> results/tables/real_vase_results.csv
python main.py --run paper-comparison  -> results/tables/paper_comparison_results.csv
```

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
results/circle_clean/panel_greedy.png
results/circle_noisy/panel_greedy.png
results/u_shape_clean/panel_greedy.png
results/u_shape_noisy/panel_greedy.png
results/<case>/panel_graph.png
results/<case>/panel_band_graph.png
results/real_vase_paper/panel_greedy.png
results/real_vase_improved/panel_greedy.png
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
docs/RESEARCH_REPORT.md         final research write-up
docs/FULL_IMPLEMENTATION_BUG_FIX_REPORT.md
docs/FINAL_PLAN.md
docs/DOXYGEN_STYLE.md
results/                        generated figures and tables
```

## Presentation Summary

Use this as the main presentation line:

> After the discussion with the professor, I focused the work on the authors' original direction: reimplementing and validating IEPS + SCF. Instead of adding unrelated preprocessing or large external methods, I investigated the parameters and implementation choices that are necessary to reproduce the paper's contour extraction behavior.

My strongest finding is that the algorithm is not difficult because of Sobel itself. The difficult part is turning the paper's SCF description into exact code: stopping tolerance, loop prevention, candidate tie-breaking, weak-gradient fallback, and contour ordering all affect whether the contour closes correctly.

## Future Work

The next improvements should still stay close to the paper:

- adaptive Sobel thresholding,
- multi-scale Sobel,
- better IEPS candidate selection for concave shapes,
- curvature-aware SCF scoring,
- graph-based contour ordering,
- Snake comparison initialized from IEPS points.

U-Net or other deep-learning methods should only be mentioned as future work, not as part of this assignment's main implementation.
