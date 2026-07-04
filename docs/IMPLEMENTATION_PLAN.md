# AI Agent Implementation Plan

## IEPS + SCF Reimplementation for Computer Vision Assignment 3

## 1. AI Agent Mission

Build a Python/OpenCV project that reimplements and validates the paper's proposed method:

1. Initial Edge Point Selection — IEPS
2. Segmental Contour Following — SCF

The implementation must follow the authors' traditional computer vision direction.

The main goal is not to create a perfect industrial contour detector.
The main goal is to reimplement the paper's algorithm from the description and identify missing implementation details that are necessary for reproducibility.

## 2. Technology Stack

Use:

```text
Python 3.10+
numpy
opencv-python
matplotlib
pandas
```

Do not use deep learning libraries.

Do not use scikit-image active contours as the main implementation.

OpenCV may be used for:

1. Sobel gradient.
2. Canny secondary baseline.
3. Drawing and image I/O.
4. Basic contour utilities for evaluation/baseline only.

Hand-code the main algorithmic parts:

1. Center of gravity.
2. Scan-line sampling.
3. IEPS.
4. IEPS refinement.
5. SCF.
6. Evaluation metrics.

## 3. Required Project Structure

Create this folder structure:

```text
cv_assignment3_ieps_scf/
│
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── image_generation.py
│   ├── gradients.py
│   ├── geometry.py
│   ├── ieps.py
│   ├── scf.py
│   ├── baseline.py
│   ├── evaluation.py
│   └── visualization.py
│
├── results/
│   ├── circle_clean/
│   ├── circle_noisy/
│   ├── u_shape_clean/
│   ├── u_shape_noisy/
│   └── tables/
│
└── docs/
    ├── RESEARCH.md
    └── IMPLEMENTATION_PLAN.md
```

## 4. Main Pipeline

Implement the following pipeline:

```text
Generate synthetic image and ground truth
→ Add optional Gaussian noise
→ Compute Sobel gradient magnitude
→ Compute center of gravity
→ Run IEPS
→ Sort initial edge points around center
→ Run SCF between neighboring points
→ Draw final contour
→ Evaluate against ground truth
→ Run parameter experiments
→ Save figures and tables
```

## 5. Module-by-Module Implementation

### 5.1 image_generation.py

Implement functions:

```python
def create_circle_image(size=(256, 256), radius=70, center=None):
    """
    Return:
        image: uint8 grayscale image
        mask: uint8 binary ground-truth object mask
        contour: uint8 binary ground-truth contour mask
    """

def create_u_shape_image(size=(256, 256)):
    """
    Create a synthetic U-shape similar to the paper's synthetic object.
    Return image, mask, contour.
    """

def add_gaussian_noise(image, sigma=20, seed=42):
    """
    Add Gaussian noise with fixed random seed for reproducibility.
    Return noisy uint8 image.
    """
```

Ground-truth contour can be produced from the binary mask using morphological gradient or OpenCV contour drawing.

Synthetic image values:

```text
background intensity: 40
object intensity: 200
```

> **Implementation deviation (documented):** the current implementation defaults
> to background 0 and object 255 because non-zero backgrounds bias the raw
> center-of-gravity moments; both functions accept `background`/`foreground`
> arguments so the 40/200 setting remains reproducible, and the `contrast`
> center mode handles non-zero backgrounds. See README "Documented Deviations
> From The Written Plan".

### 5.2 gradients.py

Implement:

```python
def sobel_gradient_magnitude(image):
    """
    Compute Sobel gradient magnitude.
    Return normalized float32 or uint8 gradient image.
    """
```

Steps:

1. Convert image to float32.
2. Compute Sobel x and Sobel y.
3. Compute magnitude: sqrt(gx^2 + gy^2).
4. Normalize to 0–255.

### 5.3 geometry.py

Implement:

```python
def compute_center_of_gravity(image):
    """
    Compute intensity-based center of gravity using image moments.
    Return (x_c, y_c).
    """
```

Use:

```text
m00 = sum(I)
m10 = sum(x * I)
m01 = sum(y * I)
x_c = m10 / m00
y_c = m01 / m00
```

Remember:

```text
OpenCV image indexing = image[y, x]
Mathematical coordinate = (x, y)
```

Implement:

```python
def sample_ray_from_center(center, angle_rad, image_shape, max_distance=None):
    """
    Sample integer pixel coordinates along a ray from center to image border.
    Return list of (x, y) coordinates.
    """

def sort_points_by_angle(points, center):
    """
    Sort points around center using atan2.
    """

def euclidean_distance(p1, p2):
    """
    Return Euclidean distance between two (x, y) points.
    """
```

### 5.4 ieps.py

Implement author-style IEPS.

Main function:

```python
def run_ieps(
    image,
    gradient,
    initial_scan_lines=4,
    iterations=3,
    threshold=64,
    center=None,
    default_scan_distance=None
):
    """
    Run Initial Edge Point Selection.

    Return:
        points: list of final IEPS points
        debug_info: dictionary with scan lines, intermediate points, center
    """
```

IEPS algorithm:

```text
1. Compute center of gravity if center is not provided.
2. Emit N initial scan lines with equal angular spacing.
3. For each scan line:
   a. sample pixels from center to image border
   b. find pixels where gradient > threshold
   c. select the farthest valid candidate from center
4. Sort points around center.
5. For each refinement iteration:
   a. for each neighboring pair of points:
      i. compute midpoint
      ii. compute segment direction between the two points
      iii. compute normal direction
      iv. sample a short scan line along the normal direction
      v. find candidate pixels where gradient > threshold
      vi. select best candidate, preferably closest to midpoint but on strong gradient
   b. merge old points and new points
   c. sort again around center
6. Return final points.
```

Important implementation rule:

If no candidate is found on a scan line:

1. Try lower local threshold, or
2. choose maximum gradient point on that line, or
3. skip point and document it.

Preferred fallback:

```text
Choose the maximum gradient point along the sampled line if no point exceeds threshold.
```

Expected output:

```text
With 4 scan lines and 3 iterations, final point count should be around 32.
```

Debug images must show:

1. Center of gravity.
2. Initial scan lines.
3. Selected IEPS points.
4. Final refined IEPS points.

### 5.5 scf.py

Implement simplified but author-aligned SCF.

Main function:

```python
def run_scf(
    gradient,
    ieps_points,
    stop_tolerance=2,
    max_step_factor=3.0,
    use_distance_weight=True
):
    """
    Connect IEPS points segment by segment.

    Return:
        contour_points: list of (x, y) points
        debug_info: dictionary with per-segment paths and stopping reasons
    """
```

Segment tracing function:

```python
def trace_segment(
    gradient,
    start,
    target,
    stop_tolerance=2,
    max_steps=None,
    use_distance_weight=True
):
    """
    Trace contour from start point to target point.
    """
```

SCF algorithm:

```text
For each neighboring pair Si → Si+1:

1. current = Si
2. target = Si+1
3. visited = empty set
4. max_steps = max_step_factor * distance(start, target)
5. while distance(current, target) > stop_tolerance:
   a. compute direction vector from current to target
   b. choose candidate pixels from 3×3 neighborhood
   c. remove candidates outside image
   d. avoid visited candidates if possible
   e. score each candidate:
        if use_distance_weight:
            score = gradient(candidate) / (distance(candidate, target)^2 + epsilon)
        else:
            score = gradient(candidate)
   f. tie-break:
        first: highest score
        second: highest gradient
        third: smallest distance to target
   g. if all candidates are weak:
        choose candidate closest to target
   h. append selected candidate to path
   i. mark selected candidate visited
   j. current = selected candidate
   k. stop if max_steps reached
6. append target if needed
7. return segment path
```

Important missing-details implementation decisions:

| Missing detail | Decision |
|---|---|
| Segment finished | distance to target <= stop_tolerance |
| Default stop tolerance | 2 pixels |
| Loop prevention | visited set per segment |
| Max steps | 3 × start-target distance |
| Tie-break | score, gradient, distance |
| Weak-gradient fallback | closest-to-target candidate |
| Border handling | ignore outside-image candidates |
| Closed contour | connect all IEPS points, including last to first |

### 5.6 baseline.py

Secondary baseline only.

Implement:

```python
def canny_contour_baseline(image):
    """
    Apply Canny + OpenCV findContours.
    Select largest external contour.
    Return contour mask and contour points.
    """
```

This must not become the main project.

Use it only for a small comparison.

### 5.7 evaluation.py

Implement:

```python
def point_accuracy(points, ground_truth_contour_mask, tolerance=2):
    """
    Count how many IEPS points are within tolerance pixels of ground-truth contour.
    """
```

Implementation idea:

1. Compute distance transform from inverse contour mask.
2. For each point, read distance to nearest true contour pixel.
3. Correct if distance <= tolerance.

```python
def contour_metrics(predicted_contour_mask, ground_truth_contour_mask, tolerance=2):
    """
    Return precision, recall, F1 using tolerance-based contour matching.
    """

def neighbor_distance_stats(points):
    """
    Return mean and standard deviation of distances between neighboring IEPS points.
    """

def measure_runtime(function, *args, **kwargs):
    """
    Measure runtime using time.perf_counter().
    """
```

### 5.8 visualization.py

Implement:

```python
def draw_points(image, points, color=(0, 0, 255)):
    """
    Draw IEPS points on image.
    """

def draw_contour_points(image, contour_points, color=(255, 0, 0)):
    """
    Draw contour points or polyline.
    """

def save_result_panel(...):
    """
    Save a panel showing:
    original image,
    noisy image,
    Sobel gradient,
    IEPS points,
    SCF contour,
    ground truth,
    optional Canny baseline.
    """
```

## 6. main.py

Main script must run all experiments.

Expected CLI:

```text
python main.py
```

It should:

1. Create test images.
2. Run IEPS + SCF with default author-style parameters.
3. Save visual outputs.
4. Run evaluation.
5. Run parameter study.
6. Save result tables as CSV.
7. Print summary to terminal.

## 7. Experiments

### 7.1 Main Author-Style Experiments

Run:

```text
circle_clean
circle_noisy
u_shape_clean
u_shape_noisy
```

Default parameters:

```text
initial_scan_lines = 4
iterations = 3
threshold = 64
scf_stop_tolerance = 2
```

For each, save:

1. Input image.
2. Sobel gradient.
3. IEPS points.
4. SCF contour.
5. Ground-truth contour.
6. Evaluation metrics.

### 7.2 Parameter Study

Run compact parameter test:

```text
threshold: 40, 64, 90
scan lines: 4, 8
iterations: 2, 3
SCF tolerance: 1, 2, 3
SCF score: gradient only vs gradient / distance²
```

Do not test every possible combination if runtime is too high.

Minimum required parameter runs:

1. Threshold study with fixed scan lines and iterations.
2. Iteration study with fixed threshold.
3. SCF tolerance study with fixed IEPS points.
4. Gravity-score comparison.

### 7.3 Secondary Canny Baseline

Run once per image:

```text
Canny + findContours + largest external contour
```

Report:

1. F1 score.
2. Runtime.
3. Visual comparison.

## 8. Output Files

Save outputs like this:

```text
results/
├── circle_clean/
│   ├── original.png
│   ├── gradient.png
│   ├── ieps_points.png
│   ├── scf_contour.png
│   ├── ground_truth.png
│   └── panel.png
│
├── circle_noisy/
├── u_shape_clean/
├── u_shape_noisy/
│
└── tables/
    ├── main_results.csv
    ├── parameter_study.csv
    └── runtime_results.csv
```

## 9. Acceptance Criteria

The implementation is successful if:

1. The project runs with `python main.py`.
2. Synthetic circle and U-shape images are generated.
3. Sobel gradient images are saved.
4. Center of gravity is computed and visualized.
5. IEPS returns approximately 32 points for 4 scan lines and 3 iterations.
6. IEPS points are drawn correctly on the contour.
7. SCF connects neighboring IEPS points into a mostly closed contour.
8. Evaluation metrics are computed.
9. Runtime is measured.
10. Parameter study results are saved.
11. README explains the missing SCF implementation choices.
12. Code is readable and documented.

## 10. README Requirements

The README must include:

1. Paper title.
2. Project goal.
3. How to run.
4. Main pipeline.
5. Explanation of IEPS.
6. Explanation of SCF.
7. Implementation choices for missing details.
8. Parameter settings.
9. Results summary.
10. Limitations.

## 11. Critical Documentation Point

The README and presentation must explicitly state:

```text
The main reproducibility challenge was SCF.
The paper describes the gravity-force and mask idea,
but does not fully specify stopping tolerance, loop prevention,
tie-breaking, weak-gradient fallback, and max-step handling.
These choices were implemented explicitly and evaluated.
```

## 12. Future Work

Only mention as future work:

1. Full comparison with Snake.
2. Full comparison with Chen method.
3. More real images.
4. Multiple-object handling.
5. U-Net or deep-learning-based segmentation.

Do not implement these in the main project.

## 13. Final Deliverable Summary

The AI agent must produce:

1. Working Python project.
2. Documented source code.
3. Results folder with images.
4. CSV tables with metrics.
5. README.
6. Clear implementation of IEPS + SCF.
7. Parameter study focused on missing reproducibility details.
