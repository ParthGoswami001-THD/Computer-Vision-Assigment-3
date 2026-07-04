"""@file image_generation.py
@brief Synthetic author-style images and ground-truth contours.

The paper evaluates controlled man-made circle and U-shape images, including
Gaussian-noisy versions. The defaults use black background and white foreground
so that center-of-gravity moments are not biased by a non-zero background.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def _contour_from_mask(mask: np.ndarray) -> np.ndarray:
    """@brief Create a contour band from a binary object mask.

    A morphological gradient is used because Sobel/Canny edge evidence usually
    lies on both sides of a sharp object boundary. This gives fairer tolerance-
    based evaluation than using only the inner eroded boundary.

    @param mask Binary object mask with values 0 and 255.
    @return Binary contour mask with values 0 and 255.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def create_circle_image(
    size: Tuple[int, int] = (256, 256),
    radius: int = 70,
    center: Optional[Point] = None,
    background: int = 0,
    foreground: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@brief Create a synthetic circle image and ground truth.

    @param size Image size as (height, width).
    @param radius Circle radius in pixels.
    @param center Optional circle center as (x, y). If None, image center is used.
    @param background Background grayscale intensity.
    @param foreground Object grayscale intensity.
    @return Tuple (image, mask, contour), uint8 arrays.
    """
    height, width = size
    if center is None:
        center = (width // 2, height // 2)
    image = np.full((height, width), background, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    image[mask > 0] = foreground
    return image, mask, _contour_from_mask(mask)


def create_u_shape_image(
    size: Tuple[int, int] = (256, 256),
    background: int = 0,
    foreground: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@brief Create a paper-style synthetic concave U-shape image.

    The ICARCV paper's U-shape is not a perfect rectangular block. It has
    slanted shoulders near the top, a narrow central notch, mostly vertical
    sides, and a rounded lower boundary. The polygon below keeps the image
    synthetic and controlled while matching that silhouette more closely than a
    simple three-rectangle U.

    @param size Image size as (height, width).
    @param background Background grayscale intensity.
    @param foreground Object grayscale intensity.
    @return Tuple (image, mask, contour), uint8 arrays.
    """
    height, width = size
    image = np.full((height, width), background, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    outer_left = width * 0.30
    outer_right = width * 0.70
    side_top = height * 0.31
    side_bottom = height * 0.70

    bottom_center = (width * 0.50, height * 0.70)
    bottom_rx = width * 0.20
    bottom_ry = height * 0.095
    bottom_curve = [
        (
            int(round(bottom_center[0] + bottom_rx * math.cos(angle))),
            int(round(bottom_center[1] + bottom_ry * math.sin(angle))),
        )
        for angle in np.linspace(0.0, math.pi, 18)
    ]

    points = [
        (int(round(outer_left)), int(round(side_top))),
        (int(round(width * 0.36)), int(round(height * 0.25))),
        (int(round(width * 0.43)), int(round(height * 0.25))),
        (int(round(width * 0.47)), int(round(height * 0.36))),
        (int(round(width * 0.46)), int(round(height * 0.44))),
        (int(round(width * 0.54)), int(round(height * 0.44))),
        (int(round(width * 0.53)), int(round(height * 0.36))),
        (int(round(width * 0.57)), int(round(height * 0.25))),
        (int(round(width * 0.64)), int(round(height * 0.25))),
        (int(round(outer_right)), int(round(side_top))),
        (int(round(outer_right)), int(round(side_bottom))),
        *bottom_curve,
        (int(round(outer_left)), int(round(side_bottom))),
    ]
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)], 255)

    image[mask > 0] = foreground
    return image, mask, _contour_from_mask(mask)


def create_vase_like_image(
    size: Tuple[int, int] = (256, 256),
    background: int = 0,
    foreground: int = 255,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """@brief Create a simple vase-like synthetic object for optional testing.

    @param size Image size as (height, width).
    @param background Background intensity.
    @param foreground Foreground intensity.
    @return Tuple (image, mask, contour), uint8 arrays.
    """
    height, width = size
    image = np.full((height, width), background, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([
        (int(width * 0.39), int(height * 0.17)),
        (int(width * 0.61), int(height * 0.17)),
        (int(width * 0.59), int(height * 0.32)),
        (int(width * 0.55), int(height * 0.40)),
        (int(width * 0.68), int(height * 0.50)),
        (int(width * 0.75), int(height * 0.72)),
        (int(width * 0.66), int(height * 0.86)),
        (int(width * 0.34), int(height * 0.86)),
        (int(width * 0.25), int(height * 0.72)),
        (int(width * 0.32), int(height * 0.50)),
        (int(width * 0.45), int(height * 0.40)),
        (int(width * 0.41), int(height * 0.32)),
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    image[mask > 0] = foreground
    return image, mask, _contour_from_mask(mask)


def _resize_max_dimension(image: np.ndarray, max_dimension: int, interpolation: int) -> np.ndarray:
    """@brief Resize an image so its largest dimension is bounded.

    @param image Input image.
    @param max_dimension Maximum output width or height.
    @param interpolation OpenCV interpolation flag.
    @return Resized image, or original copy if already small enough.
    """
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_dimension:
        return image.copy()
    scale = max_dimension / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=interpolation)


def _largest_object_component(binary: np.ndarray) -> np.ndarray:
    """@brief Keep the best single object component from a binary mask.

    Components touching the image border are penalized because they are often
    background after Otsu thresholding on real photographs.

    @param binary Binary uint8 image with values 0 and 255.
    @return Binary mask for the selected component.
    """
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    height, width = binary.shape[:2]
    best_label = 0
    best_score = -1.0
    min_area = max(16, int(0.005 * height * width))
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        score = float(area) * (0.25 if touches_border else 1.0)
        if score > best_score:
            best_label = label
            best_score = score

    mask = np.zeros_like(binary, dtype=np.uint8)
    if best_label > 0:
        mask[labels == best_label] = 255
    return mask


def estimate_single_object_mask(image: np.ndarray) -> np.ndarray:
    """@brief Estimate one foreground object mask for a real grayscale image.

    This is intended for qualitative real-vase testing when a hand mask is not
    available. It tries both Otsu polarities and keeps the most plausible
    non-border object component.

    @param image Input grayscale image.
    @return Estimated binary object mask.
    """
    gray = image.astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _threshold, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), dtype=np.uint8)

    best_mask = np.zeros_like(gray, dtype=np.uint8)
    best_area = -1
    for candidate in (otsu, cv2.bitwise_not(otsu)):
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        component = _largest_object_component(cleaned)
        area = int(cv2.countNonZero(component))
        if area > best_area:
            best_area = area
            best_mask = component

    if best_area <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    return best_mask


def load_real_vase_case(
    image_path: Path,
    mask_path: Optional[Path] = None,
    max_dimension: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """@brief Load and prepare a real vase image for IEPS + SCF testing.

    If a mask is supplied, it is used as the evaluation reference. Otherwise an
    Otsu/largest-component mask is estimated, which should be treated as a
    proxy reference rather than true ground truth. The returned grayscale image
    is oriented so the object is brighter than the background, matching the
    center-of-gravity assumption used by the synthetic experiments.

    @param image_path Path to a real vase image.
    @param mask_path Optional path to a binary vase mask.
    @param max_dimension Maximum image dimension after resizing.
    @return Tuple (image, mask, contour, mask_source).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read real vase image: {image_path}")
    image = _resize_max_dimension(image, max_dimension, cv2.INTER_AREA)

    mask_source = "otsu_estimated_mask"
    if mask_path is not None and mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read real vase mask: {mask_path}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        _threshold, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_source = "provided_mask"
    else:
        mask = estimate_single_object_mask(image)

    if int(cv2.countNonZero(mask)) > 0:
        object_mean = float(image[mask > 0].mean())
        background_pixels = image[mask == 0]
        background_mean = float(background_pixels.mean()) if background_pixels.size else object_mean
        if object_mean < background_mean:
            image = cv2.bitwise_not(image)

    return image, mask, _contour_from_mask(mask), mask_source


def add_gaussian_noise(image: np.ndarray, sigma: float = 20.0, seed: int = 42) -> np.ndarray:
    """@brief Add reproducible white Gaussian noise.

    @param image Input uint8 grayscale image.
    @param sigma Gaussian noise standard deviation.
    @param seed Random seed for reproducible validation.
    @return Noisy uint8 grayscale image.
    """
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_gaussian_noise_snr(image: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """@brief Add Gaussian noise at an approximate signal-to-noise ratio.

    SNR is computed using image variance as signal power. This gives repeatable
    paper-style noisy cases, although it may not match the paper's exact noise
    generator.

    @param image Input uint8 grayscale image.
    @param snr_db Desired signal-to-noise ratio in dB.
    @param seed Random seed.
    @return Noisy uint8 grayscale image.
    """
    signal = image.astype(np.float32)
    signal_power = float(np.var(signal))
    if signal_power <= 1e-12:
        return image.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    sigma = math.sqrt(noise_power)
    return add_gaussian_noise(image, sigma=sigma, seed=seed)
