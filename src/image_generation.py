"""@file image_generation.py
@brief Synthetic author-style images and ground-truth contours.

The paper evaluates controlled man-made circle and U-shape images, including
Gaussian-noisy versions. The defaults use black background and white foreground
so that center-of-gravity moments are not biased by a non-zero background.
"""

from __future__ import annotations

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
    """@brief Create a synthetic concave U-shape image and ground truth.

    @param size Image size as (height, width).
    @param background Background grayscale intensity.
    @param foreground Object grayscale intensity.
    @return Tuple (image, mask, contour), uint8 arrays.
    """
    height, width = size
    image = np.full((height, width), background, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    left = int(width * 0.28)
    right = int(width * 0.72)
    top = int(height * 0.22)
    bottom = int(height * 0.78)
    thickness = int(width * 0.12)
    cv2.rectangle(mask, (left, top), (left + thickness, bottom), 255, -1)
    cv2.rectangle(mask, (right - thickness, top), (right, bottom), 255, -1)
    cv2.rectangle(mask, (left, bottom - thickness), (right, bottom), 255, -1)
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
        (int(width * 0.38), int(height * 0.18)),
        (int(width * 0.62), int(height * 0.18)),
        (int(width * 0.57), int(height * 0.45)),
        (int(width * 0.74), int(height * 0.82)),
        (int(width * 0.26), int(height * 0.82)),
        (int(width * 0.43), int(height * 0.45)),
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    image[mask > 0] = foreground
    return image, mask, _contour_from_mask(mask)


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
