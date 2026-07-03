"""@file gradients.py
@brief Traditional gradient operators used by IEPS.
"""

from __future__ import annotations

import cv2
import numpy as np


def sobel_gradient_magnitude(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """@brief Compute Sobel gradient magnitude.

    @param image Input grayscale image.
    @param normalize If True, normalize magnitude to 0..255.
    @return Gradient magnitude as float32 image.
    """
    gray = image.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    if normalize:
        max_val = float(mag.max())
        if max_val > 0:
            mag = (mag / max_val) * 255.0
    return mag.astype(np.float32)
