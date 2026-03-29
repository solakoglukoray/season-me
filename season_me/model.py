"""Load and run the pre-trained SVM color season classifier.

The model operates in CIE Lab color space, which is perceptually uniform
and much better suited to skin tone analysis than HSL/HSV/RGB.

Model input:  [L*, a*, b*]  (CIE Lab values, float)
Model output: season name   ("Spring" | "Summer" | "Autumn" | "Winter")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_MODEL_PATH = Path(__file__).parent / "data" / "classifier.pkl"
_pipeline: Any = None  # lazy-loaded singleton


def _load_pipeline() -> Any:
    global _pipeline  # noqa: PLW0603
    if _pipeline is None:
        with open(_MODEL_PATH, "rb") as f:
            _pipeline = pickle.load(f)  # noqa: S301
    return _pipeline


def rgb_pixels_to_lab_mean(pixels_rgb: np.ndarray) -> np.ndarray:
    """Convert an array of RGB pixels to their mean CIE Lab values.

    Args:
        pixels_rgb: shape (N, 3), dtype uint8, values 0–255

    Returns:
        1-D array [L*, a*, b*] in CIE Lab space.
    """
    pixels_u8 = pixels_rgb.astype(np.uint8).reshape(1, -1, 3)
    # OpenCV expects BGR order for cvtColor
    pixels_bgr = cv2.cvtColor(pixels_u8, cv2.COLOR_RGB2BGR)
    lab_opencv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)

    # OpenCV scales Lab: L 0-255 (= 0-100), a/b 0-255 (= -128 to +127)
    L = lab_opencv[:, 0].astype(float) * 100.0 / 255.0
    a = lab_opencv[:, 1].astype(float) - 128.0
    b = lab_opencv[:, 2].astype(float) - 128.0

    return np.array([L.mean(), a.mean(), b.mean()], dtype=np.float32)


def predict_season(pixels_rgb: np.ndarray) -> str:
    """Predict personal color season from RGB skin pixels.

    Args:
        pixels_rgb: shape (N, 3) array of sampled skin pixels.

    Returns:
        Predicted season: "Spring", "Summer", "Autumn", or "Winter".
    """
    lab_features = rgb_pixels_to_lab_mean(pixels_rgb).reshape(1, -1)
    pipeline = _load_pipeline()
    return str(pipeline.predict(lab_features)[0])
