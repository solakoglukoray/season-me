"""Core analysis: detect personal color season from a portrait image."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .colors import SEASON_DESCRIPTIONS, SEASON_PALETTES
from .face import detect_face_region, sample_skin_pixels
from .model import predict_season, rgb_pixels_to_lab_mean


@dataclass
class SeasonResult:
    """Result of a personal color season analysis."""

    season: str
    description: str
    palette: list[dict]
    skin_tone_hex: str
    lab_L: float
    lab_a: float
    lab_b: float


def analyze(image_path: str | Path) -> SeasonResult:
    """Analyze a portrait photo and return the personal color season.

    Uses an SVM classifier trained on CIE Lab skin-tone features to
    predict one of four personal color seasons: Spring, Summer, Autumn,
    or Winter. No external API calls — fully offline.

    Args:
        image_path: Path to a JPG or PNG selfie/portrait.

    Returns:
        SeasonResult with season name, description, palette, and Lab metrics.
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    # Warn on very small images — analysis unreliable below ~400px on short side
    h_img, w_img = img.shape[:2]
    if min(h_img, w_img) < 200:
        raise ValueError(
            f"Image too small ({w_img}×{h_img}). "
            "Use a photo at least 400px on the short side for reliable results."
        )

    face_box = detect_face_region(img)
    if face_box is None:
        # No face detected — fall back to center region of image
        face_box = (w_img // 4, h_img // 4, w_img // 2, h_img // 2)

    skin_pixels = sample_skin_pixels(img, face_box)

    # Cap to 5000 pixels — median is stable with far fewer; avoids skew from
    # large face boxes that include non-skin boundary pixels
    if len(skin_pixels) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(skin_pixels), size=5000, replace=False)
        skin_pixels = skin_pixels[idx]

    # Predict season using the trained SVM (CIE Lab features)
    season = predict_season(skin_pixels)

    # Compute average skin tone hex from RGB mean
    avg_rgb = np.clip(np.mean(skin_pixels, axis=0), 0, 255).astype(int)
    r0, g0, b0 = int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])
    skin_hex = "#{:02x}{:02x}{:02x}".format(r0, g0, b0)

    # Lab features for display
    lab = rgb_pixels_to_lab_mean(skin_pixels)

    return SeasonResult(
        season=season,
        description=SEASON_DESCRIPTIONS[season],
        palette=SEASON_PALETTES[season],
        skin_tone_hex=skin_hex,
        lab_L=round(float(lab[0]), 1),
        lab_a=round(float(lab[1]), 2),
        lab_b=round(float(lab[2]), 2),
    )
