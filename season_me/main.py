"""Core analysis: detect personal color season from a portrait image."""

import colorsys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .colors import SEASON_DESCRIPTIONS, SEASON_PALETTES, classify_season
from .face import detect_face_region, sample_skin_pixels


@dataclass
class SeasonResult:
    """Result of a personal color season analysis."""

    season: str
    description: str
    palette: list[dict]
    skin_tone_hex: str
    hue: float
    saturation: float
    lightness: float


def analyze(image_path: str | Path) -> SeasonResult:
    """Analyze a portrait photo and return the personal color season.

    Args:
        image_path: Path to a JPG or PNG selfie/portrait.

    Returns:
        SeasonResult with season name, description, palette, and metrics.
    """
    img = np.array(Image.open(image_path).convert("RGB"))

    face_box = detect_face_region(img)
    if face_box is None:
        # No face detected — fall back to center region of image
        h, w = img.shape[:2]
        face_box = (w // 4, h // 4, w // 2, h // 2)

    skin_pixels = sample_skin_pixels(img, face_box)
    avg_rgb = np.clip(np.mean(skin_pixels, axis=0), 0, 255).astype(int)

    r0, g0, b0 = int(avg_rgb[0]), int(avg_rgb[1]), int(avg_rgb[2])
    skin_hex = "#{:02x}{:02x}{:02x}".format(r0, g0, b0)

    r, g, b = (c / 255.0 for c in avg_rgb)
    h_val, l_val, s_val = colorsys.rgb_to_hls(r, g, b)

    season = classify_season(h_val * 360, s_val, l_val)

    return SeasonResult(
        season=season,
        description=SEASON_DESCRIPTIONS[season],
        palette=SEASON_PALETTES[season],
        skin_tone_hex=skin_hex,
        hue=round(h_val * 360, 1),
        saturation=round(s_val, 3),
        lightness=round(l_val, 3),
    )
