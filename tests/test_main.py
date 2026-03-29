"""Tests for season-me core logic."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from season_me.colors import SEASON_DESCRIPTIONS, SEASON_PALETTES
from season_me.main import SeasonResult, analyze
from season_me.model import predict_season, rgb_pixels_to_lab_mean

# ── CIE Lab conversion ───────────────────────────────────────────────────────


def test_lab_conversion_returns_three_values():
    pixels = np.array([[200, 150, 120]], dtype=np.uint8)
    lab = rgb_pixels_to_lab_mean(pixels)
    assert lab.shape == (3,)


def test_lab_L_in_valid_range():
    pixels = np.array([[200, 160, 130]] * 10, dtype=np.uint8)
    lab = rgb_pixels_to_lab_mean(pixels)
    assert 0.0 <= lab[0] <= 100.0, f"L* out of range: {lab[0]}"


def test_lab_warm_tone_positive_b():
    # Yellow-orange skin tone should have b* > 0 (warm)
    pixels = np.array([[220, 170, 120]] * 20, dtype=np.uint8)
    lab = rgb_pixels_to_lab_mean(pixels)
    assert lab[2] > 0, f"Expected b* > 0 for warm skin, got {lab[2]}"


# ── Model prediction ─────────────────────────────────────────────────────────


def test_predict_returns_valid_season():
    pixels = np.array([[200, 155, 120]] * 50, dtype=np.uint8)
    season = predict_season(pixels)
    assert season in {"Spring", "Summer", "Autumn", "Winter"}


def test_predict_light_warm_is_spring_or_summer():
    # Light, warm-toned pixels should classify as Spring (not Autumn/Winter)
    pixels = np.array([[230, 190, 155]] * 100, dtype=np.uint8)
    season = predict_season(pixels)
    assert season in {"Spring", "Summer"}


def test_predict_dark_warm_is_autumn():
    # Deep, warm-toned pixels should map to Autumn
    pixels = np.array([[140, 95, 65]] * 100, dtype=np.uint8)
    season = predict_season(pixels)
    assert season == "Autumn"


# ── Palette and description completeness ────────────────────────────────────


def test_all_seasons_have_palettes():
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        assert season in SEASON_PALETTES
        assert len(SEASON_PALETTES[season]) >= 4
        for entry in SEASON_PALETTES[season]:
            assert "name" in entry
            assert "hex" in entry
            assert entry["hex"].startswith("#"), f"Bad hex: {entry['hex']}"


def test_all_seasons_have_descriptions():
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        assert season in SEASON_DESCRIPTIONS
        assert len(SEASON_DESCRIPTIONS[season]) > 20


# ── analyze() integration tests ─────────────────────────────────────────────


def _make_image(
    color: tuple[int, int, int],
    size: tuple[int, int] = (200, 200),
) -> Path:
    """Create a solid-color temporary image and return its path."""
    img = Image.new("RGB", size, color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return Path(tmp.name)


def test_analyze_returns_season_result():
    # warm peach — no real face, uses center region fallback
    path = _make_image((210, 160, 120))
    result = analyze(path)
    assert isinstance(result, SeasonResult)


def test_analyze_season_is_valid():
    path = _make_image((200, 150, 130))
    result = analyze(path)
    assert result.season in {"Spring", "Summer", "Autumn", "Winter"}


def test_analyze_skin_hex_format():
    path = _make_image((180, 140, 110))
    result = analyze(path)
    assert result.skin_tone_hex.startswith("#")
    assert len(result.skin_tone_hex) == 7


def test_analyze_lab_metrics_in_range():
    path = _make_image((190, 155, 125))
    result = analyze(path)
    assert 0.0 <= result.lab_L <= 100.0
    assert -128.0 <= result.lab_a <= 127.0
    assert -128.0 <= result.lab_b <= 127.0


def test_analyze_palette_not_empty():
    path = _make_image((200, 160, 130))
    result = analyze(path)
    assert len(result.palette) >= 4


def test_analyze_description_not_empty():
    path = _make_image((200, 160, 130))
    result = analyze(path)
    assert len(result.description) > 10
