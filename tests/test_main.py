"""Tests for season-me core logic."""

import tempfile
from pathlib import Path

from PIL import Image

from season_me.colors import (
    SEASON_DESCRIPTIONS,
    SEASON_PALETTES,
    classify_season,
)
from season_me.main import SeasonResult, analyze

# ── classify_season unit tests ──────────────────────────────────────────────


def test_classify_spring():
    # Warm hue (35°), medium-high saturation, light skin
    assert classify_season(hue=35.0, saturation=0.40, lightness=0.65) == "Spring"


def test_classify_summer():
    # Cool hue (5°, pinkish), low saturation, light skin
    assert classify_season(hue=5.0, saturation=0.20, lightness=0.70) == "Summer"


def test_classify_autumn():
    # Warm hue (30°), low saturation, deep/medium skin
    assert classify_season(hue=30.0, saturation=0.25, lightness=0.40) == "Autumn"


def test_classify_winter():
    # Cool hue (5°, pinkish), higher saturation, deep skin
    assert classify_season(hue=5.0, saturation=0.55, lightness=0.35) == "Winter"


def test_classify_returns_valid_season():
    for hue in [0, 20, 35, 55, 180, 350]:
        for sat in [0.1, 0.4, 0.7]:
            for light in [0.3, 0.55, 0.75]:
                result = classify_season(hue, sat, light)
                assert result in {"Spring", "Summer", "Autumn", "Winter"}


# ── palette and description completeness ───────────────────────────────────


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


def test_analyze_metrics_in_range():
    path = _make_image((190, 155, 125))
    result = analyze(path)
    assert 0.0 <= result.hue <= 360.0
    assert 0.0 <= result.saturation <= 1.0
    assert 0.0 <= result.lightness <= 1.0


def test_analyze_palette_not_empty():
    path = _make_image((200, 160, 130))
    result = analyze(path)
    assert len(result.palette) >= 4


def test_analyze_description_not_empty():
    path = _make_image((200, 160, 130))
    result = analyze(path)
    assert len(result.description) > 10
