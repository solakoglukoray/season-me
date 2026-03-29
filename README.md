# season-me

[![CI](https://github.com/solakoglukoray/season-me/actions/workflows/ci.yml/badge.svg)](https://github.com/solakoglukoray/season-me/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Drop in a selfie — get your color season (Spring / Summer / Autumn / Winter) and an 8-color flattering palette with hex codes, all from your terminal.

Personal color analysis tells you exactly which shades make you look vibrant vs washed out. Stylists charge hundreds for this. `season-me` does it offline, in seconds, with a trained ML model — no API keys, no cloud, no cost.

## How It Works

1. **Face detection** — OpenCV Haar cascade locates your face using multiple strategies (handles low-res selfies and close-up shots)
2. **Skin sampling** — Samples pixels from forehead and cheeks; a CIE Lab filter removes highlights, shadows, and hair pixels
3. **Color analysis** — Converts skin pixels to CIE Lab color space (perceptually uniform, far more accurate than HSL/HSV for skin tones)
4. **Season classification** — A pre-trained SVM classifier (94.3% cross-validation accuracy, trained on 4800 synthetic skin-tone samples from color theory literature) maps your Lab values to one of four seasons
5. **Palette output** — Prints your 8-color season palette with hex codes you can paste into any design tool

## Installation

```bash
pip install season-me
```

Or clone and install locally:

```bash
git clone https://github.com/solakoglukoray/season-me
cd season-me
pip install -e .
```

Or with Docker:

```bash
docker run --rm -v $(pwd):/photos ghcr.io/solakoglukoray/season-me /photos/portrait.jpg
```

## Usage

```bash
# Basic analysis
python -m season_me.cli portrait.jpg

# Show raw CIE Lab skin tone metrics
python -m season_me.cli portrait.jpg --verbose
```

If `season-me` is on your PATH:

```bash
season-me portrait.jpg
season-me portrait.jpg --verbose
```

**Example output:**

```
Analyzing portrait.jpg...

+----------------------------- Your Color Season -----------------------------+
| Autumn                                                                      |
|                                                                             |
| Warm, deep, and muted - your natural coloring has golden or earthy          |
| undertones with rich, complex tones. Rich, earthy, and muted shades that    |
| echo autumn foliage suit you perfectly.                                     |
+-----------------------------------------------------------------------------+

Detected skin tone: #926652

            Autumn Palette
+------------------------------------+
|  Swatch  | Color Name   | Hex Code |
|----------+--------------+----------|
|    ##    | Terracotta   | #E2725B  |
|    ##    | Rust         | #B7410E  |
|    ##    | Olive        | #808000  |
|    ##    | Deep Teal    | #008080  |
|    ##    | Burnt Orange | #CC5500  |
|    ##    | Warm Brown   | #964B00  |
|    ##    | Moss Green   | #8A9A5B  |
|    ##    | Caramel      | #C68642  |
+------------------------------------+
```

## Seasons

| Season | Undertone | Depth | Example |
|--------|-----------|-------|---------|
| **Spring** | Warm (golden) | Light | Jennifer Aniston, Blake Lively |
| **Summer** | Cool (pink) | Light | Cate Blanchett, Nicole Kidman |
| **Autumn** | Warm (earthy) | Deep | Beyoncé, Jennifer Lopez |
| **Winter** | Cool (pink/blue) | Deep | Lucy Liu, Lupita Nyong'o |

## Tips for Best Results

- **Photo size** — At least 400×400 px; tiny thumbnails give unreliable results
- **Lighting** — Neutral daylight or soft indoor white light; avoid golden-hour sun, flash, or colored ambient light
- **Skin state** — Analyze your natural skin tone, not a summer tan; tan shifts results toward warmer seasons
- **Pose** — Forward-facing, unobstructed face; the detector handles most selfie angles
- **No filters** — Avoid heavy color grading or beauty filters

## Retrain the Model

The classifier ships as `season_me/data/classifier.pkl`. To retrain from scratch:

```bash
pip install -e ".[dev]"
python scripts/train.py
```

## Development

```bash
git clone https://github.com/solakoglukoray/season-me
cd season-me
pip install -e ".[dev]"
pytest
```

## Contributing

PRs welcome. Run `ruff check .` and `pytest` before submitting.

## License

MIT
