# season-me

[![CI](https://github.com/solakoglukoray/season-me/actions/workflows/ci.yml/badge.svg)](https://github.com/solakoglukoray/season-me/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Drop in a selfie — get your color season (Spring / Summer / Autumn / Winter) and an 8-color flattering palette with hex codes, all from your terminal.

Personal color analysis tells you exactly which shades make you look vibrant vs washed out. Stylists charge hundreds for this. `season-me` does it offline, in seconds, using computer vision and color theory.

## How It Works

1. **Face detection** — OpenCV Haar cascade locates your face in the photo
2. **Skin sampling** — Samples pixels from your forehead and cheeks (avoids hair, eyes, lips)
3. **Color analysis** — Computes dominant skin tone in HSL space
4. **Season classification** — Maps undertone (warm/cool) × depth (light/deep) to your season
5. **Palette output** — Prints your 8-color season palette with hex codes you can paste anywhere

## Installation

```bash
pip install season-me
```

Or with Docker:

```bash
docker run --rm -v $(pwd):/photos ghcr.io/solakoglukoray/season-me /photos/portrait.jpg
```

## Usage

```bash
# Basic analysis
season-me portrait.jpg

# Show raw skin tone metrics
season-me portrait.jpg --verbose
```

**Example output:**

```
Analyzing portrait.jpg...

╭─────────────── Your Color Season ───────────────╮
│  Autumn                                          │
│                                                  │
│  Warm, deep, and muted — your natural coloring   │
│  has golden or earthy undertones...              │
╰──────────────────────────────────────────────────╯

Detected skin tone: #c8956a

          Autumn Palette
┌────────┬──────────────┬──────────┐
│ Swatch │  Color Name  │ Hex Code │
├────────┼──────────────┼──────────┤
│   ██   │ Terracotta   │ #E2725B  │
│   ██   │ Rust         │ #B7410E  │
│   ██   │ Burnt Orange │ #CC5500  │
│   ██   │ Caramel      │ #C68642  │
└────────┴──────────────┴──────────┘
```

## Seasons

| Season | Undertone | Depth | Example |
|--------|-----------|-------|---------|
| **Spring** | Warm (golden) | Light | Jennifer Aniston, Blake Lively |
| **Summer** | Cool (pink) | Light | Cate Blanchett, Nicole Kidman |
| **Autumn** | Warm (earthy) | Deep | Beyoncé, Jennifer Lopez |
| **Winter** | Cool (pink/blue) | Deep | Lucy Liu, Lupita Nyong'o |

## Tips for Best Results

- Use a well-lit, forward-facing portrait photo
- Avoid heavy filters or color grading
- Natural light photos work best
- The tool falls back to center-region sampling if no face is detected

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
