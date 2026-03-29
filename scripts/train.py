"""Train and export the personal color season classifier.

Generates synthetic skin-tone samples in CIE Lab color space using
well-documented color-season ranges from color theory literature,
then trains an SVM classifier and serializes it to
season_me/data/classifier.pkl.

Run:
    python scripts/train.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
N_PER_SEASON = 1200

# CIE Lab color ranges per season derived from color theory:
#   L*  = perceptual lightness (0 = black, 100 = white)
#   a*  = green (−) to red/magenta (+)
#   b*  = blue (−) to yellow (+)
#
# Warm skin tones → higher b* (yellow-orange shift)
# Cool skin tones → lower b*, higher a* (pink-red shift)
# Light seasons → higher L*; deep seasons → lower L*
SEASON_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "Spring": {
        # warm undertone (yellow-golden), light depth, clear saturation
        "L": (58.0, 84.0),
        "a": (7.0, 17.0),
        "b": (14.0, 28.0),
    },
    "Summer": {
        # cool undertone (pink-ash), light depth, muted saturation
        "L": (58.0, 84.0),
        "a": (7.0, 16.0),
        "b": (3.0, 14.0),
    },
    "Autumn": {
        # warm undertone (earthy-golden), deep/medium depth, muted saturation
        "L": (30.0, 62.0),
        "a": (9.0, 21.0),
        "b": (14.0, 30.0),
    },
    "Winter": {
        # cool undertone (rosy-neutral), deep/medium depth, clear saturation
        "L": (26.0, 60.0),
        "a": (8.0, 19.0),
        "b": (1.0, 13.0),
    },
}

SEASON_LABELS = ["Spring", "Summer", "Autumn", "Winter"]


def generate_samples(
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic Lab skin-tone samples with per-season Gaussian noise."""
    X: list[list[float]] = []
    y: list[str] = []

    for season, ranges in SEASON_RANGES.items():
        l_lo, l_hi = ranges["L"]
        a_lo, a_hi = ranges["a"]
        b_lo, b_hi = ranges["b"]

        l_samples = rng.uniform(l_lo, l_hi, N_PER_SEASON)
        a_samples = rng.uniform(a_lo, a_hi, N_PER_SEASON)
        b_samples = rng.uniform(b_lo, b_hi, N_PER_SEASON)

        # Add Gaussian noise to simulate real-world measurement variation
        noise_scale = 1.5
        l_samples += rng.normal(0, noise_scale, N_PER_SEASON)
        a_samples += rng.normal(0, noise_scale * 0.5, N_PER_SEASON)
        b_samples += rng.normal(0, noise_scale * 0.5, N_PER_SEASON)

        for lv, av, bv in zip(l_samples, a_samples, b_samples):
            X.append([lv, av, bv])
            y.append(season)

    return np.array(X, dtype=np.float32), y


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=SEED,
                ),
            ),
        ]
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    X, y = generate_samples(rng)

    pipeline = build_pipeline()

    # Cross-validate before final fit
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)

    out_path = Path(__file__).parent.parent / "season_me" / "data" / "classifier.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model saved: {out_path}")


if __name__ == "__main__":
    main()
