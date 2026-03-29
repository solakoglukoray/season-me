"""Face detection and skin pixel sampling using OpenCV Haar cascades."""

from pathlib import Path

import cv2
import numpy as np

_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)


def _try_detect(
    gray: np.ndarray,
    scale_factor: float,
    min_size: int,
    min_neighbors: int = 4,
) -> np.ndarray:
    cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
    )
    return faces if len(faces) > 0 else np.array([])


def detect_face_region(
    img_rgb: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Detect the largest face in an RGB image.

    Tries multiple strategies to handle selfies (face fills frame) and
    standard portrait photos.

    Returns:
        (x, y, w, h) bounding box of the largest detected face, or None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape

    # Strategy 1: standard detection
    faces = _try_detect(gray, scale_factor=1.1, min_size=60)

    # Strategy 2: small image / small face (low-res selfies)
    if len(faces) == 0:
        faces = _try_detect(gray, scale_factor=1.05, min_size=30, min_neighbors=3)

    # Strategy 3: selfie / close-up — face often too large for Haar,
    # try on a downscaled copy then map coordinates back
    if len(faces) == 0:
        scale = 0.4
        small = cv2.resize(gray, (int(w_img * scale), int(h_img * scale)))
        small_faces = _try_detect(small, scale_factor=1.05, min_size=20)
        if len(small_faces) > 0:
            faces = np.array(
                [
                    [
                        int(x / scale),
                        int(y / scale),
                        int(w / scale),
                        int(h / scale),
                    ]
                    for (x, y, w, h) in small_faces
                ]
            )

    # Strategy 4: very relaxed parameters
    if len(faces) == 0:
        faces = _try_detect(gray, scale_factor=1.05, min_size=20)

    if len(faces) == 0:
        return None

    areas = [w * h for (x, y, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]
    return int(x), int(y), int(w), int(h)


def _filter_skin_pixels(pixels_rgb: np.ndarray) -> np.ndarray:
    """Remove non-skin pixels (highlights, shadows, hair, background).

    Uses CIE Lab thresholds derived from skin tone literature to keep
    only plausible skin-tone pixels.
    """
    pixels_u8 = pixels_rgb.astype(np.uint8).reshape(1, -1, 3)
    bgr = cv2.cvtColor(pixels_u8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

    L = lab[:, 0] * 100.0 / 255.0
    a = lab[:, 1] - 128.0
    b = lab[:, 2] - 128.0

    # Skin tone constraints in CIE Lab:
    #   L 22–86: exclude pure shadows and specular highlights
    #   a  4–27: skin has a reddish/warm component
    #   b  5–35: skin has a yellowish component
    mask = (
        (L >= 22) & (L <= 86)
        & (a >= 4) & (a <= 27)
        & (b >= 5) & (b <= 35)
    )
    filtered = pixels_rgb[mask]
    # Fall back to unfiltered if filter is too aggressive
    return filtered if len(filtered) >= 20 else pixels_rgb


def sample_skin_pixels(
    img_rgb: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Sample skin-tone pixels from the forehead and cheek regions.

    Samples targeted sub-regions of the face (avoids hairline, eyes,
    mouth) and then filters out highlights, shadows, and hair pixels
    using CIE Lab skin-tone constraints.

    Returns:
        Array of shape (N, 3) with filtered RGB skin-tone pixels.
    """
    x, y, w, h = face_box

    # Forehead: 10-28% height, center 50% width (avoids hairline)
    forehead = img_rgb[
        y + int(h * 0.10) : y + int(h * 0.28),
        x + w // 4 : x + 3 * w // 4,
    ]

    # Left cheek: 45-68% height, left 8-28% width
    left_cheek = img_rgb[
        y + int(h * 0.45) : y + int(h * 0.68),
        x + int(w * 0.08) : x + int(w * 0.28),
    ]

    # Right cheek: 45-68% height, right 72-92% width
    right_cheek = img_rgb[
        y + int(h * 0.45) : y + int(h * 0.68),
        x + int(w * 0.72) : x + int(w * 0.92),
    ]

    regions = [r for r in (forehead, left_cheek, right_cheek) if r.size > 0]
    if not regions:
        raw = img_rgb[y : y + h, x : x + w].reshape(-1, 3)
        return _filter_skin_pixels(raw)

    raw = np.vstack([r.reshape(-1, 3) for r in regions])
    return _filter_skin_pixels(raw)
