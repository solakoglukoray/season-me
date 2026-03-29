"""Face detection and skin pixel sampling using OpenCV Haar cascades."""

from pathlib import Path

import cv2
import numpy as np

_CASCADE_PATH = (
    Path(cv2.__file__).parent / "data" / "haarcascade_frontalface_default.xml"
)


def detect_face_region(
    img_rgb: np.ndarray,
) -> tuple[int, int, int, int] | None:
    """Detect the largest face in an RGB image.

    Returns:
        (x, y, w, h) bounding box of the largest detected face, or None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    x, y, w, h = faces[int(np.argmax(areas))]
    return int(x), int(y), int(w), int(h)


def sample_skin_pixels(
    img_rgb: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Sample pixels from the forehead and cheek regions of the face.

    Avoids hair (top edge) and eyes/mouth (center strip) to get
    representative skin tone pixels.

    Returns:
        Array of shape (N, 3) with RGB pixel values.
    """
    x, y, w, h = face_box

    # Forehead: top 20-30% of face, center 50% width (avoids hairline)
    forehead = img_rgb[
        y + int(h * 0.08) : y + int(h * 0.28),
        x + w // 4 : x + 3 * w // 4,
    ]

    # Left cheek: 45-70% height, left 10-30% width
    left_cheek = img_rgb[
        y + int(h * 0.45) : y + int(h * 0.70),
        x + int(w * 0.05) : x + int(w * 0.30),
    ]

    # Right cheek: 45-70% height, right 70-95% width
    right_cheek = img_rgb[
        y + int(h * 0.45) : y + int(h * 0.70),
        x + int(w * 0.70) : x + int(w * 0.95),
    ]

    regions = [r for r in (forehead, left_cheek, right_cheek) if r.size > 0]
    if not regions:
        # Fallback: entire face box
        return img_rgb[y : y + h, x : x + w].reshape(-1, 3)

    return np.vstack([r.reshape(-1, 3) for r in regions])
