from __future__ import annotations

import io
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image, ExifTags


ORIENTATION_TAG = next(
    # Resolve the numeric EXIF key for "Orientation" once at import time.
    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    # Wrap raw bytes without copying, then let OpenCV decode compressed image data.
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _blur_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Variance of Laplacian is a standard focus metric: lower variance means blurrier image.
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _glare_ratio(img_bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Glare = very bright AND low saturation (washed-out)
    glare_mask = (v > 240) & (s < 40)

    # Convert to uint8 mask for cleanup
    mask = glare_mask.astype(np.uint8) * 255

    # Remove noise (small specks)
    mask = cv2.medianBlur(mask, 5)

    # Optional: merge nearby glare regions
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    glare_pixels = np.sum(mask == 255)
    return float(glare_pixels / mask.size)


def _orientation_hint(image_bytes: bytes) -> tuple[bool, str | None]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Pillow returns EXIF as a key/value mapping using numeric tag IDs.
        exif = image.getexif()
        if not exif or ORIENTATION_TAG is None:
            return False, None

        orientation = exif.get(ORIENTATION_TAG)
        if orientation in (3, 6, 8):
            return True, f"EXIF orientation={orientation}"
        return False, None
    except Exception:
        return False, None


def local_quality_checks(image_bytes: bytes) -> dict[str, Any]:
    img = _decode_image(image_bytes)
    blur = _blur_score(img)
    glare = _glare_ratio(img)
    needs_rotation, hint = _orientation_hint(image_bytes)

    blur_threshold = float(os.getenv("BLUR_THRESHOLD", "80.0"))
    glare_threshold = float(os.getenv("GLARE_THRESHOLD", "0.02"))

    warnings: list[str] = []
    if blur < blur_threshold:
        warnings.append("Image appears blurry; please retake with better focus.")
    if glare > glare_threshold:
        warnings.append("Image has significant glare; avoid reflective lighting.")
    if needs_rotation:
        warnings.append("Image orientation metadata indicates rotation is needed.")

    return {
        "is_blurry": blur < blur_threshold,
        "blur_score": round(blur, 2),
        "has_glare": glare > glare_threshold,
        "glare_ratio": round(glare, 4),
        "needs_rotation": needs_rotation,
        "orientation_hint": hint,
        "warnings": warnings,
    }
