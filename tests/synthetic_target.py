"""Synthetic target image helpers for tests."""

import cv2
import numpy as np


def make_synthetic_target(size: tuple[int, int] = (980, 980), center: tuple[int, int] | None = None, radius: int = 420) -> np.ndarray:
    """Create a simple BGR 90-target-like colored image."""
    height, width = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cx, cy = center or (width // 2, height // 2)
    cv2.circle(image, (cx, cy), radius, (255, 0, 0), -1)  # blue
    cv2.circle(image, (cx, cy), int(radius * 0.62), (0, 0, 255), -1)  # red
    cv2.circle(image, (cx, cy), int(radius * 0.30), (0, 255, 255), -1)  # yellow
    cv2.line(image, (cx - 45, cy), (cx + 45, cy), (0, 0, 0), 5)
    cv2.line(image, (cx, cy - 45), (cx, cy + 45), (0, 0, 0), 5)
    return image
