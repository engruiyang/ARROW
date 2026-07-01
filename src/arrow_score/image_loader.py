"""Image loading helpers that convert path-level frames into arrays."""

from pathlib import Path

import cv2
import numpy as np

from arrow_score.types import Frame, ImagePair


def load_image(path: Path) -> np.ndarray:
    """Load an image from disk as a BGR NumPy array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image as BGR data: {path}")
    return image


def load_frame_image(frame: Frame) -> Frame:
    """Return a frame with its image field populated from frame.path."""
    return Frame(path=frame.path, image=load_image(frame.path))


def load_pair_images(pair: ImagePair) -> ImagePair:
    """Return an image pair with both frame images loaded."""
    return ImagePair(prev=load_frame_image(pair.prev), curr=load_frame_image(pair.curr), index=pair.index)
