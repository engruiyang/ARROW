"""Basic OpenCV frame differencing for the TASK 1B demo pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np

from arrow_score.types import Calibration


@dataclass(frozen=True)
class DiffOutput:
    """Mask and bounding boxes produced by frame differencing."""

    mask: np.ndarray
    bounding_boxes: list[tuple[int, int, int, int]]
    changed_area: int
    debug_reason: str | None = None


class BasicFrameDiffDetector:
    """Detect changed regions between two BGR images using simple differencing."""

    def __init__(self, blur_kernel: int = 5, threshold: int = 25, min_area: int = 20) -> None:
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if threshold < 0 or threshold > 255:
            raise ValueError("threshold must be between 0 and 255")
        if min_area < 0:
            raise ValueError("min_area must be non-negative")
        self.blur_kernel = blur_kernel
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, prev_image: np.ndarray, curr_image: np.ndarray, calibration: Calibration) -> DiffOutput:
        """Return a binary change mask and original-image bounding boxes."""
        if prev_image.shape != curr_image.shape:
            raise ValueError(f"Image shapes must match, got {prev_image.shape} and {curr_image.shape}")

        x_offset = 0
        y_offset = 0
        prev_roi = prev_image
        curr_roi = curr_image
        if calibration.roi is not None:
            x, y, w, h = calibration.roi
            if w <= 0 or h <= 0:
                raise ValueError(f"Calibration ROI width and height must be positive, got {calibration.roi}")
            x_offset, y_offset = x, y
            prev_roi = prev_image[y : y + h, x : x + w]
            curr_roi = curr_image[y : y + h, x : x + w]
            if prev_roi.size == 0 or curr_roi.size == 0:
                raise ValueError(f"Calibration ROI is outside the image: {calibration.roi}")

        prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)
        prev_blur = cv2.GaussianBlur(prev_gray, (self.blur_kernel, self.blur_kernel), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (self.blur_kernel, self.blur_kernel), 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        changed_area = 0
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x + x_offset, y + y_offset, w, h))
            changed_area += area
        boxes.sort(key=lambda box: (box[1], box[0]))
        return DiffOutput(mask=mask, bounding_boxes=boxes, changed_area=changed_area)
