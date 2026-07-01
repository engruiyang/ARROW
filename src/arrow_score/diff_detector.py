"""Basic OpenCV frame differencing for the TASK 1B/1C demo pipeline."""

from dataclasses import dataclass

import cv2
import numpy as np

from arrow_score.config import DiffConfig
from arrow_score.types import Calibration


@dataclass(frozen=True)
class DiffOutput:
    """Mask, contours, and bounding boxes produced by frame differencing."""

    mask: np.ndarray
    bounding_boxes: list[tuple[int, int, int, int]]
    changed_area: int
    contours: list[np.ndarray]
    roi_offset: tuple[int, int] = (0, 0)
    debug_reason: str | None = None


class BasicFrameDiffDetector:
    """Detect changed regions between two BGR images using simple differencing."""

    def __init__(self, config: DiffConfig | None = None) -> None:
        self.config = config or DiffConfig()

    def detect(self, prev_image: np.ndarray, curr_image: np.ndarray, calibration: Calibration) -> DiffOutput:
        """Return a binary change mask plus original-image boxes and contours."""
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
        kernel_size = (self.config.blur_kernel, self.config.blur_kernel)
        prev_blur = cv2.GaussianBlur(prev_gray, kernel_size, 0)
        curr_blur = cv2.GaussianBlur(curr_gray, kernel_size, 0)
        diff = cv2.absdiff(prev_blur, curr_blur)
        _, mask = cv2.threshold(diff, self.config.threshold, 255, cv2.THRESH_BINARY)

        morph_kernel = np.ones((self.config.morph_kernel, self.config.morph_kernel), dtype=np.uint8)
        if self.config.erode_iterations:
            mask = cv2.erode(mask, morph_kernel, iterations=self.config.erode_iterations)
        if self.config.dilate_iterations:
            mask = cv2.dilate(mask, morph_kernel, iterations=self.config.dilate_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

        changed_area = int(cv2.countNonZero(mask))
        if changed_area == 0:
            return DiffOutput(mask=mask, bounding_boxes=[], changed_area=0, contours=[], roi_offset=(x_offset, y_offset), debug_reason="no_change")

        contours_roi, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        contours: list[np.ndarray] = []
        offset = np.array([[[x_offset, y_offset]]], dtype=np.int32)
        for contour in contours_roi:
            area = cv2.contourArea(contour)
            if area < self.config.min_area:
                continue
            contour_full = contour.astype(np.int32) + offset
            x, y, w, h = cv2.boundingRect(contour_full)
            boxes.append((int(x), int(y), int(w), int(h)))
            contours.append(contour_full)

        boxes_and_contours = sorted(zip(boxes, contours, strict=True), key=lambda item: (item[0][1], item[0][0]))
        if boxes_and_contours:
            boxes = [item[0] for item in boxes_and_contours]
            contours = [item[1] for item in boxes_and_contours]
        return DiffOutput(mask=mask, bounding_boxes=boxes, changed_area=changed_area, contours=contours, roi_offset=(x_offset, y_offset))
