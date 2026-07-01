import cv2
import numpy as np

from arrow_score.diff_detector import BasicFrameDiffDetector
from arrow_score.types import Calibration


def test_basic_frame_diff_detector_finds_added_line() -> None:
    prev = np.full((200, 200, 3), 255, dtype=np.uint8)
    curr = prev.copy()
    cv2.line(curr, (40, 100), (160, 100), (0, 0, 0), 3)
    calibration = Calibration("90", (100.0, 100.0), (80.0, 80.0), 0.0)

    output = BasicFrameDiffDetector().detect(prev, curr, calibration)

    assert output.mask.shape == prev.shape[:2]
    assert output.changed_area > 0
    assert len(output.bounding_boxes) >= 1
