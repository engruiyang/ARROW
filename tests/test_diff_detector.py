import cv2
import numpy as np
import pytest

from arrow_score.config import DiffConfig
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
    assert len(output.contours) >= 1


def test_basic_frame_diff_detector_roi_offsets_bbox_to_full_image() -> None:
    prev = np.full((200, 200, 3), 255, dtype=np.uint8)
    curr = prev.copy()
    cv2.line(curr, (80, 90), (140, 90), (0, 0, 0), 3)
    calibration = Calibration("90", (100.0, 100.0), (80.0, 80.0), 0.0, roi=(50, 50, 120, 80))

    output = BasicFrameDiffDetector().detect(prev, curr, calibration)

    assert output.roi_offset == (50, 50)
    assert output.bounding_boxes
    x, y, _w, _h = output.bounding_boxes[0]
    assert x >= 50
    assert y >= 50


def test_basic_frame_diff_detector_no_change_returns_empty_outputs() -> None:
    prev = np.full((200, 200, 3), 255, dtype=np.uint8)
    curr = prev.copy()
    calibration = Calibration("90", (100.0, 100.0), (80.0, 80.0), 0.0)

    output = BasicFrameDiffDetector().detect(prev, curr, calibration)

    assert output.changed_area == 0
    assert output.bounding_boxes == []
    assert output.contours == []


def test_diff_config_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        DiffConfig(blur_kernel=4)
    with pytest.raises(ValueError):
        DiffConfig(threshold=300)
    with pytest.raises(ValueError):
        DiffConfig(min_area=0)
