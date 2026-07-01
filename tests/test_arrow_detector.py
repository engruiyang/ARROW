import cv2
import numpy as np

from arrow_score.arrow_detector import SimpleArrowDetector
from arrow_score.config import ArrowDetectorConfig
from arrow_score.diff_detector import DiffOutput
from arrow_score.types import Calibration


def _contours_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def _diff_from_mask(mask: np.ndarray) -> DiffOutput:
    contours = _contours_from_mask(mask)
    boxes = [cv2.boundingRect(contour) for contour in contours]
    return DiffOutput(mask=mask, bounding_boxes=boxes, changed_area=int(cv2.countNonZero(mask)), contours=contours)


def _calibration() -> Calibration:
    return Calibration("90", (100.0, 100.0), (80.0, 80.0), 0.0)


def test_single_horizontal_contour_returns_candidate() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (40, 100), (160, 100), 255, 5)

    candidates = SimpleArrowDetector().detect(_diff_from_mask(mask), _calibration())

    assert len(candidates) == 1
    assert candidates[0].confidence == "medium"
    assert candidates[0].debug_reason is not None


def test_single_diagonal_contour_uses_fitline() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (50, 140), (150, 60), 255, 5)

    candidates = SimpleArrowDetector(ArrowDetectorConfig(min_aspect_ratio=1.1)).detect(_diff_from_mask(mask), _calibration())

    assert len(candidates) == 1
    assert candidates[0].line_start_px != candidates[0].line_end_px


def test_multiple_candidates_are_low_confidence() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (30, 80), (90, 80), 255, 5)
    cv2.line(mask, (110, 120), (170, 120), 255, 5)

    candidates = SimpleArrowDetector().detect(_diff_from_mask(mask), _calibration())

    assert len(candidates) == 2
    assert all(candidate.confidence == "low" for candidate in candidates)
    assert all(candidate.debug_reason and "multiple_candidates" in candidate.debug_reason for candidate in candidates)


def test_too_small_candidate_is_filtered() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (100, 100), (105, 100), 255, 1)

    candidates = SimpleArrowDetector(ArrowDetectorConfig(min_area=20)).detect(_diff_from_mask(mask), _calibration())

    assert candidates == []


def test_not_elongated_candidate_is_filtered() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (80, 80), (120, 120), 255, -1)

    candidates = SimpleArrowDetector(ArrowDetectorConfig(min_aspect_ratio=2.0)).detect(_diff_from_mask(mask), _calibration())

    assert candidates == []


def test_no_candidates_returns_empty_list() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    assert SimpleArrowDetector().detect(_diff_from_mask(mask), _calibration()) == []
