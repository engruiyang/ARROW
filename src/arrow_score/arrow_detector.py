"""Simple arrow candidate estimation from changed bounding boxes."""

import math

from arrow_score.diff_detector import DiffOutput
from arrow_score.types import ArrowCandidate, Calibration


class SimpleArrowDetector:
    """Estimate coarse arrow candidates from elongated changed regions."""

    def __init__(self, min_area: int = 20, min_aspect_ratio: float = 2.0) -> None:
        if min_area < 0:
            raise ValueError("min_area must be non-negative")
        if min_aspect_ratio <= 0:
            raise ValueError("min_aspect_ratio must be positive")
        self.min_area = min_area
        self.min_aspect_ratio = min_aspect_ratio

    def detect(self, diff_output: DiffOutput, calibration: Calibration) -> list[ArrowCandidate]:
        """Return coarse arrow candidates from diff bounding boxes."""
        raw_candidates: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
        for x, y, w, h in diff_output.bounding_boxes:
            area = w * h
            if area < self.min_area or w <= 0 or h <= 0:
                continue
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio < self.min_aspect_ratio:
                continue
            if w >= h:
                start = (float(x), float(y + h / 2))
                end = (float(x + w), float(y + h / 2))
            else:
                start = (float(x + w / 2), float(y))
                end = (float(x + w / 2), float(y + h))
            impact = min(start, end, key=lambda point: math.dist(point, calibration.center_px))
            raw_candidates.append((start, end, impact))

        if not raw_candidates:
            return []
        confidence = "medium" if len(raw_candidates) == 1 else "low"
        debug_reason = None if len(raw_candidates) == 1 else "multiple_candidates"
        return [
            ArrowCandidate(
                line_start_px=start,
                line_end_px=end,
                impact_point_px=impact,
                confidence=confidence,
                debug_reason=debug_reason,
            )
            for start, end, impact in raw_candidates
        ]
