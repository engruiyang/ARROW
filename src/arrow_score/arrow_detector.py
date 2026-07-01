"""Simple contour-based arrow candidate estimation."""

import math

import cv2
import numpy as np

from arrow_score.config import ArrowDetectorConfig
from arrow_score.diff_detector import DiffOutput
from arrow_score.types import ArrowCandidate, Calibration


class SimpleArrowDetector:
    """Estimate coarse arrow candidates from elongated contours using fitLine."""

    def __init__(self, config: ArrowDetectorConfig | None = None) -> None:
        self.config = config or ArrowDetectorConfig()

    def detect(self, diff_output: DiffOutput, calibration: Calibration) -> list[ArrowCandidate]:
        """Return arrow candidates sorted by a simple quality score."""
        scored_candidates: list[tuple[float, ArrowCandidate]] = []
        for contour in diff_output.contours:
            candidate = self._candidate_from_contour(contour, calibration)
            if candidate is not None:
                scored_candidates.append(candidate)

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = [candidate for _score, candidate in scored_candidates[: self.config.max_candidates]]
        if not candidates:
            return []
        if len(candidates) == 1:
            return candidates
        return [
            ArrowCandidate(
                line_start_px=candidate.line_start_px,
                line_end_px=candidate.line_end_px,
                impact_point_px=candidate.impact_point_px,
                confidence="low",
                debug_reason=f"multiple_candidates;{candidate.debug_reason}" if candidate.debug_reason else "multiple_candidates",
            )
            for candidate in candidates
        ]

    def _candidate_from_contour(self, contour: np.ndarray, calibration: Calibration) -> tuple[float, ArrowCandidate] | None:
        """Fit one contour to a line and return a scored candidate."""
        if contour.shape[0] < 2:
            return None
        area = float(cv2.contourArea(contour))
        if area < self.config.min_area:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < self.config.min_aspect_ratio:
            return None

        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 2:
            return None
        try:
            line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        except cv2.error:
            return None
        vx, vy, _x0, _y0 = (float(value) for value in line.reshape(-1))
        direction = np.array([vx, vy], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return None
        direction = direction / norm
        projections = points @ direction
        start = points[int(np.argmin(projections))]
        end = points[int(np.argmax(projections))]
        start_px = (float(start[0]), float(start[1]))
        end_px = (float(end[0]), float(end[1]))
        line_length = math.dist(start_px, end_px)
        if line_length < self.config.min_line_length:
            return None

        impact = min(start_px, end_px, key=lambda point: math.dist(point, calibration.center_px))
        quality_score = line_length * aspect_ratio
        debug_reason = f"area={area:.1f};aspect={aspect_ratio:.2f};line={line_length:.1f};quality={quality_score:.1f}"
        return (
            quality_score,
            ArrowCandidate(
                line_start_px=start_px,
                line_end_px=end_px,
                impact_point_px=impact,
                confidence="medium",
                debug_reason=debug_reason,
            ),
        )
