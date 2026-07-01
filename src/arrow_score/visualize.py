"""Minimal OpenCV visualization for scoring results."""

from pathlib import Path

import cv2
import numpy as np

from arrow_score.types import Calibration, ScoreResult, TargetSpec


class BasicVisualizer:
    """Draw target geometry and score results on a BGR image."""

    def draw_result(
        self,
        image: np.ndarray,
        calibration: Calibration,
        target_spec: TargetSpec,
        results: list[ScoreResult],
    ) -> np.ndarray:
        """Return a copy of image annotated with target rings and results."""
        output = image.copy()
        center = (int(round(calibration.center_px[0])), int(round(calibration.center_px[1])))
        axes_outer = (int(round(calibration.ellipse_axes_px[0])), int(round(calibration.ellipse_axes_px[1])))
        angle = float(calibration.ellipse_angle_deg)

        cv2.circle(output, center, 4, (0, 0, 255), -1)
        cv2.ellipse(output, center, axes_outer, angle, 0, 360, (255, 0, 0), 2)
        for boundary_mm, _score in target_spec.score_boundaries:
            scale = boundary_mm / target_spec.outer_radius_mm
            axes = (max(1, int(round(axes_outer[0] * scale))), max(1, int(round(axes_outer[1] * scale))))
            cv2.ellipse(output, center, axes, angle, 0, 360, (0, 180, 0), 1)

        for index, result in enumerate(results):
            y_text = 24 + index * 24
            label = f"pair {result.pair_index}: score={result.score} conf={result.confidence}"
            if result.needs_review:
                label += " REVIEW"
            cv2.putText(output, label, (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if result.impact_point_px is not None:
                point = (int(round(result.impact_point_px[0])), int(round(result.impact_point_px[1])))
                cv2.circle(output, point, 5, (0, 255, 255), -1)
                cv2.putText(output, str(result.score), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return output


def save_visualization(image: np.ndarray, output_path: Path) -> None:
    """Save a visualization image, creating parent directories as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Failed to write visualization image: {output_path}")
