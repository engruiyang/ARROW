"""Replaceable protocol interfaces for ARROW processing components."""

from pathlib import Path
from typing import Iterable, Protocol

from arrow_score.types import ArrowCandidate, Calibration, ImagePair, ScoreResult, TargetSpec


class FrameSource(Protocol):
    """Provides neighboring image pairs."""

    def iter_pairs(self) -> Iterable[ImagePair]:
        """Yield image pairs in processing order."""


class Calibrator(Protocol):
    """Maps image points into calibrated target geometry."""

    def point_to_radius_mm(
        self,
        point_px: tuple[float, float],
        calibration: Calibration,
        target_spec: TargetSpec,
    ) -> float:
        """Convert a pixel point to millimeter radius."""


class DiffDetector(Protocol):
    """Detects changes between two images."""

    def detect(self, prev_image: object, curr_image: object, calibration: Calibration) -> object:
        """Return an implementation-specific diff output."""


class ArrowDetector(Protocol):
    """Detects arrow candidates from a diff output."""

    def detect(self, diff_output: object, calibration: Calibration) -> list[ArrowCandidate]:
        """Return detected arrow candidates."""


class Scorer(Protocol):
    """Scores a physical target radius."""

    def score_radius(self, radius_mm: float, target_spec: TargetSpec) -> int | str:
        """Return the score for a radius in millimeters."""


class Visualizer(Protocol):
    """Draws scoring results on an image."""

    def draw_result(
        self,
        image: object,
        calibration: Calibration,
        target_spec: TargetSpec,
        results: list[ScoreResult],
    ) -> object:
        """Return an image with result annotations."""


class ResultWriter(Protocol):
    """Writes score results to disk."""

    def write(self, results: list[ScoreResult], output_dir: Path) -> None:
        """Persist results under the output directory."""
