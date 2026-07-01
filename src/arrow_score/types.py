"""Core dataclasses used across the ARROW package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    """A single input frame and optional decoded image payload."""

    path: Path
    image: object | None = None


@dataclass(frozen=True)
class ImagePair:
    """A neighboring before/after frame pair."""

    prev: Frame
    curr: Frame
    index: int


@dataclass(frozen=True)
class TargetSpec:
    """Physical target geometry and scoring boundaries."""

    name: str
    outer_radius_mm: float
    center_radius_mm: float
    ring_width_mm: float
    line_width_mm: float
    score_boundaries: list[tuple[float, int]]


@dataclass(frozen=True)
class Calibration:
    """Calibration values that map image pixels to target coordinates."""

    target_type: str
    center_px: tuple[float, float]
    ellipse_axes_px: tuple[float, float]
    ellipse_angle_deg: float
    roi: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ArrowCandidate:
    """A detected arrow candidate and its estimated impact point."""

    line_start_px: tuple[float, float]
    line_end_px: tuple[float, float]
    impact_point_px: tuple[float, float]
    confidence: str
    debug_reason: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Final score output for one image pair and one arrow candidate."""

    pair_index: int
    frame_prev: str
    frame_curr: str
    target_type: str
    impact_point_px: tuple[float, float] | None
    radius_mm: float | None
    score: int | str | None
    confidence: str
    needs_review: bool
    debug_reason: str | None = None
