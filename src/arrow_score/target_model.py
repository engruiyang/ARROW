"""Target geometry and scoring rules for supported target types."""

from arrow_score.types import TargetSpec

_TARGET_90 = TargetSpec(
    name="90",
    outer_radius_mm=245.0,
    center_radius_mm=45.0,
    ring_width_mm=50.0,
    line_width_mm=1.0,
    score_boundaries=[
        (45.0, 10),
        (95.0, 9),
        (145.0, 8),
        (195.0, 7),
        (245.0, 6),
    ],
)


def get_target_spec(target_type: str | int) -> TargetSpec:
    """Return the target specification for a supported target type."""
    if str(target_type) == "90":
        return _TARGET_90
    raise ValueError(f"Unsupported target type: {target_type!r}; only '90' is supported")


def score_radius_mm(radius_mm: float, target_spec: TargetSpec) -> int | str:
    """Score a radius in millimeters using the target boundaries."""
    if radius_mm < 0:
        raise ValueError(f"radius_mm must be non-negative, got {radius_mm}")

    for boundary_mm, score in target_spec.score_boundaries:
        if radius_mm <= boundary_mm:
            return score
    return "outside"
