"""Calibration JSON I/O and pixel-to-radius conversion helpers."""

import json
import math
from pathlib import Path
from typing import Any

from arrow_score.types import Calibration, TargetSpec

_REQUIRED_FIELDS = {"target_type", "center_px", "ellipse_axes_px", "ellipse_angle_deg", "roi"}


def _pair_of_floats(value: Any, field_name: str) -> tuple[float, float]:
    """Validate and convert a two-number sequence to a float pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item list or tuple")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric values") from exc


def _parse_roi(value: Any) -> tuple[int, int, int, int] | None:
    """Validate and convert an optional ROI."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("roi must be null or a four-item list/tuple")
    try:
        return tuple(int(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError("roi must contain integer values") from exc


def _validate_axes(axes: tuple[float, float]) -> None:
    """Ensure ellipse axes are positive."""
    if axes[0] <= 0 or axes[1] <= 0:
        raise ValueError(f"ellipse_axes_px values must be positive, got {axes}")


def load_calibration(path: Path) -> Calibration:
    """Load calibration data from a UTF-8 JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Calibration JSON must contain an object")

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Calibration JSON missing required fields: {sorted(missing)}")

    center_px = _pair_of_floats(data["center_px"], "center_px")
    axes_px = _pair_of_floats(data["ellipse_axes_px"], "ellipse_axes_px")
    _validate_axes(axes_px)
    try:
        angle = float(data["ellipse_angle_deg"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ellipse_angle_deg must be numeric") from exc

    target_type = data["target_type"]
    if not isinstance(target_type, (str, int)):
        raise ValueError("target_type must be a string or integer")

    return Calibration(
        target_type=str(target_type),
        center_px=center_px,
        ellipse_axes_px=axes_px,
        ellipse_angle_deg=angle,
        roi=_parse_roi(data["roi"]),
    )


def save_calibration(calibration: Calibration, path: Path) -> None:
    """Save calibration data as UTF-8 JSON, creating parent directories."""
    _validate_axes(calibration.ellipse_axes_px)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "target_type": calibration.target_type,
        "center_px": list(calibration.center_px),
        "ellipse_axes_px": list(calibration.ellipse_axes_px),
        "ellipse_angle_deg": calibration.ellipse_angle_deg,
        "roi": list(calibration.roi) if calibration.roi is not None else None,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def point_to_normalized(point_px: tuple[float, float], calibration: Calibration) -> tuple[float, float]:
    """Convert a pixel point to normalized ellipse coordinates."""
    _validate_axes(calibration.ellipse_axes_px)
    dx = point_px[0] - calibration.center_px[0]
    dy = point_px[1] - calibration.center_px[1]
    angle_rad = math.radians(calibration.ellipse_angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x_rot = dx * cos_a + dy * sin_a
    y_rot = -dx * sin_a + dy * cos_a
    return (x_rot / calibration.ellipse_axes_px[0], y_rot / calibration.ellipse_axes_px[1])


def point_to_radius_mm(
    point_px: tuple[float, float],
    calibration: Calibration,
    target_spec: TargetSpec,
) -> float:
    """Convert a pixel point to physical target radius in millimeters."""
    x_norm, y_norm = point_to_normalized(point_px, calibration)
    return math.sqrt(x_norm**2 + y_norm**2) * target_spec.outer_radius_mm
