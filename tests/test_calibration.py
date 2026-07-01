import json

import pytest

from arrow_score.calibration import load_calibration, point_to_normalized, point_to_radius_mm, save_calibration
from arrow_score.target_model import get_target_spec
from arrow_score.types import Calibration


def _example_calibration() -> dict[str, object]:
    return {
        "target_type": "90",
        "center_px": [960, 540],
        "ellipse_axes_px": [300, 280],
        "ellipse_angle_deg": 0,
        "roi": None,
    }


def test_load_calibration_reads_example_json(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_example_calibration()), encoding="utf-8")

    calibration = load_calibration(path)

    assert calibration.target_type == "90"
    assert calibration.center_px == (960.0, 540.0)
    assert calibration.ellipse_axes_px == (300.0, 280.0)
    assert calibration.ellipse_angle_deg == 0.0
    assert calibration.roi is None


def test_save_calibration_round_trips(tmp_path) -> None:
    calibration = Calibration(
        target_type="90",
        center_px=(960.0, 540.0),
        ellipse_axes_px=(300.0, 280.0),
        ellipse_angle_deg=0.0,
        roi=(1, 2, 3, 4),
    )
    path = tmp_path / "nested" / "calibration.json"

    save_calibration(calibration, path)

    assert load_calibration(path) == calibration


def test_point_to_normalized_center_is_zero() -> None:
    calibration = Calibration("90", (960.0, 540.0), (300.0, 280.0), 0.0)
    assert point_to_normalized((960.0, 540.0), calibration) == pytest.approx((0.0, 0.0))


def test_point_to_normalized_x_axis_endpoint_radius_is_one() -> None:
    calibration = Calibration("90", (960.0, 540.0), (300.0, 280.0), 0.0)
    x_norm, y_norm = point_to_normalized((1260.0, 540.0), calibration)
    assert (x_norm**2 + y_norm**2) ** 0.5 == pytest.approx(1.0)


def test_point_to_radius_mm_x_axis_endpoint_is_outer_radius() -> None:
    calibration = Calibration("90", (960.0, 540.0), (300.0, 280.0), 0.0)
    assert point_to_radius_mm((1260.0, 540.0), calibration, get_target_spec("90")) == pytest.approx(245.0)


def test_zero_axis_raises_value_error() -> None:
    calibration = Calibration("90", (960.0, 540.0), (0.0, 280.0), 0.0)
    with pytest.raises(ValueError):
        point_to_normalized((960.0, 540.0), calibration)
