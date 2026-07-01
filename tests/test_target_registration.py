import numpy as np
import pytest

from arrow_score.target_registration import HsvTargetRegistrar, TargetPose
from tests.synthetic_target import make_synthetic_target


def test_synthetic_color_target_detects_pose() -> None:
    image = make_synthetic_target()
    pose = HsvTargetRegistrar(min_color_area=1000).detect_pose(image)

    assert pose.confidence
    assert pose.center_px == pytest.approx((490, 490), abs=10)
    assert pose.ellipse_axes_px[0] > 350
    assert pose.ellipse_axes_px[1] > 350


def test_warp_to_canonical_output_shape() -> None:
    image = make_synthetic_target()
    registrar = HsvTargetRegistrar(min_color_area=1000)
    pose = registrar.detect_pose(image)

    canonical = registrar.warp_to_canonical(image, pose)

    assert canonical.shape == (980, 980, 3)


def test_point_mapping_round_trip_error_is_small() -> None:
    registrar = HsvTargetRegistrar()
    pose = TargetPose(center_px=(100.0, 120.0), ellipse_axes_px=(50.0, 80.0), ellipse_angle_deg=0.0, confidence="high")
    points = np.array([[100.0, 120.0], [150.0, 120.0], [100.0, 200.0]])

    canonical = registrar.original_to_canonical_points(points, pose)
    round_trip = registrar.canonical_to_original_points(canonical, pose)

    assert round_trip == pytest.approx(points, abs=1e-6)


def test_detect_pose_raises_on_blank_image() -> None:
    image = np.full((980, 980, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError):
        HsvTargetRegistrar(min_color_area=1000).detect_pose(image)
