import json

import cv2
import numpy as np

from arrow_score.config import PipelineDebugConfig
from arrow_score.pipeline import run_image_sequence_demo


def _write_smoke_inputs(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    frame_1 = np.full((200, 200, 3), 255, dtype=np.uint8)
    frame_2 = frame_1.copy()
    frame_3 = frame_1.copy()
    cv2.line(frame_2, (30, 100), (90, 100), (0, 0, 0), 3)
    cv2.line(frame_3, (30, 100), (90, 100), (0, 0, 0), 3)
    cv2.line(frame_3, (110, 100), (170, 100), (0, 0, 0), 3)
    assert cv2.imwrite(str(input_dir / "frame_001.png"), frame_1)
    assert cv2.imwrite(str(input_dir / "frame_002.png"), frame_2)
    assert cv2.imwrite(str(input_dir / "frame_003.png"), frame_3)

    calibration_path = tmp_path / "calibration_90.json"
    calibration_path.write_text(
        json.dumps(
            {
                "target_type": "90",
                "center_px": [100, 100],
                "ellipse_axes_px": [80, 80],
                "ellipse_angle_deg": 0,
                "roi": None,
            }
        ),
        encoding="utf-8",
    )
    return input_dir, calibration_path


def test_run_image_sequence_demo_smoke(tmp_path) -> None:
    input_dir, calibration_path = _write_smoke_inputs(tmp_path)
    output_dir = tmp_path / "output"

    results = run_image_sequence_demo(input_dir, calibration_path, output_dir)

    assert len(results) >= 2
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "visualized").is_dir()


def test_run_image_sequence_demo_saves_debug_images(tmp_path) -> None:
    input_dir, calibration_path = _write_smoke_inputs(tmp_path)
    output_dir = tmp_path / "output_debug"

    run_image_sequence_demo(
        input_dir,
        calibration_path,
        output_dir,
        debug_config=PipelineDebugConfig(
            save_debug_images=True,
            save_masks=True,
            save_bbox_debug=True,
            save_candidate_debug=True,
        ),
    )

    assert (output_dir / "debug" / "masks" / "pair_0000_mask.png").exists()
    assert (output_dir / "debug" / "bboxes" / "pair_0000_bbox.png").exists()
    assert (output_dir / "debug" / "candidates" / "pair_0000_candidates.png").exists()
