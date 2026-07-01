import cv2

from arrow_score.config import PipelineDebugConfig
from arrow_score.canonical_pipeline import run_canonical_image_sequence_demo
from tests.synthetic_target import make_synthetic_target


def test_run_canonical_image_sequence_demo_smoke(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    frame_1 = make_synthetic_target(center=(490, 490))
    frame_2 = make_synthetic_target(center=(496, 493))
    frame_3 = make_synthetic_target(center=(484, 496))
    cv2.line(frame_2, (430, 490), (470, 490), (0, 0, 0), 5)
    cv2.line(frame_3, (430, 490), (470, 490), (0, 0, 0), 5)
    cv2.line(frame_3, (515, 490), (555, 490), (0, 0, 0), 5)
    assert cv2.imwrite(str(input_dir / "frame_001.png"), frame_1)
    assert cv2.imwrite(str(input_dir / "frame_002.png"), frame_2)
    assert cv2.imwrite(str(input_dir / "frame_003.png"), frame_3)

    results = run_canonical_image_sequence_demo(
        input_dir,
        output_dir,
        debug_config=PipelineDebugConfig(
            save_debug_images=True,
            save_masks=True,
            save_bbox_debug=True,
            save_candidate_debug=True,
        ),
    )

    assert len(results) >= 2
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "debug" / "registration").is_dir()
    assert (output_dir / "debug" / "canonical").is_dir()
    assert (output_dir / "debug" / "masks" / "pair_0000_mask.png").exists()
