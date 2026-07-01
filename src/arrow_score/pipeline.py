"""Core TASK 1C configurable image-difference demo pipeline."""

import time
from pathlib import Path

import numpy as np

from arrow_score.arrow_detector import SimpleArrowDetector
from arrow_score.calibration import load_calibration
from arrow_score.config import ArrowDetectorConfig, DiffConfig, PipelineDebugConfig
from arrow_score.diff_detector import BasicFrameDiffDetector
from arrow_score.image_loader import load_pair_images
from arrow_score.interfaces import ArrowDetector, DiffDetector, FrameSource
from arrow_score.io import ImagePairSource, ImageSequenceSource
from arrow_score.result_writer import JsonCsvResultWriter
from arrow_score.scorer import BasicScorer, make_no_candidate_result
from arrow_score.target_model import get_target_spec
from arrow_score.types import Calibration, ScoreResult, TargetSpec
from arrow_score.visualize import BasicVisualizer, draw_candidate_debug, draw_diff_debug, save_visualization


def _image_from_loaded_pair(image: object, label: str) -> np.ndarray:
    """Validate that a loaded frame image is a NumPy array."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Loaded {label} image is not a NumPy array")
    return image


def run_pipeline(
    frame_source: FrameSource,
    calibration: Calibration,
    target_spec: TargetSpec,
    diff_detector: DiffDetector,
    arrow_detector: ArrowDetector,
    scorer: BasicScorer,
    visualizer: BasicVisualizer,
    result_writer: JsonCsvResultWriter,
    output_dir: Path,
    debug_config: PipelineDebugConfig | None = None,
) -> list[ScoreResult]:
    """Run the full image-pair demo pipeline and write outputs."""
    output_dir = Path(output_dir)
    all_results: list[ScoreResult] = []

    for pair in frame_source.iter_pairs():
        total_start = time.perf_counter()

        load_start = time.perf_counter()
        loaded_pair = load_pair_images(pair)
        prev_image = _image_from_loaded_pair(loaded_pair.prev.image, "previous")
        curr_image = _image_from_loaded_pair(loaded_pair.curr.image, "current")
        load_ms = (time.perf_counter() - load_start) * 1000

        diff_start = time.perf_counter()
        diff_output = diff_detector.detect(prev_image, curr_image, calibration)
        diff_ms = (time.perf_counter() - diff_start) * 1000

        detect_start = time.perf_counter()
        candidates = arrow_detector.detect(diff_output, calibration)
        detect_ms = (time.perf_counter() - detect_start) * 1000

        score_start = time.perf_counter()
        if candidates:
            pair_results = [scorer.score_candidate(candidate, calibration, target_spec, loaded_pair, pair.index) for candidate in candidates]
        else:
            pair_results = [make_no_candidate_result(loaded_pair, pair.index, target_spec.name, "no_candidate")]
        all_results.extend(pair_results)
        score_ms = (time.perf_counter() - score_start) * 1000

        visualize_start = time.perf_counter()
        visualized = visualizer.draw_result(curr_image, calibration, target_spec, pair_results)
        save_visualization(visualized, output_dir / "visualized" / f"pair_{pair.index:04d}.png")
        if debug_config and debug_config.save_debug_images:
            if debug_config.save_masks:
                save_visualization(diff_output.mask, output_dir / "debug" / "masks" / f"pair_{pair.index:04d}_mask.png")
            if debug_config.save_bbox_debug:
                save_visualization(draw_diff_debug(curr_image, diff_output), output_dir / "debug" / "bboxes" / f"pair_{pair.index:04d}_bbox.png")
            if debug_config.save_candidate_debug:
                save_visualization(draw_candidate_debug(curr_image, candidates), output_dir / "debug" / "candidates" / f"pair_{pair.index:04d}_candidates.png")
        visualize_ms = (time.perf_counter() - visualize_start) * 1000

        total_ms = (time.perf_counter() - total_start) * 1000
        print(
            f"pair={pair.index} load_ms={load_ms:.2f} diff_ms={diff_ms:.2f} "
            f"detect_ms={detect_ms:.2f} score_ms={score_ms:.2f} "
            f"visualize_ms={visualize_ms:.2f} total_ms={total_ms:.2f}"
        )

    result_writer.write(all_results, output_dir)
    return all_results


def run_image_sequence_demo(
    input_dir: Path,
    calibration_path: Path,
    output_dir: Path,
    target_type: str = "90",
    diff_config: DiffConfig | None = None,
    arrow_config: ArrowDetectorConfig | None = None,
    debug_config: PipelineDebugConfig | None = None,
) -> list[ScoreResult]:
    """Run the demo pipeline on a directory of sorted images."""
    calibration = load_calibration(calibration_path)
    target_spec = get_target_spec(target_type)
    return run_pipeline(
        frame_source=ImageSequenceSource(input_dir),
        calibration=calibration,
        target_spec=target_spec,
        diff_detector=BasicFrameDiffDetector(diff_config),
        arrow_detector=SimpleArrowDetector(arrow_config),
        scorer=BasicScorer(),
        visualizer=BasicVisualizer(),
        result_writer=JsonCsvResultWriter(),
        output_dir=output_dir,
        debug_config=debug_config,
    )


def run_image_pair_demo(
    prev_path: Path,
    curr_path: Path,
    calibration_path: Path,
    output_dir: Path,
    target_type: str = "90",
    diff_config: DiffConfig | None = None,
    arrow_config: ArrowDetectorConfig | None = None,
    debug_config: PipelineDebugConfig | None = None,
) -> list[ScoreResult]:
    """Run the demo pipeline on one before/after image pair."""
    calibration = load_calibration(calibration_path)
    target_spec = get_target_spec(target_type)
    return run_pipeline(
        frame_source=ImagePairSource(prev_path, curr_path),
        calibration=calibration,
        target_spec=target_spec,
        diff_detector=BasicFrameDiffDetector(diff_config),
        arrow_detector=SimpleArrowDetector(arrow_config),
        scorer=BasicScorer(),
        visualizer=BasicVisualizer(),
        result_writer=JsonCsvResultWriter(),
        output_dir=output_dir,
        debug_config=debug_config,
    )
