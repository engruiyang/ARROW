"""Canonical target registration pipeline for image sequences."""

import time
from pathlib import Path

from arrow_score.arrow_detector import SimpleArrowDetector
from arrow_score.config import ArrowDetectorConfig, DiffConfig, PipelineDebugConfig
from arrow_score.diff_detector import BasicFrameDiffDetector
from arrow_score.image_loader import load_image
from arrow_score.io import ImageSequenceSource
from arrow_score.result_writer import JsonCsvResultWriter
from arrow_score.scorer import BasicScorer, make_no_candidate_result
from arrow_score.target_model import get_target_spec
from arrow_score.target_registration import CanonicalTargetConfig, HsvTargetRegistrar
from arrow_score.types import Calibration, Frame, ImagePair, ScoreResult
from arrow_score.visualize import BasicVisualizer, draw_candidate_debug, draw_diff_debug, save_visualization


def canonical_calibration(config: CanonicalTargetConfig | None = None) -> Calibration:
    """Return fixed calibration for canonical 90-target coordinates."""
    config = config or CanonicalTargetConfig()
    center = (float(config.center_px[0]), float(config.center_px[1]))
    axes = (float(config.outer_radius_px), float(config.outer_radius_px))
    return Calibration(target_type=config.target_type, center_px=center, ellipse_axes_px=axes, ellipse_angle_deg=0.0, roi=None)


def run_canonical_image_sequence_demo(
    input_dir: Path,
    output_dir: Path,
    target_type: str = "90",
    diff_config: DiffConfig | None = None,
    arrow_config: ArrowDetectorConfig | None = None,
    debug_config: PipelineDebugConfig | None = None,
) -> list[ScoreResult]:
    """Run image sequence scoring after per-frame target registration and canonical warping."""
    output_dir = Path(output_dir)
    canonical_config = CanonicalTargetConfig(target_type=target_type)
    registrar = HsvTargetRegistrar(canonical_config=canonical_config)
    target_spec = get_target_spec(target_type)
    calibration = canonical_calibration(canonical_config)
    diff_detector = BasicFrameDiffDetector(diff_config)
    arrow_detector = SimpleArrowDetector(arrow_config)
    scorer = BasicScorer()
    visualizer = BasicVisualizer()
    result_writer = JsonCsvResultWriter()
    all_results: list[ScoreResult] = []

    for pair in ImageSequenceSource(input_dir).iter_pairs():
        total_start = time.perf_counter()
        prev_image = load_image(pair.prev.path)
        curr_image = load_image(pair.curr.path)

        prev_registration = registrar.detect_pose_debug(prev_image)
        curr_registration = registrar.detect_pose_debug(curr_image)
        prev_canonical = registrar.warp_to_canonical(prev_image, prev_registration.pose)
        curr_canonical = registrar.warp_to_canonical(curr_image, curr_registration.pose)

        save_visualization(prev_registration.debug_image, output_dir / "debug" / "registration" / f"pair_{pair.index:04d}_prev_registration.png")
        save_visualization(curr_registration.debug_image, output_dir / "debug" / "registration" / f"pair_{pair.index:04d}_curr_registration.png")
        save_visualization(prev_canonical, output_dir / "debug" / "canonical" / f"pair_{pair.index:04d}_prev_canonical.png")
        save_visualization(curr_canonical, output_dir / "debug" / "canonical" / f"pair_{pair.index:04d}_curr_canonical.png")

        diff_start = time.perf_counter()
        diff_output = diff_detector.detect(prev_canonical, curr_canonical, calibration)
        diff_ms = (time.perf_counter() - diff_start) * 1000

        detect_start = time.perf_counter()
        candidates = arrow_detector.detect(diff_output, calibration)
        detect_ms = (time.perf_counter() - detect_start) * 1000

        canonical_pair = ImagePair(
            prev=Frame(path=pair.prev.path, image=prev_canonical),
            curr=Frame(path=pair.curr.path, image=curr_canonical),
            index=pair.index,
        )
        if candidates:
            pair_results = [scorer.score_candidate(candidate, calibration, target_spec, canonical_pair, pair.index) for candidate in candidates]
        else:
            pair_results = [make_no_candidate_result(canonical_pair, pair.index, target_spec.name, "no_candidate")]
        all_results.extend(pair_results)

        visualized = visualizer.draw_result(curr_canonical, calibration, target_spec, pair_results)
        save_visualization(visualized, output_dir / "visualized" / f"pair_{pair.index:04d}.png")
        if debug_config and debug_config.save_debug_images:
            if debug_config.save_masks:
                save_visualization(diff_output.mask, output_dir / "debug" / "masks" / f"pair_{pair.index:04d}_mask.png")
            if debug_config.save_bbox_debug:
                save_visualization(draw_diff_debug(curr_canonical, diff_output), output_dir / "debug" / "bboxes" / f"pair_{pair.index:04d}_bbox.png")
            if debug_config.save_candidate_debug:
                save_visualization(draw_candidate_debug(curr_canonical, candidates), output_dir / "debug" / "candidates" / f"pair_{pair.index:04d}_candidates.png")

        total_ms = (time.perf_counter() - total_start) * 1000
        print(f"canonical_pair={pair.index} diff_ms={diff_ms:.2f} detect_ms={detect_ms:.2f} total_ms={total_ms:.2f}")

    result_writer.write(all_results, output_dir)
    return all_results
