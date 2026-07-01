"""Command-line interface for ARROW demo commands."""

import argparse
from pathlib import Path

from arrow_score.canonical_pipeline import run_canonical_image_sequence_demo
from arrow_score.config import ArrowDetectorConfig, DiffConfig, PipelineDebugConfig
from arrow_score.pipeline import run_image_pair_demo, run_image_sequence_demo


def _print_info() -> None:
    """Print project and current-stage information."""
    print("ARROW - archery scoring assistance")
    print("Current stage: TASK 1D-1 canonical target registration demo pipeline")
    print("Supported target types: 90")
    print("Supported inputs: image sequence directory, before/after image pair, or canonical sequence")
    print("Not supported: video streams, cameras, GUI windows, deep learning models")


def _print_success(pair_count: int, output_dir: Path) -> None:
    """Print common success output paths."""
    print(f"Processed pairs: {pair_count}")
    print(f"Results JSON: {output_dir / 'results.json'}")
    print(f"Results CSV: {output_dir / 'results.csv'}")
    print(f"Visualizations: {output_dir / 'visualized'}")
    print(f"Debug outputs: {output_dir / 'debug'}")


def _add_demo_options(parser: argparse.ArgumentParser) -> None:
    """Add shared detector and debug options to a subcommand parser."""
    parser.add_argument("--calibration", required=True, type=Path, help="calibration JSON path")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument("--target-type", default="90", help="target type, currently only 90")
    parser.add_argument("--diff-threshold", type=int, default=25, help="binary threshold for frame differencing")
    parser.add_argument("--diff-min-area", type=int, default=20, help="minimum contour area for diff boxes")
    parser.add_argument("--blur-kernel", type=int, default=5, help="positive odd Gaussian blur kernel")
    parser.add_argument("--arrow-min-area", type=int, default=20, help="minimum contour area for arrow candidates")
    parser.add_argument("--arrow-min-aspect", type=float, default=2.0, help="minimum arrow candidate aspect ratio")
    parser.add_argument("--arrow-min-line-length", type=float, default=10.0, help="minimum fitLine segment length")
    parser.add_argument("--save-debug", action="store_true", help="enable debug image outputs")
    parser.add_argument("--save-masks", action="store_true", help="save binary diff masks")
    parser.add_argument("--save-bbox-debug", action="store_true", help="save bbox/contour debug images")
    parser.add_argument("--save-candidate-debug", action="store_true", help="save candidate line debug images")


def _diff_config_from_args(args: argparse.Namespace) -> DiffConfig:
    """Build DiffConfig from parsed CLI args."""
    return DiffConfig(blur_kernel=args.blur_kernel, threshold=args.diff_threshold, min_area=args.diff_min_area)


def _arrow_config_from_args(args: argparse.Namespace) -> ArrowDetectorConfig:
    """Build ArrowDetectorConfig from parsed CLI args."""
    return ArrowDetectorConfig(
        min_area=args.arrow_min_area,
        min_aspect_ratio=args.arrow_min_aspect,
        min_line_length=args.arrow_min_line_length,
    )


def _debug_config_from_args(args: argparse.Namespace) -> PipelineDebugConfig:
    """Build PipelineDebugConfig from parsed CLI args."""
    return PipelineDebugConfig(
        save_debug_images=args.save_debug,
        save_masks=args.save_masks,
        save_bbox_debug=args.save_bbox_debug,
        save_candidate_debug=args.save_candidate_debug,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the arrow-score CLI."""
    parser = argparse.ArgumentParser(prog="arrow-score")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="show project information")

    sequence_parser = subparsers.add_parser("sequence", help="run demo on a sorted image directory")
    sequence_parser.add_argument("--input", required=True, type=Path, help="directory containing input images")
    _add_demo_options(sequence_parser)

    pair_parser = subparsers.add_parser("pair", help="run demo on one before/after image pair")
    pair_parser.add_argument("--prev", required=True, type=Path, help="previous/before image path")
    pair_parser.add_argument("--curr", required=True, type=Path, help="current/after image path")
    _add_demo_options(pair_parser)

    canonical_parser = subparsers.add_parser("canonical-sequence", help="auto-register targets and run demo in canonical coordinates")
    canonical_parser.add_argument("--input", required=True, type=Path, help="directory containing input images")
    canonical_parser.add_argument("--output", required=True, type=Path, help="output directory")
    canonical_parser.add_argument("--target-type", default="90", help="target type, currently only 90")
    canonical_parser.add_argument("--diff-threshold", type=int, default=25, help="binary threshold for frame differencing")
    canonical_parser.add_argument("--diff-min-area", type=int, default=20, help="minimum contour area for diff boxes")
    canonical_parser.add_argument("--blur-kernel", type=int, default=5, help="positive odd Gaussian blur kernel")
    canonical_parser.add_argument("--arrow-min-area", type=int, default=20, help="minimum contour area for arrow candidates")
    canonical_parser.add_argument("--arrow-min-aspect", type=float, default=2.0, help="minimum arrow candidate aspect ratio")
    canonical_parser.add_argument("--arrow-min-line-length", type=float, default=10.0, help="minimum fitLine segment length")
    canonical_parser.add_argument("--save-debug", action="store_true", help="enable extra diff/candidate debug image outputs")
    canonical_parser.add_argument("--save-masks", action="store_true", help="save binary diff masks")
    canonical_parser.add_argument("--save-bbox-debug", action="store_true", help="save bbox/contour debug images")
    canonical_parser.add_argument("--save-candidate-debug", action="store_true", help="save candidate line debug images")

    args = parser.parse_args(argv)

    try:
        if args.command == "info":
            _print_info()
            return 0
        if args.command == "sequence":
            results = run_image_sequence_demo(
                args.input,
                args.calibration,
                args.output,
                args.target_type,
                diff_config=_diff_config_from_args(args),
                arrow_config=_arrow_config_from_args(args),
                debug_config=_debug_config_from_args(args),
            )
            _print_success(len({result.pair_index for result in results}), args.output)
            return 0
        if args.command == "pair":
            results = run_image_pair_demo(
                args.prev,
                args.curr,
                args.calibration,
                args.output,
                args.target_type,
                diff_config=_diff_config_from_args(args),
                arrow_config=_arrow_config_from_args(args),
                debug_config=_debug_config_from_args(args),
            )
            _print_success(len({result.pair_index for result in results}), args.output)
            return 0
        if args.command == "canonical-sequence":
            results = run_canonical_image_sequence_demo(
                args.input,
                args.output,
                args.target_type,
                diff_config=_diff_config_from_args(args),
                arrow_config=_arrow_config_from_args(args),
                debug_config=_debug_config_from_args(args),
            )
            _print_success(len({result.pair_index for result in results}), args.output)
            return 0
    except Exception as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
