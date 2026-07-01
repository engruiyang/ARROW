"""Command-line interface for ARROW demo commands."""

import argparse
from pathlib import Path

from arrow_score.pipeline import run_image_pair_demo, run_image_sequence_demo


def _print_info() -> None:
    """Print project and current-stage information."""
    print("ARROW - archery scoring assistance")
    print("Current stage: TASK 1B minimal image-difference demo pipeline")
    print("Supported target types: 90")
    print("Supported inputs: image sequence directory or before/after image pair")
    print("Not supported: video streams, cameras, deep learning models")


def _print_success(pair_count: int, output_dir: Path) -> None:
    """Print common success output paths."""
    print(f"Processed pairs: {pair_count}")
    print(f"Results JSON: {output_dir / 'results.json'}")
    print(f"Results CSV: {output_dir / 'results.csv'}")
    print(f"Visualizations: {output_dir / 'visualized'}")


def main(argv: list[str] | None = None) -> int:
    """Run the arrow-score CLI."""
    parser = argparse.ArgumentParser(prog="arrow-score")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="show project information")

    sequence_parser = subparsers.add_parser("sequence", help="run demo on a sorted image directory")
    sequence_parser.add_argument("--input", required=True, type=Path, help="directory containing input images")
    sequence_parser.add_argument("--calibration", required=True, type=Path, help="calibration JSON path")
    sequence_parser.add_argument("--output", required=True, type=Path, help="output directory")
    sequence_parser.add_argument("--target-type", default="90", help="target type, currently only 90")

    pair_parser = subparsers.add_parser("pair", help="run demo on one before/after image pair")
    pair_parser.add_argument("--prev", required=True, type=Path, help="previous/before image path")
    pair_parser.add_argument("--curr", required=True, type=Path, help="current/after image path")
    pair_parser.add_argument("--calibration", required=True, type=Path, help="calibration JSON path")
    pair_parser.add_argument("--output", required=True, type=Path, help="output directory")
    pair_parser.add_argument("--target-type", default="90", help="target type, currently only 90")

    args = parser.parse_args(argv)

    try:
        if args.command == "info":
            _print_info()
            return 0
        if args.command == "sequence":
            results = run_image_sequence_demo(args.input, args.calibration, args.output, args.target_type)
            _print_success(len({result.pair_index for result in results}), args.output)
            return 0
        if args.command == "pair":
            results = run_image_pair_demo(args.prev, args.curr, args.calibration, args.output, args.target_type)
            _print_success(len({result.pair_index for result in results}), args.output)
            return 0
    except Exception as exc:
        parser.exit(status=2, message=f"error: {exc}\n")

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
