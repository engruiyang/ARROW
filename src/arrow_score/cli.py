"""Minimal command-line interface for ARROW TASK 1A."""

import argparse


def _print_info() -> None:
    """Print project and current-stage information."""
    print("ARROW - archery scoring assistance")
    print("Current stage: TASK 1A minimal Python project skeleton")
    print("Supported target types: 90")
    print("Future task: implement an image-differencing demo for new-arrow detection")


def main(argv: list[str] | None = None) -> int:
    """Run the arrow-score CLI."""
    parser = argparse.ArgumentParser(prog="arrow-score")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("info", help="show project information")
    args = parser.parse_args(argv)

    if args.command == "info":
        _print_info()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
