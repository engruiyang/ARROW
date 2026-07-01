# ARROW

ARROW is an archery scoring assistance project for civilian archery ranges. It is
intended to help identify newly shot arrows and estimate their scores while
keeping low-confidence results available for human review.

## First-version scope

The first version is intentionally narrow:

- Supports the 90 target model only.
- Accepts image sequences or a before/after image pair.
- Plans to detect new arrows with before/after image differencing.
- Estimates impact points and maps them to the 90 target geometry.

## Current status: TASK 1A

TASK 1A only provides the minimal Python project skeleton:

- Core dataclasses and protocol interfaces.
- 90 target scoring model.
- Path-level image sequence and image-pair readers.
- Calibration JSON loading/saving and radius conversion helpers.
- Minimal CLI information command.
- Basic pytest coverage.

The current code does **not** connect to video streams, cameras, capture cards, or
network media protocols. It also does **not** use deep learning and does **not**
implement the full image-differencing or arrow-detection algorithms yet.

## Installation

```bash
python -m pip install -e .
```

For test dependencies:

```bash
python -m pip install -e '.[test]'
```

## Testing

```bash
pytest
```

## CLI

```bash
python -m arrow_score.cli info
arrow-score info
```

## Next steps

Future tasks will add a small image-differencing demo, arrow-shaft detection, and
visualized output images while keeping input, calibration, detection, scoring, and
output modules independently replaceable.
