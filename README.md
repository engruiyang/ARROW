# ARROW

ARROW is an archery scoring assistance project for civilian archery ranges. It is
intended to help identify newly shot arrows and estimate their scores while
keeping low-confidence results available for human review.

## First-version scope

The first version is intentionally narrow:

- Supports the 90 target model only.
- Accepts image sequences or a before/after image pair.
- Detects newly changed regions with before/after image differencing.
- Estimates coarse arrow candidates and maps impact points to the 90 target geometry.

## Current status: TASK 1B

TASK 1B provides a minimal runnable image-difference demo pipeline:

- Core dataclasses and protocol interfaces.
- 90 target scoring model.
- Path-level image sequence and image-pair readers.
- Real image loading with OpenCV into BGR NumPy arrays.
- Basic frame differencing and changed-region extraction.
- Simple elongated-box arrow candidate estimation.
- Calibration JSON loading/saving and radius conversion helpers.
- JSON/CSV result writing and basic visualized output images.
- CLI commands for `info`, `sequence`, and `pair` modes.

The current code does **not** connect to video streams, cameras, capture cards, or
network media protocols. It also does **not** use deep learning and does **not**
attempt high-accuracy or competition-grade arrow detection.

## Local image input

`input_images/` is the local image entry directory. You can place field images in
subdirectories such as:

```text
input_images/demo_sequence/
input_images/pair/
```

For pair mode, you can also place files such as:

```text
input_images/before.jpg
input_images/after.jpg
```

Real images and subdirectories under `input_images/` are ignored by git, while
`input_images/.gitkeep` is kept so the directory exists in fresh checkouts. Do
not commit real sample images, large datasets, videos, or generated outputs.

A calibration JSON file is required. Example:

```json
{
  "target_type": "90",
  "center_px": [960, 540],
  "ellipse_axes_px": [300, 280],
  "ellipse_angle_deg": 0,
  "roi": null
}
```

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

Show project information:

```bash
python -m arrow_score.cli info
arrow-score info
```

Run an image sequence demo:

```bash
python -m arrow_score.cli sequence --input input_images/demo_sequence --calibration input_images/calibration_90.json --output outputs/demo
```

Run a before/after pair demo:

```bash
python -m arrow_score.cli pair --prev input_images/before.jpg --curr input_images/after.jpg --calibration input_images/calibration_90.json --output outputs/demo
```

## Outputs

The demo writes:

- `results.json`
- `results.csv`
- `visualized/pair_XXXX.png`

Generated outputs under `outputs/` are ignored by git except `outputs/.gitkeep`.

## Next steps

Future tasks can improve image differencing, arrow-shaft detection, calibration
assistance, review tooling, and visualization quality while keeping input,
detection, scoring, visualization, and result writing modules independently
replaceable.
