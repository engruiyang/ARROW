#!/usr/bin/env python3
"""Normalize an image sequence by auto-cropping around the detected target."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from arrow_score.image_loader import load_image  # noqa: E402
from arrow_score.io import SUPPORTED_IMAGE_SUFFIXES  # noqa: E402
from arrow_score.target_registration import HsvTargetRegistrar, TargetPose  # noqa: E402
from arrow_score.visualize import save_visualization  # noqa: E402


def collect_images(input_dir: Path) -> list[Path]:
    """Return sorted supported image paths from input_dir."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    images = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found in {input_dir}")
    return images


def compute_crop_box(pose: TargetPose, image_shape: tuple[int, ...], padding_ratio: float) -> tuple[int, int, int, int]:
    """Compute a bounded crop box around a detected target pose."""
    height, width = image_shape[:2]
    half_w = pose.ellipse_axes_px[0] * padding_ratio
    half_h = pose.ellipse_axes_px[1] * padding_ratio
    cx, cy = pose.center_px
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(width, int(round(cx + half_w)))
    y1 = min(height, int(round(cy + half_h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Computed invalid crop box: {(x0, y0, x1, y1)}")
    return (x0, y0, x1, y1)


def crop_with_bounds(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using an already bounded crop box."""
    x0, y0, x1, y1 = crop_box
    return image[y0:y1, x0:x1]


def center_crop(image: np.ndarray, output_width: int, output_height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the largest centered region matching the requested aspect ratio."""
    height, width = image.shape[:2]
    target_ratio = output_width / output_height
    current_ratio = width / height
    if current_ratio > target_ratio:
        crop_h = height
        crop_w = int(round(height * target_ratio))
    else:
        crop_w = width
        crop_h = int(round(width / target_ratio))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    box = (x0, y0, x0 + crop_w, y0 + crop_h)
    return crop_with_bounds(image, box), box


def _resize_and_write(image: np.ndarray, output_path: Path, width: int, height: int, jpeg_quality: int) -> None:
    """Resize and write one normalized image."""
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    if not cv2.imwrite(str(output_path), resized, params):
        raise ValueError(f"Failed to write normalized image: {output_path}")


def normalize_sequence(
    input_dir: Path,
    output_dir: Path,
    width: int = 1600,
    height: int = 1600,
    padding_ratio: float = 1.25,
    jpeg_quality: int = 95,
    fallback: str = "center-crop",
    save_debug: bool = False,
) -> list[dict[str, object]]:
    """Normalize an image sequence and write a manifest.csv."""
    if fallback not in {"center-crop", "resize-only", "fail"}:
        raise ValueError("fallback must be one of: center-crop, resize-only, fail")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registrar = HsvTargetRegistrar()
    manifest: list[dict[str, object]] = []

    for index, path in enumerate(collect_images(input_dir), start=1):
        image = load_image(path)
        source_height, source_width = image.shape[:2]
        registered = False
        failure_reason = ""
        crop_box: tuple[int, int, int, int]
        try:
            debug = registrar.detect_pose_debug(image)
            crop_box = compute_crop_box(debug.pose, image.shape, padding_ratio)
            cropped = crop_with_bounds(image, crop_box)
            registered = True
            if save_debug and debug.debug_image is not None:
                x0, y0, x1, y1 = crop_box
                debug_crop = debug.debug_image.copy()
                cv2.rectangle(debug_crop, (x0, y0), (x1, y1), (0, 255, 255), 3)
                save_visualization(debug_crop, output_dir / "debug_crop" / f"frame_{index:04d}_crop_debug.jpg")
        except ValueError as exc:
            failure_reason = str(exc)
            if fallback == "fail":
                raise
            if fallback == "resize-only":
                crop_box = (0, 0, source_width, source_height)
                cropped = image
            else:
                cropped, crop_box = center_crop(image, width, height)

        output_path = output_dir / f"frame_{index:04d}.jpg"
        _resize_and_write(cropped, output_path, width, height, jpeg_quality)
        manifest.append(
            {
                "source_file": path.name,
                "source_width": source_width,
                "source_height": source_height,
                "crop_box": f"{crop_box[0]},{crop_box[1]},{crop_box[2]},{crop_box[3]}",
                "output_file": output_path.name,
                "output_width": width,
                "output_height": height,
                "registration_success": registered,
                "failure_reason": failure_reason,
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Run the normalization CLI."""
    parser = argparse.ArgumentParser(description="Normalize a local ARROW image sequence")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--padding-ratio", type=float, default=1.25)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--fallback", choices=["center-crop", "resize-only", "fail"], default="center-crop")
    parser.add_argument("--save-debug", action="store_true")
    args = parser.parse_args(argv)
    normalize_sequence(
        input_dir=args.input,
        output_dir=args.output,
        width=args.width,
        height=args.height,
        padding_ratio=args.padding_ratio,
        jpeg_quality=args.jpeg_quality,
        fallback=args.fallback,
        save_debug=args.save_debug,
    )
    print(f"Normalized sequence written to {args.output}")
    print(f"Manifest: {args.output / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
