import csv

import cv2
import numpy as np
import pytest

from scripts.normalize_sequence import normalize_sequence
from tests.synthetic_target import make_synthetic_target


def test_normalize_sequence_outputs_uniform_images_and_manifest(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    images = [
        make_synthetic_target((900, 920), radius=300),
        make_synthetic_target((840, 960), radius=280),
        make_synthetic_target((1000, 880), radius=310),
    ]
    for index, image in enumerate(images, start=1):
        assert cv2.imwrite(str(input_dir / f"raw_{index}.jpg"), image)

    manifest = normalize_sequence(input_dir, output_dir, width=320, height=240, padding_ratio=1.2, save_debug=True)

    assert len(manifest) == 3
    assert (output_dir / "manifest.csv").exists()
    for index in range(1, 4):
        output_image = cv2.imread(str(output_dir / f"frame_{index:04d}.jpg"))
        assert output_image.shape[:2] == (240, 320)
    with (output_dir / "manifest.csv").open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3


def test_normalize_sequence_center_crop_fallback_handles_blank_image(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    blank = np.full((300, 500, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(input_dir / "blank.jpg"), blank)

    manifest = normalize_sequence(input_dir, output_dir, width=200, height=200, fallback="center-crop")

    assert manifest[0]["registration_success"] is False
    output_image = cv2.imread(str(output_dir / "frame_0001.jpg"))
    assert output_image.shape[:2] == (200, 200)


def test_normalize_sequence_fail_fallback_raises_on_blank_image(tmp_path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    blank = np.full((300, 500, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(input_dir / "blank.jpg"), blank)

    with pytest.raises(ValueError):
        normalize_sequence(input_dir, output_dir, fallback="fail")
