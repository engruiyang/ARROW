import cv2
import numpy as np
import pytest

from arrow_score.image_loader import load_image


def test_load_image_reads_bgr_ndarray(tmp_path) -> None:
    path = tmp_path / "image.png"
    image = np.full((10, 12, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == image.shape


def test_load_image_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")


def test_load_image_non_image_file_raises(tmp_path) -> None:
    path = tmp_path / "not_image.txt"
    path.write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError):
        load_image(path)
