import pytest

from arrow_score.io import ImagePairSource, ImageSequenceSource


def test_image_sequence_source_generates_adjacent_pairs_sorted(tmp_path) -> None:
    (tmp_path / "b.PNG").write_bytes(b"")
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "c.bmp").write_bytes(b"")

    pairs = list(ImageSequenceSource(tmp_path).iter_pairs())

    assert len(pairs) == 2
    assert pairs[0].prev.path.name == "a.jpg"
    assert pairs[0].curr.path.name == "b.PNG"
    assert pairs[1].prev.path.name == "b.PNG"
    assert pairs[1].curr.path.name == "c.bmp"
    assert pairs[0].prev.image is None


def test_image_sequence_source_requires_at_least_two_images(tmp_path) -> None:
    (tmp_path / "only.jpeg").write_bytes(b"")
    with pytest.raises(ValueError):
        ImageSequenceSource(tmp_path)


def test_image_sequence_source_missing_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageSequenceSource(tmp_path / "missing")


def test_image_pair_source_generates_one_pair(tmp_path) -> None:
    prev = tmp_path / "prev.jpg"
    curr = tmp_path / "curr.png"
    prev.write_bytes(b"")
    curr.write_bytes(b"")

    pairs = list(ImagePairSource(prev, curr).iter_pairs())

    assert len(pairs) == 1
    assert pairs[0].index == 0
    assert pairs[0].prev.path == prev
    assert pairs[0].curr.path == curr
    assert pairs[0].curr.image is None


def test_image_pair_source_missing_file_raises(tmp_path) -> None:
    prev = tmp_path / "prev.jpg"
    prev.write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        ImagePairSource(prev, tmp_path / "missing.jpg")
