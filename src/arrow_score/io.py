"""Path-level image input sources for TASK 1A."""

from pathlib import Path
from typing import Iterable

from arrow_score.types import Frame, ImagePair

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_supported_image(path: Path) -> bool:
    """Return whether a path has a supported image suffix."""
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


class ImageSequenceSource:
    """Build neighboring image pairs from a directory of image paths."""

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = Path(input_dir)
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

        self.paths = sorted(path for path in self.input_dir.iterdir() if path.is_file() and _is_supported_image(path))
        if len(self.paths) < 2:
            raise ValueError(f"At least 2 supported images are required in {self.input_dir}")

    def iter_pairs(self) -> Iterable[ImagePair]:
        """Yield adjacent image pairs sorted by file name."""
        for index, (prev_path, curr_path) in enumerate(zip(self.paths, self.paths[1:])):
            yield ImagePair(prev=Frame(prev_path), curr=Frame(curr_path), index=index)


class ImagePairSource:
    """Build a single before/after image pair from two paths."""

    def __init__(self, prev_path: Path, curr_path: Path) -> None:
        self.prev_path = Path(prev_path)
        self.curr_path = Path(curr_path)
        for path in (self.prev_path, self.curr_path):
            if not path.exists():
                raise FileNotFoundError(f"Image file does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Image path is not a file: {path}")
            if not _is_supported_image(path):
                raise ValueError(f"Unsupported image suffix for {path}; supported: {sorted(SUPPORTED_IMAGE_SUFFIXES)}")

    def iter_pairs(self) -> Iterable[ImagePair]:
        """Yield the single configured image pair."""
        yield ImagePair(prev=Frame(self.prev_path), curr=Frame(self.curr_path), index=0)
