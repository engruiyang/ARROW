"""Configuration dataclasses for the TASK 1C demo pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DiffConfig:
    """Parameters for basic frame differencing."""

    blur_kernel: int = 5
    threshold: int = 25
    min_area: int = 20
    morph_kernel: int = 3
    dilate_iterations: int = 1
    erode_iterations: int = 1

    def __post_init__(self) -> None:
        """Validate differencing parameters."""
        if self.blur_kernel <= 0 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer")
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.min_area <= 0:
            raise ValueError("min_area must be greater than 0")
        if self.morph_kernel <= 0:
            raise ValueError("morph_kernel must be greater than 0")
        if self.dilate_iterations < 0:
            raise ValueError("dilate_iterations must be non-negative")
        if self.erode_iterations < 0:
            raise ValueError("erode_iterations must be non-negative")


@dataclass(frozen=True)
class ArrowDetectorConfig:
    """Parameters for simple contour-based arrow detection."""

    min_area: int = 20
    min_aspect_ratio: float = 2.0
    min_line_length: float = 10.0
    max_candidates: int = 5
    impact_policy: str = "near_center"

    def __post_init__(self) -> None:
        """Validate arrow detector parameters."""
        if self.min_area <= 0:
            raise ValueError("min_area must be greater than 0")
        if self.min_aspect_ratio < 1:
            raise ValueError("min_aspect_ratio must be at least 1")
        if self.min_line_length <= 0:
            raise ValueError("min_line_length must be greater than 0")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be greater than 0")
        if self.impact_policy != "near_center":
            raise ValueError("impact_policy currently only supports 'near_center'")


@dataclass(frozen=True)
class PipelineDebugConfig:
    """Flags controlling optional pipeline debug image outputs."""

    save_debug_images: bool = False
    save_masks: bool = False
    save_bbox_debug: bool = False
    save_candidate_debug: bool = False
