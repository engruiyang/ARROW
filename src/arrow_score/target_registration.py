"""Automatic 90-target registration and canonical target normalization."""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class HsvRange:
    """One inclusive OpenCV HSV threshold range."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


HSV_RANGES: dict[str, tuple[HsvRange, ...]] = {
    "blue": (HsvRange((90, 60, 40), (130, 255, 255)),),
    "red": (HsvRange((0, 60, 40), (10, 255, 255)), HsvRange((170, 60, 40), (179, 255, 255))),
    "yellow": (HsvRange((18, 60, 40), (40, 255, 255)),),
}


@dataclass(frozen=True)
class TargetPose:
    """Detected target pose in original image coordinates."""

    center_px: tuple[float, float]
    ellipse_axes_px: tuple[float, float]
    ellipse_angle_deg: float
    confidence: str
    debug_reason: str | None = None
    detected_features: dict[str, object] | None = None


@dataclass(frozen=True)
class CanonicalTargetConfig:
    """Canonical 90-target coordinate system configuration."""

    target_type: str = "90"
    canonical_size_px: int = 980
    outer_radius_px: int = 490
    center_px: tuple[int, int] = (490, 490)


@dataclass(frozen=True)
class RegistrationDebugOutput:
    """Registration result plus optional debug artifacts."""

    pose: TargetPose
    color_mask: np.ndarray | None
    cross_mask: np.ndarray | None
    debug_image: np.ndarray | None


@dataclass(frozen=True)
class _EllipseCandidate:
    """Internal ellipse candidate with color and contour area metadata."""

    color: str
    center: tuple[float, float]
    axes: tuple[float, float]
    angle: float
    area: float


@dataclass(frozen=True)
class _RegistrationComputation:
    """Internal registration artifacts reused by normal and debug paths."""

    pose: TargetPose
    color_mask: np.ndarray
    cross_mask: np.ndarray | None
    ellipses: list[_EllipseCandidate] = field(default_factory=list)


class HsvTargetRegistrar:
    """Detect colored 90-target rings and warp targets to canonical coordinates."""

    def __init__(
        self,
        canonical_config: CanonicalTargetConfig | None = None,
        min_color_area: int = 5000,
        min_ellipse_points: int = 20,
        center_refine_with_cross: bool = True,
    ) -> None:
        if min_color_area <= 0:
            raise ValueError("min_color_area must be greater than 0")
        if min_ellipse_points < 5:
            raise ValueError("min_ellipse_points must be at least 5 for cv2.fitEllipse")
        self.canonical_config = canonical_config or CanonicalTargetConfig()
        self.min_color_area = min_color_area
        self.min_ellipse_points = min_ellipse_points
        self.center_refine_with_cross = center_refine_with_cross

    def detect_pose(self, image: np.ndarray) -> TargetPose:
        """Detect target pose from HSV colored rings and optional center cross."""
        return self._compute_registration(image).pose

    def detect_pose_debug(self, image: np.ndarray) -> RegistrationDebugOutput:
        """Detect target pose and return masks plus a debug image."""
        computation = self._compute_registration(image)
        debug_image = image.copy()
        for ellipse in computation.ellipses:
            center = (int(round(ellipse.center[0])), int(round(ellipse.center[1])))
            axes = (int(round(ellipse.axes[0])), int(round(ellipse.axes[1])))
            color = (255, 0, 0) if ellipse.color == "blue" else (0, 0, 255) if ellipse.color == "red" else (0, 255, 255)
            cv2.ellipse(debug_image, center, axes, ellipse.angle, 0, 360, color, 2)
        pose = computation.pose
        center = (int(round(pose.center_px[0])), int(round(pose.center_px[1])))
        axes = (int(round(pose.ellipse_axes_px[0])), int(round(pose.ellipse_axes_px[1])))
        cv2.ellipse(debug_image, center, axes, pose.ellipse_angle_deg, 0, 360, (0, 255, 0), 3)
        cv2.circle(debug_image, center, 6, (0, 0, 0), -1)
        cv2.putText(debug_image, f"confidence={pose.confidence}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if pose.debug_reason:
            cv2.putText(debug_image, pose.debug_reason[:80], (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        return RegistrationDebugOutput(pose=pose, color_mask=computation.color_mask, cross_mask=computation.cross_mask, debug_image=debug_image)

    def warp_to_canonical(self, image: np.ndarray, pose: TargetPose) -> np.ndarray:
        """Warp an original image into the canonical target coordinate system."""
        matrix = self._original_to_canonical_matrix(pose)[:2]
        size = self.canonical_config.canonical_size_px
        return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    def original_to_canonical_points(self, points: np.ndarray, pose: TargetPose) -> np.ndarray:
        """Map points from original image coordinates to canonical coordinates."""
        return _transform_points(points, self._original_to_canonical_matrix(pose))

    def canonical_to_original_points(self, points: np.ndarray, pose: TargetPose) -> np.ndarray:
        """Map points from canonical coordinates back to original image coordinates."""
        matrix = np.linalg.inv(self._original_to_canonical_matrix(pose))
        return _transform_points(points, matrix)

    def _compute_registration(self, image: np.ndarray) -> _RegistrationComputation:
        """Compute target registration and reusable debug artifacts."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("target registration expects a BGR image with 3 channels")
        color_masks = _build_color_masks(image)
        color_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in color_masks.values():
            color_mask = cv2.bitwise_or(color_mask, mask)
        if int(cv2.countNonZero(color_mask)) < self.min_color_area:
            raise ValueError("target registration failed: color mask area is insufficient")

        ellipses = self._find_ellipses(color_masks)
        if not ellipses:
            raise ValueError("target registration failed: no reliable colored ellipses found")

        centers = np.array([ellipse.center for ellipse in ellipses], dtype=np.float32)
        rough_center = tuple(float(value) for value in np.median(centers, axis=0))
        outer = max(ellipses, key=lambda ellipse: ellipse.area)
        center = rough_center
        cross_mask = None
        cross_center = None
        if self.center_refine_with_cross:
            cross_center = detect_center_cross(image, rough_center)
            if cross_center is not None:
                center = cross_center
                cross_mask = _center_cross_mask(image, rough_center)

        confidence = "high" if len(ellipses) >= 3 and cross_center is not None else "medium" if len(ellipses) >= 2 else "low"
        reason = f"ellipses={len(ellipses)};outer_color={outer.color}"
        if cross_center is None:
            reason += ";cross=missing"
        pose = TargetPose(
            center_px=(float(center[0]), float(center[1])),
            ellipse_axes_px=(float(outer.axes[0]), float(outer.axes[1])),
            ellipse_angle_deg=float(outer.angle),
            confidence=confidence,
            debug_reason=reason,
            detected_features={"ellipse_count": len(ellipses), "outer_color": outer.color, "cross_center": cross_center},
        )
        return _RegistrationComputation(pose=pose, color_mask=color_mask, cross_mask=cross_mask, ellipses=ellipses)

    def _find_ellipses(self, color_masks: dict[str, np.ndarray]) -> list[_EllipseCandidate]:
        """Find colored ellipse candidates from HSV masks."""
        ellipses: list[_EllipseCandidate] = []
        for color, mask in color_masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < self.min_color_area or contour.shape[0] < self.min_ellipse_points:
                    continue
                (cx, cy), (width, height), angle = cv2.fitEllipse(contour)
                axes = (float(width) / 2.0, float(height) / 2.0)
                if min(axes) <= 0:
                    continue
                ellipses.append(_EllipseCandidate(color=color, center=(float(cx), float(cy)), axes=axes, angle=float(angle), area=area))
        return ellipses

    def _original_to_canonical_matrix(self, pose: TargetPose) -> np.ndarray:
        """Build a homogeneous affine matrix from original to canonical coordinates."""
        cx, cy = pose.center_px
        axis_x, axis_y = pose.ellipse_axes_px
        if axis_x <= 0 or axis_y <= 0:
            raise ValueError(f"pose ellipse axes must be positive, got {pose.ellipse_axes_px}")
        ccx, ccy = self.canonical_config.center_px
        radius = float(self.canonical_config.outer_radius_px)
        angle = np.deg2rad(-pose.ellipse_angle_deg)
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        translate_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
        rotate = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float64)
        scale = np.array([[radius / axis_x, 0, 0], [0, radius / axis_y, 0], [0, 0, 1]], dtype=np.float64)
        translate_to_canonical = np.array([[1, 0, ccx], [0, 1, ccy], [0, 0, 1]], dtype=np.float64)
        return translate_to_canonical @ scale @ rotate @ translate_to_origin


def detect_center_cross(image: np.ndarray, rough_center: tuple[float, float], search_radius: int = 160) -> tuple[float, float] | None:
    """Estimate the dark center cross near rough_center, returning None on failure."""
    mask = _center_cross_mask(image, rough_center, search_radius)
    if mask is None or int(cv2.countNonZero(mask)) == 0:
        return None
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        return None
    return (float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]))


def _center_cross_mask(image: np.ndarray, rough_center: tuple[float, float], search_radius: int = 160) -> np.ndarray | None:
    """Build a dark-structure mask around the rough target center."""
    height, width = image.shape[:2]
    cx, cy = int(round(rough_center[0])), int(round(rough_center[1]))
    x0, x1 = max(0, cx - search_radius), min(width, cx + search_radius)
    y0, y1 = max(0, cy - search_radius), min(height, cy + search_radius)
    if x1 <= x0 or y1 <= y0:
        return None
    roi = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 21))
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel_h)
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel_v)
    local_mask = cv2.bitwise_or(horizontal, vertical)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = local_mask
    return mask


def _build_color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    """Build cleaned HSV masks for configured target colors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    masks: dict[str, np.ndarray] = {}
    for color, ranges in HSV_RANGES.items():
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for hsv_range in ranges:
            lower = np.array(hsv_range.lower, dtype=np.uint8)
            upper = np.array(hsv_range.upper, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masks[color] = mask
    return masks


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an Nx2 point array."""
    points_array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points_array, np.ones(points_array.shape[0], dtype=np.float64)])
    transformed = homogeneous @ matrix.T
    return transformed[:, :2]
