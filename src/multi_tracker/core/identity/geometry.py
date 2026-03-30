"""Shared geometry utilities for the identity module."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def resolve_directed_angle(
    theta: float,
    heading_hint: Optional[float] = None,
    heading_directed: bool = False,
    vx: Optional[float] = None,
    vy: Optional[float] = None,
) -> Tuple[float, bool, str]:
    """Resolve the best directed orientation angle for a crop.

    Priority:
    1. Head-tail model heading (heading_directed=True and finite heading_hint).
    2. Motion velocity (vx, vy non-negligible) — disambiguates theta ± π.
    3. OBB axis angle (undirected, 180° ambiguity retained).

    The returned angle points tail → head so that the affine-warp canonicalization
    places the head on the right side (+x) of the canonical crop.

    Returns:
        (angle_rad, is_directed, source_str)
    """
    # Priority 1: head-tail model
    if (
        heading_directed
        and heading_hint is not None
        and math.isfinite(float(heading_hint))
    ):
        return float(heading_hint) % (2.0 * math.pi), True, "head_tail_model"

    # Priority 2: motion velocity disambiguates OBB axis
    if vx is not None and vy is not None:
        fvx, fvy = float(vx), float(vy)
        if math.isfinite(fvx) and math.isfinite(fvy) and (fvx * fvx + fvy * fvy) > 1e-6:
            motion_angle = math.atan2(fvy, fvx) % (2.0 * math.pi)
            theta0 = float(theta) % (2.0 * math.pi)
            theta1 = (theta0 + math.pi) % (2.0 * math.pi)
            diff0 = abs(((motion_angle - theta0 + math.pi) % (2.0 * math.pi)) - math.pi)
            diff1 = abs(((motion_angle - theta1 + math.pi) % (2.0 * math.pi)) - math.pi)
            resolved = theta0 if diff0 <= diff1 else theta1
            return resolved, True, "motion_velocity"

    # Priority 3: undirected OBB axis (180° ambiguous)
    return float(theta) % (2.0 * math.pi), False, "tracking_theta"


def ellipse_to_obb_corners(
    cx: float, cy: float, major_axis: float, minor_axis: float, theta: float
) -> np.ndarray:
    """Convert ellipse parameters to OBB corner points.

    The OBB is the oriented bounding box that exactly fits the ellipse,
    which is a rotated rectangle with dimensions (major_axis x minor_axis).

    Args:
        cx, cy: Center coordinates of the ellipse
        major_axis: Full length of the major axis (not semi-axis)
        minor_axis: Full length of the minor axis (not semi-axis)
        theta: Rotation angle in radians (orientation of major axis)

    Returns:
        corners: numpy array of shape (4, 2) with corner coordinates
    """
    a = major_axis / 2.0
    b = minor_axis / 2.0
    local_corners = np.array([[a, b], [-a, b], [-a, -b], [a, -b]], dtype=np.float32)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    return (local_corners @ rot.T + np.array([cx, cy], dtype=np.float32)).astype(
        np.float32
    )


def ellipse_axes_from_area(area: float, aspect_ratio: float) -> Tuple[float, float]:
    """Compute ellipse major/minor axes from area and aspect ratio.

    Returns:
        (major_axis, minor_axis)
    """
    minor = math.sqrt(4.0 * float(area) / (math.pi * float(aspect_ratio)))
    major = float(aspect_ratio) * minor
    return major, minor
