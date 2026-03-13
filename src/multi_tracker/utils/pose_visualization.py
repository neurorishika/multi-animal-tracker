from __future__ import annotations

import math


def normalize_pose_render_min_conf(value: object, default: float = 0.2) -> float:
    """Return a finite, non-negative confidence threshold for pose rendering."""
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    if not math.isfinite(threshold):
        threshold = float(default)
    return max(0.0, threshold)


def is_renderable_pose_keypoint(
    x_value: object,
    y_value: object,
    confidence_value: object,
    min_confidence: object = 0.0,
) -> bool:
    """Return True only for finite coordinates with strictly positive confidence."""
    try:
        x_coord = float(x_value)
        y_coord = float(y_value)
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        return False

    threshold = normalize_pose_render_min_conf(min_confidence, default=0.0)
    return (
        math.isfinite(x_coord)
        and math.isfinite(y_coord)
        and math.isfinite(confidence)
        and confidence > 0.0
        and confidence >= threshold
    )


__all__ = [
    "is_renderable_pose_keypoint",
    "normalize_pose_render_min_conf",
]
