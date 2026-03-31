"""Shared compute runtime selection/resolution utilities."""

from .compute_runtime import (
    CANONICAL_RUNTIMES,
    allowed_runtimes_for_pipelines,
    derive_detection_runtime_settings,
    derive_pose_runtime_settings,
    infer_compute_runtime_from_legacy,
    runtime_label,
    supported_runtimes_for_pipeline,
)

__all__ = [
    "CANONICAL_RUNTIMES",
    "runtime_label",
    "supported_runtimes_for_pipeline",
    "allowed_runtimes_for_pipelines",
    "infer_compute_runtime_from_legacy",
    "derive_detection_runtime_settings",
    "derive_pose_runtime_settings",
]
