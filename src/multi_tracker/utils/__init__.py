"""
Utility modules for the Multi-Animal-Tracker.

This package contains various utility functions and classes for image processing,
CSV writing, ROI handling, video processing, GPU acceleration, and other common operations.
"""

from .image_processing import apply_image_adjustments, stabilize_lighting
from .geometry import fit_circle_to_points, wrap_angle_degs
from .frame_prefetcher import FramePrefetcher, FramePrefetcherBackward
from .gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    GPU_AVAILABLE,
    NUMBA_AVAILABLE,
    get_device_info,
    log_device_info,
)

__all__ = [
    "apply_image_adjustments",
    "stabilize_lighting",
    "fit_circle_to_points",
    "wrap_angle_degs",
    "FramePrefetcher",
    "FramePrefetcherBackward",
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "GPU_AVAILABLE",
    "NUMBA_AVAILABLE",
    "get_device_info",
    "log_device_info",
]
