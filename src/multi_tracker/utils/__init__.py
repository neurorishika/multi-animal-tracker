"""
Utility modules for the Multi-Animal Tracker.

This package contains various utility functions and classes for image processing,
CSV writing, ROI handling, video processing, and other common operations.
"""

from .csv_writer import CSVWriterThread
from .image_processing import apply_image_adjustments, stabilize_lighting
from .geometry import fit_circle_to_points, wrap_angle_degs

__all__ = [
    "CSVWriterThread",
    "apply_image_adjustments",
    "stabilize_lighting",
    "fit_circle_to_points",
    "wrap_angle_degs",
]
