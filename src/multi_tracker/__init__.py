"""
Multi-Animal-Tracker Package

A comprehensive solution for tracking multiple animals in video recordings using computer vision techniques.
The system combines background subtraction, Kalman filtering, and Hungarian algorithm for robust multi-object tracking.

Key Features:
- Real-time multi-animal tracking (up to 20 targets)
- Circular Region of Interest (ROI) definition for arena-based experiments
- Kalman filter-based motion prediction and state estimation
- Hungarian algorithm for optimal target-detection assignment
- Background subtraction using "lightest pixel" method
- Orientation tracking with anti-flip mechanisms
- CSV data export with trajectory information
- Real-time preview and parameter adjustment
- Configurable tracking parameters with persistence
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .app.launcher import main, parse_arguments, setup_logging

try:
    from .core.tracking.worker import TrackingWorker
    from .gui.main_window import MainWindow

    __all__ = ["main", "TrackingWorker", "MainWindow"]
except ImportError:
    # During development, some modules might not be available yet
    __all__ = ["main"]
