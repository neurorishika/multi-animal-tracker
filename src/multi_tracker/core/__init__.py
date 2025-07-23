"""
Core tracking algorithms and components for the Multi-Animal Tracker.

This package contains the tracking worker and supporting components for
multi-object tracking including Kalman filters, background models,
object detection, and track assignment.
"""
from .tracking_worker import TrackingWorker
from .kalman_filters import KalmanFilterManager
from .background_models import BackgroundModel
from .detection import ObjectDetector
from .assignment import TrackAssigner
from .post_processing import process_trajectories


__all__ = [
    "TrackingWorker",
    "KalmanFilterManager",
    "BackgroundModel",
    "ObjectDetector",
    "TrackAssigner",
    "process_trajectories"
]