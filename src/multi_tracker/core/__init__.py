"""
Core tracking algorithms and components for the Multi-Animal-Tracker.

This package contains the tracking worker and supporting components for
multi-object tracking including Kalman filters, background models,
object detection, and track assignment.
"""

from .tracking.worker import TrackingWorker
from .filters.kalman import KalmanFilterManager
from .background.model import BackgroundModel
from .detectors.engine import ObjectDetector
from .assigners.hungarian import TrackAssigner
from .post.processing import (
    process_trajectories,
    process_trajectories_from_csv,
    resolve_trajectories,
    interpolate_trajectories,
)
from .identity.analysis import IdentityProcessor, IndividualDatasetGenerator


__all__ = [
    "TrackingWorker",
    "KalmanFilterManager",
    "BackgroundModel",
    "ObjectDetector",
    "TrackAssigner",
    "process_trajectories",
    "process_trajectories_from_csv",
    "resolve_trajectories",
    "interpolate_trajectories",
    "IdentityProcessor",
    "IndividualDatasetGenerator",
]
