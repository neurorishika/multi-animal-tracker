"""Per-detection identity classifiers: AprilTag, CNN, and head-tail direction."""

from multi_tracker.core.identity.classification.apriltag import (
    AprilTagConfig,
    AprilTagDetector,
)
from multi_tracker.core.identity.classification.cnn import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
    TrackCNNHistory,
    apply_cnn_identity_cost,
)
from multi_tracker.core.identity.classification.headtail import HeadTailAnalyzer

__all__ = [
    "AprilTagConfig",
    "AprilTagDetector",
    "CNNIdentityBackend",
    "CNNIdentityCache",
    "CNNIdentityConfig",
    "ClassPrediction",
    "TrackCNNHistory",
    "apply_cnn_identity_cost",
    "HeadTailAnalyzer",
]
