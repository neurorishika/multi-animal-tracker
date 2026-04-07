"""Per-detection identity classifiers: AprilTag, CNN, and head-tail direction."""

from hydra_suite.core.identity.classification.apriltag import (
    AprilTagConfig,
    AprilTagDetector,
)
from hydra_suite.core.identity.classification.cnn import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
    TrackCNNHistory,
    apply_cnn_identity_cost,
)
from hydra_suite.core.identity.classification.headtail import HeadTailAnalyzer

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
