"""MAT training framework for role-aware YOLO workflows."""

from .contracts import (
    AugmentationProfile,
    CustomCNNParams,
    DatasetBuildResult,
    PublishPolicy,
    SourceDataset,
    SplitConfig,
    TinyHeadTailParams,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
    ValidationIssue,
    ValidationReport,
)
from .service import RoleRunConfig, TrainingOrchestrator, TrainingSessionResult

__all__ = [
    "AugmentationProfile",
    "CustomCNNParams",
    "DatasetBuildResult",
    "PublishPolicy",
    "RoleRunConfig",
    "SourceDataset",
    "SplitConfig",
    "TinyHeadTailParams",
    "TrainingHyperParams",
    "TrainingOrchestrator",
    "TrainingRole",
    "TrainingRunSpec",
    "TrainingSessionResult",
    "ValidationIssue",
    "ValidationReport",
]
