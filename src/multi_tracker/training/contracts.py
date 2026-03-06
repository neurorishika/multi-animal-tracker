"""Training contracts for MAT multi-role YOLO workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TrainingRole(str, Enum):
    """Canonical training roles supported by MAT."""

    OBB_DIRECT = "obb_direct"
    SEQ_DETECT = "seq_detect"
    SEQ_CROP_OBB = "seq_crop_obb"
    # Deprecated: classification training has moved to ClassKit (classkit-labeler).
    # Kept here for registry backwards-compatibility only — not exposed in any dialog.
    HEADTAIL_YOLO = "headtail_yolo"
    HEADTAIL_TINY = "headtail_tiny"

    # ClassKit classification roles
    CLASSIFY_FLAT_YOLO = "classify_flat_yolo"
    CLASSIFY_FLAT_TINY = "classify_flat_tiny"
    CLASSIFY_MULTIHEAD_YOLO = "classify_multihead_yolo"
    CLASSIFY_MULTIHEAD_TINY = "classify_multihead_tiny"


@dataclass(slots=True)
class SplitConfig:
    """Dataset split ratios."""

    train: float = 0.8
    val: float = 0.2
    test: float = 0.0


@dataclass(slots=True)
class SourceDataset:
    """Input dataset descriptor."""

    path: str
    source_type: str = "yolo_obb"
    name: str = ""


@dataclass(slots=True)
class TrainingHyperParams:
    """Generic training hyperparameters."""

    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.01
    patience: int = 30
    workers: int = 8
    cache: bool = False


@dataclass(slots=True)
class TinyHeadTailParams:
    """Tiny head-tail trainer hyperparameters."""

    epochs: int = 50
    batch: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    input_width: int = 128
    input_height: int = 64


@dataclass(slots=True)
class AugmentationProfile:
    """Augmentation settings for Ultralytics training."""

    enabled: bool = True
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PublishPolicy:
    """Post-training artifact publishing policy."""

    auto_import: bool = True
    auto_select: bool = False


@dataclass(slots=True)
class TrainingRunSpec:
    """Full run spec persisted to local registry."""

    role: TrainingRole
    source_datasets: list[SourceDataset]
    derived_dataset_dir: str
    base_model: str
    hyperparams: TrainingHyperParams
    device: str = "auto"
    seed: int = 42
    augmentation_profile: AugmentationProfile = field(
        default_factory=AugmentationProfile
    )
    publish_policy: PublishPolicy = field(default_factory=PublishPolicy)
    tiny_params: TinyHeadTailParams = field(default_factory=TinyHeadTailParams)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["role"] = self.role.value
        return out


@dataclass(slots=True)
class ValidationIssue:
    """Structured validation issue entry."""

    severity: str
    code: str
    message: str
    path: str = ""


@dataclass(slots=True)
class ValidationReport:
    """Validation summary for preflight checks."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "issues": [asdict(i) for i in self.issues],
            "stats": dict(self.stats),
        }


@dataclass(slots=True)
class DatasetBuildResult:
    """Result of a dataset-build stage."""

    dataset_dir: str
    stats: dict[str, Any] = field(default_factory=dict)
    manifest_path: str = ""
