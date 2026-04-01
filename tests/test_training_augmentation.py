from __future__ import annotations

from multi_tracker.training.contracts import (
    AugmentationProfile,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from multi_tracker.training.runner import build_ultralytics_command


def test_augmentation_args_passed_to_command():
    """Augmentation args dict entries appear as CLI flags."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=True,
            args={"flipud": 0.3, "mosaic": 0.0, "mixup": 0.1},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud=0.3" in cmd_str
    assert "mosaic=0.0" in cmd_str
    assert "mixup=0.1" in cmd_str


def test_augmentation_disabled_skips_args():
    """When enabled=False, no augmentation args are emitted."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=False,
            args={"flipud": 0.3},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud" not in cmd_str
