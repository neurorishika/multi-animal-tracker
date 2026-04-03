from __future__ import annotations

from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_detached_command_is_valid():
    """Detached training reuses the same command -- it is just launched differently."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=100, imgsz=640),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    assert len(cmd) > 0
    assert "train" in cmd
    assert any("epochs=100" in arg for arg in cmd)
