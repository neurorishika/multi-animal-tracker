from __future__ import annotations

from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_resume_flag_in_command():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="/tmp/runs/last.pt",
        hyperparams=TrainingHyperParams(epochs=200, imgsz=640),
        resume_from="/tmp/runs/last.pt",
    )
    cmd = build_ultralytics_command(spec, "/tmp/run2")
    cmd_str = " ".join(cmd)
    assert "resume=True" in cmd_str
    assert "model=/tmp/runs/last.pt" in cmd_str


def test_no_resume_flag_by_default():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=100, imgsz=640),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "resume" not in cmd_str
