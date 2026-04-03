from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Prevent heavy transitive imports (cv2, torch, etc.) that may not be
# installed in lightweight test environments.  We only need contracts +
# runner, which have no such hard dependencies themselves.
for _mod in ("cv2", "torch", "torchvision"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from hydra_suite.training.contracts import (  # noqa: E402
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command  # noqa: E402


def test_auto_batch_flag():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640, batch=-1),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "batch=-1" in cmd_str


def test_multi_gpu_device():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        device="0,1",
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "device=0,1" in cmd_str
