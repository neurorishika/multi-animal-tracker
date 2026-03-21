# ClassKit Extended Training — Custom CNN Models

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `flat_custom`/`multihead_custom` training modes to ClassKit that unify TinyClassifier and pretrained torchvision backbones (ConvNeXt, EfficientNet, ResNet, ViT) under a single UI surface, with configurable layer freezing for linear-probe or full fine-tuning.

**Architecture:** A new `training/torchvision_model.py` module owns all torchvision model construction and ONNX export; `_train_custom_classify()` in `runner.py` dispatches to it or to the existing `_train_tiny_classify()`; `CustomCNNParams` in `contracts.py` carries hyperparameters; new training modes are wired into the dialog, preset, and inference dispatch.

**Tech Stack:** Python, PyTorch (`torchvision.models`), `torch.onnx`, `onnxruntime`, PySide6, existing ClassKit training infrastructure.

**File Map:**

| File | Action | Role |
|---|---|---|
| `src/multi_tracker/training/contracts.py` | Modify | Add `CustomCNNParams` dataclass, two new `TrainingRole` values, `custom_params` field on `TrainingRunSpec` |
| `src/multi_tracker/training/torchvision_model.py` | Create | Backbone factory, layer-freezing, ONNX export, checkpoint I/O |
| `src/multi_tracker/training/runner.py` | Modify | Add `_train_custom_classify()`, update `run_training()` dispatch |
| `src/multi_tracker/training/model_publish.py` | Modify | Add repo-dir and task-usage entries for the two new roles |
| `src/multi_tracker/classkit/presets.py` | Modify | Add `flat_custom`/`multihead_custom` to all preset `training_modes` |
| `src/multi_tracker/classkit/jobs/task_workers.py` | Modify | Add `TorchvisionInferenceWorker` |
| `src/multi_tracker/classkit/gui/mainwindow.py` | Modify | Update `_load_checkpoint_from_path` dispatch and `train_classifier` role map |
| `src/multi_tracker/classkit/gui/dialogs.py` | Modify | Add Custom CNN tab to `ClassKitTrainingDialog` |
| `tests/test_classkit_extended_training.py` | Create | Full test suite for all new logic |

---

## Task 1 — `CustomCNNParams`, `TrainingRole` additions, `TrainingRunSpec.custom_params`

**Files:**
- Modify: `src/multi_tracker/training/contracts.py`
- Create (start): `tests/test_classkit_extended_training.py`

### Step 1.1 — Write failing tests

- [ ] Create `tests/test_classkit_extended_training.py` with the following content:

```python
"""Tests for ClassKit extended training — Custom CNN models."""
from __future__ import annotations


def test_custom_cnn_params_defaults():
    from multi_tracker.training.contracts import CustomCNNParams
    p = CustomCNNParams()
    assert p.backbone == "tinyclassifier"
    assert p.trainable_layers == 0
    assert p.backbone_lr_scale == 0.1
    assert p.input_size == 224
    assert p.epochs == 50
    assert p.batch == 32
    assert p.lr == 1e-3
    assert p.weight_decay == 1e-2
    assert p.patience == 10
    assert p.label_smoothing == 0.0
    assert p.class_rebalance_mode == "none"
    assert p.class_rebalance_power == 1.0
    assert p.hidden_layers == 1
    assert p.hidden_dim == 64
    assert p.dropout == 0.2
    assert p.input_width == 128
    assert p.input_height == 64


def test_new_training_roles_exist():
    from multi_tracker.training.contracts import TrainingRole
    assert TrainingRole.CLASSIFY_FLAT_CUSTOM.value == "classify_flat_custom"
    assert TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM.value == "classify_multihead_custom"


def test_training_run_spec_has_custom_params():
    from multi_tracker.training.contracts import TrainingRunSpec, TrainingRole, CustomCNNParams
    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
    )
    assert hasattr(spec, "custom_params")
    assert spec.custom_params is None

    spec2 = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="convnext_tiny"),
    )
    assert spec2.custom_params.backbone == "convnext_tiny"
```

### Step 1.2 — Run tests to verify they fail

- [ ] Run:
```bash
cd "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker"
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py -v 2>&1 | head -20
```
Expected: `ImportError` or `AttributeError` — `CustomCNNParams` does not exist yet.

### Step 1.3 — Add `CustomCNNParams` to `contracts.py`

- [ ] Open `src/multi_tracker/training/contracts.py`. The existing `TinyHeadTailParams` ends at line 76. Add `CustomCNNParams` immediately after it (before `AugmentationProfile`):

```python
@dataclass(slots=True)
class CustomCNNParams:
    """Hyperparameters for the unified Custom CNN training mode.

    Covers both TinyClassifier (backbone='tinyclassifier') and pretrained
    torchvision backbones (ConvNeXt, EfficientNet, ResNet, ViT).
    TinyClassifier-specific fields (hidden_layers, hidden_dim, dropout,
    input_width, input_height) are ignored when backbone != 'tinyclassifier'.
    """
    backbone: str = "tinyclassifier"
    trainable_layers: int = 0        # 0=frozen, -1=all, N=last N layer groups
    backbone_lr_scale: float = 0.1   # LR multiplier for unfrozen backbone layers
    input_size: int = 224            # Resize target (square) for torchvision backbones
    epochs: int = 50
    batch: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 10
    label_smoothing: float = 0.0
    class_rebalance_mode: str = "none"   # none, weighted_loss, weighted_sampler, both
    class_rebalance_power: float = 1.0
    # TinyClassifier-specific (ignored for torchvision backbones)
    hidden_layers: int = 1
    hidden_dim: int = 64
    dropout: float = 0.2
    input_width: int = 128
    input_height: int = 64
```

- [ ] In the `TrainingRole` enum (lines 10–21), add two new values after `CLASSIFY_MULTIHEAD_TINY`:

```python
    CLASSIFY_FLAT_CUSTOM = "classify_flat_custom"
    CLASSIFY_MULTIHEAD_CUSTOM = "classify_multihead_custom"
```

- [ ] In `TrainingRunSpec` (currently lines 108–129), add after the existing `tiny_params` field:

```python
    custom_params: CustomCNNParams | None = None
```

Note: `TrainingRunSpec` uses `@dataclass(slots=True)`. The new field has a default value of `None`, so it must appear after all fields without defaults. Verify that `tiny_params` has a default (`field(default_factory=TinyHeadTailParams)`) — it does. Place `custom_params` after `tiny_params`.

### Step 1.4 — Run tests to verify they pass

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_custom_cnn_params_defaults tests/test_classkit_extended_training.py::test_new_training_roles_exist tests/test_classkit_extended_training.py::test_training_run_spec_has_custom_params -v
```
Expected output:
```
PASSED tests/test_classkit_extended_training.py::test_custom_cnn_params_defaults
PASSED tests/test_classkit_extended_training.py::test_new_training_roles_exist
PASSED tests/test_classkit_extended_training.py::test_training_run_spec_has_custom_params
3 passed
```

### Step 1.5 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/training/contracts.py tests/test_classkit_extended_training.py
git commit -m "$(cat <<'EOF'
feat(classkit): add CustomCNNParams, new TrainingRole values, custom_params to TrainingRunSpec

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — `training/torchvision_model.py`

**Files:**
- Create: `src/multi_tracker/training/torchvision_model.py`
- Test: `tests/test_classkit_extended_training.py` (append)

### Step 2.1 — Write failing tests

- [ ] Append to `tests/test_classkit_extended_training.py`:

```python
import numpy as np
import torch


# ---------------------------------------------------------------------------
# torchvision_model tests
# ---------------------------------------------------------------------------

def test_build_torchvision_classifier_convnext():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("convnext_tiny", num_classes=5, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 5)


def test_build_torchvision_classifier_efficientnet():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("efficientnet_b0", num_classes=3, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 3)


def test_build_torchvision_classifier_resnet():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("resnet18", num_classes=4, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 4)


def test_build_torchvision_classifier_vit():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("vit_b_16", num_classes=6, trainable_layers=0)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 224, 224))
    assert out.shape == (1, 6)


def test_get_layer_groups_convnext_count():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, get_layer_groups
    model = build_torchvision_classifier("convnext_tiny", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "convnext_tiny")
    assert len(groups) == 4


def test_get_layer_groups_resnet_count():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, get_layer_groups
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "resnet18")
    assert len(groups) == 4


def test_get_layer_groups_efficientnet_count():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, get_layer_groups
    model = build_torchvision_classifier("efficientnet_b0", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "efficientnet_b0")
    # EfficientNet-B0 features Sequential has 9 blocks (indices 0–8)
    assert len(groups) == 9


def test_get_layer_groups_vit_count():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, get_layer_groups
    model = build_torchvision_classifier("vit_b_16", num_classes=2, trainable_layers=0)
    groups = get_layer_groups(model, "vit_b_16")
    # ViT-B/16 encoder has 12 transformer layers
    assert len(groups) == 12


def test_freeze_backbone_frozen():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    # Head (fc) must be unfrozen
    assert model.fc.weight.requires_grad is True
    # At least some backbone parameters must be frozen
    backbone_frozen = any(
        not p.requires_grad for name, p in model.named_parameters() if "fc" not in name
    )
    assert backbone_frozen


def test_freeze_backbone_all():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=-1)
    # All parameters must be trainable
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_backbone_partial():
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, get_layer_groups
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=1)
    groups = get_layer_groups(model, "resnet18")
    # Last group (layer4) must be unfrozen
    assert any(p.requires_grad for p in groups[-1].parameters())
    # First group (layer1) must be frozen
    assert all(not p.requires_grad for p in groups[0].parameters())


def test_checkpoint_format_required_keys(tmp_path):
    from multi_tracker.training.torchvision_model import build_torchvision_classifier, save_torchvision_checkpoint
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    ckpt_path = tmp_path / "test.pth"
    save_torchvision_checkpoint(
        model=model,
        arch="resnet18",
        class_names=["a", "b"],
        factor_names=[],
        input_size=(224, 224),
        best_val_acc=0.95,
        history={"train_loss": [], "val_acc": []},
        trainable_layers=0,
        backbone_lr_scale=0.1,
        path=ckpt_path,
    )
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    required = {"arch", "class_names", "factor_names", "input_size", "num_classes",
                "model_state_dict", "best_val_acc", "history", "trainable_layers", "backbone_lr_scale"}
    assert required.issubset(set(ckpt.keys()))


def test_load_torchvision_classifier_roundtrip(tmp_path):
    from multi_tracker.training.torchvision_model import (
        build_torchvision_classifier, save_torchvision_checkpoint, load_torchvision_classifier
    )
    model = build_torchvision_classifier("resnet18", num_classes=3, trainable_layers=0)
    model.eval()
    ckpt_path = tmp_path / "model.pth"
    save_torchvision_checkpoint(
        model=model, arch="resnet18", class_names=["x", "y", "z"],
        factor_names=[], input_size=(224, 224), best_val_acc=0.9,
        history={}, trainable_layers=0, backbone_lr_scale=0.1, path=ckpt_path,
    )
    loaded_model, ckpt = load_torchvision_classifier(str(ckpt_path), device="cpu")
    loaded_model.eval()
    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded_model(x)
    assert torch.allclose(orig_out, loaded_out, atol=1e-5)
    assert ckpt["arch"] == "resnet18"
    assert ckpt["class_names"] == ["x", "y", "z"]


def test_export_torchvision_to_onnx_smoke(tmp_path):
    from multi_tracker.training.torchvision_model import (
        build_torchvision_classifier, save_torchvision_checkpoint, export_torchvision_to_onnx
    )
    model = build_torchvision_classifier("resnet18", num_classes=2, trainable_layers=0)
    ckpt_path = tmp_path / "model.pth"
    save_torchvision_checkpoint(
        model=model, arch="resnet18", class_names=["a", "b"],
        factor_names=[], input_size=(224, 224), best_val_acc=0.8,
        history={}, trainable_layers=0, backbone_lr_scale=0.1, path=ckpt_path,
    )
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    onnx_path = tmp_path / "model.onnx"
    export_torchvision_to_onnx(model, ckpt, onnx_path)
    assert onnx_path.exists()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
    assert out[0].shape == (1, 2)
```

### Step 2.2 — Run tests to verify they fail

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_build_torchvision_classifier_convnext tests/test_classkit_extended_training.py::test_checkpoint_format_required_keys -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'multi_tracker.training.torchvision_model'`

### Step 2.3 — Create `src/multi_tracker/training/torchvision_model.py`

- [ ] Write the complete file at `src/multi_tracker/training/torchvision_model.py`:

```python
"""Torchvision-based classifier: model factory, freezing, ONNX export, checkpoint I/O.

This module is the sole owner of all torchvision backbone construction and
layer-freezing logic for ClassKit's Custom CNN training mode.
All functions are pure Python / PyTorch — no Qt dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

TORCHVISION_BACKBONES: dict[str, str] = {
    "convnext_tiny": "convnext_tiny",
    "convnext_small": "convnext_small",
    "convnext_base": "convnext_base",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b3": "efficientnet_b3",
    "efficientnet_b7": "efficientnet_b7",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit_b_16": "vit_b_16",
}

# Human-readable labels for the GUI
BACKBONE_DISPLAY_NAMES: dict[str, str] = {
    "convnext_tiny": "ConvNeXt-T",
    "convnext_small": "ConvNeXt-S",
    "convnext_base": "ConvNeXt-B",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "efficientnet_b7": "EfficientNet-B7",
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
}


def _load_pretrained(backbone: str) -> nn.Module:
    """Load a pretrained torchvision model by backbone key."""
    weights_map = {
        "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "convnext_small": tvm.ConvNeXt_Small_Weights.IMAGENET1K_V1,
        "convnext_base": tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1,
        "efficientnet_b0": tvm.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b3": tvm.EfficientNet_B3_Weights.IMAGENET1K_V1,
        "efficientnet_b7": tvm.EfficientNet_B7_Weights.IMAGENET1K_V1,
        "resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
        "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V1,
        "vit_b_16": tvm.ViT_B_16_Weights.IMAGENET1K_V1,
    }
    factory = getattr(tvm, backbone)
    return factory(weights=weights_map[backbone])


def _replace_head(model: nn.Module, backbone: str, num_classes: int) -> nn.Module:
    """Replace the final classifier head with a new linear layer."""
    if backbone.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "vit_b_16":
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone for head replacement: {backbone!r}")
    return model


def get_layer_groups(model: nn.Module, backbone: str) -> list[nn.Module]:
    """Return backbone layer groups in shallow-to-deep order.

    The caller can index from the end to unfreeze the last N groups.
    ConvNeXt and ResNet return exactly 4 groups.
    EfficientNet returns individual feature blocks.
    ViT-B/16 returns individual encoder layers.
    """
    if backbone.startswith("convnext"):
        # features[0]=stem, features[1..6]=stages (interleaved norms+stages)
        # Expose the 4 main ConvNeXt stages: indices 1, 3, 5, 7
        return [model.features[i] for i in [1, 3, 5, 7]]
    elif backbone.startswith("resnet"):
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    elif backbone.startswith("efficientnet"):
        # features is a Sequential; expose as individual blocks
        return list(model.features)
    elif backbone == "vit_b_16":
        return list(model.encoder.layers)
    else:
        raise ValueError(f"Unsupported backbone for layer groups: {backbone!r}")


def freeze_backbone(model: nn.Module, backbone: str, trainable_layers: int) -> None:
    """Freeze/unfreeze backbone parameters according to trainable_layers.

    Args:
        model: Model whose backbone parameters to freeze.
        backbone: Backbone key (used to determine head parameter names).
        trainable_layers: 0=frozen, -1=all, N>0=unfreeze last N groups.
    """
    # Step 1: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: always unfreeze the head
    if backbone.startswith("convnext") or backbone.startswith("efficientnet"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif backbone.startswith("resnet"):
        for p in model.fc.parameters():
            p.requires_grad = True
    elif backbone == "vit_b_16":
        for p in model.heads.parameters():
            p.requires_grad = True

    # Step 3: apply backbone unfreezing
    if trainable_layers == -1:
        for p in model.parameters():
            p.requires_grad = True
    elif trainable_layers > 0:
        groups = get_layer_groups(model, backbone)
        for group in groups[-trainable_layers:]:
            for p in group.parameters():
                p.requires_grad = True


def build_torchvision_classifier(
    backbone: str, num_classes: int, trainable_layers: int
) -> nn.Module:
    """Build a pretrained torchvision classifier with a new head.

    Args:
        backbone: One of the keys in TORCHVISION_BACKBONES.
        num_classes: Number of output classes.
        trainable_layers: 0=frozen, -1=all, N=last N groups unfrozen.

    Returns:
        nn.Module in train mode with head replaced and freezing applied.
    """
    model = _load_pretrained(backbone)
    model = _replace_head(model, backbone, num_classes)
    freeze_backbone(model, backbone, trainable_layers)
    return model


def save_torchvision_checkpoint(
    *,
    model: nn.Module,
    arch: str,
    class_names: list[str],
    factor_names: list[str],
    input_size: tuple[int, int],
    best_val_acc: float,
    history: dict[str, Any],
    trainable_layers: int,
    backbone_lr_scale: float,
    path: str | Path,
) -> Path:
    """Save a torchvision model checkpoint in the unified ClassKit .pth format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "arch": arch,
        "class_names": class_names,
        "factor_names": factor_names,
        "input_size": input_size,
        "num_classes": len(class_names),
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
        "history": history,
        "trainable_layers": trainable_layers,
        "backbone_lr_scale": backbone_lr_scale,
    }
    torch.save(ckpt, str(path))
    return path


def load_torchvision_classifier(
    path: str | Path, device: str = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a torchvision classifier from a unified ClassKit .pth checkpoint.

    Returns:
        (model_in_eval_mode_on_device, full_ckpt_dict)
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    num_classes = ckpt["num_classes"]
    # Reconstruct with trainable_layers=-1 (all), then load state — freezing
    # state is irrelevant after loading weights for inference.
    model = build_torchvision_classifier(arch, num_classes, trainable_layers=-1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def export_torchvision_to_onnx(
    model: nn.Module, ckpt: dict[str, Any], onnx_path: str | Path
) -> Path:
    """Export a torchvision classifier to ONNX format.

    Args:
        model: Model in eval mode.
        ckpt: Checkpoint dict (used for input_size).
        onnx_path: Output path for the .onnx file.

    Returns:
        Path to the exported ONNX file.
    """
    onnx_path = Path(onnx_path)
    h, w = ckpt.get("input_size", (224, 224))
    dummy = torch.zeros(1, 3, h, w)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    return onnx_path
```

### Step 2.4 — Run tests

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py -k "torchvision or classifier or layer_groups or freeze or checkpoint or roundtrip or onnx" -v 2>&1 | tail -25
```
Expected: all new tests PASS. Note: the first run may download ImageNet pretrained weights (~100 MB per backbone); subsequent runs use the local cache.

### Step 2.5 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/training/torchvision_model.py tests/test_classkit_extended_training.py
git commit -m "$(cat <<'EOF'
feat(classkit): add torchvision_model module with classifier factory, freezing, ONNX export

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — `_train_custom_classify()` in `runner.py` + dispatch

**Files:**
- Modify: `src/multi_tracker/training/runner.py`
- Test: `tests/test_classkit_extended_training.py` (append)

### Step 3.1 — Write failing tests

- [ ] Append to `tests/test_classkit_extended_training.py`:

```python
from unittest.mock import patch, MagicMock


def test_runner_flat_tiny_alias_dispatches_to_custom_classify():
    """flat_tiny role should call _train_custom_classify with backbone='tinyclassifier'."""
    from multi_tracker.training.contracts import TrainingRole, TrainingRunSpec, CustomCNNParams
    from multi_tracker.training import runner

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="tinyclassifier"),
    )
    with patch.object(runner, "_train_custom_classify", return_value={"success": True}) as mock_fn:
        runner.run_training(spec, "/tmp/run")
        mock_fn.assert_called_once()
        call_spec = mock_fn.call_args[0][0]
        assert call_spec.custom_params.backbone == "tinyclassifier"


def test_runner_flat_custom_dispatches_to_custom_classify():
    """flat_custom role should call _train_custom_classify."""
    from multi_tracker.training.contracts import TrainingRole, TrainingRunSpec, CustomCNNParams
    from multi_tracker.training import runner

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_CUSTOM,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="",
        hyperparams=None,
        custom_params=CustomCNNParams(backbone="resnet18"),
    )
    with patch.object(runner, "_train_custom_classify", return_value={"success": True}) as mock_fn:
        runner.run_training(spec, "/tmp/run")
        mock_fn.assert_called_once()
```

### Step 3.2 — Run tests to verify they fail

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_runner_flat_tiny_alias_dispatches_to_custom_classify tests/test_classkit_extended_training.py::test_runner_flat_custom_dispatches_to_custom_classify -v 2>&1 | head -20
```
Expected: FAIL — `_train_custom_classify` does not exist yet, or dispatch does not route to it.

### Step 3.3 — Add `_train_custom_classify()` to `runner.py`

- [ ] Open `src/multi_tracker/training/runner.py`. Find `_train_tiny_classify` at line 187. After the entire `_train_tiny_classify` function body (before `build_ultralytics_command`), add the new function:

```python
def _train_custom_classify(
    spec: "TrainingRunSpec",
    run_dir: Path,
    log_cb=None,
    progress_cb=None,
    should_cancel=None,
) -> dict:
    """Train a Custom CNN classifier (TinyClassifier or torchvision backbone).

    If backbone == 'tinyclassifier', delegates entirely to _train_tiny_classify().
    Otherwise trains a pretrained torchvision model with discriminative LR.
    """
    from .contracts import CustomCNNParams
    from .torchvision_model import (
        build_torchvision_classifier,
        save_torchvision_checkpoint,
        export_torchvision_to_onnx,
    )

    params: CustomCNNParams = spec.custom_params or CustomCNNParams()

    if params.backbone == "tinyclassifier":
        return _train_tiny_classify(spec, run_dir, log_cb, progress_cb, should_cancel)

    # --- Torchvision training path ---
    import json
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets

    run_dir = Path(run_dir)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    # Build dataset
    dataset_dir = Path(spec.derived_dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    sz = params.input_size

    train_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    class_names = train_ds.classes

    train_loader = DataLoader(
        train_ds, batch_size=params.batch, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=params.batch, shuffle=False, num_workers=2, pin_memory=True
    )

    # _pick_torch_device is defined in runner.py and handles "auto", MPS, CUDA fallback
    device = _pick_torch_device(spec.device)
    model = build_torchvision_classifier(params.backbone, len(class_names), params.trainable_layers)
    model.to(device)

    # Discriminative LR: backbone params at reduced LR, head at full LR
    head_params, backbone_params = [], []
    head_names = {"classifier", "fc", "heads"}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(name.startswith(h) for h in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = [{"params": head_params, "lr": params.lr}]
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": params.lr * params.backbone_lr_scale}
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=params.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=params.label_smoothing)

    best_val_acc = 0.0
    patience_count = 0
    history: dict = {"train_loss": [], "val_acc": []}
    best_ckpt_path = (
        weights_dir / f"classkit_custom_{params.backbone}_{len(class_names)}cls.pth"
    )

    for epoch in range(params.epochs):
        if should_cancel and should_cancel():
            _log("Training canceled.")
            break

        # Training
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
        val_acc = correct / max(total, 1)
        history["val_acc"].append(val_acc)

        _log(f"Epoch {epoch + 1}/{params.epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        if progress_cb:
            progress_cb(epoch + 1, params.epochs)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            save_torchvision_checkpoint(
                model=model,
                backbone=params.backbone,
                class_names=class_names,
                factor_names=[],
                input_size=(sz, sz),
                best_val_acc=best_val_acc,
                history=history,
                trainable_layers=params.trainable_layers,
                backbone_lr_scale=params.backbone_lr_scale,
                path=best_ckpt_path,
            )
        else:
            patience_count += 1
            if patience_count >= params.patience:
                _log(f"Early stopping at epoch {epoch + 1}.")
                break

    # ONNX export
    from .torchvision_model import load_torchvision_classifier
    best_model, best_ckpt = load_torchvision_classifier(str(best_ckpt_path), device="cpu")
    onnx_path = best_ckpt_path.with_suffix(".onnx")
    export_torchvision_to_onnx(best_model, best_ckpt, onnx_path)

    # Metrics
    metrics_path = run_dir / "custom_metrics.json"
    metrics_path.write_text(
        json.dumps({"best_val_acc": best_val_acc, "history": history}, indent=2)
    )
    _log(f"Training complete. Best val acc: {best_val_acc:.4f}")

    return {
        "success": True,
        "artifact_path": str(best_ckpt_path),
        "onnx_path": str(onnx_path),
        "metrics_path": str(metrics_path),
        "best_val_acc": best_val_acc,
        "command": ["custom_classify_inprocess"],
        "task": "custom_classify",
    }
```

- [ ] Update `run_training()` at line 485. Replace the existing dispatch block:

```python
    if spec.role in (
        TrainingRole.CLASSIFY_FLAT_TINY,
        TrainingRole.CLASSIFY_MULTIHEAD_TINY,
    ):
        return _train_tiny_classify(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
```

With the new unified dispatch block that also imports the two new roles:

```python
    if spec.role in (
        TrainingRole.CLASSIFY_FLAT_CUSTOM,
        TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
        TrainingRole.CLASSIFY_FLAT_TINY,
        TrainingRole.CLASSIFY_MULTIHEAD_TINY,
    ):
        # Ensure custom_params is populated; alias roles (flat_tiny, multihead_tiny)
        # inject tinyclassifier default so _train_custom_classify can dispatch correctly.
        from .contracts import CustomCNNParams
        import dataclasses as _dc
        if spec.custom_params is None:
            spec = _dc.replace(spec, custom_params=CustomCNNParams(backbone="tinyclassifier"))
        return _train_custom_classify(
            spec,
            run_dir,
            log_cb=log_cb,
            progress_cb=progress_cb,
            should_cancel=should_cancel,
        )
```

Also add the two new roles to the import at the top of `run_training()` — the existing import is `from .contracts import TrainingRole, TrainingRunSpec`. Verify this import is present; no change needed since `TrainingRole` is already imported and the new values are part of it.

### Step 3.4 — Run tests

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_runner_flat_tiny_alias_dispatches_to_custom_classify tests/test_classkit_extended_training.py::test_runner_flat_custom_dispatches_to_custom_classify -v
```
Expected output:
```
PASSED tests/test_classkit_extended_training.py::test_runner_flat_tiny_alias_dispatches_to_custom_classify
PASSED tests/test_classkit_extended_training.py::test_runner_flat_custom_dispatches_to_custom_classify
2 passed
```

### Step 3.5 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/training/runner.py tests/test_classkit_extended_training.py
git commit -m "$(cat <<'EOF'
feat(classkit): add _train_custom_classify() and update run_training() dispatch

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — `model_publish.py` + `classkit/presets.py`

**Files:**
- Modify: `src/multi_tracker/training/model_publish.py`
- Modify: `src/multi_tracker/classkit/presets.py`
- Test: `tests/test_classkit_extended_training.py` (append)

### Step 4.1 — Write failing tests

- [ ] Append to `tests/test_classkit_extended_training.py`:

```python
def test_all_presets_include_flat_custom():
    """Every single-factor preset function must include 'flat_custom' in training_modes."""
    from multi_tracker.classkit.presets import apriltag_preset, head_tail_preset, age_preset

    assert "flat_custom" in apriltag_preset("tag36h11", 9).training_modes
    assert "flat_custom" in head_tail_preset().training_modes
    assert "flat_custom" in age_preset().training_modes


def test_color_tag_preset_includes_multihead_custom():
    """Multi-factor preset must include both flat_custom and multihead_custom."""
    from multi_tracker.classkit.presets import color_tag_preset
    scheme = color_tag_preset(n_factors=2, colors=["red", "blue"])
    assert "flat_custom" in scheme.training_modes
    assert "multihead_custom" in scheme.training_modes
```

### Step 4.2 — Run tests to verify they fail

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_all_presets_include_flat_custom tests/test_classkit_extended_training.py::test_color_tag_preset_includes_multihead_custom -v 2>&1 | head -15
```
Expected: `AssertionError` — presets don't include `flat_custom` yet.

### Step 4.3 — Update `classkit/presets.py`

- [ ] Open `src/multi_tracker/classkit/presets.py`. Update each preset function's `training_modes` list:

In `head_tail_preset()` (line 19), change:
```python
        training_modes=["flat_tiny", "flat_yolo"],
```
to:
```python
        training_modes=["flat_tiny", "flat_yolo", "flat_custom"],
```

In `color_tag_preset()` (lines 40–42), change:
```python
    if n_factors == 1:
        modes = ["flat_tiny", "flat_yolo"]
    else:
        modes = ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]
```
to:
```python
    if n_factors == 1:
        modes = ["flat_tiny", "flat_yolo", "flat_custom"]
    else:
        modes = ["flat_tiny", "flat_yolo", "flat_custom", "multihead_tiny", "multihead_yolo", "multihead_custom"]
```

In `age_preset()` (line 59), change:
```python
        training_modes=["flat_tiny", "flat_yolo"],
```
to:
```python
        training_modes=["flat_tiny", "flat_yolo", "flat_custom"],
```

In `apriltag_preset()` (line 75), change:
```python
        training_modes=["flat_tiny", "flat_yolo"],
```
to:
```python
        training_modes=["flat_tiny", "flat_yolo", "flat_custom"],
```

### Step 4.4 — Update `model_publish.py`

- [ ] Open `src/multi_tracker/training/model_publish.py`. In `_repo_dir_for_role()` (line 24), the function uses an `if/elif` chain. Add two new `elif` branches before the `else` clause:

```python
    elif role == TrainingRole.CLASSIFY_FLAT_CUSTOM:
        out = root / "custom-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM:
        out = root / "custom-classify" / "multihead" / scheme_name
```

- [ ] In `_task_usage_for_role()` (line 46), the function uses `if`/`if`/`if` returns. Add two new cases before the final `raise`:

```python
    if role in (TrainingRole.CLASSIFY_FLAT_CUSTOM, TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM):
        return "classify", "classify_custom"
```

### Step 4.5 — Run tests

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py::test_all_presets_include_flat_custom tests/test_classkit_extended_training.py::test_color_tag_preset_includes_multihead_custom -v
```
Expected output:
```
PASSED tests/test_classkit_extended_training.py::test_all_presets_include_flat_custom
PASSED tests/test_classkit_extended_training.py::test_color_tag_preset_includes_multihead_custom
2 passed
```

### Step 4.6 — Run full test suite so far

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py -v 2>&1 | tail -10
```
Expected: all tests PASSED.

### Step 4.7 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/training/model_publish.py src/multi_tracker/classkit/presets.py tests/test_classkit_extended_training.py
git commit -m "$(cat <<'EOF'
feat(classkit): add flat_custom/multihead_custom to presets and model_publish

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — `TorchvisionInferenceWorker` + checkpoint dispatch fix

**Files:**
- Modify: `src/multi_tracker/classkit/jobs/task_workers.py`
- Modify: `src/multi_tracker/classkit/gui/mainwindow.py`

No Qt unit tests for the worker (consistent with the existing `TinyCNNInferenceWorker` pattern, which also has no Qt unit tests). Verification via import check only.

### Step 5.1 — Add `TorchvisionInferenceWorker` to `task_workers.py`

- [ ] Open `src/multi_tracker/classkit/jobs/task_workers.py`. Find `class TinyCNNInferenceWorker` at line 1188. After the entire `TinyCNNInferenceWorker` class body, append the new class:

```python
class TorchvisionInferenceWorker(QRunnable):
    """Run torchvision Custom CNN classification inference on all project images.

    Supports PyTorch (cpu/mps/cuda/rocm) and ONNX runtimes via
    ``compute_runtime``. Output contract: same as TinyCNNInferenceWorker —
    emits ``{"probs": ndarray(N, C), "class_names": list}`` via success signal.
    """

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        class_names: List[str],
        input_size: int = 224,
        compute_runtime: str = "cpu",
        batch_size: int = 64,
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.class_names = list(class_names)
        self.input_size = input_size
        self.compute_runtime = str(compute_runtime or "cpu")
        self.batch_size = batch_size
        self.signals = TaskSignals()

    @staticmethod
    def _torch_device(rt: str) -> str:
        """Map canonical runtime to PyTorch device string."""
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"
        return "cpu"

    @Slot()
    def run(self) -> None:
        import numpy as _np
        import torch
        from torchvision import transforms

        try:
            self.signals.started.emit()
            rt = self.compute_runtime
            sz = self.input_size
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            tf = transforms.Compose([
                transforms.Resize((sz, sz)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

            use_onnx = rt in ("onnx_cpu", "onnx_cuda", "onnx_rocm", "tensorrt")

            if use_onnx:
                import onnxruntime as ort
                onnx_path = self.model_path.with_suffix(".onnx")
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "cuda" in rt
                    else ["CPUExecutionProvider"]
                )
                sess = ort.InferenceSession(str(onnx_path), providers=providers)
                input_name = sess.get_inputs()[0].name

                def _infer(batch_np):
                    return sess.run(None, {input_name: batch_np})[0]
            else:
                from ...training.torchvision_model import load_torchvision_classifier
                device = self._torch_device(rt)
                model, _ = load_torchvision_classifier(str(self.model_path), device=device)

                def _infer(batch_np):
                    t = torch.tensor(batch_np).to(device)
                    with torch.no_grad():
                        return model(t).cpu().numpy()

            from PIL import Image
            all_probs = []
            total = len(self.image_paths)
            for i in range(0, total, self.batch_size):
                batch_paths = self.image_paths[i: i + self.batch_size]
                batch_tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(str(p)).convert("RGB")
                        batch_tensors.append(tf(img).numpy())
                    except Exception:
                        batch_tensors.append(
                            _np.zeros((3, sz, sz), dtype=_np.float32)
                        )
                batch_np = _np.stack(batch_tensors).astype(_np.float32)
                logits = _infer(batch_np)
                # Softmax
                exp = _np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                all_probs.append(probs)
                pct = int(min(i + self.batch_size, total) * 100 / total)
                self.signals.progress.emit(
                    pct, f"Inferring {min(i + self.batch_size, total)}/{total}"
                )

            all_probs_np = (
                _np.concatenate(all_probs, axis=0)
                if all_probs
                else _np.zeros((0, len(self.class_names)))
            )
            self.signals.success.emit(
                {"probs": all_probs_np, "class_names": self.class_names}
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
```

### Step 5.2 — Update checkpoint dispatch in `mainwindow.py`

- [ ] Open `src/multi_tracker/classkit/gui/mainwindow.py`. Find `_load_checkpoint_from_path` at line 4168. The current detection heuristic at line 4174 is:

```python
            is_tiny_cnn = path.suffix.lower() == ".pth" or (
                isinstance(ckpt, dict) and "model_state_dict" in ckpt
            )
```

And the branch at line 4219:
```python
            elif is_tiny_cnn:
                # ── Tiny CNN format (.pth / model_state_dict key) ─────
```

Replace the `elif is_tiny_cnn:` branch to first check `arch`:

```python
            elif is_tiny_cnn:
                arch = ckpt.get("arch", "tinyclassifier") if isinstance(ckpt, dict) else "tinyclassifier"
                if arch != "tinyclassifier":
                    # ── Torchvision Custom CNN format ──────────────────
                    ckpt_names = ckpt.get("class_names")
                    input_size = ckpt.get("input_size", (224, 224))
                    sz = input_size[0] if isinstance(input_size, (list, tuple)) else int(input_size)
                    resolved = ckpt_names or list(self.classes)
                    self._active_model_mode = "tiny"
                    self.status.showMessage(f"Loading Custom CNN ({arch}): {path.name}...")
                    self._run_torchvision_inference(
                        path,
                        class_names=resolved,
                        input_size=sz,
                        on_success=lambda r: (
                            self._evaluate_model_on_labeled(),
                            QTimer.singleShot(100, self._replot_umap_model_space),
                            QMessageBox.information(
                                self,
                                "Custom CNN Loaded",
                                f"Loaded: {path.name}\n"
                                f"Inference on {len(self.image_paths):,} images complete.\n"
                                "Metrics tab updated. Model UMAP computing...",
                            ),
                        ),
                    )
                else:
                    # ── Tiny CNN format (arch == 'tinyclassifier' or arch absent) ─
                    ckpt_names = ckpt.get("class_names")
                    db_names = None
                    if self.db_path:
                        try:
                            from ..store.db import ClassKitDB as _CKDb

                            for _entry in _CKDb(self.db_path).list_model_caches():
                                if str(path) in _entry.get("artifact_paths", []):
                                    db_names = _entry.get("class_names")
                                    break
                        except Exception:
                            pass
                    resolved = ckpt_names or db_names or list(self.classes)
                    self._active_model_mode = "tiny"
                    self.status.showMessage(f"Loading tiny CNN: {path.name}...")
                    self._run_tiny_inference(
                        path,
                        class_names=resolved,
                        on_success=lambda r: (
                            self._evaluate_model_on_labeled(),
                            QTimer.singleShot(100, self._replot_umap_model_space),
                            QMessageBox.information(
                                self,
                                "Tiny CNN Loaded",
                                f"Loaded: {path.name}\n"
                                f"Inference on {len(self.image_paths):,} images complete.\n"
                                "Metrics tab updated. Model UMAP computing...",
                            ),
                        ),
                    )
```

- [ ] Add the `_run_torchvision_inference` helper method to `mainwindow.py`. Locate the existing `_run_tiny_inference` method and add the new method immediately after it (mirror the same pattern):

```python
    def _run_torchvision_inference(self, model_path, class_names, input_size=224, on_success=None):
        """Launch TorchvisionInferenceWorker and wire signals to the standard post-inference path."""
        from ..jobs.task_workers import TorchvisionInferenceWorker

        rt = (self._last_training_settings or {}).get("compute_runtime", "cpu")
        worker = TorchvisionInferenceWorker(
            model_path=model_path,
            image_paths=self.image_paths,
            class_names=class_names,
            input_size=input_size,
            compute_runtime=rt,
        )
        worker.signals.success.connect(
            lambda result: self._on_inference_success(result, on_success=on_success)
        )
        worker.signals.error.connect(
            lambda msg: self.status.showMessage(f"Inference error: {msg}")
        )
        worker.signals.progress.connect(
            lambda pct, txt: self.status.showMessage(f"{txt} ({pct}%)")
        )
        from PySide6.QtCore import QThreadPool
        QThreadPool.globalInstance().start(worker)
```

- [ ] Update `train_classifier()` at line 3621. Find the `role_map` dict at line 3699:

```python
            role_map = {
                "flat_tiny": TrainingRole.CLASSIFY_FLAT_TINY,
                "flat_yolo": TrainingRole.CLASSIFY_FLAT_YOLO,
                "multihead_tiny": TrainingRole.CLASSIFY_MULTIHEAD_TINY,
                "multihead_yolo": TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
            }
```

Add two new entries:

```python
            role_map = {
                "flat_tiny": TrainingRole.CLASSIFY_FLAT_TINY,
                "flat_yolo": TrainingRole.CLASSIFY_FLAT_YOLO,
                "flat_custom": TrainingRole.CLASSIFY_FLAT_CUSTOM,
                "multihead_tiny": TrainingRole.CLASSIFY_MULTIHEAD_TINY,
                "multihead_yolo": TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
                "multihead_custom": TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
            }
```

- [ ] In the `_do_train()` closure inside `train_classifier()`, add the import for `CustomCNNParams` alongside the existing imports, then populate `custom_params` on the spec when the mode is `flat_custom` or `multihead_custom`. Add after the `role = role_map.get(...)` line:

```python
            from ...training.contracts import CustomCNNParams
            import dataclasses as _dc
            if mode in ("flat_custom", "multihead_custom"):
                spec = _dc.replace(spec, custom_params=CustomCNNParams(
                    backbone=settings.get("custom_backbone", "tinyclassifier"),
                    trainable_layers=settings.get("custom_trainable_layers", 0),
                    backbone_lr_scale=settings.get("custom_backbone_lr_scale", 0.1),
                    input_size=settings.get("custom_input_size", 224),
                    epochs=settings.get("epochs", 50),
                    batch=settings.get("batch", 32),
                    lr=settings.get("lr", 1e-3),
                    patience=settings.get("patience", 10),
                    weight_decay=1e-2,
                    label_smoothing=settings.get("tiny_label_smoothing", 0.0),
                    class_rebalance_mode=settings.get("tiny_rebalance_mode", "none"),
                    class_rebalance_power=settings.get("tiny_rebalance_power", 1.0),
                ))
```

Also add `TrainingRole.CLASSIFY_FLAT_CUSTOM` and `TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM` to the existing import of contracts inside `_do_train()` at line 3679.

### Step 5.3 — Fix post-training inference dispatch in `mainwindow.py`

- [ ] In `mainwindow.py`, find the `_on_success` callback inside `train_classifier()`. It currently contains an `else` branch that calls `_run_tiny_inference` for any non-YOLO artifact (approximately lines 3878–3888). This branch must be updated to distinguish torchvision checkpoints from TinyClassifier checkpoints by reading the `arch` field — otherwise a torchvision model trained via `flat_custom` would silently be inferred through the wrong worker.

Find the block that looks like:
```python
                else:
                    # Tiny CNN: run inference immediately
                    self._run_tiny_inference(
                        artifact_path,
                        class_names=...
                        ...
                    )
```

Replace it with:
```python
                else:
                    # Custom CNN: dispatch to correct inference worker based on arch field
                    import torch as _torch
                    _ckpt = _torch.load(str(artifact_path), map_location="cpu", weights_only=False)
                    _arch = _ckpt.get("arch", "tinyclassifier") if isinstance(_ckpt, dict) else "tinyclassifier"
                    _class_names = _ckpt.get("class_names") or list(self.classes)
                    if _arch != "tinyclassifier":
                        _sz = _ckpt.get("input_size", (224, 224))
                        _sz = _sz[0] if isinstance(_sz, (list, tuple)) else int(_sz)
                        self._run_torchvision_inference(
                            artifact_path,
                            class_names=_class_names,
                            input_size=_sz,
                        )
                    else:
                        self._run_tiny_inference(
                            artifact_path,
                            class_names=_class_names,
                        )
```

Note: `_run_torchvision_inference` is added in Step 5.2 — this step depends on that being in place first.

### Step 5.4 — Verify import

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.classkit.jobs.task_workers import TorchvisionInferenceWorker; print('OK')"
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.classkit.app import main; print('import OK')"
```
Expected: both print OK without errors.

### Step 5.5 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/classkit/jobs/task_workers.py src/multi_tracker/classkit/gui/mainwindow.py
git commit -m "$(cat <<'EOF'
feat(classkit): add TorchvisionInferenceWorker and update checkpoint dispatch

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6 — `ClassKitTrainingDialog` Custom CNN tab

**Files:**
- Modify: `src/multi_tracker/classkit/gui/dialogs.py`

No Qt unit tests (consistent with all other ClassKit dialogs). Verification via import check only.

### Step 6.1 — Locate insertion point

- [ ] Run to identify exact line numbers:
```bash
grep -n "def _build_ui\|self\.tabs\|_tiny_tab_idx\|addTab\|def get_settings\|def _on_mode_changed\|def _populate_modes" "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/classkit/gui/dialogs.py" | grep -A2 -B2 "1632\|1883\|2158\|2174\|2257"
```
Key reference lines (verified from reading):
- `_build_ui` starts at line 1632
- Tiny Architecture tab added with `self._tiny_tab_idx = self.tabs.addTab(self.tiny_tab, "Tiny Architecture")` at line 1883
- Space & Augmentations tab added at lines 1885+
- `_populate_modes` at line 2158
- `_on_mode_changed` at line 2174
- `get_settings` at line 2257

### Step 6.2 — Add Custom CNN tab to `_build_ui()`

- [ ] Open `src/multi_tracker/classkit/gui/dialogs.py`. After the line that adds the Tiny Architecture tab (line 1883: `self._tiny_tab_idx = self.tabs.addTab(self.tiny_tab, "Tiny Architecture")`), and before Tab 3 (`# Tab 3: Space & Augmentations` at line 1885), insert the new Custom CNN tab code:

```python
        # Tab 2b: Custom CNN
        self.custom_tab = QWidget()
        custom_form = QFormLayout(self.custom_tab)
        custom_form.setSpacing(8)

        from ...training.torchvision_model import BACKBONE_DISPLAY_NAMES

        self._custom_backbone_combo = QComboBox()
        self._custom_backbone_combo.addItem("TinyClassifier (scratch)", "tinyclassifier")
        self._custom_backbone_combo.insertSeparator(self._custom_backbone_combo.count())
        for key in ("convnext_tiny", "convnext_small", "convnext_base"):
            self._custom_backbone_combo.addItem(
                f"ConvNeXt — {BACKBONE_DISPLAY_NAMES[key]}", key
            )
        self._custom_backbone_combo.insertSeparator(self._custom_backbone_combo.count())
        for key in ("efficientnet_b0", "efficientnet_b3", "efficientnet_b7"):
            self._custom_backbone_combo.addItem(
                f"EfficientNet — {BACKBONE_DISPLAY_NAMES[key]}", key
            )
        self._custom_backbone_combo.insertSeparator(self._custom_backbone_combo.count())
        for key in ("resnet18", "resnet50"):
            self._custom_backbone_combo.addItem(
                f"ResNet — {BACKBONE_DISPLAY_NAMES[key]}", key
            )
        self._custom_backbone_combo.insertSeparator(self._custom_backbone_combo.count())
        self._custom_backbone_combo.addItem(
            f"ViT — {BACKBONE_DISPLAY_NAMES['vit_b_16']}", "vit_b_16"
        )
        custom_form.addRow("Backbone:", self._custom_backbone_combo)

        # TinyClassifier-specific controls (hidden when torchvision backbone selected)
        self._tiny_in_custom_width_spin = QSpinBox()
        self._tiny_in_custom_width_spin.setRange(32, 512)
        self._tiny_in_custom_width_spin.setValue(128)
        custom_form.addRow("Input width (px):", self._tiny_in_custom_width_spin)

        self._tiny_in_custom_height_spin = QSpinBox()
        self._tiny_in_custom_height_spin.setRange(32, 512)
        self._tiny_in_custom_height_spin.setValue(64)
        custom_form.addRow("Input height (px):", self._tiny_in_custom_height_spin)

        self._tiny_in_custom_layers_spin = QSpinBox()
        self._tiny_in_custom_layers_spin.setRange(0, 4)
        self._tiny_in_custom_layers_spin.setValue(1)
        custom_form.addRow("Hidden layers:", self._tiny_in_custom_layers_spin)

        self._tiny_in_custom_dim_spin = QSpinBox()
        self._tiny_in_custom_dim_spin.setRange(16, 512)
        self._tiny_in_custom_dim_spin.setValue(64)
        custom_form.addRow("Hidden dim:", self._tiny_in_custom_dim_spin)

        self._tiny_in_custom_dropout_spin = QDoubleSpinBox()
        self._tiny_in_custom_dropout_spin.setRange(0.0, 0.9)
        self._tiny_in_custom_dropout_spin.setSingleStep(0.05)
        self._tiny_in_custom_dropout_spin.setValue(0.2)
        custom_form.addRow("Dropout:", self._tiny_in_custom_dropout_spin)

        # Torchvision-specific controls (hidden when TinyClassifier selected)
        self._custom_trainable_layers_label = QLabel("Trainable layers (0=frozen, -1=all):")
        self._custom_trainable_layers_spin = QSpinBox()
        self._custom_trainable_layers_spin.setRange(-1, 8)
        self._custom_trainable_layers_spin.setValue(0)
        self._custom_trainable_layers_spin.setToolTip(
            "0=frozen backbone, -1=all layers, N=last N layer groups unfrozen"
        )
        custom_form.addRow(self._custom_trainable_layers_label, self._custom_trainable_layers_spin)

        self._custom_backbone_lr_label = QLabel("Backbone LR scale:")
        self._custom_backbone_lr_spin = QDoubleSpinBox()
        self._custom_backbone_lr_spin.setRange(0.001, 1.0)
        self._custom_backbone_lr_spin.setSingleStep(0.01)
        self._custom_backbone_lr_spin.setDecimals(3)
        self._custom_backbone_lr_spin.setValue(0.1)
        self._custom_backbone_lr_spin.setToolTip(
            "LR multiplier applied to unfrozen backbone layers (head LR is full LR)."
        )
        custom_form.addRow(self._custom_backbone_lr_label, self._custom_backbone_lr_spin)

        self._custom_input_size_label = QLabel("Input size (px, square):")
        self._custom_input_size_spin = QSpinBox()
        self._custom_input_size_spin.setRange(32, 512)
        self._custom_input_size_spin.setSingleStep(32)
        self._custom_input_size_spin.setValue(224)
        self._custom_input_size_spin.setToolTip(
            "Resize crops to this square size before passing to torchvision backbone."
        )
        custom_form.addRow(self._custom_input_size_label, self._custom_input_size_spin)

        # Common controls (shared between TinyClassifier and torchvision paths)
        self._custom_epochs_spin = QSpinBox()
        self._custom_epochs_spin.setRange(1, 500)
        self._custom_epochs_spin.setValue(50)
        custom_form.addRow("Epochs:", self._custom_epochs_spin)

        self._custom_batch_spin = QSpinBox()
        self._custom_batch_spin.setRange(1, 256)
        self._custom_batch_spin.setValue(32)
        custom_form.addRow("Batch size:", self._custom_batch_spin)

        self._custom_lr_spin = QDoubleSpinBox()
        self._custom_lr_spin.setRange(1e-5, 0.1)
        self._custom_lr_spin.setSingleStep(0.0001)
        self._custom_lr_spin.setDecimals(5)
        self._custom_lr_spin.setValue(1e-3)
        custom_form.addRow("Learning rate:", self._custom_lr_spin)

        self._custom_patience_spin = QSpinBox()
        self._custom_patience_spin.setRange(1, 100)
        self._custom_patience_spin.setValue(10)
        custom_form.addRow("Patience:", self._custom_patience_spin)

        self._custom_rebalance_combo = QComboBox()
        self._custom_rebalance_combo.addItems(
            ["none", "weighted_loss", "weighted_sampler", "both"]
        )
        custom_form.addRow("Class rebalance:", self._custom_rebalance_combo)

        self._custom_label_smoothing_spin = QDoubleSpinBox()
        self._custom_label_smoothing_spin.setRange(0.0, 0.5)
        self._custom_label_smoothing_spin.setSingleStep(0.01)
        self._custom_label_smoothing_spin.setValue(0.0)
        custom_form.addRow("Label smoothing:", self._custom_label_smoothing_spin)

        self._custom_tab_idx = self.tabs.addTab(self.custom_tab, "Custom CNN")

        # Connect backbone change and trainable_layers change to show/hide conditional controls
        self._custom_backbone_combo.currentIndexChanged.connect(self._on_custom_backbone_changed)
        self._custom_trainable_layers_spin.valueChanged.connect(self._on_custom_backbone_changed)
        self._on_custom_backbone_changed()  # initialize visibility on dialog open
```

### Step 6.3 — Add `_on_custom_backbone_changed` method

- [ ] Add the new method to `ClassKitTrainingDialog`. Place it immediately after `_on_mode_changed` (line 2174):

```python
    def _on_custom_backbone_changed(self) -> None:
        """Show/hide controls based on selected backbone."""
        backbone = self._custom_backbone_combo.currentData()
        is_tiny = backbone == "tinyclassifier"

        # TinyClassifier-specific controls
        for w in (
            self._tiny_in_custom_width_spin,
            self._tiny_in_custom_height_spin,
            self._tiny_in_custom_layers_spin,
            self._tiny_in_custom_dim_spin,
            self._tiny_in_custom_dropout_spin,
        ):
            w.setVisible(is_tiny)

        # Torchvision-specific controls (trainable_layers, input_size always shown)
        for w in (
            self._custom_trainable_layers_spin,
            self._custom_trainable_layers_label,
            self._custom_input_size_spin,
            self._custom_input_size_label,
        ):
            w.setVisible(not is_tiny)

        # Backbone LR scale only relevant when trainable_layers != 0
        show_lr = not is_tiny and self._custom_trainable_layers_spin.value() != 0
        self._custom_backbone_lr_spin.setVisible(show_lr)
        self._custom_backbone_lr_label.setVisible(show_lr)
```

### Step 6.4 — Update `_populate_modes()` and `_on_mode_changed()`

- [ ] In `_populate_modes()` at line 2158, add the two new mode labels:

```python
    def _populate_modes(self):
        self.mode_combo.clear()
        if self._scheme is None:
            self.mode_combo.addItem("Flat - Tiny CNN", "flat_tiny")
            self.mode_combo.addItem("Flat - YOLO-classify", "flat_yolo")
            self.mode_combo.addItem("Flat - Custom CNN", "flat_custom")
            return
        labels = {
            "flat_tiny": "Flat - Tiny CNN",
            "flat_yolo": "Flat - YOLO-classify",
            "flat_custom": "Flat - Custom CNN",
            "multihead_tiny": "Multi-head - Tiny CNN (one model per factor)",
            "multihead_yolo": "Multi-head - YOLO-classify (one model per factor)",
            "multihead_custom": "Multi-head - Custom CNN (one model per factor)",
        }
        for key in ["flat_tiny", "flat_yolo", "flat_custom", "multihead_tiny", "multihead_yolo", "multihead_custom"]:
            if key in self._scheme.training_modes:
                self.mode_combo.addItem(labels[key], key)
```

- [ ] In `_on_mode_changed()` at line 2174, add show/hide logic for the Custom CNN tab and extend the mode description dict:

In the existing body, after the `is_tiny` line, add:
```python
        is_custom = "custom" in mode
```

After the block that hides the Tiny Architecture tab:
```python
        # Show/hide Custom CNN tab
        if hasattr(self, "_custom_tab_idx"):
            self.tabs.setTabVisible(self._custom_tab_idx, is_custom)
            if not is_custom and self.tabs.currentIndex() == self._custom_tab_idx:
                self.tabs.setCurrentIndex(0)
```

Add two entries to the `_desc` dict:
```python
            "flat_custom": (
                "Custom CNN — TinyClassifier or pretrained torchvision backbone "
                "(ConvNeXt, EfficientNet, ResNet, ViT). Configurable layer freezing "
                "for linear-probe or full fine-tuning."
            ),
            "multihead_custom": (
                "Multi-head Custom CNN — one backbone per factor, with configurable "
                "layer freezing. GPU recommended for torchvision backbones."
            ),
```

### Step 6.5 — Update `get_settings()` to return Custom CNN params

- [ ] In `get_settings()` at line 2257, append the custom CNN fields to the returned dict. Find the `return {` block and add before the closing `}`:

```python
            # Custom CNN tab
            "custom_backbone": self._custom_backbone_combo.currentData(),
            "custom_trainable_layers": self._custom_trainable_layers_spin.value(),
            "custom_backbone_lr_scale": self._custom_backbone_lr_spin.value(),
            "custom_input_size": self._custom_input_size_spin.value(),
```

Also override the common hyperparams when mode is `flat_custom` or `multihead_custom`. Add this block after the `return {` dict is fully assembled — or restructure to compute them before the return:

In the function body, before `return {`, add:
```python
        _mode = self.mode_combo.currentData() or ""
        if _mode in ("flat_custom", "multihead_custom"):
            _custom_epochs = self._custom_epochs_spin.value()
            _custom_batch = self._custom_batch_spin.value()
            _custom_lr = self._custom_lr_spin.value()
            _custom_patience = self._custom_patience_spin.value()
            _custom_rebalance = self._custom_rebalance_combo.currentText()
            _custom_label_smooth = self._custom_label_smoothing_spin.value()
        else:
            _custom_epochs = self.epochs_spin.value()
            _custom_batch = self.batch_spin.value()
            _custom_lr = self.lr_spin.value()
            _custom_patience = self.patience_spin.value()
            _custom_rebalance = self.tiny_rebalance_combo.currentData()
            _custom_label_smooth = self.tiny_label_smoothing_spin.value()
```

Then in the `return {` dict, replace the existing `"epochs": self.epochs_spin.value()` etc. with the computed values only for custom modes, or add a separate key. The simplest approach: add these keys to the return dict directly, and have `train_classifier` use `settings.get("epochs")` which already exists. The intent is that when mode is custom, the custom tab's values override the General tab values. Use the following approach — add to the return dict:

```python
            "custom_backbone": self._custom_backbone_combo.currentData(),
            "custom_trainable_layers": self._custom_trainable_layers_spin.value(),
            "custom_backbone_lr_scale": self._custom_backbone_lr_spin.value(),
            "custom_input_size": self._custom_input_size_spin.value(),
```

And compute `epochs`/`batch`/`lr`/`patience` in the dict based on mode (replace the existing `"epochs": self.epochs_spin.value()` etc. with conditionals):

```python
            "epochs": (
                self._custom_epochs_spin.value()
                if (self.mode_combo.currentData() or "") in ("flat_custom", "multihead_custom")
                else self.epochs_spin.value()
            ),
            "batch": (
                self._custom_batch_spin.value()
                if (self.mode_combo.currentData() or "") in ("flat_custom", "multihead_custom")
                else self.batch_spin.value()
            ),
            "lr": (
                self._custom_lr_spin.value()
                if (self.mode_combo.currentData() or "") in ("flat_custom", "multihead_custom")
                else self.lr_spin.value()
            ),
            "patience": (
                self._custom_patience_spin.value()
                if (self.mode_combo.currentData() or "") in ("flat_custom", "multihead_custom")
                else self.patience_spin.value()
            ),
```

### Step 6.6 — Verify import

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.classkit.gui.dialogs import ClassKitTrainingDialog; print('OK')"
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.classkit.app import main; print('import OK')"
```
Expected: both print OK without errors.

### Step 6.7 — Run full test suite

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_classkit_extended_training.py tests/test_classkit_apriltag_autolabel.py tests/test_classkit_scheme.py -v 2>&1 | tail -15
```
Expected: all tests PASSED.

### Step 6.8 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/classkit/gui/dialogs.py
git commit -m "$(cat <<'EOF'
feat(classkit): add Custom CNN tab to ClassKitTrainingDialog

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
