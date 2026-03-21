# ClassKit Extended Training — Custom CNN Models

**Date:** 2026-03-21
**Status:** Draft
**Area:** ClassKit / Training

---

## Overview

This specification describes adding a unified "Custom CNN" training mode to ClassKit that encompasses both the existing TinyClassifier and new pretrained torchvision backbones (ConvNeXt, EfficientNet, ResNet, ViT). Users gain control over freezing and unfreezing backbone layers to support linear-probe and full fine-tuning workflows, all through a single UI surface.

Two new training mode strings are introduced: `flat_custom` and `multihead_custom`. The existing `flat_tiny` / `multihead_tiny` mode strings remain valid and are treated as aliases for `flat_custom` / `multihead_custom` with `backbone="tinyclassifier"` to preserve backward compatibility.

---

## Goals

- Allow users to train ConvNeXt, EfficientNet, ResNet, ViT, or TinyClassifier from ClassKit with a single unified UI surface.
- Support frozen backbone (linear probe) and full fine-tuning via a `trainable_layers` parameter.
- Produce a unified `.pth` checkpoint format that MAT can load for any supported backbone.
- Provide ONNX export for all backbone types for runtime flexibility.
- Fit into the existing training mode / preset / worker / dialog pattern without new standalone apps.

## Non-Goals

- No `timm` dependency — torchvision only.
- No new standalone application entry point.
- No changes to YOLO-classify training modes.

---

## Supported Backbones

| Key | Architecture | Pretrained | Approx. Params |
|---|---|---|---|
| `tinyclassifier` | 4-conv scratch CNN | No | 2–5M |
| `convnext_tiny` | ConvNeXt-T | ImageNet | 28M |
| `convnext_small` | ConvNeXt-S | ImageNet | 50M |
| `convnext_base` | ConvNeXt-B | ImageNet | 89M |
| `efficientnet_b0` | EfficientNet-B0 | ImageNet | 5M |
| `efficientnet_b3` | EfficientNet-B3 | ImageNet | 12M |
| `efficientnet_b7` | EfficientNet-B7 | ImageNet | 66M |
| `resnet18` | ResNet-18 | ImageNet | 11M |
| `resnet50` | ResNet-50 | ImageNet | 25M |
| `vit_b_16` | ViT-B/16 | ImageNet | 86M |

---

## Parameter Design

### `trainable_layers` Semantics

The `trainable_layers` integer controls how much of the backbone is unfrozen during training:

- `0` — Frozen backbone; only the new classifier head is trained (linear probe).
- `-1` — All parameters are trainable (full fine-tune).
- `N > 0` — Freeze all backbone parameters, then unfreeze the last `N` layer groups plus the classifier head.

"Layer groups" are architecture-specific:

| Architecture | Layer Group Definition |
|---|---|
| ConvNeXt | Stages (4 stages total) |
| ResNet | Layer blocks (`layer1` – `layer4`) |
| EfficientNet | MBConv block groups |
| ViT | Transformer encoder blocks |
| TinyClassifier | Always fully trainable; `trainable_layers` is ignored |

### `backbone_lr_scale`

When `trainable_layers != 0`, unfrozen backbone parameters use a learning rate of `lr × backbone_lr_scale` (default `0.1`). The classifier head always uses the full `lr`. This discriminative learning rate schedule prevents catastrophic forgetting of pretrained representations.

---

## New Dataclass: `CustomCNNParams`

Location: `training/contracts.py`

```python
@dataclass
class CustomCNNParams:
    backbone: str = "tinyclassifier"
    trainable_layers: int = 0        # 0=frozen, -1=all, N=last N groups
    backbone_lr_scale: float = 0.1   # LR multiplier for unfrozen backbone layers
    input_size: int = 224            # Resize target (square); TinyClassifier uses its own (H, W)
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
    input_width: int = 128           # used only when backbone="tinyclassifier"
    input_height: int = 64           # used only when backbone="tinyclassifier"
```

### `TrainingRunSpec` Field Addition

Also in `training/contracts.py`, `TrainingRunSpec` must gain the following field alongside the existing `tiny_params: TinyHeadTailParams` field:

```python
custom_params: CustomCNNParams | None = None
```

This field is populated by the caller (`ClassKitTrainingWorker`) for any role that uses `_train_custom_classify()`.

### New `TrainingRole` Values

Also in `training/contracts.py`:

```python
CLASSIFY_FLAT_CUSTOM = "classify_flat_custom"
CLASSIFY_MULTIHEAD_CUSTOM = "classify_multihead_custom"
```

`CLASSIFY_FLAT_TINY` and `CLASSIFY_MULTIHEAD_TINY` remain defined and are dispatched as aliases in the runner.

---

## New File: `training/torchvision_model.py`

Pure Python, no Qt. This module is the sole owner of all torchvision model construction, layer freezing, ONNX export, and checkpoint loading logic.

### Public API

| Symbol | Signature | Responsibility |
|---|---|---|
| `TORCHVISION_BACKBONES` | `dict[str, callable]` | Maps backbone key to torchvision factory function |
| `build_torchvision_classifier` | `(backbone, num_classes, trainable_layers) -> nn.Module` | Loads pretrained model, replaces classifier head, applies freezing |
| `get_layer_groups` | `(model, backbone) -> list[nn.Module]` | Returns backbone layer groups for progressive unfreezing |
| `freeze_backbone` | `(model, backbone, trainable_layers)` | Applies freezing logic according to `trainable_layers` semantics |
| `export_torchvision_to_onnx` | `(model, ckpt, onnx_path)` | Exports via `torch.onnx.export()` with opset 17 |
| `load_torchvision_classifier` | `(path, device) -> tuple[nn.Module, dict]` | Loads `.pth` checkpoint, reconstructs model from `arch` field |

### Freezing Logic Details

`build_torchvision_classifier()` calls `freeze_backbone()` immediately after head replacement:

1. Freeze all backbone parameters (`requires_grad = False`).
2. If `trainable_layers == -1`: unfreeze everything.
3. If `trainable_layers > 0`: call `get_layer_groups()` and unfreeze the last `trainable_layers` groups.
4. Always leave the new classifier head unfrozen.

`get_layer_groups()` must handle architectures that expose their stages/layers as named children vs. sequential blocks. It returns a flat list of `nn.Module` objects in backbone-traversal order (shallow to deep), so callers can index from the end.

---

## Unified Checkpoint Format (`.pth`)

All Custom CNN models share the following checkpoint schema. MAT reads this format to reconstruct and run any Custom CNN model.

```python
{
    "arch": str,                    # "tinyclassifier" | "convnext_tiny" | ...
    "class_names": list[str],       # flat list of all output classes
    "factor_names": list[str],      # one entry per factor (multihead only, else [])
    "input_size": tuple,            # (H, W) for tinyclassifier, (224, 224) for torchvision
    "num_classes": int,
    "model_state_dict": dict,
    "best_val_acc": float,
    "history": dict,                # {"train_loss": [...], "val_acc": [...]}
    "trainable_layers": int,        # stored for provenance
    "backbone_lr_scale": float,
}
```

The `arch` field is required so `load_torchvision_classifier()` can reconstruct the correct model architecture before loading state.

---

## `training/runner.py` Changes

### New Function: `_train_custom_classify()`

```
_train_custom_classify(spec, run_dir, log_cb, progress_cb, should_cancel) -> dict
```

Dispatch logic:
- If `spec.custom_params.backbone == "tinyclassifier"`: delegate entirely to the existing `_train_tiny_classify()`. No duplication of the training loop.
- Otherwise: use `build_torchvision_classifier()` with discriminative LR groups (head uses `lr`, unfrozen backbone uses `lr × backbone_lr_scale`), then run the same AdamW + cosine/flat LR schedule + early stopping loop already used by TinyClassifier.

### Discriminative LR Group Construction

When `trainable_layers != 0`, the optimizer receives two parameter groups:

```python
param_groups = [
    {"params": backbone_params, "lr": lr * backbone_lr_scale},
    {"params": head_params,     "lr": lr},
]
```

When `trainable_layers == 0`, only the head group is passed (backbone parameters are frozen and excluded from the optimizer).

### Multihead Architecture

`_train_custom_classify()` handles multihead by being **called once per factor** by the caller (`ClassKitTrainingWorker`), consistent with the existing multihead pattern used by `_train_tiny_classify()`. Each call receives the per-factor dataset directory and produces one model per factor. The runner function itself does not loop over factors internally.

### `run_training()` Dispatch Updates

| Role string | Action |
|---|---|
| `CLASSIFY_FLAT_CUSTOM` | Call `_train_custom_classify()` |
| `CLASSIFY_MULTIHEAD_CUSTOM` | Call `_train_custom_classify()` (called once per factor by the worker) |
| `CLASSIFY_FLAT_TINY` (alias) | Call `_train_custom_classify()` with `backbone="tinyclassifier"` |
| `CLASSIFY_MULTIHEAD_TINY` (alias) | Call `_train_custom_classify()` with `backbone="tinyclassifier"` |

---

## `training/model_publish.py` Changes

Add `_repo_dir_for_role()` entries for the new roles:

| Role | Output directory |
|---|---|
| `CLASSIFY_FLAT_CUSTOM` | `models/custom-classify/{scheme_name}/` |
| `CLASSIFY_MULTIHEAD_CUSTOM` | `models/custom-classify/multihead/{scheme_name}/{factor_name}/` |

`_task_usage_for_role()` also switches on `TrainingRole` and will raise `RuntimeError` for any unhandled role. It must be updated to handle the two new roles:

| Role | Task usage string |
|---|---|
| `CLASSIFY_FLAT_CUSTOM` | `"classify"` |
| `CLASSIFY_MULTIHEAD_CUSTOM` | `"classify"` |

---

## `classkit/presets.py` Changes

All preset functions add `"flat_custom"` and `"multihead_custom"` to their `training_modes` lists. Single-factor presets omit multihead variants, following the existing convention.

Example:

```python
def apriltag_preset() -> ClassKitPreset:
    return ClassKitPreset(
        ...
        training_modes=["flat_tiny", "flat_custom", "flat_yolo", ...],
    )
```

---

## ClassKit GUI Changes

### `classkit/gui/dialogs.py` — `ClassKitTrainingDialog`

A new "Custom CNN" tab is added to the training dialog. This tab is shown when the selected training mode is `flat_custom` or `multihead_custom`; it is hidden for all other modes.

**Controls in the Custom CNN tab:**

A backbone dropdown groups options by family:

- TinyClassifier
  - TinyClassifier (scratch)
- ConvNeXt
  - ConvNeXt-T (`convnext_tiny`)
  - ConvNeXt-S (`convnext_small`)
  - ConvNeXt-B (`convnext_base`)
- EfficientNet
  - EfficientNet-B0 (`efficientnet_b0`)
  - EfficientNet-B3 (`efficientnet_b3`)
  - EfficientNet-B7 (`efficientnet_b7`)
- ResNet
  - ResNet-18 (`resnet18`)
  - ResNet-50 (`resnet50`)
- ViT
  - ViT-B/16 (`vit_b_16`)

**Conditional control visibility:**

When backbone is `tinyclassifier`:
- Show: input width, input height, hidden layers, hidden dim, dropout (existing Tiny controls)
- Hide: trainable layers spinbox, backbone LR scale, torchvision input size

When backbone is a torchvision model:
- Show: trainable layers spinbox (range -1 to 8, default 0, label "0=frozen, -1=all, N=last N groups")
- Show: backbone LR scale spinbox (default 0.1, visible only when trainable_layers ≠ 0)
- Show: input size spinbox (default 224)
- Hide: TinyClassifier-specific controls

**Common controls (all backbones):**
- Epochs, batch size, learning rate, patience, weight decay
- Class rebalance mode (none / weighted_loss / weighted_sampler / both)
- Class rebalance power
- Label smoothing

### `classkit/gui/mainwindow.py` Changes

- Wire `flat_custom` and `multihead_custom` into the `train_classifier()` role map.
- Add inference dispatch for Custom CNN torchvision models: route `.pth` files with a torchvision `arch` field to the new `TorchvisionInferenceWorker`.

**Checkpoint detection heuristic in `_load_checkpoint_from_path`:**

The existing heuristic detects a TinyClassifier checkpoint by the presence of `model_state_dict`. Because torchvision `.pth` checkpoints also contain `model_state_dict`, this causes misrouting. The dispatch must check the `arch` field **first**:

1. If `arch` is present in the checkpoint **and** `arch != "tinyclassifier"` → route to `TorchvisionInferenceWorker`.
2. If `arch` is present **and** `arch == "tinyclassifier"` → route to the existing `TinyCNNInferenceWorker`.
3. If `arch` is **absent** (legacy TinyClassifier checkpoint saved before this field was added) → fall back to the old `model_state_dict` heuristic to select `TinyCNNInferenceWorker`.

This ordering ensures new torchvision checkpoints are never misrouted to the TinyClassifier loader.

---

## `classkit/jobs/task_workers.py` Changes

### New: `TorchvisionInferenceWorker(QRunnable)`

Same signal contract as the existing `TinyCNNInferenceWorker`:

- Signal: `result_ready(dict)` emitting `{"probs": ndarray(N, C), "class_names": list}`
- Signal: `error_occurred(str)`

Behavior:
- For PyTorch runtimes: load `.pth` → reconstruct model via `load_torchvision_classifier()` → run inference.
- For `onnx_*` runtimes: load `.onnx` via `onnxruntime.InferenceSession` → run inference.

The worker applies the same image preprocessing pipeline as the training loader (resize to `input_size`, ImageNet normalize for torchvision backbones).

---

## Testing

### New Test File: `tests/test_classkit_extended_training.py`

All tests are unit tests. Expensive torch operations are mocked where needed to keep the suite fast.

| Test | What it verifies |
|---|---|
| `test_custom_cnn_params_defaults` | `CustomCNNParams` instantiates with correct default field values |
| `test_custom_cnn_params_field_types` | All fields have correct types; no unexpected fields |
| `test_build_torchvision_classifier_convnext` | Output logits shape matches `num_classes` |
| `test_build_torchvision_classifier_efficientnet` | Output logits shape matches `num_classes` |
| `test_build_torchvision_classifier_resnet` | Output logits shape matches `num_classes` |
| `test_build_torchvision_classifier_vit` | Output logits shape matches `num_classes` |
| `test_get_layer_groups_convnext_count` | Returns exactly 4 groups for ConvNeXt |
| `test_get_layer_groups_resnet_count` | Returns exactly 4 groups for ResNet |
| `test_get_layer_groups_efficientnet_count` | Returns correct group count for EfficientNet |
| `test_get_layer_groups_vit_count` | Returns correct group count for ViT |
| `test_freeze_backbone_frozen` | `trainable_layers=0` freezes all backbone params; head remains unfrozen |
| `test_freeze_backbone_all` | `trainable_layers=-1` leaves all params unfrozen |
| `test_freeze_backbone_partial` | `trainable_layers=1` freezes all but last group + head |
| `test_export_torchvision_to_onnx_smoke` | ONNX file is created and loadable (tiny input, mocked where slow) |
| `test_load_torchvision_classifier_roundtrip` | Save then load produces identical logits for a fixed input |
| `test_checkpoint_format_required_keys` | Saved checkpoint contains all keys defined in the schema |
| `test_runner_flat_tiny_alias` | `flat_tiny` role causes `_train_custom_classify` to be called with `backbone="tinyclassifier"` |
| `test_runner_flat_custom_dispatch` | `flat_custom` role dispatches to `_train_custom_classify` with torchvision backbone |
| `test_apriltag_preset_includes_flat_custom` | `apriltag_preset()` includes `"flat_custom"` in `training_modes` |
| `test_all_presets_include_flat_custom` | Every preset function includes `"flat_custom"` in `training_modes` |

---

## Files Changed / Created

| File | Change Type | Summary |
|---|---|---|
| `training/contracts.py` | Modified | Add `CustomCNNParams` dataclass; add `CLASSIFY_FLAT_CUSTOM` and `CLASSIFY_MULTIHEAD_CUSTOM` role constants |
| `training/torchvision_model.py` | New | Model factory, layer-group enumeration, freezing logic, ONNX export, checkpoint load/save |
| `training/runner.py` | Modified | Add `_train_custom_classify()`; update `run_training()` dispatch table with new roles and backward-compat aliases |
| `training/model_publish.py` | Modified | Add `_repo_dir_for_role()` entries for `CLASSIFY_FLAT_CUSTOM` and `CLASSIFY_MULTIHEAD_CUSTOM` |
| `classkit/presets.py` | Modified | Add `flat_custom` / `multihead_custom` to `training_modes` in all preset functions |
| `classkit/gui/dialogs.py` | Modified | Add Custom CNN tab to `ClassKitTrainingDialog`; implement conditional control visibility |
| `classkit/gui/mainwindow.py` | Modified | Wire new modes into `train_classifier()` role map; add inference dispatch for torchvision models |
| `classkit/jobs/task_workers.py` | Modified | Add `TorchvisionInferenceWorker` with same signal contract as `TinyCNNInferenceWorker` |
| `tests/test_classkit_extended_training.py` | New | Unit test file covering all new logic (20 tests) |

---

## Backward Compatibility

- `flat_tiny` and `multihead_tiny` mode strings continue to work. The runner dispatch maps them to `_train_custom_classify` with `backbone="tinyclassifier"`, which in turn delegates to the existing `_train_tiny_classify()` with no behavioral change.
- Existing `.pth` files saved by TinyClassifier training do not need migration. `load_torchvision_classifier()` detects the `arch` field; files lacking this field are handled by the existing TinyClassifier loader path.
- No changes to YOLO-classify modes.

---

## Open Questions

1. Should `vit_b_16` require a minimum input size warning in the UI (ViT expects 224×224 or larger)?
2. Should the backbone LR scale spinbox use a logarithmic step (0.001 / 0.01 / 0.1) rather than a linear one?
3. For multihead training with torchvision backbones, should a single shared backbone be used with multiple heads, or should a separate model be trained per factor? (This spec assumes the existing multihead architecture is extended rather than redesigned.)
