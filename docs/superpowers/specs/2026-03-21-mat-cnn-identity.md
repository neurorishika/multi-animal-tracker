# MAT CNN Identity — ClassKit Model Integration — Design Spec

**Date:** 2026-03-21
**Status:** Approved

---

## Overview

Add a "CNN Classifier" identity method to MAT. Users import a ClassKit-trained model (`.pth` or YOLO `.pt`) via the existing "+ Add New Model" import dialog. MAT reads the checkpoint metadata automatically to display a verification summary. During tracking, a precompute phase runs the classifier on OBB crops and caches predictions. During assignment, confirmed track identities influence the Hungarian assignment cost via match bonus / mismatch penalty — the same mechanism already used for AprilTag identity.

---

## Goals

- Any ClassKit-trained model (TinyClassifier, YOLO-classify, Custom CNN / Torchvision) works out of the box.
- Import flow consistent with existing model import: copy to `models/`, register in `model_registry.json`.
- Never import ONNX directly — lazy-derive from `.pth`/`.pt` at runtime with signature caching.
- Assignment cost bonus/penalty for identity-informed tracking.
- Free-text "classification label" field for user annotation (e.g. "colortag", "apriltag").

---

## Non-Goals

- No separate "Color Tag" / "AprilTag CNN" / "General" identity sub-modes in code — one unified CNN Classifier.
- No real-time per-frame inference during tracking — precompute only.
- No changes to ClassKit training (Spec A handles that).

---

## Model Import Flow

### Dialog Trigger

A new "CNN Identity Classifier" combo box in the MAT identity settings panel with a `"＋ Add New Model…"` sentinel item, following the exact pattern of `_handle_add_new_yolo_model()` in `main_window.py`.

### Handler: `_handle_add_new_cnn_identity_model()`

New method in `gui/main_window.py`:

1. `QFileDialog.getOpenFileName()` — filter: `"ClassKit Model Files (*.pth *.pt);;All Files (*)"`.
   - Start directory: `models/classification/identity/`.
2. Read checkpoint metadata:
   - For `.pth`: load with `torch.load(..., map_location="cpu")`, extract `arch`, `class_names`, `factor_names`, `input_size`, `num_classes`.
   - For YOLO `.pt`: load with `ultralytics.YOLO()`, read `model.names` dict.
3. Show metadata import dialog — pre-filled from checkpoint, user edits:
   - Species (text field).
   - Classification label (text field, e.g. "apriltag", "colortag") — purely cosmetic.
   - Detected info shown read-only: `arch`, `num_classes`, `class_names` preview, `input_size`.
4. Copy file to `models/classification/identity/{timestamp}_{arch}_{species}_{label}.pth`.
5. Register in `model_registry.json`:
   ```json
   {
     "classification/identity/20260321-120000_convnext_tiny_ant_apriltag.pth": {
       "arch": "convnext_tiny",
       "num_classes": 11,
       "class_names": ["tag_0", "...", "no_tag"],
       "factor_names": [],
       "input_size": [224, 224],
       "species": "ant",
       "classification_label": "apriltag",
       "added_at": "2026-03-21T12:00:00",
       "task_family": "classify",
       "usage_role": "cnn_identity"
     }
   }
   ```
6. Refresh combo, select newly imported model.

### Verification Panel

A read-only block shown when a model is selected in the combo. Displays: `arch`, `num_classes`, `class_names` (truncated if >12), `input_size`, `classification_label`. Loaded from the registry entry — no re-reading `.pth` on every selection.

---

## Config Parameters

New keys added to MAT config (`configs/default.json`):

| Key | Type | Default | Notes |
|---|---|---|---|
| `CNN_CLASSIFIER_MODEL_PATH` | path | `""` | Relative path from `models/` root |
| `CNN_CLASSIFIER_CONFIDENCE` | float | `0.5` | Per-frame prediction confidence threshold |
| `CNN_CLASSIFIER_LABEL` | str | `""` | Free-text label (display only, from registry) |
| `CNN_CLASSIFIER_BATCH_SIZE` | int | `64` | Inference batch size |
| `CNN_CLASSIFIER_CROP_PADDING` | float | `0.1` | OBB crop expansion factor |
| `CNN_CLASSIFIER_MATCH_BONUS` | float | `20.0` | Assignment cost reduction on class match (absolute, same as `TAG_MATCH_BONUS`) |
| `CNN_CLASSIFIER_MISMATCH_PENALTY` | float | `50.0` | Assignment cost increase on class conflict (absolute, same as `TAG_MISMATCH_PENALTY`) |
| `CNN_CLASSIFIER_WINDOW` | int | `10` | Sliding window size for majority-vote identity accumulation |

`COLOR_TAG_MODEL_PATH` is kept as a backward-compatibility alias for `CNN_CLASSIFIER_MODEL_PATH`.

---

## ONNX / TensorRT Runtime

ONNX artifacts are never imported directly. They are lazy-derived from `.pth` / `.pt` at first run, cached alongside the source file, and signature-validated — the same pattern as the detection engine:

- `.pth` → `torch.onnx.export()` → `.onnx` cached next to source file.
- `.pt` (YOLO) → `model.export(format="onnx")` → `.onnx`.
- TensorRT: `.onnx` → TRT builder → `.engine`.
- Signature = `SHA1(model_path | runtime | batch_size | input_size)`.

The user controls runtime via MAT's global `compute_runtime` setting. No separate per-classifier runtime setting is needed.

---

## Precompute Phase

Triggered when identity method = "CNN Classifier" and an OBB detection cache exists. Runs before tracking, same lifecycle as AprilTag precompute.

**Backward mode gate:** CNN identity precompute is **skipped in backward mode** (same condition as AprilTag: `not self.backward_mode`). In backward mode the forward-pass CNN identity cache (if present) is reused as-is. The planner must apply this gate explicitly in `worker.py`.

### New File: `core/identity/cnn_identity.py`

Pure Python, no Qt dependency, fully testable in isolation.

#### `CNNIdentityConfig` dataclass

Mirrors the config keys above. Fields:

```python
@dataclass
class CNNIdentityConfig:
    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    crop_padding: float = 0.1
    match_bonus: float = 20.0
    mismatch_penalty: float = 50.0
    window: int = 10
```

#### `ClassPrediction` dataclass

```python
@dataclass
class ClassPrediction:
    class_name: str | None    # None if below confidence threshold
    confidence: float
    det_index: int            # which detection in the frame
```

#### `CNNIdentityBackend`

Wraps model loading and batch inference. Lazy-loads the model on first use.

```python
class CNNIdentityBackend:
    def __init__(self, config: CNNIdentityConfig, model_path: str, compute_runtime: str): ...
    def predict_batch(self, crops: list[np.ndarray]) -> list[ClassPrediction]: ...
    def close(self) -> None: ...
```

- `predict_batch()` returns exactly one `ClassPrediction` per input crop.
- Predictions with softmax score below `config.confidence` return `class_name=None`.
- Runtime is selected from MAT's global `compute_runtime`; ONNX/TRT artifacts are derived lazily via signature-validated caching.

#### `CNNIdentityCache`

Read/write `.npz` cache of per-frame predictions:

- Format: per-frame arrays of `(det_index, class_name, confidence)`.
- API:
  ```python
  class CNNIdentityCache:
      def save(self, frame_idx: int, predictions: list[ClassPrediction]) -> None: ...
      def load(self, frame_idx: int) -> list[ClassPrediction]: ...
      def exists(self) -> bool: ...
  ```

### Precompute Loop in `core/tracking/worker.py`

After OBB cache load, before tracking:

1. Load `CNNIdentityBackend`.
2. For each frame: extract OBB crops (same extraction as pose / AprilTag precompute) → batch → `predict_batch()` → `CNNIdentityCache.save()`.
3. Per-frame progress emitted to GUI via existing signal mechanism.

---

## Identity Assignment During Tracking

### Track State — Identity Accumulation

Each track maintains a sliding window of `CNN_CLASSIFIER_WINDOW` recent confident predictions. After each frame:

- Confident predictions (confidence >= `CNN_CLASSIFIER_CONFIDENCE`) are appended to the window.
- Majority class across the window = track's current identity.
- If the window is empty or there is no majority → identity is unassigned.

### Hungarian Assignment Cost Adjustment

Applied in `core/assigners/hungarian.py` (or equivalent assigner). When both the track has an assigned identity AND the detection has a confident CNN prediction:

- `predicted_class == track_identity` → `cost -= CNN_CLASSIFIER_MATCH_BONUS` (direct subtraction; lower cost = preferred).
- `predicted_class != track_identity` → `cost += CNN_CLASSIFIER_MISMATCH_PENALTY` (direct addition; higher cost = discouraged).
- Either side uncertain (track identity unassigned, or detection confidence below threshold) → no adjustment.

This mirrors the AprilTag bonus/penalty mechanism exactly: absolute cost deltas, no `max_cost` multiplication.

---

## MAT GUI Changes (`gui/main_window.py`)

### Identity Method Dropdown

**"Color Tags (YOLO)" is renamed to "CNN Classifier"** — it is not kept as a separate entry alongside the new option. The existing `color_tags_yolo` method key in `method_map` is updated to `cnn_classifier`. The `line_color_tag_model` UI widget is reused or replaced by the new CNN classifier model combo. Existing configs that store `color_tags_yolo` as the selected identity method are automatically migrated to `cnn_classifier` on load. `COLOR_TAG_MODEL_PATH` is retained as a backward-compatibility alias for `CNN_CLASSIFIER_MODEL_PATH` during config loading (already noted in the Config Parameters section).

The dropdown therefore contains: "None", "AprilTags", "CNN Classifier" — the last entry covering all CNN-based classification workflows regardless of what visual feature was trained on.

- When "CNN Classifier" is selected: show CNN classifier settings panel (model combo, confidence, match bonus, mismatch penalty, window size, crop padding).
- When not selected: panel hidden.

### Settings Panel Layout

| Control | Type | Notes |
|---|---|---|
| Model combo | QComboBox | Existing models from registry + `"＋ Add New Model…"` sentinel |
| Verification block | Read-only labels | Arch, num_classes, class names preview (truncated if >12), input_size |
| Confidence threshold | QDoubleSpinBox | Range 0.0–1.0 |
| Match bonus | QDoubleSpinBox | Range 0.0–200.0, absolute cost units |
| Mismatch penalty | QDoubleSpinBox | Range 0.0–200.0, absolute cost units |
| Window size | QSpinBox | Range 1–100 |
| Crop padding | QDoubleSpinBox | Range 0.0–1.0 |

---

## Data Flow

```
User selects "CNN Classifier" as identity method
    → CNN classifier settings panel becomes visible
    → User selects or imports a model via combo
        → _handle_add_new_cnn_identity_model() on sentinel selection
            → QFileDialog opens (*.pth *.pt)
            → checkpoint metadata read
            → metadata import dialog pre-filled + user edits
            → file copied to models/classification/identity/
            → model_registry.json updated
            → combo refreshed, new model selected
        → verification panel updates from registry entry (no .pth re-read)

Tracking starts with identity method = "CNN Classifier"
    → precompute phase (before tracking loop)
        → CNNIdentityBackend loaded (lazy ONNX derivation if needed)
        → OBB crops extracted per frame
        → predict_batch() run in batches
        → CNNIdentityCache.save() per frame
        → progress emitted per frame
    → tracking loop begins
        → per frame: CNN predictions loaded from cache
        → confident predictions appended to track sliding windows
        → majority vote → track identity assigned
        → Hungarian cost matrix adjusted:
            match  → cost -= match_bonus  (absolute, no max_cost multiplication)
            clash  → cost += mismatch_penalty  (absolute, no max_cost multiplication)
            uncertain → no change
        → assignment proceeds as normal
```

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/multi_tracker/core/identity/cnn_identity.py` | New — `CNNIdentityConfig`, `CNNIdentityBackend`, `ClassPrediction`, `CNNIdentityCache` |
| `src/multi_tracker/core/tracking/worker.py` | Add CNN identity precompute phase |
| `src/multi_tracker/core/assigners/hungarian.py` | Add match bonus / mismatch penalty for CNN identity |
| `src/multi_tracker/gui/main_window.py` | Add identity method option, settings panel, `_handle_add_new_cnn_identity_model()`, model registry integration |
| `configs/default.json` | Add 8 new `CNN_CLASSIFIER_*` config keys; retain `COLOR_TAG_MODEL_PATH` alias |
| `tests/test_mat_cnn_identity.py` | New test file |

---

## Testing (`tests/test_mat_cnn_identity.py`)

Unit tests — mock model inference, no real GPU required.

| Test | What it covers |
|---|---|
| `CNNIdentityConfig` defaults and field types | Config dataclass correctness |
| `ClassPrediction` dataclass fields | Prediction dataclass contract |
| `CNNIdentityCache` round-trip save / load | Cache serialization |
| `predict_batch()` returns one result per crop | Output cardinality |
| Predictions below confidence threshold return `class_name=None` | Threshold gate |
| Sliding window majority vote: 3/5 same class → identity assigned | Majority vote logic |
| Sliding window with no majority → identity unassigned | No-majority path |
| Assignment cost: match bonus applied when classes match | Cost adjustment — match |
| Assignment cost: mismatch penalty applied when classes conflict | Cost adjustment — mismatch |
| Assignment cost: no adjustment when track identity is unassigned | Cost adjustment — uncertain track |
| Assignment cost: no adjustment when detection confidence below threshold | Cost adjustment — uncertain detection |
| Checkpoint metadata read: `.pth` fields extracted correctly | Import metadata extraction |
| Registry entry format correct after import | Registry schema |
