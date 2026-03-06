# ClassKit Production Design

**Date:** 2026-03-06
**Branch:** `mat-pose-integration`
**Status:** Approved — ready for implementation planning

---

## Goal

Bring ClassKit to production-ready status as the sole animal labeling and classification training tool. This covers multi-factor labeling, preset workflows, a full training surface (replacing MAT's classification training), and publishing trained models to the shared `models/` registry for MAT discovery.

Active learning (`classkit/al/`) is explicitly out of scope — stretch goal for a future iteration.

---

## 1. Data Model

### `LabelingScheme` — new dataclasses in `classkit/config/schemas.py`

```python
@dataclass
class Factor:
    name: str               # e.g. "tag_1", "direction"
    labels: List[str]       # ordered list, e.g. ["red","blue","green","yellow","white"]
    shortcut_keys: List[str] = field(default_factory=list)  # optional key hints

@dataclass
class LabelingScheme:
    name: str               # e.g. "color_tags_2factor"
    factors: List[Factor]   # ordered; single factor = flat classification
    training_modes: List[str]  # subset of ["flat_tiny","flat_yolo","multihead_tiny","multihead_yolo"]
    description: str = ""
```

**Composite label encoding:** factor values joined by pipe — `"red|blue"`. Reversible, human-readable, valid as a YOLO class folder name.

**`ProjectConfig`** gains one new field:
```python
scheme: Optional[LabelingScheme] = None
```
When `None`, ClassKit behaves exactly as today (free-form flat label list, no stepper).

### Preset factory functions — new `classkit/presets.py`

| Preset | Factors | Classes |
|---|---|---|
| `head_tail_preset()` | 1 | left / right / up / down |
| `color_tag_preset(n_factors, colors)` | N | M colors each → M^N composites |
| `age_preset(extra_classes=[])` | 1 | young / old + extras |

Presets are pure data — they return a `LabelingScheme`. The `NewProjectDialog` gains a preset selector that populates the scheme, with a "Custom" option for free-form.

---

## 2. Labeling UX — Factor Stepper

### New widget: `classkit/gui/widgets/factor_stepper.py` — `FactorStepperWidget`

Replaces the flat label button row when a multi-factor scheme is active.

**Layout per step:**
```
Factor 1 of 2: tag_1
[Red]  [Blue]  [Green]  [Yellow]  [White]
<Back    Skip>
```

**Interaction flow:**
1. On image load, stepper resets to factor 1
2. Clicking a label (or pressing its shortcut key) commits that factor and auto-advances
3. After the final factor, composite key is assembled (`"red|blue"`) and emitted via signal
4. `MainWindow._set_label_for_index()` receives the composite string — no changes needed there
5. Back clears the last factor choice and steps back one
6. Skip leaves the image unlabeled and emits a skip signal

**Shortcut binding:**
- `MainWindow.setup_label_shortcuts()` is extended to handle multi-factor mode
- In multi-factor mode, keys 1–9 map to the active factor's labels and rebind on each factor advance
- `Factor.shortcut_keys` provides explicit key overrides per label

**UMAP / explorer coloring:**
- Default: color by full composite label (hashed to consistent color per combination)
- Toggle: color by factor 1 only (useful for large Cartesian products, e.g. 125 classes)

**History strip / undo:**
- Unchanged — composite label is stored as a single string; undo pops the whole composite assignment

---

## 3. ClassKit Training Surface

### Replaces `TrainingDialog` in `classkit/gui/dialogs.py`

New `ClassKitTrainingDialog` modeled on MAT's `TrainYoloDialog` — live log panel, progress bar, cancel, publish action.

**Dialog sections:**
1. **Scheme summary** — displays scheme name and total class count (computed from factors)
2. **Training mode selector** — radio buttons, only valid modes shown for the active scheme:
   - Flat – Tiny CNN
   - Flat – YOLO-classify
   - Multi-head Tiny (one tiny CNN per factor)
   - Multi-head YOLO (one YOLO-classify per factor)
3. **Hyperparameters** — device, base model (for YOLO), epochs, batch, LR, val fraction
4. **Live log panel** — scrolling `QPlainTextEdit`, same pattern as `TrainYoloDialog`
5. **Progress bar**
6. **Action buttons** — Start / Cancel / Publish (enabled after successful run)

**Training mode availability rules:**
- Single factor: all four modes available
- Multiple factors, total composites ≤ 100: all modes available
- Multiple factors, total composites > 100: flat YOLO recommended; flat tiny available; multi-head always available

**Worker thread:**
- Mirrors `RoleTrainingWorker` from MAT
- Flat modes: calls `training/runner.py run_training()` once with the full composite class export
- Multi-head modes: calls `run_training()` once per factor, each with a single-factor derived dataset

**Dataset preparation:**
- Reuses `classkit/export/ultralytics_classify.py export_ultralytics_classify()` before each training run
- Multi-head derives N single-factor datasets by projecting composite labels to the Nth factor

**Publish action (active after successful run):**
- Calls `training/model_publish.py publish_trained_model()` with new role values
- Flat models: single call per training run
- Multi-head models: one call per factor, each with `factor_index` in metadata

**`MainWindow.train_classifier()`** updated to open `ClassKitTrainingDialog` instead of `TrainingDialog`.

---

## 4. Model Storage

### New `TrainingRole` enum values in `training/contracts.py`

```
CLASSIFY_FLAT_YOLO
CLASSIFY_FLAT_TINY
CLASSIFY_MULTIHEAD_YOLO
CLASSIFY_MULTIHEAD_TINY
```

### Storage layout

```
models/
  YOLO-classify/
    orientation/                    # existing head-tail (unchanged)
    <scheme_name>/                  # flat YOLO ClassKit models
      20260306-143022_n_<species>_<tag>.pt
  tiny-classify/
    <scheme_name>/                  # flat tiny CNN ClassKit models
      20260306-143022_tiny_<species>_<tag>.pth
  YOLO-classify/multihead/
    <scheme_name>/
      factor_0_<factor_name>.pt     # one file per factor
      factor_1_<factor_name>.pt
  tiny-classify/multihead/
    <scheme_name>/
      factor_0_<factor_name>.pth
      factor_1_<factor_name>.pth
```

### `model_registry.json` additions

All ClassKit-published entries include:
```json
{
  "task_family": "classify",
  "usage_role": "classify_flat_yolo",   // or multihead variant
  "scheme_name": "color_tags_2factor",
  "factor_index": null,                 // or 0, 1, 2 for multi-head
  "factor_name": null,                  // or "tag_1", "tag_2"
  "factors": [...]                      // full Factor list for multi-head grouping
}
```

MAT discovers these at startup via the existing `load_model_registry()` scan — no MAT startup changes needed.

---

## 5. MAT Migration

### `TrainYoloDialog` changes

- Remove `HEADTAIL_TINY` and `HEADTAIL_YOLO` from the roles panel
- Add info banner: *"Classification training has moved to ClassKit labeler (`classkit-labeler`)"* with optional launch button
- Remove `_choose_headtail_override`, `_set_headtail_source`, and the head-tail override file picker
- Remove classify branches from `_base_model_for_role()` and `_build_role_datasets()`

### `training/contracts.py`

- `HEADTAIL_TINY` and `HEADTAIL_YOLO` kept in enum (backwards compatibility for existing registry entries)
- Marked deprecated in docstring — not exposed in any dialog

### `training/runner.py`

- `_train_tiny_headtail()` stays (needed to re-run existing published tiny models)
- `_iter_classify_samples()` generalized to accept arbitrary class folder names (not hardcoded `head_left`/`head_right`) so ClassKit's flat tiny training can reuse it

### MAT identity pipeline

No changes in this iteration. MAT auto-discovers ClassKit-published models from `model_registry.json`. Integration of multi-head bundles into the tracking pipeline is a follow-up task once models exist in practice.

---

## Out of Scope

- Active learning wiring (`classkit/al/`) — stretch goal
- MAT identity/post-processing integration for multi-head bundles — follow-up
- ONNX/TensorRT export of ClassKit models — follow-up
