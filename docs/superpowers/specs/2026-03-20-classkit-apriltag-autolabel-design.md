# ClassKit AprilTag Auto-Labeler — Design Spec

**Date:** 2026-03-20
**Status:** Approved

---

## Overview

Add an AprilTag auto-labeling capability to ClassKit. Given a loaded dataset of pre-cropped animal images (one animal per image, matching MAT's OBB crop format), the user can trigger AprilTag detection across all images to automatically generate training labels. This produces a labeled dataset that can then be used to train a classification model — one that is more robust to lighting and contrast variation than direct AprilTag detection.

---

## Goals

- Reuse MAT's `AprilTagDetector` (with its custom tag family support) as the detection backend.
- Try multiple preprocessing profiles per image so tag detection is robust to difficult lighting conditions.
- Only write labels where the detector is confident; uncertain images are left unlabeled rather than mislabeled.
- Automatically create the ClassKit labeling scheme from the user-selected tag family and max tag ID.
- Fit cleanly into the existing ClassKit background worker / Qt signal pattern.

---

## Non-Goals

- No new standalone app or entry point — this is an addition to `classkit-labeler`.
- No online (real-time) AprilTag detection during tracking — that remains in MAT's pipeline.
- No automatic hyperparameter tuning of detection parameters.

---

## Supported Tag Families

The dialog family dropdown exposes this static list (standard `apriltag` library families):

```
tag36h11  (default)
tag25h9
tag16h5
tagCircle21h7
tagCircle49h12
tagCustom48h12
tagStandard41h12
tagStandard52h13
```

---

## Architecture

### New Module: `src/multi_tracker/classkit/autolabel/apriltag.py`

Pure Python, no Qt dependency, fully testable in isolation.

#### Preprocessing Profiles and Detector Interaction

`AprilTagDetector` unconditionally applies its own unsharp mask and contrast enhancement inside `_detect_composite`. To ensure the autolabeler's preprocessing profiles produce distinct input conditions, the `AprilTagConfig` used internally by the autolabeler **must be constructed with `unsharp_amount=0.0` and `contrast_factor=1.0`** — disabling the detector's built-in preprocessing. The profiles then act as the sole preprocessing stage. The autolabeler creates this modified config from the user-supplied config before constructing the detector:

```python
internal_config = dataclasses.replace(config, unsharp_amount=0.0, contrast_factor=1.0)
detector = AprilTagDetector(internal_config)
```

Profiles are named functions with signature `(np.ndarray) → np.ndarray`. **Input and output must always be 3-channel BGR uint8.** This is required because `_detect_composite` internally calls `cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)` — a single-channel output from any profile would raise a runtime error. Profiles that conceptually work in grayscale (e.g. CLAHE) must convert to grayscale internally, apply their transform, then convert back to BGR before returning.

| Name | Description |
|---|---|
| `raw` | No preprocessing — image passed through unchanged |
| `clahe` | Convert to grayscale → CLAHE → convert back to BGR |
| `gamma_boost` | Gamma correction γ=0.5 to brighten dark images (applied per BGR channel) |
| `gamma_darken` | Gamma correction γ=2.0 to darken overexposed images (applied per BGR channel) |
| `unsharp_strong` | Unsharp mask with `amount=config.unsharp_amount` × 2, `sigma=config.unsharp_sigma`, `kernel_size=(n, n)` where n = `config.unsharp_kernel_size` |

`unsharp_strong` reads three fields from the **user-supplied** (not internal) config: `unsharp_amount`, `unsharp_sigma`, `unsharp_kernel_size`. `unsharp_kernel_size` is a single integer in the user config; it is converted to the tuple `(n, n)` when passed to the unsharp mask function (matching `AprilTagConfig`'s `Tuple[int, int]` field type).

**All profiles are always run** — no short-circuit. This ensures confidence can be computed from agreement across all profiles.

#### Detector Wrapping

`AprilTagDetector.detect_in_crops(crops, offsets_xy)` returns `list[TagObservation]`. For single-image autolabeling, wrap as:

```python
observations = detector.detect_in_crops(crops=[preprocessed_image], offsets_xy=[(0, 0)])
```

Since the offset is `(0, 0)`, coordinates are in image-local space (irrelevant for classification purposes).

When a profile returns **multiple** `TagObservation` objects (more than one tag in a single crop), the profile picks the observation with the **lowest hamming distance**. If multiple observations share the minimum hamming, the profile result is **discarded** (treated as `AMBIGUOUS`, not `NO_TAG`).

#### Profile Result Types

Each profile produces one of three outcomes (tracked as an enum or constant):

| Outcome | Meaning |
|---|---|
| `DETECTED(tag_id)` | One clear tag ID detected |
| `NO_TAG` | No observations returned by the detector |
| `AMBIGUOUS` | Multiple observations with tied minimum hamming |

#### `AprilTagAutoLabeler`

```python
class AprilTagAutoLabeler:
    def __init__(self, config: AprilTagConfig, confidence_threshold: float = 0.6):
        ...

    def label_image(self, image: np.ndarray) -> LabelResult:
        ...
```

Per-image logic:

1. Run all profiles. Record each outcome as `DETECTED(id)`, `NO_TAG`, or `AMBIGUOUS`.
2. Filter to `DETECTED` outcomes only. If none → return `LabelResult(label="no_tag", confidence=1.0, n_profiles_run=N, detected_tag_id=None, all_no_tag=True)`.
   - Note: if all profiles are `AMBIGUOUS` (not `NO_TAG`), the result is still `no_tag` with confidence 1.0. This edge case (all profiles ambiguous) is accepted as-is; in practice, consistently ambiguous crops are not useful training data.
3. Find the majority tag ID among `DETECTED` outcomes. `confidence = count_of_majority / N_total_profiles` (denominator is total profiles, including `NO_TAG` and `AMBIGUOUS` ones).
4. If `confidence >= threshold` → return `LabelResult(label=f"tag_{majority_id}", confidence=confidence, ...)`.
5. If `confidence < threshold` → return `LabelResult(label=None, confidence=confidence, ...)`.

#### `LabelResult` dataclass

```python
@dataclass
class LabelResult:
    label: str | None           # "tag_N", "no_tag", or None (unlabeled/uncertain)
    confidence: float           # count_of_majority / N_total_profiles
    n_profiles_run: int         # always len(PREPROCESSING_PROFILES)
    detected_tag_id: int | None # raw integer majority tag ID, or None
    all_no_tag: bool = False    # True when every profile returned NO_TAG (vs. AMBIGUOUS or disagreement)
```

#### Batch entrypoint

```python
def autolabel_images(
    image_paths: list[Path],
    config: AprilTagConfig,
    threshold: float,
) -> list[LabelResult]:
    ...
```

Processes images sequentially (the AprilTag detector is already multi-threaded internally via its `threads` parameter). Returns an empty list for an empty input.

#### Public exports from `autolabel/__init__.py`

```python
from .apriltag import AprilTagAutoLabeler, LabelResult, autolabel_images
```

---

### Labeling Scheme Auto-Creation

#### `apriltag_preset()` in `presets.py`

```python
def apriltag_preset(family: str, max_tag_id: int) -> LabelingScheme:
    """Single-factor scheme for AprilTag classification."""
    return LabelingScheme(
        name=f"apriltag_{family}",
        factors=[
            Factor(
                name=family,
                labels=[f"tag_{i}" for i in range(max_tag_id + 1)] + ["no_tag"],
            )
        ],
        training_modes=["flat_tiny", "flat_yolo"],  # single-factor convention
    )
```

Single-factor schemes use only flat training modes (`flat_tiny`, `flat_yolo`). Multihead modes are excluded per existing codebase convention (see `head_tail_preset()`, `age_preset()`).

#### Scheme storage and label clearing

The scheme is stored on disk as `project_path / "scheme.json"`. It is **not** stored in the DB.

When the user confirms scheme creation (including "Replace" on an existing scheme):
1. `apriltag_preset()` writes the new `scheme.json` to disk.
2. All existing labels in the DB `images` table are cleared (set to `NULL`) via a new `db.clear_all_labels()` method. This prevents stale label strings from a previous scheme from appearing as valid labels under the new scheme.
3. The main window reloads `scheme.json` and rebuilds its label buttons and `self.classes`.

The user is warned in the "Replace" dialog that existing labels will be erased.

---

### ClassKit GUI Integration

#### `AprilTagAutoLabelDialog` (added to `gui/dialogs.py`)

Two sections:

**AprilTag Detection Parameters** (mirrors `AprilTagConfig`):
- Tag family dropdown (static list above, default `tag36h11`)
- Max tag ID spinbox (default 9)
- Max hamming spinbox (default 1)
- Decimate spinbox (default 2.0)
- Blur spinbox (default 0.8)
- Contrast factor spinbox (default 1.5) — used internally by `unsharp_strong` profile only (detector's own contrast is disabled)
- Unsharp amount spinbox (default 1.0) — used by `unsharp_strong` profile
- Unsharp sigma spinbox (default 1.0) — used by `unsharp_strong` profile
- Unsharp kernel size spinbox (default 5, integer n) — maps to `(n, n)` tuple when constructing `AprilTagConfig`; used by `unsharp_strong` profile

**Labeling Parameters:**
- Confidence threshold slider (0.0–1.0, default 0.6)
- Preview button — runs detection on up to 10 randomly sampled images, shows hit rate, tag ID distribution, and skip rate before the user commits

**Scheme Preview Panel:**
- Live-updating label list showing `tag_0` … `tag_{max_tag_id}` + `no_tag` as the user adjusts max tag ID and family.

#### `AprilTagAutoLabelWorker` (added to `jobs/task_workers.py`)

Standard `QRunnable` subclass with `setAutoDelete(False)` in `__init__`:

```python
class AprilTagAutoLabelWorker(QRunnable):
    def __init__(
        self,
        image_paths: list[Path],       # pre-filtered to unlabeled images by main window
        config: AprilTagConfig,
        threshold: float,
        db: ClassKitDB,                # used only for batch label writes
    ):
        super().__init__()
        self.setAutoDelete(False)
        self.signals = TaskSignals()
        self._canceled = False
        ...

    def cancel(self) -> None:
        self._canceled = True
```

Worker logic:

1. `image_paths` is the pre-filtered unlabeled list passed in from the main window (filtering is the caller's responsibility, not the worker's). The worker uses `db` only for `update_labels_with_confidence_batch()`.
2. Call `autolabel_images()` in batches of 100.
3. For each `LabelResult`:
   - `label is not None` → accumulate into batch dict `{str(path): (label, confidence)}`
   - `label is None` → skip.
4. After each batch: call `db.update_labels_with_confidence_batch(updates)`, check `_canceled`, emit `signals.progress(percent, f"Labeled {n_labeled}, skipped {n_skipped}")`.
5. On completion emit `signals.finished()` and `signals.success(summary_string)`.
6. On cancellation emit `signals.finished()` with a cancellation note.

#### `ClassKitDB` additions (`store/db.py`)

Two new methods:

```python
def update_labels_with_confidence_batch(self, updates: dict[str, tuple[str, float]]) -> None:
    """Write label and confidence for multiple images. updates: {path: (label, confidence)}"""
    # single executemany: UPDATE images SET label=?, confidence=? WHERE path=?

def clear_all_labels(self) -> None:
    """Set label=NULL and confidence=NULL for all images."""
    # UPDATE images SET label=NULL, confidence=NULL
```

#### `mainwindow.py` trigger

An **"Auto-label AprilTags…"** action added to the existing toolbar or Edit menu:
- Disabled when no project is loaded.
- Opens `AprilTagAutoLabelDialog`.
- On confirm:
  1. Write `scheme.json` via `apriltag_preset()`.
  2. Call `db.clear_all_labels()` to erase stale labels.
  3. Reload scheme and rebuild label buttons.
  4. Collect unlabeled image paths: `[p for p, lbl in zip(db.get_all_image_paths(), db.get_all_labels()) if lbl is None]`.
  5. Pass paths + config + threshold to `AprilTagAutoLabelWorker`. Start the worker.
  6. Show progress in the existing status bar / progress widget; cancel button calls `worker.cancel()`.
  7. On worker completion, refresh label count display.

---

## Data Flow

```
User clicks "Auto-label AprilTags..."
    → AprilTagAutoLabelDialog opens
        user configures detection params + confidence threshold
        live scheme preview shows labels that will be created
        optional: preview on ≤10 images
    → user confirms
        → apriltag_preset() writes scheme.json to project dir
        → db.clear_all_labels() erases stale labels
        → main window reloads scheme, rebuilds label buttons
        → unlabeled image paths collected from db
        → AprilTagAutoLabelWorker starts (with setAutoDelete(False))
            → autolabel_images() runs all profiles per image
                detector uses internal config (unsharp_amount=0, contrast_factor=1)
            → DETECTED majority → accumulate (tag_X, confidence)
            → all NO_TAG or AMBIGUOUS → accumulate ("no_tag", 1.0)
            → profiles disagree below threshold → skip
            → batch write via update_labels_with_confidence_batch()
            → emit progress(percent, status_string)
            → check _canceled flag between batches
        → on complete: refresh UI label counts + show summary
```

---

## Testing

New file: `tests/test_classkit_apriltag_autolabel.py`

| Test | What it covers |
|---|---|
| Each profile returns correct dtype, shape, and value range | Preprocessing correctness |
| `raw` profile with internal config (no detector preprocessing) passes image unchanged | Detector double-application prevention |
| `no_tag` with confidence 1.0 when all profiles return `NO_TAG` | No-tag path |
| `no_tag` when all profiles return `AMBIGUOUS` (accepted edge case) | Ambiguous-all path |
| Correct confidence fraction (3/5 profiles agree → 0.6) | Confidence math |
| Below-threshold result returns `None` label | Threshold gate |
| Multi-tag per-profile: tied hamming discards as `AMBIGUOUS` | Multi-tag edge case |
| Multi-tag per-profile: clear winner (lower hamming) is used | Multi-tag tie-break |
| Majority tag wins when profiles return mixed IDs | Majority vote logic |
| Detector is called with `crops=[image], offsets_xy=[(0,0)]` | Wrapping contract |
| `apriltag_preset()` produces correct labels, single factor, flat training_modes | Preset correctness |
| `autolabel_images()` handles empty list | Edge case |

`AprilTagDetector` is mocked in all tests — the `apriltag` package is not required in the test environment.

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/multi_tracker/classkit/autolabel/__init__.py` | New package — exports `AprilTagAutoLabeler`, `LabelResult`, `autolabel_images` |
| `src/multi_tracker/classkit/autolabel/apriltag.py` | New — core auto-labeler logic |
| `src/multi_tracker/classkit/presets.py` | Add `apriltag_preset()` |
| `src/multi_tracker/classkit/store/db.py` | Add `update_labels_with_confidence_batch()` and `clear_all_labels()` |
| `src/multi_tracker/classkit/gui/dialogs.py` | Add `AprilTagAutoLabelDialog` |
| `src/multi_tracker/classkit/jobs/task_workers.py` | Add `AprilTagAutoLabelWorker` with `cancel()` and `setAutoDelete(False)` |
| `src/multi_tracker/classkit/gui/mainwindow.py` | Add menu action, scheme reload, label clear, worker hookup |
| `tests/test_classkit_apriltag_autolabel.py` | New test file |
