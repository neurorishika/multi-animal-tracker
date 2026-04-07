# Codebase Simplification Sprint — Design Spec

**Date:** 2026-04-04
**Author:** Rishika Mohanta
**Status:** Approved

---

## Context

The hydra-suite codebase has grown to ~100K LOC across 227 files. A structural analysis identified four compounding pain points that collectively slow down development:

1. **Monolith files** — `trackerkit/gui/main_window.py` is 19,972 lines with 11 classes and 5 embedded workers. Every edit is risky and hard to navigate.
2. **Cross-kit duplication** — patterns for workers, dialogs, and config are reinvented independently in each kit (trackerkit, posekit, classkit, refinekit, detectkit, filterkit). Fixes must be applied in 4–6 places.
3. **No shared worker pattern** — 15 `QThread` subclasses with no base class. Each implements signal definitions, error handling, and cancellation differently.
4. **No consistent config/state pattern** — only `classkit` has typed schemas. All other kits carry state as scattered widget attributes on `MainWindow`.

**Risk tolerance:** Medium — a dedicated refactoring sprint where things may be temporarily in flux, completing within a few weeks. No public API or CLI entry-point changes throughout.

---

## Target State

A codebase where:

- Every background task inherits from one `BaseWorker(QThread)` with a standard signal interface.
- Every kit's runtime state lives in a typed config schema, not scattered across widget attributes.
- Every dialog inherits from a thin `BaseDialog` that handles modal setup, button box, and accept/reject boilerplate.
- Every kit's main window is a thin coordinator of focused sub-modules, not a God object.
- Adding a new kit or feature follows an obvious template with no tribal knowledge required.

---

## Approach: Horizontal Slices

Four independent slices, executed in order. Each slice is independently mergeable. Slices 1–3 establish the abstractions that make Slice 4 mechanical.

---

## Slice 1 — Worker Pattern Unification

### Problem

15 `QThread` subclasses across trackerkit, classkit, posekit, refinekit, detectkit, core, and data. No shared base class. Error handling, signal definitions, progress reporting, and the finished-always-fires guarantee are reinvented each time.

### Solution

New file: `src/hydra_suite/widgets/workers.py`

```python
class BaseWorker(QThread):
    progress = Signal(int)   # 0–100
    status = Signal(str)     # human-readable status message
    error = Signal(str)      # error message on failure
    finished = Signal()      # always emitted on completion (success or failure)

    def run(self):
        try:
            self.execute()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def execute(self):
        raise NotImplementedError
```

Each existing worker becomes a subclass that only implements `execute()`. Workers that emit additional custom signals (e.g., `TrackingWorker` emitting per-frame results) extend the base signals — they do not lose them.

**Slice 1 scope is class-level only.** Workers are changed to inherit `BaseWorker` in-place; files are not moved in this slice. File movement for trackerkit workers (currently embedded in `main_window.py`) happens in Slice 4.

### Scope

Migrate all 15 workers:

| Worker | Location |
|--------|----------|
| `MergeWorker` | trackerkit/gui/main_window.py |
| `InterpolatedCropsWorker` | trackerkit/gui/main_window.py |
| `OrientedTrackVideoWorker` | trackerkit/gui/main_window.py |
| `DatasetGenerationWorker` | trackerkit/gui/main_window.py |
| `PreviewDetectionWorker` | trackerkit/gui/main_window.py |
| `TrackingWorker` | core/tracking/worker.py |
| `DetectionCacheBuilderWorker` | core/tracking/optimizer_workers.py |
| Optimizer workers | core/tracking/optimizer_workers.py |
| Training worker | detectkit/gui/panels/training_panel.py |
| Training worker | posekit/gui/dialogs/training.py |
| Job workers | classkit/jobs/task_workers.py |
| Track editor worker | refinekit/gui/ |
| Resolution dialog worker | refinekit/gui/ |
| Synced video grid worker | refinekit/gui/ |
| `CSVWriterThread` | data/csv_writer.py |

### Risk

Low. Workers are internal. No public API changes. Each migration is independently testable.

---

## Slice 2 — Config/Schema Unification

### Problem

Only `classkit` has a typed config schema (`ProjectConfig`, `ModelConfig` as dataclasses with `to_dict`/`from_dict`). All other kits carry state as `self.some_flag = True` on `MainWindow`. This makes state impossible to serialize, validate, or test without instantiating Qt widgets.

### Solution

Each kit gets `<kit>/config/schemas.py` following the classkit pattern:

```python
@dataclass
class TrackerConfig:
    video_path: str = ""
    model_path: str = ""
    n_animals: int = 1
    compute_runtime: str = "cpu"
    # ... etc

    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, data: dict) -> "TrackerConfig": ...
```

`MainWindow.__init__` initializes `self.config = TrackerConfig()`. The GUI reads and writes `self.config`; it no longer *is* the state. Config instances are passed to workers instead of threading the entire `MainWindow` through.

### Kits to add schemas

`trackerkit`, `posekit`, `refinekit`, `detectkit`, `filterkit` (classkit already complete).

### Side benefits

- Config can be saved/loaded to JSON for session persistence.
- Workers receive a plain dataclass, not a GUI object.
- Unit tests can validate state transitions without instantiating Qt widgets.

### Risk

Medium. Touches `MainWindow` initialization and worker invocation in each kit. Done kit-by-kit, each is independently safe to merge.

---

## Slice 3 — Dialog Pattern Unification

### Problem

No consistent dialog structure. `classkit/gui/dialogs.py` bundles 11 dialog classes in 3,282 lines. File layout varies per kit (some use `dialogs/` subdirectory, some use a single file, some embed in main_window). No shared boilerplate for modal setup, button boxes, or accept/reject handling.

### Solution

New file: `src/hydra_suite/widgets/dialogs.py`

```python
class BaseDialog(QDialog):
    def __init__(self, title: str, parent=None,
                 buttons=QDialogButtonBox.Ok | QDialogButtonBox.Cancel):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._layout = QVBoxLayout(self)
        self._content_layout = QVBoxLayout()
        self._layout.addLayout(self._content_layout)
        self._buttons = QDialogButtonBox(buttons)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._layout.addWidget(self._buttons)  # button box always at bottom, auto-added

    def add_content(self, widget: QWidget):
        self._content_layout.addWidget(widget)
```

The button box is added automatically in `__init__` — subclasses call `add_content()` and never need to finalize manually. Custom button configurations (Ok-only, custom labels) are passed via the `buttons` constructor argument.

### File layout standardization

Each kit gets a `<kit>/gui/dialogs/` subpackage (one file per dialog), matching the pattern already established in `posekit/gui/dialogs/` and `refinekit/gui/`.

`classkit/gui/dialogs.py` (3,282 lines, 11 classes) is split into 11 individual files:

- `dialogs/add_source.py`
- `dialogs/source_manager.py`
- `dialogs/class_editor.py`
- `dialogs/shortcut_editor.py`
- `dialogs/new_project.py`
- `dialogs/embedding.py`
- `dialogs/cluster.py`
- `dialogs/training.py`
- `dialogs/export.py`
- `dialogs/model_history.py`
- `dialogs/apriltag_autolabel.py`

### Risk

Low per dialog. Dialogs are self-contained. Each split is independently mergeable.

---

## Slice 4 — Monolith Decomposition (trackerkit)

### Problem

`trackerkit/gui/main_window.py` is 19,972 lines with 11 classes and 5 embedded workers. It is the single biggest drag on developer velocity. Every feature addition touches the same file, causing merge conflicts and making review impossible.

### Solution

Decompose into a `trackerkit/gui/` subpackage mirroring the structure already used by `posekit`, `detectkit`, and `refinekit`. By the time this slice runs, workers are already extracted (Slice 1), config state is in schemas (Slice 2), and dialogs are split (Slice 3). The seams are pre-cut.

**Target layout:**

```
trackerkit/gui/
    __init__.py
    main_window.py          # thin coordinator, ~200 lines
    panels/
        tracking_panel.py   # tracking controls & playback UI
        detection_panel.py  # detector/model config UI
        identity_panel.py   # identity/assignment config UI
        postprocess_panel.py # post-processing UI
    workers/
        merge_worker.py     # migrated in Slice 1
        preview_worker.py   # migrated in Slice 1
        (other workers)
    dialogs/
        training.py         # TrainYoloDialog
        parameters.py       # ParameterHelper, BgParameterHelper
        export.py           # video/export dialogs
```

`MainWindow` becomes a thin coordinator: instantiates panels, connects signals between them, delegates to `TrackerConfig`. It holds no business logic.

### Why last

Slices 1–3 pre-cut the seams. The decomposition becomes mechanical: workers are already in their own files, config state is already a separate object, dialogs are already extracted. What remains is reorganizing the panel/UI logic into focused modules.

### Risk

Medium. Largest single refactor, but done last when all abstractions are established. No public API changes — the `trackerkit` CLI entry point stays identical.

---

## Implementation Order

| Slice | New shared infrastructure | Kits touched |
|-------|--------------------------|--------------|
| 1 — Workers | `widgets/workers.py` (BaseWorker) | All 6 kits + core + data |
| 2 — Config | Per-kit `config/schemas.py` | trackerkit, posekit, refinekit, detectkit, filterkit |
| 3 — Dialogs | `widgets/dialogs.py` (BaseDialog) | classkit (split), all kits (migrate) |
| 4 — Monolith | `trackerkit/gui/` subpackage | trackerkit only |

---

## Out of Scope

- Changes to public CLI entry points or inter-kit APIs
- New features of any kind
- Refactoring `core/post/processing.py` (2,900 lines) — deferred, no shared abstraction applies yet
- Refactoring `classkit/jobs/task_workers.py` beyond worker base-class migration
- Test coverage improvements (a follow-on effort after structure stabilizes)

---

## Success Criteria

- [ ] All 15 workers inherit from `BaseWorker`; no worker defines its own `run()` boilerplate
- [ ] All 6 kits have a `config/schemas.py`; `MainWindow.__init__` initializes `self.config`
- [ ] `classkit/gui/dialogs.py` is deleted; replaced by 11 files in `classkit/gui/dialogs/`
- [ ] All dialogs that implement modal+buttonbox boilerplate inherit from `BaseDialog`
- [ ] `trackerkit/gui/main_window.py` is under 500 lines
- [ ] All existing tests pass throughout
- [ ] No public API or CLI entry-point changes
