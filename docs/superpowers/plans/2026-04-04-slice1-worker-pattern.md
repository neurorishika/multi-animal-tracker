# Slice 1 — Worker Pattern Unification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `BaseWorker(QThread)` in `hydra_suite.widgets.workers` and migrate all 15 substantive QThread background-task workers to inherit from it.

**Architecture:** `BaseWorker` owns the `run()` method, standard signals (`progress`, `status`, `error`, `finished`), and the try/except+finally guarantee. Subclasses implement `execute()` only. Migration is class-level only — files are not moved in this slice.

**Tech Stack:** PySide6 (`QThread`, `Signal`), pytest

**Scope note:** This slice targets QThread workers in app-layer packages only.
- `CSVWriterThread` (`threading.Thread`) — excluded, different model.
- posekit workers (`QObject` with moveToThread) — excluded, different model.
- classkit workers (`QRunnable`/thread pool) — excluded, different model.
- Private helpers (`_PrefetchThread`, `_PlaybackThread`, `_MultiCropLoader`, `_FrameLoader`, `_TestWorker`) — excluded, tightly coupled to single widgets.
- **`core/` workers excluded** — `TrackingWorker`, `TrackingOptimizer`, `DetectionCacheBuilderWorker`, `TrackingPreviewWorker`, `BgSubtractionOptimizer`, `BgDetectionPreviewWorker` live in `hydra_suite.core` which must not import from `hydra_suite.widgets` (app-layer boundary per CLAUDE.md). These workers already have well-defined signal interfaces and are not migrated in this slice.

---

### Task 1: Create BaseWorker

**Files:**
- Create: `src/hydra_suite/widgets/workers.py`
- Modify: `src/hydra_suite/widgets/__init__.py`
- Create: `tests/test_base_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_base_worker.py
import sys
import pytest
from PySide6.QtCore import QCoreApplication

@pytest.fixture(scope="session")
def qapp():
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication(sys.argv[:1])
    return app


def test_base_worker_execute_called(qapp):
    """execute() is called when worker is started."""
    from hydra_suite.widgets.workers import BaseWorker

    class _EchoWorker(BaseWorker):
        def execute(self):
            self.status.emit("hello")
            self.progress.emit(100)

    received = []
    worker = _EchoWorker()
    worker.status.connect(received.append)
    worker.start()
    worker.wait(3000)

    assert received == ["hello"]


def test_base_worker_finished_always_fires(qapp):
    """finished signal fires even when execute raises."""
    from hydra_suite.widgets.workers import BaseWorker

    class _CrashWorker(BaseWorker):
        def execute(self):
            raise RuntimeError("boom")

    finished_calls = []
    errors = []
    worker = _CrashWorker()
    worker.finished.connect(lambda: finished_calls.append(1))
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)

    assert len(finished_calls) == 1
    assert "boom" in errors[0]


def test_base_worker_error_emitted_on_exception(qapp):
    """error signal carries the exception message."""
    from hydra_suite.widgets.workers import BaseWorker

    class _BadWorker(BaseWorker):
        def execute(self):
            raise ValueError("bad value")

    errors = []
    worker = _BadWorker()
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)

    assert len(errors) == 1
    assert "bad value" in errors[0]


def test_base_worker_no_error_on_success(qapp):
    """error signal is not emitted when execute succeeds."""
    from hydra_suite.widgets.workers import BaseWorker

    class _OkWorker(BaseWorker):
        def execute(self):
            pass

    errors = []
    worker = _OkWorker()
    worker.error.connect(errors.append)
    worker.start()
    worker.wait(3000)

    assert errors == []


def test_base_worker_subclass_extra_signals(qapp):
    """Subclasses can add extra signals beyond the base set."""
    from hydra_suite.widgets.workers import BaseWorker
    from PySide6.QtCore import Signal

    class _ResultWorker(BaseWorker):
        result = Signal(int)

        def execute(self):
            self.result.emit(42)

    results = []
    worker = _ResultWorker()
    worker.result.connect(results.append)
    worker.start()
    worker.wait(3000)

    assert results == [42]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_base_worker.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `hydra_suite.widgets.workers` does not exist yet.

- [ ] **Step 3: Create `workers.py`**

```python
# src/hydra_suite/widgets/workers.py
"""BaseWorker — standard QThread base class for all background tasks."""
from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Base class for all background task workers.

    Subclasses implement ``execute()`` only.  ``run()`` is owned by this
    class and guarantees:
    - ``finished`` is always emitted (success or failure).
    - Unhandled exceptions in ``execute()`` emit ``error`` instead of
      crashing the thread silently.

    Standard signals
    ----------------
    progress(int)  — 0–100 completion percentage
    status(str)    — human-readable status update
    error(str)     — error message; emitted only on exception
    finished()     — always emitted when the worker stops
    """

    progress: Signal = Signal(int)
    status: Signal = Signal(str)
    error: Signal = Signal(str)
    finished: Signal = Signal()

    def run(self) -> None:  # noqa: D102
        try:
            self.execute()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def execute(self) -> None:
        """Override in subclasses with the actual work."""
        raise NotImplementedError
```

- [ ] **Step 4: Export from `widgets/__init__.py`**

Read the current `src/hydra_suite/widgets/__init__.py`, then add:

```python
from hydra_suite.widgets.workers import BaseWorker

__all__ = [...existing exports..., "BaseWorker"]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
python -m pytest tests/test_base_worker.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/widgets/workers.py src/hydra_suite/widgets/__init__.py tests/test_base_worker.py
git commit -m "feat: add BaseWorker(QThread) to widgets with standard signal interface"
```

---

### Task 2: Migrate trackerkit main_window workers (5 workers)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

The five workers below are all defined inside `main_window.py`. The migration for each is the same three-step mechanical change:
1. Add `from hydra_suite.widgets.workers import BaseWorker` at the top of the file (once, not five times).
2. Change each class header from `class FooWorker(QThread):` to `class FooWorker(BaseWorker):`.
3. Rename each `run(self)` method to `execute(self)` and remove any top-level `try/except` wrapper or explicit `self.finished.emit()` call in that method body (BaseWorker handles both).
4. Remove any `finished = Signal()` class attribute on the worker if present (it would shadow BaseWorker's).

Workers and their line numbers:
- `MergeWorker` at line 311
- `InterpolatedCropsWorker` at line 520
- `OrientedTrackVideoWorker` at line 1754
- `DatasetGenerationWorker` at line 1817
- `PreviewDetectionWorker` at line 2088

- [ ] **Step 1: Add BaseWorker import to main_window.py**

Find the block of PySide6 imports near the top of `src/hydra_suite/trackerkit/gui/main_window.py` and add:

```python
from hydra_suite.widgets.workers import BaseWorker
```

Also remove `QThread` from the PySide6 import line if it is only used as a base class for these five workers (check first — it may be used elsewhere too).

- [ ] **Step 2: Migrate MergeWorker (line 311)**

Change:
```python
class MergeWorker(QThread):
```
To:
```python
class MergeWorker(BaseWorker):
```
Rename `def run(self):` → `def execute(self):`. Remove any `self.finished.emit()` at the end and any surrounding `try/except` that only re-emits the error signal (BaseWorker handles both).

- [ ] **Step 3: Migrate InterpolatedCropsWorker (line 520)**

Same change: `(QThread)` → `(BaseWorker)`, `run` → `execute`, remove boilerplate.

- [ ] **Step 4: Migrate OrientedTrackVideoWorker (line 1754)**

Same change.

- [ ] **Step 5: Migrate DatasetGenerationWorker (line 1817)**

Same change.

- [ ] **Step 6: Migrate PreviewDetectionWorker (line 2088)**

Same change.

- [ ] **Step 7: Run the full test suite to verify nothing broke**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

Expected: same pass/fail count as before this task.

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate 5 trackerkit main_window workers to BaseWorker"
```

---

### Task 3: Migrate trackerkit train_yolo_dialog RoleTrainingWorker

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py` (line 56)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate RoleTrainingWorker**

```python
# Before (line 56):
class RoleTrainingWorker(QThread):

# After:
class RoleTrainingWorker(BaseWorker):
```

Rename `run(self)` → `execute(self)`. Remove any `self.finished.emit()` at the end and any bare `try/except` error-propagation wrapper.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py
git commit -m "refactor: migrate trackerkit RoleTrainingWorker to BaseWorker"
```

---

### Task 4: Migrate filterkit SieveWorker

**Files:**
- Modify: `src/hydra_suite/filterkit/gui/main_window.py` (line 39)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate SieveWorker**

```python
# Before (line 39):
class SieveWorker(QThread):

# After:
class SieveWorker(BaseWorker):
```

Rename `run(self)` → `execute(self)`. Remove boilerplate as in Task 2.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/filterkit/gui/main_window.py
git commit -m "refactor: migrate filterkit SieveWorker to BaseWorker"
```

---

### Task 5: Migrate detectkit RoleTrainingWorker

**Files:**
- Modify: `src/hydra_suite/detectkit/gui/panels/training_panel.py` (line 54)

Note: This `RoleTrainingWorker` is a different class from the one in trackerkit. It has signals:
```python
log_signal = Signal(str)
role_started = Signal(str)
role_finished = Signal(str, bool, str)
progress_signal = Signal(str, int, int)
```
These custom signals should be kept. Only the base `run()` boilerplate is removed.

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Change parent class and rename run**

```python
# Before (line 54):
class RoleTrainingWorker(QThread):

# After:
class RoleTrainingWorker(BaseWorker):
```

Rename `run(self)` → `execute(self)`. Keep all custom signal definitions (`log_signal`, `role_started`, etc.) — they extend the base set. Remove any `self.finished.emit()` from `execute()` if present.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/detectkit/gui/panels/training_panel.py
git commit -m "refactor: migrate detectkit RoleTrainingWorker to BaseWorker"
```

---

### Task 6: Migrate core/tracking/worker.py TrackingWorker

**Files:**
- Modify: `src/hydra_suite/core/tracking/worker.py` (line 77)

`TrackingWorker` is the largest worker (2,865 lines). It emits many custom signals for per-frame results. The migration is still mechanical: inherit `BaseWorker`, rename `run` → `execute`.

Important: `TrackingWorker` may already have `finished = Signal()` defined at class level. If so, **remove that class-level definition** — it would shadow `BaseWorker.finished` and create a second disconnected signal.

- [ ] **Step 1: Check for conflicting signal definitions**

Search `src/hydra_suite/core/tracking/worker.py` for:
```
finished = Signal(
```
If found, note the line and remove it in step 3.

- [ ] **Step 2: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 3: Migrate**

```python
# Before (line 77):
class TrackingWorker(QThread):

# After:
class TrackingWorker(BaseWorker):
```

Rename `run(self)` → `execute(self)`. Remove any duplicate `finished = Signal()` found in step 1. Remove any final `self.finished.emit()` call and any bare try/except error re-emit wrapper in `run()`.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/tracking/worker.py
git commit -m "refactor: migrate TrackingWorker to BaseWorker"
```

---

### Task 7: Migrate core/tracking/optimizer_workers.py

**Files:**
- Modify: `src/hydra_suite/core/tracking/optimizer_workers.py`

Two workers at:
- `DetectionCacheBuilderWorker` (line 40)
- `TrackingPreviewWorker` (line 205)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate DetectionCacheBuilderWorker (line 40)**

```python
# Before:
class DetectionCacheBuilderWorker(QThread):

# After:
class DetectionCacheBuilderWorker(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 3: Migrate TrackingPreviewWorker (line 205)**

```python
# Before:
class TrackingPreviewWorker(QThread):

# After:
class TrackingPreviewWorker(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/tracking/optimizer_workers.py
git commit -m "refactor: migrate optimizer workers to BaseWorker"
```

---

### Task 8: Migrate core/tracking/optimizer.py TrackingOptimizer

**Files:**
- Modify: `src/hydra_suite/core/tracking/optimizer.py` (line 71)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate**

```python
# Before (line 71):
class TrackingOptimizer(QThread):

# After:
class TrackingOptimizer(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/core/tracking/optimizer.py
git commit -m "refactor: migrate TrackingOptimizer to BaseWorker"
```

---

### Task 9: Migrate core/detectors/bg_optimizer.py

**Files:**
- Modify: `src/hydra_suite/core/detectors/bg_optimizer.py`

Two workers:
- `BgSubtractionOptimizer` (line 74)
- `BgDetectionPreviewWorker` (line 578)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate BgSubtractionOptimizer (line 74)**

```python
# Before:
class BgSubtractionOptimizer(QThread):

# After:
class BgSubtractionOptimizer(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 3: Migrate BgDetectionPreviewWorker (line 578)**

```python
# Before:
class BgDetectionPreviewWorker(QThread):

# After:
class BgDetectionPreviewWorker(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/core/detectors/bg_optimizer.py
git commit -m "refactor: migrate bg_optimizer workers to BaseWorker"
```

---

### Task 10: Migrate refinekit _ScorerWorker

**Files:**
- Modify: `src/hydra_suite/refinekit/gui/main_window.py` (line 49)

- [ ] **Step 1: Add import**

```python
from hydra_suite.widgets.workers import BaseWorker
```

- [ ] **Step 2: Migrate**

```python
# Before (line 49):
class _ScorerWorker(QThread):

# After:
class _ScorerWorker(BaseWorker):
```

Rename `run` → `execute`. Remove boilerplate.

- [ ] **Step 3: Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

Expected: same pass/fail as before Task 2.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/refinekit/gui/main_window.py
git commit -m "refactor: migrate refinekit _ScorerWorker to BaseWorker"
```

---

### Task 11: Verify and finalize

- [ ] **Step 1: Confirm no orphan QThread subclasses remain (excluding exclusions)**

```bash
grep -rn "class .*QThread" src/hydra_suite/ --include="*.py"
```

Expected remaining (intentionally excluded):
- `_MultiCropLoader`, `_PrefetchThread`, `_PlaybackThread` — video player internals
- `_FrameLoader` (×2) — dialog-internal frame loaders
- `_TestWorker` — model test dialog internal

Any other result is a missed migration — fix it before proceeding.

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 3: Commit completion marker**

```bash
git commit --allow-empty -m "chore: Slice 1 (worker unification) complete"
```
