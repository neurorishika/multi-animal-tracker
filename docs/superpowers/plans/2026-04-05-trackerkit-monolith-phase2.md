# TrackerKit Monolith Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `trackerkit/gui/main_window.py` from 16,221 lines into a thin ≤600-line coordinator by extracting workers, widget classes, dialog boilerplate, panel handlers, and cross-panel logic into focused modules.

**Architecture:** Workers and widget utility classes move verbatim into new `workers/` and `widgets/` subdirectories. Panel handler methods migrate into their respective panel classes (replacing `self._panel.widget` with `self.widget`). Cross-panel orchestration logic is collected into three plain-Python orchestrator classes (`TrackingOrchestrator`, `ConfigOrchestrator`, `SessionOrchestrator`) that `MainWindow` instantiates and wires. All four raw-`QDialog` dialogs are migrated to `BaseDialog`.

**Tech Stack:** PySide6, Python 3.13, pytest, conda env `hydra-mps`

---

## Verification gate (run after every task)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate hydra-mps

# Smoke test — catches broken imports before running the full suite
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"

# Full suite
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```

Expected baseline: **841 tests collected, 0 failed**.

---

## File Map

**New directories + files:**
```
src/hydra_suite/trackerkit/gui/
    workers/
        __init__.py
        merge_worker.py        # MergeWorker + _write_csv_artifact + _write_roi_npz
        crops_worker.py        # InterpolatedCropsWorker
        video_worker.py        # OrientedTrackVideoWorker
        dataset_worker.py      # DatasetGenerationWorker
        preview_worker.py      # PreviewDetectionWorker + all preview helper fns
    widgets/
        __init__.py
        collapsible.py         # CollapsibleGroupBox + AccordionContainer
        help_label.py          # CompactHelpLabel
        tooltip_button.py      # ImmediateTooltipButton
    orchestrators/
        __init__.py
        tracking.py            # TrackingOrchestrator
        config.py              # ConfigOrchestrator
        session.py             # SessionOrchestrator
tests/
    test_trackerkit_workers_smoke.py   # import smoke tests for all 5 workers
    test_trackerkit_orchestrators_smoke.py  # orchestrator construction smoke tests
```

**Modified files:**
```
src/hydra_suite/trackerkit/gui/main_window.py  (shrinks from 16,221 → ≤600 lines)
src/hydra_suite/trackerkit/gui/panels/*.py      (handlers added; imports updated)
src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py   → BaseDialog
src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py      → BaseDialog
src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py     → BaseDialog + BaseWorker
src/hydra_suite/trackerkit/gui/dialogs/run_history_dialog.py    → BaseDialog
src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py     → BaseDialog
tests/test_main_window_config_persistence.py   (update any method refs to orchestrators)
```

---

## Phase 1 — Mechanical Extraction

### Task 1: Scaffold new directories

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/__init__.py`
- Create: `src/hydra_suite/trackerkit/gui/widgets/__init__.py`
- Create: `src/hydra_suite/trackerkit/gui/orchestrators/__init__.py`

- [ ] **Step 1: Create the three `__init__.py` stubs**

```python
# workers/__init__.py
"""trackerkit background workers — one file per worker class."""

# widgets/__init__.py
"""trackerkit-local widget utilities."""

# orchestrators/__init__.py
"""trackerkit domain orchestrators — cross-panel workflow logic."""
```

Create all three files with the content above.

- [ ] **Step 2: Run the verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
```
Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/__init__.py \
        src/hydra_suite/trackerkit/gui/widgets/__init__.py \
        src/hydra_suite/trackerkit/gui/orchestrators/__init__.py
git commit -m "chore: scaffold workers/, widgets/, orchestrators/ subpackages in trackerkit"
```

---

### Task 2: Extract MergeWorker

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/merge_worker.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py` (remove class + helpers, add import)

- [ ] **Step 1: Write the failing import test**

Create `tests/test_trackerkit_workers_smoke.py`:

```python
"""Smoke tests: all trackerkit workers importable from workers/ subpackage."""

import pytest


def test_merge_worker_importable():
    from hydra_suite.trackerkit.gui.workers.merge_worker import MergeWorker
    assert MergeWorker is not None


def test_crops_worker_importable():
    from hydra_suite.trackerkit.gui.workers.crops_worker import InterpolatedCropsWorker
    assert InterpolatedCropsWorker is not None


def test_video_worker_importable():
    from hydra_suite.trackerkit.gui.workers.video_worker import OrientedTrackVideoWorker
    assert OrientedTrackVideoWorker is not None


def test_dataset_worker_importable():
    from hydra_suite.trackerkit.gui.workers.dataset_worker import DatasetGenerationWorker
    assert DatasetGenerationWorker is not None


def test_preview_worker_importable():
    from hydra_suite.trackerkit.gui.workers.preview_worker import PreviewDetectionWorker
    assert PreviewDetectionWorker is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_merge_worker_importable -v
```
Expected: `FAILED` — `ModuleNotFoundError: No module named 'hydra_suite.trackerkit.gui.workers.merge_worker'`

- [ ] **Step 3: Create `workers/merge_worker.py`**

Cut lines 504–543 (`_write_csv_artifact`, `_write_roi_npz`) and lines 306–503 (`MergeWorker`) from `main_window.py` verbatim. The new file needs all imports that those lines use. Create `src/hydra_suite/trackerkit/gui/workers/merge_worker.py`:

```python
"""MergeWorker — trajectory merge and CSV export background worker."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Signal

from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    process_trajectories,
    relink_trajectories_with_pose,
    resolve_trajectories,
)
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


def _write_csv_artifact(path, fieldnames, rows):
    # [paste lines 504–515 from main_window.py verbatim]
    ...


def _write_roi_npz(path, roi_rows, roi_corners):
    # [paste lines 516–543 from main_window.py verbatim]
    ...


class MergeWorker(BaseWorker):
    # [paste lines 306–503 from main_window.py verbatim]
    ...
```

**Paste the actual code from `main_window.py` lines 306–543** (do not paraphrase — copy verbatim). Verify the top-level imports in the new file cover everything those lines reference.

- [ ] **Step 4: Update `main_window.py`**

At the top of `main_window.py`, add after the `BaseWorker` import:

```python
from .workers.merge_worker import MergeWorker, _write_csv_artifact, _write_roi_npz
```

Delete lines 306–543 from `main_window.py` (the original `MergeWorker` class and the two helper functions).

- [ ] **Step 5: Run the smoke test**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_merge_worker_importable -v
```
Expected: `PASSED`

- [ ] **Step 6: Run the verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `import ok` + `841 passed`

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/merge_worker.py \
        src/hydra_suite/trackerkit/gui/main_window.py \
        tests/test_trackerkit_workers_smoke.py
git commit -m "refactor: extract MergeWorker to trackerkit/gui/workers/merge_worker.py"
```

---

### Task 3: Extract InterpolatedCropsWorker

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/crops_worker.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

> **Note:** All line numbers for `main_window.py` shift after each extraction. Use `grep -n "class InterpolatedCropsWorker" src/hydra_suite/trackerkit/gui/main_window.py` to find the current line before cutting.

- [ ] **Step 1: Confirm test already exists and fails**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_crops_worker_importable -v
```
Expected: `FAILED`

- [ ] **Step 2: Create `workers/crops_worker.py`**

Find the current line range with `grep -n "class InterpolatedCropsWorker\|class OrientedTrackVideoWorker" src/hydra_suite/trackerkit/gui/main_window.py`. Cut from `InterpolatedCropsWorker` to just before `OrientedTrackVideoWorker` verbatim. Create `src/hydra_suite/trackerkit/gui/workers/crops_worker.py`:

```python
"""InterpolatedCropsWorker — per-animal interpolated crop export worker."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class InterpolatedCropsWorker(BaseWorker):
    # [paste lines 544–1684 from main_window.py verbatim]
    ...
```

Verify that all imports used inside `InterpolatedCropsWorker` are present at the top of the new file (scan for any `from hydra_suite...` or `import` inside the class body and hoist to module level).

- [ ] **Step 3: Update `main_window.py`**

Add after the `MergeWorker` import line:

```python
from .workers.crops_worker import InterpolatedCropsWorker
```

Delete lines 544–1684 from `main_window.py`.

- [ ] **Step 4: Run smoke test + verification gate**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_crops_worker_importable -v
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/crops_worker.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract InterpolatedCropsWorker to trackerkit/gui/workers/crops_worker.py"
```

---

### Task 4: Extract OrientedTrackVideoWorker + DatasetGenerationWorker

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/video_worker.py`
- Create: `src/hydra_suite/trackerkit/gui/workers/dataset_worker.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

These two workers are short (63 and 165 lines respectively after the crops worker has been removed). Extract both in one commit.

- [ ] **Step 1: Create `workers/video_worker.py`**

Cut the `OrientedTrackVideoWorker` class (currently at approximately lines 1685–1747 after prior extractions, adjust for actual current line numbers) verbatim. Create:

```python
"""OrientedTrackVideoWorker — per-track orientation-corrected video export."""

from __future__ import annotations

import logging

from PySide6.QtCore import Signal

from hydra_suite.core.identity.dataset.oriented_video import OrientedTrackVideoExporter
from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class OrientedTrackVideoWorker(BaseWorker):
    # [paste OrientedTrackVideoWorker class verbatim]
    ...
```

- [ ] **Step 2: Create `workers/dataset_worker.py`**

Cut the `DatasetGenerationWorker` class verbatim. Create:

```python
"""DatasetGenerationWorker — active-learning dataset export worker."""

from __future__ import annotations

import logging

from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)


class DatasetGenerationWorker(BaseWorker):
    # [paste DatasetGenerationWorker class verbatim]
    ...
```

Note: `DatasetGenerationWorker.execute()` has deferred imports inside the method body (`from hydra_suite.data.dataset_generation import ...`) — leave those in place.

- [ ] **Step 3: Update `main_window.py`**

Add imports:

```python
from .workers.video_worker import OrientedTrackVideoWorker
from .workers.dataset_worker import DatasetGenerationWorker
```

Delete both class definitions from `main_window.py`.

- [ ] **Step 4: Run smoke tests + verification gate**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_video_worker_importable \
                 tests/test_trackerkit_workers_smoke.py::test_dataset_worker_importable -v
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/video_worker.py \
        src/hydra_suite/trackerkit/gui/workers/dataset_worker.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract OrientedTrackVideoWorker and DatasetGenerationWorker to workers/"
```

---

### Task 5: Extract PreviewDetectionWorker + helper functions

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

This task moves two blocks: the preview background helper functions near the top of the file (originally lines 140–305 before prior extractions) and `PreviewDetectionWorker` + `_run_preview_detection_job` + the preview drawing helpers (`_normalize_preview_model_names` through `_draw_preview_pose_points`, originally around lines 1913–2829).

- [ ] **Step 1: Confirm test fails**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py::test_preview_worker_importable -v
```
Expected: `FAILED`

- [ ] **Step 2: Create `workers/preview_worker.py`**

Create `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`. Move ALL of the following from `main_window.py` verbatim:
- Module-level constants: `_PREVIEW_BACKGROUND_CACHE_MAX_ENTRIES`, `_PREVIEW_BACKGROUND_CACHE`, `_PREVIEW_BACKGROUND_CACHE_LOCK`
- Functions: `_clear_preview_background_cache`, `_hash_preview_roi_mask`, `_preview_background_cache_key`, `_build_preview_background_params`, `_get_cached_preview_background_state`, `_store_preview_background_state`, `_build_preview_background_model`
- Functions: `_normalize_preview_model_names`, `_preview_class_label`, `_preview_label_anchor`, `_draw_preview_label_stack`, `_draw_preview_pose_points`, `_run_preview_detection_job`
- Class: `PreviewDetectionWorker`

```python
"""PreviewDetectionWorker — non-blocking single-frame detection preview."""

from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import OrderedDict

import cv2
import numpy as np
from PySide6.QtCore import Signal

from hydra_suite.widgets.workers import BaseWorker

logger = logging.getLogger(__name__)

_PREVIEW_BACKGROUND_CACHE_MAX_ENTRIES = 4
_PREVIEW_BACKGROUND_CACHE: OrderedDict = OrderedDict()
_PREVIEW_BACKGROUND_CACHE_LOCK = threading.Lock()


# [paste _clear_preview_background_cache through _build_preview_background_model verbatim]

# [paste _normalize_preview_model_names through _run_preview_detection_job verbatim]


class PreviewDetectionWorker(BaseWorker):
    # [paste PreviewDetectionWorker verbatim]
    ...
```

- [ ] **Step 3: Update `main_window.py`**

Remove the moved constants and functions from `main_window.py`. Remove `threading` from imports if it is no longer used elsewhere in `main_window.py` (grep to confirm). Add:

```python
from .workers.preview_worker import (
    PreviewDetectionWorker,
    _clear_preview_background_cache,
    _run_preview_detection_job,
)
```

- [ ] **Step 4: Run smoke test + verification gate**

```bash
python -m pytest tests/test_trackerkit_workers_smoke.py -v
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: all 5 worker smoke tests pass + 841 suite tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/workers/preview_worker.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract PreviewDetectionWorker and preview helpers to workers/preview_worker.py"
```

**Line count check:** `wc -l src/hydra_suite/trackerkit/gui/main_window.py` — expect ≤ 14,500.

---

### Task 6: Extract widget utility classes

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/widgets/collapsible.py`
- Create: `src/hydra_suite/trackerkit/gui/widgets/help_label.py`
- Create: `src/hydra_suite/trackerkit/gui/widgets/tooltip_button.py`
- Create: `src/hydra_suite/trackerkit/gui/widgets/stacked_page.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/*.py` (update imports)

- [ ] **Step 1: Create `widgets/collapsible.py`**

Cut `CollapsibleGroupBox` (currently ~line 3119) and `AccordionContainer` (~line 3274) from `main_window.py` verbatim. Create:

```python
"""CollapsibleGroupBox and AccordionContainer — expandable section widgets."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class CollapsibleGroupBox(QWidget):
    # [paste verbatim from main_window.py]
    ...


class AccordionContainer:
    # [paste verbatim from main_window.py]
    ...
```

- [ ] **Step 2: Create `widgets/help_label.py`**

Cut `CompactHelpLabel` from `main_window.py` verbatim:

```python
"""CompactHelpLabel — inline help text widget that attaches to group box titles."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget


class CompactHelpLabel(QWidget):
    # [paste verbatim from main_window.py]
    ...
```

- [ ] **Step 3: Create `widgets/tooltip_button.py`**

Cut `ImmediateTooltipButton` from `main_window.py` verbatim:

```python
"""ImmediateTooltipButton — tool button that shows tooltip instantly on hover."""

from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QToolButton, QToolTip


class ImmediateTooltipButton(QToolButton):
    # [paste verbatim from main_window.py]
    ...
```

- [ ] **Step 3b: Create `widgets/stacked_page.py`**

Cut `CurrentPageStackedWidget` from `main_window.py` verbatim:

```python
"""CurrentPageStackedWidget — stacked widget whose size hint tracks the active page."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QStackedWidget


class CurrentPageStackedWidget(QStackedWidget):
    # [paste verbatim from main_window.py]
    ...
```

- [ ] **Step 4: Update imports in `main_window.py`**

Add to `main_window.py`:

```python
from .widgets.collapsible import AccordionContainer, CollapsibleGroupBox
from .widgets.help_label import CompactHelpLabel
from .widgets.stacked_page import CurrentPageStackedWidget
from .widgets.tooltip_button import ImmediateTooltipButton
```

Delete the four class bodies from `main_window.py`.

- [ ] **Step 5: Update imports in panel files**

Each panel file currently imports these classes from `main_window`. Update all six panel files:

```python
# OLD (in panels/*.py):
from hydra_suite.trackerkit.gui.main_window import (
    CollapsibleGroupBox,
    AccordionContainer,
    CompactHelpLabel,
)

# NEW:
from hydra_suite.trackerkit.gui.widgets.collapsible import (
    AccordionContainer,
    CollapsibleGroupBox,
)
from hydra_suite.trackerkit.gui.widgets.help_label import CompactHelpLabel
from hydra_suite.trackerkit.gui.widgets.tooltip_button import ImmediateTooltipButton
```

Run `grep -rn "from hydra_suite.trackerkit.gui.main_window import" src/hydra_suite/trackerkit/` to find all files that need updating.

- [ ] **Step 6: Run verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `import ok` + 841 passed.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/widgets/ \
        src/hydra_suite/trackerkit/gui/main_window.py \
        src/hydra_suite/trackerkit/gui/panels/
git commit -m "refactor: extract CollapsibleGroupBox, CompactHelpLabel, ImmediateTooltipButton to trackerkit/gui/widgets/"
```

**Line count check:** `wc -l src/hydra_suite/trackerkit/gui/main_window.py` — expect ≤ 13,900.

---

## Phase 2 — Dialog BaseDialog Migration

### Task 7: Migrate `BgParameterHelperDialog` to `BaseDialog`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py`

`BgParameterHelperDialog` currently inherits `QDialog` and manually sets up its root layout in `__init__`. `BaseDialog` does that automatically and provides `add_content()`.

- [ ] **Step 1: Read the current `__init__` of `BgParameterHelperDialog`**

Current structure (lines 91–118 in the file):
```python
class BgParameterHelperDialog(QDialog):
    def __init__(self, video_path, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Auto-Tuner  (Background Subtraction)")
        self.setMinimumSize(1200, 640)
        # ... state setup ...
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)   # ← this becomes add_content(...)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)
        # ... widgets ...
```

- [ ] **Step 2: Apply the migration**

In `bg_parameter_helper.py`:

1. Replace `from PySide6.QtWidgets import QDialog, ...` with `from hydra_suite.widgets.dialogs import BaseDialog` (keep all other Qt imports, remove `QDialog`).

2. Change the class line:
```python
# OLD:
class BgParameterHelperDialog(QDialog):
# NEW:
class BgParameterHelperDialog(BaseDialog):
```

3. Change `__init__` to call `BaseDialog.__init__` with the title:
```python
def __init__(self, video_path, current_params, parent=None):
    super().__init__(
        title="Detection Auto-Tuner  (Background Subtraction)",
        parent=parent,
        buttons=QDialogButtonBox.NoButton,   # dialog has its own Apply/Cancel bar
        apply_dark_style=True,
    )
    self.setMinimumSize(1200, 640)
    # ... remaining state setup unchanged ...
    self._build_ui()
```

4. In `_build_ui`, replace `root = QVBoxLayout(self)` with a local `QWidget` container and pass it to `add_content`:
```python
def _build_ui(self):
    container = QWidget()
    root = QVBoxLayout(container)
    root.setSpacing(0)
    root.setContentsMargins(0, 0, 0, 0)
    # ... rest of _build_ui unchanged, all references to `root` stay ...
    self.add_content(container)
```

Note: `BaseDialog` adds a button box automatically. Since `BgParameterHelperDialog` has its own Apply/Cancel bar inside its content, pass `buttons=QDialogButtonBox.NoButton` to suppress the default button box.

- [ ] **Step 3: Run verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.dialogs.bg_parameter_helper import BgParameterHelperDialog; print('ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `ok` + 841 passed.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py
git commit -m "refactor: migrate BgParameterHelperDialog to BaseDialog"
```

---

### Task 8: Migrate `ParameterHelperDialog` to `BaseDialog`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py`

Same pattern as Task 7.

- [ ] **Step 1: Apply the migration**

In `parameter_helper.py`:

1. Add import: `from hydra_suite.widgets.dialogs import BaseDialog`
2. Change class: `class ParameterHelperDialog(BaseDialog):`
3. Change `__init__`:
```python
def __init__(self, video_path, detection_cache_path, start_frame, end_frame,
             current_params, parent=None):
    super().__init__(
        title="Tracking Auto-Tuner",
        parent=parent,
        buttons=QDialogButtonBox.NoButton,
        apply_dark_style=True,
    )
    self.video_path = video_path
    self.detection_cache_path = detection_cache_path
    self.start_frame = start_frame
    self.end_frame = end_frame
    self.base_params = current_params.copy()
    self.results = []
    self.optimizer = None
    self.preview_worker = None
    self._build_ui()
```

4. In `_build_ui`, replace the root `QVBoxLayout(self)` with a container widget passed to `add_content`:
```python
def _build_ui(self):
    container = QWidget()
    root = QVBoxLayout(container)
    # ... rest of _build_ui unchanged ...
    self.add_content(container)
```

- [ ] **Step 2: Run verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.dialogs.parameter_helper import ParameterHelperDialog; print('ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `ok` + 841 passed.

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py
git commit -m "refactor: migrate ParameterHelperDialog to BaseDialog"
```

---

### Task 9: Migrate `_TestWorker` to `BaseWorker` + `ModelTestDialog` to `BaseDialog`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py`

Two changes in one file: `_TestWorker(QThread)` → `BaseWorker`, `ModelTestDialog(QDialog)` → `BaseDialog`.

- [ ] **Step 1: Migrate `_TestWorker`**

In `model_test_dialog.py`:

1. Replace `from PySide6.QtCore import Qt, QThread, Signal` with:
```python
from PySide6.QtCore import Qt, Signal
from hydra_suite.widgets.workers import BaseWorker
```

2. Change the class:
```python
# OLD:
class _TestWorker(QThread):
    image_ready = Signal(np.ndarray)
    status = Signal(str)
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, params, image_paths):
        super().__init__()
        self.params = params
        self.image_paths = image_paths

    def run(self):
        try:
            # ... inference logic ...
        except Exception as exc:
            logger.exception("Quick test inference failed")
            self.error.emit(str(exc))
        finally:
            self.finished_all.emit()

# NEW:
class _TestWorker(BaseWorker):
    image_ready = Signal(np.ndarray)
    finished_all = Signal()

    def __init__(self, params, image_paths):
        super().__init__()
        self.params = params
        self.image_paths = image_paths

    def execute(self):
        # Move the body of run() here (the try block content only — no try/except/finally)
        # BaseWorker.run() handles the exception → error signal
        # Emit finished_all at the end:
        from hydra_suite.core.detectors import YOLOOBBDetector
        self.status.emit("Loading model...")
        detector = YOLOOBBDetector(self.params)
        for idx, img_path in enumerate(self.image_paths):
            self.status.emit(f"Running inference on image {idx + 1}/{len(self.image_paths)}...")
            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            result = detector.detect_objects(frame, frame_count=idx, return_raw=True)
            obb_corners = result[5] if len(result) > 5 else []
            annotated = frame.copy()
            for corners in obb_corners:
                pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], isClosed=True,
                              color=_OBB_COLOR, thickness=_OBB_THICKNESS)
            self.image_ready.emit(annotated)
        self.finished_all.emit()
```

Note: `BaseWorker` provides `error` and `status` signals already — remove the `error = Signal(str)` and `status = Signal(str)` redefinitions in `_TestWorker`.

- [ ] **Step 2: Migrate `ModelTestDialog`**

In the same file:

1. Add import: `from hydra_suite.widgets.dialogs import BaseDialog`
2. Change class: `class ModelTestDialog(BaseDialog):`
3. In `__init__`, call `BaseDialog.__init__` with appropriate args, then call `_build_ui`:
```python
def __init__(self, model_path, role, dataset_dir, device="cpu", imgsz=640,
             crop_pad_ratio=0.15, min_crop_size_px=64, enforce_square=True,
             detect_model_path="", parent=None):
    super().__init__(
        title="Quick Model Test",
        parent=parent,
        buttons=QDialogButtonBox.Close,
        apply_dark_style=False,
    )
    self.resize(900, 600)
    self._model_path = model_path
    self._role = role
    self._dataset_dir = dataset_dir
    self._device = device
    self._imgsz = imgsz
    self._crop_pad_ratio = crop_pad_ratio
    self._min_crop_size_px = min_crop_size_px
    self._enforce_square = enforce_square
    self._detect_model_path = detect_model_path
    self._worker = None
    self._build_ui()
    self._run_test()
```
4. In `_build_ui`, wrap the layout in a container widget and call `self.add_content(container)`.

- [ ] **Step 3: Run verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import ModelTestDialog; print('ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `ok` + 841 passed.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py
git commit -m "refactor: migrate _TestWorker to BaseWorker and ModelTestDialog to BaseDialog"
```

---

### Task 10: Migrate `RunHistoryDialog` + `TrainYoloDialog` to `BaseDialog`

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/run_history_dialog.py`
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py`

Both follow the same pattern. Do both in one commit.

- [ ] **Step 1: Migrate `RunHistoryDialog`**

In `run_history_dialog.py`:

1. Add: `from hydra_suite.widgets.dialogs import BaseDialog`
2. Change class: `class RunHistoryDialog(BaseDialog):`
3. Change `__init__`:
```python
def __init__(self, parent=None):
    super().__init__(
        title="Training Run History",
        parent=parent,
        buttons=QDialogButtonBox.Close,
        apply_dark_style=True,
    )
    self.resize(900, 520)
    from hydra_suite.training.registry import get_registry_path
    registry_path = str(get_registry_path())
    self._runs = list(reversed(load_run_history(registry_path)))
    self._build_ui()
```
4. In `_build_ui`, wrap content in container widget and call `self.add_content(container)`.

- [ ] **Step 2: Migrate `TrainYoloDialog`**

In `train_yolo_dialog.py`:

1. Add: `from hydra_suite.widgets.dialogs import BaseDialog`
2. Change class: `class TrainYoloDialog(BaseDialog):`
3. In `__init__`, replace manual `setWindowTitle`, `setModal`, layout setup with `super().__init__(title="Train YOLO Model", parent=parent, buttons=QDialogButtonBox.NoButton, apply_dark_style=True)`.
4. In `_build_ui`, wrap content in container widget and call `self.add_content(container)`.

- [ ] **Step 3: Run verification gate**

```bash
python -c "
from hydra_suite.trackerkit.gui.dialogs.run_history_dialog import RunHistoryDialog
from hydra_suite.trackerkit.gui.dialogs.train_yolo_dialog import TrainYoloDialog
print('ok')
"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `ok` + 841 passed.

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/run_history_dialog.py \
        src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py
git commit -m "refactor: migrate RunHistoryDialog and TrainYoloDialog to BaseDialog"
```

---

## Phase 3 — Panel Handler Migration

> **Note on `_panels_bundle()`:** This helper is created in Task 17. During Tasks 11–16 (panel handler migration), panels still call `self._main_window.method_name()` for cross-cutting work — `_panels_bundle` and the orchestrators do not exist yet. This is intentional: orchestrators are assembled from whatever remains in `MainWindow` after panel migration completes.

**Handler migration pattern** (same for all 6 panels):

When a method moves from `MainWindow` to a panel:
- `self._panel_name.some_widget` → `self.some_widget` (panel owns its own widgets)
- `self.some_main_window_method()` → `self._main_window.some_main_window_method()` (cross-cutting calls stay on MainWindow for now)
- Signal connections in `_build_ui()`: change `self._main_window.handler_name` → `self.handler_name`

**After each panel task**, run the full verification gate before starting the next.

---

### Task 11: `DatasetPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving from `MainWindow` to `DatasetPanel` (grep for their line numbers first):
- `_on_dataset_generation_toggled`
- `_on_xanylabeling_env_changed`
- `_open_in_xanylabeling`
- `_open_pose_label_ui`
- `_refresh_xanylabeling_envs`
- `_selected_xanylabeling_env`
- `_on_individual_dataset_toggled`

- [ ] **Step 1: Move methods to `DatasetPanel`**

Cut each method body from `MainWindow` and paste into `DatasetPanel`, applying the transformation pattern:

```python
# In DatasetPanel — example transformation of _on_dataset_generation_toggled:
def _on_dataset_generation_toggled(self, enabled):
    """Enable/disable dataset generation controls."""
    self.active_learning_content.setVisible(enabled)  # was: self._dataset_panel.active_learning_content

# In DatasetPanel — example transformation of _on_xanylabeling_env_changed:
def _on_xanylabeling_env_changed(self, _text: str) -> None:
    env_name = self._selected_xanylabeling_env()
    if not env_name:
        return
    if self._main_window.advanced_config.get("xanylabeling_env") == env_name:
        return
    self._main_window.advanced_config["xanylabeling_env"] = env_name
    self._main_window._save_advanced_config()  # cross-panel call stays on MainWindow
```

- [ ] **Step 2: Update signal connections in `DatasetPanel._build_ui`**

Find all `.connect(self._main_window.method_name)` calls in `_build_ui()` for the moved methods and change to `.connect(self.method_name)`. Example:

```python
# OLD (in _build_ui):
self.chk_enable_dataset_gen.toggled.connect(self._main_window._on_dataset_generation_toggled)
self.combo_xanylabeling_env.currentTextChanged.connect(self._main_window._on_xanylabeling_env_changed)

# NEW:
self.chk_enable_dataset_gen.toggled.connect(self._on_dataset_generation_toggled)
self.combo_xanylabeling_env.currentTextChanged.connect(self._on_xanylabeling_env_changed)
```

- [ ] **Step 3: Remove the stubs from `MainWindow`**

Delete the method bodies from `MainWindow`. If `MainWindow` still calls one of them (e.g., `self._on_dataset_generation_toggled(...)` somewhere in the config loading path), replace with `self._dataset_panel._on_dataset_generation_toggled(...)`.

- [ ] **Step 4: Run verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: 841 passed.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/dataset_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate DatasetPanel handler methods out of MainWindow"
```

---

### Task 12: `SetupPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving from `MainWindow` to `SetupPanel`:
- `_on_preset_selection_changed`
- `_add_videos_to_batch`
- `_clear_batch`
- `_detect_fps_from_current_video`
- `_refresh_compute_runtime_options`
- `_on_compute_runtime_changed`
- `_on_runtime_context_changed` (if specific to setup)
- Any other method whose body only touches `self._setup_panel.*` widgets

Cross-cutting calls (`_load_selected_preset`, `_save_custom_preset`, `_populate_preset_combo`) stay in `MainWindow` for now — they will move to `ConfigOrchestrator` in Task 19. Panel signal connections for those should remain `self._main_window.method_name`.

Apply the same transformation pattern as Task 11. Update signal connections in `SetupPanel._build_ui` for all moved methods.

- [ ] **Step 1: Move + transform each method (same pattern as Task 11)**

Example:
```python
# In SetupPanel:
def _add_videos_to_batch(self):
    from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog
    paths, _ = QFileDialog.getOpenFileNames(
        self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
    )
    for path in paths:
        if path and path not in [
            self.lst_batch_videos.item(i).text()
            for i in range(self.lst_batch_videos.count())
        ]:
            self.lst_batch_videos.addItem(path)
```

- [ ] **Step 2: Run verification gate + commit**

```bash
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
git add src/hydra_suite/trackerkit/gui/panels/setup_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate SetupPanel handler methods out of MainWindow"
```

---

### Task 13: `PostProcessPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving to `PostProcessPanel`:
- `_on_cleaning_toggled`
- `_on_video_output_toggled`
- `_select_video_pose_color`
- `select_video_output`
- Any other method whose body only touches `self._postprocess_panel.*`

Example transformation of `_on_cleaning_toggled`:
```python
# In PostProcessPanel — all self._postprocess_panel.X become self.X:
def _on_cleaning_toggled(self, state):
    enabled = self.enable_postprocessing.isChecked()
    self.spin_min_trajectory_length.setVisible(enabled)
    self.lbl_min_trajectory_length.setVisible(enabled)
    # ... rest verbatim, replacing self._postprocess_panel. → self. ...
```

- [ ] **Step 1: Move + transform, update signal connections, run gate, commit**

```bash
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
git add src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate PostProcessPanel handler methods out of MainWindow"
```

---

### Task 14: `TrackingPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving to `TrackingPanel`:
- `_on_confidence_density_map_toggled`
- `_on_visualization_mode_changed`
- `_on_parameter_changed`
- `_draw_roi_overlay`
- `_apply_roi_mask_to_image`
- `_is_visualization_enabled`
- `_is_pose_export_enabled`
- Any other method whose body only touches `self._tracking_panel.*`

Example of `_on_confidence_density_map_toggled` after move:
```python
# In TrackingPanel:
def _on_confidence_density_map_toggled(self, state):
    enabled = self.chk_enable_confidence_density_map.isChecked()
    self.g_density.setVisible(enabled)
    self.g_density.setEnabled(enabled)
```

Note: the `hasattr(self, "_tracking_panel")` guard from the original code disappears — the method is now on the panel itself and `self` always exists.

- [ ] **Step 1: Move + transform, update signal connections, run gate, commit**

```bash
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
git add src/hydra_suite/trackerkit/gui/panels/tracking_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate TrackingPanel handler methods out of MainWindow"
```

---

### Task 15: `IdentityPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving to `IdentityPanel` (largest panel handler surface, ~55 methods):
- `_on_identity_method_changed`
- `_on_identity_analysis_toggled`
- `_on_individual_analysis_toggled`
- `_on_pose_analysis_toggled`
- `_add_cnn_classifier_row`
- `_remove_cnn_classifier_row`
- `_cnn_classifier_rows`
- `_refresh_cnn_identity_model_combo`
- `_on_cnn_identity_model_selected`
- `_update_cnn_identity_verification_panel`
- `_handle_add_new_cnn_identity_model`
- `_select_color_tag_model`
- `_sync_identity_method_ui`
- `_sync_individual_analysis_mode_ui`
- `_refresh_pose_direction_keypoint_lists`
- `_refresh_pose_sleap_envs`
- `_refresh_yolo_headtail_model_combo`
- `_get_selected_yolo_headtail_model_path`
- `_get_apriltag_families`
- Any other method whose body primarily touches `self._identity_panel.*`

For methods that call `self._refresh_yolo_detect_model_combo()` or other detection-panel helpers, replace with `self._main_window._refresh_yolo_detect_model_combo()`.

- [ ] **Step 1: Move + transform, update signal connections, run gate**

After moving, run:
```bash
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: 841 passed. Note: `test_main_window_config_persistence.py` calls `window._refresh_yolo_headtail_model_combo()` — update to `window._identity_panel._refresh_yolo_headtail_model_combo()` if the test breaks.

- [ ] **Step 2: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/identity_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py \
        tests/test_main_window_config_persistence.py
git commit -m "refactor: migrate IdentityPanel handler methods out of MainWindow"
```

---

### Task 16: `DetectionPanel` handler migration

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

Methods moving to `DetectionPanel` (largest single migration, ~45 methods):
- `_on_brightness_changed`
- `_on_contrast_changed`
- `_on_gamma_changed`
- `_on_zoom_changed`
- `_on_yolo_mode_changed`
- `_on_tensorrt_toggled`
- `_on_detection_method_changed_ui`
- `on_detection_method_changed`
- `_update_body_size_info`
- `on_yolo_model_changed`
- `_refresh_yolo_detect_model_combo`
- `_refresh_yolo_crop_obb_model_combo`
- `_refresh_yolo_model_combo`
- `_test_detection_on_preview`
- `_redisplay_detection_test`
- `_collect_preview_detection_context`
- `_on_preview_detection_finished`
- `_on_preview_detection_error`
- `_on_preview_detection_worker_finished`
- `_update_preview_display`
- `_update_detection_stats`
- `_is_yolo_detection_mode`
- `_is_identity_analysis_enabled`
- `_selected_identity_method`
- `_identity_config`
- Any other method whose body primarily touches `self._detection_panel.*`

For methods that call `self._update_preview_display()` within DetectionPanel, these become `self._update_preview_display()` (no change, since they're now on the same object).

Example transformation of `_on_brightness_changed`:
```python
# In DetectionPanel:
def _on_brightness_changed(self, value):
    self.label_brightness_val.setText(str(value))  # was: self._detection_panel.label_brightness_val
    self.detection_test_result = None
    self._update_preview_display()  # now on same object — no change needed
```

- [ ] **Step 1: Move + transform, update signal connections, run gate**

```bash
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: 841 passed.

- [ ] **Step 2: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: migrate DetectionPanel handler methods out of MainWindow"
```

**Line count check after Task 16:** `wc -l src/hydra_suite/trackerkit/gui/main_window.py` — expect ≤ 5,000.

---

## Phase 4 — Orchestrators + Thinning

### Task 17: Create `TrackingOrchestrator`

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Create: `tests/test_trackerkit_orchestrators_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_trackerkit_orchestrators_smoke.py`:

```python
"""Smoke tests: trackerkit orchestrators are constructible."""
import pytest


def test_tracking_orchestrator_constructed(main_window):
    assert main_window._tracking_orch is not None


def test_config_orchestrator_constructed(main_window):
    assert main_window._config_orch is not None


def test_session_orchestrator_constructed(main_window):
    assert main_window._session_orch is not None
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py::test_tracking_orchestrator_constructed -v
```
Expected: `FAILED` — `AttributeError: 'MainWindow' object has no attribute '_tracking_orch'`

- [ ] **Step 3: Create `orchestrators/tracking.py`**

Create `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`. This class receives the methods that remain in `MainWindow` after panel migration and that relate to the tracking lifecycle:

```python
"""TrackingOrchestrator — run→merge→export→finalize lifecycle."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class TrackingOrchestrator:
    """Owns the tracking lifecycle: start, stop, merge, export, finalize."""

    def __init__(self, main_window: "MainWindow", config: "TrackerConfig", panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
```

Then cut the following methods from `MainWindow` and paste them into `TrackingOrchestrator`, replacing every `self.` that references a panel widget with `self._panels.<panel_name>.<widget>` and every `self.` that references a cross-panel utility with `self._mw.`:

Methods to move:
- `start_tracking`
- `start_backward_tracking`
- `start_full`
- `start_preview_on_video`
- `start_tracking_on_video`
- `stop_tracking`
- `_request_qthread_stop`
- `_stop_csv_writer`
- `on_tracking_finished`
- `_finish_tracking_session`
- `_finalize_tracking_session_ui`
- `on_tracking_warning`
- `merge_and_save_trajectories`
- `on_merge_finished`
- `on_merge_error`
- `on_merge_progress`
- `_generate_video_from_trajectories`
- `_generate_interpolated_individual_crops`
- `_start_pending_oriented_track_video_export`
- `_on_oriented_track_videos_finished`
- `_on_oriented_track_video_worker_thread_finished`
- `_on_oriented_track_videos_error`
- `_on_interpolated_crops_finished`
- `_store_interpolated_pose_result`
- `_store_interpolated_cnn_result`
- `_store_interpolated_tag_result`
- `_store_interpolated_headtail_result`
- `save_trajectories_to_csv`
- `_scale_trajectories_to_original_space`
- `on_new_frame`
- `on_stats_update`
- `on_progress_update`
- `on_pose_exported_model_resolved`
- `_load_video_trajectories`
- `_run_pending_video_generation_or_finalize`
- `_is_pose_export_enabled` (if not already in DetectionPanel)
- `_build_pose_augmented_dataframe`
- `_export_pose_augmented_csv`
- `_relink_final_pose_augmented_csv`
- `_generate_training_dataset`
- `on_dataset_progress`
- `on_dataset_finished`
- `on_dataset_error`
- `_on_dataset_worker_thread_finished`
- `_cleanup_thread_reference`
- `_show_session_summary`
- `show_gpu_info`

- [ ] **Step 4: Wire `_tracking_orch` in `MainWindow.__init__`**

In `MainWindow.__init__`, after panel construction, add:

```python
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator
self._tracking_orch = TrackingOrchestrator(
    main_window=self,
    config=self.config,
    panels=self._panels_bundle(),   # see Step 5
)
```

- [ ] **Step 5: Add `_panels_bundle()` helper to `MainWindow`**

```python
def _panels_bundle(self):
    """Return a simple namespace of all panels for orchestrator access."""
    import types
    ns = types.SimpleNamespace()
    ns.setup = self._setup_panel
    ns.detection = self._detection_panel
    ns.tracking = self._tracking_panel
    ns.postprocess = self._postprocess_panel
    ns.dataset = self._dataset_panel
    ns.identity = self._identity_panel
    return ns
```

- [ ] **Step 6: Update any remaining `self.method_name()` calls in `MainWindow` that now live on the orchestrator**

Search `main_window.py` for calls to moved methods:
```bash
grep -n "self\.start_tracking\|self\.stop_tracking\|self\.merge_and_save\|self\.on_tracking_finished" \
     src/hydra_suite/trackerkit/gui/main_window.py
```
Replace `self.method_name(...)` with `self._tracking_orch.method_name(...)`.

- [ ] **Step 7: Run the smoke test**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py::test_tracking_orchestrator_constructed -v
```
Expected: `PASSED`

- [ ] **Step 8: Run the verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: 841 passed.

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/tracking.py \
        src/hydra_suite/trackerkit/gui/main_window.py \
        tests/test_trackerkit_orchestrators_smoke.py
git commit -m "refactor: extract TrackingOrchestrator from MainWindow"
```

---

### Task 18: Create `ConfigOrchestrator`

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/orchestrators/config.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_main_window_config_persistence.py`

- [ ] **Step 1: Run existing smoke test to confirm failure**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py::test_config_orchestrator_constructed -v
```
Expected: `FAILED`

- [ ] **Step 2: Create `orchestrators/config.py`**

```python
"""ConfigOrchestrator — load/save config, presets, ROI, video setup."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class ConfigOrchestrator:
    """Owns config persistence, presets, ROI management, and video file setup."""

    def __init__(self, main_window: "MainWindow", config: "TrackerConfig", panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
```

Cut and move these methods from `MainWindow` into `ConfigOrchestrator`:
- `load_config`
- `_load_config_from_file`
- `save_config`
- `_resolve_config_save_path`
- `_atomic_json_write`
- `get_parameters_dict`
- `_populate_preset_combo`
- `_populate_compute_runtime_options`
- `_load_selected_preset`
- `_save_custom_preset`
- `_load_default_preset_on_startup`
- `_load_advanced_config`
- `_save_advanced_config`
- `_setup_video_file`
- `_get_ui_settings_path`
- `_load_ui_settings`
- `_queue_ui_state_save`
- `_remember_collapsible_state`
- `_restore_ui_state`
- `_save_ui_settings`
- `crop_video_to_roi`
- `_calculate_roi_bounding_box`
- `_estimate_roi_efficiency`
- `_update_roi_optimization_info`
- `_invalidate_roi_cache`
- `_find_or_plan_optimizer_cache_path`
- `_build_optimizer_detection_cache`
- `_on_optimizer_cache_built`
- `_on_preview_cache_built`
- `_apply_optimized_params`
- `_open_parameter_helper`
- `_open_bg_parameter_helper`
- `_poll_crop_stderr_progress`
- `_load_cropped_video`
- `_handle_crop_success`
- `_handle_crop_failure`
- `_check_crop_completion`
- `_resolve_source_video_fps`
- `_get_presets_dir`
- Model registry functions: `get_video_config_path`, `get_models_directory`, `get_models_root_directory`, `get_yolo_model_repository_directory`, `get_pose_models_directory`, `resolve_pose_model_path`, `make_pose_model_path_relative`, `resolve_model_path`, `make_model_path_relative`, `get_yolo_model_registry_path`, `load_yolo_model_registry`, `save_yolo_model_registry`, `get_yolo_model_metadata`, `register_yolo_model`

- [ ] **Step 3: Wire `_config_orch` in `MainWindow.__init__`**

```python
from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator
self._config_orch = ConfigOrchestrator(
    main_window=self,
    config=self.config,
    panels=self._panels_bundle(),
)
```

- [ ] **Step 4: Update remaining `MainWindow` call sites**

```bash
grep -n "self\.load_config\|self\.save_config\|self\.get_parameters_dict\|self\._setup_video_file" \
     src/hydra_suite/trackerkit/gui/main_window.py
```
Replace each with `self._config_orch.method_name(...)`.

- [ ] **Step 5: Update `test_main_window_config_persistence.py`**

The test calls `window.save_config(...)` and `window._load_config_from_file(...)`. Update:

```python
# OLD:
assert window.save_config(preset_mode=True, preset_path=str(config_path))
reloaded_window._load_config_from_file(str(config_path), preset_mode=True)
window.get_parameters_dict()

# NEW:
assert window._config_orch.save_config(preset_mode=True, preset_path=str(config_path))
reloaded_window._config_orch._load_config_from_file(str(config_path), preset_mode=True)
window._config_orch.get_parameters_dict()
```

Also update `window._selected_xanylabeling_env()` → `window._dataset_panel._selected_xanylabeling_env()` (moved to DatasetPanel in Task 11).

- [ ] **Step 6: Run smoke test + verification gate**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py::test_config_orchestrator_constructed -v
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: 841 passed.

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/config.py \
        src/hydra_suite/trackerkit/gui/main_window.py \
        tests/test_main_window_config_persistence.py
git commit -m "refactor: extract ConfigOrchestrator from MainWindow"
```

---

### Task 19: Create `SessionOrchestrator`

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/orchestrators/session.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

- [ ] **Step 1: Run smoke test to confirm failure**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py::test_session_orchestrator_constructed -v
```
Expected: `FAILED`

- [ ] **Step 2: Create `orchestrators/session.py`**

```python
"""SessionOrchestrator — UI state machine, progress tracking, session logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Owns UI state, progress display, session logging, and cleanup."""

    def __init__(self, main_window: "MainWindow", config: "TrackerConfig", panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
```

Cut and move these methods from `MainWindow` into `SessionOrchestrator`:
- `_apply_ui_state`
- `_set_ui_controls_enabled`
- `_collect_preview_controls`
- `_set_interactive_widgets_enabled`
- `_set_video_interaction_enabled`
- `_sync_contextual_controls`
- `_refresh_progress_visibility`
- `_is_worker_running`
- `_has_active_progress_task`
- `_prepare_tracking_display`
- `_show_video_logo_placeholder`
- `_setup_session_logging`
- `_cleanup_session_logging`
- `_cleanup_temporary_files`
- `_disable_spinbox_wheel_events`
- `_connect_parameter_signals`
- `_on_parameter_changed`
- `_is_pose_export_enabled` (if not already moved)
- `toggle_debug_logging`
- `_open_refinekit`
- `_on_merge_worker_thread_finished` (if present)

- [ ] **Step 3: Wire `_session_orch` in `MainWindow.__init__`**

```python
from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator
self._session_orch = SessionOrchestrator(
    main_window=self,
    config=self.config,
    panels=self._panels_bundle(),
)
```

- [ ] **Step 4: Run all three smoke tests + verification gate**

```bash
python -m pytest tests/test_trackerkit_orchestrators_smoke.py -v
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: all 3 orchestrator smoke tests pass + 841 suite tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/orchestrators/session.py \
        src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract SessionOrchestrator from MainWindow"
```

**Line count check after Task 19:** `wc -l src/hydra_suite/trackerkit/gui/main_window.py` — expect ≤ 1,000.

---

### Task 20: Thin `MainWindow` to coordinator

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

At this point `main_window.py` should contain only:
- Module-level imports
- `MainWindow.__init__` (panel + orchestrator construction)
- `MainWindow.init_ui` (tab layout assembly)
- `MainWindow._make_welcome_page` / `_show_workspace`
- `MainWindow._connect_signals` (pure signal wiring)
- `MainWindow._panels_bundle`
- A small number of Qt event handlers (`closeEvent`, `keyPressEvent`) if any

- [ ] **Step 1: Identify any remaining methods that don't belong**

```bash
grep -n "^    def " src/hydra_suite/trackerkit/gui/main_window.py
```

For each method found: decide if it belongs in a panel (touches one panel's widgets), an orchestrator (cross-cutting logic), or genuinely in `MainWindow` (pure coordination/wiring). Move any stragglers to the appropriate location.

- [ ] **Step 2: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```
Target: ≤ 600 lines.

- [ ] **Step 3: Run final verification gate**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('import ok')"
python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```
Expected: `import ok` + 841 passed.

- [ ] **Step 4: Final commit**

```bash
git add src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: thin MainWindow to coordinator — Phase 2 complete"
```

---

## Completion Checklist

- [ ] `main_window.py` ≤ 600 lines (`wc -l` confirms)
- [ ] No worker class defined in `main_window.py` (`grep "class.*BaseWorker" main_window.py` returns nothing)
- [ ] No widget utility class defined in `main_window.py` (`grep "class.*QWidget" main_window.py` returns nothing)
- [ ] All 4 dialogs inherit `BaseDialog` (`grep "class.*QDialog" dialogs/*.py` returns nothing)
- [ ] `_TestWorker` inherits `BaseWorker` (`grep "class _TestWorker" model_test_dialog.py` shows `BaseWorker`)
- [ ] 3 orchestrators constructed in `MainWindow.__init__`
- [ ] 841+ tests collected, all passing
- [ ] Import smoke test passes
