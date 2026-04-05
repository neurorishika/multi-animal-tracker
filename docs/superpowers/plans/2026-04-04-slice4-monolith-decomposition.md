# Slice 4 — trackerkit Monolith Decomposition

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `trackerkit/gui/main_window.py` (19 972 lines) into a `trackerkit/gui/` subpackage of focused panel and widget modules. `MainWindow` becomes a thin coordinator under ~500 lines.

**Architecture:** `MainWindow` instantiates panels, wires their signals together, and delegates to `TrackerConfig`. Business logic moves to panels. Workers and dialogs are already in separate files after Slices 1–3 — this slice carves out the panel/UI logic.

**Tech Stack:** PySide6 (`QWidget`, `QVBoxLayout`, `Signal`), pytest

**Prerequisite:** Slices 1, 2, and 3 must be complete. Workers are already extracted (Slice 1), `self.config = TrackerConfig()` exists (Slice 2), dialogs are already split (Slice 3).

**Existing structure:** `trackerkit/gui/` already has:
- `dialogs/` — train_yolo, model_test, parameter_helper, bg_parameter_helper, cnn_identity_import, run_history
- `widgets/` — existing shared widgets

---

### Task 1: Audit main_window.py and map responsibilities

**Files:**
- Read: `src/hydra_suite/trackerkit/gui/main_window.py`

Before moving anything, map the current `MainWindow` into logical groups. This task produces no code — only a commented note in the plan (update this file) showing the decomposition map you'll use.

- [ ] **Step 1: Scan for top-level class definitions**

```bash
grep -n "^class " src/hydra_suite/trackerkit/gui/main_window.py
```

Record the class names and line numbers.

- [ ] **Step 2: Scan for method groups inside MainWindow**

Search for method name patterns that suggest panel boundaries:
```bash
grep -n "def _setup_\|def _init_\|def _create_\|def _build_" src/hydra_suite/trackerkit/gui/main_window.py
```

Methods that set up "tracking controls", "detection config", "identity config", "post-processing" are candidates for panel extraction.

- [ ] **Step 3: Map panels**

Based on steps 1–2, update the table in this plan with the actual method boundaries you found:

| Panel file | Methods/responsibilities to move | Approx lines |
|-----------|----------------------------------|--------------|
| `panels/tracking_panel.py` | (fill in from audit) | ? |
| `panels/detection_panel.py` | (fill in from audit) | ? |
| `panels/identity_panel.py` | (fill in from audit) | ? |
| `panels/postprocess_panel.py` | (fill in from audit) | ? |

- [ ] **Step 4: Note signal wiring**

List every `Signal` defined at class level in `MainWindow`. These stay in `MainWindow` — panels communicate up via signals.

---

### Task 2: Create panels/ package scaffold

**Files:**
- Create: `src/hydra_suite/trackerkit/gui/panels/__init__.py`
- Create: `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py`
- Create: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`
- Create: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py`
- Create: `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py`

- [ ] **Step 1: Create `panels/__init__.py`**

```python
# src/hydra_suite/trackerkit/gui/panels/__init__.py
"""trackerkit GUI panels — each panel owns one functional area of the UI."""
from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel
from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel
from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel
from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel

__all__ = ["TrackingPanel", "DetectionPanel", "IdentityPanel", "PostProcessPanel"]
```

- [ ] **Step 2: Create stub panel files**

Each panel follows this template — create all four:

```python
# src/hydra_suite/trackerkit/gui/panels/tracking_panel.py
"""TrackingPanel — playback controls and tracking run management."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from hydra_suite.trackerkit.config.schemas import TrackerConfig


class TrackingPanel(QWidget):
    """Controls for starting/stopping tracking and playback.

    Signals
    -------
    config_changed(TrackerConfig)
        Emitted when the user edits a tracking parameter.
    """

    config_changed: Signal = Signal(object)

    def __init__(self, config: TrackerConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""
        pass

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

Create `detection_panel.py`, `identity_panel.py`, `postprocess_panel.py` with the same structure but their own class names (`DetectionPanel`, `IdentityPanel`, `PostProcessPanel`) and docstrings.

- [ ] **Step 3: Run tests to confirm scaffold doesn't break anything**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 4: Commit scaffold**

```bash
git add src/hydra_suite/trackerkit/gui/panels/
git commit -m "feat: scaffold trackerkit/gui/panels/ subpackage with stub panel classes"
```

---

### Task 3: Extract TrackingPanel

- [ ] **Step 1: Find tracking-related setup methods**

```bash
grep -n "def .*track\|def .*play\|def .*video" src/hydra_suite/trackerkit/gui/main_window.py -i | head -40
```

Identify the methods that build the tracking controls section of the UI (play/pause controls, tracking start/stop buttons, progress bar, video preview).

- [ ] **Step 2: Move methods to TrackingPanel._build_ui()**

For each method identified in step 1 that purely sets up tracking UI:
1. Copy the method body into `TrackingPanel._build_ui()`.
2. Replace `self.some_control = ...` with `self._some_control = ...` (panel owns its widgets).
3. Where the method previously emitted signals from `MainWindow`, emit `self.config_changed` from `TrackingPanel` instead.
4. Delete or stub the method in `main_window.py` with a delegation call:

```python
# In MainWindow — stub after extraction:
def _setup_tracking_controls(self):
    # Moved to TrackingPanel
    self._tracking_panel._build_ui()
```

- [ ] **Step 3: Wire TrackingPanel in MainWindow**

In `MainWindow.__init__`:

```python
from hydra_suite.trackerkit.gui.panels import TrackingPanel

self._tracking_panel = TrackingPanel(self.config, parent=self)
self._tracking_panel.config_changed.connect(self._on_config_changed)
# Add to layout where tracking controls previously lived
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/tracking_panel.py src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract TrackingPanel from trackerkit MainWindow"
```

---

### Task 4: Extract DetectionPanel

- [ ] **Step 1: Find detection-related setup methods**

```bash
grep -n "def .*detect\|def .*model\|def .*camera" src/hydra_suite/trackerkit/gui/main_window.py -i | head -40
```

Identify methods that build the detector/model configuration section of the UI (model path picker, detector type selector, confidence threshold, preview button).

- [ ] **Step 2: Move methods to DetectionPanel._build_ui()**

Same procedure as Task 3 step 2. For model path changes, emit `config_changed` with updated config.

- [ ] **Step 3: Wire DetectionPanel in MainWindow**

```python
from hydra_suite.trackerkit.gui.panels import DetectionPanel

self._detection_panel = DetectionPanel(self.config, parent=self)
self._detection_panel.config_changed.connect(self._on_config_changed)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract DetectionPanel from trackerkit MainWindow"
```

---

### Task 5: Extract IdentityPanel

- [ ] **Step 1: Find identity-related setup methods**

```bash
grep -n "def .*identity\|def .*assign\|def .*embed" src/hydra_suite/trackerkit/gui/main_window.py -i | head -40
```

Identify methods building the identity/assignment configuration section.

- [ ] **Step 2: Move to IdentityPanel._build_ui()**

Same procedure as previous panels.

- [ ] **Step 3: Wire IdentityPanel in MainWindow**

```python
from hydra_suite.trackerkit.gui.panels import IdentityPanel

self._identity_panel = IdentityPanel(self.config, parent=self)
self._identity_panel.config_changed.connect(self._on_config_changed)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/identity_panel.py src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract IdentityPanel from trackerkit MainWindow"
```

---

### Task 6: Extract PostProcessPanel

- [ ] **Step 1: Find post-processing setup methods**

```bash
grep -n "def .*post\|def .*relink\|def .*interpolat\|def .*merge" src/hydra_suite/trackerkit/gui/main_window.py -i | head -40
```

- [ ] **Step 2: Move to PostProcessPanel._build_ui()**

- [ ] **Step 3: Wire PostProcessPanel in MainWindow**

```python
from hydra_suite.trackerkit.gui.panels import PostProcessPanel

self._postprocess_panel = PostProcessPanel(self.config, parent=self)
self._postprocess_panel.config_changed.connect(self._on_config_changed)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "refactor: extract PostProcessPanel from trackerkit MainWindow"
```

---

### Task 7: Clean up MainWindow and verify line count

After all four panels are extracted, `MainWindow` should be a thin coordinator.

- [ ] **Step 1: Review what remains in main_window.py**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Target: under 500 lines. If above 500, identify what still belongs in a panel and extract it.

- [ ] **Step 2: Remove dead code**

Search for any methods in `MainWindow` that are now empty stubs or fully delegated:

```bash
grep -n "def " src/hydra_suite/trackerkit/gui/main_window.py
```

Review each method. Any method that only calls `pass` or delegates 100% to a panel can be deleted; update callers to call the panel directly.

- [ ] **Step 3: Verify `MainWindow.__init__` structure**

`MainWindow.__init__` should follow this pattern and nothing more:

```python
def __init__(self, parent=None):
    super().__init__(parent)
    self.config = TrackerConfig()

    self._tracking_panel = TrackingPanel(self.config, parent=self)
    self._detection_panel = DetectionPanel(self.config, parent=self)
    self._identity_panel = IdentityPanel(self.config, parent=self)
    self._postprocess_panel = PostProcessPanel(self.config, parent=self)

    self._wire_signals()
    self._build_layout()
    self._apply_welcome_state()
```

If `__init__` is longer than ~80 lines, move setup logic into the named helpers above.

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 5: Confirm line count target**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 500 lines. Record the number in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/
git commit -m "refactor: trackerkit MainWindow reduced to <N> lines after panel extraction"
```

---

### Task 8: Verify and finalize

- [ ] **Step 1: Check imports from outside trackerkit still work**

```bash
python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
```

Expected: `ok` (no import errors).

- [ ] **Step 2: Run full test suite one final time**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 3: Confirm success criteria from design spec**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
# Must be ≤ 500
```

- [ ] **Step 4: Commit completion marker**

```bash
git commit --allow-empty -m "chore: Slice 4 (trackerkit monolith decomposition) complete — all sprint goals achieved"
```
