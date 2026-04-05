# Sprint 2 Panel Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract all 6 `setup_*_ui()` methods from `trackerkit/gui/main_window.py` into the scaffolded panel classes in `trackerkit/gui/panels/`, reducing `main_window.py` from 19,975 lines to ≤ 16,050 lines.

**Architecture:** Each panel class owns its widget tree (widgets are public attributes). MainWindow instantiates panels in `init_ui()`, adds them to the tab widget, and accesses their widgets via `self._panel_name.widget_name`. No business logic moves — only UI construction code.

**Tech Stack:** PySide6 (`QWidget`, `QVBoxLayout`, `Signal`), pytest, Python `re` module for widget reference substitution.

**Prerequisite:** Sprint 1 scaffold exists — `src/hydra_suite/trackerkit/gui/panels/` with 6 stub files. Branch: create a new worktree from `refactor/src-reorganization`.

---

## File Map

| File | Change |
|---|---|
| `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/panels/setup_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/panels/identity_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/panels/detection_panel.py` | Replace stub with real `_build_ui()` |
| `src/hydra_suite/trackerkit/gui/main_window.py` | Delete 6 setup methods; replace 6 tab blocks with panel instantiation; update ~1,200 widget refs |
| `tests/test_main_window_config_persistence.py` | Update 5 widget refs to panel-qualified paths |
| `tests/test_trackerkit_panels_smoke.py` | Create with 6 panel smoke tests |

---

## Transformation Rules (apply to every panel)

When moving `setup_X_ui()` body into panel's `_build_ui()`, apply these changes:

| Old (in main_window.py method) | New (in panel's _build_ui) |
|---|---|
| `layout = QVBoxLayout(self.tab_XXX)` (first line) | `layout = self._layout` |
| `self._create_help_label(` | `self._main_window._create_help_label(` |
| `self._set_compact_scroll_layout(` | `self._main_window._set_compact_scroll_layout(` |
| `self._set_compact_section_widget(` | `self._main_window._set_compact_section_widget(` |
| `self._remember_collapsible_state(` | `self._main_window._remember_collapsible_state(` |
| `.connect(self.HANDLER)` where HANDLER is a MainWindow method | `.connect(self._main_window.HANDLER)` |
| `self.WIDGET = QSomething(...)` (widget assignment) | **No change** — panel owns the widget |

**How to identify MainWindow method handlers:** Every handler listed in the "Handlers" section of each task is a MainWindow method. All other `self.X` in connect calls are either widget refs (which stay) or lambda expressions (which stay unchanged).

---

### Task 1: Set Up Worktree and Verify Baseline

**Files:** None — environment setup only.

- [ ] **Step 1: Create worktree**

```bash
cd /Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker
git worktree add .worktrees/panel-extraction -b refactor/panel-extraction-sprint2
cd .worktrees/panel-extraction
```

- [ ] **Step 2: Verify baseline test count**

```bash
conda run -n hydra-mps python -m pytest tests/ -q --tb=no -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -3
```

Expected output contains: `N passed` (record N — it must not decrease during this sprint).

- [ ] **Step 3: Verify panels package imports**

```bash
conda run -n hydra-mps python -c "
from hydra_suite.trackerkit.gui.panels import (
    DatasetPanel, SetupPanel, TrackingPanel,
    PostProcessPanel, IdentityPanel, DetectionPanel
)
print('ok')
"
```

Expected: `ok`

---

### Task 2: Update All Panel Stubs — Add `main_window` Parameter

All 6 panel stubs must accept a `main_window` argument before any extraction can begin. Update all 6 files now.

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`

- [ ] **Step 1: Update `dataset_panel.py`**

Replace the entire file content:

```python
"""DatasetPanel — active learning dataset generation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class DatasetPanel(QWidget):
    """Active learning dataset generation: frame selection, export, and controls."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 2: Update `setup_panel.py`**

Replace the entire file content:

```python
"""SetupPanel — preset selection, video files, display, and ROI configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class SetupPanel(QWidget):
    """Preset picker, video/batch file selection, ROI, and display options."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 3: Update `tracking_panel.py`**

Replace the entire file content:

```python
"""TrackingPanel — core tracking parameters, Kalman filter, and assignment config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class TrackingPanel(QWidget):
    """Kalman filter parameters, identity assignment, and backward pass controls."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 4: Update `postprocess_panel.py`**

Replace the entire file content:

```python
"""PostProcessPanel — trajectory cleaning, relinking, and interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class PostProcessPanel(QWidget):
    """Trajectory post-processing: cleaning, velocity breaks, and interpolation."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 5: Update `identity_panel.py`**

Replace the entire file content:

```python
"""IdentityPanel — identity classification, pose analysis, and keypoint config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class IdentityPanel(QWidget):
    """CNN/appearance identity assignment and pose backend configuration."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 6: Update `detection_panel.py`**

Replace the entire file content:

```python
"""DetectionPanel — detection method, image preprocessing, and model config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class DetectionPanel(QWidget):
    """Detection method selector, image-processing pipeline, and YOLO config."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
```

- [ ] **Step 7: Create smoke test file**

Create `tests/test_trackerkit_panels_smoke.py`:

```python
"""Smoke tests: each panel instantiates and exposes expected key widgets."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()
```

Note: the individual panel test functions will be added in later tasks as each panel is extracted. The fixture is added now so it's ready.

- [ ] **Step 8: Run tests to confirm stubs still pass**

```bash
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: same pass count as baseline.

- [ ] **Step 9: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/ tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "feat: update panel stubs to accept main_window parameter + smoke test fixture"
```

---

### Task 3: Extract DatasetPanel (lines 7448–7827, 30 widgets)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_main_window_config_persistence.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by DatasetPanel:**
`active_learning_content`, `btn_open_pose_label`, `btn_open_xanylabeling`, `btn_refresh_envs`, `chk_dataset_include_context`, `chk_dataset_probabilistic`, `chk_enable_dataset_gen`, `chk_enable_individual_dataset`, `chk_generate_individual_track_videos`, `chk_metric_count_mismatch`, `chk_metric_high_assignment_cost`, `chk_metric_high_uncertainty`, `chk_metric_low_confidence`, `chk_metric_track_loss`, `chk_suppress_foreign_obb_dataset`, `combo_individual_format`, `combo_xanylabeling_env`, `g_active_learning`, `g_dataset_config`, `g_frame_selection`, `g_individual_dataset`, `g_pose_label`, `g_quality_metrics`, `g_xanylabeling`, `ind_output_group`, `lbl_individual_info`, `line_dataset_class_name`, `spin_dataset_conf_threshold`, `spin_dataset_diversity_window`, `spin_dataset_max_frames`, `spin_individual_interval`

**MainWindow handlers connected in this panel:**
`_on_dataset_generation_toggled`, `_open_in_xanylabeling`, `_open_pose_label_ui`, `_refresh_xanylabeling_envs`

- [ ] **Step 1: Add required imports to `dataset_panel.py`**

Add these imports after the existing imports in `dataset_panel.py` (all the Qt widgets used by `setup_dataset_ui` — get the full list by running `grep "^from\|^import" main_window.py | head -60` and copying relevant PySide6 imports):

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
)
```

- [ ] **Step 2: Populate `_build_ui()` in `dataset_panel.py`**

Copy the body of `setup_dataset_ui` from `main_window.py` (lines 7449–7826, everything after the `def` line) into `_build_ui()`. Then apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_dataset)` → `layout = self._layout`
2. Replace all helper calls (14 occurrences total):
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
3. Replace handler connections (4 occurrences):
   - `.connect(self._on_dataset_generation_toggled)` → `.connect(self._main_window._on_dataset_generation_toggled)`
   - `.connect(self._open_in_xanylabeling)` → `.connect(self._main_window._open_in_xanylabeling)`
   - `.connect(self._open_pose_label_ui)` → `.connect(self._main_window._open_pose_label_ui)`
   - `.connect(self._refresh_xanylabeling_envs)` → `.connect(self._main_window._refresh_xanylabeling_envs)`

All other `self.something` lines stay unchanged — panel's `self` owns those widgets.

- [ ] **Step 3: Replace the tab block in `main_window.py`**

In `main_window.py` around line 4190, find and replace:

```python
        # Tab 6: Dataset Generation (Active Learning)
        self.tab_dataset = QWidget()
        self.setup_dataset_ui()
        self.tabs.addTab(self.tab_dataset, "Build Dataset")
```

With:

```python
        # Tab 6: Dataset Generation (Active Learning)
        from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel
        self._dataset_panel = DatasetPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._dataset_panel, "Build Dataset")
```

- [ ] **Step 4: Delete `setup_dataset_ui` from `main_window.py`**

Find the method definition `def setup_dataset_ui(self: object) -> object:` (line ~7448) and delete it and all its body lines (7448–7827). The next method after it is `def setup_individual_analysis_ui`.

- [ ] **Step 5: Run widget reference substitution script**

Save this script as `/tmp/subst_dataset.py` and run it:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_dataset_panel"
widgets = [
    "active_learning_content", "btn_open_pose_label", "btn_open_xanylabeling",
    "btn_refresh_envs", "chk_dataset_include_context", "chk_dataset_probabilistic",
    "chk_enable_dataset_gen", "chk_enable_individual_dataset",
    "chk_generate_individual_track_videos", "chk_metric_count_mismatch",
    "chk_metric_high_assignment_cost", "chk_metric_high_uncertainty",
    "chk_metric_low_confidence", "chk_metric_track_loss",
    "chk_suppress_foreign_obb_dataset", "combo_individual_format",
    "combo_xanylabeling_env", "g_active_learning", "g_dataset_config",
    "g_frame_selection", "g_individual_dataset", "g_pose_label",
    "g_quality_metrics", "g_xanylabeling", "ind_output_group",
    "lbl_individual_info", "line_dataset_class_name",
    "spin_dataset_conf_threshold", "spin_dataset_diversity_window",
    "spin_dataset_max_frames", "spin_individual_interval",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_dataset.py
```

Expected output: `Done — N substitutions made` where N > 0 (typically 80–120 for this panel).

- [ ] **Step 6: Update `test_main_window_config_persistence.py`**

Find all occurrences of `window.combo_xanylabeling_env` and `reloaded_window.combo_xanylabeling_env` in the file and replace with `window._dataset_panel.combo_xanylabeling_env` and `reloaded_window._dataset_panel.combo_xanylabeling_env` respectively.

There are 2 occurrences on lines ~112 and ~114. After the edit those lines should read:

```python
    assert window._dataset_panel.combo_xanylabeling_env.currentText() == "x-anylabeling-beta"
    window._dataset_panel.combo_xanylabeling_env.setCurrentText("x-anylabeling-alpha")
```

- [ ] **Step 7: Add DatasetPanel smoke test to `tests/test_trackerkit_panels_smoke.py`**

Append this test function to the file:

```python
def test_dataset_panel_instantiated(main_window):
    """DatasetPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    assert hasattr(main_window, "_dataset_panel")
    assert isinstance(main_window._dataset_panel, DatasetPanel)
    assert hasattr(main_window._dataset_panel, "combo_xanylabeling_env")
    assert hasattr(main_window._dataset_panel, "chk_enable_dataset_gen")
```

- [ ] **Step 8: Import smoke test**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
```

Expected: `ok` (no AttributeError, no ImportError).

- [ ] **Step 9: Run full test suite**

```bash
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: pass count ≥ baseline (from Task 1 Step 2). Zero new failures.

- [ ] **Step 10: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 19,600 lines.

- [ ] **Step 11: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/dataset_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_main_window_config_persistence.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract DatasetPanel from trackerkit MainWindow"
```

---

### Task 4: Extract SetupPanel (lines 4341–5037, 58 widgets)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/setup_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by SetupPanel:**
`btn_add_batch`, `btn_clear_batch`, `btn_csv`, `btn_detect_fps`, `btn_export_batch`, `btn_file`, `btn_first_frame`, `btn_import_batch`, `btn_last_frame`, `btn_load_config`, `btn_load_preset`, `btn_next_frame`, `btn_play_pause`, `btn_prev_frame`, `btn_random_seek`, `btn_remove_batch`, `btn_reset_range`, `btn_save_config`, `btn_save_custom`, `btn_set_end_current`, `btn_set_start_current`, `btn_show_gpu_info`, `check_save_confidence`, `chk_debug_logging`, `chk_enable_profiling`, `chk_show_circles`, `chk_show_kalman_uncertainty`, `chk_show_labels`, `chk_show_orientation`, `chk_show_state`, `chk_show_trajectories`, `chk_use_cached_detections`, `chk_visualization_free`, `combo_compute_runtime`, `combo_playback_speed`, `combo_presets`, `config_status_label`, `container_batch`, `csv_line`, `file_line`, `g_batch`, `g_display`, `g_video_player`, `label_fps_info`, `lbl_batch_warning`, `lbl_current_frame`, `lbl_range_info`, `lbl_video_info`, `list_batch_videos`, `preset_description_label`, `preset_status_label`, `slider_timeline`, `spin_end_frame`, `spin_fps`, `spin_max_targets`, `spin_resize`, `spin_start_frame`, `spin_traj_hist`

**MainWindow handlers connected in this panel:**
`_add_videos_to_batch`, `_clear_batch`, `_detect_fps_from_current_video`, `_export_batch_list`, `_goto_first_frame`, `_goto_last_frame`, `_goto_next_frame`, `_goto_prev_frame`, `_goto_random_frame`, `_import_batch_list`, `_load_selected_preset`, `_on_batch_mode_toggled`, `_on_batch_video_selected`, `_on_frame_range_changed`, `_on_timeline_changed`, `_remove_from_batch`, `_reset_frame_range`, `_save_custom_preset`, `_set_end_to_current`, `_set_start_to_current`, `_toggle_playback`, `_update_fps_info`, `load_config`, `save_config`, `select_csv`, `select_file`, `show_gpu_info`, `toggle_debug_logging`

- [ ] **Step 1: Add required imports to `setup_panel.py`**

Add after the existing imports:

```python
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
)
```

- [ ] **Step 2: Populate `_build_ui()` in `setup_panel.py`**

Copy the body of `setup_setup_ui` from `main_window.py` (lines 4342–5036) into `_build_ui()`. Apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_setup)` → `layout = self._layout`
2. Replace all helper calls:
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
   - `self._remember_collapsible_state(` → `self._main_window._remember_collapsible_state(`
   - `self._populate_preset_combo(` → `self._main_window._populate_preset_combo(`
   - `self._populate_compute_runtime_options(` → `self._main_window._populate_compute_runtime_options(`
3. Replace all handler connections. Every `.connect(self.X)` where X is one of the 28 handlers listed above becomes `.connect(self._main_window.X)`.

- [ ] **Step 3: Replace the tab block in `main_window.py`**

Find (around line 4164):

```python
        # Tab 1: Setup (Files & Performance)
        self.tab_setup = QWidget()
        self.setup_setup_ui()
        self.tabs.addTab(self.tab_setup, "Get Started")
```

Replace with:

```python
        # Tab 1: Setup (Files & Performance)
        from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel
        self._setup_panel = SetupPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._setup_panel, "Get Started")
```

- [ ] **Step 4: Delete `setup_setup_ui` from `main_window.py`**

Delete the entire method `def setup_setup_ui(self: object) -> object:` and its body (lines ~4341–5037).

- [ ] **Step 5: Run widget reference substitution script**

Save as `/tmp/subst_setup.py` and run:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_setup_panel"
widgets = [
    "btn_add_batch", "btn_clear_batch", "btn_csv", "btn_detect_fps",
    "btn_export_batch", "btn_file", "btn_first_frame", "btn_import_batch",
    "btn_last_frame", "btn_load_config", "btn_load_preset", "btn_next_frame",
    "btn_play_pause", "btn_prev_frame", "btn_random_seek", "btn_remove_batch",
    "btn_reset_range", "btn_save_config", "btn_save_custom", "btn_set_end_current",
    "btn_set_start_current", "btn_show_gpu_info", "check_save_confidence",
    "chk_debug_logging", "chk_enable_profiling", "chk_show_circles",
    "chk_show_kalman_uncertainty", "chk_show_labels", "chk_show_orientation",
    "chk_show_state", "chk_show_trajectories", "chk_use_cached_detections",
    "chk_visualization_free", "combo_compute_runtime", "combo_playback_speed",
    "combo_presets", "config_status_label", "container_batch", "csv_line",
    "file_line", "g_batch", "g_display", "g_video_player", "label_fps_info",
    "lbl_batch_warning", "lbl_current_frame", "lbl_range_info", "lbl_video_info",
    "list_batch_videos", "preset_description_label", "preset_status_label",
    "slider_timeline", "spin_end_frame", "spin_fps", "spin_max_targets",
    "spin_resize", "spin_start_frame", "spin_traj_hist",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_setup.py
```

- [ ] **Step 6: Add SetupPanel smoke test to `tests/test_trackerkit_panels_smoke.py`**

Append:

```python
def test_setup_panel_instantiated(main_window):
    """SetupPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel

    assert hasattr(main_window, "_setup_panel")
    assert isinstance(main_window._setup_panel, SetupPanel)
    assert hasattr(main_window._setup_panel, "combo_presets")
    assert hasattr(main_window._setup_panel, "btn_file")
```

- [ ] **Step 7: Import smoke test**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
```

- [ ] **Step 8: Run full test suite**

```bash
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: pass count ≥ baseline. Zero new failures.

- [ ] **Step 9: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 18,900 lines.

- [ ] **Step 10: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/setup_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract SetupPanel from trackerkit MainWindow"
```

---

### Task 5: Extract TrackingPanel (lines 6052–6846, 44 widgets)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_main_window_config_persistence.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by TrackingPanel:**
`btn_param_helper`, `chk_directed_orient_smoothing`, `chk_enable_backward`, `chk_enable_confidence_density_map`, `chk_enable_pose_rejection`, `chk_instant_flip`, `chk_spatial_optimization`, `chk_use_mahal`, `combo_assignment_method`, `g_density`, `spin_assoc_gate_multiplier`, `spin_assoc_high_conf_threshold`, `spin_assoc_max_area_ratio`, `spin_assoc_max_aspect_diff`, `spin_continuity_thresh`, `spin_density_binarize_threshold`, `spin_density_conservative_factor`, `spin_density_downsample_factor`, `spin_density_gaussian_sigma_scale`, `spin_density_min_area_bodies`, `spin_density_min_duration`, `spin_density_temporal_sigma`, `spin_directed_orient_flip_conf`, `spin_directed_orient_flip_persist`, `spin_kalman_damping`, `spin_kalman_initial_velocity_retention`, `spin_kalman_lateral_noise`, `spin_kalman_longitudinal_noise`, `spin_kalman_maturity_age`, `spin_kalman_max_velocity`, `spin_kalman_meas`, `spin_kalman_noise`, `spin_lost_thresh`, `spin_max_dist`, `spin_max_orient`, `spin_min_detect`, `spin_min_detections_to_start`, `spin_min_respawn_distance`, `spin_min_track`, `spin_pose_rejection_min_visibility`, `spin_pose_rejection_threshold`, `spin_track_feature_ema_alpha`, `spin_velocity`, `tracking_accordion`

**MainWindow handlers connected in this panel:**
`_open_parameter_helper`

- [ ] **Step 1: Add required imports to `tracking_panel.py`**

Add after existing imports:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
)
from hydra_suite.trackerkit.gui.main_window import AccordionContainer, CollapsibleGroupBox
```

- [ ] **Step 2: Populate `_build_ui()` in `tracking_panel.py`**

Copy the body of `setup_tracking_ui` from `main_window.py` (lines 6053–6845) into `_build_ui()`. Apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_tracking)` → `layout = self._layout`
2. Replace all helper calls:
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
   - `self._remember_collapsible_state(` → `self._main_window._remember_collapsible_state(`
3. Replace handler connection (1 occurrence):
   - `.connect(self._open_parameter_helper)` → `.connect(self._main_window._open_parameter_helper)`

- [ ] **Step 3: Replace the tab block in `main_window.py`**

Find (around line 4180):

```python
        # Tab 4: Tracking (Kalman, Logic, Lifecycle)
        self.tab_tracking = QWidget()
        self.setup_tracking_ui()
        self.tabs.addTab(self.tab_tracking, "Track Movement")
```

Replace with:

```python
        # Tab 4: Tracking (Kalman, Logic, Lifecycle)
        from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel
        self._tracking_panel = TrackingPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._tracking_panel, "Track Movement")
```

- [ ] **Step 4: Delete `setup_tracking_ui` from `main_window.py`**

Delete the entire method `def setup_tracking_ui` and its body (lines ~6052–6846).

- [ ] **Step 5: Run widget reference substitution script**

Save as `/tmp/subst_tracking.py` and run:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_tracking_panel"
widgets = [
    "btn_param_helper", "chk_directed_orient_smoothing", "chk_enable_backward",
    "chk_enable_confidence_density_map", "chk_enable_pose_rejection",
    "chk_instant_flip", "chk_spatial_optimization", "chk_use_mahal",
    "combo_assignment_method", "g_density", "spin_assoc_gate_multiplier",
    "spin_assoc_high_conf_threshold", "spin_assoc_max_area_ratio",
    "spin_assoc_max_aspect_diff", "spin_continuity_thresh",
    "spin_density_binarize_threshold", "spin_density_conservative_factor",
    "spin_density_downsample_factor", "spin_density_gaussian_sigma_scale",
    "spin_density_min_area_bodies", "spin_density_min_duration",
    "spin_density_temporal_sigma", "spin_directed_orient_flip_conf",
    "spin_directed_orient_flip_persist", "spin_kalman_damping",
    "spin_kalman_initial_velocity_retention", "spin_kalman_lateral_noise",
    "spin_kalman_longitudinal_noise", "spin_kalman_maturity_age",
    "spin_kalman_max_velocity", "spin_kalman_meas", "spin_kalman_noise",
    "spin_lost_thresh", "spin_max_dist", "spin_max_orient", "spin_min_detect",
    "spin_min_detections_to_start", "spin_min_respawn_distance", "spin_min_track",
    "spin_pose_rejection_min_visibility", "spin_pose_rejection_threshold",
    "spin_track_feature_ema_alpha", "spin_velocity", "tracking_accordion",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_tracking.py
```

- [ ] **Step 6: Update `test_main_window_config_persistence.py`**

Find all occurrences of `window.g_density`, `reloaded_window.g_density`, `window.chk_enable_confidence_density_map`, and `reloaded_window.chk_enable_confidence_density_map` and add the panel prefix:

```python
# Before:
assert not window.g_density.isHidden()
window.chk_enable_confidence_density_map.setChecked(False)
assert window.g_density.isHidden()
# ...
reloaded_window.chk_enable_confidence_density_map.isChecked() is False
reloaded_window.g_density.isHidden()
reloaded_window.chk_enable_confidence_density_map.setChecked(True)
assert not reloaded_window.g_density.isHidden()

# After:
assert not window._tracking_panel.g_density.isHidden()
window._tracking_panel.chk_enable_confidence_density_map.setChecked(False)
assert window._tracking_panel.g_density.isHidden()
# ...
reloaded_window._tracking_panel.chk_enable_confidence_density_map.isChecked() is False
reloaded_window._tracking_panel.g_density.isHidden()
reloaded_window._tracking_panel.chk_enable_confidence_density_map.setChecked(True)
assert not reloaded_window._tracking_panel.g_density.isHidden()
```

- [ ] **Step 7: Add TrackingPanel smoke test**

Append to `tests/test_trackerkit_panels_smoke.py`:

```python
def test_tracking_panel_instantiated(main_window):
    """TrackingPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

    assert hasattr(main_window, "_tracking_panel")
    assert isinstance(main_window._tracking_panel, TrackingPanel)
    assert hasattr(main_window._tracking_panel, "g_density")
    assert hasattr(main_window._tracking_panel, "chk_enable_confidence_density_map")
```

- [ ] **Step 8: Import smoke test**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
```

- [ ] **Step 9: Run full test suite**

```bash
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: pass count ≥ baseline. Zero new failures.

- [ ] **Step 10: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 18,100 lines.

- [ ] **Step 11: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/tracking_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_main_window_config_persistence.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract TrackingPanel from trackerkit MainWindow"
```

---

### Task 6: Extract PostProcessPanel (lines 6847–7447, 67 widgets)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by PostProcessPanel:**
`_btn_open_refinekit`, `_video_pose_color`, `btn_video_out`, `btn_video_pose_color`, `check_show_labels`, `check_show_orientation`, `check_show_trails`, `check_video_output`, `check_video_show_pose`, `chk_cleanup_temp_files`, `chk_enable_tracklet_relinking`, `combo_interpolation_method`, `combo_video_pose_color_mode`, `enable_postprocessing`, `lbl_arrow_length`, `lbl_enable_tracklet_relinking`, `lbl_heading_flip_max_burst`, `lbl_interpolation_max_gap`, `lbl_interpolation_method`, `lbl_marker_size`, `lbl_max_occlusion_gap`, `lbl_max_velocity_break`, `lbl_max_velocity_zscore`, `lbl_merge_overlap_multiplier`, `lbl_min_overlap_frames`, `lbl_min_trajectory_length`, `lbl_pose_export_min_valid_fraction`, `lbl_pose_export_min_valid_keypoints`, `lbl_pose_postproc_max_gap`, `lbl_pose_temporal_outlier_zscore`, `lbl_relink_min_pose_quality`, `lbl_relink_pose_max_distance`, `lbl_text_scale`, `lbl_trail_duration`, `lbl_velocity_zscore_min_vel`, `lbl_velocity_zscore_window`, `lbl_video_path`, `lbl_video_pose_color`, `lbl_video_pose_color_label`, `lbl_video_pose_color_mode`, `lbl_video_pose_disabled_hint`, `lbl_video_pose_line_thickness`, `lbl_video_pose_point_radius`, `lbl_video_pose_point_thickness`, `lbl_video_pose_settings`, `lbl_video_viz_settings`, `spin_arrow_length`, `spin_heading_flip_max_burst`, `spin_interpolation_max_gap`, `spin_marker_size`, `spin_max_occlusion_gap`, `spin_max_velocity_break`, `spin_max_velocity_zscore`, `spin_merge_overlap_multiplier`, `spin_min_overlap_frames`, `spin_min_trajectory_length`, `spin_pose_export_min_valid_fraction`, `spin_pose_export_min_valid_keypoints`, `spin_pose_postproc_max_gap`, `spin_pose_temporal_outlier_zscore`, `spin_relink_min_pose_quality`, `spin_relink_pose_max_distance`, `spin_text_scale`, `spin_trail_duration`, `spin_velocity_zscore_min_vel`, `spin_velocity_zscore_window`, `spin_video_pose_line_thickness`, `spin_video_pose_point_radius`, `spin_video_pose_point_thickness`, `video_out_line`

**Note:** `_btn_open_refinekit` and `_video_pose_color` begin with underscore — include them in the widget list as-is (the substitution script will match them correctly).

**MainWindow handlers connected in this panel:**
`_on_cleaning_toggled`, `_on_video_output_toggled`, `_open_refinekit`, `_select_video_pose_color`, `select_video_output`

- [ ] **Step 1: Add required imports to `postprocess_panel.py`**

Add after existing imports:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)
from hydra_suite.trackerkit.gui.main_window import CollapsibleGroupBox
```

- [ ] **Step 2: Populate `_build_ui()` in `postprocess_panel.py`**

Copy the body of `setup_data_ui` from `main_window.py` (lines 6848–7446) into `_build_ui()`. Apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_data)` → `layout = self._layout`
2. Replace all helper calls:
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
   - `self._remember_collapsible_state(` → `self._main_window._remember_collapsible_state(`
3. Replace handler connections (5 occurrences):
   - `.connect(self._on_cleaning_toggled)` → `.connect(self._main_window._on_cleaning_toggled)`
   - `.connect(self._on_video_output_toggled)` → `.connect(self._main_window._on_video_output_toggled)`
   - `.connect(self._open_refinekit)` → `.connect(self._main_window._open_refinekit)`
   - `.connect(self._select_video_pose_color)` → `.connect(self._main_window._select_video_pose_color)`
   - `.connect(self.select_video_output)` → `.connect(self._main_window.select_video_output)`

- [ ] **Step 3: Replace the tab block in `main_window.py`**

Find:

```python
        # Tab 5: Data (Post-proc)
        self.tab_data = QWidget()
        self.setup_data_ui()
        self.tabs.addTab(self.tab_data, "Clean Results")
```

Replace with:

```python
        # Tab 5: Data (Post-proc)
        from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel
        self._postprocess_panel = PostProcessPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._postprocess_panel, "Clean Results")
```

- [ ] **Step 4: Delete `setup_data_ui` from `main_window.py`**

Delete the entire method `def setup_data_ui` and its body (lines ~6847–7447).

- [ ] **Step 5: Run widget reference substitution script**

Save as `/tmp/subst_postprocess.py` and run:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_postprocess_panel"
widgets = [
    "_btn_open_refinekit", "_video_pose_color", "btn_video_out",
    "btn_video_pose_color", "check_show_labels", "check_show_orientation",
    "check_show_trails", "check_video_output", "check_video_show_pose",
    "chk_cleanup_temp_files", "chk_enable_tracklet_relinking",
    "combo_interpolation_method", "combo_video_pose_color_mode",
    "enable_postprocessing", "lbl_arrow_length", "lbl_enable_tracklet_relinking",
    "lbl_heading_flip_max_burst", "lbl_interpolation_max_gap",
    "lbl_interpolation_method", "lbl_marker_size", "lbl_max_occlusion_gap",
    "lbl_max_velocity_break", "lbl_max_velocity_zscore",
    "lbl_merge_overlap_multiplier", "lbl_min_overlap_frames",
    "lbl_min_trajectory_length", "lbl_pose_export_min_valid_fraction",
    "lbl_pose_export_min_valid_keypoints", "lbl_pose_postproc_max_gap",
    "lbl_pose_temporal_outlier_zscore", "lbl_relink_min_pose_quality",
    "lbl_relink_pose_max_distance", "lbl_text_scale", "lbl_trail_duration",
    "lbl_velocity_zscore_min_vel", "lbl_velocity_zscore_window", "lbl_video_path",
    "lbl_video_pose_color", "lbl_video_pose_color_label",
    "lbl_video_pose_color_mode", "lbl_video_pose_disabled_hint",
    "lbl_video_pose_line_thickness", "lbl_video_pose_point_radius",
    "lbl_video_pose_point_thickness", "lbl_video_pose_settings",
    "lbl_video_viz_settings", "spin_arrow_length", "spin_heading_flip_max_burst",
    "spin_interpolation_max_gap", "spin_marker_size", "spin_max_occlusion_gap",
    "spin_max_velocity_break", "spin_max_velocity_zscore",
    "spin_merge_overlap_multiplier", "spin_min_overlap_frames",
    "spin_min_trajectory_length", "spin_pose_export_min_valid_fraction",
    "spin_pose_export_min_valid_keypoints", "spin_pose_postproc_max_gap",
    "spin_pose_temporal_outlier_zscore", "spin_relink_min_pose_quality",
    "spin_relink_pose_max_distance", "spin_text_scale", "spin_trail_duration",
    "spin_velocity_zscore_min_vel", "spin_velocity_zscore_window",
    "spin_video_pose_line_thickness", "spin_video_pose_point_radius",
    "spin_video_pose_point_thickness", "video_out_line",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_postprocess.py
```

- [ ] **Step 6: Add PostProcessPanel smoke test**

Append to `tests/test_trackerkit_panels_smoke.py`:

```python
def test_postprocess_panel_instantiated(main_window):
    """PostProcessPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel

    assert hasattr(main_window, "_postprocess_panel")
    assert isinstance(main_window._postprocess_panel, PostProcessPanel)
    assert hasattr(main_window._postprocess_panel, "enable_postprocessing")
    assert hasattr(main_window._postprocess_panel, "combo_interpolation_method")
```

- [ ] **Step 7: Import smoke test + full test suite**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: `ok` then pass count ≥ baseline.

- [ ] **Step 8: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 17,500 lines.

- [ ] **Step 9: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract PostProcessPanel from trackerkit MainWindow"
```

---

### Task 7: Extract IdentityPanel (lines 7828–8324, 55 widgets)

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/identity_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_main_window_config_persistence.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by IdentityPanel:**
`_background_color`, `apriltag_settings_widget`, `btn_add_cnn_classifier`, `btn_background_color`, `btn_browse_pose_skeleton_file`, `btn_median_color`, `btn_refresh_pose_sleap_envs`, `chk_enable_pose_extractor`, `chk_individual_interpolate`, `chk_pose_overrides_headtail`, `chk_sleap_experimental_features`, `chk_suppress_foreign_obb`, `cnn_rows_layout`, `cnn_scroll_area`, `combo_apriltag_family`, `combo_cnn_identity_model`, `combo_pose_model`, `combo_pose_model_type`, `combo_pose_runtime_flavor`, `combo_pose_sleap_env`, `combo_yolo_headtail_model`, `combo_yolo_headtail_model_type`, `form_pose_runtime`, `g_apriltags`, `g_cnn_classifiers`, `g_identity`, `g_individual_pipeline_common`, `g_pose_runtime`, `headtail_model_row_widget`, `identity_content`, `lbl_background_color`, `lbl_cnn_arch`, `lbl_cnn_class_names`, `lbl_cnn_input_size`, `lbl_cnn_label`, `lbl_cnn_num_classes`, `lbl_identity_help`, `lbl_individual_yolo_only_notice`, `lbl_pose_runtime_help`, `line_color_tag_model`, `line_pose_skeleton_file`, `list_pose_direction_anterior`, `list_pose_direction_posterior`, `list_pose_ignore_keypoints`, `pose_runtime_content`, `pose_sleap_env_row_widget`, `pose_sleap_experimental_row_widget`, `spin_apriltag_decimate`, `spin_cnn_confidence`, `spin_cnn_window`, `spin_color_tag_conf`, `spin_identity_match_bonus`, `spin_identity_mismatch_penalty`, `spin_individual_padding`, `spin_pose_batch`, `spin_pose_min_kpt_conf_valid`, `spin_yolo_headtail_conf`

**MainWindow handlers connected in this panel:**
`_add_cnn_classifier_row`, `_compute_median_background_color`, `_on_cleaning_toggled`, `_on_identity_analysis_toggled`, `_on_pose_analysis_toggled`, `_on_runtime_context_changed`, `_refresh_pose_sleap_envs`, `_sync_video_pose_overlay_controls`, `on_pose_model_changed`

- [ ] **Step 1: Add required imports to `identity_panel.py`**

Add after existing imports:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
)
from hydra_suite.trackerkit.gui.main_window import CollapsibleGroupBox
```

- [ ] **Step 2: Populate `_build_ui()` in `identity_panel.py`**

Copy the body of `setup_individual_analysis_ui` from `main_window.py` (lines 7829–8323) into `_build_ui()`. Apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_individual)` → `layout = self._layout`
2. Replace all helper calls:
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
   - `self._remember_collapsible_state(` → `self._main_window._remember_collapsible_state(`
3. Replace handler connections (9 handler names listed above):
   - Each `.connect(self.HANDLER)` → `.connect(self._main_window.HANDLER)`

- [ ] **Step 3: Replace the tab block in `main_window.py`**

Find:

```python
        # Tab 3: Individual Analysis (Identity)
        self.tab_individual = QWidget()
        self.setup_individual_analysis_ui()
        self.tabs.addTab(self.tab_individual, "Analyze Individuals")
        self._sync_individual_analysis_mode_ui()
```

Replace with:

```python
        # Tab 3: Individual Analysis (Identity)
        from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel
        self._identity_panel = IdentityPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._identity_panel, "Analyze Individuals")
        self._sync_individual_analysis_mode_ui()
```

- [ ] **Step 4: Delete `setup_individual_analysis_ui` from `main_window.py`**

Delete the entire method `def setup_individual_analysis_ui` and its body (lines ~7828–8324).

- [ ] **Step 5: Run widget reference substitution script**

Save as `/tmp/subst_identity.py` and run:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_identity_panel"
widgets = [
    "_background_color", "apriltag_settings_widget", "btn_add_cnn_classifier",
    "btn_background_color", "btn_browse_pose_skeleton_file", "btn_median_color",
    "btn_refresh_pose_sleap_envs", "chk_enable_pose_extractor",
    "chk_individual_interpolate", "chk_pose_overrides_headtail",
    "chk_sleap_experimental_features", "chk_suppress_foreign_obb",
    "cnn_rows_layout", "cnn_scroll_area", "combo_apriltag_family",
    "combo_cnn_identity_model", "combo_pose_model", "combo_pose_model_type",
    "combo_pose_runtime_flavor", "combo_pose_sleap_env",
    "combo_yolo_headtail_model", "combo_yolo_headtail_model_type",
    "form_pose_runtime", "g_apriltags", "g_cnn_classifiers", "g_identity",
    "g_individual_pipeline_common", "g_pose_runtime", "headtail_model_row_widget",
    "identity_content", "lbl_background_color", "lbl_cnn_arch",
    "lbl_cnn_class_names", "lbl_cnn_input_size", "lbl_cnn_label",
    "lbl_cnn_num_classes", "lbl_identity_help", "lbl_individual_yolo_only_notice",
    "lbl_pose_runtime_help", "line_color_tag_model", "line_pose_skeleton_file",
    "list_pose_direction_anterior", "list_pose_direction_posterior",
    "list_pose_ignore_keypoints", "pose_runtime_content",
    "pose_sleap_env_row_widget", "pose_sleap_experimental_row_widget",
    "spin_apriltag_decimate", "spin_cnn_confidence", "spin_cnn_window",
    "spin_color_tag_conf", "spin_identity_match_bonus",
    "spin_identity_mismatch_penalty", "spin_individual_padding",
    "spin_pose_batch", "spin_pose_min_kpt_conf_valid", "spin_yolo_headtail_conf",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_identity.py
```

- [ ] **Step 6: Update `test_main_window_config_persistence.py`**

Replace `window.combo_yolo_headtail_model_type`, `window.combo_yolo_headtail_model`, `reloaded_window.combo_yolo_headtail_model_type`, `reloaded_window.combo_yolo_headtail_model` with the panel-qualified versions:

```python
# Before:
window.combo_yolo_headtail_model_type.setCurrentText("tiny")
# ...
window.combo_yolo_headtail_model,
# ...
reloaded_window.combo_yolo_headtail_model_type.currentText() == "tiny"

# After:
window._identity_panel.combo_yolo_headtail_model_type.setCurrentText("tiny")
# ...
window._identity_panel.combo_yolo_headtail_model,
# ...
reloaded_window._identity_panel.combo_yolo_headtail_model_type.currentText() == "tiny"
```

- [ ] **Step 7: Add IdentityPanel smoke test**

Append to `tests/test_trackerkit_panels_smoke.py`:

```python
def test_identity_panel_instantiated(main_window):
    """IdentityPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel

    assert hasattr(main_window, "_identity_panel")
    assert isinstance(main_window._identity_panel, IdentityPanel)
    assert hasattr(main_window._identity_panel, "combo_yolo_headtail_model")
    assert hasattr(main_window._identity_panel, "combo_yolo_headtail_model_type")
```

- [ ] **Step 8: Import smoke test + full test suite**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: `ok` then pass count ≥ baseline.

- [ ] **Step 9: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 17,000 lines.

- [ ] **Step 10: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/identity_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_main_window_config_persistence.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract IdentityPanel from trackerkit MainWindow"
```

---

### Task 8: Extract DetectionPanel (lines 5038–6051, 74 widgets) — largest panel

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`
- Modify: `tests/test_trackerkit_panels_smoke.py`

**Widgets owned by DetectionPanel:**
`bg_accordion`, `btn_auto_set_aspect_ratio`, `btn_auto_set_body_size`, `btn_bg_autotune`, `chk_adaptive_bg`, `chk_additional_dilation`, `chk_conservative_split`, `chk_dark_on_light`, `chk_enable_aspect_ratio_filtering`, `chk_enable_tensorrt`, `chk_enable_yolo_batching`, `chk_lighting_stab`, `chk_show_bg`, `chk_show_fg`, `chk_show_yolo_obb`, `chk_size_filtering`, `chk_use_custom_obb_iou`, `chk_yolo_seq_square_crop`, `chk_yolo_seq_stage2_pow2_pad`, `combo_detection_method`, `combo_device`, `combo_yolo_batch_mode`, `combo_yolo_crop_obb_model`, `combo_yolo_detect_model`, `combo_yolo_model`, `combo_yolo_obb_mode`, `g_gpu_accel`, `g_img`, `g_overlays_bg`, `g_overlays_yolo`, `label_body_size_info`, `label_brightness_val`, `label_contrast_val`, `label_detection_stats`, `label_gamma_val`, `lbl_obb_mode_warning`, `lbl_tensorrt_batch`, `lbl_yolo_batch_mode`, `lbl_yolo_batch_size`, `line_yolo_classes`, `slider_brightness`, `slider_contrast`, `slider_gamma`, `spin_bg_learning`, `spin_bg_prime`, `spin_conservative_erode`, `spin_conservative_kernel`, `spin_dilation_iterations`, `spin_dilation_kernel_size`, `spin_lighting_median`, `spin_lighting_smooth`, `spin_max_ar_multiplier`, `spin_max_contour_multiplier`, `spin_max_object_size`, `spin_min_ar_multiplier`, `spin_min_contour`, `spin_min_object_size`, `spin_morph_size`, `spin_reference_aspect_ratio`, `spin_reference_body_size`, `spin_tensorrt_batch`, `spin_threshold`, `spin_yolo_batch_size`, `spin_yolo_confidence`, `spin_yolo_iou`, `spin_yolo_seq_crop_pad`, `spin_yolo_seq_detect_conf`, `spin_yolo_seq_min_crop_px`, `spin_yolo_seq_stage2_imgsz`, `stack_detection`, `yolo_group`, `yolo_seq_advanced`, `yolo_seq_advanced_content`

**MainWindow handlers connected in this panel:**
`_on_brightness_changed`, `_on_contrast_changed`, `_on_gamma_changed`, `_on_tensorrt_toggled`, `_on_yolo_mode_changed`, `_open_bg_parameter_helper`, `_update_body_size_info`, `on_yolo_model_changed`

- [ ] **Step 1: Add required imports to `detection_panel.py`**

Add after existing imports:

```python
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
)
from hydra_suite.trackerkit.gui.main_window import AccordionContainer, CollapsibleGroupBox
```

Also add the GPU availability flags that the method uses directly — check the top of `main_window.py` (around lines 1–50) to find where these are imported, then add them to `detection_panel.py`:

```python
# These are set at module level in main_window.py — import them here too
from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

try:
    from hydra_suite.utils.gpu_utils import TENSORRT_AVAILABLE
except ImportError:
    TENSORRT_AVAILABLE = False
```

Verify the exact import path by reading the top of `main_window.py` for how `TENSORRT_AVAILABLE` is imported there.

- [ ] **Step 2: Populate `_build_ui()` in `detection_panel.py`**

Copy the body of `setup_detection_ui` from `main_window.py` (lines 5039–6050) into `_build_ui()`. Apply these transformations:

1. Replace first line: `layout = QVBoxLayout(self.tab_detection)` → `layout = self._layout`
2. Replace all helper calls:
   - `self._create_help_label(` → `self._main_window._create_help_label(`
   - `self._set_compact_scroll_layout(` → `self._main_window._set_compact_scroll_layout(`
   - `self._set_compact_section_widget(` → `self._main_window._set_compact_section_widget(`
   - `self._remember_collapsible_state(` → `self._main_window._remember_collapsible_state(`
3. Replace handler connections (8 handlers listed above):
   - Each `.connect(self.HANDLER)` → `.connect(self._main_window.HANDLER)`

- [ ] **Step 3: Replace the tab block in `main_window.py`**

Find:

```python
        # Tab 2: Detection (Image, Method, Params)
        self.tab_detection = QWidget()
        self.setup_detection_ui()
        self.tabs.addTab(self.tab_detection, "Find Animals")
```

Replace with:

```python
        # Tab 2: Detection (Image, Method, Params)
        from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel
        self._detection_panel = DetectionPanel(main_window=self, config=self.config, parent=self)
        self.tabs.addTab(self._detection_panel, "Find Animals")
```

- [ ] **Step 4: Delete `setup_detection_ui` from `main_window.py`**

Delete the entire method `def setup_detection_ui` and its body (lines ~5038–6051).

- [ ] **Step 5: Run widget reference substitution script**

Save as `/tmp/subst_detection.py` and run:

```python
import re

with open("src/hydra_suite/trackerkit/gui/main_window.py") as f:
    content = f.read()

panel = "_detection_panel"
widgets = [
    "bg_accordion", "btn_auto_set_aspect_ratio", "btn_auto_set_body_size",
    "btn_bg_autotune", "chk_adaptive_bg", "chk_additional_dilation",
    "chk_conservative_split", "chk_dark_on_light",
    "chk_enable_aspect_ratio_filtering", "chk_enable_tensorrt",
    "chk_enable_yolo_batching", "chk_lighting_stab", "chk_show_bg",
    "chk_show_fg", "chk_show_yolo_obb", "chk_size_filtering",
    "chk_use_custom_obb_iou", "chk_yolo_seq_square_crop",
    "chk_yolo_seq_stage2_pow2_pad", "combo_detection_method", "combo_device",
    "combo_yolo_batch_mode", "combo_yolo_crop_obb_model", "combo_yolo_detect_model",
    "combo_yolo_model", "combo_yolo_obb_mode", "g_gpu_accel", "g_img",
    "g_overlays_bg", "g_overlays_yolo", "label_body_size_info",
    "label_brightness_val", "label_contrast_val", "label_detection_stats",
    "label_gamma_val", "lbl_obb_mode_warning", "lbl_tensorrt_batch",
    "lbl_yolo_batch_mode", "lbl_yolo_batch_size", "line_yolo_classes",
    "slider_brightness", "slider_contrast", "slider_gamma", "spin_bg_learning",
    "spin_bg_prime", "spin_conservative_erode", "spin_conservative_kernel",
    "spin_dilation_iterations", "spin_dilation_kernel_size",
    "spin_lighting_median", "spin_lighting_smooth", "spin_max_ar_multiplier",
    "spin_max_contour_multiplier", "spin_max_object_size", "spin_min_ar_multiplier",
    "spin_min_contour", "spin_min_object_size", "spin_morph_size",
    "spin_reference_aspect_ratio", "spin_reference_body_size",
    "spin_tensorrt_batch", "spin_threshold", "spin_yolo_batch_size",
    "spin_yolo_confidence", "spin_yolo_iou", "spin_yolo_seq_crop_pad",
    "spin_yolo_seq_detect_conf", "spin_yolo_seq_min_crop_px",
    "spin_yolo_seq_stage2_imgsz", "stack_detection", "yolo_group",
    "yolo_seq_advanced", "yolo_seq_advanced_content",
]

count = 0
for w in widgets:
    new, n = re.subn(
        r"\bself\." + re.escape(w) + r"(?![a-zA-Z0-9_])",
        f"self.{panel}.{w}",
        content,
    )
    content = new
    count += n

with open("src/hydra_suite/trackerkit/gui/main_window.py", "w") as f:
    f.write(content)

print(f"Done — {count} substitutions made")
```

```bash
conda run -n hydra-mps python /tmp/subst_detection.py
```

- [ ] **Step 6: Add DetectionPanel smoke test**

Append to `tests/test_trackerkit_panels_smoke.py`:

```python
def test_detection_panel_instantiated(main_window):
    """DetectionPanel is accessible and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel

    assert hasattr(main_window, "_detection_panel")
    assert isinstance(main_window._detection_panel, DetectionPanel)
    assert hasattr(main_window._detection_panel, "combo_detection_method")
    assert hasattr(main_window._detection_panel, "stack_detection")
```

- [ ] **Step 7: Import smoke test + full test suite**

```bash
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"
conda run -n hydra-mps python -m pytest tests/ -q --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -5
```

Expected: `ok` then pass count ≥ baseline.

- [ ] **Step 8: Verify line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 16,050 lines.

- [ ] **Step 9: Commit**

```bash
conda run -n hydra-mps git add src/hydra_suite/trackerkit/gui/panels/detection_panel.py \
  src/hydra_suite/trackerkit/gui/main_window.py \
  tests/test_trackerkit_panels_smoke.py
conda run -n hydra-mps git commit -m "refactor: extract DetectionPanel from trackerkit MainWindow"
```

---

### Task 9: Final Verification

**Files:** None — verification only.

- [ ] **Step 1: Confirm all 6 setup methods are gone**

```bash
grep -n "def setup_setup_ui\|def setup_detection_ui\|def setup_tracking_ui\|def setup_data_ui\|def setup_dataset_ui\|def setup_individual_analysis_ui" \
  src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: no output (all 6 methods deleted).

- [ ] **Step 2: Confirm all 6 tab_X attributes are gone**

```bash
grep -n "self\.tab_setup\|self\.tab_detection\|self\.tab_individual\|self\.tab_tracking\|self\.tab_data\|self\.tab_dataset" \
  src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: no output.

- [ ] **Step 3: Confirm all 6 panels are instantiated**

```bash
grep -n "_setup_panel\|_detection_panel\|_tracking_panel\|_postprocess_panel\|_identity_panel\|_dataset_panel" \
  src/hydra_suite/trackerkit/gui/main_window.py | grep "= SetupPanel\|= DetectionPanel\|= TrackingPanel\|= PostProcessPanel\|= IdentityPanel\|= DatasetPanel"
```

Expected: 6 lines (one instantiation per panel).

- [ ] **Step 4: Final line count**

```bash
wc -l src/hydra_suite/trackerkit/gui/main_window.py
```

Expected: ≤ 16,050 lines.

- [ ] **Step 5: Full test suite**

```bash
conda run -n hydra-mps python -m pytest tests/ -v --tb=short -m "not benchmark" \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py 2>&1 | tail -10
```

Expected: pass count ≥ baseline (from Task 1 Step 2). Zero new failures.

- [ ] **Step 6: Completion marker commit**

```bash
conda run -n hydra-mps git commit --allow-empty -m "chore: Sprint 2 panel extraction complete — main_window.py reduced to <N> lines"
```

Replace `<N>` with the actual line count from Step 4.
