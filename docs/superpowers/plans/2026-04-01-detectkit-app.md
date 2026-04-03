# DetectKit Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create DetectKit, an independent application for OBB detection model training -- dataset curation, visualization, X-AnyLabeling integration, training, and evaluation -- without requiring MAT. Replaces the Training Center dialog in MAT entirely.

**Architecture:** DetectKit follows the ClassKit standalone-workspace model (own project directory, not embedded in datasets) with PoseKit's UI patterns (three-panel layout, QGraphicsView canvas, project persistence). Left panel: dataset list + image list + X-AnyLabeling launch. Center panel: read-only OBB image viewer with zoom/pan/scroll. Right panel: training config, augmentation, run controls, loss plot, history. The existing `hydra_suite.training` framework is reused as-is.

**Tech Stack:** Python 3.10+, PySide6 (QGraphicsView for canvas), cv2, numpy. Reuses `hydra_suite.training`, `hydra_suite.paths`, `hydra_suite.integrations.xanylabeling`.

---

## Scope: Two-Phase Delivery

This plan covers **Phase 1: App Skeleton + Core Panels**. This gets DetectKit fully functional.

Phase 2 (separate plan, later): advanced features (active learning, model comparison, annotation editing).

---

## File Map

```
src/hydra_suite/detectkit/
    __init__.py                          # Package marker
    app.py                               # Entry point (QApplication + startup)
    ui/
        __init__.py
        models.py                        # Project, OBBSource dataclasses
        project.py                       # Project lifecycle (create/open/save/recent)
        constants.py                     # Shared constants
        utils.py                         # UI settings persistence, helpers
        main_window.py                   # QMainWindow: three-panel layout
        canvas.py                        # OBBCanvas(QGraphicsView): image + OBB viewer
        panels/
            __init__.py
            dataset_panel.py             # Left panel: sources + image list
            training_panel.py            # Right panel: config + augmentation + run
            evaluation_panel.py          # Right sub-panel: quick test + analysis
            history_panel.py             # Right sub-panel: run history browser
```

| Task | Files Created | Files Modified |
|------|--------------|----------------|
| 1. Package skeleton + entry point | `detectkit/__init__.py`, `detectkit/app.py`, `detectkit/ui/__init__.py`, `detectkit/ui/constants.py` | `pyproject.toml` |
| 2. Project model + persistence | `detectkit/ui/models.py`, `detectkit/ui/project.py`, `detectkit/ui/utils.py` | |
| 3. OBB canvas viewer | `detectkit/ui/canvas.py` | |
| 4. Main window shell | `detectkit/ui/main_window.py`, `detectkit/ui/panels/__init__.py` | |
| 5. Dataset panel (left) | `detectkit/ui/panels/dataset_panel.py` | |
| 6. Training panel (right) | `detectkit/ui/panels/training_panel.py` | |
| 7. Evaluation + history panels | `detectkit/ui/panels/evaluation_panel.py`, `detectkit/ui/panels/history_panel.py` | |
| 8. Remove Training Center from MAT | | `mat/gui/main_window.py`, `pyproject.toml` |

All paths relative to `src/hydra_suite/`. Tests in `tests/test_detectkit_*.py`.

---

## Task 1: Package Skeleton + Entry Point

**Why:** Get `detectkit` launchable as a console command that shows an empty window.

**Files:**
- Create: `src/hydra_suite/detectkit/__init__.py`
- Create: `src/hydra_suite/detectkit/app.py`
- Create: `src/hydra_suite/detectkit/ui/__init__.py`
- Create: `src/hydra_suite/detectkit/ui/panels/__init__.py`
- Create: `src/hydra_suite/detectkit/ui/constants.py`
- Modify: `pyproject.toml`
- Test: `tests/test_detectkit_skeleton.py`

- [ ] **Step 1: Create package directory structure**

```bash
mkdir -p src/hydra_suite/detectkit/ui/panels
```

- [ ] **Step 2: Create __init__.py files**

```python
# src/hydra_suite/detectkit/__init__.py
"""DetectKit -- OBB detection model training and dataset curation."""
```

```python
# src/hydra_suite/detectkit/ui/__init__.py
```

```python
# src/hydra_suite/detectkit/ui/panels/__init__.py
```

- [ ] **Step 3: Create constants.py**

```python
# src/hydra_suite/detectkit/ui/constants.py
"""DetectKit constants."""
from __future__ import annotations

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

DEFAULT_PROJECT_DIR_NAME = "detectkit_project"
DEFAULT_PROJECT_FILENAME = "detectkit_project.json"

# Default location for new projects (user picks during creation)
DEFAULT_PROJECTS_ROOT_NAME = "DetectKit"

# OBB label format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 fields, normalized)
OBB_LABEL_FIELDS = 9

# UI
DEFAULT_OBB_LINE_WIDTH = 2.0
DEFAULT_OBB_FONT_SIZE = 10
CANVAS_BG_COLOR = "#121212"

# Dataset analysis
MAX_ANALYSIS_IMAGES = 500
```

- [ ] **Step 4: Create app.py entry point**

```python
# src/hydra_suite/detectkit/app.py
"""DetectKit application entry point."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication


def main() -> None:
    """Launch the DetectKit application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("DetectKit")
    app.setApplicationDisplayName("DetectKit")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("detectkit")

    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("detectkit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    from hydra_suite.detectkit.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1600, 1000)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create minimal main window placeholder** (will be replaced in Task 4)

```python
# src/hydra_suite/detectkit/ui/main_window.py
"""DetectKit main window -- placeholder for Task 4."""
from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DetectKit")
        self.setCentralWidget(QLabel("DetectKit -- coming soon"))
```

- [ ] **Step 6: Register console script in pyproject.toml**

In the `[project.scripts]` section (after the `afterhours` line), add:
```toml
detectkit = "hydra_suite.detectkit.app:main"
```

- [ ] **Step 7: Write test**

```python
# tests/test_detectkit_skeleton.py
"""Verify DetectKit package structure and imports."""
from __future__ import annotations


def test_detectkit_package_imports():
    import hydra_suite.detectkit
    from hydra_suite.detectkit.ui.constants import (
        DEFAULT_PROJECT_FILENAME,
        IMG_EXTS,
        OBB_LABEL_FIELDS,
    )
    assert OBB_LABEL_FIELDS == 9
    assert ".jpg" in IMG_EXTS
    assert DEFAULT_PROJECT_FILENAME == "detectkit_project.json"
```

- [ ] **Step 8: Run test**

Run: `python -m pytest tests/test_detectkit_skeleton.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/detectkit/ tests/test_detectkit_skeleton.py pyproject.toml
git commit -m "feat(detectkit): create package skeleton and entry point"
```

---

## Task 2: Project Model + Persistence

**Why:** DetectKit needs persistent projects that store source dataset paths, class names, training settings, and session state. Follows ClassKit standalone-workspace pattern.

**Files:**
- Create: `src/hydra_suite/detectkit/ui/models.py`
- Create: `src/hydra_suite/detectkit/ui/project.py`
- Create: `src/hydra_suite/detectkit/ui/utils.py`
- Test: `tests/test_detectkit_project.py`

- [ ] **Step 1: Write test for project model**

```python
# tests/test_detectkit_project.py
"""Tests for DetectKit project model and persistence."""
from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.detectkit.ui.models import DetectKitProject, OBBSource


def test_project_roundtrip(tmp_path: Path):
    proj = DetectKitProject(
        project_dir=tmp_path,
        class_name="ant",
        sources=[
            OBBSource(path=str(tmp_path / "ds1"), name="ds1"),
            OBBSource(path=str(tmp_path / "ds2"), name="ds2"),
        ],
    )
    proj_file = tmp_path / "detectkit_project.json"
    proj.save(proj_file)
    assert proj_file.exists()

    loaded = DetectKitProject.load(proj_file)
    assert loaded.class_name == "ant"
    assert len(loaded.sources) == 2
    assert loaded.sources[0].name == "ds1"


def test_project_defaults():
    proj = DetectKitProject(project_dir=Path("/tmp/test"))
    assert proj.class_name == "object"
    assert proj.sources == []
    assert proj.split_train == 0.8
    assert proj.split_val == 0.2
    assert proj.seed == 42


def test_obb_source_roundtrip():
    src = OBBSource(path="/data/obb_ds", name="my_dataset")
    d = src.to_dict()
    restored = OBBSource.from_dict(d)
    assert restored.path == "/data/obb_ds"
    assert restored.name == "my_dataset"
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_detectkit_project.py -v`
Expected: FAIL (models module does not exist)

- [ ] **Step 3: Create models.py**

`DetectKitProject` dataclass with all training settings as fields (class_name, sources, split, seed, dedup, crop params, per-role imgsz, base models, hyperparams, augmentation, roles, device, publish, session state). Methods: `to_dict()`, `save(path)`, `load(path)` static.

`OBBSource` dataclass: path, name, validated. Methods: `to_dict()`, `from_dict()` static.

The `load()` method should iterate over known field names, type-cast based on the existing default type (bool/int/float/str), and set via `setattr`. Sources are deserialized from the "sources" list.

- [ ] **Step 4: Create project.py**

Functions for project lifecycle:
- `get_recent_projects_path()` -- returns path to `~/.local/share/hydra-suite/detectkit/recent_projects.json`
- `load_recent_projects()` / `save_recent_projects(paths)` / `add_to_recent(project_dir)` -- manage recent projects list (max 20)
- `project_file_path(project_dir)` -- returns `project_dir / DEFAULT_PROJECT_FILENAME`
- `open_project(project_dir)` -- load existing project, add to recent
- `create_project(project_dir, class_name)` -- create new project with defaults, save, add to recent
- `save_project(proj)` -- save project to its directory

Uses `hydra_suite.paths._user_data_dir()` for the recent projects storage path, with fallback to `~/.detectkit/`.

- [ ] **Step 5: Create utils.py**

Utility functions:
- `get_ui_settings_path()` / `load_ui_settings()` / `save_ui_settings()` -- UI settings persistence
- `list_images_in_source(source_path)` -- list all image files in an OBB source dataset (checks `images/` subdir first, falls back to root)
- `find_label_for_image(image_path, source_path)` -- find matching OBB label by relative path structure or stem match
- `parse_obb_label(label_path, img_w, img_h)` -- parse OBB label file into list of `{"class_id": int, "polygon_px": [(x,y), ...]}` dicts

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_detectkit_project.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/detectkit/ui/models.py src/hydra_suite/detectkit/ui/project.py src/hydra_suite/detectkit/ui/utils.py tests/test_detectkit_project.py
git commit -m "feat(detectkit): add project model and persistence layer"
```

---

## Task 3: OBB Canvas Viewer

**Why:** The center panel needs a read-only image viewer that draws OBB polygons with zoom, pan, and scroll. Based on PoseKit's QGraphicsView pattern.

**Files:**
- Create: `src/hydra_suite/detectkit/ui/canvas.py`
- Test: `tests/test_detectkit_canvas.py`

- [ ] **Step 1: Write test for OBB label parsing** (canvas rendering is tested manually)

```python
# tests/test_detectkit_canvas.py
"""Tests for OBB label parsing (canvas drawing tested manually)."""
from __future__ import annotations

from pathlib import Path

from hydra_suite.detectkit.ui.utils import parse_obb_label


def test_parse_obb_label(tmp_path: Path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    assert len(dets[0]["polygon_px"]) == 4
    assert abs(dets[0]["polygon_px"][0][0] - 10.0) < 0.1
    assert abs(dets[0]["polygon_px"][0][1] - 20.0) < 0.1


def test_parse_obb_label_empty(tmp_path: Path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert dets == []


def test_parse_obb_label_invalid_line(tmp_path: Path):
    lbl = tmp_path / "bad.txt"
    lbl.write_text("0 0.1 0.2\n0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1  # skips the bad line
```

- [ ] **Step 2: Run test to verify pass** (parse_obb_label already defined in utils.py from Task 2)

Run: `python -m pytest tests/test_detectkit_canvas.py -v`
Expected: PASS

- [ ] **Step 3: Create canvas.py**

`OBBCanvas(QGraphicsView)` class with:
- `QGraphicsScene` with a `QGraphicsPixmapItem` for the image
- `load_image(image_path)` -- load and display BGR image via cv2
- `set_image_array(bgr)` -- display a numpy array
- `set_detections(detections, class_name)` -- draw OBB polygons using `QGraphicsPolygonItem` and class labels using `QGraphicsTextItem`, with per-class colors from an 8-color palette
- `clear_detections()` / `clear_all()` -- remove overlays
- `fit_in_view()` -- fit image to viewport
- `wheelEvent` -- zoom (factor 1.15 per scroll, clamped to 0.05-30.0)
- `mousePressEvent/mouseMoveEvent/mouseReleaseEvent` -- pan via middle-click or Ctrl+left-click
- `resizeEvent` -- auto fit-in-view

Use the same dark background as PoseKit (`#121212`). Pen width from `DEFAULT_OBB_LINE_WIDTH` constant.

- [ ] **Step 4: Run all tests and commit**

Run: `python -m pytest tests/test_detectkit_canvas.py tests/test_detectkit_project.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/detectkit/ui/canvas.py tests/test_detectkit_canvas.py
git commit -m "feat(detectkit): add read-only OBB canvas viewer"
```

---

## Task 4: Main Window Shell

**Why:** Replace the placeholder main window with the real three-panel layout: left (dataset), center (canvas), right (training). Includes welcome page, project open/create, and menu bar.

**Files:**
- Replace: `src/hydra_suite/detectkit/ui/main_window.py`
- Create: stub panels (empty QWidgets with `set_project` and `collect_state` method signatures)

- [ ] **Step 1: Create stub panels for compilation**

Create minimal stub panels at `dataset_panel.py`, `training_panel.py`, `evaluation_panel.py`, `history_panel.py`. Each should be a QWidget with a QVBoxLayout containing a placeholder QLabel and empty `set_project(proj, main_window)` and `collect_state(proj)` methods.

- [ ] **Step 2: Replace main_window.py with full implementation**

The main window should have:

**Menu bar:** File menu with New Project, Open Project, Recent Projects submenu, Save, Quit

**QStackedWidget** switching between:
- **Welcome page** (index 0): Logo (detectkit.svg via `get_brand_icon_bytes`), app title "DetectKit", subtitle, "New Project" button, "Open Project" button, recent projects QListWidget (double-click to open)
- **Main workspace** (index 1): QSplitter with three sections

**Main workspace layout:**
- Left: `DatasetPanel` (min 280px, max 450px)
- Center: `OBBCanvas` (stretches)
- Right: `QTabWidget` with tabs "Training", "Evaluation", "History" (min 380px, max 550px)

**Project lifecycle methods:**
- `new_project()` -- QFileDialog for directory + QInputDialog for class name
- `open_project_dialog()` -- QFileDialog for directory
- `_load_project(proj)` -- wire project into all panels via `panel.set_project(proj, self)`, switch to workspace view
- `_save_current_project()` -- collect state from panels, call `save_project()`
- `show_image(source_path, image_path)` -- load image + parse OBB labels into canvas
- `closeEvent` -- auto-save on close

**Public accessors:** `project()`, `canvas()`

**Dark stylesheet** matching PoseKit: `#1e1e1e` background, `#252526` widgets, `#0e639c` buttons, `#007acc` status bar.

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/hydra_suite/detectkit/ui/main_window.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/detectkit/ui/main_window.py src/hydra_suite/detectkit/ui/panels/
git commit -m "feat(detectkit): add main window with three-panel layout and welcome page"
```

---

## Task 5: Dataset Panel (Left)

**Why:** The left panel manages source datasets: add/remove sources, save/load dataset lists, browse images, launch X-AnyLabeling. Clicking an image shows it in the center canvas with OBB overlays.

**Files:**
- Replace: `src/hydra_suite/detectkit/ui/panels/dataset_panel.py`
- Test: `tests/test_detectkit_dataset_panel.py`

- [ ] **Step 1: Write test**

```python
# tests/test_detectkit_dataset_panel.py
"""Tests for dataset panel utilities."""
from __future__ import annotations

from pathlib import Path

from hydra_suite.detectkit.ui.utils import find_label_for_image, list_images_in_source


def test_list_images_in_source_with_images_dir(tmp_path: Path):
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").write_text("fake")
    (img_dir / "b.png").write_text("fake")
    (img_dir / "c.txt").write_text("not an image")
    images = list_images_in_source(str(tmp_path))
    assert len(images) == 2


def test_find_label_for_image(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    img = tmp_path / "images" / "train" / "frame001.jpg"
    img.write_text("fake")
    lbl = tmp_path / "labels" / "train" / "frame001.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8")
    result = find_label_for_image(img, str(tmp_path))
    assert result is not None
    assert result.name == "frame001.txt"
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_detectkit_dataset_panel.py -v`
Expected: PASS (utilities already implemented in Task 2)

- [ ] **Step 3: Implement full dataset panel**

The panel widget contains, top to bottom:

1. **Source list** (QListWidget): Shows dataset source names/paths. Each item stores source path as UserRole data.

2. **Source action buttons** (QHBoxLayout):
   - "Add Sources..." -- multi-directory picker (reuse the non-native QFileDialog pattern from train_yolo_dialog `_get_multiple_dirs`)
   - "Remove" -- remove selected source from list and project
   - "Save List..." / "Load List..." -- JSON file with source paths (same format as train_yolo_dialog: `[{"source_type": "obb", "path": "..."}]`)

3. **Image list** (QListWidget): Shows image filenames within the selected source. Populated when source selection changes. Item data stores full image path.

4. **"Open in X-AnyLabeling" button**: Launches xanylabeling in external terminal for the selected source directory. Reuse the cross-platform terminal launch pattern from MAT's `_open_in_xanylabeling()` method. On button click:
   - Get the selected source directory
   - Check for `classes.txt` (create if missing using project class_name)
   - Run xlabel2yolo conversion if JSON annotations found in `images/`
   - Launch `xanylabeling --filename ./images` in terminal

5. **Validation**: When a new source is added, auto-validate it using `inspect_obb_or_detect_dataset()`. If X-AnyLabeling JSON files are found alongside images but no YOLO labels exist, auto-run `convert_project()` to generate them.

Key behaviors:
- `set_project(proj, main_window)`: Populate source list from `proj.sources`, store main_window reference
- `collect_state(proj)`: Write current sources back to `proj.sources`
- Source item clicked -> populate image list with `list_images_in_source()`
- Image item clicked -> call `main_window.show_image(source_path, image_path)`
- After X-AnyLabeling closes (or on Refresh), re-validate sources to pick up new labels

- [ ] **Step 4: Run all tests and commit**

Run: `python -m pytest tests/test_detectkit_dataset_panel.py tests/test_detectkit_project.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/detectkit/ui/panels/dataset_panel.py tests/test_detectkit_dataset_panel.py
git commit -m "feat(detectkit): implement dataset panel with source management and image browser"
```

---

## Task 6: Training Panel (Right)

**Why:** The right panel contains all training configuration -- migrated from the Training Center dialog. This is the largest panel.

**Files:**
- Replace: `src/hydra_suite/detectkit/ui/panels/training_panel.py`

- [ ] **Step 1: Implement training panel**

Organized as a scrollable vertical layout with collapsible QGroupBox sections:

1. **Roles** (checkboxes: obb_direct, seq_detect, seq_crop_obb)
2. **Config** (class name QLineEdit, workspace browse, split spinners, seed, dedup checkbox, crop derivation spinners, device combo -- editable for multi-GPU)
3. **Hyperparameters** (epochs, batch + auto checkbox, lr0, patience, workers, cache, per-role imgsz spinners)
4. **Base Models** (editable combos for each role: yolo26n/s/m/l/x-obb.pt etc.)
5. **Augmentation** (checkable QGroupBox with fliplr, flipud, degrees, mosaic, mixup, hsv_h/s/v spinners)
6. **Publish** (species, model tag, auto-import, auto-select checkboxes)
7. **Run Controls** row of buttons: Build, Train, Stop, Resume, Detach, Quick Test, Run History, Save/Load Config
8. **Loss Plot** (import and embed `LossPlotWidget` from `hydra_suite.tracker.gui.widgets.loss_plot_widget`)
9. **Log** (read-only QTextEdit)

Key behaviors:
- `set_project(proj, main_window)`: Populate all widgets from project fields. Store references.
- `collect_state(proj)`: Write all widget values back to project fields.
- Training execution: Uses `RoleTrainingWorker` and `TrainingOrchestrator` from `hydra_suite.training` -- same backend as the Training Center dialog.
- Sources for training come from the project's source list (accessed via `main_window.project().sources`).
- Build/Train/Stop/Resume/Detach: Same logic as train_yolo_dialog methods, adapted to read from project and widgets.
- Quick Test: Opens `ModelTestDialog`.
- Run History: Opens `RunHistoryDialog`.
- Save/Load Config: Collects all widgets to JSON, same format as train_yolo_dialog's config export.
- Loss plot: Fed by `_append_log` which calls `self.loss_plot.ingest_log_line()`.
- Analyze/Preview: Runs analysis and sends crop previews to `main_window.canvas()` instead of a popup.

The panel must NOT depend on MAT's main window. No `_try_auto_select_parent_models`. No `conda_envs` parameter.

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/hydra_suite/detectkit/ui/panels/training_panel.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/detectkit/ui/panels/training_panel.py
git commit -m "feat(detectkit): implement training panel with full config and run controls"
```

---

## Task 7: Evaluation + History Panels

**Why:** The evaluation panel shows dataset analysis results and quick-test capability. The history panel shows the run history table. Both are thin wrappers around existing code.

**Files:**
- Replace: `src/hydra_suite/detectkit/ui/panels/evaluation_panel.py`
- Replace: `src/hydra_suite/detectkit/ui/panels/history_panel.py`

- [ ] **Step 1: Implement evaluation panel**

Contains:
1. **Analyze Dataset** button -- runs `analyze_obb_sizes()` + `format_size_analysis()` from `hydra_suite.training.dataset_inspector`, displays results in a QTextEdit, and shows crop previews in the center canvas via `main_window.canvas()`
2. **Analysis results** (read-only QTextEdit showing size stats and warnings)
3. **Quick Test** button -- opens `ModelTestDialog` from `hydra_suite.tracker.gui.dialogs.model_test_dialog`
4. **Validation report** (read-only QTextEdit)

Key behavior:
- `set_project(proj, main_window)`: Store references
- Analyze button: Collect sources from `main_window.project().sources`, inspect each, run `analyze_obb_sizes`, format report, display in text area. For crop preview: generate sample crops and display them in the canvas using `main_window.canvas().set_image_array()` with a mosaic/grid of crops.

- [ ] **Step 2: Implement history panel**

Embeds `load_run_history` from `hydra_suite.tracker.gui.dialogs.run_history_dialog` directly as a panel widget.

Contains:
1. **Refresh** button
2. **QTableWidget** with columns: Run ID, Role, Status, Started, Base Model, Epochs (newest first, color-coded status)
3. **Detail view** (read-only QTextEdit) showing full JSON of selected run

Key behavior:
- `set_project(proj, main_window)`: Load registry via `get_registry_path()` and populate table
- Row click: Show full run JSON in detail view
- Refresh button: Reload registry

- [ ] **Step 3: Verify syntax and commit**

Run: `python -c "import ast; ast.parse(open('src/hydra_suite/detectkit/ui/panels/evaluation_panel.py').read()); print('OK')"` and same for history_panel.py

```bash
git add src/hydra_suite/detectkit/ui/panels/evaluation_panel.py src/hydra_suite/detectkit/ui/panels/history_panel.py
git commit -m "feat(detectkit): implement evaluation and history panels"
```

---

## Task 8: Remove Training Center from MAT

**Why:** DetectKit replaces the Training Center entirely. Remove the launch button from MAT.

**Files:**
- Modify: `src/hydra_suite/mat/gui/main_window.py`

- [ ] **Step 1: Find and remove Training Center launch code**

Search `main_window.py` for references to `TrainYoloDialog` or `train_yolo_dialog`. Remove:
- The import of `TrainYoloDialog` (if any top-level import exists)
- The button that opens the training dialog (search for "Training Center" or "Train" button text)
- The `clicked.connect` handler for that button
- The method that instantiates `TrainYoloDialog` (search for `TrainYoloDialog(`)

Do NOT delete `train_yolo_dialog.py` itself -- it stays as reference code and is still importable.

- [ ] **Step 2: Verify MAT still parses**

Run: `python -c "import ast; ast.parse(open('src/hydra_suite/mat/gui/main_window.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/mat/gui/main_window.py
git commit -m "refactor(mat): remove Training Center dialog, replaced by DetectKit app"
```

---

## Dependency Order

Tasks must be executed in strict order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8.

- Task 1 establishes the package and entry point
- Task 2 provides the project model that all panels use
- Task 3 provides the canvas that the main window and panels reference
- Task 4 provides the main window shell that panels plug into
- Tasks 5-7 fill in the panel implementations
- Task 8 removes the old Training Center from MAT (only after DetectKit is functional)

**Priority tiers:**
- Tasks 1-4: **Critical** -- gets the app launchable with the three-panel skeleton
- Tasks 5-6: **High** -- the two main functional panels (dataset management + training)
- Task 7: **Medium** -- evaluation and history (existing dialogs can be used standalone meanwhile)
- Task 8: **Low** -- cleanup, do last
