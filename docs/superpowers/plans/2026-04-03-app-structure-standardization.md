# App Structure Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Standardize the internal directory structure of all 6 hydra-suite apps so each follows the same `<app>/gui/main_window.py` pattern, and rename `tracker/` → `trackerkit/`.

**Architecture:** Each app package uses `gui/main_window.py` as the main window, `app.py` (or `app/`) as the launcher, and `main.py` as the `__main__` entry. Internal cross-app imports and pyproject.toml entry points are updated to match.

**Tech Stack:** Python, PySide6, git mv (history preservation), pyproject.toml

---

### Task 1: Rename tracker/ to trackerkit/ and flatten launcher

**Files:**
- Rename: src/hydra_suite/tracker/ to src/hydra_suite/trackerkit/
- Move: src/hydra_suite/trackerkit/app/launcher.py to src/hydra_suite/trackerkit/app.py
- Modify: pyproject.toml (entry point)
- Modify: src/hydra_suite/launcher/app.py (entry string)
- Modify: src/hydra_suite/detectkit/ui/panels/history_panel.py
- Modify: src/hydra_suite/detectkit/ui/panels/evaluation_panel.py
- Modify: src/hydra_suite/detectkit/ui/panels/training_panel.py
- Modify: src/hydra_suite/trackerkit/gui/main_window.py (self-imports)
- Modify: src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py
- Modify: tests/test_loss_plot_widget.py
- Modify: tests/test_model_test_dialog.py
- Modify: tests/test_main_window_config_persistence.py
- Modify: tests/test_run_history.py
- Modify: tests/test_preview_background_cache.py

### Task 2: Rename posekit/ui/ to posekit/gui/

**Files:**
- Rename: src/hydra_suite/posekit/ui/ to src/hydra_suite/posekit/gui/
- Modify: src/hydra_suite/posekit/gui/main_window.py (self-imports)
- Modify: src/hydra_suite/launcher/app.py (entry string)
- Modify: pyproject.toml (entry point)
- Modify: src/hydra_suite/trackerkit/gui/main_window.py (posekit.ui references)

### Task 3: Rename classkit/gui/mainwindow.py to main_window.py

**Files:**
- Rename: src/hydra_suite/classkit/gui/mainwindow.py to src/hydra_suite/classkit/gui/main_window.py
- Modify: src/hydra_suite/classkit/gui/__init__.py (update import if needed)
- Modify: src/hydra_suite/classkit/app.py (import if needed)

### Task 4: Rename detectkit/ui/ to detectkit/gui/

**Files:**
- Rename: src/hydra_suite/detectkit/ui/ to src/hydra_suite/detectkit/gui/
- Modify: src/hydra_suite/detectkit/app.py (import)
- Modify: src/hydra_suite/detectkit/gui/main_window.py (self-imports)
- Modify: tests/test_detectkit_canvas.py
- Modify: tests/test_detectkit_project.py
- Modify: tests/test_detectkit_skeleton.py
- Modify: tests/test_detectkit_dataset_panel.py

### Task 5: Split filterkit/gui.py into filterkit/gui/ package

**Files:**
- Create: src/hydra_suite/filterkit/gui/__init__.py
- Create: src/hydra_suite/filterkit/gui/main_window.py (all classes)
- Create: src/hydra_suite/filterkit/app.py (main() function)
- Delete: src/hydra_suite/filterkit/gui.py
- Modify: src/hydra_suite/filterkit/main.py (update import)
- Modify: pyproject.toml (entry point)

### Task 6: Update CLAUDE.md, run full tests, lint

**Files:**
- Modify: CLAUDE.md (tracker -> trackerkit in Architecture table)
