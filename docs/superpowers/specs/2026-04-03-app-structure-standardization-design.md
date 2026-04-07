# App Structure Standardization Design

## Goal

Rename `hydra_suite.tracker` to `hydra_suite.trackerkit` and standardize the internal directory structure of all 6 kit apps to a consistent pattern.

## Target Structure (all apps)

```
{appkit}/
    __init__.py
    app.py              <- entry point, contains main()
    core/               <- kept where it already exists
    gui/
        __init__.py
        main_window.py
        dialogs/
        widgets/
```

## Per-App Changes

### 1. tracker -> trackerkit
- Rename src/hydra_suite/tracker/ -> src/hydra_suite/trackerkit/
- Flatten app/launcher.py -> app.py (collapse app/ subdirectory)
- Update all internal imports

### 2. posekit
- Rename src/hydra_suite/posekit/ui/ -> src/hydra_suite/posekit/gui/
- Update __init__.py lazy import: from .ui.main -> from .gui.main

### 3. classkit
- Rename gui/mainwindow.py -> gui/main_window.py
- Update internal imports

### 4. detectkit
- Rename src/hydra_suite/detectkit/ui/ -> src/hydra_suite/detectkit/gui/
- panels/ moves with it

### 5. refinekit
- No changes (already correct)

### 6. filterkit
- Split gui.py: main() -> app.py, FilterKitWindow -> gui/main_window.py
- gui/__init__.py re-exports FilterKitWindow for backward compat
- Update filterkit/main.py

## Cross-Package References

pyproject.toml:
  hydra_suite.tracker.app.launcher:main -> hydra_suite.trackerkit.app:main
  hydra_suite.posekit.ui.main:main -> hydra_suite.posekit.gui.main:main
  hydra_suite.filterkit.gui:main -> hydra_suite.filterkit.app:main

src/hydra_suite/__init__.py:
  .tracker.app.launcher -> .trackerkit.app
  .tracker.gui.main_window -> .trackerkit.gui.main_window

src/hydra_suite/launcher/app.py (3 entry strings): same as pyproject

detectkit panels (5 occurrences):
  hydra_suite.tracker.gui.* -> hydra_suite.trackerkit.gui.*

trackerkit/gui/main_window.py:
  hydra_suite.posekit.ui.* -> hydra_suite.posekit.gui.*

filterkit/main.py:
  from hydra_suite.filterkit.gui import main -> from hydra_suite.filterkit.app import main

tests/ (5 files):
  hydra_suite.tracker.* -> hydra_suite.trackerkit.*

CLAUDE.md: all hydra_suite.tracker refs updated
