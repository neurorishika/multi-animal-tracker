# Pip-Publishable Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hydra-suite` installable via `pip install hydra-suite` with all assets bundled, user-writable data dirs, and proper dependency declarations — while preserving the existing conda+Makefile developer workflow.

**Architecture:** Create a `hydra_suite.resources` sub-package that bundles read-only assets (brand icons, default configs, skeletons) using `importlib.resources`. User-writable data (models, training runs, advanced config) moves to platform-appropriate directories via `platformdirs`. A central `hydra_suite.paths` module replaces all 24 `parents[N]` path-resolution patterns. `pyproject.toml` gains full dependency declarations.

**Tech Stack:** Python 3.11+ `importlib.resources.files()`, `platformdirs>=3.0`, setuptools `package-data`, PySide6 `QPixmap.loadFromData()`

---

## File Structure

### New files to create

| File | Responsibility |
|------|---------------|
| `src/hydra_suite/paths.py` | Central path resolution: bundled assets via `importlib.resources`, user dirs via `platformdirs` |
| `src/hydra_suite/resources/__init__.py` | Package marker for importlib.resources |
| `src/hydra_suite/resources/brand/__init__.py` | Package marker |
| `src/hydra_suite/resources/brand/*.svg` | Moved from `src/brand/` |
| `src/hydra_suite/resources/brand/*.png` | Moved from `src/brand/` |
| `src/hydra_suite/resources/configs/__init__.py` | Package marker |
| `src/hydra_suite/resources/configs/default.json` | Copied from `configs/default.json` |
| `src/hydra_suite/resources/configs/ooceraea_biroi.json` | Copied from `configs/ooceraea_biroi.json` |
| `src/hydra_suite/resources/configs/skeletons/__init__.py` | Package marker |
| `src/hydra_suite/resources/configs/skeletons/ooceraea_biroi.json` | Copied from `configs/skeletons/ooceraea_biroi.json` |
| `tests/test_paths.py` | Tests for the paths module |
| `tests/test_packaging.py` | Tests that package-data is correctly included |

### Files to modify

| File | Change |
|------|--------|
| `src/hydra_suite/paths.py` | NEW — all path logic lives here |
| `src/hydra_suite/__init__.py` | Fix placeholder metadata, use `importlib.metadata` for version |
| `pyproject.toml` | Add `dependencies`, `optional-dependencies`, `package-data`, remove unused `setuptools_scm` |
| `src/hydra_suite/datasieve/gui.py` | Replace 4x `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/afterhours/app.py` | Replace `parents[2] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/afterhours/gui/main_window.py` | Replace `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/mat/app/launcher.py` | Replace `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/mat/gui/main_window.py` | Replace 7 path patterns (brand, models, configs, advanced_config) |
| `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py` | Replace `parents[5]` with `paths.get_training_dir()` |
| `src/hydra_suite/classkit/gui/mainwindow.py` | Replace `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/posekit/ui/main_window.py` | Replace 2x `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/posekit/ui/main.py` | Replace `parents[3] / "brand"` with `paths.get_brand_icon()` |
| `src/hydra_suite/posekit/ui/utils.py` | Replace `parents[4]` skeleton dir with `paths.get_skeleton_dir()` |
| `src/hydra_suite/training/registry.py` | Replace `_project_root()` with `paths.get_training_runs_dir()` |
| `src/hydra_suite/training/model_publish.py` | Replace `_project_root()` with `paths.get_models_dir()` |

---

## Task 1: Create the `paths` module and resource sub-package structure

**Files:**
- Create: `src/hydra_suite/paths.py`
- Create: `src/hydra_suite/resources/__init__.py`
- Create: `src/hydra_suite/resources/brand/__init__.py`
- Create: `src/hydra_suite/resources/configs/__init__.py`
- Create: `src/hydra_suite/resources/configs/skeletons/__init__.py`
- Test: `tests/test_paths.py`

- [ ] **Step 1: Write the failing tests for the paths module**

```python
# tests/test_paths.py
"""Tests for hydra_suite.paths — central path resolution."""

import json
from pathlib import Path

import pytest


def test_get_brand_icon_bytes_returns_bytes():
    """Brand icon loader returns non-empty bytes for a known icon."""
    from hydra_suite.paths import get_brand_icon_bytes

    data = get_brand_icon_bytes("hydra.svg")
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert b"<svg" in data or b"<?xml" in data


def test_get_brand_icon_bytes_missing_returns_none():
    """Missing brand icon returns None instead of raising."""
    from hydra_suite.paths import get_brand_icon_bytes

    result = get_brand_icon_bytes("nonexistent_icon.svg")
    assert result is None


def test_get_default_config_returns_dict():
    """Loading a bundled default config returns a valid dict."""
    from hydra_suite.paths import get_default_config

    cfg = get_default_config("default.json")
    assert isinstance(cfg, dict)
    assert "preset_name" in cfg


def test_get_default_config_missing_returns_none():
    """Missing config returns None."""
    from hydra_suite.paths import get_default_config

    result = get_default_config("nonexistent.json")
    assert result is None


def test_get_skeleton_config_returns_dict():
    """Loading a bundled skeleton config returns a valid dict."""
    from hydra_suite.paths import get_skeleton_config

    cfg = get_skeleton_config("ooceraea_biroi.json")
    assert isinstance(cfg, dict)


def test_get_bundled_config_names_returns_list():
    """Listing bundled configs returns at least default.json."""
    from hydra_suite.paths import get_bundled_config_names

    names = get_bundled_config_names()
    assert isinstance(names, list)
    assert "default.json" in names


def test_get_bundled_skeleton_names_returns_list():
    """Listing bundled skeletons returns at least one entry."""
    from hydra_suite.paths import get_bundled_skeleton_names

    names = get_bundled_skeleton_names()
    assert isinstance(names, list)
    assert "ooceraea_biroi.json" in names


def test_user_config_dir_is_writable(tmp_path, monkeypatch):
    """User config dir is created and writable."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Re-import to pick up env change
    from hydra_suite.paths import _user_config_dir

    d = _user_config_dir()
    assert d.exists()
    assert d.is_dir()
    # Should be able to write a file
    test_file = d / "test.txt"
    test_file.write_text("ok")
    assert test_file.read_text() == "ok"


def test_user_data_dir_is_writable(tmp_path, monkeypatch):
    """User data dir is created and writable."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    from hydra_suite.paths import _user_data_dir

    d = _user_data_dir()
    assert d.exists()
    assert d.is_dir()


def test_get_models_dir_returns_path():
    """Models dir returns a Path under user data dir."""
    from hydra_suite.paths import get_models_dir

    d = get_models_dir()
    assert isinstance(d, Path)
    assert "models" in str(d)


def test_get_training_runs_dir_returns_path():
    """Training runs dir returns a Path under user data dir."""
    from hydra_suite.paths import get_training_runs_dir

    d = get_training_runs_dir()
    assert isinstance(d, Path)
    assert "training" in str(d)


def test_get_advanced_config_path_returns_path():
    """Advanced config path is under user config dir."""
    from hydra_suite.paths import get_advanced_config_path

    p = get_advanced_config_path()
    assert isinstance(p, Path)
    assert p.name == "advanced_config.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.paths'`

- [ ] **Step 3: Create empty resource package directories with `__init__.py` markers**

```python
# src/hydra_suite/resources/__init__.py
"""Bundled read-only resources for hydra-suite."""
```

```python
# src/hydra_suite/resources/brand/__init__.py
"""Brand icon assets (SVG, PNG)."""
```

```python
# src/hydra_suite/resources/configs/__init__.py
"""Bundled default configuration presets."""
```

```python
# src/hydra_suite/resources/configs/skeletons/__init__.py
"""Bundled skeleton configuration files."""
```

- [ ] **Step 4: Copy brand assets into the resource package**

```bash
cp src/brand/*.svg src/hydra_suite/resources/brand/
cp src/brand/*.png src/hydra_suite/resources/brand/
```

Verify with:
```bash
ls src/hydra_suite/resources/brand/
```

Expected: `__init__.py`, `classkit.png`, `classkit.svg`, `datasieve.png`, `datasieve.svg`, `hydra.png`, `hydra.svg`, `hydraafterhours.png`, `hydraafterhours.svg`, `posekit.png`, `posekit.svg`

- [ ] **Step 5: Copy config presets and skeletons into the resource package**

```bash
cp configs/default.json src/hydra_suite/resources/configs/
cp configs/ooceraea_biroi.json src/hydra_suite/resources/configs/
cp configs/skeletons/ooceraea_biroi.json src/hydra_suite/resources/configs/skeletons/
```

Verify with:
```bash
ls src/hydra_suite/resources/configs/
ls src/hydra_suite/resources/configs/skeletons/
```

- [ ] **Step 6: Write the `paths.py` module**

```python
# src/hydra_suite/paths.py
"""Central path resolution for hydra-suite.

Bundled read-only assets are accessed via importlib.resources.
User-writable directories (models, training, config) use platformdirs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir, user_data_dir

logger = logging.getLogger(__name__)

APP_NAME = "hydra-suite"
APP_AUTHOR = "Kronauer Lab"


# ---------------------------------------------------------------------------
# User-writable directories (models, training runs, config)
# ---------------------------------------------------------------------------


def _user_config_dir() -> Path:
    """Return user config directory, creating it if needed."""
    d = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _user_data_dir() -> Path:
    """Return user data directory, creating it if needed."""
    d = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_models_dir() -> Path:
    """Return user-writable models directory."""
    d = _user_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_training_runs_dir() -> Path:
    """Return user-writable training runs directory."""
    d = _user_data_dir() / "training" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_training_workspace_dir(subdir: str = "YOLO") -> Path:
    """Return user-writable training workspace directory."""
    d = _user_data_dir() / "training" / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_advanced_config_path() -> Path:
    """Return path for user's advanced_config.json (writable)."""
    return _user_config_dir() / "advanced_config.json"


def get_presets_dir() -> Path:
    """Return user-writable presets directory, seeding bundled defaults on first use."""
    d = _user_config_dir() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    # Seed bundled configs on first use
    marker = d / ".seeded"
    if not marker.exists():
        for name in get_bundled_config_names():
            target = d / name
            if not target.exists():
                cfg = get_default_config(name)
                if cfg is not None:
                    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        marker.write_text("1", encoding="utf-8")
    return d


def get_skeleton_dir() -> Path:
    """Return user-writable skeleton config directory, seeding bundled defaults."""
    d = _user_config_dir() / "skeletons"
    d.mkdir(parents=True, exist_ok=True)
    marker = d / ".seeded"
    if not marker.exists():
        for name in get_bundled_skeleton_names():
            target = d / name
            if not target.exists():
                cfg = get_skeleton_config(name)
                if cfg is not None:
                    target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        marker.write_text("1", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# Bundled read-only assets (brand icons, default configs)
# ---------------------------------------------------------------------------


def get_brand_icon_bytes(name: str) -> Optional[bytes]:
    """Load a brand icon (SVG or PNG) as bytes. Returns None if not found."""
    from importlib.resources import files

    try:
        return files("hydra_suite.resources.brand").joinpath(name).read_bytes()
    except (FileNotFoundError, TypeError, ModuleNotFoundError):
        return None


def get_default_config(name: str) -> Optional[dict]:
    """Load a bundled default config as a dict. Returns None if not found."""
    from importlib.resources import files

    try:
        text = files("hydra_suite.resources.configs").joinpath(name).read_text(
            encoding="utf-8"
        )
        return json.loads(text)
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_skeleton_config(name: str) -> Optional[dict]:
    """Load a bundled skeleton config as a dict. Returns None if not found."""
    from importlib.resources import files

    try:
        text = (
            files("hydra_suite.resources.configs.skeletons")
            .joinpath(name)
            .read_text(encoding="utf-8")
        )
        return json.loads(text)
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_bundled_config_names() -> list[str]:
    """List available bundled config file names."""
    from importlib.resources import files

    try:
        pkg = files("hydra_suite.resources.configs")
        return sorted(
            r.name for r in pkg.iterdir()
            if r.name.endswith(".json")
        )
    except Exception:
        return []


def get_bundled_skeleton_names() -> list[str]:
    """List available bundled skeleton config file names."""
    from importlib.resources import files

    try:
        pkg = files("hydra_suite.resources.configs.skeletons")
        return sorted(
            r.name for r in pkg.iterdir()
            if r.name.endswith(".json")
        )
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Qt helpers (brand icons as QIcon)
# ---------------------------------------------------------------------------


def get_brand_qicon(name: str):
    """Load a brand icon as a QIcon via byte loading (no filesystem path needed).

    Returns a QIcon, or an empty QIcon if the asset is missing or Qt is unavailable.
    """
    data = get_brand_icon_bytes(name)
    if data is None:
        try:
            from PySide6.QtGui import QIcon
            return QIcon()
        except ImportError:
            return None
    try:
        from PySide6.QtCore import QByteArray
        from PySide6.QtGui import QIcon, QPixmap
        from PySide6.QtSvg import QSvgRenderer

        if name.endswith(".svg"):
            from PySide6.QtCore import QSize
            from PySide6.QtGui import QPainter

            renderer = QSvgRenderer(QByteArray(data))
            pixmap = QPixmap(QSize(256, 256))
            pixmap.fill()
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.end()
            return QIcon(pixmap)
        else:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            return QIcon(pixmap)
    except Exception:
        try:
            from PySide6.QtGui import QIcon
            return QIcon()
        except ImportError:
            return None
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths.py -v`
Expected: All tests PASS (the `_user_config_dir` and `_user_data_dir` tests may need `platformdirs` installed — it will be added to deps in Task 2)

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/paths.py src/hydra_suite/resources/ tests/test_paths.py
git commit -m "feat: add paths module and bundled resources sub-package

Central path resolution using importlib.resources for read-only assets
and platformdirs for user-writable directories (models, training, config).
Copies brand icons and default configs into the package."
```

---

## Task 2: Update `pyproject.toml` — dependencies, package-data, metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/hydra_suite/__init__.py`
- Test: `tests/test_packaging.py`

- [ ] **Step 1: Write the failing packaging test**

```python
# tests/test_packaging.py
"""Tests that packaging metadata and bundled data are correct."""

from pathlib import Path

import pytest


def test_version_matches_metadata():
    """Package version from importlib.metadata matches __init__.__version__."""
    from importlib.metadata import version

    from hydra_suite import __version__

    assert __version__ == version("hydra-suite")


def test_brand_svgs_exist_in_package():
    """Brand SVG files are accessible via importlib.resources."""
    from importlib.resources import files

    brand = files("hydra_suite.resources.brand")
    names = [r.name for r in brand.iterdir()]
    assert "hydra.svg" in names
    assert "datasieve.svg" in names
    assert "classkit.svg" in names
    assert "posekit.svg" in names
    assert "hydraafterhours.svg" in names


def test_default_configs_exist_in_package():
    """Default config JSON files are accessible via importlib.resources."""
    from importlib.resources import files

    configs = files("hydra_suite.resources.configs")
    names = [r.name for r in configs.iterdir()]
    assert "default.json" in names


def test_skeleton_configs_exist_in_package():
    """Skeleton JSON files are accessible via importlib.resources."""
    from importlib.resources import files

    skeletons = files("hydra_suite.resources.configs.skeletons")
    names = [r.name for r in skeletons.iterdir()]
    assert "ooceraea_biroi.json" in names


def test_platformdirs_importable():
    """platformdirs is available as a dependency."""
    import platformdirs

    assert hasattr(platformdirs, "user_config_dir")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_packaging.py -v`
Expected: FAIL — `importlib.metadata.version("hydra-suite")` may fail, `platformdirs` may not be installed yet

- [ ] **Step 3: Update `pyproject.toml` — add dependencies, package-data, fix build-requires**

Replace the `[build-system]` and `[project]` sections in `pyproject.toml`. The key changes:

1. Remove `setuptools_scm` from build-requires (not configured, never used)
2. Add `[project.dependencies]` with all common packages (NOT including torch — users install it themselves)
3. Add `[project.optional-dependencies]` for GPU and dev extras
4. Add `[tool.setuptools.packages.find]` and `[tool.setuptools.package-data]`
5. Add `dynamic = ["version"]` or keep version static (keep static for simplicity)

Edit `pyproject.toml` `[build-system]`:

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"
```

Add after `[project.urls]` and before `[project.scripts]`:

```toml
dependencies = [
    # Core scientific stack
    "numpy>=1.24",
    "scipy>=1.10",
    "pandas>=2.0",
    "numba>=0.58",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    # Computer vision
    "opencv-python-headless>=4.8",
    # GUI
    "PySide6>=6.5",
    # ML (torch deliberately excluded — users install correct variant themselves)
    "timm",
    "ultralytics>=8.0",
    # Nearest-neighbor search
    "hnswlib",
    "usearch",
    "annoy",
    # AprilTags
    "pupil-apriltags",
    # Path resolution
    "platformdirs>=3.0",
]

[project.optional-dependencies]
cuda = [
    "onnxruntime-gpu>=1.24",
    "cupy-cuda12x",
]
mps = [
    "onnxruntime>=1.16",
]
rocm = [
    "cupy-rocm-6-0",
    "onnxruntime",
]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "black>=23.0",
    "isort>=5.12",
    "flake8>=6.0",
    "mypy>=1.5",
    "vulture>=2.7",
]
```

Add after `[tool.deadcode]` section:

```toml
# ============================================================================
# setuptools - Package discovery and data
# ============================================================================
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"hydra_suite" = [
    "resources/brand/*.svg",
    "resources/brand/*.png",
    "resources/configs/*.json",
    "resources/configs/skeletons/*.json",
    "py.typed",
]
```

- [ ] **Step 4: Update `__init__.py` — fix metadata, use importlib.metadata for version**

Replace the top of `src/hydra_suite/__init__.py`:

```python
"""
HYDRA Suite Package

A comprehensive solution for tracking multiple animals in video recordings using computer vision techniques.
The system combines background subtraction, Kalman filtering, and Hungarian algorithm for robust multi-object tracking.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("hydra-suite")
except Exception:
    __version__ = "1.0.0"  # Fallback for editable installs without metadata

__author__ = "Rishika Mohanta"
__email__ = "neurorishika@gmail.com"
```

Keep the rest of the file unchanged.

- [ ] **Step 5: Create `py.typed` marker**

```bash
touch src/hydra_suite/py.typed
```

- [ ] **Step 6: Install platformdirs and reinstall package**

```bash
uv pip install platformdirs>=3.0
uv pip install -e .
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_packaging.py tests/test_paths.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/hydra_suite/__init__.py src/hydra_suite/py.typed tests/test_packaging.py
git commit -m "feat: add pip dependencies, package-data, and fix metadata

- Add [project.dependencies] with all common packages (torch excluded)
- Add [project.optional-dependencies] for cuda/mps/rocm/dev
- Add [tool.setuptools.package-data] for brand/config resources
- Fix placeholder __author__/__email__ in __init__.py
- Use importlib.metadata for version resolution
- Add py.typed marker for type checker support"
```

---

## Task 3: Migrate brand icon loading in all GUI modules

This task replaces all `Path(__file__).resolve().parents[N] / "brand"` patterns with `paths.get_brand_qicon()`.

**Files:**
- Modify: `src/hydra_suite/mat/app/launcher.py:168-181`
- Modify: `src/hydra_suite/mat/gui/main_window.py:4199,13625`
- Modify: `src/hydra_suite/datasieve/gui.py:604,660,1057,1674`
- Modify: `src/hydra_suite/afterhours/app.py:25`
- Modify: `src/hydra_suite/afterhours/gui/main_window.py:437`
- Modify: `src/hydra_suite/classkit/gui/mainwindow.py:462`
- Modify: `src/hydra_suite/posekit/ui/main_window.py:769,1196`
- Modify: `src/hydra_suite/posekit/ui/main.py:57`

- [ ] **Step 1: Migrate `mat/app/launcher.py` — replace brand icon resolution**

Replace lines 168-181:

```python
        # OLD:
        # project_root = Path(__file__).resolve().parents[3]
        # brand_icon = project_root / "brand" / "hydra.svg"
        # fallback_icon = (
        #     Path(__file__).resolve().parent.parent / "resources" / "icon.png"
        # )
        # if brand_icon.exists():
        #     app.setWindowIcon(QIcon(str(brand_icon)))
        # elif fallback_icon.exists():
        #     app.setWindowIcon(QIcon(str(fallback_icon)))
```

With:

```python
        # Set application icon if available
        try:
            from hydra_suite.paths import get_brand_qicon

            icon = get_brand_qicon("hydra.svg")
            if icon and not icon.isNull():
                app.setWindowIcon(icon)
        except Exception:
            pass  # Icon not critical
```

Remove the now-unused `from pathlib import Path` import if it was only used for this purpose (check if Path is used elsewhere in the file first).

- [ ] **Step 2: Migrate `mat/gui/main_window.py` — brand icon at line 4199**

Replace the block around line 4199:

```python
        # OLD:
        # project_root = Path(__file__).resolve().parents[3]
        # logo_path = project_root / "brand" / "hydra.svg"
```

With:

```python
        from hydra_suite.paths import get_brand_icon_bytes
```

Then where the SVG is loaded into a `QSvgRenderer` or `QPixmap`, use:

```python
        logo_data = get_brand_icon_bytes("hydra.svg")
        if logo_data:
            # Use QSvgRenderer with byte data instead of file path
            renderer = QSvgRenderer(QByteArray(logo_data))
            # ... rest of existing rendering logic
```

Repeat the same pattern for line 13625 (same file, same icon).

- [ ] **Step 3: Migrate `datasieve/gui.py` — 4 brand icon references**

For each of the 4 occurrences (lines 604, 660, 1057, 1674), replace:

```python
        # OLD:
        # project_root = Path(__file__).resolve().parents[3]
        # icon_path = project_root / "brand" / "datasieve.svg"
```

With:

```python
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("datasieve.svg")
```

Adjust usage based on context — if the code uses `QIcon(str(icon_path))`, replace with the `icon` object directly. If it checks `.exists()` first, replace with a `not icon.isNull()` check.

- [ ] **Step 4: Migrate `afterhours/app.py` and `afterhours/gui/main_window.py`**

In `afterhours/app.py` line 25, replace:

```python
        # OLD:
        # icon_path = (
        #     Path(__file__).resolve().parents[2]
        #     / "brand"
        #     / "hydraafterhours.svg"
        # )
```

With:

```python
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("hydraafterhours.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
```

In `afterhours/gui/main_window.py` line 437, same pattern — replace `parents[3] / "brand"` with `get_brand_icon_bytes("hydraafterhours.svg")` and load via byte data.

- [ ] **Step 5: Migrate `classkit/gui/mainwindow.py` line 462**

Replace:

```python
        # OLD:
        # logo_path = Path(__file__).resolve().parents[3] / "brand" / "classkit.svg"
```

With:

```python
        from hydra_suite.paths import get_brand_icon_bytes

        logo_data = get_brand_icon_bytes("classkit.svg")
```

Adjust the rendering code to use byte data instead of file path.

- [ ] **Step 6: Migrate `posekit/ui/main_window.py` lines 769, 1196 and `posekit/ui/main.py` line 57**

In `posekit/ui/main_window.py`, replace both occurrences:

```python
        # OLD:
        # logo_path = Path(__file__).resolve().parents[3] / "brand" / "posekit.svg"
```

With:

```python
        from hydra_suite.paths import get_brand_icon_bytes

        logo_data = get_brand_icon_bytes("posekit.svg")
```

In `posekit/ui/main.py` line 57:

```python
        # OLD:
        # project_root = Path(__file__).resolve().parents[3]
        # icon_path = project_root / "brand" / "posekit.svg"
```

With:

```python
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("posekit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
```

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All existing tests PASS plus new tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/mat/app/launcher.py \
        src/hydra_suite/mat/gui/main_window.py \
        src/hydra_suite/datasieve/gui.py \
        src/hydra_suite/afterhours/app.py \
        src/hydra_suite/afterhours/gui/main_window.py \
        src/hydra_suite/classkit/gui/mainwindow.py \
        src/hydra_suite/posekit/ui/main_window.py \
        src/hydra_suite/posekit/ui/main.py
git commit -m "refactor: replace all brand icon parents[N] paths with paths module

Migrated 14 brand icon resolution patterns across 8 files to use
importlib.resources via hydra_suite.paths.get_brand_qicon().
Icons load as bytes, no filesystem path dependency."
```

---

## Task 4: Migrate models, training, and config path resolution

**Files:**
- Modify: `src/hydra_suite/training/registry.py:15-21`
- Modify: `src/hydra_suite/training/model_publish.py:14-20`
- Modify: `src/hydra_suite/mat/gui/main_window.py:2817-2826,8164-8172,19191-19198,19428-19490`
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py:136-138`
- Modify: `src/hydra_suite/posekit/ui/utils.py:64-76`

- [ ] **Step 1: Migrate `training/registry.py`**

Replace:

```python
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_runs_root() -> Path:
    root = _project_root() / "training" / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root
```

With:

```python
def get_runs_root() -> Path:
    from hydra_suite.paths import get_training_runs_dir

    return get_training_runs_dir()
```

Remove the `_project_root` function entirely.

- [ ] **Step 2: Migrate `training/model_publish.py`**

Replace:

```python
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_models_root() -> Path:
    root = _project_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root
```

With:

```python
def get_models_root() -> Path:
    from hydra_suite.paths import get_models_dir

    return get_models_dir()
```

Remove the `_project_root` function entirely.

- [ ] **Step 3: Migrate `mat/gui/main_window.py` — `get_models_root_directory()` at line 2817**

Replace:

```python
def get_models_root_directory() -> object:
    """Return project-local models/ root and create it when missing."""
    project_root = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
    )
    models_root = os.path.join(project_root, "models")
    os.makedirs(models_root, exist_ok=True)
    return models_root
```

With:

```python
def get_models_root_directory() -> str:
    """Return user-local models/ root and create it when missing."""
    from hydra_suite.paths import get_models_dir

    return str(get_models_dir())
```

- [ ] **Step 4: Migrate `mat/gui/main_window.py` — skeleton config at line 8164**

Replace:

```python
            candidate = (
                Path(__file__).resolve().parents[4]
                / "configs"
                / "skeletons"
                / "ooceraea_biroi.json"
            )
            if candidate.exists():
                default_skeleton = str(candidate)
```

With:

```python
            from hydra_suite.paths import get_skeleton_dir

            candidate = get_skeleton_dir() / "ooceraea_biroi.json"
            if candidate.exists():
                default_skeleton = str(candidate)
```

- [ ] **Step 5: Migrate `mat/gui/main_window.py` — `_get_presets_dir()` at line 19191**

Replace:

```python
    def _get_presets_dir(self):
        """Get the presets directory path."""
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        )
        presets_dir = os.path.join(repo_root, "configs")
        return presets_dir
```

With:

```python
    def _get_presets_dir(self):
        """Get the presets directory path."""
        from hydra_suite.paths import get_presets_dir

        return str(get_presets_dir())
```

- [ ] **Step 6: Migrate `mat/gui/main_window.py` — `_load_advanced_config()` at line 19428**

Replace the path resolution in `_load_advanced_config()`:

```python
        # OLD:
        # package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # config_path = os.path.join(package_dir, "advanced_config.json")
```

With:

```python
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())
```

Keep the rest of the method (default_config dict, loading, auto-create logic) unchanged — only the path changes.

Also update `_save_advanced_config()` at line 19475 the same way:

```python
        # OLD:
        # config_path = os.path.join(
        #     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        #     "advanced_config.json",
        # )
```

With:

```python
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())
```

- [ ] **Step 7: Migrate `mat/gui/dialogs/train_yolo_dialog.py` line 136**

Replace:

```python
        self.repo_root = Path(__file__).resolve().parents[5]
        self.workspace_default = self.repo_root / "training" / "YOLO"
```

With:

```python
        from hydra_suite.paths import get_training_workspace_dir

        self.workspace_default = get_training_workspace_dir("YOLO")
```

Remove `self.repo_root` if it's not used elsewhere in the file (check first).

- [ ] **Step 8: Migrate `posekit/ui/utils.py` — `get_default_skeleton_dir()` at line 64**

Replace:

```python
def get_default_skeleton_dir() -> Optional[Path]:
    """Return the repository-level skeleton config directory if available."""
    here = Path(__file__).resolve()
    repo_root = here.parents[4] if len(here.parents) >= 5 else here.parent
    cfg = repo_root / DEFAULT_SKELETON_DIRNAME
    if cfg.exists() and cfg.is_dir():
        return cfg
    return None
```

With:

```python
def get_default_skeleton_dir() -> Optional[Path]:
    """Return the user-level skeleton config directory if available."""
    from hydra_suite.paths import get_skeleton_dir

    d = get_skeleton_dir()
    if d.exists() and d.is_dir():
        return d
    return None
```

- [ ] **Step 9: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add src/hydra_suite/training/registry.py \
        src/hydra_suite/training/model_publish.py \
        src/hydra_suite/mat/gui/main_window.py \
        src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py \
        src/hydra_suite/posekit/ui/utils.py
git commit -m "refactor: migrate models/training/config paths to paths module

Replaced all remaining parents[N] and os.path.dirname chains:
- Models dir → ~/.local/share/hydra-suite/models/
- Training runs → ~/.local/share/hydra-suite/training/runs/
- Presets → ~/.config/hydra-suite/presets/ (seeded from bundled)
- Skeletons → ~/.config/hydra-suite/skeletons/ (seeded from bundled)
- Advanced config → ~/.config/hydra-suite/advanced_config.json"
```

---

## Task 5: Add `platformdirs` to requirements files and update install docs

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-mps.txt`
- Modify: `requirements-cuda.txt`
- Modify: `requirements-rocm.txt`
- Modify: `docs/getting-started/installation.md`

- [ ] **Step 1: Add `platformdirs` to all requirements files**

Add `platformdirs>=3.0` to each requirements file, right before the `-e .` line:

In `requirements.txt`, add after `ultralytics`:
```
platformdirs>=3.0
```

In `requirements-mps.txt`, add after `ultralytics`:
```
platformdirs>=3.0
```

In `requirements-cuda.txt`, add after `ultralytics`:
```
platformdirs>=3.0
```

In `requirements-rocm.txt`, add after `ultralytics`:
```
platformdirs>=3.0
```

- [ ] **Step 2: Update installation docs to mention pip install option**

Add a new section to `docs/getting-started/installation.md` after the "Before You Start" section:

```markdown
## Quick Install (CPU, pip only)

If you don't need GPU acceleration and want the simplest install:

```bash
pip install hydra-suite
```

For GPU support, continue with the full installation below.
```

- [ ] **Step 3: Verify install works**

```bash
uv pip install -e .
python -c "from hydra_suite.paths import get_models_dir; print(get_models_dir())"
```

Expected: Prints something like `/Users/<you>/Library/Application Support/hydra-suite/models`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt requirements-mps.txt requirements-cuda.txt requirements-rocm.txt \
        docs/getting-started/installation.md
git commit -m "chore: add platformdirs to all requirements files and document pip install"
```

---

## Task 6: Data migration helper for existing users

Existing developer users have models in `<repo>/models/` and training runs in `<repo>/training/`. They need a one-time migration.

**Files:**
- Create: `src/hydra_suite/paths_migrate.py`
- Test: `tests/test_paths_migrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths_migrate.py
"""Tests for the one-time data migration helper."""

import json
from pathlib import Path

import pytest


def test_migrate_copies_models(tmp_path, monkeypatch):
    """Migration copies models from old repo location to new user data dir."""
    # Set up fake old repo structure
    old_models = tmp_path / "old_repo" / "models"
    old_models.mkdir(parents=True)
    (old_models / "test_model.pt").write_bytes(b"fake model data")

    # Set up fake new location
    new_data = tmp_path / "new_data"
    monkeypatch.setenv("XDG_DATA_HOME", str(new_data))

    from hydra_suite.paths_migrate import migrate_repo_data

    result = migrate_repo_data(repo_root=tmp_path / "old_repo", dry_run=False)

    assert result["models_copied"] >= 1
    # Verify file exists in new location
    from hydra_suite.paths import get_models_dir

    assert (get_models_dir() / "test_model.pt").exists()


def test_migrate_dry_run_copies_nothing(tmp_path, monkeypatch):
    """Dry run reports what would happen without copying."""
    old_models = tmp_path / "old_repo" / "models"
    old_models.mkdir(parents=True)
    (old_models / "test_model.pt").write_bytes(b"fake model data")

    new_data = tmp_path / "new_data"
    monkeypatch.setenv("XDG_DATA_HOME", str(new_data))

    from hydra_suite.paths_migrate import migrate_repo_data

    result = migrate_repo_data(repo_root=tmp_path / "old_repo", dry_run=True)

    assert result["models_copied"] >= 1
    from hydra_suite.paths import get_models_dir

    # Nothing actually copied
    assert not (get_models_dir() / "test_model.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths_migrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydra_suite.paths_migrate'`

- [ ] **Step 3: Write the migration module**

```python
# src/hydra_suite/paths_migrate.py
"""One-time migration helper: copy models/training data from repo to user dirs.

Usage:
    python -m hydra_suite.paths_migrate /path/to/hydra-suite
    python -m hydra_suite.paths_migrate /path/to/hydra-suite --dry-run
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .paths import get_models_dir, get_presets_dir, get_training_runs_dir

logger = logging.getLogger(__name__)


def migrate_repo_data(
    repo_root: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    """Migrate data from old repo-relative locations to user dirs.

    Returns a summary dict with counts of items found/copied.
    """
    repo_root = Path(repo_root).resolve()
    summary: dict[str, Any] = {
        "models_copied": 0,
        "training_copied": 0,
        "configs_copied": 0,
    }

    # Migrate models/
    old_models = repo_root / "models"
    if old_models.exists():
        new_models = get_models_dir()
        for src in old_models.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(old_models)
            dst = new_models / rel
            summary["models_copied"] += 1
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                    logger.info("Copied model: %s -> %s", rel, dst)
                else:
                    logger.info("Skipped (exists): %s", rel)

    # Migrate training/runs/
    old_runs = repo_root / "training" / "runs"
    if old_runs.exists():
        new_runs = get_training_runs_dir()
        for src in old_runs.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(old_runs)
            dst = new_runs / rel
            summary["training_copied"] += 1
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)

    # Migrate configs/ (user presets)
    old_configs = repo_root / "configs"
    if old_configs.exists():
        new_presets = get_presets_dir()
        for src in old_configs.glob("*.json"):
            dst = new_presets / src.name
            summary["configs_copied"] += 1
            if not dry_run:
                if not dst.exists():
                    shutil.copy2(src, dst)

    action = "Would copy" if dry_run else "Copied"
    logger.info(
        "%s: %d models, %d training items, %d configs",
        action,
        summary["models_copied"],
        summary["training_copied"],
        summary["configs_copied"],
    )
    return summary


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Migrate data from repo to user directories"
    )
    parser.add_argument("repo_root", help="Path to hydra-suite repo root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without copying",
    )
    args = parser.parse_args()

    if not Path(args.repo_root).is_dir():
        print(f"Error: {args.repo_root} is not a directory", file=sys.stderr)
        sys.exit(1)

    result = migrate_repo_data(args.repo_root, dry_run=args.dry_run)
    print(f"Models: {result['models_copied']}")
    print(f"Training: {result['training_copied']}")
    print(f"Configs: {result['configs_copied']}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths_migrate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/paths_migrate.py tests/test_paths_migrate.py
git commit -m "feat: add one-time migration helper for repo-to-user-dir data

Existing users can run:
  python -m hydra_suite.paths_migrate /path/to/repo --dry-run
  python -m hydra_suite.paths_migrate /path/to/repo

Copies models/, training/runs/, and configs/ to platformdirs locations."
```

---

## Task 7: Clean up old assets and verify wheel build

**Files:**
- Modify: none (verification only)

- [ ] **Step 1: Verify the wheel includes all assets**

```bash
python -m build --wheel
```

Then inspect the wheel:

```bash
unzip -l dist/multi_animal_tracker-*.whl | grep -E "(brand|configs|skeletons|py\.typed)"
```

Expected output should show:
```
hydra_suite/resources/brand/hydra.svg
hydra_suite/resources/brand/datasieve.svg
hydra_suite/resources/brand/classkit.svg
hydra_suite/resources/brand/posekit.svg
hydra_suite/resources/brand/hydraafterhours.svg
hydra_suite/resources/brand/*.png
hydra_suite/resources/configs/default.json
hydra_suite/resources/configs/ooceraea_biroi.json
hydra_suite/resources/configs/skeletons/ooceraea_biroi.json
hydra_suite/py.typed
```

- [ ] **Step 2: Test wheel install in a clean environment**

```bash
python -m venv /tmp/mat-test-venv
/tmp/mat-test-venv/bin/pip install dist/multi_animal_tracker-*.whl
/tmp/mat-test-venv/bin/python -c "from hydra_suite.paths import get_brand_icon_bytes; assert get_brand_icon_bytes('hydra.svg') is not None; print('OK: assets bundled')"
/tmp/mat-test-venv/bin/python -c "from hydra_suite.paths import get_default_config; cfg = get_default_config('default.json'); assert cfg is not None; print('OK: configs bundled')"
/tmp/mat-test-venv/bin/python -c "from hydra_suite.paths import get_models_dir; print('Models dir:', get_models_dir())"
```

Expected: All print "OK" and show the platformdirs-managed path.

- [ ] **Step 3: Run full test suite one final time**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Run format and lint**

```bash
make format
make lint
```

Expected: Clean pass.

- [ ] **Step 5: Commit any formatting changes**

```bash
git add -u
git commit -m "style: format after packaging refactor"
```

---

## Task 8: Update documentation for new path structure and pip install

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/getting-started/installation.md`
- Modify: `docs/developer-guide/architecture.md`
- Modify: `docs/developer-guide/module-map.md`

- [ ] **Step 1: Update `README.md` — add pip install, fix banner path**

The banner image reference at line 4 uses `brand/banner.png` which is the repo-root relative path. This still works for GitHub rendering (reads from repo), so keep it. But update the install section to show both pip and conda options:

Replace the "Install (Quick)" section:

```markdown
## Install (Quick)

### Option A: pip (CPU, simplest)

```bash
pip install hydra-suite
```

For GPU (NVIDIA), install PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install hydra-suite[cuda]
```

### Option B: Full environment (GPU, recommended for development)

```bash
mamba env create -f environment.yml          # or environment-mps.yml / environment-cuda.yml
conda activate hydra-suite          # or -mps / -cuda
uv pip install -r requirements.txt           # or requirements-mps.txt / requirements-cuda13.txt
```

Platform-specific environments are documented in `ENVIRONMENTS.md` and in the online docs.
```

- [ ] **Step 2: Update `CLAUDE.md` — architecture section**

In the Architecture section, update the System Layers table to include the new `Resources` and `Paths` modules:

Add these rows to the table after the `Utils` row:

```markdown
| Resources | `hydra_suite.resources` | Bundled read-only assets (brand icons, default configs, skeletons) |
| Paths | `hydra_suite.paths` | Central path resolution: bundled assets via `importlib.resources`, user dirs via `platformdirs` |
```

Add to the **Key boundary rules** section:

```markdown
- All path resolution must go through `hydra_suite.paths`. No module should use `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are accessed via `importlib.resources` through the `paths` module.
- User-writable data (models, training, config) goes to platform-appropriate directories via `platformdirs`.
```

Update the **Build & Environment Setup** section to include the pip install option:

```markdown
## Quick Install (pip, CPU only)

```bash
pip install hydra-suite
```

For GPU variants:

```bash
# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install hydra-suite[cuda]

# Apple Silicon
pip install hydra-suite[mps]
```
```

Update the **Key Source Files for Auditing Behavior** list to add:

```markdown
- `src/hydra_suite/paths.py` — central path resolution (all asset/data paths)
```

- [ ] **Step 3: Update `docs/getting-started/installation.md` — comprehensive install rewrite**

Add a new section after "Before You Start" and before "Choose Your Installation Path":

```markdown
## Quick Install (pip)

If you want the simplest install without GPU acceleration:

```bash
pip install hydra-suite
```

This installs all dependencies and bundled assets. Launch with `hydra` or `posekit-labeler`.

### GPU via pip

Install the correct PyTorch variant first, then install the package with GPU extras:

```bash
# NVIDIA CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install hydra-suite[cuda]

# NVIDIA CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install hydra-suite[cuda]

# Apple Silicon (MPS) — torch includes MPS by default
pip install torch torchvision
pip install hydra-suite[mps]

# AMD ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install hydra-suite[rocm]
```

> **Note:** PyTorch GPU builds are not hosted on PyPI. You must install torch from PyTorch's own index first. The `pip install hydra-suite` step will not overwrite your existing torch installation.

### Data directories

After installation, user data is stored in platform-appropriate locations:

| Data | macOS | Linux |
|------|-------|-------|
| Config | `~/Library/Application Support/hydra-suite/` | `~/.config/hydra-suite/` |
| Models | `~/Library/Application Support/hydra-suite/models/` | `~/.local/share/hydra-suite/models/` |
| Training | `~/Library/Application Support/hydra-suite/training/` | `~/.local/share/hydra-suite/training/` |

Default config presets and skeleton definitions are bundled with the package and automatically seeded to your config directory on first run.

### Migrating from a repo checkout

If you previously used a cloned repo with models in `<repo>/models/` and training data in `<repo>/training/`, migrate with:

```bash
python -m hydra_suite.paths_migrate /path/to/hydra-suite --dry-run  # preview
python -m hydra_suite.paths_migrate /path/to/hydra-suite            # copy
```
```

Keep the existing "Full Environment Setup" sections (conda + requirements) below — label them clearly:

```markdown
---

## Full Environment Setup (recommended for GPU and development)

The full environment setup uses conda/mamba for system libraries and pip for Python packages. This is recommended for GPU users and developers.
```

- [ ] **Step 4: Update `docs/developer-guide/architecture.md` — system layers**

Replace the System Layers section:

```markdown
## System Layers

- `hydra_suite.tracker`: MAT launcher, GUI, dialogs, widgets.
- `hydra_suite.posekit`: pose-labeling application and related dialogs/inference flows.
- `hydra_suite.classkit`: classification/embedding toolkit.
- `hydra_suite.afterhours`: interactive proofreading.
- `hydra_suite.datasieve`: data sieve tool.
- `hydra_suite.integrations`: external tool bridges (SLEAP, X-AnyLabeling).
- `hydra_suite.core`: detection, filtering, assignment, post-processing, worker orchestration.
- `hydra_suite.runtime`: compute runtime selection and GPU utilities.
- `hydra_suite.data`: CSV/cache/dataset generation and merging.
- `hydra_suite.training`: dataset builders, training runner, registry, model publishing.
- `hydra_suite.utils`: shared helpers (GPU detection, geometry, image processing, batching, prefetch).
- `hydra_suite.paths`: central path resolution — bundled assets via `importlib.resources`, user dirs via `platformdirs`.
- `hydra_suite.resources`: bundled read-only assets (brand icons, default configs, skeletons).
```

Add to "Key Operational Boundaries":

```markdown
- All path resolution goes through `hydra_suite.paths`. No module uses `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are in `hydra_suite.resources` and accessed via `importlib.resources`.
- User-writable data (models, training runs, config) lives in platform directories (`~/.config/`, `~/.local/share/`), never inside the installed package.
```

- [ ] **Step 5: Update `docs/developer-guide/module-map.md` — add paths and resources**

Add new sections after "Integrations and Utils":

```markdown
## Paths and Resources

- `hydra_suite.paths` — central path resolver (brand icons, configs, models dir, training dir)
- `hydra_suite.paths_migrate` — one-time migration helper for repo-to-user-dir data
- `hydra_suite.resources` — bundled read-only assets package
- `hydra_suite.resources.brand` — SVG/PNG brand icons
- `hydra_suite.resources.configs` — default config presets
- `hydra_suite.resources.configs.skeletons` — skeleton keypoint definitions
```

- [ ] **Step 6: Run docs build to verify no broken links**

```bash
make docs-build
```

Expected: Clean build with no errors.

- [ ] **Step 7: Commit**

```bash
git add README.md CLAUDE.md \
        docs/getting-started/installation.md \
        docs/developer-guide/architecture.md \
        docs/developer-guide/module-map.md
git commit -m "docs: update documentation for pip-publishable packaging

- README: add pip install instructions alongside conda workflow
- CLAUDE.md: add paths/resources modules to architecture, add path rules
- installation.md: add pip install section, data directory table, migration guide
- architecture.md: update system layers with paths/resources, add boundary rules
- module-map.md: add paths and resources modules"
```

---

## Summary of directory changes

### Before
```
hydra-suite/
  src/brand/                    ← icons, outside package
  configs/                      ← presets, outside package
  models/                       ← user data, in repo root
  training/                     ← user data, in repo root
  src/hydra_suite/            ← no resources/, no paths.py
```

### After
```
hydra-suite/
  src/brand/                    ← kept for development reference (not deleted)
  configs/                      ← kept for development reference (not deleted)
  src/hydra_suite/
    paths.py                    ← central path resolution
    paths_migrate.py            ← one-time migration helper
    py.typed                    ← type checker marker
    resources/
      brand/*.svg, *.png        ← copied from src/brand/
      configs/*.json            ← copied from configs/
      configs/skeletons/*.json  ← copied from configs/skeletons/

User home (created at runtime):
  ~/.config/hydra-suite/
    advanced_config.json
    presets/                    ← seeded from bundled configs
    skeletons/                  ← seeded from bundled skeletons
  ~/.local/share/hydra-suite/
    models/                     ← user-trained models
    training/runs/              ← training run registry
```
