"""Central path resolution for multi-animal-tracker.

User-writable directories are managed via *platformdirs*.
Bundled read-only assets are accessed via *importlib.resources*.
Qt helpers provide lazy-loaded QIcon construction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from platformdirs import user_config_dir, user_data_dir

logger = logging.getLogger(__name__)

APP_NAME = "multi-animal-tracker"
APP_AUTHOR = "Rishika Mohanta"

# ---------------------------------------------------------------------------
# Internal helpers — user-writable directories
# ---------------------------------------------------------------------------


def _user_config_dir() -> Path:
    """Return (and create) the user configuration directory."""
    p = Path(user_config_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _user_data_dir() -> Path:
    """Return (and create) the user data directory."""
    p = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Public — user-writable directories
# ---------------------------------------------------------------------------


def get_models_dir() -> Path:
    """Return (and create) the models directory."""
    p = _user_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_training_runs_dir() -> Path:
    """Return (and create) the training runs directory."""
    p = _user_data_dir() / "training" / "runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_training_workspace_dir(subdir: str = "YOLO") -> Path:
    """Return (and create) the training workspace directory."""
    p = _user_data_dir() / "training" / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_advanced_config_path() -> Path:
    """Return the path to the advanced configuration file."""
    return _user_config_dir() / "advanced_config.json"


def get_presets_dir() -> Path:
    """Return (and create) the presets directory, seeding bundled defaults on first use."""
    p = _user_config_dir() / "presets"
    p.mkdir(parents=True, exist_ok=True)
    marker = p / ".seeded"
    if not marker.exists():
        _seed_bundled_configs(p)
        marker.touch()
    return p


def get_skeleton_dir() -> Path:
    """Return (and create) the skeletons directory, seeding bundled defaults on first use."""
    p = _user_config_dir() / "skeletons"
    p.mkdir(parents=True, exist_ok=True)
    marker = p / ".seeded"
    if not marker.exists():
        _seed_bundled_skeletons(p)
        marker.touch()
    return p


# ---------------------------------------------------------------------------
# Internal — seed helpers
# ---------------------------------------------------------------------------


def _seed_bundled_configs(dest: Path) -> None:
    """Copy bundled config presets into *dest*."""
    for name in get_bundled_config_names():
        data = get_default_config(name)
        if data is not None:
            (dest / name).write_text(json.dumps(data, indent=4), encoding="utf-8")


def _seed_bundled_skeletons(dest: Path) -> None:
    """Copy bundled skeleton configs into *dest*."""
    for name in get_bundled_skeleton_names():
        data = get_skeleton_config(name)
        if data is not None:
            (dest / name).write_text(json.dumps(data, indent=4), encoding="utf-8")


# ---------------------------------------------------------------------------
# Bundled read-only assets (importlib.resources)
# ---------------------------------------------------------------------------


def _resources_files(package: str):
    """Get importlib.resources files handle for a package."""
    from importlib.resources import files

    return files(package)


def get_brand_icon_bytes(name: str) -> Optional[bytes]:
    """Read a brand icon asset by filename. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("multi_tracker.resources.brand").joinpath(name)
        return ref.read_bytes()
    except (FileNotFoundError, TypeError, ModuleNotFoundError):
        return None


def get_default_config(name: str) -> Optional[dict]:
    """Read a bundled config preset as a dict. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("multi_tracker.resources.configs").joinpath(name)
        return json.loads(ref.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_skeleton_config(name: str) -> Optional[dict]:
    """Read a bundled skeleton config as a dict. Returns None if not found."""
    if "/" in name or "\\" in name or name.startswith("."):
        return None
    try:
        ref = _resources_files("multi_tracker.resources.configs.skeletons").joinpath(
            name
        )
        return json.loads(ref.read_text(encoding="utf-8"))
    except (FileNotFoundError, TypeError, ModuleNotFoundError, json.JSONDecodeError):
        return None


def get_bundled_config_names() -> list[str]:
    """List .json files bundled in multi_tracker.resources.configs."""
    try:
        root = _resources_files("multi_tracker.resources.configs")
        return sorted(
            item.name for item in root.iterdir() if item.name.endswith(".json")
        )
    except (TypeError, ModuleNotFoundError):
        return []


def get_bundled_skeleton_names() -> list[str]:
    """List .json files bundled in multi_tracker.resources.configs.skeletons."""
    try:
        root = _resources_files("multi_tracker.resources.configs.skeletons")
        return sorted(
            item.name for item in root.iterdir() if item.name.endswith(".json")
        )
    except (TypeError, ModuleNotFoundError):
        return []


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------


def get_brand_qicon(name: str):
    """Load a brand asset as a QIcon. Returns an empty QIcon on failure.

    All Qt imports are lazy to avoid import-time failures when Qt is not
    installed or when running headless.
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

        if name.lower().endswith(".svg"):
            from PySide6.QtCore import QSize
            from PySide6.QtGui import QImage, QPainter
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(QByteArray(data))
            image = QImage(QSize(256, 256), QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(0)
            painter = QPainter(image)
            renderer.render(painter)
            painter.end()
            pixmap = QPixmap.fromImage(image)
        else:
            pixmap = QPixmap()
            pixmap.loadFromData(QByteArray(data))

        return QIcon(pixmap)
    except (ImportError, Exception):
        try:
            from PySide6.QtGui import QIcon

            return QIcon()
        except ImportError:
            return None
