"""DetectKit UI utility functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .constants import IMG_EXTS, OBB_LABEL_FIELDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UI settings persistence
# ---------------------------------------------------------------------------


def get_ui_settings_path() -> Path:
    """Return the path to the DetectKit UI-settings JSON file."""
    try:
        from multi_tracker.paths import _user_data_dir

        return _user_data_dir() / "detectkit" / "ui_settings.json"
    except Exception:
        return Path.home() / ".detectkit" / "ui_settings.json"


def load_ui_settings() -> dict:
    """Load saved UI settings (window size, last dirs, etc.)."""
    sp = get_ui_settings_path()
    if not sp.exists():
        return {}
    try:
        data = json.loads(sp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        logger.debug("Failed to read UI settings", exc_info=True)
    return {}


def save_ui_settings(settings: dict) -> None:
    """Persist UI settings."""
    sp = get_ui_settings_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Image / label discovery
# ---------------------------------------------------------------------------


def list_images_in_source(source_path: str) -> list[Path]:
    """Return sorted list of image files found under *source_path*.

    Checks ``source_path/images/`` first; falls back to the source root.
    """
    root = Path(source_path)
    images_dir = root / "images"
    search_root = images_dir if images_dir.is_dir() else root

    results: list[Path] = []
    for p in search_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            results.append(p)
    results.sort()
    return results


def find_label_for_image(
    image_path: Path,
    source_path: str,
) -> Optional[Path]:
    """Locate the OBB label file corresponding to *image_path*.

    Strategies tried in order:
    1. Mirror the ``images/`` sub-path into ``labels/`` (YOLO convention).
    2. Stem match directly inside ``<source>/labels/``.
    3. Recursive search under ``<source>/labels/`` for stem match.
    """
    root = Path(source_path)
    labels_dir = root / "labels"
    stem = image_path.stem

    # Strategy 1: mirror images -> labels
    images_dir = root / "images"
    if images_dir.is_dir():
        try:
            rel = image_path.relative_to(images_dir)
            candidate = labels_dir / rel.with_suffix(".txt")
            if candidate.exists():
                return candidate
        except ValueError:
            pass

    # Strategy 2: direct stem match in labels/
    if labels_dir.is_dir():
        candidate = labels_dir / f"{stem}.txt"
        if candidate.exists():
            return candidate

    # Strategy 3: recursive search
    if labels_dir.is_dir():
        for p in labels_dir.rglob(f"{stem}.txt"):
            return p

    return None


def parse_obb_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> list[dict]:
    """Parse an OBB label file and return pixel-coordinate polygons.

    Each valid line has *OBB_LABEL_FIELDS* (9) space-separated values:
    ``class_id x1 y1 x2 y2 x3 y3 x4 y4`` where coordinates are
    normalised [0, 1].  Returns a list of dicts with keys ``class_id``
    (int) and ``polygon_px`` (list of four ``(x, y)`` tuples in pixels).
    Invalid lines are silently skipped.
    """
    results: list[dict] = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except Exception:
        return results

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != OBB_LABEL_FIELDS:
            continue
        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            polygon_px = [
                (coords[i] * img_w, coords[i + 1] * img_h) for i in range(0, 8, 2)
            ]
            results.append(
                {
                    "class_id": class_id,
                    "polygon_px": polygon_px,
                }
            )
        except (ValueError, IndexError):
            continue

    return results
