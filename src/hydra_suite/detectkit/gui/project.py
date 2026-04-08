"""DetectKit project lifecycle: create, open, save, recent projects."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .constants import DEFAULT_PROJECT_FILENAME, DEFAULT_PROJECTS_ROOT_NAME
from .models import DetectKitProject, normalize_class_names

logger = logging.getLogger(__name__)

_MAX_RECENT = 20


# ---------------------------------------------------------------------------
# Recent-projects persistence
# ---------------------------------------------------------------------------


def get_recent_projects_path() -> Path:
    """Return the path to the recent-projects JSON file."""
    try:
        from hydra_suite.paths import _user_data_dir

        return _user_data_dir() / "detectkit" / "recent_projects.json"
    except Exception:
        return Path.home() / ".detectkit" / "recent_projects.json"


def load_recent_projects() -> list[str]:
    """Load the list of recent project directory paths."""
    rp = get_recent_projects_path()
    if not rp.exists():
        return []
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(p) for p in data]
    except Exception:
        logger.debug("Failed to read recent projects file", exc_info=True)
    return []


def save_recent_projects(paths: list[str]) -> None:
    """Persist at most *_MAX_RECENT* recent project paths."""
    rp = get_recent_projects_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(paths[:_MAX_RECENT], indent=2), encoding="utf-8")


def add_to_recent(project_dir: str) -> None:
    """Add *project_dir* to the top of the recent list, de-duplicating."""
    paths = load_recent_projects()
    # Remove any existing occurrence
    paths = [p for p in paths if p != project_dir]
    paths.insert(0, project_dir)
    save_recent_projects(paths)


# ---------------------------------------------------------------------------
# Project file helpers
# ---------------------------------------------------------------------------


def project_file_path(project_dir: Path) -> Path:
    """Return the canonical project-file path inside *project_dir*."""
    return project_dir / DEFAULT_PROJECT_FILENAME


def default_project_parent_dir() -> Path:
    """Return the default parent directory for new DetectKit projects."""
    from hydra_suite.paths import get_projects_dir

    parent = get_projects_dir() / DEFAULT_PROJECTS_ROOT_NAME
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
    return parent


def open_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Open an existing project from *project_dir*."""
    pf = project_file_path(project_dir)
    if not pf.exists():
        logger.warning("Project file not found: %s", pf)
        return None
    proj = DetectKitProject.load(pf)
    proj.project_dir = project_dir
    add_to_recent(str(project_dir))
    return proj


def create_project(
    project_dir: Path,
    class_name: str = "object",
    *,
    class_names: list[str] | None = None,
) -> DetectKitProject:
    """Create a new project in *project_dir* and persist defaults."""
    project_dir.mkdir(parents=True, exist_ok=True)
    resolved_class_names = normalize_class_names(
        class_names if class_names is not None else [class_name]
    )
    proj = DetectKitProject(project_dir=project_dir, class_names=resolved_class_names)
    save_project(proj)
    add_to_recent(str(project_dir))
    return proj


def save_project(proj: DetectKitProject) -> None:
    """Save *proj* to its canonical project file."""
    proj.save(project_file_path(proj.project_dir))
