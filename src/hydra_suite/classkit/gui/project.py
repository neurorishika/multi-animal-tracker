"""ClassKit project-path helpers."""

from __future__ import annotations

from pathlib import Path


def default_project_parent_dir() -> Path:
    """Return the default parent directory for new ClassKit projects."""
    from hydra_suite.paths import get_projects_dir

    parent = get_projects_dir() / "ClassKit"
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        pass
    return parent
