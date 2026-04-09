"""DetectKit project lifecycle: create, open, save, recent projects."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from hydra_suite.data.project_bundle import (
    DEFAULT_BUNDLE_HISTORY_DIRNAME,
    DEFAULT_BUNDLE_STATE_DIRNAME,
    ProjectBundleManifest,
    bundle_paths,
    ensure_bundle_subdirectories,
    ensure_project_bundle_layout,
    load_project_bundle_manifest,
    save_project_bundle_manifest,
)

from .constants import DEFAULT_PROJECT_FILENAME, DEFAULT_PROJECTS_ROOT_NAME
from .models import DetectKitProject, normalize_class_names

logger = logging.getLogger(__name__)

_MAX_RECENT = 20
_KIT_NAME = "detectkit"
_LEGACY_ARCHIVE_PREFIX = "legacy_"
_DETECTKIT_ARTIFACT_DIRS = {
    "training_runs": "artifacts/training_runs",
    "evaluation": "artifacts/evaluation",
    "exports": "artifacts/exports",
}


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
    return bundle_paths(project_dir).state_dir / DEFAULT_PROJECT_FILENAME


def legacy_project_file_path(project_dir: Path) -> Path:
    """Return the legacy DetectKit project-file path at the project root."""
    return project_dir / DEFAULT_PROJECT_FILENAME


def project_exists(project_dir: Path) -> bool:
    """Return True when *project_dir* contains either a bundle or legacy project."""
    manifest_path = bundle_paths(project_dir).manifest_path
    return (
        manifest_path.exists()
        or project_file_path(project_dir).exists()
        or legacy_project_file_path(project_dir).exists()
    )


def _manifest_for_project(project_dir: Path) -> ProjectBundleManifest:
    """Build the shared bundle manifest for a DetectKit project."""
    return ProjectBundleManifest(
        kit=_KIT_NAME,
        display_name=project_dir.name,
        state_path=str(Path(DEFAULT_BUNDLE_STATE_DIRNAME) / DEFAULT_PROJECT_FILENAME),
        artifacts_dir="artifacts",
        history_dir=DEFAULT_BUNDLE_HISTORY_DIRNAME,
        meta={
            "state_format": DEFAULT_PROJECT_FILENAME,
            "artifact_dirs": dict(_DETECTKIT_ARTIFACT_DIRS),
        },
    )


def detectkit_artifact_paths(project_dir: Path) -> dict[str, Path]:
    """Return typed artifact directories for DetectKit bundle projects."""
    created = ensure_bundle_subdirectories(
        project_dir,
        tuple(_DETECTKIT_ARTIFACT_DIRS.values()),
    )
    return {
        name: created[relative] for name, relative in _DETECTKIT_ARTIFACT_DIRS.items()
    }


def _state_path_from_manifest(
    project_dir: Path, manifest: ProjectBundleManifest
) -> Path:
    """Resolve the DetectKit state file from the shared bundle manifest."""
    return project_dir / Path(manifest.state_path)


def _archive_legacy_project_file(project_dir: Path) -> None:
    """Move the legacy root project file into the bundle history directory."""
    legacy_path = legacy_project_file_path(project_dir)
    canonical_path = project_file_path(project_dir)
    if not legacy_path.exists() or legacy_path == canonical_path:
        return

    history_dir = ensure_project_bundle_layout(project_dir).history_dir
    archive_path = history_dir / f"{_LEGACY_ARCHIVE_PREFIX}{DEFAULT_PROJECT_FILENAME}"
    if archive_path.exists():
        legacy_path.unlink()
        return
    shutil.move(str(legacy_path), str(archive_path))


def _load_bundle_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Load a DetectKit project from the shared bundle layout if present."""
    manifest = load_project_bundle_manifest(project_dir)
    if manifest is None:
        return None
    if manifest.kit and manifest.kit != _KIT_NAME:
        logger.warning("Bundle manifest kit mismatch for %s", project_dir)
        return None

    state_path = _state_path_from_manifest(project_dir, manifest)
    if not state_path.exists():
        logger.warning("DetectKit state file not found: %s", state_path)
        return None

    proj = DetectKitProject.load(state_path)
    proj.project_dir = project_dir
    return proj


def _load_legacy_project(project_dir: Path) -> Optional[DetectKitProject]:
    """Load and migrate a legacy DetectKit root project file if present."""
    legacy_path = legacy_project_file_path(project_dir)
    if not legacy_path.exists():
        return None

    proj = DetectKitProject.load(legacy_path)
    proj.project_dir = project_dir
    save_project(proj)
    return proj


def _ensure_bundle_manifest(project_dir: Path) -> None:
    """Create or refresh the shared bundle manifest for *project_dir*."""
    detectkit_artifact_paths(project_dir)
    save_project_bundle_manifest(project_dir, _manifest_for_project(project_dir))


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
    project_dir = project_dir.expanduser().resolve()
    proj = _load_bundle_project(project_dir)
    if proj is None:
        proj = _load_legacy_project(project_dir)
    if proj is None and project_file_path(project_dir).exists():
        proj = DetectKitProject.load(project_file_path(project_dir))
        proj.project_dir = project_dir
        _ensure_bundle_manifest(project_dir)
    if proj is None:
        logger.warning("Project file not found in: %s", project_dir)
        return None

    add_to_recent(str(project_dir))
    return proj


def create_project(
    project_dir: Path,
    class_name: str = "object",
    *,
    class_names: list[str] | None = None,
) -> DetectKitProject:
    """Create a new project in *project_dir* and persist defaults."""
    project_dir = project_dir.expanduser().resolve()
    ensure_project_bundle_layout(project_dir)
    resolved_class_names = normalize_class_names(
        class_names if class_names is not None else [class_name]
    )
    proj = DetectKitProject(project_dir=project_dir, class_names=resolved_class_names)
    save_project(proj)
    add_to_recent(str(project_dir))
    return proj


def save_project(proj: DetectKitProject) -> None:
    """Save *proj* to its canonical project file."""
    ensure_project_bundle_layout(proj.project_dir)
    proj.save(project_file_path(proj.project_dir))
    _ensure_bundle_manifest(proj.project_dir)
    _archive_legacy_project_file(proj.project_dir)
