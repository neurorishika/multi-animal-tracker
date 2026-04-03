from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtWidgets import QDialog, QMessageBox

from .constants import (
    DEFAULT_DATASET_IMAGES_DIR,
    DEFAULT_POSEKIT_PROJECT_DIR,
    DEFAULT_PROJECT_NAME,
    LARGE_DATASET_SIEVE_THRESHOLD,
)
from .models import DataSource, Project
from .utils import _resolve_project_path, list_images

# -----------------------------
# Bootstrap / project discovery
# -----------------------------


def _resolve_images_dir(folder: Path) -> Path:
    """Auto-detect where images live within *folder*.

    Returns ``folder/images/`` when that subdirectory exists **and** contains
    image files (legacy layout), otherwise returns ``folder`` itself so that
    collections organised with images directly in the root are accepted
    without any additional structure requirements.
    """
    candidate = folder / DEFAULT_DATASET_IMAGES_DIR
    if candidate.is_dir() and list_images(candidate):
        return candidate
    return folder


def resolve_dataset_paths(dataset_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    """Resolve canonical dataset structure paths.

    Expected structure:
      dataset_dir/
        images/
        posekit_project/
          pose_project.json
          labels/
    """
    ds = dataset_dir.expanduser().resolve()
    images_dir = ds / DEFAULT_DATASET_IMAGES_DIR
    out_root = ds / DEFAULT_POSEKIT_PROJECT_DIR
    labels_dir = out_root / "labels"
    project_path = out_root / DEFAULT_PROJECT_NAME
    return ds, images_dir, out_root, labels_dir, project_path


def find_project(dataset_dir: Path) -> Optional[Path]:
    """Find an existing project for a dataset root.

    Primary location:
      dataset_dir/posekit_project/pose_project.json

    The primary location is trusted by virtue of its path — if the file
    exists there it belongs to this dataset even if the stored ``images_dir``
    is stale (e.g. after moving between machines).  ``load_project_with_repairs``
    will fix the paths afterwards.

    Legacy fallback paths are only accepted when the stored ``images_dir``
    matches, to avoid accidentally loading an unrelated project.
    """
    dataset_dir, images_dir, out_root, _labels_dir, project_path = (
        resolve_dataset_paths(dataset_dir)
    )

    # ── Primary location: trust by filesystem position ─────────────────────
    if project_path.exists():
        try:
            json.loads(project_path.read_text(encoding="utf-8"))  # validate JSON
            return project_path
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file — fall through to legacy search

    # ── Legacy fallbacks: require images_dir match to avoid false positives ─
    legacy_candidates: List[Path] = [
        out_root / "labels" / DEFAULT_PROJECT_NAME,
        dataset_dir / DEFAULT_PROJECT_NAME,
        dataset_dir / "labels" / DEFAULT_PROJECT_NAME,
    ]
    for p in legacy_candidates:
        if not p.exists():
            continue
        try:
            proj_data = json.loads(p.read_text(encoding="utf-8"))
            base = p.parent
            proj_images_dir = _resolve_project_path(
                Path(proj_data["images_dir"]), base, proj_data
            )
            if proj_images_dir.resolve() == images_dir.resolve():
                return p
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return None


def _repair_project_paths(proj: Project, dataset_dir: Path) -> bool:
    """Repair project paths when moved across machines. Returns True if changed."""
    (
        _dataset_dir,
        expected_images_dir,
        expected_out_root,
        expected_labels_dir,
        expected_project_path,
    ) = resolve_dataset_paths(dataset_dir)
    changed = False
    if proj.images_dir.resolve() != expected_images_dir.resolve():
        proj.images_dir = expected_images_dir
        changed = True
    if proj.labels_dir.resolve() != expected_labels_dir.resolve():
        proj.labels_dir = expected_labels_dir
        changed = True
    if proj.out_root.resolve() != expected_out_root.resolve():
        proj.out_root = expected_out_root
        changed = True
    if proj.project_path.resolve() != expected_project_path.resolve():
        proj.project_path = expected_project_path
        changed = True
    expected_out_root.mkdir(parents=True, exist_ok=True)
    expected_labels_dir.mkdir(parents=True, exist_ok=True)
    return changed


def load_project_with_repairs(project_path: Path, dataset_dir: Path) -> Project:
    """Load a project and repair any stale paths for the current machine."""
    proj = Project.from_json(project_path)
    if _repair_project_paths(proj, dataset_dir):
        proj.project_path.write_text(
            json.dumps(proj.to_json(), indent=2), encoding="utf-8"
        )
    return proj


def create_project_via_wizard(dataset_dir: Path) -> Optional[Project]:
    """Run the ProjectWizard dialog and create a new project from user input."""
    # Import here to avoid a circular dependency between project.py and dialogs.
    from .dialogs.project_wizard import ProjectWizard

    (
        dataset_dir,
        images_dir,
        _default_out_root,
        _default_labels_dir,
        default_project_path,
    ) = resolve_dataset_paths(dataset_dir)
    if not images_dir.exists() or not images_dir.is_dir():
        QMessageBox.warning(
            None,
            "Invalid Dataset",
            f"Dataset is missing `{DEFAULT_DATASET_IMAGES_DIR}/`:\n{dataset_dir}",
        )
        return None

    # Suggest FilterKit preprocessing for very large datasets.
    image_count = len(list_images(images_dir))
    if image_count > LARGE_DATASET_SIEVE_THRESHOLD:
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Large Dataset Detected")
        msg.setText(
            f"This dataset has {image_count:,} images.\n\n"
            "For large datasets, FilterKit can reduce near-duplicates and create "
            "a smaller representative subset before labeling."
        )
        msg.setInformativeText("Open this dataset in FilterKit now?")
        btn_open = msg.addButton("Open in FilterKit", QMessageBox.AcceptRole)
        btn_continue = msg.addButton("Continue in PoseKit", QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_open:
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "multi_tracker.filterkit.gui",
                        str(dataset_dir),
                    ],
                    start_new_session=True,
                )
            except Exception as exc:
                QMessageBox.warning(
                    None,
                    "Launch Failed",
                    f"Could not launch FilterKit:\n{exc}",
                )
            return None
        if clicked != btn_continue:
            return None

    wiz = ProjectWizard(dataset_dir, existing=None)
    if wiz.exec() != QDialog.Accepted:
        return None

    out_root, labels_dir = wiz.get_paths()
    class_names = wiz.get_classes()
    kpt_names = wiz.get_keypoints()
    edges = wiz.get_edges()
    autosave, pad = wiz.get_options()

    out_root.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    project_path = default_project_path
    proj = Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=project_path,
        class_names=class_names,
        keypoint_names=kpt_names,
        skeleton_edges=edges,
        autosave=autosave,
        bbox_pad_frac=pad,
    )
    project_path.write_text(json.dumps(proj.to_json(), indent=2), encoding="utf-8")
    return proj


def create_standalone_project_via_wizard(parent_widget=None) -> Optional[Project]:
    """Run NewProjectDialog and build a standalone multi-source project.

    Unlike ``create_project_via_wizard()``, the project file lives at a
    user-chosen location independent of any dataset folder, and one or more
    sources are added during creation rather than as an afterthought.
    """
    from .dialogs.project_wizard import NewProjectDialog

    dlg = NewProjectDialog(parent=parent_widget)
    if dlg.exec() != QDialog.Accepted:
        return None

    project_location = dlg.get_project_location()
    if project_location is None:
        return None

    class_names = dlg.get_classes()
    kpt_names = dlg.get_keypoints()
    edges = dlg.get_edges()
    autosave, pad = dlg.get_options()
    sources_to_add = dlg.get_sources()  # List[Tuple[Path, str]]

    if not sources_to_add:
        QMessageBox.warning(
            parent_widget, "No Sources", "At least one dataset source is required."
        )
        return None

    out_root = project_location
    out_root.mkdir(parents=True, exist_ok=True)
    labels_dir = out_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    project_path = out_root / DEFAULT_PROJECT_NAME

    # Build DataSource objects directly to avoid the legacy-promotion path inside
    # add_source_to_project (which would double-count the first source).
    sources: List[DataSource] = []
    for i, (ds_dir, desc) in enumerate(sources_to_add):
        source_id = f"source_{i}"
        src_images_dir = _resolve_images_dir(ds_dir)
        src_labels_dir = labels_dir / source_id
        src_labels_dir.mkdir(parents=True, exist_ok=True)
        sources.append(
            DataSource(
                source_id=source_id,
                dataset_root=ds_dir,
                images_dir=src_images_dir,
                labels_dir=src_labels_dir,
                description=desc or ds_dir.name,
            )
        )

    # Use the first source's images dir as the legacy images_dir field so
    # older code paths that read project.images_dir still work.
    first_images_dir = sources[0].images_dir

    proj = Project(
        images_dir=first_images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=project_path,
        class_names=class_names,
        keypoint_names=kpt_names,
        skeleton_edges=edges,
        autosave=autosave,
        bbox_pad_frac=pad,
        sources=sources,
    )
    project_path.write_text(json.dumps(proj.to_json(), indent=2), encoding="utf-8")

    return proj


def open_project_from_path(project_path: Path) -> Optional[Project]:
    """Load a project directly from its .json path (no dataset_dir required).

    Used when the user browses to a standalone project file.  Path repairs
    are skipped because the project location is trusted as-is.
    """
    try:
        proj = Project.from_json(project_path)
        return proj
    except Exception as exc:
        QMessageBox.critical(
            None, "Failed to open project", f"Could not load:\n{project_path}\n\n{exc}"
        )
        return None


def build_image_list(project: Project) -> list:
    """Return all image paths for *project*, spanning every registered source.

    Multi-source projects aggregate images from all ``project.sources`` in
    registration order.  Legacy single-source projects fall back to the
    top-level ``project.images_dir``.
    """
    if project.sources:
        imgs = []
        for src in project.sources:
            imgs.extend(list_images(src.images_dir))
        return imgs
    return list_images(project.images_dir)


def add_source_to_project(
    project: Project,
    dataset_dir: Path,
    description: str = "",
) -> DataSource:
    """Ingest a new source dataset into *project* and persist the change.

    If the project is currently in legacy single-source mode (``sources=[]``)
    the existing ``images_dir`` / ``labels_dir`` is first promoted to
    ``source_0`` so that routing remains consistent.

    Returns the newly created :class:`DataSource`.
    """
    dataset_dir = dataset_dir.expanduser().resolve()
    images_dir = _resolve_images_dir(dataset_dir)

    # --- promote legacy project to multi-source on first add ---
    if not project.sources:
        src0 = DataSource(
            source_id="source_0",
            dataset_root=project.images_dir.parent,
            images_dir=project.images_dir,
            labels_dir=project.labels_dir,
            description=project.images_dir.parent.name,
        )
        project.sources.append(src0)

    # --- generate a unique source_id ---
    existing_ids = {s.source_id for s in project.sources}
    i = len(project.sources)
    while f"source_{i}" in existing_ids:
        i += 1
    source_id = f"source_{i}"

    # --- per-source labels dir ---
    labels_dir = project.out_root / "labels" / source_id
    labels_dir.mkdir(parents=True, exist_ok=True)

    src = DataSource(
        source_id=source_id,
        dataset_root=dataset_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        description=description or dataset_dir.name,
    )
    project.sources.append(src)
    project.project_path.write_text(
        json.dumps(project.to_json(), indent=2), encoding="utf-8"
    )
    return src


def create_empty_startup_project() -> Project:
    """Create a minimal placeholder project for startup-without-dataset mode."""
    dataset_root = (Path.home() / ".posekit" / "startup").resolve()
    images_dir = dataset_root / DEFAULT_DATASET_IMAGES_DIR
    out_root = dataset_root / DEFAULT_POSEKIT_PROJECT_DIR
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=out_root / DEFAULT_PROJECT_NAME,
        class_names=["object"],
        keypoint_names=["kp1", "kp2"],
        skeleton_edges=[(0, 1)],
        autosave=True,
    )
