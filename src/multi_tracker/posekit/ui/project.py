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
from .models import Project
from .utils import _resolve_project_path, list_images

# -----------------------------
# Bootstrap / project discovery
# -----------------------------


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
    Legacy fallbacks are still checked for migration.
    """
    dataset_dir, images_dir, out_root, _labels_dir, project_path = (
        resolve_dataset_paths(dataset_dir)
    )
    candidates: List[Path] = [
        project_path,
        out_root / "labels" / DEFAULT_PROJECT_NAME,  # Legacy
        dataset_dir / DEFAULT_PROJECT_NAME,  # Legacy
        dataset_dir / "labels" / DEFAULT_PROJECT_NAME,  # Legacy
    ]
    for p in candidates:
        if p.exists():
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

    # Suggest DataSieve preprocessing for very large datasets.
    image_count = len(list_images(images_dir))
    if image_count > LARGE_DATASET_SIEVE_THRESHOLD:
        msg = QMessageBox(None)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Large Dataset Detected")
        msg.setText(
            f"This dataset has {image_count:,} images.\n\n"
            "For large datasets, DataSieve can reduce near-duplicates and create "
            "a smaller representative subset before labeling."
        )
        msg.setInformativeText("Open this dataset in DataSieve now?")
        btn_open = msg.addButton("Open in DataSieve", QMessageBox.AcceptRole)
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
                        "multi_tracker.tools.data_sieve.gui",
                        str(dataset_dir),
                    ],
                    start_new_session=True,
                )
            except Exception as exc:
                QMessageBox.warning(
                    None,
                    "Launch Failed",
                    f"Could not launch DataSieve:\n{exc}",
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
