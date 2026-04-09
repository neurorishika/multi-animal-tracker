"""Tests for PoseKit bundle persistence and legacy migration helpers."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.posekit.gui.constants import (
    DEFAULT_DATASET_IMAGES_DIR,
    DEFAULT_POSEKIT_PROJECT_DIR,
    DEFAULT_PROJECT_NAME,
)
from hydra_suite.posekit.gui.models import Project
from hydra_suite.posekit.gui.project import (
    canonical_project_path,
    find_project,
    open_project_from_path,
    save_project_state,
)


def _make_posekit_project(dataset_root: Path) -> Project:
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
        class_names=["ant"],
        keypoint_names=["head", "tail"],
        skeleton_edges=[(0, 1)],
    )


def test_posekit_save_project_state_uses_bundle_layout(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)

    saved_path = save_project_state(project)

    assert saved_path == canonical_project_path(project.out_root)
    assert saved_path.exists()
    assert (project.out_root / "hydra_project.json").exists()
    assert (project.out_root / "artifacts" / "cache").is_dir()
    assert (project.out_root / "artifacts" / "exports").is_dir()


def test_posekit_find_project_prefers_bundle_state_path(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    save_project_state(project)

    found = find_project(dataset_root)

    assert found == canonical_project_path(project.out_root)


def test_posekit_open_project_from_manifest_loads_bundled_state(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    save_project_state(project)

    loaded = open_project_from_path(project.out_root / "hydra_project.json")

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert loaded.class_names == ["ant"]


def test_posekit_open_project_from_legacy_file_migrates_to_bundle(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    legacy_path = project.out_root / DEFAULT_PROJECT_NAME
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(project.to_json(), indent=2), encoding="utf-8")

    loaded = open_project_from_path(legacy_path)

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert canonical_project_path(project.out_root).exists()
    assert not legacy_path.exists()
    assert (project.out_root / "history" / f"legacy_{DEFAULT_PROJECT_NAME}").exists()


def test_posekit_open_project_from_manifest_recovers_using_legacy_file(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    project = _make_posekit_project(dataset_root)
    legacy_path = project.out_root / DEFAULT_PROJECT_NAME
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(project.to_json(), indent=2), encoding="utf-8")
    (project.out_root / "hydra_project.json").write_text(
        "{bad-manifest", encoding="utf-8"
    )

    loaded = open_project_from_path(project.out_root / "hydra_project.json")

    assert loaded is not None
    assert loaded.project_path == canonical_project_path(project.out_root)
    assert canonical_project_path(project.out_root).exists()
    assert (project.out_root / "history" / f"legacy_{DEFAULT_PROJECT_NAME}").exists()
