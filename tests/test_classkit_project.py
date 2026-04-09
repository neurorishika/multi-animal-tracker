"""Tests for ClassKit project-path helpers."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.classkit.gui.project import (
    classkit_config_path,
    classkit_db_path,
    classkit_scheme_path,
    default_project_parent_dir,
    prepare_project_directory,
    project_exists,
)


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "ClassKit"


def test_prepare_project_directory_creates_bundle_layout(tmp_path: Path) -> None:
    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert (tmp_path / "hydra_project.json").exists()
    assert classkit_config_path(tmp_path).parent.is_dir()
    assert classkit_scheme_path(tmp_path).parent.is_dir()
    assert (tmp_path / "artifacts" / "models").is_dir()
    assert (tmp_path / "artifacts" / "exports").is_dir()
    assert project_exists(tmp_path) is True


def test_prepare_project_directory_migrates_legacy_root_files(tmp_path: Path) -> None:
    (tmp_path / "classkit.db").write_text("db", encoding="utf-8")
    (tmp_path / "project.json").write_text(
        json.dumps({"name": "colony", "classes": ["a", "b"]}, indent=2),
        encoding="utf-8",
    )
    (tmp_path / "scheme.json").write_text(
        json.dumps({"name": "scheme"}, indent=2),
        encoding="utf-8",
    )

    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert classkit_db_path(tmp_path).read_text(encoding="utf-8") == "db"
    assert (
        json.loads(classkit_config_path(tmp_path).read_text(encoding="utf-8"))["name"]
        == "colony"
    )
    assert (
        json.loads(classkit_scheme_path(tmp_path).read_text(encoding="utf-8"))["name"]
        == "scheme"
    )
    assert not (tmp_path / "classkit.db").exists()
    assert not (tmp_path / "project.json").exists()
    assert not (tmp_path / "scheme.json").exists()


def test_prepare_project_directory_recovers_from_malformed_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "hydra_project.json").write_text("{bad-manifest", encoding="utf-8")
    (tmp_path / "classkit.db").write_text("db", encoding="utf-8")

    db_path = prepare_project_directory(tmp_path)

    assert db_path == classkit_db_path(tmp_path)
    assert classkit_db_path(tmp_path).read_text(encoding="utf-8") == "db"
    assert (tmp_path / "hydra_project.json").exists()
