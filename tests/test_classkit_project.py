"""Tests for ClassKit project-path helpers."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.classkit.gui.project import default_project_parent_dir


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "ClassKit"
