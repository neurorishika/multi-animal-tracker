"""Tests for DetectKit project model and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource
from hydra_suite.detectkit.gui.project import default_project_parent_dir


def test_project_roundtrip(tmp_path: Path):
    proj = DetectKitProject(
        project_dir=tmp_path,
        class_names=["ant", "bee"],
        sources=[
            OBBSource(path=str(tmp_path / "ds1"), name="ds1"),
            OBBSource(path=str(tmp_path / "ds2"), name="ds2"),
        ],
    )
    proj_file = tmp_path / "detectkit_project.json"
    proj.save(proj_file)
    assert proj_file.exists()

    loaded = DetectKitProject.load(proj_file)
    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant", "bee"]
    assert len(loaded.sources) == 2
    assert loaded.sources[0].name == "ds1"


def test_project_loads_legacy_single_class_field(tmp_path: Path):
    proj_file = tmp_path / "detectkit_project.json"
    proj_file.write_text(
        json.dumps(
            {
                "version": 1,
                "project_dir": str(tmp_path),
                "class_name": "ant",
                "sources": [],
            }
        ),
        encoding="utf-8",
    )

    loaded = DetectKitProject.load(proj_file)

    assert loaded.class_name == "ant"
    assert loaded.class_names == ["ant"]


def test_project_defaults():
    proj = DetectKitProject(project_dir=Path("/tmp/test"))
    assert proj.class_name == "object"
    assert proj.class_names == ["object"]
    assert proj.sources == []
    assert proj.split_train == 0.8
    assert proj.split_val == 0.2
    assert proj.seed == 42


def test_obb_source_roundtrip():
    src = OBBSource(path="/data/obb_ds", name="my_dataset")
    d = src.to_dict()
    restored = OBBSource.from_dict(d)
    assert restored.path == "/data/obb_ds"
    assert restored.name == "my_dataset"


def test_default_project_parent_dir_uses_hydra_projects_root(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    assert default_project_parent_dir() == tmp_path / "hydra-projects" / "DetectKit"
