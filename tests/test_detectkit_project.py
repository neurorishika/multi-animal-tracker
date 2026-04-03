"""Tests for DetectKit project model and persistence."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.detectkit.ui.models import DetectKitProject, OBBSource


def test_project_roundtrip(tmp_path: Path):
    proj = DetectKitProject(
        project_dir=tmp_path,
        class_name="ant",
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
    assert len(loaded.sources) == 2
    assert loaded.sources[0].name == "ds1"


def test_project_defaults():
    proj = DetectKitProject(project_dir=Path("/tmp/test"))
    assert proj.class_name == "object"
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
