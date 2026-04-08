"""Tests for dataset panel utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from hydra_suite.detectkit.gui.utils import (
    ensure_detectkit_source_structure,
    find_label_for_image,
    list_images_in_source,
    parse_obb_label,
    source_class_id_map,
)


def test_list_images_in_source_with_images_dir(tmp_path: Path):
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    (img_dir / "a.jpg").write_text("fake")
    (img_dir / "b.png").write_text("fake")
    (img_dir / "c.txt").write_text("not an image")
    images = list_images_in_source(str(tmp_path))
    assert len(images) == 2


def test_find_label_for_image(tmp_path: Path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    img = tmp_path / "images" / "train" / "frame001.jpg"
    img.write_text("fake")
    lbl = tmp_path / "labels" / "train" / "frame001.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8")
    result = find_label_for_image(img, str(tmp_path))
    assert result is not None
    assert result.name == "frame001.txt"


def test_detectkit_source_structure_requires_images_labels_and_classes(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()

    with pytest.raises(RuntimeError, match="classes.txt"):
        ensure_detectkit_source_structure(tmp_path)


def test_source_class_id_map_accepts_source_superset(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "classes.txt").write_text("bee\nant\nwasp\n", encoding="utf-8")

    class_id_map = source_class_id_map(tmp_path, ["ant", "bee"])

    assert class_id_map == {1: 0, 0: 1}


def test_parse_obb_label_filters_and_remaps_by_project_classes(tmp_path: Path):
    lbl = tmp_path / "filtered.txt"
    lbl.write_text(
        "1 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n" "2 0.2 0.2 0.8 0.2 0.8 0.7 0.2 0.7\n",
        encoding="utf-8",
    )

    dets = parse_obb_label(lbl, img_w=100, img_h=100, class_id_map={1: 0})

    assert [det["class_id"] for det in dets] == [0]
