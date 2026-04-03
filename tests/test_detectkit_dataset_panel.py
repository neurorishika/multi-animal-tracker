"""Tests for dataset panel utilities."""

from __future__ import annotations

from pathlib import Path

from hydra_suite.detectkit.ui.utils import find_label_for_image, list_images_in_source


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
