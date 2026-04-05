from __future__ import annotations

import json
import os
from pathlib import Path

from tests.helpers.module_loader import load_src_module


def _load_mod():
    return load_src_module(
        "hydra_suite/data/dataset_merge.py",
        "dataset_merge_under_test",
    )


def _write_obb_label(path: Path, class_id: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",
        encoding="utf-8",
    )


def _write_source_dataset(
    root: Path,
    *,
    image_name: str = "frame.jpg",
    image_bytes: bytes = b"image-data",
    class_id: int = 0,
) -> None:
    image_path = root / "images" / image_name
    label_path = root / "labels" / f"{Path(image_name).stem}.txt"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(image_bytes)
    _write_obb_label(label_path, class_id=class_id)


def test_detect_dataset_layout_uses_yaml_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_mod()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "dataset.yaml").write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "_read_dataset_yaml",
        lambda _path: {
            "train": "images/train",
            "val": str(tmp_path / "external" / "images" / "val"),
        },
    )

    layout = mod.detect_dataset_layout(dataset_root)

    assert layout["train"] == (
        os.path.join(str(dataset_root), "images/train"),
        os.path.join(str(dataset_root), "labels/train"),
    )
    assert layout["val"] == (
        str(tmp_path / "external" / "images" / "val"),
        str(tmp_path / "external" / "labels" / "val"),
    )


def test_validate_and_rewrite_labels_round_trip(tmp_path: Path) -> None:
    mod = _load_mod()
    labels_dir = tmp_path / "labels"
    _write_obb_label(labels_dir / "a.txt", class_id=1)
    _write_obb_label(labels_dir / "nested" / "b.txt", class_id=3)

    class_ids, total = mod.validate_labels(labels_dir)
    assert class_ids == {1, 3}
    assert total == 2

    mod.rewrite_labels_to_single_class(labels_dir, class_id=7)

    class_ids, total = mod.validate_labels(labels_dir)
    assert class_ids == {7}
    assert total == 2


def test_merge_datasets_deduplicates_duplicate_images(tmp_path: Path) -> None:
    mod = _load_mod()
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    _write_source_dataset(src1, image_name="dup.jpg", image_bytes=b"same-bytes")
    _write_source_dataset(src2, image_name="dup.jpg", image_bytes=b"same-bytes")

    merged_dir = Path(
        mod.merge_datasets(
            [
                {"name": "s1", "path": str(src1)},
                {"name": "s2", "path": str(src2)},
            ],
            tmp_path / "out",
            class_name="ant",
            split_cfg={"train": 1.0, "val": 0.0},
            dedup=True,
        )
    )

    manifest = json.loads((merged_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stats"] == {
        "total_images": 2,
        "unique_images": 1,
        "duplicates_skipped": 1,
    }
    assert (merged_dir / "classes.txt").read_text(encoding="utf-8").strip() == "ant"
    assert len(list((merged_dir / "images" / "train").glob("*.jpg"))) == 1
    assert len(list((merged_dir / "labels" / "train").glob("*.txt"))) == 1


def test_merge_datasets_renames_colliding_unique_images(tmp_path: Path) -> None:
    mod = _load_mod()
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    _write_source_dataset(src1, image_name="shared.jpg", image_bytes=b"first")
    _write_source_dataset(src2, image_name="shared.jpg", image_bytes=b"second")

    merged_dir = Path(
        mod.merge_datasets(
            [
                {"name": "s1", "path": str(src1)},
                {"name": "s2", "path": str(src2)},
            ],
            tmp_path / "out",
            class_name="ant",
            split_cfg={"train": 1.0, "val": 0.0},
            dedup=False,
        )
    )

    output_names = sorted(
        path.name for path in (merged_dir / "images" / "train").glob("*.jpg")
    )
    assert output_names == ["shared.jpg", "shared_1.jpg"]
