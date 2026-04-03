# tests/test_stratified_split.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.dataset_inspector import DatasetItem, stratified_split_items


def _write_label(path: Path, class_ids: list[int]):
    """Write OBB labels with given class IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cid in class_ids:
        lines.append(str(cid) + " 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stratified_split_preserves_class_proportions(tmp_path: Path):
    """Each split should have roughly proportional class representation."""
    items = []
    for i in range(20):
        img_path = tmp_path / ("img_" + str(i) + ".jpg")
        lbl_path = tmp_path / ("img_" + str(i) + ".txt")
        img = np.full((80, 120, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        cls_id = 0 if i < 14 else 1
        _write_label(lbl_path, [cls_id])
        items.append(
            DatasetItem(image_path=str(img_path), label_path=str(lbl_path), split="all")
        )

    result = stratified_split_items(items, split_cfg=(0.7, 0.3, 0.0), seed=42)
    train_items = result["train"]
    val_items = result["val"]

    def count_classes(split_items):
        counts = {0: 0, 1: 0}
        for item in split_items:
            for line in Path(item.label_path).read_text().strip().splitlines():
                cid = int(float(line.split()[0]))
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    train_counts = count_classes(train_items)
    val_counts = count_classes(val_items)

    assert val_counts.get(1, 0) >= 1, "Class 1 should appear in val split"
    assert train_counts.get(1, 0) >= 1, "Class 1 should appear in train split"


def test_stratified_split_handles_single_class():
    """Single-class datasets should still split normally."""
    items = [
        DatasetItem(
            image_path="/img_" + str(i) + ".jpg",
            label_path="/lbl_" + str(i) + ".txt",
            split="all",
        )
        for i in range(10)
    ]
    result = stratified_split_items(items, split_cfg=(0.7, 0.3, 0.0), seed=42)
    assert len(result["train"]) + len(result["val"]) == 10
