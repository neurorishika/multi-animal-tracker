from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from hydra_suite.classkit.jobs.task_workers import ExportWorker


def _run_worker_and_collect_error(worker: ExportWorker) -> list[str]:
    errors: list[str] = []
    worker.signals.error.connect(errors.append)
    worker.run()
    return errors


def test_label_expansion_auto_creates_temp_dir(tmp_path: Path) -> None:
    """Label expansion should auto-create a temp dir when none is provided."""
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
    )

    errors = _run_worker_and_collect_error(worker)

    # Should fail because img_0.jpg doesn't exist, but NOT because of
    # a missing canonical guard (which was removed).
    assert errors
    assert "Label expansion requires canonical training space" not in errors[0]


def test_export_worker_no_labeled_samples(
    tmp_path: Path,
) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[-1],  # no valid labeled samples
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
    )

    errors = _run_worker_and_collect_error(worker)

    assert errors
    assert "No labeled samples found to export" in errors[0]


def test_export_worker_collect_valid_labels_uses_stratified_split(
    tmp_path: Path,
) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(6)],
        labels=[0, 0, 0, 0, 1, 1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        val_fraction=0.33,
    )

    _image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    label_split_counts = Counter(zip(labels, splits))
    assert len(labels) == 6
    assert splits.count("val") == 2
    assert label_split_counts[(0, "val")] == 1
    assert label_split_counts[(1, "val")] == 1


def test_export_worker_split_planning_ignores_unlabeled_items(tmp_path: Path) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(8)],
        labels=[0, 0, 0, 0, 1, 1, -1, -1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        val_fraction=0.33,
    )

    image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    assert len(image_paths) == 6
    assert len(labels) == 6
    assert len(splits) == 6
    assert splits.count("val") == 2


def test_export_worker_split_planning_uses_requested_strategy(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_dataset_splits(
        labels, *, strategy, val_fraction, test_fraction, seed=42
    ):
        captured["labels"] = list(labels)
        captured["strategy"] = strategy
        captured["val_fraction"] = val_fraction
        captured["test_fraction"] = test_fraction
        return ["train"] * len(labels)

    monkeypatch.setattr(
        "hydra_suite.classkit.jobs.task_workers.build_dataset_splits",
        _fake_build_dataset_splits,
    )

    worker = ExportWorker(
        image_paths=[tmp_path / f"img_{idx}.jpg" for idx in range(4)],
        labels=[0, 0, 1, 1],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left", 1: "right"},
        split_strategy="random",
        val_fraction=0.25,
    )

    _image_paths, labels, splits, _class_names = worker._collect_valid_labels()

    assert labels == [0, 0, 1, 1]
    assert splits == ["train", "train", "train", "train"]
    assert captured == {
        "labels": [0, 0, 1, 1],
        "strategy": "random",
        "val_fraction": 0.25,
        "test_fraction": 0.0,
    }


def test_export_worker_force_monochrome_materializes_grayscale_copies(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "color.png"
    Image.new("RGB", (12, 10), color=(200, 80, 20)).save(image_path)

    worker = ExportWorker(
        image_paths=[image_path],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        force_monochrome=True,
    )

    worker._prepare_export_workspace()
    image_paths, labels, splits, _class_names = worker._collect_valid_labels()
    converted_paths, converted_labels, converted_splits = worker._apply_monochrome_mode(
        image_paths, labels, splits
    )

    converted = np.asarray(Image.open(converted_paths[0]).convert("RGB"))

    assert converted_paths[0] != image_path
    assert converted_labels == labels
    assert converted_splits == splits
    assert np.array_equal(converted[..., 0], converted[..., 1])
    assert np.array_equal(converted[..., 1], converted[..., 2])


def test_export_worker_ultralytics_allows_empty_validation_split(
    tmp_path: Path,
) -> None:
    image_paths = []
    for idx, color in enumerate(((200, 80, 20), (20, 120, 220))):
        image_path = tmp_path / f"sample_{idx}.png"
        Image.new("RGB", (12, 10), color=color).save(image_path)
        image_paths.append(image_path)

    output_path = tmp_path / "ultralytics_out"
    worker = ExportWorker(
        image_paths=image_paths,
        labels=[0, 0],
        output_path=output_path,
        format="ultralytics",
        class_names={0: "left"},
        val_fraction=0.0,
    )

    errors = _run_worker_and_collect_error(worker)

    train_files = sorted((output_path / "train" / "left").glob("*.png"))
    val_files = list((output_path / "val").rglob("*.png"))

    assert errors == []
    assert len(train_files) == 2
    assert val_files == []
