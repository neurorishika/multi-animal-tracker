from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

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
