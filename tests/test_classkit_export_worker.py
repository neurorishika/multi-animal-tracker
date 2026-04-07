from __future__ import annotations

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
