from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from multi_tracker.classkit.jobs.task_workers import ExportWorker


def _run_worker_and_collect_error(worker: ExportWorker) -> list[str]:
    errors: list[str] = []
    worker.signals.error.connect(errors.append)
    worker.run()
    return errors


def test_label_expansion_requires_canonical_space(tmp_path: Path) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[0],
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
        canonicalize=False,
    )

    errors = _run_worker_and_collect_error(worker)

    assert errors
    assert "Label expansion requires canonical training space" in errors[0]


def test_label_expansion_does_not_raise_canonical_guard_when_enabled(
    tmp_path: Path,
) -> None:
    worker = ExportWorker(
        image_paths=[tmp_path / "img_0.jpg"],
        labels=[-1],  # no valid labeled samples
        output_path=tmp_path / "out.csv",
        format="csv",
        class_names={0: "left"},
        label_expansion={"fliplr": {"left": "right"}},
        canonicalize=True,
    )

    errors = _run_worker_and_collect_error(worker)

    assert errors
    assert "Label expansion requires canonical training space" not in errors[0]
    assert "No labeled samples found to export" in errors[0]
