from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_tracker.classkit.store.db import ClassKitDB


def _seed_model_entry(
    db: ClassKitDB, artifact_path: Path, *, mode: str = "flat_tiny"
) -> int:
    artifact_path.write_bytes(b"weights")
    return int(
        db.save_model_cache(
            mode=mode,
            artifact_paths=[str(artifact_path)],
            class_names=["alpha", "beta"],
            canonicalize_mat=True,
            best_val_acc=0.91,
            num_classes=2,
            meta={"owner": "test"},
        )
    )


def test_set_model_cache_display_name_round_trip(tmp_path: Path) -> None:
    db = ClassKitDB(tmp_path / "classkit.db")
    model_id = _seed_model_entry(db, tmp_path / "best.pth")

    updated = db.set_model_cache_display_name(model_id, "Experiment A")
    assert updated == 1

    entries = db.list_model_caches()
    assert len(entries) == 1
    assert entries[0]["id"] == model_id
    assert entries[0].get("display_name") == "Experiment A"
    assert entries[0].get("meta", {}).get("owner") == "test"

    # Clearing the name removes it from metadata while preserving other keys.
    cleared = db.set_model_cache_display_name(model_id, "   ")
    assert cleared == 1

    entries_after_clear = db.list_model_caches()
    assert len(entries_after_clear) == 1
    assert "display_name" not in entries_after_clear[0]
    assert entries_after_clear[0].get("meta", {}).get("owner") == "test"


def test_set_model_cache_display_name_missing_id_returns_zero(tmp_path: Path) -> None:
    db = ClassKitDB(tmp_path / "classkit.db")

    updated = db.set_model_cache_display_name(999999, "unused")

    assert updated == 0


def test_delete_model_cache_entry_removes_row(tmp_path: Path) -> None:
    db = ClassKitDB(tmp_path / "classkit.db")
    model_a = _seed_model_entry(db, tmp_path / "a.pth", mode="flat_tiny")
    model_b = _seed_model_entry(db, tmp_path / "b.pth", mode="flat_yolo")

    deleted = db.delete_model_cache_entry(model_a)
    assert deleted == 1

    remaining_entries = db.list_model_caches()
    remaining_ids = {int(entry["id"]) for entry in remaining_entries}
    assert remaining_ids == {model_b}

    deleted_again = db.delete_model_cache_entry(model_a)
    assert deleted_again == 0


def test_list_model_caches_infers_tiny_best_val_acc(tmp_path: Path) -> None:
    db = ClassKitDB(tmp_path / "classkit.db")

    run_dir = tmp_path / "run_tiny"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    artifact = weights_dir / "best.pth"
    artifact.write_bytes(b"ckpt")
    (run_dir / "tiny_metrics.json").write_text(
        json.dumps({"best_val_acc": 0.83, "history": [{"val_acc": 0.80}]}),
        encoding="utf-8",
    )

    db.save_model_cache(
        mode="flat_tiny",
        artifact_paths=[str(artifact)],
        class_names=["a", "b"],
        best_val_acc=0.0,
        num_classes=2,
    )

    entries = db.list_model_caches()
    assert len(entries) == 1
    assert entries[0]["best_val_acc"] == pytest.approx(0.83)


def test_list_model_caches_infers_yolo_best_val_acc(tmp_path: Path) -> None:
    db = ClassKitDB(tmp_path / "classkit.db")

    run_dir = tmp_path / "run_yolo"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    artifact = weights_dir / "best.pt"
    artifact.write_bytes(b"ckpt")
    (run_dir / "results.csv").write_text(
        "epoch,metrics/accuracy_top1,metrics/accuracy_top5\n"
        "1,0.60,0.88\n"
        "2,0.79,0.94\n",
        encoding="utf-8",
    )

    db.save_model_cache(
        mode="flat_yolo",
        artifact_paths=[str(artifact)],
        class_names=["a", "b"],
        best_val_acc=0.0,
        num_classes=2,
    )

    entries = db.list_model_caches()
    assert len(entries) == 1
    assert entries[0]["best_val_acc"] == pytest.approx(0.79)
