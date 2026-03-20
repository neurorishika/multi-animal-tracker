"""Tests for ClassKit AprilTag auto-labeler."""

from __future__ import annotations

from multi_tracker.classkit.presets import apriltag_preset


def test_apriltag_preset_labels():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    labels = scheme.factors[0].labels
    assert labels[:10] == [f"tag_{i}" for i in range(10)]
    assert labels[-1] == "no_tag"
    assert len(labels) == 11  # tag_0..tag_9 + no_tag


def test_apriltag_preset_total_classes():
    scheme = apriltag_preset("tag36h11", max_tag_id=4)
    assert scheme.total_classes == 6  # tag_0..tag_4 + no_tag


def test_apriltag_preset_single_factor():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert len(scheme.factors) == 1


def test_apriltag_preset_factor_name_matches_family():
    scheme = apriltag_preset("tag25h9", max_tag_id=5)
    assert scheme.factors[0].name == "tag25h9"


def test_apriltag_preset_name():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.name == "apriltag_tag36h11"


def test_apriltag_preset_flat_training_modes_only():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.training_modes == ["flat_tiny", "flat_yolo"]


def test_apriltag_preset_max_tag_id_zero():
    scheme = apriltag_preset("tag36h11", max_tag_id=0)
    assert scheme.factors[0].labels == ["tag_0", "no_tag"]
    assert scheme.total_classes == 2


import sqlite3
from pathlib import Path

from multi_tracker.classkit.store.db import ClassKitDB


def _make_db(tmp_path: Path) -> ClassKitDB:
    """Create a DB with two test images."""
    db = ClassKitDB(tmp_path / "test.db")
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/a.jpg", "aaa"),
        )
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/b.jpg", "bbb"),
        )
        conn.commit()
    return db


def test_update_labels_with_confidence_batch_writes_label_and_confidence(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {
            "/img/a.jpg": ("tag_3", 0.8),
            "/img/b.jpg": ("no_tag", 1.0),
        }
    )
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT file_path, label, confidence FROM images ORDER BY file_path"
        ).fetchall()
    assert rows[0] == ("/img/a.jpg", "tag_3", 0.8)
    assert rows[1] == ("/img/b.jpg", "no_tag", 1.0)


def test_update_labels_with_confidence_batch_empty_is_noop(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch({})  # must not raise
    labels = db.get_all_labels()
    assert all(lbl is None for lbl in labels)


def test_clear_all_labels_sets_null(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch(
        {
            "/img/a.jpg": ("tag_3", 0.8),
        }
    )
    db.clear_all_labels()
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute("SELECT label, confidence FROM images").fetchall()
    assert all(row[0] is None and row[1] is None for row in rows)


def test_clear_all_labels_on_empty_db_is_noop(tmp_path):
    db = ClassKitDB(tmp_path / "empty.db")
    db.clear_all_labels()  # must not raise
