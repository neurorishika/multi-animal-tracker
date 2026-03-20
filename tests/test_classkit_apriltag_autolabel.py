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
