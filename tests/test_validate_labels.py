from __future__ import annotations

from pathlib import Path

from tests.helpers.module_loader import load_src_module


def _load_mod():
    return load_src_module(
        "hydra_suite/trackerkit/gui/validate_labels.py",
        "trackerkit_validate_labels_under_test",
    )


def test_validate_label_file_accepts_valid_pose_label(tmp_path: Path) -> None:
    mod = _load_mod()
    label_path = tmp_path / "sample.txt"
    label_path.write_text(
        "0 0.5 0.5 0.2 0.2 0.2 0.3 2 0.8 0.7 1\n",
        encoding="utf-8",
    )

    is_valid, issues = mod.validate_label_file(label_path, kpt_count=2)

    assert is_valid is True
    assert issues == []


def test_validate_label_file_reports_bbox_and_keypoint_issues(tmp_path: Path) -> None:
    mod = _load_mod()
    label_path = tmp_path / "bad.txt"
    label_path.write_text(
        "0 nan 0.5 0.0001 0.0 1.2 0.2 3 0.6 1.5 0\n",
        encoding="utf-8",
    )

    is_valid, issues = mod.validate_label_file(label_path, kpt_count=2)

    assert is_valid is False
    assert any("Non-finite bbox cx" in issue for issue in issues)
    assert any("Bbox width too small" in issue for issue in issues)
    assert any("Bbox height too small" in issue for issue in issues)
    assert any("Invalid visibility 3" in issue for issue in issues)
    assert any("Keypoint 0 x out of range" in issue for issue in issues)


def test_validate_dataset_counts_valid_and_invalid_files(tmp_path: Path) -> None:
    mod = _load_mod()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "good.txt").write_text(
        "0 0.5 0.5 0.2 0.2 0.2 0.3 2 0.8 0.7 1\n",
        encoding="utf-8",
    )
    (labels_dir / "bad.txt").write_text("", encoding="utf-8")

    valid_count, invalid_count, fixed_count = mod.validate_dataset(
        labels_dir,
        kpt_count=2,
    )

    assert valid_count == 1
    assert invalid_count == 1
    assert fixed_count == 0
