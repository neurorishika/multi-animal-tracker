"""Tests for DetectKit TrainingDialog — full feature set."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


def _make_proj(tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject, OBBSource

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    proj.sources = [OBBSource(path=str(tmp_path / "ds1"), name="ds1")]
    return proj


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------


def test_training_dialog_imports(qapp):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import (  # noqa: F401
        TrainingDialog,
        _TrainingWorker,
    )


def test_training_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_training_dialog_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_training_dialog_has_training_completed_signal(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "training_completed")


def test_training_dialog_has_start_cancel_buttons(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_start")
    assert hasattr(dlg, "btn_cancel")


def test_training_dialog_has_progress_bar(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "progress")


def test_training_dialog_has_log_view(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "log_view")


def test_training_dialog_start_always_enabled(qapp, tmp_path):
    """Start button is enabled by default; sources are validated at click time."""
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert dlg.btn_start.isEnabled()


def test_training_worker_is_base_worker(qapp):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import _TrainingWorker
    from hydra_suite.widgets.workers import BaseWorker

    assert issubclass(_TrainingWorker, BaseWorker)


# ---------------------------------------------------------------------------
# Roles group
# ---------------------------------------------------------------------------


def test_training_dialog_has_role_checkboxes(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "chk_role_obb_direct")
    assert hasattr(dlg, "chk_role_seq_detect")
    assert hasattr(dlg, "chk_role_seq_crop_obb")


def test_training_dialog_roles_roundtrip(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.role_obb_direct = True
    proj.role_seq_detect = False
    proj.role_seq_crop_obb = True
    dlg = TrainingDialog(proj)
    assert dlg.chk_role_obb_direct.isChecked() is True
    assert dlg.chk_role_seq_detect.isChecked() is False
    assert dlg.chk_role_seq_crop_obb.isChecked() is True


# ---------------------------------------------------------------------------
# Config group
# ---------------------------------------------------------------------------


def test_training_dialog_has_class_names_edit(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "class_names_edit")


def test_training_dialog_class_names_loaded(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.class_names = ["worker", "queen"]
    dlg = TrainingDialog(proj)
    text = dlg.class_names_edit.toPlainText()
    assert "worker" in text
    assert "queen" in text


def test_training_dialog_has_workspace_line(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "line_workspace")


def test_training_dialog_has_seed_spinbox(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.seed = 77
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_seed")
    assert dlg.spin_seed.value() == 77


def test_training_dialog_has_dedup_checkbox(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.dedup = False
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "chk_dedup")
    assert dlg.chk_dedup.isChecked() is False


def test_training_dialog_has_crop_derivation_widgets(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.crop_pad_ratio = 0.25
    proj.min_crop_size_px = 128
    proj.enforce_square = False
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_crop_pad")
    assert hasattr(dlg, "spin_crop_min_px")
    assert hasattr(dlg, "chk_crop_square")
    assert abs(dlg.spin_crop_pad.value() - 0.25) < 0.001
    assert dlg.spin_crop_min_px.value() == 128
    assert dlg.chk_crop_square.isChecked() is False


def test_training_dialog_has_device_combo(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "combo_device")


# ---------------------------------------------------------------------------
# Hyperparams group — extended fields
# ---------------------------------------------------------------------------


def test_training_dialog_has_workers_cache_auto_batch(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.workers = 4
    proj.cache = True
    proj.auto_batch = True
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_workers")
    assert hasattr(dlg, "chk_cache")
    assert hasattr(dlg, "chk_auto_batch")
    assert dlg.spin_workers.value() == 4
    assert dlg.chk_cache.isChecked() is True
    assert dlg.chk_auto_batch.isChecked() is True


def test_training_dialog_has_per_role_imgsz(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.imgsz_obb_direct = 800
    proj.imgsz_seq_detect = 960
    proj.imgsz_seq_crop_obb = 256
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "spin_imgsz_obb_direct")
    assert hasattr(dlg, "spin_imgsz_seq_detect")
    assert hasattr(dlg, "spin_imgsz_seq_crop_obb")
    assert dlg.spin_imgsz_obb_direct.value() == 800
    assert dlg.spin_imgsz_seq_detect.value() == 960
    assert dlg.spin_imgsz_seq_crop_obb.value() == 256


def test_training_dialog_load_from_project_populates_fields(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.epochs = 50
    proj.split_train = 0.7
    dlg = TrainingDialog(proj)
    assert dlg.spin_epochs.value() == 50


# ---------------------------------------------------------------------------
# Base Models group
# ---------------------------------------------------------------------------


def test_training_dialog_has_per_role_model_combos(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.model_obb_direct = "yolo26m-obb.pt"
    proj.model_seq_detect = "yolo26n.pt"
    proj.model_seq_crop_obb = "yolo26n-obb.pt"
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "combo_model_obb_direct")
    assert hasattr(dlg, "combo_model_seq_detect")
    assert hasattr(dlg, "combo_model_seq_crop_obb")
    assert dlg.combo_model_obb_direct.currentText() == "yolo26m-obb.pt"
    assert dlg.combo_model_seq_detect.currentText() == "yolo26n.pt"
    assert dlg.combo_model_seq_crop_obb.currentText() == "yolo26n-obb.pt"


# ---------------------------------------------------------------------------
# Augmentation group
# ---------------------------------------------------------------------------


def test_training_dialog_has_augmentation_group(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "aug_group")


def test_training_dialog_augmentation_fields_present(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    for attr in (
        "aug_fliplr",
        "aug_flipud",
        "aug_degrees",
        "aug_mosaic",
        "aug_mixup",
        "aug_hsv_h",
        "aug_hsv_s",
        "aug_hsv_v",
    ):
        assert hasattr(dlg, attr), f"Missing augmentation widget: {attr}"


def test_training_dialog_augmentation_roundtrip(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.aug_enabled = False
    proj.aug_fliplr = 0.0
    proj.aug_degrees = 45.0
    dlg = TrainingDialog(proj)
    assert dlg.aug_group.isChecked() is False
    assert abs(dlg.aug_fliplr.value() - 0.0) < 0.001
    assert abs(dlg.aug_degrees.value() - 45.0) < 0.001


# ---------------------------------------------------------------------------
# Publish group
# ---------------------------------------------------------------------------


def test_training_dialog_has_publish_widgets(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    proj.species = "ant"
    proj.model_tag = "v1"
    proj.auto_import = False
    proj.auto_select = True
    dlg = TrainingDialog(proj)
    assert hasattr(dlg, "line_species")
    assert hasattr(dlg, "line_model_tag")
    assert hasattr(dlg, "chk_auto_import")
    assert hasattr(dlg, "chk_auto_select")
    assert dlg.line_species.text() == "ant"
    assert dlg.line_model_tag.text() == "v1"
    assert dlg.chk_auto_import.isChecked() is False
    assert dlg.chk_auto_select.isChecked() is True


# ---------------------------------------------------------------------------
# Loss plot
# ---------------------------------------------------------------------------


def test_training_dialog_has_loss_plot(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    dlg = TrainingDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "loss_plot")


# ---------------------------------------------------------------------------
# Write-to-project round-trip
# ---------------------------------------------------------------------------


def test_training_dialog_write_to_project(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.training_dialog import TrainingDialog

    proj = _make_proj(tmp_path)
    dlg = TrainingDialog(proj)

    dlg.chk_role_seq_detect.setChecked(False)
    dlg.spin_epochs.setValue(25)
    dlg.spin_seed.setValue(99)
    dlg.aug_group.setChecked(False)
    dlg.line_species.setText("bee")

    dlg._write_to_project()

    assert proj.role_seq_detect is False
    assert proj.epochs == 25
    assert proj.seed == 99
    assert proj.aug_enabled is False
    assert proj.species == "bee"
