from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication

ClassKitTrainingDialog = pytest.importorskip(
    "hydra_suite.classkit.gui.dialogs.training"
).ClassKitTrainingDialog


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def test_training_dialog_auto_sizes_from_average_image_dimensions(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        average_image_size=(101.0, 67.0),
    )

    assert dialog.tiny_width_spin.value() == 96
    assert dialog.tiny_height_spin.value() == 64
    assert dialog._tiny_in_custom_width_spin.value() == 96
    assert dialog._tiny_in_custom_height_spin.value() == 64
    assert dialog._custom_input_size_spin.value() == 96


def test_training_dialog_restores_initial_settings(qapp) -> None:
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        average_image_size=(101.0, 67.0),
        initial_settings={
            "mode": "flat_custom",
            "device": "cpu",
            "compute_runtime": "cpu",
            "custom_backbone": "resnet18",
            "custom_input_size": 192,
            "tiny_width": 160,
            "tiny_height": 96,
            "epochs": 30,
            "batch": 16,
            "lr": 0.002,
            "patience": 7,
            "custom_fine_tune_method": "layerwise_lr_decay",
            "custom_layerwise_lr_decay": 0.6,
            "brightness": 0.2,
            "contrast": 0.15,
            "initial_model_path": "/tmp/previous_model.pth",
        },
    )

    assert dialog.mode_combo.currentData() == "flat_custom"
    assert dialog._custom_backbone_combo.currentData() == "resnet18"
    assert dialog._custom_input_size_spin.value() == 192
    assert dialog._custom_fine_tune_method_combo.currentData() == "layerwise_lr_decay"
    assert dialog._custom_layerwise_decay_spin.value() == pytest.approx(0.6)
    assert dialog.brightness_spin.value() == pytest.approx(0.2)
    assert dialog.contrast_spin.value() == pytest.approx(0.15)
    assert dialog.tiny_width_spin.value() == 160
    assert dialog.tiny_height_spin.value() == 96
    assert dialog._custom_epochs_spin.value() == 30
    assert dialog._custom_batch_spin.value() == 16
    assert dialog._custom_patience_spin.value() == 7
    assert dialog.get_settings()["initial_model_path"] == "/tmp/previous_model.pth"


def test_training_dialog_exposes_custom_and_yolo_modes_only(qapp) -> None:
    dialog = ClassKitTrainingDialog(n_labeled=8, class_choices=["a", "b"])

    modes = [dialog.mode_combo.itemData(i) for i in range(dialog.mode_combo.count())]

    assert modes == ["flat_custom", "flat_yolo"]


def test_training_dialog_filters_recent_model_choices_by_mode(qapp) -> None:
    recent_yolo = str(Path("/tmp/run_a/weights/best.pt").resolve())
    recent_custom = str(Path("/tmp/run_b/weights/best.pth").resolve())
    dialog = ClassKitTrainingDialog(
        n_labeled=8,
        class_choices=["a", "b"],
        recent_model_paths=[
            "/tmp/run_a/weights/best.pt",
            "/tmp/run_b/weights/best.pth",
        ],
    )

    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("flat_custom"))
    assert dialog._compatible_recent_model_paths() == [recent_custom]

    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("flat_yolo"))
    assert dialog._compatible_recent_model_paths() == [recent_yolo]
