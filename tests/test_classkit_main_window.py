from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QPixmap = QtGui.QPixmap
QApplication = QtWidgets.QApplication

MainWindow = pytest.importorskip("hydra_suite.classkit.gui.main_window").MainWindow


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


def test_reset_analysis_view_restores_explore_embedding_mode(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window._show_model_umap = True
    window._show_model_pca = True
    window.btn_umap_model.setEnabled(True)
    window.btn_pca_model.setEnabled(True)
    window.btn_umap_embedding.setChecked(False)
    window.btn_umap_model.setChecked(True)
    window.btn_pca_model.setChecked(True)

    window._reset_analysis_view()

    assert window.explorer_mode == "explore"
    assert window._show_model_umap is False
    assert window._show_model_pca is False
    assert window.btn_umap_embedding.isChecked() is True
    assert window.btn_umap_model.isChecked() is False
    assert window.btn_pca_model.isChecked() is False


def test_outline_control_only_visible_in_labeling_mode(qapp) -> None:
    window = MainWindow()

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True

    window.set_explorer_mode("labeling")
    assert window.outline_threshold_label.isHidden() is False
    assert window.outline_threshold_spin.isHidden() is False

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True


def test_empty_hover_clears_preview_without_active_label_selection(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = None
    window.hover_locked = False
    window.last_preview_index = 5
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    window.on_explorer_empty_hover()

    assert window.last_preview_index is None
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "Hover a point to preview the source image" in window.preview_info.text()


def test_training_settings_persist_in_project_config(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    window.project_path = tmp_path
    window._last_training_settings = {
        "mode": "flat_custom",
        "custom_input_size": 192,
        "tiny_width": 160,
        "tiny_height": 96,
        "device": "cpu",
    }

    window._save_last_training_settings()

    config = json.loads((tmp_path / "project.json").read_text())
    assert config["last_training_settings"]["custom_input_size"] == 192

    restored = MainWindow()
    restored._apply_project_config(config)

    assert restored._last_training_settings["mode"] == "flat_custom"
    assert restored._last_training_settings["tiny_width"] == 160


def test_default_training_settings_use_average_image_dimensions(
    qapp, tmp_path: Path
) -> None:
    from PIL import Image

    image_a = tmp_path / "img_a.png"
    image_b = tmp_path / "img_b.png"
    Image.new("RGB", (104, 60), color=(32, 64, 96)).save(image_a)
    Image.new("RGB", (136, 68), color=(96, 64, 32)).save(image_b)

    window = MainWindow()
    window.image_paths = [image_a, image_b]

    defaults = window._default_training_settings_from_project()

    assert defaults["tiny_width"] == 128
    assert defaults["tiny_height"] == 64
    assert defaults["custom_input_size"] == 96


def test_make_training_spec_uses_selected_initial_model_path(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()

    yolo_start = tmp_path / "previous_yolo.pt"
    yolo_start.write_text("weights", encoding="utf-8")
    yolo_spec = window._make_training_spec(
        {
            "base_model": "yolo26n-cls.pt",
            "initial_model_path": str(yolo_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_yolo"),
        "flat_yolo",
        True,
        tmp_path / "export_yolo",
    )

    assert yolo_spec.base_model == str(yolo_start)
    assert yolo_spec.resume_from == ""

    custom_start = tmp_path / "previous_custom.pth"
    custom_start.write_text("weights", encoding="utf-8")
    custom_spec = window._make_training_spec(
        {
            "custom_backbone": "resnet18",
            "initial_model_path": str(custom_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_custom"),
        "flat_custom",
        False,
        tmp_path / "export_custom",
    )

    assert custom_spec.base_model == ""
    assert custom_spec.resume_from == str(custom_start)


def test_validate_training_start_model_rejects_classkit_checkpoint_for_yolo(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    start_model = tmp_path / "wrong_start.pt"
    start_model.write_text("weights", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QMessageBox.warning",
        lambda _parent, title, text, *args, **kwargs: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        MainWindow,
        "_inspect_training_start_model",
        staticmethod(lambda _path: ("classkit_custom", "resnet18", None)),
    )

    ok = window._validate_training_start_model(
        {"mode": "flat_yolo", "initial_model_path": str(start_model)}
    )

    assert ok is False
    assert warnings
    assert warnings[0][0] == "Unsupported Starting Model"
    assert "not a YOLO model" in warnings[0][1]


def test_validate_training_start_model_rejects_non_classkit_checkpoint_for_custom(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    start_model = tmp_path / "wrong_start.pth"
    start_model.write_text("weights", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QMessageBox.warning",
        lambda _parent, title, text, *args, **kwargs: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        MainWindow,
        "_inspect_training_start_model",
        staticmethod(lambda _path: ("invalid", None, "not a ClassKit checkpoint")),
    )

    ok = window._validate_training_start_model(
        {
            "mode": "flat_custom",
            "initial_model_path": str(start_model),
            "custom_backbone": "resnet18",
        }
    )

    assert ok is False
    assert warnings
    assert warnings[0][0] == "Unsupported Starting Model"
    assert "not a compatible ClassKit CNN checkpoint" in warnings[0][1]


def test_validate_training_start_model_rejects_backbone_mismatch(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = MainWindow()
    start_model = tmp_path / "start_model.pth"
    start_model.write_text("weights", encoding="utf-8")

    warnings = []
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QMessageBox.warning",
        lambda _parent, title, text, *args, **kwargs: warnings.append((title, text)),
    )
    monkeypatch.setattr(
        MainWindow,
        "_inspect_training_start_model",
        staticmethod(lambda _path: ("classkit_custom", "tinyclassifier", None)),
    )

    ok = window._validate_training_start_model(
        {
            "mode": "flat_custom",
            "initial_model_path": str(start_model),
            "custom_backbone": "resnet18",
        }
    )

    assert ok is False
    assert warnings
    assert warnings[0][0] == "Backbone Mismatch"
    assert "Checkpoint backbone: tinyclassifier" in warnings[0][1]
