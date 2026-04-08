"""Tests for ClassKit class-editor preset persistence."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hydra_suite.classkit.gui.dialogs.class_editor import (  # noqa: E402
    ClassEditorDialog,
)


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


def test_class_editor_can_save_current_scheme_as_custom_preset(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.class_editor.QInputDialog.getText",
        lambda *_args, **_kwargs: ("Shared Tags", True),
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.class_editor.QMessageBox.information",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Ok,
    )

    dialog = ClassEditorDialog(
        scheme_dict={
            "name": "shared_tags",
            "description": "Shared tag scheme",
            "factors": [
                {
                    "name": "tag",
                    "labels": ["red", "blue"],
                    "shortcut_keys": ["r", "b"],
                }
            ],
            "training_modes": ["flat_custom"],
        }
    )

    dialog._save_current_as_preset()

    preset_index = dialog._preset_combo.findData("custom:shared_tags")
    assert preset_index >= 0
