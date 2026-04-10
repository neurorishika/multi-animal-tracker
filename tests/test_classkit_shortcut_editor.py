from __future__ import annotations

import gc
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication
QKeySequence = QtGui.QKeySequence
Qt = QtCore.Qt

ShortcutEditorDialog = pytest.importorskip(
    "hydra_suite.classkit.gui.dialogs.shortcut_editor"
).ShortcutEditorDialog


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


def test_review_shortcut_defaults_use_plus_and_minus(qapp) -> None:
    dialog = ShortcutEditorDialog()
    shortcuts = dialog.get_shortcuts()

    assert shortcuts["Approve review label"] == QKeySequence(Qt.Key.Key_Plus).toString()
    assert shortcuts["Reject review label"] == QKeySequence(Qt.Key.Key_Minus).toString()
