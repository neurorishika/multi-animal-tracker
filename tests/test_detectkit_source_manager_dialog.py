"""Tests for DetectKit SourceManagerDialog."""

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
    from hydra_suite.detectkit.gui.models import DetectKitProject

    return DetectKitProject(project_dir=tmp_path, class_names=["ant"])


def test_source_manager_dialog_imports(qapp):
    pass  # noqa: F401


def test_source_manager_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_source_manager_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    # Should have a Close button, not Ok/Cancel
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_source_manager_shows_existing_sources(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource

    proj = _make_proj(tmp_path)
    proj.sources = [OBBSource(path=str(tmp_path), name="ds1")]
    dlg = SourceManagerDialog(proj)
    assert dlg._source_list.count() == 1


def test_source_manager_remove_selected(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog
    from hydra_suite.detectkit.gui.models import OBBSource

    proj = _make_proj(tmp_path)
    proj.sources = [
        OBBSource(path=str(tmp_path / "a"), name="a"),
        OBBSource(path=str(tmp_path / "b"), name="b"),
    ]
    dlg = SourceManagerDialog(proj)
    dlg._source_list.setCurrentRow(0)
    dlg._remove_selected()
    assert len(proj.sources) == 1
    assert dlg._source_list.count() == 1


def test_source_manager_has_add_remove_buttons(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.source_manager import SourceManagerDialog

    dlg = SourceManagerDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_add")
    assert hasattr(dlg, "btn_remove")
