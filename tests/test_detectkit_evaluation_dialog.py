"""Tests for DetectKit EvaluationDialog."""

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


def test_evaluation_dialog_imports(qapp):
    pass  # noqa: F401


def test_evaluation_dialog_is_base_dialog(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert isinstance(dlg, BaseDialog)


def test_evaluation_dialog_has_close_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    close_btn = dlg._buttons.button(QDialogButtonBox.StandardButton.Close)
    assert close_btn is not None


def test_evaluation_dialog_has_analyze_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_analyze")


def test_evaluation_dialog_has_analysis_view(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "_analysis_view")


def test_evaluation_dialog_no_sources_message(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    dlg = EvaluationDialog(proj)
    dlg._run_dataset_analysis()
    assert "No dataset sources" in dlg._analysis_view.toPlainText()


def test_evaluation_dialog_has_quick_test_button(qapp, tmp_path):
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    dlg = EvaluationDialog(_make_proj(tmp_path))
    assert hasattr(dlg, "btn_quick_test")
    assert dlg.btn_quick_test.isEnabled()


def test_evaluation_dialog_quick_test_no_model_shows_message(
    qapp, tmp_path, monkeypatch
):
    """Quick test with no active_model_path shows an informative message (not a crash)."""
    from hydra_suite.detectkit.gui.dialogs.evaluation_dialog import EvaluationDialog

    proj = _make_proj(tmp_path)
    proj.active_model_path = ""  # no active model
    dlg = EvaluationDialog(proj)

    shown = []
    monkeypatch.setattr(
        "hydra_suite.detectkit.gui.dialogs.evaluation_dialog.QMessageBox.information",
        lambda *a, **kw: shown.append(a[2]),
    )
    dlg._quick_test()
    assert shown, "Expected an informative message when no model is active"
