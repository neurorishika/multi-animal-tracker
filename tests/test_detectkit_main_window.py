"""Tests for the refactored DetectKit MainWindow."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTabWidget, QToolBar  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(scope="module")
def main_win(qapp):
    """Single shared DetectKitMainWindow — avoids per-test SVG GC crash."""
    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    win = DetectKitMainWindow()
    yield win


def test_main_window_has_toolbar(main_win):
    toolbars = main_win.findChildren(QToolBar)
    assert toolbars, "MainWindow must have at least one QToolBar"


def test_main_window_no_tab_widget(main_win):
    tabs = main_win.findChildren(QTabWidget)
    assert not tabs, "QTabWidget must be removed from MainWindow"


def test_main_window_has_tools_panel(main_win):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panels = main_win.findChildren(ToolsPanel)
    assert panels, "MainWindow must contain a ToolsPanel"


def test_main_window_tools_panel_fixed_width(main_win):
    from hydra_suite.detectkit.gui.panels.tools_panel import ToolsPanel

    panel = main_win.findChildren(ToolsPanel)[0]
    assert panel.maximumWidth() == 280
    assert panel.minimumWidth() == 280


def test_main_window_has_open_source_manager(main_win):
    assert hasattr(main_win, "_open_source_manager")


def test_main_window_has_open_training_dialog(main_win):
    assert hasattr(main_win, "_open_training_dialog")


def test_main_window_has_open_evaluation_dialog(main_win):
    assert hasattr(main_win, "_open_evaluation_dialog")


def test_main_window_has_open_history_dialog(main_win):
    assert hasattr(main_win, "_open_history_dialog")


def test_main_window_toolbar_hidden_on_welcome(qapp):
    """Fresh window (welcome screen) must have toolbar explicitly hidden."""
    from hydra_suite.detectkit.gui.main_window import DetectKitMainWindow

    fresh = DetectKitMainWindow()
    # _toolbar must be set explicitly invisible (not just hidden by parent)
    assert not fresh._toolbar.isVisibleTo(
        fresh
    ), "Toolbar should be hidden on welcome screen"


def test_main_window_toolbar_visible_after_project_load(qapp, main_win, tmp_path):
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path, class_names=["ant"])
    main_win._load_project(proj)
    # isVisibleTo(parent) returns True if widget would be visible when parent is shown
    assert main_win._toolbar.isVisibleTo(
        main_win
    ), "Toolbar should be visible after loading a project"


def test_save_project_no_deleted_panels(qapp, main_win, tmp_path):
    """_save_current_project must not raise (no old training/eval/history panels)."""
    from hydra_suite.detectkit.gui.models import DetectKitProject

    proj = DetectKitProject(project_dir=tmp_path / "proj2", class_names=["ant"])
    main_win._load_project(proj)
    main_win._save_current_project()  # Must not raise


def test_dialogs_init_exports(qapp):
    from hydra_suite.detectkit.gui import dialogs

    assert hasattr(dialogs, "NewProjectDialog")
    assert hasattr(dialogs, "SourceManagerDialog")
    assert hasattr(dialogs, "TrainingDialog")
    assert hasattr(dialogs, "EvaluationDialog")
    assert hasattr(dialogs, "HistoryDialog")
