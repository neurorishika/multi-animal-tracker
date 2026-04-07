"""Smoke tests: trackerkit orchestrators are constructible."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def test_tracking_orchestrator_constructed(main_window):
    assert main_window._tracking_orch is not None


def test_config_orchestrator_constructed(main_window):
    assert main_window._config_orch is not None


def test_session_orchestrator_constructed(main_window):
    assert main_window._session_orch is None  # placeholder until Task 19
