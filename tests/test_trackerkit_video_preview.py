"""Focused tests for TrackerKit preview seeking behavior."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    window = MainWindow()
    yield window
    window.close()


def test_timeline_navigation_renders_once(main_window, monkeypatch):
    main_window.video_total_frames = 10
    main_window.video_current_frame_idx = 1
    main_window._setup_panel.slider_timeline.setMaximum(9)
    main_window._setup_panel.slider_timeline.setValue(1)

    render_spy = MagicMock()
    monkeypatch.setattr(main_window._session_orch, "_display_current_frame", render_spy)

    main_window._goto_next_frame()

    assert main_window.video_current_frame_idx == 2
    assert main_window._setup_panel.slider_timeline.value() == 2
    assert render_spy.call_count == 1


def test_timeline_move_updates_label_without_render(main_window, monkeypatch):
    render_spy = MagicMock()
    monkeypatch.setattr(main_window._session_orch, "_display_current_frame", render_spy)

    main_window.video_total_frames = 10
    main_window._session_orch._on_timeline_moved(7)

    if main_window._setup_panel.slider_timeline.hasTracking():
        assert (
            main_window._setup_panel.lbl_current_frame.text()
            != "Frame: 7/9 (release to seek)"
        )
    else:
        assert (
            main_window._setup_panel.lbl_current_frame.text()
            == "Frame: 7/9 (release to seek)"
        )
    assert render_spy.call_count == 0
