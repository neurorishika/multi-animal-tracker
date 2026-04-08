"""Tests for the ClassKit launcher metadata and icon wiring."""

import sys
import types

import pytest


def test_classkit_launcher_sets_application_name_and_icon(
    monkeypatch: pytest.MonkeyPatch,
):
    app_module = pytest.importorskip("hydra_suite.classkit.app")

    calls = {
        "application_name": None,
        "display_name": None,
        "organization": None,
        "desktop_file_name": None,
        "window_icon": None,
        "window_icon_requests": 0,
        "resize": None,
        "show_maximized": False,
        "exec_called": False,
        "exit_code": None,
        "main_window_icon": None,
        "brand_icon_name": None,
    }

    class FakeIcon:
        def isNull(self):
            return False

    fake_icon = FakeIcon()

    class FakeApp:
        def __init__(self, _args):
            self._window_icon = None

        def setApplicationName(self, value):
            calls["application_name"] = value

        def setApplicationDisplayName(self, value):
            calls["display_name"] = value

        def setOrganizationName(self, value):
            calls["organization"] = value

        def setDesktopFileName(self, value):
            calls["desktop_file_name"] = value

        def setWindowIcon(self, icon):
            calls["window_icon"] = icon
            self._window_icon = icon

        def windowIcon(self):
            calls["window_icon_requests"] += 1
            return self._window_icon

        def exec(self):
            calls["exec_called"] = True
            return 0

    class FakeMainWindow:
        def setWindowIcon(self, icon):
            calls["main_window_icon"] = icon

        def resize(self, width, height):
            calls["resize"] = (width, height)

        def showMaximized(self):
            calls["show_maximized"] = True

    monkeypatch.setattr(app_module, "QApplication", FakeApp)
    fake_main_window_module = types.ModuleType("hydra_suite.classkit.gui.main_window")
    fake_main_window_module.MainWindow = FakeMainWindow
    monkeypatch.setitem(
        sys.modules,
        "hydra_suite.classkit.gui.main_window",
        fake_main_window_module,
    )
    monkeypatch.setattr(
        "hydra_suite.paths.get_brand_qicon",
        lambda name: calls.__setitem__("brand_icon_name", name) or fake_icon,
    )
    monkeypatch.setattr(
        app_module.sys, "exit", lambda code: calls.__setitem__("exit_code", code)
    )

    app_module.main()

    assert calls["application_name"] == "ClassKit"
    assert calls["display_name"] == "ClassKit"
    assert calls["organization"] == "NeuroRishika"
    assert calls["desktop_file_name"] == "classkit"
    assert calls["brand_icon_name"] == "classkit.svg"
    assert calls["window_icon"] is fake_icon
    assert calls["main_window_icon"] is fake_icon
    assert calls["window_icon_requests"] == 1
    assert calls["resize"] == (1600, 1000)
    assert calls["show_maximized"] is True
    assert calls["exec_called"] is True
    assert calls["exit_code"] == 0
