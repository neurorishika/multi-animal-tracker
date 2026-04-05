"""Smoke tests: each panel instantiates and exposes expected key widgets."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from hydra_suite.trackerkit.config.schemas import TrackerConfig  # noqa: E402

_PANEL_MAP = {
    "DatasetPanel": "hydra_suite.trackerkit.gui.panels.dataset_panel",
    "DetectionPanel": "hydra_suite.trackerkit.gui.panels.detection_panel",
    "IdentityPanel": "hydra_suite.trackerkit.gui.panels.identity_panel",
    "PostProcessPanel": "hydra_suite.trackerkit.gui.panels.postprocess_panel",
    "SetupPanel": "hydra_suite.trackerkit.gui.panels.setup_panel",
    "TrackingPanel": "hydra_suite.trackerkit.gui.panels.tracking_panel",
}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def main_window(qapp):
    from hydra_suite.trackerkit.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


@pytest.mark.parametrize("class_name,module_path", list(_PANEL_MAP.items()))
def test_panel_instantiates(qapp, class_name, module_path):
    """Each panel must instantiate without raising an exception."""
    import importlib

    mock_mw = MagicMock()
    config = TrackerConfig()
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    panel = cls(main_window=mock_mw, config=config)
    assert isinstance(panel, QWidget)


@pytest.mark.parametrize("class_name,module_path", list(_PANEL_MAP.items()))
def test_panel_has_config_changed_signal(class_name, module_path):
    """Each panel class must declare a config_changed signal."""
    import importlib

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    assert hasattr(cls, "config_changed")
