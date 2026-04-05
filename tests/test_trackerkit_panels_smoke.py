"""Smoke tests: each panel instantiates and exposes expected key widgets."""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from hydra_suite.trackerkit.config.schemas import TrackerConfig  # noqa: E402

# All panels — used for signal-presence checks.
_PANEL_MAP = {
    "DatasetPanel": "hydra_suite.trackerkit.gui.panels.dataset_panel",
    "DetectionPanel": "hydra_suite.trackerkit.gui.panels.detection_panel",
    "IdentityPanel": "hydra_suite.trackerkit.gui.panels.identity_panel",
    "PostProcessPanel": "hydra_suite.trackerkit.gui.panels.postprocess_panel",
    "SetupPanel": "hydra_suite.trackerkit.gui.panels.setup_panel",
    "TrackingPanel": "hydra_suite.trackerkit.gui.panels.tracking_panel",
}

# Panels whose _build_ui is still a stub (safe to instantiate with MagicMock).
# Remove a panel from this dict once its _build_ui is populated.
_STUB_PANEL_MAP = {
    k: v
    for k, v in _PANEL_MAP.items()
    if k not in {"DatasetPanel", "SetupPanel", "TrackingPanel"}
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


@pytest.mark.parametrize("class_name,module_path", list(_STUB_PANEL_MAP.items()))
def test_panel_instantiates(qapp, class_name, module_path):
    """Stub panels must instantiate without raising an exception (mock main_window is safe)."""
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


def test_dataset_panel_wired_in_main_window(main_window):
    """DatasetPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

    assert hasattr(main_window, "_dataset_panel")
    assert isinstance(main_window._dataset_panel, DatasetPanel)
    assert hasattr(main_window._dataset_panel, "combo_xanylabeling_env")
    assert hasattr(main_window._dataset_panel, "chk_enable_dataset_gen")


def test_setup_panel_wired_in_main_window(main_window):
    """SetupPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel

    assert hasattr(main_window, "_setup_panel")
    assert isinstance(main_window._setup_panel, SetupPanel)
    assert hasattr(main_window._setup_panel, "combo_presets")
    assert hasattr(main_window._setup_panel, "btn_file")


def test_tracking_panel_wired_in_main_window(main_window):
    """TrackingPanel is accessible on MainWindow and exposes key widgets."""
    from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

    assert hasattr(main_window, "_tracking_panel")
    assert isinstance(main_window._tracking_panel, TrackingPanel)
    assert hasattr(main_window._tracking_panel, "g_density")
    assert hasattr(main_window._tracking_panel, "chk_enable_confidence_density_map")
