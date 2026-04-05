"""trackerkit GUI panels — each panel owns one functional area of the UI."""

from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel
from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel
from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel
from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel
from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel
from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

__all__ = [
    "DetectionPanel",
    "DatasetPanel",
    "IdentityPanel",
    "PostProcessPanel",
    "SetupPanel",
    "TrackingPanel",
]
