"""TrackingPanel — core tracking parameters, Kalman filter, and assignment config."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig


class TrackingPanel(QWidget):
    """Kalman filter parameters, identity assignment, and backward pass controls.

    Signals
    -------
    config_changed(TrackerConfig)
        Emitted when the user edits a tracking parameter.
    """

    config_changed: Signal = Signal(object)

    def __init__(self, config: TrackerConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout. Filled in during extraction."""

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
