"""SetupPanel — preset selection, video files, display, and ROI configuration."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from hydra_suite.trackerkit.config.schemas import TrackerConfig


class SetupPanel(QWidget):
    """Preset picker, video/batch file selection, ROI, and display options.

    Signals
    -------
    config_changed(TrackerConfig)
        Emitted when the user edits a setup parameter.
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
