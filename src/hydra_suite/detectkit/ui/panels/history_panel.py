"""History panel -- browse training run records."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)

_STATUS_COLORS = {
    "completed": "#228B22",
    "failed": "#CC0000",
    "canceled": "#CC9900",
}

_COLUMNS = ["Run ID", "Role", "Status", "Started", "Base Model", "Epochs"]


class HistoryPanel(QWidget):
    """Browse training run history from the registry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proj = None
        self._main_window = None
        self._runs: list[dict] = []
        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top: Refresh
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._load_registry)
        layout.addWidget(self.btn_refresh)

        # Middle: Table
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self.table)

        # Bottom: Detail view
        layout.addWidget(QLabel("Run details:"))
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(200)
        layout.addWidget(self.detail_view)

    # ---- Public API ----

    def set_project(self, proj: DetectKitProject, main_window) -> None:
        self._proj = proj
        self._main_window = main_window
        self._load_registry()

    def collect_state(self, proj: DetectKitProject) -> None:
        pass  # nothing to persist

    # ---- Internal ----

    def _load_registry(self) -> None:
        try:
            from hydra_suite.training.registry import get_registry_path
        except ImportError:
            logger.warning("Training registry module not available")
            return

        registry_path = str(get_registry_path())

        try:
            from hydra_suite.tracker.gui.dialogs.run_history_dialog import (
                load_run_history,
            )
        except ImportError:
            logger.warning("Run history loader not available")
            return

        self._runs = list(reversed(load_run_history(registry_path)))
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._runs))

        for row_idx, run in enumerate(self._runs):
            spec = run.get("spec", {})
            hyperparams = spec.get("hyperparams", {})

            values = [
                run.get("run_id", ""),
                run.get("role", ""),
                run.get("status", ""),
                run.get("started_at", ""),
                spec.get("base_model", ""),
                str(hyperparams.get("epochs", "")),
            ]
            for col_idx, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # Color-code the status column
                if col_idx == 2:
                    color = _STATUS_COLORS.get(val)
                    if color:
                        item.setForeground(QColor(color))
                self.table.setItem(row_idx, col_idx, item)

        self.detail_view.clear()

    def _on_row_changed(self, current_row, _col, _prev_row, _prev_col):
        if 0 <= current_row < len(self._runs):
            self.detail_view.setPlainText(json.dumps(self._runs[current_row], indent=2))
        else:
            self.detail_view.clear()
