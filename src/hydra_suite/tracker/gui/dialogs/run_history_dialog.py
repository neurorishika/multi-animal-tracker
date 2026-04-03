"""Training run history viewer dialog."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)


def load_run_history(registry_path: str) -> list[dict]:
    """Load run records from a registry JSON file."""
    path = Path(registry_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        return []
    return runs


_STATUS_COLORS = {
    "completed": "#228B22",
    "failed": "#CC0000",
    "canceled": "#CC9900",
}

_COLUMNS = ["Run ID", "Role", "Status", "Started", "Base Model", "Epochs"]


class RunHistoryDialog(QDialog):
    """Browse training runs recorded in the registry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Run History")
        self.resize(900, 520)

        from hydra_suite.training.registry import get_registry_path

        registry_path = str(get_registry_path())
        self._runs = load_run_history(registry_path)
        # Newest first
        self._runs = list(reversed(self._runs))

        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Table
        self.table = QTableWidget(len(self._runs), len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

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
                # Color the status column
                if col_idx == 2:
                    color = _STATUS_COLORS.get(val)
                    if color:
                        from PySide6.QtGui import QColor

                        item.setForeground(QColor(color))
                self.table.setItem(row_idx, col_idx, item)

        layout.addWidget(self.table)

        # Detail view
        layout.addWidget(QLabel("Run details:"))
        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(160)
        layout.addWidget(self.detail_view)

        # Close button
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        # Connections
        self.table.currentCellChanged.connect(self._on_row_changed)

    def _on_row_changed(self, current_row, _col, _prev_row, _prev_col):
        if 0 <= current_row < len(self._runs):
            self.detail_view.setPlainText(json.dumps(self._runs[current_row], indent=2))
        else:
            self.detail_view.clear()
