"""HistoryDialog — browse training run history and select a model for inference."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


def _load_runs(project: "DetectKitProject") -> list[dict]:
    """Load training run history for *project*. Monkeypatchable in tests."""
    try:
        from hydra_suite.trackerkit.gui.dialogs.run_history_dialog import (
            load_run_history,
        )
        from hydra_suite.training.registry import get_registry_path

        registry_path = str(get_registry_path())
        return list(reversed(load_run_history(registry_path)))
    except Exception as exc:
        logger.warning("Could not load run history: %s", exc)
        return []


class HistoryDialog(BaseDialog):
    """Browse training run history; optionally load a model for inference."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Training History",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._runs: list[dict] = []
        self.resize(760, 480)
        self._build_content()
        self._refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: run list + action buttons
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        self._run_list = QListWidget()
        self._run_list.currentRowChanged.connect(self._on_row_changed)
        lv.addWidget(self._run_list)

        btn_row = QHBoxLayout()
        self._btn_load = QPushButton("Use for Inference")
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._load_for_inference)
        self._btn_delete = QPushButton("Delete Run")
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._delete_run)
        btn_row.addWidget(self._btn_load)
        btn_row.addWidget(self._btn_delete)
        lv.addLayout(btn_row)

        splitter.addWidget(left)

        # Right: detail view
        self._detail_view = QTextEdit()
        self._detail_view.setReadOnly(True)
        self._detail_view.setPlaceholderText("Select a run to see details.")
        splitter.addWidget(self._detail_view)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.add_content(splitter)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._runs = _load_runs(self._project)
        self._run_list.clear()
        for run in self._runs:
            label = f"{run.get('run_id', '?')}  [{run.get('role', '')}]  {run.get('status', '')}"
            self._run_list.addItem(label)
        self._detail_view.clear()
        self._btn_load.setEnabled(False)
        self._btn_delete.setEnabled(False)

    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._runs):
            self._detail_view.setPlainText(json.dumps(self._runs[row], indent=2))
            has_model = bool(self._runs[row].get("published_model_path"))
            self._btn_load.setEnabled(has_model)
            self._btn_delete.setEnabled(True)
        else:
            self._detail_view.clear()
            self._btn_load.setEnabled(False)
            self._btn_delete.setEnabled(False)

    def _load_for_inference(self) -> None:
        row = self._run_list.currentRow()
        if 0 <= row < len(self._runs):
            self._project.active_model_path = self._runs[row].get(
                "published_model_path", ""
            )
            self.accept()

    def _delete_run(self) -> None:
        row = self._run_list.currentRow()
        if row < 0 or row >= len(self._runs):
            return
        run_id = self._runs[row].get("run_id", "?")
        ans = QMessageBox.question(
            self,
            "Delete Run",
            f"Delete run '{run_id}'? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        try:
            from hydra_suite.training.registry import delete_run

            delete_run(run_id)
        except Exception as exc:
            logger.warning("Could not delete run %s: %s", run_id, exc)
        self._refresh()
