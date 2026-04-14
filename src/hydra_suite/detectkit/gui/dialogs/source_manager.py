"""SourceManagerDialog — add/remove/scan dataset source directories."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.widgets.dialogs import BaseDialog

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class SourceManagerDialog(BaseDialog):
    """Manage dataset source directories for a DetectKit project."""

    def __init__(self, project: "DetectKitProject", parent=None) -> None:
        super().__init__(
            "Manage Sources",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Close,
        )
        self._project = project
        self._build_content()
        self._refresh_list()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_content(self) -> None:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        v.addWidget(QLabel("Dataset source directories:"))

        self._source_list = QListWidget()
        self._source_list.setMinimumHeight(200)
        v.addWidget(self._source_list)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Source…")
        self.btn_add.clicked.connect(self._add_source)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        v.addLayout(btn_row)

        self.add_content(container)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _refresh_list(self) -> None:
        self._source_list.clear()
        for src in self._project.sources:
            display = src.name if src.name else src.path
            self._source_list.addItem(display)

    def _add_source(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Select Source Directory", ""
        )
        if not directory:
            return
        from ..models import OBBSource

        # Avoid duplicates
        existing_paths = {s.path for s in self._project.sources}
        if directory in existing_paths:
            QMessageBox.information(self, "Add Source", "Source already added.")
            return

        import os

        name = os.path.basename(directory)
        self._project.sources.append(OBBSource(path=directory, name=name))
        self._refresh_list()

    def _remove_selected(self) -> None:
        row = self._source_list.currentRow()
        if row < 0 or row >= len(self._project.sources):
            return
        self._project.sources.pop(row)
        self._refresh_list()
