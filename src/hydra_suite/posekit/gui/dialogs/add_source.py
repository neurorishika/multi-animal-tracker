"""Dialog for adding a new dataset source to a multi-source PoseKit project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ..constants import DEFAULT_DATASET_IMAGES_DIR, LARGE_DATASET_SIEVE_THRESHOLD
from ..utils import list_images

_FILTERKIT_TRANSACTION_FILE = ".filterkit_last_transaction.json"


def _get_transaction_mtime(dataset_root: Path) -> "float | None":
    """Return the mtime of FilterKit's transaction file, or None if absent."""
    p = dataset_root / _FILTERKIT_TRANSACTION_FILE
    try:
        return p.stat().st_mtime
    except OSError:
        return None


class AddSourceDialog(QDialog):
    """Let the user pick a dataset folder to add as a new source to the project."""

    def __init__(self, project, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Dataset Source")
        self.setMinimumWidth(520)
        self._project = project
        self._selected_dir: Optional[Path] = None

        layout = QVBoxLayout(self)

        info = QLabel(
            "Select any folder containing images — either directly in the folder "
            f"or inside an <b>{DEFAULT_DATASET_IMAGES_DIR}/</b> subdirectory.\n"
            "Its images will be added to the current project for labeling."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        form = QFormLayout()

        # -- folder picker row --
        folder_row = QHBoxLayout()
        self._le_folder = QLineEdit()
        self._le_folder.setPlaceholderText("Dataset root folder…")
        self._le_folder.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        folder_row.addWidget(self._le_folder, 1)
        folder_row.addWidget(btn_browse)
        form.addRow("Dataset folder", folder_row)

        # -- description --
        self._le_desc = QLineEdit()
        self._le_desc.setPlaceholderText("e.g. Species A – run 2025-01-01")
        form.addRow("Description (optional)", self._le_desc)

        # -- discovered image count --
        self._lbl_count = QLabel("No folder selected.")
        form.addRow("Discovered images", self._lbl_count)

        layout.addLayout(form)

        # -- existing sources info --
        if project.sources:
            src_names = ", ".join(s.description or s.source_id for s in project.sources)
            note = QLabel(
                f"<i>Current sources ({len(project.sources)}): {src_names}</i>"
            )
            note.setWordWrap(True)
            layout.addWidget(note)

        # -- buttons --
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_ok = QPushButton("Add Source")
        self._btn_ok.setEnabled(False)
        btn_cancel = QPushButton("Cancel")
        btn_row.addWidget(self._btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._btn_ok.clicked.connect(self._accept)
        btn_cancel.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _browse(self):
        start = str(Path.home())
        if self._project.sources:
            start = str(self._project.sources[-1].dataset_root)
        elif self._project.images_dir:
            start = str(self._project.images_dir.parent)

        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", start
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()

        # Auto-detect: prefer images/ subdir if it exists with images
        candidate = d / DEFAULT_DATASET_IMAGES_DIR
        if candidate.is_dir() and list_images(candidate):
            resolved = candidate
            location_note = f"(from {DEFAULT_DATASET_IMAGES_DIR}/ subdirectory)"
        else:
            resolved = d
            location_note = "(directly in folder)"

        count = len(list_images(resolved))
        if count == 0:
            QMessageBox.warning(
                self,
                "No Images Found",
                f"No images were found in:\n{resolved}\n\n"
                "Please select a folder that contains image files.",
            )
            return

        # Check for duplicate
        for src in self._project.sources:
            if src.images_dir.resolve() == resolved.resolve():
                QMessageBox.warning(
                    self,
                    "Already Added",
                    f"This dataset is already registered as source '{src.source_id}'.",
                )
                return

        # Warn when the folder is very large and offer to open FilterKit instead.
        if count > LARGE_DATASET_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images — that is a lot to label by hand.\n\n"
                "FilterKit can reduce near-duplicates and create a smaller representative "
                "subset before labeling."
            )
            msg.setInformativeText(
                "Open this folder in FilterKit now, or continue adding it as-is?"
            )
            btn_sieve = msg.addButton("Open in FilterKit", QMessageBox.AcceptRole)
            btn_add = msg.addButton("Add Anyway", QMessageBox.DestructiveRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()

            clicked = msg.clickedButton()
            if clicked == btn_sieve:
                pre_mtime = _get_transaction_mtime(d)
                proc = QProcess(self)
                self._filterkit_proc = proc
                proc.finished.connect(
                    lambda _ec, _es, _d=d, _r=resolved, _m=pre_mtime: (
                        self._on_filterkit_closed(_d, _r, _m)
                        if self.isVisible()
                        else None
                    )
                )
                proc.errorOccurred.connect(
                    lambda _err: QMessageBox.warning(
                        self, "Launch Failed", "Could not launch FilterKit."
                    )
                )
                proc.start(
                    sys.executable,
                    ["-m", "hydra_suite.filterkit.gui", str(d)],
                )
                return
            if clicked != btn_add:
                return

        self._selected_dir = d
        self._le_folder.setText(str(d))
        if not self._le_desc.text().strip():
            self._le_desc.setText(d.name)

        self._lbl_count.setText(f"{count:,} image(s) found {location_note}")
        self._btn_ok.setEnabled(True)

    def _on_filterkit_closed(self, d: Path, resolved: Path, pre_mtime) -> None:
        """Called when FilterKit exits. Auto-populate if filtering was applied."""
        post_mtime = _get_transaction_mtime(d)
        if post_mtime == pre_mtime:
            reply = QMessageBox.question(
                self,
                "FilterKit Closed",
                "FilterKit was closed without applying a filter.\n\n"
                "Add the original folder as-is?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        count = len(list_images(resolved))
        location_note = (
            f"(from {DEFAULT_DATASET_IMAGES_DIR}/ subdirectory)"
            if resolved != d
            else "(directly in folder)"
        )
        self._selected_dir = d
        self._le_folder.setText(str(d))
        if not self._le_desc.text().strip():
            self._le_desc.setText(d.name)
        self._lbl_count.setText(f"{count:,} image(s) found {location_note}")
        self._btn_ok.setEnabled(True)

    def _accept(self):
        if self._selected_dir is None:
            return
        self.accept()

    # ------------------------------------------------------------------
    @property
    def selected_dir(self) -> Optional[Path]:
        """The dataset root folder selected by the user."""
        return self._selected_dir

    @property
    def description(self) -> str:
        """Return the human-readable description text entered by the user for this source."""
        return self._le_desc.text().strip()
