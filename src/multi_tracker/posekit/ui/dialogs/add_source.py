"""Dialog for adding a new dataset source to a multi-source PoseKit project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

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

from ..constants import DEFAULT_DATASET_IMAGES_DIR
from ..utils import list_images


class AddSourceDialog(QDialog):
    """Let the user pick a dataset folder to add as a new source to the project."""

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Dataset Source")
        self.setMinimumWidth(520)
        self._project = project
        self._selected_dir: Optional[Path] = None
        self._resolved_images_dir: Optional[Path] = None

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

        self._selected_dir = d
        self._resolved_images_dir = resolved
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
    def selected_images_dir(self) -> Optional[Path]:
        """The resolved images folder (may be inside selected_dir/images/ or selected_dir itself)."""
        return self._resolved_images_dir

    @property
    def description(self) -> str:
        return self._le_desc.text().strip()
