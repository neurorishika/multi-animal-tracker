"""AddSourceDialog — pick image source folders for a ClassKit project."""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

CLASSKIT_IMAGES_SUBDIR = "images"
CLASSKIT_SIEVE_THRESHOLD = 5000

_DARK_STYLE = """
    QDialog { background-color: #1e1e1e; }
    QGroupBox {
        border: 1px solid #3e3e42; border-radius: 6px;
        margin-top: 12px; padding-top: 12px; color: #cccccc;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QLabel { color: #cccccc; }
    QLineEdit, QTextEdit, QPlainTextEdit, QListWidget {
        background-color: #252526; color: #e0e0e0;
        border: 1px solid #3e3e42; border-radius: 4px; padding: 6px;
    }
    QLineEdit:focus, QTextEdit:focus { border: 1px solid #007acc; }
    QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #252526; color: #e0e0e0;
        border: 1px solid #3e3e42; border-radius: 4px; padding: 6px;
    }
    QComboBox:focus, QSpinBox:focus { border: 1px solid #007acc; }
    QCheckBox { color: #cccccc; }
    QPushButton {
        background-color: #0e639c; color: #ffffff;
        border: none; border-radius: 4px;
        padding: 8px 16px; font-weight: 500;
    }
    QPushButton:hover { background-color: #1177bb; }
    QPushButton:pressed { background-color: #0d5a8f; }
    QPushButton:disabled { background-color: #3e3e42; color: #888888; }
"""


class AddSourceDialog(QDialog):
    """Pick one or more image source folders for a ClassKit project.

    For each folder the user picks, the ``images/`` subdirectory is preferred
    (if it exists and contains images), otherwise the root folder itself is
    used — matching the PoseKit convention.

    If a folder contains more images than ``CLASSKIT_SIEVE_THRESHOLD``, the
    dialog offers to open FilterKit before continuing.
    """

    def __init__(self, existing_sources: Optional[List[Path]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Image Sources")
        self.setMinimumWidth(580)
        self.setStyleSheet(_DARK_STYLE)

        # Resolved image dirs already added (for duplicate checks).
        self._existing: List[Path] = [
            p.expanduser().resolve() for p in (existing_sources or [])
        ]
        # (dataset_root, resolved_images_dir, description)
        self._sources: List[Tuple[Path, Path, str]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel(
            "Add one or more folders containing images.  "
            f"An <b>{CLASSKIT_IMAGES_SUBDIR}/</b> subdirectory is preferred when present; "
            "otherwise the folder root is used."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QListWidget()
        self._list.setMinimumHeight(120)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Folder\u2026")
        btn_add.clicked.connect(self._browse)
        btn_row.addWidget(btn_add)
        btn_row.addStretch(1)
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.setStyleSheet(
            "QPushButton { background-color:#6b1c1c; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:8px 16px; }"
            "QPushButton:hover { background-color:#8b2424; }"
        )
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_remove.setEnabled(False)
        btn_row.addWidget(self._btn_remove)
        layout.addLayout(btn_row)

        self._list.itemSelectionChanged.connect(
            lambda: self._btn_remove.setEnabled(bool(self._list.selectedItems()))
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Done")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", str(Path.home())
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()

        # Prefer images/ subdirectory (PoseKit convention)
        candidate = d / CLASSKIT_IMAGES_SUBDIR
        if candidate.is_dir() and self._has_images(candidate):
            resolved = candidate
            location_note = f"(from {CLASSKIT_IMAGES_SUBDIR}/ subdirectory)"
        else:
            resolved = d
            location_note = "(folder root)"

        count = self._count_images(resolved)
        if count == 0:
            QMessageBox.warning(
                self,
                "No Images Found",
                f"No images were found in:\n{resolved}\n\n"
                "Please select a folder that contains .jpg / .jpeg / .png files.",
            )
            return

        # Duplicate check
        if resolved in self._existing or any(
            r == resolved for _, r, _ in self._sources
        ):
            QMessageBox.warning(
                self, "Already Added", "That folder has already been added."
            )
            return

        # FilterKit check
        if count > CLASSKIT_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images \u2014 that is a lot to label.\n\n"
                "FilterKit can reduce near-duplicates and create a smaller "
                "representative subset before labeling."
            )
            msg.setInformativeText(
                "Open this folder in FilterKit now, or add it as-is?"
            )
            btn_sieve = msg.addButton("Open in FilterKit", QMessageBox.AcceptRole)
            btn_add = msg.addButton("Add Anyway", QMessageBox.DestructiveRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_sieve:
                try:
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "hydra_suite.filterkit.gui",
                            str(d),
                        ],
                        start_new_session=True,
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self, "Launch Failed", f"Could not launch FilterKit:\n{exc}"
                    )
                return
            if clicked != btn_add:
                return

        self._sources.append((d, resolved, d.name))
        item = QListWidgetItem(
            f"{d.name}  \u2014  {count:,} images  {location_note}\n{d}"
        )
        self._list.addItem(item)

    def _remove_selected(self):
        for item in self._list.selectedItems():
            row = self._list.row(item)
            self._list.takeItem(row)
            if row < len(self._sources):
                self._sources.pop(row)

    @staticmethod
    def _has_images(folder: Path) -> bool:
        exts = {".jpg", ".jpeg", ".png"}
        return any(p.suffix.lower() in exts for p in folder.iterdir() if p.is_file())

    @staticmethod
    def _count_images(folder: Path) -> int:
        exts = {".jpg", ".jpeg", ".png"}
        return sum(
            1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    @property
    def sources(self) -> List[Tuple[Path, Path, str]]:
        """List of (dataset_root, resolved_images_dir, description)."""
        return list(self._sources)
