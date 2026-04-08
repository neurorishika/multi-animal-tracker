"""SourceManagerDialog — view / add / remove image source folders."""

import subprocess
import sys
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
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

from hydra_suite.classkit.gui.dialogs.source_validation import (
    count_classkit_images,
    inspect_classkit_source_dir,
    resolve_classkit_images_dir,
    standardize_classkit_source_dir,
)
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

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


class SourceManagerDialog(QDialog):
    """Full source manager showing existing sources with add/remove capability."""

    def __init__(self, db_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Source Manager")
        self.setMinimumSize(640, 480)
        self.setStyleSheet(_DARK_STYLE)

        self._db_path = db_path
        self._to_add: List[Path] = []
        self._to_remove: List[str] = []
        self._existing: List[dict] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel(
            "<b>Image Sources</b><br>"
            "Manage the folders that supply images to this project. "
            "Each source folder should contain an images/ subdirectory. "
            "If a folder contains images directly, ClassKit can standardize it "
            "by creating images/ and copying those files first. "
            "Adding a folder will ingest its images; removing one will "
            "delete those images from the database."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.setMinimumHeight(200)
        self._list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()

        btn_add = QPushButton("Add Folder\u2026")
        btn_add.setStyleSheet(
            "QPushButton { background-color:#1a4a1a; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:8px 16px; }"
            "QPushButton:hover { background-color:#235a23; }"
        )
        btn_add.clicked.connect(self._browse_add)
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

        self._summary = QLabel("")
        self._summary.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self._summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_existing_sources()

    def _load_existing_sources(self):
        from hydra_suite.classkit.core.store.db import ClassKitDB

        db = ClassKitDB(self._db_path)
        self._existing = db.get_source_folders()
        self._rebuild_list()

    def _rebuild_list(self):
        self._list.clear()
        total_images = 0

        for src in self._existing:
            folder = src["folder"]
            count = src["count"]
            is_pending_remove = folder in self._to_remove
            item = QListWidgetItem()
            display = f"{Path(folder).name}  \u2014  {count:,} images\n{folder}"
            if is_pending_remove:
                display = f"[REMOVING]  {display}"
                item.setForeground(Qt.GlobalColor.darkRed)
            item.setText(display)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            self._list.addItem(item)
            if not is_pending_remove:
                total_images += count

        for add_path in self._to_add:
            count = self._count_images(add_path)
            item = QListWidgetItem()
            item.setText(
                f"[NEW]  {add_path.name}  \u2014  {count:,} images\n{add_path}"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(add_path))
            item.setForeground(Qt.GlobalColor.darkGreen)
            self._list.addItem(item)
            total_images += count

        n_sources = len(self._existing) - len(self._to_remove) + len(self._to_add)
        self._summary.setText(
            f"{n_sources} source(s)  \u00b7  ~{total_images:,} images total"
        )

    def _browse_add(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", str(Path.home())
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()

        try:
            resolved = self._resolve_selected_source(d)
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid Source Folder",
                str(exc),
            )
            return
        if resolved is None:
            return

        count = count_classkit_images(resolved)

        existing_folders = {s["folder"] for s in self._existing}
        pending_add_folders = {str(p) for p in self._to_add}
        if str(resolved) in existing_folders or str(resolved) in pending_add_folders:
            QMessageBox.warning(
                self, "Already Added", "That folder is already a source."
            )
            return

        if count > CLASSKIT_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images \u2014 that is a lot to label.\n\n"
                "FilterKit can reduce near-duplicates and create a smaller representative subset before labeling."
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
                        [sys.executable, "-m", "hydra_suite.filterkit", str(d)],
                        start_new_session=True,
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self, "Launch Failed", f"Could not launch FilterKit:\n{exc}"
                    )
                return
            if clicked != btn_add:
                return

        self._to_add.append(resolved)
        self._rebuild_list()

    def _resolve_selected_source(self, dataset_root: Path) -> Path | None:
        """Resolve or standardize a selected source folder into dataset_root/images."""
        inspection = inspect_classkit_source_dir(dataset_root)
        if not inspection.needs_standardization:
            return resolve_classkit_images_dir(dataset_root)

        if not self._confirm_standardization(inspection):
            return None

        try:
            return standardize_classkit_source_dir(dataset_root)
        except Exception as exc:
            raise ValueError(f"Failed to standardize source folder:\n{exc}") from exc

    def _confirm_standardization(self, inspection) -> bool:
        """Ask whether a flat image folder should be converted into dataset_root/images."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Standardize Source Folder")
        msg.setText(
            "This folder contains compatible images directly in the selected "
            "directory instead of inside images/."
        )
        msg.setInformativeText(
            f"ClassKit can create images/ and copy {inspection.images_count:,} "
            "image(s) into it before import.\n\n"
            "This may require additional disk space. Continue?"
        )
        btn_standardize = msg.addButton("Standardize and Add", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.exec()
        return msg.clickedButton() == btn_standardize

    def _remove_selected(self):
        for item in self._list.selectedItems():
            folder = item.data(Qt.ItemDataRole.UserRole)
            if not folder:
                continue
            folder_path = Path(folder).resolve()
            for i, p in enumerate(self._to_add):
                if p == folder_path or str(p) == folder:
                    self._to_add.pop(i)
                    break
            else:
                if folder not in self._to_remove:
                    self._to_remove.append(folder)
        self._rebuild_list()

    @property
    def folders_to_add(self) -> List[Path]:
        return list(self._to_add)

    @property
    def folders_to_remove(self) -> List[str]:
        return list(self._to_remove)

    @property
    def has_changes(self) -> bool:
        return bool(self._to_add) or bool(self._to_remove)
