"""Dataset panel -- source management and image browser (left panel)."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..models import OBBSource
from ..utils import list_images_in_source

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class DatasetPanel(QWidget):
    """Left panel: source list, image browser, X-AnyLabeling launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = None
        self._project: DetectKitProject | None = None

        layout = QVBoxLayout(self)

        # --- Source list ---
        layout.addWidget(QLabel("Sources"))
        self.source_list = QListWidget()
        self.source_list.currentRowChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_list)

        # --- Source action buttons ---
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add...")
        self.btn_remove = QPushButton("Remove")
        self.btn_save_list = QPushButton("Save List...")
        self.btn_load_list = QPushButton("Load List...")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_save_list)
        btn_row.addWidget(self.btn_load_list)
        layout.addLayout(btn_row)

        self.btn_add.clicked.connect(self._add_sources)
        self.btn_remove.clicked.connect(self._remove_source)
        self.btn_save_list.clicked.connect(self._save_list)
        self.btn_load_list.clicked.connect(self._load_list)

        # --- Image list ---
        layout.addWidget(QLabel("Images"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._on_image_changed)
        layout.addWidget(self.image_list)

        # --- X-AnyLabeling button ---
        self.btn_xanylabeling = QPushButton("Open in X-AnyLabeling")
        self.btn_xanylabeling.clicked.connect(self._open_xanylabeling)
        layout.addWidget(self.btn_xanylabeling)

        # --- Refresh Labels button ---
        self.btn_refresh = QPushButton("Refresh Labels")
        self.btn_refresh.clicked.connect(self._refresh_labels)
        layout.addWidget(self.btn_refresh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_project(self, proj: DetectKitProject, main_window) -> None:
        """Populate from project state."""
        self._project = proj
        self._main_window = main_window

        self.source_list.blockSignals(True)
        self.source_list.clear()
        for src in proj.sources:
            self._add_source_item(src)
        self.source_list.blockSignals(False)

        # Restore last selection
        if 0 <= proj.last_source_index < self.source_list.count():
            self.source_list.setCurrentRow(proj.last_source_index)
        elif self.source_list.count() > 0:
            self.source_list.setCurrentRow(0)

    def collect_state(self, proj: DetectKitProject) -> None:
        """Write panel state back into the project."""
        proj.sources = []
        for i in range(self.source_list.count()):
            item = self.source_list.item(i)
            path = item.data(Qt.UserRole)
            proj.sources.append(OBBSource(path=path, name=item.text()))
        proj.last_source_index = max(self.source_list.currentRow(), 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_source_item(self, src: OBBSource) -> None:
        """Add an OBBSource as a list widget item."""
        item = QListWidgetItem(src.name or Path(src.path).name)
        item.setData(Qt.UserRole, src.path)
        self.source_list.addItem(item)

    def _selected_source_path(self) -> str | None:
        """Return the path of the currently selected source, or None."""
        item = self.source_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _get_multiple_dirs(self, title: str) -> list[str]:
        """Open a non-native file dialog that allows multi-directory selection."""
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QFileDialog.Accepted:
            return []
        return dlg.selectedFiles()

    def _validate_source(self, path: str) -> None:
        """Run validation and auto-convert xlabel JSON if needed."""
        try:
            from multi_tracker.training.dataset_inspector import (
                inspect_obb_or_detect_dataset,
            )

            inspect_obb_or_detect_dataset(path)
        except Exception:
            # Validation failed -- try xlabel->YOLO conversion
            self._try_xlabel_convert(path)

    def _try_xlabel_convert(self, path: str) -> None:
        """Attempt to convert xlabel JSON labels to YOLO format."""
        try:
            from multi_tracker.integrations.xanylabeling.cli import convert_project

            ok, msg = convert_project(path, path)
            if ok:
                logger.info("Auto-converted xlabel labels in %s: %s", path, msg)
            else:
                logger.debug("xlabel conversion not applicable for %s: %s", path, msg)
        except Exception:
            logger.debug("xlabel conversion failed for %s", path, exc_info=True)

    def _ensure_classes_txt(self, source_dir: Path) -> None:
        """Create classes.txt in source dir if missing, using project class_name."""
        classes_file = source_dir / "classes.txt"
        if not classes_file.exists() and self._project is not None:
            classes_file.write_text(self._project.class_name + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_source_changed(self, row: int) -> None:
        """Populate image list when source selection changes."""
        self.image_list.clear()
        if row < 0:
            return
        item = self.source_list.item(row)
        if item is None:
            return
        source_path = item.data(Qt.UserRole)
        images = list_images_in_source(source_path)
        self.image_list.blockSignals(True)
        for img in images:
            img_item = QListWidgetItem(img.name)
            img_item.setData(Qt.UserRole, str(img))
            self.image_list.addItem(img_item)
        self.image_list.blockSignals(False)
        if self.image_list.count() > 0:
            self.image_list.setCurrentRow(0)

    def _on_image_changed(self, row: int) -> None:
        """Show selected image in canvas."""
        if row < 0 or self._main_window is None:
            return
        img_item = self.image_list.item(row)
        if img_item is None:
            return
        source_path = self._selected_source_path()
        if source_path is None:
            return
        image_path = img_item.data(Qt.UserRole)
        self._main_window.show_image(source_path, str(image_path))

    def _add_sources(self) -> None:
        """Add one or more source directories."""
        dirs = self._get_multiple_dirs("Select OBB Source Datasets")
        if not dirs:
            return
        for d in dirs:
            p = Path(d)
            src = OBBSource(path=str(p), name=p.name)
            self._validate_source(str(p))
            self._add_source_item(src)
        # Select the first newly added source
        self.source_list.setCurrentRow(self.source_list.count() - len(dirs))

    def _remove_source(self) -> None:
        """Remove the selected source from the list."""
        row = self.source_list.currentRow()
        if row >= 0:
            self.source_list.takeItem(row)

    def _save_list(self) -> None:
        """Save source paths to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Source List", "", "JSON Files (*.json)"
        )
        if not path:
            return
        data = []
        for i in range(self.source_list.count()):
            item = self.source_list.item(i)
            data.append({"source_type": "obb", "path": item.data(Qt.UserRole)})
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_list(self) -> None:
        """Load sources from a JSON file and append to current list."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Source List", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("Expected a JSON array")
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", str(exc))
            return
        for entry in data:
            p = str(entry.get("path", ""))
            if not p:
                continue
            src = OBBSource(path=p, name=Path(p).name)
            self._add_source_item(src)

    def _open_xanylabeling(self) -> None:
        """Launch X-AnyLabeling for the selected source."""
        source_path = self._selected_source_path()
        if source_path is None:
            QMessageBox.information(self, "No Source", "Select a source first.")
            return

        source_dir = Path(source_path)
        self._ensure_classes_txt(source_dir)
        self._try_xlabel_convert(source_path)

        images_dir = source_dir / "images"
        if not images_dir.is_dir():
            images_dir = source_dir

        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", "-a", "Terminal", str(source_dir)])
            else:
                subprocess.Popen(["xanylabeling", "--filename", str(images_dir)])
        except Exception as exc:
            QMessageBox.warning(
                self, "Launch Error", f"Failed to open X-AnyLabeling:\n{exc}"
            )

    def _refresh_labels(self) -> None:
        """Re-validate selected source and refresh image list."""
        source_path = self._selected_source_path()
        if source_path is None:
            return
        self._validate_source(source_path)
        # Refresh image list
        self._on_source_changed(self.source_list.currentRow())
