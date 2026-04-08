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
    QComboBox,
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

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from ..models import OBBSource
from ..utils import (
    ensure_detectkit_source_structure,
    list_images_in_source,
    source_class_id_map,
)

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class DatasetPanel(QWidget):
    """Left panel: source list, image browser, X-AnyLabeling launch."""

    def __init__(self, parent=None) -> None:
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

        # --- X-AnyLabeling section ---
        layout.addWidget(QLabel("X-AnyLabeling"))
        env_row = QHBoxLayout()
        self.combo_xal_env = QComboBox()
        self.combo_xal_env.setToolTip("Conda environment with X-AnyLabeling installed.")
        self.btn_refresh_envs = QPushButton("⟳")
        self.btn_refresh_envs.setFixedWidth(30)
        self.btn_refresh_envs.setToolTip("Rescan conda environments")
        self.btn_refresh_envs.clicked.connect(self._refresh_xal_envs)
        env_row.addWidget(self.combo_xal_env, 1)
        env_row.addWidget(self.btn_refresh_envs)
        layout.addLayout(env_row)

        xal_btn_row = QHBoxLayout()
        self.btn_xanylabeling = QPushButton("Open in X-AnyLabeling")
        self.btn_xanylabeling.clicked.connect(self._open_xanylabeling)
        self.btn_refresh = QPushButton("Refresh Labels")
        self.btn_refresh.clicked.connect(self._refresh_labels)
        xal_btn_row.addWidget(self.btn_xanylabeling)
        xal_btn_row.addWidget(self.btn_refresh)
        layout.addLayout(xal_btn_row)

        self._refresh_xal_envs()

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

    def _selected_xal_env(self) -> str | None:
        """Return the selected X-AnyLabeling conda env name, or None."""
        env = self.combo_xal_env.currentText().strip()
        if (
            not env
            or env.startswith("No ")
            or env.startswith("Conda ")
            or env.startswith("Error")
        ):
            return None
        return env

    def _refresh_xal_envs(self) -> None:
        """Scan for conda environments starting with 'x-anylabeling-'."""
        self.combo_xal_env.clear()
        try:
            result = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                envs = []
                for line in result.stdout.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if parts and parts[0].startswith("x-anylabeling"):
                        envs.append(parts[0])
                if envs:
                    self.combo_xal_env.addItems(envs)
                    self.btn_xanylabeling.setEnabled(True)
                    logger.info("Found %d X-AnyLabeling conda env(s)", len(envs))
                else:
                    self.combo_xal_env.addItem("No X-AnyLabeling envs found")
                    self.btn_xanylabeling.setEnabled(False)
                    logger.warning(
                        "No conda envs starting with 'x-anylabeling-' found. "
                        "Create one: conda create -n x-anylabeling-cpu python=3.10 "
                        "&& conda activate x-anylabeling-cpu && pip install x-anylabeling"
                    )
            else:
                self.combo_xal_env.addItem("Conda not available")
                self.btn_xanylabeling.setEnabled(False)
        except FileNotFoundError:
            self.combo_xal_env.addItem("Conda not installed")
            self.btn_xanylabeling.setEnabled(False)
        except Exception as exc:
            self.combo_xal_env.addItem("Error detecting envs")
            self.btn_xanylabeling.setEnabled(False)
            logger.warning("Failed to scan conda envs: %s", exc)

    def _validate_source(self, path: str) -> None:
        """Validate a candidate DetectKit source against the current project scheme."""
        source_dir = ensure_detectkit_source_structure(path)
        try:
            from hydra_suite.training.dataset_inspector import (
                inspect_obb_or_detect_dataset,
            )

            inspect_obb_or_detect_dataset(source_dir)
        except Exception:
            self._try_xlabel_convert(path)
            inspect_obb_or_detect_dataset(source_dir)

        if self._project is not None:
            source_class_id_map(source_dir, self._project.class_names)

    def _try_xlabel_convert(self, path: str) -> None:
        """Attempt to convert xlabel JSON labels to YOLO format via conda env."""
        env = self._selected_xal_env()
        try:
            from hydra_suite.integrations.xanylabeling.cli import convert_project

            ok, msg = convert_project(path, path, conda_env=env)
            if ok:
                logger.info("Auto-converted xlabel labels in %s: %s", path, msg)
            else:
                logger.debug("xlabel conversion not applicable for %s: %s", path, msg)
        except Exception:
            logger.debug("xlabel conversion failed for %s", path, exc_info=True)

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
        added_rows: list[int] = []
        skipped: list[str] = []
        for d in dirs:
            p = Path(d)
            try:
                self._validate_source(str(p))
            except Exception as exc:
                skipped.append(f"{p.name}: {exc}")
                continue
            src = OBBSource(path=str(p), name=p.name, validated=True)
            self._add_source_item(src)
            added_rows.append(self.source_list.count() - 1)

        if added_rows:
            self.source_list.setCurrentRow(added_rows[0])

        if skipped:
            QMessageBox.warning(
                self,
                "Skipped Invalid Sources",
                "\n\n".join(skipped),
            )

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
            try:
                self._validate_source(p)
            except Exception as exc:
                skipped = f"{Path(p).name}: {exc}"
                QMessageBox.warning(self, "Skipped Invalid Source", skipped)
                continue
            src = OBBSource(path=p, name=Path(p).name, validated=True)
            self._add_source_item(src)

    def _open_xanylabeling(self) -> None:
        """Launch X-AnyLabeling for the selected source in the selected conda env."""
        source_path = self._selected_source_path()
        if source_path is None:
            QMessageBox.information(self, "No Source", "Select a source first.")
            return

        env = self._selected_xal_env()
        if env is None:
            QMessageBox.warning(
                self,
                "No Environment",
                "Select a valid conda environment with X-AnyLabeling installed.\n\n"
                "Create one with:\n"
                "  conda create -n x-anylabeling-cpu python=3.10\n"
                "  conda activate x-anylabeling-cpu\n"
                "  pip install x-anylabeling",
            )
            return

        source_dir = Path(source_path)
        try:
            self._validate_source(str(source_dir))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Source", str(exc))
            return

        # Build the shell command: activate conda env, convert yolo->xlabel, open GUI
        convert_cmd = (
            "xanylabeling convert --task yolo2xlabel --mode obb "
            "--images ./images --labels ./labels --output ./images "
            "--classes classes.txt"
        )
        open_cmd = "xanylabeling --filename ./images"
        full_cmd = f"{convert_cmd} && {open_cmd}"

        system = platform.system()
        try:
            if system == "Darwin":
                # macOS: open Terminal with conda activation via AppleScript
                script = (
                    'tell application "Terminal"\n'
                    "    activate\n"
                    '    do script "source $(conda info --base)/etc/profile.d/conda.sh '
                    f"&& conda activate {env} "
                    f"&& cd '{source_dir}' "
                    f'&& {full_cmd}"\n'
                    "end tell"
                )
                subprocess.Popen(["osascript", "-e", script])
            elif system == "Windows":
                cmd = (
                    f'start cmd /k "conda activate {env} '
                    f"&& cd /d {source_dir} "
                    f'&& {full_cmd}"'
                )
                subprocess.Popen(cmd, shell=True)  # noqa: S602
            else:
                # Linux: try common terminal emulators
                shell_cmd = (
                    f"source $(conda info --base)/etc/profile.d/conda.sh "
                    f"&& conda activate {env} "
                    f"&& cd '{source_dir}' "
                    f"&& {full_cmd}"
                )
                for term_cmd in [
                    ["gnome-terminal", "--", "bash", "-c", shell_cmd],
                    ["konsole", "-e", "bash", "-c", shell_cmd],
                    ["xterm", "-e", "bash", "-c", shell_cmd],
                ]:
                    try:
                        subprocess.Popen(term_cmd)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    QMessageBox.warning(
                        self,
                        "No Terminal",
                        "Could not find a terminal emulator "
                        "(gnome-terminal, konsole, or xterm).",
                    )
        except Exception as exc:
            QMessageBox.warning(
                self, "Launch Error", f"Failed to open X-AnyLabeling:\n{exc}"
            )

    def _refresh_labels(self) -> None:
        """Convert xlabel JSONs to YOLO labels, then refresh image list.

        Always attempts xlabel→YOLO conversion first (in case the user
        edited annotations in X-AnyLabeling), then re-validates.
        """
        source_path = self._selected_source_path()
        if source_path is None:
            return
        self._try_xlabel_convert(source_path)
        self._validate_source(source_path)
        # Refresh image list
        self._on_source_changed(self.source_list.currentRow())
