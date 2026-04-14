"""DetectKit main window — thin coordinator with VS Code-style toolbar."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QWidget,
)

from hydra_suite.detectkit.config.schemas import DetectKitConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

from .canvas import OBBCanvas
from .models import DetectKitProject
from .panels.dataset_panel import DatasetPanel
from .panels.tools_panel import ToolsPanel
from .project import (
    create_project,
    default_project_parent_dir,
    open_project,
    project_exists,
    save_project,
)
from .utils import find_label_for_image, parse_obb_label, source_class_id_map

logger = logging.getLogger(__name__)

_DATASET_PANEL_MIN_WIDTH = 220
_DATASET_PANEL_MAX_WIDTH = 360
_CANVAS_MIN_WIDTH = 480
_WORKSPACE_MIN_HEIGHT = 700
_WORKSPACE_MIN_WIDTH = _DATASET_PANEL_MIN_WIDTH + _CANVAS_MIN_WIDTH + 280 + 24

_DARK_STYLESHEET = """
QMainWindow { background-color: #1e1e1e; }
QWidget { background-color: #1e1e1e; color: #ddd; }
QMenuBar { background-color: #252526; color: #ddd; }
QMenuBar::item:selected { background-color: #094771; }
QMenu { background-color: #252526; color: #ddd; }
QMenu::item:selected { background-color: #094771; }
QPushButton { background-color: #0e639c; color: white; border: none;
              padding: 6px 14px; border-radius: 4px; }
QPushButton:hover { background-color: #1177bb; }
QPushButton:disabled { background-color: #444; color: #888; }
QListWidget { background-color: #252526; border: 1px solid #333; color: #ddd; }
QListWidget::item:selected { background-color: #094771; }
QSplitter::handle { background-color: #333; width: 2px; }
QStatusBar { background-color: #007acc; color: white; }
QScrollArea { border: none; }
QGroupBox { border: 1px solid #444; border-radius: 4px;
            margin-top: 8px; padding-top: 12px; color: #ddd; }
QGroupBox::title { padding: 0 4px; }
QToolBar { background-color: #252526; border-bottom: 1px solid #333; spacing: 4px; }
QToolBar QToolButton { color: #ddd; padding: 4px 8px; }
QToolBar QToolButton:hover { background-color: #094771; border-radius: 3px; }
QComboBox { background-color: #3c3c3c; border: 1px solid #555; color: #ddd;
            padding: 2px 6px; border-radius: 3px; }
"""


class DetectKitMainWindow(QMainWindow):
    """DetectKit main application window — thin coordinator."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("DetectKit")
        self.setStyleSheet(_DARK_STYLESHEET)
        self.setMinimumSize(_WORKSPACE_MIN_WIDTH, _WORKSPACE_MIN_HEIGHT)

        self.config = DetectKitConfig()
        self._project: Optional[DetectKitProject] = None

        # Build workspace panels first (toolbar actions need them)
        self._dataset_panel = DatasetPanel()
        self._canvas = OBBCanvas()
        self._tools_panel = ToolsPanel()

        # Toolbar (hidden until project loaded)
        self._toolbar = self._build_toolbar()
        self.addToolBar(self._toolbar)
        self._toolbar.setVisible(False)

        # Central stacked widget: welcome (0) vs workspace (1)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._build_welcome_page()
        self._build_workspace_page()
        self._build_menu_bar()

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        self._stack.setCurrentIndex(0)
        self.menuBar().hide()

        # Connect ToolsPanel signals
        self._dataset_panel.manage_sources_requested.connect(self._open_source_manager)
        self._tools_panel.overlay_settings_changed.connect(self._on_overlay_changed)
        self._tools_panel.prev_requested.connect(self._dataset_panel.navigate_prev)
        self._tools_panel.next_requested.connect(self._dataset_panel.navigate_next)
        self._tools_panel.train_requested.connect(self._open_training_dialog)
        self._tools_panel.evaluate_requested.connect(self._open_evaluation_dialog)
        self._tools_panel.history_requested.connect(self._open_history_dialog)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)

        act_new = QAction("New", self)
        act_new.triggered.connect(self.new_project)
        tb.addAction(act_new)

        act_open = QAction("Open", self)
        act_open.triggered.connect(self.open_project_dialog)
        tb.addAction(act_open)

        act_save = QAction("Save", self)
        act_save.triggered.connect(self._save_current_project)
        tb.addAction(act_save)

        tb.addSeparator()

        act_sources = QAction("Sources", self)
        act_sources.triggered.connect(self._open_source_manager)
        tb.addAction(act_sources)

        tb.addSeparator()

        act_prev = QAction("Prev", self)
        act_prev.triggered.connect(self._dataset_panel.navigate_prev)
        tb.addAction(act_prev)

        act_next = QAction("Next", self)
        act_next.triggered.connect(self._dataset_panel.navigate_next)
        tb.addAction(act_next)

        tb.addSeparator()

        act_train = QAction("Train", self)
        act_train.triggered.connect(self._open_training_dialog)
        tb.addAction(act_train)

        act_evaluate = QAction("Evaluate", self)
        act_evaluate.triggered.connect(self._open_evaluation_dialog)
        tb.addAction(act_evaluate)

        act_history = QAction("History", self)
        act_history.triggered.connect(self._open_history_dialog)
        tb.addAction(act_history)

        tb.addSeparator()

        act_export = QAction("Export", self)
        act_export.triggered.connect(self._export_stub)
        tb.addAction(act_export)

        return tb

    # ------------------------------------------------------------------
    # Welcome page
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> None:
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("detectkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="detectkit.svg",
            tagline="OBB Detection Model Training & Dataset Curation",
            buttons=[
                ButtonDef(label="New Project", callback=self.new_project),
                ButtonDef(label="Open Project", callback=self.open_project_dialog),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        self._stack.addWidget(self._welcome_page)  # index 0

    def _open_recent_project(self, path: str) -> None:
        project_dir = Path(path)
        if project_dir.exists():
            proj = open_project(project_dir)
            if proj is not None:
                self._load_project(proj)
            else:
                QMessageBox.warning(
                    self, "Open Failed", f"Could not open project at:\n{path}"
                )
                self._remove_from_recents(path)
        else:
            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            self._remove_from_recents(path)

    def _remove_from_recents(self, path: str) -> None:
        if hasattr(self, "_recents_store"):
            self._recents_store.remove(path)
            if hasattr(self, "_welcome_page"):
                self._welcome_page.refresh_recents()

    # ------------------------------------------------------------------
    # Workspace page
    # ------------------------------------------------------------------

    def _build_workspace_page(self) -> None:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter

        self._dataset_panel.setMinimumWidth(_DATASET_PANEL_MIN_WIDTH)
        self._dataset_panel.setMaximumWidth(_DATASET_PANEL_MAX_WIDTH)
        splitter.addWidget(self._dataset_panel)

        self._canvas.setMinimumWidth(_CANVAS_MIN_WIDTH)
        splitter.addWidget(self._canvas)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([_DATASET_PANEL_MIN_WIDTH, _CANVAS_MIN_WIDTH + 200])

        layout.addWidget(splitter)

        self._tools_panel.setFixedWidth(280)
        layout.addWidget(self._tools_panel)

        self._stack.addWidget(page)  # index 1

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        act_new = QAction("New Project...", self)
        act_new.triggered.connect(self.new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project...", self)
        act_open.triggered.connect(self.open_project_dialog)
        file_menu.addAction(act_open)

        self._recent_menu = QMenu("Recent Projects", self)
        file_menu.addMenu(self._recent_menu)
        self._refresh_recent_menu()

        file_menu.addSeparator()

        act_save = QAction("Save Project", self)
        act_save.triggered.connect(self._save_current_project)
        file_menu.addAction(act_save)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        if hasattr(self, "_recents_store"):
            for p in self._recents_store.load():
                action = self._recent_menu.addAction(p)
                action.setData(p)
                action.triggered.connect(self._on_recent_menu_action)

    def _on_recent_menu_action(self) -> None:
        action = self.sender()
        if action is None:
            return
        path_str = action.data()
        if path_str:
            proj = open_project(Path(path_str))
            if proj is not None:
                self._load_project(proj)
            else:
                QMessageBox.warning(
                    self, "Open Failed", f"Could not open project at:\n{path_str}"
                )

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def new_project(self) -> None:
        from .dialogs import NewProjectDialog

        dialog = NewProjectDialog(self)
        result = dialog.exec()
        if result != dialog.DialogCode.Accepted:
            return

        project_info = dialog.get_project_info()
        proj_dir = Path(project_info["path"]).expanduser()

        if project_exists(proj_dir):
            ans = QMessageBox.question(
                self,
                "Project Exists",
                f"A project already exists in:\n{proj_dir}\n\nOpen it instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.Yes:
                proj = open_project(proj_dir)
                if proj is not None:
                    self._load_project(proj)
            return

        proj = create_project(
            proj_dir,
            project_info["class_name"],
            class_names=list(project_info.get("class_names", [])),
        )
        self._load_project(proj)

    def open_project_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Open DetectKit Project", str(default_project_parent_dir())
        )
        if not directory:
            return
        proj = open_project(Path(directory))
        if proj is not None:
            self._load_project(proj)
        else:
            QMessageBox.warning(
                self, "Open Failed", f"No DetectKit project found in:\n{directory}"
            )

    def _load_project(self, proj: DetectKitProject) -> None:
        """Activate proj: wire panels, show toolbar, switch to workspace."""
        self._project = proj

        self._dataset_panel.set_project(proj, self)
        self._tools_panel.set_project(proj)

        self._toolbar.setVisible(True)
        self._stack.setCurrentIndex(1)
        self.menuBar().show()

        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(proj.project_dir))
            if hasattr(self, "_welcome_page"):
                self._welcome_page.refresh_recents()
        self._refresh_recent_menu()

        self.statusBar().showMessage(f"Loaded project: {proj.project_dir}", 5000)

    def _save_current_project(self) -> None:
        if self._project is None:
            return
        self._dataset_panel.collect_state(self._project)
        save_project(self._project)
        self.statusBar().showMessage("Project saved.", 3000)

    # ------------------------------------------------------------------
    # Dialog launchers
    # ------------------------------------------------------------------

    def _open_source_manager(self) -> None:
        if self._project is None:
            return
        from .dialogs.source_manager import SourceManagerDialog

        dlg = SourceManagerDialog(self._project, parent=self)
        dlg.exec()
        self._dataset_panel.refresh_sources(self._project)
        self._tools_panel.refresh_overview()

    def _open_training_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.training_dialog import TrainingDialog

        dlg = TrainingDialog(self._project, parent=self)
        dlg.training_completed.connect(self._on_training_completed)
        dlg.exec()

    def _open_evaluation_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.evaluation_dialog import EvaluationDialog

        dlg = EvaluationDialog(self._project, parent=self)
        dlg.exec()

    def _open_history_dialog(self) -> None:
        if self._project is None:
            return
        from .dialogs.history_dialog import HistoryDialog

        dlg = HistoryDialog(self._project, parent=self)
        result = dlg.exec()
        if result == dlg.DialogCode.Accepted:
            model_paths = (
                [self._project.active_model_path]
                if self._project.active_model_path
                else []
            )
            self._tools_panel.refresh_model_selector(model_paths)

    def _on_training_completed(self, results: list) -> None:
        model_paths = [
            r.get("published_model_path", "")
            for r in results
            if r.get("published_model_path")
        ]
        self._tools_panel.refresh_model_selector(model_paths)
        self._save_current_project()

    def _export_stub(self) -> None:
        QMessageBox.information(
            self, "Export", "Export is not yet implemented in this release."
        )

    def _on_overlay_changed(self) -> None:
        settings = self._tools_panel.get_overlay_settings()
        self._canvas.set_overlay_visibility(settings.show_gt, settings.show_pred)
        self._canvas.set_class_filter(settings.visible_class_ids)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image and overlay GT labels."""
        self._canvas.clear_gt_detections()
        self._canvas.clear_pred_detections()
        ok = self._canvas.load_image(image_path)
        if not ok:
            return

        label_path = find_label_for_image(Path(image_path), source_path)
        if label_path is not None:
            import cv2

            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                class_names = self._project.class_names if self._project else ["object"]
                class_id_map = None
                if self._project is not None:
                    try:
                        class_id_map = source_class_id_map(
                            source_path, self._project.class_names
                        )
                    except Exception:
                        class_id_map = {}
                        logger.warning(
                            "Skipping incompatible source labels for preview: %s",
                            source_path,
                            exc_info=True,
                        )
                dets = parse_obb_label(label_path, w, h, class_id_map=class_id_map)
                self._canvas.set_gt_detections(dets, class_names=class_names)

        self._canvas.fit_in_view()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def project(self) -> Optional[DetectKitProject]:
        return self._project

    def canvas(self) -> OBBCanvas:
        return self._canvas

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_current_project()
        super().closeEvent(event)


# Backward-compat alias
MainWindow = DetectKitMainWindow
