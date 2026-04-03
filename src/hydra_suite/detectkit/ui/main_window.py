"""DetectKit main window -- three-panel layout with welcome page."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .canvas import OBBCanvas
from .models import DetectKitProject
from .panels.dataset_panel import DatasetPanel
from .panels.evaluation_panel import EvaluationPanel
from .panels.history_panel import HistoryPanel
from .panels.training_panel import TrainingPanel
from .project import (
    add_to_recent,
    create_project,
    load_recent_projects,
    open_project,
    project_file_path,
    save_project,
)
from .utils import find_label_for_image, parse_obb_label

logger = logging.getLogger(__name__)

_DARK_STYLESHEET = """
QMainWindow { background-color: #1e1e1e; }
QWidget { background-color: #1e1e1e; color: #ddd; }
QMenuBar { background-color: #252526; color: #ddd; }
QMenuBar::item:selected { background-color: #094771; }
QMenu { background-color: #252526; color: #ddd; }
QMenu::item:selected { background-color: #094771; }
QPushButton { background-color: #0e639c; color: white; border: none;
              padding: 8px 16px; border-radius: 4px; }
QPushButton:hover { background-color: #1177bb; }
QPushButton:disabled { background-color: #444; color: #888; }
QTabWidget::pane { border: 1px solid #333; }
QTabBar::tab { background-color: #252526; color: #aaa;
               padding: 6px 12px; border: 1px solid #333; }
QTabBar::tab:selected { background-color: #1e1e1e; color: #fff; }
QListWidget { background-color: #252526; border: 1px solid #333; color: #ddd; }
QListWidget::item:selected { background-color: #094771; }
QSplitter::handle { background-color: #333; width: 2px; }
QStatusBar { background-color: #007acc; color: white; }
QScrollArea { border: none; }
QGroupBox { border: 1px solid #444; border-radius: 4px;
            margin-top: 8px; padding-top: 12px; color: #ddd; }
QGroupBox::title { padding: 0 4px; }
"""


class MainWindow(QMainWindow):
    """DetectKit main application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DetectKit")
        self.setStyleSheet(_DARK_STYLESHEET)

        self._project: Optional[DetectKitProject] = None

        # Central stacked widget: welcome (0) vs workspace (1)
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Build pages
        self._build_welcome_page()
        self._build_workspace_page()

        # Menu bar
        self._build_menu_bar()

        # Status bar
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

        # Start on welcome page
        self._stack.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Welcome page
    # ------------------------------------------------------------------

    def _build_welcome_page(self) -> None:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo
        try:
            from multi_tracker.paths import get_brand_icon_bytes
            from PySide6.QtCore import QByteArray
            from PySide6.QtSvgWidgets import QSvgWidget

            logo_data = get_brand_icon_bytes("detectkit.svg")
            if logo_data:
                svg_widget = QSvgWidget()
                svg_widget.renderer().load(QByteArray(logo_data))
                svg_widget.setFixedSize(120, 120)
                outer.addWidget(svg_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        except Exception:
            pass

        # Title
        title = QLabel("DetectKit")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #eee;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        # Subtitle
        subtitle = QLabel("OBB Detection Model Training & Dataset Curation")
        subtitle.setStyleSheet("font-size: 14px; color: #aaa;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(subtitle)

        outer.addSpacing(20)

        # New Project button
        btn_new = QPushButton("New Project")
        btn_new.setFixedWidth(250)
        btn_new.clicked.connect(self.new_project)
        outer.addWidget(btn_new, alignment=Qt.AlignmentFlag.AlignCenter)

        # Open Project button
        btn_open = QPushButton("Open Project")
        btn_open.setFixedWidth(250)
        btn_open.clicked.connect(self.open_project_dialog)
        outer.addWidget(btn_open, alignment=Qt.AlignmentFlag.AlignCenter)

        outer.addSpacing(16)

        # Recent projects
        lbl_recent = QLabel("Recent Projects")
        lbl_recent.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(lbl_recent, alignment=Qt.AlignmentFlag.AlignCenter)

        self._welcome_recent_list = QListWidget()
        self._welcome_recent_list.setFixedWidth(400)
        self._welcome_recent_list.setMaximumHeight(200)
        self._welcome_recent_list.itemDoubleClicked.connect(
            self._on_recent_double_clicked
        )
        outer.addWidget(
            self._welcome_recent_list, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self._refresh_welcome_recent()
        self._stack.addWidget(page)  # index 0

    def _refresh_welcome_recent(self) -> None:
        self._welcome_recent_list.clear()
        for p in load_recent_projects():
            item = QListWidgetItem(p)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self._welcome_recent_list.addItem(item)

    def _on_recent_double_clicked(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if path_str:
            proj = open_project(Path(path_str))
            if proj is not None:
                self._load_project(proj)
            else:
                QMessageBox.warning(
                    self,
                    "Open Failed",
                    f"Could not open project at:\n{path_str}",
                )

    # ------------------------------------------------------------------
    # Workspace page
    # ------------------------------------------------------------------

    def _build_workspace_page(self) -> None:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Dataset panel
        self._dataset_panel = DatasetPanel()
        self._dataset_panel.setMinimumWidth(280)
        self._dataset_panel.setMaximumWidth(450)
        splitter.addWidget(self._dataset_panel)

        # Center: OBB Canvas
        self._canvas = OBBCanvas()
        splitter.addWidget(self._canvas)

        # Right: Tab widget
        self._right_tabs = QTabWidget()
        self._right_tabs.setMinimumWidth(380)
        self._right_tabs.setMaximumWidth(550)

        self._training_panel = TrainingPanel()
        self._evaluation_panel = EvaluationPanel()
        self._history_panel = HistoryPanel()

        self._right_tabs.addTab(self._training_panel, "Training")
        self._right_tabs.addTab(self._evaluation_panel, "Evaluation")
        self._right_tabs.addTab(self._history_panel, "History")

        splitter.addWidget(self._right_tabs)

        # Stretch factors: 0, 1, 0
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)
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

        # Recent Projects submenu
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
        for p in load_recent_projects():
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
                    self,
                    "Open Failed",
                    f"Could not open project at:\n{path_str}",
                )

    # ------------------------------------------------------------------
    # Project lifecycle
    # ------------------------------------------------------------------

    def new_project(self) -> None:
        """Create a new DetectKit project via directory + class name dialogs."""
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if not directory:
            return

        proj_dir = Path(directory)
        pf = project_file_path(proj_dir)

        # If a project already exists, offer to open it instead.
        if pf.exists():
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

        class_name, ok = QInputDialog.getText(
            self,
            "Class Name",
            "Enter the object class name (e.g. 'ant', 'bee'):",
            text="object",
        )
        if not ok or not class_name.strip():
            return

        proj = create_project(proj_dir, class_name.strip())
        self._load_project(proj)

    def open_project_dialog(self) -> None:
        """Open an existing project via directory picker."""
        directory = QFileDialog.getExistingDirectory(self, "Open DetectKit Project")
        if not directory:
            return

        proj = open_project(Path(directory))
        if proj is not None:
            self._load_project(proj)
        else:
            QMessageBox.warning(
                self,
                "Open Failed",
                f"No DetectKit project found in:\n{directory}",
            )

    def _load_project(self, proj: DetectKitProject) -> None:
        """Activate *proj*: wire panels, update title, switch to workspace."""
        self._project = proj
        self.setWindowTitle(f"DetectKit - {proj.project_dir.name}")

        # Wire panels
        self._dataset_panel.set_project(proj, self)
        self._training_panel.set_project(proj, self)
        self._evaluation_panel.set_project(proj, self)
        self._history_panel.set_project(proj, self)

        # Switch to workspace
        self._stack.setCurrentIndex(1)

        # Refresh recent lists
        add_to_recent(str(proj.project_dir))
        self._refresh_welcome_recent()
        self._refresh_recent_menu()

        self.statusBar().showMessage(f"Loaded project: {proj.project_dir}", 5000)

    def _save_current_project(self) -> None:
        """Collect state from panels and save the project."""
        if self._project is None:
            return

        self._dataset_panel.collect_state(self._project)
        self._training_panel.collect_state(self._project)
        self._evaluation_panel.collect_state(self._project)
        self._history_panel.collect_state(self._project)

        save_project(self._project)
        self.statusBar().showMessage("Project saved.", 3000)

    # ------------------------------------------------------------------
    # Image display
    # ------------------------------------------------------------------

    def show_image(self, source_path: str, image_path: str) -> None:
        """Load an image on the canvas and overlay any existing OBB labels."""
        self._canvas.clear_detections()
        ok = self._canvas.load_image(image_path)
        if not ok:
            return

        label_path = find_label_for_image(Path(image_path), source_path)
        if label_path is not None:
            # Need image dimensions for coordinate conversion
            import cv2

            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                class_name = self._project.class_name if self._project else "object"
                dets = parse_obb_label(label_path, w, h)
                self._canvas.set_detections(dets, class_name=class_name)

        self._canvas.fit_in_view()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def project(self) -> Optional[DetectKitProject]:
        """Return the currently loaded project, or *None*."""
        return self._project

    def canvas(self) -> OBBCanvas:
        """Return the OBB canvas widget."""
        return self._canvas

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        """Auto-save the project on close."""
        self._save_current_project()
        super().closeEvent(event)
