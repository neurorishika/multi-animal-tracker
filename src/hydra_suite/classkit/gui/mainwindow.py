"""
ClassKit Main Window - Polished and feature-complete UI
"""

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QThreadPool, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .color_utils import best_text_color, build_category_color_map, to_hex


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClassKit: Active Learning Dataset Builder")
        self.resize(1600, 1000)

        # Project state
        self.project_path = None
        self.db_path = None
        self.embeddings = None
        self.umap_coords = None
        self.cluster_assignments = None
        self.image_paths = []
        self.image_labels = []
        self.image_confidences = []
        self.classes = ["class_1", "class_2"]
        self.selected_point_index = None
        self.candidate_indices = []
        self.round_labeled_indices = []
        self.explorer_mode = "explore"
        self.hover_locked = False
        self._label_shortcuts = []
        self.label_history = []
        self.last_assigned_stack = []
        self.last_umap_params = {"n_neighbors": 15, "min_dist": 0.1}
        self.last_preview_index = None
        self._history_icon_cache = {}
        self._history_thumb_load_budget = 2
        self._command_busy = False
        self._command_block_until = 0.0
        self._command_squelch_pending = False
        self._active_jobs = 0  # reference count: number of running background workers
        self._pending_label_updates = {}
        self._autosave_interval_ms = 5000
        self._autosave_last_save_time = None
        self._autosave_heartbeat_on = False
        self._trained_classifier = None
        self._last_training_settings = None
        self._yolo_model_path = None  # Path to loaded YOLO classification model
        self._model_probs = None  # (N, C) per-image class probabilities
        self._model_class_names = None  # class names from (YOLO/tiny) model
        self.umap_model_coords = None  # UMAP computed in model logits space
        self.pca_model_coords = None  # PCA computed in model logits/probability space
        self._show_model_umap = False  # Explorer toggle: embedding vs model UMAP
        self._show_model_pca = False  # Explorer toggle: embedding vs model PCA
        self._al_candidates = None  # np.ndarray of selected AL batch indices
        self._active_model_mode = None  # "yolo", "tiny", or None
        self._current_knn_neighbors = []
        self._stepper = None
        self._custom_shortcuts: dict = {}  # action_name → key sequence string
        self._outline_threshold = 0.60

        # Display enhancement settings (sync with PoseKit)
        self.clahe_clip = 2.0
        self.clahe_grid = (8, 8)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(False)
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        self._autosave_timer.timeout.connect(self._flush_pending_label_updates)

        self._autosave_heartbeat_timer = QTimer(self)
        self._autosave_heartbeat_timer.setSingleShot(False)
        self._autosave_heartbeat_timer.setInterval(700)
        self._autosave_heartbeat_timer.timeout.connect(self._tick_autosave_heartbeat)

        self._history_refresh_timer = QTimer(self)
        self._history_refresh_timer.setSingleShot(True)
        self._history_refresh_timer.setInterval(40)
        self._history_refresh_timer.timeout.connect(self.refresh_label_history_strip)

        self._plot_refresh_pending = False
        self._plot_refresh_force_fit = False
        self._plot_refresh_timer = QTimer(self)
        self._plot_refresh_timer.setSingleShot(True)
        self._plot_refresh_timer.setInterval(24)
        self._plot_refresh_timer.timeout.connect(self._flush_explorer_update)

        self._pending_preview_index = None
        self._pending_preview_source = "hover"
        self._preview_refresh_timer = QTimer(self)
        self._preview_refresh_timer.setSingleShot(True)
        self._preview_refresh_timer.setInterval(20)
        self._preview_refresh_timer.timeout.connect(self._flush_preview_update)

        self._context_refresh_pending = False
        self._context_refresh_timer = QTimer(self)
        self._context_refresh_timer.setSingleShot(True)
        self._context_refresh_timer.setInterval(60)
        self._context_refresh_timer.timeout.connect(self._flush_context_update)

        # Threadpool
        self.threadpool = QThreadPool.globalInstance()

        # Apply dark theme stylesheet
        self.apply_stylesheet()

        # Setup UI
        self.setup_menus()
        self.setup_toolbar()
        self.setup_central_widget()
        self.setup_statusbar()

    def apply_stylesheet(self):
        """Apply modern dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }
            QListWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-radius: 4px;
                margin: 2px 0px;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #2a2d2e;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #888888;
            }
            QLabel {
                color: #cccccc;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
                spacing: 8px;
                padding: 6px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 8px 12px;
                color: #cccccc;
            }
            QToolButton:hover {
                background-color: #2a2d2e;
            }
            QToolButton:pressed {
                background-color: #094771;
            }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
            }
            QMenuBar {
                background-color: #252526;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 4px;
            }
            QMenuBar::item {
                padding: 6px 12px;
                background-color: transparent;
            }
            QMenuBar::item:selected {
                background-color: #2a2d2e;
            }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 8px 24px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QSplitter::handle {
                background-color: #3e3e42;
            }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
            }
            QProgressBar::chunk {
                background-color: #0e639c;
                border-radius: 3px;
            }
        """)

    def setup_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_project = QAction("&New Project...", self)
        new_project.setShortcut("Ctrl+N")
        new_project.triggered.connect(self.new_project)
        file_menu.addAction(new_project)

        open_project = QAction("&Open Project...", self)
        open_project.setShortcut("Ctrl+O")
        open_project.triggered.connect(self.open_project)
        file_menu.addAction(open_project)

        save_project = QAction("&Save Project", self)
        save_project.setShortcut("Ctrl+S")
        save_project.triggered.connect(self.save_project)
        file_menu.addAction(save_project)

        file_menu.addSeparator()

        source_mgr_action = QAction("&Source Manager...", self)
        source_mgr_action.setShortcut("Ctrl+I")
        source_mgr_action.triggered.connect(self.manage_sources)
        file_menu.addAction(source_mgr_action)

        file_menu.addSeparator()

        export_action = QAction("&Export Dataset...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_dataset)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Compute menu
        compute_menu = menubar.addMenu("&Compute")

        embed_action = QAction("Compute &Embeddings...", self)
        embed_action.triggered.connect(self.compute_embeddings)
        compute_menu.addAction(embed_action)

        cluster_action = QAction("&Cluster Data...", self)
        cluster_action.triggered.connect(self.cluster_data)
        compute_menu.addAction(cluster_action)

        umap_action = QAction("Compute &UMAP...", self)
        umap_action.triggered.connect(self.compute_umap)
        compute_menu.addAction(umap_action)

        compute_menu.addSeparator()

        train_action = QAction("&Train Classifier...", self)
        train_action.setShortcut("Ctrl+T")
        train_action.triggered.connect(self.train_classifier)
        compute_menu.addAction(train_action)

        load_checkpoint_action = QAction("&Load Classifier Checkpoint...", self)
        load_checkpoint_action.triggered.connect(self.load_classifier_checkpoint)
        compute_menu.addAction(load_checkpoint_action)

        model_history_action = QAction("&Previously Trained Models...", self)
        model_history_action.setShortcut("Ctrl+Shift+H")
        model_history_action.setStatusTip(
            "Browse and load past trained models registered in this project"
        )
        model_history_action.triggered.connect(self._open_model_history)
        compute_menu.addAction(model_history_action)

        build_al_action = QAction("Build Active &Learning Batch...", self)
        build_al_action.setShortcut("Ctrl+Shift+B")
        build_al_action.triggered.connect(self._build_al_batch)
        compute_menu.addAction(build_al_action)

        compute_menu.addSeparator()

        self._autolabel_apriltag_action = QAction("Auto-label AprilTags\u2026", self)
        self._autolabel_apriltag_action.setEnabled(False)
        self._autolabel_apriltag_action.triggered.connect(self._run_apriltag_autolabel)
        compute_menu.addAction(self._autolabel_apriltag_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)

        view_menu.addSeparator()

        self.act_enhance = QAction("&Enhance Contrast (CLAHE)", self)
        self.act_enhance.setShortcut("Ctrl+Shift+E")
        self.act_enhance.setCheckable(True)
        self.act_enhance.setChecked(False)
        self.act_enhance.setStatusTip(
            "Toggle CLAHE contrast enhancement for better visibility"
        )
        self.act_enhance.toggled.connect(self.on_enhance_toggled)
        view_menu.addAction(self.act_enhance)

        enhance_settings = QAction("Contrast &Settings...", self)
        enhance_settings.triggered.connect(self.open_contrast_settings)
        view_menu.addAction(enhance_settings)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About ClassKit", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setMovable(False)
        self.toolbar = toolbar  # keep reference for busy-blocking
        self.addToolBar(toolbar)

        # Project section
        new_btn = QAction("New", self)
        new_btn.setStatusTip("Create a new ClassKit project")
        new_btn.triggered.connect(self.new_project)
        toolbar.addAction(new_btn)

        open_btn = QAction("Open", self)
        open_btn.setStatusTip("Open an existing project")
        open_btn.triggered.connect(self.open_project)
        toolbar.addAction(open_btn)

        save_btn = QAction("Save", self)
        save_btn.setStatusTip("Save project labels")
        save_btn.triggered.connect(self.save_project)
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        # Data section
        source_mgr_btn = QAction("Source Manager", self)
        source_mgr_btn.setStatusTip("View, add, or remove image source folders")
        source_mgr_btn.triggered.connect(self.manage_sources)
        toolbar.addAction(source_mgr_btn)

        toolbar.addSeparator()

        # Compute section
        embed_btn = QAction("Embed", self)
        embed_btn.setStatusTip("Compute embeddings")
        embed_btn.triggered.connect(self.compute_embeddings)
        toolbar.addAction(embed_btn)

        cluster_btn = QAction("Cluster", self)
        cluster_btn.setStatusTip("Cluster embeddings")
        cluster_btn.triggered.connect(self.cluster_data)
        toolbar.addAction(cluster_btn)

        toolbar.addSeparator()

        # Training section
        train_btn = QAction("Train", self)
        train_btn.setStatusTip("Train classifier")
        train_btn.triggered.connect(self.train_classifier)
        toolbar.addAction(train_btn)

        history_btn = QAction("Previously Trained Models", self)
        history_btn.setStatusTip("Browse and load past trained models")
        history_btn.triggered.connect(self._open_model_history)
        toolbar.addAction(history_btn)

        toolbar.addSeparator()

        # Export section
        export_btn = QAction("Export", self)
        export_btn.setStatusTip("Export labeled dataset")
        export_btn.triggered.connect(self.export_dataset)
        toolbar.addAction(export_btn)

    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown before any project is opened."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("classkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="classkit.svg",
            tagline="Active Learning Dataset Builder",
            buttons=[
                ButtonDef(label="New Project\u2026", callback=self.new_project),
                ButtonDef(label="Open Project\u2026", callback=self.open_project),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        project_path = Path(path)
        if project_path.exists():
            self.project_path = project_path
            self.db_path = project_path / "classkit.db"
            if self.db_path.exists():
                self.load_project_data()
                self.update_context_panel()
                self.status.showMessage(f"Opened project: {self.project_path.name}")
            else:
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Invalid Project",
                    "This directory does not contain a valid ClassKit project.\n\n"
                    "Expected file: classkit.db",
                )
                if hasattr(self, "_recents_store"):
                    self._recents_store.remove(path)
                    if hasattr(self, "_welcome_page"):
                        self._welcome_page.refresh_recents()
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    def setup_central_widget(self):
        """Setup main UI layout for UMAP exploration and fast labeling."""
        self.splitter = QSplitter(Qt.Horizontal)
        # Stacked widget: page 0 = welcome logo, page 1 = working UI
        self._stacked = QStackedWidget()
        self._stacked.addWidget(self._make_welcome_page())  # index 0
        self._stacked.addWidget(self.splitter)  # index 1
        self.setCentralWidget(self._stacked)

        from .widgets.explorer import ExplorerView
        from .widgets.image_viewer import ImageCanvas

        # 1) Left: project metadata + settings + label controls
        self.context_panel = QScrollArea()
        self.context_panel.setWidgetResizable(True)
        self.context_panel.setFixedWidth(380)
        self.context_panel.setStyleSheet(
            "QScrollArea { border: none; background: #1e1e1e; }"
        )

        self.context_content = QWidget()
        self.context_layout = QVBoxLayout(self.context_content)
        self.context_layout.setContentsMargins(16, 16, 16, 16)
        self.context_layout.setSpacing(14)

        # ── Group 1: Project Info ─────────────────────────────────────
        group_info = QGroupBox("Project Info")
        layout_info = QVBoxLayout(group_info)

        self.context_info = QLabel("No project loaded.")
        self.context_info.setWordWrap(True)
        self.context_info.setStyleSheet(
            "color: #cccccc; font-size: 12px; line-height: 1.5;"
        )
        layout_info.addWidget(self.context_info)

        autosave_row = QHBoxLayout()
        autosave_row.addWidget(QLabel("Autosave (s):"))
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 300)
        self.autosave_spin.setValue(max(1, self._autosave_interval_ms // 1000))
        self.autosave_spin.valueChanged.connect(self.on_autosave_interval_changed)
        autosave_row.addWidget(self.autosave_spin)
        layout_info.addLayout(autosave_row)

        self.context_layout.addWidget(group_info)

        # ── Group 2: Selection & Neighbors ────────────────────────────
        group_selection = QGroupBox("Selection & Neighbors")
        layout_selection = QVBoxLayout(group_selection)

        self.selection_info = QLabel("No point selected.")
        self.selection_info.setWordWrap(True)
        self.selection_info.setStyleSheet("color: #aaaaaa; padding: 4px;")
        layout_selection.addWidget(self.selection_info)

        self.knn_info = QTextEdit()
        self.knn_info.setReadOnly(True)
        self.knn_info.setMaximumHeight(100)
        self.knn_info.setStyleSheet(
            "background: #111; border: 1px solid #333; border-radius: 4px;"
        )
        layout_selection.addWidget(self.knn_info)

        knn_actions = QHBoxLayout()
        self.knn_jump_btn = QPushButton("Jump")
        self.knn_jump_btn.clicked.connect(self.jump_to_nearest_neighbor)
        self.knn_jump_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_jump_btn)

        self.knn_bulk_btn = QPushButton("Bulk Label 5")
        self.knn_bulk_btn.clicked.connect(self.bulk_label_nearest_neighbors)
        self.knn_bulk_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_bulk_btn)
        layout_selection.addLayout(knn_actions)

        self.context_layout.addWidget(group_selection)

        # ── Group 3: Labeling ─────────────────────────────────────────
        group_labeling = QGroupBox("Labeling")
        layout_labeling = QVBoxLayout(group_labeling)

        # Search / Filter classes
        self.class_search = QLineEdit()
        self.class_search.setPlaceholderText("Filter classes...")
        self.class_search.textChanged.connect(self.filter_label_buttons)
        layout_labeling.addWidget(self.class_search)

        self.label_buttons_container = QWidget()
        self.label_buttons_layout = QGridLayout(self.label_buttons_container)
        self.label_buttons_layout.setSpacing(6)
        layout_labeling.addWidget(self.label_buttons_container)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.prev_btn.clicked.connect(self.on_prev_image)
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.on_next_image)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        layout_labeling.addLayout(nav_row)

        self.context_layout.addWidget(group_labeling)

        # ── Actions ───────────────────────────────────────────────────
        edit_row = QHBoxLayout()
        btn_edit_classes = QPushButton("Edit Scheme")
        btn_edit_classes.clicked.connect(self.open_class_editor)
        btn_edit_classes.setStyleSheet("background: #1a3a5e; padding: 5px;")
        edit_row.addWidget(btn_edit_classes)

        btn_edit_shortcuts = QPushButton("Shortcuts")
        btn_edit_shortcuts.clicked.connect(self.open_shortcut_editor)
        btn_edit_shortcuts.setStyleSheet("background: #3a3a1a; padding: 5px;")
        edit_row.addWidget(btn_edit_shortcuts)
        self.context_layout.addLayout(edit_row)

        self.shortcut_help = QLabel()
        self.shortcut_help.setWordWrap(True)
        self.shortcut_help.setStyleSheet("font-size: 11px; color: #888; padding: 4px;")
        self.context_layout.addWidget(self.shortcut_help)
        self._refresh_shortcut_help()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.context_layout.addWidget(self.progress_bar)

        self.context_layout.addStretch()
        self.context_panel.setWidget(self.context_content)

        # 2) Center: main UMAP explorer + Tabs for Metrics/Batch
        self.center_tabs = QWidget()
        center_tab_layout = QVBoxLayout(self.center_tabs)
        center_tab_layout.setContentsMargins(0, 0, 0, 0)

        from PySide6.QtWidgets import QTabWidget

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; background: #1e1e1e; }
            QTabBar::tab { background: #252526; color: #888888; padding: 10px 20px; border: 1px solid #3e3e42; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; border-bottom: 1px solid #1e1e1e; }
            QTabBar::tab:hover { background: #2d2d2d; }
        """)

        # Tab 1: Explorer
        self.explorer_page = QWidget()
        explorer_layout = QVBoxLayout(self.explorer_page)
        explorer_layout.setContentsMargins(8, 8, 8, 8)
        explorer_layout.setSpacing(8)

        self.explorer = ExplorerView()
        self.explorer.set_uncertainty_outline_threshold(self._outline_threshold)
        self.explorer.point_clicked.connect(self.on_explorer_point_clicked)
        self.explorer.point_hovered.connect(self.on_explorer_point_hovered)
        self.explorer.empty_double_clicked.connect(
            self.on_explorer_background_double_click
        )
        explorer_layout.addWidget(self.explorer, 1)

        # UMAP space toggle row (embedding vs model logits)
        umap_toggle_row = QHBoxLayout()
        umap_toggle_row.setSpacing(8)
        umap_toggle_label = QLabel("<b>UMAP Space:</b>")
        umap_toggle_row.addWidget(umap_toggle_label)
        self.btn_umap_embedding = QPushButton("Embeddings")
        self.btn_umap_embedding.setCheckable(True)
        self.btn_umap_embedding.setChecked(True)
        self.btn_umap_embedding.setFixedWidth(110)
        self.btn_umap_embedding.clicked.connect(
            lambda: self._switch_projection_space("embedding")
        )
        umap_toggle_row.addWidget(self.btn_umap_embedding)
        self.btn_umap_model = QPushButton("Model UMAP")
        self.btn_umap_model.setCheckable(True)
        self.btn_umap_model.setChecked(False)
        self.btn_umap_model.setEnabled(False)
        self.btn_umap_model.setFixedWidth(110)
        self.btn_umap_model.setToolTip(
            "Switch explorer to UMAP of trained model predictions (computed automatically after checkpoint load)"
        )
        self.btn_umap_model.clicked.connect(
            lambda: self._switch_projection_space("model_umap")
        )
        umap_toggle_row.addWidget(self.btn_umap_model)

        self.btn_pca_model = QPushButton("Model PCA")
        self.btn_pca_model.setCheckable(True)
        self.btn_pca_model.setChecked(False)
        self.btn_pca_model.setEnabled(False)
        self.btn_pca_model.setFixedWidth(110)
        self.btn_pca_model.setToolTip(
            "Switch explorer to PCA of model predictions (computed on demand)"
        )
        self.btn_pca_model.clicked.connect(
            lambda: self._switch_projection_space("model_pca")
        )
        umap_toggle_row.addWidget(self.btn_pca_model)

        umap_toggle_row.addStretch(1)
        explorer_layout.addLayout(umap_toggle_row)

        # Labeling controls row
        self.label_controls_row = QHBoxLayout()
        self.label_controls_row.setSpacing(12)

        self.label_controls_row.addWidget(QLabel("<b>Set:</b>"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(6)
        self.sample_spin.setToolTip("Unlabeled images to sample from each cluster")
        self.label_controls_row.addWidget(self.sample_spin)

        self.btn_sample_next = QPushButton("Sample Next")
        self.btn_sample_next.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 6px 12px;"
        )
        self.btn_sample_next.clicked.connect(self.on_sample_next_triggered)
        self.label_controls_row.addWidget(self.btn_sample_next)

        self.btn_clear_candidates = QPushButton("Clear Candidates")
        self.btn_clear_candidates.setStyleSheet(
            "background: #3e3e42; padding: 6px 12px;"
        )
        self.btn_clear_candidates.clicked.connect(self.on_clear_candidates_triggered)
        self.label_controls_row.addWidget(self.btn_clear_candidates)

        self.label_controls_row.addStretch(1)

        self.mode_toggle_lbl = QLabel("Mode:")
        self.label_controls_row.addWidget(self.mode_toggle_lbl)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Explore (clusters)", "explore")
        self.view_mode_combo.addItem("Labeling (labels)", "labeling")
        self.view_mode_combo.addItem("Predictions", "predictions")
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        self.label_controls_row.addWidget(self.view_mode_combo)

        self.label_controls_row.addSpacing(8)
        self.label_controls_row.addWidget(QLabel("Outline <"))
        self.outline_threshold_spin = QDoubleSpinBox()
        self.outline_threshold_spin.setRange(0.0, 1.0)
        self.outline_threshold_spin.setSingleStep(0.05)
        self.outline_threshold_spin.setDecimals(2)
        self.outline_threshold_spin.setValue(self._outline_threshold)
        self.outline_threshold_spin.setFixedWidth(72)
        self.outline_threshold_spin.setToolTip(
            "Confidence threshold for white uncertainty outlines in Explorer mode.\n"
            "Set to 0 to disable uncertainty outlines."
        )
        self.outline_threshold_spin.valueChanged.connect(
            self.on_outline_threshold_changed
        )
        self.label_controls_row.addWidget(self.outline_threshold_spin)

        explorer_layout.addLayout(self.label_controls_row)

        # ── Inline Active Learning panel ────────────────────────────────

        al_group = QGroupBox("Active Learning")
        al_group.setStyleSheet("""
            QGroupBox { color: #777; border: 1px solid #3e3e42; border-radius: 4px;
                        margin-top: 8px; font-size: 11px; padding-top: 2px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        """)
        al_group_layout = QVBoxLayout(al_group)
        al_group_layout.setContentsMargins(8, 4, 8, 6)
        al_group_layout.setSpacing(4)

        al_ctrl_row = QHBoxLayout()
        al_ctrl_row.setSpacing(8)

        self.al_status_label = QLabel(
            "Model: none  |  Labeled: 0 / 0  |  Predictions: none"
        )
        self.al_status_label.setStyleSheet("color: #777; font-size: 11px;")
        al_ctrl_row.addWidget(self.al_status_label, 1)

        al_ctrl_row.addWidget(QLabel("<b>Batch:</b>"))
        self.al_batch_spin = QSpinBox()
        self.al_batch_spin.setRange(5, 500)
        self.al_batch_spin.setValue(50)
        self.al_batch_spin.setFixedWidth(68)
        self.al_batch_spin.setStyleSheet("background: #252526;")
        al_ctrl_row.addWidget(self.al_batch_spin)

        self.al_build_btn = QPushButton("⚙  Build Batch")
        self.al_build_btn.setStyleSheet(
            "background: #0e639c; font-weight: bold; padding: 4px 12px;"
        )
        self.al_build_btn.setToolTip(
            "Select the highest-value unlabeled images for labeling —\n"
            "40% uncertain · 35% diverse · 15% representative · 10% audit"
        )
        self.al_build_btn.clicked.connect(self._build_al_batch)
        al_ctrl_row.addWidget(self.al_build_btn)

        self.al_candidates_badge = QLabel("")
        self.al_candidates_badge.setStyleSheet(
            "color: #ccc; font-size: 11px; font-weight: bold;"
        )
        al_ctrl_row.addWidget(self.al_candidates_badge)

        self.al_start_btn = QPushButton("▶  Label")
        self.al_start_btn.setStyleSheet(
            "background: #28a745; color: white; font-weight: bold; padding: 4px 12px;"
        )
        self.al_start_btn.setEnabled(False)
        self.al_start_btn.setToolTip(
            "Switch to Labeling mode with AL candidates as the active set"
        )
        self.al_start_btn.clicked.connect(self._start_labeling_al_batch)
        al_ctrl_row.addWidget(self.al_start_btn)

        self.al_highlight_btn = QPushButton("◆ Highlight")
        self.al_highlight_btn.setStyleSheet("background: #3e3e42; padding: 4px 10px;")
        self.al_highlight_btn.setEnabled(False)
        self.al_highlight_btn.setToolTip(
            "Show AL candidates highlighted on the UMAP without entering Labeling mode"
        )
        self.al_highlight_btn.clicked.connect(self._highlight_al_batch_on_map)
        al_ctrl_row.addWidget(self.al_highlight_btn)

        al_group_layout.addLayout(al_ctrl_row)

        # Candidate list — hidden until a batch is built
        self.al_candidate_list = QListWidget()
        self.al_candidate_list.setFixedHeight(100)
        self.al_candidate_list.setStyleSheet(
            "background: #1a1a1a; color: #ccc; font-size: 11px; font-family: monospace;"
        )
        self.al_candidate_list.setAlternatingRowColors(True)
        self.al_candidate_list.itemDoubleClicked.connect(self._al_candidate_goto)
        self.al_candidate_list.hide()
        al_group_layout.addWidget(self.al_candidate_list)

        explorer_layout.addWidget(al_group)
        # ──────────────────────────────────────────────────────────────────

        self.history_title = QLabel("<b>Recent Labels (click to undo + relabel)</b>")
        explorer_layout.addWidget(self.history_title)

        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFixedHeight(140)
        self.history_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.history_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_scroll_content = QWidget()
        self.history_scroll_layout = QHBoxLayout(self.history_scroll_content)
        self.history_scroll_layout.setContentsMargins(8, 8, 8, 8)
        self.history_scroll_layout.setSpacing(8)
        self._history_slots = []
        for _ in range(24):
            card = QFrame()
            card.setFixedSize(110, 110)
            card.setToolTip("")
            card.setStyleSheet(
                "padding: 4px; border: 1px solid #3e3e42; border-radius: 6px;"
            )
            card.setProperty("history_index", None)
            card.installEventFilter(self)

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(4, 4, 4, 4)
            card_layout.setSpacing(4)

            thumb = QLabel()
            thumb.setFixedSize(72, 72)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setAttribute(Qt.WA_TransparentForMouseEvents, True)

            caption = QLabel("")
            caption.setAlignment(Qt.AlignCenter)
            caption.setWordWrap(True)
            caption.setAttribute(Qt.WA_TransparentForMouseEvents, True)

            card_layout.addWidget(thumb, 0, Qt.AlignCenter)
            card_layout.addWidget(caption, 0, Qt.AlignCenter)
            card.hide()

            self._history_slots.append((card, thumb, caption))
            self.history_scroll_layout.addWidget(card)

        self.history_scroll_layout.addStretch(1)
        self.history_scroll.setWidget(self.history_scroll_content)
        explorer_layout.addWidget(self.history_scroll)

        self.tabs.addTab(self.explorer_page, "Explorer")

        # Tab 2: Metrics
        self.metrics_page = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_page)
        metrics_layout.setContentsMargins(8, 8, 8, 8)
        metrics_layout.setSpacing(6)

        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMaximumHeight(180)
        self.metrics_view.setPlaceholderText(
            "Train a model to see evaluation metrics here."
        )
        self.metrics_view.setStyleSheet(
            "font-family: 'SF Mono', 'Roboto Mono', monospace; font-size: 12px; background: #111;"
        )
        metrics_layout.addWidget(self.metrics_view)

        # Matplotlib figure area (confusion matrix + per-class bars)
        self.metrics_figure_label = QLabel("(Train a model to see visualizations)")
        self.metrics_figure_label.setAlignment(Qt.AlignCenter)
        self.metrics_figure_label.setStyleSheet(
            "background: #111; color: #555; border-radius: 4px; padding: 20px;"
        )
        metrics_figure_scroll = QScrollArea()
        metrics_figure_scroll.setWidgetResizable(True)
        metrics_figure_scroll.setWidget(self.metrics_figure_label)
        metrics_layout.addWidget(metrics_figure_scroll, 1)

        self.tabs.addTab(self.metrics_page, "Metrics")

        center_tab_layout.addWidget(self.tabs)

        # 3) Right: large hover/selection image preview
        self.preview_panel = QWidget()
        self.preview_panel.setFixedWidth(420)
        preview_layout = QVBoxLayout(self.preview_panel)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(10)

        preview_title = QLabel(
            "<b style='font-size:16px; color:#ffffff;'>Image Preview</b>"
        )
        preview_layout.addWidget(preview_title)

        # Enhance toggle

        self.preview_canvas = ImageCanvas()
        self.preview_canvas.setMinimumHeight(320)
        preview_layout.addWidget(self.preview_canvas, 1)

        self.cb_enhance = QCheckBox("Enhance contrast (CLAHE)")
        self.cb_enhance.setStyleSheet("color: #aaa; font-size: 11px;")
        self.cb_enhance.toggled.connect(self.on_enhance_toggled)
        preview_layout.addWidget(self.cb_enhance)

        self.preview_info = QLabel(
            "<div style='line-height:1.55; color:#aaaaaa;'>"
            "Hover a point to preview the source image.<br>"
            "Select a point, then label using 1-9 or buttons."
            "</div>"
        )
        self.preview_info.setWordWrap(True)
        self.preview_info.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px;"
        )
        preview_layout.addWidget(self.preview_info)

        self.splitter.addWidget(self.context_panel)
        self.splitter.addWidget(self.center_tabs)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setSizes([380, 920, 420])

        self.setup_label_shortcuts()
        self.rebuild_label_buttons()

    def setup_statusbar(self):
        """Setup status bar."""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.labeling_progress_label = QLabel()
        self.labeling_progress_label.setVisible(False)
        self.labeling_progress_label.setStyleSheet(
            "color: #7ec8e3; font-size: 12px; padding: 0 8px;"
        )
        self.status.addPermanentWidget(self.labeling_progress_label)
        self.autosave_indicator = QLabel("Autosave ○ no project")
        self.status.addPermanentWidget(self.autosave_indicator)
        self.status.showMessage("Ready • No project loaded")
        self._autosave_heartbeat_timer.start()

    # ================== Project Management ==================

    def new_project(self):
        """Create a new ClassKit project."""
        from .dialogs import NewProjectDialog

        try:
            self._flush_pending_label_updates(force=True)
            dialog = NewProjectDialog(self)
            if dialog.exec():
                project_info = dialog.get_project_info()
                self.project_path = Path(project_info["path"])
                self.db_path = self.project_path / "classkit.db"

                # Create project directory
                self.project_path.mkdir(parents=True, exist_ok=True)

                # Initialize database
                from ..store.db import ClassKitDB

                ClassKitDB(self.db_path)

                # Save project config
                config = {
                    "name": project_info["name"],
                    "version": "1.0",
                    "classes": project_info.get("classes", []),
                    "autosave_interval_ms": self._autosave_interval_ms,
                }
                with open(self.project_path / "project.json", "w") as f:
                    json.dump(config, f, indent=2)

                # Save labeling scheme if one was selected
                scheme = project_info.get("scheme")
                if scheme is not None:
                    with open(self.project_path / "scheme.json", "w") as f:
                        json.dump(scheme.to_dict(), f, indent=2)

                self.classes = project_info.get("classes", []) or ["class_1", "class_2"]
                self.rebuild_label_buttons()
                self.setup_label_shortcuts()
                self._refresh_shortcut_help()

                self.update_context_panel()
                self.status.showMessage(f"Created project: {project_info['name']}")

                # If no scheme was set, offer to open the editor now
                if project_info.get("scheme") is None:
                    reply = QMessageBox.question(
                        self,
                        "Project Created",
                        f"Project created:\n{self.project_path}\n\n"
                        "No labeling scheme was defined yet.\n"
                        "Would you like to open the Class Scheme Editor now?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        self.open_class_editor()
                else:
                    QMessageBox.information(
                        self,
                        "Project Created",
                        f"Successfully created project:\n\n{self.project_path}",
                    )

                # Always prompt to add sources after a brand new project
                QTimer.singleShot(100, self._prompt_adjust_sources_if_empty)
                if hasattr(self, "_recents_store"):
                    self._recents_store.add(str(self.project_path))
                if hasattr(self, "_stacked"):
                    self._stacked.setCurrentIndex(1)  # reveal working UI
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\\n{e}")

    def open_project(self):
        """Open an existing project."""
        self._flush_pending_label_updates(force=True)
        project_dir = QFileDialog.getExistingDirectory(
            self, "Open ClassKit Project", str(Path.home()), QFileDialog.ShowDirsOnly
        )

        if project_dir:
            self.project_path = Path(project_dir)
            self.db_path = self.project_path / "classkit.db"

            if not self.db_path.exists():
                QMessageBox.warning(
                    self,
                    "Invalid Project",
                    "This directory does not contain a valid ClassKit project.\\n\\n"
                    + "Expected file: classkit.db",
                )
                return

            self.load_project_data()
            self.update_context_panel()
            self.status.showMessage(f"Opened project: {self.project_path.name}")
            QTimer.singleShot(200, self._guide_to_next_step)

    def save_project(self):
        """Force immediate flush of all pending label updates."""
        if not self.db_path:
            self.status.showMessage("No project loaded; nothing to save")
            return

        self.status.showMessage("Saving project...")
        self._flush_pending_label_updates(force=True)
        self.status.showMessage("Project saved", 2000)

    def load_project_data(self):
        """Load project data from database."""
        if not self.db_path:
            return

        try:
            from pathlib import Path

            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

            # Migrate any non-resolved paths from older ingests (no-op if already resolved)
            db.migrate_paths_to_resolved()

            # Load image paths
            path_strings = db.get_all_image_paths()
            self.image_paths = [Path(p) for p in path_strings]
            self.image_labels = db.get_all_labels()
            self.image_confidences = [None] * len(self.image_paths)

            # Load class names from project config
            project_config_path = self.project_path / "project.json"
            if project_config_path.exists():
                with open(project_config_path, "r") as f:
                    config = json.load(f)
                self.classes = config.get("classes", []) or ["class_1", "class_2"]
                autosave_interval = int(
                    config.get("autosave_interval_ms", self._autosave_interval_ms)
                )
                autosave_interval = max(1000, min(300000, autosave_interval))
                self._autosave_interval_ms = autosave_interval
                self._autosave_timer.setInterval(self._autosave_interval_ms)
                if hasattr(self, "autosave_spin"):
                    self.autosave_spin.blockSignals(True)
                    self.autosave_spin.setValue(self._autosave_interval_ms // 1000)
                    self.autosave_spin.blockSignals(False)
                # Load saved custom shortcuts
                saved_shortcuts = config.get("custom_shortcuts", {})
                if isinstance(saved_shortcuts, dict):
                    self._custom_shortcuts = saved_shortcuts

                # Load CLAHE settings
                self.clahe_clip = float(config.get("clahe_clip", 2.0))
                self.clahe_grid = tuple(config.get("clahe_grid", [8, 8]))
                if hasattr(self, "preview_canvas"):
                    self.preview_canvas.set_clahe_params(
                        self.clahe_clip, self.clahe_grid
                    )
            else:
                self.classes = ["class_1", "class_2"]

            self.rebuild_label_buttons()
            self.setup_label_shortcuts()
            self._refresh_shortcut_help()
            self.request_refresh_label_history_strip()
            self._pending_label_updates = {}
            self._autosave_last_save_time = None
            self._update_autosave_heartbeat_text()
            self.try_autoload_cached_artifacts(db)
            self.update_explorer_plot()

            if hasattr(self, "_autolabel_apriltag_action"):
                self._autolabel_apriltag_action.setEnabled(
                    self.project_path is not None
                )

            self.status.showMessage(
                f"Loaded {len(self.image_paths):,} images from database"
            )
            if hasattr(self, "_recents_store"):
                self._recents_store.add(str(self.project_path))
            if hasattr(self, "_stacked"):
                self._stacked.setCurrentIndex(1)  # reveal working UI
        except Exception as e:
            QMessageBox.warning(
                self, "Load Error", f"Failed to load project data:\\n{e}"
            )

    def update_context_panel(self):
        """Update context panel with project info."""
        if not self.project_path:
            self.context_info.setText(
                "<div style='line-height: 1.65;'>"
                "No project loaded.<br><br>"
                "Get started:<br>"
                "• <b>File → New Project</b> to create<br>"
                "• <b>File → Open Project</b> to load<br>"
                "</div>"
            )
            self._update_labeling_progress_indicator()
            return

        labeled_count = sum(1 for label in self.image_labels if label)
        total_count = len(self.image_paths)
        unlabeled_count = max(0, total_count - labeled_count)
        n_clusters = (
            len(set(self.cluster_assignments))
            if self.cluster_assignments is not None
            else 0
        )

        info_html = "<div style='line-height: 1.7;'>"
        info_html += f"<b style='color: #ffffff; font-size: 15px;'>{self.project_path.name}</b><br>"
        info_html += f"<span style='color: #888888; font-size: 11px;'>{self.project_path}</span><br><br>"
        info_html += f"<b>Database:</b> {'Connected' if self.db_path and self.db_path.exists() else 'Not ready'}<br>"
        info_html += f"<b>Total Images:</b> {total_count:,}<br>"
        info_html += f"<b>Labeled:</b> {labeled_count:,}<br>"
        info_html += f"<b>Unlabeled:</b> {unlabeled_count:,}<br>"
        info_html += f"<b>Classes:</b> {len(self.classes)} ({', '.join(self.classes[:5])}{'...' if len(self.classes) > 5 else ''})<br>"
        info_html += f"<b>Embeddings:</b> {'ready' if self.embeddings is not None else 'not computed'}<br>"
        info_html += (
            f"<b>Clusters:</b> {n_clusters if n_clusters else 'not clustered'}<br>"
        )
        info_html += f"<b>UMAP:</b> {'ready' if self.umap_coords is not None else 'not computed'}<br>"
        info_html += (
            f"<b>Current Mode:</b> {self._mode_display_name(self.explorer_mode)}<br>"
        )
        info_html += (
            f"<b>Candidate Set:</b> {len(self.candidate_indices)} points"
            if self.candidate_indices
            else "<b>Candidate Set:</b> none sampled"
        )
        info_html += "</div>"

        # ── Next Step guidance ──────────────────────────────────────────
        next_step = ""
        if total_count == 0:
            next_step = "• <b>File → Source Manager</b>"
        elif self.embeddings is None:
            next_step = "• <b>Actions → Compute Embeddings</b>"
        elif self.cluster_assignments is None:
            next_step = "• <b>Actions → Cluster Embeddings</b>"
        elif self.umap_coords is None:
            next_step = "• <b>Actions → Compute UMAP</b>"
        elif labeled_count < 10:
            next_step = "• <b>Labeling → Sample Randomly</b>"
        elif len(self.candidate_indices) == 0:
            next_step = "• <b>Labeling → Sample next candidates</b>"
        else:
            next_step = "• <b>Labeling → Enter Labeling Mode (L)</b>"

        info_html += "<div style='margin-top:16px; padding:10px; background:#2d2d2d; border-radius:6px; border-left:3px solid #0e639c;'>"
        info_html += "<b style='color:#0e639c; font-size:11px; text-transform:uppercase;'>Suggested Next Step:</b><br>"
        info_html += f"<span style='color:#ffffff; font-size:12px;'>{next_step}</span>"
        info_html += "</div>"

        self.context_info.setText(info_html)
        self._update_labeling_progress_indicator()

    def _update_labeling_progress_indicator(self):
        """Update the status bar progress label and context panel progress bar."""
        in_labeling = self.explorer_mode == "labeling" and bool(
            self.candidate_indices or self.round_labeled_indices
        )
        if not in_labeling:
            self.labeling_progress_label.setVisible(False)
            self.progress_bar.setVisible(False)
            return

        # Batch progress: how many candidates in current set have been labeled
        candidate_set = set(self.candidate_indices) | set(self.round_labeled_indices)
        batch_total = len(candidate_set)
        labels = self.image_labels or []
        batch_labeled = sum(
            1 for i in self.round_labeled_indices if i < len(labels) and labels[i]
        )

        # Overall progress across entire dataset
        total_count = len(self.image_paths)
        total_labeled = sum(1 for lbl in labels if lbl)

        # Status bar label
        remaining = batch_total - batch_labeled
        self.labeling_progress_label.setText(
            f"Batch: {batch_labeled}/{batch_total}  ·  "
            f"Total: {total_labeled:,}/{total_count:,}  ·  "
            f"{remaining} remaining"
        )
        self.labeling_progress_label.setVisible(True)

        # Context panel progress bar
        self.progress_bar.setMaximum(batch_total)
        self.progress_bar.setValue(batch_labeled)
        pct = int(100 * batch_labeled / batch_total) if batch_total else 0
        self.progress_bar.setFormat(f"Batch: {batch_labeled}/{batch_total} ({pct}%)")
        self.progress_bar.setVisible(True)

    def _ask_yes_no(self, title: str, text: str) -> bool:
        """Reusable yes/no prompt helper."""
        reply = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _tick_autosave_heartbeat(self):
        """Animate autosave heartbeat indicator in the status bar."""
        self._autosave_heartbeat_on = not self._autosave_heartbeat_on
        self._update_autosave_heartbeat_text()

    def _update_autosave_heartbeat_text(self):
        """Render autosave status in bottom-right status bar indicator."""
        indicator = "●" if self._autosave_heartbeat_on else "○"
        pending = len(self._pending_label_updates)
        period_sec = max(1, self._autosave_interval_ms // 1000)
        if not self.project_path:
            text = f"Autosave {indicator} no project"
        elif pending > 0:
            text = f"Autosave {indicator} {period_sec}s pending {pending}"
        elif self._autosave_last_save_time is not None:
            last_saved = time.strftime(
                "%H:%M:%S", time.localtime(self._autosave_last_save_time)
            )
            text = f"Autosave {indicator} {period_sec}s saved {last_saved}"
        else:
            text = f"Autosave {indicator} {period_sec}s idle"
        self.autosave_indicator.setText(text)

    def _save_project_runtime_settings(self):
        """Persist runtime settings such as autosave interval into project config."""
        if not self.project_path:
            return
        project_config_path = self.project_path / "project.json"
        config = {}
        if project_config_path.exists():
            with open(project_config_path, "r") as f:
                config = json.load(f)
        config["autosave_interval_ms"] = int(self._autosave_interval_ms)
        with open(project_config_path, "w") as f:
            json.dump(config, f, indent=2)

    def on_autosave_interval_changed(self, seconds: int):
        """Handle autosave interval changes from the UI."""
        self._autosave_interval_ms = max(1000, int(seconds) * 1000)
        self._autosave_timer.setInterval(self._autosave_interval_ms)
        self._save_project_runtime_settings()
        self._update_autosave_heartbeat_text()
        self.status.showMessage(f"Autosave interval set to {seconds}s")

    def _flush_pending_label_updates(self, force: bool = False):
        """Persist buffered label updates to DB on autosave cadence or forced flush."""
        if not self._pending_label_updates:
            self._update_autosave_heartbeat_text()
            return
        if not self.db_path:
            if force:
                self._pending_label_updates = {}
            self._update_autosave_heartbeat_text()
            return

        try:
            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            # Create a local copy to avoid mutation during iteration
            updates = dict(self._pending_label_updates)
            count = len(updates)

            updated_count = db.update_labels_batch(updates)

            if updated_count < count:
                import logging

                logging.getLogger(__name__).warning(
                    f"Autosave: only {updated_count} of {count} labels were updated in DB. "
                    "This usually means some image paths in memory don't match the DB exactly."
                )

            # Only clear the ones we just successfully updated
            for path in updates:
                self._pending_label_updates.pop(path, None)

            self._autosave_last_save_time = time.time()
            if self._autosave_timer.isActive():
                self._autosave_timer.stop()

            self.status.showMessage(f"Autosaved {updated_count} labels", 2000)
        except Exception as exc:
            self.status.showMessage(f"Autosave failed: {exc}")
            import logging

            logging.getLogger(__name__).error(f"Autosave failure: {exc}", exc_info=True)

        self._update_autosave_heartbeat_text()

    def try_autoload_cached_artifacts(self, db=None):
        """Offer to autoload latest embeddings, cluster assignments, and UMAP."""
        if not self.db_path:
            return

        if db is None:
            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

        try:
            cached_embeddings = db.get_most_recent_embeddings()
            if cached_embeddings is not None and self.embeddings is None:
                embeddings, metadata = cached_embeddings
                if self._ask_yes_no(
                    "Load Cached Embeddings",
                    "Most recent embeddings cache found. Load it now?\n\n"
                    f"Model: {metadata.get('model_name', 'unknown')}\n"
                    f"Timestamp: {metadata.get('timestamp', 'unknown')}\n"
                    f"Shape: {embeddings.shape[0]:,} × {embeddings.shape[1]}",
                ):
                    self.embeddings = embeddings

            cached_cluster = db.get_most_recent_cluster_cache()
            if cached_cluster is not None and self.cluster_assignments is None:
                if self._ask_yes_no(
                    "Load Cached Clusters",
                    "Most recent cluster cache found. Load it now?\n\n"
                    f"Method: {cached_cluster.get('method', 'unknown')}\n"
                    f"Timestamp: {cached_cluster.get('timestamp', 'unknown')}\n"
                    f"Clusters: {cached_cluster.get('n_clusters', 'unknown')}",
                ):
                    self.cluster_assignments = cached_cluster["assignments"]

            cached_umap = db.get_most_recent_umap_cache()
            if cached_umap is not None and self.umap_coords is None:
                if self._ask_yes_no(
                    "Load Cached UMAP",
                    "Most recent UMAP cache found. Load it now?\n\n"
                    f"Timestamp: {cached_umap.get('timestamp', 'unknown')}\n"
                    f"n_neighbors: {cached_umap.get('n_neighbors', 'unknown')}\n"
                    f"min_dist: {cached_umap.get('min_dist', 'unknown')}",
                ):
                    self.umap_coords = cached_umap["coords"]
                    self.last_umap_params = {
                        "n_neighbors": cached_umap.get("n_neighbors", 15),
                        "min_dist": cached_umap.get("min_dist", 0.1),
                    }

            cached_candidates = db.get_most_recent_candidate_cache()
            if cached_candidates is not None and not self.candidate_indices:
                if self._ask_yes_no(
                    "Load Cached Candidates",
                    "Most recent labeling candidate set found. Load it now?\n\n"
                    f"Timestamp: {cached_candidates.get('timestamp', 'unknown')}\n"
                    f"Count: {len(cached_candidates['candidate_indices'])} images",
                ):
                    self.candidate_indices = cached_candidates["candidate_indices"]
                    self.set_explorer_mode("labeling")
                    self.status.showMessage(
                        f"Loaded {len(self.candidate_indices)} candidates"
                    )

            # Auto-offer to load the published YOLO classifier
            if self.project_path:
                yolo_ckpt = self.project_path / "models" / "yolo_classifier_latest.pt"
                if (
                    yolo_ckpt.exists()
                    and self._yolo_model_path is None
                    and self._model_probs is None
                ):
                    # First try loading cached predictions so we don't rerun inference
                    cached_preds = db.get_most_recent_prediction_cache()
                    if cached_preds and self._ask_yes_no(
                        "Load Cached Predictions",
                        f"Cached predictions found from last session.\n"
                        f"Mode: {cached_preds.get('active_model_mode', '?')}  |  "
                        f"Classes: {len(cached_preds.get('class_names', []))}\n"
                        f"Saved: {str(cached_preds.get('timestamp', ''))[:19]}\n\n"
                        "Load predictions without re-running inference?",
                    ):
                        self._model_probs = cached_preds["probs"]
                        self._model_class_names = cached_preds["class_names"]
                        self._active_model_mode = cached_preds.get(
                            "active_model_mode", "yolo"
                        )
                        self.image_confidences = list(
                            self._model_probs.max(axis=1).astype(float)
                        )
                        self._set_model_projection_buttons_enabled(True)
                        self._update_al_status()
                        # Also restore model-space UMAP if available
                        cached_mumap = db.get_most_recent_umap_cache(kind="model")
                        if cached_mumap:
                            self.umap_model_coords = cached_mumap["coords"]
                        cached_mpca = db.get_most_recent_umap_cache(kind="model_pca")
                        if cached_mpca:
                            self.pca_model_coords = cached_mpca["coords"]
                    elif self._ask_yes_no(
                        "Load Trained YOLO Model",
                        f"A published YOLO classifier was found.\n"
                        f"Load it and run inference?\n\n{yolo_ckpt}",
                    ):
                        self._yolo_model_path = yolo_ckpt
                        self._active_model_mode = "yolo"
                        QTimer.singleShot(
                            200, lambda p=yolo_ckpt: self._run_yolo_inference(p)
                        )
                        return  # skip DB check after filesystem hit

            # Auto-offer to load the most recent model from the project DB
            if self.db_path and self._model_probs is None:
                try:
                    from ..store.db import ClassKitDB as _CKDb

                    _db2 = _CKDb(self.db_path)
                    # Try loading cached predictions first
                    cached_preds = _db2.get_most_recent_prediction_cache()
                    if cached_preds and self._ask_yes_no(
                        "Load Cached Predictions",
                        f"Cached predictions found from last session.\n"
                        f"Mode: {cached_preds.get('active_model_mode', '?')}  |  "
                        f"Classes: {len(cached_preds.get('class_names', []))}\n"
                        f"Saved: {str(cached_preds.get('timestamp', ''))[:19]}\n\n"
                        "Load predictions without re-running inference?",
                    ):
                        self._model_probs = cached_preds["probs"]
                        self._model_class_names = cached_preds["class_names"]
                        self._active_model_mode = cached_preds.get(
                            "active_model_mode", ""
                        )
                        self.image_confidences = list(
                            self._model_probs.max(axis=1).astype(float)
                        )
                        self._set_model_projection_buttons_enabled(True)
                        self._update_al_status()
                        cached_mumap = _db2.get_most_recent_umap_cache(kind="model")
                        if cached_mumap:
                            self.umap_model_coords = cached_mumap["coords"]
                        cached_mpca = _db2.get_most_recent_umap_cache(kind="model_pca")
                        if cached_mpca:
                            self.pca_model_coords = cached_mpca["coords"]
                    else:
                        _recent = _db2.get_most_recent_model_cache()
                        if _recent and self._ask_yes_no(
                            "Load Trained Model",
                            f"A trained model was found in the project records.\n"
                            f"Mode: {_recent.get('mode', '?')}  |  "
                            f"Classes: {', '.join(_recent.get('class_names') or [])}\n"
                            f"Trained: {str(_recent.get('timestamp', ''))[:19]}\n\n"
                            "Load it and run inference?",
                        ):
                            QTimer.singleShot(
                                200,
                                lambda e=_recent: self._load_model_from_cache_entry(e),
                            )
                except Exception:
                    pass

            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()
        except Exception:
            pass

    def setup_label_shortcuts(self):
        """Create keyboard shortcuts for labeling and mode switching."""
        for shortcut in self._label_shortcuts:
            shortcut.setParent(None)
        self._label_shortcuts = []

        from .dialogs import ShortcutEditorDialog

        defaults = dict(ShortcutEditorDialog.DEFAULT_SHORTCUTS)
        active = {**defaults, **self._custom_shortcuts}

        def _key(action: str) -> QKeySequence:
            return QKeySequence(active.get(action, defaults.get(action, "")))

        # Load scheme-specific shortcuts if available
        scheme_shortcuts = {}
        if self.project_path:
            try:
                scheme_path = self.project_path / "scheme.json"
                if scheme_path.exists():
                    with open(scheme_path) as _f:
                        scheme_dict = json.load(_f)
                        factors = scheme_dict.get("factors", [])
                        if factors:
                            # Use shortcuts from the first factor if it's a flat/single-factor scheme
                            f = factors[0]
                            labels = f.get("labels", [])
                            keys = f.get("shortcut_keys", [])
                            for lbl, k in zip(labels, keys):
                                if k:
                                    scheme_shortcuts[lbl] = k
            except Exception:
                pass

        # Only install label shortcuts when NOT in stepper (multi-factor) mode.
        # In stepper mode, key routing goes through the stepper's handle_key().
        if self._stepper is None:
            # 1) Try standard 1-9 fallback
            for i, class_name in enumerate(self.classes[:9], start=1):
                # Only use digit fallback if no explicit scheme shortcut exists for this class
                if class_name not in scheme_shortcuts:
                    shortcut = QShortcut(QKeySequence(str(i)), self)
                    shortcut.setAutoRepeat(False)
                    shortcut.activated.connect(
                        lambda c=class_name: self.assign_label_to_selected(c)
                    )
                    self._label_shortcuts.append(shortcut)

            # 2) Apply explicit scheme shortcuts (e.g. 'A', 'W', 'S', 'D' for head/tail)
            for lbl, k in scheme_shortcuts.items():
                shortcut = QShortcut(QKeySequence(k), self)
                shortcut.setAutoRepeat(False)
                shortcut.activated.connect(
                    lambda c=lbl: self.assign_label_to_selected(c)
                )
                self._label_shortcuts.append(shortcut)

            # Always register '0' for unknown
            unknown_key = QShortcut(QKeySequence("0"), self)
            unknown_key.setAutoRepeat(False)
            unknown_key.activated.connect(
                lambda: self.assign_label_to_selected("unknown")
            )
            self._label_shortcuts.append(unknown_key)

        explore_shortcut = QShortcut(_key("Explore mode"), self)
        explore_shortcut.setAutoRepeat(False)
        explore_shortcut.activated.connect(lambda: self.set_explorer_mode("explore"))
        self._label_shortcuts.append(explore_shortcut)

        label_shortcut = QShortcut(_key("Labeling mode"), self)
        label_shortcut.setAutoRepeat(False)
        label_shortcut.activated.connect(lambda: self.set_explorer_mode("labeling"))
        self._label_shortcuts.append(label_shortcut)

        prediction_shortcut = QShortcut(_key("Predictions mode"), self)
        prediction_shortcut.setAutoRepeat(False)
        prediction_shortcut.activated.connect(
            lambda: self.set_explorer_mode("predictions")
        )
        self._label_shortcuts.append(prediction_shortcut)

        sample_shortcut = QShortcut(_key("Sample next candidates"), self)
        sample_shortcut.setAutoRepeat(False)
        sample_shortcut.activated.connect(self.on_sample_next_triggered)
        self._label_shortcuts.append(sample_shortcut)

        prev_shortcut = QShortcut(_key("Previous unlabeled"), self)
        prev_shortcut.setAutoRepeat(False)
        prev_shortcut.activated.connect(self.on_prev_image)
        self._label_shortcuts.append(prev_shortcut)

        next_shortcut = QShortcut(_key("Next unlabeled"), self)
        next_shortcut.setAutoRepeat(False)
        next_shortcut.activated.connect(self.on_next_image)
        self._label_shortcuts.append(next_shortcut)

        undo_shortcut = QShortcut(_key("Undo last label (Ctrl+Z)"), self)
        undo_shortcut.setAutoRepeat(False)
        undo_shortcut.activated.connect(self.undo_last_assignment)
        self._label_shortcuts.append(undo_shortcut)

    def _begin_command(self) -> bool:
        """Return False when command input should be ignored during active/cooldown windows."""
        now = time.monotonic()
        if self._command_busy:
            self._command_squelch_pending = True
            return False
        if now < self._command_block_until:
            self._command_squelch_pending = True
            return False
        self._command_busy = True
        self._set_command_controls_enabled(False)
        return True

    def _end_command(self, cooldown_seconds: float = 0.10):
        """Release command processing and apply short cooldown to drop queued key events."""
        self._command_busy = False
        now = time.monotonic()
        hold = cooldown_seconds
        if self._command_squelch_pending:
            hold = max(hold, 0.35)
            self._command_squelch_pending = False

        self._command_block_until = max(self._command_block_until, now + hold)
        QTimer.singleShot(int(hold * 1000), self._restore_command_controls)

    def _restore_command_controls(self):
        """Re-enable interactive command controls after cooldown expires."""
        if self._command_busy:
            return
        now = time.monotonic()
        if now < self._command_block_until:
            remaining_ms = max(10, int((self._command_block_until - now) * 1000))
            QTimer.singleShot(remaining_ms, self._restore_command_controls)
            return
        self._set_command_controls_enabled(True)

    # ── Background-job UI locking ─────────────────────────────────────────

    def _set_ui_busy(self, busy: bool) -> None:
        """Disable/re-enable toolbar and menu bar while a background job is running."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        toolbar = getattr(self, "toolbar", None)
        if toolbar is not None:
            toolbar.setEnabled(not busy)
        menubar = self.menuBar()
        if menubar is not None:
            menubar.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _job_start(self) -> None:
        """Increment active-job counter and lock the UI on the first job."""
        self._active_jobs += 1
        if self._active_jobs == 1:
            self._set_ui_busy(True)

    def _job_done(self) -> None:
        """Decrement active-job counter and unlock the UI when all jobs finish."""
        self._active_jobs = max(0, self._active_jobs - 1)
        if self._active_jobs == 0:
            self._set_ui_busy(False)

    def _threadpool_start(self, worker) -> None:
        """Register a job, connect the cleanup hook, and submit to the thread pool."""
        self._job_start()
        worker.signals.finished.connect(self._job_done)
        self.threadpool.start(worker)

    def _set_command_controls_enabled(self, enabled: bool):
        """Enable/disable controls and shortcuts that can generate rapid command bursts."""
        for attr in ("sample_btn", "prev_btn", "next_btn"):
            button = getattr(self, attr, None)
            if button is not None:
                button.setEnabled(enabled)

        labels_enabled = enabled and self.explorer_mode == "labeling"
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setEnabled(labels_enabled)

        for shortcut in self._label_shortcuts:
            shortcut.setEnabled(enabled)

    def request_update_explorer_plot(self, force_fit: bool = False):
        """Coalesce expensive UMAP redraws into a single latest update."""
        self._plot_refresh_pending = True
        self._plot_refresh_force_fit = self._plot_refresh_force_fit or force_fit
        if not self._plot_refresh_timer.isActive():
            self._plot_refresh_timer.start()

    def _flush_explorer_update(self):
        """Apply the latest queued explorer redraw request."""
        if not self._plot_refresh_pending:
            return
        force_fit = self._plot_refresh_force_fit
        self._plot_refresh_pending = False
        self._plot_refresh_force_fit = False
        self.update_explorer_plot(force_fit=force_fit)

    def request_preview_for_index(self, index: int, source: str = "hover"):
        """Coalesce preview updates so rapid input keeps only the latest target."""
        self._pending_preview_index = index
        self._pending_preview_source = source
        if not self._preview_refresh_timer.isActive():
            self._preview_refresh_timer.start()

    def request_update_context_panel(self):
        """Coalesce context panel updates to avoid frequent expensive HTML rebuilds."""
        self._context_refresh_pending = True
        if not self._context_refresh_timer.isActive():
            self._context_refresh_timer.start()

    def _flush_context_update(self):
        """Apply latest queued context panel update request."""
        if not self._context_refresh_pending:
            return
        self._context_refresh_pending = False
        self.update_context_panel()

    def _flush_preview_update(self):
        """Apply the latest queued preview update request."""
        index = self._pending_preview_index
        source = self._pending_preview_source
        self._pending_preview_index = None
        if index is None:
            return
        self.load_preview_for_index(index, source=source)

    def request_update_explorer_selection(self, selected_index: int | None):
        """Fast-path selection-only update; falls back to full redraw when needed."""
        if not hasattr(self, "explorer"):
            return
        if self.explorer.set_selected_index(selected_index):
            return
        self.request_update_explorer_plot()

    def refresh_label_history_strip(self):
        """Refresh bottom strip of recently labeled items for quick undo/relabel."""
        recent = list(reversed(self.label_history[-24:]))
        allow_thumbnail_load = (not self._command_busy) and (
            time.monotonic() >= self._command_block_until
        )
        loads_remaining = self._history_thumb_load_budget
        pending_uncached = False

        for slot_idx, slot in enumerate(self._history_slots):
            card, thumb, caption = slot
            if slot_idx >= len(recent):
                card.setProperty("history_index", None)
                card.setToolTip("")
                thumb.clear()
                caption.clear()
                card.hide()
                continue

            entry = recent[slot_idx]
            index = entry["index"]
            label = entry["label"]
            image_path = (
                self.image_paths[index] if index < len(self.image_paths) else None
            )

            card.setProperty("history_index", int(index))
            card.setToolTip(f"Click to undo label and relabel: {label}")
            caption.setText(f"{label}\n#{index}")
            thumb.clear()

            if image_path and image_path.exists():
                icon_key = str(image_path)
                pixmap = self._history_icon_cache.get(icon_key)
                if pixmap is None and allow_thumbnail_load and loads_remaining > 0:
                    pixmap = QPixmap(str(image_path))
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(
                            72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
                        )
                        self._history_icon_cache[icon_key] = pixmap
                    loads_remaining -= 1
                elif pixmap is None:
                    pending_uncached = True
                if pixmap is not None and not pixmap.isNull():
                    thumb.setPixmap(pixmap)

            card.show()

        if pending_uncached and not self._history_refresh_timer.isActive():
            self._history_refresh_timer.start(80)

    def request_refresh_label_history_strip(self):
        """Coalesce strip rebuilds to avoid repeated widget churn during rapid labeling."""
        self._history_refresh_timer.start()

    def keyPressEvent(self, event):
        """Route digit keys to stepper when in multi-factor labeling mode."""
        if self._stepper is not None:
            key_text = event.text()
            if key_text and self._stepper.handle_key(key_text):
                event.accept()
                return
        super().keyPressEvent(event)

    def eventFilter(self, watched, event):
        """Handle clicks on history cards without QPushButton construction."""
        if event.type() == QEvent.MouseButtonRelease:
            index = watched.property("history_index") if watched is not None else None
            if index is not None:
                self.undo_label_from_history(int(index))
                return True
        return super().eventFilter(watched, event)

    def _remove_history_for_index(self, index: int):
        """Remove history entries for a point index."""
        self.label_history = [
            entry for entry in self.label_history if entry.get("index") != index
        ]

    def undo_label_from_history(self, index: int):
        """Undo a specific label from history and move selection back for relabel."""
        if not self._begin_command():
            return
        try:
            if index < 0 or index >= len(self.image_paths):
                return
            previous = (
                self.image_labels[index] if index < len(self.image_labels) else None
            )
            if not previous:
                return

            self._set_label_for_index(index, None)
            self.selected_point_index = index
            if index not in self.candidate_indices:
                self.candidate_indices.insert(0, index)
            self.round_labeled_indices = [
                i for i in self.round_labeled_indices if i != index
            ]
            self._remove_history_for_index(index)
            self.request_refresh_label_history_strip()
            self.hover_locked = True
            self.request_preview_for_index(index, source="undo")
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.status.showMessage(
                f"Reverted label on point {index}; ready to relabel"
            )
        finally:
            self._end_command(0.12)

    def undo_last_assignment(self):
        """Undo the most recent assignment (Ctrl+Z)."""
        if not self._begin_command():
            return
        try:
            if not self.last_assigned_stack:
                self.status.showMessage("No recent assignment to undo")
                return

            last = self.last_assigned_stack.pop()
            index = last["index"]
            previous_label = last.get("previous_label")
            self._set_label_for_index(index, previous_label)

            if previous_label:
                if index not in self.round_labeled_indices:
                    self.round_labeled_indices.append(index)
            else:
                self.round_labeled_indices = [
                    i for i in self.round_labeled_indices if i != index
                ]
                if index not in self.candidate_indices:
                    self.candidate_indices.insert(0, index)

            if previous_label is None:
                self._remove_history_for_index(index)

            self.request_refresh_label_history_strip()

            self.selected_point_index = index
            self.hover_locked = True
            self.request_preview_for_index(index, source="undo")
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.status.showMessage(f"Undid assignment for point {index}")
        finally:
            self._end_command(0.12)

    def _on_stepper_label_committed(self, composite_label: str) -> None:
        """Called when the stepper completes all factors for the current image."""
        if self.selected_point_index is None:
            return
        index = self.selected_point_index
        self._set_label_for_index(index, composite_label)
        self.round_labeled_indices = [
            i for i in self.round_labeled_indices if i != index
        ]
        self.round_labeled_indices.append(index)
        self._remove_history_for_index(index)
        self.label_history.append({"index": index, "label": composite_label})
        self.request_refresh_label_history_strip()
        self.on_label_assigned(composite_label)
        self.request_update_explorer_plot()
        self.request_update_context_panel()
        # Flush immediately after each composite label is finished to ensure data safety
        self._flush_pending_label_updates()
        self.on_next_image()

    def _set_label_for_index(self, index: int, label):
        """Set or clear label for an index and persist to DB."""
        if index < 0 or index >= len(self.image_paths):
            return

        if self.db_path:
            # Key must match the file_path string stored in DB exactly
            self._pending_label_updates[str(self.image_paths[index])] = label
            if not self._autosave_timer.isActive():
                self._autosave_timer.start()
            self._update_autosave_heartbeat_text()

        if not self.image_labels or len(self.image_labels) != len(self.image_paths):
            self.image_labels = [None] * len(self.image_paths)
        self.image_labels[index] = label

    def on_explorer_background_double_click(self):
        """Return to hover mode when empty region is double-clicked."""
        self.selected_point_index = None
        self.hover_locked = False
        self.update_knn_panel(None)
        self.request_update_explorer_plot()
        self.selection_info.setText(
            "<div style='line-height:1.5;'>"
            "<b>Selected Point:</b> none<br>"
            "<b>Hovered Point:</b> none<br>"
            "<b>Current Label:</b> unlabeled<br>"
            "Selection cleared. Hover candidates to preview; click one to select for labeling."
            "</div>"
        )
        self.status.showMessage("Hover mode restored")

    def _get_navigation_pool(self):
        """Return indices eligible for next/prev navigation in labeling context."""
        if self.candidate_indices:
            return self.candidate_indices
        return list(range(len(self.image_paths)))

    def rebuild_label_buttons(self):
        """Rebuild class buttons shown in the left settings panel."""
        # Clear existing widgets from the grid layout
        for i in reversed(range(self.label_buttons_layout.count())):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Determine if project uses a multi-factor labeling scheme
        scheme = None
        scheme_shortcuts = {}
        if self.project_path:
            try:
                from ..config.schemas import LabelingScheme

                scheme_path = self.project_path / "scheme.json"
                if scheme_path.exists():
                    with open(scheme_path) as _f:
                        scheme = LabelingScheme.from_dict(json.load(_f))
                        factors = scheme.factors
                        if factors:
                            f = factors[0]
                            for lbl, k in zip(f.labels, f.shortcut_keys):
                                if k:
                                    scheme_shortcuts[lbl] = k
            except Exception:
                scheme = None

        if scheme is not None and len(scheme.factors) > 1:
            # Multi-factor: use FactorStepperWidget
            try:
                from .widgets.factor_stepper import _build_qt_widget

                FactorStepperWidget = _build_qt_widget(scheme)
                stepper = FactorStepperWidget(
                    scheme, parent=self.label_buttons_container
                )
                stepper.label_committed.connect(self._on_stepper_label_committed)
                stepper.skipped.connect(self.on_next_image)
                self.label_buttons_layout.addWidget(stepper, 0, 0, 1, 2)
                self._stepper = stepper
            except Exception:
                self._stepper = None
        else:
            # Single-factor or free-form: flat buttons
            self._stepper = None
            class_color_map = build_category_color_map([*self.classes, "unknown"])

            # Use all project classes, not just first 9
            for i, class_name in enumerate(self.classes):
                # Determine display shortcut
                shortcut = scheme_shortcuts.get(class_name)
                if not shortcut and i < 9:
                    shortcut = str(i + 1)

                btn_text = f"[{shortcut}] {class_name}" if shortcut else class_name
                button = QPushButton(btn_text)
                bg = class_color_map.get(class_name)
                if bg is None:
                    bg = class_color_map.get(str(class_name), None)
                if bg is None:
                    bg = class_color_map.get("unknown")
                fg = best_text_color(bg)
                button.setStyleSheet(
                    "text-align: left; padding: 4px; "
                    f"background-color: {to_hex(bg)}; color: {to_hex(fg)};"
                    "border: 1px solid #2f2f2f;"
                )
                button.clicked.connect(
                    lambda checked=False, c=class_name: self.assign_label_to_selected(c)
                )
                # Store class name for filtering
                button.setProperty("class_name", class_name)

                row = i // 2
                col = i % 2
                self.label_buttons_layout.addWidget(button, row, col)

            # Always add 'unknown' button at the end
            unknown_shortcut = "0"
            unknown_btn = QPushButton(f"[{unknown_shortcut}] unknown")
            unknown_bg = class_color_map.get("unknown")
            unknown_fg = best_text_color(unknown_bg)
            unknown_btn.setStyleSheet(
                "text-align: left; padding: 4px; "
                f"background-color: {to_hex(unknown_bg)}; color: {to_hex(unknown_fg)};"
                "border: 1px solid #2f2f2f;"
            )
            unknown_btn.clicked.connect(
                lambda checked=False: self.assign_label_to_selected("unknown")
            )
            unknown_btn.setProperty("class_name", "unknown")

            # Add to next available position
            total_btns = len(self.classes)
            self.label_buttons_layout.addWidget(
                unknown_btn, total_btns // 2, total_btns % 2
            )

            # Initial filter update
            if hasattr(self, "class_search"):
                self.filter_label_buttons(self.class_search.text())

    def filter_label_buttons(self, text: str):
        """Filter visible label buttons based on search text."""
        text = text.lower()
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget and hasattr(widget, "property"):
                class_name = widget.property("class_name")
                if class_name:
                    widget.setVisible(text in class_name.lower())

    @staticmethod
    def _mode_display_name(mode: str) -> str:
        return {
            "explore": "Explore",
            "labeling": "Labeling",
            "predictions": "Predictions",
        }.get(str(mode), str(mode))

    def _prediction_labels_for_plot(self) -> list:
        """Return per-image predicted class labels for prediction-colored explorer mode."""
        if self._model_probs is None:
            return [None] * len(self.image_paths)

        probs = np.asarray(self._model_probs)
        if probs.ndim != 2:
            return [None] * len(self.image_paths)

        num_images = len(self.image_paths)
        out = [None] * num_images
        eval_n = min(num_images, probs.shape[0])

        names = list(self._model_class_names or [])
        for i in range(eval_n):
            pred_idx = int(np.argmax(probs[i]))
            if 0 <= pred_idx < len(names):
                out[i] = names[pred_idx]
            else:
                out[i] = f"pred_{pred_idx}"
        return out

    def set_explorer_mode(self, mode: str):
        """Set explorer mode to cluster, labeling, or prediction coloring."""
        if mode not in {"explore", "labeling", "predictions"}:
            return

        if mode == "predictions" and self._model_probs is None:
            QMessageBox.information(
                self,
                "Predictions Unavailable",
                "Load a checkpoint first. Predictions and model-space UMAP are computed automatically on load.",
            )
            return

        self.explorer_mode = mode

        # Label assignment is only allowed in labeling mode.
        labels_enabled = mode == "labeling"
        for i in range(self.label_buttons_layout.count()):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setEnabled(labels_enabled)

        if hasattr(self, "view_mode_combo"):
            idx = self.view_mode_combo.findData(mode)
            if idx >= 0 and self.view_mode_combo.currentIndex() != idx:
                self.view_mode_combo.blockSignals(True)
                self.view_mode_combo.setCurrentIndex(idx)
                self.view_mode_combo.blockSignals(False)

        self.request_update_explorer_plot()
        self.update_knn_panel(self.selected_point_index)
        self.update_context_panel()
        self.status.showMessage(f"Mode: {self._mode_display_name(mode)}")

    def on_view_mode_changed(self, index):
        """Handle mode selection from combo box."""
        mode = self.view_mode_combo.itemData(index)
        self.set_explorer_mode(mode)

    def on_outline_threshold_changed(self, value: float) -> None:
        """Apply uncertainty outline threshold used by Explorer point styling."""
        self._outline_threshold = float(value)
        if hasattr(self, "explorer") and self.explorer is not None:
            self.explorer.set_uncertainty_outline_threshold(self._outline_threshold)
            self.request_update_explorer_plot()
        self.status.showMessage(
            f"Uncertainty outline threshold: {self._outline_threshold:.2f}"
        )

    def on_sample_next_triggered(self):
        """Action for 'Sample Next' button."""
        self.sample_candidates_for_labeling()

    def on_clear_candidates_triggered(self):
        """Clear current candidate set and return to explore mode."""
        self.candidate_indices = []
        self.round_labeled_indices = []

        if self.db_path:
            try:
                from ..store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                db.save_candidate_cache([])  # Save empty list to effectively clear
            except Exception:
                pass

        self.set_explorer_mode("explore")
        self.status.showMessage("Candidates cleared")

    def sample_candidates_for_labeling(self):
        """Sample diverse unlabeled points from each cluster for fast labeling."""
        if not self._begin_command():
            return
        try:
            if self.umap_coords is None or self.cluster_assignments is None:
                QMessageBox.information(
                    self,
                    "Need UMAP + Clusters",
                    "Compute clustering and UMAP first, then sample candidate sets.",
                )
                return

            if not self.image_labels or len(self.image_labels) != len(self.image_paths):
                self.image_labels = [None] * len(self.image_paths)

            per_cluster = self.sample_spin.value()
            assignments = list(self.cluster_assignments)
            labels = self.image_labels

            sampled = []
            for cluster_id in sorted(set(assignments)):
                indices = [
                    idx
                    for idx, value in enumerate(assignments)
                    if value == cluster_id and (idx < len(labels) and not labels[idx])
                ]
                sampled.extend(indices[:per_cluster])

            # Preserve already-labeled candidates from the current set so they
            # remain visible after re-sampling (don't hide work already done).
            seen = set(sampled)
            for i in self.candidate_indices:
                if i not in seen and i < len(labels) and labels[i]:
                    sampled.append(i)
                    seen.add(i)

            self.candidate_indices = sampled
            # Do NOT clear round_labeled_indices — keep current-session progress.

            if self.db_path:
                try:
                    from ..store.db import ClassKitDB

                    db = ClassKitDB(self.db_path)
                    db.save_candidate_cache(self.candidate_indices)
                except Exception:
                    pass

            if self.candidate_indices:
                self.set_explorer_mode("labeling")
                self.selected_point_index = None
                self.hover_locked = False
                self.selection_info.setText(
                    "<div style='line-height:1.5;'>"
                    "<b>Selected Point:</b> none<br>"
                    "<b>Hovered Point:</b> none<br>"
                    "<b>Current Label:</b> unlabeled<br>"
                    "Hover over candidate points to preview, then click to select one for labeling."
                    "</div>"
                )
                self.status.showMessage(
                    f"Sampled {len(self.candidate_indices):,} unlabeled candidates across {len(set(assignments))} clusters"
                )
            else:
                self.status.showMessage("No unlabeled points left in current clusters")

            self.request_update_explorer_plot()
            self.request_update_context_panel()
        finally:
            self._end_command(0.15)

    def load_preview_for_index(self, index: int, source: str = "hover"):
        """Load preview image and metadata for a UMAP point index."""
        if index is None or index < 0 or index >= len(self.image_paths):
            return

        if source == "hover" and self.last_preview_index == index:
            return

        image_path = self.image_paths[index]

        self.preview_canvas.set_image(str(image_path))

        self.last_preview_index = index
        current_label = (
            self.image_labels[index] if index < len(self.image_labels) else None
        )
        cluster_id = None
        if self.cluster_assignments is not None and index < len(
            self.cluster_assignments
        ):
            cluster_id = self.cluster_assignments[index]

        self.preview_info.setText(
            "<div style='line-height:1.55;'>"
            f"<b>Point:</b> {index}<br>"
            f"<b>Cluster:</b> {cluster_id if cluster_id is not None else 'n/a'}<br>"
            f"<b>Label:</b> {current_label if current_label else 'unlabeled'}<br>"
            f"<b>Source:</b> {source}<br>"
            f"<span style='color:#9e9e9e; font-size:11px;'>{image_path.name}</span>"
            "</div>"
        )

        self.selection_info.setText(
            "<div style='line-height:1.5;'>"
            f"<b>Selected Point:</b> {self.selected_point_index if self.selected_point_index is not None else 'none'}<br>"
            f"<b>Hovered Point:</b> {index}<br>"
            f"<b>Current Label:</b> {current_label if current_label else 'unlabeled'}<br>"
            "Assign using number keys 1-9 or class buttons."
            "</div>"
        )

        knn_anchor = self.selected_point_index
        if knn_anchor is None and source in {"click", "next", "prev", "undo", "label"}:
            knn_anchor = index
        self.update_knn_panel(knn_anchor)

    def _compute_knn_neighbors(self, anchor_index: int, k: int = 8):
        """Compute nearest neighbors from embedding space for a selected point."""
        if (
            self.embeddings is None
            or anchor_index is None
            or anchor_index < 0
            or anchor_index >= len(self.image_paths)
        ):
            return []

        query = self.embeddings[anchor_index]
        deltas = self.embeddings - query
        distances = np.sum(deltas * deltas, axis=1)
        ranking = np.argsort(distances)

        neighbors = []
        for candidate in ranking:
            candidate = int(candidate)
            if candidate == anchor_index:
                continue
            label = (
                self.image_labels[candidate]
                if candidate < len(self.image_labels)
                else None
            )
            neighbors.append(
                {
                    "index": candidate,
                    "distance": float(distances[candidate]),
                    "label": label,
                }
            )
            if len(neighbors) >= k:
                break
        return neighbors

    def update_knn_panel(self, anchor_index: int | None):
        """Render nearest-neighbor context for the current selected point."""
        if (
            anchor_index is None
            or self.explorer_mode != "labeling"
            or self.embeddings is None
        ):
            self._current_knn_neighbors = []
            self.knn_info.setHtml(
                "<span style='color:#9f9f9f;'>Select a point in labeling mode to view nearest neighbors.</span>"
            )
            self.knn_jump_btn.setEnabled(False)
            self.knn_bulk_btn.setEnabled(False)
            return

        neighbors = self._compute_knn_neighbors(anchor_index, k=8)
        self._current_knn_neighbors = neighbors

        if not neighbors:
            self.knn_info.setHtml(
                "<span style='color:#9f9f9f;'>No neighbors available.</span>"
            )
            self.knn_jump_btn.setEnabled(False)
            self.knn_bulk_btn.setEnabled(False)
            return

        lines = [
            f"<b>Anchor:</b> #{anchor_index}<br>",
            "<b>Nearest points:</b><br>",
        ]
        unlabeled_neighbors = 0
        for rank, item in enumerate(neighbors, start=1):
            label = item["label"] if item["label"] else "unlabeled"
            if not item["label"]:
                unlabeled_neighbors += 1
            lines.append(
                f"{rank}. #{item['index']} · <i>{label}</i> · d={item['distance']:.4f}<br>"
            )

        self.knn_info.setHtml("".join(lines))
        self.knn_jump_btn.setEnabled(True)

        anchor_label = (
            self.image_labels[anchor_index]
            if anchor_index < len(self.image_labels)
            else None
        )
        self.knn_bulk_btn.setEnabled(bool(anchor_label) and unlabeled_neighbors > 0)

    def jump_to_nearest_neighbor(self):
        """Jump preview/selection to the nearest neighbor of current anchor."""
        if not self._current_knn_neighbors:
            self.status.showMessage("No neighbors available for jump")
            return

        neighbor_index = int(self._current_knn_neighbors[0]["index"])
        self.selected_point_index = neighbor_index
        self.hover_locked = True
        self.request_preview_for_index(neighbor_index, source="knn-jump")
        self.request_update_explorer_selection(neighbor_index)
        self.status.showMessage(f"Jumped to nearest neighbor #{neighbor_index}")

    def bulk_label_nearest_neighbors(self):
        """Apply anchor label to nearby unlabeled neighbors after confirmation."""
        if not self._begin_command():
            return

        try:
            anchor_index = self.selected_point_index
            if anchor_index is None or anchor_index >= len(self.image_labels):
                self.status.showMessage("Select a point first")
                return

            anchor_label = self.image_labels[anchor_index]
            if not anchor_label:
                self.status.showMessage(
                    "Anchor point must be labeled before bulk assist"
                )
                return

            unlabeled_neighbors = [
                item for item in self._current_knn_neighbors if not item.get("label")
            ]
            if not unlabeled_neighbors:
                self.status.showMessage("No unlabeled neighbors available")
                return

            to_apply = unlabeled_neighbors[:5]
            if not self._ask_yes_no(
                "Confirm Bulk Label Assist",
                f"Apply label '{anchor_label}' to {len(to_apply)} nearest unlabeled neighbors?\n\n"
                "You can undo these assignments with Ctrl+Z.",
            ):
                return

            for item in to_apply:
                idx = int(item["index"])
                prev = self.image_labels[idx] if idx < len(self.image_labels) else None
                self.last_assigned_stack.append(
                    {
                        "index": idx,
                        "previous_label": prev,
                        "new_label": anchor_label,
                    }
                )
                self._set_label_for_index(idx, anchor_label)
                self._remove_history_for_index(idx)
                self.label_history.append({"index": idx, "label": anchor_label})
                self.candidate_indices = [i for i in self.candidate_indices if i != idx]

            self.request_refresh_label_history_strip()
            self.request_update_explorer_plot()
            self.request_update_context_panel()
            self.update_knn_panel(anchor_index)
            self.status.showMessage(
                f"Applied '{anchor_label}' to {len(to_apply)} nearest unlabeled neighbors"
            )
        finally:
            self._end_command(0.15)

    def assign_label_to_selected(self, label: str):
        """Assign label to selected point and persist in DB."""
        if not self._begin_command():
            return
        if self.explorer_mode != "labeling":
            self.status.showMessage(
                "Label assignment is disabled outside Labeling mode"
            )
            self._end_command(0.08)
            return

        next_action = None
        try:
            if self.selected_point_index is None:
                self.status.showMessage("Select a point first before assigning a label")
                return

            index = self.selected_point_index
            if index < 0 or index >= len(self.image_paths):
                return

            previous_label = (
                self.image_labels[index] if index < len(self.image_labels) else None
            )
            self.last_assigned_stack.append(
                {
                    "index": index,
                    "previous_label": previous_label,
                    "new_label": label,
                }
            )

            self._set_label_for_index(index, label)
            self.round_labeled_indices = [
                i for i in self.round_labeled_indices if i != index
            ]
            self.round_labeled_indices.append(index)
            self._remove_history_for_index(index)
            self.label_history.append({"index": index, "label": label})
            self.request_refresh_label_history_strip()

            self.on_label_assigned(label)

            # Drop from candidate list once labeled
            self.candidate_indices = [i for i in self.candidate_indices if i != index]
            labels = self.image_labels or []
            next_unlabeled = None
            for i in self.candidate_indices:
                if i >= len(labels) or not labels[i]:
                    next_unlabeled = i
                    break

            if next_unlabeled is not None:
                self.selected_point_index = next_unlabeled
                self.request_preview_for_index(
                    self.selected_point_index, source="next-unlabeled"
                )
                self.hover_locked = True
            else:
                self.selected_point_index = None
                self.hover_locked = False
                self.request_update_explorer_selection(None)
                self.request_preview_for_index(index, source="label")
                next_action = self._prompt_after_label_set_complete()

            self.request_update_explorer_plot()
            self.request_update_context_panel()
        finally:
            self._end_command(0.12)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "train":
            QTimer.singleShot(0, self.train_classifier)

    def _prompt_after_label_set_complete(self) -> str | None:
        """Prompt next step when all points in current sampled set are labeled."""
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Labeling Set Complete")
        message.setText("All points in the current labeling set are labeled.")
        message.setInformativeText(
            "Choose what to do next: sample more smart selections, build an Active Learning batch, or start training."
        )
        sample_btn = message.addButton("Sample Another Set", QMessageBox.AcceptRole)
        al_btn = message.addButton("Build AL Batch", QMessageBox.ActionRole)
        train_btn = message.addButton("Train Classifier", QMessageBox.ActionRole)
        message.addButton(QMessageBox.Close)
        message.setDefaultButton(sample_btn)
        message.exec()

        clicked = message.clickedButton()
        if clicked == sample_btn:
            return "sample"
        if clicked == al_btn:
            return "al"
        if clicked == train_btn:
            return "train"
        return None

    def update_explorer_plot(self, force_fit: bool = False):
        """Refresh explorer points with current mode, labels, and candidate emphasis."""
        if not hasattr(self, "explorer"):
            return

        # Choose which UMAP coordinates to display
        if self._show_model_umap and self.umap_model_coords is not None:
            coords = self.umap_model_coords
        elif self._show_model_pca and self.pca_model_coords is not None:
            coords = self.pca_model_coords
        else:
            coords = self.umap_coords
        if coords is None:
            return

        if self.explorer_mode == "explore":
            color_values = self.cluster_assignments
            candidate_indices = []
        elif self.explorer_mode == "labeling":
            seeded_labels = list(self.image_labels or [])
            seeded_labels.extend(list(self.classes or []))
            seeded_labels.append("unknown")
            color_values = seeded_labels
            candidate_indices = self.candidate_indices
        else:
            seeded_preds = self._prediction_labels_for_plot()
            seeded_preds.extend(list(self._model_class_names or []))
            color_values = seeded_preds
            candidate_indices = []

        if not force_fit:
            if self.explorer.update_state(
                labels=color_values,
                confidences=self.image_confidences,
                candidate_indices=candidate_indices,
                round_labeled_indices=self.round_labeled_indices,
                selected_index=self.selected_point_index,
                labeling_mode=(self.explorer_mode == "labeling"),
            ):
                return

        self.explorer.set_data(
            coords,
            color_values,
            confidences=self.image_confidences,
            candidate_indices=candidate_indices,
            round_labeled_indices=self.round_labeled_indices,
            selected_index=self.selected_point_index,
            labeling_mode=(self.explorer_mode == "labeling"),
            preserve_view=(not force_fit),
        )

    # ================== Data Operations ==================

    def manage_sources(self):
        """Open the Source Manager to view, add, or remove image source folders."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first.\n\n"
                "Use File → New Project to get started.",
            )
            return

        from .dialogs import SourceManagerDialog

        dlg = SourceManagerDialog(db_path=self.db_path, parent=self)
        if dlg.exec() != SourceManagerDialog.Accepted or not dlg.has_changes:
            return

        folders_to_remove = dlg.folders_to_remove
        folders_to_add = dlg.folders_to_add

        # ── removals ────────────────────────────────────────────────
        if folders_to_remove:
            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            total_removed = 0
            for folder in folders_to_remove:
                total_removed += db.remove_images_by_folder_exact(folder)
            if total_removed:
                self.status.showMessage(
                    f"Removed {total_removed:,} images from {len(folders_to_remove)} source(s)"
                )

        # ── additions ───────────────────────────────────────────────
        if folders_to_add:
            self._ingest_queue = list(folders_to_add)
            self._ingest_batch_total = len(self._ingest_queue)
            self._run_next_ingest()
        elif folders_to_remove:
            # Removals only — reload and offer to redo pipeline
            self._flush_pending_label_updates(force=True)
            self.load_project_data()
            self.update_context_panel()
            # Caches are now stale (image count changed) — offer auto-redo
            self.embeddings = None
            self.cluster_assignments = None
            self.umap_coords = None
            QTimer.singleShot(200, self._auto_pipeline_after_source_change)

    # keep legacy name as alias for internal callers
    ingest_images = manage_sources

    def _run_next_ingest(self):
        """Start the next item in the ingest queue."""
        if not self._ingest_queue:
            return
        folder = self._ingest_queue[0]

        from ..jobs.task_workers import IngestWorker

        worker = IngestWorker(folder, self.db_path)
        worker.signals.started.connect(
            lambda f=folder: self.status.showMessage(f"[Step 1/5] Ingesting {f.name}…")
        )
        worker.signals.progress.connect(self.on_ingest_progress)
        worker.signals.success.connect(self._on_ingest_batch_item_done)
        worker.signals.error.connect(self.on_ingest_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _on_ingest_batch_item_done(self, result):
        """Called when one folder in the ingest batch finishes."""
        self._ingest_queue.pop(0)
        if self._ingest_queue:
            # More folders to go
            self._run_next_ingest()
        else:
            # All done — flush any in-flight labels before reloading so they are preserved
            self._flush_pending_label_updates(force=True)
            num_images = result.get("num_images", 0)
            total = (
                num_images  # accumulative count not tracked per-folder; just show last
            )
            QMessageBox.information(
                self,
                "Ingestion Complete",
                f"Ingested images from {self._ingest_batch_total} folder(s).\n"
                f"Last batch: {total:,} images added.",
            )
            self.status.showMessage(
                f"Ingestion complete ({self._ingest_batch_total} folder(s))"
            )
            self.load_project_data()
            self.update_context_panel()
            # Caches are now stale — invalidate and offer to redo
            self.embeddings = None
            self.cluster_assignments = None
            self.umap_coords = None
            QTimer.singleShot(200, self._auto_pipeline_after_source_change)

    # ── auto-pipeline after source changes ─────────────────────────────

    def _auto_pipeline_after_source_change(self):
        """After sources changed: offer to re-run the full embed → cluster → UMAP
        pipeline using existing settings (without showing configuration dialogs)."""
        if not self.image_paths:
            return

        # Retrieve the most recent embedding settings from the DB so we can
        # re-run without prompting for configuration.
        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)

        # Look for last embedding metadata to recover settings
        last_embed_meta = self._get_last_embedding_settings(db)
        last_cluster_meta = self._get_last_cluster_settings(db)

        if last_embed_meta is None:
            # No previous pipeline — fall back to the interactive flow
            self._auto_pipeline_check()
            return

        model_name = last_embed_meta.get("model_name", "dinov2_vitb14")
        device = last_embed_meta.get("device", "cpu")
        batch_size = last_embed_meta.get("batch_size", 32)

        n_clusters = 500
        cluster_method = "minibatch"
        if last_cluster_meta:
            n_clusters = last_cluster_meta.get("n_clusters", 500)
            cluster_method = last_cluster_meta.get("method", "minibatch")

        n_neighbors = self.last_umap_params.get("n_neighbors", 15)
        min_dist = self.last_umap_params.get("min_dist", 0.1)

        summary = (
            f"Sources have changed. Re-run the pipeline with previous settings?\n\n"
            f"  Embedding: {model_name} on {device} (batch {batch_size})\n"
            f"  Clustering: {cluster_method}, k={n_clusters}\n"
            f"  UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}\n\n"
            f"Images: {len(self.image_paths):,}"
        )

        reply = QMessageBox.question(
            self,
            "Re-run Pipeline?",
            summary,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        # Chain: embed → cluster → UMAP using saved settings (no dialogs)
        self._auto_rerun_embed(
            model_name,
            device,
            batch_size,
            n_clusters,
            cluster_method,
            n_neighbors,
            min_dist,
        )

    def _get_last_embedding_settings(self, db):
        """Return the metadata dict from the most recent embedding run, or None."""
        with __import__("sqlite3").connect(db.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT model_name, device, batch_size, meta_json
                FROM embeddings
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = c.fetchone()
        if not row:
            return None
        model_name, device, batch_size, meta_json = row
        result = {"model_name": model_name, "device": device, "batch_size": batch_size}
        if meta_json:
            import json as _json

            try:
                meta = _json.loads(meta_json)
                if isinstance(meta, dict):
                    result.update(meta)
            except Exception:
                pass
        return result

    def _get_last_cluster_settings(self, db):
        """Return the metadata from the most recent cluster run, or None."""
        with __import__("sqlite3").connect(db.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                SELECT n_clusters, method
                FROM cluster_cache
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            row = c.fetchone()
        if not row:
            return None
        return {"n_clusters": row[0], "method": row[1]}

    def _auto_rerun_embed(
        self,
        model_name,
        device,
        batch_size,
        n_clusters,
        cluster_method,
        n_neighbors,
        min_dist,
    ):
        """Start embeddings computation with given settings, then chain cluster + UMAP."""
        from ..jobs.task_workers import EmbeddingWorker

        worker = EmbeddingWorker(
            self.image_paths,
            model_name,
            device,
            batch_size,
            db_path=self.db_path,
            force_recompute=True,
        )
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-computing embeddings…")
        )
        worker.signals.progress.connect(self.on_embedding_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_embed_success(result):
            self.on_embedding_success(result)
            # Chain to clustering
            QTimer.singleShot(
                200,
                lambda: self._auto_rerun_cluster(
                    n_clusters,
                    cluster_method,
                    n_neighbors,
                    min_dist,
                ),
            )

        worker.signals.success.connect(_on_embed_success)
        worker.signals.error.connect(self.on_embedding_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _auto_rerun_cluster(self, n_clusters, method, n_neighbors, min_dist):
        """Run clustering with given settings, then chain UMAP."""
        if self.embeddings is None:
            return

        from ..jobs.task_workers import ClusteringWorker

        worker = ClusteringWorker(self.embeddings, n_clusters, method)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-clustering data…")
        )
        worker.signals.progress.connect(self.on_clustering_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_cluster_success(result):
            self.on_clustering_success(result)
            # Chain to UMAP
            QTimer.singleShot(
                200,
                lambda: self._auto_rerun_umap(n_neighbors, min_dist),
            )

        worker.signals.success.connect(_on_cluster_success)
        worker.signals.error.connect(self.on_clustering_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _auto_rerun_umap(self, n_neighbors, min_dist):
        """Run UMAP with given settings."""
        if self.embeddings is None:
            return

        from ..jobs.task_workers import UMAPWorker

        self.last_umap_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}
        worker = UMAPWorker(self.embeddings, n_neighbors, min_dist)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Re-computing UMAP…")
        )
        worker.signals.progress.connect(self.on_umap_progress)
        worker.signals.success.connect(self.on_umap_success)
        worker.signals.error.connect(self.on_umap_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    # ── guided-setup helpers ───────────────────────────────────────────────

    def _prompt_adjust_sources_if_empty(self):
        """After new-project creation: automatically open Source Manager if no images yet."""
        if not self.image_paths:
            self.manage_sources()

    def _guide_to_next_step(self):
        """After opening an existing project, guide the user to the next incomplete step."""
        if not self.project_path:
            return

        if not self.image_paths:
            reply = QMessageBox.question(
                self,
                "No Images Found",
                f"Project '{self.project_path.name}' has no images ingested yet.\n\n"
                "Would you like to add image sources now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.ingest_images()
            return

        if self.embeddings is None:
            reply = QMessageBox.question(
                self,
                "Embeddings Not Computed",
                f"{len(self.image_paths):,} images are loaded but embeddings have "
                "not been computed yet.\n\nCompute embeddings now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._compute_embeddings_then(callback=self._auto_cluster_check)
            return

        if self.cluster_assignments is None:
            reply = QMessageBox.question(
                self,
                "Clustering Not Done",
                "Embeddings are ready but clustering has not been run.\n\n"
                "Run clustering now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._cluster_data_then(callback=self._auto_umap_check)
            return

        if self.umap_coords is None:
            reply = QMessageBox.question(
                self,
                "UMAP Not Computed",
                "Clustering is done but UMAP has not been computed yet.\n\n"
                "Compute UMAP now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.compute_umap()

    def _auto_pipeline_check(self):
        """After ingestion: offer to run embed → cluster → UMAP if not yet done."""
        if not self.image_paths:
            return

        # Step 1: embeddings
        if self.embeddings is None:
            reply = QMessageBox.question(
                self,
                "Compute Embeddings?",
                f"{len(self.image_paths):,} images have been ingested.\n\n"
                "Embeddings are required before clustering and UMAP visualisation.\n"
                "Compute them now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._compute_embeddings_then(callback=self._auto_cluster_check)
        else:
            self._auto_cluster_check()

    def _auto_cluster_check(self):
        """After embeddings: offer to run cluster → UMAP."""
        if self.embeddings is None:
            return
        if self.cluster_assignments is None:
            reply = QMessageBox.question(
                self,
                "Cluster Data?",
                "Embeddings are ready.\n\n"
                "Clustering groups images into visual clusters to guide labeling.\n"
                "Run clustering now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self._cluster_data_then(callback=self._auto_umap_check)
        else:
            self._auto_umap_check()

    def _auto_umap_check(self):
        """After clustering: offer to compute UMAP."""
        if self.cluster_assignments is None:
            return
        if self.umap_coords is None:
            reply = QMessageBox.question(
                self,
                "Compute UMAP?",
                "Clustering complete.\n\n"
                "Computing a 2-D UMAP projection allows you to visually explore and "
                "label your dataset.  Run UMAP now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.compute_umap()

    def _compute_embeddings_then(self, callback=None):
        """Like compute_embeddings() but calls *callback* on success."""
        if not self.image_paths:
            return

        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached = db.get_most_recent_embeddings()
        if cached is not None:
            embeddings, metadata = cached
            if self._ask_yes_no(
                "Use Cached Embeddings",
                "Cached embeddings found. Load them instead of recomputing?\n\n"
                f"Model: {metadata.get('model_name', 'unknown')}\n"
                f"Timestamp: {metadata.get('timestamp', 'unknown')}",
            ):
                self.embeddings = embeddings
                self.status.showMessage(
                    f"Loaded {embeddings.shape[0]:,} cached embeddings"
                )
                self.update_context_panel()
                if callback:
                    QTimer.singleShot(200, callback)
                return

        from .dialogs import EmbeddingDialog

        dialog = EmbeddingDialog(self)
        if not dialog.exec():
            return
        model_name, device, batch_size, force_recompute = dialog.get_settings()

        from ..jobs.task_workers import EmbeddingWorker

        worker = EmbeddingWorker(
            self.image_paths,
            model_name,
            device,
            batch_size,
            db_path=self.db_path,
            force_recompute=force_recompute,
        )
        worker.signals.started.connect(
            lambda: self.status.showMessage("Computing embeddings…")
        )
        worker.signals.progress.connect(self.on_embedding_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_success(result):
            self.on_embedding_success(result)
            if callback:
                QTimer.singleShot(200, callback)

        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(self.on_embedding_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def _cluster_data_then(self, callback=None):
        """Like cluster_data() but calls *callback* on success."""
        if self.embeddings is None:
            return

        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_cluster = db.get_most_recent_cluster_cache()
        if cached_cluster is not None:
            if self._ask_yes_no(
                "Use Cached Clusters",
                "Cached cluster assignments found. Load them?\n\n"
                f"Method: {cached_cluster.get('method', 'unknown')}\n"
                f"Timestamp: {cached_cluster.get('timestamp', 'unknown')}",
            ):
                self.cluster_assignments = cached_cluster["assignments"]
                self.status.showMessage("Loaded cached cluster assignments")
                self.update_explorer_plot()
                self.update_context_panel()
                if callback:
                    QTimer.singleShot(200, callback)
                return

        from .dialogs import ClusterDialog

        dialog = ClusterDialog(self)
        if not dialog.exec():
            return
        n_clusters, method = dialog.get_settings()

        from ..jobs.task_workers import ClusteringWorker

        worker = ClusteringWorker(self.embeddings, n_clusters, method)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Clustering data…")
        )
        worker.signals.progress.connect(self.on_clustering_progress)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        def _on_success(result):
            self.on_clustering_success(result)
            if callback:
                QTimer.singleShot(200, callback)

        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(self.on_clustering_error)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    # ── class editor ─────────────────────────────────────────────────────

    def open_class_editor(self):
        """Open the dedicated class / labeling-scheme editor dialog."""
        from .dialogs import ClassEditorDialog

        scheme_dict = None
        if self.project_path:
            scheme_path = self.project_path / "scheme.json"
            if scheme_path.exists():
                try:
                    import json as _json

                    with open(scheme_path) as _f:
                        scheme_dict = _json.load(_f)
                except Exception:
                    pass

        dlg = ClassEditorDialog(
            classes=self.classes,
            scheme_dict=scheme_dict,
            parent=self,
        )
        if dlg.exec() != ClassEditorDialog.Accepted:
            return

        new_scheme = dlg.get_scheme_dict()
        flat_classes = dlg.flat_classes

        # Persist to project
        if self.project_path:
            import json as _json

            config_path = self.project_path / "project.json"
            config = {}
            if config_path.exists():
                with open(config_path) as _f:
                    config = _json.load(_f)
            config["classes"] = flat_classes
            with open(config_path, "w") as _f:
                _json.dump(config, _f, indent=2)

            if new_scheme:
                with open(self.project_path / "scheme.json", "w") as _f:
                    _json.dump(new_scheme, _f, indent=2)

        self.classes = flat_classes
        self.rebuild_label_buttons()
        self.setup_label_shortcuts()
        self.update_context_panel()
        self.status.showMessage(f"Classes updated ({len(self.classes)})")

    # ── shortcut editor ──────────────────────────────────────────────────

    def open_shortcut_editor(self):
        """Open the keyboard-shortcut editor dialog."""
        from .dialogs import ShortcutEditorDialog

        dlg = ShortcutEditorDialog(current=self._custom_shortcuts, parent=self)
        if dlg.exec() != ShortcutEditorDialog.Accepted:
            return

        self._custom_shortcuts = dlg.get_shortcuts()
        self.setup_label_shortcuts()
        self._refresh_shortcut_help()

        if self.project_path:
            import json as _json

            config_path = self.project_path / "project.json"
            config = {}
            if config_path.exists():
                with open(config_path) as _f:
                    config = _json.load(_f)
            config["custom_shortcuts"] = self._custom_shortcuts
            with open(config_path, "w") as _f:
                _json.dump(config, _f, indent=2)

        self.status.showMessage("Keyboard shortcuts updated")

    def open_contrast_settings(self):
        """Open the CLAHE contrast enhancement settings dialog."""
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
        )

        class ContrastSettingsDialog(QDialog):
            def __init__(self, clip, grid, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Contrast Enhancement Settings")
                layout = QVBoxLayout(self)
                layout.setSpacing(16)

                form = QFormLayout()
                self.clip_spin = QDoubleSpinBox()
                self.clip_spin.setRange(0.1, 40.0)
                self.clip_spin.setSingleStep(0.1)
                self.clip_spin.setValue(clip)
                form.addRow("<b>CLAHE Clip Limit:</b>", self.clip_spin)

                grid_row = QHBoxLayout()
                self.grid_x = QSpinBox()
                self.grid_x.setRange(1, 64)
                self.grid_x.setValue(grid[0])
                self.grid_y = QSpinBox()
                self.grid_y.setRange(1, 64)
                self.grid_y.setValue(grid[1])
                grid_row.addWidget(self.grid_x)
                grid_row.addWidget(QLabel("x"))
                grid_row.addWidget(self.grid_y)
                form.addRow("<b>CLAHE Grid Size:</b>", grid_row)

                layout.addLayout(form)

                info = QLabel(
                    "<span style='color: #888;'>"
                    "CLAHE (Contrast Limited Adaptive Histogram Equalization) "
                    "improves visibility in dark or low-contrast images."
                    "</span>"
                )
                info.setWordWrap(True)
                layout.addWidget(info)

                buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self
                )
                buttons.accepted.connect(self.accept)
                buttons.rejected.connect(self.reject)
                layout.addWidget(buttons)

        dlg = ContrastSettingsDialog(self.clahe_clip, self.clahe_grid, self)
        if dlg.exec() == QDialog.Accepted:
            self.clahe_clip = dlg.clip_spin.value()
            self.clahe_grid = (dlg.grid_x.value(), dlg.grid_y.value())

            # Update canvas
            if hasattr(self, "preview_canvas"):
                self.preview_canvas.set_clahe_params(self.clahe_clip, self.clahe_grid)

            # Persist to project
            if self.project_path:
                config_path = self.project_path / "project.json"
                config = {}
                if config_path.exists():
                    with open(config_path) as _f:
                        config = json.load(_f)
                config["clahe_clip"] = self.clahe_clip
                config["clahe_grid"] = list(self.clahe_grid)
                with open(config_path, "w") as _f:
                    json.dump(config, _f, indent=2)

            self.status.showMessage("Contrast enhancement settings updated")

            # Refresh current preview if visible
            if self.last_preview_index is not None:
                self.load_preview_for_index(self.last_preview_index, source="enhance")

    def _refresh_shortcut_help(self):
        """Rebuild the shortcut reminder label from current bindings."""
        from .dialogs import ShortcutEditorDialog

        defaults = dict(ShortcutEditorDialog.DEFAULT_SHORTCUTS)
        active = {**defaults, **self._custom_shortcuts}

        label_instr = "1–9 (fallback)"
        if self._stepper is not None:
            label_instr = "stepper keys"
        else:
            # Check if we have scheme shortcuts
            try:
                scheme_path = self.project_path / "scheme.json"
                if scheme_path.exists():
                    with open(scheme_path) as _f:
                        scheme_dict = json.load(_f)
                        factors = scheme_dict.get("factors", [])
                        if factors:
                            f = factors[0]
                            keys = [k for k in f.get("shortcut_keys", []) if k]
                            if keys:
                                label_instr = f"defined keys: {', '.join(keys[:5])}..."
            except Exception:
                pass

        lines = [
            "<div style='line-height:1.5; color:#bcbcbc;'>",
            "<b>Controls:</b><br>",
            f"• <b>{active.get('Explore mode', 'E')}</b> / <b>{active.get('Labeling mode', 'L')}</b> / <b>{active.get('Predictions mode', 'P')}</b>: set mode<br>",
            f"• <b>{label_instr}</b>: assign class<br>",
            "• <b>0</b>: mark as unknown<br>",
            f"• <b>{active.get('Sample next candidates', 'Space')}</b>: sample candidates<br>",
            f"• <b>{active.get('Previous unlabeled', 'Left')}</b> / <b>{active.get('Next unlabeled', 'Right')}</b>: navigate<br>",
            f"• <b>{active.get('Undo last label (Ctrl+Z)', 'Ctrl+Z')}</b>: undo<br>",
            "</div>",
        ]
        if hasattr(self, "shortcut_help"):
            self.shortcut_help.setText("".join(lines))

    # ── startup overlay ──────────────────────────────────────────────────

    def compute_embeddings(self):
        """Compute embeddings for all images."""
        if not self.project_path:
            QMessageBox.warning(
                self, "No Project", "Please create or open a project first."
            )
            return

        # Check if we have images
        if not self.image_paths:
            QMessageBox.warning(
                self,
                "No Images",
                "Please ingest images first.\\n\\nUse File → Ingest Images.",
            )
            return

        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached = db.get_most_recent_embeddings()
        if cached is not None:
            embeddings, metadata = cached
            if self._ask_yes_no(
                "Use Cached Embeddings",
                "Most recent cached embeddings are available. Load them instead of recomputing?\n\n"
                f"Model: {metadata.get('model_name', 'unknown')}\n"
                f"Timestamp: {metadata.get('timestamp', 'unknown')}",
            ):
                self.embeddings = embeddings
                self.status.showMessage(
                    f"Loaded {embeddings.shape[0]:,} cached embeddings"
                )
                self.update_context_panel()
                return

        from .dialogs import EmbeddingDialog

        dialog = EmbeddingDialog(self)
        if dialog.exec():
            model_name, device, batch_size, force_recompute = dialog.get_settings()

            from ..jobs.task_workers import EmbeddingWorker

            worker = EmbeddingWorker(
                self.image_paths,
                model_name,
                device,
                batch_size,
                db_path=self.db_path,
                force_recompute=force_recompute,
            )
            worker.signals.started.connect(
                lambda: self.status.showMessage("Computing embeddings...")
            )
            worker.signals.progress.connect(self.on_embedding_progress)
            worker.signals.success.connect(self.on_embedding_success)
            worker.signals.error.connect(self.on_embedding_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._threadpool_start(worker)

    def cluster_data(self):
        """Cluster embeddings."""
        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Please compute embeddings first.\\n\\nUse Compute → Compute Embeddings.",
            )
            return

        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_cluster = db.get_most_recent_cluster_cache()
        if cached_cluster is not None:
            if self._ask_yes_no(
                "Use Cached Clusters",
                "Most recent cluster assignments are available. Load them instead of reclustering?\n\n"
                f"Method: {cached_cluster.get('method', 'unknown')}\n"
                f"Timestamp: {cached_cluster.get('timestamp', 'unknown')}",
            ):
                self.cluster_assignments = cached_cluster["assignments"]
                self.status.showMessage("Loaded cached cluster assignments")
                self.update_explorer_plot()
                self.update_context_panel()
                return

        from .dialogs import ClusterDialog

        dialog = ClusterDialog(self)
        if dialog.exec():
            n_clusters, method = dialog.get_settings()

            from ..jobs.task_workers import ClusteringWorker

            worker = ClusteringWorker(self.embeddings, n_clusters, method)
            worker.signals.started.connect(
                lambda: self.status.showMessage("Clustering data...")
            )
            worker.signals.progress.connect(self.on_clustering_progress)
            worker.signals.success.connect(self.on_clustering_success)
            worker.signals.error.connect(self.on_clustering_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self._threadpool_start(worker)

    def compute_umap(self):
        """Compute UMAP projection."""
        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Please compute embeddings first.\\n\\nUse Compute → Compute Embeddings.",
            )
            return

        from ..store.db import ClassKitDB

        db = ClassKitDB(self.db_path)
        cached_umap = db.get_most_recent_umap_cache()
        if cached_umap is not None:
            if self._ask_yes_no(
                "Use Cached UMAP",
                "Most recent UMAP projection is available. Load it instead of recomputing?\n\n"
                f"Timestamp: {cached_umap.get('timestamp', 'unknown')}\n"
                f"n_neighbors: {cached_umap.get('n_neighbors', 'unknown')}\n"
                f"min_dist: {cached_umap.get('min_dist', 'unknown')}",
            ):
                self.umap_coords = cached_umap["coords"]
                self.last_umap_params = {
                    "n_neighbors": cached_umap.get("n_neighbors", 15),
                    "min_dist": cached_umap.get("min_dist", 0.1),
                }
                self.status.showMessage("Loaded cached UMAP projection")
                self.update_explorer_plot(force_fit=True)
                self.update_context_panel()
                return

        from ..jobs.task_workers import UMAPWorker

        worker = UMAPWorker(self.embeddings)
        worker.signals.started.connect(
            lambda: self.status.showMessage("Computing UMAP...")
        )
        worker.signals.progress.connect(self.on_umap_progress)
        worker.signals.success.connect(self.on_umap_success)
        worker.signals.error.connect(self.on_umap_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def train_classifier(self):
        """Open ClassKitTrainingDialog, export dataset, run training, offer publish."""
        self._flush_pending_label_updates(force=True)

        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Compute embeddings before training.",
            )
            return

        labeled_pairs = [
            (p, l) for p, l in zip(self.image_paths, self.image_labels) if l
        ]
        if len(labeled_pairs) < 4:
            QMessageBox.warning(
                self,
                "Not Enough Labels",
                "Need at least 4 labeled images.",
            )
            return

        # Resolve scheme from project directory (if present)
        scheme = None
        if self.project_path:
            try:
                from ..config.schemas import LabelingScheme

                scheme_path = self.project_path / "scheme.json"
                if scheme_path.exists():
                    with open(scheme_path) as _f:
                        scheme = LabelingScheme.from_dict(json.load(_f))
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to load scheme.json; proceeding without scheme",
                    exc_info=True,
                )
                scheme = None

        from .dialogs import ClassKitTrainingDialog

        project_class_choices = sorted(
            {str(lbl).strip() for _, lbl in labeled_pairs if str(lbl).strip()}
        )

        dialog = ClassKitTrainingDialog(
            scheme=scheme,
            n_labeled=len(labeled_pairs),
            class_choices=project_class_choices,
            parent=self,
        )

        def _do_train():
            from pathlib import Path as _Path

            from ...training.contracts import (
                AugmentationProfile,
                TinyHeadTailParams,
                TrainingHyperParams,
                TrainingRole,
                TrainingRunSpec,
            )
            from ..jobs.task_workers import ClassKitTrainingWorker, ExportWorker

            settings = dialog.get_settings()
            # Persist latest training settings so post-train inference reuses
            # the selected training device/runtime choices.
            self._last_training_settings = dict(settings)
            mode = settings.get("mode") or "flat_tiny"
            is_yolo = "yolo" in mode
            multi_head = mode.startswith("multihead")

            role_map = {
                "flat_tiny": TrainingRole.CLASSIFY_FLAT_TINY,
                "flat_yolo": TrainingRole.CLASSIFY_FLAT_YOLO,
                "multihead_tiny": TrainingRole.CLASSIFY_MULTIHEAD_TINY,
                "multihead_yolo": TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
                "flat_custom": TrainingRole.CLASSIFY_FLAT_CUSTOM,
                "multihead_custom": TrainingRole.CLASSIFY_MULTIHEAD_CUSTOM,
            }
            role = role_map.get(mode, TrainingRole.CLASSIFY_FLAT_TINY)

            project_path = (
                _Path(self.project_path) if self.project_path else _Path.cwd()
            )

            # Use a unique run directory for this training session
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            run_dir = project_path / ".classkit_runs" / f"{mode}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            images = [_Path(p) for p, _ in labeled_pairs]
            labels_str = [lbl for _, lbl in labeled_pairs]
            unique = sorted(set(labels_str))
            label_map_int = {s: i for i, s in enumerate(unique)}
            int_labels = [label_map_int[lbl] for lbl in labels_str]
            class_names_int = {i: s for s, i in label_map_int.items()}

            def _make_spec(dataset_dir):
                import dataclasses

                from ...training.contracts import CustomCNNParams

                aug = AugmentationProfile(
                    enabled=True,
                    flipud=settings.get("flipud", 0.0),
                    fliplr=settings.get("fliplr", 0.5),
                    rotate=settings.get("rotate", 0.0),
                    label_expansion=settings.get("label_expansion") or {},
                )

                spec = TrainingRunSpec(
                    role=role,
                    source_datasets=[],
                    derived_dataset_dir=str(dataset_dir),
                    base_model=settings.get("base_model", "") if is_yolo else "",
                    hyperparams=TrainingHyperParams(
                        epochs=settings.get("epochs", 50),
                        batch=settings.get("batch", 32),
                        lr0=settings.get("lr", 0.001),
                        patience=settings.get("patience", 10),
                    ),
                    tiny_params=TinyHeadTailParams(
                        epochs=settings.get("epochs", 50),
                        batch=settings.get("batch", 32),
                        lr=settings.get("lr", 0.001),
                        patience=settings.get("patience", 10),
                        hidden_layers=settings.get("tiny_layers", 1),
                        hidden_dim=settings.get("tiny_dim", 64),
                        dropout=settings.get("tiny_dropout", 0.2),
                        input_width=settings.get("tiny_width", 128),
                        input_height=settings.get("tiny_height", 64),
                        class_rebalance_mode=settings.get(
                            "tiny_rebalance_mode", "none"
                        ),
                        class_rebalance_power=settings.get("tiny_rebalance_power", 1.0),
                        label_smoothing=settings.get("tiny_label_smoothing", 0.0),
                    ),
                    device=settings.get("device", "cpu"),
                    training_space="original",
                    augmentation_profile=aug,
                )
                if mode in ("flat_custom", "multihead_custom"):
                    spec = dataclasses.replace(
                        spec,
                        custom_params=CustomCNNParams(
                            backbone=settings.get("custom_backbone", "tinyclassifier"),
                            trainable_layers=settings.get("custom_trainable_layers", 0),
                            backbone_lr_scale=settings.get(
                                "custom_backbone_lr_scale", 0.1
                            ),
                            input_size=settings.get("custom_input_size", 224),
                            epochs=settings.get("epochs", 50),
                            batch=settings.get("batch", 32),
                            lr=settings.get("lr", 1e-3),
                            patience=settings.get("patience", 10),
                            weight_decay=1e-2,
                            label_smoothing=settings.get("tiny_label_smoothing", 0.0),
                            class_rebalance_mode=settings.get(
                                "tiny_rebalance_mode", "none"
                            ),
                            class_rebalance_power=settings.get(
                                "tiny_rebalance_power", 1.0
                            ),
                        ),
                    )
                return spec

            # Start export → then chain to training
            def _on_export_success(result):
                specs = []
                if multi_head and scheme is not None:
                    # In multi-head, each factor gets its own spec pointing to its export folder
                    for fi in range(len(scheme.factors)):
                        factor_export_dir = run_dir / f"export_f{fi}"
                        specs.append(_make_spec(factor_export_dir))
                else:
                    specs = [_make_spec(run_dir / "export")]

                # Transition to training worker
                train_worker = ClassKitTrainingWorker(
                    role=role, specs=specs, run_dir=str(run_dir), multi_head=multi_head
                )
                dialog._worker = train_worker
                train_worker.signals.progress.connect(_on_progress)
                train_worker.signals.success.connect(_on_success)
                train_worker.signals.error.connect(_on_error)
                self._threadpool_start(train_worker)

            def _on_progress(pct: int, msg: str) -> None:
                if pct >= 0:
                    dialog.progress_bar.setValue(pct)
                if msg:
                    dialog.append_log(msg)

            def _on_success(results: list) -> None:
                dialog._train_results = results
                dialog.publish_btn.setEnabled(True)
                dialog.append_log("Training complete.")
                dialog.start_btn.setEnabled(True)
                dialog.cancel_btn.setEnabled(False)

                # ── Save to project model cache (DB) ──────────────────
                if self.db_path and results:
                    try:
                        from ..store.db import ClassKitDB as _CKDb

                        _db = _CKDb(self.db_path)
                        artifact_paths = [
                            r.get("artifact_path", "")
                            for r in results
                            if r.get("artifact_path")
                            and Path(r["artifact_path"]).exists()
                        ]
                        if artifact_paths:
                            all_classes = sorted(set(labels_str))
                            acc_values = []
                            for r in results:
                                value = r.get("best_val_acc")
                                if value is None:
                                    continue
                                try:
                                    acc_values.append(float(value))
                                except Exception:
                                    continue
                            best_acc = max(acc_values) if acc_values else None
                            _db.save_model_cache(
                                mode=mode,
                                artifact_paths=artifact_paths,
                                class_names=all_classes,
                                best_val_acc=best_acc,
                                num_classes=len(all_classes),
                            )
                    except Exception:
                        pass  # non-fatal

                # ── Auto-run inference + UMAP post-training ───────────
                def _post_inference_chain(r):
                    """After inference: update metrics, auto-compute model UMAP."""
                    self._evaluate_model_on_labeled()
                    dialog.append_log("Inference complete — Metrics tab updated.")
                    dialog.append_log("Auto-computing model-space UMAP...")
                    QTimer.singleShot(100, self._replot_umap_model_space)

                if results:
                    artifact = results[0].get("artifact_path", "")
                    if artifact and Path(artifact).exists():
                        if is_yolo:
                            if multi_head:
                                all_artifacts = [
                                    Path(r["artifact_path"])
                                    for r in results
                                    if r.get("artifact_path")
                                    and Path(r["artifact_path"]).exists()
                                ]
                                self._active_model_mode = "yolo_multihead"
                                dialog.append_log(
                                    f"Running multi-head YOLO inference ({len(all_artifacts)} models)..."
                                )
                                self._run_multihead_yolo_inference(
                                    all_artifacts, on_success=_post_inference_chain
                                )
                            else:
                                self._yolo_model_path = Path(artifact)
                                self._active_model_mode = "yolo"
                                dialog.append_log(
                                    f"Running YOLO inference: {Path(artifact).name}..."
                                )
                                self._run_yolo_inference(
                                    Path(artifact), on_success=_post_inference_chain
                                )
                        else:
                            # Custom CNN or Tiny CNN .pth — dispatch based on arch field
                            import torch as _torch

                            _ckpt = _torch.load(
                                str(artifact), map_location="cpu", weights_only=False
                            )
                            _arch = (
                                _ckpt.get("arch", "tinyclassifier")
                                if isinstance(_ckpt, dict)
                                else "tinyclassifier"
                            )
                            _class_names = _ckpt.get("class_names") or sorted(
                                set(labels_str)
                            )
                            self._active_model_mode = "tiny"
                            if _arch != "tinyclassifier":
                                _sz = _ckpt.get("input_size", (224, 224))
                                _sz = (
                                    _sz[0]
                                    if isinstance(_sz, (list, tuple))
                                    else int(_sz)
                                )
                                dialog.append_log(
                                    f"Running Custom CNN inference ({_arch}): {Path(artifact).name}..."
                                )
                                self._run_torchvision_inference(
                                    Path(artifact),
                                    class_names=_class_names,
                                    input_size=_sz,
                                    on_success=_post_inference_chain,
                                )
                            else:
                                dialog.append_log(
                                    f"Running tiny CNN inference: {Path(artifact).name}..."
                                )
                                self._run_tiny_inference(
                                    Path(artifact),
                                    class_names=_class_names,
                                    on_success=_post_inference_chain,
                                )

            def _on_error(err: str) -> None:
                dialog.append_log(f"ERROR: {err}")
                dialog.start_btn.setEnabled(True)
                dialog.cancel_btn.setEnabled(False)

            # Determine export path and mode
            if multi_head and scheme is not None:
                # Multi-head export is special: one folder per factor
                # For now, let's keep it simple and implement multi-head export in a loop or specialized worker
                # (Re-using logic from the old _do_train but in background)
                _exp_label_expansion = settings.get("label_expansion") or {}

                class MultiHeadExportWorker(ExportWorker):
                    def __init__(self, *args, **kwargs):
                        self.scheme = kwargs.pop("scheme")
                        self.labels_str = kwargs.pop("labels_str")
                        super().__init__(*args, **kwargs)

                    @Slot()
                    def run(self):
                        try:
                            self.signals.started.emit()
                            for fi, factor in enumerate(self.scheme.factors):
                                self.signals.progress.emit(
                                    0, f"Exporting factor {fi}..."
                                )
                                f_labels_str = [
                                    self.scheme.decode_label(lbl)[fi]
                                    for lbl in self.labels_str
                                ]
                                f_unique = sorted(set(f_labels_str))
                                f_map = {s: i for i, s in enumerate(f_unique)}
                                f_int = [f_map[lbl] for lbl in f_labels_str]
                                f_names = {i: s for s, i in f_map.items()}

                                factor_dir = _Path(self.output_path) / f"export_f{fi}"
                                sub_worker = ExportWorker(
                                    image_paths=self.image_paths,
                                    labels=f_int,
                                    output_path=factor_dir,
                                    format="ultralytics",
                                    class_names=f_names,
                                    val_fraction=self.val_fraction,
                                    label_expansion=_exp_label_expansion,
                                )
                                # Run synchronously within this thread
                                sub_worker.run()

                            self.signals.success.emit({})
                        except Exception as e:
                            self.signals.error.emit(str(e))
                        finally:
                            self.signals.finished.emit()

                worker = MultiHeadExportWorker(
                    image_paths=images,
                    labels=[0] * len(images),  # unused
                    output_path=run_dir,
                    format="ultralytics",
                    val_fraction=settings.get("val_fraction", 0.2),
                    scheme=scheme,
                    labels_str=labels_str,
                )
            else:
                worker = ExportWorker(
                    image_paths=images,
                    labels=int_labels,
                    output_path=run_dir / "export",
                    format="ultralytics",
                    class_names=class_names_int,
                    val_fraction=settings.get("val_fraction", 0.2),
                    label_expansion=settings.get("label_expansion") or {},
                )

            dialog._worker = worker
            worker.signals.progress.connect(_on_progress)
            worker.signals.success.connect(_on_export_success)
            worker.signals.error.connect(_on_error)

            dialog.start_btn.setEnabled(False)
            dialog.cancel_btn.setEnabled(True)
            dialog.append_log("Starting dataset export...")
            self._threadpool_start(worker)

        scheme_name = scheme.name if scheme else "classkit"

        def _on_publish():
            results = getattr(dialog, "_train_results", None) or []
            settings = dialog.get_settings()
            mode = settings.get("mode") or "flat_tiny"
            is_yolo = "yolo" in mode
            multi_head = mode.startswith("multihead")
            role_map = {
                "flat_tiny": "classify_flat_tiny",
                "flat_yolo": "classify_flat_yolo",
                "multihead_tiny": "classify_multihead_tiny",
                "multihead_yolo": "classify_multihead_yolo",
                "flat_custom": "classify_flat_custom",
                "multihead_custom": "classify_multihead_custom",
            }
            from ...training.contracts import TrainingRole
            from ...training.model_publish import publish_trained_model

            role_val = role_map.get(mode, "classify_flat_tiny")
            role = TrainingRole(role_val)
            for fi, result in enumerate(results):
                artifact = result.get("artifact_path", "")
                if not artifact:
                    continue
                try:
                    publish_trained_model(
                        role=role,
                        artifact_path=artifact,
                        size="tiny" if "tiny" in mode else "n",
                        species=(
                            self.project_path.name if self.project_path else "species"
                        ),
                        model_info=mode,
                        trained_from_run_id="",
                        dataset_fingerprint="",
                        base_model=settings.get("base_model", "") if is_yolo else "",
                        scheme_name=scheme_name,
                        factor_index=fi if multi_head else None,
                        factor_name=(
                            scheme.factors[fi].name if (multi_head and scheme) else None
                        ),
                    )
                    from pathlib import Path as _Path

                    dialog.append_log(f"Published: {_Path(artifact).name}")

                    # Copy single-head YOLO model to project models/ as the "official" latest
                    if is_yolo and not multi_head and fi == 0 and self.project_path:
                        try:
                            import shutil

                            model_dir = self.project_path / "models"
                            model_dir.mkdir(parents=True, exist_ok=True)
                            dest = model_dir / "yolo_classifier_latest.pt"
                            shutil.copy2(artifact, dest)
                            self._yolo_model_path = dest
                            dialog.append_log(f"Saved to project models: {dest.name}")
                        except Exception as copy_exc:
                            dialog.append_log(f"Model copy warning: {copy_exc}")

                except Exception as exc:
                    dialog.append_log(f"Publish error: {exc}")

        dialog.start_btn.clicked.connect(_do_train)
        dialog.publish_btn.clicked.connect(_on_publish)
        dialog.exec()

    def export_dataset(self):
        """Export labeled dataset."""
        self._flush_pending_label_updates(force=True)

        if not self.project_path or not self.image_paths:
            QMessageBox.warning(
                self,
                "No Project Data",
                "Open a project with ingested images before exporting.",
            )
            return

        from .dialogs import ExportDialog

        default_output = str(self.project_path / "exports")
        dialog = ExportDialog(default_output=default_output, parent=self)
        if not dialog.exec():
            return

        settings = dialog.get_settings()

        label_to_index = {name: idx for idx, name in enumerate(self.classes)}
        class_names = {idx: name for idx, name in enumerate(self.classes)}
        include_unlabeled = settings.get("include_unlabeled", False)
        unlabeled_index = len(self.classes)
        if include_unlabeled:
            class_names[unlabeled_index] = "unlabeled"

        export_paths = []
        export_labels = []
        for path, label_name in zip(self.image_paths, self.image_labels):
            if label_name in label_to_index:
                export_paths.append(path)
                export_labels.append(label_to_index[label_name])
            elif include_unlabeled:
                export_paths.append(path)
                export_labels.append(unlabeled_index)

        if not export_paths:
            QMessageBox.warning(
                self,
                "No Exportable Data",
                "No labeled images available to export with the selected options.",
            )
            return

        from ..jobs.task_workers import ExportWorker

        worker = ExportWorker(
            image_paths=export_paths,
            labels=export_labels,
            output_path=Path(settings["output_dir"]),
            format=settings["format"],
            class_names=class_names,
            val_fraction=float(settings.get("val_fraction", 0.2)),
            test_fraction=float(settings.get("test_fraction", 0.0)),
            copy_files=bool(settings.get("copy_files", True)),
        )

        worker.signals.started.connect(
            lambda: self.status.showMessage("Starting dataset export...")
        )
        worker.signals.progress.connect(self.on_export_progress)
        worker.signals.success.connect(self.on_export_success)
        worker.signals.error.connect(self.on_export_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)

    def load_classifier_checkpoint(self):
        """Load a trained classifier checkpoint (embedding head or YOLO model)."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Open a ClassKit project first.",
            )
            return

        model_dir = self.project_path / "models"
        latest_checkpoint = model_dir / "classifier_latest.pt"
        yolo_checkpoint = model_dir / "yolo_classifier_latest.pt"
        selected_path = None

        if yolo_checkpoint.exists() and self._ask_yes_no(
            "Load YOLO Classifier",
            f"Load latest saved YOLO classifier?\n\n{yolo_checkpoint}",
        ):
            selected_path = yolo_checkpoint
        elif latest_checkpoint.exists() and self._ask_yes_no(
            "Load Latest Checkpoint",
            f"Load the latest saved classifier checkpoint?\n\n{latest_checkpoint}",
        ):
            selected_path = latest_checkpoint
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Classifier Checkpoint",
                str(model_dir if model_dir.exists() else self.project_path),
                "PyTorch Checkpoint (*.pt *.pth)",
            )
            if not file_path:
                return
            selected_path = Path(file_path)

        self._load_checkpoint_from_path(selected_path)

    def _load_checkpoint_from_path(self, path: Path):
        """Load a checkpoint, auto-detecting YOLO vs embedding-head vs tiny CNN format."""
        try:
            import torch

            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
            is_tiny_cnn = path.suffix.lower() == ".pth" or (
                isinstance(ckpt, dict) and "model_state_dict" in ckpt
            )

            if isinstance(ckpt, dict) and "model_state" in ckpt:
                # ── Embedding head format ──────────────────────────────
                from ..train.trainer import EmbeddingHeadTrainer

                input_dim = (
                    int(self.embeddings.shape[1])
                    if self.embeddings is not None
                    else 768
                )
                trainer = EmbeddingHeadTrainer(
                    model_type=ckpt.get("model_type", "linear"),
                    input_dim=input_dim,
                    num_classes=max(2, len(self.classes)),
                    device=(self._last_training_settings or {}).get("device", "cpu"),
                )
                trainer.load(path)
                self._trained_classifier = trainer
                self.status.showMessage(f"Loaded embedding head: {path.name}")

                if self.embeddings is not None:
                    probs = trainer.predict_proba(self.embeddings, calibrated=True)
                    self._model_probs = probs
                    self._model_class_names = list(self.classes)
                    self.umap_model_coords = None
                    self.pca_model_coords = None
                    self._show_model_umap = False
                    self._show_model_pca = False
                    self.btn_umap_embedding.setChecked(True)
                    self.btn_umap_model.setChecked(False)
                    if hasattr(self, "btn_pca_model"):
                        self.btn_pca_model.setChecked(False)
                    self.image_confidences = list(probs.max(axis=1).astype(float))
                    self.update_explorer_plot()
                    self._set_model_projection_buttons_enabled(True)
                    self._update_al_status()
                    self._evaluate_model_on_labeled()
                    QTimer.singleShot(100, self._replot_umap_model_space)

                QMessageBox.information(
                    self, "Checkpoint Loaded", f"Loaded embedding head: {path.name}"
                )
            elif is_tiny_cnn:
                arch = (
                    ckpt.get("arch", "tinyclassifier")
                    if isinstance(ckpt, dict)
                    else "tinyclassifier"
                )
                if arch != "tinyclassifier":
                    # ── Torchvision Custom CNN format ──────────────────
                    ckpt_names = ckpt.get("class_names")
                    input_size = ckpt.get("input_size", (224, 224))
                    sz = (
                        input_size[0]
                        if isinstance(input_size, (list, tuple))
                        else int(input_size)
                    )
                    resolved = ckpt_names or list(self.classes)
                    self._active_model_mode = "custom_cnn"
                    self.status.showMessage(
                        f"Loading Custom CNN ({arch}): {path.name}..."
                    )
                    self._run_torchvision_inference(
                        path,
                        class_names=resolved,
                        input_size=sz,
                        on_success=lambda r: (
                            self._evaluate_model_on_labeled(),
                            QTimer.singleShot(100, self._replot_umap_model_space),
                            QMessageBox.information(
                                self,
                                "Custom CNN Loaded",
                                f"Loaded: {path.name}\n"
                                f"Inference on {len(self.image_paths):,} images complete.\n"
                                "Metrics tab updated. Model UMAP computing...",
                            ),
                        ),
                    )
                else:
                    # ── Tiny CNN format (arch == 'tinyclassifier' or arch absent) ─
                    ckpt_names = ckpt.get("class_names")
                    db_names = None
                    if self.db_path:
                        try:
                            from ..store.db import ClassKitDB as _CKDb

                            for _entry in _CKDb(self.db_path).list_model_caches():
                                if str(path) in _entry.get("artifact_paths", []):
                                    db_names = _entry.get("class_names")
                                    break
                        except Exception:
                            pass
                    resolved = ckpt_names or db_names or list(self.classes)
                    self._active_model_mode = "tiny"
                    self.status.showMessage(f"Loading tiny CNN: {path.name}...")
                    self._run_tiny_inference(
                        path,
                        class_names=resolved,
                        on_success=lambda r: (
                            self._evaluate_model_on_labeled(),
                            QTimer.singleShot(100, self._replot_umap_model_space),
                            QMessageBox.information(
                                self,
                                "Tiny CNN Loaded",
                                f"Loaded: {path.name}\n"
                                f"Inference on {len(self.image_paths):,} images complete.\n"
                                "Metrics tab updated. Model UMAP computing...",
                            ),
                        ),
                    )
            else:
                # ── YOLO model format ──────────────────────────────────
                self._yolo_model_path = path
                self.status.showMessage(f"Loading YOLO model: {path.name}...")
                self._run_yolo_inference(
                    path,
                    on_success=lambda r: (
                        self._evaluate_model_on_labeled(),
                        QTimer.singleShot(100, self._replot_umap_model_space),
                        QMessageBox.information(
                            self,
                            "YOLO Model Loaded",
                            f"Loaded: {path.name}\n"
                            f"Inference on {len(self.image_paths):,} images complete.\n"
                            "Metrics tab updated. Model UMAP is being computed automatically.",
                        ),
                    ),
                )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Checkpoint Load Failed",
                f"Failed to load checkpoint:\n\n{exc}",
            )

    def on_export_progress(self, percentage, message):
        """Update progress for export workflow."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

    def on_export_success(self, result):
        """Handle successful export operation."""
        output_path = result.get("output_path")
        num_exported = int(result.get("num_exported", 0))
        num_classes = int(result.get("num_classes", 0))
        export_format = result.get("format", "unknown")

        self.status.showMessage(
            f"Export complete: {num_exported:,} images ({num_classes} classes)"
        )
        QMessageBox.information(
            self,
            "Export Complete",
            f"Export format: {export_format}\n"
            f"Images exported: {num_exported:,}\n"
            f"Classes: {num_classes}\n\n"
            f"Output:\n{output_path}",
        )

    def on_export_error(self, error_msg):
        """Handle export failure."""
        QMessageBox.critical(
            self,
            "Export Failed",
            f"Dataset export failed:\n\n{error_msg}",
        )
        self.status.showMessage("Export failed")

    # ================== UI Callbacks ==================

    def on_explorer_point_clicked(self, index):
        """Handle point click in explorer."""
        if self.explorer_mode != "labeling":
            self.status.showMessage("Selection is disabled outside Labeling mode")
            return
        self.selected_point_index = index
        self.hover_locked = True
        self.request_preview_for_index(index, source="click")
        self.request_update_explorer_selection(index)
        self.status.showMessage(f"Selected point {index}")

    def on_explorer_point_hovered(self, index):
        """Handle point hover in explorer."""
        if (
            self.hover_locked
            and self.selected_point_index is not None
            and index != self.selected_point_index
        ):
            return
        self.request_preview_for_index(index, source="hover")

    def on_label_assigned(self, label):
        """Handle label assignment."""
        idx = (
            self.selected_point_index
            if self.selected_point_index is not None
            else "n/a"
        )
        self.status.showMessage(f"Assigned label '{label}' to point {idx}")

    def on_next_image(self):
        """Navigate to next candidate or point."""
        if not self._begin_command():
            return
        next_action = None
        try:
            if self.selected_point_index is None:
                self.status.showMessage(
                    "No selected point; arrow keys are inactive until you click a point"
                )
                return

            pool = self._get_navigation_pool()
            if pool:
                if self.explorer_mode == "labeling":
                    # In labeling mode only advance to unlabeled candidates.
                    labels = self.image_labels or []
                    unlabeled = [i for i in pool if i >= len(labels) or not labels[i]]
                    unlabeled_set = set(unlabeled)
                    if not unlabeled:
                        # All candidates are labeled — prompt for next action.
                        self.selected_point_index = None
                        self.hover_locked = False
                        self.request_update_explorer_selection(None)
                        next_action = self._prompt_after_label_set_complete()
                    else:
                        # Find the next unlabeled item after the current position.
                        try:
                            current_pos = pool.index(self.selected_point_index)
                        except ValueError:
                            current_pos = -1
                        # Walk forward from current_pos wrapping only within unlabeled.
                        candidate = None
                        for offset in range(1, len(pool) + 1):
                            idx = pool[(current_pos + offset) % len(pool)]
                            if idx in unlabeled_set:
                                candidate = idx
                                break
                        if candidate is not None:
                            self.selected_point_index = candidate
                        else:
                            self.selected_point_index = None
                            self.hover_locked = False
                            self.request_update_explorer_selection(None)
                            next_action = self._prompt_after_label_set_complete()
                        # else: all unlabeled already tried, handled by empty check above
                else:
                    try:
                        current_pos = pool.index(self.selected_point_index)
                        next_pos = (current_pos + 1) % len(pool)
                    except ValueError:
                        next_pos = 0
                    self.selected_point_index = pool[next_pos]
            else:
                self.selected_point_index = min(
                    len(self.image_paths) - 1, (self.selected_point_index or 0) + 1
                )
            if self.selected_point_index is not None:
                self.hover_locked = True
                self.request_preview_for_index(self.selected_point_index, source="next")
                self.request_update_explorer_selection(self.selected_point_index)
        finally:
            self._end_command(0.08)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "train":
            QTimer.singleShot(0, self.train_classifier)

    def on_prev_image(self):
        """Navigate to previous candidate or point."""
        if not self._begin_command():
            return
        next_action = None
        try:
            if self.selected_point_index is None:
                self.status.showMessage(
                    "No selected point; arrow keys are inactive until you click a point"
                )
                return

            pool = self._get_navigation_pool()
            if pool:
                if self.explorer_mode == "labeling":
                    # In labeling mode only move through unlabeled candidates.
                    labels = self.image_labels or []
                    unlabeled = [i for i in pool if i >= len(labels) or not labels[i]]
                    unlabeled_set = set(unlabeled)
                    if not unlabeled:
                        self.selected_point_index = None
                        self.hover_locked = False
                        self.request_update_explorer_selection(None)
                        next_action = self._prompt_after_label_set_complete()
                    else:
                        try:
                            current_pos = pool.index(self.selected_point_index)
                        except ValueError:
                            current_pos = 0
                        candidate = None
                        for offset in range(1, len(pool) + 1):
                            idx = pool[(current_pos - offset) % len(pool)]
                            if idx in unlabeled_set:
                                candidate = idx
                                break
                        if candidate is not None:
                            self.selected_point_index = candidate
                        else:
                            self.selected_point_index = None
                            self.hover_locked = False
                            self.request_update_explorer_selection(None)
                            next_action = self._prompt_after_label_set_complete()
                else:
                    try:
                        current_pos = pool.index(self.selected_point_index)
                        prev_pos = (current_pos - 1) % len(pool)
                    except ValueError:
                        prev_pos = 0
                    self.selected_point_index = pool[prev_pos]
            else:
                current = (
                    self.selected_point_index
                    if self.selected_point_index is not None
                    else 0
                )
                self.selected_point_index = max(0, current - 1)
            if self.selected_point_index is not None:
                self.hover_locked = True
                self.request_preview_for_index(self.selected_point_index, source="prev")
                self.request_update_explorer_selection(self.selected_point_index)
        finally:
            self._end_command(0.08)

        if next_action == "sample":
            QTimer.singleShot(0, self.sample_candidates_for_labeling)
        elif next_action == "al":
            QTimer.singleShot(0, self._build_al_batch)
        elif next_action == "train":
            QTimer.singleShot(0, self.train_classifier)

    def refresh_view(self):
        """Refresh current view."""
        self.update_explorer_plot()
        self.update_context_panel()
        self.status.showMessage("Refreshed view")

    def on_enhance_toggled(self, checked: bool):
        """Handle CLAHE enhancement toggle."""
        if hasattr(self, "preview_canvas"):
            self.preview_canvas.use_clahe = checked
            # Refresh status
            status = "enabled" if checked else "disabled"
            self.status.showMessage(f"Contrast enhancement {status}", 2000)

            # Sync menu action
            if hasattr(self, "act_enhance") and self.act_enhance.isChecked() != checked:
                self.act_enhance.blockSignals(True)
                self.act_enhance.setChecked(checked)
                self.act_enhance.blockSignals(False)

            # Sync checkbox
            if hasattr(self, "cb_enhance") and self.cb_enhance.isChecked() != checked:
                self.cb_enhance.blockSignals(True)
                self.cb_enhance.setChecked(checked)
                self.cb_enhance.blockSignals(False)

            # Re-display current preview with the new setting
            if self.last_preview_index is not None:
                self.load_preview_for_index(self.last_preview_index, source="enhance")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About ClassKit",
            "<h2>ClassKit</h2>"
            + "<p>Active Learning Dataset Builder</p>"
            + "<p>Version 1.0</p>"
            + "<p>Sister framework to PoseKit for image classification "
            + "with smart active learning workflows.</p>",
        )

    # ================== Worker Callbacks ==================

    def on_ingest_progress(self, percentage, message):
        """Update progress for ingestion."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 1/5] Ingesting: {message}")

    def on_ingest_error(self, error_msg):
        """Handle ingestion error."""
        QMessageBox.critical(
            self, "Ingestion Error", f"Failed to ingest images:\n\n{error_msg}"
        )
        self.status.showMessage("Ingestion failed")

    def on_embedding_progress(self, percentage, message):
        """Update progress for embeddings."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 2/5] Embedding: {message}")

    def on_embedding_success(self, result):
        """Handle successful embedding computation."""
        self.embeddings = result["embeddings"]
        cached = result.get("cached", False)

        if cached:
            self.status.showMessage(
                f"Loaded {self.embeddings.shape[0]:,} cached embeddings"
            )
        else:
            self.status.showMessage(f"Computed {self.embeddings.shape[0]:,} embeddings")

        self.update_context_panel()
        self._auto_start_umap_if_ready()

    def on_embedding_error(self, error_msg):
        """Handle embedding error."""
        QMessageBox.critical(
            self, "Embedding Error", f"Failed to compute embeddings:\n\n{error_msg}"
        )
        self.status.showMessage("Embedding failed")

    def on_clustering_progress(self, percentage, message):
        """Update progress for clustering."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 3/5] Clustering: {message}")

    def on_clustering_success(self, result):
        """Handle successful clustering."""
        self.cluster_assignments = result["assignments"]
        self.candidate_indices = []
        n_clusters = len(set(self.cluster_assignments))

        if self.db_path:
            try:
                from ..store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                db.save_cluster_cache(
                    self.cluster_assignments,
                    result.get("centers"),
                    n_clusters,
                    result.get("method", "unknown"),
                )
            except Exception:
                pass

        self.status.showMessage(f"Clustered into {n_clusters} groups")
        self.update_explorer_plot()
        self.update_context_panel()
        self._auto_start_umap_if_ready()

    def on_clustering_error(self, error_msg):
        """Handle clustering error."""
        QMessageBox.critical(
            self, "Clustering Error", f"Failed to cluster:\n\n{error_msg}"
        )
        self.status.showMessage("Clustering failed")

    def on_umap_progress(self, percentage, message):
        """Update progress for UMAP."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(f"[Step 4/5] UMAP: {message}")

    def on_umap_success(self, result):
        """Handle successful UMAP computation."""
        self.umap_coords = result["coords"]

        if self.db_path:
            try:
                from ..store.db import ClassKitDB

                db = ClassKitDB(self.db_path)
                db.save_umap_cache(
                    self.umap_coords,
                    self.last_umap_params.get("n_neighbors", 15),
                    self.last_umap_params.get("min_dist", 0.1),
                )
            except Exception:
                pass

        self.status.showMessage("UMAP projection complete")
        self.update_explorer_plot(force_fit=True)
        self.update_context_panel()

    def on_umap_error(self, error_msg):
        """Handle UMAP error."""
        QMessageBox.critical(
            self, "UMAP Error", f"Failed to compute UMAP:\n\n{error_msg}"
        )
        self.status.showMessage("UMAP failed")
        self.update_context_panel()

    def _auto_start_umap_if_ready(self):
        """Automatically trigger UMAP if embeddings and clustering are complete."""
        if (
            self.embeddings is not None
            and self.cluster_assignments is not None
            and self.umap_coords is None
        ):
            self.status.showMessage("Auto-triggering UMAP projection...")
            self.compute_umap()

    def closeEvent(self, event):
        """Ensure pending label updates are flushed before close."""
        self._flush_pending_label_updates(force=True)
        super().closeEvent(event)

    # ================== Model Inference & Evaluation ==================

    def _persist_prediction_cache(self, probs, class_names: list, mode: str) -> None:
        """Save inference probs + class names to the project DB prediction cache."""
        if not self.db_path or probs is None:
            return
        try:
            from ..store.db import ClassKitDB

            ClassKitDB(self.db_path).save_prediction_cache(
                probs=probs,
                class_names=class_names or [],
                active_model_mode=mode,
            )
        except Exception:
            pass  # non-fatal — just means cache won't be available next session

    def _run_yolo_inference(self, model_path: Path, on_success=None):
        """Run YOLO inference on all images in background and update confidences."""
        if not self.image_paths:
            return
        compute_runtime = (self._last_training_settings or {}).get(
            "compute_runtime", "cpu"
        )

        from ..jobs.task_workers import YoloInferenceWorker

        worker = YoloInferenceWorker(
            model_path,
            self.image_paths,
            compute_runtime=compute_runtime,
            batch_size=64,
        )

        def _inference_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Inference done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "yolo"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_inference_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Inference error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Inference] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_tiny_inference(self, model_path: Path, class_names: list, on_success=None):
        """Run TinyCNN inference on all images in background and update confidences."""
        if not self.image_paths:
            return
        compute_runtime = (self._last_training_settings or {}).get(
            "compute_runtime", "cpu"
        )

        from ..jobs.task_workers import TinyCNNInferenceWorker

        worker = TinyCNNInferenceWorker(
            model_path,
            self.image_paths,
            class_names,
            compute_runtime=compute_runtime,
            batch_size=64,
        )

        def _tiny_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Tiny CNN done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "tiny"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_tiny_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Tiny CNN error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Tiny CNN] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_torchvision_inference(
        self,
        model_path: Path,
        class_names: list,
        input_size: int = 224,
        on_success=None,
    ):
        """Launch TorchvisionInferenceWorker and wire signals to the standard post-inference path."""
        if not self.image_paths:
            return

        from ..jobs.task_workers import TorchvisionInferenceWorker

        rt = (self._last_training_settings or {}).get("compute_runtime", "cpu")
        worker = TorchvisionInferenceWorker(
            model_path=model_path,
            image_paths=self.image_paths,
            class_names=class_names,
            input_size=input_size,
            compute_runtime=rt,
        )

        def _torchvision_success(result):
            self._model_probs = result["probs"]
            self._model_class_names = result["class_names"]
            self.umap_model_coords = None
            self.pca_model_coords = None
            self._show_model_umap = False
            self._show_model_pca = False
            self.btn_umap_embedding.setChecked(True)
            self.btn_umap_model.setChecked(False)
            if hasattr(self, "btn_pca_model"):
                self.btn_pca_model.setChecked(False)
            self.image_confidences = list(self._model_probs.max(axis=1).astype(float))
            self._set_model_projection_buttons_enabled(True)
            self._update_al_status()
            self.update_explorer_plot()
            self.status.showMessage(
                f"Custom CNN done: {len(self.image_paths):,} images, "
                f"{len(self._model_class_names)} classes"
            )
            self._active_model_mode = "custom_cnn"
            self._persist_prediction_cache(
                self._model_probs, self._model_class_names, "custom_cnn"
            )
            if on_success:
                on_success(result)

        worker.signals.success.connect(_torchvision_success)
        worker.signals.error.connect(
            lambda e: self.status.showMessage(f"Custom CNN error: {e}")
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[Custom CNN] {m}") if m else None
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)

    def _run_multihead_yolo_inference(self, model_paths: list, on_success=None):
        """Run YOLO inference on each factor model, concatenate log-prob columns."""
        import numpy as np

        if not model_paths or not self.image_paths:
            return

        collected: list = []  # list of (probs ndarray, class_names list) per head
        remaining: list = [len(model_paths)]

        def _head_done(result):
            collected.append((result["probs"], result["class_names"]))
            remaining[0] -= 1
            if remaining[0] == 0:
                # Merge all heads column-wise
                all_probs = np.concatenate([p for p, _ in collected], axis=1)
                all_names = [n for _, names in collected for n in names]
                merged = {"probs": all_probs, "class_names": all_names}
                self._model_probs = all_probs
                self._model_class_names = all_names
                self.umap_model_coords = None
                self.pca_model_coords = None
                self._show_model_umap = False
                self._show_model_pca = False
                self.btn_umap_embedding.setChecked(True)
                self.btn_umap_model.setChecked(False)
                if hasattr(self, "btn_pca_model"):
                    self.btn_pca_model.setChecked(False)
                self.image_confidences = list(all_probs.max(axis=1).astype(float))
                self._set_model_projection_buttons_enabled(True)
                self._update_al_status()
                self.update_explorer_plot()
                self.status.showMessage(
                    f"Multi-head inference done: {len(all_names)} combined classes"
                )
                self._persist_prediction_cache(all_probs, all_names, "yolo_multihead")
                if on_success:
                    on_success(merged)

        for mp in model_paths:
            self._run_yolo_inference(mp, on_success=_head_done)

    def _open_model_history(self):
        """Open the Model History dialog and load a selected past model."""
        if not self.db_path:
            QMessageBox.warning(self, "No Project", "Open a project first.")
            return
        from ..store.db import ClassKitDB
        from .dialogs import ModelHistoryDialog

        entries = ClassKitDB(self.db_path).list_model_caches()
        if not entries:
            QMessageBox.information(
                self, "No History", "No trained models found in project history yet."
            )
            return
        dlg = ModelHistoryDialog(
            entries,
            project_path=self.project_path,
            db_path=self.db_path,
            parent=self,
        )
        if dlg.exec() and dlg.selected_entry():
            self._load_model_from_cache_entry(dlg.selected_entry())

    def _load_model_from_cache_entry(self, entry: dict):
        """Load a model from a DB cache entry and run inference + UMAP."""
        mode = entry.get("mode", "")
        paths = entry.get("artifact_paths") or []
        if not paths or not Path(paths[0]).exists():
            QMessageBox.warning(
                self,
                "Missing Artifact",
                f"Model file not found:\n{paths[0] if paths else 'unknown'}",
            )
            return
        class_names = entry.get("class_names") or list(self.classes)

        def _after(r):
            self._evaluate_model_on_labeled()
            QTimer.singleShot(100, self._replot_umap_model_space)

        if "yolo" in mode:
            if mode.startswith("multihead"):
                all_paths = [Path(p) for p in paths if Path(p).exists()]
                self._active_model_mode = "yolo_multihead"
                self._run_multihead_yolo_inference(all_paths, on_success=_after)
            else:
                self._yolo_model_path = Path(paths[0])
                self._active_model_mode = "yolo"
                self._run_yolo_inference(Path(paths[0]), on_success=_after)
        else:
            self._active_model_mode = "tiny"
            self._run_tiny_inference(Path(paths[0]), class_names, on_success=_after)

    def _evaluate_model_on_labeled(self):
        """Compute metrics from _model_probs on all labeled images and update Metrics tab."""
        if self._model_probs is None:
            return
        labeled_indices = [i for i, lbl in enumerate(self.image_labels) if lbl]
        if len(labeled_indices) < 2:
            return
        try:
            import numpy as np

            from ..train.metrics import compute_metrics

            class_to_id = {c: i for i, c in enumerate(self.classes)}
            y_true = np.array(
                [class_to_id.get(self.image_labels[i], -1) for i in labeled_indices]
            )
            valid = y_true >= 0
            if valid.sum() < 2:
                return
            idx_arr = np.array(labeled_indices)[valid]
            y_true = y_true[valid]

            probs_rows = np.asarray(self._model_probs[idx_arr])

            # Align model probability columns to project class order by name.
            # Missing classes remain all-zero instead of throwing index errors.
            if self._model_class_names:
                name_to_col = {
                    str(name): i for i, name in enumerate(self._model_class_names)
                }
                probs_subset = np.zeros(
                    (len(idx_arr), len(self.classes)), dtype=probs_rows.dtype
                )
                for target_col, class_name in enumerate(self.classes):
                    source_col = name_to_col.get(str(class_name))
                    if source_col is None:
                        continue
                    if 0 <= int(source_col) < probs_rows.shape[1]:
                        probs_subset[:, target_col] = probs_rows[:, int(source_col)]
            else:
                # Fallback when model classes are unknown: clamp width safely.
                usable_cols = min(probs_rows.shape[1], len(self.classes))
                probs_subset = np.zeros(
                    (len(idx_arr), len(self.classes)), dtype=probs_rows.dtype
                )
                if usable_cols > 0:
                    probs_subset[:, :usable_cols] = probs_rows[:, :usable_cols]

            y_pred = probs_subset.argmax(axis=1)
            metrics = compute_metrics(y_pred, y_true, class_names=self.classes)
            self._update_metrics_display(metrics)
        except Exception as e:
            self.metrics_view.setPlainText(f"Evaluation error: {e}")

    def _update_metrics_display(self, metrics):
        """Update Metrics tab: text report + matplotlib confusion matrix / per-class bars."""
        from ..train.metrics import format_metrics_report

        self.metrics_view.setPlainText(format_metrics_report(metrics))
        self.tabs.setCurrentWidget(self.metrics_page)

        try:
            import io

            import matplotlib
            import numpy as np

            matplotlib.use("Agg")
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from PySide6.QtGui import QImage, QPixmap

            n_classes = len(metrics.per_class)
            fig = Figure(figsize=(12, 4.5), facecolor="#1e1e1e")
            names = [c.class_name for c in metrics.per_class]

            # ── Left: confusion matrix ──
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_facecolor("#252526")
            cm = metrics.confusion_matrix.astype(float)
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
            img = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax1.set_title(
                "Confusion Matrix (row-normalised)", color="#ddd", fontsize=10
            )
            ax1.set_xlabel("Predicted", color="#aaa", fontsize=9)
            ax1.set_ylabel("True", color="#aaa", fontsize=9)
            ax1.set_xticks(range(n_classes))
            ax1.set_yticks(range(n_classes))
            ax1.set_xticklabels(
                names, rotation=45, ha="right", color="#ccc", fontsize=8
            )
            ax1.set_yticklabels(names, color="#ccc", fontsize=8)
            for i in range(n_classes):
                for j in range(n_classes):
                    ax1.text(
                        j,
                        i,
                        f"{cm[i, j]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white" if cm_norm[i, j] > 0.5 else "#999",
                    )
            ax1.tick_params(colors="#aaa")
            cbar = fig.colorbar(img, ax=ax1)
            cbar.ax.tick_params(colors="#aaa")

            # ── Right: per-class P/R/F1 bars ──
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_facecolor("#252526")
            x = np.arange(n_classes)
            w = 0.25
            ax2.bar(
                x - w,
                [c.precision for c in metrics.per_class],
                w,
                label="Precision",
                color="#4e9de0",
            )
            ax2.bar(
                x,
                [c.recall for c in metrics.per_class],
                w,
                label="Recall",
                color="#5dbea3",
            )
            ax2.bar(
                x + w, [c.f1 for c in metrics.per_class], w, label="F1", color="#e07b4e"
            )
            ax2.axhline(
                metrics.accuracy,
                color="#ffff66",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
            )
            ax2.set_title(
                "Per-class Precision / Recall / F1", color="#ddd", fontsize=10
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                names, rotation=45, ha="right", color="#ccc", fontsize=8
            )
            ax2.set_ylim(0, 1.08)
            ax2.tick_params(colors="#aaa")
            ax2.spines[:].set_color("#555")
            legend = ax2.legend(
                loc="upper right", fontsize=8, facecolor="#252526", labelcolor="#ddd"
            )
            legend.get_frame().set_edgecolor("#555")

            fig.text(
                0.5,
                0.01,
                f"Accuracy: {metrics.accuracy:.3f}  |  Macro F1: {metrics.macro_f1:.3f}  |  "
                f"Weighted F1: {metrics.weighted_f1:.3f}  |  n={metrics.num_samples}",
                ha="center",
                color="#aaa",
                fontsize=9,
            )
            fig.tight_layout(rect=[0, 0.05, 1, 1])

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", dpi=110, bbox_inches="tight", facecolor="#1e1e1e"
            )
            buf.seek(0)
            qimg = QImage.fromData(buf.read())
            pixmap = QPixmap.fromImage(qimg)
            self.metrics_figure_label.setPixmap(pixmap)
            self.metrics_figure_label.setFixedSize(pixmap.size())
        except Exception as fig_exc:
            self.metrics_figure_label.setText(f"Figure error: {fig_exc}")

    # ================== Model-Space Projections ==================

    def _set_model_projection_buttons_enabled(self, enabled: bool) -> None:
        """Enable/disable model-space projection toggles together."""
        self.btn_umap_model.setEnabled(enabled)
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setEnabled(enabled)

    def _switch_projection_space(self, target: str) -> None:
        """Switch explorer between embedding UMAP, model UMAP, and model PCA."""
        target = str(target or "embedding")

        if target == "model_umap" and self.umap_model_coords is None:
            if self._model_probs is not None:
                self._replot_umap_model_space(auto_switch=True)
                return
            QMessageBox.information(
                self,
                "No Model UMAP",
                "Load a model checkpoint first. Model-space UMAP is computed automatically.",
            )
            target = "embedding"

        if target == "model_pca" and self.pca_model_coords is None:
            if self._model_probs is not None:
                self._replot_pca_model_space(auto_switch=True)
                return
            QMessageBox.information(
                self,
                "No Model PCA",
                "Load a model checkpoint first, then compute model-space PCA.",
            )
            target = "embedding"

        self._show_model_umap = target == "model_umap"
        self._show_model_pca = target == "model_pca"

        self.btn_umap_embedding.setChecked(target == "embedding")
        self.btn_umap_model.setChecked(target == "model_umap")
        if hasattr(self, "btn_pca_model"):
            self.btn_pca_model.setChecked(target == "model_pca")

        self.update_explorer_plot(force_fit=True)

        label = {
            "embedding": "Embeddings (UMAP)",
            "model_umap": "Model Logits (UMAP)",
            "model_pca": "Model Logits (PCA)",
        }.get(target, target)
        self.status.showMessage(f"Explorer space → {label}")

    def _replot_umap_model_space(self, auto_switch: bool = True):
        """Compute UMAP from current model probabilities and switch explorer to it."""
        if self._model_probs is None:
            QMessageBox.warning(
                self,
                "No Model Predictions",
                "Load a model first (Load Ckpt or train one).\n"
                "Predictions are computed automatically during model load.",
            )
            return

        from ..jobs.task_workers import LogitsUMAPWorker

        worker = LogitsUMAPWorker(
            self._model_probs,
            n_neighbors=self.last_umap_params.get("n_neighbors", 15),
            min_dist=self.last_umap_params.get("min_dist", 0.1),
        )

        def _on_model_umap_success(result):
            self.umap_model_coords = result["coords"]
            self._set_model_projection_buttons_enabled(True)
            if auto_switch:
                self._switch_projection_space("model_umap")
            self.status.showMessage(
                "Model-space UMAP ready — Explorer switched to model view"
            )
            # Persist so it's skipped on next project open
            if self.db_path:
                try:
                    from ..store.db import ClassKitDB

                    ClassKitDB(self.db_path).save_umap_cache(
                        result["coords"],
                        n_neighbors=self.last_umap_params.get("n_neighbors", 15),
                        min_dist=self.last_umap_params.get("min_dist", 0.1),
                        kind="model",
                    )
                except Exception:
                    pass

        worker.signals.success.connect(_on_model_umap_success)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(
                self, "UMAP Error", f"Model UMAP failed:\n{e}"
            )
        )
        worker.signals.progress.connect(
            lambda p, m: (
                self.progress_bar.setValue(p),
                self.status.showMessage(f"[Model UMAP] {m}") if m else None,
            )
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)
        self.status.showMessage("Computing model-space UMAP…")
        self.tabs.setCurrentIndex(0)

    def _replot_pca_model_space(self, auto_switch: bool = True):
        """Compute PCA from current model probabilities and optionally switch to it."""
        if self._model_probs is None:
            QMessageBox.warning(
                self,
                "No Model Predictions",
                "Load a model first (Load Ckpt or train one).",
            )
            return

        from ..jobs.task_workers import LogitsPCAWorker

        worker = LogitsPCAWorker(self._model_probs)

        def _on_model_pca_success(result):
            self.pca_model_coords = result["coords"]
            self._set_model_projection_buttons_enabled(True)
            if auto_switch:
                self._switch_projection_space("model_pca")
            self.status.showMessage("Model-space PCA ready")

            if self.db_path:
                try:
                    from ..store.db import ClassKitDB

                    ClassKitDB(self.db_path).save_umap_cache(
                        result["coords"],
                        n_neighbors=2,
                        min_dist=0.0,
                        kind="model_pca",
                    )
                except Exception:
                    pass

        worker.signals.success.connect(_on_model_pca_success)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(self, "PCA Error", f"Model PCA failed:\n{e}")
        )
        worker.signals.progress.connect(
            lambda p, m: (
                self.progress_bar.setValue(p),
                self.status.showMessage(f"[Model PCA] {m}") if m else None,
            )
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self._threadpool_start(worker)
        self.status.showMessage("Computing model-space PCA…")
        self.tabs.setCurrentIndex(0)

    # ================== Active Learning Batch Builder ==================

    def _update_al_status(self):
        """Refresh the Active Learning status label."""
        if not hasattr(self, "al_status_label"):
            return
        labeled = sum(1 for lbl in self.image_labels if lbl)
        total = len(self.image_paths)
        if self._active_model_mode == "yolo" and self._yolo_model_path:
            model_name = Path(self._yolo_model_path).name
        elif self._active_model_mode == "yolo_multihead":
            model_name = "multi-head YOLO"
        elif self._active_model_mode == "tiny":
            model_name = "tiny CNN"
        elif self._active_model_mode == "custom_cnn":
            model_name = "custom CNN"
        elif self._trained_classifier is not None:
            model_name = "embedding head"
        else:
            model_name = "none"
        preds_ready = self._model_probs is not None
        pct = f"{labeled / total:.0%}" if total else "—"
        _preds_span = (
            "<span style='color:#4ec943'>ready</span>"
            if preds_ready
            else "<span style='color:#888'>none</span>"
        )
        self.al_status_label.setText(
            f"Model: <b>{model_name}</b>  \u00b7  "
            f"Labeled: {labeled:,} / {total:,} ({pct})  \u00b7  "
            f"Predictions: {_preds_span}"
        )
        self.al_status_label.setTextFormat(Qt.RichText)

    def _build_al_batch(self):
        """Launch ALBatchWorker to select the next high-value labeling batch."""
        if self._model_probs is None:
            QMessageBox.warning(
                self,
                "No Predictions",
                "Load a trained model first.\n\n"
                "Use 'Load Ckpt' or finish a training run, then predictions will be "
                "computed automatically.",
            )
            return
        if self.embeddings is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        batch_size = (
            self.al_batch_spin.value() if hasattr(self, "al_batch_spin") else 50
        )
        labeled_mask = np.array([bool(lbl) for lbl in self.image_labels])

        from ..jobs.task_workers import ALBatchWorker

        worker = ALBatchWorker(
            embeddings=self.embeddings,
            probs=self._model_probs,
            labeled_mask=labeled_mask,
            cluster_assignments=self.cluster_assignments,
            batch_size=batch_size,
        )
        worker.signals.success.connect(self._on_al_batch_success)
        worker.signals.error.connect(
            lambda e: QMessageBox.critical(self, "AL Batch Error", e)
        )
        worker.signals.progress.connect(
            lambda p, m: self.status.showMessage(f"[AL] {m}") if m else None
        )
        self.al_candidates_badge.setText("  building…")
        self.al_candidate_list.clear()
        self.al_candidate_list.show()
        self.al_start_btn.setEnabled(False)
        self.al_highlight_btn.setEnabled(False)
        self._threadpool_start(worker)

    def _on_al_batch_success(self, result):
        """Populate the Batch Builder candidate list."""
        self._al_candidates = result["selected_indices"]
        breakdown = result["breakdown"]

        # Build per-index reason map
        reason_map = {}
        for reason, indices in breakdown.items():
            for idx in indices:
                reason_map[int(idx)] = reason

        self.al_candidate_list.clear()
        for idx in self._al_candidates:
            idx = int(idx)
            reason = reason_map.get(idx, "selected")
            path_name = (
                self.image_paths[idx].name
                if idx < len(self.image_paths)
                else f"img_{idx}"
            )
            conf = (
                float(self._model_probs[idx].max())
                if self._model_probs is not None
                else 0.0
            )
            pred_col = (
                int(self._model_probs[idx].argmax())
                if self._model_probs is not None
                else 0
            )
            pred_class = (
                self._model_class_names[pred_col]
                if self._model_class_names and pred_col < len(self._model_class_names)
                else f"cls_{pred_col}"
            )
            item_text = (
                f"#{idx:5d}  {path_name:<30s}  conf={conf:.3f}  "
                f"pred={pred_class:<12s}  [{reason}]"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, idx)
            if conf < 0.6:
                from PySide6.QtGui import QColor

                item.setForeground(QColor("#ff9944"))
            self.al_candidate_list.addItem(item)

        n = len(self._al_candidates)
        self.al_candidates_badge.setText(f"  {n} selected")
        self.al_candidate_list.show()
        self.al_start_btn.setEnabled(n > 0)
        self.al_highlight_btn.setEnabled(n > 0)
        self.status.showMessage(
            f"AL batch ready: {n} candidates — click ▶ Label to start"
        )

    def _start_labeling_al_batch(self):
        """Set AL candidates as the active labeling set and enter labeling mode."""
        if self._al_candidates is None or len(self._al_candidates) == 0:
            return
        new_candidates = [int(i) for i in self._al_candidates]
        # Preserve already-labeled images from the current candidate set so
        # they remain visible after switching to the AL batch.
        labels = self.image_labels or []
        seen = set(new_candidates)
        for i in self.candidate_indices:
            if i not in seen and i < len(labels) and labels[i]:
                new_candidates.append(i)
                seen.add(i)
        self.candidate_indices = new_candidates
        self.set_explorer_mode("labeling")
        self.tabs.setCurrentIndex(0)
        self.update_explorer_plot(force_fit=True)
        self.status.showMessage(
            f"Labeling {len(self.candidate_indices)} AL candidates — "
            "use label buttons or 1-9 keys"
        )

    def _highlight_al_batch_on_map(self):
        """Show AL candidates as the candidate set on the Explorer without entering labeling mode."""
        if self._al_candidates is None or len(self._al_candidates) == 0:
            return
        self.candidate_indices = [int(i) for i in self._al_candidates]
        self.tabs.setCurrentIndex(0)
        self.update_explorer_plot(force_fit=True)
        self.status.showMessage(
            f"Highlighted {len(self.candidate_indices)} AL candidates on map"
        )

    def _al_candidate_goto(self, item):
        """Jump explorer to the selected candidate when double-clicked in Batch Builder."""
        idx = item.data(Qt.UserRole)
        if idx is not None and 0 <= int(idx) < len(self.image_paths):
            self.selected_point_index = int(idx)
            self.tabs.setCurrentIndex(0)
            self.update_explorer_plot()
            self.load_preview_for_index(int(idx))
            self.load_preview_for_index(int(idx))

    def _run_apriltag_autolabel(self) -> None:
        """Open the AprilTag auto-label dialog and start the background worker."""
        from ..gui.dialogs import AprilTagAutoLabelDialog
        from ..jobs.task_workers import AprilTagAutoLabelWorker
        from ..presets import apriltag_preset
        from ..store.db import ClassKitDB

        if self.project_path is None:
            return

        # Collect all image paths (dialog needs them for preview)
        db = ClassKitDB(self.db_path)
        all_paths = [Path(p) for p in db.get_all_image_paths()]

        dlg = AprilTagAutoLabelDialog(image_paths=all_paths, parent=self)
        if dlg.exec() != AprilTagAutoLabelDialog.DialogCode.Accepted:
            return

        config = dlg.get_config()
        threshold = dlg.get_threshold()
        max_tag_id = config.max_tag_id

        # Warn if existing scheme will be replaced
        scheme_path = self.project_path / "scheme.json"
        if scheme_path.exists():
            reply = QMessageBox.question(
                self,
                "Replace Existing Scheme?",
                "A labeling scheme already exists.\n\n"
                "Replacing it will ERASE all existing labels.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # 1. Write the new scheme to disk
        scheme = apriltag_preset(config.family, max_tag_id)
        with open(scheme_path, "w") as f:
            json.dump(scheme.to_dict(), f, indent=2)

        # 2. Clear all existing labels (prevents stale labels under new scheme)
        db.clear_all_labels()

        # 3. Update self.classes and rebuild label buttons
        self.classes = scheme.factors[0].labels
        self.rebuild_label_buttons()

        # 4. Collect unlabeled image paths
        labels = db.get_all_labels()
        unlabeled = [p for p, lbl in zip(all_paths, labels) if lbl is None]

        if not unlabeled:
            QMessageBox.information(
                self, "Auto-label", "No unlabeled images to process."
            )
            return

        # 5. Start worker
        worker = AprilTagAutoLabelWorker(
            image_paths=unlabeled,
            config=config,
            threshold=threshold,
            db=db,
        )

        def _on_progress(pct: int, msg: str) -> None:
            self.statusBar().showMessage(f"AprilTag auto-label: {msg} ({pct}%)")

        def _on_success(result: dict) -> None:
            n_tag = result.get("n_labeled", 0)
            n_no = result.get("n_no_tag", 0)
            n_skip = result.get("n_skipped", 0)
            self.statusBar().showMessage(
                f"Auto-label complete: {n_tag} tagged, {n_no} no_tag, {n_skip} uncertain"
            )
            self.image_labels = db.get_all_labels()
            self._update_labeling_progress_indicator()

        def _on_error(msg: str) -> None:
            self.statusBar().showMessage(f"Auto-label error: {msg}")

        worker.signals.progress.connect(_on_progress)
        worker.signals.success.connect(_on_success)
        worker.signals.error.connect(_on_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self._threadpool_start(worker)
