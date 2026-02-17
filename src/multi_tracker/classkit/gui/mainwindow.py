"""
ClassKit Main Window - Polished and feature-complete UI
"""

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


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
        self.current_image_index = 0
        self.image_paths = []
        self.image_labels = []
        self.classes = ["class_1", "class_2"]
        self.selected_point_index = None
        self.candidate_indices = []
        self.round_labeled_indices = []
        self.explorer_mode = "explore"
        self.hover_locked = False
        self._label_shortcuts = []
        self.label_history = []
        self.last_assigned_stack = []
        self.last_cluster_result = None
        self.last_umap_params = {"n_neighbors": 15, "min_dist": 0.1}
        self.last_preview_index = None
        self._history_refresh_pending = False
        self._history_icon_cache = {}
        self._history_thumb_load_budget = 2
        self._command_busy = False
        self._command_block_until = 0.0
        self._command_squelch_pending = False
        self._pending_label_updates = {}
        self._autosave_interval_ms = 5000
        self._autosave_last_save_time = None
        self._autosave_heartbeat_on = False
        self._trained_classifier = None
        self._last_training_settings = None
        self._current_knn_neighbors = []

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
                font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
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

        file_menu.addSeparator()

        ingest_action = QAction("&Ingest Images...", self)
        ingest_action.setShortcut("Ctrl+I")
        ingest_action.triggered.connect(self.ingest_images)
        file_menu.addAction(ingest_action)

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

        predict_unlabeled_action = QAction("Predict &Unlabeled...", self)
        predict_unlabeled_action.triggered.connect(self.predict_unlabeled_images)
        compute_menu.addAction(predict_unlabeled_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        refresh_action = QAction("&Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)

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

        toolbar.addSeparator()

        # Data section
        ingest_btn = QAction("Ingest", self)
        ingest_btn.setStatusTip("Ingest images from folder")
        ingest_btn.triggered.connect(self.ingest_images)
        toolbar.addAction(ingest_btn)

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

        umap_btn = QAction("UMAP", self)
        umap_btn.setStatusTip("Compute UMAP projection")
        umap_btn.triggered.connect(self.compute_umap)
        toolbar.addAction(umap_btn)

        toolbar.addSeparator()

        # Training section
        train_btn = QAction("Train", self)
        train_btn.setStatusTip("Train classifier")
        train_btn.triggered.connect(self.train_classifier)
        toolbar.addAction(train_btn)

        load_ckpt_btn = QAction("Load Ckpt", self)
        load_ckpt_btn.setStatusTip("Load a classifier checkpoint")
        load_ckpt_btn.triggered.connect(self.load_classifier_checkpoint)
        toolbar.addAction(load_ckpt_btn)

        predict_btn = QAction("Predict", self)
        predict_btn.setStatusTip("Predict labels for unlabeled images")
        predict_btn.triggered.connect(self.predict_unlabeled_images)
        toolbar.addAction(predict_btn)

        toolbar.addSeparator()

        # Export section
        export_btn = QAction("Export", self)
        export_btn.setStatusTip("Export labeled dataset")
        export_btn.triggered.connect(self.export_dataset)
        toolbar.addAction(export_btn)

    def setup_central_widget(self):
        """Setup main UI layout for UMAP exploration and fast labeling."""
        self.splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.splitter)

        from .widgets.explorer import ExplorerView
        from .widgets.image_viewer import ImageCanvas

        # 1) Left: project metadata + settings + label controls
        self.context_panel = QWidget()
        self.context_panel.setFixedWidth(380)
        self.context_layout = QVBoxLayout(self.context_panel)
        self.context_layout.setContentsMargins(16, 16, 16, 16)
        self.context_layout.setSpacing(10)

        title_label = QLabel(
            "<b style='font-size: 18px; color: #ffffff;'>Project Metadata & Settings</b>"
        )
        self.context_layout.addWidget(title_label)

        self.context_info = QLabel(
            "<div style='line-height: 1.65;'>"
            "No project loaded.<br><br>"
            "Workflow:<br>"
            "1) Ingest images<br>"
            "2) Compute embeddings<br>"
            "3) Cluster + UMAP<br>"
            "4) Sample candidates<br>"
            "5) Hover + label via shortcuts<br>"
            "6) Retrain and iterate"
            "</div>"
        )
        self.context_info.setWordWrap(True)
        self.context_info.setStyleSheet(
            "padding: 14px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c;"
        )
        self.context_layout.addWidget(self.context_info)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("<b>UMAP View Mode:</b>"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("Explore (cluster colors)", "explore")
        self.view_mode_combo.addItem("Labeling (assigned labels)", "labeling")
        self.view_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        mode_row.addWidget(self.view_mode_combo)
        self.context_layout.addLayout(mode_row)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("<b>Candidates / cluster:</b>"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(6)
        sample_row.addWidget(self.sample_spin)
        self.sample_btn = QPushButton("Sample Labeling Set")
        self.sample_btn.clicked.connect(self.sample_candidates_for_labeling)
        sample_row.addWidget(self.sample_btn)
        self.context_layout.addLayout(sample_row)

        autosave_row = QHBoxLayout()
        autosave_row.addWidget(QLabel("<b>Autosave (sec):</b>"))
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 300)
        self.autosave_spin.setValue(max(1, self._autosave_interval_ms // 1000))
        self.autosave_spin.valueChanged.connect(self.on_autosave_interval_changed)
        autosave_row.addWidget(self.autosave_spin)
        self.context_layout.addLayout(autosave_row)

        self.selection_info = QLabel(
            "<div style='line-height:1.5; color:#aaaaaa;'>"
            "No point selected.<br>"
            "Click a UMAP point to select.<br>"
            "Hover points for quick preview."
            "</div>"
        )
        self.selection_info.setWordWrap(True)
        self.selection_info.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px;"
        )
        self.context_layout.addWidget(self.selection_info)

        self.context_layout.addWidget(QLabel("<b>kNN Neighbors</b>"))
        self.knn_info = QTextEdit()
        self.knn_info.setReadOnly(True)
        self.knn_info.setMaximumHeight(140)
        self.knn_info.setStyleSheet(
            "padding: 8px; background-color: #252526; border-radius: 6px;"
        )
        self.knn_info.setHtml(
            "<span style='color:#9f9f9f;'>Select a point in labeling mode to view nearest neighbors.</span>"
        )
        self.context_layout.addWidget(self.knn_info)

        knn_actions = QHBoxLayout()
        self.knn_jump_btn = QPushButton("Jump to Nearest")
        self.knn_jump_btn.clicked.connect(self.jump_to_nearest_neighbor)
        self.knn_jump_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_jump_btn)

        self.knn_bulk_btn = QPushButton("Bulk Label 5 Nearest")
        self.knn_bulk_btn.clicked.connect(self.bulk_label_nearest_neighbors)
        self.knn_bulk_btn.setEnabled(False)
        knn_actions.addWidget(self.knn_bulk_btn)
        self.context_layout.addLayout(knn_actions)

        self.label_buttons_container = QWidget()
        self.label_buttons_layout = QGridLayout(self.label_buttons_container)
        self.label_buttons_layout.setSpacing(8)
        self.context_layout.addWidget(QLabel("<b>Class Labels (keyboard 1-9)</b>"))
        self.context_layout.addWidget(self.label_buttons_container)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev Unlabeled ←")
        self.prev_btn.clicked.connect(self.on_prev_image)
        self.next_btn = QPushButton("Next Unlabeled →")
        self.next_btn.clicked.connect(self.on_next_image)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        self.context_layout.addLayout(nav_row)

        self.context_layout.addWidget(QLabel("<b>Edit Classes (one per line)</b>"))
        self.class_editor = QTextEdit()
        self.class_editor.setPlaceholderText("class_1\nclass_2")
        self.class_editor.setMaximumHeight(120)
        self.context_layout.addWidget(self.class_editor)
        self.apply_classes_btn = QPushButton("Apply Class Changes")
        self.apply_classes_btn.clicked.connect(self.apply_class_changes)
        self.context_layout.addWidget(self.apply_classes_btn)

        self.shortcut_help = QLabel(
            "<div style='line-height:1.55; color:#bcbcbc;'>"
            "Shortcuts:<br>"
            "• 1-9: assign class to selected point<br>"
            "• Space: sample next candidate set<br>"
            "• E: Explore mode, L: Labeling mode"
            "</div>"
        )
        self.shortcut_help.setWordWrap(True)
        self.shortcut_help.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px;"
        )
        self.context_layout.addWidget(self.shortcut_help)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.context_layout.addWidget(self.progress_bar)
        self.context_layout.addStretch()

        # 2) Center: main UMAP explorer
        self.center_panel = QWidget()
        center_layout = QVBoxLayout(self.center_panel)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        self.explorer = ExplorerView()
        self.explorer.point_clicked.connect(self.on_explorer_point_clicked)
        self.explorer.point_hovered.connect(self.on_explorer_point_hovered)
        self.explorer.empty_double_clicked.connect(
            self.on_explorer_background_double_click
        )
        center_layout.addWidget(self.explorer, 1)

        self.history_title = QLabel("<b>Recent Labels (click to undo + relabel)</b>")
        center_layout.addWidget(self.history_title)

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
        center_layout.addWidget(self.history_scroll)

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

        self.preview_canvas = ImageCanvas()
        self.preview_canvas.setMinimumHeight(320)
        preview_layout.addWidget(self.preview_canvas, 1)

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
        self.splitter.addWidget(self.center_panel)
        self.splitter.addWidget(self.preview_panel)
        self.splitter.setSizes([380, 920, 420])

        self.setup_label_shortcuts()
        self.rebuild_label_buttons()
        self.class_editor.setPlainText("\n".join(self.classes))

    def create_placeholder_page(self, title: str, description: str) -> QWidget:
        """Create a styled placeholder page."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)

        title_label = QLabel(
            f"<h1 style='color: #ffffff; font-weight: 300;'>{title}</h1>"
        )
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description.replace("\\n", "<br>"))
        desc_label.setAlignment(Qt.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "color: #aaaaaa; font-size: 12px; padding: 24px; "
            + "background-color: #252526; border-radius: 8px; line-height: 1.8;"
        )
        layout.addWidget(desc_label)

        layout.addStretch()
        return widget

    def setup_statusbar(self):
        """Setup status bar."""
        self.status = QStatusBar()
        self.setStatusBar(self.status)
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

                self.classes = project_info.get("classes", []) or ["class_1", "class_2"]
                self.rebuild_label_buttons()
                self.setup_label_shortcuts()

                self.update_context_panel()
                self.status.showMessage(f"Created project: {project_info['name']}")
                QMessageBox.information(
                    self,
                    "Project Created",
                    f"Successfully created project:\\n\\n{self.project_path}",
                )
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

    def load_project_data(self):
        """Load project data from database."""
        if not self.db_path:
            return

        try:
            from pathlib import Path

            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)

            # Load image paths
            path_strings = db.get_all_image_paths()
            self.image_paths = [Path(p) for p in path_strings]
            self.image_labels = db.get_all_labels()

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
            else:
                self.classes = ["class_1", "class_2"]

            self.rebuild_label_buttons()
            self.setup_label_shortcuts()
            self.class_editor.setPlainText("\n".join(self.classes))
            self.request_refresh_label_history_strip()
            self._pending_label_updates = {}
            self._autosave_last_save_time = None
            self._update_autosave_heartbeat_text()
            self.try_autoload_cached_artifacts(db)
            self.update_explorer_plot()

            self.status.showMessage(
                f"Loaded {len(self.image_paths):,} images from database"
            )
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
        info_html += f"<b>Current Mode:</b> {'Explore' if self.explorer_mode == 'explore' else 'Labeling'}<br>"
        info_html += (
            f"<b>Candidate Set:</b> {len(self.candidate_indices)} points"
            if self.candidate_indices
            else "<b>Candidate Set:</b> none sampled"
        )
        info_html += "</div>"
        self.context_info.setText(info_html)

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
            pending_items = list(self._pending_label_updates.items())
            for image_path, label in pending_items:
                db.update_label(image_path, label)

            self._pending_label_updates = {}
            self._autosave_last_save_time = time.time()
            if self._autosave_timer.isActive():
                self._autosave_timer.stop()
        except Exception as exc:
            self.status.showMessage(f"Autosave failed: {exc}")

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

            self.update_explorer_plot(force_fit=True)
            self.update_context_panel()
        except Exception:
            pass

    def setup_label_shortcuts(self):
        """Create keyboard shortcuts for labeling and mode switching."""
        for shortcut in self._label_shortcuts:
            shortcut.setParent(None)
        self._label_shortcuts = []

        for i, class_name in enumerate(self.classes[:9], start=1):
            shortcut = QShortcut(QKeySequence(str(i)), self)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(
                lambda c=class_name: self.assign_label_to_selected(c)
            )
            self._label_shortcuts.append(shortcut)

        explore_shortcut = QShortcut(QKeySequence("E"), self)
        explore_shortcut.setAutoRepeat(False)
        explore_shortcut.activated.connect(lambda: self.set_explorer_mode("explore"))
        self._label_shortcuts.append(explore_shortcut)

        label_shortcut = QShortcut(QKeySequence("L"), self)
        label_shortcut.setAutoRepeat(False)
        label_shortcut.activated.connect(lambda: self.set_explorer_mode("labeling"))
        self._label_shortcuts.append(label_shortcut)

        sample_shortcut = QShortcut(QKeySequence("Space"), self)
        sample_shortcut.setAutoRepeat(False)
        sample_shortcut.activated.connect(self.sample_candidates_for_labeling)
        self._label_shortcuts.append(sample_shortcut)

        prev_shortcut = QShortcut(QKeySequence("Left"), self)
        prev_shortcut.setAutoRepeat(False)
        prev_shortcut.activated.connect(self.on_prev_image)
        self._label_shortcuts.append(prev_shortcut)

        next_shortcut = QShortcut(QKeySequence("Right"), self)
        next_shortcut.setAutoRepeat(False)
        next_shortcut.activated.connect(self.on_next_image)
        self._label_shortcuts.append(next_shortcut)

        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
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

    def apply_class_changes(self):
        """Apply class name edits and persist to project config."""
        text = self.class_editor.toPlainText().strip()
        updated = [line.strip() for line in text.split("\n") if line.strip()]
        if not updated:
            QMessageBox.warning(
                self, "Invalid Classes", "Please provide at least one class name."
            )
            return

        self.classes = updated
        self.rebuild_label_buttons()
        self.setup_label_shortcuts()

        if self.project_path:
            project_config_path = self.project_path / "project.json"
            config = {}
            if project_config_path.exists():
                with open(project_config_path, "r") as f:
                    config = json.load(f)
            config["classes"] = self.classes
            with open(project_config_path, "w") as f:
                json.dump(config, f, indent=2)

        self.update_context_panel()
        self.status.showMessage(f"Updated classes ({len(self.classes)})")

    def refresh_label_history_strip(self):
        """Refresh bottom strip of recently labeled items for quick undo/relabel."""
        self._history_refresh_pending = False
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
        self._history_refresh_pending = True
        self._history_refresh_timer.start()

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

    def _set_label_for_index(self, index: int, label):
        """Set or clear label for an index and persist to DB."""
        if index < 0 or index >= len(self.image_paths):
            return

        if self.db_path:
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

    def _get_unlabeled_navigation_pool(self):
        """Return indices eligible for next/prev navigation in labeling context."""
        if self.candidate_indices:
            return [
                idx
                for idx in self.candidate_indices
                if idx < len(self.image_labels) and not self.image_labels[idx]
            ]
        return [idx for idx, label in enumerate(self.image_labels) if not label]

    def rebuild_label_buttons(self):
        """Rebuild class buttons shown in the left settings panel."""
        for i in reversed(range(self.label_buttons_layout.count())):
            widget = self.label_buttons_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        for i, class_name in enumerate(self.classes[:9], start=1):
            button = QPushButton(f"{i}: {class_name}")
            button.setStyleSheet("text-align: left;")
            button.clicked.connect(
                lambda checked=False, c=class_name: self.assign_label_to_selected(c)
            )
            row = (i - 1) // 2
            col = (i - 1) % 2
            self.label_buttons_layout.addWidget(button, row, col)

    def set_explorer_mode(self, mode: str):
        """Set explorer mode to explore or labeling."""
        if mode not in {"explore", "labeling"}:
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
        self.status.showMessage(
            f"Mode: {'Explore' if mode == 'explore' else 'Labeling'}"
        )

    def on_view_mode_changed(self, index):
        """Handle mode selection from combo box."""
        mode = self.view_mode_combo.itemData(index)
        self.set_explorer_mode(mode)

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

            self.candidate_indices = sampled
            self.round_labeled_indices = []
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
        self.current_image_index = neighbor_index
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
            self.status.showMessage("Label assignment is disabled in Explore mode")
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
            if self.candidate_indices:
                self.selected_point_index = self.candidate_indices[0]
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
        elif next_action == "train":
            QTimer.singleShot(0, self.train_classifier)

    def _prompt_after_label_set_complete(self) -> str | None:
        """Prompt next step when all points in current sampled set are labeled."""
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Information)
        message.setWindowTitle("Labeling Set Complete")
        message.setText("All points in the current labeling set are labeled.")
        message.setInformativeText(
            "Choose what to do next: sample another set per cluster or move on to training."
        )
        sample_btn = message.addButton("Sample Another Set", QMessageBox.AcceptRole)
        train_btn = message.addButton("Train Classifier", QMessageBox.ActionRole)
        message.addButton(QMessageBox.Close)
        message.setDefaultButton(sample_btn)
        message.exec()

        clicked = message.clickedButton()
        if clicked == sample_btn:
            return "sample"
        if clicked == train_btn:
            return "train"
        return None

    def update_explorer_plot(self, force_fit: bool = False):
        """Refresh explorer points with current mode, labels, and candidate emphasis."""
        if self.umap_coords is None or not hasattr(self, "explorer"):
            return

        if self.explorer_mode == "explore":
            color_values = self.cluster_assignments
            candidate_indices = []
        else:
            color_values = self.image_labels
            candidate_indices = self.candidate_indices

        if not force_fit:
            if self.explorer.update_state(
                labels=color_values,
                candidate_indices=candidate_indices,
                round_labeled_indices=self.round_labeled_indices,
                selected_index=self.selected_point_index,
                labeling_mode=(self.explorer_mode == "labeling"),
            ):
                return

        self.explorer.set_data(
            self.umap_coords,
            color_values,
            candidate_indices=candidate_indices,
            round_labeled_indices=self.round_labeled_indices,
            selected_index=self.selected_point_index,
            labeling_mode=(self.explorer_mode == "labeling"),
            preserve_view=(not force_fit),
        )

    # ================== Data Operations ==================

    def ingest_images(self):
        """Ingest images from a folder."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Please create or open a project first.\\n\\n"
                + "Use File → New Project to get started.",
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", str(Path.home())
        )

        if folder:
            from ..jobs.task_workers import IngestWorker

            worker = IngestWorker(Path(folder), self.db_path)
            worker.signals.started.connect(
                lambda: self.status.showMessage("Ingesting images...")
            )
            worker.signals.progress.connect(self.on_ingest_progress)
            worker.signals.success.connect(self.on_ingest_success)
            worker.signals.error.connect(self.on_ingest_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.threadpool.start(worker)

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
            self.threadpool.start(worker)

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
            n_clusters, use_gpu = dialog.get_settings()

            from ..jobs.task_workers import ClusteringWorker

            worker = ClusteringWorker(self.embeddings, n_clusters, use_gpu)
            worker.signals.started.connect(
                lambda: self.status.showMessage("Clustering data...")
            )
            worker.signals.progress.connect(self.on_clustering_progress)
            worker.signals.success.connect(self.on_clustering_success)
            worker.signals.error.connect(self.on_clustering_error)
            worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.threadpool.start(worker)

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
        self.threadpool.start(worker)

    def train_classifier(self):
        """Train a classifier on labeled data."""
        self._flush_pending_label_updates(force=True)

        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Compute embeddings before training a classifier.",
            )
            return

        if not self.image_labels or len(self.image_labels) != len(self.image_paths):
            QMessageBox.warning(
                self,
                "No Labels",
                "No valid labels found. Label some points first.",
            )
            return

        label_to_index = {name: idx for idx, name in enumerate(self.classes)}
        labeled_indices = []
        numeric_labels = []
        for idx, label in enumerate(self.image_labels):
            if label and label in label_to_index:
                labeled_indices.append(idx)
                numeric_labels.append(label_to_index[label])

        if len(labeled_indices) < 4:
            QMessageBox.warning(
                self,
                "Not Enough Labels",
                "Need at least 4 labeled examples before training.",
            )
            return

        class_counts = {}
        for value in numeric_labels:
            class_counts[value] = class_counts.get(value, 0) + 1
        if len(class_counts) < 2:
            QMessageBox.warning(
                self,
                "Need Multiple Classes",
                "Training requires labels from at least two classes.",
            )
            return

        from .dialogs import TrainingDialog

        dialog = TrainingDialog(self)
        if not dialog.exec():
            return
        settings = dialog.get_settings()
        self._last_training_settings = settings

        x = self.embeddings[np.array(labeled_indices, dtype=np.int64)]
        y = np.array(numeric_labels, dtype=np.int64)

        train_indices, val_indices = self._split_train_val_indices(
            y,
            val_fraction=settings["val_fraction"],
        )

        train_embeddings = x[train_indices]
        train_labels = y[train_indices]
        val_embeddings = x[val_indices] if len(val_indices) > 0 else None
        val_labels = y[val_indices] if len(val_indices) > 0 else None

        from ..jobs.task_workers import TrainingWorker

        worker = TrainingWorker(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            val_embeddings=val_embeddings,
            val_labels=val_labels,
            num_classes=len(self.classes),
            model_type=settings["model_type"],
            device=settings["device"],
            hidden_dim=settings["hidden_dim"],
            dropout=settings["dropout"],
            batch_size=settings["batch_size"],
            epochs=settings["epochs"],
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            early_stop_patience=settings["early_stop_patience"],
            calibrate=settings["calibrate"],
        )

        worker.signals.started.connect(
            lambda: self.status.showMessage("Training embedding head classifier...")
        )
        worker.signals.progress.connect(self.on_training_progress)
        worker.signals.success.connect(self.on_training_success)
        worker.signals.error.connect(self.on_training_error)
        worker.signals.finished.connect(lambda: self.progress_bar.setVisible(False))

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.threadpool.start(worker)

    def _split_train_val_indices(self, y: np.ndarray, val_fraction: float):
        """Create stratified train/val index split from label vector."""
        rng = np.random.default_rng(42)
        train_parts = []
        val_parts = []

        for class_id in np.unique(y):
            class_indices = np.where(y == class_id)[0]
            rng.shuffle(class_indices)

            if len(class_indices) < 3 or val_fraction <= 0.0:
                train_parts.append(class_indices)
                continue

            n_val = int(round(len(class_indices) * val_fraction))
            n_val = max(1, n_val)
            n_val = min(n_val, len(class_indices) - 1)

            val_parts.append(class_indices[:n_val])
            train_parts.append(class_indices[n_val:])

        train_idx = (
            np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
        )
        val_idx = (
            np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
        )
        return train_idx, val_idx

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
        self.threadpool.start(worker)

    def load_classifier_checkpoint(self):
        """Load a trained classifier checkpoint from the project."""
        if not self.project_path:
            QMessageBox.warning(
                self,
                "No Project",
                "Open a ClassKit project first.",
            )
            return

        model_dir = self.project_path / "models"
        latest_checkpoint = model_dir / "classifier_latest.pt"
        selected_path = None

        if latest_checkpoint.exists() and self._ask_yes_no(
            "Load Latest Checkpoint",
            "Load the latest saved classifier checkpoint?\n\n" f"{latest_checkpoint}",
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

        try:
            from ..train.trainer import EmbeddingHeadTrainer

            input_dim = (
                int(self.embeddings.shape[1]) if self.embeddings is not None else 768
            )
            trainer = EmbeddingHeadTrainer(
                model_type="linear",
                input_dim=input_dim,
                num_classes=max(2, len(self.classes)),
                device=(self._last_training_settings or {}).get("device", "cpu"),
            )
            trainer.load(selected_path)
            self._trained_classifier = trainer
            self.status.showMessage(
                f"Loaded classifier checkpoint: {selected_path.name}"
            )
            QMessageBox.information(
                self,
                "Checkpoint Loaded",
                f"Loaded classifier checkpoint:\n{selected_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Checkpoint Load Failed",
                f"Failed to load checkpoint:\n\n{exc}",
            )

    def predict_unlabeled_images(self):
        """Run classifier predictions on unlabeled items and persist confidences."""
        self._flush_pending_label_updates(force=True)

        if self.embeddings is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Compute embeddings before running predictions.",
            )
            return

        if self._trained_classifier is None:
            QMessageBox.warning(
                self,
                "No Classifier",
                "Train or load a classifier checkpoint first.",
            )
            return

        unlabeled_indices = [
            idx for idx, label in enumerate(self.image_labels) if not label
        ]
        if not unlabeled_indices:
            QMessageBox.information(
                self,
                "No Unlabeled Images",
                "All images already have labels.",
            )
            return

        try:
            idx_array = np.array(unlabeled_indices, dtype=np.int64)
            unlabeled_embeddings = self.embeddings[idx_array]
            probs = self._trained_classifier.predict_proba(
                unlabeled_embeddings,
                calibrated=True,
            )

            pred_ids = probs.argmax(axis=1).astype(int)
            confidences = probs.max(axis=1).astype(float)
            pred_labels = [
                (
                    self.classes[pred_idx]
                    if pred_idx < len(self.classes)
                    else f"class_{pred_idx}"
                )
                for pred_idx in pred_ids
            ]

            from ..store.db import ClassKitDB

            db = ClassKitDB(self.db_path)
            pred_paths = [str(self.image_paths[idx]) for idx in unlabeled_indices]
            db.save_predictions(
                paths=pred_paths,
                predicted_labels=pred_labels,
                predicted_indices=pred_ids.tolist(),
                confidences=confidences.tolist(),
            )

            predictions_dir = self.project_path / "models"
            predictions_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = predictions_dir / "predictions_latest.csv"
            with open(artifact_path, "w") as f:
                f.write("image_path,predicted_label,predicted_index,confidence\n")
                for path, label, pred_idx, conf in zip(
                    pred_paths, pred_labels, pred_ids.tolist(), confidences.tolist()
                ):
                    safe_path = path.replace(",", "%2C")
                    safe_label = str(label).replace(",", "%2C")
                    f.write(f"{safe_path},{safe_label},{pred_idx},{conf:.6f}\n")

            high_conf = int(np.sum(confidences >= 0.8))
            self.status.showMessage(
                f"Predicted {len(unlabeled_indices):,} unlabeled images (>=0.8 conf: {high_conf:,})"
            )
            QMessageBox.information(
                self,
                "Prediction Complete",
                f"Scored {len(unlabeled_indices):,} unlabeled images.\n"
                f"High confidence (>= 0.8): {high_conf:,}\n\n"
                f"Saved predictions to:\n{artifact_path}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Prediction Failed",
                f"Failed to run predictions:\n\n{exc}",
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

    def on_nav_changed(self, index):
        """Legacy callback retained for compatibility."""
        self.status.showMessage("Explorer layout is active")

    def on_explorer_point_clicked(self, index):
        """Handle point click in explorer."""
        if self.explorer_mode != "labeling":
            self.status.showMessage("Selection is disabled in Explore mode")
            return
        self.selected_point_index = index
        self.current_image_index = index
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
        try:
            if self.selected_point_index is None:
                self.status.showMessage(
                    "No selected point; arrow keys are inactive until you click a point"
                )
                return

            pool = self._get_unlabeled_navigation_pool()
            if pool:
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
            self.hover_locked = True
            self.request_preview_for_index(self.selected_point_index, source="next")
            self.request_update_explorer_selection(self.selected_point_index)
        finally:
            self._end_command(0.08)

    def on_prev_image(self):
        """Navigate to previous candidate or point."""
        if not self._begin_command():
            return
        try:
            if self.selected_point_index is None:
                self.status.showMessage(
                    "No selected point; arrow keys are inactive until you click a point"
                )
                return

            pool = self._get_unlabeled_navigation_pool()
            if pool:
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
            self.hover_locked = True
            self.request_preview_for_index(self.selected_point_index, source="prev")
            self.request_update_explorer_selection(self.selected_point_index)
        finally:
            self._end_command(0.08)

    def refresh_view(self):
        """Refresh current view."""
        self.update_explorer_plot()
        self.update_context_panel()
        self.status.showMessage("Refreshed view")

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
        self.status.showMessage(message)

    def on_ingest_success(self, result):
        """Handle successful ingestion."""
        num_images = result.get("num_images", 0)
        QMessageBox.information(
            self, "Ingestion Complete", f"Successfully ingested {num_images:,} images."
        )
        self.status.showMessage(f"Ingested {num_images:,} images")
        self.load_project_data()
        self.update_context_panel()

    def on_ingest_error(self, error_msg):
        """Handle ingestion error."""
        QMessageBox.critical(
            self, "Ingestion Error", f"Failed to ingest images:\\n\\n{error_msg}"
        )
        self.status.showMessage("Ingestion failed")

    def on_embedding_progress(self, percentage, message):
        """Update progress for embeddings."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

    def on_embedding_success(self, result):
        """Handle successful embedding computation."""
        self.embeddings = result["embeddings"]
        dimension = result["dimension"]
        cached = result.get("cached", False)

        if cached:
            metadata = result.get("metadata", {})
            timestamp = metadata.get("timestamp", "unknown")
            QMessageBox.information(
                self,
                "Embeddings Loaded",
                f"Loaded cached embeddings from {timestamp}\\n\\n"
                + f"Shape: {self.embeddings.shape[0]:,} × {dimension}\\n"
                + f"Model: {metadata.get('model_name', 'unknown')}\\n"
                + f"Device: {metadata.get('device', 'unknown')}",
            )
            self.status.showMessage(
                f"Loaded {self.embeddings.shape[0]:,} cached embeddings"
            )
        else:
            QMessageBox.information(
                self,
                "Embeddings Complete",
                "Successfully computed embeddings.\\n\\n"
                + f"Shape: {self.embeddings.shape[0]:,} × {dimension}\\n"
                + "Cached for future use",
            )
            self.status.showMessage(f"Computed {self.embeddings.shape[0]:,} embeddings")

        self.update_context_panel()

    def on_embedding_error(self, error_msg):
        """Handle embedding error."""
        QMessageBox.critical(
            self, "Embedding Error", f"Failed to compute embeddings:\\n\\n{error_msg}"
        )
        self.status.showMessage("Embedding failed")

    def on_clustering_progress(self, percentage, message):
        """Update progress for clustering."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

    def on_clustering_success(self, result):
        """Handle successful clustering."""
        self.cluster_assignments = result["assignments"]
        self.last_cluster_result = result
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
        QMessageBox.information(
            self,
            "Clustering Complete",
            f"Successfully clustered into {n_clusters} clusters.\\n\\n"
            + f"Total points: {len(self.cluster_assignments):,}",
        )
        self.status.showMessage(f"Clustered into {n_clusters} groups")
        self.update_explorer_plot()
        self.update_context_panel()

    def on_clustering_error(self, error_msg):
        """Handle clustering error."""
        QMessageBox.critical(
            self, "Clustering Error", f"Failed to cluster:\\n\\n{error_msg}"
        )
        self.status.showMessage("Clustering failed")

    def on_umap_progress(self, percentage, message):
        """Update progress for UMAP."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

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
        QMessageBox.information(
            self,
            "UMAP Complete",
            "Successfully computed UMAP projection.\\n\\n"
            + f"Coordinates: {self.umap_coords.shape[0]:,} × 2",
        )
        self.status.showMessage("UMAP projection complete")
        self.update_explorer_plot(force_fit=True)
        self.update_context_panel()

    def on_umap_error(self, error_msg):
        """Handle UMAP error."""
        QMessageBox.critical(
            self, "UMAP Error", f"Failed to compute UMAP:\\n\\n{error_msg}"
        )
        self.status.showMessage("UMAP failed")

    def on_training_progress(self, percentage, message):
        """Update progress for classifier training."""
        self.progress_bar.setValue(percentage)
        self.status.showMessage(message)

    def on_training_success(self, result):
        """Handle successful classifier training."""
        trainer = result.get("trainer")
        history = result.get("history", {})
        self._trained_classifier = trainer

        val_acc_values = history.get("val_acc", []) if isinstance(history, dict) else []
        best_val_acc = max(val_acc_values) if val_acc_values else None
        summary = (
            f"Best validation accuracy: {best_val_acc:.3f}"
            if best_val_acc is not None
            else "No validation split used"
        )

        saved_note = ""
        if trainer is not None and self.project_path is not None:
            try:
                model_dir = self.project_path / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                latest_path = model_dir / "classifier_latest.pt"
                snapshot_path = model_dir / f"classifier_{timestamp}.pt"
                trainer.save(latest_path)
                trainer.save(snapshot_path)

                metadata = {
                    "timestamp": timestamp,
                    "best_val_acc": (
                        float(best_val_acc) if best_val_acc is not None else None
                    ),
                    "history": history,
                    "settings": self._last_training_settings or {},
                    "latest_checkpoint": str(latest_path),
                    "snapshot_checkpoint": str(snapshot_path),
                }
                with open(model_dir / "classifier_latest.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                saved_note = f"\nCheckpoints saved to {model_dir}"
            except Exception as exc:
                saved_note = f"\nModel save warning: {exc}"

        QMessageBox.information(
            self,
            "Training Complete",
            "Embedding head training finished successfully.\\n\\n"
            + summary
            + saved_note,
        )
        self.status.showMessage("Classifier training complete")

        unlabeled_count = sum(1 for label in self.image_labels if not label)
        if unlabeled_count > 0 and self._ask_yes_no(
            "Run Predictions Now?",
            f"There are {unlabeled_count:,} unlabeled images.\n\n"
            "Run classifier predictions for unlabeled images now?",
        ):
            QTimer.singleShot(0, self.predict_unlabeled_images)

    def on_training_error(self, error_msg):
        """Handle classifier training error."""
        QMessageBox.critical(
            self,
            "Training Error",
            f"Failed to train classifier:\\n\\n{error_msg}",
        )
        self.status.showMessage("Classifier training failed")

    def closeEvent(self, event):
        """Ensure pending label updates are flushed before close."""
        self._flush_pending_label_updates(force=True)
        super().closeEvent(event)
