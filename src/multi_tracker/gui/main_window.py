#!/usr/bin/env python3
"""
Main application window for the Multi-Animal Tracker.

Refactored for improved UX with Tabbed interface and logical grouping.
"""

import sys, os, json, math, logging
import numpy as np
import pandas as pd
import cv2
from collections import deque
import gc
import csv

from PySide2.QtCore import Qt, Slot, Signal, QThread, QMutex
from PySide2.QtGui import QImage, QPixmap, QPainter, QPen, QIcon, QColor
from PySide2.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QScrollArea,
    QProgressBar,
    QProgressDialog,
    QApplication,
    QComboBox,
    QTabWidget,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QFrame,
    QSlider,
)
import matplotlib.pyplot as plt

from ..core.tracking_worker import TrackingWorker
from ..core.post_processing import process_trajectories, resolve_trajectories
from ..utils.csv_writer import CSVWriterThread
from ..utils.geometry import fit_circle_to_points
from ..utils.video_io import VideoReversalWorker
from .histogram_widgets import HistogramPanel

# Configuration file for saving/loading tracking parameters
CONFIG_FILENAME = "tracking_config.json"  # Fallback for manual load/save


def get_video_config_path(video_path):
    """Get the config file path for a given video file."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window providing GUI interface for multi-animal tracking.
    """

    parameters_changed = Signal(dict)

    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        self.setWindowTitle("Multi-Animal Tracker Pro")
        self.resize(1360, 850)

        # Set comprehensive dark mode styling
        self.setStyleSheet(
            """
            /* Main window and widgets */
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: -apple-system, system-ui, sans-serif; }
            
            /* Tabs */
            QTabWidget::pane { border: 1px solid #444; top: -1px; }
            QTabBar::tab {
                background: #353535; color: #aaa; padding: 8px 12px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #4a9eff; color: white; font-weight: bold; }
            QTabBar::tab:hover { background: #404040; }

            /* Group boxes */
            QGroupBox {
                font-weight: bold; border: 1px solid #555; border-radius: 6px;
                margin-top: 20px; padding-top: 10px; background-color: #323232;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4a9eff;
            }
            
            /* Buttons */
            QPushButton {
                background-color: #444; border: 1px solid #555; color: #fff;
                padding: 6px 12px; border-radius: 4px; min-height: 25px;
            }
            QPushButton:hover { background-color: #555; border-color: #666; }
            QPushButton:pressed { background-color: #2a75c4; }
            QPushButton:checked { background-color: #2a75c4; border: 1px solid #4a9eff; }
            QPushButton:disabled { background-color: #333; color: #666; border-color: #333; }
            
            /* Specific Action Buttons */
            QPushButton#ActionBtn { background-color: #4a9eff; font-weight: bold; font-size: 13px; }
            QPushButton#ActionBtn:hover { background-color: #3d8bdb; }
            QPushButton#StopBtn { background-color: #d9534f; font-weight: bold; }
            QPushButton#StopBtn:hover { background-color: #c9302c; }

            /* Inputs */
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #222; border: 1px solid #555; border-radius: 3px;
                padding: 4px; color: #fff; selection-background-color: #4a9eff;
                min-width: 120px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 1px solid #4a9eff; }
            
            /* ComboBox dropdown */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555;
                background-color: #555;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::drop-down:hover { background-color: #666; }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #fff;
                width: 6px;
                height: 6px;
                border-top: none;
                border-right: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                border: 1px solid #555;
                selection-background-color: #4a9eff;
                selection-color: #fff;
                color: #fff;
                padding: 4px;
                min-width: 200px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 8px;
                min-height: 24px;
                border: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3d8bdb;
                color: #fff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #4a9eff;
                color: #fff;
            }
            
            /* SpinBox arrows */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #555;
                background-color: #555;
                border-top-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #666;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #4a9eff;
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border-left: 1px solid #555;
                background-color: #555;
                border-bottom-right-radius: 3px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #666;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #4a9eff;
            }
            
            /* Scrollbars */
            QScrollBar:vertical { background: #2b2b2b; width: 12px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 6px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #666; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            
            /* Progress Bar */
            QProgressBar {
                border: 1px solid #555; border-radius: 4px; text-align: center; background: #222;
            }
            QProgressBar::chunk { background-color: #4a9eff; width: 10px; margin: 0.5px; }
            
            QSplitter::handle { background-color: #444; }
            """
        )

        # === STATE VARIABLES ===
        self.roi_base_frame = None
        self.roi_points = []
        self.roi_mask = None
        self.roi_selection_active = False
        self.roi_fitted_circle = None
        # Enhanced ROI support: multiple shapes with inclusion/exclusion
        self.roi_shapes = (
            []
        )  # List of dicts: {'type': 'circle'/'polygon', 'params': ..., 'mode': 'include'/'exclude'}
        self.roi_current_mode = "circle"  # 'circle' or 'polygon'
        self.roi_current_zone_type = "include"  # 'include' or 'exclude'

        self.histogram_panel = None
        self.histogram_window = None
        self.current_worker = None

        self.tracking_worker = None
        self.csv_writer_thread = None
        self.reversal_worker = None
        self.final_full_trajs = []
        self.temporary_files = []  # Track temporary files for cleanup
        self.session_log_handler = None  # Track current session log file handler

        # Preview frame for live image adjustments
        self.preview_frame_original = None  # Original frame without adjustments
        self.detection_test_result = None  # Store detection test result
        self.current_video_path = None
        self.detected_sizes = None  # Store detected object sizes for statistics

        # ROI optimization tracking
        self.roi_crop_warning_shown = (
            False  # Track if we've warned about cropping this session
        )

        # ROI display caching (for performance)
        self._roi_masked_cache = {}  # Cache: {(frame_id, roi_hash): masked_image}
        self._roi_hash = None  # Hash of current ROI configuration

        # Interactive pan/zoom state
        self._is_panning = False
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        
        # Track first frame for auto-fit during tracking
        self._tracking_first_frame = True
        self._tracking_frame_size = None  # (width, height) of resized tracking frames

        # Advanced configuration (for power users)
        self.advanced_config = self._load_advanced_config()

        # === UI CONSTRUCTION ===
        self.init_ui()

        # === POST-INIT ===
        # Disable wheel events on all spinboxes to prevent accidental value changes
        self._disable_spinbox_wheel_events()

        # Config is now loaded automatically when a video is selected
        # instead of at startup
        self._connect_parameter_signals()

    def init_ui(self):
        """Build the structured UI using Splitter and Tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout is a horizontal splitter (Video Left | Controls Right)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANEL: Video & ROI ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Video Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #000; border: none;")
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label = QLabel("Load a video to begin...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #666; font-size: 16px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll.setWidget(self.video_label)

        # Enable mouse tracking and events for interactive pan/zoom
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._handle_video_mouse_press
        self.video_label.mouseMoveEvent = self._handle_video_mouse_move
        self.video_label.mouseReleaseEvent = self._handle_video_mouse_release
        self.video_label.mouseDoubleClickEvent = self._handle_video_double_click
        self.video_label.wheelEvent = self._handle_video_wheel

        # Enable pinch-to-zoom gesture
        self.video_label.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.video_label.grabGesture(Qt.PinchGesture)
        self.video_label.event = self._handle_video_event

        # ROI Toolbar (Contextual to video)
        roi_frame = QFrame()
        roi_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        roi_main_layout = QVBoxLayout(roi_frame)
        roi_main_layout.setContentsMargins(10, 5, 10, 5)

        # Top row: mode selection and controls
        roi_layout = QHBoxLayout()
        roi_label = QLabel("ROI Controls:")
        roi_label.setStyleSheet("font-weight: bold; color: #bbb;")

        # Mode selector
        self.combo_roi_mode = QComboBox()
        self.combo_roi_mode.addItems(["Circle", "Polygon"])
        self.combo_roi_mode.setToolTip("Select shape type for ROI")
        self.combo_roi_mode.currentIndexChanged.connect(self._on_roi_mode_changed)

        # Zone type selector (Include/Exclude)
        self.combo_roi_zone = QComboBox()
        self.combo_roi_zone.addItems(["Include Zone", "Exclude Zone"])
        self.combo_roi_zone.setToolTip(
            "Include: Area where tracking is active\n"
            "Exclude: Area to remove from tracking (applied after inclusions)"
        )
        self.combo_roi_zone.currentIndexChanged.connect(self._on_roi_zone_changed)

        self.btn_start_roi = QPushButton("Add Shape")
        self.btn_start_roi.clicked.connect(self.start_roi_selection)
        self.btn_start_roi.setShortcut("Ctrl+R")
        self.btn_start_roi.setToolTip("Start adding ROI shape (Ctrl+R)")

        self.btn_finish_roi = QPushButton("Confirm Shape")
        self.btn_finish_roi.clicked.connect(self.finish_roi_selection)
        self.btn_finish_roi.setEnabled(False)
        self.btn_finish_roi.setShortcut("Ctrl+F")
        self.btn_finish_roi.setToolTip("Finish current shape (Ctrl+F)")

        self.btn_undo_roi = QPushButton("Undo Last")
        self.btn_undo_roi.clicked.connect(self.undo_last_roi_shape)
        self.btn_undo_roi.setEnabled(False)
        self.btn_undo_roi.setToolTip("Remove last added shape")

        self.btn_clear_roi = QPushButton("Clear All")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setShortcut("Ctrl+C")
        self.btn_clear_roi.setToolTip("Clear all ROI shapes (Ctrl+C)")

        self.btn_crop_video = QPushButton("Crop Video to ROI")
        self.btn_crop_video.clicked.connect(self.crop_video_to_roi)
        self.btn_crop_video.setEnabled(False)
        self.btn_crop_video.setToolTip(
            "Generate a cropped video containing only the ROI area\n"
            "This can significantly improve tracking performance"
        )
        self.btn_crop_video.setStyleSheet(
            "QPushButton { background-color: #2d7a3e; }"
            "QPushButton:hover { background-color: #3a9150; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )

        self.roi_status_label = QLabel("No ROI")
        self.roi_status_label.setStyleSheet("color: #888; margin-left: 10px;")

        roi_layout.addWidget(roi_label)
        roi_layout.addWidget(self.combo_roi_mode)
        roi_layout.addWidget(self.combo_roi_zone)
        roi_layout.addWidget(self.btn_start_roi)
        roi_layout.addWidget(self.btn_finish_roi)
        roi_layout.addWidget(self.btn_undo_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addWidget(self.btn_crop_video)
        roi_layout.addStretch()

        roi_main_layout.addLayout(roi_layout)

        # Second row: status and optimization info
        roi_status_layout = QHBoxLayout()
        roi_status_layout.addWidget(self.roi_status_label)
        self.roi_optimization_label = QLabel("")
        self.roi_optimization_label.setStyleSheet(
            "color: #f0ad4e; margin-left: 10px; font-weight: bold;"
        )
        roi_status_layout.addWidget(self.roi_optimization_label)
        roi_main_layout.addLayout(roi_status_layout)

        # Instructions (Hidden unless active)
        self.roi_instructions = QLabel(""
        )
        self.roi_instructions.setWordWrap(True)
        self.roi_instructions.setStyleSheet(
            "color: #4a9eff; font-size: 11px; font-weight: bold; "
            "padding: 6px; background-color: #1a3a5a; border-radius: 4px;"
        )
        roi_main_layout.addWidget(self.roi_instructions)

        left_layout.addWidget(self.scroll, stretch=1)

        # Interactive instructions
        self.interaction_help = QLabel(
            "Double-click: Fit to screen  •  Drag: Pan  •  Ctrl+Scroll/Pinch: Zoom"
        )
        self.interaction_help.setAlignment(Qt.AlignCenter)
        self.interaction_help.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic; "
            "padding: 4px; background-color: #1a1a1a; border-radius: 3px;"
        )
        left_layout.addWidget(self.interaction_help)

        left_layout.addWidget(roi_frame)

        # Zoom control under video
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(10, 5, 10, 5)

        zoom_label = QLabel("Zoom:")
        zoom_label.setStyleSheet("font-weight: bold; color: #bbb;")

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 500)  # 0.1x to 5.0x, scaled by 100
        self.slider_zoom.setValue(100)  # 1.0x
        self.slider_zoom.setTickPosition(QSlider.TicksBelow)
        self.slider_zoom.setTickInterval(50)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)

        self.label_zoom_val = QLabel("1.00x")
        self.label_zoom_val.setStyleSheet(
            "color: #4a9eff; font-weight: bold; min-width: 50px;"
        )

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.slider_zoom, stretch=1)
        zoom_layout.addWidget(self.label_zoom_val)

        left_layout.addWidget(zoom_frame)

        # Preview refresh button
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 5, 10, 5)

        self.btn_refresh_preview = QPushButton("Load Random Frame for Preview")
        self.btn_refresh_preview.clicked.connect(self._load_preview_frame)
        self.btn_refresh_preview.setEnabled(False)
        self.btn_refresh_preview.setToolTip(
            "Load a random frame from your video.\n\n"
            "For size estimation: Choose a frame with:\n"
            "• Many animals visible\n"
            "• Animals well-separated (not overlapping)\n"
            "• Representative of typical body sizes"
        )
        preview_layout.addWidget(self.btn_refresh_preview)

        self.btn_test_detection = QPushButton("Test Detection on Preview")
        self.btn_test_detection.clicked.connect(self._test_detection_on_preview)
        self.btn_test_detection.setEnabled(False)
        self.btn_test_detection.setStyleSheet(
            "background-color: #4a9eff; color: white; font-weight: bold;"
        )
        preview_layout.addWidget(self.btn_test_detection)

        left_layout.addWidget(preview_frame)

        # --- RIGHT PANEL: Configuration Tabs & Actions ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setUsesScrollButtons(
            True
        )  # Enable scroll buttons when tabs don't fit
        self.tabs.setElideMode(Qt.ElideNone)  # Don't truncate tab text

        # Tab 1: Setup (Files & Performance)
        self.tab_setup = QWidget()
        self.setup_setup_ui()
        self.tabs.addTab(self.tab_setup, "Setup")

        # Tab 2: Detection (Image, Method, Params)
        self.tab_detection = QWidget()
        self.setup_detection_ui()
        self.tabs.addTab(self.tab_detection, "Detection")

        # Tab 3: Tracking (Kalman, Logic, Lifecycle)
        self.tab_tracking = QWidget()
        self.setup_tracking_ui()
        self.tabs.addTab(self.tab_tracking, "Tracking")

        # Tab 4: Data (Post-proc, Histograms)
        self.tab_data = QWidget()
        self.setup_data_ui()
        self.tabs.addTab(self.tab_data, "Processing")

        # Tab 5: Visuals (Overlays, Debug)
        self.tab_viz = QWidget()
        self.setup_viz_ui()
        self.tabs.addTab(self.tab_viz, "Visuals")

        # Tab 6: Dataset Generation (Active Learning)
        self.tab_dataset = QWidget()
        self.setup_dataset_ui()
        self.tabs.addTab(self.tab_dataset, "Dataset Generation")

        # Tab 7: Individual Analysis (Identity & Pose)
        self.tab_individual = QWidget()
        self.setup_individual_analysis_ui()
        self.tabs.addTab(self.tab_individual, "Individual Analysis")

        right_layout.addWidget(self.tabs, stretch=1)

        # Persistent Action Panel (Bottom Right)
        action_frame = QFrame()
        action_frame.setStyleSheet(
            "background-color: #252525; border-top: 1px solid #444; border-radius: 0px;"
        )
        action_layout = QVBoxLayout(action_frame)

        # Progress Info
        prog_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)

        # Real-time stats display
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(5, 5, 5, 5)

        self.label_current_fps = QLabel("FPS: --")
        self.label_current_fps.setStyleSheet(
            "color: #4a9eff; font-weight: bold; font-size: 11px;"
        )
        self.label_current_fps.setVisible(False)
        stats_layout.addWidget(self.label_current_fps)

        self.label_elapsed_time = QLabel("Elapsed: --")
        self.label_elapsed_time.setStyleSheet("color: #888; font-size: 11px;")
        self.label_elapsed_time.setVisible(False)
        stats_layout.addWidget(self.label_elapsed_time)

        self.label_eta = QLabel("ETA: --")
        self.label_eta.setStyleSheet("color: #888; font-size: 11px;")
        self.label_eta.setVisible(False)
        stats_layout.addWidget(self.label_eta)

        stats_layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_preview = QPushButton("Preview Mode")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(lambda ch: self.toggle_preview(ch))
        self.btn_preview.setMinimumHeight(40)

        self.btn_start = QPushButton("Start Full Tracking")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.clicked.connect(self.start_full)
        self.btn_start.setMinimumHeight(40)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.clicked.connect(self.stop_tracking)
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)  # Start disabled

        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        action_layout.addLayout(prog_layout)
        action_layout.addLayout(stats_layout)
        action_layout.addLayout(btn_layout)

        right_layout.addWidget(action_frame)

        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)

        # Set initial splitter ratio (60% Video, 40% Controls) and minimum sizes
        total_width = 1360  # Default window width
        self.splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        main_layout.addWidget(self.splitter)

    # =========================================================================
    # TAB UI BUILDERS
    # =========================================================================

    def setup_setup_ui(self):
        """Tab 1: Setup - Files & Basic Config."""
        layout = QVBoxLayout(self.tab_setup)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)

        # File Inputs
        g_files = QGroupBox("File Management")
        vl_files = QVBoxLayout(g_files)
        vl_files.addWidget(
            self._create_help_label(
                "Select your input video and output locations. Configuration is auto-saved per video - "
                "next time you load the same video, your settings will be restored automatically."
            )
        )
        fl = QFormLayout()
        fl.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.btn_file = QPushButton("Select Input Video...")
        self.btn_file.clicked.connect(self.select_file)
        self.file_line = QLineEdit()
        self.file_line.setPlaceholderText("path/to/video.mp4")
        self.file_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl.addRow(self.btn_file, self.file_line)

        self.btn_csv = QPushButton("Select CSV Output...")
        self.btn_csv.clicked.connect(self.select_csv)
        self.csv_line = QLineEdit()
        self.csv_line.setPlaceholderText("path/to/output.csv")
        self.csv_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl.addRow(self.btn_csv, self.csv_line)

        self.check_video_output = QCheckBox("Enable Video Output")
        self.check_video_output.setChecked(False)
        self.check_video_output.toggled.connect(self._on_video_output_toggled)
        fl.addRow("", self.check_video_output)

        self.btn_video_out = QPushButton("Select Video Output...")
        self.btn_video_out.clicked.connect(self.select_video_output)
        self.btn_video_out.setEnabled(False)
        self.video_out_line = QLineEdit()
        self.video_out_line.setPlaceholderText("Optional visualization export")
        self.video_out_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_out_line.setEnabled(False)
        fl.addRow(self.btn_video_out, self.video_out_line)

        # Config Management
        config_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Config...")
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_load_config.setToolTip("Manually load configuration from a JSON file")
        config_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Config...")
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_save_config.setToolTip("Save current settings to a JSON file")
        config_layout.addWidget(self.btn_save_config)

        config_layout.addStretch()
        fl.addRow("Configuration:", config_layout)

        # Config status label
        self.config_status_label = QLabel("No config loaded (using defaults)")
        self.config_status_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 10px;"
        )
        fl.addRow("", self.config_status_label)
        vl_files.addLayout(fl)

        form.addWidget(g_files)

        # Reference Parameters
        g_ref = QGroupBox("Reference Parameters")
        vl_ref = QVBoxLayout(g_ref)
        vl_ref.addWidget(
            self._create_help_label(
                "These parameters define the time and spatial scale for tracking. "
                "Frame rate controls time-dependent parameters (velocities, durations). "
                "Body size makes all distance/size parameters portable across videos."
            )
        )
        fl_ref = QFormLayout()
        fl_ref.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # FPS with detect button
        fps_layout = QHBoxLayout()
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 240.0)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(30.0)
        self.spin_fps.setDecimals(2)
        self.spin_fps.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_fps.setToolTip(
            "Acquisition frame rate (frames per second) at which the video was recorded.\n"
            "NOTE: This may differ from the video file's playback framerate.\n"
            "Use 'Detect from Video' to read from file metadata as a starting point.\n"
            "Time-dependent parameters (velocity, durations) scale with this.\n"
            "Affects: motion prediction, track lifecycle, velocity thresholds."
        )
        self.spin_fps.valueChanged.connect(self._update_fps_info)
        fps_layout.addWidget(self.spin_fps)

        self.btn_detect_fps = QPushButton("Detect from Video")
        self.btn_detect_fps.clicked.connect(self._detect_fps_from_current_video)
        self.btn_detect_fps.setEnabled(False)
        self.btn_detect_fps.setToolTip(
            "Auto-detect frame rate from video metadata (may differ from actual acquisition rate)"
        )
        fps_layout.addWidget(self.btn_detect_fps)
        fl_ref.addRow("Acquisition Frame Rate (FPS):", fps_layout)

        # FPS info label
        self.label_fps_info = QLabel()
        self.label_fps_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        fl_ref.addRow("", self.label_fps_info)
        vl_ref.addLayout(fl_ref)
        form.addWidget(g_ref)

        # System Performance
        g_sys = QGroupBox("System Performance")
        vl_sys = QVBoxLayout(g_sys)
        vl_sys.addWidget(
            self._create_help_label(
                "Resize factor reduces computational cost by downscaling frames. "
                "Lower values speed up processing but reduce spatial accuracy."
            )
        )
        fl_sys = QFormLayout()
        fl_sys.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_resize = QDoubleSpinBox()
        self.spin_resize.setRange(0.1, 1.0)
        self.spin_resize.setSingleStep(0.1)
        self.spin_resize.setValue(1.0)
        self.spin_resize.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_resize.setToolTip(
            "Downscale video frames for faster processing.\n"
            "1.0 = full resolution, 0.5 = half resolution (4× faster).\n"
            "All body-size-based parameters auto-scale with this value."
        )
        fl_sys.addRow("Processing Resize Factor:", self.spin_resize)

        self.check_save_confidence = QCheckBox("Save Confidence Metrics (slower)")
        self.check_save_confidence.setChecked(True)
        self.check_save_confidence.setToolTip(
            "Save detection, assignment, and position uncertainty metrics to CSV.\n"
            "Useful for post-hoc quality control but adds ~10-20% processing time.\n"
            "Disable for maximum tracking speed."
        )
        fl_sys.addRow("", self.check_save_confidence)

        # Visualization-Free Mode
        self.chk_visualization_free = QCheckBox(
            "Enable Visualization-Free Mode (Maximum Speed)"
        )
        self.chk_visualization_free.setChecked(False)
        self.chk_visualization_free.setToolTip(
            "Skip all frame visualization and rendering.\n"
            "Significantly faster processing (2-4× speedup).\n"
            "Real-time FPS/ETA stats still shown in UI.\n"
            "Recommended for large batch processing."
        )
        self.chk_visualization_free.stateChanged.connect(
            self._on_visualization_mode_changed
        )
        fl_sys.addRow("", self.chk_visualization_free)

        vl_sys.addLayout(fl_sys)
        form.addWidget(g_sys)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_detection_ui(self):
        """Tab 2: Detection - Method, Image Proc, Algo specific."""
        layout = QVBoxLayout(self.tab_detection)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # 1. Detection Method Selector
        g_method = QGroupBox("Detection Strategy")
        l_method_outer = QVBoxLayout(g_method)
        l_method_outer.addWidget(
            self._create_help_label(
                "Choose how to detect animals in each frame. Background Subtraction works by modeling "
                "the static background and finding moving objects. YOLO uses deep learning to detect animals directly."
            )
        )
        l_method = QHBoxLayout()
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.currentIndexChanged.connect(
            self._on_detection_method_changed_ui
        )
        l_method.addWidget(QLabel("Method:"))
        l_method.addWidget(self.combo_detection_method)
        l_method.addStretch()
        l_method_outer.addLayout(l_method)
        vbox.addWidget(g_method)

        # 2. Common Size Filtering (Applies to both methods)
        g_size = QGroupBox("Size Filtering")
        vl_size = QVBoxLayout(g_size)
        vl_size.addWidget(
            self._create_help_label(
                "Filter detections by size relative to your reference body size. This removes noise (too small) "
                "and erroneous clusters (too large). Most effective when animals are similar size."
            )
        )
        f_size = QFormLayout()
        self.chk_size_filtering = QCheckBox("Enable Size Constraints")
        self.chk_size_filtering.setToolTip(
            "Filter detected objects by area to remove noise and artifacts.\n"
            "Recommended: Enable for cleaner tracking."
        )
        f_size.addRow(self.chk_size_filtering)

        h_sf = QHBoxLayout()
        self.spin_min_object_size = QDoubleSpinBox()
        self.spin_min_object_size.setRange(0.1, 5.0)
        self.spin_min_object_size.setSingleStep(0.1)
        self.spin_min_object_size.setDecimals(2)
        self.spin_min_object_size.setValue(0.3)
        self.spin_min_object_size.setToolTip(
            "Minimum object area as multiple of reference body area.\n"
            "Filters out small noise/artifacts.\n"
            "Recommended: 0.2-0.5× (allows partial occlusion)"
        )
        self.spin_max_object_size = QDoubleSpinBox()
        self.spin_max_object_size.setRange(0.5, 10.0)
        self.spin_max_object_size.setSingleStep(0.1)
        self.spin_max_object_size.setDecimals(2)
        self.spin_max_object_size.setValue(3.0)
        self.spin_max_object_size.setToolTip(
            "Maximum object area as multiple of reference body area.\n"
            "Filters out large clusters or artifacts.\n"
            "Recommended: 2-4× (handles overlapping animals)"
        )
        h_sf.addWidget(QLabel("Min (×body):"))
        h_sf.addWidget(self.spin_min_object_size)
        h_sf.addWidget(QLabel("Max (×body):"))
        h_sf.addWidget(self.spin_max_object_size)
        f_size.addRow(h_sf)
        vl_size.addLayout(f_size)
        vbox.addWidget(g_size)

        # Reference Body Size (Spatial Scale)
        g_body_size = QGroupBox("Reference Body Size (Spatial Scale)")
        vl_body_size = QVBoxLayout(g_body_size)
        vl_body_size.addWidget(
            self._create_help_label(
                "Define the spatial scale for tracking. This reference size makes all distance/size "
                "parameters portable across videos. Set this BEFORE configuring tracking parameters."
            )
        )
        fl_body = QFormLayout()
        fl_body.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_reference_body_size = QDoubleSpinBox()
        self.spin_reference_body_size.setRange(1.0, 500.0)
        self.spin_reference_body_size.setSingleStep(1.0)
        self.spin_reference_body_size.setValue(20.0)
        self.spin_reference_body_size.setDecimals(2)
        self.spin_reference_body_size.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.spin_reference_body_size.setToolTip(
            "Reference animal body diameter in pixels (at resize=1.0).\n"
            "All distance/size parameters are scaled relative to this value."
        )
        self.spin_reference_body_size.valueChanged.connect(self._update_body_size_info)
        fl_body.addRow("Reference Body Size (px):", self.spin_reference_body_size)

        # Info label showing calculated area
        self.label_body_size_info = QLabel()
        self.label_body_size_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        fl_body.addRow("", self.label_body_size_info)
        vl_body_size.addLayout(fl_body)
        vbox.addWidget(g_body_size)

        # Detection size statistics panel
        g_detect_stats = QGroupBox("Detection Size Statistics & Auto-Configuration")
        vl_stats = QVBoxLayout(g_detect_stats)
        vl_stats.addWidget(
            self._create_help_label(
                "Workflow for accurate size estimation:\n"
                "1. Configure your detection method below\n"
                "2. IMPORTANT: Disable 'Enable Size Filtering' above (it biases estimates)\n"
                "3. Click 'Load Random Frame for Preview' (bottom of page)\n"
                "4. Choose a frame with many animals well-separated\n"
                "5. Click 'Test Detection' to analyze sizes\n"
                "6. Use 'Auto-Set' to apply the recommended body size"
            )
        )

        self.label_detection_stats = QLabel(
            "No detection data yet.\nRun 'Test Detection' to estimate sizes."
        )
        self.label_detection_stats.setStyleSheet(
            "color: #aaa; font-size: 11px; padding: 8px; "
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        self.label_detection_stats.setWordWrap(True)
        vl_stats.addWidget(self.label_detection_stats)

        # Auto-set button
        btn_layout = QHBoxLayout()
        self.btn_auto_set_body_size = QPushButton("Auto-Set Body Size from Median")
        self.btn_auto_set_body_size.clicked.connect(
            self._auto_set_body_size_from_detection
        )
        self.btn_auto_set_body_size.setEnabled(False)
        self.btn_auto_set_body_size.setToolTip(
            "Automatically set reference body size to the median detected diameter"
        )
        btn_layout.addWidget(self.btn_auto_set_body_size)
        btn_layout.addStretch()
        vl_stats.addLayout(btn_layout)

        vbox.addWidget(g_detect_stats)

        # 3. Image Pre-processing (Common) with Live Preview
        # Only shown for Background Subtraction, not YOLO
        self.g_img = QGroupBox("Image Enhancement (Pre-processing)")
        vl_img = QVBoxLayout(self.g_img)
        vl_img.addWidget(
            self._create_help_label(
                "Adjust image properties before detection to improve contrast between animals and background. "
                "Start with default values and adjust only if animals are hard to distinguish."
            )
        )

        # Brightness slider
        bright_layout = QVBoxLayout()
        bright_label_row = QHBoxLayout()
        bright_label_row.addWidget(QLabel("Brightness:"))
        self.label_brightness_val = QLabel("0")
        self.label_brightness_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        bright_label_row.addWidget(self.label_brightness_val)
        bright_label_row.addStretch()
        bright_layout.addLayout(bright_label_row)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-255, 255)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTickPosition(QSlider.TicksBelow)
        self.slider_brightness.setTickInterval(50)
        self.slider_brightness.valueChanged.connect(self._on_brightness_changed)
        self.slider_brightness.setToolTip(
            "Adjust overall image brightness.\n"
            "Positive = lighter, Negative = darker.\n"
            "Use to improve contrast between animals and background."
        )
        bright_layout.addWidget(self.slider_brightness)
        vl_img.addLayout(bright_layout)

        # Contrast slider
        contrast_layout = QVBoxLayout()
        contrast_label_row = QHBoxLayout()
        contrast_label_row.addWidget(QLabel("Contrast:"))
        self.label_contrast_val = QLabel("1.0")
        self.label_contrast_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        contrast_label_row.addWidget(self.label_contrast_val)
        contrast_label_row.addStretch()
        contrast_layout.addLayout(contrast_label_row)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 300)  # 0.0 to 3.0, scaled by 100
        self.slider_contrast.setValue(100)  # 1.0
        self.slider_contrast.setTickPosition(QSlider.TicksBelow)
        self.slider_contrast.setTickInterval(50)
        self.slider_contrast.valueChanged.connect(self._on_contrast_changed)
        self.slider_contrast.setToolTip(
            "Adjust image contrast (difference between light and dark).\n"
            "1.0 = original, >1.0 = more contrast, <1.0 = less contrast.\n"
            "Increase to make animals stand out from background."
        )
        contrast_layout.addWidget(self.slider_contrast)
        vl_img.addLayout(contrast_layout)

        # Gamma slider
        gamma_layout = QVBoxLayout()
        gamma_label_row = QHBoxLayout()
        gamma_label_row.addWidget(QLabel("Gamma:"))
        self.label_gamma_val = QLabel("1.0")
        self.label_gamma_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        gamma_label_row.addWidget(self.label_gamma_val)
        gamma_label_row.addStretch()
        gamma_layout.addLayout(gamma_label_row)

        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 300)  # 0.1 to 3.0, scaled by 100
        self.slider_gamma.setValue(100)  # 1.0
        self.slider_gamma.setTickPosition(QSlider.TicksBelow)
        self.slider_gamma.setTickInterval(50)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        self.slider_gamma.setToolTip(
            "Adjust gamma correction (mid-tone brightness).\n"
            "1.0 = original, >1.0 = brighter mid-tones, <1.0 = darker mid-tones.\n"
            "Use to enhance detail in shadowed or bright areas."
        )
        gamma_layout.addWidget(self.slider_gamma)
        vl_img.addLayout(gamma_layout)

        # Dark on light checkbox
        self.chk_dark_on_light = QCheckBox("Dark Animals on Light Background")
        self.chk_dark_on_light.setChecked(True)
        self.chk_dark_on_light.setToolTip(
            "Check if animals are darker than background (most common).\n"
            "Uncheck if animals are lighter than background.\n"
            "This inverts the foreground detection."
        )
        vl_img.addWidget(self.chk_dark_on_light)

        vbox.addWidget(self.g_img)

        # 3. Stacked Widget for Method Specific Params
        self.stack_detection = QStackedWidget()

        # --- Page 0: Background Subtraction Params ---
        page_bg = QWidget()
        l_bg = QVBoxLayout(page_bg)
        l_bg.setContentsMargins(0, 0, 0, 0)
        l_bg.addWidget(
            self._create_help_label(
                "Background subtraction identifies moving animals by comparing each frame to a learned background model. "
                "Start with defaults and increase threshold if you see too much noise, decrease if animals are missed."
            )
        )

        # Background Model
        g_bg_model = QGroupBox("Background Model")
        vl_bg_model = QVBoxLayout(g_bg_model)
        vl_bg_model.addWidget(
            self._create_help_label(
                "Build a model of the static background. Priming frames establish initial model, learning rate "
                "controls adaptation speed, threshold sets sensitivity. Lower threshold = more sensitive detection."
            )
        )
        f_bg = QFormLayout()
        f_bg.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_bg_prime = QSpinBox()
        self.spin_bg_prime.setRange(0, 5000)
        self.spin_bg_prime.setValue(10)
        self.spin_bg_prime.setToolTip(
            "Number of initial frames to build background model.\n"
            "Recommended: 10-100 frames.\n"
            "Use more if background varies or animals are present initially."
        )
        f_bg.addRow("Priming Frames:", self.spin_bg_prime)

        self.chk_adaptive_bg = QCheckBox("Adaptive Background (Update over time)")
        self.chk_adaptive_bg.setChecked(True)
        self.chk_adaptive_bg.setToolTip(
            "Continuously update background model during tracking.\n"
            "Recommended: Enable for videos with changing lighting.\n"
            "Disable for static background to improve performance."
        )
        f_bg.addRow(self.chk_adaptive_bg)

        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setValue(0.001)
        self.spin_bg_learning.setToolTip(
            "How quickly background adapts to changes (0.0001-0.1).\n"
            "Lower = slower adaptation (stable, good for mostly static background).\n"
            "Higher = faster adaptation (use for variable lighting/shadows)."
        )
        f_bg.addRow("Learning Rate:", self.spin_bg_learning)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setToolTip(
            "Pixel intensity difference to detect foreground (0-255).\n"
            "Lower = more sensitive (detects subtle animals, more noise).\n"
            "Higher = less sensitive (cleaner, may miss animals).\n"
            "Recommended: 30-70 depending on contrast."
        )
        f_bg.addRow("Subtraction Threshold:", self.spin_threshold)
        vl_bg_model.addLayout(f_bg)
        l_bg.addWidget(g_bg_model)

        # Lighting Stab
        g_light = QGroupBox("Lighting Stabilization")
        vl_light = QVBoxLayout(g_light)
        vl_light.addWidget(
            self._create_help_label(
                "Compensate for gradual lighting changes (clouds, time of day). Smoothing factor controls "
                "adaptation speed - higher = slower/more stable. Enable for outdoor or variable-light videos."
            )
        )
        f_light = QFormLayout()
        f_light.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.chk_lighting_stab = QCheckBox("Enable Stabilization")
        self.chk_lighting_stab.setChecked(True)
        self.chk_lighting_stab.setToolTip(
            "Compensate for gradual lighting changes over time.\n"
            "Recommended: Enable for videos with variable lighting.\n"
            "Disable for consistent illumination to improve speed."
        )
        f_light.addRow(self.chk_lighting_stab)

        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setValue(0.95)
        self.spin_lighting_smooth.setToolTip(
            "Temporal smoothing factor for lighting correction (0.8-0.999).\n"
            "Higher = smoother, slower adaptation to lighting changes.\n"
            "Lower = faster response to sudden lighting shifts.\n"
            "Recommended: 0.9-0.98"
        )
        f_light.addRow("Smooth Factor:", self.spin_lighting_smooth)

        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setSingleStep(2)
        self.spin_lighting_median.setValue(5)
        self.spin_lighting_median.setToolTip(
            "Median filter window size (odd number, 3-15).\n"
            "Larger window = smoother lighting estimate, slower response.\n"
            "Smaller window = faster response, less smoothing.\n"
            "Recommended: 5-9"
        )
        f_light.addRow("Median Window:", self.spin_lighting_median)
        vl_light.addLayout(f_light)
        l_bg.addWidget(g_light)

        # Morphology (Standard)
        g_morph = QGroupBox("Morphology & Noise")
        vl_morph = QVBoxLayout(g_morph)
        vl_morph.addWidget(
            self._create_help_label(
                "Clean up detected blobs using morphological operations. Closing fills small holes, opening removes "
                "small noise. Larger kernels = stronger effect but may distort shape. Use odd numbers only."
            )
        )
        f_morph = QFormLayout()
        f_morph.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 25)
        self.spin_morph_size.setSingleStep(2)
        self.spin_morph_size.setValue(5)
        self.spin_morph_size.setToolTip(
            "Morphological operation kernel size (odd number, 1-25).\n"
            "Larger = more aggressive noise removal, may merge nearby animals.\n"
            "Smaller = preserves detail, may leave noise.\n"
            "Recommended: 3-7 for typical tracking scenarios."
        )
        f_morph.addRow("Main Kernel Size:", self.spin_morph_size)

        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        self.spin_min_contour.setToolTip(
            "Minimum contour area in pixels² to keep.\n"
            "Filters out small noise blobs after morphology.\n"
            "Recommended: 20-100 depending on animal size and zoom.\n"
            "Note: Similar to min object size but in absolute pixels."
        )
        f_morph.addRow("Min Contour Area:", self.spin_min_contour)

        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        self.spin_max_contour_multiplier.setToolTip(
            "Maximum contour area as multiplier of minimum (5-100).\n"
            "Max area = min_contour × this multiplier.\n"
            "Filters out very large blobs (clusters, shadows, artifacts).\n"
            "Recommended: 10-30"
        )
        f_morph.addRow("Max Contour Multiplier:", self.spin_max_contour_multiplier)
        vl_morph.addLayout(f_morph)
        l_bg.addWidget(g_morph)

        # Morphology (Advanced/Splitting)
        g_split = QGroupBox("Advanced Separation")
        vl_split = QVBoxLayout(g_split)
        vl_split.addWidget(
            self._create_help_label(
                "Split touching animals using erosion/dilation. Conservative split uses watershed, aggressive uses "
                "multi-stage erosion/dilation. Enable only if animals frequently touch."
            )
        )
        f_split = QFormLayout()
        self.chk_conservative_split = QCheckBox("Conservative Splitting (Erosion)")
        self.chk_conservative_split.setChecked(True)
        self.chk_conservative_split.setToolTip(
            "Use erosion to separate touching animals more conservatively.\n"
            "Recommended: Enable to avoid over-splitting single animals.\n"
            "Disable for aggressive separation of tightly clustered animals."
        )
        f_split.addRow(self.chk_conservative_split)

        h_split = QHBoxLayout()
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setRange(1, 15)
        self.spin_conservative_kernel.setSingleStep(2)
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_kernel.setToolTip(
            "Erosion kernel size (odd number, 1-15).\n"
            "Larger = more aggressive separation.\n"
            "Recommended: 3-5"
        )
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setRange(1, 10)
        self.spin_conservative_erode.setValue(1)
        self.spin_conservative_erode.setToolTip(
            "Number of erosion iterations (1-10).\n"
            "More iterations = stronger separation effect.\n"
            "Recommended: 1-2"
        )
        h_split.addWidget(QLabel("K-Size:"))
        h_split.addWidget(self.spin_conservative_kernel)
        h_split.addWidget(QLabel("Iters:"))
        h_split.addWidget(self.spin_conservative_erode)
        f_split.addRow(h_split)

        self.spin_merge_threshold = QSpinBox()
        self.spin_merge_threshold.setRange(100, 10000)
        self.spin_merge_threshold.setValue(1000)
        self.spin_merge_threshold.setToolTip(
            "Maximum area (px²) of small blobs to merge with nearby animals.\n"
            "Helps reconnect fragmented detections.\n"
            "Lower = merge more aggressively, Higher = keep fragments separate.\n"
            "Recommended: 500-2000"
        )
        f_split.addRow("Merge Area Threshold:", self.spin_merge_threshold)

        self.chk_additional_dilation = QCheckBox("Reconnect Thin Parts (Dilation)")
        self.chk_additional_dilation.setToolTip(
            "Use dilation to reconnect thin parts (e.g., legs, antennae).\n"
            "Recommended: Enable if animals have thin appendages.\n"
            "Disable to maintain accurate body shape."
        )
        f_split.addRow(self.chk_additional_dilation)

        h_dil = QHBoxLayout()
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setRange(1, 15)
        self.spin_dilation_kernel_size.setSingleStep(2)
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_kernel_size.setToolTip(
            "Dilation kernel size (odd number, 1-15).\n"
            "Larger = thicker reconnection.\n"
            "Recommended: 3-5"
        )
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setRange(1, 10)
        self.spin_dilation_iterations.setValue(2)
        self.spin_dilation_iterations.setToolTip(
            "Number of dilation iterations (1-10).\n"
            "More iterations = thicker result.\n"
            "Recommended: 1-3"
        )
        h_dil.addWidget(QLabel("K-Size:"))
        h_dil.addWidget(self.spin_dilation_kernel_size)
        h_dil.addWidget(QLabel("Iters:"))
        h_dil.addWidget(self.spin_dilation_iterations)
        f_split.addRow(h_dil)
        vl_split.addLayout(f_split)

        l_bg.addWidget(g_split)

        # --- Page 1: YOLO Params ---
        page_yolo = QWidget()
        l_yolo = QVBoxLayout(page_yolo)
        l_yolo.setContentsMargins(0, 0, 0, 0)
        l_yolo.addWidget(
            self._create_help_label(
                "YOLO uses a trained neural network to detect animals. Choose your model file and adjust confidence "
                "threshold to balance detection sensitivity vs false positives. Higher confidence = fewer false detections."
            )
        )

        self.yolo_group = QGroupBox("YOLO Configuration")
        f_yolo = QFormLayout(self.yolo_group)

        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.addItems(
            [
                "yolo26s-obb.pt (Balanced)",
                "yolo26n-obb.pt (Fastest)",
                "yolov11s-obb.pt",
                "Custom Model...",
            ]
        )
        self.combo_yolo_model.currentIndexChanged.connect(self.on_yolo_model_changed)
        self.combo_yolo_model.setToolTip(
            "YOLO model for oriented bounding box detection.\n"
            "yolo26s = balanced speed/accuracy, yolo26n = fastest.\n"
            "Select 'Custom Model...' to use your own trained model."
        )
        f_yolo.addRow("Model:", self.combo_yolo_model)

        # Custom model container (hidden by default)
        self.yolo_custom_model_widget = QWidget()
        h_cust = QHBoxLayout(self.yolo_custom_model_widget)
        h_cust.setContentsMargins(0, 0, 0, 0)
        self.yolo_custom_model_line = QLineEdit()
        self.btn_yolo_custom_model = QPushButton("...")
        self.btn_yolo_custom_model.clicked.connect(self.select_yolo_custom_model)
        h_cust.addWidget(self.yolo_custom_model_line)
        h_cust.addWidget(self.btn_yolo_custom_model)
        f_yolo.addRow("Path:", self.yolo_custom_model_widget)
        self.yolo_custom_model_widget.setVisible(False)

        self.spin_yolo_confidence = QDoubleSpinBox()
        self.spin_yolo_confidence.setRange(0.01, 1.0)
        self.spin_yolo_confidence.setValue(0.25)
        self.spin_yolo_confidence.setToolTip(
            "Minimum confidence score for YOLO detections (0.01-1.0).\n"
            "Lower = more detections (more false positives).\n"
            "Higher = fewer detections (may miss animals).\n"
            "Recommended: 0.2-0.4"
        )
        f_yolo.addRow("Confidence:", self.spin_yolo_confidence)

        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        self.spin_yolo_iou.setToolTip(
            "Intersection-over-Union threshold for non-max suppression (0.01-1.0).\n"
            "Lower = more aggressive duplicate removal.\n"
            "Higher = keep more overlapping detections.\n"
            "Recommended: 0.5-0.8"
        )
        f_yolo.addRow("IOU Threshold:", self.spin_yolo_iou)

        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setPlaceholderText("e.g. 15, 16 (Empty for all)")
        self.line_yolo_classes.setToolTip(
            "Comma-separated class IDs to detect (leave empty for all classes).\n"
            "Example: '0,1,2' to detect only classes 0, 1, and 2.\n"
            "Refer to your model's class definitions."
        )
        f_yolo.addRow("Target Classes:", self.line_yolo_classes)

        self.combo_yolo_device = QComboBox()
        self.combo_yolo_device.addItems(["auto", "cpu", "cuda:0", "mps"])
        self.combo_yolo_device.setToolTip(
            "Hardware device for YOLO inference.\n"
            "auto = automatic selection, cpu = CPU only.\n"
            "cuda:0 = NVIDIA GPU, mps = Apple Silicon GPU.\n"
            "GPU dramatically improves YOLO speed."
        )
        f_yolo.addRow("Device:", self.combo_yolo_device)

        l_yolo.addWidget(self.yolo_group)
        l_yolo.addStretch()

        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
        self.stack_detection.addWidget(page_yolo)

        vbox.addWidget(self.stack_detection)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_tracking_ui(self):
        """Tab 3: Tracking Logic."""
        layout = QVBoxLayout(self.tab_tracking)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # Core Params
        g_core = QGroupBox("Core Tracking Parameters")
        vl_core = QVBoxLayout(g_core)
        vl_core.addWidget(
            self._create_help_label(
                "These control basic track-to-detection matching. Max assignment distance sets how far an animal can "
                "move between frames. Recovery search distance helps reconnect lost tracks."
            )
        )
        f_core = QFormLayout()
        f_core.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 200)
        self.spin_max_targets.setValue(4)
        self.spin_max_targets.setToolTip(
            "Maximum number of animals to track simultaneously (1-200).\n"
            "Set this to the expected number of animals in your video.\n"
            "Higher values use more memory and may slow down processing."
        )
        f_core.addRow("Max Targets (Animals):", self.spin_max_targets)

        self.spin_max_dist = QDoubleSpinBox()
        self.spin_max_dist.setRange(0.1, 20.0)
        self.spin_max_dist.setSingleStep(0.1)
        self.spin_max_dist.setDecimals(2)
        self.spin_max_dist.setValue(1.5)
        self.spin_max_dist.setToolTip(
            "Maximum distance for track-to-detection assignment (×body size).\n"
            "Animals can move at most this distance between frames.\n"
            "Too low = tracks break frequently, Too high = identity swaps.\n"
            "Recommended: 1-2× for normal motion, 3-5× for fast motion."
        )
        f_core.addRow("Max Assignment Dist (×body):", self.spin_max_dist)

        self.spin_continuity_thresh = QDoubleSpinBox()
        self.spin_continuity_thresh.setRange(0.1, 10.0)
        self.spin_continuity_thresh.setSingleStep(0.1)
        self.spin_continuity_thresh.setDecimals(2)
        self.spin_continuity_thresh.setValue(0.5)
        self.spin_continuity_thresh.setToolTip(
            "Search radius for recovering lost tracks (×body size).\n"
            "When a track is lost, looks backward within this distance.\n"
            "Smaller = more conservative recovery (fewer false merges).\n"
            "Recommended: 0.3-1.0×"
        )
        f_core.addRow("Recovery Search Distance (×body):", self.spin_continuity_thresh)

        self.chk_enable_backward = QCheckBox("Run Backward Tracking after Forward")
        self.chk_enable_backward.setChecked(True)
        self.chk_enable_backward.setToolTip(
            "Run tracking in reverse after forward pass to improve accuracy.\n"
            "Recommended: Enable for best results (takes 2× time).\n"
            "Disable for faster processing if accuracy is sufficient."
        )
        f_core.addRow("", self.chk_enable_backward)
        vl_core.addLayout(f_core)
        vbox.addWidget(g_core)

        # Kalman
        g_kf = QGroupBox("Kalman Filter (Motion Model)")
        vl_kf = QVBoxLayout(g_kf)
        vl_kf.addWidget(
            self._create_help_label(
                "Kalman filter predicts animal positions using motion history. Process noise controls smoothing, "
                "measurement noise controls responsiveness. Age-dependent damping helps stabilize newly initialized tracks."
            )
        )
        f_kf = QFormLayout()
        f_kf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setToolTip(
            "Process noise covariance (0.0-1.0) for motion prediction.\n"
            "Lower = trust motion model more (smooth, may lag).\n"
            "Higher = trust measurements more (responsive, less smooth).\n"
            "Note: Optimal value depends on frame rate (time step).\n"
            "Recommended: 0.01-0.05 for predictable motion."
        )
        f_kf.addRow("Process Noise:", self.spin_kalman_noise)

        self.spin_kalman_meas = QDoubleSpinBox()
        self.spin_kalman_meas.setRange(0.0, 1.0)
        self.spin_kalman_meas.setValue(0.1)
        self.spin_kalman_meas.setToolTip(
            "Measurement noise covariance (0.0-1.0).\n"
            "Lower = trust detections more (accurate, may be jittery).\n"
            "Higher = trust predictions more (smooth, may drift).\n"
            "Recommended: 0.05-0.15"
        )
        f_kf.addRow("Measurement Noise:", self.spin_kalman_meas)

        self.spin_kalman_damping = QDoubleSpinBox()
        self.spin_kalman_damping.setRange(0.5, 0.99)
        self.spin_kalman_damping.setSingleStep(0.01)
        self.spin_kalman_damping.setDecimals(2)
        self.spin_kalman_damping.setValue(0.95)
        self.spin_kalman_damping.setToolTip(
            "Velocity damping coefficient (0.5-0.99).\n"
            "Controls how quickly velocity decays each frame.\n"
            "Lower = faster decay (better for stop-and-go behavior).\n"
            "Higher = slower decay (better for continuous motion).\n"
            "Recommended: 0.90-0.95"
        )
        f_kf.addRow("Velocity Damping:", self.spin_kalman_damping)

        # Age-dependent velocity damping
        age_label = QLabel("Age-Dependent Damping (New Tracks):")
        age_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(age_label)

        self.spin_kalman_maturity_age = QSpinBox()
        self.spin_kalman_maturity_age.setRange(1, 30)
        self.spin_kalman_maturity_age.setValue(5)
        self.spin_kalman_maturity_age.setToolTip(
            "Number of frames for a track to reach maturity (1-30).\n"
            "Young tracks use conservative velocity estimates.\n"
            "After this many successful updates, tracks use full dynamics.\n"
            "Lower = faster adaptation, Higher = more conservative.\n"
            "Recommended: 3-10 frames"
        )
        f_kf.addRow("  Maturity Age (frames):", self.spin_kalman_maturity_age)

        self.spin_kalman_initial_velocity_retention = QDoubleSpinBox()
        self.spin_kalman_initial_velocity_retention.setRange(0.0, 1.0)
        self.spin_kalman_initial_velocity_retention.setSingleStep(0.05)
        self.spin_kalman_initial_velocity_retention.setDecimals(2)
        self.spin_kalman_initial_velocity_retention.setValue(0.2)
        self.spin_kalman_initial_velocity_retention.setToolTip(
            "Initial velocity retention for brand new tracks (0.0-1.0).\n"
            "0.0 = assume stationary (no velocity)\n"
            "1.0 = use full velocity estimate\n"
            "Gradually increases to 1.0 as track ages to maturity.\n"
            "Lower = more conservative (prevents wild predictions).\n"
            "Recommended: 0.1-0.3"
        )
        f_kf.addRow(
            "  Initial Velocity Retention:", self.spin_kalman_initial_velocity_retention
        )

        self.spin_kalman_max_velocity = QDoubleSpinBox()
        self.spin_kalman_max_velocity.setRange(0.5, 10.0)
        self.spin_kalman_max_velocity.setSingleStep(0.1)
        self.spin_kalman_max_velocity.setDecimals(1)
        self.spin_kalman_max_velocity.setValue(2.0)
        self.spin_kalman_max_velocity.setToolTip(
            "Maximum velocity constraint (body size multiplier).\n"
            "Prevents unrealistic predictions during occlusions.\n"
            "velocity_max = this_value × reference_body_size (pixels/frame)\n"
            "Lower = more conservative, Higher = allows faster movement.\n"
            "Recommended: 1.5-3.0 depending on animal speed"
        )
        f_kf.addRow("  Max Velocity (×body):", self.spin_kalman_max_velocity)

        # Anisotropic process noise
        aniso_label = QLabel("Anisotropic Process Noise:")
        aniso_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(aniso_label)

        self.spin_kalman_longitudinal_noise = QDoubleSpinBox()
        self.spin_kalman_longitudinal_noise.setRange(0.1, 20.0)
        self.spin_kalman_longitudinal_noise.setSingleStep(0.5)
        self.spin_kalman_longitudinal_noise.setDecimals(1)
        self.spin_kalman_longitudinal_noise.setValue(5.0)
        self.spin_kalman_longitudinal_noise.setToolTip(
            "Forward/longitudinal noise multiplier (0.1-20.0).\n"
            "Controls uncertainty in direction of movement.\n"
            "Higher = more uncertainty forward (smoother forward motion).\n"
            "Multiplies base process noise for forward direction.\n"
            "Recommended: 3.0-7.0"
        )
        f_kf.addRow("  Longitudinal Multiplier:", self.spin_kalman_longitudinal_noise)

        self.spin_kalman_lateral_noise = QDoubleSpinBox()
        self.spin_kalman_lateral_noise.setRange(0.01, 5.0)
        self.spin_kalman_lateral_noise.setSingleStep(0.05)
        self.spin_kalman_lateral_noise.setDecimals(2)
        self.spin_kalman_lateral_noise.setValue(0.1)
        self.spin_kalman_lateral_noise.setToolTip(
            "Sideways/lateral noise multiplier (0.01-5.0).\n"
            "Controls uncertainty perpendicular to movement.\n"
            "Lower = less uncertainty sideways (constrains lateral drift).\n"
            "Multiplies base process noise for sideways direction.\n"
            "Recommended: 0.05-0.2"
        )
        f_kf.addRow("  Lateral Multiplier:", self.spin_kalman_lateral_noise)

        vl_kf.addLayout(f_kf)
        vbox.addWidget(g_kf)

        # Weights
        g_weights = QGroupBox("Cost Function Weights")
        l_weights = QVBoxLayout(g_weights)
        l_weights.addWidget(
            self._create_help_label(
                "Control how different factors influence track-to-detection matching. Position is primary; orientation, "
                "area, and aspect help resolve ambiguities. Increase Mahalanobis to trust Kalman predictions more."
            )
        )

        row1 = QHBoxLayout()
        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setRange(0.0, 10.0)
        self.spin_Wp.setValue(1.0)
        self.spin_Wp.setToolTip(
            "Weight for position distance in assignment cost.\n"
            "Higher = prioritize spatial proximity.\n"
            "Recommended: 1.0 (primary factor)"
        )
        row1.addWidget(QLabel("Position:"))
        row1.addWidget(self.spin_Wp)

        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setRange(0.0, 10.0)
        self.spin_Wo.setValue(1.0)
        self.spin_Wo.setToolTip(
            "Weight for orientation difference in assignment cost.\n"
            "Higher = penalize large orientation changes.\n"
            "Recommended: 0.5-2.0 (helps maintain correct identity)"
        )
        row1.addWidget(QLabel("Orientation:"))
        row1.addWidget(self.spin_Wo)
        l_weights.addLayout(row1)

        row2 = QHBoxLayout()
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setRange(0.0, 1.0)
        self.spin_Wa.setSingleStep(0.001)
        self.spin_Wa.setDecimals(4)
        self.spin_Wa.setValue(0.001)
        self.spin_Wa.setToolTip(
            "Weight for area difference in assignment cost.\n"
            "Higher = penalize size changes.\n"
            "Recommended: 0.001-0.01 (prevents size-based swaps)"
        )
        row2.addWidget(QLabel("Area:"))
        row2.addWidget(self.spin_Wa)

        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setRange(0.0, 10.0)
        self.spin_Wasp.setValue(0.1)
        self.spin_Wasp.setToolTip(
            "Weight for aspect ratio difference in assignment cost.\n"
            "Higher = penalize shape changes.\n"
            "Recommended: 0.05-0.2 (helps with occlusions)"
        )
        row2.addWidget(QLabel("Aspect Ratio:"))
        row2.addWidget(self.spin_Wasp)
        l_weights.addLayout(row2)

        self.chk_use_mahal = QCheckBox("Use Mahalanobis Distance")
        self.chk_use_mahal.setChecked(True)
        self.chk_use_mahal.setToolTip(
            "Use Mahalanobis distance instead of Euclidean for position.\n"
            "Accounts for velocity and uncertainty in motion prediction.\n"
            "Recommended: Enable for better handling of motion variability."
        )
        l_weights.addWidget(self.chk_use_mahal)
        vbox.addWidget(g_weights)

        # Assignment Algorithm (for large N optimization)
        g_assign = QGroupBox("Assignment Algorithm (for Large N)")
        vl_assign = QVBoxLayout(g_assign)
        vl_assign.addWidget(
            self._create_help_label(
                "Choose matching algorithm. Hungarian is optimal but slow for many animals (N>100). "
                "Greedy approximation is faster but may produce suboptimal assignments."
            )
        )
        f_assign = QFormLayout()

        self.combo_assignment_method = QComboBox()
        self.combo_assignment_method.addItems(
            ["Hungarian (Optimal)", "Greedy (Fast for N>100)"]
        )
        self.combo_assignment_method.setCurrentIndex(0)
        self.combo_assignment_method.setToolTip(
            "Hungarian: Optimal global assignment (slow for N>100)\n"
            "Greedy: Fast approximation for large N (200+)"
        )
        f_assign.addRow("Method:", self.combo_assignment_method)

        self.chk_spatial_optimization = QCheckBox(
            "Enable Spatial Optimization (KD-Tree)"
        )
        self.chk_spatial_optimization.setChecked(False)
        self.chk_spatial_optimization.setToolTip(
            "Uses KD-tree to reduce comparisons for large N (50+).\n"
            "Disable for small N (8-50) to reduce overhead."
        )
        f_assign.addRow(self.chk_spatial_optimization)

        vl_assign.addLayout(f_assign)
        vbox.addWidget(g_assign)

        # Orientation & Lifecycle
        g_misc = QGroupBox("Orientation & Lifecycle")
        vl_misc = QVBoxLayout(g_misc)
        vl_misc.addWidget(
            self._create_help_label(
                "Control how orientation is calculated based on movement. Moving animals can flip orientation instantly, "
                "stationary animals change orientation gradually within max angle limit."
            )
        )
        f_misc = QFormLayout()

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.1, 100.0)
        self.spin_velocity.setSingleStep(0.5)
        self.spin_velocity.setDecimals(2)
        self.spin_velocity.setValue(5.0)
        self.spin_velocity.setToolTip(
            "Velocity threshold (body-sizes/second) to classify as 'moving'.\n"
            "Below this = stationary (allows larger orientation changes).\n"
            "Above this = moving (instant orientation flip possible).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 2-10 body-sizes/s depending on animal speed."
        )
        f_misc.addRow("Motion Velocity Threshold (body/s):", self.spin_velocity)

        self.chk_instant_flip = QCheckBox("Instant Flip (Fast Motion)")
        self.chk_instant_flip.setChecked(True)
        self.chk_instant_flip.setToolTip(
            "Allow instant 180° orientation flip when moving quickly.\n"
            "Recommended: Enable for animals that can turn rapidly.\n"
            "Disable for slowly rotating animals."
        )
        f_misc.addRow(self.chk_instant_flip)

        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1, 180)
        self.spin_max_orient.setValue(30)
        self.spin_max_orient.setToolTip(
            "Maximum orientation change (degrees) when stationary (1-180).\n"
            "Larger = allow more rotation while stopped.\n"
            "Recommended: 20-45° (prevents orientation jitter)."
        )
        f_misc.addRow("Max Orient Δ (Stopped):", self.spin_max_orient)
        vl_misc.addLayout(f_misc)
        vbox.addWidget(g_misc)

        # Track Lifecycle
        g_lifecycle = QGroupBox("Track Lifecycle")
        vl_lifecycle = QVBoxLayout(g_lifecycle)
        vl_lifecycle.addWidget(
            self._create_help_label(
                "Control when tracks start and end. Lost frames determines how long to wait before terminating a track. "
                "Min respawn distance prevents creating duplicate IDs near existing animals."
            )
        )
        f_lifecycle = QFormLayout()

        self.spin_lost_thresh = QSpinBox()
        self.spin_lost_thresh.setRange(1, 100)
        self.spin_lost_thresh.setValue(10)
        self.spin_lost_thresh.setToolTip(
            "Number of frames without detection before track is terminated (1-100).\n"
            "Higher = tracks persist longer during occlusions.\n"
            "Lower = tracks end quickly, creating fragments.\n"
            "Recommended: 5-20 frames."
        )
        f_lifecycle.addRow("Lost Frames Threshold:", self.spin_lost_thresh)

        self.spin_min_respawn_distance = QDoubleSpinBox()
        self.spin_min_respawn_distance.setRange(0.0, 20.0)
        self.spin_min_respawn_distance.setSingleStep(0.5)
        self.spin_min_respawn_distance.setDecimals(2)
        self.spin_min_respawn_distance.setValue(2.5)
        self.spin_min_respawn_distance.setToolTip(
            "Minimum distance from existing tracks to spawn new track (×body size).\n"
            "Prevents creating duplicate tracks near existing animals.\n"
            "Recommended: 2-4× body size."
        )
        f_lifecycle.addRow("Min Respawn Dist (×body):", self.spin_min_respawn_distance)
        vl_lifecycle.addLayout(f_lifecycle)
        vbox.addWidget(g_lifecycle)

        # Stability
        g_stab = QGroupBox("Initialization Stability")
        vl_stab = QVBoxLayout(g_stab)
        vl_stab.addWidget(
            self._create_help_label(
                "Filter out unreliable tracks. Min detections to start prevents creating tracks from noise. "
                "Min detect/tracking frames removes short-lived false tracks in post-processing."
            )
        )
        f_stab = QFormLayout()
        self.spin_min_detections_to_start = QSpinBox()
        self.spin_min_detections_to_start.setRange(1, 50)
        self.spin_min_detections_to_start.setValue(1)
        self.spin_min_detections_to_start.setToolTip(
            "Minimum consecutive detections before starting a new track (1-50).\n"
            "Higher = fewer false tracks from noise, slower to start tracking.\n"
            "Lower = faster tracking startup, more noise-based tracks.\n"
            "Recommended: 1-3"
        )
        f_stab.addRow("Min Detections to Start:", self.spin_min_detections_to_start)

        self.spin_min_detect = QSpinBox()
        self.spin_min_detect.setRange(1, 500)
        self.spin_min_detect.setValue(10)
        self.spin_min_detect.setToolTip(
            "Minimum total detection frames to keep a track (1-500).\n"
            "Filters out short-lived false tracks in post-processing.\n"
            "Recommended: 5-20 frames."
        )
        f_stab.addRow("Min Detect Frames:", self.spin_min_detect)

        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(1, 500)
        self.spin_min_track.setValue(10)
        self.spin_min_track.setToolTip(
            "Minimum tracking frames (including predicted) to keep (1-500).\n"
            "Filters out tracks with too many gaps/predictions.\n"
            "Recommended: Similar to min detect frames."
        )
        f_stab.addRow("Min Tracking Frames:", self.spin_min_track)
        vl_stab.addLayout(f_stab)
        vbox.addWidget(g_stab)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_data_ui(self):
        """Tab 4: Post-Processing."""
        layout = QVBoxLayout(self.tab_data)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # Post-Processing
        g_pp = QGroupBox("Trajectory Post-Processing")
        vl_pp = QVBoxLayout(g_pp)
        vl_pp.addWidget(
            self._create_help_label(
                "Clean trajectories after tracking by removing outliers and splitting at identity swaps. "
                "Velocity/distance breaks detect unrealistic jumps that indicate ID switching."
            )
        )
        f_pp = QFormLayout()
        f_pp.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.enable_postprocessing = QCheckBox("Enable Automatic Cleaning")
        self.enable_postprocessing.setChecked(True)
        self.enable_postprocessing.setToolTip(
            "Automatically clean trajectories by removing outliers and fragments.\n"
            "Uses velocity and distance thresholds to detect anomalies.\n"
            "Recommended: Enable for cleaner data output."
        )
        f_pp.addRow(self.enable_postprocessing)

        self.spin_min_trajectory_length = QSpinBox()
        self.spin_min_trajectory_length.setRange(1, 1000)
        self.spin_min_trajectory_length.setValue(10)
        self.spin_min_trajectory_length.setToolTip(
            "Remove trajectories shorter than this (1-1000 frames).\n"
            "Filters out brief false detections and transient tracks.\n"
            "Recommended: 5-30 frames depending on video length."
        )
        f_pp.addRow("Min Length (frames):", self.spin_min_trajectory_length)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1.0, 500.0)
        self.spin_max_velocity_break.setSingleStep(5.0)
        self.spin_max_velocity_break.setDecimals(1)
        self.spin_max_velocity_break.setValue(50.0)
        self.spin_max_velocity_break.setToolTip(
            "Maximum velocity (body-sizes/second) before breaking trajectory.\n"
            "Splits tracks at unrealistic speed jumps (likely identity swaps).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 30-100 body-sizes/s for typical animal motion."
        )
        f_pp.addRow("Max Velocity Break (body/s):", self.spin_max_velocity_break)

        self.spin_max_distance_break = QDoubleSpinBox()
        self.spin_max_distance_break.setRange(1.0, 50.0)
        self.spin_max_distance_break.setSingleStep(0.5)
        self.spin_max_distance_break.setDecimals(2)
        self.spin_max_distance_break.setValue(15.0)
        self.spin_max_distance_break.setToolTip(
            "Maximum distance jump before breaking trajectory (×body size, 1-50).\n"
            "Splits tracks at unrealistic position jumps (likely identity swaps).\n"
            "Recommended: 10-20× body size."
        )
        f_pp.addRow("Max Distance Break (×body):", self.spin_max_distance_break)

        self.spin_max_occlusion_gap = QSpinBox()
        self.spin_max_occlusion_gap.setRange(0, 200)
        self.spin_max_occlusion_gap.setValue(30)
        self.spin_max_occlusion_gap.setToolTip(
            "Maximum consecutive occluded/lost frames before splitting trajectory (0-200).\n"
            "Prevents unreliable interpolation across long gaps.\n"
            "Set to 0 to disable occlusion-based splitting.\n"
            "Recommended: 20-50 frames for typical tracking scenarios."
        )
        f_pp.addRow("Max Occlusion Gap (frames):", self.spin_max_occlusion_gap)

        # Interpolation settings
        self.combo_interpolation_method = QComboBox()
        self.combo_interpolation_method.addItems(["None", "Linear", "Cubic", "Spline"])
        self.combo_interpolation_method.setCurrentText("None")
        self.combo_interpolation_method.setToolTip(
            "Interpolation method for filling gaps in trajectories:\n"
            "• None: No interpolation (keep NaN values)\n"
            "• Linear: Simple linear interpolation\n"
            "• Cubic: Smooth cubic spline interpolation\n"
            "• Spline: Smoothing spline with automatic smoothing\n"
            "Applied to X, Y positions and heading (circular interpolation)."
        )
        f_pp.addRow("Interpolation Method:", self.combo_interpolation_method)

        self.spin_interpolation_max_gap = QSpinBox()
        self.spin_interpolation_max_gap.setRange(1, 100)
        self.spin_interpolation_max_gap.setValue(10)
        self.spin_interpolation_max_gap.setToolTip(
            "Maximum gap size to interpolate (1-100 frames).\n"
            "Gaps larger than this will remain as NaN.\n"
            "Prevents interpolation across large occlusions.\n"
            "Recommended: 5-15 frames."
        )
        f_pp.addRow("Max Interpolation Gap:", self.spin_interpolation_max_gap)

        # Cleanup option
        self.chk_cleanup_temp_files = QCheckBox("Auto-cleanup temporary files")
        self.chk_cleanup_temp_files.setChecked(True)
        self.chk_cleanup_temp_files.setToolTip(
            "Automatically delete temporary files after successful tracking:\n"
            "• Reversed video files (*_reversed.mp4)\n"
            "• Intermediate CSV files (*_forward.csv, *_backward.csv)\n"
            "Keeps only final merged/processed output files."
        )
        f_pp.addRow("", self.chk_cleanup_temp_files)

        vl_pp.addLayout(f_pp)
        vbox.addWidget(g_pp)

        # Histograms
        g_hist = QGroupBox("Real-Time Analytics")
        vl_hist = QVBoxLayout(g_hist)
        vl_hist.addWidget(
            self._create_help_label(
                "Collect and visualize statistics during tracking. Useful for monitoring behavior patterns in real-time. "
                "History window controls how many recent frames to include in the analysis."
            )
        )
        f_hist = QFormLayout()
        self.enable_histograms = QCheckBox("Collect Histogram Data")
        self.enable_histograms.setToolTip(
            "Collect real-time statistics during tracking.\n"
            "Tracks speed, direction, and spatial distributions.\n"
            "Slight performance overhead but useful for monitoring."
        )
        f_hist.addRow(self.enable_histograms)

        self.spin_histogram_history = QSpinBox()
        self.spin_histogram_history.setRange(50, 5000)
        self.spin_histogram_history.setValue(300)
        self.spin_histogram_history.setToolTip(
            "Number of frames to include in rolling statistics (50-5000).\n"
            "Larger window = smoother trends but slower response.\n"
            "Recommended: 100-500 frames for most videos."
        )
        f_hist.addRow("History Window:", self.spin_histogram_history)

        self.btn_show_histograms = QPushButton("Open Plot Window")
        self.btn_show_histograms.setCheckable(True)
        self.btn_show_histograms.clicked.connect(self.toggle_histogram_window)
        f_hist.addRow(self.btn_show_histograms)
        vl_hist.addLayout(f_hist)
        vbox.addWidget(g_hist)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_viz_ui(self):
        """Tab 5: Visualization & Debug."""
        layout = QVBoxLayout(self.tab_viz)
        layout.setContentsMargins(10, 10, 10, 10)

        g_overlays = QGroupBox("Video Overlays - Common")
        v_ov = QVBoxLayout(g_overlays)
        v_ov.addWidget(
            self._create_help_label(
                "Choose which tracking information to display on the video. Toggle these on/off to reduce clutter "
                "or focus on specific aspects like trajectories or orientation."
            )
        )

        self.chk_show_circles = QCheckBox("Show Track Markers (Circles)")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip("Draw circles around tracked animals.")
        v_ov.addWidget(self.chk_show_circles)

        self.chk_show_orientation = QCheckBox("Show Orientation Lines")
        self.chk_show_orientation.setChecked(True)
        self.chk_show_orientation.setToolTip("Draw lines showing heading direction.")
        v_ov.addWidget(self.chk_show_orientation)

        self.chk_show_trajectories = QCheckBox("Show Trajectory Trails")
        self.chk_show_trajectories.setChecked(True)
        self.chk_show_trajectories.setToolTip(
            "Draw recent path history for each track."
        )
        v_ov.addWidget(self.chk_show_trajectories)

        self.chk_show_labels = QCheckBox("Show ID Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip("Display unique track IDs on each animal.")
        v_ov.addWidget(self.chk_show_labels)

        self.chk_show_state = QCheckBox("Show State Text")
        self.chk_show_state.setChecked(True)
        self.chk_show_state.setToolTip(
            "Display tracking state (ACTIVE, PREDICTED, etc.)."
        )
        v_ov.addWidget(self.chk_show_state)

        self.chk_show_kalman_uncertainty = QCheckBox("Show Kalman Uncertainty")
        self.chk_show_kalman_uncertainty.setChecked(False)
        self.chk_show_kalman_uncertainty.setToolTip(
            "Draw ellipses showing Kalman filter position uncertainty.\n"
            "Larger ellipse = more uncertainty in predicted position.\n"
            "Useful for debugging tracking quality and filter convergence."
        )
        v_ov.addWidget(self.chk_show_kalman_uncertainty)

        layout.addWidget(g_overlays)

        # Background Subtraction specific overlays
        self.g_overlays_bg = QGroupBox("Video Overlays - Background Subtraction")
        v_ov_bg = QVBoxLayout(self.g_overlays_bg)
        v_ov_bg.addWidget(
            self._create_help_label(
                "Debug background subtraction by viewing the foreground mask (detected movement) "
                "and background model (learned static image)."
            )
        )

        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        v_ov_bg.addWidget(self.chk_show_fg)

        self.chk_show_bg = QCheckBox("Show Background Model")
        self.chk_show_bg.setChecked(True)
        v_ov_bg.addWidget(self.chk_show_bg)

        layout.addWidget(self.g_overlays_bg)

        # YOLO specific overlays
        self.g_overlays_yolo = QGroupBox("Video Overlays - YOLO")
        v_ov_yolo = QVBoxLayout(self.g_overlays_yolo)
        v_ov_yolo.addWidget(
            self._create_help_label(
                "Show oriented bounding boxes from YOLO detection. Useful for debugging detection quality "
                "and verifying model performance."
            )
        )

        self.chk_show_yolo_obb = QCheckBox("Show YOLO OBB Detection Boxes")
        self.chk_show_yolo_obb.setChecked(False)
        v_ov_yolo.addWidget(self.chk_show_yolo_obb)

        layout.addWidget(self.g_overlays_yolo)

        g_settings = QGroupBox("Display Settings")
        vl_settings = QVBoxLayout(g_settings)
        vl_settings.addWidget(
            self._create_help_label(
                "Control how much trajectory history to display. Longer trails show more path context "
                "but can clutter the view when many animals are tracked."
            )
        )
        f_disp = QFormLayout()
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setRange(1, 60)
        self.spin_traj_hist.setValue(5)
        self.spin_traj_hist.setToolTip(
            "Length of trajectory trails to display (1-60 seconds).\n"
            "Longer = more visible path history but more cluttered.\n"
            "Recommended: 3-10 seconds."
        )
        f_disp.addRow("Trail History (sec):", self.spin_traj_hist)
        vl_settings.addLayout(f_disp)
        layout.addWidget(g_settings)

        g_debug = QGroupBox("Advanced / Debug")
        v_dbg = QVBoxLayout(g_debug)
        v_dbg.addWidget(
            self._create_help_label(
                "Enable verbose logging to see detailed tracking decisions. Useful for troubleshooting "
                "but generates large log files. Disable for production runs."
            )
        )
        self.chk_debug_logging = QCheckBox("Enable Verbose Debug Logging")
        self.chk_debug_logging.stateChanged.connect(self.toggle_debug_logging)
        v_dbg.addWidget(self.chk_debug_logging)
        layout.addWidget(g_debug)

        layout.addStretch()

    def setup_dataset_ui(self):
        """Tab 6: Dataset Generation for Active Learning."""
        layout = QVBoxLayout(self.tab_dataset)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(10, 10, 10, 10)

        # Info box
        info_box = QGroupBox("Active Learning - Training Dataset Generation")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(
            self._create_help_label(
                "Automatically identify challenging frames during tracking and export them "
                "for annotation. This helps improve YOLO model performance through active learning.\n\n"
                "The system will:\n"
                "• Identify frames with low detection confidence, assignment issues, or tracking failures\n"
                "• Select the worst N frames while ensuring visual diversity\n"
                "• Export frames with YOLO-format annotations for initial labeling\n"
                "• Package everything in a zip file ready for x-AnyLabeling\n\n"
                "Recommended workflow:\n"
                "1. Run tracking with dataset generation enabled\n"
                "2. Use x-AnyLabeling to correct/refine the exported annotations\n"
                "3. Train an improved YOLO model with the new labeled data"
            )
        )
        form.addWidget(info_box)

        # Enable dataset generation
        self.chk_enable_dataset_gen = QCheckBox(
            "Enable Active Learning Dataset Generation"
        )
        self.chk_enable_dataset_gen.setChecked(False)
        self.chk_enable_dataset_gen.setToolTip(
            "Enable automatic generation of training dataset from difficult frames."
        )
        self.chk_enable_dataset_gen.stateChanged.connect(
            self._on_dataset_generation_toggled
        )
        form.addWidget(self.chk_enable_dataset_gen)

        # Dataset configuration
        self.g_dataset_config = QGroupBox("Dataset Configuration")
        f_config = QFormLayout(self.g_dataset_config)
        f_config.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Dataset name
        self.line_dataset_name = QLineEdit()
        self.line_dataset_name.setPlaceholderText("e.g., my_dataset_v1")
        self.line_dataset_name.setToolTip(
            "Name for the dataset (used for folder and zip file naming)."
        )
        f_config.addRow("Dataset Name:", self.line_dataset_name)

        # Class name
        self.line_dataset_class_name = QLineEdit()
        self.line_dataset_class_name.setPlaceholderText("e.g., ant")
        self.line_dataset_class_name.setText("object")
        self.line_dataset_class_name.setToolTip(
            "Name of the object class being tracked.\n"
            "This will be used in the classes.txt file for YOLO training.\n"
            "Examples: ant, bee, mouse, fish, etc."
        )
        f_config.addRow("Class Name:", self.line_dataset_class_name)

        # Output directory
        h_output = QHBoxLayout()
        self.line_dataset_output = QLineEdit()
        self.line_dataset_output.setPlaceholderText("Select output directory...")
        self.line_dataset_output.setToolTip(
            "Directory where the dataset will be saved."
        )
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self._select_dataset_output_dir)
        h_output.addWidget(self.line_dataset_output)
        h_output.addWidget(btn_browse_output)
        f_config.addRow("Output Directory:", h_output)

        form.addWidget(self.g_dataset_config)

        # Frame selection parameters
        self.g_frame_selection = QGroupBox("Frame Selection Criteria")
        f_selection = QFormLayout(self.g_frame_selection)
        f_selection.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Number of frames to export
        self.spin_dataset_max_frames = QSpinBox()
        self.spin_dataset_max_frames.setRange(10, 1000)
        self.spin_dataset_max_frames.setValue(100)
        self.spin_dataset_max_frames.setToolTip(
            "Maximum number of frames to export (10-1000).\n"
            "Higher values provide more training data but increase annotation time.\n"
            "Recommended: 50-200 frames for initial improvement."
        )
        f_selection.addRow("Max Frames to Export:", self.spin_dataset_max_frames)

        # Confidence threshold
        self.spin_dataset_conf_threshold = QDoubleSpinBox()
        self.spin_dataset_conf_threshold.setRange(0.0, 1.0)
        self.spin_dataset_conf_threshold.setSingleStep(0.05)
        self.spin_dataset_conf_threshold.setDecimals(2)
        self.spin_dataset_conf_threshold.setValue(0.5)
        self.spin_dataset_conf_threshold.setToolTip(
            "Flag frames where YOLO detection confidence is below this threshold (0.0-1.0).\n"
            "Lower = only export very uncertain detections.\n"
            "Higher = include moderately uncertain detections.\n"
            "Recommended: 0.4-0.6"
        )
        f_selection.addRow("Confidence Threshold:", self.spin_dataset_conf_threshold)

        # Visual diversity window
        self.spin_dataset_diversity_window = QSpinBox()
        self.spin_dataset_diversity_window.setRange(10, 500)
        self.spin_dataset_diversity_window.setValue(30)
        self.spin_dataset_diversity_window.setToolTip(
            "Minimum frame separation for visual diversity (10-500 frames).\n"
            "Prevents selecting too many consecutive similar frames.\n"
            "Higher = more spread out frames, more visual variety.\n"
            "Recommended: 20-50 frames (depends on video frame rate)."
        )
        f_selection.addRow(
            "Diversity Window (frames):", self.spin_dataset_diversity_window
        )

        # Include context frames
        self.chk_dataset_include_context = QCheckBox("Include Context Frames (±1)")
        self.chk_dataset_include_context.setChecked(True)
        self.chk_dataset_include_context.setToolTip(
            "Export the frame before and after each selected frame.\n"
            "Provides temporal context which can improve annotation quality.\n"
            "Increases dataset size by 3x."
        )
        f_selection.addRow("Context Frames:", self.chk_dataset_include_context)

        self.chk_dataset_probabilistic = QCheckBox("Probabilistic Sampling")
        self.chk_dataset_probabilistic.setChecked(True)
        self.chk_dataset_probabilistic.setToolTip(
            "Use rank-based probabilistic sampling instead of greedy selection.\n"
            "Probabilistic: Higher quality scores = higher probability (more variety).\n"
            "Greedy: Always select absolute worst frames first (may be too extreme).\n"
            "Recommended: Enabled for better training data diversity."
        )
        f_selection.addRow("Sampling Strategy:", self.chk_dataset_probabilistic)

        form.addWidget(self.g_frame_selection)

        # Quality metrics
        self.g_quality_metrics = QGroupBox("Quality Metrics (Frame Scoring)")
        v_metrics = QVBoxLayout(self.g_quality_metrics)
        v_metrics.addWidget(
            self._create_help_label(
                "Select which quality metrics to use for identifying problematic frames:"
            )
        )

        self.chk_metric_low_confidence = QCheckBox("Low Detection Confidence")
        self.chk_metric_low_confidence.setChecked(True)
        self.chk_metric_low_confidence.setToolTip(
            "Flag frames where YOLO confidence is below threshold."
        )
        v_metrics.addWidget(self.chk_metric_low_confidence)

        self.chk_metric_count_mismatch = QCheckBox("Detection Count Mismatch")
        self.chk_metric_count_mismatch.setChecked(True)
        self.chk_metric_count_mismatch.setToolTip(
            "Flag frames where detected count doesn't match expected number of animals."
        )
        v_metrics.addWidget(self.chk_metric_count_mismatch)

        self.chk_metric_high_assignment_cost = QCheckBox("High Assignment Cost")
        self.chk_metric_high_assignment_cost.setChecked(True)
        self.chk_metric_high_assignment_cost.setToolTip(
            "Flag frames where tracker struggles to match detections to tracks."
        )
        v_metrics.addWidget(self.chk_metric_high_assignment_cost)

        self.chk_metric_track_loss = QCheckBox("Frequent Track Losses")
        self.chk_metric_track_loss.setChecked(True)
        self.chk_metric_track_loss.setToolTip(
            "Flag frames where tracks are frequently lost."
        )
        v_metrics.addWidget(self.chk_metric_track_loss)

        self.chk_metric_high_uncertainty = QCheckBox("High Position Uncertainty")
        self.chk_metric_high_uncertainty.setChecked(False)
        self.chk_metric_high_uncertainty.setToolTip(
            "Flag frames where Kalman filter is very uncertain about positions."
        )
        v_metrics.addWidget(self.chk_metric_high_uncertainty)

        form.addWidget(self.g_quality_metrics)

        # Add visual separator between sections
        separator_frame = QFrame()
        separator_frame.setFrameShape(QFrame.HLine)
        separator_frame.setFrameShadow(QFrame.Sunken)
        separator_frame.setStyleSheet("background-color: #555; margin: 20px 0px;")
        separator_frame.setMinimumHeight(2)
        form.addWidget(separator_frame)

        # Section header for pose tracking
        pose_section_label = QLabel("Pose Tracking Dataset Export")
        pose_section_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #4a9eff; margin-top: 10px;"
        )
        form.addWidget(pose_section_label)

        # Pose Tracking Dataset Export
        self.g_pose_export = QGroupBox("Configuration")
        vl_pose = QVBoxLayout(self.g_pose_export)
        vl_pose.addWidget(
            self._create_help_label(
                "Export individual trajectory videos for pose estimation training. "
                "After tracking is complete, generates cropped videos centered on each animal, "
                "ready for annotation with DeepLabCut, SLEAP, or other pose tracking tools.\n\n"
                "This is independent from active learning dataset generation above."
            )
        )

        # Enable checkbox
        self.chk_enable_pose_export = QCheckBox("Enable Pose Tracking Export")
        self.chk_enable_pose_export.setChecked(False)
        self.chk_enable_pose_export.toggled.connect(self._on_pose_export_toggled)
        vl_pose.addWidget(self.chk_enable_pose_export)

        # Configuration group
        pose_config_group = QGroupBox("Export Configuration")
        fl_pose = QFormLayout(pose_config_group)
        fl_pose.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl_pose.setSpacing(8)

        # Output Directory
        self.line_pose_output_dir = QLineEdit()
        self.line_pose_output_dir.setPlaceholderText("directory/for/pose/datasets")
        self.line_pose_output_dir.setEnabled(False)
        btn_select_pose_dir = QPushButton("Browse...")
        btn_select_pose_dir.clicked.connect(self._select_pose_output_dir)
        btn_select_pose_dir.setEnabled(False)
        self.btn_select_pose_dir = btn_select_pose_dir
        pose_dir_layout = QHBoxLayout()
        pose_dir_layout.addWidget(self.line_pose_output_dir, 1)
        pose_dir_layout.addWidget(btn_select_pose_dir)
        fl_pose.addRow("Output Directory:", pose_dir_layout)

        # Dataset Name
        self.line_pose_dataset_name = QLineEdit()
        self.line_pose_dataset_name.setPlaceholderText("my_pose_dataset")
        self.line_pose_dataset_name.setText("pose_dataset")
        self.line_pose_dataset_name.setEnabled(False)
        self.line_pose_dataset_name.setToolTip(
            "Base name for pose tracking dataset (timestamp will be appended)"
        )
        fl_pose.addRow("Dataset Name:", self.line_pose_dataset_name)

        vl_pose.addWidget(pose_config_group)

        # Parameters group
        pose_params_group = QGroupBox("Export Parameters")
        fl_params = QFormLayout(pose_params_group)
        fl_params.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl_params.setSpacing(8)

        # Crop Multiplier
        self.spin_pose_crop_multiplier = QDoubleSpinBox()
        self.spin_pose_crop_multiplier.setRange(1.0, 10.0)
        self.spin_pose_crop_multiplier.setValue(4.0)
        self.spin_pose_crop_multiplier.setSingleStep(0.5)
        self.spin_pose_crop_multiplier.setDecimals(1)
        self.spin_pose_crop_multiplier.setEnabled(False)
        self.spin_pose_crop_multiplier.setToolTip(
            "Crop size for pose videos = body_size × multiplier\n"
            "Should include full body + margin for pose estimation"
        )
        fl_params.addRow("Crop Multiplier:", self.spin_pose_crop_multiplier)

        # Min Trajectory Length
        self.spin_pose_min_length = QSpinBox()
        self.spin_pose_min_length.setRange(5, 1000)
        self.spin_pose_min_length.setValue(30)
        self.spin_pose_min_length.setSingleStep(5)
        self.spin_pose_min_length.setEnabled(False)
        self.spin_pose_min_length.setToolTip(
            "Minimum trajectory length (frames) to export.\n"
            "Shorter trajectories are skipped."
        )
        fl_params.addRow("Min Length (frames):", self.spin_pose_min_length)

        # Export FPS
        self.spin_pose_export_fps = QSpinBox()
        self.spin_pose_export_fps.setRange(1, 120)
        self.spin_pose_export_fps.setValue(30)
        self.spin_pose_export_fps.setSingleStep(1)
        self.spin_pose_export_fps.setEnabled(False)
        self.spin_pose_export_fps.setToolTip("Frame rate for exported pose videos")
        fl_params.addRow("Export FPS:", self.spin_pose_export_fps)

        vl_pose.addWidget(pose_params_group)

        # Add note about automatic export
        note_label = QLabel(
            "Note: Pose dataset will be automatically exported after tracking completes."
        )
        note_label.setStyleSheet("color: #888; font-style: italic; margin-top: 5px;")
        vl_pose.addWidget(note_label)

        form.addWidget(self.g_pose_export)

        # Initially disable dataset generation config widgets
        # Pose export is independent and stays enabled
        self.g_dataset_config.setEnabled(False)
        self.g_frame_selection.setEnabled(False)
        self.g_quality_metrics.setEnabled(False)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_individual_analysis_ui(self):
        """Tab 7: Individual Analysis - Real-time Identity & Post-hoc Pose Analysis."""
        layout = QVBoxLayout(self.tab_individual)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)

        # Info box
        info_box = QGroupBox("Individual-Level Processing")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(
            self._create_help_label(
                "Process individual animals for identity classification and pose estimation.\n\n"
                "• Real-time Identity: Classify individual animals during tracking (color tags, AprilTags)\n"
                "• Post-hoc Pose Tracking: Run pose estimation on exported trajectory videos (coming soon)\n\n"
                "Note: Pose dataset export is available in the 'Dataset Generation' tab."
            )
        )
        form.addWidget(info_box)

        # Main Enable Checkbox
        self.chk_enable_individual_analysis = QCheckBox(
            "Enable Real-time Identity Classification"
        )
        self.chk_enable_individual_analysis.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4a9eff;"
        )
        self.chk_enable_individual_analysis.toggled.connect(
            self._on_individual_analysis_toggled
        )
        form.addWidget(self.chk_enable_individual_analysis)

        # Identity Classification Section
        self.g_identity = QGroupBox("Identity Classification (Real-time)")
        vl_identity = QVBoxLayout(self.g_identity)
        vl_identity.addWidget(
            self._create_help_label(
                "Classify individual identity during tracking. Extracts crops around each detection "
                "and processes them with the selected method."
            )
        )

        fl_identity = QFormLayout()
        fl_identity.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Identity Method
        self.combo_identity_method = QComboBox()
        self.combo_identity_method.addItems(
            ["None (Disabled)", "Color Tags (YOLO)", "AprilTags", "Custom"]
        )
        self.combo_identity_method.setToolTip(
            "Select method for identifying individual animals:\n"
            "• Color Tags: Detect color markers using YOLO model\n"
            "• AprilTags: Detect fiducial markers\n"
            "• Custom: Implement your own classifier"
        )
        self.combo_identity_method.currentIndexChanged.connect(
            self._on_identity_method_changed
        )
        fl_identity.addRow("Identity Method:", self.combo_identity_method)

        # Model/Config for identity (stacked widget for different methods)
        self.identity_config_stack = QStackedWidget()

        # Page 0: None
        none_widget = QWidget()
        none_layout = QVBoxLayout(none_widget)
        none_layout.addWidget(
            self._create_help_label("Identity classification is disabled.")
        )
        self.identity_config_stack.addWidget(none_widget)

        # Page 1: Color Tags (YOLO)
        color_widget = QWidget()
        color_layout = QFormLayout(color_widget)
        self.line_color_tag_model = QLineEdit()
        self.line_color_tag_model.setPlaceholderText("path/to/color_tag_model.pt")
        btn_select_color_model = QPushButton("Browse...")
        btn_select_color_model.clicked.connect(self._select_color_tag_model)
        color_model_layout = QHBoxLayout()
        color_model_layout.addWidget(self.line_color_tag_model)
        color_model_layout.addWidget(btn_select_color_model)
        color_layout.addRow("Model File:", color_model_layout)
        self.spin_color_tag_conf = QDoubleSpinBox()
        self.spin_color_tag_conf.setRange(0.01, 1.0)
        self.spin_color_tag_conf.setValue(0.5)
        self.spin_color_tag_conf.setSingleStep(0.05)
        self.spin_color_tag_conf.setToolTip(
            "Minimum confidence for color tag detection"
        )
        color_layout.addRow("Confidence:", self.spin_color_tag_conf)
        self.identity_config_stack.addWidget(color_widget)

        # Page 2: AprilTags
        apriltag_widget = QWidget()
        apriltag_layout = QFormLayout(apriltag_widget)
        self.combo_apriltag_family = QComboBox()
        self.combo_apriltag_family.addItems(
            ["tag36h11", "tag25h9", "tag16h5", "tagCircle21h7", "tagStandard41h12"]
        )
        self.combo_apriltag_family.setToolTip("AprilTag family to detect")
        apriltag_layout.addRow("Tag Family:", self.combo_apriltag_family)
        self.spin_apriltag_decimate = QDoubleSpinBox()
        self.spin_apriltag_decimate.setRange(1.0, 4.0)
        self.spin_apriltag_decimate.setValue(1.0)
        self.spin_apriltag_decimate.setSingleStep(0.5)
        self.spin_apriltag_decimate.setToolTip(
            "Decimation factor for faster detection (higher = faster but less accurate)"
        )
        apriltag_layout.addRow("Decimate:", self.spin_apriltag_decimate)
        self.identity_config_stack.addWidget(apriltag_widget)

        # Page 3: Custom
        custom_widget = QWidget()
        custom_layout = QVBoxLayout(custom_widget)
        custom_layout.addWidget(
            self._create_help_label(
                "Implement custom identity classifier in:\n"
                "src/multi_tracker/core/individual_analysis.py"
            )
        )
        self.identity_config_stack.addWidget(custom_widget)

        vl_identity.addLayout(fl_identity)
        vl_identity.addWidget(self.identity_config_stack)

        # Crop Parameters
        crop_group = QGroupBox("Crop Parameters")
        crop_layout = QFormLayout(crop_group)

        self.spin_identity_crop_multiplier = QDoubleSpinBox()
        self.spin_identity_crop_multiplier.setRange(1.0, 10.0)
        self.spin_identity_crop_multiplier.setValue(3.0)
        self.spin_identity_crop_multiplier.setSingleStep(0.5)
        self.spin_identity_crop_multiplier.setDecimals(1)
        self.spin_identity_crop_multiplier.setToolTip(
            "Crop size = body_size × multiplier\n"
            "Larger values include more context, smaller values focus on the animal"
        )
        crop_layout.addRow("Size Multiplier:", self.spin_identity_crop_multiplier)

        self.spin_identity_crop_min = QSpinBox()
        self.spin_identity_crop_min.setRange(32, 512)
        self.spin_identity_crop_min.setValue(64)
        self.spin_identity_crop_min.setSingleStep(16)
        self.spin_identity_crop_min.setToolTip("Minimum crop size in pixels")
        crop_layout.addRow("Min Size (px):", self.spin_identity_crop_min)

        self.spin_identity_crop_max = QSpinBox()
        self.spin_identity_crop_max.setRange(64, 1024)
        self.spin_identity_crop_max.setValue(256)
        self.spin_identity_crop_max.setSingleStep(16)
        self.spin_identity_crop_max.setToolTip("Maximum crop size in pixels")
        crop_layout.addRow("Max Size (px):", self.spin_identity_crop_max)

        vl_identity.addWidget(crop_group)

        form.addWidget(self.g_identity)

        # ============================================================
        # Individual Dataset Generator Section (Real-time OBB crops)
        # ============================================================
        self.g_individual_dataset = QGroupBox("Real-time Individual Dataset (YOLO OBB)")
        vl_ind_dataset = QVBoxLayout(self.g_individual_dataset)
        vl_ind_dataset.addWidget(
            self._create_help_label(
                "Generate a clean dataset of isolated individuals during tracking.\n\n"
                "• Extracts OBB-masked crops in real-time as tracking runs\n"
                "• Only the detected animal (within OBB) is visible, rest is masked\n"
                "• Perfect for training identity classifiers or pose models\n\n"
                "Note: Only available when using YOLO OBB detection."
            )
        )

        self.chk_enable_individual_dataset = QCheckBox(
            "Enable Real-time Individual Dataset Generation"
        )
        self.chk_enable_individual_dataset.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4a9eff;"
        )
        self.chk_enable_individual_dataset.toggled.connect(
            self._on_individual_dataset_toggled
        )
        vl_ind_dataset.addWidget(self.chk_enable_individual_dataset)

        # Output Configuration
        ind_output_group = QGroupBox("Output Configuration")
        ind_output_layout = QFormLayout(ind_output_group)
        ind_output_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Output directory
        h_ind_output = QHBoxLayout()
        self.line_individual_output = QLineEdit()
        self.line_individual_output.setPlaceholderText("Select output directory...")
        self.line_individual_output.setToolTip(
            "Directory where individual crops will be saved."
        )
        self.line_individual_output.setEnabled(False)
        btn_browse_ind_output = QPushButton("Browse...")
        btn_browse_ind_output.clicked.connect(self._select_individual_output_dir)
        h_ind_output.addWidget(self.line_individual_output)
        h_ind_output.addWidget(btn_browse_ind_output)
        ind_output_layout.addRow("Output Directory:", h_ind_output)

        # Output format
        self.combo_individual_format = QComboBox()
        self.combo_individual_format.addItems(["PNG", "JPEG"])
        self.combo_individual_format.setToolTip(
            "PNG: Lossless, larger files\nJPEG: Smaller files, slight quality loss"
        )
        self.combo_individual_format.setEnabled(False)
        ind_output_layout.addRow("Image Format:", self.combo_individual_format)

        # Save interval
        self.spin_individual_interval = QSpinBox()
        self.spin_individual_interval.setRange(1, 100)
        self.spin_individual_interval.setValue(1)
        self.spin_individual_interval.setSingleStep(1)
        self.spin_individual_interval.setToolTip(
            "Save crops every N frames.\n"
            "1 = every frame, 10 = every 10th frame, etc."
        )
        self.spin_individual_interval.setEnabled(False)
        ind_output_layout.addRow("Save Every N Frames:", self.spin_individual_interval)

        # Padding fraction (only crop parameter needed - size is determined by OBB)
        self.spin_individual_padding = QDoubleSpinBox()
        self.spin_individual_padding.setRange(0.0, 0.5)
        self.spin_individual_padding.setValue(0.1)
        self.spin_individual_padding.setSingleStep(0.05)
        self.spin_individual_padding.setDecimals(2)
        self.spin_individual_padding.setToolTip(
            "Padding around OBB bounding box as fraction of size.\n"
            "0.1 = 10% padding on each side."
        )
        self.spin_individual_padding.setEnabled(False)
        ind_output_layout.addRow("Padding Fraction:", self.spin_individual_padding)

        vl_ind_dataset.addWidget(ind_output_group)

        # Info label about filtering
        info_label = self._create_help_label(
            "Note: Crops use detections already filtered by ROI and size settings.\n"
            "No additional filtering parameters needed."
        )
        vl_ind_dataset.addWidget(info_label)

        form.addWidget(self.g_individual_dataset)

        # Post-hoc Pose Analysis Section (Placeholder)
        self.g_pose_analysis = QGroupBox("Post-hoc Pose Tracking Analysis")
        vl_pose_analysis = QVBoxLayout(self.g_pose_analysis)
        vl_pose_analysis.addWidget(
            self._create_help_label(
                "Run pose estimation on exported trajectory videos.\n\n"
                "Workflow:\n"
                "1. Export pose dataset from 'Dataset Generation' tab\n"
                "2. Annotate keypoints in DeepLabCut/SLEAP\n"
                "3. Train pose estimation model\n"
                "4. Use this section to apply trained model to all trajectories (coming soon)"
            )
        )

        self.chk_enable_pose_analysis = QCheckBox("Enable Post-hoc Pose Analysis")
        self.chk_enable_pose_analysis.setChecked(False)
        self.chk_enable_pose_analysis.setEnabled(False)
        self.chk_enable_pose_analysis.setToolTip(
            "Feature coming soon: Apply trained pose models to trajectory videos"
        )
        vl_pose_analysis.addWidget(self.chk_enable_pose_analysis)

        placeholder_label = QLabel("⚠️ This feature is under development")
        placeholder_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 10px;"
        )
        vl_pose_analysis.addWidget(placeholder_label)

        form.addWidget(self.g_pose_analysis)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Initially disable all controls
        self.g_identity.setEnabled(False)
        self.g_pose_analysis.setEnabled(False)

    def _on_dataset_generation_toggled(self, state):
        """Enable/disable dataset generation controls."""
        enabled = state == Qt.Checked
        self.g_dataset_config.setEnabled(enabled)
        self.g_frame_selection.setEnabled(enabled)
        self.g_quality_metrics.setEnabled(enabled)
        # Pose export is independent, doesn't need to be enabled with dataset gen

    def _select_dataset_output_dir(self):
        """Browse for dataset output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Dataset Output Directory"
        )
        if directory:
            self.line_dataset_output.setText(directory)

    def _on_individual_analysis_toggled(self, state):
        """Enable/disable individual analysis controls."""
        enabled = state == Qt.Checked
        self.g_identity.setEnabled(enabled)
        # g_pose_analysis stays disabled for now (placeholder)

    def _on_identity_method_changed(self, index):
        """Update identity configuration stack when method changes."""
        self.identity_config_stack.setCurrentIndex(index)

    def _select_color_tag_model(self):
        """Browse for color tag YOLO model."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Color Tag YOLO Model", "", "YOLO Models (*.pt *.onnx)"
        )
        if filepath:
            self.line_color_tag_model.setText(filepath)

    def _on_pose_export_toggled(self, state):
        """Enable/disable pose export controls."""
        enabled = state  # toggled signal passes boolean, not Qt.CheckState
        self.line_pose_output_dir.setEnabled(enabled)
        self.btn_select_pose_dir.setEnabled(enabled)
        self.line_pose_dataset_name.setEnabled(enabled)
        self.spin_pose_crop_multiplier.setEnabled(enabled)
        self.spin_pose_min_length.setEnabled(enabled)
        self.spin_pose_export_fps.setEnabled(enabled)

    def _select_pose_output_dir(self):
        """Browse for pose dataset output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Pose Dataset Output Directory"
        )
        if directory:
            self.line_pose_output_dir.setText(directory)

    def _on_individual_dataset_toggled(self, enabled):
        """Enable/disable individual dataset generation controls."""
        self.line_individual_output.setEnabled(enabled)
        self.combo_individual_format.setEnabled(enabled)
        self.spin_individual_interval.setEnabled(enabled)
        self.spin_individual_padding.setEnabled(enabled)

    def _select_individual_output_dir(self):
        """Browse for individual dataset output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Individual Dataset Output Directory"
        )
        if directory:
            self.line_individual_output.setText(directory)

    def _export_pose_dataset(self):
        """Export pose tracking dataset from trajectories."""
        from ..core.individual_analysis import PoseTrackingExporter

        # Get parameters
        video_path = self.file_line.text()
        raw_csv_path = self.csv_line.text()
        output_dir = self.line_pose_output_dir.text()
        dataset_name = self.line_pose_dataset_name.text() or "pose_dataset"

        # Determine which CSV to use based on tracking mode
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            # Check for merged file first (backward tracking completed)
            merged_csv = f"{base}_merged.csv"
            if os.path.exists(merged_csv):
                csv_path = merged_csv
                logger.info("Using merged trajectories for pose export")
            else:
                # Use forward processed file (forward-only tracking)
                processed_csv = f"{base}_forward_processed{ext}"
                if os.path.exists(processed_csv):
                    csv_path = processed_csv
                    logger.info("Using forward processed trajectories for pose export")
                else:
                    # Fallback to raw CSV
                    csv_path = raw_csv_path
                    logger.warning(
                        "Using raw CSV for pose export (no processed file found)"
                    )
        else:
            csv_path = raw_csv_path

        # Validate inputs
        if not video_path or not os.path.exists(video_path):
            logger.error("No valid video file selected")
            return

        if not csv_path or not os.path.exists(csv_path):
            logger.error("No tracking CSV found. Complete tracking first.")
            return

        if not output_dir:
            logger.error("No output directory selected")
            return

        # Create parameters dict
        params = {
            "ENABLE_POSE_EXPORT": True,
            "POSE_CROP_SIZE_MULTIPLIER": self.spin_pose_crop_multiplier.value(),
            "POSE_MIN_TRAJECTORY_LENGTH": self.spin_pose_min_length.value(),
            "POSE_EXPORT_FPS": self.spin_pose_export_fps.value(),
            "REFERENCE_BODY_SIZE": self.spin_reference_body_size.value(),
        }

        # Create exporter with progress callback
        def update_progress(percentage, status):
            self.progress_bar.setValue(percentage)
            self.progress_label.setText(status)
            QApplication.processEvents()

        exporter = PoseTrackingExporter(params, progress_callback=update_progress)

        # Show progress UI
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting pose dataset export...")
        QApplication.processEvents()

        logger.info("Starting pose tracking dataset export...")

        try:
            export_path = exporter.export_trajectories(
                video_path, csv_path, output_dir, dataset_name
            )
            if export_path:
                logger.info(f"✓ Pose dataset exported successfully to: {export_path}")
        except Exception as e:
            logger.error(f"Failed to export pose dataset: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Hide progress UI
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    # =========================================================================
    # EVENT HANDLERS (Identical Logic to Original)
    # =========================================================================

    def _on_detection_method_changed_ui(self, index):
        """Update stack widget when detection method changes."""
        self.stack_detection.setCurrentIndex(index)
        # Show image adjustments only for Background Subtraction (index 0)
        is_background_subtraction = index == 0
        self.g_img.setVisible(is_background_subtraction)
        # Show/hide method-specific overlay groups
        self.g_overlays_bg.setVisible(is_background_subtraction)
        self.g_overlays_yolo.setVisible(not is_background_subtraction)
        # Refresh preview to show correct mode
        self._update_preview_display()
        self.on_detection_method_changed(index)

    def select_file(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if fp:
            self.file_line.setText(fp)
            self.current_video_path = fp
            if self.roi_selection_active:
                self.clear_roi()

            # Auto-generate output paths based on video name
            video_dir = os.path.dirname(fp)
            video_name = os.path.splitext(os.path.basename(fp))[0]

            # Auto-populate CSV output
            csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
            self.csv_line.setText(csv_path)

            # Auto-populate video output and enable it
            video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
            self.video_out_line.setText(video_out_path)
            self.check_video_output.setChecked(True)

            # Enable preview refresh button and load a random frame
            self.btn_refresh_preview.setEnabled(True)
            self.btn_test_detection.setEnabled(True)
            self.btn_detect_fps.setEnabled(True)

            self._load_preview_frame()

            # Auto-load config if it exists for this video
            config_path = get_video_config_path(fp)
            if config_path and os.path.isfile(config_path):
                self._load_config_from_file(config_path)
                self.config_status_label.setText(
                    f"✓ Loaded: {os.path.basename(config_path)}"
                )
                self.config_status_label.setStyleSheet(
                    "color: #4a9eff; font-style: italic; font-size: 10px;"
                )
                logger.info(
                    f"Video selected: {fp} (auto-loaded config from {config_path})"
                )
            else:
                self.config_status_label.setText(
                    "No config found (using current settings)"
                )
                self.config_status_label.setStyleSheet(
                    "color: #f39c12; font-style: italic; font-size: 10px;"
                )
                logger.info(
                    f"Video selected: {fp} (no config found, using current settings)"
                )

    def select_csv(self):
        fp, _ = QFileDialog.getSaveFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if fp:
            self.csv_line.setText(fp)

    def select_video_output(self):
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self.video_out_line.setText(fp)

    def _load_preview_frame(self):
        """Load a random frame from the video for live preview."""
        if not self.current_video_path:
            return

        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for preview")
            return

        # Get total frames and pick a random one
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            random_frame_idx = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)

        ret, frame = cap.read()
        cap.release()

        if ret:
            # Store original frame for adjustments
            self.preview_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Clear any previous detection test result
            self.detection_test_result = None
            self._update_preview_display()
            # Auto-fit to screen - use QTimer to ensure display is updated first
            from PySide2.QtCore import QTimer
            QTimer.singleShot(10, self._fit_image_to_screen)
            logger.info(f"Loaded preview frame {random_frame_idx}/{total_frames}")
        else:
            logger.warning("Failed to read preview frame")

    def _on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.label_brightness_val.setText(str(value))
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_contrast_changed(self, value):
        """Handle contrast slider change."""
        contrast_val = value / 100.0
        self.label_contrast_val.setText(f"{contrast_val:.2f}")
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_gamma_changed(self, value):
        """Handle gamma slider change."""
        gamma_val = value / 100.0
        self.label_gamma_val.setText(f"{gamma_val:.2f}")
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_zoom_changed(self, value):
        """Handle zoom slider change."""
        zoom_val = value / 100.0
        self.label_zoom_val.setText(f"{zoom_val:.2f}x")
        # If detection test result exists, redisplay it; otherwise show preview
        if self.detection_test_result is not None:
            self._redisplay_detection_test()
        elif self.roi_base_frame is not None and self.roi_shapes:
            # If ROI is active but no preview frame, show ROI base frame with mask
            self._display_roi_with_zoom()
        else:
            self._update_preview_display()

    def _update_body_size_info(self):
        """Update the info label showing calculated body area."""
        import math

        body_size = self.spin_reference_body_size.value()
        body_area = math.pi * (body_size / 2.0) ** 2
        self.label_body_size_info.setText(
            f"≈ {body_area:.1f} px² area (all size/distance params scale with this)"
        )

    def _update_fps_info(self):
        """Update the FPS info label with time per frame."""
        fps = self.spin_fps.value()
        time_per_frame = 1000.0 / fps  # milliseconds
        self.label_fps_info.setText(f"= {time_per_frame:.2f} ms per frame")

    def _detect_fps_from_current_video(self):
        """Detect and set FPS from the currently loaded video."""
        if not self.current_video_path:
            QMessageBox.warning(
                self, "No Video Loaded", "Please load a video file first."
            )
            return

        detected_fps = self._auto_detect_fps(self.current_video_path)
        if detected_fps is not None:
            self.spin_fps.setValue(detected_fps)
            QMessageBox.information(
                self,
                "FPS Detected",
                f"Frame rate detected: {detected_fps:.2f} FPS\n\n"
                f"Time per frame: {1000.0/detected_fps:.2f} ms",
            )

    def _auto_detect_fps(self, video_path):
        """Auto-detect FPS from video metadata and return the value."""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    logger.info(f"Detected FPS from video: {fps:.2f}")
                    return fps
                else:
                    logger.warning("Could not detect FPS from video metadata")
                    return None
            else:
                logger.warning("Could not open video for FPS detection")
                return None
        except Exception as e:
            logger.error(f"Error detecting FPS: {e}")

    def _update_detection_stats(self, detected_dimensions, resize_factor=1.0):
        """Update detection statistics display.

        Args:
            detected_dimensions: List of (major_axis, minor_axis) tuples in pixels
            resize_factor: Factor by which frame was resized (to scale back to original)
        """
        if not detected_dimensions or len(detected_dimensions) == 0:
            self.label_detection_stats.setText(
                "No detections found.\nAdjust parameters and try again."
            )
            self.btn_auto_set_body_size.setEnabled(False)
            self.detected_sizes = None
            return

        # Scale dimensions back to original resolution (resize_factor < 1 means frame was downscaled)
        # Linear dimensions scale with resize_factor
        scale_factor = 1.0 / resize_factor
        major_axes = [dims[0] * scale_factor for dims in detected_dimensions]
        minor_axes = [dims[1] * scale_factor for dims in detected_dimensions]

        # Calculate aspect ratios
        aspect_ratios = [
            major / minor if minor > 0 else 1.0
            for major, minor in zip(major_axes, minor_axes)
        ]

        # Calculate geometric mean as representative body size (better than assuming circular)
        geometric_means = [
            math.sqrt(major * minor) for major, minor in zip(major_axes, minor_axes)
        ]

        # Calculate statistics for each dimension
        stats = {
            "major": {
                "mean": np.mean(major_axes),
                "median": np.median(major_axes),
                "std": np.std(major_axes),
                "min": np.min(major_axes),
                "max": np.max(major_axes),
            },
            "minor": {
                "mean": np.mean(minor_axes),
                "median": np.median(minor_axes),
                "std": np.std(minor_axes),
                "min": np.min(minor_axes),
                "max": np.max(minor_axes),
            },
            "aspect_ratio": {
                "mean": np.mean(aspect_ratios),
                "median": np.median(aspect_ratios),
                "std": np.std(aspect_ratios),
            },
            "geometric_mean": {
                "mean": np.mean(geometric_means),
                "median": np.median(geometric_means),
                "std": np.std(geometric_means),
            },
        }

        # Store for auto-set
        self.detected_sizes = {
            "major_axes": major_axes,
            "minor_axes": minor_axes,
            "aspect_ratios": aspect_ratios,
            "geometric_means": geometric_means,
            "stats": stats,
            "count": len(detected_dimensions),
            "resize_factor": resize_factor,
            "recommended_body_size": stats["geometric_mean"][
                "median"
            ],  # Use geometric mean median
        }

        # Update label with comprehensive statistics
        stats_text = (
            f"Analyzed {len(detected_dimensions)} detections:\n\n"
            f"Major Axis (length):\n"
            f"  • Median: {stats['major']['median']:.1f} px  (range: {stats['major']['min']:.1f} - {stats['major']['max']:.1f})\n"
            f"  • Mean: {stats['major']['mean']:.1f} ± {stats['major']['std']:.1f} px\n\n"
            f"Minor Axis (width):\n"
            f"  • Median: {stats['minor']['median']:.1f} px  (range: {stats['minor']['min']:.1f} - {stats['minor']['max']:.1f})\n"
            f"  • Mean: {stats['minor']['mean']:.1f} ± {stats['minor']['std']:.1f} px\n\n"
            f"Aspect Ratio (length/width):\n"
            f"  • Median: {stats['aspect_ratio']['median']:.2f}  Mean: {stats['aspect_ratio']['mean']:.2f} ± {stats['aspect_ratio']['std']:.2f}\n\n"
            f"Recommended Body Size: {stats['geometric_mean']['median']:.1f} px\n"
            f"  (geometric mean of dimensions)"
        )
        self.label_detection_stats.setText(stats_text)
        self.btn_auto_set_body_size.setEnabled(True)

    def _auto_set_body_size_from_detection(self):
        """Auto-set reference body size from detected geometric mean."""
        if self.detected_sizes is None:
            return

        recommended_size = self.detected_sizes["recommended_body_size"]
        stats = self.detected_sizes["stats"]
        self.spin_reference_body_size.setValue(recommended_size)

        # Show confirmation with aspect ratio info
        QMessageBox.information(
            self,
            "Body Size Updated",
            f"Reference body size set to {recommended_size:.1f} px\n"
            f"(geometric mean of {self.detected_sizes['count']} detections)\n\n"
            f"Detected dimensions:\n"
            f"  • Major axis: {stats['major']['median']:.1f} px\n"
            f"  • Minor axis: {stats['minor']['median']:.1f} px\n"
            f"  • Aspect ratio: {stats['aspect_ratio']['median']:.2f}\n\n"
            f"All distance/size parameters will now scale relative to this value.",
        )

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        self.btn_video_out.setEnabled(checked)
        self.video_out_line.setEnabled(checked)

    def _update_preview_display(self):
        """Update the video display with current brightness/contrast/gamma settings."""
        if self.preview_frame_original is None:
            return

        # If we have a detection test result, redisplay it with the new zoom
        if self.detection_test_result is not None:
            self._redisplay_detection_test()
            return

        # Get current adjustment values
        brightness = self.slider_brightness.value()
        contrast = self.slider_contrast.value() / 100.0
        gamma = self.slider_gamma.value() / 100.0

        # Get detection method
        detection_method = self.combo_detection_method.currentText()
        is_background_subtraction = detection_method == "Background Subtraction"

        # Apply adjustments
        from ..utils.image_processing import apply_image_adjustments

        if is_background_subtraction:
            # Background subtraction uses grayscale with adjustments
            gray = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2GRAY)
            adjusted = apply_image_adjustments(gray, brightness, contrast, gamma)
            # Convert back to RGB for display
            adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB)
        else:
            # YOLO uses color frames directly without brightness/contrast/gamma adjustments
            # Just show the original color frame
            adjusted_rgb = self.preview_frame_original.copy()

        # Display the adjusted frame
        h, w, ch = adjusted_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(adjusted_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Apply ROI mask if exists
        if self.roi_mask is not None:
            qimg = self._apply_roi_mask_to_image(qimg)

        # Apply zoom
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _redisplay_detection_test(self):
        """Redisplay the stored detection test result with current zoom."""
        if self.detection_test_result is None:
            return

        test_frame_rgb, resize_f = self.detection_test_result
        h, w, ch = test_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Apply zoom
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f

        if effective_scale != 1.0:
            orig_h, orig_w = self.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _test_detection_on_preview(self):
        """Test detection algorithm on the current preview frame."""
        if self.preview_frame_original is None:
            logger.warning("No preview frame loaded")
            return

        # If size filtering is enabled, ask user whether to use it for the test
        use_size_filtering = False
        if self.chk_size_filtering.isChecked():
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Size Filtering Options")
            msg.setText("Size filtering is currently enabled!")
            msg.setInformativeText(
                "For accurate size estimation, it's recommended to run detection\n"
                "WITHOUT size constraints. However, you can test with constraints\n"
                "if you want to see how filtering affects the results.\n\n"
                "How would you like to proceed?"
            )

            btn_without = msg.addButton(
                "NO Size Filtering (Recommended)", QMessageBox.AcceptRole
            )
            btn_with = msg.addButton("WITH Size Filtering", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_without)

            msg.exec_()
            clicked = msg.clickedButton()

            if clicked == btn_cancel:
                return
            elif clicked == btn_with:
                use_size_filtering = True
                logger.info("Running detection test WITH size filtering enabled")
            else:  # btn_without
                use_size_filtering = False
                logger.info(
                    "Running detection test WITHOUT size filtering (recommended for size estimation)"
                )

        from ..utils.image_processing import apply_image_adjustments
        from ..core.background_models import BackgroundModel
        from ..core.detection import YOLOOBBDetector

        # Convert RGB preview to BGR for OpenCV
        frame_bgr = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2BGR)

        # Get current parameters
        detection_method = self.combo_detection_method.currentIndex()
        is_background_subtraction = detection_method == 0

        # Create a copy for visualization
        test_frame = frame_bgr.copy()

        try:
            if is_background_subtraction:
                # Build actual background model using priming frames
                logger.info("Building background model for test detection...")

                # Open video to sample priming frames
                video_path = self.file_line.text()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error("Cannot open video for background priming")
                    return

                # Build parameters dict for BackgroundModel
                bg_params = {
                    "BACKGROUND_PRIME_FRAMES": self.spin_bg_prime.value(),
                    "BRIGHTNESS": self.slider_brightness.value(),
                    "CONTRAST": self.slider_contrast.value() / 100.0,
                    "GAMMA": self.slider_gamma.value() / 100.0,
                    "ROI_MASK": self.roi_mask,
                    "RESIZE_FACTOR": self.spin_resize.value(),
                    "DARK_ON_LIGHT_BACKGROUND": self.chk_dark_on_light.isChecked(),
                    "THRESHOLD_VALUE": self.spin_threshold.value(),
                    "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
                    "ENABLE_ADDITIONAL_DILATION": self.chk_additional_dilation.isChecked(),
                    "DILATION_KERNEL_SIZE": self.spin_dilation_kernel_size.value(),
                    "DILATION_ITERATIONS": self.spin_dilation_iterations.value(),
                }

                # Create and prime background model
                bg_model = BackgroundModel(bg_params)
                bg_model.prime_background(cap)

                if bg_model.lightest_background is None:
                    logger.error("Failed to build background model")
                    cap.release()
                    return

                # Now process the preview frame with the primed background
                # Need to resize frame to match background dimensions if resize factor is set
                resize_f = bg_params["RESIZE_FACTOR"]
                frame_to_process = frame_bgr.copy()
                if resize_f < 1.0:
                    frame_to_process = cv2.resize(
                        frame_to_process,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )
                    # Also resize the test_frame for visualization
                    test_frame = cv2.resize(
                        test_frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
                gray = apply_image_adjustments(
                    gray,
                    bg_params["BRIGHTNESS"],
                    bg_params["CONTRAST"],
                    bg_params["GAMMA"],
                )

                # Apply ROI mask if exists (resize it too if needed)
                roi_for_test = self.roi_mask
                if self.roi_mask is not None:
                    if resize_f < 1.0:
                        roi_for_test = cv2.resize(
                            self.roi_mask,
                            (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    gray = cv2.bitwise_and(gray, gray, mask=roi_for_test)

                # Get background (use lightest_background as starting point)
                bg_u8 = bg_model.lightest_background.astype(np.uint8)

                # Generate foreground mask (includes morphology operations)
                fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)

                # Find contours
                cnts, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                min_contour = self.spin_min_contour.value()
                detections = []
                detected_dimensions = (
                    []
                )  # Collect (major, minor) axis pairs for statistics

                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < min_contour or len(c) < 5:
                        continue

                    # Apply size filtering based on user choice
                    if use_size_filtering:
                        min_size = self.spin_min_object_size.value()
                        max_size = self.spin_max_object_size.value()
                        if not (min_size <= area <= max_size):
                            continue

                    # Fit ellipse
                    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
                    detections.append(((cx, cy), (ax1, ax2), ang, area))
                    # Store major and minor axes (fitEllipse returns full axes, not semi-axes)
                    major_axis = max(ax1, ax2)
                    minor_axis = min(ax1, ax2)
                    detected_dimensions.append((major_axis, minor_axis))

                    # Draw ellipse
                    cv2.ellipse(
                        test_frame,
                        ((int(cx), int(cy)), (int(ax1), int(ax2)), ang),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(test_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                # Show foreground mask in corner
                small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
                test_frame[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
                    small_fg, cv2.COLOR_GRAY2BGR
                )

                # Show estimated background in opposite corner
                small_bg = cv2.resize(bg_u8, (0, 0), fx=0.3, fy=0.3)
                bg_bgr = cv2.cvtColor(small_bg, cv2.COLOR_GRAY2BGR)
                test_frame[0 : bg_bgr.shape[0], -bg_bgr.shape[1] :] = bg_bgr

                # Add detection count and note
                cv2.putText(
                    test_frame,
                    f"Detections: {len(detections)} (BG from {self.spin_bg_prime.value()} frames)",
                    (10, test_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cap.release()
                logger.info(
                    f"Background subtraction test complete: {len(detections)} detections"
                )

                # Update detection statistics (scale dimensions back to original resolution)
                self._update_detection_stats(detected_dimensions, resize_f)
            else:
                # YOLO Detection
                # Apply resize factor (same as tracking does)
                resize_f = self.spin_resize.value()
                frame_to_process = frame_bgr.copy()
                if resize_f < 1.0:
                    frame_to_process = cv2.resize(
                        frame_to_process,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )
                    # Also resize the test_frame for visualization
                    test_frame = cv2.resize(
                        test_frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                # Build parameters for YOLO
                yolo_params = {
                    "YOLO_MODEL_PATH": (
                        self.yolo_custom_model_line.text()
                        if self.combo_yolo_model.currentText() == "Custom Model..."
                        else self.combo_yolo_model.currentText().split(" ")[0]
                    ),
                    "YOLO_CONFIDENCE_THRESHOLD": self.spin_yolo_confidence.value(),
                    "YOLO_IOU_THRESHOLD": self.spin_yolo_iou.value(),
                    "YOLO_TARGET_CLASSES": (
                        [
                            int(x.strip())
                            for x in self.line_yolo_classes.text().split(",")
                        ]
                        if self.line_yolo_classes.text().strip()
                        else None
                    ),
                    "YOLO_DEVICE": self.combo_yolo_device.currentText().split(" ")[0],
                    "MAX_TARGETS": self.spin_max_targets.value(),
                    "MAX_CONTOUR_MULTIPLIER": self.spin_max_contour_multiplier.value(),
                    "ENABLE_SIZE_FILTERING": use_size_filtering,  # Use the user's choice
                    "MIN_OBJECT_SIZE": self.spin_min_object_size.value(),
                    "MAX_OBJECT_SIZE": self.spin_max_object_size.value(),
                }

                # Prepare ROI mask for filtering (resize if needed)
                roi_for_yolo = None
                if self.roi_mask is not None:
                    roi_for_yolo = self.roi_mask
                    if resize_f < 1.0:
                        roi_for_yolo = cv2.resize(
                            self.roi_mask,
                            (frame_to_process.shape[1], frame_to_process.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                # Create detector and run detection on FULL frame (no masking)
                # This preserves natural image context for better YOLO confidence
                detector = YOLOOBBDetector(yolo_params)
                meas, sizes, shapes, yolo_results, detection_confidences = (
                    detector.detect_objects(frame_to_process, 0)
                )
                
                # Filter detections by ROI AFTER detection (vectorized)
                if roi_for_yolo is not None and len(meas) > 0:
                    # Vectorized filtering using NumPy for efficiency with large n
                    meas_arr = np.array(meas)
                    cx_arr = meas_arr[:, 0].astype(np.int32)
                    cy_arr = meas_arr[:, 1].astype(np.int32)
                    
                    # Bounds check
                    h, w = roi_for_yolo.shape[:2]
                    in_bounds = (cy_arr >= 0) & (cy_arr < h) & (cx_arr >= 0) & (cx_arr < w)
                    
                    # ROI check (clip to bounds for safe indexing, then apply bounds mask)
                    cy_safe = np.clip(cy_arr, 0, h - 1)
                    cx_safe = np.clip(cx_arr, 0, w - 1)
                    in_roi = roi_for_yolo[cy_safe, cx_safe] > 0
                    
                    # Combined mask
                    keep_mask = in_bounds & in_roi
                    filtered_indices = np.where(keep_mask)[0].tolist()
                    
                    # Apply filter using boolean indexing
                    # Keep meas as list of numpy arrays for consistency
                    meas = [meas_arr[i] for i in filtered_indices]
                    sizes = np.array(sizes)[keep_mask].tolist()
                    shapes = np.array(shapes)[keep_mask].tolist()
                    detection_confidences = np.array(detection_confidences)[keep_mask].tolist()
                else:
                    filtered_indices = list(range(len(meas)))  # Keep all

                # Collect detected dimensions for statistics (only for filtered detections)
                # Extract actual width/height from YOLO OBB results
                detected_dimensions = []
                if (
                    yolo_results
                    and hasattr(yolo_results, "obb")
                    and len(yolo_results.obb) > 0
                ):
                    obb_data = yolo_results.obb
                    for i in filtered_indices:
                        # xywhr gives [center_x, center_y, width, height, rotation]
                        xywhr = obb_data.xywhr[i].cpu().numpy()
                        _, _, w, h, _ = xywhr
                        major_axis = max(w, h)
                        minor_axis = min(w, h)
                        detected_dimensions.append((major_axis, minor_axis))

                # Visualize YOLO detections (only filtered ones)
                if yolo_results and hasattr(yolo_results, "obb"):
                    obb_data = yolo_results.obb
                    for i in filtered_indices:
                        corners = obb_data.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                        cv2.polylines(
                            test_frame,
                            [corners],
                            isClosed=True,
                            color=(0, 255, 255),
                            thickness=2,
                        )

                        if hasattr(obb_data, "conf"):
                            conf = obb_data.conf[i].cpu().item()
                            cx = int(corners[:, 0].mean())
                            cy = int(corners[:, 1].mean())
                            cv2.putText(
                                test_frame,
                                f"{conf:.2f}",
                                (cx - 15, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                2,
                            )

                # Draw detection centers and orientations
                for i, m in enumerate(meas):
                    cx, cy, angle_rad = m
                    cv2.circle(test_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    # Draw orientation
                    import math

                    ex = int(cx + 30 * math.cos(angle_rad))
                    ey = int(cy + 30 * math.sin(angle_rad))
                    cv2.line(test_frame, (int(cx), int(cy)), (ex, ey), (0, 255, 0), 2)

                # Add detection count
                cv2.putText(
                    test_frame,
                    f"Detections: {len(meas)}",
                    (10, test_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Update detection statistics (scale dimensions back to original resolution)
                self._update_detection_stats(detected_dimensions, resize_f)

            # Convert BGR to RGB for Qt display
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)

            # Store the detection test result for redisplay when zoom changes
            self.detection_test_result = (test_frame_rgb.copy(), resize_f)

            h, w, ch = test_frame_rgb.shape
            bytes_per_line = ch * w

            qimg = QImage(
                test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

            # Apply zoom (zoom is applied after any resize_factor processing)
            # The zoom should be relative to the original frame size, not the resized one
            zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
            effective_scale = zoom_val * resize_f

            if effective_scale != 1.0:
                # Get original dimensions
                orig_h, orig_w = self.preview_frame_original.shape[:2]
                scaled_w = int(orig_w * effective_scale)
                scaled_h = int(orig_h * effective_scale)
                qimg = qimg.scaled(
                    scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

            # Auto-fit to screen after detection
            self._fit_image_to_screen()

            logger.info("Detection test completed on preview frame")

        except Exception as e:
            logger.error(f"Detection test failed: {e}")
            import traceback

            traceback.print_exc()

    def _on_roi_mode_changed(self, index):
        """Handle ROI mode selection change."""
        self.roi_current_mode = "circle" if index == 0 else "polygon"
        if self.roi_selection_active:
            # If actively selecting, update instructions
            if self.roi_current_mode == "circle":
                self.roi_instructions.setText(
                    "Circle: Left-click 3+ points on boundary  •  Right-click to undo  •  ESC to cancel"
                )
            else:
                self.roi_instructions.setText(
                    "Polygon: Left-click vertices  •  Right-click to undo  •  Double-click to finish  •  ESC to cancel"
                )

    def _on_roi_zone_changed(self, index):
        """Handle ROI zone type selection change."""
        self.roi_current_zone_type = "include" if index == 0 else "exclude"

    def _handle_video_mouse_press(self, evt):
        """Handle mouse press on video - either ROI selection or pan/zoom."""
        # ROI selection takes priority
        if self.roi_selection_active:
            self.record_roi_click(evt)
            return

        # Pan mode: Left button or Middle button
        from PySide2.QtCore import Qt

        if evt.button() == Qt.LeftButton or evt.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = evt.globalPos()
            self._scroll_start_h = self.scroll.horizontalScrollBar().value()
            self._scroll_start_v = self.scroll.verticalScrollBar().value()
            self.video_label.setCursor(Qt.ClosedHandCursor)
            evt.accept()

    def _handle_video_mouse_move(self, evt):
        """Handle mouse move - update pan if active."""
        if self._is_panning and self._pan_start_pos:
            from PySide2.QtCore import Qt

            delta = evt.globalPos() - self._pan_start_pos
            self.scroll.horizontalScrollBar().setValue(self._scroll_start_h - delta.x())
            self.scroll.verticalScrollBar().setValue(self._scroll_start_v - delta.y())
            evt.accept()
        elif not self.roi_selection_active:
            # Show open hand cursor to indicate draggable
            from PySide2.QtCore import Qt

            self.video_label.setCursor(Qt.OpenHandCursor)

    def _handle_video_mouse_release(self, evt):
        """Handle mouse release - end pan."""
        from PySide2.QtCore import Qt

        if self._is_panning:
            self._is_panning = False
            self._pan_start_pos = None
            # Return to open hand (still draggable) if not in ROI mode
            if not self.roi_selection_active:
                self.video_label.setCursor(Qt.OpenHandCursor)
            else:
                self.video_label.setCursor(Qt.ArrowCursor)
            evt.accept()

    def _handle_video_double_click(self, evt):
        """Handle double-click on video to fit to screen."""
        if evt.button() == Qt.LeftButton:
            self._fit_image_to_screen()

    def _handle_video_wheel(self, evt):
        """Handle mouse wheel - zoom in/out."""
        from PySide2.QtCore import Qt
        
        # Block zoom during ROI selection
        if self.roi_selection_active:
            evt.ignore()
            return

        # Ctrl+Wheel for zoom
        if evt.modifiers() == Qt.ControlModifier:
            delta = evt.angleDelta().y()

            # Get current zoom
            current_zoom = self.slider_zoom.value()

            # Calculate new zoom (10% per wheel step)
            zoom_change = 10 if delta > 0 else -10
            new_zoom = max(10, min(400, current_zoom + zoom_change))

            # Update zoom slider (will trigger zoom change)
            self.slider_zoom.setValue(new_zoom)
            evt.accept()
        else:
            # Normal scroll - pass to scroll area
            evt.ignore()

    def _handle_video_event(self, evt):
        """Handle video events including pinch gestures."""
        from PySide2.QtCore import QEvent, Qt

        if evt.type() == QEvent.Gesture:
            return self._handle_gesture_event(evt)

        # Pass other events to default handler
        return QLabel.event(self.video_label, evt)

    def _handle_gesture_event(self, evt):
        """Handle pinch-to-zoom gesture."""
        from PySide2.QtCore import Qt
        
        # Block gestures during ROI selection
        if self.roi_selection_active:
            return False

        gesture = evt.gesture(Qt.PinchGesture)
        if gesture:
            from PySide2.QtWidgets import QGesture

            if gesture.state() == Qt.GestureUpdated:
                # Get scale factor
                scale_factor = gesture.scaleFactor()

                # Get current zoom
                current_zoom = self.slider_zoom.value()

                # Calculate new zoom based on pinch scale
                # Scale factor > 1 = zoom in, < 1 = zoom out
                zoom_delta = int((scale_factor - 1.0) * 50)  # Sensitivity adjustment
                new_zoom = max(10, min(400, current_zoom + zoom_delta))

                # Update zoom slider
                self.slider_zoom.setValue(new_zoom)

            return True

        return False

    def _display_roi_with_zoom(self):
        """Display the ROI base frame with mask and current zoom applied."""
        if self.roi_base_frame is None or not self.roi_shapes:
            return
        
        # Apply ROI mask to base frame
        qimg_masked = self._apply_roi_mask_to_image(self.roi_base_frame)
        
        # Apply zoom
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            w = qimg_masked.width()
            h = qimg_masked.height()
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg_masked = qimg_masked.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        
        pixmap = QPixmap.fromImage(qimg_masked)
        self.video_label.setPixmap(pixmap)

    def _fit_image_to_screen(self):
        """Fit the image to the available screen space."""
        # Determine which frame to use for sizing and whether resize factor applies
        # Resize factor applies differently depending on display mode:
        # - ROI selection / preview display: full resolution, resize factor NOT yet applied
        # - Detection test / tracking preview: resize factor already applied to displayed frame
        
        # Check if tracking worker is running (frames are already resized)
        tracking_active = self.tracking_worker is not None and self.tracking_worker.isRunning()
        
        if tracking_active and self._tracking_frame_size is not None:
            # During tracking/preview, use the actual frame size from the worker
            # These frames are already resized, so use dimensions directly
            effective_width, effective_height = self._tracking_frame_size
        elif self.detection_test_result is not None:
            # Detection test shows resized content
            if self.preview_frame_original is not None:
                h, w = self.preview_frame_original.shape[:2]
                resize_factor = self.spin_resize.value()
                effective_width = int(w * resize_factor)
                effective_height = int(h * resize_factor)
            else:
                return
        elif self.preview_frame_original is not None:
            # Preview frame at original resolution (pre-test detection)
            h, w = self.preview_frame_original.shape[:2]
            effective_width = w
            effective_height = h
        elif self.roi_base_frame is not None:
            # ROI base frame is always full resolution
            effective_width = self.roi_base_frame.width()
            effective_height = self.roi_base_frame.height()
        else:
            return

        # Get the scroll area viewport size
        viewport_width = self.scroll.viewport().width()
        viewport_height = self.scroll.viewport().height()

        # Calculate zoom to fit the effective dimensions
        zoom_w = viewport_width / effective_width
        zoom_h = viewport_height / effective_height
        zoom_fit = min(zoom_w, zoom_h) * 0.95  # 95% to leave some margin

        # Clamp to valid range
        zoom_fit = max(0.1, min(5.0, zoom_fit))

        # Set the zoom slider
        self.slider_zoom.setValue(int(zoom_fit * 100))

        # Reset scroll position to center
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def record_roi_click(self, evt):
        if not self.roi_selection_active or self.roi_base_frame is None:
            return
        
        # Right-click to undo last point
        if evt.button() == Qt.RightButton:
            if len(self.roi_points) > 0:
                removed = self.roi_points.pop()
                logger.info(f"Undid last ROI point: ({removed[0]}, {removed[1]})")
                self.update_roi_preview()
            return
        
        # Left-click to add point
        if evt.button() != Qt.LeftButton:
            return
            
        x, y = evt.pos().x(), evt.pos().y()

        # Double-click detection for polygon closing
        if self.roi_current_mode == "polygon" and len(self.roi_points) >= 3:
            if hasattr(self, "_last_click_pos") and hasattr(self, "_last_click_time"):
                import time

                current_time = time.time()
                last_x, last_y = self._last_click_pos
                # Check if double-click (within 0.5s and 10 pixels)
                if (
                    current_time - self._last_click_time < 0.5
                    and abs(x - last_x) < 10
                    and abs(y - last_y) < 10
                ):
                    self.finish_roi_selection()
                    return
            import time

            self._last_click_pos = (x, y)
            self._last_click_time = time.time()

        self.roi_points.append((x, y))
        self.update_roi_preview()

    def update_roi_preview(self):
        if self.roi_base_frame is None:
            return
        pix = QPixmap.fromImage(self.roi_base_frame).toImage().copy()
        painter = QPainter(pix)

        # Draw existing shapes first (color-coded by mode)
        for shape in self.roi_shapes:
            # Choose color based on inclusion/exclusion
            is_include = shape.get("mode", "include") == "include"
            color = Qt.cyan if is_include else Qt.red

            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
            elif shape["type"] == "polygon":
                from PySide2.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(color, 2))
                painter.drawPolygon(points)

        # Draw current selection
        painter.setPen(QPen(Qt.red, 6))
        for i, (px, py) in enumerate(self.roi_points):
            painter.drawPoint(px, py)
            painter.setPen(QPen(Qt.black, 3))
            painter.drawText(px + 12, py - 12, str(i + 1))
            painter.setPen(QPen(Qt.white, 2))
            painter.drawText(px + 10, py - 10, str(i + 1))
            painter.setPen(QPen(Qt.red, 6))

        can_finish = False
        # Color for current shape preview (lighter version of final color)
        preview_color = (
            Qt.green if self.roi_current_zone_type == "include" else QColor(255, 165, 0)
        )  # Orange for exclude

        if self.roi_current_mode == "circle" and len(self.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self.roi_points)
            if circle_fit:
                cx, cy, radius = circle_fit
                self.roi_fitted_circle = circle_fit
                painter.setPen(QPen(preview_color, 3))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
                painter.setPen(QPen(Qt.blue, 8))
                painter.drawPoint(int(cx), int(cy))
                zone_type = (
                    "Include" if self.roi_current_zone_type == "include" else "Exclude"
                )
                self.roi_status_label.setText(
                    f"Preview {zone_type} Circle: R={radius:.1f}px"
                )
                can_finish = True
            else:
                self.roi_status_label.setText("Invalid circle fit")
        elif self.roi_current_mode == "polygon" and len(self.roi_points) >= 3:
            # Draw preview polygon
            from PySide2.QtCore import QPoint

            points = [QPoint(int(x), int(y)) for x, y in self.roi_points]
            painter.setPen(QPen(preview_color, 3))
            painter.drawPolygon(points)
            zone_type = (
                "Include" if self.roi_current_zone_type == "include" else "Exclude"
            )
            self.roi_status_label.setText(
                f"Preview {zone_type} Polygon: {len(self.roi_points)} vertices"
            )
            can_finish = True
        else:
            min_pts = 3
            self.roi_status_label.setText(
                f"Points: {len(self.roi_points)} (Need {min_pts}+)"
            )

        self.btn_finish_roi.setEnabled(can_finish)
        painter.end()
        self.video_label.setPixmap(QPixmap.fromImage(pix))

    def start_roi_selection(self):
        if not self.file_line.text():
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        # Load base frame if not already loaded
        if self.roi_base_frame is None:
            cap = cv2.VideoCapture(self.file_line.text())
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open video file.")
                return
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self, "Error", "Cannot read video frame.")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.roi_base_frame = qt_image

        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = True
        self.btn_start_roi.setEnabled(False)
        self.btn_finish_roi.setEnabled(False)
        self.combo_roi_mode.setEnabled(False)
        self.combo_roi_zone.setEnabled(False)
        
        # Disable zoom slider during ROI selection
        self.slider_zoom.setEnabled(False)
        
        # Set crosshair cursor for precise ROI selection
        self.video_label.setCursor(Qt.CrossCursor)

        zone_type = (
            "INCLUSION" if self.roi_current_zone_type == "include" else "EXCLUSION"
        )
        if self.roi_current_mode == "circle":
            self.roi_status_label.setText(
                f"Click points on {zone_type.lower()} circle boundary"
            )
            self.roi_instructions.setText(
                f"{zone_type} Circle: Left-click 3+ points on boundary  •  Right-click to undo  •  ESC to cancel"
            )
        else:
            self.roi_status_label.setText(f"Click {zone_type.lower()} polygon vertices")
            self.roi_instructions.setText(
                f"{zone_type} Polygon: Left-click vertices  •  Right-click to undo  •  Double-click to finish  •  ESC to cancel"
            )

        self.update_roi_preview()

    def finish_roi_selection(self):
        if not self.roi_base_frame:
            return

        fh, fw = self.roi_base_frame.height(), self.roi_base_frame.width()

        if self.roi_current_mode == "circle":
            if not self.roi_fitted_circle:
                QMessageBox.warning(self, "No ROI", "No valid circle fit available.")
                return
            cx, cy, radius = self.roi_fitted_circle
            self.roi_shapes.append(
                {
                    "type": "circle",
                    "params": (cx, cy, radius),
                    "mode": self.roi_current_zone_type,
                }
            )
            zone_type = (
                "inclusion" if self.roi_current_zone_type == "include" else "exclusion"
            )
            logger.info(
                f"Added circle {zone_type} zone: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
            )

        elif self.roi_current_mode == "polygon":
            if len(self.roi_points) < 3:
                QMessageBox.warning(
                    self, "No ROI", "Need at least 3 points for polygon."
                )
                return
            self.roi_shapes.append(
                {
                    "type": "polygon",
                    "params": list(self.roi_points),
                    "mode": self.roi_current_zone_type,
                }
            )
            zone_type = (
                "inclusion" if self.roi_current_zone_type == "include" else "exclusion"
            )
            logger.info(
                f"Added polygon {zone_type} zone with {len(self.roi_points)} vertices"
            )

        # Generate combined mask from all shapes
        self._generate_combined_roi_mask(fh, fw)

        # Reset for next shape
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = False
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        self.combo_roi_mode.setEnabled(True)
        self.combo_roi_zone.setEnabled(True)
        self.roi_instructions.setText("")
        
        # Re-enable zoom slider
        self.slider_zoom.setEnabled(True)
        
        # Restore open hand cursor (for pan/zoom)
        if hasattr(Qt, 'OpenHandCursor'):
            self.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self.video_label.unsetCursor()

        # Update status to show inclusion/exclusion counts
        include_count = sum(
            1 for s in self.roi_shapes if s.get("mode", "include") == "include"
        )
        exclude_count = sum(
            1 for s in self.roi_shapes if s.get("mode", "include") == "exclude"
        )
        self.roi_status_label.setText(
            f"Active ROI: {include_count} inclusion, {exclude_count} exclusion zone(s)"
        )

        # Enable crop button and show optimization info
        self.btn_crop_video.setEnabled(True)
        self._update_roi_optimization_info()

        # Auto-fit to screen after ROI change - use QTimer to ensure proper sequencing
        if self.roi_base_frame:
            from PySide2.QtCore import QTimer
            # First fit the screen (sets zoom slider value)
            QTimer.singleShot(10, self._fit_image_to_screen)
            # Then display with the new zoom applied
            QTimer.singleShot(50, self._display_roi_with_zoom)

    def _generate_combined_roi_mask(self, height, width):
        """Generate a combined mask from all ROI shapes with inclusion/exclusion support."""
        if not self.roi_shapes:
            self.roi_mask = None
            return

        # Create blank mask
        combined_mask = np.zeros((height, width), np.uint8)

        # First pass: apply all inclusion zones (OR operation)
        for shape in self.roi_shapes:
            if shape.get("mode", "include") == "include":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 255, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 255)

        # Second pass: subtract all exclusion zones (AND NOT operation)
        for shape in self.roi_shapes:
            if shape.get("mode", "include") == "exclude":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 0, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 0)

        self.roi_mask = combined_mask
        logger.info(f"Generated combined ROI mask from {len(self.roi_shapes)} shape(s)")

        # Invalidate cache when ROI changes
        self._invalidate_roi_cache()

    def undo_last_roi_shape(self):
        """Remove the last added ROI shape."""
        if not self.roi_shapes:
            return

        removed = self.roi_shapes.pop()
        logger.info(f"Removed last ROI shape: {removed['type']}")

        if self.roi_base_frame:
            fh, fw = self.roi_base_frame.height(), self.roi_base_frame.width()
            self._generate_combined_roi_mask(fh, fw)
        else:
            self.roi_mask = None

        # Update UI
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        if self.roi_shapes:
            num_shapes = len(self.roi_shapes)
            shape_summary = ", ".join([s["type"] for s in self.roi_shapes])
            self.roi_status_label.setText(
                f"Active ROI: {num_shapes} shape(s) ({shape_summary})"
            )
            # Show the updated masked result
            if self.roi_base_frame:
                qimg_masked = self._apply_roi_mask_to_image(self.roi_base_frame)
                self.video_label.setPixmap(QPixmap.fromImage(qimg_masked))
        else:
            self.roi_status_label.setText("No ROI")
            # Show original frame without masking
            if self.roi_base_frame:
                self.video_label.setPixmap(QPixmap.fromImage(self.roi_base_frame))

        self.update_roi_preview()

    def clear_roi(self):
        self.roi_mask = None
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_shapes = []
        self.roi_selection_active = False
        self.roi_base_frame = None
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        self.btn_undo_roi.setEnabled(False)
        self.combo_roi_mode.setEnabled(True)
        self.roi_status_label.setText("No ROI")
        self.roi_instructions.setText("")
        self.video_label.setText("ROI Cleared.")
        
        # Re-enable zoom slider
        self.slider_zoom.setEnabled(True)
        
        # Restore open hand cursor
        if hasattr(Qt, 'OpenHandCursor'):
            self.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self.video_label.unsetCursor()
        
        logger.info("All ROI shapes cleared")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.roi_selection_active:
            self.clear_roi()
        else:
            super().keyPressEvent(event)

    def on_detection_method_changed(self, index):
        is_yolo = index == 1
        # In new UI, this is handled by StackedWidget, but we keep this for compatibility logic
        pass

    def on_yolo_model_changed(self, index):
        is_custom = self.combo_yolo_model.currentText() == "Custom Model..."
        self.yolo_custom_model_widget.setVisible(is_custom)

    def select_yolo_custom_model(self):
        start_dir = (
            os.path.dirname(self.yolo_custom_model_line.text())
            if self.yolo_custom_model_line.text()
            else os.path.expanduser("~")
        )
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            start_dir,
            "PyTorch Model Files (*.pt *.pth);;All Files (*)",
        )
        if fp:
            self.yolo_custom_model_line.setText(fp)

    def toggle_histogram_window(self):
        if self.histogram_window is None:
            if self.histogram_panel is None:
                self.histogram_panel = HistogramPanel(
                    history_frames=self.spin_histogram_history.value()
                )
            self.histogram_window = QMainWindow()
            self.histogram_window.setWindowTitle("Real-Time Parameter Histograms")
            self.histogram_window.setCentralWidget(self.histogram_panel)
            self.histogram_window.resize(900, 700)
            self.histogram_window.setStyleSheet(self.styleSheet())

            def on_close():
                self.btn_show_histograms.setChecked(False)
                self.histogram_window.hide()

            self.histogram_window.closeEvent = lambda event: (
                on_close(),
                event.accept(),
            )

        if self.btn_show_histograms.isChecked():
            self.histogram_window.show()
            self.histogram_window.raise_()
            self.histogram_window.activateWindow()
        else:
            self.histogram_window.hide()

    def toggle_preview(self, checked):
        if checked:
            # Warn user that preview doesn't save config
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Preview Mode")
            msg.setText(
                "Preview mode will run forward tracking only without saving configuration."
            )
            msg.setInformativeText(
                "Preview features:\n"
                "• Forward pass only (no backward tracking)\n"
                "• Configuration is NOT saved\n"
                "• No CSV output\n\n"
                "Use 'Run Full Tracking' to save results and config."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)

            if msg.exec_() == QMessageBox.Ok:
                self.start_tracking(preview_mode=True)
                self.btn_preview.setText("Stop Preview")
                self.btn_start.setEnabled(False)
            else:
                self.btn_preview.setChecked(False)
        else:
            self.stop_tracking()
            self.btn_preview.setText("Preview Mode")
            self.btn_start.setEnabled(True)

    def toggle_debug_logging(self, checked):
        if checked:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging disabled")

    def _on_visualization_mode_changed(self, state):
        """Handle visualization-free mode toggle."""
        is_viz_free = self.chk_visualization_free.isChecked()

        # Disable all visualization options when in viz-free mode
        self.chk_show_circles.setEnabled(not is_viz_free)
        self.chk_show_orientation.setEnabled(not is_viz_free)
        self.chk_show_trajectories.setEnabled(not is_viz_free)
        self.chk_show_labels.setEnabled(not is_viz_free)
        self.chk_show_state.setEnabled(not is_viz_free)
        self.chk_show_kalman_uncertainty.setEnabled(not is_viz_free)
        self.chk_show_fg.setEnabled(not is_viz_free)
        self.chk_show_bg.setEnabled(not is_viz_free)
        self.chk_show_yolo_obb.setEnabled(not is_viz_free)

        # Show/hide video preview
        if is_viz_free:
            # Store current preview if any, then show placeholder
            self._stored_preview_text = (
                self.video_label.text() if not self.video_label.pixmap() else None
            )
            self.video_label.clear()
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #888; font-size: 14px;")
            logger.info("Visualization-Free Mode enabled - Maximum speed processing")
        else:
            # Restore previous state or default message
            if hasattr(self, "_stored_preview_text") and self._stored_preview_text:
                self.video_label.setText(self._stored_preview_text)
            elif not self.video_label.pixmap():
                self.video_label.setText("Load a video to begin...")
            self.video_label.setStyleSheet("color: #666; font-size: 16px;")

    def start_full(self):
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            self.stop_tracking()

        # Set up comprehensive session logging once for entire tracking session
        video_path = self.file_line.text()
        if video_path:
            self._setup_session_logging(video_path, backward_mode=False)

        self.start_tracking(preview_mode=False)

    def stop_tracking(self):
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
        self._set_ui_controls_enabled(True)
        self.btn_preview.setChecked(False)
        self.btn_preview.setText("Preview Mode")
        self.btn_start.setEnabled(True)

        # Hide stats labels when tracking stops
        self.label_current_fps.setVisible(False)
        self.label_elapsed_time.setVisible(False)
        self.label_eta.setVisible(False)
        
        # Reset tracking frame size
        self._tracking_frame_size = None

    def _set_ui_controls_enabled(self, enabled: bool):
        # Disable Main setup
        self.btn_file.setEnabled(enabled)
        self.btn_csv.setEnabled(enabled)
        self.btn_video_out.setEnabled(enabled)
        self.spin_resize.setEnabled(enabled)

        # Disable tabs except for Visualization and maybe some display settings
        self.tab_detection.setEnabled(enabled)
        self.tab_tracking.setEnabled(enabled)
        self.tab_data.setEnabled(enabled)

        # ROI
        self.btn_start_roi.setEnabled(enabled)
        self.btn_finish_roi.setEnabled(enabled and self.roi_selection_active)
        self.btn_clear_roi.setEnabled(enabled)

        # Buttons
        self.btn_preview.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled)

        # Zoom is always enabled
        self.slider_zoom.setEnabled(True)

    def _draw_roi_overlay(self, qimage):
        """Draw ROI shapes overlay on a QImage."""
        if not self.roi_shapes:
            return qimage

        # Create a copy to draw on
        pix = QPixmap.fromImage(qimage).copy()
        painter = QPainter(pix)

        # Draw all ROI shapes
        for shape in self.roi_shapes:
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
                # Draw center point
                painter.setPen(QPen(Qt.cyan, 6))
                painter.drawPoint(int(cx), int(cy))
            elif shape["type"] == "polygon":
                from PySide2.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawPolygon(points)

        painter.end()
        return pix.toImage()

    def _apply_roi_mask_to_image(self, qimage):
        """Apply ROI visualization based on detection method.
        
        For background subtraction: mask with darkening (image is actually masked)
        For YOLO: just draw boundary overlay (image is NOT masked, only detections filtered)
        """
        if self.roi_mask is None or not self.roi_shapes:
            return qimage
        
        # Check detection method to determine visualization style
        # Safe check - default to YOLO overlay if no method selected yet
        try:
            detection_method = self.combo_detection_method.currentText()
        except:
            detection_method = "YOLO OBB"
        
        if detection_method == "YOLO OBB":
            # YOLO processes full image - just show boundary overlay
            return self._draw_roi_overlay(qimage)
        else:
            # Background subtraction masks image - show masked visualization
            return self._apply_roi_mask_darkening(qimage)
    
    def _apply_roi_mask_darkening(self, qimage):
        """Apply ROI mask to darken areas outside the ROI (with caching).
        Used for background subtraction where the image is actually masked.
        """
        if self.roi_mask is None or not self.roi_shapes:
            return qimage

        # Generate cache key from image pointer and ROI hash
        frame_id = id(qimage)
        roi_hash = self._get_roi_hash()
        cache_key = (frame_id, roi_hash)

        # Return cached result if available
        if cache_key in self._roi_masked_cache:
            return self._roi_masked_cache[cache_key]

        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()

        # Ensure image is in RGB888 format
        if qimage.format() != QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)

        # Convert to numpy array using buffer protocol
        ptr = qimage.bits()
        if hasattr(ptr, "setsize"):
            # Older PySide2 versions (sip.voidptr)
            ptr.setsize(height * width * 3)
            arr = np.array(ptr).reshape(height, width, 3)
        else:
            # Newer PySide2 versions (memoryview)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 3)

        # Create a copy to modify
        arr_copy = arr.copy()

        # Resize ROI mask to match image dimensions if needed
        if self.roi_mask.shape != (height, width):
            roi_resized = cv2.resize(
                self.roi_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        else:
            roi_resized = self.roi_mask

        # Darken areas outside ROI (multiply by 0.3 for 70% darkening)
        mask_inv = roi_resized == 0
        arr_copy[mask_inv] = (arr_copy[mask_inv] * 0.3).astype(np.uint8)

        # Create new QImage from modified array
        result = QImage(arr_copy.data, width, height, width * 3, QImage.Format_RGB888)
        # Make a copy to ensure data persistence
        result_copy = result.copy()

        # Cache the result (limit cache size to prevent memory bloat)
        if len(self._roi_masked_cache) > 50:
            # Remove oldest entries
            self._roi_masked_cache.clear()
        self._roi_masked_cache[cache_key] = result_copy

        return result_copy

    @Slot(int, str)
    def on_progress_update(self, percentage, status_text):
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status_text)

    @Slot(dict)
    def on_stats_update(self, stats):
        """Update real-time tracking statistics."""
        # Update FPS
        if "fps" in stats:
            self.label_current_fps.setText(f"FPS: {stats['fps']:.1f}")
            self.label_current_fps.setVisible(True)

        # Update elapsed time
        if "elapsed" in stats:
            elapsed_sec = stats["elapsed"]
            hours = int(elapsed_sec // 3600)
            minutes = int((elapsed_sec % 3600) // 60)
            seconds = int(elapsed_sec % 60)
            if hours > 0:
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                elapsed_str = f"{minutes:02d}:{seconds:02d}"
            self.label_elapsed_time.setText(f"Elapsed: {elapsed_str}")
            self.label_elapsed_time.setVisible(True)

        # Update ETA
        if "eta" in stats:
            eta_sec = stats["eta"]
            if eta_sec > 0:
                hours = int(eta_sec // 3600)
                minutes = int((eta_sec % 3600) // 60)
                seconds = int(eta_sec % 60)
                if hours > 0:
                    eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    eta_str = f"{minutes:02d}:{seconds:02d}"
                self.label_eta.setText(f"ETA: {eta_str}")
            else:
                self.label_eta.setText("ETA: calculating...")
            self.label_eta.setVisible(True)

    @Slot(np.ndarray)
    def on_new_frame(self, rgb):
        z = max(self.slider_zoom.value() / 100.0, 0.1)
        h, w, _ = rgb.shape
        
        # Store tracking frame size for fit-to-screen calculation
        self._tracking_frame_size = (w, h)
        
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        # ROI masking is now done in tracking worker - no need to duplicate here
        scaled = qimg.scaled(
            int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))
        
        # Auto-fit to screen on first frame of tracking
        if self._tracking_first_frame:
            self._tracking_first_frame = False
            # Use QTimer to ensure frame is displayed first
            from PySide2.QtCore import QTimer
            QTimer.singleShot(50, self._fit_image_to_screen)

    def _scale_trajectories_to_original_space(self, trajectories_df, resize_factor):
        """Scale trajectory coordinates from resized space back to original video space."""
        if trajectories_df is None or trajectories_df.empty:
            return trajectories_df

        if resize_factor == 1.0:
            return trajectories_df  # No scaling needed

        # Scale factor to go from resized -> original is 1/resize_factor
        scale_factor = 1.0 / resize_factor

        logger.info(
            f"Scaling trajectories to original video space (resize_factor={resize_factor:.3f}, scale_factor={scale_factor:.3f})"
        )

        result_df = trajectories_df.copy()

        # Scale X, Y coordinates
        result_df["X"] = result_df["X"] * scale_factor
        result_df["Y"] = result_df["Y"] * scale_factor

        # Theta doesn't need scaling (it's an angle)
        # FrameID doesn't need scaling

        logger.info(
            f"Scaled {len(result_df)} trajectory points to original video coordinates"
        )
        return result_df

    def save_trajectories_to_csv(self, trajectories, output_path):
        """Save processed trajectories to CSV.

        Args:
            trajectories: Either list of tuples (old format) or pandas DataFrame (new format with confidence)
            output_path: Path to save CSV file
        """
        if trajectories is None:
            logger.warning("No post-processed trajectories to save (None).")
            return False

        # Check if input is a DataFrame (new format with confidence)
        if isinstance(trajectories, pd.DataFrame):
            if trajectories.empty:
                logger.warning(
                    "No post-processed trajectories to save (empty DataFrame)."
                )
                return False
            try:
                # DataFrame already has all columns including confidence metrics
                # Convert X and Y to integers where possible (non-NaN values)
                df_to_save = trajectories.copy()
                for col in ["X", "Y", "FrameID"]:
                    if col in df_to_save.columns:
                        # Convert to float first to handle any issues, then to Int64 (nullable integer)
                        df_to_save[col] = pd.to_numeric(
                            df_to_save[col], errors="coerce"
                        )
                        # Use Int64 dtype which supports NaN values
                        df_to_save[col] = df_to_save[col].round().astype("Int64")

                # Drop unwanted columns from raw tracking data
                unwanted_cols = ["TrackID", "Index"]
                df_to_save = df_to_save.drop(
                    columns=[col for col in unwanted_cols if col in df_to_save.columns],
                    errors="ignore",
                )

                # Reorder columns to put basic trajectory info first
                base_cols = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
                other_cols = [col for col in df_to_save.columns if col not in base_cols]
                ordered_cols = base_cols + other_cols
                df_to_save[ordered_cols].to_csv(output_path, index=False)
                logger.info(
                    f"Successfully saved {df_to_save['TrajectoryID'].nunique()} post-processed trajectories "
                    f"({len(df_to_save)} rows) with {len(ordered_cols)} columns to {output_path}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to save processed trajectories to {output_path}: {e}"
                )
                return False

        # Old format (list of tuples) - for backward compatibility
        if not trajectories:
            logger.warning("No post-processed trajectories to save.")
            return False
        header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for trajectory_id, segment in enumerate(trajectories):
                    for x, y, theta, frame_id in segment:
                        # Handle NaN values - write as empty string or keep as float
                        x_val = int(x) if not np.isnan(x) else ""
                        y_val = int(y) if not np.isnan(y) else ""
                        frame_val = int(frame_id) if not np.isnan(frame_id) else ""
                        writer.writerow([trajectory_id, x_val, y_val, theta, frame_val])
            logger.info(
                f"Successfully saved {len(trajectories)} post-processed trajectories to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    def merge_and_save_trajectories(self):
        logger.info(f"=" * 80)
        logger.info("Starting trajectory merging process...")
        logger.info(f"=" * 80)

        forward_trajs = getattr(self, "forward_processed_trajs", None)
        backward_trajs = getattr(self, "backward_processed_trajs", None)

        # Check if trajectories exist and are not empty (handle both DataFrame and list)
        forward_empty = (
            forward_trajs is None
            or (isinstance(forward_trajs, pd.DataFrame) and forward_trajs.empty)
            or (isinstance(forward_trajs, list) and len(forward_trajs) == 0)
        )
        backward_empty = (
            backward_trajs is None
            or (isinstance(backward_trajs, pd.DataFrame) and backward_trajs.empty)
            or (isinstance(backward_trajs, list) and len(backward_trajs) == 0)
        )

        if forward_empty or backward_empty:
            QMessageBox.warning(
                self,
                "No Trajectories",
                "No forward or backward trajectories available to merge.",
            )
            return

        video_fp = self.file_line.text()
        if not video_fp:
            return
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        current_params = self.get_parameters_dict()

        # Convert DataFrames to list of DataFrames (one per trajectory) for resolve_trajectories
        def prepare_trajs_for_merge(trajs):
            """Convert trajectories to format expected by resolve_trajectories."""
            if isinstance(trajs, pd.DataFrame):
                # Split DataFrame by TrajectoryID into list of DataFrames
                return [group for _, group in trajs.groupby("TrajectoryID")]
            else:
                # Already in list format
                return trajs

        forward_prepared = prepare_trajs_for_merge(forward_trajs)
        backward_prepared = prepare_trajs_for_merge(backward_trajs)

        resolved_trajectories = resolve_trajectories(
            forward_prepared,
            backward_prepared,
            video_length=total_frames,
            params=current_params,
        )

        # Convert resolved trajectories to DataFrame for interpolation
        # resolve_trajectories now returns list of DataFrames, concatenate them
        from ..core.post_processing import interpolate_trajectories

        # Convert list of DataFrames to single DataFrame
        if resolved_trajectories and isinstance(resolved_trajectories, list):
            if isinstance(resolved_trajectories[0], pd.DataFrame):
                # Concatenate all trajectory DataFrames
                # Reassign TrajectoryID to ensure unique IDs
                for new_id, traj_df in enumerate(resolved_trajectories):
                    traj_df["TrajectoryID"] = new_id
                resolved_trajectories = pd.concat(
                    resolved_trajectories, ignore_index=True
                )
            else:
                # Fallback for old tuple format (shouldn't happen)
                logger.warning(
                    "Received tuple format from resolve_trajectories, converting..."
                )
                all_data = []
                for traj_id, traj in enumerate(resolved_trajectories):
                    for x, y, theta, frame in traj:
                        all_data.append(
                            {
                                "TrajectoryID": traj_id,
                                "X": x,
                                "Y": y,
                                "Theta": theta,
                                "FrameID": frame,
                            }
                        )
                if all_data:
                    resolved_trajectories = pd.DataFrame(all_data)
                else:
                    resolved_trajectories = []

        # Apply interpolation if enabled
        if isinstance(resolved_trajectories, pd.DataFrame):
            interp_method = self.combo_interpolation_method.currentText().lower()
            if interp_method != "none":
                max_gap = self.spin_interpolation_max_gap.value()
                resolved_trajectories = interpolate_trajectories(
                    resolved_trajectories, method=interp_method, max_gap=max_gap
                )

        # Scale coordinates back to original video space
        resize_factor = self.spin_resize.value()
        if isinstance(resolved_trajectories, pd.DataFrame):
            resolved_trajectories = self._scale_trajectories_to_original_space(
                resolved_trajectories, resize_factor
            )

        raw_csv_path = self.csv_line.text()
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            merged_csv_path = f"{base}_merged.csv"
            if self.save_trajectories_to_csv(resolved_trajectories, merged_csv_path):
                # Track initial tracking CSV as temporary (will be replaced by merged version)
                if raw_csv_path not in self.temporary_files:
                    self.temporary_files.append(raw_csv_path)
                logger.info(f"✓ Merged trajectory data saved to: {merged_csv_path}")

    @Slot(bool, list, list)
    def on_tracking_finished(self, finished_normally, fps_list, full_traj):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if self.csv_writer_thread:
            self.csv_writer_thread.stop()
            self.csv_writer_thread.join()

        # Check if this was preview mode
        was_preview_mode = self.btn_preview.isChecked()

        if was_preview_mode:
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            # Hide stats labels and re-enable UI for preview mode
            self.label_current_fps.setVisible(False)
            self.label_elapsed_time.setVisible(False)
            self.label_eta.setVisible(False)
            self._set_ui_controls_enabled(True)
            if finished_normally:
                logger.info("Preview completed.")
            else:
                QMessageBox.warning(
                    self,
                    "Preview Interrupted",
                    "Preview was stopped or encountered an error.",
                )
            gc.collect()
            return  # Exit early - no post-processing, no backward tracking for preview

        if finished_normally:
            logger.info("Tracking completed successfully.")
            is_backward_mode = (
                hasattr(self.tracking_worker, "backward_mode")
                and self.tracking_worker.backward_mode
            )
            is_backward_enabled = self.chk_enable_backward.isChecked()

            processed_trajectories = full_traj
            if self.enable_postprocessing.isChecked():
                params = self.get_parameters_dict()
                raw_csv_path = self.csv_line.text()

                if is_backward_mode and raw_csv_path:
                    # Use backward CSV for processing
                    base, ext = os.path.splitext(raw_csv_path)
                    csv_to_process = f"{base}_backward{ext}"
                else:
                    csv_to_process = raw_csv_path

                if csv_to_process and os.path.exists(csv_to_process):
                    # Use CSV-based processing to preserve confidence columns
                    from ..core.post_processing import (
                        process_trajectories_from_csv,
                        interpolate_trajectories,
                    )

                    processed_trajectories, stats = process_trajectories_from_csv(
                        csv_to_process, params
                    )
                    logger.info(f"Post-processing stats: {stats}")

                    # Apply interpolation if enabled
                    interp_method = (
                        self.combo_interpolation_method.currentText().lower()
                    )
                    if interp_method != "none":
                        max_gap = self.spin_interpolation_max_gap.value()
                        processed_trajectories = interpolate_trajectories(
                            processed_trajectories,
                            method=interp_method,
                            max_gap=max_gap,
                        )

                    # NOTE: Do NOT scale to original space yet if backward tracking will happen
                    # Scaling will be done after merging or in forward-only mode
                else:
                    # Fallback to old method if CSV not available
                    processed_trajectories, stats = process_trajectories(
                        full_traj, params
                    )
                    logger.info(f"Post-processing stats (fallback): {stats}")

            if not is_backward_mode:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate forward CSV as temporary (always removed)
                    forward_csv = f"{base}_forward{ext}"
                    if forward_csv not in self.temporary_files:
                        self.temporary_files.append(forward_csv)

                    processed_csv_path = f"{base}_forward_processed{ext}"
                    # Only track processed CSV as temporary if backward tracking will run
                    # (it will be merged into final file). Otherwise, this IS the final file.
                    if (
                        is_backward_enabled
                        and processed_csv_path not in self.temporary_files
                    ):
                        self.temporary_files.append(processed_csv_path)

                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )

                if is_backward_enabled:
                    self.forward_processed_trajs = processed_trajectories
                    self.start_backward_tracking()
                else:
                    # Scale coordinates to original video space (forward-only mode)
                    resize_factor = self.spin_resize.value()
                    processed_trajectories = self._scale_trajectories_to_original_space(
                        processed_trajectories, resize_factor
                    )

                    # Re-save the scaled trajectories
                    if raw_csv_path:
                        base, ext = os.path.splitext(raw_csv_path)
                        processed_csv_path = f"{base}_forward_processed{ext}"
                        self.save_trajectories_to_csv(
                            processed_trajectories, processed_csv_path
                        )

                    # Generate dataset if enabled (BEFORE cleanup so files are still available)
                    if self.chk_enable_dataset_gen.isChecked():
                        self._generate_training_dataset()

                    # Automatically export pose dataset if enabled
                    if (
                        self.chk_enable_pose_export.isChecked()
                        and self.line_pose_output_dir.text()
                    ):
                        self._export_pose_dataset()

                    # Clean up session logging - forward-only tracking complete
                    self._cleanup_session_logging()
                    self._cleanup_temporary_files()

                    # Hide stats labels
                    self.label_current_fps.setVisible(False)
                    self.label_elapsed_time.setVisible(False)
                    self.label_eta.setVisible(False)
                    self._set_ui_controls_enabled(True)
                    logger.info("✓ Tracking complete.")
            else:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate backward CSV as temporary
                    backward_csv = f"{base}_backward{ext}"
                    if backward_csv not in self.temporary_files:
                        self.temporary_files.append(backward_csv)

                    processed_csv_path = f"{base}_backward_processed{ext}"
                    # Track processed CSV as temporary
                    if processed_csv_path not in self.temporary_files:
                        self.temporary_files.append(processed_csv_path)
                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )
                self.backward_processed_trajs = processed_trajectories

                # Check if both forward and backward trajectories exist for merging
                has_forward = self.forward_processed_trajs is not None and (
                    isinstance(self.forward_processed_trajs, pd.DataFrame)
                    and not self.forward_processed_trajs.empty
                    or isinstance(self.forward_processed_trajs, list)
                    and len(self.forward_processed_trajs) > 0
                )
                has_backward = self.backward_processed_trajs is not None and (
                    isinstance(self.backward_processed_trajs, pd.DataFrame)
                    and not self.backward_processed_trajs.empty
                    or isinstance(self.backward_processed_trajs, list)
                    and len(self.backward_processed_trajs) > 0
                )

                if has_forward and has_backward:
                    self.merge_and_save_trajectories()

                # Generate dataset if enabled (BEFORE cleanup so files are still available)
                if self.chk_enable_dataset_gen.isChecked():
                    self._generate_training_dataset()

                # Automatically export pose dataset if enabled
                if (
                    self.chk_enable_pose_export.isChecked()
                    and self.line_pose_output_dir.text()
                ):
                    self._export_pose_dataset()

                # Clean up session logging - backward tracking and merging complete
                self._cleanup_session_logging()
                self._cleanup_temporary_files()

                # Hide stats labels
                self.label_current_fps.setVisible(False)
                self.label_elapsed_time.setVisible(False)
                self.label_eta.setVisible(False)
                self._set_ui_controls_enabled(True)
                logger.info("✓ Backward tracking and merging complete.")
        else:
            # Hide stats labels
            self.label_current_fps.setVisible(False)
            self.label_elapsed_time.setVisible(False)
            self.label_eta.setVisible(False)
            self._set_ui_controls_enabled(True)
            if not finished_normally:
                QMessageBox.warning(
                    self,
                    "Tracking Interrupted",
                    "Tracking was stopped or encountered an error.",
                )
        gc.collect()

    @Slot(dict)
    def on_histogram_data(self, histogram_data):
        if (
            self.enable_histograms.isChecked()
            and self.histogram_window is not None
            and self.histogram_window.isVisible()
        ):

            current_history = self.spin_histogram_history.value()
            if self.histogram_panel.history_frames != current_history:
                self.histogram_panel.set_history_frames(current_history)

            if "velocities" in histogram_data:
                self.histogram_panel.update_velocity_data(histogram_data["velocities"])
            if "sizes" in histogram_data:
                self.histogram_panel.update_size_data(histogram_data["sizes"])
            if "orientations" in histogram_data:
                self.histogram_panel.update_orientation_data(
                    histogram_data["orientations"]
                )
            if "assignment_costs" in histogram_data:
                self.histogram_panel.update_assignment_cost_data(
                    histogram_data["assignment_costs"]
                )

    def start_backward_tracking(self):
        logger.info(f"=" * 80)
        logger.info("Starting backward tracking pass...")
        logger.info(f"=" * 80)

        video_fp = self.file_line.text()
        if not video_fp:
            return
        base_name, ext = os.path.splitext(video_fp)
        reversed_video_path = f"{base_name}_reversed{ext}"

        # Track reversed video as temporary file
        if reversed_video_path not in self.temporary_files:
            self.temporary_files.append(reversed_video_path)

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Creating reversed video (FFmpeg)...")
        QApplication.processEvents()

        self.reversal_worker = VideoReversalWorker(video_fp, reversed_video_path)
        self.reversal_worker.finished.connect(self.on_reversal_finished)
        self.reversal_worker.start()

    @Slot(bool, str, str)
    def on_reversal_finished(self, success, output_path, error_message):
        self.progress_bar.setRange(0, 100)
        if not success:
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            QMessageBox.critical(self, "Error", error_message)
            return
        self.progress_label.setText("Starting backward tracking...")
        QApplication.processEvents()
        self.start_tracking_on_video(output_path, backward_mode=True)

    def start_tracking(self, preview_mode: bool, backward_mode: bool = False):
        # Only save config when NOT in preview mode
        if not preview_mode:
            if not self.save_config():
                # User cancelled config save, abort tracking
                return
        video_fp = self.file_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No video", "Please select a video file first.")
            return
        if preview_mode:
            self.start_preview_on_video(video_fp)
        else:
            self.start_tracking_on_video(video_fp, backward_mode=False)

    def start_preview_on_video(self, video_path):
        if self.tracking_worker and self.tracking_worker.isRunning():
            return
        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True
        self.csv_writer_thread = None
        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=False,
        )
        self.tracking_worker.set_parameters(self.get_parameters_dict())
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preview Mode Active")

        # Update video preview for viz-free mode
        if self.get_parameters_dict().get("VISUALIZATION_FREE_MODE", False):
            self.video_label.clear()
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #888; font-size: 14px;")

        self._set_ui_controls_enabled(False)
        self.tracking_worker.start()

    def start_tracking_on_video(self, video_path, backward_mode=False):
        if self.tracking_worker and self.tracking_worker.isRunning():
            return

        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True

        # Session logging is already set up in start_full() - don't duplicate here
        # For backward mode, we reuse the same session log

        self.csv_writer_thread = None
        if self.csv_line.text():
            # Determine header based on confidence tracking setting
            save_confidence = self.check_save_confidence.isChecked()
            if save_confidence:
                hdr = [
                    "TrackID",
                    "TrajectoryID",
                    "Index",
                    "X",
                    "Y",
                    "Theta",
                    "FrameID",
                    "State",
                    "DetectionConfidence",
                    "AssignmentConfidence",
                    "PositionUncertainty",
                ]
            else:
                hdr = [
                    "TrackID",
                    "TrajectoryID",
                    "Index",
                    "X",
                    "Y",
                    "Theta",
                    "FrameID",
                    "State",
                ]
            csv_path = self.csv_line.text()
            if backward_mode:
                base, ext = os.path.splitext(csv_path)
                csv_path = f"{base}_backward{ext}"
            self.csv_writer_thread = CSVWriterThread(csv_path, header=hdr)
            self.csv_writer_thread.start()

        video_output_path = (
            self.video_out_line.text()
            if (self.check_video_output.isChecked() and self.video_out_line.text())
            else None
        )

        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=self.csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
        )
        self.tracking_worker.set_parameters(self.get_parameters_dict())
        self.parameters_changed.connect(self.tracking_worker.update_parameters)
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Backward Tracking..." if backward_mode else "Forward Tracking..."
        )

        # Update video preview for viz-free mode
        if self.get_parameters_dict().get("VISUALIZATION_FREE_MODE", False):
            self.video_label.clear()
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #888; font-size: 14px;")

        self._set_ui_controls_enabled(False)
        self.tracking_worker.start()

    def get_parameters_dict(self):
        N = self.spin_max_targets.value()
        np.random.seed(42)
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]

        det_method = (
            "background_subtraction"
            if self.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

        yolo_path = "yolo26s-obb.pt"
        if self.combo_yolo_model.currentText() == "Custom Model...":
            yolo_path = self.yolo_custom_model_line.text() or "yolo26s-obb.pt"
        else:
            yolo_path = self.combo_yolo_model.currentText().split(" ")[0]

        yolo_cls = None
        if self.line_yolo_classes.text().strip():
            try:
                yolo_cls = [
                    int(x.strip()) for x in self.line_yolo_classes.text().split(",")
                ]
            except:
                pass

        # Calculate actual pixel values from body-size multipliers
        reference_body_size = self.spin_reference_body_size.value()
        resize_factor = self.spin_resize.value()
        scaled_body_size = reference_body_size * resize_factor

        # Area is π * (diameter/2)^2
        import math

        reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
        scaled_body_area = reference_body_area * (resize_factor**2)

        # Convert multipliers to actual pixels
        min_object_size_pixels = int(
            self.spin_min_object_size.value() * scaled_body_area
        )
        max_object_size_pixels = int(
            self.spin_max_object_size.value() * scaled_body_area
        )
        max_distance_pixels = self.spin_max_dist.value() * scaled_body_size
        recovery_search_distance_pixels = (
            self.spin_continuity_thresh.value() * scaled_body_size
        )
        min_respawn_distance_pixels = (
            self.spin_min_respawn_distance.value() * scaled_body_size
        )
        max_distance_break_pixels = (
            self.spin_max_distance_break.value() * scaled_body_size
        )

        # Convert time-based velocities to frame-based for tracking
        fps = self.spin_fps.value()
        velocity_threshold_pixels_per_frame = (
            self.spin_velocity.value() * scaled_body_size / fps
        )
        max_velocity_break_pixels_per_frame = (
            self.spin_max_velocity_break.value() * scaled_body_size / fps
        )

        return {
            "DETECTION_METHOD": det_method,
            "FPS": fps,  # Acquisition frame rate
            "YOLO_MODEL_PATH": yolo_path,
            "YOLO_CONFIDENCE_THRESHOLD": self.spin_yolo_confidence.value(),
            "YOLO_IOU_THRESHOLD": self.spin_yolo_iou.value(),
            "YOLO_TARGET_CLASSES": yolo_cls,
            "YOLO_DEVICE": self.combo_yolo_device.currentText().split(" ")[0],
            "MAX_TARGETS": N,
            "THRESHOLD_VALUE": self.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
            "ENABLE_SIZE_FILTERING": self.chk_size_filtering.isChecked(),
            "MIN_OBJECT_SIZE": min_object_size_pixels,
            "MAX_OBJECT_SIZE": max_object_size_pixels,
            "MAX_CONTOUR_MULTIPLIER": self.spin_max_contour_multiplier.value(),
            "MAX_DISTANCE_THRESHOLD": max_distance_pixels,
            "ENABLE_POSTPROCESSING": self.enable_postprocessing.isChecked(),
            "MIN_TRAJECTORY_LENGTH": self.spin_min_trajectory_length.value(),
            "MAX_VELOCITY_BREAK": max_velocity_break_pixels_per_frame,
            "MAX_DISTANCE_BREAK": max_distance_break_pixels,
            "MAX_OCCLUSION_GAP": self.spin_max_occlusion_gap.value(),
            "CONTINUITY_THRESHOLD": recovery_search_distance_pixels,
            "MIN_RESPAWN_DISTANCE": min_respawn_distance_pixels,
            "MIN_DETECTION_COUNTS": self.spin_min_detect.value(),
            "MIN_DETECTIONS_TO_START": self.spin_min_detections_to_start.value(),
            "MIN_TRACKING_COUNTS": self.spin_min_track.value(),
            "TRAJECTORY_HISTORY_SECONDS": self.spin_traj_hist.value(),
            "BACKGROUND_PRIME_FRAMES": self.spin_bg_prime.value(),
            "ENABLE_LIGHTING_STABILIZATION": self.chk_lighting_stab.isChecked(),
            "ENABLE_ADAPTIVE_BACKGROUND": self.chk_adaptive_bg.isChecked(),
            "BACKGROUND_LEARNING_RATE": self.spin_bg_learning.value(),
            "LIGHTING_SMOOTH_FACTOR": self.spin_lighting_smooth.value(),
            "LIGHTING_MEDIAN_WINDOW": self.spin_lighting_median.value(),
            "KALMAN_NOISE_COVARIANCE": self.spin_kalman_noise.value(),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": self.spin_kalman_meas.value(),
            "KALMAN_DAMPING": self.spin_kalman_damping.value(),
            "KALMAN_MATURITY_AGE": self.spin_kalman_maturity_age.value(),
            "KALMAN_INITIAL_VELOCITY_RETENTION": self.spin_kalman_initial_velocity_retention.value(),
            "KALMAN_MAX_VELOCITY_MULTIPLIER": self.spin_kalman_max_velocity.value(),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": self.spin_kalman_longitudinal_noise.value(),
            "KALMAN_LATERAL_NOISE_MULTIPLIER": self.spin_kalman_lateral_noise.value(),
            "RESIZE_FACTOR": self.spin_resize.value(),
            "ENABLE_CONSERVATIVE_SPLIT": self.chk_conservative_split.isChecked(),
            "MERGE_AREA_THRESHOLD": self.spin_merge_threshold.value(),
            "CONSERVATIVE_KERNEL_SIZE": self.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self.spin_conservative_erode.value(),
            "ENABLE_ADDITIONAL_DILATION": self.chk_additional_dilation.isChecked(),
            "DILATION_ITERATIONS": self.spin_dilation_iterations.value(),
            "DILATION_KERNEL_SIZE": self.spin_dilation_kernel_size.value(),
            "BRIGHTNESS": self.slider_brightness.value(),
            "CONTRAST": self.slider_contrast.value() / 100.0,
            "GAMMA": self.slider_gamma.value() / 100.0,
            "DARK_ON_LIGHT_BACKGROUND": self.chk_dark_on_light.isChecked(),
            "VELOCITY_THRESHOLD": velocity_threshold_pixels_per_frame,
            "INSTANT_FLIP_ORIENTATION": self.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": self.spin_max_orient.value(),
            "LOST_THRESHOLD_FRAMES": self.spin_lost_thresh.value(),
            "W_POSITION": self.spin_Wp.value(),
            "W_ORIENTATION": self.spin_Wo.value(),
            "W_AREA": self.spin_Wa.value(),
            "W_ASPECT": self.spin_Wasp.value(),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "ENABLE_GREEDY_ASSIGNMENT": self.combo_assignment_method.currentIndex()
            == 1,
            "ENABLE_SPATIAL_OPTIMIZATION": self.chk_spatial_optimization.isChecked(),
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self.chk_show_orientation.isChecked(),
            "SHOW_YOLO_OBB": self.chk_show_yolo_obb.isChecked(),
            "SHOW_TRAJECTORIES": self.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self.chk_show_labels.isChecked(),
            "SHOW_STATE": self.chk_show_state.isChecked(),
            "SHOW_KALMAN_UNCERTAINTY": self.chk_show_kalman_uncertainty.isChecked(),
            "VISUALIZATION_FREE_MODE": self.chk_visualization_free.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            "ENABLE_HISTOGRAMS": self.enable_histograms.isChecked(),
            "HISTOGRAM_HISTORY_FRAMES": self.spin_histogram_history.value(),
            "ROI_MASK": self.roi_mask,
            # Dataset generation parameters
            "ENABLE_DATASET_GENERATION": self.chk_enable_dataset_gen.isChecked(),
            "DATASET_NAME": self.line_dataset_name.text(),
            "DATASET_CLASS_NAME": self.line_dataset_class_name.text(),
            "DATASET_OUTPUT_DIR": self.line_dataset_output.text(),
            "DATASET_MAX_FRAMES": self.spin_dataset_max_frames.value(),
            "DATASET_CONF_THRESHOLD": self.spin_dataset_conf_threshold.value(),
            "DATASET_DIVERSITY_WINDOW": self.spin_dataset_diversity_window.value(),
            "DATASET_INCLUDE_CONTEXT": self.chk_dataset_include_context.isChecked(),
            "DATASET_PROBABILISTIC_SAMPLING": self.chk_dataset_probabilistic.isChecked(),
            "METRIC_LOW_CONFIDENCE": self.chk_metric_low_confidence.isChecked(),
            "METRIC_COUNT_MISMATCH": self.chk_metric_count_mismatch.isChecked(),
            "METRIC_HIGH_ASSIGNMENT_COST": self.chk_metric_high_assignment_cost.isChecked(),
            "METRIC_TRACK_LOSS": self.chk_metric_track_loss.isChecked(),
            "METRIC_HIGH_UNCERTAINTY": self.chk_metric_high_uncertainty.isChecked(),
            # Individual analysis parameters
            "ENABLE_IDENTITY_ANALYSIS": self.chk_enable_individual_analysis.isChecked(),
            "IDENTITY_METHOD": self.combo_identity_method.currentText()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""),
            "IDENTITY_CROP_SIZE_MULTIPLIER": self.spin_identity_crop_multiplier.value(),
            "IDENTITY_CROP_MIN_SIZE": self.spin_identity_crop_min.value(),
            "IDENTITY_CROP_MAX_SIZE": self.spin_identity_crop_max.value(),
            "COLOR_TAG_MODEL_PATH": self.line_color_tag_model.text(),
            "COLOR_TAG_CONFIDENCE": self.spin_color_tag_conf.value(),
            "APRILTAG_FAMILY": self.combo_apriltag_family.currentText(),
            "APRILTAG_DECIMATE": self.spin_apriltag_decimate.value(),
            "ENABLE_POSE_EXPORT": self.chk_enable_pose_export.isChecked(),
            "POSE_OUTPUT_DIR": self.line_pose_output_dir.text(),
            "POSE_DATASET_NAME": self.line_pose_dataset_name.text(),
            "POSE_CROP_SIZE_MULTIPLIER": self.spin_pose_crop_multiplier.value(),
            "POSE_MIN_TRAJECTORY_LENGTH": self.spin_pose_min_length.value(),
            "POSE_EXPORT_FPS": self.spin_pose_export_fps.value(),
            # Real-time Individual Dataset Generation parameters
            "ENABLE_INDIVIDUAL_DATASET": self.chk_enable_individual_dataset.isChecked(),
            "INDIVIDUAL_DATASET_OUTPUT_DIR": self.line_individual_output.text(),
            "INDIVIDUAL_OUTPUT_FORMAT": self.combo_individual_format.currentText().lower(),
            "INDIVIDUAL_SAVE_INTERVAL": self.spin_individual_interval.value(),
            "INDIVIDUAL_CROP_PADDING": self.spin_individual_padding.value(),
        }

    def load_config(self):
        """Manually load config from file dialog."""
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if config_path:
            self._load_config_from_file(config_path)
            self.config_status_label.setText(
                f"✓ Loaded: {os.path.basename(config_path)}"
            )
            self.config_status_label.setStyleSheet(
                "color: #4a9eff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Configuration loaded from {config_path}")

    def _load_config_from_file(self, config_path):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.
        """
        if not os.path.isfile(config_path):
            return

        def get_cfg(new_key, *legacy_keys, default=None):
            """Helper to get config value with fallback to legacy keys."""
            if new_key in cfg:
                return cfg[new_key]
            for key in legacy_keys:
                if key in cfg:
                    return cfg[key]
            return default

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # === FILE MANAGEMENT ===
            # Only set paths if they're currently empty (preserve existing paths)
            if not self.file_line.text().strip():
                self.file_line.setText(get_cfg("file_path", default=""))
            if not self.csv_line.text().strip():
                self.csv_line.setText(get_cfg("csv_path", default=""))
            self.check_video_output.setChecked(
                get_cfg("video_output_enabled", default=False)
            )
            saved_video_path = get_cfg("video_output_path", default="")
            if saved_video_path and not self.video_out_line.text().strip():
                self.video_out_line.setText(saved_video_path)

            # === REFERENCE PARAMETERS ===
            self.spin_fps.setValue(get_cfg("fps", default=30.0))
            self.spin_reference_body_size.setValue(
                get_cfg("reference_body_size", default=20.0)
            )

            # === SYSTEM PERFORMANCE ===
            self.spin_resize.setValue(get_cfg("resize_factor", default=1.0))
            self.check_save_confidence.setChecked(
                get_cfg("save_confidence_metrics", default=True)
            )
            self.chk_visualization_free.setChecked(
                get_cfg("visualization_free_mode", default=False)
            )

            # === DETECTION STRATEGY ===
            det_method = get_cfg("detection_method", default="background_subtraction")
            self.combo_detection_method.setCurrentIndex(
                0 if det_method == "background_subtraction" else 1
            )

            # === SIZE FILTERING ===
            self.chk_size_filtering.setChecked(
                get_cfg("enable_size_filtering", default=False)
            )
            self.spin_min_object_size.setValue(
                get_cfg("min_object_size_multiplier", default=0.3)
            )
            self.spin_max_object_size.setValue(
                get_cfg("max_object_size_multiplier", default=3.0)
            )

            # === IMAGE ENHANCEMENT ===
            self.slider_brightness.setValue(int(get_cfg("brightness", default=0.0)))
            self.slider_contrast.setValue(int(get_cfg("contrast", default=1.0) * 100))
            self.slider_gamma.setValue(int(get_cfg("gamma", default=1.0) * 100))
            self.chk_dark_on_light.setChecked(
                get_cfg("dark_on_light_background", default=True)
            )

            # === BACKGROUND SUBTRACTION ===
            self.spin_bg_prime.setValue(
                get_cfg("background_prime_frames", "bg_prime_frames", default=10)
            )
            self.chk_adaptive_bg.setChecked(
                get_cfg(
                    "enable_adaptive_background", "adaptive_background", default=True
                )
            )
            self.spin_bg_learning.setValue(
                get_cfg("background_learning_rate", default=0.001)
            )
            self.spin_threshold.setValue(
                get_cfg("subtraction_threshold", "threshold_value", default=50)
            )

            # === LIGHTING STABILIZATION ===
            self.chk_lighting_stab.setChecked(
                get_cfg(
                    "enable_lighting_stabilization",
                    "lighting_stabilization",
                    default=True,
                )
            )
            self.spin_lighting_smooth.setValue(
                get_cfg("lighting_smooth_factor", default=0.95)
            )
            self.spin_lighting_median.setValue(
                get_cfg("lighting_median_window", default=5)
            )

            # === MORPHOLOGY & NOISE ===
            self.spin_morph_size.setValue(get_cfg("morph_kernel_size", default=5))
            self.spin_min_contour.setValue(get_cfg("min_contour_area", default=50))
            self.spin_max_contour_multiplier.setValue(
                get_cfg("max_contour_multiplier", default=20)
            )

            # === ADVANCED SEPARATION ===
            self.chk_conservative_split.setChecked(
                get_cfg("enable_conservative_split", default=True)
            )
            self.spin_conservative_kernel.setValue(
                get_cfg("conservative_kernel_size", default=3)
            )
            self.spin_conservative_erode.setValue(
                get_cfg(
                    "conservative_erode_iterations",
                    "conservative_erode_iter",
                    default=1,
                )
            )
            self.spin_merge_threshold.setValue(
                get_cfg("merge_area_threshold", default=1000)
            )
            self.chk_additional_dilation.setChecked(
                get_cfg("enable_additional_dilation", default=False)
            )
            self.spin_dilation_kernel_size.setValue(
                get_cfg("dilation_kernel_size", default=3)
            )
            self.spin_dilation_iterations.setValue(
                get_cfg("dilation_iterations", default=2)
            )

            # === YOLO CONFIGURATION ===
            yolo_model = get_cfg("yolo_model_path", default="yolo26s-obb.pt")
            found = False
            for i in range(self.combo_yolo_model.count() - 1):
                if self.combo_yolo_model.itemText(i).startswith(yolo_model):
                    self.combo_yolo_model.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.combo_yolo_model.setCurrentIndex(self.combo_yolo_model.count() - 1)
                if not self.yolo_custom_model_line.text().strip():
                    self.yolo_custom_model_line.setText(yolo_model)

            self.spin_yolo_confidence.setValue(
                get_cfg("yolo_confidence_threshold", default=0.25)
            )
            self.spin_yolo_iou.setValue(get_cfg("yolo_iou_threshold", default=0.7))
            yolo_cls = get_cfg("yolo_target_classes", default=None)
            if yolo_cls:
                self.line_yolo_classes.setText(",".join(map(str, yolo_cls)))
            yolo_dev = get_cfg("yolo_device", default="auto")
            idx = self.combo_yolo_device.findText(yolo_dev, Qt.MatchStartsWith)
            if idx >= 0:
                self.combo_yolo_device.setCurrentIndex(idx)

            # === CORE TRACKING ===
            self.spin_max_targets.setValue(get_cfg("max_targets", default=4))
            self.spin_max_dist.setValue(
                get_cfg(
                    "max_assignment_distance_multiplier",
                    "max_dist_multiplier",
                    default=1.5,
                )
            )
            self.spin_continuity_thresh.setValue(
                get_cfg(
                    "recovery_search_distance_multiplier",
                    "continuity_threshold_multiplier",
                    default=0.5,
                )
            )
            self.chk_enable_backward.setChecked(
                get_cfg("enable_backward_tracking", default=True)
            )

            # === KALMAN FILTER ===
            self.spin_kalman_noise.setValue(
                get_cfg("kalman_process_noise", "kalman_noise", default=0.03)
            )
            self.spin_kalman_meas.setValue(
                get_cfg("kalman_measurement_noise", "kalman_meas_noise", default=0.1)
            )
            self.spin_kalman_damping.setValue(
                get_cfg("kalman_velocity_damping", "kalman_damping", default=0.95)
            )
            self.spin_kalman_maturity_age.setValue(
                get_cfg("kalman_maturity_age", default=5)
            )
            self.spin_kalman_initial_velocity_retention.setValue(
                get_cfg("kalman_initial_velocity_retention", default=0.2)
            )
            self.spin_kalman_max_velocity.setValue(
                get_cfg("kalman_max_velocity_multiplier", default=2.0)
            )
            self.spin_kalman_longitudinal_noise.setValue(
                get_cfg("kalman_longitudinal_noise_multiplier", default=5.0)
            )
            self.spin_kalman_lateral_noise.setValue(
                get_cfg("kalman_lateral_noise_multiplier", default=0.1)
            )

            # === COST FUNCTION WEIGHTS ===
            self.spin_Wp.setValue(get_cfg("weight_position", "W_POSITION", default=1.0))
            self.spin_Wo.setValue(
                get_cfg("weight_orientation", "W_ORIENTATION", default=1.0)
            )
            self.spin_Wa.setValue(get_cfg("weight_area", "W_AREA", default=0.001))
            self.spin_Wasp.setValue(
                get_cfg("weight_aspect_ratio", "W_ASPECT", default=0.1)
            )
            self.chk_use_mahal.setChecked(
                get_cfg("use_mahalanobis_distance", "USE_MAHALANOBIS", default=True)
            )

            # === ASSIGNMENT ALGORITHM ===
            self.combo_assignment_method.setCurrentIndex(
                1 if get_cfg("enable_greedy_assignment", default=False) else 0
            )
            self.chk_spatial_optimization.setChecked(
                get_cfg("enable_spatial_optimization", default=False)
            )

            # === ORIENTATION & MOTION ===
            self.spin_velocity.setValue(get_cfg("velocity_threshold", default=5.0))
            self.chk_instant_flip.setChecked(
                get_cfg("enable_instant_flip", "instant_flip", default=True)
            )
            self.spin_max_orient.setValue(
                get_cfg(
                    "max_orientation_delta_stopped",
                    "max_orient_delta_stopped",
                    default=30.0,
                )
            )

            # === TRACK LIFECYCLE ===
            self.spin_lost_thresh.setValue(
                get_cfg("lost_frames_threshold", "lost_threshold_frames", default=10)
            )
            self.spin_min_respawn_distance.setValue(
                get_cfg("min_respawn_distance_multiplier", default=2.5)
            )
            self.spin_min_detections_to_start.setValue(
                get_cfg("min_detections_to_start", default=1)
            )
            self.spin_min_detect.setValue(
                get_cfg("min_detect_frames", "min_detect_counts", default=10)
            )
            self.spin_min_track.setValue(
                get_cfg("min_track_frames", "min_track_counts", default=10)
            )

            # === POST-PROCESSING ===
            self.enable_postprocessing.setChecked(
                get_cfg("enable_postprocessing", default=True)
            )
            self.spin_min_trajectory_length.setValue(
                get_cfg("min_trajectory_length", default=10)
            )
            self.spin_max_velocity_break.setValue(
                get_cfg("max_velocity_break", default=50.0)
            )
            self.spin_max_distance_break.setValue(
                get_cfg("max_distance_break_multiplier", default=15.0)
            )
            self.spin_max_occlusion_gap.setValue(
                get_cfg("max_occlusion_gap", default=30)
            )
            interp_method = get_cfg("interpolation_method", default="None")
            idx = self.combo_interpolation_method.findText(
                interp_method, Qt.MatchFixedString
            )
            if idx >= 0:
                self.combo_interpolation_method.setCurrentIndex(idx)
            self.spin_interpolation_max_gap.setValue(
                get_cfg("interpolation_max_gap", default=10)
            )
            self.chk_cleanup_temp_files.setChecked(
                get_cfg("cleanup_temp_files", default=True)
            )

            # === REAL-TIME ANALYTICS ===
            self.enable_histograms.setChecked(
                get_cfg("enable_histograms", default=False)
            )
            self.spin_histogram_history.setValue(
                get_cfg("histogram_history_frames", default=300)
            )

            # === VISUALIZATION OVERLAYS ===
            self.chk_show_circles.setChecked(
                get_cfg("show_track_markers", "show_circles", default=True)
            )
            self.chk_show_orientation.setChecked(
                get_cfg("show_orientation_lines", "show_orientation", default=True)
            )
            self.chk_show_trajectories.setChecked(
                get_cfg("show_trajectory_trails", "show_trajectories", default=True)
            )
            self.chk_show_labels.setChecked(
                get_cfg("show_id_labels", "show_labels", default=True)
            )
            self.chk_show_state.setChecked(
                get_cfg("show_state_text", "show_state", default=True)
            )
            self.chk_show_kalman_uncertainty.setChecked(
                get_cfg("show_kalman_uncertainty", default=False)
            )
            self.chk_show_fg.setChecked(
                get_cfg("show_foreground_mask", "show_fg", default=True)
            )
            self.chk_show_bg.setChecked(
                get_cfg("show_background_model", "show_bg", default=True)
            )
            self.chk_show_yolo_obb.setChecked(get_cfg("show_yolo_obb", default=False))
            self.spin_traj_hist.setValue(
                get_cfg("trajectory_history_seconds", "traj_history", default=5)
            )
            self.chk_debug_logging.setChecked(get_cfg("debug_logging", default=False))
            self.slider_zoom.setValue(int(get_cfg("zoom_factor", default=1.0) * 100))

            # === DATASET GENERATION ===
            self.chk_enable_dataset_gen.setChecked(
                get_cfg("enable_dataset_generation", default=False)
            )
            self.line_dataset_name.setText(get_cfg("dataset_name", default=""))
            self.line_dataset_class_name.setText(
                get_cfg("dataset_class_name", default="object")
            )
            if not self.line_dataset_output.text().strip():
                self.line_dataset_output.setText(
                    get_cfg("dataset_output_dir", default="")
                )
            self.spin_dataset_max_frames.setValue(
                get_cfg("dataset_max_frames", default=100)
            )
            self.spin_dataset_conf_threshold.setValue(
                get_cfg(
                    "dataset_confidence_threshold",
                    "dataset_conf_threshold",
                    default=0.5,
                )
            )
            self.spin_dataset_diversity_window.setValue(
                get_cfg("dataset_diversity_window", default=30)
            )
            self.chk_dataset_include_context.setChecked(
                get_cfg("dataset_include_context", default=True)
            )
            self.chk_dataset_probabilistic.setChecked(
                get_cfg("dataset_probabilistic_sampling", default=True)
            )
            self.chk_metric_low_confidence.setChecked(
                get_cfg("metric_low_confidence", default=True)
            )
            self.chk_metric_count_mismatch.setChecked(
                get_cfg("metric_count_mismatch", default=True)
            )
            self.chk_metric_high_assignment_cost.setChecked(
                get_cfg("metric_high_assignment_cost", default=True)
            )
            self.chk_metric_track_loss.setChecked(
                get_cfg("metric_track_loss", default=True)
            )
            self.chk_metric_high_uncertainty.setChecked(
                get_cfg("metric_high_uncertainty", default=False)
            )

            # === INDIVIDUAL ANALYSIS ===
            self.chk_enable_individual_analysis.setChecked(
                get_cfg("enable_identity_analysis", default=False)
            )
            method_map = {
                "none_disabled": 0,
                "color_tags_yolo": 1,
                "apriltags": 2,
                "custom": 3,
            }
            identity_method = get_cfg("identity_method", default="none_disabled")
            self.combo_identity_method.setCurrentIndex(
                method_map.get(identity_method, 0)
            )
            self.spin_identity_crop_multiplier.setValue(
                get_cfg("identity_crop_size_multiplier", default=3.0)
            )
            self.spin_identity_crop_min.setValue(
                get_cfg("identity_crop_min_size", default=64)
            )
            self.spin_identity_crop_max.setValue(
                get_cfg("identity_crop_max_size", default=256)
            )
            if not self.line_color_tag_model.text().strip():
                self.line_color_tag_model.setText(
                    get_cfg("color_tag_model_path", default="")
                )
            self.spin_color_tag_conf.setValue(
                get_cfg("color_tag_confidence", default=0.5)
            )
            apriltag_family = get_cfg("apriltag_family", default="tag36h11")
            families = [
                "tag36h11",
                "tag25h9",
                "tag16h5",
                "tagCircle21h7",
                "tagStandard41h12",
            ]
            if apriltag_family in families:
                self.combo_apriltag_family.setCurrentIndex(
                    families.index(apriltag_family)
                )
            self.spin_apriltag_decimate.setValue(
                get_cfg("apriltag_decimate", default=1.0)
            )

            # === POSE EXPORT ===
            self.chk_enable_pose_export.setChecked(
                get_cfg("enable_pose_export", default=False)
            )
            if not self.line_pose_output_dir.text().strip():
                self.line_pose_output_dir.setText(
                    get_cfg("pose_output_dir", default="")
                )
            self.line_pose_dataset_name.setText(
                get_cfg("pose_dataset_name", default="pose_dataset")
            )
            self.spin_pose_crop_multiplier.setValue(
                get_cfg("pose_crop_size_multiplier", default=4.0)
            )
            self.spin_pose_min_length.setValue(
                get_cfg("pose_min_trajectory_length", default=30)
            )
            self.spin_pose_export_fps.setValue(get_cfg("pose_export_fps", default=30))

            # === REAL-TIME INDIVIDUAL DATASET ===
            self.chk_enable_individual_dataset.setChecked(
                get_cfg("enable_individual_dataset", default=False)
            )
            if not self.line_individual_output.text().strip():
                self.line_individual_output.setText(
                    get_cfg("individual_dataset_output_dir", default="")
                )
            format_text = get_cfg("individual_output_format", default="png").upper()
            format_idx = self.combo_individual_format.findText(format_text)
            if format_idx >= 0:
                self.combo_individual_format.setCurrentIndex(format_idx)
            self.spin_individual_interval.setValue(
                get_cfg("individual_save_interval", default=1)
            )
            self.spin_individual_padding.setValue(
                get_cfg("individual_crop_padding", default=0.1)
            )

            # === ROI ===
            self.roi_shapes = cfg.get("roi_shapes", [])
            if self.roi_shapes:
                # Regenerate the combined mask from loaded shapes
                # Need to get video frame dimensions first
                video_path = cfg.get("file_path", "")
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            fh, fw = frame.shape[:2]
                            self._generate_combined_roi_mask(fh, fw)
                            num_shapes = len(self.roi_shapes)
                            shape_summary = ", ".join(
                                [s["type"] for s in self.roi_shapes]
                            )
                            self.roi_status_label.setText(
                                f"Loaded ROI: {num_shapes} shape(s) ({shape_summary})"
                            )
                            self.btn_undo_roi.setEnabled(True)
                            logger.info(f"Loaded {num_shapes} ROI shapes from config")
                        cap.release()
                else:
                    # Video not available yet, just store shapes for later
                    logger.info(
                        f"Loaded {len(self.roi_shapes)} ROI shapes (mask will be generated when video loads)"
                    )
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")

    def save_config(self):
        """Save current configuration to JSON file.

        Returns:
            bool: True if config was saved successfully, False if cancelled or failed
        """
        yolo_path = (
            self.yolo_custom_model_line.text()
            if self.combo_yolo_model.currentText() == "Custom Model..."
            else self.combo_yolo_model.currentText().split(" ")[0]
        )
        yolo_cls = (
            [int(x.strip()) for x in self.line_yolo_classes.text().split(",")]
            if self.line_yolo_classes.text().strip()
            else None
        )

        cfg = {
            # === FILE MANAGEMENT ===
            "file_path": self.file_line.text(),
            "csv_path": self.csv_line.text(),
            "video_output_enabled": self.check_video_output.isChecked(),
            "video_output_path": self.video_out_line.text(),
            # === REFERENCE PARAMETERS ===
            "fps": self.spin_fps.value(),
            "reference_body_size": self.spin_reference_body_size.value(),
            # === SYSTEM PERFORMANCE ===
            "resize_factor": self.spin_resize.value(),
            "save_confidence_metrics": self.check_save_confidence.isChecked(),
            "visualization_free_mode": self.chk_visualization_free.isChecked(),
            # === DETECTION STRATEGY ===
            "detection_method": (
                "background_subtraction"
                if self.combo_detection_method.currentIndex() == 0
                else "yolo_obb"
            ),
            # === SIZE FILTERING ===
            "enable_size_filtering": self.chk_size_filtering.isChecked(),
            "min_object_size_multiplier": self.spin_min_object_size.value(),
            "max_object_size_multiplier": self.spin_max_object_size.value(),
            # === IMAGE ENHANCEMENT ===
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value() / 100.0,
            "gamma": self.slider_gamma.value() / 100.0,
            "dark_on_light_background": self.chk_dark_on_light.isChecked(),
            # === BACKGROUND SUBTRACTION ===
            "background_prime_frames": self.spin_bg_prime.value(),
            "enable_adaptive_background": self.chk_adaptive_bg.isChecked(),
            "background_learning_rate": self.spin_bg_learning.value(),
            "subtraction_threshold": self.spin_threshold.value(),
            # === LIGHTING STABILIZATION ===
            "enable_lighting_stabilization": self.chk_lighting_stab.isChecked(),
            "lighting_smooth_factor": self.spin_lighting_smooth.value(),
            "lighting_median_window": self.spin_lighting_median.value(),
            # === MORPHOLOGY & NOISE ===
            "morph_kernel_size": self.spin_morph_size.value(),
            "min_contour_area": self.spin_min_contour.value(),
            "max_contour_multiplier": self.spin_max_contour_multiplier.value(),
            # === ADVANCED SEPARATION ===
            "enable_conservative_split": self.chk_conservative_split.isChecked(),
            "conservative_kernel_size": self.spin_conservative_kernel.value(),
            "conservative_erode_iterations": self.spin_conservative_erode.value(),
            "merge_area_threshold": self.spin_merge_threshold.value(),
            "enable_additional_dilation": self.chk_additional_dilation.isChecked(),
            "dilation_kernel_size": self.spin_dilation_kernel_size.value(),
            "dilation_iterations": self.spin_dilation_iterations.value(),
            # === YOLO CONFIGURATION ===
            "yolo_model_path": yolo_path,
            "yolo_confidence_threshold": self.spin_yolo_confidence.value(),
            "yolo_iou_threshold": self.spin_yolo_iou.value(),
            "yolo_target_classes": yolo_cls,
            "yolo_device": self.combo_yolo_device.currentText().split(" ")[0],
            # === CORE TRACKING ===
            "max_targets": self.spin_max_targets.value(),
            "max_assignment_distance_multiplier": self.spin_max_dist.value(),
            "recovery_search_distance_multiplier": self.spin_continuity_thresh.value(),
            "enable_backward_tracking": self.chk_enable_backward.isChecked(),
            # === KALMAN FILTER ===
            "kalman_process_noise": self.spin_kalman_noise.value(),
            "kalman_measurement_noise": self.spin_kalman_meas.value(),
            "kalman_velocity_damping": self.spin_kalman_damping.value(),
            "kalman_maturity_age": self.spin_kalman_maturity_age.value(),
            "kalman_initial_velocity_retention": self.spin_kalman_initial_velocity_retention.value(),
            "kalman_max_velocity_multiplier": self.spin_kalman_max_velocity.value(),
            "kalman_longitudinal_noise_multiplier": self.spin_kalman_longitudinal_noise.value(),
            "kalman_lateral_noise_multiplier": self.spin_kalman_lateral_noise.value(),
            # === COST FUNCTION WEIGHTS ===
            "weight_position": self.spin_Wp.value(),
            "weight_orientation": self.spin_Wo.value(),
            "weight_area": self.spin_Wa.value(),
            "weight_aspect_ratio": self.spin_Wasp.value(),
            "use_mahalanobis_distance": self.chk_use_mahal.isChecked(),
            # === ASSIGNMENT ALGORITHM ===
            "enable_greedy_assignment": self.combo_assignment_method.currentIndex()
            == 1,
            "enable_spatial_optimization": self.chk_spatial_optimization.isChecked(),
            # === ORIENTATION & MOTION ===
            "velocity_threshold": self.spin_velocity.value(),
            "enable_instant_flip": self.chk_instant_flip.isChecked(),
            "max_orientation_delta_stopped": self.spin_max_orient.value(),
            # === TRACK LIFECYCLE ===
            "lost_frames_threshold": self.spin_lost_thresh.value(),
            "min_respawn_distance_multiplier": self.spin_min_respawn_distance.value(),
            "min_detections_to_start": self.spin_min_detections_to_start.value(),
            "min_detect_frames": self.spin_min_detect.value(),
            "min_track_frames": self.spin_min_track.value(),
            # === POST-PROCESSING ===
            "enable_postprocessing": self.enable_postprocessing.isChecked(),
            "min_trajectory_length": self.spin_min_trajectory_length.value(),
            "max_velocity_break": self.spin_max_velocity_break.value(),
            "max_distance_break_multiplier": self.spin_max_distance_break.value(),
            "max_occlusion_gap": self.spin_max_occlusion_gap.value(),
            "interpolation_method": self.combo_interpolation_method.currentText(),
            "interpolation_max_gap": self.spin_interpolation_max_gap.value(),
            "cleanup_temp_files": self.chk_cleanup_temp_files.isChecked(),
            # === REAL-TIME ANALYTICS ===
            "enable_histograms": self.enable_histograms.isChecked(),
            "histogram_history_frames": self.spin_histogram_history.value(),
            # === VISUALIZATION OVERLAYS ===
            "show_track_markers": self.chk_show_circles.isChecked(),
            "show_orientation_lines": self.chk_show_orientation.isChecked(),
            "show_trajectory_trails": self.chk_show_trajectories.isChecked(),
            "show_id_labels": self.chk_show_labels.isChecked(),
            "show_state_text": self.chk_show_state.isChecked(),
            "show_kalman_uncertainty": self.chk_show_kalman_uncertainty.isChecked(),
            "show_foreground_mask": self.chk_show_fg.isChecked(),
            "show_background_model": self.chk_show_bg.isChecked(),
            "show_yolo_obb": self.chk_show_yolo_obb.isChecked(),
            "trajectory_history_seconds": self.spin_traj_hist.value(),
            "debug_logging": self.chk_debug_logging.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            # === ROI ===
            "roi_shapes": self.roi_shapes,
            # === DATASET GENERATION ===
            "enable_dataset_generation": self.chk_enable_dataset_gen.isChecked(),
            "dataset_name": self.line_dataset_name.text(),
            "dataset_class_name": self.line_dataset_class_name.text(),
            "dataset_output_dir": self.line_dataset_output.text(),
            "dataset_max_frames": self.spin_dataset_max_frames.value(),
            "dataset_confidence_threshold": self.spin_dataset_conf_threshold.value(),
            "dataset_diversity_window": self.spin_dataset_diversity_window.value(),
            "dataset_include_context": self.chk_dataset_include_context.isChecked(),
            "dataset_probabilistic_sampling": self.chk_dataset_probabilistic.isChecked(),
            "metric_low_confidence": self.chk_metric_low_confidence.isChecked(),
            "metric_count_mismatch": self.chk_metric_count_mismatch.isChecked(),
            "metric_high_assignment_cost": self.chk_metric_high_assignment_cost.isChecked(),
            "metric_track_loss": self.chk_metric_track_loss.isChecked(),
            "metric_high_uncertainty": self.chk_metric_high_uncertainty.isChecked(),
            # === INDIVIDUAL ANALYSIS ===
            "enable_identity_analysis": self.chk_enable_individual_analysis.isChecked(),
            "identity_method": self.combo_identity_method.currentText()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""),
            "identity_crop_size_multiplier": self.spin_identity_crop_multiplier.value(),
            "identity_crop_min_size": self.spin_identity_crop_min.value(),
            "identity_crop_max_size": self.spin_identity_crop_max.value(),
            "color_tag_model_path": self.line_color_tag_model.text(),
            "color_tag_confidence": self.spin_color_tag_conf.value(),
            "apriltag_family": self.combo_apriltag_family.currentText(),
            "apriltag_decimate": self.spin_apriltag_decimate.value(),
            # === POSE EXPORT ===
            "enable_pose_export": self.chk_enable_pose_export.isChecked(),
            "pose_output_dir": self.line_pose_output_dir.text(),
            "pose_dataset_name": self.line_pose_dataset_name.text(),
            "pose_crop_size_multiplier": self.spin_pose_crop_multiplier.value(),
            "pose_min_trajectory_length": self.spin_pose_min_length.value(),
            "pose_export_fps": self.spin_pose_export_fps.value(),
            # === REAL-TIME INDIVIDUAL DATASET ===
            "enable_individual_dataset": self.chk_enable_individual_dataset.isChecked(),
            "individual_dataset_output_dir": self.line_individual_output.text(),
            "individual_output_format": self.combo_individual_format.currentText().lower(),
            "individual_save_interval": self.spin_individual_interval.value(),
            "individual_crop_padding": self.spin_individual_padding.value(),
        }

        # Determine save path: video-based if video selected, otherwise ask user
        video_path = self.file_line.text()
        if video_path:
            default_path = get_video_config_path(video_path)
        else:
            default_path = CONFIG_FILENAME

        config_path = None

        # If default path exists, ask user whether to replace or save elsewhere
        if default_path and os.path.exists(default_path):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Configuration File Exists")
            msg.setText(
                f"A configuration file already exists:\n{os.path.basename(default_path)}"
            )
            msg.setInformativeText(
                "Do you want to replace it or save to a different location?"
            )

            replace_btn = msg.addButton("Replace Existing", QMessageBox.AcceptRole)
            save_as_btn = msg.addButton("Save As...", QMessageBox.ActionRole)
            cancel_btn = msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(replace_btn)

            result = msg.exec_()
            clicked = msg.clickedButton()

            if clicked == replace_btn:
                config_path = default_path
            elif clicked == save_as_btn:
                config_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration As", default_path, "JSON Files (*.json)"
                )
            else:
                # User clicked Cancel or closed dialog - return False to cancel operation
                return False
        else:
            # No existing file, save directly to default path if available
            if default_path:
                config_path = default_path
            else:
                # No video selected, ask user where to save
                config_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration", CONFIG_FILENAME, "JSON Files (*.json)"
                )

        if config_path:
            try:
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                logger.info(
                    f"Configuration saved to {config_path} (including ROI shapes)"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to save configuration: {e}")
                return False
        else:
            # User cancelled file dialog
            return False

    def _setup_session_logging(self, video_path, backward_mode=False):
        """Set up comprehensive logging for the entire tracking session."""
        from datetime import datetime
        from pathlib import Path

        # Close existing session log if any
        self._cleanup_session_logging()

        # Only set up logging if not already set up
        if self.session_log_handler is not None:
            logger.info(f"=" * 80)
            logger.info(f"Session log already active, continuing...")
            logger.info(f"=" * 80)
            return

        # Create log file next to the video
        video_path = Path(video_path)
        log_dir = video_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{video_path.stem}_tracking_{timestamp}.log"
        log_path = log_dir / log_filename

        # Create file handler for session
        self.session_log_handler = logging.FileHandler(log_path, mode="w")
        self.session_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.session_log_handler.setFormatter(formatter)

        # Add to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self.session_log_handler)

        logger.info(f"=" * 80)
        logger.info(f"TRACKING SESSION STARTED")
        logger.info(f"Session log: {log_path}")
        logger.info(f"Video: {video_path}")
        logger.info(f"=" * 80)

    def _cleanup_session_logging(self):
        """Remove session log handler from root logger."""
        if self.session_log_handler:
            logger.info("=" * 80)
            logger.info("Tracking session completed")
            logger.info("=" * 80)

            root_logger = logging.getLogger()
            root_logger.removeHandler(self.session_log_handler)
            self.session_log_handler.close()
            self.session_log_handler = None

    def _generate_training_dataset(self):
        """Generate training dataset from tracking results for active learning."""
        try:
            from ..utils.dataset_generation import export_dataset, FrameQualityScorer

            logger.info("Starting training dataset generation...")

            # Validate parameters
            dataset_name = self.line_dataset_name.text().strip()
            output_dir = self.line_dataset_output.text().strip()

            if not dataset_name:
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Please enter a dataset name."
                )
                return

            if not output_dir:
                QMessageBox.warning(
                    self,
                    "Dataset Generation Error",
                    "Please select an output directory.",
                )
                return

            if not os.path.exists(output_dir):
                QMessageBox.warning(
                    self,
                    "Dataset Generation Error",
                    f"Output directory does not exist: {output_dir}",
                )
                return

            video_path = self.file_line.text()
            csv_path = self.csv_line.text()

            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Source video file not found."
                )
                return

            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Tracking CSV file not found."
                )
                return

            # Get parameters
            params = self.get_parameters_dict()
            max_frames = self.spin_dataset_max_frames.value()
            diversity_window = self.spin_dataset_diversity_window.value()
            include_context = self.chk_dataset_include_context.isChecked()

            # Load tracking CSV to compute quality scores
            import pandas as pd

            df = pd.read_csv(csv_path)

            # Initialize quality scorer
            scorer = FrameQualityScorer(params)

            # Score each frame
            for frame_id in df["FrameID"].unique():
                frame_data = df[df["FrameID"] == frame_id]

                # Detection data
                detection_data = {
                    "confidences": (
                        frame_data["DetectionConfidence"].tolist()
                        if "DetectionConfidence" in frame_data.columns
                        else []
                    ),
                    "count": len(frame_data),
                }

                # Tracking data (simplified - could be enhanced with actual assignment costs)
                tracking_data = {
                    "lost_tracks": int((frame_data["State"] == "lost").sum()),
                    "uncertainties": (
                        frame_data["PositionUncertainty"].tolist()
                        if "PositionUncertainty" in frame_data.columns
                        else []
                    ),
                }

                scorer.score_frame(frame_id, detection_data, tracking_data)

            # Select worst frames with diversity
            probabilistic = self.chk_dataset_probabilistic.isChecked()
            selected_frames = scorer.get_worst_frames(
                max_frames, diversity_window, probabilistic=probabilistic
            )

            if not selected_frames:
                logger.info("No frames met the quality criteria for export.")
                return

            # Get class name
            class_name = self.line_dataset_class_name.text().strip()
            if not class_name:
                class_name = "object"

            # Export dataset (non-blocking - runs in background)
            logger.info(f"Exporting {len(selected_frames)} frames to dataset...")
            zip_path = export_dataset(
                video_path=video_path,
                csv_path=csv_path,
                frame_ids=selected_frames,
                output_dir=output_dir,
                dataset_name=dataset_name,
                class_name=class_name,
                params=params,
                include_context=include_context,
            )

            logger.info(f"Dataset generation complete: {zip_path}")
            logger.info(f"Frames exported: {len(selected_frames)}")
            logger.info(
                "Next steps: Extract zip, use x-AnyLabeling to review/correct annotations, train improved YOLO model"
            )

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Dataset Generation Error",
                f"Failed to generate dataset:\n{str(e)}",
            )

    def _cleanup_temporary_files(self):
        """Remove temporary files if cleanup is enabled."""
        if not self.chk_cleanup_temp_files.isChecked():
            logger.info("Temporary file cleanup disabled, keeping intermediate files.")
            return

        if not self.temporary_files:
            logger.info("No temporary files to clean up.")
            return

        cleaned = []
        failed = []
        for temp_file in self.temporary_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    cleaned.append(os.path.basename(temp_file))
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    failed.append(os.path.basename(temp_file))
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        # Clear the list after cleanup attempt
        self.temporary_files.clear()

        if cleaned:
            logger.info(
                f"Cleaned up {len(cleaned)} temporary file(s): {', '.join(cleaned)}"
            )
        if failed:
            logger.warning(
                f"Failed to clean {len(failed)} file(s): {', '.join(failed)}"
            )

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _connect_parameter_signals(self):
        widgets_to_connect = (
            self.findChildren(QSpinBox)
            + self.findChildren(QDoubleSpinBox)
            + self.findChildren(QCheckBox)
        )
        for widget in widgets_to_connect:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_parameter_changed)
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._on_parameter_changed)

    @Slot()
    def _on_parameter_changed(self):
        params = self.get_parameters_dict()
        self.parameters_changed.emit(params)

    def _create_help_label(self, text):
        """Create a styled help label for section guidance."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(
            "color: #aaa; font-size: 11px; font-weight: normal; "
            "font-style: italic; padding: 4px 2px; margin: 2px 0px;"
        )
        return label

    def _get_roi_hash(self):
        """Generate a hash of current ROI configuration for caching."""
        if not self.roi_shapes:
            return None

        # Create a simple hash from ROI shapes
        roi_str = str(
            [
                (
                    s["type"],
                    (
                        tuple(s["params"])
                        if isinstance(s["params"], list)
                        else s["params"]
                    ),
                    s.get("mode", "include"),
                )
                for s in self.roi_shapes
            ]
        )
        return hash(roi_str)

    def _invalidate_roi_cache(self):
        """Invalidate ROI display cache when ROI changes."""
        self._roi_masked_cache.clear()
        self._roi_hash = self._get_roi_hash()

    # =========================================================================
    # ROI OPTIMIZATION AND VIDEO CROPPING
    # =========================================================================

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        # Store config in the package directory (where this file is located)
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_dir, "advanced_config.json")

        default_config = {
            "roi_crop_warning_threshold": 0.6,  # Warn if ROI is <60% of frame
            "roi_crop_auto_suggest": True,  # Auto-suggest cropping
            "roi_crop_remind_every_session": False,  # Remind every time or once
            "roi_crop_padding_fraction": 0.05,  # Padding as fraction of min(width, height) - typically 5%
            "video_crop_codec": "libx264",  # Codec for cropped videos (libx264 for quality)
            "video_crop_crf": 18,  # CRF quality (lower = better, 18 = visually lossless)
            "video_crop_preset": "medium",  # ffmpeg preset (ultrafast, fast, medium, slow, veryslow)
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded advanced config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load advanced config: {e}")

        return default_config

    def _save_advanced_config(self):
        """Save advanced configuration."""
        config_path = os.path.expanduser("~/.multi_tracker_advanced.json")
        try:
            with open(config_path, "w") as f:
                json.dump(self.advanced_config, f, indent=2)
            logger.info(f"Saved advanced config to {config_path}")
        except Exception as e:
            logger.error(f"Could not save advanced config: {e}")

    def _calculate_roi_bounding_box(self, padding=None):
        """Calculate the bounding box of the current ROI mask with optional padding.

        Args:
            padding: Fraction of min(width, height) to add as padding (e.g., 0.05 = 5%).
                    If None, uses value from advanced config.

        Returns:
            Tuple (x, y, w, h) or None if no ROI
        """
        if self.roi_mask is None:
            return None

        # Find all non-zero points in the mask
        points = cv2.findNonZero(self.roi_mask)
        if points is None or len(points) == 0:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)

        # Apply padding if requested (helps with detection by providing context)
        if padding is None:
            padding = self.advanced_config.get("roi_crop_padding_fraction", 0.05)

        if padding > 0:
            # Get frame dimensions
            frame_h, frame_w = self.roi_mask.shape[:2]

            # Calculate padding in pixels based on smaller dimension
            min_dim = min(frame_w, frame_h)
            padding_pixels = int(min_dim * padding)

            # Add padding while staying within frame bounds
            x = max(0, x - padding_pixels)
            y = max(0, y - padding_pixels)
            w = min(frame_w - x, w + 2 * padding_pixels)
            h = min(frame_h - y, h + 2 * padding_pixels)

        return (x, y, w, h)

    def _estimate_roi_efficiency(self):
        """Estimate the efficiency gain from cropping to ROI.

        Returns:
            tuple: (roi_coverage_percent, potential_speedup_factor) or (None, None)
        """
        if self.roi_mask is None or self.preview_frame_original is None:
            return (None, None)

        bbox = self._calculate_roi_bounding_box()
        if bbox is None:
            return (None, None)

        x, y, w, h = bbox
        frame_h, frame_w = self.roi_mask.shape

        frame_area = frame_w * frame_h
        roi_area = w * h

        roi_coverage = roi_area / frame_area
        # Speedup is roughly inverse of area ratio (simplification, but good estimate)
        potential_speedup = 1.0 / roi_coverage if roi_coverage > 0 else 1.0

        return (roi_coverage * 100, potential_speedup)

    def _update_roi_optimization_info(self):
        """Update the ROI optimization label with efficiency information."""
        coverage, speedup = self._estimate_roi_efficiency()

        if coverage is None:
            if hasattr(self, "roi_optimization_label"):
                self.roi_optimization_label.setText("")
            return

        threshold = self.advanced_config.get("roi_crop_warning_threshold", 0.6) * 100

        if coverage < threshold and hasattr(self, "roi_optimization_label"):
            self.roi_optimization_label.setText(
                f"⚡ ROI is {coverage:.1f}% of frame - up to {speedup:.1f}x faster if cropped!"
            )
        elif hasattr(self, "roi_optimization_label"):
            self.roi_optimization_label.setText("")

    def _check_roi_optimization_warning(self):
        """Check if we should warn the user about ROI optimization."""
        if not self.advanced_config.get("roi_crop_auto_suggest", True):
            return

        # Don't warn if already shown this session (unless configured otherwise)
        if self.roi_crop_warning_shown and not self.advanced_config.get(
            "roi_crop_remind_every_session", False
        ):
            return

        coverage, speedup = self._estimate_roi_efficiency()
        if coverage is None:
            return

        threshold = self.advanced_config.get("roi_crop_warning_threshold", 0.6) * 100

        if coverage < threshold:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("ROI Optimization Opportunity")
            msg.setText(f"⚡ Performance Optimization Available")
            msg.setInformativeText(
                f"Your ROI covers only {coverage:.1f}% of the video frame.\\n\\n"
                f"Cropping the video to the ROI bounding box could provide\\n"
                f"up to {speedup:.1f}x speedup in tracking performance!\\n\\n"
                f"Would you like to:"
            )

            btn_crop_now = msg.addButton("Crop Video Now", QMessageBox.AcceptRole)
            btn_remind_later = msg.addButton(
                "Remind Me When Tracking", QMessageBox.ActionRole
            )
            btn_dont_show = msg.addButton("Don't Show Again", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_crop_now)

            msg.exec_()
            clicked = msg.clickedButton()

            if clicked == btn_crop_now:
                self.crop_video_to_roi()
            elif clicked == btn_remind_later:
                self.roi_crop_warning_shown = True
            elif clicked == btn_dont_show:
                self.advanced_config["roi_crop_auto_suggest"] = False
                self._save_advanced_config()
                self.roi_crop_warning_shown = True

    def crop_video_to_roi(self):
        """Crop the video to the ROI bounding box and save as new file."""
        if self.roi_mask is None:
            QMessageBox.warning(self, "No ROI", "Please define an ROI before cropping.")
            return

        video_path = self.file_line.text()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        # Get padding fraction from advanced config (default 5% of min dimension)
        padding_fraction = self.advanced_config.get("roi_crop_padding_fraction", 0.05)

        bbox = self._calculate_roi_bounding_box(padding=padding_fraction)
        if bbox is None:
            QMessageBox.warning(
                self, "Invalid ROI", "Could not calculate ROI bounding box."
            )
            return

        x, y, w, h = bbox
        padding_percent = padding_fraction * 100
        logger.info(
            f"ROI crop with {padding_percent:.1f}% padding: x={x}, y={y}, w={w}, h={h}"
        )

        # Suggest output filename
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        suggested_name = f"{video_name}_cropped_roi.mp4"
        suggested_path = os.path.join(video_dir, suggested_name)

        # Ask user for output location
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cropped Video",
            suggested_path,
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )

        if not output_path:
            return  # User cancelled

        try:
            # Use ffmpeg for high-quality cropping that preserves video properties
            # This is much faster and maintains quality better than re-encoding with OpenCV
            import subprocess

            # Get video properties for the success message
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open source video")

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Build ffmpeg command for high-quality cropping
            # Get settings from advanced config
            codec = self.advanced_config.get("video_crop_codec", "libx264")
            crf = str(self.advanced_config.get("video_crop_crf", 18))
            preset = self.advanced_config.get("video_crop_preset", "medium")

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter:v",
                f"crop={w}:{h}:{x}:{y}",  # Crop to ROI bounding box
                "-c:v",
                codec,  # Video codec from config
                "-crf",
                crf,  # Quality (lower = better, 18 is visually lossless)
                "-preset",
                preset,  # Encoding speed preset
                "-c:a",
                "copy",  # Copy audio without re-encoding
                "-movflags",
                "+faststart",  # Enable streaming/fast preview
                "-y",  # Overwrite output file
                output_path,
            ]

            # Log the ffmpeg command for debugging
            logger.info(
                f"Starting video crop: {frame_w}x{frame_h} -> {w}x{h} (padding: {padding_percent:.1f}%)"
            )
            logger.info(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")

            # Show non-blocking message
            QMessageBox.information(
                self,
                "Cropping Video",
                f"Video cropping has started in the background.\n\n"
                f"Original: {frame_w}x{frame_h}\n"
                f"Cropped: {w}x{h}\n"
                f"Padding: {padding_percent:.1f}% of frame (improves detection)\n\n"
                f"Note: Padding provides visual context for better YOLO confidence.\n"
                f"Adjust 'roi_crop_padding_fraction' in advanced_config.json if needed.\n\n"
                f"This may take a few minutes. The application will remain responsive.\n"
                f"You'll be notified when cropping is complete.",
            )

            # Run ffmpeg in background with progress logging

            # Run ffmpeg in background with progress logging
            # Capture stderr for progress tracking but don't block UI
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,  # Detach from parent process
            )

            # Get total frames for progress tracking
            cap_temp = cv2.VideoCapture(video_path)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_temp.release()

            # Store process info for potential future use
            self._crop_process = {
                "process": process,
                "output_path": output_path,
                "original_size": (frame_w, frame_h),
                "cropped_size": (w, h),
                "total_frames": total_frames,
                "last_logged_progress": 0,
            }

            # Set up a timer to check when process completes
            from PySide2.QtCore import QTimer

            self._crop_check_timer = QTimer()
            self._crop_check_timer.timeout.connect(self._check_crop_completion)
            self._crop_check_timer.start(2000)  # Check every 2 seconds

            # Disable UI controls while cropping is in progress
            self._set_ui_controls_enabled(False)
            # Also disable crop button specifically
            if hasattr(self, "btn_crop_video"):
                self.btn_crop_video.setText("Cropping...")
                self.btn_crop_video.setEnabled(False)

            logger.info(f"Started background video crop: {video_path} -> {output_path}")

        except Exception as e:
            # Re-enable UI if crop failed to start
            self._set_ui_controls_enabled(True)
            if hasattr(self, "btn_crop_video"):
                self.btn_crop_video.setText("Crop Video to ROI")
                self.btn_crop_video.setEnabled(True)

            QMessageBox.critical(
                self,
                "Crop Failed",
                f"Failed to start video crop:\n{str(e)}",
            )
            logger.error(f"Video crop failed: {e}")
            import traceback

            traceback.print_exc()

    def _check_crop_completion(self):
        """Check if background crop process has completed."""
        if not hasattr(self, "_crop_process"):
            if hasattr(self, "_crop_check_timer"):
                self._crop_check_timer.stop()
            return

        process = self._crop_process["process"]

        # Read and log any new stderr output (ffmpeg progress)
        try:
            # Read available lines without blocking (non-blocking I/O)
            if process.stderr:
                import fcntl
                import os as os_module

                # Set stderr to non-blocking mode
                fd = process.stderr.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os_module.O_NONBLOCK)

                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break

                        # Parse progress from ffmpeg output
                        if "frame=" in line:
                            try:
                                frame_str = line.split("frame=")[1].split()[0]
                                current_frame = int(frame_str)
                                total_frames = self._crop_process.get("total_frames", 0)

                                # Log every 10% of progress
                                if total_frames > 0:
                                    progress_pct = int(
                                        (current_frame / total_frames) * 100
                                    )
                                    last_logged = self._crop_process.get(
                                        "last_logged_progress", 0
                                    )

                                    if progress_pct >= last_logged + 10:
                                        logger.info(
                                            f"Video crop progress: {progress_pct}% ({current_frame}/{total_frames} frames)"
                                        )
                                        self._crop_process["last_logged_progress"] = (
                                            progress_pct
                                        )
                            except (ValueError, IndexError):
                                pass
                except (IOError, OSError):
                    # No data available right now
                    pass
        except Exception:
            # Don't let logging errors break the process
            pass

        return_code = process.poll()  # Non-blocking check

        if return_code is not None:  # Process has finished
            self._crop_check_timer.stop()
            output_path = self._crop_process["output_path"]
            orig_w, orig_h = self._crop_process["original_size"]
            crop_w, crop_h = self._crop_process["cropped_size"]

            if return_code == 0 and os.path.exists(output_path):
                # Success - ask if user wants to load the cropped video
                reply = QMessageBox.question(
                    self,
                    "Crop Complete",
                    f"Video successfully cropped to ROI!\n\n"
                    f"Original: {orig_w}x{orig_h}\n"
                    f"Cropped: {crop_w}x{crop_h}\n"
                    f"Saved to: {os.path.basename(output_path)}\n\n"
                    f"Would you like to load the cropped video now?\n"
                    f"(Note: ROI will be cleared since the video is already cropped)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    # Load the cropped video with full initialization (same as select_file)
                    self.file_line.setText(output_path)
                    self.current_video_path = output_path
                    self.clear_roi()  # Clear ROI since we're loading the cropped version

                    # Auto-generate output paths based on cropped video name
                    video_dir = os.path.dirname(output_path)
                    video_name = os.path.splitext(os.path.basename(output_path))[0]

                    # Auto-populate CSV output
                    csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
                    self.csv_line.setText(csv_path)

                    # Auto-populate video output and enable it
                    video_out_path = os.path.join(
                        video_dir, f"{video_name}_tracking.mp4"
                    )
                    self.video_out_line.setText(video_out_path)
                    self.check_video_output.setChecked(True)

                    # Enable preview buttons
                    self.btn_refresh_preview.setEnabled(True)
                    self.btn_test_detection.setEnabled(True)
                    self.btn_detect_fps.setEnabled(True)

                    # Disable crop button and clear optimization info (no ROI anymore)
                    self.btn_crop_video.setEnabled(False)
                    if hasattr(self, "roi_optimization_label"):
                        self.roi_optimization_label.setText("")
                    self.roi_crop_warning_shown = False

                    # Load preview frame
                    self._load_preview_frame()

                    # Auto-load config if it exists
                    config_path = get_video_config_path(output_path)
                    if config_path and os.path.isfile(config_path):
                        self._load_config_from_file(config_path)
                        self.config_status_label.setText(
                            f"✓ Loaded: {os.path.basename(config_path)}"
                        )
                        self.config_status_label.setStyleSheet(
                            "color: #4a9eff; font-style: italic; font-size: 10px;"
                        )
                        logger.info(
                            f"Cropped video loaded: {output_path} (auto-loaded config)"
                        )
                    else:
                        self.config_status_label.setText(
                            "No config found (using current settings)"
                        )
                        self.config_status_label.setStyleSheet(
                            "color: #f39c12; font-style: italic; font-size: 10px;"
                        )
                        logger.info(
                            f"Cropped video loaded: {output_path} (no config found)"
                        )

                # Re-enable UI controls after successful crop
                self._set_ui_controls_enabled(True)
                if hasattr(self, "btn_crop_video"):
                    self.btn_crop_video.setText("Crop Video to ROI")

                logger.info(f"Successfully cropped video to {output_path}")
            else:
                # Process failed - re-enable UI
                self._set_ui_controls_enabled(True)
                if hasattr(self, "btn_crop_video"):
                    self.btn_crop_video.setText("Crop Video to ROI")
                    self.btn_crop_video.setEnabled(True)

                logger.error(f"Video crop failed with return code {return_code}")
                QMessageBox.critical(
                    self,
                    "Crop Failed",
                    f"Video cropping failed (return code: {return_code})\n\n"
                    f"Check that ffmpeg is installed and the video is valid.",
                )

            # Clean up
            del self._crop_process

    def plot_fps(self, fps_list):
        if len(fps_list) < 2:
            return
        plt.figure()
        plt.plot(fps_list)
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.title("Tracking FPS Over Time")
        plt.show()
