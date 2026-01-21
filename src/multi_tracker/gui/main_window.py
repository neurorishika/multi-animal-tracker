#!/usr/bin/env python3
"""
Main application window for the Multi-Animal Tracker.

Refactored for improved UX with Tabbed interface and logical grouping.
"""

import sys, os, json, math, logging
import numpy as np
import cv2
from collections import deque
import gc
import csv

from PySide2.QtCore import Qt, Slot, Signal, QThread, QMutex
from PySide2.QtGui import QImage, QPixmap, QPainter, QPen, QIcon
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
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
            
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
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 1px solid #4a9eff; }
            
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
        # Enhanced ROI support: multiple shapes
        self.roi_shapes = (
            []
        )  # List of dicts: {'type': 'circle'/'polygon', 'params': ...}
        self.roi_current_mode = "circle"  # 'circle' or 'polygon'

        self.histogram_panel = None
        self.histogram_window = None
        self.current_worker = None

        self.tracking_worker = None
        self.csv_writer_thread = None
        self.reversal_worker = None
        self.final_full_trajs = []

        # Preview frame for live image adjustments
        self.preview_frame_original = None  # Original frame without adjustments
        self.current_video_path = None

        # === UI CONSTRUCTION ===
        self.init_ui()

        # === POST-INIT ===
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
        self.video_label.mousePressEvent = self.record_roi_click

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

        self.roi_status_label = QLabel("No ROI")
        self.roi_status_label.setStyleSheet("color: #888; margin-left: 10px;")

        roi_layout.addWidget(roi_label)
        roi_layout.addWidget(self.combo_roi_mode)
        roi_layout.addWidget(self.btn_start_roi)
        roi_layout.addWidget(self.btn_finish_roi)
        roi_layout.addWidget(self.btn_undo_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addStretch()

        roi_main_layout.addLayout(roi_layout)

        # Second row: status
        roi_status_layout = QHBoxLayout()
        roi_status_layout.addWidget(self.roi_status_label)
        roi_main_layout.addLayout(roi_status_layout)

        # Instructions (Hidden unless active)
        self.roi_instructions = QLabel("")
        self.roi_instructions.setStyleSheet("color: #4a9eff; font-size: 11px;")
        roi_main_layout.addWidget(self.roi_instructions)

        left_layout.addWidget(self.scroll, stretch=1)
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
        preview_layout.addWidget(self.btn_refresh_preview)

        left_layout.addWidget(preview_frame)

        # --- RIGHT PANEL: Configuration Tabs & Actions ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Tab 1: Setup (Files & Performance)
        self.tab_setup = QWidget()
        self.setup_tab_ui()
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
        self.tabs.addTab(self.tab_data, "Data & Stats")

        # Tab 5: Visuals (Overlays, Debug)
        self.tab_viz = QWidget()
        self.setup_viz_ui()
        self.tabs.addTab(self.tab_viz, "Visuals")

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

        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        action_layout.addLayout(prog_layout)
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

    def setup_tab_ui(self):
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
        fl = QFormLayout(g_files)
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

        form.addWidget(g_files)

        # System Performance
        g_sys = QGroupBox("System Performance")
        fl_sys = QFormLayout(g_sys)
        fl_sys.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_resize = QDoubleSpinBox()
        self.spin_resize.setRange(0.1, 1.0)
        self.spin_resize.setSingleStep(0.1)
        self.spin_resize.setValue(1.0)
        self.spin_resize.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl_sys.addRow("Processing Resize Factor:", self.spin_resize)

        # Threads/Device hints could go here in future
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
        l_method = QHBoxLayout(g_method)
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.currentIndexChanged.connect(
            self._on_detection_method_changed_ui
        )
        l_method.addWidget(QLabel("Method:"))
        l_method.addWidget(self.combo_detection_method)
        l_method.addStretch()
        vbox.addWidget(g_method)

        # 2. Common Size Filtering (Applies to both methods)
        g_size = QGroupBox("Size Filtering")
        f_size = QFormLayout(g_size)
        self.chk_size_filtering = QCheckBox("Enable Size Constraints")
        f_size.addRow(self.chk_size_filtering)

        h_sf = QHBoxLayout()
        self.spin_min_object_size = QSpinBox()
        self.spin_min_object_size.setRange(0, 100000)
        self.spin_min_object_size.setValue(100)
        self.spin_max_object_size = QSpinBox()
        self.spin_max_object_size.setRange(0, 1000000)
        self.spin_max_object_size.setValue(5000)
        h_sf.addWidget(QLabel("Min:"))
        h_sf.addWidget(self.spin_min_object_size)
        h_sf.addWidget(QLabel("Max:"))
        h_sf.addWidget(self.spin_max_object_size)
        f_size.addRow(h_sf)
        vbox.addWidget(g_size)

        # 3. Image Pre-processing (Common) with Live Preview
        # Only shown for Background Subtraction, not YOLO
        self.g_img = QGroupBox("Image Enhancement (Pre-processing)")
        vl_img = QVBoxLayout(self.g_img)

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
        gamma_layout.addWidget(self.slider_gamma)
        vl_img.addLayout(gamma_layout)

        # Dark on light checkbox
        self.chk_dark_on_light = QCheckBox("Dark Animals on Light Background")
        self.chk_dark_on_light.setChecked(True)
        vl_img.addWidget(self.chk_dark_on_light)

        vbox.addWidget(self.g_img)

        # 3. Stacked Widget for Method Specific Params
        self.stack_detection = QStackedWidget()

        # --- Page 0: Background Subtraction Params ---
        page_bg = QWidget()
        l_bg = QVBoxLayout(page_bg)
        l_bg.setContentsMargins(0, 0, 0, 0)

        # Background Model
        g_bg_model = QGroupBox("Background Model")
        f_bg = QFormLayout(g_bg_model)
        f_bg.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_bg_prime = QSpinBox()
        self.spin_bg_prime.setRange(0, 5000)
        self.spin_bg_prime.setValue(10)
        f_bg.addRow("Priming Frames:", self.spin_bg_prime)

        self.chk_adaptive_bg = QCheckBox("Adaptive Background (Update over time)")
        self.chk_adaptive_bg.setChecked(True)
        f_bg.addRow(self.chk_adaptive_bg)

        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setValue(0.001)
        f_bg.addRow("Learning Rate:", self.spin_bg_learning)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        f_bg.addRow("Subtraction Threshold:", self.spin_threshold)
        l_bg.addWidget(g_bg_model)

        # Lighting Stab
        g_light = QGroupBox("Lighting Stabilization")
        f_light = QFormLayout(g_light)
        f_light.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.chk_lighting_stab = QCheckBox("Enable Stabilization")
        self.chk_lighting_stab.setChecked(True)
        f_light.addRow(self.chk_lighting_stab)

        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setValue(0.95)
        f_light.addRow("Smooth Factor:", self.spin_lighting_smooth)

        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setValue(5)
        f_light.addRow("Median Window:", self.spin_lighting_median)
        l_bg.addWidget(g_light)

        # Morphology (Standard)
        g_morph = QGroupBox("Morphology & Noise")
        f_morph = QFormLayout(g_morph)
        f_morph.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 50)
        self.spin_morph_size.setValue(5)
        f_morph.addRow("Main Kernel Size:", self.spin_morph_size)

        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        f_morph.addRow("Min Contour Area:", self.spin_min_contour)

        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        f_morph.addRow("Max Contour Multiplier:", self.spin_max_contour_multiplier)
        l_bg.addWidget(g_morph)

        # Morphology (Advanced/Splitting)
        g_split = QGroupBox("Advanced Separation")
        f_split = QFormLayout(g_split)
        self.chk_conservative_split = QCheckBox("Conservative Splitting (Erosion)")
        self.chk_conservative_split.setChecked(True)
        f_split.addRow(self.chk_conservative_split)

        h_split = QHBoxLayout()
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setValue(1)
        h_split.addWidget(QLabel("K-Size:"))
        h_split.addWidget(self.spin_conservative_kernel)
        h_split.addWidget(QLabel("Iters:"))
        h_split.addWidget(self.spin_conservative_erode)
        f_split.addRow(h_split)

        self.spin_merge_threshold = QSpinBox()
        self.spin_merge_threshold.setRange(100, 10000)
        self.spin_merge_threshold.setValue(1000)
        f_split.addRow("Merge Area Threshold:", self.spin_merge_threshold)

        self.chk_additional_dilation = QCheckBox("Reconnect Thin Parts (Dilation)")
        f_split.addRow(self.chk_additional_dilation)

        h_dil = QHBoxLayout()
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setValue(2)
        h_dil.addWidget(QLabel("K-Size:"))
        h_dil.addWidget(self.spin_dilation_kernel_size)
        h_dil.addWidget(QLabel("Iters:"))
        h_dil.addWidget(self.spin_dilation_iterations)
        f_split.addRow(h_dil)

        l_bg.addWidget(g_split)

        # --- Page 1: YOLO Params ---
        page_yolo = QWidget()
        l_yolo = QVBoxLayout(page_yolo)
        l_yolo.setContentsMargins(0, 0, 0, 0)

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
        f_yolo.addRow("Confidence:", self.spin_yolo_confidence)

        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        f_yolo.addRow("IOU Threshold:", self.spin_yolo_iou)

        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setPlaceholderText("e.g. 15, 16 (Empty for all)")
        f_yolo.addRow("Target Classes:", self.line_yolo_classes)

        self.combo_yolo_device = QComboBox()
        self.combo_yolo_device.addItems(["auto", "cpu", "cuda:0", "mps"])
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
        f_core = QFormLayout(g_core)
        f_core.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 20)
        self.spin_max_targets.setValue(4)
        f_core.addRow("Max Targets (Animals):", self.spin_max_targets)

        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 2000)
        self.spin_max_dist.setValue(25)
        f_core.addRow("Max Assignment Distance:", self.spin_max_dist)

        self.spin_continuity_thresh = QSpinBox()
        self.spin_continuity_thresh.setValue(10)
        f_core.addRow("Continuity Threshold:", self.spin_continuity_thresh)
        vbox.addWidget(g_core)

        # Kalman
        g_kf = QGroupBox("Kalman Filter (Motion Model)")
        f_kf = QFormLayout(g_kf)
        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setValue(0.03)
        f_kf.addRow("Process Noise:", self.spin_kalman_noise)

        self.spin_kalman_meas = QDoubleSpinBox()
        self.spin_kalman_meas.setRange(0.0, 1.0)
        self.spin_kalman_meas.setValue(0.1)
        f_kf.addRow("Measurement Noise:", self.spin_kalman_meas)
        vbox.addWidget(g_kf)

        # Weights
        g_weights = QGroupBox("Cost Function Weights")
        l_weights = QVBoxLayout(g_weights)

        row1 = QHBoxLayout()
        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setValue(1.0)
        row1.addWidget(QLabel("Position:"))
        row1.addWidget(self.spin_Wp)

        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setValue(1.0)
        row1.addWidget(QLabel("Orientation:"))
        row1.addWidget(self.spin_Wo)
        l_weights.addLayout(row1)

        row2 = QHBoxLayout()
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setSingleStep(0.001)
        self.spin_Wa.setDecimals(4)
        self.spin_Wa.setValue(0.001)
        row2.addWidget(QLabel("Area:"))
        row2.addWidget(self.spin_Wa)

        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setValue(0.1)
        row2.addWidget(QLabel("Aspect Ratio:"))
        row2.addWidget(self.spin_Wasp)
        l_weights.addLayout(row2)

        self.chk_use_mahal = QCheckBox("Use Mahalanobis Distance")
        self.chk_use_mahal.setChecked(True)
        l_weights.addWidget(self.chk_use_mahal)
        vbox.addWidget(g_weights)

        # Orientation & Lifecycle
        g_misc = QGroupBox("Orientation & Lifecycle")
        f_misc = QFormLayout(g_misc)

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setValue(2.0)
        f_misc.addRow("Motion Velocity Threshold:", self.spin_velocity)

        self.chk_instant_flip = QCheckBox("Instant Flip (Fast Motion)")
        self.chk_instant_flip.setChecked(True)
        f_misc.addRow(self.chk_instant_flip)

        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1, 180)
        self.spin_max_orient.setValue(30)
        f_misc.addRow("Max Orient Δ (Stopped):", self.spin_max_orient)

        self.spin_lost_thresh = QSpinBox()
        self.spin_lost_thresh.setValue(10)
        f_misc.addRow("Lost Frames Threshold:", self.spin_lost_thresh)

        self.spin_min_respawn_distance = QSpinBox()
        self.spin_min_respawn_distance.setValue(50)
        f_misc.addRow("Min Respawn Dist:", self.spin_min_respawn_distance)
        vbox.addWidget(g_misc)

        # Stability
        g_stab = QGroupBox("Initialization Stability")
        f_stab = QFormLayout(g_stab)
        self.spin_min_detections_to_start = QSpinBox()
        self.spin_min_detections_to_start.setValue(1)
        f_stab.addRow("Min Detections to Start:", self.spin_min_detections_to_start)

        self.spin_min_detect = QSpinBox()
        self.spin_min_detect.setValue(10)
        f_stab.addRow("Min Detect Frames:", self.spin_min_detect)

        self.spin_min_track = QSpinBox()
        self.spin_min_track.setValue(10)
        f_stab.addRow("Min Tracking Frames:", self.spin_min_track)
        vbox.addWidget(g_stab)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_data_ui(self):
        """Tab 4: Data & Post-Processing."""
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
        f_pp = QFormLayout(g_pp)
        f_pp.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.enable_postprocessing = QCheckBox("Enable Automatic Cleaning")
        self.enable_postprocessing.setChecked(True)
        f_pp.addRow(self.enable_postprocessing)

        self.spin_min_trajectory_length = QSpinBox()
        self.spin_min_trajectory_length.setRange(1, 1000)
        self.spin_min_trajectory_length.setValue(10)
        f_pp.addRow("Min Length (frames):", self.spin_min_trajectory_length)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1, 1000)
        self.spin_max_velocity_break.setValue(100.0)
        f_pp.addRow("Max Velocity Break:", self.spin_max_velocity_break)

        self.spin_max_distance_break = QDoubleSpinBox()
        self.spin_max_distance_break.setRange(1, 2000)
        self.spin_max_distance_break.setValue(300.0)
        f_pp.addRow("Max Distance Break:", self.spin_max_distance_break)
        vbox.addWidget(g_pp)

        # Histograms
        g_hist = QGroupBox("Real-Time Analytics")
        f_hist = QFormLayout(g_hist)
        self.enable_histograms = QCheckBox("Collect Histogram Data")
        f_hist.addRow(self.enable_histograms)

        self.spin_histogram_history = QSpinBox()
        self.spin_histogram_history.setRange(50, 5000)
        self.spin_histogram_history.setValue(300)
        f_hist.addRow("History Window:", self.spin_histogram_history)

        self.btn_show_histograms = QPushButton("Open Plot Window")
        self.btn_show_histograms.setCheckable(True)
        self.btn_show_histograms.clicked.connect(self.toggle_histogram_window)
        f_hist.addRow(self.btn_show_histograms)
        vbox.addWidget(g_hist)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_viz_ui(self):
        """Tab 5: Visualization & Debug."""
        layout = QVBoxLayout(self.tab_viz)
        layout.setContentsMargins(10, 10, 10, 10)

        g_overlays = QGroupBox("Video Overlays")
        v_ov = QVBoxLayout(g_overlays)

        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        v_ov.addWidget(self.chk_show_fg)

        self.chk_show_bg = QCheckBox("Show Background Model")
        self.chk_show_bg.setChecked(True)
        v_ov.addWidget(self.chk_show_bg)

        self.chk_show_circles = QCheckBox("Show Track Markers (Circles)")
        self.chk_show_circles.setChecked(True)
        v_ov.addWidget(self.chk_show_circles)

        self.chk_show_orientation = QCheckBox("Show Orientation Lines")
        self.chk_show_orientation.setChecked(True)
        v_ov.addWidget(self.chk_show_orientation)

        self.chk_show_trajectories = QCheckBox("Show Trajectory Trails")
        self.chk_show_trajectories.setChecked(True)
        v_ov.addWidget(self.chk_show_trajectories)

        self.chk_show_labels = QCheckBox("Show ID Labels")
        self.chk_show_labels.setChecked(True)
        v_ov.addWidget(self.chk_show_labels)

        self.chk_show_state = QCheckBox("Show State Text")
        self.chk_show_state.setChecked(True)
        v_ov.addWidget(self.chk_show_state)

        layout.addWidget(g_overlays)

        g_settings = QGroupBox("Display Settings")
        f_disp = QFormLayout(g_settings)
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setValue(5)
        f_disp.addRow("Trail History (sec):", self.spin_traj_hist)
        layout.addWidget(g_settings)

        g_debug = QGroupBox("Advanced / Debug")
        v_dbg = QVBoxLayout(g_debug)
        self.chk_enable_backward = QCheckBox("Run Backward Tracking after Forward")
        self.chk_enable_backward.setChecked(True)
        v_dbg.addWidget(self.chk_enable_backward)

        self.chk_debug_logging = QCheckBox("Enable Verbose Debug Logging")
        self.chk_debug_logging.stateChanged.connect(self.toggle_debug_logging)
        v_dbg.addWidget(self.chk_debug_logging)
        layout.addWidget(g_debug)

        layout.addStretch()

    # =========================================================================
    # EVENT HANDLERS (Identical Logic to Original)
    # =========================================================================

    def _on_detection_method_changed_ui(self, index):
        """Update stack widget when detection method changes."""
        self.stack_detection.setCurrentIndex(index)
        # Show image adjustments only for Background Subtraction (index 0)
        is_background_subtraction = index == 0
        self.g_img.setVisible(is_background_subtraction)
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
            self._update_preview_display()
            logger.info(f"Loaded preview frame {random_frame_idx}/{total_frames}")
        else:
            logger.warning("Failed to read preview frame")

    def _on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.label_brightness_val.setText(str(value))
        self._update_preview_display()

    def _on_contrast_changed(self, value):
        """Handle contrast slider change."""
        contrast_val = value / 100.0
        self.label_contrast_val.setText(f"{contrast_val:.2f}")
        self._update_preview_display()

    def _on_gamma_changed(self, value):
        """Handle gamma slider change."""
        gamma_val = value / 100.0
        self.label_gamma_val.setText(f"{gamma_val:.2f}")
        self._update_preview_display()

    def _on_zoom_changed(self, value):
        """Handle zoom slider change."""
        zoom_val = value / 100.0
        self.label_zoom_val.setText(f"{zoom_val:.2f}x")
        self._update_preview_display()

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        self.btn_video_out.setEnabled(checked)
        self.video_out_line.setEnabled(checked)

    def _update_preview_display(self):
        """Update the video display with current brightness/contrast/gamma settings."""
        if self.preview_frame_original is None:
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

    def _on_roi_mode_changed(self, index):
        """Handle ROI mode selection change."""
        self.roi_current_mode = "circle" if index == 0 else "polygon"
        if self.roi_selection_active:
            # If actively selecting, update instructions
            if self.roi_current_mode == "circle":
                self.roi_instructions.setText(
                    "Circle: Click 3+ points on boundary. ESC to cancel."
                )
            else:
                self.roi_instructions.setText(
                    "Polygon: Click vertices. Double-click or press Confirm to close. ESC to cancel."
                )

    def record_roi_click(self, evt):
        if not self.roi_selection_active or self.roi_base_frame is None:
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

        # Draw existing shapes first
        for shape in self.roi_shapes:
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.setPen(QPen(Qt.cyan, 2))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
            elif shape["type"] == "polygon":
                from PySide2.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(Qt.cyan, 2))
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
        if self.roi_current_mode == "circle" and len(self.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self.roi_points)
            if circle_fit:
                cx, cy, radius = circle_fit
                self.roi_fitted_circle = circle_fit
                painter.setPen(QPen(Qt.green, 3))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
                painter.setPen(QPen(Qt.blue, 8))
                painter.drawPoint(int(cx), int(cy))
                self.roi_status_label.setText(f"Preview Circle: R={radius:.1f}px")
                can_finish = True
            else:
                self.roi_status_label.setText("Invalid circle fit")
        elif self.roi_current_mode == "polygon" and len(self.roi_points) >= 3:
            # Draw preview polygon
            from PySide2.QtCore import QPoint

            points = [QPoint(int(x), int(y)) for x, y in self.roi_points]
            painter.setPen(QPen(Qt.green, 3))
            painter.drawPolygon(points)
            self.roi_status_label.setText(
                f"Preview Polygon: {len(self.roi_points)} vertices"
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

        if self.roi_current_mode == "circle":
            self.roi_status_label.setText("Click points on circle boundary")
            self.roi_instructions.setText(
                "Circle: Click 3+ points on boundary. Press ESC to cancel."
            )
        else:
            self.roi_status_label.setText("Click polygon vertices")
            self.roi_instructions.setText(
                "Polygon: Click vertices. Double-click or press Confirm to close. ESC to cancel."
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
            self.roi_shapes.append({"type": "circle", "params": (cx, cy, radius)})
            logger.info(
                f"Added circle ROI: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
            )

        elif self.roi_current_mode == "polygon":
            if len(self.roi_points) < 3:
                QMessageBox.warning(
                    self, "No ROI", "Need at least 3 points for polygon."
                )
                return
            self.roi_shapes.append({"type": "polygon", "params": list(self.roi_points)})
            logger.info(f"Added polygon ROI with {len(self.roi_points)} vertices")

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
        self.roi_instructions.setText("")

        # Update status
        num_shapes = len(self.roi_shapes)
        shape_summary = ", ".join([s["type"] for s in self.roi_shapes])
        self.roi_status_label.setText(
            f"Active ROI: {num_shapes} shape(s) ({shape_summary})"
        )

        # Show the masked result - what detector will see
        if self.roi_base_frame:
            qimg_masked = self._apply_roi_mask_to_image(self.roi_base_frame)
            self.video_label.setPixmap(QPixmap.fromImage(qimg_masked))

    def _generate_combined_roi_mask(self, height, width):
        """Generate a combined mask from all ROI shapes."""
        if not self.roi_shapes:
            self.roi_mask = None
            return

        # Create blank mask
        combined_mask = np.zeros((height, width), np.uint8)

        # Add each shape to the mask (OR operation)
        for shape in self.roi_shapes:
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 255, -1)
            elif shape["type"] == "polygon":
                points = np.array(shape["params"], dtype=np.int32)
                cv2.fillPoly(combined_mask, [points], 255)

        self.roi_mask = combined_mask
        logger.info(f"Generated combined ROI mask from {len(self.roi_shapes)} shape(s)")

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
        else:
            self.roi_status_label.setText("No ROI")

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
            self.start_tracking(preview_mode=True)
            self.btn_preview.setText("Stop Preview")
            self.btn_start.setEnabled(False)
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

    def start_full(self):
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            self.stop_tracking()
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

    @Slot(int, str)
    def on_progress_update(self, percentage, status_text):
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status_text)

    @Slot(np.ndarray)
    def on_new_frame(self, rgb):
        z = max(self.slider_zoom.value() / 100.0, 0.1)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        # ROI masking is now done in tracking worker - no need to duplicate here
        scaled = qimg.scaled(
            int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def save_trajectories_to_csv(self, trajectories, output_path):
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
                        writer.writerow(
                            [trajectory_id, int(x), int(y), theta, int(frame_id)]
                        )
            logger.info(
                f"Successfully saved {len(trajectories)} post-processed trajectories to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    def merge_and_save_trajectories(self):
        forward_trajs = getattr(self, "forward_processed_trajs", None)
        backward_trajs = getattr(self, "backward_processed_trajs", None)
        if not forward_trajs or not backward_trajs:
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
        resolved_trajectories = resolve_trajectories(
            forward_trajs,
            backward_trajs,
            video_length=total_frames,
            params=current_params,
        )

        raw_csv_path = self.csv_line.text()
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            merged_csv_path = f"{base}_merged.csv"
            if self.save_trajectories_to_csv(resolved_trajectories, merged_csv_path):
                QMessageBox.information(
                    self,
                    "Merged Data Saved",
                    f"Merged trajectory data has been saved to:\n{merged_csv_path}",
                )

    @Slot(bool, list, list)
    def on_tracking_finished(self, finished_normally, fps_list, full_traj):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if self.csv_writer_thread:
            self.csv_writer_thread.stop()
            self.csv_writer_thread.join()

        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")

        if finished_normally and not self.btn_preview.isChecked():
            logger.info("Tracking completed successfully.")
            is_backward_mode = (
                hasattr(self.tracking_worker, "backward_mode")
                and self.tracking_worker.backward_mode
            )
            is_backward_enabled = self.chk_enable_backward.isChecked()

            processed_trajectories = full_traj
            if self.enable_postprocessing.isChecked():
                params = self.get_parameters_dict()
                processed_trajectories, stats = process_trajectories(full_traj, params)
                logger.info(f"Post-processing stats: {stats}")

            if not is_backward_mode:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    processed_csv_path = f"{base}_forward_processed{ext}"
                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )

                if is_backward_enabled:
                    self.forward_processed_trajs = processed_trajectories
                    self.start_backward_tracking()
                else:
                    self._set_ui_controls_enabled(True)
                    QMessageBox.information(self, "Done", "Tracking complete.")
                    self.plot_fps(fps_list)
            else:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    processed_csv_path = f"{base}_backward_processed{ext}"
                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )
                self.backward_processed_trajs = processed_trajectories

                if self.forward_processed_trajs and self.backward_processed_trajs:
                    self.merge_and_save_trajectories()

                self._set_ui_controls_enabled(True)
                QMessageBox.information(
                    self, "Done", "Backward tracking and merging complete."
                )
                self.plot_fps(fps_list)
        else:
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
        video_fp = self.file_line.text()
        if not video_fp:
            return
        base_name, ext = os.path.splitext(video_fp)
        reversed_video_path = f"{base_name}_reversed{ext}"

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
        self.save_config()
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

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preview Mode Active")
        self._set_ui_controls_enabled(False)
        self.tracking_worker.start()

    def start_tracking_on_video(self, video_path, backward_mode=False):
        if self.tracking_worker and self.tracking_worker.isRunning():
            return

        self.csv_writer_thread = None
        if self.csv_line.text():
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

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Backward Tracking..." if backward_mode else "Forward Tracking..."
        )
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

        return {
            "DETECTION_METHOD": det_method,
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
            "MIN_OBJECT_SIZE": self.spin_min_object_size.value(),
            "MAX_OBJECT_SIZE": self.spin_max_object_size.value(),
            "MAX_CONTOUR_MULTIPLIER": self.spin_max_contour_multiplier.value(),
            "MAX_DISTANCE_THRESHOLD": self.spin_max_dist.value(),
            "ENABLE_POSTPROCESSING": self.enable_postprocessing.isChecked(),
            "MIN_TRAJECTORY_LENGTH": self.spin_min_trajectory_length.value(),
            "MAX_VELOCITY_BREAK": self.spin_max_velocity_break.value(),
            "MAX_DISTANCE_BREAK": self.spin_max_distance_break.value(),
            "CONTINUITY_THRESHOLD": self.spin_continuity_thresh.value(),
            "MIN_RESPAWN_DISTANCE": self.spin_min_respawn_distance.value(),
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
            "VELOCITY_THRESHOLD": self.spin_velocity.value(),
            "INSTANT_FLIP_ORIENTATION": self.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": self.spin_max_orient.value(),
            "LOST_THRESHOLD_FRAMES": self.spin_lost_thresh.value(),
            "W_POSITION": self.spin_Wp.value(),
            "W_ORIENTATION": self.spin_Wo.value(),
            "W_AREA": self.spin_Wa.value(),
            "W_ASPECT": self.spin_Wasp.value(),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self.chk_show_orientation.isChecked(),
            "SHOW_TRAJECTORIES": self.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self.chk_show_labels.isChecked(),
            "SHOW_STATE": self.chk_show_state.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            "ENABLE_HISTOGRAMS": self.enable_histograms.isChecked(),
            "HISTOGRAM_HISTORY_FRAMES": self.spin_histogram_history.value(),
            "ROI_MASK": self.roi_mask,
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
        """Internal method to load config from a specific file path."""
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            self.file_line.setText(cfg.get("file_path", ""))
            self.csv_line.setText(cfg.get("csv_path", ""))
            self.check_video_output.setChecked(cfg.get("video_output_enabled", False))
            # Only override video output path if one is saved in config
            saved_video_path = cfg.get("video_output_path", "")
            if saved_video_path:
                self.video_out_line.setText(saved_video_path)

            det_method = cfg.get("detection_method", "background_subtraction")
            self.combo_detection_method.setCurrentIndex(
                0 if det_method == "background_subtraction" else 1
            )

            yolo_model = cfg.get("yolo_model_path", "yolo26s-obb.pt")
            found = False
            for i in range(self.combo_yolo_model.count() - 1):
                if self.combo_yolo_model.itemText(i).startswith(yolo_model):
                    self.combo_yolo_model.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.combo_yolo_model.setCurrentIndex(self.combo_yolo_model.count() - 1)
                self.yolo_custom_model_line.setText(yolo_model)

            self.spin_yolo_confidence.setValue(
                cfg.get("yolo_confidence_threshold", 0.25)
            )
            self.spin_yolo_iou.setValue(cfg.get("yolo_iou_threshold", 0.7))

            yolo_cls = cfg.get("yolo_target_classes", None)
            if yolo_cls:
                self.line_yolo_classes.setText(",".join(map(str, yolo_cls)))

            yolo_dev = cfg.get("yolo_device", "auto")
            idx = self.combo_yolo_device.findText(yolo_dev, Qt.MatchStartsWith)
            if idx >= 0:
                self.combo_yolo_device.setCurrentIndex(idx)

            self.spin_max_targets.setValue(cfg.get("max_targets", 4))
            self.spin_threshold.setValue(cfg.get("threshold_value", 50))
            self.spin_morph_size.setValue(cfg.get("morph_kernel_size", 5))
            self.spin_min_contour.setValue(cfg.get("min_contour_area", 50))
            self.chk_size_filtering.setChecked(cfg.get("enable_size_filtering", False))
            self.spin_min_object_size.setValue(cfg.get("min_object_size", 100))
            self.spin_max_object_size.setValue(cfg.get("max_object_size", 5000))
            self.spin_max_contour_multiplier.setValue(
                cfg.get("max_contour_multiplier", 20)
            )
            self.spin_max_dist.setValue(cfg.get("max_dist_thresh", 25))
            self.enable_postprocessing.setChecked(
                cfg.get("enable_postprocessing", True)
            )
            self.spin_min_trajectory_length.setValue(
                cfg.get("min_trajectory_length", 10)
            )
            self.spin_max_velocity_break.setValue(cfg.get("max_velocity_break", 100.0))
            self.spin_max_distance_break.setValue(cfg.get("max_distance_break", 300.0))
            self.spin_continuity_thresh.setValue(cfg.get("continuity_thresh", 10))
            self.spin_min_detect.setValue(cfg.get("min_detect_counts", 10))
            self.spin_min_detections_to_start.setValue(
                cfg.get("min_detections_to_start", 1)
            )
            self.spin_min_track.setValue(cfg.get("min_track_counts", 10))
            self.spin_traj_hist.setValue(cfg.get("traj_history", 5))
            self.spin_bg_prime.setValue(cfg.get("bg_prime_frames", 10))
            self.chk_lighting_stab.setChecked(cfg.get("lighting_stabilization", True))
            self.chk_adaptive_bg.setChecked(cfg.get("adaptive_background", True))
            self.spin_bg_learning.setValue(cfg.get("background_learning_rate", 0.001))
            self.spin_lighting_smooth.setValue(cfg.get("lighting_smooth_factor", 0.95))
            self.spin_lighting_median.setValue(cfg.get("lighting_median_window", 5))
            self.spin_kalman_noise.setValue(cfg.get("kalman_noise", 0.03))
            self.spin_kalman_meas.setValue(cfg.get("kalman_meas_noise", 0.1))
            self.spin_resize.setValue(cfg.get("resize_factor", 1.0))
            self.chk_conservative_split.setChecked(
                cfg.get("enable_conservative_split", True)
            )
            self.spin_merge_threshold.setValue(cfg.get("merge_area_threshold", 1000))
            self.spin_conservative_kernel.setValue(
                cfg.get("conservative_kernel_size", 3)
            )
            self.spin_conservative_erode.setValue(cfg.get("conservative_erode_iter", 1))
            self.enable_histograms.setChecked(cfg.get("enable_histograms", False))
            self.spin_histogram_history.setValue(
                cfg.get("histogram_history_frames", 300)
            )
            self.chk_additional_dilation.setChecked(
                cfg.get("enable_additional_dilation", False)
            )
            self.spin_dilation_iterations.setValue(cfg.get("dilation_iterations", 2))
            self.spin_dilation_kernel_size.setValue(cfg.get("dilation_kernel_size", 3))
            self.slider_brightness.setValue(int(cfg.get("brightness", 0.0)))
            self.slider_contrast.setValue(int(cfg.get("contrast", 1.0) * 100))
            self.slider_gamma.setValue(int(cfg.get("gamma", 1.0) * 100))
            self.chk_dark_on_light.setChecked(cfg.get("dark_on_light_background", True))
            self.spin_velocity.setValue(cfg.get("velocity_threshold", 2.0))
            self.chk_instant_flip.setChecked(cfg.get("instant_flip", True))
            self.spin_max_orient.setValue(cfg.get("max_orient_delta_stopped", 30.0))
            self.spin_lost_thresh.setValue(cfg.get("lost_threshold_frames", 10))
            self.spin_min_respawn_distance.setValue(cfg.get("min_respawn_distance", 50))
            self.spin_Wp.setValue(cfg.get("W_POSITION", 1.0))
            self.spin_Wo.setValue(cfg.get("W_ORIENTATION", 1.0))
            self.spin_Wa.setValue(cfg.get("W_AREA", 0.001))
            self.spin_Wasp.setValue(cfg.get("W_ASPECT", 0.1))
            self.chk_use_mahal.setChecked(cfg.get("USE_MAHALANOBIS", True))
            self.chk_show_fg.setChecked(cfg.get("show_fg", True))
            self.chk_show_bg.setChecked(cfg.get("show_bg", True))
            self.chk_show_circles.setChecked(cfg.get("show_circles", True))
            self.chk_show_orientation.setChecked(cfg.get("show_orientation", True))
            self.chk_show_trajectories.setChecked(cfg.get("show_trajectories", True))
            self.chk_show_labels.setChecked(cfg.get("show_labels", True))
            self.chk_show_state.setChecked(cfg.get("show_state", True))
            self.chk_debug_logging.setChecked(cfg.get("debug_logging", False))
            self.chk_enable_backward.setChecked(
                cfg.get("enable_backward_tracking", True)
            )
            self.slider_zoom.setValue(int(cfg.get("zoom_factor", 1.0) * 100))

            # Load ROI shapes
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
            "file_path": self.file_line.text(),
            "csv_path": self.csv_line.text(),
            "video_output_enabled": self.check_video_output.isChecked(),
            "video_output_path": self.video_out_line.text(),
            "detection_method": (
                "background_subtraction"
                if self.combo_detection_method.currentIndex() == 0
                else "yolo_obb"
            ),
            "yolo_model_path": yolo_path,
            "yolo_confidence_threshold": self.spin_yolo_confidence.value(),
            "yolo_iou_threshold": self.spin_yolo_iou.value(),
            "yolo_target_classes": yolo_cls,
            "yolo_device": self.combo_yolo_device.currentText().split(" ")[0],
            "max_targets": self.spin_max_targets.value(),
            "threshold_value": self.spin_threshold.value(),
            "morph_kernel_size": self.spin_morph_size.value(),
            "min_contour_area": self.spin_min_contour.value(),
            "enable_size_filtering": self.chk_size_filtering.isChecked(),
            "min_object_size": self.spin_min_object_size.value(),
            "max_object_size": self.spin_max_object_size.value(),
            "max_contour_multiplier": self.spin_max_contour_multiplier.value(),
            "max_dist_thresh": self.spin_max_dist.value(),
            "enable_postprocessing": self.enable_postprocessing.isChecked(),
            "min_trajectory_length": self.spin_min_trajectory_length.value(),
            "max_velocity_break": self.spin_max_velocity_break.value(),
            "max_distance_break": self.spin_max_distance_break.value(),
            "continuity_thresh": self.spin_continuity_thresh.value(),
            "min_detect_counts": self.spin_min_detect.value(),
            "min_detections_to_start": self.spin_min_detections_to_start.value(),
            "min_track_counts": self.spin_min_track.value(),
            "traj_history": self.spin_traj_hist.value(),
            "bg_prime_frames": self.spin_bg_prime.value(),
            "lighting_stabilization": self.chk_lighting_stab.isChecked(),
            "adaptive_background": self.chk_adaptive_bg.isChecked(),
            "background_learning_rate": self.spin_bg_learning.value(),
            "lighting_smooth_factor": self.spin_lighting_smooth.value(),
            "lighting_median_window": self.spin_lighting_median.value(),
            "kalman_noise": self.spin_kalman_noise.value(),
            "kalman_meas_noise": self.spin_kalman_meas.value(),
            "resize_factor": self.spin_resize.value(),
            "enable_conservative_split": self.chk_conservative_split.isChecked(),
            "merge_area_threshold": self.spin_merge_threshold.value(),
            "conservative_kernel_size": self.spin_conservative_kernel.value(),
            "conservative_erode_iter": self.spin_conservative_erode.value(),
            "enable_additional_dilation": self.chk_additional_dilation.isChecked(),
            "dilation_iterations": self.spin_dilation_iterations.value(),
            "dilation_kernel_size": self.spin_dilation_kernel_size.value(),
            "brightness": self.slider_brightness.value(),
            "contrast": self.slider_contrast.value() / 100.0,
            "gamma": self.slider_gamma.value() / 100.0,
            "dark_on_light_background": self.chk_dark_on_light.isChecked(),
            "velocity_threshold": self.spin_velocity.value(),
            "instant_flip": self.chk_instant_flip.isChecked(),
            "max_orient_delta_stopped": self.spin_max_orient.value(),
            "lost_threshold_frames": self.spin_lost_thresh.value(),
            "min_respawn_distance": self.spin_min_respawn_distance.value(),
            "W_POSITION": self.spin_Wp.value(),
            "W_ORIENTATION": self.spin_Wo.value(),
            "W_AREA": self.spin_Wa.value(),
            "W_ASPECT": self.spin_Wasp.value(),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "show_fg": self.chk_show_fg.isChecked(),
            "show_bg": self.chk_show_bg.isChecked(),
            "show_circles": self.chk_show_circles.isChecked(),
            "show_orientation": self.chk_show_orientation.isChecked(),
            "show_trajectories": self.chk_show_trajectories.isChecked(),
            "show_labels": self.chk_show_labels.isChecked(),
            "show_state": self.chk_show_state.isChecked(),
            "debug_logging": self.chk_debug_logging.isChecked(),
            "enable_backward_tracking": self.chk_enable_backward.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            "enable_histograms": self.enable_histograms.isChecked(),
            "histogram_history_frames": self.spin_histogram_history.value(),
            "roi_shapes": self.roi_shapes,  # Save ROI shapes
        }

        # Determine save path: video-based if video selected, otherwise ask user
        video_path = self.file_line.text()
        if video_path:
            default_path = get_video_config_path(video_path)
        else:
            default_path = CONFIG_FILENAME

        config_path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", default_path, "JSON Files (*.json)"
        )

        if config_path:
            try:
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                logger.info(
                    f"Configuration saved to {config_path} (including ROI shapes)"
                )
            except Exception as e:
                logger.warning(f"Failed to save configuration: {e}")

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

    def plot_fps(self, fps_list):
        if len(fps_list) < 2:
            return
        plt.figure()
        plt.plot(fps_list)
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.title("Tracking FPS Over Time")
        plt.show()
