#!/usr/bin/env python3
"""
Main application window for the Multi-Animal Tracker.

This module contains the MainWindow class which provides the complete GUI interface
for video selection, parameter configuration, ROI definition, and tracking control.
"""

import sys, os, json, math, logging
import numpy as np
import cv2
from collections import deque
import gc
import csv

from PySide2.QtCore import Qt, Slot, Signal, QThread, QMutex
from PySide2.QtGui import QImage, QPixmap, QPainter, QPen
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
)
import matplotlib.pyplot as plt

from ..core.tracking_worker import TrackingWorker
from ..core.post_processing import process_trajectories, resolve_trajectories
from ..utils.csv_writer import CSVWriterThread
from ..utils.geometry import fit_circle_to_points
from ..utils.video_io import VideoReversalWorker
from .histogram_widgets import HistogramPanel


# Configuration file for saving/loading tracking parameters
CONFIG_FILENAME = "tracking_config.json"

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window providing GUI interface for multi-animal tracking.

    This class creates a comprehensive user interface that allows users to:

    **File Management:**
    - Select input video files
    - Choose output CSV file for data export
    - Load/save configuration parameters

    **ROI Definition:**
    - Define circular Region of Interest by clicking center and boundary
    - Visual feedback for ROI selection process

    **Parameter Control:**
    - Extensive parameter adjustment through GUI controls
    - Real-time parameter updates during tracking
    - Parameter persistence across sessions

    **Tracking Control:**
    - Preview mode for parameter tuning
    - Full tracking mode with data export
    - Real-time tracking visualization with zoom control

    **Data Export:**
    - CSV output with comprehensive tracking data
    - FPS performance monitoring and visualization

    The interface is organized into:
    - Left panel: Video display with ROI interaction
    - Right panel: Parameter controls and action buttons
    """

    parameters_changed = Signal(dict)

    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        self.setWindowTitle("Multi-Animal Tracking w/ Circular ROI")
        self.resize(1200, 800)

        # Set comprehensive dark mode styling for all UI elements
        self.setStyleSheet(
            """
            /* Main window and widgets */
            QMainWindow, QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            /* Group boxes with dark theme */
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: #353535;
                color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #ffffff;
                background-color: #353535;
            }
            
            /* Buttons with modern dark styling */
            QPushButton {
                background-color: #4a9eff;
                border: none;
                color: #ffffff;
                padding: 10px 18px;
                text-align: center;
                font-size: 12px;
                font-weight: 500;
                margin: 3px;
                border-radius: 6px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3d8bdb;
                border: 1px solid #5daaff;
            }
            QPushButton:pressed {
                background-color: #2a75c4;
            }
            QPushButton:checked {
                background-color: #ff6b35;
                border: 1px solid #ff8559;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
                border: 1px solid #444444;
            }
            
            /* Input controls */
            QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                selection-background-color: #4a9eff;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #4a9eff;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                background-color: #555555;
                border: none;
                border-radius: 2px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #555555;
                border: none;
                border-radius: 2px;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                color: #ffffff;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                color: #ffffff;
            }
            
            /* Checkboxes */
            QCheckBox {
                color: #ffffff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #404040;
            }
            QCheckBox::indicator:checked {
                background-color: #4a9eff;
                border: 2px solid #4a9eff;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #3d8bdb;
            }
            
            /* Combo boxes */
            QComboBox {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                min-width: 120px;
            }
            QComboBox:focus {
                border: 2px solid #4a9eff;
            }
            QComboBox::drop-down {
                background-color: #555555;
                border: none;
                border-radius: 2px;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #ffffff;
                width: 8px;
                height: 8px;
                border-top: none;
                border-right: none;
                margin-top: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #404040;
                border: 2px solid #555555;
                selection-background-color: #4a9eff;
                selection-color: #ffffff;
                color: #ffffff;
            }
            
            /* Labels */
            QLabel {
                color: #ffffff;
                background-color: transparent;
            }
            
            /* Line edits */
            QLineEdit {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                selection-background-color: #4a9eff;
            }
            QLineEdit:focus {
                border: 2px solid #4a9eff;
            }
            
            /* Progress bars */
            QProgressBar {
                background-color: #404040;
                border: 2px solid #555555;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4a9eff;
                border-radius: 2px;
            }
            
            /* Scroll areas and scroll bars */
            QScrollArea {
                background-color: #2b2b2b;
                border: 1px solid #555555;
            }
            QScrollBar:vertical {
                background-color: #404040;
                width: 16px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #666666;
                border-radius: 8px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #404040;
                height: 16px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #666666;
                border-radius: 8px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #777777;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            
            /* Form layout styling */
            QFormLayout QLabel {
                color: #cccccc;
                font-weight: 500;
            }
        """
        )

        # === ROI DEFINITION STATE ===
        # State variables for circular ROI selection
        self.roi_base_frame = None  # Frame displayed during ROI selection
        self.roi_points = []  # List of clicked points on circle circumference
        self.roi_mask = None  # Binary mask for circular ROI
        self.roi_selection_active = False  # Whether ROI selection is in progress
        self.roi_fitted_circle = (
            None  # Current best-fit circle (center_x, center_y, radius)
        )

        # === REAL-TIME HISTOGRAM STATE ===
        # Initialize histogram panel (hidden by default)
        self.histogram_panel = None  # Initialize as None first
        self.histogram_window = None  # Will be created when needed

        # === TRACKING WORKER STATE ===
        # Initialize tracking worker reference for histogram data collection
        self.current_worker = None

        # === VIDEO DISPLAY AREA ===
        # Scrollable area for video display with zoom support
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)

        # Video display label with click event handling for ROI selection
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black; color:white; font-size:14px;")
        self.video_label.setText("Select a video file to begin...")
        self.scroll.setWidget(self.video_label)

        # Override mouse click handler for ROI point selection
        self.video_label.mousePressEvent = self.record_roi_click

        # === CONTROL PANEL LAYOUT ===
        control = QVBoxLayout()

        # === FILE SELECTION CONTROLS ===
        # Video file selection
        self.btn_file = QPushButton("Select Video...")
        self.btn_file.clicked.connect(self.select_file)
        self.btn_file.setToolTip(
            "Browse and select the input video file for tracking. Supports common video formats (mp4, avi, mov, etc.)"
        )
        self.file_line = QLineEdit()
        self.file_line.setPlaceholderText("No video selected")
        hl = QHBoxLayout()
        hl.addWidget(self.btn_file)
        hl.addWidget(self.file_line)
        control.addLayout(hl)

        # CSV output file selection
        self.btn_csv = QPushButton("CSV Output...")
        self.btn_csv.clicked.connect(self.select_csv)
        self.btn_csv.setToolTip(
            "Specify output file path for tracking data in CSV format. Includes TrackID, TrajectoryID, position, orientation, and state."
        )
        self.csv_line = QLineEdit()
        self.csv_line.setPlaceholderText("No CSV selected")
        hl2 = QHBoxLayout()
        hl2.addWidget(self.btn_csv)
        hl2.addWidget(self.csv_line)
        control.addLayout(hl2)

        # Video output file selection
        self.btn_video_out = QPushButton("Video Output...")
        self.btn_video_out.clicked.connect(self.select_video_output)
        self.btn_video_out.setToolTip(
            "Optional: Specify output video file path to save tracking visualization. Video shows trajectories, labels, and tracking overlays."
        )
        self.video_out_line = QLineEdit()
        self.video_out_line.setPlaceholderText("No video output selected (optional)")
        hl3 = QHBoxLayout()
        hl3.addWidget(self.btn_video_out)
        hl3.addWidget(self.video_out_line)
        control.addLayout(hl3)

        # === TRACKING PARAMETERS PANEL ===
        # Create scrollable area for parameters
        params_scroll = QScrollArea()
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)

        # === DETECTION METHOD SELECTION ===
        detection_group = QGroupBox("Detection Method")
        detection_form = QFormLayout(detection_group)

        # Detection method dropdown
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.setCurrentIndex(0)
        self.combo_detection_method.setToolTip(
            "Choose detection method: Background Subtraction (fast, CPU-friendly) or YOLO OBB (deep learning, better for stationary animals)"
        )
        self.combo_detection_method.currentIndexChanged.connect(
            self.on_detection_method_changed
        )
        detection_form.addRow("Detection Method:", self.combo_detection_method)

        params_layout.addWidget(detection_group)

        # === YOLO-SPECIFIC PARAMETERS ===
        self.yolo_group = QGroupBox("YOLO Detection Parameters")
        yolo_form = QFormLayout(self.yolo_group)

        # YOLO model selection
        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.addItems(
            [
                "yolo26n-obb.pt (YOLO26 Nano - Fastest, 43% faster CPU)",
                "yolo26s-obb.pt (YOLO26 Small - Balanced)",
                "yolo26m-obb.pt (YOLO26 Medium)",
                "yolo26l-obb.pt (YOLO26 Large)",
                "yolo26x-obb.pt (YOLO26 Extra Large)",
                "yolov11n-obb.pt (YOLO11 Nano)",
                "yolov11s-obb.pt (YOLO11 Small)",
                "yolov11m-obb.pt (YOLO11 Medium)",
                "yolov11l-obb.pt (YOLO11 Large)",
                "yolov11x-obb.pt (YOLO11 Extra Large)",
                "Custom Model...",
            ]
        )
        self.combo_yolo_model.setCurrentIndex(1)  # Default to yolo26s-obb.pt
        self.combo_yolo_model.setToolTip(
            "Select YOLO26 (latest - Jan 2026, end-to-end, 43% faster CPU) or YOLO11 model. Models auto-download on first use. Choose 'Custom Model' for your own trained model."
        )
        self.combo_yolo_model.currentIndexChanged.connect(self.on_yolo_model_changed)
        yolo_form.addRow("YOLO Model:", self.combo_yolo_model)

        # Custom model path (hidden by default)
        self.yolo_custom_model_line = QLineEdit()
        self.yolo_custom_model_line.setPlaceholderText(
            "Path to custom YOLO model (.pt file)"
        )
        self.yolo_custom_model_line.setToolTip(
            "Full path to your custom-trained YOLO model file"
        )

        self.btn_yolo_custom_model = QPushButton("Browse...")
        self.btn_yolo_custom_model.clicked.connect(self.select_yolo_custom_model)
        self.btn_yolo_custom_model.setToolTip("Browse for custom YOLO model file")

        custom_model_layout = QHBoxLayout()
        custom_model_layout.addWidget(self.yolo_custom_model_line)
        custom_model_layout.addWidget(self.btn_yolo_custom_model)

        self.yolo_custom_model_widget = QWidget()
        self.yolo_custom_model_widget.setLayout(custom_model_layout)
        self.yolo_custom_model_widget.setVisible(False)
        yolo_form.addRow("Custom Model Path:", self.yolo_custom_model_widget)

        # YOLO confidence threshold
        self.spin_yolo_confidence = QDoubleSpinBox()
        self.spin_yolo_confidence.setRange(0.01, 1.0)
        self.spin_yolo_confidence.setValue(0.25)
        self.spin_yolo_confidence.setSingleStep(0.05)
        self.spin_yolo_confidence.setDecimals(2)
        self.spin_yolo_confidence.setToolTip(
            "Minimum confidence score for YOLO detections. Lower = more detections, higher = fewer but more confident detections"
        )
        yolo_form.addRow("Confidence Threshold:", self.spin_yolo_confidence)

        # YOLO IOU threshold
        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        self.spin_yolo_iou.setSingleStep(0.05)
        self.spin_yolo_iou.setDecimals(2)
        self.spin_yolo_iou.setToolTip(
            "IoU threshold for Non-Maximum Suppression. Lower = more aggressive overlap removal"
        )
        yolo_form.addRow("IOU Threshold:", self.spin_yolo_iou)

        # YOLO target classes (optional)
        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setPlaceholderText(
            "Leave empty for all classes, or enter comma-separated IDs (e.g., 15,16,17)"
        )
        self.line_yolo_classes.setToolTip(
            "Specify COCO class IDs to detect (e.g., 14=bird, 15=cat, 16=dog). Leave empty to detect all classes."
        )
        yolo_form.addRow("Target Classes:", self.line_yolo_classes)

        # Initially hide YOLO parameters
        self.yolo_group.setVisible(False)
        params_layout.addWidget(self.yolo_group)

        # === CORE TRACKING PARAMETERS ===
        core_group = QGroupBox("Core Tracking")
        core_form = QFormLayout()

        # Maximum number of simultaneous tracks
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 20)
        self.spin_max_targets.setValue(4)
        self.spin_max_targets.setToolTip(
            "Maximum number of animals to track simultaneously. Should match the actual number of animals in your video."
        )
        core_form.addRow("Max Targets:", self.spin_max_targets)

        # Global threshold for foreground detection
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setToolTip(
            "Intensity threshold for foreground detection. Higher values = more selective detection. Adjust based on contrast between animals and background."
        )
        core_form.addRow("Global Threshold:", self.spin_threshold)

        # Maximum assignment distance for track-detection matching
        self.spin_max_dist = QSpinBox()
        self.spin_max_dist.setRange(0, 2000)
        self.spin_max_dist.setValue(25)
        self.spin_max_dist.setToolTip(
            "Maximum distance (pixels) between predicted and detected positions for track assignment. Lower values = stricter matching."
        )
        core_form.addRow("Max Distance Thresh:", self.spin_max_dist)

        # === POST-PROCESSING PARAMETERS ===
        postproc_group = QGroupBox("Trajectory Post-Processing")
        postproc_form = QFormLayout(postproc_group)

        self.enable_postprocessing = QCheckBox()
        self.enable_postprocessing.setChecked(True)
        self.enable_postprocessing.setToolTip(
            "Enable automatic trajectory cleaning and optimization"
        )
        postproc_form.addRow("Enable Post-Processing:", self.enable_postprocessing)

        self.spin_min_trajectory_length = QSpinBox()
        self.spin_min_trajectory_length.setRange(1, 1000)
        self.spin_min_trajectory_length.setValue(10)
        self.spin_min_trajectory_length.setSuffix(" frames")
        self.spin_min_trajectory_length.setToolTip(
            "Remove trajectories shorter than this (likely noise)"
        )
        postproc_form.addRow("Min Trajectory Length:", self.spin_min_trajectory_length)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1.0, 1000.0)
        self.spin_max_velocity_break.setValue(100.0)
        self.spin_max_velocity_break.setDecimals(1)
        self.spin_max_velocity_break.setSuffix(" px/frame")
        self.spin_max_velocity_break.setToolTip(
            "Break trajectories at points exceeding this velocity (identity switches)"
        )
        postproc_form.addRow("Max Velocity (Break):", self.spin_max_velocity_break)

        self.spin_max_distance_break = QDoubleSpinBox()
        self.spin_max_distance_break.setRange(1.0, 2000.0)
        self.spin_max_distance_break.setValue(300.0)
        self.spin_max_distance_break.setDecimals(1)
        self.spin_max_distance_break.setSuffix(" px")
        self.spin_max_distance_break.setToolTip(
            "Break trajectories at points exceeding this distance jump"
        )
        postproc_form.addRow("Max Distance (Break):", self.spin_max_distance_break)

        # Add the post-processing group to the parameters layout
        params_layout.addWidget(postproc_group)

        # === REAL-TIME HISTOGRAM CONTROLS ===
        histogram_group = QGroupBox("Real-Time Histograms")
        histogram_form = QFormLayout(histogram_group)

        self.enable_histograms = QCheckBox()
        self.enable_histograms.setChecked(False)
        self.enable_histograms.setToolTip(
            "Show real-time parameter histograms for debugging and parameter tuning"
        )
        histogram_form.addRow("Enable Histograms:", self.enable_histograms)

        self.spin_histogram_history = QSpinBox()
        self.spin_histogram_history.setRange(50, 1000)
        self.spin_histogram_history.setValue(300)
        self.spin_histogram_history.setSuffix(" frames")
        self.spin_histogram_history.setToolTip(
            "Number of recent frames to include in histogram displays"
        )
        histogram_form.addRow("History Window:", self.spin_histogram_history)

        # Button to show/hide histogram window
        self.btn_show_histograms = QPushButton("Show Histograms")
        self.btn_show_histograms.setCheckable(True)
        self.btn_show_histograms.clicked.connect(self.toggle_histogram_window)
        self.btn_show_histograms.setToolTip("Open/close the histogram display window")
        histogram_form.addRow("Display:", self.btn_show_histograms)

        params_layout.addWidget(histogram_group)

        # Continuity threshold for Hungarian vs Priority assignment
        self.spin_continuity_thresh = QSpinBox()
        self.spin_continuity_thresh.setRange(1, 100)
        self.spin_continuity_thresh.setValue(10)
        self.spin_continuity_thresh.setToolTip(
            "Number of frames needed to consider a track 'established'. Established tracks get priority in assignment algorithms."
        )
        core_form.addRow("Continuity Threshold:", self.spin_continuity_thresh)

        core_group.setLayout(core_form)
        params_layout.addWidget(core_group)

        # === MORPHOLOGICAL PROCESSING ===
        morph_group = QGroupBox("Morphological Processing")
        morph_form = QFormLayout()

        # Morphological kernel size for noise removal
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 50)
        self.spin_morph_size.setValue(5)
        self.spin_morph_size.setToolTip(
            "Size of morphological kernel for noise removal. Larger values remove more noise but may merge nearby objects."
        )
        morph_form.addRow("Kernel Size:", self.spin_morph_size)

        # Minimum contour area to filter noise
        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        self.spin_min_contour.setToolTip(
            "Minimum contour area (pixels) to be considered a valid detection. Helps filter out noise and artifacts."
        )
        morph_form.addRow("Min Contour Area:", self.spin_min_contour)

        # === SIZE-BASED FILTERING ===
        # Enable/disable size-based filtering
        self.chk_size_filtering = QCheckBox("Enable Size Filtering")
        self.chk_size_filtering.setChecked(False)
        self.chk_size_filtering.setToolTip(
            "Filter detections by size range to exclude objects that are too small or too large to be your target animals."
        )
        morph_form.addRow(self.chk_size_filtering)

        # Minimum object size for detection
        self.spin_min_object_size = QSpinBox()
        self.spin_min_object_size.setRange(0, 100000)
        self.spin_min_object_size.setValue(100)
        self.spin_min_object_size.setToolTip(
            "Minimum object size (pixels) for size filtering. Objects smaller than this will be ignored."
        )
        morph_form.addRow("Min Object Size:", self.spin_min_object_size)

        # Maximum object size for detection
        self.spin_max_object_size = QSpinBox()
        self.spin_max_object_size.setRange(100, 1000000)
        self.spin_max_object_size.setValue(5000)
        self.spin_max_object_size.setToolTip(
            "Maximum object size (pixels) for size filtering. Objects larger than this will be ignored."
        )
        morph_form.addRow("Max Object Size:", self.spin_max_object_size)

        # === FRAME QUALITY CONTROL ===
        # Maximum contour multiplier for frame quality check
        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        self.spin_max_contour_multiplier.setToolTip(
            "Skip frames with too many contours (more than this multiplier × max targets). Prevents processing of extremely noisy frames."
        )
        morph_form.addRow("Max Contour Multiplier:", self.spin_max_contour_multiplier)

        # === CONSERVATIVE SPLITTING CONTROLS ===
        # Enable/disable conservative object splitting
        self.chk_conservative_split = QCheckBox("Enable Conservative Object Splitting")
        self.chk_conservative_split.setChecked(True)
        self.chk_conservative_split.setToolTip(
            "Attempt to split merged objects using erosion. Useful when animals are close together or touching."
        )
        morph_form.addRow(self.chk_conservative_split)

        # Conservative kernel size
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setRange(1, 20)
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_kernel.setToolTip(
            "Kernel size for conservative splitting operations. Smaller values = more gentle splitting."
        )
        morph_form.addRow("Conservative Kernel Size:", self.spin_conservative_kernel)

        # Conservative erosion iterations
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setRange(1, 10)
        self.spin_conservative_erode.setValue(1)
        self.spin_conservative_erode.setToolTip(
            "Number of erosion iterations for conservative splitting. More iterations = stronger separation."
        )
        morph_form.addRow("Conservative Erode Iter:", self.spin_conservative_erode)

        # Merge area threshold for triggering splitting
        self.spin_merge_threshold = QSpinBox()
        self.spin_merge_threshold.setRange(100, 10000)
        self.spin_merge_threshold.setValue(1000)
        self.spin_merge_threshold.setToolTip(
            "Area threshold (pixels) above which objects are considered potentially merged and splitting is attempted."
        )
        morph_form.addRow("Merge Area Threshold:", self.spin_merge_threshold)

        # === ADDITIONAL DILATION FOR THIN ANIMALS ===
        # Enable/disable additional dilation to connect split animal parts
        self.chk_additional_dilation = QCheckBox("Enable Additional Dilation")
        self.chk_additional_dilation.setChecked(False)
        self.chk_additional_dilation.setToolTip(
            "Apply extra dilation to connect separated parts of thin animals (e.g., legs, antennae). May merge nearby objects."
        )
        morph_form.addRow(self.chk_additional_dilation)

        # Number of additional dilation iterations
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setRange(1, 10)
        self.spin_dilation_iterations.setValue(2)
        self.spin_dilation_iterations.setToolTip(
            "Number of dilation iterations to apply. More iterations = stronger connection of separated parts."
        )
        morph_form.addRow("Dilation Iterations:", self.spin_dilation_iterations)

        # Dilation kernel size (can be different from main morph kernel)
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setRange(1, 20)
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_kernel_size.setToolTip(
            "Size of dilation kernel. Larger kernels connect parts that are further apart."
        )
        morph_form.addRow("Dilation Kernel Size:", self.spin_dilation_kernel_size)

        morph_group.setLayout(morph_form)
        params_layout.addWidget(morph_group)

        # === SYSTEM STABILITY ===
        stability_group = QGroupBox("System Stability")
        stability_form = QFormLayout()

        # Frames needed for detection system initialization
        self.spin_min_detect = QSpinBox()
        self.spin_min_detect.setRange(0, 1000)
        self.spin_min_detect.setValue(10)
        self.spin_min_detect.setToolTip(
            "Number of consecutive frames with good detections needed to initialize tracking."
        )
        stability_form.addRow("Min Detection Counts:", self.spin_min_detect)

        # Minimum detections needed to start tracking
        self.spin_min_detections_to_start = QSpinBox()
        self.spin_min_detections_to_start.setRange(1, 20)
        self.spin_min_detections_to_start.setValue(1)
        self.spin_min_detections_to_start.setToolTip(
            "Minimum number of detections required in a frame to begin tracking (1 = start immediately when any animal is detected)."
        )
        stability_form.addRow(
            "Min Detections to Start:", self.spin_min_detections_to_start
        )

        # Frames needed for tracking system stabilization
        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(0, 1000)
        self.spin_min_track.setValue(10)
        self.spin_min_track.setToolTip(
            "Number of consecutive frames with good tracking performance needed to consider the system stabilized."
        )
        stability_form.addRow("Min Tracking Counts:", self.spin_min_track)

        # Trajectory history length for visualization (seconds)
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setRange(0, 300)
        self.spin_traj_hist.setValue(5)
        self.spin_traj_hist.setToolTip(
            "Length of trajectory trails to display (in seconds of video). Longer trails use more memory."
        )
        stability_form.addRow("Trajectory History (sec):", self.spin_traj_hist)

        stability_group.setLayout(stability_form)
        params_layout.addWidget(stability_group)

        # === BACKGROUND MODEL ===
        bg_group = QGroupBox("Background Model")
        bg_form = QFormLayout()

        # Number of frames for background initialization
        self.spin_bg_prime = QSpinBox()
        self.spin_bg_prime.setRange(0, 5000)
        self.spin_bg_prime.setValue(10)
        self.spin_bg_prime.setToolTip(
            "Number of random frames to sample for building the initial background model. More frames = better background model."
        )
        bg_form.addRow("Prime Frames:", self.spin_bg_prime)

        # Enable adaptive background learning
        self.chk_adaptive_bg = QCheckBox("Enable Adaptive Background")
        self.chk_adaptive_bg.setChecked(True)
        self.chk_adaptive_bg.setToolTip(
            "Continuously update background model to adapt to lighting changes. Disable for static lighting conditions."
        )
        bg_form.addRow(self.chk_adaptive_bg)

        # Background learning rate for lighting adaptation
        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setSingleStep(0.0001)
        self.spin_bg_learning.setValue(0.001)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setToolTip(
            "Learning rate for adaptive background updates. Higher values adapt faster to changes but are less stable."
        )
        bg_form.addRow("Learning Rate:", self.spin_bg_learning)

        bg_group.setLayout(bg_form)
        params_layout.addWidget(bg_group)

        # === LIGHTING STABILIZATION ===
        lighting_group = QGroupBox("Lighting Stabilization")
        lighting_form = QFormLayout()

        # Enable automatic lighting stabilization
        self.chk_lighting_stab = QCheckBox("Enable Lighting Stabilization")
        self.chk_lighting_stab.setChecked(True)
        self.chk_lighting_stab.setToolTip(
            "Automatically compensate for gradual lighting changes throughout the video. Improves detection consistency."
        )
        lighting_form.addRow(self.chk_lighting_stab)

        # Lighting smoothing factor
        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setSingleStep(0.01)
        self.spin_lighting_smooth.setValue(0.95)
        self.spin_lighting_smooth.setToolTip(
            "Smoothing factor for lighting stabilization. Higher values = slower adaptation to lighting changes."
        )
        lighting_form.addRow("Smooth Factor:", self.spin_lighting_smooth)

        # Median filter window for lighting stabilization
        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setValue(5)
        self.spin_lighting_median.setToolTip(
            "Window size for median filtering in lighting stabilization. Larger windows remove more noise but respond slower."
        )
        lighting_form.addRow("Median Window:", self.spin_lighting_median)

        lighting_group.setLayout(lighting_form)
        params_layout.addWidget(lighting_group)

        # === KALMAN FILTER PARAMETERS ===
        kalman_group = QGroupBox("Kalman Filter")
        kalman_form = QFormLayout()

        # Process noise covariance (motion model uncertainty)
        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setSingleStep(0.01)
        self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setToolTip(
            "Process noise for Kalman filter. Higher values make predictions less confident, allowing for more erratic motion."
        )
        kalman_form.addRow("Process Noise:", self.spin_kalman_noise)

        # Measurement noise covariance (observation uncertainty)
        self.spin_kalman_meas = QDoubleSpinBox()
        self.spin_kalman_meas.setRange(0.0, 1.0)
        self.spin_kalman_meas.setSingleStep(0.01)
        self.spin_kalman_meas.setValue(0.1)
        self.spin_kalman_meas.setToolTip(
            "Measurement noise for Kalman filter. Higher values trust observations less, relying more on predictions."
        )
        kalman_form.addRow("Measurement Noise:", self.spin_kalman_meas)

        kalman_group.setLayout(kalman_form)
        params_layout.addWidget(kalman_group)

        # === PERFORMANCE OPTIMIZATION ===
        performance_group = QGroupBox("Performance")
        performance_form = QFormLayout()

        # Video resize factor for performance optimization
        self.spin_resize = QDoubleSpinBox()
        self.spin_resize.setRange(0.1, 1.0)
        self.spin_resize.setSingleStep(0.1)
        self.spin_resize.setValue(1.0)
        self.spin_resize.setToolTip(
            "Resize factor for video processing. Values < 1.0 improve performance but reduce accuracy. 0.5 = half resolution."
        )
        performance_form.addRow("Resize Factor:", self.spin_resize)

        performance_group.setLayout(performance_form)
        params_layout.addWidget(performance_group)

        # === IMAGE ENHANCEMENT ===
        image_group = QGroupBox("Image Enhancement")
        image_form = QFormLayout()

        # Brightness adjustment (-255 to +255)
        self.spin_brightness = QDoubleSpinBox()
        self.spin_brightness.setRange(-255, 255)
        self.spin_brightness.setSingleStep(5)
        self.spin_brightness.setValue(0)
        self.spin_brightness.setToolTip(
            "Brightness adjustment for the video. Positive values brighten, negative values darken. Use to improve contrast."
        )
        image_form.addRow("Brightness:", self.spin_brightness)

        # Contrast multiplier (0.0 to 3.0+)
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.0, 3.0)
        self.spin_contrast.setSingleStep(0.1)
        self.spin_contrast.setValue(1.0)
        self.spin_contrast.setToolTip(
            "Contrast multiplier. Values > 1.0 increase contrast, < 1.0 decrease contrast. Use to enhance animal-background separation."
        )
        image_form.addRow("Contrast:", self.spin_contrast)

        # Gamma correction factor (0.1 to 3.0+)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 3.0)
        self.spin_gamma.setSingleStep(0.1)
        self.spin_gamma.setValue(1.0)
        self.spin_gamma.setToolTip(
            "Gamma correction factor. Values > 1.0 brighten mid-tones, < 1.0 darken mid-tones. Fine-tunes brightness curve."
        )
        image_form.addRow("Gamma:", self.spin_gamma)

        # === BACKGROUND TYPE SELECTION ===
        # Select whether animals are dark on light background or light on dark background
        self.chk_dark_on_light = QCheckBox("Dark Animals on Light Background")
        self.chk_dark_on_light.setChecked(True)  # Default: dark flies on light surface
        self.chk_dark_on_light.setToolTip(
            "Check if animals are darker than background (e.g., black flies on white paper). Uncheck for light animals on dark background."
        )
        image_form.addRow(self.chk_dark_on_light)

        image_group.setLayout(image_form)
        params_layout.addWidget(image_group)

        # === ORIENTATION TRACKING ===
        orientation_group = QGroupBox("Orientation Tracking")
        orientation_form = QFormLayout()

        # Velocity threshold for orientation logic switching
        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.0, 100.0)
        self.spin_velocity.setSingleStep(0.1)
        self.spin_velocity.setValue(2.0)
        self.spin_velocity.setToolTip(
            "Velocity threshold (pixels/frame) for switching orientation tracking modes. Below this: smooth orientation changes. Above: align with motion direction."
        )
        orientation_form.addRow("Velocity Threshold:", self.spin_velocity)

        # Enable instant flip correction for fast-moving objects
        self.chk_instant_flip = QCheckBox("Instant Flip Orientation")
        self.chk_instant_flip.setChecked(True)
        self.chk_instant_flip.setToolTip(
            "For fast-moving animals, instantly flip orientation to align with motion direction. Prevents 180° orientation errors."
        )
        orientation_form.addRow(self.chk_instant_flip)

        # Maximum orientation change for stationary objects (degrees)
        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1.0, 180.0)
        self.spin_max_orient.setSingleStep(1.0)
        self.spin_max_orient.setValue(30.0)
        self.spin_max_orient.setToolTip(
            "Maximum orientation change per frame (degrees) for slow-moving animals. Prevents jittery orientation tracking."
        )
        orientation_form.addRow("Max Orient Δ (deg):", self.spin_max_orient)

        orientation_group.setLayout(orientation_form)
        params_layout.addWidget(orientation_group)

        # === TRACK LIFECYCLE ===
        lifecycle_group = QGroupBox("Track Lifecycle")
        lifecycle_form = QFormLayout()

        # Frames before declaring track lost
        self.spin_lost_thresh = QSpinBox()
        self.spin_lost_thresh.setRange(1, 1000)
        self.spin_lost_thresh.setValue(10)
        self.spin_lost_thresh.setToolTip(
            "Number of consecutive missed frames before declaring a track 'lost' and available for reassignment."
        )
        lifecycle_form.addRow("Lost Threshold (frames):", self.spin_lost_thresh)

        # Minimum distance for track respawning to prevent duplicate IDs
        self.spin_min_respawn_distance = QSpinBox()
        self.spin_min_respawn_distance.setRange(0, 500)
        self.spin_min_respawn_distance.setValue(50)
        self.spin_min_respawn_distance.setToolTip(
            "Minimum distance (pixels) between new detections and existing tracks for respawning. Prevents duplicate tracking."
        )
        lifecycle_form.addRow("Min Respawn Distance:", self.spin_min_respawn_distance)

        lifecycle_group.setLayout(lifecycle_form)
        params_layout.addWidget(lifecycle_group)

        # === COST FUNCTION WEIGHTS ===
        weights_group = QGroupBox("Cost Function Weights")
        weights_form = QFormLayout()

        # Weight for position cost in assignment
        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setRange(0.0, 10.0)
        self.spin_Wp.setSingleStep(0.1)
        self.spin_Wp.setValue(1.0)
        self.spin_Wp.setToolTip(
            "Weight for position cost in track assignment. Higher values prioritize position accuracy over other factors."
        )
        weights_form.addRow("Position Weight:", self.spin_Wp)

        # Weight for orientation cost in assignment
        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setRange(0.0, 10.0)
        self.spin_Wo.setSingleStep(0.1)
        self.spin_Wo.setValue(1.0)
        self.spin_Wo.setToolTip(
            "Weight for orientation cost in track assignment. Higher values prioritize orientation consistency."
        )
        weights_form.addRow("Orientation Weight:", self.spin_Wo)

        # Weight for area cost in assignment
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setRange(0.0, 1.0)
        self.spin_Wa.setSingleStep(0.0005)
        self.spin_Wa.setValue(0.001)
        self.spin_Wa.setToolTip(
            "Weight for area cost in track assignment. Higher values enforce size consistency between frames."
        )
        weights_form.addRow("Area Weight:", self.spin_Wa)

        # Weight for aspect ratio cost in assignment
        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setRange(0.0, 10.0)
        self.spin_Wasp.setSingleStep(0.1)
        self.spin_Wasp.setValue(0.1)
        self.spin_Wasp.setToolTip(
            "Weight for aspect ratio cost in track assignment. Higher values enforce shape consistency."
        )
        weights_form.addRow("Aspect Weight:", self.spin_Wasp)

        weights_group.setLayout(weights_form)
        params_layout.addWidget(weights_group)

        # === ALGORITHM OPTIONS ===
        algorithm_group = QGroupBox("Algorithm Options")
        algorithm_form = QFormLayout()

        # Use Mahalanobis distance vs Euclidean for position cost
        self.chk_use_mahal = QCheckBox("Use Mahalanobis Distance")
        self.chk_use_mahal.setChecked(True)
        self.chk_use_mahal.setToolTip(
            "Use Mahalanobis distance (considers prediction uncertainty) vs simple Euclidean distance for position cost calculation."
        )
        algorithm_form.addRow(self.chk_use_mahal)

        algorithm_group.setLayout(algorithm_form)
        params_layout.addWidget(algorithm_group)

        # === VISUALIZATION OPTIONS ===
        viz_group = QGroupBox("Visualization")
        viz_form = QFormLayout()

        # Show foreground mask overlay
        self.chk_show_fg = QCheckBox("Show FG Mask")
        self.chk_show_fg.setChecked(True)
        self.chk_show_fg.setToolTip(
            "Display the foreground mask showing detected animal pixels after background subtraction and morphological processing."
        )
        viz_form.addRow(self.chk_show_fg)

        # Show background model overlay
        self.chk_show_bg = QCheckBox("Show Background")
        self.chk_show_bg.setChecked(True)
        self.chk_show_bg.setToolTip(
            "Display the learned background model used for foreground detection. Helps verify background quality and detect lighting changes."
        )
        viz_form.addRow(self.chk_show_bg)

        # === TRACKING VISUALIZATION ELEMENTS ===
        # Show circle markers at animal positions
        self.chk_show_circles = QCheckBox("Show Circle Markers")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip(
            "Draw colored circle markers at each animal's current position. Each trajectory has a unique color for easy identification."
        )
        viz_form.addRow(self.chk_show_circles)

        # Show orientation lines
        self.chk_show_orientation = QCheckBox("Show Orientation Lines")
        self.chk_show_orientation.setChecked(True)
        self.chk_show_orientation.setToolTip(
            "Draw lines indicating animal orientation/heading direction. Useful for analyzing movement patterns and directional behavior."
        )
        viz_form.addRow(self.chk_show_orientation)

        # Show trajectory trails
        self.chk_show_trajectories = QCheckBox("Show Trajectory Trails")
        self.chk_show_trajectories.setChecked(True)
        self.chk_show_trajectories.setToolTip(
            "Draw colored trail lines showing recent path history for each animal. Trail length determined by trajectory history buffer."
        )
        viz_form.addRow(self.chk_show_trajectories)

        # Show track ID and continuity info
        self.chk_show_labels = QCheckBox("Show Trajectory ID & Continuity Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip(
            "Show persistent trajectory ID (T#) that maintains identity across track losses and tracking continuity count (C:#)"
        )
        viz_form.addRow(self.chk_show_labels)

        # Show track state (active/occluded)
        self.chk_show_state = QCheckBox("Show Track State")
        self.chk_show_state.setChecked(True)
        self.chk_show_state.setToolTip(
            "Display tracking state information (Active/Lost/Predicted) for debugging track lifecycle and loss detection."
        )
        viz_form.addRow(self.chk_show_state)

        # Enable debug logging
        self.chk_debug_logging = QCheckBox("Enable Debug Logging")
        self.chk_debug_logging.setChecked(False)
        self.chk_debug_logging.stateChanged.connect(self.toggle_debug_logging)
        viz_form.addRow(self.chk_debug_logging)

        # Enable backward tracking
        self.chk_enable_backward = QCheckBox("Enable Backward Tracking")
        self.chk_enable_backward.setChecked(True)
        self.chk_enable_backward.setToolTip(
            "Automatically run backward tracking after forward tracking completes (requires FFmpeg)"
        )
        viz_form.addRow(self.chk_enable_backward)

        viz_group.setLayout(viz_form)
        params_layout.addWidget(viz_group)

        # Set up scrollable parameters area
        params_scroll.setWidget(params_widget)
        params_scroll.setWidgetResizable(True)
        params_scroll.setMinimumWidth(350)
        control.addWidget(params_scroll)

        # === DISPLAY CONTROLS ===
        display_group = QGroupBox("Display Controls")
        display_form = QFormLayout()

        # Display zoom factor
        self.spin_zoom = QDoubleSpinBox()
        self.spin_zoom.setRange(0.1, 5.0)
        self.spin_zoom.setSingleStep(0.1)
        self.spin_zoom.setValue(1.0)
        display_form.addRow("Zoom Factor:", self.spin_zoom)

        display_group.setLayout(display_form)
        control.addWidget(display_group)

        # === ROI CONTROLS ===
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QVBoxLayout()

        # ROI selection buttons
        roi_btn_layout = QHBoxLayout()
        self.btn_start_roi = QPushButton("Start ROI Selection")
        self.btn_start_roi.clicked.connect(self.start_roi_selection)
        self.btn_start_roi.setToolTip(
            "Begin interactive circular ROI selection. Click and drag on the video display to define tracking region."
        )
        self.btn_finish_roi = QPushButton("Finish ROI")
        self.btn_finish_roi.clicked.connect(self.finish_roi_selection)
        self.btn_finish_roi.setEnabled(False)
        self.btn_finish_roi.setToolTip(
            "Complete ROI selection and apply the defined circular region for tracking analysis."
        )
        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setToolTip(
            "Remove current ROI selection and revert to full-frame tracking."
        )

        roi_btn_layout.addWidget(self.btn_start_roi)
        roi_btn_layout.addWidget(self.btn_finish_roi)
        roi_btn_layout.addWidget(self.btn_clear_roi)
        roi_layout.addLayout(roi_btn_layout)

        # ROI status display
        self.roi_status_label = QLabel("No ROI selected")
        self.roi_status_label.setStyleSheet(
            "font-weight: bold; color: #666; padding: 5px; border: 1px solid #ccc; background: #f9f9f9;"
        )
        roi_layout.addWidget(self.roi_status_label)

        # ROI instructions
        self.roi_instructions = QLabel(
            "Click 'Start ROI Selection', then click points on the circle boundary in the video. A preview circle will appear after 3+ points."
        )
        self.roi_instructions.setWordWrap(True)
        self.roi_instructions.setStyleSheet("color: #888; font-size: 10px;")
        roi_layout.addWidget(self.roi_instructions)

        roi_group.setLayout(roi_layout)
        control.addWidget(roi_group)

        # === ACTION BUTTONS ===
        # Preview mode: real-time tracking without data export
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(lambda ch: self.toggle_preview(ch))
        self.btn_preview.setToolTip(
            "Toggle real-time preview mode for parameter tuning. Shows tracking without saving data to CSV."
        )

        # Full tracking mode: complete tracking with CSV export
        self.btn_start = QPushButton("Full Tracking")
        self.btn_start.clicked.connect(self.start_full)
        self.btn_start.setToolTip(
            "Start complete tracking analysis with CSV data export. Processes entire video and saves trajectory data."
        )

        # Emergency stop button
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_tracking)
        self.btn_stop.setToolTip(
            "Stop current tracking operation immediately. Use this to abort processing if needed."
        )

        # Progress bars for tracking operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        # Progress label to show current operation and frame info
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)  # Hidden by default
        self.progress_label.setStyleSheet("color: #666; font-size: 10px;")

        control.addWidget(self.btn_preview)
        control.addWidget(self.btn_start)
        control.addWidget(self.btn_stop)
        control.addWidget(self.progress_label)
        control.addWidget(self.progress_bar)
        control.addStretch(1)  # Push buttons to top

        # === MAIN LAYOUT ASSEMBLY ===
        main_l = QHBoxLayout()
        main_l.addWidget(self.scroll, stretch=1)  # Video display (expandable)
        main_l.addLayout(control, stretch=0)  # Control panel (fixed width)

        central_widget = QWidget()
        central_widget.setLayout(main_l)
        self.setCentralWidget(central_widget)

        # === KEYBOARD SHORTCUTS ===
        # Add keyboard shortcuts for common actions
        self.btn_start_roi.setShortcut("Ctrl+R")
        self.btn_start_roi.setToolTip("Start ROI Selection (Ctrl+R)")
        self.btn_finish_roi.setShortcut("Ctrl+F")
        self.btn_finish_roi.setToolTip("Finish ROI Selection (Ctrl+F)")
        self.btn_clear_roi.setShortcut("Ctrl+C")
        self.btn_clear_roi.setToolTip("Clear ROI (Ctrl+C)")

        # === APPLICATION STATE ===
        self.tracking_worker = None  # Background tracking thread
        self.csv_writer_thread = None  # Background CSV writer thread
        self.final_full_trajs = []  # Complete trajectory data after tracking

        # Add reversal worker for backward tracking
        self.reversal_worker = None

        # Load saved configuration if available
        self.load_config()
        self._connect_parameter_signals()

    def select_file(self):
        """
        Handle video file selection.

        Opens file dialog for video selection and updates the file path display.
        Users can then separately define ROI using the ROI controls.
        """
        # Open file dialog for video selection
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if not fp:
            return

        self.file_line.setText(fp)

        # Clear any existing ROI selection
        if self.roi_selection_active:
            self.clear_roi()

        # Update display
        self.video_label.clear()
        self.video_label.setText(
            "Video loaded. Click 'Start ROI Selection' to define tracking area."
        )

        logger.info(f"Video selected: {fp}")

    def record_roi_click(self, evt):
        """
        Handle mouse clicks for circular ROI definition with live preview.

        This method collects multiple points on the circle circumference and
        fits a circle in real-time, showing a preview of the current best fit.
        Users can click multiple points for better accuracy.

        Args:
            evt: Mouse click event with position information
        """
        # Only process clicks during ROI selection mode
        if not self.roi_selection_active or self.roi_base_frame is None:
            return

        # Extract click coordinates
        x = evt.pos().x()
        y = evt.pos().y()

        # Add point to collection
        self.roi_points.append((x, y))

        # Update live preview
        self.update_roi_preview()

    def update_roi_preview(self):
        """
        Update the live preview of the ROI circle based on current points.

        Fits a circle to the current set of points and displays the result
        with visual feedback about the quality of the fit.
        """
        if self.roi_base_frame is None:
            return

        # Start with the base frame
        pix = QPixmap.fromImage(self.roi_base_frame).toImage().copy()
        painter = QPainter(pix)

        # Draw all clicked points
        painter.setPen(QPen(Qt.red, 6))
        for i, (px, py) in enumerate(self.roi_points):
            painter.drawPoint(px, py)
            # Number the points for clarity with better visibility
            painter.setPen(QPen(Qt.black, 3))  # Black outline
            painter.drawText(px + 12, py - 12, str(i + 1))
            painter.setPen(QPen(Qt.white, 2))  # White text
            painter.drawText(px + 10, py - 10, str(i + 1))
            painter.setPen(QPen(Qt.red, 6))  # Reset for next point

        # Try to fit circle if we have enough points
        if len(self.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self.roi_points)
            if circle_fit:
                cx, cy, radius = circle_fit
                self.roi_fitted_circle = circle_fit

                # Draw preview circle
                painter.setPen(QPen(Qt.green, 3))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )

                # Draw center point
                painter.setPen(QPen(Qt.blue, 8))
                painter.drawPoint(int(cx), int(cy))

                # Show circle info
                info_text = f"Center: ({cx:.1f}, {cy:.1f}), Radius: {radius:.1f}"
                painter.setPen(QPen(Qt.white, 2))
                painter.drawText(10, 30, info_text)

                # Update status
                self.roi_status_label.setText(
                    f"Preview: {len(self.roi_points)} points, radius {radius:.1f}px"
                )
                self.btn_finish_roi.setEnabled(True)
            else:
                self.roi_status_label.setText(
                    f"Invalid fit with {len(self.roi_points)} points"
                )
                self.btn_finish_roi.setEnabled(False)
        else:
            # Not enough points yet
            self.roi_status_label.setText(
                f"Need {3 - len(self.roi_points)} more points (minimum 3)"
            )
            self.btn_finish_roi.setEnabled(False)

        painter.end()
        self.video_label.setPixmap(QPixmap.fromImage(pix))

    def start_roi_selection(self):
        """
        Start the ROI selection process.

        Loads the first frame of the video for ROI definition and
        enables click handling for point collection.
        """
        if not self.file_line.text():
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        # Load first frame for ROI selection
        cap = cv2.VideoCapture(self.file_line.text())
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open video file.")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            QMessageBox.warning(self, "Error", "Cannot read video frame.")
            return

        # Convert frame for display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Store frame and initialize ROI selection
        self.roi_base_frame = qt_image
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = True

        # Update UI state
        self.btn_start_roi.setEnabled(False)
        self.btn_finish_roi.setEnabled(False)
        self.roi_status_label.setText("Click points on circle boundary...")
        self.roi_instructions.setText(
            "Click points around the circular boundary. Preview circle appears after 3 points. Press Escape to cancel, or click 'Finish ROI' when satisfied."
        )

        # Display frame
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def finish_roi_selection(self):
        """
        Complete the ROI selection using the current fitted circle.

        Creates the binary mask from the fitted circle and exits ROI selection mode.
        """
        if not self.roi_fitted_circle:
            QMessageBox.warning(self, "No ROI", "No valid circle fit available.")
            return

        cx, cy, radius = self.roi_fitted_circle

        # Create binary circular mask
        if self.roi_base_frame:
            fh, fw = self.roi_base_frame.height(), self.roi_base_frame.width()
            mask = np.zeros((fh, fw), np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(radius), 255, -1)  # Filled circle
            self.roi_mask = mask

            # Exit ROI selection mode
            self.roi_selection_active = False
            self.roi_base_frame = None

            # Update UI state
            self.btn_start_roi.setEnabled(True)
            self.btn_finish_roi.setEnabled(False)
            self.roi_status_label.setText(
                f"ROI set: center ({cx:.1f}, {cy:.1f}), radius {radius:.1f}px"
            )
            self.roi_instructions.setText(
                "ROI successfully defined. You can now start tracking."
            )

            # Clear video display
            self.video_label.clear()
            self.video_label.setText("ROI defined. Ready for tracking.")

            logger.info(
                f"ROI defined: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
            )

    def clear_roi(self):
        """
        Clear the current ROI selection and reset to no ROI state.
        """
        self.roi_mask = None
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = False
        self.roi_base_frame = None

        # Update UI state
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        self.roi_status_label.setText("No ROI selected")
        self.roi_instructions.setText(
            "Click 'Start ROI Selection', then click points on the circle boundary in the video. A preview circle will appear after 3+ points."
        )

        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Click 'Start ROI Selection' to define tracking area.")

        logger.info("ROI cleared")

    def keyPressEvent(self, event):
        """
        Handle keyboard shortcuts for ROI selection.

        Args:
            event: Keyboard event
        """
        if event.key() == Qt.Key_Escape and self.roi_selection_active:
            # Cancel ROI selection with Escape key
            self.clear_roi()
        else:
            super().keyPressEvent(event)

    def select_csv(self):
        """Open file dialog to select CSV output file for data export."""
        fp, _ = QFileDialog.getSaveFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if fp:
            self.csv_line.setText(fp)

    def select_video_output(self):
        """Open file dialog to select video output file for tracking visualization."""
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self.video_out_line.setText(fp)

    def on_detection_method_changed(self, index):
        """
        Handle detection method selection change.

        Shows/hides YOLO parameters based on selected method.
        Also updates visibility of background subtraction parameters.
        """
        is_yolo = index == 1  # 0 = Background Subtraction, 1 = YOLO OBB
        self.yolo_group.setVisible(is_yolo)

        # Optionally show/hide background subtraction specific params
        # (Currently all tracking params are shared)

    def on_yolo_model_changed(self, index):
        """
        Handle YOLO model selection change.

        Shows custom model path input if 'Custom Model' is selected.
        """
        is_custom = self.combo_yolo_model.currentText() == "Custom Model..."
        self.yolo_custom_model_widget.setVisible(is_custom)

    def select_yolo_custom_model(self):
        """Open file dialog to select custom YOLO model file."""
        # Start from home directory or current custom model path if set
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
        """
        Toggle the display of the real-time histogram window.

        Creates and shows the histogram window if it doesn't exist,
        or shows/hides it if it does exist.
        """
        if self.histogram_window is None:
            # Create histogram panel if it doesn't exist
            if self.histogram_panel is None:
                self.histogram_panel = HistogramPanel(
                    history_frames=self.spin_histogram_history.value()
                )

            # Create new histogram window
            self.histogram_window = QMainWindow()
            self.histogram_window.setWindowTitle("Real-Time Parameter Histograms")
            self.histogram_window.setCentralWidget(self.histogram_panel)
            self.histogram_window.resize(900, 700)

            # Apply dark theme to histogram window
            self.histogram_window.setStyleSheet(self.styleSheet())

            # Handle window close event
            def on_close():
                self.btn_show_histograms.setChecked(False)
                self.histogram_window.hide()

            self.histogram_window.closeEvent = lambda event: (
                on_close(),
                event.accept(),
            )

        # Toggle visibility
        if self.btn_show_histograms.isChecked():
            self.histogram_window.show()
            self.histogram_window.raise_()
            self.histogram_window.activateWindow()
        else:
            self.histogram_window.hide()

    def load_config(self):
        """
        Load tracking parameters from JSON configuration file.

        Restores all GUI control values from saved configuration,
        allowing users to resume work with previous settings.
        Falls back gracefully if config file doesn't exist or is corrupted.
        """
        if not os.path.isfile(CONFIG_FILENAME):
            return

        try:
            with open(CONFIG_FILENAME, "r") as f:
                cfg = json.load(f)

            # Restore file paths
            self.file_line.setText(cfg.get("file_path", ""))
            self.csv_line.setText(cfg.get("csv_path", ""))

            # Restore detection method and YOLO parameters
            detection_method = cfg.get("detection_method", "background_subtraction")
            self.combo_detection_method.setCurrentIndex(
                0 if detection_method == "background_subtraction" else 1
            )

            # Restore YOLO model selection
            yolo_model_path = cfg.get("yolo_model_path", "yolov8s-obb.pt")
            # Try to match against predefined models
            model_matched = False
            for i in range(
                self.combo_yolo_model.count() - 1
            ):  # -1 to exclude "Custom Model..."
                if self.combo_yolo_model.itemText(i).startswith(yolo_model_path):
                    self.combo_yolo_model.setCurrentIndex(i)
                    model_matched = True
                    break

            # If no match, it's a custom model
            if not model_matched:
                self.combo_yolo_model.setCurrentIndex(
                    self.combo_yolo_model.count() - 1
                )  # "Custom Model..."
                self.yolo_custom_model_line.setText(yolo_model_path)

            self.spin_yolo_confidence.setValue(
                cfg.get("yolo_confidence_threshold", 0.25)
            )
            self.spin_yolo_iou.setValue(cfg.get("yolo_iou_threshold", 0.7))

            # Restore YOLO target classes
            yolo_classes = cfg.get("yolo_target_classes", None)
            if yolo_classes is not None and isinstance(yolo_classes, list):
                self.line_yolo_classes.setText(",".join(map(str, yolo_classes)))
            else:
                self.line_yolo_classes.setText("")

            # Core tracking parameters
            self.spin_max_targets.setValue(cfg.get("max_targets", 4))
            self.spin_threshold.setValue(cfg.get("threshold_value", 50))
            self.spin_morph_size.setValue(cfg.get("morph_kernel_size", 5))
            self.spin_min_contour.setValue(cfg.get("min_contour_area", 50))

            # Restore size filtering parameters
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

            # Restore background model parameters
            self.spin_bg_prime.setValue(cfg.get("bg_prime_frames", 10))

            # Restore lighting stabilization parameters
            self.chk_lighting_stab.setChecked(cfg.get("lighting_stabilization", True))
            self.chk_adaptive_bg.setChecked(cfg.get("adaptive_background", True))
            self.spin_bg_learning.setValue(cfg.get("background_learning_rate", 0.001))
            self.spin_lighting_smooth.setValue(cfg.get("lighting_smooth_factor", 0.95))
            self.spin_lighting_median.setValue(cfg.get("lighting_median_window", 5))

            # Restore Kalman filter parameters
            self.spin_kalman_noise.setValue(cfg.get("kalman_noise", 0.03))
            self.spin_kalman_meas.setValue(cfg.get("kalman_meas_noise", 0.1))

            # Restore performance and conservative splitting parameters
            self.spin_resize.setValue(cfg.get("resize_factor", 1.0))
            self.chk_conservative_split.setChecked(
                cfg.get("enable_conservative_split", True)
            )
            self.spin_merge_threshold.setValue(cfg.get("merge_area_threshold", 1000))
            self.spin_conservative_kernel.setValue(
                cfg.get("conservative_kernel_size", 3)
            )
            self.spin_conservative_erode.setValue(cfg.get("conservative_erode_iter", 1))

            # Restore histogram parameters
            self.enable_histograms.setChecked(cfg.get("enable_histograms", False))
            self.spin_histogram_history.setValue(
                cfg.get("histogram_history_frames", 300)
            )

            # Restore additional dilation parameters for thin animals
            self.chk_additional_dilation.setChecked(
                cfg.get("enable_additional_dilation", False)
            )
            self.spin_dilation_iterations.setValue(cfg.get("dilation_iterations", 2))
            self.spin_dilation_kernel_size.setValue(cfg.get("dilation_kernel_size", 3))

            # Restore image enhancement parameters
            self.spin_brightness.setValue(cfg.get("brightness", 0.0))
            self.spin_contrast.setValue(cfg.get("contrast", 1.0))
            self.spin_gamma.setValue(cfg.get("gamma", 1.0))
            self.chk_dark_on_light.setChecked(cfg.get("dark_on_light_background", True))

            # Restore orientation tracking parameters
            self.spin_velocity.setValue(cfg.get("velocity_threshold", 2.0))
            self.chk_instant_flip.setChecked(cfg.get("instant_flip", True))
            self.spin_max_orient.setValue(cfg.get("max_orient_delta_stopped", 30.0))
            self.spin_lost_thresh.setValue(cfg.get("lost_threshold_frames", 10))
            self.spin_min_respawn_distance.setValue(cfg.get("min_respawn_distance", 50))

            # Restore cost function weights
            self.spin_Wp.setValue(cfg.get("W_POSITION", 1.0))
            self.spin_Wo.setValue(cfg.get("W_ORIENTATION", 1.0))
            self.spin_Wa.setValue(cfg.get("W_AREA", 0.001))
            self.spin_Wasp.setValue(cfg.get("W_ASPECT", 0.1))

            # Restore algorithm options
            self.chk_use_mahal.setChecked(cfg.get("USE_MAHALANOBIS", True))

            # Restore visualization options
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
            self.spin_zoom.setValue(cfg.get("zoom_factor", 1.0))

            logger.info("Configuration loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")

    def save_config(self):
        """
        Save current tracking parameters to JSON configuration file.

        Captures all GUI control values and saves them for future sessions,
        enabling parameter persistence across application restarts.
        """
        cfg = {
            # File paths
            "file_path": self.file_line.text(),
            "csv_path": self.csv_line.text(),
            # Detection method and YOLO parameters
            "detection_method": (
                "background_subtraction"
                if self.combo_detection_method.currentIndex() == 0
                else "yolo_obb"
            ),
            "yolo_model_path": (
                self.yolo_custom_model_line.text()
                if self.combo_yolo_model.currentText() == "Custom Model..."
                else self.combo_yolo_model.currentText().split(" ")[0]
            ),
            "yolo_confidence_threshold": self.spin_yolo_confidence.value(),
            "yolo_iou_threshold": self.spin_yolo_iou.value(),
            "yolo_target_classes": (
                [int(x.strip()) for x in self.line_yolo_classes.text().split(",")]
                if self.line_yolo_classes.text().strip()
                else None
            ),
            # Core tracking parameters
            "max_targets": self.spin_max_targets.value(),
            "threshold_value": self.spin_threshold.value(),
            "morph_kernel_size": self.spin_morph_size.value(),
            "min_contour_area": self.spin_min_contour.value(),
            # Size filtering parameters
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
            # Background model parameters
            "bg_prime_frames": self.spin_bg_prime.value(),
            # Lighting stabilization parameters
            "lighting_stabilization": self.chk_lighting_stab.isChecked(),
            "adaptive_background": self.chk_adaptive_bg.isChecked(),
            "background_learning_rate": self.spin_bg_learning.value(),
            "lighting_smooth_factor": self.spin_lighting_smooth.value(),
            "lighting_median_window": self.spin_lighting_median.value(),
            # Kalman filter parameters
            "kalman_noise": self.spin_kalman_noise.value(),
            "kalman_meas_noise": self.spin_kalman_meas.value(),
            # Performance and conservative splitting parameters
            "resize_factor": self.spin_resize.value(),
            "enable_conservative_split": self.chk_conservative_split.isChecked(),
            "merge_area_threshold": self.spin_merge_threshold.value(),
            "conservative_kernel_size": self.spin_conservative_kernel.value(),
            "conservative_erode_iter": self.spin_conservative_erode.value(),
            # Additional dilation parameters for thin animals
            "enable_additional_dilation": self.chk_additional_dilation.isChecked(),
            "dilation_iterations": self.spin_dilation_iterations.value(),
            "dilation_kernel_size": self.spin_dilation_kernel_size.value(),
            # Image enhancement parameters
            "brightness": self.spin_brightness.value(),
            "contrast": self.spin_contrast.value(),
            "gamma": self.spin_gamma.value(),
            "dark_on_light_background": self.chk_dark_on_light.isChecked(),
            # Orientation tracking parameters
            "velocity_threshold": self.spin_velocity.value(),
            "instant_flip": self.chk_instant_flip.isChecked(),
            "max_orient_delta_stopped": self.spin_max_orient.value(),
            "lost_threshold_frames": self.spin_lost_thresh.value(),
            "min_respawn_distance": self.spin_min_respawn_distance.value(),
            # Cost function weights
            "W_POSITION": self.spin_Wp.value(),
            "W_ORIENTATION": self.spin_Wo.value(),
            "W_AREA": self.spin_Wa.value(),
            "W_ASPECT": self.spin_Wasp.value(),
            # Algorithm options
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            # Visualization options
            "show_fg": self.chk_show_fg.isChecked(),
            "show_bg": self.chk_show_bg.isChecked(),
            "show_circles": self.chk_show_circles.isChecked(),
            "show_orientation": self.chk_show_orientation.isChecked(),
            "show_trajectories": self.chk_show_trajectories.isChecked(),
            "show_labels": self.chk_show_labels.isChecked(),
            "show_state": self.chk_show_state.isChecked(),
            "debug_logging": self.chk_debug_logging.isChecked(),
            "enable_backward_tracking": self.chk_enable_backward.isChecked(),
            "zoom_factor": self.spin_zoom.value(),
            # Histogram parameters
            "enable_histograms": self.enable_histograms.isChecked(),
            "histogram_history_frames": self.spin_histogram_history.value(),
        }

        try:
            with open(CONFIG_FILENAME, "w") as f:
                json.dump(cfg, f, indent=2)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save configuration: {e}")

    def get_parameters_dict(self):
        """
        Convert GUI control values to parameter dictionary for tracking engine.

        Collects all current GUI values and formats them into the dictionary
        structure expected by the TrackingWorker thread. Also generates
        random colors for trajectory visualization.

        Returns:
            dict: Complete parameter set for tracking algorithm
        """
        N = self.spin_max_targets.value()

        # Generate random colors for trajectory visualization
        np.random.seed(42)  # Consistent colors across runs
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]

        # Get detection method and YOLO parameters
        detection_method = (
            "background_subtraction"
            if self.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

        # Get YOLO model path
        yolo_model_path = "yolo26s-obb.pt"  # Default
        if self.combo_yolo_model.currentText() == "Custom Model...":
            yolo_model_path = self.yolo_custom_model_line.text() or "yolo26s-obb.pt"
        else:
            # Extract model name from combo box text (format: "model.pt (description)")
            model_text = self.combo_yolo_model.currentText()
            yolo_model_path = model_text.split(" ")[0]

        # Parse YOLO target classes
        yolo_target_classes = None
        if self.line_yolo_classes.text().strip():
            try:
                yolo_target_classes = [
                    int(x.strip()) for x in self.line_yolo_classes.text().split(",")
                ]
            except ValueError:
                logger.warning(
                    "Invalid YOLO target classes format. Using None (all classes)."
                )
                yolo_target_classes = None

        return {
            # Detection method selection
            "DETECTION_METHOD": detection_method,
            # YOLO-specific parameters
            "YOLO_MODEL_PATH": yolo_model_path,
            "YOLO_CONFIDENCE_THRESHOLD": float(self.spin_yolo_confidence.value()),
            "YOLO_IOU_THRESHOLD": float(self.spin_yolo_iou.value()),
            "YOLO_TARGET_CLASSES": yolo_target_classes,
            # Core tracking parameters
            "MAX_TARGETS": N,
            "THRESHOLD_VALUE": self.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
            # Size filtering parameters
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
            # Lighting stabilization parameters
            "ENABLE_LIGHTING_STABILIZATION": self.chk_lighting_stab.isChecked(),
            "ENABLE_ADAPTIVE_BACKGROUND": self.chk_adaptive_bg.isChecked(),
            "BACKGROUND_LEARNING_RATE": float(self.spin_bg_learning.value()),
            "LIGHTING_SMOOTH_FACTOR": float(self.spin_lighting_smooth.value()),
            "LIGHTING_MEDIAN_WINDOW": self.spin_lighting_median.value(),
            # Kalman filter parameters
            "KALMAN_NOISE_COVARIANCE": float(self.spin_kalman_noise.value()),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": float(self.spin_kalman_meas.value()),
            # Performance parameters
            "RESIZE_FACTOR": float(self.spin_resize.value()),
            # Conservative splitting parameters
            "ENABLE_CONSERVATIVE_SPLIT": self.chk_conservative_split.isChecked(),
            "MERGE_AREA_THRESHOLD": self.spin_merge_threshold.value(),
            "CONSERVATIVE_KERNEL_SIZE": self.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self.spin_conservative_erode.value(),
            # Additional dilation parameters for thin animals
            "ENABLE_ADDITIONAL_DILATION": self.chk_additional_dilation.isChecked(),
            "DILATION_ITERATIONS": self.spin_dilation_iterations.value(),
            "DILATION_KERNEL_SIZE": self.spin_dilation_kernel_size.value(),
            # Image enhancement parameters
            "BRIGHTNESS": float(self.spin_brightness.value()),
            "CONTRAST": float(self.spin_contrast.value()),
            "GAMMA": float(self.spin_gamma.value()),
            "DARK_ON_LIGHT_BACKGROUND": self.chk_dark_on_light.isChecked(),
            # Orientation tracking parameters
            "VELOCITY_THRESHOLD": float(self.spin_velocity.value()),
            "INSTANT_FLIP_ORIENTATION": self.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": float(self.spin_max_orient.value()),
            "LOST_THRESHOLD_FRAMES": self.spin_lost_thresh.value(),
            # Cost function weights
            "W_POSITION": float(self.spin_Wp.value()),
            "W_ORIENTATION": float(self.spin_Wo.value()),
            "W_AREA": float(self.spin_Wa.value()),
            "W_ASPECT": float(self.spin_Wasp.value()),
            # Algorithm options
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            # Visualization parameters
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self.chk_show_orientation.isChecked(),
            "SHOW_TRAJECTORIES": self.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self.chk_show_labels.isChecked(),
            "SHOW_STATE": self.chk_show_state.isChecked(),
            "zoom_factor": self.spin_zoom.value(),
            # Real-time histogram parameters
            "ENABLE_HISTOGRAMS": self.enable_histograms.isChecked(),
            "HISTOGRAM_HISTORY_FRAMES": self.spin_histogram_history.value(),
            # ROI mask (if defined)
            "ROI_MASK": self.roi_mask,
        }

    def toggle_preview(self, checked):
        """
        Handle preview mode toggle button.

        Preview mode allows real-time parameter tuning without data export.

        Args:
            checked (bool): True if preview mode activated, False if deactivated
        """
        if checked:
            self.start_tracking(preview_mode=True)
            self.btn_preview.setText("Stop Preview")
        else:
            self.stop_tracking()
            self.btn_preview.setText("Preview")

    def toggle_debug_logging(self, checked):
        """
        Toggle debug logging level based on checkbox state.

        Args:
            checked (bool): True to enable debug logging, False for info level
        """
        if checked:
            # setup_logging(log_level=logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            # setup_logging(log_level=logging.INFO)
            logger.info("Debug logging disabled")

    def start_full(self):
        """
        Start full tracking mode with data export.

        Stops any active preview mode and initiates complete tracking
        with CSV data export enabled.
        """
        # Stop preview mode if active
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")
        self.start_tracking(preview_mode=False)

    def start_tracking(self, preview_mode: bool, backward_mode: bool = False):
        """
        Initialize and start tracking worker thread.

        Sets up CSV writer (if not preview mode), creates tracking worker,
        and starts background processing with current parameters.

        Args:
            preview_mode (bool): If True, skip CSV export; if False, enable full logging
            backward_mode (bool): If True, process video from end to start
        """
        # Save current parameters
        self.save_config()

        # Validate video file selection
        video_fp = self.file_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No video", "Please select a video file first.")
            return

        # For forward tracking or preview mode, use original video
        if preview_mode:
            # Preview mode - no CSV export, use original video
            self.start_preview_on_video(video_fp)
        else:
            # Full tracking mode on original video
            self.start_tracking_on_video(video_fp, backward_mode=False)

    def start_preview_on_video(self, video_path):
        """
        Start preview mode on a specific video file.

        Args:
            video_path (str): Path to video file to preview
        """
        # Check for existing tracking session
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Tracking already running.")
            return

        # No CSV writer for preview mode
        self.csv_writer_thread = None

        # Create and configure tracking worker for preview (no output files)
        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=False,
        )
        self.tracking_worker.set_parameters(self.get_parameters_dict())

        # Connect signals for real-time updates
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)

        # Show progress bar and label
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preview Mode: Initializing...")

        # Disable UI controls during tracking
        self._set_ui_controls_enabled(False)

        # Start background tracking
        self.tracking_worker.start()

    def stop_tracking(self):
        """Stop any active tracking session gracefully."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            self.tracking_worker.stop()
            # Hide progress bar when manually stopped
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

        # Re-enable UI controls
        self._set_ui_controls_enabled(True)

    def _set_ui_controls_enabled(self, enabled: bool):
        """
        Enable or disable all UI controls to prevent changes during tracking.

        Args:
            enabled (bool): True to enable controls, False to disable
        """
        # File selection controls
        self.btn_file.setEnabled(enabled)
        self.btn_csv.setEnabled(enabled)
        self.btn_video_out.setEnabled(enabled)

        # ROI controls
        self.btn_start_roi.setEnabled(enabled)
        self.btn_finish_roi.setEnabled(enabled and self.roi_selection_active)
        self.btn_clear_roi.setEnabled(enabled)

        # Action buttons (except stop)
        self.btn_preview.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        # Stop button should always be enabled when tracking is running
        self.btn_stop.setEnabled(not enabled)

        # Parameter controls - disable all tracking parameters
        self.spin_max_targets.setEnabled(enabled)
        self.spin_threshold.setEnabled(enabled)
        self.spin_max_dist.setEnabled(enabled)
        self.spin_continuity_thresh.setEnabled(enabled)

        # Morphological processing
        self.spin_morph_size.setEnabled(enabled)
        self.spin_min_contour.setEnabled(enabled)
        self.chk_size_filtering.setEnabled(enabled)
        self.spin_min_object_size.setEnabled(enabled)
        self.spin_max_object_size.setEnabled(enabled)
        self.spin_max_contour_multiplier.setEnabled(enabled)
        self.chk_conservative_split.setEnabled(enabled)
        self.spin_conservative_kernel.setEnabled(enabled)
        self.spin_conservative_erode.setEnabled(enabled)
        self.spin_merge_threshold.setEnabled(enabled)
        self.chk_additional_dilation.setEnabled(enabled)
        self.spin_dilation_iterations.setEnabled(enabled)
        self.spin_dilation_kernel_size.setEnabled(enabled)

        # System stability
        self.spin_min_detect.setEnabled(enabled)
        self.spin_min_detections_to_start.setEnabled(enabled)
        self.spin_min_track.setEnabled(enabled)
        self.spin_traj_hist.setEnabled(enabled)

        # Background model
        self.spin_bg_prime.setEnabled(enabled)
        self.chk_adaptive_bg.setEnabled(enabled)
        self.spin_bg_learning.setEnabled(enabled)

        # Lighting stabilization
        self.chk_lighting_stab.setEnabled(enabled)
        self.spin_lighting_smooth.setEnabled(enabled)
        self.spin_lighting_median.setEnabled(enabled)

        # Kalman filter
        self.spin_kalman_noise.setEnabled(enabled)
        self.spin_kalman_meas.setEnabled(enabled)

        # Performance
        self.spin_resize.setEnabled(enabled)

        # Image enhancement
        self.spin_brightness.setEnabled(enabled)
        self.spin_contrast.setEnabled(enabled)
        self.spin_gamma.setEnabled(enabled)
        self.chk_dark_on_light.setEnabled(enabled)

        # Orientation tracking
        self.spin_velocity.setEnabled(enabled)
        self.chk_instant_flip.setEnabled(enabled)
        self.spin_max_orient.setEnabled(enabled)

        # Track lifecycle
        self.spin_lost_thresh.setEnabled(enabled)
        self.spin_min_respawn_distance.setEnabled(enabled)

        # Cost function weights
        self.spin_Wp.setEnabled(enabled)
        self.spin_Wo.setEnabled(enabled)
        self.spin_Wa.setEnabled(enabled)
        self.spin_Wasp.setEnabled(enabled)

        # Algorithm options
        self.chk_use_mahal.setEnabled(enabled)

        # Visualization options - disable all during tracking
        self.chk_show_fg.setEnabled(enabled)
        self.chk_show_bg.setEnabled(enabled)
        self.chk_show_circles.setEnabled(enabled)
        self.chk_show_orientation.setEnabled(enabled)
        self.chk_show_trajectories.setEnabled(enabled)
        self.chk_show_labels.setEnabled(enabled)
        self.chk_show_state.setEnabled(enabled)
        self.chk_debug_logging.setEnabled(enabled)
        self.chk_enable_backward.setEnabled(enabled)

        # Display controls (zoom can always be changed during tracking)
        self.spin_zoom.setEnabled(True)  # Always allow zoom changes

        # Post-processing parameters
        if hasattr(self, "enable_postprocessing"):
            self.enable_postprocessing.setEnabled(enabled)
        if hasattr(self, "spin_min_trajectory_length"):
            self.spin_min_trajectory_length.setEnabled(enabled)
        if hasattr(self, "spin_max_velocity_break"):
            self.spin_max_velocity_break.setEnabled(enabled)
        if hasattr(self, "spin_max_distance_break"):
            self.spin_max_distance_break.setEnabled(enabled)

        # Histogram controls
        if hasattr(self, "enable_histograms"):
            self.enable_histograms.setEnabled(enabled)
        if hasattr(self, "spin_histogram_history"):
            self.spin_histogram_history.setEnabled(enabled)
        if hasattr(self, "btn_show_histograms"):
            self.btn_show_histograms.setEnabled(enabled)

    @Slot(int, str)
    def on_progress_update(self, percentage, status_text):
        """
        Handle progress updates from tracking worker.

        Args:
            percentage (int): Progress percentage (0-100)
            status_text (str): Status description text
        """
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status_text)

    @Slot(np.ndarray)
    def on_new_frame(self, rgb):
        """
        Handle new frame from tracking worker for display.

        Applies zoom factor and updates video display with processed frame.

        Args:
            rgb (np.ndarray): RGB frame from tracking worker
        """
        # Apply user-specified zoom factor
        z = max(self.spin_zoom.value(), 0.1)
        h, w, _ = rgb.shape

        # Convert to Qt image format and apply zoom
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        scaled = qimg.scaled(
            int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

    def save_trajectories_to_csv(self, trajectories, output_path):
        """
        Saves a list of cleaned trajectories to a CSV file.

        Args:
            trajectories (list): A list of trajectory segments. Each segment is a list of points.
            output_path (str): The path to the output CSV file.
        """
        if not trajectories:
            logger.warning("No post-processed trajectories to save.")
            return

        # Define the header for the clean output file
        header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]

        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

                # Assign a new, unique ID to each cleaned trajectory segment
                # as post-processing can split one raw track into multiple clean ones.
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
        """
        Use this method to merge forward and backward trajectories to find consensus trajectories.
        """
        forward_trajs = getattr(self, "forward_processed_trajs", None)
        backward_trajs = getattr(self, "backward_processed_trajs", None)
        if not forward_trajs or not backward_trajs:
            QMessageBox.warning(
                self,
                "No Trajectories",
                "No forward or backward trajectories available to merge.",
            )
            return

        # get the number of frames in the video using the video file path
        video_fp = self.file_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            QMessageBox.warning(self, "Error", "Cannot open video file.")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Merge forward and backward trajectories with current parameters
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
        """
        Handle tracking completion and cleanup.

        Manages CSV writer shutdown, UI state updates, and result presentation.

        Args:
            finished_normally (bool): True if tracking completed successfully
            fps_list (list): Frame processing rates over time
            full_traj (list): Complete trajectory data for all tracks
        """
        # Hide progress bar and label
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        # Shutdown CSV writer if active
        if self.csv_writer_thread:
            self.csv_writer_thread.stop()
            self.csv_writer_thread.join()
            logger.info("CSV writer stopped.")

        # Reset preview button state
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview")

        if finished_normally and not self.btn_preview.isChecked():
            logger.info("Tracking completed successfully.")
            # Get tracking context
            is_backward_mode = (
                hasattr(self.tracking_worker, "backward_mode")
                and self.tracking_worker.backward_mode
            )
            is_backward_enabled = self.chk_enable_backward.isChecked()

            # Always run post-processing if enabled
            processed_trajectories = full_traj  # Default to raw data
            if self.enable_postprocessing.isChecked():
                logger.info("Running final post-processing on trajectories...")
                params = self.get_parameters_dict()
                processed_trajectories, stats = process_trajectories(full_traj, params)
                logger.info(f"Post-processing complete. Stats: {stats}")

                # Now, decide what to do with the processed data
                if not is_backward_mode:
                    # This was the FORWARD pass.
                    # Save its processed data to a file.
                    raw_csv_path = self.csv_line.text()
                    if raw_csv_path:
                        base, ext = os.path.splitext(raw_csv_path)
                        processed_csv_path = f"{base}_forward_processed{ext}"
                        self.save_trajectories_to_csv(
                            processed_trajectories, processed_csv_path
                        )

                    # If backward tracking is next, store the data and start the next pass.
                    if is_backward_enabled:
                        self.forward_processed_trajs = processed_trajectories
                        logger.info(
                            "Forward tracking complete, starting backward tracking..."
                        )
                        self.start_backward_tracking()
                    else:
                        # Not doing backward tracking, so we are done.
                        self._set_ui_controls_enabled(True)  # Re-enable UI controls
                        QMessageBox.information(self, "Done", "Tracking complete.")
                        self.plot_fps(fps_list)
                else:
                    # This was the BACKWARD pass.
                    # We now have both self.forward_processed_trajs and the new processed_trajectories.

                    # Save the processed backward data for inspection.
                    raw_csv_path = self.csv_line.text()
                    if raw_csv_path:
                        base, ext = os.path.splitext(raw_csv_path)
                        processed_csv_path = f"{base}_backward_processed{ext}"
                        self.save_trajectories_to_csv(
                            processed_trajectories, processed_csv_path
                        )
                    self.backward_processed_trajs = processed_trajectories

                    # Now, MERGE the results.
                    if self.forward_processed_trajs and self.backward_processed_trajs:
                        logger.info("Merging forward and backward trajectories...")
                        self.merge_and_save_trajectories()

                    self._set_ui_controls_enabled(True)  # Re-enable UI controls
                    QMessageBox.information(
                        self, "Done", "Backward tracking and merging complete."
                    )
                    self.plot_fps(fps_list)
        else:
            logger.warning("Tracking did not finish normally.")
            # Tracking didn't finish normally (error, interruption, etc.)
            self._set_ui_controls_enabled(True)  # Re-enable UI controls
            if not finished_normally:
                QMessageBox.warning(
                    self,
                    "Tracking Interrupted",
                    "Tracking was stopped or encountered an error.",
                )

        gc.collect()

    @Slot(dict)
    def on_histogram_data(self, histogram_data):
        """
        Handle real-time histogram data updates from tracking worker.

        Updates the histogram panel with new parameter data if histograms
        are enabled and the histogram window is open.

        Args:
            histogram_data (dict): Dictionary containing parameter arrays for histograms
                - 'velocities': List of velocity values (px/frame)
                - 'sizes': List of object size values (px²)
                - 'orientations': List of orientation values (radians)
                - 'assignment_costs': List of assignment cost values
        """
        # Only update histograms if enabled and window is open
        if (
            self.enable_histograms.isChecked()
            and self.histogram_window is not None
            and self.histogram_window.isVisible()
        ):

            # Update histogram history window size if changed
            current_history = self.spin_histogram_history.value()
            if self.histogram_panel.history_frames != current_history:
                self.histogram_panel.set_history_frames(current_history)

            # Update each histogram with new data
            if "velocities" in histogram_data and histogram_data["velocities"]:
                self.histogram_panel.update_velocity_data(histogram_data["velocities"])

            if "sizes" in histogram_data and histogram_data["sizes"]:
                self.histogram_panel.update_size_data(histogram_data["sizes"])

            if "orientations" in histogram_data and histogram_data["orientations"]:
                self.histogram_panel.update_orientation_data(
                    histogram_data["orientations"]
                )

            if (
                "assignment_costs" in histogram_data
                and histogram_data["assignment_costs"]
            ):
                self.histogram_panel.update_assignment_cost_data(
                    histogram_data["assignment_costs"]
                )

    def start_backward_tracking(self):
        """
        Start backward tracking by creating a reversed video and tracking it.

        This method:
        1. Creates a reversed video using FFmpeg
        2. Runs normal tracking on the reversed video
        3. Outputs results with _backward suffix
        """
        # Get the original video path
        video_fp = self.file_line.text()
        if not video_fp:
            QMessageBox.warning(
                self, "No video", "No video file available for backward tracking."
            )
            return

        # Create paths for reversed video
        base_name, ext = os.path.splitext(video_fp)
        reversed_video_path = f"{base_name}_reversed{ext}"

        # Show progress for video reversal
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Set to indeterminate mode
        self.progress_label.setText(
            "Creating reversed video with FFmpeg (this may take a while)..."
        )

        # Process events to update UI
        QApplication.processEvents()

        # Create reversed video using FFmpeg
        # Use the non-blocking worker
        self.reversal_worker = VideoReversalWorker(video_fp, reversed_video_path)
        self.reversal_worker.finished.connect(self.on_reversal_finished)
        self.reversal_worker.start()

    @Slot(bool, str, str)
    def on_reversal_finished(self, success, output_path, error_message):
        """Handle completion of the video reversal worker."""
        self.progress_bar.setRange(0, 100)  # Reset progress bar

        if not success:
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            QMessageBox.critical(self, "Error Creating Reversed Video", error_message)
            return

        self.progress_label.setText(
            "Reversed video created. Starting backward tracking..."
        )
        QApplication.processEvents()

        # Now run normal tracking on the reversed video
        self.start_tracking_on_video(output_path, backward_mode=True)

    def _connect_parameter_signals(self):
        """Connect all parameter widgets to the update handler."""
        # PySide2's findChildren doesn't accept a tuple of types, so we call it
        # for each type individually and combine the resulting lists.
        widgets_to_connect = (
            self.findChildren(QSpinBox)
            + self.findChildren(QDoubleSpinBox)
            + self.findChildren(QCheckBox)
        )

        for widget in widgets_to_connect:
            # For SpinBoxes, the signal is 'valueChanged'
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_parameter_changed)
            # For CheckBoxes, the signal is 'stateChanged'
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._on_parameter_changed)

    @Slot()
    def _on_parameter_changed(self):
        """
        Slot to handle any parameter change. Emits a signal with all current params.
        Also updates the worker if it's running.
        """
        params = self.get_parameters_dict()
        self.parameters_changed.emit(params)

    def start_tracking_on_video(self, video_path, backward_mode=False):
        """
        Start tracking on a specific video file.

        Args:
            video_path (str): Path to video file to track
            backward_mode (bool): Whether this is backward tracking mode
        """
        # Check for existing tracking session
        if self.tracking_worker and self.tracking_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Tracking already running.")
            return

        # Initialize CSV writer for backward tracking
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

            # Modify CSV path for backward mode
            csv_path = self.csv_line.text()
            if backward_mode:
                base, ext = os.path.splitext(csv_path)
                csv_path = f"{base}_backward{ext}"

            self.csv_writer_thread = CSVWriterThread(csv_path, header=hdr)
            self.csv_writer_thread.start()
            logger.info(f"CSV writer started -> {csv_path}")

        # Get video output path if specified
        video_output_path = None
        if self.video_out_line.text():
            video_output_path = self.video_out_line.text()

        # Create and configure tracking worker
        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=self.csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
        )
        self.tracking_worker.set_parameters(self.get_parameters_dict())

        # Connect the new signal/slot for live updates
        self.parameters_changed.connect(self.tracking_worker.update_parameters)

        # Connect signals for real-time updates
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)

        # Show progress bar and label
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        mode_text = "Backward Tracking" if backward_mode else "Forward Tracking"
        self.progress_label.setText(f"{mode_text}: Initializing...")

        # Disable UI controls during tracking
        self._set_ui_controls_enabled(False)

        # Start background tracking
        self.tracking_worker.start()

    def plot_fps(self, fps_list):
        """
        Display FPS performance plot using matplotlib.

        Shows frame processing rate over time to help assess
        tracking performance and identify bottlenecks.

        Args:
            fps_list (list): Frame processing rates over time
        """
        if len(fps_list) < 2:
            QMessageBox.information(self, "FPS Plot", "Not enough data.")
            return

        plt.figure()
        plt.plot(fps_list)
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.title("Tracking FPS Over Time")
        plt.show()
