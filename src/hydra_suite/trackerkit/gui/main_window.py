#!/usr/bin/env python3
"""
Main application window for the HYDRA.

Refactored for improved UX with Tabbed interface and logical grouping.
"""

import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionGroupBox,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.core.identity.dataset.generator import IndividualDatasetGenerator
from hydra_suite.core.identity.dataset.oriented_video import (
    OrientedTrackVideoExporter,
    resolve_individual_dataset_dir,
)
from hydra_suite.core.identity.pose.quality import (
    apply_quality_to_dataframe,
    apply_temporal_pose_postprocessing,
    calibrate_body_length_prior,
    calibrate_edge_length_priors,
)
from hydra_suite.core.identity.properties.export import (
    POSE_SUMMARY_COLUMNS,
    augment_trajectories_with_pose_cache,
    build_pose_keypoint_labels,
    flatten_pose_keypoints_row,
    merge_interpolated_pose_df,
    pose_wide_columns_for_labels,
)
from hydra_suite.core.post.processing import (
    interpolate_trajectories,
    process_trajectories,
    relink_trajectories_with_pose,
    resolve_trajectories,
)
from hydra_suite.core.tracking.optimizer_workers import DetectionCacheBuilderWorker
from hydra_suite.core.tracking.worker import TrackingWorker
from hydra_suite.data.csv_writer import CSVWriterThread
from hydra_suite.data.detection_cache import DetectionCache
from hydra_suite.runtime.compute_runtime import (
    CANONICAL_RUNTIMES,
    allowed_runtimes_for_pipelines,
    derive_detection_runtime_settings,
    derive_pose_runtime_settings,
    infer_compute_runtime_from_legacy,
    runtime_label,
)
from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.utils.geometry import fit_circle_to_points, wrap_angle_degs
from hydra_suite.utils.pose_visualization import (
    is_renderable_pose_keypoint,
    normalize_pose_render_min_conf,
)
from hydra_suite.utils.video_artifacts import (
    build_detection_cache_path,
    build_optimizer_detection_cache_path,
    build_tracking_session_log_path,
    candidate_artifact_base_dirs,
    choose_writable_artifact_base_dir,
    find_existing_detection_cache_path,
    iter_detection_cache_candidates,
)
from hydra_suite.widgets.workers import BaseWorker

from .workers.merge_worker import MergeWorker, _write_csv_artifact, _write_roi_npz
from .workers.crops_worker import InterpolatedCropsWorker
from .workers.video_worker import OrientedTrackVideoWorker
from .workers.dataset_worker import DatasetGenerationWorker
from .workers.preview_worker import (
    PreviewDetectionWorker,
    _build_preview_background_model,
    _clear_preview_background_cache,
    _run_preview_detection_job,
)
from .dialogs.bg_parameter_helper import BgParameterHelperDialog
from .dialogs.parameter_helper import ParameterHelperDialog
from .widgets.collapsible import AccordionContainer, CollapsibleGroupBox
from .widgets.help_label import CompactHelpLabel
from .widgets.stacked_page import CurrentPageStackedWidget
from .widgets.tooltip_button import ImmediateTooltipButton

try:
    from hydra_suite.posekit.gui.dialogs.utils import get_available_devices
except ImportError:

    def get_available_devices():
        return ["auto", "cpu", "cuda", "mps"]


# Configuration file for saving/loading tracking parameters
CONFIG_FILENAME = "tracking_config.json"  # Fallback for manual load/save


def get_video_config_path(video_path: object) -> object:
    """Get the config file path for a given video file."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


def get_models_directory() -> object:
    """
    Get the path to the default YOLO OBB model repository.

    Returns models/obb (direct OBB models).
    Creates the directory if it doesn't exist.
    """
    return get_yolo_model_repository_directory(
        task_family="obb", usage_role="obb_direct"
    )


def get_models_root_directory() -> str:
    """Return user-local models/ root and create it when missing."""
    from hydra_suite.paths import get_models_dir

    return str(get_models_dir())


def get_yolo_model_repository_directory(
    task_family: str | None = None, usage_role: str | None = None
) -> object:
    """Return repository directory for a YOLO model role."""
    tf = str(task_family or "").strip().lower()
    ur = str(usage_role or "").strip().lower()
    models_root = get_models_root_directory()

    if ur == "seq_detect" or tf == "detect":
        repo_dir = os.path.join(models_root, "detection")
    elif ur == "seq_crop_obb":
        repo_dir = os.path.join(models_root, "obb", "cropped")
    elif ur == "headtail":
        # Parent dir; YOLO/ and tiny/ subdirs are resolved at import time.
        repo_dir = os.path.join(models_root, "classification", "orientation")
    elif ur == "colortag" or (tf == "classify" and ur not in ("headtail",)):
        # Parent dir; YOLO/ and tiny/ subdirs are resolved at import time.
        repo_dir = os.path.join(models_root, "classification", "colortag")
    else:
        repo_dir = os.path.join(models_root, "obb")

    os.makedirs(repo_dir, exist_ok=True)
    return repo_dir


def get_pose_models_directory(backend: str | None = None) -> object:
    """
    Get the local pose-model repository directory.

    Layout:
      models/pose/YOLO/
      models/pose/SLEAP/
      models/pose/ViTPose/
    """
    models_root = get_models_root_directory()
    pose_root = os.path.join(models_root, "pose")
    os.makedirs(pose_root, exist_ok=True)
    if not backend:
        return pose_root
    key = str(backend or "").strip().lower()
    if key == "sleap":
        backend_dirname = "SLEAP"
    elif key == "vitpose":
        backend_dirname = "ViTPose"
    else:
        backend_dirname = "YOLO"
    backend_dir = os.path.join(pose_root, backend_dirname)
    os.makedirs(backend_dir, exist_ok=True)
    return backend_dir


def resolve_pose_model_path(model_path: object, backend: str | None = None) -> object:
    """Resolve a pose model path (relative or absolute) to an absolute path when possible."""
    if not model_path:
        return model_path

    path_str = str(model_path).strip()
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    models_root = get_models_root_directory()
    candidates = [os.path.join(models_root, path_str)]
    if backend:
        candidates.append(os.path.join(get_pose_models_directory(backend), path_str))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return path_str


def make_pose_model_path_relative(model_path: object) -> object:
    """Convert absolute pose-model paths under models/ into relative paths."""
    if not model_path or not os.path.isabs(str(model_path)):
        return model_path
    models_root = get_models_root_directory()
    try:
        rel_path = os.path.relpath(str(model_path), models_root)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass
    return model_path


def resolve_model_path(model_path: object) -> object:
    """
    Resolve a model path to an absolute path.

    If the path is relative, look for it in the models directory.
    If absolute and exists, return as-is.

    Args:
        model_path: Relative or absolute model path

    Returns:
        Absolute path to the model file, or original path if not found
    """
    if not model_path:
        return model_path

    path_str = str(model_path).strip()

    # If already absolute and exists, return it
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    models_root = get_models_root_directory()
    candidate = os.path.join(models_root, path_str)
    if os.path.exists(candidate):
        return candidate

    # If relative path doesn't exist in models dir, try as-is
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Return original if nothing works (will fail later with clear error)
    return model_path


def make_model_path_relative(model_path: object) -> object:
    """
    Convert an absolute model path to relative if it's in the models directory.

    This allows presets to be portable across devices.

    Args:
        model_path: Absolute or relative model path

    Returns:
        Relative path if model is in archive, otherwise absolute path
    """
    if not model_path or not os.path.isabs(model_path):
        return model_path

    models_root = get_models_root_directory()

    # Store path relative to models/ root (e.g. obb/model.pt, detection/model.pt).
    try:
        rel_path = os.path.relpath(model_path, models_root)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass

    # Return absolute path if not in models directory
    return model_path


def get_yolo_model_registry_path() -> object:
    """Return path to the local YOLO model metadata registry JSON."""
    return os.path.join(get_models_root_directory(), "model_registry.json")


def _sanitize_model_token(text: object) -> object:
    """Sanitize a species/info token for filenames and metadata."""
    raw = str(text or "").strip()
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    return cleaned.strip("_")


def _normalize_yolo_model_metadata(metadata: object) -> object:
    """Normalize legacy model metadata to species + model_info schema."""
    if not isinstance(metadata, dict):
        return {}

    normalized = dict(metadata)
    species = _sanitize_model_token(normalized.get("species", ""))
    model_info = _sanitize_model_token(normalized.get("model_info", ""))

    if species:
        normalized["species"] = species
    if model_info:
        normalized["model_info"] = model_info

    task_family = _sanitize_model_token(normalized.get("task_family", "")).lower()
    usage_role = _sanitize_model_token(normalized.get("usage_role", "")).lower()
    if task_family:
        normalized["task_family"] = task_family
    else:
        normalized.pop("task_family", None)
    if usage_role:
        normalized["usage_role"] = usage_role
    else:
        normalized.pop("usage_role", None)
    return normalized


def load_yolo_model_registry() -> object:
    """Load YOLO model metadata registry (path -> metadata)."""
    registry_path = get_yolo_model_registry_path()
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): _normalize_yolo_model_metadata(v) for k, v in data.items()}
    except Exception as e:
        logger.warning(f"Failed to load YOLO model registry: {e}")
        return {}


def save_yolo_model_registry(registry: object) -> object:
    """Persist YOLO model metadata registry JSON."""
    registry_path = get_yolo_model_registry_path()
    try:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save YOLO model registry: {e}")


def get_yolo_model_metadata(model_path: object) -> object:
    """Get metadata for a model path if registered."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    return _normalize_yolo_model_metadata(registry.get(rel_path, {}))


def register_yolo_model(model_path: object, metadata: object) -> object:
    """Register/overwrite metadata entry for a model path."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    registry[rel_path] = _normalize_yolo_model_metadata(metadata)
    save_yolo_model_registry(registry)


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window providing GUI interface for HYDRA configuration, model management, and execution.
    """

    parameters_changed = Signal(dict)

    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        self.config = TrackerConfig()
        self.setWindowTitle("HYDRA")
        self.resize(1360, 850)

        # Apply consistent VSCode dark theme (matches PoseKit and ClassKit)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }

            /* Tabs */
            QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; background-color: #1e1e1e; }
            QTabBar::tab {
                background: #2d2d30; color: #cccccc; padding: 6px 10px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                border: 1px solid #3e3e42; border-bottom: none;
            }
            QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; font-weight: 600; border-bottom: 2px solid #007acc; }
            QTabBar::tab:hover:!selected { background: #37373d; }

            /* Group boxes */
            QGroupBox {
                font-weight: 600; border: 1px solid #3e3e42; border-radius: 6px;
                margin-top: 8px; padding: 6px; background-color: #252526;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top left;
                left: 8px; padding: 1px 6px;
                background-color: #1e1e1e; color: #9cdcfe; border-radius: 3px;
            }

            /* Buttons */
            QPushButton {
                background-color: #0e639c; border: none; color: #ffffff;
                padding: 5px 12px; border-radius: 4px; min-height: 22px; font-weight: 500;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5a8f; }
            QPushButton:checked { background-color: #094771; border: 1px solid #007acc; }
            QPushButton:disabled { background-color: #3e3e42; color: #777777; border: none; }

            /* Action / Stop buttons */
            QPushButton#ActionBtn {
                background-color: #0e639c; font-weight: bold; font-size: 11px;
            }
            QPushButton#ActionBtn:hover { background-color: #1177bb; }
            QPushButton#StopBtn { background-color: #d9534f; font-weight: bold; }
            QPushButton#StopBtn:hover { background-color: #c9302c; }
            QPushButton#SecondaryBtn {
                background-color: #3a3a3a; color: #d6d6d6;
                font-size: 10px; font-weight: 500; min-height: 20px;
                padding: 4px 10px;
            }
            QPushButton#SecondaryBtn:hover { background-color: #4a4a4a; }

            /* Inputs */
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #3c3c3c; border: 1px solid #3e3e42; border-radius: 4px;
                padding: 4px 8px; color: #e0e0e0;
                selection-background-color: #094771; min-width: 100px; min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, QComboBox:hover {
                border-color: #0e639c;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus {
                border-color: #007acc;
            }

            /* ComboBox dropdown */
            QComboBox::drop-down {
                subcontrol-origin: padding; subcontrol-position: top right;
                width: 20px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a;
                border-top-right-radius: 4px; border-bottom-right-radius: 4px;
            }
            QComboBox::drop-down:hover { background-color: #5a5a5a; }
            QComboBox QAbstractItemView {
                background-color: #252526; border: 1px solid #3e3e42;
                selection-background-color: #094771; selection-color: #ffffff;
                color: #e0e0e0; outline: none;
            }
            QComboBox QAbstractItemView::item { padding: 6px 10px; min-height: 22px; }
            QComboBox QAbstractItemView::item:hover { background-color: #2a2d2e; }
            QComboBox QAbstractItemView::item:selected { background-color: #094771; color: #ffffff; }

            /* SpinBox arrows */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                width: 18px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a; border-top-right-radius: 4px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover { background-color: #0e639c; }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 18px; border-left: 1px solid #3e3e42;
                background-color: #4a4a4a; border-bottom-right-radius: 4px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #0e639c; }

            /* Checkboxes and Radio buttons */
            QCheckBox, QRadioButton { color: #cccccc; spacing: 8px; }
            QCheckBox::indicator, QRadioButton::indicator {
                width: 14px; height: 14px; border: 1px solid #3e3e42;
                border-radius: 3px; background-color: #3c3c3c;
            }
            QRadioButton::indicator { border-radius: 7px; }
            QCheckBox::indicator:checked, QRadioButton::indicator:checked {
                background-color: #0e639c; border-color: #007acc;
            }
            QCheckBox::indicator:hover, QRadioButton::indicator:hover { border-color: #007acc; }

            /* Labels */
            QLabel { color: #cccccc; background-color: transparent; }

            /* Toolbar */
            QToolBar {
                background-color: #252526; border-bottom: 1px solid #3e3e42;
                spacing: 6px; padding: 4px 6px;
            }
            QToolButton {
                background-color: transparent; border: none; border-radius: 4px;
                padding: 6px 10px; color: #cccccc;
            }
            QToolButton:hover { background-color: #2a2d2e; }
            QToolButton:pressed, QToolButton:checked { background-color: #094771; color: #4fc1ff; }

            /* Status bar */
            QStatusBar {
                background-color: #007acc; color: #ffffff;
                border-top: 1px solid #0098ff; font-weight: 500; font-size: 11px;
            }
            QStatusBar QLabel { background-color: transparent; color: #ffffff; padding: 0px 4px; }

            /* Menu */
            QMenuBar {
                background-color: #252526; color: #cccccc;
                border-bottom: 1px solid #3e3e42; padding: 2px;
            }
            QMenuBar::item { padding: 5px 10px; background-color: transparent; border-radius: 3px; }
            QMenuBar::item:selected { background-color: #2a2d2e; }
            QMenu {
                background-color: #252526; color: #cccccc;
                border: 1px solid #3e3e42; border-radius: 4px; padding: 4px;
            }
            QMenu::item { padding: 6px 20px 6px 12px; border-radius: 3px; }
            QMenu::item:selected { background-color: #094771; color: #ffffff; }
            QMenu::separator { height: 1px; background-color: #3e3e42; margin: 4px 8px; }

            /* Splitter */
            QSplitter::handle { background-color: #3e3e42; }
            QSplitter::handle:hover { background-color: #007acc; }

            /* Scrollbars */
            QScrollBar:vertical { background-color: #252526; width: 10px; border-radius: 5px; margin: 0px; }
            QScrollBar::handle:vertical { background-color: #5a5a5a; border-radius: 5px; min-height: 24px; }
            QScrollBar::handle:vertical:hover { background-color: #007acc; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal { background-color: #252526; height: 10px; border-radius: 5px; margin: 0px; }
            QScrollBar::handle:horizontal { background-color: #5a5a5a; border-radius: 5px; min-width: 24px; }
            QScrollBar::handle:horizontal:hover { background-color: #007acc; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

            /* Progress bar */
            QProgressBar {
                border: 1px solid #3e3e42; border-radius: 4px;
                text-align: center; background-color: #252526; color: #cccccc; font-size: 11px;
            }
            QProgressBar::chunk { background-color: #0e639c; border-radius: 3px; }

            /* Lists and Tables */
            QListWidget, QTableWidget {
                background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; outline: none;
            }
            QListWidget::item, QTableWidget::item { padding: 4px 8px; }
            QListWidget::item:selected, QTableWidget::item:selected { background-color: #094771; color: #ffffff; }
            QListWidget::item:hover:!selected, QTableWidget::item:hover:!selected { background-color: #2a2d2e; }
            QHeaderView::section {
                background-color: #2d2d30; color: #cccccc;
                border: none; border-right: 1px solid #3e3e42; border-bottom: 1px solid #3e3e42;
                padding: 4px 8px; font-weight: 600;
            }

            /* Text edits */
            QPlainTextEdit, QTextEdit {
                background-color: #252526; color: #e0e0e0;
                border: 1px solid #3e3e42; border-radius: 4px; padding: 4px;
            }
            QPlainTextEdit:focus, QTextEdit:focus { border-color: #007acc; }

            /* Sliders */
            QSlider::groove:horizontal {
                border: 1px solid #3e3e42; height: 4px;
                background-color: #3c3c3c; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #007acc; border: none;
                width: 14px; height: 14px; border-radius: 7px; margin: -5px 0;
            }
            QSlider::handle:horizontal:hover { background-color: #1177bb; }
            QSlider::sub-page:horizontal { background-color: #0e639c; border-radius: 2px; }
            """)

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

        self.tracking_worker = None
        self.merge_worker = None
        self.csv_writer_thread = None
        self.dataset_worker = None
        self.interp_worker = None
        self.oriented_video_worker = None
        self.preview_detection_worker = None
        self.temporary_files = []  # Track temporary files for cleanup
        self.session_log_handler = None  # Track current session log file handler
        self._individual_dataset_run_id = None
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None
        self.current_interpolated_roi_npz_path = None
        self.current_interpolated_pose_csv_path = None
        self.current_interpolated_pose_df = None
        self.current_interpolated_tag_csv_path = None
        self.current_interpolated_tag_df = None
        self.current_interpolated_cnn_csv_paths = {}
        self.current_interpolated_cnn_dfs = {}
        self.current_interpolated_headtail_csv_path = None
        self.current_interpolated_headtail_df = None
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False
        self._pending_finish_after_track_videos = False

        # Preview frame for live image adjustments
        self.preview_frame_original = None  # Original frame without adjustments
        self.detection_test_result = None  # Store detection test result
        self.current_video_path = None
        self.detected_sizes = None  # Store detected object sizes for statistics
        self.batch_videos = []  # List of video paths for batch mode
        self.current_batch_index = -1

        # ROI display caching (for performance)
        self._roi_masked_cache = {}  # Cache: {(frame_id, roi_hash): masked_image}
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

        # UI interaction state
        self._video_interactions_enabled = True
        self._saved_widget_enabled_states = {}
        self._pending_finish_after_interp = False
        self._stop_all_requested = False

        # Per-session summary state (reset at the start of each forward tracking run)
        self._session_result_dataset = None
        self._dataset_was_started = False
        self._show_summary_on_dataset_done = False
        self._session_wall_start = None
        self._session_final_csv_path = None
        self._session_fps_list = []
        self._session_frames_processed = 0
        self._ui_settings = self._load_ui_settings()
        self._ui_state_save_timer = QTimer(self)
        self._ui_state_save_timer.setSingleShot(True)
        self._ui_state_save_timer.setInterval(250)
        self._ui_state_save_timer.timeout.connect(self._save_ui_settings)
        self._collapsible_state_widgets = {}

        # Advanced configuration (for power users)
        self.advanced_config = self._load_advanced_config()

        # Video player state
        self.video_cap = None  # cv2.VideoCapture for video playback
        self.video_total_frames = 0
        self.video_current_frame_idx = 0
        self.last_read_frame_idx = (
            -1
        )  # Track last frame read for sequential optimization
        self.is_playing = False
        self.playback_timer = None  # QTimer for playback

        # === UI CONSTRUCTION ===
        self.init_ui()

        # === POST-INIT ===
        # Disable wheel events on all spinboxes to prevent accidental value changes
        self._disable_spinbox_wheel_events()

        # Config is now loaded automatically when a video is selected
        # instead of at startup
        self._connect_parameter_signals()

        # Cache preview-related controls for UI state transitions
        self._preview_controls = self._collect_preview_controls()

        # Restore persisted layout preferences after widgets are fully built.
        QTimer.singleShot(0, self._restore_ui_state)

        # Welcome page is visible at startup — workspace UI state is applied
        # when the user loads a video and _show_workspace() is called.

    def _get_ui_settings_path(self) -> Path:
        """Return the HYDRA UI settings path used for persistent layout preferences."""
        config_dir = Path.home() / ".hydra-suite"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "ui_settings.json"

    def _load_ui_settings(self) -> dict:
        """Load persistent HYDRA UI settings."""
        path = self._get_ui_settings_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _queue_ui_state_save(self) -> None:
        """Debounce HYDRA UI settings writes while the user resizes or switches tabs."""
        if hasattr(self, "_ui_state_save_timer"):
            self._ui_state_save_timer.start()

    def _remember_collapsible_state(
        self, key: str, collapsible: CollapsibleGroupBox
    ) -> None:
        """Restore and track expanded state for a collapsible section."""
        self._collapsible_state_widgets[key] = collapsible
        saved = self._ui_settings.get("collapsed_sections", {}).get(key)
        if isinstance(saved, bool):
            collapsible.setExpanded(saved)
        collapsible.toggled.connect(
            lambda _expanded, _key=key: self._queue_ui_state_save()
        )

    def _restore_ui_state(self) -> None:
        """Apply persisted HYDRA UI layout preferences after construction."""
        settings = self._ui_settings or {}

        detection_index = settings.get("detection_method_index")
        if isinstance(detection_index, int) and hasattr(self, "_detection_panel"):
            self._detection_panel.combo_detection_method.setCurrentIndex(
                max(
                    0,
                    min(
                        detection_index,
                        self._detection_panel.combo_detection_method.count() - 1,
                    ),
                )
            )

        tab_index = settings.get("active_tab_index")
        if isinstance(tab_index, int) and hasattr(self, "tabs"):
            tab_index = max(0, min(tab_index, self.tabs.count() - 1))
            if self.tabs.isTabEnabled(tab_index):
                self.tabs.setCurrentIndex(tab_index)

        splitter_sizes = settings.get("splitter_sizes")
        if (
            isinstance(splitter_sizes, list)
            and len(splitter_sizes) == 2
            and all(isinstance(size, int) and size > 0 for size in splitter_sizes)
            and hasattr(self, "splitter")
        ):
            self.splitter.setSizes(splitter_sizes)

    def _save_ui_settings(self) -> None:
        """Persist HYDRA UI layout preferences without touching tracking configs."""
        if not hasattr(self, "tabs") or not hasattr(self, "splitter"):
            return

        collapsed_sections = {
            key: widget.isExpanded()
            for key, widget in self._collapsible_state_widgets.items()
        }
        settings = {
            "active_tab_index": int(self.tabs.currentIndex()),
            "splitter_sizes": [int(size) for size in self.splitter.sizes()],
            "detection_method_index": (
                int(self._detection_panel.combo_detection_method.currentIndex())
                if hasattr(self, "_detection_panel")
                else 0
            ),
            "collapsed_sections": collapsed_sections,
        }

        path = self._get_ui_settings_path()
        try:
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            self._ui_settings = settings
        except Exception:
            logger.debug("Failed to save HYDRA UI settings", exc_info=True)

    def _set_compact_scroll_layout(self, layout: QLayout) -> None:
        """Prevent scroll-area content layouts from stretching sparse sections vertically."""
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

    def _set_compact_section_widget(self, widget: QWidget) -> None:
        """Make top-level section widgets hug their content vertically."""
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def _make_setup_divider(self) -> QFrame:
        """Create a subtle divider line used inside the Get Started tab."""
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Plain)
        divider.setStyleSheet("color: #343434; background-color: #343434;")
        divider.setFixedHeight(1)
        return divider

    def init_ui(self: object) -> object:
        """Build the structured UI using Splitter and Tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.main_stack = QStackedWidget()
        root_layout.addWidget(self.main_stack)

        self._welcome_page_index = self.main_stack.addWidget(self._make_welcome_page())

        workspace_page = QWidget()
        self._workspace_page_index = self.main_stack.addWidget(workspace_page)
        self.main_stack.setCurrentIndex(self._welcome_page_index)

        # Main Layout is a horizontal splitter (Video Left | Controls Right)
        main_layout = QHBoxLayout(workspace_page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(6)

        # --- LEFT PANEL: Video & ROI ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        # Video Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #121212; border: none;")
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #6a6a6a; font-size: 16px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll.setWidget(self.video_label)
        self._show_video_logo_placeholder()

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
        roi_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        roi_main_layout = QVBoxLayout(roi_frame)
        roi_main_layout.setContentsMargins(8, 4, 8, 4)
        roi_main_layout.setSpacing(4)

        # Top row: mode selection and controls
        roi_layout = QHBoxLayout()
        roi_label = QLabel("ROI controls")
        roi_label.setStyleSheet("font-weight: bold; color: #cccccc;")

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
            "QPushButton:disabled { background-color: #3e3e42; color: #777777; }"
        )

        self.roi_status_label = QLabel("No ROI")
        self.roi_status_label.setStyleSheet("color: #6a6a6a; margin-left: 10px;")

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
        self.roi_instructions = QLabel("")
        self.roi_instructions.setWordWrap(True)
        self.roi_instructions.setStyleSheet(
            "color: #4fc1ff; font-size: 11px; font-weight: bold; "
            "padding: 6px; background-color: #0d3354; border-radius: 4px;"
        )
        roi_main_layout.addWidget(self.roi_instructions)

        left_layout.addWidget(self.scroll, stretch=1)

        # Interactive instructions
        self.interaction_help = QLabel(
            "Double-click: Fit to screen  •  Drag: Pan  •  Ctrl+Scroll/Pinch: Zoom"
        )
        self.interaction_help.setAlignment(Qt.AlignCenter)
        self.interaction_help.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic; "
            "padding: 4px; background-color: #1e1e1e; border-radius: 3px;"
        )
        left_layout.addWidget(self.interaction_help)

        left_layout.addWidget(roi_frame)

        # Zoom control under video
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(8, 4, 8, 4)

        zoom_label = QLabel("Zoom")
        zoom_label.setStyleSheet("font-weight: bold; color: #cccccc;")

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 500)  # 0.1x to 5.0x, scaled by 100
        self.slider_zoom.setValue(100)  # 1.0x
        self.slider_zoom.setTickPosition(QSlider.TicksBelow)
        self.slider_zoom.setTickInterval(50)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)

        self.label_zoom_val = QLabel("1.00x")
        self.label_zoom_val.setStyleSheet(
            "color: #4fc1ff; font-weight: bold; min-width: 50px;"
        )

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.slider_zoom, stretch=1)
        zoom_layout.addWidget(self.label_zoom_val)

        left_layout.addWidget(zoom_frame)

        # Preview detection button (uses current player frame)
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #252526; border-radius: 6px;")
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 4, 8, 4)

        self.btn_test_detection = QPushButton("Test Detection on Preview")
        self.btn_test_detection.clicked.connect(self._test_detection_on_preview)
        self.btn_test_detection.setEnabled(False)
        self.btn_test_detection.setStyleSheet(
            "background-color: #0e639c; color: white; font-weight: bold;"
        )
        preview_layout.addWidget(self.btn_test_detection)

        left_layout.addWidget(preview_frame)

        # --- RIGHT PANEL: Configuration Tabs & Actions ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setUsesScrollButtons(
            True
        )  # Enable scroll buttons when tabs don't fit
        self.tabs.setElideMode(Qt.ElideNone)  # Don't truncate tab text

        # Tab 1: Setup (Files & Performance)
        from hydra_suite.trackerkit.gui.panels.setup_panel import SetupPanel

        self._setup_panel = SetupPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._setup_panel, "Get Started")
        # Bootstrap calls that require _setup_panel to be assigned first
        self._populate_preset_combo()
        self._populate_compute_runtime_options(preferred="cpu")
        self._on_runtime_context_changed()

        # Tab 2: Detection (Image, Method, Params)
        from hydra_suite.trackerkit.gui.panels.detection_panel import DetectionPanel

        self._detection_panel = DetectionPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._detection_panel, "Find Animals")
        # Post-construction bootstrap: populate YOLO model combos now that _detection_panel is assigned
        self._refresh_yolo_model_combo()
        self._refresh_yolo_detect_model_combo()
        self._refresh_yolo_crop_obb_model_combo()

        # Tab 3: Individual Analysis (Identity)
        from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel

        self._identity_panel = IdentityPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._identity_panel, "Analyze Individuals")
        # Post-construction bootstrap calls now that _identity_panel is assigned
        self._refresh_cnn_identity_model_combo()
        self._refresh_yolo_headtail_model_combo()
        self._update_background_color_button()
        self._populate_pose_runtime_flavor_options(backend="yolo")
        self._set_form_row_visible(
            self._identity_panel.form_pose_runtime,
            self._identity_panel.combo_pose_runtime_flavor,
            False,
        )
        self._refresh_pose_model_combo()
        self._refresh_pose_sleap_envs()
        self._refresh_pose_direction_keypoint_lists()
        self._sync_pose_backend_ui()
        self._sync_individual_analysis_mode_ui()

        # Tab 4: Tracking (Kalman, Logic, Lifecycle)
        from hydra_suite.trackerkit.gui.panels.tracking_panel import TrackingPanel

        self._tracking_panel = TrackingPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._tracking_panel, "Track Movement")

        # Tab 5: Data (Post-proc)
        from hydra_suite.trackerkit.gui.panels.postprocess_panel import PostProcessPanel

        self._postprocess_panel = PostProcessPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._postprocess_panel, "Clean Results")

        # Tab 6: Dataset Generation (Active Learning)
        from hydra_suite.trackerkit.gui.panels.dataset_panel import DatasetPanel

        self._dataset_panel = DatasetPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._dataset_panel, "Build Dataset")
        # Populate conda environments now that the panel (and its combo widget) exists
        self._dataset_panel._refresh_xanylabeling_envs()

        right_layout.addWidget(self.tabs, stretch=1)

        # Persistent Action Panel (Bottom Right)
        action_frame = QFrame()
        action_frame.setStyleSheet(
            "background-color: #1e1e1e; border-top: 1px solid #3e3e42; border-radius: 0px;"
        )
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(6, 6, 6, 6)
        action_layout.setSpacing(4)

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
        stats_layout.setContentsMargins(2, 2, 2, 2)

        self.label_current_fps = QLabel("FPS: --")
        self.label_current_fps.setStyleSheet(
            "color: #4fc1ff; font-weight: bold; font-size: 11px;"
        )
        self.label_current_fps.setVisible(False)
        stats_layout.addWidget(self.label_current_fps)

        self.label_elapsed_time = QLabel("Elapsed: --")
        self.label_elapsed_time.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        self.label_elapsed_time.setVisible(False)
        stats_layout.addWidget(self.label_elapsed_time)

        self.label_eta = QLabel("ETA: --")
        self.label_eta.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        self.label_eta.setVisible(False)
        stats_layout.addWidget(self.label_eta)

        stats_layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_preview = QPushButton("Preview Mode")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(lambda ch: self.toggle_preview(ch))
        self.btn_preview.setMinimumHeight(34)

        self.btn_start = QPushButton("Start Full Tracking")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(lambda ch: self.toggle_tracking(ch))
        self.btn_start.setMinimumHeight(34)

        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_start)

        action_layout.addLayout(prog_layout)
        action_layout.addLayout(stats_layout)
        action_layout.addLayout(btn_layout)

        right_layout.addWidget(action_frame)

        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)

        # Set initial splitter ratio (60% Video, 40% Controls) and minimum sizes
        total_width = 1360  # Default window width
        self.splitter.setSizes([int(total_width * 0.63), int(total_width * 0.37)])
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.splitter.splitterMoved.connect(lambda *_args: self._queue_ui_state_save())
        self.tabs.currentChanged.connect(lambda _index: self._queue_ui_state_save())

        main_layout.addWidget(self.splitter)

        # =====================================================================
        # INITIALIZE PRESETS
        # =====================================================================
        # Populate preset combo box with available presets
        self._populate_preset_combo()

        # Load default preset (custom if available, otherwise default.json)
        self._load_default_preset_on_startup()

    def _make_welcome_page(self):
        """Create startup splash page with primary HYDRA session actions."""
        from hydra_suite.widgets import (
            ButtonDef,
            RecentItemsStore,
            WelcomeConfig,
            WelcomePage,
        )

        store = RecentItemsStore("tracker")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="trackerkit.svg",
            tagline="Track  |  Analyze  |  Refine",
            buttons=[
                ButtonDef(label="Load Video\u2026", callback=self.select_file),
                ButtonDef(
                    label="Load Video List\u2026", callback=self._import_batch_list
                ),
                ButtonDef(label="Load Config\u2026", callback=self.load_config),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=self._open_recent_video,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page

    def _open_recent_video(self, path: str):
        """Open a video file from the recent items list."""
        from pathlib import Path

        video_path = Path(path)
        if video_path.exists():
            self._setup_video_file(str(video_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "File Not Found", f"Video not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()

    def _show_workspace(self):
        """Switch to the main HYDRA workspace view."""
        if hasattr(self, "main_stack") and hasattr(self, "_workspace_page_index"):
            self.main_stack.setCurrentIndex(self._workspace_page_index)

    # =========================================================================
    # TAB UI BUILDERS
    # =========================================================================

    def _find_or_plan_optimizer_cache_path(
        self, video_path: str, params: dict, start_frame: int, end_frame: int
    ) -> tuple:
        """
        Return (cache_path, already_valid) where:
          - cache_path  : best path to use (existing covering cache, or new build target)
          - already_valid: True when that path exists AND covers start_frame..end_frame

        Search order:
          1. current_detection_cache_path (set by the last tracking run in this session)
             — only if it was produced by the *same* detection method.
          2. Any compatible detection cache in the video's dedicated cache directory
             whose filename matches the current detection method, with legacy flat
             files still considered as a fallback.
          3. A freshly-computed optimizer-specific path in the dedicated cache directory
             (name encodes the detection method so different methods never collide).
        """
        import re

        from hydra_suite.data.detection_cache import DetectionCache

        detection_method = params.get("DETECTION_METHOD", "background_subtraction")
        method_tag = "yolo" if detection_method == "yolo_obb" else "bgsub"

        def _cache_matches_method(path_str: str) -> bool:
            """Return True when *path_str*'s filename was produced by *method_tag*."""
            if not path_str:
                return False
            basename = os.path.basename(path_str)
            # Production tracking caches: ..._detection_cache_{method_tag}_...
            if "_detection_cache_" in basename:
                after = basename.split("_detection_cache_", 1)[1]
                return after.startswith(f"{method_tag}_") or after.startswith(
                    f"{method_tag}."
                )
            # Optimizer caches (new naming): ..._opt_cache.npz with method_tag
            if "_opt_cache" in basename:
                before_opt = basename.split("_opt_cache")[0]
                # Accept if the method tag appears as a delimited segment
                segments = before_opt.split("_")
                return method_tag in segments
            return False

        def _is_valid(path: str) -> bool:
            """Return True if *path* is a compatible cache covering the frame range."""
            if not path or not os.path.exists(path):
                return False
            try:
                dc = DetectionCache(path, mode="r")
                ok = dc.is_compatible() and dc.covers_frame_range(
                    start_frame, end_frame
                )
                dc.close()
                return ok
            except Exception:
                return False

        # 1. Production cache from current session — only if method-compatible
        if _cache_matches_method(self.current_detection_cache_path) and _is_valid(
            self.current_detection_cache_path
        ):
            return self.current_detection_cache_path, True

        csv_dir = (
            os.path.dirname(self._setup_panel.csv_line.text())
            if self._setup_panel.csv_line.text()
            else ""
        )
        artifact_base_dirs = candidate_artifact_base_dirs(
            video_path,
            preferred_base_dirs=[csv_dir],
        )

        # 2. Scan known cache directories — accept only method-compatible caches.
        for candidate in iter_detection_cache_candidates(
            video_path,
            artifact_base_dirs=artifact_base_dirs,
        ):
            candidate_str = str(candidate)
            if _cache_matches_method(candidate_str) and _is_valid(candidate_str):
                return candidate_str, True

        # 3. Fallback: compute a write-target path for a new detection-only build.
        #    Include the detection method so different methods never share a cache.
        if detection_method == "yolo_obb":
            model_raw = os.path.splitext(
                os.path.basename(params.get("YOLO_MODEL_PATH", "model"))
            )[0]
            model = re.sub(r"[^A-Za-z0-9_-]", "_", model_raw)
            opt_model_name = f"yolo_{model}"
        else:
            opt_model_name = "bgsub"
        resize = int(params.get("RESIZE_FACTOR", 1.0) * 100)
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        new_path = build_optimizer_detection_cache_path(
            video_path,
            opt_model_name,
            resize,
            artifact_base_dir=artifact_base_dir,
        )
        return str(new_path), False

    def _build_optimizer_detection_cache(
        self, video_path: str, cache_path: str, params: dict
    ):
        """Spin up a DetectionCacheBuilderWorker and show progress in the main window."""
        self._cache_builder_worker = DetectionCacheBuilderWorker(
            video_path,
            cache_path,
            params,
            self._setup_panel.spin_start_frame.value(),
            self._setup_panel.spin_end_frame.value(),
        )
        self._cache_builder_worker.progress_signal.connect(self.on_progress_update)
        self._cache_builder_worker.finished_signal.connect(
            self._on_optimizer_cache_built
        )
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Building detection cache for optimizer...")
        self._cache_builder_worker.start()

    def _on_optimizer_cache_built(self, ok: bool, cache_path: str):
        """Called when DetectionCacheBuilderWorker finishes."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if getattr(self, "_stop_all_requested", False):
            return

        if not ok:
            QMessageBox.critical(
                self,
                "Detection Failed",
                "Could not build the detection cache. Check that the YOLO model "
                "path is correct and that the video is readable.",
            )
            return
        # Store built cache as the current cache so _open_parameter_helper finds it
        self.current_detection_cache_path = cache_path
        # Re-open the helper now that the cache is ready
        self._open_parameter_helper()

    def _on_preview_cache_built(self, ok: bool, cache_path: str):
        """Called when DetectionCacheBuilderWorker finishes building the preview cache."""
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if getattr(self, "_stop_all_requested", False):
            return

        if not ok:
            QMessageBox.critical(
                self,
                "Detection Failed",
                "Could not build the detection cache. Check that the YOLO model "
                "path is correct and that the video is readable.",
            )
            return
        self.current_detection_cache_path = cache_path
        video_path = getattr(
            self,
            "_pending_preview_video_path",
            self._setup_panel.file_line.text().strip(),
        )
        self.start_preview_on_video(video_path)

    def _apply_optimized_params(self, new_params):
        """Apply optimized parameter values from the helper dialog to UI widgets."""
        _direct_mappings = [
            ("YOLO_CONFIDENCE_THRESHOLD", self._detection_panel.spin_yolo_confidence),
            ("YOLO_IOU_THRESHOLD", self._detection_panel.spin_yolo_iou),
            ("MAX_DISTANCE_MULTIPLIER", self._tracking_panel.spin_max_dist),
            ("KALMAN_NOISE_COVARIANCE", self._tracking_panel.spin_kalman_noise),
            (
                "KALMAN_MEASUREMENT_NOISE_COVARIANCE",
                self._tracking_panel.spin_kalman_meas,
            ),
            ("W_POSITION", self._tracking_panel.spin_Wp),
            ("W_ORIENTATION", self._tracking_panel.spin_Wo),
            ("W_AREA", self._tracking_panel.spin_Wa),
            ("W_ASPECT", self._tracking_panel.spin_Wasp),
            ("KALMAN_DAMPING", self._tracking_panel.spin_kalman_damping),
            (
                "KALMAN_MAX_VELOCITY_MULTIPLIER",
                self._tracking_panel.spin_kalman_max_velocity,
            ),
            (
                "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER",
                self._tracking_panel.spin_kalman_longitudinal_noise,
            ),
        ]
        for key, widget in _direct_mappings:
            if key in new_params:
                widget.setValue(new_params[key])

        # Frame-count-to-seconds conversions
        _opt_fps = self._setup_panel.spin_fps.value()
        if "KALMAN_MATURITY_AGE" in new_params:
            self._tracking_panel.spin_kalman_maturity_age.setValue(
                new_params["KALMAN_MATURITY_AGE"] / _opt_fps
            )
        if "LOST_THRESHOLD_FRAMES" in new_params:
            self._tracking_panel.spin_lost_thresh.setValue(
                new_params["LOST_THRESHOLD_FRAMES"] / _opt_fps
            )

    def _open_parameter_helper(self):
        """Open the tracking parameter selection helper dialog."""
        video_path = self._setup_panel.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        start_frame = self._setup_panel.spin_start_frame.value()
        end_frame = self._setup_panel.spin_end_frame.value()

        if (end_frame - start_frame) > 1000:
            QMessageBox.warning(
                self,
                "Range Too Large",
                "The selected range is very large. For faster optimization, "
                "please select a smaller slice (e.g., 100-500 frames) using "
                "the 'Start Frame' and 'End Frame' boxes.",
            )
            return

        params = self.get_parameters_dict()

        cache_path, already_valid = self._find_or_plan_optimizer_cache_path(
            video_path, params, start_frame, end_frame
        )

        if not already_valid:
            res = QMessageBox.question(
                self,
                "Detection Required",
                "No detection cache covering frames "
                f"{start_frame}\u2013{end_frame} was found.\n\n"
                "Run a quick detection-only scan now?\n"
                "(No config save, no pose inference, no CSV output \u2014 "
                "detections only.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                self._build_optimizer_detection_cache(video_path, cache_path, params)
            return  # dialog opens via _on_optimizer_cache_built when ready

        dialog = ParameterHelperDialog(
            video_path, cache_path, start_frame, end_frame, params, self
        )

        if dialog.exec() == QDialog.Accepted:
            new_params = dialog.get_selected_params()
            if new_params:
                self._apply_optimized_params(new_params)
                QMessageBox.information(
                    self,
                    "Parameters Applied",
                    "The optimized parameters have been applied to the UI.",
                )

    # ── BG-subtraction auto-tuner ─────────────────────────────────────────

    def _open_bg_parameter_helper(self):
        """Open the BG-subtraction parameter auto-tuner dialog."""
        video_path = self._setup_panel.file_line.text().strip()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        params = self.get_parameters_dict()

        dialog = BgParameterHelperDialog(video_path, params, self)
        if dialog.exec() == QDialog.Accepted:
            new_p = dialog.get_selected_params()
            if not new_p:
                return
            # Apply optimised values back to the UI widgets
            if "THRESHOLD_VALUE" in new_p:
                self._detection_panel.spin_threshold.setValue(new_p["THRESHOLD_VALUE"])
            if "MORPH_KERNEL_SIZE" in new_p:
                self._detection_panel.spin_morph_size.setValue(
                    new_p["MORPH_KERNEL_SIZE"]
                )
            if "MIN_CONTOUR_AREA" in new_p:
                self._detection_panel.spin_min_contour.setValue(
                    new_p["MIN_CONTOUR_AREA"]
                )
            if "ENABLE_ADDITIONAL_DILATION" in new_p:
                self._detection_panel.chk_additional_dilation.setChecked(
                    new_p["ENABLE_ADDITIONAL_DILATION"]
                )
            if "DILATION_KERNEL_SIZE" in new_p:
                self._detection_panel.spin_dilation_kernel_size.setValue(
                    new_p["DILATION_KERNEL_SIZE"]
                )
            if "DILATION_ITERATIONS" in new_p:
                self._detection_panel.spin_dilation_iterations.setValue(
                    new_p["DILATION_ITERATIONS"]
                )
            if "ENABLE_CONSERVATIVE_SPLIT" in new_p:
                self._detection_panel.chk_conservative_split.setChecked(
                    new_p["ENABLE_CONSERVATIVE_SPLIT"]
                )
            if "CONSERVATIVE_KERNEL_SIZE" in new_p:
                self._detection_panel.spin_conservative_kernel.setValue(
                    new_p["CONSERVATIVE_KERNEL_SIZE"]
                )
            if "CONSERVATIVE_ERODE_ITER" in new_p:
                self._detection_panel.spin_conservative_erode.setValue(
                    new_p["CONSERVATIVE_ERODE_ITER"]
                )
            QMessageBox.information(
                self,
                "Parameters Applied",
                "Detection parameters have been applied to the UI.\n"
                "Use 'Preview Detection' to verify the results.",
            )

    def _on_individual_analysis_toggled(self, state):
        """Enable/disable individual analysis controls."""
        self._sync_individual_analysis_mode_ui()

    def _on_identity_method_changed(self, index):
        """Update identity configuration stack when method changes."""
        self._sync_identity_method_ui()

    def _on_identity_analysis_toggled(self, state):
        """Enable/disable identity-method controls inside individual analysis."""
        self._sync_identity_method_ui()

    def _on_pose_analysis_toggled(self, state):
        """Enable/disable pose-extraction controls inside individual analysis."""
        self._sync_pose_analysis_ui()

    @staticmethod
    def _get_apriltag_families() -> list:
        """Return the list of tag families supported by the installed apriltag library.

        Probes the library by passing an intentionally invalid family name and
        parsing the error message, which always lists every known family.  Falls
        back to a static list if the library is not installed.
        """
        _FALLBACK = [
            "tag36h11",
            "tag25h9",
            "tag16h5",
            "tagCircle21h7",
            "tagCircle49h12",
            "tagStandard41h12",
            "tagStandard52h13",
            "tagCustom48h12",
        ]
        try:
            import apriltag as _at  # type: ignore[import-untyped]
        except ImportError:
            return _FALLBACK
        try:
            _at.apriltag("__probe__")
        except Exception as exc:
            import re

            families = re.findall(r"^\s+(\S+)$", str(exc), re.MULTILINE)
            if families:
                return sorted(families)
        return _FALLBACK

    def _select_color_tag_model(self):
        """Browse for color tag YOLO model."""
        # Default to models directory
        start_dir = get_models_directory()
        if self._identity_panel.line_color_tag_model.text():
            current_path = resolve_model_path(
                self._identity_panel.line_color_tag_model.text()
            )
            if os.path.exists(current_path):
                start_dir = os.path.dirname(current_path)

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Color Tag YOLO Model", start_dir, "YOLO Models (*.pt *.onnx)"
        )
        if filepath:
            # Check if model is outside the archive
            models_dir = get_models_directory()
            try:
                rel_path = os.path.relpath(filepath, models_dir)
                is_in_archive = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_archive = False

            if not is_in_archive:
                # Ask user if they want to copy to archive
                reply = QMessageBox.question(
                    self,
                    "Copy Model to Archive?",
                    f"The selected model is outside the local model archive.\n\n"
                    f"Would you like to copy it to the archive at:\n{models_dir}\n\n"
                    f"This makes presets portable across devices.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    import shutil

                    filename = os.path.basename(filepath)
                    dest_path = os.path.join(models_dir, filename)

                    # Handle duplicate filenames
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                models_dir, f"{base}_{counter}{ext}"
                            )
                            counter += 1

                    try:
                        shutil.copy2(filepath, dest_path)
                        filepath = dest_path
                        logger.info(f"Copied color tag model to archive: {dest_path}")
                        QMessageBox.information(
                            self,
                            "Model Copied",
                            f"Model copied to archive as:\n{os.path.basename(dest_path)}",
                        )
                    except Exception as e:
                        logger.error(f"Failed to copy model: {e}")
                        QMessageBox.warning(
                            self,
                            "Copy Failed",
                            f"Could not copy model to archive:\n{e}\n\nUsing original path.",
                        )

            self._identity_panel.line_color_tag_model.setText(filepath)

    class CNNClassifierRow(QWidget):
        """Self-contained widget for one CNN classifier configuration row."""

        remove_requested = Signal(object)

        def __init__(self, main_window, parent=None):
            super().__init__(parent)
            self._main_window = main_window
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(4, 4, 4, 4)
            outer.setSpacing(4)

            # Header row: model combo + remove button
            header_row = QHBoxLayout()
            self.combo_model = QComboBox()
            self.combo_model.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self._populate_model_combo()
            self.combo_model.activated.connect(self._on_model_selected)
            header_row.addWidget(self.combo_model)
            self.btn_remove = QPushButton("\u2715")
            self.btn_remove.setMaximumWidth(28)
            self.btn_remove.setToolTip("Remove this CNN classifier")
            self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
            header_row.addWidget(self.btn_remove)
            outer.addLayout(header_row)

            # Verification labels
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
            self.lbl_arch = QLabel("\u2014")
            self.lbl_num_classes = QLabel("\u2014")
            self.lbl_class_names = QLabel("\u2014")
            self.lbl_input_size = QLabel("\u2014")
            self.lbl_label = QLabel("\u2014")
            form.addRow("Architecture", self.lbl_arch)
            form.addRow("Num classes", self.lbl_num_classes)
            form.addRow("Class names", self.lbl_class_names)
            form.addRow("Input size", self.lbl_input_size)
            form.addRow("Classification label", self.lbl_label)

            # Inference settings
            self.spin_confidence = QDoubleSpinBox()
            self.spin_confidence.setRange(0.0, 1.0)
            self.spin_confidence.setSingleStep(0.05)
            self.spin_confidence.setValue(0.5)
            form.addRow("Confidence threshold", self.spin_confidence)

            self.spin_window = QSpinBox()
            self.spin_window.setRange(1, 100)
            self.spin_window.setValue(10)
            form.addRow("History window", self.spin_window)

            outer.addLayout(form)

            # Separator line
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            outer.addWidget(line)

        def _populate_model_combo(self):
            """Populate combo from model_registry.json (usage_role == 'cnn_identity')."""
            registry_path = get_yolo_model_registry_path()
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except (FileNotFoundError, ValueError):
                registry = {}
            self.combo_model.blockSignals(True)
            current = self.combo_model.currentData()
            self.combo_model.clear()
            self.combo_model.addItem("\u2014 select model \u2014", "")
            for rel_path, meta in registry.items():
                if meta.get("usage_role") != "cnn_identity":
                    continue
                arch = meta.get("arch", "?")
                label = meta.get("classification_label", "")
                n_cls = meta.get("num_classes", "?")
                display = f"{arch} | {n_cls} cls"
                if label:
                    display += f" | {label}"
                self.combo_model.addItem(display, rel_path)
            self.combo_model.addItem("\uff0b Add New Model\u2026", "__add_new__")
            idx = self.combo_model.findData(current)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
            self.combo_model.blockSignals(False)

        def _on_model_selected(self, index: int):
            rel_path = self.combo_model.itemData(index)
            if rel_path == "__add_new__":
                self._main_window._handle_add_new_cnn_identity_model()
                self._populate_model_combo()
                return
            self._update_verification_labels(rel_path)

        def _update_verification_labels(self, rel_path: str):
            try:
                with open(get_yolo_model_registry_path()) as f:
                    registry = json.load(f)
            except Exception:
                return
            meta = registry.get(rel_path or "", {})
            self.lbl_arch.setText(str(meta.get("arch", "\u2014")))
            self.lbl_num_classes.setText(str(meta.get("num_classes", "\u2014")))
            class_names = meta.get("class_names", [])
            preview = ", ".join(class_names[:12])
            if len(class_names) > 12:
                preview += f", \u2026 ({len(class_names)} total)"
            self.lbl_class_names.setText(preview or "\u2014")
            self.lbl_input_size.setText(str(meta.get("input_size", "\u2014")))
            self.lbl_label.setText(str(meta.get("classification_label", "\u2014")))

        def to_config(self):
            """Return config dict or None if no model selected."""
            rel_path = self.combo_model.currentData()
            if not rel_path or rel_path == "__add_new__":
                return None
            registry_path = get_yolo_model_registry_path()
            try:
                with open(registry_path) as f:
                    registry = json.load(f)
            except Exception:
                registry = {}
            meta = registry.get(rel_path, {})
            models_root = get_models_root_directory()
            abs_path = os.path.join(models_root, rel_path)
            label = str(meta.get("classification_label", "") or "cnn_identity")
            return {
                "model_path": abs_path,
                "label": label,
                "confidence": self.spin_confidence.value(),
                "window": self.spin_window.value(),
                "batch_size": 64,
                "rel_path": rel_path,  # for config save/load roundtrip
            }

        def load_from_config(self, cfg: dict):
            """Populate from a config dict entry."""
            rel_path = cfg.get("rel_path", "")
            if not rel_path:
                # Try to find by abs model_path
                abs_path = cfg.get("model_path", "")
                models_root = get_models_root_directory()
                try:
                    with open(get_yolo_model_registry_path()) as f:
                        registry = json.load(f)
                    for rp, meta in registry.items():
                        if os.path.normpath(
                            os.path.join(models_root, rp)
                        ) == os.path.normpath(abs_path):
                            rel_path = rp
                            break
                except Exception:
                    pass
            if rel_path:
                self._populate_model_combo()
                idx = self.combo_model.findData(rel_path)
                if idx >= 0:
                    self.combo_model.setCurrentIndex(idx)
                    self._update_verification_labels(rel_path)
            if "confidence" in cfg:
                self.spin_confidence.setValue(float(cfg["confidence"]))
            if "window" in cfg:
                self.spin_window.setValue(int(cfg["window"]))

    def _add_cnn_classifier_row(self) -> "CNNClassifierRow":
        """Add a new CNN classifier row and return it."""
        row = self.CNNClassifierRow(self)
        row.remove_requested.connect(self._remove_cnn_classifier_row)
        self._identity_panel.cnn_rows_layout.addWidget(row)
        self._identity_panel.cnn_scroll_area.setVisible(True)
        return row

    def _remove_cnn_classifier_row(self, row: "CNNClassifierRow"):
        """Remove a CNN classifier row."""
        self._identity_panel.cnn_rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._identity_panel.cnn_scroll_area.setVisible(
            bool(self._cnn_classifier_rows())
        )

    def _cnn_classifier_rows(self) -> list:
        """Return list of all CNNClassifierRow instances."""
        rows = []
        for i in range(self._identity_panel.cnn_rows_layout.count()):
            item = self._identity_panel.cnn_rows_layout.itemAt(i)
            if item and isinstance(item.widget(), self.CNNClassifierRow):
                rows.append(item.widget())
        return rows

    def _refresh_cnn_identity_model_combo(self) -> None:
        """Populate the CNN identity model combo from model_registry.json."""
        if not hasattr(self, "_identity_panel"):
            return
        registry_path = get_yolo_model_registry_path()
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            registry = {}

        self._identity_panel.combo_cnn_identity_model.blockSignals(True)
        current_path = self._identity_panel.combo_cnn_identity_model.currentData()
        self._identity_panel.combo_cnn_identity_model.clear()
        self._identity_panel.combo_cnn_identity_model.addItem("— select model —", "")
        for rel_path, meta in registry.items():
            if meta.get("usage_role") != "cnn_identity":
                continue
            arch = meta.get("arch", "?")
            label = meta.get("classification_label", "")
            species = meta.get("species", "")
            n_cls = meta.get("num_classes", "?")
            display = f"{arch} | {n_cls} cls"
            if species:
                display += f" | {species}"
            if label:
                display += f" | {label}"
            self._identity_panel.combo_cnn_identity_model.addItem(display, rel_path)
        self._identity_panel.combo_cnn_identity_model.addItem(
            "\uff0b Add New Model\u2026", "__add_new__"
        )

        idx = self._identity_panel.combo_cnn_identity_model.findData(current_path)
        if idx >= 0:
            self._identity_panel.combo_cnn_identity_model.setCurrentIndex(idx)
        self._identity_panel.combo_cnn_identity_model.blockSignals(False)

    def _on_cnn_identity_model_selected(self, index: int) -> None:
        """Handle combo activation — sentinel triggers import dialog."""
        rel_path = self._identity_panel.combo_cnn_identity_model.itemData(index)
        if rel_path == "__add_new__":
            self._handle_add_new_cnn_identity_model()
            return
        self._update_cnn_identity_verification_panel(rel_path)

    def _update_cnn_identity_verification_panel(self, rel_path: str) -> None:
        """Populate the read-only verification labels from the registry entry."""
        registry_path = get_yolo_model_registry_path()
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except Exception:
            return
        meta = registry.get(rel_path or "", {})
        self._identity_panel.lbl_cnn_arch.setText(str(meta.get("arch", "\u2014")))
        self._identity_panel.lbl_cnn_num_classes.setText(
            str(meta.get("num_classes", "\u2014"))
        )
        class_names = meta.get("class_names", [])
        preview = ", ".join(class_names[:12])
        if len(class_names) > 12:
            preview += f", \u2026 ({len(class_names)} total)"
        self._identity_panel.lbl_cnn_class_names.setText(preview or "\u2014")
        raw_size = meta.get("input_size", "\u2014")
        self._identity_panel.lbl_cnn_input_size.setText(str(raw_size))
        self._identity_panel.lbl_cnn_label.setText(
            str(meta.get("classification_label", "\u2014"))
        )

    def _handle_add_new_cnn_identity_model(self) -> None:
        """Import a ClassKit-trained .pth or YOLO .pt model for CNN identity."""
        prev_data = self._identity_panel.combo_cnn_identity_model.currentData()

        src_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import ClassKit Model for CNN Identity",
            os.path.join(get_models_root_directory(), "classification", "identity"),
            "ClassKit Model Files (*.pth *.pt);;All Files (*)",
        )
        if not src_path:
            idx = self._identity_panel.combo_cnn_identity_model.findData(prev_data)
            self._identity_panel.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return

        # Read checkpoint metadata
        meta: dict = {}
        try:
            if src_path.endswith(".pth"):
                import torch

                ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
                meta["arch"] = ckpt.get("arch", "tinyclassifier")
                meta["class_names"] = ckpt.get("class_names", [])
                meta["factor_names"] = ckpt.get("factor_names", [])
                raw_size = ckpt.get("input_size", (224, 224))
                meta["input_size"] = (
                    list(raw_size)
                    if isinstance(raw_size, (list, tuple))
                    else [raw_size, raw_size]
                )
                meta["num_classes"] = ckpt.get("num_classes", len(meta["class_names"]))
            else:  # .pt (YOLO)
                from ultralytics import YOLO as _YOLO

                yolo = _YOLO(src_path)
                names = yolo.names
                meta["arch"] = "yolo"
                meta["class_names"] = [names[i] for i in sorted(names.keys())]
                meta["factor_names"] = []
                meta["input_size"] = [224, 224]
                meta["num_classes"] = len(meta["class_names"])
                del yolo  # explicitly release; YOLO may hold GPU tensors
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Import Error",
                f"Could not read checkpoint metadata:\n{exc}",
            )
            idx = self._identity_panel.combo_cnn_identity_model.findData(prev_data)
            self._identity_panel.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return

        from hydra_suite.trackerkit.gui.dialogs import CNNIdentityImportDialog

        dlg = CNNIdentityImportDialog(meta, parent=self)
        if dlg.exec() != QDialog.Accepted:
            idx = self._identity_panel.combo_cnn_identity_model.findData(prev_data)
            self._identity_panel.combo_cnn_identity_model.setCurrentIndex(max(idx, 0))
            return

        species = dlg.species()
        classification_label = dlg.classification_label()

        dest_dir = os.path.join(
            get_models_root_directory(), "classification", "identity"
        )
        os.makedirs(dest_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ext = Path(src_path).suffix
        label_part = f"_{classification_label}" if classification_label else ""
        filename = f"{timestamp}_{meta['arch']}_{species}{label_part}{ext}"
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(src_path, dest_path)

        rel_path = os.path.relpath(dest_path, get_models_root_directory())
        registry_path = get_yolo_model_registry_path()
        try:
            with open(registry_path) as f:
                registry = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            registry = {}

        registry[rel_path] = {
            "arch": meta["arch"],
            "num_classes": meta["num_classes"],
            "class_names": meta["class_names"],
            "factor_names": meta["factor_names"],
            "input_size": meta["input_size"],
            "species": species,
            "classification_label": classification_label,
            "added_at": datetime.now().isoformat(),
            "task_family": "classify",
            "usage_role": "cnn_identity",
        }
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        self._refresh_cnn_identity_model_combo()
        idx = self._identity_panel.combo_cnn_identity_model.findData(rel_path)
        if idx >= 0:
            self._identity_panel.combo_cnn_identity_model.setCurrentIndex(idx)
            self._update_cnn_identity_verification_panel(rel_path)

    def _ensure_pose_model_path_store(self):
        if not hasattr(self, "_pose_model_path_by_backend"):
            self._pose_model_path_by_backend = {"yolo": "", "sleap": "", "vitpose": ""}

    def _current_pose_backend_key(self):
        if not hasattr(self, "_identity_panel"):
            return "yolo"
        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        if backend == "sleap":
            return "sleap"
        if backend == "vitpose":
            return "vitpose"
        return "yolo"

    def _pose_model_path_for_backend(self, backend=None):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        if key not in ("sleap", "vitpose"):
            key = "yolo"
        return str(self._pose_model_path_by_backend.get(key, "")).strip()

    def _set_pose_model_path_for_backend(self, path, backend=None, update_combo=False):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        if key not in ("sleap", "vitpose"):
            key = "yolo"
        value = str(path or "").strip()
        if value:
            resolved = str(resolve_pose_model_path(value, backend=key)).strip()
            if resolved and os.path.exists(resolved):
                value = str(make_pose_model_path_relative(os.path.abspath(resolved)))
        self._pose_model_path_by_backend[key] = value
        if update_combo:
            self._refresh_pose_model_combo(preferred_model_path=value)

    def _handle_add_new_pose_model(self):
        """Browse for a pose model, import it if outside repo, refresh combo, and select it.

        Restores the previous selection if the user cancels.
        """
        combo = getattr(self, "combo_pose_model", None)
        prev_data = None
        if combo is not None:
            for i in range(combo.count()):
                d = combo.itemData(i, Qt.UserRole)
                if d and d not in ("__add_new__", "__none__"):
                    prev_data = d
                    break

        def _restore():
            if combo is not None:
                combo.blockSignals(True)
                self._set_model_selection_for_selector(combo, prev_data)
                combo.blockSignals(False)

        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        backend_key = (
            "sleap"
            if backend == "sleap"
            else ("vitpose" if backend == "vitpose" else "yolo")
        )
        current = self._pose_model_path_for_backend(backend)
        if current:
            resolved_current = str(resolve_pose_model_path(current, backend=backend))
            start = (
                resolved_current
                if os.path.isdir(resolved_current)
                else (os.path.dirname(resolved_current) or str(Path.home()))
            )
        else:
            start = get_pose_models_directory(backend_key)

        if backend == "sleap":
            selected = QFileDialog.getExistingDirectory(
                self, "Select SLEAP Model Directory", start
            )
            if not selected:
                _restore()
                return
            selected_abs = os.path.abspath(selected)
            pose_root = get_pose_models_directory(backend_key)
            try:
                rel_path = os.path.relpath(selected_abs, pose_root)
                is_in_repo = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_repo = False
            if is_in_repo:
                final_path = make_pose_model_path_relative(selected_abs)
            else:
                final_path = self._import_pose_model_to_repository(
                    selected_abs, backend=backend_key
                )
                if not final_path:
                    _restore()
                    return
                QMessageBox.information(
                    self,
                    "Model Added",
                    f"SLEAP model added to repository:\n{final_path}",
                )
            self._set_pose_model_path_for_backend(
                final_path, backend=backend, update_combo=True
            )
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pose Weights",
            start,
            "PyTorch Weights (*.pt *.pth);;All Files (*)",
        )
        if not selected:
            _restore()
            return
        selected_abs = os.path.abspath(selected)
        pose_root = get_pose_models_directory(backend_key)
        try:
            rel_path = os.path.relpath(selected_abs, pose_root)
            is_in_repo = not rel_path.startswith("..")
        except (ValueError, TypeError):
            is_in_repo = False
        if is_in_repo:
            final_path = make_pose_model_path_relative(selected_abs)
        else:
            final_path = self._import_pose_model_to_repository(
                selected_abs, backend=backend_key
            )
            if not final_path:
                _restore()
                return
            QMessageBox.information(
                self,
                "Model Added",
                f"Pose model added to repository:\n{final_path}",
            )
        self._set_pose_model_path_for_backend(
            final_path, backend=backend, update_combo=True
        )

    def _import_pose_model_to_repository(self, source_path, backend="yolo"):
        """Copy a selected pose model into models/pose/{YOLO|SLEAP|ViTPose} and return relative path."""
        src = str(source_path or "").strip()
        if not src or not os.path.exists(src):
            return None

        bk = str(backend).strip().lower()
        backend_key = (
            "sleap" if bk == "sleap" else ("vitpose" if bk == "vitpose" else "yolo")
        )
        dest_dir = get_pose_models_directory(backend_key)

        try:
            src_path = Path(src).expanduser().resolve()
        except Exception:
            src_path = Path(src)

        try:
            rel_existing = os.path.relpath(str(src_path), str(Path(dest_dir).resolve()))
            if not rel_existing.startswith(".."):
                return make_pose_model_path_relative(str(src_path))
        except Exception:
            pass

        # Metadata collection (same style as YOLO OBB import dialog).
        now_preview = datetime.now()
        dlg = QDialog(self)
        dlg.setWindowTitle("Pose Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        stem_tokens = [t for t in src_path.stem.replace("-", "_").split("_") if t]
        default_species = (
            self._sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            self._sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        size_combo = None
        type_line = None
        if backend_key == "yolo":
            size_combo = QComboBox(dlg)
            size_combo.addItems(
                ["26n", "26s", "26m", "26l", "26x", "custom", "unknown"]
            )
            size_combo.setCurrentText("26s")
            dlg_form.addRow("YOLO model size:", size_combo)
        else:
            default_type = self._sanitize_model_token(src_path.name) or "sleap_model"
            type_line = QLineEdit(default_type, dlg)
            type_line.setPlaceholderText("model-type")
            dlg_form.addRow("Model type:", type_line)

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_species = self._sanitize_model_token(species_line.text())
        model_info = self._sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        if backend_key == "sleap":
            model_type = (
                self._sanitize_model_token(type_line.text()) if type_line else ""
            )
            if not model_type:
                QMessageBox.warning(
                    self,
                    "Invalid Metadata",
                    "SLEAP model type must be provided.",
                )
                return None
            target_name = f"{timestamp}_{model_type}_{model_species}_{model_info}"
            dest_path = Path(dest_dir) / target_name
            counter = 1
            while dest_path.exists():
                dest_path = Path(dest_dir) / f"{target_name}_{counter}"
                counter += 1
            try:
                shutil.copytree(src_path, dest_path)
            except Exception as exc:
                logger.error("Failed to copy SLEAP model directory: %s", exc)
                QMessageBox.warning(
                    self,
                    "Import Failed",
                    f"Could not import SLEAP model directory:\n{exc}",
                )
                return None
            return make_pose_model_path_relative(str(dest_path))

        model_size = size_combo.currentText().strip() if size_combo else "unknown"
        model_size = self._sanitize_model_token(model_size) or "unknown"
        ext = src_path.suffix or ".pt"
        target_name = f"{timestamp}_{model_size}_{model_species}_{model_info}{ext}"
        dest_path = Path(dest_dir) / target_name
        counter = 1
        while dest_path.exists():
            dest_path = (
                Path(dest_dir)
                / f"{timestamp}_{model_size}_{model_species}_{model_info}_{counter}{ext}"
            )
            counter += 1
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as exc:
            logger.error("Failed to copy pose model: %s", exc)
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Could not import pose model:\n{exc}",
            )
            return None
        return make_pose_model_path_relative(str(dest_path))

    def _select_pose_skeleton_file(self):
        """Select pose skeleton JSON file."""
        start = self._identity_panel.line_pose_skeleton_file.text().strip() or str(
            Path.home()
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pose Skeleton JSON",
            start,
            "JSON Files (*.json);;All Files (*)",
        )
        if selected:
            self._identity_panel.line_pose_skeleton_file.setText(selected)

    def _load_pose_skeleton_keypoint_names(self):
        """Load keypoint names from selected skeleton JSON."""
        skeleton_file = self._identity_panel.line_pose_skeleton_file.text().strip()
        if not skeleton_file:
            return []
        try:
            path = Path(skeleton_file).expanduser().resolve()
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("keypoint_names", data.get("keypoints", []))
            names = [str(v).strip() for v in raw if str(v).strip()]
            return names
        except Exception:
            return []

    def _selected_pose_group_keypoints(self, list_widget):
        """Return selected keypoint names for a pose direction list widget."""
        if list_widget is None:
            return []
        return [
            item.text().strip()
            for item in list_widget.selectedItems()
            if item.text().strip()
        ]

    def _set_pose_group_selection(self, list_widget, values):
        """Select keypoints in list widget from config-provided values."""
        if list_widget is None:
            return
        tokens = self._parse_pose_keypoint_tokens(values)
        if not tokens:
            return
        items_by_name = {}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            items_by_name[item.text().strip().lower()] = item
        for tok in tokens:
            if isinstance(tok, int):
                if 0 <= tok < list_widget.count():
                    list_widget.item(tok).setSelected(True)
                continue
            item = items_by_name.get(str(tok).strip().lower())
            if item is not None:
                item.setSelected(True)

    def _apply_pose_keypoint_selection_set(self, list_widget, selected_names):
        """Set list selection to exact keypoint-name set."""
        if list_widget is None:
            return
        target = {str(v).strip().lower() for v in selected_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            item.setSelected(item_name in target)

    def _set_pose_keypoints_enabled(self, list_widget, disabled_names):
        """Disable keypoints in a list by name while preserving others."""
        if list_widget is None:
            return
        disabled = {str(v).strip().lower() for v in disabled_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            flags = item.flags()
            if item_name in disabled:
                item.setSelected(False)
                item.setFlags(flags & ~Qt.ItemIsEnabled)
            else:
                item.setFlags(flags | Qt.ItemIsEnabled)

    def _apply_pose_keypoint_selection_constraints(self, changed_group="ignore"):
        """Enforce exclusivity across ignore/anterior/posterior keypoint selections."""
        ignore_list = getattr(self, "list_pose_ignore_keypoints", None)
        ant_list = getattr(self, "list_pose_direction_anterior", None)
        post_list = getattr(self, "list_pose_direction_posterior", None)
        if ignore_list is None or ant_list is None or post_list is None:
            return

        ignore = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ignore_list)
            if k.strip()
        }
        anterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ant_list)
            if k.strip()
        }
        posterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(post_list)
            if k.strip()
        }

        # Ignore selection always wins over directional groups.
        anterior -= ignore
        posterior -= ignore

        # Anterior/posterior remain mutually exclusive. The changed group wins.
        if changed_group == "posterior":
            anterior -= posterior
        else:
            posterior -= anterior

        ant_list.blockSignals(True)
        post_list.blockSignals(True)
        self._apply_pose_keypoint_selection_set(ant_list, anterior)
        self._apply_pose_keypoint_selection_set(post_list, posterior)
        self._set_pose_keypoints_enabled(ant_list, ignore)
        self._set_pose_keypoints_enabled(post_list, ignore)
        ant_list.blockSignals(False)
        post_list.blockSignals(False)

    def _on_pose_keypoint_group_changed(self, changed_group):
        """Handle keypoint group updates and keep list constraints synchronized."""
        self._apply_pose_keypoint_selection_constraints(changed_group)

    def _refresh_pose_direction_keypoint_lists(self):
        """Populate ignore/anterior/posterior keypoint pickers from skeleton file."""
        if not hasattr(self, "_identity_panel"):
            return

        prev_ignore = self._selected_pose_group_keypoints(
            self._identity_panel.list_pose_ignore_keypoints
        )
        prev_anterior = self._selected_pose_group_keypoints(
            self._identity_panel.list_pose_direction_anterior
        )
        prev_posterior = self._selected_pose_group_keypoints(
            self._identity_panel.list_pose_direction_posterior
        )
        names = self._load_pose_skeleton_keypoint_names()

        self._identity_panel.list_pose_ignore_keypoints.blockSignals(True)
        self._identity_panel.list_pose_direction_anterior.blockSignals(True)
        self._identity_panel.list_pose_direction_posterior.blockSignals(True)
        self._identity_panel.list_pose_ignore_keypoints.clear()
        self._identity_panel.list_pose_direction_anterior.clear()
        self._identity_panel.list_pose_direction_posterior.clear()
        self._identity_panel.list_pose_ignore_keypoints.addItems(names)
        self._identity_panel.list_pose_direction_anterior.addItems(names)
        self._identity_panel.list_pose_direction_posterior.addItems(names)
        self._set_pose_group_selection(
            self._identity_panel.list_pose_ignore_keypoints, prev_ignore
        )
        self._set_pose_group_selection(
            self._identity_panel.list_pose_direction_anterior, prev_anterior
        )
        self._set_pose_group_selection(
            self._identity_panel.list_pose_direction_posterior, prev_posterior
        )
        enabled = len(names) > 0
        self._identity_panel.list_pose_ignore_keypoints.setEnabled(enabled)
        self._identity_panel.list_pose_direction_anterior.setEnabled(enabled)
        self._identity_panel.list_pose_direction_posterior.setEnabled(enabled)
        self._identity_panel.list_pose_ignore_keypoints.blockSignals(False)
        self._identity_panel.list_pose_direction_anterior.blockSignals(False)
        self._identity_panel.list_pose_direction_posterior.blockSignals(False)
        self._apply_pose_keypoint_selection_constraints("ignore")

    def _parse_pose_keypoint_tokens(self, raw):
        """Parse comma-separated keypoint names and/or indices."""
        if not raw:
            return []
        if isinstance(raw, (list, tuple, set)):
            tokens = []
            for value in raw:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                try:
                    tokens.append(int(text))
                except ValueError:
                    tokens.append(text)
            return tokens
        out = []
        for token in str(raw).split(","):
            t = token.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                out.append(t)
        return out

    def _parse_pose_ignore_keypoints(self):
        """Parse keypoints to ignore from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_ignore_keypoints", None)
        )

    def _parse_pose_direction_anterior_keypoints(self):
        """Parse anterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_direction_anterior", None)
        )

    def _parse_pose_direction_posterior_keypoints(self):
        """Parse posterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_direction_posterior", None)
        )

    def _refresh_pose_sleap_envs(self):
        """Refresh conda environments starting with 'sleap'."""
        if not hasattr(self, "_identity_panel"):
            return

        self._identity_panel.combo_pose_sleap_env.clear()
        self._identity_panel.combo_pose_sleap_env.setEnabled(True)
        envs = []
        preferred = str(self.advanced_config.get("pose_sleap_env", "sleap")).strip()
        try:
            import subprocess

            res = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[0].strip()
                    if name.lower().startswith("sleap"):
                        envs.append(name)
        except Exception:
            envs = []

        if not envs:
            self._identity_panel.combo_pose_sleap_env.addItem("No sleap envs found")
            self._identity_panel.combo_pose_sleap_env.setEnabled(False)
            return

        self._identity_panel.combo_pose_sleap_env.addItems(envs)
        if preferred and preferred in envs:
            self._identity_panel.combo_pose_sleap_env.setCurrentText(preferred)

    def _selected_pose_sleap_env(self):
        """Return valid selected SLEAP env name or default."""
        if not hasattr(self, "_identity_panel"):
            return "sleap"
        txt = self._identity_panel.combo_pose_sleap_env.currentText().strip()
        if not txt or txt.lower().startswith("no sleap envs"):
            return "sleap"
        return txt

    def _sleap_experimental_features_enabled(self):
        """Return True if SLEAP experimental features (ONNX/TensorRT) are allowed."""
        if not hasattr(self, "_identity_panel"):
            return False
        return self._identity_panel.chk_sleap_experimental_features.isChecked()

    def _on_sleap_experimental_toggled(self):
        """Handle experimental features checkbox toggle."""
        if not hasattr(self, "_identity_panel"):
            return
        # Refresh runtime options to show warning if needed
        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
            if hasattr(self, "_identity_panel")
            else "yolo"
        )
        if backend == "sleap":
            current_flavor = self._selected_pose_runtime_flavor()
            if (
                current_flavor in ("onnx", "tensorrt")
                and not self._sleap_experimental_features_enabled()
            ):
                # Show warning that runtime will revert to native
                from PySide6.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Experimental Features Disabled",
                    f"SLEAP {current_flavor.upper()} runtime is experimental.\\n"
                    "With experimental features disabled, the runtime will revert to native.\\n\\n"
                    "To use ONNX/TensorRT for SLEAP, enable experimental features.",
                )

    def _runtime_pipelines_for_current_ui(self):
        pipelines = []
        if self._is_yolo_detection_mode():
            pipelines.append("yolo_obb_detection")
        if self._is_pose_inference_enabled():
            backend = (
                self._identity_panel.combo_pose_model_type.currentText().strip().lower()
            )
            if backend == "sleap":
                pipelines.append("sleap_pose")
            else:
                pipelines.append("yolo_pose")
        return pipelines

    def _compute_runtime_options_for_current_ui(self):
        allowed = allowed_runtimes_for_pipelines(
            self._runtime_pipelines_for_current_ui()
        )
        if not allowed:
            allowed = ["cpu"]
        return [(runtime_label(rt), rt) for rt in allowed if rt in CANONICAL_RUNTIMES]

    def _populate_compute_runtime_options(self, preferred: str | None = None):
        if not hasattr(self, "_setup_panel"):
            return
        combo = self._setup_panel.combo_compute_runtime
        selected = (
            str(preferred or self._selected_compute_runtime() or "cpu").strip().lower()
        )
        options = self._compute_runtime_options_for_current_ui()
        values = [value for _label, value in options]
        if selected not in values:
            selected = values[0] if values else "cpu"
        combo.blockSignals(True)
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _selected_compute_runtime(self) -> str:
        if not hasattr(self, "_setup_panel"):
            return "cpu"
        data = self._setup_panel.combo_compute_runtime.currentData()
        if data:
            return str(data).strip().lower()
        txt = self._setup_panel.combo_compute_runtime.currentText().strip().lower()
        if txt in CANONICAL_RUNTIMES:
            return txt
        return "cpu"

    def _runtime_requires_fixed_yolo_batch(self, runtime: str | None = None) -> bool:
        rt = str(runtime or self._selected_compute_runtime() or "").strip().lower()
        return rt == "tensorrt" or rt.startswith("onnx")

    @staticmethod
    def _preview_safe_runtime(runtime: str) -> str:
        """Downgrade ONNX/TensorRT runtimes to their native equivalents.

        Preview mode and test-detection-on-preview do not benefit from
        ONNX/TensorRT (single-frame, no warm-up) and those runtimes add
        unnecessary latency.  Map them back to the underlying device runtime.
        """
        rt = str(runtime or "cpu").strip().lower()
        if rt == "onnx_cpu":
            return "cpu"
        if rt in ("onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "onnx_rocm":
            return "rocm"
        return rt

    def _on_runtime_context_changed(self, *_args):
        previous = self._selected_compute_runtime()
        self._populate_compute_runtime_options(preferred=previous)
        selected_runtime = self._selected_compute_runtime()
        self._update_obb_mode_warning()
        # Keep hidden legacy controls synchronized for compatibility paths.
        derived = derive_detection_runtime_settings(selected_runtime)
        if hasattr(self, "_detection_panel"):
            idx = self._detection_panel.combo_device.findText(
                str(derived.get("yolo_device", "cpu")), Qt.MatchStartsWith
            )
            if idx >= 0:
                self._detection_panel.combo_device.setCurrentIndex(idx)
        if hasattr(self, "_detection_panel"):
            self._detection_panel.chk_enable_tensorrt.setChecked(
                bool(derived.get("enable_tensorrt", False))
            )
        if (
            self._runtime_requires_fixed_yolo_batch(selected_runtime)
            and hasattr(self, "combo_yolo_batch_mode")
            and hasattr(self, "spin_yolo_batch_size")
            and hasattr(self, "chk_enable_yolo_batching")
        ):
            self._detection_panel.chk_enable_yolo_batching.setChecked(True)
            self._detection_panel.chk_enable_yolo_batching.setEnabled(False)
            self._detection_panel.combo_yolo_batch_mode.setCurrentIndex(1)  # Manual
            self._detection_panel.combo_yolo_batch_mode.setEnabled(False)
            self._detection_panel.spin_yolo_batch_size.setEnabled(True)
            if hasattr(self, "spin_tensorrt_batch"):
                self._detection_panel.spin_tensorrt_batch.setValue(
                    self._detection_panel.spin_yolo_batch_size.value()
                )
        elif hasattr(self, "combo_yolo_batch_mode") and hasattr(
            self, "chk_enable_yolo_batching"
        ):
            self._detection_panel.chk_enable_yolo_batching.setEnabled(True)
            self._detection_panel.combo_yolo_batch_mode.setEnabled(
                self._detection_panel.chk_enable_yolo_batching.isChecked()
            )
            self._on_yolo_batch_mode_changed(
                self._detection_panel.combo_yolo_batch_mode.currentIndex()
            )
        if hasattr(self, "_identity_panel"):
            self._populate_pose_runtime_flavor_options(
                backend=self._identity_panel.combo_pose_model_type.currentText()
                .strip()
                .lower(),
                preferred=self._selected_pose_runtime_flavor(),
            )

    def _pose_runtime_options_for_backend(self, backend: str):
        derived = derive_pose_runtime_settings(
            self._selected_compute_runtime(), backend_family=backend
        )
        flavor = str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()
        return [(runtime_label(self._selected_compute_runtime()), flavor)]

    def _populate_pose_runtime_flavor_options(
        self, backend: str, preferred: str | None = None
    ):
        if not hasattr(self, "_identity_panel"):
            return
        combo = self._identity_panel.combo_pose_runtime_flavor
        selected = (
            str(preferred or self._selected_pose_runtime_flavor() or "auto")
            .strip()
            .lower()
        )
        options = self._pose_runtime_options_for_backend(backend)
        values = [value for _label, value in options]
        if selected not in values:
            selected = values[0] if values else "cpu"
        combo.blockSignals(True)
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _selected_pose_runtime_flavor(self) -> str:
        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
            if hasattr(self, "_identity_panel")
            else "yolo"
        )
        derived = derive_pose_runtime_settings(
            self._selected_compute_runtime(), backend_family=backend
        )
        return str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()

    def _set_form_row_visible(self, form_layout, field_widget, visible: bool):
        """Show/hide a QFormLayout row by field widget."""
        if form_layout is None or field_widget is None:
            return
        label = form_layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(bool(visible))
        field_widget.setVisible(bool(visible))

    def _sync_pose_backend_ui(self):
        """Show/hide backend-specific pose controls."""
        if not hasattr(self, "_identity_panel"):
            return
        backend = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        self._populate_pose_runtime_flavor_options(backend=backend)
        is_sleap = backend == "sleap"
        if hasattr(self, "_identity_panel") and hasattr(
            self._identity_panel, "pose_sleap_env_row_widget"
        ):
            self._set_form_row_visible(
                self._identity_panel.form_pose_runtime,
                self._identity_panel.pose_sleap_env_row_widget,
                is_sleap,
            )
        if hasattr(self, "_identity_panel") and hasattr(
            self._identity_panel, "pose_sleap_experimental_row_widget"
        ):
            self._set_form_row_visible(
                self._identity_panel.form_pose_runtime,
                self._identity_panel.pose_sleap_experimental_row_widget,
                is_sleap,
            )
        # Refresh pose model combo to show models for the selected backend.
        self._refresh_pose_model_combo(
            preferred_model_path=self._pose_model_path_for_backend(backend)
        )
        self._on_runtime_context_changed()

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        return bool(
            self._is_individual_pipeline_enabled()
            and hasattr(self, "_identity_panel")
            and self._identity_panel.chk_enable_pose_extractor.isChecked()
        )

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        has_controls = hasattr(self, "check_video_show_pose") and hasattr(
            self, "combo_video_pose_color_mode"
        )
        if not has_controls:
            return

        video_visible = bool(
            hasattr(self, "_postprocess_panel")
            and self._postprocess_panel.check_video_output.isChecked()
        )
        pose_enabled = self._is_pose_inference_enabled()
        enabled = bool(video_visible and pose_enabled)

        self._postprocess_panel.check_video_show_pose.setEnabled(enabled)
        show_pose = bool(
            enabled and self._postprocess_panel.check_video_show_pose.isChecked()
        )
        fixed_color_mode = (
            self._postprocess_panel.combo_video_pose_color_mode.currentIndex() == 1
        )

        # Show detailed controls only when pose overlay is on.
        self._postprocess_panel.lbl_video_pose_color_mode.setVisible(show_pose)
        self._postprocess_panel.combo_video_pose_color_mode.setVisible(show_pose)
        self._postprocess_panel.lbl_video_pose_point_radius.setVisible(show_pose)
        self._postprocess_panel.spin_video_pose_point_radius.setVisible(show_pose)
        self._postprocess_panel.lbl_video_pose_point_thickness.setVisible(show_pose)
        self._postprocess_panel.spin_video_pose_point_thickness.setVisible(show_pose)
        self._postprocess_panel.lbl_video_pose_line_thickness.setVisible(show_pose)
        self._postprocess_panel.spin_video_pose_line_thickness.setVisible(show_pose)

        show_fixed_color = bool(show_pose and fixed_color_mode)
        self._postprocess_panel.lbl_video_pose_color_label.setVisible(show_fixed_color)
        self._postprocess_panel.btn_video_pose_color.setVisible(show_fixed_color)
        self._postprocess_panel.lbl_video_pose_color.setVisible(show_fixed_color)

        self._postprocess_panel.combo_video_pose_color_mode.setEnabled(show_pose)
        self._postprocess_panel.spin_video_pose_point_radius.setEnabled(show_pose)
        self._postprocess_panel.spin_video_pose_point_thickness.setEnabled(show_pose)
        self._postprocess_panel.spin_video_pose_line_thickness.setEnabled(show_pose)
        self._postprocess_panel.btn_video_pose_color.setEnabled(show_fixed_color)

        self._postprocess_panel.lbl_video_pose_disabled_hint.setVisible(video_visible)
        if enabled:
            self._postprocess_panel.lbl_video_pose_disabled_hint.setText(
                "Pose overlay will use keypoints from pose-augmented tracking output."
            )
        else:
            self._postprocess_panel.lbl_video_pose_disabled_hint.setText(
                "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings."
            )

    def _is_yolo_detection_mode(self) -> bool:
        """Return True when current detection mode is YOLO OBB."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel.combo_detection_method.currentIndex() == 1

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline."""
        return self._is_yolo_detection_mode()

    def _is_identity_analysis_enabled(self) -> bool:
        """Return effective runtime state for identity classification."""
        if not hasattr(self, "_identity_panel"):
            return False
        return bool(
            self._identity_panel.g_identity.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _selected_identity_method(self) -> str:
        """Return canonical identity-method key for runtime/config usage.

        Preserved for backward compat. Returns 'apriltags' if AprilTags is the
        only active method, 'cnn_classifier' if only CNNs are active, or
        'none_disabled' if nothing is active.
        """
        if not self._is_identity_analysis_enabled():
            return "none_disabled"
        cfg = self._identity_config()
        has_apriltags = cfg.get("use_apriltags", False)
        has_cnn = bool(cfg.get("cnn_classifiers", []))
        if has_apriltags and not has_cnn:
            return "apriltags"
        if has_cnn and not has_apriltags:
            return "cnn_classifier"
        if has_apriltags or has_cnn:
            return "cnn_classifier"  # multi-method: report as cnn_classifier for compat
        return "none_disabled"

    def _identity_config(self) -> dict:
        """Return use_apriltags + cnn_classifiers config dict (replaces _selected_identity_method).

        match_bonus and mismatch_penalty are shared across all identity sources and injected
        into each CNN phase here so the worker/hungarian receive per-phase values already set.
        """
        if not self._is_identity_analysis_enabled():
            return {"use_apriltags": False, "cnn_classifiers": []}
        use_apriltags = (
            hasattr(self, "_identity_panel")
            and self._identity_panel.g_apriltags.isChecked()
        )
        match_bonus = float(
            self._identity_panel.spin_identity_match_bonus.value()
            if hasattr(self, "_identity_panel")
            else 20.0
        )
        mismatch_penalty = float(
            self._identity_panel.spin_identity_mismatch_penalty.value()
            if hasattr(self, "_identity_panel")
            else 50.0
        )
        cnn_classifiers = []
        if hasattr(self, "_cnn_classifier_rows"):
            for row in self._cnn_classifier_rows():
                cfg = row.to_config()
                if cfg is not None:
                    cfg["match_bonus"] = match_bonus
                    cfg["mismatch_penalty"] = mismatch_penalty
                    cnn_classifiers.append(cfg)
        return {
            "use_apriltags": use_apriltags,
            "cnn_classifiers": cnn_classifiers,
            "match_bonus": match_bonus,
            "mismatch_penalty": mismatch_penalty,
        }

    def _sync_identity_method_ui(self):
        """Show the active identity configuration only when enabled."""
        identity_enabled = self._is_identity_analysis_enabled()
        if hasattr(self, "_identity_panel"):
            self._identity_panel.identity_content.setVisible(identity_enabled)
            self._identity_panel.identity_content.setEnabled(identity_enabled)

    def _sync_pose_analysis_ui(self):
        """Show pose controls only when pose extraction is enabled."""
        pose_enabled = self._is_pose_inference_enabled()
        if hasattr(self, "_identity_panel"):
            self._identity_panel.pose_runtime_content.setVisible(pose_enabled)
            self._identity_panel.pose_runtime_content.setEnabled(pose_enabled)

    def _is_individual_image_save_enabled(self) -> bool:
        """Return effective runtime state for saving individual crops."""
        if not hasattr(self, "_dataset_panel"):
            return False
        return bool(
            self._dataset_panel.chk_enable_individual_dataset.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _should_generate_oriented_track_videos(self) -> bool:
        """Return True when final per-track oriented videos should be exported."""
        if not hasattr(self, "_dataset_panel"):
            return False
        return bool(
            self._dataset_panel.chk_generate_individual_track_videos.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _should_run_interpolated_postpass(self) -> bool:
        """
        Return True when interpolated post-pass should run.

        We run this pass when interpolation is enabled and either:
        - individual crop saving is enabled, or
        - pose export is enabled (to fill occluded-frame pose rows in final CSV), or
        - oriented track video export is enabled (to cache interpolated ROI geometry).
        """
        if not hasattr(self, "_identity_panel"):
            return False
        if not self._identity_panel.chk_individual_interpolate.isChecked():
            return False
        if not self._is_individual_pipeline_enabled():
            return False
        return bool(
            self._is_individual_image_save_enabled()
            or self._is_pose_export_enabled()
            or self._should_generate_oriented_track_videos()
        )

    def _sync_individual_analysis_mode_ui(self):
        """Enforce YOLO-only pipeline and run/save dependency in UI."""
        has_save_toggle = hasattr(self, "_dataset_panel")
        is_yolo = self._is_yolo_detection_mode()

        if hasattr(self, "tabs") and hasattr(self, "_identity_panel"):
            tab_index = self.tabs.indexOf(self._identity_panel)
            if tab_index >= 0:
                if not is_yolo and self.tabs.currentWidget() is self._identity_panel:
                    fallback_index = self.tabs.indexOf(
                        getattr(self, "_detection_panel", self._setup_panel)
                    )
                    if fallback_index >= 0:
                        self.tabs.setCurrentIndex(fallback_index)
                if hasattr(self.tabs, "setTabVisible"):
                    self.tabs.setTabVisible(tab_index, is_yolo)
                elif hasattr(self.tabs, "tabBar") and hasattr(
                    self.tabs.tabBar(), "setTabVisible"
                ):
                    self.tabs.tabBar().setTabVisible(tab_index, is_yolo)
                self.tabs.setTabEnabled(tab_index, is_yolo)

        pipeline_enabled = self._is_individual_pipeline_enabled()

        if hasattr(self, "_identity_panel"):
            self._identity_panel.lbl_individual_yolo_only_notice.setVisible(not is_yolo)

        if hasattr(self, "_identity_panel"):
            self._identity_panel.g_identity.setVisible(pipeline_enabled)
            self._identity_panel.g_identity.setEnabled(pipeline_enabled)
        if hasattr(self, "_identity_panel"):
            self._identity_panel.g_pose_runtime.setVisible(pipeline_enabled)
            self._identity_panel.g_pose_runtime.setEnabled(pipeline_enabled)
        if hasattr(self, "_identity_panel"):
            self._identity_panel.g_individual_pipeline_common.setVisible(
                pipeline_enabled
            )
            self._identity_panel.g_individual_pipeline_common.setEnabled(
                pipeline_enabled
            )
        if hasattr(self, "_dataset_panel"):
            self._dataset_panel.g_individual_dataset.setVisible(pipeline_enabled)
            self._dataset_panel.g_individual_dataset.setEnabled(pipeline_enabled)
        self._sync_identity_method_ui()
        self._sync_pose_analysis_ui()
        self._sync_pose_backend_ui()

        if has_save_toggle:
            self._dataset_panel.chk_enable_individual_dataset.setEnabled(
                pipeline_enabled
            )

        save_enabled = self._is_individual_image_save_enabled()
        if hasattr(self, "_dataset_panel"):
            self._dataset_panel.ind_output_group.setVisible(save_enabled)
            self._dataset_panel.ind_output_group.setEnabled(save_enabled)
        if hasattr(self, "_dataset_panel"):
            self._dataset_panel.lbl_individual_info.setVisible(save_enabled)
        if hasattr(self, "_dataset_panel"):
            self._dataset_panel.chk_generate_individual_track_videos.setVisible(
                pipeline_enabled
            )
            has_headtail = bool(
                str(self._get_selected_yolo_headtail_model_path() or "").strip()
            )
            oriented_enabled = pipeline_enabled and has_headtail
            self._dataset_panel.chk_generate_individual_track_videos.setEnabled(
                oriented_enabled
            )
            if not oriented_enabled:
                self._dataset_panel.chk_generate_individual_track_videos.setChecked(
                    False
                )
                self._dataset_panel.chk_generate_individual_track_videos.setToolTip(
                    "Requires a head-tail model to be configured."
                )
            else:
                self._dataset_panel.chk_generate_individual_track_videos.setToolTip(
                    "After final cleaning completes, export one orientation-fixed video per\n"
                    "final TrajectoryID by streaming the source video and using the detection\n"
                    "cache plus interpolated ROI cache. Independent from saved crop files."
                )
        self._sync_video_pose_overlay_controls()
        self._on_runtime_context_changed()

    def _select_individual_background_color(self):
        """Open color picker for individual dataset background color."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        # Convert current BGR to RGB for QColorDialog
        b, g, r = self._identity_panel._background_color
        initial_color = QColor(r, g, b)

        color = QColorDialog.getColor(initial_color, self, "Choose Background Color")
        if color.isValid():
            # Convert RGB back to BGR for OpenCV
            self._identity_panel._background_color = (
                color.blue(),
                color.green(),
                color.red(),
            )
            self._update_background_color_button()

    def _select_video_pose_color(self):
        """Open color picker for fixed pose overlay color (BGR)."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        b, g, r = self._postprocess_panel._video_pose_color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(initial_color, self, "Choose Pose Overlay Color")
        if color.isValid():
            self._postprocess_panel._video_pose_color = (
                color.blue(),
                color.green(),
                color.red(),
            )
            self._update_video_pose_color_button()

    def _update_video_pose_color_button(self):
        """Update fixed pose-color preview button and text label."""
        b, g, r = self._postprocess_panel._video_pose_color
        self._postprocess_panel.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self._postprocess_panel.lbl_video_pose_color.setText(
            f"{self._postprocess_panel._video_pose_color}"
        )

    def _update_background_color_button(self):
        """Update the color button display and label."""
        b, g, r = self._identity_panel._background_color
        # Set button color
        self._identity_panel.btn_background_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        # Update label with BGR values
        self._identity_panel.lbl_background_color.setText(
            f"{self._identity_panel._background_color}"
        )

    def _compute_median_background_color(self):
        """Compute median color from current preview frame or load from video."""
        frame = None

        # Try to use preview frame first
        if (
            hasattr(self, "preview_frame_original")
            and self.preview_frame_original is not None
        ):
            # preview_frame_original is in RGB, convert to BGR for processing
            frame = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2BGR)
        # Otherwise, try to load from video if available
        elif self.current_video_path:
            cap = cv2.VideoCapture(self.current_video_path)
            if cap.isOpened():
                ret, frame_bgr = cap.read()
                cap.release()
                if ret:
                    frame = frame_bgr

        if frame is None:
            QMessageBox.warning(
                self, "No Frame", "Please load a video first to compute median color."
            )
            return

        try:
            from hydra_suite.utils.image_processing import (
                compute_median_color_from_frame,
            )

            # Compute median color
            median_color = compute_median_color_from_frame(frame)
            # Convert numpy.uint8 to regular int for JSON serialization
            self._identity_panel._background_color = tuple(int(c) for c in median_color)
            self._update_background_color_button()

            QMessageBox.information(
                self,
                "Median Color Computed",
                f"Background color set to median:\nBGR: {median_color}",
            )
        except Exception as e:
            logger.error(f"Failed to compute median color: {e}")
            QMessageBox.warning(self, "Error", f"Failed to compute median color:\n{e}")

    def _on_yolo_batching_toggled(self, state):
        """Enable/disable YOLO batching controls based on checkbox."""
        if self._runtime_requires_fixed_yolo_batch():
            # TensorRT/ONNX runtimes require explicit fixed batch size.
            if not self._detection_panel.chk_enable_yolo_batching.isChecked():
                self._detection_panel.chk_enable_yolo_batching.setChecked(True)
            self._detection_panel.chk_enable_yolo_batching.setEnabled(False)
            self._detection_panel.combo_yolo_batch_mode.setVisible(True)
            self._detection_panel.lbl_yolo_batch_mode.setVisible(True)
            self._detection_panel.spin_yolo_batch_size.setVisible(True)
            self._detection_panel.lbl_yolo_batch_size.setVisible(True)
            self._detection_panel.combo_yolo_batch_mode.setCurrentIndex(1)
            self._detection_panel.combo_yolo_batch_mode.setEnabled(False)
            self._detection_panel.spin_yolo_batch_size.setEnabled(True)
            return

        # Directly check checkbox state for reliability
        enabled = self._detection_panel.chk_enable_yolo_batching.isChecked()

        # Hide/show batching widgets
        self._detection_panel.combo_yolo_batch_mode.setVisible(enabled)
        self._detection_panel.lbl_yolo_batch_mode.setVisible(enabled)
        self._detection_panel.spin_yolo_batch_size.setVisible(enabled)
        self._detection_panel.lbl_yolo_batch_size.setVisible(enabled)

        # Also control enable state
        self._detection_panel.combo_yolo_batch_mode.setEnabled(enabled)
        # Manual batch size only enabled if batching is on AND mode is Manual
        manual_mode = self._detection_panel.combo_yolo_batch_mode.currentIndex() == 1
        self._detection_panel.spin_yolo_batch_size.setEnabled(enabled and manual_mode)

    def _on_yolo_manual_batch_size_changed(self, value: int):
        """Keep legacy fixed-batch field synchronized for fixed runtimes."""
        if self._runtime_requires_fixed_yolo_batch() and hasattr(
            self, "spin_tensorrt_batch"
        ):
            self._detection_panel.spin_tensorrt_batch.setValue(int(value))

    def _on_yolo_batch_mode_changed(self, index):
        """Show/hide manual batch size based on selected mode."""
        if self._runtime_requires_fixed_yolo_batch():
            # TensorRT/ONNX runtimes require explicit fixed batch size.
            if self._detection_panel.combo_yolo_batch_mode.currentIndex() != 1:
                self._detection_panel.combo_yolo_batch_mode.setCurrentIndex(1)
            self._detection_panel.spin_yolo_batch_size.setEnabled(True)
            return
        # index 0 = Auto, index 1 = Manual
        is_manual = index == 1
        batching_enabled = self._detection_panel.chk_enable_yolo_batching.isChecked()
        self._detection_panel.spin_yolo_batch_size.setEnabled(
            batching_enabled and is_manual
        )

    def _on_tensorrt_toggled(self, state):
        """Enable/disable TensorRT batch size control based on checkbox."""
        # TensorRT toggles are now derived from canonical compute runtime.
        # Keep legacy widgets hidden from UI.
        if not self._detection_panel.chk_enable_tensorrt.isVisible():
            self._detection_panel.spin_tensorrt_batch.setVisible(False)
            self._detection_panel.lbl_tensorrt_batch.setVisible(False)
            return

        # Directly check checkbox state for reliability
        enabled = self._detection_panel.chk_enable_tensorrt.isChecked()

        # Hide/show TensorRT batch size widgets
        self._detection_panel.spin_tensorrt_batch.setVisible(enabled)
        self._detection_panel.lbl_tensorrt_batch.setVisible(enabled)

        # Also control enable state
        self._detection_panel.spin_tensorrt_batch.setEnabled(enabled)
        self._detection_panel.lbl_tensorrt_batch.setEnabled(enabled)

    def _on_confidence_density_map_toggled(self, state):
        """Show or hide the density-map controls from the top-level tracking toggle."""
        if not hasattr(self, "_tracking_panel"):
            return

        enabled = self._tracking_panel.chk_enable_confidence_density_map.isChecked()
        self._tracking_panel.g_density.setVisible(enabled)
        self._tracking_panel.g_density.setEnabled(enabled)

    def _on_cleaning_toggled(self, state):
        """Enable/disable trajectory cleaning controls based on checkbox."""
        enabled = self._postprocess_panel.enable_postprocessing.isChecked()

        # Hide/show all cleaning parameter widgets
        self._postprocess_panel.spin_min_trajectory_length.setVisible(enabled)
        self._postprocess_panel.lbl_min_trajectory_length.setVisible(enabled)
        self._postprocess_panel.spin_max_velocity_break.setVisible(enabled)
        self._postprocess_panel.lbl_max_velocity_break.setVisible(enabled)
        self._postprocess_panel.spin_max_occlusion_gap.setVisible(enabled)
        self._postprocess_panel.lbl_max_occlusion_gap.setVisible(enabled)
        self._postprocess_panel.chk_enable_tracklet_relinking.setVisible(enabled)
        self._postprocess_panel.lbl_enable_tracklet_relinking.setVisible(enabled)
        self._postprocess_panel.spin_relink_pose_max_distance.setVisible(enabled)
        self._postprocess_panel.lbl_relink_pose_max_distance.setVisible(enabled)
        self._postprocess_panel.spin_max_velocity_zscore.setVisible(enabled)
        self._postprocess_panel.lbl_max_velocity_zscore.setVisible(enabled)
        self._postprocess_panel.spin_velocity_zscore_window.setVisible(enabled)
        self._postprocess_panel.lbl_velocity_zscore_window.setVisible(enabled)
        self._postprocess_panel.spin_velocity_zscore_min_vel.setVisible(enabled)
        self._postprocess_panel.lbl_velocity_zscore_min_vel.setVisible(enabled)
        self._postprocess_panel.combo_interpolation_method.setVisible(enabled)
        self._postprocess_panel.lbl_interpolation_method.setVisible(enabled)
        self._postprocess_panel.spin_interpolation_max_gap.setVisible(enabled)
        self._postprocess_panel.lbl_interpolation_max_gap.setVisible(enabled)
        self._postprocess_panel.spin_heading_flip_max_burst.setVisible(enabled)
        self._postprocess_panel.lbl_heading_flip_max_burst.setVisible(enabled)
        self._postprocess_panel.spin_merge_overlap_multiplier.setVisible(enabled)
        self._postprocess_panel.lbl_merge_overlap_multiplier.setVisible(enabled)
        self._postprocess_panel.spin_min_overlap_frames.setVisible(enabled)
        self._postprocess_panel.lbl_min_overlap_frames.setVisible(enabled)
        self._postprocess_panel.chk_cleanup_temp_files.setVisible(enabled)

        # Also control enable state
        self._postprocess_panel.spin_min_trajectory_length.setEnabled(enabled)
        self._postprocess_panel.spin_max_velocity_break.setEnabled(enabled)
        self._postprocess_panel.spin_max_occlusion_gap.setEnabled(enabled)
        self._postprocess_panel.chk_enable_tracklet_relinking.setEnabled(enabled)
        self._postprocess_panel.spin_relink_pose_max_distance.setEnabled(enabled)
        self._postprocess_panel.spin_max_velocity_zscore.setEnabled(enabled)
        self._postprocess_panel.spin_velocity_zscore_window.setEnabled(enabled)
        self._postprocess_panel.spin_velocity_zscore_min_vel.setEnabled(enabled)
        self._postprocess_panel.combo_interpolation_method.setEnabled(enabled)
        self._postprocess_panel.spin_interpolation_max_gap.setEnabled(enabled)
        self._postprocess_panel.spin_heading_flip_max_burst.setEnabled(enabled)
        self._postprocess_panel.spin_merge_overlap_multiplier.setEnabled(enabled)
        self._postprocess_panel.spin_min_overlap_frames.setEnabled(enabled)
        self._postprocess_panel.chk_cleanup_temp_files.setEnabled(enabled)

        # Pose quality widgets — visible only when post-processing AND pose export are active
        pose_enabled = enabled and self._is_pose_export_enabled()
        self._postprocess_panel.spin_pose_export_min_valid_fraction.setVisible(
            pose_enabled
        )
        self._postprocess_panel.lbl_pose_export_min_valid_fraction.setVisible(
            pose_enabled
        )
        self._postprocess_panel.spin_pose_export_min_valid_keypoints.setVisible(
            pose_enabled
        )
        self._postprocess_panel.lbl_pose_export_min_valid_keypoints.setVisible(
            pose_enabled
        )
        self._postprocess_panel.spin_relink_min_pose_quality.setVisible(pose_enabled)
        self._postprocess_panel.lbl_relink_min_pose_quality.setVisible(pose_enabled)
        self._postprocess_panel.spin_pose_postproc_max_gap.setVisible(pose_enabled)
        self._postprocess_panel.lbl_pose_postproc_max_gap.setVisible(pose_enabled)
        self._postprocess_panel.spin_pose_temporal_outlier_zscore.setVisible(
            pose_enabled
        )
        self._postprocess_panel.lbl_pose_temporal_outlier_zscore.setVisible(
            pose_enabled
        )

        self._postprocess_panel.spin_pose_export_min_valid_fraction.setEnabled(
            pose_enabled
        )
        self._postprocess_panel.spin_pose_export_min_valid_keypoints.setEnabled(
            pose_enabled
        )
        self._postprocess_panel.spin_relink_min_pose_quality.setEnabled(pose_enabled)
        self._postprocess_panel.spin_pose_postproc_max_gap.setEnabled(pose_enabled)
        self._postprocess_panel.spin_pose_temporal_outlier_zscore.setEnabled(
            pose_enabled
        )

    # =========================================================================
    # EVENT HANDLERS (Identical Logic to Original)
    # =========================================================================

    def _on_detection_method_changed_ui(self, index):
        """Update stack widget when detection method changes."""
        self._detection_panel.stack_detection.setCurrentIndex(index)
        # Show image adjustments only for Background Subtraction (index 0)
        is_background_subtraction = index == 0
        self._detection_panel.g_img.setVisible(is_background_subtraction)
        # Show/hide method-specific overlay groups
        self._detection_panel.g_overlays_bg.setVisible(is_background_subtraction)
        self._detection_panel.g_overlays_yolo.setVisible(not is_background_subtraction)
        # Refresh preview to show correct mode
        self._update_preview_display()
        self.on_detection_method_changed(index)
        self._on_runtime_context_changed()
        self._queue_ui_state_save()

    def closeEvent(self, event):
        """Persist MAT-specific UI layout state on close."""
        self._save_ui_settings()
        super().closeEvent(event)

    def select_file(self: object) -> object:
        """Select video file via file dialog."""
        from hydra_suite.paths import get_projects_dir

        start_fp = (
            self._setup_panel.file_line.text().strip()
            if hasattr(self, "_setup_panel")
            else ""
        )
        start_dir = os.path.dirname(start_fp) if start_fp else str(get_projects_dir())
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Video", start_dir, "Video Files (*.mp4 *.avi *.mov)"
        )
        if fp:
            # If batch mode is checked, update the keystone
            if self._setup_panel.g_batch.isChecked():
                if not self.batch_videos:
                    self.batch_videos = [fp]
                else:
                    if fp in self.batch_videos:
                        self.batch_videos.remove(fp)
                    self.batch_videos.insert(0, fp)
                self._sync_batch_list_ui()

            self._setup_video_file(fp)

    def _setup_video_file(self, fp, skip_config_load=False):
        """
        Setup a video file for tracking.

        Args:
            fp: Path to the video file
            skip_config_load: If True, skip auto-loading config (used when loading config itself)
        """
        self._setup_panel.file_line.setText(fp)
        self.current_video_path = fp

        # Reset caches for the new video
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None

        if self.roi_selection_active:
            self.clear_roi()

        # Auto-generate output paths based on video name
        video_dir = os.path.dirname(fp)
        video_name = os.path.splitext(os.path.basename(fp))[0]

        # Auto-populate CSV output
        csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
        self._setup_panel.csv_line.setText(csv_path)

        # Auto-populate video output and enable it
        video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
        self._postprocess_panel.video_out_line.setText(video_out_path)
        self._postprocess_panel.check_video_output.setChecked(True)

        # Enable preview detection button
        self.btn_test_detection.setEnabled(True)
        self._setup_panel.btn_detect_fps.setEnabled(True)

        # Initialize video player
        self._init_video_player(fp)

        # Update window title
        self.setWindowTitle(f"HYDRA - {os.path.basename(fp)}")

        # Update Start/End frame spins
        self._setup_panel.spin_start_frame.setValue(0)
        self._setup_panel.spin_end_frame.setValue(self.video_total_frames - 1)

        # Auto-load config if it exists for this video (unless explicitly skipped)
        if not skip_config_load:
            config_path = get_video_config_path(fp)
            if config_path and os.path.isfile(config_path):
                self._load_config_from_file(config_path)
                self._setup_panel.config_status_label.setText(
                    f"✓ Loaded: {os.path.basename(config_path)}"
                )
        else:
            self._setup_panel.config_status_label.setText(
                "ℹ️ Using current UI parameters (Keystone)"
            )
            self._setup_panel.config_status_label.setStyleSheet(
                "color: #f39c12; font-style: italic; font-size: 10px;"
            )

        # Enable controls
        self._apply_ui_state("idle")
        if hasattr(self, "_recents_store"):
            self._recents_store.add(fp)
        self._show_workspace()

    def _on_batch_mode_toggled(self, checked):
        """Handle showing/hiding batch controls and syncing keystone video."""
        self._setup_panel.lbl_batch_warning.setVisible(checked)
        self._setup_panel.container_batch.setVisible(checked)
        if checked:
            self._sync_keystone_to_batch()
        else:
            # If turning off batch mode, keep the current video but reset batch tracking state
            self.current_batch_index = -1

    def _sync_keystone_to_batch(self):
        """Ensure the currently loaded video is the FIRST item in the batch list."""
        video_path = self._setup_panel.file_line.text().strip()
        if not video_path:
            return

        if not self.batch_videos:
            self.batch_videos = [video_path]
        else:
            # If current video is already in list, move it to top
            if video_path in self.batch_videos:
                self.batch_videos.remove(video_path)
            self.batch_videos.insert(0, video_path)

        self._sync_batch_list_ui()

    def _sync_batch_list_ui(self):
        """Refresh the batch list widget with markers for the keystone."""
        self._setup_panel.list_batch_videos.clear()
        current_fp = (
            os.path.normpath(self._setup_panel.file_line.text().strip())
            if self._setup_panel.file_line.text().strip()
            else ""
        )

        for i, fp in enumerate(self.batch_videos):
            norm_fp = os.path.normpath(fp)
            if i == 0:
                item_text = f"⭐ KEYSTONE: {fp}"
            else:
                item_text = fp

            if norm_fp == current_fp:
                item_text = f"▶ CURRENT: {item_text}"

            item = QListWidgetItem(item_text)
            item.setToolTip(fp)

            if norm_fp == current_fp:
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self._setup_panel.list_batch_videos.addItem(item)

            if norm_fp == current_fp:
                self._setup_panel.list_batch_videos.setCurrentItem(item)

    def _on_batch_video_selected(self, *args):
        """Load a video from the batch list for preview/tuning."""
        row = self._setup_panel.list_batch_videos.currentRow()
        if 0 <= row < len(self.batch_videos):
            fp = self.batch_videos[row]
            # If it's already the current video, do nothing
            if fp == self.current_video_path:
                return

            # Skip config load so we keep using the keystone's parameters in the UI
            self._setup_video_file(fp, skip_config_load=True)
            self._sync_batch_list_ui()

    def _remove_from_batch(self):
        """Remove selected additional video from the batch list."""
        row = self._setup_panel.list_batch_videos.currentRow()
        if row == 0:
            QMessageBox.information(
                self,
                "Cannot Remove",
                "The Keystone video cannot be removed from the batch.",
            )
            return
        if row > 0:
            self.batch_videos.pop(row)
            self._sync_batch_list_ui()

    def _export_batch_list(self):
        """Save the current batch video list to a plain-text file (one path per line, keystone first)."""
        if not self.batch_videos:
            QMessageBox.information(
                self,
                "Nothing to Export",
                "The batch list is empty. Add videos first.",
            )
            return

        fp, _ = QFileDialog.getSaveFileName(
            self,
            "Export Batch Video List",
            "",
            "Batch List (*.txt);;All Files (*)",
        )
        if not fp:
            return

        try:
            with open(fp, "w", encoding="utf-8") as fh:
                for path in self.batch_videos:
                    fh.write(path + "\n")
            QMessageBox.information(
                self,
                "Exported",
                f"Batch list saved to:\n{fp}\n\n"
                f"{len(self.batch_videos)} video(s) listed (first = keystone).",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))

    def _import_batch_list(self):
        """Load a batch video list from a plain-text file and set up keystone + additional videos."""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Import Batch Video List",
            "",
            "Batch List (*.txt);;All Files (*)",
        )
        if not fp:
            return

        try:
            with open(fp, "r", encoding="utf-8") as fh:
                lines = [ln.rstrip("\n").strip() for ln in fh if ln.strip()]
        except OSError as exc:
            QMessageBox.critical(self, "Import Failed", str(exc))
            return

        if not lines:
            QMessageBox.warning(
                self, "Empty File", "The selected file contains no video paths."
            )
            return

        missing = [p for p in lines if not os.path.isfile(p)]
        valid = [p for p in lines if os.path.isfile(p)]

        if missing:
            missing_summary = "\n".join(f"  • {p}" for p in missing[:20])
            if len(missing) > 20:
                missing_summary += f"\n  … and {len(missing) - 20} more"

            if not valid:
                QMessageBox.critical(
                    self,
                    "Import Failed – All Paths Missing",
                    f"None of the {len(missing)} path(s) in the file could be found:\n\n{missing_summary}",
                )
                return

            if lines[0] not in valid:
                QMessageBox.critical(
                    self,
                    "Import Failed – Keystone Missing",
                    f"The keystone video (first line) does not exist:\n\n  {lines[0]}\n\n"
                    "Cannot import without a valid keystone.",
                )
                return

            reply = QMessageBox.question(
                self,
                "Missing Videos",
                f"{len(missing)} path(s) could not be found and will be skipped:\n\n"
                f"{missing_summary}\n\nProceed with {len(valid)} valid video(s)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        if not valid:
            QMessageBox.warning(
                self, "Nothing Imported", "No valid video paths were found."
            )
            return

        keystone = valid[0]

        # Activate batch mode if not already on, then load the keystone
        self._setup_panel.g_batch.setChecked(True)

        # Replace the batch list with the imported paths (keystone first)
        self.batch_videos = valid

        # Trigger full keystone setup (loads config, autofills outputs, etc.)
        self._setup_video_file(keystone)

        # Sync the list widget now that keystone is loaded
        self._sync_batch_list_ui()

        QMessageBox.information(
            self,
            "Imported",
            f"Loaded {len(valid)} video(s).\n\nKeystone: {keystone}",
        )

    def select_csv(self: object) -> object:
        """select_csv method documentation."""
        fp, _ = QFileDialog.getSaveFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if fp:
            self._setup_panel.csv_line.setText(fp)

    def select_video_output(self: object) -> object:
        """select_video_output method documentation."""
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self._postprocess_panel.video_out_line.setText(fp)

    # =========================================================================
    # VIDEO PLAYER FUNCTIONS
    # =========================================================================

    def _init_video_player(self, video_path):
        """Initialize video player with the loaded video."""
        # Release any existing video capture
        if self.video_cap is not None:
            self.video_cap.release()

        # Stop any active playback
        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer = None
        self.is_playing = False

        # Open video
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update UI
        self._setup_panel.lbl_video_info.setText(
            f"Video: {self.video_total_frames} frames, {width}x{height}, {fps:.2f} FPS"
        )

        # Enable controls
        self._setup_panel.slider_timeline.setMaximum(self.video_total_frames - 1)
        self._setup_panel.slider_timeline.setEnabled(True)
        self._setup_panel.btn_first_frame.setEnabled(True)
        self._setup_panel.btn_prev_frame.setEnabled(True)
        self._setup_panel.btn_play_pause.setEnabled(True)
        self._setup_panel.btn_next_frame.setEnabled(True)
        self._setup_panel.btn_last_frame.setEnabled(True)
        self._setup_panel.btn_random_seek.setEnabled(True)
        self._setup_panel.combo_playback_speed.setEnabled(True)

        # Enable frame range controls
        self._setup_panel.spin_start_frame.setMaximum(self.video_total_frames - 1)
        self._setup_panel.spin_start_frame.setEnabled(True)
        self._setup_panel.spin_end_frame.setMaximum(self.video_total_frames - 1)
        self._setup_panel.spin_end_frame.setValue(self.video_total_frames - 1)
        self._setup_panel.spin_end_frame.setEnabled(True)
        self._setup_panel.btn_set_start_current.setEnabled(True)
        self._setup_panel.btn_set_end_current.setEnabled(True)
        self._setup_panel.btn_reset_range.setEnabled(True)

        # Show video player group
        self._setup_panel.g_video_player.setVisible(True)

        # Go to first frame
        self.video_current_frame_idx = 0
        self._display_current_frame()
        self._update_range_info()

        logger.info(f"Video player initialized: {self.video_total_frames} frames")

    def _display_current_frame(self):
        """Display the current frame in the video label."""
        if self.video_cap is None:
            return

        # Only seek if not reading sequentially (seeking is slow)
        if self.last_read_frame_idx != self.video_current_frame_idx - 1:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_current_frame_idx)

        ret, frame = self.video_cap.read()

        if not ret:
            return

        self.last_read_frame_idx = self.video_current_frame_idx

        # Convert to RGB and update preview
        self.preview_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detection_test_result = None  # Clear any detection overlay
        self._update_preview_display()

        # Update UI
        self._setup_panel.lbl_current_frame.setText(
            f"Frame: {self.video_current_frame_idx}/{self.video_total_frames - 1}"
        )
        self._setup_panel.slider_timeline.blockSignals(True)
        self._setup_panel.slider_timeline.setValue(self.video_current_frame_idx)
        self._setup_panel.slider_timeline.blockSignals(False)

    def _on_timeline_changed(self, value):
        """Handle timeline slider change."""
        # Only stop playback if this is a manual user change (not from playback itself)
        if self.is_playing and not self._setup_panel.slider_timeline.signalsBlocked():
            self._stop_playback()

        self.video_current_frame_idx = value
        self._display_current_frame()

    def _goto_first_frame(self):
        """Go to the first frame."""
        if self.is_playing:
            self._stop_playback()
        self.video_current_frame_idx = 0
        self._setup_panel.slider_timeline.setValue(0)
        self._display_current_frame()

    def _goto_prev_frame(self):
        """Go to the previous frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_current_frame_idx > 0:
            self.video_current_frame_idx -= 1
            self._setup_panel.slider_timeline.setValue(self.video_current_frame_idx)
            self._display_current_frame()

    def _goto_next_frame(self):
        """Go to the next frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_current_frame_idx < self.video_total_frames - 1:
            self.video_current_frame_idx += 1
            self._setup_panel.slider_timeline.setValue(self.video_current_frame_idx)
            self._display_current_frame()

    def _goto_last_frame(self):
        """Go to the last frame."""
        if self.is_playing:
            self._stop_playback()
        self.video_current_frame_idx = self.video_total_frames - 1
        self._setup_panel.slider_timeline.setValue(self.video_current_frame_idx)
        self._display_current_frame()

    def _goto_random_frame(self):
        """Jump to a random frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_total_frames <= 0:
            return
        self.video_current_frame_idx = np.random.randint(0, self.video_total_frames)
        self._setup_panel.slider_timeline.setValue(self.video_current_frame_idx)
        self._display_current_frame()

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start video playback."""
        if self.video_cap is None or self.is_playing:
            return

        self.is_playing = True
        self._setup_panel.btn_play_pause.setText("⏸ Pause")

        # Get playback speed
        speed_text = self._setup_panel.combo_playback_speed.currentText()
        speed = float(speed_text.replace("x", ""))

        # Calculate interval based on FPS and speed
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default

        interval_ms = max(1, int((1000.0 / fps) / speed))

        # Create timer if needed (use as single-shot timer)
        if self.playback_timer is None:
            self.playback_timer = QTimer(self)

        # Start first frame with single-shot
        self.playback_timer.singleShot(interval_ms, self._playback_step)
        logger.debug(f"Started playback at {speed}x speed ({interval_ms}ms interval)")

    def _stop_playback(self):
        """Stop video playback."""
        if not self.is_playing:
            return

        self.is_playing = False
        self._setup_panel.btn_play_pause.setText("▶ Play")

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()

        logger.debug("Stopped playback")

    def _playback_step(self):
        """Advance one frame during playback."""
        # Stop timer first to prevent event queueing
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()

        # Check if still playing (user might have stopped it)
        if not self.is_playing:
            return

        if self.video_current_frame_idx < self.video_total_frames - 1:
            self.video_current_frame_idx += 1
            self._display_current_frame()

            # Process events to keep UI responsive
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

            # Re-check if still playing after processing events
            if self.is_playing and self.playback_timer:
                # Calculate next interval
                speed_text = self._setup_panel.combo_playback_speed.currentText()
                speed = float(speed_text.replace("x", ""))
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                interval_ms = max(1, int((1000.0 / fps) / speed))

                # Schedule next frame
                self.playback_timer.singleShot(interval_ms, self._playback_step)
        else:
            # Reached end of video
            self._stop_playback()

    def _on_frame_range_changed(self):
        """Handle frame range spinbox changes."""
        # Ensure start <= end
        if (
            self._setup_panel.spin_start_frame.value()
            > self._setup_panel.spin_end_frame.value()
        ):
            self._setup_panel.spin_end_frame.setValue(
                self._setup_panel.spin_start_frame.value()
            )

        self._update_range_info()

    def _update_range_info(self):
        """Update the frame range info label."""
        start = self._setup_panel.spin_start_frame.value()
        end = self._setup_panel.spin_end_frame.value()
        num_frames = end - start + 1

        fps = self._setup_panel.spin_fps.value()
        duration_sec = num_frames / fps if fps > 0 else 0

        self._setup_panel.lbl_range_info.setText(
            f"Tracking {num_frames} frames ({duration_sec:.2f} seconds)"
        )

    def _set_start_to_current(self):
        """Set start frame to current frame."""
        self._setup_panel.spin_start_frame.setValue(self.video_current_frame_idx)

    def _set_end_to_current(self):
        """Set end frame to current frame."""
        self._setup_panel.spin_end_frame.setValue(self.video_current_frame_idx)

    def _reset_frame_range(self):
        """Reset frame range to full video."""
        self._setup_panel.spin_start_frame.setValue(0)
        self._setup_panel.spin_end_frame.setValue(self.video_total_frames - 1)

    def _on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self._detection_panel.label_brightness_val.setText(str(value))
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_contrast_changed(self, value):
        """Handle contrast slider change."""
        contrast_val = value / 100.0
        self._detection_panel.label_contrast_val.setText(f"{contrast_val:.2f}")
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_gamma_changed(self, value):
        """Handle gamma slider change."""
        gamma_val = value / 100.0
        self._detection_panel.label_gamma_val.setText(f"{gamma_val:.2f}")
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

        body_size = self._detection_panel.spin_reference_body_size.value()
        body_area = math.pi * (body_size / 2.0) ** 2
        self._detection_panel.label_body_size_info.setText(
            f"≈ {body_area:.1f} px² area (all size/distance params scale with this)"
        )

    def _update_fps_info(self):
        """Update the FPS info label with time per frame."""
        fps = self._setup_panel.spin_fps.value()
        time_per_frame = 1000.0 / fps  # milliseconds
        self._setup_panel.label_fps_info.setText(f"= {time_per_frame:.2f} ms per frame")

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
            self._detection_panel.label_detection_stats.setText(
                "No detections found.\nAdjust parameters and try again."
            )
            self._detection_panel.btn_auto_set_body_size.setEnabled(False)
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
            "recommended_aspect_ratio": stats["aspect_ratio"]["median"],
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
        self._detection_panel.label_detection_stats.setText(stats_text)
        self._detection_panel.btn_auto_set_body_size.setEnabled(True)
        self._detection_panel.btn_auto_set_aspect_ratio.setEnabled(True)

    def _auto_set_body_size_from_detection(self):
        """Auto-set reference body size from detected geometric mean."""
        if self.detected_sizes is None:
            return

        recommended_size = self.detected_sizes["recommended_body_size"]
        stats = self.detected_sizes["stats"]
        self._detection_panel.spin_reference_body_size.setValue(recommended_size)

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

    def _auto_set_aspect_ratio_from_detection(self):
        """Auto-set reference aspect ratio from detected median."""
        if self.detected_sizes is None:
            return
        recommended_ar = self.detected_sizes["recommended_aspect_ratio"]
        self._detection_panel.spin_reference_aspect_ratio.setValue(recommended_ar)
        QMessageBox.information(
            self,
            "Aspect Ratio Updated",
            f"Reference aspect ratio set to {recommended_ar:.2f}\n"
            f"(median of {self.detected_sizes['count']} detections)\n\n"
            f"Head-tail crop dimensions will adapt to this ratio.\n"
            f"Aspect ratio filtering (if enabled) will use this as the centre.",
        )

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        # Hide/show all video output widgets
        self._postprocess_panel.btn_video_out.setVisible(checked)
        self._postprocess_panel.video_out_line.setVisible(checked)
        self._postprocess_panel.lbl_video_path.setVisible(checked)
        self._postprocess_panel.lbl_video_viz_settings.setVisible(checked)
        self._postprocess_panel.check_show_labels.setVisible(checked)
        self._postprocess_panel.check_show_orientation.setVisible(checked)
        self._postprocess_panel.check_show_trails.setVisible(checked)
        self._postprocess_panel.spin_trail_duration.setVisible(checked)
        self._postprocess_panel.lbl_trail_duration.setVisible(checked)
        self._postprocess_panel.spin_marker_size.setVisible(checked)
        self._postprocess_panel.lbl_marker_size.setVisible(checked)
        self._postprocess_panel.spin_text_scale.setVisible(checked)
        self._postprocess_panel.lbl_text_scale.setVisible(checked)
        self._postprocess_panel.spin_arrow_length.setVisible(checked)
        self._postprocess_panel.lbl_arrow_length.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_settings.setVisible(checked)
        self._postprocess_panel.check_video_show_pose.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_color_mode.setVisible(checked)
        self._postprocess_panel.combo_video_pose_color_mode.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_color_label.setVisible(checked)
        self._postprocess_panel.btn_video_pose_color.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_color.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_point_radius.setVisible(checked)
        self._postprocess_panel.spin_video_pose_point_radius.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_point_thickness.setVisible(checked)
        self._postprocess_panel.spin_video_pose_point_thickness.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_line_thickness.setVisible(checked)
        self._postprocess_panel.spin_video_pose_line_thickness.setVisible(checked)
        self._postprocess_panel.lbl_video_pose_disabled_hint.setVisible(checked)

        # Also control enable state
        self._postprocess_panel.btn_video_out.setEnabled(checked)
        self._postprocess_panel.video_out_line.setEnabled(checked)
        self._sync_video_pose_overlay_controls()

    def _update_preview_display(self):
        """Update the video display with current brightness/contrast/gamma settings."""
        if self.preview_frame_original is None:
            return

        # If we have a detection test result, redisplay it with the new zoom
        if self.detection_test_result is not None:
            self._redisplay_detection_test()
            return

        # Get current adjustment values
        brightness = self._detection_panel.slider_brightness.value()
        contrast = self._detection_panel.slider_contrast.value() / 100.0
        gamma = self._detection_panel.slider_gamma.value() / 100.0

        # Get detection method
        detection_method = self._detection_panel.combo_detection_method.currentText()
        is_background_subtraction = detection_method == "Background Subtraction"

        # Apply adjustments
        from hydra_suite.utils.image_processing import apply_image_adjustments

        if is_background_subtraction:
            # Background subtraction uses grayscale with adjustments
            gray = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2GRAY)
            # GPU not needed for preview - single frame
            adjusted = apply_image_adjustments(
                gray, brightness, contrast, gamma, use_gpu=False
            )
            # Convert back to RGB for display
            adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB)
        else:
            # YOLO uses color frames directly without brightness/contrast/gamma adjustments
            adjusted_rgb = self.preview_frame_original

        # Display the adjusted frame
        h, w, ch = adjusted_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(adjusted_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Apply ROI mask if exists
        if self.roi_mask is not None:
            qimg = self._apply_roi_mask_to_image(qimg)

        # Apply zoom (always use fast transformation for responsive UI)
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
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

        # Apply zoom (always use fast transformation for responsive UI)
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f

        if effective_scale != 1.0:
            orig_h, orig_w = self.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _test_detection_on_preview(self):
        """Test detection algorithm on the current preview frame."""
        if self.preview_frame_original is None:
            logger.warning("No preview frame loaded")
            return

        if self.preview_detection_worker and self.preview_detection_worker.isRunning():
            logger.info("Preview detection is already running")
            return

        # If detection filters are enabled, ask user whether to use them for the test.
        use_detection_filters = False
        detection_filters_enabled = bool(
            self._detection_panel.chk_size_filtering.isChecked()
            or self._detection_panel.chk_enable_aspect_ratio_filtering.isChecked()
        )
        if detection_filters_enabled:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Detection Filter Options")
            msg.setText("Detection filters are currently enabled!")
            msg.setInformativeText(
                "For accurate size estimation, it's recommended to run detection\n"
                "WITHOUT detection constraints. However, you can test with constraints\n"
                "if you want to see how filtering affects the results.\n\n"
                "This includes both size and aspect-ratio filtering.\n\n"
                "How would you like to proceed?"
            )

            btn_without = msg.addButton(
                "NO Detection Filtering (Recommended)", QMessageBox.AcceptRole
            )
            btn_with = msg.addButton("WITH Detection Filtering", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_without)

            msg.exec()
            clicked = msg.clickedButton()

            if clicked == btn_cancel:
                return
            elif clicked == btn_with:
                use_detection_filters = True
                logger.info("Running detection test WITH detection filtering enabled")
            else:  # btn_without
                use_detection_filters = False
                logger.info(
                    "Running detection test WITHOUT detection filtering (recommended for size estimation)"
                )

        context = self._collect_preview_detection_context()
        if (
            int(context.get("detection_method", 0)) == 1
            and str(context.get("yolo_obb_mode", "direct")).strip().lower()
            == "sequential"
        ):
            detect_model = str(context.get("yolo_detect_model_path", "")).strip()
            crop_obb_model = str(context.get("yolo_crop_obb_model_path", "")).strip()
            if not detect_model or not crop_obb_model:
                QMessageBox.warning(
                    self,
                    "Missing Sequential Models",
                    "Sequential YOLO OBB mode in detection preview requires both a detect model and a crop OBB model.",
                )
                return
        self.preview_detection_worker = PreviewDetectionWorker(
            self.preview_frame_original.copy(),
            context,
            use_detection_filters,
        )
        self.preview_detection_worker.finished_signal.connect(
            self._on_preview_detection_finished
        )
        self.preview_detection_worker.error_signal.connect(
            self._on_preview_detection_error
        )
        self.preview_detection_worker.finished.connect(
            self._on_preview_detection_worker_finished
        )
        self._set_preview_test_running(True)
        self.preview_detection_worker.start()

    def _collect_preview_detection_context(self) -> dict:
        """Capture current UI values for async preview detection."""
        # ONNX/TensorRT are not used for single-frame preview detection.
        selected_runtime = self._preview_safe_runtime(self._selected_compute_runtime())
        runtime_detection = derive_detection_runtime_settings(selected_runtime)
        identity_cfg = self._identity_config()
        pose_backend_family = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        runtime_pose = derive_pose_runtime_settings(
            selected_runtime, backend_family=pose_backend_family
        )
        trt_batch_size = (
            self._detection_panel.spin_yolo_batch_size.value()
            if self._runtime_requires_fixed_yolo_batch(selected_runtime)
            else self._detection_panel.spin_tensorrt_batch.value()
        )
        class_text = self._detection_panel.line_yolo_classes.text().strip()
        target_classes = None
        if class_text:
            try:
                target_classes = [int(x.strip()) for x in class_text.split(",")]
            except ValueError:
                target_classes = None

        return {
            "detection_method": self._detection_panel.combo_detection_method.currentIndex(),
            "video_path": self._setup_panel.file_line.text(),
            "bg_prime_seconds": self._detection_panel.spin_bg_prime.value(),
            "fps": self._setup_panel.spin_fps.value(),
            "brightness": self._detection_panel.slider_brightness.value(),
            "contrast": self._detection_panel.slider_contrast.value() / 100.0,
            "gamma": self._detection_panel.slider_gamma.value() / 100.0,
            "roi_mask": self.roi_mask.copy() if self.roi_mask is not None else None,
            "resize_factor": self._setup_panel.spin_resize.value(),
            "dark_on_light": self._detection_panel.chk_dark_on_light.isChecked(),
            "threshold_value": self._detection_panel.spin_threshold.value(),
            "morph_kernel_size": self._detection_panel.spin_morph_size.value(),
            "enable_additional_dilation": self._detection_panel.chk_additional_dilation.isChecked(),
            "dilation_kernel_size": self._detection_panel.spin_dilation_kernel_size.value(),
            "dilation_iterations": self._detection_panel.spin_dilation_iterations.value(),
            "min_contour": self._detection_panel.spin_min_contour.value(),
            "reference_body_size": self._detection_panel.spin_reference_body_size.value(),
            "reference_aspect_ratio": self._detection_panel.spin_reference_aspect_ratio.value(),
            "enable_aspect_ratio_filtering": self._detection_panel.chk_enable_aspect_ratio_filtering.isChecked(),
            "min_aspect_ratio_multiplier": self._detection_panel.spin_min_ar_multiplier.value(),
            "max_aspect_ratio_multiplier": self._detection_panel.spin_max_ar_multiplier.value(),
            "min_object_size": self._detection_panel.spin_min_object_size.value(),
            "max_object_size": self._detection_panel.spin_max_object_size.value(),
            "compute_runtime": selected_runtime,
            "yolo_obb_mode": (
                "sequential"
                if self._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
                else "direct"
            ),
            "yolo_model_path": self._get_selected_yolo_model_path(),
            "yolo_obb_direct_model_path": self._get_selected_yolo_model_path(),
            "yolo_detect_model_path": self._get_selected_yolo_detect_model_path(),
            "yolo_crop_obb_model_path": self._get_selected_yolo_crop_obb_model_path(),
            "yolo_headtail_model_path": self._get_selected_yolo_headtail_model_path(),
            "pose_overrides_headtail": self._identity_panel.chk_pose_overrides_headtail.isChecked(),
            "yolo_seq_crop_pad_ratio": self._detection_panel.spin_yolo_seq_crop_pad.value(),
            "yolo_seq_min_crop_size_px": self._detection_panel.spin_yolo_seq_min_crop_px.value(),
            "yolo_seq_enforce_square_crop": self._detection_panel.chk_yolo_seq_square_crop.isChecked(),
            "yolo_seq_stage2_imgsz": self._detection_panel.spin_yolo_seq_stage2_imgsz.value(),
            "yolo_seq_stage2_pow2_pad": self._detection_panel.chk_yolo_seq_stage2_pow2_pad.isChecked(),
            "yolo_seq_detect_conf_threshold": self._detection_panel.spin_yolo_seq_detect_conf.value(),
            "yolo_headtail_conf_threshold": self._identity_panel.spin_yolo_headtail_conf.value(),
            "yolo_confidence": self._detection_panel.spin_yolo_confidence.value(),
            "yolo_iou": self._detection_panel.spin_yolo_iou.value(),
            "yolo_target_classes": target_classes,
            "yolo_device": runtime_detection["yolo_device"],
            "enable_gpu_background": runtime_detection["enable_gpu_background"],
            "enable_tensorrt": runtime_detection["enable_tensorrt"],
            "enable_onnx_runtime": runtime_detection["enable_onnx_runtime"],
            "tensorrt_max_batch_size": trt_batch_size,
            "max_targets": self._setup_panel.spin_max_targets.value(),
            "max_contour_multiplier": self._detection_panel.spin_max_contour_multiplier.value(),
            "enable_conservative_split": self._detection_panel.chk_conservative_split.isChecked(),
            "conservative_kernel_size": self._detection_panel.spin_conservative_kernel.value(),
            "conservative_erode_iterations": self._detection_panel.spin_conservative_erode.value(),
            "use_apriltags": identity_cfg.get("use_apriltags", False),
            "cnn_classifiers": identity_cfg.get("cnn_classifiers", []),
            "apriltag_family": self._identity_panel.combo_apriltag_family.currentText(),
            "apriltag_decimate": self._identity_panel.spin_apriltag_decimate.value(),
            "enable_pose_extractor": self._is_pose_inference_enabled(),
            "pose_model_type": pose_backend_family,
            "pose_model_dir": resolve_pose_model_path(
                self._pose_model_path_for_backend(pose_backend_family),
                backend=pose_backend_family,
            ),
            "pose_runtime_flavor": runtime_pose["pose_runtime_flavor"],
            "pose_min_kpt_conf_valid": self._identity_panel.spin_pose_min_kpt_conf_valid.value(),
            "pose_skeleton_file": self._identity_panel.line_pose_skeleton_file.text().strip(),
            "pose_ignore_keypoints": self._parse_pose_ignore_keypoints(),
            "pose_direction_anterior_keypoints": self._parse_pose_direction_anterior_keypoints(),
            "pose_direction_posterior_keypoints": self._parse_pose_direction_posterior_keypoints(),
            "pose_batch_size": self._identity_panel.spin_pose_batch.value(),
            "pose_sleap_env": self._selected_pose_sleap_env(),
            "pose_sleap_device": runtime_pose["pose_sleap_device"],
            "pose_sleap_experimental_features": self._sleap_experimental_features_enabled(),
            "individual_crop_padding": self._identity_panel.spin_individual_padding.value(),
            "individual_background_color": [
                int(c) for c in self._identity_panel._background_color
            ],
            "suppress_foreign_obb_regions": self._identity_panel.chk_suppress_foreign_obb.isChecked(),
        }

    def _validate_yolo_model_requirements(self, params: dict, mode_label: str) -> bool:
        """Validate YOLO mode-specific model requirements before starting runs."""
        if str(params.get("DETECTION_METHOD", "")) != "yolo_obb":
            return True

        yolo_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        if yolo_mode != "sequential":
            return True

        detect_model = str(params.get("YOLO_DETECT_MODEL_PATH", "")).strip()
        crop_obb_model = str(params.get("YOLO_CROP_OBB_MODEL_PATH", "")).strip()
        if detect_model and crop_obb_model:
            return True

        QMessageBox.warning(
            self,
            "Missing Sequential Models",
            (
                f"Sequential YOLO OBB mode in {mode_label} requires both a detect model "
                "and a crop OBB model."
            ),
        )
        return False

    def _set_preview_test_running(self, running: bool):
        """Lock/unlock UI while async preview detection is running."""
        if running:
            self._set_interactive_widgets_enabled(False, remember_state=True)
            self._set_video_interaction_enabled(False)
            self.btn_test_detection.setText("Testing Detection...")
            self.btn_test_detection.setEnabled(False)
            self.progress_label.setText("Testing detection on preview...")
            self.progress_label.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setVisible(True)
            return

        self._set_interactive_widgets_enabled(True, remember_state=True)
        self._set_video_interaction_enabled(True)
        self.btn_test_detection.setText("Test Detection on Preview")
        self.btn_test_detection.setEnabled(self.preview_frame_original is not None)
        self.progress_bar.setRange(0, 100)
        self._refresh_progress_visibility()

    @Slot(dict)
    def _on_preview_detection_finished(self, result: dict):
        """Handle successful async preview detection completion."""
        test_frame_rgb = result.get("test_frame_rgb")
        resize_f = float(result.get("resize_factor", 1.0))
        detected_dimensions = result.get("detected_dimensions") or []

        if test_frame_rgb is None:
            logger.warning("Preview detection completed without image result")
            return

        self._update_detection_stats(detected_dimensions, resize_f)
        self.detection_test_result = (test_frame_rgb.copy(), resize_f)

        h, w, ch = test_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f
        if effective_scale != 1.0 and self.preview_frame_original is not None:
            orig_h, orig_w = self.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        self._fit_image_to_screen()
        logger.info("Detection test completed on preview frame")

    @Slot(str)
    def _on_preview_detection_error(self, error_message: str):
        """Handle async preview detection failure."""
        logger.error(f"Detection test failed: {error_message}")
        QMessageBox.warning(
            self,
            "Detection Test Failed",
            "Detection test failed on preview frame. Check logs for details.",
        )

    @Slot()
    def _on_preview_detection_worker_finished(self):
        """Finalize async preview detection UI state and worker lifecycle."""
        sender = self.sender()
        if sender is self.preview_detection_worker:
            try:
                sender.deleteLater()
            except Exception:
                pass
            self.preview_detection_worker = None
        self._set_preview_test_running(False)

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
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        # ROI selection takes priority
        if self.roi_selection_active:
            self.record_roi_click(evt)
            return

        # Pan mode: Left button or Middle button
        from PySide6.QtCore import Qt

        if evt.button() == Qt.LeftButton or evt.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = evt.globalPosition().toPoint()
            self._scroll_start_h = self.scroll.horizontalScrollBar().value()
            self._scroll_start_v = self.scroll.verticalScrollBar().value()
            self.video_label.setCursor(Qt.ClosedHandCursor)
            evt.accept()

    def _handle_video_mouse_move(self, evt):
        """Handle mouse move - update pan if active."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        if self._is_panning and self._pan_start_pos:
            from PySide6.QtCore import Qt

            delta = evt.globalPosition().toPoint() - self._pan_start_pos
            self.scroll.horizontalScrollBar().setValue(self._scroll_start_h - delta.x())
            self.scroll.verticalScrollBar().setValue(self._scroll_start_v - delta.y())
            evt.accept()
        elif not self.roi_selection_active:
            # Show open hand cursor to indicate draggable
            from PySide6.QtCore import Qt

            self.video_label.setCursor(Qt.OpenHandCursor)

    def _handle_video_mouse_release(self, evt):
        """Handle mouse release - end pan."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        from PySide6.QtCore import Qt

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
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        if evt.button() == Qt.LeftButton:
            self._fit_image_to_screen()

    def _handle_video_wheel(self, evt):
        """Handle mouse wheel - zoom in/out."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        from PySide6.QtCore import Qt

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
        from PySide6.QtCore import QEvent

        if evt.type() == QEvent.Gesture:
            if not self._video_interactions_enabled:
                evt.ignore()
                return False
            return self._handle_gesture_event(evt)

        # Always pass non-gesture events through so paint/layout still work
        # even when interactions are disabled (no-video placeholder state).
        return QLabel.event(self.video_label, evt)

    def _handle_gesture_event(self, evt):
        """Handle pinch-to-zoom gesture."""
        if not self._video_interactions_enabled:
            return False
        from PySide6.QtCore import Qt

        # Block gestures during ROI selection
        if self.roi_selection_active:
            return False

        gesture = evt.gesture(Qt.PinchGesture)
        if gesture:

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
        tracking_active = (
            self.tracking_worker is not None and self.tracking_worker.isRunning()
        )

        if tracking_active and self._tracking_frame_size is not None:
            # During tracking/preview, use the actual frame size from the worker
            # These frames are already resized, so use dimensions directly
            effective_width, effective_height = self._tracking_frame_size
        elif self.detection_test_result is not None:
            # Detection test shows resized content
            if self.preview_frame_original is not None:
                h, w = self.preview_frame_original.shape[:2]
                resize_factor = self._setup_panel.spin_resize.value()
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

    def record_roi_click(self: object, evt: object) -> object:
        """record_roi_click method documentation."""
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

        pos = evt.position().toPoint()
        x, y = pos.x(), pos.y()

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

    def update_roi_preview(self: object) -> object:
        """update_roi_preview method documentation."""
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
                from PySide6.QtCore import QPoint

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
            from PySide6.QtCore import QPoint

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

    def start_roi_selection(self: object) -> object:
        """start_roi_selection method documentation."""
        if not self._setup_panel.file_line.text():
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        # Load base frame if not already loaded
        if self.roi_base_frame is None:
            cap = cv2.VideoCapture(self._setup_panel.file_line.text())
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

    def finish_roi_selection(self: object) -> object:
        """finish_roi_selection method documentation."""
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
        if hasattr(Qt, "OpenHandCursor"):
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
            from PySide6.QtCore import QTimer

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

    def undo_last_roi_shape(self: object) -> object:
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

    def clear_roi(self: object) -> object:
        """clear_roi method documentation."""
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
        if hasattr(Qt, "OpenHandCursor"):
            self.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self.video_label.unsetCursor()

        logger.info("All ROI shapes cleared")

    def keyPressEvent(self: object, event: object) -> object:
        """keyPressEvent method documentation."""
        if event.key() == Qt.Key_Escape and self.roi_selection_active:
            self.clear_roi()
        else:
            super().keyPressEvent(event)

    def on_detection_method_changed(self: object, index: object) -> object:
        """on_detection_method_changed method documentation."""
        # Keep compatibility hook and synchronize YOLO-only individual-analysis controls.
        self._sync_individual_analysis_mode_ui()

    def _sanitize_model_token(self, text: object) -> object:
        """Sanitize model metadata token for safe filename use."""
        return _sanitize_model_token(text)

    def _format_yolo_model_label(self, model_path: object) -> object:
        """Build combo-box label for a model path, including metadata if available."""
        rel_path = make_model_path_relative(model_path)
        filename = os.path.basename(rel_path)
        metadata = get_yolo_model_metadata(rel_path) or {}

        size = metadata.get("size") or metadata.get("model_size")
        species = metadata.get("species")
        model_info = metadata.get("model_info")
        task_family = str(metadata.get("task_family", "")).strip().lower()
        usage_role = str(metadata.get("usage_role", "")).strip().lower()
        model_id = None
        if species and model_info:
            model_id = f"{species}_{model_info}"
        elif species:
            model_id = species
        suffix_parts = []
        if size:
            suffix_parts.append(str(size))
        if model_id:
            suffix_parts.append(str(model_id))
        if usage_role:
            suffix_parts.append(usage_role)
        elif task_family:
            suffix_parts.append(task_family)
        if suffix_parts:
            return f"{filename} ({', '.join(suffix_parts)})"
        return filename

    @staticmethod
    def _yolo_model_matches_filter(
        metadata: object, task_family: str | None = None, usage_role: str | None = None
    ) -> bool:
        if not isinstance(metadata, dict):
            return True
        meta_task = str(metadata.get("task_family", "")).strip().lower()
        meta_role = str(metadata.get("usage_role", "")).strip().lower()
        if not meta_task and not meta_role:
            # Backward compatibility for legacy entries with no role/type metadata.
            return True
        if task_family and meta_task and meta_task != task_family:
            return False
        if usage_role and meta_role and meta_role != usage_role:
            return False
        return True

    def _populate_yolo_model_combo(
        self,
        combo: object,
        preferred_model_path: object = None,
        default_path: str = "",
        include_none: bool = False,
        task_family: str | None = None,
        usage_role: str | None = None,
        repository_dir: str | None = None,
        recursive: bool = False,
    ) -> object:
        """Populate a YOLO-model combo with optional metadata role filtering."""
        selected_path = preferred_model_path
        if selected_path is None:
            selected_data = combo.currentData(Qt.UserRole)
            if selected_data and selected_data not in ("__add_new__", "__none__"):
                selected_path = str(selected_data)

        entries = {}

        models_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        try:
            if recursive:
                local_model_paths = []
                for dirpath, _dirnames, filenames in os.walk(models_dir):
                    for fn in sorted(filenames):
                        if os.path.splitext(fn)[1].lower() in (".pt", ".pth"):
                            local_model_paths.append(os.path.join(dirpath, fn))
            else:
                local_model_paths = sorted(
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if os.path.splitext(f)[1].lower() in (".pt", ".pth")
                )
        except Exception as e:
            logger.warning(f"Failed to list YOLO model directory '{models_dir}': {e}")
            local_model_paths = []

        for model_abs in local_model_paths:
            rel_path = make_model_path_relative(model_abs)
            metadata = get_yolo_model_metadata(rel_path) or {}
            if not self._yolo_model_matches_filter(
                metadata, task_family=task_family, usage_role=usage_role
            ):
                continue
            entries[rel_path] = self._format_yolo_model_label(rel_path)

        combo.blockSignals(True)
        combo.clear()
        for model_path, label in entries.items():
            combo.addItem(label, model_path)
        if include_none:
            combo.insertItem(0, "— None —", "__none__")
        combo.addItem("＋ Add New Model…", "__add_new__")
        combo.blockSignals(False)

        self._set_model_selection_for_selector(
            combo,
            selected_path,
            default_path=default_path,
        )

    def _set_model_selection_for_selector(
        self,
        combo: object,
        model_path: object,
        default_path: str = "",
    ) -> object:
        target_path = make_model_path_relative(model_path or "")
        if not target_path:
            target_path = str(default_path or "")

        for i in range(combo.count()):
            item_data = combo.itemData(i, Qt.UserRole)
            if item_data == target_path:
                combo.setCurrentIndex(i)
                return

        # Target not found — fall back to "— None —" or first item.
        none_idx = combo.findData("__none__", Qt.UserRole)
        if none_idx >= 0:
            combo.setCurrentIndex(none_idx)
        else:
            combo.setCurrentIndex(0)

    def _get_selected_model_path_from_selector(
        self, combo: object, default_path: str = ""
    ) -> object:
        selected_data = combo.currentData(Qt.UserRole)
        if selected_data and selected_data not in ("__add_new__", "__none__"):
            return str(selected_data)
        return str(default_path or "")

    def _import_yolo_model_to_repository(
        self,
        source_path: object,
        task_family: str | None = None,
        usage_role: str | None = None,
        repository_dir: str | None = None,
    ) -> object:
        """Import a YOLO model file into models/obb with metadata."""
        src = str(source_path or "")
        if not src or not os.path.exists(src):
            return None

        src_abs = os.path.abspath(src)
        models_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        try:
            rel_existing = os.path.relpath(src_abs, models_dir)
            if not rel_existing.startswith(".."):
                return make_model_path_relative(src_abs)
        except Exception:
            pass

        # Single dialog for metadata collection (size + species + model info + timestamp preview)
        now_preview = datetime.now()
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        size_combo = QComboBox(dlg)
        size_combo.addItems(["26n", "26s", "26m", "26l", "26x", "custom", "unknown"])
        size_combo.setCurrentText("26s")
        dlg_form.addRow("YOLO model size:", size_combo)

        stem_tokens = [t for t in Path(src).stem.replace("-", "_").split("_") if t]
        default_species = (
            self._sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            self._sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_size = size_combo.currentText().strip() or "unknown"
        model_species = self._sanitize_model_token(species_line.text())
        model_info = self._sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp_token = now.strftime("%Y%m%d-%H%M%S")
        added_at = now.isoformat(timespec="seconds")
        ext = os.path.splitext(src)[1].lower() or ".pt"

        model_slug = f"{model_species}_{model_info}"
        base_name = f"{timestamp_token}_{model_size}_{model_slug}"
        dest_path = os.path.join(models_dir, f"{base_name}{ext}")
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(models_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        try:
            shutil.copy2(src, dest_path)
        except Exception as e:
            logger.error(f"Failed to copy model to repository: {e}")
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Could not import model into repository:\n{e}",
            )
            return None

        rel_path = make_model_path_relative(dest_path)
        metadata = {
            "size": model_size,
            "species": model_species,
            "model_info": model_info,
            "added_at": added_at,
            "source_path": src,
            "stored_filename": os.path.basename(dest_path),
        }
        if task_family:
            metadata["task_family"] = str(task_family).strip().lower()
        if usage_role:
            metadata["usage_role"] = str(usage_role).strip().lower()
        register_yolo_model(rel_path, metadata)
        logger.info(f"Imported model to repository: {dest_path}")
        return rel_path

    def _refresh_yolo_model_combo(self, preferred_model_path: object = None) -> object:
        """Populate direct OBB model combo from repository models."""
        self._populate_yolo_model_combo(
            self._detection_panel.combo_yolo_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=False,
            task_family="obb",
            usage_role="obb_direct",
            repository_dir=get_yolo_model_repository_directory(
                task_family="obb", usage_role="obb_direct"
            ),
        )

    def _refresh_yolo_detect_model_combo(self, preferred_model_path: object = None):
        self._populate_yolo_model_combo(
            self._detection_panel.combo_yolo_detect_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="detect",
            usage_role="seq_detect",
            repository_dir=get_yolo_model_repository_directory(
                task_family="detect", usage_role="seq_detect"
            ),
        )

    def _refresh_yolo_crop_obb_model_combo(self, preferred_model_path: object = None):
        self._populate_yolo_model_combo(
            self._detection_panel.combo_yolo_crop_obb_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="obb",
            usage_role="seq_crop_obb",
            repository_dir=get_yolo_model_repository_directory(
                task_family="obb", usage_role="seq_crop_obb"
            ),
        )

    def _refresh_yolo_headtail_model_combo(self, preferred_model_path: object = None):
        ht_type = getattr(self, "combo_yolo_headtail_model_type", None)
        subdir = ht_type.currentText() if ht_type else "YOLO"
        repo_dir = os.path.join(
            get_yolo_model_repository_directory(
                task_family="classify", usage_role="headtail"
            ),
            subdir,
        )
        os.makedirs(repo_dir, exist_ok=True)
        self._populate_yolo_model_combo(
            self._identity_panel.combo_yolo_headtail_model,
            preferred_model_path=preferred_model_path,
            default_path="",
            include_none=True,
            task_family="classify",
            usage_role="headtail",
            repository_dir=repo_dir,
        )

    @staticmethod
    def _infer_yolo_headtail_model_type(model_path: object) -> str:
        """Infer the head-tail model family from its stored path."""
        normalized = str(make_model_path_relative(model_path or "")).replace("\\", "/")
        normalized_lower = f"/{normalized.lower().strip('/')}" if normalized else ""
        if "/tiny/" in normalized_lower:
            return "tiny"
        return "YOLO"

    def _populate_pose_model_combo(
        self,
        combo: object,
        backend: str,
        preferred_model_path: object = None,
    ) -> None:
        """Populate the pose model combo for the given backend."""
        selected_path = preferred_model_path
        if selected_path is None:
            selected_data = combo.currentData(Qt.UserRole)
            if selected_data and selected_data not in ("__add_new__", "__none__"):
                selected_path = str(selected_data)

        backend_key = (
            "sleap"
            if backend == "sleap"
            else ("vitpose" if backend == "vitpose" else "yolo")
        )
        repo_dir = get_pose_models_directory(backend_key)
        os.makedirs(repo_dir, exist_ok=True)

        entries = {}
        try:
            if backend_key == "sleap":
                for name in sorted(os.listdir(repo_dir)):
                    full = os.path.join(repo_dir, name)
                    if os.path.isdir(full):
                        rel = make_pose_model_path_relative(full)
                        entries[rel] = name
            else:
                for fn in sorted(os.listdir(repo_dir)):
                    if os.path.splitext(fn)[1].lower() in (".pt", ".pth"):
                        full = os.path.join(repo_dir, fn)
                        rel = make_pose_model_path_relative(full)
                        entries[rel] = self._format_yolo_model_label(rel)
        except Exception as e:
            logger.warning(f"Failed to list pose model directory '{repo_dir}': {e}")

        combo.blockSignals(True)
        combo.clear()
        for path, label in entries.items():
            combo.addItem(label, path)
        combo.insertItem(0, "— None —", "__none__")
        combo.addItem("＋ Add New Model…", "__add_new__")
        combo.blockSignals(False)

        self._set_model_selection_for_selector(combo, selected_path, default_path="")

    def _refresh_pose_model_combo(self, preferred_model_path: object = None) -> None:
        """Refresh the pose model combo for the current backend."""
        if not hasattr(self, "_identity_panel"):
            return
        backend = self._current_pose_backend_key()
        self._populate_pose_model_combo(
            self._identity_panel.combo_pose_model,
            backend=backend,
            preferred_model_path=preferred_model_path,
        )

    def on_pose_model_changed(self, index: int) -> None:
        """Handle selection change in the pose model combo."""
        if not hasattr(self, "_identity_panel"):
            return
        selected_data = self._identity_panel.combo_pose_model.itemData(
            index, Qt.UserRole
        )
        if selected_data == "__add_new__":
            self._handle_add_new_pose_model()
            return
        path = selected_data if selected_data and selected_data != "__none__" else ""
        self._set_pose_model_path_for_backend(
            path, backend=self._current_pose_backend_key(), update_combo=False
        )

    def _get_selected_yolo_model_path(self) -> object:
        """Return currently selected direct OBB model path."""
        if not hasattr(self, "_detection_panel"):
            return ""
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_model,
            default_path="",
        )

    def _get_selected_yolo_detect_model_path(self) -> object:
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_detect_model,
            default_path="",
        )

    def _get_selected_yolo_crop_obb_model_path(self) -> object:
        return self._get_selected_model_path_from_selector(
            self._detection_panel.combo_yolo_crop_obb_model,
            default_path="",
        )

    def _get_selected_yolo_headtail_model_path(self) -> object:
        return self._get_selected_model_path_from_selector(
            self._identity_panel.combo_yolo_headtail_model,
            default_path="",
        )

    def _set_yolo_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_model,
            model_path,
        )

    def _set_yolo_detect_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_detect_model,
            model_path,
        )

    def _set_yolo_crop_obb_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._detection_panel.combo_yolo_crop_obb_model,
            model_path,
        )

    def _set_yolo_headtail_model_selection(self, model_path: object) -> object:
        self._set_model_selection_for_selector(
            self._identity_panel.combo_yolo_headtail_model,
            model_path,
        )

    def _on_yolo_mode_changed(self, _index: object) -> object:
        """Toggle direct/sequential model controls."""
        form = (
            self._detection_panel.yolo_group.layout()
            if hasattr(self, "_detection_panel")
            else None
        )

        def _set_row_visible(widget: object, visible: bool):
            if widget is None:
                return
            widget.setVisible(bool(visible))
            if form is None:
                return
            try:
                label = form.labelForField(widget)
            except Exception:
                label = None
            if label is not None:
                label.setVisible(bool(visible))

        sequential = (
            hasattr(self, "_detection_panel")
            and self._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
        )

        _set_row_visible(
            (
                getattr(self._detection_panel, "combo_yolo_model", None)
                if hasattr(self, "_detection_panel")
                else None
            ),
            not sequential,
        )

        _set_row_visible(
            (
                getattr(self._detection_panel, "combo_yolo_detect_model", None)
                if hasattr(self, "_detection_panel")
                else None
            ),
            sequential,
        )
        _set_row_visible(
            (
                getattr(self._detection_panel, "combo_yolo_crop_obb_model", None)
                if hasattr(self, "_detection_panel")
                else None
            ),
            sequential,
        )
        _set_row_visible(
            (
                getattr(self._detection_panel, "yolo_seq_advanced", None)
                if hasattr(self, "_detection_panel")
                else None
            ),
            sequential,
        )
        _set_row_visible(
            getattr(self, "headtail_model_row_widget", None),
            True,
        )
        _set_row_visible(getattr(self, "chk_pose_overrides_headtail", None), True)
        if hasattr(self, "_identity_panel"):
            self._identity_panel.spin_yolo_headtail_conf.setEnabled(
                bool(self._get_selected_yolo_headtail_model_path().strip())
            )
        self._update_obb_mode_warning()

    def _on_headtail_model_type_changed(self, _index: object = None) -> None:
        """Refresh the head-tail model combo when the user switches YOLO ↔ tiny."""
        self._refresh_yolo_headtail_model_combo()

    def _update_obb_mode_warning(self) -> None:
        """Show a performance hint when device/mode is a suboptimal combination."""
        if not hasattr(self, "_detection_panel"):
            return
        runtime = (
            self._selected_compute_runtime() if hasattr(self, "_setup_panel") else ""
        )
        sequential = (
            hasattr(self, "_detection_panel")
            and self._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
        )
        is_mps = "mps" in runtime.lower()
        is_cuda = "cuda" in runtime.lower()
        if is_mps and sequential:
            msg = (
                "⚠ Sequential mode is significantly slower on Apple Silicon (MPS). "
                "Direct mode is recommended for MPS — it runs ~4× faster."
            )
        elif is_cuda and not sequential:
            msg = (
                "⚠ Sequential mode is typically faster on CUDA GPUs. "
                "Consider switching to Sequential for better throughput."
            )
        else:
            msg = ""
        self._detection_panel.lbl_obb_mode_warning.setText(msg)
        self._detection_panel.lbl_obb_mode_warning.setVisible(bool(msg))

    def on_yolo_model_changed(self: object, index: object) -> object:
        """Handle direct OBB model selection — triggers import when 'Add New' chosen."""
        if (
            self._detection_panel.combo_yolo_model.itemData(index, Qt.UserRole)
            == "__add_new__"
        ):
            self._handle_add_new_yolo_model(
                combo=self._detection_panel.combo_yolo_model,
                refresh_callback=self._refresh_yolo_model_combo,
                selection_callback=self._set_yolo_model_selection,
                task_family="obb",
                usage_role="obb_direct",
                dialog_title="Add Direct OBB Model",
            )
            return
        self._on_yolo_mode_changed(index)

    def on_yolo_detect_model_changed(self: object, index: object) -> object:
        if (
            self._detection_panel.combo_yolo_detect_model.itemData(index, Qt.UserRole)
            == "__add_new__"
        ):
            self._handle_add_new_yolo_model(
                combo=self._detection_panel.combo_yolo_detect_model,
                refresh_callback=self._refresh_yolo_detect_model_combo,
                selection_callback=self._set_yolo_detect_model_selection,
                task_family="detect",
                usage_role="seq_detect",
                dialog_title="Add Sequential Detect Model",
            )
            return
        self._on_yolo_mode_changed(index)

    def on_yolo_crop_obb_model_changed(self: object, index: object) -> object:
        if (
            self._detection_panel.combo_yolo_crop_obb_model.itemData(index, Qt.UserRole)
            == "__add_new__"
        ):
            self._handle_add_new_yolo_model(
                combo=self._detection_panel.combo_yolo_crop_obb_model,
                refresh_callback=self._refresh_yolo_crop_obb_model_combo,
                selection_callback=self._set_yolo_crop_obb_model_selection,
                task_family="obb",
                usage_role="seq_crop_obb",
                dialog_title="Add Sequential Crop OBB Model",
            )
            return
        self._on_yolo_mode_changed(index)
        self._apply_crop_obb_training_params()

    def _apply_crop_obb_training_params(self):
        """Auto-configure sequential inference params from model training metadata."""
        model_path = self._get_selected_yolo_crop_obb_model_path()
        if not model_path:
            return
        meta = get_yolo_model_metadata(model_path) or {}
        tp = meta.get("training_params")
        if not isinstance(tp, dict):
            return

        applied = []
        if "imgsz" in tp:
            val = int(tp["imgsz"])
            self._detection_panel.spin_yolo_seq_stage2_imgsz.setValue(val)
            applied.append(f"stage2_imgsz={val}")
        if "crop_pad_ratio" in tp:
            val = float(tp["crop_pad_ratio"])
            self._detection_panel.spin_yolo_seq_crop_pad.setValue(val)
            applied.append(f"crop_pad={val}")
        if "min_crop_size_px" in tp:
            val = int(tp["min_crop_size_px"])
            self._detection_panel.spin_yolo_seq_min_crop_px.setValue(val)
            applied.append(f"min_crop={val}")
        if "enforce_square" in tp:
            val = bool(tp["enforce_square"])
            self._detection_panel.chk_yolo_seq_square_crop.setChecked(val)
            applied.append(f"square={val}")
        if applied:
            logger.info(
                "Auto-configured sequential params from model metadata: %s",
                ", ".join(applied),
            )

    def on_yolo_headtail_model_changed(self: object, index: object) -> object:
        if (
            self._identity_panel.combo_yolo_headtail_model.itemData(index, Qt.UserRole)
            == "__add_new__"
        ):
            ht_type = getattr(self, "combo_yolo_headtail_model_type", None)
            subdir = ht_type.currentText() if ht_type else "YOLO"
            repo_dir = os.path.join(
                get_yolo_model_repository_directory(
                    task_family="classify", usage_role="headtail"
                ),
                subdir,
            )
            os.makedirs(repo_dir, exist_ok=True)
            self._handle_add_new_yolo_model(
                combo=self._identity_panel.combo_yolo_headtail_model,
                refresh_callback=self._refresh_yolo_headtail_model_combo,
                selection_callback=self._set_yolo_headtail_model_selection,
                task_family="classify",
                usage_role="headtail",
                dialog_title="Add Head-Tail Classifier",
                repository_dir=repo_dir,
            )
            return
        self._on_yolo_mode_changed(index)
        self._sync_individual_analysis_mode_ui()

    def _handle_add_new_yolo_model(
        self,
        combo: object,
        refresh_callback: object,
        selection_callback: object,
        task_family: str,
        usage_role: str,
        dialog_title: str,
        repository_dir: str | None = None,
    ) -> object:
        """Browse for a model, import it, refresh the combo, and select it.

        Restores the previous selection if the user cancels.
        """
        # Remember what was selected before this action item was triggered.
        prev_data = None
        for i in range(combo.count()):
            d = combo.itemData(i, Qt.UserRole)
            if d not in ("__add_new__", "__none__", None):
                prev_data = d
                break

        start_dir = str(
            repository_dir
            or get_yolo_model_repository_directory(
                task_family=task_family, usage_role=usage_role
            )
        )
        fp, _ = QFileDialog.getOpenFileName(
            self,
            dialog_title,
            start_dir,
            "PyTorch Model Files (*.pt *.pth);;All Files (*)",
        )
        if not fp:
            # Cancelled — restore previous selection silently.
            combo.blockSignals(True)
            self._set_model_selection_for_selector(combo, prev_data)
            combo.blockSignals(False)
            return

        selected_abs = os.path.abspath(fp)
        try:
            rel_existing = os.path.relpath(selected_abs, start_dir)
            is_in_repo = not rel_existing.startswith("..")
        except (ValueError, TypeError):
            is_in_repo = False

        if is_in_repo:
            final_path = make_model_path_relative(selected_abs)
        else:
            final_path = self._import_yolo_model_to_repository(
                selected_abs,
                task_family=task_family,
                usage_role=usage_role,
                repository_dir=start_dir,
            )
            if not final_path:
                combo.blockSignals(True)
                self._set_model_selection_for_selector(combo, prev_data)
                combo.blockSignals(False)
                return
            QMessageBox.information(
                self,
                "Model Added",
                f"Model added to repository:\n{os.path.basename(final_path)}",
            )

        refresh_callback(preferred_model_path=final_path)
        selection_callback(final_path)

    def toggle_preview(self: object, checked: object) -> object:
        """toggle_preview method documentation."""
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

            if msg.exec() == QMessageBox.Ok:
                self.start_tracking(preview_mode=True)
                self.btn_preview.setText("Stop Preview")
                self.btn_start.setEnabled(False)
            else:
                self.btn_preview.setChecked(False)
        else:
            self.stop_tracking()
            self.btn_preview.setText("Preview Mode")
            self.btn_start.setEnabled(True)

    def toggle_tracking(self: object, checked: object) -> object:
        """toggle_tracking method documentation."""
        if checked:
            # If preview is active, stop it first
            if self.btn_preview.isChecked():
                self.btn_preview.setChecked(False)
                self.btn_preview.setText("Preview Mode")
                self.stop_tracking()

            self.btn_start.setText("Stop Tracking")
            self.btn_preview.setEnabled(False)
            self.start_full()

            if not (self.tracking_worker and self.tracking_worker.isRunning()):
                # Tracking did not start (e.g., no video or config save cancelled)
                self.btn_start.blockSignals(True)
                self.btn_start.setChecked(False)
                self.btn_start.blockSignals(False)
                self.btn_start.setText("Start Full Tracking")
                self.btn_preview.setEnabled(True)
        else:
            self.stop_tracking()

    def toggle_debug_logging(self: object, checked: object) -> object:
        """toggle_debug_logging method documentation."""
        if checked:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging disabled")

    def _on_visualization_mode_changed(self, state):
        """Handle visualization-free mode toggle."""
        is_viz_free = self._setup_panel.chk_visualization_free.isChecked()
        is_preview_active = self.btn_preview.isChecked()
        is_tracking_active = self.tracking_worker and self.tracking_worker.isRunning()

        # Keep display settings visible; only gate their effect at runtime
        self._setup_panel.g_display.setVisible(True)

        # Keep individual checkboxes enabled for pre-configuration
        self._setup_panel.chk_show_circles.setEnabled(True)
        self._setup_panel.chk_show_orientation.setEnabled(True)
        self._setup_panel.chk_show_trajectories.setEnabled(True)
        self._setup_panel.chk_show_labels.setEnabled(True)
        self._setup_panel.chk_show_state.setEnabled(True)
        self._setup_panel.chk_show_kalman_uncertainty.setEnabled(True)
        self._detection_panel.chk_show_fg.setEnabled(True)
        self._detection_panel.chk_show_bg.setEnabled(True)
        self._detection_panel.chk_show_yolo_obb.setEnabled(True)

        # Only affect display during active tracking (not setup/preview)
        if is_tracking_active and is_viz_free and not is_preview_active:
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
            self.video_label.setStyleSheet("color: #9a9a9a; font-size: 14px;")
            logger.info("Visualization-Free Mode enabled - Maximum speed processing")
        elif is_tracking_active and not is_viz_free:
            # Restore previous state or default message
            if hasattr(self, "_stored_preview_text") and self._stored_preview_text:
                self.video_label.setText(self._stored_preview_text)
            elif not self.video_label.pixmap():
                self._show_video_logo_placeholder()
            self.video_label.setStyleSheet("color: #6a6a6a; font-size: 16px;")

    def start_full(self: object) -> object:
        """start_full method documentation."""
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            self.stop_tracking()

        # Set up comprehensive session logging once for entire tracking session
        video_path = self._setup_panel.file_line.text()
        if video_path:
            self._setup_session_logging(video_path, backward_mode=False)
            from datetime import datetime

            self._individual_dataset_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_detection_cache_path = None
            self.current_individual_properties_cache_path = None
            self.current_interpolated_roi_npz_path = None
            self.current_interpolated_pose_csv_path = None
            self.current_interpolated_pose_df = None
            self.current_interpolated_tag_csv_path = None
            self.current_interpolated_tag_df = None
            self.current_interpolated_cnn_csv_paths = {}
            self.current_interpolated_cnn_dfs = {}
            self.current_interpolated_headtail_csv_path = None
            self.current_interpolated_headtail_df = None
            self._pending_pose_export_csv_path = None
            self._pending_video_csv_path = None
            self._pending_video_generation = False
            self._pending_finish_after_track_videos = False

        self.start_tracking(preview_mode=False)

    def _request_qthread_stop(
        self,
        worker,
        worker_name: str,
        *,
        timeout_ms: int = 1500,
        force_terminate: bool = True,
    ) -> None:
        """Stop a QThread cooperatively, then force terminate if needed."""
        if worker is None:
            return
        try:
            if not worker.isRunning():
                return
        except Exception:
            return

        try:
            if hasattr(worker, "stop"):
                worker.stop()
        except Exception:
            logger.debug("Failed to call stop() on %s", worker_name, exc_info=True)

        try:
            worker.requestInterruption()
        except Exception:
            pass

        stopped = False
        try:
            stopped = bool(worker.wait(int(timeout_ms)))
        except Exception:
            stopped = False

        if stopped:
            logger.info("%s stopped.", worker_name)
            return

        if not force_terminate:
            logger.warning(
                "%s did not stop within %d ms (cooperative stop only).",
                worker_name,
                int(timeout_ms),
            )
            return

        logger.warning(
            "%s did not stop cooperatively; forcing terminate().", worker_name
        )
        try:
            worker.terminate()
        except Exception:
            logger.debug("terminate() failed for %s", worker_name, exc_info=True)
        try:
            worker.wait(max(500, int(timeout_ms)))
        except Exception:
            pass

    def _stop_csv_writer(self, timeout_sec: float = 2.0) -> None:
        """Stop background CSV writer thread safely without indefinite blocking."""
        writer = self.csv_writer_thread
        if writer is None:
            return
        try:
            writer.stop()
        except Exception:
            logger.debug("Failed to request CSV writer stop.", exc_info=True)
        try:
            if writer.is_alive():
                writer.join(timeout=timeout_sec)
                if writer.is_alive():
                    logger.warning("CSV writer did not stop within %.1fs.", timeout_sec)
        except Exception:
            logger.debug("Failed to join CSV writer thread.", exc_info=True)
        finally:
            self.csv_writer_thread = None

    def _cleanup_thread_reference(self, attr_name: str) -> None:
        """Delete finished QThread references safely."""
        worker = getattr(self, attr_name, None)
        if worker is None:
            return
        try:
            running = bool(worker.isRunning())
        except Exception:
            running = False
        if not running:
            try:
                worker.deleteLater()
            except Exception:
                pass
            setattr(self, attr_name, None)

    def stop_tracking(self: object) -> object:
        """stop_tracking method documentation."""
        self._stop_all_requested = True
        self._pending_finish_after_interp = False
        self._pending_finish_after_track_videos = False
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False

        # Stop all active workers and subprocess-like threads.
        self._request_qthread_stop(
            getattr(self, "_cache_builder_worker", None), "DetectionCacheBuilderWorker"
        )
        self._request_qthread_stop(
            getattr(self, "merge_worker", None), "MergeWorker", timeout_ms=1200
        )
        self._request_qthread_stop(self.dataset_worker, "DatasetGenerationWorker")
        self._request_qthread_stop(self.interp_worker, "InterpolatedCropsWorker")
        self._request_qthread_stop(
            self.oriented_video_worker, "OrientedTrackVideoWorker"
        )
        self._request_qthread_stop(self.tracking_worker, "TrackingWorker")
        self._stop_csv_writer()

        self._cleanup_thread_reference("_cache_builder_worker")
        self._cleanup_thread_reference("merge_worker")
        self._cleanup_thread_reference("dataset_worker")
        self._cleanup_thread_reference("interp_worker")
        self._cleanup_thread_reference("oriented_video_worker")

        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self._set_ui_controls_enabled(True)
        # Ensure UI state is restored after stopping
        if self.current_video_path:
            self._apply_ui_state("idle")
        else:
            self._apply_ui_state("no_video")
        self.btn_preview.setChecked(False)
        self.btn_preview.setText("Preview Mode")
        self.btn_start.blockSignals(True)
        self.btn_start.setChecked(False)
        self.btn_start.blockSignals(False)
        self.btn_start.setText("Start Full Tracking")
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self._individual_dataset_run_id = None
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None
        self.current_interpolated_roi_npz_path = None
        self.current_interpolated_pose_csv_path = None
        self.current_interpolated_pose_df = None
        self.current_interpolated_tag_csv_path = None
        self.current_interpolated_tag_df = None
        self.current_interpolated_cnn_csv_paths = {}
        self.current_interpolated_cnn_dfs = {}
        self.current_interpolated_headtail_csv_path = None
        self.current_interpolated_headtail_df = None

        # Hide stats labels when tracking stops
        self.label_current_fps.setVisible(False)
        self.label_elapsed_time.setVisible(False)
        self.label_eta.setVisible(False)

        # Reset tracking frame size
        self._tracking_frame_size = None
        self._cleanup_session_logging()

    def _set_ui_controls_enabled(self, enabled: bool):
        if enabled:
            if self.current_video_path:
                self._apply_ui_state("idle")
            else:
                self._apply_ui_state("no_video")
            return

        # Disabled state - choose mode based on tracking/preview status
        if self.tracking_worker and self.tracking_worker.isRunning():
            if self.btn_preview.isChecked():
                self._apply_ui_state("preview")
            else:
                self._apply_ui_state("tracking")
        else:
            self._apply_ui_state("locked")

    def _collect_preview_controls(self):
        return [
            self.btn_test_detection,
            self._setup_panel.slider_timeline,
            self._setup_panel.btn_first_frame,
            self._setup_panel.btn_prev_frame,
            self._setup_panel.btn_play_pause,
            self._setup_panel.btn_next_frame,
            self._setup_panel.btn_last_frame,
            self._setup_panel.btn_random_seek,
            self._setup_panel.combo_playback_speed,
            self._setup_panel.spin_start_frame,
            self._setup_panel.spin_end_frame,
            self._setup_panel.btn_set_start_current,
            self._setup_panel.btn_set_end_current,
            self._setup_panel.btn_reset_range,
        ]

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        allow = set(allowlist or [])
        block = set(blocklist or [])
        interactive_types = (
            QAbstractButton,
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QSlider,
        )
        # Exclude welcome-page widgets — they are managed by WelcomePage itself
        welcome = getattr(self, "_welcome_page", None)
        widgets = []
        for widget_type in interactive_types:
            for w in self.findChildren(widget_type):
                if welcome is not None and welcome.isAncestorOf(w):
                    continue
                widgets.append(w)

        if enabled and remember_state and self._saved_widget_enabled_states:
            for widget in widgets:
                if widget in block:
                    widget.setEnabled(False)
                elif widget in allow:
                    widget.setEnabled(True)
                elif widget in self._saved_widget_enabled_states:
                    widget.setEnabled(self._saved_widget_enabled_states[widget])
            self._saved_widget_enabled_states = {}
            return

        if not enabled and remember_state:
            for widget in widgets:
                if widget in block or widget in allow:
                    continue
                self._saved_widget_enabled_states[widget] = widget.isEnabled()

        for widget in widgets:
            if widget in block:
                widget.setEnabled(False)
            elif widget in allow:
                widget.setEnabled(True)
            else:
                widget.setEnabled(enabled)

    def _set_video_interaction_enabled(self, enabled: bool):
        self._video_interactions_enabled = enabled
        self.slider_zoom.setEnabled(enabled)
        # Keep the viewport enabled so placeholder/logo rendering is not dimmed
        # by disabled-widget styling (notably on macOS).
        self.scroll.setEnabled(True)
        if not enabled:
            self.video_label.unsetCursor()

    def _prepare_tracking_display(self):
        """Clear any stale frame before tracking starts."""
        self.video_label.clear()
        if self._is_visualization_enabled():
            self.video_label.setText("")
            self.video_label.setStyleSheet("color: #6a6a6a; font-size: 16px;")
        else:
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #9a9a9a; font-size: 14px;")

    def _show_video_logo_placeholder(self):
        """Show HYDRA logo in the video panel when no video is loaded."""
        try:
            from PySide6.QtCore import QByteArray

            from hydra_suite.paths import get_brand_icon_bytes

            logo_data = get_brand_icon_bytes("trackerkit.svg")
            vw = max(640, self.scroll.viewport().width())
            vh = max(420, self.scroll.viewport().height())
            canvas = QPixmap(vw, vh)
            canvas.fill(QColor(0, 0, 0, 0))

            renderer = (
                QSvgRenderer(QByteArray(logo_data)) if logo_data else QSvgRenderer()
            )
            if renderer.isValid():
                view_box = renderer.viewBoxF()
                if view_box.isEmpty():
                    default_size = renderer.defaultSize()
                    view_box = QRectF(
                        0,
                        0,
                        max(1, default_size.width()),
                        max(1, default_size.height()),
                    )

                # Preserve source aspect ratio and size it prominently.
                max_w = max(1, int(vw * 0.9))
                max_h = max(1, int(vh * 0.8))
                scale = min(max_w / view_box.width(), max_h / view_box.height())
                logo_w = max(1, int(view_box.width() * scale))
                logo_h = max(1, int(view_box.height() * scale))
                x = (vw - logo_w) // 2
                y = (vh - logo_h) // 2

                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(x, y, logo_w, logo_h))
                painter.end()
                self.video_label.setPixmap(canvas)
                self.video_label.setText("")
                return
        except Exception:
            pass
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("HYDRA\n\nLoad a video to begin...")

    def _is_visualization_enabled(self) -> bool:
        # Preview should always render frames regardless of visualization-free toggle
        return (
            not self._setup_panel.chk_visualization_free.isChecked()
            or self.btn_preview.isChecked()
        )

    def _sync_contextual_controls(self):
        # ROI
        self.btn_finish_roi.setEnabled(self.roi_selection_active)
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        self.btn_clear_roi.setEnabled(
            len(self.roi_shapes) > 0 or self.roi_selection_active
        )

        # Crop video only if ROI exists and video loaded
        if hasattr(self, "btn_crop_video"):
            self.btn_crop_video.setEnabled(
                bool(self.roi_shapes) and bool(self.current_video_path)
            )

    def _apply_ui_state(self, state: str):
        if state == "no_video":
            extra_allowed = [
                self._dataset_panel.combo_xanylabeling_env,
                self._dataset_panel.btn_refresh_envs,
                self._dataset_panel.btn_open_xanylabeling,
                self._dataset_panel.btn_open_pose_label,
            ]
            self._set_interactive_widgets_enabled(
                False,
                allowlist=[
                    self._setup_panel.btn_file,
                    self._setup_panel.btn_load_config,
                ]
                + extra_allowed,
                remember_state=False,
            )
            self.btn_start.setEnabled(False)
            self.btn_preview.setEnabled(False)
            if hasattr(self, "_tracking_panel"):
                self._tracking_panel.btn_param_helper.setEnabled(False)
            self._set_video_interaction_enabled(False)
            self._setup_panel.g_video_player.setVisible(False)
            self._show_video_logo_placeholder()
            return

        if state == "idle":
            self._set_interactive_widgets_enabled(True)
            self.btn_start.setEnabled(True)
            self.btn_preview.setEnabled(True)
            if hasattr(self, "_tracking_panel"):
                self._tracking_panel.btn_param_helper.setEnabled(True)
            self._set_video_interaction_enabled(True)
            self._sync_contextual_controls()
            return

        if state == "tracking":
            allow = [self.btn_start]
            if self._is_visualization_enabled():
                allow.append(self.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self.btn_start.setEnabled(True)
            self._set_video_interaction_enabled(self._is_visualization_enabled())
            return

        if state == "preview":
            allow = [self.btn_preview] + list(self._preview_controls)
            if self._is_visualization_enabled():
                allow.append(self.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self.btn_preview.setEnabled(True)
            self._set_video_interaction_enabled(self._is_visualization_enabled())
            return

        # Locked (non-tracking) state: disable all interactive widgets
        if state == "locked":
            self._set_interactive_widgets_enabled(False)
            self._set_video_interaction_enabled(False)
            return

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
                from PySide6.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawPolygon(points)

        painter.end()
        return pix.toImage()

    def _apply_roi_mask_to_image(self, qimage):
        """Apply ROI visualization - draw boundary overlay for all detection methods.

        Both YOLO and Background Subtraction now show the same cyan boundary overlay
        for consistent UI experience. The actual masking happens in the tracking pipeline.
        """
        if self.roi_mask is None or not self.roi_shapes:
            return qimage

        # Use boundary overlay for all detection methods
        return self._draw_roi_overlay(qimage)

    @Slot(int, str)
    def on_progress_update(
        self: object, percentage: object, status_text: object
    ) -> object:
        """on_progress_update method documentation."""
        if self._stop_all_requested:
            return
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status_text)

    @Slot(str)
    def on_pose_exported_model_resolved(self, artifact_path: str) -> None:
        """Update pose exported-model UI/config when runtime resolves an artifact path."""
        if self._stop_all_requested:
            return
        path = str(artifact_path or "").strip()
        if not path:
            return
        logger.info("Pose runtime resolved exported model artifact: %s", path)
        try:
            # Persist run metadata immediately.
            self.save_config(prompt_if_exists=False)
        except Exception:
            logger.debug(
                "Failed to persist resolved pose runtime artifact metadata.",
                exc_info=True,
            )

    @Slot(str, str)
    def on_tracking_warning(self: object, title: object, message: object) -> object:
        """Display tracking warnings in the UI."""
        if self._stop_all_requested:
            return
        QMessageBox.information(self, title, message)

    def show_gpu_info(self: object) -> object:
        """Display GPU and acceleration information dialog."""
        from hydra_suite.utils.gpu_utils import get_device_info

        info = get_device_info()

        # Build formatted message
        lines = ["<b>GPU & Acceleration Status</b><br>"]

        # CUDA
        cuda_status = "✓ Available" if info["cuda_available"] else "✗ Not Available"
        lines.append(f"<br><b>NVIDIA CUDA:</b> {cuda_status}")
        if info["cuda_available"] and info.get("cuda_device_count", 0) > 0:
            lines.append(f"&nbsp;&nbsp;• Devices: {info['cuda_device_count']}")
            if "cupy_version" in info:
                lines.append(f"&nbsp;&nbsp;• CuPy: {info['cupy_version']}")

        # TensorRT
        tensorrt_status = (
            "✓ Available"
            if info.get("tensorrt_available", False)
            else "✗ Not Available"
        )
        lines.append(f"<br><b>NVIDIA TensorRT:</b> {tensorrt_status}")
        if info.get("tensorrt_available", False):
            lines.append("&nbsp;&nbsp;• 2-5× faster YOLO inference")

        # MPS (Apple Silicon)
        mps_status = "✓ Available" if info["mps_available"] else "✗ Not Available"
        lines.append(f"<br><b>Apple MPS:</b> {mps_status}")
        if info.get("torch_available", False) and "torch_version" in info:
            lines.append(f"&nbsp;&nbsp;• PyTorch: {info['torch_version']}")

        # CPU Acceleration
        numba_status = "✓ Available" if info["numba_available"] else "✗ Not Available"
        lines.append(f"<br><b>CPU JIT (Numba):</b> {numba_status}")
        if info["numba_available"] and "numba_version" in info:
            lines.append(f"&nbsp;&nbsp;• Version: {info['numba_version']}")

        # Overall status
        lines.append("<br><b>Overall Status:</b>")
        if info["cuda_available"]:
            lines.append("&nbsp;&nbsp;• Using NVIDIA GPU acceleration")
        elif info["mps_available"]:
            lines.append("&nbsp;&nbsp;• Using Apple Silicon GPU acceleration")
        elif info["numba_available"]:
            lines.append("&nbsp;&nbsp;• Using CPU JIT compilation")
        else:
            lines.append("&nbsp;&nbsp;• Using NumPy (no acceleration)")

        message = "<br>".join(lines)

        # Create message box with rich text
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("GPU & Acceleration Info")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec()

    @Slot(dict)
    def on_stats_update(self: object, stats: object) -> object:
        """Update real-time tracking statistics."""
        if self._stop_all_requested:
            return
        phase = str(stats.get("phase", "tracking"))
        is_precompute = phase == "individual_precompute"

        # Update FPS
        if "fps" in stats:
            if is_precompute:
                self.label_current_fps.setText(f"Precompute Rate: {stats['fps']:.1f}/s")
            else:
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
            if is_precompute:
                self.label_elapsed_time.setText(f"Precompute Elapsed: {elapsed_str}")
            else:
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
                if is_precompute:
                    self.label_eta.setText(f"Precompute ETA: {eta_str}")
                else:
                    self.label_eta.setText(f"ETA: {eta_str}")
            else:
                if is_precompute:
                    self.label_eta.setText("Precompute ETA: calculating...")
                else:
                    self.label_eta.setText("ETA: calculating...")
            self.label_eta.setVisible(True)

    @Slot(np.ndarray)
    def on_new_frame(self: object, rgb: object) -> object:
        """on_new_frame method documentation."""
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
            from PySide6.QtCore import QTimer

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

    def save_trajectories_to_csv(
        self: object, trajectories: object, output_path: object
    ) -> object:
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

    def merge_and_save_trajectories(self: object) -> object:
        """merge_and_save_trajectories method documentation."""
        if self._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting trajectory merging process...")
        logger.info("=" * 80)

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

        video_fp = self._setup_panel.file_line.text()
        if not video_fp:
            return
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        current_params = self.get_parameters_dict()
        resize_factor = self._setup_panel.spin_resize.value()
        interp_method = (
            self._postprocess_panel.combo_interpolation_method.currentText().lower()
        )
        max_gap = max(
            1,
            round(
                self._postprocess_panel.spin_interpolation_max_gap.value()
                * self._setup_panel.spin_fps.value()
            ),
        )
        heading_flip_max_burst = (
            self._postprocess_panel.spin_heading_flip_max_burst.value()
        )

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Merging trajectories...")

        # Create and start merge worker thread
        # Discover tag observation cache for AprilTag identity resolution
        _tag_cache_path = None
        if (
            bool(current_params.get("USE_APRILTAGS", False))
            or str(current_params.get("IDENTITY_METHOD", "")).lower() == "apriltags"
        ):
            _det_cache = getattr(self, "current_detection_cache_path", None)
            if _det_cache and os.path.exists(str(_det_cache)):
                import glob as _glob

                _pattern = str(_det_cache).replace(".npz", "") + "_tags_*.npz"
                _candidates = sorted(_glob.glob(_pattern))
                if _candidates:
                    _tag_cache_path = _candidates[-1]

        # Determine profiling settings for MergeWorker
        _enable_profiling = current_params.get("ENABLE_PROFILING", False)
        _merge_profile_path = None
        if _enable_profiling:
            _det_cache = getattr(self, "current_detection_cache_path", None)
            if _det_cache:
                _merge_profile_path = str(
                    Path(_det_cache).parent / "merge_profile.json"
                )
            elif video_fp:
                _merge_profile_path = str(Path(video_fp).parent / "merge_profile.json")

        self.merge_worker = MergeWorker(
            forward_trajs,
            backward_trajs,
            total_frames,
            current_params,
            resize_factor,
            interp_method,
            max_gap,
            tag_cache_path=_tag_cache_path,
            heading_flip_max_burst=heading_flip_max_burst,
            enable_profiling=_enable_profiling,
            profile_export_path=_merge_profile_path,
        )
        self.merge_worker.progress_signal.connect(self.on_merge_progress)
        self.merge_worker.finished_signal.connect(self.on_merge_finished)
        self.merge_worker.error_signal.connect(self.on_merge_error)
        self.merge_worker.start()

    def on_merge_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during merge."""
        if self._stop_all_requested:
            return
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _store_interpolated_pose_result(self, pose_csv_path, pose_rows):
        """Store interpolated pose results from CSV path or in-memory rows."""
        if pose_csv_path:
            self.current_interpolated_pose_csv_path = pose_csv_path
            self.current_interpolated_pose_df = None
            logger.info(f"Interpolated pose CSV saved: {pose_csv_path}")
        elif pose_rows:
            try:
                self.current_interpolated_pose_df = pd.DataFrame(pose_rows)
                self.current_interpolated_pose_csv_path = None
                logger.info(
                    "Interpolated pose rows kept in-memory: %d",
                    len(self.current_interpolated_pose_df),
                )
            except Exception:
                self.current_interpolated_pose_df = None

    def _store_interpolated_tag_result(self, tag_csv_path, tag_rows):
        """Store interpolated AprilTag results from CSV path or in-memory rows."""
        if tag_csv_path:
            self.current_interpolated_tag_csv_path = tag_csv_path
            self.current_interpolated_tag_df = None
            logger.info(f"Interpolated tag CSV saved: {tag_csv_path}")
        elif tag_rows:
            try:
                self.current_interpolated_tag_df = pd.DataFrame(tag_rows)
                self.current_interpolated_tag_csv_path = None
            except Exception:
                self.current_interpolated_tag_df = None

    def _store_interpolated_cnn_result(self, cnn_csv_paths, cnn_rows):
        """Store interpolated CNN identity results from CSV paths or in-memory rows."""
        if cnn_csv_paths:
            self.current_interpolated_cnn_csv_paths = cnn_csv_paths
            self.current_interpolated_cnn_dfs = {}
            logger.info(f"Interpolated CNN CSVs: {cnn_csv_paths}")
        elif cnn_rows:
            try:
                self.current_interpolated_cnn_dfs = {
                    label: pd.DataFrame(rows)
                    for label, rows in cnn_rows.items()
                    if rows
                }
                self.current_interpolated_cnn_csv_paths = {}
            except Exception:
                self.current_interpolated_cnn_dfs = {}

    def _store_interpolated_headtail_result(self, headtail_csv_path, headtail_rows):
        """Store interpolated head-tail results from CSV path or in-memory rows."""
        if headtail_csv_path:
            self.current_interpolated_headtail_csv_path = headtail_csv_path
            self.current_interpolated_headtail_df = None
            logger.info(f"Interpolated head-tail CSV saved: {headtail_csv_path}")
        elif headtail_rows:
            try:
                self.current_interpolated_headtail_df = pd.DataFrame(headtail_rows)
                self.current_interpolated_headtail_csv_path = None
            except Exception:
                self.current_interpolated_headtail_df = None

    def _on_interpolated_crops_finished(self, result):
        sender = self.sender()
        if (
            sender is not None
            and self.interp_worker is not None
            and sender is not self.interp_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("interp_worker")
            self._refresh_progress_visibility()
            return

        saved = 0
        gaps = 0
        try:
            saved = int(result.get("saved", 0))
            gaps = int(result.get("gaps", 0))
        except Exception:
            pass

        self._refresh_progress_visibility()
        logger.info(f"Interpolated individual crops saved: {saved} (gaps: {gaps})")

        mapping_path = result.get("mapping_path")
        if mapping_path:
            logger.info(f"Interpolated mapping saved: {mapping_path}")

        roi_csv_path = result.get("roi_csv_path")
        if roi_csv_path:
            logger.info(f"Interpolated ROIs CSV saved: {roi_csv_path}")

        roi_npz_path = result.get("roi_npz_path")
        if roi_npz_path:
            self.current_interpolated_roi_npz_path = roi_npz_path
            logger.info(f"Interpolated ROIs cache saved: {roi_npz_path}")

        self._store_interpolated_pose_result(
            result.get("pose_csv_path"), result.get("pose_rows")
        )
        self._store_interpolated_tag_result(
            result.get("tag_csv_path"), result.get("tag_rows")
        )
        self._store_interpolated_cnn_result(
            result.get("cnn_csv_paths"), result.get("cnn_rows")
        )
        self._store_interpolated_headtail_result(
            result.get("headtail_csv_path"), result.get("headtail_rows")
        )

        self._cleanup_thread_reference("interp_worker")
        self._refresh_progress_visibility()

        if self._pending_pose_export_csv_path:
            self._relink_final_pose_augmented_csv(self._pending_pose_export_csv_path)

        if self._pending_finish_after_interp:
            self._pending_finish_after_interp = False
            if self._start_pending_oriented_track_video_export(
                self._session_final_csv_path
            ):
                return
            self._run_pending_video_generation_or_finalize()

    def _resolve_source_video_fps(self) -> float:
        """Return the source video FPS, falling back to the UI value."""
        fps = 0.0
        video_path = (
            self._setup_panel.file_line.text().strip()
            if hasattr(self, "_setup_panel")
            else ""
        )
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            try:
                if cap.isOpened():
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            finally:
                cap.release()
        if fps <= 0.0 and hasattr(self, "_setup_panel"):
            fps = float(self._setup_panel.spin_fps.value() or 0.0)
        return max(1.0, fps or 1.0)

    def _resolve_current_individual_dataset_dir(self):
        """Resolve the active per-session individual dataset directory."""
        params = self.get_parameters_dict()
        dataset_dir = resolve_individual_dataset_dir(
            params.get("INDIVIDUAL_DATASET_OUTPUT_DIR"),
            params.get("INDIVIDUAL_DATASET_NAME"),
            self._individual_dataset_run_id,
        )
        if dataset_dir is None:
            return None
        return Path(dataset_dir).expanduser()

    def _generate_oriented_track_videos(self, final_csv_path):
        """Export orientation-fixed videos for final trajectories."""
        try:
            if self._stop_all_requested:
                return False
            if not self._should_generate_oriented_track_videos():
                return False
            if not final_csv_path or not os.path.exists(final_csv_path):
                return False

            dataset_dir = self._resolve_current_individual_dataset_dir()
            if dataset_dir is None:
                logger.warning(
                    "Skipping oriented track video export: no individual dataset directory found."
                )
                return False
            if not self.current_detection_cache_path or not os.path.exists(
                self.current_detection_cache_path
            ):
                logger.warning(
                    "Skipping oriented track video export: no compatible detection cache is available."
                )
                return False

            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Generating oriented track videos...")

            if (
                self.oriented_video_worker is not None
                and self.oriented_video_worker.isRunning()
            ):
                logger.warning(
                    "Oriented track video export already running; skipping duplicate request."
                )
                return True
            if (
                self.oriented_video_worker is not None
                and not self.oriented_video_worker.isRunning()
            ):
                self.oriented_video_worker.deleteLater()
                self.oriented_video_worker = None

            padding_fraction = (
                float(self._identity_panel.spin_individual_padding.value())
                if hasattr(self, "_identity_panel")
                else 0.1
            )
            self.oriented_video_worker = OrientedTrackVideoWorker(
                final_csv_path,
                str(dataset_dir),
                self._setup_panel.file_line.text().strip(),
                self.current_detection_cache_path,
                self.current_interpolated_roi_npz_path,
                self._resolve_source_video_fps(),
                max(0.0, padding_fraction),
                tuple(int(c) for c in self._identity_panel._background_color),
                bool(self._dataset_panel.chk_suppress_foreign_obb_dataset.isChecked()),
            )
            self.oriented_video_worker.progress_signal.connect(self.on_progress_update)
            self.oriented_video_worker.finished_signal.connect(
                self._on_oriented_track_videos_finished
            )
            self.oriented_video_worker.error_signal.connect(
                self._on_oriented_track_videos_error
            )
            self.oriented_video_worker.finished.connect(
                self._on_oriented_track_video_worker_thread_finished
            )
            self.oriented_video_worker.start()
            return True
        except Exception as e:
            logger.warning(f"Oriented track video export failed to start: {e}")
            return False

    def _start_pending_oriented_track_video_export(self, final_csv_path) -> bool:
        """Start optional oriented track video export and hold the finish pipeline."""
        started = self._generate_oriented_track_videos(final_csv_path)
        if started:
            self._pending_finish_after_track_videos = True
        return started

    def _on_oriented_track_video_worker_thread_finished(self):
        """Release completed oriented track video worker safely."""
        sender = self.sender()
        if (
            sender is not None
            and self.oriented_video_worker is not None
            and sender is not self.oriented_video_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("oriented_video_worker")
        self._refresh_progress_visibility()

    def _on_oriented_track_videos_finished(self, result):
        """Handle completion of oriented track video export."""
        sender = self.sender()
        if (
            sender is not None
            and self.oriented_video_worker is not None
            and sender is not self.oriented_video_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("oriented_video_worker")
            self._refresh_progress_visibility()
            return

        try:
            exported_videos = int(result.get("exported_videos", 0))
            exported_frames = int(result.get("exported_frames", 0))
            exported_tracks = int(result.get("exported_tracks", 0))
            missing_rows = int(result.get("missing_rows", 0))
            output_dir = str(result.get("output_dir", "")).strip()
        except Exception:
            exported_videos = 0
            exported_frames = 0
            exported_tracks = 0
            missing_rows = 0
            output_dir = ""

        if output_dir:
            logger.info(
                "Oriented track videos exported to %s (%d/%d tracks, %d frames, missing rows=%d)",
                output_dir,
                exported_videos,
                exported_tracks,
                exported_frames,
                missing_rows,
            )
        else:
            logger.info(
                "Oriented track video export complete (%d/%d tracks, %d frames, missing rows=%d)",
                exported_videos,
                exported_tracks,
                exported_frames,
                missing_rows,
            )

        self._cleanup_thread_reference("oriented_video_worker")
        self._refresh_progress_visibility()

        if self._pending_finish_after_track_videos:
            self._pending_finish_after_track_videos = False
            self._run_pending_video_generation_or_finalize()

    def _on_oriented_track_videos_error(self, error_message):
        """Handle oriented track video export errors without aborting the session."""
        sender = self.sender()
        if (
            sender is not None
            and self.oriented_video_worker is not None
            and sender is not self.oriented_video_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        logger.warning("Oriented track video export failed: %s", error_message)
        self._cleanup_thread_reference("oriented_video_worker")
        self._refresh_progress_visibility()
        if self._pending_finish_after_track_videos:
            self._pending_finish_after_track_videos = False
            self._run_pending_video_generation_or_finalize()

    def on_merge_error(self: object, error_message: object) -> object:
        """Handle merge errors."""
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._stop_all_requested:
            self._refresh_progress_visibility()
            return
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.critical(
            self, "Merge Error", f"Error during trajectory merging:\n{error_message}"
        )
        logger.error(f"Trajectory merge error: {error_message}")

    def on_merge_finished(self: object, resolved_trajectories: object) -> object:
        """Handle completion of trajectory merging."""
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._stop_all_requested:
            self._refresh_progress_visibility()
            return
        self.progress_label.setText("Saving merged trajectories...")

        raw_csv_path = self._setup_panel.csv_line.text()
        merged_csv_path = None
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            merged_csv_path = f"{base}_final.csv"
            if self.save_trajectories_to_csv(resolved_trajectories, merged_csv_path):
                # Track initial tracking CSV as temporary (only if cleanup enabled)
                if (
                    self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                    and raw_csv_path not in self.temporary_files
                ):
                    self.temporary_files.append(raw_csv_path)
                logger.info(f"✓ Merged trajectory data saved to: {merged_csv_path}")

        # Complete session pipeline. Video generation is deferred to the very end
        # after pose export and interpolated individual analysis complete.
        self._finish_tracking_session(final_csv_path=merged_csv_path)

    def _generate_video_from_trajectories(
        self, trajectories_df, csv_path=None, finalize_on_complete=True
    ):
        """
        Generate annotated video from post-processed trajectories.

        Args:
            trajectories_df: DataFrame with merged/interpolated trajectories
            csv_path: Path to the CSV file (optional, for logging)
            finalize_on_complete: If True, continue full finish pipeline after render.
        """
        logger.info("=" * 80)
        logger.info("Generating video from post-processed trajectories...")
        logger.info("=" * 80)

        # Ensure progress UI is visible for video generation
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Generating video...")
        QApplication.processEvents()

        video_path = self._setup_panel.file_line.text()
        output_path = self._postprocess_panel.video_out_line.text()

        def _complete_after_video():
            if finalize_on_complete:
                self._finish_tracking_session(final_csv_path=csv_path)
            else:
                self._finalize_tracking_session_ui()

        if not video_path or not output_path:
            logger.error("Video input or output path not specified")
            _complete_after_video()
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            _complete_after_video()
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            cap.release()
            _complete_after_video()
            return

        logger.info(f"Writing video: {frame_width}x{frame_height} @ {fps} FPS")

        # Get visualization parameters
        params = self.get_parameters_dict()
        start_frame = int(params.get("START_FRAME", 0) or 0)
        end_frame = params.get("END_FRAME", None)
        if end_frame is None:
            end_frame = total_video_frames - 1 if total_video_frames > 0 else 0
        end_frame = int(end_frame)

        # Clamp to video bounds and export only the tracked subset.
        if total_video_frames > 0:
            start_frame = max(0, min(start_frame, total_video_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_video_frames - 1))
        total_frames = max(0, end_frame - start_frame + 1)
        logger.info(
            f"Exporting tracked frame range: {start_frame}-{end_frame} ({total_frames} frames)"
        )

        if total_frames <= 0:
            logger.error("Invalid frame range for video generation.")
            cap.release()
            out.release()
            _complete_after_video()
            return

        # Seek once to the tracked start frame so we don't write unrelated frames.
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        colors = params.get("TRAJECTORY_COLORS", [])
        reference_body_size = params.get("REFERENCE_BODY_SIZE", 30.0)

        # Get video visualization settings
        show_labels = self._postprocess_panel.check_show_labels.isChecked()
        show_orientation = self._postprocess_panel.check_show_orientation.isChecked()
        show_trails = self._postprocess_panel.check_show_trails.isChecked()
        trail_duration_sec = self._postprocess_panel.spin_trail_duration.value()
        trail_duration_frames = int(
            trail_duration_sec * fps
        )  # Convert seconds to frames
        marker_size = self._postprocess_panel.spin_marker_size.value()
        text_scale = self._postprocess_panel.spin_text_scale.value()
        arrow_length = self._postprocess_panel.spin_arrow_length.value()
        advanced_config = params.get("ADVANCED_CONFIG", {})

        # NOTE: Trajectories in merged CSV are already scaled to original coordinates
        # (see MergeWorker line ~173: coordinates divided by resize_factor)
        # So we use them directly without additional scaling

        # Scale drawing parameters by body size
        # reference_body_size is in original (unresized) coordinates
        # Video is at original resolution, so use body size directly
        marker_radius = int(marker_size * reference_body_size)
        arrow_len = int(arrow_length * reference_body_size)
        text_size = 0.5 * text_scale
        marker_thickness = max(2, int(0.15 * reference_body_size))
        pose_point_radius = int(
            max(
                1,
                advanced_config.get(
                    "video_pose_point_radius", max(2, marker_radius // 3)
                ),
            )
        )
        pose_point_thickness = int(
            advanced_config.get("video_pose_point_thickness", -1)
        )
        pose_line_thickness = int(
            max(1, advanced_config.get("video_pose_line_thickness", 2))
        )
        pose_color_mode = (
            str(advanced_config.get("video_pose_color_mode", "track")).strip().lower()
        )
        pose_fixed_color_raw = advanced_config.get("video_pose_color", [255, 255, 255])
        if (
            isinstance(pose_fixed_color_raw, (list, tuple))
            and len(pose_fixed_color_raw) == 3
        ):
            try:
                pose_fixed_color = tuple(
                    int(max(0, min(255, float(v)))) for v in pose_fixed_color_raw
                )
            except Exception:
                pose_fixed_color = (255, 255, 255)
        else:
            pose_fixed_color = (255, 255, 255)
        pose_min_conf = normalize_pose_render_min_conf(
            params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
        )
        pose_edges = []
        pose_column_triplets = []
        show_pose = bool(advanced_config.get("video_show_pose", True))
        pose_col_pattern = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")
        pose_labels_available = {}
        for col in trajectories_df.columns:
            m = pose_col_pattern.match(str(col))
            if m is None:
                continue
            label = m.group(1)
            axis = m.group(2)
            pose_labels_available.setdefault(label, set()).add(axis)
        if not pose_labels_available:
            show_pose = False
        if show_pose:
            skeleton_names = []
            skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
            if skeleton_file and os.path.exists(skeleton_file):
                try:
                    with open(skeleton_file, "r", encoding="utf-8") as f:
                        skeleton_data = json.load(f)
                    names_raw = skeleton_data.get(
                        "keypoint_names", skeleton_data.get("keypoints", [])
                    )
                    skeleton_names = [str(n) for n in names_raw]
                    raw_edges = skeleton_data.get(
                        "skeleton_edges", skeleton_data.get("edges", [])
                    )
                    for edge in raw_edges:
                        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                            try:
                                pose_edges.append((int(edge[0]), int(edge[1])))
                            except Exception:
                                continue
                except Exception:
                    pose_edges = []
            ordered_labels = build_pose_keypoint_labels(
                skeleton_names, len(skeleton_names)
            )
            # Add any extra labels present in CSV but absent from skeleton.
            extras = sorted(
                [
                    lbl
                    for lbl in pose_labels_available.keys()
                    if lbl not in ordered_labels
                ]
            )
            ordered_labels.extend(extras)
            for label in ordered_labels:
                axes = pose_labels_available.get(label, set())
                if {"X", "Y", "Conf"}.issubset(axes):
                    pose_column_triplets.append(
                        (
                            f"PoseKpt_{label}_X",
                            f"PoseKpt_{label}_Y",
                            f"PoseKpt_{label}_Conf",
                        )
                    )
            if not pose_column_triplets:
                show_pose = False

        # ── Pre-extract trajectory arrays for O(1)/O(log N) lookups ─────────────
        _frame_ids = trajectories_df["FrameID"].to_numpy(dtype=np.int32)
        _track_ids = trajectories_df["TrajectoryID"].to_numpy(dtype=np.int32)
        _xs = trajectories_df["X"].to_numpy(dtype=np.float64)
        _ys = trajectories_df["Y"].to_numpy(dtype=np.float64)
        _thetas = (
            trajectories_df["Theta"].to_numpy(dtype=np.float64)
            if "Theta" in trajectories_df.columns
            else np.full(len(trajectories_df), np.nan)
        )

        # Pose arrays: shape (K, N, 3) extracted once to avoid per-row pandas overhead
        _pose_kpts = None
        if show_pose and pose_column_triplets:
            _K = len(pose_column_triplets)
            _N = len(trajectories_df)
            _pose_kpts = np.full((_K, _N, 3), np.nan, dtype=np.float32)
            for _k, (_x_col, _y_col, _c_col) in enumerate(pose_column_triplets):
                if _x_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 0] = trajectories_df[_x_col].to_numpy(
                        dtype=np.float32
                    )
                if _y_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 1] = trajectories_df[_y_col].to_numpy(
                        dtype=np.float32
                    )
                if _c_col in trajectories_df.columns:
                    _pose_kpts[_k, :, 2] = trajectories_df[_c_col].to_numpy(
                        dtype=np.float32
                    )

        # Frame → row-index list (replaces slow iterrows + pandas Series access)
        traj_indices_by_frame: dict = {}
        for _i in range(len(_frame_ids)):
            _fid = int(_frame_ids[_i])
            if _fid not in traj_indices_by_frame:
                traj_indices_by_frame[_fid] = []
            traj_indices_by_frame[_fid].append(_i)

        # Per-track sorted arrays for O(log N) trail window lookup via binary search
        _track_sorted_row_indices: dict = {}
        _track_sorted_frame_vals: dict = {}
        if show_trails:
            _tmp_track: dict = {}
            for _i in range(len(_track_ids)):
                _tid = int(_track_ids[_i])
                if _tid not in _tmp_track:
                    _tmp_track[_tid] = []
                _tmp_track[_tid].append(_i)
            for _tid, _idxs in _tmp_track.items():
                _idx_arr = np.asarray(_idxs, dtype=np.int32)
                _order = np.argsort(_frame_ids[_idx_arr])
                _track_sorted_row_indices[_tid] = _idx_arr[_order]
                _track_sorted_frame_vals[_tid] = _frame_ids[_idx_arr[_order]]

        # Pre-compute palette and one color per track ID (avoid rebuilding every frame)
        _category20_colors = [
            (127, 127, 31),
            (188, 189, 34),
            (140, 86, 75),
            (255, 127, 14),
            (214, 39, 40),
            (255, 152, 150),
            (197, 176, 213),
            (148, 103, 189),
            (196, 156, 148),
            (227, 119, 194),
            (199, 199, 199),
            (140, 140, 140),
            (23, 190, 207),
            (158, 218, 229),
            (57, 59, 121),
            (82, 84, 163),
            (107, 110, 207),
            (156, 158, 222),
            (99, 121, 57),
            (140, 162, 82),
        ]
        _n_cat = len(_category20_colors)
        _max_track = int(_track_ids.max()) if len(_track_ids) > 0 else 0
        _precomputed_colors = [
            (
                colors[_tid]
                if colors and _tid < len(colors)
                else _category20_colors[_tid % _n_cat]
            )
            for _tid in range(_max_track + 1)
        ]

        # Threaded writer: overlaps disk I/O with CPU rendering.
        # maxsize=4 caps in-flight frames so fast rendering on Linux cannot
        # exhaust memory before the writer thread drains the queue.
        import queue as _queue
        import threading as _threading

        _write_q: _queue.Queue = _queue.Queue(maxsize=4)

        def _writer_thread():
            while True:
                _item = _write_q.get()
                if _item is None:
                    break
                out.write(_item)

        _writer = _threading.Thread(target=_writer_thread, daemon=True)
        _writer.start()

        # Process only the tracked frame range.
        for rel_idx in range(total_frames):
            frame_idx = start_frame + rel_idx
            ret, frame = cap.read()
            if not ret:
                break

            # Get row indices for this frame
            frame_row_indices = traj_indices_by_frame.get(frame_idx, [])

            # Draw trails first (underneath current positions)
            if show_trails:
                for row_i in frame_row_indices:
                    track_id = int(_track_ids[row_i])
                    color = (
                        _precomputed_colors[track_id]
                        if track_id < len(_precomputed_colors)
                        else _category20_colors[track_id % _n_cat]
                    )
                    # Binary-search trail window: O(log N) instead of O(N) per frame
                    if track_id in _track_sorted_frame_vals:
                        _sfv = _track_sorted_frame_vals[track_id]
                        _sri = _track_sorted_row_indices[track_id]
                        _lo = int(
                            np.searchsorted(
                                _sfv, frame_idx - trail_duration_frames, side="left"
                            )
                        )
                        _hi = int(np.searchsorted(_sfv, frame_idx, side="left"))
                        if _hi - _lo >= 2:
                            _trail_xs = _xs[_sri[_lo:_hi]]
                            _trail_ys = _ys[_sri[_lo:_hi]]
                            _trail_fs = _sfv[_lo:_hi]
                            _trail_lw = max(1, marker_thickness // 2)
                            for _seg in range(_hi - _lo - 1):
                                _px1, _py1 = _trail_xs[_seg], _trail_ys[_seg]
                                _px2, _py2 = _trail_xs[_seg + 1], _trail_ys[_seg + 1]
                                if (
                                    np.isnan(_px1)
                                    or np.isnan(_py1)
                                    or np.isnan(_px2)
                                    or np.isnan(_py2)
                                ):
                                    continue
                                _age = frame_idx - int(_trail_fs[_seg])
                                _alpha = 1.0 - (_age / trail_duration_frames)
                                cv2.line(
                                    frame,
                                    (int(_px1), int(_py1)),
                                    (int(_px2), int(_py2)),
                                    (
                                        int(color[0] * _alpha),
                                        int(color[1] * _alpha),
                                        int(color[2] * _alpha),
                                    ),
                                    _trail_lw,
                                )

            # Draw current positions
            for row_i in frame_row_indices:
                track_id = int(_track_ids[row_i])
                cx_f, cy_f = _xs[row_i], _ys[row_i]

                # Skip if NaN
                if np.isnan(cx_f) or np.isnan(cy_f):
                    continue

                cx, cy = int(cx_f), int(cy_f)
                color = (
                    _precomputed_colors[track_id]
                    if track_id < len(_precomputed_colors)
                    else _category20_colors[track_id % _n_cat]
                )

                # Draw circle at position
                cv2.circle(frame, (cx, cy), marker_radius, color, marker_thickness)

                # Draw label
                if show_labels:
                    label_offset = int(marker_radius + 5)
                    cv2.putText(
                        frame,
                        f"ID{track_id}",
                        (cx + label_offset, cy - label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_size,
                        color,
                        max(1, int(text_scale * 2)),
                    )

                # Draw orientation if available
                if show_orientation:
                    _theta = _thetas[row_i]
                    if not np.isnan(_theta):
                        cv2.arrowedLine(
                            frame,
                            (cx, cy),
                            (
                                int(cx + arrow_len * np.cos(_theta)),
                                int(cy + arrow_len * np.sin(_theta)),
                            ),
                            color,
                            marker_thickness,
                            tipLength=0.3,
                        )

                # Draw pose keypoints/skeleton (uses pre-extracted numpy slice)
                if show_pose and _pose_kpts is not None:
                    kpts_arr = _pose_kpts[:, row_i, :]  # shape (K, 3) — zero-copy view
                    if np.any(np.isfinite(kpts_arr[:, 2])):
                        pose_color = (
                            color if pose_color_mode == "track" else pose_fixed_color
                        )
                        if pose_edges:
                            for e0, e1 in pose_edges:
                                if (
                                    e0 < 0
                                    or e1 < 0
                                    or e0 >= len(kpts_arr)
                                    or e1 >= len(kpts_arr)
                                ):
                                    continue
                                if not is_renderable_pose_keypoint(
                                    kpts_arr[e0, 0],
                                    kpts_arr[e0, 1],
                                    kpts_arr[e0, 2],
                                    pose_min_conf,
                                ) or not is_renderable_pose_keypoint(
                                    kpts_arr[e1, 0],
                                    kpts_arr[e1, 1],
                                    kpts_arr[e1, 2],
                                    pose_min_conf,
                                ):
                                    continue
                                cv2.line(
                                    frame,
                                    (
                                        int(round(float(kpts_arr[e0, 0]))),
                                        int(round(float(kpts_arr[e0, 1]))),
                                    ),
                                    (
                                        int(round(float(kpts_arr[e1, 0]))),
                                        int(round(float(kpts_arr[e1, 1]))),
                                    ),
                                    pose_color,
                                    pose_line_thickness,
                                )
                        for kpt in kpts_arr:
                            if not is_renderable_pose_keypoint(
                                kpt[0], kpt[1], kpt[2], pose_min_conf
                            ):
                                continue
                            cv2.circle(
                                frame,
                                (int(round(float(kpt[0]))), int(round(float(kpt[1])))),
                                pose_point_radius,
                                pose_color,
                                pose_point_thickness,
                            )

            # Enqueue frame for background write (overlaps disk I/O with CPU rendering)
            _write_q.put(frame)

            # Update progress every 30 frames
            if rel_idx % 30 == 0:
                progress = int(((rel_idx + 1) / total_frames) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()

        # Signal writer thread to finish and wait for all frames to be flushed
        _write_q.put(None)
        _writer.join()

        # Cleanup
        cap.release()
        out.release()

        logger.info(f"✓ Video saved to: {output_path}")
        logger.info("=" * 80)

        _complete_after_video()

    @Slot(bool, list, list)
    def on_tracking_finished(
        self: object, finished_normally: object, fps_list: object, full_traj: object
    ) -> object:
        """on_tracking_finished method documentation."""
        sender = self.sender()
        if (
            sender is not None
            and self.tracking_worker is not None
            and sender is not self.tracking_worker
        ):
            logger.debug(
                "Ignoring stale tracking finished signal from previous worker."
            )
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        self._stop_csv_writer()

        if self._stop_all_requested:
            logger.info("Tracking stop requested; skipping post-processing pipeline.")
            self._cleanup_thread_reference("tracking_worker")
            self._refresh_progress_visibility()
            gc.collect()
            return

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
            self.btn_start.blockSignals(True)
            self.btn_start.setChecked(False)
            self.btn_start.blockSignals(False)
            self.btn_start.setText("Start Full Tracking")
            self._apply_ui_state("idle" if self.current_video_path else "no_video")
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

        worker_props_path = ""
        if self.tracking_worker is not None:
            worker_props_path = str(
                getattr(self.tracking_worker, "individual_properties_cache_path", "")
                or ""
            ).strip()
        if worker_props_path:
            self.current_individual_properties_cache_path = worker_props_path
            logger.info(
                "Using individual properties cache for export: %s",
                worker_props_path,
            )

        if finished_normally:
            logger.info("Tracking completed successfully.")
            is_backward_mode = (
                hasattr(self.tracking_worker, "backward_mode")
                and self.tracking_worker.backward_mode
            )
            # Accumulate fps_list across forward and backward passes
            if isinstance(fps_list, (list, tuple)) and fps_list:
                self._session_fps_list = list(self._session_fps_list) + [
                    f for f in fps_list if f and f > 0
                ]
            if not is_backward_mode:
                self._session_frames_processed = (
                    len(fps_list) if isinstance(fps_list, (list, tuple)) else 0
                )
            is_backward_enabled = self._tracking_panel.chk_enable_backward.isChecked()

            processed_trajectories = full_traj
            if self._postprocess_panel.enable_postprocessing.isChecked():
                params = self.get_parameters_dict()
                raw_csv_path = self._setup_panel.csv_line.text()

                if is_backward_mode and raw_csv_path:
                    # Use backward CSV for processing
                    base, ext = os.path.splitext(raw_csv_path)
                    csv_to_process = f"{base}_backward{ext}"
                elif is_backward_enabled and raw_csv_path:
                    # Forward mode with backward enabled: use _forward.csv
                    # (tracking writes to _forward.csv when backward is enabled)
                    base, ext = os.path.splitext(raw_csv_path)
                    csv_to_process = f"{base}_forward{ext}"
                else:
                    # Forward-only mode: use base path
                    csv_to_process = raw_csv_path

                from hydra_suite.core.post.processing import (
                    interpolate_trajectories,
                    process_trajectories_from_csv,
                )

                if csv_to_process and os.path.exists(csv_to_process):
                    # Use CSV-based processing to preserve confidence columns
                    processed_trajectories, stats = process_trajectories_from_csv(
                        csv_to_process, params
                    )
                    logger.info(f"Post-processing stats: {stats}")

                    # NOTE: Do NOT apply interpolation here if backward tracking will follow
                    # Interpolation should only be applied AFTER merging (in MergeWorker)
                    # or in forward-only mode (below). Pre-merge interpolation can affect
                    # merge candidate detection by filling gaps that should remain as gaps.

                    # NOTE: Do NOT scale to original space yet if backward tracking will happen
                    # Scaling will be done after merging or in forward-only mode
                else:
                    # Fallback to old method if CSV not available
                    processed_trajectories, stats = process_trajectories(
                        full_traj, params
                    )
                    logger.info(f"Post-processing stats (fallback): {stats}")

            if not is_backward_mode:
                raw_csv_path = self._setup_panel.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate forward CSV as temporary (only if cleanup enabled)
                    forward_csv = f"{base}_forward{ext}"
                    if (
                        self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                        and forward_csv not in self.temporary_files
                    ):
                        self.temporary_files.append(forward_csv)

                    processed_csv_path = f"{base}_forward_processed{ext}"
                    # Only track processed CSV as temporary if backward tracking will run
                    # and cleanup is enabled (it will be merged into final file).
                    # Otherwise, this IS the final file.
                    if (
                        is_backward_enabled
                        and self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                        and processed_csv_path not in self.temporary_files
                    ):
                        self.temporary_files.append(processed_csv_path)

                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )

                if is_backward_enabled:
                    self.forward_processed_trajs = processed_trajectories
                    # Log coordinate ranges for debugging
                    if (
                        isinstance(processed_trajectories, pd.DataFrame)
                        and not processed_trajectories.empty
                    ):
                        logger.info(
                            f"Forward trajectories stored for merge: "
                            f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                            f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
                        )
                    self.start_backward_tracking()
                else:
                    # Forward-only mode: Apply interpolation here (no merge step)
                    interp_method = (
                        self._postprocess_panel.combo_interpolation_method.currentText().lower()
                    )
                    if interp_method != "none":
                        max_gap = max(
                            1,
                            round(
                                self._postprocess_panel.spin_interpolation_max_gap.value()
                                * self._setup_panel.spin_fps.value()
                            ),
                        )
                        heading_flip_max_burst = (
                            self._postprocess_panel.spin_heading_flip_max_burst.value()
                        )
                        processed_trajectories = interpolate_trajectories(
                            processed_trajectories,
                            method=interp_method,
                            max_gap=max_gap,
                            heading_flip_max_burst=heading_flip_max_burst,
                        )

                    # Scale coordinates to original video space (forward-only mode)
                    resize_factor = self._setup_panel.spin_resize.value()
                    processed_trajectories = self._scale_trajectories_to_original_space(
                        processed_trajectories, resize_factor
                    )

                    # Re-save the scaled trajectories
                    final_csv_path = None
                    if raw_csv_path:
                        base, ext = os.path.splitext(raw_csv_path)
                        # In forward-only mode, the final output is _forward_processed.csv
                        final_csv_path = f"{base}_forward_processed{ext}"
                        self.save_trajectories_to_csv(
                            processed_trajectories, final_csv_path
                        )
                        # Track initial tracking CSV as temporary (only if cleanup enabled)
                        if (
                            self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                            and raw_csv_path not in self.temporary_files
                        ):
                            self.temporary_files.append(raw_csv_path)

                    # Complete session pipeline. Video generation is deferred to
                    # the final step after pose export + interpolation.
                    self._finish_tracking_session(final_csv_path=final_csv_path)
                    return
            else:
                raw_csv_path = self._setup_panel.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate backward CSV as temporary (only if cleanup enabled)
                    backward_csv = f"{base}_backward{ext}"
                    if (
                        self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                        and backward_csv not in self.temporary_files
                    ):
                        self.temporary_files.append(backward_csv)

                    processed_csv_path = f"{base}_backward_processed{ext}"
                    # Track processed CSV as temporary (only if cleanup enabled)
                    if (
                        self._postprocess_panel.chk_cleanup_temp_files.isChecked()
                        and processed_csv_path not in self.temporary_files
                    ):
                        self.temporary_files.append(processed_csv_path)
                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )
                self.backward_processed_trajs = processed_trajectories
                # Log coordinate ranges for debugging
                if (
                    isinstance(processed_trajectories, pd.DataFrame)
                    and not processed_trajectories.empty
                ):
                    logger.info(
                        f"Backward trajectories stored for merge: "
                        f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                        f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
                    )

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
                    # Start merge in background thread (will handle cleanup when done)
                    self.merge_and_save_trajectories()
                else:
                    # No merge needed, do cleanup now
                    # Pass the correct CSV path based on what we processed
                    self._finish_tracking_session(final_csv_path=processed_csv_path)
        else:
            logger.error("Tracking did not finish normally.")
            QMessageBox.warning(
                self,
                "Tracking Failed",
                "An error occurred during tracking. Check logs for details.",
            )
            if self._setup_panel.g_batch.isChecked():
                self.current_batch_index = -1
                logger.info("Batch mode aborted due to error.")
            self._finish_tracking_session(final_csv_path=None)

    def _is_pose_export_enabled(self) -> bool:
        """Return True when pose extraction export should be produced."""
        return bool(
            self._is_individual_pipeline_enabled()
            and hasattr(self, "_identity_panel")
            and self._identity_panel.chk_enable_pose_extractor.isChecked()
            and self._is_yolo_detection_mode()
        )

    def _build_pose_augmented_dataframe(self, final_csv_path):
        """Load final CSV and merge available cached/interpolated pose columns."""
        if not final_csv_path or not os.path.exists(final_csv_path):
            return None

        # Check for any available analysis source (pose, tag, cnn, headtail)
        _has_interp_tag = bool(
            (getattr(self, "current_interpolated_tag_csv_path", None))
            or (
                isinstance(
                    getattr(self, "current_interpolated_tag_df", None),
                    pd.DataFrame,
                )
            )
        )
        _has_interp_cnn = bool(
            getattr(self, "current_interpolated_cnn_csv_paths", None)
            or getattr(self, "current_interpolated_cnn_dfs", None)
        )
        _has_interp_ht = bool(
            (getattr(self, "current_interpolated_headtail_csv_path", None))
            or (
                isinstance(
                    getattr(self, "current_interpolated_headtail_df", None),
                    pd.DataFrame,
                )
            )
        )
        _has_other_analyses = _has_interp_tag or _has_interp_cnn or _has_interp_ht

        if not self._is_pose_export_enabled() and not _has_other_analyses:
            return None

        cache_path = str(self.current_individual_properties_cache_path or "").strip()
        cache_available = bool(cache_path and os.path.exists(cache_path))
        interp_pose_path = str(self.current_interpolated_pose_csv_path or "").strip()
        interp_available = bool(interp_pose_path and os.path.exists(interp_pose_path))
        interp_pose_df_mem = getattr(self, "current_interpolated_pose_df", None)
        interp_mem_available = (
            isinstance(interp_pose_df_mem, pd.DataFrame)
            and not interp_pose_df_mem.empty
        )
        if (
            not cache_available
            and not interp_available
            and not interp_mem_available
            and not _has_other_analyses
        ):
            logger.warning(
                "Pose export skipped: no pose sources found (cache=%s, interpolated=%s, in_memory=%s).",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
                bool(interp_mem_available),
            )
            return None

        try:
            trajectories_df = pd.read_csv(final_csv_path)
        except Exception:
            logger.exception(
                "Pose export skipped: failed to load trajectories CSV: %s",
                final_csv_path,
            )
            return None

        try:
            with_pose_df = trajectories_df
            if cache_available:
                min_valid_conf = float(
                    self._identity_panel.spin_pose_min_kpt_conf_valid.value()
                )
                _resize_factor = float(
                    self.get_parameters_dict().get("RESIZE_FACTOR", 1.0)
                )
                _coord_scale = (
                    1.0 / _resize_factor
                    if _resize_factor and _resize_factor != 1.0
                    else 1.0
                )
                with_pose_df = augment_trajectories_with_pose_cache(
                    with_pose_df,
                    cache_path,
                    ignore_keypoints=self._parse_pose_ignore_keypoints(),
                    min_valid_conf=min_valid_conf,
                    coordinate_scale=_coord_scale,
                )
            if interp_available:
                interp_pose_df = pd.read_csv(interp_pose_path)
                with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df)
            elif interp_mem_available:
                with_pose_df = merge_interpolated_pose_df(
                    with_pose_df, interp_pose_df_mem
                )

            # --- Merge interpolated AprilTag observations ---
            _interp_tag_path = str(
                getattr(self, "current_interpolated_tag_csv_path", None) or ""
            ).strip()
            _interp_tag_df = getattr(self, "current_interpolated_tag_df", None)
            try:
                from hydra_suite.core.identity.properties.export import (
                    merge_interpolated_apriltag_df,
                )

                if _interp_tag_path and os.path.exists(_interp_tag_path):
                    _tag_df = pd.read_csv(_interp_tag_path)
                    with_pose_df = merge_interpolated_apriltag_df(with_pose_df, _tag_df)
                elif (
                    isinstance(_interp_tag_df, pd.DataFrame)
                    and not _interp_tag_df.empty
                ):
                    with_pose_df = merge_interpolated_apriltag_df(
                        with_pose_df, _interp_tag_df
                    )
            except Exception:
                logger.debug("Interpolated AprilTag merge skipped.", exc_info=True)

            # --- Merge interpolated CNN identity predictions ---
            _interp_cnn_paths = (
                getattr(self, "current_interpolated_cnn_csv_paths", {}) or {}
            )
            _interp_cnn_dfs = getattr(self, "current_interpolated_cnn_dfs", {}) or {}
            try:
                from hydra_suite.core.identity.properties.export import (
                    merge_interpolated_cnn_df,
                )

                _all_cnn_labels = set(_interp_cnn_paths.keys()) | set(
                    _interp_cnn_dfs.keys()
                )
                for _cnn_label in _all_cnn_labels:
                    _cnn_path = str(_interp_cnn_paths.get(_cnn_label, "")).strip()
                    if _cnn_path and os.path.exists(_cnn_path):
                        _cnn_df = pd.read_csv(_cnn_path)
                        with_pose_df = merge_interpolated_cnn_df(
                            with_pose_df, _cnn_df, label=_cnn_label
                        )
                    elif _cnn_label in _interp_cnn_dfs:
                        _cnn_df = _interp_cnn_dfs[_cnn_label]
                        if isinstance(_cnn_df, pd.DataFrame) and not _cnn_df.empty:
                            with_pose_df = merge_interpolated_cnn_df(
                                with_pose_df, _cnn_df, label=_cnn_label
                            )
            except Exception:
                logger.debug("Interpolated CNN merge skipped.", exc_info=True)

            # --- Merge interpolated head-tail directions ---
            _interp_ht_path = str(
                getattr(self, "current_interpolated_headtail_csv_path", None) or ""
            ).strip()
            _interp_ht_df = getattr(self, "current_interpolated_headtail_df", None)
            try:
                from hydra_suite.core.identity.properties.export import (
                    merge_interpolated_headtail_df,
                )

                if _interp_ht_path and os.path.exists(_interp_ht_path):
                    _ht_df = pd.read_csv(_interp_ht_path)
                    with_pose_df = merge_interpolated_headtail_df(with_pose_df, _ht_df)
                elif (
                    isinstance(_interp_ht_df, pd.DataFrame) and not _interp_ht_df.empty
                ):
                    with_pose_df = merge_interpolated_headtail_df(
                        with_pose_df, _interp_ht_df
                    )
            except Exception:
                logger.debug("Interpolated head-tail merge skipped.", exc_info=True)
        except Exception:
            logger.exception(
                "Pose export skipped: failed while merging pose sources (cache=%s, interpolated=%s)",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
            )
            return None

        if with_pose_df is None or with_pose_df.empty:
            logger.warning(
                "Pose export skipped: merged pose dataframe is empty for %s",
                final_csv_path,
            )
            return None

        # Extract pose labels from the merged DataFrame
        import re as _re

        _kpt_re = _re.compile(r"^PoseKpt_(.+)_X$")
        pose_labels = [
            m.group(1) for col in with_pose_df.columns if (m := _kpt_re.match(str(col)))
        ]

        if pose_labels:
            params = self.get_parameters_dict()
            # Resolve anterior/posterior indices for body-length calibration
            from hydra_suite.core.identity.pose.features import (
                resolve_pose_group_indices,
            )

            kpt_names = []
            try:
                from hydra_suite.core.identity.properties.cache import (
                    IndividualPropertiesCache,
                )

                _cache_path = str(
                    self.current_individual_properties_cache_path or ""
                ).strip()
                if _cache_path and os.path.exists(_cache_path):
                    _cache = IndividualPropertiesCache(_cache_path, mode="r")
                    try:
                        kpt_names = [
                            str(v)
                            for v in (
                                _cache.metadata.get("pose_keypoint_names", []) or []
                            )
                        ]
                    finally:
                        _cache.close()
            except Exception:
                pass
            anterior_indices = resolve_pose_group_indices(
                params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), kpt_names
            )
            posterior_indices = resolve_pose_group_indices(
                params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), kpt_names
            )

            # Load skeleton edges from the skeleton JSON for anatomy checks
            skeleton_edges = []
            try:
                _skel_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
                if _skel_file and os.path.exists(_skel_file):
                    with open(_skel_file, "r", encoding="utf-8") as _sf:
                        _skel_data = json.load(_sf)
                    for _edge in _skel_data.get(
                        "skeleton_edges", _skel_data.get("edges", [])
                    ):
                        if isinstance(_edge, (list, tuple)) and len(_edge) >= 2:
                            try:
                                skeleton_edges.append((int(_edge[0]), int(_edge[1])))
                            except Exception:
                                pass
            except Exception:
                logger.exception(
                    "Failed to load skeleton edges for anatomy check; skipping."
                )
                skeleton_edges = []

            # Calibrate body-length prior from high-confidence frames
            body_length_prior = None
            if anterior_indices and posterior_indices:
                try:
                    body_length_prior = calibrate_body_length_prior(
                        with_pose_df,
                        pose_labels,
                        anterior_indices,
                        posterior_indices,
                        min_valid_conf=float(
                            params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
                        ),
                    )
                    if body_length_prior.is_valid:
                        logger.info(
                            "Body-length prior calibrated: median=%.1f px, MAD=%.1f px, n=%d",
                            body_length_prior.median_px,
                            body_length_prior.mad_px,
                            body_length_prior.n_samples,
                        )
                except Exception:
                    logger.exception(
                        "Body-length prior calibration failed; skipping anatomy check."
                    )
                    body_length_prior = None

            # Calibrate per-edge length priors from high-confidence frames
            edge_length_priors = None
            if skeleton_edges:
                try:
                    edge_length_priors = calibrate_edge_length_priors(
                        with_pose_df,
                        pose_labels,
                        skeleton_edges,
                        min_valid_conf=float(
                            params.get("POSE_MIN_KPT_CONF_VALID", 0.2)
                        ),
                    )
                    if edge_length_priors.is_valid:
                        logger.info(
                            "Edge-length priors calibrated for %d edges.",
                            len(edge_length_priors.priors),
                        )
                except Exception:
                    logger.exception(
                        "Edge-length prior calibration failed; skipping skeleton check."
                    )
                    edge_length_priors = None

            # Per-frame quality gate
            try:
                with_pose_df = apply_quality_to_dataframe(
                    with_pose_df,
                    pose_labels,
                    params,
                    body_length_prior=body_length_prior,
                    anterior_indices=anterior_indices if anterior_indices else None,
                    posterior_indices=posterior_indices if posterior_indices else None,
                    skeleton_edges=skeleton_edges if skeleton_edges else None,
                    edge_length_priors=edge_length_priors,
                )
            except Exception:
                logger.exception("Pose quality gating failed; using unfiltered pose.")

            # Temporal post-processing per trajectory
            max_gap = int(params.get("POSE_POSTPROC_MAX_GAP", 5))
            z_threshold = float(params.get("POSE_TEMPORAL_OUTLIER_ZSCORE", 3.0))
            if z_threshold > 0.0 and "TrajectoryID" in with_pose_df.columns:
                try:
                    parts = []
                    for _, traj_group in with_pose_df.groupby(
                        "TrajectoryID", sort=False
                    ):
                        parts.append(
                            apply_temporal_pose_postprocessing(
                                traj_group,
                                pose_labels,
                                max_gap=max_gap,
                                z_score_threshold=z_threshold,
                            )
                        )
                    if parts:
                        with_pose_df = (
                            pd.concat(parts, ignore_index=True)
                            .sort_values(["TrajectoryID", "FrameID"], kind="stable")
                            .reset_index(drop=True)
                        )
                except Exception:
                    logger.exception(
                        "Pose temporal post-processing failed; using unfiltered pose."
                    )

        return with_pose_df

    def _export_pose_augmented_csv(self, final_csv_path):
        """Write a pose-augmented trajectories CSV next to the final CSV."""
        with_pose_df = self._build_pose_augmented_dataframe(final_csv_path)
        if with_pose_df is None or with_pose_df.empty:
            return None

        base, ext = os.path.splitext(final_csv_path)
        with_pose_path = f"{base}_with_pose{ext or '.csv'}"
        try:
            with_pose_df.to_csv(with_pose_path, index=False)
        except Exception:
            logger.exception("Failed to save pose-augmented CSV to: %s", with_pose_path)
            return None

        logger.info("Pose-augmented trajectories saved to: %s", with_pose_path)
        return with_pose_path

    def _relink_final_pose_augmented_csv(self, final_csv_path):
        """Rewrite final CSV IDs after pose-aware relinking and regenerate _with_pose.csv."""
        if not final_csv_path or not os.path.exists(final_csv_path):
            return None

        with_pose_df = self._build_pose_augmented_dataframe(final_csv_path)
        params = self.get_parameters_dict()

        try:
            base_df = pd.read_csv(final_csv_path)
        except Exception:
            logger.exception(
                "Relinking skipped: failed to reload final CSV: %s", final_csv_path
            )
            return self._export_pose_augmented_csv(final_csv_path)

        relink_input_df = (
            with_pose_df
            if with_pose_df is not None and not with_pose_df.empty
            else base_df
        )
        relinked_with_pose = relink_trajectories_with_pose(relink_input_df, params)
        if relinked_with_pose is None or relinked_with_pose.empty:
            relinked_with_pose = relink_input_df

        common_cols = [
            col for col in base_df.columns if col in relinked_with_pose.columns
        ]
        relinked_base = relinked_with_pose.loc[:, common_cols].copy()
        relinked_base = relinked_base.sort_values(
            ["TrajectoryID", "FrameID"], kind="stable"
        ).reset_index(drop=True)
        relinked_with_pose = relinked_with_pose.sort_values(
            ["TrajectoryID", "FrameID"], kind="stable"
        ).reset_index(drop=True)

        try:
            relinked_base.to_csv(final_csv_path, index=False)
        except Exception:
            logger.exception("Failed to rewrite relinked final CSV: %s", final_csv_path)
            return None

        base, ext = os.path.splitext(final_csv_path)
        with_pose_path = f"{base}_with_pose{ext or '.csv'}"
        if with_pose_df is not None and not with_pose_df.empty:
            try:
                relinked_with_pose.to_csv(with_pose_path, index=False)
            except Exception:
                logger.exception(
                    "Failed to rewrite relinked pose-augmented CSV: %s", with_pose_path
                )
                return None
        elif os.path.exists(with_pose_path):
            try:
                os.remove(with_pose_path)
            except Exception:
                logger.warning(
                    "Failed to remove stale pose-augmented CSV: %s", with_pose_path
                )

        logger.info(
            "Relinked final CSV rewritten: %s (%d trajectories)",
            final_csv_path,
            (
                int(relinked_base["TrajectoryID"].nunique())
                if "TrajectoryID" in relinked_base.columns
                else 0
            ),
        )
        if with_pose_df is not None and not with_pose_df.empty:
            logger.info("Relinked pose-augmented CSV saved: %s", with_pose_path)
            return with_pose_path
        return final_csv_path

    def _load_video_trajectories(self, final_csv_path):
        """Load best available trajectories for video generation (prefers pose-augmented CSV)."""
        if not final_csv_path:
            return None, None
        base, ext = os.path.splitext(final_csv_path)
        with_pose_path = f"{base}_with_pose{ext or '.csv'}"
        candidate = with_pose_path if os.path.exists(with_pose_path) else final_csv_path
        if not os.path.exists(candidate):
            return None, None
        try:
            return pd.read_csv(candidate), candidate
        except Exception:
            logger.exception("Failed to load video trajectories from: %s", candidate)
            return None, None

    def _run_pending_video_generation_or_finalize(self):
        """Run video generation if queued; otherwise finalize UI/session cleanup."""
        if self._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        csv_path = self._pending_video_csv_path
        should_render_video = bool(self._pending_video_generation and csv_path)
        self._pending_video_generation = False
        self._pending_video_csv_path = None
        self._pending_pose_export_csv_path = None

        if should_render_video:
            trajectories_df, loaded_path = self._load_video_trajectories(csv_path)
            if trajectories_df is None or trajectories_df.empty:
                logger.warning(
                    "Skipping final video generation: no trajectories loaded from %s",
                    csv_path,
                )
                self._finalize_tracking_session_ui()
                return
            logger.info(
                "Final video rendering uses trajectories from: %s",
                loaded_path or csv_path,
            )
            self._generate_video_from_trajectories(
                trajectories_df,
                csv_path=csv_path,
                finalize_on_complete=False,
            )
            return

        self._finalize_tracking_session_ui()

    def _finish_tracking_session(self, final_csv_path=None):
        """Complete tracking session cleanup and UI updates."""
        if self._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        self._session_final_csv_path = final_csv_path
        # Hide progress elements
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if final_csv_path:
            self._pending_pose_export_csv_path = final_csv_path
            self._export_pose_augmented_csv(final_csv_path)

        self._pending_video_csv_path = final_csv_path
        self._pending_video_generation = bool(
            final_csv_path
            and self._postprocess_panel.check_video_output.isChecked()
            and self._postprocess_panel.video_out_line.text().strip()
        )

        # Generate dataset if enabled (BEFORE cleanup so files are still available)
        if self._dataset_panel.chk_enable_dataset_gen.isChecked():
            self._generate_training_dataset(override_csv_path=final_csv_path)
            self._dataset_was_started = True

        # Interpolate occlusions for individual analysis (post-pass).
        # This also powers pose enrichment on occluded frames in final CSV.
        if self._should_run_interpolated_postpass():
            started = self._generate_interpolated_individual_crops(final_csv_path)
            if started:
                # Hold final UI/session completion until interpolation finishes.
                self._pending_finish_after_interp = True
                return

        if final_csv_path:
            self._relink_final_pose_augmented_csv(final_csv_path)

        if self._start_pending_oriented_track_video_export(final_csv_path):
            return

        self._run_pending_video_generation_or_finalize()

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False
        self._pending_finish_after_track_videos = False
        self.current_interpolated_pose_df = None
        self.current_interpolated_roi_npz_path = None
        # Force-clear progress UI at terminal session state.
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        # Clean up session logging
        self._cleanup_session_logging()
        self._cleanup_temporary_files()

        # Hide stats labels
        self.label_current_fps.setVisible(False)
        self.label_elapsed_time.setVisible(False)
        self.label_eta.setVisible(False)

        # Determine if we are continuing a batch
        is_batch_continuing = (
            self._setup_panel.g_batch.isChecked()
            and self.current_batch_index >= 0
            and (self.current_batch_index + 1) < len(self.batch_videos)
        )

        if not is_batch_continuing:
            self._set_ui_controls_enabled(True)
            self.btn_start.blockSignals(True)
            self.btn_start.setChecked(False)
            self.btn_start.blockSignals(False)
            self.btn_start.setText("Start Full Tracking")
            self._apply_ui_state("idle" if self.current_video_path else "no_video")
            logger.info("✓ Tracking session complete.")

            # Show end-of-session summary. If the dataset worker is still running,
            # defer the summary until it finishes so we can include its result.
            if getattr(self, "_dataset_was_started", False) and self._is_worker_running(
                self.dataset_worker
            ):
                self._show_summary_on_dataset_done = True
            else:
                self._show_session_summary()
        else:
            logger.info("✓ Video complete. Continuing batch...")
            # Disable deferred summary for intermediate batch items so it doesn't block
            self._show_summary_on_dataset_done = False

        # --- Batch Mode Continuation ---
        if self._setup_panel.g_batch.isChecked() and self.current_batch_index >= 0:
            self.current_batch_index += 1
            if self.current_batch_index < len(self.batch_videos):
                # Load next video
                fp = self.batch_videos[self.current_batch_index]
                self._setup_panel.list_batch_videos.setCurrentRow(
                    self.current_batch_index
                )

                # We MUST skip_config_load here to preserve the keystone parameters
                # currently in the UI so they are applied to this video.
                self._setup_video_file(fp, skip_config_load=True)

                # Small delay to ensure UI updates before starting next
                logger.info(
                    f"Batch Mode: Starting next video ({self.current_batch_index + 1}/{len(self.batch_videos)})"
                )
                QTimer.singleShot(1000, lambda: self.start_tracking(preview_mode=False))
            else:
                # Batch complete
                self.current_batch_index = -1
                QMessageBox.information(
                    self,
                    "Batch Complete",
                    f"Finished processing {len(self.batch_videos)} videos.",
                )
        else:
            # Ensure reset if batch mode is disabled mid-run or not used
            self.current_batch_index = -1

    def _generate_interpolated_individual_crops(self, csv_path):
        """Post-pass interpolation for occluded segments in individual dataset."""
        try:
            if self._stop_all_requested:
                return False
            if not self._identity_panel.chk_individual_interpolate.isChecked():
                return False

            target_csv = None
            if csv_path and os.path.exists(csv_path):
                target_csv = csv_path
            elif self._setup_panel.csv_line.text() and os.path.exists(
                self._setup_panel.csv_line.text()
            ):
                target_csv = self._setup_panel.csv_line.text()
            if not target_csv or not os.path.exists(target_csv):
                return False

            video_path = self._setup_panel.file_line.text()
            if not video_path or not os.path.exists(video_path):
                return False

            params = self.get_parameters_dict()
            output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
            if not output_dir:
                # Keep interpolated analysis available even when image-save toggle is off.
                csv_dir = os.path.dirname(target_csv) if target_csv else ""
                fallback_output = (
                    os.path.join(csv_dir, "training_data") if csv_dir else ""
                )
                if fallback_output:
                    try:
                        os.makedirs(fallback_output, exist_ok=True)
                    except Exception:
                        pass
                    params["INDIVIDUAL_DATASET_OUTPUT_DIR"] = fallback_output
                    logger.info(
                        "Interpolated analysis output dir not set; using fallback: %s",
                        fallback_output,
                    )

            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Interpolating occluded crops...")

            if self.interp_worker is not None and self.interp_worker.isRunning():
                logger.warning(
                    "Interpolated crop generation already in progress; skipping duplicate request."
                )
                return True
            if self.interp_worker is not None and not self.interp_worker.isRunning():
                self.interp_worker.deleteLater()
                self.interp_worker = None

            self.current_interpolated_roi_npz_path = None
            self.current_interpolated_pose_csv_path = None
            self.current_interpolated_pose_df = None
            self.current_interpolated_tag_csv_path = None
            self.current_interpolated_tag_df = None
            self.current_interpolated_cnn_csv_paths = {}
            self.current_interpolated_cnn_dfs = {}
            self.current_interpolated_headtail_csv_path = None
            self.current_interpolated_headtail_df = None

            _interp_profiling = bool(params.get("ENABLE_PROFILING", False))
            _interp_profile_path = None
            if _interp_profiling:
                if self.current_detection_cache_path:
                    _interp_profile_path = str(
                        Path(self.current_detection_cache_path).parent
                        / "interp_profile.json"
                    )
                elif video_path:
                    _interp_profile_path = str(
                        Path(video_path).parent / "interp_profile.json"
                    )

            self.interp_worker = InterpolatedCropsWorker(
                target_csv,
                video_path,
                self.current_detection_cache_path,
                params,
                enable_profiling=_interp_profiling,
                profile_export_path=_interp_profile_path,
            )
            self.interp_worker.progress_signal.connect(self.on_progress_update)
            self.interp_worker.finished_signal.connect(
                self._on_interpolated_crops_finished
            )
            self.interp_worker.start()
            return True
        except Exception as e:
            logger.warning(f"Interpolated individual crops failed: {e}")
            return False

    def _interp_angle(self, theta_start, theta_end, t):
        deg0 = math.degrees(theta_start)
        deg1 = math.degrees(theta_end)
        # Treat OBB angles as 180-degree periodic; pick shortest path.
        candidates = (deg1, deg1 + 180.0, deg1 - 180.0)
        best_delta = None
        for cand in candidates:
            delta = wrap_angle_degs(cand - deg0)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        return math.radians(deg0 + (best_delta or 0.0) * t)

    def _get_detection_size(self, detection_cache, frame_id, detection_id, params):
        if detection_cache is None or detection_id is None or pd.isna(detection_id):
            return None, None
        try:
            _, _, shapes, _, obb_corners, detection_ids, *_ = detection_cache.get_frame(
                int(frame_id)
            )
        except Exception:
            return None, None

        idx = None
        try:
            for i, did in enumerate(detection_ids):
                if int(did) == int(detection_id):
                    idx = i
                    break
        except Exception:
            idx = None

        if idx is None:
            return None, None

        # YOLO OBB corners
        if obb_corners and idx < len(obb_corners):
            c = np.asarray(obb_corners[idx], dtype=np.float32)
            if c.shape[0] >= 4:
                w = float(np.linalg.norm(c[1] - c[0]))
                h = float(np.linalg.norm(c[2] - c[1]))
                if w < h:
                    w, h = h, w
                return w, h

        # Background subtraction shapes
        if shapes and idx < len(shapes):
            area, aspect_ratio = shapes[idx][0], shapes[idx][1]
            if aspect_ratio > 0 and area > 0:
                ax2 = math.sqrt(4 * area / (math.pi * aspect_ratio))
                ax1 = aspect_ratio * ax2
                return ax1, ax2

        return None, None

    def start_backward_tracking(self: object) -> object:
        """start_backward_tracking method documentation."""
        if self._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting backward tracking pass (using cached detections)...")
        logger.info("=" * 80)

        video_fp = self._setup_panel.file_line.text()
        if not video_fp:
            return

        # Use original video (no reversal needed with detection caching)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Starting backward tracking (using cached detections)..."
        )
        QApplication.processEvents()

        # Start backward tracking directly on original video with cached detections
        self.start_tracking_on_video(video_fp, backward_mode=True)

    def start_tracking(
        self: object, preview_mode: bool, backward_mode: bool = False
    ) -> object:
        """start_tracking method documentation."""
        if not preview_mode:
            # If batch mode group is checked, initialize batch processing
            if self._setup_panel.g_batch.isChecked():
                if self.current_batch_index < 0:
                    res = QMessageBox.question(
                        self,
                        "Start Batch Process",
                        f"This will process {len(self.batch_videos)} videos sequentially using the CURRENT parameters.\n\n"
                        "Each video will have its own CSV and configuration file saved in its source directory.\n\n"
                        "Continue?",
                        QMessageBox.Yes | QMessageBox.No,
                    )
                    if res == QMessageBox.No:
                        return

                    # Start at the first video (Keystone)
                    self.current_batch_index = 0
                    self._sync_keystone_to_batch()
                    fp = self.batch_videos[0]
                    self._setup_panel.list_batch_videos.setCurrentRow(0)

                    # Ensure the keystone video is loaded WITHOUT overwriting current UI params
                    if self.current_video_path != fp:
                        self._setup_video_file(fp, skip_config_load=True)

            # Save config for the CURRENTLY LOADED video (this persists the keystone's params to the current video)
            # In batch mode, we automatically overwrite to avoid halting the automated process.
            if not self.save_config(
                prompt_if_exists=not self._setup_panel.g_batch.isChecked()
            ):
                # User cancelled config save, abort tracking
                self.current_batch_index = -1  # Reset batch if cancelled
                return

        video_fp = self._setup_panel.file_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No video", "Please select a video file first.")
            return
        if preview_mode:
            self.start_preview_on_video(video_fp)
        else:
            self.start_tracking_on_video(video_fp, backward_mode=False)

    def start_preview_on_video(self: object, video_path: object) -> object:
        """start_preview_on_video method documentation."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return
        self._stop_all_requested = False
        self._pending_finish_after_interp = False

        # Stop video playback if active
        if self.is_playing:
            self._stop_playback()

        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True
        self.csv_writer_thread = None

        params = self.get_parameters_dict()
        if not self._validate_yolo_model_requirements(
            params, mode_label="tracking preview"
        ):
            return
        # Preview should always render frames regardless of visualization-free toggle
        params["VISUALIZATION_FREE_MODE"] = False
        # Preview must not use ONNX/TensorRT — downgrade to the native device runtime.
        safe_rt = self._preview_safe_runtime(params.get("COMPUTE_RUNTIME", "cpu"))
        if safe_rt != params.get("COMPUTE_RUNTIME"):
            safe_det = derive_detection_runtime_settings(safe_rt)
            params["COMPUTE_RUNTIME"] = safe_rt
            params["YOLO_DEVICE"] = safe_det["yolo_device"]
            params["ENABLE_GPU_BACKGROUND"] = safe_det["enable_gpu_background"]
            params["ENABLE_TENSORRT"] = safe_det["enable_tensorrt"]
            params["ENABLE_ONNX_RUNTIME"] = safe_det["enable_onnx_runtime"]
            safe_pose = derive_pose_runtime_settings(
                safe_rt, backend_family=params.get("POSE_MODEL_TYPE", "yolo")
            )
            params["POSE_RUNTIME_FLAVOR"] = safe_pose["pose_runtime_flavor"]

        # Reuse an existing detection cache when one covers the current frame
        # range — this ensures preview uses the same detections as the autotune
        # preview instead of re-running live YOLO inference, which can produce
        # slightly different detection sets and cause visible jumps.
        start_frame = params.get("START_FRAME", 0)
        end_frame = params.get("END_FRAME", 0)
        cache_path, cache_valid = self._find_or_plan_optimizer_cache_path(
            video_path, params, start_frame, end_frame
        )

        if not cache_valid:
            res = QMessageBox.question(
                self,
                "Build Detection Cache",
                "No detection cache found for this frame range.\n\n"
                "Build one now for a consistent preview?\n"
                "(Detection-only scan — no CSV output, no config save.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                self._pending_preview_video_path = video_path
                self._cache_builder_worker = DetectionCacheBuilderWorker(
                    video_path,
                    cache_path,
                    params,
                    start_frame,
                    end_frame,
                )
                self._cache_builder_worker.progress_signal.connect(
                    self.on_progress_update
                )
                self._cache_builder_worker.finished_signal.connect(
                    self._on_preview_cache_built
                )
                self.progress_bar.setVisible(True)
                self.progress_label.setVisible(True)
                self.progress_bar.setValue(0)
                self.progress_label.setText("Building detection cache for preview...")
                self._cache_builder_worker.start()
                return  # Will resume via _on_preview_cache_built

        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=False,
            detection_cache_path=cache_path if cache_valid else None,
            preview_mode=True,
            use_cached_detections=cache_valid,
        )
        self.tracking_worker.set_parameters(params)
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)
        self.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preview Mode Active")

        self._prepare_tracking_display()
        self._apply_ui_state("preview")
        self.tracking_worker.start()

    def start_tracking_on_video(
        self: object, video_path: object, backward_mode: object = False
    ) -> object:
        """start_tracking_on_video method documentation."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return
        self._stop_all_requested = False
        self._pending_finish_after_interp = False
        if not backward_mode:
            # Reset per-session summary state for each new forward tracking run.
            self._session_result_dataset = None
            self._dataset_was_started = False
            self._show_summary_on_dataset_done = False
            self._session_wall_start = time.time()
            self._session_final_csv_path = None
            self._session_fps_list = []
            self._session_frames_processed = 0

        # Stop video playback if active
        if self.is_playing:
            self._stop_playback()

        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True

        # Session logging is already set up in start_full() - don't duplicate here
        # For backward mode, we reuse the same session log

        self.csv_writer_thread = None
        if self._setup_panel.csv_line.text():
            # Determine header based on confidence tracking setting
            save_confidence = self._setup_panel.check_save_confidence.isChecked()
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
                    "DetectionID",
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
                    "DetectionID",
                ]
            # Append TagID column when AprilTag identity is active
            if self._selected_identity_method() == "apriltags":
                hdr.append("TagID")
            csv_path = self._setup_panel.csv_line.text()
            base, ext = os.path.splitext(csv_path)
            if backward_mode:
                csv_path = f"{base}_backward{ext}"
            elif self._tracking_panel.chk_enable_backward.isChecked():
                # Forward mode with backward tracking enabled - save as _forward.csv
                csv_path = f"{base}_forward{ext}"
            self.csv_writer_thread = CSVWriterThread(csv_path, header=hdr)
            self.csv_writer_thread.start()

        # Video output is no longer generated during tracking
        # Instead, it's generated from post-processed trajectories after merging
        # This ensures the video shows clean, merged trajectories with stable IDs
        video_output_path = None

        # Generate detection cache path based on video and detection method
        # Cache is always created for forward tracking to allow reuse on reruns
        detection_cache_path = None
        params = self.get_parameters_dict()
        logger.info(
            f"Launching {'backward' if backward_mode else 'forward'} tracking for frame range "
            f"{params.get('START_FRAME')}..{params.get('END_FRAME')}"
        )
        detection_method = params.get("DETECTION_METHOD", "background_subtraction")
        use_cached_detections = self._setup_panel.chk_use_cached_detections.isChecked()
        if not self._validate_yolo_model_requirements(params, mode_label="tracking"):
            return

        # Generate model-specific cache name
        def get_cache_model_ids() -> object:
            """Generate raw-detection and TensorRT-engine cache identity keys."""
            # Include resize factor in cache ID since detections are scale-dependent
            resize_factor = params.get("RESIZE_FACTOR", 1.0)
            resize_str = f"r{int(resize_factor * 100)}"

            def normalize_for_hash(value: object) -> object:
                """Convert values to deterministic, JSON-safe forms for hashing."""
                if isinstance(value, np.ndarray):
                    arr = np.ascontiguousarray(value)
                    return {
                        "type": "ndarray",
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                        "digest": hashlib.md5(arr.tobytes()).hexdigest(),
                    }
                if isinstance(value, np.integer):
                    return int(value)
                if isinstance(value, np.floating):
                    if np.isnan(value):
                        return "NaN"
                    if np.isinf(value):
                        return "Infinity" if value > 0 else "-Infinity"
                    return float(value)
                if isinstance(value, np.bool_):
                    return bool(value)
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, dict):
                    return {
                        str(k): normalize_for_hash(v)
                        for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                    }
                if isinstance(value, (list, tuple)):
                    return [normalize_for_hash(v) for v in value]
                return value

            def extract_hash_params(keys: object) -> object:
                return {k: normalize_for_hash(params.get(k)) for k in keys}

            def build_cache_id(
                prefix: str, cache_params: object, model_stem: str = ""
            ) -> str:
                digest = hashlib.md5(
                    json.dumps(cache_params, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
                if model_stem:
                    return f"{prefix}_{model_stem}_{resize_str}_{digest}"
                return f"{prefix}_{resize_str}_{digest}"

            def get_model_fingerprint(model_path: object) -> object:
                configured = str(model_path or "")
                resolved = str(resolve_model_path(configured))
                fingerprint = {
                    "configured_path": configured,
                    "resolved_path": resolved,
                }
                if resolved and os.path.exists(resolved):
                    try:
                        stat = os.stat(resolved)
                        fingerprint["size_bytes"] = stat.st_size
                        fingerprint["mtime_ns"] = stat.st_mtime_ns
                    except OSError:
                        fingerprint["size_bytes"] = None
                        fingerprint["mtime_ns"] = None
                else:
                    fingerprint["size_bytes"] = None
                    fingerprint["mtime_ns"] = None
                return fingerprint

            # Common inference settings that affect detections for both methods.
            common_detection_keys = (
                "DETECTION_METHOD",
                "RESIZE_FACTOR",
                "MAX_TARGETS",
                "COMPUTE_RUNTIME",
            )

            if detection_method == "yolo_obb":
                yolo_mode = str(params.get("YOLO_OBB_MODE", "direct")).strip().lower()
                direct_model = params.get(
                    "YOLO_OBB_DIRECT_MODEL_PATH",
                    params.get("YOLO_MODEL_PATH", "best.pt"),
                )
                crop_obb_model = params.get(
                    "YOLO_CROP_OBB_MODEL_PATH", params.get("YOLO_MODEL_PATH", "best.pt")
                )
                active_obb_model = (
                    direct_model if yolo_mode == "direct" else crop_obb_model
                )
                model_fingerprint = get_model_fingerprint(active_obb_model)
                model_name = os.path.basename(
                    model_fingerprint["resolved_path"]
                    or model_fingerprint["configured_path"]
                )
                model_stem = os.path.splitext(model_name)[0] or "model"
                safe_model_stem = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_" for c in model_stem
                )

                yolo_inference_keys = (
                    "YOLO_TARGET_CLASSES",
                    "YOLO_DEVICE",
                    "ENABLE_TENSORRT",
                    "TENSORRT_MAX_BATCH_SIZE",
                    "YOLO_OBB_MODE",
                    "YOLO_SEQ_CROP_PAD_RATIO",
                    "YOLO_SEQ_MIN_CROP_SIZE_PX",
                    "YOLO_SEQ_ENFORCE_SQUARE_CROP",
                    "YOLO_SEQ_STAGE2_IMGSZ",
                    "YOLO_SEQ_STAGE2_POW2_PAD",
                    "YOLO_HEADTAIL_CONF_THRESHOLD",
                    "POSE_OVERRIDES_HEADTAIL",
                )
                cache_params = {
                    "common": extract_hash_params(common_detection_keys),
                    "yolo": extract_hash_params(yolo_inference_keys),
                    "models": normalize_for_hash(
                        {
                            "active_obb": model_fingerprint,
                            "direct_obb": get_model_fingerprint(direct_model),
                            "detect": get_model_fingerprint(
                                params.get("YOLO_DETECT_MODEL_PATH", "")
                            ),
                            "crop_obb": get_model_fingerprint(crop_obb_model),
                            "headtail": get_model_fingerprint(
                                params.get("YOLO_HEADTAIL_MODEL_PATH", "")
                            ),
                        }
                    ),
                    # Bump when raw detection extraction/filtering semantics change.
                    "raw_detection_cache_version": 4,
                }
                # Class order should not change cache identity.
                classes = cache_params["yolo"].get("YOLO_TARGET_CLASSES")
                if classes is not None:
                    if isinstance(classes, str):
                        raw_classes = [
                            c.strip() for c in classes.split(",") if c.strip()
                        ]
                    elif isinstance(classes, (list, tuple)):
                        raw_classes = list(classes)
                    else:
                        raw_classes = [classes]
                    try:
                        cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                            int(c) for c in raw_classes
                        )
                    except (TypeError, ValueError):
                        cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                            str(c) for c in raw_classes
                        )

                build_batch_size = params.get(
                    "TENSORRT_BUILD_BATCH_SIZE",
                    params.get("TENSORRT_MAX_BATCH_SIZE", 1),
                )
                try:
                    build_batch_size = max(1, int(build_batch_size or 1))
                except (TypeError, ValueError):
                    build_batch_size = max(
                        1, int(params.get("TENSORRT_MAX_BATCH_SIZE", 1) or 1)
                    )
                try:
                    build_workspace_gb = float(
                        params.get("TENSORRT_BUILD_WORKSPACE_GB", 4.0)
                    )
                except (TypeError, ValueError):
                    build_workspace_gb = 4.0

                engine_cache_params = {
                    "engine": {
                        "runtime": "tensorrt",
                        "device": normalize_for_hash(params.get("YOLO_DEVICE")),
                        "build_batch_size": build_batch_size,
                        "workspace_gb": round(max(0.5, build_workspace_gb), 3),
                        "active_obb": model_fingerprint,
                        "export_profile": "trt_fp16_static_v1",
                    },
                    "engine_cache_version": 1,
                }

                return {
                    "inference": build_cache_id(
                        "yolo", cache_params, model_stem=safe_model_stem
                    ),
                    "engine": build_cache_id(
                        "yolo_engine",
                        engine_cache_params,
                        model_stem=safe_model_stem,
                    ),
                }

            bg_detection_keys = (
                "MAX_CONTOUR_MULTIPLIER",
                "ENABLE_SIZE_FILTERING",
                "MIN_OBJECT_SIZE",
                "MAX_OBJECT_SIZE",
                "ROI_MASK",
                "BACKGROUND_PRIME_FRAMES",
                "ENABLE_ADAPTIVE_BACKGROUND",
                "BACKGROUND_LEARNING_RATE",
                "ENABLE_GPU_BACKGROUND",
                "GPU_DEVICE_ID",
                "THRESHOLD_VALUE",
                "MORPH_KERNEL_SIZE",
                "ENABLE_ADDITIONAL_DILATION",
                "DILATION_ITERATIONS",
                "DILATION_KERNEL_SIZE",
                "BRIGHTNESS",
                "CONTRAST",
                "GAMMA",
                "DARK_ON_LIGHT_BACKGROUND",
                "ENABLE_LIGHTING_STABILIZATION",
                "LIGHTING_SMOOTH_FACTOR",
                "LIGHTING_MEDIAN_WINDOW",
                "ENABLE_CONSERVATIVE_SPLIT",
                "CONSERVATIVE_KERNEL_SIZE",
                "CONSERVATIVE_ERODE_ITER",
                "MIN_CONTOUR_AREA",
                "MIN_DETECTIONS_TO_START",
                "MIN_DETECTION_COUNTS",
            )
            cache_params = {
                "common": extract_hash_params(common_detection_keys),
                "background_subtraction": extract_hash_params(bg_detection_keys),
            }
            return {
                "inference": build_cache_id("bgsub", cache_params),
                "engine": None,
            }

        cache_ids = get_cache_model_ids()
        model_id = cache_ids["inference"]
        params["INFERENCE_MODEL_ID"] = model_id
        if cache_ids.get("engine"):
            params["ENGINE_MODEL_ID"] = cache_ids["engine"]
        csv_dir = (
            os.path.dirname(self._setup_panel.csv_line.text())
            if self._setup_panel.csv_line.text()
            else ""
        )
        artifact_base_dirs = candidate_artifact_base_dirs(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        if artifact_base_dir != Path(video_path).parent:
            logger.warning(
                "Video directory not writable; using artifact root: %s",
                artifact_base_dir,
            )

        existing_detection_cache = find_existing_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dirs=artifact_base_dirs,
        )
        if existing_detection_cache is not None:
            detection_cache_path = str(existing_detection_cache)
        else:
            detection_cache_path = str(
                build_detection_cache_path(
                    video_path,
                    model_id,
                    artifact_base_dir=artifact_base_dir,
                )
            )

        # Do NOT delete old detection caches; keep all for reuse
        self.current_detection_cache_path = detection_cache_path

        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=self.csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
            detection_cache_path=detection_cache_path,
            preview_mode=False,  # Full tracking mode - batching enabled if applicable
            use_cached_detections=use_cached_detections,
        )
        self.tracking_worker.set_parameters(params)
        self.parameters_changed.connect(self.tracking_worker.update_parameters)
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)
        self.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Backward Tracking..." if backward_mode else "Forward Tracking..."
        )

        self._prepare_tracking_display()
        self._apply_ui_state("tracking")
        self.tracking_worker.start()

    def get_parameters_dict(self: object) -> object:
        """get_parameters_dict method documentation."""
        N = self._setup_panel.spin_max_targets.value()
        np.random.seed(42)
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]

        det_method = (
            "background_subtraction"
            if self._detection_panel.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

        yolo_mode = (
            "sequential"
            if self._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
            else "direct"
        )
        yolo_direct_path = resolve_model_path(
            self._get_selected_yolo_model_path() or ""
        )
        yolo_detect_path = resolve_model_path(
            self._get_selected_yolo_detect_model_path() or ""
        )
        yolo_crop_obb_path = resolve_model_path(
            self._get_selected_yolo_crop_obb_model_path() or ""
        )
        yolo_headtail_path = resolve_model_path(
            self._get_selected_yolo_headtail_model_path() or ""
        )
        yolo_path = yolo_direct_path if yolo_mode == "direct" else yolo_crop_obb_path

        yolo_cls = None
        if self._detection_panel.line_yolo_classes.text().strip():
            try:
                yolo_cls = [
                    int(x.strip())
                    for x in self._detection_panel.line_yolo_classes.text().split(",")
                ]
            except ValueError:
                pass

        # Calculate actual pixel values from body-size multipliers
        reference_body_size = self._detection_panel.spin_reference_body_size.value()
        resize_factor = self._setup_panel.spin_resize.value()
        scaled_body_size = reference_body_size * resize_factor

        # Area is π * (diameter/2)^2
        import math

        reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
        scaled_body_area = reference_body_area * (resize_factor**2)

        # Convert multipliers to actual pixels
        min_object_size_pixels = int(
            self._detection_panel.spin_min_object_size.value() * scaled_body_area
        )
        max_object_size_pixels = int(
            self._detection_panel.spin_max_object_size.value() * scaled_body_area
        )
        max_distance_pixels = (
            self._tracking_panel.spin_max_dist.value() * scaled_body_size
        )
        recovery_search_distance_pixels = (
            self._tracking_panel.spin_continuity_thresh.value() * scaled_body_size
        )
        min_respawn_distance_pixels = (
            self._tracking_panel.spin_min_respawn_distance.value() * scaled_body_size
        )

        # Convert time-based velocities to frame-based for tracking
        fps = self._setup_panel.spin_fps.value()
        velocity_threshold_pixels_per_frame = (
            self._tracking_panel.spin_velocity.value() * scaled_body_size / fps
        )
        max_velocity_break_pixels_per_frame = (
            self._postprocess_panel.spin_max_velocity_break.value()
            * scaled_body_size
            / fps
        )

        # Convert time-based durations (seconds) to frame counts
        def _seconds_to_frames(seconds: float, min_frames: int = 1) -> int:
            """Convert a duration in seconds to an integer frame count."""
            return max(min_frames, round(seconds * fps))

        lost_threshold_frames = _seconds_to_frames(
            self._tracking_panel.spin_lost_thresh.value()
        )
        kalman_maturity_age = _seconds_to_frames(
            self._tracking_panel.spin_kalman_maturity_age.value()
        )
        bg_prime_frames = _seconds_to_frames(
            self._detection_panel.spin_bg_prime.value(), min_frames=0
        )
        min_detections_to_start = _seconds_to_frames(
            self._tracking_panel.spin_min_detections_to_start.value()
        )
        min_detection_counts = _seconds_to_frames(
            self._tracking_panel.spin_min_detect.value()
        )
        min_tracking_counts = _seconds_to_frames(
            self._tracking_panel.spin_min_track.value()
        )
        min_trajectory_length = _seconds_to_frames(
            self._postprocess_panel.spin_min_trajectory_length.value()
        )
        max_occlusion_gap = _seconds_to_frames(
            self._postprocess_panel.spin_max_occlusion_gap.value(), min_frames=0
        )
        velocity_zscore_window = _seconds_to_frames(
            self._postprocess_panel.spin_velocity_zscore_window.value(), min_frames=5
        )
        # YOLO Batching settings from UI (overrides advanced_config defaults)
        advanced_config = self.advanced_config.copy()
        advanced_config["enable_yolo_batching"] = (
            self._detection_panel.chk_enable_yolo_batching.isChecked()
        )
        advanced_config["yolo_batch_size_mode"] = (
            "auto"
            if self._detection_panel.combo_yolo_batch_mode.currentIndex() == 0
            else "manual"
        )
        advanced_config["yolo_manual_batch_size"] = (
            self._detection_panel.spin_yolo_batch_size.value()
        )
        advanced_config["video_show_pose"] = (
            self._postprocess_panel.check_video_show_pose.isChecked()
        )
        advanced_config["video_pose_point_radius"] = (
            self._postprocess_panel.spin_video_pose_point_radius.value()
        )
        advanced_config["video_pose_point_thickness"] = (
            self._postprocess_panel.spin_video_pose_point_thickness.value()
        )
        advanced_config["video_pose_line_thickness"] = (
            self._postprocess_panel.spin_video_pose_line_thickness.value()
        )
        advanced_config["video_pose_color_mode"] = (
            "track"
            if self._postprocess_panel.combo_video_pose_color_mode.currentIndex() == 0
            else "fixed"
        )
        advanced_config["video_pose_color"] = [
            int(self._postprocess_panel._video_pose_color[0]),
            int(self._postprocess_panel._video_pose_color[1]),
            int(self._postprocess_panel._video_pose_color[2]),
        ]
        # Canonical crop / aspect ratio params (from UI widgets)
        advanced_config["reference_aspect_ratio"] = (
            self._detection_panel.spin_reference_aspect_ratio.value()
        )
        advanced_config["enable_aspect_ratio_filtering"] = (
            self._detection_panel.chk_enable_aspect_ratio_filtering.isChecked()
        )
        advanced_config["min_aspect_ratio_multiplier"] = (
            self._detection_panel.spin_min_ar_multiplier.value()
        )
        advanced_config["max_aspect_ratio_multiplier"] = (
            self._detection_panel.spin_max_ar_multiplier.value()
        )

        individual_pipeline_enabled = self._is_individual_pipeline_enabled()
        individual_image_save_enabled = self._is_individual_image_save_enabled()
        pose_extractor_enabled = self._is_pose_inference_enabled()
        identity_cfg = self._identity_config()
        identity_method = self._selected_identity_method()  # kept for backward compat
        compute_runtime = self._selected_compute_runtime()
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        trt_batch_size = (
            self._detection_panel.spin_yolo_batch_size.value()
            if self._runtime_requires_fixed_yolo_batch(compute_runtime)
            else self._detection_panel.spin_tensorrt_batch.value()
        )
        trt_build_batch_size_raw = advanced_config.get(
            "tensorrt_build_batch_size", None
        )
        if trt_build_batch_size_raw in (None, "", 0, "0"):
            trt_build_batch_size = None
        else:
            try:
                trt_build_batch_size = max(1, int(trt_build_batch_size_raw))
            except (TypeError, ValueError):
                trt_build_batch_size = None
        pose_backend_family = (
            self._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        runtime_pose = derive_pose_runtime_settings(
            compute_runtime, backend_family=pose_backend_family
        )

        p = {
            "ADVANCED_CONFIG": advanced_config,  # Include advanced config for batch optimization
            "DETECTION_METHOD": det_method,
            "FPS": fps,  # Acquisition frame rate
            # Keep selected frame range stable even when controls are disabled
            # during tracking/backward pass.
            "START_FRAME": self._setup_panel.spin_start_frame.value(),
            "END_FRAME": self._setup_panel.spin_end_frame.value(),
            "YOLO_MODEL_PATH": yolo_path,
            "YOLO_OBB_MODE": yolo_mode,
            "YOLO_OBB_DIRECT_MODEL_PATH": yolo_direct_path,
            "YOLO_DETECT_MODEL_PATH": yolo_detect_path,
            "YOLO_CROP_OBB_MODEL_PATH": yolo_crop_obb_path,
            "YOLO_HEADTAIL_MODEL_PATH": yolo_headtail_path,
            "POSE_OVERRIDES_HEADTAIL": self._identity_panel.chk_pose_overrides_headtail.isChecked(),
            "YOLO_SEQ_CROP_PAD_RATIO": self._detection_panel.spin_yolo_seq_crop_pad.value(),
            "YOLO_SEQ_MIN_CROP_SIZE_PX": self._detection_panel.spin_yolo_seq_min_crop_px.value(),
            "YOLO_SEQ_ENFORCE_SQUARE_CROP": self._detection_panel.chk_yolo_seq_square_crop.isChecked(),
            "YOLO_SEQ_STAGE2_IMGSZ": self._detection_panel.spin_yolo_seq_stage2_imgsz.value(),
            "YOLO_SEQ_STAGE2_POW2_PAD": self._detection_panel.chk_yolo_seq_stage2_pow2_pad.isChecked(),
            "YOLO_SEQ_DETECT_CONF_THRESHOLD": self._detection_panel.spin_yolo_seq_detect_conf.value(),
            "YOLO_HEADTAIL_CONF_THRESHOLD": self._identity_panel.spin_yolo_headtail_conf.value(),
            "YOLO_CONFIDENCE_THRESHOLD": self._detection_panel.spin_yolo_confidence.value(),
            "YOLO_IOU_THRESHOLD": self._detection_panel.spin_yolo_iou.value(),
            "USE_CUSTOM_OBB_IOU_FILTERING": True,
            "YOLO_TARGET_CLASSES": yolo_cls,
            "COMPUTE_RUNTIME": compute_runtime,
            "YOLO_DEVICE": runtime_detection["yolo_device"],
            "ENABLE_GPU_BACKGROUND": runtime_detection["enable_gpu_background"],
            "ENABLE_TENSORRT": runtime_detection["enable_tensorrt"],
            "ENABLE_ONNX_RUNTIME": runtime_detection["enable_onnx_runtime"],
            "TENSORRT_MAX_BATCH_SIZE": trt_batch_size,
            "TENSORRT_BUILD_WORKSPACE_GB": float(
                advanced_config.get("tensorrt_build_workspace_gb", 4.0)
            ),
            "TENSORRT_BUILD_BATCH_SIZE": trt_build_batch_size,
            "MAX_TARGETS": N,
            "THRESHOLD_VALUE": self._detection_panel.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self._detection_panel.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self._detection_panel.spin_min_contour.value(),
            "ENABLE_SIZE_FILTERING": self._detection_panel.chk_size_filtering.isChecked(),
            "MIN_OBJECT_SIZE": min_object_size_pixels,
            "MAX_OBJECT_SIZE": max_object_size_pixels,
            "MAX_CONTOUR_MULTIPLIER": self._detection_panel.spin_max_contour_multiplier.value(),
            "MAX_DISTANCE_THRESHOLD": max_distance_pixels,
            "MAX_DISTANCE_MULTIPLIER": self._tracking_panel.spin_max_dist.value(),
            "ENABLE_POSTPROCESSING": self._postprocess_panel.enable_postprocessing.isChecked(),
            "MIN_TRAJECTORY_LENGTH": min_trajectory_length,
            "MAX_VELOCITY_BREAK": max_velocity_break_pixels_per_frame,
            "MAX_OCCLUSION_GAP": max_occlusion_gap,
            "ENABLE_TRACKLET_RELINKING": self._postprocess_panel.chk_enable_tracklet_relinking.isChecked(),
            "RELINK_POSE_MAX_DISTANCE": self._postprocess_panel.spin_relink_pose_max_distance.value(),
            "POSE_EXPORT_MIN_VALID_FRACTION": self._postprocess_panel.spin_pose_export_min_valid_fraction.value(),
            "POSE_EXPORT_MIN_VALID_KEYPOINTS": self._postprocess_panel.spin_pose_export_min_valid_keypoints.value(),
            "RELINK_MIN_POSE_QUALITY": self._postprocess_panel.spin_relink_min_pose_quality.value(),
            "POSE_POSTPROC_MAX_GAP": self._postprocess_panel.spin_pose_postproc_max_gap.value(),
            "POSE_TEMPORAL_OUTLIER_ZSCORE": self._postprocess_panel.spin_pose_temporal_outlier_zscore.value(),
            "MAX_VELOCITY_ZSCORE": self._postprocess_panel.spin_max_velocity_zscore.value(),
            "VELOCITY_ZSCORE_WINDOW": velocity_zscore_window,
            "VELOCITY_ZSCORE_MIN_VELOCITY": self._postprocess_panel.spin_velocity_zscore_min_vel.value()
            * scaled_body_size
            / fps,
            "CONTINUITY_THRESHOLD": recovery_search_distance_pixels,
            "MIN_RESPAWN_DISTANCE": min_respawn_distance_pixels,
            "MIN_DETECTION_COUNTS": min_detection_counts,
            "MIN_DETECTIONS_TO_START": min_detections_to_start,
            "MIN_TRACKING_COUNTS": min_tracking_counts,
            "TRAJECTORY_HISTORY_SECONDS": self._setup_panel.spin_traj_hist.value(),
            "BACKGROUND_PRIME_FRAMES": bg_prime_frames,
            "ENABLE_LIGHTING_STABILIZATION": self._detection_panel.chk_lighting_stab.isChecked(),
            "ENABLE_ADAPTIVE_BACKGROUND": self._detection_panel.chk_adaptive_bg.isChecked(),
            "BACKGROUND_LEARNING_RATE": self._detection_panel.spin_bg_learning.value(),
            "LIGHTING_SMOOTH_FACTOR": self._detection_panel.spin_lighting_smooth.value(),
            "LIGHTING_MEDIAN_WINDOW": self._detection_panel.spin_lighting_median.value(),
            "KALMAN_NOISE_COVARIANCE": self._tracking_panel.spin_kalman_noise.value(),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": self._tracking_panel.spin_kalman_meas.value(),
            "KALMAN_DAMPING": self._tracking_panel.spin_kalman_damping.value(),
            "KALMAN_MATURITY_AGE": kalman_maturity_age,
            "KALMAN_INITIAL_VELOCITY_RETENTION": self._tracking_panel.spin_kalman_initial_velocity_retention.value(),
            "KALMAN_MAX_VELOCITY_MULTIPLIER": self._tracking_panel.spin_kalman_max_velocity.value(),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": self._tracking_panel.spin_kalman_longitudinal_noise.value(),
            "KALMAN_LATERAL_NOISE_MULTIPLIER": self._tracking_panel.spin_kalman_lateral_noise.value(),
            # Derived anisotropy ratio for the autotune domain banner.
            # Lateral = Longitudinal / ratio, so ratio = long / lat (clamped ≥ 1).
            "KALMAN_ANISOTROPY_RATIO": max(
                1.0,
                self._tracking_panel.spin_kalman_longitudinal_noise.value()
                / max(self._tracking_panel.spin_kalman_lateral_noise.value(), 1e-6),
            ),
            "RESIZE_FACTOR": self._setup_panel.spin_resize.value(),
            "ENABLE_CONSERVATIVE_SPLIT": self._detection_panel.chk_conservative_split.isChecked(),
            "CONSERVATIVE_KERNEL_SIZE": self._detection_panel.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self._detection_panel.spin_conservative_erode.value(),
            "ENABLE_ADDITIONAL_DILATION": self._detection_panel.chk_additional_dilation.isChecked(),
            "DILATION_ITERATIONS": self._detection_panel.spin_dilation_iterations.value(),
            "DILATION_KERNEL_SIZE": self._detection_panel.spin_dilation_kernel_size.value(),
            "BRIGHTNESS": self._detection_panel.slider_brightness.value(),
            "CONTRAST": self._detection_panel.slider_contrast.value() / 100.0,
            "GAMMA": self._detection_panel.slider_gamma.value() / 100.0,
            "DARK_ON_LIGHT_BACKGROUND": self._detection_panel.chk_dark_on_light.isChecked(),
            "VELOCITY_THRESHOLD": velocity_threshold_pixels_per_frame,
            "INSTANT_FLIP_ORIENTATION": self._tracking_panel.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": self._tracking_panel.spin_max_orient.value(),
            "DIRECTED_ORIENT_SMOOTHING": self._tracking_panel.chk_directed_orient_smoothing.isChecked(),
            "DIRECTED_ORIENT_FLIP_CONFIDENCE": self._tracking_panel.spin_directed_orient_flip_conf.value(),
            "DIRECTED_ORIENT_FLIP_PERSISTENCE": self._tracking_panel.spin_directed_orient_flip_persist.value(),
            "LOST_THRESHOLD_FRAMES": lost_threshold_frames,
            "W_POSITION": self._tracking_panel.spin_Wp.value(),
            "W_ORIENTATION": self._tracking_panel.spin_Wo.value(),
            "W_AREA": self._tracking_panel.spin_Wa.value(),
            "W_ASPECT": self._tracking_panel.spin_Wasp.value(),
            "W_POSE_DIRECTION": 0.5,
            "W_POSE_LENGTH": 0.0,
            "POSE_VALID_ORIENTATION_SCALE": 0.15,
            "USE_MAHALANOBIS": self._tracking_panel.chk_use_mahal.isChecked(),
            "ENABLE_GREEDY_ASSIGNMENT": self._tracking_panel.combo_assignment_method.currentIndex()
            == 1,
            "ENABLE_SPATIAL_OPTIMIZATION": self._tracking_panel.chk_spatial_optimization.isChecked(),
            "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER": self._tracking_panel.spin_assoc_gate_multiplier.value(),
            "ASSOCIATION_STAGE1_MAX_AREA_RATIO": self._tracking_panel.spin_assoc_max_area_ratio.value(),
            "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF": self._tracking_panel.spin_assoc_max_aspect_diff.value(),
            "ENABLE_POSE_REJECTION": self._tracking_panel.chk_enable_pose_rejection.isChecked(),
            "POSE_REJECTION_THRESHOLD": self._tracking_panel.spin_pose_rejection_threshold.value(),
            "POSE_REJECTION_MIN_VISIBILITY": self._tracking_panel.spin_pose_rejection_min_visibility.value(),
            "TRACK_FEATURE_EMA_ALPHA": self._tracking_panel.spin_track_feature_ema_alpha.value(),
            "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD": self._tracking_panel.spin_assoc_high_conf_threshold.value(),
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self._detection_panel.chk_show_fg.isChecked(),
            "SHOW_BG": self._detection_panel.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self._setup_panel.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self._setup_panel.chk_show_orientation.isChecked(),
            "SHOW_YOLO_OBB": self._detection_panel.chk_show_yolo_obb.isChecked(),
            "SHOW_TRAJECTORIES": self._setup_panel.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self._setup_panel.chk_show_labels.isChecked(),
            "SHOW_STATE": self._setup_panel.chk_show_state.isChecked(),
            "SHOW_KALMAN_UNCERTAINTY": self._setup_panel.chk_show_kalman_uncertainty.isChecked(),
            "VISUALIZATION_FREE_MODE": self._setup_panel.chk_visualization_free.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            "ROI_MASK": self.roi_mask,
            "REFERENCE_BODY_SIZE": reference_body_size,
            # Conservative trajectory merging parameters (in resized coordinate space)
            # These are used in resolve_trajectories() for bidirectional merging
            # AGREEMENT_DISTANCE: max distance for frames to be considered "agreeing"
            # MIN_OVERLAP_FRAMES: minimum agreeing frames to consider merge candidates
            "AGREEMENT_DISTANCE": self._postprocess_panel.spin_merge_overlap_multiplier.value()
            * scaled_body_size,
            "MIN_OVERLAP_FRAMES": self._postprocess_panel.spin_min_overlap_frames.value(),
            # Dataset generation parameters
            "ENABLE_DATASET_GENERATION": self._dataset_panel.chk_enable_dataset_gen.isChecked(),
            "DATASET_NAME": "",
            "DATASET_CLASS_NAME": self._dataset_panel.line_dataset_class_name.text(),
            "DATASET_OUTPUT_DIR": (
                os.path.join(
                    os.path.dirname(self.current_video_path),
                    f"{os.path.splitext(os.path.basename(self.current_video_path))[0]}_datasets",
                    "active_learning",
                )
                if self.current_video_path
                else ""
            ),
            "DATASET_MAX_FRAMES": self._dataset_panel.spin_dataset_max_frames.value(),
            "DATASET_CONF_THRESHOLD": self._dataset_panel.spin_dataset_conf_threshold.value(),
            # Dataset-specific YOLO parameters from advanced config (for export, not tracking)
            "DATASET_YOLO_CONFIDENCE_THRESHOLD": self.advanced_config.get(
                "dataset_yolo_confidence_threshold", 0.05
            ),
            "DATASET_YOLO_IOU_THRESHOLD": self.advanced_config.get(
                "dataset_yolo_iou_threshold", 0.5
            ),
            "DATASET_DIVERSITY_WINDOW": self._dataset_panel.spin_dataset_diversity_window.value(),
            "DATASET_INCLUDE_CONTEXT": self._dataset_panel.chk_dataset_include_context.isChecked(),
            "DATASET_PROBABILISTIC_SAMPLING": self._dataset_panel.chk_dataset_probabilistic.isChecked(),
            "METRIC_LOW_CONFIDENCE": self._dataset_panel.chk_metric_low_confidence.isChecked(),
            "METRIC_COUNT_MISMATCH": self._dataset_panel.chk_metric_count_mismatch.isChecked(),
            "METRIC_HIGH_ASSIGNMENT_COST": self._dataset_panel.chk_metric_high_assignment_cost.isChecked(),
            "METRIC_TRACK_LOSS": self._dataset_panel.chk_metric_track_loss.isChecked(),
            "METRIC_HIGH_UNCERTAINTY": self._dataset_panel.chk_metric_high_uncertainty.isChecked(),
            # Individual analysis parameters
            "ENABLE_IDENTITY_ANALYSIS": individual_pipeline_enabled,
            "ENABLE_INDIVIDUAL_PIPELINE": individual_pipeline_enabled,
            "IDENTITY_METHOD": identity_method,
            "USE_APRILTAGS": identity_cfg.get("use_apriltags", False),
            "CNN_CLASSIFIERS": identity_cfg.get("cnn_classifiers", []),
            "COLOR_TAG_MODEL_PATH": self._identity_panel.line_color_tag_model.text(),
            "COLOR_TAG_CONFIDENCE": self._identity_panel.spin_color_tag_conf.value(),
            "CNN_CLASSIFIER_MODEL_PATH": "",
            "CNN_CLASSIFIER_CONFIDENCE": 0.5,
            "CNN_CLASSIFIER_LABEL": "",
            "CNN_CLASSIFIER_BATCH_SIZE": 64,
            "IDENTITY_MATCH_BONUS": self._identity_panel.spin_identity_match_bonus.value(),
            "IDENTITY_MISMATCH_PENALTY": self._identity_panel.spin_identity_mismatch_penalty.value(),
            "CNN_CLASSIFIER_MATCH_BONUS": self._identity_panel.spin_identity_match_bonus.value(),
            "CNN_CLASSIFIER_MISMATCH_PENALTY": self._identity_panel.spin_identity_mismatch_penalty.value(),
            "CNN_CLASSIFIER_WINDOW": 10,
            "APRILTAG_FAMILY": self._identity_panel.combo_apriltag_family.currentText(),
            "APRILTAG_DECIMATE": self._identity_panel.spin_apriltag_decimate.value(),
            "TAG_MATCH_BONUS": self._identity_panel.spin_identity_match_bonus.value(),
            "TAG_MISMATCH_PENALTY": self._identity_panel.spin_identity_mismatch_penalty.value(),
            "ENABLE_POSE_EXTRACTOR": pose_extractor_enabled,
            "POSE_MODEL_TYPE": self._identity_panel.combo_pose_model_type.currentText()
            .strip()
            .lower(),
            "POSE_MODEL_DIR": resolve_pose_model_path(
                self._pose_model_path_for_backend(
                    self._identity_panel.combo_pose_model_type.currentText()
                    .strip()
                    .lower()
                ),
                backend=self._identity_panel.combo_pose_model_type.currentText()
                .strip()
                .lower(),
            ),
            "POSE_RUNTIME_FLAVOR": runtime_pose["pose_runtime_flavor"],
            "POSE_EXPORTED_MODEL_PATH": "",
            "POSE_MIN_KPT_CONF_VALID": self._identity_panel.spin_pose_min_kpt_conf_valid.value(),
            "POSE_SKELETON_FILE": self._identity_panel.line_pose_skeleton_file.text().strip(),
            "POSE_IGNORE_KEYPOINTS": self._parse_pose_ignore_keypoints(),
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": self._parse_pose_direction_anterior_keypoints(),
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": self._parse_pose_direction_posterior_keypoints(),
            "POSE_YOLO_BATCH": self._identity_panel.spin_pose_batch.value(),
            "POSE_BATCH_SIZE": self._identity_panel.spin_pose_batch.value(),
            "POSE_SLEAP_ENV": self._selected_pose_sleap_env(),
            "POSE_SLEAP_DEVICE": runtime_pose["pose_sleap_device"],
            "POSE_SLEAP_BATCH": self._identity_panel.spin_pose_batch.value(),
            "POSE_SLEAP_MAX_INSTANCES": 1,
            "POSE_SLEAP_EXPERIMENTAL_FEATURES": self._sleap_experimental_features_enabled(),
            "INDIVIDUAL_PROPERTIES_CACHE_PATH": str(
                self.current_individual_properties_cache_path or ""
            ).strip(),
            # Real-time Individual Dataset Generation parameters
            "ENABLE_INDIVIDUAL_DATASET": individual_image_save_enabled,
            "ENABLE_INDIVIDUAL_IMAGE_SAVE": individual_image_save_enabled,
            "GENERATE_ORIENTED_TRACK_VIDEOS": self._should_generate_oriented_track_videos(),
            "INDIVIDUAL_DATASET_NAME": (
                ""
                if str(self._get_selected_yolo_headtail_model_path() or "").strip()
                else "unoriented"
            ),
            "INDIVIDUAL_DATASET_OUTPUT_DIR": (
                os.path.join(
                    os.path.dirname(self.current_video_path),
                    f"{os.path.splitext(os.path.basename(self.current_video_path))[0]}_datasets",
                    "individual_crops",
                )
                if self.current_video_path
                else ""
            ),
            "INDIVIDUAL_OUTPUT_FORMAT": self._dataset_panel.combo_individual_format.currentText().lower(),
            "INDIVIDUAL_SAVE_INTERVAL": self._dataset_panel.spin_individual_interval.value(),
            "INDIVIDUAL_INTERPOLATE_OCCLUSIONS": self._identity_panel.chk_individual_interpolate.isChecked(),
            "INDIVIDUAL_CROP_PADDING": self._identity_panel.spin_individual_padding.value(),
            "INDIVIDUAL_BACKGROUND_COLOR": [
                int(c) for c in self._identity_panel._background_color
            ],  # Ensure JSON serializable
            "SUPPRESS_FOREIGN_OBB_REGIONS": self._identity_panel.chk_suppress_foreign_obb.isChecked(),
            "SUPPRESS_FOREIGN_OBB_DATASET": self._dataset_panel.chk_suppress_foreign_obb_dataset.isChecked(),
            "INDIVIDUAL_DATASET_RUN_ID": self._individual_dataset_run_id,
            "ENABLE_CONFIDENCE_DENSITY_MAP": self._tracking_panel.chk_enable_confidence_density_map.isChecked(),
            "DENSITY_GAUSSIAN_SIGMA_SCALE": self._tracking_panel.spin_density_gaussian_sigma_scale.value(),
            "DENSITY_TEMPORAL_SIGMA": self._tracking_panel.spin_density_temporal_sigma.value(),
            "DENSITY_BINARIZE_THRESHOLD": self._tracking_panel.spin_density_binarize_threshold.value(),
            "DENSITY_CONSERVATIVE_FACTOR": self._tracking_panel.spin_density_conservative_factor.value(),
            "DENSITY_MIN_FRAME_DURATION": self._tracking_panel.spin_density_min_duration.value(),
            "DENSITY_MIN_AREA_BODIES": self._tracking_panel.spin_density_min_area_bodies.value(),
            "DENSITY_DOWNSAMPLE_FACTOR": self._tracking_panel.spin_density_downsample_factor.value(),
            "ENABLE_PROFILING": self._setup_panel.chk_enable_profiling.isChecked(),
        }

        # Backward compat: map old color_tag keys to new cnn_classifier keys
        if not p.get("CNN_CLASSIFIER_MODEL_PATH"):
            p["CNN_CLASSIFIER_MODEL_PATH"] = p.get("COLOR_TAG_MODEL_PATH", "")

        return p

    def load_config(self: object) -> object:
        """Manually load config from file dialog."""
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if config_path:
            self._load_config_from_file(config_path)
            self._show_workspace()
            self._setup_panel.config_status_label.setText(
                f"✓ Loaded: {os.path.basename(config_path)}"
            )
            self._setup_panel.config_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Configuration loaded from {config_path}")

    def _load_config_from_file(self, config_path, preset_mode=False):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.

        Args:
            config_path: Path to the config/preset file
            preset_mode: If True, skip loading video paths and ROI data (for organism presets)
        """
        if not os.path.isfile(config_path):
            return

        def get_cfg(
            new_key: object, *legacy_keys: object, default: object = None
        ) -> object:
            """Helper to get config value with fallback to legacy keys."""
            if new_key in cfg:
                return cfg[new_key]
            for key in legacy_keys:
                if key in cfg:
                    return cfg[key]
            return default

        def get_cfg_time(
            seconds_key: str,
            *frame_keys: str,
            default_seconds: float,
        ) -> float:
            """Load a time parameter, with backward-compat from frame-based configs.

            Tries ``seconds_key`` first (new-style, value in seconds).
            Falls back to legacy ``frame_keys`` (old-style, value in frames),
            converting ``frames / config_fps`` to seconds.
            """
            val = get_cfg(seconds_key, default=None)
            if val is not None:
                return float(val)
            # Try legacy frame-based keys and convert
            config_fps = float(get_cfg("fps", default=30.0))
            for fk in frame_keys:
                if fk in cfg:
                    return float(cfg[fk]) / config_fps
            return default_seconds

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # === FILE MANAGEMENT ===
            # Skip file paths when loading presets (only load for full configs)
            if not preset_mode:
                # Only set paths if they're currently empty (preserve existing paths)
                if not self._setup_panel.file_line.text().strip():
                    video_path = get_cfg("file_path", default="")
                    if video_path:
                        # Use the same setup logic as browsing for a file
                        self._setup_video_file(video_path, skip_config_load=True)
                if not self._setup_panel.csv_line.text().strip():
                    self._setup_panel.csv_line.setText(get_cfg("csv_path", default=""))
                self._postprocess_panel.check_video_output.setChecked(
                    get_cfg("video_output_enabled", default=False)
                )
                saved_video_path = get_cfg("video_output_path", default="")
                if (
                    saved_video_path
                    and not self._postprocess_panel.video_out_line.text().strip()
                ):
                    self._postprocess_panel.video_out_line.setText(saved_video_path)

            # === REFERENCE PARAMETERS ===
            # Only load video-specific reference parameters from configs (not presets)
            if not preset_mode:
                # Load FPS if saved in config
                saved_fps = get_cfg("fps", default=None)
                if saved_fps is not None:
                    self._setup_panel.spin_fps.setValue(saved_fps)

                # Load reference body size if saved in config
                saved_body_size = get_cfg("reference_body_size", default=None)
                if saved_body_size is not None:
                    self._detection_panel.spin_reference_body_size.setValue(
                        saved_body_size
                    )

                # Load frame range if saved in config
                saved_start_frame = get_cfg("start_frame", default=None)
                if (
                    saved_start_frame is not None
                    and self._setup_panel.spin_start_frame.isEnabled()
                ):
                    self._setup_panel.spin_start_frame.setValue(saved_start_frame)

                saved_end_frame = get_cfg("end_frame", default=None)
                if (
                    saved_end_frame is not None
                    and self._setup_panel.spin_end_frame.isEnabled()
                ):
                    self._setup_panel.spin_end_frame.setValue(saved_end_frame)

            # === SYSTEM PERFORMANCE ===
            self._setup_panel.spin_resize.setValue(
                get_cfg("resize_factor", default=1.0)
            )
            self._setup_panel.check_save_confidence.setChecked(
                get_cfg("save_confidence_metrics", default=True)
            )
            self._setup_panel.chk_use_cached_detections.setChecked(
                get_cfg("use_cached_detections", default=True)
            )
            self._setup_panel.chk_visualization_free.setChecked(
                get_cfg("visualization_free_mode", default=False)
            )

            # === DETECTION STRATEGY ===
            det_method = get_cfg("detection_method", default="background_subtraction")
            self._detection_panel.combo_detection_method.setCurrentIndex(
                0 if det_method == "background_subtraction" else 1
            )

            # === SIZE FILTERING ===
            self._detection_panel.chk_size_filtering.setChecked(
                get_cfg("enable_size_filtering", default=False)
            )
            self._detection_panel.spin_min_object_size.setValue(
                get_cfg("min_object_size_multiplier", default=0.3)
            )
            self._detection_panel.spin_max_object_size.setValue(
                get_cfg("max_object_size_multiplier", default=3.0)
            )

            # === IMAGE ENHANCEMENT ===
            self._detection_panel.slider_brightness.setValue(
                int(get_cfg("brightness", default=0.0))
            )
            self._detection_panel.slider_contrast.setValue(
                int(get_cfg("contrast", default=1.0) * 100)
            )
            self._detection_panel.slider_gamma.setValue(
                int(get_cfg("gamma", default=1.0) * 100)
            )
            self._detection_panel.chk_dark_on_light.setChecked(
                get_cfg("dark_on_light_background", default=True)
            )

            # === BACKGROUND SUBTRACTION ===
            self._detection_panel.spin_bg_prime.setValue(
                get_cfg_time(
                    "background_prime_seconds",
                    "background_prime_frames",
                    "bg_prime_frames",
                    default_seconds=0.33,
                )
            )
            self._detection_panel.chk_adaptive_bg.setChecked(
                get_cfg(
                    "enable_adaptive_background", "adaptive_background", default=True
                )
            )
            self._detection_panel.spin_bg_learning.setValue(
                get_cfg("background_learning_rate", default=0.001)
            )
            self._detection_panel.spin_threshold.setValue(
                get_cfg("subtraction_threshold", "threshold_value", default=50)
            )

            # === LIGHTING STABILIZATION ===
            self._detection_panel.chk_lighting_stab.setChecked(
                get_cfg(
                    "enable_lighting_stabilization",
                    "lighting_stabilization",
                    default=True,
                )
            )
            self._detection_panel.spin_lighting_smooth.setValue(
                get_cfg("lighting_smooth_factor", default=0.95)
            )
            self._detection_panel.spin_lighting_median.setValue(
                get_cfg("lighting_median_window", default=5)
            )

            # === MORPHOLOGY & NOISE ===
            self._detection_panel.spin_morph_size.setValue(
                get_cfg("morph_kernel_size", default=5)
            )
            self._detection_panel.spin_min_contour.setValue(
                get_cfg("min_contour_area", default=50)
            )
            self._detection_panel.spin_max_contour_multiplier.setValue(
                get_cfg("max_contour_multiplier", default=20)
            )

            # === ADVANCED SEPARATION ===
            self._detection_panel.chk_conservative_split.setChecked(
                get_cfg("enable_conservative_split", default=True)
            )
            self._detection_panel.spin_conservative_kernel.setValue(
                get_cfg("conservative_kernel_size", default=3)
            )
            self._detection_panel.spin_conservative_erode.setValue(
                get_cfg(
                    "conservative_erode_iterations",
                    "conservative_erode_iter",
                    default=1,
                )
            )
            self._detection_panel.chk_additional_dilation.setChecked(
                get_cfg("enable_additional_dilation", default=False)
            )
            self._detection_panel.spin_dilation_kernel_size.setValue(
                get_cfg("dilation_kernel_size", default=3)
            )
            self._detection_panel.spin_dilation_iterations.setValue(
                get_cfg("dilation_iterations", default=2)
            )

            # === YOLO CONFIGURATION ===
            yolo_mode = str(get_cfg("yolo_obb_mode", default="direct")).strip().lower()
            if yolo_mode not in {"direct", "sequential"}:
                yolo_mode = "direct"
            self._detection_panel.combo_yolo_obb_mode.setCurrentIndex(
                1 if yolo_mode == "sequential" else 0
            )

            yolo_direct_model = get_cfg(
                "yolo_obb_direct_model_path",
                "yolo_model_path",
                default="",
            )
            yolo_detect_model = get_cfg("yolo_detect_model_path", default="")
            yolo_crop_obb_model = get_cfg(
                "yolo_crop_obb_model_path",
                default=yolo_direct_model,
            )
            yolo_headtail_model = get_cfg("yolo_headtail_model_path", default="")
            yolo_headtail_model_type = str(
                get_cfg(
                    "yolo_headtail_model_type",
                    default=self._infer_yolo_headtail_model_type(yolo_headtail_model),
                )
            ).strip()

            resolved_yolo_direct = resolve_model_path(yolo_direct_model)
            resolved_yolo_detect = resolve_model_path(yolo_detect_model)
            resolved_yolo_crop_obb = resolve_model_path(yolo_crop_obb_model)
            resolved_yolo_headtail = resolve_model_path(yolo_headtail_model)

            if preset_mode:
                for model_key, model_cfg, model_resolved in (
                    ("Direct OBB model", yolo_direct_model, resolved_yolo_direct),
                    (
                        "Sequential detect model",
                        yolo_detect_model,
                        resolved_yolo_detect,
                    ),
                    (
                        "Sequential crop OBB model",
                        yolo_crop_obb_model,
                        resolved_yolo_crop_obb,
                    ),
                    ("Head-tail model", yolo_headtail_model, resolved_yolo_headtail),
                ):
                    if not model_cfg:
                        continue
                    if os.path.isabs(str(model_cfg)):
                        continue
                    if os.path.exists(model_resolved):
                        continue
                    logger.warning(
                        "Preset references non-existent %s: %s (expected: %s)",
                        model_key,
                        model_cfg,
                        model_resolved,
                    )

            self._refresh_yolo_model_combo(preferred_model_path=yolo_direct_model)
            self._set_yolo_model_selection(resolved_yolo_direct)
            self._refresh_yolo_detect_model_combo(
                preferred_model_path=yolo_detect_model
            )
            self._set_yolo_detect_model_selection(resolved_yolo_detect)
            self._refresh_yolo_crop_obb_model_combo(
                preferred_model_path=yolo_crop_obb_model
            )
            self._set_yolo_crop_obb_model_selection(resolved_yolo_crop_obb)
            headtail_type_idx = (
                self._identity_panel.combo_yolo_headtail_model_type.findText(
                    "tiny" if yolo_headtail_model_type.lower() == "tiny" else "YOLO"
                )
            )
            if headtail_type_idx >= 0:
                self._identity_panel.combo_yolo_headtail_model_type.setCurrentIndex(
                    headtail_type_idx
                )
            self._refresh_yolo_headtail_model_combo(
                preferred_model_path=yolo_headtail_model
            )
            self._set_yolo_headtail_model_selection(resolved_yolo_headtail)
            self._identity_panel.chk_pose_overrides_headtail.setChecked(
                bool(get_cfg("pose_overrides_headtail", default=True))
            )
            self._detection_panel.spin_yolo_seq_crop_pad.setValue(
                float(get_cfg("yolo_seq_crop_pad_ratio", default=0.15))
            )
            self._detection_panel.spin_yolo_seq_min_crop_px.setValue(
                int(get_cfg("yolo_seq_min_crop_size_px", default=64))
            )
            self._detection_panel.chk_yolo_seq_square_crop.setChecked(
                bool(get_cfg("yolo_seq_enforce_square_crop", default=True))
            )
            self._detection_panel.spin_yolo_seq_stage2_imgsz.setValue(
                int(get_cfg("yolo_seq_stage2_imgsz", default=160))
            )
            self._detection_panel.chk_yolo_seq_stage2_pow2_pad.setChecked(
                bool(get_cfg("yolo_seq_stage2_pow2_pad", default=False))
            )
            self._detection_panel.spin_yolo_seq_detect_conf.setValue(
                float(get_cfg("yolo_seq_detect_conf_threshold", default=0.25))
            )
            self._identity_panel.spin_yolo_headtail_conf.setValue(
                float(get_cfg("yolo_headtail_conf_threshold", default=0.50))
            )
            self._detection_panel.spin_reference_aspect_ratio.setValue(
                float(get_cfg("reference_aspect_ratio", default=2.0))
            )
            self._detection_panel.chk_enable_aspect_ratio_filtering.setChecked(
                bool(get_cfg("enable_aspect_ratio_filtering", default=False))
            )
            self._detection_panel.spin_min_ar_multiplier.setValue(
                float(get_cfg("min_aspect_ratio_multiplier", default=0.5))
            )
            self._detection_panel.spin_max_ar_multiplier.setValue(
                float(get_cfg("max_aspect_ratio_multiplier", default=2.0))
            )
            self._on_yolo_mode_changed(
                self._detection_panel.combo_yolo_obb_mode.currentIndex()
            )

            self._detection_panel.spin_yolo_confidence.setValue(
                get_cfg("yolo_confidence_threshold", default=0.25)
            )
            self._detection_panel.spin_yolo_iou.setValue(
                get_cfg("yolo_iou_threshold", default=0.7)
            )
            self._detection_panel.chk_use_custom_obb_iou.setChecked(True)
            yolo_cls = get_cfg("yolo_target_classes", default=None)
            if yolo_cls:
                self._detection_panel.line_yolo_classes.setText(
                    ",".join(map(str, yolo_cls))
                )
            else:
                self._detection_panel.line_yolo_classes.clear()

            compute_runtime_cfg = (
                str(
                    get_cfg(
                        "compute_runtime",
                        default=infer_compute_runtime_from_legacy(
                            yolo_device=str(get_cfg("yolo_device", default="auto")),
                            enable_tensorrt=bool(
                                get_cfg("enable_tensorrt", default=False)
                            ),
                            pose_runtime_flavor=str(
                                get_cfg("pose_runtime_flavor", default="auto")
                            ),
                        ),
                    )
                )
                .strip()
                .lower()
            )
            self._populate_compute_runtime_options(preferred=compute_runtime_cfg)
            self._on_runtime_context_changed()

            # TensorRT batch size is still configurable (runtime-derived usage).
            self._detection_panel.spin_tensorrt_batch.setValue(
                get_cfg("tensorrt_max_batch_size", default=16)
            )
            self._detection_panel.spin_tensorrt_batch.setEnabled(
                bool(
                    derive_detection_runtime_settings(self._selected_compute_runtime())[
                        "enable_tensorrt"
                    ]
                )
            )
            self._detection_panel.lbl_tensorrt_batch.setEnabled(
                bool(
                    derive_detection_runtime_settings(self._selected_compute_runtime())[
                        "enable_tensorrt"
                    ]
                )
            )

            # YOLO Batching settings
            self._detection_panel.chk_enable_yolo_batching.setChecked(
                get_cfg("enable_yolo_batching", default=True)
            )
            batch_mode = get_cfg("yolo_batch_size_mode", default="auto")
            self._detection_panel.combo_yolo_batch_mode.setCurrentIndex(
                0 if batch_mode == "auto" else 1
            )
            self._detection_panel.spin_yolo_batch_size.setValue(
                get_cfg("yolo_manual_batch_size", default=16)
            )
            # Re-apply runtime-derived constraints (e.g., TensorRT => manual batch mode).
            self._on_runtime_context_changed()

            # === CORE TRACKING ===
            self._setup_panel.spin_max_targets.setValue(
                get_cfg("max_targets", default=4)
            )
            self._tracking_panel.spin_max_dist.setValue(
                get_cfg(
                    "max_assignment_distance_multiplier",
                    "max_dist_multiplier",
                    default=1.5,
                )
            )
            self._tracking_panel.spin_continuity_thresh.setValue(
                get_cfg(
                    "recovery_search_distance_multiplier",
                    "continuity_threshold_multiplier",
                    default=0.5,
                )
            )
            self._tracking_panel.chk_enable_backward.setChecked(
                get_cfg("enable_backward_tracking", default=True)
            )

            # === KALMAN FILTER ===
            self._tracking_panel.spin_kalman_noise.setValue(
                get_cfg("kalman_process_noise", "kalman_noise", default=0.03)
            )
            self._tracking_panel.spin_kalman_meas.setValue(
                get_cfg("kalman_measurement_noise", "kalman_meas_noise", default=0.1)
            )
            self._tracking_panel.spin_kalman_damping.setValue(
                get_cfg("kalman_velocity_damping", "kalman_damping", default=0.95)
            )
            self._tracking_panel.spin_kalman_maturity_age.setValue(
                get_cfg_time(
                    "kalman_maturity_age_seconds",
                    "kalman_maturity_age",
                    default_seconds=0.17,
                )
            )
            self._tracking_panel.spin_kalman_initial_velocity_retention.setValue(
                get_cfg("kalman_initial_velocity_retention", default=0.2)
            )
            self._tracking_panel.spin_kalman_max_velocity.setValue(
                get_cfg("kalman_max_velocity_multiplier", default=2.0)
            )
            self._tracking_panel.spin_kalman_longitudinal_noise.setValue(
                get_cfg("kalman_longitudinal_noise_multiplier", default=5.0)
            )
            self._tracking_panel.spin_kalman_lateral_noise.setValue(
                get_cfg("kalman_lateral_noise_multiplier", default=0.1)
            )

            # === COST FUNCTION WEIGHTS ===
            self._tracking_panel.spin_Wp.setValue(
                get_cfg("weight_position", "W_POSITION", default=1.0)
            )
            self._tracking_panel.spin_Wo.setValue(
                get_cfg("weight_orientation", "W_ORIENTATION", default=1.0)
            )
            self._tracking_panel.spin_Wa.setValue(
                get_cfg("weight_area", "W_AREA", default=0.001)
            )
            self._tracking_panel.spin_Wasp.setValue(
                get_cfg("weight_aspect_ratio", "W_ASPECT", default=0.1)
            )
            self._tracking_panel.chk_use_mahal.setChecked(
                get_cfg("use_mahalanobis_distance", "USE_MAHALANOBIS", default=True)
            )

            # === ASSIGNMENT ALGORITHM ===
            self._tracking_panel.combo_assignment_method.setCurrentIndex(
                1 if get_cfg("enable_greedy_assignment", default=False) else 0
            )
            self._tracking_panel.chk_spatial_optimization.setChecked(
                get_cfg("enable_spatial_optimization", default=False)
            )
            self._tracking_panel.spin_assoc_gate_multiplier.setValue(
                get_cfg(
                    "association_stage1_motion_gate_multiplier",
                    "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER",
                    default=1.4,
                )
            )
            self._tracking_panel.spin_assoc_max_area_ratio.setValue(
                get_cfg(
                    "association_stage1_max_area_ratio",
                    "ASSOCIATION_STAGE1_MAX_AREA_RATIO",
                    default=2.5,
                )
            )
            self._tracking_panel.spin_assoc_max_aspect_diff.setValue(
                get_cfg(
                    "association_stage1_max_aspect_diff",
                    "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF",
                    default=0.8,
                )
            )
            self._tracking_panel.chk_enable_pose_rejection.setChecked(
                get_cfg(
                    "enable_pose_rejection",
                    "ENABLE_POSE_REJECTION",
                    default=True,
                )
            )
            self._tracking_panel.spin_pose_rejection_threshold.setValue(
                get_cfg(
                    "pose_rejection_threshold",
                    "POSE_REJECTION_THRESHOLD",
                    default=0.5,
                )
            )
            self._tracking_panel.spin_pose_rejection_min_visibility.setValue(
                get_cfg(
                    "pose_rejection_min_visibility",
                    "POSE_REJECTION_MIN_VISIBILITY",
                    default=0.5,
                )
            )
            self._tracking_panel.spin_track_feature_ema_alpha.setValue(
                get_cfg(
                    "track_feature_ema_alpha",
                    "TRACK_FEATURE_EMA_ALPHA",
                    default=0.85,
                )
            )
            self._tracking_panel.spin_assoc_high_conf_threshold.setValue(
                get_cfg(
                    "association_high_confidence_threshold",
                    "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD",
                    default=0.7,
                )
            )

            # === ORIENTATION & MOTION ===
            self._tracking_panel.spin_velocity.setValue(
                get_cfg("velocity_threshold", default=5.0)
            )
            self._tracking_panel.chk_instant_flip.setChecked(
                get_cfg("enable_instant_flip", "instant_flip", default=True)
            )
            self._tracking_panel.spin_max_orient.setValue(
                get_cfg(
                    "max_orientation_delta_stopped",
                    "max_orient_delta_stopped",
                    default=30.0,
                )
            )
            self._tracking_panel.chk_directed_orient_smoothing.setChecked(
                bool(get_cfg("directed_orient_smoothing", default=True))
            )
            self._tracking_panel.spin_directed_orient_flip_conf.setValue(
                float(get_cfg("directed_orient_flip_confidence", default=0.7))
            )
            self._tracking_panel.spin_directed_orient_flip_persist.setValue(
                int(get_cfg("directed_orient_flip_persistence", default=3))
            )

            # === TRACK LIFECYCLE ===
            self._tracking_panel.spin_lost_thresh.setValue(
                get_cfg_time(
                    "lost_threshold_seconds",
                    "lost_frames_threshold",
                    "lost_threshold_frames",
                    default_seconds=0.33,
                )
            )
            self._tracking_panel.spin_min_respawn_distance.setValue(
                get_cfg("min_respawn_distance_multiplier", default=2.5)
            )
            self._tracking_panel.spin_min_detections_to_start.setValue(
                get_cfg_time(
                    "min_detections_to_start_seconds",
                    "min_detections_to_start",
                    default_seconds=0.03,
                )
            )
            self._tracking_panel.spin_min_detect.setValue(
                get_cfg_time(
                    "min_detect_seconds",
                    "min_detect_frames",
                    "min_detect_counts",
                    default_seconds=0.33,
                )
            )
            self._tracking_panel.spin_min_track.setValue(
                get_cfg_time(
                    "min_track_seconds",
                    "min_track_frames",
                    "min_track_counts",
                    default_seconds=0.33,
                )
            )

            # === POST-PROCESSING ===
            self._postprocess_panel.enable_postprocessing.setChecked(
                get_cfg("enable_postprocessing", default=True)
            )
            self._postprocess_panel.spin_min_trajectory_length.setValue(
                get_cfg_time(
                    "min_trajectory_length_seconds",
                    "min_trajectory_length",
                    default_seconds=0.33,
                )
            )
            self._postprocess_panel.spin_max_velocity_break.setValue(
                get_cfg("max_velocity_break", default=50.0)
            )
            self._postprocess_panel.spin_max_occlusion_gap.setValue(
                get_cfg_time(
                    "max_occlusion_gap_seconds",
                    "max_occlusion_gap",
                    default_seconds=1.0,
                )
            )
            self._postprocess_panel.chk_enable_tracklet_relinking.setChecked(
                get_cfg("enable_tracklet_relinking", default=False)
            )
            self._postprocess_panel.spin_relink_pose_max_distance.setValue(
                get_cfg("relink_pose_max_distance", default=0.45)
            )
            self._postprocess_panel.spin_pose_export_min_valid_fraction.setValue(
                get_cfg("pose_export_min_valid_fraction", default=0.5)
            )
            self._postprocess_panel.spin_pose_export_min_valid_keypoints.setValue(
                get_cfg("pose_export_min_valid_keypoints", default=3)
            )
            self._postprocess_panel.spin_relink_min_pose_quality.setValue(
                get_cfg("relink_min_pose_quality", default=0.6)
            )
            self._postprocess_panel.spin_pose_postproc_max_gap.setValue(
                get_cfg("pose_postproc_max_gap", default=5)
            )
            self._postprocess_panel.spin_pose_temporal_outlier_zscore.setValue(
                get_cfg("pose_temporal_outlier_zscore", default=3.0)
            )
            self._tracking_panel.chk_enable_confidence_density_map.setChecked(
                get_cfg("enable_confidence_density_map", default=True)
            )
            self._tracking_panel.spin_density_gaussian_sigma_scale.setValue(
                get_cfg("density_gaussian_sigma_scale", default=1.0)
            )
            self._tracking_panel.spin_density_temporal_sigma.setValue(
                get_cfg("density_temporal_sigma", default=2.0)
            )
            self._tracking_panel.spin_density_binarize_threshold.setValue(
                get_cfg("density_binarize_threshold", default=0.3)
            )
            self._tracking_panel.spin_density_conservative_factor.setValue(
                get_cfg("density_conservative_factor", default=0.7)
            )
            self._tracking_panel.spin_density_min_duration.setValue(
                int(get_cfg("density_min_frame_duration", default=3))
            )
            self._tracking_panel.spin_density_min_area_bodies.setValue(
                float(get_cfg("density_min_area_bodies", default=0.25))
            )
            self._tracking_panel.spin_density_downsample_factor.setValue(
                int(get_cfg("density_downsample_factor", default=8))
            )
            self._on_confidence_density_map_toggled(
                self._tracking_panel.chk_enable_confidence_density_map.checkState()
            )
            self._postprocess_panel.spin_max_velocity_zscore.setValue(
                get_cfg("max_velocity_zscore", default=0.0)
            )
            self._postprocess_panel.spin_velocity_zscore_window.setValue(
                get_cfg_time(
                    "velocity_zscore_window_seconds",
                    "velocity_zscore_window",
                    default_seconds=0.33,
                )
            )
            self._postprocess_panel.spin_velocity_zscore_min_vel.setValue(
                get_cfg("velocity_zscore_min_velocity", default=2.0)
            )
            interp_method = get_cfg("interpolation_method", default="None")
            idx = self._postprocess_panel.combo_interpolation_method.findText(
                interp_method, Qt.MatchFixedString
            )
            if idx >= 0:
                self._postprocess_panel.combo_interpolation_method.setCurrentIndex(idx)
            self._postprocess_panel.spin_interpolation_max_gap.setValue(
                get_cfg_time(
                    "interpolation_max_gap_seconds",
                    "interpolation_max_gap",
                    default_seconds=0.33,
                )
            )
            self._postprocess_panel.spin_heading_flip_max_burst.setValue(
                int(get_cfg("heading_flip_max_burst", default=5))
            )
            self._postprocess_panel.chk_cleanup_temp_files.setChecked(
                get_cfg("cleanup_temp_files", default=True)
            )

            # === TRAJECTORY MERGING (Conservative Strategy) ===
            # Agreement distance and min overlap frames for conservative merging
            self._postprocess_panel.spin_merge_overlap_multiplier.setValue(
                get_cfg("merge_agreement_distance_multiplier", default=0.5)
            )
            self._postprocess_panel.spin_min_overlap_frames.setValue(
                get_cfg("min_overlap_frames", default=5)
            )

            # === VIDEO VISUALIZATION ===
            self._postprocess_panel.check_show_labels.setChecked(
                get_cfg("video_show_labels", default=True)
            )
            self._postprocess_panel.check_show_orientation.setChecked(
                get_cfg("video_show_orientation", default=True)
            )
            self._postprocess_panel.check_show_trails.setChecked(
                get_cfg("video_show_trails", default=False)
            )
            self._postprocess_panel.spin_trail_duration.setValue(
                get_cfg("video_trail_duration", default=1.0)
            )
            self._postprocess_panel.spin_marker_size.setValue(
                get_cfg("video_marker_size", default=0.3)
            )
            self._postprocess_panel.spin_text_scale.setValue(
                get_cfg("video_text_scale", default=0.5)
            )
            self._postprocess_panel.spin_arrow_length.setValue(
                get_cfg("video_arrow_length", default=0.7)
            )
            self._postprocess_panel.check_video_show_pose.setChecked(
                get_cfg(
                    "video_show_pose",
                    default=bool(self.advanced_config.get("video_show_pose", True)),
                )
            )
            pose_color_mode = str(
                get_cfg(
                    "video_pose_color_mode",
                    default=self.advanced_config.get("video_pose_color_mode", "track"),
                )
            ).strip()
            self._postprocess_panel.combo_video_pose_color_mode.setCurrentIndex(
                0 if pose_color_mode == "track" else 1
            )
            self._postprocess_panel.spin_video_pose_point_radius.setValue(
                int(
                    get_cfg(
                        "video_pose_point_radius",
                        default=self.advanced_config.get("video_pose_point_radius", 3),
                    )
                )
            )
            self._postprocess_panel.spin_video_pose_point_thickness.setValue(
                int(
                    get_cfg(
                        "video_pose_point_thickness",
                        default=self.advanced_config.get(
                            "video_pose_point_thickness", -1
                        ),
                    )
                )
            )
            self._postprocess_panel.spin_video_pose_line_thickness.setValue(
                int(
                    get_cfg(
                        "video_pose_line_thickness",
                        default=self.advanced_config.get(
                            "video_pose_line_thickness", 2
                        ),
                    )
                )
            )
            pose_color = get_cfg(
                "video_pose_color",
                default=self.advanced_config.get("video_pose_color", [255, 255, 255]),
            )
            if isinstance(pose_color, (list, tuple)) and len(pose_color) == 3:
                self._postprocess_panel._video_pose_color = tuple(
                    int(max(0, min(255, float(v)))) for v in pose_color
                )
                self._update_video_pose_color_button()
            self._sync_video_pose_overlay_controls()

            # === VISUALIZATION OVERLAYS ===
            self._setup_panel.chk_show_circles.setChecked(
                get_cfg("show_track_markers", "show_circles", default=True)
            )
            self._setup_panel.chk_show_orientation.setChecked(
                get_cfg("show_orientation_lines", "show_orientation", default=True)
            )
            self._setup_panel.chk_show_trajectories.setChecked(
                get_cfg("show_trajectory_trails", "show_trajectories", default=True)
            )
            self._setup_panel.chk_show_labels.setChecked(
                get_cfg("show_id_labels", "show_labels", default=True)
            )
            self._setup_panel.chk_show_state.setChecked(
                get_cfg("show_state_text", "show_state", default=True)
            )
            self._setup_panel.chk_show_kalman_uncertainty.setChecked(
                get_cfg("show_kalman_uncertainty", default=False)
            )
            self._detection_panel.chk_show_fg.setChecked(
                get_cfg("show_foreground_mask", "show_fg", default=True)
            )
            self._detection_panel.chk_show_bg.setChecked(
                get_cfg("show_background_model", "show_bg", default=True)
            )
            self._detection_panel.chk_show_yolo_obb.setChecked(
                get_cfg("show_yolo_obb", default=False)
            )
            self._setup_panel.spin_traj_hist.setValue(
                get_cfg("trajectory_history_seconds", "traj_history", default=5)
            )
            self._setup_panel.chk_debug_logging.setChecked(
                get_cfg("debug_logging", default=False)
            )
            self._setup_panel.chk_enable_profiling.setChecked(
                get_cfg("enable_profiling", default=False)
            )
            self.slider_zoom.setValue(int(get_cfg("zoom_factor", default=1.0) * 100))

            # === DATASET GENERATION ===
            self._dataset_panel.chk_enable_dataset_gen.setChecked(
                get_cfg("enable_dataset_generation", default=False)
            )
            self._dataset_panel.line_dataset_class_name.setText(
                get_cfg("dataset_class_name", default="object")
            )
            self._dataset_panel.spin_dataset_max_frames.setValue(
                get_cfg("dataset_max_frames", default=100)
            )
            self._dataset_panel.spin_dataset_conf_threshold.setValue(
                get_cfg(
                    "dataset_confidence_threshold",
                    "dataset_conf_threshold",
                    default=0.5,
                )
            )
            # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
            self._dataset_panel.spin_dataset_diversity_window.setValue(
                get_cfg("dataset_diversity_window", default=30)
            )
            self._dataset_panel.chk_dataset_include_context.setChecked(
                get_cfg("dataset_include_context", default=True)
            )
            self._dataset_panel.chk_dataset_probabilistic.setChecked(
                get_cfg("dataset_probabilistic_sampling", default=True)
            )
            self._dataset_panel.chk_metric_low_confidence.setChecked(
                get_cfg("metric_low_confidence", default=True)
            )
            self._dataset_panel.chk_metric_count_mismatch.setChecked(
                get_cfg("metric_count_mismatch", default=True)
            )
            self._dataset_panel.chk_metric_high_assignment_cost.setChecked(
                get_cfg("metric_high_assignment_cost", default=True)
            )
            self._dataset_panel.chk_metric_track_loss.setChecked(
                get_cfg("metric_track_loss", default=True)
            )
            self._dataset_panel.chk_metric_high_uncertainty.setChecked(
                get_cfg("metric_high_uncertainty", default=False)
            )

            # === INDIVIDUAL ANALYSIS ===
            old_method = str(
                get_cfg("identity_method", default="none_disabled")
            ).lower()
            # Backward compat: rename color_tags_yolo -> cnn_classifier on load
            if old_method == "color_tags_yolo":
                old_method = "cnn_classifier"
            self._identity_panel.g_identity.setChecked(old_method != "none_disabled")

            # --- New format or migrate from old format ---
            _new_cnn_classifiers = get_cfg("cnn_classifiers", default=None)
            if _new_cnn_classifiers is not None:
                # New format: load use_apriltags + cnn_classifiers list
                self._identity_panel.g_apriltags.setChecked(
                    bool(get_cfg("use_apriltags", default=False))
                )
                for entry in _new_cnn_classifiers or []:
                    row = self._add_cnn_classifier_row()
                    row.load_from_config(entry)
            else:
                # Old single-method config: migrate
                if old_method == "apriltags":
                    self._identity_panel.g_apriltags.setChecked(True)
                elif old_method in ("cnn_classifier",):
                    cnn_model_rel = get_cfg("cnn_classifier_model_path", default="")
                    if cnn_model_rel:
                        row = self._add_cnn_classifier_row()
                        row.load_from_config(
                            {
                                "rel_path": cnn_model_rel,
                                "label": get_cfg(
                                    "cnn_classifier_label", default="identity"
                                ),
                                "confidence": float(
                                    get_cfg("cnn_classifier_confidence", default=0.5)
                                ),
                                "window": int(
                                    get_cfg("cnn_classifier_window", default=10)
                                ),
                                "match_bonus": float(
                                    get_cfg(
                                        "cnn_classifier_match_bonus",
                                        "identity_match_bonus",
                                        default=20.0,
                                    )
                                ),
                                "mismatch_penalty": float(
                                    get_cfg(
                                        "cnn_classifier_mismatch_penalty",
                                        "identity_mismatch_penalty",
                                        default=50.0,
                                    )
                                ),
                            }
                        )

            # Shared identity cost settings
            self._identity_panel.spin_identity_match_bonus.setValue(
                float(
                    get_cfg(
                        "identity_match_bonus",
                        "tag_match_bonus",
                        "cnn_classifier_match_bonus",
                        default=20.0,
                    )
                )
            )
            self._identity_panel.spin_identity_mismatch_penalty.setValue(
                float(
                    get_cfg(
                        "identity_mismatch_penalty",
                        "tag_mismatch_penalty",
                        "cnn_classifier_mismatch_penalty",
                        default=50.0,
                    )
                )
            )
            apriltag_family = get_cfg("apriltag_family", default="tag36h11")
            idx = self._identity_panel.combo_apriltag_family.findText(apriltag_family)
            self._identity_panel.combo_apriltag_family.setCurrentIndex(max(0, idx))
            self._identity_panel.spin_apriltag_decimate.setValue(
                float(get_cfg("apriltag_decimate", default=1.0))
            )

            # Warn users who had a non-default cnn_classifier_crop_padding in their config
            _legacy_crop_padding = get_cfg("cnn_classifier_crop_padding", default=None)
            if _legacy_crop_padding is not None and float(_legacy_crop_padding) != 0.1:
                logger.warning(
                    "Config key 'cnn_classifier_crop_padding' (value=%.2f) is no longer used. "
                    "All precompute phases now use 'individual_crop_padding'. "
                    "Update your crop padding setting in the Individual Analysis panel.",
                    float(_legacy_crop_padding),
                )

            self._identity_panel.chk_enable_pose_extractor.setChecked(
                get_cfg("enable_pose_extractor", default=False)
            )
            pose_backend = (
                str(get_cfg("pose_model_type", default="yolo")).strip().upper()
            )
            pose_backend_idx = self._identity_panel.combo_pose_model_type.findText(
                pose_backend
            )
            if pose_backend_idx >= 0:
                self._identity_panel.combo_pose_model_type.setCurrentIndex(
                    pose_backend_idx
                )
            yolo_pose_model = str(get_cfg("pose_yolo_model_dir", default="")).strip()
            sleap_pose_model = str(get_cfg("pose_sleap_model_dir", default="")).strip()
            legacy_pose_model = str(get_cfg("pose_model_dir", default="")).strip()
            if not yolo_pose_model and pose_backend.lower() == "yolo":
                yolo_pose_model = legacy_pose_model
            if not sleap_pose_model and pose_backend.lower() == "sleap":
                sleap_pose_model = legacy_pose_model
            self._set_pose_model_path_for_backend(yolo_pose_model, backend="yolo")
            self._set_pose_model_path_for_backend(sleap_pose_model, backend="sleap")
            active_backend = (
                self._identity_panel.combo_pose_model_type.currentText().strip().lower()
            )
            self._refresh_pose_model_combo(
                preferred_model_path=self._pose_model_path_for_backend(active_backend)
            )
            pose_runtime_flavor = derive_pose_runtime_settings(
                self._selected_compute_runtime(),
                backend_family=self._identity_panel.combo_pose_model_type.currentText()
                .strip()
                .lower(),
            )["pose_runtime_flavor"]
            self._populate_pose_runtime_flavor_options(
                backend=self._identity_panel.combo_pose_model_type.currentText()
                .strip()
                .lower(),
                preferred=pose_runtime_flavor,
            )
            self._identity_panel.spin_pose_min_kpt_conf_valid.setValue(
                get_cfg("pose_min_kpt_conf_valid", default=0.2)
            )
            self._identity_panel.line_pose_skeleton_file.setText(
                get_cfg("pose_skeleton_file", default="")
            )
            self._refresh_pose_direction_keypoint_lists()
            ignore_kpts = get_cfg("pose_ignore_keypoints", default=[])
            self._set_pose_group_selection(
                self._identity_panel.list_pose_ignore_keypoints, ignore_kpts
            )
            ant_kpts = get_cfg("pose_direction_anterior_keypoints", default=[])
            self._set_pose_group_selection(
                self._identity_panel.list_pose_direction_anterior, ant_kpts
            )
            post_kpts = get_cfg("pose_direction_posterior_keypoints", default=[])
            self._set_pose_group_selection(
                self._identity_panel.list_pose_direction_posterior, post_kpts
            )
            self._apply_pose_keypoint_selection_constraints("ignore")
            self.advanced_config["pose_sleap_env"] = str(
                get_cfg("pose_sleap_env", default="sleap")
            )
            self._refresh_pose_sleap_envs()
            if hasattr(self, "_identity_panel"):
                self._identity_panel.chk_sleap_experimental_features.setChecked(
                    get_cfg("pose_sleap_experimental_features", default=False)
                )
            shared_pose_batch = int(
                get_cfg(
                    "pose_batch_size",
                    default=get_cfg(
                        "pose_yolo_batch",
                        default=get_cfg("pose_sleap_batch", default=4),
                    ),
                )
            )
            self._identity_panel.spin_pose_batch.setValue(shared_pose_batch)

            # === REAL-TIME INDIVIDUAL DATASET ===
            self._dataset_panel.chk_suppress_foreign_obb_dataset.setChecked(
                get_cfg("suppress_foreign_obb_dataset", default=False)
            )
            self._dataset_panel.chk_enable_individual_dataset.setChecked(
                get_cfg(
                    "enable_individual_image_save",
                    "enable_individual_dataset",
                    default=False,
                )
            )
            self._dataset_panel.chk_generate_individual_track_videos.setChecked(
                get_cfg("generate_oriented_track_videos", default=False)
            )
            format_text = get_cfg("individual_output_format", default="png").upper()
            format_idx = self._dataset_panel.combo_individual_format.findText(
                format_text
            )
            if format_idx >= 0:
                self._dataset_panel.combo_individual_format.setCurrentIndex(format_idx)
            self._dataset_panel.spin_individual_interval.setValue(
                get_cfg("individual_save_interval", default=1)
            )
            self._identity_panel.chk_individual_interpolate.setChecked(
                get_cfg("individual_interpolate_occlusions", default=True)
            )
            self._identity_panel.spin_individual_padding.setValue(
                get_cfg("individual_crop_padding", default=0.1)
            )
            # Load background color
            bg_color = get_cfg("individual_background_color", default=[0, 0, 0])
            if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
                self._identity_panel._background_color = tuple(bg_color)
            self._update_background_color_button()
            self._identity_panel.chk_suppress_foreign_obb.setChecked(
                get_cfg("suppress_foreign_obb_regions", default=True)
            )
            self._sync_individual_analysis_mode_ui()

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

    @staticmethod
    def _atomic_json_write(cfg, path):
        """Write a JSON config atomically. Returns (success, error_message)."""
        import tempfile as _tempfile

        tmp_path = None
        try:
            with _tempfile.NamedTemporaryFile(
                mode="w",
                dir=os.path.dirname(path),
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(cfg, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
            return True, ""
        except Exception as e:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False, str(e)

    def _resolve_config_save_path(self, prompt_if_exists):
        """Determine the config file save path, prompting the user if needed."""
        video_path = self._setup_panel.file_line.text()
        default_path = (
            get_video_config_path(video_path) if video_path else CONFIG_FILENAME
        )

        if default_path and os.path.exists(default_path) and prompt_if_exists:
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
            msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(replace_btn)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == replace_btn:
                return default_path
            if clicked == save_as_btn:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration As", default_path, "JSON Files (*.json)"
                )
                return path or None
            return None

        if default_path:
            return default_path
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration", CONFIG_FILENAME, "JSON Files (*.json)"
        )
        return path or None

    def save_config(
        self: object,
        preset_mode: object = False,
        preset_path: object = None,
        preset_name: object = None,
        preset_description: object = None,
        prompt_if_exists: bool = True,
    ) -> object:
        """Save current configuration to JSON file.

        Args:
            preset_mode: If True, skip video paths, device settings, and ROI data (for organism presets)
            preset_path: If provided, save directly to this path without prompting
            preset_name: Name for the preset (only used in preset_mode)
            preset_description: Description for the preset (only used in preset_mode)
            prompt_if_exists: If False, overwrite default config path without interactive replace dialog.

        Returns:
            bool: True if config was saved successfully, False if cancelled or failed
        """
        yolo_mode = (
            "sequential"
            if self._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
            else "direct"
        )
        yolo_direct_path = self._get_selected_yolo_model_path()
        yolo_detect_path = self._get_selected_yolo_detect_model_path()
        yolo_crop_obb_path = self._get_selected_yolo_crop_obb_model_path()
        yolo_headtail_path = self._get_selected_yolo_headtail_model_path()
        yolo_path = yolo_direct_path if yolo_mode == "direct" else yolo_crop_obb_path
        yolo_cls = (
            [
                int(x.strip())
                for x in self._detection_panel.line_yolo_classes.text().split(",")
            ]
            if self._detection_panel.line_yolo_classes.text().strip()
            else None
        )

        cfg = {}

        # === PRESET METADATA ===
        # Add name and description when saving as preset
        if preset_mode:
            cfg.update(
                {
                    "preset_name": preset_name or "Custom",
                    "description": preset_description or "User-defined custom preset",
                }
            )

        # === FILE MANAGEMENT ===
        # Skip file paths when saving as preset
        if not preset_mode:
            cfg.update(
                {
                    "file_path": self._setup_panel.file_line.text(),
                    "csv_path": self._setup_panel.csv_line.text(),
                    "video_output_enabled": self._postprocess_panel.check_video_output.isChecked(),
                    "video_output_path": self._postprocess_panel.video_out_line.text(),
                    # Video-specific reference parameters
                    "fps": self._setup_panel.spin_fps.value(),
                    "reference_body_size": self._detection_panel.spin_reference_body_size.value(),
                    # Frame range
                    "start_frame": (
                        self._setup_panel.spin_start_frame.value()
                        if self._setup_panel.spin_start_frame.isEnabled()
                        else 0
                    ),
                    "end_frame": (
                        self._setup_panel.spin_end_frame.value()
                        if self._setup_panel.spin_end_frame.isEnabled()
                        else None
                    ),
                }
            )

        compute_runtime = self._selected_compute_runtime()
        pose_runtime_derived = derive_pose_runtime_settings(
            compute_runtime,
            backend_family=self._identity_panel.combo_pose_model_type.currentText()
            .strip()
            .lower(),
        )

        cfg.update(
            {
                # === SYSTEM PERFORMANCE ===
                "resize_factor": self._setup_panel.spin_resize.value(),
                "save_confidence_metrics": self._setup_panel.check_save_confidence.isChecked(),
                "use_cached_detections": self._setup_panel.chk_use_cached_detections.isChecked(),
                "visualization_free_mode": self._setup_panel.chk_visualization_free.isChecked(),
                # === DETECTION STRATEGY ===
                "detection_method": (
                    "background_subtraction"
                    if self._detection_panel.combo_detection_method.currentIndex() == 0
                    else "yolo_obb"
                ),
                # === SIZE FILTERING ===
                "enable_size_filtering": self._detection_panel.chk_size_filtering.isChecked(),
                "min_object_size_multiplier": self._detection_panel.spin_min_object_size.value(),
                "max_object_size_multiplier": self._detection_panel.spin_max_object_size.value(),
                # === IMAGE ENHANCEMENT ===
                "brightness": self._detection_panel.slider_brightness.value(),
                "contrast": self._detection_panel.slider_contrast.value() / 100.0,
                "gamma": self._detection_panel.slider_gamma.value() / 100.0,
                "dark_on_light_background": self._detection_panel.chk_dark_on_light.isChecked(),
                # === BACKGROUND SUBTRACTION ===
                "background_prime_seconds": self._detection_panel.spin_bg_prime.value(),
                "enable_adaptive_background": self._detection_panel.chk_adaptive_bg.isChecked(),
                "background_learning_rate": self._detection_panel.spin_bg_learning.value(),
                "subtraction_threshold": self._detection_panel.spin_threshold.value(),
                # === LIGHTING STABILIZATION ===
                "enable_lighting_stabilization": self._detection_panel.chk_lighting_stab.isChecked(),
                "lighting_smooth_factor": self._detection_panel.spin_lighting_smooth.value(),
                "lighting_median_window": self._detection_panel.spin_lighting_median.value(),
                # === MORPHOLOGY & NOISE ===
                "morph_kernel_size": self._detection_panel.spin_morph_size.value(),
                "min_contour_area": self._detection_panel.spin_min_contour.value(),
                "max_contour_multiplier": self._detection_panel.spin_max_contour_multiplier.value(),
                # === ADVANCED SEPARATION ===
                "enable_conservative_split": self._detection_panel.chk_conservative_split.isChecked(),
                "conservative_kernel_size": self._detection_panel.spin_conservative_kernel.value(),
                "conservative_erode_iterations": self._detection_panel.spin_conservative_erode.value(),
                "enable_additional_dilation": self._detection_panel.chk_additional_dilation.isChecked(),
                "dilation_kernel_size": self._detection_panel.spin_dilation_kernel_size.value(),
                "dilation_iterations": self._detection_panel.spin_dilation_iterations.value(),
                # === YOLO CONFIGURATION ===
                # Store relative path if model is in archive, otherwise absolute
                "yolo_model_path": make_model_path_relative(yolo_path),
                "yolo_obb_mode": yolo_mode,
                "yolo_obb_direct_model_path": make_model_path_relative(
                    yolo_direct_path
                ),
                "yolo_detect_model_path": make_model_path_relative(yolo_detect_path),
                "yolo_crop_obb_model_path": make_model_path_relative(
                    yolo_crop_obb_path
                ),
                "yolo_headtail_model_path": make_model_path_relative(
                    yolo_headtail_path
                ),
                "yolo_headtail_model_type": self._identity_panel.combo_yolo_headtail_model_type.currentText(),
                "pose_overrides_headtail": self._identity_panel.chk_pose_overrides_headtail.isChecked(),
                "yolo_seq_crop_pad_ratio": self._detection_panel.spin_yolo_seq_crop_pad.value(),
                "yolo_seq_min_crop_size_px": self._detection_panel.spin_yolo_seq_min_crop_px.value(),
                "yolo_seq_enforce_square_crop": self._detection_panel.chk_yolo_seq_square_crop.isChecked(),
                "yolo_seq_stage2_imgsz": self._detection_panel.spin_yolo_seq_stage2_imgsz.value(),
                "yolo_seq_stage2_pow2_pad": self._detection_panel.chk_yolo_seq_stage2_pow2_pad.isChecked(),
                "yolo_seq_detect_conf_threshold": self._detection_panel.spin_yolo_seq_detect_conf.value(),
                "yolo_headtail_conf_threshold": self._identity_panel.spin_yolo_headtail_conf.value(),
                "reference_aspect_ratio": self._detection_panel.spin_reference_aspect_ratio.value(),
                "enable_aspect_ratio_filtering": self._detection_panel.chk_enable_aspect_ratio_filtering.isChecked(),
                "min_aspect_ratio_multiplier": self._detection_panel.spin_min_ar_multiplier.value(),
                "max_aspect_ratio_multiplier": self._detection_panel.spin_max_ar_multiplier.value(),
                "yolo_confidence_threshold": self._detection_panel.spin_yolo_confidence.value(),
                "yolo_iou_threshold": self._detection_panel.spin_yolo_iou.value(),
                "use_custom_obb_iou_filtering": self._detection_panel.chk_use_custom_obb_iou.isChecked(),
                "yolo_target_classes": yolo_cls,
            }
        )
        yolo_meta = get_yolo_model_metadata(yolo_path) or {}
        if yolo_meta:
            cfg["yolo_model_size"] = yolo_meta.get("size", "")
            cfg["yolo_model_species"] = yolo_meta.get("species", "")
            cfg["yolo_model_info"] = yolo_meta.get("model_info", "")
            cfg["yolo_model_added_at"] = yolo_meta.get("added_at", "")

        role_models = {
            "yolo_obb_direct": yolo_direct_path,
            "yolo_seq_detect": yolo_detect_path,
            "yolo_seq_crop_obb": yolo_crop_obb_path,
            "yolo_headtail": yolo_headtail_path,
        }
        for role_key, model_path in role_models.items():
            if not model_path:
                continue
            role_meta = get_yolo_model_metadata(model_path) or {}
            if not role_meta:
                continue
            cfg[f"{role_key}_model_size"] = role_meta.get("size", "")
            cfg[f"{role_key}_model_species"] = role_meta.get("species", "")
            cfg[f"{role_key}_model_info"] = role_meta.get("model_info", "")
            cfg[f"{role_key}_model_added_at"] = role_meta.get("added_at", "")
            cfg[f"{role_key}_task_family"] = role_meta.get("task_family", "")
            cfg[f"{role_key}_usage_role"] = role_meta.get("usage_role", "")

        # === COMPUTE RUNTIME ===
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        cfg["compute_runtime"] = compute_runtime
        # Keep legacy fields writable for backward compatibility.
        if not preset_mode:
            cfg["yolo_device"] = runtime_detection["yolo_device"]

        cfg.update(
            {
                # TensorRT
                "enable_tensorrt": runtime_detection["enable_tensorrt"],
                "tensorrt_max_batch_size": (
                    self._detection_panel.spin_yolo_batch_size.value()
                    if self._runtime_requires_fixed_yolo_batch(compute_runtime)
                    else self._detection_panel.spin_tensorrt_batch.value()
                ),
                # YOLO Batching
                "enable_yolo_batching": self._detection_panel.chk_enable_yolo_batching.isChecked(),
                "yolo_batch_size_mode": (
                    "auto"
                    if self._detection_panel.combo_yolo_batch_mode.currentIndex() == 0
                    else "manual"
                ),
                "yolo_manual_batch_size": self._detection_panel.spin_yolo_batch_size.value(),
                # === CORE TRACKING ===
                "max_targets": self._setup_panel.spin_max_targets.value(),
                "max_assignment_distance_multiplier": self._tracking_panel.spin_max_dist.value(),
                "recovery_search_distance_multiplier": self._tracking_panel.spin_continuity_thresh.value(),
                "enable_backward_tracking": self._tracking_panel.chk_enable_backward.isChecked(),
                # === KALMAN FILTER ===
                "kalman_process_noise": self._tracking_panel.spin_kalman_noise.value(),
                "kalman_measurement_noise": self._tracking_panel.spin_kalman_meas.value(),
                "kalman_velocity_damping": self._tracking_panel.spin_kalman_damping.value(),
                "kalman_maturity_age_seconds": self._tracking_panel.spin_kalman_maturity_age.value(),
                "kalman_initial_velocity_retention": self._tracking_panel.spin_kalman_initial_velocity_retention.value(),
                "kalman_max_velocity_multiplier": self._tracking_panel.spin_kalman_max_velocity.value(),
                "kalman_longitudinal_noise_multiplier": self._tracking_panel.spin_kalman_longitudinal_noise.value(),
                "kalman_lateral_noise_multiplier": self._tracking_panel.spin_kalman_lateral_noise.value(),
                # === COST FUNCTION WEIGHTS ===
                "weight_position": self._tracking_panel.spin_Wp.value(),
                "weight_orientation": self._tracking_panel.spin_Wo.value(),
                "weight_area": self._tracking_panel.spin_Wa.value(),
                "weight_aspect_ratio": self._tracking_panel.spin_Wasp.value(),
                "weight_pose_direction": 0.5,
                "weight_pose_length": 0.0,
                "pose_valid_orientation_scale": 0.15,
                "use_mahalanobis_distance": self._tracking_panel.chk_use_mahal.isChecked(),
                # === ASSIGNMENT ALGORITHM ===
                "enable_greedy_assignment": self._tracking_panel.combo_assignment_method.currentIndex()
                == 1,
                "enable_spatial_optimization": self._tracking_panel.chk_spatial_optimization.isChecked(),
                "association_stage1_motion_gate_multiplier": self._tracking_panel.spin_assoc_gate_multiplier.value(),
                "association_stage1_max_area_ratio": self._tracking_panel.spin_assoc_max_area_ratio.value(),
                "association_stage1_max_aspect_diff": self._tracking_panel.spin_assoc_max_aspect_diff.value(),
                "enable_pose_rejection": self._tracking_panel.chk_enable_pose_rejection.isChecked(),
                "pose_rejection_threshold": self._tracking_panel.spin_pose_rejection_threshold.value(),
                "pose_rejection_min_visibility": self._tracking_panel.spin_pose_rejection_min_visibility.value(),
                "track_feature_ema_alpha": self._tracking_panel.spin_track_feature_ema_alpha.value(),
                "association_high_confidence_threshold": self._tracking_panel.spin_assoc_high_conf_threshold.value(),
                # === ORIENTATION & MOTION ===
                "velocity_threshold": self._tracking_panel.spin_velocity.value(),
                "enable_instant_flip": self._tracking_panel.chk_instant_flip.isChecked(),
                "max_orientation_delta_stopped": self._tracking_panel.spin_max_orient.value(),
                "directed_orient_smoothing": self._tracking_panel.chk_directed_orient_smoothing.isChecked(),
                "directed_orient_flip_confidence": self._tracking_panel.spin_directed_orient_flip_conf.value(),
                "directed_orient_flip_persistence": self._tracking_panel.spin_directed_orient_flip_persist.value(),
                # === TRACK LIFECYCLE ===
                "lost_threshold_seconds": self._tracking_panel.spin_lost_thresh.value(),
                "min_respawn_distance_multiplier": self._tracking_panel.spin_min_respawn_distance.value(),
                "min_detections_to_start_seconds": self._tracking_panel.spin_min_detections_to_start.value(),
                "min_detect_seconds": self._tracking_panel.spin_min_detect.value(),
                "min_track_seconds": self._tracking_panel.spin_min_track.value(),
                # === POST-PROCESSING ===
                "enable_postprocessing": self._postprocess_panel.enable_postprocessing.isChecked(),
                "min_trajectory_length_seconds": self._postprocess_panel.spin_min_trajectory_length.value(),
                "max_velocity_break": self._postprocess_panel.spin_max_velocity_break.value(),
                "max_occlusion_gap_seconds": self._postprocess_panel.spin_max_occlusion_gap.value(),
                "enable_tracklet_relinking": self._postprocess_panel.chk_enable_tracklet_relinking.isChecked(),
                "relink_pose_max_distance": self._postprocess_panel.spin_relink_pose_max_distance.value(),
                "pose_export_min_valid_fraction": self._postprocess_panel.spin_pose_export_min_valid_fraction.value(),
                "pose_export_min_valid_keypoints": self._postprocess_panel.spin_pose_export_min_valid_keypoints.value(),
                "relink_min_pose_quality": self._postprocess_panel.spin_relink_min_pose_quality.value(),
                "pose_postproc_max_gap": self._postprocess_panel.spin_pose_postproc_max_gap.value(),
                "pose_temporal_outlier_zscore": self._postprocess_panel.spin_pose_temporal_outlier_zscore.value(),
                "enable_confidence_density_map": self._tracking_panel.chk_enable_confidence_density_map.isChecked(),
                "density_gaussian_sigma_scale": self._tracking_panel.spin_density_gaussian_sigma_scale.value(),
                "density_temporal_sigma": self._tracking_panel.spin_density_temporal_sigma.value(),
                "density_binarize_threshold": self._tracking_panel.spin_density_binarize_threshold.value(),
                "density_conservative_factor": self._tracking_panel.spin_density_conservative_factor.value(),
                "density_min_frame_duration": self._tracking_panel.spin_density_min_duration.value(),
                "density_min_area_bodies": self._tracking_panel.spin_density_min_area_bodies.value(),
                "density_downsample_factor": self._tracking_panel.spin_density_downsample_factor.value(),
                "max_velocity_zscore": self._postprocess_panel.spin_max_velocity_zscore.value(),
                "velocity_zscore_window_seconds": self._postprocess_panel.spin_velocity_zscore_window.value(),
                "velocity_zscore_min_velocity": self._postprocess_panel.spin_velocity_zscore_min_vel.value(),
                "interpolation_method": self._postprocess_panel.combo_interpolation_method.currentText(),
                "interpolation_max_gap_seconds": self._postprocess_panel.spin_interpolation_max_gap.value(),
                "heading_flip_max_burst": self._postprocess_panel.spin_heading_flip_max_burst.value(),
                "cleanup_temp_files": self._postprocess_panel.chk_cleanup_temp_files.isChecked(),
                # === TRAJECTORY MERGING (Conservative Strategy) ===
                # Agreement distance and min overlap frames for conservative merging
                "merge_agreement_distance_multiplier": self._postprocess_panel.spin_merge_overlap_multiplier.value(),
                "min_overlap_frames": self._postprocess_panel.spin_min_overlap_frames.value(),
                # === VIDEO VISUALIZATION ===
                "video_show_labels": self._postprocess_panel.check_show_labels.isChecked(),
                "video_show_orientation": self._postprocess_panel.check_show_orientation.isChecked(),
                "video_show_trails": self._postprocess_panel.check_show_trails.isChecked(),
                "video_trail_duration": self._postprocess_panel.spin_trail_duration.value(),
                "video_marker_size": self._postprocess_panel.spin_marker_size.value(),
                "video_text_scale": self._postprocess_panel.spin_text_scale.value(),
                "video_arrow_length": self._postprocess_panel.spin_arrow_length.value(),
                "video_show_pose": self._postprocess_panel.check_video_show_pose.isChecked(),
                "video_pose_color_mode": (
                    "track"
                    if self._postprocess_panel.combo_video_pose_color_mode.currentIndex()
                    == 0
                    else "fixed"
                ),
                "video_pose_color": [
                    int(self._postprocess_panel._video_pose_color[0]),
                    int(self._postprocess_panel._video_pose_color[1]),
                    int(self._postprocess_panel._video_pose_color[2]),
                ],
                "video_pose_point_radius": self._postprocess_panel.spin_video_pose_point_radius.value(),
                "video_pose_point_thickness": self._postprocess_panel.spin_video_pose_point_thickness.value(),
                "video_pose_line_thickness": self._postprocess_panel.spin_video_pose_line_thickness.value(),
                # === VISUALIZATION OVERLAYS ===
                "show_track_markers": self._setup_panel.chk_show_circles.isChecked(),
                "show_orientation_lines": self._setup_panel.chk_show_orientation.isChecked(),
                "show_trajectory_trails": self._setup_panel.chk_show_trajectories.isChecked(),
                "show_id_labels": self._setup_panel.chk_show_labels.isChecked(),
                "show_state_text": self._setup_panel.chk_show_state.isChecked(),
                "show_kalman_uncertainty": self._setup_panel.chk_show_kalman_uncertainty.isChecked(),
                "show_foreground_mask": self._detection_panel.chk_show_fg.isChecked(),
                "show_background_model": self._detection_panel.chk_show_bg.isChecked(),
                "show_yolo_obb": self._detection_panel.chk_show_yolo_obb.isChecked(),
                "trajectory_history_seconds": self._setup_panel.spin_traj_hist.value(),
                "debug_logging": self._setup_panel.chk_debug_logging.isChecked(),
                "enable_profiling": self._setup_panel.chk_enable_profiling.isChecked(),
                "zoom_factor": self.slider_zoom.value() / 100.0,
            }
        )

        # === ROI ===
        # Skip ROI when saving as preset
        if not preset_mode:
            cfg["roi_shapes"] = self.roi_shapes

        cfg.update(
            {
                # === DATASET GENERATION ===
                "enable_dataset_generation": self._dataset_panel.chk_enable_dataset_gen.isChecked(),
                "dataset_class_name": self._dataset_panel.line_dataset_class_name.text(),
                "dataset_max_frames": self._dataset_panel.spin_dataset_max_frames.value(),
            }
        )

        cfg.update(
            {
                "dataset_confidence_threshold": self._dataset_panel.spin_dataset_conf_threshold.value(),
                # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
                "dataset_diversity_window": self._dataset_panel.spin_dataset_diversity_window.value(),
                "dataset_include_context": self._dataset_panel.chk_dataset_include_context.isChecked(),
                "dataset_probabilistic_sampling": self._dataset_panel.chk_dataset_probabilistic.isChecked(),
                "metric_low_confidence": self._dataset_panel.chk_metric_low_confidence.isChecked(),
                "metric_count_mismatch": self._dataset_panel.chk_metric_count_mismatch.isChecked(),
                "metric_high_assignment_cost": self._dataset_panel.chk_metric_high_assignment_cost.isChecked(),
                "metric_track_loss": self._dataset_panel.chk_metric_track_loss.isChecked(),
                "metric_high_uncertainty": self._dataset_panel.chk_metric_high_uncertainty.isChecked(),
                # === INDIVIDUAL ANALYSIS ===
                "enable_identity_analysis": self._is_individual_pipeline_enabled(),
                "enable_individual_pipeline": self._is_individual_pipeline_enabled(),
                "identity_method": self._selected_identity_method(),
                "use_apriltags": self._identity_config().get("use_apriltags", False),
                "cnn_classifiers": self._identity_config().get("cnn_classifiers", []),
                # Legacy CNN Classifier settings (for backward compat on load)
                "cnn_classifier_confidence": self._identity_panel.spin_cnn_confidence.value(),
                "identity_match_bonus": self._identity_panel.spin_identity_match_bonus.value(),
                "identity_mismatch_penalty": self._identity_panel.spin_identity_mismatch_penalty.value(),
                "cnn_classifier_match_bonus": self._identity_panel.spin_identity_match_bonus.value(),
                "cnn_classifier_mismatch_penalty": self._identity_panel.spin_identity_mismatch_penalty.value(),
                "cnn_classifier_window": self._identity_panel.spin_cnn_window.value(),
            }
        )

        cfg.update(
            {
                "apriltag_family": self._identity_panel.combo_apriltag_family.currentText(),
                "apriltag_decimate": self._identity_panel.spin_apriltag_decimate.value(),
                "tag_match_bonus": self._identity_panel.spin_identity_match_bonus.value(),
                "tag_mismatch_penalty": self._identity_panel.spin_identity_mismatch_penalty.value(),
                "enable_pose_extractor": self._identity_panel.chk_enable_pose_extractor.isChecked(),
                "pose_model_type": self._identity_panel.combo_pose_model_type.currentText()
                .strip()
                .lower(),
                "pose_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend(
                        self._identity_panel.combo_pose_model_type.currentText()
                        .strip()
                        .lower()
                    )
                ),
                "pose_yolo_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend("yolo")
                ),
                "pose_sleap_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend("sleap")
                ),
                "pose_runtime_flavor": pose_runtime_derived["pose_runtime_flavor"],
                "pose_exported_model_path": "",
                "pose_min_kpt_conf_valid": self._identity_panel.spin_pose_min_kpt_conf_valid.value(),
                "pose_skeleton_file": self._identity_panel.line_pose_skeleton_file.text().strip(),
                "pose_ignore_keypoints": self._parse_pose_ignore_keypoints(),
                "pose_direction_anterior_keypoints": self._parse_pose_direction_anterior_keypoints(),
                "pose_direction_posterior_keypoints": self._parse_pose_direction_posterior_keypoints(),
                "pose_batch_size": self._identity_panel.spin_pose_batch.value(),
                "pose_yolo_batch": self._identity_panel.spin_pose_batch.value(),
                "pose_sleap_env": self._selected_pose_sleap_env(),
                "pose_sleap_device": pose_runtime_derived["pose_sleap_device"],
                "pose_sleap_batch": self._identity_panel.spin_pose_batch.value(),
                "pose_sleap_max_instances": 1,
                "pose_sleap_experimental_features": self._sleap_experimental_features_enabled(),
                # === REAL-TIME INDIVIDUAL DATASET ===
                "enable_individual_dataset": self._is_individual_image_save_enabled(),
                "enable_individual_image_save": self._is_individual_image_save_enabled(),
                "generate_oriented_track_videos": bool(
                    self._dataset_panel.chk_generate_individual_track_videos.isChecked()
                ),
                "individual_output_format": self._dataset_panel.combo_individual_format.currentText().lower(),
            }
        )

        cfg.update(
            {
                "individual_save_interval": self._dataset_panel.spin_individual_interval.value(),
                "individual_interpolate_occlusions": self._identity_panel.chk_individual_interpolate.isChecked(),
                "individual_crop_padding": self._identity_panel.spin_individual_padding.value(),
                "individual_background_color": [
                    int(c) for c in self._identity_panel._background_color
                ],  # Ensure JSON serializable
                "suppress_foreign_obb_regions": self._identity_panel.chk_suppress_foreign_obb.isChecked(),
                "suppress_foreign_obb_dataset": self._dataset_panel.chk_suppress_foreign_obb_dataset.isChecked(),
            }
        )

        # If preset mode with path provided, save directly
        if preset_mode and preset_path:
            os.makedirs(os.path.dirname(preset_path), exist_ok=True)
            ok, err = self._atomic_json_write(cfg, preset_path)
            if ok:
                logger.info(f"Saved preset to {preset_path}")
                return True
            logger.error(f"Failed to save preset: {err}")
            QMessageBox.critical(self, "Save Error", f"Failed to save preset:\n{err}")
            return False

        config_path = self._resolve_config_save_path(prompt_if_exists)
        if not config_path:
            return False

        ok, err = self._atomic_json_write(cfg, config_path)
        if ok:
            logger.info(f"Configuration saved to {config_path} (including ROI shapes)")
            return True
        logger.warning(f"Failed to save configuration: {err}")
        return False

    def _setup_session_logging(self, video_path, backward_mode=False):
        """Set up comprehensive logging for the entire tracking session."""
        from datetime import datetime
        from pathlib import Path

        # Close existing session log if any
        self._cleanup_session_logging()

        # Only set up logging if not already set up
        if self.session_log_handler is not None:
            logger.info("=" * 80)
            logger.info("Session log already active, continuing...")
            logger.info("=" * 80)
            return

        # Create a session log in the video's dedicated log directory.
        video_path = Path(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = (
            os.path.dirname(self._setup_panel.csv_line.text())
            if self._setup_panel.csv_line.text()
            else ""
        )
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        log_path = build_tracking_session_log_path(
            video_path,
            timestamp,
            artifact_base_dir=artifact_base_dir,
            create_dir=True,
        )

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

        logger.info("=" * 80)
        logger.info("TRACKING SESSION STARTED")
        logger.info(f"Session log: {log_path}")
        logger.info(f"Video: {video_path}")
        logger.info("=" * 80)

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

    def _generate_training_dataset(self, override_csv_path=None):
        """Generate training dataset from tracking results for active learning."""
        try:
            if self._stop_all_requested:
                return
            logger.info("Starting training dataset generation...")

            # Prevent launching overlapping dataset threads; this can lead to
            # QThread destruction while still running if references are replaced.
            if self.dataset_worker is not None and self.dataset_worker.isRunning():
                logger.warning(
                    "Dataset generation already in progress; skipping duplicate request."
                )
                return
            if self.dataset_worker is not None and not self.dataset_worker.isRunning():
                self.dataset_worker.deleteLater()
                self.dataset_worker = None

            video_path = self._setup_panel.file_line.text()
            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Source video file not found."
                )
                return

            # Validate parameters
            # Auto-compute output directory
            output_dir = os.path.join(
                os.path.dirname(video_path),
                f"{os.path.splitext(os.path.basename(video_path))[0]}_datasets",
                "active_learning",
            )

            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Dataset Generation Error",
                        f"Could not create output directory: {output_dir}\nError: {e}",
                    )
                    return

            # Use override path if provided (e.g. valid processed CSV), otherwise fallback to UI field
            csv_path = (
                override_csv_path
                if override_csv_path
                else self._setup_panel.csv_line.text()
            )

            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Tracking CSV file not found."
                )
                return

            # Get parameters
            params = self.get_parameters_dict()
            max_frames = self._dataset_panel.spin_dataset_max_frames.value()
            diversity_window = self._dataset_panel.spin_dataset_diversity_window.value()
            include_context = (
                self._dataset_panel.chk_dataset_include_context.isChecked()
            )
            probabilistic = self._dataset_panel.chk_dataset_probabilistic.isChecked()

            # Get class name
            class_name = self._dataset_panel.line_dataset_class_name.text().strip()
            if not class_name:
                class_name = "object"

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Preparing dataset generation...")

            # Create and start dataset generation worker thread
            self.dataset_worker = DatasetGenerationWorker(
                video_path=video_path,
                csv_path=csv_path,
                detection_cache_path=self.current_detection_cache_path,
                output_dir=output_dir,
                dataset_name="",
                class_name=class_name,
                params=params,
                max_frames=max_frames,
                diversity_window=diversity_window,
                include_context=include_context,
                probabilistic=probabilistic,
            )
            self.dataset_worker.progress_signal.connect(self.on_dataset_progress)
            self.dataset_worker.finished_signal.connect(self.on_dataset_finished)
            self.dataset_worker.error_signal.connect(self.on_dataset_error)
            self.dataset_worker.finished.connect(
                self._on_dataset_worker_thread_finished
            )
            self.dataset_worker.start()

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Dataset Generation Error",
                f"Failed to generate dataset:\n{str(e)}",
            )

    def on_dataset_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during dataset generation."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            return
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_dataset_finished(
        self: object, dataset_dir: object, num_frames: object
    ) -> object:
        """Handle dataset generation completion."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._refresh_progress_visibility()
            return
        self._refresh_progress_visibility()

        logger.info(f"Dataset generation complete: {dataset_dir}")
        logger.info(f"Frames exported: {num_frames}")
        logger.info(
            "Use 'Open Dataset in X-AnyLabeling' button to review/correct annotations"
        )

        # Store result; popup is deferred to end-of-session summary.
        self._session_result_dataset = {
            "success": True,
            "num_frames": num_frames,
            "dir": dataset_dir,
        }
        if getattr(self, "_show_summary_on_dataset_done", False):
            self._show_summary_on_dataset_done = False
            self._show_session_summary()

    def on_dataset_error(self: object, error_message: object) -> object:
        """Handle dataset generation errors."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._refresh_progress_visibility()
            return
        self._refresh_progress_visibility()

        logger.error(f"Dataset generation error: {error_message}")

        # Store result; popup is deferred to end-of-session summary.
        self._session_result_dataset = {
            "success": False,
            "error": error_message,
        }
        if getattr(self, "_show_summary_on_dataset_done", False):
            self._show_summary_on_dataset_done = False
            self._show_session_summary()

    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        lines = []

        # --- Timing ---
        if self._session_wall_start is not None:
            elapsed = time.time() - self._session_wall_start
            h = int(elapsed // 3600)
            m = int((elapsed % 3600) // 60)
            s = int(elapsed % 60)
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            lines.append(f"Duration: {elapsed_str}")

        # --- Frames / FPS ---
        frames = self._session_frames_processed
        if frames > 0:
            lines.append(f"Frames processed: {frames}")
        fps_vals = [f for f in self._session_fps_list if f and f > 0]
        if fps_vals:
            avg_fps = sum(fps_vals) / len(fps_vals)
            lines.append(f"Average FPS: {avg_fps:.1f}")

        # --- Video / CSV ---
        video_path = self._setup_panel.file_line.text()
        if video_path:
            lines.append(f"Video: {os.path.basename(video_path)}")
        csv_path = self._session_final_csv_path or self._setup_panel.csv_line.text()
        if csv_path:
            lines.append(f"Output CSV: {os.path.basename(csv_path)}")

        # --- Trajectory / track count ---
        if csv_path and os.path.exists(csv_path):
            try:
                _df = pd.read_csv(csv_path, usecols=["TrajectoryID"])
                n_trajs = int(_df["TrajectoryID"].nunique())
                lines.append(f"Trajectories: {n_trajs}")
            except Exception:
                pass

        # --- Pipelines run ---
        pipelines = []
        if self._postprocess_panel.enable_postprocessing.isChecked():
            pipelines.append("Post-processing")
        if self._tracking_panel.chk_enable_backward.isChecked():
            pipelines.append("Backward tracking")
        if self._is_individual_pipeline_enabled():
            pipelines.append("Individual analysis")
            if self._identity_panel.chk_enable_pose_extractor.isChecked():
                pipelines.append("Pose extraction")
        if pipelines:
            lines.append("Pipelines: " + ", ".join(pipelines))

        # --- Separator before optional sub-results ---
        lines.append("")

        # --- Dataset generation result ---
        result = getattr(self, "_session_result_dataset", None)
        if result is not None:
            if result.get("success"):
                lines.append(
                    f"\u2713 Dataset generated: {result['num_frames']} frame(s)"
                    f"\n  Location: {result['dir']}"
                )
            else:
                lines.append(
                    f"\u2717 Dataset generation failed: {result.get('error', 'unknown error')}"
                )

        # Clean up state
        self._session_result_dataset = None
        self._dataset_was_started = False
        self._show_summary_on_dataset_done = False

        QMessageBox.information(self, "Tracking Complete", "\n".join(lines))

        # Offer to open RefineKit for interactive proofreading
        self._postprocess_panel._btn_open_refinekit.setEnabled(
            bool(self.current_video_path)
        )
        if self.current_video_path:
            reply = QMessageBox.question(
                self,
                "Open RefineKit?",
                "Tracking complete. Open in RefineKit for "
                "interactive identity proofreading?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._open_refinekit()

    def _open_refinekit(self):
        """Launch RefineKit as a subprocess pointing at the current video."""
        import subprocess
        import sys

        video_path = self.current_video_path
        if not video_path:
            QMessageBox.warning(
                self,
                "No Video",
                "No video is currently loaded. Please load a video first.",
            )
            return
        cmd = [sys.executable, "-m", "hydra_suite.refinekit.app", str(video_path)]
        logger.info("Launching RefineKit: %s", " ".join(cmd))
        subprocess.Popen(cmd)

    def _on_dataset_worker_thread_finished(self):
        """Release completed dataset worker safely."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("dataset_worker")
        self._refresh_progress_visibility()

    def _is_worker_running(self, worker):
        """Safely check whether a worker thread-like object is running."""
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    def _has_active_progress_task(self) -> bool:
        """Return True if any async task that owns progress UI is still active."""
        return any(
            [
                self._is_worker_running(self.tracking_worker),
                self._is_worker_running(getattr(self, "merge_worker", None)),
                self._is_worker_running(self.dataset_worker),
                self._is_worker_running(self.interp_worker),
                self._is_worker_running(self.oriented_video_worker),
            ]
        )

    def _refresh_progress_visibility(self):
        """Keep progress UI visible while any async tracking task is still running."""
        has_active_task = self._has_active_progress_task()
        self.progress_bar.setVisible(has_active_task)
        self.progress_label.setVisible(has_active_task)

    def _cleanup_temporary_files(self):
        """Remove temporary files if cleanup is enabled."""
        if not self._postprocess_panel.chk_cleanup_temp_files.isChecked():
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

        # Also clean up posekit directories if they exist
        params = self.get_parameters_dict()
        output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
        if output_dir and os.path.exists(output_dir):
            posekit_dir = os.path.join(output_dir, "posekit")
            if os.path.exists(posekit_dir) and os.path.isdir(posekit_dir):
                try:
                    import shutil

                    shutil.rmtree(posekit_dir)
                    logger.info(f"Removed posekit directory: {posekit_dir}")
                    cleaned.append("posekit/")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove posekit directory {posekit_dir}: {e}"
                    )
                    failed.append("posekit/")

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

    def _create_help_label(self, text, attach_to_title=True):
        """Create a styled help label for section guidance."""
        label = CompactHelpLabel(text, attach_to_title=attach_to_title)
        return label

    def _invalidate_roi_cache(self):
        """Invalidate ROI display cache when ROI changes."""
        self._roi_masked_cache.clear()

    # =========================================================================
    # PRESET MANAGEMENT
    # =========================================================================

    def _get_presets_dir(self):
        """Get the presets directory path."""
        from hydra_suite.paths import get_presets_dir

        return str(get_presets_dir())

    def _populate_preset_combo(self):
        """Populate the preset combo box by auto-scanning configs folder."""
        presets_dir = self._get_presets_dir()

        if not os.path.exists(presets_dir):
            return

        presets = []
        custom_preset = None

        # Scan all JSON files in configs directory
        for filename in sorted(os.listdir(presets_dir)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(presets_dir, filename)

            try:
                with open(filepath, "r") as f:
                    cfg = json.load(f)

                # Get preset name from file, fallback to filename
                preset_name = cfg.get(
                    "preset_name",
                    filename.replace(".json", "").replace("_", " ").title(),
                )

                # Custom preset gets special treatment (star marker, shown first)
                if filename == "custom.json":
                    custom_preset = (f"{preset_name} ★", filepath)
                else:
                    presets.append((preset_name, filepath))
            except Exception as e:
                logger.warning(f"Failed to load preset {filename}: {e}")
                continue

        # Populate combo box (custom first, then others alphabetically)
        self._setup_panel.combo_presets.clear()
        if custom_preset:
            self._setup_panel.combo_presets.addItem(custom_preset[0], custom_preset[1])
        for name, filepath in presets:
            self._setup_panel.combo_presets.addItem(name, filepath)

    def _load_selected_preset(self):
        """Load the currently selected preset."""
        filepath = self._setup_panel.combo_presets.currentData()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(
                self, "Preset Not Found", f"Preset file not found: {filepath}"
            )
            return

        # Confirm if current settings differ significantly
        reply = QMessageBox.question(
            self,
            "Load Preset",
            f"Load preset: {self._setup_panel.combo_presets.currentText()}?\n\n"
            "This will replace your current parameter values.\n"
            "(Video-specific configs will still override presets when loading videos)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            # Use existing config loader in preset mode
            self._load_config_from_file(filepath, preset_mode=True)

            # Update status
            preset_name = self._setup_panel.combo_presets.currentText()
            self._setup_panel.preset_status_label.setText(f"✓ Loaded: {preset_name}")
            self._setup_panel.preset_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            self._setup_panel.preset_status_label.setVisible(True)
            logger.info(f"Loaded preset: {preset_name} from {filepath}")

    def _save_custom_preset(self):
        """Save current settings as custom preset with user-defined name and description."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit

        # Create dialog for preset metadata
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Preset")
        dialog.setModal(True)
        dialog_layout = QVBoxLayout(dialog)

        # Name input
        name_label = QLabel("Preset name (e.g., danio rerio / zebrafish)")
        name_label.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        name_input = QLineEdit()
        name_input.setPlaceholderText("Scientific Name (Common Name)")
        name_input.setText("Custom")

        # Description input
        desc_label = QLabel("Description (optional)")
        desc_label.setStyleSheet("color: #e0e0e0; font-weight: bold; margin-top: 10px;")
        desc_input = QTextEdit()
        desc_input.setPlaceholderText(
            "Describe the optimizations or use case for this preset..."
        )
        desc_input.setMaximumHeight(80)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        dialog_layout.addWidget(name_label)
        dialog_layout.addWidget(name_input)
        dialog_layout.addWidget(desc_label)
        dialog_layout.addWidget(desc_input)
        dialog_layout.addWidget(button_box)

        if dialog.exec() != QDialog.Accepted:
            return

        preset_name = name_input.text().strip() or "Custom"
        description = desc_input.toPlainText().strip() or "User-defined custom preset"

        # Ask user for filename
        presets_dir = self._get_presets_dir()
        os.makedirs(presets_dir, exist_ok=True)

        # Generate suggested filename from preset name
        suggested_filename = preset_name.lower()
        suggested_filename = suggested_filename.split("(")[
            0
        ].strip()  # Remove common name part
        suggested_filename = suggested_filename.replace(" ", "_").replace(".", "")
        suggested_filename = "".join(
            c for c in suggested_filename if c.isalnum() or c == "_"
        )
        suggested_filename = suggested_filename or "custom"
        suggested_path = os.path.join(presets_dir, f"{suggested_filename}.json")

        preset_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset As", suggested_path, "JSON Files (*.json)"
        )

        if not preset_path:
            return

        custom_path = preset_path

        # Use existing config saver in preset mode with custom metadata
        success = self.save_config(
            preset_mode=True,
            preset_path=custom_path,
            preset_name=preset_name,
            preset_description=description,
        )

        if success:
            # Refresh combo box to show new preset
            self._populate_preset_combo()
            # Select the newly saved preset
            for i in range(self._setup_panel.combo_presets.count()):
                item_path = self._setup_panel.combo_presets.itemData(i)
                if item_path == custom_path:
                    self._setup_panel.combo_presets.setCurrentIndex(i)
                    break

            self._setup_panel.preset_status_label.setText(f"✓ Saved: {preset_name}")
            self._setup_panel.preset_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            self._setup_panel.preset_status_label.setVisible(True)

            filename = os.path.basename(custom_path)
            is_custom = filename == "custom.json"

            QMessageBox.information(
                self,
                "Preset Saved",
                f"Your settings have been saved as:\n{preset_name}\n\n"
                f"Location: {custom_path}\n\n"
                + (
                    "This preset will be loaded automatically on startup and will appear\n"
                    "at the top of the preset selector with a ★ indicator."
                    if is_custom
                    else "This preset is now available in the preset selector."
                ),
            )

    def _load_default_preset_on_startup(self):
        """Load default preset on application startup."""
        presets_dir = self._get_presets_dir()

        # Try custom preset first
        custom_path = os.path.join(presets_dir, "custom.json")
        if os.path.exists(custom_path):
            logger.info("Loading custom preset on startup")
            self._load_config_from_file(custom_path, preset_mode=True)
            return

        # Fall back to default preset
        default_path = os.path.join(presets_dir, "default.json")
        if os.path.exists(default_path):
            logger.info("Loading default preset on startup")
            self._load_config_from_file(default_path, preset_mode=True)
            return

        logger.info("No preset found, using hardcoded defaults")

    # =========================================================================
    # ROI OPTIMIZATION AND VIDEO CROPPING
    # =========================================================================

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        # Store config in the package directory (where this file is located)
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())

        default_config = {
            "roi_crop_warning_threshold": 0.6,  # Warn if ROI is <60% of frame
            "roi_crop_auto_suggest": True,  # Auto-suggest cropping
            "roi_crop_remind_every_session": False,  # Remind every time or once
            "roi_crop_padding_fraction": 0.05,  # Padding as fraction of min(width, height) - typically 5%
            "video_crop_codec": "libx264",  # Codec for cropped videos (libx264 for quality)
            "video_crop_crf": 18,  # CRF quality (lower = better, 18 = visually lossless)
            "video_crop_preset": "medium",  # ffmpeg preset (ultrafast, fast, medium, slow, veryslow)
            # YOLO Batching - Memory Fractions (device-specific optimization)
            "mps_memory_fraction": 0.3,  # Conservative 30% of unified memory for MPS (Apple Silicon)
            "cuda_memory_fraction": 0.7,  # 70% of VRAM for CUDA (NVIDIA GPUs)
            "tensorrt_build_workspace_gb": 4.0,  # TensorRT builder workspace limit in GB
            "tensorrt_build_batch_size": None,  # Optional fixed TensorRT build batch override
            # Dataset Generation - YOLO Detection Parameters (separate from tracking)
            "dataset_yolo_confidence_threshold": 0.05,  # Very low - detect all animals including uncertain ones for annotation
            "dataset_yolo_iou_threshold": 0.5,  # Moderate - remove obvious duplicates but keep borderline cases for manual review
            "xanylabeling_env": "",  # Preferred X-AnyLabeling conda env
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded advanced config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load advanced config: {e}")
        else:
            # Auto-create advanced_config.json with defaults on first run
            # This makes the file discoverable for power users who want to customize
            try:
                config_dir = os.path.dirname(config_path)
                os.makedirs(config_dir, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default advanced config at {config_path}")
            except Exception as e:
                logger.warning(f"Could not create advanced config file: {e}")

        return default_config

    def _save_advanced_config(self):
        """Save advanced configuration."""
        from hydra_suite.paths import get_advanced_config_path

        config_path = str(get_advanced_config_path())
        try:
            import tempfile

            # Write to temp file first, then rename (atomic on most filesystems)
            config_dir = os.path.dirname(config_path)
            os.makedirs(config_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=config_dir, delete=False, suffix=".tmp"
            ) as tmp:
                json.dump(self.advanced_config, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)  # Atomic rename
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

    def crop_video_to_roi(self: object) -> object:
        """Crop the video to the ROI bounding box and save as new file."""
        if self.roi_mask is None:
            QMessageBox.warning(self, "No ROI", "Please define an ROI before cropping.")
            return

        video_path = self._setup_panel.file_line.text()
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
            from PySide6.QtCore import QTimer

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

    def _poll_crop_stderr_progress(self, process):
        """Read ffmpeg stderr in non-blocking mode and log progress."""
        if not process.stderr:
            return
        try:
            import fcntl
            import os as os_module

            fd = process.stderr.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os_module.O_NONBLOCK)
            try:
                while True:
                    line = process.stderr.readline()
                    if not line:
                        break
                    if "frame=" in line:
                        try:
                            frame_str = line.split("frame=")[1].split()[0]
                            current_frame = int(frame_str)
                            total_frames = self._crop_process.get("total_frames", 0)
                            if total_frames > 0:
                                progress_pct = int((current_frame / total_frames) * 100)
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
                pass
        except Exception:
            pass

    def _load_cropped_video(self, output_path):
        """Set up the UI to use the newly cropped video."""
        self._setup_panel.file_line.setText(output_path)
        self.current_video_path = output_path
        self.clear_roi()

        video_dir = os.path.dirname(output_path)
        video_name = os.path.splitext(os.path.basename(output_path))[0]

        csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
        self._setup_panel.csv_line.setText(csv_path)

        video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
        self._postprocess_panel.video_out_line.setText(video_out_path)
        self._postprocess_panel.check_video_output.setChecked(True)

        self.btn_test_detection.setEnabled(True)
        self._setup_panel.btn_detect_fps.setEnabled(True)
        self.btn_crop_video.setEnabled(False)
        if hasattr(self, "roi_optimization_label"):
            self.roi_optimization_label.setText("")

        config_path = get_video_config_path(output_path)
        if config_path and os.path.isfile(config_path):
            self._load_config_from_file(config_path)
            self._setup_panel.config_status_label.setText(
                f"\u2713 Loaded: {os.path.basename(config_path)}"
            )
            self._setup_panel.config_status_label.setStyleSheet(
                "color: #4fc1ff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Cropped video loaded: {output_path} (auto-loaded config)")
        else:
            self._setup_panel.config_status_label.setText(
                "No config found (using current settings)"
            )
            self._setup_panel.config_status_label.setStyleSheet(
                "color: #f39c12; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Cropped video loaded: {output_path} (no config found)")

    def _handle_crop_success(self, output_path, orig_w, orig_h, crop_w, crop_h):
        """Handle a successful crop completion."""
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
            self._load_cropped_video(output_path)

        self._set_ui_controls_enabled(True)
        if hasattr(self, "btn_crop_video"):
            self.btn_crop_video.setText("Crop Video to ROI")
        logger.info(f"Successfully cropped video to {output_path}")

    def _handle_crop_failure(self, return_code):
        """Handle a failed crop completion."""
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

    def _check_crop_completion(self):
        """Check if background crop process has completed."""
        if not hasattr(self, "_crop_process"):
            if hasattr(self, "_crop_check_timer"):
                self._crop_check_timer.stop()
            return

        process = self._crop_process["process"]
        self._poll_crop_stderr_progress(process)

        return_code = process.poll()
        if return_code is not None:
            self._crop_check_timer.stop()
            output_path = self._crop_process["output_path"]
            orig_w, orig_h = self._crop_process["original_size"]
            crop_w, crop_h = self._crop_process["cropped_size"]

            if return_code == 0 and os.path.exists(output_path):
                self._handle_crop_success(output_path, orig_w, orig_h, crop_w, crop_h)
            else:
                self._handle_crop_failure(return_code)

            del self._crop_process
