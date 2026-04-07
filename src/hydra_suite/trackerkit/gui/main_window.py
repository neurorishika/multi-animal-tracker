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
        if hasattr(self, "_session_orch"):
            return self._session_orch._load_ui_settings()
        # Fallback during __init__ before orchestrators exist
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
        if hasattr(self, "_session_orch"):
            self._session_orch._queue_ui_state_save()
            return
        if hasattr(self, "_ui_state_save_timer"):
            self._ui_state_save_timer.start()

    def _remember_collapsible_state(
        self, key: str, collapsible: CollapsibleGroupBox
    ) -> None:
        """Restore and track expanded state for a collapsible section."""
        if hasattr(self, "_session_orch"):
            self._session_orch._remember_collapsible_state(key, collapsible)
            return
        # Fallback during early init before orchestrators exist
        self._collapsible_state_widgets[key] = collapsible
        saved = self._ui_settings.get("collapsed_sections", {}).get(key)
        if isinstance(saved, bool):
            collapsible.setExpanded(saved)
        collapsible.toggled.connect(
            lambda _expanded, _key=key: self._queue_ui_state_save()
        )

    def _restore_ui_state(self) -> None:
        self._session_orch._restore_ui_state()

    def _save_ui_settings(self) -> None:
        self._session_orch._save_ui_settings()

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
        self.slider_zoom.valueChanged.connect(
            lambda v: self._detection_panel._on_zoom_changed(v) if hasattr(self, "_detection_panel") else None
        )

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
        self.btn_test_detection.clicked.connect(
            lambda: self._detection_panel._test_detection_on_preview() if hasattr(self, "_detection_panel") else None
        )
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
        self._detection_panel._refresh_yolo_model_combo()
        self._detection_panel._refresh_yolo_detect_model_combo()
        self._detection_panel._refresh_yolo_crop_obb_model_combo()

        # Tab 3: Individual Analysis (Identity)
        from hydra_suite.trackerkit.gui.panels.identity_panel import IdentityPanel

        self._identity_panel = IdentityPanel(
            main_window=self, config=self.config, parent=self
        )
        self.tabs.addTab(self._identity_panel, "Analyze Individuals")
        # Post-construction bootstrap calls now that _identity_panel is assigned
        self._identity_panel._refresh_cnn_identity_model_combo()
        self._identity_panel._refresh_yolo_headtail_model_combo()
        self._update_background_color_button()
        self._populate_pose_runtime_flavor_options(backend="yolo")
        self._set_form_row_visible(
            self._identity_panel.form_pose_runtime,
            self._identity_panel.combo_pose_runtime_flavor,
            False,
        )
        self._refresh_pose_model_combo()
        self._identity_panel._refresh_pose_sleap_envs()
        self._identity_panel._refresh_pose_direction_keypoint_lists()
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
        # Cross-panel wire: pose-toggle in IdentityPanel must re-run cleaning visibility
        self._identity_panel.g_pose_runtime.toggled.connect(
            self._postprocess_panel._on_cleaning_toggled
        )

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
        # INITIALIZE ORCHESTRATORS
        # =====================================================================
        from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator

        self._tracking_orch = TrackingOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )
        from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator

        self._config_orch = ConfigOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )
        from hydra_suite.trackerkit.gui.orchestrators.session import SessionOrchestrator

        self._session_orch = SessionOrchestrator(
            main_window=self,
            config=self.config,
            panels=self._panels_bundle(),
        )

        # =====================================================================
        # INITIALIZE PRESETS
        # =====================================================================
        # Populate preset combo box with available presets
        self._populate_preset_combo()

        # Load default preset (custom if available, otherwise default.json)
        self._load_default_preset_on_startup()

    def _panels_bundle(self):
        """Return a namespace of all panels for orchestrator access."""
        import types

        ns = types.SimpleNamespace()
        ns.setup = self._setup_panel
        ns.detection = self._detection_panel
        ns.tracking = self._tracking_panel
        ns.postprocess = self._postprocess_panel
        ns.dataset = self._dataset_panel
        ns.identity = self._identity_panel
        return ns

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
        return self._config_orch._find_or_plan_optimizer_cache_path()

    def _build_optimizer_detection_cache(
        self, video_path: str, cache_path: str, params: dict
    ):
        """Spin up a DetectionCacheBuilderWorker and show progress in the main window."""
        self._config_orch._build_optimizer_detection_cache()

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
        self._config_orch._apply_optimized_params(new_params)

    def _open_parameter_helper(self):
        """Open the tracking parameter selection helper dialog."""
        self._config_orch._open_parameter_helper()

    def _open_bg_parameter_helper(self):
        """Open the BG-subtraction parameter auto-tuner dialog."""
        self._config_orch._open_bg_parameter_helper()

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

    def _get_resolved_pose_model_dir(self, backend: str) -> object:
        """Resolve pose model path for context dicts — callable by DetectionPanel."""
        return resolve_pose_model_path(
            self._pose_model_path_for_backend(backend),
            backend=backend,
        )

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
        if hasattr(self, "_config_orch"):
            self._config_orch._populate_compute_runtime_options(preferred=preferred)
            return
        # Fallback during early init before orchestrators exist
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
            self._detection_panel._on_yolo_batch_mode_changed(
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
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_pose_backend_ui()
            return
        # Fallback during early init before orchestrators exist
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
        self._refresh_pose_model_combo(
            preferred_model_path=self._pose_model_path_for_backend(backend)
        )
        self._on_runtime_context_changed()

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_pose_inference_enabled()

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_video_pose_overlay_controls(*_args)

    def _is_yolo_detection_mode(self) -> bool:
        """Return True when current detection mode is YOLO OBB."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel._is_yolo_detection_mode()

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_individual_pipeline_enabled()

    def _is_identity_analysis_enabled(self) -> bool:
        """Return effective runtime state for identity classification."""
        if not hasattr(self, "_detection_panel"):
            return False
        return self._detection_panel._is_identity_analysis_enabled()

    def _selected_identity_method(self) -> str:
        """Return canonical identity-method key for runtime/config usage."""
        if not hasattr(self, "_detection_panel"):
            return "none_disabled"
        return self._detection_panel._selected_identity_method()

    def _identity_config(self) -> dict:
        """Return use_apriltags + cnn_classifiers config dict."""
        if not hasattr(self, "_detection_panel"):
            return {"use_apriltags": False, "cnn_classifiers": []}
        return self._detection_panel._identity_config()

    def _is_individual_image_save_enabled(self) -> bool:
        """Return effective runtime state for saving individual crops."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._is_individual_image_save_enabled()

    def _should_generate_oriented_track_videos(self) -> bool:
        """Return True when final per-track oriented videos should be exported."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._should_generate_oriented_track_videos()

    def _should_run_interpolated_postpass(self) -> bool:
        """Return True when interpolated post-pass should run."""
        if not hasattr(self, "_session_orch"):
            return False
        return self._session_orch._should_run_interpolated_postpass()

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
        if hasattr(self, "_identity_panel"):
            self._identity_panel._sync_identity_method_ui()
        if hasattr(self, "_identity_panel"):
            self._identity_panel._sync_pose_analysis_ui()
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
                str(self._identity_panel._get_selected_yolo_headtail_model_path() or "").strip()
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
        self._config_orch._setup_video_file(fp, skip_config_load=skip_config_load)

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
        if hasattr(self, "_session_orch"):
            self._session_orch._sync_batch_list_ui()

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
        if hasattr(self, "_detection_panel"):
            self._detection_panel._update_preview_display()

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
        if hasattr(self, "_session_orch"):
            self._session_orch._update_range_info()

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

    def _on_headtail_model_type_changed(self, _index: object = None) -> None:
        """Refresh the head-tail model combo when the user switches YOLO ↔ tiny."""
        if hasattr(self, "_identity_panel"):
            self._identity_panel._refresh_yolo_headtail_model_combo()

    def _update_obb_mode_warning(self) -> None:
        """Show a performance hint when device/mode is a suboptimal combination."""
        if hasattr(self, "_session_orch"):
            self._session_orch._update_obb_mode_warning()

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
            ht_type = getattr(self._identity_panel, "combo_yolo_headtail_model_type", None)
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
                refresh_callback=self._identity_panel._refresh_yolo_headtail_model_combo,
                selection_callback=self._set_yolo_headtail_model_selection,
                task_family="classify",
                usage_role="headtail",
                dialog_title="Add Head-Tail Classifier",
                repository_dir=repo_dir,
            )
            return
        if hasattr(self, "_detection_panel"):
            self._detection_panel._on_yolo_mode_changed(index)
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
        self._tracking_orch.start_full()

    def _request_qthread_stop(
        self,
        worker,
        worker_name: str,
        *,
        timeout_ms: int = 1500,
        force_terminate: bool = True,
    ) -> None:
        """Stop a QThread cooperatively, then force terminate if needed."""
        self._tracking_orch._request_qthread_stop(
                    worker, worker_name, timeout_ms=timeout_ms, force_terminate=force_terminate
                )

    def _stop_csv_writer(self, timeout_sec: float = 2.0) -> None:
        """Stop background CSV writer thread safely without indefinite blocking."""
        self._tracking_orch._stop_csv_writer(timeout_sec)

    def _cleanup_thread_reference(self, attr_name: str) -> None:
        """Delete finished QThread references safely."""
        self._tracking_orch._cleanup_thread_reference(attr_name)

    def stop_tracking(self: object) -> object:
        """stop_tracking method documentation."""
        self._tracking_orch.stop_tracking()

    def _set_ui_controls_enabled(self, enabled: bool):
        self._session_orch._set_ui_controls_enabled(enabled)

    def _collect_preview_controls(self):
        return self._session_orch._collect_preview_controls()

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        self._session_orch._set_interactive_widgets_enabled(
            enabled, allowlist=allowlist, blocklist=blocklist, remember_state=remember_state
        )

    def _set_video_interaction_enabled(self, enabled: bool):
        self._session_orch._set_video_interaction_enabled(enabled)

    def _prepare_tracking_display(self):
        self._session_orch._prepare_tracking_display()

    def _show_video_logo_placeholder(self):
        """Show HYDRA logo in the video panel when no video is loaded."""
        if hasattr(self, "_session_orch"):
            self._session_orch._show_video_logo_placeholder()
            return
        # Fallback during early init before orchestrators exist
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("HYDRA\n\nLoad a video to begin...")

    def _is_visualization_enabled(self) -> bool:
        # Preview should always render frames regardless of visualization-free toggle
        return (
            not self._setup_panel.chk_visualization_free.isChecked()
            or self.btn_preview.isChecked()
        )

    def _sync_contextual_controls(self):
        self._session_orch._sync_contextual_controls()

    def _apply_ui_state(self, state: str):
        self._session_orch._apply_ui_state(state)

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
        self._tracking_orch.on_progress_update(percentage, status_text)

    def on_pose_exported_model_resolved(self, artifact_path: str) -> None:
        """Update pose exported-model UI/config when runtime resolves an artifact path."""
        self._tracking_orch.on_pose_exported_model_resolved(artifact_path)

    def on_tracking_warning(self: object, title: object, message: object) -> object:
        """Display tracking warnings in the UI."""
        self._tracking_orch.on_tracking_warning(title, message)

    def show_gpu_info(self: object) -> object:
        """Display GPU and acceleration information dialog."""
        self._tracking_orch.show_gpu_info()

    def on_stats_update(self: object, stats: object) -> object:
        """Update real-time tracking statistics."""
        self._tracking_orch.on_stats_update(stats)

    def on_new_frame(self: object, rgb: object) -> object:
        """on_new_frame method documentation."""
        self._tracking_orch.on_new_frame(rgb)

    def _scale_trajectories_to_original_space(self, trajectories_df, resize_factor):
        """Scale trajectory coordinates from resized space back to original video space."""
        return self._tracking_orch._scale_trajectories_to_original_space(trajectories_df, resize_factor)

    def save_trajectories_to_csv(
        self: object, trajectories: object, output_path: object
    ) -> object:
        """Save processed trajectories to CSV.

        Args:
            trajectories: Either list of tuples (old format) or pandas DataFrame (new format with confidence)
            output_path: Path to save CSV file
        """
        self._tracking_orch.save_trajectories_to_csv(trajectories, output_path)

    def merge_and_save_trajectories(self: object) -> object:
        """merge_and_save_trajectories method documentation."""
        self._tracking_orch.merge_and_save_trajectories()

    def on_merge_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during merge."""
        self._tracking_orch.on_merge_progress(value, message)

    def _store_interpolated_pose_result(self, pose_csv_path, pose_rows):
        """Store interpolated pose results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_pose_result(pose_csv_path, pose_rows)

    def _store_interpolated_tag_result(self, tag_csv_path, tag_rows):
        """Store interpolated AprilTag results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_tag_result(tag_csv_path, tag_rows)

    def _store_interpolated_cnn_result(self, cnn_csv_paths, cnn_rows):
        """Store interpolated CNN identity results from CSV paths or in-memory rows."""
        self._tracking_orch._store_interpolated_cnn_result(cnn_csv_paths, cnn_rows)

    def _store_interpolated_headtail_result(self, headtail_csv_path, headtail_rows):
        """Store interpolated head-tail results from CSV path or in-memory rows."""
        self._tracking_orch._store_interpolated_headtail_result(headtail_csv_path, headtail_rows)

    def _on_interpolated_crops_finished(self, result):
        self._tracking_orch._on_interpolated_crops_finished(result)

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
        return self._tracking_orch._start_pending_oriented_track_video_export(final_csv_path)

    def _on_oriented_track_video_worker_thread_finished(self):
        """Release completed oriented track video worker safely."""
        self._tracking_orch._on_oriented_track_video_worker_thread_finished()

    def _on_oriented_track_videos_finished(self, result):
        """Handle completion of oriented track video export."""
        self._tracking_orch._on_oriented_track_videos_finished(result)

    def _on_oriented_track_videos_error(self, error_message):
        """Handle oriented track video export errors without aborting the session."""
        self._tracking_orch._on_oriented_track_videos_error(error_message)

    def on_merge_error(self: object, error_message: object) -> object:
        """Handle merge errors."""
        self._tracking_orch.on_merge_error(error_message)

    def on_merge_finished(self: object, resolved_trajectories: object) -> object:
        """Handle completion of trajectory merging."""
        self._tracking_orch.on_merge_finished(resolved_trajectories)

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
        self._tracking_orch._generate_video_from_trajectories(trajectories_df, csv_path, finalize_on_complete)

    def on_tracking_finished(
        self: object, finished_normally: object, fps_list: object, full_traj: object
    ) -> object:
        """on_tracking_finished method documentation."""
        self._tracking_orch.on_tracking_finished(finished_normally, fps_list, full_traj)

    def _is_pose_export_enabled(self) -> bool:
        """Return True when pose extraction export should be produced."""
        if not hasattr(self, '_detection_panel'):
            return False
        return self._detection_panel._is_yolo_detection_mode() and bool(
            hasattr(self, '_identity_panel')
            and self._identity_panel.chk_enable_pose_extractor.isChecked()
        )

    def _build_pose_augmented_dataframe(self, final_csv_path):
        """Load final CSV and merge available cached/interpolated pose columns."""
        return self._tracking_orch._build_pose_augmented_dataframe(final_csv_path)

    def _export_pose_augmented_csv(self, final_csv_path):
        """Write a pose-augmented trajectories CSV next to the final CSV."""
        self._tracking_orch._export_pose_augmented_csv(final_csv_path)

    def _relink_final_pose_augmented_csv(self, final_csv_path):
        """Rewrite final CSV IDs after pose-aware relinking and regenerate _with_pose.csv."""
        self._tracking_orch._relink_final_pose_augmented_csv(final_csv_path)

    def _load_video_trajectories(self, final_csv_path):
        """Load best available trajectories for video generation (prefers pose-augmented CSV)."""
        return self._tracking_orch._load_video_trajectories(final_csv_path)

    def _run_pending_video_generation_or_finalize(self):
        """Run video generation if queued; otherwise finalize UI/session cleanup."""
        self._tracking_orch._run_pending_video_generation_or_finalize()

    def _finish_tracking_session(self, final_csv_path=None):
        """Complete tracking session cleanup and UI updates."""
        self._tracking_orch._finish_tracking_session(final_csv_path=final_csv_path)

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._tracking_orch._finalize_tracking_session_ui()

    def _generate_interpolated_individual_crops(self, csv_path):
        """Post-pass interpolation for occluded segments in individual dataset."""
        self._tracking_orch._generate_interpolated_individual_crops(csv_path)

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
        self._tracking_orch.start_backward_tracking()

    def start_tracking(
        self: object, preview_mode: bool, backward_mode: bool = False
    ) -> object:
        """start_tracking method documentation."""
        self._tracking_orch.start_tracking(preview_mode=preview_mode, backward_mode=backward_mode)

    def start_preview_on_video(self: object, video_path: object) -> object:
        """start_preview_on_video method documentation."""
        self._tracking_orch.start_preview_on_video(video_path)

    def start_tracking_on_video(
        self: object, video_path: object, backward_mode: object = False
    ) -> object:
        """start_tracking_on_video method documentation."""
        self._tracking_orch.start_tracking_on_video(video_path, backward_mode=backward_mode)

    def get_parameters_dict(self: object) -> object:
        """get_parameters_dict method documentation."""
        return self._config_orch.get_parameters_dict()

    def load_config(self: object) -> object:
        """Manually load config from file dialog."""
        self._config_orch.load_config()

    def _load_config_from_file(self, config_path, preset_mode=False):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.

        Args:
            config_path: Path to the config/preset file
            preset_mode: If True, skip loading video paths and ROI data (for organism presets)
        """
        self._config_orch._load_config_from_file(config_path, preset_mode=preset_mode)

    def _atomic_json_write(cfg, path):
        """Write a JSON config atomically. Returns (success, error_message)."""
        self._config_orch._atomic_json_write(cfg, path)

    def _resolve_config_save_path(self, prompt_if_exists):
        """Determine the config file save path, prompting the user if needed."""
        return self._config_orch._resolve_config_save_path(prompt_if_exists)

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
        self._config_orch.save_config(prompt_if_exists=prompt_if_exists)

    def _setup_session_logging(self, video_path, backward_mode=False):
        self._session_orch._setup_session_logging(video_path, backward_mode=backward_mode)

    def _cleanup_session_logging(self):
        self._session_orch._cleanup_session_logging()

    def _generate_training_dataset(self, override_csv_path=None):
        """Generate training dataset from tracking results for active learning."""
        self._tracking_orch._generate_training_dataset(override_csv_path=override_csv_path)

    def on_dataset_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during dataset generation."""
        self._tracking_orch.on_dataset_progress(value, message)

    def on_dataset_finished(
        self: object, dataset_dir: object, num_frames: object
    ) -> object:
        """Handle dataset generation completion."""
        self._tracking_orch.on_dataset_finished(dataset_dir, num_frames)

    def on_dataset_error(self: object, error_message: object) -> object:
        """Handle dataset generation errors."""
        self._tracking_orch.on_dataset_error(error_message)

    def _show_session_summary(self):
        """Show a single end-of-session summary dialog listing completed processes."""
        self._tracking_orch._show_session_summary()

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
        self._tracking_orch._on_dataset_worker_thread_finished()

    def _is_worker_running(self, worker):
        return self._session_orch._is_worker_running(worker)

    def _has_active_progress_task(self) -> bool:
        return self._session_orch._has_active_progress_task()

    def _refresh_progress_visibility(self):
        self._session_orch._refresh_progress_visibility()

    def _cleanup_temporary_files(self):
        self._session_orch._cleanup_temporary_files()

    def _disable_spinbox_wheel_events(self):
        self._session_orch._disable_spinbox_wheel_events()

    def _connect_parameter_signals(self):
        self._session_orch._connect_parameter_signals()

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
        self._config_orch._invalidate_roi_cache()

    def _get_presets_dir(self):
        """Get the presets directory path."""
        from hydra_suite.paths import get_presets_dir

        return str(get_presets_dir())

    def _populate_preset_combo(self):
        """Populate the preset combo box by auto-scanning configs folder."""
        if hasattr(self, "_config_orch"):
            self._config_orch._populate_preset_combo()
            return
        # Fallback during early init before orchestrators exist
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
        self._config_orch._load_selected_preset()

    def _save_custom_preset(self):
        """Save current settings as custom preset with user-defined name and description."""
        self._config_orch._save_custom_preset()

    def _load_default_preset_on_startup(self):
        """Load default preset on application startup."""
        self._config_orch._load_default_preset_on_startup()

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        if hasattr(self, "_config_orch"):
            return self._config_orch._load_advanced_config()
        # Fallback during __init__ before orchestrators exist
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
        self._config_orch._save_advanced_config()

    def _calculate_roi_bounding_box(self, padding=None):
        """Calculate the bounding box of the current ROI mask with optional padding.

        Args:
            padding: Fraction of min(width, height) to add as padding (e.g., 0.05 = 5%).
                    If None, uses value from advanced config.

        Returns:
            Tuple (x, y, w, h) or None if no ROI
        """
        return self._config_orch._calculate_roi_bounding_box(padding=padding)

    def _estimate_roi_efficiency(self):
        """Estimate the efficiency gain from cropping to ROI.

        Returns:
            tuple: (roi_coverage_percent, potential_speedup_factor) or (None, None)
        """
        return self._config_orch._estimate_roi_efficiency()

    def _update_roi_optimization_info(self):
        """Update the ROI optimization label with efficiency information."""
        self._config_orch._update_roi_optimization_info()

    def crop_video_to_roi(self: object) -> object:
        """Crop the video to the ROI bounding box and save as new file."""
        self._config_orch.crop_video_to_roi()

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
