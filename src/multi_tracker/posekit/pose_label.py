#!/usr/bin/env python3
"""
PoseKit Labeler: an all-in-one PySide6 labeler for Ultralytics YOLO Pose.

- Start with only images: launches a Setup Wizard to create a project at runtime.
- If a project exists: loads it automatically.
- Project Settings: edit classes/keypoints later, with optional label migration.
- Writes Ultralytics YOLO Pose labels (.txt) with normalized bbox + keypoints + visibility.

Label line format:
<class> <cx> <cy> <w> <h> <x1> <y1> <v1> ... <xK> <yK> <vK>

Visibility convention:
0 = missing / not labeled
1 = labeled but occluded
2 = labeled and visible

Hotkeys:
- A / D: prev / next frame
- Q / E: prev / next keypoint
- Space: advance (frame-mode -> next keypoint; keypoint-mode -> next frame)
- Ctrl+S: save
- V / O / N: click mode Visible / Occluded / Missing
- Delete: clear current keypoint
- Ctrl+F: next unlabeled
- Ctrl+G: skeleton editor
- Ctrl+P: project settings
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import subprocess
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import csv
import gc

logger = logging.getLogger("pose_label")

import cv2
import numpy as np
import yaml

# Handle both package imports and direct script execution
try:
    from .pose_label_extensions import (
        FrameMetadata,
        MetadataManager,
        CrashSafeWriter,
        LabelVersioning,
        cluster_stratified_split,
        cluster_kfold_split,
        save_split_files,
        IncrementalEmbeddingCache,
        Keypoint,
        load_yolo_pose_label,
        save_yolo_pose_label,
        migrate_labels_keypoints,
        build_yolo_pose_dataset,
    )
    from .pose_inference import PoseInferenceService

    from .pose_label_dialogs import (
        DatasetSplitDialog,
        FrameMetadataDialog,
        SmartSelectDialog,
        TrainingRunnerDialog,
        EvaluationDashboardDialog,
        ActiveLearningDialog,
        get_available_devices,
    )
except ImportError:
    # Direct script execution - use absolute imports
    from pose_label_extensions import (
        FrameMetadata,
        MetadataManager,
        CrashSafeWriter,
        LabelVersioning,
        cluster_stratified_split,
        cluster_kfold_split,
        save_split_files,
        IncrementalEmbeddingCache,
        Keypoint,
        load_yolo_pose_label,
        save_yolo_pose_label,
        migrate_labels_keypoints,
        build_yolo_pose_dataset,
    )
    from pose_inference import PoseInferenceService

    from pose_label_dialogs import (
        DatasetSplitDialog,
        FrameMetadataDialog,
        SmartSelectDialog,
        TrainingRunnerDialog,
        EvaluationDashboardDialog,
        ActiveLearningDialog,
        get_available_devices,
    )

from PySide6.QtCore import Qt, QRect, QRectF, QSize, QObject, Signal, Slot, QThread, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPalette,
    QPainter,
    QPen,
    QPixmap,
    QBrush,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QPlainTextEdit,
    QScrollArea,
    QFrame,
    QGroupBox,
    QInputDialog,
    QProgressBar,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROJECT_NAME = "pose_project.json"
DEFAULT_SKELETON_DIRNAME = "configs"
DEFAULT_KPT_RADIUS = 5.0
DEFAULT_LABEL_FONT_SIZE = 10
DEFAULT_AUTOSAVE_DELAY_MS = 3000


# -----------------------------
# Data model
# -----------------------------
@dataclass
class Keypoint:
    """Single keypoint annotation in pixel space."""

    x: float = 0.0  # pixel coords
    y: float = 0.0
    v: int = 0  # 0 missing, 1 occluded, 2 visible


@dataclass
class FrameAnn:
    """Single-frame annotation containing class, box, and keypoints."""

    cls: int
    bbox_xyxy: Optional[Tuple[float, float, float, float]]  # pixel coords
    kpts: List[Keypoint]


@dataclass
class Project:
    """Persistent PoseKit project configuration and UI/session state."""

    images_dir: Path
    out_root: Path
    labels_dir: Path
    project_path: Path

    class_names: List[str]
    keypoint_names: List[str]
    skeleton_edges: List[Tuple[int, int]]  # 0-based indices

    dims: int = 3
    bbox_pad_frac: float = 0.03
    autosave: bool = True
    last_index: int = 0
    labeling_frames: List[int] = field(default_factory=list)

    # Display enhancement (CLAHE + unsharp mask)
    enhance_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    sharpen_amt: float = 1.2
    blur_sigma: float = 1.0

    # Display sizing
    kpt_radius: float = DEFAULT_KPT_RADIUS
    label_font_size: int = DEFAULT_LABEL_FONT_SIZE

    # Opacity
    kpt_opacity: float = 1.0
    edge_opacity: float = 0.7
    latest_pose_weights: Optional[Path] = None
    latest_sleap_dataset: Optional[Path] = None

    def to_json(self) -> dict:
        """Serialize project state to a JSON-compatible dictionary."""
        base = self.project_path.parent
        images_dir = _relativize_path(self.images_dir, base)
        labels_dir = _relativize_path(self.labels_dir, base)
        out_root = _relativize_path(self.out_root, base)
        latest_weights = (
            _relativize_path(self.latest_pose_weights, base)
            if self.latest_pose_weights
            else None
        )
        latest_sleap_dataset = (
            _relativize_path(self.latest_sleap_dataset, base)
            if self.latest_sleap_dataset
            else None
        )
        paths_relative = (
            not Path(images_dir).is_absolute()
            and not Path(labels_dir).is_absolute()
            and not Path(out_root).is_absolute()
        )
        return {
            "images_dir": str(images_dir),
            "out_root": str(out_root),
            "labels_dir": str(labels_dir),
            "latest_pose_weights": str(latest_weights) if latest_weights else "",
            "latest_sleap_dataset": str(latest_sleap_dataset)
            if latest_sleap_dataset
            else "",
            "paths_relative": bool(paths_relative),
            "class_names": self.class_names,
            "keypoint_names": self.keypoint_names,
            "skeleton_edges": [[a, b] for a, b in self.skeleton_edges],
            "dims": self.dims,
            "bbox_pad_frac": self.bbox_pad_frac,
            "autosave": self.autosave,
            "last_index": self.last_index,
            "labeling_frames": sorted({int(i) for i in self.labeling_frames}),
            "enhance_enabled": self.enhance_enabled,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": list(self.clahe_grid),
            "sharpen_amt": self.sharpen_amt,
            "blur_sigma": self.blur_sigma,
            "kpt_radius": self.kpt_radius,
            "label_font_size": self.label_font_size,
            "kpt_opacity": self.kpt_opacity,
            "edge_opacity": self.edge_opacity,
        }

    @staticmethod
    def from_json(project_path: Path) -> "Project":
        """Load project configuration from disk."""
        data = json.loads(project_path.read_text(encoding="utf-8"))
        grid = data.get("clahe_grid", [8, 8])
        if not isinstance(grid, (list, tuple)) or len(grid) != 2:
            grid = [8, 8]
        base = project_path.parent
        images_dir = _resolve_project_path(Path(data["images_dir"]), base, data)
        labels_dir = _resolve_project_path(Path(data["labels_dir"]), base, data)
        out_root_raw = Path(data.get("out_root", labels_dir.parent))
        out_root = _resolve_project_path(out_root_raw, base, data)
        latest_raw = Path(data.get("latest_pose_weights", "") or "")
        latest_pose_weights = (
            _resolve_project_path(latest_raw, base, data) if str(latest_raw) else None
        )
        if latest_pose_weights and latest_pose_weights.suffix != ".pt":
            latest_pose_weights = None
        sleap_raw = Path(data.get("latest_sleap_dataset", "") or "")
        latest_sleap_dataset = (
            _resolve_project_path(sleap_raw, base, data) if str(sleap_raw) else None
        )
        if latest_sleap_dataset and latest_sleap_dataset.suffix != ".slp":
            latest_sleap_dataset = None
        return Project(
            images_dir=images_dir,
            out_root=out_root,
            labels_dir=labels_dir,
            project_path=project_path,
            class_names=list(data.get("class_names", ["object"])),
            keypoint_names=list(data.get("keypoint_names", ["kp1", "kp2"])),
            skeleton_edges=[
                (int(a), int(b)) for a, b in data.get("skeleton_edges", [])
            ],
            dims=int(data.get("dims", 3)),
            bbox_pad_frac=float(data.get("bbox_pad_frac", 0.03)),
            autosave=bool(data.get("autosave", True)),
            last_index=int(data.get("last_index", 0)),
            labeling_frames=[int(i) for i in data.get("labeling_frames", [])],
            enhance_enabled=bool(data.get("enhance_enabled", False)),
            clahe_clip=float(data.get("clahe_clip", 2.0)),
            clahe_grid=(int(grid[0]), int(grid[1])),
            sharpen_amt=float(data.get("sharpen_amt", 1.2)),
            blur_sigma=float(data.get("blur_sigma", 1.0)),
            kpt_radius=float(data.get("kpt_radius", DEFAULT_KPT_RADIUS)),
            label_font_size=int(data.get("label_font_size", DEFAULT_LABEL_FONT_SIZE)),
            kpt_opacity=float(data.get("kpt_opacity", 1.0)),
            edge_opacity=float(data.get("edge_opacity", 0.7)),
            latest_pose_weights=latest_pose_weights,
            latest_sleap_dataset=latest_sleap_dataset,
        )


class PosePredictWorker(QObject):
    """Background worker for one-image pose prediction."""

    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        model_path: Path,
        image_path: Path,
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
        backend: str = "yolo",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.image_path = Path(image_path)
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        self.num_kpts = len(self.keypoint_names)
        self.backend = (backend or "yolo").lower()
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        # Enforce single-instance predictions for PoseKit.
        self.sleap_max_instances = 1

    def _extract_best_prediction(
        self, result: Any
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract the highest-confidence keypoint set from a model result."""
        if result is None or result.keypoints is None:
            return None, None
        kpts = result.keypoints
        try:
            xy = kpts.xy
            conf = getattr(kpts, "conf", None)
            xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.array(xy)
            if conf is not None:
                conf = conf.cpu().numpy() if hasattr(conf, "cpu") else np.array(conf)
        except Exception:
            return None, None

        if xy.ndim == 2:
            xy = xy[None, :, :]
        if conf is not None and conf.ndim == 1:
            conf = conf[None, :]

        if xy.size == 0:
            return None, None

        if conf is not None:
            scores = np.nanmean(conf, axis=1)
        else:
            scores = None
            try:
                if result.boxes is not None and hasattr(result.boxes, "conf"):
                    scores = result.boxes.conf.cpu().numpy()
            except Exception:
                scores = None
            if scores is None:
                scores = np.zeros((xy.shape[0],), dtype=np.float32)

        best = int(np.argmax(scores)) if len(scores) > 0 else 0
        pred_xy = xy[best]
        pred_conf = conf[best] if conf is not None else np.zeros((self.num_kpts,))

        if pred_xy.shape[0] != self.num_kpts:
            num = min(pred_xy.shape[0], self.num_kpts)
            tmp_xy = np.zeros((self.num_kpts, 2), dtype=np.float32)
            tmp_conf = np.zeros((self.num_kpts,), dtype=np.float32)
            tmp_xy[:num] = pred_xy[:num]
            tmp_conf[:num] = pred_conf[:num]
            pred_xy = tmp_xy
            pred_conf = tmp_conf

        return pred_xy, pred_conf

    def run(self) -> None:
        """Run inference and emit either predicted keypoints or an error."""
        try:
            infer = PoseInferenceService(
                self.out_root, self.keypoint_names, self.skeleton_edges
            )
            cached = infer.get_cached_pred(
                self.model_path, self.image_path, backend=self.backend
            )
            if cached is not None:
                self.finished.emit(cached)
                return
            preds_map, err = infer.predict(
                self.model_path,
                [self.image_path],
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                batch=1,
                progress_cb=None,
                cancel_cb=None,
                backend=self.backend,
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
            )
            if preds_map is None:
                self.failed.emit(err or "Prediction failed.")
                return
            preds = preds_map.get(str(self.image_path)) or preds_map.get(
                str(self.image_path.resolve())
            )
            if preds is None:
                preds = [(0.0, 0.0, 0.0) for _ in range(len(self.keypoint_names))]
            self.finished.emit(preds)
        except Exception as e:
            _maybe_empty_cuda_cache()
            self.failed.emit(str(e))


class BulkPosePredictWorker(QObject):
    """Background worker for multi-image pose prediction."""

    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        model_path: Path,
        image_paths: List[Path],
        out_root: Path,
        keypoint_names: List[str],
        skeleton_edges: Optional[List[Tuple[int, int]]] = None,
        backend: str = "yolo",
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.25,
        batch: int = 16,
        sleap_env: Optional[str] = None,
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
    ):
        super().__init__()
        self.model_path = Path(model_path)
        self.image_paths = list(image_paths)
        self.out_root = Path(out_root)
        self.keypoint_names = list(keypoint_names)
        self.skeleton_edges = list(skeleton_edges or [])
        self.backend = (backend or "yolo").lower()
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.batch = int(batch)
        self.sleap_env = sleap_env
        self.sleap_device = sleap_device
        self.sleap_batch = int(sleap_batch)
        # Enforce single-instance predictions for PoseKit.
        self.sleap_max_instances = 1
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation for the running prediction batch."""
        self._cancel = True

    def run(self) -> None:
        """Run batch inference and stream progress updates."""
        try:
            infer = PoseInferenceService(
                self.out_root, self.keypoint_names, self.skeleton_edges
            )
            preds, err = infer.predict(
                self.model_path,
                self.image_paths,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.conf,
                batch=self.batch,
                progress_cb=lambda d, t: self.progress.emit(d, t),
                cancel_cb=lambda: self._cancel,
                backend=self.backend,
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
            )
            if preds is None:
                self.failed.emit(err or "Prediction failed.")
                return
            self.finished.emit(preds)
        except Exception as e:
            _maybe_empty_cuda_cache()
            self.failed.emit(str(e))


class SleapServiceWorker(QObject):
    """Worker that starts and validates the SLEAP backend service."""

    finished = Signal(bool, str, str)

    def __init__(self, env_name: str, out_root: Path):
        super().__init__()
        self.env_name = env_name
        self.out_root = Path(out_root)

    def run(self) -> None:
        """Start SLEAP service and emit status tuple."""
        try:
            ok, err, log_path = PoseInferenceService.start_sleap_service(
                self.env_name, self.out_root
            )
            self.finished.emit(bool(ok), str(err or ""), str(log_path or ""))
        except Exception as e:
            self.finished.emit(False, str(e), "")


# -----------------------------
# Utils
# -----------------------------
def _relativize_path(path: Path, base: Path) -> Path:
    try:
        return Path(os.path.relpath(path, base))
    except Exception:
        return path


def _resolve_project_path(path: Path, base: Path, data: dict) -> Path:
    if path.is_absolute():
        return path
    if bool(data.get("paths_relative", False)):
        return (base / path).resolve()
    return (base / path).resolve()


def list_images(images_dir: Path) -> List[Path]:
    """Recursively list supported image files in sorted order."""
    paths: List[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def enhance_for_pose(
    img_bgr: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    sharpen_amt: float = 1.2,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Apply CLAHE and unsharp masking to improve pose keypoint visibility."""
    # 1) CLAHE on L channel (LAB)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 2) Unsharp mask (mild sharpening)
    blur = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    sharp = cv2.addWeighted(img, 1.0 + sharpen_amt, blur, -sharpen_amt, 0)

    return sharp


def get_default_skeleton_dir() -> Optional[Path]:
    """Return the repository-level skeleton config directory if available."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3] if len(here.parents) >= 4 else here.parent
    cfg = repo_root / DEFAULT_SKELETON_DIRNAME
    if cfg.exists() and cfg.is_dir():
        return cfg
    return None


def get_keypoint_palette() -> List[QColor]:
    """Return a stable high-contrast color palette for keypoint overlays."""
    return [
        QColor(255, 99, 71),
        QColor(30, 144, 255),
        QColor(60, 179, 113),
        QColor(238, 130, 238),
        QColor(255, 165, 0),
        QColor(0, 206, 209),
        QColor(255, 215, 0),
        QColor(199, 21, 133),
        QColor(127, 255, 0),
        QColor(70, 130, 180),
        QColor(255, 105, 180),
        QColor(64, 224, 208),
    ]


def get_ui_settings_path() -> Path:
    """Get path to UI settings file in user's home directory."""
    config_dir = Path.home() / "posekit"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "ui_settings.json"


def load_ui_settings() -> Dict[str, Any]:
    """Load persistent UI settings."""
    path = get_ui_settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ui_settings(settings: Dict[str, Any]) -> None:
    """Save persistent UI settings."""
    path = get_ui_settings_path()
    try:
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        pass


def _clamp01(value: float) -> float:
    """Clamp value to [0, 1] range with NaN protection."""
    if not np.isfinite(value):
        logger.warning(f"Non-finite value clamped to 0: {value}")
        return 0.0
    return max(0.0, min(1.0, value))


def _xyxy_to_cxcywh(
    x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """Convert bounding box from (x1, y1, x2, y2) to (center_x, center_y, width, height)."""
    # Validate inputs
    if not all(np.isfinite(v) for v in (x1, y1, x2, y2)):
        logger.error(f"Non-finite bbox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return (0.5, 0.5, 0.1, 0.1)  # Return safe default

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1

    # Ensure positive dimensions
    bw = max(bw, 0.001)
    bh = max(bh, 0.001)

    return cx, cy, bw, bh


def compute_bbox_from_kpts(
    kpts: List[Keypoint], pad_frac: float, w: int, h: int
) -> Optional[Tuple[float, float, float, float]]:
    """Compute bounding box from visible keypoints with minimum size guarantee."""
    MIN_BBOX_SIZE = 2.0  # Minimum bbox size in pixels

    pts = [(kp.x, kp.y) for kp in kpts if kp.v > 0]
    if not pts:
        return None

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Check for valid coordinates
    if not all(np.isfinite(x) for x in xs) or not all(np.isfinite(y) for y in ys):
        logger.error(f"Non-finite keypoint coordinates detected")
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # Add padding
    pad = pad_frac * max((x2 - x1), (y2 - y1), 1.0)
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(w - 1), x2 + pad)
    y2 = min(float(h - 1), y2 + pad)

    # Ensure minimum bbox size to prevent numerical issues
    if x2 - x1 < MIN_BBOX_SIZE:
        center_x = (x1 + x2) / 2.0
        x1 = max(0.0, center_x - MIN_BBOX_SIZE / 2.0)
        x2 = min(float(w - 1), center_x + MIN_BBOX_SIZE / 2.0)

    if y2 - y1 < MIN_BBOX_SIZE:
        center_y = (y1 + y2) / 2.0
        y1 = max(0.0, center_y - MIN_BBOX_SIZE / 2.0)
        y2 = min(float(h - 1), center_y + MIN_BBOX_SIZE / 2.0)

    # Final validation
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        logger.warning(
            f"Computed bbox too small: ({x1:.1f}, {y1:.1f}) to ({x2:.1f}, {y2:.1f})"
        )
        return None

    return (x1, y1, x2, y2)


# ------------------------------------
# Helpers for Embedding and Clustering
# ------------------------------------


def _stable_hash_dict(d: dict) -> str:
    s = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _choose_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _is_cuda_device(device: str) -> bool:
    d = (device or "").strip().lower()
    if d in {"cuda", "gpu"}:
        return True
    if d.startswith("cuda:"):
        return True
    return d.isdigit()


def _maybe_limit_cuda_memory(fraction: float = 0.9):
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(fraction))
    except Exception:
        pass


def _maybe_empty_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_image_pil(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGB")


def _maybe_downscale_pil(img, max_side: int):
    if max_side <= 0:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return img.resize((nw, nh), resample=2)  # BILINEAR


def _enhance_pil_for_pose(img_pil):
    # enhance_for_pose expects BGR np array
    arr = np.array(img_pil)  # RGB
    bgr = arr[:, :, ::-1].copy()
    bgr2 = enhance_for_pose(bgr)
    rgb2 = bgr2[:, :, ::-1].copy()
    from PIL import Image

    return Image.fromarray(rgb2)


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # assumes A, B are L2-normalized
    return A @ B.T


def _farthest_point_centers(emb: np.ndarray, k: int, seed: int = 0) -> List[int]:
    # emb: (n,d) L2-normalized
    n = emb.shape[0]
    if k <= 0:
        return []
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centers = [first]
    # maintain best similarity to any chosen center (higher is closer)
    best_sim = _cosine_sim_matrix(emb, emb[[first]]).reshape(-1)
    for _ in range(1, min(k, n)):
        # farthest => smallest best_sim
        idx = int(np.argmin(best_sim))
        centers.append(idx)
        sim_to_new = _cosine_sim_matrix(emb, emb[[idx]]).reshape(-1)
        best_sim = np.maximum(best_sim, sim_to_new)
    return centers


def cluster_embeddings_cosine(
    emb: np.ndarray,
    k: int,
    method: str = "hierarchical",
    seed: int = 0,
    hierarchical_limit: int = 2500,
) -> np.ndarray:
    """
    Returns cluster labels in [0..k-1].
    """
    n = emb.shape[0]
    k = max(1, min(int(k), n))

    if method == "hierarchical" and n <= hierarchical_limit:
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, fcluster

        # condensed cosine distances
        d = pdist(emb, metric="cosine")
        Z = linkage(d, method="average")  # average linkage works well with cosine
        lab = fcluster(Z, t=k, criterion="maxclust")
        return (lab - 1).astype(np.int32)

    # fallback: farthest-point centers + nearest-center assignment
    centers = _farthest_point_centers(emb, k=k, seed=seed)
    C = emb[centers]
    sims = _cosine_sim_matrix(emb, C)  # (n,k)
    return np.argmax(sims, axis=1).astype(np.int32)


def pick_frames_stratified(
    emb: np.ndarray,
    cluster_id: np.ndarray,
    want_n: int,
    eligible_indices: List[int],
    min_per_cluster: int = 1,
    seed: int = 0,
    strategy: str = "centroid_then_diverse",  # or "centroid"
) -> List[int]:
    """
    emb: embeddings for eligible_indices (same order)
    cluster_id: cluster labels for eligible_indices
    returns: selected *global* indices (from eligible_indices mapping)
    """
    rng = np.random.default_rng(seed)
    want_n = max(0, min(int(want_n), len(eligible_indices)))
    if want_n == 0:
        return []

    # cluster -> local positions
    clusters: Dict[int, List[int]] = {}
    for pos, cid in enumerate(cluster_id):
        clusters.setdefault(int(cid), []).append(pos)

    # order clusters by size (largest first)
    cluster_keys = sorted(clusters.keys(), key=lambda c: len(clusters[c]), reverse=True)

    # If more clusters than want_n, keep only top want_n clusters (still diverse-ish)
    if len(cluster_keys) > want_n:
        cluster_keys = cluster_keys[:want_n]

    # initial quotas
    quotas = {c: 0 for c in cluster_keys}
    remaining = want_n

    # give each cluster min_per_cluster (as possible)
    for c in cluster_keys:
        q = min(min_per_cluster, len(clusters[c]), remaining)
        quotas[c] += q
        remaining -= q
        if remaining <= 0:
            break

    if remaining > 0:
        # distribute remaining proportional to cluster size
        sizes = np.array([len(clusters[c]) for c in cluster_keys], dtype=np.float64)
        weights = sizes / (sizes.sum() + 1e-12)
        extra = rng.multinomial(remaining, weights)
        for c, add in zip(cluster_keys, extra):
            quotas[c] += int(min(add, len(clusters[c]) - quotas[c]))

        # if rounding left some slack, fill greedily
        used = sum(quotas.values())
        slack = want_n - used
        if slack > 0:
            for c in cluster_keys:
                avail = len(clusters[c]) - quotas[c]
                take = min(avail, slack)
                quotas[c] += take
                slack -= take
                if slack <= 0:
                    break

    selected_local: List[int] = []

    for c in cluster_keys:
        local_positions = clusters[c]
        if not local_positions:
            continue
        q = quotas[c]
        if q <= 0:
            continue

        X = emb[local_positions]  # (m,d) L2-normalized

        # 1) representative: closest to centroid (max cosine similarity to centroid)
        centroid = X.mean(axis=0, keepdims=True)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = (X @ centroid.T).reshape(-1)
        rep_idx = int(np.argmax(sims))
        chosen = [local_positions[rep_idx]]

        if q > 1:
            if strategy == "centroid":
                # pick next-best by centroid similarity (less diverse)
                order = np.argsort(-sims)
                for oi in order:
                    if len(chosen) >= q:
                        break
                    cand = local_positions[int(oi)]
                    if cand not in chosen:
                        chosen.append(cand)
            else:
                # centroid_then_diverse: greedy farthest within the cluster
                # track best similarity to chosen
                chosen_X = emb[[chosen[0]]]
                best_sim = (X @ chosen_X.T).reshape(-1)
                while len(chosen) < q:
                    # farthest => smallest best_sim
                    far = int(np.argmin(best_sim))
                    cand = local_positions[far]
                    if cand in chosen:
                        best_sim[far] = 1.0
                        continue
                    chosen.append(cand)
                    # update best_sim with new chosen point
                    sim_new = (X @ emb[[cand]].T).reshape(-1)
                    best_sim = np.maximum(best_sim, sim_new)

        selected_local.extend(chosen[:q])

    # map local positions back to global frame indices
    out = [eligible_indices[pos] for pos in selected_local]
    # ensure exact want_n (rare over/under due to edge cases)
    out = out[:want_n]
    return out


# -----------------------------
# YOLO pose I/O
# -----------------------------
def load_yolo_pose_label(
    label_path: Path, k: int
) -> Optional[Tuple[int, List[Keypoint], Optional[Tuple[float, float, float, float]]]]:
    """Load a YOLO pose label file and convert entries into Keypoint objects."""
    if not label_path.exists():
        return None
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    line = txt.splitlines()[0].strip()
    parts = line.split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, bw, bh = map(float, parts[1:5])
    rest = parts[5:]

    kpts: List[Keypoint] = []
    if len(rest) == 2 * k:
        for i in range(k):
            x = float(rest[2 * i + 0])
            y = float(rest[2 * i + 1])
            kpts.append(Keypoint(x=x, y=y, v=2))
    elif len(rest) == 3 * k:
        for i in range(k):
            x = float(rest[3 * i + 0])
            y = float(rest[3 * i + 1])
            v = int(float(rest[3 * i + 2]))
            kpts.append(Keypoint(x=x, y=y, v=v))
    else:
        # best-effort parse triples
        n = min(len(rest) // 3, k)
        for i in range(k):
            if i < n:
                x = float(rest[3 * i + 0])
                y = float(rest[3 * i + 1])
                v = int(float(rest[3 * i + 2]))
                kpts.append(Keypoint(x=x, y=y, v=v))
            else:
                kpts.append(Keypoint(0.0, 0.0, 0))

    return cls, kpts, (cx, cy, bw, bh)


def save_yolo_pose_label(
    label_path: Path,
    cls: int,
    img_w: int,
    img_h: int,
    kpts_px: List[Keypoint],
    bbox_xyxy_px: Optional[Tuple[float, float, float, float]],
    pad_frac: float,
    create_backup: bool = True,
) -> None:
    """Write one annotation in YOLO pose format with optional versioned backup."""
    # Create backup of existing label before overwriting
    if create_backup and label_path.exists():
        try:
            versioning = LabelVersioning(label_path.parent)
            versioning.backup_label(label_path)
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")

    if bbox_xyxy_px is None:
        bbox_xyxy_px = compute_bbox_from_kpts(kpts_px, pad_frac, img_w, img_h)

    if bbox_xyxy_px is None:
        label_path.parent.mkdir(parents=True, exist_ok=True)
        CrashSafeWriter.write_label(label_path, "")
        return

    x1, y1, x2, y2 = bbox_xyxy_px
    cx, cy, bw, bh = _xyxy_to_cxcywh(x1, y1, x2, y2)

    cxn = _clamp01(cx / img_w)
    cyn = _clamp01(cy / img_h)
    bwn = _clamp01(bw / img_w)
    bhn = _clamp01(bh / img_h)

    vals = [str(int(cls)), f"{cxn:.6f}", f"{cyn:.6f}", f"{bwn:.6f}", f"{bhn:.6f}"]
    for kp in kpts_px:
        if kp.v == 0:
            vals += ["0.000000", "0.000000", "0"]
        else:
            vals += [
                f"{_clamp01(kp.x / img_w):.6f}",
                f"{_clamp01(kp.y / img_h):.6f}",
                str(int(kp.v)),
            ]

    # Use crash-safe atomic write
    content = " ".join(vals) + "\n"
    CrashSafeWriter.write_label(label_path, content)


# -----------------------------
# Migration when keypoints change
# -----------------------------
def migrate_labels_keypoints(
    labels_dir: Path,
    old_kpt_names: List[str],
    new_kpt_names: List[str],
    mode: str = "name",  # "name" or "index"
) -> Tuple[int, int]:
    """
    Update all .txt labels in labels_dir to match new keypoint list length/order.

    mode="name": map by unique keypoint names, fallback to 0 if missing
    mode="index": preserve positions by old index (truncate/extend)
    Returns: (files_modified, files_total)
    """
    txts = sorted(labels_dir.glob("*.txt"))
    if not txts:
        return (0, 0)

    old_k = len(old_kpt_names)
    new_k = len(new_kpt_names)

    # build mapping new_index -> old_index
    mapping: Dict[int, Optional[int]] = {}
    if mode == "index":
        for ni in range(new_k):
            mapping[ni] = ni if ni < old_k else None
    else:
        # name-based mapping (only safe if names are unique-ish)
        name_to_old = {}
        for i, n in enumerate(old_kpt_names):
            if n not in name_to_old:
                name_to_old[n] = i
        for ni, n in enumerate(new_kpt_names):
            mapping[ni] = name_to_old.get(n, None)

    files_modified = 0
    for lp in txts:
        raw = lp.read_text(encoding="utf-8").strip()
        if not raw:
            continue
        line = raw.splitlines()[0].strip()
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        cx, cy, bw, bh = parts[1:5]
        rest = parts[5:]

        # parse old kpts triples if possible; otherwise skip
        old_trip = []
        if len(rest) >= 3 * old_k:
            for i in range(old_k):
                x = rest[3 * i + 0]
                y = rest[3 * i + 1]
                v = rest[3 * i + 2]
                old_trip.append((x, y, v))
        elif len(rest) >= 2 * old_k:
            # no visibility; assume v=2
            for i in range(old_k):
                x = rest[2 * i + 0]
                y = rest[2 * i + 1]
                old_trip.append((x, y, "2"))
        else:
            # can't safely migrate
            continue

        new_trip = []
        for ni in range(new_k):
            oi = mapping.get(ni, None)
            if oi is None or oi >= len(old_trip):
                new_trip.extend(["0.000000", "0.000000", "0"])
            else:
                x, y, v = old_trip[oi]
                new_trip.extend([x, y, v])

        out_line = " ".join([cls, cx, cy, bw, bh] + new_trip) + "\n"
        if out_line.strip() != raw.strip():
            lp.write_text(out_line, encoding="utf-8")
            files_modified += 1

    return files_modified, len(txts)


# EmbeddingWorker and SmartSelectDialog moved to pose_label_dialogs


# -----------------------------
# Qt graphics canvas
# -----------------------------
class PoseCanvas(QGraphicsView):
    """Interactive graphics canvas for keypoint editing and prediction overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self.kpt_items: List[Optional[QGraphicsEllipseItem]] = []
        self.kpt_labels: List[Optional[QGraphicsTextItem]] = []
        self.edge_items: List[QGraphicsLineItem] = []
        self.pred_items: List[Optional[QGraphicsEllipseItem]] = []
        self.pred_edge_items: List[Optional[QGraphicsLineItem]] = []
        self.pred_labels: List[Optional[QGraphicsTextItem]] = []

        self._img_w = 1
        self._img_h = 1
        self._current_kpt = 0
        self._click_vis = 2
        self._kpt_radius = DEFAULT_KPT_RADIUS
        self._label_font_size = DEFAULT_LABEL_FONT_SIZE
        self._palette = get_keypoint_palette()
        self._kpt_opacity = 1.0
        self._edge_opacity = 0.7
        self._zoom_factor = 1.0
        self._min_zoom = 0.3
        self._max_zoom = 30.0

        self._on_place = None
        self._on_move = None
        self._on_select = None
        self._dragging_kpt = None
        self._drag_start_pos = None
        self._dragging_pred = None
        self._pred_edges: List[Tuple[int, int]] = []

    def _ellipse_item_at(self, scene_pos) -> Optional[QGraphicsEllipseItem]:
        for it in self.scene.items(scene_pos):
            if isinstance(it, QGraphicsEllipseItem):
                return it
        return None

    def set_callbacks(self: object, on_place: object, on_move: object, on_select: object = None) -> None:
        """Register callbacks for placement, drag movement, and optional selection."""
        self._on_place = on_place
        self._on_move = on_move
        self._on_select = on_select

    def set_current_keypoint(self, idx: int) -> None:
        """Set the active keypoint index used for click placement."""
        self._current_kpt = idx

    def set_click_visibility(self, v: int) -> None:
        """Set visibility code assigned to newly placed keypoints."""
        self._click_vis = v

    def set_kpt_radius(self, r: float) -> None:
        """Update rendered keypoint radius."""
        self._kpt_radius = max(0.5, float(r))

    def set_label_font_size(self, size: int) -> None:
        """Update keypoint label font size."""
        self._label_font_size = max(4, int(size))

    def set_kpt_opacity(self, opacity: float) -> None:
        """Update keypoint alpha channel used for overlay drawing."""
        self._kpt_opacity = max(0.0, min(1.0, float(opacity)))

    def set_edge_opacity(self, opacity: float) -> None:
        """Update skeleton edge alpha channel used for overlay drawing."""
        self._edge_opacity = max(0.0, min(1.0, float(opacity)))

    def fit_to_view(self: object) -> object:
        """Fit the image to the view."""
        if self.pix_item.pixmap().isNull():
            return
        self.fitInView(self.pix_item, Qt.KeepAspectRatio)
        # Update zoom factor
        self._zoom_factor = self.transform().m11()

    def set_image(self, img_bgr: np.ndarray) -> None:
        """Load a BGR image into the canvas and fit it to the current viewport."""
        h, w = img_bgr.shape[:2]
        self._img_w, self._img_h = w, h
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def rebuild_overlays(
        self,
        kpts: List[Keypoint],
        kpt_names: List[str],
        edges: List[Tuple[int, int]],
        pred_kpts: Optional[List[Keypoint]] = None,
        pred_confs: Optional[List[float]] = None,
        show_pred_conf: bool = False,
    ) -> None:
        """Rebuild all keypoint/skeleton and optional prediction overlay items."""
        for item in (
            self.kpt_items
            + self.kpt_labels
            + self.edge_items
            + self.pred_items
            + self.pred_edge_items
            + self.pred_labels
        ):
            if item is not None:
                self.scene.removeItem(item)
        self.kpt_items = [None] * len(kpts)
        self.kpt_labels = [None] * len(kpts)
        self.edge_items = [None] * len(edges)
        self.pred_items = [None] * (len(pred_kpts) if pred_kpts else 0)
        self.pred_edge_items = [None] * len(edges)
        self.pred_labels = [None] * (len(pred_kpts) if pred_kpts else 0)

        # Create edges with opacity
        edge_alpha = int(255 * self._edge_opacity)
        for ei, (a, b) in enumerate(edges):
            pen = QPen(QColor(100, 100, 100, edge_alpha), 2)
            edge = QGraphicsLineItem()
            edge.setPen(pen)
            edge.setZValue(3)
            self.scene.addItem(edge)
            self.edge_items[ei] = edge

        # Create keypoints and labels
        label_positions = []  # Track label positions for collision avoidance
        for i, kp in enumerate(kpts):
            if kp.v == 0:
                continue

            color = self._palette[i % len(self._palette)]
            base = color
            alpha = int(255 * self._kpt_opacity)

            # Different appearance for occluded keypoints (v=1)
            if kp.v == 1:
                alpha = int(alpha * 0.6)

            color = QColor(base.red(), base.green(), base.blue(), alpha)

            r = self._kpt_radius
            circ = QGraphicsEllipseItem(kp.x - r, kp.y - r, 2 * r, 2 * r)
            pen_alpha = int(200 * self._kpt_opacity)

            # Use different border style for occluded keypoints
            if kp.v == 1:
                # Dashed border for occluded keypoints
                pen = QPen(QColor(255, 100, 0, pen_alpha), 2.0)
                pen.setStyle(Qt.DashLine)
            else:
                # Solid border for visible keypoints
                pen = QPen(QColor(0, 0, 0, pen_alpha), 1.5)

            circ.setPen(pen)
            circ.setBrush(color)
            circ.setZValue(4)
            # Movement is handled manually to keep edges/labels in sync.
            circ.setData(0, i)
            self.scene.addItem(circ)
            self.kpt_items[i] = circ

            nm = kpt_names[i] if i < len(kpt_names) else "kp"
            lab = QGraphicsTextItem(f"{i}:{nm}")
            label_alpha = int(220 * self._kpt_opacity)
            lab.setDefaultTextColor(
                QColor(base.red(), base.green(), base.blue(), label_alpha)
            )
            lab.setFont(QFont("Arial", self._label_font_size))
            lab.setAcceptedMouseButtons(Qt.NoButton)
            lab.setFlag(QGraphicsTextItem.ItemIsSelectable, False)

            # Smart label positioning with collision avoidance
            dx, dy = self._find_best_label_offset(kp.x, kp.y, label_positions)
            lab.setPos(kp.x + dx, kp.y + dy)
            lab.setZValue(5)

            # Track this label's bounding box
            label_rect = lab.boundingRect()
            label_positions.append(
                (kp.x + dx, kp.y + dy, label_rect.width(), label_rect.height())
            )

            self.scene.addItem(lab)
            self.kpt_labels[i] = lab

        # Prediction overlays (distinct style)
        self._pred_edges = list(edges)
        if pred_kpts:
            pred_edge_alpha = int(160 * self._edge_opacity)
            for ei, (a, b) in enumerate(edges):
                if (
                    a < len(pred_kpts)
                    and b < len(pred_kpts)
                    and pred_kpts[a].v > 0
                    and pred_kpts[b].v > 0
                ):
                    pen = QPen(QColor(0, 220, 255, pred_edge_alpha), 1.5)
                    pen.setStyle(Qt.DashLine)
                    edge = QGraphicsLineItem(
                        pred_kpts[a].x,
                        pred_kpts[a].y,
                        pred_kpts[b].x,
                        pred_kpts[b].y,
                    )
                    edge.setPen(pen)
                    edge.setZValue(1)
                    self.scene.addItem(edge)
                    self.pred_edge_items[ei] = edge

            pred_alpha = int(140 * self._kpt_opacity)
            for i, kp in enumerate(pred_kpts):
                if kp.v == 0:
                    continue
                r = max(2.0, self._kpt_radius - 1.5)
                circ = QGraphicsEllipseItem(kp.x - r, kp.y - r, 2 * r, 2 * r)
                pen = QPen(QColor(0, 220, 255, pred_alpha), 1.5)
                pen.setStyle(Qt.DashLine)
                circ.setPen(pen)
                circ.setBrush(Qt.NoBrush)
                circ.setZValue(2)
                circ.setData(0, i)
                circ.setData(1, "pred")
                self.scene.addItem(circ)
                self.pred_items[i] = circ
                if show_pred_conf and pred_confs and i < len(pred_confs):
                    conf = pred_confs[i]
                    lab = QGraphicsTextItem(f"{conf:.3f}")
                    lab.setDefaultTextColor(QColor(0, 220, 255, 200))
                    lab.setFont(QFont("Arial", max(4, int(self._label_font_size * 0.8))))
                    lab.setPos(kp.x + r + 2, kp.y - r - 2)
                    lab.setZValue(2)
                    lab.setAcceptedMouseButtons(Qt.NoButton)
                    self.scene.addItem(lab)
                    self.pred_labels[i] = lab

        self._update_edges_from_points(edges)

    def _update_edges_from_points(self, edges: List[Tuple[int, int]]):
        for ei, (a, b) in enumerate(edges):
            if ei >= len(self.edge_items):
                break
            pa = self.kpt_items[a] if 0 <= a < len(self.kpt_items) else None
            pb = self.kpt_items[b] if 0 <= b < len(self.kpt_items) else None
            if pa is None or pb is None:
                self.edge_items[ei].setVisible(False)
                continue
            self.edge_items[ei].setVisible(True)
            ra = pa.rect()
            rb = pb.rect()
            ax = ra.x() + ra.width() / 2
            ay = ra.y() + ra.height() / 2
            bx = rb.x() + rb.width() / 2
            by = rb.y() + rb.height() / 2
            self.edge_items[ei].setLine(ax, ay, bx, by)

    def _find_best_label_offset(
        self,
        x: float,
        y: float,
        existing_labels: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        """Find the best offset for a label to avoid collisions."""
        r = self._kpt_radius
        # Try multiple candidate positions in order of preference
        candidates = [
            (r + 8, -(r + 14)),  # top-right
            (r + 8, r + 4),  # bottom-right
            (-(r + 60), -(r + 14)),  # top-left
            (-(r + 60), r + 4),  # bottom-left
            (-(r + 26), -(r + 28)),  # far top-left
            (r + 22, -(r + 28)),  # far top-right
            (-(r + 26), r + 18),  # far bottom-left
            (r + 22, r + 18),  # far bottom-right
        ]

        # Approximate label size (will be refined when created)
        label_w = 50
        label_h = 16

        for dx, dy in candidates:
            label_x = x + dx
            label_y = y + dy

            # Check for collision with existing labels
            collision = False
            for ex, ey, ew, eh in existing_labels:
                # Check if rectangles overlap
                if not (
                    label_x + label_w < ex
                    or label_x > ex + ew
                    or label_y + label_h < ey
                    or label_y > ey + eh
                ):
                    collision = True
                    break

            if not collision:
                return (dx, dy)

        # If all positions collide, return the first one anyway
        return candidates[0]

    def wheelEvent(self: object, event: object) -> None:
        """Zoom in/out around cursor while clamping to sane zoom bounds."""
        # Use smaller zoom increments for smoother control
        factor = 1.01 if event.angleDelta().y() > 0 else 1 / 1.01
        new_zoom = self._zoom_factor * factor

        # Clamp zoom to reasonable bounds
        if new_zoom < self._min_zoom:
            new_zoom = self._min_zoom
            factor = new_zoom / self._zoom_factor
        elif new_zoom > self._max_zoom:
            new_zoom = self._max_zoom
            factor = new_zoom / self._zoom_factor

        self.scale(factor, factor)
        self._zoom_factor = new_zoom
        event.accept()

    def mousePressEvent(self: object, event: object) -> None:
        """Handle placement, selection, visibility toggle, and drag-start behavior."""
        if event.button() == Qt.RightButton:
            pos = self.mapToScene(event.position().toPoint())
            item = self._ellipse_item_at(pos)
            if item is not None:
                idx = int(item.data(0))
                is_pred = item.data(1) == "pred"
                if not is_pred:
                    parent = self.parent()
                    if parent and not hasattr(parent, "mode"):
                        parent = self.window()
                    if parent and hasattr(parent, "_toggle_kpt_visibility"):
                        parent._toggle_kpt_visibility(idx)
                        return
            if self._on_place:
                # Right-click marks keypoint as occluded (v=1)
                self._on_place(self._current_kpt, float(pos.x()), float(pos.y()), 1)
            return

        pos = self.mapToScene(event.position().toPoint())
        item = self._ellipse_item_at(pos)
        if item is not None:
            # Clicked on existing keypoint
            idx = int(item.data(0))
            is_pred = item.data(1) == "pred"

            # Get parent's mode
            parent = self.parent()
            # If parent is splitter (default Qt behavior when added to splitter), try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            mode = getattr(parent, "mode", "frame") if parent else "frame"

            if is_pred:
                # Start dragging prediction (adopt on release)
                self._dragging_pred = idx
                self._drag_start_pos = self.mapToScene(event.position().toPoint())
                return super().mousePressEvent(event)

            if mode == "keypoint":
                # In keypoint mode: NEVER allow dragging - treat click as placing next unlabeled keypoint
                ann = getattr(parent, "_ann", None) if parent else None
                if ann and hasattr(ann, "kpts"):
                    # Find next unlabeled keypoint
                    next_unlabeled = None
                    for i, kp in enumerate(ann.kpts):
                        if kp.v == 0:
                            next_unlabeled = i
                            break

                    if next_unlabeled is not None:
                        # Update canvas internal state FIRST
                        self._current_kpt = next_unlabeled

                        # Get position
                        pos = self.mapToScene(event.position().toPoint())

                        # Update parent state
                        if parent:
                            parent.current_kpt = next_unlabeled
                            parent.kpt_list.setCurrentRow(next_unlabeled)
                            # Update canvas visual state
                            parent.canvas.set_current_keypoint(next_unlabeled)
                            # Force UI update
                            QApplication.processEvents()

                        # Now place the keypoint with the updated state
                        if self._on_place:
                            self._on_place(
                                next_unlabeled,
                                float(pos.x()),
                                float(pos.y()),
                                self._click_vis,
                            )
                    # If all labeled, ignore click
                return super().mousePressEvent(event)
            else:
                # In frame mode: start dragging existing keypoint
                if self._on_select:
                    self._on_select(idx)
                self._dragging_kpt = idx
                self._drag_start_pos = self.mapToScene(event.position().toPoint())
                return super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            # Clicking on empty space - always place next unlabeled keypoint in both modes
            parent = self.parent()
            # If parent is splitter, try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            ann = getattr(parent, "_ann", None) if parent else None

            if ann and hasattr(ann, "kpts"):
                # Check if all keypoints are already labeled
                all_labeled = all(kp.v > 0 for kp in ann.kpts)
                if all_labeled:
                    # Ignore click - all keypoints exist
                    super().mousePressEvent(event)
                    return

                # Find next unlabeled keypoint
                next_unlabeled = None
                for i, kp in enumerate(ann.kpts):
                    if kp.v == 0:
                        next_unlabeled = i
                        break

                if next_unlabeled is not None:
                    # Update canvas internal state FIRST
                    self._current_kpt = next_unlabeled

                    # Update parent state
                    if parent:
                        parent.current_kpt = next_unlabeled
                        parent.kpt_list.setCurrentRow(next_unlabeled)
                        # Update canvas visual state
                        parent.canvas.set_current_keypoint(next_unlabeled)
                        # Force UI update
                        QApplication.processEvents()

                    # Now place the keypoint with the updated state
                    if self._on_place:
                        self._on_place(
                            next_unlabeled,
                            float(pos.x()),
                            float(pos.y()),
                            self._click_vis,
                        )
            else:
                # No annotation data, place normally
                if self._on_place:
                    self._on_place(
                        self._current_kpt,
                        float(pos.x()),
                        float(pos.y()),
                        self._click_vis,
                    )
        super().mousePressEvent(event)

    def mouseMoveEvent(self: object, event: object) -> None:
        """Update dragged keypoint/prediction position during pointer motion."""
        if self._dragging_pred is not None:
            pos = self.mapToScene(event.position().toPoint())
            pred_item = (
                self.pred_items[self._dragging_pred]
                if self._dragging_pred < len(self.pred_items)
                else None
            )
            if pred_item is not None:
                r = pred_item.rect().width() / 2.0
                pred_item.setRect(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
                self._update_pred_edges_from_items()
            super().mouseMoveEvent(event)
            return
        if self._dragging_kpt is not None and self._drag_start_pos is not None:
            # Never allow drag in keypoint mode
            parent = self.parent()
            # If parent is splitter, try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            mode = getattr(parent, "mode", "frame") if parent else "frame"
            if mode == "keypoint":
                # Keypoint mode: no dragging allowed
                super().mouseMoveEvent(event)
                return
            # Update keypoint position during drag (frame mode only)
            pos = self.mapToScene(event.position().toPoint())
            if self._on_move:
                self._on_move(self._dragging_kpt, float(pos.x()), float(pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self: object, event: object) -> None:
        """Commit drag interactions and optional prediction adoption on release."""
        if self._dragging_pred is not None:
            idx = self._dragging_pred
            self._dragging_pred = None
            pos = self.mapToScene(event.position().toPoint())
            parent = self.parent()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()
            if parent and hasattr(parent, "_adopt_prediction_kpt"):
                parent._adopt_prediction_kpt(idx, float(pos.x()), float(pos.y()))
            super().mouseReleaseEvent(event)
            return
        if self._dragging_kpt is not None:
            # Complete drag operation
            self._dragging_kpt = None
            self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _update_pred_edges_from_items(self):
        if not self.pred_edge_items or not self.pred_items:
            return
        for ei, edge in enumerate(self.pred_edge_items):
            if edge is None:
                continue
            if ei >= len(self._pred_edges):
                continue
            a, b = self._pred_edges[ei]
            if a >= len(self.pred_items) or b >= len(self.pred_items):
                continue
            a_item = self.pred_items[a]
            b_item = self.pred_items[b]
            if a_item is None or b_item is None:
                continue
            a_pos = a_item.rect().center()
            b_pos = b_item.rect().center()
            edge.setLine(a_pos.x(), a_pos.y(), b_pos.x(), b_pos.y())


# -----------------------------
# Frame list delegate
# -----------------------------
class FrameListDelegate(QStyledItemDelegate):
    """Render frame name with right-aligned prediction confidence."""

    CONF_ROLE = Qt.UserRole + 2
    KP_COUNT_ROLE = Qt.UserRole + 3
    CLUSTER_ROLE = Qt.UserRole + 4

    def paint(self: object, painter: object, option: object, index: object) -> None:
        """Render frame row with elided name plus confidence and label counters."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        text = index.data(Qt.DisplayRole) or ""
        conf_val = index.data(self.CONF_ROLE)
        conf_text = "" if conf_val in (None, "") else f"{float(conf_val):.3f}"
        kpt_val = index.data(self.KP_COUNT_ROLE)
        kpt_text = "" if kpt_val in (None, "") else f"k{int(kpt_val)}"
        cluster_val = index.data(self.CLUSTER_ROLE)
        if cluster_val in (None, "", -1):
            cluster_text = ""
        else:
            try:
                cluster_text = f"c{int(cluster_val)}"
            except Exception:
                cluster_text = ""

        fg = index.data(Qt.ForegroundRole)
        if isinstance(fg, QBrush):
            base_color = fg.color()
        elif isinstance(fg, QColor):
            base_color = fg
        else:
            base_color = opt.palette.color(QPalette.Text)

        if opt.state & QStyle.State_Selected:
            text_color = opt.palette.color(QPalette.HighlightedText)
            conf_color = opt.palette.color(QPalette.HighlightedText)
        else:
            text_color = base_color
            conf_color = QColor(140, 140, 140)

        r = opt.rect.adjusted(6, 0, -6, 0)
        conf_w = 54 if conf_text else 0
        kpt_w = 40 if kpt_text else 0
        cluster_w = 40 if cluster_text else 0
        available = max(0, r.width())
        total = conf_w + kpt_w + cluster_w
        if available < conf_w:
            conf_w = available
            kpt_w = 0
            cluster_w = 0
        elif available < conf_w + kpt_w:
            cluster_w = 0
        elif available < total:
            cluster_w = 0
        right_total = conf_w + kpt_w + cluster_w
        name_rect = QRect(r.left(), r.top(), max(0, r.width() - right_total), r.height())
        conf_rect = QRect(
            max(r.left(), r.right() - conf_w + 1), r.top(), max(0, conf_w), r.height()
        )
        kpt_rect = QRect(
            max(r.left(), conf_rect.left() - kpt_w), r.top(), max(0, kpt_w), r.height()
        )
        cluster_rect = QRect(
            max(r.left(), kpt_rect.left() - cluster_w),
            r.top(),
            max(0, cluster_w),
            r.height(),
        )

        painter.save()
        painter.setPen(text_color)
        if name_rect.width() > 0:
            fm = painter.fontMetrics()
            elided = fm.elidedText(text, Qt.ElideRight, max(0, name_rect.width()))
            painter.drawText(
                name_rect, Qt.AlignVCenter | Qt.AlignLeft, elided
            )
        if cluster_text and cluster_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            cluster_elided = fm.elidedText(
                cluster_text, Qt.ElideLeft, cluster_rect.width()
            )
            painter.drawText(
                cluster_rect, Qt.AlignVCenter | Qt.AlignRight, cluster_elided
            )
        if kpt_text and kpt_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            kpt_elided = fm.elidedText(kpt_text, Qt.ElideLeft, kpt_rect.width())
            painter.drawText(
                kpt_rect, Qt.AlignVCenter | Qt.AlignRight, kpt_elided
            )
        if conf_text and conf_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            conf_elided = fm.elidedText(conf_text, Qt.ElideLeft, conf_rect.width())
            painter.drawText(conf_rect, Qt.AlignVCenter | Qt.AlignRight, conf_elided)
        painter.restore()


# -----------------------------
# Skeleton editor
# -----------------------------
class SkeletonEditorDialog(QDialog):
    """Dialog to edit keypoint names and skeleton edge topology."""

    def __init__(
        self,
        keypoint_names: List[str],
        edges: List[Tuple[int, int]],
        parent=None,
        default_dir: Optional[Path] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Skeleton Editor")
        self.setMinimumSize(QSize(720, 420))

        self.kpt_names = list(keypoint_names)
        self.edges = list(edges)
        self.default_dir = default_dir

        # Main layout
        outer = QVBoxLayout(self)

        # Top section with keypoints and edges side-by-side
        root = QHBoxLayout()

        left = QVBoxLayout()
        left.addWidget(QLabel("What should keypoints (order matters) be?"))
        self.kpt_table = QTableWidget(len(self.kpt_names), 2)
        self.kpt_table.setHorizontalHeaderLabels(["Index", "Name"])
        self.kpt_table.verticalHeader().setVisible(False)
        self.kpt_table.setColumnWidth(0, 60)
        self.kpt_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.kpt_table.setSelectionMode(QTableWidget.SingleSelection)
        for i, name in enumerate(self.kpt_names):
            self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.kpt_table.setItem(i, 1, QTableWidgetItem(name))
        left.addWidget(self.kpt_table)

        kp_btns = QHBoxLayout()
        self.btn_kp_add = QPushButton("Add")
        self.btn_kp_del = QPushButton("Remove")
        self.btn_kp_ren = QPushButton("Rename")
        self.btn_kp_up = QPushButton("Up")
        self.btn_kp_down = QPushButton("Down")
        kp_btns.addWidget(self.btn_kp_add)
        kp_btns.addWidget(self.btn_kp_del)
        kp_btns.addWidget(self.btn_kp_ren)
        kp_btns.addWidget(self.btn_kp_up)
        kp_btns.addWidget(self.btn_kp_down)
        left.addLayout(kp_btns)

        btn_row = QHBoxLayout()
        self.btn_chain = QPushButton("Make Chain (i→i+1)")
        self.btn_clear_edges = QPushButton("Clear Edges")
        btn_row.addWidget(self.btn_chain)
        btn_row.addWidget(self.btn_clear_edges)
        left.addLayout(btn_row)

        io_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Skeleton…")
        self.btn_save = QPushButton("Save Skeleton…")
        io_row.addWidget(self.btn_load)
        io_row.addWidget(self.btn_save)
        left.addLayout(io_row)

        right = QVBoxLayout()
        right.addWidget(QLabel("What should edges (0-based indices) be?"))
        self.edge_table = QTableWidget(0, 2)
        self.edge_table.setHorizontalHeaderLabels(["A", "B"])
        self.edge_table.verticalHeader().setVisible(False)
        self.edge_table.setColumnWidth(0, 60)
        self.edge_table.setColumnWidth(1, 60)
        right.addWidget(self.edge_table)

        add_row = QHBoxLayout()
        self.a_spin = QSpinBox()
        self.b_spin = QSpinBox()
        mx = max(0, len(self.kpt_names) - 1)
        self.a_spin.setRange(0, mx)
        self.b_spin.setRange(0, mx)
        self.btn_add = QPushButton("Add Edge")
        self.btn_del = QPushButton("Delete Selected")
        add_row.addWidget(QLabel("What should a be?"))
        add_row.addWidget(self.a_spin)
        add_row.addWidget(QLabel("What should b be?"))
        add_row.addWidget(self.b_spin)
        add_row.addWidget(self.btn_add)
        add_row.addWidget(self.btn_del)
        right.addLayout(add_row)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        outer.addLayout(root)

        # Bottom buttons
        bottom = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        bottom.addStretch(1)
        bottom.addWidget(self.btn_ok)
        bottom.addWidget(self.btn_cancel)

        outer.addLayout(bottom)

        self.btn_add.clicked.connect(self._add_edge)
        self.btn_del.clicked.connect(self._del_edges)
        self.btn_chain.clicked.connect(self._make_chain)
        self.btn_clear_edges.clicked.connect(self._clear_edges)
        self.btn_load.clicked.connect(self._load_config)
        self.btn_save.clicked.connect(self._save_config)
        self.btn_kp_add.clicked.connect(self._kp_add)
        self.btn_kp_del.clicked.connect(self._kp_del)
        self.btn_kp_ren.clicked.connect(self._kp_rename)
        self.btn_kp_up.clicked.connect(lambda: self._kp_move(-1))
        self.btn_kp_down.clicked.connect(lambda: self._kp_move(1))
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)

        self._refresh_edges()

    def _refresh_edges(self):
        self.edge_table.setRowCount(len(self.edges))
        for r, (a, b) in enumerate(self.edges):
            self.edge_table.setItem(r, 0, QTableWidgetItem(str(a)))
            self.edge_table.setItem(r, 1, QTableWidgetItem(str(b)))

    def _refresh_kpt_indices(self):
        for i in range(self.kpt_table.rowCount()):
            idx_item = self.kpt_table.item(i, 0)
            if idx_item is None:
                idx_item = QTableWidgetItem(str(i))
                self.kpt_table.setItem(i, 0, idx_item)
            idx_item.setText(str(i))
            idx_item.setFlags(Qt.ItemIsEnabled)

    def _update_kpt_ranges(self):
        mx = max(0, self.kpt_table.rowCount() - 1)
        self.a_spin.setRange(0, mx)
        self.b_spin.setRange(0, mx)

    def _selected_kpt_row(self) -> int:
        rows = {i.row() for i in self.kpt_table.selectedIndexes()}
        return next(iter(rows), -1)

    def _kp_add(self):
        i = self.kpt_table.rowCount()
        self.kpt_table.insertRow(i)
        self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
        self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
        self.kpt_table.setItem(i, 1, QTableWidgetItem(f"kp{i+1}"))
        self.kpt_table.setCurrentCell(i, 1)
        self._update_kpt_ranges()

    def _kp_del(self):
        r = self._selected_kpt_row()
        if r < 0:
            return
        self.kpt_table.removeRow(r)
        self.edges = self._remap_edges_on_remove(self.edges, r)
        self._refresh_kpt_indices()
        self._update_kpt_ranges()
        self._refresh_edges()

    def _kp_rename(self):
        r = self._selected_kpt_row()
        if r < 0:
            return
        item = self.kpt_table.item(r, 1)
        current = item.text() if item else f"kp{r+1}"
        name, ok = self._simple_text_prompt("Rename keypoint", "New name:", current)
        if ok and name.strip():
            if item is None:
                item = QTableWidgetItem(name.strip())
                self.kpt_table.setItem(r, 1, item)
            else:
                item.setText(name.strip())

    def _kp_move(self, delta: int):
        r = self._selected_kpt_row()
        if r < 0:
            return
        nr = r + delta
        if nr < 0 or nr >= self.kpt_table.rowCount():
            return

        name_item = self.kpt_table.takeItem(r, 1)
        if name_item is None:
            name_item = QTableWidgetItem(f"kp{r+1}")
        self.kpt_table.removeRow(r)
        self.kpt_table.insertRow(nr)
        self.kpt_table.setItem(nr, 0, QTableWidgetItem(str(nr)))
        self.kpt_table.item(nr, 0).setFlags(Qt.ItemIsEnabled)
        self.kpt_table.setItem(nr, 1, name_item)
        self.kpt_table.setCurrentCell(nr, 1)

        self.edges = self._remap_edges_on_move(self.edges, r, nr)
        self._refresh_kpt_indices()
        self._refresh_edges()

    @staticmethod
    def _remap_edges_on_remove(
        edges: List[Tuple[int, int]], removed: int
    ) -> List[Tuple[int, int]]:
        out = []
        for a, b in edges:
            if a == removed or b == removed:
                continue
            na = a - 1 if a > removed else a
            nb = b - 1 if b > removed else b
            if na != nb:
                out.append((na, nb))
        return out

    @staticmethod
    def _remap_edges_on_move(
        edges: List[Tuple[int, int]], old: int, new: int
    ) -> List[Tuple[int, int]]:
        if old == new:
            return list(edges)

        def remap_idx(i: int) -> int:
            """remap_idx method documentation."""
            if i == old:
                return new
            if new > old and old < i <= new:
                return i - 1
            if new < old and new <= i < old:
                return i + 1
            return i

        out = []
        for a, b in edges:
            na = remap_idx(a)
            nb = remap_idx(b)
            if na != nb:
                out.append((na, nb))
        return out

    def _add_edge(self):
        a = int(self.a_spin.value())
        b = int(self.b_spin.value())
        if a == b:
            return
        if (a, b) not in self.edges and (b, a) not in self.edges:
            self.edges.append((a, b))
            self._refresh_edges()

    def _del_edges(self):
        rows = sorted(
            {i.row() for i in self.edge_table.selectedIndexes()}, reverse=True
        )
        for r in rows:
            if 0 <= r < len(self.edges):
                self.edges.pop(r)
        self._refresh_edges()

    def _make_chain(self):
        self.edges = [(i, i + 1) for i in range(max(0, len(self.kpt_names) - 1))]
        self._refresh_edges()

    def _clear_edges(self):
        self.edges = []
        self._refresh_edges()

    def _simple_text_prompt(
        self, title: str, label: str, default: str
    ) -> Tuple[str, bool]:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        lay = QFormLayout(dlg)
        le = QLineEdit(default)
        lay.addRow(label, le)
        row = QHBoxLayout()
        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addRow(row)
        ok.clicked.connect(dlg.accept)
        cancel.clicked.connect(dlg.reject)
        if dlg.exec() == QDialog.Accepted:
            return le.text(), True
        return default, False

    def _pick_skeleton_path(self, save: bool) -> Optional[Path]:
        start_dir = str(self.default_dir) if self.default_dir else os.getcwd()
        if save:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Skeleton",
                os.path.join(start_dir, "skeleton.json"),
                "Skeleton JSON (*.json)",
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Skeleton",
                start_dir,
                "Skeleton JSON (*.json)",
            )
        return Path(path) if path else None

    def _apply_config(self, names: List[str], edges: List[Tuple[int, int]]):
        if not names:
            return
        if len(names) != self.kpt_table.rowCount():
            resp = QMessageBox.question(
                self,
                "Replace keypoints?",
                f"Skeleton has {len(names)} keypoints but editor has {self.kpt_table.rowCount()}\n"
                "Replace keypoints with loaded list?",
            )
            if resp != QMessageBox.Yes:
                return
            self.kpt_table.setRowCount(len(names))

        self.kpt_names = list(names)
        for i, name in enumerate(self.kpt_names):
            self.kpt_table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.kpt_table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.kpt_table.setItem(i, 1, QTableWidgetItem(name))

        self._update_kpt_ranges()

        k = len(self.kpt_names)
        self.edges = [
            (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
        ]
        self._refresh_edges()

    def _load_config(self):
        path = self._pick_skeleton_path(save=False)
        if path is None:
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"Failed to read file:\n{exc}")
            return

        names = data.get("keypoint_names") or data.get("keypoints")
        edges = data.get("skeleton_edges") or data.get("edges")
        if not isinstance(names, list) or not isinstance(edges, list):
            QMessageBox.critical(
                self, "Invalid file", "Missing keypoint_names or edges."
            )
            return

        parsed_edges = []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                try:
                    parsed_edges.append((int(e[0]), int(e[1])))
                except Exception:
                    continue

        self._apply_config([str(n) for n in names], parsed_edges)

    def _save_config(self):
        path = self._pick_skeleton_path(save=True)
        if path is None:
            return
        names, edges = self.get_result()
        data = {
            "keypoint_names": names,
            "skeleton_edges": [[a, b] for a, b in edges],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Failed to write file:\n{exc}")

    def get_result(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """get_result method documentation."""
        out_names = []
        for i in range(self.kpt_table.rowCount()):
            item = self.kpt_table.item(i, 1)
            out_names.append(item.text().strip() if item else f"kp{i+1}")
        # Return at least the defaults if empty
        if not out_names:
            out_names = ["kp1", "kp2"]
        return out_names, list(self.edges)


# -----------------------------
# Setup wizard / Project settings dialog
# -----------------------------
class ProjectWizard(QDialog):
    """
    Used on first run (no project file) OR to edit project settings.
    """

    def __init__(
        self, images_dir: Path, existing: Optional[Project] = None, parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle("PoseKit Setup" if existing is None else "Project Settings")
        self.setMinimumSize(QSize(820, 560))

        self.images_dir = images_dir
        self.existing = existing

        # Defaults
        if existing is None:
            default_root = images_dir.parent / "pose_project"
            default_labels = default_root / "labels"
            default_classes = ["object"]
            default_kpts = ["kp1", "kp2"]
            default_pad = 0.03
            default_autosave = True
            default_edges: List[Tuple[int, int]] = []
        else:
            default_root = existing.out_root
            default_labels = existing.labels_dir
            default_classes = existing.class_names
            default_kpts = existing.keypoint_names
            default_pad = existing.bbox_pad_frac
            default_autosave = existing.autosave
            default_edges = existing.skeleton_edges

        self._edges = list(default_edges)
        self._kpt_names = list(default_kpts)

        layout = QVBoxLayout(self)

        # Paths
        form = QFormLayout()
        self.out_root = QLineEdit(str(default_root))
        btn_root = QPushButton("Choose…")
        row_root = QHBoxLayout()
        row_root.addWidget(self.out_root, 1)
        row_root.addWidget(btn_root)
        form.addRow("Where should PoseKit project outputs be stored?", row_root)

        self.labels_dir = QLineEdit(str(default_labels))
        btn_labels = QPushButton("Choose…")
        row_labels = QHBoxLayout()
        row_labels.addWidget(self.labels_dir, 1)
        row_labels.addWidget(btn_labels)
        form.addRow("Where should annotation label files be written?", row_labels)

        self.autosave_cb = QCheckBox("Autosave when changing frames")
        self.autosave_cb.setChecked(default_autosave)
        form.addRow("", self.autosave_cb)

        self.pad_spin = QDoubleSpinBox()
        self.pad_spin.setRange(0.0, 0.25)
        self.pad_spin.setSingleStep(0.01)
        self.pad_spin.setValue(default_pad)
        form.addRow("How much padding should be added around keypoints for bbox export?", self.pad_spin)

        layout.addLayout(form)

        # Classes
        cls_box = QWidget()
        cls_layout = QVBoxLayout(cls_box)
        cls_layout.addWidget(QLabel("What should classes (one per line) be?"))
        self.classes_edit = QPlainTextEdit("\n".join(default_classes))
        cls_layout.addWidget(self.classes_edit, 1)
        layout.addWidget(cls_box, 1)

        # Skeleton Editor button
        skel_box = QWidget()
        skel_layout = QVBoxLayout(skel_box)
        skel_layout.addWidget(QLabel("What should keypoints & skeleton be?"))
        self.btn_skel = QPushButton("Edit Keypoints & Skeleton…")
        skel_layout.addWidget(self.btn_skel)

        # Summary label
        self.skel_summary = QLabel(self._get_skeleton_summary())
        self.skel_summary.setWordWrap(True)
        skel_layout.addWidget(self.skel_summary)
        skel_layout.addStretch(1)

        # Migration (only if editing existing project)
        self.mig_box = QWidget()
        mig_layout = QVBoxLayout(self.mig_box)
        self.migrate_cb = QCheckBox(
            "Migrate existing label files to new keypoint layout"
        )
        self.migrate_cb.setChecked(True)
        mig_layout.addWidget(self.migrate_cb)

        self.mig_mode_group = QButtonGroup(self)
        self.rb_by_name = QRadioButton(
            "Map by keypoint NAME (recommended if names stable)"
        )
        self.rb_by_index = QRadioButton(
            "Map by INDEX (recommended if you append/remove at end)"
        )
        self.rb_by_name.setChecked(True)
        self.mig_mode_group.addButton(self.rb_by_name)
        self.mig_mode_group.addButton(self.rb_by_index)
        mig_layout.addWidget(self.rb_by_name)
        mig_layout.addWidget(self.rb_by_index)

        if existing is None:
            self.mig_box.setVisible(False)

        skel_layout.addWidget(self.mig_box)
        layout.addWidget(skel_box)

        # OK / Cancel
        bottom = QHBoxLayout()
        self.ok_btn = QPushButton("Create Project" if existing is None else "Apply")
        self.cancel_btn = QPushButton("Cancel")
        bottom.addStretch(1)
        bottom.addWidget(self.ok_btn)
        bottom.addWidget(self.cancel_btn)
        layout.addLayout(bottom)

        # wiring
        btn_root.clicked.connect(self._pick_root)
        btn_labels.clicked.connect(self._pick_labels)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.btn_skel.clicked.connect(self._edit_skeleton)

    def _pick_root(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select output root", self.out_root.text()
        )
        if d:
            self.out_root.setText(d)

    def _pick_labels(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select labels directory", self.labels_dir.text()
        )
        if d:
            self.labels_dir.setText(d)

    def _get_skeleton_summary(self) -> str:
        k = len(self._kpt_names)
        e = len(self._edges)
        return f"{k} keypoint{'s' if k != 1 else ''}, {e} edge{'s' if e != 1 else ''}"

    def _edit_skeleton(self):
        dlg = SkeletonEditorDialog(
            self._kpt_names, self._edges, self, default_dir=get_default_skeleton_dir()
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()
            self._kpt_names = list(names) if names else ["kp1", "kp2"]
            k = len(self._kpt_names)
            self._edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self.skel_summary.setText(self._get_skeleton_summary())

    def get_classes(self) -> List[str]:
        """get_classes method documentation."""
        lines = [s.strip() for s in self.classes_edit.toPlainText().splitlines()]
        out = [s for s in lines if s]
        return out if out else ["object"]

    def get_keypoints(self) -> List[str]:
        """get_keypoints method documentation."""
        return list(self._kpt_names) if self._kpt_names else ["kp1", "kp2"]

    def get_edges(self) -> List[Tuple[int, int]]:
        """get_edges method documentation."""
        return list(self._edges)

    def get_paths(self) -> Tuple[Path, Path]:
        """get_paths method documentation."""
        root = Path(self.out_root.text()).expanduser().resolve()
        labels = Path(self.labels_dir.text()).expanduser().resolve()
        return root, labels

    def get_options(self) -> Tuple[bool, float]:
        """get_options method documentation."""
        return bool(self.autosave_cb.isChecked()), float(self.pad_spin.value())

    def get_migration(self) -> Tuple[bool, str]:
        """get_migration method documentation."""
        if self.existing is None:
            return False, "name"
        do = bool(self.migrate_cb.isChecked())
        mode = "name" if self.rb_by_name.isChecked() else "index"
        return do, mode


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QMainWindow):
    """MainWindow API surface documentation."""
    def __init__(self, project: Project, image_paths: List[Path]):
        super().__init__()
        self.setWindowTitle("PoseKit Labeler")

        self.project = project
        self.image_paths = image_paths
        self.current_index = max(0, min(project.last_index, len(image_paths) - 1))
        self.current_kpt = 0
        self.mode = "frame"  # frame | keypoint
        self.click_vis = 2
        self._img_bgr = None
        self._img_display = None
        self._img_wh = (1, 1)
        self._ann: Optional[FrameAnn] = None
        self._dirty = False
        self._undo_stack: List[List[Keypoint]] = []
        self._undo_max = 50
        self._frame_cache: Dict[int, FrameAnn] = {}
        self._suppress_list_rebuild = False
        self._pred_thread: Optional[QThread] = None
        self._pred_worker: Optional[PosePredictWorker] = None
        self._bulk_pred_thread: Optional[QThread] = None
        self._bulk_pred_worker: Optional[BulkPosePredictWorker] = None
        self._sleap_service_thread: Optional[QThread] = None
        self._sleap_service_worker: Optional[SleapServiceWorker] = None
        self.show_predictions = True
        self.show_pred_conf = False
        self._sleap_env_pref = ""
        self._pred_conf_cache_key: Optional[str] = None
        self._pred_conf_map: Dict[str, Optional[float]] = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map: Dict[str, Optional[int]] = {}
        self._list_items: Dict[int, QListWidgetItem] = {}
        self._cluster_ids_cache: Optional[List[int]] = None
        self._cluster_ids_mtime: Optional[float] = None
        self.infer = PoseInferenceService(
            self.project.out_root,
            self.project.keypoint_names,
            self.project.skeleton_edges,
        )

        # Track which frames are in the labeling set (empty by default)
        self.labeling_frames: set = set()
        if getattr(project, "labeling_frames", None):
            self.labeling_frames = {
                int(i)
                for i in project.labeling_frames
                if 0 <= int(i) < len(image_paths)
            }
        self.metadata_manager = MetadataManager(
            self.project.out_root / "posekit" / "metadata.json"
        )
        self.autosave_delay_ms = DEFAULT_AUTOSAVE_DELAY_MS
        self._rebuild_path_index()

        splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(splitter)

        # Frames lists - dual list with drag-drop
        left = QWidget()
        left_layout = QVBoxLayout(left)

        left_layout.addWidget(QLabel("Labeling Frames"))
        self.labeling_list = QListWidget()
        self.labeling_list.setDragDropMode(QListWidget.DragDrop)
        self.labeling_list.setDefaultDropAction(Qt.MoveAction)
        self.labeling_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.labeling_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.labeling_list.setTextElideMode(Qt.ElideRight)
        self.labeling_list.setItemDelegate(FrameListDelegate(self.labeling_list))
        left_layout.addWidget(self.labeling_list, 1)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Find frame...")
        left_layout.addWidget(self.search_edit)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("How should frames be sorted?"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(
            [
                "Default",
                "Pred conf (high to low)",
                "Pred conf (low to high)",
                "Detected kpts (high to low)",
                "Detected kpts (low to high)",
                "Cluster id (low to high)",
                "Cluster id (high to low)",
            ]
        )
        sort_row.addWidget(self.sort_combo, 1)
        left_layout.addLayout(sort_row)

        left_layout.addWidget(QLabel("All Frames"))
        self.frame_list = QListWidget()
        self.frame_list.setDragDropMode(QListWidget.DragDrop)
        self.frame_list.setDefaultDropAction(Qt.MoveAction)
        self.frame_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.frame_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frame_list.setTextElideMode(Qt.ElideRight)
        self.frame_list.setItemDelegate(FrameListDelegate(self.frame_list))
        left_layout.addWidget(self.frame_list, 1)

        # Frame management buttons
        frame_btns = QVBoxLayout()
        self.btn_unlabeled_to_labeling = QPushButton("Unlabeled → Labeling")
        self.btn_unlabeled_to_labeling.setToolTip(
            "Move all unlabeled frames to labeling list"
        )
        frame_btns.addWidget(self.btn_unlabeled_to_labeling)

        self.btn_unlabeled_to_all = QPushButton("Unlabeled → All")
        self.btn_unlabeled_to_all.setToolTip(
            "Move unlabeled frames from labeling to all frames list"
        )
        frame_btns.addWidget(self.btn_unlabeled_to_all)

        random_row = QHBoxLayout()
        self.btn_random_to_labeling = QPushButton("Random")
        self.btn_random_to_labeling.setToolTip(
            "Add random unlabeled frames to labeling"
        )
        self.spin_random_count = QSpinBox()
        self.spin_random_count.setRange(1, 1000)
        self.spin_random_count.setValue(10)
        random_row.addWidget(self.btn_random_to_labeling)
        random_row.addWidget(self.spin_random_count)
        frame_btns.addLayout(random_row)

        self.btn_smart_select = QPushButton("Smart Select…")
        self.btn_smart_select.setToolTip(
            "Select diverse frames using embeddings + clustering"
        )
        frame_btns.addWidget(self.btn_smart_select)
        self.btn_smart_select.clicked.connect(self.open_smart_select)

        self.btn_delete_frames = QPushButton("Delete Selected…")
        self.btn_delete_frames.setToolTip(
            "Permanently delete selected images (and labels) from the dataset"
        )
        frame_btns.addWidget(self.btn_delete_frames)

        left_layout.addLayout(frame_btns)

        # Canvas
        # Load UI settings - will be applied after widgets are created
        self._ui_settings = load_ui_settings()

        self.canvas = PoseCanvas(parent=self)
        self.canvas.set_callbacks(
            self.on_place_kpt, self.on_move_kpt, self.on_select_kpt
        )
        self.canvas.set_kpt_radius(self.project.kpt_radius)
        self.canvas.set_label_font_size(self.project.label_font_size)
        self.canvas.set_kpt_opacity(self.project.kpt_opacity)
        self.canvas.set_edge_opacity(self.project.edge_opacity)
        self.canvas_hint = QLabel(
            "Left click: place/move  •  Right click: toggle vis / place occluded  •  "
            "Wheel: zoom  •  A/D: prev/next  •  Q/E: prev/next keypoint  •  "
            "Space: advance  •  Ctrl+S: save"
        )
        self.canvas_hint.setWordWrap(True)
        self.canvas_hint.setAlignment(Qt.AlignCenter)
        self.canvas_hint.setStyleSheet("QLabel { color: #666; padding: 6px; }")

        self._setting_meta = False
        self.meta_tags_label = QLabel("What should frame tags be?")
        self.meta_tags = {}
        tags_row = QHBoxLayout()
        for tag in [
            "occluded",
            "weird_posture",
            "motion_blur",
            "poor_lighting",
            "partial_view",
            "unclear",
        ]:
            cb = QCheckBox(tag)
            cb.toggled.connect(self._on_meta_changed)
            self.meta_tags[tag] = cb
            tags_row.addWidget(cb)
        tags_row.addStretch(1)

        self.meta_notes = QLineEdit()
        self.meta_notes.setPlaceholderText("Notes for this frame…")
        self.meta_notes.textEdited.connect(self._on_meta_changed)

        meta_box = QWidget()
        meta_layout = QVBoxLayout(meta_box)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(4)
        meta_layout.addWidget(self.meta_tags_label)
        meta_layout.addLayout(tags_row)
        meta_layout.addWidget(self.meta_notes)
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(4)
        canvas_layout.addWidget(self.canvas, 1)
        canvas_layout.addWidget(self.canvas_hint, 0)
        canvas_layout.addWidget(meta_box, 0)

        # Tools
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        # Annotation
        ann_group = QGroupBox("Annotation")
        ann_layout = QVBoxLayout(ann_group)
        ann_layout.addWidget(QLabel("Which class is being annotated?"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.project.class_names)
        ann_layout.addWidget(self.class_combo)
        ann_layout.addSpacing(4)
        ann_layout.addWidget(QLabel("Which keypoints are being edited?"))
        self.kpt_list = QListWidget()
        self._rebuild_kpt_list()
        ann_layout.addWidget(self.kpt_list, 1)
        ann_layout.addSpacing(4)
        ann_layout.addWidget(QLabel("How should annotation progression advance?"))
        self.rb_frame = QRadioButton("Frame-by-frame")
        self.rb_kpt = QRadioButton("Keypoint-by-keypoint")
        self.rb_frame.setChecked(True)
        ann_layout.addWidget(self.rb_frame)
        ann_layout.addWidget(self.rb_kpt)
        ann_layout.addSpacing(4)
        ann_layout.addWidget(QLabel("What visibility state should clicks assign?"))
        self.vis_group = QButtonGroup(self)
        self.rb_vis = QRadioButton("Visible (2)")
        self.rb_occ = QRadioButton("Occluded (1)")
        self.rb_miss = QRadioButton("Missing (0)")
        self.rb_vis.setChecked(True)
        self.vis_group.addButton(self.rb_vis, 2)
        self.vis_group.addButton(self.rb_occ, 1)
        self.vis_group.addButton(self.rb_miss, 0)
        ann_layout.addWidget(self.rb_vis)
        ann_layout.addWidget(self.rb_occ)
        ann_layout.addWidget(self.rb_miss)
        right_layout.addWidget(ann_group)

        # Display
        disp_group = QGroupBox("Display")
        disp_layout = QVBoxLayout(disp_group)
        self.cb_enhance = QCheckBox("Enhance contrast (CLAHE)")
        self.cb_enhance.setChecked(bool(self.project.enhance_enabled))
        self.btn_enhance_settings = QPushButton("Enhancement settings…")
        disp_layout.addWidget(self.cb_enhance)
        disp_layout.addWidget(self.btn_enhance_settings)
        self.cb_show_preds = QCheckBox("Show predictions")
        self.cb_show_preds.setChecked(True)
        disp_layout.addWidget(self.cb_show_preds)
        self.cb_show_pred_conf = QCheckBox("Show pred confidence")
        self.cb_show_pred_conf.setChecked(False)
        disp_layout.addWidget(self.cb_show_pred_conf)

        autosave_row = QHBoxLayout()
        autosave_row.addWidget(QLabel("How often should autosave run (sec)?"))
        self.sp_autosave_delay = QDoubleSpinBox()
        self.sp_autosave_delay.setRange(0.5, 30.0)
        self.sp_autosave_delay.setSingleStep(0.5)
        self.sp_autosave_delay.setValue(self.autosave_delay_ms / 1000.0)
        autosave_row.addWidget(self.sp_autosave_delay)
        disp_layout.addLayout(autosave_row)

        opacity_row1 = QHBoxLayout()
        opacity_row1.addWidget(QLabel("Keypoint opacity"))
        self.sp_kpt_opacity = QDoubleSpinBox()
        self.sp_kpt_opacity.setRange(0.0, 1.0)
        self.sp_kpt_opacity.setSingleStep(0.05)
        self.sp_kpt_opacity.setValue(self.project.kpt_opacity)
        opacity_row1.addWidget(self.sp_kpt_opacity)
        disp_layout.addLayout(opacity_row1)

        opacity_row2 = QHBoxLayout()
        opacity_row2.addWidget(QLabel("Edge opacity"))
        self.sp_edge_opacity = QDoubleSpinBox()
        self.sp_edge_opacity.setRange(0.0, 1.0)
        self.sp_edge_opacity.setSingleStep(0.05)
        self.sp_edge_opacity.setValue(self.project.edge_opacity)
        opacity_row2.addWidget(self.sp_edge_opacity)
        disp_layout.addLayout(opacity_row2)

        size_row = QHBoxLayout()
        self.sp_kpt_size = QDoubleSpinBox()
        self.sp_kpt_size.setRange(0.5, 20.0)
        self.sp_kpt_size.setSingleStep(0.5)
        self.sp_kpt_size.setValue(float(self.project.kpt_radius))
        self.sp_label_size = QSpinBox()
        self.sp_label_size.setRange(4, 20)
        self.sp_label_size.setValue(int(self.project.label_font_size))
        size_row.addWidget(QLabel("What should point be?"))
        size_row.addWidget(self.sp_kpt_size)
        size_row.addSpacing(6)
        size_row.addWidget(QLabel("What should text be?"))
        size_row.addWidget(self.sp_label_size)
        disp_layout.addLayout(size_row)

        self.btn_fit_view = QPushButton("Fit to View (Ctrl+0)")
        disp_layout.addWidget(self.btn_fit_view)
        right_layout.addWidget(disp_group)

        # Navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_group)
        row1 = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev (A)")
        self.btn_next = QPushButton("Next (D) ▶")
        row1.addWidget(self.btn_prev)
        row1.addWidget(self.btn_next)
        nav_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_save = QPushButton("Save (Ctrl+S)")
        self.btn_next_unl = QPushButton("Next Unlabeled (Ctrl+F)")
        row2.addWidget(self.btn_save)
        row2.addWidget(self.btn_next_unl)
        nav_layout.addLayout(row2)
        right_layout.addWidget(nav_group)

        # Model
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Which inference backend should generate predictions?"))
        self.combo_pred_backend = QComboBox()
        self.combo_pred_backend.addItems(["YOLO", "SLEAP"])
        backend_row.addWidget(self.combo_pred_backend, 1)
        model_layout.addLayout(backend_row)

        pred_conf_row = QHBoxLayout()
        pred_conf_row.addWidget(QLabel("What minimum prediction confidence should be shown/applied?"))
        self.sp_pred_conf = QDoubleSpinBox()
        self.sp_pred_conf.setRange(0.0, 1.0)
        self.sp_pred_conf.setSingleStep(0.05)
        self.sp_pred_conf.setValue(0.25)
        self.sp_pred_conf.setToolTip(
            "Minimum keypoint confidence to display/apply (YOLO + SLEAP)."
        )
        pred_conf_row.addWidget(self.sp_pred_conf)
        model_layout.addLayout(pred_conf_row)

        # YOLO settings
        self.yolo_pred_widget = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_pred_widget)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.addWidget(QLabel("Pose weights (.pt)"))
        pred_weights_row = QHBoxLayout()
        self.pred_weights_edit = QLineEdit("")
        self.pred_weights_edit.setPlaceholderText("Select weights (.pt)")
        self.btn_pred_weights = QPushButton("Browse…")
        self.btn_pred_weights_latest = QPushButton("Use Latest")
        pred_weights_row.addWidget(self.pred_weights_edit, 1)
        pred_weights_row.addWidget(self.btn_pred_weights)
        pred_weights_row.addWidget(self.btn_pred_weights_latest)
        yolo_layout.addLayout(pred_weights_row)
        model_layout.addWidget(self.yolo_pred_widget)

        # SLEAP settings
        self.sleap_pred_widget = QWidget()
        sleap_layout = QFormLayout(self.sleap_pred_widget)
        sleap_layout.setContentsMargins(0, 0, 0, 0)
        env_row = QHBoxLayout()
        self.combo_sleap_env = QComboBox()
        self.combo_sleap_env.setToolTip("Environment name must start with 'sleap'.")
        self.btn_sleap_refresh = QPushButton("↻")
        self.btn_sleap_refresh.setMaximumWidth(40)
        self.btn_sleap_refresh.setToolTip("Refresh conda environments list")
        env_row.addWidget(self.combo_sleap_env, 1)
        env_row.addWidget(self.btn_sleap_refresh)
        sleap_layout.addRow("What should conda env be?", env_row)
        self.lbl_sleap_env_status = QLabel("")
        self.lbl_sleap_env_status.setStyleSheet("QLabel { color: #b00; }")
        sleap_layout.addRow("", self.lbl_sleap_env_status)

        model_row = QHBoxLayout()
        self.sleap_model_edit = QLineEdit("")
        self.sleap_model_edit.setPlaceholderText("Select SLEAP model directory")
        self.btn_sleap_model = QPushButton("Browse…")
        self.btn_sleap_model_latest = QPushButton("Use Latest")
        model_row.addWidget(self.sleap_model_edit, 1)
        model_row.addWidget(self.btn_sleap_model)
        model_row.addWidget(self.btn_sleap_model_latest)
        sleap_layout.addRow("Where should model dir be set?", model_row)

        self.combo_sleap_device = QComboBox()
        self.combo_sleap_device.addItems(get_available_devices())
        sleap_layout.addRow("Which device should be used?", self.combo_sleap_device)

        sleap_btns = QHBoxLayout()
        self.btn_sleap_start = QPushButton("Start SLEAP Service")
        self.btn_sleap_stop = QPushButton("Stop SLEAP Service")
        sleap_btns.addWidget(self.btn_sleap_start)
        sleap_btns.addWidget(self.btn_sleap_stop)
        sleap_layout.addRow("", sleap_btns)

        model_layout.addWidget(self.sleap_pred_widget)

        self.btn_train = QPushButton("Train / Fine-tune…")
        self.btn_eval = QPushButton("Evaluate…")
        self.btn_active = QPushButton("Active Learning…")
        self.btn_predict = QPushButton("Predict Keypoints…")
        self.btn_predict_bulk = QPushButton("Predict Dataset…")
        self.btn_apply_preds = QPushButton("Apply Predictions")
        self.btn_clear_pred_cache = QPushButton("Clear Prediction Cache")
        model_layout.addWidget(self.btn_train)
        model_layout.addWidget(self.btn_eval)
        model_layout.addWidget(self.btn_active)
        model_layout.addWidget(self.btn_predict)
        model_layout.addWidget(self.btn_predict_bulk)
        model_layout.addWidget(self.btn_apply_preds)
        model_layout.addWidget(self.btn_clear_pred_cache)
        right_layout.addWidget(model_group)

        # Project
        proj_group = QGroupBox("Project")
        proj_layout = QVBoxLayout(proj_group)
        self.btn_skel = QPushButton("Skeleton Editor (Ctrl+G)")
        self.btn_proj = QPushButton("Project Settings (Ctrl+P)")
        self.btn_export = QPushButton("Export dataset.yaml + splits…")
        proj_layout.addWidget(self.btn_skel)
        proj_layout.addWidget(self.btn_proj)
        proj_layout.addWidget(self.btn_export)
        right_layout.addWidget(proj_group)

        # Shortcuts
        self.controls_group = QGroupBox("Shortcuts")
        controls_layout = QVBoxLayout(self.controls_group)
        self.controls_label = QLabel(
            "Left click: place/move keypoint\n"
            "Right click: toggle vis (on keypoint)\n"
            "Wheel: zoom\n"
            "A/D: prev/next frame\n"
            "Q/E: prev/next keypoint\n"
            "Space: advance\n"
            "V/O/N: set visibility\n"
            "Ctrl+S: save\n"
            "Ctrl+F: next unlabeled"
        )
        self.controls_label.setWordWrap(True)
        controls_layout.addWidget(self.controls_label)
        right_layout.addWidget(self.controls_group)

        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        right_layout.addWidget(self.lbl_info)
        right_layout.addStretch(1)

        # Wrap left and right panels in ScrollAreas to allow scaling
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setMinimumWidth(320)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_scroll = QScrollArea()
        right_scroll.setWidget(right)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setMinimumWidth(320)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        splitter.addWidget(left_scroll)
        splitter.addWidget(canvas_container)
        splitter.addWidget(right_scroll)

        # Give significantly more space to canvas (center is 10x larger than side panels)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 10)
        splitter.setStretchFactor(2, 1)

        splitter.setHandleWidth(8)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #b0b0b0; }"
        )

        # Set initial sizes: ~200px per side panel, rest to center
        # This will be adjusted when window is shown based on total width
        splitter.setSizes([320, 840, 320])

        # Set minimum sizes to 0 to allow full collapsing/scaling
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, True)

        self.setStatusBar(QStatusBar())
        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setFixedWidth(160)
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)
        self.status_sleap = QLabel("SLEAP: off")
        self.status_sleap.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_sleap)
        self._last_saved_at = None
        self.status_autosave = QLabel("Saved")
        self.statusBar().addPermanentWidget(self.status_autosave)

        self._build_actions()

        # Periodic garbage collection to keep memory pressure down after heavy ops.
        self._gc_timer = QTimer(self)
        self._gc_timer.setInterval(60000)
        self._gc_timer.timeout.connect(lambda: gc.collect())
        self._gc_timer.start()

        # Timed autosave to avoid frequent blocking writes.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._perform_autosave)

        # Signals
        self.labeling_list.currentRowChanged.connect(self._on_labeling_frame_selected)
        self.frame_list.currentRowChanged.connect(self._on_all_frame_selected)
        self.labeling_list.model().rowsMoved.connect(self._on_labeling_list_changed)
        self.frame_list.model().rowsMoved.connect(self._on_all_list_changed)
        self.kpt_list.currentRowChanged.connect(self._on_kpt_selected)
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_next_unl.clicked.connect(self.next_unlabeled)
        self.btn_skel.clicked.connect(self.open_skeleton_editor)
        self.btn_proj.clicked.connect(self.open_project_settings)
        self.btn_export.clicked.connect(self.export_dataset_dialog)
        self.sp_autosave_delay.valueChanged.connect(self._update_autosave_delay)
        self.sp_pred_conf.valueChanged.connect(self._on_pred_conf_changed)
        self.btn_train.clicked.connect(self.open_training_runner)
        self.btn_eval.clicked.connect(self.open_evaluation_dashboard)
        self.btn_active.clicked.connect(self.open_active_learning)
        self.btn_predict.clicked.connect(self.predict_current_frame)
        self.btn_predict_bulk.clicked.connect(self.predict_dataset)
        self.cb_show_preds.toggled.connect(self._toggle_show_predictions)
        self.cb_show_pred_conf.toggled.connect(self._toggle_show_pred_conf)
        self.btn_apply_preds.clicked.connect(self.apply_predictions_current)
        self.btn_clear_pred_cache.clicked.connect(self._clear_prediction_cache)
        self.btn_pred_weights.clicked.connect(self._browse_pred_weights)
        self.btn_pred_weights_latest.clicked.connect(self._use_latest_pred_weights)
        self.combo_pred_backend.currentTextChanged.connect(
            self._update_pred_backend_ui
        )
        self.btn_sleap_refresh.clicked.connect(self._refresh_sleap_envs)
        self.btn_sleap_model.clicked.connect(self._browse_sleap_model_dir)
        self.btn_sleap_model_latest.clicked.connect(self._use_latest_sleap_model)
        self.btn_sleap_start.clicked.connect(self._start_sleap_service)
        self.btn_sleap_stop.clicked.connect(self._stop_sleap_service)
        self.btn_unlabeled_to_labeling.clicked.connect(self._move_unlabeled_to_labeling)
        self.btn_unlabeled_to_all.clicked.connect(self._move_unlabeled_to_all)
        self.btn_random_to_labeling.clicked.connect(self._add_random_to_labeling)
        self.btn_delete_frames.clicked.connect(self._delete_selected_frames)
        self.search_edit.textChanged.connect(self._populate_frames)
        self.sort_combo.currentTextChanged.connect(self._populate_frames)
        self.rb_frame.toggled.connect(self._update_mode)
        self.vis_group.buttonClicked.connect(self._update_vis_mode)
        self.class_combo.currentIndexChanged.connect(self._mark_dirty)
        self.cb_enhance.toggled.connect(self._toggle_enhancement)
        self.btn_enhance_settings.clicked.connect(self._open_enhancement_settings)
        self.sp_kpt_size.valueChanged.connect(self._update_kpt_size)
        self.sp_label_size.valueChanged.connect(self._update_label_size)
        self.sp_kpt_opacity.valueChanged.connect(self._update_kpt_opacity)
        self.sp_edge_opacity.valueChanged.connect(self._update_edge_opacity)
        self.btn_fit_view.clicked.connect(self.fit_to_view)

        # Load UI settings now that all widgets are created
        self._refresh_sleap_envs()
        self._sleap_batch_predict = 4
        self._load_ui_settings()
        self._update_pred_backend_ui()

        # Populate list + load
        self._populate_frames()
        # Don't auto-load a frame on startup to avoid odd zoom; user clicks to load.
        self.frame_list.setCurrentRow(-1)
        self.labeling_list.setCurrentRow(-1)
        self.lbl_info.setText("Select a frame to display.")

    def show_startup_open_overlay(self: object) -> object:
        """Show startup chooser overlay for opening a project or image directory."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Open PoseKit Dataset")
        dlg.setModal(True)
        dlg.setWindowFlag(Qt.WindowCloseButtonHint, False)

        layout = QVBoxLayout(dlg)
        msg = QLabel(
            "No dataset is open.\n\nChoose one option to continue:"
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        btn_open_project = QPushButton("Open Project…")
        btn_open_images = QPushButton("Open Images Folder…")
        btn_quit = QPushButton("Quit")
        layout.addWidget(btn_open_project)
        layout.addWidget(btn_open_images)
        layout.addWidget(btn_quit)

        def open_project():
            dlg.accept()
            self.open_project_dialog()
            if not self.image_paths:
                QTimer.singleShot(0, self.show_startup_open_overlay)

        def open_images():
            dlg.accept()
            self.open_images_folder()
            if not self.image_paths:
                QTimer.singleShot(0, self.show_startup_open_overlay)

        btn_open_project.clicked.connect(open_project)
        btn_open_images.clicked.connect(open_images)
        btn_quit.clicked.connect(lambda: (dlg.reject(), self.close()))
        dlg.exec()

    def closeEvent(self: object, event: object) -> object:
        """Save UI settings when window closes."""
        self._perform_autosave()
        self.save_project()
        self._save_ui_settings()
        try:
            PoseInferenceService.shutdown_sleap_service()
        except Exception:
            pass
        super().closeEvent(event)

    def keyPressEvent(self: object, event: object) -> object:
        """Handle arrow key nudging of keypoints with modifier keys."""
        # Only process arrow keys
        if event.key() not in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            super().keyPressEvent(event)
            return

        # Get current keypoint value
        if self._ann is None or self.current_kpt >= len(self._ann.kpts):
            super().keyPressEvent(event)
            return

        x, y, v = (
            self._ann.kpts[self.current_kpt].x,
            self._ann.kpts[self.current_kpt].y,
            self._ann.kpts[self.current_kpt].v,
        )

        # Only nudge if keypoint is placed
        if v <= 0:
            super().keyPressEvent(event)
            return

        # In keypoint mode, only allow nudging if all keypoints are labeled
        if self.mode == "keypoint":
            all_labeled = all(kp.v > 0 for kp in self._ann.kpts)
            if not all_labeled:
                super().keyPressEvent(event)
                return

        # Calculate base nudge amount (0.5% of average image dimension)
        if self._img_bgr is not None:
            h, w = self._img_bgr.shape[:2]
            base_nudge = 0.005 * ((w + h) / 2)
        else:
            base_nudge = 2.0  # fallback

        # Apply modifiers
        modifiers = event.modifiers()
        if modifiers & Qt.ShiftModifier:
            nudge = base_nudge * 5.0  # 5x faster
        elif modifiers & Qt.ControlModifier:
            nudge = base_nudge * 0.2  # 0.2x slower (pixel-level precision)
        else:
            nudge = base_nudge  # normal speed

        # Apply nudge based on arrow key
        if event.key() == Qt.Key_Left:
            x -= nudge
        elif event.key() == Qt.Key_Right:
            x += nudge
        elif event.key() == Qt.Key_Up:
            y -= nudge
        elif event.key() == Qt.Key_Down:
            y += nudge

        # Bounds checking
        if self._img_bgr is not None:
            h, w = self._img_bgr.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))

        # Update keypoint
        self._ann.kpts[self.current_kpt] = Keypoint(x, y, v)
        self._set_dirty()

        # Rebuild overlays
        self._rebuild_canvas()

        # Save/cache based on mode
        if self.mode == "frame":
            self.save_current(refresh_ui=False)
        else:  # keypoint mode
            self._cache_current_frame()

        event.accept()

    def _save_ui_settings(self):
        """Save UI settings to persistent storage."""
        settings = {
            "enhance_enabled": self.project.enhance_enabled,
            "kpt_radius": self.project.kpt_radius,
            "label_font_size": self.project.label_font_size,
            "kpt_opacity": self.project.kpt_opacity,
            "edge_opacity": self.project.edge_opacity,
            "clahe_clip": self.project.clahe_clip,
            "clahe_grid": list(self.project.clahe_grid),
            "sharpen_amt": self.project.sharpen_amt,
            "blur_sigma": self.project.blur_sigma,
            "controls_open": bool(self.controls_group.isChecked()),
            "frame_search": self.search_edit.text().strip(),
            "autosave_delay_ms": int(self.autosave_delay_ms),
            "pred_conf": float(self.sp_pred_conf.value()),
            "pred_weights": self.pred_weights_edit.text().strip(),
            "pred_backend": self.combo_pred_backend.currentText().strip(),
            "sleap_env": self.combo_sleap_env.currentText().strip(),
            "sleap_model_dir": self.sleap_model_edit.text().strip(),
            "sleap_device": self.combo_sleap_device.currentText().strip(),
            "sleap_batch_predict": int(getattr(self, "_sleap_batch_predict", 4)),
            "show_predictions": bool(self.cb_show_preds.isChecked()),
            "show_pred_conf": bool(self.cb_show_pred_conf.isChecked()),
        }
        save_ui_settings(settings)

    def _load_ui_settings(self):
        """Load UI settings from persistent storage and apply defaults."""
        settings = load_ui_settings()
        if settings:
            # Apply loaded settings to project
            if "kpt_radius" in settings:
                self.project.kpt_radius = float(settings["kpt_radius"])
                self.sp_kpt_size.setValue(self.project.kpt_radius)
            if "label_font_size" in settings:
                self.project.label_font_size = int(settings["label_font_size"])
                self.sp_label_size.setValue(self.project.label_font_size)
            if "kpt_opacity" in settings:
                self.project.kpt_opacity = float(settings["kpt_opacity"])
                self.sp_kpt_opacity.setValue(self.project.kpt_opacity)
            if "edge_opacity" in settings:
                self.project.edge_opacity = float(settings["edge_opacity"])
                self.sp_edge_opacity.setValue(self.project.edge_opacity)
            if "controls_open" in settings:
                self.controls_group.setChecked(bool(settings["controls_open"]))
            if "frame_search" in settings:
                self.search_edit.setText(str(settings["frame_search"]))
            if "autosave_delay_ms" in settings:
                self.autosave_delay_ms = int(settings["autosave_delay_ms"])
                self.sp_autosave_delay.setValue(self.autosave_delay_ms / 1000.0)
            if "pred_conf" in settings:
                self.sp_pred_conf.setValue(float(settings["pred_conf"]))
            if "pred_weights" in settings:
                self.pred_weights_edit.setText(str(settings["pred_weights"]))
            if "pred_backend" in settings:
                backend = str(settings["pred_backend"])
                if backend:
                    self.combo_pred_backend.setCurrentText(backend)
            if "sleap_env" in settings:
                self._sleap_env_pref = str(settings["sleap_env"]).strip()
                if self._sleap_env_pref:
                    if self._sleap_env_pref in [
                        self.combo_sleap_env.itemText(i)
                        for i in range(self.combo_sleap_env.count())
                    ]:
                        self.combo_sleap_env.setCurrentText(self._sleap_env_pref)
            if "sleap_model_dir" in settings:
                self.sleap_model_edit.setText(str(settings["sleap_model_dir"]))
            if "sleap_device" in settings:
                self.combo_sleap_device.setCurrentText(
                    str(settings["sleap_device"])
                )
            if "sleap_batch_predict" in settings:
                self._sleap_batch_predict = int(settings["sleap_batch_predict"])
            elif "sleap_batch" in settings:
                # Backwards compatibility for older settings key.
                self._sleap_batch_predict = int(settings["sleap_batch"])
            if "show_predictions" in settings:
                self.cb_show_preds.setChecked(bool(settings["show_predictions"]))
                self.show_predictions = bool(self.cb_show_preds.isChecked())
            if "show_pred_conf" in settings:
                self.cb_show_pred_conf.setChecked(bool(settings["show_pred_conf"]))
                self.show_pred_conf = bool(self.cb_show_pred_conf.isChecked())
            if not self.pred_weights_edit.text().strip():
                if (
                    self.project.latest_pose_weights
                    and Path(self.project.latest_pose_weights).exists()
                ):
                    self.pred_weights_edit.setText(str(self.project.latest_pose_weights))

    # ----- menus / shortcuts -----
    def _build_actions(self):
        menubar = self.menuBar()
        m_file = menubar.addMenu("&File")
        m_nav = menubar.addMenu("&Navigate")
        m_tools = menubar.addMenu("&Tools")
        m_model = menubar.addMenu("&Model")

        act_save = QAction("Save", self)
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self.save_all_labeling_frames)

        act_open_proj = QAction("Open Project…", self)
        act_open_proj.setShortcut(QKeySequence.Open)
        act_open_proj.triggered.connect(self.open_project_dialog)

        act_open_images = QAction("Open Images Folder…", self)
        act_open_images.triggered.connect(self.open_images_folder)

        act_export = QAction("Export dataset.yaml + splits…", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.export_dataset_dialog)

        act_proj = QAction("Project Settings…", self)
        act_proj.setShortcut(QKeySequence("Ctrl+P"))
        act_proj.triggered.connect(self.open_project_settings)

        act_prev = QAction("Prev Frame", self)
        act_prev.setShortcut(QKeySequence("A"))
        act_prev.triggered.connect(self.prev_frame)

        act_next = QAction("Next Frame", self)
        act_next.setShortcut(QKeySequence("D"))
        act_next.triggered.connect(self.next_frame)

        act_prev_k = QAction("Prev Keypoint", self)
        act_prev_k.setShortcut(QKeySequence("Q"))
        act_prev_k.triggered.connect(self.prev_keypoint)

        act_next_k = QAction("Next Keypoint", self)
        act_next_k.setShortcut(QKeySequence("E"))
        act_next_k.triggered.connect(self.next_keypoint)

        act_next_unl = QAction("Next Unlabeled", self)
        act_next_unl.setShortcut(QKeySequence("Ctrl+F"))
        act_next_unl.triggered.connect(self.next_unlabeled)

        act_skel = QAction("Skeleton Editor", self)
        act_skel.setShortcut(QKeySequence("Ctrl+G"))
        act_skel.triggered.connect(self.open_skeleton_editor)

        act_vis = QAction("Click Visible", self)
        act_vis.setShortcut(QKeySequence("V"))
        act_occ = QAction("Click Occluded", self)
        act_occ.setShortcut(QKeySequence("O"))
        act_mis = QAction("Click Missing", self)
        act_mis.setShortcut(QKeySequence("N"))
        act_vis.triggered.connect(lambda: self._set_vis_radio(2))
        act_occ.triggered.connect(lambda: self._set_vis_radio(1))
        act_mis.triggered.connect(lambda: self._set_vis_radio(0))

        act_clear = QAction("Clear Current Keypoint", self)
        act_clear.setShortcut(QKeySequence(Qt.Key_Delete))
        act_clear.triggered.connect(self.clear_current_keypoint)

        act_clear_all = QAction("Clear All Keypoints", self)
        act_clear_all.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        act_clear_all.triggered.connect(self.clear_all_keypoints)

        act_undo = QAction("Undo", self)
        act_undo.setShortcut(QKeySequence.Undo)
        act_undo.triggered.connect(self.undo_last)

        act_fit = QAction("Fit to View", self)
        act_fit.setShortcut(QKeySequence("Ctrl+0"))
        act_fit.triggered.connect(self.fit_to_view)

        self.act_enhance = QAction("Enhance Contrast (CLAHE)", self)
        self.act_enhance.setCheckable(True)
        self.act_enhance.setChecked(bool(self.project.enhance_enabled))
        self.act_enhance.setShortcut(QKeySequence("Ctrl+H"))
        self.act_enhance.triggered.connect(
            lambda checked: self._toggle_enhancement(checked)
        )

        self.act_enhance_settings = QAction("Enhancement Settings…", self)
        self.act_enhance_settings.triggered.connect(self._open_enhancement_settings)

        self.act_show_pred_conf = QAction("Show Prediction Confidence", self)
        self.act_show_pred_conf.setCheckable(True)
        self.act_show_pred_conf.setChecked(bool(self.show_pred_conf))
        self.act_show_pred_conf.triggered.connect(self._toggle_show_pred_conf)

        act_train = QAction("Training Runner…", self)
        act_train.triggered.connect(self.open_training_runner)

        act_eval = QAction("Evaluation Dashboard…", self)
        act_eval.triggered.connect(self.open_evaluation_dashboard)

        act_active = QAction("Active Learning…", self)
        act_active.triggered.connect(self.open_active_learning)

        m_file.addAction(act_save)
        m_file.addAction(act_open_proj)
        m_file.addAction(act_open_images)
        m_file.addAction(act_proj)
        m_file.addSeparator()
        m_file.addAction(act_export)

        m_nav.addAction(act_prev)
        m_nav.addAction(act_next)
        m_nav.addAction(act_prev_k)
        m_nav.addAction(act_next_k)
        m_nav.addSeparator()
        m_nav.addSeparator()
        m_nav.addAction(act_next_unl)

        m_tools.addAction(act_skel)
        m_tools.addAction(act_vis)
        m_tools.addAction(act_occ)
        m_tools.addAction(act_mis)
        m_tools.addAction(act_clear)
        m_tools.addAction(act_clear_all)
        m_tools.addSeparator()
        m_tools.addAction(act_undo)
        m_tools.addSeparator()
        m_tools.addAction(act_fit)
        m_tools.addSeparator()
        m_tools.addAction(self.act_enhance)
        m_tools.addAction(self.act_enhance_settings)
        m_tools.addSeparator()
        m_tools.addAction(self.act_show_pred_conf)

        m_model.addAction(act_train)
        m_model.addAction(act_eval)
        m_model.addAction(act_active)

        tb = QToolBar("Main", self)
        self.addToolBar(tb)
        tb.addAction(act_prev)
        tb.addAction(act_next)
        tb.addAction(act_save)
        tb.addAction(act_next_unl)
        tb.addSeparator()
        tb.addAction(act_skel)
        tb.addAction(act_proj)
        tb.addAction(act_export)
        tb.addSeparator()
        tb.addAction(self.act_enhance)
        tb.addSeparator()
        tb.addAction(act_undo)

    # ----- file paths -----
    def _label_path_for(self, img_path: Path) -> Path:
        return self.project.labels_dir / (img_path.stem + ".txt")

    def _is_labeled(self, img_path: Path) -> bool:
        lp = self._label_path_for(img_path)
        if not lp.exists():
            return False
        return bool(lp.read_text(encoding="utf-8").strip())

    def _count_labeled_kpts(self, img_path: Path) -> int:
        try:
            loaded = load_yolo_pose_label(
                self._label_path_for(img_path), len(self.project.keypoint_names)
            )
        except Exception:
            return 0
        if loaded is None:
            return 0
        _cls, kpts_norm, _bbox = loaded
        return sum(1 for kp in kpts_norm if kp.v > 0)

    def save_project(self: object) -> object:
        """save_project method documentation."""
        if hasattr(self.project, "labeling_frames"):
            self.project.labeling_frames = sorted(
                {int(i) for i in self.labeling_frames}
            )
        self.project.last_index = self.current_index
        self.project.project_path.write_text(
            json.dumps(self.project.to_json(), indent=2), encoding="utf-8"
        )

    # ----- list / info -----
    def _populate_frames(self):
        self._suppress_list_rebuild = True
        self.labeling_list.blockSignals(True)
        self.frame_list.blockSignals(True)
        self.labeling_list.setUpdatesEnabled(False)
        self.frame_list.setUpdatesEnabled(False)
        self.labeling_list.clear()
        self.frame_list.clear()
        self._list_items = {}

        query = self.search_edit.text().strip().lower()
        indices = []
        for idx, img_path in enumerate(self.image_paths):
            if query and query not in img_path.name.lower():
                continue
            indices.append(idx)

        conf_map = self._get_pred_conf_for_indices(indices)
        kpt_map = self._get_pred_kpt_count_for_indices(indices)
        cluster_map = self._get_cluster_id_for_indices(indices)
        sort_mode = (
            self.sort_combo.currentText().strip()
            if hasattr(self, "sort_combo")
            else "Default"
        )
        reverse = sort_mode in {
            "Pred conf (high to low)",
            "Detected kpts (high to low)",
            "Cluster id (high to low)",
        }

        items = []
        for idx in indices:
            img_path = self.image_paths[idx]
            # Check if saved to disk
            is_saved = self._is_labeled(img_path)
            in_cache = idx in self._frame_cache

            # Determine marker: tick (saved), asterisk (modified but unsaved), or empty
            if in_cache and not is_saved:
                tick = "* "  # Modified but not saved
            elif is_saved:
                tick = "✓ "  # Saved to disk
            else:
                tick = "  "  # No changes

            # Count labeled keypoints - prefer cache over disk
            num_labeled = 0
            total_kpts = len(self.project.keypoint_names)
            if in_cache:
                # Use cache if available (most up-to-date)
                cached = self._frame_cache[idx]
                num_labeled = sum(1 for kp in cached.kpts if kp.v > 0)
            elif is_saved:
                num_labeled = self._count_labeled_kpts(img_path)

            # Determine color: Green=all labeled, Orange=some labeled, White=none
            if num_labeled == total_kpts:
                color = QColor(0, 200, 0)  # Green
            elif num_labeled > 0:
                color = QColor(255, 165, 0)  # Orange
            else:
                color = QColor(220, 220, 220)  # White

            item_text = f"{tick}{img_path.name}"
            pred_conf = conf_map.get(idx)
            pred_kpt_count = kpt_map.get(idx)
            cluster_id = cluster_map.get(idx)
            items.append(
                {
                    "idx": idx,
                    "is_saved": is_saved,
                    "in_labeling": idx in self.labeling_frames,
                    "item_text": item_text,
                    "color": color,
                    "pred_conf": pred_conf,
                    "pred_kpt_count": pred_kpt_count,
                    "cluster_id": cluster_id,
                }
            )

        if sort_mode != "Default":
            if sort_mode.startswith("Pred conf"):
                items.sort(
                    key=lambda it: (
                        it["pred_conf"]
                        if it["pred_conf"] is not None
                        else (-1.0 if reverse else 1e9)
                    ),
                    reverse=reverse,
                )
            elif sort_mode.startswith("Detected kpts"):
                items.sort(
                    key=lambda it: (
                        it["pred_kpt_count"]
                        if it["pred_kpt_count"] is not None
                        else (-1 if reverse else 1_000_000)
                    ),
                    reverse=reverse,
                )
            elif sort_mode.startswith("Cluster id"):
                items.sort(
                    key=lambda it: (
                        it["cluster_id"]
                        if it["cluster_id"] is not None
                        else (-1 if reverse else 1_000_000)
                    ),
                    reverse=reverse,
                )

        for it in items:
            idx = it["idx"]
            # Labeled frames always go to labeling list
            if it["is_saved"] or it["in_labeling"]:
                if it["is_saved"]:
                    self.labeling_frames.add(idx)
                item = QListWidgetItem(it["item_text"])
                item.setData(Qt.UserRole, idx)
                item.setData(FrameListDelegate.CONF_ROLE, it["pred_conf"])
                item.setData(FrameListDelegate.KP_COUNT_ROLE, it["pred_kpt_count"])
                item.setData(FrameListDelegate.CLUSTER_ROLE, it["cluster_id"])
                item.setForeground(it["color"])
                self.labeling_list.addItem(item)
                self._list_items[idx] = item
            else:
                item = QListWidgetItem(it["item_text"])
                item.setData(Qt.UserRole, idx)
                item.setData(FrameListDelegate.CONF_ROLE, it["pred_conf"])
                item.setData(FrameListDelegate.KP_COUNT_ROLE, it["pred_kpt_count"])
                item.setData(FrameListDelegate.CLUSTER_ROLE, it["cluster_id"])
                item.setForeground(it["color"])
                self.frame_list.addItem(item)
                self._list_items[idx] = item

        self.labeling_list.blockSignals(False)
        self.frame_list.blockSignals(False)
        self.labeling_list.setUpdatesEnabled(True)
        self.frame_list.setUpdatesEnabled(True)
        self._suppress_list_rebuild = False

    def _update_frame_item(
        self,
        idx: int,
        pred_conf: Optional[float] = None,
        pred_kpt_count: Optional[int] = None,
        cluster_id: Optional[int] = None,
        conf_only: bool = False,
    ):
        """Update a single frame item in the lists without rebuilding everything."""
        if conf_only:
            item = self._list_items.get(idx)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
                if cluster_id is not None:
                    item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)
            return
        img_path = self.image_paths[idx]
        is_saved = self._is_labeled(img_path)
        in_cache = idx in self._frame_cache

        # Determine marker
        if in_cache and not is_saved:
            tick = "* "  # Modified but not saved
        elif is_saved:
            tick = "✓ "  # Saved to disk
        else:
            tick = "  "  # No changes

        # Count labeled keypoints
        num_labeled = 0
        total_kpts = len(self.project.keypoint_names)
        if in_cache:
            cached = self._frame_cache[idx]
            num_labeled = sum(1 for kp in cached.kpts if kp.v > 0)
        elif is_saved:
            num_labeled = self._count_labeled_kpts(img_path)

        # Determine color
        if num_labeled == total_kpts:
            color = QColor(0, 200, 0)  # Green
        elif num_labeled > 0:
            color = QColor(255, 165, 0)  # Orange
        else:
            color = QColor(220, 220, 220)  # White

        item_text = f"{tick}{img_path.name}"
        if pred_conf is None:
            pred_conf = self._get_pred_conf_for_indices([idx]).get(idx)
        if pred_kpt_count is None:
            pred_kpt_count = self._get_pred_kpt_count_for_indices([idx]).get(idx)
        if cluster_id is None:
            cluster_id = self._get_cluster_id_for_indices([idx]).get(idx)

        # Find and update the item in the appropriate list
        item = self._list_items.get(idx)
        if item is not None:
            item.setForeground(color)
            item.setText(item_text)
            item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
            item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
            item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)
            return

        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            if item.data(Qt.UserRole) == idx:
                item.setForeground(color)
                item.setText(item_text)
                item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
                item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)
                return

        for i in range(self.frame_list.count()):
            item = self.frame_list.item(i)
            if item.data(Qt.UserRole) == idx:
                item.setForeground(color)
                item.setText(item_text)
                item.setData(FrameListDelegate.CONF_ROLE, pred_conf)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, pred_kpt_count)
                item.setData(FrameListDelegate.CLUSTER_ROLE, cluster_id)
                return

    def _clear_conf_display(self) -> None:
        if self._list_items:
            for item in self._list_items.values():
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)
            return
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)
        for i in range(self.frame_list.count()):
            item = self.frame_list.item(i)
            if item is not None:
                item.setData(FrameListDelegate.CONF_ROLE, None)
                item.setData(FrameListDelegate.KP_COUNT_ROLE, None)

    def _rebuild_path_index(self):
        self._path_to_index: Dict[str, int] = {}
        for i, p in enumerate(self.image_paths):
            try:
                self._path_to_index[str(p)] = i
                self._path_to_index[str(p.resolve())] = i
            except Exception:
                self._path_to_index[str(p)] = i

    def _get_pred_conf_for_indices(self, indices: List[int]) -> Dict[int, Optional[float]]:
        self._ensure_pred_conf_cache()
        out: Dict[int, Optional[float]] = {}
        for idx in indices:
            img_path = self.image_paths[idx]
            key = str(img_path)
            val = self._pred_conf_map.get(key)
            if val is None:
                try:
                    val = self._pred_conf_map.get(str(Path(img_path).resolve()))
                except Exception:
                    val = None
            out[idx] = val
        return out

    def _get_pred_kpt_count_for_indices(
        self, indices: List[int]
    ) -> Dict[int, Optional[int]]:
        self._ensure_pred_conf_cache()
        out: Dict[int, Optional[int]] = {}
        for idx in indices:
            img_path = self.image_paths[idx]
            key = str(img_path)
            val = self._pred_kpt_count_map.get(key)
            if val is None:
                try:
                    val = self._pred_kpt_count_map.get(
                        str(Path(img_path).resolve())
                    )
                except Exception:
                    val = None
            out[idx] = val
        return out

    def _ensure_cluster_ids_cache(self) -> None:
        csv_path = self.project.out_root / "posekit" / "clusters" / "clusters.csv"
        if not csv_path.exists():
            self._cluster_ids_cache = None
            self._cluster_ids_mtime = None
            return
        try:
            mtime = float(csv_path.stat().st_mtime)
        except Exception:
            mtime = None
        refreshed = not (
            self._cluster_ids_cache is not None
            and self._cluster_ids_mtime == mtime
            and len(self._cluster_ids_cache) == len(self.image_paths)
        )
        if not refreshed:
            return
        cluster_ids = self._load_cluster_ids_from_csv(csv_path)
        self._cluster_ids_cache = cluster_ids
        self._cluster_ids_mtime = mtime
        if not self._suppress_list_rebuild and self._list_items:
            self._refresh_cluster_roles_all()

    def _get_cluster_id_for_indices(
        self, indices: List[int]
    ) -> Dict[int, Optional[int]]:
        self._ensure_cluster_ids_cache()
        out: Dict[int, Optional[int]] = {}
        for idx in indices:
            cid = None
            if self._cluster_ids_cache and idx < len(self._cluster_ids_cache):
                cid_val = self._cluster_ids_cache[idx]
                cid = cid_val if cid_val is not None and cid_val >= 0 else None
            out[idx] = cid
        return out

    def _refresh_cluster_roles_all(self) -> None:
        if not self._cluster_ids_cache:
            return
        for idx, item in self._list_items.items():
            if idx < 0 or idx >= len(self.image_paths):
                continue
            cid_val = None
            if idx < len(self._cluster_ids_cache):
                raw = self._cluster_ids_cache[idx]
                cid_val = raw if raw is not None and raw >= 0 else None
            item.setData(FrameListDelegate.CLUSTER_ROLE, cid_val)

    def _pred_cache_key_for(self, model: Optional[Path], backend: str) -> Optional[str]:
        if not model:
            return None
        try:
            sig = self.infer._model_sig(model, backend)
        except Exception:
            sig = None
        conf_thr = self._pred_conf_default()
        return f"{backend}|{model}|{sig}|thr={conf_thr:.6f}"

    def _current_pred_cache_key(self) -> Optional[str]:
        model = self._get_pred_model_silent()
        backend = self._pred_backend()
        return self._pred_cache_key_for(model, backend)

    def _ensure_pred_conf_cache(self) -> None:
        key = self._current_pred_cache_key()
        if not key:
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            return
        if key == self._pred_conf_cache_key and self._pred_conf_complete:
            return
        model = self._get_pred_model_silent()
        if not model:
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            return
        backend = self._pred_backend()
        preds_cache = self.infer.load_cache(model, backend=backend)
        self._pred_conf_cache_key = key
        conf_map, kpt_map = self._build_pred_stats_maps(preds_cache)
        self._pred_conf_map = conf_map
        self._pred_kpt_count_map = kpt_map
        self._pred_conf_complete = True

    def _build_pred_stats_maps(
        self, preds_cache: Dict[str, List[Tuple[float, float, float]]]
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[int]]]:
        conf_map: Dict[str, Optional[float]] = {}
        kpt_map: Dict[str, Optional[int]] = {}
        conf_thr = self._pred_conf_default()
        for path, pred_list in preds_cache.items():
            confs = [
                float(p[2])
                for p in pred_list
                if len(p) >= 3 and np.isfinite(p[2])
            ]
            val = float(np.mean(confs)) if confs else None
            if confs:
                kpt_count = int(sum(1 for c in confs if c > conf_thr))
            else:
                kpt_count = 0 if pred_list else None
            conf_map[path] = val
            kpt_map[path] = kpt_count
            try:
                resolved = str(Path(path).resolve())
                conf_map[resolved] = val
                kpt_map[resolved] = kpt_count
            except Exception:
                pass
        return conf_map, kpt_map

    def _update_pred_conf_map_from_preds(
        self,
        preds: Dict[str, List[Tuple[float, float, float]]],
        model: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> None:
        if not preds:
            return
        if backend is None:
            backend = self._pred_backend()
        if model is None:
            model = self._get_pred_model_silent()
        key = self._pred_cache_key_for(model, backend)
        if not key:
            return
        if key != self._pred_conf_cache_key:
            self._pred_conf_cache_key = key
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
        conf_thr = self._pred_conf_default()
        for path, pred_list in preds.items():
            confs = [
                float(p[2])
                for p in pred_list
                if len(p) >= 3 and np.isfinite(p[2])
            ]
            val = float(np.mean(confs)) if confs else None
            if confs:
                kpt_count = int(sum(1 for c in confs if c > conf_thr))
            else:
                kpt_count = 0 if pred_list else None
            self._pred_conf_map[path] = val
            self._pred_kpt_count_map[path] = kpt_count
            try:
                resolved = str(Path(path).resolve())
                self._pred_conf_map[resolved] = val
                self._pred_kpt_count_map[resolved] = kpt_count
            except Exception:
                pass

    def _rebuild_kpt_list(self):
        self.kpt_list.clear()
        for i, nm in enumerate(self.project.keypoint_names):
            self.kpt_list.addItem(f"{i}: {nm}")
        self.kpt_list.setCurrentRow(min(self.current_kpt, self.kpt_list.count() - 1))

    def _update_info(self):
        img_path = self.image_paths[self.current_index]
        labeled = self._is_labeled(img_path)
        done = sum(1 for kp in self._ann.kpts if kp.v > 0) if self._ann else 0
        total = len(self.project.keypoint_names)
        self.lbl_info.setText(
            f"{img_path.name}\n"
            f"Frame {self.current_index + 1}/{len(self.image_paths)} | "
            f"Keypoints {done}/{total} | "
            f"{'LABELED' if labeled else 'unlabeled'}"
        )
        self.statusBar().showMessage(f"{img_path.name} ({done}/{total} kpts)")

    # ----- load frame -----
    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return img

    def _clone_ann(self, ann: FrameAnn) -> FrameAnn:
        return FrameAnn(
            cls=int(ann.cls),
            bbox_xyxy=ann.bbox_xyxy,
            kpts=[Keypoint(kp.x, kp.y, kp.v) for kp in ann.kpts],
        )

    def _cache_current_frame(self):
        if self._ann is None:
            return
        self._frame_cache[self.current_index] = self._clone_ann(self._ann)
        logger.debug(
            "Cached frame %d (kpts=%d)", self.current_index, len(self._ann.kpts)
        )
        # Update the frame item display to reflect cache state
        self._update_frame_item(self.current_index)

    def _prime_cache_for_labeling(self):
        if not self.labeling_frames:
            return
        logger.debug("Priming cache for %d labeling frames", len(self.labeling_frames))
        for idx in sorted(self.labeling_frames):
            if idx in self._frame_cache:
                continue
            try:
                ann = self._load_ann_from_disk(idx)
            except Exception:
                continue
            self._frame_cache[idx] = self._clone_ann(ann)
        logger.debug("Cache size after prime: %d", len(self._frame_cache))

    def _load_ann_from_disk(self, idx: int) -> FrameAnn:
        img_path = self.image_paths[idx]
        img = self._read_image(img_path)
        h, w = img.shape[:2]
        k = len(self.project.keypoint_names)
        kpts = [Keypoint(0.0, 0.0, 0) for _ in range(k)]
        bbox = None
        cls = int(self.class_combo.currentIndex())

        loaded = load_yolo_pose_label(self._label_path_for(img_path), k)
        if loaded is not None:
            cls_loaded, kpts_norm, bbox_cxcywh = loaded
            cls = int(cls_loaded)
            kpts = []
            for kp in kpts_norm:
                if kp.v == 0:
                    kpts.append(Keypoint(0.0, 0.0, 0))
                else:
                    kpts.append(Keypoint(kp.x * w, kp.y * h, int(kp.v)))

            if bbox_cxcywh is not None:
                cx, cy, bw, bh = bbox_cxcywh
                cx *= w
                cy *= h
                bw *= w
                bh *= h
                x1 = cx - bw / 2
                x2 = cx + bw / 2
                y1 = cy - bh / 2
                y2 = cy + bh / 2
                bbox = (x1, y1, x2, y2)

        return FrameAnn(cls=cls, bbox_xyxy=bbox, kpts=kpts)

    def load_frame(self: object, idx: int) -> object:
        """load_frame method documentation."""
        idx = max(0, min(idx, len(self.image_paths) - 1))

        logger.debug("Load frame requested: idx=%d", idx)

        # Cache current annotations before switching
        self._cache_current_frame()

        self.current_index = idx
        img_path = self.image_paths[idx]
        self._img_bgr = self._read_image(img_path)
        self._img_display = None
        h, w = self._img_bgr.shape[:2]
        self._img_wh = (w, h)

        if idx in self._frame_cache:
            cached = self._frame_cache[idx]
            cls = int(cached.cls)
            kpts = [Keypoint(kp.x, kp.y, kp.v) for kp in cached.kpts]
            bbox = cached.bbox_xyxy
            logger.debug("Loaded frame %d from cache", idx)
        else:
            ann = self._load_ann_from_disk(idx)
            cls = int(ann.cls)
            kpts = ann.kpts
            bbox = ann.bbox_xyxy
            if self.mode == "keypoint":
                self._frame_cache[idx] = self._clone_ann(ann)
            logger.debug(
                "Loaded frame %d from disk (cached=%s)", idx, self.mode == "keypoint"
            )

        self.class_combo.blockSignals(True)
        self.class_combo.setCurrentIndex(max(0, min(cls, self.class_combo.count() - 1)))
        self.class_combo.blockSignals(False)

        self._ann = FrameAnn(cls=cls, bbox_xyxy=bbox, kpts=kpts)
        self._dirty = False

        # Clear undo stack when changing frames to prevent cross-frame undo
        self._undo_stack.clear()

        # In keypoint mode, auto-switch to first unlabeled keypoint
        if self.mode == "keypoint":
            first_unlabeled = None
            for i, kp in enumerate(self._ann.kpts):
                if kp.v == 0:
                    first_unlabeled = i
                    break

            if first_unlabeled is not None and first_unlabeled != self.current_kpt:
                # Notify user that we're switching keypoints
                old_kpt_name = self.project.keypoint_names[self.current_kpt]
                new_kpt_name = self.project.keypoint_names[first_unlabeled]
                self.statusBar().showMessage(
                    f"Switched from '{old_kpt_name}' to unlabeled '{new_kpt_name}'",
                    3000,
                )
                self.current_kpt = first_unlabeled
                self.kpt_list.setCurrentRow(first_unlabeled)

        self.canvas.set_current_keypoint(self.current_kpt)
        self.canvas.set_click_visibility(self.click_vis)
        self._refresh_canvas_image()
        self._rebuild_canvas()
        self._update_info()
        self._load_metadata_ui()

    # ----- events -----
    def _on_labeling_frame_selected(self, row: int):
        if row < 0:
            return
        # Get the actual index from the item
        item = self.labeling_list.item(row)
        if item:
            actual_idx = item.data(Qt.UserRole)
            if actual_idx is not None:
                logger.debug("Labeling frame selected: row=%s idx=%s", row, actual_idx)
                # Ensure frame is in labeling_frames set (for navigation to work)
                self.labeling_frames.add(actual_idx)
                self._maybe_autosave()
                self.load_frame(actual_idx)
                # Deselect all frames list
                self.frame_list.clearSelection()

    def _on_all_frame_selected(self, row: int):
        if row < 0:
            return
        # Get the actual index from the item
        item = self.frame_list.item(row)
        if item:
            actual_idx = item.data(Qt.UserRole)
            if actual_idx is not None:
                logger.debug("All frame selected: row=%s idx=%s", row, actual_idx)
                self._maybe_autosave()
                self.load_frame(actual_idx)
                # Deselect labeling list
                self.labeling_list.clearSelection()

    def _on_labeling_list_changed(self, parent, start, end, dest, row):
        """Update labeling_frames when items are moved."""
        if self._suppress_list_rebuild:
            logger.debug("Labeling list rowsMoved suppressed")
            return
        self._rebuild_labeling_set()

    def _on_all_list_changed(self, parent, start, end, dest, row):
        """Update labeling_frames when items are moved."""
        if self._suppress_list_rebuild:
            logger.debug("All list rowsMoved suppressed")
            return
        self._rebuild_labeling_set()

    def _rebuild_labeling_set(self):
        """Rebuild the labeling_frames set from current list contents."""
        logger.debug("Rebuild labeling set (before): %s", sorted(self.labeling_frames))
        self.labeling_frames.clear()
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            idx = item.data(Qt.UserRole)
            if idx is not None:
                self.labeling_frames.add(idx)

        # Ensure labeled frames are always in labeling set (prevent drag-out)
        for idx, img_path in enumerate(self.image_paths):
            if self._is_labeled(img_path):
                self.labeling_frames.add(idx)

        # Repopulate to enforce labeled frames stay in labeling list
        self._populate_frames()
        logger.debug("Rebuild labeling set (after): %s", sorted(self.labeling_frames))

    def _move_unlabeled_to_labeling(self):
        """Move all unlabeled frames from all frames to labeling frames."""
        for idx, img_path in enumerate(self.image_paths):
            if not self._is_labeled(img_path) and idx not in self.labeling_frames:
                self.labeling_frames.add(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

    def _move_unlabeled_to_all(self):
        """Move unlabeled frames from labeling to all frames."""
        unlabeled_to_remove = []
        for idx in list(self.labeling_frames):
            if not self._is_labeled(self.image_paths[idx]):
                unlabeled_to_remove.append(idx)
        for idx in unlabeled_to_remove:
            self.labeling_frames.remove(idx)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

    def _add_random_to_labeling(self):
        """Add random unlabeled frames from All Frames list to labeling set."""
        import random

        count = self.spin_random_count.value()

        # Get all unlabeled frames from All Frames list (not in labeling set)
        candidates = []
        for idx, img_path in enumerate(self.image_paths):
            if not self._is_labeled(img_path) and idx not in self.labeling_frames:
                candidates.append(idx)

        if not candidates:
            QMessageBox.information(
                self, "No frames", "No unlabeled frames available in All Frames list."
            )
            return

        # Randomly select up to 'count' frames
        to_add = random.sample(candidates, min(count, len(candidates)))
        for idx in to_add:
            self.labeling_frames.add(idx)

        self._populate_frames()
        self._select_frame_in_list(self.current_index)
        QMessageBox.information(
            self, "Added frames", f"Added {len(to_add)} frames to labeling set."
        )

    def _on_kpt_selected(self, row: int):
        if row < 0:
            return
        self.current_kpt = row
        self.canvas.set_current_keypoint(row)

    def _update_mode(self):
        prev_mode = self.mode
        self.mode = "frame" if self.rb_frame.isChecked() else "keypoint"
        logger.debug("Mode update: %s -> %s", prev_mode, self.mode)
        if self.mode == "keypoint" and prev_mode != "keypoint":
            # Prime cache for keypoint-by-keypoint workflow
            logger.debug(
                "Priming cache for keypoint mode. labeling_frames=%d",
                len(self.labeling_frames),
            )
            self._cache_current_frame()
            self._prime_cache_for_labeling()

    def _update_vis_mode(self):
        self.click_vis = int(self.vis_group.checkedId())
        self.canvas.set_click_visibility(self.click_vis)

    def _toggle_enhancement(self, checked: bool):
        self.project.enhance_enabled = bool(checked)
        if self.cb_enhance.isChecked() != self.project.enhance_enabled:
            self.cb_enhance.setChecked(self.project.enhance_enabled)
        if self.act_enhance.isChecked() != self.project.enhance_enabled:
            self.act_enhance.setChecked(self.project.enhance_enabled)
        self._img_display = None
        self._refresh_canvas_image()
        self.save_project()

    def _update_kpt_size(self, value: float):
        self.project.kpt_radius = float(value)
        self.canvas.set_kpt_radius(self.project.kpt_radius)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_label_size(self, value: int):
        self.project.label_font_size = int(value)
        self.canvas.set_label_font_size(self.project.label_font_size)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_kpt_opacity(self, value: float):
        self.project.kpt_opacity = float(value)
        self.canvas.set_kpt_opacity(self.project.kpt_opacity)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def _update_edge_opacity(self, value: float):
        self.project.edge_opacity = float(value)
        self.canvas.set_edge_opacity(self.project.edge_opacity)
        if self._ann is not None:
            self._rebuild_canvas()
        self.save_project()

    def fit_to_view(self: object) -> object:
        """Fit image to view."""
        self.canvas.fit_to_view()

    def _get_display_image(self) -> Optional[np.ndarray]:
        if self._img_bgr is None:
            return None
        if self._img_display is not None:
            return self._img_display
        if not self.project.enhance_enabled:
            self._img_display = self._img_bgr
            return self._img_display

        try:
            self._img_display = enhance_for_pose(
                self._img_bgr,
                clahe_clip=self.project.clahe_clip,
                clahe_grid=self.project.clahe_grid,
                sharpen_amt=self.project.sharpen_amt,
                blur_sigma=self.project.blur_sigma,
            )
        except Exception:
            self._img_display = self._img_bgr
        return self._img_display

    def _refresh_canvas_image(self):
        img = self._get_display_image()
        if img is None:
            return
        self.canvas.set_image(img)

    def _open_enhancement_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Enhancement Settings")
        layout = QFormLayout(dlg)

        clip = QDoubleSpinBox()
        clip.setRange(0.1, 10.0)
        clip.setSingleStep(0.1)
        clip.setValue(float(self.project.clahe_clip))

        grid_x = QSpinBox()
        grid_x.setRange(2, 64)
        grid_x.setValue(int(self.project.clahe_grid[0]))

        grid_y = QSpinBox()
        grid_y.setRange(2, 64)
        grid_y.setValue(int(self.project.clahe_grid[1]))

        sharpen = QDoubleSpinBox()
        sharpen.setRange(0.0, 3.0)
        sharpen.setSingleStep(0.1)
        sharpen.setValue(float(self.project.sharpen_amt))

        blur = QDoubleSpinBox()
        blur.setRange(0.0, 5.0)
        blur.setSingleStep(0.1)
        blur.setValue(float(self.project.blur_sigma))

        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("X"))
        grid_row.addWidget(grid_x)
        grid_row.addWidget(QLabel("Y"))
        grid_row.addWidget(grid_y)

        layout.addRow("What should clahe clip be?", clip)
        layout.addRow("What should clahe grid be?", grid_row)
        layout.addRow("What should sharpen amount be?", sharpen)
        layout.addRow("What should blur sigma be?", blur)

        btns = QHBoxLayout()
        ok = QPushButton("Apply")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addRow(btns)

        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)

        if dlg.exec() != QDialog.Accepted:
            return

        self.project.clahe_clip = float(clip.value())
        self.project.clahe_grid = (int(grid_x.value()), int(grid_y.value()))
        self.project.sharpen_amt = float(sharpen.value())
        self.project.blur_sigma = float(blur.value())
        self._img_display = None
        self._refresh_canvas_image()
        self.save_project()

    def _set_vis_radio(self, v: int):
        if v == 2:
            self.rb_vis.setChecked(True)
        elif v == 1:
            self.rb_occ.setChecked(True)
        else:
            self.rb_miss.setChecked(True)

    def _set_autosave_status(self, text: str):
        if hasattr(self, "status_autosave") and self.status_autosave is not None:
            self.status_autosave.setText(text)

    def _set_saved_status(self):
        ts = datetime.now().strftime("%H:%M:%S")
        self._last_saved_at = ts
        self._set_autosave_status(f"Saved {ts}")

    def _set_dirty(self):
        self._dirty = True
        self._set_autosave_status("Unsaved changes…")

    def _mark_dirty(self, *_):
        self._set_dirty()
        self._schedule_autosave()

    # ----- undo -----
    def _snapshot_kpts(self) -> Optional[List[Keypoint]]:
        if self._ann is None:
            return None
        return [Keypoint(kp.x, kp.y, kp.v) for kp in self._ann.kpts]

    def _push_undo(self):
        snap = self._snapshot_kpts()
        if snap is None:
            return
        self._undo_stack.append(snap)
        if len(self._undo_stack) > self._undo_max:
            self._undo_stack = self._undo_stack[-self._undo_max :]

    def undo_last(self: object) -> object:
        """undo_last method documentation."""
        if not self._undo_stack or self._ann is None:
            return
        self._ann.kpts = self._undo_stack.pop()
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()

    # ----- edits -----
    def on_place_kpt(self: object, kpt_idx: int, x: float, y: float, v: int) -> object:
        """on_place_kpt method documentation."""
        if self._ann is None:
            return
        logger.debug(
            "Place kpt: idx=%d v=%d at (%.1f, %.1f) mode=%s frame=%d",
            kpt_idx,
            v,
            x,
            y,
            self.mode,
            self.current_index,
        )
        w, h = self._img_wh
        kpt_idx = max(0, min(kpt_idx, len(self._ann.kpts) - 1))
        # In frame-by-frame mode, enforce sequential keypoint placement
        if self.mode == "frame" and kpt_idx > 0:
            # Check if all previous keypoints have been placed
            for i in range(kpt_idx):
                if self._ann.kpts[i].v == 0:
                    QMessageBox.warning(
                        self,
                        "Sequential Labeling",
                        f"Please label keypoint {i} ({self.project.keypoint_names[i]}) before keypoint {kpt_idx}.",
                    )
                    return

        self._push_undo()

        if v == 0:
            self._ann.kpts[kpt_idx] = Keypoint(0.0, 0.0, 0)
        else:
            self._ann.kpts[kpt_idx] = Keypoint(
                x=max(0.0, min(float(w - 1), x)),
                y=max(0.0, min(float(h - 1), y)),
                v=int(v),
            )
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()

        if self.mode == "keypoint":
            # In keypoint mode, keep in-memory cache only
            self._cache_current_frame()
            # Frame item is already updated by _cache_current_frame
            # Find next frame that needs this keypoint
            self._advance_keypoint_mode()
        else:
            # Save immediately after each placement in frame mode
            # Don't refresh UI since we're about to navigate away
            self.save_current(refresh_ui=False)

            # Check if all keypoints are now labeled
            all_labeled = all(kp.v > 0 for kp in self._ann.kpts)

            # In frame mode: if all labeled, advance to next frame
            # Otherwise, jump to first unlabeled keypoint
            if all_labeled:
                self.next_frame(prefer_missing=True)
            else:
                # Find first unlabeled keypoint
                for i, kp in enumerate(self._ann.kpts):
                    if kp.v == 0:
                        self.current_kpt = i
                        self.kpt_list.setCurrentRow(i)
                        self.canvas.set_current_keypoint(i)
                        # Force UI update to prevent race condition with next click
                        QApplication.processEvents()
                        break

    def _advance_keypoint_mode(self):
        """Find next frame/keypoint to label in keypoint-by-keypoint mode."""
        if not self.labeling_frames:
            return

        # Find next frame in labeling set that needs current keypoint
        next_frame_idx = self._find_next_frame_needing_keypoint(self.current_kpt)

        if next_frame_idx is not None:
            # Found a frame that needs this keypoint
            self._select_frame_in_list(next_frame_idx)
        else:
            # All frames have this keypoint - move to next keypoint
            kpt_name = self.project.keypoint_names[self.current_kpt]

            # Check if all frames have all keypoints
            if self._all_frames_fully_labeled():
                QMessageBox.information(
                    self,
                    "Labeling Complete",
                    "All keypoints have been labeled on all frames in the labeling set!",
                )
                return

            # Move to next keypoint
            next_kpt = (self.current_kpt + 1) % len(self.project.keypoint_names)
            self.current_kpt = next_kpt
            self.kpt_list.setCurrentRow(next_kpt)
            self.canvas.set_current_keypoint(next_kpt)
            # Force UI update to prevent race condition
            QApplication.processEvents()

            next_kpt_name = self.project.keypoint_names[next_kpt]
            QMessageBox.information(
                self,
                "Keypoint Complete",
                f"All frames have keypoint '{kpt_name}'.\nMoving to keypoint '{next_kpt_name}'.",
            )

            # Find first frame that needs the new keypoint
            next_frame_idx = self._find_next_frame_needing_keypoint(next_kpt)
            if next_frame_idx is not None:
                self._select_frame_in_list(next_frame_idx)

    def _find_next_frame_needing_keypoint(self, kpt_idx: int) -> Optional[int]:
        """Find next frame in labeling set that doesn't have the specified keypoint."""
        labeling_indices = sorted(self.labeling_frames)

        # Start from current frame and wrap around
        try:
            current_pos = labeling_indices.index(self.current_index)
            search_order = (
                labeling_indices[current_pos + 1 :]
                + labeling_indices[: current_pos + 1]
            )
        except ValueError:
            # Current frame not in labeling set, search all
            search_order = labeling_indices

        for idx in search_order:
            # Check cache first, then disk
            if idx in self._frame_cache:
                ann = self._frame_cache[idx]
                if ann.kpts[kpt_idx].v == 0:
                    return idx
            else:
                # Load from disk to check
                try:
                    ann = self._load_ann_from_disk(idx)
                    if ann.kpts[kpt_idx].v == 0:
                        return idx
                except Exception:
                    # If we can't load, assume it needs the keypoint
                    return idx

        return None

    def _all_frames_fully_labeled(self) -> bool:
        """Check if all frames in labeling set have all keypoints labeled."""
        num_kpts = len(self.project.keypoint_names)

        for idx in self.labeling_frames:
            if idx in self._frame_cache:
                ann = self._frame_cache[idx]
            else:
                try:
                    ann = self._load_ann_from_disk(idx)
                except Exception:
                    return False

            # Check if any keypoint is missing
            if any(kp.v == 0 for kp in ann.kpts):
                return False

        return True

    def _frame_has_missing_labels(self, idx: int) -> bool:
        """Return True if any keypoint is missing for the given frame."""
        if idx in self._frame_cache:
            ann = self._frame_cache[idx]
        else:
            try:
                ann = self._load_ann_from_disk(idx)
            except Exception:
                # If we can't load, assume it needs labeling.
                return True
        return any(kp.v == 0 for kp in ann.kpts)

    def _find_next_frame_with_missing_labels(self) -> Optional[int]:
        """Find next frame in labeling set with any missing keypoints."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            return None

        try:
            current_pos = labeling_indices.index(self.current_index)
            search_order = (
                labeling_indices[current_pos + 1 :]
                + labeling_indices[: current_pos + 1]
            )
        except ValueError:
            search_order = labeling_indices

        for idx in search_order:
            if self._frame_has_missing_labels(idx):
                return idx
        return None

    def on_move_kpt(self: object, kpt_idx: int, x: float, y: float) -> object:
        """on_move_kpt method documentation."""
        if self._ann is None:
            return
        logger.debug(
            "Move kpt: idx=%d to (%.1f, %.1f) mode=%s frame=%d",
            kpt_idx,
            x,
            y,
            self.mode,
            self.current_index,
        )
        self._push_undo()
        w, h = self._img_wh
        kp = self._ann.kpts[kpt_idx]
        if kp.v == 0:
            kp.v = 2
        kp.x = max(0.0, min(float(w - 1), x))
        kp.y = max(0.0, min(float(h - 1), y))
        self._mark_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()

    def on_select_kpt(self: object, idx: int) -> object:
        """Called when user clicks on an existing keypoint - make it current."""
        if idx >= 0 and idx < len(self.project.keypoint_names):
            self.current_kpt = idx
            self.kpt_list.setCurrentRow(idx)
            self.canvas.set_current_keypoint(idx)

    def clear_current_keypoint(self: object) -> object:
        """clear_current_keypoint method documentation."""
        self.on_place_kpt(self.current_kpt, 0.0, 0.0, 0)

    def clear_all_keypoints(self: object) -> object:
        """Clear all keypoints from the current frame."""
        if self._ann is None:
            return
        self._push_undo()
        for i in range(len(self._ann.kpts)):
            self._ann.kpts[i] = Keypoint(0.0, 0.0, 0)
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()
        else:
            self.save_current()

    # ----- navigation -----
    def _maybe_autosave(self):
        if self.mode == "keypoint":
            # In keypoint mode, keep in-memory cache only
            self._cache_current_frame()
            if self.project.autosave and self._dirty:
                self._schedule_autosave()
            return
        if self.project.autosave and self._dirty:
            self._schedule_autosave()

    def _schedule_autosave(self):
        if not self.project.autosave:
            return
        if self._dirty:
            self._autosave_timer.start(self.autosave_delay_ms)

    def _perform_autosave(self):
        if self.project.autosave and self._dirty:
            self._set_autosave_status("Autosaving…")
            self.save_current()

    def _update_autosave_delay(self, seconds: float):
        self.autosave_delay_ms = int(max(0.5, float(seconds)) * 1000)
        if self._autosave_timer.isActive():
            self._autosave_timer.start(self.autosave_delay_ms)

    def prev_frame(self: object) -> object:
        """prev_frame method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            return

        # Refresh frame lists to show updated labeling status
        prev_idx = self.current_index

        try:
            current_pos = labeling_indices.index(self.current_index)
            if current_pos > 0:
                prev_idx_target = labeling_indices[current_pos - 1]
                self._select_frame_in_list(prev_idx_target)
                # Reset to first keypoint in frame mode
                if self.mode == "frame":
                    self.current_kpt = 0
                    self.kpt_list.setCurrentRow(0)
                    self.canvas.set_current_keypoint(0)
        except ValueError:
            # Current not in labeling set, go to last
            if labeling_indices:
                self._select_frame_in_list(labeling_indices[-1])
                if self.mode == "frame":
                    self.current_kpt = 0
                    self.kpt_list.setCurrentRow(0)
                    self.canvas.set_current_keypoint(0)

        # Refresh lists after navigation to show updated status
        if prev_idx != self.current_index:
            # Block signals to prevent triggering frame load during list rebuild
            self.labeling_list.blockSignals(True)
            self.frame_list.blockSignals(True)
            self._populate_frames()
            self.labeling_list.blockSignals(False)
            self.frame_list.blockSignals(False)
            # Restore selection without triggering load (already loaded)
            for i in range(self.labeling_list.count()):
                item = self.labeling_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.labeling_list.setCurrentRow(i)
                    break

    def _toggle_kpt_visibility(self, kpt_idx: int):
        if self._ann is None:
            return
        if kpt_idx < 0 or kpt_idx >= len(self._ann.kpts):
            return
        kp = self._ann.kpts[kpt_idx]
        if kp.v <= 0:
            return
        self._push_undo()
        if kp.v == 2:
            kp.v = 1
        elif kp.v == 1:
            kp.v = 0
            kp.x = 0.0
            kp.y = 0.0
        else:
            kp.v = 2
        self._set_dirty()
        self._rebuild_canvas()
        self._update_info()
        if self.mode == "keypoint":
            self._cache_current_frame()
            for i in range(self.frame_list.count()):
                item = self.frame_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.frame_list.setCurrentRow(i)
                    break

    def next_frame(self: object, prefer_missing: bool = False) -> object:
        """next_frame method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        logger.debug(
            "next_frame: current=%d labeling_frames=%s cache_size=%d",
            self.current_index,
            labeling_indices,
            len(self._frame_cache),
        )
        if not labeling_indices:
            logger.debug("next_frame: no labeling frames, returning")
            return

        # Refresh frame lists to show updated labeling status
        prev_idx = self.current_index

        if prefer_missing and self.mode == "frame":
            next_missing = self._find_next_frame_with_missing_labels()
            if next_missing is not None:
                self._select_frame_in_list(next_missing)
                self.current_kpt = 0
                self.kpt_list.setCurrentRow(0)
                self.canvas.set_current_keypoint(0)
            else:
                self.statusBar().showMessage(
                    "All frames in labeling set are fully labeled.", 2000
                )
                return
        else:
            try:
                current_pos = labeling_indices.index(self.current_index)
                if current_pos < len(labeling_indices) - 1:
                    next_idx = labeling_indices[current_pos + 1]
                    self._select_frame_in_list(next_idx)
                    # Reset to first keypoint in frame mode
                    if self.mode == "frame":
                        self.current_kpt = 0
                        self.kpt_list.setCurrentRow(0)
                        self.canvas.set_current_keypoint(0)
            except ValueError:
                # Current not in labeling set, go to first
                if labeling_indices:
                    self._select_frame_in_list(labeling_indices[0])
                    if self.mode == "frame":
                        self.current_kpt = 0
                        self.kpt_list.setCurrentRow(0)
                        self.canvas.set_current_keypoint(0)

        # Refresh lists after navigation to show updated status
        if prev_idx != self.current_index:
            # Block signals to prevent triggering frame load during list rebuild
            self.labeling_list.blockSignals(True)
            self.frame_list.blockSignals(True)
            self._populate_frames()
            self.labeling_list.blockSignals(False)
            self.frame_list.blockSignals(False)
            # Restore selection without triggering load (already loaded)
            for i in range(self.labeling_list.count()):
                item = self.labeling_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.labeling_list.setCurrentRow(i)
                    break
            for i in range(self.frame_list.count()):
                item = self.frame_list.item(i)
                if item.data(Qt.UserRole) == self.current_index:
                    self.frame_list.setCurrentRow(i)
                    break

    def _select_frame_in_list(self, idx: int):
        """Select a frame by its actual index in the appropriate list."""
        # Check labeling list first
        for i in range(self.labeling_list.count()):
            item = self.labeling_list.item(i)
            if item.data(Qt.UserRole) == idx:
                self.labeling_list.setCurrentRow(i)
                return
        # Check all frames list
        for i in range(self.frame_list.count()):
            item = self.frame_list.item(i)
            if item.data(Qt.UserRole) == idx:
                self.frame_list.setCurrentRow(i)
                return

    def prev_keypoint(self: object) -> object:
        """prev_keypoint method documentation."""
        if self.current_kpt > 0:
            self.current_kpt -= 1
            self.kpt_list.setCurrentRow(self.current_kpt)

    def next_keypoint(self: object) -> object:
        """next_keypoint method documentation."""
        if self.current_kpt < self.kpt_list.count() - 1:
            self.current_kpt += 1
            self.kpt_list.setCurrentRow(self.current_kpt)

    def next_unlabeled(self: object) -> object:
        """next_unlabeled method documentation."""
        labeling_indices = sorted(self.labeling_frames)
        if not labeling_indices:
            QMessageBox.information(
                self, "No labeling frames", "No frames in labeling set."
            )
            return

        # Find current position in labeling set
        try:
            current_pos = labeling_indices.index(self.current_index)
            search_start = current_pos + 1
        except ValueError:
            search_start = 0

        # Search forward in labeling frames
        for i in range(search_start, len(labeling_indices)):
            idx = labeling_indices[i]
            if not self._is_labeled(self.image_paths[idx]):
                self._maybe_autosave()
                self._select_frame_in_list(idx)
                return

        QMessageBox.information(
            self,
            "Done",
            "No unlabeled frames found in labeling set after current frame.",
        )

    # ----- save / export -----
    def save_current(self: object, refresh_ui: object = True) -> object:
        """save_current method documentation."""
        if self._ann is None:
            return
        # Keep cache in sync
        self._cache_current_frame()
        logger.debug(
            "Save current frame=%d refresh_ui=%s", self.current_index, refresh_ui
        )
        img_path = self.image_paths[self.current_index]
        label_path = self._label_path_for(img_path)

        w, h = self._img_wh
        cls = int(self.class_combo.currentIndex())
        self._ann.cls = cls

        bbox = compute_bbox_from_kpts(self._ann.kpts, self.project.bbox_pad_frac, w, h)

        save_yolo_pose_label(
            label_path=label_path,
            cls=cls,
            img_w=w,
            img_h=h,
            kpts_px=self._ann.kpts,
            bbox_xyxy_px=bbox,
            pad_frac=self.project.bbox_pad_frac,
        )
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
        self._dirty = False

        # Only refresh UI if we're staying on the current frame
        if refresh_ui:
            self._populate_frames()
            self._select_frame_in_list(self.current_index)

        self.statusBar().showMessage(f"Saved: {label_path.name}", 2000)
        self._set_saved_status()
        self.save_project()

    def save_all_labeling_frames(self: object) -> object:
        """Save all labeling frames to disk using current in-memory state."""
        if not self.labeling_frames:
            return

        logger.debug("Save all labeling frames: count=%d", len(self.labeling_frames))

        # Cache current frame state before batch save
        self._cache_current_frame()

        current_idx = self.current_index
        saved_count = 0

        for idx in sorted(self.labeling_frames):
            ann = None
            if idx == current_idx and self._ann is not None:
                ann = self._clone_ann(self._ann)
            elif idx in self._frame_cache:
                ann = self._clone_ann(self._frame_cache[idx])
            else:
                ann = self._load_ann_from_disk(idx)

            logger.debug(
                "Saving frame %d (from %s)",
                idx,
                (
                    "current"
                    if idx == current_idx
                    else ("cache" if idx in self._frame_cache else "disk")
                ),
            )

            img_path = self.image_paths[idx]
            img = self._read_image(img_path)
            h, w = img.shape[:2]

            bbox = compute_bbox_from_kpts(ann.kpts, self.project.bbox_pad_frac, w, h)
            save_yolo_pose_label(
                label_path=self._label_path_for(img_path),
                cls=int(ann.cls),
                img_w=w,
                img_h=h,
                kpts_px=ann.kpts,
                bbox_xyxy_px=bbox,
                pad_frac=self.project.bbox_pad_frac,
            )
            saved_count += 1

        self._dirty = False
        self._populate_frames()
        self._select_frame_in_list(current_idx)
        self.statusBar().showMessage(f"Saved {saved_count} labeling frames", 2000)
        self._set_saved_status()
        self.save_project()

    def open_skeleton_editor(self: object) -> object:
        """open_skeleton_editor method documentation."""
        old_kpts = list(self.project.keypoint_names)
        dlg = SkeletonEditorDialog(
            self.project.keypoint_names,
            self.project.skeleton_edges,
            self,
            default_dir=get_default_skeleton_dir(),
        )
        if dlg.exec() == QDialog.Accepted:
            names, edges = dlg.get_result()

            if names != old_kpts:
                resp = QMessageBox.question(
                    self,
                    "Update keypoints",
                    "Keypoints changed in Skeleton Editor.\n"
                    "Migrate existing label files to the new layout?",
                )
                if resp == QMessageBox.Yes:
                    modified, total = migrate_labels_keypoints(
                        self.project.labels_dir, old_kpts, names, mode="name"
                    )
                    QMessageBox.information(
                        self,
                        "Keypoint migration",
                        f"Migrated {modified} / {total} label files.",
                    )

            self.project.keypoint_names = names
            self._rebuild_kpt_list()

            if self._ann is not None:
                k = len(self.project.keypoint_names)
                if len(self._ann.kpts) < k:
                    self._ann.kpts.extend(
                        [Keypoint(0.0, 0.0, 0) for _ in range(k - len(self._ann.kpts))]
                    )
                else:
                    self._ann.kpts = self._ann.kpts[:k]

            k = len(self.project.keypoint_names)
            self.project.skeleton_edges = [
                (a, b) for (a, b) in edges if 0 <= a < k and 0 <= b < k and a != b
            ]
            self.infer = PoseInferenceService(
                self.project.out_root,
                self.project.keypoint_names,
                self.project.skeleton_edges,
            )
            if self._ann is not None:
                self._rebuild_canvas()
            self._populate_frames()
            self._update_info()
            self.save_project()

    def open_project_settings(self: object) -> object:
        """open_project_settings method documentation."""
        wiz = ProjectWizard(self.project.images_dir, existing=self.project, parent=self)
        if wiz.exec() != QDialog.Accepted:
            return

        old_classes = list(self.project.class_names)
        old_kpts = list(self.project.keypoint_names)

        new_root, new_labels = wiz.get_paths()
        new_classes = wiz.get_classes()
        new_kpts = wiz.get_keypoints()
        new_edges = wiz.get_edges()
        autosave, pad = wiz.get_options()
        do_mig, mig_mode = wiz.get_migration()

        # Apply paths
        self.project.out_root = new_root
        self.project.labels_dir = new_labels
        self.project.labels_dir.mkdir(parents=True, exist_ok=True)

        # Warning if classes changed
        if new_classes != old_classes:
            QMessageBox.information(
                self,
                "Note on classes",
                "Changing class ordering can change the meaning of existing labels.\n"
                "If you already labeled data, consider keeping class order stable.",
            )
        self.project.class_names = new_classes

        # Keypoints changed -> optional migration
        if new_kpts != old_kpts and do_mig:
            modified, total = migrate_labels_keypoints(
                self.project.labels_dir, old_kpts, new_kpts, mode=mig_mode
            )
            QMessageBox.information(
                self,
                "Keypoint migration",
                f"Migrated {modified} / {total} label files using mode='{mig_mode}'.",
            )

        self.project.keypoint_names = new_kpts
        # clamp edges to new range
        k = len(new_kpts)
        self.project.skeleton_edges = [
            (a, b) for (a, b) in new_edges if 0 <= a < k and 0 <= b < k and a != b
        ]
        self.infer = PoseInferenceService(
            self.project.out_root,
            self.project.keypoint_names,
            self.project.skeleton_edges,
        )
        self.project.autosave = autosave
        self.project.bbox_pad_frac = pad

        # Refresh UI
        self.class_combo.clear()
        self.class_combo.addItems(self.project.class_names)
        self._rebuild_kpt_list()

        # Ensure current annotation matches new K
        if self._ann is not None:
            if len(self._ann.kpts) < k:
                self._ann.kpts.extend(
                    [Keypoint(0.0, 0.0, 0) for _ in range(k - len(self._ann.kpts))]
                )
            else:
                self._ann.kpts = self._ann.kpts[:k]

        self._rebuild_canvas()
        self._populate_frames()
        self.save_project()
        self._update_info()

    def _switch_project_window(self, proj: Project):
        imgs = list_images(proj.images_dir)
        if not imgs:
            QMessageBox.critical(
                self, "No images", f"No images found under: {proj.images_dir}"
            )
            return
        try:
            self._perform_autosave()
            self.save_project()
        except Exception:
            pass
        new_win = MainWindow(proj, imgs)
        new_win.resize(self.size())
        new_win.showMaximized()
        app = QApplication.instance()
        if app is not None:
            if not hasattr(app, "_posekit_windows"):
                app._posekit_windows = []
            app._posekit_windows.append(new_win)
        self.close()

    def open_project_dialog(self: object) -> object:
        """open_project_dialog method documentation."""
        start = str(self.project.project_path.parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open pose_project.json", start, "pose_project.json;All Files (*)"
        )
        if not path:
            return
        project_path = Path(path).expanduser().resolve()
        if not project_path.exists():
            QMessageBox.warning(self, "Missing", "Project file not found.")
            return
        try:
            data = json.loads(project_path.read_text(encoding="utf-8"))
            base = project_path.parent
            images_dir = _resolve_project_path(Path(data["images_dir"]), base, data)
        except Exception as e:
            QMessageBox.warning(self, "Invalid project", str(e))
            return
        proj = load_project_with_repairs(project_path, images_dir)
        self._switch_project_window(proj)

    def open_images_folder(self: object) -> object:
        """open_images_folder method documentation."""
        start = str(self.project.images_dir) if self.project.images_dir else ""
        path = QFileDialog.getExistingDirectory(self, "Select images folder", start)
        if not path:
            return
        images_dir = Path(path).expanduser().resolve()
        out_root = (images_dir.parent / "pose_project").resolve()
        proj = None
        found = find_project(images_dir, out_root)
        if found:
            proj = load_project_with_repairs(found, images_dir)
        else:
            proj = create_project_via_wizard(images_dir, out_root_hint=out_root)
        if proj is None:
            return
        self._switch_project_window(proj)

    def export_dataset_dialog(self: object) -> object:
        """export_dataset_dialog method documentation."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Export dataset.yaml + copied images/labels")
        layout = QFormLayout(dlg)

        out_root = QLineEdit(str(self.project.out_root))
        split = QDoubleSpinBox()
        split.setRange(0.05, 0.95)
        split.setSingleStep(0.05)
        split.setValue(0.8)

        seed = QSpinBox()
        seed.setRange(0, 999999)
        seed.setValue(0)

        split_method = QComboBox()
        split_method.addItems(["Random", "Cluster-stratified"])

        cluster_csv = QLineEdit("")
        cluster_csv.setPlaceholderText("Optional: clusters.csv (auto-detected)")
        btn_cluster = QPushButton("Choose…")

        def pick_dir() -> object:
            """pick_dir method documentation."""
            d = QFileDialog.getExistingDirectory(
                self, "Select output root", out_root.text()
            )
            if d:
                out_root.setText(d)

        def pick_cluster() -> object:
            """pick_cluster method documentation."""
            path, _ = QFileDialog.getOpenFileName(
                self, "Select cluster CSV", str(self.project.out_root), "CSV (*.csv)"
            )
            if path:
                cluster_csv.setText(path)

        btn_pick = QPushButton("Choose…")
        btn_pick.clicked.connect(pick_dir)
        btn_cluster.clicked.connect(pick_cluster)

        row = QHBoxLayout()
        row.addWidget(out_root, 1)
        row.addWidget(btn_pick)

        cl_row = QHBoxLayout()
        cl_row.addWidget(cluster_csv, 1)
        cl_row.addWidget(btn_cluster)

        layout.addRow("Where should output root be set?", row)
        layout.addRow("Which split method should be used?", split_method)
        layout.addRow("What train fraction should be used?", split)
        layout.addRow("What should random seed be?", seed)
        layout.addRow("What should cluster csv be?", cl_row)

        btns = QHBoxLayout()
        ok = QPushButton("Export")
        cancel = QPushButton("Cancel")
        btns.addStretch(1)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addRow(btns)

        cancel.clicked.connect(dlg.reject)
        ok.clicked.connect(dlg.accept)

        if dlg.exec() != QDialog.Accepted:
            return

        root = Path(out_root.text()).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        train_items = None
        val_items = None
        if split_method.currentText() == "Cluster-stratified":
            csv_path = (
                Path(cluster_csv.text().strip()) if cluster_csv.text().strip() else None
            )
            if csv_path is None or not csv_path.exists():
                default_csv = (
                    self.project.out_root / "posekit" / "clusters" / "clusters.csv"
                )
                csv_path = default_csv if default_csv.exists() else None
            if not csv_path:
                QMessageBox.warning(
                    self,
                    "Missing clusters",
                    "No cluster CSV found. Run Smart Select → Clustering first.",
                )
                return

            cluster_ids = self._load_cluster_ids_from_csv(csv_path)
            if not cluster_ids:
                QMessageBox.warning(
                    self,
                    "Missing clusters",
                    "Could not load cluster IDs from CSV.",
                )
                return

            labeled_indices = [
                i for i, p in enumerate(self.image_paths) if self._is_labeled(p)
            ]
            items = []
            item_cluster_ids = []
            for i in labeled_indices:
                img = self.image_paths[i]
                lbl = self._label_path_for(img)
                if lbl.exists():
                    items.append((img, lbl))
                    item_cluster_ids.append(cluster_ids[i])

            if len(items) < 2:
                QMessageBox.warning(
                    self, "Not enough labels", "Need at least 2 labeled frames."
                )
                return

            train_idx, val_idx, _ = cluster_stratified_split(
                [p for p, _ in items],
                item_cluster_ids,
                train_frac=float(split.value()),
                val_frac=1.0 - float(split.value()),
                test_frac=0.0,
                min_per_cluster=1,
                seed=int(seed.value()),
            )
            train_items = [items[i] for i in train_idx]
            val_items = [items[i] for i in val_idx]

        try:
            info = build_yolo_pose_dataset(
                self.image_paths,
                self.project.labels_dir,
                root,
                float(split.value()),
                int(seed.value()),
                self.project.class_names,
                self.project.keypoint_names,
                train_items=train_items,
                val_items=val_items,
                ignore_occluded_train=False,
                ignore_occluded_val=False,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
            return

        QMessageBox.information(
            self,
            "Exported",
            "Wrote:\n"
            f"- {info['yaml_path']}\n"
            f"- images/train + labels/train\n"
            f"- images/val + labels/val\n"
            f"- {info.get('manifest', '')}",
        )

    def open_smart_select(self: object) -> object:
        """open_smart_select method documentation."""
        dlg = SmartSelectDialog(self, self.project, self.image_paths, self._is_labeled)
        if dlg.exec() != QDialog.Accepted or not getattr(dlg, "_did_add", False):
            return

        # If they added nothing, ignore
        picked = getattr(dlg, "selected_indices", None)
        if not picked:
            return

        self._add_indices_to_labeling(picked, "Smart Select")

    def _add_indices_to_labeling(self, indices: List[int], title: str):
        if not indices:
            return
        for idx in indices:
            self.labeling_frames.add(int(idx))
        self._populate_frames()
        self._select_frame_in_list(self.current_index)
        QMessageBox.information(
            self, title, f"Added {len(indices)} frames to labeling set."
        )

    def _collect_selected_indices(self) -> List[int]:
        idxs = set()
        for item in self.labeling_list.selectedItems():
            try:
                idx = int(item.data(Qt.UserRole))
                idxs.add(idx)
            except Exception:
                continue
        for item in self.frame_list.selectedItems():
            try:
                idx = int(item.data(Qt.UserRole))
                idxs.add(idx)
            except Exception:
                continue
        return sorted(idxs)

    def _delete_selected_frames(self):
        indices = self._collect_selected_indices()
        if not indices:
            QMessageBox.information(self, "Delete", "No frames selected.")
            return

        names = [self.image_paths[i].name for i in indices if i < len(self.image_paths)]
        preview = "\n".join(names[:10])
        if len(names) > 10:
            preview += f"\n… (+{len(names) - 10} more)"

        resp = QMessageBox.warning(
            self,
            "Delete selected frames?",
            "This will permanently delete the selected images and any corresponding labels.\n\n"
            f"Count: {len(indices)}\n\n{preview}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return

        # Keep track of current frame path for remapping
        cur_path = None
        if 0 <= self.current_index < len(self.image_paths):
            cur_path = self.image_paths[self.current_index]

        deleted_paths: List[Path] = []

        # Delete files
        for idx in indices:
            if idx < 0 or idx >= len(self.image_paths):
                continue
            img_path = self.image_paths[idx]
            deleted_paths.append(img_path)
            try:
                if img_path.exists():
                    img_path.unlink()
            except Exception:
                pass
            try:
                lbl_path = self._label_path_for(img_path)
                if lbl_path.exists():
                    lbl_path.unlink()
            except Exception:
                pass

        self._cleanup_deleted_cache(deleted_paths)

        # Rebuild image list and labeling set by path
        old_labeling_paths = {
            self.image_paths[i]
            for i in self.labeling_frames
            if 0 <= i < len(self.image_paths)
        }
        self.image_paths = list_images(self.project.images_dir)
        self._rebuild_path_index()
        self.labeling_frames = set()
        for i, p in enumerate(self.image_paths):
            if p in old_labeling_paths or self._is_labeled(p):
                self.labeling_frames.add(i)

        # Reset caches and current state
        self._frame_cache.clear()
        self._ann = None
        self._dirty = False
        self._undo_stack.clear()
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._cluster_ids_cache = None
        self._cluster_ids_mtime = None

        # Remap current index
        self.current_index = 0
        if cur_path:
            try:
                self.current_index = self.image_paths.index(cur_path)
            except ValueError:
                self.current_index = 0

        self._populate_frames()
        if self.image_paths:
            self.load_frame(self.current_index)
        else:
            self.canvas.set_image(np.zeros((1, 1, 3), dtype=np.uint8))
            self.lbl_info.setText("No images found.")

    def _cleanup_deleted_cache(self, deleted_paths: List[Path]) -> None:
        if not deleted_paths:
            return
        deleted_keys = set()
        for p in deleted_paths:
            try:
                deleted_keys.add(str(p))
                deleted_keys.add(str(Path(p).resolve()))
            except Exception:
                deleted_keys.add(str(p))

        # Metadata cleanup
        try:
            keys = list(self.metadata_manager.metadata.keys())
            for k in keys:
                try:
                    if str(Path(k).resolve()) in deleted_keys or k in deleted_keys:
                        self.metadata_manager.metadata.pop(k, None)
                except Exception:
                    if k in deleted_keys:
                        self.metadata_manager.metadata.pop(k, None)
            self.metadata_manager.save()
        except Exception:
            pass

        # Prediction cache cleanup
        pred_dir = self.project.out_root / "posekit" / "predictions"
        if pred_dir.exists():
            for cache_path in pred_dir.glob("*.json"):
                try:
                    data = json.loads(cache_path.read_text(encoding="utf-8"))
                    preds = data.get("preds", {})
                    if not preds:
                        continue
                    changed = False
                    for k in list(preds.keys()):
                        if k in deleted_keys:
                            preds.pop(k, None)
                            changed = True
                    if changed:
                        data["preds"] = preds
                        cache_path.write_text(
                            json.dumps(data), encoding="utf-8"
                        )
                except Exception:
                    continue

        # Embeddings cache cleanup (path-based)
        try:
            emb_root = self.project.out_root / "posekit" / "embeddings"
            if emb_root.exists():
                for child in emb_root.iterdir():
                    if not child.is_dir():
                        continue
                    try:
                        cache = IncrementalEmbeddingCache(
                            self.project.out_root / "posekit", child.name
                        )
                        cache.remove_paths(deleted_keys)
                    except Exception:
                        continue
        except Exception:
            pass

        # Clusters cache cleanup (path-based CSV)
        try:
            cluster_csv = (
                self.project.out_root / "posekit" / "clusters" / "clusters.csv"
            )
            if cluster_csv.exists():
                rows = []
                changed = False
                with cluster_csv.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img = row.get("image") or row.get("image_path") or row.get("path")
                        if not img:
                            continue
                        key = img
                        try:
                            if (
                                key in deleted_keys
                                or str(Path(key).resolve()) in deleted_keys
                            ):
                                changed = True
                                continue
                        except Exception:
                            if key in deleted_keys:
                                changed = True
                                continue
                        rows.append({"image": img, "cluster_id": row.get("cluster_id")})
                if changed:
                    cluster_csv.parent.mkdir(parents=True, exist_ok=True)
                    with cluster_csv.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["image", "cluster_id"])
                        for row in rows:
                            writer.writerow([row["image"], row["cluster_id"]])
        except Exception:
            pass

    def _preds_to_keypoints(
        self, preds: List[Tuple[float, float, float]], conf_thr: float = 0.25
    ) -> List[Keypoint]:
        kpts = []
        for x, y, c in preds:
            if not np.isfinite(c):
                c = 0.0
            if not np.isfinite(x) or not np.isfinite(y):
                x, y, c = 0.0, 0.0, 0.0
            if c > conf_thr and c > 0.0:
                kpts.append(Keypoint(float(x), float(y), 2))
            else:
                kpts.append(Keypoint(0.0, 0.0, 0))
        return kpts

    def _pred_conf_default(self) -> float:
        try:
            return float(self.sp_pred_conf.value())
        except Exception:
            return 0.25

    def _on_pred_conf_changed(self, _value: float):
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._populate_frames()
        self._rebuild_canvas()

    def _pred_backend(self) -> str:
        try:
            txt = self.combo_pred_backend.currentText().strip().lower()
            return "sleap" if txt.startswith("sleap") else "yolo"
        except Exception:
            return "yolo"

    def _set_sleap_status(self, text: str, visible: Optional[bool] = None):
        if not hasattr(self, "status_sleap"):
            return
        if visible is not None:
            self.status_sleap.setVisible(bool(visible))
        self.status_sleap.setText(text)

    def _set_busy_progress(self, msg: Optional[str] = None):
        self.status_progress.setRange(0, 0)
        self.status_progress.setVisible(True)
        if msg:
            self.statusBar().showMessage(msg)

    def _clear_progress(self):
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setVisible(False)

    def _update_pred_backend_ui(self):
        backend = self._pred_backend()
        is_sleap = backend == "sleap"
        if hasattr(self, "yolo_pred_widget"):
            self.yolo_pred_widget.setVisible(not is_sleap)
        if hasattr(self, "sleap_pred_widget"):
            self.sleap_pred_widget.setVisible(is_sleap)
        if is_sleap:
            running = PoseInferenceService.sleap_service_running()
            self._set_sleap_status(
                "SLEAP: running" if running else "SLEAP: idle", visible=True
            )
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(not running)
                self.btn_sleap_stop.setEnabled(running)
        else:
            self._set_sleap_status("SLEAP: off", visible=False)
        if not is_sleap:
            if PoseInferenceService.sleap_service_running():
                self._stop_sleap_service()
            else:
                try:
                    PoseInferenceService.shutdown_sleap_service()
                except Exception:
                    pass

    def _refresh_sleap_envs(self):
        if not hasattr(self, "combo_sleap_env"):
            return
        self.combo_sleap_env.clear()
        self.combo_sleap_env.setEnabled(True)
        envs: List[str] = []
        try:
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
                    name = parts[0]
                    if name.lower().startswith("sleap"):
                        envs.append(name)
            else:
                self.lbl_sleap_env_status.setText("Unable to list conda environments.")
        except FileNotFoundError:
            self.lbl_sleap_env_status.setText("Conda not found on PATH.")
            envs = []
        except Exception as e:
            self.lbl_sleap_env_status.setText(f"Env scan failed: {e}")
            envs = []

        if not envs:
            self.combo_sleap_env.addItem("No sleap envs found")
            self.combo_sleap_env.setEnabled(False)
            if not self.lbl_sleap_env_status.text():
                self.lbl_sleap_env_status.setText(
                    "No conda envs starting with 'sleap' found."
                )
        else:
            self.combo_sleap_env.addItems(envs)
            if self._sleap_env_pref and self._sleap_env_pref in envs:
                self.combo_sleap_env.setCurrentText(self._sleap_env_pref)
            self.lbl_sleap_env_status.setText("")

    def _browse_sleap_model_dir(self):
        start = self.sleap_model_edit.text().strip() or str(self.project.out_root)
        path = QFileDialog.getExistingDirectory(
            self, "Select SLEAP model directory", start
        )
        if path:
            self._validate_sleap_model_dir(Path(path), notify=True)

    def _is_sleap_model_dir(self, path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        return (path / "training_config.yaml").exists() or (
            path / "training_config.json"
        ).exists()

    def _find_latest_sleap_model_dir_from(self, base: Path) -> Optional[Path]:
        models_dir = base / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            return None
        best_ts = None
        best_path = None
        for child in models_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            prefix = name.split(".", 1)[0]
            try:
                ts = datetime.strptime(prefix, "%y%m%d_%H%M%S")
            except Exception:
                continue
            if not self._is_sleap_model_dir(child):
                continue
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_path = child
        return best_path

    def _resolve_sleap_model_dir(self, path: Path) -> Optional[Path]:
        try:
            path = path.expanduser().resolve()
        except Exception:
            pass
        if path.is_file():
            path = path.parent
        if self._is_sleap_model_dir(path):
            return path
        found = self._find_latest_sleap_model_dir_from(path)
        if found:
            return found
        try:
            parent = path.parent
        except Exception:
            parent = None
        if parent:
            found = self._find_latest_sleap_model_dir_from(parent)
            if found:
                return found
        return None

    def _validate_sleap_model_dir(
        self, path: Path, notify: bool = True
    ) -> Optional[Path]:
        resolved = self._resolve_sleap_model_dir(path)
        if not resolved:
            if notify:
                QMessageBox.warning(
                    self,
                    "Invalid SLEAP model dir",
                    "Selected folder does not contain training_config.yaml/json.\n"
                    "Choose a model folder under models/<YYMMDD_HHMMSS>.*",
                )
            return None
        if notify and resolved != path:
            QMessageBox.information(
                self,
                "SLEAP model folder resolved",
                f"'{path}' is not a model folder.\nUsing:\n{resolved}",
            )
        self.sleap_model_edit.setText(str(resolved))
        return resolved

    def _find_latest_sleap_model_dir(self, slp_path: Path) -> Optional[Path]:
        try:
            base = slp_path.parent
        except Exception:
            return None
        return self._find_latest_sleap_model_dir_from(base)

    def _use_latest_sleap_model(self):
        slp = getattr(self.project, "latest_sleap_dataset", None)
        if not slp:
            QMessageBox.information(
                self,
                "No SLEAP dataset",
                "No latest SLEAP dataset recorded. Export a SLEAP dataset first.",
            )
            return
        slp_path = Path(slp)
        if not slp_path.exists():
            QMessageBox.warning(
                self,
                "Missing dataset",
                f"Latest SLEAP dataset not found:\n{slp_path}",
            )
            return
        model_dir = self._find_latest_sleap_model_dir(slp_path)
        if not model_dir:
            QMessageBox.warning(
                self,
                "No models found",
                f"No models folder with timestamped runs found next to:\n{slp_path}",
            )
            return
        self._validate_sleap_model_dir(model_dir, notify=False)

    def _get_sleap_env(self) -> Optional[str]:
        env = self.combo_sleap_env.currentText().strip()
        if not env or env.lower().startswith("no sleap"):
            return None
        return env

    def _get_sleap_model_or_prompt(self) -> Optional[Path]:
        txt = self.sleap_model_edit.text().strip()
        if txt:
            resolved = self._validate_sleap_model_dir(Path(txt), notify=True)
            if resolved:
                return resolved
            return None
        self._browse_sleap_model_dir()
        txt = self.sleap_model_edit.text().strip()
        if not txt:
            return None
        resolved = self._validate_sleap_model_dir(Path(txt), notify=True)
        if resolved:
            return resolved
        return None

    def _get_sleap_model_silent(self) -> Optional[Path]:
        txt = self.sleap_model_edit.text().strip()
        if not txt:
            return None
        return self._validate_sleap_model_dir(Path(txt), notify=False)

    def _start_sleap_service(self):
        env = self._get_sleap_env()
        if not env:
            QMessageBox.warning(
                self,
                "No SLEAP env",
                "Select a conda env starting with 'sleap' for SLEAP inference.",
            )
            return
        if PoseInferenceService.sleap_service_running():
            self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            return
        if self._sleap_service_thread and self._sleap_service_thread.isRunning():
            return
        self._set_sleap_status("SLEAP: starting", visible=True)
        self._set_busy_progress("Starting SLEAP service…")
        self.btn_sleap_start.setEnabled(False)
        self.btn_sleap_stop.setEnabled(False)
        self._sleap_service_thread = QThread()
        self._sleap_service_worker = SleapServiceWorker(env, self.project.out_root)
        self._sleap_service_worker.moveToThread(self._sleap_service_thread)
        self._sleap_service_thread.started.connect(self._sleap_service_worker.run)
        self._sleap_service_worker.finished.connect(self._on_sleap_service_started)
        self._sleap_service_worker.finished.connect(
            self._sleap_service_thread.quit
        )
        self._sleap_service_thread.finished.connect(
            self._sleap_service_thread.deleteLater
        )
        self._sleap_service_thread.start()

    def _stop_sleap_service(self):
        if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
            self.btn_sleap_start.setEnabled(False)
            self.btn_sleap_stop.setEnabled(False)
        self._set_sleap_status("SLEAP: stopping", visible=True)
        self.statusBar().showMessage("Stopping SLEAP service…", 2000)

        try:
            PoseInferenceService.shutdown_sleap_service()
        except Exception as e:
            self._set_sleap_status("SLEAP: error", visible=True)
            self.statusBar().showMessage(f"SLEAP stop failed: {e}", 4000)
        if self._sleap_service_thread and self._sleap_service_thread.isRunning():
            try:
                self._sleap_service_thread.quit()
            except Exception:
                pass

        running = PoseInferenceService.sleap_service_running()
        if running:
            self._set_sleap_status("SLEAP: running", visible=True)
            self.statusBar().showMessage("SLEAP service still running.", 4000)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            return

        self._set_sleap_status("SLEAP: idle", visible=True)
        self.statusBar().showMessage("SLEAP service stopped.", 3000)
        if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
            self.btn_sleap_start.setEnabled(True)
            self.btn_sleap_stop.setEnabled(False)

    def _on_sleap_service_started(self, ok: bool, err: str, log_path: str):
        self._clear_progress()
        self.btn_sleap_start.setEnabled(True)
        self.btn_sleap_stop.setEnabled(True)
        if not ok:
            self._set_sleap_status("SLEAP: error", visible=True)
            QMessageBox.warning(self, "SLEAP service failed", err)
            return
        self._set_sleap_status("SLEAP: running", visible=True)
        self.btn_sleap_start.setEnabled(False)
        self.btn_sleap_stop.setEnabled(True)
        if log_path:
            self.statusBar().showMessage(
                f"SLEAP service running. Log: {log_path}", 4000
            )

    def _get_pred_model_or_prompt(self) -> Optional[Path]:
        if self._pred_backend() == "sleap":
            return self._get_sleap_model_or_prompt()
        return self._get_pred_weights_or_prompt()

    def _get_pred_model_silent(self) -> Optional[Path]:
        if self._pred_backend() == "sleap":
            return self._get_sleap_model_silent()
        return self._get_pred_weights_silent()

    def _browse_pred_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select pose weights", "", "*.pt")
        if path:
            self.pred_weights_edit.setText(path)

    def _use_latest_pred_weights(self):
        if (
            self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.pred_weights_edit.setText(str(self.project.latest_pose_weights))

    def _get_pred_weights_or_prompt(self) -> Optional[Path]:
        txt = self.pred_weights_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
            QMessageBox.warning(
                self, "Invalid weights", "Prediction weights not found."
            )
            return None
        weights = self._get_latest_weights_or_prompt()
        if weights:
            self.pred_weights_edit.setText(str(weights))
        return weights

    def _get_pred_weights_silent(self) -> Optional[Path]:
        txt = self.pred_weights_edit.text().strip()
        if txt:
            p = Path(txt).expanduser().resolve()
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
            return None
        if self.project.latest_pose_weights:
            p = Path(self.project.latest_pose_weights)
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p
        return None

    def _get_pred_overlay_for_current(
        self,
    ) -> Tuple[Optional[List[Keypoint]], Optional[List[float]]]:
        if not self.show_predictions:
            return None, None
        model = self._get_pred_model_silent()
        if not model:
            return None, None
        backend = self._pred_backend()
        preds = self._get_pred_for_frame(
            model, self.image_paths[self.current_index], backend=backend
        )
        if not preds:
            return None, None
        conf_thr = self._pred_conf_default()
        pred_kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
        pred_confs = []
        for p in preds:
            if len(p) >= 3 and np.isfinite(p[2]):
                pred_confs.append(float(p[2]))
            else:
                pred_confs.append(0.0)
        return pred_kpts, pred_confs

    def _rebuild_canvas(self):
        if self._ann is None:
            return
        pred_kpts, pred_confs = self._get_pred_overlay_for_current()
        self.canvas.rebuild_overlays(
            self._ann.kpts,
            self.project.keypoint_names,
            self.project.skeleton_edges,
            pred_kpts=pred_kpts,
            pred_confs=pred_confs,
            show_pred_conf=self.show_pred_conf,
        )

    def _toggle_show_predictions(self, checked: bool):
        self.show_predictions = bool(checked)
        self._rebuild_canvas()

    def _toggle_show_pred_conf(self, checked: bool):
        checked = bool(checked)
        self.show_pred_conf = checked
        if hasattr(self, "cb_show_pred_conf") and self.cb_show_pred_conf.isChecked() != checked:
            self.cb_show_pred_conf.blockSignals(True)
            self.cb_show_pred_conf.setChecked(checked)
            self.cb_show_pred_conf.blockSignals(False)
        if hasattr(self, "act_show_pred_conf") and self.act_show_pred_conf.isChecked() != checked:
            self.act_show_pred_conf.blockSignals(True)
            self.act_show_pred_conf.setChecked(checked)
            self.act_show_pred_conf.blockSignals(False)
        self._rebuild_canvas()

    def _get_pred_for_frame(
        self, model: Path, img_path: Path, backend: str = "yolo"
    ) -> Optional[List[Tuple[float, float, float]]]:
        return self.infer.get_cached_pred(model, img_path, backend=backend)

    def _current_frame_has_labels(self) -> bool:
        if self._ann is not None and any(kp.v > 0 for kp in self._ann.kpts):
            return True
        img_path = self.image_paths[self.current_index]
        if self._is_labeled(img_path):
            try:
                ann = self._load_ann_from_disk(self.current_index)
                return any(kp.v > 0 for kp in ann.kpts)
            except Exception:
                return True
        return False

    def _get_latest_weights_or_prompt(self) -> Optional[Path]:
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
        ):
            p = Path(self.project.latest_pose_weights)
            if p.exists() and p.is_file() and p.suffix == ".pt":
                return p

        path, _ = QFileDialog.getOpenFileName(self, "Select pose weights", "", "*.pt")
        if not path:
            return None
        weights = Path(path).resolve()
        self.project.latest_pose_weights = weights
        self.save_project()
        return weights

    def predict_current_frame(self: object) -> object:
        """predict_current_frame method documentation."""
        if not self.image_paths or self._ann is None:
            QMessageBox.information(self, "No frame", "Select a frame first.")
            return

        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        if not model:
            return
        sleap_env = None
        if backend == "sleap":
            sleap_env = self._get_sleap_env()
            if not sleap_env:
                QMessageBox.warning(
                    self,
                    "No SLEAP env",
                    "Select a conda env starting with 'sleap' for SLEAP inference.",
                )
                return
            if not PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: starting", visible=True)
                self._set_busy_progress("Starting SLEAP service…")
            else:
                self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)
            self._last_pred_conf = None
        else:
            self._last_pred_conf = self._pred_conf_default()
        self._last_pred_model = model
        self._last_pred_backend = backend

        cached = self._get_pred_for_frame(
            model, self.image_paths[self.current_index], backend=backend
        )
        if cached is not None:
            self._on_pred_finished(cached)
            return

        if backend != "sleap":
            self.statusBar().showMessage(f"Predicting keypoints… ({model.name})")
        self.btn_predict.setEnabled(False)
        if backend == "sleap":
            if PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: running", visible=True)

        self._pred_thread = QThread()
        self._pred_worker = PosePredictWorker(
            model_path=model,
            image_path=self.image_paths[self.current_index],
            out_root=self.project.out_root,
            keypoint_names=self.project.keypoint_names,
            skeleton_edges=self.project.skeleton_edges,
            backend=backend,
            device="auto",
            imgsz=640,
            conf=self._last_pred_conf or 0.0,
            sleap_env=sleap_env,
            sleap_device=self.combo_sleap_device.currentText().strip(),
            sleap_batch=1,
            sleap_max_instances=1,
        )
        self._pred_worker.moveToThread(self._pred_thread)
        self._pred_thread.started.connect(self._pred_worker.run)
        self._pred_worker.finished.connect(self._on_pred_finished)
        self._pred_worker.failed.connect(self._on_pred_failed)
        self._pred_worker.finished.connect(self._pred_thread.quit)
        self._pred_worker.failed.connect(self._pred_thread.quit)
        self._pred_thread.finished.connect(self._pred_thread.deleteLater)
        self._pred_thread.start()

    def _on_pred_finished(self, preds: List[Tuple[float, float, float]]):
        self.btn_predict.setEnabled(True)
        if not preds:
            backend = getattr(self, "_last_pred_backend", "yolo")
            conf = getattr(self, "_last_pred_conf", None)
            if backend == "sleap":
                msg = "No predictions found."
            elif conf is not None:
                msg = f"No predictions above conf={conf:.2f}. Try a lower conf."
            else:
                msg = "No predictions above confidence threshold. Try a lower conf."
            self.statusBar().showMessage(msg, 3000)
            if backend == "sleap":
                self._set_sleap_status("SLEAP: idle", visible=True)
                self._clear_progress()
            return
        self.statusBar().showMessage("Prediction loaded. Adjust and save.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
            self._clear_progress()
        if self._ann is None:
            return
        if self.mode != "frame":
            self.rb_frame.setChecked(True)
        model = getattr(self, "_last_pred_model", None)
        backend = getattr(self, "_last_pred_backend", "yolo")
        if model:
            img_path = str(self.image_paths[self.current_index])
            self.infer.merge_cache(model, {img_path: preds}, backend=backend)
            self._update_pred_conf_map_from_preds(
                {img_path: preds}, model=model, backend=backend
            )
            conf_val = self._pred_conf_map.get(img_path)
            kpt_val = self._pred_kpt_count_map.get(img_path)
            self._update_frame_item(
                self.current_index,
                pred_conf=conf_val,
                pred_kpt_count=kpt_val,
                conf_only=True,
            )
        self._rebuild_canvas()

    def _on_pred_failed(self, msg: str):
        self.btn_predict.setEnabled(True)
        self.statusBar().showMessage("Prediction failed.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
            self._clear_progress()
        QMessageBox.warning(self, "Prediction failed", msg)

    def _clear_prediction_cache(self):
        backend = self._pred_backend()
        model = self._get_pred_model_silent()
        if model:
            removed = self.infer.clear_cache(model, backend=backend)
            self.statusBar().showMessage(
                f"Cleared prediction cache ({removed} file(s)).", 3000
            )
            self._pred_conf_cache_key = None
            self._pred_conf_map = {}
            self._pred_conf_complete = False
            self._pred_kpt_count_map = {}
            self._clear_conf_display()
            self._rebuild_canvas()
            return

        resp = QMessageBox.question(
            self,
            "Clear prediction cache",
            "No prediction model selected. Clear all cached predictions?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if resp != QMessageBox.Yes:
            return
        removed = self.infer.clear_cache(None)
        self.statusBar().showMessage(
            f"Cleared all prediction caches ({removed} file(s)).", 3000
        )
        self._pred_conf_cache_key = None
        self._pred_conf_map = {}
        self._pred_conf_complete = False
        self._pred_kpt_count_map = {}
        self._clear_conf_display()
        self._rebuild_canvas()

    def _adopt_prediction_kpt(self, kpt_idx: int, x: float, y: float):
        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        preds = None
        if model:
            preds = self.infer.get_cached_pred(
                model, self.image_paths[self.current_index], backend=backend
            )

        if not self._current_frame_has_labels():
            if not preds:
                return
            # Update dragged keypoint in predictions before adopting all.
            if kpt_idx < len(preds):
                conf = preds[kpt_idx][2] if len(preds[kpt_idx]) >= 3 else 1.0
                preds[kpt_idx] = (float(x), float(y), float(conf))
            self._push_undo()
            conf_thr = self._pred_conf_default()
            self._ann.kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
            self._set_dirty()
            self._rebuild_canvas()
            self._update_info()
            self.labeling_frames.add(self.current_index)
            self._populate_frames()
            self._select_frame_in_list(self.current_index)
            if model:
                self.infer.merge_cache(
                    model,
                    {str(self.image_paths[self.current_index]): preds},
                    backend=backend,
                )
            return

        # Prompt only when user tries to drag a prediction and GT exists
        res = QMessageBox.question(
            self,
            "Replace keypoint?",
            "This frame already has labels. Replace this keypoint with the prediction you adjusted?",
        )
        if res != QMessageBox.Yes:
            self._rebuild_canvas()
            return

        self.on_place_kpt(kpt_idx, x, y, 2)
        self.labeling_frames.add(self.current_index)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

        # Update prediction cache with adjusted point (preserve existing conf if present)
        if model and preds and kpt_idx < len(preds):
            conf = preds[kpt_idx][2] if len(preds[kpt_idx]) >= 3 else 1.0
            preds[kpt_idx] = (float(x), float(y), float(conf))
            self.infer.merge_cache(
                model,
                {str(self.image_paths[self.current_index]): preds},
                backend=backend,
            )

    def apply_predictions_current(self: object) -> object:
        """apply_predictions_current method documentation."""
        if self._ann is None:
            QMessageBox.information(self, "No frame", "Select a frame first.")
            return
        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        if not model:
            return
        preds = self.infer.get_cached_pred(
            model, self.image_paths[self.current_index], backend=backend
        )
        if not preds:
            QMessageBox.information(
                self,
                "No predictions",
                "No cached predictions for this frame. Run Predict Keypoints first.",
            )
            return
        if self._current_frame_has_labels():
            res = QMessageBox.question(
                self,
                "Replace labels?",
                "This frame already has labels. Replace all keypoints with predictions?",
            )
            if res != QMessageBox.Yes:
                return
        self._push_undo()
        conf_thr = self._pred_conf_default()
        self._ann.kpts = self._preds_to_keypoints(preds, conf_thr=conf_thr)
        self._mark_dirty()
        self._rebuild_canvas()
        self._update_info()
        self.labeling_frames.add(self.current_index)
        self._populate_frames()
        self._select_frame_in_list(self.current_index)

    def predict_dataset(self: object) -> object:
        """predict_dataset method documentation."""
        if not self.image_paths:
            QMessageBox.information(self, "No frames", "No images loaded.")
            return
        backend = self._pred_backend()
        model = self._get_pred_model_or_prompt()
        if not model:
            return
        sleap_env = None
        if backend == "sleap":
            sleap_env = self._get_sleap_env()
            if not sleap_env:
                QMessageBox.warning(
                    self,
                    "No SLEAP env",
                    "Select a conda env starting with 'sleap' for SLEAP inference.",
                )
                return
            if not PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: starting", visible=True)
                self._set_busy_progress("Starting SLEAP service…")
            else:
                self._set_sleap_status("SLEAP: running", visible=True)
            if hasattr(self, "btn_sleap_start") and hasattr(self, "btn_sleap_stop"):
                self.btn_sleap_start.setEnabled(False)
                self.btn_sleap_stop.setEnabled(True)

        items = ["All frames", "Labeling set", "Unlabeled only"]
        scope, ok = QInputDialog.getItem(
            self, "Predict dataset", "Scope:", items, 0, False
        )
        if not ok:
            return
        if scope == "Labeling set":
            indices = sorted(self.labeling_frames)
        elif scope == "Unlabeled only":
            indices = [
                i for i, p in enumerate(self.image_paths) if not self._is_labeled(p)
            ]
        else:
            indices = list(range(len(self.image_paths)))

        paths = [self.image_paths[i] for i in indices]
        if not paths:
            QMessageBox.information(self, "No frames", "No frames in selected scope.")
            return

        if backend != "sleap":
            self.statusBar().showMessage(f"Predicting dataset… ({model.name})")
        self.btn_predict_bulk.setEnabled(False)
        if backend == "sleap":
            if PoseInferenceService.sleap_service_running():
                self._set_sleap_status("SLEAP: running", visible=True)

        self._last_pred_model = model
        self._last_pred_backend = backend
        sleap_batch = 4
        if backend == "sleap":
            default_batch = int(getattr(self, "_sleap_batch_predict", 4))
            batch_val, ok = QInputDialog.getInt(
                self,
                "SLEAP Batch Size",
                "Batch size:",
                default_batch,
                1,
                512,
                1,
            )
            if not ok:
                self.btn_predict_bulk.setEnabled(True)
                return
            self._sleap_batch_predict = int(batch_val)
            sleap_batch = self._sleap_batch_predict
        self._bulk_pred_thread = QThread()
        self._bulk_pred_worker = BulkPosePredictWorker(
            model_path=model,
            image_paths=paths,
            out_root=self.project.out_root,
            keypoint_names=self.project.keypoint_names,
            skeleton_edges=self.project.skeleton_edges,
            backend=backend,
            device="auto",
            imgsz=640,
            conf=self._pred_conf_default() if backend == "yolo" else 0.0,
            batch=16,
            sleap_env=sleap_env,
            sleap_device=self.combo_sleap_device.currentText().strip(),
            sleap_batch=int(sleap_batch),
            sleap_max_instances=1,
        )
        self._bulk_pred_worker.moveToThread(self._bulk_pred_thread)
        self._bulk_pred_thread.started.connect(self._bulk_pred_worker.run)
        self._bulk_pred_worker.progress.connect(self._on_bulk_pred_progress)
        self._bulk_pred_worker.finished.connect(self._on_bulk_pred_finished)
        self._bulk_pred_worker.failed.connect(self._on_bulk_pred_failed)
        self._bulk_pred_worker.finished.connect(self._bulk_pred_thread.quit)
        self._bulk_pred_worker.failed.connect(self._bulk_pred_thread.quit)
        self._bulk_pred_thread.finished.connect(self._bulk_pred_thread.deleteLater)
        self._bulk_pred_thread.start()

    def _on_bulk_pred_progress(self, done: int, total: int):
        if total > 0:
            if self.status_progress.maximum() == 0:
                self.status_progress.setRange(0, 100)
            pct = int((done / total) * 100)
            self.status_progress.setVisible(True)
            self.status_progress.setValue(min(100, max(0, pct)))
            self.statusBar().showMessage(f"Predicting dataset… {done}/{total}")
            if getattr(self, "_last_pred_backend", "yolo") == "sleap":
                self._set_sleap_status("SLEAP: running", visible=True)

    def _on_bulk_pred_finished(
        self, preds: Dict[str, List[Tuple[float, float, float]]]
    ):
        self.btn_predict_bulk.setEnabled(True)
        self._clear_progress()
        self.statusBar().showMessage("Predictions cached.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
        model = getattr(self, "_last_pred_model", None)
        backend = getattr(self, "_last_pred_backend", "yolo")
        if model:
            self.infer.merge_cache(model, preds, backend=backend)
            self._update_pred_conf_map_from_preds(preds, model=model, backend=backend)
            for path in preds.keys():
                idx = self._path_to_index.get(path)
                if idx is None:
                    try:
                        idx = self._path_to_index.get(str(Path(path).resolve()))
                    except Exception:
                        idx = None
                if idx is None:
                    continue
                conf_val = self._pred_conf_map.get(path)
                if conf_val is None:
                    try:
                        conf_val = self._pred_conf_map.get(str(Path(path).resolve()))
                    except Exception:
                        conf_val = None
                kpt_val = self._pred_kpt_count_map.get(path)
                if kpt_val is None:
                    try:
                        kpt_val = self._pred_kpt_count_map.get(
                            str(Path(path).resolve())
                        )
                    except Exception:
                        kpt_val = None
                self._update_frame_item(
                    idx, pred_conf=conf_val, pred_kpt_count=kpt_val, conf_only=True
                )
        self._rebuild_canvas()

    def _on_bulk_pred_failed(self, msg: str):
        self.btn_predict_bulk.setEnabled(True)
        self._clear_progress()
        self.statusBar().showMessage("Prediction failed.", 2000)
        if getattr(self, "_last_pred_backend", "yolo") == "sleap":
            self._set_sleap_status("SLEAP: idle", visible=True)
        QMessageBox.warning(self, "Prediction failed", msg)

    def open_training_runner(self: object) -> object:
        """open_training_runner method documentation."""
        dlg = TrainingRunnerDialog(self, self.project, self.image_paths)
        try:
            backend = self._pred_backend()
            if backend == "sleap":
                dlg.backend_combo.setCurrentText("SLEAP")
            else:
                dlg.backend_combo.setCurrentText("YOLO Pose")
            if hasattr(dlg, "_update_backend_ui"):
                dlg._update_backend_ui()
        except Exception:
            pass
        dlg.exec()

    def open_evaluation_dashboard(self: object) -> object:
        """open_evaluation_dashboard method documentation."""
        dlg = EvaluationDashboardDialog(
            self,
            self.project,
            self.image_paths,
            weights_path=(
                str(self.project.latest_pose_weights)
                if self.project.latest_pose_weights
                else None
            ),
            add_frames_callback=lambda idxs, reason="Evaluation": self._add_indices_to_labeling(
                idxs, reason
            ),
        )
        try:
            backend = self._pred_backend()
            dlg.backend_combo.setCurrentText("SLEAP" if backend == "sleap" else "YOLO")
            if backend == "sleap":
                model_path = self.sleap_model_edit.text().strip()
            else:
                model_path = self.pred_weights_edit.text().strip()
            if model_path:
                dlg.lock_model_path(model_path)
            if hasattr(dlg, "_apply_backend_ui"):
                dlg._apply_backend_ui()
        except Exception:
            pass
        dlg.exec()

    def open_active_learning(self: object) -> object:
        """open_active_learning method documentation."""
        dlg = ActiveLearningDialog(
            self,
            self.project,
            self.image_paths,
            self._is_labeled,
            set(self.labeling_frames),
            add_frames_callback=lambda idxs, reason="Active learning": self._add_indices_to_labeling(
                idxs, reason
            ),
        )
        try:
            backend = self._pred_backend()
            dlg.backend_combo.setCurrentText("SLEAP" if backend == "sleap" else "YOLO")
            if backend == "sleap":
                model_path = self.sleap_model_edit.text().strip()
            else:
                model_path = self.pred_weights_edit.text().strip()
            if model_path:
                dlg.lock_model_path(model_path)
            if hasattr(dlg, "_apply_backend_ui"):
                dlg._apply_backend_ui()
        except Exception:
            pass
        dlg.exec()

    # Backwards/alternate name used in older menu wiring.
    def open_active_learning_sampler(self: object) -> object:
        """open_active_learning_sampler method documentation."""
        self.open_active_learning()

    def _load_metadata_ui(self):
        if not self.image_paths:
            return
        self._setting_meta = True
        img_path = str(self.image_paths[self.current_index])
        meta = self.metadata_manager.get_metadata(img_path)
        for tag, cb in self.meta_tags.items():
            cb.setChecked(tag in meta.tags)
        self.meta_notes.setText(meta.notes or "")
        self._setting_meta = False

    def _on_meta_changed(self, *_args):
        if self._setting_meta or not self.image_paths:
            return
        img_path = str(self.image_paths[self.current_index])
        meta = self.metadata_manager.get_metadata(img_path)
        meta.tags = {t for t, cb in self.meta_tags.items() if cb.isChecked()}
        meta.notes = self.meta_notes.text().strip()
        self.metadata_manager.save()

    def _load_cluster_ids_from_csv(self, csv_path: Path) -> Optional[List[int]]:
        if not csv_path.exists():
            return None
        mapping = {}
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img = row.get("image") or row.get("image_path") or row.get("path")
                    cid = row.get("cluster_id") or row.get("cluster")
                    if img is None or cid is None:
                        continue
                    try:
                        mapping[str(Path(img).resolve())] = int(float(cid))
                    except Exception:
                        continue
        except Exception:
            return None

        if not mapping:
            return None

        cluster_ids: List[int] = []
        for p in self.image_paths:
            key = str(p.resolve())
            if key in mapping:
                cluster_ids.append(mapping[key])
            else:
                cluster_ids.append(-1)
        return cluster_ids

    def _on_clusters_updated(self):
        self._cluster_ids_cache = None
        self._cluster_ids_mtime = None
        self._populate_frames()


# -----------------------------
# Bootstrap / project discovery
# -----------------------------
def find_project(images_dir: Path, out_root: Optional[Path]) -> Optional[Path]:
    """
    Tries to find an existing project json.
    Search order:
      1) out_root/pose_project.json
      2) out_root/labels/pose_project.json
      3) images_dir/pose_project.json
      4) images_dir/labels/pose_project.json
    """
    candidates: List[Path] = []
    if out_root:
        candidates += [
            out_root / DEFAULT_PROJECT_NAME,
            out_root / "labels" / DEFAULT_PROJECT_NAME,
        ]
    candidates += [
        images_dir / DEFAULT_PROJECT_NAME,
        images_dir / "labels" / DEFAULT_PROJECT_NAME,
        images_dir.parent / "labels" / DEFAULT_PROJECT_NAME,  # Also check parent/labels
    ]
    for p in candidates:
        if p.exists():
            # Verify this project actually matches our images_dir
            try:
                proj_data = json.loads(p.read_text(encoding="utf-8"))
                base = p.parent
                proj_images_dir = _resolve_project_path(
                    Path(proj_data["images_dir"]), base, proj_data
                )
                if proj_images_dir.resolve() == images_dir.resolve():
                    return p
                # If the project file is colocated with images_dir/labels,
                # allow opening and repair paths for portability.
                if p.parent in {
                    images_dir,
                    images_dir / "labels",
                    images_dir.parent / "labels",
                }:
                    return p
            except (json.JSONDecodeError, KeyError, OSError):
                continue
    return None


def _repair_project_paths(proj: Project, project_path: Path, images_dir: Path) -> bool:
    """Repair project paths when moved across machines. Returns True if changed."""
    changed = False
    if proj.images_dir.resolve() != images_dir.resolve():
        proj.images_dir = images_dir
        changed = True

    labels_dir = project_path.parent
    if proj.labels_dir.resolve() != labels_dir.resolve():
        proj.labels_dir = labels_dir
        changed = True

    out_root = labels_dir.parent
    if proj.out_root.resolve() != out_root.resolve():
        proj.out_root = out_root
        changed = True

    if proj.project_path.resolve() != project_path.resolve():
        proj.project_path = project_path
        changed = True

    return changed


def load_project_with_repairs(project_path: Path, images_dir: Path) -> Project:
    """load_project_with_repairs function documentation."""
    proj = Project.from_json(project_path)
    if _repair_project_paths(proj, project_path, images_dir):
        proj.project_path.write_text(
            json.dumps(proj.to_json(), indent=2), encoding="utf-8"
        )
    return proj


def create_project_via_wizard(
    images_dir: Path, out_root_hint: Optional[Path] = None
) -> Optional[Project]:
    """create_project_via_wizard function documentation."""
    wiz = ProjectWizard(images_dir, existing=None)
    if out_root_hint is not None:
        wiz.out_root.setText(str(out_root_hint.resolve()))
        wiz.labels_dir.setText(str((out_root_hint.resolve() / "labels")))

    if wiz.exec() != QDialog.Accepted:
        return None

    out_root, labels_dir = wiz.get_paths()
    class_names = wiz.get_classes()
    kpt_names = wiz.get_keypoints()
    edges = wiz.get_edges()
    autosave, pad = wiz.get_options()

    out_root.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    project_path = labels_dir / DEFAULT_PROJECT_NAME
    proj = Project(
        images_dir=images_dir,
        out_root=out_root,
        labels_dir=labels_dir,
        project_path=project_path,
        class_names=class_names,
        keypoint_names=kpt_names,
        skeleton_edges=edges,
        autosave=autosave,
        bbox_pad_frac=pad,
    )
    project_path.write_text(json.dumps(proj.to_json(), indent=2), encoding="utf-8")
    return proj


def create_empty_startup_project() -> Project:
    """Create a minimal placeholder project for startup-without-dataset mode."""
    root = (Path.home() / ".posekit" / "startup").resolve()
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return Project(
        images_dir=root,
        out_root=root,
        labels_dir=labels_dir,
        project_path=labels_dir / DEFAULT_PROJECT_NAME,
        class_names=["object"],
        keypoint_names=["kp1", "kp2"],
        skeleton_edges=[(0, 1)],
        autosave=True,
    )


def parse_args() -> object:
    """parse_args function documentation."""
    ap = argparse.ArgumentParser(description="PoseKit labeler")
    ap.add_argument(
        "images",
        nargs="?",
        default=None,
        help="Folder of images (recursively scanned)",
    )
    ap.add_argument(
        "--out", default=None, help="Output root (default: <images>/../pose_project)"
    )
    ap.add_argument("--project", default=None, help="Explicit project json path")
    ap.add_argument(
        "--new", action="store_true", help="Force setup wizard even if project exists"
    )
    return ap.parse_args()


def main() -> object:
    """main function documentation."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    app = QApplication(sys.argv)

    # Startup-without-input mode: open full UI first, then show chooser overlay.
    if not args.images and not args.project:
        proj = create_empty_startup_project()
        win = MainWindow(proj, [])
        win.resize(1500, 860)
        win.showMaximized()
        QTimer.singleShot(0, win.show_startup_open_overlay)
        sys.exit(app.exec())

    if args.images:
        images_dir = Path(args.images).expanduser().resolve()
        if not images_dir.exists():
            print(f"Images dir not found: {images_dir}", file=sys.stderr)
            sys.exit(2)
    else:
        images_dir = None

    out_root = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (
            (images_dir.parent / "pose_project").resolve()
            if images_dir is not None
            else None
        )
    )

    proj: Optional[Project] = None

    if args.project:
        project_path = Path(args.project).expanduser().resolve()
        if project_path.exists() and not args.new:
            if images_dir is None:
                try:
                    data = json.loads(project_path.read_text(encoding="utf-8"))
                    base = project_path.parent
                    images_dir = _resolve_project_path(
                        Path(data["images_dir"]), base, data
                    )
                except Exception as e:
                    print(f"Failed to read project images_dir: {e}", file=sys.stderr)
                    sys.exit(2)
            proj = load_project_with_repairs(project_path, images_dir)

    if proj is None and not args.new:
        found = find_project(images_dir, out_root) if images_dir is not None else None
        if found:
            proj = load_project_with_repairs(found, images_dir)

    if proj is None:
        if images_dir is None:
            print("No images directory available to create a new project.", file=sys.stderr)
            sys.exit(2)
        proj = create_project_via_wizard(images_dir, out_root_hint=out_root)
        if proj is None:
            sys.exit(0)

    imgs = list_images(proj.images_dir)
    if not imgs:
        QMessageBox.critical(
            None, "No images", f"No images found under: {proj.images_dir}"
        )
        sys.exit(2)

    win = MainWindow(proj, imgs)
    win.resize(1500, 860)
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
