#!/usr/bin/env python3
"""
Dialogs for PoseKit Labeler extensions.
"""

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging
import subprocess
import sys
import tempfile
import os
import shutil
import re
import numpy as np
import yaml
import gc

from PySide6.QtCore import Qt, QSize, QThread, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage, QPen

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
    QFormLayout,
    QGroupBox,
    QTextEdit,
    QLineEdit,
    QPlainTextEdit,
    QFileDialog,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QStackedWidget,
    QWidget,
    QScrollArea,
    QGridLayout,
)

# Settings helpers
try:
    from .pose_inference import PoseInferenceService
except ImportError:
    try:
        from pose_inference import PoseInferenceService
    except Exception:
        PoseInferenceService = None

try:
    from .pose_label import load_ui_settings, save_ui_settings
except ImportError:
    try:
        from pose_label import load_ui_settings, save_ui_settings
    except Exception:
        def load_ui_settings():
            return {}

        def save_ui_settings(_settings: Dict):
            return

# Handle both package imports and direct script execution
try:
    from .pose_label_extensions import (
        MetadataManager,
        cluster_stratified_split,
        cluster_kfold_split,
        save_split_files,
        EmbeddingWorker,
        cluster_embeddings_cosine,
        pick_frames_stratified,
        list_labeled_indices,
        build_yolo_pose_dataset,
        load_yolo_pose_label,
    )
    from ..utils.gpu_utils import (
        CUDA_AVAILABLE,
        MPS_AVAILABLE,
        TORCH_CUDA_AVAILABLE,
        ROCM_AVAILABLE,
    )
except ImportError:
    from pose_label_extensions import (
        MetadataManager,
        cluster_stratified_split,
        cluster_kfold_split,
        save_split_files,
        EmbeddingWorker,
        cluster_embeddings_cosine,
        pick_frames_stratified,
        list_labeled_indices,
        build_yolo_pose_dataset,
        load_yolo_pose_label,
    )

    try:
        from multi_tracker.utils.gpu_utils import (
            CUDA_AVAILABLE,
            MPS_AVAILABLE,
            TORCH_CUDA_AVAILABLE,
            ROCM_AVAILABLE,
        )
    except ImportError:
        # Fallback if gpu_utils not available
        CUDA_AVAILABLE = False
        MPS_AVAILABLE = False
        TORCH_CUDA_AVAILABLE = False
        ROCM_AVAILABLE = False

logger = logging.getLogger("pose_label.dialogs")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _make_pose_infer(out_root: Path, keypoint_names: List[str]):
    if PoseInferenceService is None:
        raise RuntimeError("PoseInferenceService not available. Check imports.")
    return PoseInferenceService(out_root, keypoint_names)


def _make_loss_plot_image(
    train_vals: List[float],
    val_vals: List[float],
    width: int = 520,
    height: int = 220,
) -> QImage:
    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(30, 30, 30))
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)

    pad = 24
    w = width - 2 * pad
    h = height - 2 * pad
    painter.setPen(QPen(QColor(80, 80, 80), 1))
    painter.drawRect(pad, pad, w, h)

    all_vals = [v for v in train_vals + val_vals if v is not None]
    if not all_vals:
        painter.end()
        return img

    vmin = min(all_vals)
    vmax = max(all_vals)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    def _plot_series(vals: List[float], color: QColor):
        if len(vals) < 2:
            return
        pen = QPen(color, 2)
        painter.setPen(pen)
        n = len(vals)
        for i in range(1, n):
            if vals[i - 1] is None or vals[i] is None:
                continue
            x0 = pad + (w * (i - 1) / (n - 1))
            x1 = pad + (w * i / (n - 1))
            y0 = pad + h * (1.0 - (vals[i - 1] - vmin) / (vmax - vmin))
            y1 = pad + h * (1.0 - (vals[i] - vmin) / (vmax - vmin))
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))

    _plot_series(train_vals, QColor(80, 200, 120))
    _plot_series(val_vals, QColor(255, 140, 80))

    painter.end()
    return img


def get_available_devices():
    """Get list of available compute devices based on gpu_utils flags."""
    devices = ["auto", "cpu"]
    if CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE or ROCM_AVAILABLE:
        devices.append("cuda")
    if MPS_AVAILABLE:
        devices.append("mps")
    return devices




def _is_cuda_device(device: str) -> bool:
    d = (device or "").strip().lower()
    if d in {"cuda", "gpu"}:
        return True
    if d.startswith("cuda:"):
        return True
    return d.isdigit()


def _is_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def _maybe_limit_cuda_memory(log_fn=None, fraction: float = 0.9):
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(fraction))
            if log_fn:
                log_fn(f"CUDA memory cap set to {int(fraction * 100)}% of GPU.")
    except Exception:
        pass


def _maybe_empty_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_dialog_settings(key: str) -> Dict:
    settings = load_ui_settings()
    return settings.get("dialogs", {}).get(key, {})


def _save_dialog_settings(key: str, data: Dict) -> None:
    settings = load_ui_settings()
    dialogs = settings.get("dialogs", {})
    dialogs[key] = data
    settings["dialogs"] = dialogs
    save_ui_settings(settings)


def get_yolo_pose_base_models() -> List[str]:
    """Known YOLO Pose base models from Ultralytics docs."""
    return [
        "yolo26n-pose.pt",
        "yolo26s-pose.pt",
        "yolo26m-pose.pt",
        "yolo26l-pose.pt",
        "yolo26x-pose.pt",
    ]


def list_images_in_dir(images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths


def _label_path_for_image(base_dir: Path, img_path: Path) -> Optional[Path]:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        lbl = Path(*parts).with_suffix(".txt")
        if lbl.exists():
            return lbl
    # fallback to base/labels/<stem>.txt
    lbl = base_dir / "labels" / f"{img_path.stem}.txt"
    if lbl.exists():
        return lbl
    return None


def load_yolo_dataset_items(
    dataset_yaml: Path,
) -> Tuple[List[Tuple[Path, Path]], Dict[str, object]]:
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    base = Path(data.get("path", dataset_yaml.parent)).expanduser().resolve()
    train = data.get("train")
    val = data.get("val")
    names = data.get("names", {})
    kpt_shape = data.get("kpt_shape")
    kpt_names = data.get("kpt_names")

    def _resolve_source(src):
        if src is None:
            return []
        p = Path(src)
        if not p.is_absolute():
            p = (base / p).resolve()
        if p.is_file() and p.suffix.lower() in [".txt"]:
            lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()]
            out = []
            for l in lines:
                if not l:
                    continue
                lp = Path(l)
                if not lp.is_absolute():
                    lp = (base / lp).resolve()
                out.append(lp)
            return out
        if p.is_dir():
            return list_images_in_dir(p)
        return []

    images = _resolve_source(train) + _resolve_source(val)
    items: List[Tuple[Path, Path]] = []
    for img in images:
        lbl = _label_path_for_image(base, img)
        if lbl is not None:
            items.append((img, lbl))

    info = {
        "base": str(base),
        "names": names,
        "kpt_shape": kpt_shape,
        "kpt_names": kpt_names,
    }
    return items, info


class DatasetSplitDialog(QDialog):
    """Dialog for creating cluster-stratified train/val/test splits."""

    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        cluster_ids: Optional[List[int]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Dataset Split (Cluster-Stratified)")
        self.setMinimumSize(QSize(600, 400))

        self.project = project
        self.image_paths = image_paths
        self.cluster_ids = cluster_ids

        layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Train/Val/Test", "K-Fold Cross-Validation"])
        content_layout.addWidget(self.mode_combo)

        # Train/Val/Test parameters
        self.tvt_widget = QGroupBox("Train/Val/Test Parameters")
        tvt_layout = QFormLayout(self.tvt_widget)

        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.7)
        tvt_layout.addRow("Train fraction:", self.train_spin)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.15)
        tvt_layout.addRow("Val fraction:", self.val_spin)

        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.15)
        tvt_layout.addRow("Test fraction:", self.test_spin)

        content_layout.addWidget(self.tvt_widget)

        # K-Fold parameters
        self.kfold_widget = QGroupBox("K-Fold Parameters")
        kfold_layout = QFormLayout(self.kfold_widget)

        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_layout.addRow("Number of folds:", self.kfold_spin)

        content_layout.addWidget(self.kfold_widget)
        self.kfold_widget.setVisible(False)

        # Common parameters
        common_group = QGroupBox("Common Parameters")
        common_layout = QFormLayout(common_group)

        self.min_per_cluster_spin = QSpinBox()
        self.min_per_cluster_spin.setRange(1, 100)
        self.min_per_cluster_spin.setValue(1)
        common_layout.addRow("Min per cluster:", self.min_per_cluster_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        common_layout.addRow("Random seed:", self.seed_spin)

        self.split_name_edit = QLineEdit("split")
        common_layout.addRow("Split name:", self.split_name_edit)

        content_layout.addWidget(common_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Split")
        self.btn_close = QPushButton("Close")
        btn_layout.addWidget(self.btn_generate)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        content_layout.addLayout(btn_layout)

        # Wiring
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_generate.clicked.connect(self._generate_split)
        self.btn_close.clicked.connect(self.reject)

        self._apply_settings()

    def _apply_settings(self):
        settings = _load_dialog_settings("dataset_split")
        if not settings:
            return
        mode = settings.get("mode_index")
        if mode is not None:
            self.mode_combo.setCurrentIndex(int(mode))
        self.train_spin.setValue(float(settings.get("train_frac", self.train_spin.value())))
        self.val_spin.setValue(float(settings.get("val_frac", self.val_spin.value())))
        self.test_spin.setValue(float(settings.get("test_frac", self.test_spin.value())))
        self.kfold_spin.setValue(int(settings.get("kfold", self.kfold_spin.value())))
        self.min_per_cluster_spin.setValue(
            int(settings.get("min_per_cluster", self.min_per_cluster_spin.value()))
        )
        self.seed_spin.setValue(int(settings.get("seed", self.seed_spin.value())))
        self.split_name_edit.setText(settings.get("split_name", self.split_name_edit.text()))

    def _save_settings(self):
        _save_dialog_settings(
            "dataset_split",
            {
                "mode_index": int(self.mode_combo.currentIndex()),
                "train_frac": float(self.train_spin.value()),
                "val_frac": float(self.val_spin.value()),
                "test_frac": float(self.test_spin.value()),
                "kfold": int(self.kfold_spin.value()),
                "min_per_cluster": int(self.min_per_cluster_spin.value()),
                "seed": int(self.seed_spin.value()),
                "split_name": self.split_name_edit.text().strip(),
            },
        )

    def closeEvent(self, event):
        # Drop large arrays to reduce memory footprint between sessions.
        self._emb = None
        self._eligible_indices = None
        self._cluster = None
        try:
            gc.collect()
        except Exception:
            pass
        self._save_settings()
        super().closeEvent(event)

    def _on_mode_changed(self, index: int):
        """Toggle visibility of parameter groups."""
        is_kfold = index == 1
        self.tvt_widget.setVisible(not is_kfold)
        self.kfold_widget.setVisible(is_kfold)

    def _generate_split(self):
        """Generate the dataset split."""
        if not self.cluster_ids:
            QMessageBox.warning(
                self,
                "No Clusters",
                "No cluster information available. Please run clustering first in Smart Select.",
            )
            return

        output_dir = self.project.out_root / "splits"
        split_name = self.split_name_edit.text().strip() or "split"
        seed = self.seed_spin.value()

        try:
            if self.mode_combo.currentIndex() == 0:
                # Train/Val/Test
                train_frac = self.train_spin.value()
                val_frac = self.val_spin.value()
                test_frac = self.test_spin.value()
                min_per = self.min_per_cluster_spin.value()

                train_idx, val_idx, test_idx = cluster_stratified_split(
                    self.image_paths,
                    self.cluster_ids,
                    train_frac=train_frac,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    min_per_cluster=min_per,
                    seed=seed,
                )

                save_split_files(
                    output_dir,
                    self.image_paths,
                    train_idx,
                    val_idx,
                    test_idx,
                    split_name=split_name,
                )

                QMessageBox.information(
                    self,
                    "Split Generated",
                    f"Train/Val/Test split created:\n"
                    f"Train: {len(train_idx)} frames\n"
                    f"Val: {len(val_idx)} frames\n"
                    f"Test: {len(test_idx)} frames\n\n"
                    f"Saved to: {output_dir}",
                )

            else:
                # K-Fold
                n_folds = self.kfold_spin.value()

                splits = cluster_kfold_split(
                    self.image_paths,
                    self.cluster_ids,
                    n_folds=n_folds,
                    seed=seed,
                )

                # Save each fold
                for fold_idx, (train_idx, val_idx) in enumerate(splits):
                    fold_name = f"{split_name}_fold{fold_idx + 1}"
                    save_split_files(
                        output_dir,
                        self.image_paths,
                        train_idx,
                        val_idx,
                        [],  # No test set in k-fold
                        split_name=fold_name,
                    )

                QMessageBox.information(
                    self,
                    "Splits Generated",
                    f"{n_folds}-Fold cross-validation splits created.\n\n"
                    f"Saved to: {output_dir}",
                )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate split:\n{str(e)}")
            logger.error(f"Split generation failed: {e}", exc_info=True)


class FrameMetadataDialog(QDialog):
    """Dialog for viewing/editing frame metadata and tags."""

    def __init__(self, parent, image_path: str, metadata_manager: MetadataManager):
        super().__init__(parent)
        self.setWindowTitle(f"Frame Metadata: {Path(image_path).name}")
        self.setMinimumSize(QSize(500, 400))

        self.image_path = image_path
        self.metadata_manager = metadata_manager
        self.metadata = metadata_manager.get_metadata(image_path)

        layout = QVBoxLayout(self)

        # Tags section
        tags_group = QGroupBox("Tags")
        tags_layout = QVBoxLayout(tags_group)

        # Common tags
        self.tag_checkboxes = {}
        common_tags = [
            "occluded",
            "weird_posture",
            "motion_blur",
            "poor_lighting",
            "partial_view",
            "unclear",
        ]

        for tag in common_tags:
            cb = QCheckBox(tag)
            cb.setChecked(tag in self.metadata.tags)
            self.tag_checkboxes[tag] = cb
            tags_layout.addWidget(cb)

        layout.addWidget(tags_group)

        # Notes section
        notes_group = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_group)

        self.notes_edit = QTextEdit()
        self.notes_edit.setPlainText(self.metadata.notes)
        self.notes_edit.setMaximumHeight(100)
        notes_layout.addWidget(self.notes_edit)

        layout.addWidget(notes_group)

        # Cluster info
        if self.metadata.cluster_id is not None:
            cluster_label = QLabel(f"Cluster ID: {self.metadata.cluster_id}")
            layout.addWidget(cluster_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save")
        self.btn_cancel = QPushButton("Cancel")
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

        # Wiring
        self.btn_save.clicked.connect(self._save)
        self.btn_cancel.clicked.connect(self.reject)

    def _save(self):
        """Save metadata changes."""
        # Update tags
        for tag, cb in self.tag_checkboxes.items():
            if cb.isChecked():
                self.metadata_manager.add_tag(self.image_path, tag)
            else:
                self.metadata_manager.remove_tag(self.image_path, tag)

        # Update notes
        notes = self.notes_edit.toPlainText().strip()
        self.metadata_manager.set_notes(self.image_path, notes)

        self.accept()


# -----------------------------
# Smart Select Dialog
# -----------------------------


class SmartSelectDialog(QDialog):
    """Dialog for smart frame selection using embeddings and clustering."""

    def __init__(self, parent, project, image_paths: List[Path], is_labeled_fn):
        super().__init__(parent)
        self.setWindowTitle("Smart Select (Embeddings)")
        self.setMinimumSize(QSize(720, 420))
        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn

        self._emb = None
        self._eligible_indices = None
        self._cluster = None

        layout = QVBoxLayout(self)

        # --- scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.cb_scope = QComboBox()
        self.cb_scope.addItems(["Unlabeled only", "All frames", "Labeling set"])
        self.cb_scope.setToolTip(
            "Which frames to select from:\n"
            "• Unlabeled only: Only frames not yet labeled\n"
            "• All frames: All available frames\n"
            "• Labeling set: Only frames in current labeling set"
        )
        scope_row.addWidget(self.cb_scope, 1)
        self.cb_exclude_in_labeling = QCheckBox("Exclude already in labeling set")
        self.cb_exclude_in_labeling.setChecked(True)
        self.cb_exclude_in_labeling.setToolTip(
            "Don't select frames already being labeled"
        )
        scope_row.addWidget(self.cb_exclude_in_labeling)
        layout.addLayout(scope_row)

        # --- embeddings
        emb_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            [
                # DINO v2 models (recommended for general image understanding)
                "timm/vit_base_patch14_dinov2.lvd142m",  # ~86M params, excellent quality
                "timm/vit_small_patch14_dinov2.lvd142m",  # ~22M params, good balance
                "timm/vit_large_patch14_dinov2.lvd142m",  # ~304M params, best quality
                "timm/vit_giant_patch14_dinov2.lvd142m",  # ~1.1B params, huge model
                # CLIP models (text+image understanding, good for semantic search)
                "timm/vit_base_patch32_clip_224.openai",  # ~88M params, good balance
                "timm/vit_bigG_14_clip_224.laion400M_e32",  # ~2B params, best quality
                "timm/convnext_base_w_clip.laion2b_s29B_b131k_ft_in1k",  # ~88M params, efficient
                # ResNet models (faster, smaller)
                "timm/resnet50.a1_in1k",  # ~25M params, fast
                "timm/resnet18.a1_in1k",  # ~11M params, very fast
                "timm/resnet101.a1_in1k",  # ~44M params, more capacity
                # EfficientNet models (efficient, good accuracy/speed tradeoff)
                "timm/efficientnet_b0.ra_in1k",  # ~5M params, very efficient
                "timm/efficientnet_b3.ra2_in1k",  # ~12M params, good balance
                "timm/efficientnet_b5.sw_in12k_ft_in1k",  # ~30M params, high quality
                # MobileNet models (fastest, smallest)
                "timm/mobilenetv3_small_100.lamb_in1k",  # ~2M params, mobile
                "timm/mobilenetv3_large_100.ra_in1k",  # ~5M params, mobile
                # ConvNeXt models (modern convnets)
                "timm/convnext_tiny.fb_in1k",  # ~28M params, modern
                "timm/convnext_base.fb_in1k",  # ~88M params, powerful
            ]
        )
        self.model_combo.setToolTip(
            "Choose embedding model:\n"
            "• DINO v2 (vit_*): Best for detailed feature understanding, good for animal pose\n"
            "• CLIP (vit_base_patch32): Good for semantic/contextual understanding\n"
            "• ResNet: Balanced speed/quality, well-tested\n"
            "• EfficientNet: Good speed/quality tradeoff\n"
            "• MobileNet: Fastest, smallest - for resource-constrained systems\n"
            "• ConvNeXt: Modern architecture, good quality"
        )
        emb_form.addRow("Model:", self.model_combo)

        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(get_available_devices())
        self.dev_combo.setToolTip(
            "Compute device:\n"
            "• auto: Automatically select available GPU (CUDA/MPS) or CPU\n"
            "• cpu: Use CPU (slow but reliable)\n"
            "• cuda: NVIDIA GPU (fastest if available)\n"
            "• mps: Apple Metal Performance Shaders (fast on Mac)"
        )
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        self.batch_spin.setToolTip(
            "Batch size: Larger = faster but more memory.\n"
            "Start with 32, increase if OOM errors don't occur."
        )
        dev_row.addWidget(self.batch_spin)
        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        self.max_side_spin.setToolTip(
            "Max image size (pixels): Resize crops to this dimension.\n"
            "Larger = more detail but slower and more memory.\n"
            "0 = auto-detect from crop size."
        )
        dev_row.addWidget(self.max_side_spin)
        emb_form.addRow("Device / batch:", dev_row)

        self.cb_use_enhance = QCheckBox("Use Enhance (CLAHE+unsharp)")
        self.cb_use_enhance.setChecked(bool(self.project.enhance_enabled))
        self.cb_use_enhance.setToolTip(
            "Apply image enhancement:\n"
            "• CLAHE: Adaptive contrast enhancement\n"
            "• Unsharp: Sharpening filter\n"
            "Improves visibility of fine details in dark/low-contrast images."
        )
        emb_form.addRow("", self.cb_use_enhance)

        layout.addLayout(emb_form)

        # --- compute button + progress
        row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute / Load cached embeddings")
        self.btn_cancel = QPushButton("Cancel compute")
        self.btn_cancel.setEnabled(False)
        row.addWidget(self.btn_compute, 1)
        row.addWidget(self.btn_cancel)
        layout.addLayout(row)

        self.progress = QDoubleSpinBox()
        self.progress.setRange(0.0, 1.0)
        self.progress.setSingleStep(0.01)
        self.progress.setValue(0.0)
        self.progress.setEnabled(False)
        layout.addWidget(self.progress)

        self.lbl_status = QLabel("")
        layout.addWidget(self.lbl_status)

        # --- selection params
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Add frames:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50000)
        self.n_spin.setValue(50)
        self.n_spin.setToolTip("Number of frames to select for labeling")
        sel_row.addWidget(self.n_spin)

        sel_row.addWidget(QLabel("Clusters:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 5000)
        self.k_spin.setValue(20)
        self.k_spin.setToolTip(
            "Number of clusters for diversity. Higher = more diversity"
        )
        sel_row.addWidget(self.k_spin)

        sel_row.addWidget(QLabel("Min/cluster:"))
        self.min_per_spin = QSpinBox()
        self.min_per_spin.setRange(1, 1000)
        self.min_per_spin.setValue(1)
        self.min_per_spin.setToolTip(
            "Minimum frames per cluster. Min frames = clusters × min/cluster"
        )
        sel_row.addWidget(self.min_per_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["centroid_then_diverse", "centroid"])
        self.strategy_combo.setToolTip(
            "Selection strategy:\n"
            "• centroid_then_diverse: Pick cluster centers first, then fill with diverse samples\n"
            "• centroid: Only pick cluster centers (fastest, fewer frames)"
        )
        sel_row.addWidget(QLabel("Strategy:"))
        sel_row.addWidget(self.strategy_combo, 1)

        layout.addLayout(sel_row)

        # Connect signals to update minimum frames dynamically
        self.k_spin.valueChanged.connect(self._update_min_frames)
        self.min_per_spin.valueChanged.connect(self._update_min_frames)

        # --- Sampling options
        opt_row = QHBoxLayout()

        self.cb_filter_duplicates = QCheckBox("Filter near-duplicates")
        opt_row.addWidget(self.cb_filter_duplicates)

        opt_row.addWidget(QLabel("Threshold:"))
        self.dup_threshold_spin = QDoubleSpinBox()
        self.dup_threshold_spin.setRange(0.5, 1.0)
        self.dup_threshold_spin.setSingleStep(0.05)
        self.dup_threshold_spin.setValue(0.95)
        self.dup_threshold_spin.setEnabled(False)
        opt_row.addWidget(self.dup_threshold_spin)
        self.cb_filter_duplicates.toggled.connect(self.dup_threshold_spin.setEnabled)

        opt_row.addWidget(QLabel("   "))
        self.cb_prefer_unlabeled = QCheckBox("Prefer unlabeled")
        self.cb_prefer_unlabeled.setChecked(True)
        opt_row.addWidget(self.cb_prefer_unlabeled)

        opt_row.addStretch()
        layout.addLayout(opt_row)

        # --- clustering method row
        cluster_row = QHBoxLayout()
        cluster_row.addWidget(QLabel("Clustering method:"))
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["hierarchical", "linkage"])
        self.cluster_method_combo.setCurrentIndex(0)  # default to hierarchical
        cluster_row.addWidget(self.cluster_method_combo, 1)
        cluster_row.addStretch(2)
        layout.addLayout(cluster_row)

        # --- preview text
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        layout.addWidget(self.preview, 1)

        # --- bottom buttons
        bottom = QHBoxLayout()
        self.btn_preview = QPushButton("Run Clustering")
        self.btn_add = QPushButton("Add to Labeling Set && Close")
        self.btn_save_csv = QPushButton("Save clusters CSV…")
        self.btn_explorer = QPushButton("Embedding Explorer…")
        self.btn_explorer.setEnabled(False)
        self.btn_close = QPushButton("Close Without Saving")
        bottom.addWidget(self.btn_preview)
        bottom.addWidget(self.btn_add)
        bottom.addWidget(self.btn_save_csv)
        bottom.addWidget(self.btn_explorer)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

        self.btn_preview.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.btn_explorer.setEnabled(False)

        # wiring
        self.btn_close.clicked.connect(self.reject)
        self.btn_add.clicked.connect(self.accept)
        self.btn_compute.clicked.connect(self._compute)
        self.btn_cancel.clicked.connect(self._cancel_compute)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_explorer.clicked.connect(self._open_explorer)

        self.selected_indices: List[int] = []

        self._thread = None
        self._worker = None

        # Connect signals to update minimum frames dynamically
        self.k_spin.valueChanged.connect(self._update_min_frames)
        self.min_per_spin.valueChanged.connect(self._update_min_frames)
        # Set initial minimum
        self._update_min_frames()

        # Connect signals to update minimum frames dynamically
        self.k_spin.valueChanged.connect(self._update_min_frames)
        self.min_per_spin.valueChanged.connect(self._update_min_frames)
        # Set initial minimum
        self._update_min_frames()

        self._apply_settings()

    def _apply_settings(self):
        settings = _load_dialog_settings("smart_select")
        if not settings:
            return
        self.cb_scope.setCurrentText(settings.get("scope", self.cb_scope.currentText()))
        self.cb_exclude_in_labeling.setChecked(
            bool(settings.get("exclude_labeling", self.cb_exclude_in_labeling.isChecked()))
        )
        self.model_combo.setCurrentText(settings.get("model", self.model_combo.currentText()))
        self.dev_combo.setCurrentText(settings.get("device", self.dev_combo.currentText()))
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.max_side_spin.setValue(int(settings.get("max_side", self.max_side_spin.value())))
        self.cb_use_enhance.setChecked(
            bool(settings.get("use_enhance", self.cb_use_enhance.isChecked()))
        )
        self.n_spin.setValue(int(settings.get("n", self.n_spin.value())))
        self.k_spin.setValue(int(settings.get("k", self.k_spin.value())))
        self.min_per_spin.setValue(int(settings.get("min_per", self.min_per_spin.value())))
        self.strategy_combo.setCurrentText(settings.get("strategy", self.strategy_combo.currentText()))
        self.cb_filter_duplicates.setChecked(
            bool(settings.get("filter_dups", self.cb_filter_duplicates.isChecked()))
        )
        self.dup_threshold_spin.setValue(float(settings.get("dup_thresh", self.dup_threshold_spin.value())))
        self.cb_prefer_unlabeled.setChecked(
            bool(settings.get("prefer_unlabeled", self.cb_prefer_unlabeled.isChecked()))
        )
        self.cluster_method_combo.setCurrentText(
            settings.get("cluster_method", self.cluster_method_combo.currentText())
        )

    def _save_settings(self):
        _save_dialog_settings(
            "smart_select",
            {
                "scope": self.cb_scope.currentText(),
                "exclude_labeling": bool(self.cb_exclude_in_labeling.isChecked()),
                "model": self.model_combo.currentText().strip(),
                "device": self.dev_combo.currentText(),
                "batch": int(self.batch_spin.value()),
                "max_side": int(self.max_side_spin.value()),
                "use_enhance": bool(self.cb_use_enhance.isChecked()),
                "n": int(self.n_spin.value()),
                "k": int(self.k_spin.value()),
                "min_per": int(self.min_per_spin.value()),
                "strategy": self.strategy_combo.currentText(),
                "filter_dups": bool(self.cb_filter_duplicates.isChecked()),
                "dup_thresh": float(self.dup_threshold_spin.value()),
                "prefer_unlabeled": bool(self.cb_prefer_unlabeled.isChecked()),
                "cluster_method": self.cluster_method_combo.currentText(),
            },
        )

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _build_eligible_indices(self) -> List[int]:
        scope = self.cb_scope.currentText()
        if scope == "All frames":
            idxs = list(range(len(self.image_paths)))
        elif scope == "Labeling set":
            # parent (MainWindow) should set this property before opening, but fallback:
            idxs = list(getattr(self.parent(), "labeling_frames", set()))
        else:
            # Unlabeled only
            idxs = [
                i for i, p in enumerate(self.image_paths) if not self.is_labeled_fn(p)
            ]
        # optionally exclude already in labeling set
        if self.cb_exclude_in_labeling.isChecked():
            in_labeling = set(getattr(self.parent(), "labeling_frames", set()))
            idxs = [i for i in idxs if i not in in_labeling]
        return sorted(idxs)

    def _update_min_frames(self):
        """Update minimum frames to be at least clusters * min_per_cluster."""
        min_frames = self.k_spin.value() * self.min_per_spin.value()
        self.n_spin.setMinimum(min_frames)
        if self.n_spin.value() < min_frames:
            self.n_spin.setValue(min_frames)

    def _update_min_frames(self):
        """Update minimum frames to be at least clusters * min_per_cluster."""
        min_frames = self.k_spin.value() * self.min_per_spin.value()
        self.n_spin.setMinimum(min_frames)
        # If current value is below minimum, update it
        if self.n_spin.value() < min_frames:
            self.n_spin.setValue(min_frames)

    def _compute(self):
        eligible = self._build_eligible_indices()
        if not eligible:
            QMessageBox.information(
                self, "No frames", "No eligible frames in this scope."
            )
            return

        cache_dir = self.project.out_root / ".posekit" / "embeddings"

        self.btn_compute.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_preview.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.preview.setPlainText("")
        self.lbl_status.setText("Starting…")
        self.progress.setValue(0.0)

        self._thread = QThread(self)
        self._worker = EmbeddingWorker(
            image_paths=self.image_paths,
            eligible_indices=eligible,
            cache_dir=cache_dir,
            model_name=self.model_combo.currentText().strip(),
            device_pref=self.dev_combo.currentText(),
            batch_size=int(self.batch_spin.value()),
            use_enhance=bool(self.cb_use_enhance.isChecked()),
            max_side=int(self.max_side_spin.value()),
            cache_ok=True,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.status.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _cancel_compute(self):
        if self._worker:
            self._worker.cancel()

    def _on_progress(self, done: int, total: int):
        self.progress.setValue(done / max(1, total))

    def _on_status(self, s: str):
        self.lbl_status.setText(s)

    def _on_failed(self, msg: str):
        self._cleanup_thread()
        QMessageBox.critical(self, "Embedding failed", msg)
        self.btn_compute.setEnabled(True)
        self.btn_cancel.setEnabled(False)

    def _on_finished(self, emb, eligible_indices, meta):
        self._cleanup_thread()
        self._emb = emb
        self._eligible_indices = eligible_indices
        self.lbl_status.setText(f"Embeddings ready: {emb.shape[0]} × {emb.shape[1]}")
        self.btn_compute.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_preview.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_explorer.setEnabled(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _preview(self):
        if self._emb is None or self._eligible_indices is None:
            return
        n = int(self.n_spin.value())
        k = int(self.k_spin.value())
        min_per = int(self.min_per_spin.value())
        strategy = self.strategy_combo.currentText().strip()
        cluster_method = self.cluster_method_combo.currentText().strip()

        # Check if we should warn about large dataset with hierarchical clustering
        n_frames = len(self._eligible_indices)
        if cluster_method == "hierarchical" and n_frames > 2500:
            reply = QMessageBox.warning(
                self,
                "Large Dataset Warning",
                f"You have {n_frames} frames selected.\n\n"
                "Hierarchical clustering may be slow or memory-intensive for large datasets.\n\n"
                "Would you like to switch to 'linkage' method instead?\n"
                "(You can also change this manually in the dropdown)",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.cluster_method_combo.setCurrentText("linkage")
                cluster_method = "linkage"
            elif reply == QMessageBox.Cancel:
                return

        # Compute clusters
        cluster = cluster_embeddings_cosine(
            self._emb, k=k, method=cluster_method, seed=0
        )
        self._cluster = cluster
        self._autosave_clusters()
        self.btn_explorer.setEnabled(True)
        self.lbl_status.setText("Clusters saved. Use Export → Split method to apply.")

        # Use existing pick_frames_stratified with strategy
        picked = pick_frames_stratified(
            emb=self._emb,
            cluster_id=cluster,
            want_n=n,
            eligible_indices=self._eligible_indices,
            min_per_cluster=min_per,
            seed=0,
            strategy=strategy,
        )

        # Apply duplicate filtering if requested
        if self.cb_filter_duplicates.isChecked():
            try:
                try:
                    from .pose_label_extensions import filter_near_duplicates
                except ImportError:
                    from pose_label_extensions import filter_near_duplicates

                threshold = self.dup_threshold_spin.value()
                embeddings_picked = self._emb[
                    [self._eligible_indices.index(i) for i in picked]
                ]
                picked = filter_near_duplicates(
                    embeddings_picked,
                    list(range(len(picked))),
                    threshold=threshold,
                )
                # Map back to original indices
                picked = [picked[i] for i in picked]
            except Exception as e:
                logger.warning(f"Duplicate filtering failed: {e}")

        # Prefer unlabeled if requested
        if self.cb_prefer_unlabeled.isChecked() and self.is_labeled_fn:
            labeled = [i for i in picked if self.is_labeled_fn(self.image_paths[i])]
            unlabeled = [
                i for i in picked if not self.is_labeled_fn(self.image_paths[i])
            ]
            # Prioritize unlabeled but keep labeled if needed
            picked = unlabeled + labeled[: max(0, n - len(unlabeled))]

        self.selected_indices = picked[:n]

        lines = []
        for idx in self.selected_indices[:300]:  # avoid huge previews
            # find local pos to show cluster
            local = self._eligible_indices.index(idx)
            cid = int(cluster[local])
            labeled_str = (
                " [L]"
                if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[idx])
                else ""
            )
            lines.append(f"[c{cid:03d}]{labeled_str} {self.image_paths[idx].name}")
        if len(self.selected_indices) > 300:
            lines.append(f"... ({len(self.selected_indices)-300} more)")
        self.preview.setPlainText("\n".join(lines))

    def _open_explorer(self):
        """Open the embedding explorer dialog."""
        if self._emb is None:
            QMessageBox.warning(
                self,
                "No Embeddings",
                "Please generate embeddings first by clicking 'Compute / Load cached embeddings'.",
            )
            return

        # Launch the explorer dialog (clusters are optional, will compute if needed)
        dialog = EmbeddingExplorerDialog(
            embeddings=self._emb,
            image_paths=[self.image_paths[i] for i in self._eligible_indices],
            cluster_ids=self._cluster,  # May be None, dialog will handle it
            is_labeled_fn=self.is_labeled_fn if self.is_labeled_fn else lambda p: False,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            # Get selected indices from explorer
            explorer_selected = dialog.selected_indices
            if explorer_selected:
                # Map back to global indices
                global_selected = [self._eligible_indices[i] for i in explorer_selected]

                # Merge with existing selections (avoid duplicates)
                existing = set(self.selected_indices)
                for idx in global_selected:
                    if idx not in existing:
                        self.selected_indices.append(idx)
                        existing.add(idx)

                # Update preview to show added frames
                lines = []
                for idx in self.selected_indices[:300]:
                    local = self._eligible_indices.index(idx)
                    cid = int(self._cluster[local])
                    labeled_str = (
                        " [L]"
                        if self.is_labeled_fn
                        and self.is_labeled_fn(self.image_paths[idx])
                        else ""
                    )
                    lines.append(
                        f"[c{cid:03d}]{labeled_str} {self.image_paths[idx].name}"
                    )
                if len(self.selected_indices) > 300:
                    lines.append(f"... ({len(self.selected_indices)-300} more)")
                self.preview.setPlainText("\n".join(lines))

                QMessageBox.information(
                    self,
                    "Added from Explorer",
                    f"Added {len(explorer_selected)} frames from embedding explorer.\n"
                    f"Total selected: {len(self.selected_indices)}",
                )

    def _save_csv(self):
        if self._cluster is None or self._eligible_indices is None:
            QMessageBox.information(self, "No clusters", "Run Clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cluster assignments",
            str(
                (self.project.out_root / ".posekit" / "clusters").resolve()
                / "clusters.csv"
            ),
            "CSV (*.csv)",
        )
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # write: image_path, cluster_id
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cluster_id"])
            for local_pos, idx in enumerate(self._eligible_indices):
                w.writerow([str(self.image_paths[idx]), int(self._cluster[local_pos])])

        QMessageBox.information(self, "Saved", f"Wrote cluster CSV:\n{out}")


    def _autosave_clusters(self):
        if self._cluster is None or self._eligible_indices is None:
            return
        out = (
            self.project.out_root / ".posekit" / "clusters" / "clusters.csv"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "cluster_id"])
                for local_pos, idx in enumerate(self._eligible_indices):
                    w.writerow([str(self.image_paths[idx]), int(self._cluster[local_pos])])
            self.lbl_status.setText(f"Clusters saved: {out}")
            win = self.window()
            if hasattr(win, "statusBar"):
                win.statusBar().showMessage(
                    f"Clusters saved → {out}", 4000
                )
        except Exception:
            # Silent fail; user can still export manually.
            pass


# -----------------------------
# Interactive Embedding Explorer
# -----------------------------


class EmbeddingExplorerDialog(QDialog):
    """Interactive UMAP visualization with Bokeh - image hover support."""

    def __init__(
        self,
        embeddings: np.ndarray,
        image_paths: List[Path],
        cluster_ids: Optional[np.ndarray] = None,
        is_labeled_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Embedding Explorer (UMAP + Bokeh)")
        self.setMinimumSize(QSize(1000, 700))

        self.embeddings = embeddings
        self.image_paths = image_paths
        self.cluster_ids = cluster_ids
        self.is_labeled_fn = is_labeled_fn

        self.umap_projection = None
        self.difficulty_scores = None
        self.selected_indices: List[int] = []
        self.bokeh_html_path = None

        # UMAP worker thread
        self._umap_thread = None
        self._umap_worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Top controls
        controls = QHBoxLayout()

        controls.addWidget(QLabel("UMAP neighbors:"))
        self.neighbors_spin = QSpinBox()
        self.neighbors_spin.setRange(5, 50)
        self.neighbors_spin.setValue(15)
        controls.addWidget(self.neighbors_spin)

        self.btn_compute_umap = QPushButton("Compute UMAP")
        self.btn_compute_umap.clicked.connect(self._compute_umap)
        controls.addWidget(self.btn_compute_umap)

        self.btn_cancel_umap = QPushButton("Cancel")
        self.btn_cancel_umap.setEnabled(False)
        self.btn_cancel_umap.clicked.connect(self._cancel_umap)
        controls.addWidget(self.btn_cancel_umap)

        controls.addStretch()
        layout.addLayout(controls)

        # Progress bar for UMAP
        self.umap_progress = QLabel(
            "UMAP not computed yet. Click 'Compute UMAP' to visualize."
        )
        layout.addWidget(self.umap_progress)

        # Selection info
        info_layout = QHBoxLayout()

        info_layout.addWidget(QLabel("Tip: Use Box Select tool to select frames"))

        self.btn_refresh = QPushButton("Refresh Visualization")
        self.btn_refresh.clicked.connect(self._refresh_viz)
        self.btn_refresh.setEnabled(False)
        info_layout.addWidget(self.btn_refresh)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        info_layout.addWidget(btn_close)

        layout.addLayout(info_layout)

        # Visualization opens in the system browser.

    def _compute_umap(self):
        """Compute UMAP in background thread."""
        self.btn_compute_umap.setEnabled(False)
        self.btn_cancel_umap.setEnabled(True)
        self.umap_progress.setText("Computing UMAP projection...")

        # Create worker thread
        self._umap_thread = QThread(self)
        self._umap_worker = UMAPWorker(
            embeddings=self.embeddings,
            n_neighbors=self.neighbors_spin.value(),
            random_state=42,
        )
        self._umap_worker.moveToThread(self._umap_thread)
        self._umap_thread.started.connect(self._umap_worker.run)
        self._umap_worker.progress.connect(self._on_umap_progress)
        self._umap_worker.finished.connect(self._on_umap_finished)
        self._umap_worker.failed.connect(self._on_umap_failed)
        self._umap_thread.start()

    def _cancel_umap(self):
        """Cancel UMAP computation."""
        if self._umap_worker:
            self._umap_worker.cancel()

    def _on_umap_progress(self, msg: str):
        """Update progress message."""
        self.umap_progress.setText(msg)

    def _on_umap_finished(self, projection: np.ndarray):
        """Handle UMAP completion."""
        self._cleanup_umap_thread()
        self.umap_projection = projection
        self.umap_progress.setText(
            f"UMAP complete! Generated {projection.shape[0]} 2D points."
        )
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        self.btn_refresh.setEnabled(True)

        # Generate and open Bokeh visualization
        self._generate_bokeh_viz()

    def _on_umap_failed(self, error_msg: str):
        """Handle UMAP failure."""
        self._cleanup_umap_thread()
        self.umap_progress.setText(f"UMAP failed: {error_msg}")
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        QMessageBox.critical(
            self, "UMAP Error", f"UMAP computation failed:\\n{error_msg}"
        )

    def _cleanup_umap_thread(self):
        """Clean up UMAP thread."""
        if self._umap_thread:
            self._umap_thread.quit()
            self._umap_thread.wait(2000)
        self._umap_thread = None
        self._umap_worker = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass

    def _generate_bokeh_viz(self):
        """Generate Bokeh visualization with image hover."""
        if self.umap_projection is None:
            return

        try:
            from bokeh.plotting import figure, output_file, save
            from bokeh.models import HoverTool, ColumnDataSource, BoxSelectTool, TapTool
            from bokeh.palettes import Category20_20, Turbo256
            from PIL import Image
            import base64
            from io import BytesIO
            import tempfile

            self.umap_progress.setText(
                "Generating Bokeh visualization (encoding images)..."
            )

            # Convert images to base64
            def img_to_base64(path, max_size=150):
                try:
                    img = Image.open(path)
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    return base64.b64encode(buffered.getvalue()).decode()
                except Exception:
                    return ""

            img_b64 = [img_to_base64(p) for p in self.image_paths]
            image_names = [p.name for p in self.image_paths]

            # Generate colors based on clusters or labeled status
            if self.cluster_ids is not None:
                n_clusters = len(set(self.cluster_ids)) - (
                    1 if -1 in self.cluster_ids else 0
                )
                if n_clusters <= 20:
                    palette = Category20_20
                else:
                    palette = Turbo256

                colors = []
                for cid in self.cluster_ids:
                    if cid == -1:
                        colors.append("#808080")  # Gray for noise
                    else:
                        colors.append(palette[int(cid) % len(palette)])

                cluster_labels = [
                    f"Cluster {int(c)}" if c != -1 else "Noise"
                    for c in self.cluster_ids
                ]
            else:
                # Color by labeled/unlabeled
                if self.is_labeled_fn:
                    colors = [
                        "#4CAF50" if self.is_labeled_fn(p) else "#FF9800"
                        for p in self.image_paths
                    ]
                    cluster_labels = [
                        "Labeled" if self.is_labeled_fn(p) else "Unlabeled"
                        for p in self.image_paths
                    ]
                else:
                    colors = ["#2196F3"] * len(self.image_paths)
                    cluster_labels = ["Frame"] * len(self.image_paths)

            # Create ColumnDataSource
            source = ColumnDataSource(
                data=dict(
                    x=self.umap_projection[:, 0],
                    y=self.umap_projection[:, 1],
                    names=image_names,
                    imgs=["data:image/png;base64," + b64 for b64 in img_b64],
                    clusters=cluster_labels,
                    colors=colors,
                    indices=list(range(len(self.image_paths))),
                )
            )

            # Create figure
            p = figure(
                width=900,
                height=700,
                title=f"UMAP Embedding Space ({len(self.image_paths)} frames)",
                tools="pan,wheel_zoom,box_zoom,box_select,tap,reset,save",
                active_drag="box_select",
            )

            # Add scatter plot
            p.circle(
                "x",
                "y",
                size=8,
                alpha=0.7,
                color="colors",
                line_color="black",
                line_width=0.5,
                source=source,
                selection_color="red",
                selection_alpha=1.0,
                nonselection_alpha=0.3,
            )

            # Add hover tool with image
            hover = HoverTool(
                tooltips="""
                <div>
                    <div>
                        <img src="@imgs" style="width:150px; height:auto;"></img>
                    </div>
                    <div>
                        <span style="font-size: 12px; font-weight: bold;">@names</span><br>
                        <span style="font-size: 11px; color: #666;">@clusters</span><br>
                        <span style="font-size: 10px; color: #999;">Index: @indices</span>
                    </div>
                </div>
            """
            )
            p.add_tools(hover)

            p.xaxis.axis_label = "UMAP 1"
            p.yaxis.axis_label = "UMAP 2"

            # Save to temp HTML file
            if self.bokeh_html_path is None:
                fd, self.bokeh_html_path = tempfile.mkstemp(
                    suffix=".html", prefix="embedding_explorer_"
                )
                import os

                os.close(fd)

            output_file(self.bokeh_html_path, title="Embedding Explorer")
            save(p)

            self.umap_progress.setText(f"✓ Visualization ready! HTML: {self.bokeh_html_path}")

            # Load in web view or open in browser
            self._load_visualization()

        except ImportError as e:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                f"Bokeh is not installed. Install with:\\n\\npip install bokeh\\n\\nError: {e}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Visualization Error",
                f"Failed to generate Bokeh visualization:\\n{str(e)}",
            )

    def _load_visualization(self):
        """Open Bokeh HTML in the system browser."""
        if not self.bokeh_html_path:
            return

        import webbrowser

        webbrowser.open("file://" + self.bokeh_html_path)

    def _refresh_viz(self):
        """Refresh the visualization."""
        self._load_visualization()

    def closeEvent(self, event):
        self.embeddings = None
        self.umap_projection = None
        self.cluster_ids = None
        try:
            gc.collect()
        except Exception:
            pass
        super().closeEvent(event)


class UMAPWorker(QObject):
    """Background worker for UMAP computation."""

    progress = Signal(str)
    finished = Signal(np.ndarray)
    failed = Signal(str)

    def __init__(self, embeddings: np.ndarray, n_neighbors: int, random_state: int):
        super().__init__()
        self.embeddings = embeddings
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            if self._cancelled:
                return

            self.progress.emit("Importing UMAP...")
            import umap

            if self._cancelled:
                return

            self.progress.emit(
                f"Running UMAP (n_neighbors={self.n_neighbors}, this may take a minute)..."
            )

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=self.random_state,
                verbose=False,
            )

            if self._cancelled:
                return

            projection = reducer.fit_transform(self.embeddings)

            if self._cancelled:
                return

            self.finished.emit(projection.astype(np.float32))

        except ImportError:
            self.failed.emit("UMAP not installed. Install with: pip install umap-learn")
        except Exception as e:
            self.failed.emit(str(e))


# -----------------------------
# Evaluation Dashboard
# -----------------------------


def _read_image_size(img_path: Path) -> Optional[Tuple[int, int]]:
    try:
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


def _load_kpts_px(
    label_path: Path, k: int, img_path: Path
) -> Optional[List[Tuple[float, float, float]]]:
    size = _read_image_size(img_path)
    if size is None:
        return None
    w, h = size
    parsed = load_yolo_pose_label(label_path, k)
    if parsed is None:
        return None
    _, kpts, _ = parsed
    out = []
    for kp in kpts:
        out.append((float(kp.x) * w, float(kp.y) * h, float(kp.v)))
    return out


def _confidence_from_v(v: float) -> float:
    if v <= 1.0:
        return max(0.0, min(1.0, float(v)))
    if v >= 2.0:
        return 1.0
    return 0.5


def evaluate_pose_predictions(
    image_paths: List[Path],
    gt_labels_dir: Path,
    pred_labels_dir: Path,
    k: int,
    pck_thresh_frac: float,
    oks_sigma: float,
) -> Dict[str, object]:
    per_kpt_errors: List[List[float]] = [[] for _ in range(k)]
    per_kpt_oks: List[List[float]] = [[] for _ in range(k)]
    per_kpt_pck: List[List[int]] = [[] for _ in range(k)]

    frame_errors: List[Tuple[int, float]] = []
    all_errors: List[float] = []

    for idx, img_path in enumerate(image_paths):
        gt_path = gt_labels_dir / f"{img_path.stem}.txt"
        pred_path = pred_labels_dir / f"{img_path.stem}.txt"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = _load_kpts_px(gt_path, k, img_path)
        pred = _load_kpts_px(pred_path, k, img_path)
        if gt is None or pred is None:
            continue

        size = _read_image_size(img_path)
        if size is None:
            continue
        w, h = size
        thresh = pck_thresh_frac * max(w, h)
        sigma = oks_sigma * max(w, h)
        sigma = max(1e-6, sigma)

        frame_errs = []
        for j in range(k):
            gx, gy, gv = gt[j]
            px, py, pv = pred[j]
            if gv <= 0 or pv <= 0:
                continue
            d = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            all_errors.append(d)
            per_kpt_errors[j].append(d)
            frame_errs.append(d)

            per_kpt_pck[j].append(1 if d <= thresh else 0)
            oks = math.exp(-((d * d) / (2.0 * sigma * sigma)))
            per_kpt_oks[j].append(oks)

        if frame_errs:
            frame_errors.append((idx, float(np.mean(frame_errs))))

    def _safe_mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    per_kpt_stats = []
    for j in range(k):
        per_kpt_stats.append(
            {
                "mean_error": _safe_mean(per_kpt_errors[j]),
                "pck": _safe_mean(per_kpt_pck[j]),
                "oks": _safe_mean(per_kpt_oks[j]),
                "n": len(per_kpt_errors[j]),
            }
        )

    overall = {
        "mean_error": _safe_mean(all_errors),
        "pck": _safe_mean([x for sub in per_kpt_pck for x in sub]),
        "oks": _safe_mean([x for sub in per_kpt_oks for x in sub]),
        "n": len(all_errors),
    }

    return {
        "overall": overall,
        "per_kpt": per_kpt_stats,
        "frame_errors": frame_errors,
        "all_errors": all_errors,
    }


def _render_histogram(data: List[float], bins: int = 20, w: int = 360, h: int = 120):
    if not data:
        pm = QPixmap(w, h)
        pm.fill(QColor(240, 240, 240))
        return pm
    hist, edges = np.histogram(np.array(data), bins=bins)
    maxv = max(1, int(hist.max()))
    pm = QPixmap(w, h)
    pm.fill(QColor(255, 255, 255))
    painter = QPainter(pm)
    painter.setPen(QColor(60, 60, 60))
    bar_w = w / float(bins)
    for i, v in enumerate(hist):
        bh = int((v / maxv) * (h - 10))
        x = int(i * bar_w)
        painter.fillRect(x, h - bh - 5, int(bar_w) - 2, bh, QColor(100, 160, 220))
    painter.end()
    return pm


class EvaluationDashboardDialog(QDialog):
    """Evaluate predictions against labeled data and show summaries."""

    add_indices = Signal(list)

    def __init__(self, parent, project, image_paths: List[Path]):
        super().__init__(parent)
        self.setWindowTitle("Evaluation Dashboard")
        self.setMinimumSize(QSize(920, 640))
        self.project = project
        self.image_paths = image_paths
        self._last_results = None

        layout = QVBoxLayout(self)

        # Paths
        path_group = QGroupBox("Paths")
        path_layout = QFormLayout(path_group)
        self.gt_dir_edit = QLineEdit(str(self.project.labels_dir))
        self.pred_dir_edit = QLineEdit("")
        self.btn_pick_pred = QPushButton("Choose Predictions Dir…")
        pred_row = QHBoxLayout()
        pred_row.addWidget(self.pred_dir_edit, 1)
        pred_row.addWidget(self.btn_pick_pred)
        path_layout.addRow("GT labels:", self.gt_dir_edit)
        path_layout.addRow("Pred labels:", pred_row)
        layout.addWidget(path_group)

        # Metrics config
        cfg_group = QGroupBox("Metrics")
        cfg_layout = QFormLayout(cfg_group)
        self.pck_spin = QDoubleSpinBox()
        self.pck_spin.setRange(0.005, 0.5)
        self.pck_spin.setSingleStep(0.005)
        self.pck_spin.setValue(0.05)
        cfg_layout.addRow("PCK threshold (frac of max side):", self.pck_spin)
        self.oks_spin = QDoubleSpinBox()
        self.oks_spin.setRange(0.005, 0.5)
        self.oks_spin.setSingleStep(0.005)
        self.oks_spin.setValue(0.05)
        cfg_layout.addRow("OKS sigma (frac of max side):", self.oks_spin)
        layout.addWidget(cfg_group)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_eval = QPushButton("Run Evaluation")
        self.btn_add_worst = QPushButton("Add Worst Frames to Labeling Set")
        self.btn_add_worst.setEnabled(False)
        btn_row.addWidget(self.btn_eval)
        btn_row.addWidget(self.btn_add_worst)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Output layout
        out_row = QHBoxLayout()
        self.tbl_metrics = QTableWidget(0, 4)
        self.tbl_metrics.setHorizontalHeaderLabels(
            ["Keypoint", "Mean Error", "PCK", "OKS"]
        )
        out_row.addWidget(self.tbl_metrics, 2)

        right_col = QVBoxLayout()
        self.lbl_overall = QLabel("Overall: n=0")
        right_col.addWidget(self.lbl_overall)
        self.lbl_hist = QLabel()
        right_col.addWidget(self.lbl_hist)
        right_col.addWidget(QLabel("Worst frames"))
        self.list_worst = QListWidget()
        right_col.addWidget(self.list_worst, 1)
        out_row.addLayout(right_col, 1)
        layout.addLayout(out_row, 1)

        self.btn_pick_pred.clicked.connect(self._pick_pred_dir)
        self.btn_eval.clicked.connect(self._run_eval)
        self.btn_add_worst.clicked.connect(self._add_worst)

    def _pick_pred_dir(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select predictions directory", self.project.out_root.as_posix()
        )
        if d:
            self.pred_dir_edit.setText(d)

    def _run_eval(self):
        pred_dir = Path(self.pred_dir_edit.text().strip())
        gt_dir = Path(self.gt_dir_edit.text().strip())
        if not pred_dir.exists():
            QMessageBox.warning(self, "Missing", "Predictions directory not found.")
            return
        if not gt_dir.exists():
            QMessageBox.warning(
                self, "Missing", "Ground-truth labels directory not found."
            )
            return

        res = evaluate_pose_predictions(
            self.image_paths,
            gt_dir,
            pred_dir,
            len(self.project.keypoint_names),
            self.pck_spin.value(),
            self.oks_spin.value(),
        )
        self._last_results = res
        self._update_ui()

    def _update_ui(self):
        if not self._last_results:
            return
        res = self._last_results
        overall = res["overall"]
        self.lbl_overall.setText(
            f"Overall: n={overall['n']}  mean_error={overall['mean_error']:.2f}  "
            f"PCK={overall['pck']:.3f}  OKS={overall['oks']:.3f}"
        )

        self.tbl_metrics.setRowCount(len(self.project.keypoint_names))
        for i, name in enumerate(self.project.keypoint_names):
            stats = res["per_kpt"][i]
            self.tbl_metrics.setItem(i, 0, QTableWidgetItem(name))
            self.tbl_metrics.setItem(
                i, 1, QTableWidgetItem(f"{stats['mean_error']:.2f}")
            )
            self.tbl_metrics.setItem(i, 2, QTableWidgetItem(f"{stats['pck']:.3f}"))
            self.tbl_metrics.setItem(i, 3, QTableWidgetItem(f"{stats['oks']:.3f}"))

        hist_pm = _render_histogram(res["all_errors"])
        self.lbl_hist.setPixmap(hist_pm)

        self.list_worst.clear()
        worst = sorted(res["frame_errors"], key=lambda x: x[1], reverse=True)[:50]
        for idx, err in worst:
            item = QListWidgetItem(f"{self.image_paths[idx].name}  err={err:.2f}")
            item.setData(Qt.UserRole, idx)
            self.list_worst.addItem(item)
        self.btn_add_worst.setEnabled(bool(worst))

    def _add_worst(self):
        if not self._last_results:
            return
        selected = []
        for i in range(self.list_worst.count()):
            item = self.list_worst.item(i)
            selected.append(int(item.data(Qt.UserRole)))
        if not selected:
            return
        self.add_indices.emit(selected)
        QMessageBox.information(
            self, "Added", f"Added {len(selected)} frames to labeling set."
        )


# -----------------------------
# Active Learning Sampler
# -----------------------------


def _score_low_confidence(
    image_paths: List[Path],
    pred_dir: Path,
    k: int,
) -> Dict[int, float]:
    scores = {}
    for idx, img_path in enumerate(image_paths):
        pred_path = pred_dir / f"{img_path.stem}.txt"
        if not pred_path.exists():
            continue
        pred = _load_kpts_px(pred_path, k, img_path)
        if pred is None:
            continue
        confs = []
        for _, _, v in pred:
            confs.append(_confidence_from_v(v))
        if confs:
            scores[idx] = float(1.0 - np.mean(confs))
    return scores


def _score_disagreement(
    image_paths: List[Path],
    pred_a: Path,
    pred_b: Path,
    k: int,
) -> Dict[int, float]:
    scores = {}
    for idx, img_path in enumerate(image_paths):
        pa = pred_a / f"{img_path.stem}.txt"
        pb = pred_b / f"{img_path.stem}.txt"
        if not pa.exists() or not pb.exists():
            continue
        a = _load_kpts_px(pa, k, img_path)
        b = _load_kpts_px(pb, k, img_path)
        if a is None or b is None:
            continue
        errs = []
        for j in range(k):
            ax, ay, av = a[j]
            bx, by, bv = b[j]
            if av <= 0 or bv <= 0:
                continue
            errs.append(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2))
        if errs:
            scores[idx] = float(np.mean(errs))
    return scores


def _score_error_vs_gt(
    image_paths: List[Path],
    gt_dir: Path,
    pred_dir: Path,
    k: int,
) -> Dict[int, float]:
    scores = {}
    for idx, img_path in enumerate(image_paths):
        gt_path = gt_dir / f"{img_path.stem}.txt"
        pred_path = pred_dir / f"{img_path.stem}.txt"
        if not gt_path.exists() or not pred_path.exists():
            continue
        gt = _load_kpts_px(gt_path, k, img_path)
        pred = _load_kpts_px(pred_path, k, img_path)
        if gt is None or pred is None:
            continue
        errs = []
        for j in range(k):
            gx, gy, gv = gt[j]
            px, py, pv = pred[j]
            if gv <= 0 or pv <= 0:
                continue
            errs.append(math.sqrt((gx - px) ** 2 + (gy - py) ** 2))
        if errs:
            scores[idx] = float(np.mean(errs))
    return scores


class ActiveLearningSamplerDialog(QDialog):
    """Suggest frames to label based on model outputs."""

    add_indices = Signal(list)

    def __init__(
        self, parent, project, image_paths: List[Path], is_labeled_fn, labeling_set_fn
    ):
        super().__init__(parent)
        self.setWindowTitle("Active Learning Sampler")
        self.setMinimumSize(QSize(900, 600))
        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn
        self.labeling_set_fn = labeling_set_fn

        layout = QVBoxLayout(self)

        # Scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Scope:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Unlabeled only", "All frames", "Labeling set"])
        scope_row.addWidget(self.scope_combo, 1)
        layout.addLayout(scope_row)

        # Strategy
        strat_group = QGroupBox("Strategy")
        strat_layout = QFormLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            [
                "Lowest confidence",
                "Disagreement between two models",
                "Largest error vs GT",
            ]
        )
        strat_layout.addRow("Strategy:", self.strategy_combo)
        layout.addWidget(strat_group)

        # Paths
        path_group = QGroupBox("Prediction Paths")
        path_layout = QFormLayout(path_group)
        self.pred_a_edit = QLineEdit("")
        self.pred_b_edit = QLineEdit("")
        self.gt_dir_edit = QLineEdit(str(self.project.labels_dir))
        self.btn_pick_a = QPushButton("Choose A…")
        self.btn_pick_b = QPushButton("Choose B…")
        self.btn_pick_gt = QPushButton("Choose GT…")

        row_a = QHBoxLayout()
        row_a.addWidget(self.pred_a_edit, 1)
        row_a.addWidget(self.btn_pick_a)
        path_layout.addRow("Predictions A:", row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(self.pred_b_edit, 1)
        row_b.addWidget(self.btn_pick_b)
        path_layout.addRow("Predictions B:", row_b)

        row_gt = QHBoxLayout()
        row_gt.addWidget(self.gt_dir_edit, 1)
        row_gt.addWidget(self.btn_pick_gt)
        path_layout.addRow("GT labels:", row_gt)
        layout.addWidget(path_group)

        # Count
        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Suggest count:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 5000)
        self.count_spin.setValue(50)
        count_row.addWidget(self.count_spin)
        count_row.addStretch(1)
        layout.addLayout(count_row)

        # Results
        self.list_suggest = QListWidget()
        layout.addWidget(self.list_suggest, 1)

        # Buttons
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Suggest Frames")
        self.btn_add = QPushButton("Add to Labeling Set")
        self.btn_add.setEnabled(False)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_add)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.btn_pick_a.clicked.connect(lambda: self._pick_dir(self.pred_a_edit))
        self.btn_pick_b.clicked.connect(lambda: self._pick_dir(self.pred_b_edit))
        self.btn_pick_gt.clicked.connect(lambda: self._pick_dir(self.gt_dir_edit))
        self.btn_run.clicked.connect(self._run)
        self.btn_add.clicked.connect(self._add)

    def _pick_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(
            self, "Select directory", self.project.out_root.as_posix()
        )
        if d:
            edit.setText(d)

    def _eligible_indices(self) -> List[int]:
        scope = self.scope_combo.currentText()
        if scope == "All frames":
            return list(range(len(self.image_paths)))
        if scope == "Labeling set":
            return list(self.labeling_set_fn())
        return [
            i
            for i in range(len(self.image_paths))
            if not self.is_labeled_fn(self.image_paths[i])
        ]

    def _run(self):
        strat = self.strategy_combo.currentText()
        pred_a = (
            Path(self.pred_a_edit.text().strip())
            if self.pred_a_edit.text().strip()
            else None
        )
        pred_b = (
            Path(self.pred_b_edit.text().strip())
            if self.pred_b_edit.text().strip()
            else None
        )
        gt_dir = (
            Path(self.gt_dir_edit.text().strip())
            if self.gt_dir_edit.text().strip()
            else None
        )

        k = len(self.project.keypoint_names)
        scores = {}

        if strat == "Lowest confidence":
            if not pred_a or not pred_a.exists():
                QMessageBox.warning(
                    self, "Missing", "Predictions A directory is required."
                )
                return
            scores = _score_low_confidence(self.image_paths, pred_a, k)
        elif strat == "Disagreement between two models":
            if not pred_a or not pred_a.exists() or not pred_b or not pred_b.exists():
                QMessageBox.warning(
                    self, "Missing", "Predictions A and B directories are required."
                )
                return
            scores = _score_disagreement(self.image_paths, pred_a, pred_b, k)
        else:
            if not pred_a or not pred_a.exists() or not gt_dir or not gt_dir.exists():
                QMessageBox.warning(
                    self,
                    "Missing",
                    "Predictions A and GT labels directories are required.",
                )
                return
            scores = _score_error_vs_gt(self.image_paths, gt_dir, pred_a, k)

        eligible = set(self._eligible_indices())
        filtered = [(idx, s) for idx, s in scores.items() if idx in eligible]
        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[: self.count_spin.value()]

        self.list_suggest.clear()
        for idx, s in filtered:
            item = QListWidgetItem(f"{self.image_paths[idx].name}  score={s:.3f}")
            item.setData(Qt.UserRole, idx)
            self.list_suggest.addItem(item)
        self.btn_add.setEnabled(bool(filtered))

    def _add(self):
        picked = []
        for i in range(self.list_suggest.count()):
            item = self.list_suggest.item(i)
            picked.append(int(item.data(Qt.UserRole)))
        if not picked:
            return
        self.add_indices.emit(picked)
        QMessageBox.information(
            self, "Added", f"Added {len(picked)} frames to labeling set."
        )


# -----------------------------
# Training / Evaluation / Active Learning
# -----------------------------


class _SignalLogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self._signal = signal

    def emit(self, record):
        try:
            msg = self.format(record)
            self._signal.emit(msg)
        except Exception:
            pass


def _make_histogram_image(values, bins: int = 20, width: int = 360, height: int = 120):
    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(248, 248, 248))
    if not values:
        return img

    hist, _ = np.histogram(values, bins=bins)
    max_count = int(hist.max()) if hist.size else 1
    if max_count <= 0:
        return img

    painter = QPainter(img)
    bar_w = max(1, width // bins)
    for i, count in enumerate(hist):
        x = i * bar_w
        h = int((count / max_count) * (height - 8))
        painter.fillRect(x, height - h, bar_w - 1, h, QColor(60, 120, 200))
    painter.end()
    return img


def _make_heatmap_image(matrix: np.ndarray, width: int = 360, height: int = 140):
    img = QImage(width, height, QImage.Format_ARGB32)
    img.fill(QColor(248, 248, 248))
    if matrix.size == 0:
        return img

    rows, cols = matrix.shape
    max_val = float(matrix.max()) if matrix.size else 1.0
    if max_val <= 0:
        max_val = 1.0

    cell_w = max(1, width // cols)
    cell_h = max(1, height // rows)
    painter = QPainter(img)
    for r in range(rows):
        for c in range(cols):
            v = float(matrix[r, c]) / max_val
            color = QColor(255 - int(180 * v), 80 + int(120 * v), 80 + int(120 * v))
            painter.fillRect(c * cell_w, r * cell_h, cell_w, cell_h, color)
    painter.end()
    return img


def _format_float(val, digits: int = 3):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return "n/a"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return "n/a"


class TrainingWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        model_weights: str,
        dataset_yaml: Path,
        run_dir: Path,
        epochs: int,
        patience: int,
        batch: int,
        imgsz: int,
        device: str,
        augment: bool,
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        degrees: float,
        translate: float,
        scale: float,
        fliplr: float,
        mosaic: float,
        mixup: float,
    ):
        super().__init__()
        self.model_weights = model_weights
        self.dataset_yaml = Path(dataset_yaml)
        self.run_dir = Path(run_dir)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch = int(batch)
        self.imgsz = int(imgsz)
        self.device = device
        self.augment = bool(augment)
        self.hsv_h = float(hsv_h)
        self.hsv_s = float(hsv_s)
        self.hsv_v = float(hsv_v)
        self.degrees = float(degrees)
        self.translate = float(translate)
        self.scale = float(scale)
        self.mosaic = float(mosaic)
        self.mixup = float(mixup)
        self.fliplr = float(fliplr)
        self._cancel = False
        self._log_handler = None
        self._model = None
        self._proc = None

    def cancel(self):
        self._cancel = True
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass

    def _on_epoch_end(self, trainer):
        try:
            self.progress.emit(int(trainer.epoch) + 1, int(trainer.epochs))
        except Exception:
            pass

    def _on_batch_end(self, trainer):
        if self._cancel:
            try:
                trainer.stop = True
            except Exception:
                pass

    def _attach_logger(self):
        try:
            self._log_handler = _SignalLogHandler(self.log)
            self._log_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            ul_logger = logging.getLogger("ultralytics")
            ul_logger.addHandler(self._log_handler)
            ul_logger.setLevel(logging.INFO)
        except Exception:
            self._log_handler = None

    def _detach_logger(self):
        if not self._log_handler:
            return
        try:
            ul_logger = logging.getLogger("ultralytics")
            ul_logger.removeHandler(self._log_handler)
        except Exception:
            pass
        self._log_handler = None

    def run(self):
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Training run dir: {self.run_dir}")
            self.log.emit("Starting training (subprocess)…")
            self.progress.emit(0, max(1, int(self.epochs)))

            cli_args = [
                "task=pose",
                "mode=train",
                f"model={self.model_weights}",
                f"data={str(self.dataset_yaml)}",
                f"epochs={self.epochs}",
                f"patience={int(getattr(self, 'patience', 10))}",
                f"batch={self.batch}",
                f"imgsz={self.imgsz}",
                f"project={str(self.run_dir.parent)}",
                f"name={self.run_dir.name}",
                "exist_ok=True",
            ]
            if self.device and self.device != "auto":
                cli_args.append(f"device={self.device}")
            if self.augment:
                cli_args += [
                    "augment=True",
                    f"hsv_h={self.hsv_h}",
                    f"hsv_s={self.hsv_s}",
                    f"hsv_v={self.hsv_v}",
                    f"degrees={self.degrees}",
                    f"translate={self.translate}",
                    f"scale={self.scale}",
                    f"fliplr={self.fliplr}",
                    f"mosaic={self.mosaic}",
                    f"mixup={self.mixup}",
                ]

            def _run_cmd(cmd):
                return subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

            yolo_bin = shutil.which("yolo")
            if yolo_bin:
                self._proc = _run_cmd([yolo_bin] + cli_args)
            else:
                # Fallback: run via python -c using ultralytics.YOLO API
                py_code = (
                    "from ultralytics import YOLO\n"
                    "model=YOLO(r'''{model}''')\n"
                    "model.train(\n"
                    "  data=r'''{data}''',\n"
                    "  epochs={epochs},\n"
                    "  patience={patience},\n"
                    "  batch={batch},\n"
                    "  imgsz={imgsz},\n"
                    "  project=r'''{project}''',\n"
                    "  name=r'''{name}''',\n"
                    "  exist_ok=True,\n"
                    "  device=r'''{device}''',\n"
                    "  augment={augment},\n"
                    "  hsv_h={hsv_h},\n"
                    "  hsv_s={hsv_s},\n"
                    "  hsv_v={hsv_v},\n"
                    "  degrees={degrees},\n"
                    "  translate={translate},\n"
                    "  scale={scale},\n"
                    "  fliplr={fliplr},\n"
                    "  mosaic={mosaic},\n"
                    "  mixup={mixup},\n"
                    ")\n"
                ).format(
                    model=self.model_weights,
                    data=str(self.dataset_yaml),
                    epochs=self.epochs,
                    patience=self.patience,
                    batch=self.batch,
                    imgsz=self.imgsz,
                    project=str(self.run_dir.parent),
                    name=self.run_dir.name,
                    device=self.device if self.device else "auto",
                    augment=bool(self.augment),
                    hsv_h=self.hsv_h,
                    hsv_s=self.hsv_s,
                    hsv_v=self.hsv_v,
                    degrees=self.degrees,
                    translate=self.translate,
                    scale=self.scale,
                    fliplr=self.fliplr,
                    mosaic=self.mosaic,
                    mixup=self.mixup,
                )
                self._proc = _run_cmd([sys.executable, "-c", py_code])
            assert self._proc.stdout is not None
            ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
            for line in self._proc.stdout:
                if self._cancel:
                    break
                msg = ansi_re.sub("", line).rstrip()
                self.log.emit(msg)
                # Parse epoch progress from Ultralytics logs (e.g., "Epoch 3/50")
                m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", msg)
                if not m:
                    m = re.search(r"^\s*(\d+)\s*/\s*(\d+)\s", msg)
                if m:
                    try:
                        cur = int(m.group(1))
                        total = int(m.group(2))
                        if total > 0:
                            self.progress.emit(cur, total)
                    except Exception:
                        pass

            rc = self._proc.wait()
            if self._cancel:
                self.failed.emit("Canceled.")
                return
            if rc != 0:
                self.failed.emit(f"Training failed (exit code {rc}).")
                return

            weights_dir = self.run_dir / "weights"
            best = weights_dir / "best.pt"
            last = weights_dir / "last.pt"
            weights_path = best if best.exists() else last

            self.log.emit("Training finished.")
            self.finished.emit(
                {
                    "run_dir": str(self.run_dir),
                    "weights": str(weights_path) if weights_path.exists() else "",
                }
            )
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            self._cleanup_gpu()

    def _cleanup_gpu(self):
        try:
            # Best-effort free of ultralytics/torch GPU memory between runs
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        except Exception:
            pass


class TrainingRunnerDialog(QDialog):
    def __init__(self, parent, project, image_paths: List[Path]):
        super().__init__(parent)
        self.setWindowTitle("Training Runner")
        self.setMinimumSize(QSize(760, 600))

        self.project = project
        self.image_paths = image_paths
        self._thread = None
        self._worker = None
        self._last_run_dir = None
        self._last_weights = None
        self._loss_timer = QTimer(self)
        self._loss_timer.setInterval(1000)
        self._loss_timer.timeout.connect(self._update_loss_plot)
        self._loss_timer = QTimer(self)
        self._loss_timer.setInterval(1000)
        self._loss_timer.timeout.connect(self._update_loss_plot)

        layout = QVBoxLayout(self)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Backend
        backend_group = QGroupBox("Backend")
        backend_layout = QFormLayout(backend_group)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["YOLO Pose", "ViTPose (soon)", "SLEAP (soon)"])
        backend_layout.addRow("Backend:", self.backend_combo)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(get_yolo_pose_base_models())
        self.model_combo.setCurrentText("yolo26n-pose.pt")
        self.btn_model_browse = QPushButton("Browse…")
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.btn_model_browse)
        backend_layout.addRow("Base weights:", model_row)

        content_layout.addWidget(backend_group)

        # Config
        cfg_group = QGroupBox("Config")
        cfg_layout = QFormLayout(cfg_group)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(16)
        cfg_layout.addRow("Batch size:", self.batch_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(50)
        cfg_layout.addRow("Epochs:", self.epochs_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 1000)
        self.patience_spin.setValue(10)
        cfg_layout.addRow("Early stopping (patience):", self.patience_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        cfg_layout.addRow("Image size:", self.imgsz_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        cfg_layout.addRow("Device:", self.device_combo)

        content_layout.addWidget(cfg_group)

        # Augmentations
        aug_group = QGroupBox("Augmentations")
        aug_layout = QVBoxLayout(aug_group)
        self.cb_augment = QCheckBox("Enable augmentations")
        self.cb_augment.setChecked(True)
        aug_layout.addWidget(self.cb_augment)

        self.aug_widgets = []

        grid = QGridLayout()
        aug_layout.addLayout(grid)

        row = 0
        col = 0

        def add_row(label: str, widget: QWidget, tip: str):
            nonlocal row, col
            lbl = QLabel(label)
            lbl.setToolTip(tip)
            widget.setToolTip(tip)
            grid.addWidget(lbl, row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
            if col == 0:
                col = 1
            else:
                col = 0
                row += 1

        self.hsv_h_spin = QDoubleSpinBox()
        self.hsv_h_spin.setRange(0.0, 1.0)
        self.hsv_h_spin.setSingleStep(0.005)
        self.hsv_h_spin.setValue(0.01)
        add_row(
            "hsv_h:",
            self.hsv_h_spin,
            "Hue jitter (fraction). Small values recommended.",
        )
        self.aug_widgets.append(self.hsv_h_spin)

        self.hsv_s_spin = QDoubleSpinBox()
        self.hsv_s_spin.setRange(0.0, 1.0)
        self.hsv_s_spin.setSingleStep(0.05)
        self.hsv_s_spin.setValue(0.2)
        add_row(
            "hsv_s:",
            self.hsv_s_spin,
            "Saturation jitter (fraction).",
        )
        self.aug_widgets.append(self.hsv_s_spin)

        self.hsv_v_spin = QDoubleSpinBox()
        self.hsv_v_spin.setRange(0.0, 1.0)
        self.hsv_v_spin.setSingleStep(0.05)
        self.hsv_v_spin.setValue(0.1)
        add_row(
            "hsv_v:",
            self.hsv_v_spin,
            "Value/brightness jitter (fraction).",
        )
        self.aug_widgets.append(self.hsv_v_spin)

        self.degrees_spin = QDoubleSpinBox()
        self.degrees_spin.setRange(0.0, 180.0)
        self.degrees_spin.setSingleStep(1.0)
        self.degrees_spin.setValue(5.0)
        add_row(
            "degrees:",
            self.degrees_spin,
            "Rotation range in degrees.",
        )
        self.aug_widgets.append(self.degrees_spin)

        self.translate_spin = QDoubleSpinBox()
        self.translate_spin.setRange(0.0, 1.0)
        self.translate_spin.setSingleStep(0.05)
        self.translate_spin.setValue(0.05)
        add_row(
            "translate:",
            self.translate_spin,
            "Translation range as fraction of image size.",
        )
        self.aug_widgets.append(self.translate_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(0.2)
        add_row(
            "scale:",
            self.scale_spin,
            "Scale variation (fraction).",
        )
        self.aug_widgets.append(self.scale_spin)

        self.mosaic_spin = QDoubleSpinBox()
        self.mosaic_spin.setRange(0.0, 1.0)
        self.mosaic_spin.setSingleStep(0.05)
        self.mosaic_spin.setValue(0.1)
        add_row(
            "mosaic:",
            self.mosaic_spin,
            "Mosaic augmentation probability.",
        )
        self.aug_widgets.append(self.mosaic_spin)

        self.mixup_spin = QDoubleSpinBox()
        self.mixup_spin.setRange(0.0, 1.0)
        self.mixup_spin.setSingleStep(0.05)
        self.mixup_spin.setValue(0.0)
        add_row(
            "mixup:",
            self.mixup_spin,
            "MixUp probability.",
        )
        self.aug_widgets.append(self.mixup_spin)

        self.fliplr_spin = QDoubleSpinBox()
        self.fliplr_spin.setRange(0.0, 1.0)
        self.fliplr_spin.setSingleStep(0.05)
        self.fliplr_spin.setValue(0.2)
        add_row(
            "fliplr:",
            self.fliplr_spin,
            "Horizontal flip probability.",
        )
        self.aug_widgets.append(self.fliplr_spin)

        content_layout.addWidget(aug_group)

        # Dataset
        data_group = QGroupBox("Dataset")
        data_layout = QFormLayout(data_group)

        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.05, 0.95)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.8)
        data_layout.addRow("Train fraction:", self.train_split_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        data_layout.addRow("Random seed:", self.seed_spin)

        self._aux_datasets: List[Dict[str, object]] = []
        self._aux_items: List[Tuple[Path, Path]] = []
        self._last_aux_path = ""

        self.lbl_labeled = QLabel("")
        self._refresh_labeled_count()
        data_layout.addRow("Status:", self.lbl_labeled)

        self.aux_list = QListWidget()
        self.btn_add_aux = QPushButton("Add auxiliary project…")
        self.btn_remove_aux = QPushButton("Remove selected")
        aux_row = QHBoxLayout()
        aux_row.addWidget(self.btn_add_aux)
        aux_row.addWidget(self.btn_remove_aux)
        data_layout.addRow("Auxiliary datasets:", self.aux_list)
        data_layout.addRow("", aux_row)

        content_layout.addWidget(data_group)

        # Run info
        info_group = QGroupBox("Run")
        info_layout = QFormLayout(info_group)
        self.lbl_run_dir = QLabel("Run dir: (not started)")
        info_layout.addRow(self.lbl_run_dir)
        content_layout.addWidget(info_group)

        # Logs + progress
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.lbl_loss_plot = QLabel()
        self.lbl_loss_plot.setMinimumHeight(220)
        content_layout.addWidget(self.lbl_loss_plot)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Buttons
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_open_eval = QPushButton("Open Evaluation")
        self.btn_open_eval.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        btns.addWidget(self.btn_open_eval)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.btn_model_browse.clicked.connect(self._browse_model)
        self.btn_start.clicked.connect(self._start_training)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_close.clicked.connect(self.reject)
        self.btn_open_eval.clicked.connect(self._open_eval)
        self.btn_add_aux.clicked.connect(self._add_aux_project)
        self.btn_remove_aux.clicked.connect(self._remove_aux_project)
        self.cb_augment.toggled.connect(self._toggle_aug_widgets)

        self._toggle_aug_widgets(self.cb_augment.isChecked())
        self._apply_settings()
        self._apply_latest_weights_default()

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
        if path:
            self.model_combo.setCurrentText(path)

    def _toggle_aug_widgets(self, enabled: bool):
        for w in self.aug_widgets:
            w.setEnabled(enabled)

    def _add_aux_project(self):
        start_dir = self._last_aux_path or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose_project.json or dataset.yaml",
            start_dir,
            "pose_project.json;dataset.yaml;*.yaml;*.yml",
        )
        if not path:
            return
        self._last_aux_path = str(Path(path).parent)
        path_obj = Path(path)
        if path_obj.name == "pose_project.json":
            try:
                data = json.loads(path_obj.read_text(encoding="utf-8"))
                images_dir = Path(data["images_dir"]).expanduser().resolve()
                labels_dir = Path(data["labels_dir"]).expanduser().resolve()
                class_names = data.get("class_names", [])
                keypoint_names = data.get("keypoint_names", [])
            except Exception as e:
                QMessageBox.warning(self, "Invalid project", str(e))
                return

            if (
                class_names != self.project.class_names
                or keypoint_names != self.project.keypoint_names
            ):
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "Aux project classes/keypoints do not match the current project.",
                )
                return

            image_paths = list_images_in_dir(images_dir)
            if not image_paths:
                QMessageBox.warning(self, "Empty", "No images found in aux project.")
                return

            item = QListWidgetItem(f"{images_dir}  (labels: {labels_dir})")
            self.aux_list.addItem(item)
            self._aux_datasets.append(
                {"images": image_paths, "labels_dir": labels_dir, "project_path": path}
            )
        else:
            try:
                items, info = load_yolo_dataset_items(path_obj)
            except Exception as e:
                QMessageBox.warning(self, "Invalid dataset", str(e))
                return

            names = info.get("names") or {}
            kpt_shape = info.get("kpt_shape")
            kpt_names = info.get("kpt_names")
            if list(names.values()) != self.project.class_names:
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "YOLO dataset class names do not match the current project.",
                )
                return
            if not kpt_shape or not kpt_names:
                QMessageBox.warning(
                    self,
                    "Missing keypoints",
                    "YOLO dataset is missing keypoint metadata.",
                )
                return
            if list(kpt_names.values())[0] != self.project.keypoint_names:
                QMessageBox.warning(
                    self,
                    "Mismatch",
                    "YOLO dataset keypoint names do not match the current project.",
                )
                return
            if not items:
                QMessageBox.warning(self, "Empty", "No labeled items found in dataset.")
                return

            item = QListWidgetItem(f"{path_obj}  (YOLO dataset)")
            self.aux_list.addItem(item)
            self._aux_items.extend(items)
            self._aux_datasets.append({"project_path": str(path_obj), "items": items})

        self._refresh_labeled_count()

    def _apply_settings(self):
        settings = _load_dialog_settings("training_runner")
        if not settings:
            return
        self.backend_combo.setCurrentText(settings.get("backend", self.backend_combo.currentText()))
        self.model_combo.setCurrentText(settings.get("model", self.model_combo.currentText()))
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.epochs_spin.setValue(int(settings.get("epochs", self.epochs_spin.value())))
        self.imgsz_spin.setValue(int(settings.get("imgsz", self.imgsz_spin.value())))
        self.patience_spin.setValue(int(settings.get("patience", self.patience_spin.value())))
        self.device_combo.setCurrentText(settings.get("device", self.device_combo.currentText()))
        self.cb_augment.setChecked(bool(settings.get("augment", self.cb_augment.isChecked())))
        self.hsv_h_spin.setValue(float(settings.get("hsv_h", self.hsv_h_spin.value())))
        self.hsv_s_spin.setValue(float(settings.get("hsv_s", self.hsv_s_spin.value())))
        self.hsv_v_spin.setValue(float(settings.get("hsv_v", self.hsv_v_spin.value())))
        self.degrees_spin.setValue(float(settings.get("degrees", self.degrees_spin.value())))
        self.translate_spin.setValue(float(settings.get("translate", self.translate_spin.value())))
        self.scale_spin.setValue(float(settings.get("scale", self.scale_spin.value())))
        self.fliplr_spin.setValue(float(settings.get("fliplr", self.fliplr_spin.value())))
        self.mosaic_spin.setValue(float(settings.get("mosaic", self.mosaic_spin.value())))
        self.mixup_spin.setValue(float(settings.get("mixup", self.mixup_spin.value())))
        self.train_split_spin.setValue(
            float(settings.get("train_split", self.train_split_spin.value()))
        )
        self.seed_spin.setValue(int(settings.get("seed", self.seed_spin.value())))
        self._last_aux_path = settings.get("last_aux_path", self._last_aux_path)

    def _apply_latest_weights_default(self):
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.model_combo.setCurrentText(str(self.project.latest_pose_weights))

    def _save_settings(self):
        _save_dialog_settings(
            "training_runner",
            {
                "backend": self.backend_combo.currentText(),
                "model": self.model_combo.currentText().strip(),
                "batch": int(self.batch_spin.value()),
                "epochs": int(self.epochs_spin.value()),
                "patience": int(self.patience_spin.value()),
                "imgsz": int(self.imgsz_spin.value()),
                "device": self.device_combo.currentText(),
                "augment": bool(self.cb_augment.isChecked()),
                "hsv_h": float(self.hsv_h_spin.value()),
                "hsv_s": float(self.hsv_s_spin.value()),
                "hsv_v": float(self.hsv_v_spin.value()),
                "degrees": float(self.degrees_spin.value()),
                "translate": float(self.translate_spin.value()),
                "scale": float(self.scale_spin.value()),
                "fliplr": float(self.fliplr_spin.value()),
                "mosaic": float(self.mosaic_spin.value()),
                "mixup": float(self.mixup_spin.value()),
                "train_split": float(self.train_split_spin.value()),
                "seed": int(self.seed_spin.value()),
                "last_aux_path": self._last_aux_path,
            },
        )

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _remove_aux_project(self):
        row = self.aux_list.currentRow()
        if row < 0:
            return
        self.aux_list.takeItem(row)
        try:
            removed = self._aux_datasets.pop(row)
            if "items" in removed:
                items = set(removed.get("items") or [])
                self._aux_items = [i for i in self._aux_items if i not in items]
        except Exception:
            pass
        self._refresh_labeled_count()

    def _refresh_labeled_count(self):
        labeled_count = len(
            list_labeled_indices(self.image_paths, self.project.labels_dir)
        )
        labeled_count += len(self._aux_items)
        for aux in self._aux_datasets:
            if "images" in aux and "labels_dir" in aux:
                labeled_count += len(
                    list_labeled_indices(aux["images"], aux["labels_dir"])
                )
        self.lbl_labeled.setText(f"Labeled frames: {labeled_count}")

    def _start_training(self):
        backend = self.backend_combo.currentText()
        if backend != "YOLO Pose":
            QMessageBox.information(
                self,
                "Not Implemented",
                "Only YOLO Pose is wired up right now. Other backends are coming soon.",
            )
            return

        labeled_count = len(
            list_labeled_indices(self.image_paths, self.project.labels_dir)
        )
        for aux in self._aux_datasets:
            labeled_count += len(
                list_labeled_indices(aux["images"], aux["labels_dir"])
            )
        if labeled_count < 2:
            QMessageBox.warning(
                self,
                "Not enough labels",
                "Need at least 2 labeled frames to train.",
            )
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.project.out_root / "runs" / "yolo_pose" / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Build dataset
        data_dir = run_dir / "data"
        try:
            dataset_info = build_yolo_pose_dataset(
                self.image_paths,
                self.project.labels_dir,
                data_dir,
                self.train_split_spin.value(),
                self.seed_spin.value(),
                self.project.class_names,
                self.project.keypoint_names,
                extra_datasets=[
                    (d["images"], d["labels_dir"]) for d in self._aux_datasets
                    if "images" in d and "labels_dir" in d
                ],
                extra_items=list(self._aux_items),
            )
        except Exception as e:
            QMessageBox.critical(self, "Dataset error", str(e))
            return

        config = {
            "backend": "yolo_pose",
            "model_weights": self.model_combo.currentText().strip(),
            "epochs": int(self.epochs_spin.value()),
            "patience": int(self.patience_spin.value()),
            "batch": int(self.batch_spin.value()),
            "imgsz": int(self.imgsz_spin.value()),
            "device": self.device_combo.currentText(),
            "augment": bool(self.cb_augment.isChecked()),
            "hsv_h": float(self.hsv_h_spin.value()),
            "hsv_s": float(self.hsv_s_spin.value()),
            "hsv_v": float(self.hsv_v_spin.value()),
            "degrees": float(self.degrees_spin.value()),
            "translate": float(self.translate_spin.value()),
            "scale": float(self.scale_spin.value()),
            "fliplr": float(self.fliplr_spin.value()),
            "mosaic": float(self.mosaic_spin.value()),
            "mixup": float(self.mixup_spin.value()),
            "dataset": {
                "yaml": str(dataset_info["yaml_path"]),
                    "train": str(dataset_info["train_list"]),
                    "val": str(dataset_info["val_list"]),
                    "train_count": dataset_info["train_count"],
                    "val_count": dataset_info["val_count"],
                    "manifest": str(dataset_info.get("manifest", "")),
                },
            "aux_datasets": [d["project_path"] for d in self._aux_datasets],
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        self._last_run_dir = run_dir
        self.lbl_run_dir.setText(f"Run dir: {run_dir}")
        self.log_view.clear()
        self.progress.setValue(0)
        self._update_loss_plot()
        self._loss_timer.start()

        # Start worker
        self._thread = QThread()
        self._worker = TrainingWorker(
            model_weights=self.model_combo.currentText().strip(),
            dataset_yaml=Path(dataset_info["yaml_path"]),
            run_dir=run_dir,
            epochs=self.epochs_spin.value(),
            patience=self.patience_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=self.imgsz_spin.value(),
            device=self.device_combo.currentText(),
            augment=self.cb_augment.isChecked(),
            hsv_h=self.hsv_h_spin.value(),
            hsv_s=self.hsv_s_spin.value(),
            hsv_v=self.hsv_v_spin.value(),
            degrees=self.degrees_spin.value(),
            translate=self.translate_spin.value(),
            scale=self.scale_spin.value(),
            fliplr=self.fliplr_spin.value(),
            mosaic=self.mosaic_spin.value(),
            mixup=self.mixup_spin.value(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_open_eval.setEnabled(False)

    def _stop_training(self):
        if self._worker:
            self._worker.cancel()
            self._append_log("Stop requested. Waiting for trainer to stop...")
        self.btn_stop.setEnabled(False)
        if self._loss_timer.isActive():
            self._loss_timer.stop()

    def _on_progress(self, epoch: int, epochs: int):
        if epochs > 0:
            pct = int((epoch / epochs) * 100)
            self.progress.setValue(min(100, max(0, pct)))

    def _on_finished(self, info: dict):
        self._append_log("Training complete.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        run_dir_info = info.get("run_dir")
        if run_dir_info:
            try:
                self._last_run_dir = Path(run_dir_info)
            except Exception:
                pass
        if self._loss_timer.isActive():
            self._loss_timer.stop()
        weights = info.get("weights") or ""
        self._last_weights = weights if weights else None
        if self._last_run_dir:
            run_dir = Path(self._last_run_dir)
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            if best.exists():
                self._last_weights = str(best)
            elif last.exists():
                self._last_weights = str(last)
        if weights:
            self._append_log(f"Weights: {weights}")
            self.btn_open_eval.setEnabled(True)
            self._set_latest_weights(weights)
        elif self._last_weights:
            self._append_log(f"Weights: {self._last_weights}")
            self.btn_open_eval.setEnabled(True)
            self._set_latest_weights(self._last_weights)
        elif (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.btn_open_eval.setEnabled(True)
        if self._last_weights:
            self._set_latest_weights(self._last_weights)
        if self._last_run_dir:
            run_dir = Path(self._last_run_dir)
            if list(run_dir.glob("**/events.out.tfevents.*")):
                self._append_log(f"TensorBoard: tensorboard --logdir {run_dir}")
            if (run_dir / "wandb").exists():
                self._append_log(
                    "W&B run detected in wandb/ (check your W&B dashboard)."
                )

    def _set_latest_weights(self, weights: str):
        try:
            w = Path(weights).resolve()
            if not w.is_file() or w.suffix != ".pt":
                self._append_log(f"Warning: ignoring non-.pt weights: {w}")
                return
            self.project.latest_pose_weights = w
            self.project.project_path.write_text(
                json.dumps(self.project.to_json(), indent=2), encoding="utf-8"
            )
        except Exception as e:
            self._append_log(f"Warning: failed to save latest weights: {e}")

    def _on_failed(self, msg: str):
        self._append_log(msg)
        QMessageBox.critical(self, "Training failed", msg)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if self._loss_timer.isActive():
            self._loss_timer.stop()

    def _open_eval(self):
        weights = self._last_weights
        if not weights and hasattr(self.project, "latest_pose_weights"):
            if self.project.latest_pose_weights and Path(self.project.latest_pose_weights).exists():
                weights = str(self.project.latest_pose_weights)
        if not weights:
            QMessageBox.information(self, "Missing weights", "No valid weights found to evaluate.")
            return
        dlg = EvaluationDashboardDialog(
            self,
            self.project,
            self.image_paths,
            weights_path=weights,
        )
        dlg.exec()

    def _update_loss_plot(self):
        if not self._last_run_dir:
            return
        results_path = Path(self._last_run_dir) / "results.csv"
        if not results_path.exists():
            return
        try:
            rows = list(
                csv.reader(
                    results_path.read_text(encoding="utf-8").splitlines()
                )
            )
            if len(rows) < 2:
                return
            header = rows[0]
            train_cols = [
                i for i, h in enumerate(header) if "train/" in h and "loss" in h
            ]
            val_cols = [i for i, h in enumerate(header) if "val/" in h and "loss" in h]
            train_vals = []
            val_vals = []
            for row in rows[1:]:
                if not row:
                    continue
                t = 0.0
                v = 0.0
                tcount = 0
                vcount = 0
                for i in train_cols:
                    if i < len(row) and row[i]:
                        t += float(row[i])
                        tcount += 1
                for i in val_cols:
                    if i < len(row) and row[i]:
                        v += float(row[i])
                        vcount += 1
                train_vals.append(t if tcount else None)
                val_vals.append(v if vcount else None)
            img = _make_loss_plot_image(train_vals, val_vals)
            self.lbl_loss_plot.setPixmap(QPixmap.fromImage(img))
        except Exception:
            return


class EvaluationWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        weights_path: Path,
        eval_paths: List[Path],
        labels_dir: Path,
        keypoint_names: List[str],
        run_dir: Path,
        out_root: Path,
        device: str,
        imgsz: int,
        conf: float,
        pck_thr: float,
        oks_sigma: float,
        batch: int,
        pred_cache: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    ):
        super().__init__()
        self.weights_path = Path(weights_path)
        self.eval_paths = list(eval_paths)
        self.labels_dir = Path(labels_dir)
        self.keypoint_names = list(keypoint_names)
        self.run_dir = Path(run_dir)
        self.out_root = Path(out_root)
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.pck_thr = float(pck_thr)
        self.oks_sigma = float(oks_sigma)
        self.batch = int(batch)
        self.pred_cache = pred_cache
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _extract_best_prediction(self, result, num_kpts: int):
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
        pred_conf = conf[best] if conf is not None else np.zeros((num_kpts,))

        if pred_xy.shape[0] != num_kpts:
            num = min(pred_xy.shape[0], num_kpts)
            tmp_xy = np.zeros((num_kpts, 2), dtype=np.float32)
            tmp_conf = np.zeros((num_kpts,), dtype=np.float32)
            tmp_xy[:num] = pred_xy[:num]
            tmp_conf[:num] = pred_conf[:num]
            pred_xy = tmp_xy
            pred_conf = tmp_conf

        return pred_xy, pred_conf

    def run(self):
        try:
            try:
                from ultralytics import YOLO
            except Exception as e:
                self.failed.emit(
                    f"Ultralytics not available. Install with: pip install ultralytics\n{e}"
                )
                return

            if not self.weights_path.exists() and not self.pred_cache:
                self.failed.emit(f"Weights not found: {self.weights_path}")
                return

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.log.emit(f"Eval run dir: {self.run_dir}")
            self.log.emit(f"Evaluating {len(self.eval_paths)} frames...")

            total = len(self.eval_paths)
            num_kpts = len(self.keypoint_names)

            per_frame = []
            per_kpt_errors = [[] for _ in range(num_kpts)]
            per_kpt_confs = [[] for _ in range(num_kpts)]
            kpt_counts = [0] * num_kpts
            kpt_pck = [0] * num_kpts
            kpt_oks = [0.0] * num_kpts
            kpt_err_sum = [0.0] * num_kpts
            kpt_conf_sum = [0.0] * num_kpts

            total_kpts = 0
            total_pck = 0
            total_oks = 0.0
            total_err = 0.0
            total_conf = 0.0

            if self.pred_cache is None:
                infer = _make_pose_infer(self.out_root, self.keypoint_names)
                preds, err = infer.predict(
                    self.weights_path,
                    self.eval_paths,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    batch=self.batch,
                    progress_cb=self.progress.emit,
                )
                if preds is None:
                    self.failed.emit(err or "Prediction failed.")
                    return
                self.pred_cache = preds

            if self.pred_cache:
                try:
                    from PIL import Image
                except Exception as e:
                    self.failed.emit(f"PIL not available: {e}")
                    return

                for idx, img_path in enumerate(self.eval_paths):
                    if self._cancel:
                        self.failed.emit("Canceled.")
                        return

                    label_path = self.labels_dir / f"{img_path.stem}.txt"
                    gt = load_yolo_pose_label(label_path, num_kpts)
                    if not gt:
                        self.progress.emit(idx + 1, total)
                        continue

                    _, gt_kpts, bbox = gt
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        w, h = (1, 1)
                    if bbox is not None:
                        _, _, bw, bh = bbox
                        scale = max(bw * w, bh * h)
                    else:
                        scale = max(w, h)
                    if scale <= 0:
                        scale = max(w, h, 1)

                    pred_list = self.pred_cache.get(str(img_path))
                    if pred_list is None:
                        pred_list = self.pred_cache.get(str(img_path.resolve()))
                    if not pred_list:
                        self.progress.emit(idx + 1, total)
                        continue

                    pred_xy = np.array([[p[0], p[1]] for p in pred_list], dtype=np.float32)
                    pred_conf = np.array([p[2] for p in pred_list], dtype=np.float32)
                    if pred_xy.size == 0:
                        self.progress.emit(idx + 1, total)
                        continue

                    frame_errs = [None] * num_kpts
                    frame_confs = [None] * num_kpts
                    for k in range(num_kpts):
                        if gt_kpts[k].v <= 0:
                            continue

                        px, py = pred_xy[k]
                        err = float(math.hypot(px - (gt_kpts[k].x * w), py - (gt_kpts[k].y * h)))
                        conf = float(pred_conf[k]) if pred_conf is not None else 0.0

                        ok = err <= (self.pck_thr * scale)
                        oks = math.exp(-((err**2) / (2 * (self.oks_sigma * scale) ** 2)))

                        total_kpts += 1
                        total_pck += 1 if ok else 0
                        total_oks += oks
                        total_err += err
                        total_conf += conf

                        kpt_counts[k] += 1
                        kpt_pck[k] += 1 if ok else 0
                        kpt_oks[k] += oks
                        kpt_err_sum[k] += err
                        kpt_conf_sum[k] += conf

                        per_kpt_errors[k].append(err)
                        per_kpt_confs[k].append(conf)
                        frame_errs[k] = err
                        frame_confs[k] = conf

                    valid_errs = [e for e in frame_errs if e is not None]
                    valid_confs = [c for c in frame_confs if c is not None]
                    if valid_errs:
                        mean_err = float(np.mean(valid_errs))
                        mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
                        mean_err_norm = mean_err / (scale + 1e-6)
                        per_frame.append(
                            {
                                "image_path": str(img_path),
                                "mean_error_px": mean_err,
                                "mean_error_norm": mean_err_norm,
                                "mean_conf": mean_conf,
                                "kpt_errors": frame_errs,
                                "kpt_confs": frame_confs,
                            }
                        )

                    self.progress.emit(idx + 1, total)
                # prediction handled via subprocess + cache only

            overall = {
                "frames": len(per_frame),
                "total_kpts": total_kpts,
                "pck": (total_pck / total_kpts) if total_kpts else 0.0,
                "oks": (total_oks / total_kpts) if total_kpts else 0.0,
                "mean_error_px": (total_err / total_kpts) if total_kpts else 0.0,
                "mean_conf": (total_conf / total_kpts) if total_kpts else 0.0,
            }

            per_kpt = []
            for i, name in enumerate(self.keypoint_names):
                count = kpt_counts[i]
                per_kpt.append(
                    {
                        "name": name,
                        "count": count,
                        "pck": (kpt_pck[i] / count) if count else 0.0,
                        "oks": (kpt_oks[i] / count) if count else 0.0,
                        "mean_error_px": (kpt_err_sum[i] / count) if count else 0.0,
                        "mean_conf": (kpt_conf_sum[i] / count) if count else 0.0,
                    }
                )

            per_frame_sorted = sorted(
                per_frame, key=lambda x: x.get("mean_error_norm", 0.0), reverse=True
            )
            worst = per_frame_sorted[:50]

            # Save results
            (self.run_dir / "eval_metrics.json").write_text(
                json.dumps({"overall": overall, "per_keypoint": per_kpt}, indent=2),
                encoding="utf-8",
            )

            csv_path = self.run_dir / "eval_per_frame.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    "image_path",
                    "mean_error_px",
                    "mean_error_norm",
                    "mean_conf",
                ]
                for i in range(num_kpts):
                    header.append(f"kpt_err_{i}")
                for i in range(num_kpts):
                    header.append(f"kpt_conf_{i}")
                writer.writerow(header)
                for row in per_frame:
                    errs = row.get("kpt_errors", [])
                    confs = row.get("kpt_confs", [])
                    errs = ["" if e is None else e for e in errs]
                    confs = ["" if c is None else c for c in confs]
                    errs = errs + [""] * max(0, num_kpts - len(errs))
                    confs = confs + [""] * max(0, num_kpts - len(confs))
                    writer.writerow(
                        [
                            row.get("image_path", ""),
                            row.get("mean_error_px", ""),
                            row.get("mean_error_norm", ""),
                            row.get("mean_conf", ""),
                            *errs[:num_kpts],
                            *confs[:num_kpts],
                        ]
                    )

            worst_path = self.run_dir / "eval_worst_frames.csv"
            with worst_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["image_path", "mean_error_norm", "mean_error_px"])
                for row in worst:
                    writer.writerow(
                        [
                            row.get("image_path", ""),
                            row.get("mean_error_norm", 0.0),
                            row.get("mean_error_px", 0.0),
                        ]
                    )

            self.log.emit("Evaluation finished.")
            self.finished.emit(
                {
                    "overall": overall,
                    "per_keypoint": per_kpt,
                    "per_frame": per_frame,
                    "worst": worst,
                    "per_kpt_errors": per_kpt_errors,
                }
            )

        except Exception as e:
            self.failed.emit(str(e))


class EvaluationDashboardDialog(QDialog):
    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        weights_path: Optional[str] = None,
        add_frames_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Evaluation Dashboard")
        self.setMinimumSize(QSize(900, 700))

        self.project = project
        self.image_paths = image_paths
        self.add_frames_callback = add_frames_callback
        self._thread = None
        self._worker = None
        self._path_to_index = {str(p): i for i, p in enumerate(image_paths)}
        self.infer = _make_pose_infer(
            self.project.out_root, self.project.keypoint_names
        )

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Config
        cfg_group = QGroupBox("Config")
        cfg_layout = QFormLayout(cfg_group)

        self.weights_edit = QLineEdit(weights_path or "")
        self.btn_weights_browse = QPushButton("Browse…")
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_edit, 1)
        weights_row.addWidget(self.btn_weights_browse)
        cfg_layout.addRow("Weights:", weights_row)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        cfg_layout.addRow("Device:", self.device_combo)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        cfg_layout.addRow("Image size:", self.imgsz_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        cfg_layout.addRow("Batch:", self.batch_spin)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        cfg_layout.addRow("Conf threshold:", self.conf_spin)

        self.pck_spin = QDoubleSpinBox()
        self.pck_spin.setRange(0.01, 0.5)
        self.pck_spin.setSingleStep(0.01)
        self.pck_spin.setValue(0.05)
        cfg_layout.addRow("PCK threshold:", self.pck_spin)

        self.oks_spin = QDoubleSpinBox()
        self.oks_spin.setRange(0.01, 1.0)
        self.oks_spin.setSingleStep(0.05)
        self.oks_spin.setValue(0.1)
        cfg_layout.addRow("OKS sigma:", self.oks_spin)

        self.cb_use_cache = QCheckBox("Use cached predictions (if available)")
        self.cb_use_cache.setChecked(True)
        cfg_layout.addRow("", self.cb_use_cache)

        self.out_dir_edit = QLineEdit("")
        self.btn_out_browse = QPushButton("Browse…")
        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(self.btn_out_browse)
        cfg_layout.addRow("Output dir:", out_row)

        content_layout.addWidget(cfg_group)

        # Progress + logs
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.lbl_loss_plot = QLabel()
        self.lbl_loss_plot.setMinimumHeight(220)
        content_layout.addWidget(self.lbl_loss_plot)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.lbl_overall = QLabel("Overall: --")
        results_layout.addWidget(self.lbl_overall)

        self.kpt_table = QTableWidget(0, 5)
        self.kpt_table.setHorizontalHeaderLabels(
            ["Keypoint", "Mean Err (px)", "PCK", "OKS", "Mean Conf"]
        )
        results_layout.addWidget(self.kpt_table)

        charts_row = QHBoxLayout()
        self.lbl_hist = QLabel()
        self.lbl_heat = QLabel()
        charts_row.addWidget(self.lbl_hist, 1)
        charts_row.addWidget(self.lbl_heat, 1)
        results_layout.addLayout(charts_row)

        self.worst_list = QListWidget()
        results_layout.addWidget(QLabel("Worst frames"))
        results_layout.addWidget(self.worst_list, 1)

        self.btn_add_selected = None
        self.btn_add_top = None

        content_layout.addWidget(results_group, 2)

        # Buttons
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Run Evaluation")
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_run)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.btn_weights_browse.clicked.connect(self._browse_weights)
        self.btn_out_browse.clicked.connect(self._browse_out)
        self.btn_run.clicked.connect(self._run_eval)
        self.btn_close.clicked.connect(self.reject)

        self._update_default_out_dir()
        self._apply_settings()
        self._apply_latest_weights_default()

    def _apply_settings(self):
        settings = _load_dialog_settings("evaluation_dashboard")
        if not settings:
            return
        self.weights_edit.setText(settings.get("weights", self.weights_edit.text()))
        self.device_combo.setCurrentText(settings.get("device", self.device_combo.currentText()))
        self.imgsz_spin.setValue(int(settings.get("imgsz", self.imgsz_spin.value())))
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.conf_spin.setValue(float(settings.get("conf", self.conf_spin.value())))
        self.pck_spin.setValue(float(settings.get("pck", self.pck_spin.value())))
        self.oks_spin.setValue(float(settings.get("oks", self.oks_spin.value())))
        self.out_dir_edit.setText(settings.get("out_dir", self.out_dir_edit.text()))

    def _apply_latest_weights_default(self):
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and not self.weights_edit.text().strip()
            and Path(self.project.latest_pose_weights).exists()
        ):
            self.weights_edit.setText(str(self.project.latest_pose_weights))
            self._update_default_out_dir()

    def _save_settings(self):
        _save_dialog_settings(
            "evaluation_dashboard",
            {
                "weights": self.weights_edit.text().strip(),
                "device": self.device_combo.currentText(),
                "imgsz": int(self.imgsz_spin.value()),
                "batch": int(self.batch_spin.value()),
                "conf": float(self.conf_spin.value()),
                "pck": float(self.pck_spin.value()),
                "oks": float(self.oks_spin.value()),
                "out_dir": self.out_dir_edit.text().strip(),
            },
        )

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _browse_weights(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
        if path:
            self.weights_edit.setText(path)
            self._update_default_out_dir()

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Select output dir")
        if path:
            self.out_dir_edit.setText(path)

    def _update_default_out_dir(self):
        weights = Path(self.weights_edit.text().strip())
        if weights.exists() and weights.parent.name == "weights":
            out_dir = weights.parent.parent
        else:
            out_dir = (
                self.project.out_root
                / "runs"
                / "eval"
                / datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        self.out_dir_edit.setText(str(out_dir))

    def _collect_eval_paths(self) -> List[Path]:
        # all labeled frames
        labeled = list_labeled_indices(self.image_paths, self.project.labels_dir)
        return [self.image_paths[i] for i in labeled]

    def _run_eval(self):
        weights = Path(self.weights_edit.text().strip())
        if not weights.exists():
            QMessageBox.warning(self, "Missing weights", "Please select weights file.")
            return

        eval_paths = self._collect_eval_paths()
        if not eval_paths:
            QMessageBox.warning(self, "No data", "No labeled frames found to evaluate.")
            return

        out_dir = Path(self.out_dir_edit.text().strip())
        out_dir.mkdir(parents=True, exist_ok=True)

        pred_cache = None
        if self.cb_use_cache.isChecked():
            cache = self.infer.get_cache_for_paths(weights, eval_paths)
            if cache is not None:
                pred_cache = cache
            else:
                self._append_log("Cached predictions incomplete. Using model.")

        self.log_view.clear()
        self.progress.setValue(0)
        self.worst_list.clear()
        self.kpt_table.setRowCount(0)
        self.lbl_overall.setText("Overall: --")
        # worst list is informational only for labeled frames

        self._thread = QThread()
        self._worker = EvaluationWorker(
            weights_path=weights,
            eval_paths=eval_paths,
            labels_dir=self.project.labels_dir,
            keypoint_names=self.project.keypoint_names,
            run_dir=out_dir,
            out_root=self.project.out_root,
            device=self.device_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            conf=self.conf_spin.value(),
            pck_thr=self.pck_spin.value(),
            oks_sigma=self.oks_spin.value(),
            batch=self.batch_spin.value(),
            pred_cache=pred_cache,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = int((done / total) * 100)
            self.progress.setValue(min(100, max(0, pct)))

    def _on_finished(self, payload: dict):
        overall = payload.get("overall", {})
        per_kpt = payload.get("per_keypoint", [])
        per_kpt_errors = payload.get("per_kpt_errors", [])
        worst = payload.get("worst", [])

        self.lbl_overall.setText(
            "Overall: PCK="
            + _format_float(overall.get("pck"))
            + " OKS="
            + _format_float(overall.get("oks"))
            + " MeanErr(px)="
            + _format_float(overall.get("mean_error_px"))
        )

        self.kpt_table.setRowCount(len(per_kpt))
        for row, kpt in enumerate(per_kpt):
            self.kpt_table.setItem(row, 0, QTableWidgetItem(kpt.get("name", "")))
            self.kpt_table.setItem(
                row, 1, QTableWidgetItem(_format_float(kpt.get("mean_error_px")))
            )
            self.kpt_table.setItem(
                row, 2, QTableWidgetItem(_format_float(kpt.get("pck")))
            )
            self.kpt_table.setItem(
                row, 3, QTableWidgetItem(_format_float(kpt.get("oks")))
            )
            self.kpt_table.setItem(
                row, 4, QTableWidgetItem(_format_float(kpt.get("mean_conf")))
            )

        # Charts
        all_errors = [e for k in per_kpt_errors for e in k]
        hist_img = _make_histogram_image(all_errors)
        self.lbl_hist.setPixmap(QPixmap.fromImage(hist_img))

        # Heatmap: keypoint x error bins
        bins = 20
        heat = np.zeros((len(per_kpt_errors), bins), dtype=np.float32)
        if all_errors:
            max_err = max(all_errors)
            max_err = max(max_err, 1e-6)
            for i, errs in enumerate(per_kpt_errors):
                if not errs:
                    continue
                hist, _ = np.histogram(errs, bins=bins, range=(0, max_err))
                heat[i, :] = hist
        heat_img = _make_heatmap_image(heat)
        self.lbl_heat.setPixmap(QPixmap.fromImage(heat_img))

        # Worst frames
        self.worst_list.clear()
        for row in worst:
            item = QListWidgetItem(
                f"{Path(row.get('image_path', '')).name} | err={_format_float(row.get('mean_error_norm'))}"
            )
            item.setData(Qt.UserRole, row.get("image_path"))
            self.worst_list.addItem(item)

        self._append_log("Evaluation complete.")
    def _on_failed(self, msg: str):
        self._append_log(msg)
        QMessageBox.critical(self, "Evaluation failed", msg)


class ActiveLearningWorker(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        strategy: str,
        image_paths: List[Path],
        candidate_indices: List[int],
        num_kpts: int,
        keypoint_names: List[str],
        out_root: Path,
        weights_a: Optional[str],
        weights_b: Optional[str],
        device: str,
        imgsz: int,
        conf: float,
        batch: int,
        eval_csv: Optional[str],
        preds_cache_a: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
        preds_cache_b: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    ):
        super().__init__()
        self.strategy = strategy
        self.image_paths = image_paths
        self.candidate_indices = candidate_indices
        self.num_kpts = int(num_kpts)
        self.keypoint_names = list(keypoint_names)
        self.out_root = Path(out_root)
        self.weights_a = weights_a
        self.weights_b = weights_b
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.batch = int(batch)
        self.eval_csv = eval_csv
        self.preds_cache_a = preds_cache_a
        self.preds_cache_b = preds_cache_b
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _predict_keypoints(self, weights_path: str, paths: List[Path]):
        infer = _make_pose_infer(self.out_root, self.keypoint_names)
        preds_list, err = infer.predict(
            Path(weights_path),
            paths,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            batch=self.batch,
            progress_cb=self.progress.emit,
        )
        if preds_list is None:
            self.failed.emit(err or "Prediction failed.")
            return None

        preds = {}
        total = len(paths)
        for i, p in enumerate(paths):
            pred = preds_list.get(str(p)) or preds_list.get(str(p.resolve()))
            if not pred:
                preds[str(p)] = (None, None)
            else:
                pred_xy = np.array([[x, y] for x, y, _ in pred], dtype=np.float32)
                pred_conf = np.array([c for _, _, c in pred], dtype=np.float32)
                preds[str(p)] = (pred_xy, pred_conf)
            self.progress.emit(i + 1, total)
        return preds

    def _extract_best_prediction(self, result):
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

    def run(self):
        try:
            paths = [self.image_paths[i] for i in self.candidate_indices]
            if not paths:
                self.failed.emit("No candidate frames.")
                return

            if self.strategy == "eval_error":
                if not self.eval_csv or not Path(self.eval_csv).exists():
                    self.failed.emit("Eval CSV not found.")
                    return
                path_set = {str(p.resolve()) for p in paths}
                rows = []
                with Path(self.eval_csv).open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                scores = []
                for row in rows:
                    raw = row.get("image_path", "")
                    if not raw:
                        continue
                    p = Path(raw)
                    if not p.is_absolute():
                        p = (Path(self.eval_csv).parent / p).resolve()
                    else:
                        p = p.resolve()
                    path = str(p)
                    if path not in path_set:
                        continue
                    score = float(row.get("mean_error_norm", 0.0) or 0.0)
                    scores.append((path, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                self.finished.emit(scores)
                return

            if self.strategy == "lowest_conf":
                if self.preds_cache_a:
                    scores = []
                    for p in paths:
                        pred = self.preds_cache_a.get(str(p)) or self.preds_cache_a.get(
                            str(p.resolve())
                        )
                        if not pred:
                            score = 0.0
                        else:
                            confs = [c for _, _, c in pred]
                            score = float(np.mean(confs)) if confs else 0.0
                        scores.append((str(p), score))
                    scores.sort(key=lambda x: x[1])
                    self.finished.emit(scores)
                    return
                else:
                    if not self.weights_a or not Path(self.weights_a).exists():
                        self.failed.emit("Weights not found.")
                        return
                    preds = self._predict_keypoints(self.weights_a, paths)
                    if preds is None:
                        self.failed.emit("Canceled.")
                        return
                scores = []
                for p in paths:
                    key = str(p)
                    pred = preds.get(key)
                    if pred is None or pred[0] is None:
                        score = 0.0
                    else:
                        confs = pred[1]
                        score = float(np.mean(confs)) if confs is not None else 0.0
                    scores.append((key, score))
                scores.sort(key=lambda x: x[1])
                self.finished.emit(scores)
                return

            if self.strategy == "disagreement":
                if self.preds_cache_a and self.preds_cache_b:
                    scores = []
                    for p in paths:
                        a = self.preds_cache_a.get(str(p)) or self.preds_cache_a.get(
                            str(p.resolve())
                        )
                        b = self.preds_cache_b.get(str(p)) or self.preds_cache_b.get(
                            str(p.resolve())
                        )
                        if not a or not b:
                            score = 1e9
                        else:
                            a_xy = np.array([[x, y] for x, y, _ in a], dtype=np.float32)
                            b_xy = np.array([[x, y] for x, y, _ in b], dtype=np.float32)
                            diff = a_xy - b_xy
                            dists = np.linalg.norm(diff, axis=1)
                            score = float(np.mean(dists)) if dists.size else 0.0
                        scores.append((str(p), score))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    self.finished.emit(scores)
                    return
                else:
                    if not self.weights_a or not Path(self.weights_a).exists():
                        self.failed.emit("Weights A not found.")
                        return
                    if not self.weights_b or not Path(self.weights_b).exists():
                        self.failed.emit("Weights B not found.")
                        return
                    preds_a = self._predict_keypoints(self.weights_a, paths)
                    if preds_a is None:
                        self.failed.emit("Canceled.")
                        return
                    preds_b = self._predict_keypoints(self.weights_b, paths)
                    if preds_b is None:
                        self.failed.emit("Canceled.")
                        return

                scores = []
                for p in paths:
                    key = str(p)
                    a = preds_a.get(key)
                    b = preds_b.get(key)
                    if a is None or b is None or a[0] is None or b[0] is None:
                        score = 1e9
                    else:
                        diff = a[0] - b[0]
                        dists = np.linalg.norm(diff, axis=1)
                        score = float(np.mean(dists)) if dists.size else 0.0
                    scores.append((key, score))
                scores.sort(key=lambda x: x[1], reverse=True)
                self.finished.emit(scores)
                return

            self.failed.emit("Unknown strategy")

        except Exception as e:
            self.failed.emit(str(e))


class ActiveLearningDialog(QDialog):
    def __init__(
        self,
        parent,
        project,
        image_paths: List[Path],
        is_labeled_fn,
        labeling_indices,
        add_frames_callback=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Active Learning Sampler")
        self.setMinimumSize(QSize(860, 650))

        self.project = project
        self.image_paths = image_paths
        self.is_labeled_fn = is_labeled_fn
        self.labeling_indices = set(labeling_indices)
        self.add_frames_callback = add_frames_callback
        self._thread = None
        self._worker = None
        self._path_to_index = {str(p): i for i, p in enumerate(image_paths)}
        self._scores = []
        self.infer = _make_pose_infer(
            self.project.out_root, self.project.keypoint_names
        )

        layout = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)

        # Strategy
        strat_group = QGroupBox("Strategy")
        strat_layout = QFormLayout(strat_group)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(
            [
                "Lowest keypoint confidence",
                "Largest model disagreement",
                "Largest train/val error",
            ]
        )
        strat_layout.addRow("Strategy:", self.strategy_combo)
        content_layout.addWidget(strat_group)

        # Common config
        common_group = QGroupBox("Selection")
        common_layout = QFormLayout(common_group)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Unlabeled only", "All frames", "Labeling set"])
        common_layout.addRow("Scope:", self.scope_combo)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 5000)
        self.n_spin.setValue(50)
        common_layout.addRow("Suggest N frames:", self.n_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(get_available_devices())
        common_layout.addRow("Device:", self.device_combo)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(64, 4096)
        self.imgsz_spin.setValue(640)
        common_layout.addRow("Image size:", self.imgsz_spin)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        common_layout.addRow("Conf threshold:", self.conf_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)
        common_layout.addRow("Batch:", self.batch_spin)

        self.cb_use_cache = QCheckBox("Use cached predictions (if available)")
        self.cb_use_cache.setChecked(True)
        common_layout.addRow("", self.cb_use_cache)

        content_layout.addWidget(common_group)

        # Strategy-specific controls
        self.stack = QStackedWidget()

        # Lowest confidence
        w1 = QWidget()
        w1_layout = QFormLayout(w1)
        self.weights_a_edit = QLineEdit("")
        self.btn_weights_a = QPushButton("Browse…")
        row_a = QHBoxLayout()
        row_a.addWidget(self.weights_a_edit, 1)
        row_a.addWidget(self.btn_weights_a)
        w1_layout.addRow("Weights:", row_a)

        # Disagreement
        w2 = QWidget()
        w2_layout = QFormLayout(w2)
        self.weights_b1_edit = QLineEdit("")
        self.btn_weights_b1 = QPushButton("Browse…")
        row_b1 = QHBoxLayout()
        row_b1.addWidget(self.weights_b1_edit, 1)
        row_b1.addWidget(self.btn_weights_b1)
        w2_layout.addRow("Weights A:", row_b1)

        self.weights_b2_edit = QLineEdit("")
        self.btn_weights_b2 = QPushButton("Browse…")
        row_b2 = QHBoxLayout()
        row_b2.addWidget(self.weights_b2_edit, 1)
        row_b2.addWidget(self.btn_weights_b2)
        w2_layout.addRow("Weights B:", row_b2)

        # Eval error
        w3 = QWidget()
        w3_layout = QFormLayout(w3)
        self.eval_csv_edit = QLineEdit("")
        self.btn_eval_csv = QPushButton("Browse…")
        row_c = QHBoxLayout()
        row_c.addWidget(self.eval_csv_edit, 1)
        row_c.addWidget(self.btn_eval_csv)
        w3_layout.addRow("Eval CSV:", row_c)

        self.stack.addWidget(w1)
        self.stack.addWidget(w2)
        self.stack.addWidget(w3)
        content_layout.addWidget(self.stack)

        # Progress + log
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        content_layout.addWidget(self.progress)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        content_layout.addWidget(self.log_view, 1)

        # Results
        self.results_list = QListWidget()
        content_layout.addWidget(QLabel("Suggested frames"))
        content_layout.addWidget(self.results_list, 2)

        # Buttons
        btns = QHBoxLayout()
        self.btn_run = QPushButton("Suggest Frames")
        self.btn_add = QPushButton("Add Selected to Labeling")
        self.btn_add.setEnabled(False)
        self.btn_close = QPushButton("Close")
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_add)
        btns.addStretch(1)
        btns.addWidget(self.btn_close)
        content_layout.addLayout(btns)

        # Wiring
        self.strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self.btn_weights_a.clicked.connect(
            lambda: self._browse_weights(self.weights_a_edit)
        )
        self.btn_weights_b1.clicked.connect(
            lambda: self._browse_weights(self.weights_b1_edit)
        )
        self.btn_weights_b2.clicked.connect(
            lambda: self._browse_weights(self.weights_b2_edit)
        )
        self.btn_eval_csv.clicked.connect(self._browse_eval_csv)
        self.btn_run.clicked.connect(self._run)
        self.btn_add.clicked.connect(self._add_selected)
        self.btn_close.clicked.connect(self.reject)

        self._on_strategy_changed(0)
        self._apply_settings()
        self._apply_latest_weights_default()

    def _apply_settings(self):
        settings = _load_dialog_settings("active_learning")
        if not settings:
            return
        self.strategy_combo.setCurrentIndex(int(settings.get("strategy_index", 0)))
        self.scope_combo.setCurrentText(settings.get("scope", self.scope_combo.currentText()))
        self.n_spin.setValue(int(settings.get("n", self.n_spin.value())))
        self.device_combo.setCurrentText(settings.get("device", self.device_combo.currentText()))
        self.imgsz_spin.setValue(int(settings.get("imgsz", self.imgsz_spin.value())))
        self.conf_spin.setValue(float(settings.get("conf", self.conf_spin.value())))
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.weights_a_edit.setText(settings.get("weights_a", self.weights_a_edit.text()))
        self.weights_b1_edit.setText(settings.get("weights_b1", self.weights_b1_edit.text()))
        self.weights_b2_edit.setText(settings.get("weights_b2", self.weights_b2_edit.text()))
        self.eval_csv_edit.setText(settings.get("eval_csv", self.eval_csv_edit.text()))

    def _apply_latest_weights_default(self):
        if (
            hasattr(self.project, "latest_pose_weights")
            and self.project.latest_pose_weights
            and Path(self.project.latest_pose_weights).exists()
        ):
            latest = str(self.project.latest_pose_weights)
            if not self.weights_a_edit.text().strip():
                self.weights_a_edit.setText(latest)
            if not self.weights_b1_edit.text().strip():
                self.weights_b1_edit.setText(latest)

    def _save_settings(self):
        _save_dialog_settings(
            "active_learning",
            {
                "strategy_index": int(self.strategy_combo.currentIndex()),
                "scope": self.scope_combo.currentText(),
                "n": int(self.n_spin.value()),
                "device": self.device_combo.currentText(),
                "imgsz": int(self.imgsz_spin.value()),
                "conf": float(self.conf_spin.value()),
                "batch": int(self.batch_spin.value()),
                "weights_a": self.weights_a_edit.text().strip(),
                "weights_b1": self.weights_b1_edit.text().strip(),
                "weights_b2": self.weights_b2_edit.text().strip(),
                "eval_csv": self.eval_csv_edit.text().strip(),
            },
        )

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def _append_log(self, msg: str):
        self.log_view.appendPlainText(msg)
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum()
        )

    def _browse_weights(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Select weights", "", "*.pt")
        if path:
            line_edit.setText(path)

    def _browse_eval_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select eval CSV", "", "*.csv")
        if path:
            self.eval_csv_edit.setText(path)

    def _on_strategy_changed(self, idx: int):
        self.stack.setCurrentIndex(idx)

    def _collect_candidates(self) -> List[int]:
        scope = self.scope_combo.currentText()
        indices = []
        for i, p in enumerate(self.image_paths):
            if scope == "Unlabeled only" and self.is_labeled_fn(p):
                continue
            if scope == "Labeling set" and i not in self.labeling_indices:
                continue
            indices.append(i)
        return indices

    def _run(self):
        candidates = self._collect_candidates()
        if not candidates:
            QMessageBox.warning(self, "No candidates", "No frames match the scope.")
            return

        self.results_list.clear()
        self.log_view.clear()
        self.progress.setValue(0)

        strat_idx = self.strategy_combo.currentIndex()
        if strat_idx == 0:
            strategy = "lowest_conf"
            weights_a = self.weights_a_edit.text().strip()
            weights_b = None
            eval_csv = None
        elif strat_idx == 1:
            strategy = "disagreement"
            weights_a = self.weights_b1_edit.text().strip()
            weights_b = self.weights_b2_edit.text().strip()
            eval_csv = None
        else:
            strategy = "eval_error"
            weights_a = None
            weights_b = None
            eval_csv = self.eval_csv_edit.text().strip()

        preds_cache_a = None
        preds_cache_b = None
        if self.cb_use_cache.isChecked():
            if weights_a:
                cache_a = self.infer.get_cache_for_paths(
                    Path(weights_a), [self.image_paths[i] for i in candidates]
                )
                if cache_a is not None:
                    preds_cache_a = cache_a
            if weights_b:
                cache_b = self.infer.get_cache_for_paths(
                    Path(weights_b), [self.image_paths[i] for i in candidates]
                )
                if cache_b is not None:
                    preds_cache_b = cache_b

        self._thread = QThread()
        self._worker = ActiveLearningWorker(
            strategy=strategy,
            image_paths=self.image_paths,
            candidate_indices=candidates,
            num_kpts=len(self.project.keypoint_names),
            keypoint_names=self.project.keypoint_names,
            out_root=self.project.out_root,
            weights_a=weights_a,
            weights_b=weights_b,
            device=self.device_combo.currentText(),
            imgsz=self.imgsz_spin.value(),
            conf=self.conf_spin.value(),
            batch=self.batch_spin.value(),
            eval_csv=eval_csv,
            preds_cache_a=preds_cache_a,
            preds_cache_b=preds_cache_b,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = int((done / total) * 100)
            self.progress.setValue(min(100, max(0, pct)))

    def _on_finished(self, scores: List[Tuple[str, float]]):
        self._scores = scores
        n = min(self.n_spin.value(), len(scores))
        for path, score in scores[:n]:
            name = Path(path).name
            item = QListWidgetItem(f"{name} | score={_format_float(score)}")
            item.setData(Qt.UserRole, path)
            self.results_list.addItem(item)
        self.btn_add.setEnabled(True if n > 0 else False)
        self._append_log("Suggestions ready.")

    def _on_failed(self, msg: str):
        self._append_log(msg)
        QMessageBox.critical(self, "Active learning failed", msg)

    def _add_selected(self):
        selected = self.results_list.selectedItems()
        if not selected:
            return
        indices = []
        for item in selected:
            path = item.data(Qt.UserRole)
            if path in self._path_to_index:
                indices.append(self._path_to_index[path])
        if not indices:
            return
        if self.add_frames_callback:
            self.add_frames_callback(indices, "Active learning")
            return
        if hasattr(self.parent(), "_add_indices_to_labeling"):
            self.parent()._add_indices_to_labeling(indices, "Active learning")
