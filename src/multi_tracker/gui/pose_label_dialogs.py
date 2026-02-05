#!/usr/bin/env python3
"""
Dialogs for PoseKit Labeler extensions.
"""

import csv
from pathlib import Path
from typing import List, Optional
import logging
import numpy as np

from PySide6.QtCore import Qt, QSize, QThread, QObject, Signal, QUrl

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False
    QWebEngineView = None

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
)

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
    )

logger = logging.getLogger("pose_label.dialogs")


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

        # Info
        info_label = QLabel(
            f"Total frames: {len(image_paths)}\n"
            + (
                f"Clusters detected: {len(set(cluster_ids))}"
                if cluster_ids
                else "No cluster info available"
            )
        )
        layout.addWidget(info_label)

        # Split mode
        mode_group = QGroupBox("Split Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Train/Val/Test", "K-Fold Cross-Validation"])
        mode_layout.addWidget(self.mode_combo)

        layout.addWidget(mode_group)

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

        layout.addWidget(self.tvt_widget)

        # K-Fold parameters
        self.kfold_widget = QGroupBox("K-Fold Parameters")
        kfold_layout = QFormLayout(self.kfold_widget)

        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_layout.addRow("Number of folds:", self.kfold_spin)

        layout.addWidget(self.kfold_widget)
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

        layout.addWidget(common_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Split")
        self.btn_close = QPushButton("Close")
        btn_layout.addWidget(self.btn_generate)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

        # Wiring
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_generate.clicked.connect(self._generate_split)
        self.btn_close.clicked.connect(self.reject)

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
        scope_row.addWidget(self.cb_scope, 1)
        self.cb_exclude_in_labeling = QCheckBox("Exclude already in labeling set")
        self.cb_exclude_in_labeling.setChecked(True)
        scope_row.addWidget(self.cb_exclude_in_labeling)
        layout.addLayout(scope_row)

        # --- embeddings
        emb_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(
            [
                "timm/vit_base_patch14_dinov2.lvd142m",
                "timm/vit_small_patch14_dinov2.lvd142m",
                "timm/resnet50.a1_in1k",
                "timm/mobilenetv3_small_100.lamb_in1k",
            ]
        )
        emb_form.addRow("Model:", self.model_combo)

        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(["auto", "cpu", "cuda", "mps"])
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        dev_row.addWidget(self.batch_spin)
        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        dev_row.addWidget(self.max_side_spin)
        emb_form.addRow("Device / batch:", dev_row)

        self.cb_use_enhance = QCheckBox("Use Enhance (CLAHE+unsharp)")
        self.cb_use_enhance.setChecked(bool(self.project.enhance_enabled))
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
        sel_row.addWidget(self.n_spin)

        sel_row.addWidget(QLabel("Clusters:"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 5000)
        self.k_spin.setValue(20)
        sel_row.addWidget(self.k_spin)

        sel_row.addWidget(QLabel("Min/cluster:"))
        self.min_per_spin = QSpinBox()
        self.min_per_spin.setRange(1, 1000)
        self.min_per_spin.setValue(1)
        sel_row.addWidget(self.min_per_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["centroid_then_diverse", "centroid"])
        sel_row.addWidget(QLabel("Strategy:"))
        sel_row.addWidget(self.strategy_combo, 1)

        layout.addLayout(sel_row)

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
        self.btn_explorer.setEnabled(True)  # Enable explorer now!

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None

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

        # Placeholder for Bokeh visualization
        self.viz_label = QLabel(
            "Bokeh visualization will appear in your browser after UMAP computation."
        )
        self.viz_label.setWordWrap(True)
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setMinimumHeight(400)
        self.viz_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 1px solid #ccc; padding: 20px; }"
        )
        layout.addWidget(self.viz_label, 1)

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

        # Add web view for Bokeh visualization (if available)
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView()
            layout.addWidget(self.web_view, 2)  # Give it more stretch
        else:
            self.web_view = None
            layout.addWidget(
                QLabel("QWebEngineView not available. Install PySide6-WebEngine.")
            )

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

            self.umap_progress.setText(f"✓ Visualization ready!")
            self.viz_label.setText(
                f"Use Box Select tool to select frames.\\n\\nHTML: {self.bokeh_html_path}"
            )

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
        """Load Bokeh HTML in web view or open in browser."""
        if not self.bokeh_html_path:
            return

        if HAS_WEBENGINE and self.web_view:
            # Load in QWebEngineView
            self.web_view.setUrl(QUrl.fromLocalFile(self.bokeh_html_path))
        else:
            # Fallback to external browser
            import webbrowser

            webbrowser.open("file://" + self.bokeh_html_path)

    def _refresh_viz(self):
        """Refresh the visualization."""
        self._load_visualization()


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
