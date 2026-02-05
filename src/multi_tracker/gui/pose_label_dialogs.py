#!/usr/bin/env python3
"""
Dialogs for PoseKit Labeler extensions.
"""

import csv
from pathlib import Path
from typing import List, Optional
import logging

from PySide6.QtCore import Qt, QSize, QThread
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
        self.btn_preview = QPushButton("Preview selection")
        self.btn_add = QPushButton("Add to labeling set")
        self.btn_save_csv = QPushButton("Save clusters CSV…")
        self.btn_close = QPushButton("Close")
        bottom.addWidget(self.btn_preview)
        bottom.addWidget(self.btn_add)
        bottom.addWidget(self.btn_save_csv)
        bottom.addStretch(1)
        bottom.addWidget(self.btn_close)
        layout.addLayout(bottom)

        self.btn_preview.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_save_csv.setEnabled(False)

        # wiring
        self.btn_close.clicked.connect(self.reject)
        self.btn_compute.clicked.connect(self._compute)
        self.btn_cancel.clicked.connect(self._cancel_compute)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_save_csv.clicked.connect(self._save_csv)

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

        cluster = cluster_embeddings_cosine(
            self._emb, k=k, method=cluster_method, seed=0
        )
        self._cluster = cluster

        picked = pick_frames_stratified(
            emb=self._emb,
            cluster_id=cluster,
            want_n=n,
            eligible_indices=self._eligible_indices,
            min_per_cluster=min_per,
            seed=0,
            strategy=strategy,
        )
        self.selected_indices = picked

        lines = []
        for idx in picked[:300]:  # avoid huge previews
            # find local pos to show cluster
            local = self._eligible_indices.index(idx)
            cid = int(cluster[local])
            lines.append(f"[c{cid:03d}] {self.image_paths[idx].name}")
        if len(picked) > 300:
            lines.append(f"... ({len(picked)-300} more)")
        self.preview.setPlainText("\n".join(lines))

    def _save_csv(self):
        if self._cluster is None or self._eligible_indices is None:
            QMessageBox.information(self, "No clusters", "Run Preview selection first.")
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
