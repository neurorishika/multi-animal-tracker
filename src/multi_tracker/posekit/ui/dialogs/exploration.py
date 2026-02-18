#!/usr/bin/env python3
"""
Exploration dialogs: Smart Select, Embedding Explorer, Frame Metadata.
"""

import csv
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, QSize, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from ...core.extensions import (
    EmbeddingWorker,
    cluster_embeddings_cosine,
    pick_frames_stratified,
)
from .utils import _load_dialog_settings, _save_dialog_settings, get_available_devices

logger = logging.getLogger("pose_label.dialogs.exploration")


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
        scope_row.addWidget(QLabel("Scope"))
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
                "timm/vit_base_patch14_dinov2.lvd142m",
                "timm/vit_small_patch14_dinov2.lvd142m",
                "timm/vit_large_patch14_dinov2.lvd142m",
                "timm/vit_giant_patch14_dinov2.lvd142m",
                # CLIP models
                "timm/vit_base_patch32_clip_224.openai",
                "timm/vit_bigG_14_clip_224.laion400M_e32",
                "timm/convnext_base_w_clip.laion2b_s29B_b131k_ft_in1k",
                # ResNet models
                "timm/resnet50.a1_in1k",
                "timm/resnet18.a1_in1k",
                "timm/resnet101.a1_in1k",
                # EfficientNet models
                "timm/efficientnet_b0.ra_in1k",
                "timm/efficientnet_b3.ra2_in1k",
                "timm/efficientnet_b5.sw_in12k_ft_in1k",
                # MobileNet models
                "timm/mobilenetv3_small_100.lamb_in1k",
                "timm/mobilenetv3_large_100.ra_in1k",
                # ConvNeXt models
                "timm/convnext_tiny.fb_in1k",
                "timm/convnext_base.fb_in1k",
            ]
        )
        emb_form.addRow("Embedding model", self.model_combo)

        dev_row = QHBoxLayout()
        self.dev_combo = QComboBox()
        self.dev_combo.addItems(get_available_devices())
        dev_row.addWidget(self.dev_combo)
        dev_row.addWidget(QLabel("Batch size"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(32)
        dev_row.addWidget(self.batch_spin)
        self.cb_auto_batch = QCheckBox("Auto")
        self.cb_auto_batch.setChecked(True)
        dev_row.addWidget(self.cb_auto_batch)

        dev_row.addWidget(QLabel("Max side"))
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(0, 2048)
        self.max_side_spin.setValue(512)
        dev_row.addWidget(self.max_side_spin)
        emb_form.addRow("Device and batch size", dev_row)

        self.cb_use_enhance = QCheckBox("Enhance images (CLAHE + unsharp)")
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
        sel_row.addWidget(QLabel("Frames to add"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 50000)
        self.n_spin.setValue(50)
        sel_row.addWidget(self.n_spin)

        sel_row.addWidget(QLabel("Clusters"))
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 5000)
        self.k_spin.setValue(20)
        sel_row.addWidget(self.k_spin)

        sel_row.addWidget(QLabel("Min per cluster"))
        self.min_per_spin = QSpinBox()
        self.min_per_spin.setRange(1, 1000)
        self.min_per_spin.setValue(1)
        sel_row.addWidget(self.min_per_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["centroid_then_diverse", "centroid"])
        sel_row.addWidget(QLabel("Strategy"))
        sel_row.addWidget(self.strategy_combo, 1)

        layout.addLayout(sel_row)

        self.k_spin.valueChanged.connect(self._update_min_frames)
        self.min_per_spin.valueChanged.connect(self._update_min_frames)

        # --- Sampling options
        opt_row = QHBoxLayout()

        self.cb_filter_duplicates = QCheckBox("Filter near-duplicates")
        opt_row.addWidget(self.cb_filter_duplicates)

        opt_row.addWidget(QLabel("Threshold"))
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
        cluster_row.addWidget(QLabel("Which clustering method should be used?"))
        self.cluster_method_combo = QComboBox()
        self.cluster_method_combo.addItems(["hierarchical", "linkage"])
        self.cluster_method_combo.setCurrentIndex(0)
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
        self.btn_close.clicked.connect(self._on_close)
        self.btn_add.clicked.connect(self._on_add)
        self.btn_compute.clicked.connect(self._compute)
        self.btn_cancel.clicked.connect(self._cancel_compute)
        self.btn_preview.clicked.connect(self._preview)
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_explorer.clicked.connect(self._open_explorer)

        self.selected_indices: List[int] = []
        self._did_add = False

        self._thread = None
        self._worker = None

        self._update_min_frames()
        self._apply_settings()

    def _on_add(self):
        self._did_add = True
        self.accept()

    def _on_close(self):
        self.selected_indices = []
        self._did_add = False
        self.reject()

    def _apply_settings(self):
        settings = _load_dialog_settings("smart_select")
        if not settings:
            return
        self.cb_scope.setCurrentText(settings.get("scope", self.cb_scope.currentText()))
        self.cb_exclude_in_labeling.setChecked(
            bool(
                settings.get(
                    "exclude_labeling", self.cb_exclude_in_labeling.isChecked()
                )
            )
        )
        self.model_combo.setCurrentText(
            settings.get("model", self.model_combo.currentText())
        )
        self.dev_combo.setCurrentText(
            settings.get("device", self.dev_combo.currentText())
        )
        self.batch_spin.setValue(int(settings.get("batch", self.batch_spin.value())))
        self.cb_auto_batch.setChecked(
            bool(settings.get("auto_batch", self.cb_auto_batch.isChecked()))
        )
        self.max_side_spin.setValue(
            int(settings.get("max_side", self.max_side_spin.value()))
        )
        self.cb_use_enhance.setChecked(
            bool(settings.get("use_enhance", self.cb_use_enhance.isChecked()))
        )
        self.n_spin.setValue(int(settings.get("n", self.n_spin.value())))
        self.k_spin.setValue(int(settings.get("k", self.k_spin.value())))
        self.min_per_spin.setValue(
            int(settings.get("min_per", self.min_per_spin.value()))
        )
        self.strategy_combo.setCurrentText(
            settings.get("strategy", self.strategy_combo.currentText())
        )
        self.cb_filter_duplicates.setChecked(
            bool(settings.get("filter_dups", self.cb_filter_duplicates.isChecked()))
        )
        self.dup_threshold_spin.setValue(
            float(settings.get("dup_thresh", self.dup_threshold_spin.value()))
        )
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
                "auto_batch": bool(self.cb_auto_batch.isChecked()),
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

    def closeEvent(self: object, event: object) -> object:
        self._save_settings()
        super().closeEvent(event)

    def _build_eligible_indices(self) -> List[int]:
        scope = self.cb_scope.currentText()
        if scope == "All frames":
            idxs = list(range(len(self.image_paths)))
        elif scope == "Labeling set":
            idxs = list(getattr(self.parent(), "labeling_frames", set()))
        else:
            # Unlabeled only
            idxs = [
                i for i, p in enumerate(self.image_paths) if not self.is_labeled_fn(p)
            ]
        if self.cb_exclude_in_labeling.isChecked():
            in_labeling = set(getattr(self.parent(), "labeling_frames", set()))
            idxs = [i for i in idxs if i not in in_labeling]
        return sorted(idxs)

    def _update_min_frames(self):
        min_frames = self.k_spin.value() * self.min_per_spin.value()
        self.n_spin.setMinimum(min_frames)
        if self.n_spin.value() < min_frames:
            self.n_spin.setValue(min_frames)

    def _compute(self):
        eligible = self._build_eligible_indices()
        if not eligible:
            QMessageBox.information(
                self, "No frames", "No eligible frames in this scope."
            )
            return

        cache_dir = self.project.out_root / "posekit" / "embeddings"

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

        n_frames = len(self._eligible_indices)
        if cluster_method == "hierarchical" and n_frames > 2500:
            reply = QMessageBox.warning(
                self,
                "Large Dataset Warning",
                f"You have {n_frames} frames selected.\n\n"
                "Hierarchical clustering may be slow or memory-intensive for large datasets.\n\n"
                "Would you like to switch to 'linkage' method instead?",
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
        self._autosave_clusters()
        self.btn_explorer.setEnabled(True)
        self.lbl_status.setText("Clusters saved. Use Export → Split method to apply.")

        picked = pick_frames_stratified(
            emb=self._emb,
            cluster_id=cluster,
            want_n=n,
            eligible_indices=self._eligible_indices,
            min_per_cluster=min_per,
            seed=0,
            strategy=strategy,
        )

        if self.cb_filter_duplicates.isChecked():
            try:
                # Try simple cosine similarity filtering if imported
                from ...core.extensions import filter_near_duplicates

                threshold = self.dup_threshold_spin.value()
                embeddings_picked = self._emb[
                    [self._eligible_indices.index(i) for i in picked]
                ]
                picked = filter_near_duplicates(
                    embeddings_picked,
                    list(range(len(picked))),
                    threshold=threshold,
                )
                picked = [picked[i] for i in picked]
            except Exception as e:
                logger.warning(f"Duplicate filtering failed: {e}")

        if self.cb_prefer_unlabeled.isChecked() and self.is_labeled_fn:
            labeled = [i for i in picked if self.is_labeled_fn(self.image_paths[i])]
            unlabeled = [
                i for i in picked if not self.is_labeled_fn(self.image_paths[i])
            ]
            picked = unlabeled + labeled[: max(0, n - len(unlabeled))]

        self.selected_indices = picked[:n]

        lines = []
        for idx in self.selected_indices[:300]:
            local = self._eligible_indices.index(idx)
            cid = int(cluster[local])
            labeled_str = (
                " [L]"
                if self.is_labeled_fn and self.is_labeled_fn(self.image_paths[idx])
                else ""
            )
            lines.append(f"[c{cid:03d}]{labeled_str} {self.image_paths[idx].name}")
        if len(self.selected_indices) > 300:
            lines.append(f"... ({len(self.selected_indices) - 300} more)")
        self.preview.setPlainText("\n".join(lines))

    def _open_explorer(self):
        if self._emb is None:
            QMessageBox.warning(self, "No Embeddings", "Compute embeddings first.")
            return

        dialog = EmbeddingExplorerDialog(
            embeddings=self._emb,
            image_paths=[self.image_paths[i] for i in self._eligible_indices],
            cluster_ids=self._cluster,
            is_labeled_fn=self.is_labeled_fn if self.is_labeled_fn else lambda p: False,
            parent=self,
        )

        if dialog.exec_() == QDialog.Accepted:
            explorer_selected = dialog.selected_indices
            if explorer_selected:
                global_selected = [self._eligible_indices[i] for i in explorer_selected]
                existing = set(self.selected_indices)
                for idx in global_selected:
                    if idx not in existing:
                        self.selected_indices.append(idx)
                        existing.add(idx)

                # Update preview... (simplified for brevity)
                QMessageBox.information(
                    self,
                    "Added from Explorer",
                    f"Added {len(explorer_selected)} frames.",
                )

    def _save_csv(self):
        if self._cluster is None:
            QMessageBox.information(self, "No clusters", "Run Clustering first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cluster assignments",
            str(
                (self.project.out_root / "posekit" / "clusters").resolve()
                / "clusters.csv"
            ),
            "CSV (*.csv)",
        )
        if not path:
            return
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "cluster_id"])
            for local_pos, idx in enumerate(self._eligible_indices):
                w.writerow([str(self.image_paths[idx]), int(self._cluster[local_pos])])
        QMessageBox.information(self, "Saved", f"Wrote cluster CSV:\n{out}")

    def _autosave_clusters(self):
        if self._cluster is None:
            return
        out = self.project.out_root / "posekit" / "clusters" / "clusters.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "cluster_id"])
                for local_pos, idx in enumerate(self._eligible_indices):
                    w.writerow(
                        [str(self.image_paths[idx]), int(self._cluster[local_pos])]
                    )
        except Exception:
            pass


class EmbeddingExplorerDialog(QDialog):
    """Interactive UMAP visualization with Bokeh."""

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
        self.selected_indices: List[int] = []
        self.bokeh_html_path = None
        self._umap_thread = None
        self._umap_worker = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("UMAP neighbors"))
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

        self.umap_progress = QLabel("UMAP not computed yet.")
        layout.addWidget(self.umap_progress)

        info_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh Visualization")
        self.btn_refresh.clicked.connect(self._refresh_viz)
        self.btn_refresh.setEnabled(False)
        info_layout.addWidget(self.btn_refresh)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        info_layout.addWidget(btn_close)
        layout.addLayout(info_layout)

    def _compute_umap(self):
        self.btn_compute_umap.setEnabled(False)
        self.btn_cancel_umap.setEnabled(True)
        self.umap_progress.setText("Computing UMAP projection...")
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
        if self._umap_worker:
            self._umap_worker.cancel()

    def _on_umap_progress(self, msg: str):
        self.umap_progress.setText(msg)

    def _on_umap_finished(self, projection: np.ndarray):
        self._cleanup_umap_thread()
        self.umap_projection = projection
        self.umap_progress.setText(f"UMAP complete! {projection.shape[0]} points.")
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        self.btn_refresh.setEnabled(True)
        self._generate_bokeh_viz()

    def _on_umap_failed(self, error_msg: str):
        self._cleanup_umap_thread()
        self.umap_progress.setText(f"UMAP failed: {error_msg}")
        self.btn_compute_umap.setEnabled(True)
        self.btn_cancel_umap.setEnabled(False)
        QMessageBox.critical(self, "UMAP Error", str(error_msg))

    def _cleanup_umap_thread(self):
        if self._umap_thread:
            self._umap_thread.quit()
            self._umap_thread.wait(2000)
        self._umap_thread = None
        self._umap_worker = None

    def _generate_bokeh_viz(self):
        if self.umap_projection is None:
            return
        try:
            from bokeh.models import ColumnDataSource, HoverTool
            from bokeh.plotting import figure, output_file, save

            self.umap_progress.setText("Generating Bokeh visualization...")

            # Simplified generation for brevity
            image_names = [p.name for p in self.image_paths]
            source = ColumnDataSource(
                data=dict(
                    x=self.umap_projection[:, 0],
                    y=self.umap_projection[:, 1],
                    names=image_names,
                    indices=list(range(len(self.image_paths))),
                )
            )

            p = figure(
                width=900,
                height=700,
                title="UMAP Embedding Space",
                tools="pan,wheel_zoom,box_select,reset,save",
            )
            p.circle("x", "y", size=6, alpha=0.6, source=source)
            p.add_tools(HoverTool(tooltips=[("Name", "@names")]))

            if self.bokeh_html_path is None:
                fd, self.bokeh_html_path = tempfile.mkstemp(
                    suffix=".html", prefix="umap_"
                )
                import os

                os.close(fd)

            output_file(self.bokeh_html_path, title="Embedding Explorer")
            save(p)
            self._load_visualization()
            self.umap_progress.setText(f"Visualization ready: {self.bokeh_html_path}")

        except Exception as e:
            QMessageBox.critical(self, "Viz Error", str(e))

    def _load_visualization(self):
        if self.bokeh_html_path:
            import webbrowser

            webbrowser.open("file://" + self.bokeh_html_path)

    def _refresh_viz(self):
        self._load_visualization()

    def closeEvent(self, event):
        self.embeddings = None
        super().closeEvent(event)


class UMAPWorker(QObject):
    """Background worker for UMAP."""

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
            self.progress.emit("Running UMAP...")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=0.1,
                metric="cosine",
                random_state=self.random_state,
            )
            projection = reducer.fit_transform(self.embeddings)
            if self._cancelled:
                return
            self.finished.emit(projection.astype(np.float32))
        except Exception as e:
            self.failed.emit(str(e))
