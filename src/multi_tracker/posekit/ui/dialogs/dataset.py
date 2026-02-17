#!/usr/bin/env python3
"""Dataset splitting dialog."""

import gc
import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.extensions import (
    cluster_kfold_split,
    cluster_stratified_split,
    save_split_files,
)
from .utils import _load_dialog_settings, _save_dialog_settings

logger = logging.getLogger("pose_label.dialogs.dataset")


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
        self.tvt_widget = QGroupBox("Train/Val/Test split")
        tvt_layout = QFormLayout(self.tvt_widget)

        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.7)
        tvt_layout.addRow("Train fraction", self.train_spin)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.15)
        tvt_layout.addRow("Validation fraction", self.val_spin)

        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.15)
        tvt_layout.addRow("Test fraction", self.test_spin)

        content_layout.addWidget(self.tvt_widget)

        # K-Fold parameters
        self.kfold_widget = QGroupBox("K-fold split")
        kfold_layout = QFormLayout(self.kfold_widget)

        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_layout.addRow("Number of folds", self.kfold_spin)

        content_layout.addWidget(self.kfold_widget)
        self.kfold_widget.setVisible(False)

        # Common parameters
        common_group = QGroupBox("Common settings")
        common_layout = QFormLayout(common_group)

        self.min_per_cluster_spin = QSpinBox()
        self.min_per_cluster_spin.setRange(1, 100)
        self.min_per_cluster_spin.setValue(1)
        common_layout.addRow("Min frames per cluster", self.min_per_cluster_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        common_layout.addRow("Random seed", self.seed_spin)

        self.split_name_edit = QLineEdit("split")
        common_layout.addRow("Split name", self.split_name_edit)

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
        self.train_spin.setValue(
            float(settings.get("train_frac", self.train_spin.value()))
        )
        self.val_spin.setValue(float(settings.get("val_frac", self.val_spin.value())))
        self.test_spin.setValue(
            float(settings.get("test_frac", self.test_spin.value()))
        )
        self.kfold_spin.setValue(int(settings.get("kfold", self.kfold_spin.value())))
        self.min_per_cluster_spin.setValue(
            int(settings.get("min_per_cluster", self.min_per_cluster_spin.value()))
        )
        self.seed_spin.setValue(int(settings.get("seed", self.seed_spin.value())))
        self.split_name_edit.setText(
            settings.get("split_name", self.split_name_edit.text())
        )

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

    def closeEvent(self: object, event: object) -> object:
        """closeEvent method documentation."""
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
