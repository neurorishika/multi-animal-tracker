"""Evaluation panel -- dataset analysis and quick model testing."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from ..models import DetectKitProject

logger = logging.getLogger(__name__)


class EvaluationPanel(QWidget):
    """Evaluate dataset quality and run quick model tests."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proj = None
        self._main_window = None
        self._build_ui()

    # ---- UI construction ----

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top section: Analyze
        self.btn_analyze = QPushButton("Analyze Dataset")
        self.btn_analyze.clicked.connect(self._analyze)
        layout.addWidget(self.btn_analyze)

        self.analysis_view = QTextEdit()
        self.analysis_view.setReadOnly(True)
        self.analysis_view.setPlaceholderText(
            "Run analysis to see dataset size statistics and compatibility warnings."
        )
        layout.addWidget(self.analysis_view)

        # Bottom section: Quick Test
        self.btn_quick_test = QPushButton("Quick Test\u2026")
        self.btn_quick_test.setEnabled(False)
        self.btn_quick_test.clicked.connect(self._quick_test)
        layout.addWidget(self.btn_quick_test)

    # ---- Public API ----

    def set_project(self, proj: DetectKitProject, main_window) -> None:
        self._proj = proj
        self._main_window = main_window

    def collect_state(self, proj: DetectKitProject) -> None:
        pass  # nothing to persist

    # ---- Handlers ----

    def _analyze(self) -> None:
        if self._main_window is None:
            return
        proj = self._main_window.project()
        sources = proj.sources
        if not sources:
            self.analysis_view.setPlainText("No dataset sources configured.")
            return

        try:
            from hydra_suite.training.dataset_inspector import (
                DatasetInspection,
                analyze_obb_sizes,
                format_size_analysis,
                inspect_obb_or_detect_dataset,
            )
        except ImportError:
            self.analysis_view.setPlainText(
                "Dataset inspector is not available. "
                "Please install training dependencies."
            )
            return

        # Inspect each source and merge
        merged = DatasetInspection(root_dir="(merged)")
        for src in sources:
            if not src.path:
                continue
            try:
                insp = inspect_obb_or_detect_dataset(src.path)
            except Exception as exc:
                logger.warning("Failed to inspect %s: %s", src.path, exc)
                continue
            for split_name, items in insp.splits.items():
                merged.splits.setdefault(split_name, []).extend(items)
            merged.class_names.update(insp.class_names)

        if not any(merged.splits.values()):
            self.analysis_view.setPlainText(
                "No valid dataset items found in the configured sources."
            )
            return

        # Compute size stats
        stats = analyze_obb_sizes(
            merged,
            pad_ratio=proj.crop_pad_ratio,
            min_crop_size_px=proj.min_crop_size_px,
            enforce_square=proj.enforce_square,
        )

        # Format for seq_crop_obb pipeline
        report_seq, warnings_seq = format_size_analysis(
            stats, training_imgsz=proj.imgsz_seq_crop_obb
        )
        # Format for obb_direct pipeline
        report_direct, warnings_direct = format_size_analysis(
            stats, training_imgsz=proj.imgsz_obb_direct
        )

        # Build combined report
        lines = []
        lines.append("=== Seq Crop OBB Pipeline ===")
        lines.append(f"(imgsz = {proj.imgsz_seq_crop_obb})")
        lines.append("")
        lines.append(report_seq)

        if warnings_seq:
            lines.append("")
            lines.append("WARNINGS:")
            for w in warnings_seq:
                lines.append(f"  \u26a0 {w}")

        lines.append("")
        lines.append("=== OBB Direct Pipeline ===")
        lines.append(f"(imgsz = {proj.imgsz_obb_direct})")
        lines.append("")
        lines.append(report_direct)

        if warnings_direct:
            lines.append("")
            lines.append("WARNINGS:")
            for w in warnings_direct:
                lines.append(f"  \u26a0 {w}")

        self.analysis_view.setPlainText("\n".join(lines))

        # Show prominent warning dialog if any
        all_warnings = warnings_seq + warnings_direct
        if all_warnings:
            QMessageBox.warning(
                self,
                "Dataset Analysis Warnings",
                "\n".join(all_warnings),
            )

    def _quick_test(self) -> None:
        try:
            from hydra_suite.tracker.gui.dialogs.model_test_dialog import (
                ModelTestDialog,
            )
        except ImportError:
            QMessageBox.information(
                self,
                "Not Available",
                "Model test dialog is not available.",
            )
            return

        # TODO: detect trained model path and pass it in.
        # For now, inform the user that a training run is needed first.
        _ = ModelTestDialog  # will be used once model path detection is wired
        QMessageBox.information(
            self,
            "Quick Test",
            "No trained model is available yet. Complete a training run first.",
        )
