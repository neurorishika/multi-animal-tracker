"""Qt dialog for building and training YOLO-OBB datasets from GUI sources.

This dialog coordinates conversion, validation, dataset merging, and training
invocation workflows used by the MAT application.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ...data.dataset_merge import (
    detect_dataset_layout,
    get_dataset_class_name,
    merge_datasets,
    rewrite_labels_to_single_class,
    update_dataset_class_name,
    validate_labels,
)
from ...integrations.xanylabeling_cli import HARD_CODED_CMD, convert_project
from ...utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


class BuildDatasetWorker(QThread):
    """Background worker that converts and merges selected dataset sources."""

    status_signal = Signal(int, str)
    log_signal = Signal(str)
    done_signal = Signal(bool, str)

    def __init__(
        self,
        sources,
        output_dir,
        class_name,
        split_cfg,
        seed,
        dedup,
        conda_env,
        rewrite_classes=False,
    ):
        super().__init__()
        self.sources = sources
        self.output_dir = output_dir
        self.class_name = class_name
        self.split_cfg = split_cfg
        self.seed = seed
        self.dedup = dedup
        self.conda_env = conda_env
        self.rewrite_classes = rewrite_classes
        self.converted_sources = []

    def run(self: object) -> object:
        """run method documentation."""
        try:
            converted = []
            # Convert X-AnyLabeling sources
            for src in self.sources:
                row = src.get("row", 0)
                if src["type"] != "xany":
                    converted.append(src)
                    continue
                self.status_signal.emit(row, "Converting")
                project_name = Path(src["path"]).name
                ok, log = convert_project(src["path"], self.output_dir, self.conda_env)
                self.log_signal.emit(log)
                if not ok:
                    self.status_signal.emit(row, "Failed")
                    self.done_signal.emit(
                        False, f"Conversion failed for {project_name}"
                    )
                    return
                self.status_signal.emit(row, "Converted")
                converted.append(
                    {"name": project_name, "path": src["path"], "row": row}
                )

            # Optionally rewrite classes to single class
            if self.rewrite_classes:
                for src in converted:
                    update_dataset_class_name(src["path"], self.class_name)
                    layout = detect_dataset_layout(src["path"])
                    for _, (_, lbl_dir) in layout.items():
                        rewrite_labels_to_single_class(lbl_dir, 0)

            # Validate datasets & class count
            for src in converted:
                row = src.get("row", 0)
                self.status_signal.emit(row, "Validating")
                layout = detect_dataset_layout(src["path"])
                class_name = get_dataset_class_name(src["path"])
                if class_name and class_name != self.class_name:
                    self.done_signal.emit(False, "CLASS_MISMATCH")
                    return
                for split, (_, lbl_dir) in layout.items():
                    class_ids, _ = validate_labels(lbl_dir)
                    if len(class_ids) > 1 or (
                        len(class_ids) == 1 and 0 not in class_ids
                    ):
                        self.done_signal.emit(False, "CLASS_MISMATCH")
                        return
                self.status_signal.emit(row, "Validated")

            # Merge datasets
            merged_dir = merge_datasets(
                sources=converted,
                output_dir=self.output_dir,
                class_name=self.class_name,
                split_cfg=self.split_cfg,
                seed=self.seed,
                dedup=self.dedup,
            )
            self.done_signal.emit(True, merged_dir)
        except Exception as e:
            self.log_signal.emit(str(e))
            self.done_signal.emit(False, str(e))


class TrainYoloDialog(QDialog):
    """Interactive workflow dialog for YOLO dataset build and training runs."""

    def __init__(self, parent=None, class_name="object", conda_envs=None):
        """Initialize dialog UI state and defaults for dataset/training controls."""
        super().__init__(parent)
        self.setWindowTitle("Train YOLO-OBB Model")
        self.resize(900, 700)
        self.class_name = class_name
        self.conda_envs = conda_envs or []
        self.merged_dataset_dir = None
        self.train_process = None
        self.pending_build_args = None

        self._build_ui()

    def _build_ui(self):
        outer_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)

        # Sources
        gb_sources = QGroupBox("Sources")
        v_sources = QVBoxLayout(gb_sources)
        self.table_sources = QTableWidget(0, 3)
        self.table_sources.setHorizontalHeaderLabels(["Type", "Path", "Status"])
        self.table_sources.horizontalHeader().setStretchLastSection(True)
        v_sources.addWidget(self.table_sources)

        btn_row = QHBoxLayout()
        self.btn_add_xany = QPushButton("Add X-AnyLabeling Projects…")
        self.btn_add_yolo = QPushButton("Add YOLO-OBB Datasets…")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_clear = QPushButton("Clear All")
        btn_row.addWidget(self.btn_add_xany)
        btn_row.addWidget(self.btn_add_yolo)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_clear)
        v_sources.addLayout(btn_row)

        # Conversion
        gb_conv = QGroupBox("X-AnyLabeling Conversion")
        v_conv = QVBoxLayout(gb_conv)
        h_env = QHBoxLayout()
        h_env.addWidget(QLabel("Conda Env:"))
        self.combo_env = QComboBox()
        self.combo_env.addItems(self.conda_envs or ["(none)"])
        h_env.addWidget(self.combo_env, 1)
        v_conv.addLayout(h_env)

        v_conv.addWidget(QLabel("Converter: X-AnyLabeling XLABEL → YOLO-OBB"))

        # Merge
        gb_merge = QGroupBox("Merge + Validation")
        f_merge = QVBoxLayout(gb_merge)
        h_out = QHBoxLayout()
        repo_root = Path(__file__).resolve().parents[3]
        self.repo_output_dir = repo_root / "training" / "YOLO-obb"
        self.line_output = QLineEdit(str(self.repo_output_dir))
        self.btn_output = QPushButton("Browse…")
        h_out.addWidget(QLabel("Output Dir:"))
        h_out.addWidget(self.line_output, 1)
        h_out.addWidget(self.btn_output)
        f_merge.addLayout(h_out)

        h_class = QHBoxLayout()
        self.line_class = QLineEdit(self.class_name)
        h_class.addWidget(QLabel("Class Name:"))
        h_class.addWidget(self.line_class)
        f_merge.addLayout(h_class)

        h_split = QHBoxLayout()
        self.spin_train = QDoubleSpinBox()
        self.spin_val = QDoubleSpinBox()
        for spin, val in [(self.spin_train, 0.8), (self.spin_val, 0.2)]:
            spin.setRange(0.0, 1.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(val)
        h_split.addWidget(QLabel("Train"))
        h_split.addWidget(self.spin_train)
        h_split.addWidget(QLabel("Val"))
        h_split.addWidget(self.spin_val)
        f_merge.addLayout(h_split)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(42)
        f_merge.addWidget(QLabel("Seed:"))
        f_merge.addWidget(self.spin_seed)

        self.chk_dedup = QCheckBox("Deduplicate images by content")
        self.chk_dedup.setChecked(True)
        f_merge.addWidget(self.chk_dedup)

        self.btn_build = QPushButton("Build Combined Dataset")
        f_merge.addWidget(self.btn_build)

        # Training
        gb_train = QGroupBox("Training")
        v_train = QVBoxLayout(gb_train)
        h_weights = QHBoxLayout()
        self.combo_weights = QComboBox()
        self.combo_weights.setEditable(True)
        self.combo_weights.addItems(self._build_model_options())
        self.combo_weights.addItem("Select weights file…")
        self.line_weights = QLineEdit()
        self.line_weights.setPlaceholderText("weights file path")
        self.btn_weights = QPushButton("Browse…")
        h_weights.addWidget(self.combo_weights)
        h_weights.addWidget(self.line_weights, 1)
        h_weights.addWidget(self.btn_weights)
        v_train.addLayout(h_weights)

        # Hyperparams
        h_params = QHBoxLayout()
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(320, 2048)
        self.spin_imgsz.setValue(640)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(1e-5, 1.0)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 200)
        self.spin_patience.setValue(30)
        self.combo_device = QComboBox()
        self.combo_device.addItems(self._build_device_options())
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        self.chk_cache = QCheckBox("Cache")
        h_params.addWidget(QLabel("epochs"))
        h_params.addWidget(self.spin_epochs)
        h_params.addWidget(QLabel("imgsz"))
        h_params.addWidget(self.spin_imgsz)
        h_params.addWidget(QLabel("batch"))
        h_params.addWidget(self.spin_batch)
        h_params.addWidget(QLabel("lr0"))
        h_params.addWidget(self.spin_lr0)
        h_params.addWidget(QLabel("patience"))
        h_params.addWidget(self.spin_patience)
        v_train.addLayout(h_params)
        h_params2 = QHBoxLayout()
        h_params2.addWidget(QLabel("device"))
        h_params2.addWidget(self.combo_device)
        h_params2.addWidget(QLabel("workers"))
        h_params2.addWidget(self.spin_workers)
        h_params2.addWidget(self.chk_cache)
        v_train.addLayout(h_params2)

        h_aug = QHBoxLayout()
        self.chk_default_aug = QCheckBox("Use default augmentations")
        self.chk_default_aug.setChecked(True)
        self.line_aug = QLineEdit()
        self.line_aug.setPlaceholderText(
            "Custom aug args e.g. degrees=10 translate=0.1 scale=0.5"
        )
        self.line_aug.setEnabled(False)
        h_aug.addWidget(self.chk_default_aug)
        h_aug.addWidget(self.line_aug, 1)
        v_train.addLayout(h_aug)

        h_train_btn = QHBoxLayout()
        self.btn_start_train = QPushButton("Start Training")
        self.btn_stop_train = QPushButton("Stop Training")
        self.btn_stop_train.setEnabled(False)
        h_train_btn.addWidget(self.btn_start_train)
        h_train_btn.addWidget(self.btn_stop_train)
        v_train.addLayout(h_train_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        v_train.addWidget(self.progress)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        v_train.addWidget(self.log_view)

        # Wire up
        self.btn_add_xany.clicked.connect(self._add_xany)
        self.btn_add_yolo.clicked.connect(self._add_yolo)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_output.clicked.connect(self._choose_output)
        self.btn_build.clicked.connect(self._build_dataset)
        self.btn_weights.clicked.connect(self._choose_weights)
        self.btn_start_train.clicked.connect(self._start_training)
        self.btn_stop_train.clicked.connect(self._stop_training)
        self.combo_weights.currentIndexChanged.connect(self._on_weights_choice)
        self.chk_default_aug.toggled.connect(self._on_aug_toggle)

        layout.addWidget(gb_sources)
        layout.addWidget(gb_conv)
        layout.addWidget(gb_merge)
        layout.addWidget(gb_train)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll)
        self._on_weights_choice()
        self._on_aug_toggle(self.chk_default_aug.isChecked())

    def _add_xany(self):
        for d in self._get_multiple_dirs("Select X-AnyLabeling Projects"):
            self._add_source("xany", d)

    def _add_yolo(self):
        for d in self._get_multiple_dirs("Select YOLO-OBB Datasets"):
            self._add_source("yolo", d)

    def _add_source(self, typ, path):
        row = self.table_sources.rowCount()
        self.table_sources.insertRow(row)
        self.table_sources.setItem(
            row, 0, QTableWidgetItem("X-AnyLabeling" if typ == "xany" else "YOLO-OBB")
        )
        self.table_sources.setItem(row, 1, QTableWidgetItem(path))
        self.table_sources.setItem(row, 2, QTableWidgetItem("Pending"))
        self.table_sources.item(row, 0).setData(Qt.UserRole, typ)

    def _remove_selected(self):
        rows = {idx.row() for idx in self.table_sources.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table_sources.removeRow(row)

    def _clear_all(self):
        self.table_sources.setRowCount(0)

    def _choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.line_output.setText(directory)

    def _choose_weights(self):
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Weights", "", "Weights (*.pt)"
        )
        if fp:
            self.line_weights.setText(fp)

    def _build_dataset(self):
        if self.table_sources.rowCount() == 0:
            QMessageBox.warning(self, "No sources", "Please add at least one source.")
            return
        output_dir = self.line_output.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "Output directory", "Please select an output directory."
            )
            return
        class_name = self.line_class.text().strip() or "object"
        split_cfg = {
            "train": self.spin_train.value(),
            "val": self.spin_val.value(),
            "test": 0.0,
        }
        sources = []
        for row in range(self.table_sources.rowCount()):
            typ = self.table_sources.item(row, 0).data(Qt.UserRole)
            path = self.table_sources.item(row, 1).text()
            name = Path(path).name
            sources.append({"type": typ, "path": path, "name": name, "row": row})

        conda_env = self.combo_env.currentText()
        if conda_env == "(none)":
            conda_env = None

        self.pending_build_args = dict(
            sources=sources,
            output_dir=output_dir,
            class_name=class_name,
            split_cfg=split_cfg,
            seed=self.spin_seed.value(),
            dedup=self.chk_dedup.isChecked(),
            conda_env=conda_env,
            rewrite_classes=False,
        )
        self.worker = BuildDatasetWorker(**self.pending_build_args)
        self.worker.status_signal.connect(self._update_status)
        self.worker.log_signal.connect(self._append_log)
        self.worker.done_signal.connect(self._build_done)
        self.worker.start()

    def _update_status(self, row, status):
        self.table_sources.setItem(row, 2, QTableWidgetItem(status))

    def _append_log(self, text):
        self.log_view.append(text)

    def _build_done(self, ok, message):
        if ok and message != "CLASS_MISMATCH":
            self.merged_dataset_dir = message
            QMessageBox.information(self, "Dataset Ready", f"Merged dataset: {message}")
            return
        if message == "CLASS_MISMATCH":
            resp = QMessageBox.question(
                self,
                "Class Mismatch",
                "One or more datasets have multiple classes or mismatched class IDs.\n"
                "Rewrite all labels to the Active Learning class?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                if self.pending_build_args:
                    self.pending_build_args["rewrite_classes"] = True
                    self.worker = BuildDatasetWorker(**self.pending_build_args)
                    self.worker.status_signal.connect(self._update_status)
                    self.worker.log_signal.connect(self._append_log)
                    self.worker.done_signal.connect(self._build_done)
                    self.worker.start()
            return
        QMessageBox.critical(self, "Build Failed", message)

    def _start_training(self):
        if not self.merged_dataset_dir:
            QMessageBox.warning(self, "No dataset", "Build a dataset first.")
            return
        weights = self._get_weights_path()
        if not weights:
            QMessageBox.warning(
                self, "Weights", "Select weights or choose auto-download."
            )
            return

        yaml_path = os.path.join(self.merged_dataset_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            QMessageBox.warning(
                self, "Dataset", "dataset.yaml not found in merged dataset."
            )
            return

        output_dir = os.path.join(self.line_output.text().strip(), "training_runs")
        os.makedirs(output_dir, exist_ok=True)
        run_name = os.path.basename(self.merged_dataset_dir)

        cmd = self._build_train_command(
            yaml_path=yaml_path,
            weights=weights,
            output_dir=output_dir,
            run_name=run_name,
        )
        if self.chk_cache.isChecked():
            cmd.append("cache=True")
        if not self.chk_default_aug.isChecked():
            custom_aug = self.line_aug.text().strip()
            if not custom_aug:
                QMessageBox.warning(
                    self,
                    "Augmentations",
                    "Custom augmentations enabled, but no args provided.",
                )
                return
            cmd.extend(custom_aug.split())

        self.log_view.append("Running: " + " ".join(cmd))

        self.train_process = QProcess(self)
        self.train_process.setProcessChannelMode(QProcess.MergedChannels)
        self.train_process.readyReadStandardOutput.connect(self._read_train_log)
        self.train_process.finished.connect(self._train_finished)
        self.train_process.start(cmd[0], cmd[1:])

        self.progress.setVisible(True)
        self.btn_start_train.setEnabled(False)
        self.btn_stop_train.setEnabled(True)

    def _get_weights_path(self):
        choice = self.combo_weights.currentText().strip()
        if choice == "Select weights file…":
            return self.line_weights.text().strip()
        return choice

    def _read_train_log(self):
        if not self.train_process:
            return
        text = (
            self.train_process.readAllStandardOutput()
            .data()
            .decode("utf-8", errors="ignore")
        )
        if text:
            self.log_view.append(text)

    def _train_finished(self):
        self.progress.setVisible(False)
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.log_view.append("Training finished.")

    def _stop_training(self):
        if self.train_process:
            self.train_process.terminate()
            if not self.train_process.waitForFinished(3000):
                self.train_process.kill()

    def _get_multiple_dirs(self, title):
        dialog = QFileDialog(self, title)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        # Enable multi-selection in the view
        for view in dialog.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dialog.exec() != QFileDialog.Accepted:
            return []
        return dialog.selectedFiles()

    def _on_weights_choice(self):
        is_custom = self.combo_weights.currentText().strip() == "Select weights file…"
        self.line_weights.setEnabled(is_custom)
        self.btn_weights.setEnabled(is_custom)

    def _on_aug_toggle(self, checked):
        # Checked = use defaults, disable custom input.
        self.line_aug.setEnabled(not checked)

    def _build_model_options(self):
        options = [
            "yolo26n-obb.pt",
            "yolo26s-obb.pt",
            "yolo26m-obb.pt",
            "yolo26l-obb.pt",
            "yolo26x-obb.pt",
        ]
        # Include any local .pt files in repo models/YOLO-obb
        try:
            for fp in (self.repo_output_dir).glob("*.pt"):
                options.append(str(fp))
        except Exception:
            pass
        return options

    def _build_train_command(self, yaml_path, weights, output_dir, run_name):
        common = [
            "obb",
            "train",
            f"data={yaml_path}",
            f"model={weights}",
            f"epochs={self.spin_epochs.value()}",
            f"imgsz={self.spin_imgsz.value()}",
            f"batch={self.spin_batch.value()}",
            f"lr0={self.spin_lr0.value()}",
            f"patience={self.spin_patience.value()}",
            f"device={self.combo_device.currentText()}",
            f"workers={self.spin_workers.value()}",
            f"project={output_dir}",
            f"name={run_name}",
        ]
        if self.chk_cache.isChecked():
            common.append("cache=True")

        # Prefer `yolo` executable if available, else ultralytics.cli module
        yolo_exe = shutil.which("yolo")
        if yolo_exe:
            return [yolo_exe] + common
        return [sys.executable, "-m", "ultralytics.cli"] + common

    def _build_device_options(self):
        info = get_device_info()
        options = ["auto", "cpu"]
        if info.get("torch_cuda_available"):
            options.append("cuda")
            count = info.get("torch_cuda_device_count", 0)
            if count:
                for i in range(count):
                    options.append(f"cuda:{i}")
                if count > 1:
                    options.append(",".join([f"cuda:{i}" for i in range(count)]))
        if info.get("mps_available"):
            options.append("mps")
        if info.get("rocm_available"):
            options.append("rocm")
        return options
