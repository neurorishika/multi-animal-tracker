"""Role-aware MAT Training Center dialog for multi-model YOLO workflows."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
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

from multi_tracker.training import (
    PublishPolicy,
    SourceDataset,
    SplitConfig,
    TrainingHyperParams,
    TrainingOrchestrator,
    TrainingRole,
    TrainingRunSpec,
)
from multi_tracker.training.validation import format_validation_report
from multi_tracker.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


class RoleTrainingWorker(QThread):
    """Run selected role trainings sequentially in a background thread."""

    log_signal = Signal(str)
    role_started = Signal(str)
    role_finished = Signal(str, bool, str)
    progress_signal = Signal(str, int, int)
    done_signal = Signal(list)

    def __init__(self, orchestrator, role_entries):
        super().__init__()
        self.orchestrator = orchestrator
        self.role_entries = role_entries
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _should_cancel(self) -> bool:
        return bool(self._cancel)

    def run(self):
        results = []
        parent_run = ""
        for entry in self.role_entries:
            if self._cancel:
                break
            role = entry["role"]
            spec = entry["spec"]
            publish_meta = entry["publish_meta"]
            self.role_started.emit(role.value)

            def _log(msg: str, _role=role):
                self.log_signal.emit(f"[{_role.value}] {msg}")

            def _prog(cur: int, total: int, _role=role):
                self.progress_signal.emit(_role.value, int(cur), int(total))

            try:
                result = self.orchestrator.run_role_training(
                    spec,
                    parent_run_id=parent_run,
                    publish_metadata=publish_meta,
                    log_cb=_log,
                    progress_cb=_prog,
                    should_cancel=self._should_cancel,
                )
            except Exception as exc:
                result = {
                    "run_id": "",
                    "success": False,
                    "error": str(exc),
                    "published_registry_key": "",
                    "published_model_path": "",
                }

            result["role"] = role.value
            results.append(result)
            ok = bool(result.get("success", False))
            msg = (
                f"run_id={result.get('run_id', '')}"
                if ok
                else (
                    result.get("error") or f"exit={result.get('exit_code', 'unknown')}"
                )
            )
            self.role_finished.emit(role.value, ok, msg)
            if result.get("run_id"):
                parent_run = str(result["run_id"])

        self.done_signal.emit(results)


class TrainYoloDialog(QDialog):
    """MAT role-aware training center replacing legacy OBB-only trainer."""

    def __init__(self, parent=None, class_name="object", conda_envs=None):
        super().__init__(parent)
        self.setWindowTitle("Training Center (YOLO Multi-Role)")
        self.resize(1100, 820)

        self.class_name = str(class_name or "object")
        self.conda_envs = conda_envs or []
        self.worker = None

        from multi_tracker.paths import get_training_workspace_dir

        self.workspace_default = get_training_workspace_dir("YOLO")
        self.orchestrator = TrainingOrchestrator(self.workspace_default)

        self.role_dataset_dirs: dict[str, str] = {}

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_sources_group())
        layout.addWidget(self._build_roles_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_run_group())

        scroll.setWidget(content)
        outer.addWidget(scroll)

    def _build_sources_group(self):
        gb = QGroupBox("Step 1: Sources + Validation")
        v = QVBoxLayout(gb)

        self.table_sources = QTableWidget(0, 3)
        self.table_sources.setHorizontalHeaderLabels(["Type", "Path", "Status"])
        self.table_sources.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table_sources)

        row = QHBoxLayout()
        self.btn_add_obb = QPushButton("Add OBB Source Datasets…")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_clear = QPushButton("Clear All")
        self.btn_validate = QPushButton("Validate Sources")
        row.addWidget(self.btn_add_obb)
        row.addWidget(self.btn_remove)
        row.addWidget(self.btn_clear)
        row.addWidget(self.btn_validate)
        v.addLayout(row)

        self.validation_view = QTextEdit()
        self.validation_view.setReadOnly(True)
        self.validation_view.setPlaceholderText("Validation report appears here.")
        v.addWidget(self.validation_view)

        self.btn_add_obb.clicked.connect(self._add_obb_sources)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_sources)
        self.btn_validate.clicked.connect(self._validate_sources)

        return gb

    def _build_roles_group(self):
        gb = QGroupBox("Step 2: Role Selection")
        g = QGridLayout(gb)

        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")

        self.chk_role_obb_direct.setChecked(True)
        self.chk_role_seq_detect.setChecked(True)
        self.chk_role_seq_crop_obb.setChecked(True)

        g.addWidget(self.chk_role_obb_direct, 0, 0)
        g.addWidget(self.chk_role_seq_detect, 0, 1)
        g.addWidget(self.chk_role_seq_crop_obb, 0, 2)

        info = QLabel(
            "<b>Classification training has moved to ClassKit.</b><br>"
            "Use <code>classkit-labeler</code> to label and train classifiers."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 8px; background: #1a2a1a; border-left: 3px solid #4caf50;"
            " color: #aaa; border-radius: 3px;"
        )
        g.addWidget(info, 1, 0, 1, 3)

        note = QLabel("Sequential datasets are auto-derived from OBB sources.")
        note.setWordWrap(True)
        g.addWidget(note, 2, 0, 1, 3)
        return gb

    def _build_config_group(self):
        gb = QGroupBox("Step 3: Config")
        v = QVBoxLayout(gb)

        form = QFormLayout()
        self.line_workspace = QLineEdit(str(self.workspace_default))
        self.btn_workspace = QPushButton("Browse…")
        h_ws = QHBoxLayout()
        h_ws.addWidget(self.line_workspace, 1)
        h_ws.addWidget(self.btn_workspace)
        form.addRow("Workspace Root", h_ws)

        self.line_class = QLineEdit(self.class_name)
        form.addRow("Class Name", self.line_class)

        self.spin_train = QDoubleSpinBox()
        self.spin_train.setRange(0.05, 0.95)
        self.spin_train.setSingleStep(0.05)
        self.spin_train.setValue(0.8)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.05, 0.95)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.2)
        h_split = QHBoxLayout()
        h_split.addWidget(QLabel("train"))
        h_split.addWidget(self.spin_train)
        h_split.addWidget(QLabel("val"))
        h_split.addWidget(self.spin_val)
        form.addRow("Split", h_split)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(42)
        form.addRow("Seed", self.spin_seed)

        self.chk_dedup = QCheckBox("Deduplicate source images by content hash")
        self.chk_dedup.setChecked(True)
        form.addRow("", self.chk_dedup)

        self.spin_crop_pad = QDoubleSpinBox()
        self.spin_crop_pad.setRange(0.0, 1.0)
        self.spin_crop_pad.setSingleStep(0.01)
        self.spin_crop_pad.setValue(0.15)
        self.spin_crop_min_px = QSpinBox()
        self.spin_crop_min_px.setRange(8, 2048)
        self.spin_crop_min_px.setValue(64)
        self.chk_crop_square = QCheckBox("Enforce square crop")
        self.chk_crop_square.setChecked(True)
        h_crop = QHBoxLayout()
        h_crop.addWidget(QLabel("pad"))
        h_crop.addWidget(self.spin_crop_pad)
        h_crop.addWidget(QLabel("min px"))
        h_crop.addWidget(self.spin_crop_min_px)
        h_crop.addWidget(self.chk_crop_square)
        form.addRow("Sequential crop derivation", h_crop)

        self.combo_device = QComboBox()
        self.combo_device.addItems(self._build_device_options())
        form.addRow("Device", self.combo_device)

        v.addLayout(form)

        gb_train = QGroupBox("Training Hyperparameters")
        tr = QGridLayout(gb_train)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(64, 2048)
        self.spin_imgsz.setValue(640)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(1e-5, 1.0)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 500)
        self.spin_patience.setValue(30)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        self.chk_cache = QCheckBox("Cache")
        tr.addWidget(QLabel("epochs"), 0, 0)
        tr.addWidget(self.spin_epochs, 0, 1)
        tr.addWidget(QLabel("imgsz"), 0, 2)
        tr.addWidget(self.spin_imgsz, 0, 3)
        tr.addWidget(QLabel("batch"), 0, 4)
        tr.addWidget(self.spin_batch, 0, 5)
        tr.addWidget(QLabel("lr0"), 1, 0)
        tr.addWidget(self.spin_lr0, 1, 1)
        tr.addWidget(QLabel("patience"), 1, 2)
        tr.addWidget(self.spin_patience, 1, 3)
        tr.addWidget(QLabel("workers"), 1, 4)
        tr.addWidget(self.spin_workers, 1, 5)
        tr.addWidget(self.chk_cache, 2, 0, 1, 2)
        v.addWidget(gb_train)

        gb_models = QGroupBox("Base Models")
        fm = QFormLayout(gb_models)
        self.combo_model_obb_direct = QComboBox()
        self.combo_model_obb_direct.setEditable(True)
        self.combo_model_obb_direct.addItems(
            [
                "yolo26n-obb.pt",
                "yolo26s-obb.pt",
                "yolo26m-obb.pt",
                "yolo26l-obb.pt",
                "yolo26x-obb.pt",
            ]
        )
        self.combo_model_obb_direct.setCurrentText("yolo26s-obb.pt")

        self.combo_model_seq_detect = QComboBox()
        self.combo_model_seq_detect.setEditable(True)
        self.combo_model_seq_detect.addItems(
            ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]
        )
        self.combo_model_seq_detect.setCurrentText("yolo26s.pt")

        self.combo_model_seq_crop_obb = QComboBox()
        self.combo_model_seq_crop_obb.setEditable(True)
        self.combo_model_seq_crop_obb.addItems(
            ["yolo26n-obb.pt", "yolo26s-obb.pt", "yolo26m-obb.pt"]
        )
        self.combo_model_seq_crop_obb.setCurrentText("yolo26s-obb.pt")

        fm.addRow("obb_direct", self.combo_model_obb_direct)
        fm.addRow("seq_detect", self.combo_model_seq_detect)
        fm.addRow("seq_crop_obb", self.combo_model_seq_crop_obb)
        v.addWidget(gb_models)

        gb_publish = QGroupBox("Publish Metadata")
        fp = QFormLayout(gb_publish)
        self.line_species = QLineEdit(self.class_name)
        self.line_model_tag = QLineEdit("train")
        self.chk_auto_import = QCheckBox(
            "Auto-import successful models into repository"
        )
        self.chk_auto_import.setChecked(True)
        self.chk_auto_select = QCheckBox("Auto-select newly imported models in main UI")
        self.chk_auto_select.setChecked(False)
        fp.addRow("species", self.line_species)
        fp.addRow("model tag", self.line_model_tag)
        fp.addRow("", self.chk_auto_import)
        fp.addRow("", self.chk_auto_select)
        v.addWidget(gb_publish)

        self.btn_workspace.clicked.connect(self._choose_workspace)

        return gb

    def _build_run_group(self):
        gb = QGroupBox("Step 4: Build + Train + Monitor")
        v = QVBoxLayout(gb)

        row = QHBoxLayout()
        self.btn_build = QPushButton("Build Role Datasets")
        self.btn_train = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_build)
        row.addWidget(self.btn_train)
        row.addWidget(self.btn_stop)
        v.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        v.addWidget(self.log_view)

        self.btn_build.clicked.connect(self._build_role_datasets)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_stop.clicked.connect(self._stop_training)

        return gb

    def _build_device_options(self):
        info = get_device_info()
        options = ["auto", "cpu"]
        if info.get("torch_cuda_available"):
            options.append("cuda")
            count = int(info.get("torch_cuda_device_count", 0) or 0)
            for i in range(count):
                options.append(f"cuda:{i}")
        if info.get("mps_available"):
            options.append("mps")
        if info.get("rocm_available"):
            options.append("rocm")
        return options

    def _append_log(self, text: str):
        self.log_view.append(str(text))

    def _choose_workspace(self):
        d = QFileDialog.getExistingDirectory(self, "Select Workspace Root")
        if d:
            self.line_workspace.setText(d)
            self.orchestrator = TrainingOrchestrator(d)

    def _get_multiple_dirs(self, title):
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        for view in dlg.findChildren(QListView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for view in dlg.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QFileDialog.Accepted:
            return []
        return dlg.selectedFiles()

    def _add_source_row(self, source_type: str, path: str):
        row = self.table_sources.rowCount()
        self.table_sources.insertRow(row)
        pretty = "OBB Source"
        self.table_sources.setItem(row, 0, QTableWidgetItem(pretty))
        self.table_sources.setItem(row, 1, QTableWidgetItem(path))
        self.table_sources.setItem(row, 2, QTableWidgetItem("Pending"))
        self.table_sources.item(row, 0).setData(Qt.UserRole, source_type)

    def _add_obb_sources(self):
        for d in self._get_multiple_dirs("Select OBB Source Datasets"):
            self._add_source_row("obb", d)

    def _remove_selected(self):
        rows = {idx.row() for idx in self.table_sources.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table_sources.removeRow(row)

    def _clear_sources(self):
        self.table_sources.setRowCount(0)

    def _collect_sources(self):
        obb_sources = []
        for row in range(self.table_sources.rowCount()):
            st = self.table_sources.item(row, 0).data(Qt.UserRole)
            p = self.table_sources.item(row, 1).text().strip()
            if st == "obb":
                obb_sources.append(
                    SourceDataset(path=p, source_type="yolo_obb", name=Path(p).name)
                )
        return obb_sources

    def _validate_sources(self):
        obb_sources = self._collect_sources()
        if not obb_sources:
            QMessageBox.warning(
                self, "No OBB Sources", "Add at least one OBB source dataset."
            )
            return False

        try:
            report = self.orchestrator.preflight_obb_sources(
                obb_sources, require_train_val=False
            )
        except Exception as exc:
            QMessageBox.critical(self, "Validation Error", str(exc))
            return False

        self.validation_view.setPlainText(format_validation_report(report))

        status = "Validated" if report.valid else "Validation Failed"
        for row in range(self.table_sources.rowCount()):
            self.table_sources.setItem(row, 2, QTableWidgetItem(status))

        if not report.valid:
            QMessageBox.warning(
                self,
                "Validation Failed",
                "Source validation failed. See report for actionable issues.",
            )
            return False
        return True

    def _selected_roles(self):
        roles = []
        if self.chk_role_obb_direct.isChecked():
            roles.append(TrainingRole.OBB_DIRECT)
        if self.chk_role_seq_detect.isChecked():
            roles.append(TrainingRole.SEQ_DETECT)
        if self.chk_role_seq_crop_obb.isChecked():
            roles.append(TrainingRole.SEQ_CROP_OBB)
        return roles

    def _build_role_datasets(self):
        if not self._validate_sources():
            return False

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return False

        obb_sources = self._collect_sources()
        try:
            split = SplitConfig(
                train=self.spin_train.value(), val=self.spin_val.value(), test=0.0
            )
            merged = self.orchestrator.build_merged_obb_dataset(
                obb_sources,
                class_name=self.line_class.text().strip() or "object",
                split_cfg=split,
                seed=self.spin_seed.value(),
                dedup=self.chk_dedup.isChecked(),
            )
            self.role_dataset_dirs = {}
            self._append_log(f"Merged dataset: {merged.dataset_dir}")

            for role in roles:
                build = self.orchestrator.build_role_dataset(
                    role,
                    merged.dataset_dir,
                    class_name=self.line_class.text().strip() or "object",
                    crop_pad_ratio=self.spin_crop_pad.value(),
                    min_crop_size_px=self.spin_crop_min_px.value(),
                    enforce_square=self.chk_crop_square.isChecked(),
                )
                self.role_dataset_dirs[role.value] = build.dataset_dir
                self._append_log(
                    f"Prepared [{role.value}] dataset: {build.dataset_dir}"
                )
        except Exception as exc:
            QMessageBox.critical(self, "Build Failed", str(exc))
            return False

        QMessageBox.information(
            self, "Datasets Ready", "Role datasets built successfully."
        )
        return True

    def _base_model_for_role(self, role: TrainingRole) -> str:
        if role == TrainingRole.OBB_DIRECT:
            return self.combo_model_obb_direct.currentText().strip()
        if role == TrainingRole.SEQ_DETECT:
            return self.combo_model_seq_detect.currentText().strip()
        if role == TrainingRole.SEQ_CROP_OBB:
            return self.combo_model_seq_crop_obb.currentText().strip()
        return ""

    @staticmethod
    def _infer_size_token(model_path: str) -> str:
        name = Path(str(model_path or "")).name.lower()
        for token in (
            "26n",
            "26s",
            "26m",
            "26l",
            "26x",
            "11n",
            "11s",
            "11m",
            "11l",
            "11x",
        ):
            if token in name:
                return token
        return "unknown"

    def _publish_meta_for_role(self, role: TrainingRole, base_model: str) -> dict:
        species = self.line_species.text().strip() or "species"
        tag = self.line_model_tag.text().strip() or "train"
        role_suffix = role.value
        return {
            "size": self._infer_size_token(base_model),
            "species": species,
            "model_info": f"{tag}_{role_suffix}",
        }

    def _start_training(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Training is already running.")
            return

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return

        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        if not self.chk_auto_import.isChecked() and self.chk_auto_select.isChecked():
            QMessageBox.warning(
                self,
                "Invalid Publish Settings",
                "Auto-select requires auto-import to be enabled.",
            )
            return

        source_obb = self._collect_sources()
        role_entries = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                QMessageBox.warning(
                    self,
                    "Missing Dataset",
                    f"No dataset prepared for role: {role.value}",
                )
                return

            base_model = self._base_model_for_role(role)
            if not base_model:
                QMessageBox.warning(
                    self, "Base Model", f"Set base model for role: {role.value}"
                )
                return

            spec = TrainingRunSpec(
                role=role,
                source_datasets=source_obb,
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self.spin_imgsz.value(),
                    batch=self.spin_batch.value(),
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
                publish_policy=PublishPolicy(
                    auto_import=self.chk_auto_import.isChecked(),
                    auto_select=self.chk_auto_select.isChecked(),
                ),
            )
            role_entries.append(
                {
                    "role": role,
                    "spec": spec,
                    "publish_meta": self._publish_meta_for_role(role, base_model),
                }
            )

        self.worker = RoleTrainingWorker(self.orchestrator, role_entries)
        self.worker.log_signal.connect(self._append_log)
        self.worker.role_started.connect(self._on_role_started)
        self.worker.role_finished.connect(self._on_role_finished)
        self.worker.progress_signal.connect(self._on_role_progress)
        self.worker.done_signal.connect(self._on_training_done)

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.worker.start()

    def _stop_training(self):
        if self.worker is not None:
            self.worker.cancel()
            self._append_log("Cancellation requested…")

    def _on_role_started(self, role: str):
        self._append_log(f"=== START {role} ===")

    def _on_role_finished(self, role: str, ok: bool, message: str):
        self._append_log(f"=== {'OK' if ok else 'FAIL'} {role}: {message} ===")

    def _on_role_progress(self, role: str, cur: int, total: int):
        total = max(1, int(total))
        cur = max(0, min(total, int(cur)))
        pct = int((cur / total) * 100.0)
        self.progress.setValue(pct)
        self.progress.setFormat(f"{role}: {cur}/{total} ({pct}%)")

    def _on_training_done(self, results: list):
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setFormat("Done")

        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        self._append_log(
            f"Session complete: {len(succeeded)} success, {len(failed)} failed"
        )

        if self.chk_auto_select.isChecked():
            self._try_auto_select_parent_models(results)

        if failed:
            QMessageBox.warning(
                self,
                "Training Completed with Failures",
                f"Succeeded: {len(succeeded)}\nFailed: {len(failed)}\nSee logs for details.",
            )
        else:
            QMessageBox.information(
                self,
                "Training Completed",
                f"All {len(succeeded)} selected roles completed successfully.",
            )

    def _try_auto_select_parent_models(self, results: list[dict]):
        parent = self.parent()
        if parent is None:
            return

        for r in results:
            if not r.get("success"):
                continue
            key = str(r.get("published_registry_key") or "").strip()
            if not key:
                continue
            role = str(r.get("role") or "")
            try:
                if role == TrainingRole.OBB_DIRECT.value:
                    if hasattr(parent, "_refresh_yolo_model_combo"):
                        parent._refresh_yolo_model_combo(preferred_model_path=key)
                    if hasattr(parent, "_set_yolo_model_selection"):
                        parent._set_yolo_model_selection(key)
                elif role == TrainingRole.SEQ_DETECT.value:
                    if hasattr(parent, "_refresh_yolo_detect_model_combo"):
                        parent._refresh_yolo_detect_model_combo(
                            preferred_model_path=key
                        )
                    if hasattr(parent, "_set_yolo_detect_model_selection"):
                        parent._set_yolo_detect_model_selection(key)
                elif role == TrainingRole.SEQ_CROP_OBB.value:
                    if hasattr(parent, "_refresh_yolo_crop_obb_model_combo"):
                        parent._refresh_yolo_crop_obb_model_combo(
                            preferred_model_path=key
                        )
                    if hasattr(parent, "_set_yolo_crop_obb_model_selection"):
                        parent._set_yolo_crop_obb_model_selection(key)
            except Exception as exc:
                logger.warning("Auto-select model failed for role %s: %s", role, exc)
