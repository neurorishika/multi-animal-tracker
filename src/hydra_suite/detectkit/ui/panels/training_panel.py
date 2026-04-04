"""Training panel (right) for DetectKit -- full config, run controls, loss plot, log."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.gui.widgets.loss_plot_widget import LossPlotWidget
from hydra_suite.training import (
    AugmentationProfile,
    PublishPolicy,
    SourceDataset,
    SplitConfig,
    TrainingHyperParams,
    TrainingOrchestrator,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.validation import format_validation_report
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.utils.gpu_utils import get_device_info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker -- copied from train_yolo_dialog to avoid cross-app dependency
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Training Panel
# ---------------------------------------------------------------------------


class TrainingPanel(QWidget):
    """Right-side training panel for DetectKit."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._main_window = None
        self._proj = None
        self.worker = None
        self._last_training_results: list[dict] = []
        self.role_dataset_dirs: dict[str, str] = {}

        from hydra_suite.paths import get_training_workspace_dir

        self.workspace_default = get_training_workspace_dir("YOLO")
        self.orchestrator = TrainingOrchestrator(self.workspace_default)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        layout.addWidget(self._build_roles_group())
        layout.addWidget(self._build_config_group())
        layout.addWidget(self._build_hyperparams_group())
        layout.addWidget(self._build_base_models_group())
        layout.addWidget(self._build_augmentation_group())
        layout.addWidget(self._build_publish_group())
        layout.addWidget(self._build_run_group())
        layout.addWidget(self._build_loss_plot())
        layout.addWidget(self._build_log())

        layout.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll)

    # --- 1. Roles ---

    def _build_roles_group(self):
        gb = QGroupBox("Roles")
        v = QVBoxLayout(gb)

        h = QHBoxLayout()
        self.chk_role_obb_direct = QCheckBox("obb_direct")
        self.chk_role_seq_detect = QCheckBox("seq_detect")
        self.chk_role_seq_crop_obb = QCheckBox("seq_crop_obb")
        self.chk_role_obb_direct.setChecked(True)
        self.chk_role_seq_detect.setChecked(True)
        self.chk_role_seq_crop_obb.setChecked(True)
        h.addWidget(self.chk_role_obb_direct)
        h.addWidget(self.chk_role_seq_detect)
        h.addWidget(self.chk_role_seq_crop_obb)
        v.addLayout(h)

        note = QLabel("Sequential datasets are auto-derived from OBB sources.")
        note.setWordWrap(True)
        v.addWidget(note)
        return gb

    # --- 2. Config ---

    def _build_config_group(self):
        gb = QGroupBox("Config")
        form = QFormLayout(gb)

        self.line_class = QLineEdit("object")
        form.addRow("Class name", self.line_class)

        self.line_workspace = QLineEdit(str(self.workspace_default))
        self.btn_workspace = QPushButton("Browse...")
        h_ws = QHBoxLayout()
        h_ws.addWidget(self.line_workspace, 1)
        h_ws.addWidget(self.btn_workspace)
        form.addRow("Workspace root", h_ws)
        self.btn_workspace.clicked.connect(self._choose_workspace)

        # Split
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

        # Crop derivation
        self.spin_crop_pad = QDoubleSpinBox()
        self.spin_crop_pad.setRange(0.0, 1.0)
        self.spin_crop_pad.setSingleStep(0.01)
        self.spin_crop_pad.setValue(0.15)
        self.spin_crop_min_px = QSpinBox()
        self.spin_crop_min_px.setRange(8, 2048)
        self.spin_crop_min_px.setValue(64)
        self.chk_crop_square = QCheckBox("Square crop")
        self.chk_crop_square.setChecked(True)
        h_crop = QHBoxLayout()
        h_crop.addWidget(QLabel("pad"))
        h_crop.addWidget(self.spin_crop_pad)
        h_crop.addWidget(QLabel("min px"))
        h_crop.addWidget(self.spin_crop_min_px)
        h_crop.addWidget(self.chk_crop_square)
        form.addRow("Crop derivation", h_crop)

        # Device
        self.combo_device = QComboBox()
        self.combo_device.setEditable(True)
        self.combo_device.setToolTip(
            "Select compute device. For multi-GPU, type comma-separated list."
        )
        self.combo_device.addItems(self._build_device_options())
        form.addRow("Device", self.combo_device)

        return gb

    # --- 3. Hyperparameters ---

    def _build_hyperparams_group(self):
        gb = QGroupBox("Hyperparameters")
        g = QGridLayout(gb)

        # Row 0: epochs, batch + auto, lr0
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        g.addWidget(QLabel("epochs"), 0, 0)
        g.addWidget(self.spin_epochs, 0, 1)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 256)
        self.spin_batch.setValue(16)
        self.chk_auto_batch = QCheckBox("Auto")
        self.chk_auto_batch.setToolTip(
            "Let Ultralytics auto-detect optimal batch size (batch=-1)."
        )
        self.chk_auto_batch.toggled.connect(
            lambda checked: self.spin_batch.setEnabled(not checked)
        )
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(self.spin_batch)
        batch_layout.addWidget(self.chk_auto_batch)
        batch_widget = QWidget()
        batch_widget.setLayout(batch_layout)
        g.addWidget(QLabel("batch"), 0, 2)
        g.addWidget(batch_widget, 0, 3)

        self.spin_lr0 = QDoubleSpinBox()
        self.spin_lr0.setRange(1e-5, 1.0)
        self.spin_lr0.setDecimals(5)
        self.spin_lr0.setValue(0.01)
        g.addWidget(QLabel("lr0"), 0, 4)
        g.addWidget(self.spin_lr0, 0, 5)

        # Row 1: patience, workers, cache
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(1, 500)
        self.spin_patience.setValue(30)
        g.addWidget(QLabel("patience"), 1, 0)
        g.addWidget(self.spin_patience, 1, 1)

        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(0, 32)
        self.spin_workers.setValue(8)
        g.addWidget(QLabel("workers"), 1, 2)
        g.addWidget(self.spin_workers, 1, 3)

        self.chk_cache = QCheckBox("Cache")
        g.addWidget(self.chk_cache, 1, 4, 1, 2)

        # Row 2: imgsz per role
        self.spin_imgsz_obb_direct = QSpinBox()
        self.spin_imgsz_obb_direct.setRange(64, 2048)
        self.spin_imgsz_obb_direct.setValue(640)
        g.addWidget(QLabel("imgsz (obb_direct)"), 2, 0)
        g.addWidget(self.spin_imgsz_obb_direct, 2, 1)

        self.spin_imgsz_seq_detect = QSpinBox()
        self.spin_imgsz_seq_detect.setRange(64, 2048)
        self.spin_imgsz_seq_detect.setValue(640)
        g.addWidget(QLabel("imgsz (seq_detect)"), 2, 2)
        g.addWidget(self.spin_imgsz_seq_detect, 2, 3)

        self.spin_imgsz_seq_crop_obb = QSpinBox()
        self.spin_imgsz_seq_crop_obb.setRange(64, 2048)
        self.spin_imgsz_seq_crop_obb.setValue(160)
        self.spin_imgsz_seq_crop_obb.setToolTip(
            "Must match YOLO_SEQ_STAGE2_IMGSZ used during inference (default 160)."
        )
        g.addWidget(QLabel("imgsz (seq_crop_obb)"), 2, 4)
        g.addWidget(self.spin_imgsz_seq_crop_obb, 2, 5)

        return gb

    # --- 4. Base Models ---

    def _build_base_models_group(self):
        gb = QGroupBox("Base Models")
        form = QFormLayout(gb)

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
        form.addRow("obb_direct", self.combo_model_obb_direct)

        self.combo_model_seq_detect = QComboBox()
        self.combo_model_seq_detect.setEditable(True)
        self.combo_model_seq_detect.addItems(
            [
                "yolo26n.pt",
                "yolo26s.pt",
                "yolo26m.pt",
                "yolo26l.pt",
                "yolo26x.pt",
            ]
        )
        self.combo_model_seq_detect.setCurrentText("yolo26s.pt")
        form.addRow("seq_detect", self.combo_model_seq_detect)

        self.combo_model_seq_crop_obb = QComboBox()
        self.combo_model_seq_crop_obb.setEditable(True)
        self.combo_model_seq_crop_obb.addItems(
            [
                "yolo26n-obb.pt",
                "yolo26s-obb.pt",
                "yolo26m-obb.pt",
            ]
        )
        self.combo_model_seq_crop_obb.setCurrentText("yolo26s-obb.pt")
        form.addRow("seq_crop_obb", self.combo_model_seq_crop_obb)

        return gb

    # --- 5. Augmentation ---

    def _build_augmentation_group(self):
        self.aug_group = QGroupBox("Augmentation")
        self.aug_group.setCheckable(True)
        self.aug_group.setChecked(True)
        v = QVBoxLayout(self.aug_group)

        note = QLabel(
            "These are passed directly to Ultralytics. "
            "Set fliplr=0 for asymmetric animals."
        )
        note.setWordWrap(True)
        v.addWidget(note)

        form = QFormLayout()

        def _spin(default: float, maximum: float = 1.0) -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setRange(0.0, maximum)
            sb.setDecimals(3)
            sb.setSingleStep(0.05)
            sb.setValue(default)
            return sb

        self.aug_fliplr = _spin(0.5)
        form.addRow("fliplr", self.aug_fliplr)
        self.aug_flipud = _spin(0.0)
        form.addRow("flipud", self.aug_flipud)
        self.aug_degrees = _spin(0.0, 360.0)
        form.addRow("degrees", self.aug_degrees)
        self.aug_mosaic = _spin(1.0)
        form.addRow("mosaic", self.aug_mosaic)
        self.aug_mixup = _spin(0.0)
        form.addRow("mixup", self.aug_mixup)
        self.aug_hsv_h = _spin(0.015)
        form.addRow("hsv_h", self.aug_hsv_h)
        self.aug_hsv_s = _spin(0.7)
        form.addRow("hsv_s", self.aug_hsv_s)
        self.aug_hsv_v = _spin(0.4)
        form.addRow("hsv_v", self.aug_hsv_v)

        v.addLayout(form)
        return self.aug_group

    # --- 6. Publish ---

    def _build_publish_group(self):
        gb = QGroupBox("Publish")
        form = QFormLayout(gb)

        self.line_species = QLineEdit("")
        form.addRow("species", self.line_species)

        self.line_model_tag = QLineEdit("train")
        form.addRow("model tag", self.line_model_tag)

        self.chk_auto_import = QCheckBox(
            "Auto-import successful models into repository"
        )
        self.chk_auto_import.setChecked(True)
        form.addRow("", self.chk_auto_import)

        self.chk_auto_select = QCheckBox("Auto-select newly imported models in main UI")
        self.chk_auto_select.setChecked(False)
        form.addRow("", self.chk_auto_select)

        return gb

    # --- 7. Run Controls ---

    def _build_run_group(self):
        gb = QGroupBox("Run Controls")
        v = QVBoxLayout(gb)

        row1 = QHBoxLayout()
        self.btn_build = QPushButton("Build Role Datasets")
        self.btn_train = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_resume = QPushButton("Resume")
        self.btn_resume.setEnabled(False)
        self.btn_resume.setToolTip(
            "Resume training from last.pt checkpoint of the most recent run."
        )
        self.btn_detach = QPushButton("Start Detached")
        self.btn_detach.setToolTip("Launch training as a background process.")
        row1.addWidget(self.btn_build)
        row1.addWidget(self.btn_train)
        row1.addWidget(self.btn_stop)
        row1.addWidget(self.btn_resume)
        row1.addWidget(self.btn_detach)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_quick_test = QPushButton("Quick Test")
        self.btn_quick_test.setEnabled(False)
        self.btn_history = QPushButton("Run History")
        self.btn_save_config = QPushButton("Save Config")
        self.btn_load_config = QPushButton("Load Config")
        row2.addWidget(self.btn_quick_test)
        row2.addWidget(self.btn_history)
        row2.addWidget(self.btn_save_config)
        row2.addWidget(self.btn_load_config)
        v.addLayout(row2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        v.addWidget(self.progress)

        # Connect signals
        self.btn_build.clicked.connect(self._build_role_datasets)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_resume.clicked.connect(self._resume_training)
        self.btn_detach.clicked.connect(self._start_detached)
        self.btn_quick_test.clicked.connect(self._quick_test)
        self.btn_history.clicked.connect(self._show_history)
        self.btn_save_config.clicked.connect(self._save_training_config)
        self.btn_load_config.clicked.connect(self._load_training_config)

        return gb

    # --- 8. Loss Plot ---

    def _build_loss_plot(self):
        self.loss_plot = LossPlotWidget()
        self.loss_plot.setMinimumHeight(180)
        return self.loss_plot

    # --- 9. Log ---

    def _build_log(self):
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Training log output appears here.")
        return self.log_view

    # ------------------------------------------------------------------
    # Project interface
    # ------------------------------------------------------------------

    def set_project(self, proj, main_window):
        """Populate all widgets from *proj* fields. Store references."""
        self._proj = proj
        self._main_window = main_window

        self.line_class.setText(proj.class_name or "object")
        self.line_species.setText(proj.species or "")
        self.line_model_tag.setText(proj.model_tag or "train")

        # Roles
        self.chk_role_obb_direct.setChecked(proj.role_obb_direct)
        self.chk_role_seq_detect.setChecked(proj.role_seq_detect)
        self.chk_role_seq_crop_obb.setChecked(proj.role_seq_crop_obb)

        # Split
        self.spin_train.setValue(proj.split_train)
        self.spin_val.setValue(proj.split_val)
        self.spin_seed.setValue(proj.seed)
        self.chk_dedup.setChecked(proj.dedup)

        # Crop
        self.spin_crop_pad.setValue(proj.crop_pad_ratio)
        self.spin_crop_min_px.setValue(proj.min_crop_size_px)
        self.chk_crop_square.setChecked(proj.enforce_square)

        # Imgsz
        self.spin_imgsz_obb_direct.setValue(proj.imgsz_obb_direct)
        self.spin_imgsz_seq_detect.setValue(proj.imgsz_seq_detect)
        self.spin_imgsz_seq_crop_obb.setValue(proj.imgsz_seq_crop_obb)

        # Base models
        self.combo_model_obb_direct.setCurrentText(proj.model_obb_direct)
        self.combo_model_seq_detect.setCurrentText(proj.model_seq_detect)
        self.combo_model_seq_crop_obb.setCurrentText(proj.model_seq_crop_obb)

        # Hyperparams
        self.spin_epochs.setValue(proj.epochs)
        self.spin_batch.setValue(proj.batch)
        self.chk_auto_batch.setChecked(proj.auto_batch)
        self.spin_lr0.setValue(proj.lr0)
        self.spin_patience.setValue(proj.patience)
        self.spin_workers.setValue(proj.workers)
        self.chk_cache.setChecked(proj.cache)

        # Augmentation
        self.aug_group.setChecked(proj.aug_enabled)
        self.aug_fliplr.setValue(proj.aug_fliplr)
        self.aug_flipud.setValue(proj.aug_flipud)
        self.aug_degrees.setValue(proj.aug_degrees)
        self.aug_mosaic.setValue(proj.aug_mosaic)
        self.aug_mixup.setValue(proj.aug_mixup)
        self.aug_hsv_h.setValue(proj.aug_hsv_h)
        self.aug_hsv_s.setValue(proj.aug_hsv_s)
        self.aug_hsv_v.setValue(proj.aug_hsv_v)

        # Device
        self.combo_device.setCurrentText(proj.device or "auto")

        # Publish
        self.chk_auto_import.setChecked(proj.auto_import)
        self.chk_auto_select.setChecked(proj.auto_select)

    def collect_state(self, proj):
        """Write all widget values back to *proj* fields."""
        proj.class_name = self.line_class.text().strip() or "object"
        proj.species = self.line_species.text().strip()
        proj.model_tag = self.line_model_tag.text().strip() or "train"

        # Roles
        proj.role_obb_direct = self.chk_role_obb_direct.isChecked()
        proj.role_seq_detect = self.chk_role_seq_detect.isChecked()
        proj.role_seq_crop_obb = self.chk_role_seq_crop_obb.isChecked()

        # Split
        proj.split_train = self.spin_train.value()
        proj.split_val = self.spin_val.value()
        proj.seed = self.spin_seed.value()
        proj.dedup = self.chk_dedup.isChecked()

        # Crop
        proj.crop_pad_ratio = self.spin_crop_pad.value()
        proj.min_crop_size_px = self.spin_crop_min_px.value()
        proj.enforce_square = self.chk_crop_square.isChecked()

        # Imgsz
        proj.imgsz_obb_direct = self.spin_imgsz_obb_direct.value()
        proj.imgsz_seq_detect = self.spin_imgsz_seq_detect.value()
        proj.imgsz_seq_crop_obb = self.spin_imgsz_seq_crop_obb.value()

        # Base models
        proj.model_obb_direct = self.combo_model_obb_direct.currentText()
        proj.model_seq_detect = self.combo_model_seq_detect.currentText()
        proj.model_seq_crop_obb = self.combo_model_seq_crop_obb.currentText()

        # Hyperparams
        proj.epochs = self.spin_epochs.value()
        proj.batch = self.spin_batch.value()
        proj.auto_batch = self.chk_auto_batch.isChecked()
        proj.lr0 = self.spin_lr0.value()
        proj.patience = self.spin_patience.value()
        proj.workers = self.spin_workers.value()
        proj.cache = self.chk_cache.isChecked()

        # Augmentation
        proj.aug_enabled = self.aug_group.isChecked()
        proj.aug_fliplr = self.aug_fliplr.value()
        proj.aug_flipud = self.aug_flipud.value()
        proj.aug_degrees = self.aug_degrees.value()
        proj.aug_mosaic = self.aug_mosaic.value()
        proj.aug_mixup = self.aug_mixup.value()
        proj.aug_hsv_h = self.aug_hsv_h.value()
        proj.aug_hsv_s = self.aug_hsv_s.value()
        proj.aug_hsv_v = self.aug_hsv_v.value()

        # Device
        proj.device = self.combo_device.currentText().strip() or "auto"

        # Publish
        proj.auto_import = self.chk_auto_import.isChecked()
        proj.auto_select = self.chk_auto_select.isChecked()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _imgsz_for_role(self, role: TrainingRole) -> int:
        if role == TrainingRole.OBB_DIRECT:
            return self.spin_imgsz_obb_direct.value()
        if role == TrainingRole.SEQ_DETECT:
            return self.spin_imgsz_seq_detect.value()
        if role == TrainingRole.SEQ_CROP_OBB:
            return self.spin_imgsz_seq_crop_obb.value()
        return 640

    def _base_model_for_role(self, role: TrainingRole) -> str:
        if role == TrainingRole.OBB_DIRECT:
            return self.combo_model_obb_direct.currentText().strip()
        if role == TrainingRole.SEQ_DETECT:
            return self.combo_model_seq_detect.currentText().strip()
        if role == TrainingRole.SEQ_CROP_OBB:
            return self.combo_model_seq_crop_obb.currentText().strip()
        return ""

    def _selected_roles(self):
        roles = []
        if self.chk_role_obb_direct.isChecked():
            roles.append(TrainingRole.OBB_DIRECT)
        if self.chk_role_seq_detect.isChecked():
            roles.append(TrainingRole.SEQ_DETECT)
        if self.chk_role_seq_crop_obb.isChecked():
            roles.append(TrainingRole.SEQ_CROP_OBB)
        return roles

    def _collect_sources(self):
        """Get sources from the main window project, convert to SourceDataset list."""
        if self._main_window is None:
            return []
        proj = self._main_window.project()
        if proj is None:
            return []
        obb_sources = []
        for src in proj.sources:
            p = src.path.strip()
            if p:
                obb_sources.append(
                    SourceDataset(
                        path=p,
                        source_type="yolo_obb",
                        name=Path(p).name,
                    )
                )
        return obb_sources

    def _append_log(self, text: str):
        self.log_view.append(str(text))
        if hasattr(self, "loss_plot"):
            self.loss_plot.ingest_log_line(str(text))

    def _choose_workspace(self):
        d = QFileDialog.getExistingDirectory(self, "Select Workspace Root")
        if d:
            self.line_workspace.setText(d)
            self.orchestrator = TrainingOrchestrator(d)

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
        training_params: dict = {
            "imgsz": self._imgsz_for_role(role),
        }
        if role == TrainingRole.SEQ_CROP_OBB:
            training_params["crop_pad_ratio"] = self.spin_crop_pad.value()
            training_params["min_crop_size_px"] = self.spin_crop_min_px.value()
            training_params["enforce_square"] = self.chk_crop_square.isChecked()
        return {
            "size": self._infer_size_token(base_model),
            "species": species,
            "model_info": f"{tag}_{role.value}",
            "training_params": training_params,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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

        text = format_validation_report(report)
        self._append_log(text)

        if not report.valid:
            QMessageBox.warning(
                self,
                "Validation Failed",
                "Source validation failed. See log for details.",
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Dataset building
    # ------------------------------------------------------------------

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
                train=self.spin_train.value(),
                val=self.spin_val.value(),
                test=0.0,
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

    # ------------------------------------------------------------------
    # Training execution
    # ------------------------------------------------------------------

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

            aug_args: dict[str, float] = {}
            if self.aug_group.isChecked():
                aug_args = {
                    "fliplr": self.aug_fliplr.value(),
                    "flipud": self.aug_flipud.value(),
                    "degrees": self.aug_degrees.value(),
                    "mosaic": self.aug_mosaic.value(),
                    "mixup": self.aug_mixup.value(),
                    "hsv_h": self.aug_hsv_h.value(),
                    "hsv_s": self.aug_hsv_s.value(),
                    "hsv_v": self.aug_hsv_v.value(),
                }

            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            spec = TrainingRunSpec(
                role=role,
                source_datasets=source_obb,
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
                augmentation_profile=AugmentationProfile(
                    enabled=self.aug_group.isChecked(),
                    args=aug_args,
                ),
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
        self.loss_plot.clear()
        self.worker.start()

    def _stop_training(self):
        if self.worker is not None:
            self.worker.cancel()
            self._append_log("Cancellation requested...")

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

        self._last_training_results = results
        for r in results:
            run_dir = ""
            artifact = r.get("artifact_path", "")
            if artifact:
                _wdir = Path(artifact).parent
                if _wdir.name == "weights":
                    run_dir = str(_wdir.parent)
            r["_run_dir"] = run_dir

        self.btn_resume.setEnabled(
            any(
                r.get("_run_dir")
                and Path(r["_run_dir"]).joinpath("weights", "last.pt").exists()
                for r in results
            )
        )

        succeeded = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]
        self.btn_quick_test.setEnabled(bool(succeeded))

        self._append_log(
            f"Session complete: {len(succeeded)} success, {len(failed)} failed"
        )

        if failed:
            QMessageBox.warning(
                self,
                "Training Completed with Failures",
                f"Succeeded: {len(succeeded)}\nFailed: {len(failed)}\n"
                "See logs for details.",
            )
        else:
            QMessageBox.information(
                self,
                "Training Completed",
                f"All {len(succeeded)} selected roles completed successfully.",
            )

    def _resume_training(self):
        """Resume training from last.pt checkpoint of the most recent run."""
        last_pt = None
        resume_result = None
        for r in reversed(self._last_training_results):
            run_dir = r.get("_run_dir", "")
            if run_dir:
                candidate = Path(run_dir) / "weights" / "last.pt"
                if candidate.exists():
                    last_pt = candidate
                    resume_result = r
                    break

        if last_pt is None:
            QMessageBox.warning(
                self,
                "No Checkpoint Found",
                "Could not find a last.pt checkpoint from the previous run.",
            )
            return

        role_str = str(resume_result.get("role", ""))
        try:
            role = TrainingRole(role_str)
        except ValueError:
            QMessageBox.warning(
                self, "Resume Failed", f"Unknown training role: {role_str}"
            )
            return

        resume_batch_val = (
            -1 if self.chk_auto_batch.isChecked() else int(self.spin_batch.value())
        )
        spec = TrainingRunSpec(
            role=role,
            source_datasets=[],
            derived_dataset_dir=resume_result.get("_run_dir", ""),
            base_model=str(last_pt),
            hyperparams=TrainingHyperParams(
                epochs=int(self.spin_epochs.value()),
                imgsz=self._imgsz_for_role(role),
                batch=resume_batch_val,
                lr0=float(self.spin_lr0.value()),
                patience=int(self.spin_patience.value()),
                workers=int(self.spin_workers.value()),
            ),
            resume_from=str(last_pt),
        )

        publish_meta = {
            "class_name": self.line_class.text().strip() or "object",
            "resumed_from": str(last_pt),
        }
        entry = {"role": role, "spec": spec, "publish_meta": publish_meta}

        self._append_log(f"Resuming training from {last_pt}")
        self.btn_train.setEnabled(False)
        self.btn_resume.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setFormat("Resuming...")

        self.worker = RoleTrainingWorker(self.orchestrator, [entry])
        self.worker.log_signal.connect(self._append_log)
        self.worker.role_started.connect(self._on_role_started)
        self.worker.role_finished.connect(self._on_role_finished)
        self.worker.progress_signal.connect(self._on_role_progress)
        self.worker.done_signal.connect(self._on_training_done)
        self.worker.start()

    def _start_detached(self):
        """Launch training as a detached subprocess."""
        import subprocess as _subprocess

        roles = self._selected_roles()
        if not roles:
            QMessageBox.warning(self, "No Roles", "Select at least one training role.")
            return
        if not self.role_dataset_dirs:
            if not self._build_role_datasets():
                return

        launched = []
        for role in roles:
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                continue
            base_model = self._base_model_for_role(role)
            if not base_model:
                continue

            batch_val = (
                -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
            )
            spec = TrainingRunSpec(
                role=role,
                source_datasets=self._collect_sources(),
                derived_dataset_dir=ds,
                base_model=base_model,
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=batch_val,
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
            )
            from hydra_suite.training.runner import build_ultralytics_command

            ws_text = self.line_workspace.text().strip()
            ws = Path(ws_text) if ws_text else self.workspace_default
            run_dir = ws / "runs" / ("detached_" + role.value)
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_ultralytics_command(spec, str(run_dir))

            log_file = run_dir / "detached_output.log"
            with open(log_file, "w") as log_fh:
                proc = _subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=_subprocess.STDOUT,
                    start_new_session=True,
                )
            launched.append((role.value, proc.pid, str(log_file)))
            self._append_log(
                f"Detached {role.value} training: PID={proc.pid}, log={log_file}"
            )

        if launched:
            msg = "\n".join(
                f"* {role}: PID {pid}\n  Log: {log}" for role, pid, log in launched
            )
            QMessageBox.information(
                self,
                "Detached Training Started",
                f"Training launched in background:\n\n{msg}\n\n"
                "You can close this panel. Check Run History for results.",
            )

    # ------------------------------------------------------------------
    # Quick Test / History
    # ------------------------------------------------------------------

    def _quick_test(self):
        """Open ModelTestDialog for the last successful training result."""
        from hydra_suite.trackerkit.gui.dialogs.model_test_dialog import ModelTestDialog

        results = getattr(self, "_last_training_results", [])
        succeeded = [r for r in results if r.get("success")]
        if not succeeded:
            QMessageBox.warning(
                self,
                "No Model Available",
                "No successfully trained model found. Run training first.",
            )
            return

        result = succeeded[0]
        model_path = result.get("published_model_path") or result.get(
            "artifact_path", ""
        )
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(
                self, "Model Not Found", f"Model file not found: {model_path}"
            )
            return

        role = result.get("role", "obb_direct")
        dataset_dir = self.role_dataset_dirs.get(role, "")
        if not dataset_dir:
            QMessageBox.warning(
                self, "No Dataset", f"No dataset directory found for role '{role}'."
            )
            return

        device = self.combo_device.currentText() or "cpu"
        imgsz_map = {
            "obb_direct": self.spin_imgsz_obb_direct.value(),
            "seq_detect": self.spin_imgsz_seq_detect.value(),
            "seq_crop_obb": self.spin_imgsz_seq_crop_obb.value(),
        }
        imgsz = imgsz_map.get(role, 640)

        dlg = ModelTestDialog(
            model_path=model_path,
            role=role,
            dataset_dir=dataset_dir,
            device=device,
            imgsz=imgsz,
            crop_pad_ratio=self.spin_crop_pad.value(),
            min_crop_size_px=self.spin_crop_min_px.value(),
            enforce_square=self.chk_crop_square.isChecked(),
            parent=self,
        )
        dlg.exec()

    def _show_history(self):
        """Open the training run history viewer dialog."""
        from hydra_suite.trackerkit.gui.dialogs.run_history_dialog import (
            RunHistoryDialog,
        )

        dlg = RunHistoryDialog(parent=self)
        dlg.exec()

    # ------------------------------------------------------------------
    # Save / Load training config
    # ------------------------------------------------------------------

    def _save_training_config(self):
        """Export all panel settings to a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Config",
            str(self.workspace_default),
            "JSON Files (*.json)",
        )
        if not path:
            return

        roles = []
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            chk = getattr(self, f"chk_role_{role_key}", None)
            if chk and chk.isChecked():
                roles.append(role_key)

        config = {
            "version": 1,
            "class_name": self.line_class.text().strip(),
            "roles": roles,
            "hyperparams": {
                "epochs": self.spin_epochs.value(),
                "batch": self.spin_batch.value(),
                "auto_batch": self.chk_auto_batch.isChecked(),
                "lr0": self.spin_lr0.value(),
                "patience": self.spin_patience.value(),
                "workers": self.spin_workers.value(),
                "cache": self.chk_cache.isChecked(),
            },
            "imgsz": {
                "obb_direct": self.spin_imgsz_obb_direct.value(),
                "seq_detect": self.spin_imgsz_seq_detect.value(),
                "seq_crop_obb": self.spin_imgsz_seq_crop_obb.value(),
            },
            "split": {
                "train": self.spin_train.value(),
                "val": self.spin_val.value(),
            },
            "seed": self.spin_seed.value(),
            "dedup": self.chk_dedup.isChecked(),
            "crop_derivation": {
                "pad_ratio": self.spin_crop_pad.value(),
                "min_crop_size_px": self.spin_crop_min_px.value(),
                "enforce_square": self.chk_crop_square.isChecked(),
            },
            "base_models": {
                "obb_direct": self.combo_model_obb_direct.currentText(),
                "seq_detect": self.combo_model_seq_detect.currentText(),
                "seq_crop_obb": self.combo_model_seq_crop_obb.currentText(),
            },
            "augmentation": {
                "enabled": self.aug_group.isChecked(),
                "fliplr": self.aug_fliplr.value(),
                "flipud": self.aug_flipud.value(),
                "degrees": self.aug_degrees.value(),
                "mosaic": self.aug_mosaic.value(),
                "mixup": self.aug_mixup.value(),
                "hsv_h": self.aug_hsv_h.value(),
                "hsv_s": self.aug_hsv_s.value(),
                "hsv_v": self.aug_hsv_v.value(),
            },
            "device": self.combo_device.currentText(),
            "publish": {
                "species": self.line_species.text().strip(),
                "model_tag": self.line_model_tag.text().strip(),
                "auto_import": self.chk_auto_import.isChecked(),
                "auto_select": self.chk_auto_select.isChecked(),
            },
        }

        try:
            Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")
            self._append_log(f"Config saved to {path}")
        except Exception as exc:
            self._append_log(f"Failed to save config: {exc}")

    def _load_training_config(self):
        """Restore panel settings from a JSON config file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Training Config",
            str(self.workspace_default),
            "JSON Files (*.json)",
        )
        if not path:
            return

        try:
            config = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self._append_log(f"Failed to load config: {exc}")
            return

        if "class_name" in config:
            self.line_class.setText(config["class_name"])

        self._apply_roles_config(config.get("roles", []))
        self._apply_hyperparams_config(config.get("hyperparams", {}))
        self._apply_imgsz_config(config.get("imgsz", {}))
        self._apply_split_config(config.get("split", {}))
        self._apply_seed_dedup_config(config)
        self._apply_crop_derivation_config(config.get("crop_derivation", {}))
        self._apply_base_models_config(config.get("base_models", {}))
        self._apply_augmentation_config(config.get("augmentation", {}))

        if "device" in config:
            self.combo_device.setCurrentText(config["device"])

        self._apply_publish_config(config.get("publish", {}))
        self._append_log(f"Config loaded from {path}")

    def _apply_roles_config(self, roles):
        """Apply role checkboxes from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            chk = getattr(self, f"chk_role_{role_key}", None)
            if chk:
                chk.setChecked(role_key in roles)

    def _apply_hyperparams_config(self, hp):
        """Apply hyperparameter widgets from loaded config."""
        for key, widget_name in (
            ("epochs", "spin_epochs"),
            ("batch", "spin_batch"),
            ("lr0", "spin_lr0"),
            ("patience", "spin_patience"),
            ("workers", "spin_workers"),
        ):
            if key in hp:
                w = getattr(self, widget_name, None)
                if w:
                    w.setValue(hp[key])
        if "cache" in hp:
            self.chk_cache.setChecked(hp["cache"])
        if "auto_batch" in hp:
            self.chk_auto_batch.setChecked(hp["auto_batch"])

    def _apply_imgsz_config(self, isz):
        """Apply image-size spinners from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            if role_key in isz:
                w = getattr(self, f"spin_imgsz_{role_key}", None)
                if w:
                    w.setValue(isz[role_key])

    def _apply_split_config(self, sp):
        """Apply train/val split values from loaded config."""
        if "train" in sp:
            self.spin_train.setValue(sp["train"])
        if "val" in sp:
            self.spin_val.setValue(sp["val"])

    def _apply_seed_dedup_config(self, config):
        """Apply seed and dedup settings from loaded config."""
        if "seed" in config:
            self.spin_seed.setValue(config["seed"])
        if "dedup" in config:
            self.chk_dedup.setChecked(config["dedup"])

    def _apply_crop_derivation_config(self, cd):
        """Apply crop derivation settings from loaded config."""
        if "pad_ratio" in cd:
            self.spin_crop_pad.setValue(cd["pad_ratio"])
        if "min_crop_size_px" in cd:
            self.spin_crop_min_px.setValue(cd["min_crop_size_px"])
        if "enforce_square" in cd:
            self.chk_crop_square.setChecked(cd["enforce_square"])

    def _apply_base_models_config(self, bm):
        """Apply base model combo boxes from loaded config."""
        for role_key in ("obb_direct", "seq_detect", "seq_crop_obb"):
            if role_key in bm:
                combo = getattr(self, f"combo_model_{role_key}", None)
                if combo:
                    combo.setCurrentText(bm[role_key])

    def _apply_augmentation_config(self, aug):
        """Apply augmentation settings from loaded config."""
        if "enabled" in aug:
            self.aug_group.setChecked(aug["enabled"])
        for key in (
            "fliplr",
            "flipud",
            "degrees",
            "mosaic",
            "mixup",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        ):
            if key in aug:
                w = getattr(self, f"aug_{key}", None)
                if w:
                    w.setValue(aug[key])

    def _apply_publish_config(self, pub):
        """Apply publish settings from loaded config."""
        if "species" in pub:
            self.line_species.setText(pub["species"])
        if "model_tag" in pub:
            self.line_model_tag.setText(pub["model_tag"])
        if "auto_import" in pub:
            self.chk_auto_import.setChecked(pub["auto_import"])
        if "auto_select" in pub:
            self.chk_auto_select.setChecked(pub["auto_select"])

    # ------------------------------------------------------------------
    # Analysis / preview (delegate to canvas if available)
    # ------------------------------------------------------------------

    def _analyze_and_preview(self):
        """Run analysis and show crop previews in canvas."""
        obb_sources = self._collect_sources()
        if not obb_sources:
            QMessageBox.warning(
                self, "No Sources", "Add at least one OBB source dataset."
            )
            return

        from hydra_suite.training.dataset_inspector import (
            DatasetInspection,
            analyze_obb_sizes,
            format_size_analysis,
            inspect_obb_or_detect_dataset,
        )

        pad = self.spin_crop_pad.value()
        min_px = self.spin_crop_min_px.value()
        square = self.chk_crop_square.isChecked()
        crop_imgsz = self.spin_imgsz_seq_crop_obb.value()

        merged = DatasetInspection(root_dir="(merged)")
        for src in obb_sources:
            try:
                insp = inspect_obb_or_detect_dataset(src.path)
            except Exception as exc:
                self._append_log(f"Skip {src.path}: {exc}")
                continue
            for split, items in insp.splits.items():
                merged.splits.setdefault(split, []).extend(items)

        if not any(merged.splits.values()):
            QMessageBox.warning(self, "No Data", "No valid images found in sources.")
            return

        try:
            stats = analyze_obb_sizes(
                merged,
                pad_ratio=pad,
                min_crop_size_px=min_px,
                enforce_square=square,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            return

        report, warnings = format_size_analysis(stats, training_imgsz=crop_imgsz)
        self._append_log(report)
        if warnings:
            for w in warnings:
                self._append_log(f"  WARNING: {w}")
