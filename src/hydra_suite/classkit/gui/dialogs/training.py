"""ClassKitTrainingDialog — training dialog for ClassKit."""

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.training.torchvision_model import (
    BACKBONE_DISPLAY_NAMES,
    TORCHVISION_BACKBONES,
)


class ClassKitTrainingDialog(QDialog):
    """Training dialog for ClassKit: flat or multi-head, tiny CNN or YOLO-classify."""

    def __init__(
        self,
        scheme=None,
        n_labeled: int = 0,
        class_choices: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._scheme = scheme
        self._n_labeled = n_labeled
        self._class_choices = self._resolve_class_choices(class_choices)
        self._train_results = None
        self._worker = None
        self.setWindowTitle("Train Classifier")
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)
        self._build_ui()

    def _resolve_class_choices(
        self, class_choices: Optional[List[str]] = None
    ) -> List[str]:
        ordered: List[str] = []

        def _add(value: object) -> None:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

        for value in class_choices or []:
            _add(value)

        if self._scheme is not None:
            factors = getattr(self._scheme, "factors", None) or []
            if len(factors) == 1:
                for label in getattr(factors[0], "labels", []) or []:
                    _add(label)

        return ordered

    def _build_general_tab(self):
        """Build the General settings tab and return its widget."""
        self.general_tab = QWidget()
        form = QFormLayout()
        form.setSpacing(8)

        self.mode_combo = QComboBox()
        self._populate_modes()
        form.addRow("<b>Training Mode:</b>", self.mode_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        try:
            from hydra_suite.utils.gpu_utils import (
                MPS_AVAILABLE,
                ROCM_AVAILABLE,
                TORCH_CUDA_AVAILABLE,
            )

            if TORCH_CUDA_AVAILABLE:
                self.device_combo.addItem("CUDA GPU", "cuda")
            if ROCM_AVAILABLE:
                self.device_combo.addItem("ROCm GPU", "rocm")
            if MPS_AVAILABLE:
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass
        self.device_combo.setToolTip(
            "Hardware device used for model training (PyTorch backend)."
        )
        form.addRow("<b>Training Device:</b>", self.device_combo)

        self._build_compute_runtime_combo(form)
        self._build_base_model_combo(form)
        self._build_hyperparams_widgets(form)

        self._mode_desc_label = QLabel("")
        self._mode_desc_label.setWordWrap(True)
        self._mode_desc_label.setStyleSheet(
            "color: #888; font-size: 11px; padding: 4px 8px; "
            "border-left: 2px solid #3e3e42; margin-top: 2px;"
        )

        layout_gen = QVBoxLayout(self.general_tab)
        layout_gen.addLayout(form)
        layout_gen.addWidget(self._mode_desc_label)
        layout_gen.addStretch()
        return self.general_tab

    def _build_compute_runtime_combo(self, form):
        """Build the inference runtime combo box."""
        self.compute_runtime_combo = QComboBox()
        try:
            from hydra_suite.runtime.compute_runtime import (
                runtime_label,
                supported_runtimes_for_pipeline,
            )

            _runtimes = supported_runtimes_for_pipeline("tiny_classify")
            for _rt in _runtimes:
                self.compute_runtime_combo.addItem(runtime_label(_rt), _rt)

            _train_dev = str(self.device_combo.currentData() or "cpu").strip().lower()
            if _train_dev == "mps":
                _preferred_rt = "mps"
            elif _train_dev == "rocm":
                _preferred_rt = "onnx_rocm" if "onnx_rocm" in _runtimes else "rocm"
            elif _train_dev == "cuda":
                _preferred_rt = "onnx_cuda" if "onnx_cuda" in _runtimes else "cuda"
            else:
                _preferred_rt = "cpu"

            _idx = self.compute_runtime_combo.findData(_preferred_rt)
            if _idx >= 0:
                self.compute_runtime_combo.setCurrentIndex(_idx)
        except Exception:
            self.compute_runtime_combo.addItem("CPU", "cpu")

        self.compute_runtime_combo.setToolTip(
            "Runtime used for Tiny CNN inference in ClassKit (and MAT integration).\n"
            "ONNX / TensorRT runtimes use exported artifacts (auto-exported after training)."
        )
        form.addRow("<b>Inference Runtime:</b>", self.compute_runtime_combo)

    def _build_base_model_combo(self, form):
        """Build the YOLO base model combo box."""
        self.base_model_combo = QComboBox()
        for m in [
            "yolo26n-cls.pt",
            "yolo26s-cls.pt",
            "yolo26m-cls.pt",
            "yolo26l-cls.pt",
            "yolo26x-cls.pt",
        ]:
            self.base_model_combo.addItem(m, m)
        self.base_model_combo.setToolTip(
            "Pretrained YOLO backbone to fine-tune.\n"
            "'n' = nano (fastest, least RAM)  'x' = extra-large (most accurate, needs GPU)"
        )
        self._base_model_row_label = QLabel("<b>Base Model (YOLO):</b>")
        form.addRow(self._base_model_row_label, self.base_model_combo)

    def _build_hyperparams_widgets(self, form):
        """Build epoch, batch, lr, val fraction, and patience widgets."""
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        form.addRow("<b>Epochs:</b>", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 512)
        self.batch_spin.setValue(32)
        form.addRow("<b>Batch Size:</b>", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("<b>Learning Rate:</b>", self.lr_spin)

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.0, 0.5)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.2)
        form.addRow("<b>Val Fraction:</b>", self.val_fraction_spin)

        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(0, 500)
        self.patience_spin.setValue(10)
        form.addRow("<b>Patience:</b>", self.patience_spin)

        self.epochs_spin.setToolTip(
            "Training epochs. More = slower but potentially better.\nDefault: 50."
        )
        self.batch_spin.setToolTip(
            "Mini-batch size.\nTiny CNN: 32 typical.  YOLO: 16 on GPU, 8 on CPU."
        )
        self.lr_spin.setToolTip(
            "Initial learning rate. 0.001 is a robust default for both Tiny and YOLO."
        )
        self.val_fraction_spin.setToolTip(
            "Fraction of labeled images reserved for validation (0.2 = 20%).\n"
            "Set to 0 only if you have very few labels."
        )
        self.patience_spin.setToolTip(
            "Early stopping: halt if val accuracy doesn't improve for N consecutive epochs.\n"
            "Set to 0 to disable early stopping."
        )

    @staticmethod
    def _safe_class_text(value: object) -> str:
        if value is None or isinstance(value, bool):
            return ""
        return str(value).strip()

    def _build_class_combo(self, initial_text: object = "") -> QComboBox:
        combo = QComboBox()
        combo.setMinimumWidth(160)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)

        if self._class_choices:
            for cls in self._class_choices:
                combo.addItem(cls, cls)
        else:
            combo.addItem("(no classes)", "")
            combo.setEnabled(False)

        initial = self._safe_class_text(initial_text)
        if initial:
            idx = combo.findData(initial)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        return combo

    def _add_reverse_mapping(
        self,
        rows: List[Tuple[QComboBox, QComboBox]],
        add_row,
        src: QComboBox,
        dst: QComboBox,
    ) -> None:
        src_name = self._safe_class_text(src.currentData() or src.currentText())
        dst_name = self._safe_class_text(dst.currentData() or dst.currentText())
        if not src_name or not dst_name or src_name == dst_name:
            return
        for row_src, row_dst in rows:
            row_src_name = self._safe_class_text(
                row_src.currentData() or row_src.currentText()
            )
            row_dst_name = self._safe_class_text(
                row_dst.currentData() or row_dst.currentText()
            )
            if row_src_name == dst_name and row_dst_name == src_name:
                return
        add_row(dst_name, src_name)

    def _add_mapping_row(
        self,
        rows: List[Tuple[QComboBox, QComboBox]],
        rows_layout: QVBoxLayout,
        add_row,
        src_text: str = "",
        dst_text: str = "",
    ) -> None:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        src = self._build_class_combo(src_text)
        dst = self._build_class_combo(dst_text)
        add_rev = QPushButton("Reverse")
        add_rev.setFixedWidth(72)
        add_rev.setToolTip("Add the reverse mapping (destination \u2192 source)")
        remove_btn = QPushButton("Remove")
        remove_btn.setFixedWidth(72)
        remove_btn.setStyleSheet("color:#c66;")

        row_layout.addWidget(src, 1)
        row_layout.addWidget(QLabel("\u2192"))
        row_layout.addWidget(dst, 1)
        row_layout.addWidget(add_rev)
        row_layout.addWidget(remove_btn)
        rows_layout.addWidget(row_widget)

        pair = (src, dst)
        rows.append(pair)

        def _remove_row() -> None:
            if pair in rows:
                rows.remove(pair)
            row_widget.setParent(None)

        remove_btn.clicked.connect(_remove_row)
        add_rev.clicked.connect(
            lambda: self._add_reverse_mapping(rows, add_row, src, dst)
        )

    def _build_mapping_section(
        self,
        layout: QVBoxLayout,
        title: str,
        add_row,
    ) -> tuple[QWidget, QVBoxLayout]:
        header_row = QHBoxLayout()
        header_row.addWidget(QLabel(title))
        add_btn = QPushButton("+ Add pair")
        add_btn.setFixedWidth(90)
        add_btn.clicked.connect(lambda _checked=False: add_row())
        header_row.addWidget(add_btn)
        header_row.addStretch()
        layout.addLayout(header_row)

        rows_widget = QWidget()
        rows_layout = QVBoxLayout(rows_widget)
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setSpacing(3)
        layout.addWidget(rows_widget)
        return rows_widget, rows_layout

    def _build_expansion_group(self) -> QGroupBox:
        exp_group = QGroupBox("Label-Switching Expansion  (advanced)")
        exp_group.setCheckable(True)
        exp_group.setChecked(False)
        exp_group.setToolTip(
            "Generate deterministic mirrored copies of training images with remapped labels.\n"
            "Useful when a flip transforms one class into another (e.g. 'left' \u2194 'right'\n"
            "on a horizontal flip).  Expanded copies are only added to the train split."
        )
        exp_vbox = QVBoxLayout(exp_group)

        exp_note = QLabel(
            "<i>Each row: source label \u2192 destination label when that flip is applied.<br>"
            "Pairs are applied to train images only; evaluation data is never flipped.<br>"
            "When enabled, all stochastic augmentations (flip/rotate) are disabled.</i>"
        )
        exp_note.setWordWrap(True)
        exp_note.setStyleSheet("color:#aaa; font-size:11px;")
        exp_vbox.addWidget(exp_note)

        self._exp_constraints_label = QLabel("")
        self._exp_constraints_label.setWordWrap(True)
        self._exp_constraints_label.setStyleSheet("color:#e0c070; font-size:11px;")
        exp_vbox.addWidget(self._exp_constraints_label)

        self._lr_mapping_rows = []
        self._ud_mapping_rows = []

        def _add_lr_row(src_text="", dst_text=""):
            self._add_mapping_row(
                self._lr_mapping_rows,
                self._lr_rows_layout,
                _add_lr_row,
                src_text,
                dst_text,
            )

        self._lr_rows_widget, self._lr_rows_layout = self._build_mapping_section(
            exp_vbox,
            "<b>Horizontal flip (LR) mappings:</b>",
            _add_lr_row,
        )
        self._add_lr_row = _add_lr_row

        def _add_ud_row(src_text="", dst_text=""):
            self._add_mapping_row(
                self._ud_mapping_rows,
                self._ud_rows_layout,
                _add_ud_row,
                src_text,
                dst_text,
            )

        self._ud_rows_widget, self._ud_rows_layout = self._build_mapping_section(
            exp_vbox,
            "<b>Vertical flip (UD) mappings:</b>",
            _add_ud_row,
        )
        self._add_ud_row = _add_ud_row
        return exp_group

    def _build_tiny_architecture_tab(self) -> None:
        self.tiny_tab = QWidget()
        tiny_layout = QVBoxLayout(self.tiny_tab)
        tiny_form = QFormLayout()

        self.tiny_layers_spin = QSpinBox()
        self.tiny_layers_spin.setRange(1, 10)
        self.tiny_layers_spin.setValue(1)
        tiny_form.addRow("<b>Hidden Layers:</b>", self.tiny_layers_spin)

        self.tiny_dim_spin = QSpinBox()
        self.tiny_dim_spin.setRange(16, 1024)
        self.tiny_dim_spin.setValue(64)
        self.tiny_dim_spin.setSingleStep(16)
        tiny_form.addRow("<b>Hidden Dim:</b>", self.tiny_dim_spin)

        self.tiny_dropout_spin = QDoubleSpinBox()
        self.tiny_dropout_spin.setRange(0.0, 0.9)
        self.tiny_dropout_spin.setValue(0.2)
        self.tiny_dropout_spin.setSingleStep(0.05)
        tiny_form.addRow("<b>Dropout:</b>", self.tiny_dropout_spin)

        self.tiny_width_spin = QSpinBox()
        self.tiny_width_spin.setRange(32, 512)
        self.tiny_width_spin.setValue(128)
        self.tiny_width_spin.setSingleStep(32)
        tiny_form.addRow("<b>Input Width:</b>", self.tiny_width_spin)

        self.tiny_height_spin = QSpinBox()
        self.tiny_height_spin.setRange(32, 512)
        self.tiny_height_spin.setValue(64)
        self.tiny_height_spin.setSingleStep(32)
        tiny_form.addRow("<b>Input Height:</b>", self.tiny_height_spin)

        self.tiny_rebalance_combo = QComboBox()
        self.tiny_rebalance_combo.addItem("None", "none")
        self.tiny_rebalance_combo.addItem("Weighted Loss", "weighted_loss")
        self.tiny_rebalance_combo.addItem("Weighted Sampler", "weighted_sampler")
        self.tiny_rebalance_combo.addItem("Weighted Loss + Sampler", "both")
        self.tiny_rebalance_combo.setCurrentIndex(1)
        tiny_form.addRow("<b>Class Rebalancing:</b>", self.tiny_rebalance_combo)

        self.tiny_rebalance_power_spin = QDoubleSpinBox()
        self.tiny_rebalance_power_spin.setRange(0.0, 3.0)
        self.tiny_rebalance_power_spin.setSingleStep(0.1)
        self.tiny_rebalance_power_spin.setValue(1.0)
        self.tiny_rebalance_power_spin.setDecimals(2)
        tiny_form.addRow("<b>Rebalance Power:</b>", self.tiny_rebalance_power_spin)

        self.tiny_label_smoothing_spin = QDoubleSpinBox()
        self.tiny_label_smoothing_spin.setRange(0.0, 0.4)
        self.tiny_label_smoothing_spin.setSingleStep(0.01)
        self.tiny_label_smoothing_spin.setValue(0.0)
        self.tiny_label_smoothing_spin.setDecimals(2)
        tiny_form.addRow("<b>Label Smoothing:</b>", self.tiny_label_smoothing_spin)

        self.tiny_layers_spin.setToolTip("Number of hidden layers in the MLP head.")
        self.tiny_dim_spin.setToolTip(
            "Neurons per hidden layer. Larger = more capacity, slower."
        )
        self.tiny_dropout_spin.setToolTip(
            "Dropout regularization probability (0 = disabled)."
        )
        self.tiny_width_spin.setToolTip(
            "Input image width (px). Match to your typical crop size."
        )
        self.tiny_height_spin.setToolTip(
            "Input image height (px). Match to your typical crop size."
        )
        self.tiny_rebalance_combo.setToolTip(
            "Class imbalance handling for Tiny CNN training.\n"
            "Weighted Loss improves minority-class gradients.\n"
            "Weighted Sampler oversamples minority classes per mini-batch."
        )
        self.tiny_rebalance_power_spin.setToolTip(
            "Strength of rebalancing (0 disables effect, 1 = inverse-frequency baseline)."
        )
        self.tiny_label_smoothing_spin.setToolTip(
            "Cross-entropy label smoothing for Tiny CNN training (0.0 = disabled)."
        )

        tiny_layout.addLayout(tiny_form)
        tiny_layout.addStretch()
        self._tiny_tab_idx = self.tabs.addTab(self.tiny_tab, "Tiny Architecture")

    def _build_custom_cnn_tab(self) -> None:
        self.custom_tab = QWidget()
        custom_form = QFormLayout(self.custom_tab)
        custom_form.setSpacing(8)

        self._custom_backbone_combo = QComboBox()
        for key in TORCHVISION_BACKBONES:
            if key == "tinyclassifier":
                self._custom_backbone_combo.addItem("TinyClassifier (scratch)", key)
                self._custom_backbone_combo.insertSeparator(
                    self._custom_backbone_combo.count()
                )
            else:
                self._custom_backbone_combo.addItem(BACKBONE_DISPLAY_NAMES[key], key)
        custom_form.addRow("Backbone:", self._custom_backbone_combo)

        self._tiny_in_custom_width_spin = QSpinBox()
        self._tiny_in_custom_width_spin.setRange(32, 512)
        self._tiny_in_custom_width_spin.setValue(128)
        custom_form.addRow("Input width (px):", self._tiny_in_custom_width_spin)

        self._tiny_in_custom_height_spin = QSpinBox()
        self._tiny_in_custom_height_spin.setRange(32, 512)
        self._tiny_in_custom_height_spin.setValue(64)
        custom_form.addRow("Input height (px):", self._tiny_in_custom_height_spin)

        self._tiny_in_custom_layers_spin = QSpinBox()
        self._tiny_in_custom_layers_spin.setRange(0, 4)
        self._tiny_in_custom_layers_spin.setValue(1)
        custom_form.addRow("Hidden layers:", self._tiny_in_custom_layers_spin)

        self._tiny_in_custom_dim_spin = QSpinBox()
        self._tiny_in_custom_dim_spin.setRange(16, 512)
        self._tiny_in_custom_dim_spin.setValue(64)
        custom_form.addRow("Hidden dim:", self._tiny_in_custom_dim_spin)

        self._tiny_in_custom_dropout_spin = QDoubleSpinBox()
        self._tiny_in_custom_dropout_spin.setRange(0.0, 0.9)
        self._tiny_in_custom_dropout_spin.setSingleStep(0.05)
        self._tiny_in_custom_dropout_spin.setValue(0.2)
        custom_form.addRow("Dropout:", self._tiny_in_custom_dropout_spin)

        self._custom_trainable_layers_label = QLabel(
            "Trainable layers (0=frozen, -1=all):"
        )
        self._custom_trainable_layers_spin = QSpinBox()
        self._custom_trainable_layers_spin.setRange(-1, 8)
        self._custom_trainable_layers_spin.setValue(0)
        self._custom_trainable_layers_spin.setToolTip(
            "0=frozen backbone, -1=all layers, N=last N layer groups unfrozen"
        )
        custom_form.addRow(
            self._custom_trainable_layers_label, self._custom_trainable_layers_spin
        )

        self._custom_backbone_lr_label = QLabel("Backbone LR scale:")
        self._custom_backbone_lr_spin = QDoubleSpinBox()
        self._custom_backbone_lr_spin.setRange(0.001, 1.0)
        self._custom_backbone_lr_spin.setSingleStep(0.01)
        self._custom_backbone_lr_spin.setDecimals(3)
        self._custom_backbone_lr_spin.setValue(0.1)
        self._custom_backbone_lr_spin.setToolTip(
            "LR multiplier applied to unfrozen backbone layers (head LR is full LR)."
        )
        custom_form.addRow(
            self._custom_backbone_lr_label, self._custom_backbone_lr_spin
        )

        self._custom_input_size_label = QLabel("Input size (px, square):")
        self._custom_input_size_spin = QSpinBox()
        self._custom_input_size_spin.setRange(32, 512)
        self._custom_input_size_spin.setSingleStep(32)
        self._custom_input_size_spin.setValue(224)
        self._custom_input_size_spin.setToolTip(
            "Resize crops to this square size before passing to torchvision backbone."
        )
        custom_form.addRow(self._custom_input_size_label, self._custom_input_size_spin)

        self._custom_epochs_spin = QSpinBox()
        self._custom_epochs_spin.setRange(1, 500)
        self._custom_epochs_spin.setValue(50)
        custom_form.addRow("Epochs:", self._custom_epochs_spin)

        self._custom_batch_spin = QSpinBox()
        self._custom_batch_spin.setRange(1, 256)
        self._custom_batch_spin.setValue(32)
        custom_form.addRow("Batch size:", self._custom_batch_spin)

        self._custom_lr_spin = QDoubleSpinBox()
        self._custom_lr_spin.setRange(1e-5, 0.1)
        self._custom_lr_spin.setSingleStep(0.0001)
        self._custom_lr_spin.setDecimals(5)
        self._custom_lr_spin.setValue(1e-3)
        custom_form.addRow("Learning rate:", self._custom_lr_spin)

        self._custom_patience_spin = QSpinBox()
        self._custom_patience_spin.setRange(1, 100)
        self._custom_patience_spin.setValue(10)
        custom_form.addRow("Patience:", self._custom_patience_spin)

        self._custom_tab_idx = self.tabs.addTab(self.custom_tab, "Custom CNN")

        self._custom_backbone_combo.activated.connect(self._on_custom_backbone_changed)
        self._custom_trainable_layers_spin.valueChanged.connect(
            self._on_custom_backbone_changed
        )
        self._on_custom_backbone_changed()

    def _build_space_and_augmentations_tab(self) -> None:
        self.aug_tab = QWidget()
        aug_tab_layout = QVBoxLayout(self.aug_tab)
        aug_tab_layout.setContentsMargins(0, 0, 0, 0)
        self._aug_scroll = QScrollArea()
        self._aug_scroll.setWidgetResizable(True)
        self._aug_scroll.setFrameShape(QFrame.NoFrame)
        aug_scroll_content = QWidget()
        aug_layout = QVBoxLayout(aug_scroll_content)

        aug_group = QGroupBox("Augmentations")
        aug_layout.addWidget(aug_group)
        aug_form = QFormLayout(aug_group)

        self.flip_lr_spin = QDoubleSpinBox()
        self.flip_lr_spin.setRange(0.0, 1.0)
        self.flip_lr_spin.setValue(0.5)
        aug_form.addRow("<b>Horizontal Flip Prob:</b>", self.flip_lr_spin)

        self.flip_ud_spin = QDoubleSpinBox()
        self.flip_ud_spin.setRange(0.0, 1.0)
        self.flip_ud_spin.setValue(0.0)
        aug_form.addRow("<b>Vertical Flip Prob:</b>", self.flip_ud_spin)

        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(0.0, 180.0)
        self.rotate_spin.setValue(0.0)
        aug_form.addRow("<b>Max Rotation (deg):</b>", self.rotate_spin)

        self._exp_group = self._build_expansion_group()
        aug_layout.addWidget(self._exp_group)
        aug_layout.addStretch()
        self._aug_scroll.setWidget(aug_scroll_content)
        aug_tab_layout.addWidget(self._aug_scroll)
        self.tabs.addTab(self.aug_tab, "Space and Augmentations")

    def _build_log_panel(self, layout: QVBoxLayout) -> None:
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)
        self.log_view.setStyleSheet(
            "background:#111; color:#ccc; font-family:monospace; font-size:11px;"
        )
        layout.addWidget(self.log_view)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

    def _build_action_row(self, layout: QVBoxLayout) -> None:
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.publish_btn = QPushButton("Publish to models/")
        self.publish_btn.setEnabled(False)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.publish_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        total = self._scheme.total_classes if self._scheme else "?"
        name = self._scheme.name if self._scheme else "free-form"
        header = QLabel(
            f"<h2 style='color:#ffffff;margin:0'>Train Classifier</h2>"
            f"<p style='color:#888;margin:4px 0 0 0'>Scheme: <b>{name}</b>"
            f" &nbsp;|&nbsp; Classes: <b>{total}</b>"
            f" &nbsp;|&nbsp; Labeled samples: <b>{self._n_labeled}</b></p>"
        )
        header.setStyleSheet("background: #252526; padding: 10px; border-radius: 4px;")
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_general_tab(), "General")
        self._build_tiny_architecture_tab()
        self._build_custom_cnn_tab()
        self._build_space_and_augmentations_tab()

        layout.addWidget(self.tabs, 1)
        self._build_log_panel(layout)
        self._build_action_row(layout)

        self.cancel_btn.clicked.connect(self._on_cancel)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.tiny_rebalance_combo.currentIndexChanged.connect(
            self._on_rebalance_mode_changed
        )
        self._on_mode_changed()
        self._on_rebalance_mode_changed()

        self._exp_group.toggled.connect(
            lambda _checked: self._sync_expansion_constraints()
        )
        self._sync_expansion_constraints()

    def _populate_modes(self):
        self.mode_combo.clear()
        if self._scheme is None:
            self.mode_combo.addItem("Flat - Tiny CNN", "flat_tiny")
            self.mode_combo.addItem("Flat - YOLO-classify", "flat_yolo")
            self.mode_combo.addItem("Flat - Custom CNN", "flat_custom")
            return
        labels = {
            "flat_tiny": "Flat - Tiny CNN",
            "flat_yolo": "Flat - YOLO-classify",
            "flat_custom": "Flat - Custom CNN",
            "multihead_tiny": "Multi-head - Tiny CNN (one model per factor)",
            "multihead_yolo": "Multi-head - YOLO-classify (one model per factor)",
            "multihead_custom": "Multi-head - Custom CNN (one model per factor)",
        }
        for key in [
            "flat_tiny",
            "flat_yolo",
            "flat_custom",
            "multihead_tiny",
            "multihead_yolo",
            "multihead_custom",
        ]:
            if key in self._scheme.training_modes:
                self.mode_combo.addItem(labels[key], key)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData() or ""
        is_yolo = "yolo" in mode
        is_tiny = "tiny" in mode
        is_custom = "custom" in mode

        self.base_model_combo.setVisible(is_yolo)
        if hasattr(self, "_base_model_row_label"):
            self._base_model_row_label.setVisible(is_yolo)

        if hasattr(self, "_tiny_tab_idx"):
            self.tabs.setTabVisible(self._tiny_tab_idx, is_tiny)
            if not is_tiny and self.tabs.currentIndex() == self._tiny_tab_idx:
                self.tabs.setCurrentIndex(0)

        if hasattr(self, "_custom_tab_idx"):
            self.tabs.setTabVisible(self._custom_tab_idx, is_custom)
            if not is_custom and self.tabs.currentIndex() == self._custom_tab_idx:
                self.tabs.setCurrentIndex(0)

        _desc = {
            "flat_tiny": (
                "Tiny CNN \u2014 lightweight MLP trained on image crops. Fast to train; "
                "ideal for rapid iteration and CPU-only environments."
            ),
            "flat_yolo": (
                "YOLO Classify \u2014 fine-tuned pretrained backbone. Higher accuracy; "
                "GPU strongly recommended. Slower to train."
            ),
            "multihead_tiny": (
                "Multi-head Tiny CNN \u2014 one independent model per factor in the labeling scheme. "
                "Each factor is trained separately."
            ),
            "multihead_yolo": (
                "Multi-head YOLO Classify \u2014 one fine-tuned backbone per factor. "
                "Highest accuracy; GPU required."
            ),
            "flat_custom": (
                "Custom CNN \u2014 TinyClassifier or pretrained torchvision backbone "
                "(ConvNeXt, EfficientNet, ResNet, ViT). Configurable layer freezing "
                "for linear-probe or full fine-tuning."
            ),
            "multihead_custom": (
                "Multi-head Custom CNN \u2014 one backbone per factor, with configurable "
                "layer freezing. GPU recommended for torchvision backbones."
            ),
        }
        if hasattr(self, "_mode_desc_label"):
            self._mode_desc_label.setText(_desc.get(mode, ""))

    def _on_custom_backbone_changed(self) -> None:
        backbone = self._custom_backbone_combo.currentData()
        is_tiny = backbone == "tinyclassifier"

        for w in (
            self._tiny_in_custom_width_spin,
            self._tiny_in_custom_height_spin,
            self._tiny_in_custom_layers_spin,
            self._tiny_in_custom_dim_spin,
            self._tiny_in_custom_dropout_spin,
        ):
            w.setVisible(is_tiny)

        for w in (
            self._custom_trainable_layers_spin,
            self._custom_trainable_layers_label,
            self._custom_input_size_spin,
            self._custom_input_size_label,
        ):
            w.setVisible(not is_tiny)

        show_lr = not is_tiny and self._custom_trainable_layers_spin.value() != 0
        self._custom_backbone_lr_spin.setVisible(show_lr)
        self._custom_backbone_lr_label.setVisible(show_lr)

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_rebalance_mode_changed(self):
        mode = str(self.tiny_rebalance_combo.currentData() or "none")
        enable_power = mode != "none"
        self.tiny_rebalance_power_spin.setEnabled(enable_power)

    def _sync_expansion_constraints(self) -> None:
        expansion_enabled = bool(self._exp_group.isChecked())

        for widget in (self.flip_lr_spin, self.flip_ud_spin, self.rotate_spin):
            widget.setEnabled(not expansion_enabled)

        if expansion_enabled:
            self.flip_lr_spin.setValue(0.0)
            self.flip_ud_spin.setValue(0.0)
            self.rotate_spin.setValue(0.0)
            self._exp_constraints_label.setText(
                "Label expansion ON: all random augmentations are disabled."
            )
        else:
            self._exp_constraints_label.setText(
                "Label expansion OFF: choose augmentation probabilities freely."
            )

    def append_log(self, msg: str):
        if msg:
            self.log_view.appendPlainText(msg)
            self.log_view.ensureCursorVisible()

    def get_settings(self) -> dict:
        label_expansion: dict = {}

        def _combo_text(cb: QComboBox) -> str:
            return str(cb.currentData() or cb.currentText() or "").strip()

        if self._exp_group.isChecked():
            lr_map = {
                _combo_text(s): _combo_text(d)
                for s, d in self._lr_mapping_rows
                if _combo_text(s)
                and _combo_text(d)
                and _combo_text(s) != _combo_text(d)
            }
            if lr_map:
                label_expansion["fliplr"] = lr_map
            ud_map = {
                _combo_text(s): _combo_text(d)
                for s, d in self._ud_mapping_rows
                if _combo_text(s)
                and _combo_text(d)
                and _combo_text(s) != _combo_text(d)
            }
            if ud_map:
                label_expansion["flipud"] = ud_map

        expansion_enabled = bool(self._exp_group.isChecked())

        flipud_value = 0.0 if expansion_enabled else self.flip_ud_spin.value()
        fliplr_value = 0.0 if expansion_enabled else self.flip_lr_spin.value()
        rotate_value = 0.0 if expansion_enabled else self.rotate_spin.value()

        _rt = str(self.compute_runtime_combo.currentData() or "cpu")
        _train_device = str(self.device_combo.currentData() or "cpu")
        if _train_device == "rocm":
            _train_device = "cuda"

        _mode = self.mode_combo.currentData() or ""
        _is_custom = _mode in ("flat_custom", "multihead_custom")

        return {
            "mode": self.mode_combo.currentData(),
            "compute_runtime": _rt,
            "device": _train_device,
            "base_model": self.base_model_combo.currentData(),
            "epochs": (
                self._custom_epochs_spin.value()
                if _is_custom
                else self.epochs_spin.value()
            ),
            "batch": (
                self._custom_batch_spin.value()
                if _is_custom
                else self.batch_spin.value()
            ),
            "lr": (
                self._custom_lr_spin.value() if _is_custom else self.lr_spin.value()
            ),
            "val_fraction": self.val_fraction_spin.value(),
            "patience": (
                self._custom_patience_spin.value()
                if _is_custom
                else self.patience_spin.value()
            ),
            "tiny_layers": self.tiny_layers_spin.value(),
            "tiny_dim": self.tiny_dim_spin.value(),
            "tiny_dropout": self.tiny_dropout_spin.value(),
            "tiny_width": self.tiny_width_spin.value(),
            "tiny_height": self.tiny_height_spin.value(),
            "tiny_rebalance_mode": self.tiny_rebalance_combo.currentData(),
            "tiny_rebalance_power": self.tiny_rebalance_power_spin.value(),
            "tiny_label_smoothing": self.tiny_label_smoothing_spin.value(),
            "custom_backbone": self._custom_backbone_combo.currentData(),
            "custom_trainable_layers": self._custom_trainable_layers_spin.value(),
            "custom_backbone_lr_scale": self._custom_backbone_lr_spin.value(),
            "custom_input_size": self._custom_input_size_spin.value(),
            "flipud": flipud_value,
            "fliplr": fliplr_value,
            "rotate": rotate_value,
            "label_expansion": label_expansion,
        }
