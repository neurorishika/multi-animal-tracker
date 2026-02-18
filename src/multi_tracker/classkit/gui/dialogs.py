"""
Polished dialogs for ClassKit
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from ..cluster.metalfaiss_backend import probe_metalfaiss_backend


class NewProjectDialog(QDialog):
    """Dialog for creating a new ClassKit project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(500)

        # Apply styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #cccccc;
                font-size: 13px;
            }
            QLineEdit {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
            QTextEdit {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        header = QLabel("<h2 style='color: #ffffff;'>Create New Project</h2>")
        layout.addWidget(header)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("MyDataset")
        form.addRow("<b>Project Name:</b>", self.name_edit)

        # Location selector
        location_layout = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Select project location...")
        self.location_edit.setText(str(Path.home() / "ClassKit" / "projects"))

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_location)

        location_layout.addWidget(self.location_edit, 1)
        location_layout.addWidget(browse_btn)
        form.addRow("<b>Location:</b>", location_layout)

        # Classes
        self.classes_edit = QTextEdit()
        self.classes_edit.setPlaceholderText(
            "Enter class names, one per line:\\ndog\\ncat\\nbird"
        )
        self.classes_edit.setMaximumHeight(100)
        form.addRow("<b>Classes (optional):</b>", self.classes_edit)

        layout.addLayout(form)

        # Info box
        info = QLabel(
            "<b>Tip:</b> You can add or modify classes later.\\n"
            + "The project will be created as a new folder at the specified location."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #aaaaaa; line-height: 1.6;"
        )
        layout.addWidget(info)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Validation
        self.name_edit.textChanged.connect(self.validate_input)
        self.validate_input()

    def browse_location(self):
        """Browse for project location."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Location",
            self.location_edit.text() or str(Path.home()),
        )
        if folder:
            self.location_edit.setText(folder)

    def validate_input(self):
        """Validate form inputs."""
        name = self.name_edit.text().strip()
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        ok_button.setEnabled(len(name) > 0)

    def get_project_info(self):
        """Get project configuration."""
        name = self.name_edit.text().strip()
        location = Path(self.location_edit.text())
        project_path = location / name

        classes_text = self.classes_edit.toPlainText().strip()
        classes = [c.strip() for c in classes_text.split("\\n") if c.strip()]

        return {"name": name, "path": str(project_path), "classes": classes}


class EmbeddingDialog(QDialog):
    """Dialog for configuring embedding computation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute Embeddings")
        self.setMinimumWidth(500)

        # Apply styling (same as NewProjectDialog)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #cccccc;
            }
            QComboBox, QSpinBox {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox:focus, QSpinBox:focus {
                border: 1px solid #007acc;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #cccccc;
                margin-right: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        header = QLabel("<h2 style='color: #ffffff;'>Embedding Configuration</h2>")
        layout.addWidget(header)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        # Model selection
        self.model_combo = QComboBox()
        models = [
            ("DINOv2 Base (Recommended)", "vit_base_patch14_dinov2.lvd142m"),
            ("DINOv2 Small (Faster)", "vit_small_patch14_dinov2.lvd142m"),
            ("DINOv2 Large (Best Quality)", "vit_large_patch14_dinov2.lvd142m"),
            ("CLIP Base", "vit_base_patch32_clip_224.openai"),
            ("ResNet-50 (Fast)", "resnet50.a1_in1k"),
            ("EfficientNet-B3", "efficientnet_b3.ra2_in1k"),
        ]
        for display_name, model_name in models:
            self.model_combo.addItem(display_name, model_name)
        form.addRow("<b>Model:</b>", self.model_combo)

        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")

        try:
            from ...utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

            if TORCH_CUDA_AVAILABLE:
                self.device_combo.addItem("CUDA GPU", "cuda")
            if MPS_AVAILABLE:
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass

        form.addRow("<b>Device:</b>", self.device_combo)

        # Batch size
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        self.batch_spin.setSuffix(" images")
        form.addRow("<b>Batch Size:</b>", self.batch_spin)

        # Force recompute checkbox
        from PySide6.QtWidgets import QCheckBox

        self.force_recompute_check = QCheckBox("Force recompute (ignore cache)")
        self.force_recompute_check.setStyleSheet("color: #cccccc;")
        form.addRow("", self.force_recompute_check)

        layout.addLayout(form)

        # Info
        info_text = (
            "<b>Model Guide:</b><br>"
            + "• <b>DINOv2:</b> Best for general visual understanding<br>"
            + "• <b>CLIP:</b> Good for semantic similarity<br>"
            + "• <b>ResNet/EfficientNet:</b> Faster, less memory<br><br>"
            + "<b>Device:</b> GPU is recommended for faster processing.<br>"
            + "<b>Cache:</b> Embeddings are cached automatically. Uncheck to recompute."
        )
        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #aaaaaa; line-height: 1.8;"
        )
        layout.addWidget(info)

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        """Get embedding settings."""
        model_name = self.model_combo.currentData()
        device = self.device_combo.currentData()
        batch_size = self.batch_spin.value()
        force_recompute = self.force_recompute_check.isChecked()

        return model_name, device, batch_size, force_recompute


class ClusterDialog(QDialog):
    """Dialog for configuring clustering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Configuration")
        self.setMinimumWidth(500)

        # Detect available backends
        import platform

        is_macos = platform.system() == "Darwin"

        metalfaiss_probe = probe_metalfaiss_backend()
        metalfaiss_installed = metalfaiss_probe["installed"]
        metalfaiss_backend_ready = metalfaiss_probe["ready"]

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2>Clustering Settings</h2>")
        layout.addWidget(header)

        form = QFormLayout()

        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(10, 10000)
        self.n_clusters_spin.setValue(500)
        self.n_clusters_spin.setSuffix(" clusters")
        form.addRow("<b>Number of Clusters:</b>", self.n_clusters_spin)

        self.gpu_combo = QComboBox()

        if is_macos:
            if metalfaiss_backend_ready:
                self.gpu_combo.addItem("Metal-accelerated (metalfaiss)", False)
            else:
                self.gpu_combo.addItem("CPU (sklearn - stable)", False)
        else:
            self.gpu_combo.addItem("CPU (FAISS)", False)
            self.gpu_combo.addItem("GPU (FAISS-CUDA)", True)

        form.addRow("<b>Backend:</b>", self.gpu_combo)

        layout.addLayout(form)

        # Build info message based on platform and available backends
        if is_macos:
            if metalfaiss_backend_ready:
                source_note = ""
                if metalfaiss_probe.get("local_path_added"):
                    source_note = "<br><br><b>Source:</b> Local Faiss-mlx checkout detected and loaded."
                info_text = (
                    "<b>Metal FAISS detected.</b><br>"
                    + "Using GPU-accelerated clustering via Apple Metal.<br><br>"
                    + "<b>Tips:</b><br>"
                    + "• Metal is 5-10× faster than CPU clustering<br>"
                    + "• Use 100s-1000s of clusters for best results<br>"
                    + "• Clusters ≠ classes (fine-grained visual structure)"
                    + source_note
                )
            elif metalfaiss_installed:
                details = (
                    metalfaiss_probe.get("error")
                    or "Unknown backend initialization error"
                )
                origin = metalfaiss_probe.get("origin") or "unknown"
                remediation = metalfaiss_probe.get("remediation")
                shadow_note = ""
                if metalfaiss_probe.get("likely_local_shadow"):
                    shadow_note = (
                        "<br><br><b>Likely cause:</b> a local Faiss-mlx checkout is shadowing site-packages "
                        "and missing compiled modules (for example: metalfaiss.index_pointer)."
                    )
                remediation_note = (
                    f"<br><br><b>How to fix:</b> {remediation}" if remediation else ""
                )
                info_text = (
                    "<b>MetalFaiss package detected, but runtime backend is unavailable.</b><br>"
                    + "Clustering will fall back to sklearn in this session.<br><br>"
                    + f"<b>Import source:</b> <code>{origin}</code><br>"
                    + f"<b>Error:</b> <code>{details}</code>"
                    + shadow_note
                    + remediation_note
                )
            else:
                info_text = (
                    "<b>Metal FAISS not installed</b><br>"
                    + "Using sklearn (CPU-based, slower but stable).<br><br>"
                    + "<b>For faster clustering, install:</b><br>"
                    + "<code>pip install mlx metalfaiss</code><br><br>"
                    + "See METALFAISS_SETUP.md for details."
                )
        else:
            info_text = (
                "<b>Tips:</b><br>"
                + "• Use many clusters (100s-1000s) to capture visual modes<br>"
                + "• Clusters ≠ classes (fine-grained structure)<br>"
                + "• GPU mode requires CUDA-enabled FAISS"
            )

        info = QLabel(info_text)
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #aaaaaa; line-height: 1.8;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        """Get cluster settings."""
        return self.n_clusters_spin.value(), self.gpu_combo.currentData()


class TrainingDialog(QDialog):
    """Dialog for configuring embedding-head training."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Classifier")
        self.setMinimumWidth(520)

        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #cccccc; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #007acc;
            }
            QCheckBox { color: #cccccc; }
            """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2 style='color: #ffffff;'>Embedding Head Training</h2>")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(10)

        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("Linear Head (fast)", "linear")
        self.model_type_combo.addItem("MLP Head (stronger)", "mlp")
        form.addRow("<b>Model Type:</b>", self.model_type_combo)

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        try:
            from ...utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

            if TORCH_CUDA_AVAILABLE:
                self.device_combo.addItem("CUDA GPU", "cuda")
            if MPS_AVAILABLE:
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass
        form.addRow("<b>Device:</b>", self.device_combo)

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 4096)
        self.hidden_dim_spin.setValue(512)
        form.addRow("<b>MLP Hidden Dim:</b>", self.hidden_dim_spin)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(0.1)
        form.addRow("<b>Dropout:</b>", self.dropout_spin)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(8, 4096)
        self.batch_size_spin.setValue(256)
        form.addRow("<b>Batch Size:</b>", self.batch_size_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(100)
        form.addRow("<b>Epochs:</b>", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        form.addRow("<b>Learning Rate:</b>", self.lr_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(0.001)
        self.weight_decay_spin.setValue(0.01)
        form.addRow("<b>Weight Decay:</b>", self.weight_decay_spin)

        self.early_stop_spin = QSpinBox()
        self.early_stop_spin.setRange(1, 100)
        self.early_stop_spin.setValue(10)
        form.addRow("<b>Early Stop Patience:</b>", self.early_stop_spin)

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.0, 0.5)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.2)
        form.addRow("<b>Validation Fraction:</b>", self.val_fraction_spin)

        self.calibrate_check = QCheckBox("Run calibration when validation split exists")
        self.calibrate_check.setChecked(True)
        form.addRow("", self.calibrate_check)

        layout.addLayout(form)

        info = QLabel(
            "<b>Notes:</b><br>"
            "• Linear is fastest and works well for small datasets.<br>"
            "• MLP can improve accuracy with enough labels.<br>"
            "• Validation fraction 0 disables validation and calibration."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #aaaaaa; line-height: 1.8;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.model_type_combo.currentIndexChanged.connect(self._on_model_type_changed)
        self._on_model_type_changed()

    def _on_model_type_changed(self):
        """Enable MLP-only options when needed."""
        is_mlp = self.model_type_combo.currentData() == "mlp"
        self.hidden_dim_spin.setEnabled(is_mlp)

    def get_settings(self):
        """Return training hyperparameters from dialog."""
        return {
            "model_type": self.model_type_combo.currentData(),
            "device": self.device_combo.currentData(),
            "hidden_dim": self.hidden_dim_spin.value(),
            "dropout": float(self.dropout_spin.value()),
            "batch_size": self.batch_size_spin.value(),
            "epochs": self.epochs_spin.value(),
            "lr": float(self.lr_spin.value()),
            "weight_decay": float(self.weight_decay_spin.value()),
            "early_stop_patience": self.early_stop_spin.value(),
            "val_fraction": float(self.val_fraction_spin.value()),
            "calibrate": self.calibrate_check.isChecked(),
        }


class ExportDialog(QDialog):
    """Dialog for configuring dataset export."""

    def __init__(self, default_output: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Dataset")
        self.setMinimumWidth(560)

        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #cccccc; }
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
                background-color: #252526;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 6px;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
                border: 1px solid #007acc;
            }
            QCheckBox { color: #cccccc; }
            """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2 style='color:#ffffff;'>Export Labeled Dataset</h2>")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(10)

        self.format_combo = QComboBox()
        self.format_combo.addItem("ImageFolder", "imagefolder")
        self.format_combo.addItem("CSV", "csv")
        self.format_combo.addItem("Parquet", "parquet")
        self.format_combo.addItem("Ultralytics Classify", "ultralytics")
        form.addRow("<b>Format:</b>", self.format_combo)

        output_row = QHBoxLayout()
        self.output_edit = QLineEdit(default_output)
        self.output_edit.setPlaceholderText("Choose output directory...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(browse_btn)
        form.addRow("<b>Output Dir:</b>", output_row)

        self.copy_check = QCheckBox(
            "Copy files (disable to use symlinks where supported)"
        )
        self.copy_check.setChecked(True)
        form.addRow("", self.copy_check)

        self.include_unlabeled_check = QCheckBox(
            "Include unlabeled images (as class 'unlabeled')"
        )
        self.include_unlabeled_check.setChecked(False)
        form.addRow("", self.include_unlabeled_check)

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.0, 0.6)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.2)
        form.addRow("<b>Validation Fraction:</b>", self.val_fraction_spin)

        self.test_fraction_spin = QDoubleSpinBox()
        self.test_fraction_spin.setRange(0.0, 0.4)
        self.test_fraction_spin.setSingleStep(0.05)
        self.test_fraction_spin.setDecimals(2)
        self.test_fraction_spin.setValue(0.0)
        form.addRow("<b>Test Fraction:</b>", self.test_fraction_spin)

        layout.addLayout(form)

        info = QLabel(
            "<b>Export notes:</b><br>"
            "• ImageFolder/Ultralytics export directory trees by split and class.<br>"
            "• CSV/Parquet include split and class metadata columns.<br>"
            "• Fractions are applied with a fixed seed for reproducible splits."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding: 12px; background-color: #252526; border-radius: 6px; "
            + "border-left: 3px solid #0e639c; color: #aaaaaa; line-height: 1.7;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _browse_output(self):
        """Browse for export output directory."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Export Output Directory",
            self.output_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_edit.setText(folder)

    def _validate_and_accept(self):
        """Validate user settings before accepting dialog."""
        output_dir = self.output_edit.text().strip()
        if not output_dir:
            QMessageBox.warning(
                self, "Invalid Output", "Please choose an output directory."
            )
            return

        val_fraction = float(self.val_fraction_spin.value())
        test_fraction = float(self.test_fraction_spin.value())
        if val_fraction + test_fraction >= 0.95:
            QMessageBox.warning(
                self,
                "Invalid Split Fractions",
                "Validation + test fraction must be less than 0.95.",
            )
            return

        self.accept()

    def get_settings(self):
        """Get export settings."""
        return {
            "format": self.format_combo.currentData(),
            "output_dir": self.output_edit.text().strip(),
            "copy_files": self.copy_check.isChecked(),
            "include_unlabeled": self.include_unlabeled_check.isChecked(),
            "val_fraction": float(self.val_fraction_spin.value()),
            "test_fraction": float(self.test_fraction_spin.value()),
        }
