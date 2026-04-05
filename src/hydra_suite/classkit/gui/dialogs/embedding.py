"""EmbeddingDialog — configure embedding computation for ClassKit."""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class EmbeddingDialog(QDialog):
    """Dialog for configuring embedding computation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compute Embeddings")
        self.setMinimumWidth(500)

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

        header = QLabel("<h2 style='color: #ffffff;'>Embedding Configuration</h2>")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(12)

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

        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")

        try:
            from hydra_suite.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

            if TORCH_CUDA_AVAILABLE:
                self.device_combo.addItem("CUDA GPU", "cuda")
            if MPS_AVAILABLE:
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass

        form.addRow("<b>Device:</b>", self.device_combo)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        self.batch_spin.setSuffix(" images")
        form.addRow("<b>Batch Size:</b>", self.batch_spin)

        self.force_recompute_check = QCheckBox("Force recompute (ignore cache)")
        self.force_recompute_check.setStyleSheet("color: #cccccc;")
        form.addRow("", self.force_recompute_check)

        layout.addLayout(form)

        info_text = (
            "<b>Model Guide:</b><br>"
            + "\u2022 <b>DINOv2:</b> Best for general visual understanding<br>"
            + "\u2022 <b>CLIP:</b> Good for semantic similarity<br>"
            + "\u2022 <b>ResNet/EfficientNet:</b> Faster, less memory<br><br>"
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

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def get_settings(self):
        model_name = self.model_combo.currentData()
        device = self.device_combo.currentData()
        batch_size = self.batch_spin.value()
        force_recompute = self.force_recompute_check.isChecked()
        return model_name, device, batch_size, force_recompute
