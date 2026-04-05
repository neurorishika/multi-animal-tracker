"""ExportDialog — configure dataset export for ClassKit."""

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811


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
            "\u2022 ImageFolder/Ultralytics export directory trees by split and class.<br>"
            "\u2022 CSV/Parquet include split and class metadata columns.<br>"
            "\u2022 Fractions are applied with a fixed seed for reproducible splits."
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
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Export Output Directory",
            self.output_edit.text() or str(Path.home()),
        )
        if folder:
            self.output_edit.setText(folder)

    def _validate_and_accept(self):
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
        return {
            "format": self.format_combo.currentData(),
            "output_dir": self.output_edit.text().strip(),
            "copy_files": self.copy_check.isChecked(),
            "include_unlabeled": self.include_unlabeled_check.isChecked(),
            "val_fraction": float(self.val_fraction_spin.value()),
            "test_fraction": float(self.test_fraction_spin.value()),
        }
