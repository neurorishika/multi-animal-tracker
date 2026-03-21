"""Dialog for importing a ClassKit-trained model for CNN identity classification."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit


class CNNIdentityImportDialog(QDialog):
    """Pre-filled dialog for user to verify and annotate CNN identity model import."""

    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import CNN Identity Model")
        layout = QFormLayout(self)

        # Read-only metadata display
        layout.addRow("Architecture:", QLabel(str(meta.get("arch", "—"))))
        layout.addRow("Num classes:", QLabel(str(meta.get("num_classes", "—"))))
        class_names = meta.get("class_names", [])
        class_preview = ", ".join(class_names[:8])
        if len(class_names) > 8:
            class_preview += f" … ({len(class_names)} total)"
        layout.addRow("Classes:", QLabel(class_preview or "—"))
        layout.addRow("Input size:", QLabel(str(meta.get("input_size", "—"))))

        # Editable fields
        self._species_edit = QLineEdit()
        self._species_edit.setPlaceholderText("e.g. ant")
        layout.addRow("Species:", self._species_edit)

        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("e.g. apriltag, colortag (optional)")
        layout.addRow("Classification label:", self._label_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def species(self) -> str:
        return self._species_edit.text().strip() or "unknown"

    def classification_label(self) -> str:
        return self._label_edit.text().strip()
