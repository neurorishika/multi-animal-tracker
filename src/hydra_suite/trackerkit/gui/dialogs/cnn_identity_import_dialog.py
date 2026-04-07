"""Dialog for importing a ClassKit-trained model for CNN identity classification."""

from __future__ import annotations

from PySide6.QtWidgets import QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QWidget

from hydra_suite.widgets import BaseDialog


class CNNIdentityImportDialog(BaseDialog):
    """Pre-filled dialog for user to verify and annotate CNN identity model import."""

    def __init__(self, meta: dict, parent=None) -> None:
        super().__init__(
            "Import CNN Identity Model",
            parent=parent,
            buttons=QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
        )

        form_widget = QWidget()
        layout = QFormLayout(form_widget)

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

        self.add_content(form_widget)

    def species(self) -> str:
        """Return the entered species name, defaulting to 'unknown' if blank."""
        return self._species_edit.text().strip() or "unknown"

    def classification_label(self) -> str:
        """Return the optional classification tag entered for the imported model."""
        return self._label_edit.text().strip()
