"""TIMM backbone browser dialog for ClassKit custom CNN training."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class TimmBackboneBrowserDialog(QDialog):
    """Searchable browser for pretrained TIMM image-classification backbones."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add TIMM Backbones")
        self.setMinimumWidth(760)
        self.setMinimumHeight(560)
        self._all_models: list[str] = []
        self._build_ui()
        self._load_models()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QLabel(
            "Choose one or more pretrained TIMM image-classification backbones to add "
            "to the Custom CNN picker. Added models persist across projects and sessions."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Filter:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(
            "Search timm model names, e.g. convnext, dinov2, eva, efficientnet"
        )
        search_row.addWidget(self.search_edit, 1)
        layout.addLayout(search_row)

        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.model_list, 1)

        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)

        button_box = QDialogButtonBox()
        self.add_button = QPushButton("Add Selected")
        self.add_button.setEnabled(False)
        button_box.addButton(self.add_button, QDialogButtonBox.AcceptRole)
        button_box.addButton(QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        self.search_edit.textChanged.connect(self._apply_filter)
        self.model_list.itemSelectionChanged.connect(self._update_selection_state)
        self.model_list.itemDoubleClicked.connect(lambda _item: self.accept())
        self.add_button.clicked.connect(self.accept)
        button_box.rejected.connect(self.reject)

    def _load_models(self) -> None:
        try:
            import timm

            self._all_models = sorted(timm.list_models(pretrained=True))
        except Exception as exc:
            QMessageBox.warning(
                self,
                "TIMM Unavailable",
                f"Unable to list TIMM models.\n\n{exc}",
            )
            self._all_models = []
        self._apply_filter()

    def _apply_filter(self) -> None:
        query = self.search_edit.text().strip().lower()
        self.model_list.clear()
        count = 0
        for model_name in self._all_models:
            if query and query not in model_name.lower():
                continue
            item = QListWidgetItem(f"timm/{model_name}")
            item.setData(Qt.UserRole, f"timm/{model_name}")
            self.model_list.addItem(item)
            count += 1
        self.summary_label.setText(f"{count} TIMM models shown")
        self._update_selection_state()

    def _update_selection_state(self) -> None:
        selected = len(self.model_list.selectedItems())
        self.add_button.setEnabled(selected > 0)
        if selected:
            self.summary_label.setText(
                f"{selected} model{'s' if selected != 1 else ''} selected"
            )

    def selected_backbones(self) -> list[str]:
        return [
            str(item.data(Qt.UserRole))
            for item in self.model_list.selectedItems()
            if str(item.data(Qt.UserRole) or "").startswith("timm/")
        ]
