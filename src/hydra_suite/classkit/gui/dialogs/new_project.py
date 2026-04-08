"""NewProjectDialog — dialog for creating a new ClassKit project."""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.classkit.config.presets import (
    flatten_scheme_labels,
    get_available_scheme_presets,
)
from hydra_suite.classkit.gui.dialogs._helpers import _SchemeWrapper
from hydra_suite.classkit.gui.dialogs.class_editor import ClassEditorDialog
from hydra_suite.classkit.gui.project import default_project_parent_dir
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811
from hydra_suite.widgets.dialogs import BaseDialog

_BTN_NEUTRAL = (
    "QPushButton { background-color:#3e3e42; color:#e0e0e0; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#555558; }"
)


class NewProjectDialog(BaseDialog):
    """Dialog for creating a new ClassKit project."""

    def __init__(self, parent=None) -> None:
        super().__init__("Create New Project", parent=parent)
        self.setMinimumWidth(580)

        self._custom_scheme: Optional[dict] = None
        self._preset_lookup: dict[str, object] = {}

        content = QWidget(self)
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        header = QLabel("<h2 style='margin:0;'>Create New Project</h2>")
        intro = QLabel(
            "Choose the project folder first, then pick a starter labeling scheme or define one in full."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(header)
        layout.addWidget(intro)

        details_group = QGroupBox("Project Details")
        details_layout = QVBoxLayout(details_group)
        details_layout.setSpacing(10)
        details_form = QFormLayout()
        details_form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("MyDataset")
        self.name_edit.textChanged.connect(self._update_project_preview)
        self.name_edit.textChanged.connect(self._validate)
        details_form.addRow("Project Name", self.name_edit)

        location_row = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Select project location\u2026")
        self.location_edit.setText(str(default_project_parent_dir()))
        self.location_edit.textChanged.connect(self._update_project_preview)
        self.location_edit.textChanged.connect(self._validate)
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setMaximumWidth(90)
        browse_btn.clicked.connect(self._browse_location)
        location_row.addWidget(self.location_edit, 1)
        location_row.addWidget(browse_btn)
        details_form.addRow("Location", location_row)
        details_layout.addLayout(details_form)

        preview_label = QLabel("Project Folder")
        preview_label.setStyleSheet("color: #cfcfcf;")
        details_layout.addWidget(preview_label)

        self.project_preview = QLabel()
        self.project_preview.setWordWrap(True)
        self.project_preview.setStyleSheet(
            "padding: 10px; background-color: #252526; border-radius: 6px; "
            "border-left: 3px solid #0e639c;"
        )
        details_layout.addWidget(self.project_preview)
        layout.addWidget(details_group)

        scheme_group = QGroupBox("Labeling Setup")
        scheme_vlayout = QVBoxLayout(scheme_group)
        scheme_vlayout.setSpacing(10)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self._populate_presets()
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        preset_row.addWidget(QLabel("Quick preset:"))
        preset_row.addWidget(self.preset_combo, 1)
        scheme_vlayout.addLayout(preset_row)

        self._scheme_info = QLabel("")
        self._scheme_info.setWordWrap(True)
        self._scheme_info.setStyleSheet("color:#888; font-size:11px; padding-left:4px;")
        scheme_vlayout.addWidget(self._scheme_info)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color:#3e3e42;")
        scheme_vlayout.addWidget(div)

        editor_row = QHBoxLayout()
        editor_row.addWidget(QLabel("Or open the full scheme editor:"))
        editor_row.addStretch(1)
        self._btn_define_scheme = QPushButton("Define Full Scheme\u2026")
        self._btn_define_scheme.setStyleSheet(_BTN_NEUTRAL)
        self._btn_define_scheme.clicked.connect(self._open_scheme_editor)
        editor_row.addWidget(self._btn_define_scheme)
        scheme_vlayout.addLayout(editor_row)

        self._custom_scheme_lbl = QLabel("")
        self._custom_scheme_lbl.setStyleSheet(
            "color:#4ec94e; font-size:11px; padding-left:4px;"
        )
        self._custom_scheme_lbl.setWordWrap(True)
        scheme_vlayout.addWidget(self._custom_scheme_lbl)

        layout.addWidget(scheme_group)

        info = QLabel(
            "You can refine the labeling scheme later from the project workspace. ClassKit will create the project folder at the selected location."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding:10px; background-color:#252526; border-radius:6px; "
            "border-left:3px solid #0e639c; color:#aaaaaa; line-height:1.6;"
        )
        layout.addWidget(info)

        self.add_content(content)

        create_button = self._buttons.button(QDialogButtonBox.Ok)
        create_button.setText("Create Project")

        self._validate()
        self._update_project_preview()
        self._on_preset_changed()

    def _on_preset_changed(self):
        key = self.preset_combo.currentData()
        preset = self._preset_lookup.get(key)
        if key == "none":
            self._scheme_info.setText(
                "Free-form — define labels manually after the project is created."
            )
        elif preset is not None:
            self._scheme_info.setText(getattr(preset, "description", ""))
        else:
            self._scheme_info.setText("")
        if self._custom_scheme is not None:
            self._custom_scheme = None
            self._custom_scheme_lbl.setText("")
            self._btn_define_scheme.setText("Define Full Scheme\u2026")

    def _open_scheme_editor(self):
        dlg = ClassEditorDialog(parent=self)
        if dlg.exec():
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(0)
            self.preset_combo.blockSignals(False)
            self._populate_presets()
            self._custom_scheme = dlg.get_scheme_dict()
            factors = self._custom_scheme.get("factors", [])
            total_labels = sum(len(f.get("labels", [])) for f in factors)
            factor_names = ", ".join(f.get("name", "?") for f in factors)
            self._custom_scheme_lbl.setText(
                f"\u2713 Custom scheme: {len(factors)} factor(s)  \u00b7  {total_labels} labels  "
                f"({factor_names})"
            )
            self._btn_define_scheme.setText("Edit Full Scheme\u2026")
            self._scheme_info.setText(
                "Custom scheme selected. Edit it again to change factors or labels."
            )

    def _populate_presets(self) -> None:
        current_key = (
            self.preset_combo.currentData() if self.preset_combo.count() else "none"
        )
        self._preset_lookup = {}
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("None — define manually after creation", "none")
        for preset in get_available_scheme_presets():
            self._preset_lookup[preset.key] = preset
            self.preset_combo.addItem(preset.label, preset.key)
        target_index = self.preset_combo.findData(current_key)
        self.preset_combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        self.preset_combo.blockSignals(False)

    def _browse_location(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Location",
            self.location_edit.text() or str(default_project_parent_dir()),
        )
        if folder:
            self.location_edit.setText(folder)

    def _validate(self):
        ok = self._buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(
            bool(self.name_edit.text().strip())
            and bool(self.location_edit.text().strip())
        )

    def _update_project_preview(self) -> None:
        project_path = self.get_project_path()
        if project_path is None:
            self.project_preview.setText(
                "Project folder: choose a location and enter a project name."
            )
            return
        self.project_preview.setText(f"Project folder: {project_path}")

    def get_project_path(self) -> Path | None:
        """Return the full ClassKit project directory, or None if incomplete."""
        name = self.name_edit.text().strip()
        location = self.location_edit.text().strip()
        if not name or not location:
            return None
        return Path(location).expanduser() / name

    def get_project_info(self) -> dict:
        name = self.name_edit.text().strip()
        project_path = self.get_project_path()

        if self._custom_scheme is not None:
            scheme_dict = self._custom_scheme
            classes = flatten_scheme_labels(scheme_dict)
            return {
                "name": name,
                "path": str(project_path) if project_path is not None else "",
                "classes": classes,
                "scheme": _SchemeWrapper(scheme_dict),
            }

        preset = self._preset_lookup.get(self.preset_combo.currentData())
        scheme_obj = getattr(preset, "scheme", None)
        classes = flatten_scheme_labels(scheme_obj) if scheme_obj is not None else []
        return {
            "name": name,
            "path": str(project_path) if project_path is not None else "",
            "classes": classes,
            "scheme": scheme_obj,
        }
