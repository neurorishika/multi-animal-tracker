"""NewProjectDialog — dialog for creating a new ClassKit project."""

from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from hydra_suite.classkit.gui.dialogs._helpers import _SchemeWrapper
from hydra_suite.classkit.gui.dialogs.class_editor import ClassEditorDialog
from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

_DARK_STYLE = """
    QDialog { background-color: #1e1e1e; }
    QGroupBox {
        border: 1px solid #3e3e42; border-radius: 6px;
        margin-top: 12px; padding-top: 12px; color: #cccccc;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
    QLabel { color: #cccccc; }
    QLineEdit, QTextEdit, QPlainTextEdit, QListWidget {
        background-color: #252526; color: #e0e0e0;
        border: 1px solid #3e3e42; border-radius: 4px; padding: 6px;
    }
    QLineEdit:focus, QTextEdit:focus { border: 1px solid #007acc; }
    QComboBox, QSpinBox, QDoubleSpinBox {
        background-color: #252526; color: #e0e0e0;
        border: 1px solid #3e3e42; border-radius: 4px; padding: 6px;
    }
    QComboBox:focus, QSpinBox:focus { border: 1px solid #007acc; }
    QCheckBox { color: #cccccc; }
    QPushButton {
        background-color: #0e639c; color: #ffffff;
        border: none; border-radius: 4px;
        padding: 8px 16px; font-weight: 500;
    }
    QPushButton:hover { background-color: #1177bb; }
    QPushButton:pressed { background-color: #0d5a8f; }
    QPushButton:disabled { background-color: #3e3e42; color: #888888; }
"""

_BTN_NEUTRAL = (
    "QPushButton { background-color:#3e3e42; color:#e0e0e0; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#555558; }"
)


class NewProjectDialog(QDialog):
    """Dialog for creating a new ClassKit project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(520)
        self.setStyleSheet(_DARK_STYLE)

        self._custom_scheme: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        header = QLabel("<h2 style='color:#ffffff; margin:0;'>Create New Project</h2>")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("MyDataset")
        form.addRow("<b>Project Name:</b>", self.name_edit)

        location_row = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Select project location\u2026")
        self.location_edit.setText(str(Path.home() / "ClassKit" / "projects"))
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.setMaximumWidth(90)
        browse_btn.clicked.connect(self._browse_location)
        location_row.addWidget(self.location_edit, 1)
        location_row.addWidget(browse_btn)
        form.addRow("<b>Location:</b>", location_row)

        layout.addLayout(form)

        scheme_group = QGroupBox("Labeling Scheme")
        scheme_vlayout = QVBoxLayout(scheme_group)
        scheme_vlayout.setSpacing(10)

        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("None \u2014 define manually after creation", "none")
        self.preset_combo.addItem(
            "Head / Tail  (4 directions \u00b7 A D W S)", "head_tail"
        )
        self.preset_combo.addItem(
            "Color tag \u2014 1 factor  (5 colors)", "color_tag_1"
        )
        self.preset_combo.addItem(
            "Color tag \u2014 2 factors  (25 composites)", "color_tag_2"
        )
        self.preset_combo.addItem(
            "Color tag \u2014 3 factors  (125 composites)", "color_tag_3"
        )
        self.preset_combo.addItem("Age  (young / old)", "age")
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
            "<b>Tip:</b> You can always reconfigure the scheme later from the "
            "left panel.  The project folder will be created at the specified location."
        )
        info.setWordWrap(True)
        info.setStyleSheet(
            "padding:10px; background-color:#252526; border-radius:6px; "
            "border-left:3px solid #0e639c; color:#aaaaaa; line-height:1.6;"
        )
        layout.addWidget(info)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create Project")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.name_edit.textChanged.connect(self._validate)
        self._validate()
        self._on_preset_changed()

    def _on_preset_changed(self):
        key = self.preset_combo.currentData()
        _C = ["red", "blue", "green", "yellow", "white"]
        info_map = {
            "none": "Free-form \u2014 define labels manually after the project is created.",
            "head_tail": "1 factor \u00b7 4 labels: left, right, up, down  (keys A D W S).",
            "color_tag_1": f"1 factor \u00b7 5 labels: {', '.join(_C)}.",
            "color_tag_2": "2 factors \u00d7 5 colors = 25 composite labels.",
            "color_tag_3": "3 factors \u00d7 5 colors = 125 composite labels.",
            "age": "1 factor \u00b7 2 labels: young, old.",
        }
        self._scheme_info.setText(info_map.get(key, ""))
        if self._custom_scheme is not None:
            self._custom_scheme = None
            self._custom_scheme_lbl.setText("")
            self._btn_define_scheme.setText("Define Full Scheme\u2026")

    def _open_scheme_editor(self):
        dlg = ClassEditorDialog(parent=self)
        if dlg.exec():
            self._custom_scheme = dlg.get_scheme_dict()
            factors = self._custom_scheme.get("factors", [])
            total_labels = sum(len(f.get("labels", [])) for f in factors)
            factor_names = ", ".join(f.get("name", "?") for f in factors)
            self._custom_scheme_lbl.setText(
                f"\u2713 Custom scheme: {len(factors)} factor(s)  \u00b7  {total_labels} labels  "
                f"({factor_names})"
            )
            self._btn_define_scheme.setText("Edit Full Scheme\u2026")
            self.preset_combo.setCurrentIndex(0)

    def _browse_location(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Project Location",
            self.location_edit.text() or str(Path.home()),
        )
        if folder:
            self.location_edit.setText(folder)

    def _validate(self):
        ok = self.buttons.button(QDialogButtonBox.Ok)
        ok.setEnabled(len(self.name_edit.text().strip()) > 0)

    def get_project_info(self) -> dict:
        name = self.name_edit.text().strip()
        project_path = Path(self.location_edit.text()) / name

        if self._custom_scheme is not None:
            scheme_dict = self._custom_scheme
            classes: List[str] = []
            for f in scheme_dict.get("factors", []):
                classes.extend(f.get("labels", []))
            return {
                "name": name,
                "path": str(project_path),
                "classes": classes,
                "scheme": _SchemeWrapper(scheme_dict),
            }

        from hydra_suite.classkit.presets import (
            age_preset,
            color_tag_preset,
            head_tail_preset,
        )

        _C = ["red", "blue", "green", "yellow", "white"]
        key = self.preset_combo.currentData()
        preset_map = {
            "head_tail": head_tail_preset(),
            "color_tag_1": color_tag_preset(1, _C),
            "color_tag_2": color_tag_preset(2, _C),
            "color_tag_3": color_tag_preset(3, _C),
            "age": age_preset(),
        }
        scheme_obj = preset_map.get(key)
        classes = []
        if scheme_obj is not None:
            for f in scheme_obj.factors:
                classes.extend(f.labels)
        return {
            "name": name,
            "path": str(project_path),
            "classes": classes,
            "scheme": scheme_obj,
        }
