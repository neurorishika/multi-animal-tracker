"""
Polished dialogs for ClassKit
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..cluster.metalfaiss_backend import probe_ann_backend


class _KeyCapture(QLineEdit):
    """Drop-in replacement for QKeySequenceEdit.

    ``QKeySequenceEdit`` registers with the macOS Text Services Manager (TSM)
    as a text-input client.  On macOS + Python 3.13 + Qt 6 this registration
    races with the TSM UI-server port and produces a SIGBUS crash.

    This widget avoids all native IME / TSM hooks by:
    • staying permanently read-only (no system text-input session),
    • disabling the Qt input-method bridge (``WA_InputMethodEnabled = False``),
    • capturing key presses directly in ``keyPressEvent``.

    Public interface is compatible with ``QKeySequenceEdit``:
    ``setKeySequence(QKeySequence)``, ``keySequence() -> QKeySequence``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seq = QKeySequence()
        self._capturing = False
        self.setReadOnly(True)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        self.setPlaceholderText("click → press key")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ── public interface MatchQKeySequenceEdit ────────────────────────────────

    def setKeySequence(self, seq):
        if not isinstance(seq, QKeySequence):
            seq = QKeySequence(seq)
        self._seq = seq
        self._refresh()

    def keySequence(self) -> QKeySequence:
        return self._seq

    # ── internal ──────────────────────────────────────────────────────────────

    def _refresh(self):
        self.setText(self._seq.toString() if self._seq else "")

    def mousePressEvent(self, event):
        self._capturing = True
        self.setText("⌨  press key…")
        self.setFocus()

    def keyPressEvent(self, event):
        if not self._capturing:
            super().keyPressEvent(event)
            return
        key = event.key()
        # Ignore bare modifier presses
        if key in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
            Qt.Key.Key_unknown,
        ):
            return
        self._seq = QKeySequence(int(event.modifiers()) | int(key))
        self._capturing = False
        self._refresh()

    def focusOutEvent(self, event):
        if self._capturing:
            self._capturing = False
            self._refresh()
        super().focusOutEvent(event)


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

CLASSKIT_IMAGES_SUBDIR = "images"
CLASSKIT_SIEVE_THRESHOLD = 5000


# ──────────────────────────────────────────────────────────────────────────────
# Startup chooser
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Add-source dialog (pick a folder of images, optionally run DataSieve)
# ──────────────────────────────────────────────────────────────────────────────


class AddSourceDialog(QDialog):
    """Pick one or more image source folders for a ClassKit project.

    For each folder the user picks, the ``images/`` subdirectory is preferred
    (if it exists and contains images), otherwise the root folder itself is
    used — matching the PoseKit convention.

    If a folder contains more images than ``CLASSKIT_SIEVE_THRESHOLD``, the
    dialog offers to open DataSieve before continuing.
    """

    def __init__(self, existing_sources: Optional[List[Path]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Image Sources")
        self.setMinimumWidth(580)
        self.setStyleSheet(_DARK_STYLE)

        # Resolved image dirs already added (for duplicate checks).
        self._existing: List[Path] = [
            p.expanduser().resolve() for p in (existing_sources or [])
        ]
        # (dataset_root, resolved_images_dir, description)
        self._sources: List[Tuple[Path, Path, str]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info = QLabel(
            "Add one or more folders containing images.  "
            f"An <b>{CLASSKIT_IMAGES_SUBDIR}/</b> subdirectory is preferred when present; "
            "otherwise the folder root is used."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self._list = QListWidget()
        self._list.setMinimumHeight(120)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add Folder…")
        btn_add.clicked.connect(self._browse)
        btn_row.addWidget(btn_add)
        btn_row.addStretch(1)
        self._btn_remove = QPushButton("Remove Selected")
        self._btn_remove.setStyleSheet(
            "QPushButton { background-color:#6b1c1c; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:8px 16px; }"
            "QPushButton:hover { background-color:#8b2424; }"
        )
        self._btn_remove.clicked.connect(self._remove_selected)
        self._btn_remove.setEnabled(False)
        btn_row.addWidget(self._btn_remove)
        layout.addLayout(btn_row)

        self._list.itemSelectionChanged.connect(
            lambda: self._btn_remove.setEnabled(bool(self._list.selectedItems()))
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Done")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    def _browse(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select folder containing images", str(Path.home())
        )
        if not path:
            return

        d = Path(path).expanduser().resolve()

        # Prefer images/ subdirectory (PoseKit convention)
        candidate = d / CLASSKIT_IMAGES_SUBDIR
        if candidate.is_dir() and self._has_images(candidate):
            resolved = candidate
            location_note = f"(from {CLASSKIT_IMAGES_SUBDIR}/ subdirectory)"
        else:
            resolved = d
            location_note = "(folder root)"

        count = self._count_images(resolved)
        if count == 0:
            QMessageBox.warning(
                self,
                "No Images Found",
                f"No images were found in:\n{resolved}\n\n"
                "Please select a folder that contains .jpg / .jpeg / .png files.",
            )
            return

        # Duplicate check
        if resolved in self._existing or any(
            r == resolved for _, r, _ in self._sources
        ):
            QMessageBox.warning(
                self, "Already Added", "That folder has already been added."
            )
            return

        # DataSieve check
        if count > CLASSKIT_SIEVE_THRESHOLD:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Large Dataset Detected")
            msg.setText(
                f"This folder contains {count:,} images — that is a lot to label.\n\n"
                "DataSieve can reduce near-duplicates and create a smaller "
                "representative subset before labeling."
            )
            msg.setInformativeText(
                "Open this folder in DataSieve now, or add it as-is?"
            )
            btn_sieve = msg.addButton("Open in DataSieve", QMessageBox.AcceptRole)
            btn_add = msg.addButton("Add Anyway", QMessageBox.DestructiveRole)
            msg.addButton(QMessageBox.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == btn_sieve:
                try:
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "multi_tracker.tools.data_sieve.gui",
                            str(d),
                        ],
                        start_new_session=True,
                    )
                except Exception as exc:
                    QMessageBox.warning(
                        self, "Launch Failed", f"Could not launch DataSieve:\n{exc}"
                    )
                return
            if clicked != btn_add:
                return

        self._sources.append((d, resolved, d.name))
        item = QListWidgetItem(f"{d.name}  —  {count:,} images  {location_note}\n{d}")
        self._list.addItem(item)

    def _remove_selected(self):
        for item in self._list.selectedItems():
            row = self._list.row(item)
            self._list.takeItem(row)
            if row < len(self._sources):
                self._sources.pop(row)

    @staticmethod
    def _has_images(folder: Path) -> bool:
        exts = {".jpg", ".jpeg", ".png"}
        return any(p.suffix.lower() in exts for p in folder.iterdir() if p.is_file())

    @staticmethod
    def _count_images(folder: Path) -> int:
        exts = {".jpg", ".jpeg", ".png"}
        return sum(
            1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts
        )

    @property
    def sources(self) -> List[Tuple[Path, Path, str]]:
        """List of (dataset_root, resolved_images_dir, description)."""
        return list(self._sources)


# ──────────────────────────────────────────────────────────────────────────────
# Class / labeling-scheme editor dialog  (polished two-panel design)
# ──────────────────────────────────────────────────────────────────────────────

_BTN_ADD = (
    "QPushButton { background-color:#1a4a1a; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#235a23; }"
)
_BTN_DEL = (
    "QPushButton { background-color:#4a1a1a; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#6b2424; }"
)
_BTN_NEUTRAL = (
    "QPushButton { background-color:#3e3e42; color:#e0e0e0; padding:4px 12px; border-radius:4px; }"
    "QPushButton:hover { background-color:#555558; }"
)


class _LabelRow(QWidget):
    """A single label row: name QLineEdit + QKeySequenceEdit for shortcut."""

    def __init__(self, label: str = "", shortcut: str = "", parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._name_edit = QLineEdit(label)
        self._name_edit.setPlaceholderText("label name")
        self._name_edit.setMinimumWidth(120)
        row.addWidget(self._name_edit, 2)

        key_lbl = QLabel("key:")
        key_lbl.setStyleSheet("color:#777; font-size:11px;")
        key_lbl.setFixedWidth(28)
        row.addWidget(key_lbl)

        self._key_edit = _KeyCapture()
        if shortcut:
            try:
                self._key_edit.setKeySequence(QKeySequence(shortcut))
            except Exception:
                pass
        self._key_edit.setFixedWidth(110)
        self._key_edit.setToolTip(
            "Press any key: letters, digits, arrows, symbols (e.g. ↑ ↓ ← →, Space, +, -, …)"
        )
        row.addWidget(self._key_edit)

        self._btn_del = QPushButton("✕")
        self._btn_del.setFixedSize(26, 26)
        self._btn_del.setStyleSheet(_BTN_DEL)
        self._btn_del.setToolTip("Remove this label")
        row.addWidget(self._btn_del)

    def label(self) -> str:
        return self._name_edit.text().strip()

    def shortcut(self) -> str:
        ks = self._key_edit.keySequence()
        return ks.toString() if ks else ""

    def delete_button(self) -> QPushButton:
        return self._btn_del


class ClassEditorDialog(QDialog):
    """Polished two-panel labeling class / multi-factor scheme editor.

    Left panel: factor list with add/remove/rename controls.
    Right panel: per-label rows, each with a name field and QKeySequenceEdit
    shortcut capture (supports letters, digits, arrows, symbols, etc.).
    """

    _DEFAULT_COLORS = ["red", "blue", "green", "yellow", "white"]
    _PRESETS = {
        "head_tail": [
            {
                "name": "direction",
                "labels": ["left", "right", "up", "down"],
                "shortcuts": ["A", "D", "W", "S"],
            },
        ],
        "color_tag_1": [
            {"name": "tag_1", "labels": _DEFAULT_COLORS, "shortcuts": []},
        ],
        "color_tag_2": [
            {"name": "tag_1", "labels": _DEFAULT_COLORS, "shortcuts": []},
            {"name": "tag_2", "labels": _DEFAULT_COLORS, "shortcuts": []},
        ],
        "color_tag_3": [
            {"name": "tag_1", "labels": _DEFAULT_COLORS, "shortcuts": []},
            {"name": "tag_2", "labels": _DEFAULT_COLORS, "shortcuts": []},
            {"name": "tag_3", "labels": _DEFAULT_COLORS, "shortcuts": []},
        ],
        "age": [
            {"name": "age", "labels": ["young", "old"], "shortcuts": []},
        ],
    }

    def __init__(
        self,
        classes: Optional[List[str]] = None,
        scheme_dict: Optional[dict] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Edit Classes and Labeling Scheme")
        self.resize(820, 540)
        self.setMinimumWidth(680)
        self.setMinimumHeight(400)
        self.setStyleSheet(_DARK_STYLE)

        # ── internal state ─────────────────────────────────────────────
        self._factors: List[dict] = []

        if scheme_dict and scheme_dict.get("factors"):
            self._factors = [
                {
                    "name": f.get("name", f"factor_{i}"),
                    "labels": list(f.get("labels", [])),
                    "shortcuts": list(f.get("shortcut_keys", [])),
                }
                for i, f in enumerate(scheme_dict["factors"])
            ]
        elif classes:
            self._factors = [
                {"name": "class", "labels": list(classes), "shortcuts": []}
            ]
        else:
            self._factors = [
                {"name": "class", "labels": ["class_1", "class_2"], "shortcuts": []}
            ]

        # ── outer layout ────────────────────────────────────────────────
        outer = QVBoxLayout(self)
        outer.setSpacing(8)

        # Preset bar at the top
        preset_bar = QHBoxLayout()
        lbl_preset = QLabel("Quick preset:")
        lbl_preset.setStyleSheet("color:#888; font-size:11px;")
        preset_bar.addWidget(lbl_preset)
        self._preset_combo = QComboBox()
        self._preset_combo.setMaximumWidth(320)
        self._preset_combo.addItem("— choose preset —", None)
        self._preset_combo.addItem("Head / Tail  (4 directions · A D W S)", "head_tail")
        self._preset_combo.addItem("Color tag — 1 factor  (5 colors)", "color_tag_1")
        self._preset_combo.addItem(
            "Color tag — 2 factors  (25 composites)", "color_tag_2"
        )
        self._preset_combo.addItem(
            "Color tag — 3 factors  (125 composites)", "color_tag_3"
        )
        self._preset_combo.addItem("Age  (young / old)", "age")
        preset_bar.addWidget(self._preset_combo)
        btn_apply_preset = QPushButton("Apply")
        btn_apply_preset.setMaximumWidth(70)
        btn_apply_preset.clicked.connect(self._apply_preset)
        preset_bar.addWidget(btn_apply_preset)
        preset_bar.addStretch(1)
        outer.addLayout(preset_bar)

        # ── main splitter ───────────────────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet("QSplitter::handle { background-color:#3e3e42; }")
        outer.addWidget(splitter, 1)

        # ──── LEFT PANEL: factor list ───────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        left_header = QLabel("<b>Factors</b>")
        left_layout.addWidget(left_header)

        self._factor_list = QListWidget()
        self._factor_list.setMinimumWidth(150)
        self._factor_list.setMaximumWidth(220)
        self._factor_list.setStyleSheet(
            "QListWidget { background:#252526; border:1px solid #3e3e42; border-radius:4px; }"
            "QListWidget::item { padding:6px 10px; color:#cccccc; }"
            "QListWidget::item:selected { background-color:#094771; color:#ffffff; }"
        )
        self._factor_list.currentRowChanged.connect(self._on_factor_selected)
        left_layout.addWidget(self._factor_list, 1)

        left_btn_row = QHBoxLayout()
        btn_add_factor = QPushButton("+ Factor")
        btn_add_factor.setStyleSheet(_BTN_ADD)
        btn_add_factor.clicked.connect(self._add_factor)
        btn_add_factor.setToolTip("Add a new factor")
        left_btn_row.addWidget(btn_add_factor)
        self._btn_del_factor = QPushButton("− Factor")
        self._btn_del_factor.setStyleSheet(_BTN_DEL)
        self._btn_del_factor.clicked.connect(self._remove_selected_factor)
        self._btn_del_factor.setToolTip("Remove selected factor")
        left_btn_row.addWidget(self._btn_del_factor)
        left_layout.addLayout(left_btn_row)
        splitter.addWidget(left_widget)

        # ──── RIGHT PANEL: factor detail ────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)
        right_layout.setSpacing(6)

        # Factor name row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Factor name:"))
        self._factor_name_edit = QLineEdit()
        self._factor_name_edit.setPlaceholderText("e.g. direction")
        self._factor_name_edit.setMaximumWidth(200)
        self._factor_name_edit.textChanged.connect(self._on_name_edited)
        name_row.addWidget(self._factor_name_edit)
        name_row.addStretch(1)
        right_layout.addLayout(name_row)

        # Label rows in a scroll area
        lbl_header = QHBoxLayout()
        lbl_header.addWidget(QLabel("<b>Labels</b>"))
        lbl_header.addWidget(
            QLabel(
                "<i style='color:#777; font-size:11px;'>"
                "  Key: letter, digit, arrow (↑↓←→), symbol (+  −  Space …)</i>"
            )
        )
        lbl_header.addStretch(1)
        btn_add_label = QPushButton("+ Label")
        btn_add_label.setStyleSheet(_BTN_ADD)
        btn_add_label.setMaximumWidth(80)
        btn_add_label.clicked.connect(self._add_label_row)
        lbl_header.addWidget(btn_add_label)
        right_layout.addLayout(lbl_header)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(
            "QScrollArea { border:1px solid #3e3e42; border-radius:4px; background:#252526; }"
        )
        self._labels_container = QWidget()
        self._labels_layout = QVBoxLayout(self._labels_container)
        self._labels_layout.setContentsMargins(8, 8, 8, 8)
        self._labels_layout.setSpacing(4)
        self._labels_layout.addStretch(1)
        scroll_area.setWidget(self._labels_container)
        right_layout.addWidget(scroll_area, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([180, 600])

        # ── bottom buttons ───────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._do_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        # ── populate ─────────────────────────────────────────────────────
        self._current_factor_idx: int = -1
        self._label_rows: List[_LabelRow] = []
        self._populating = False
        self._populate_factor_list()
        if self._factors:
            self._factor_list.setCurrentRow(0)

    # ── factor list management ───────────────────────────────────────────

    def _populate_factor_list(self):
        self._factor_list.blockSignals(True)
        self._factor_list.clear()
        for f in self._factors:
            self._factor_list.addItem(f.get("name", "factor"))
        self._factor_list.blockSignals(False)

    def _on_factor_selected(self, row: int):
        if row < 0 or row >= len(self._factors):
            self._factor_name_edit.clear()
            self._clear_label_rows()
            self._current_factor_idx = -1
            return
        # Save current edits first
        self._flush_current_factor()
        self._current_factor_idx = row
        self._load_factor_into_panel(self._factors[row])

    def _flush_current_factor(self):
        """Write panel state back to self._factors[_current_factor_idx]."""
        idx = self._current_factor_idx
        if idx < 0 or idx >= len(self._factors):
            return
        self._factors[idx]["name"] = self._factor_name_edit.text().strip() or "factor"
        self._factors[idx]["labels"] = [
            r.label() for r in self._label_rows if r.label()
        ]
        self._factors[idx]["shortcuts"] = [
            r.shortcut() for r in self._label_rows if r.label()
        ]
        # Keep list item label in sync
        item = self._factor_list.item(idx)
        if item:
            item.setText(self._factors[idx]["name"])

    def _load_factor_into_panel(self, factor: dict):
        self._populating = True
        self._factor_name_edit.setText(factor.get("name", ""))
        self._clear_label_rows()
        labels = factor.get("labels", [])
        shortcuts = factor.get("shortcuts", [])
        for i, lbl in enumerate(labels):
            key = shortcuts[i] if i < len(shortcuts) else ""
            self._append_label_row(lbl, key)
        self._populating = False

    def _on_name_edited(self, text: str):
        idx = self._current_factor_idx
        if idx < 0 or self._populating:
            return
        if idx < len(self._factors):
            self._factors[idx]["name"] = text.strip() or "factor"
        item = self._factor_list.item(idx)
        if item:
            item.setText(text.strip() or "factor")

    def _add_factor(self):
        self._flush_current_factor()
        new_name = f"factor_{len(self._factors) + 1}"
        self._factors.append(
            {"name": new_name, "labels": ["label_1"], "shortcuts": [""]}
        )
        self._populate_factor_list()
        self._factor_list.setCurrentRow(len(self._factors) - 1)

    def _remove_selected_factor(self):
        idx = self._factor_list.currentRow()
        if idx < 0 or len(self._factors) <= 1:
            QMessageBox.information(
                self, "Cannot Remove", "A scheme must have at least one factor."
            )
            return
        self._flush_current_factor()
        self._factors.pop(idx)
        self._current_factor_idx = -1
        self._populate_factor_list()
        new_row = min(idx, len(self._factors) - 1)
        self._factor_list.setCurrentRow(new_row)

    # ── label row management ─────────────────────────────────────────────

    def _clear_label_rows(self):
        for row in self._label_rows:
            row.setParent(None)
        self._label_rows = []

    def _append_label_row(self, label: str = "", shortcut: str = ""):
        row = _LabelRow(label, shortcut, self._labels_container)
        # Insert before the trailing stretch item
        insert_pos = max(0, self._labels_layout.count() - 1)
        self._labels_layout.insertWidget(insert_pos, row)
        self._label_rows.append(row)
        row.delete_button().clicked.connect(lambda: self._remove_label_row(row))

    def _add_label_row(self):
        self._append_label_row("", "")

    def _remove_label_row(self, row: "_LabelRow"):
        if len(self._label_rows) <= 1:
            QMessageBox.information(
                self, "Cannot Remove", "A factor must have at least one label."
            )
            return
        self._label_rows.remove(row)
        row.setParent(None)

    # ── preset loading ─────────────────────────────────────────────────────

    def _apply_preset(self):
        key = self._preset_combo.currentData()
        if key is None:
            return
        preset = self._PRESETS.get(key)
        if preset is None:
            return
        reply = QMessageBox.question(
            self,
            "Apply Preset",
            f"Replace current scheme with the '{key}' preset?",
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if reply != QMessageBox.Yes:
            return
        self._current_factor_idx = -1
        self._factors = [dict(f) for f in preset]
        self._populate_factor_list()
        self._clear_label_rows()
        if self._factors:
            self._factor_list.setCurrentRow(0)

    # ── result accessors ────────────────────────────────────────────────────

    def _do_accept(self):
        self._flush_current_factor()
        for i, f in enumerate(self._factors):
            if not f.get("labels"):
                QMessageBox.warning(
                    self,
                    "Empty Labels",
                    f"Factor '{f.get('name', i + 1)}' has no labels. "
                    "Add at least one label or remove the factor.",
                )
                return
        self.accept()

    @property
    def flat_classes(self) -> List[str]:
        """Labels of the first factor (single-factor / flat-class mode)."""
        if not self._factors:
            return ["class_1", "class_2"]
        return self._factors[0].get("labels", [])

    def get_scheme_dict(self) -> Optional[dict]:
        """Return a scheme dict compatible with ``LabelingScheme.from_dict()``."""
        factors = self._factors
        if len(factors) == 1:
            f = factors[0]
            return {
                "name": f.get("name", "custom"),
                "description": "Custom 1-factor labeling scheme",
                "factors": [
                    {
                        "name": f.get("name", "class"),
                        "labels": f.get("labels", []),
                        "shortcut_keys": f.get("shortcuts", []),
                    }
                ],
                "training_modes": ["flat_tiny", "flat_yolo"],
            }
        modes = ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]
        return {
            "name": "_".join(f.get("name", "f") for f in factors),
            "description": f"{len(factors)}-factor labeling scheme",
            "factors": [
                {
                    "name": f.get("name", f"factor_{i}"),
                    "labels": f.get("labels", []),
                    "shortcut_keys": f.get("shortcuts", []),
                }
                for i, f in enumerate(factors)
            ],
            "training_modes": modes,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Keyboard shortcut editor dialog
# ──────────────────────────────────────────────────────────────────────────────


class ShortcutEditorDialog(QDialog):
    """Edit keyboard shortcut assignments for non-label global actions.

    Shows a table of action → current key.  The user clicks a row and types a
    new key (single character or Qt key name via QKeySequenceEdit).
    """

    # Default global action shortcuts (name, default_key_sequence)
    DEFAULT_SHORTCUTS = [
        ("Explore mode", "E"),
        ("Labeling mode", "L"),
        ("Predictions mode", "P"),
        ("Sample next candidates", "Space"),
        ("Previous unlabeled", "Left"),
        ("Next unlabeled", "Right"),
        ("Undo last label (Ctrl+Z)", "Ctrl+Z"),
    ]

    def __init__(self, current: Optional[dict] = None, parent=None):
        """
        Args:
            current: dict mapping action name → key sequence string.
                     Falls back to DEFAULT_SHORTCUTS for missing entries.
        """
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumWidth(500)
        self.setStyleSheet(_DARK_STYLE)

        self._shortcuts: dict = {name: seq for name, seq in self.DEFAULT_SHORTCUTS}
        if current:
            self._shortcuts.update(current)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        hdr = QLabel("<b>Global Navigation Shortcuts</b>")
        layout.addWidget(hdr)

        info = QLabel(
            "Click a row and press the desired key combination.  "
            "All key types are supported: letters, digits, "
            "<b>arrow keys</b> (↑ ↓ ← →), <b>symbols</b> (+  −  Space  Tab …), "
            "and modifier combos (Ctrl+Z, Shift+Left, …).  "
            "Label-specific shortcuts are defined in the <b>Class Scheme Editor</b>."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#888; font-size:11px;")
        layout.addWidget(info)

        # Row list
        form = QFormLayout()
        form.setSpacing(8)
        self._key_edits: dict = {}
        for name, default_key in self.DEFAULT_SHORTCUTS:
            current_key = self._shortcuts.get(name, default_key)
            edit = _KeyCapture()
            try:
                edit.setKeySequence(QKeySequence(current_key))
            except Exception:
                pass
            self._key_edits[name] = edit
            form.addRow(QLabel(name), edit)
        layout.addLayout(form)

        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setStyleSheet(
            "QPushButton { background-color:#3e3e42; color:#e0e0e0; "
            "border:none; border-radius:4px; padding:6px 14px; }"
            "QPushButton:hover { background-color:#555558; }"
        )
        btn_reset.clicked.connect(self._reset_defaults)
        layout.addWidget(btn_reset)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reset_defaults(self):
        for name, default_key in self.DEFAULT_SHORTCUTS:
            edit = self._key_edits.get(name)
            if edit is not None:
                try:
                    edit.setKeySequence(QKeySequence(default_key))
                except Exception:
                    pass

    def get_shortcuts(self) -> dict:
        """Return dict of action_name → key sequence string."""
        out = {}
        for name, edit in self._key_edits.items():
            ks = edit.keySequence()
            out[name] = ks.toString() if ks else self._shortcuts.get(name, "")
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Thin wrapper so a raw scheme dict looks like a LabelingScheme (for mainwindow)
# ──────────────────────────────────────────────────────────────────────────────


class _SchemeWrapper:
    """Wraps a raw scheme dict to satisfy the ``scheme.to_dict()`` contract."""

    def __init__(self, d: dict):
        self._d = d

    def to_dict(self) -> dict:
        return self._d

    @property
    def factors(self):
        return self._d.get("factors", [])


class NewProjectDialog(QDialog):
    """Dialog for creating a new ClassKit project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Project")
        self.setMinimumWidth(520)
        self.setStyleSheet(_DARK_STYLE)

        # Scheme from the embedded ClassEditorDialog (overrides preset when set)
        self._custom_scheme: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Header
        header = QLabel("<h2 style='color:#ffffff; margin:0;'>Create New Project</h2>")
        layout.addWidget(header)

        # ── name + location ──────────────────────────────────────────
        form = QFormLayout()
        form.setSpacing(12)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("MyDataset")
        form.addRow("<b>Project Name:</b>", self.name_edit)

        location_row = QHBoxLayout()
        self.location_edit = QLineEdit()
        self.location_edit.setPlaceholderText("Select project location…")
        self.location_edit.setText(str(Path.home() / "ClassKit" / "projects"))
        browse_btn = QPushButton("Browse…")
        browse_btn.setMaximumWidth(90)
        browse_btn.clicked.connect(self._browse_location)
        location_row.addWidget(self.location_edit, 1)
        location_row.addWidget(browse_btn)
        form.addRow("<b>Location:</b>", location_row)

        layout.addLayout(form)

        # ── scheme section ───────────────────────────────────────────
        scheme_group = QGroupBox("Labeling Scheme")
        scheme_vlayout = QVBoxLayout(scheme_group)
        scheme_vlayout.setSpacing(10)

        # Quick-preset row
        preset_row = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("None — define manually after creation", "none")
        self.preset_combo.addItem("Head / Tail  (4 directions · A D W S)", "head_tail")
        self.preset_combo.addItem("Color tag — 1 factor  (5 colors)", "color_tag_1")
        self.preset_combo.addItem(
            "Color tag — 2 factors  (25 composites)", "color_tag_2"
        )
        self.preset_combo.addItem(
            "Color tag — 3 factors  (125 composites)", "color_tag_3"
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

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color:#3e3e42;")
        scheme_vlayout.addWidget(div)

        # Full scheme editor button row
        editor_row = QHBoxLayout()
        editor_row.addWidget(QLabel("Or open the full scheme editor:"))
        editor_row.addStretch(1)
        self._btn_define_scheme = QPushButton("Define Full Scheme…")
        self._btn_define_scheme.setStyleSheet(_BTN_NEUTRAL)
        self._btn_define_scheme.clicked.connect(self._open_scheme_editor)
        editor_row.addWidget(self._btn_define_scheme)
        scheme_vlayout.addLayout(editor_row)

        # Custom-scheme status label
        self._custom_scheme_lbl = QLabel("")
        self._custom_scheme_lbl.setStyleSheet(
            "color:#4ec94e; font-size:11px; padding-left:4px;"
        )
        self._custom_scheme_lbl.setWordWrap(True)
        scheme_vlayout.addWidget(self._custom_scheme_lbl)

        layout.addWidget(scheme_group)

        # Info box
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

        # Buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create Project")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Validation
        self.name_edit.textChanged.connect(self._validate)
        self._validate()
        self._on_preset_changed()

    # ── helpers ──────────────────────────────────────────────────────────

    def _on_preset_changed(self):
        key = self.preset_combo.currentData()
        _C = ["red", "blue", "green", "yellow", "white"]
        info_map = {
            "none": "Free-form — define labels manually after the project is created.",
            "head_tail": "1 factor · 4 labels: left, right, up, down  (keys A D W S).",
            "color_tag_1": f"1 factor · 5 labels: {', '.join(_C)}.",
            "color_tag_2": "2 factors × 5 colors = 25 composite labels.",
            "color_tag_3": "3 factors × 5 colors = 125 composite labels.",
            "age": "1 factor · 2 labels: young, old.",
        }
        self._scheme_info.setText(info_map.get(key, ""))
        # Reset custom scheme when user picks a preset
        if self._custom_scheme is not None:
            self._custom_scheme = None
            self._custom_scheme_lbl.setText("")
            self._btn_define_scheme.setText("Define Full Scheme…")

    def _open_scheme_editor(self):
        dlg = ClassEditorDialog(parent=self)
        if dlg.exec():
            self._custom_scheme = dlg.get_scheme_dict()
            factors = self._custom_scheme.get("factors", [])
            total_labels = sum(len(f.get("labels", [])) for f in factors)
            factor_names = ", ".join(f.get("name", "?") for f in factors)
            self._custom_scheme_lbl.setText(
                f"✓ Custom scheme: {len(factors)} factor(s)  ·  {total_labels} labels  "
                f"({factor_names})"
            )
            self._btn_define_scheme.setText("Edit Full Scheme…")
            # Reset preset combo to "none" to avoid confusion
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

    # ── result ───────────────────────────────────────────────────────────

    def get_project_info(self) -> dict:
        """Return project configuration dict."""
        name = self.name_edit.text().strip()
        project_path = Path(self.location_edit.text()) / name

        # Custom scheme (from editor) takes priority over quick preset
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

        # Quick preset path
        from ...classkit.presets import age_preset, color_tag_preset, head_tail_preset

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

        self.canonicalize_check = QCheckBox(
            "Canonicalize MAT individual datasets using metadata.json when available"
        )
        self.canonicalize_check.setStyleSheet("color: #cccccc;")
        form.addRow("", self.canonicalize_check)

        layout.addLayout(form)

        # Info
        info_text = (
            "<b>Model Guide:</b><br>"
            + "• <b>DINOv2:</b> Best for general visual understanding<br>"
            + "• <b>CLIP:</b> Good for semantic similarity<br>"
            + "• <b>ResNet/EfficientNet:</b> Faster, less memory<br><br>"
            + "<b>Device:</b> GPU is recommended for faster processing.<br>"
            + "<b>Canonicalize:</b> Rotates/crops MAT individual-analysis crops using OBB metadata when available.<br>"
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
        canonicalize_mat = self.canonicalize_check.isChecked()

        return model_name, device, batch_size, force_recompute, canonicalize_mat


class ClusterDialog(QDialog):
    """Dialog for configuring clustering."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Configuration")
        self.setMinimumWidth(500)

        # Detect available backends
        ann_probe = probe_ann_backend()
        ann_ready = (
            ann_probe.get("hdbscan_available", False)
            or ann_probe.get("best_ann") != "numpy"
        )

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
        self.gpu_combo.addItem("CPU (scikit-learn)", False)
        if ann_ready:
            self.gpu_combo.addItem(f"Accelerated ({ann_probe['best_ann']})", True)

        form.addRow("<b>Backend:</b>", self.gpu_combo)

        layout.addLayout(form)

        # Build info message
        info_text = (
            "<b>Clustering Tips:</b><br>"
            + "• Use many clusters (100s-1000s) to capture visual modes<br>"
            + "• Clusters ≠ classes (fine-grained structure)<br>"
            + f"• Current Best Backend: <b>{ann_probe.get('best_ann', 'numpy')}</b>"
        )
        if ann_probe.get("hdbscan_available"):
            info_text += (
                "<br>• HDBSCAN (density-based) is available for advanced users."
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
        """Resolve a de-duplicated class list for mapping dropdowns."""
        ordered: List[str] = []

        def _add(value: object) -> None:
            text = str(value).strip()
            if text and text not in ordered:
                ordered.append(text)

        for value in class_choices or []:
            _add(value)

        # For single-factor schemes include declared labels even if not yet observed.
        if self._scheme is not None:
            factors = getattr(self._scheme, "factors", None) or []
            if len(factors) == 1:
                for label in getattr(factors[0], "labels", []) or []:
                    _add(label)

        return ordered

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
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

        # Tabs container — must be created before any tab widget is referenced below
        self.tabs = QTabWidget()
        self.general_tab = QWidget()

        form = QFormLayout()
        form.setSpacing(8)

        # Training mode
        self.mode_combo = QComboBox()
        self._populate_modes()
        form.addRow("<b>Training Mode:</b>", self.mode_combo)

        # Training device (PyTorch execution target during optimization)
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        try:
            from ...utils.gpu_utils import (
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

        # Inference runtime (used by TinyCNN inference after training / on load)
        self.compute_runtime_combo = QComboBox()
        try:
            from ...core.runtime.compute_runtime import (
                runtime_label,
                supported_runtimes_for_pipeline,
            )

            _runtimes = supported_runtimes_for_pipeline("tiny_classify")
            for _rt in _runtimes:
                self.compute_runtime_combo.addItem(runtime_label(_rt), _rt)

            # Choose a sensible default based on selected training device.
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

        # Base model (YOLO only)
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

        # Hyperparams
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

        # Tooltips for General tab fields
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

        # Mode description label (updated by _on_mode_changed)
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
        self.tabs.addTab(self.general_tab, "General")

        # Tab 2: Tiny Architecture
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

        # Tooltips for Tiny Architecture fields
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

        # Tab 3: Space & Augmentations
        self.aug_tab = QWidget()
        aug_tab_layout = QVBoxLayout(self.aug_tab)
        aug_tab_layout.setContentsMargins(0, 0, 0, 0)
        self._aug_scroll = QScrollArea()
        self._aug_scroll.setWidgetResizable(True)
        self._aug_scroll.setFrameShape(QFrame.NoFrame)
        aug_scroll_content = QWidget()
        aug_layout = QVBoxLayout(aug_scroll_content)

        space_group = QGroupBox("Training Space")
        space_layout = QHBoxLayout(space_group)
        self.space_combo = QComboBox()
        self.space_combo.addItem("Canonical (rotation corrected)", "canonical")
        self.space_combo.addItem("Original (raw frames)", "original")
        space_layout.addWidget(QLabel("<b>Space:</b>"))
        space_layout.addWidget(self.space_combo)
        aug_layout.addWidget(space_group)

        aug_group = QGroupBox("Augmentations")
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

        aug_layout.addWidget(aug_group)

        # ---- Label-Switching Expansion (advanced, collapsible) ----
        exp_group = QGroupBox("Label-Switching Expansion  (advanced)")
        exp_group.setCheckable(True)
        exp_group.setChecked(False)
        exp_group.setToolTip(
            "Generate deterministic mirrored copies of training images with remapped labels.\n"
            "Useful when a flip transforms one class into another (e.g. 'left' ↔ 'right'\n"
            "on a horizontal flip).  Expanded copies are only added to the train split."
        )
        exp_vbox = QVBoxLayout(exp_group)

        exp_note = QLabel(
            "<i>Each row: source label → destination label when that flip is applied.<br>"
            "Pairs are applied to train images only; evaluation data is never flipped.<br>"
            "Label expansion is available only in canonical space.<br>"
            "When enabled, all stochastic augmentations (flip/rotate) are disabled.</i>"
        )
        exp_note.setWordWrap(True)
        exp_note.setStyleSheet("color:#aaa; font-size:11px;")
        exp_vbox.addWidget(exp_note)

        self._exp_constraints_label = QLabel("")
        self._exp_constraints_label.setWordWrap(True)
        self._exp_constraints_label.setStyleSheet("color:#e0c070; font-size:11px;")
        exp_vbox.addWidget(self._exp_constraints_label)

        # -- Horizontal flip (LR) mapping rows --
        lr_hdr_row = QHBoxLayout()
        lr_hdr_row.addWidget(QLabel("<b>Horizontal flip (LR) mappings:</b>"))
        lr_add_btn = QPushButton("+ Add pair")
        lr_add_btn.setFixedWidth(90)
        lr_hdr_row.addWidget(lr_add_btn)
        lr_hdr_row.addStretch()
        exp_vbox.addLayout(lr_hdr_row)

        self._lr_mapping_rows: List[Tuple[QComboBox, QComboBox]] = []
        self._lr_rows_widget = QWidget()
        self._lr_rows_layout = QVBoxLayout(self._lr_rows_widget)
        self._lr_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._lr_rows_layout.setSpacing(3)
        exp_vbox.addWidget(self._lr_rows_widget)

        def _safe_class_text(value: object) -> str:
            if value is None or isinstance(value, bool):
                return ""
            return str(value).strip()

        def _build_class_combo(initial_text: object = "") -> QComboBox:
            combo = QComboBox()
            combo.setMinimumWidth(160)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)

            if self._class_choices:
                for cls in self._class_choices:
                    combo.addItem(cls, cls)
            else:
                combo.addItem("(no classes)", "")
                combo.setEnabled(False)

            initial = _safe_class_text(initial_text)
            if initial:
                idx = combo.findData(initial)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            return combo

        def _add_lr_row(src_text="", dst_text=""):
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            src = _build_class_combo(src_text)
            arr = QLabel("→")
            dst = _build_class_combo(dst_text)
            add_rev = QPushButton("Reverse")
            add_rev.setFixedWidth(72)
            add_rev.setToolTip("Add the reverse mapping (destination → source)")
            rm = QPushButton("Remove")
            rm.setFixedWidth(72)
            rm.setStyleSheet("color:#c66;")
            row_h.addWidget(src, 1)
            row_h.addWidget(arr)
            row_h.addWidget(dst, 1)
            row_h.addWidget(add_rev)
            row_h.addWidget(rm)
            self._lr_rows_layout.addWidget(row_w)
            pair = (src, dst)
            self._lr_mapping_rows.append(pair)

            def _remove_row() -> None:
                if pair in self._lr_mapping_rows:
                    self._lr_mapping_rows.remove(pair)
                row_w.setParent(None)

            rm.clicked.connect(_remove_row)

            def _add_reverse_pair() -> None:
                s = _safe_class_text(src.currentData() or src.currentText())
                d = _safe_class_text(dst.currentData() or dst.currentText())
                if not s or not d or s == d:
                    return
                for rs, rd in self._lr_mapping_rows:
                    rs_name = _safe_class_text(rs.currentData() or rs.currentText())
                    rd_name = _safe_class_text(rd.currentData() or rd.currentText())
                    if rs_name == d and rd_name == s:
                        return
                _add_lr_row(d, s)

            add_rev.clicked.connect(_add_reverse_pair)

        # clicked(bool) provides a bool arg; ignore it to keep defaults intact.
        lr_add_btn.clicked.connect(lambda _checked=False: _add_lr_row())
        self._add_lr_row = _add_lr_row  # expose for programmatic seeding

        # -- Vertical flip (UD) mapping rows --
        ud_hdr_row = QHBoxLayout()
        ud_hdr_row.addWidget(QLabel("<b>Vertical flip (UD) mappings:</b>"))
        ud_add_btn = QPushButton("+ Add pair")
        ud_add_btn.setFixedWidth(90)
        ud_hdr_row.addWidget(ud_add_btn)
        ud_hdr_row.addStretch()
        exp_vbox.addLayout(ud_hdr_row)

        self._ud_mapping_rows: List[Tuple[QComboBox, QComboBox]] = []
        self._ud_rows_widget = QWidget()
        self._ud_rows_layout = QVBoxLayout(self._ud_rows_widget)
        self._ud_rows_layout.setContentsMargins(0, 0, 0, 0)
        self._ud_rows_layout.setSpacing(3)
        exp_vbox.addWidget(self._ud_rows_widget)

        def _add_ud_row(src_text="", dst_text=""):
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(0, 0, 0, 0)
            src = _build_class_combo(src_text)
            arr = QLabel("→")
            dst = _build_class_combo(dst_text)
            add_rev = QPushButton("Reverse")
            add_rev.setFixedWidth(72)
            add_rev.setToolTip("Add the reverse mapping (destination → source)")
            rm = QPushButton("Remove")
            rm.setFixedWidth(72)
            rm.setStyleSheet("color:#c66;")
            row_h.addWidget(src, 1)
            row_h.addWidget(arr)
            row_h.addWidget(dst, 1)
            row_h.addWidget(add_rev)
            row_h.addWidget(rm)
            self._ud_rows_layout.addWidget(row_w)
            pair = (src, dst)
            self._ud_mapping_rows.append(pair)

            def _remove_row() -> None:
                if pair in self._ud_mapping_rows:
                    self._ud_mapping_rows.remove(pair)
                row_w.setParent(None)

            rm.clicked.connect(_remove_row)

            def _add_reverse_pair() -> None:
                s = _safe_class_text(src.currentData() or src.currentText())
                d = _safe_class_text(dst.currentData() or dst.currentText())
                if not s or not d or s == d:
                    return
                for rs, rd in self._ud_mapping_rows:
                    rs_name = _safe_class_text(rs.currentData() or rs.currentText())
                    rd_name = _safe_class_text(rd.currentData() or rd.currentText())
                    if rs_name == d and rd_name == s:
                        return
                _add_ud_row(d, s)

            add_rev.clicked.connect(_add_reverse_pair)

        # clicked(bool) provides a bool arg; ignore it to keep defaults intact.
        ud_add_btn.clicked.connect(lambda _checked=False: _add_ud_row())
        self._add_ud_row = _add_ud_row  # expose for programmatic seeding

        aug_layout.addWidget(exp_group)
        self._exp_group = exp_group  # so get_settings() can check isChecked()

        aug_layout.addStretch()
        self._aug_scroll.setWidget(aug_scroll_content)
        aug_tab_layout.addWidget(self._aug_scroll)
        self.tabs.addTab(self.aug_tab, "Space and Augmentations")

        layout.addWidget(self.tabs, 1)

        # Log panel
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)
        self.log_view.setStyleSheet(
            "background:#111; color:#ccc; font-family:monospace; font-size:11px;"
        )
        layout.addWidget(self.log_view)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons
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

        self.cancel_btn.clicked.connect(self._on_cancel)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.tiny_rebalance_combo.currentIndexChanged.connect(
            self._on_rebalance_mode_changed
        )
        self._on_mode_changed()
        self._on_rebalance_mode_changed()

        # Enforce canonical-only label expansion and explicit augmentation lockout.
        self._exp_group.toggled.connect(
            lambda _checked: self._sync_expansion_constraints()
        )
        self.space_combo.currentIndexChanged.connect(
            lambda _idx: self._sync_expansion_constraints()
        )
        self._sync_expansion_constraints()

    def _populate_modes(self):
        self.mode_combo.clear()
        if self._scheme is None:
            self.mode_combo.addItem("Flat - Tiny CNN", "flat_tiny")
            self.mode_combo.addItem("Flat - YOLO-classify", "flat_yolo")
            return
        labels = {
            "flat_tiny": "Flat - Tiny CNN",
            "flat_yolo": "Flat - YOLO-classify",
            "multihead_tiny": "Multi-head - Tiny CNN (one model per factor)",
            "multihead_yolo": "Multi-head - YOLO-classify (one model per factor)",
        }
        for key in ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]:
            if key in self._scheme.training_modes:
                self.mode_combo.addItem(labels[key], key)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData() or ""
        is_yolo = "yolo" in mode
        is_tiny = "tiny" in mode

        # Show/hide YOLO base-model form row
        self.base_model_combo.setVisible(is_yolo)
        if hasattr(self, "_base_model_row_label"):
            self._base_model_row_label.setVisible(is_yolo)

        # Show/hide Tiny Architecture tab
        if hasattr(self, "_tiny_tab_idx"):
            self.tabs.setTabVisible(self._tiny_tab_idx, is_tiny)
            # If tiny tab was active and is now hidden, fall back to General
            if not is_tiny and self.tabs.currentIndex() == self._tiny_tab_idx:
                self.tabs.setCurrentIndex(0)

        # Update mode description
        _desc = {
            "flat_tiny": (
                "Tiny CNN — lightweight MLP trained on image crops. Fast to train; "
                "ideal for rapid iteration and CPU-only environments."
            ),
            "flat_yolo": (
                "YOLO Classify — fine-tuned pretrained backbone. Higher accuracy; "
                "GPU strongly recommended. Slower to train."
            ),
            "multihead_tiny": (
                "Multi-head Tiny CNN — one independent model per factor in the labeling scheme. "
                "Each factor is trained separately."
            ),
            "multihead_yolo": (
                "Multi-head YOLO Classify — one fine-tuned backbone per factor. "
                "Highest accuracy; GPU required."
            ),
        }
        if hasattr(self, "_mode_desc_label"):
            self._mode_desc_label.setText(_desc.get(mode, ""))

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.cancel()

    def _on_rebalance_mode_changed(self):
        mode = str(self.tiny_rebalance_combo.currentData() or "none")
        enable_power = mode != "none"
        self.tiny_rebalance_power_spin.setEnabled(enable_power)

    def _sync_expansion_constraints(self) -> None:
        """Keep label-expansion settings valid and explicit in the UI."""
        expansion_enabled = bool(self._exp_group.isChecked())

        if expansion_enabled and self.space_combo.currentData() != "canonical":
            idx = self.space_combo.findData("canonical")
            if idx >= 0:
                self.space_combo.blockSignals(True)
                self.space_combo.setCurrentIndex(idx)
                self.space_combo.blockSignals(False)

        # Expansion in this workflow is deterministic augmentation, so disable
        # random flips/rotations to avoid conflicting semantics.
        for widget in (self.flip_lr_spin, self.flip_ud_spin, self.rotate_spin):
            widget.setEnabled(not expansion_enabled)

        if expansion_enabled:
            self.flip_lr_spin.setValue(0.0)
            self.flip_ud_spin.setValue(0.0)
            self.rotate_spin.setValue(0.0)
            self.space_combo.setEnabled(False)
            self._exp_constraints_label.setText(
                "Label expansion ON: canonical space is locked and all random augmentations are disabled."
            )
        else:
            self.space_combo.setEnabled(True)
            self._exp_constraints_label.setText(
                "Label expansion OFF: choose canonical/original space and augmentation probabilities freely."
            )

    def append_log(self, msg: str):
        if msg:
            self.log_view.appendPlainText(msg)
            self.log_view.ensureCursorVisible()

    def get_settings(self) -> dict:
        # Build label_expansion dict from UI rows (only when group is checked)
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
        training_space = self.space_combo.currentData()
        if expansion_enabled:
            training_space = "canonical"

        # Label expansion creates deterministic flipped copies with remapped
        # labels, so stochastic augmentations are disabled while it is enabled.
        flipud_value = 0.0 if expansion_enabled else self.flip_ud_spin.value()
        fliplr_value = 0.0 if expansion_enabled else self.flip_lr_spin.value()
        rotate_value = 0.0 if expansion_enabled else self.rotate_spin.value()

        _rt = str(self.compute_runtime_combo.currentData() or "cpu")
        _train_device = str(self.device_combo.currentData() or "cpu")
        if _train_device == "rocm":
            # PyTorch ROCm uses the CUDA device namespace.
            _train_device = "cuda"

        return {
            "mode": self.mode_combo.currentData(),
            "compute_runtime": _rt,
            "device": _train_device,
            "base_model": self.base_model_combo.currentData(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "lr": self.lr_spin.value(),
            "val_fraction": self.val_fraction_spin.value(),
            "patience": self.patience_spin.value(),
            # Tiny Architecture
            "tiny_layers": self.tiny_layers_spin.value(),
            "tiny_dim": self.tiny_dim_spin.value(),
            "tiny_dropout": self.tiny_dropout_spin.value(),
            "tiny_width": self.tiny_width_spin.value(),
            "tiny_height": self.tiny_height_spin.value(),
            "tiny_rebalance_mode": self.tiny_rebalance_combo.currentData(),
            "tiny_rebalance_power": self.tiny_rebalance_power_spin.value(),
            "tiny_label_smoothing": self.tiny_label_smoothing_spin.value(),
            # Space & Augmentations
            "training_space": training_space,
            "flipud": flipud_value,
            "fliplr": fliplr_value,
            "rotate": rotate_value,
            # Label-switching expansion
            "label_expansion": label_expansion,
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


class ModelHistoryDialog(QDialog):
    """Browse, rename, delete, and load trained models from project history."""

    def __init__(
        self,
        model_entries: list,
        project_path=None,
        db_path: Optional[Path] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Previously Trained Models")
        self.setMinimumSize(940, 620)
        self.setStyleSheet("background-color: #1e1e1e; color: #cccccc;")
        self._entries = list(model_entries)
        self._project_path = project_path
        self._db_path = Path(db_path) if db_path else None
        self._selected_entry_data = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        header = QLabel(
            "Select, rename, delete, or export trained checkpoints before loading one for inference."
        )
        header.setStyleSheet("color: #aaaaaa; font-size: 12px;")
        layout.addWidget(header)

        from PySide6.QtCore import Qt as _Qt
        from PySide6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

        self._qt_align = _Qt.AlignVCenter | _Qt.AlignLeft
        self._table_item_type = QTableWidgetItem

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Timestamp", "Mode", "Classes", "Val Acc", "Canonicalize"]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            5, QHeaderView.ResizeToContents
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #252526; color: #cccccc; "
            "gridline-color: #3a3a3a; border: none; }"
            "QTableWidget::item:selected { background-color: #094771; }"
            "QHeaderView::section { background-color: #2d2d2d; color: #cccccc; "
            "padding: 4px; border: none; border-right: 1px solid #3a3a3a; }"
        )
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        # Detail panel
        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setFixedHeight(104)
        self.detail_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail_label.setStyleSheet(
            "background: #252526; color: #cccccc; font-size: 11px; "
            "padding: 7px 10px; border-radius: 4px;"
        )
        layout.addWidget(self.detail_label)

        btn_row = QHBoxLayout()

        self.rename_btn = QPushButton("Rename")
        self.rename_btn.setFixedHeight(36)
        self.rename_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #cccccc; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        self.rename_btn.clicked.connect(self._rename_selected)
        btn_row.addWidget(self.rename_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setFixedHeight(36)
        self.delete_btn.setStyleSheet(
            "QPushButton { background-color: #553333; color: #ffcccc; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #7a3f3f; color: white; }"
        )
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.delete_btn)

        self.export_btn = QPushButton("Export to models/")
        self.export_btn.setFixedHeight(36)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #cccccc; border-radius: 4px; "
            "padding: 0 14px; }"
            "QPushButton:hover { background-color: #4a6b4a; color: white; }"
        )
        self.export_btn.setToolTip(
            "Copy this model's artifact files to the project models/ directory"
        )
        self.export_btn.clicked.connect(self._export_selected)
        btn_row.addWidget(self.export_btn)

        btn_row.addStretch(1)

        self.load_btn = QPushButton("Load and Run Inference")
        self.load_btn.setFixedHeight(36)
        self.load_btn.setStyleSheet(
            "QPushButton { background-color: #0e639c; color: white; border-radius: 4px; "
            "padding: 0 16px; font-weight: bold; }"
            "QPushButton:hover { background-color: #1177bb; }"
            "QPushButton:disabled { background-color: #3a3a3a; color: #666; }"
        )
        self.load_btn.clicked.connect(self._accept_selection)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedHeight(36)
        cancel_btn.setStyleSheet(
            "QPushButton { background-color: #3a3a3a; color: #cccccc; border-radius: 4px; "
            "padding: 0 16px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
        )
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.load_btn)
        layout.addLayout(btn_row)

        self._refresh_table(select_row=0)

    def _display_name(self, entry: dict) -> str:
        name = str(entry.get("display_name") or "").strip()
        if name:
            return name
        return str(entry.get("mode") or "model")

    def _refresh_table(self, select_row: int | None = None) -> None:
        self.table.setRowCount(0)
        for row, entry in enumerate(self._entries):
            self.table.insertRow(row)
            ts = str(entry.get("timestamp", ""))[:19]
            mode = str(entry.get("mode", ""))
            names = ", ".join(entry.get("class_names") or [])
            acc = entry.get("best_val_acc")
            acc_str = f"{acc:.3f}" if acc is not None else "—"
            canon = "Yes" if entry.get("canonicalize_mat") else "No"
            display_name = self._display_name(entry)

            values = [display_name, ts, mode, names, acc_str, canon]
            for col, text in enumerate(values):
                item = self._table_item_type(text)
                item.setTextAlignment(self._qt_align)
                self.table.setItem(row, col, item)

        if self._entries:
            target = (
                0
                if select_row is None
                else max(0, min(select_row, len(self._entries) - 1))
            )
            self.table.selectRow(target)
        else:
            self.detail_label.setText("No model history entries remain.")
            self.rename_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.load_btn.setEnabled(False)

    def _selected_row(self) -> int:
        return int(self.table.currentRow())

    def _get_selected_entry(self) -> dict | None:
        row = self._selected_row()
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def _on_double_click(self, index):
        self._accept_selection()

    def _accept_selection(self):
        entry = self._get_selected_entry()
        if entry is not None:
            self._selected_entry_data = entry
            self.accept()

    def selected_entry(self):
        """Return the chosen model cache entry dict, or None if dialog was cancelled."""
        return self._selected_entry_data

    def _on_selection_changed(self):
        """Update detail panel and button states on row changes."""
        entry = self._get_selected_entry()
        has_entry = entry is not None
        self.rename_btn.setEnabled(has_entry)
        self.delete_btn.setEnabled(has_entry)
        self.export_btn.setEnabled(has_entry)
        self.load_btn.setEnabled(has_entry)

        if not has_entry:
            self.detail_label.setText("")
            return

        ts = str(entry.get("timestamp", ""))
        mode = str(entry.get("mode", ""))
        display_name = self._display_name(entry)
        names = entry.get("class_names") or []
        acc = entry.get("best_val_acc")
        acc_str = f"{acc:.4f}" if acc is not None else "—"
        artifact_paths = entry.get("artifact_paths") or []
        filenames = [Path(p).name for p in artifact_paths]
        classes_html = ", ".join(names) if names else "—"
        files_html = (
            "  ".join(f"<span style='color:#9cdcfe'>{f}</span>" for f in filenames)
            if filenames
            else "—"
        )
        canon = "Yes" if entry.get("canonicalize_mat") else "No"
        self.detail_label.setText(
            f"<b>{display_name}</b> &nbsp;&bull;&nbsp; Mode: <b>{mode}</b>"
            f" &nbsp;&bull;&nbsp; Val acc: <b>{acc_str}</b>"
            f" &nbsp;&bull;&nbsp; Canonicalize: <b>{canon}</b><br>"
            f"<span style='color:#888'>Classes:</span> {classes_html}<br>"
            f"<span style='color:#888'>Files:</span> {files_html}<br>"
            f"<span style='color:#555;font-size:10px'>{ts}</span>"
        )

    def _rename_selected(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        if self._db_path is None:
            QMessageBox.warning(
                self, "Rename Failed", "No project database is available."
            )
            return

        current_name = self._display_name(entry)
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model Entry",
            "Display name:",
            text=current_name,
        )
        if not ok:
            return

        from ..store.db import ClassKitDB

        updated = ClassKitDB(self._db_path).set_model_cache_display_name(
            int(entry.get("id", -1)),
            str(new_name).strip(),
        )
        if updated < 1:
            QMessageBox.warning(
                self,
                "Rename Failed",
                "Could not update this history entry in the database.",
            )
            return

        entry["display_name"] = str(new_name).strip()
        row = self._selected_row()
        self._refresh_table(select_row=row)

    def _delete_selected(self) -> None:
        entry = self._get_selected_entry()
        if entry is None:
            return
        if self._db_path is None:
            QMessageBox.warning(
                self, "Delete Failed", "No project database is available."
            )
            return

        prompt = QMessageBox(self)
        prompt.setIcon(QMessageBox.Warning)
        prompt.setWindowTitle("Delete Model Entry")
        prompt.setText("Delete this checkpoint entry from model history?")
        prompt.setInformativeText(
            "You can remove only the DB record, or delete both the record and artifact files."
        )
        remove_record_btn = prompt.addButton(
            "Remove History Entry", QMessageBox.AcceptRole
        )
        delete_files_btn = prompt.addButton(
            "Delete Entry + Files", QMessageBox.DestructiveRole
        )
        prompt.addButton(QMessageBox.Cancel)
        prompt.setDefaultButton(remove_record_btn)
        prompt.exec()

        clicked = prompt.clickedButton()
        if clicked not in {remove_record_btn, delete_files_btn}:
            return

        delete_files = clicked == delete_files_btn
        deleted_files = 0
        file_errors: list[str] = []

        if delete_files:
            for src in entry.get("artifact_paths") or []:
                src_path = Path(src)
                if not src_path.exists():
                    continue
                try:
                    src_path.unlink()
                    deleted_files += 1
                except Exception as exc:
                    file_errors.append(f"{src_path.name}: {exc}")

        from ..store.db import ClassKitDB

        deleted_rows = ClassKitDB(self._db_path).delete_model_cache_entry(
            int(entry.get("id", -1))
        )
        if deleted_rows < 1:
            QMessageBox.warning(
                self,
                "Delete Failed",
                "Could not remove this history entry from the database.",
            )
            return

        keep_id = int(entry.get("id", -1))
        self._entries = [e for e in self._entries if int(e.get("id", -1)) != keep_id]
        row = self._selected_row()
        self._refresh_table(select_row=max(0, row - 1))

        if file_errors:
            QMessageBox.warning(
                self,
                "Entry Deleted With Warnings",
                f"Removed history entry. Deleted files: {deleted_files}\n\n"
                + "\n".join(file_errors),
            )
        elif delete_files:
            self.detail_label.setText(
                f"<span style='color:#4ec9b0'>Removed history entry and deleted {deleted_files} artifact file(s).</span>"
            )
        else:
            self.detail_label.setText(
                "<span style='color:#4ec9b0'>Removed history entry from the database.</span>"
            )

    def _export_selected(self):
        """Copy selected artifacts to project ``models/`` using descriptive names."""
        import shutil

        entry = self._get_selected_entry()
        if entry is None:
            return
        artifact_paths = entry.get("artifact_paths") or []
        if not artifact_paths:
            QMessageBox.warning(
                self, "No Artifacts", "No model files are recorded for this entry."
            )
            return

        # Export is intentionally non-interactive: always use project-local models/.
        if self._project_path:
            dest_dir = Path(str(self._project_path)) / "models"
        else:
            dest_dir = Path.cwd() / "models"
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Export Failed",
                f"Could not create export directory:\n{dest_dir}\n\n{exc}",
            )
            return

        def _slug(text: object, max_len: int = 48) -> str:
            raw = str(text or "").strip().lower()
            out = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
            if not out:
                return ""
            while "__" in out:
                out = out.replace("__", "_")
            return out[:max_len]

        mode_slug = _slug(entry.get("mode", "model"), max_len=32) or "model"
        class_names = entry.get("class_names") or []
        class_slug = "-".join(
            part for part in (_slug(name, max_len=16) for name in class_names) if part
        )
        if not class_slug:
            class_slug = "classes"
        if len(class_slug) > 48:
            class_slug = f"{max(1, len(class_names))}cls"
        ts_raw = str(entry.get("timestamp", "")).strip()
        ts_slug = "".join(ch for ch in ts_raw if ch.isdigit())[:14] or "unknown"
        base_stem = f"classkit_{mode_slug}_{class_slug}_{ts_slug}"

        copied, failed = [], []
        total_artifacts = len(artifact_paths)
        for idx, src in enumerate(artifact_paths):
            src_path = Path(src)
            if src_path.exists():
                ext = src_path.suffix or ".bin"
                hint = _slug(src_path.stem, max_len=24)
                if hint in {"", "best", "last", "weights", "model", "checkpoint"}:
                    hint = ""

                if total_artifacts == 1:
                    desired_name = f"{base_stem}{ext}"
                else:
                    desired_name = (
                        f"{base_stem}_{hint}{ext}"
                        if hint
                        else f"{base_stem}_f{idx + 1}{ext}"
                    )

                dst = dest_dir / desired_name
                suffix_counter = 1
                while dst.exists():
                    dst = dest_dir / f"{dst.stem}_{suffix_counter}{ext}"
                    suffix_counter += 1

                try:
                    shutil.copy2(str(src_path), str(dst))
                    copied.append(dst.name)
                except Exception as exc:
                    failed.append(f"{src_path.name}: {exc}")
            else:
                failed.append(f"{src_path.name}: file not found")

        if copied:
            self.detail_label.setText(
                f"<span style='color:#4ec9b0'>Exported → {dest_dir}:</span> "
                + "  ".join(f"<b>{f}</b>" for f in copied)
                + (
                    f"<br><span style='color:#f88'>Skipped: {'; '.join(failed)}</span>"
                    if failed
                    else ""
                )
            )
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {len(copied)} file(s) to:\n{dest_dir}"
                + ("\n\nSkipped:\n" + "\n".join(failed) if failed else ""),
            )
        else:
            QMessageBox.warning(
                self,
                "Export Failed",
                "No model files could be exported.\n\n" + "\n".join(failed),
            )
