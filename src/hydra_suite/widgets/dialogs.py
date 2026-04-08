# src/hydra_suite/widgets/dialogs.py
"""BaseDialog — standard QDialog base class for all kit dialogs."""

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

_DARK_STYLE = """
QDialog {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QLabel {
    color: #e0e0e0;
}
QLineEdit,
QPlainTextEdit,
QListWidget,
QComboBox,
QDoubleSpinBox {
    background-color: #252526;
    color: #e0e0e0;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px;
}
QLineEdit:focus,
QPlainTextEdit:focus,
QListWidget:focus,
QComboBox:focus,
QDoubleSpinBox:focus {
    border: 1px solid #0e639c;
}
QGroupBox {
    color: #cfcfcf;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #2d2d2d;
    color: #e0e0e0;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 4px 12px;
}
QPushButton:hover {
    background-color: #3a3a3a;
}
QPushButton:pressed {
    background-color: #1a1a1a;
}
QPushButton:disabled {
    background-color: #252526;
    color: #7f7f7f;
    border-color: #333333;
}
QCheckBox {
    color: #d0d0d0;
}
QFrame {
    color: #3e3e42;
}
"""


class BaseDialog(QDialog):
    """Base class for all kit dialogs.

    Subclasses call ``add_content(widget)`` to insert their UI above the
    button box.  The button box is always created automatically in the
    correct position.

    Parameters
    ----------
    title:
        Window title shown in the title bar.
    parent:
        Parent widget (passed to QDialog).
    buttons:
        ``QDialogButtonBox.StandardButtons`` flags.  Defaults to
        ``Ok | Cancel``.
    apply_dark_style:
        Whether to apply the shared dark stylesheet.  Default ``True``.
    """

    def __init__(
        self,
        title: str,
        parent: QWidget | None = None,
        buttons: QDialogButtonBox.StandardButtons = (
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        ),
        apply_dark_style: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        if apply_dark_style:
            self.setStyleSheet(_DARK_STYLE)

        self._root_layout = QVBoxLayout(self)
        self._root_layout.setContentsMargins(16, 16, 16, 16)
        self._root_layout.setSpacing(16)
        self._content_layout = QVBoxLayout()
        self._content_layout.setSpacing(12)
        self._root_layout.addLayout(self._content_layout)

        self._buttons = QDialogButtonBox(buttons)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._root_layout.addWidget(self._buttons)

    def add_content(self, widget: QWidget) -> None:
        """Insert *widget* above the button box."""
        self._content_layout.addWidget(widget)
