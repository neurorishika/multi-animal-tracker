# tests/test_base_dialog.py
import sys

import pytest


@pytest.fixture()
def qapp():
    """Provide a QApplication for dialog tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_base_dialog_has_title_and_is_modal(qapp):
    """BaseDialog sets title and modal flag."""
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = BaseDialog("Test Title")
    assert dlg.windowTitle() == "Test Title"
    assert dlg.isModal()


def test_base_dialog_has_button_box(qapp):
    """BaseDialog always creates a button box."""
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = BaseDialog("Test")
    assert dlg._buttons is not None


def test_base_dialog_add_content(qapp):
    """add_content inserts a widget above the button box."""
    from PySide6.QtWidgets import QLabel

    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = BaseDialog("Content Test")
    label = QLabel("hello")
    dlg.add_content(label)
    found = dlg.findChild(QLabel)
    assert found is label


def test_base_dialog_custom_buttons_ok_only(qapp):
    """ok_only variant has Ok but no Cancel button."""
    from PySide6.QtWidgets import QDialogButtonBox

    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = BaseDialog("Ok Only", buttons=QDialogButtonBox.Ok)
    ok_btn = dlg._buttons.button(QDialogButtonBox.Ok)
    cancel_btn = dlg._buttons.button(QDialogButtonBox.Cancel)
    assert ok_btn is not None
    assert cancel_btn is None


def test_base_dialog_subclass_pattern(qapp):
    """Subclasses call add_content() and the dialog works."""
    from PySide6.QtWidgets import QLineEdit

    from hydra_suite.widgets.dialogs import BaseDialog

    class _NameDialog(BaseDialog):
        def __init__(self, parent=None):
            super().__init__("Enter Name", parent=parent)
            self._edit = QLineEdit()
            self.add_content(self._edit)

        def value(self):
            return self._edit.text()

    dlg = _NameDialog()
    dlg._edit.setText("hydra")
    assert dlg.value() == "hydra"


def test_base_dialog_no_dark_style(qapp):
    """apply_dark_style=False leaves stylesheet empty."""
    from hydra_suite.widgets.dialogs import BaseDialog

    dlg = BaseDialog("Plain", apply_dark_style=False)
    assert dlg.styleSheet() == ""
