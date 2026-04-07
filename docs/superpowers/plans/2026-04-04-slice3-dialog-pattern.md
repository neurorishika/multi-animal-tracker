# Slice 3 — Dialog Pattern Unification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce `BaseDialog(QDialog)` in `hydra_suite.widgets.dialogs`, split `classkit/gui/dialogs.py` (11 classes, 3 282 lines) into 11 individual files, and migrate all dialogs across the codebase to inherit from `BaseDialog`.

**Architecture:** `BaseDialog` owns the dark stylesheet, modal flag, window title, and `QDialogButtonBox`. Subclasses call `add_content()` to insert their layout. Custom button sets are passed via the `buttons` constructor argument. `classkit/gui/dialogs.py` is deleted and replaced by a `classkit/gui/dialogs/` package.

**Tech Stack:** PySide6 (`QDialog`, `QDialogButtonBox`, `QVBoxLayout`), pytest

---

### Task 1: Create BaseDialog

**Files:**
- Create: `src/hydra_suite/widgets/dialogs.py`
- Modify: `src/hydra_suite/widgets/__init__.py`
- Create: `tests/test_base_dialog.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_base_dialog.py
import sys
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication, QLabel, QDialogButtonBox


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_base_dialog_has_button_box(qapp):
    """BaseDialog always creates a button box."""
    from hydra_suite.widgets.dialogs import BaseDialog
    dlg = BaseDialog("Test Dialog")
    assert dlg.windowTitle() == "Test Dialog"
    assert dlg._buttons is not None


def test_base_dialog_is_modal(qapp):
    """BaseDialog is modal by default."""
    from hydra_suite.widgets.dialogs import BaseDialog
    dlg = BaseDialog("Modal Test")
    assert dlg.isModal()


def test_base_dialog_add_content(qapp):
    """add_content inserts a widget above the button box."""
    from hydra_suite.widgets.dialogs import BaseDialog
    dlg = BaseDialog("Content Test")
    label = QLabel("hello")
    dlg.add_content(label)
    # label should be findable as a child
    found = dlg.findChild(QLabel)
    assert found is label


def test_base_dialog_custom_buttons(qapp):
    """ok_only variant has only Ok button."""
    from hydra_suite.widgets.dialogs import BaseDialog
    dlg = BaseDialog("Ok Only", buttons=QDialogButtonBox.Ok)
    ok_btn = dlg._buttons.button(QDialogButtonBox.Ok)
    cancel_btn = dlg._buttons.button(QDialogButtonBox.Cancel)
    assert ok_btn is not None
    assert cancel_btn is None


def test_base_dialog_subclass(qapp):
    """Subclasses can add content in __init__ and accept/reject works."""
    from hydra_suite.widgets.dialogs import BaseDialog
    from PySide6.QtWidgets import QLineEdit

    class _NameDialog(BaseDialog):
        def __init__(self, parent=None):
            super().__init__("Enter Name", parent=parent)
            self._edit = QLineEdit()
            self.add_content(self._edit)

        def value(self) -> str:
            return self._edit.text()

    dlg = _NameDialog()
    dlg._edit.setText("hydra")
    assert dlg.value() == "hydra"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_base_dialog.py -v
```

Expected: `ImportError` — `hydra_suite.widgets.dialogs` does not exist.

- [ ] **Step 3: Create `src/hydra_suite/widgets/dialogs.py`**

```python
# src/hydra_suite/widgets/dialogs.py
"""BaseDialog — standard QDialog base class for all kit dialogs."""
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QWidget,
)

_DARK_STYLE = """
QDialog {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QLabel {
    color: #e0e0e0;
}
QGroupBox {
    color: #aaaaaa;
    border: 1px solid #333333;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
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
"""


class BaseDialog(QDialog):
    """Base class for all kit dialogs.

    Subclasses call ``add_content(widget)`` to insert their UI above the
    button box.  The button box is created automatically.

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
        self._content_layout = QVBoxLayout()
        self._root_layout.addLayout(self._content_layout)

        self._buttons = QDialogButtonBox(buttons)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._root_layout.addWidget(self._buttons)

    def add_content(self, widget: QWidget) -> None:
        """Insert *widget* above the button box."""
        self._content_layout.addWidget(widget)
```

- [ ] **Step 4: Export from `widgets/__init__.py`**

Read the current `src/hydra_suite/widgets/__init__.py` and add:

```python
from hydra_suite.widgets.dialogs import BaseDialog

__all__ = [...existing exports..., "BaseDialog"]
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_base_dialog.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/widgets/dialogs.py src/hydra_suite/widgets/__init__.py tests/test_base_dialog.py
git commit -m "feat: add BaseDialog(QDialog) to widgets with standard dark-theme interface"
```

---

### Task 2: Split classkit/gui/dialogs.py into a package

**Files:**
- Create: `src/hydra_suite/classkit/gui/dialogs/__init__.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/add_source.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/source_manager.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/class_editor.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/shortcut_editor.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/new_project.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/embedding.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/cluster.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/training.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/export.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/model_history.py`
- Create: `src/hydra_suite/classkit/gui/dialogs/apriltag_autolabel.py`
- Delete: `src/hydra_suite/classkit/gui/dialogs.py`

**Note:** `classkit/gui/` currently has `dialogs.py` as a file. It must be converted to a package (`dialogs/`). Python cannot have both a `dialogs.py` file and a `dialogs/` directory at the same path — delete the file first, then create the directory.

- [ ] **Step 1: Read `classkit/gui/dialogs.py` fully**

Open `src/hydra_suite/classkit/gui/dialogs.py` and note:
- The module-level imports (all of them will be needed in individual files)
- The `_KeyCapture` helper class (shared by multiple dialogs — goes into `dialogs/_helpers.py`)
- The `_DARK_STYLE` constant (no longer needed — `BaseDialog` provides the style)
- The exact class boundaries (line numbers from the grep: 161, 335, 666, 1059, 1175, 1386, 1515, 1583, 2489, 2627, 3099)

- [ ] **Step 2: Create `dialogs/_helpers.py` for the `_KeyCapture` widget**

```python
# src/hydra_suite/classkit/gui/dialogs/_helpers.py
"""Shared internal helpers for classkit dialogs."""
# Move the _KeyCapture class verbatim from the old dialogs.py here.
# _KeyCapture is a QLineEdit subclass that replaces QKeySequenceEdit
# to avoid a macOS TSM crash with Python 3.13 + Qt6.
```

Copy the `_KeyCapture` class (and its imports) verbatim from `dialogs.py` to this file.

- [ ] **Step 3: Create each dialog file**

For each class, create its file. Use this template for each:

```python
# src/hydra_suite/classkit/gui/dialogs/add_source.py
"""AddSourceDialog — add a new source to the classkit project."""
# Imports: copy only what AddSourceDialog needs from the old dialogs.py
from PySide6.QtWidgets import ...
from hydra_suite.widgets.dialogs import BaseDialog
# ... other imports

class AddSourceDialog(BaseDialog):
    def __init__(self, ..., parent=None):
        super().__init__("Add Source", parent=parent)
        # Move the body of the old __init__ here,
        # replacing QDialog.__init__ call and manual stylesheet with
        # the super().__init__ call above.
        # Replace self._layout / self.buttonBox setup with self.add_content().
        ...
```

Repeat for all 11 dialog classes. Class-to-file mapping:

| Class | File | Old line |
|-------|------|----------|
| `AddSourceDialog` | `add_source.py` | 161 |
| `SourceManagerDialog` | `source_manager.py` | 335 |
| `ClassEditorDialog` | `class_editor.py` | 666 |
| `ShortcutEditorDialog` | `shortcut_editor.py` | 1059 |
| `NewProjectDialog` | `new_project.py` | 1175 |
| `EmbeddingDialog` | `embedding.py` | 1386 |
| `ClusterDialog` | `cluster.py` | 1515 |
| `ClassKitTrainingDialog` | `training.py` | 1583 |
| `ExportDialog` | `export.py` | 2489 |
| `ModelHistoryDialog` | `model_history.py` | 2627 |
| `AprilTagAutoLabelDialog` | `apriltag_autolabel.py` | 3099 |

**For each dialog, update its `__init__`:** Replace the old `QDialog.__init__` + manual `setModal(True)` + `setStyleSheet(_DARK_STYLE)` + manual `QDialogButtonBox` setup with:
```python
super().__init__("Dialog Title", parent=parent)
```
Then call `self.add_content(main_widget)` instead of directly adding to `self._layout`.

- [ ] **Step 4: Create `dialogs/__init__.py` to re-export everything**

```python
# src/hydra_suite/classkit/gui/dialogs/__init__.py
"""classkit dialog package — one dialog per file."""
from hydra_suite.classkit.gui.dialogs.add_source import AddSourceDialog
from hydra_suite.classkit.gui.dialogs.apriltag_autolabel import AprilTagAutoLabelDialog
from hydra_suite.classkit.gui.dialogs.class_editor import ClassEditorDialog
from hydra_suite.classkit.gui.dialogs.cluster import ClusterDialog
from hydra_suite.classkit.gui.dialogs.embedding import EmbeddingDialog
from hydra_suite.classkit.gui.dialogs.export import ExportDialog
from hydra_suite.classkit.gui.dialogs.model_history import ModelHistoryDialog
from hydra_suite.classkit.gui.dialogs.new_project import NewProjectDialog
from hydra_suite.classkit.gui.dialogs.shortcut_editor import ShortcutEditorDialog
from hydra_suite.classkit.gui.dialogs.source_manager import SourceManagerDialog
from hydra_suite.classkit.gui.dialogs.training import ClassKitTrainingDialog

__all__ = [
    "AddSourceDialog",
    "AprilTagAutoLabelDialog",
    "ClassEditorDialog",
    "ClusterDialog",
    "EmbeddingDialog",
    "ExportDialog",
    "ModelHistoryDialog",
    "NewProjectDialog",
    "ShortcutEditorDialog",
    "SourceManagerDialog",
    "ClassKitTrainingDialog",
]
```

- [ ] **Step 5: Delete `classkit/gui/dialogs.py`**

```bash
git rm src/hydra_suite/classkit/gui/dialogs.py
```

- [ ] **Step 6: Verify imports in classkit main_window still work**

Search for all imports of `classkit.gui.dialogs` across the codebase:

```bash
grep -rn "from hydra_suite.classkit.gui.dialogs" src/ --include="*.py"
grep -rn "from hydra_suite.classkit.gui import dialogs" src/ --include="*.py"
```

Any import that used to import from `dialogs.py` still works because `dialogs/__init__.py` re-exports everything by the same names.

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 8: Commit**

```bash
git add src/hydra_suite/classkit/gui/dialogs/ tests/
git commit -m "refactor: split classkit/gui/dialogs.py into 11-file dialogs/ package with BaseDialog"
```

---

### Task 3: Migrate remaining dialogs across kits to BaseDialog

**Files:**
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/run_history_dialog.py` (line 46)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/train_yolo_dialog.py` (line 129)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py` (line 82)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/cnn_identity_import_dialog.py` (line 8)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/model_test_dialog.py` (line 197)
- Modify: `src/hydra_suite/trackerkit/gui/dialogs/bg_parameter_helper.py` (line 91)
- Modify: `src/hydra_suite/refinekit/gui/dialogs/track_editor_dialog.py` (line 145)
- Modify: `src/hydra_suite/refinekit/gui/dialogs/merge_wizard.py` (line 1035)
- Modify: `src/hydra_suite/refinekit/gui/dialogs/bbox_selector.py` (line 110)

For each dialog the migration is:
1. Add `from hydra_suite.widgets.dialogs import BaseDialog`
2. Change `class FooDialog(QDialog):` → `class FooDialog(BaseDialog):`
3. Replace the `__init__` preamble:
   - Remove `super().__init__(parent)` and any `self.setModal(True)`, `self.setWindowTitle(...)`, `self.setStyleSheet(...)`, manual `QDialogButtonBox` creation
   - Replace with `super().__init__("Title", parent=parent)`
4. Replace `self.layout.addWidget(main_widget)` and `self.layout.addWidget(self.button_box)` patterns with `self.add_content(main_widget)`

- [ ] **Step 1: Migrate trackerkit dialogs (6 dialogs)**

For each of the 6 files listed above, apply the migration described. Read each file to understand its exact `__init__` before modifying.

After all 6 are done, run:

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 2: Migrate refinekit dialogs (3 dialogs)**

Apply the same migration to the 3 refinekit dialogs.

After all 3:

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 3: Find any remaining posekit/detectkit dialogs to migrate**

```bash
grep -rn "class .*Dialog(QDialog)" src/hydra_suite/posekit src/hydra_suite/detectkit --include="*.py"
```

For any result, apply the same migration (add import, change parent class to `BaseDialog`, replace `__init__` preamble with `super().__init__("Title", parent=parent)`, use `add_content()`).

- [ ] **Step 4: Verify no bare QDialog subclasses remain in standalone dialog files**

```bash
grep -rn "class .*Dialog(QDialog)" src/hydra_suite/ --include="*.py"
```

Any result here is a missed migration — fix it. Exception: dialogs defined inline inside a method body (like `ContrastSettingsDialog` inside `classkit/gui/main_window.py:3281`) — these are acceptable as-is and should be left alone.

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/trackerkit/gui/dialogs/ src/hydra_suite/refinekit/gui/dialogs/ src/hydra_suite/posekit/ src/hydra_suite/detectkit/
git commit -m "refactor: migrate all standalone dialogs to BaseDialog"
```

---

### Task 4: Verify and finalize

- [ ] **Step 1: Confirm `dialogs.py` is gone from classkit**

```bash
ls src/hydra_suite/classkit/gui/
```

Expected: `dialogs/` directory, no `dialogs.py` file.

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 3: Commit completion marker**

```bash
git commit --allow-empty -m "chore: Slice 3 (dialog unification) complete"
```
