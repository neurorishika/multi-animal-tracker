"""Tests for ClassKit source-folder selection dialogs."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from hydra_suite.classkit.gui.dialogs.add_source import AddSourceDialog  # noqa: E402
from hydra_suite.classkit.gui.dialogs.source_manager import (  # noqa: E402
    SourceManagerDialog,
)
from hydra_suite.widgets.dialogs import (  # noqa: E402
    HYDRA_DIALOG_MUTED_TEXT_COLOR,
    HYDRA_DIALOG_STYLE,
)


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def _capture_warning(
    monkeypatch: pytest.MonkeyPatch, module_path: str
) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []

    def _warning(_parent, title: str, text: str, *args, **kwargs):
        warnings.append((title, text))
        return QMessageBox.Ok

    monkeypatch.setattr(f"{module_path}.QMessageBox.warning", _warning)
    return warnings


def test_add_source_dialog_rejects_folder_without_images_subdir(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    (source_root / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.add_source"
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == []
    assert warnings == [
        (
            "Invalid Source Folder",
            "Selected source folder must contain an images/ subdirectory, or the "
            "folder itself must contain compatible .jpg / .jpeg / .png files.\n\n"
            f"{source_root}",
        )
    ]


def test_add_source_dialog_uses_images_subdir_when_present(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    images_dir = source_root / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == [
        (source_root.resolve(), images_dir.resolve(), source_root.name)
    ]


def test_add_source_dialog_standardizes_flat_image_folder_when_user_accepts(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    image_path = source_root / "frame001.jpg"
    image_path.write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    monkeypatch.setattr(
        AddSourceDialog,
        "_confirm_standardization",
        lambda self, inspection: True,
    )

    dialog = AddSourceDialog()
    dialog._browse()

    images_dir = source_root / "images"
    copied_image = images_dir / image_path.name
    assert copied_image.exists() is True
    assert copied_image.read_bytes() == image_path.read_bytes()
    assert dialog.sources == [
        (source_root.resolve(), images_dir.resolve(), source_root.name)
    ]


def test_add_source_dialog_does_not_standardize_when_user_declines(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    (source_root / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.add_source.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    monkeypatch.setattr(
        AddSourceDialog,
        "_confirm_standardization",
        lambda self, inspection: False,
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.add_source"
    )

    dialog = AddSourceDialog()
    dialog._browse()

    assert dialog.sources == []
    assert (source_root / "images").exists() is False
    assert warnings == []


def test_source_manager_dialog_rejects_folder_without_images_subdir(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    (source_root / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        SourceManagerDialog,
        "_load_existing_sources",
        lambda self: None,
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.source_manager"
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == []
    assert warnings == [
        (
            "Invalid Source Folder",
            "Selected source folder must contain an images/ subdirectory, or the "
            "folder itself must contain compatible .jpg / .jpeg / .png files.\n\n"
            f"{source_root}",
        )
    ]


def test_source_manager_dialog_standardizes_flat_image_folder_when_user_accepts(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    image_path = source_root / "frame001.jpg"
    image_path.write_bytes(b"not-an-image")

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    monkeypatch.setattr(
        SourceManagerDialog,
        "_confirm_standardization",
        lambda self, inspection: True,
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    images_dir = source_root / "images"
    copied_image = images_dir / image_path.name
    assert copied_image.exists() is True
    assert copied_image.read_bytes() == image_path.read_bytes()
    assert dialog.folders_to_add == [images_dir.resolve()]


def test_source_manager_dialog_does_not_standardize_when_user_declines(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_root = tmp_path / "dataset_root"
    source_root.mkdir()
    (source_root / "frame001.jpg").write_bytes(b"not-an-image")

    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.dialogs.source_manager.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(source_root),
    )
    monkeypatch.setattr(
        SourceManagerDialog,
        "_confirm_standardization",
        lambda self, inspection: False,
    )
    warnings = _capture_warning(
        monkeypatch, "hydra_suite.classkit.gui.dialogs.source_manager"
    )

    dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")
    dialog._browse_add()

    assert dialog.folders_to_add == []
    assert (source_root / "images").exists() is False
    assert warnings == []


def test_source_dialogs_use_shared_hydra_theme(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        SourceManagerDialog, "_load_existing_sources", lambda self: None
    )

    add_dialog = AddSourceDialog()
    manager_dialog = SourceManagerDialog(db_path=tmp_path / "classkit.db")

    assert add_dialog.styleSheet() == HYDRA_DIALOG_STYLE
    assert manager_dialog.styleSheet() == HYDRA_DIALOG_STYLE
    assert HYDRA_DIALOG_MUTED_TEXT_COLOR in manager_dialog._summary.styleSheet()
