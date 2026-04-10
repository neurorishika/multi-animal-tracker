"""Tests for the standalone PoseKit new-project dialog."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QDialogButtonBox, QScrollArea  # noqa: E402

from hydra_suite.posekit.gui.dialogs.project_wizard import (  # noqa: E402
    NewProjectDialog,
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


def test_posekit_new_project_dialog_defaults_to_posekit_projects_root(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    create_button = dialog._buttons.button(QDialogButtonBox.Ok)

    assert dialog._le_parent.text() == str(tmp_path / "hydra-projects" / "PoseKit")
    assert create_button.isEnabled() is False

    dialog._le_name.setText("ant_pose_project")

    assert (
        dialog.get_project_location()
        == tmp_path / "hydra-projects" / "PoseKit" / "ant_pose_project"
    )
    assert create_button.isEnabled() is True


def test_posekit_new_project_dialog_enables_create_after_name_entered(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    create_button = dialog._buttons.button(QDialogButtonBox.Ok)
    dialog._le_name.setText("ant_pose_project")
    dialog._refresh_create_btn()

    assert create_button.isEnabled() is True
    assert dialog.get_classes() == ["object"]
    assert dialog.get_keypoints() == ["kp1", "kp2"]
    assert dialog.get_options() == (True, 0.03)


def test_posekit_new_project_dialog_wraps_form_in_scroll_area(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("HYDRA_PROJECTS_DIR", str(tmp_path / "hydra-projects"))

    dialog = NewProjectDialog()
    scroll_areas = dialog.findChildren(QScrollArea)

    assert scroll_areas
    assert any(scroll.widgetResizable() for scroll in scroll_areas)
