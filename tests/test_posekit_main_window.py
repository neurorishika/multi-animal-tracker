"""Regression tests for PoseKit main-window frame switching."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hydra_suite.posekit.gui.main_window import MainWindow
from hydra_suite.posekit.gui.models import FrameAnn


class _DummyCombo:
    def blockSignals(self, _blocked: bool) -> None:
        return None

    def setCurrentIndex(self, _index: int) -> None:
        return None

    def count(self) -> int:
        return 1


class _DummyCanvas:
    def set_current_keypoint(self, _index: int) -> None:
        return None


def test_load_frame_defers_previous_frame_list_refresh_after_save() -> None:
    saved_refresh_flags = []
    scheduled_indices = []
    cache_calls = []

    window = SimpleNamespace(
        image_paths=[Path("frame_0.png"), Path("frame_1.png")],
        current_index=0,
        current_kpt=0,
        _dirty=True,
        _ann=object(),
        _frame_cache={1: FrameAnn(cls=0, bbox_xyxy=None, kpts=[])},
        _img_bgr=None,
        _img_display=None,
        _img_wh=(1, 1),
        mode="frame",
        class_combo=_DummyCombo(),
        canvas=_DummyCanvas(),
        _undo_stack=[],
        _cache_current_frame=lambda: cache_calls.append("cached"),
        save_current=lambda refresh_ui=False: saved_refresh_flags.append(refresh_ui),
        _read_image=lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
        _load_ann_from_disk=lambda _idx: FrameAnn(cls=0, bbox_xyxy=None, kpts=[]),
        _refresh_canvas_image=lambda: None,
        _rebuild_canvas=lambda: None,
        _update_info=lambda: None,
        _load_metadata_ui=lambda: None,
        _schedule_frame_item_refresh=lambda idx: scheduled_indices.append(idx),
        _update_frame_item=lambda _idx: (_ for _ in ()).throw(
            AssertionError("load_frame should defer list-item refresh")
        ),
    )

    MainWindow.load_frame(window, 1)

    assert cache_calls == ["cached"]
    assert saved_refresh_flags == [False]
    assert scheduled_indices == [0]
    assert window.current_index == 1


def test_open_recent_project_uses_posekit_gui_project_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_path = tmp_path / "pose_project.json"
    project_path.write_text("{}", encoding="utf-8")

    opened = []
    switched = []
    sentinel_project = object()

    monkeypatch.setattr(
        "hydra_suite.posekit.gui.main_window.open_project_from_path",
        lambda path: opened.append(path) or sentinel_project,
    )

    window = SimpleNamespace(
        _switch_project_window=lambda project: switched.append(project),
    )

    MainWindow._open_recent_project(window, str(project_path))

    assert opened == [project_path]
    assert switched == [sentinel_project]


def test_recent_project_display_name_prefers_project_directory() -> None:
    path = "/Users/example/projects/ant_pose_project/pose_project.json"

    assert MainWindow._recent_project_display_name(path) == "ant_pose_project"


def test_recent_project_display_name_handles_bundle_state_path() -> None:
    path = "/Users/example/projects/ant_pose_project/state/pose_project.json"

    assert MainWindow._recent_project_display_name(path) == "ant_pose_project"
