from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from hydra_suite.trackerkit.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _fake_conda_env_list(*_args, **_kwargs) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["conda", "env", "list"],
        returncode=0,
        stdout=(
            "# conda environments:\n"
            "base                  *  /fake/base\n"
            "x-anylabeling-alpha      /fake/alpha\n"
            "x-anylabeling-beta       /fake/beta\n"
        ),
        stderr="",
    )


def _make_main_window(
    monkeypatch: pytest.MonkeyPatch,
    advanced_config: dict[str, object] | None = None,
) -> MainWindow:
    monkeypatch.setattr(MainWindow, "_save_advanced_config", lambda self: None)
    monkeypatch.setattr(
        MainWindow,
        "_load_advanced_config",
        lambda self: dict(advanced_config or {}),
    )
    monkeypatch.setattr(subprocess, "run", _fake_conda_env_list)
    return MainWindow()


def _select_first_model_with_suffix(combo, suffix: str) -> str:
    for index in range(combo.count()):
        item_data = combo.itemData(index)
        if isinstance(item_data, str) and item_data.endswith(suffix):
            combo.setCurrentIndex(index)
            return item_data
    raise AssertionError(f"No model ending with {suffix!r} was available in the combo")


def test_headtail_model_type_roundtrip_preserves_tiny_selection(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)
    window.combo_yolo_headtail_model_type.setCurrentText("tiny")
    window._refresh_yolo_headtail_model_combo()

    selected_model = _select_first_model_with_suffix(
        window.combo_yolo_headtail_model,
        ".pth",
    )
    assert selected_model

    config_path = tmp_path / "headtail_roundtrip.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["yolo_headtail_model_type"] == "tiny"
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window.combo_yolo_headtail_model_type.currentText() == "tiny"
    assert reloaded_window._get_selected_yolo_headtail_model_path() == selected_model
    reloaded_window.close()


def test_xanylabeling_env_preference_restores_and_updates(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
) -> None:
    saved_preferences: list[dict[str, object]] = []

    def _record_advanced_config(self: MainWindow) -> None:
        saved_preferences.append(dict(self.advanced_config))

    monkeypatch.setattr(MainWindow, "_save_advanced_config", _record_advanced_config)
    monkeypatch.setattr(
        MainWindow,
        "_load_advanced_config",
        lambda self: {"xanylabeling_env": "x-anylabeling-beta"},
    )
    monkeypatch.setattr(subprocess, "run", _fake_conda_env_list)

    window = MainWindow()

    assert (
        window._dataset_panel.combo_xanylabeling_env.currentText()
        == "x-anylabeling-beta"
    )

    window._dataset_panel.combo_xanylabeling_env.setCurrentText("x-anylabeling-alpha")

    assert window._selected_xanylabeling_env() == "x-anylabeling-alpha"
    assert window.advanced_config["xanylabeling_env"] == "x-anylabeling-alpha"
    assert saved_preferences
    assert saved_preferences[-1]["xanylabeling_env"] == "x-anylabeling-alpha"
    window.close()


def test_confidence_density_toggle_roundtrip_updates_visibility(
    monkeypatch: pytest.MonkeyPatch,
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    window = _make_main_window(monkeypatch)

    assert not window.g_density.isHidden()

    window.chk_enable_confidence_density_map.setChecked(False)

    assert window.g_density.isHidden()
    assert window.get_parameters_dict()["ENABLE_CONFIDENCE_DENSITY_MAP"] is False

    config_path = tmp_path / "confidence_density_toggle.json"
    assert window.save_config(preset_mode=True, preset_path=str(config_path))
    saved_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_cfg["enable_confidence_density_map"] is False
    window.close()

    reloaded_window = _make_main_window(monkeypatch)
    reloaded_window._load_config_from_file(str(config_path), preset_mode=True)

    assert reloaded_window.chk_enable_confidence_density_map.isChecked() is False
    assert reloaded_window.g_density.isHidden()

    reloaded_window.chk_enable_confidence_density_map.setChecked(True)

    assert not reloaded_window.g_density.isHidden()
    reloaded_window.close()
