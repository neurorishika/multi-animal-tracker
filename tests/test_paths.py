"""Tests for multi_tracker.paths module."""

from __future__ import annotations

import os
from pathlib import Path

from multi_tracker.paths import (
    get_advanced_config_path,
    get_brand_icon_bytes,
    get_bundled_config_names,
    get_bundled_skeleton_names,
    get_default_config,
    get_models_dir,
    get_skeleton_config,
    get_training_runs_dir,
)


def test_get_brand_icon_bytes_returns_bytes():
    data = get_brand_icon_bytes("multianimaltracker.svg")
    assert isinstance(data, bytes)
    assert b"<svg" in data or b"<?xml" in data


def test_get_brand_icon_bytes_missing_returns_none():
    data = get_brand_icon_bytes("nonexistent_icon.svg")
    assert data is None


def test_get_default_config_returns_dict():
    cfg = get_default_config("default.json")
    assert isinstance(cfg, dict)
    assert "preset_name" in cfg


def test_get_default_config_missing_returns_none():
    cfg = get_default_config("nonexistent.json")
    assert cfg is None


def test_get_skeleton_config_returns_dict():
    cfg = get_skeleton_config("ooceraea_biroi.json")
    assert isinstance(cfg, dict)


def test_get_bundled_config_names_returns_list():
    names = get_bundled_config_names()
    assert isinstance(names, list)
    assert "default.json" in names


def test_get_bundled_skeleton_names_returns_list():
    names = get_bundled_skeleton_names()
    assert isinstance(names, list)
    assert "ooceraea_biroi.json" in names


def test_user_config_dir_is_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Clear any cached values by reimporting
    import importlib

    import multi_tracker.paths as paths_mod

    importlib.reload(paths_mod)

    d = paths_mod._user_config_dir()
    assert d.exists()
    assert os.access(d, os.W_OK)


def test_user_data_dir_is_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    import importlib

    import multi_tracker.paths as paths_mod

    importlib.reload(paths_mod)

    d = paths_mod._user_data_dir()
    assert d.exists()
    assert os.access(d, os.W_OK)


def test_get_models_dir_returns_path():
    p = get_models_dir()
    assert isinstance(p, Path)
    assert "models" in str(p)


def test_get_training_runs_dir_returns_path():
    p = get_training_runs_dir()
    assert isinstance(p, Path)
    assert "training" in str(p)


def test_get_advanced_config_path_returns_path():
    p = get_advanced_config_path()
    assert isinstance(p, Path)
    assert p.name == "advanced_config.json"


def test_get_training_workspace_dir_returns_path():
    from multi_tracker.paths import get_training_workspace_dir

    p = get_training_workspace_dir("YOLO")
    assert isinstance(p, Path)
    assert "training" in str(p)


def test_get_presets_dir_seeds_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from multi_tracker.paths import get_presets_dir

    p = get_presets_dir()
    assert (p / ".seeded").exists()
    # Should contain at least default.json
    assert any(f.suffix == ".json" for f in p.iterdir())


def test_get_skeleton_dir_seeds_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from multi_tracker.paths import get_skeleton_dir

    p = get_skeleton_dir()
    assert (p / ".seeded").exists()
    assert any(f.suffix == ".json" for f in p.iterdir())
