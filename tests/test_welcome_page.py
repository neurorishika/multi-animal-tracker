"""Tests for the shared welcome-page widgets and recents store."""

from __future__ import annotations

import json

import pytest

from hydra_suite.widgets.recents import RecentItemsStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Create a RecentItemsStore backed by a temp directory."""
    monkeypatch.setenv("HYDRA_DATA_DIR", str(tmp_path))
    return RecentItemsStore("testapp")


class TestRecentItemsStore:
    def test_load_empty(self, store):
        assert store.load() == []

    def test_add_and_load(self, store):
        store.add("/path/to/project1")
        store.add("/path/to/project2")
        result = store.load()
        assert result == ["/path/to/project2", "/path/to/project1"]

    def test_deduplication(self, store):
        store.add("/path/a")
        store.add("/path/b")
        store.add("/path/a")
        result = store.load()
        assert result == ["/path/a", "/path/b"]

    def test_max_items(self, store):
        for i in range(25):
            store.add(f"/path/{i}")
        result = store.load()
        assert len(result) == 20
        assert result[0] == "/path/24"

    def test_remove(self, store):
        store.add("/path/a")
        store.add("/path/b")
        store.remove("/path/a")
        assert store.load() == ["/path/b"]

    def test_clear(self, store):
        store.add("/path/a")
        store.clear()
        assert store.load() == []

    def test_respects_hydra_data_dir(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom"
        monkeypatch.setenv("HYDRA_DATA_DIR", str(custom))
        s = RecentItemsStore("myapp")
        s.add("/some/path")
        expected_file = custom / "myapp" / "recents.json"
        assert expected_file.exists()
        data = json.loads(expected_file.read_text())
        assert data == ["/some/path"]

    def test_corrupted_file_returns_empty(self, store):
        store.add("/path/a")
        # Corrupt the file
        store._json_path().write_text("NOT JSON", encoding="utf-8")
        assert store.load() == []
