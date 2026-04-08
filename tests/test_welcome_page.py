"""Tests for the shared welcome-page widgets and recents store."""

from __future__ import annotations

import json
import sys

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

    def test_ignores_transient_pytest_temp_paths(self, store, tmp_path):
        transient = str(tmp_path / "test_load_project_data_clears_0")
        store.add(transient)
        store.add("/Users/example/projects/real-project")

        assert store.load() == ["/Users/example/projects/real-project"]

    def test_load_scrubs_existing_transient_pytest_entries(self, store, tmp_path):
        transient = str(tmp_path / "test_open_project_0")
        payload = [transient, "/Users/example/projects/real-project"]
        store._json_path().parent.mkdir(parents=True, exist_ok=True)
        store._json_path().write_text(json.dumps(payload), encoding="utf-8")

        assert store.load() == ["/Users/example/projects/real-project"]
        saved = json.loads(store._json_path().read_text(encoding="utf-8"))
        assert saved == ["/Users/example/projects/real-project"]


# Guard Qt tests — skip if display not available
pytest.importorskip("PySide6")


@pytest.fixture()
def qapp():
    """Provide a QApplication for widget tests."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


class TestWelcomePage:
    def test_creates_with_minimal_config(self, qapp, store):
        from hydra_suite.widgets.welcome_page import (
            ButtonDef,
            WelcomeConfig,
            WelcomePage,
        )

        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Test Tagline",
            buttons=[ButtonDef(label="Test", callback=lambda: None)],
            recents_label="Recent Items",
            recents_store=store,
            on_recent_clicked=lambda p: None,
        )
        page = WelcomePage(config)
        assert page is not None
        page.close()

    def test_buttons_rendered(self, qapp, store):
        from PySide6.QtWidgets import QPushButton

        from hydra_suite.widgets.welcome_page import (
            ButtonDef,
            WelcomeConfig,
            WelcomePage,
        )

        clicked = []
        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Test",
            buttons=[
                ButtonDef(label="Alpha", callback=lambda: clicked.append("a")),
                ButtonDef(label="Beta", callback=lambda: clicked.append("b")),
                ButtonDef(label="Gamma", callback=lambda: clicked.append("c")),
            ],
            recents_label="Recent",
            recents_store=store,
            on_recent_clicked=lambda p: None,
        )
        page = WelcomePage(config)
        btns = page.findChildren(QPushButton)
        labels = [b.text() for b in btns]
        assert "Alpha" in labels
        assert "Beta" in labels
        assert "Gamma" in labels
        page.close()

    def test_recents_displayed(self, qapp, store):
        from hydra_suite.widgets.welcome_page import (
            ButtonDef,
            WelcomeConfig,
            WelcomePage,
        )

        store.add("/data/videos/experiment1.mp4")
        store.add("/data/videos/experiment2.mp4")

        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Test",
            buttons=[ButtonDef(label="Open", callback=lambda: None)],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=lambda p: None,
        )
        page = WelcomePage(config)
        all_text = page.grab()  # Just verify it renders without error
        assert all_text is not None
        page.close()

    def test_refresh_recents(self, qapp, store):
        from hydra_suite.widgets.welcome_page import (
            ButtonDef,
            WelcomeConfig,
            WelcomePage,
        )

        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Test",
            buttons=[ButtonDef(label="Open", callback=lambda: None)],
            recents_label="Recent",
            recents_store=store,
            on_recent_clicked=lambda p: None,
        )
        page = WelcomePage(config)
        store.add("/new/path")
        page.refresh_recents()
        page.close()
