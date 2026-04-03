# Unified Splash Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a shared `WelcomePage` widget and `RecentItemsStore` in a new `hydra_suite/widgets/` package, then migrate all 6 apps (tracker, posekit, classkit, detectkit, refinekit, filterkit) to use it — producing consistent splash pages with big logos, uniform button styling, and VS Code-style recent items lists.

**Architecture:** A declarative-config approach where each app constructs a `WelcomeConfig` dataclass with its logo, tagline, buttons, and recents settings, then passes it to a shared `WelcomePage` widget. A separate `RecentItemsStore` handles JSON persistence through `hydra_suite.paths._user_data_dir()` (respecting `HYDRA_DATA_DIR`). Each app replaces its `_make_welcome_page()` with ~15 lines that build a config and instantiate `WelcomePage`.

**Tech Stack:** PySide6 (QtWidgets, QtSvg, QtGui, QtCore), hydra_suite.paths, JSON persistence, dataclasses

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/hydra_suite/widgets/__init__.py` | Package init, public exports |
| Create | `src/hydra_suite/widgets/welcome_page.py` | `WelcomePage` widget, `WelcomeConfig`/`ButtonDef` dataclasses |
| Create | `src/hydra_suite/widgets/recents.py` | `RecentItemsStore` JSON persistence |
| Create | `tests/test_welcome_page.py` | Tests for WelcomePage and RecentItemsStore |
| Modify | `src/hydra_suite/tracker/gui/main_window.py:4180-4295` | Replace `_make_welcome_page` |
| Modify | `src/hydra_suite/posekit/ui/main_window.py:758-834` | Replace `_make_welcome_page` |
| Modify | `src/hydra_suite/classkit/gui/mainwindow.py:447-518` | Replace `_make_welcome_page` |
| Modify | `src/hydra_suite/detectkit/ui/main_window.py:106-171` | Replace `_build_welcome_page` |
| Modify | `src/hydra_suite/refinekit/gui/main_window.py:425-494` | Replace `_make_welcome_page` |
| Modify | `src/hydra_suite/filterkit/gui.py:649-725` | Replace `_make_welcome_page` |

---

### Task 1: Create `RecentItemsStore` with tests

**Files:**
- Create: `src/hydra_suite/widgets/__init__.py`
- Create: `src/hydra_suite/widgets/recents.py`
- Create: `tests/test_welcome_page.py`

- [ ] **Step 1: Create the widgets package init**

```python
# src/hydra_suite/widgets/__init__.py
"""Shared GUI widgets for hydra-suite applications."""
```

- [ ] **Step 2: Write failing tests for RecentItemsStore**

```python
# tests/test_welcome_page.py
"""Tests for the shared welcome-page widgets and recents store."""

from __future__ import annotations

import json
from pathlib import Path

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_welcome_page.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.widgets'`

- [ ] **Step 4: Implement RecentItemsStore**

```python
# src/hydra_suite/widgets/recents.py
"""Persistent recent-items store for hydra-suite applications."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_RECENT = 20


class RecentItemsStore:
    """JSON-backed recent-items list for a specific app.

    Storage location respects ``HYDRA_DATA_DIR`` via ``hydra_suite.paths``.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def _json_path(self) -> Path:
        from hydra_suite.paths import _user_data_dir

        return _user_data_dir() / self._app_name / "recents.json"

    def load(self) -> list[str]:
        """Return recent items, most-recent first."""
        p = self._json_path()
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            logger.debug("Failed to read recents for %s", self._app_name, exc_info=True)
        return []

    def add(self, path: str) -> None:
        """Add *path* to the top, de-duplicating and trimming to max."""
        items = self.load()
        items = [x for x in items if x != path]
        items.insert(0, path)
        self._save(items)

    def remove(self, path: str) -> None:
        """Remove *path* from the list."""
        items = self.load()
        items = [x for x in items if x != path]
        self._save(items)

    def clear(self) -> None:
        """Remove all recent items."""
        self._save([])

    def _save(self, items: list[str]) -> None:
        p = self._json_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items[:_MAX_RECENT], indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_welcome_page.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/widgets/__init__.py src/hydra_suite/widgets/recents.py tests/test_welcome_page.py
git commit -m "feat: add RecentItemsStore for shared recent-items persistence"
```

---

### Task 2: Create `WelcomePage` widget

**Files:**
- Modify: `src/hydra_suite/widgets/__init__.py`
- Create: `src/hydra_suite/widgets/welcome_page.py`
- Modify: `tests/test_welcome_page.py`

- [ ] **Step 1: Write failing tests for WelcomePage**

Append to `tests/test_welcome_page.py`:

```python
import sys

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
        from hydra_suite.widgets.welcome_page import ButtonDef, WelcomeConfig, WelcomePage

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

        from hydra_suite.widgets.welcome_page import ButtonDef, WelcomeConfig, WelcomePage

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
        from hydra_suite.widgets.welcome_page import ButtonDef, WelcomeConfig, WelcomePage

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
        # The page should contain text from the recent items
        all_text = page.grab()  # Just verify it renders without error
        assert all_text is not None
        page.close()

    def test_refresh_recents(self, qapp, store):
        from hydra_suite.widgets.welcome_page import ButtonDef, WelcomeConfig, WelcomePage

        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Test",
            buttons=[ButtonDef(label="Open", callback=lambda: None)],
            recents_label="Recent",
            recents_store=store,
            on_recent_clicked=lambda p: None,
        )
        page = WelcomePage(config)
        # Add an item after construction, then refresh
        store.add("/new/path")
        page.refresh_recents()
        page.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_welcome_page.py::TestWelcomePage -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydra_suite.widgets.welcome_page'`

- [ ] **Step 3: Implement WelcomePage widget**

```python
# src/hydra_suite/widgets/welcome_page.py
"""Shared welcome/splash page widget for hydra-suite applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.paths import get_brand_icon_bytes
from hydra_suite.widgets.recents import RecentItemsStore

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ButtonDef:
    """Definition for a single action button on the welcome page."""

    label: str
    callback: Callable
    tooltip: str = ""


@dataclass
class WelcomeConfig:
    """Declarative configuration for a WelcomePage instance."""

    logo_svg: str
    tagline: str
    buttons: List[ButtonDef]
    recents_label: str
    recents_store: RecentItemsStore
    on_recent_clicked: Callable[[str], None]
    logo_max_height: int = 450


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

_BG = "#121212"
_TAGLINE_COLOR = "#555555"
_BTN_STYLE = """
    QPushButton {
        background-color: #1e1e1e;
        color: #cccccc;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 10px 24px;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #2a2a2a;
        border-color: #4d9eff;
        color: #ffffff;
    }
    QPushButton:pressed {
        background-color: #333333;
    }
"""
_RECENT_NAME_COLOR = "#4d9eff"
_RECENT_PATH_COLOR = "#888888"
_RECENT_HOVER_BG = "#1e1e1e"
_SECTION_LABEL_COLOR = "#555555"


# ---------------------------------------------------------------------------
# Helper: middle-ellipsis for paths
# ---------------------------------------------------------------------------


def _middle_ellipsis(text: str, max_len: int = 50) -> str:
    """Shorten *text* with an ellipsis in the middle if it exceeds *max_len*."""
    if len(text) <= max_len:
        return text
    keep = (max_len - 3) // 2
    return text[:keep] + "..." + text[len(text) - keep:]


# ---------------------------------------------------------------------------
# Recent-item row widget
# ---------------------------------------------------------------------------


class _RecentItemRow(QWidget):
    """Single row in the recents list: clickable name + dimmed path."""

    def __init__(
        self, display_name: str, full_path: str, on_click: Callable[[str], None], parent=None
    ):
        super().__init__(parent)
        self._full_path = full_path
        self._on_click = on_click
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet(
            f"_RecentItemRow {{ background: transparent; border-radius: 4px; }}"
            f"_RecentItemRow:hover {{ background: {_RECENT_HOVER_BG}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        name_label = QLabel(display_name)
        name_label.setStyleSheet(
            f"color: {_RECENT_NAME_COLOR}; font-size: 14px; font-weight: bold;"
            " background: transparent;"
        )
        name_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        path_display = _middle_ellipsis(full_path, 55)
        path_label = QLabel(path_display)
        path_label.setStyleSheet(
            f"color: {_RECENT_PATH_COLOR}; font-size: 13px; background: transparent;"
        )
        path_label.setToolTip(full_path)

        layout.addWidget(name_label)
        layout.addWidget(path_label, stretch=1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self._full_path)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# WelcomePage widget
# ---------------------------------------------------------------------------


class WelcomePage(QWidget):
    """Shared welcome/splash page for all hydra-suite applications."""

    def __init__(self, config: WelcomeConfig, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self.setStyleSheet(f"background-color: {_BG};")

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.setSpacing(0)
        root.addStretch(1)

        # --- Logo ---
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._render_logo(logo_label, config.logo_svg, config.logo_max_height)
        root.addWidget(logo_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # --- Tagline ---
        tagline = QLabel(config.tagline)
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tagline.setStyleSheet(
            f"color: {_TAGLINE_COLOR}; font-size: 14px; letter-spacing: 2px;"
            " margin-top: 12px; background: transparent;"
        )
        root.addWidget(tagline, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(32)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        btn_row.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_row.setSpacing(16)

        for bdef in config.buttons:
            btn = QPushButton(bdef.label)
            btn.setStyleSheet(_BTN_STYLE)
            if bdef.tooltip:
                btn.setToolTip(bdef.tooltip)
            btn.clicked.connect(bdef.callback)
            btn_row.addWidget(btn)

        btn_container = QWidget()
        btn_container.setStyleSheet("background: transparent;")
        btn_container.setLayout(btn_row)
        root.addWidget(btn_container, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.addSpacing(36)

        # --- Recents section ---
        self._recents_container = QVBoxLayout()
        self._recents_container.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._recents_container.setSpacing(0)

        # Section label
        self._recents_header = QLabel(config.recents_label)
        self._recents_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._recents_header.setStyleSheet(
            f"color: {_SECTION_LABEL_COLOR}; font-size: 12px;"
            " letter-spacing: 1px; margin-bottom: 8px; background: transparent;"
        )
        self._recents_container.addWidget(
            self._recents_header, alignment=Qt.AlignmentFlag.AlignHCenter
        )

        # Scroll area for recent items (frameless, transparent)
        self._recents_scroll = QScrollArea()
        self._recents_scroll.setWidgetResizable(True)
        self._recents_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._recents_scroll.setStyleSheet("background: transparent;")
        self._recents_scroll.setFixedWidth(600)
        self._recents_scroll.setMaximumHeight(300)

        self._recents_list_widget = QWidget()
        self._recents_list_widget.setStyleSheet("background: transparent;")
        self._recents_list_layout = QVBoxLayout(self._recents_list_widget)
        self._recents_list_layout.setContentsMargins(0, 0, 0, 0)
        self._recents_list_layout.setSpacing(2)
        self._recents_scroll.setWidget(self._recents_list_widget)

        self._recents_container.addWidget(
            self._recents_scroll, alignment=Qt.AlignmentFlag.AlignHCenter
        )

        recents_wrapper = QWidget()
        recents_wrapper.setStyleSheet("background: transparent;")
        recents_wrapper.setLayout(self._recents_container)
        root.addWidget(recents_wrapper, alignment=Qt.AlignmentFlag.AlignHCenter)

        root.addStretch(1)

        # Populate recents
        self.refresh_recents()

    def refresh_recents(self) -> None:
        """Reload and redisplay the recent items list."""
        # Clear existing rows
        while self._recents_list_layout.count():
            item = self._recents_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        items = self._config.recents_store.load()
        if not items:
            self._recents_header.hide()
            self._recents_scroll.hide()
            return

        self._recents_header.show()
        self._recents_scroll.show()

        for path_str in items[:10]:
            p = Path(path_str)
            display_name = p.name
            row = _RecentItemRow(
                display_name, path_str, self._config.on_recent_clicked, self
            )
            self._recents_list_layout.addWidget(row)

        self._recents_list_layout.addStretch()

    @staticmethod
    def _render_logo(label: QLabel, svg_name: str, max_height: int) -> None:
        """Render an SVG brand icon onto *label* at up to *max_height* px."""
        logo_data = get_brand_icon_bytes(svg_name)
        if logo_data is None:
            return
        renderer = QSvgRenderer(QByteArray(logo_data))
        if not renderer.isValid():
            return
        vb = renderer.viewBoxF()
        if vb.isEmpty():
            ds = renderer.defaultSize()
            vb = QRectF(0, 0, max(1, ds.width()), max(1, ds.height()))

        # Scale to fit max_height while preserving aspect ratio
        scale = max_height / max(vb.height(), 1)
        draw_w = max(1, int(vb.width() * scale))
        draw_h = max(1, int(vb.height() * scale))

        canvas = QPixmap(draw_w, draw_h)
        canvas.fill(QColor(0, 0, 0, 0))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        renderer.render(painter, QRectF(0, 0, draw_w, draw_h))
        painter.end()
        label.setPixmap(canvas)
```

- [ ] **Step 4: Update widgets __init__.py with public exports**

```python
# src/hydra_suite/widgets/__init__.py
"""Shared GUI widgets for hydra-suite applications."""

from hydra_suite.widgets.recents import RecentItemsStore
from hydra_suite.widgets.welcome_page import ButtonDef, WelcomeConfig, WelcomePage

__all__ = [
    "ButtonDef",
    "RecentItemsStore",
    "WelcomeConfig",
    "WelcomePage",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_welcome_page.py -v`
Expected: All tests PASS (store tests + widget tests)

- [ ] **Step 6: Commit**

```bash
git add src/hydra_suite/widgets/ tests/test_welcome_page.py
git commit -m "feat: add shared WelcomePage widget with VS Code-style recents"
```

---

### Task 3: Migrate Tracker splash page

**Files:**
- Modify: `src/hydra_suite/tracker/gui/main_window.py:4180-4295`

- [ ] **Step 1: Replace `_make_welcome_page` in tracker MainWindow**

Replace lines 4180–4295 of `src/hydra_suite/tracker/gui/main_window.py` (the entire `_make_welcome_page` method) with:

```python
    def _make_welcome_page(self):
        """Create startup splash page with primary HYDRA session actions."""
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("tracker")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="hydra.svg",
            tagline="Track  |  Analyze  |  Refine",
            buttons=[
                ButtonDef(label="Load Video\u2026", callback=self.select_file),
                ButtonDef(label="Load Video List\u2026", callback=self._import_batch_list),
                ButtonDef(label="Load Config\u2026", callback=self.load_config),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=self._open_recent_video,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page
```

- [ ] **Step 2: Add `_open_recent_video` helper and recents tracking**

Add after `_make_welcome_page` (before `_show_workspace`):

```python
    def _open_recent_video(self, path: str):
        """Open a video file from the recent items list."""
        from pathlib import Path

        video_path = Path(path)
        if video_path.exists():
            self._load_video_path(str(video_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "File Not Found", f"Video not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

Also, find where the tracker successfully loads a video (the method that transitions to workspace — `_show_workspace` or wherever the video path is confirmed) and add a call to record the recent:

```python
        if hasattr(self, "_recents_store"):
            self._recents_store.add(str(video_path))
```

The exact insertion point: search for where `self.main_stack.setCurrentIndex(self._workspace_page_index)` is called after a video loads, and add the recents tracking just before it.

- [ ] **Step 3: Remove old `_splash_buttons` references**

The old code stored `self._splash_buttons`. Search for any references to `_splash_buttons` in the file and remove them if they exist only for the old welcome page.

- [ ] **Step 4: Verify tracker launches**

Run: `python -c "from hydra_suite.widgets import WelcomePage; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/hydra_suite/tracker/gui/main_window.py
git commit -m "refactor: migrate tracker splash page to shared WelcomePage"
```

---

### Task 4: Migrate PoseKit splash page

**Files:**
- Modify: `src/hydra_suite/posekit/ui/main_window.py:758-834`

- [ ] **Step 1: Replace `_make_welcome_page` in PoseKit MainWindow**

Replace lines 758–834 with:

```python
    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown when PoseKit starts without a loaded project."""
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("posekit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="posekit.svg",
            tagline="Pose Labeling Workspace",
            buttons=[
                ButtonDef(label="New Project\u2026", callback=self.new_project_wizard),
                ButtonDef(label="Open Existing Project\u2026", callback=self.open_project),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page
```

- [ ] **Step 2: Add `_open_recent_project` helper and recents tracking**

Add after the method:

```python
    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        project_path = Path(path)
        if project_path.exists():
            self._load_project_dir(str(project_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

Also add `self._recents_store.add(path)` at the point where a project is successfully opened (search for where `self._content_stack.setCurrentIndex(1)` is triggered after project load).

- [ ] **Step 3: Verify PoseKit import works**

Run: `python -c "from hydra_suite.posekit.ui.main_window import PoseKitWindow; print('OK')"`

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/posekit/ui/main_window.py
git commit -m "refactor: migrate posekit splash page to shared WelcomePage"
```

---

### Task 5: Migrate ClassKit splash page

**Files:**
- Modify: `src/hydra_suite/classkit/gui/mainwindow.py:447-518`

- [ ] **Step 1: Replace `_make_welcome_page` in ClassKit MainWindow**

Replace lines 447–518 with:

```python
    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown before any project is opened."""
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("classkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="classkit.svg",
            tagline="Active Learning Dataset Builder",
            buttons=[
                ButtonDef(label="New Project\u2026", callback=self.new_project),
                ButtonDef(label="Open Project\u2026", callback=self.open_project),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page
```

- [ ] **Step 2: Add `_open_recent_project` helper and recents tracking**

Add after the method:

```python
    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        project_path = Path(path)
        if project_path.exists():
            self._load_project(str(project_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

Add `self._recents_store.add(path)` where ClassKit transitions to workspace after a successful project load (near `self._stacked.setCurrentIndex(1)`).

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/classkit/gui/mainwindow.py
git commit -m "refactor: migrate classkit splash page to shared WelcomePage"
```

---

### Task 6: Migrate DetectKit splash page

**Files:**
- Modify: `src/hydra_suite/detectkit/ui/main_window.py:106-171`

- [ ] **Step 1: Replace `_build_welcome_page` in DetectKit MainWindow**

Replace lines 106–171 with:

```python
    def _build_welcome_page(self) -> None:
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("detectkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="detectkit.svg",
            tagline="OBB Detection Model Training & Dataset Curation",
            buttons=[
                ButtonDef(label="New Project", callback=self.new_project),
                ButtonDef(label="Open Project", callback=self.open_project_dialog),
            ],
            recents_label="Recent Projects",
            recents_store=store,
            on_recent_clicked=self._open_recent_project,
        )
        self._welcome_page = WelcomePage(config)
        self._stack.addWidget(self._welcome_page)  # index 0
```

- [ ] **Step 2: Add `_open_recent_project` helper**

Add after the method:

```python
    def _open_recent_project(self, path: str):
        """Open a project from the recent items list."""
        from pathlib import Path

        project_dir = Path(path)
        if project_dir.exists():
            self._open_project(project_dir)
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Project not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

- [ ] **Step 3: Update recents tracking to use shared store**

DetectKit currently calls `add_to_recent()` from `detectkit/ui/project.py`. Update those call sites to use `self._recents_store.add(path)` instead. Remove or deprecate the old `_refresh_welcome_recent` method and `self._welcome_recent_list` references.

Search for `add_to_recent` and `_refresh_welcome_recent` in the detectkit main_window and replace:
- `add_to_recent(str(project_dir))` → `self._recents_store.add(str(project_dir))`
- `self._refresh_welcome_recent()` → `self._welcome_page.refresh_recents()`
- Remove the old `_refresh_welcome_recent` method
- Remove `self._welcome_recent_list` references

- [ ] **Step 4: Commit**

```bash
git add src/hydra_suite/detectkit/ui/main_window.py
git commit -m "refactor: migrate detectkit splash page to shared WelcomePage"
```

---

### Task 7: Migrate RefineKit splash page

**Files:**
- Modify: `src/hydra_suite/refinekit/gui/main_window.py:425-494`

- [ ] **Step 1: Replace `_make_welcome_page` in RefineKit MainWindow**

Replace lines 425–494 with:

```python
    def _make_welcome_page(self) -> QWidget:
        """Logo/welcome screen shown before any session is loaded."""
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("refinekit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="refinekit.svg",
            tagline="Review  \u00b7  Correct  \u00b7  Verify",
            buttons=[
                ButtonDef(
                    label="Load Video\u2026",
                    callback=self._load_single_video,
                    tooltip="Open a single video file for review",
                ),
                ButtonDef(
                    label="Load Video List\u2026",
                    callback=self._load_video_list,
                    tooltip="Open a .txt file listing one video path per line",
                ),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Videos",
            recents_store=store,
            on_recent_clicked=self._open_recent_video,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page
```

- [ ] **Step 2: Add `_open_recent_video` helper and recents tracking**

Add after the method:

```python
    def _open_recent_video(self, path: str):
        """Open a video from the recent items list."""
        from pathlib import Path

        video_path = Path(path)
        if video_path.exists():
            self._load_video_path(str(video_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Video not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

Add `self._recents_store.add(path)` where a video is successfully loaded (near `self._content_stack.setCurrentIndex(1)`).

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/refinekit/gui/main_window.py
git commit -m "refactor: migrate refinekit splash page to shared WelcomePage"
```

---

### Task 8: Migrate FilterKit splash page

**Files:**
- Modify: `src/hydra_suite/filterkit/gui.py:649-725`

- [ ] **Step 1: Replace `_make_welcome_page` in FilterKit**

Replace lines 649–725 with:

```python
    def _make_welcome_page(self):
        """Logo/welcome screen shown before a dataset is loaded."""
        from hydra_suite.widgets import ButtonDef, RecentItemsStore, WelcomeConfig, WelcomePage

        store = RecentItemsStore("filterkit")
        self._recents_store = store

        config = WelcomeConfig(
            logo_svg="filterkit.svg",
            tagline="Dataset Subsampling Workspace",
            buttons=[
                ButtonDef(label="Load Dataset Folder\u2026", callback=self.load_dataset_dialog),
                ButtonDef(label="Quit", callback=self.close),
            ],
            recents_label="Recent Datasets",
            recents_store=store,
            on_recent_clicked=self._open_recent_dataset,
        )
        self._welcome_page = WelcomePage(config)
        return self._welcome_page
```

- [ ] **Step 2: Add `_open_recent_dataset` helper and recents tracking**

Add after the method:

```python
    def _open_recent_dataset(self, path: str):
        """Open a dataset folder from the recent items list."""
        from pathlib import Path

        dataset_path = Path(path)
        if dataset_path.exists():
            self._load_dataset(str(dataset_path))
        else:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Not Found", f"Dataset not found:\n{path}")
            if hasattr(self, "_recents_store"):
                self._recents_store.remove(path)
                if hasattr(self, "_welcome_page"):
                    self._welcome_page.refresh_recents()
```

Add `self._recents_store.add(path)` where a dataset folder is successfully loaded (near `self._content_stack.setCurrentIndex(1)`).

- [ ] **Step 3: Commit**

```bash
git add src/hydra_suite/filterkit/gui.py
git commit -m "refactor: migrate filterkit splash page to shared WelcomePage"
```

---

### Task 9: Remove dead imports and run quality checks

**Files:**
- All modified app files

- [ ] **Step 1: Clean up unused imports**

In each migrated file, the old `_make_welcome_page` methods imported `QSvgRenderer`, `QRectF`, `QColor`, `QPainter`, `QPixmap`, `get_brand_icon_bytes` directly. Check each file for imports that are now only used by the removed code and delete them. Common candidates:

- `from PySide6.QtSvg import QSvgRenderer` (if not used elsewhere)
- `from PySide6.QtCore import QByteArray` (if only used in welcome page)

Do NOT remove imports that are still used by other parts of the file.

- [ ] **Step 2: Run formatter**

```bash
make format
```

- [ ] **Step 3: Run linter**

```bash
make lint
```

Fix any issues reported.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_welcome_page.py -v
```

Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "chore: clean up dead imports from splash page migration"
```
