"""Shared welcome/splash page widget for hydra-suite applications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPixmap
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
    return text[:keep] + "..." + text[len(text) - keep :]


# ---------------------------------------------------------------------------
# Recent-item row widget
# ---------------------------------------------------------------------------


class _RecentItemRow(QWidget):
    """Single row in the recents list: clickable name + dimmed path."""

    def __init__(
        self,
        display_name: str,
        full_path: str,
        on_click: Callable[[str], None],
        parent=None,
    ):
        super().__init__(parent)
        self._full_path = full_path
        self._on_click = on_click
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
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
        self.setObjectName("welcomePage")
        self.setStyleSheet(f"#welcomePage {{ background-color: {_BG}; }}")

        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.setSpacing(0)
        root.addStretch(1)

        # --- Logo ---
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_label.setStyleSheet("background: transparent; border: none;")
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
