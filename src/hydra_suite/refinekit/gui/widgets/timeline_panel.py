"""Per-animal timeline panel for RefineKit.

Draws one horizontal bar per trajectory with a label column on the left.
Clicking on a bar emits ``split_requested(track_id, frame)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QLabel, QMenu, QScrollArea, QVBoxLayout, QWidget

# Colour palette (RGB) — same order as video player but in RGB for Qt.
_PALETTE_RGB = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (128, 255, 0),
    (255, 0, 128),
    (0, 255, 128),
]

_LABEL_WIDTH = 60
_DEFAULT_ROW_HEIGHT = 22
_BAR_MARGIN = 2
_MAX_MANUAL_REGION = 300  # max frames selectable for manual review


# ---------------------------------------------------------------------------
# _TimelineCanvas
# ---------------------------------------------------------------------------


class _TimelineCanvas(QWidget):
    """Custom-painted widget showing one horizontal bar per track."""

    split_at = Signal(int, int)  # (track_id, frame)
    region_edit_requested = Signal(int, int)  # (frame_start, frame_end)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tracks: Dict[int, Tuple[int, int]] = {}
        self._track_order: List[int] = []
        self._total_frames: int = 1
        self._row_height: int = _DEFAULT_ROW_HEIGHT

        # Right-click drag selection state
        self._sel_start_x: Optional[int] = None
        self._sel_end_x: Optional[int] = None
        self._is_right_dragging: bool = False

        self.setMouseTracking(True)
        self.setMinimumHeight(50)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_tracks(
        self,
        tracks: Dict[int, Tuple[int, int]],
        total_frames: int,
    ) -> None:
        """Populate the panel with track presence data and set the total frame count for coordinate mapping."""
        self._tracks = dict(tracks)
        self._track_order = sorted(tracks.keys())
        self._total_frames = max(total_frames, 1)
        self._update_size()
        self.update()

    def _update_size(self) -> None:
        self.setMinimumHeight(max(len(self._track_order) * self._row_height + 4, 50))

    def set_highlight_range(self, frame_range: Optional[Tuple[int, int]]) -> None:
        """Update the highlighted frame range and repaint the widget."""
        self.update()

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _bar_area_width(self) -> int:
        return max(self.width() - _LABEL_WIDTH, 1)

    def _frame_to_x(self, frame: int) -> int:
        return _LABEL_WIDTH + int(frame / self._total_frames * self._bar_area_width())

    def _x_to_frame(self, x: int) -> int:
        bar_w = self._bar_area_width()
        frac = max(0.0, min((x - _LABEL_WIDTH) / bar_w, 1.0))
        return int(frac * self._total_frames)

    def _y_to_row(self, y: int) -> int:
        return y // self._row_height

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Emit a split-at signal on left-click over a track bar, or begin a right-drag range selection."""
        pos = event.position().toPoint()

        if event.button() == Qt.MouseButton.LeftButton:
            if pos.x() < _LABEL_WIDTH:
                return super().mousePressEvent(event)
            row = self._y_to_row(pos.y())
            if 0 <= row < len(self._track_order):
                tid = self._track_order[row]
                frame = self._x_to_frame(pos.x())
                self.split_at.emit(tid, frame)

        elif event.button() == Qt.MouseButton.RightButton:
            if pos.x() >= _LABEL_WIDTH:
                self._sel_start_x = pos.x()
                self._sel_end_x = pos.x()
                self._is_right_dragging = True
                self.update()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._is_right_dragging:
            x = max(_LABEL_WIDTH, event.position().toPoint().x())
            self._sel_end_x = x
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.RightButton and self._is_right_dragging:
            self._is_right_dragging = False
            sx1 = min(self._sel_start_x or 0, self._sel_end_x or 0)
            sx2 = max(self._sel_start_x or 0, self._sel_end_x or 0)
            f1 = self._x_to_frame(sx1)
            f2 = self._x_to_frame(sx2)
            self._sel_start_x = None
            self._sel_end_x = None
            self.update()
            if f2 - f1 >= 2:
                self._show_region_menu(event.position().toPoint(), f1, f2)
        super().mouseReleaseEvent(event)

    def _show_region_menu(self, pos: QPoint, f1: int, f2: int) -> None:
        span = f2 - f1
        label = f"Review region  [{f1}\u2013{f2}]  ({span} frames)"
        if span > _MAX_MANUAL_REGION:
            label += f"  \u26a0 will be capped to {_MAX_MANUAL_REGION}"
        menu = QMenu(self)
        act = QAction(label, self)
        act.triggered.connect(lambda: self.region_edit_requested.emit(f1, f2))
        menu.addAction(act)
        menu.exec(self.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Wheel — Ctrl+scroll scales row height
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            step = 2 if delta > 0 else -2
            new_h = max(12, min(80, self._row_height + step))
            if new_h != self._row_height:
                self._row_height = new_h
                self._update_size()
                self.update()
            event.accept()
        else:
            super().wheelEvent(event)


# ---------------------------------------------------------------------------
# TimelinePanelWidget
# ---------------------------------------------------------------------------


class TimelinePanelWidget(QWidget):
    """Per-animal timeline bars in a scrollable container."""

    split_requested = Signal(int, int)  # (track_id, frame)
    region_edit_requested = Signal(int, int)  # (frame_start, frame_end)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._canvas = _TimelineCanvas()
        self._canvas.split_at.connect(self.split_requested)
        self._canvas.region_edit_requested.connect(self.region_edit_requested)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

        hint = QLabel(
            "Right-click drag to select a region for manual review  \u00b7"
            "  Ctrl+scroll to resize rows"
        )
        hint.setStyleSheet("color: #555555; font-size: 10px; padding: 1px 4px;")
        layout.addWidget(hint)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_trajectories(self, df: pd.DataFrame) -> None:
        """Compute per-track frame ranges from a trajectory DataFrame."""
        tracks: Dict[int, Tuple[int, int]] = {}
        total_frames = 0

        for tid, grp in df.groupby("TrajectoryID"):
            fmin = int(grp["FrameID"].min())
            fmax = int(grp["FrameID"].max())
            tracks[int(tid)] = (fmin, fmax)
            total_frames = max(total_frames, fmax + 1)

        self._canvas.set_tracks(tracks, total_frames)

    def highlight_event(self, event) -> None:
        """Highlight a swap suspicion event's frame range."""
        if event is not None and hasattr(event, "frame_range"):
            self._canvas.set_highlight_range(event.frame_range)
        else:
            self._canvas.set_highlight_range(None)
