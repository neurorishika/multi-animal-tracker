"""Per-animal timeline panel for MAT-afterhours.

Draws one horizontal bar per trajectory with a label column on the left.
Clicking on a bar emits ``split_requested(track_id, frame)``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

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
_ROW_HEIGHT = 22
_BAR_MARGIN = 2


# ---------------------------------------------------------------------------
# _TimelineCanvas
# ---------------------------------------------------------------------------


class _TimelineCanvas(QWidget):
    """Custom-painted widget showing one horizontal bar per track."""

    split_at = Signal(int, int)  # (track_id, frame)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        # track_id -> (min_frame, max_frame)
        self._tracks: Dict[int, Tuple[int, int]] = {}
        self._track_order: List[int] = []
        self._total_frames: int = 1
        self._highlight_range: Optional[Tuple[int, int]] = None

        self.setMinimumHeight(50)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_tracks(
        self,
        tracks: Dict[int, Tuple[int, int]],
        total_frames: int,
    ) -> None:
        """Set per-track frame ranges and total frame count."""
        self._tracks = dict(tracks)
        self._track_order = sorted(tracks.keys())
        self._total_frames = max(total_frames, 1)
        self.setMinimumHeight(max(len(self._track_order) * _ROW_HEIGHT + 4, 50))
        self.update()

    def set_highlight_range(self, frame_range: Optional[Tuple[int, int]]) -> None:
        """Store a highlight range for rendering (future enhancement)."""
        self._highlight_range = frame_range
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
        return y // _ROW_HEIGHT

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        for row, tid in enumerate(self._track_order):
            y_top = row * _ROW_HEIGHT + _BAR_MARGIN
            bar_h = _ROW_HEIGHT - 2 * _BAR_MARGIN

            # Label
            painter.setPen(QPen(Qt.GlobalColor.black))
            label_rect = QRect(0, y_top, _LABEL_WIDTH - 4, bar_h)
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(tid),
            )

            # Bar
            fmin, fmax = self._tracks[tid]
            x1 = self._frame_to_x(fmin)
            x2 = self._frame_to_x(fmax)
            r, g, b = _PALETTE_RGB[tid % len(_PALETTE_RGB)]
            painter.fillRect(
                QRect(x1, y_top, max(x2 - x1, 2), bar_h),
                QColor(r, g, b, 180),
            )

        # Draw highlight range if set
        if self._highlight_range is not None:
            hl_x1 = self._frame_to_x(self._highlight_range[0])
            hl_x2 = self._frame_to_x(self._highlight_range[1])
            painter.setPen(QPen(QColor(255, 165, 0, 200), 2))
            painter.drawRect(
                QRect(
                    hl_x1,
                    0,
                    max(hl_x2 - hl_x1, 2),
                    len(self._track_order) * _ROW_HEIGHT,
                )
            )

        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        pos = event.position()
        x = int(pos.x())
        y = int(pos.y())

        if x < _LABEL_WIDTH:
            return super().mousePressEvent(event)

        row = self._y_to_row(y)
        if 0 <= row < len(self._track_order):
            tid = self._track_order[row]
            frame = self._x_to_frame(x)
            self.split_at.emit(tid, frame)

        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# TimelinePanelWidget
# ---------------------------------------------------------------------------


class TimelinePanelWidget(QWidget):
    """Per-animal timeline bars in a scrollable container."""

    split_requested = Signal(int, int)  # (track_id, frame)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._canvas = _TimelineCanvas()
        self._canvas.split_at.connect(self.split_requested)
        self._scroll.setWidget(self._canvas)
        layout.addWidget(self._scroll)

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
