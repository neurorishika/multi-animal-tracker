"""Video player widget for MAT-afterhours.

Displays video frames with trajectory overlay (coloured circles + ID labels).
Includes an in-memory frame cache and a horizontal slider for seeking.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional, Set

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

# Colour palette (BGR order for OpenCV drawing).
_PALETTE_BGR = [
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

_MAX_CACHE = 200


class VideoPlayerWidget(QWidget):
    """Video frame display with trajectory overlay and frame slider."""

    frame_changed = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._cap: Optional[cv2.VideoCapture] = None
        self._total_frames: int = 0
        self._current_frame: int = 0
        self._df: Optional[pd.DataFrame] = None
        self._df_by_frame: dict = {}
        self._highlight_ids: Set[int] = set()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        # Debounce timer: only seek after the slider stops moving for 80 ms.
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(80)
        self._seek_timer.timeout.connect(self._do_seek)
        self._pending_frame: int = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Canvas
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(320, 240)
        self._canvas.setStyleSheet("background-color: #000;")
        layout.addWidget(self._canvas, stretch=1)

        # Slider row
        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, stretch=1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(90)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_video(self, path: str) -> None:
        """Open a video file for playback."""
        if self._cap is not None:
            self._cap.release()
        self._cache.clear()

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            self._cap = None
            self._total_frames = 0
            return

        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._slider.setMaximum(max(self._total_frames - 1, 0))
        self.seek_to(0)

    def load_trajectories(self, df: pd.DataFrame) -> None:
        """Store trajectory DataFrame for overlay rendering."""
        self._df = df
        # Pre-group by frame for O(1) overlay lookups.
        self._df_by_frame = (
            {fid: grp for fid, grp in df.groupby("FrameID")} if df is not None else {}
        )

    def seek_to(self, frame: int) -> None:
        """Seek to a specific frame and refresh display."""
        frame = max(0, min(frame, self._total_frames - 1))
        self._current_frame = frame
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._refresh()

    def highlight_tracks(self, track_ids: List[int]) -> None:
        """Draw extra ring around specified track IDs."""
        self._highlight_ids = set(track_ids)
        self._refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._pending_frame = value
        self._seek_timer.start()  # restarts the timer on each move

    def _do_seek(self) -> None:
        self._current_frame = self._pending_frame
        self._refresh()
        self.frame_changed.emit(self._current_frame)

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        """Read frame from cache or video capture."""
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        if self._cap is None:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None

        # Evict oldest if cache full.
        while len(self._cache) >= _MAX_CACHE:
            self._cache.popitem(last=False)
        self._cache[idx] = frame
        return frame

    def _refresh(self) -> None:
        """Read frame, draw overlay, update canvas."""
        self._frame_label.setText(f"{self._current_frame} / {self._total_frames}")

        frame = self._read_frame(self._current_frame)
        if frame is None:
            return

        display = frame.copy()
        self._draw_overlay(display, self._current_frame)
        self._show_frame(display)

    def _draw_overlay(self, img: np.ndarray, frame_idx: int) -> None:
        """Draw trajectory circles and ID labels on *img* (BGR)."""
        if not self._df_by_frame:
            return

        rows = self._df_by_frame.get(frame_idx)
        if rows is None:
            return
        for _, row in rows.iterrows():
            if pd.isna(row["X"]) or pd.isna(row["Y"]):
                continue
            tid = int(row["TrajectoryID"])
            cx = int(round(row["X"]))
            cy = int(round(row["Y"]))

            colour = _PALETTE_BGR[tid % len(_PALETTE_BGR)]
            cv2.circle(img, (cx, cy), 8, colour, 2)

            # ID label
            cv2.putText(
                img,
                str(tid),
                (cx + 10, cy - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                1,
            )

            # Extra highlight ring
            if tid in self._highlight_ids:
                cv2.circle(img, (cx, cy), 14, colour, 2)

    def _show_frame(self, bgr: np.ndarray) -> None:
        """Convert BGR numpy array to QPixmap and display on canvas."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit canvas while keeping aspect ratio.
        scaled = pixmap.scaled(
            self._canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._canvas.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().closeEvent(event)
