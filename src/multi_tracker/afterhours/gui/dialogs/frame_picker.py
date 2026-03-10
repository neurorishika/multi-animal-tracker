"""Frame picker dialog for MAT-afterhours.

Lets the user choose a split frame within a swap suspicion event's frame
range by scrubbing a slider over cropped video thumbnails.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
)

_CROP_MARGIN = 80


class FramePickerDialog(QDialog):
    """Dialog for selecting the split frame within an event's range.

    Shows a cropped view around the two tracks involved in the event.
    The slider covers the event's ``frame_range``.

    Parameters
    ----------
    video_path:
        Path to the video file.
    df:
        Trajectory DataFrame (must contain ``TrajectoryID``, ``FrameID``,
        ``X``, ``Y``).
    track_a, track_b:
        The two trajectory IDs involved.
    frame_range:
        Inclusive ``(start, end)`` frame range to browse.
    parent:
        Parent widget.
    """

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        track_a: int,
        track_b: int,
        frame_range: Tuple[int, int],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pick split frame")
        self.setMinimumSize(500, 400)

        self._df = df
        self._track_a = track_a
        self._track_b = track_b
        self._frame_range = frame_range
        self._selected: int = (frame_range[0] + frame_range[1]) // 2
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

        self._cap = cv2.VideoCapture(video_path)

        layout = QVBoxLayout(self)

        # Instructions
        instr = QLabel(
            "Scrub the slider to find the frame where the identity swap "
            "most likely occurred, then click OK."
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Canvas
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(400, 300)
        self._canvas.setStyleSheet("background-color: #000;")
        layout.addWidget(self._canvas, stretch=1)

        # Slider row
        slider_row = QHBoxLayout()
        self._frame_label = QLabel(str(self._selected))
        self._frame_label.setMinimumWidth(60)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(frame_range[0])
        self._slider.setMaximum(frame_range[1])
        self._slider.setValue(self._selected)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, stretch=1)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Show initial frame
        self._refresh()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def selected_frame(self) -> int:
        """Return the frame index chosen by the user."""
        return self._selected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._selected = value
        self._frame_label.setText(str(value))
        self._refresh()

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        if self._cap is None or not self._cap.isOpened():
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return None
        while len(self._cache) >= 100:
            self._cache.popitem(last=False)
        self._cache[idx] = frame
        return frame

    def _compute_crop(self, frame_idx: int) -> Tuple[int, int, int, int]:
        """Bounding box of both tracks at *frame_idx* plus margin."""
        rows = self._df[
            (self._df["FrameID"] == frame_idx)
            & (self._df["TrajectoryID"].isin([self._track_a, self._track_b]))
        ]
        if rows.empty:
            return 0, 0, 640, 480

        x_vals = rows["X"].values
        y_vals = rows["Y"].values
        x1 = int(x_vals.min()) - _CROP_MARGIN
        y1 = int(y_vals.min()) - _CROP_MARGIN
        x2 = int(x_vals.max()) + _CROP_MARGIN
        y2 = int(y_vals.max()) + _CROP_MARGIN
        return max(x1, 0), max(y1, 0), x2, y2

    def _refresh(self) -> None:
        frame = self._read_frame(self._selected)
        if frame is None:
            return

        x1, y1, x2, y2 = self._compute_crop(self._selected)
        h, w = frame.shape[:2]
        x2 = min(x2, w)
        y2 = min(y2, h)
        crop = frame[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
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

    def reject(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().reject()

    def accept(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().accept()
