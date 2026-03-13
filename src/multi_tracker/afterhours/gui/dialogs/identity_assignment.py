"""Identity assignment dialog for MAT-afterhours.

Shows "before" and "after" thumbnails around the split frame and asks the
user whether the post-split identities should be swapped.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from multi_tracker.afterhours.gui.widgets.interactive_canvas import InteractiveCanvas
from multi_tracker.afterhours.gui.widgets.video_player import draw_overlay

_CROP_MARGIN = 80
_OFFSET_FRAMES = 5


class IdentityAssignmentDialog(QDialog):
    """Ask the user whether to swap post-split identities.

    Shows two side-by-side thumbnails: ``split_frame - 5`` ("Before split")
    and ``split_frame + 5`` ("After split"), cropped around the two tracks.

    Parameters
    ----------
    video_path:
        Path to the video file.
    df:
        Trajectory DataFrame.
    track_a, track_b:
        The two trajectory IDs.
    split_frame:
        The chosen split frame.
    parent:
        Parent widget.
    """

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        track_a: int,
        track_b: int,
        split_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Assign identities")
        self.setMinimumSize(600, 350)

        self._df = df
        self._track_a = track_a
        self._track_b = track_b
        self._split_frame = split_frame

        self._cap = cv2.VideoCapture(video_path)

        layout = QVBoxLayout(self)

        # Instructions
        instr = QLabel(
            f"Review the frames before and after the split at frame "
            f"{split_frame}. Choose whether to swap the identities of "
            f"tracks {track_a} and {track_b} after the split."
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Side-by-side thumbnails
        thumbs_layout = QHBoxLayout()

        # Before group
        before_group = QGroupBox("Before split")
        before_layout = QVBoxLayout(before_group)
        self._before_canvas = InteractiveCanvas()
        self._before_canvas.setMinimumSize(250, 200)
        before_layout.addWidget(self._before_canvas)
        before_frame_label = QLabel(f"Frame {split_frame - _OFFSET_FRAMES}")
        before_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        before_layout.addWidget(before_frame_label)
        thumbs_layout.addWidget(before_group)

        # After group
        after_group = QGroupBox("After split")
        after_layout = QVBoxLayout(after_group)
        self._after_canvas = InteractiveCanvas()
        self._after_canvas.setMinimumSize(250, 200)
        after_layout.addWidget(self._after_canvas)
        after_frame_label = QLabel(f"Frame {split_frame + _OFFSET_FRAMES}")
        after_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        after_layout.addWidget(after_frame_label)
        thumbs_layout.addWidget(after_group)

        layout.addLayout(thumbs_layout)

        # Radio buttons
        self._radio_keep = QRadioButton("Keep order (no swap)")
        self._radio_swap = QRadioButton("Swap identities")
        self._radio_keep.setChecked(True)
        layout.addWidget(self._radio_keep)
        layout.addWidget(self._radio_swap)

        # Buttons
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Render thumbnails
        self._render_thumbnail(split_frame - _OFFSET_FRAMES, self._before_canvas)
        self._render_thumbnail(split_frame + _OFFSET_FRAMES, self._after_canvas)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def should_swap(self) -> bool:
        """Return True if the user chose to swap identities."""
        return self._radio_swap.isChecked()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if self._cap is None or not self._cap.isOpened():
            return None
        idx = max(idx, 0)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        return frame if ret else None

    def _compute_crop(self, frame_idx: int) -> Tuple[int, int, int, int]:
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

    def _render_thumbnail(self, frame_idx: int, canvas: InteractiveCanvas) -> None:
        frame = self._read_frame(frame_idx)
        if frame is None:
            return

        x1, y1, x2, y2 = self._compute_crop(frame_idx)
        h, w = frame.shape[:2]
        x2 = min(x2, w)
        y2 = min(y2, h)
        crop = frame[y1:y2, x1:x2].copy()

        if crop.size == 0:
            return

        # Build a per-frame lookup offset by the crop origin so overlay
        # coordinates line up with the cropped image.
        rows = self._df[
            (self._df["FrameID"] == frame_idx)
            & (self._df["TrajectoryID"].isin([self._track_a, self._track_b]))
        ].copy()
        rows["X"] = rows["X"] - x1
        rows["Y"] = rows["Y"] - y1
        df_by_frame = {frame_idx: rows}
        draw_overlay(crop, df_by_frame, frame_idx, set())

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format.Format_RGB888)
        canvas.set_pixmap(QPixmap.fromImage(qimg))

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
