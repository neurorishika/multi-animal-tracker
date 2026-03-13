"""Frame picker dialog for MAT-afterhours.

Lets the user choose a split frame within a swap suspicion event's frame
range. All frames in the range are pre-decoded in a background thread so
the slider seeks instantly. Trajectory overlays for the two tracks under
review are baked in at load time so the user can clearly see both animals.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSlider,
    QVBoxLayout,
)

from multi_tracker.afterhours.gui.widgets.interactive_canvas import InteractiveCanvas

_CROP_MARGIN = 80
_CONTEXT_FRAMES = 15  # extra frames loaded before / after the event window


# ---------------------------------------------------------------------------
# Background frame loader
# ---------------------------------------------------------------------------


class _FrameRangeLoader(QThread):
    """Decodes every frame in a range, draws track overlays, stores cropped results.

    The crop region is computed once from the union of both tracks' positions
    across the entire range so the view is stable while scrubbing.
    """

    progress = Signal(int, int)  # (loaded_so_far, total)
    finished = Signal()

    _COLOR_A = (0, 0, 220)  # red (BGR) for track_a
    _COLOR_B = (220, 0, 0)  # blue (BGR) for track_b

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        track_a: int,
        track_b: int,
        crop_box: Tuple[int, int, int, int],
        frame_start: int,
        frame_end: int,
        parent=None,
    ):
        super().__init__(parent)
        self._path = video_path
        self._df_by_frame = {fid: grp for fid, grp in df.groupby("FrameID")}
        self._track_a = track_a
        self._track_b = track_b
        self._crop_box = crop_box
        self._start = frame_start
        self._end = frame_end
        # Public result — populated during run(); safe to read after finished().
        self.crops: dict = {}

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self.finished.emit()
            return
        x1, y1, x2, y2 = self._crop_box
        total = self._end - self._start + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        for i in range(total):
            if self.isInterruptionRequested():
                break
            ret, frame = cap.read()
            if not ret:
                break
            idx = self._start + i
            h, w = frame.shape[:2]
            crop = frame[y1 : min(y2, h), x1 : min(x2, w)].copy()
            rows = self._df_by_frame.get(idx)
            if rows is not None:
                ref = min(crop.shape[:2])
                radius = max(3, int(ref / 80.0))
                font_scale = max(0.2, ref / 900.0)
                thickness = max(1, int(ref / 500.0))
                grey_r = max(2, int(radius * 0.75))
                for _, row in rows.iterrows():
                    if pd.isna(row["X"]) or pd.isna(row["Y"]):
                        continue
                    tid = int(row["TrajectoryID"])
                    cx = int(round(row["X"])) - x1
                    cy = int(round(row["Y"])) - y1
                    if tid == self._track_a:
                        color, r = self._COLOR_A, radius
                    elif tid == self._track_b:
                        color, r = self._COLOR_B, radius
                    else:
                        color, r = (128, 128, 128), grey_r
                    cv2.circle(crop, (cx, cy), r, color, thickness)
                    cv2.putText(
                        crop,
                        f"T{tid}",
                        (cx + r + 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        thickness,
                    )
            self.crops[idx] = crop
            self.progress.emit(i + 1, total)
        cap.release()
        self.finished.emit()


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------


def _compute_fixed_crop(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    frame_range: Tuple[int, int],
    video_path: str,
) -> Tuple[int, int, int, int]:
    """Return fixed (x1, y1, x2, y2) covering both tracks across the range."""
    rows = df.loc[
        (df["FrameID"].between(frame_range[0], frame_range[1]))
        & (df["TrajectoryID"].isin([track_a, track_b]))
    ]
    valid = rows.dropna(subset=["X", "Y"])
    cap = cv2.VideoCapture(video_path)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()
    if valid.empty:
        return 0, 0, vid_w, vid_h
    x1 = max(int(valid["X"].min()) - _CROP_MARGIN, 0)
    y1 = max(int(valid["Y"].min()) - _CROP_MARGIN, 0)
    x2 = min(int(valid["X"].max()) + _CROP_MARGIN, vid_w)
    y2 = min(int(valid["Y"].max()) + _CROP_MARGIN, vid_h)
    return x1, y1, x2, y2


class FramePickerDialog(QDialog):
    """Dialog for selecting the split frame within an event's range.

    All frames in the range are pre-decoded in a background thread so the
    slider seeks instantly.  Track overlays are baked into the crops so the
    user can clearly see which animals are track_a vs track_b.

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
        self.setMinimumSize(520, 460)

        self._track_a = track_a
        self._track_b = track_b
        self._frame_range = frame_range
        # Extend the display window to give temporal context
        _cap = cv2.VideoCapture(video_path)
        _total = max(int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_range[1])
        _cap.release()
        load_start = max(0, frame_range[0] - _CONTEXT_FRAMES)
        load_end = min(_total, frame_range[1] + _CONTEXT_FRAMES)
        self._selected: int = (frame_range[0] + frame_range[1]) // 2

        layout = QVBoxLayout(self)

        # Instructions with colour-coded track legend
        instr = QLabel(
            f"Scrub to the frame where the swap most likely occurred, then click OK.<br>"
            f"<span style='color:red; font-weight:bold'>&#9679; Track {track_a}</span>"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:#0055cc; font-weight:bold'>&#9679; Track {track_b}</span>"
        )
        instr.setTextFormat(Qt.TextFormat.RichText)
        instr.setWordWrap(True)
        layout.addWidget(instr)

        # Canvas
        self._canvas = InteractiveCanvas()
        self._canvas.setMinimumSize(420, 300)
        layout.addWidget(self._canvas, stretch=1)

        # Progress bar (hidden once loading is complete)
        total_frames = load_end - load_start + 1
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, total_frames)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Pre-loading frames\u2026 %v / %m")
        layout.addWidget(self._progress_bar)

        # Slider (disabled until all frames are loaded)
        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(load_start)
        self._slider.setMaximum(load_end)
        self._slider.setValue(self._selected)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, stretch=1)
        self._frame_label = QLabel(str(self._selected))
        self._frame_label.setMinimumWidth(70)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        # Buttons (OK disabled until loaded)
        self._btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        layout.addWidget(self._btns)

        # Start background loader
        crop_box = _compute_fixed_crop(df, track_a, track_b, frame_range, video_path)
        self._loader = _FrameRangeLoader(
            video_path,
            df,
            track_a,
            track_b,
            crop_box,
            load_start,
            load_end,
            self,
        )
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def selected_frame(self) -> int:
        """Return the frame index chosen by the user."""
        return self._selected

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_progress(self, loaded: int, total: int) -> None:
        self._progress_bar.setValue(loaded)

    def _on_load_finished(self) -> None:
        self._progress_bar.hide()
        self._slider.setEnabled(True)
        self._btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._refresh()

    def _on_slider(self, value: int) -> None:
        self._selected = value
        self._frame_label.setText(str(value))
        self._refresh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Display the pre-rendered crop for the current slider position."""
        crop = self._loader.crops.get(self._selected)
        if crop is None or crop.size == 0:
            return
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format.Format_RGB888)
        self._canvas.set_pixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def _free_frames(self) -> None:
        """Stop the loader thread and free pre-loaded frame memory."""
        self._loader.requestInterruption()
        self._loader.crops.clear()

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def accept(self):
        self._free_frames()
        super().accept()

    def reject(self):
        self._free_frames()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        self._free_frames()
        super().closeEvent(event)
