"""Unified event resolution dialog for MAT-afterhours.

Handles all event types: swap, flicker, fragmentation, absorption,
phantom, multi-shuffle.  Embeds a frame picker, event info header,
action radio buttons, and a reclassify combo.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.event_types import (
    EVENT_TYPE_COLOR,
    EVENT_TYPE_LABEL,
    EventType,
    SuspicionEvent,
)
from multi_tracker.afterhours.gui.widgets.interactive_canvas import InteractiveCanvas

_CROP_MARGIN = 80
_CONTEXT_FRAMES = 15  # extra frames loaded before / after the event window

# Map event types to available action labels
_ACTIONS: Dict[EventType, List[str]] = {
    EventType.SWAP: [
        "Swap IDs at split frame",
        "Keep order (no swap)",
        "Skip (no correction)",
    ],
    EventType.FLICKER: [
        "Erase flicker (undo both swaps)",
        "Skip (no correction)",
    ],
    EventType.FRAGMENTATION: [
        "Merge fragments into one ID",
        "Skip (no correction)",
    ],
    EventType.ABSORPTION: [
        "Split + swap at re-appearance",
        "Keep order (no swap)",
        "Skip (no correction)",
    ],
    EventType.PHANTOM: [
        "Delete phantom track",
        "Delete in frame range only",
        "Skip (no correction)",
    ],
    EventType.MULTI_SHUFFLE: [
        "Swap IDs at split frame (pairwise)",
        "Skip (no correction)",
    ],
}


# ---------------------------------------------------------------------------
# Background frame loader (simplified — crops around all involved tracks)
# ---------------------------------------------------------------------------


class _FrameLoader(QThread):
    """Decode frames in event range with track overlays."""

    progress = Signal(int, int)
    finished = Signal()

    _PALETTE = [
        (0, 0, 220),  # red
        (220, 0, 0),  # blue
        (0, 180, 0),  # green
        (220, 180, 0),  # cyan
        (0, 128, 220),  # orange
        (180, 0, 180),  # magenta
    ]

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        involved_tracks: List[int],
        crop_box: Tuple[int, int, int, int],
        frame_start: int,
        frame_end: int,
        parent=None,
    ):
        super().__init__(parent)
        self._path = video_path
        self._df_by_frame: Dict[int, pd.DataFrame] = {
            fid: grp for fid, grp in df.groupby("FrameID")
        }
        self._target_tracks = set(involved_tracks)
        self._tracks = involved_tracks
        self._crop = crop_box
        self._start = frame_start
        self._end = frame_end
        self.crops: Dict[int, np.ndarray] = {}

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self.finished.emit()
            return
        x1, y1, x2, y2 = self._crop
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
                    if tid in self._target_tracks:
                        ci = self._tracks.index(tid)
                        color = self._PALETTE[ci % len(self._PALETTE)]
                        r = radius
                    else:
                        color = (128, 128, 128)
                        r = grey_r
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


def _compute_crop(
    df: pd.DataFrame,
    tracks: List[int],
    frame_range: Tuple[int, int],
    video_path: str,
) -> Tuple[int, int, int, int]:
    rows = df.loc[
        (df["FrameID"].between(frame_range[0], frame_range[1]))
        & (df["TrajectoryID"].isin(tracks))
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


# ---------------------------------------------------------------------------
# ResolutionDialog
# ---------------------------------------------------------------------------


class ResolutionDialog(QDialog):
    """Unified event resolution dialog.

    Parameters
    ----------
    video_path:
        Path to the video file.
    df:
        Trajectory DataFrame.
    event:
        The :class:`SuspicionEvent` being reviewed.
    parent:
        Parent widget.
    """

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        event: SuspicionEvent,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._df = df
        self._event = event
        self._selected_frame: int = event.frame_peak
        # Extend the display window to give temporal context
        _cap = cv2.VideoCapture(video_path)
        _total = max(int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, event.frame_range[1])
        _cap.release()
        _load_start = max(0, event.frame_range[0] - _CONTEXT_FRAMES)
        _load_end = min(_total, event.frame_range[1] + _CONTEXT_FRAMES)
        self.setWindowTitle("Resolve Event")
        self.setMinimumSize(620, 520)

        layout = QVBoxLayout(self)

        # --- Event info header ---
        header = self._build_header(event)
        layout.addWidget(header)

        # --- Canvas ---
        self._canvas = InteractiveCanvas()
        self._canvas.setMinimumSize(420, 300)
        layout.addWidget(self._canvas, stretch=1)

        # --- Progress bar ---
        total = _load_end - _load_start + 1
        self._progress = QProgressBar()
        self._progress.setRange(0, total)
        self._progress.setFormat("Loading frames\u2026 %v / %m")
        layout.addWidget(self._progress)

        # --- Frame slider ---
        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(_load_start)
        self._slider.setMaximum(_load_end)
        self._slider.setValue(self._selected_frame)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, stretch=1)
        self._frame_label = QLabel(str(self._selected_frame))
        self._frame_label.setMinimumWidth(70)
        slider_row.addWidget(self._frame_label)
        layout.addLayout(slider_row)

        # --- Reclassify combo ---
        reclass_row = QHBoxLayout()
        reclass_row.addWidget(QLabel("Classification:"))
        self._reclass_combo = QComboBox()
        for et in EventType:
            self._reclass_combo.addItem(EVENT_TYPE_LABEL[et], et)
        self._reclass_combo.setCurrentIndex(list(EventType).index(event.event_type))
        self._reclass_combo.currentIndexChanged.connect(self._on_reclassify)
        reclass_row.addWidget(self._reclass_combo, stretch=1)
        layout.addLayout(reclass_row)

        # --- Action radios ---
        self._action_group = QGroupBox("Action")
        self._action_layout = QVBoxLayout(self._action_group)
        self._action_radios: List[QRadioButton] = []
        layout.addWidget(self._action_group)
        self._populate_actions(event.event_type)

        # --- Buttons ---
        self._btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        self._btns.accepted.connect(self.accept)
        self._btns.rejected.connect(self.reject)
        layout.addWidget(self._btns)

        # --- Start background loader ---
        crop_box = _compute_crop(
            df, event.involved_tracks, event.frame_range, video_path
        )
        self._loader = _FrameLoader(
            video_path,
            df,
            event.involved_tracks,
            crop_box,
            _load_start,
            _load_end,
            self,
        )
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def event(self) -> SuspicionEvent:
        """Return the (possibly reclassified) event."""
        return self._event

    def selected_frame(self) -> int:
        """The frame the user chose (for split-based actions)."""
        return self._selected_frame

    def selected_action(self) -> str:
        """Label of the chosen action radio."""
        for radio in self._action_radios:
            if radio.isChecked():
                return radio.text()
        return "Skip (no correction)"

    def effective_event_type(self) -> EventType:
        """Return the (possibly reclassified) event type."""
        return self._event.event_type

    # ------------------------------------------------------------------
    # Header builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(event: SuspicionEvent) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        # Type badge
        color = EVENT_TYPE_COLOR.get(event.event_type, "#cccccc")
        badge = QLabel(EVENT_TYPE_LABEL.get(event.event_type, "Unknown"))
        badge.setStyleSheet(
            f"background-color: {color}; color: #1e1e1e; font-weight: bold; "
            f"padding: 2px 8px; border-radius: 4px; font-size: 12px;"
        )
        lay.addWidget(badge)

        # Score
        score_lbl = QLabel(f"Score: {event.score:.2f}")
        score_lbl.setStyleSheet("font-weight: bold; color: #f48771; font-size: 12px;")
        lay.addWidget(score_lbl)

        # Signals
        sig = "+".join(event.signals) if event.signals else "—"
        sig_lbl = QLabel(sig)
        sig_lbl.setStyleSheet("color: #9cdcfe; font-size: 12px;")
        lay.addWidget(sig_lbl)

        # Tracks
        tracks_text = ", ".join(f"T{t}" for t in event.involved_tracks)
        lay.addWidget(QLabel(tracks_text))

        # Frame range
        lay.addWidget(QLabel(f"frames {event.frame_range[0]}–{event.frame_range[1]}"))
        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _populate_actions(self, event_type: EventType) -> None:
        """Fill action radios for the given event type."""
        # Clear existing
        for radio in self._action_radios:
            self._action_layout.removeWidget(radio)
            radio.deleteLater()
        self._action_radios.clear()

        actions = _ACTIONS.get(event_type, ["Skip (no correction)"])
        for i, label in enumerate(actions):
            radio = QRadioButton(label)
            if i == 0:
                radio.setChecked(True)
            self._action_radios.append(radio)
            self._action_layout.addWidget(radio)

    def _on_reclassify(self, index: int) -> None:
        new_type = self._reclass_combo.itemData(index)
        if new_type is None:
            return
        self._event = SuspicionEvent(
            event_type=new_type,
            involved_tracks=self._event.involved_tracks,
            frame_peak=self._event.frame_peak,
            frame_range=self._event.frame_range,
            score=self._event.score,
            signals=self._event.signals,
            region_label=self._event.region_label,
            region_boundary=self._event.region_boundary,
        )
        self._populate_actions(new_type)

    # ------------------------------------------------------------------
    # Loader slots
    # ------------------------------------------------------------------

    def _on_load_progress(self, loaded: int, total: int) -> None:
        self._progress.setValue(loaded)

    def _on_load_finished(self) -> None:
        self._progress.hide()
        self._slider.setEnabled(True)
        self._btns.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self._refresh()

    # ------------------------------------------------------------------
    # Slider / refresh
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._selected_frame = value
        self._frame_label.setText(str(value))
        self._refresh()

    def _refresh(self) -> None:
        crop = self._loader.crops.get(self._selected_frame)
        if crop is None or crop.size == 0:
            return
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format.Format_RGB888)
        self._canvas.set_pixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _free_frames(self) -> None:
        self._loader.requestInterruption()
        self._loader.crops.clear()

    def accept(self):
        self._free_frames()
        super().accept()

    def reject(self):
        self._free_frames()
        super().reject()

    def closeEvent(self, event):  # noqa: N802
        self._free_frames()
        super().closeEvent(event)
