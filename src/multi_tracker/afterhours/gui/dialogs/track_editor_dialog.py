"""Track editor dialog for MAT-afterhours.

Replaces the old ``ResolutionDialog`` / ``FramePickerDialog`` pair with a
single **timeline-based** editor.  The user sees:

* A video preview (top) that updates as they scrub the timeline.
* An interactive fragment timeline (middle) where they can split, delete,
  and drag-reassign fragments.
* An event header (top-right) showing *why* they were brought here.
* An explicit **Apply** button that writes changes to disk.

The dialog receives the full trajectory DataFrame; the crop and frame
window are computed from the event's involved tracks ±
:data:`_CONTEXT_FRAMES` padding.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.event_types import SuspicionEvent
from multi_tracker.afterhours.core.track_editor_model import EditOp, TrackEditorModel
from multi_tracker.afterhours.gui.widgets.interactive_canvas import InteractiveCanvas
from multi_tracker.afterhours.gui.widgets.timeline_editor import (
    PALETTE_RGB,
    TimelineEditorWidget,
)

logger = logging.getLogger(__name__)

_CONTEXT_FRAMES = 15
_CROP_MARGIN = 80


# ---------------------------------------------------------------------------
# Background frame loader
# ---------------------------------------------------------------------------


class _FrameLoader(QThread):
    """Decode raw frames (no overlay) in the visible range.

    Frames are stored as raw BGR crops so the dialog can re-draw overlays
    live as the user edits fragments.
    """

    progress = Signal(int, int)
    finished = Signal()

    def __init__(
        self,
        video_path: str,
        crop_box: Tuple[int, int, int, int],
        frame_start: int,
        frame_end: int,
        parent=None,
    ):
        super().__init__(parent)
        self._path = video_path
        self._crop = crop_box
        self._start = frame_start
        self._end = frame_end
        self.frames: Dict[int, np.ndarray] = {}

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
            self.frames[idx] = crop
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
    """Fixed crop box covering *tracks* across *frame_range*."""
    rows = df.loc[
        df["FrameID"].between(frame_range[0], frame_range[1])
        & df["TrajectoryID"].isin(tracks)
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


# BGR palette derived from the canonical RGB palette in timeline_editor.
_PALETTE_BGR: List[Tuple[int, int, int]] = [(b, g, r) for r, g, b in PALETTE_RGB]


# ---------------------------------------------------------------------------
# TrackEditorDialog
# ---------------------------------------------------------------------------


class TrackEditorDialog(QDialog):
    """Timeline-based track editor dialog.

    Parameters
    ----------
    video_path:
        Path to the video file.
    df:
        Full trajectory DataFrame.
    event:
        The :class:`SuspicionEvent` that brought the user here.
    parent:
        Parent widget.

    After the dialog closes, call :meth:`ops` to get the list of
    :class:`EditOp` objects (empty if the user cancelled or made no edits).
    Call :meth:`applied` to check if the user clicked Apply.
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
        self._applied = False
        self._edit_ops: List[EditOp] = []

        self.setWindowTitle("Track Editor")
        self.setMinimumSize(700, 600)

        # --- Compute expanded frame range ---
        cap = cv2.VideoCapture(video_path)
        total_frames = max(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, event.frame_range[1]
        )
        cap.release()

        self._load_start = max(0, event.frame_range[0] - _CONTEXT_FRAMES)
        self._load_end = min(total_frames, event.frame_range[1] + _CONTEXT_FRAMES)

        # --- Crop box (computed from target tracks only) ---
        self._crop_box = _compute_crop(
            df,
            event.involved_tracks,
            event.frame_range,
            video_path,
        )
        self._crop_origin = (self._crop_box[0], self._crop_box[1])

        # --- Discover tracks that appear inside the crop during the segment ---
        x1, y1, x2, y2 = self._crop_box
        region_df = df[df["FrameID"].between(self._load_start, self._load_end)]
        in_crop = region_df[
            region_df["X"].between(x1, x2) & region_df["Y"].between(y1, y2)
        ]
        all_visible = sorted(in_crop["TrajectoryID"].unique().tolist())
        # Always include the target tracks even if NaN
        for t in event.involved_tracks:
            if t not in all_visible:
                all_visible.append(t)
        all_visible = sorted(all_visible)

        # --- Build fragment model ---
        self._model = TrackEditorModel(
            df,
            all_visible,
            (self._load_start, self._load_end),
        )

        # --- Build UI ---
        layout = QVBoxLayout(self)

        # Event header
        header = self._build_header(event)
        layout.addWidget(header)

        # Video preview
        self._canvas = InteractiveCanvas()
        self._canvas.setMinimumSize(420, 260)
        layout.addWidget(self._canvas, stretch=2)

        # Progress bar (hidden once frames are loaded)
        self._progress = QProgressBar()
        total = self._load_end - self._load_start + 1
        self._progress.setRange(0, total)
        self._progress.setFormat("Loading frames\u2026 %v / %m")
        layout.addWidget(self._progress)

        # Frame slider
        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(self._load_start)
        self._slider.setMaximum(self._load_end)
        self._slider.setValue(event.frame_peak)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self._slider, stretch=1)
        self._frame_label = QLabel(str(event.frame_peak))
        self._frame_label.setMinimumWidth(70)
        slider_row.addWidget(self._frame_label)

        # Marker size control
        slider_row.addWidget(QLabel("Marker:"))
        self._marker_spin = QSpinBox()
        self._marker_spin.setRange(50, 400)
        self._marker_spin.setValue(150)
        self._marker_spin.setSuffix("%")
        self._marker_spin.setSingleStep(25)
        self._marker_spin.setToolTip("Marker and label size")
        self._marker_spin.valueChanged.connect(lambda _: self._refresh_preview())
        slider_row.addWidget(self._marker_spin)

        layout.addLayout(slider_row)

        # Timeline editor
        self._timeline = TimelineEditorWidget()
        self._timeline.set_model(self._model)
        self._timeline.setMinimumHeight(100)
        self._timeline.setMaximumHeight(300)
        self._timeline.model_changed.connect(self._on_model_changed)
        self._timeline.frame_cursor_changed.connect(self._on_timeline_cursor)
        layout.addWidget(self._timeline, stretch=1)

        # Instruction text
        instr = QLabel(
            "<b>Right-click</b> a bar to split · "
            "<b>Drag</b> a bar to move it to another lane · "
            "<b>Delete</b> key to remove · "
            "<b>Ctrl-Z</b> to undo"
        )
        instr.setTextFormat(Qt.TextFormat.RichText)
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #9cdcfe; font-size: 11px; padding: 2px 4px;")
        layout.addWidget(instr)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Write changes to disk and refresh trajectories")
        self._apply_btn.clicked.connect(self._on_apply)
        self._apply_btn.setEnabled(False)
        btn_row.addWidget(self._apply_btn)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._on_undo)
        self._undo_btn.setEnabled(False)
        btn_row.addWidget(self._undo_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # --- Current frame state ---
        self._current_frame = event.frame_peak

        # --- Start background loader ---
        self._loader = _FrameLoader(
            video_path,
            self._crop_box,
            self._load_start,
            self._load_end,
            self,
        )
        self._loader.progress.connect(self._on_load_progress)
        self._loader.finished.connect(self._on_load_finished)
        self._loader.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def applied(self) -> bool:
        """True if the user clicked Apply."""
        return self._applied

    @property
    def edit_ops(self) -> List[EditOp]:
        """Edit operations produced by the last Apply."""
        return self._edit_ops

    @property
    def reviewed_range(self) -> Tuple[int, int]:
        """The frame range the user reviewed (for deprioritization)."""
        return (self._load_start, self._load_end)

    @property
    def reviewed_tracks(self) -> List[int]:
        """All tracks visible in the editor."""
        return self._model.visible_tracks

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(event: SuspicionEvent) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        flag = QLabel("\u26a0")
        flag.setStyleSheet("font-size: 13px; color: #f48771;")
        flag.setToolTip(event.event_type.value)
        lay.addWidget(flag)

        score_lbl = QLabel(f"Score: {event.score:.2f}")
        score_lbl.setStyleSheet("font-weight: bold; color: #f48771; font-size: 12px;")
        lay.addWidget(score_lbl)

        tracks_text = ", ".join(f"T{t}" for t in event.involved_tracks)
        lay.addWidget(QLabel(tracks_text))

        lay.addWidget(
            QLabel(f"frames {event.frame_range[0]}\u2013{event.frame_range[1]}")
        )
        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _on_load_progress(self, loaded: int, total: int) -> None:
        self._progress.setValue(loaded)

    def _on_load_finished(self) -> None:
        self._progress.hide()
        self._slider.setEnabled(True)
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._frame_label.setText(str(value))
        self._timeline.set_cursor_frame(value)
        self._refresh_preview()

    def _on_timeline_cursor(self, frame: int) -> None:
        self._current_frame = frame
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._frame_label.setText(str(frame))
        self._refresh_preview()

    def _on_model_changed(self) -> None:
        self._apply_btn.setEnabled(True)
        self._undo_btn.setEnabled(self._model.can_undo)
        self._timeline.refresh()
        self._refresh_preview()

    def _on_undo(self) -> None:
        if self._model.undo():
            self._on_model_changed()

    def _on_apply(self) -> None:
        self._edit_ops = self._model.compute_ops()
        self._applied = True
        self.accept()

    # ------------------------------------------------------------------
    # Preview rendering
    # ------------------------------------------------------------------

    def _refresh_preview(self) -> None:
        """Draw the current frame with fragment-aware overlays."""
        raw = self._loader.frames.get(self._current_frame)
        if raw is None or raw.size == 0:
            return

        display = raw.copy()
        self._draw_overlays(display, self._current_frame)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        ch, cw = rgb.shape[:2]
        qimg = QImage(rgb.data, cw, ch, 3 * cw, QImage.Format.Format_RGB888)
        self._canvas.set_pixmap(QPixmap.fromImage(qimg))

    def _draw_overlays(self, crop: np.ndarray, frame_idx: int) -> None:
        """Draw markers on *crop* using the current fragment model state."""
        ox, oy = self._crop_origin
        ref = min(crop.shape[:2])
        scale = self._marker_spin.value() / 100.0
        radius = max(4, int(ref / 40.0 * scale))
        font_scale = max(0.35, ref / 400.0 * scale)
        thickness = max(1, int(ref / 200.0 * scale))
        outline_th = max(1, thickness + 1)

        # Build mapping: original_track_id → current lane for non-deleted
        # fragments covering this frame
        original_ids: Dict[int, int] = {}
        for frag in self._model.fragments:
            if frag.deleted:
                continue
            if frag.frame_start <= frame_idx <= frag.frame_end:
                original_ids[frag.original_track_id] = frag.track_id

        # Get rows for this frame (only model-visible tracks)
        sub = self._df[
            (self._df["FrameID"] == frame_idx)
            & self._df["TrajectoryID"].isin(self._model.visible_tracks)
        ]

        for _, row in sub.iterrows():
            if pd.isna(row["X"]) or pd.isna(row["Y"]):
                continue
            orig_tid = int(row["TrajectoryID"])
            cx = int(round(row["X"])) - ox
            cy = int(round(row["Y"])) - oy

            current_lane = original_ids.get(orig_tid)
            if current_lane is None:
                continue

            b, g, r = _PALETTE_BGR[current_lane % len(_PALETTE_BGR)]
            color = (b, g, r)

            # Filled circle with dark outline for contrast
            cv2.circle(crop, (cx, cy), radius, (0, 0, 0), outline_th, cv2.LINE_AA)
            cv2.circle(crop, (cx, cy), radius, color, cv2.FILLED, cv2.LINE_AA)

            # Text with dark outline for readability
            label = f"T{current_lane}"
            tx, ty = cx + radius + 3, cy + 2
            cv2.putText(
                crop,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                outline_th,
                cv2.LINE_AA,
            )
            cv2.putText(
                crop,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def reject(self):
        self._loader.requestInterruption()
        self._loader.frames.clear()
        super().reject()

    def accept(self):
        self._loader.requestInterruption()
        self._loader.frames.clear()
        super().accept()

    def closeEvent(self, event):  # noqa: N802
        self._loader.requestInterruption()
        self._loader.frames.clear()
        super().closeEvent(event)
