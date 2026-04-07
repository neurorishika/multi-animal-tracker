"""Unified event resolution dialog for RefineKit.

Handles all event types: swap, flicker, fragmentation, absorption,
phantom, multi-shuffle.  Embeds a frame picker, event info header,
action radio buttons, and a reclassify combo.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal

from hydra_suite.refinekit.core.event_types import EventType

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
        """Decode frames in the configured range, crop to the bounding box, draw track markers, and store results in ``self.crops``."""
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
