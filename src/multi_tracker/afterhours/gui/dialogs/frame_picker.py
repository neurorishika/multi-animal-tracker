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
from PySide6.QtCore import QThread, Signal

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
