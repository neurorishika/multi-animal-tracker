"""Fragment Merge Wizard dialog for MAT-afterhours.

Presents merge hypotheses side-by-side with synchronized video playback,
allowing the user to accept, reject, or skip merge decisions.

Each accepted merge is applied immediately via
:meth:`CorrectionWriter.apply_merge`, and the candidate graph is updated
so subsequent decisions reflect previous merges.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from multi_tracker.afterhours.core.correction_writer import CorrectionWriter
from multi_tracker.afterhours.core.event_types import EventType, SuspicionEvent
from multi_tracker.afterhours.core.merge_candidates import (
    MergeCandidate,
    SwapCandidate,
    TrackSegment,
    build_candidates,
    build_swap_candidates,
    extract_segments,
    update_after_merge,
)
from multi_tracker.afterhours.gui.widgets.synced_video_grid import SyncedVideoGrid
from multi_tracker.data.detection_cache import DetectionCache

# Type alias for hypotheses that can be either merge or swap
Hypothesis = MergeCandidate | SwapCandidate

logger = logging.getLogger(__name__)

_CONTEXT_FRAMES = 30
_CROP_MARGIN = 80
_HYPOTHESES_PER_PAGE = 1
_TRAIL_LENGTH = 15  # frames of visible "tail" behind the head dot

# Overlay colours (BGR)
_MERGED_COLOR = (0, 220, 0)  # green — the "what-if" merged trail
_GAP_COLOR = (0, 220, 0)  # same green, dashed in gap
_CONTEXT_COLOR = (100, 100, 100)  # gray for surrounding tracks
# Merge hypothesis colours (green family — "join / continue")
_MERGE_COLORS = [
    (0, 220, 0),  # vivid green
    (0, 200, 80),  # green-teal
    (80, 200, 0),  # yellow-green
]
# Swap hypothesis colours (amber/orange family — "identity change")
_SWAP_COLORS = [
    (0, 160, 255),  # orange
    (0, 100, 240),  # red-orange
    (0, 130, 215),  # amber-orange
]
_ORPHAN_COLOR = (80, 80, 80)  # dim gray for the orphaned pre-swap tail
_SWAP_MARKER_COLOR = (0, 200, 255)  # yellow for swap frame marker

# Detection overlay alpha (0–1); 0.5 gives a translucent ellipse/OBB.
_DET_ALPHA = 0.45


# ---------------------------------------------------------------------------
# Non-Qt model
# ---------------------------------------------------------------------------


class MergeWizardModel:
    """State machine for the merge wizard, independent of Qt."""

    def __init__(
        self,
        df: pd.DataFrame,
        segments: List[TrackSegment],
        candidates: Dict[int, List[MergeCandidate]],
        writer: CorrectionWriter,
        swap_candidates: Optional[Dict[int, List[SwapCandidate]]] = None,
    ):
        self._df = df
        self._segments = segments
        self._candidates = candidates
        self._swap_candidates: Dict[int, List[SwapCandidate]] = swap_candidates or {}
        self._writer = writer

        # Order: process most confident hypotheses first (merge + swap)
        all_source_ids = set(candidates.keys()) | set(self._swap_candidates.keys())
        self._merge_order = sorted(
            all_source_ids,
            key=lambda sid: self._best_score(sid),
            reverse=True,
        )
        self._current_idx = 0
        self._hypothesis_page = 0
        self._undo_stack: List[Tuple[int, int, pd.DataFrame]] = []
        self._merges_applied = 0
        self._skipped: set = set()
        # (source_id, target_id, frame_peak) for flagged pairs
        self._flagged: List[Tuple[int, int, int]] = []

    def _best_score(self, sid: int) -> float:
        """Best hypothesis score for a source across merge + swap."""
        mc = self._candidates.get(sid, [])
        sc = self._swap_candidates.get(sid, [])
        best_mc = mc[0].score if mc else 0.0
        best_sc = sc[0].score if sc else 0.0
        return max(best_mc, best_sc)

    @property
    def current_source_id(self) -> Optional[int]:
        """Track ID of the current dying track, or None if finished."""
        while self._current_idx < len(self._merge_order):
            sid = self._merge_order[self._current_idx]
            has_hyps = sid in self._candidates or sid in self._swap_candidates
            if has_hyps and sid not in self._skipped:
                return sid
            self._current_idx += 1
        return None

    @property
    def current_source(self) -> Optional[TrackSegment]:
        sid = self.current_source_id
        if sid is None:
            return None
        for s in self._segments:
            if s.track_id == sid:
                return s
        return None

    def _all_hypotheses(self) -> List[Hypothesis]:
        """All deduplicated hypotheses for the current source, sorted by score.

        If the same target appears via both merge *and* swap, only the
        higher-scoring mechanism is kept so every displayed option is
        mutually exclusive.
        """
        sid = self.current_source_id
        if sid is None:
            return []
        merge = list(self._candidates.get(sid, []))
        swap = list(self._swap_candidates.get(sid, []))
        combined = merge + swap
        # Merges first (MergeCandidate → False=0, SwapCandidate → True=1),
        # then within each group sort by descending score.
        combined.sort(key=lambda c: (isinstance(c, SwapCandidate), -c.score))
        # Deduplicate by target_id — keep highest-scoring per target
        seen_targets: set = set()
        unique: List[Hypothesis] = []
        for h in combined:
            if h.target_id not in seen_targets:
                seen_targets.add(h.target_id)
                unique.append(h)
        return unique

    @property
    def current_hypotheses(self) -> List[Hypothesis]:
        """Return current page of hypotheses (deduplicated, sorted)."""
        all_h = self._all_hypotheses()
        start = self._hypothesis_page * _HYPOTHESES_PER_PAGE
        return all_h[start : start + _HYPOTHESES_PER_PAGE]

    @property
    def _total_hypotheses(self) -> int:
        return len(self._all_hypotheses())

    @property
    def has_more_hypotheses(self) -> bool:
        return (
            self._hypothesis_page + 1
        ) * _HYPOTHESES_PER_PAGE < self._total_hypotheses

    @property
    def has_prev_hypotheses(self) -> bool:
        return self._hypothesis_page > 0

    def accept(self, hypothesis_idx: int) -> Optional[Tuple[int, int]]:
        """Accept a hypothesis (merge or swap). Returns (source_id, target_id)."""
        hyps = self.current_hypotheses
        if hypothesis_idx >= len(hyps):
            return None
        candidate = hyps[hypothesis_idx]

        # Save undo state
        self._undo_stack.append(
            (candidate.source_id, candidate.target_id, self._df.copy())
        )

        if isinstance(candidate, SwapCandidate):
            self._writer.apply_swap_merge(
                candidate.source_id,
                candidate.target_id,
                candidate.swap_frame,
            )
            self._df = self._writer.df
            # Full rebuild — swap creates a new dead fragment
            self._rebuild_graph()
        else:
            # Merge: efficient incremental update
            self._writer.apply_merge([candidate.source_id, candidate.target_id])
            self._df = self._writer.df
            self._segments, self._candidates = update_after_merge(
                self._segments,
                self._candidates,
                candidate.source_id,
                candidate.target_id,
            )
            # Clean stale swap references
            self._swap_candidates.pop(candidate.source_id, None)
            self._swap_candidates.pop(candidate.target_id, None)
            for k in list(self._swap_candidates.keys()):
                self._swap_candidates[k] = [
                    c
                    for c in self._swap_candidates[k]
                    if c.target_id not in (candidate.source_id, candidate.target_id)
                ]
                if not self._swap_candidates[k]:
                    del self._swap_candidates[k]
            self._rebuild_order()

        self._merges_applied += 1
        self._hypothesis_page = 0
        self._advance()
        return (candidate.source_id, candidate.target_id)

    def flag(self) -> Optional[int]:
        """Flag the current source track for detailed fixing in the editor.

        Records the source with its best hypothesis as a max-score
        FRAGMENTATION suspicion event and advances without applying any
        merge.  Returns the source track ID or None.
        """
        sid = self.current_source_id
        if sid is None:
            return None
        seg_by_id = {s.track_id: s for s in self._segments}
        src = seg_by_id.get(sid)
        hyps = self.current_hypotheses
        if src and hyps:
            best = hyps[0]
            tgt = seg_by_id.get(best.target_id)
            frame_peak = (
                (src.frame_death + tgt.frame_birth) // 2 if tgt else src.frame_death
            )
            self._flagged.append((sid, best.target_id, frame_peak))
        elif src:
            self._flagged.append((sid, sid, src.frame_death))
        self._hypothesis_page = 0
        self._advance()
        return sid

    def skip(self) -> None:
        """Skip the current source track."""
        sid = self.current_source_id
        if sid is not None:
            self._skipped.add(sid)
        self._hypothesis_page = 0
        self._advance()

    def next_page(self) -> None:
        """Show next page of hypotheses for current source."""
        if self.has_more_hypotheses:
            self._hypothesis_page += 1

    def prev_page(self) -> None:
        """Show previous page of hypotheses for current source."""
        if self._hypothesis_page > 0:
            self._hypothesis_page -= 1

    def undo(self) -> bool:
        """Undo the last merge/swap. Returns True if successful."""
        if not self._undo_stack:
            return False
        source_id, target_id, old_df = self._undo_stack.pop()
        # Restore DataFrame
        self._writer._df = old_df
        self._writer._write_atomic()
        self._df = old_df

        # Full rebuild
        self._rebuild_graph()

        self._merges_applied -= 1
        # Position cursor back to the source that was just undone
        for i, sid in enumerate(self._merge_order):
            if sid == source_id:
                self._current_idx = i
                break
        self._hypothesis_page = 0
        return True

    def _rebuild_graph(self) -> None:
        """Rebuild segments, merge candidates, and swap candidates from df."""
        last_frame = int(self._df["FrameID"].max())
        self._segments = extract_segments(self._df, last_frame)
        self._candidates = build_candidates(self._segments)
        self._swap_candidates = build_swap_candidates(
            self._df,
            self._segments,
        )
        self._rebuild_order()

    def _rebuild_order(self) -> None:
        """Rebuild merge_order from current candidate dicts."""
        all_ids = set(self._candidates.keys()) | set(self._swap_candidates.keys())
        self._merge_order = sorted(
            all_ids,
            key=lambda sid: self._best_score(sid),
            reverse=True,
        )

    @property
    def is_finished(self) -> bool:
        return self.current_source_id is None

    @property
    def progress(self) -> Tuple[int, int]:
        """(completed + skipped, total)."""
        total = len(self._merge_order)
        done = self._current_idx
        return (min(done, total), total)

    @property
    def merges_applied(self) -> int:
        return self._merges_applied

    @property
    def flagged_events(self) -> List[SuspicionEvent]:
        """Return FRAGMENTATION SuspicionEvents for all flagged pairs."""
        events: List[SuspicionEvent] = []
        seg_by_id = {s.track_id: s for s in self._segments}
        for src_id, tgt_id, frame_peak in self._flagged:
            src = seg_by_id.get(src_id)
            tgt = seg_by_id.get(tgt_id)
            f_start = src.frame_birth if src else max(0, frame_peak - 30)
            f_end = tgt.frame_death if tgt else frame_peak + 30
            events.append(
                SuspicionEvent(
                    event_type=EventType.FRAGMENTATION,
                    involved_tracks=[src_id, tgt_id],
                    frame_peak=frame_peak,
                    frame_range=(f_start, f_end),
                    score=1.0,
                    signals=["flag"],
                )
            )
        return events

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def _advance(self) -> None:
        """Move to next unresolved source."""
        self._current_idx += 1


# ---------------------------------------------------------------------------
# Crop helper (shared with track_editor_dialog)
# ---------------------------------------------------------------------------

# Cache video dimensions to avoid re-opening VideoCapture on every navigation.
_video_dims_cache: Dict[str, Tuple[int, int]] = {}


def _compute_crop_for_merge(
    df: pd.DataFrame,
    source: TrackSegment,
    targets: List[Hypothesis],
    video_path: str,
) -> Tuple[int, int, int, int]:
    """Crop box covering source death region + all target regions."""
    if video_path not in _video_dims_cache:
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()
        _video_dims_cache[video_path] = (w, h)
    vid_w, vid_h = _video_dims_cache[video_path]

    # Collect all relevant positions
    xs = [source.pos_death[0]]
    ys = [source.pos_death[1]]
    all_track_ids = [source.track_id] + [t.target_id for t in targets]

    frame_start = max(0, source.frame_death - _CONTEXT_FRAMES)
    frame_end = source.frame_death + _CONTEXT_FRAMES
    for hyp in targets:
        if isinstance(hyp, SwapCandidate):
            frame_start = min(
                frame_start,
                max(0, hyp.swap_frame - _CONTEXT_FRAMES),
            )
            frame_end = max(frame_end, hyp.swap_frame + _CONTEXT_FRAMES)
        else:
            frame_end = max(
                frame_end,
                hyp.gap_frames + source.frame_death + _CONTEXT_FRAMES,
            )

    rows = df.loc[
        df["FrameID"].between(frame_start, frame_end)
        & df["TrajectoryID"].isin(all_track_ids)
    ]
    valid = rows.dropna(subset=["X", "Y"])
    if not valid.empty:
        xs.extend(valid["X"].values.tolist())
        ys.extend(valid["Y"].values.tolist())

    x1 = max(int(min(xs)) - _CROP_MARGIN, 0)
    y1 = max(int(min(ys)) - _CROP_MARGIN, 0)
    x2 = min(int(max(xs)) + _CROP_MARGIN, vid_w)
    y2 = min(int(max(ys)) + _CROP_MARGIN, vid_h)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------


def _tail_mask(frames: np.ndarray, frame_idx: int) -> np.ndarray:
    """Boolean mask selecting frames in [frame_idx - _TRAIL_LENGTH, frame_idx]."""
    return (frames >= frame_idx - _TRAIL_LENGTH) & (frames <= frame_idx)


# ---------------------------------------------------------------------------
# Detection-cache helpers
# ---------------------------------------------------------------------------


def _discover_detection_cache(video_path: str) -> Optional[Path]:
    """Find the detection cache ``.npz`` for *video_path*.

    Searches ``{stem}_caches/{stem}_detection_cache_*.npz`` first, then the
    legacy flat layout ``{dir}/{stem}_detection_cache_*.npz``.
    Returns ``None`` if no cache is found.
    """
    vp = Path(video_path).expanduser()
    stem = vp.stem
    parent = vp.parent

    # Current layout: {parent}/{stem}_caches/{stem}_detection_cache_*.npz
    cache_dir = parent / f"{stem}_caches"
    if cache_dir.is_dir():
        hits = sorted(cache_dir.glob(f"{stem}_detection_cache_*.npz"))
        if hits:
            return hits[-1]  # newest / last alphabetically

    # Legacy flat layout
    hits = sorted(parent.glob(f"{stem}_detection_cache_*.npz"))
    if hits:
        return hits[-1]

    return None


def _load_resize_factor(video_path: str) -> float:
    """Read the ``resize_factor`` from the tracking config saved next to *video_path*.

    Falls back to ``1.0`` if the config file doesn't exist or the key is absent.
    """
    vp = Path(video_path).expanduser()
    cfg_path = vp.parent / f"{vp.stem}_config.json"
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r") as fh:
                data = json.load(fh)
            return float(data.get("resize_factor", 1.0))
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    return 1.0


class _FrameDetections:
    """Read-only wrapper around a :class:`DetectionCache` opened for reading.

    Pre-computes per-detection semi-axes from the cached ``shapes`` array so
    that drawing is cheap at render time.  Coordinates are scaled from the
    *resized* space stored in the cache to the *original* pixel space used by
    the afterhours CSV.
    """

    def __init__(self, cache: DetectionCache, inv_resize: float):
        self._cache = cache
        self._inv_resize = inv_resize  # 1.0 / resize_factor

    def get(self, frame_idx: int):
        """Return ``(meas, semi_axes, obb_corners)`` or ``None``.

        * *meas* — ``(N, 3)`` float32 ``[cx, cy, theta]`` in original-res pixels.
        * *semi_axes* — ``(N, 2)`` float32 ``[semi_major, semi_minor]``.
        * *obb_corners* — list of ``(4, 2)`` float32 arrays (empty when BG-sub).
        """
        try:
            meas, _sizes, shapes, _conf, obb, _ids, _hints, _dmask = (
                self._cache.get_frame(frame_idx)
            )
        except Exception:
            return None
        if not meas:
            return None

        s = self._inv_resize
        meas_arr = np.array(meas, dtype=np.float32)
        meas_arr[:, 0] *= s
        meas_arr[:, 1] *= s

        # Recover semi-axes from (ellipse_area, aspect_ratio)
        shapes_arr = (
            np.array(shapes, dtype=np.float32)
            if shapes
            else np.empty((0, 2), dtype=np.float32)
        )
        semi_axes = np.empty((len(meas_arr), 2), dtype=np.float32)
        for i in range(len(meas_arr)):
            if i < len(shapes_arr) and shapes_arr.ndim == 2:
                area, ar = shapes_arr[i]
                ar = max(ar, 0.01)
                # area = π * (a/2)*(b/2)  → a*b = 4*area/π
                b = np.sqrt(4.0 * abs(area) / (np.pi * ar))
                a = ar * b
                semi_axes[i] = [a * 0.5 * s, b * 0.5 * s]
            else:
                semi_axes[i] = [8.0, 4.0]  # minimal fallback

        obb_scaled: list = []
        if obb:
            for corners in obb:
                obb_scaled.append((np.asarray(corners, dtype=np.float32) * s))

        return meas_arr, semi_axes, obb_scaled


def _build_det_index_map(
    df: pd.DataFrame,
    track_ids: set,
    frame_start: int,
    frame_end: int,
) -> Dict[int, Set[int]]:
    """Build a per-frame map of detection indices for the given tracks.

    Returns ``{FrameID: {det_index, ...}}`` extracted from the
    ``DetectionID`` column.  Detection index is ``int(DetectionID) % 10000``.
    Frames/tracks with missing ``DetectionID`` are silently skipped.
    """
    if "DetectionID" not in df.columns:
        return {}
    sub = df[
        df["TrajectoryID"].isin(track_ids)
        & df["FrameID"].between(frame_start, frame_end)
    ]
    valid = sub["DetectionID"].notna()
    if not valid.any():
        return {}
    frames = sub.loc[valid, "FrameID"].values.astype(int)
    det_indices = sub.loc[valid, "DetectionID"].values.astype(int) % 10000
    result: Dict[int, Set[int]] = {}
    for f, d in zip(frames, det_indices):
        result.setdefault(f, set()).add(d)
    return result


def _draw_detections(
    img: np.ndarray,
    dets: _FrameDetections,
    frame_idx: int,
    ox: float,
    oy: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    allowed_indices: Optional[Set[int]] = None,
) -> None:
    """Draw detections for *frame_idx* onto *img*.

    If OBB corners are available (YOLO), draws rotated rectangles.
    Otherwise draws oriented ellipses from the cached shape data.
    Coordinates are shifted by ``(ox, oy)`` to convert from full-frame
    to crop-relative space.

    When *allowed_indices* is not ``None``, only detections whose within-frame
    index is in the set are drawn (i.e. only those assigned to displayed tracks).
    """
    result = dets.get(frame_idx)
    if result is None:
        return

    meas_arr, semi_axes, obb_corners = result

    overlay = img.copy()

    if obb_corners:
        # Draw OBB rotated rectangles
        for i, corners in enumerate(obb_corners):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            pts = corners.copy()
            pts[:, 0] -= ox
            pts[:, 1] -= oy
            ipts = pts.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [ipts], True, color, thickness, cv2.LINE_AA)
    else:
        # Draw oriented ellipses from shape data
        for i in range(len(meas_arr)):
            if allowed_indices is not None and i not in allowed_indices:
                continue
            cx = int(round(meas_arr[i, 0] - ox))
            cy = int(round(meas_arr[i, 1] - oy))
            theta = meas_arr[i, 2]
            a = int(round(semi_axes[i, 0]))
            b = int(round(semi_axes[i, 1]))
            if a < 2 or b < 2:
                continue
            angle_deg = -np.degrees(theta)
            cv2.ellipse(
                overlay,
                (cx, cy),
                (a, b),
                angle_deg,
                0,
                360,
                color,
                thickness,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, _DET_ALPHA, img, 1.0 - _DET_ALPHA, 0, img)


def _make_overlay_fn(
    df: pd.DataFrame,
    source_id: int,
    target_id: int,
    crop_box: Tuple[int, int, int, int],
    merged_color: Tuple[int, int, int],
    frame_start: int,
    frame_end: int,
    frame_dets: Optional[_FrameDetections] = None,
) -> Callable:
    """Create a 'what if merged' overlay for one hypothesis panel.

    Draws context tracks as short tails in gray, then the hypothetical
    merged track (source + gap bridge + target) in *merged_color*.
    Only tracks that actually pass through the crop region within the
    visible time window are drawn.  When *frame_dets* is provided the
    real detection outlines are rendered for every frame.
    """
    ox, oy = crop_box[0], crop_box[1]
    cx1, cy1, cx2, cy2 = crop_box

    # --- Pre-extract source & target trajectories -----------------------
    source_df = df[df["TrajectoryID"] == source_id].dropna(subset=["X", "Y"])
    source_df = source_df[(source_df["X"] > 0) & (source_df["Y"] > 0)]
    target_df = df[df["TrajectoryID"] == target_id].dropna(subset=["X", "Y"])
    target_df = target_df[(target_df["X"] > 0) & (target_df["Y"] > 0)]

    src_frames = source_df["FrameID"].values
    src_xs = source_df["X"].values
    src_ys = source_df["Y"].values

    tgt_frames = target_df["FrameID"].values
    tgt_xs = target_df["X"].values
    tgt_ys = target_df["Y"].values

    # --- Pre-extract context tracks (crop + time bounded) ---------------
    ctx_df = df[
        df["FrameID"].between(frame_start, frame_end)
        & df["X"].between(cx1, cx2)
        & df["Y"].between(cy1, cy2)
        & (df["X"] > 0)
        & (df["Y"] > 0)
    ]
    ctx_ids = sorted(
        set(ctx_df["TrajectoryID"].dropna().unique()) - {source_id, target_id}
    )
    ctx_data: list = []
    for cid in ctx_ids:
        cgrp = ctx_df[ctx_df["TrajectoryID"] == cid]
        if len(cgrp) < 2:
            continue
        ctx_data.append((cgrp["FrameID"].values, cgrp["X"].values, cgrp["Y"].values))

    # --- Per-frame detection index map (only tracked detections) --------
    _visible_ids = {source_id, target_id} | set(ctx_ids)
    _det_map = _build_det_index_map(df, _visible_ids, frame_start, frame_end)

    def overlay(bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        # 0. Detection outlines (only those assigned to displayed tracks)
        if frame_dets is not None:
            _draw_detections(
                bgr,
                frame_dets,
                frame_idx,
                ox,
                oy,
                _CONTEXT_COLOR,
                1,
                allowed_indices=_det_map.get(frame_idx),
            )

        # 1. Draw context track tails (semi-transparent)
        ctx_layer = bgr.copy()
        for c_frames, c_xs, c_ys in ctx_data:
            cmask = _tail_mask(c_frames, frame_idx)
            if cmask.sum() < 2:
                continue
            cpts = np.column_stack((c_xs[cmask] - ox, c_ys[cmask] - oy)).astype(
                np.int32
            )
            cv2.polylines(ctx_layer, [cpts], False, _CONTEXT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(ctx_layer, tuple(cpts[-1]), 3, _CONTEXT_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(ctx_layer, 0.4, bgr, 0.6, 0, bgr)

        # 2. Source tail
        mask_s = _tail_mask(src_frames, frame_idx)
        pts_s = None
        if mask_s.any():
            pts_s = np.column_stack((src_xs[mask_s] - ox, src_ys[mask_s] - oy)).astype(
                np.int32
            )
            cv2.polylines(bgr, [pts_s], False, merged_color, 2, cv2.LINE_AA)

        #    Bridge dashed line (visible once source is dead)
        if pts_s is not None and frame_idx >= src_frames[-1] and len(tgt_frames) > 0:
            # Show static end-point of source
            src_last = (int(src_xs[-1] - ox), int(src_ys[-1] - oy))
            tgt_start = (int(tgt_xs[0] - ox), int(tgt_ys[0] - oy))
            _draw_dashed_line(bgr, src_last, tgt_start, merged_color, 2, dash_len=6)

        #    Target tail
        mask_t = _tail_mask(tgt_frames, frame_idx)
        pts_t = None
        if mask_t.any():
            pts_t = np.column_stack((tgt_xs[mask_t] - ox, tgt_ys[mask_t] - oy)).astype(
                np.int32
            )
            cv2.polylines(bgr, [pts_t], False, merged_color, 2, cv2.LINE_AA)

        # 3. Head dot + label
        head_pt = None
        if pts_t is not None and len(pts_t) > 0:
            head_pt = tuple(pts_t[-1])
        elif (
            len(tgt_frames) > 0
            and frame_idx > src_frames[-1]
            and frame_idx < tgt_frames[0]
        ):
            # Animate head dot linearly through the gap
            gap_len = max(1, int(tgt_frames[0]) - int(src_frames[-1]))
            t = (frame_idx - int(src_frames[-1])) / gap_len
            ix = int(round(src_xs[-1] + t * (tgt_xs[0] - src_xs[-1]) - ox))
            iy = int(round(src_ys[-1] + t * (tgt_ys[0] - src_ys[-1]) - oy))
            head_pt = (ix, iy)
        elif pts_s is not None and len(pts_s) > 0:
            head_pt = tuple(pts_s[-1])
        if head_pt is not None:
            cv2.circle(bgr, head_pt, 5, merged_color, -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"T{source_id}+T{target_id}",
                (head_pt[0] + 8, head_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                merged_color,
                1,
                cv2.LINE_AA,
            )

        return bgr

    return overlay


def _make_swap_overlay_fn(
    df: pd.DataFrame,
    source_id: int,
    target_id: int,
    swap_frame: int,
    crop_box: Tuple[int, int, int, int],
    merged_color: Tuple[int, int, int],
    frame_start: int,
    frame_end: int,
    frame_dets: Optional[_FrameDetections] = None,
) -> Callable:
    """Create a 'what if swapped' overlay for one hypothesis panel.

    Shows both source and target tracks clearly with short tails so the
    user can see exactly which identities are involved:
    * Real detection outlines when *frame_dets* is available.
    * Context tracks as short gray tails (semi-transparent).
    * Source trail in *merged_color* (the identity being continued).
    * Target's pre-swap trail in a distinct colour (cyan).
    * Target's post-swap trail relabeled in *merged_color* (continuation).
    * Dashed bridge from source death to target at swap point.
    * Diamond marker at the swap frame.
    """
    ox, oy = crop_box[0], crop_box[1]
    cx1, cy1, cx2, cy2 = crop_box
    _TARGET_PRE_COLOR = (255, 200, 0)  # cyan-ish in BGR for target's own trail

    # --- Pre-extract source & target trajectories ---
    source_df = df[df["TrajectoryID"] == source_id].dropna(subset=["X", "Y"])
    source_df = source_df[(source_df["X"] > 0) & (source_df["Y"] > 0)]
    target_df = df[df["TrajectoryID"] == target_id].dropna(subset=["X", "Y"])
    target_df = target_df[(target_df["X"] > 0) & (target_df["Y"] > 0)]

    src_frames = source_df["FrameID"].values
    src_xs = source_df["X"].values
    src_ys = source_df["Y"].values

    tgt_frames = target_df["FrameID"].values
    tgt_xs = target_df["X"].values
    tgt_ys = target_df["Y"].values

    # Split target into pre-swap and post-swap
    tgt_pre_mask = tgt_frames < swap_frame
    tgt_post_mask = tgt_frames >= swap_frame

    tgt_pre_frames = tgt_frames[tgt_pre_mask]
    tgt_pre_xs = tgt_xs[tgt_pre_mask]
    tgt_pre_ys = tgt_ys[tgt_pre_mask]

    tgt_post_frames = tgt_frames[tgt_post_mask]
    tgt_post_xs = tgt_xs[tgt_post_mask]
    tgt_post_ys = tgt_ys[tgt_post_mask]

    # --- Pre-extract context tracks (crop + time bounded) ---
    ctx_df = df[
        df["FrameID"].between(frame_start, frame_end)
        & df["X"].between(cx1, cx2)
        & df["Y"].between(cy1, cy2)
        & (df["X"] > 0)
        & (df["Y"] > 0)
    ]
    ctx_ids = sorted(
        set(ctx_df["TrajectoryID"].dropna().unique()) - {source_id, target_id}
    )
    ctx_data: list = []
    for cid in ctx_ids:
        cgrp = ctx_df[ctx_df["TrajectoryID"] == cid]
        if len(cgrp) < 2:
            continue
        ctx_data.append((cgrp["FrameID"].values, cgrp["X"].values, cgrp["Y"].values))

    # --- Per-frame detection index map (only tracked detections) --------
    _visible_ids = {source_id, target_id} | set(ctx_ids)
    _det_map = _build_det_index_map(df, _visible_ids, frame_start, frame_end)

    def overlay(bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        # 0. Detection outlines (only those assigned to displayed tracks)
        if frame_dets is not None:
            _draw_detections(
                bgr,
                frame_dets,
                frame_idx,
                ox,
                oy,
                _CONTEXT_COLOR,
                1,
                allowed_indices=_det_map.get(frame_idx),
            )

        # 1. Context track tails (semi-transparent)
        ctx_layer = bgr.copy()
        for c_frames, c_xs, c_ys in ctx_data:
            cmask = _tail_mask(c_frames, frame_idx)
            if cmask.sum() < 2:
                continue
            cpts = np.column_stack((c_xs[cmask] - ox, c_ys[cmask] - oy)).astype(
                np.int32
            )
            cv2.polylines(ctx_layer, [cpts], False, _CONTEXT_COLOR, 1, cv2.LINE_AA)
            cv2.circle(ctx_layer, tuple(cpts[-1]), 3, _CONTEXT_COLOR, -1, cv2.LINE_AA)
        cv2.addWeighted(ctx_layer, 0.4, bgr, 0.6, 0, bgr)

        # 2. Target pre-swap trail in distinct colour (its own identity)
        mask_pre = _tail_mask(tgt_pre_frames, frame_idx)
        pts_pre = None
        if mask_pre.sum() >= 2:
            pts_pre = np.column_stack(
                (tgt_pre_xs[mask_pre] - ox, tgt_pre_ys[mask_pre] - oy)
            ).astype(np.int32)
            cv2.polylines(bgr, [pts_pre], False, _TARGET_PRE_COLOR, 2, cv2.LINE_AA)
        # Label target head (pre-swap)
        if pts_pre is not None and len(pts_pre) > 0 and frame_idx < swap_frame:
            tp = tuple(pts_pre[-1])
            cv2.circle(bgr, tp, 5, _TARGET_PRE_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"T{target_id}",
                (tp[0] + 8, tp[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                _TARGET_PRE_COLOR,
                1,
                cv2.LINE_AA,
            )

        # 3. Source trail in merged_color
        mask_s = _tail_mask(src_frames, frame_idx)
        pts_s = None
        if mask_s.any():
            pts_s = np.column_stack((src_xs[mask_s] - ox, src_ys[mask_s] - oy)).astype(
                np.int32
            )
            cv2.polylines(bgr, [pts_s], False, merged_color, 2, cv2.LINE_AA)
        # Label source head
        if pts_s is not None and len(pts_s) > 0 and frame_idx <= src_frames[-1]:
            sp = tuple(pts_s[-1])
            cv2.circle(bgr, sp, 5, merged_color, -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"T{source_id}",
                (sp[0] + 8, sp[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                merged_color,
                1,
                cv2.LINE_AA,
            )

        # 4. Dashed bridge from source death to target at swap_frame
        if (
            frame_idx >= src_frames[-1]
            and len(tgt_post_frames) > 0
            and len(src_frames) > 0
        ):
            src_end = (int(src_xs[-1] - ox), int(src_ys[-1] - oy))
            swap_pt = (
                int(tgt_post_xs[0] - ox),
                int(tgt_post_ys[0] - oy),
            )
            _draw_dashed_line(bgr, src_end, swap_pt, merged_color, 2, dash_len=6)

        # 5. Target post-swap tail in merged_color (relabeled continuation)
        mask_post = _tail_mask(tgt_post_frames, frame_idx)
        pts_post = None
        if mask_post.any():
            pts_post = np.column_stack(
                (tgt_post_xs[mask_post] - ox, tgt_post_ys[mask_post] - oy)
            ).astype(np.int32)
            cv2.polylines(bgr, [pts_post], False, merged_color, 2, cv2.LINE_AA)

        # 6. Swap frame diamond marker
        if frame_idx >= swap_frame and len(tgt_post_frames) > 0:
            sx = int(tgt_post_xs[0] - ox)
            sy = int(tgt_post_ys[0] - oy)
            cv2.drawMarker(
                bgr,
                (sx, sy),
                _SWAP_MARKER_COLOR,
                cv2.MARKER_DIAMOND,
                10,
                2,
                cv2.LINE_AA,
            )

        # 7. Head dot + label for the "continued" track after swap
        if pts_post is not None and len(pts_post) > 0:
            head_pt = tuple(pts_post[-1])
            cv2.circle(bgr, head_pt, 5, merged_color, -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"T{source_id}(was T{target_id})",
                (head_pt[0] + 8, head_pt[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                merged_color,
                1,
                cv2.LINE_AA,
            )

        # 8. After swap, show orphaned target-pre tail endpoint label
        if frame_idx >= swap_frame and len(tgt_pre_frames) > 0:
            oend = (int(tgt_pre_xs[-1] - ox), int(tgt_pre_ys[-1] - oy))
            cv2.circle(bgr, oend, 3, _ORPHAN_COLOR, -1, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"T{target_id}(orphan)",
                (oend[0] + 6, oend[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                _ORPHAN_COLOR,
                1,
                cv2.LINE_AA,
            )

        return bgr

    return overlay


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    color: tuple,
    thickness: int,
    dash_len: int = 8,
) -> None:
    """Draw a dashed line on *img*."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if dist < 1:
        return
    n_dashes = max(1, int(dist / dash_len))
    for i in range(0, n_dashes, 2):
        t0 = i / n_dashes
        t1 = min((i + 1) / n_dashes, 1.0)
        p0 = (int(x1 + t0 * (x2 - x1)), int(y1 + t0 * (y2 - y1)))
        p1 = (int(x1 + t1 * (x2 - x1)), int(y1 + t1 * (y2 - y1)))
        cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MergeWizardDialog
# ---------------------------------------------------------------------------


class MergeWizardDialog(QDialog):
    """Fragment Merge Wizard — side-by-side hypothesis comparison.

    Parameters
    ----------
    video_path:
        Path to the video file.
    df:
        Full trajectory DataFrame.
    segments:
        Pre-computed track segments.
    candidates:
        Pre-computed merge candidate graph.
    writer:
        Correction writer for applying merges.
    parent:
        Parent widget.
    """

    def __init__(
        self,
        video_path: str,
        df: pd.DataFrame,
        segments: List[TrackSegment],
        candidates: Dict[int, List[MergeCandidate]],
        writer: CorrectionWriter,
        parent=None,
        swap_candidates: Optional[Dict[int, List[SwapCandidate]]] = None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._model = MergeWizardModel(
            df,
            segments,
            candidates,
            writer,
            swap_candidates=swap_candidates,
        )

        # --- Discover and load the detection cache (read-only) ----------
        self._frame_dets: Optional[_FrameDetections] = None
        cache_path = _discover_detection_cache(video_path)
        if cache_path is not None:
            try:
                cache = DetectionCache(str(cache_path), mode="r")
                if cache.is_compatible() and cache._loaded_data is not None:
                    inv_resize = 1.0 / _load_resize_factor(video_path)
                    self._frame_dets = _FrameDetections(cache, inv_resize)
                    logger.info("Detection cache loaded: %s", cache_path)
                else:
                    logger.warning("Detection cache incompatible: %s", cache_path)
            except Exception:
                logger.warning(
                    "Failed to load detection cache: %s", cache_path, exc_info=True
                )
        else:
            logger.info("No detection cache found for %s", video_path)

        self.setWindowTitle("Fragment Merge Wizard")
        self.setMinimumSize(900, 600)

        self._build_ui()
        self._update_view()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Header: "Track T42 died at frame 1234 — Hypothesis 2 of 7 (Merge)"
        self._header = QLabel()
        self._header.setStyleSheet(
            "font-weight: bold; font-size: 14px; padding: 6px; color: #9cdcfe;"
        )
        root.addWidget(self._header)

        # Single hypothesis detail badge
        self._stat_label = QLabel()
        self._stat_label.setStyleSheet(
            "background: #2d2d2d; border-radius: 4px; padding: 6px 10px; "
            "font-size: 12px; color: #ccc;"
        )
        self._stat_label.setTextFormat(Qt.TextFormat.RichText)
        self._stat_label.setVisible(False)
        root.addWidget(self._stat_label)

        # --- Video panel flanked by Prev/Next hypothesis buttons ---
        grid_row = QHBoxLayout()
        grid_row.setSpacing(4)

        self._btn_prev_page = QPushButton("\u25c0\nPrev")
        self._btn_prev_page.setToolTip("Previous hypothesis (P)")
        self._btn_prev_page.setFixedWidth(52)
        self._btn_prev_page.setStyleSheet(
            "font-size: 11px; padding: 6px; border-radius: 4px;"
        )
        self._btn_prev_page.clicked.connect(self._on_prev_page)
        grid_row.addWidget(self._btn_prev_page)

        self._grid = SyncedVideoGrid(n_panels=1)
        grid_row.addWidget(self._grid, stretch=1)

        self._btn_next_page = QPushButton("\u25b6\nNext")
        self._btn_next_page.setToolTip("Next hypothesis (N)")
        self._btn_next_page.setFixedWidth(52)
        self._btn_next_page.setStyleSheet(
            "font-size: 11px; padding: 6px; border-radius: 4px;"
        )
        self._btn_next_page.clicked.connect(self._on_next_page)
        grid_row.addWidget(self._btn_next_page)

        root.addLayout(grid_row, stretch=1)

        # Single accept button for the displayed hypothesis
        accept_row = QHBoxLayout()
        accept_row.addSpacing(56)
        self._accept_btn = QPushButton("\u2714 Accept This Hypothesis")
        self._accept_btn.setToolTip("Accept the displayed hypothesis (Enter or 1)")
        self._accept_btn.setStyleSheet(
            "background: #2d5a2d; color: white; font-weight: bold; "
            "padding: 8px 16px; border-radius: 4px; font-size: 13px;"
        )
        self._accept_btn.clicked.connect(lambda: self._on_accept(0))
        self._accept_btn.setVisible(False)
        accept_row.addWidget(self._accept_btn, stretch=1)
        accept_row.addSpacing(56)
        root.addLayout(accept_row)

        # Timeline strip
        self._timeline_label = QLabel()
        self._timeline_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; color: #ccc; "
            "padding: 4px; background: #1e1e1e; border-radius: 3px;"
        )
        self._timeline_label.setWordWrap(True)
        root.addWidget(self._timeline_label)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._flag_btn = QPushButton("\U0001f6a9 Flag for Editor")
        self._flag_btn.setToolTip("Flag this track for detailed editing (F)")
        self._flag_btn.setStyleSheet(
            "background: #6b5a1a; color: #ffd700; font-weight: bold; "
            "padding: 6px 12px; border-radius: 4px;"
        )
        self._flag_btn.clicked.connect(self._on_flag)
        btn_row.addWidget(self._flag_btn)

        self._btn_skip = QPushButton("Skip Track")
        self._btn_skip.setToolTip("No good merge \u2014 skip (S)")
        self._btn_skip.setStyleSheet(
            "background: #5a3a2d; color: white; padding: 6px 12px; "
            "border-radius: 4px;"
        )
        self._btn_skip.clicked.connect(self._on_skip)
        btn_row.addWidget(self._btn_skip)

        btn_row.addStretch()

        self._btn_undo = QPushButton("Undo")
        self._btn_undo.setToolTip("Undo last merge (Ctrl+Z)")
        self._btn_undo.clicked.connect(self._on_undo)
        btn_row.addWidget(self._btn_undo)

        self._btn_finish = QPushButton("Finish")
        self._btn_finish.setToolTip("Close wizard and proceed to swap proofreading")
        self._btn_finish.clicked.connect(self.accept)
        btn_row.addWidget(self._btn_finish)

        root.addLayout(btn_row)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setFormat("%v / %m tracks reviewed")
        root.addWidget(self._progress_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def merges_applied(self) -> int:
        return self._model.merges_applied

    # ------------------------------------------------------------------
    # View update
    # ------------------------------------------------------------------

    def _update_view(self) -> None:
        """Refresh all UI elements for the current model state."""
        if self._model.is_finished:
            self._on_finished()
            return

        source = self._model.current_source
        hyps = self._model.current_hypotheses
        done, total = self._model.progress
        n_merged = self._model.merges_applied
        n_flagged = len(self._model._flagged)
        total_hyps = self._model._total_hypotheses
        hyp_idx = self._model._hypothesis_page + 1

        # Current hypothesis (single)
        hyp = hyps[0] if hyps else None
        is_swap = isinstance(hyp, SwapCandidate) if hyp else False
        kind = "Swap" if is_swap else "Merge"

        # Header
        self._header.setText(
            f"Track T{source.track_id} died at frame {source.frame_death}  "
            f"\u2014  Hypothesis {hyp_idx} of {total_hyps} ({kind})  |  "
            f"Track {done + 1}/{total}  \u00b7  "
            f"{n_merged} merged  \u00b7  {n_flagged} flagged"
        )

        # Progress
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(done)

        # Stat badge
        if hyp is not None:
            if is_swap:
                self._stat_label.setText(
                    f"<b>SWAP: T{source.track_id} \u2190 T{hyp.target_id}</b>  "
                    f"Score: {hyp.score:.2f}  |  "
                    f"swap@f{hyp.swap_frame}  |  "
                    f"dist: {hyp.min_distance:.1f}px  |  "
                    f"heading: {hyp.heading_continuity:.2f}"
                )
                self._stat_label.setStyleSheet(
                    "background: #2d2d2d; border-radius: 4px; padding: 6px 10px; "
                    "font-size: 12px; color: #ccc; border-left: 4px solid #ffa040;"
                )
                self._accept_btn.setText("\u2714 Accept Swap")
                self._accept_btn.setStyleSheet(
                    "background: #5a3d00; color: #ffd27a; font-weight: bold; "
                    "padding: 8px 16px; border-radius: 4px; font-size: 13px;"
                )
            else:
                self._stat_label.setText(
                    f"<b>MERGE: T{source.track_id} \u2192 T{hyp.target_id}</b>  "
                    f"Score: {hyp.score:.2f}  |  "
                    f"gap: {hyp.gap_frames}f  |  "
                    f"dist: {hyp.spatial_dist:.1f}px  |  "
                    f"heading: {hyp.heading_agreement:.2f}"
                )
                self._stat_label.setStyleSheet(
                    "background: #2d2d2d; border-radius: 4px; padding: 6px 10px; "
                    "font-size: 12px; color: #ccc; border-left: 4px solid #00dc00;"
                )
                self._accept_btn.setText("\u2714 Accept Merge")
                self._accept_btn.setStyleSheet(
                    "background: #2d5a2d; color: white; font-weight: bold; "
                    "padding: 8px 16px; border-radius: 4px; font-size: 13px;"
                )
            self._stat_label.setVisible(True)
            self._accept_btn.setVisible(True)
            self._accept_btn.setEnabled(True)
        else:
            self._stat_label.setVisible(False)
            self._accept_btn.setVisible(False)

        # Navigation buttons
        self._btn_prev_page.setEnabled(self._model.has_prev_hypotheses)
        self._btn_next_page.setEnabled(self._model.has_more_hypotheses)
        self._btn_undo.setEnabled(self._model.merges_applied > 0)

        # Timeline strip
        self._update_timeline_strip(source, hyps)

        # Load video
        self._load_hypothesis_video(source, hyps)

    def _update_timeline_strip(
        self, source: TrackSegment, hyps: List[Hypothesis]
    ) -> None:
        """Update the timeline label showing fragment timelines."""
        lines = [
            f"  T{source.track_id}: frame {source.frame_birth}\u2013{source.frame_death}"
        ]
        seg_by_id = {s.track_id: s for s in self._model._segments}
        if hyps:
            hyp = hyps[0]
            tgt = seg_by_id.get(hyp.target_id)
            if tgt:
                tag = "SWAP " if isinstance(hyp, SwapCandidate) else ""
                extra = ""
                if isinstance(hyp, SwapCandidate):
                    extra = f"  swap@f{hyp.swap_frame}"
                lines.append(
                    f"  {tag}T{hyp.target_id}: "
                    f"frame {tgt.frame_birth}\u2013{tgt.frame_death}{extra}"
                )
        lines.append(
            "  Keys: Enter/1=Accept  F=Flag  S=Skip  N/P/\u2190/\u2192=Hypothesis  "
            "Space=Play  Ctrl+\u2190/\u2192=Step  Ctrl+Z=Undo"
        )
        self._timeline_label.setText("\n".join(lines))

    def _load_hypothesis_video(
        self,
        source: TrackSegment,
        hyps: List[Hypothesis],
    ) -> None:
        """Configure the video grid for the current hypotheses."""
        if not hyps:
            return

        df = self._model.df

        # Determine frame range encompassing all hypotheses
        seg_by_id = {s.track_id: s for s in self._model._segments}
        frame_start = max(0, source.frame_death - _CONTEXT_FRAMES)
        frame_end = source.frame_death + _CONTEXT_FRAMES

        for hyp in hyps:
            tgt = seg_by_id.get(hyp.target_id)
            if tgt:
                if isinstance(hyp, SwapCandidate):
                    # Focus the clip around the swap frame only
                    frame_start = min(
                        frame_start,
                        max(0, hyp.swap_frame - _CONTEXT_FRAMES),
                    )
                    frame_end = max(frame_end, hyp.swap_frame + _CONTEXT_FRAMES)
                else:
                    # For merge candidates, focus on the gap:
                    # source.frame_death → target.frame_birth
                    frame_end = max(frame_end, tgt.frame_birth + _CONTEXT_FRAMES)

        # Compute crop box (shared across all panels for fair comparison)
        crop_box = _compute_crop_for_merge(df, source, hyps, self._video_path)

        # Each panel gets the same crop but a single overlay
        crop_boxes = [crop_box]
        hyp = hyps[0]
        if isinstance(hyp, MergeCandidate):
            color = _MERGE_COLORS[0]
        else:
            color = _SWAP_COLORS[0]
        if isinstance(hyp, SwapCandidate):
            fn = _make_swap_overlay_fn(
                df,
                source.track_id,
                hyp.target_id,
                hyp.swap_frame,
                crop_box,
                color,
                frame_start,
                frame_end,
                frame_dets=self._frame_dets,
            )
        else:
            fn = _make_overlay_fn(
                df,
                source.track_id,
                hyp.target_id,
                crop_box,
                color,
                frame_start,
                frame_end,
                frame_dets=self._frame_dets,
            )
        overlay_fns = [fn]

        self._grid.configure(
            self._video_path,
            crop_boxes,
            frame_start,
            frame_end,
            overlay_fns,
            auto_play=True,
        )

        # Kick off background prefetch for adjacent hypothesis pages
        QTimer.singleShot(0, lambda: self._prefetch_adjacent(source))

    def _prefetch_adjacent(self, source: TrackSegment) -> None:
        """Pre-load video frames for the previous / next hypothesis pages."""
        all_hyps = self._model._all_hypotheses()
        current_page = self._model._hypothesis_page
        for page_offset in (-1, +1):
            page = current_page + page_offset
            if page < 0:
                continue
            start = page * _HYPOTHESES_PER_PAGE
            hyps = all_hyps[start : start + _HYPOTHESES_PER_PAGE]
            if not hyps:
                continue
            config = self._build_hypothesis_frame_range(source, hyps)
            if config:
                vp, cb, fs, fe = config
                self._grid.prefetch(vp, cb, fs, fe)

    def _build_hypothesis_frame_range(
        self,
        source: TrackSegment,
        hyps: List[Hypothesis],
    ) -> Optional[Tuple[str, list, int, int]]:
        """Return (video_path, crop_boxes, frame_start, frame_end) without I/O side-effects."""
        if not hyps:
            return None
        df = self._model.df
        seg_by_id = {s.track_id: s for s in self._model._segments}
        frame_start = max(0, source.frame_death - _CONTEXT_FRAMES)
        frame_end = source.frame_death + _CONTEXT_FRAMES
        for hyp in hyps:
            tgt = seg_by_id.get(hyp.target_id)
            if tgt:
                if isinstance(hyp, SwapCandidate):
                    frame_start = min(
                        frame_start, max(0, hyp.swap_frame - _CONTEXT_FRAMES)
                    )
                    frame_end = max(frame_end, hyp.swap_frame + _CONTEXT_FRAMES)
                else:
                    frame_end = max(frame_end, tgt.frame_birth + _CONTEXT_FRAMES)
        crop_box = _compute_crop_for_merge(df, source, hyps, self._video_path)
        return (self._video_path, [crop_box], frame_start, frame_end)

    def _on_finished(self) -> None:
        """All tracks processed."""
        n = self._model.merges_applied
        n_flagged = len(self._model._flagged)
        self._header.setText(
            f"Merge wizard complete \u2014 {n} merge{'s' if n != 1 else ''} applied"
            f"  \u00b7  {n_flagged} flagged"
        )
        self._stat_label.setVisible(False)
        self._accept_btn.setVisible(False)
        self._flag_btn.setEnabled(False)
        self._btn_prev_page.setEnabled(False)
        self._btn_next_page.setEnabled(False)
        self._btn_skip.setEnabled(False)
        self._timeline_label.setText("")

        done, total = self._model.progress
        self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(total)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_accept(self, idx: int) -> None:
        result = self._model.accept(idx)
        if result:
            logger.info("Merge accepted: T%d → T%d", result[0], result[1])
        self._update_view()

    def _on_flag(self) -> None:
        result = self._model.flag()
        if result is not None:
            logger.info("Flagged track T%d for editor", result)
        self._update_view()

    def _on_skip(self) -> None:
        self._model.skip()
        self._update_view()

    def _on_next_page(self) -> None:
        self._model.next_page()
        self._update_view()

    def _on_prev_page(self) -> None:
        self._model.prev_page()
        self._update_view()

    def _on_undo(self) -> None:
        if self._model.undo():
            self._update_view()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        mod = event.modifiers()

        # 1 or Enter → accept the displayed hypothesis
        if key in (Qt.Key.Key_1, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._model.current_hypotheses:
                self._on_accept(0)
                return

        # F → flag current track for detailed editor
        if key == Qt.Key.Key_F:
            self._on_flag()
            return

        # N → next page, P → prev page
        if key == Qt.Key.Key_N:
            self._on_next_page()
            return
        if key == Qt.Key.Key_P:
            self._on_prev_page()
            return

        # S → skip
        if key == Qt.Key.Key_S:
            self._on_skip()
            return

        # Space → play/pause
        if key == Qt.Key.Key_Space:
            self._grid.toggle_play()
            return

        # Left/Right → navigate hypotheses (Ctrl+Left/Right → frame step)
        if key == Qt.Key.Key_Left:
            if mod & Qt.KeyboardModifier.ControlModifier:
                self._grid.step_back()
            else:
                self._on_prev_page()
            return
        if key == Qt.Key.Key_Right:
            if mod & Qt.KeyboardModifier.ControlModifier:
                self._grid.step_forward()
            else:
                self._on_next_page()
            return

        # Ctrl+Z → undo
        if key == Qt.Key.Key_Z and mod & Qt.KeyboardModifier.ControlModifier:
            self._on_undo()
            return

        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._grid.cleanup()
        super().closeEvent(event)
