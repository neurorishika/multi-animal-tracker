"""
Per-frame tag feature helpers for the tracking loop.

These functions translate tag observations from the :class:`TagObservationCache`
into the ``association_data`` dictionaries consumed by the assigner.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for "no tag observed"
NO_TAG: int = -1


def build_tag_detection_map(
    tag_cache: Any,
    frame_idx: int,
) -> Dict[int, int]:
    """Map detection-slot index → tag_id for a single frame.

    Parameters
    ----------
    tag_cache:
        An open :class:`TagObservationCache` in read mode (or *None*).
    frame_idx:
        The video frame index to query.

    Returns
    -------
    dict mapping ``det_index`` → ``tag_id`` for every tag observed in this
    frame.  Empty dict if the cache is *None* or the frame has no tags.
    """
    if tag_cache is None:
        return {}
    try:
        obs = tag_cache.get_frame(frame_idx)
    except Exception:
        return {}

    tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
    det_indices = obs.get("det_indices", np.array([], dtype=np.int32))

    if len(tag_ids) == 0:
        return {}

    result: Dict[int, int] = {}
    for tid, didx in zip(tag_ids.tolist(), det_indices.tolist()):
        # If multiple tags land in the same detection slot, keep the first.
        if didx not in result:
            result[didx] = int(tid)
    return result


def build_detection_tag_id_list(
    tag_det_map: Dict[int, int],
    num_detections: int,
) -> List[int]:
    """Return a list aligned with detections: ``tag_id`` or :data:`NO_TAG`.

    This is what goes into ``association_data["detection_tag_ids"]``.
    """
    return [tag_det_map.get(j, NO_TAG) for j in range(num_detections)]


# ---------------------------------------------------------------------------
# Track-side tag history
# ---------------------------------------------------------------------------


class TrackTagHistory:
    """Maintains a recent-window tag-ID history for every track slot.

    Tracks accumulate tag observations over a rolling window of *n_frames*.
    The "current" tag for a track is the majority-vote winner inside the window
    (or :data:`NO_TAG` if no observations are available).
    """

    def __init__(self, n_tracks: int, window: int = 30):
        self._window = max(1, window)
        # _history[track_idx] is a list of (frame_idx, tag_id) pairs, newest last
        self._history: List[List[tuple]] = [[] for _ in range(n_tracks)]

    @property
    def n_tracks(self) -> int:
        return len(self._history)

    def resize(self, n_tracks: int) -> None:
        """Grow (never shrink) the history to accommodate *n_tracks*."""
        while len(self._history) < n_tracks:
            self._history.append([])

    def record(self, track_idx: int, frame_idx: int, tag_id: int) -> None:
        """Record (or update) the tag observation for a track on this frame."""
        if tag_id == NO_TAG:
            return
        if track_idx >= len(self._history):
            self.resize(track_idx + 1)
        self._history[track_idx].append((frame_idx, tag_id))
        # Trim old entries beyond the window
        cutoff = frame_idx - self._window
        hist = self._history[track_idx]
        while hist and hist[0][0] < cutoff:
            hist.pop(0)

    def majority_tag(self, track_idx: int) -> int:
        """Return the majority-vote tag for *track_idx*, or :data:`NO_TAG`."""
        if track_idx >= len(self._history):
            return NO_TAG
        hist = self._history[track_idx]
        if not hist:
            return NO_TAG
        counts = Counter(tid for _, tid in hist)
        winner, cnt = counts.most_common(1)[0]
        return int(winner)

    def build_track_tag_id_list(self, n_tracks: int) -> List[int]:
        """Return a list of length *n_tracks*: majority tag per slot.

        This is what goes into ``association_data["track_last_tag_ids"]``.
        """
        self.resize(n_tracks)
        return [self.majority_tag(i) for i in range(n_tracks)]

    def clear_track(self, track_idx: int) -> None:
        """Reset history for a track slot (e.g. on respawn / lost)."""
        if track_idx < len(self._history):
            self._history[track_idx].clear()
