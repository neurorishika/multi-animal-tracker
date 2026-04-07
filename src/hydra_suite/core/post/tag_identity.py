"""
Post-processing tag identity utilities.

Functions to:
1. Assign a majority-vote TagID to each trajectory segment.
2. Detect identity swaps based on tag observations.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _resolve_tag_frame_range(
    tag_cache: Any,
    start_frame: int | None,
    end_frame: int | None,
) -> Tuple[int, int]:
    """Resolve the inclusive frame range to scan from a tag cache."""
    start = int(start_frame) if start_frame is not None else 0
    if end_frame is not None:
        return start, int(end_frame)
    total_frames = (
        int(tag_cache.get_total_frames())
        if hasattr(tag_cache, "get_total_frames")
        else 0
    )
    return start, max(start, total_frames - 1)


def _collect_tag_observations(
    tag_cache: Any,
    start_frame: int,
    end_frame: int,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Collect per-tag observations as ``(frame, x, y)`` tuples."""
    observations: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for frame_idx in range(start_frame, end_frame + 1):
        obs = tag_cache.get_frame(frame_idx)
        tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
        centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
        n_items = min(len(tag_ids), len(centers))
        for item_idx in range(n_items):
            x, y = centers[item_idx]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            observations[int(tag_ids[item_idx])].append((frame_idx, float(x), float(y)))
    return observations


def _dedupe_tag_observations(
    observations: List[Tuple[int, float, float]],
) -> List[Tuple[int, float, float]]:
    """Collapse repeated observations for the same tag/frame by averaging XY."""
    by_frame: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for frame_idx, x, y in observations:
        by_frame[int(frame_idx)].append((float(x), float(y)))

    deduped: List[Tuple[int, float, float]] = []
    for frame_idx in sorted(by_frame):
        coords = by_frame[frame_idx]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        deduped.append((frame_idx, float(np.mean(xs)), float(np.mean(ys))))
    return deduped


def _split_tag_segments(
    observations: List[Tuple[int, float, float]],
    max_gap: int,
) -> List[List[Tuple[int, float, float]]]:
    """Split observations into contiguous segments separated by large gaps."""
    if not observations:
        return []

    segments: List[List[Tuple[int, float, float]]] = []
    current = [observations[0]]
    for obs in observations[1:]:
        if obs[0] - current[-1][0] > max_gap:
            segments.append(current)
            current = [obs]
        else:
            current.append(obs)
    segments.append(current)
    return segments


def _interpolate_segment_rows(
    trajectory_id: int,
    tag_id: int,
    segment: List[Tuple[int, float, float]],
) -> List[Dict[str, Any]]:
    """Expand one observed tag segment into per-frame rows with interpolation."""
    rows: List[Dict[str, Any]] = []
    for idx, (frame_a, x_a, y_a) in enumerate(segment):
        rows.append(
            {
                "TrajectoryID": trajectory_id,
                "FrameID": int(frame_a),
                "X": float(x_a),
                "Y": float(y_a),
                "Theta": np.nan,
                "State": "active",
                "TagID": int(tag_id),
                "Interpolated": False,
            }
        )
        if idx == len(segment) - 1:
            continue

        frame_b, x_b, y_b = segment[idx + 1]
        gap = int(frame_b - frame_a)
        if gap <= 1:
            continue

        for interp_frame in range(frame_a + 1, frame_b):
            alpha = (interp_frame - frame_a) / float(gap)
            rows.append(
                {
                    "TrajectoryID": trajectory_id,
                    "FrameID": int(interp_frame),
                    "X": float(x_a + alpha * (x_b - x_a)),
                    "Y": float(y_a + alpha * (y_b - y_a)),
                    "Theta": np.nan,
                    "State": "active",
                    "TagID": int(tag_id),
                    "Interpolated": True,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# 1.  Majority-vote tag identity per trajectory
# ---------------------------------------------------------------------------


def resolve_tag_identities(
    trajectories_df: pd.DataFrame,
    tag_cache: Any,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """Add ``TagID`` and ``TagVotes`` columns to *trajectories_df*.

    For each trajectory segment the tag observation cache is queried on every
    frame where the trajectory has a valid position.  The tag whose detection
    index is spatially closest to the trajectory's XY is assigned.  A majority
    vote across all frames determines the final ``TagID``.

    Parameters
    ----------
    trajectories_df:
        Standard MAT trajectory DataFrame (columns: ``FrameID``,
        ``TrajectoryID``, ``X``, ``Y``, …).
    tag_cache:
        An open :class:`TagObservationCache` in read mode.
    params:
        Tracking parameters dict.  Relevant keys:

        - ``TAG_ASSOCIATION_RADIUS`` (float, default 50.0) – max distance
          (pixels) between a trajectory position and a tag centre to count
          as an association.

    Returns
    -------
    A *copy* of *trajectories_df* with ``TagID`` (int, -1 = no tag) and
    ``TagVotes`` (int, number of frames that voted for the winning tag).
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if tag_cache is None:
        df = trajectories_df.copy()
        df["TagID"] = -1
        df["TagVotes"] = 0
        return df

    assoc_radius = float(params.get("TAG_ASSOCIATION_RADIUS", 50.0))
    df = trajectories_df.copy()
    df["TagID"] = -1
    df["TagVotes"] = 0

    for traj_id in df["TrajectoryID"].unique():
        mask = df["TrajectoryID"] == traj_id
        traj = df.loc[mask]
        votes: List[int] = []

        for _, row in traj.iterrows():
            fid = int(row["FrameID"])
            x, y = row["X"], row["Y"]
            if pd.isna(x) or pd.isna(y):
                continue
            obs = tag_cache.get_frame(fid)
            tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
            centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
            if len(tag_ids) == 0:
                continue
            # Find the closest tag centre
            dists = np.sqrt(
                (centers[:, 0] - float(x)) ** 2 + (centers[:, 1] - float(y)) ** 2
            )
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= assoc_radius:
                votes.append(int(tag_ids[best_idx]))

        if votes:
            counter = Counter(votes)
            winner, cnt = counter.most_common(1)[0]
            df.loc[mask, "TagID"] = winner
            df.loc[mask, "TagVotes"] = cnt

    return df


# ---------------------------------------------------------------------------
# 2.  Detect tag-based identity swaps
# ---------------------------------------------------------------------------


def _build_frame_trajs(
    trajectories_df: pd.DataFrame,
) -> Dict[int, List[Tuple[int, float, float]]]:
    """Build per-frame trajectory positions lookup."""
    frame_trajs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for _, row in trajectories_df.iterrows():
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            continue
        frame_trajs[int(row["FrameID"])].append(
            (int(row["TrajectoryID"]), float(row["X"]), float(row["Y"]))
        )
    return frame_trajs


def _process_tag_observation(
    tid: int,
    assoc_traj: int,
    tag_last_traj: Dict[int, int],
    tag_streak: Dict[int, int],
    min_streak: int,
    fid: int,
    swaps: List[Dict[str, Any]],
) -> None:
    """Update tag tracking state and record swap if detected."""
    prev = tag_last_traj.get(tid)
    if prev is not None and prev != assoc_traj:
        streak = tag_streak.get(tid, 0)
        if streak >= min_streak:
            confidence = min(1.0, streak / (2.0 * min_streak))
            swaps.append(
                {
                    "frame": fid,
                    "tag_id": tid,
                    "from_traj": prev,
                    "to_traj": assoc_traj,
                    "confidence": confidence,
                }
            )
        tag_streak[tid] = 1
    else:
        tag_streak[tid] = tag_streak.get(tid, 0) + 1

    tag_last_traj[tid] = assoc_traj


def detect_tag_swaps(
    trajectories_df: pd.DataFrame,
    tag_cache: Any,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return a list of suspected identity swap events.

    A swap is reported when a tag that was consistently associated with
    trajectory *A* suddenly appears closest to trajectory *B*.

    Each swap event is a dict::

        {
            "frame": int,
            "tag_id": int,
            "from_traj": int,
            "to_traj": int,
            "confidence": float,   # 0–1
        }
    """
    if trajectories_df is None or trajectories_df.empty or tag_cache is None:
        return []

    assoc_radius = float(params.get("TAG_ASSOCIATION_RADIUS", 50.0))
    min_streak = int(params.get("TAG_SWAP_MIN_STREAK", 3))

    frame_trajs = _build_frame_trajs(trajectories_df)

    tag_last_traj: Dict[int, int] = {}
    tag_streak: Dict[int, int] = {}
    swaps: List[Dict[str, Any]] = []

    for fid in sorted(frame_trajs.keys()):
        obs = tag_cache.get_frame(fid)
        tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
        centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
        if len(tag_ids) == 0:
            continue

        trajs = frame_trajs.get(fid, [])
        if not trajs:
            continue

        traj_ids_arr = np.array([t[0] for t in trajs])
        traj_xy = np.array([[t[1], t[2]] for t in trajs], dtype=np.float32)

        for ti in range(len(tag_ids)):
            tid = int(tag_ids[ti])
            tc = centers[ti]
            dists = np.sqrt(np.sum((traj_xy - tc) ** 2, axis=1))
            best = int(np.argmin(dists))
            if dists[best] > assoc_radius:
                continue
            assoc_traj = int(traj_ids_arr[best])
            _process_tag_observation(
                tid, assoc_traj, tag_last_traj, tag_streak, min_streak, fid, swaps
            )

    return swaps


def build_tag_only_trajectories(
    tag_cache: Any,
    params: Dict[str, Any],
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> pd.DataFrame:
    """Construct simple trajectories directly from tag observations.

    Each unique tag becomes one or more linear-interpolated trajectory segments.
    Segments are broken when the observation gap exceeds ``TAG_ONLY_MAX_GAP``.
    """
    columns = [
        "TrajectoryID",
        "FrameID",
        "X",
        "Y",
        "Theta",
        "State",
        "TagID",
        "Interpolated",
    ]
    if tag_cache is None:
        return pd.DataFrame(columns=columns)

    min_obs = int(params.get("TAG_ONLY_MIN_OBS", 3))
    max_gap = max(1, int(params.get("TAG_ONLY_MAX_GAP", 5)))
    frame_start, frame_end = _resolve_tag_frame_range(tag_cache, start_frame, end_frame)
    if frame_end < frame_start:
        return pd.DataFrame(columns=columns)

    observations_by_tag = _collect_tag_observations(tag_cache, frame_start, frame_end)
    rows: List[Dict[str, Any]] = []
    next_trajectory_id = 0

    for tag_id in sorted(observations_by_tag):
        observations = _dedupe_tag_observations(observations_by_tag[tag_id])
        if len(observations) < min_obs:
            continue
        for segment in _split_tag_segments(observations, max_gap=max_gap):
            if len(segment) < min_obs:
                continue
            rows.extend(_interpolate_segment_rows(next_trajectory_id, tag_id, segment))
            next_trajectory_id += 1

    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows)
    result = result.sort_values(["TrajectoryID", "FrameID"]).reset_index(drop=True)
    return result
