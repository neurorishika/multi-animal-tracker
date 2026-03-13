"""
Post-processing tag identity utilities.

Functions to:
1. Assign a majority-vote TagID to each trajectory segment.
2. Detect identity swaps based on tag observations.
3. Build tag-only trajectories from tag observations + interpolation
   (standalone alternative to YOLO+Kalman tracking).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


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

    # Build per-frame trajectory positions lookup
    # {frame_id: [(traj_id, x, y), …]}
    frame_trajs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    for _, row in trajectories_df.iterrows():
        if pd.isna(row["X"]) or pd.isna(row["Y"]):
            continue
        frame_trajs[int(row["FrameID"])].append(
            (int(row["TrajectoryID"]), float(row["X"]), float(row["Y"]))
        )

    # Track which traj each tag was last associated with
    tag_last_traj: Dict[int, int] = {}
    tag_streak: Dict[int, int] = {}  # how many consecutive frames
    swaps: List[Dict[str, Any]] = []

    frames_sorted = sorted(frame_trajs.keys())
    for fid in frames_sorted:
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

    return swaps


# ---------------------------------------------------------------------------
# 3.  Tag-only trajectories (standalone mode)
# ---------------------------------------------------------------------------


def build_tag_only_trajectories(
    tag_cache: Any,
    params: Dict[str, Any],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Build trajectory DataFrame purely from tag observations + interpolation.

    Each unique ``tag_id`` becomes its own trajectory.  Gaps up to
    ``TAG_ONLY_MAX_GAP`` frames are interpolated linearly.

    Parameters
    ----------
    tag_cache:
        Open :class:`TagObservationCache` in read mode.
    params:
        Dict with optional keys:

        - ``TAG_ONLY_MAX_GAP`` (int, default 30)
        - ``TAG_ONLY_MIN_OBS`` (int, default 5) – min observations to keep

    Returns
    -------
    DataFrame with columns ``FrameID``, ``TrajectoryID``, ``X``, ``Y``,
    ``Theta``, ``TagID``, ``State``.
    """
    if tag_cache is None:
        return pd.DataFrame(
            columns=["FrameID", "TrajectoryID", "X", "Y", "Theta", "TagID", "State"]
        )

    max_gap = int(params.get("TAG_ONLY_MAX_GAP", 30))
    min_obs = int(params.get("TAG_ONLY_MIN_OBS", 5))

    ef = end_frame if end_frame is not None else tag_cache.get_total_frames() - 1

    # Collect all observations: tag_id → [(frame, x, y)]
    tag_obs: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)

    for fid in range(start_frame, ef + 1):
        try:
            obs = tag_cache.get_frame(fid)
        except Exception:
            continue
        tag_ids = obs.get("tag_ids", np.array([], dtype=np.int32))
        centers = obs.get("centers_xy", np.zeros((0, 2), dtype=np.float32))
        for i in range(len(tag_ids)):
            tid = int(tag_ids[i])
            tag_obs[tid].append((fid, float(centers[i, 0]), float(centers[i, 1])))

    rows: List[Dict[str, Any]] = []
    traj_id = 0

    for tid in sorted(tag_obs.keys()):
        obs_list = sorted(tag_obs[tid], key=lambda t: t[0])
        if len(obs_list) < min_obs:
            continue

        # Split into segments at gaps > max_gap
        segments: List[List[Tuple[int, float, float]]] = []
        current_seg: List[Tuple[int, float, float]] = [obs_list[0]]
        for k in range(1, len(obs_list)):
            if obs_list[k][0] - obs_list[k - 1][0] > max_gap:
                segments.append(current_seg)
                current_seg = []
            current_seg.append(obs_list[k])
        segments.append(current_seg)

        for seg in segments:
            if len(seg) < min_obs:
                continue

            frames = np.array([s[0] for s in seg], dtype=np.int64)
            xs = np.array([s[1] for s in seg], dtype=np.float64)
            ys = np.array([s[2] for s in seg], dtype=np.float64)

            f_min, f_max = int(frames[0]), int(frames[-1])
            all_frames = np.arange(f_min, f_max + 1)

            # Interpolate
            if len(frames) > 1:
                interp_x = interp1d(frames, xs, kind="linear", fill_value="extrapolate")
                interp_y = interp1d(frames, ys, kind="linear", fill_value="extrapolate")
                all_x = interp_x(all_frames)
                all_y = interp_y(all_frames)
            else:
                all_x = np.full_like(all_frames, xs[0], dtype=np.float64)
                all_y = np.full_like(all_frames, ys[0], dtype=np.float64)

            # Compute heading from consecutive positions
            dx = np.diff(all_x, prepend=all_x[0])
            dy = np.diff(all_y, prepend=all_y[0])
            theta = np.arctan2(dy, dx)

            observed_set = set(frames.tolist())

            for i, f in enumerate(all_frames):
                rows.append(
                    {
                        "FrameID": int(f),
                        "TrajectoryID": traj_id,
                        "X": float(all_x[i]),
                        "Y": float(all_y[i]),
                        "Theta": float(theta[i]),
                        "TagID": tid,
                        "State": "active" if int(f) in observed_set else "interpolated",
                    }
                )

            traj_id += 1

    return pd.DataFrame(rows)
