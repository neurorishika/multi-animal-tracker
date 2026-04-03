"""Utilities for exporting pose properties into wide trajectory CSV outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .cache import IndividualPropertiesCache

POSE_SUMMARY_COLUMNS = [
    "PoseMeanConf",
    "PoseValidFraction",
    "PoseNumValid",
    "PoseNumKeypoints",
]


def _pose_value_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("Pose")]


def _sanitize_keypoint_name(name: str, idx: int) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    if not token:
        token = f"kp{idx:03d}"
    return token


def build_pose_keypoint_labels(
    keypoint_names: Optional[Sequence[str]], max_keypoints: int
) -> List[str]:
    names = list(keypoint_names or [])
    labels: List[str] = []
    used = set()
    for idx, name in enumerate(names):
        base = _sanitize_keypoint_name(name, idx)
        label = base
        if label in used:
            label = f"{base}_{idx:03d}"
        used.add(label)
        labels.append(label)

    total = max(int(max_keypoints), len(labels))
    for idx in range(len(labels), total):
        label = f"kp{idx:03d}"
        while label in used:
            label = f"{label}_{idx:03d}"
        used.add(label)
        labels.append(label)
    return labels


def pose_wide_columns_for_labels(labels: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for label in labels:
        cols.extend(
            [
                f"PoseKpt_{label}_X",
                f"PoseKpt_{label}_Y",
                f"PoseKpt_{label}_Conf",
            ]
        )
    return cols


def _parse_ignore_keypoint_tokens(ignore_keypoints: Any) -> List[Any]:
    if ignore_keypoints is None:
        return []
    if isinstance(ignore_keypoints, str):
        out: List[Any] = []
        for token in ignore_keypoints.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                out.append(t)
        return out
    if isinstance(ignore_keypoints, (list, tuple, set)):
        out: List[Any] = []
        for value in ignore_keypoints:
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            try:
                out.append(int(text))
            except ValueError:
                out.append(text)
        return out
    return []


def _resolve_ignored_keypoint_indices(
    ignore_keypoints: Any, keypoint_names: Optional[Sequence[str]]
) -> Set[int]:
    tokens = _parse_ignore_keypoint_tokens(ignore_keypoints)
    if not tokens:
        return set()
    names = [str(v) for v in (keypoint_names or [])]
    name_to_idx = {name.lower(): idx for idx, name in enumerate(names)}
    ignore_idxs: Set[int] = set()
    for token in tokens:
        if isinstance(token, int):
            ignore_idxs.add(int(token))
            continue
        idx = name_to_idx.get(str(token).strip().lower())
        if idx is not None:
            ignore_idxs.add(int(idx))
    return ignore_idxs


def _apply_ignore_to_keypoints(
    keypoints: np.ndarray, ignore_indices: Set[int]
) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    if not ignore_indices:
        return arr
    keep = [i for i in range(len(arr)) if i not in ignore_indices]
    if not keep:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(arr[keep], dtype=np.float32)


def _filter_labels_for_ignore(
    labels: Sequence[str],
    keypoint_names: Optional[Sequence[str]],
    ignore_indices: Set[int],
) -> Tuple[List[str], Set[int]]:
    if not labels:
        return [], set()
    if not ignore_indices:
        return list(labels), set()

    # Indices align with pose keypoint index order.
    dropped_label_indices = {
        idx for idx in ignore_indices if 0 <= int(idx) < len(labels)
    }
    filtered = [
        label for idx, label in enumerate(labels) if idx not in dropped_label_indices
    ]
    return filtered, dropped_label_indices


def flatten_pose_keypoints_row(
    keypoints: Any, labels: Sequence[str]
) -> Dict[str, float]:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        arr = np.zeros((0, 3), dtype=np.float32)

    row: Dict[str, float] = {}
    for idx, label in enumerate(labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        if idx < len(arr):
            row[x_col] = float(arr[idx, 0])
            row[y_col] = float(arr[idx, 1])
            row[c_col] = float(np.clip(arr[idx, 2], 0.0, 1.0))
        else:
            row[x_col] = np.nan
            row[y_col] = np.nan
            row[c_col] = np.nan
    return row


def build_pose_lookup_dataframe(
    cache: IndividualPropertiesCache,
    keypoint_names: Optional[Sequence[str]] = None,
    ignore_keypoints: Any = None,
    min_valid_conf: float = 0.2,
) -> pd.DataFrame:
    """Flatten cache entries into frame+detection keyed wide pose rows.

    Args:
        cache: Cache to read from
        keypoint_names: Optional keypoint names for labeling
        ignore_keypoints: Keypoints to ignore
        min_valid_conf: Minimum confidence threshold for keypoint validity
    """
    entries: List[Dict[str, Any]] = []
    max_keypoints = 0
    ignore_indices = _resolve_ignored_keypoint_indices(ignore_keypoints, keypoint_names)

    for frame_idx in cache.get_cached_frames():
        # Pass min_valid_conf to compute stats on-demand
        frame = cache.get_frame(int(frame_idx), min_valid_conf=min_valid_conf)
        ids = frame.get("detection_ids", [])
        mean_conf = frame.get("pose_mean_conf", [])
        valid_fraction = frame.get("pose_valid_fraction", [])
        num_valid = frame.get("pose_num_valid", [])
        num_keypoints = frame.get("pose_num_keypoints", [])
        keypoints = frame.get("pose_keypoints", [])
        count = min(
            len(ids),
            len(mean_conf),
            len(valid_fraction),
            len(num_valid),
            len(num_keypoints),
            len(keypoints),
        )
        for idx in range(count):
            try:
                det_id_int = int(ids[idx])
            except Exception:
                continue
            kpts = np.asarray(keypoints[idx], dtype=np.float32)
            kpts = _apply_ignore_to_keypoints(kpts, ignore_indices)
            max_keypoints = max(max_keypoints, int(len(kpts)))
            entries.append(
                {
                    "_pose_frame_id": int(frame_idx),
                    "_pose_detection_id": det_id_int,
                    "PoseMeanConf": float(np.clip(mean_conf[idx], 0.0, 1.0)),
                    "PoseValidFraction": float(np.clip(valid_fraction[idx], 0.0, 1.0)),
                    "PoseNumValid": int(num_valid[idx]),
                    "PoseNumKeypoints": int(num_keypoints[idx]),
                    "_pose_keypoints": kpts,
                }
            )

    labels = build_pose_keypoint_labels(keypoint_names, max_keypoints)
    labels, _ = _filter_labels_for_ignore(labels, keypoint_names, ignore_indices)
    keypoint_cols = pose_wide_columns_for_labels(labels)
    if not entries:
        return pd.DataFrame(
            columns=[
                "_pose_frame_id",
                "_pose_detection_id",
                *POSE_SUMMARY_COLUMNS,
                *keypoint_cols,
            ]
        )

    rows: List[Dict[str, Any]] = []
    for entry in entries:
        row = {
            "_pose_frame_id": entry["_pose_frame_id"],
            "_pose_detection_id": entry["_pose_detection_id"],
            "PoseMeanConf": entry["PoseMeanConf"],
            "PoseValidFraction": entry["PoseValidFraction"],
            "PoseNumValid": entry["PoseNumValid"],
            "PoseNumKeypoints": entry["PoseNumKeypoints"],
        }
        row.update(flatten_pose_keypoints_row(entry["_pose_keypoints"], labels))
        rows.append(row)

    return pd.DataFrame(rows)


def augment_trajectories_with_pose_df(
    trajectories_df: pd.DataFrame,
    pose_lookup_df: pd.DataFrame,
    coordinate_scale: float = 1.0,
) -> pd.DataFrame:
    """Merge pose properties into trajectory rows using (FrameID, DetectionID).

    Args:
        trajectories_df: Trajectories dataframe to augment.
        pose_lookup_df: Wide pose lookup dataframe from build_pose_lookup_dataframe.
        coordinate_scale: Scale factor applied to all PoseKpt_*_X and PoseKpt_*_Y
            columns after merging.  Set to ``1.0 / RESIZE_FACTOR`` when the pose
            cache was built on down-scaled frames but the trajectory CSV already
            contains original-resolution coordinates.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        "FrameID" not in trajectories_df.columns
        or "DetectionID" not in trajectories_df.columns
    ):
        return trajectories_df.copy()

    out = trajectories_df.copy()
    out = out.drop(columns=_pose_value_columns(out), errors="ignore")

    pose_cols = (
        _pose_value_columns(pose_lookup_df) if pose_lookup_df is not None else []
    )
    if pose_lookup_df is None or pose_lookup_df.empty:
        for col in POSE_SUMMARY_COLUMNS:
            out[col] = np.nan
        return out

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_detection_join"] = (
        pd.to_numeric(out["DetectionID"], errors="coerce").round().astype("Int64")
    )

    lookup = pose_lookup_df.copy()
    lookup["_pose_frame_id"] = (
        pd.to_numeric(lookup["_pose_frame_id"], errors="coerce").round().astype("Int64")
    )
    lookup["_pose_detection_id"] = (
        pd.to_numeric(lookup["_pose_detection_id"], errors="coerce")
        .round()
        .astype("Int64")
    )

    merged = out.merge(
        lookup[["_pose_frame_id", "_pose_detection_id", *pose_cols]],
        how="left",
        left_on=["_frame_join", "_detection_join"],
        right_on=["_pose_frame_id", "_pose_detection_id"],
        sort=False,
    )
    merged.drop(
        columns=[
            "_frame_join",
            "_detection_join",
            "_pose_frame_id",
            "_pose_detection_id",
        ],
        inplace=True,
        errors="ignore",
    )
    for col in POSE_SUMMARY_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Rescale keypoint X/Y when the cache was built on down-scaled frames
    if coordinate_scale != 1.0 and coordinate_scale > 0.0:
        for col in merged.columns:
            col_str = str(col)
            if col_str.startswith("PoseKpt_") and (
                col_str.endswith("_X") or col_str.endswith("_Y")
            ):
                merged[col] = (
                    pd.to_numeric(merged[col], errors="coerce") * coordinate_scale
                )

    return merged


def augment_trajectories_with_pose_cache(
    trajectories_df: pd.DataFrame,
    cache_path: str,
    ignore_keypoints: Any = None,
    min_valid_conf: float = 0.2,
    coordinate_scale: float = 1.0,
) -> pd.DataFrame:
    """Load properties cache and merge wide pose columns into trajectory rows.

    Args:
        trajectories_df: Trajectories dataframe to augment
        cache_path: Path to individual properties cache
        ignore_keypoints: Keypoints to ignore
        min_valid_conf: Minimum confidence threshold for keypoint validity
        coordinate_scale: Scale factor applied to PoseKpt_*_X / _Y columns.
            Set to ``1.0 / RESIZE_FACTOR`` when the cache was built on
            down-scaled frames.
    """
    cache = IndividualPropertiesCache(cache_path, mode="r")
    try:
        if not cache.is_compatible():
            raise RuntimeError(
                f"Incompatible individual-properties cache: {cache_path}"
            )
        names = cache.metadata.get("pose_keypoint_names", [])
        if not isinstance(names, (list, tuple)):
            names = []
        lookup = build_pose_lookup_dataframe(
            cache,
            keypoint_names=names,
            ignore_keypoints=ignore_keypoints,
            min_valid_conf=min_valid_conf,
        )
    finally:
        cache.close()
    return augment_trajectories_with_pose_df(
        trajectories_df, lookup, coordinate_scale=coordinate_scale
    )


def _ensure_pose_columns(
    df: pd.DataFrame, extra_cols: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    out = df.copy()
    for col in list(POSE_SUMMARY_COLUMNS) + list(extra_cols or []):
        if col.startswith("Pose") and col not in out.columns:
            out[col] = np.nan
    return out


def merge_interpolated_pose_df(
    trajectories_df: pd.DataFrame, interpolated_pose_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Fill missing pose rows from interpolated analysis keyed by (FrameID, TrajectoryID).

    Existing detection-keyed pose values are preserved.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        interpolated_pose_df is None
        or interpolated_pose_df.empty
        or "FrameID" not in trajectories_df.columns
        or "TrajectoryID" not in trajectories_df.columns
    ):
        return _ensure_pose_columns(trajectories_df)

    pose_cols_interp = [
        c for c in interpolated_pose_df.columns if str(c).startswith("Pose")
    ]
    if not pose_cols_interp or not {"frame_id", "trajectory_id"}.issubset(
        interpolated_pose_df.columns
    ):
        return _ensure_pose_columns(trajectories_df)

    out = _ensure_pose_columns(trajectories_df, extra_cols=pose_cols_interp)
    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )

    interp = interpolated_pose_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )
    interp_lookup = interp[
        ["_frame_join", "_traj_join", *pose_cols_interp]
    ].drop_duplicates(subset=["_frame_join", "_traj_join"], keep="first")

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_interp"),
        sort=False,
    )
    for col in pose_cols_interp:
        interp_col = f"{col}_interp"
        if interp_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged[col].where(merged[col].notna(), merged[interp_col])
    merged.drop(
        columns=[
            "_frame_join",
            "_traj_join",
            *[f"{c}_interp" for c in pose_cols_interp],
        ],
        inplace=True,
        errors="ignore",
    )
    return _ensure_pose_columns(merged)


# ---------------------------------------------------------------------------
# AprilTag interpolated merge
# ---------------------------------------------------------------------------

APRILTAG_INTERP_COLUMNS = ["InterpTagID", "InterpTagHamming", "InterpTagConf"]


def merge_interpolated_apriltag_df(
    trajectories_df: pd.DataFrame,
    interp_tag_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated AprilTag observations into final trajectories.

    Existing ``TagID`` values (from pre-tracking) are preserved.
    Interpolated rows receive ``InterpTagID`` / ``InterpTagHamming``.
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        interp_tag_df is None
        or interp_tag_df.empty
        or "FrameID" not in trajectories_df.columns
        or "TrajectoryID" not in trajectories_df.columns
    ):
        return trajectories_df

    needed = {"frame_id", "trajectory_id", "tag_id"}
    if not needed.issubset(interp_tag_df.columns):
        return trajectories_df

    out = trajectories_df.copy()
    for col in APRILTAG_INTERP_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )

    interp = interp_tag_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )

    tag_cols = ["tag_id"]
    if "hamming" in interp.columns:
        tag_cols.append("hamming")
    if "confidence" in interp.columns:
        tag_cols.append("confidence")

    interp_lookup = interp[["_frame_join", "_traj_join", *tag_cols]].drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_itag"),
        sort=False,
    )

    if "tag_id" in merged.columns:
        merged["InterpTagID"] = merged["InterpTagID"].where(
            merged["InterpTagID"].notna(), merged["tag_id"]
        )
        merged.drop(columns=["tag_id"], inplace=True, errors="ignore")
    if "hamming" in merged.columns:
        merged["InterpTagHamming"] = merged["InterpTagHamming"].where(
            merged["InterpTagHamming"].notna(), merged["hamming"]
        )
        merged.drop(columns=["hamming"], inplace=True, errors="ignore")
    if "confidence" in merged.columns:
        merged["InterpTagConf"] = merged["InterpTagConf"].where(
            merged["InterpTagConf"].notna(), merged["confidence"]
        )
        merged.drop(columns=["confidence"], inplace=True, errors="ignore")

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged


# ---------------------------------------------------------------------------
# CNN identity interpolated merge
# ---------------------------------------------------------------------------


def merge_interpolated_cnn_df(
    trajectories_df: pd.DataFrame,
    interp_cnn_df: Optional[pd.DataFrame],
    label: str = "cnn_identity",
) -> pd.DataFrame:
    """Merge interpolated CNN identity predictions into final trajectories.

    Each classifier label gets its own column pair: ``CNN_{label}_Class``,
    ``CNN_{label}_Conf``.
    """
    col_class = f"CNN_{label}_Class"
    col_conf = f"CNN_{label}_Conf"

    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        interp_cnn_df is None
        or interp_cnn_df.empty
        or "FrameID" not in trajectories_df.columns
        or "TrajectoryID" not in trajectories_df.columns
    ):
        out = trajectories_df.copy()
        for c in (col_class, col_conf):
            if c not in out.columns:
                out[c] = np.nan
        return out

    needed = {"frame_id", "trajectory_id", "class_name", "confidence"}
    if not needed.issubset(interp_cnn_df.columns):
        out = trajectories_df.copy()
        for c in (col_class, col_conf):
            if c not in out.columns:
                out[c] = np.nan
        return out

    out = trajectories_df.copy()
    for c in (col_class, col_conf):
        if c not in out.columns:
            out[c] = np.nan

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )

    interp = interp_cnn_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )

    interp_lookup = interp[
        ["_frame_join", "_traj_join", "class_name", "confidence"]
    ].drop_duplicates(subset=["_frame_join", "_traj_join"], keep="first")

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_icnn"),
        sort=False,
    )

    if "class_name" in merged.columns:
        merged[col_class] = merged[col_class].where(
            merged[col_class].notna(), merged["class_name"]
        )
        merged.drop(columns=["class_name"], inplace=True, errors="ignore")
    if "confidence" in merged.columns:
        merged[col_conf] = merged[col_conf].where(
            merged[col_conf].notna(), merged["confidence"]
        )
        merged.drop(columns=["confidence"], inplace=True, errors="ignore")

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged


# ---------------------------------------------------------------------------
# Head-tail interpolated merge
# ---------------------------------------------------------------------------

HEADTAIL_INTERP_COLUMNS = [
    "InterpHeadingRad",
    "InterpHeadingConf",
    "InterpHeadingDirected",
]


def merge_interpolated_headtail_df(
    trajectories_df: pd.DataFrame,
    interp_ht_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge interpolated head-tail direction into final trajectories."""
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df
    if (
        interp_ht_df is None
        or interp_ht_df.empty
        or "FrameID" not in trajectories_df.columns
        or "TrajectoryID" not in trajectories_df.columns
    ):
        out = trajectories_df.copy()
        for c in HEADTAIL_INTERP_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
        return out

    needed = {"frame_id", "trajectory_id", "heading_rad"}
    if not needed.issubset(interp_ht_df.columns):
        out = trajectories_df.copy()
        for c in HEADTAIL_INTERP_COLUMNS:
            if c not in out.columns:
                out[c] = np.nan
        return out

    out = trajectories_df.copy()
    for c in HEADTAIL_INTERP_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan

    out["_frame_join"] = (
        pd.to_numeric(out["FrameID"], errors="coerce").round().astype("Int64")
    )
    out["_traj_join"] = (
        pd.to_numeric(out["TrajectoryID"], errors="coerce").round().astype("Int64")
    )

    interp = interp_ht_df.copy()
    interp["_frame_join"] = (
        pd.to_numeric(interp["frame_id"], errors="coerce").round().astype("Int64")
    )
    interp["_traj_join"] = (
        pd.to_numeric(interp["trajectory_id"], errors="coerce").round().astype("Int64")
    )

    ht_cols = ["heading_rad"]
    if "heading_conf" in interp.columns:
        ht_cols.append("heading_conf")
    if "heading_directed" in interp.columns:
        ht_cols.append("heading_directed")

    interp_lookup = interp[["_frame_join", "_traj_join", *ht_cols]].drop_duplicates(
        subset=["_frame_join", "_traj_join"], keep="first"
    )

    merged = out.merge(
        interp_lookup,
        how="left",
        on=["_frame_join", "_traj_join"],
        suffixes=("", "_iht"),
        sort=False,
    )

    if "heading_rad" in merged.columns:
        merged["InterpHeadingRad"] = merged["InterpHeadingRad"].where(
            merged["InterpHeadingRad"].notna(), merged["heading_rad"]
        )
        merged.drop(columns=["heading_rad"], inplace=True, errors="ignore")
    if "heading_conf" in merged.columns:
        merged["InterpHeadingConf"] = merged["InterpHeadingConf"].where(
            merged["InterpHeadingConf"].notna(), merged["heading_conf"]
        )
        merged.drop(columns=["heading_conf"], inplace=True, errors="ignore")
    if "heading_directed" in merged.columns:
        merged["InterpHeadingDirected"] = merged["InterpHeadingDirected"].where(
            merged["InterpHeadingDirected"].notna(), merged["heading_directed"]
        )
        merged.drop(columns=["heading_directed"], inplace=True, errors="ignore")

    merged.drop(
        columns=["_frame_join", "_traj_join"],
        inplace=True,
        errors="ignore",
    )
    return merged
