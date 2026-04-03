"""
Pose quality assessment and post-processing for the HYDRA suite.

All functions are pure (stateless) and can be imported from both the GUI
and any script-based pipeline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from hydra_suite.core.identity.pose.features import compute_pose_geometry_from_keypoints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PoseQualityResult:
    """Result of a single-row pose quality assessment."""

    cleaned_keypoints: np.ndarray  # [K, 3] float32; bad kpts have conf=0, X/Y kept
    valid_mask: np.ndarray  # [K] bool
    quality_score: float  # [0, 1]
    quality_state: str  # "good" | "partial" | "bad" | "rejected"
    quality_flags: List[str]  # e.g. ["low_conf:3", "too_few_valid"]
    was_cleaned: bool


@dataclass
class BodyLengthPrior:
    """Statistics-based body-length calibration prior."""

    median_px: float
    mad_px: float  # median absolute deviation
    n_samples: int
    is_valid: bool  # True only when n_samples >= 20


@dataclass
class EdgeLengthPriors:
    """Per-skeleton-edge length calibration priors.

    ``priors`` maps a canonical edge key ``(min_idx, max_idx)`` to a dict with
    keys ``median_px``, ``mad_px``, and ``n_samples``.  ``is_valid`` is True
    when at least one edge has been calibrated with >= 20 samples.
    """

    priors: Dict[Tuple[int, int], Dict]
    is_valid: bool


# ---------------------------------------------------------------------------
# assess_pose_row
# ---------------------------------------------------------------------------


def assess_pose_row(
    keypoints: np.ndarray,
    min_valid_conf: float,
    min_valid_fraction: float,
    min_valid_keypoints: int,
    ignore_indices: Optional[List[int]] = None,
    body_length_prior: Optional[BodyLengthPrior] = None,
    anterior_indices: Optional[List[int]] = None,
    posterior_indices: Optional[List[int]] = None,
    body_length_z_threshold: float = 3.5,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    edge_length_priors: Optional[EdgeLengthPriors] = None,
    edge_length_z_threshold: float = 4.0,
) -> PoseQualityResult:
    """Assess pose quality for a single keypoint array.

    Cleans raw pose keypoints by zeroing confidence for bad keypoints while
    retaining X/Y coordinates as a best guess.  Never drops rows — rejected
    rows get all conf=0 plus quality flags.

    Args:
        keypoints: [K, 3] float32 array of (x, y, conf) per keypoint.
        min_valid_conf: Minimum confidence threshold to consider a keypoint valid.
        min_valid_fraction: Minimum fraction of (considered) keypoints that must
            be valid for the row to be accepted.
        min_valid_keypoints: Minimum absolute count of valid keypoints required.
        ignore_indices: Keypoint indices excluded from all quality computations.
        body_length_prior: Optional prior for body-length outlier detection.
        anterior_indices: Keypoint indices defining the anterior region.
        posterior_indices: Keypoint indices defining the posterior region.
        body_length_z_threshold: Z-score threshold above which body length is
            flagged as an outlier.
        skeleton_edges: List of (i, j) index pairs defining connected keypoints.
        edge_length_priors: Calibrated per-edge length priors for anatomy checks.
        edge_length_z_threshold: MAD-based z-score above which an edge length is
            flagged as an anatomical outlier.

    Returns:
        PoseQualityResult with cleaned keypoints and quality metadata.
    """
    K_dummy = 0
    ignore_set = {int(i) for i in (ignore_indices or [])}

    def _rejected_result(n_kpts: int, flags: List[str]) -> PoseQualityResult:
        kpts_out = (
            np.zeros((n_kpts, 3), dtype=np.float32)
            if n_kpts > 0
            else np.zeros((0, 3), dtype=np.float32)
        )
        mask = np.zeros(n_kpts, dtype=bool)
        return PoseQualityResult(
            cleaned_keypoints=kpts_out,
            valid_mask=mask,
            quality_score=0.0,
            quality_state="rejected",
            quality_flags=flags,
            was_cleaned=True,
        )

    # ------------------------------------------------------------------
    # 1. Validate input
    # ------------------------------------------------------------------
    if keypoints is None:
        return _rejected_result(K_dummy, ["null_input"])

    try:
        arr = np.asarray(keypoints, dtype=np.float32)
    except Exception:
        return _rejected_result(K_dummy, ["invalid_input"])

    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return _rejected_result(K_dummy, ["invalid_shape"])

    # All-NaN check
    if not np.any(np.isfinite(arr)):
        return _rejected_result(len(arr), ["all_nan"])

    K = len(arr)
    cleaned = arr.copy()
    valid_mask = np.zeros(K, dtype=bool)
    flags: List[str] = []
    was_cleaned = False

    low_conf_count = 0
    invalid_coords_count = 0

    # ------------------------------------------------------------------
    # 2. Per-keypoint cleaning
    # ------------------------------------------------------------------
    for i in range(K):
        if i in ignore_set:
            # Leave ignored keypoints untouched; do not mark valid or invalid
            continue
        x, y, conf = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])
        coords_ok = np.isfinite(x) and np.isfinite(y)
        conf_ok = np.isfinite(conf) and conf >= float(min_valid_conf)

        if not conf_ok:
            cleaned[i, 2] = 0.0
            was_cleaned = True
            low_conf_count += 1
            # valid_mask[i] remains False
        elif not coords_ok:
            cleaned[i, 2] = 0.0
            was_cleaned = True
            invalid_coords_count += 1
            # valid_mask[i] remains False
        else:
            valid_mask[i] = True

    if low_conf_count > 0:
        flags.append(f"low_conf:{low_conf_count}")
    if invalid_coords_count > 0:
        flags.append(f"invalid_coords:{invalid_coords_count}")

    # ------------------------------------------------------------------
    # 3-4. Count valid keypoints (ignoring ignored indices in denominator)
    # ------------------------------------------------------------------
    num_considered = sum(1 for i in range(K) if i not in ignore_set)
    num_valid = int(np.sum(valid_mask))
    valid_fraction = (
        float(num_valid) / float(num_considered) if num_considered > 0 else 0.0
    )

    # ------------------------------------------------------------------
    # 5. Rejection check
    # ------------------------------------------------------------------
    rejected = False
    if valid_fraction < float(min_valid_fraction) or num_valid < int(
        min_valid_keypoints
    ):
        rejected = True
        flags.append("too_few_valid")
        # Zero all confs in output
        for i in range(K):
            if i not in ignore_set:
                cleaned[i, 2] = 0.0
        was_cleaned = True

    # ------------------------------------------------------------------
    # 6. Body-length outlier check (on ORIGINAL keypoints before zeroing)
    # ------------------------------------------------------------------
    body_length_outlier = False
    if (
        not rejected
        and body_length_prior is not None
        and body_length_prior.is_valid
        and anterior_indices
        and posterior_indices
    ):
        geom = compute_pose_geometry_from_keypoints(
            arr,  # original, before any zeroing
            anterior_indices,
            posterior_indices,
            min_valid_conf,
            list(ignore_set) if ignore_set else None,
        )
        if geom is not None and geom.get("body_length") is not None:
            bl = float(geom["body_length"])
            denom = max(float(body_length_prior.mad_px), 1.0)
            z_score = abs(bl - float(body_length_prior.median_px)) / denom
            if z_score > float(body_length_z_threshold):
                flags.append("body_length_outlier")
                body_length_outlier = True

    # ------------------------------------------------------------------
    # 6b. Per-edge skeleton length check
    # ------------------------------------------------------------------
    n_edge_outliers = 0
    if (
        not rejected
        and skeleton_edges
        and edge_length_priors is not None
        and edge_length_priors.is_valid
    ):
        for edge in skeleton_edges:
            try:
                ei, ej = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if ei >= K or ej >= K:
                continue
            if not (valid_mask[ei] and valid_mask[ej]):
                continue
            key = (min(ei, ej), max(ei, ej))
            prior = edge_length_priors.priors.get(key)
            if prior is None or int(prior.get("n_samples", 0)) < 20:
                continue
            dx = float(cleaned[ei, 0]) - float(cleaned[ej, 0])
            dy = float(cleaned[ei, 1]) - float(cleaned[ej, 1])
            edge_len = math.sqrt(dx * dx + dy * dy)
            denom = max(float(prior.get("mad_px", 1.0)), 1.0)
            z = abs(edge_len - float(prior["median_px"])) / denom
            if z > float(edge_length_z_threshold):
                n_edge_outliers += 1
        if n_edge_outliers > 0:
            flags.append(f"edge_outlier:{n_edge_outliers}")

    # ------------------------------------------------------------------
    # 7. quality_score computation
    # ------------------------------------------------------------------
    if rejected:
        quality_score = 0.0
    else:
        # mean conf of valid keypoints in cleaned output
        valid_confs = [float(cleaned[i, 2]) for i in range(K) if valid_mask[i]]
        mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
        quality_score = valid_fraction * mean_conf
        if body_length_outlier:
            quality_score *= 0.7
        if n_edge_outliers > 0:
            # Each outlier edge applies a 15% penalty, capped at 50% total
            quality_score *= max(0.5, 1.0 - 0.15 * n_edge_outliers)
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 8. quality_state
    # ------------------------------------------------------------------
    if rejected:
        quality_state = "rejected"
    elif quality_score < 0.2:
        quality_state = "bad"
    elif quality_score < 0.7:
        quality_state = "partial"
    else:
        quality_state = "good"

    return PoseQualityResult(
        cleaned_keypoints=cleaned,
        valid_mask=valid_mask,
        quality_score=quality_score,
        quality_state=quality_state,
        quality_flags=flags,
        was_cleaned=was_cleaned,
    )


# ---------------------------------------------------------------------------
# calibrate_body_length_prior
# ---------------------------------------------------------------------------


def calibrate_body_length_prior(
    df: pd.DataFrame,
    pose_labels: List[str],
    anterior_indices: List[int],
    posterior_indices: List[int],
    min_valid_conf: float,
    high_conf_floor: float = 0.7,
) -> BodyLengthPrior:
    """Estimate a body-length prior from high-confidence frames.

    Filters to rows where PoseMeanConf >= high_conf_floor and computes the
    median and MAD of the body length across those rows.

    Args:
        df: DataFrame with pose keypoint columns (PoseKpt_{label}_X/Y/Conf)
            and a PoseMeanConf summary column.
        pose_labels: Ordered list of keypoint label strings.
        anterior_indices: Indices of anterior keypoints (for body-length calc).
        posterior_indices: Indices of posterior keypoints.
        min_valid_conf: Minimum confidence for individual keypoints.
        high_conf_floor: Minimum PoseMeanConf to include a row in calibration.

    Returns:
        BodyLengthPrior; is_valid=True only when n_samples >= 20.
    """
    _invalid = BodyLengthPrior(median_px=0.0, mad_px=0.0, n_samples=0, is_valid=False)

    if df is None or df.empty:
        return _invalid
    if not anterior_indices or not posterior_indices:
        return _invalid
    if not pose_labels:
        return _invalid

    # Filter to high-confidence rows
    if "PoseMeanConf" not in df.columns:
        high_conf_df = df
    else:
        try:
            high_conf_df = df[df["PoseMeanConf"] >= float(high_conf_floor)]
        except Exception:
            high_conf_df = df

    if high_conf_df.empty:
        return _invalid

    body_lengths: List[float] = []

    for _, row in high_conf_df.iterrows():
        kpts = _extract_keypoints_from_row(row, pose_labels)
        if kpts is None:
            continue
        geom = compute_pose_geometry_from_keypoints(
            kpts,
            anterior_indices,
            posterior_indices,
            min_valid_conf,
        )
        if geom is None:
            continue
        bl = geom.get("body_length")
        if bl is not None and float(bl) > 0.0:
            body_lengths.append(float(bl))

    n = len(body_lengths)
    if n == 0:
        return _invalid

    arr = np.asarray(body_lengths, dtype=np.float64)
    median_px = float(np.median(arr))
    mad_px = float(np.median(np.abs(arr - median_px)))

    return BodyLengthPrior(
        median_px=median_px,
        mad_px=mad_px,
        n_samples=n,
        is_valid=n >= 20,
    )


# ---------------------------------------------------------------------------
# calibrate_edge_length_priors
# ---------------------------------------------------------------------------


def calibrate_edge_length_priors(
    df: pd.DataFrame,
    pose_labels: List[str],
    skeleton_edges: List[Tuple[int, int]],
    min_valid_conf: float,
    high_conf_floor: float = 0.7,
) -> EdgeLengthPriors:
    """Estimate per-skeleton-edge length distributions from high-confidence frames.

    For each edge (i, j) defined in *skeleton_edges*, measures the pixel
    distance between the two keypoints in every high-confidence row and
    computes the median and MAD of those distances.

    Args:
        df: DataFrame with PoseKpt_{label}_X/Y/Conf columns and PoseMeanConf.
        pose_labels: Ordered list of keypoint label strings.
        skeleton_edges: List of (i, j) keypoint index pairs.
        min_valid_conf: Minimum confidence to treat a keypoint as valid.
        high_conf_floor: Minimum PoseMeanConf to include a row in calibration.

    Returns:
        EdgeLengthPriors; is_valid=True when at least one edge has >= 20 samples.
    """
    _invalid = EdgeLengthPriors(priors={}, is_valid=False)

    if df is None or df.empty or not pose_labels or not skeleton_edges:
        return _invalid

    # Filter to high-confidence rows
    if "PoseMeanConf" in df.columns:
        try:
            high_conf_df = df[df["PoseMeanConf"] >= float(high_conf_floor)]
        except Exception:
            high_conf_df = df
    else:
        high_conf_df = df

    if high_conf_df.empty:
        return _invalid

    # Accumulate distances per canonical edge key
    edge_samples: Dict[Tuple[int, int], List[float]] = {}
    K = len(pose_labels)

    for _, row in high_conf_df.iterrows():
        kpts = _extract_keypoints_from_row(row, pose_labels)
        if kpts is None or len(kpts) < K:
            continue
        for edge in skeleton_edges:
            try:
                ei, ej = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if ei >= K or ej >= K:
                continue
            xi, yi, ci = float(kpts[ei, 0]), float(kpts[ei, 1]), float(kpts[ei, 2])
            xj, yj, cj = float(kpts[ej, 0]), float(kpts[ej, 1]), float(kpts[ej, 2])
            if ci < float(min_valid_conf) or cj < float(min_valid_conf):
                continue
            if not (math.isfinite(xi) and math.isfinite(yi)):
                continue
            if not (math.isfinite(xj) and math.isfinite(yj)):
                continue
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist <= 0.0:
                continue
            key = (min(ei, ej), max(ei, ej))
            edge_samples.setdefault(key, []).append(dist)

    priors: Dict[Tuple[int, int], Dict] = {}
    any_valid = False
    for key, samples in edge_samples.items():
        n = len(samples)
        arr = np.asarray(samples, dtype=np.float64)
        median_px = float(np.median(arr))
        mad_px = float(np.median(np.abs(arr - median_px)))
        priors[key] = {
            "median_px": median_px,
            "mad_px": mad_px,
            "n_samples": n,
        }
        if n >= 20:
            any_valid = True

    return EdgeLengthPriors(priors=priors, is_valid=any_valid)


# ---------------------------------------------------------------------------
# normalize_pose_keypoints_for_relink
# ---------------------------------------------------------------------------


def normalize_pose_keypoints_for_relink(
    window_df: pd.DataFrame,
    pose_labels: List[str],
    min_valid_conf: float,
) -> tuple:
    """Aggregate and normalize pose keypoints across a short window of rows.

    This is the canonical replacement for ``_normalize_pose_keypoints_window``
    in ``processing.py``.  Results are identical to that function.

    Args:
        window_df: DataFrame of up to ~3 rows for a single trajectory window.
        pose_labels: Ordered list of keypoint label strings.
        min_valid_conf: Minimum confidence to include a keypoint in aggregation.

    Returns:
        (normalized_array, visibility) where normalized_array is [K, 3] float32
        or None, and visibility is a float in [0, 1].
    """
    if window_df is None or window_df.empty or not pose_labels:
        return None, 0.0

    K = len(pose_labels)
    keypoints = np.full((K, 3), np.nan, dtype=np.float32)
    keypoints[:, 2] = 0.0
    valid_points = []
    valid_weights = []

    for idx, label in enumerate(pose_labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        if (
            x_col not in window_df.columns
            or y_col not in window_df.columns
            or c_col not in window_df.columns
        ):
            continue

        vals = window_df[[x_col, y_col, c_col]].copy()
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        vals = vals[vals[c_col] >= float(min_valid_conf)]
        if vals.empty:
            continue

        x = float(vals[x_col].median())
        y = float(vals[y_col].median())
        conf = float(vals[c_col].median())
        keypoints[idx] = np.array([x, y, conf], dtype=np.float32)
        valid_points.append((x, y))
        valid_weights.append(max(1e-6, conf))

    if not valid_points:
        return None, 0.0

    pts_arr = np.asarray(valid_points, dtype=np.float64)
    w_arr = np.asarray(valid_weights, dtype=np.float64)
    cx = float(np.average(pts_arr[:, 0], weights=w_arr))
    cy = float(np.average(pts_arr[:, 1], weights=w_arr))
    centered = pts_arr - np.array([[cx, cy]], dtype=np.float64)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    scale = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
    scale = max(scale, 1.0)

    normalized = np.full((K, 3), np.nan, dtype=np.float32)
    normalized[:, 2] = 0.0
    valid_counter = 0
    for idx in range(K):
        x, y, conf = keypoints[idx]
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
            continue
        normalized[idx, 0] = np.float32((float(x) - cx) / scale)
        normalized[idx, 1] = np.float32((float(y) - cy) / scale)
        normalized[idx, 2] = np.float32(conf)
        valid_counter += 1

    visibility = float(valid_counter) / float(K) if K > 0 else 0.0
    return normalized, float(np.clip(visibility, 0.0, 1.0))


# ---------------------------------------------------------------------------
# apply_quality_to_dataframe
# ---------------------------------------------------------------------------


def apply_quality_to_dataframe(
    df: pd.DataFrame,
    pose_labels: List[str],
    params: dict,
    body_length_prior: Optional[BodyLengthPrior] = None,
    anterior_indices: Optional[List[int]] = None,
    posterior_indices: Optional[List[int]] = None,
    skeleton_edges: Optional[List[Tuple[int, int]]] = None,
    edge_length_priors: Optional[EdgeLengthPriors] = None,
) -> pd.DataFrame:
    """Apply pose quality assessment to all rows of a trajectory DataFrame.

    Modifies confidence columns in-place on a copy, adds quality metadata
    columns, and returns the modified copy.  Never raises exceptions for bad
    rows — they are marked as rejected.

    Args:
        df: Input DataFrame with PoseKpt_{label}_X/Y/Conf columns.
        pose_labels: Ordered list of keypoint label strings.
        params: Tracking parameter dict.  Relevant keys:
            POSE_MIN_KPT_CONF_VALID, POSE_EXPORT_MIN_VALID_FRACTION,
            POSE_EXPORT_MIN_VALID_KEYPOINTS, POSE_IGNORE_KEYPOINTS.
        body_length_prior: Optional prior for body-length outlier filtering.
        anterior_indices: Anterior keypoint indices for body-length calc.
        posterior_indices: Posterior keypoint indices for body-length calc.
        skeleton_edges: List of (i, j) index pairs for connected-keypoint
            anatomy checks.
        edge_length_priors: Calibrated per-edge length priors.

    Returns:
        Modified copy of df with added/updated quality columns.
    """
    if df is None or df.empty:
        return df

    # ------------------------------------------------------------------
    # Extract parameters
    # ------------------------------------------------------------------
    min_valid_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    min_valid_fraction = float(params.get("POSE_EXPORT_MIN_VALID_FRACTION", 0.5))
    min_valid_keypoints = int(params.get("POSE_EXPORT_MIN_VALID_KEYPOINTS", 3))
    ignore_kpts_raw = params.get("POSE_IGNORE_KEYPOINTS", [])
    ignore_indices: List[int] = [int(i) for i in (ignore_kpts_raw or [])]

    # ------------------------------------------------------------------
    # Build column triplets
    # ------------------------------------------------------------------
    col_triplets = []
    for label in pose_labels:
        col_triplets.append(
            (f"PoseKpt_{label}_X", f"PoseKpt_{label}_Y", f"PoseKpt_{label}_Conf")
        )

    out = df.copy()

    # ------------------------------------------------------------------
    # Initialize new columns
    # ------------------------------------------------------------------
    out["PoseQualityScore"] = float("nan")
    out["PoseQualityState"] = ""
    out["PoseQualityFlags"] = ""
    out["PoseSource"] = ""
    out["PoseWasCleaned"] = 0

    # Determine which rows have any pose data
    all_conf_cols = [c for _, _, c in col_triplets if c in out.columns]

    for row_idx in out.index:
        row = out.loc[row_idx]

        # Check if this row has any pose data at all
        has_pose = False
        if all_conf_cols:
            try:
                has_pose = any(
                    not pd.isna(row[c]) for c in all_conf_cols if c in out.columns
                )
            except Exception:
                has_pose = False

        if not has_pose:
            out.at[row_idx, "PoseQualityState"] = "no_pose"
            out.at[row_idx, "PoseQualityScore"] = 0.0
            out.at[row_idx, "PoseSource"] = ""
            continue

        # Extract keypoints for this row
        kpts = _extract_keypoints_from_row(row, pose_labels)

        result = assess_pose_row(
            kpts,
            min_valid_conf=min_valid_conf,
            min_valid_fraction=min_valid_fraction,
            min_valid_keypoints=min_valid_keypoints,
            ignore_indices=ignore_indices if ignore_indices else None,
            body_length_prior=body_length_prior,
            anterior_indices=anterior_indices,
            posterior_indices=posterior_indices,
            skeleton_edges=skeleton_edges,
            edge_length_priors=edge_length_priors,
        )

        # Write cleaned confidences back
        for i, label in enumerate(pose_labels):
            c_col = f"PoseKpt_{label}_Conf"
            if c_col in out.columns and i < len(result.cleaned_keypoints):
                out.at[row_idx, c_col] = float(result.cleaned_keypoints[i, 2])

        out.at[row_idx, "PoseQualityScore"] = result.quality_score
        out.at[row_idx, "PoseQualityState"] = result.quality_state
        out.at[row_idx, "PoseQualityFlags"] = "|".join(result.quality_flags)
        out.at[row_idx, "PoseWasCleaned"] = int(result.was_cleaned)
        out.at[row_idx, "PoseSource"] = "cache"

    return out


# ---------------------------------------------------------------------------
# apply_temporal_pose_postprocessing
# ---------------------------------------------------------------------------


def apply_temporal_pose_postprocessing(
    trajectory_df: pd.DataFrame,
    pose_labels: List[str],
    max_gap: int,
    z_score_threshold: float,
    fill_interpolated: bool = True,
) -> pd.DataFrame:
    """Apply temporal outlier suppression and gap-filling to one trajectory.

    Input must be a single trajectory's rows, ideally sorted by FrameID.
    This function sorts by FrameID internally.

    Args:
        trajectory_df: DataFrame for one animal trajectory.
        pose_labels: Ordered list of keypoint label strings.
        max_gap: Maximum number of consecutive bad frames to gap-fill.
        z_score_threshold: Rolling z-score above which a keypoint position
            is flagged as a temporal outlier.
        fill_interpolated: When True, linearly interpolate keypoint X/Y across
            gaps of at most ``max_gap`` frames.

    Returns:
        Modified copy of trajectory_df.
    """
    if trajectory_df is None or trajectory_df.empty or not pose_labels:
        return trajectory_df

    out = trajectory_df.copy()

    # Sort by FrameID
    if "FrameID" in out.columns:
        out = out.sort_values("FrameID").reset_index(drop=True)

    _VALID_STATES = {"good", "partial"}
    rolling_window = max(5, max_gap * 2)

    # ------------------------------------------------------------------
    # 3. Per-label outlier suppression
    # ------------------------------------------------------------------
    for label in pose_labels:
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"

        if (
            x_col not in out.columns
            or y_col not in out.columns
            or c_col not in out.columns
        ):
            continue

        # Determine valid rows for this label
        if "PoseQualityState" in out.columns:
            quality_ok = out["PoseQualityState"].isin(_VALID_STATES)
        else:
            quality_ok = pd.Series([True] * len(out), index=out.index)

        conf_ok = out[c_col].apply(lambda v: pd.notna(v) and float(v) > 0.0)
        valid_idx = out.index[quality_ok & conf_ok].tolist()

        if len(valid_idx) < 3:
            # Not enough data to do rolling stats
            continue

        # Compute rolling z-score on valid X and Y series
        x_series = out.loc[valid_idx, x_col].astype(float)
        y_series = out.loc[valid_idx, y_col].astype(float)

        for series, col in [(x_series, x_col), (y_series, y_col)]:
            roll_mean = series.rolling(
                rolling_window, min_periods=3, center=True
            ).mean()
            roll_std = series.rolling(rolling_window, min_periods=3, center=True).std()

            for idx_val in valid_idx:
                if idx_val not in roll_mean.index:
                    continue
                mean_v = roll_mean.loc[idx_val]
                std_v = roll_std.loc[idx_val]
                if pd.isna(mean_v):
                    continue
                std_v = float(std_v) if not pd.isna(std_v) else 0.0
                z = abs(float(series.loc[idx_val]) - float(mean_v)) / max(std_v, 1e-6)
                if z > float(z_score_threshold):
                    # Mark this row as an outlier — zero conf, update flags
                    out.at[idx_val, c_col] = 0.0
                    _add_flag(out, idx_val, "temporal_outlier")
                    out.at[idx_val, "PoseWasCleaned"] = 1

    # ------------------------------------------------------------------
    # 4. Gap-fill via linear interpolation
    # ------------------------------------------------------------------
    if fill_interpolated:
        for label in pose_labels:
            x_col = f"PoseKpt_{label}_X"
            y_col = f"PoseKpt_{label}_Y"
            c_col = f"PoseKpt_{label}_Conf"

            if (
                x_col not in out.columns
                or y_col not in out.columns
                or c_col not in out.columns
            ):
                continue

            # Identify valid rows (conf > 0 after outlier suppression)
            valid_mask = out[c_col].apply(lambda v: pd.notna(v) and float(v) > 0.0)
            valid_positions = out.index[valid_mask].tolist()

            if len(valid_positions) < 2:
                continue

            # Find contiguous gaps between valid positions
            for seg_start_pos, seg_end_pos in zip(
                valid_positions[:-1], valid_positions[1:]
            ):
                # Get the integer positions in the DataFrame
                start_iloc = out.index.get_loc(seg_start_pos)
                end_iloc = out.index.get_loc(seg_end_pos)
                gap_length = end_iloc - start_iloc - 1

                if gap_length <= 0 or gap_length > max_gap:
                    continue

                # Linear interpolation of X and Y over the gap
                x_start = float(out.at[seg_start_pos, x_col])
                x_end = float(out.at[seg_end_pos, x_col])
                y_start = float(out.at[seg_start_pos, y_col])
                y_end = float(out.at[seg_end_pos, y_col])

                gap_indices = out.index[start_iloc + 1 : end_iloc]
                for step, gap_idx in enumerate(gap_indices, start=1):
                    t = float(step) / float(gap_length + 1)
                    out.at[gap_idx, x_col] = x_start + t * (x_end - x_start)
                    out.at[gap_idx, y_col] = y_start + t * (y_end - y_start)
                    out.at[gap_idx, c_col] = 0.3  # low-trust interpolated conf
                    out.at[gap_idx, "PoseSource"] = "cleaned"
                    out.at[gap_idx, "PoseWasCleaned"] = 1

    # ------------------------------------------------------------------
    # 5. Recompute PoseMeanConf and PoseValidFraction for modified rows
    # ------------------------------------------------------------------
    _recompute_pose_summary(out, pose_labels)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_keypoints_from_row(row, pose_labels: List[str]) -> Optional[np.ndarray]:
    """Extract a [K, 3] float32 keypoints array from a DataFrame row."""
    K = len(pose_labels)
    kpts = np.full((K, 3), np.nan, dtype=np.float32)
    kpts[:, 2] = 0.0

    any_found = False
    for i, label in enumerate(pose_labels):
        x_col = f"PoseKpt_{label}_X"
        y_col = f"PoseKpt_{label}_Y"
        c_col = f"PoseKpt_{label}_Conf"
        try:
            x = row[x_col]
            y = row[y_col]
            conf = row[c_col]
        except (KeyError, TypeError):
            continue
        try:
            fx, fy, fc = float(x), float(y), float(conf)
        except (ValueError, TypeError):
            continue
        kpts[i] = np.array([fx, fy, fc], dtype=np.float32)
        any_found = True

    return kpts if any_found else None


def _add_flag(df: pd.DataFrame, idx, flag: str) -> None:
    """Append a quality flag to PoseQualityFlags for the given row index."""
    if "PoseQualityFlags" not in df.columns:
        return
    current = (
        str(df.at[idx, "PoseQualityFlags"])
        if pd.notna(df.at[idx, "PoseQualityFlags"])
        else ""
    )
    if flag in current.split("|"):
        return
    new_val = f"{current}|{flag}" if current else flag
    df.at[idx, "PoseQualityFlags"] = new_val


def _recompute_pose_summary(df: pd.DataFrame, pose_labels: List[str]) -> None:
    """Recompute PoseMeanConf and PoseValidFraction columns in-place."""
    if not pose_labels:
        return
    conf_cols = [f"PoseKpt_{label}_Conf" for label in pose_labels]
    present_conf_cols = [c for c in conf_cols if c in df.columns]
    if not present_conf_cols:
        return

    K = len(pose_labels)

    if "PoseMeanConf" in df.columns or "PoseValidFraction" in df.columns:
        for idx in df.index:
            row = df.loc[idx]
            confs = []
            valid_count = 0
            for c in present_conf_cols:
                v = row[c]
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        confs.append(fv)
                        if fv > 0.0:
                            valid_count += 1
                except (ValueError, TypeError):
                    pass
            if "PoseMeanConf" in df.columns:
                df.at[idx, "PoseMeanConf"] = float(np.mean(confs)) if confs else 0.0
            if "PoseValidFraction" in df.columns:
                df.at[idx, "PoseValidFraction"] = (
                    float(valid_count) / float(K) if K > 0 else 0.0
                )
