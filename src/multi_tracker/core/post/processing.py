# src/multi_tracker/core/post/processing.py
"""
Trajectory post-processing utilities for cleaning and refining tracking data.

Optimizations:
- NumPy vectorization for distance calculations
- Numba JIT compilation for inner loops (if available)
- Parallel processing for independent trajectory operations
"""
import numpy as np
import logging
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from concurrent.futures import ThreadPoolExecutor
import warnings

# Import Numba from gpu_utils (handles availability detection)
from multi_tracker.utils.gpu_utils import NUMBA_AVAILABLE, njit, prange

logger = logging.getLogger(__name__)

# Create jit decorator based on availability
if NUMBA_AVAILABLE:
    from numba import jit
else:
    # Create no-op decorator when Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# ============================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# ============================================================================


@jit(nopython=True, cache=True)
def _compute_pairwise_distances_numba(x1, y1, frames1, x2, y2, frames2, threshold):
    """
    Compute agreeing frame count between two trajectories using Numba.

    Returns (agreeing_count, common_count)
    """
    agreeing = 0
    common = 0

    # Create frame lookup for trajectory 2
    for i in range(len(frames1)):
        f1 = frames1[i]
        x1_val = x1[i]
        y1_val = y1[i]

        # Check if NaN
        if np.isnan(x1_val) or np.isnan(y1_val):
            continue

        # Find matching frame in trajectory 2
        for j in range(len(frames2)):
            if frames2[j] == f1:
                x2_val = x2[j]
                y2_val = y2[j]

                if np.isnan(x2_val) or np.isnan(y2_val):
                    break

                common += 1
                dist = np.sqrt((x1_val - x2_val) ** 2 + (y1_val - y2_val) ** 2)
                if dist <= threshold:
                    agreeing += 1
                break

    return agreeing, common


@jit(nopython=True, cache=True)
def _compute_all_merge_candidates_numba(
    fwd_x_list,
    fwd_y_list,
    fwd_frames_list,
    fwd_starts,
    fwd_ends,
    bwd_x_list,
    bwd_y_list,
    bwd_frames_list,
    bwd_starts,
    bwd_ends,
    threshold,
    min_overlap,
):
    """
    Find all merge candidates between forward and backward trajectories.

    NOTE: Not using parallel=True due to race conditions with candidate_count.
    The sequential version is still fast due to Numba JIT.

    Returns arrays of (forward_idx, backward_idx, agreeing_count, common_count)
    """
    n_fwd = len(fwd_starts)
    n_bwd = len(bwd_starts)

    # Pre-allocate maximum possible candidates
    max_candidates = n_fwd * n_bwd
    candidates_fi = np.empty(max_candidates, dtype=np.int32)
    candidates_bi = np.empty(max_candidates, dtype=np.int32)
    candidates_agreeing = np.empty(max_candidates, dtype=np.int32)
    candidates_common = np.empty(max_candidates, dtype=np.int32)

    candidate_count = 0

    for fi in range(n_fwd):
        f_start, f_end = fwd_starts[fi], fwd_ends[fi]
        fwd_x = fwd_x_list[f_start:f_end]
        fwd_y = fwd_y_list[f_start:f_end]
        fwd_frames = fwd_frames_list[f_start:f_end]

        for bi in range(n_bwd):
            b_start, b_end = bwd_starts[bi], bwd_ends[bi]
            bwd_x = bwd_x_list[b_start:b_end]
            bwd_y = bwd_y_list[b_start:b_end]
            bwd_frames = bwd_frames_list[b_start:b_end]

            agreeing, common = _compute_pairwise_distances_numba(
                fwd_x, fwd_y, fwd_frames, bwd_x, bwd_y, bwd_frames, threshold
            )

            if agreeing >= min_overlap:
                candidates_fi[candidate_count] = fi
                candidates_bi[candidate_count] = bi
                candidates_agreeing[candidate_count] = agreeing
                candidates_common[candidate_count] = common
                candidate_count += 1

    return (
        candidates_fi[:candidate_count],
        candidates_bi[:candidate_count],
        candidates_agreeing[:candidate_count],
        candidates_common[:candidate_count],
    )


def _prepare_trajectory_arrays(traj_dfs):
    """
    Convert list of trajectory DataFrames to flat NumPy arrays for Numba.

    Returns: (x_array, y_array, frames_array, start_indices, end_indices)
    """
    if not traj_dfs:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Calculate total size
    total_size = sum(len(df) for df in traj_dfs)

    # Pre-allocate arrays
    x_array = np.empty(total_size, dtype=np.float64)
    y_array = np.empty(total_size, dtype=np.float64)
    frames_array = np.empty(total_size, dtype=np.int64)
    starts = np.empty(len(traj_dfs), dtype=np.int64)
    ends = np.empty(len(traj_dfs), dtype=np.int64)

    offset = 0
    for i, df in enumerate(traj_dfs):
        n = len(df)
        starts[i] = offset
        ends[i] = offset + n

        x_array[offset : offset + n] = df["X"].values
        y_array[offset : offset + n] = df["Y"].values
        frames_array[offset : offset + n] = df["FrameID"].values.astype(np.int64)

        offset += n

    return x_array, y_array, frames_array, starts, ends


def _find_merge_candidates_python(
    forward_dfs, backward_dfs, agreement_distance, min_overlap
):
    """
    Pure Python fallback for finding merge candidates.
    Used when Numba is not available or for small trajectory counts.

    Optimized with vectorized NumPy operations where possible.
    """
    merge_candidates = []

    for fi, fwd in enumerate(forward_dfs):
        # Extract arrays for vectorized operations
        fwd_frames = fwd["FrameID"].values
        fwd_x = fwd["X"].values
        fwd_y = fwd["Y"].values
        fwd_frame_set = set(fwd_frames)

        for bi, bwd in enumerate(backward_dfs):
            bwd_frames = bwd["FrameID"].values
            bwd_frame_set = set(bwd_frames)
            common_frames = fwd_frame_set.intersection(bwd_frame_set)

            if len(common_frames) < min_overlap:
                continue

            # Build index mappings for fast lookup
            fwd_frame_to_idx = {f: i for i, f in enumerate(fwd_frames)}
            bwd_frame_to_idx = {f: i for i, f in enumerate(bwd_frames)}

            bwd_x = bwd["X"].values
            bwd_y = bwd["Y"].values

            # Count agreeing frames using vectorized operations
            agreeing_frames = 0
            for frame in common_frames:
                fi_idx = fwd_frame_to_idx[frame]
                bi_idx = bwd_frame_to_idx[frame]

                fx, fy = fwd_x[fi_idx], fwd_y[fi_idx]
                bx, by = bwd_x[bi_idx], bwd_y[bi_idx]

                # Skip if either has NaN positions
                if np.isnan(fx) or np.isnan(bx):
                    continue

                dist = np.sqrt((fx - bx) ** 2 + (fy - by) ** 2)
                if dist <= agreement_distance:
                    agreeing_frames += 1

            if agreeing_frames >= min_overlap:
                merge_candidates.append((fi, bi, agreeing_frames, len(common_frames)))

    return merge_candidates


# ============================================================================
# TRAJECTORY PROCESSING FUNCTIONS
# ============================================================================


def _compute_velocity_zscore_breaks(
    traj_df,
    zscore_threshold=3.0,
    window_size=10,
    min_velocity_threshold=2.0,
    std_regularization=0.5,
    active_velocity_threshold=0.5,
):
    """
    Identify trajectory break points based on velocity z-scores.

    Detects sudden, statistically significant changes in velocity that often indicate
    identity swaps. For each point, calculates the z-score of its velocity relative to
    the rolling mean and std of recent past velocities.

    CRITICAL SAFEGUARDS to avoid false positives:
    1. Minimum velocity threshold - prevents breaking when animal starts moving from rest
    2. Regularized std - prevents extreme z-scores from low-variability periods
    3. Active velocity filtering - excludes near-stationary periods from statistics

    Args:
        traj_df (pd.DataFrame): Trajectory with 'Velocity' column already computed
        zscore_threshold (float): Z-score threshold for breaking (default: 3.0)
        window_size (int): Number of past velocities to use for statistics (default: 10)
        min_velocity_threshold (float): Minimum velocity to consider for z-score breaking (pixels/frame)
        std_regularization (float): Regularization added to std to prevent extreme z-scores (pixels/frame)
        active_velocity_threshold (float): Minimum velocity to be considered "active" (pixels/frame)

    Returns:
        list: Indices where velocity z-score exceeds threshold
    """
    if len(traj_df) < window_size + 2:
        return []  # Not enough data for z-score analysis

    velocities = traj_df["Velocity"].values
    break_indices = []

    # Start from window_size+1 to ensure we have enough history
    for i in range(window_size + 1, len(velocities)):
        current_vel = velocities[i]

        # Skip if current velocity is NaN (happens at trajectory start or after gaps)
        if pd.isna(current_vel):
            continue

        # SAFEGUARD 1: Skip if current velocity is too low
        # This prevents false breaks when animal transitions from rest to normal movement
        if current_vel < min_velocity_threshold:
            continue

        # Get past velocities within the window (excluding NaN values)
        past_velocities = velocities[max(0, i - window_size) : i]
        past_velocities = past_velocities[~pd.isna(past_velocities)]

        # Need at least 3 past velocities for meaningful statistics
        if len(past_velocities) < 3:
            continue

        # SAFEGUARD 3: Calculate statistics using only "active" velocities
        # This prevents stationary periods from creating artificially low baseline
        active_velocities = past_velocities[
            past_velocities >= active_velocity_threshold
        ]

        # If too few active velocities, fall back to all velocities
        if len(active_velocities) >= 3:
            mean_vel = np.mean(active_velocities)
            std_vel = np.std(active_velocities, ddof=1)
        else:
            mean_vel = np.mean(past_velocities)
            std_vel = np.std(past_velocities, ddof=1)

        # SAFEGUARD 2: Regularize std to prevent extreme z-scores
        # When std is very small (consistent low movement), regularization prevents
        # normal movement from triggering breaks
        regularized_std = std_vel + std_regularization

        # Calculate z-score with regularized std
        zscore = (current_vel - mean_vel) / regularized_std

        # Break if z-score exceeds threshold (only positive - sudden acceleration)
        # This now requires BOTH unusually high velocity relative to history
        # AND sufficient absolute velocity (via min_velocity_threshold check above)
        if zscore > zscore_threshold:
            break_indices.append(traj_df.index[i])

    return break_indices


def process_trajectories_from_csv(csv_path, params):
    """
    Cleans and refines trajectory data from CSV file, preserving all columns including confidence metrics.

    This function performs several steps:
    - Removes trajectories that are too short.
    - Breaks trajectories at points of impossibly high velocity or large jumps,
      which often indicate identity switches or tracking errors.
    - Preserves all columns from the input CSV (including confidence metrics)

    Args:
        csv_path (str): Path to the raw CSV file from tracking
        params (dict): The dictionary of tracking parameters.

    Returns:
        tuple: (final_trajectories_df, statistics_dict)
    """
    if pd is None:
        logger.warning("pandas is not available. Skipping trajectory post-processing.")
        return None, {}

    min_len = params.get("MIN_TRAJECTORY_LENGTH", 10)
    max_vel_break = params.get("MAX_VELOCITY_BREAK", 100.0)
    # MAX_DISTANCE_BREAK is derived from MAX_VELOCITY_BREAK * frame_diff (computed per-point)
    max_occlusion_gap = params.get("MAX_OCCLUSION_GAP", 30)
    max_vel_zscore = params.get("MAX_VELOCITY_ZSCORE", 0.0)  # 0.0 means disabled
    vel_zscore_window = params.get("VELOCITY_ZSCORE_WINDOW", 10)
    vel_zscore_min_vel = params.get("VELOCITY_ZSCORE_MIN_VELOCITY", 2.0)  # pixels/frame

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(
            f"Loaded {len(df)} rows from {csv_path} with columns: {list(df.columns)}"
        )

        # Drop unnecessary columns from raw tracking data
        columns_to_drop = ["TrackID", "Index"]
        df = df.drop(
            columns=[col for col in columns_to_drop if col in df.columns],
            errors="ignore",
        )
        if columns_to_drop:
            logger.info(
                f"Dropped columns: {[col for col in columns_to_drop if col in df.columns]}"
            )
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return None, {}

    # Check if confidence columns exist
    has_confidence = "DetectionConfidence" in df.columns

    # Set X, Y, Theta to NaN for occluded/lost states to enable proper interpolation
    if "State" in df.columns:
        occluded_mask = df["State"].isin(["occluded", "lost"])
        if occluded_mask.any():
            logger.info(
                f"Setting X, Y, Theta to NaN for {occluded_mask.sum()} occluded/lost detections"
            )
            df.loc[occluded_mask, ["X", "Y", "Theta"]] = np.nan

    stats = {
        "original_count": df["TrajectoryID"].nunique(),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_velocity_zscore": 0,
        "broken_occlusion": 0,
        "broken_spatial_gap": 0,  # New: breaks due to spatial jumps across NaN gaps
        "final_count": 0,
    }

    cleaned_segments = []
    new_traj_id = 0

    for traj_id in df["TrajectoryID"].unique():
        traj_df = (
            df[df["TrajectoryID"] == traj_id]
            .sort_values("FrameID")
            .reset_index(drop=True)
        )
        if len(traj_df) < min_len:
            stats["removed_short"] += 1
            continue

        # Calculate frame difference, distance, and velocity between consecutive points
        traj_df["FrameDiff"] = traj_df["FrameID"].diff()
        traj_df["DistDiff"] = np.sqrt(
            traj_df["X"].diff() ** 2 + traj_df["Y"].diff() ** 2
        )
        traj_df["Velocity"] = traj_df["DistDiff"] / traj_df["FrameDiff"]
        # Compute max allowed distance per frame gap (velocity-based threshold)
        traj_df["MaxAllowedDist"] = max_vel_break * traj_df["FrameDiff"]

        # Identify break points: velocity exceeds threshold (handles variable frame gaps)
        break_indices = traj_df[traj_df["Velocity"] > max_vel_break].index.tolist()

        stats["broken_velocity"] += len(break_indices)

        # Add z-score based velocity breaks if enabled
        if max_vel_zscore > 0:
            zscore_break_indices = _compute_velocity_zscore_breaks(
                traj_df, zscore_threshold=max_vel_zscore, window_size=vel_zscore_window
            )
            stats["broken_velocity_zscore"] += len(zscore_break_indices)
            # Combine break indices and remove duplicates
            break_indices = sorted(set(break_indices + zscore_break_indices))

        # Create segments based on break points
        segment_start_idx = 0
        for break_idx in break_indices:
            segment = traj_df.iloc[segment_start_idx:break_idx].copy()
            if len(segment) >= min_len:
                # Assign new trajectory ID
                segment["TrajectoryID"] = new_traj_id
                new_traj_id += 1
                # Drop temporary columns
                segment = segment.drop(
                    columns=["FrameDiff", "DistDiff", "Velocity", "MaxAllowedDist"],
                    errors="ignore",
                )
                cleaned_segments.append(segment)
            segment_start_idx = break_idx

        # Add the last segment
        last_segment = traj_df.iloc[segment_start_idx:].copy()
        if len(last_segment) >= min_len:
            last_segment["TrajectoryID"] = new_traj_id
            new_traj_id += 1
            last_segment = last_segment.drop(
                columns=["FrameDiff", "DistDiff", "Velocity", "MaxAllowedDist"],
                errors="ignore",
            )
            cleaned_segments.append(last_segment)

    # Further split segments based on long occlusion gaps
    if "State" in df.columns and max_occlusion_gap > 0:
        final_segments = []
        for segment in cleaned_segments:
            # Find runs of consecutive occluded/lost frames
            is_occluded = segment["State"].isin(["occluded", "lost"])

            # Find where occlusion state changes
            state_changes = is_occluded.ne(is_occluded.shift()).cumsum()

            # Group by state change to get runs
            runs = segment.groupby(state_changes)

            # Find occlusion runs that are too long
            split_indices = []
            for run_id, run_group in runs:
                if (
                    is_occluded.loc[run_group.index[0]]
                    and len(run_group) > max_occlusion_gap
                ):
                    # Split at the start of this long occlusion run
                    split_indices.append(run_group.index[0])
                    stats["broken_occlusion"] += 1

            if not split_indices:
                # No long occlusion gaps, keep segment as is
                final_segments.append(segment)
            else:
                # Split segment at occlusion gaps
                split_indices = sorted(split_indices)
                subseg_start_idx = segment.index[0]

                for split_idx in split_indices:
                    subseg = segment.loc[subseg_start_idx : split_idx - 1].copy()
                    if len(subseg) >= min_len:
                        subseg["TrajectoryID"] = new_traj_id
                        new_traj_id += 1
                        final_segments.append(subseg)
                    subseg_start_idx = split_idx

                # Add the last subsegment
                last_subseg = segment.loc[subseg_start_idx:].copy()
                if len(last_subseg) >= min_len:
                    last_subseg["TrajectoryID"] = new_traj_id
                    new_traj_id += 1
                    final_segments.append(last_subseg)

        cleaned_segments = final_segments

    # Split segments on large spatial discontinuities across NaN gaps (hidden jumps)
    # These would create unrealistic velocities when interpolated
    final_segments_spatial = []
    for segment in cleaned_segments:
        # Find all valid (non-NaN) positions in this segment
        valid_mask = segment["X"].notna()
        if valid_mask.sum() < 2:
            # Not enough valid positions to check for jumps
            final_segments_spatial.append(segment)
            continue

        # Get indices of valid positions
        valid_indices = segment[valid_mask].index.tolist()

        # Check for spatial jumps between consecutive valid positions
        split_indices = []
        for i in range(1, len(valid_indices)):
            curr_idx = valid_indices[i]
            prev_idx = valid_indices[i - 1]

            curr_row = segment.loc[curr_idx]
            prev_row = segment.loc[prev_idx]

            frame_gap = curr_row["FrameID"] - prev_row["FrameID"]

            # Only check if there's a gap (NaN frames between valid positions)
            if frame_gap > 1:
                distance = np.sqrt(
                    (curr_row["X"] - prev_row["X"]) ** 2
                    + (curr_row["Y"] - prev_row["Y"]) ** 2
                )
                velocity = distance / frame_gap  # Average velocity across gap

                # If velocity exceeds threshold, mark for split
                if velocity > max_vel_break:
                    split_indices.append(curr_idx)
                    stats["broken_spatial_gap"] += 1

        if not split_indices:
            # No spatial jumps, keep segment as is
            final_segments_spatial.append(segment)
        else:
            # Split segment at spatial jump points
            split_indices = sorted(split_indices)
            subseg_start_idx = segment.index[0]

            for split_idx in split_indices:
                subseg = segment.loc[subseg_start_idx : split_idx - 1].copy()
                if len(subseg) >= min_len:
                    subseg["TrajectoryID"] = new_traj_id
                    new_traj_id += 1
                    final_segments_spatial.append(subseg)
                subseg_start_idx = split_idx

            # Add the last subsegment
            last_subseg = segment.loc[subseg_start_idx:].copy()
            if len(last_subseg) >= min_len:
                last_subseg["TrajectoryID"] = new_traj_id
                new_traj_id += 1
                final_segments_spatial.append(last_subseg)

    cleaned_segments = final_segments_spatial

    stats["final_count"] = len(cleaned_segments)

    # Concatenate all segments into one dataframe
    if cleaned_segments:
        result_df = pd.concat(cleaned_segments, ignore_index=True)

        # Clean trajectories: remove those with no valid detections and trim trailing lost/occluded states
        if "State" in result_df.columns:
            cleaned_traj_list = []
            for traj_id in result_df["TrajectoryID"].unique():
                traj_df = result_df[result_df["TrajectoryID"] == traj_id]
                cleaned_traj = _clean_trajectory(traj_df)
                if cleaned_traj is not None and len(cleaned_traj) >= min_len:
                    cleaned_traj_list.append(cleaned_traj)

            if cleaned_traj_list:
                result_df = pd.concat(cleaned_traj_list, ignore_index=True)
                # Reassign trajectory IDs sequentially
                old_ids = result_df["TrajectoryID"].unique()
                id_mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
                result_df["TrajectoryID"] = result_df["TrajectoryID"].map(id_mapping)
                stats["final_count"] = len(old_ids)
            else:
                result_df = pd.DataFrame()
                stats["final_count"] = 0
    else:
        result_df = pd.DataFrame()

    logger.info(f"Post-processing stats: {stats}")
    return result_df, stats


def process_trajectories(trajectories_full, params):
    """
    Cleans and refines raw trajectory data.

    This function performs several steps:
    - Removes trajectories that are too short.
    - Breaks trajectories at points of impossibly high velocity or large jumps,
      which often indicate identity switches or tracking errors.

    Args:
        trajectories_full (list of lists): The raw trajectory data from the tracker.
        params (dict): The dictionary of tracking parameters.

    Returns:
        tuple: (final_trajectories, statistics_dict)
    """
    if pd is None:
        logger.warning("pandas is not available. Skipping trajectory post-processing.")
        return trajectories_full, {}

    min_len = params.get("MIN_TRAJECTORY_LENGTH", 10)
    max_vel_break = params.get("MAX_VELOCITY_BREAK", 100.0)
    # MAX_DISTANCE_BREAK is derived from MAX_VELOCITY_BREAK * frame_diff (computed per-point)
    max_vel_zscore = params.get("MAX_VELOCITY_ZSCORE", 0.0)  # 0.0 means disabled
    vel_zscore_window = params.get("VELOCITY_ZSCORE_WINDOW", 10)
    vel_zscore_min_vel = params.get("VELOCITY_ZSCORE_MIN_VELOCITY", 2.0)  # pixels/frame

    all_data = []
    # Note: Using original track_id as TrajectoryID for this initial dataframe.
    # A more advanced version could use the persistent trajectory_ids from the tracker.
    for track_id, traj in enumerate(trajectories_full):
        for point in traj:
            x, y, theta, frame = point
            all_data.append(
                {
                    "TrajectoryID": track_id,
                    "X": int(x),
                    "Y": int(y),
                    "Theta": theta,
                    "FrameID": int(frame),
                }
            )

    if not all_data:
        return [], {
            "original_count": 0,
            "removed_short": 0,
            "broken_velocity": 0,
            "broken_velocity_zscore": 0,
            "final_count": 0,
        }

    df = pd.DataFrame(all_data)

    stats = {
        "original_count": len([t for t in trajectories_full if t]),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_velocity_zscore": 0,
        "final_count": 0,
    }

    cleaned_segments = []
    for traj_id in df["TrajectoryID"].unique():
        traj_df = (
            df[df["TrajectoryID"] == traj_id]
            .sort_values("FrameID")
            .reset_index(drop=True)
        )
        if len(traj_df) < min_len:
            stats["removed_short"] += 1
            continue

        # Calculate frame difference, distance, and velocity between consecutive points
        traj_df["FrameDiff"] = traj_df["FrameID"].diff()
        traj_df["DistDiff"] = np.sqrt(
            traj_df["X"].diff() ** 2 + traj_df["Y"].diff() ** 2
        )
        traj_df["Velocity"] = traj_df["DistDiff"] / traj_df["FrameDiff"]

        # Identify break points: velocity exceeds threshold (handles variable frame gaps)
        break_indices = traj_df[traj_df["Velocity"] > max_vel_break].index.tolist()

        stats["broken_velocity"] += len(break_indices)

        # Add z-score based velocity breaks if enabled
        if max_vel_zscore > 0:
            # Calculate body-size-relative thresholds
            # std_regularization and active_threshold are set to ~25% and ~12.5% of min_velocity
            zscore_break_indices = _compute_velocity_zscore_breaks(
                traj_df,
                zscore_threshold=max_vel_zscore,
                window_size=vel_zscore_window,
                min_velocity_threshold=vel_zscore_min_vel,
                std_regularization=vel_zscore_min_vel * 0.25,
                active_velocity_threshold=vel_zscore_min_vel * 0.125,
            )
            stats["broken_velocity_zscore"] += len(zscore_break_indices)
            # Combine break indices and remove duplicates
            break_indices = sorted(set(break_indices + zscore_break_indices))

        # Create segments based on break points
        segment_start_idx = 0
        for break_idx in break_indices:
            segment = traj_df.iloc[segment_start_idx:break_idx]
            if len(segment) >= min_len:
                cleaned_segments.append(segment)
            segment_start_idx = break_idx

        # Add the last segment
        last_segment = traj_df.iloc[segment_start_idx:]
        if len(last_segment) >= min_len:
            cleaned_segments.append(last_segment)

    stats["final_count"] = len(cleaned_segments)

    # Convert dataframe segments back to list of tuples format
    final_trajectories = [
        [tuple(row) for row in seg_df[["X", "Y", "Theta", "FrameID"]].to_numpy()]
        for seg_df in cleaned_segments
    ]

    logger.info(f"Post-processing stats: {stats}")
    return final_trajectories, stats


def resolve_trajectories(forward_trajs, backward_trajs, params=None):
    """
    Merges forward and backward trajectories using conservative consensus-based merging.

    This function prioritizes identity confidence over trajectory completeness:
    1. Only considers trajectory pairs as merge candidates if they have sufficient
       overlapping frames where positions agree (within AGREEMENT_DISTANCE)
    2. Merges only the agreeing segments - disagreeing frames cause trajectory splits
    3. Results in more trajectory fragments but higher confidence in identity

    Algorithm:
    - For each forward/backward pair, count frames where both have valid positions
      within AGREEMENT_DISTANCE of each other
    - If count >= MIN_OVERLAP_FRAMES, they are merge candidates
    - During merge: agreeing frames are averaged, disagreeing frames cause splits
      into separate trajectory segments

    Args:
        forward_trajs (list): List of forward trajectory DataFrames or lists of tuples
        backward_trajs (list): List of backward trajectory DataFrames or lists of tuples
        params (dict, optional): Parameters for merging thresholds

    Returns:
        list: Final merged trajectories as list of DataFrames
    """

    if not forward_trajs and not backward_trajs:
        return []

    # Default parameters
    if params is None:
        params = {}

    # Get parameters - AGREEMENT_DISTANCE should be ~0.5 * body_size (in scaled pixels)
    AGREEMENT_DISTANCE = params.get("AGREEMENT_DISTANCE", 15.0)
    MIN_OVERLAP_FRAMES = params.get("MIN_OVERLAP_FRAMES", 5)
    MIN_LENGTH = params.get("MIN_TRAJECTORY_LENGTH", 5)

    logger.info(
        f"Starting conservative trajectory resolution with {len(forward_trajs)} forward "
        f"and {len(backward_trajs)} backward trajectories"
    )
    logger.info(
        f"Parameters: AGREEMENT_DISTANCE={AGREEMENT_DISTANCE:.2f}px, "
        f"MIN_OVERLAP_FRAMES={MIN_OVERLAP_FRAMES}, MIN_LENGTH={MIN_LENGTH}"
    )

    # Convert and prepare forward trajectories
    forward_dfs = []
    for i, traj in enumerate(forward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"forward_{i}")
        if len(df) >= MIN_LENGTH:
            df["Theta"] = df["Theta"] % (2 * np.pi)
            df["_source"] = "forward"
            forward_dfs.append(df)

    # Convert and prepare backward trajectories
    backward_dfs = []
    for i, traj in enumerate(backward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"backward_{i}")
        if len(df) >= MIN_LENGTH:
            # No frame adjustment needed - tracking worker now saves actual frame indices
            # Rotate theta by 180 degrees for backward trajectories
            df["Theta"] = (df["Theta"] + np.pi) % (2 * np.pi)
            df["_source"] = "backward"
            backward_dfs.append(df)

    # Clean trajectories
    forward_dfs = _clean_trajectories(forward_dfs, MIN_LENGTH)
    backward_dfs = _clean_trajectories(backward_dfs, MIN_LENGTH)

    logger.info(
        f"After cleaning: {len(forward_dfs)} forward, {len(backward_dfs)} backward"
    )

    if not forward_dfs and not backward_dfs:
        logger.warning("No valid trajectories found for merging")
        return []

    # If only one direction has trajectories, return those
    if not forward_dfs:
        for traj in backward_dfs:
            if "_source" in traj.columns:
                traj.drop(columns=["_source"], inplace=True)
        return backward_dfs
    if not backward_dfs:
        for traj in forward_dfs:
            if "_source" in traj.columns:
                traj.drop(columns=["_source"], inplace=True)
        return forward_dfs

    # Find merge candidates based on overlap counting
    # Use Numba-accelerated version if available
    if NUMBA_AVAILABLE and len(forward_dfs) > 5 and len(backward_dfs) > 5:
        logger.debug("Using Numba-accelerated merge candidate search")

        # Prepare trajectory arrays for Numba
        fwd_x, fwd_y, fwd_frames, fwd_starts, fwd_ends = _prepare_trajectory_arrays(
            forward_dfs
        )
        bwd_x, bwd_y, bwd_frames, bwd_starts, bwd_ends = _prepare_trajectory_arrays(
            backward_dfs
        )

        try:
            # Run Numba-accelerated candidate search
            fi_arr, bi_arr, agreeing_arr, common_arr = (
                _compute_all_merge_candidates_numba(
                    fwd_x,
                    fwd_y,
                    fwd_frames,
                    fwd_starts,
                    fwd_ends,
                    bwd_x,
                    bwd_y,
                    bwd_frames,
                    bwd_starts,
                    bwd_ends,
                    AGREEMENT_DISTANCE,
                    MIN_OVERLAP_FRAMES,
                )
            )
            merge_candidates = list(zip(fi_arr, bi_arr, agreeing_arr, common_arr))
        except Exception as e:
            logger.warning(f"Numba acceleration failed, falling back to Python: {e}")
            merge_candidates = _find_merge_candidates_python(
                forward_dfs, backward_dfs, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
            )
    else:
        merge_candidates = _find_merge_candidates_python(
            forward_dfs, backward_dfs, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
        )

    logger.info(f"Found {len(merge_candidates)} merge candidates")

    # Now merge candidates using conservative strategy
    used_forward = set()
    used_backward = set()
    result_trajectories = []

    # Sort by number of agreeing frames (most agreement first)
    merge_candidates.sort(key=lambda x: -x[2])

    for fi, bi, agreeing, total_common in merge_candidates:
        if fi in used_forward or bi in used_backward:
            continue

        used_forward.add(fi)
        used_backward.add(bi)

        logger.debug(
            f"Merging forward_{fi} with backward_{bi}: "
            f"{agreeing}/{total_common} agreeing frames"
        )

        # Perform conservative merge
        merged_segments = _conservative_merge(
            forward_dfs[fi], backward_dfs[bi], AGREEMENT_DISTANCE, MIN_LENGTH
        )
        result_trajectories.extend(merged_segments)

    # Add unused forward trajectories
    for fi, fwd in enumerate(forward_dfs):
        if fi not in used_forward:
            result_trajectories.append(fwd.copy())

    # Add unused backward trajectories
    for bi, bwd in enumerate(backward_dfs):
        if bi not in used_backward:
            result_trajectories.append(bwd.copy())

    # Note: Filtering by MIN_LENGTH is deferred until after stitching
    # to allow small fragments to be reconnected.

    # CRITICAL: Remove duplicate trajectories that are spatially contained within others
    # This can happen when a forward trajectory matches multiple backward trajectories,
    # and the "unused" ones cover the same physical location
    result_trajectories = _remove_spatially_redundant_trajectories(
        result_trajectories, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES
    )

    # CRITICAL: Merge overlapping trajectories that agree spatially
    # This handles fragments that overlap in the middle (neither contains the other)
    result_trajectories = _merge_overlapping_agreeing_trajectories(
        result_trajectories, AGREEMENT_DISTANCE, MIN_OVERLAP_FRAMES, MIN_LENGTH
    )

    # NEW: Stitch consecutive fragments that are spatially close
    # This fixes tracking breaks during turns or fast movements
    # Use 2x agreement distance to allow for fast movements/ambiguities across small gaps
    # Allow larger gap (3 frames) to jump over short occlusions
    result_trajectories = _stitch_broken_trajectory_fragments(
        result_trajectories,
        AGREEMENT_DISTANCE * 2,
        max_gap=3,
    )

    # FINAL CLEANING: Now that stitching is done, remove trajectories that are still too short
    result_trajectories = [t for t in result_trajectories if len(t) >= MIN_LENGTH]
    result_trajectories = _clean_trajectories(result_trajectories, MIN_LENGTH)

    # Reassign trajectory IDs and remove internal columns
    for new_id, traj in enumerate(result_trajectories):
        traj["TrajectoryID"] = new_id
        if "_source" in traj.columns:
            traj.drop(columns=["_source"], inplace=True)

    logger.info(f"Final result: {len(result_trajectories)} trajectories")

    return result_trajectories


def _conservative_merge(traj1, traj2, agreement_distance, min_length):
    """
    Conservatively merge two trajectories.

    - Agreeing frames (both exist, distance <= threshold): Average positions
    - Disagreeing frames (both exist, distance > threshold): Split into separate segments
    - Unique frames: Keep from whichever trajectory has them

    Returns list of trajectory DataFrames (may be more than input if splits occur).
    """
    # Build frame dictionaries (OPTIMIZED: avoid iterrows)
    t1_by_frame = {}
    for idx in range(len(traj1)):
        row = traj1.iloc[idx]
        t1_by_frame[row["FrameID"]] = row.to_dict()

    t2_by_frame = {}
    for idx in range(len(traj2)):
        row = traj2.iloc[idx]
        t2_by_frame[row["FrameID"]] = row.to_dict()

    frames1 = set(traj1["FrameID"])
    frames2 = set(traj2["FrameID"])
    all_frames = sorted(frames1.union(frames2))
    common_frames = frames1.intersection(frames2)

    # Classify each frame
    # Returns: ("agree", merged_row), ("disagree", (r1, r2)), ("t1_only", row), ("t2_only", row)
    frame_classifications = {}

    for frame in all_frames:
        if frame in common_frames:
            r1, r2 = t1_by_frame[frame], t2_by_frame[frame]

            # Handle NaN positions
            x1_valid = not pd.isna(r1.get("X"))
            x2_valid = not pd.isna(r2.get("X"))

            if not x1_valid and not x2_valid:
                # Both NaN, treat as agree with averaged metadata
                merged = _average_trajectory_rows(r1, r2)
                frame_classifications[frame] = ("agree", merged)
            elif not x1_valid:
                frame_classifications[frame] = ("t2_only", r2)
            elif not x2_valid:
                frame_classifications[frame] = ("t1_only", r1)
            else:
                dist = np.sqrt((r1["X"] - r2["X"]) ** 2 + (r1["Y"] - r2["Y"]) ** 2)
                if dist <= agreement_distance:
                    merged = _average_trajectory_rows(r1, r2)
                    frame_classifications[frame] = ("agree", merged)
                else:
                    frame_classifications[frame] = ("disagree", (r1, r2))
        elif frame in frames1:
            frame_classifications[frame] = ("t1_only", t1_by_frame[frame])
        else:
            frame_classifications[frame] = ("t2_only", t2_by_frame[frame])

    # Build trajectory segments using state machine
    # State: "merged" = building single merged segment
    #        "split" = building two parallel segments (after disagreement)
    result_segments = []
    state = "merged"
    current_segment = []
    split_t1_segment = []
    split_t2_segment = []

    for frame in all_frames:
        classification, data = frame_classifications[frame]

        if state == "merged":
            if classification == "agree":
                current_segment.append(data)
            elif classification == "t1_only":
                current_segment.append(data)
            elif classification == "t2_only":
                current_segment.append(data)
            elif classification == "disagree":
                # End current merged segment and start split
                if len(current_segment) >= min_length:
                    result_segments.append(pd.DataFrame(current_segment))
                current_segment = []

                # Start split segments
                state = "split"
                r1, r2 = data
                split_t1_segment = [r1.copy()]
                split_t2_segment = [r2.copy()]

        elif state == "split":
            if classification == "agree":
                # End split, save segments, start new merged segment
                if len(split_t1_segment) >= min_length:
                    result_segments.append(pd.DataFrame(split_t1_segment))
                if len(split_t2_segment) >= min_length:
                    result_segments.append(pd.DataFrame(split_t2_segment))
                split_t1_segment = []
                split_t2_segment = []

                state = "merged"
                current_segment = [data]

            elif classification == "disagree":
                r1, r2 = data
                split_t1_segment.append(r1.copy())
                split_t2_segment.append(r2.copy())

            elif classification == "t1_only":
                split_t1_segment.append(data.copy())
                # t2 segment gets a gap (handled by frame continuity later)

            elif classification == "t2_only":
                split_t2_segment.append(data.copy())
                # t1 segment gets a gap

    # Finalize remaining segments
    if state == "merged":
        if len(current_segment) >= min_length:
            result_segments.append(pd.DataFrame(current_segment))
    elif state == "split":
        if len(split_t1_segment) >= min_length:
            result_segments.append(pd.DataFrame(split_t1_segment))
        if len(split_t2_segment) >= min_length:
            result_segments.append(pd.DataFrame(split_t2_segment))

    # Further split any segments with large gaps OR spatial jumps
    max_spatial_jump = agreement_distance * 5  # ~50px for 9.62 agreement_distance
    final_segments = []
    for seg in result_segments:
        if seg.empty:
            continue
        sub_segments = _split_dataframe_into_segments(
            seg, max_gap=5, max_spatial_jump=max_spatial_jump
        )
        final_segments.extend(sub_segments)

    return final_segments


def _average_trajectory_rows(r1, r2):
    """
    Average two trajectory row dictionaries.
    For positions: average X, Y. For theta: circular mean.
    For confidence: take max. For uncertainty: take min.
    For state: prefer 'active'.
    """
    result = r1.copy()

    # Average positions
    if not pd.isna(r1.get("X")) and not pd.isna(r2.get("X")):
        result["X"] = (r1["X"] + r2["X"]) / 2
        result["Y"] = (r1["Y"] + r2["Y"]) / 2
    elif pd.isna(r1.get("X")):
        result["X"] = r2.get("X")
        result["Y"] = r2.get("Y")

    # Average theta (circular)
    if "Theta" in r1 and "Theta" in r2:
        t1, t2 = r1.get("Theta"), r2.get("Theta")
        if not pd.isna(t1) and not pd.isna(t2):
            result["Theta"] = _circular_mean([t1, t2])
        elif pd.isna(t1):
            result["Theta"] = t2

    # For confidence metrics, take the max (more informative)
    for col in ["DetectionConfidence", "AssignmentConfidence"]:
        if col in r1 and col in r2:
            v1, v2 = r1.get(col), r2.get(col)
            if pd.isna(v1):
                result[col] = v2
            elif pd.isna(v2):
                result[col] = v1
            else:
                result[col] = max(v1, v2)

    # For uncertainty, take the min (lower = more confident)
    if "PositionUncertainty" in r1 and "PositionUncertainty" in r2:
        v1, v2 = r1.get("PositionUncertainty"), r2.get("PositionUncertainty")
        if pd.isna(v1):
            result["PositionUncertainty"] = v2
        elif pd.isna(v2):
            result["PositionUncertainty"] = v1
        else:
            result["PositionUncertainty"] = min(v1, v2)

    # State: prefer "active" over "occluded"/"lost"
    if "State" in r1 and "State" in r2:
        s1, s2 = r1.get("State", "active"), r2.get("State", "active")
        if s1 == "active" or s2 == "active":
            result["State"] = "active"
        else:
            result["State"] = s1

    return result


def _remove_spatially_redundant_trajectories(
    trajectories, agreement_distance, min_overlap
):
    """
    Remove trajectories that are spatially redundant (covered by another trajectory).

    A trajectory B is considered redundant relative to trajectory A if:
    - B has significant frame overlap with A
    - At those overlapping frames, B's positions are within agreement_distance of A's

    When redundancy is detected, we keep the LONGER trajectory and remove the shorter one.
    This handles cases where an "unused" trajectory is actually a subset of a merged one.

    OPTIMIZED: Uses NumPy arrays instead of iterrows() for speed.
    """
    if not trajectories:
        return trajectories

    # Sort by length (longest first) - longer trajectories are preferred
    sorted_trajs = sorted(enumerate(trajectories), key=lambda x: -len(x[1]))

    # Track which trajectories to keep
    redundant_indices = set()

    # Pre-extract arrays for all trajectories
    traj_arrays = []
    for idx, traj in sorted_trajs:
        frames = traj["FrameID"].values
        x = traj["X"].values
        y = traj["Y"].values
        # Create frame -> position lookup using dict comprehension on arrays
        valid_mask = ~np.isnan(x)
        frame_to_pos = {
            frames[i]: (x[i], y[i]) for i in range(len(frames)) if valid_mask[i]
        }
        traj_arrays.append((idx, frame_to_pos, np.sum(valid_mask)))

    for i, (idx_a, a_by_frame, _) in enumerate(traj_arrays):
        if idx_a in redundant_indices:
            continue

        # Check all shorter trajectories against this one
        for idx_b, b_by_frame, total_b_frames in traj_arrays[i + 1 :]:
            if idx_b in redundant_indices:
                continue

            if total_b_frames == 0:
                continue

            # Count overlapping frames where positions agree
            agreeing_frames = 0
            for frame, (bx, by) in b_by_frame.items():
                if frame in a_by_frame:
                    ax, ay = a_by_frame[frame]
                    dist = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
                    if dist <= agreement_distance:
                        agreeing_frames += 1

            # If most of B's frames agree with A, B is redundant
            # We use a high threshold (70%) to be conservative
            if agreeing_frames >= min(min_overlap, total_b_frames):
                agreement_ratio = agreeing_frames / total_b_frames
                if agreement_ratio >= 0.7:
                    redundant_indices.add(idx_b)
                    logger.debug(
                        f"Marking trajectory as redundant: {agreeing_frames}/{total_b_frames} "
                        f"({agreement_ratio:.1%}) frames agree"
                    )

    # Return non-redundant trajectories
    result = [t for i, t in enumerate(trajectories) if i not in redundant_indices]
    if redundant_indices:
        logger.info(
            f"Removed {len(redundant_indices)} spatially redundant trajectories"
        )

    return result


def _merge_overlapping_agreeing_trajectories_old(
    trajectories, agreement_distance, min_overlap, min_length
):
    """
    Merge trajectories that overlap in time and agree spatially.

    CONSERVATIVE approach: Only merge the frames that actually agree.
    Non-overlapping parts that are spatially distant become separate trajectories.

    For two overlapping trajectories:
    - Frames where BOTH agree (within distance threshold): Merge into averaged position
    - Frames where only ONE exists: Keep, but split if there's a spatial discontinuity
    - Frames where BOTH exist but DISAGREE: Each becomes its own trajectory segment

    CRITICAL: Also checks spatial continuity when adding single-source frames.
    If an a_only or b_only frame is too far from the current segment, start a new segment.

    Uses iterative processing until no more merges are possible.
    """
    if not trajectories:
        return trajectories

    # Spatial jump threshold - if a frame is this far from previous, break the segment
    max_spatial_jump = agreement_distance * 5  # ~50px for 9.62 agreement_distance

    def get_last_position(segment):
        """Get the (X, Y) of the last frame in a segment."""
        if not segment:
            return None, None
        last = segment[-1]
        return last.get("X"), last.get("Y")

    def is_spatially_continuous(segment, new_row, threshold):
        """Check if new_row is spatially close to the end of segment."""
        if not segment:
            return True
        last_x, last_y = get_last_position(segment)
        if last_x is None or pd.isna(last_x):
            return True
        new_x, new_y = new_row.get("X"), new_row.get("Y")
        if new_x is None or pd.isna(new_x):
            return True
        dist = np.sqrt((new_x - last_x) ** 2 + (new_y - last_y) ** 2)
        return dist <= threshold

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []

        # Build lookup for each trajectory (OPTIMIZED: use NumPy arrays)
        traj_lookups = []
        for traj in trajectories:
            frames = traj["FrameID"].values
            x = traj["X"].values
            valid_mask = ~np.isnan(x)

            # Build lookup dict from arrays - faster than iterrows
            lookup = {}
            for idx in np.where(valid_mask)[0]:
                row_dict = {col: traj.iloc[idx][col] for col in traj.columns}
                lookup[frames[idx]] = row_dict
            traj_lookups.append(lookup)

        for i in range(len(trajectories)):
            if i in used:
                continue

            traj_a = trajectories[i]
            lookup_a = traj_lookups[i]

            for j in range(i + 1, len(trajectories)):
                if j in used:
                    continue

                traj_b = trajectories[j]
                lookup_b = traj_lookups[j]

                # Find overlapping frames
                common_frames = set(lookup_a.keys()).intersection(set(lookup_b.keys()))
                if len(common_frames) < min_overlap:
                    continue

                # Count agreeing frames in the overlap
                agreeing = 0
                for frame in common_frames:
                    ra, rb = lookup_a[frame], lookup_b[frame]
                    dist = np.sqrt((ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2)
                    if dist <= agreement_distance:
                        agreeing += 1

                # Only process if there's significant agreement
                if agreeing < min_overlap:
                    continue

                # CONSERVATIVE MERGE with spatial continuity checks
                all_frames_set = set(lookup_a.keys()).union(set(lookup_b.keys()))
                all_frames = sorted(all_frames_set)

                # Classify frames
                frame_classifications = {}
                for frame in all_frames:
                    in_a = frame in lookup_a
                    in_b = frame in lookup_b

                    if in_a and in_b:
                        ra, rb = lookup_a[frame], lookup_b[frame]
                        dist = np.sqrt(
                            (ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2
                        )
                        if dist <= agreement_distance:
                            merged = _average_trajectory_rows(ra, rb)
                            frame_classifications[frame] = ("agree", merged)
                        else:
                            frame_classifications[frame] = ("disagree", (ra, rb))
                    elif in_a:
                        frame_classifications[frame] = ("a_only", lookup_a[frame])
                    else:
                        frame_classifications[frame] = ("b_only", lookup_b[frame])

                # Build segments using state machine WITH spatial continuity checks
                result_segments = []
                current_segment = []

                for frame in all_frames:
                    classification, data = frame_classifications[frame]

                    if classification == "agree":
                        # Check spatial continuity
                        if is_spatially_continuous(
                            current_segment, data, max_spatial_jump
                        ):
                            current_segment.append(data)
                        else:
                            # Spatial discontinuity - save current and start new
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = [data]

                    elif classification in ("a_only", "b_only"):
                        # Single-source frame: only add if spatially continuous
                        if is_spatially_continuous(
                            current_segment, data, max_spatial_jump
                        ):
                            current_segment.append(data)
                        else:
                            # Spatially discontinuous - save current, start new
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = [data]

                    elif classification == "disagree":
                        # Both exist but disagree - split into separate trajectories
                        ra, rb = data

                        # Check which (if any) is continuous with current segment
                        a_continuous = is_spatially_continuous(
                            current_segment, ra, max_spatial_jump
                        )
                        b_continuous = is_spatially_continuous(
                            current_segment, rb, max_spatial_jump
                        )

                        if a_continuous and not b_continuous:
                            # Keep a in current segment, start new for b if long enough later
                            current_segment.append(ra.copy())
                        elif b_continuous and not a_continuous:
                            # Keep b in current segment
                            current_segment.append(rb.copy())
                        elif a_continuous and b_continuous:
                            # Both continuous - this shouldn't happen if they disagree
                            # but if it does, just save current and start fresh
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = []
                        else:
                            # Neither continuous - save current, drop both as noise
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = []

                # Finalize remaining segment
                if len(current_segment) >= min_length:
                    result_segments.append(pd.DataFrame(current_segment))

                # If we produced any segments, mark both as used
                if result_segments:
                    used.add(i)
                    used.add(j)
                    new_trajectories.extend(result_segments)
                    merged_any = True
                    break

            # If trajectory i wasn't merged with anything, keep it
            if i not in used:
                if len(traj_a) >= min_length:
                    new_trajectories.append(traj_a)
                used.add(i)

        trajectories = new_trajectories

        if not merged_any:
            break

    if iteration > 1:
        logger.info(f"Processed overlapping trajectories in {iteration} iterations")

    return trajectories


def _merge_overlapping_agreeing_trajectories(
    trajectories, agreement_distance, min_overlap, min_length
):
    """
    Merge trajectories that overlap in time and agree spatially or share DetectionIDs.

    ENHANCED with DetectionID support for deterministic merging:
    - Trajectories sharing DetectionIDs are guaranteed to be the same animal
    - DetectionID matches override spatial discontinuity checks
    - Falls back to spatial agreement for frames without DetectionID

    CONSERVATIVE approach: Only merge the frames that actually agree.
    Non-overlapping parts that are spatially distant become separate trajectories.

    For two overlapping trajectories:
    - Frames where BOTH share same DetectionID: MUST merge (same detection)
    - Frames where BOTH agree spatially (within distance threshold): Merge into averaged position
    - Frames where only ONE exists: Keep, but split if there's a spatial discontinuity
    - Frames where BOTH exist but DISAGREE: Each becomes its own trajectory segment

    Uses iterative processing until no more merges are possible.
    """
    if not trajectories:
        return trajectories

    # Check if DetectionID column is available
    has_detection_id = (
        "DetectionID" in trajectories[0].columns if trajectories else False
    )

    # Spatial jump threshold - if a frame is this far from previous, break the segment
    # UNLESS DetectionID confirms continuity
    max_spatial_jump = agreement_distance * 5  # ~50px for 9.62 agreement_distance

    def get_last_position(segment):
        """Get the (X, Y) of the last frame in a segment."""
        if not segment:
            return None, None
        last = segment[-1]
        return last.get("X"), last.get("Y")

    def is_spatially_continuous(segment, new_row, threshold, check_detection_id=True):
        """
        Check if new_row is spatially close to the end of segment.

        If both have valid DetectionID and they match consecutive detections,
        spatial continuity is enforced more strictly. If DetectionID suggests
        a jump in detection sequence, we're more lenient with spatial distance.
        """
        if not segment:
            return True
        last_x, last_y = get_last_position(segment)
        if last_x is None or pd.isna(last_x):
            return True
        new_x, new_y = new_row.get("X"), new_row.get("Y")
        if new_x is None or pd.isna(new_x):
            return True
        dist = np.sqrt((new_x - last_x) ** 2 + (new_y - last_y) ** 2)

        # If DetectionID is available and both rows have it, use it to inform continuity
        if has_detection_id and check_detection_id:
            last_det_id = segment[-1].get("DetectionID")
            new_det_id = new_row.get("DetectionID")

            # If both have valid DetectionIDs (not NaN)
            if (
                last_det_id is not None
                and not pd.isna(last_det_id)
                and new_det_id is not None
                and not pd.isna(new_det_id)
            ):
                # If DetectionIDs differ, it's a different detection - be lenient with spatial distance
                # This handles cases where tracking switched between detections
                if last_det_id != new_det_id:
                    # Allow larger spatial jumps when DetectionID changes
                    return dist <= threshold * 2

        return dist <= threshold

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []

        # Build lookup for each trajectory (OPTIMIZED: use NumPy arrays)
        traj_lookups = []
        for traj in trajectories:
            frames = traj["FrameID"].values
            x = traj["X"].values
            valid_mask = ~np.isnan(x)

            # Build lookup dict from arrays - faster than iterrows
            lookup = {}
            for idx in np.where(valid_mask)[0]:
                row_dict = {col: traj.iloc[idx][col] for col in traj.columns}
                lookup[frames[idx]] = row_dict
            traj_lookups.append(lookup)

        for i in range(len(trajectories)):
            if i in used:
                continue

            traj_a = trajectories[i]
            lookup_a = traj_lookups[i]

            for j in range(i + 1, len(trajectories)):
                if j in used:
                    continue

                traj_b = trajectories[j]
                lookup_b = traj_lookups[j]

                # Find overlapping frames
                common_frames = set(lookup_a.keys()).intersection(set(lookup_b.keys()))
                if len(common_frames) < min_overlap:
                    continue

                # Count agreeing frames in the overlap
                # Enhanced: Also count DetectionID matches
                agreeing = 0
                detection_id_matches = 0
                for frame in common_frames:
                    ra, rb = lookup_a[frame], lookup_b[frame]

                    # Check DetectionID match first (strongest evidence)
                    if has_detection_id:
                        det_a = ra.get("DetectionID")
                        det_b = rb.get("DetectionID")
                        # If both have valid DetectionIDs and they match, it's the same detection
                        if (
                            det_a is not None
                            and not pd.isna(det_a)
                            and det_b is not None
                            and not pd.isna(det_b)
                            and det_a == det_b
                        ):
                            detection_id_matches += 1
                            agreeing += 1
                            continue

                    # Fall back to spatial agreement
                    dist = np.sqrt((ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2)
                    if dist <= agreement_distance:
                        agreeing += 1

                # Only process if there's significant agreement
                # DetectionID matches are stronger evidence, so require fewer matches
                required_matches = (
                    min_overlap
                    if detection_id_matches < 2
                    else max(2, min_overlap // 2)
                )
                if agreeing < required_matches:
                    continue

                # Log if we found DetectionID matches
                if detection_id_matches > 0:
                    logger.debug(
                        f"Found {detection_id_matches} DetectionID matches out of {len(common_frames)} overlapping frames"
                    )

                # CONSERVATIVE MERGE with spatial continuity checks
                all_frames_set = set(lookup_a.keys()).union(set(lookup_b.keys()))
                all_frames = sorted(all_frames_set)

                # Classify frames
                frame_classifications = {}
                for frame in all_frames:
                    in_a = frame in lookup_a
                    in_b = frame in lookup_b

                    if in_a and in_b:
                        ra, rb = lookup_a[frame], lookup_b[frame]

                        # Check DetectionID match first (strongest evidence of same animal)
                        detection_id_match = False
                        if has_detection_id:
                            det_a = ra.get("DetectionID")
                            det_b = rb.get("DetectionID")
                            if (
                                det_a is not None
                                and not pd.isna(det_a)
                                and det_b is not None
                                and not pd.isna(det_b)
                                and det_a == det_b
                            ):
                                detection_id_match = True

                        # If DetectionID matches, they MUST agree (same physical detection)
                        if detection_id_match:
                            merged = _average_trajectory_rows(ra, rb)
                            frame_classifications[frame] = ("agree_detid", merged)
                        else:
                            # Fall back to spatial agreement
                            dist = np.sqrt(
                                (ra["X"] - rb["X"]) ** 2 + (ra["Y"] - rb["Y"]) ** 2
                            )
                            if dist <= agreement_distance:
                                merged = _average_trajectory_rows(ra, rb)
                                frame_classifications[frame] = ("agree", merged)
                            else:
                                frame_classifications[frame] = ("disagree", (ra, rb))
                    elif in_a:
                        frame_classifications[frame] = ("a_only", lookup_a[frame])
                    else:
                        frame_classifications[frame] = ("b_only", lookup_b[frame])

                # Build segments using state machine WITH spatial continuity checks
                # ENHANCED: DetectionID matches bypass spatial continuity checks
                result_segments = []
                current_segment = []
                # Keep disagreeing observations separate by source to avoid FrameID duplicates
                disagree_a_observations = []  # From trajectory A
                disagree_b_observations = []  # From trajectory B

                for frame in all_frames:
                    classification, data = frame_classifications[frame]

                    if classification == "agree_detid":
                        # DetectionID match - ALWAYS continuous (same physical detection)
                        # This overrides spatial discontinuity
                        current_segment.append(data)

                    elif classification == "agree":
                        # Spatial agreement without DetectionID confirmation
                        # Check spatial continuity
                        if is_spatially_continuous(
                            current_segment, data, max_spatial_jump
                        ):
                            current_segment.append(data)
                        else:
                            # Spatial discontinuity - save current and start new
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = [data]

                    elif classification in ("a_only", "b_only"):
                        # Single-source frame: be LENIENT with spatial continuity
                        # These frames don't have disagreement, they're just continuing one trajectory
                        # Only break if the jump is VERY large (indicates genuine trajectory switch)
                        lenient_threshold = (
                            max_spatial_jump  # 3x more lenient for single-source
                        )
                        if is_spatially_continuous(
                            current_segment,
                            data,
                            lenient_threshold,
                            check_detection_id=False,
                        ):
                            current_segment.append(data)
                        else:
                            # VERY large spatial jump - likely different animal
                            if len(current_segment) >= min_length:
                                result_segments.append(pd.DataFrame(current_segment))
                            current_segment = [data]

                    elif classification == "disagree":
                        # Both exist but disagree - check if one is occluded
                        ra, rb = data

                        # Check state of each observation
                        state_a = ra.get("State", "active")
                        state_b = rb.get("State", "active")

                        # If one is active and the other is occluded/lost, prefer the active one
                        # Don't create separate trajectories for predicted vs detected positions
                        if state_a == "active" and state_b in ("occluded", "lost"):
                            # Prefer A (actual detection), discard B (prediction)
                            lenient_threshold = max_spatial_jump * 2
                            if is_spatially_continuous(
                                current_segment,
                                ra,
                                lenient_threshold,
                                check_detection_id=False,
                            ):
                                current_segment.append(ra.copy())
                            else:
                                if len(current_segment) >= min_length:
                                    result_segments.append(
                                        pd.DataFrame(current_segment)
                                    )
                                current_segment = [ra.copy()]
                            # B is discarded (it's just a prediction during occlusion)

                        elif state_b == "active" and state_a in ("occluded", "lost"):
                            # Prefer B (actual detection), discard A (prediction)
                            lenient_threshold = max_spatial_jump * 2
                            if is_spatially_continuous(
                                current_segment,
                                rb,
                                lenient_threshold,
                                check_detection_id=False,
                            ):
                                current_segment.append(rb.copy())
                            else:
                                if len(current_segment) >= min_length:
                                    result_segments.append(
                                        pd.DataFrame(current_segment)
                                    )
                                current_segment = [rb.copy()]
                            # A is discarded (it's just a prediction during occlusion)

                        else:
                            # Both active or both occluded - genuine disagreement
                            # Preserve BOTH as separate trajectories
                            lenient_threshold = max_spatial_jump * 2
                            a_continuous = is_spatially_continuous(
                                current_segment,
                                ra,
                                lenient_threshold,
                                check_detection_id=False,
                            )
                            b_continuous = is_spatially_continuous(
                                current_segment,
                                rb,
                                lenient_threshold,
                                check_detection_id=False,
                            )

                            if a_continuous and not b_continuous:
                                # Keep a in current segment, save b for separate trajectory
                                current_segment.append(ra.copy())
                                disagree_b_observations.append(rb.copy())
                            elif b_continuous and not a_continuous:
                                # Keep b in current segment, save a for separate trajectory
                                current_segment.append(rb.copy())
                                disagree_a_observations.append(ra.copy())
                            elif a_continuous and b_continuous:
                                # Both could continue - ambiguous case
                                # NEW: Prefer the one minimizing spatial jump (smoother path) instead of splitting
                                last_x, last_y = get_last_position(current_segment)

                                if last_x is None or pd.isna(last_x):
                                    # Start of segment or last position unknown, prefer A
                                    current_segment.append(ra.copy())
                                    disagree_b_observations.append(rb.copy())
                                else:
                                    # Handle potential NaNs in current observations
                                    xa, ya = ra.get("X"), ra.get("Y")
                                    xb, yb = rb.get("X"), rb.get("Y")

                                    dist_a = float("inf")
                                    dist_b = float("inf")

                                    if not pd.isna(xa):
                                        dist_a = np.sqrt(
                                            (xa - last_x) ** 2 + (ya - last_y) ** 2
                                        )
                                    if not pd.isna(xb):
                                        dist_b = np.sqrt(
                                            (xb - last_x) ** 2 + (yb - last_y) ** 2
                                        )

                                    # If both have valid distances, compare
                                    if dist_a <= dist_b:
                                        current_segment.append(ra.copy())
                                        disagree_b_observations.append(rb.copy())
                                    else:
                                        current_segment.append(rb.copy())
                                        disagree_a_observations.append(ra.copy())
                            else:
                                # Neither continuous - save current and preserve both separately
                                if len(current_segment) >= min_length:
                                    result_segments.append(
                                        pd.DataFrame(current_segment)
                                    )
                                current_segment = []
                                disagree_a_observations.append(ra.copy())
                                disagree_b_observations.append(rb.copy())

                # Finalize remaining segment
                if len(current_segment) >= min_length:
                    result_segments.append(pd.DataFrame(current_segment))

                # Convert disagree observations to separate trajectories
                # Process each source separately to avoid FrameID duplicates
                def build_disagree_trajectory(obs_list, source_name):
                    """Build trajectory segments from a list of disagree observations."""
                    if not obs_list:
                        return []

                    segments = []
                    # Sort by FrameID
                    obs_list.sort(key=lambda x: x.get("FrameID", 0))

                    # Build trajectory segments
                    current_seg = []
                    for obs in obs_list:
                        if not current_seg:
                            current_seg = [obs]
                        else:
                            last_frame = current_seg[-1].get("FrameID", 0)
                            curr_frame = obs.get("FrameID", 0)

                            # Skip if same frame (shouldn't happen but safety check)
                            if curr_frame == last_frame:
                                continue

                            # If frames are close in time and space, continue segment
                            if curr_frame - last_frame <= 5:  # Within 5 frames
                                if is_spatially_continuous(
                                    current_seg,
                                    obs,
                                    max_spatial_jump * 3,
                                    check_detection_id=False,
                                ):
                                    current_seg.append(obs)
                                else:
                                    # Spatially discontinuous - save and start new
                                    if len(current_seg) >= min_length:
                                        segments.append(pd.DataFrame(current_seg))
                                    current_seg = [obs]
                            else:
                                # Temporal gap - save current and start new
                                if len(current_seg) >= min_length:
                                    segments.append(pd.DataFrame(current_seg))
                                current_seg = [obs]

                    # Finalize last segment
                    if len(current_seg) >= min_length:
                        segments.append(pd.DataFrame(current_seg))

                    return segments

                # Build trajectories from disagreeing observations (kept separate by source)
                disagree_segs_a = build_disagree_trajectory(
                    disagree_a_observations, "A"
                )
                disagree_segs_b = build_disagree_trajectory(
                    disagree_b_observations, "B"
                )
                result_segments.extend(disagree_segs_a)
                result_segments.extend(disagree_segs_b)

                # If we produced any segments, mark both as used
                if result_segments:
                    used.add(i)
                    used.add(j)
                    new_trajectories.extend(result_segments)
                    merged_any = True

                    # Log merge results
                    total_merged_frames = sum(len(seg) for seg in result_segments)
                    disagree_count = len(disagree_a_observations) + len(
                        disagree_b_observations
                    )
                    logger.debug(
                        f"Merged trajectories {i} ({len(lookup_a)} frames) + {j} ({len(lookup_b)} frames) "
                        f"→ {len(result_segments)} segments with {total_merged_frames} total frames "
                        f"(DetectionID matches: {detection_id_matches}, disagree preserved: {disagree_count})"
                    )
                    break

            # If trajectory i wasn't merged with anything, keep it
            if i not in used:
                if len(traj_a) >= min_length:
                    new_trajectories.append(traj_a)
                used.add(i)

        trajectories = new_trajectories

        if not merged_any:
            break

    if iteration > 1:
        logger.info(f"Processed overlapping trajectories in {iteration} iterations")

    return trajectories


def _stitch_broken_trajectory_fragments(trajectories, agreement_distance, max_gap=2):
    """
    Stitch together trajectories that are likely fragmented parts of the same track.

    This handles sequential fragments where T2 starts shortly after T1 ends,
    and they are spatially close (within agreement_distance).

    Args:
        trajectories: List of trajectory DataFrames
        agreement_distance: Max spatial distance to allow stitching
        max_gap: Max frame gap between end of T1 and start of T2
    """
    if not trajectories:
        return trajectories

    # Iterative stitching
    max_iterations = 20
    iteration = 0
    current_trajectories = trajectories

    while iteration < max_iterations:
        iteration += 1
        merged_any = False
        used = set()
        new_trajectories = []

        # Sort by start frame for efficient searching
        sorted_indices = sorted(
            range(len(current_trajectories)),
            key=lambda i: current_trajectories[i].iloc[0]["FrameID"],
        )

        # Build lookup for start/end info
        traj_info = {}
        for idx in sorted_indices:
            df = current_trajectories[idx]
            if df.empty:
                continue
            traj_info[idx] = {
                "start_frame": df.iloc[0]["FrameID"],
                "end_frame": df.iloc[-1]["FrameID"],
                "start_pos": (df.iloc[0]["X"], df.iloc[0]["Y"]),
                "end_pos": (df.iloc[-1]["X"], df.iloc[-1]["Y"]),
                "df": df,
            }

        idx_ptr = 0
        while idx_ptr < len(sorted_indices):
            idx_a = sorted_indices[idx_ptr]

            if idx_a in used:
                idx_ptr += 1
                continue

            info_a = traj_info[idx_a]
            best_idx_b = -1
            min_dist = float("inf")

            # Look ahead for stitch candidates
            for idx_next in range(idx_ptr + 1, len(sorted_indices)):
                idx_b = sorted_indices[idx_next]
                if idx_b in used:
                    continue

                info_b = traj_info[idx_b]

                # Check frame gap
                gap = info_b["start_frame"] - info_a["end_frame"]

                if gap <= 0:
                    continue  # Overlapping (handled by other function)

                if gap > max_gap:
                    break  # Too far ahead (list is sorted)

                # Check spatial distance
                ax, ay = info_a["end_pos"]
                bx, by = info_b["start_pos"]

                if pd.isna(ax) or pd.isna(bx):
                    continue

                dist = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

                if dist <= agreement_distance:
                    if dist < min_dist:
                        min_dist = dist
                        best_idx_b = idx_b

            if best_idx_b != -1:
                # Merge sequential fragments
                merged = pd.concat(
                    [info_a["df"], traj_info[best_idx_b]["df"]], ignore_index=True
                )
                new_trajectories.append(merged)
                used.add(idx_a)
                used.add(best_idx_b)
                merged_any = True
                logger.debug(
                    f"Stitched fragment {idx_a} -> {best_idx_b} (gap: {traj_info[best_idx_b]['start_frame'] - info_a['end_frame']} frames, dist: {min_dist:.2f}px)"
                )
            else:
                new_trajectories.append(info_a["df"])
                used.add(idx_a)

            idx_ptr += 1

        current_trajectories = new_trajectories
        if not merged_any:
            break

    if iteration > 1:
        logger.info(f"Stitched broken fragments in {iteration} iterations")

    return current_trajectories


def _convert_trajectory_to_dataframe(traj, traj_id):
    """Convert trajectory from list of tuples to DataFrame format."""
    if isinstance(traj, pd.DataFrame):
        df = traj.copy()
        if "TrajectoryID" not in df.columns:
            df["TrajectoryID"] = traj_id
        # Ensure required columns exist
        if "State" not in df.columns:
            df["State"] = "active"
        return df
    else:
        # Convert from list of tuples format
        data = []
        for point in traj:
            if len(point) >= 4:
                x, y, theta, frame = point[:4]
                data.append(
                    {
                        "TrajectoryID": traj_id,
                        "X": int(x) if not np.isnan(x) else x,
                        "Y": int(y) if not np.isnan(y) else y,
                        "Theta": theta,
                        "FrameID": int(frame),
                        "State": "active",
                    }
                )
        return pd.DataFrame(data)


def _calculate_trajectory_distance_matrix(
    traj_list_1,
    traj_list_2,
    frame_id_col="FrameID",
    x_col="X",
    y_col="Y",
    summary_func=np.mean,
):
    """
    Calculate distance matrix between two lists of trajectory DataFrames.

    Parameters:
    -----------
    traj_list_1, traj_list_2 : list of pandas.DataFrame
        Lists containing individual trajectory DataFrames
    frame_id_col : str
        Column name for frame IDs
    x_col, y_col : str
        Column names for X and Y positions
    summary_func : callable
        Function to summarize distances (e.g., np.mean, np.median, lambda x: np.percentile(x, 75))

    Returns:
    --------
    dist_matrix : numpy.ndarray
        Distance matrix where dist_matrix[i,j] is the distance between trajectory i from list 1 and trajectory j from list 2
    """
    # Initialize distance matrix
    dist_matrix = np.full((len(traj_list_1), len(traj_list_2)), np.inf)

    # Calculate distances efficiently
    for i, traj_1 in enumerate(traj_list_1):
        # Filter out rows with NaN in X or Y positions
        traj_1_valid = traj_1.dropna(subset=[x_col, y_col])
        frameids_1 = set(traj_1_valid[frame_id_col])

        # Pre-compute trajectory 1 positions indexed by FrameID
        pos_dict_1 = dict(
            zip(
                traj_1_valid[frame_id_col],
                zip(traj_1_valid[x_col], traj_1_valid[y_col]),
            )
        )

        for j, traj_2 in enumerate(traj_list_2):
            # Filter out rows with NaN in X or Y positions
            traj_2_valid = traj_2.dropna(subset=[x_col, y_col])
            frameids_2 = set(traj_2_valid[frame_id_col])

            # Check temporal overlap
            common_frameids = frameids_1.intersection(frameids_2)

            if not common_frameids:
                continue  # Keep as inf (no overlap)

            # Vectorized distance calculation for overlapping frames
            positions_1 = []
            positions_2 = []

            # Pre-compute trajectory 2 positions for common frames
            traj_2_common = traj_2_valid[
                traj_2_valid[frame_id_col].isin(common_frameids)
            ]
            pos_dict_2 = dict(
                zip(
                    traj_2_common[frame_id_col],
                    zip(traj_2_common[x_col], traj_2_common[y_col]),
                )
            )

            # Collect positions for common frames
            for frameid in common_frameids:
                if frameid in pos_dict_1 and frameid in pos_dict_2:
                    positions_1.append(pos_dict_1[frameid])
                    positions_2.append(pos_dict_2[frameid])

            if positions_1:
                # Convert to numpy arrays and calculate distances vectorized
                pos_array_1 = np.array(positions_1)
                pos_array_2 = np.array(positions_2)

                # Calculate Euclidean distances for all frame pairs at once
                distances = np.linalg.norm(pos_array_1 - pos_array_2, axis=1)
                dist_matrix[i, j] = summary_func(distances)

    return dist_matrix


def _circular_mean(values):
    """Calculate the circular mean of a list of angles in radians."""
    sin_sum = np.sum(np.sin(values))
    cos_sum = np.sum(np.cos(values))
    return np.arctan2(sin_sum, cos_sum) % (2 * np.pi)


def _clean_trajectory(traj_df):
    """
    Clean a trajectory DataFrame by:
    1. Removing leading 'lost' and 'occluded' states
    2. Removing trailing 'lost' and 'occluded' states
    3. Returning None if no valid detections remain

    Args:
        traj_df: DataFrame with trajectory data

    Returns:
        Cleaned DataFrame or None if trajectory has no valid detections
    """
    if traj_df.empty:
        return None

    # Check if State column exists
    if "State" not in traj_df.columns:
        return traj_df

    # Check if all states are occluded/lost (no valid detections)
    valid_states = ~traj_df["State"].isin(["occluded", "lost"])
    if not valid_states.any():
        return None

    # Find first and last valid detections
    valid_indices = traj_df[valid_states].index
    first_valid_idx = valid_indices[0]
    last_valid_idx = valid_indices[-1]

    # Trim trajectory to valid detection range
    cleaned_df = traj_df.loc[first_valid_idx:last_valid_idx].copy()

    return cleaned_df


def _clean_trajectories(traj_list, min_length=5):
    """
    Clean a list of trajectory DataFrames by removing useless ones.

    Args:
        traj_list: List of trajectory DataFrames
        min_length: Minimum length for valid trajectories

    Returns:
        List of cleaned trajectories
    """
    cleaned = []
    for traj in traj_list:
        cleaned_traj = _clean_trajectory(traj)
        if cleaned_traj is not None and len(cleaned_traj) >= min_length:
            cleaned.append(cleaned_traj)
    return cleaned


def _merge_trajectories(traj1, traj2, distance_threshold=None):
    """
    Combine two trajectories into one, with improved handling of ID swaps.
    Returns list of DataFrames instead of dictionaries.
    Preserves all columns including confidence metrics.
    """
    # Identify confidence columns
    confidence_cols = [
        col
        for col in traj1.columns
        if col.lower().endswith("confidence") or col == "PositionUncertainty"
    ]

    # Create row dictionaries indexed by FrameID for full data access
    traj1_rows = {row["FrameID"]: row for _, row in traj1.iterrows()}
    traj2_rows = {row["FrameID"]: row for _, row in traj2.iterrows()}

    # Find frame relationships
    frames1 = set(traj1["FrameID"])
    frames2 = set(traj2["FrameID"])
    common_frames = frames1.intersection(frames2)

    if not common_frames:
        # No overlap - return both trajectories as separate segments
        return [traj1.copy(), traj2.copy()]

    unique_1 = frames1.difference(common_frames)
    unique_2 = frames2.difference(common_frames)

    # Classify common frames by distance threshold
    good_frames = set()
    bad_frames = set()

    if distance_threshold is not None:
        for frame in common_frames:
            row1 = traj1_rows[frame]
            row2 = traj2_rows[frame]
            distance = np.linalg.norm([row1["X"] - row2["X"], row1["Y"] - row2["Y"]])

            if distance <= distance_threshold:
                good_frames.add(frame)
            else:
                bad_frames.add(frame)
    else:
        good_frames = common_frames.copy()

    # Process mergeable frames (good frames + unique frames)
    mergeable_frames = good_frames.union(unique_1).union(unique_2)

    # Create merged trajectory from mergeable frames, preserving all columns
    merged_rows = []

    for frame in sorted(mergeable_frames):
        if frame in good_frames:
            # Both trajectories have this frame and distance is acceptable
            row1 = traj1_rows[frame]
            row2 = traj2_rows[frame]
            state1 = row1.get("State", "occluded")  # Default to occluded if missing
            state2 = row2.get("State", "occluded")

            # Merge based on states
            merged_row = {"FrameID": frame}

            # If either trajectory is active, the merged state is active
            if state1 == "active" or state2 == "active":
                # Use the active detection, or average if both active
                if state1 == "active" and state2 == "active":
                    merged_row["X"] = (row1["X"] + row2["X"]) / 2
                    merged_row["Y"] = (row1["Y"] + row2["Y"]) / 2
                    merged_row["Theta"] = _circular_mean([row1["Theta"], row2["Theta"]])
                    merged_row["State"] = "active"
                    # Average confidence metrics
                    for col in confidence_cols:
                        if col in row1 and col in row2:
                            val1, val2 = row1[col], row2[col]
                            if pd.notna(val1) and pd.notna(val2):
                                merged_row[col] = (val1 + val2) / 2
                            elif pd.notna(val1):
                                merged_row[col] = val1
                            elif pd.notna(val2):
                                merged_row[col] = val2
                            else:
                                merged_row[col] = np.nan
                elif state1 == "active":
                    merged_row.update(row1.to_dict())
                else:  # state2 == "active"
                    merged_row.update(row2.to_dict())
            else:
                # Both occluded/lost - merged state is occluded, positions set to NaN
                merged_row["X"] = np.nan
                merged_row["Y"] = np.nan
                merged_row["Theta"] = np.nan
                merged_row["State"] = "occluded"
                # Average confidence metrics if available
                for col in confidence_cols:
                    if col in row1 and col in row2:
                        val1, val2 = row1[col], row2[col]
                        if pd.notna(val1) and pd.notna(val2):
                            merged_row[col] = (val1 + val2) / 2
                        elif pd.notna(val1):
                            merged_row[col] = val1
                        elif pd.notna(val2):
                            merged_row[col] = val2
                        else:
                            merged_row[col] = np.nan

        elif frame in unique_1:
            merged_row = traj1_rows[frame].to_dict()
            # Set positions to NaN if occluded/lost
            if merged_row.get("State", "occluded") in ["occluded", "lost"]:
                merged_row["X"] = np.nan
                merged_row["Y"] = np.nan
                merged_row["Theta"] = np.nan
        else:  # frame in unique_2
            merged_row = traj2_rows[frame].to_dict()
            # Set positions to NaN if occluded/lost
            if merged_row.get("State", "occluded") in ["occluded", "lost"]:
                merged_row["X"] = np.nan
                merged_row["Y"] = np.nan
                merged_row["Theta"] = np.nan

        # Ensure confidence columns are present (State should already be set above)
        for col in confidence_cols:
            if col not in merged_row:
                merged_row[col] = np.nan

        merged_rows.append(merged_row)

    # Create main merged trajectory as DataFrame
    all_trajectories = []

    if merged_rows:
        # Convert to DataFrame
        merged_df = (
            pd.DataFrame(merged_rows).sort_values("FrameID").reset_index(drop=True)
        )

        # Split into continuous segments
        segments = _split_dataframe_into_segments(merged_df, max_gap=5)
        all_trajectories.extend(segments)

    # Handle bad frames (create separate trajectories for each animal)
    if bad_frames:
        logger.debug(
            f"Merge created {len(bad_frames)} bad frames - creating separate trajectory segments"
        )
        # Get rows from traj1 for bad frames
        bad_rows_1 = [
            traj1_rows[f].to_dict() for f in sorted(bad_frames) if f in traj1_rows
        ]
        if bad_rows_1:
            bad_df_1 = (
                pd.DataFrame(bad_rows_1).sort_values("FrameID").reset_index(drop=True)
            )
            segments_1 = _split_dataframe_into_segments(bad_df_1, max_gap=5)
            all_trajectories.extend(segments_1)

        # Get rows from traj2 for bad frames
        bad_rows_2 = [
            traj2_rows[f].to_dict() for f in sorted(bad_frames) if f in traj2_rows
        ]
        if bad_rows_2:
            bad_df_2 = (
                pd.DataFrame(bad_rows_2).sort_values("FrameID").reset_index(drop=True)
            )
            segments_2 = _split_dataframe_into_segments(bad_df_2, max_gap=5)
            all_trajectories.extend(segments_2)

    return all_trajectories


def _split_dataframe_into_segments(df, max_gap=5, max_spatial_jump=50.0):
    """
    Split a DataFrame into continuous segments based on FrameID gaps AND spatial jumps.
    Preserves all columns.

    Args:
        df: DataFrame with FrameID, X, Y columns
        max_gap: Maximum allowed frame gap before splitting
        max_spatial_jump: Maximum allowed distance between consecutive frames before splitting
    """
    if df.empty:
        return []

    segments = []
    current_segment = []

    for idx, row in df.iterrows():
        should_split = False

        if current_segment:
            prev_row = current_segment[-1]
            frame_gap = row["FrameID"] - prev_row["FrameID"]

            # Check temporal gap
            if frame_gap > max_gap:
                should_split = True

            # Check spatial jump (only for consecutive or near-consecutive frames)
            if not should_split and frame_gap <= max_gap:
                prev_x, prev_y = prev_row.get("X"), prev_row.get("Y")
                curr_x, curr_y = row.get("X"), row.get("Y")

                # Only check if both positions are valid
                if (
                    not pd.isna(prev_x)
                    and not pd.isna(curr_x)
                    and not pd.isna(prev_y)
                    and not pd.isna(curr_y)
                ):
                    dist = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                    if dist > max_spatial_jump:
                        should_split = True

        if not current_segment or not should_split:
            current_segment.append(row.to_dict())
        else:
            # Save current segment and start new one
            if current_segment:
                segments.append(pd.DataFrame(current_segment))
            current_segment = [row.to_dict()]

    # Add the last segment
    if current_segment:
        segments.append(pd.DataFrame(current_segment))

    return segments


def _split_into_continuous_segments(frame_list, x_list, y_list, theta_list, max_gap=5):
    """Split frame sequence into continuous segments."""
    if not frame_list:
        return []

    segments = []
    sorted_indices = np.argsort(frame_list)

    current_segment = {"FrameID": [], "X": [], "Y": [], "Theta": []}

    for i, idx in enumerate(sorted_indices):
        frame = frame_list[idx]

        # Check if this frame continues the current segment
        if (
            not current_segment["FrameID"]
            or frame <= current_segment["FrameID"][-1] + max_gap
        ):
            current_segment["FrameID"].append(frame)
            current_segment["X"].append(x_list[idx])
            current_segment["Y"].append(y_list[idx])
            current_segment["Theta"].append(theta_list[idx])
        else:
            # Save current segment and start new one
            if current_segment["FrameID"]:
                segments.append(current_segment.copy())

            current_segment = {
                "FrameID": [frame],
                "X": [x_list[idx]],
                "Y": [y_list[idx]],
                "Theta": [theta_list[idx]],
            }

    # Add the last segment
    if current_segment["FrameID"]:
        segments.append(current_segment)

    return segments


def _create_segments_from_frames(frame_list, traj_dict, max_gap=5):
    """Create trajectory segments from a list of frames."""
    if not frame_list:
        return []

    segments = []
    current_segment = {"FrameID": [], "X": [], "Y": [], "Theta": []}

    for frame in frame_list:
        if frame not in traj_dict:
            continue

        x, y, theta, _ = traj_dict[frame]

        # Check if this frame continues the current segment
        if (
            not current_segment["FrameID"]
            or frame <= current_segment["FrameID"][-1] + max_gap
        ):
            current_segment["FrameID"].append(frame)
            current_segment["X"].append(x)
            current_segment["Y"].append(y)
            current_segment["Theta"].append(theta)
        else:
            # Save current segment and start new one
            if current_segment["FrameID"]:
                segments.append(current_segment.copy())

            current_segment = {"FrameID": [frame], "X": [x], "Y": [y], "Theta": [theta]}

    # Add the last segment
    if current_segment["FrameID"]:
        segments.append(current_segment)

    return segments


def interpolate_trajectories(trajectories_df, method="linear", max_gap=10):
    """
    Interpolate missing values in trajectories using various methods.

    Args:
        trajectories_df: DataFrame with trajectory data (must have X, Y, Theta, FrameID columns)
        method: Interpolation method - 'linear', 'cubic', 'spline', or 'none'
        max_gap: Maximum gap size to interpolate (frames). Larger gaps are left as NaN.

    Returns:
        DataFrame with interpolated values
    """
    if trajectories_df is None or trajectories_df.empty:
        return trajectories_df

    if method == "none" or method is None:
        return trajectories_df

    logger.info(f"Interpolating trajectories using {method} method (max_gap={max_gap})")

    result_df = trajectories_df.copy()

    # Process each trajectory separately
    for traj_id in result_df["TrajectoryID"].unique():
        mask = result_df["TrajectoryID"] == traj_id
        traj_data = result_df[mask].copy()

        # Sort by FrameID and remove duplicates (keep first occurrence)
        traj_data = traj_data.sort_values("FrameID").reset_index(drop=True)

        # Check for and remove duplicate FrameIDs
        if traj_data["FrameID"].duplicated().any():
            n_duplicates = traj_data["FrameID"].duplicated().sum()
            logger.warning(
                f"Trajectory {traj_id} has {n_duplicates} duplicate FrameID(s). "
                f"Keeping first occurrence of each frame."
            )
            traj_data = traj_data.drop_duplicates(subset="FrameID", keep="first")
            traj_data = traj_data.reset_index(drop=True)

        # Get frame range
        min_frame = traj_data["FrameID"].min()
        max_frame = traj_data["FrameID"].max()

        # Create complete frame range
        all_frames = np.arange(min_frame, max_frame + 1)

        # Reindex to include all frames
        traj_data = traj_data.set_index("FrameID").reindex(all_frames).reset_index()
        traj_data["TrajectoryID"] = traj_id

        # Fill State column for interpolated frames
        # Frames added by reindex will have NaN for State - set them to "occluded"
        if "State" in traj_data.columns:
            traj_data["State"] = traj_data["State"].fillna("occluded")
        else:
            traj_data["State"] = "occluded"

        # Ensure confidence columns exist (will be NaN for interpolated frames)
        confidence_cols = [
            col
            for col in result_df.columns
            if col.lower().endswith("confidence") or col == "PositionUncertainty"
        ]
        for col in confidence_cols:
            if col not in traj_data.columns:
                traj_data[col] = np.nan

        # Interpolate X and Y positions
        for col in ["X", "Y"]:
            traj_data[col] = _interpolate_column(
                traj_data["FrameID"].values,
                traj_data[col].values,
                method=method,
                max_gap=max_gap,
            )

        # Interpolate Theta using circular interpolation
        traj_data["Theta"] = _interpolate_angle(
            traj_data["FrameID"].values,
            traj_data["Theta"].values,
            method=method,
            max_gap=max_gap,
        )

        # Update the result dataframe - use proper indexing to avoid dtype warnings
        # Get the indices where this trajectory exists
        traj_indices = result_df[result_df["TrajectoryID"] == traj_id].index

        # Remove old rows for this trajectory
        result_df = result_df[result_df["TrajectoryID"] != traj_id]

        # Append the interpolated data
        result_df = pd.concat([result_df, traj_data], ignore_index=True)

    # Sort the final result by TrajectoryID and FrameID
    result_df = result_df.sort_values(["TrajectoryID", "FrameID"]).reset_index(
        drop=True
    )

    logger.info(f"Interpolation complete")
    return result_df


def _interpolate_column(frames, values, method="linear", max_gap=10):
    """Interpolate a single column with gap limit."""
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(values)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return values  # Need at least 2 points to interpolate

    valid_frames = frames[valid_indices]
    valid_values = values[valid_indices]

    # Create interpolation function
    try:
        if method == "linear":
            interp_func = interp1d(
                valid_frames,
                valid_values,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
        elif method == "cubic":
            if len(valid_indices) < 4:
                # Fall back to linear if not enough points
                interp_func = interp1d(
                    valid_frames,
                    valid_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
            else:
                interp_func = CubicSpline(
                    valid_frames, valid_values, bc_type="natural", extrapolate=False
                )
        elif method == "spline":
            if len(valid_indices) < 4:
                # Fall back to linear if not enough points
                interp_func = interp1d(
                    valid_frames,
                    valid_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value=np.nan,
                )
            else:
                # Smoothing spline with automatic smoothing factor
                interp_func = UnivariateSpline(valid_frames, valid_values, s=None, k=3)
        else:
            return values
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, keeping original values")
        return values

    # Interpolate all frames
    result = values.copy()

    # Only interpolate within gaps smaller than max_gap
    for i in range(len(valid_indices) - 1):
        start_idx = valid_indices[i]
        end_idx = valid_indices[i + 1]
        gap_size = end_idx - start_idx - 1

        if gap_size > 0 and gap_size <= max_gap:
            # Interpolate this gap
            gap_frames = frames[start_idx + 1 : end_idx]
            try:
                result[start_idx + 1 : end_idx] = interp_func(gap_frames)
            except:
                pass  # Keep NaN if interpolation fails

    return result


def _interpolate_angle(frames, angles, method="linear", max_gap=10):
    """
    Interpolate angles using circular interpolation to handle wraparound.
    """
    # Find valid (non-NaN) indices
    valid_mask = ~np.isnan(angles)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        return angles

    # Convert to Cartesian coordinates for interpolation
    sin_values = np.sin(angles)
    cos_values = np.cos(angles)

    # Interpolate sin and cos separately
    sin_interp = _interpolate_column(frames, sin_values, method=method, max_gap=max_gap)
    cos_interp = _interpolate_column(frames, cos_values, method=method, max_gap=max_gap)

    # Convert back to angles
    result = np.arctan2(sin_interp, cos_interp) % (2 * np.pi)

    # Preserve original NaN values where both sin and cos are NaN
    nan_mask = np.isnan(sin_interp) & np.isnan(cos_interp)
    result[nan_mask] = np.nan

    return result
