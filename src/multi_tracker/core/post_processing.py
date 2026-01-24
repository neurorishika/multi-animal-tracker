# src/multi_tracker/core/post_processing.py
"""
Trajectory post-processing utilities for cleaning and refining tracking data.
"""
import numpy as np
import logging
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline

logger = logging.getLogger(__name__)


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
    max_dist_break = params.get("MAX_DISTANCE_BREAK", 300.0)

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        logger.info(
            f"Loaded {len(df)} rows from {csv_path} with columns: {list(df.columns)}"
        )
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        return None, {}

    # Check if confidence columns exist
    has_confidence = "DetectionConfidence" in df.columns

    stats = {
        "original_count": df["TrajectoryID"].nunique(),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_distance": 0,
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

        # Identify break points
        break_indices = traj_df[
            (traj_df["Velocity"] > max_vel_break)
            | (traj_df["DistDiff"] > max_dist_break)
        ].index.tolist()

        stats["broken_velocity"] += traj_df[traj_df["Velocity"] > max_vel_break].shape[
            0
        ]
        stats["broken_distance"] += traj_df[
            (traj_df["Velocity"] <= max_vel_break)
            & (traj_df["DistDiff"] > max_dist_break)
        ].shape[0]

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
                    columns=["FrameDiff", "DistDiff", "Velocity"], errors="ignore"
                )
                cleaned_segments.append(segment)
            segment_start_idx = break_idx

        # Add the last segment
        last_segment = traj_df.iloc[segment_start_idx:].copy()
        if len(last_segment) >= min_len:
            last_segment["TrajectoryID"] = new_traj_id
            new_traj_id += 1
            last_segment = last_segment.drop(
                columns=["FrameDiff", "DistDiff", "Velocity"], errors="ignore"
            )
            cleaned_segments.append(last_segment)

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
    max_dist_break = params.get("MAX_DISTANCE_BREAK", 300.0)

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
            "broken_distance": 0,
            "final_count": 0,
        }

    df = pd.DataFrame(all_data)

    stats = {
        "original_count": len([t for t in trajectories_full if t]),
        "removed_short": 0,
        "broken_velocity": 0,
        "broken_distance": 0,
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

        # Identify break points
        break_indices = traj_df[
            (traj_df["Velocity"] > max_vel_break)
            | (traj_df["DistDiff"] > max_dist_break)
        ].index.tolist()

        stats["broken_velocity"] += traj_df[traj_df["Velocity"] > max_vel_break].shape[
            0
        ]
        stats["broken_distance"] += traj_df[
            (traj_df["Velocity"] <= max_vel_break)
            & (traj_df["DistDiff"] > max_dist_break)
        ].shape[0]

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


def resolve_trajectories(forward_trajs, backward_trajs, video_length=None, params=None):
    """
    Merges forward and backward trajectories by creating a consensus set of trajectories.

    This function implements an iterative merging algorithm that:
    1. Adjusts backward trajectory frame numbers and orientations
    2. Calculates distance matrices between all trajectory pairs
    3. Uses Hungarian assignment to find optimal matches
    4. Merges trajectories based on conservative and liberal distance thresholds
    5. Handles identity swaps and conflicts during merging

    Args:
        forward_trajs (list): List of forward trajectory DataFrames or lists of tuples
        backward_trajs (list): List of backward trajectory DataFrames or lists of tuples
        video_length (int, optional): Total number of frames in video for frame adjustment
        params (dict, optional): Parameters for merging thresholds

    Returns:
        list: Final merged trajectories as list of DataFrames
    """

    if not forward_trajs and not backward_trajs:
        return []

    # Default parameters
    if params is None:
        params = {}

    TRUE_OVERLAP_THRESHOLD = params.get("TRUE_OVERLAP_THRESHOLD", 30.0)
    COMMONALITY_THRESHOLD = params.get("COMMONALITY_THRESHOLD", 10.0)
    MIN_LENGTH = params.get("MIN_TRAJECTORY_LENGTH", 5)
    MAX_ITERATIONS = params.get("MAX_MERGE_ITERATIONS", 10)

    logger.info(
        f"Starting trajectory resolution with {len(forward_trajs)} forward and {len(backward_trajs)} backward trajectories"
    )

    # Convert trajectory formats and prepare data
    all_trajs = []

    # Process forward trajectories
    for i, traj in enumerate(forward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"forward_{i}")
        if len(df) >= MIN_LENGTH:
            # Ensure theta is in [0, 2*pi]
            df["Theta"] = df["Theta"] % (2 * np.pi)
            all_trajs.append(df)

    # Process backward trajectories
    for i, traj in enumerate(backward_trajs):
        df = _convert_trajectory_to_dataframe(traj, f"backward_{i}")
        if len(df) >= MIN_LENGTH:
            # Adjust frame numbers if video_length is provided
            if video_length is not None:
                df["FrameID"] = video_length - df["FrameID"]
            # Rotate theta by 180 degrees for backward trajectories
            df["Theta"] = (df["Theta"] + np.pi) % (2 * np.pi)
            all_trajs.append(df)

    # Clean trajectories before merging (remove trailing lost/occluded states)
    all_trajs = _clean_trajectories(all_trajs, MIN_LENGTH)

    if not all_trajs:
        logger.warning("No valid trajectories found for merging")
        return []

    logger.info(f"Initial number of trajectories for merging: {len(all_trajs)}")

    # Iterative merging process
    iteration = 0
    while iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info(f"Merging iteration {iteration}")

        new_trajectories = []
        current_id = 0

        # Calculate distance matrices with different summary functions (NaN-aware)
        dist_matrix_conservative = _calculate_trajectory_distance_matrix(
            all_trajs, all_trajs, summary_func=lambda x: np.nanpercentile(x, 95)
        )
        dist_matrix_liberal = _calculate_trajectory_distance_matrix(
            all_trajs, all_trajs, summary_func=lambda x: np.nanpercentile(x, 5)
        )

        # Prepare assignment matrix
        dist_matrix_assignment = dist_matrix_conservative.copy()

        # Get max value, handling case where all values might be inf or NaN
        valid_values = dist_matrix_assignment[
            (dist_matrix_assignment != np.inf) & ~np.isnan(dist_matrix_assignment)
        ]
        if len(valid_values) > 0:
            max_value = np.max(valid_values) * 10
        else:
            max_value = 1e6  # Fallback if no valid values

        dist_matrix_assignment[dist_matrix_assignment == np.inf] = max_value
        dist_matrix_assignment[np.isnan(dist_matrix_assignment)] = (
            max_value  # Replace NaN with max_value
        )
        np.fill_diagonal(dist_matrix_assignment, max_value)  # Avoid self-matching

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(dist_matrix_assignment)

        # Track merges and used trajectories
        merges_made = 0
        used = set()

        # Process assignments and merge trajectories
        for i, j in zip(row_ind, col_ind):
            if i in used or j in used:
                continue

            # Skip if this is a self-pairing or invalid pairing
            if i == j:
                continue

            conservative_dist = dist_matrix_conservative[i, j]
            liberal_dist = dist_matrix_liberal[i, j]

            if conservative_dist < TRUE_OVERLAP_THRESHOLD:
                # High confidence merge - no distance threshold
                merged_trajs = _merge_trajectories(
                    all_trajs[i], all_trajs[j], distance_threshold=None
                )
                for new_traj in merged_trajs:
                    new_traj["TrajectoryID"] = current_id
                    current_id += 1
                    new_trajectories.append(new_traj)
                used.add(i)
                used.add(j)
                merges_made += 1

            elif liberal_dist < COMMONALITY_THRESHOLD:
                # Moderate confidence merge - with distance threshold
                merged_trajs = _merge_trajectories(
                    all_trajs[i],
                    all_trajs[j],
                    distance_threshold=TRUE_OVERLAP_THRESHOLD,
                )
                for new_traj in merged_trajs:
                    new_traj["TrajectoryID"] = current_id
                    current_id += 1
                    new_trajectories.append(new_traj)
                used.add(i)
                used.add(j)
                merges_made += 1

        # Add unused trajectories
        for i in range(len(all_trajs)):
            if i not in used:
                new_traj = all_trajs[i].copy()
                new_traj["TrajectoryID"] = current_id
                new_trajectories.append(new_traj)
                current_id += 1

        # Filter out trajectories that are too short
        all_trajs = [traj for traj in new_trajectories if len(traj) >= MIN_LENGTH]

        # Clean trajectories (remove trailing lost/occluded states)
        all_trajs = _clean_trajectories(all_trajs, MIN_LENGTH)

        logger.info(f"Merges made: {merges_made}, Total trajectories: {len(all_trajs)}")

        # Stop if no merges were made
        if merges_made == 0:
            logger.info("No more merges possible. Stopping iteration.")
            break

    # Final cleanup before returning
    all_trajs = _clean_trajectories(all_trajs, MIN_LENGTH)

    logger.info(
        f"Final result: {len(all_trajs)} trajectories after {iteration} iterations"
    )

    # Convert back to the expected format (list of tuples)
    final_trajectories = []
    for traj_df in all_trajs:
        traj_list = [
            (row["X"], row["Y"], row["Theta"], row["FrameID"])
            for _, row in traj_df.iterrows()
        ]
        final_trajectories.append(traj_list)

    return final_trajectories


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
    1. Removing trailing 'lost' and 'occluded' states
    2. Returning None if no valid detections remain

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

    # Find last valid detection
    last_valid_idx = traj_df[valid_states].index[-1]

    # Trim trajectory to last valid detection
    cleaned_df = traj_df.loc[:last_valid_idx].copy()

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
    """
    # Convert to dictionaries for O(1) lookup
    traj1_dict = dict(
        zip(
            traj1["FrameID"],
            zip(
                traj1["X"],
                traj1["Y"],
                traj1["Theta"],
                traj1.get("State", ["active"] * len(traj1)),
            ),
        )
    )
    traj2_dict = dict(
        zip(
            traj2["FrameID"],
            zip(
                traj2["X"],
                traj2["Y"],
                traj2["Theta"],
                traj2.get("State", ["active"] * len(traj2)),
            ),
        )
    )

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
            x1, y1, theta1, state1 = traj1_dict[frame]
            x2, y2, theta2, state2 = traj2_dict[frame]
            distance = np.linalg.norm([x1 - x2, y1 - y2])

            if distance <= distance_threshold:
                good_frames.add(frame)
            else:
                bad_frames.add(frame)
    else:
        good_frames = common_frames.copy()

    # Process mergeable frames (good frames + unique frames)
    mergeable_frames = good_frames.union(unique_1).union(unique_2)

    # Create merged trajectory from mergeable frames
    frame_list = []
    x_list = []
    y_list = []
    theta_list = []

    for frame in sorted(mergeable_frames):
        if frame in good_frames:
            # Both trajectories have this frame and distance is acceptable
            x1, y1, theta1, state1 = traj1_dict[frame]
            x2, y2, theta2, state2 = traj2_dict[frame]

            # Merge based on states
            if state1 == "active" and state2 == "active":
                x_val = (x1 + x2) / 2
                y_val = (y1 + y2) / 2
                theta_val = _circular_mean([theta1, theta2])
            elif state1 == "active":
                x_val, y_val, theta_val = x1, y1, theta1
            elif state2 == "active":
                x_val, y_val, theta_val = x2, y2, theta2
            else:
                x_val = (x1 + x2) / 2
                y_val = (y1 + y2) / 2
                theta_val = _circular_mean([theta1, theta2])

        elif frame in unique_1:
            x_val, y_val, theta_val, _ = traj1_dict[frame]
        else:  # frame in unique_2
            x_val, y_val, theta_val, _ = traj2_dict[frame]

        frame_list.append(frame)
        x_list.append(x_val)
        y_list.append(y_val)
        theta_list.append(theta_val)

    # Create main merged trajectory as DataFrame
    all_trajectories = []

    if frame_list:
        # Split merged frames into continuous segments
        segments = _split_into_continuous_segments(
            frame_list, x_list, y_list, theta_list
        )
        # Convert segments to DataFrames
        for segment in segments:
            df_segment = pd.DataFrame(segment)
            all_trajectories.append(df_segment)

    # Handle bad frames (create separate trajectories for each animal)
    if bad_frames:
        bad_segments_1 = _create_segments_from_frames(sorted(bad_frames), traj1_dict)
        bad_segments_2 = _create_segments_from_frames(sorted(bad_frames), traj2_dict)

        # Convert bad segments to DataFrames
        for segment in bad_segments_1:
            df_segment = pd.DataFrame(segment)
            all_trajectories.append(df_segment)
        for segment in bad_segments_2:
            df_segment = pd.DataFrame(segment)
            all_trajectories.append(df_segment)

    return all_trajectories


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

        # Sort by FrameID
        traj_data = traj_data.sort_values("FrameID").reset_index(drop=True)

        # Get frame range
        min_frame = traj_data["FrameID"].min()
        max_frame = traj_data["FrameID"].max()

        # Create complete frame range
        all_frames = np.arange(min_frame, max_frame + 1)

        # Reindex to include all frames
        traj_data = traj_data.set_index("FrameID").reindex(all_frames).reset_index()
        traj_data["TrajectoryID"] = traj_id

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

        # Update the result dataframe
        result_df.loc[mask] = traj_data.values

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
