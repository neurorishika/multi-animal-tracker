# src/multi_tracker/core/post_processing.py
"""
Trajectory post-processing utilities for cleaning and refining tracking data.
"""
import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(__name__)

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
            all_data.append({
                'TrajectoryID': track_id, 'X': int(x), 'Y': int(y), 
                'Theta': theta, 'FrameID': int(frame)
            })
    
    if not all_data:
        return [], {'original_count': 0, 'removed_short': 0, 'broken_velocity': 0, 'broken_distance': 0, 'final_count': 0}
        
    df = pd.DataFrame(all_data)
    
    stats = {
        'original_count': len([t for t in trajectories_full if t]),
        'removed_short': 0,
        'broken_velocity': 0,
        'broken_distance': 0,
        'final_count': 0
    }
    
    cleaned_segments = []
    for traj_id in df['TrajectoryID'].unique():
        traj_df = df[df['TrajectoryID'] == traj_id].sort_values('FrameID').reset_index(drop=True)
        if len(traj_df) < min_len:
            stats['removed_short'] += 1
            continue
        
        # Calculate frame difference, distance, and velocity between consecutive points
        traj_df['FrameDiff'] = traj_df['FrameID'].diff()
        traj_df['DistDiff'] = np.sqrt(traj_df['X'].diff()**2 + traj_df['Y'].diff()**2)
        traj_df['Velocity'] = traj_df['DistDiff'] / traj_df['FrameDiff']
        
        # Identify break points
        break_indices = traj_df[
            (traj_df['Velocity'] > max_vel_break) | 
            (traj_df['DistDiff'] > max_dist_break)
        ].index.tolist()
        
        stats['broken_velocity'] += traj_df[traj_df['Velocity'] > max_vel_break].shape[0]
        stats['broken_distance'] += traj_df[(traj_df['Velocity'] <= max_vel_break) & (traj_df['DistDiff'] > max_dist_break)].shape[0]

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

    stats['final_count'] = len(cleaned_segments)
    
    # Convert dataframe segments back to list of tuples format
    final_trajectories = [
        [tuple(row) for row in seg_df[['X', 'Y', 'Theta', 'FrameID']].to_numpy()] 
        for seg_df in cleaned_segments
    ]
    
    logger.info(f"Post-processing stats: {stats}")
    return final_trajectories, stats