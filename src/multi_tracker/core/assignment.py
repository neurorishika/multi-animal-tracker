"""
Track assignment utilities for multi-object tracking.

This module handles the data association between detections and tracks using
cost matrices, Hungarian algorithm, and hybrid assignment strategies.
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackAssigner:
    """
    Handles assignment of detections to tracks using various strategies.
    
    Implements hybrid assignment system with Hungarian algorithm for established
    tracks and priority-based assignment for unstable tracks.
    """
    
    def __init__(self, params):
        """
        Initialize track assigner.
        
        Args:
            params (dict): Assignment parameters
        """
        self.params = params
        
    def compute_cost_matrix(self, N, measurements, predictions, shapes, 
                           kalman_filters, last_shape_info, use_mahalanobis, 
                           w_position, w_orientation, w_area, w_aspect):
        """
        Compute cost matrix for track-detection assignment.
        
        Args:
            N (int): Number of tracks
            measurements (list): Current detections [x, y, theta]
            predictions (np.ndarray): Predicted states from Kalman filters
            shapes (list): Shape information for detections
            kalman_filters (list): Kalman filter objects
            last_shape_info (list): Previous shape information for tracks
            use_mahalanobis (bool): Whether to use Mahalanobis distance
            w_position (float): Weight for position cost
            w_orientation (float): Weight for orientation cost
            w_area (float): Weight for area cost
            w_aspect (float): Weight for aspect ratio cost
            
        Returns:
            np.ndarray: Cost matrix (N_tracks x N_detections)
        """
        M = len(measurements)  # Number of current detections
        
        if M == 0:
            return np.zeros((N, 0), np.float32)
        
        cost = np.zeros((N, M), np.float32)
        
        for i in range(N):  # For each track
            for j in range(M):  # For each detection
                # === POSITION COST ===
                if use_mahalanobis:
                    # Use Mahalanobis distance considering prediction uncertainty
                    Pcov = kalman_filters[i].errorCovPre[:2, :2]  # Position covariance
                    diff = measurements[j][:2] - predictions[i][:2]
                    try: 
                        invP = np.linalg.inv(Pcov)
                        posc = np.sqrt(diff.T @ invP @ diff)
                    except: 
                        posc = np.linalg.norm(diff)  # Fallback to Euclidean
                else:
                    # Simple Euclidean distance
                    posc = np.linalg.norm(measurements[j][:2] - predictions[i][:2])
                
                # === ORIENTATION COST ===
                # Handle angular wraparound for orientation difference
                odiff = abs(predictions[i][2] - measurements[j][2])
                odiff = min(odiff, 2*np.pi - odiff)  # Minimum angular distance
                
                # === SHAPE CONSISTENCY COST ===
                # Use previous shape info or current detection if no history
                prev_a, prev_as = last_shape_info[i] if last_shape_info[i] else shapes[j]
                area_diff = abs(shapes[j][0] - prev_a)      # Area difference
                asp_diff = abs(shapes[j][1] - prev_as)      # Aspect ratio difference
            
                # === COMBINED COST ===
                # Weighted combination of all cost components
                cost[i, j] = w_position * posc + w_orientation * odiff + w_area * area_diff + w_aspect * asp_diff
        
        return cost
    
    def _compute_mahalanobis_distance(self, pred_pos, meas_pos, kalman_filter):
        """
        Compute Mahalanobis distance considering prediction uncertainty.
        
        Args:
            pred_pos (np.ndarray): Predicted position [x, y]
            meas_pos (np.ndarray): Measured position [x, y]
            kalman_filter: Kalman filter object (can be None)
            
        Returns:
            float: Mahalanobis distance or Euclidean fallback
        """
        if kalman_filter is None:
            return np.linalg.norm(meas_pos - pred_pos)
            
        try:
            pos_cov = kalman_filter.errorCovPre[:2, :2]
            diff = meas_pos - pred_pos
            inv_cov = np.linalg.inv(pos_cov)
            return np.sqrt(diff.T @ inv_cov @ diff)
        except (np.linalg.LinAlgError, AttributeError):
            return np.linalg.norm(meas_pos - pred_pos)
    
    def assign_tracks(self, N, M, cost, meas, tracking_continuity, track_states,
                     kalman_filters, trajectory_ids, trajectory_info, params):
        """
        Assign detections to tracks using hybrid strategy.
        
        Args:
            N (int): Number of tracks
            M (int): Number of detections
            cost (np.ndarray): Cost matrix (N_tracks x N_detections)
            meas (list): Current measurements
            tracking_continuity (list): Continuity counters for each track
            track_states (list): Current states of tracks
            kalman_filters (list): Kalman filter objects  
            trajectory_ids (list): Current trajectory IDs for tracks
            trajectory_info (dict): Dictionary containing 'next_id' key for next trajectory ID
            params (dict): Tracking parameters
            
        Returns:
            tuple: (track_indices, detection_indices) of valid assignments
        """
        if M == 0:
            return [], []
        
        next_trajectory_id = trajectory_info['next_id']
        
        # Define continuity threshold for established vs new tracks
        CONTINUITY_THRESHOLD = params.get("CONTINUITY_THRESHOLD", 10)  # Frames of continuous tracking
        
        # Separate tracks into established (high continuity) and new/unstable (low continuity)
        established_tracks = [i for i in range(N) if tracking_continuity[i] >= CONTINUITY_THRESHOLD and track_states[i] != 'lost']
        unstable_tracks = [i for i in range(N) if tracking_continuity[i] < CONTINUITY_THRESHOLD and track_states[i] != 'lost']
        lost_tracks = [i for i in range(N) if track_states[i] == 'lost']
        
        logger.debug(f"Established={len(established_tracks)}, Unstable={len(unstable_tracks)}, Lost={len(lost_tracks)}")
        
        assigned_detections = set()
        assigned_tracks = set()
        all_assignments = []
        
        # === PHASE 1: HUNGARIAN FOR ESTABLISHED TRACKS ===
        if established_tracks and M > 0:
            # Create sub-cost matrix for established tracks only
            est_cost = np.zeros((len(established_tracks), M), np.float32)
            for i, track_idx in enumerate(established_tracks):
                for j in range(M):
                    est_cost[i, j] = cost[track_idx, j]
            
            # Apply Hungarian algorithm to established tracks
            est_rows, est_cols = linear_sum_assignment(est_cost)
            
            # Process assignments for established tracks
            for i, j in zip(est_rows, est_cols):
                track_idx = established_tracks[i]
                det_idx = j  # j is already the detection index from est_cols
                
                # Only accept assignment if cost is reasonable
                if cost[track_idx, det_idx] < params["MAX_DISTANCE_THRESHOLD"]:
                    all_assignments.append((track_idx, det_idx))
                    assigned_detections.add(det_idx)
                    assigned_tracks.add(track_idx)
                    logger.debug(f"Hungarian assignment: Track {track_idx} (continuity={tracking_continuity[track_idx]}) -> Detection {det_idx} (cost={cost[track_idx, det_idx]:.2f})")
                else:
                    logger.debug(f"Rejected Hungarian assignment: Track {track_idx} -> Detection {det_idx} (cost too high: {cost[track_idx, det_idx]:.2f})")
        
        # === PHASE 2: PRIORITY-BASED FOR UNSTABLE TRACKS ===
        # Sort unstable tracks by continuity (longest first, even if below threshold)
        unstable_tracks_sorted = sorted(unstable_tracks, key=lambda i: tracking_continuity[i], reverse=True)
        
        for track_idx in unstable_tracks_sorted:
            best_detection = None
            best_cost = float('inf')
            
            # Find best available detection for this track
            for det_idx in range(M):
                if det_idx in assigned_detections:
                    continue  # Detection already taken by established track
                
                track_cost = cost[track_idx, det_idx]
                if track_cost < params["MAX_DISTANCE_THRESHOLD"] and track_cost < best_cost:
                    best_cost = track_cost
                    best_detection = det_idx
            
            # Assign best detection to this track if found
            if best_detection is not None:
                all_assignments.append((track_idx, best_detection))
                assigned_detections.add(best_detection)
                assigned_tracks.add(track_idx)
                logger.debug(f"Priority assignment: Track {track_idx} (continuity={tracking_continuity[track_idx]}) -> Detection {best_detection} (cost={best_cost:.2f})")
        
        # === PHASE 3: RESPAWN LOST TRACKS WITH REMAINING DETECTIONS ===
        unassigned_detections = [i for i in range(M) if i not in assigned_detections]
        MIN_RESPAWN_DISTANCE = params.get("MIN_RESPAWN_DISTANCE", params["MAX_DISTANCE_THRESHOLD"] * 0.8)
        
        for det_idx in unassigned_detections:
            if not lost_tracks:
                break  # No more lost tracks available
            
            # Check if this detection is far enough from all assigned detections
            detection_pos = meas[det_idx][:2]  # [x, y] position
            is_good_detection = True
            
            # Calculate minimum distance to any assigned detection
            min_distance_to_assigned = float('inf')
            for assigned_det_idx in assigned_detections:
                assigned_pos = meas[assigned_det_idx][:2]
                distance = np.linalg.norm(detection_pos - assigned_pos)
                min_distance_to_assigned = min(min_distance_to_assigned, distance)
            
            # Only respawn if detection is sufficiently far from assigned detections
            if min_distance_to_assigned < MIN_RESPAWN_DISTANCE:
                is_good_detection = False
                logger.debug(f"Rejected respawn: Detection {det_idx} too close to assigned detection (distance={min_distance_to_assigned:.2f} < {MIN_RESPAWN_DISTANCE:.2f})")
            
            # Also check if detection is within reasonable bounds for any lost track
            if is_good_detection:
                best_lost_track = None
                best_respawn_cost = float('inf')
                
                # Find the best lost track for this detection (if any)
                for track_idx in lost_tracks:
                    # Use last known position if available, otherwise skip cost check
                    if kalman_filters[track_idx].statePost is not None:
                        last_known_pos = kalman_filters[track_idx].statePost[:2].flatten()
                        respawn_distance = np.linalg.norm(detection_pos - last_known_pos)
                        
                        # Only consider if within reasonable respawn distance
                        if respawn_distance < params["MAX_DISTANCE_THRESHOLD"] * 2.0:  # More lenient for respawning
                            if respawn_distance < best_respawn_cost:
                                best_respawn_cost = respawn_distance
                                best_lost_track = track_idx
                    else:
                        # If no last known position, any lost track is eligible
                        best_lost_track = track_idx
                        break
                
                # Respawn the best lost track if found
                if best_lost_track is not None:
                    all_assignments.append((best_lost_track, det_idx))
                    assigned_tracks.add(best_lost_track)
                    lost_tracks.remove(best_lost_track)
                    # Assign new trajectory ID for respawned track
                    trajectory_ids[best_lost_track] = next_trajectory_id
                    next_trajectory_id += 1
                    logger.debug(f"Respawn assignment: Track {best_lost_track} -> Detection {det_idx} (distance from assigned={min_distance_to_assigned:.2f}, respawn_cost={best_respawn_cost:.2f}) - New trajectory ID: {trajectory_ids[best_lost_track]}")
                else:
                    logger.debug(f"No suitable lost track for detection {det_idx}")
            
            # If detection wasn't good enough for respawning, it remains unassigned
            if not is_good_detection or best_lost_track is None:
                logger.debug(f"Detection {det_idx} left unassigned")
        
        # Convert assignments back to the format expected by rest of code
        if all_assignments:
            rows, cols = zip(*all_assignments)
            rows, cols = list(rows), list(cols)
        else:
            rows, cols = [], []
        
        # Update the trajectory info with the new next_trajectory_id
        trajectory_info['next_id'] = next_trajectory_id
        
        return rows, cols
    
    def _assign_established_tracks(self, cost_matrix, established_tracks, max_distance):
        """
        Assign established tracks using Hungarian algorithm.
        
        Args:
            cost_matrix (np.ndarray): Full cost matrix
            established_tracks (list): Indices of established tracks
            max_distance (float): Maximum assignment distance
            
        Returns:
            list: List of (track_idx, detection_idx) assignments
        """
        if not established_tracks:
            return []
            
        num_detections = cost_matrix.shape[1]
        
        # Create sub-cost matrix for established tracks only
        sub_cost = np.zeros((len(established_tracks), num_detections), np.float32)
        for i, track_idx in enumerate(established_tracks):
            sub_cost[i, :] = cost_matrix[track_idx, :]
        
        # Apply Hungarian algorithm
        try:
            row_indices, col_indices = linear_sum_assignment(sub_cost)
        except ValueError:
            return []
        
        assignments = []
        for i, j in zip(row_indices, col_indices):
            track_idx = established_tracks[i]
            det_idx = j
            
            if cost_matrix[track_idx, det_idx] < max_distance:
                assignments.append((track_idx, det_idx))
                logger.debug(f"Hungarian assignment: Track {track_idx} -> Detection {det_idx} "
                           f"(cost={cost_matrix[track_idx, det_idx]:.2f})")
            else:
                logger.debug(f"Rejected Hungarian assignment: Track {track_idx} -> Detection {det_idx} "
                           f"(cost too high: {cost_matrix[track_idx, det_idx]:.2f})")
        
        return assignments
    
    def _assign_unstable_tracks(self, cost_matrix, unstable_tracks, tracking_continuity,
                               assigned_detections, max_distance):
        """
        Assign unstable tracks using priority-based strategy.
        
        Args:
            cost_matrix (np.ndarray): Full cost matrix
            unstable_tracks (list): Indices of unstable tracks
            tracking_continuity (list): Continuity counters
            assigned_detections (set): Already assigned detection indices
            max_distance (float): Maximum assignment distance
            
        Returns:
            list: List of (track_idx, detection_idx) assignments
        """
        # Sort by continuity (longest first)
        unstable_sorted = sorted(unstable_tracks, 
                               key=lambda i: tracking_continuity[i], reverse=True)
        
        assignments = []
        num_detections = cost_matrix.shape[1]
        
        for track_idx in unstable_sorted:
            best_detection = None
            best_cost = float('inf')
            
            # Find best available detection
            for det_idx in range(num_detections):
                if det_idx in assigned_detections:
                    continue
                    
                cost = cost_matrix[track_idx, det_idx]
                if cost < max_distance and cost < best_cost:
                    best_cost = cost
                    best_detection = det_idx
            
            # Assign if suitable detection found
            if best_detection is not None:
                assignments.append((track_idx, best_detection))
                assigned_detections.add(best_detection)
                logger.debug(f"Priority assignment: Track {track_idx} "
                           f"(continuity={tracking_continuity[track_idx]}) -> "
                           f"Detection {best_detection} (cost={best_cost:.2f})")
        
        return assignments
    
    def _respawn_lost_tracks(self, cost_matrix, lost_tracks, assigned_detections, max_distance):
        """
        Respawn lost tracks with remaining detections.
        
        Args:
            cost_matrix (np.ndarray): Full cost matrix
            lost_tracks (list): Indices of lost tracks
            assigned_detections (set): Already assigned detection indices
            max_distance (float): Maximum assignment distance
            
        Returns:
            list: List of (track_idx, detection_idx) assignments
        """
        num_detections = cost_matrix.shape[1]
        unassigned_detections = [i for i in range(num_detections) 
                               if i not in assigned_detections]
        
        min_respawn_distance = self.params.get("MIN_RESPAWN_DISTANCE", max_distance * 0.8)
        assignments = []
        available_lost_tracks = lost_tracks.copy()
        
        for det_idx in unassigned_detections:
            if not available_lost_tracks:
                break
                
            # Find best lost track for this detection
            best_track = None
            best_cost = float('inf')
            
            for track_idx in available_lost_tracks:
                cost = cost_matrix[track_idx, det_idx]
                # More lenient distance threshold for respawning
                if cost < max_distance * 2.0 and cost < best_cost:
                    best_cost = cost
                    best_track = track_idx
            
            if best_track is not None:
                assignments.append((best_track, det_idx))
                available_lost_tracks.remove(best_track)
                logger.debug(f"Respawn assignment: Track {best_track} -> Detection {det_idx} "
                           f"(cost={best_cost:.2f})")
        
        return assignments
