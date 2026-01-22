"""
Track assignment utilities for multi-object tracking.
Functionally identical to the original implementation's assignment logic.
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)
def _compute_cost_matrix_numba(
    N, M, meas_pos, meas_ori, pred_pos, pred_ori, 
    shapes_area, shapes_asp, prev_areas, prev_asps,
    covariances_arr, use_maha, Wp, Wo, Wa, Wasp, cull_threshold
):
    """Numba-compiled cost matrix computation for maximum speed."""
    cost = np.zeros((N, M), dtype=np.float32)
    
    for i in range(N):
        # Position cost
        if use_maha:
            Pcov = covariances_arr[i]
            try:
                invP = np.linalg.inv(Pcov)
                # Mahalanobis distance for all measurements
                for j in range(M):
                    diff = meas_pos[j] - pred_pos[i]
                    pos_cost_j = np.sqrt(diff @ invP @ diff)
                    
                    # Early exit for distant measurements
                    if pos_cost_j > cull_threshold:
                        cost[i, j] = Wp * pos_cost_j
                        continue
                    
                    # Orientation cost
                    odiff = abs(pred_ori[i] - meas_ori[j])
                    odiff = min(odiff, 2 * np.pi - odiff)
                    
                    # Shape costs
                    area_diff = abs(shapes_area[j] - prev_areas[i])
                    asp_diff = abs(shapes_asp[j] - prev_asps[i])
                    
                    cost[i, j] = Wp * pos_cost_j + Wo * odiff + Wa * area_diff + Wasp * asp_diff
            except:
                # Fallback to Euclidean if inversion fails
                for j in range(M):
                    diff = meas_pos[j] - pred_pos[i]
                    pos_cost_j = np.sqrt(diff[0]**2 + diff[1]**2)
                    
                    if pos_cost_j > cull_threshold:
                        cost[i, j] = Wp * pos_cost_j
                        continue
                    
                    odiff = abs(pred_ori[i] - meas_ori[j])
                    odiff = min(odiff, 2 * np.pi - odiff)
                    area_diff = abs(shapes_area[j] - prev_areas[i])
                    asp_diff = abs(shapes_asp[j] - prev_asps[i])
                    
                    cost[i, j] = Wp * pos_cost_j + Wo * odiff + Wa * area_diff + Wasp * asp_diff
        else:
            # Euclidean distance
            for j in range(M):
                diff = meas_pos[j] - pred_pos[i]
                pos_cost_j = np.sqrt(diff[0]**2 + diff[1]**2)
                
                if pos_cost_j > cull_threshold:
                    cost[i, j] = Wp * pos_cost_j
                    continue
                
                odiff = abs(pred_ori[i] - meas_ori[j])
                odiff = min(odiff, 2 * np.pi - odiff)
                area_diff = abs(shapes_area[j] - prev_areas[i])
                asp_diff = abs(shapes_asp[j] - prev_asps[i])
                
                cost[i, j] = Wp * pos_cost_j + Wo * odiff + Wa * area_diff + Wasp * asp_diff
    
    return cost


class TrackAssigner:
    """Handles assignment of detections to tracks."""

    def __init__(self, params):
        self.params = params

    def compute_cost_matrix(
        self, N, measurements, predictions, shapes, covariances, last_shape_info
    ):
        """Computes cost matrix for track-detection assignment.

        Args:
            covariances: List of (2, 2) covariance matrices, one per track
        """
        p = self.params
        M = len(measurements)
        if M == 0:
            return np.zeros((N, 0), np.float32)

        # Extract parameters
        Wp, Wo, Wa, Wasp = (
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
        )
        use_maha = p["USE_MAHALANOBIS"]
        MAX_DIST = p.get("MAX_DISTANCE_THRESHOLD", 1000.0)
        cull_threshold = MAX_DIST / Wp if Wp > 0 else float("inf")

        # Pre-extract arrays for Numba
        meas_pos = np.array([m[:2] for m in measurements], dtype=np.float32)  # (M, 2)
        meas_ori = np.array([m[2] for m in measurements], dtype=np.float32)  # (M,)
        pred_pos = np.array([p[:2] for p in predictions], dtype=np.float32)  # (N, 2)
        pred_ori = np.array([p[2] for p in predictions], dtype=np.float32)  # (N,)
        shapes_area = np.array([s[0] for s in shapes], dtype=np.float32)  # (M,)
        shapes_asp = np.array([s[1] for s in shapes], dtype=np.float32)  # (M,)
        
        # Extract previous shapes as arrays
        prev_areas = np.array([
            last_shape_info[i][0] if last_shape_info[i] is not None else shapes[0][0]
            for i in range(N)
        ], dtype=np.float32)
        prev_asps = np.array([
            last_shape_info[i][1] if last_shape_info[i] is not None else shapes[0][1]
            for i in range(N)
        ], dtype=np.float32)
        
        # Convert covariances list to 3D array for Numba
        covariances_arr = np.array(covariances, dtype=np.float32)  # (N, 2, 2)
        
        # Call Numba-compiled function
        if NUMBA_AVAILABLE:
            cost = _compute_cost_matrix_numba(
                N, M, meas_pos, meas_ori, pred_pos, pred_ori,
                shapes_area, shapes_asp, prev_areas, prev_asps,
                covariances_arr, use_maha, Wp, Wo, Wa, Wasp, cull_threshold
            )
        else:
            # Fallback to non-Numba version
            logger.warning("Numba not available, using slower Python implementation")
            cost = self._compute_cost_matrix_python(
                N, M, meas_pos, meas_ori, pred_pos, pred_ori,
                shapes_area, shapes_asp, prev_areas, prev_asps,
                covariances_arr, use_maha, Wp, Wo, Wa, Wasp, cull_threshold
            )

        return cost
    
    def _compute_cost_matrix_python(
        self, N, M, meas_pos, meas_ori, pred_pos, pred_ori,
        shapes_area, shapes_asp, prev_areas, prev_asps,
        covariances_arr, use_maha, Wp, Wo, Wa, Wasp, cull_threshold
    ):
        """Python fallback for cost matrix computation (same logic as Numba version)."""
        cost = np.zeros((N, M), np.float32)
        
        for i in range(N):
            if use_maha:
                Pcov = covariances_arr[i]
                diff = meas_pos - pred_pos[i]  # (M, 2)
                try:
                    invP = np.linalg.inv(Pcov)
                    pos_cost = np.sqrt(np.sum(diff @ invP * diff, axis=1))  # (M,)
                except:
                    pos_cost = np.linalg.norm(diff, axis=1)  # (M,)
            else:
                diff = meas_pos - pred_pos[i]  # (M, 2)
                pos_cost = np.linalg.norm(diff, axis=1)  # (M,)

            far_mask = pos_cost > cull_threshold
            odiff = np.abs(pred_ori[i] - meas_ori)
            odiff = np.minimum(odiff, 2 * np.pi - odiff)
            odiff[far_mask] = 0
            
            area_diff = np.abs(shapes_area - prev_areas[i])
            area_diff[far_mask] = 0
            asp_diff = np.abs(shapes_asp - prev_asps[i])
            asp_diff[far_mask] = 0

            cost[i, :] = Wp * pos_cost + Wo * odiff + Wa * area_diff + Wasp * asp_diff
        
        return cost

    def assign_tracks(
        self,
        cost,
        N,
        M,
        meas,
        track_states,
        tracking_continuity,
        kalman_filters,
        trajectory_ids,
        next_trajectory_id,
    ):
        """Assigns detections to tracks using the hybrid strategy from the original code."""
        p = self.params
        if M == 0:
            return [], [], [], next_trajectory_id

        CONTINUITY_THRESHOLD = p.get("CONTINUITY_THRESHOLD", 10)
        MAX_DIST = p["MAX_DISTANCE_THRESHOLD"]

        established = [
            i
            for i in range(N)
            if tracking_continuity[i] >= CONTINUITY_THRESHOLD
            and track_states[i] != "lost"
        ]
        unstable = [
            i
            for i in range(N)
            if tracking_continuity[i] < CONTINUITY_THRESHOLD
            and track_states[i] != "lost"
        ]
        lost = [i for i in range(N) if track_states[i] == "lost"]

        all_assignments = []
        assigned_dets = set()

        # Phase 1: Hungarian for established tracks
        if established and M > 0:
            est_cost = cost[established, :]
            rows_idx, cols_idx = linear_sum_assignment(est_cost)
            for r, c in zip(rows_idx, cols_idx):
                track_idx = established[r]
                if cost[track_idx, c] < MAX_DIST:
                    all_assignments.append((track_idx, c))
                    assigned_dets.add(c)

        # Phase 2: Priority for unstable tracks
        unstable_sorted = sorted(
            unstable, key=lambda i: tracking_continuity[i], reverse=True
        )
        for track_idx in unstable_sorted:
            avail_dets = [j for j in range(M) if j not in assigned_dets]
            if not avail_dets:
                break

            costs = cost[track_idx][avail_dets]
            best_local_idx = np.argmin(costs)
            if costs[best_local_idx] < MAX_DIST:
                det_idx = avail_dets[best_local_idx]
                all_assignments.append((track_idx, det_idx))
                assigned_dets.add(det_idx)

        # Phase 3: Respawn lost tracks
        unassigned_dets = [j for j in range(M) if j not in assigned_dets]
        MIN_RESPAWN_DIST = p.get("MIN_RESPAWN_DISTANCE", MAX_DIST * 0.8)

        for det_idx in unassigned_dets:
            if not lost:
                break

            min_dist_to_assigned = float("inf")
            if assigned_dets:
                for ad in assigned_dets:
                    dist = np.linalg.norm(meas[det_idx][:2] - meas[ad][:2])
                    min_dist_to_assigned = min(min_dist_to_assigned, dist)

            if min_dist_to_assigned >= MIN_RESPAWN_DIST:
                best_lost_track, best_cost = None, float("inf")
                for track_idx in lost:
                    if kalman_filters[track_idx].statePost is not None:
                        last_pos = kalman_filters[track_idx].statePost[:2].flatten()
                        respawn_dist = np.linalg.norm(meas[det_idx][:2] - last_pos)
                        if respawn_dist < best_cost:
                            best_cost, best_lost_track = respawn_dist, track_idx
                    else:  # No previous position, any is fine
                        best_lost_track = track_idx
                        break

                if best_lost_track is not None and (
                    kalman_filters[best_lost_track].statePost is None
                    or best_cost < MAX_DIST * 2.0
                ):
                    all_assignments.append((best_lost_track, det_idx))
                    assigned_dets.add(det_idx)
                    lost.remove(best_lost_track)
                    trajectory_ids[best_lost_track] = next_trajectory_id
                    next_trajectory_id += 1

        if not all_assignments:
            rows, cols = [], []
        else:
            rows, cols = zip(*all_assignments)
            rows, cols = list(rows), list(cols)

        valid_assignments, high_cost_tracks = [], []
        for r, c in zip(rows, cols):
            if cost[r, c] < MAX_DIST:
                valid_assignments.append((r, c))
            else:
                high_cost_tracks.append(r)

        if valid_assignments:
            final_rows, final_cols = zip(*valid_assignments)
        else:
            final_rows, final_cols = [], []

        all_dets_set = set(range(M))
        matched_dets_set = set(final_cols)
        free_dets = list(all_dets_set - matched_dets_set)

        return (
            list(final_rows),
            list(final_cols),
            free_dets,
            next_trajectory_id,
            high_cost_tracks,
        )
