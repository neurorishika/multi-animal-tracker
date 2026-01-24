"""
Track assignment utilities for multi-object tracking.
Functionally identical to the original implementation's assignment logic.
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class TrackAssigner:
    """Handles assignment of detections to tracks."""

    def __init__(self, params):
        self.params = params

    def compute_cost_matrix(
        self, N, measurements, predictions, shapes, kalman_filters, last_shape_info
    ):
        """Computes cost matrix for track-detection assignment using vectorized operations."""
        p = self.params
        M = len(measurements)
        if M == 0:
            return np.zeros((N, 0), np.float32)

        # Convert to arrays for vectorization
        meas_arr = np.array(measurements, dtype=np.float32)  # (M, 3)
        pred_arr = np.array(predictions, dtype=np.float32)  # (N, 3)
        shapes_arr = np.array(shapes, dtype=np.float32)  # (M, 2)

        # Weights
        Wp, Wo, Wa, Wasp = (
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
        )
        use_maha = p["USE_MAHALANOBIS"]

        # 1. Position cost - vectorized
        if use_maha:
            # Mahalanobis distance - compute for each track
            pos_cost = np.zeros((N, M), dtype=np.float32)
            for i in range(N):
                Pcov = kalman_filters[i].errorCovPre[:2, :2]
                diff = meas_arr[:, :2] - pred_arr[i, :2]  # (M, 2)
                try:
                    invP = np.linalg.inv(Pcov)
                    # Vectorized Mahalanobis: sqrt(diff @ inv @ diff.T) for each measurement
                    pos_cost[i, :] = np.sqrt(np.sum(diff @ invP * diff, axis=1))
                except:
                    # Fallback to Euclidean
                    pos_cost[i, :] = np.linalg.norm(diff, axis=1)
        else:
            # Euclidean distance - fully vectorized
            # diff shape: (N, M, 2) via broadcasting
            diff = meas_arr[np.newaxis, :, :2] - pred_arr[:, np.newaxis, :2]
            pos_cost = np.linalg.norm(diff, axis=2)  # (N, M)

        # 2. Orientation cost - vectorized with circular distance
        # diff shape: (N, M)
        odiff = np.abs(pred_arr[:, np.newaxis, 2] - meas_arr[np.newaxis, :, 2])
        odiff = np.minimum(odiff, 2 * np.pi - odiff)

        # 3. Shape costs - vectorized
        # Get previous shapes, using current measurement shapes as fallback
        prev_shapes = np.array(
            [
                last_shape_info[i] if last_shape_info[i] is not None else shapes[0]
                for i in range(N)
            ],
            dtype=np.float32,
        )  # (N, 2)

        # Area difference: (N, M)
        area_diff = np.abs(prev_shapes[:, np.newaxis, 0] - shapes_arr[np.newaxis, :, 0])

        # Aspect ratio difference: (N, M)
        asp_diff = np.abs(prev_shapes[:, np.newaxis, 1] - shapes_arr[np.newaxis, :, 1])

        # 4. Combine all costs
        cost = Wp * pos_cost + Wo * odiff + Wa * area_diff + Wasp * asp_diff

        return cost.astype(np.float32)

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
