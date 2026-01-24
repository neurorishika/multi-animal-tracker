"""
SOTA Optimized Track Assigner.
Compatible with Vectorized Kalman Filter.
Uses batch Mahalanobis distance and Numba-accelerated spatial assignment.
"""

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)
def _compute_cost_matrix_sota(
    N,
    M,
    meas_pos,
    meas_ori,
    pred_pos,
    pred_ori,
    shapes_area,
    shapes_asp,
    prev_areas,
    prev_asps,
    S_inv_batch,
    use_maha,
    Wp,
    Wo,
    Wa,
    Wasp,
    cull_threshold,
):
    """SOTA Numba kernel using pre-calculated batch Inverse Covariances."""
    cost = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        # Extract pre-calculated 2x2 inverse position covariance from the 3x3 S_inv
        # (This avoids N*M matrix inversions inside the loop)
        inv_S_pos = S_inv_batch[i, :2, :2]

        for j in range(M):
            diff = meas_pos[j] - pred_pos[i]

            # 1. Position Cost
            if use_maha:
                # Mahalanobis: sqrt(d^T * S_inv * d)
                pos_dist = np.sqrt(
                    diff[0] * (diff[0] * inv_S_pos[0, 0] + diff[1] * inv_S_pos[1, 0])
                    + diff[1] * (diff[0] * inv_S_pos[0, 1] + diff[1] * inv_S_pos[1, 1])
                )
            else:
                pos_dist = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

            # Spatial Culling
            if pos_dist > cull_threshold:
                cost[i, j] = 1e6  # Large penalty
                continue

            # 2. Orientation Cost (Circular wrap)
            odiff = abs(pred_ori[i] - meas_ori[j])
            if odiff > np.pi:
                odiff = 2 * np.pi - odiff

            # 3. Shape Costs
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])

            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff

    return cost


class TrackAssigner:
    """Handles assignment of detections to tracks with SOTA optimizations."""

    def __init__(self, params):
        self.params = params

    def _get_spatial_candidates(self, N, M, pred_pos, meas_pos, max_dist):
        """Use KD-tree to find candidate matches within max_dist for large N."""
        if M == 0 or N == 0:
            return {}
        tree = cKDTree(meas_pos)
        candidates = {}
        for i in range(N):
            indices = tree.query_ball_point(pred_pos[i], max_dist)
            if indices:
                candidates[i] = indices
        return candidates

    def compute_cost_matrix(
        self, N, measurements, predictions, shapes, kf_manager, last_shape_info
    ):
        """
        Computes cost matrix. Compatible with Vectorized Kalman Filter.
        """
        p = self.params
        M = len(measurements)
        if M == 0:
            return np.zeros((N, 0), np.float32), {}

        # SOTA: Get pre-calculated Inverse Innovation Covariances from Manager
        S_inv_batch = kf_manager.get_mahalanobis_matrices()

        # Pre-extract arrays for Numba (Avoids attribute access in loop)
        meas_pos = np.array([m[:2] for m in measurements], dtype=np.float32)
        meas_ori = np.array([m[2] for m in measurements], dtype=np.float32)
        pred_pos = predictions[:, :2]  # Predictions are already (N, 3)
        pred_ori = predictions[:, 2]

        shapes_area = np.array([s[0] for s in shapes], dtype=np.float32)
        shapes_asp = np.array([s[1] for s in shapes], dtype=np.float32)

        # Optimized fill for previous shape info
        prev_areas = np.zeros(N, dtype=np.float32)
        prev_asps = np.zeros(N, dtype=np.float32)
        for i in range(N):
            if last_shape_info[i] is not None:
                prev_areas[i], prev_asps[i] = last_shape_info[i]
            else:
                prev_areas[i], prev_asps[i] = shapes_area[0], shapes_asp[0]

        MAX_DIST = p.get("MAX_DISTANCE_THRESHOLD", 1000.0)
        cull_threshold = (
            max(MAX_DIST / p["W_POSITION"], 50.0) if p["W_POSITION"] > 0 else 1e6
        )

        if p.get("ENABLE_SPATIAL_OPTIMIZATION", False) and N > 50:
            spatial_candidates = self._get_spatial_candidates(
                N, M, pred_pos, meas_pos, cull_threshold
            )
            # KD-Tree mode uses a hybrid approach
            cost = self._compute_cost_python_fallback(
                N,
                M,
                meas_pos,
                meas_ori,
                pred_pos,
                pred_ori,
                shapes_area,
                shapes_asp,
                prev_areas,
                prev_asps,
                S_inv_batch,
                p,
                spatial_candidates,
            )
            return cost, spatial_candidates

        # Default: Full SOTA Numba Matrix
        cost = _compute_cost_matrix_sota(
            N,
            M,
            meas_pos,
            meas_ori,
            pred_pos,
            pred_ori,
            shapes_area,
            shapes_asp,
            prev_areas,
            prev_asps,
            S_inv_batch,
            p["USE_MAHALANOBIS"],
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
            cull_threshold,
        )

        return cost, {}

    def compute_assignment_confidence(self, cost, matched_pairs):
        """Compute confidence scores for assignments."""
        if not matched_pairs:
            return {}
        scale = self.params.get("MAX_DISTANCE_THRESHOLD", 100.0) * 0.5
        return {r: 1.0 / (1.0 + cost[r, c] / scale) for r, c in matched_pairs}

    def assign_tracks(
        self,
        cost,
        N,
        M,
        meas,
        track_states,
        tracking_continuity,
        kf_manager,
        trajectory_ids,
        next_trajectory_id,
        spatial_candidates=None,
    ):
        """
        Drop-in replacement for track assignment logic.
        Compatible with kf_manager.X state access.
        """
        p = self.params
        if M == 0:
            return [], [], [], next_trajectory_id

        THRESH = p.get("CONTINUITY_THRESHOLD", 10)
        MAX_DIST = p["MAX_DISTANCE_THRESHOLD"]
        USE_GREEDY = p.get("ENABLE_GREEDY_ASSIGNMENT", False)

        # 1. Split tracks by state
        est = [
            i
            for i in range(N)
            if tracking_continuity[i] >= THRESH and track_states[i] != "lost"
        ]
        unst = [
            i
            for i in range(N)
            if tracking_continuity[i] < THRESH and track_states[i] != "lost"
        ]
        lost = [i for i in range(N) if track_states[i] == "lost"]

        all_assignments = []
        assigned_dets = set()

        # Phase 1: Established Tracks
        if est:
            if USE_GREEDY:
                # Optimized Greedy
                track_det_costs = []
                for r in est:
                    for c in range(M):
                        if cost[r, c] < MAX_DIST:
                            track_det_costs.append((cost[r, c], r, c))
                track_det_costs.sort()
                assigned_r = set()
                for _, r, c in track_det_costs:
                    if r not in assigned_r and c not in assigned_dets:
                        all_assignments.append((r, c))
                        assigned_dets.add(c)
                        assigned_r.add(r)
            else:
                # Hungarian
                rows, cols = linear_sum_assignment(cost[est, :])
                for r_idx, c in zip(rows, cols):
                    r = est[r_idx]
                    if cost[r, c] < MAX_DIST:
                        all_assignments.append((r, c))
                        assigned_dets.add(c)

        # Phase 2: Unstable Tracks
        for r in sorted(unst, key=lambda i: tracking_continuity[i], reverse=True):
            avail = [j for j in range(M) if j not in assigned_dets]
            if not avail:
                break
            best_c = avail[np.argmin(cost[r, avail])]
            if cost[r, best_c] < MAX_DIST:
                all_assignments.append((r, best_c))
                assigned_dets.add(best_c)

        # Phase 3: Respawn Lost Tracks
        unassigned = [j for j in range(M) if j not in assigned_dets]
        respawn_dist_limit = p.get("MIN_RESPAWN_DISTANCE", MAX_DIST * 0.8)

        for c in unassigned:
            if not lost:
                break
            # Check proximity to existing active tracks to avoid shadow tracking
            min_dist_active = (
                min(
                    [np.linalg.norm(meas[c][:2] - meas[ad][:2]) for ad in assigned_dets]
                )
                if assigned_dets
                else 1e6
            )

            if min_dist_active >= respawn_dist_limit:
                best_r, best_c_val = None, 1e6
                for r in lost:
                    # Accessing vectorized manager state X: [x, y, theta, vx, vy]
                    last_pos = kf_manager.X[r, :2]
                    dist = np.linalg.norm(meas[c][:2] - last_pos)
                    if dist < best_c_val:
                        best_c_val, best_r = dist, r

                if best_r is not None and best_c_val < MAX_DIST * 2.0:
                    all_assignments.append((best_r, c))
                    assigned_dets.add(c)
                    lost.remove(best_r)
                    trajectory_ids[best_r] = next_trajectory_id
                    next_trajectory_id += 1

        if not all_assignments:
            return [], [], [], next_trajectory_id, []

        final_r, final_c = zip(*all_assignments)
        free_dets = list(set(range(M)) - set(final_c))
        return list(final_r), list(final_c), free_dets, next_trajectory_id, []

    def _compute_cost_python_fallback(
        self,
        N,
        M,
        meas_pos,
        meas_ori,
        pred_pos,
        pred_ori,
        sh_area,
        sh_asp,
        pr_area,
        pr_asp,
        S_inv,
        p,
        candidates,
    ):
        """Python fallback for spatial optimization."""
        cost = np.full((N, M), 1e6, dtype=np.float32)
        Wp, Wo, Wa, Wasp = (
            p["W_POSITION"],
            p["W_ORIENTATION"],
            p["W_AREA"],
            p["W_ASPECT"],
        )

        for r, det_indices in candidates.items():
            inv_S = S_inv[r, :2, :2]
            for c in det_indices:
                diff = meas_pos[c] - pred_pos[r]
                pos_c = (
                    np.sqrt(diff @ inv_S @ diff)
                    if p["USE_MAHALANOBIS"]
                    else np.linalg.norm(diff)
                )

                odiff = abs(pred_ori[r] - meas_ori[c])
                if odiff > np.pi:
                    odiff = 2 * np.pi - odiff

                cost[r, c] = (
                    Wp * pos_c
                    + Wo * odiff
                    + Wa * abs(sh_area[c] - pr_area[r])
                    + Wasp * abs(sh_asp[c] - pr_asp[r])
                )
        return cost
