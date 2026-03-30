"""
Optimized Track Assigner.
Compatible with Vectorized Kalman Filter.
Uses batch Mahalanobis distance and Numba-accelerated spatial assignment.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from multi_tracker.core.identity.classification.cnn import apply_cnn_identity_cost

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
def _compute_cost_matrix_numba(
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
    meas_ori_directed,
):
    """Numba kernel using pre-calculated batch Inverse Covariances."""
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
            # OBB theta is an axis (0/180 equivalent) unless pose provides directed heading.
            if meas_ori_directed[j] == 0:
                alt = np.pi - odiff
                if alt < odiff:
                    odiff = alt

            # 3. Shape Costs
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])

            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff

    return cost


class TrackAssigner:
    """Handles assignment of detections to tracks with optimizations."""

    def __init__(self, params, worker=None):
        self.params = params
        self.worker = worker
        self._large_n_warning_shown = False  # Track if we've shown the warning

    def _spatial_optimization_enabled(self) -> bool:
        """Support both the current flag and the legacy alias."""
        return bool(
            self.params.get(
                "ENABLE_SPATIAL_OPTIMIZATION",
                self.params.get("USE_SPATIAL_PRUNING", False),
            )
        )

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

    @staticmethod
    def _orientation_diff(pred_theta, meas_theta, directed: bool) -> float:
        odiff = abs(float(pred_theta) - float(meas_theta))
        if odiff > np.pi:
            odiff = 2 * np.pi - odiff
        if not directed:
            alt = np.pi - odiff
            if alt < odiff:
                odiff = alt
        return float(max(0.0, odiff))

    @staticmethod
    def _pose_paired_stats(
        det_pose, track_pose, min_shared: int = 3
    ) -> tuple[float | None, int]:
        if det_pose is None or track_pose is None:
            return None, 0
        det_arr = np.asarray(det_pose, dtype=np.float32)
        track_arr = np.asarray(track_pose, dtype=np.float32)
        if (
            det_arr.shape != track_arr.shape
            or det_arr.ndim != 2
            or det_arr.shape[1] < 2
        ):
            return None, 0

        dists = []
        for kp_idx in range(len(det_arr)):
            det_valid = np.isfinite(det_arr[kp_idx, 0]) and np.isfinite(
                det_arr[kp_idx, 1]
            )
            track_valid = np.isfinite(track_arr[kp_idx, 0]) and np.isfinite(
                track_arr[kp_idx, 1]
            )
            if not (det_valid and track_valid):
                continue
            dist = float(np.linalg.norm(det_arr[kp_idx, :2] - track_arr[kp_idx, :2]))
            if np.isfinite(dist):
                dists.append(dist)

        if len(dists) < min_shared:
            return None, len(dists)

        dists_arr = np.asarray(dists, dtype=np.float32)
        med = float(np.median(dists_arr))
        abs_dev = np.abs(dists_arr - med)
        mad = float(np.median(abs_dev))
        if mad > 1e-6:
            keep = abs_dev <= (2.5 * mad)
            filtered = dists_arr[keep]
            if len(filtered) >= min_shared:
                dists_arr = filtered
        if len(dists_arr) >= 5:
            cutoff = max(1, int(np.floor(len(dists_arr) * 0.2)))
            dists_arr = (
                np.sort(dists_arr)[:-cutoff] if cutoff < len(dists_arr) else dists_arr
            )
        return float(np.mean(dists_arr)), int(len(dists_arr))

    def _advanced_association_enabled(self, association_data) -> bool:
        if not association_data:
            return False
        # A list full of None values means pose is disabled — only activate the
        # advanced cost matrix when there is at least one actual (non-None) keypoint
        # or prototype.  Using the key's mere presence (e.g. [None, None, …]) caused
        # the worker to always take the advanced path while the optimizer/preview took
        # the simple Numba path, making tuned parameters diverge in production.
        kpts = association_data.get("detection_pose_keypoints")
        protos = association_data.get("track_pose_prototypes")
        has_kpts = kpts is not None and any(k is not None for k in kpts)
        has_protos = protos is not None and any(p is not None for p in protos)
        return has_kpts or has_protos

    def _compute_stage1_gate(
        self,
        N,
        M,
        meas_pos,
        pred_pos,
        shapes_area,
        shapes_asp,
        prev_areas,
        prev_asps,
        S_inv_batch,
        track_uncertainty,
        track_avg_step,
        cull_threshold,
    ):
        p = self.params
        reference_body_size = max(1.0, float(p.get("REFERENCE_BODY_SIZE", 20.0)))
        gate_multiplier = float(p.get("ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER", 1.4))
        max_area_ratio = float(p.get("ASSOCIATION_STAGE1_MAX_AREA_RATIO", 2.5))
        max_aspect_diff = float(p.get("ASSOCIATION_STAGE1_MAX_ASPECT_DIFF", 0.8))
        uncertainty_ref = max(1.0, reference_body_size**2)

        # --- Vectorized position distances (N × M) ---
        diff = meas_pos[None, :, :] - pred_pos[:, None, :]  # (N, M, 2)
        if p["USE_MAHALANOBIS"]:
            S_inv_2x2 = S_inv_batch[:, :2, :2]  # (N, 2, 2)
            pos_dist = np.sqrt(np.einsum("nmd,nde,nme->nm", diff, S_inv_2x2, diff))
        else:
            pos_dist = np.linalg.norm(diff, axis=2)  # (N, M)

        # --- Per-track adaptive gate threshold ---
        unc_scale = np.minimum(2.0, track_uncertainty / uncertainty_ref)
        mot_scale = np.minimum(2.0, track_avg_step / reference_body_size)
        local_gates = (
            cull_threshold
            * gate_multiplier
            * (1.0 + 0.5 * unc_scale + 0.35 * mot_scale)
        )  # (N,)

        # --- Vectorized area ratio and aspect diff ---
        _prev = np.maximum(prev_areas, 1e-6)[:, None]  # (N, 1)
        _curr = np.maximum(shapes_area, 1e-6)[None, :]  # (1, M)
        area_ratio = np.maximum(_prev, _curr) / np.maximum(
            np.minimum(_prev, _curr), 1e-6
        )
        asp_diff = np.abs(shapes_asp[None, :] - prev_asps[:, None])

        # --- Build boolean pass mask and extract candidates ---
        mask = (
            (pos_dist <= local_gates[:, None])
            & (area_ratio <= max_area_ratio)
            & (asp_diff <= max_aspect_diff)
        )

        candidates = {}
        for i in range(N):
            indices = np.where(mask[i])[0]
            if len(indices) > 0:
                candidates[i] = indices.tolist()
        return candidates

    def _compute_advanced_cost_matrix(
        self,
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
        meas_ori_directed,
        association_data,
    ):
        p = self.params
        cost = np.full((N, M), 1e6, dtype=np.float32)
        track_uncertainty = np.asarray(
            association_data.get("track_position_uncertainty", np.zeros(N)),
            dtype=np.float32,
        )
        track_avg_step = np.asarray(
            association_data.get("track_avg_step", np.zeros(N)), dtype=np.float32
        )
        candidates = self._compute_stage1_gate(
            N,
            M,
            meas_pos,
            pred_pos,
            shapes_area,
            shapes_asp,
            prev_areas,
            prev_asps,
            S_inv_batch,
            track_uncertainty,
            track_avg_step,
            min(
                max(
                    self.params.get("MAX_DISTANCE_THRESHOLD", 1000.0)
                    / max(self.params.get("W_POSITION", 1.0), 1e-6),
                    50.0,
                ),
                self.params.get("MAX_DISTANCE_THRESHOLD", 1000.0) * 3.0,
            ),
        )

        detection_pose_keypoints = list(
            association_data.get("detection_pose_keypoints", [None] * M)
        )
        detection_pose_visibility = np.asarray(
            association_data.get("detection_pose_visibility", np.zeros(M)),
            dtype=np.float32,
        )
        track_pose_prototypes = list(
            association_data.get("track_pose_prototypes", [None] * N)
        )
        pose_rejection_enabled = bool(p.get("ENABLE_POSE_REJECTION", True))
        pose_veto_threshold = float(p.get("POSE_REJECTION_THRESHOLD", 0.5))
        pose_min_visibility = float(p.get("POSE_REJECTION_MIN_VISIBILITY", 0.5))

        # --- AprilTag identity cost ---
        _det_tag_ids = association_data.get("detection_tag_ids", [])
        _track_tag_ids = association_data.get("track_last_tag_ids", [])
        _tag_match_bonus = float(p.get("TAG_MATCH_BONUS", 20.0))
        _tag_mismatch_penalty = float(p.get("TAG_MISMATCH_PENALTY", 50.0))
        _NO_TAG = -1

        # --- CNN Classifier identity cost config (multi-phase) ---
        _cnn_phases = association_data.get("cnn_phases", None)
        # Backward compat: old flat keys
        if _cnn_phases is None:
            _old_det = association_data.get("detection_cnn_classes")
            _old_trk = association_data.get("track_cnn_identities")
            if _old_det is not None and _old_trk is not None:
                _cnn_phases = [
                    {
                        "label": "cnn_identity",
                        "match_bonus": float(p.get("CNN_CLASSIFIER_MATCH_BONUS", 20.0)),
                        "mismatch_penalty": float(
                            p.get("CNN_CLASSIFIER_MISMATCH_PENALTY", 50.0)
                        ),
                        "detection_classes": _old_det,
                        "track_identities": _old_trk,
                    }
                ]
            else:
                _cnn_phases = []

        # Identity-scale normalization:
        # N_total = AprilTags (1 if any det has a tag) + number of CNN phases
        # Each factor contributes 1/N_total of its bonus/penalty.
        # Full bonus/penalty only when ALL factors agree.
        # When N_total==1, scale==1.0 (no change from current single-method behavior).
        _n_identity_factors = (1 if _det_tag_ids else 0) + len(_cnn_phases)
        _identity_scale = 1.0 / _n_identity_factors if _n_identity_factors > 1 else 1.0

        for track_idx, det_indices in candidates.items():
            inv_S = S_inv_batch[track_idx, :2, :2]
            pred_theta = pred_ori[track_idx]
            for det_idx in det_indices:
                diff = meas_pos[det_idx] - pred_pos[track_idx]
                pos_cost = (
                    float(np.sqrt(diff @ inv_S @ diff))
                    if p["USE_MAHALANOBIS"]
                    else float(np.linalg.norm(diff))
                )

                visibility = (
                    float(detection_pose_visibility[det_idx])
                    if det_idx < len(detection_pose_visibility)
                    else 0.0
                )
                visibility = float(np.clip(visibility, 0.0, 1.0))
                det_pose_proto = (
                    detection_pose_keypoints[det_idx]
                    if det_idx < len(detection_pose_keypoints)
                    else None
                )
                track_pose_proto = track_pose_prototypes[track_idx]
                pose_dist, shared_keypoints = self._pose_paired_stats(
                    det_pose_proto, track_pose_proto
                )
                adaptive_pose_threshold = pose_veto_threshold
                if shared_keypoints > 0:
                    if shared_keypoints <= 3 or visibility < min(
                        1.0, pose_min_visibility + 0.15
                    ):
                        adaptive_pose_threshold *= 1.2
                if (
                    pose_rejection_enabled
                    and pose_dist is not None
                    and visibility >= pose_min_visibility
                    and pose_dist > adaptive_pose_threshold
                ):
                    continue

                orientation_cost = float(p["W_ORIENTATION"]) * self._orientation_diff(
                    pred_theta,
                    meas_ori[det_idx],
                    (
                        bool(meas_ori_directed[det_idx])
                        if det_idx < len(meas_ori_directed)
                        else False
                    ),
                )

                area_cost = float(p["W_AREA"]) * abs(
                    float(shapes_area[det_idx]) - float(prev_areas[track_idx])
                )
                aspect_cost = float(p["W_ASPECT"]) * abs(
                    float(shapes_asp[det_idx]) - float(prev_asps[track_idx])
                )

                motion_core_cost = (
                    float(p["W_POSITION"]) * pos_cost
                    + orientation_cost
                    + area_cost
                    + aspect_cost
                )

                # --- AprilTag identity bonus / penalty ---
                _dt = _det_tag_ids[det_idx] if det_idx < len(_det_tag_ids) else _NO_TAG
                _tt = (
                    _track_tag_ids[track_idx]
                    if track_idx < len(_track_tag_ids)
                    else _NO_TAG
                )
                if _dt != _NO_TAG and _tt != _NO_TAG:
                    if _dt == _tt:
                        motion_core_cost -= _tag_match_bonus * _identity_scale
                    else:
                        motion_core_cost += _tag_mismatch_penalty * _identity_scale

                # --- CNN identity bonus / penalty (multi-phase) ---
                for _phase in _cnn_phases:
                    _dc = _phase["detection_classes"]
                    _ti = _phase["track_identities"]
                    _det_cls = _dc[det_idx] if det_idx < len(_dc) else None
                    _trk_cls = _ti[track_idx] if track_idx < len(_ti) else None
                    motion_core_cost = apply_cnn_identity_cost(
                        motion_core_cost,
                        _det_cls,
                        _trk_cls,
                        _phase["match_bonus"] * _identity_scale,
                        _phase["mismatch_penalty"] * _identity_scale,
                    )

                cost[track_idx, det_idx] = motion_core_cost

        return cost, candidates

    def compute_cost_matrix(
        self,
        N: int,
        measurements: List[np.ndarray],
        predictions: np.ndarray,
        shapes: List[Tuple[float, float]],
        kf_manager: Any,
        last_shape_info: List[Any],
        meas_ori_directed: np.ndarray | None = None,
        association_data: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Computes cost matrix. Compatible with Vectorized Kalman Filter.
        """
        p = self.params
        M = len(measurements)
        if M == 0:
            return np.zeros((N, 0), np.float32), {}

        # Warn about spatial indexing for large N
        if (
            N > 25
            and not self._spatial_optimization_enabled()
            and not self._large_n_warning_shown
        ):
            warning_msg = (
                f"Tracking {N} objects without spatial indexing may be slow.\n\n"
                f"Consider enabling these optimizations in tracking_config.json:\n"
                f"  • ENABLE_SPATIAL_OPTIMIZATION: true\n"
                f"  • ENABLE_GREEDY_ASSIGNMENT: true\n\n"
                f"Expected performance improvement: 10-30% for {N}+ objects."
            )
            logger.warning(warning_msg.replace("\n", " "))
            if self.worker is not None:
                self.worker.warning_signal.emit(
                    "Performance Optimization Available", warning_msg
                )
            self._large_n_warning_shown = True

        # Get pre-calculated Inverse Innovation Covariances from Manager
        S_inv_batch = kf_manager.get_mahalanobis_matrices()

        # Pre-extract arrays for Numba (Avoids attribute access in loop)
        meas_pos = np.array([m[:2] for m in measurements], dtype=np.float32)
        meas_ori = np.array([m[2] for m in measurements], dtype=np.float32)
        if meas_ori_directed is None:
            meas_ori_directed_arr = np.zeros(M, dtype=np.uint8)
        else:
            meas_ori_directed_arr = np.asarray(meas_ori_directed, dtype=np.uint8)
            if len(meas_ori_directed_arr) != M:
                logger.warning(
                    "meas_ori_directed length mismatch (%d != %d); falling back to axis mode.",
                    len(meas_ori_directed_arr),
                    M,
                )
                meas_ori_directed_arr = np.zeros(M, dtype=np.uint8)
        pred_pos = predictions[:, :2]  # Predictions are already (N, 3)
        pred_ori = predictions[:, 2]

        # Override meas_ori with the directed heading where pose/headtail detected direction.
        # This ensures orientation cost in ALL paths (Numba, spatial, advanced) uses the
        # correct directed angle rather than the 180°-ambiguous OBB axis angle.
        if association_data is not None:
            _dh = association_data.get("detection_pose_heading")
            if _dh is not None:
                _dh_arr = np.asarray(_dh, dtype=np.float32)
                for _j in range(min(M, len(_dh_arr))):
                    if meas_ori_directed_arr[_j] == 1 and np.isfinite(_dh_arr[_j]):
                        meas_ori[_j] = _dh_arr[_j]

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
            min(
                max(MAX_DIST / max(p.get("W_POSITION", 1.0), 1e-6), 50.0),
                MAX_DIST * 3.0,  # never search beyond 3× the hard distance limit
            )
            if p.get("W_POSITION", 1.0) > 0
            else 1e6
        )

        if self._advanced_association_enabled(association_data):
            track_uncertainty = (
                np.asarray(kf_manager.get_position_uncertainties(), dtype=np.float32)
                if hasattr(kf_manager, "get_position_uncertainties")
                else np.trace(kf_manager.P[:, :2, :2], axis1=1, axis2=2).astype(
                    np.float32
                )
            )
            advanced_data = dict(association_data or {})
            advanced_data["track_position_uncertainty"] = track_uncertainty
            return self._compute_advanced_cost_matrix(
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
                meas_ori_directed_arr,
                advanced_data,
            )

        if self._spatial_optimization_enabled() and N > 50:
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
                meas_ori_directed_arr,
            )
            return cost, spatial_candidates

        # Default: Full Numba Matrix
        cost = _compute_cost_matrix_numba(
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
            meas_ori_directed_arr,
        )

        return cost, {}

    def compute_assignment_confidence(
        self: object, cost: object, matched_pairs: object
    ) -> object:
        """Compute confidence scores for assignments."""
        if not matched_pairs:
            return {}
        scale = self.params.get("MAX_DISTANCE_THRESHOLD", 100.0) * 0.5
        return {r: 1.0 / (1.0 + cost[r, c] / scale) for r, c in matched_pairs}

    def assign_tracks(
        self: object,
        cost: object,
        N: object,
        M: object,
        meas: object,
        track_states: object,
        tracking_continuity: object,
        kf_manager: object,
        trajectory_ids: object,
        next_trajectory_id: object,
        spatial_candidates: object = None,
        association_data: Dict[str, Any] | None = None,
    ) -> object:
        """
        Drop-in replacement for track assignment logic.
        Compatible with kf_manager.X state access.
        """
        p = self.params
        if M == 0:
            return [], [], [], next_trajectory_id, []

        # KALMAN_MATURITY_AGE is a frame count — use it to gate Phase-1 (Hungarian)
        # vs Phase-2 (greedy). CONTINUITY_THRESHOLD stores a pixel value for the
        # recovery search distance and must NOT be used here.
        THRESH = p.get("KALMAN_MATURITY_AGE", 10)
        MAX_DIST = p["MAX_DISTANCE_THRESHOLD"]
        USE_GREEDY = p.get("ENABLE_GREEDY_ASSIGNMENT", False)

        # Velocity-plausibility gate for established (Phase-1) tracks.
        #
        # After predict(), kf_manager.X[r, :2] is already at the KF-predicted
        # position for this frame — it includes the track's own velocity step.
        # The "innovation" (distance from predicted pos → matched detection) must
        # therefore be bounded by at most one frame of maximum physical movement.
        # Any larger innovation means the solver is trying to snap the track to a
        # detection that requires a physically impossible acceleration — almost
        # always a cross-identity swap.  When rejected, the track is simply left
        # unmatched and treated as occluded for this frame.
        #
        #   VEL_GATE = KALMAN_MAX_VELOCITY_MULTIPLIER × body_size (pixels / frame)
        #
        # Phase-2 (unstable) and Phase-3 (respawn) use the looser MAX_DIST gate
        # because those tracks do not yet have a reliable velocity estimate.
        _body_size = p.get("REFERENCE_BODY_SIZE", 20.0) * p.get("RESIZE_FACTOR", 1.0)
        VEL_GATE = p.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0) * _body_size

        # Pre-gate: block any (track, detection) pair whose raw Euclidean distance
        # exceeds the per-track gate before the Hungarian algorithm runs.
        # When USE_MAHALANOBIS=True and a track's covariance P is large (coasting),
        # Mahalanobis distances collapse toward zero for ALL detections — even those
        # on the other side of the arena.  These near-zero costs corrupt the global
        # optimisation, causing perfectly-tracked nearby tracks to lose their
        # detections to the uncertain coasting track.  Setting cost=1e9 here
        # prevents physically impossible assignments from being selected.
        #
        # Young tracks (continuity < KALMAN_MATURITY_AGE) have noisy KF predictions
        # because they lack reliable velocity estimates; their predictions can drift
        # far from the actual animal within a few frames.  We expand the gate for
        # those tracks only, using KALMAN_YOUNG_GATE_MULTIPLIER, so the caller
        # does not need to inflate MAX_DISTANCE_THRESHOLD globally (which would also
        # widen the gate for mature tracks and increase swap probability there).
        # Default is 1.0 so that existing configs without this key keep the old
        # behaviour (uniform MAX_DIST gate for all tracks).  The auto-tuner's
        # _DEFAULTS starts at 1.5, giving it the right search direction without
        # breaking runs that haven't been re-tuned.
        _young_mult = max(1.0, float(p.get("KALMAN_YOUNG_GATE_MULTIPLIER", 1.0)))
        _per_track_gate = np.where(
            np.array([tracking_continuity[r] for r in range(N)], dtype=np.float32)
            < THRESH,
            MAX_DIST * _young_mult,
            MAX_DIST,
        )  # shape (N,); young tracks get wider gate
        meas_xy = np.array([meas[j][:2] for j in range(M)], dtype=np.float32)
        raw_dist_mat = np.linalg.norm(
            np.asarray(kf_manager.X[:N, :2], dtype=np.float32)[:, None, :]
            - meas_xy[None, :, :],
            axis=2,
        )  # shape (N, M)
        cost[raw_dist_mat >= _per_track_gate[:, None]] = 1e9

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
                # Optimized Greedy — velocity gate applied to established tracks.
                track_det_costs = []
                for r in est:
                    for c in range(M):
                        if cost[r, c] < MAX_DIST and raw_dist_mat[r, c] < VEL_GATE:
                            track_det_costs.append((cost[r, c], r, c))
                track_det_costs.sort()
                assigned_r = set()
                for _, r, c in track_det_costs:
                    if r not in assigned_r and c not in assigned_dets:
                        all_assignments.append((r, c))
                        assigned_dets.add(c)
                        assigned_r.add(r)
            else:
                # Hungarian — velocity gate replaces the old raw_dist < MAX_DIST check.
                rows, cols = linear_sum_assignment(cost[est, :])
                for r_idx, c in zip(rows, cols):
                    r = est[r_idx]
                    if cost[r, c] < MAX_DIST and raw_dist_mat[r, c] < VEL_GATE:
                        all_assignments.append((r, c))
                        assigned_dets.add(c)

        # Phase 2: Unstable Tracks — expanded per-track gate (velocity not yet reliable).
        for r in sorted(unst, key=lambda i: tracking_continuity[i], reverse=True):
            avail = [j for j in range(M) if j not in assigned_dets]
            if not avail:
                break
            best_c = avail[np.argmin(cost[r, avail])]
            raw_dist = float(
                np.linalg.norm(np.asarray(meas[best_c][:2]) - kf_manager.X[r, :2])
            )
            if cost[r, best_c] < MAX_DIST and raw_dist < float(_per_track_gate[r]):
                all_assignments.append((r, best_c))
                assigned_dets.add(best_c)

        # Phase 3: Respawn Lost Tracks
        unassigned = [j for j in range(M) if j not in assigned_dets]
        respawn_dist_limit = p.get("MIN_RESPAWN_DISTANCE", MAX_DIST * 0.8)
        non_lost_positions = [
            np.asarray(kf_manager.X[r, :2], dtype=np.float32)
            for r in range(N)
            if track_states[r] != "lost"
        ]

        for c in unassigned:
            if not lost:
                break
            # Block respawns near any currently active or occluded track prediction.
            min_dist_non_lost = (
                min(
                    np.linalg.norm(meas[c][:2] - track_pos)
                    for track_pos in non_lost_positions
                )
                if non_lost_positions
                else 1e6
            )

            if min_dist_non_lost >= respawn_dist_limit:
                best_r, best_c_val = None, 1e6
                for r in lost:
                    # Accessing vectorized manager state X: [x, y, theta, vx, vy]
                    last_pos = kf_manager.X[r, :2]
                    dist = np.linalg.norm(meas[c][:2] - last_pos)
                    if dist < best_c_val:
                        best_c_val, best_r = dist, r

                if best_r is not None and best_c_val < MAX_DIST:
                    all_assignments.append((best_r, c))
                    assigned_dets.add(c)
                    lost.remove(best_r)
                    trajectory_ids[best_r] = next_trajectory_id
                    next_trajectory_id += 1

        if not all_assignments:
            return [], [], list(range(M)), next_trajectory_id, []

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
        meas_ori_directed,
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
                if meas_ori_directed[c] == 0:
                    odiff = min(odiff, np.pi - odiff)

                cost[r, c] = (
                    Wp * pos_c
                    + Wo * odiff
                    + Wa * abs(sh_area[c] - pr_area[r])
                    + Wasp * abs(sh_asp[c] - pr_asp[r])
                )
        return cost
