from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module

assigner_mod = load_src_module(
    "multi_tracker/core/assigners/hungarian.py",
    "assigner_under_test",
)


def _compute_cost_matrix_sota_py(
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
    cost = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        inv_S_pos = S_inv_batch[i, :2, :2]
        for j in range(M):
            diff = meas_pos[j] - pred_pos[i]
            if use_maha:
                pos_dist = float(np.sqrt(diff @ inv_S_pos @ diff))
            else:
                pos_dist = float(np.linalg.norm(diff))
            if pos_dist > cull_threshold:
                cost[i, j] = 1e6
                continue
            odiff = abs(pred_ori[i] - meas_ori[j])
            if odiff > np.pi:
                odiff = 2 * np.pi - odiff
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])
            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff
    return cost


assigner_mod._compute_cost_matrix_sota = _compute_cost_matrix_sota_py
TrackAssigner = assigner_mod.TrackAssigner


class _DummyKF:
    def __init__(self, n: int):
        self.X = np.zeros((n, 5), dtype=np.float32)

    def get_mahalanobis_matrices(self):
        mats = np.zeros((len(self.X), 3, 3), dtype=np.float32)
        for i in range(len(self.X)):
            mats[i] = np.eye(3, dtype=np.float32)
        return mats


def _params() -> dict:
    return {
        "USE_MAHALANOBIS": True,
        "W_POSITION": 1.0,
        "W_ORIENTATION": 0.2,
        "W_AREA": 0.02,
        "W_ASPECT": 0.02,
        "MAX_DISTANCE_THRESHOLD": 50.0,
        "CONTINUITY_THRESHOLD": 3,
        "ENABLE_GREEDY_ASSIGNMENT": False,
        "ENABLE_SPATIAL_OPTIMIZATION": False,
        "MIN_RESPAWN_DISTANCE": 15.0,
    }


def test_compute_cost_matrix_prefers_nearest_consistent_pairing() -> None:
    assigner = TrackAssigner(_params())
    kf = _DummyKF(2)
    predictions = np.array([[10.0, 10.0, 0.1], [100.0, 100.0, 1.0]], dtype=np.float32)
    measurements = [
        np.array([11.0, 10.5, 0.11], dtype=np.float32),
        np.array([101.0, 99.5, 1.02], dtype=np.float32),
    ]
    shapes = [(20.0, 1.3), (21.0, 1.2)]
    last_shape_info = [(20.0, 1.3), (21.0, 1.2)]

    cost, _ = assigner.compute_cost_matrix(
        N=2,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
    )

    assert cost.shape == (2, 2)
    assert cost[0, 0] < cost[0, 1]
    assert cost[1, 1] < cost[1, 0]


def test_assign_tracks_respawns_lost_track_when_valid_detection_exists() -> None:
    params = _params()
    assigner = TrackAssigner(params)
    kf = _DummyKF(2)
    kf.X[0, :2] = [10.0, 10.0]
    kf.X[1, :2] = [300.0, 300.0]

    measurements = [
        np.array([12.0, 11.0, 0.1], dtype=np.float32),
        np.array([330.0, 330.0, 1.5], dtype=np.float32),
    ]

    cost = np.array(
        [
            [2.0, 300.0],
            [300.0, 20.0],
        ],
        dtype=np.float32,
    )

    track_states = ["active", "lost"]
    continuity = [10, 0]
    trajectory_ids = [0, 1]

    rows, cols, free_dets, next_id, _ = assigner.assign_tracks(
        cost=cost,
        N=2,
        M=2,
        meas=measurements,
        track_states=track_states,
        tracking_continuity=continuity,
        kf_manager=kf,
        trajectory_ids=trajectory_ids,
        next_trajectory_id=2,
        spatial_candidates={},
    )

    assigned = list(zip(rows, cols))
    assert (0, 0) in assigned
    assert (1, 1) in assigned
    assert free_dets == []
    assert next_id == 3
    assert trajectory_ids[1] == 2
