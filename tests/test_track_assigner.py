from __future__ import annotations

import numpy as np

from tests.helpers.module_loader import load_src_module

assigner_mod = load_src_module(
    "hydra_suite/core/assigners/hungarian.py",
    "assigner_under_test",
)


def _compute_cost_matrix_numba_py(
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
            if meas_ori_directed[j] == 0:
                odiff = min(odiff, np.pi - odiff)
            area_diff = abs(shapes_area[j] - prev_areas[i])
            asp_diff = abs(shapes_asp[j] - prev_asps[i])
            cost[i, j] = Wp * pos_dist + Wo * odiff + Wa * area_diff + Wasp * asp_diff
    return cost


assigner_mod._compute_cost_matrix_numba = _compute_cost_matrix_numba_py
TrackAssigner = assigner_mod.TrackAssigner


class _DummyKF:
    def __init__(self, n: int):
        self.X = np.zeros((n, 5), dtype=np.float32)
        self.P = np.stack([np.eye(5, dtype=np.float32) for _ in range(n)])

    def get_mahalanobis_matrices(self):
        mats = np.zeros((len(self.X), 3, 3), dtype=np.float32)
        for i in range(len(self.X)):
            mats[i] = np.eye(3, dtype=np.float32)
        return mats

    def get_position_uncertainties(self):
        return [float(np.trace(self.P[i, :2, :2])) for i in range(len(self.X))]


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
        "REFERENCE_BODY_SIZE": 20.0,
        "W_POSE_DIRECTION": 0.75,
        "W_POSE_LENGTH": 0.1,
        "POSE_VALID_ORIENTATION_SCALE": 0.15,
        "ASSOCIATION_STAGE1_MOTION_GATE_MULTIPLIER": 1.4,
        "ASSOCIATION_STAGE1_MAX_AREA_RATIO": 2.5,
        "ASSOCIATION_STAGE1_MAX_ASPECT_DIFF": 0.8,
        "TRACK_FEATURE_EMA_ALPHA": 0.85,
        "ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD": 0.7,
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


def test_assign_tracks_does_not_respawn_near_non_lost_track() -> None:
    params = _params()
    assigner = TrackAssigner(params)
    kf = _DummyKF(2)
    kf.X[0, :2] = [10.0, 10.0]
    kf.X[1, :2] = [14.0, 14.0]

    measurements = [
        np.array([12.0, 12.0, 0.1], dtype=np.float32),
    ]

    cost = np.array(
        [
            [80.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    track_states = ["active", "lost"]
    continuity = [10, 0]
    trajectory_ids = [0, 1]

    rows, cols, free_dets, next_id, _ = assigner.assign_tracks(
        cost=cost,
        N=2,
        M=1,
        meas=measurements,
        track_states=track_states,
        tracking_continuity=continuity,
        kf_manager=kf,
        trajectory_ids=trajectory_ids,
        next_trajectory_id=2,
        spatial_candidates={},
    )

    assert rows == []
    assert cols == []
    assert free_dets == [0]
    assert next_id == 2
    assert trajectory_ids == [0, 1]


def test_assign_tracks_returns_five_values_when_no_measurements() -> None:
    assigner = TrackAssigner(_params())
    kf = _DummyKF(2)
    result = assigner.assign_tracks(
        cost=np.zeros((2, 0), dtype=np.float32),
        N=2,
        M=0,
        meas=[],
        track_states=["active", "lost"],
        tracking_continuity=[5, 0],
        kf_manager=kf,
        trajectory_ids=[0, 1],
        next_trajectory_id=2,
        spatial_candidates={},
    )

    assert len(result) == 5
    rows, cols, free_dets, next_id, high_cost_tracks = result
    assert rows == []
    assert cols == []
    assert free_dets == []
    assert next_id == 2
    assert high_cost_tracks == []


def test_orientation_cost_uses_axis_equivalence_by_default() -> None:
    assigner = TrackAssigner(_params())
    kf = _DummyKF(1)
    predictions = np.array([[20.0, 20.0, 0.1]], dtype=np.float32)
    measurements = [
        np.array([20.0, 20.0, 0.1], dtype=np.float32),
        np.array([20.0, 20.0, 0.1 + np.pi], dtype=np.float32),
    ]
    shapes = [(30.0, 1.2), (30.0, 1.2)]
    last_shape_info = [(30.0, 1.2)]

    cost, _ = assigner.compute_cost_matrix(
        N=1,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
    )

    assert abs(float(cost[0, 0]) - float(cost[0, 1])) < 1e-6


def test_orientation_cost_respects_directed_measurements_when_flagged() -> None:
    assigner = TrackAssigner(_params())
    kf = _DummyKF(1)
    predictions = np.array([[20.0, 20.0, 0.1]], dtype=np.float32)
    measurements = [
        np.array([20.0, 20.0, 0.1], dtype=np.float32),
        np.array([20.0, 20.0, 0.1 + np.pi], dtype=np.float32),
    ]
    shapes = [(30.0, 1.2), (30.0, 1.2)]
    last_shape_info = [(30.0, 1.2)]

    cost, _ = assigner.compute_cost_matrix(
        N=1,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        meas_ori_directed=np.array([1, 1], dtype=np.uint8),
    )

    assert float(cost[0, 0]) < float(cost[0, 1])


def test_advanced_cost_matrix_uses_pose_as_veto_only() -> None:
    params = _params()
    assigner = TrackAssigner(params)
    kf = _DummyKF(2)
    kf.X[0, :3] = [10.0, 10.0, 0.0]
    kf.X[1, :3] = [12.0, 10.0, np.pi]
    predictions = np.array([[10.0, 10.0, 0.0], [12.0, 10.0, np.pi]], dtype=np.float32)
    measurements = [
        np.array([11.0, 10.0, 0.0], dtype=np.float32),
        np.array([11.5, 10.0, np.pi], dtype=np.float32),
    ]
    shapes = [(20.0, 1.3), (20.0, 1.3)]
    last_shape_info = [(20.0, 1.3), (20.0, 1.3)]
    association_data = {
        "detection_pose_heading": np.array([0.0, np.pi], dtype=np.float32),
        "detection_pose_keypoints": [
            np.array(
                [[0.0, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [1.2, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.4, 1.0], [0.0, 0.8, 1.0], [0.0, 1.2, 1.0]],
                dtype=np.float32,
            ),
        ],
        "detection_pose_visibility": np.array([1.0, 1.0], dtype=np.float32),
        "track_pose_prototypes": [
            np.array(
                [[0.0, 0.0, 1.0], [0.42, 0.0, 1.0], [0.82, 0.0, 1.0], [1.22, 0.0, 1.0]],
                dtype=np.float32,
            ),
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 0.42, 1.0], [0.0, 0.82, 1.0], [0.0, 1.22, 1.0]],
                dtype=np.float32,
            ),
        ],
        "track_avg_step": [4.0, 4.0],
    }

    cost, candidates = assigner.compute_cost_matrix(
        N=2,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        meas_ori_directed=np.array([1, 1], dtype=np.uint8),
        association_data=association_data,
    )

    assert 0 in candidates and 1 in candidates
    assert float(cost[0, 0]) < 1e6
    assert float(cost[1, 1]) < 1e6
    assert float(cost[0, 1]) >= 1e6
    assert float(cost[1, 0]) >= 1e6


def test_lost_track_respawn_uses_nearest_lost_track_only() -> None:
    params = _params()
    assigner = TrackAssigner(params)
    kf = _DummyKF(2)
    kf.X[0, :2] = [10.0, 10.0]
    kf.X[1, :2] = [14.0, 10.0]

    cost = np.full((2, 1), 1e6, dtype=np.float32)
    measurements = [np.array([12.0, 10.0, 0.0], dtype=np.float32)]
    track_states = ["lost", "lost"]
    continuity = [0, 0]
    trajectory_ids = [3, 4]

    rows, cols, free_dets, next_id, _ = assigner.assign_tracks(
        cost=cost,
        N=2,
        M=1,
        meas=measurements,
        track_states=track_states,
        tracking_continuity=continuity,
        kf_manager=kf,
        trajectory_ids=trajectory_ids,
        next_trajectory_id=5,
        spatial_candidates={},
    )

    assert rows == [0]
    assert cols == [0]
    assert free_dets == []
    assert next_id == 6
    assert trajectory_ids[0] == 5


def test_pose_rejection_is_more_permissive_with_weak_visibility() -> None:
    params = _params()
    params["POSE_REJECTION_THRESHOLD"] = 0.5
    assigner = TrackAssigner(params)
    kf = _DummyKF(2)
    predictions = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
    measurements = [
        np.array([2.9, 0.0, 0.0], dtype=np.float32),
        np.array([3.1, 0.0, 0.0], dtype=np.float32),
    ]
    shapes = [(20.0, 1.3), (20.0, 1.3)]
    last_shape_info = [(20.0, 1.3), (20.0, 1.3)]
    track_pose = np.array(
        [[0.0, 0.0, 1.0], [0.4, 0.0, 1.0], [0.8, 0.0, 1.0], [1.2, 0.0, 1.0]],
        dtype=np.float32,
    )
    crossing_det_pose = np.array(
        [[0.0, 0.0, 1.0], [0.9, 0.0, 1.0], [1.45, 0.0, 1.0], [2.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    strict_cost, _ = assigner.compute_cost_matrix(
        N=2,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        association_data={
            "track_avg_step": [2.0, 2.0],
            "detection_pose_keypoints": [crossing_det_pose, crossing_det_pose],
            "detection_pose_visibility": np.array([0.95, 0.95], dtype=np.float32),
            "track_pose_prototypes": [track_pose, track_pose],
        },
    )

    permissive_cost, _ = assigner.compute_cost_matrix(
        N=2,
        measurements=measurements,
        predictions=predictions,
        shapes=shapes,
        kf_manager=kf,
        last_shape_info=last_shape_info,
        association_data={
            "track_avg_step": [2.0, 2.0],
            "detection_pose_keypoints": [crossing_det_pose, crossing_det_pose],
            "detection_pose_visibility": np.array([0.45, 0.45], dtype=np.float32),
            "track_pose_prototypes": [track_pose, track_pose],
        },
    )

    assert float(strict_cost[0, 0]) >= 1e6
    assert float(permissive_cost[0, 0]) < 1e6


def test_advanced_association_disabled_when_all_keypoints_none():
    """All-None keypoint list must NOT trigger the advanced cost matrix path."""
    params = _params()
    params["MAX_TARGETS"] = 2
    assigner = TrackAssigner(params)
    data_all_none = {
        "detection_pose_keypoints": [None, None],
        "track_pose_prototypes": [None, None],
    }
    assert not assigner._advanced_association_enabled(data_all_none)


def test_advanced_association_enabled_when_keypoint_present():
    """Non-None keypoint must trigger the advanced cost matrix path."""
    params = _params()
    params["MAX_TARGETS"] = 2
    assigner = TrackAssigner(params)
    kpt = np.zeros((4, 3), dtype=np.float32)
    data_with_kpt = {
        "detection_pose_keypoints": [kpt, None],
        "track_pose_prototypes": [None, None],
    }
    assert assigner._advanced_association_enabled(data_with_kpt)
