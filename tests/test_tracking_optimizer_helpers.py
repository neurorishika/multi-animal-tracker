from __future__ import annotations

import types

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


class _StubDetectionFilter:
    def __init__(self, params):
        self.params = params

    def filter_raw_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        directed_mask=None,
    ):
        return (
            meas,
            sizes,
            shapes,
            confidences,
            obb,
            detection_ids or [],
            heading_hints or [],
            directed_mask or [],
        )


class _StubKalmanFilterManager:
    last_instance = None

    def __init__(self, n_targets, params):
        self.X = np.zeros((n_targets, 5), dtype=np.float32)
        self.P = np.tile(np.eye(5, dtype=np.float32)[None, :, :], (n_targets, 1, 1))
        self.corrected_measurements = []
        _StubKalmanFilterManager.last_instance = self

    def predict(self):
        return self.X

    def correct(self, track_idx, measurement):
        measurement = np.asarray(measurement, dtype=np.float32)
        self.corrected_measurements.append((int(track_idx), measurement.copy()))
        self.X[track_idx, :3] = measurement

    def initialize_filter(self, track_idx, state):
        self.X[track_idx] = np.asarray(state, dtype=np.float32)

    def get_mahalanobis_matrices(self):
        return np.tile(np.eye(3, dtype=np.float32)[None, :, :], (len(self.X), 1, 1))

    def get_position_uncertainties(self):
        return np.ones(len(self.X), dtype=np.float32)


class _StubTrackAssigner:
    last_association_data = None
    last_meas_ori_directed = None

    def __init__(self, params):
        self.params = params

    def compute_cost_matrix(
        self,
        N,
        meas,
        preds,
        shapes,
        kf_manager,
        last_shape_info,
        meas_ori_directed=None,
        association_data=None,
    ):
        _StubTrackAssigner.last_association_data = association_data
        _StubTrackAssigner.last_meas_ori_directed = np.asarray(
            meas_ori_directed, dtype=np.uint8
        )
        return np.zeros((N, len(meas)), dtype=np.float32), {}

    def assign_tracks(self, cost, N, M, meas, *args, **kwargs):
        next_trajectory_id = (
            args[4] if len(args) > 4 else kwargs.get("next_trajectory_id", N)
        )
        return [0], [0], [], next_trajectory_id, []


class _FakeCache:
    def get_frame(self, frame_idx):
        return (
            [np.array([10.0, 20.0, 0.0], dtype=np.float32)],
            [50.0],
            [(50.0, 1.0)],
            [0.95],
            [np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=np.float32)],
            [101],
            [1.25],
            [1],
            None,
            None,
            None,
        )


def _load_optimizer_module():
    multi_tracker_pkg = types.ModuleType("multi_tracker")
    multi_tracker_pkg.__path__ = []
    core_pkg = types.ModuleType("multi_tracker.core")
    core_pkg.__path__ = []
    core_tracking = types.ModuleType("multi_tracker.core.tracking")
    core_tracking.__path__ = []
    data_pkg = types.ModuleType("multi_tracker.data")
    data_pkg.__path__ = []

    pose_features = load_src_module(
        "multi_tracker/core/tracking/pose_features.py",
        "pose_features_for_optimizer_test",
    )

    qtcore = types.ModuleType("PySide6.QtCore")

    class Signal:
        def __init__(self, *args, **kwargs):
            self.emissions = []

        def emit(self, *args, **kwargs):
            self.emissions.append((args, kwargs))

    class QThread:
        def __init__(self, parent=None):
            self.parent = parent

    qtcore.Signal = Signal
    qtcore.QThread = QThread

    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore

    optuna = types.ModuleType("optuna")
    optuna.logging = types.SimpleNamespace(WARNING=0, set_verbosity=lambda *_args: None)
    optuna.samplers = types.SimpleNamespace(
        QMCSampler=object,
        RandomSampler=object,
        GPSampler=object,
        TPESampler=object,
    )
    optuna.create_study = lambda **_kwargs: None

    assigner = types.ModuleType("multi_tracker.core.assigners.hungarian")
    assigner.TrackAssigner = _StubTrackAssigner

    detectors_engine = types.ModuleType("multi_tracker.core.detectors.engine")
    detectors_engine.DetectionFilter = _StubDetectionFilter

    kalman = types.ModuleType("multi_tracker.core.filters.kalman")
    kalman.KalmanFilterManager = _StubKalmanFilterManager

    detection_cache = types.ModuleType("multi_tracker.data.detection_cache")
    detection_cache.DetectionCache = object

    stubs = {
        "cv2": make_cv2_stub(),
        "optuna": optuna,
        "PySide6": pyside,
        "PySide6.QtCore": qtcore,
        "multi_tracker": multi_tracker_pkg,
        "multi_tracker.core": core_pkg,
        "multi_tracker.core.tracking": core_tracking,
        "multi_tracker.data": data_pkg,
        "multi_tracker.core.tracking.pose_features": pose_features,
        "multi_tracker.core.assigners.hungarian": assigner,
        "multi_tracker.core.detectors.engine": detectors_engine,
        "multi_tracker.core.filters.kalman": kalman,
        "multi_tracker.data.detection_cache": detection_cache,
    }

    return load_src_module(
        "multi_tracker/core/tracking/optimizer.py",
        "tracking_optimizer_under_test",
        stubs=stubs,
    )


def test_optimizer_replay_uses_directed_heading_for_assignment_and_kf() -> None:
    mod = _load_optimizer_module()
    optimizer = mod.TrackingOptimizer(
        video_path="dummy.mp4",
        detection_cache_path="dummy.npz",
        start_frame=0,
        end_frame=0,
        base_params={},
        tuning_config={},
    )
    optimizer.cache = _FakeCache()
    optimizer._pose_run_context = (None, [], [], [], False)
    optimizer._pose_frame_cache = {}
    optimizer._stop_requested = False

    params = {
        "MAX_TARGETS": 1,
        "REFERENCE_BODY_SIZE": 20.0,
        "RESIZE_FACTOR": 1.0,
        "LOST_THRESHOLD_FRAMES": 5,
        "POSE_OVERRIDES_HEADTAIL": True,
    }

    score, _, _ = optimizer._run_tracking_loop(params)

    assert np.isfinite(score)
    np.testing.assert_allclose(
        _StubTrackAssigner.last_association_data["detection_pose_heading"],
        np.array([1.25], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(_StubTrackAssigner.last_meas_ori_directed, [1])
    _, corrected = _StubKalmanFilterManager.last_instance.corrected_measurements[0]
    assert corrected[2] == np.float32(1.25)
