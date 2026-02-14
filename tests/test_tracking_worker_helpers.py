from __future__ import annotations

import types

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


def _load_worker_module():
    # Minimal QtCore stub
    qtcore = types.ModuleType("PySide6.QtCore")

    class Signal:
        def __init__(self, *args, **kwargs):
            self.emissions = []

        def emit(self, *args, **kwargs):
            self.emissions.append((args, kwargs))

    class QThread:
        def __init__(self, parent=None):
            self.parent = parent

    class QMutex:
        def lock(self):
            return None

        def unlock(self):
            return None

    def Slot(*args, **kwargs):
        def deco(fn):
            return fn

        return deco

    qtcore.Signal = Signal
    qtcore.QThread = QThread
    qtcore.QMutex = QMutex
    qtcore.Slot = Slot

    pyside = types.ModuleType("PySide6")
    pyside.QtCore = qtcore

    # Utility and core dependency stubs to avoid importing heavy runtime modules.
    image_processing = types.ModuleType("multi_tracker.utils.image_processing")
    image_processing.apply_image_adjustments = lambda *args, **kwargs: args[0]
    image_processing.stabilize_lighting = lambda *args, **kwargs: (args[0], None, 0.0)

    geometry = types.ModuleType("multi_tracker.utils.geometry")
    geometry.wrap_angle_degs = lambda x: x

    detection_cache = types.ModuleType("multi_tracker.data.detection_cache")
    detection_cache.DetectionCache = object

    batch_optimizer = types.ModuleType("multi_tracker.utils.batch_optimizer")
    batch_optimizer.BatchOptimizer = object

    frame_prefetcher = types.ModuleType("multi_tracker.utils.frame_prefetcher")

    class FramePrefetcher:
        def __init__(self, cap, buffer_size=2):
            self.cap = cap
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True

        def read(self):
            return self.cap.read()

        def stop(self):
            self.stopped = True

    frame_prefetcher.FramePrefetcher = FramePrefetcher

    # Stub core submodules imported by worker.
    core_pkg = types.ModuleType("multi_tracker.core")
    core_filters = types.ModuleType("multi_tracker.core.filters")
    core_background = types.ModuleType("multi_tracker.core.background")
    core_detectors = types.ModuleType("multi_tracker.core.detectors")
    core_assigners = types.ModuleType("multi_tracker.core.assigners")
    core_identity = types.ModuleType("multi_tracker.core.identity")

    kalman = types.ModuleType("multi_tracker.core.filters.kalman")
    kalman.KalmanFilterManager = object

    background_model = types.ModuleType("multi_tracker.core.background.model")
    background_model.BackgroundModel = object

    detectors_engine = types.ModuleType("multi_tracker.core.detectors.engine")
    detectors_engine.create_detector = lambda *_args, **_kwargs: None

    assigner = types.ModuleType("multi_tracker.core.assigners.hungarian")
    assigner.TrackAssigner = object

    identity = types.ModuleType("multi_tracker.core.identity.analysis")
    identity.IndividualDatasetGenerator = object

    stubs = {
        "cv2": make_cv2_stub(),
        "PySide6": pyside,
        "PySide6.QtCore": qtcore,
        "multi_tracker.utils.image_processing": image_processing,
        "multi_tracker.utils.geometry": geometry,
        "multi_tracker.data.detection_cache": detection_cache,
        "multi_tracker.utils.batch_optimizer": batch_optimizer,
        "multi_tracker.utils.frame_prefetcher": frame_prefetcher,
        "multi_tracker.core": core_pkg,
        "multi_tracker.core.filters": core_filters,
        "multi_tracker.core.background": core_background,
        "multi_tracker.core.detectors": core_detectors,
        "multi_tracker.core.assigners": core_assigners,
        "multi_tracker.core.identity": core_identity,
        "multi_tracker.core.filters.kalman": kalman,
        "multi_tracker.core.background.model": background_model,
        "multi_tracker.core.detectors.engine": detectors_engine,
        "multi_tracker.core.assigners.hungarian": assigner,
        "multi_tracker.core.identity.analysis": identity,
    }

    return load_src_module(
        "multi_tracker/core/tracking/worker.py",
        "tracking_worker_under_test",
        stubs=stubs,
    )


class _DummyCap:
    def __init__(self, frames):
        self.frames = list(frames)
        self.idx = 0

    def read(self):
        if self.idx >= len(self.frames):
            return False, None
        frame = self.frames[self.idx]
        self.idx += 1
        return True, frame


def test_cached_detection_iterator_frame_counts() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    rows = list(
        worker._cached_detection_iterator(
            total_frames=4, start_frame=5, end_frame=8, backward=False
        )
    )
    assert len(rows) == 4
    assert [count for _, count in rows] == [1, 2, 3, 4]
    assert all(frame is None for frame, _ in rows)

    rows_b = list(
        worker._cached_detection_iterator(
            total_frames=3, start_frame=10, end_frame=12, backward=True
        )
    )
    assert len(rows_b) == 3
    assert [count for _, count in rows_b] == [1, 2, 3]


def test_forward_frame_iterator_sync_and_prefetch_paths() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    cap_sync = _DummyCap(frames=["f1", "f2", "f3"])
    sync_rows = list(worker._forward_frame_iterator(cap_sync, use_prefetcher=False))
    assert sync_rows == [("f1", 1), ("f2", 2), ("f3", 3)]

    cap_prefetch = _DummyCap(frames=["a", "b"])
    prefetch_rows = list(
        worker._forward_frame_iterator(cap_prefetch, use_prefetcher=True)
    )
    assert prefetch_rows == [("a", 1), ("b", 2)]
    assert worker.frame_prefetcher is None


def test_collapse_obb_axis_theta_chooses_nearest_branch() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    theta_axis = np.deg2rad(12.0)
    reference = np.deg2rad(205.0)
    collapsed = worker._collapse_obb_axis_theta(theta_axis, reference)
    expected = (theta_axis + np.pi) % (2 * np.pi)
    diff = ((collapsed - expected + np.pi) % (2 * np.pi)) - np.pi
    assert abs(float(diff)) < 1e-6


def test_pose_heading_from_keypoints_uses_weighted_centroids() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    keypoints = np.array(
        [
            [9.0, 0.0, 0.9],  # anterior
            [11.0, 0.0, 0.8],  # anterior
            [0.0, 0.0, 0.7],  # posterior
            [0.0, 1.0, 0.1],  # posterior but below threshold
        ],
        dtype=np.float32,
    )
    theta = worker._compute_pose_heading_from_keypoints(
        keypoints=keypoints,
        anterior_indices=[0, 1],
        posterior_indices=[2, 3],
        min_valid_conf=0.2,
    )
    assert theta is not None
    assert abs(float(theta)) < 1e-6

    theta_none = worker._compute_pose_heading_from_keypoints(
        keypoints=keypoints,
        anterior_indices=[3],  # low confidence only
        posterior_indices=[2],
        min_valid_conf=0.2,
    )
    assert theta_none is None


def test_resolve_pose_group_indices_accepts_names_and_indices() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    names = ["head", "thorax", "abdomen"]
    idxs = worker._resolve_pose_group_indices(["head", 2, "HEAD", "missing"], names)
    assert idxs == [0, 2]


def test_backward_orientation_flip_applies_only_to_motion_based_theta() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    worker.backward_mode = True
    base_theta = worker._normalize_theta(np.deg2rad(35.0))

    motion_theta_out = (base_theta + np.pi) % (2 * np.pi)
    pose_theta_out = base_theta

    expected_motion = worker._normalize_theta(np.deg2rad(215.0))
    diff_motion = (
        (float(motion_theta_out) - float(expected_motion) + np.pi) % (2 * np.pi)
    ) - np.pi
    assert abs(float(diff_motion)) < 1e-6

    diff_pose = (
        (float(pose_theta_out) - float(base_theta) + np.pi) % (2 * np.pi)
    ) - np.pi
    assert abs(float(diff_pose)) < 1e-6
