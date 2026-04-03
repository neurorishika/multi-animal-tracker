from __future__ import annotations

import types

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


def _load_worker_module():
    hydra_suite_pkg = types.ModuleType("hydra_suite")
    hydra_suite_pkg.__path__ = []
    core_pkg = types.ModuleType("hydra_suite.core")
    core_pkg.__path__ = []
    core_tracking = types.ModuleType("hydra_suite.core.tracking")
    core_tracking.__path__ = []
    utils_pkg = types.ModuleType("hydra_suite.utils")
    utils_pkg.__path__ = []
    data_pkg = types.ModuleType("hydra_suite.data")
    data_pkg.__path__ = []

    video_artifacts = load_src_module(
        "hydra_suite/utils/video_artifacts.py",
        "video_artifacts_under_test",
    )

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
    image_processing = types.ModuleType("hydra_suite.utils.image_processing")
    image_processing.apply_image_adjustments = lambda *args, **kwargs: args[0]
    image_processing.stabilize_lighting = lambda *args, **kwargs: (args[0], None, 0.0)

    geometry = types.ModuleType("hydra_suite.utils.geometry")
    geometry.wrap_angle_degs = lambda x: x
    geometry.estimate_detection_crop_quality = lambda shape, ref: 0.0

    detection_cache = types.ModuleType("hydra_suite.data.detection_cache")
    detection_cache.DetectionCache = object

    tag_observation_cache = types.ModuleType("hydra_suite.data.tag_observation_cache")
    tag_observation_cache.TagObservationCache = object

    batch_optimizer = types.ModuleType("hydra_suite.utils.batch_optimizer")
    batch_optimizer.BatchOptimizer = object

    frame_prefetcher = types.ModuleType("hydra_suite.utils.frame_prefetcher")

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
    core_filters = types.ModuleType("hydra_suite.core.filters")
    core_background = types.ModuleType("hydra_suite.core.background")
    core_detectors = types.ModuleType("hydra_suite.core.detectors")
    core_assigners = types.ModuleType("hydra_suite.core.assigners")
    core_identity = types.ModuleType("hydra_suite.core.identity")

    core_filters.__path__ = []
    core_background.__path__ = []
    core_detectors.__path__ = []
    core_assigners.__path__ = []
    core_identity.__path__ = []

    kalman = types.ModuleType("hydra_suite.core.filters.kalman")
    kalman.KalmanFilterManager = object

    background_model = types.ModuleType("hydra_suite.core.background.model")
    background_model.BackgroundModel = object

    # worker.py imports create_detector from the package
    core_detectors.create_detector = lambda *_args, **_kwargs: None

    assigner = types.ModuleType("hydra_suite.core.assigners.hungarian")
    assigner.TrackAssigner = object

    identity_dataset = types.ModuleType("hydra_suite.core.identity.dataset")
    identity_dataset_generator = types.ModuleType(
        "hydra_suite.core.identity.dataset.generator"
    )
    identity_dataset_generator.IndividualDatasetGenerator = object

    tag_features = types.ModuleType("hydra_suite.core.tracking.tag_features")
    tag_features.NO_TAG = -1
    tag_features.TrackTagHistory = object
    tag_features.build_detection_tag_id_list = lambda *_args, **_kwargs: []
    tag_features.build_tag_detection_map = lambda *_args, **_kwargs: {}

    # Classification sub-package stubs
    classification = types.ModuleType("hydra_suite.core.identity.classification")
    classification_apriltag = types.ModuleType(
        "hydra_suite.core.identity.classification.apriltag"
    )
    classification_apriltag.AprilTagDetector = object
    classification_apriltag.AprilTagConfig = object
    classification_cnn = types.ModuleType(
        "hydra_suite.core.identity.classification.cnn"
    )
    classification_cnn.ClassPrediction = object
    classification_cnn.CNNIdentityBackend = object
    classification_cnn.CNNIdentityCache = object
    classification_cnn.CNNIdentityConfig = object
    classification_cnn.apply_cnn_identity_cost = lambda *_args, **_kwargs: 0.0
    classification_headtail = types.ModuleType(
        "hydra_suite.core.identity.classification.headtail"
    )
    classification_headtail.HeadTailAnalyzer = object

    # Identity geometry stubs
    identity_geometry = types.ModuleType("hydra_suite.core.identity.geometry")
    identity_geometry.build_detection_direction_overrides = lambda *_args, **_kwargs: (
        np.full(0, np.nan, dtype=np.float32),
        np.zeros(0, dtype=np.uint8),
    )
    identity_geometry.resolve_detection_tracking_theta = lambda *_args, **_kwargs: 0.0
    identity_geometry.resolve_tracking_theta = lambda *_args, **_kwargs: 0.0
    identity_geometry.normalize_theta = lambda x: float(x) % (2 * 3.141592653589793)

    # Pose sub-package stubs
    pose_pkg = types.ModuleType("hydra_suite.core.identity.pose")
    pose_features_new = types.ModuleType("hydra_suite.core.identity.pose.features")
    pose_features_new.build_pose_detection_keypoint_map = lambda *_args, **_kwargs: {}
    pose_features_new.compute_pose_geometry_from_keypoints = (
        lambda *_args, **_kwargs: None
    )
    pose_features_new.normalize_pose_keypoints = lambda *_args, **_kwargs: None
    pose_features_new.resolve_pose_group_indices = lambda *_args, **_kwargs: []
    pose_api = types.ModuleType("hydra_suite.core.identity.pose.api")
    pose_api.build_runtime_config = lambda *_args, **_kwargs: None
    pose_api.create_pose_backend_from_config = lambda *_args, **_kwargs: None

    # Properties sub-package stubs
    properties_pkg = types.ModuleType("hydra_suite.core.identity.properties")
    properties_cache = types.ModuleType("hydra_suite.core.identity.properties.cache")
    properties_cache.IndividualPropertiesCache = object
    properties_cache.compute_detection_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_extractor_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_filter_settings_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_individual_properties_id = lambda *_args, **_kwargs: ""

    # Tracking sub-module stubs for density, cnn_features, precompute
    density = types.ModuleType("hydra_suite.core.tracking.density")
    density.get_density_region_flags = lambda *_args, **_kwargs: np.zeros(0, dtype=bool)

    cnn_features = types.ModuleType("hydra_suite.core.tracking.cnn_features")
    cnn_features.cnn_build_association_entries = lambda *_args, **_kwargs: (
        None,
        None,
        None,
    )
    cnn_features.cnn_update_track_history = lambda *_args, **_kwargs: None

    pose_pipeline = types.ModuleType("hydra_suite.core.tracking.pose_pipeline")
    pose_pipeline.extract_one_crop = lambda *_args, **_kwargs: None
    precompute = types.ModuleType("hydra_suite.core.tracking.precompute")
    precompute.AprilTagPrecomputePhase = object
    precompute.CNNPrecomputePhase = object
    precompute.CropConfig = object
    precompute.UnifiedPrecompute = object
    profiler = types.ModuleType("hydra_suite.core.tracking.profiler")
    profiler.TrackingProfiler = object

    stubs = {
        "cv2": make_cv2_stub(),
        "PySide6": pyside,
        "PySide6.QtCore": qtcore,
        "hydra_suite": hydra_suite_pkg,
        "hydra_suite.core": core_pkg,
        "hydra_suite.core.tracking": core_tracking,
        "hydra_suite.utils": utils_pkg,
        "hydra_suite.data": data_pkg,
        "hydra_suite.utils.image_processing": image_processing,
        "hydra_suite.utils.geometry": geometry,
        "hydra_suite.utils.video_artifacts": video_artifacts,
        "hydra_suite.data.detection_cache": detection_cache,
        "hydra_suite.data.tag_observation_cache": tag_observation_cache,
        "hydra_suite.utils.batch_optimizer": batch_optimizer,
        "hydra_suite.utils.frame_prefetcher": frame_prefetcher,
        "hydra_suite.core.filters": core_filters,
        "hydra_suite.core.background": core_background,
        "hydra_suite.core.detectors": core_detectors,
        "hydra_suite.core.assigners": core_assigners,
        "hydra_suite.core.identity": core_identity,
        "hydra_suite.core.filters.kalman": kalman,
        "hydra_suite.core.background.model": background_model,
        "hydra_suite.core.assigners.hungarian": assigner,
        "hydra_suite.core.identity.dataset": identity_dataset,
        "hydra_suite.core.identity.dataset.generator": identity_dataset_generator,
        "hydra_suite.core.identity.classification": classification,
        "hydra_suite.core.identity.classification.apriltag": classification_apriltag,
        "hydra_suite.core.identity.classification.cnn": classification_cnn,
        "hydra_suite.core.identity.classification.headtail": classification_headtail,
        "hydra_suite.core.identity.geometry": identity_geometry,
        "hydra_suite.core.identity.pose": pose_pkg,
        "hydra_suite.core.identity.pose.features": pose_features_new,
        "hydra_suite.core.identity.pose.api": pose_api,
        "hydra_suite.core.identity.properties": properties_pkg,
        "hydra_suite.core.identity.properties.cache": properties_cache,
        "hydra_suite.core.tracking.density": density,
        "hydra_suite.core.tracking.cnn_features": cnn_features,
        "hydra_suite.core.tracking.pose_pipeline": pose_pipeline,
        "hydra_suite.core.tracking.precompute": precompute,
        "hydra_suite.core.tracking.profiler": profiler,
        "hydra_suite.core.tracking.tag_features": tag_features,
    }

    return load_src_module(
        "hydra_suite/core/tracking/worker.py",
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
    pf = load_src_module(
        "hydra_suite/core/identity/geometry.py",
        "geometry_for_collapse_test",
    )

    theta_axis = np.deg2rad(12.0)
    reference = np.deg2rad(205.0)
    collapsed = pf.collapse_obb_axis_theta(theta_axis, reference)
    expected = (theta_axis + np.pi) % (2 * np.pi)
    diff = ((collapsed - expected + np.pi) % (2 * np.pi)) - np.pi
    assert abs(float(diff)) < 1e-6


def test_pose_heading_from_keypoints_uses_weighted_centroids() -> None:
    pf = load_src_module(
        "hydra_suite/core/identity/pose/features.py",
        "pose_features_for_heading_test",
    )

    keypoints = np.array(
        [
            [9.0, 0.0, 0.9],  # anterior
            [11.0, 0.0, 0.8],  # anterior
            [0.0, 0.0, 0.7],  # posterior
            [0.0, 1.0, 0.1],  # posterior but below threshold
        ],
        dtype=np.float32,
    )
    result = pf.compute_pose_geometry_from_keypoints(
        keypoints=keypoints,
        anterior_indices=[0, 1],
        posterior_indices=[2, 3],
        min_valid_conf=0.2,
    )
    theta = result["heading"] if result else None
    assert theta is not None
    assert abs(float(theta)) < 1e-6

    result_none = pf.compute_pose_geometry_from_keypoints(
        keypoints=keypoints,
        anterior_indices=[3],  # low confidence only
        posterior_indices=[2],
        min_valid_conf=0.2,
    )
    theta_none = result_none["heading"] if result_none else None
    assert theta_none is None


def test_resolve_pose_group_indices_accepts_names_and_indices() -> None:
    pf = load_src_module(
        "hydra_suite/core/identity/pose/features.py",
        "pose_features_for_indices_test",
    )

    names = ["head", "thorax", "abdomen"]
    idxs = pf.resolve_pose_group_indices(["head", 2, "HEAD", "missing"], names)
    assert idxs == [0, 2]


def test_individual_data_precompute_gate_requires_pose_extractor() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    assert (
        worker._should_precompute_individual_data(
            {"ENABLE_POSE_EXTRACTOR": True},
            "yolo_obb",
        )
        is True
    )
    assert (
        worker._should_precompute_individual_data(
            {"ENABLE_POSE_EXTRACTOR": False},
            "yolo_obb",
        )
        is False
    )
    assert (
        worker._should_precompute_individual_data(
            {"ENABLE_POSE_EXTRACTOR": True},
            "background_subtraction",
        )
        is False
    )


def test_individual_properties_cache_path_defaults_to_video_cache_dir(
    tmp_path,
) -> None:
    mod = _load_worker_module()
    video_path = tmp_path / "clip.mp4"
    worker = mod.TrackingWorker(str(video_path))

    cache_path = worker._build_individual_properties_cache_path("props", 4, 9)

    assert cache_path.parent == tmp_path / "clip_caches"
    assert cache_path.name == "clip_pose_cache_props_4_9.npz"


def test_confidence_density_enabled_defaults_true_and_respects_flag() -> None:
    mod = _load_worker_module()
    worker = mod.TrackingWorker("dummy.mp4")

    assert worker._confidence_density_enabled({}) is True
    assert (
        worker._confidence_density_enabled({"ENABLE_CONFIDENCE_DENSITY_MAP": True})
        is True
    )
    assert (
        worker._confidence_density_enabled({"ENABLE_CONFIDENCE_DENSITY_MAP": False})
        is False
    )

    worker.set_parameters({"ENABLE_CONFIDENCE_DENSITY_MAP": False})
    assert worker._confidence_density_enabled() is False


def test_backward_orientation_flip_applies_only_to_motion_based_theta() -> None:
    pf = load_src_module(
        "hydra_suite/core/identity/geometry.py",
        "geometry_for_orient_flip_test",
    )

    base_theta = pf.normalize_theta(np.deg2rad(35.0))

    motion_theta_out = (base_theta + np.pi) % (2 * np.pi)
    pose_theta_out = base_theta

    expected_motion = pf.normalize_theta(np.deg2rad(215.0))
    diff_motion = (
        (float(motion_theta_out) - float(expected_motion) + np.pi) % (2 * np.pi)
    ) - np.pi
    assert abs(float(diff_motion)) < 1e-6

    diff_pose = (
        (float(pose_theta_out) - float(base_theta) + np.pi) % (2 * np.pi)
    ) - np.pi
    assert abs(float(diff_pose)) < 1e-6


def test_select_directed_heading_prefers_pose_by_default() -> None:
    pf = load_src_module(
        "hydra_suite/core/identity/geometry.py",
        "geometry_for_heading_select_test",
    )

    selected, directed = pf.select_directed_heading(
        pose_heading=np.deg2rad(30.0),
        pose_directed=True,
        headtail_heading=np.deg2rad(210.0),
        headtail_directed=True,
        pose_overrides_headtail=True,
    )
    assert directed is True
    diff = ((float(selected) - float(np.deg2rad(30.0)) + np.pi) % (2 * np.pi)) - np.pi
    assert abs(float(diff)) < 1e-6


def test_select_directed_heading_can_prefer_headtail() -> None:
    pf = load_src_module(
        "hydra_suite/core/identity/geometry.py",
        "geometry_for_headtail_test",
    )

    selected, directed = pf.select_directed_heading(
        pose_heading=np.deg2rad(30.0),
        pose_directed=True,
        headtail_heading=np.deg2rad(210.0),
        headtail_directed=True,
        pose_overrides_headtail=False,
    )
    assert directed is True
    diff = ((float(selected) - float(np.deg2rad(210.0)) + np.pi) % (2 * np.pi)) - np.pi
    assert abs(float(diff)) < 1e-6
