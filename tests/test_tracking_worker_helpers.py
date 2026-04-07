from __future__ import annotations

import types

import numpy as np

from tests.helpers.module_loader import load_src_module, make_cv2_stub


def _make_namespace_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _build_base_package_stubs() -> dict[str, types.ModuleType]:
    return {
        "hydra_suite": _make_namespace_package("hydra_suite"),
        "hydra_suite.core": _make_namespace_package("hydra_suite.core"),
        "hydra_suite.core.tracking": _make_namespace_package(
            "hydra_suite.core.tracking"
        ),
        "hydra_suite.utils": _make_namespace_package("hydra_suite.utils"),
        "hydra_suite.data": _make_namespace_package("hydra_suite.data"),
    }


def _build_qt_stubs() -> dict[str, types.ModuleType]:
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
    return {"PySide6": pyside, "PySide6.QtCore": qtcore}


def _build_runtime_stubs(video_artifacts) -> dict[str, types.ModuleType]:
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

    return {
        "hydra_suite.utils.image_processing": image_processing,
        "hydra_suite.utils.geometry": geometry,
        "hydra_suite.utils.video_artifacts": video_artifacts,
        "hydra_suite.data.detection_cache": detection_cache,
        "hydra_suite.data.tag_observation_cache": tag_observation_cache,
        "hydra_suite.utils.batch_optimizer": batch_optimizer,
        "hydra_suite.utils.frame_prefetcher": frame_prefetcher,
    }


def _build_core_dependency_stubs() -> dict[str, types.ModuleType]:
    core_filters = _make_namespace_package("hydra_suite.core.filters")
    core_background = _make_namespace_package("hydra_suite.core.background")
    core_detectors = _make_namespace_package("hydra_suite.core.detectors")
    core_assigners = _make_namespace_package("hydra_suite.core.assigners")
    core_identity = _make_namespace_package("hydra_suite.core.identity")

    kalman = types.ModuleType("hydra_suite.core.filters.kalman")
    kalman.KalmanFilterManager = object

    background_model = types.ModuleType("hydra_suite.core.background.model")
    background_model.BackgroundModel = object

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

    return {
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
        "hydra_suite.core.tracking.tag_features": tag_features,
    }


def _build_identity_and_tracking_stubs() -> dict[str, types.ModuleType]:
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

    identity_geometry = types.ModuleType("hydra_suite.core.identity.geometry")
    identity_geometry.build_detection_direction_overrides = lambda *_args, **_kwargs: (
        np.full(0, np.nan, dtype=np.float32),
        np.zeros(0, dtype=np.uint8),
    )
    identity_geometry.resolve_detection_tracking_theta = lambda *_args, **_kwargs: 0.0
    identity_geometry.resolve_tracking_theta = lambda *_args, **_kwargs: 0.0
    identity_geometry.normalize_theta = lambda x: float(x) % (2 * 3.141592653589793)

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

    properties_pkg = types.ModuleType("hydra_suite.core.identity.properties")
    properties_cache = types.ModuleType("hydra_suite.core.identity.properties.cache")
    properties_cache.IndividualPropertiesCache = object
    properties_cache.compute_detection_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_extractor_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_filter_settings_hash = lambda *_args, **_kwargs: ""
    properties_cache.compute_individual_properties_id = lambda *_args, **_kwargs: ""

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

    return {
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
    }


def _load_worker_module():
    video_artifacts = load_src_module(
        "hydra_suite/utils/video_artifacts.py",
        "video_artifacts_under_test",
    )
    stubs = {"cv2": make_cv2_stub()}
    stubs.update(_build_base_package_stubs())
    stubs.update(_build_qt_stubs())
    stubs.update(_build_runtime_stubs(video_artifacts))
    stubs.update(_build_core_dependency_stubs())
    stubs.update(_build_identity_and_tracking_stubs())
    return load_src_module(
        "hydra_suite/core/tracking/worker.py",
        "tracking_worker_under_test",
        stubs=stubs,
    )


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
