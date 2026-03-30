"""
Core tracking engine running in separate thread for real-time performance.
This is the main orchestrator, functionally identical to the original.
"""

import gc
import logging
import math
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QMutex, QThread, Signal, Slot

from multi_tracker.core.assigners.hungarian import TrackAssigner
from multi_tracker.core.background.model import BackgroundModel
from multi_tracker.core.detectors.engine import create_detector
from multi_tracker.core.filters.kalman import KalmanFilterManager
from multi_tracker.core.identity.analysis import IndividualDatasetGenerator
from multi_tracker.core.tracking.pose_features import (
    build_detection_direction_overrides as _pf_build_direction_overrides,
)
from multi_tracker.core.tracking.pose_features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from multi_tracker.core.tracking.pose_features import (
    compute_pose_geometry_from_keypoints as _pf_compute_geometry,
)
from multi_tracker.core.tracking.pose_features import (
    normalize_pose_keypoints as _pf_normalize_keypoints,
)
from multi_tracker.core.tracking.pose_features import (
    resolve_detection_tracking_theta as _pf_resolve_detection_tracking_theta,
)
from multi_tracker.core.tracking.pose_features import (
    resolve_pose_group_indices as _pf_resolve_indices,
)
from multi_tracker.core.tracking.precompute import (
    AprilTagPrecomputePhase,
    CNNPrecomputePhase,
    CropConfig,
    UnifiedPrecompute,
)
from multi_tracker.core.tracking.profiler import TrackingProfiler
from multi_tracker.core.tracking.tag_features import (
    NO_TAG,
    TrackTagHistory,
    build_detection_tag_id_list,
    build_tag_detection_map,
)
from multi_tracker.data.detection_cache import DetectionCache
from multi_tracker.data.tag_observation_cache import TagObservationCache
from multi_tracker.utils.batch_optimizer import BatchOptimizer
from multi_tracker.utils.frame_prefetcher import FramePrefetcher
from multi_tracker.utils.geometry import wrap_angle_degs
from multi_tracker.utils.image_processing import (
    apply_image_adjustments,
    stabilize_lighting,
)
from multi_tracker.utils.video_artifacts import build_individual_properties_cache_path

logger = logging.getLogger(__name__)


def get_density_region_flags(
    meas,
    regions,
    frame_idx: int,
) -> np.ndarray:
    """Return a boolean mask indicating which detections fall inside a density region.

    Parameters
    ----------
    meas:
        List/array of detection measurements.  Each element must be indexable
        with ``[0]`` (x) and ``[1]`` (y).
    regions:
        List of :class:`DensityRegion` to test against.
    frame_idx:
        Current frame index.

    Returns
    -------
    np.ndarray
        Shape ``(M,)`` bool array — ``True`` for detections inside a flagged
        region.
    """
    M = len(meas)
    flags = np.zeros(M, dtype=bool)
    if not regions:
        return flags

    for j in range(M):
        cx, cy = float(meas[j][0]), float(meas[j][1])
        for region in regions:
            if region.contains(frame_idx, cx, cy):
                flags[j] = True
                break
    return flags


def _cnn_build_association_entries(
    cnn_identity_cache, cnn_track_history, frame_idx, n_det, N
):
    """Return (det_classes, track_identities, frame_preds) for CNN identity assigner fields.

    Returns three values ready for use by the caller.  Either cache argument
    may be None; in that case (None, None, None) is returned and the caller
    should skip the update.  The returned *frame_preds* list can be passed
    directly to _cnn_update_track_history to avoid a second cache.load() call.
    """
    if cnn_identity_cache is None or cnn_track_history is None:
        return None, None, None
    frame_preds = cnn_identity_cache.load(frame_idx)
    det_classes = [None] * n_det
    for pred in frame_preds:
        if pred.det_index < n_det:
            det_classes[pred.det_index] = pred.class_name
    track_identities = cnn_track_history.build_track_identity_list(N)
    return det_classes, track_identities, frame_preds


def _cnn_update_track_history(cnn_track_history, frame_preds, frame_idx, N, rows, cols):
    """Record CNN predictions for the matched (track, detection) pairs.

    *frame_preds* should be the list already loaded by
    _cnn_build_association_entries so that the cache is not read twice per
    frame.
    """
    if cnn_track_history is None or frame_preds is None:
        return
    cnn_track_history.resize(N)
    pred_by_det = {pred.det_index: pred for pred in frame_preds}
    for r, c in zip(rows, cols):
        pred = pred_by_det.get(c)
        if pred is not None and pred.class_name is not None:
            cnn_track_history.record(r, frame_idx, pred.class_name)


class TrackingWorker(QThread):
    """
    Core tracking engine. Orchestrates tracking components to be functionally
    identical to the original monolithic implementation.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal(bool, list, list)
    progress_signal = Signal(int, str)
    stats_signal = Signal(dict)  # Real-time FPS/ETA stats
    warning_signal = Signal(str, str)  # (title, message) for UI warnings
    pose_exported_model_resolved_signal = Signal(str)

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=None,
        preview_mode=False,
        use_cached_detections=False,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.video_output_path = video_output_path
        self.backward_mode = backward_mode
        self.detection_cache_path = detection_cache_path
        self.preview_mode = preview_mode
        self.use_cached_detections = use_cached_detections
        self.video_writer = None
        self.params_mutex = QMutex()
        self.parameters = {}
        self.individual_properties_cache_path = None
        # Stats tracking for FPS/ETA
        self.start_time = None
        self.frame_times = deque(maxlen=30)  # Keep last 30 frames for FPS calculation
        self._stop_requested = False

        # Internal state variables that helper methods depend on
        self.frame_count = 0
        self.trajectories_full = []

        # Confidence density regions (computed after pre-detection phase)
        self._density_regions = []

        # Frame prefetcher for async I/O
        self.frame_prefetcher = None
        self.frame_prefetcher = None

    def set_parameters(self: object, p: dict) -> object:
        """Set full tracking parameter dictionary in a thread-safe way."""
        self.params_mutex.lock()
        self.parameters = p
        self.params_mutex.unlock()

    @Slot(dict)
    def update_parameters(self: object, new_params: dict) -> object:
        """Slot to safely update parameters from the GUI thread."""
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()
        logger.info("Tracking worker parameters updated.")

    def get_current_params(self: object) -> object:
        """Return a shallow copy of current tracking parameters."""
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def _confidence_density_enabled(self, params=None) -> bool:
        """Return whether confidence-density mapping should be active for this run."""
        p = self.get_current_params() if params is None else params
        return bool(p.get("ENABLE_CONFIDENCE_DENSITY_MAP", True))

    def stop(self: object) -> object:
        """Request cooperative stop for current processing loop."""
        self._stop_requested = True

    def _forward_frame_iterator(self, cap, use_prefetcher=False):
        """Iterate through frames in forward direction.

        Args:
            cap: OpenCV VideoCapture object
            use_prefetcher (bool): Use frame prefetching for better I/O performance
        """
        frame_num = 0

        if use_prefetcher:
            # Use async frame prefetching for better performance
            self.frame_prefetcher = FramePrefetcher(cap, buffer_size=2)
            self.frame_prefetcher.start()

            while not self._stop_requested:
                ret, frame = self.frame_prefetcher.read()
                if not ret:
                    break
                frame_num += 1
                yield frame, frame_num

            self.frame_prefetcher.stop()
            self.frame_prefetcher = None
        else:
            # Standard synchronous frame reading
            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_num += 1
                yield frame, frame_num

    def _cached_detection_iterator(
        self, total_frames, start_frame=0, end_frame=None, backward=False
    ):
        """Iterate through frame indices for cached detection mode (no actual frames needed).

        Args:
            total_frames: Total number of frames to process
            start_frame: Starting frame index (0-based, actual video frame)
            end_frame: Ending frame index (0-based, actual video frame)
            backward: If True, iterate in reverse order (for backward tracking)
        """
        if end_frame is None:
            end_frame = start_frame + total_frames - 1

        if backward:
            # Backward mode: iterate from end_frame down to start_frame
            # This matches the cache keys which are actual video frame indices
            for relative_idx in range(total_frames):
                if self._stop_requested:
                    break
                yield None, relative_idx + 1  # Return None for frame, 1-indexed count
        else:
            # Forward cached mode: iterate from start_frame to end_frame
            for relative_idx in range(total_frames):
                if self._stop_requested:
                    break
                yield None, relative_idx + 1  # Return None for frame, 1-indexed count

    def emit_frame(self: object, bgr: object) -> object:
        """Emit current frame to GUI in RGB format."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb)

    def _build_individual_properties_cache_path(
        self, properties_id: str, start_frame: int, end_frame: int
    ) -> Path:
        """Build deterministic path for individual-properties cache artifact."""
        return build_individual_properties_cache_path(
            self.video_path,
            properties_id,
            start_frame,
            end_frame,
            detection_cache_path=self.detection_cache_path,
        )

    def _extract_expanded_obb_crop(
        self, frame: np.ndarray, corners: np.ndarray, padding_fraction: float
    ):
        """Extract axis-aligned crop around expanded OBB polygon.

        Returns:
            tuple[np.ndarray | None, tuple[int, int] | None]:
                (crop, (x_min, y_min)) where offsets map crop-local -> frame-global.
        """
        if frame is None or corners is None:
            return None, None
        if corners.shape[0] < 4:
            return None, None

        frame_h, frame_w = frame.shape[:2]
        centroid = corners.mean(axis=0)
        expanded = corners.copy()
        for i in range(4):
            direction = corners[i] - centroid
            expanded[i] = centroid + direction * (1.0 + padding_fraction)

        expanded[:, 0] = np.clip(expanded[:, 0], 0, frame_w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, frame_h - 1)

        x_min = max(0, int(np.floor(expanded[:, 0].min())))
        x_max = min(frame_w, int(np.ceil(expanded[:, 0].max())) + 1)
        y_min = max(0, int(np.floor(expanded[:, 1].min())))
        y_max = min(frame_h, int(np.ceil(expanded[:, 1].max())) + 1)
        if x_max <= x_min or y_max <= y_min:
            return None, None

        return frame[y_min:y_max, x_min:x_max].copy(), (x_min, y_min)

    @staticmethod
    def _normalize_theta(theta):
        """Normalize radians to [0, 2*pi)."""
        try:
            value = float(theta)
        except Exception:
            value = 0.0
        return value % (2 * math.pi)

    @staticmethod
    def _circular_abs_diff_rad(theta_a, theta_b):
        """Return absolute circular difference in radians in [0, pi]."""
        a = float(theta_a)
        b = float(theta_b)
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(d)

    def _collapse_obb_axis_theta(self, theta_axis, reference_theta):
        """
        Resolve 180-degree OBB ambiguity by picking theta or theta+pi nearest reference.
        """
        theta0 = self._normalize_theta(theta_axis)
        theta1 = self._normalize_theta(theta0 + math.pi)
        if reference_theta is None:
            return theta0
        try:
            ref = float(reference_theta)
            if not np.isfinite(ref):
                return theta0
            ref = self._normalize_theta(ref)
        except Exception:
            return theta0
        d0 = self._circular_abs_diff_rad(theta0, ref)
        d1 = self._circular_abs_diff_rad(theta1, ref)
        return theta0 if d0 <= d1 else theta1

    @staticmethod
    def _estimate_detection_crop_quality(shape, reference_body_size):
        """Estimate crop quality from detection geometry."""
        try:
            area = float(shape[0])
            aspect = float(shape[1])
        except Exception:
            return 0.0
        if not np.isfinite(area) or area <= 0:
            return 0.0
        aspect = max(1e-3, float(abs(aspect)))
        minor = math.sqrt(max(1e-6, (4.0 * area) / (math.pi * max(aspect, 1e-3))))
        ref = max(1.0, float(reference_body_size) * 0.75)
        return float(np.clip(minor / ref, 0.0, 1.0))

    def _resolve_tracking_theta(
        self,
        track_idx,
        measured_theta,
        pose_directed,
        orientation_last,
        fallback_theta=None,
    ):
        """Resolve directed vs axis-aligned orientation consistently for one track."""
        if pose_directed:
            return self._normalize_theta(measured_theta)

        reference_theta = None
        if orientation_last is not None and 0 <= int(track_idx) < len(orientation_last):
            reference_theta = orientation_last[int(track_idx)]

        if reference_theta is None and fallback_theta is not None:
            try:
                candidate = float(fallback_theta)
            except Exception:
                candidate = None
            if candidate is not None and np.isfinite(candidate):
                reference_theta = candidate

        return self._collapse_obb_axis_theta(measured_theta, reference_theta)

    def _build_cnn_identity_cache_path(
        self, label: str, classify_id: str, start_frame: int, end_frame: int
    ):
        """Build an independent, hash-keyed classify cache path."""
        from multi_tracker.utils.video_artifacts import build_classify_cache_path

        return str(
            build_classify_cache_path(
                self.video_path,
                classify_id,
                label,
                start_frame,
                end_frame,
                artifact_base_dir=(
                    Path(self.detection_cache_path).parent
                    if self.detection_cache_path
                    else None
                ),
                create_dir=True,
            )
        )

    def _build_tag_cache_path(self, apriltag_id, start_frame, end_frame):
        """Build an independent, hash-keyed AprilTag cache path."""
        from multi_tracker.utils.video_artifacts import build_apriltag_cache_path

        return str(
            build_apriltag_cache_path(
                self.video_path,
                apriltag_id,
                start_frame,
                end_frame,
                artifact_base_dir=(
                    Path(self.detection_cache_path).parent
                    if self.detection_cache_path
                    else None
                ),
                create_dir=True,
            )
        )

    def _build_precompute_phases(
        self,
        params: dict,
        detection_method: str,
        detection_cache,
        start_frame: int,
        end_frame: int,
    ) -> list:
        """Build the list of enabled precompute phases for a tracking run.

        Returns [] when precompute should be skipped entirely (backward mode,
        preview mode, wrong detection method, or no detection cache).
        """
        if detection_method != "yolo_obb":
            return []
        if self.backward_mode or self.preview_mode:
            return []
        if detection_cache is None:
            return []

        phases = []

        # --- Pose ---
        pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
        if pose_enabled:
            from multi_tracker.core.identity.properties_cache import (
                IndividualPropertiesCache,
                compute_detection_hash,
                compute_extractor_hash,
                compute_filter_settings_hash,
                compute_individual_properties_id,
            )
            from multi_tracker.core.identity.runtime_api import (
                build_runtime_config,
                create_pose_backend_from_config,
            )

            detection_hash = compute_detection_hash(
                params.get("INFERENCE_MODEL_ID", ""),
                self.video_path,
                start_frame,
                end_frame,
                detection_cache_version="2.0",
            )
            filter_hash = compute_filter_settings_hash(params)
            extractor_hash = compute_extractor_hash(params)
            properties_id = compute_individual_properties_id(
                detection_hash, filter_hash, extractor_hash
            )
            pose_cache_path = self._build_individual_properties_cache_path(
                properties_id, start_frame, end_frame
            )
            self.individual_properties_cache_path = str(pose_cache_path)
            params["INDIVIDUAL_PROPERTIES_ID"] = properties_id
            params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(pose_cache_path)

            # Check cache hit
            pose_cache_hit = False
            if pose_cache_path.exists():
                existing = IndividualPropertiesCache(str(pose_cache_path), mode="r")
                try:
                    pose_cache_hit = existing.is_compatible()
                finally:
                    existing.close()

            pose_backend = None
            pose_cache_writer = None
            finalize_metadata = {}

            if not pose_cache_hit:
                pose_out_root = str(
                    params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")
                ).strip()
                if not pose_out_root:
                    pose_out_root = str(pose_cache_path.parent)

                pose_config = build_runtime_config(params, out_root=pose_out_root)
                pose_backend = create_pose_backend_from_config(pose_config)
                pose_backend.warmup()

                runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "")).lower()
                if runtime_flavor.startswith("onnx") or runtime_flavor.startswith(
                    "tensorrt"
                ):
                    try:
                        resolved = str(
                            getattr(pose_backend, "exported_model_path", "")
                            or getattr(pose_backend, "model_path", "")
                        ).strip()
                    except Exception:
                        resolved = ""
                    if resolved:
                        params["POSE_EXPORTED_MODEL_PATH"] = resolved
                        self.pose_exported_model_resolved_signal.emit(resolved)

                pose_cache_writer = IndividualPropertiesCache(
                    str(pose_cache_path), mode="w"
                )
                keypoint_names = list(
                    getattr(pose_backend, "output_keypoint_names", []) or []
                )
                finalize_metadata = {
                    "individual_properties_id": properties_id,
                    "detection_hash": detection_hash,
                    "filter_settings_hash": filter_hash,
                    "extractor_hash": extractor_hash,
                    "pose_keypoint_names": keypoint_names,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "video_path": str(Path(self.video_path).expanduser().resolve()),
                }

            _POSE_CROSS_FRAME_BATCH = int(params.get("POSE_PRECOMPUTE_BATCH_SIZE", 64))
            _bg_raw = params.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])
            _pose_bg_color = (
                tuple(int(c) for c in _bg_raw)
                if isinstance(_bg_raw, (list, tuple)) and len(_bg_raw) == 3
                else (0, 0, 0)
            )
            _suppress_foreign_obb = bool(
                params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)
            )
            _crop_workers = int(params.get("POSE_PIPELINE_CROP_WORKERS", 4))
            _pre_resize = int(params.get("POSE_PIPELINE_PRE_RESIZE", 0))
            if _pre_resize <= 0 and pose_backend is not None:
                _pre_resize = int(getattr(pose_backend, "preferred_input_size", 0) or 0)

            from multi_tracker.core.tracking.pose_pipeline import PosePipeline

            pipeline = PosePipeline(
                pose_backend,
                pose_cache_writer,
                cross_frame_batch=_POSE_CROSS_FRAME_BATCH,
                crop_workers=_crop_workers,
                pre_resize_target=_pre_resize,
                bg_color=_pose_bg_color,
                suppress_foreign_obb=_suppress_foreign_obb,
                padding_fraction=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
                cache_hit=pose_cache_hit,
                cache_path=str(pose_cache_path),
                finalize_metadata=finalize_metadata,
            )
            phases.append(pipeline)

        # --- AprilTag ---
        if bool(params.get("USE_APRILTAGS", False)):
            from multi_tracker.core.identity.apriltag_detector import AprilTagConfig
            from multi_tracker.core.identity.properties_cache import (
                compute_apriltag_cache_id,
            )

            cfg = AprilTagConfig.from_params(params)
            apriltag_id = compute_apriltag_cache_id(
                params,
                inference_model_id=str(params.get("INFERENCE_MODEL_ID", "")),
            )
            tag_cache_path = self._build_tag_cache_path(
                apriltag_id, start_frame, end_frame
            )
            if tag_cache_path is not None:
                try:
                    phase = AprilTagPrecomputePhase(
                        detector_config=cfg,
                        cache_path=tag_cache_path,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        video_path=str(Path(self.video_path).expanduser().resolve()),
                    )
                    phases.append(phase)
                except ImportError as exc:
                    logger.warning("AprilTag precompute skipped: %s", exc)
                    self.warning_signal.emit("AprilTag Unavailable", str(exc))

        # --- CNN Identity (multi-phase) ---
        for cnn_cfg_dict in params.get("CNN_CLASSIFIERS", []):
            label = str(cnn_cfg_dict.get("label", "cnn_identity"))
            model_path = str(cnn_cfg_dict.get("model_path", ""))
            if not model_path or not os.path.exists(model_path):
                logger.warning(
                    "CNN identity precompute skipped (%s): model not found: %s",
                    label,
                    model_path,
                )
                continue
            from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig
            from multi_tracker.core.identity.properties_cache import (
                compute_classify_cache_id,
            )

            cnn_cfg = CNNIdentityConfig(
                model_path=model_path,
                confidence=float(cnn_cfg_dict.get("confidence", 0.5)),
                batch_size=int(cnn_cfg_dict.get("batch_size", 64)),
            )
            classify_id = compute_classify_cache_id(
                model_path=model_path,
                compute_runtime=str(params.get("COMPUTE_RUNTIME", "cpu")),
                inference_model_id=str(params.get("INFERENCE_MODEL_ID", "")),
            )
            cnn_cache_path = self._build_cnn_identity_cache_path(
                label, classify_id, start_frame, end_frame
            )
            if cnn_cache_path:
                phase = CNNPrecomputePhase(
                    config=cnn_cfg,
                    model_path=model_path,
                    cache_path=cnn_cache_path,
                    compute_runtime=str(params.get("COMPUTE_RUNTIME", "cpu")),
                    name=label,
                )
                phases.append(phase)

        return phases

    def _run_batched_detection_phase(
        self,
        cap,
        detection_cache,
        detector,
        params,
        start_frame,
        end_frame,
        profiler=None,
    ):
        """
        Phase 1: Run batched YOLO detection on specified frame range and cache results.

        Args:
            cap: OpenCV VideoCapture object
            detection_cache: DetectionCache instance for writing
            detector: YOLOOBBDetector instance
            params: Configuration parameters
            start_frame: Starting frame index (0-based)
            end_frame: Ending frame index (0-based)

        Returns:
            int: Total frames processed
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Batched YOLO Detection")
        logger.info("=" * 80)

        # Get batch size using advanced config
        # Include TensorRT settings from top-level params
        advanced_config = params.get("ADVANCED_CONFIG", {}).copy()
        advanced_config["enable_tensorrt"] = params.get("ENABLE_TENSORRT", False)
        advanced_config["tensorrt_max_batch_size"] = params.get(
            "TENSORRT_MAX_BATCH_SIZE", 16
        )
        batch_optimizer = BatchOptimizer(advanced_config)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = end_frame - start_frame + 1  # Process only the specified range

        logger.info(
            f"Processing frame range: {start_frame} to {end_frame} ({total_frames} frames)"
        )

        # Seek to start frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Account for resize factor in batch size estimation
        resize_factor = params.get("RESIZE_FACTOR", 1.0)
        effective_width = int(frame_width * resize_factor)
        effective_height = int(frame_height * resize_factor)

        # Estimate optimal batch size using effective (resized) dimensions
        model_name = params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt")
        batch_size = batch_optimizer.estimate_batch_size(
            effective_width, effective_height, model_name
        )

        logger.info(f"Video: {frame_width}x{frame_height}, {total_frames} frames")
        if resize_factor < 1.0:
            logger.info(
                f"Resize factor: {resize_factor} → Effective: {effective_width}x{effective_height}"
            )
        logger.info(f"Batch size: {batch_size}")

        # Initialize timing stats for detection phase
        detection_start_time = time.time()
        batch_times = deque(maxlen=30)  # Track last 30 batch times for FPS calculation

        # Process video in batches
        frame_idx = 0
        batch_count = 0
        total_batches = (total_frames + batch_size - 1) // batch_size

        # Note: resize_factor already retrieved above

        while not self._stop_requested:
            batch_start_time = time.time()

            # Read a batch of frames
            batch_frames = []
            batch_start_idx = frame_idx

            if profiler:
                profiler.tick("batched_frame_read")
            for _ in range(batch_size):
                if self._stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if we've exceeded end_frame
                current_frame_index = start_frame + frame_idx
                if current_frame_index > end_frame:
                    break

                # Apply resize if needed (same as single-frame mode)
                if resize_factor < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_factor,
                        fy=resize_factor,
                        interpolation=cv2.INTER_AREA,
                    )

                batch_frames.append(frame)
                frame_idx += 1
            if profiler:
                profiler.tock("batched_frame_read")

            if not batch_frames:
                break  # No more frames

            if self._stop_requested:
                break

            # Run batched detection
            batch_count += 1
            logger.info(
                f"Processing batch {batch_count}/{total_batches} ({len(batch_frames)} frames)"
            )

            # Progress callback for within-batch updates
            def progress_cb(
                current,
                total,
                msg,
                _batch_start=batch_start_idx,
                _batch_num=batch_count,
                _total_batches=total_batches,
            ):
                if self._stop_requested:
                    return
                if total <= 0:
                    return
                # Keep UI responsive without flooding signals.
                if current != total and current % 10 != 0:
                    return
                batch_fraction = float(current) / float(total)
                overall_processed = _batch_start + current
                overall_pct = (
                    int((overall_processed * 100) / total_frames)
                    if total_frames > 0
                    else 0
                )
                self.progress_signal.emit(
                    overall_pct,
                    "Detecting objects: "
                    f"batch {_batch_num}/{_total_batches}, "
                    f"within-batch {int(batch_fraction * 100)}% "
                    f"({current}/{total})",
                )

            batch_results = detector.detect_objects_batched(
                batch_frames,
                batch_start_idx,
                progress_cb,
                return_raw=True,
                profiler=profiler,
            )

            # Cache each frame's detections
            for local_idx, (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                raw_heading_hints,
                raw_directed_mask,
                raw_canonical_affines,
            ) in enumerate(batch_results):
                relative_idx = batch_start_idx + local_idx
                actual_frame_idx = (
                    start_frame + relative_idx
                )  # Convert to actual video frame
                # Calculate DetectionID for each detection using actual frame index
                detection_ids = [
                    actual_frame_idx * 10000 + i for i in range(len(raw_meas))
                ]
                detection_cache.add_frame(
                    actual_frame_idx,  # Use actual frame index for cache key
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    detection_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                    canonical_affines=raw_canonical_affines,
                )

            # Track batch timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            # Calculate stats
            elapsed = time.time() - detection_start_time

            # Calculate FPS based on recent batch times
            if len(batch_times) > 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                frames_per_batch = (
                    batch_size if len(batch_frames) == batch_size else len(batch_frames)
                )
                current_fps = (
                    frames_per_batch / avg_batch_time if avg_batch_time > 0 else 0
                )
            else:
                current_fps = 0

            # Calculate ETA
            if current_fps > 0:
                remaining_frames = total_frames - frame_idx
                eta = remaining_frames / current_fps
            else:
                eta = 0

            # Emit progress and stats
            percentage = (
                int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
            )
            status_text = f"Detecting objects: batch {batch_count}/{total_batches} ({percentage}%)"
            self.progress_signal.emit(percentage, status_text)

            # Emit stats signal for FPS/elapsed/ETA display
            self.stats_signal.emit({"fps": current_fps, "elapsed": elapsed, "eta": eta})

        logger.info(
            f"Detection phase complete: {frame_idx} frames processed in {batch_count} batches"
        )
        return frame_idx

    def run(self: object) -> object:  # noqa: C901
        """Execute tracking pipeline for the configured video and parameters."""
        # === 1. INITIALIZATION (Identical to Original) ===
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()

        # Create profiler early so initialization timing is captured.
        # The profiler is configured with metadata later, once all params are known.
        _profiling_enabled = bool(p.get("ENABLE_PROFILING", False))
        profiler = TrackingProfiler(enabled=_profiling_enabled)
        profiler.phase_start("initialization")

        density_map_enabled = self._confidence_density_enabled(p)
        if not density_map_enabled:
            self._density_regions = []

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return

        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_video_frames <= 0:
            total_video_frames = None

        # Get frame range parameters early (before video writer init)
        start_frame = p.get("START_FRAME", 0)
        end_frame = p.get("END_FRAME", None)
        if end_frame is None:
            end_frame = total_video_frames - 1 if total_video_frames else 0

        # Validate frame range
        if total_video_frames:
            start_frame = max(0, min(start_frame, total_video_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_video_frames - 1))

        # Set total_frames to the range we'll actually process
        total_frames = end_frame - start_frame + 1

        logger.info(f"Video has {total_video_frames} frames total")
        logger.info(
            f"Processing frame range: {start_frame} to {end_frame} ({total_frames} frames)"
        )

        if self.video_output_path:
            fps, resize_f = cap.get(cv2.CAP_PROP_FPS), p.get("RESIZE_FACTOR", 1.0)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_f), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_f
            )
            out_path = self.video_output_path
            if self.backward_mode:
                base, ext = os.path.splitext(out_path)
                out_path = f"{base}_backward{ext}"
            self.video_writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        # Initialize detector using factory function.
        # Preview mode remains compatible with fixed-batch runtimes by using
        # single-frame padding in the detector path.
        detector = create_detector(p)

        # Determine if we should use batched detection
        # Batching is only used for YOLO in full tracking mode (not preview, not backward)
        detection_method = p.get("DETECTION_METHOD", "background_subtraction")
        advanced_config = p.get("ADVANCED_CONFIG", {})
        use_batched_detection = (
            not self.preview_mode  # Not preview mode
            and not self.backward_mode  # Not backward mode (uses cache)
            and detection_method == "yolo_obb"  # Only YOLO benefits from batching
            and advanced_config.get(
                "enable_yolo_batching", True
            )  # Batching enabled in config
            and self.detection_cache_path
            is not None  # Need cache path for two-phase approach
        )

        if use_batched_detection:
            logger.info("Using batched YOLO detection (two-phase approach)")
        elif detection_method == "yolo_obb" and not self.preview_mode:
            logger.info("Using frame-by-frame YOLO detection")

        # Initialize background model only if using background subtraction
        # Skip priming in backward mode since it uses cached detections.
        bg_model = None
        if detection_method == "background_subtraction" and not self.backward_mode:
            bg_model = BackgroundModel(p)
            bg_model.prime_background(cap)

        # Seek to start frame if not at beginning
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Seeking to start frame {start_frame}")

        individual_pipeline_enabled = bool(
            p.get(
                "ENABLE_INDIVIDUAL_PIPELINE", p.get("ENABLE_IDENTITY_ANALYSIS", False)
            )
        )
        individual_image_save_enabled = bool(
            p.get(
                "ENABLE_INDIVIDUAL_IMAGE_SAVE",
                p.get("ENABLE_INDIVIDUAL_DATASET", False),
            )
        )

        # Individual analysis is YOLO-only in this phase. Keep tracking behavior
        # unchanged for background subtraction and skip analysis outputs there.
        if individual_pipeline_enabled and detection_method != "yolo_obb":
            msg = (
                "Individual analysis requires YOLO OBB mode. "
                "Background subtraction mode runs tracking without individual-analysis outputs."
            )
            logger.info(msg)
            self.warning_signal.emit("Individual Analysis Disabled", msg)
            individual_pipeline_enabled = False
            individual_image_save_enabled = False

        # Whether any precompute phase will be needed (pose, AprilTag, CNN identity).
        # Used to gate detection-cache requirements and force two-phase YOLO detection.
        individual_data_precompute_enabled = bool(
            not self.backward_mode
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and (
                bool(p.get("ENABLE_POSE_EXTRACTOR", False))
                or bool(p.get("USE_APRILTAGS", False))
                or bool(p.get("CNN_CLASSIFIERS", []))
            )
        )
        if individual_data_precompute_enabled and not self.detection_cache_path:
            logger.error(
                "Individual precompute requires detection caching, but no detection cache path is configured."
            )
            cap.release()
            self.finished_signal.emit(False, [], [])
            return

        # Individual precompute needs full raw detections before tracking starts.
        # Force two-phase detection in YOLO mode so precompute can run on cached detections.
        if individual_data_precompute_enabled and not use_batched_detection:
            use_batched_detection = True
            logger.info("Enabling batched YOLO prepass for precompute.")

        # Initialize individual dataset generator for image persistence only.
        individual_generator = None
        if (
            individual_pipeline_enabled
            and individual_image_save_enabled
            and not self.backward_mode  # Only generate dataset in forward pass
            and not self.preview_mode  # Never generate in preview
        ):
            output_dir = p.get("INDIVIDUAL_DATASET_OUTPUT_DIR")
            video_name = Path(self.video_path).stem
            dataset_name = p.get("INDIVIDUAL_DATASET_NAME", "individual_dataset")
            if output_dir:
                individual_generator = IndividualDatasetGenerator(
                    p, output_dir, video_name, dataset_name
                )
                logger.info(
                    f"Individual dataset generator enabled for {detection_method}, output: {output_dir}"
                )

        self.kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)
        assigner = TrackAssigner(p, worker=self)

        N = p["MAX_TARGETS"]
        # Start all tracks as "lost" so the free_dets loop bootstraps each slot
        # from the first frame's real detections via initialize_filter.  Starting
        # as "active" with a zero-initialised KF state causes every track to sit
        # at (0, 0) = top-left corner for LOST_THRESHOLD_FRAMES frames before it
        # can be properly placed — producing a warm-up gap that the optimizer and
        # TrackingPreviewWorker do not have, and causing parameter divergence.
        track_states, missed_frames = ["lost"] * N, [0] * N
        self.trajectories_full = [[] for _ in range(N)]
        trajectories_pruned = [[] for _ in range(N)]
        position_deques = [
            deque(maxlen=2) for _ in range(N)
        ]  # entries: (x, y, frame_count)
        orientation_last, last_shape_info = [None] * N, [None] * N
        heading_flip_counters = [0] * N  # Hysteresis for directed heading flips
        track_pose_prototypes = [None] * N
        track_avg_step = [0.0] * N
        tracking_continuity = [0] * N
        trajectory_ids, next_trajectory_id = list(range(N)), N

        detection_initialized, tracking_stabilized = False, False
        detection_counts, tracking_counts = 0, 0

        # Diagnostic: log gate parameters for debugging jumps
        _diag_body = float(
            p.get("REFERENCE_BODY_SIZE", 20.0) * p.get("RESIZE_FACTOR", 1.0)
        )
        _diag_max_dist = float(p.get("MAX_DISTANCE_THRESHOLD", 1000.0))
        _diag_vel_gate = (
            float(p.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * _diag_body
        )
        _diag_young_mult = float(p.get("KALMAN_YOUNG_GATE_MULTIPLIER", 1.0))
        _diag_maturity = int(p.get("KALMAN_MATURITY_AGE", 10))
        _diag_lost_thresh = int(p.get("LOST_THRESHOLD_FRAMES", 5))
        logger.info(
            f"Assignment gates: body={_diag_body:.1f}px "
            f"MAX_DIST={_diag_max_dist:.1f}px "
            f"VEL_GATE={_diag_vel_gate:.1f}px "
            f"young_mult={_diag_young_mult:.1f} "
            f"maturity={_diag_maturity} "
            f"lost_thresh={_diag_lost_thresh} "
            f"density_regions={len(self._density_regions)}"
        )

        start_time, self.frame_count, fps_list = time.time(), 0, []
        local_counts, intensity_history, lighting_state = [0] * N, deque(maxlen=50), {}
        roi_fill_color = None  # Average color outside ROI for visualization overlay

        # Pipeline profiler — store run metadata now that all params are known,
        # and close the initialization phase.
        profiler.set_config(
            detection_method=detection_method,
            n_targets=N,
            resize_factor=float(p.get("RESIZE_FACTOR", 1.0)),
            compute_runtime=str(p.get("COMPUTE_RUNTIME", "cpu")),
            start_frame=start_frame,
            end_frame=end_frame,
            backward_mode=self.backward_mode,
            preview_mode=self.preview_mode,
            batched_detection=use_batched_detection,
            density_map_enabled=density_map_enabled,
            precompute_enabled=individual_data_precompute_enabled,
        )
        profiler.phase_end("initialization")

        # Initialize detection cache
        detection_cache = None
        use_cached_detections = False
        cached_frame_indices = set()
        if self.detection_cache_path:
            # Check if we should load existing cache
            cache_exists = os.path.exists(self.detection_cache_path)
            should_load_cache = self.backward_mode or (
                (self.use_cached_detections or individual_data_precompute_enabled)
                and cache_exists
            )

            if should_load_cache and cache_exists:
                # Load cached detections and validate frame range
                detection_cache = DetectionCache(self.detection_cache_path, mode="r")
                if not detection_cache.is_compatible():
                    logger.warning(
                        "Detection cache format is incompatible; deleting and regenerating."
                    )
                    detection_cache.close()
                    detection_cache = None
                    os.remove(self.detection_cache_path)
                    cache_exists = False
                else:
                    cached_start, cached_end = detection_cache.get_frame_range()

                    # Check if cache fully covers requested frame range
                    if detection_cache.covers_frame_range(start_frame, end_frame):
                        requested_total_frames = end_frame - start_frame + 1
                        cache_total_frames = detection_cache.get_total_frames()
                        # Progress and iteration should always reflect the requested subset,
                        # not the full cached file span.
                        total_frames = requested_total_frames
                        use_cached_detections = True
                        if self.backward_mode:
                            logger.info(
                                f"Backward pass using cached detections for requested range "
                                f"{start_frame}-{end_frame} ({requested_total_frames} frames; cache has {cache_total_frames})"
                            )
                        else:
                            logger.info(
                                f"Reusing cached detections for requested range "
                                f"{start_frame}-{end_frame} ({requested_total_frames} frames; cache has {cache_total_frames})"
                            )
                    else:
                        # Frame range mismatch - invalidate cache
                        missing = detection_cache.get_missing_frames(
                            start_frame, end_frame
                        )
                        if missing:
                            logger.warning(
                                f"Cache missing {len(missing)}+ frame(s) in requested range (sample: {missing[:5]})"
                            )
                        logger.warning(
                            f"Cache frame range mismatch! Cache: {cached_start}-{cached_end}, Requested: {start_frame}-{end_frame}"
                        )
                        logger.warning(
                            "Deleting old cache and regenerating detections..."
                        )
                        detection_cache.close()
                        detection_cache = None
                        os.remove(self.detection_cache_path)
                        cache_exists = False

            if self.backward_mode and not use_cached_detections:
                logger.error(
                    "Backward tracking requires a compatible forward detection cache. "
                    "Please run forward tracking first."
                )
                if detection_cache:
                    detection_cache.close()
                cap.release()
                self.finished_signal.emit(False, [], [])
                return

            # Load density regions for backward pass from the sidecar JSON written
            # by the forward pass (backward mode skips pre-detection, so regions are
            # not computed here — they are loaded from disk instead).
            if density_map_enabled and self.backward_mode and not self._density_regions:
                try:
                    from pathlib import Path as _Path

                    from multi_tracker.afterhours.core.confidence_density import (
                        load_regions as _load_regions,
                    )

                    _regions_path = _Path(self.detection_cache_path).with_name(
                        _Path(self.detection_cache_path).stem
                        + "_confidence_regions.json"
                    )
                    if _regions_path.exists():
                        self._density_regions = _load_regions(_regions_path)
                        logger.info(
                            f"Backward pass: loaded {len(self._density_regions)} "
                            f"density regions from {_regions_path}"
                        )
                    else:
                        logger.info(
                            "Backward pass: no density regions sidecar found "
                            f"({_regions_path}); density-aware assignment disabled."
                        )
                except Exception:
                    logger.exception(
                        "Failed to load density regions for backward pass (non-fatal)"
                    )
                    self._density_regions = []

            # Create new cache for writing if needed
            if not use_cached_detections:
                detection_cache = DetectionCache(
                    self.detection_cache_path,
                    mode="w",
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                logger.info(
                    f"Forward pass caching detections for range {start_frame}-{end_frame}"
                )

        # === RUN BATCHED DETECTION PHASE (if applicable) ===
        # Only run batched detection if we don't already have cached detections
        if use_batched_detection and not use_cached_detections:
            # Phase 1: Batched YOLO detection
            profiler.phase_start("batched_detection")
            frames_processed = self._run_batched_detection_phase(
                cap,
                detection_cache,
                detector,
                p,
                start_frame,
                end_frame,
                profiler=profiler,
            )
            profiler.phase_end("batched_detection")

            # Save detection cache after phase 1
            detection_cache.save()
            logger.info("Detection cache saved after batched phase")

            # Reopen cache in read mode for phase 2
            detection_cache.close()
            detection_cache = DetectionCache(self.detection_cache_path, mode="r")
            total_frames = frames_processed
            use_cached_detections = True  # Phase 2 uses cached detections

            # Reset video capture to start frame for phase 2 (tracking + visualization)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            logger.info(f"Reset video to start frame {start_frame} for phase 2")

            logger.info("=" * 80)
            logger.info("PHASE 2: Tracking and Visualization")
            logger.info("=" * 80)

        # === COMPUTE CONFIDENCE DENSITY MAP ===
        # Runs for BOTH fresh and cached detections (forward pass only).
        # Backward pass loads regions from the sidecar JSON instead.
        if (
            density_map_enabled
            and not self.backward_mode
            and self.detection_cache_path
            and detection_cache is not None
            and use_cached_detections
        ):
            profiler.phase_start("confidence_density")
            from pathlib import Path as _Path

            _regions_path = _Path(self.detection_cache_path).with_name(
                _Path(self.detection_cache_path).stem + "_confidence_regions.json"
            )
            if _regions_path.exists():
                # Regions already computed — just load them.
                try:
                    from multi_tracker.afterhours.core.confidence_density import (
                        load_regions as _load_regions,
                    )

                    self._density_regions = _load_regions(_regions_path)
                    logger.info(
                        f"Loaded {len(self._density_regions)} existing density "
                        f"regions from {_regions_path}"
                    )
                except Exception:
                    logger.exception(
                        "Failed to load existing density regions (non-fatal)"
                    )
                    self._density_regions = []
            else:
                # Compute density map from detection cache.
                try:
                    import cv2 as _cv2

                    from multi_tracker.afterhours.core.confidence_density import (
                        compute_density_map_from_cache,
                        export_diagnostic_video,
                        save_regions,
                    )

                    _frame_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
                    _frame_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))

                    # Build a plain dict {frame_idx: (meas_arr, confs_arr, sizes_arr)}
                    _cache_frames = sorted(detection_cache._cached_frames or [])
                    _cache_dict = {}
                    for _fidx in _cache_frames:
                        (
                            _meas_list,
                            _sizes_list,
                            _shapes_list,
                            _confs_list,
                            _obb_corners,
                            _det_ids,
                            _heading_hints,
                            _directed_mask,
                            _canonical_affines,
                            _canvas_dims,
                            _M_inverse,
                        ) = detection_cache.get_frame(_fidx)
                        if _meas_list:
                            _meas_arr = np.array(_meas_list, dtype=np.float32)
                        else:
                            _meas_arr = np.zeros((0, 3), dtype=np.float32)
                        _confs_arr = np.array(_confs_list, dtype=np.float32)
                        _sizes_arr = np.array(_sizes_list, dtype=np.float32)
                        _cache_dict[_fidx] = (_meas_arr, _confs_arr, _sizes_arr)

                    def _density_progress(pct, msg):
                        logger.info(msg)
                        self.progress_signal.emit(pct, msg)

                    logger.info("Computing confidence density map...")
                    self.progress_signal.emit(0, "Computing confidence density map...")

                    # Compute min_area_px in grid-pixel units from body-size fraction.
                    _density_ds = int(p.get("DENSITY_DOWNSAMPLE_FACTOR", 8))
                    _body_size_px = float(p.get("REFERENCE_BODY_SIZE", 20.0)) * float(
                        p.get("RESIZE_FACTOR", 1.0)
                    )
                    _body_size_grid = _body_size_px / max(1, _density_ds)
                    _density_min_area_px = int(
                        float(p.get("DENSITY_MIN_AREA_BODIES", 0.25))
                        * _body_size_grid**2
                    )

                    _dm, _raw_grids = compute_density_map_from_cache(
                        detection_cache=_cache_dict,
                        frame_h=_frame_h,
                        frame_w=_frame_w,
                        sigma_scale=float(p.get("DENSITY_GAUSSIAN_SIGMA_SCALE", 1.0)),
                        temporal_sigma=float(p.get("DENSITY_TEMPORAL_SIGMA", 2.0)),
                        threshold=float(p.get("DENSITY_BINARIZE_THRESHOLD", 0.3)),
                        downsample_factor=int(p.get("DENSITY_DOWNSAMPLE_FACTOR", 8)),
                        min_frame_duration=int(p.get("DENSITY_MIN_FRAME_DURATION", 3)),
                        min_area_px=_density_min_area_px,
                        progress_callback=_density_progress,
                    )
                    self._density_regions = _dm.regions

                    save_regions(self._density_regions, _regions_path)
                    logger.info(
                        f"Density map: {len(self._density_regions)} regions, "
                        f"saved to {_regions_path}"
                    )

                    # Export diagnostic video at reduced resolution with
                    # sequential frame reading (avoids expensive random seeks
                    # on large videos).  Saved next to the source video so it
                    # is easy to find regardless of where the cache lives.
                    _diag_path = _Path(self.video_path).parent / (
                        _Path(self.video_path).stem + "_confidence_map.mp4"
                    )
                    _fps = cap.get(_cv2.CAP_PROP_FPS) or 25.0

                    # Use a sequential reader: seek to start_frame so the
                    # diagnostic video only covers the selected subset.
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

                    def _diag_reader(_fidx, _cap=cap):
                        # Sequential read — _fidx is expected to increase
                        # monotonically.  Just grab the next frame.
                        _ok, _fr = _cap.read()
                        return _fr if _ok else None

                    logger.info("Exporting confidence density diagnostic video...")
                    self.progress_signal.emit(
                        50, "Exporting confidence density video..."
                    )

                    # Output at reduced resolution for speed.
                    _diag_ds = 4  # 4× downsample for diagnostic video (independent of density grid ds)
                    _out_w = max(1, _frame_w // _diag_ds)
                    _out_h = max(1, _frame_h // _diag_ds)

                    export_diagnostic_video(
                        frame_reader=_diag_reader,
                        n_frames=total_frames,
                        frame_h=_out_h,
                        frame_w=_out_w,
                        density_grids=_raw_grids,
                        regions=self._density_regions,
                        output_path=_diag_path,
                        fps=_fps,
                        output_scale=1.0 / _diag_ds,
                        binary_volume=_dm.binary_volume,
                        progress_callback=_density_progress,
                    )
                    self.progress_signal.emit(100, "Density map complete")
                    logger.info(f"Diagnostic video exported: {_diag_path}")

                    # Reset video capture for subsequent phases.
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, start_frame)

                except Exception:
                    logger.exception(
                        "Confidence density map generation failed (non-fatal)"
                    )
                    self._density_regions = []
                    # Ensure video is reset even on failure.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            profiler.phase_end("confidence_density")

        # === UNIFIED PRECOMPUTE ===
        props_path = None
        tag_observation_cache_path = None
        cnn_identity_cache_path = None

        phases = self._build_precompute_phases(
            p, detection_method, detection_cache, start_frame, end_frame
        )
        if phases:
            _bg_raw = p.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])
            _adv = p.get("ADVANCED_CONFIG", {})
            _ref_ar = float(_adv.get("reference_aspect_ratio", 2.0))
            crop_config = CropConfig(
                padding_fraction=float(p.get("INDIVIDUAL_CROP_PADDING", 0.1)),
                suppress_foreign=bool(p.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)),
                bg_color=(
                    tuple(int(c) for c in _bg_raw)
                    if isinstance(_bg_raw, (list, tuple)) and len(_bg_raw) == 3
                    else (0, 0, 0)
                ),
                reference_aspect_ratio=_ref_ar,
            )
            precompute = UnifiedPrecompute(phases, crop_config)
            profiler.phase_start("precompute")
            try:
                results = precompute.run(
                    cap,
                    detection_cache,
                    detector,
                    start_frame,
                    end_frame,
                    float(p.get("RESIZE_FACTOR", 1.0)),
                    p.get("ROI_MASK", None),
                    progress_cb=lambda pct, msg: self.progress_signal.emit(pct, msg),
                    stop_check=lambda: self._stop_requested,
                    warning_cb=lambda title, msg: self.warning_signal.emit(title, msg),
                    profiler=profiler,
                )
            except Exception as exc:
                profiler.phase_end("precompute")
                logger.exception("Unified precompute failed (fatal phase).")
                self.warning_signal.emit(
                    "Precompute Failed",
                    f"Tracking aborted because precompute failed:\n{exc}",
                )
                if detection_cache:
                    detection_cache.close()
                cap.release()
                if self.video_writer:
                    self.video_writer.release()
                self.finished_signal.emit(False, [], [])
                return

            props_path = results.get("pose")
            tag_observation_cache_path = results.get("apriltag")
            cnn_identity_cache_path = results.get("cnn_identity")
            profiler.phase_end("precompute")

            if props_path:
                logger.info("Individual properties cache: %s", props_path)

            # Reset cap position after precompute consumed all frames.
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Open tag observation cache for reading during tracking loop.
        tag_obs_cache = None
        track_tag_history = None
        if tag_observation_cache_path and os.path.exists(tag_observation_cache_path):
            tag_obs_cache = TagObservationCache(tag_observation_cache_path, mode="r")
            if not tag_obs_cache.is_compatible():
                logger.warning("Tag observation cache incompatible; ignoring.")
                tag_obs_cache.close()
                tag_obs_cache = None
            else:
                track_tag_history = TrackTagHistory(
                    N, window=int(p.get("TAG_HISTORY_WINDOW", 30))
                )
                logger.info(
                    "Tag observation cache loaded for tracking: %s",
                    tag_observation_cache_path,
                )

        # Open CNN identity caches for reading during tracking loop (multi-phase).
        _cnn_phase_states = (
            []
        )  # list of (label, cache, history, match_bonus, mismatch_penalty)
        for cnn_cfg_dict in p.get("CNN_CLASSIFIERS", []):
            label = str(cnn_cfg_dict.get("label", "cnn_identity"))
            model_path = str(cnn_cfg_dict.get("model_path", ""))
            from multi_tracker.core.identity.properties_cache import (
                compute_classify_cache_id,
            )

            classify_id = compute_classify_cache_id(
                model_path=model_path,
                compute_runtime=str(p.get("COMPUTE_RUNTIME", "cpu")),
                inference_model_id=str(p.get("INFERENCE_MODEL_ID", "")),
            )
            _path = self._build_cnn_identity_cache_path(
                label, classify_id, start_frame, end_frame
            )
            if _path and os.path.exists(_path):
                from multi_tracker.core.identity.cnn_identity import (
                    CNNIdentityCache,
                    TrackCNNHistory,
                )

                _cache = CNNIdentityCache(_path)
                _hist = TrackCNNHistory(N, window=int(cnn_cfg_dict.get("window", 10)))
                _cnn_phase_states.append(
                    (
                        label,
                        _cache,
                        _hist,
                        float(cnn_cfg_dict.get("match_bonus", 20.0)),
                        float(cnn_cfg_dict.get("mismatch_penalty", 50.0)),
                    )
                )
                logger.info("CNN identity cache loaded (%s): %s", label, _path)

        # Optional pose-properties reader for directional orientation override.
        pose_props_cache = None
        pose_direction_enabled = False
        pose_direction_applied_count = 0
        pose_direction_fallback_count = 0
        pose_direction_anterior_indices = []
        pose_direction_posterior_indices = []
        pose_ignore_indices = []
        pose_keypoint_names = []
        pose_min_valid_conf = float(p.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        pose_frame_keypoints_map = {}
        pose_frame_keypoints_map_frame = None

        pose_cache_candidate = str(
            self.individual_properties_cache_path
            or p.get("INDIVIDUAL_PROPERTIES_CACHE_PATH", "")
            or ""
        ).strip()
        pose_extractor_enabled = bool(p.get("ENABLE_POSE_EXTRACTOR", False))
        if (
            pose_extractor_enabled
            and detection_method == "yolo_obb"
            and pose_cache_candidate
            and os.path.exists(pose_cache_candidate)
        ):
            from multi_tracker.core.identity.properties_cache import (
                IndividualPropertiesCache,
            )

            pose_props_cache = IndividualPropertiesCache(pose_cache_candidate, mode="r")
            if not pose_props_cache.is_compatible():
                logger.warning(
                    "Pose direction override disabled: incompatible properties cache: %s",
                    pose_cache_candidate,
                )
                pose_props_cache.close()
                pose_props_cache = None
            else:
                names = pose_props_cache.metadata.get("pose_keypoint_names", [])
                if isinstance(names, (list, tuple)):
                    pose_keypoint_names = [str(v) for v in names]
                pose_ignore_indices = _pf_resolve_indices(
                    p.get("POSE_IGNORE_KEYPOINTS", []), pose_keypoint_names
                )
                pose_direction_anterior_indices = _pf_resolve_indices(
                    p.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), pose_keypoint_names
                )
                pose_direction_posterior_indices = _pf_resolve_indices(
                    p.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), pose_keypoint_names
                )
                if (
                    len(pose_direction_anterior_indices) > 0
                    and len(pose_direction_posterior_indices) > 0
                ):
                    pose_direction_enabled = True
                    logger.info(
                        "Pose direction override enabled: anterior=%s, posterior=%s",
                        pose_direction_anterior_indices,
                        pose_direction_posterior_indices,
                    )
                else:
                    logger.info(
                        "Pose direction override disabled: define both anterior/posterior keypoint groups."
                    )

        # === 2. FRAME PROCESSING LOOP ===
        # Determine whether to use frame prefetcher
        # Enable for forward passes where we're not batching detection (to avoid double buffering)
        # Prefetching is most beneficial when frame I/O competes with processing time
        use_prefetcher = (
            not use_batched_detection  # Not in batched detection phase 1
            and not self.backward_mode  # Not backward mode (uses cache iterator)
            and not self.preview_mode  # Not preview (latency-sensitive)
            and p.get("ENABLE_FRAME_PREFETCH", True)  # User hasn't disabled it
        )

        # Choose appropriate frame iterator
        if use_cached_detections:
            # Check if we are in forward mode (either Reuse or Batched Phase 2) or backward mode
            # If forward mode, we might need frames for visualization/video/dataset
            if not self.backward_mode:
                # Phase 2 of batched detection OR Cached Reuse: only read frames if we need visualization OR individual analysis
                # Update condition: Check for NOT visualization_free_mode (since ENABLE_VISUALIZATION isn't used)
                # Also check self.video_output_path (since ENABLE_VIDEO_OUTPUT isn't reliably in params)
                needs_frames = (
                    not p.get("VISUALIZATION_FREE_MODE", False)
                    or (self.video_output_path is not None)
                    or individual_generator
                    is not None  # Need frames for cropping individuals
                )

                if needs_frames:
                    frame_iterator = self._forward_frame_iterator(
                        cap, use_prefetcher=use_prefetcher
                    )
                    skip_visualization = False
                    logger.info(
                        "Forward Cached: Using cached detections with frame reading"
                    )
                else:
                    # No visualization or individual analysis - skip frame reading entirely
                    frame_iterator = self._cached_detection_iterator(
                        total_frames, start_frame, end_frame, backward=False
                    )
                    skip_visualization = True
                    use_prefetcher = False
                    logger.info(
                        "Forward Cached: Skipping frame reading (no visualization/analysis needed, using cached detections)"
                    )
            else:
                # Backward pass: no frames needed, skip visualization
                frame_iterator = self._cached_detection_iterator(
                    total_frames, start_frame, end_frame, backward=True
                )
                skip_visualization = True
                use_prefetcher = False  # No frames to prefetch
                logger.info(
                    "Backward pass: Skipping frame reading and visualization for maximum speed"
                )
        else:
            # Standard frame-by-frame with detection
            frame_iterator = self._forward_frame_iterator(
                cap, use_prefetcher=use_prefetcher
            )
            skip_visualization = False

        if use_prefetcher:
            logger.info("Frame prefetching ENABLED (background I/O buffering)")
        else:
            logger.info("Frame prefetching disabled")

        # Pre-compute ROI contours once (the mask is static for the entire run).
        _roi_contours_cache = None

        profiler.phase_start("tracking_loop")
        for frame, _ in frame_iterator:

            params = self.get_current_params()
            self.frame_count += 1

            # Calculate actual frame index (0-based) accounting for start_frame offset
            # In backward mode, go from end_frame backward to start_frame
            if self.backward_mode:
                actual_frame_index = end_frame - (self.frame_count - 1)
            else:
                actual_frame_index = start_frame + (self.frame_count - 1)

            # Check if we've reached the boundary
            if self.backward_mode:
                if actual_frame_index < start_frame:
                    logger.info(
                        f"Reached start frame {start_frame}, stopping backward tracking"
                    )
                    break
            else:
                if actual_frame_index > end_frame:
                    logger.info(f"Reached end frame {end_frame}, stopping tracking")
                    break

            # --- Preprocessing & Detection ---
            profiler.tick("preprocessing")
            resize_f = params["RESIZE_FACTOR"]

            # Skip preprocessing if no frame (cached detection mode)
            if frame is not None:
                # Keep original frame for individual dataset generation (high resolution).
                # When resize_f >= 1.0 the frame won't be replaced below, so a
                # copy is unnecessary — the same buffer can be re-used read-only.
                if individual_generator:
                    original_frame = frame if resize_f >= 1.0 else frame.copy()
                else:
                    original_frame = None

                if resize_f < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )
            else:
                original_frame = None

            profiler.tock("preprocessing")

            detection_method = params.get("DETECTION_METHOD", "background_subtraction")

            # Prepare ROI masks once for both detection and visualization
            ROI_mask = params.get("ROI_MASK", None)
            ROI_mask_current = None

            if ROI_mask is not None:
                if frame is not None:
                    target_w, target_h = frame.shape[1], frame.shape[0]
                else:
                    base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    target_w = max(1, int(base_w * resize_f))
                    target_h = max(1, int(base_h * resize_f))

                if ROI_mask.shape[1] != target_w or ROI_mask.shape[0] != target_h:
                    ROI_mask_current = cv2.resize(
                        ROI_mask, (target_w, target_h), cv2.INTER_NEAREST
                    )
                else:
                    ROI_mask_current = ROI_mask

                # Calculate fill color on first frame if not yet done
                if frame is not None and roi_fill_color is None:
                    mask_inv = ROI_mask_current == 0
                    outside_pixels = frame[mask_inv]
                    if len(outside_pixels) > 0:
                        roi_fill_color = np.mean(outside_pixels, axis=0).astype(
                            np.uint8
                        )
                    else:
                        roi_fill_color = np.array([0, 0, 0], dtype=np.uint8)

            profiler.tick("detection")

            # Initialize detection-related variables (in case no detection occurs)
            detection_ids = []
            raw_detection_ids = []
            filtered_obb_corners = []
            detection_confidences = []
            pose_directed_mask = np.zeros(0, dtype=np.uint8)
            detection_headtail_heading = np.full(0, np.nan, dtype=np.float32)
            headtail_directed_mask = np.zeros(0, dtype=np.uint8)
            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
                [],
                [],
                [],
                [],
                [],
            )
            raw_heading_hints = []
            raw_directed_mask = []
            yolo_results = None
            fg_mask = None
            bg_u8 = None

            # Get detections either from cache or by detection
            if use_cached_detections:
                # Load cached detections using actual frame index
                # The cache keys are actual video frame indices, so we use actual_frame_index directly
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                    raw_canonical_affines,
                    _raw_canvas_dims,
                    _raw_M_inverse,
                ) = detection_cache.get_frame(actual_frame_index)

                if detection_method == "yolo_obb":
                    (
                        meas,
                        sizes,
                        shapes,
                        detection_confidences,
                        filtered_obb_corners,
                        detection_ids,
                        filtered_heading_hints,
                        filtered_directed_mask,
                    ) = detector.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confidences,
                        raw_obb_corners,
                        roi_mask=ROI_mask_current,
                        detection_ids=raw_detection_ids,
                        heading_hints=raw_heading_hints,
                        directed_mask=raw_directed_mask,
                    )
                    detection_headtail_heading = np.asarray(
                        filtered_heading_hints, dtype=np.float32
                    )
                    headtail_directed_mask = np.asarray(
                        filtered_directed_mask, dtype=np.uint8
                    )
                else:
                    meas = raw_meas
                    sizes = raw_sizes
                    shapes = raw_shapes
                    detection_confidences = raw_confidences
                    filtered_obb_corners = raw_obb_corners
                    detection_ids = raw_detection_ids

            elif detection_method == "background_subtraction" and frame is not None:
                # Background subtraction detection pipeline
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = apply_image_adjustments(
                    gray,
                    params["BRIGHTNESS"],
                    params["CONTRAST"],
                    params["GAMMA"],
                    params.get("ENABLE_GPU_BACKGROUND", False),
                )

                if params.get("ENABLE_LIGHTING_STABILIZATION", True):
                    gray, intensity_history, _ = stabilize_lighting(
                        gray,
                        bg_model.reference_intensity,
                        intensity_history,
                        params.get("LIGHTING_SMOOTH_FACTOR", 0.95),
                        ROI_mask_current,
                        params.get("LIGHTING_MEDIAN_WINDOW", 5),
                        lighting_state,
                        params.get("ENABLE_GPU_BACKGROUND", False),
                    )

                bg_u8 = bg_model.update_and_get_background(
                    gray, ROI_mask_current, tracking_stabilized
                )
                if bg_u8 is None:
                    if frame is not None:
                        self.emit_frame(frame)
                    continue

                fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)

                # Apply ROI mask to foreground mask
                if ROI_mask_current is not None:
                    fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=ROI_mask_current)
                if detection_initialized and params.get(
                    "ENABLE_CONSERVATIVE_SPLIT", True
                ):
                    fg_mask = detector.apply_conservative_split(fg_mask, gray, bg_u8)
                meas, sizes, shapes, yolo_results, detection_confidences = (
                    detector.detect_objects(fg_mask, actual_frame_index)
                )
                # No OBB corners for background subtraction
                filtered_obb_corners = []
                # Calculate DetectionID for each detection using actual frame index
                detection_ids = [
                    actual_frame_index * 10000 + i for i in range(len(meas))
                ]
                raw_meas = meas
                raw_sizes = sizes
                raw_shapes = shapes
                raw_confidences = detection_confidences
                raw_obb_corners = filtered_obb_corners
                raw_detection_ids = detection_ids
                raw_heading_hints = []
                raw_directed_mask = []
                raw_canonical_affines = None

            elif (
                detection_method == "yolo_obb" and frame is not None
            ):  # YOLO OBB detection
                # YOLO uses the original BGR frame directly without masking
                # This preserves natural image context for better confidence estimates
                # Note: no frame.copy() needed — Ultralytics letterboxing already
                # copies/transforms the input internally.

                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    yolo_results,
                    raw_confidences,
                    raw_obb_corners,
                    raw_heading_hints,
                    raw_directed_mask,
                    raw_canonical_affines,
                ) = detector.detect_objects(
                    frame,
                    self.frame_count,
                    return_raw=True,
                    profiler=profiler,
                )

                raw_detection_ids = [
                    actual_frame_index * 10000 + i for i in range(len(raw_meas))
                ]
                (
                    meas,
                    sizes,
                    shapes,
                    detection_confidences,
                    filtered_obb_corners,
                    detection_ids,
                    filtered_heading_hints,
                    filtered_directed_mask,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=ROI_mask_current,
                    detection_ids=raw_detection_ids,
                    heading_hints=raw_heading_hints,
                    directed_mask=raw_directed_mask,
                )
                detection_headtail_heading = np.asarray(
                    filtered_heading_hints, dtype=np.float32
                )
                headtail_directed_mask = np.asarray(
                    filtered_directed_mask, dtype=np.uint8
                )
                if yolo_results is not None:
                    pass

            else:
                # No frame and no cached detections - skip this iteration
                if not use_cached_detections:
                    logger.warning(
                        f"Frame {self.frame_count}: No frame available and no cached detections"
                    )
                    continue

            # Cache detections during forward pass (only when actively detecting, not when loading from cache)
            if detection_cache and not self.backward_mode and not use_cached_detections:
                # Cache raw detections so confidence/IOU/ROI filtering can be tuned without re-running inference.
                detection_cache.add_frame(
                    actual_frame_index,
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners if raw_obb_corners else None,
                    raw_detection_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                    canonical_affines=raw_canonical_affines,
                )
                cached_frame_indices.add(actual_frame_index)

            profiler.tock("detection")

            profiler.tick("features")
            detection_crop_quality = np.zeros(len(meas), dtype=np.float32)
            detection_pose_heading = np.full(len(meas), np.nan, dtype=np.float32)
            detection_pose_keypoints = [None] * len(meas)
            detection_pose_visibility = np.zeros(len(meas), dtype=np.float32)
            detection_directed_heading = np.full(len(meas), np.nan, dtype=np.float32)
            detection_directed_mask = np.zeros(len(meas), dtype=np.uint8)

            if meas and shapes:
                reference_body_size = float(params.get("REFERENCE_BODY_SIZE", 20.0))
                for det_idx in range(min(len(meas), len(shapes))):
                    detection_crop_quality[det_idx] = (
                        self._estimate_detection_crop_quality(
                            shapes[det_idx], reference_body_size
                        )
                    )

            # Optional pose-based geometry features and direction override.
            if pose_direction_enabled and meas and detection_ids:
                if pose_frame_keypoints_map_frame != actual_frame_index:
                    pose_frame_keypoints_map = _pf_build_keypoint_map(
                        pose_props_cache, actual_frame_index
                    )
                    pose_frame_keypoints_map_frame = actual_frame_index

                pose_directed_mask = np.zeros(len(meas), dtype=np.uint8)
                n_det = min(len(meas), len(detection_ids))
                for det_idx in range(n_det):
                    try:
                        det_id = int(detection_ids[det_idx])
                    except Exception:
                        continue
                    keypoints = pose_frame_keypoints_map.get(det_id)
                    pose_features = _pf_compute_geometry(
                        keypoints,
                        pose_direction_anterior_indices,
                        pose_direction_posterior_indices,
                        pose_min_valid_conf,
                        ignore_indices=pose_ignore_indices,
                    )
                    if pose_features is None:
                        continue
                    visibility = float(pose_features.get("visibility", 0.0) or 0.0)
                    detection_pose_visibility[det_idx] = visibility
                    detection_pose_keypoints[det_idx] = _pf_normalize_keypoints(
                        keypoints,
                        pose_min_valid_conf,
                        ignore_indices=pose_ignore_indices,
                    )
                    pose_theta = pose_features.get("heading")
                    if pose_theta is None:
                        continue
                    detection_pose_heading[det_idx] = np.float32(pose_theta)
                    pose_directed_mask[det_idx] = 1

            pose_overrides_headtail = bool(params.get("POSE_OVERRIDES_HEADTAIL", True))
            if len(meas) > 0:
                detection_directed_heading, detection_directed_mask = (
                    _pf_build_direction_overrides(
                        len(meas),
                        detection_pose_heading,
                        pose_directed_mask,
                        detection_headtail_heading,
                        headtail_directed_mask,
                        pose_overrides_headtail=pose_overrides_headtail,
                    )
                )

            if len(meas) >= params.get("MIN_DETECTIONS_TO_START", 1):
                detection_counts += 1
            else:
                detection_counts = 0
            if (
                detection_counts >= max(1, params["MIN_DETECTION_COUNTS"] // 2)
                and not detection_initialized
            ):
                detection_initialized = True
                logger.info(f"Tracking initialized with {len(meas)} detections.")

            # === VISUALIZATION (Skip in cached detection mode) ===
            if not skip_visualization and frame is not None:
                overlay = frame.copy()

                # Apply ROI visualization - draw cyan boundary for all detection methods
                # The actual masking for background subtraction happens earlier in the pipeline
                if ROI_mask_current is not None:
                    # Compute ROI contours once and cache (mask is static).
                    if _roi_contours_cache is None:
                        _roi_contours_cache, _ = cv2.findContours(
                            ROI_mask_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                    if _roi_contours_cache:
                        # Draw cyan boundary (BGR: 255, 255, 0)
                        cv2.drawContours(
                            overlay, _roi_contours_cache, -1, (255, 255, 0), 2
                        )
            else:
                overlay = None

            profiler.tock("features")

            if detection_initialized and meas:
                # --- Assignment ---
                profiler.tick("kf_predict")
                preds = self.kf_manager.get_predictions()
                profiler.tock("kf_predict")

                profiler.tick("cost_matrix")

                # --- AprilTag detection map for this frame ---
                _tag_det_map = build_tag_detection_map(
                    tag_obs_cache, actual_frame_index
                )
                _det_tag_ids = build_detection_tag_id_list(_tag_det_map, len(meas))
                _track_tag_ids = (
                    track_tag_history.build_track_tag_id_list(N)
                    if track_tag_history is not None
                    else [NO_TAG] * N
                )

                # The Assigner now takes the kf_manager directly to access X and S_inv
                association_data = {
                    "detection_confidences": detection_confidences,
                    "detection_crop_quality": detection_crop_quality,
                    "detection_pose_heading": detection_directed_heading,
                    "detection_pose_keypoints": detection_pose_keypoints,
                    "detection_pose_visibility": detection_pose_visibility,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": track_avg_step,
                    "detection_tag_ids": _det_tag_ids,
                    "track_last_tag_ids": _track_tag_ids,
                }

                # CNN identity data for assigner (multi-phase)
                _cnn_phases_assoc = []
                _cnn_frame_preds_all = []
                for label, _cache, _history, _bonus, _penalty in _cnn_phase_states:
                    _det_cls, _trk_ids, _frame_preds = _cnn_build_association_entries(
                        _cache,
                        _history,
                        actual_frame_index,
                        len(meas),
                        N,
                    )
                    _cnn_frame_preds_all.append(_frame_preds)
                    if _det_cls is not None:
                        _cnn_phases_assoc.append(
                            {
                                "label": label,
                                "match_bonus": _bonus,
                                "mismatch_penalty": _penalty,
                                "detection_classes": _det_cls,
                                "track_identities": _trk_ids,
                            }
                        )
                if _cnn_phases_assoc:
                    association_data["cnn_phases"] = _cnn_phases_assoc

                # --- Density-aware pre-gate ---
                # For detections inside a high-density region, apply a tighter
                # distance threshold in the cost matrix.  This blocks long-range
                # matches into crowded zones (the track goes occluded instead)
                # while leaving short-range matches intact.
                _density_flags = None
                if self._density_regions and len(meas) > 0:
                    try:
                        _density_flags = get_density_region_flags(
                            meas,
                            self._density_regions,
                            frame_idx=actual_frame_index,
                        )
                    except Exception:
                        logger.debug(
                            "Density region flag computation failed (non-fatal)",
                            exc_info=True,
                        )

                cost, spatial_candidates = assigner.compute_cost_matrix(
                    N,
                    meas,
                    preds,
                    shapes,
                    self.kf_manager,
                    last_shape_info,
                    meas_ori_directed=(
                        detection_directed_mask
                        if len(detection_directed_mask) == len(meas)
                        else None
                    ),
                    association_data=association_data,
                )

                # Tighter distance gate for density-region detections.
                # Block (track, detection) pairs where the detection is in a
                # density region AND the raw Euclidean distance from the track's
                # predicted position exceeds the tighter threshold.  This
                # prevents long-range jumps into crowded zones without
                # distorting the cost matrix (which can push assignments to
                # wrong detections farther away).
                if _density_flags is not None and np.any(_density_flags):
                    _density_factor = float(
                        params.get("DENSITY_CONSERVATIVE_FACTOR", 0.7)
                    )
                    if _density_factor < 1.0:
                        _base_max_dist = float(
                            assigner.params.get("MAX_DISTANCE_THRESHOLD", 1000.0)
                        )
                        _density_max_dist = _base_max_dist * _density_factor
                        _pred_xy = np.asarray(
                            self.kf_manager.X[:N, :2], dtype=np.float32
                        )
                        _meas_xy = np.array(
                            [meas[j][:2] for j in range(len(meas))],
                            dtype=np.float32,
                        )
                        _raw_dist = np.linalg.norm(
                            _pred_xy[:, None, :] - _meas_xy[None, :, :], axis=2
                        )
                        # Block long-range matches to density-region detections.
                        _flagged_cols = np.where(_density_flags)[0]
                        for _c in _flagged_cols:
                            cost[_raw_dist[:, _c] >= _density_max_dist, _c] = 1e9

                profiler.tock("cost_matrix")
                profiler.tick("hungarian")
                rows, cols, free_dets, next_id, high_cost_tracks = (
                    assigner.assign_tracks(
                        cost,
                        N,
                        len(meas),
                        meas,
                        track_states,
                        tracking_continuity,
                        self.kf_manager,  # <--- Pass the manager, not .filters
                        trajectory_ids,
                        next_trajectory_id,
                        spatial_candidates,
                        association_data=association_data,
                    )
                )
                next_trajectory_id = next_id
                respawned_matches = {r for r in rows if track_states[r] == "lost"}
                profiler.tock("hungarian")

                # --- Diagnostic: log large-distance assignments ---
                if rows and self.frame_count % 10 == 0:
                    _body = float(
                        params.get("REFERENCE_BODY_SIZE", 20.0)
                        * params.get("RESIZE_FACTOR", 1.0)
                    )
                    _max_d = float(
                        assigner.params.get("MAX_DISTANCE_THRESHOLD", 1000.0)
                    )
                    _vel_gate = (
                        float(params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)) * _body
                    )
                    for _r, _c in zip(rows, cols):
                        _det_xy = np.array(meas[_c][:2], dtype=np.float32)
                        _pred_xy_diag = self.kf_manager.X[_r, :2].copy()
                        _raw_d = float(np.linalg.norm(_det_xy - _pred_xy_diag))
                        _last_d = float("nan")
                        if self.trajectories_full[_r]:
                            _lp = self.trajectories_full[_r][-1]
                            _last_d = float(
                                np.linalg.norm(
                                    _det_xy
                                    - np.array([_lp[0], _lp[1]], dtype=np.float32)
                                )
                            )
                        _state = track_states[_r]
                        _cont = tracking_continuity[_r]
                        _is_respawn = _r in respawned_matches
                        if _raw_d > _body * 1.5 or _last_d > _body * 2.0:
                            logger.warning(
                                f"JUMP? frame={actual_frame_index} slot={_r} "
                                f"traj={trajectory_ids[_r]} state={_state} "
                                f"cont={_cont} respawn={_is_respawn} "
                                f"raw_dist={_raw_d:.1f} last_dist={_last_d:.1f} "
                                f"body={_body:.1f} MAX_DIST={_max_d:.1f} "
                                f"VEL_GATE={_vel_gate:.1f}"
                            )

                # Conditionally compute confidence metrics (for performance)
                save_confidence = params.get("SAVE_CONFIDENCE_METRICS", True)
                if save_confidence:
                    # Compute assignment confidence for matched pairs
                    matched_pairs = list(zip(rows, cols))
                    assignment_confidences = assigner.compute_assignment_confidence(
                        cost, matched_pairs
                    )

                    # Get Kalman filter position uncertainties
                    position_uncertainties = (
                        self.kf_manager.get_position_uncertainties()
                    )
                else:
                    assignment_confidences = {}
                    position_uncertainties = []

                # --- State Management (Identical to Original) ---
                profiler.tick("state_update")
                matched = set(rows)
                unmatched = list((set(range(N)) - matched) | set(high_cost_tracks))
                for r in matched:
                    missed_frames[r], track_states[r] = 0, "active"
                for r in unmatched:
                    missed_frames[r] += 1
                    if missed_frames[r] >= params["LOST_THRESHOLD_FRAMES"]:
                        track_states[r], tracking_continuity[r] = "lost", 0
                    elif track_states[r] != "lost":
                        track_states[r] = "occluded"

                # --- Record tag observations for matched tracks ---
                if track_tag_history is not None:
                    track_tag_history.resize(N)
                    # Clear history for respawned tracks first (new trajectory)
                    for r in respawned_matches:
                        track_tag_history.clear_track(r)
                    for r, c in zip(rows, cols):
                        track_tag_history.record(r, actual_frame_index, _det_tag_ids[c])

                # Update CNN track histories after assignment (multi-phase)
                for (label, _cache, _history, _, _), _frame_preds in zip(
                    _cnn_phase_states, _cnn_frame_preds_all
                ):
                    for r in respawned_matches:
                        _history.clear_track(r)
                    _cnn_update_track_history(
                        _history,
                        _frame_preds,
                        actual_frame_index,
                        N,
                        rows,
                        cols,
                    )

                # --- KF Update & State Update ---
                profiler.tock("state_update")
                profiler.tick("kf_update")
                total_cost = 0.0
                for r, c in zip(rows, cols):
                    meas_x = float(meas[c][0])
                    meas_y = float(meas[c][1])
                    measured_theta = float(meas[c][2])
                    directed_heading = bool(
                        c < len(detection_directed_mask)
                        and detection_directed_mask[c] == 1
                    )
                    theta_for_tracking = _pf_resolve_detection_tracking_theta(
                        r,
                        measured_theta,
                        (
                            detection_directed_heading[c]
                            if c < len(detection_directed_heading)
                            else float("nan")
                        ),
                        directed_heading,
                        orientation_last,
                        fallback_theta=preds[r, 2] if r < len(preds) else None,
                    )
                    if directed_heading:
                        pose_direction_applied_count += 1
                    else:
                        pose_direction_fallback_count += 1

                    if r in respawned_matches:
                        # Hard KF reset: Phase 3 assigns a new trajectory ID so
                        # the KF must also start clean — mirrors the free_dets loop.
                        self.trajectories_full[r].clear()
                        trajectories_pruned[r].clear()
                        position_deques[r].clear()
                        track_avg_step[r] = 0.0
                        local_counts[r] = 0
                        orientation_last[r] = theta_for_tracking
                        heading_flip_counters[r] = 0
                        track_pose_prototypes[r] = None
                        self.kf_manager.initialize_filter(
                            r,
                            np.array(
                                [meas_x, meas_y, theta_for_tracking, 0.0, 0.0],
                                dtype=np.float32,
                            ),
                        )

                    corrected_meas = np.asarray(
                        [meas_x, meas_y, theta_for_tracking], dtype=np.float32
                    )
                    # Scale theta measurement noise by heading confidence:
                    # low-confidence headings get inflated R[2,2] so the KF
                    # trusts its own prediction more than a noisy measurement.
                    _orient_conf_for_r = 1.0
                    if directed_heading:
                        if c < len(pose_directed_mask) and pose_directed_mask[c]:
                            _orient_conf_for_r = (
                                float(detection_pose_visibility[c])
                                if c < len(detection_pose_visibility)
                                else 1.0
                            )
                        else:
                            _orient_conf_for_r = (
                                float(detection_confidences[c])
                                if c < len(detection_confidences)
                                else 1.0
                            )
                    _theta_r_scale = 1.0 / max(_orient_conf_for_r, 0.1)
                    self.kf_manager.correct(
                        r, corrected_meas, theta_r_scale=_theta_r_scale
                    )
                    track_x = float(self.kf_manager.X[r, 0])
                    track_y = float(self.kf_manager.X[r, 1])
                    if not (np.isfinite(track_x) and np.isfinite(track_y)):
                        track_x, track_y = meas_x, meas_y

                    tracking_continuity[r] += 1
                    position_deques[r].append((track_x, track_y, self.frame_count))
                    if len(position_deques[r]) == 2:
                        (px1, py1, pf1), (px2, py2, pf2) = position_deques[r]
                        speed = math.hypot(px2 - px1, py2 - py1) / max(1, pf2 - pf1)
                    else:
                        speed = 0
                    # Confidence for the directed-heading flip gate:
                    # pose-directed → pose visibility; head-tail-directed →
                    # detector confidence; undirected → not used (1.0).
                    orient_confidence = 1.0
                    if directed_heading:
                        if c < len(pose_directed_mask) and pose_directed_mask[c]:
                            orient_confidence = (
                                float(detection_pose_visibility[c])
                                if c < len(detection_pose_visibility)
                                else 1.0
                            )
                        else:
                            orient_confidence = (
                                float(detection_confidences[c])
                                if c < len(detection_confidences)
                                else 1.0
                            )
                    orientation_last[r] = self._smooth_orientation(
                        r,
                        theta_for_tracking,
                        speed,
                        params,
                        orientation_last,
                        position_deques,
                        directed_heading=directed_heading,
                        orient_confidence=orient_confidence,
                        heading_flip_counters=heading_flip_counters,
                    )
                    # Feed smoothed heading back into the Kalman state so that
                    # the KF prediction stays consistent with the orientation
                    # actually used for tracking/display.  Without this, the KF
                    # state diverges from orientation_last and subsequent
                    # innovations are computed against a stale reference.
                    if orientation_last[r] is not None and r < len(self.kf_manager.X):
                        self.kf_manager.X[r, 2] = float(orientation_last[r])
                    last_shape_info[r] = shapes[c]
                    feature_alpha = float(
                        np.clip(params.get("TRACK_FEATURE_EMA_ALPHA", 0.85), 0.0, 0.999)
                    )
                    high_conf_thresh = float(
                        np.clip(
                            params.get("ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD", 0.7),
                            0.0,
                            1.0,
                        )
                    )
                    det_conf_for_track = (
                        float(detection_confidences[c])
                        if c < len(detection_confidences)
                        else 0.0
                    )
                    if det_conf_for_track >= high_conf_thresh:
                        prev_avg = float(track_avg_step[r])
                        track_avg_step[r] = (
                            feature_alpha * prev_avg + (1.0 - feature_alpha) * speed
                        )

                    det_pose_proto = (
                        detection_pose_keypoints[c]
                        if c < len(detection_pose_keypoints)
                        else None
                    )
                    if det_pose_proto is not None:
                        det_pose_proto = np.asarray(det_pose_proto, dtype=np.float32)
                        prev_pose_proto = track_pose_prototypes[r]
                        if prev_pose_proto is None or np.shape(
                            prev_pose_proto
                        ) != np.shape(det_pose_proto):
                            track_pose_prototypes[r] = det_pose_proto.copy()
                        else:
                            prev_pose_proto = np.asarray(
                                prev_pose_proto, dtype=np.float32
                            )
                            updated = prev_pose_proto.copy()
                            for kp_idx in range(len(det_pose_proto)):
                                det_valid = np.isfinite(
                                    det_pose_proto[kp_idx, 0]
                                ) and np.isfinite(det_pose_proto[kp_idx, 1])
                                prev_valid = np.isfinite(
                                    updated[kp_idx, 0]
                                ) and np.isfinite(updated[kp_idx, 1])
                                if det_valid and prev_valid:
                                    updated[kp_idx, :2] = (
                                        feature_alpha * updated[kp_idx, :2]
                                        + (1.0 - feature_alpha)
                                        * det_pose_proto[kp_idx, :2]
                                    )
                                    updated[kp_idx, 2] = max(
                                        float(updated[kp_idx, 2]),
                                        float(det_pose_proto[kp_idx, 2]),
                                    )
                                elif det_valid:
                                    updated[kp_idx] = det_pose_proto[kp_idx]
                            track_pose_prototypes[r] = updated

                    theta_out = orientation_last[r]
                    # Backward fallback orientation historically needs a 180-degree correction.
                    if self.backward_mode and not directed_heading:
                        theta_out = (theta_out + np.pi) % (2 * np.pi)

                    # Update trajectory with actual frame index
                    pt = (int(track_x), int(track_y), theta_out, actual_frame_index)
                    self.trajectories_full[r].append(pt)
                    trajectories_pruned[r].append(pt)

                    if self.csv_writer_thread:
                        # Build base data row with actual frame index
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            pt[0],
                            pt[1],
                            pt[2],
                            pt[3],
                            track_states[r],
                        ]

                        # Add confidence values if enabled
                        if save_confidence:
                            det_conf = (
                                detection_confidences[c]
                                if c < len(detection_confidences)
                                else 0.0
                            )
                            assign_conf = assignment_confidences.get(r, 0.0)
                            pos_uncertainty = (
                                position_uncertainties[r]
                                if r < len(position_uncertainties)
                                else 0.0
                            )
                            row_data.extend([det_conf, assign_conf, pos_uncertainty])

                        # Add DetectionID (can be NaN for unmatched)
                        det_id = (
                            detection_ids[c] if c < len(detection_ids) else float("nan")
                        )
                        row_data.append(det_id)

                        # Add TagID (majority-vote tag for this track, or NaN)
                        if track_tag_history is not None:
                            _tag = track_tag_history.majority_tag(r)
                            row_data.append(_tag if _tag != NO_TAG else float("nan"))

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1
                    current_cost = cost[r, c]
                    total_cost += current_cost

                # --- CSV for Unmatched & Final Respawn (Identical to Original) ---
                if self.csv_writer_thread:
                    for r in unmatched:
                        last_pos = (
                            self.trajectories_full[r][-1]
                            if self.trajectories_full[r]
                            else (float("nan"),) * 4
                        )
                        # Build base data row with actual frame index
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            last_pos[0],
                            last_pos[1],
                            last_pos[2],
                            actual_frame_index,
                            track_states[r],
                        ]

                        # Add confidence values if enabled (unmatched = 0)
                        if save_confidence:
                            det_conf = 0.0
                            assign_conf = 0.0
                            pos_uncertainty = (
                                position_uncertainties[r]
                                if r < len(position_uncertainties)
                                else 0.0
                            )
                            row_data.extend([det_conf, assign_conf, pos_uncertainty])

                        # Add DetectionID (NaN for unmatched tracks)
                        row_data.append(float("nan"))

                        # Add TagID (majority-vote tag for this track, or NaN)
                        if track_tag_history is not None:
                            _tag = track_tag_history.majority_tag(r)
                            row_data.append(_tag if _tag != NO_TAG else float("nan"))

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1

                for d_idx in free_dets:
                    for track_idx in range(N):
                        if track_states[track_idx] == "lost":
                            # Diagnostic: log slot reuse distance
                            if self.trajectories_full[track_idx]:
                                _old = self.trajectories_full[track_idx][-1]
                                _new_xy = meas[d_idx][:2]
                                _reuse_d = float(
                                    np.linalg.norm(
                                        np.array([_old[0], _old[1]])
                                        - np.array(_new_xy[:2])
                                    )
                                )
                                _body_d = float(
                                    params.get("REFERENCE_BODY_SIZE", 20.0)
                                    * params.get("RESIZE_FACTOR", 1.0)
                                )
                                if _reuse_d > _body_d * 3.0:
                                    logger.warning(
                                        f"SLOT_REUSE frame={actual_frame_index} "
                                        f"slot={track_idx} old_traj="
                                        f"{trajectory_ids[track_idx]} "
                                        f"new_traj={next_trajectory_id} "
                                        f"reuse_dist={_reuse_d:.1f} "
                                        f"old=({_old[0]:.0f},{_old[1]:.0f}) "
                                        f"new=({_new_xy[0]:.0f},"
                                        f"{_new_xy[1]:.0f})"
                                    )
                            directed_heading = bool(
                                d_idx < len(detection_directed_mask)
                                and detection_directed_mask[d_idx] == 1
                            )
                            theta_measurement = float(meas[d_idx][2])
                            if directed_heading and d_idx < len(
                                detection_directed_heading
                            ):
                                directed_theta = float(
                                    detection_directed_heading[d_idx]
                                )
                                if np.isfinite(directed_theta):
                                    theta_measurement = directed_theta
                            theta_init = self._resolve_tracking_theta(
                                track_idx,
                                theta_measurement,
                                pose_directed=directed_heading,
                                orientation_last=orientation_last,
                                fallback_theta=(
                                    self.kf_manager.X[track_idx, 2]
                                    if track_idx < len(self.kf_manager.X)
                                    else None
                                ),
                            )
                            self.kf_manager.initialize_filter(
                                track_idx,
                                np.array(
                                    [
                                        meas[d_idx][0],
                                        meas[d_idx][1],
                                        theta_init,
                                        0,
                                        0,
                                    ],
                                    np.float32,
                                ),
                            )
                            (
                                track_states[track_idx],
                                missed_frames[track_idx],
                                tracking_continuity[track_idx],
                            ) = ("active", 0, 0)
                            self.trajectories_full[track_idx].clear()
                            trajectories_pruned[track_idx].clear()
                            trajectory_ids[track_idx] = next_trajectory_id
                            orientation_last[track_idx] = theta_init
                            last_shape_info[track_idx] = (
                                shapes[d_idx] if d_idx < len(shapes) else None
                            )
                            track_pose_prototypes[track_idx] = (
                                np.asarray(
                                    detection_pose_keypoints[d_idx], dtype=np.float32
                                ).copy()
                                if (
                                    d_idx < len(detection_pose_keypoints)
                                    and detection_pose_keypoints[d_idx] is not None
                                )
                                else None
                            )
                            track_avg_step[track_idx] = 0.0
                            position_deques[track_idx].clear()
                            position_deques[track_idx].append(
                                (
                                    float(meas[d_idx][0]),
                                    float(meas[d_idx][1]),
                                    self.frame_count,
                                )
                            )
                            local_counts[track_idx] = 0
                            next_trajectory_id += 1
                            break

                avg_cost = total_cost / len(rows) if rows else float("inf")
                if avg_cost < params["MAX_DISTANCE_THRESHOLD"]:
                    tracking_counts += 1
                else:
                    tracking_counts = 0
                if (
                    tracking_counts >= params["MIN_TRACKING_COUNTS"]
                    and not tracking_stabilized
                ):
                    tracking_stabilized = True
                    logger.info(f"Tracking stabilized (avg cost={avg_cost:.2f})")

                profiler.tock("kf_update")

            # --- Individual Dataset Generation (supports YOLO OBB and BG subtraction) ---
            profiler.tick("individual_dataset")
            if individual_generator is not None and meas:
                # Get track and trajectory IDs for matched detections
                # cols contains the detection indices that were matched to tracks (rows)
                matched_track_ids = []
                matched_traj_ids = []

                if (
                    detection_initialized
                    and meas
                    and "cols" in locals()
                    and "rows" in locals()
                ):
                    # Create mapping from detection index to track info
                    det_to_track = {}
                    for r, c in zip(rows, cols):
                        det_to_track[c] = (r, trajectory_ids[r])

                    # Build lists in detection order
                    for det_idx in range(len(meas)):
                        if det_idx in det_to_track:
                            track_id, traj_id = det_to_track[det_idx]
                            matched_track_ids.append(track_id)
                            matched_traj_ids.append(traj_id)
                        else:
                            matched_track_ids.append(-1)
                            matched_traj_ids.append(-1)

                # Export all detections that already passed filtering
                # (confidence/IOU/ROI/size), regardless of assignment state.
                # Track/trajectory IDs are attached when available.
                track_ids_for_dataset = (
                    matched_track_ids if len(matched_track_ids) == len(meas) else None
                )
                traj_ids_for_dataset = (
                    matched_traj_ids if len(matched_traj_ids) == len(meas) else None
                )
                conf_for_dataset = (
                    detection_confidences if detection_confidences else None
                )
                detection_ids_for_dataset = detection_ids if detection_ids else None

                # Heading hints from head-tail model for directed canonicalization.
                _ht_hints = (
                    list(filtered_heading_hints)
                    if (
                        "filtered_heading_hints" in locals()
                        and len(filtered_heading_hints) == len(meas)
                    )
                    else None
                )
                _ht_directed = (
                    list(headtail_directed_mask)
                    if (
                        "headtail_directed_mask" in locals()
                        and len(headtail_directed_mask) == len(meas)
                    )
                    else None
                )

                # Motion-based velocity fallback: derive (vx, vy) per detection from
                # the two most-recent positions stored in each track's position_deque.
                _velocities_for_dataset = None
                if matched_track_ids and "position_deques" in locals():
                    _vel_list = []
                    for _tid in matched_track_ids:
                        if (
                            _tid >= 0
                            and _tid < len(position_deques)
                            and len(position_deques[_tid]) == 2
                        ):
                            (_x1, _y1, _f1), (_x2, _y2, _f2) = position_deques[_tid]
                            _dt = _f2 - _f1
                            if _dt != 0:
                                _vel_list.append(((_x2 - _x1) / _dt, (_y2 - _y1) / _dt))
                            else:
                                _vel_list.append(None)
                        else:
                            _vel_list.append(None)
                    if any(v is not None for v in _vel_list):
                        _velocities_for_dataset = _vel_list

                # Use original-frame coordinates for crop extraction.
                coord_scale_factor = 1.0 / resize_f

                # Filter canonical affines to match filtered detections.
                _canon_for_dataset = None
                if (
                    raw_canonical_affines is not None
                    and raw_detection_ids
                    and detection_ids
                ):
                    _raw_id_map = {}
                    for _ri, _rid in enumerate(raw_detection_ids):
                        _raw_id_map[int(_rid)] = _ri
                    _canon_for_dataset = []
                    for _did in detection_ids:
                        _ri2 = _raw_id_map.get(int(_did))
                        if (
                            _ri2 is not None
                            and _ri2 < len(raw_canonical_affines)
                            and raw_canonical_affines[_ri2] is not None
                        ):
                            _canon_for_dataset.append(raw_canonical_affines[_ri2])
                        else:
                            _canon_for_dataset.append(None)

                if filtered_obb_corners:
                    # YOLO OBB detection - use filtered OBB corners directly
                    individual_generator.process_frame(
                        frame=original_frame,
                        frame_id=actual_frame_index,
                        meas=meas,
                        obb_corners=filtered_obb_corners,
                        ellipse_params=None,
                        confidences=conf_for_dataset,
                        track_ids=track_ids_for_dataset,
                        trajectory_ids=traj_ids_for_dataset,
                        coord_scale_factor=coord_scale_factor,
                        detection_ids=detection_ids_for_dataset,
                        heading_hints=_ht_hints,
                        directed_mask=_ht_directed,
                        velocities=_velocities_for_dataset,
                        canonical_affines=_canon_for_dataset,
                    )
                elif shapes:
                    # Background subtraction - compute ellipse params from filtered shapes
                    ellipse_params = []
                    for shape in shapes:
                        area, aspect_ratio = shape[0], shape[1]
                        if aspect_ratio > 0 and area > 0:
                            ax2 = np.sqrt(4 * area / (np.pi * aspect_ratio))
                            ax1 = aspect_ratio * ax2
                            ellipse_params.append(
                                [ax1, ax2]
                            )  # [major_axis, minor_axis]
                        else:
                            # Fallback to small circle if invalid
                            ellipse_params.append([10.0, 10.0])

                    individual_generator.process_frame(
                        frame=original_frame,
                        frame_id=actual_frame_index,
                        meas=meas,
                        obb_corners=None,
                        ellipse_params=ellipse_params,
                        confidences=conf_for_dataset,
                        track_ids=track_ids_for_dataset,
                        trajectory_ids=traj_ids_for_dataset,
                        coord_scale_factor=coord_scale_factor,
                        detection_ids=detection_ids_for_dataset,
                        heading_hints=_ht_hints,
                        directed_mask=_ht_directed,
                        velocities=_velocities_for_dataset,
                        canonical_affines=_canon_for_dataset,
                    )

            profiler.tock("individual_dataset")

            # Emit progress signal periodically to avoid overwhelming the GUI thread
            # We also check that total_frames is valid
            if total_frames and total_frames > 0:
                # Emit more frequently (every 10 frames) to show better progress feedback
                # Especially important for batched detection where users need ETA
                if self.frame_count % 10 == 0:
                    percentage = int((self.frame_count * 100) / total_frames)

                    # Add mode information to status text
                    if use_cached_detections:
                        if use_batched_detection:
                            status_text = (
                                f"Tracking (batched): Frame {self.frame_count}/{total_frames} "
                                f"(abs {actual_frame_index})"
                            )
                        else:
                            status_text = (
                                f"Tracking (cached): Frame {self.frame_count}/{total_frames} "
                                f"(abs {actual_frame_index})"
                            )
                    else:
                        status_text = (
                            f"Processing: Frame {self.frame_count}/{total_frames} "
                            f"(abs {actual_frame_index})"
                        )

                    self.progress_signal.emit(percentage, status_text)

            # --- Visualization, Output & Loop Maintenance ---
            viz_free_mode = params.get("VISUALIZATION_FREE_MODE", False)

            if not viz_free_mode and overlay is not None:
                profiler.tick("visualization")
                # Incrementally prune old trajectory points instead of
                # rebuilding from trajectories_full every frame.
                _traj_horizon = params["TRAJECTORY_HISTORY_SECONDS"]
                _cutoff = self.frame_count - _traj_horizon
                for _tp_list in trajectories_pruned:
                    while _tp_list and _tp_list[0][3] < _cutoff:
                        _tp_list.pop(0)

                self._draw_overlays(
                    overlay,
                    params,
                    trajectories_pruned,
                    track_states,
                    trajectory_ids,
                    tracking_continuity,
                    fg_mask,
                    bg_u8,
                    yolo_results,
                    filtered_obb_corners,  # Pass OBB corners for visualization
                )
                profiler.tock("visualization")

                profiler.tick("video_write")
                if self.video_writer:
                    self.video_writer.write(overlay)
                profiler.tock("video_write")

                profiler.tick("gui_emit")
                # For YOLO with ROI, draw boundary overlay before emitting
                if (
                    detection_method != "background_subtraction"
                    and ROI_mask_current is not None
                ):
                    # Reuse cached ROI contours (computed once above).
                    if _roi_contours_cache is None:
                        _roi_contours_cache, _ = cv2.findContours(
                            ROI_mask_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                    if _roi_contours_cache:
                        # Draw cyan boundary
                        cv2.drawContours(
                            overlay, _roi_contours_cache, -1, (0, 255, 255), 2
                        )

                self.emit_frame(overlay)
                profiler.tock("gui_emit")

            # === Real-time Stats (Always emit, even in viz-free mode) ===
            current_time = time.time()
            self.frame_times.append(current_time)

            # Calculate FPS from recent frames
            if len(self.frame_times) >= 2:
                time_span = self.frame_times[-1] - self.frame_times[0]
                current_fps = (
                    (len(self.frame_times) - 1) / time_span if time_span > 0 else 0
                )
            else:
                current_fps = 0

            # Calculate elapsed and ETA
            if self.start_time is None:
                self.start_time = start_time
            elapsed = current_time - self.start_time

            if total_frames and self.frame_count > 0:
                frames_remaining = total_frames - self.frame_count
                eta = (
                    (frames_remaining / self.frame_count) * elapsed
                    if self.frame_count > 0
                    else 0
                )
            else:
                eta = 0

            # Emit stats every 10 frames to avoid overwhelming the UI
            if self.frame_count % 10 == 0:
                self.stats_signal.emit(
                    {"fps": current_fps, "elapsed": elapsed, "eta": eta}
                )

            # Finalize profiling for this frame and log periodically
            profiler.end_frame()
            profiler.log_periodic(100)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_list.append(self.frame_count / elapsed)

        profiler.phase_end("tracking_loop")

        # Ensure cache has entries for all frames in the requested range (forward pass)
        if detection_cache and not self.backward_mode and not use_cached_detections:
            for frame_idx in range(start_frame, end_frame + 1):
                if frame_idx not in cached_frame_indices:
                    detection_cache.add_frame(
                        frame_idx,
                        [],
                        [],
                        [],
                        [],
                        None,
                        [],
                        [],
                        [],
                    )

        # === 3. CLEANUP (Identical to Original) ===
        profiler.phase_start("cleanup")
        # Stop frame prefetcher if still running
        if self.frame_prefetcher is not None:
            self.frame_prefetcher.stop()
            self.frame_prefetcher = None

        cap.release()
        if self.video_writer:
            self.video_writer.release()

        if pose_props_cache is not None:
            try:
                pose_props_cache.close()
            except Exception:
                pass
        # Save or close detection cache
        if detection_cache:
            if not self.backward_mode and not use_cached_detections:
                # Forward pass Phase 1 (detection phase): save cache to disk
                # Note: In batched detection, cache is already saved after Phase 1
                detection_cache.save()
                logger.info("Detection cache saved successfully")
            else:
                # Backward pass or Phase 2: just close cache (read-only mode)
                detection_cache.close()

        # Finalize individual dataset if enabled
        if individual_generator is not None:
            dataset_path = individual_generator.finalize()
            if dataset_path:
                logger.info(f"Individual dataset saved to: {dataset_path}")

        if pose_direction_applied_count > 0 or pose_direction_fallback_count > 0:
            logger.info(
                "Directed heading summary: applied=%d, fallback=%d",
                int(pose_direction_applied_count),
                int(pose_direction_fallback_count),
            )

        # --- Profiling: final summary and JSON export ---
        profiler.phase_end("cleanup")
        profiler.log_final_summary()
        # Export JSON next to the video output or detection cache, whichever is available.
        # Use a direction suffix so forward and backward profiles are kept separate.
        _dir_tag = "backward" if self.backward_mode else "forward"
        profile_export_path = None
        if self.video_output_path:
            _pbase = Path(self.video_output_path).with_suffix("")
            profile_export_path = Path(f"{_pbase}_{_dir_tag}.profile.json")
        elif self.detection_cache_path:
            profile_export_path = (
                Path(self.detection_cache_path).parent
                / f"tracking_profile_{_dir_tag}.json"
            )
        elif self.video_path:
            profile_export_path = (
                Path(self.video_path).parent / f"tracking_profile_{_dir_tag}.json"
            )
        if profile_export_path is not None:
            profiler.export_summary(profile_export_path)

        logger.info("Tracking worker finished. Emitting raw trajectory data.")

        self.finished_signal.emit(
            not self._stop_requested, fps_list, self.trajectories_full
        )

    def _smooth_orientation(
        self,
        r,
        theta,
        speed,
        p,
        orientation_last,
        position_deques,
        directed_heading=False,
        orient_confidence=1.0,
        heading_flip_counters=None,
    ):
        final_theta, old = theta, orientation_last[r]

        if directed_heading and p.get("DIRECTED_ORIENT_SMOOTHING", True):
            # Directed headings (pose/head-tail) are noisy estimates: trust the
            # axis but gate 180-degree flips via temporal continuity and
            # per-detection confidence.  Small changes (<=90°) are accepted
            # as-is — unlike undirected OBB headings, there is no axis jitter
            # to rate-limit when stopped.
            if old is None:
                if heading_flip_counters is not None:
                    heading_flip_counters[r] = 0
                return theta
            flip_conf_thresh = float(p.get("DIRECTED_ORIENT_FLIP_CONFIDENCE", 0.7))
            flip_persistence = int(p.get("DIRECTED_ORIENT_FLIP_PERSISTENCE", 3))
            old_deg = math.degrees(old)
            new_deg = math.degrees(theta)
            delta = wrap_angle_degs(new_deg - old_deg)
            if abs(delta) > 90:
                # Likely head-tail swap: check motion evidence + confidence.
                single_frame_accept = False
                if speed >= p["VELOCITY_THRESHOLD"] and len(position_deques[r]) == 2:
                    (x1, y1, _), (x2, y2, _) = position_deques[r]
                    ang_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    motion_favors_new = abs(wrap_angle_degs(new_deg - ang_deg)) < abs(
                        wrap_angle_degs(old_deg - ang_deg)
                    )
                    # Accept only when motion corroborates the flip AND
                    # confidence is adequate.  When motion contradicts (animal
                    # still going the same way), never accept regardless of
                    # confidence — the flip is almost certainly spurious.
                    single_frame_accept = (
                        motion_favors_new and orient_confidence >= flip_conf_thresh
                    )
                else:
                    # Stopped or no motion history: confidence alone decides.
                    single_frame_accept = orient_confidence >= flip_conf_thresh

                # Temporal hysteresis: require flip_persistence consecutive
                # frames requesting a flip before actually accepting it.
                if single_frame_accept and heading_flip_counters is not None:
                    heading_flip_counters[r] += 1
                    if heading_flip_counters[r] >= flip_persistence:
                        heading_flip_counters[r] = 0
                        # Accept the flip
                    else:
                        # Not yet enough evidence — reject this frame's flip
                        new_deg = (new_deg + 180.0) % 360.0
                elif not single_frame_accept:
                    if heading_flip_counters is not None:
                        heading_flip_counters[r] = 0
                    new_deg = (new_deg + 180.0) % 360.0
                else:
                    # No hysteresis array — fall back to original behavior
                    if not single_frame_accept:
                        new_deg = (new_deg + 180.0) % 360.0
            else:
                # No flip detected — reset the counter
                if heading_flip_counters is not None:
                    heading_flip_counters[r] = 0
            return math.radians(new_deg % 360.0)

        # --- Original undirected smoothing (axis-only, no direction signal) ---
        if speed < p["VELOCITY_THRESHOLD"] and old is not None:
            old_deg, new_deg = math.degrees(old), math.degrees(theta)
            delta = wrap_angle_degs(new_deg - old_deg)
            if abs(delta) > 90:
                new_deg = (new_deg + 180) % 360
            elif abs(delta) > p["MAX_ORIENT_DELTA_STOPPED"]:
                new_deg = old_deg + math.copysign(p["MAX_ORIENT_DELTA_STOPPED"], delta)
            final_theta = math.radians(new_deg)
        elif speed >= p["VELOCITY_THRESHOLD"] and p["INSTANT_FLIP_ORIENTATION"]:
            (x1, y1, _), (x2, y2, _) = position_deques[r]
            ang = math.atan2(y2 - y1, x2 - x1)
            diff = (ang - theta + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > math.pi / 2:
                final_theta = (theta + math.pi) % (2 * math.pi)
        return final_theta

    def _draw_uncertainty_ellipses(self, overlay, params, track_states):
        """Draw Kalman filter uncertainty ellipses for debugging."""
        if not hasattr(self, "kf_manager"):
            return

        # Get covariance matrices for all tracks
        P = self.kf_manager.P  # Shape: (N, 5, 5)
        X = self.kf_manager.X  # Shape: (N, 5)
        colors = params.get("TRAJECTORY_COLORS", [(255, 0, 0)] * len(X))

        for i in range(len(X)):
            # Skip lost tracks - they are not being actively predicted
            if track_states[i] == "lost":
                continue

            # Extract position (x, y)
            x, y = X[i, 0], X[i, 1]

            # Extract 2x2 position covariance
            P_pos = P[i, :2, :2]

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(P_pos)

            # Sort by eigenvalue magnitude
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # Convert to ellipse parameters (95% confidence ~= 2.45 sigma)
            # Using chi-square distribution for 2D: 95% confidence is sqrt(5.991)
            scale = np.sqrt(5.991)
            width = 2 * scale * np.sqrt(max(0, eigenvalues[0]))
            height = 2 * scale * np.sqrt(max(0, eigenvalues[1]))

            # Calculate rotation angle
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            # Draw ellipse
            center = (int(x), int(y))
            axes = (int(width), int(height))

            # Use track color (BGR format)
            color = tuple(int(c) for c in colors[i])

            # Draw with thicker line and more opacity for visibility
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, 2)

    def _draw_overlays(
        self,
        overlay,
        p,
        trajectories,
        track_states,
        ids,
        continuity,
        fg,
        bg,
        yolo_results=None,  # YOLO results object (direct detection)
        obb_corners=None,  # OBB corners list (cached detections)
    ):
        # Draw YOLO OBB boxes if enabled and available
        if p.get("SHOW_YOLO_OBB", False):
            # First try to use filtered OBB corners (works with cached detections)
            if obb_corners is not None and len(obb_corners) > 0:
                for corners in obb_corners:
                    if corners is not None:
                        # corners is already a numpy array of shape (4, 2)
                        corners_int = corners.astype(np.int32)
                        cv2.polylines(
                            overlay,
                            [corners_int],
                            isClosed=True,
                            color=(0, 255, 255),  # Yellow
                            thickness=2,
                        )
            # Fall back to yolo_results object (direct detection mode)
            elif yolo_results is not None:
                if (
                    hasattr(yolo_results, "obb")
                    and yolo_results.obb is not None
                    and len(yolo_results.obb) > 0
                ):
                    obb_data = yolo_results.obb
                    for i in range(len(obb_data)):
                        # Get the 4 corner points of the OBB
                        corners = obb_data.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                        # Draw the OBB as a polygon
                        cv2.polylines(
                            overlay,
                            [corners],
                            isClosed=True,
                            color=(0, 255, 255),
                            thickness=2,
                        )

                        # Optionally draw confidence score
                        if hasattr(obb_data, "conf"):
                            conf = obb_data.conf[i].cpu().item()
                            cx = int(corners[:, 0].mean())
                            cy = int(corners[:, 1].mean())
                            cv2.putText(
                                overlay,
                                f"{conf:.2f}",
                                (cx - 15, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 255, 255),
                                1,
                            )

        # Draw Kalman uncertainty ellipses if enabled (for debugging)
        if p.get("SHOW_KALMAN_UNCERTAINTY", False):
            self._draw_uncertainty_ellipses(overlay, p, track_states)

        if any(
            p.get(k)
            for k in [
                "SHOW_CIRCLES",
                "SHOW_ORIENTATION",
                "SHOW_TRAJECTORIES",
                "SHOW_LABELS",
                "SHOW_STATE",
            ]
        ):
            for i, tr in enumerate(trajectories):
                if not tr or track_states[i] == "lost":
                    continue
                x, y, th, _ = tr[-1]
                pt = (int(x), int(y))
                if math.isnan(x):
                    continue
                col = tuple(
                    int(c)
                    for c in p["TRAJECTORY_COLORS"][i % len(p["TRAJECTORY_COLORS"])]
                )
                if p.get("SHOW_CIRCLES"):
                    cv2.circle(overlay, pt, 8, col, -1)
                if p.get("SHOW_ORIENTATION"):
                    ex, ey = int(x + 20 * math.cos(th)), int(y + 20 * math.sin(th))
                    cv2.line(overlay, pt, (ex, ey), col, 2)
                if p.get("SHOW_TRAJECTORIES"):
                    pts = np.array(
                        [(pt[0], pt[1]) for pt in tr if not math.isnan(pt[0])],
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))
                    if len(pts) > 1:
                        cv2.polylines(
                            overlay, [pts], isClosed=False, color=col, thickness=2
                        )
                if p.get("SHOW_LABELS") or p.get("SHOW_STATE"):
                    label = (
                        f"T{ids[i]} C:{continuity[i]}" if p.get("SHOW_LABELS") else ""
                    )
                    state = f" [{track_states[i]}]" if p.get("SHOW_STATE") else ""
                    cv2.putText(
                        overlay,
                        f"{label}{state}",
                        (pt[0] + 15, pt[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        col,
                        2,
                    )
        if p.get("SHOW_FG") and fg is not None:
            small_fg = cv2.resize(fg, (0, 0), fx=0.3, fy=0.3)
            overlay[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
                small_fg, cv2.COLOR_GRAY2BGR
            )
        if p.get("SHOW_BG") and bg is not None:
            bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            small_bg = cv2.resize(bg_bgr, (0, 0), fx=0.3, fy=0.3)
            overlay[0 : small_bg.shape[0], -small_bg.shape[1] :] = small_bg
