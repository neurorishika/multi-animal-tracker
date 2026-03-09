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
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from multi_tracker.core.tracking.pose_features import (
    compute_pose_geometry_from_keypoints as _pf_compute_geometry,
)
from multi_tracker.core.tracking.pose_features import (
    normalize_pose_keypoints as _pf_normalize_keypoints,
)
from multi_tracker.core.tracking.pose_features import (
    resolve_pose_group_indices as _pf_resolve_indices,
)
from multi_tracker.data.detection_cache import DetectionCache
from multi_tracker.utils.batch_optimizer import BatchOptimizer
from multi_tracker.utils.frame_prefetcher import FramePrefetcher
from multi_tracker.utils.geometry import wrap_angle_degs
from multi_tracker.utils.image_processing import (
    apply_image_adjustments,
    stabilize_lighting,
)

logger = logging.getLogger(__name__)


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
        self.individual_properties_id = ""

        # Stats tracking for FPS/ETA
        self.start_time = None
        self.frame_times = deque(maxlen=30)  # Keep last 30 frames for FPS calculation
        self._stop_requested = False

        # Internal state variables that helper methods depend on
        self.frame_count = 0
        self.trajectories_full = []

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
        if self.detection_cache_path:
            base_dir = Path(self.detection_cache_path).parent
        else:
            base_dir = Path(self.video_path).parent
        video_stem = Path(self.video_path).stem
        fname = (
            f"{video_stem}_individual_properties_{properties_id}_"
            f"{int(start_frame)}_{int(end_frame)}.npz"
        )
        return base_dir / fname

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
    def _parse_pose_group_tokens(raw_spec):
        """Parse keypoint group spec from list/tuple/string into tokens."""
        if raw_spec is None:
            return []
        if isinstance(raw_spec, str):
            raw_tokens = raw_spec.split(",")
        elif isinstance(raw_spec, (list, tuple)):
            raw_tokens = list(raw_spec)
        else:
            raw_tokens = [raw_spec]

        tokens = []
        for token in raw_tokens:
            t = str(token).strip()
            if not t:
                continue
            try:
                tokens.append(int(t))
            except Exception:
                tokens.append(t)
        return tokens

    def _resolve_pose_group_indices(self, raw_spec, keypoint_names):
        """Resolve keypoint group names/indices to deduplicated index list."""
        names = [str(v) for v in (keypoint_names or [])]
        tokens = self._parse_pose_group_tokens(raw_spec)
        if not tokens:
            return []

        lower_map = {name.lower(): idx for idx, name in enumerate(names)}
        indices = []
        seen = set()
        for tok in tokens:
            idx = None
            if isinstance(tok, int):
                if 0 <= tok < len(names):
                    idx = int(tok)
            else:
                idx = lower_map.get(str(tok).strip().lower(), None)
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        return indices

    def _compute_pose_heading_from_keypoints(
        self,
        keypoints,
        anterior_indices,
        posterior_indices,
        min_valid_conf,
    ):
        """
        Estimate directed heading (posterior -> anterior) from pose keypoints.

        Returns None if either group has no valid keypoints above min_valid_conf.
        """
        if keypoints is None:
            return None
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            return None

        def weighted_centroid(indices):
            pts = []
            weights = []
            for idx in indices:
                if idx < 0 or idx >= len(arr):
                    continue
                x, y, conf = arr[idx]
                if (
                    not np.isfinite(x)
                    or not np.isfinite(y)
                    or not np.isfinite(conf)
                    or float(conf) < float(min_valid_conf)
                ):
                    continue
                pts.append((float(x), float(y)))
                weights.append(max(1e-6, float(conf)))
            if not pts:
                return None
            pts_arr = np.asarray(pts, dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64)
            cx = float(np.average(pts_arr[:, 0], weights=w_arr))
            cy = float(np.average(pts_arr[:, 1], weights=w_arr))
            return cx, cy

        ant = weighted_centroid(anterior_indices)
        post = weighted_centroid(posterior_indices)
        if ant is None or post is None:
            return None

        dx = ant[0] - post[0]
        dy = ant[1] - post[1]
        if not np.isfinite(dx) or not np.isfinite(dy):
            return None
        return self._normalize_theta(math.atan2(dy, dx))

    def _build_pose_detection_keypoint_map(self, pose_props_cache, frame_idx):
        """Build detection_id -> pose_keypoints map for one frame from cache."""
        if pose_props_cache is None:
            return {}
        try:
            frame = pose_props_cache.get_frame(int(frame_idx))
        except Exception:
            return {}
        ids = frame.get("detection_ids", [])
        keypoints = frame.get("pose_keypoints", [])
        n = min(len(ids), len(keypoints))
        out = {}
        for i in range(n):
            try:
                det_id = int(ids[i])
            except Exception:
                continue
            out[det_id] = keypoints[i]
        return out

    def _compute_pose_geometry_from_keypoints(
        self,
        keypoints,
        anterior_indices,
        posterior_indices,
        min_valid_conf,
        ignore_indices=None,
    ):
        """Extract heading, body length, and visibility from pose keypoints."""
        if keypoints is None:
            return None
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            return None

        ignore_set = {int(idx) for idx in (ignore_indices or [])}

        def weighted_centroid(indices):
            pts = []
            weights = []
            for idx in indices:
                if idx in ignore_set or idx < 0 or idx >= len(arr):
                    continue
                x, y, conf = arr[idx]
                if (
                    not np.isfinite(x)
                    or not np.isfinite(y)
                    or not np.isfinite(conf)
                    or float(conf) < float(min_valid_conf)
                ):
                    continue
                pts.append((float(x), float(y)))
                weights.append(max(1e-6, float(conf)))
            if not pts:
                return None
            pts_arr = np.asarray(pts, dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64)
            cx = float(np.average(pts_arr[:, 0], weights=w_arr))
            cy = float(np.average(pts_arr[:, 1], weights=w_arr))
            return cx, cy

        valid_total = 0
        visible_total = 0
        for idx in range(len(arr)):
            if idx in ignore_set:
                continue
            valid_total += 1
            conf = arr[idx, 2]
            if np.isfinite(conf) and float(conf) >= float(min_valid_conf):
                visible_total += 1
        visibility = (
            float(visible_total) / float(valid_total) if valid_total > 0 else 0.0
        )

        ant = weighted_centroid(anterior_indices)
        post = weighted_centroid(posterior_indices)
        if ant is None or post is None:
            return {
                "heading": None,
                "body_length": None,
                "visibility": float(np.clip(visibility, 0.0, 1.0)),
            }

        dx = ant[0] - post[0]
        dy = ant[1] - post[1]
        if not np.isfinite(dx) or not np.isfinite(dy):
            heading = None
            body_length = None
        else:
            heading = self._normalize_theta(math.atan2(dy, dx))
            body_length = float(math.hypot(dx, dy))

        return {
            "heading": heading,
            "body_length": body_length,
            "visibility": float(np.clip(visibility, 0.0, 1.0)),
        }

    def _normalize_pose_keypoints(self, keypoints, min_valid_conf, ignore_indices=None):
        """Center and scale pose keypoints for same-keypoint shape comparison."""
        if keypoints is None:
            return None
        arr = np.asarray(keypoints, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
            return None

        ignore_set = {int(idx) for idx in (ignore_indices or [])}
        valid = np.zeros(len(arr), dtype=bool)
        valid_points = []
        valid_weights = []
        for idx in range(len(arr)):
            if idx in ignore_set:
                continue
            x, y, conf = arr[idx]
            if (
                np.isfinite(x)
                and np.isfinite(y)
                and np.isfinite(conf)
                and float(conf) >= float(min_valid_conf)
            ):
                valid[idx] = True
                valid_points.append((float(x), float(y)))
                valid_weights.append(max(1e-6, float(conf)))

        if not valid_points:
            return None

        pts_arr = np.asarray(valid_points, dtype=np.float64)
        w_arr = np.asarray(valid_weights, dtype=np.float64)
        cx = float(np.average(pts_arr[:, 0], weights=w_arr))
        cy = float(np.average(pts_arr[:, 1], weights=w_arr))
        centered = pts_arr - np.array([[cx, cy]], dtype=np.float64)
        radii = np.sqrt(np.sum(centered**2, axis=1))
        scale = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
        scale = max(scale, 1.0)

        out = np.full((len(arr), 3), np.nan, dtype=np.float32)
        out[:, 2] = 0.0
        valid_indices = np.where(valid)[0]
        for src_idx, kp_idx in enumerate(valid_indices):
            out[kp_idx, 0] = np.float32(centered[src_idx, 0] / scale)
            out[kp_idx, 1] = np.float32(centered[src_idx, 1] / scale)
            out[kp_idx, 2] = np.float32(arr[kp_idx, 2])
        return out

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

    def _should_precompute_individual_data(self, params, detection_method):
        """Run pose precompute when the pose extractor is enabled."""
        pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
        return bool(
            not self.backward_mode
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and pose_enabled
        )

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

    @staticmethod
    def _select_directed_heading(
        pose_heading,
        pose_directed,
        headtail_heading,
        headtail_directed,
        pose_overrides_headtail=True,
    ):
        """Choose directed heading source (pose/head-tail) according to precedence."""
        pose_valid = bool(pose_directed) and np.isfinite(float(pose_heading))
        headtail_valid = bool(headtail_directed) and np.isfinite(
            float(headtail_heading)
        )
        if pose_overrides_headtail:
            if pose_valid:
                return float(pose_heading), True
            if headtail_valid:
                return float(headtail_heading), True
            return float("nan"), False
        if headtail_valid:
            return float(headtail_heading), True
        if pose_valid:
            return float(pose_heading), True
        return float("nan"), False

    def _precompute_pose_data(
        self,
        detector,
        params,
        detection_cache,
        start_frame,
        end_frame,
    ):
        """
        Precompute pose keypoints in a single video pass.

        Returns:
            (pose_cache_path, pose_cache_hit)
        """
        import time

        pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
        detection_method = params.get("DETECTION_METHOD", "background_subtraction")

        if detection_method != "yolo_obb":
            if pose_enabled:
                logger.warning("Pose precompute requires YOLO OBB detection mode.")
            return None, True

        if not pose_enabled:
            return None, True

        pose_cache_path = None
        pose_cache_hit = True
        pose_backend = None
        pose_cache_writer = None
        properties_id = None
        detection_hash = None
        filter_hash = None
        extractor_hash = None

        if pose_enabled:
            from multi_tracker.core.identity.properties_cache import (
                IndividualPropertiesCache,
                compute_detection_hash,
                compute_extractor_hash,
                compute_filter_settings_hash,
                compute_individual_properties_id,
            )
            from multi_tracker.core.identity.runtime_api import (
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
            self.individual_properties_id = str(properties_id)
            self.individual_properties_cache_path = str(pose_cache_path)
            params["INDIVIDUAL_PROPERTIES_ID"] = properties_id
            params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(pose_cache_path)

            # Check for existing pose cache
            if pose_cache_path.exists():
                existing = IndividualPropertiesCache(str(pose_cache_path), mode="r")
                try:
                    if existing.is_compatible():
                        logger.info(
                            "Pose properties cache hit: %s", str(pose_cache_path)
                        )
                    else:
                        pose_cache_hit = False
                finally:
                    existing.close()
            else:
                pose_cache_hit = False

        if pose_cache_hit:
            self.progress_signal.emit(
                100,
                "Precompute: using existing pose cache.",
            )
            return str(pose_cache_path) if pose_cache_path else None, True

        logger.info("=" * 80)
        logger.info("POSE PRECOMPUTE: single video pass")
        logger.info("=" * 80)

        if pose_enabled and not pose_cache_hit:
            logger.info("Initializing pose backend...")
            self.progress_signal.emit(0, "Precompute: loading pose backend...")
            from multi_tracker.core.identity.runtime_api import (
                build_runtime_config,
                create_pose_backend_from_config,
            )

            pose_out_root = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
            if not pose_out_root:
                pose_out_root = str(pose_cache_path.parent) if pose_cache_path else "."

            pose_config = build_runtime_config(params, out_root=pose_out_root)
            pose_backend = create_pose_backend_from_config(pose_config)
            pose_backend.warmup()
            logger.info("Pose backend ready.")

            # Store resolved artifact path if applicable
            runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "")).lower()
            if runtime_flavor.startswith("onnx") or runtime_flavor.startswith(
                "tensorrt"
            ):
                try:
                    resolved_artifact = str(
                        getattr(pose_backend, "exported_model_path", "")
                        or getattr(pose_backend, "model_path", "")
                    ).strip()
                except Exception:
                    resolved_artifact = ""
                if resolved_artifact:
                    params["POSE_EXPORTED_MODEL_PATH"] = resolved_artifact
                    self.pose_exported_model_resolved_signal.emit(resolved_artifact)

        resize_f = float(params.get("RESIZE_FACTOR", 1.0))
        padding_fraction = float(params.get("INDIVIDUAL_CROP_PADDING", 0.1))

        roi_mask = params.get("ROI_MASK", None)
        if roi_mask is not None:
            dims_cap = cv2.VideoCapture(self.video_path)
            base_h = int(dims_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            base_w = int(dims_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            dims_cap.release()
            target_w = max(1, int(base_w * resize_f))
            target_h = max(1, int(base_h * resize_f))
            if roi_mask.shape[1] != target_w or roi_mask.shape[0] != target_h:
                roi_mask = cv2.resize(roi_mask, (target_w, target_h), cv2.INTER_NEAREST)

        # Open video once
        video_cap = cv2.VideoCapture(self.video_path)
        if not video_cap.isOpened():
            raise RuntimeError(
                f"Failed to open video for unified precompute: {self.video_path}"
            )
        if start_frame > 0:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Open cache writers
        if pose_enabled and not pose_cache_hit:
            from multi_tracker.core.identity.properties_cache import (
                IndividualPropertiesCache,
            )

            pose_cache_writer = IndividualPropertiesCache(
                str(pose_cache_path), mode="w"
            )

        # Cross-frame batch size: accumulate this many crops across frames before
        # calling pose_backend.predict_batch.  Larger values → better GPU utilisation
        # at the cost of slightly delayed writes.  64 is a good default.
        _POSE_CROSS_FRAME_BATCH = int(params.get("POSE_PRECOMPUTE_BATCH_SIZE", 64))
        _bg_raw = params.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])
        _pose_bg_color = (
            tuple(int(c) for c in _bg_raw)
            if isinstance(_bg_raw, (list, tuple)) and len(_bg_raw) == 3
            else (0, 0, 0)
        )

        total_frames = max(1, end_frame - start_frame + 1)
        precompute_start_ts = time.time()
        cancelled = False

        self.progress_signal.emit(
            1,
            f"Pose precompute: processing {total_frames} frame(s)...",
        )

        # Accumulators for cross-frame batching
        _pending: list = []  # list of per-frame dicts
        _flat_crops: list = []  # every pending crop in arrival order

        def _flush_pose_batch():
            """Run inference on all accumulated crops and write results to cache."""
            if not _pending:
                return
            if _flat_crops and pose_backend:
                all_pred = pose_backend.predict_batch(_flat_crops)
            else:
                all_pred = []
            out_offset = 0
            for pf in _pending:
                n_crops = len(pf["crops"])
                batch_slice = all_pred[out_offset : out_offset + n_crops]
                out_offset += n_crops
                pose_outputs: list = [{} for _ in range(pf["n_dets"])]
                for ci, det_idx in enumerate(pf["crop_to_det"]):
                    if ci >= len(batch_slice):
                        break
                    out = batch_slice[ci]
                    kpts = out.keypoints
                    crop_offset = pf["crop_offsets"].get(det_idx)
                    if kpts is not None and crop_offset is not None and len(kpts) > 0:
                        x0, y0 = crop_offset
                        gkpts = np.asarray(kpts, dtype=np.float32).copy()
                        gkpts[:, 0] += float(x0)
                        gkpts[:, 1] += float(y0)
                        all_obbs = pf.get("all_obb_corners", [])
                        if len(all_obbs) > 1:
                            from multi_tracker.core.tracking.pose_features import (
                                filter_keypoints_by_foreign_obbs,
                            )

                            gkpts = filter_keypoints_by_foreign_obbs(
                                gkpts, all_obbs, target_idx=det_idx
                            )
                    else:
                        gkpts = kpts
                    pose_outputs[det_idx] = {"keypoints": gkpts}
                kp_list = [
                    pose_outputs[d].get("keypoints", None) for d in range(pf["n_dets"])
                ]
                if pose_cache_writer:
                    pose_cache_writer.add_frame(
                        pf["frame_idx"], pf["det_ids"], pose_keypoints=kp_list
                    )
            _pending.clear()
            _flat_crops.clear()

        try:
            for rel_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if self._stop_requested:
                    cancelled = True
                    break

                # Get filtered detections from cache
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                ) = detection_cache.get_frame(frame_idx)

                (
                    meas,
                    _sizes,
                    _shapes,
                    _confs,
                    filtered_obb_corners,
                    detection_ids,
                    _heading_hints,
                    _directed_mask,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=roi_mask,
                    detection_ids=raw_detection_ids,
                    heading_hints=raw_heading_hints,
                    directed_mask=raw_directed_mask,
                )

                # Read frame
                ret, frame = video_cap.read()
                if ret and resize_f < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                # Build per-frame record (crop extraction only — no inference yet)
                _all_obb_corners = [
                    np.asarray(c, dtype=np.float32)
                    for c in (filtered_obb_corners or [])
                ]
                pf: dict = {
                    "frame_idx": frame_idx,
                    "det_ids": detection_ids,
                    "n_dets": len(meas),
                    "crops": [],
                    "crop_to_det": [],
                    "crop_offsets": {},
                    "all_obb_corners": _all_obb_corners,
                }
                if ret and meas and filtered_obb_corners:
                    from multi_tracker.core.tracking.pose_features import (
                        apply_foreign_obb_mask,
                    )

                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            if len(_all_obb_corners) > 1:
                                other_corners = [
                                    _all_obb_corners[j]
                                    for j in range(len(_all_obb_corners))
                                    if j != det_idx
                                ]
                                crop = apply_foreign_obb_mask(
                                    crop,
                                    crop_offset[0],
                                    crop_offset[1],
                                    other_corners,
                                    background_color=_pose_bg_color,
                                )
                            pf["crops"].append(crop)
                            pf["crop_to_det"].append(det_idx)
                            pf["crop_offsets"][det_idx] = crop_offset

                _pending.append(pf)
                _flat_crops.extend(pf["crops"])

                is_last = rel_idx == total_frames - 1
                if len(_flat_crops) >= _POSE_CROSS_FRAME_BATCH or is_last:
                    if self._stop_requested:
                        cancelled = True
                        break
                    _flush_pose_batch()

                processed_count = rel_idx + 1
                if rel_idx % 10 == 0 or is_last:
                    elapsed = max(1e-6, time.time() - precompute_start_ts)
                    rate_fps = processed_count / elapsed
                    remaining = max(0, total_frames - processed_count)
                    eta = (remaining / rate_fps) if rate_fps > 1e-9 else 0.0
                    pct = int((processed_count * 100) / total_frames)
                    self.progress_signal.emit(
                        pct,
                        f"Pose precompute: {processed_count}/{total_frames}",
                    )
                    self.stats_signal.emit(
                        {
                            "phase": "pose_precompute",
                            "fps": rate_fps,
                            "elapsed": elapsed,
                            "eta": eta,
                        }
                    )

            if cancelled or self._stop_requested:
                logger.info("Unified precompute cancelled.")
                self.progress_signal.emit(0, "Precompute cancelled.")
                return None, False  # was incorrectly returning 4 values before

            # Save caches
            if pose_cache_writer:
                pose_cache_writer.save(
                    metadata={
                        "individual_properties_id": properties_id,
                        "detection_hash": detection_hash,
                        "filter_settings_hash": filter_hash,
                        "extractor_hash": extractor_hash,
                        "pose_keypoint_names": list(
                            getattr(pose_backend, "output_keypoint_names", []) or []
                        ),
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "video_path": str(Path(self.video_path).expanduser().resolve()),
                    }
                )
                logger.info("Pose properties cache saved: %s", str(pose_cache_path))

            self.progress_signal.emit(
                100,
                "Pose precompute complete: pose cache saved.",
            )

        finally:
            if pose_cache_writer:
                pose_cache_writer.close()
            video_cap.release()
            if pose_backend:
                try:
                    pose_backend.close()
                except Exception:
                    pass

        return (
            str(pose_cache_path) if pose_cache_path else None
        ), not pose_enabled or pose_cache_hit

    def _run_batched_detection_phase(
        self, cap, detection_cache, detector, params, start_frame, end_frame
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

    def run(self: object) -> object:
        """Execute tracking pipeline for the configured video and parameters."""
        # === 1. INITIALIZATION (Identical to Original) ===
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()

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
        bg_model = None
        if detection_method == "background_subtraction":
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

        individual_data_precompute_enabled = self._should_precompute_individual_data(
            p, detection_method
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
            logger.info("Enabling batched YOLO prepass for pose precompute.")

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
        track_pose_prototypes = [None] * N
        track_avg_step = [0.0] * N
        tracking_continuity = [0] * N
        trajectory_ids, next_trajectory_id = list(range(N)), N

        detection_initialized, tracking_stabilized = False, False
        detection_counts, tracking_counts = 0, 0

        start_time, self.frame_count, fps_list = time.time(), 0, []
        local_counts, intensity_history, lighting_state = [0] * N, deque(maxlen=50), {}
        roi_fill_color = None  # Average color outside ROI for visualization overlay

        # Profiling accumulators
        profile_times = {
            "frame_read": 0.0,
            "preprocessing": 0.0,
            "detection": 0.0,
            "assignment": 0.0,
            "tracking_update": 0.0,
            "visualization": 0.0,
            "video_write": 0.0,
            "gui_emit": 0.0,
        }
        profile_counts = 0
        PROFILE_INTERVAL = 100  # Log every 100 frames

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
            frames_processed = self._run_batched_detection_phase(
                cap, detection_cache, detector, p, start_frame, end_frame
            )

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

        if individual_data_precompute_enabled:
            if detection_cache is None or not use_cached_detections:
                logger.error(
                    "Individual properties precompute requires cached raw detections."
                )
                if detection_cache:
                    detection_cache.close()
                cap.release()
                if self.video_writer:
                    self.video_writer.release()
                self.finished_signal.emit(False, [], [])
                return

            try:
                props_path, props_cache_hit = self._precompute_pose_data(
                    detector, p, detection_cache, start_frame, end_frame
                )

                if props_path:
                    state = "hit" if props_cache_hit else "miss"
                    logger.info("Individual properties cache %s: %s", state, props_path)

            except Exception as exc:
                logger.exception("Individual data precompute failed.")
                self.warning_signal.emit(
                    "Individual Precompute Failed",
                    f"Aborting tracking run because individual precompute failed:\n{exc}",
                )
                if detection_cache:
                    detection_cache.close()
                cap.release()
                if self.video_writer:
                    self.video_writer.release()
                self.finished_signal.emit(False, [], [])
                return

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

        for frame, _ in frame_iterator:
            loop_start = time.time()

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
            prep_start = time.time()
            resize_f = params["RESIZE_FACTOR"]

            # Skip preprocessing if no frame (cached detection mode)
            if frame is not None:
                # Keep original frame for individual dataset generation (high resolution)
                original_frame = frame.copy() if individual_generator else None

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

            profile_times["preprocessing"] += time.time() - prep_start

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

            detect_start = time.time()

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
                use_gpu = params.get("ENABLE_GPU_BACKGROUND", False)
                gray = apply_image_adjustments(
                    gray,
                    params["BRIGHTNESS"],
                    params["CONTRAST"],
                    params["GAMMA"],
                    use_gpu,
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
                        use_gpu,
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
                    fg_mask = detector.apply_conservative_split(fg_mask)
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

            elif (
                detection_method == "yolo_obb" and frame is not None
            ):  # YOLO OBB detection
                # YOLO uses the original BGR frame directly without masking
                # This preserves natural image context for better confidence estimates
                yolo_frame = frame.copy()

                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    yolo_results,
                    raw_confidences,
                    raw_obb_corners,
                    raw_heading_hints,
                    raw_directed_mask,
                ) = detector.detect_objects(
                    yolo_frame,
                    self.frame_count,
                    return_raw=True,
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
                    yolo_results._filtered_obb_corners = filtered_obb_corners

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
                )
                cached_frame_indices.add(actual_frame_index)

            profile_times["detection"] += time.time() - detect_start

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
                for det_idx in range(len(meas)):
                    pose_heading = (
                        float(detection_pose_heading[det_idx])
                        if det_idx < len(detection_pose_heading)
                        else float("nan")
                    )
                    pose_directed = bool(
                        det_idx < len(pose_directed_mask)
                        and pose_directed_mask[det_idx] == 1
                    )
                    headtail_heading = (
                        float(detection_headtail_heading[det_idx])
                        if det_idx < len(detection_headtail_heading)
                        else float("nan")
                    )
                    headtail_directed = bool(
                        det_idx < len(headtail_directed_mask)
                        and headtail_directed_mask[det_idx] == 1
                    )
                    selected_heading, is_directed = self._select_directed_heading(
                        pose_heading=pose_heading,
                        pose_directed=pose_directed,
                        headtail_heading=headtail_heading,
                        headtail_directed=headtail_directed,
                        pose_overrides_headtail=pose_overrides_headtail,
                    )
                    if is_directed:
                        detection_directed_heading[det_idx] = selected_heading
                        detection_directed_mask[det_idx] = 1

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
                    # Draw cyan dashed boundary around ROI using contours from the mask
                    contours, _ = cv2.findContours(
                        ROI_mask_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # Draw cyan boundary (BGR: 255, 255, 0)
                        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)
            else:
                overlay = None

            if detection_initialized and meas:
                # --- Assignment ---
                assign_start = time.time()
                preds = self.kf_manager.get_predictions()

                # The Assigner now takes the kf_manager directly to access X and S_inv
                association_data = {
                    "detection_confidences": detection_confidences,
                    "detection_crop_quality": detection_crop_quality,
                    "detection_pose_heading": detection_directed_heading,
                    "detection_pose_keypoints": detection_pose_keypoints,
                    "detection_pose_visibility": detection_pose_visibility,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": track_avg_step,
                }

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

                # --- KF Update & State Update ---
                total_cost = 0.0
                for r, c in zip(rows, cols):
                    meas_x = float(meas[c][0])
                    meas_y = float(meas[c][1])
                    measured_theta = float(meas[c][2])
                    directed_heading = bool(
                        c < len(detection_directed_mask)
                        and detection_directed_mask[c] == 1
                    )
                    tracking_theta_measurement = measured_theta
                    if directed_heading and c < len(detection_directed_heading):
                        directed_theta = float(detection_directed_heading[c])
                        if np.isfinite(directed_theta):
                            tracking_theta_measurement = directed_theta

                    theta_for_tracking = self._resolve_tracking_theta(
                        r,
                        tracking_theta_measurement,
                        pose_directed=directed_heading,
                        orientation_last=orientation_last,
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
                    self.kf_manager.correct(r, corrected_meas)
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
                    orientation_last[r] = self._smooth_orientation(
                        r,
                        theta_for_tracking,
                        speed,
                        params,
                        orientation_last,
                        position_deques,
                    )
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

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1

                for d_idx in free_dets:
                    for track_idx in range(N):
                        if track_states[track_idx] == "lost":
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

            profile_times["assignment"] += (
                time.time() - assign_start if detection_initialized and meas else 0
            )

            # --- Individual Dataset Generation (supports YOLO OBB and BG subtraction) ---
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

                # Use original-frame coordinates for crop extraction.
                coord_scale_factor = 1.0 / resize_f

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
                    )

            # --- Tracking State Updates ---
            track_start = time.time()
            # (All the tracking state updates happen here - already in code)
            profile_times["tracking_update"] += time.time() - track_start

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
                viz_start = time.time()
                trajectories_pruned = [
                    [
                        pt
                        for pt in tr
                        if self.frame_count - pt[3]
                        <= params["TRAJECTORY_HISTORY_SECONDS"]
                    ]
                    for tr in self.trajectories_full
                ]

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
                profile_times["visualization"] += time.time() - viz_start

                write_start = time.time()
                if self.video_writer:
                    self.video_writer.write(overlay)
                profile_times["video_write"] += time.time() - write_start

                emit_start = time.time()
                # For YOLO with ROI, draw boundary overlay before emitting
                if (
                    detection_method != "background_subtraction"
                    and ROI_mask_current is not None
                ):
                    # Find contours of ROI mask
                    contours, _ = cv2.findContours(
                        ROI_mask_current, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    # Draw cyan boundary
                    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

                self.emit_frame(overlay)
                profile_times["gui_emit"] += time.time() - emit_start

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

            # Calculate frame read time (total loop time - all other operations)
            loop_time = time.time() - loop_start
            other_time = sum(profile_times.values()) - profile_times["frame_read"]
            profile_times["frame_read"] += max(0, loop_time - other_time)

            profile_counts += 1

            # Log profiling summary periodically
            if profile_counts % PROFILE_INTERVAL == 0:
                total_time = sum(profile_times.values())
                if total_time > 0:
                    logger.info(
                        "=== PROFILING SUMMARY (last %d frames) ===", PROFILE_INTERVAL
                    )
                    for key in sorted(profile_times.keys()):
                        pct = (profile_times[key] / total_time) * 100
                        avg_ms = (profile_times[key] / PROFILE_INTERVAL) * 1000
                        logger.info("  %s: %.1f%% (%.2fms/frame)", key, pct, avg_ms)
                    logger.info(
                        "  Total: %.2fms/frame", (total_time / PROFILE_INTERVAL) * 1000
                    )
                    logger.info("===========================================")
                    # Reset counters
                    for key in profile_times:
                        profile_times[key] = 0.0
                    profile_counts = 0

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_list.append(self.frame_count / elapsed)

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

        logger.info("Tracking worker finished. Emitting raw trajectory data.")

        self.finished_signal.emit(
            not self._stop_requested, fps_list, self.trajectories_full
        )

    def _smooth_orientation(
        self, r, theta, speed, p, orientation_last, position_deques
    ):
        final_theta, old = theta, orientation_last[r]
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
