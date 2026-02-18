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
    histogram_data_signal = Signal(dict)
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

    def _predict_pose_on_crops(
        self,
        pose_model,
        crops,
        model_device: str,
        min_valid_conf: float,
    ):
        """Run pose model on crops and return per-crop summary + keypoints arrays."""
        if not crops:
            return []

        results = pose_model.predict(
            source=crops,
            conf=1e-4,
            iou=0.7,
            max_det=1,
            verbose=False,
            device=model_device,
        )

        outputs = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                outputs.append(
                    {
                        "keypoints": None,
                        "mean_conf": 0.0,
                        "valid_fraction": 0.0,
                        "num_valid": 0,
                        "num_keypoints": 0,
                    }
                )
                continue

            try:
                xy = keypoints.xy
                xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
                conf = getattr(keypoints, "conf", None)
                if conf is not None:
                    conf = (
                        conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
                    )
            except Exception:
                xy = np.empty((0, 0, 2), dtype=np.float32)
                conf = None

            if xy.ndim == 2:
                xy = xy[None, :, :]
            if conf is not None and conf.ndim == 1:
                conf = conf[None, :]

            if xy.size == 0:
                outputs.append(
                    {
                        "keypoints": None,
                        "mean_conf": 0.0,
                        "valid_fraction": 0.0,
                        "num_valid": 0,
                        "num_keypoints": 0,
                    }
                )
                continue

            if conf is None:
                conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)
            mean_per_instance = np.nanmean(conf, axis=1)
            best_idx = (
                int(np.nanargmax(mean_per_instance)) if len(mean_per_instance) else 0
            )
            pred_xy = np.asarray(xy[best_idx], dtype=np.float32)
            pred_conf = np.asarray(conf[best_idx], dtype=np.float32)
            if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
                outputs.append(
                    {
                        "keypoints": None,
                        "mean_conf": 0.0,
                        "valid_fraction": 0.0,
                        "num_valid": 0,
                        "num_keypoints": 0,
                    }
                )
                continue

            keypoints_arr = np.column_stack((pred_xy, pred_conf)).astype(np.float32)
            if len(keypoints_arr) > 0:
                valid_mask = keypoints_arr[:, 2] >= float(min_valid_conf)
                mean_conf = float(np.nanmean(keypoints_arr[:, 2]))
                valid_fraction = float(np.mean(valid_mask))
                num_valid = int(np.sum(valid_mask))
                num_keypoints = int(len(keypoints_arr))
            else:
                mean_conf = 0.0
                valid_fraction = 0.0
                num_valid = 0
                num_keypoints = 0

            outputs.append(
                {
                    "keypoints": keypoints_arr,
                    "mean_conf": mean_conf,
                    "valid_fraction": valid_fraction,
                    "num_valid": num_valid,
                    "num_keypoints": num_keypoints,
                }
            )

        return outputs

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

    def _precompute_individual_data_unified(
        self,
        detector,
        params,
        detection_cache,
        start_frame,
        end_frame,
    ):
        """
        Unified precompute for pose keypoints and appearance embeddings in a single video pass.

        This method combines pose and appearance extraction to avoid multiple video reads and
        redundant crop extractions, significantly improving efficiency.

        Returns:
            (pose_cache_path, pose_cache_hit, appearance_cache_path, appearance_cache_hit)
        """
        import time

        # Check what's enabled
        pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
        appearance_enabled = bool(params.get("APPEARANCE_ENABLED", False))
        detection_method = params.get("DETECTION_METHOD", "background_subtraction")

        if detection_method != "yolo_obb":
            if pose_enabled or appearance_enabled:
                logger.warning(
                    "Individual data precompute requires YOLO OBB detection mode."
                )
            return None, True, None, True

        if not pose_enabled and not appearance_enabled:
            return None, True, None, True

        # ===== POSE SETUP =====
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

        # ===== APPEARANCE SETUP =====
        appearance_cache_path = None
        appearance_cache_hit = True
        appearance_backend = None
        appearance_cache_writer = None
        embedding_id = None
        embedding_dim = None
        appearance_config = None

        if appearance_enabled:
            from multi_tracker.core.identity.appearance_cache import (
                AppearanceEmbeddingCache,
                build_appearance_cache_path,
                compute_appearance_embedding_id,
                compute_appearance_extractor_hash,
            )
            from multi_tracker.core.identity.properties_cache import (
                compute_detection_hash,
                compute_filter_settings_hash,
            )
            from multi_tracker.core.identity.runtime_api import (
                AppearanceRuntimeConfig,
                create_appearance_backend_from_config,
            )

            detection_hash = compute_detection_hash(
                params.get("INFERENCE_MODEL_ID", ""),
                self.video_path,
                start_frame,
                end_frame,
                detection_cache_version="2.0",
            )
            filter_hash = compute_filter_settings_hash(params)
            appearance_hash = compute_appearance_extractor_hash(params)
            embedding_id = compute_appearance_embedding_id(
                detection_hash, filter_hash, appearance_hash
            )

            video_stem = Path(self.video_path).stem
            output_dir = params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "").strip()
            if not output_dir:
                output_dir = str(Path(self.video_path).parent / "appearance_embeddings")

            appearance_cache_path = build_appearance_cache_path(
                output_dir, video_stem, embedding_id, start_frame, end_frame
            )

            # Check for existing appearance cache
            if appearance_cache_path.exists():
                existing = AppearanceEmbeddingCache(
                    str(appearance_cache_path), mode="r"
                )
                try:
                    if existing.is_compatible():
                        meta = existing.metadata or {}
                        if (
                            str(meta.get("model_name", ""))
                            == params.get("APPEARANCE_MODEL_NAME", "")
                            and int(meta.get("start_frame", -1)) == int(start_frame)
                            and int(meta.get("end_frame", -1)) == int(end_frame)
                        ):
                            logger.info(
                                "Appearance embedding cache hit: %s",
                                str(appearance_cache_path),
                            )
                        else:
                            appearance_cache_hit = False
                    else:
                        appearance_cache_hit = False
                finally:
                    existing.close()
            else:
                appearance_cache_hit = False

        # If both are cache hits, return early
        if pose_cache_hit and appearance_cache_hit:
            self.progress_signal.emit(
                100,
                "Precompute: using existing pose and appearance caches.",
            )
            return (
                str(pose_cache_path) if pose_cache_path else None,
                True,
                str(appearance_cache_path) if appearance_cache_path else None,
                True,
            )

        # ===== UNIFIED EXTRACTION =====
        logger.info("=" * 80)
        logger.info("UNIFIED PRECOMPUTE: Pose + Appearance (single video pass)")
        logger.info("=" * 80)

        # Initialize backends
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

        if appearance_enabled and not appearance_cache_hit:
            logger.info("Initializing appearance backend...")
            self.progress_signal.emit(0, "Precompute: loading appearance model...")

            from multi_tracker.core.identity.runtime_api import (
                AppearanceRuntimeConfig,
                create_appearance_backend_from_config,
            )

            appearance_config = AppearanceRuntimeConfig(
                model_name=params.get(
                    "APPEARANCE_MODEL_NAME", "timm/vit_base_patch14_dinov2.lvd142m"
                ),
                runtime_flavor=params.get("APPEARANCE_RUNTIME_FLAVOR", "auto"),
                batch_size=params.get("APPEARANCE_BATCH_SIZE", 32),
                max_image_side=params.get("APPEARANCE_MAX_IMAGE_SIDE", 512),
                use_clahe=params.get("APPEARANCE_USE_CLAHE", False),
                normalize_embeddings=params.get("APPEARANCE_NORMALIZE", True),
                compute_runtime=str(
                    params.get("COMPUTE_RUNTIME", params.get("compute_runtime", ""))
                ),
            )

            appearance_backend = create_appearance_backend_from_config(
                appearance_config
            )
            appearance_backend.warmup()
            embedding_dim = appearance_backend.output_dimension
            logger.info(
                f"Appearance model loaded. Embedding dimension: {embedding_dim}"
            )

        # Common setup
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
        if appearance_enabled and not appearance_cache_hit:
            from multi_tracker.core.identity.appearance_cache import (
                AppearanceEmbeddingCache,
            )

            appearance_cache_writer = AppearanceEmbeddingCache(
                str(appearance_cache_path), mode="w"
            )

        # Process frames
        total_frames = max(1, end_frame - start_frame + 1)
        precompute_start_ts = time.time()
        cancelled = False

        self.progress_signal.emit(
            1,
            f"Unified precompute: processing {total_frames} frame(s)...",
        )

        try:
            for rel_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if self._stop_requested:
                    cancelled = True
                    break

                # Get detections
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                ) = detection_cache.get_frame(frame_idx)

                (
                    meas,
                    _sizes,
                    _shapes,
                    _confs,
                    filtered_obb_corners,
                    detection_ids,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=roi_mask,
                    detection_ids=raw_detection_ids,
                )

                # Read frame
                ret, frame = video_cap.read()
                if not ret:
                    if pose_cache_writer:
                        pose_cache_writer.add_frame(
                            frame_idx, detection_ids or [], pose_keypoints=[]
                        )
                    if appearance_cache_writer:
                        appearance_cache_writer.add_frame(
                            frame_idx, detection_ids or [], embeddings=[]
                        )
                    continue

                if resize_f < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                # Extract crops once
                pose_keypoints = []
                embeddings_list = []

                if meas and filtered_obb_corners:
                    crops = []
                    crop_to_det = []
                    crop_offsets = {}

                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            crops.append(crop)
                            crop_to_det.append(det_idx)
                            crop_offsets[det_idx] = crop_offset

                    if crops:
                        if self._stop_requested:
                            cancelled = True
                            break

                        # Process pose
                        if pose_backend:
                            pose_outputs = [{} for _ in range(len(meas))]
                            pred_outputs = pose_backend.predict_batch(crops)
                            for crop_idx, det_idx in enumerate(crop_to_det):
                                if crop_idx < len(pred_outputs):
                                    out = pred_outputs[crop_idx]
                                    crop_offset = crop_offsets.get(det_idx)
                                    kpts = out.keypoints
                                    if (
                                        kpts is not None
                                        and crop_offset is not None
                                        and len(kpts) > 0
                                    ):
                                        x0, y0 = crop_offset
                                        gkpts = np.asarray(
                                            kpts, dtype=np.float32
                                        ).copy()
                                        gkpts[:, 0] += float(x0)
                                        gkpts[:, 1] += float(y0)
                                    else:
                                        gkpts = kpts
                                    pose_outputs[det_idx] = {"keypoints": gkpts}

                            for det_idx in range(len(meas)):
                                out = pose_outputs[det_idx]
                                pose_keypoints.append(out.get("keypoints", None))

                        # Process appearance
                        if appearance_backend:
                            embeddings_list = [None] * len(meas)
                            appearance_results = appearance_backend.predict_crops(crops)
                            for crop_idx, det_idx in enumerate(crop_to_det):
                                if crop_idx < len(appearance_results):
                                    result = appearance_results[crop_idx]
                                    embeddings_list[det_idx] = result.embedding

                # Write to caches
                if pose_cache_writer:
                    pose_cache_writer.add_frame(
                        frame_idx,
                        detection_ids,
                        pose_keypoints=pose_keypoints,
                    )

                if appearance_cache_writer:
                    appearance_cache_writer.add_frame(
                        frame_idx,
                        detection_ids or [],
                        embeddings=embeddings_list,
                    )

                # Progress update
                processed_count = rel_idx + 1
                if rel_idx % 10 == 0 or rel_idx == total_frames - 1:
                    elapsed = max(1e-6, time.time() - precompute_start_ts)
                    rate_fps = processed_count / elapsed
                    remaining = max(0, total_frames - processed_count)
                    eta = (remaining / rate_fps) if rate_fps > 1e-9 else 0.0
                    pct = int((processed_count * 100) / total_frames)
                    self.progress_signal.emit(
                        pct,
                        f"Unified precompute: {processed_count}/{total_frames}",
                    )
                    self.stats_signal.emit(
                        {
                            "phase": "unified_precompute",
                            "fps": rate_fps,
                            "elapsed": elapsed,
                            "eta": eta,
                        }
                    )

            if cancelled or self._stop_requested:
                logger.info("Unified precompute cancelled.")
                self.progress_signal.emit(0, "Precompute cancelled.")
                return None, False, None, False

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

            if appearance_cache_writer:
                appearance_cache_writer.save(
                    metadata={
                        "model_name": appearance_config.model_name,
                        "embedding_dimension": embedding_dim,
                        "device": appearance_config.device,
                        "batch_size": appearance_config.batch_size,
                        "max_image_side": appearance_config.max_image_side,
                        "use_clahe": appearance_config.use_clahe,
                        "normalize_embeddings": appearance_config.normalize_embeddings,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "video_path": str(Path(self.video_path).expanduser().resolve()),
                        "detection_cache_path": str(self.detection_cache_path),
                    }
                )
                logger.info(
                    "Appearance embedding cache saved: %s", str(appearance_cache_path)
                )

            self.progress_signal.emit(
                100,
                "Unified precompute complete: pose and appearance caches saved.",
            )

        finally:
            if pose_cache_writer:
                pose_cache_writer.close()
            if appearance_cache_writer:
                appearance_cache_writer.close()
            video_cap.release()
            if pose_backend:
                try:
                    pose_backend.close()
                except Exception:
                    pass
            if appearance_backend:
                try:
                    appearance_backend.close()
                except Exception:
                    pass

        return (
            str(pose_cache_path) if pose_cache_path else None,
            not pose_enabled or pose_cache_hit,
            str(appearance_cache_path) if appearance_cache_path else None,
            not appearance_enabled or appearance_cache_hit,
        )

    def _precompute_individual_properties(
        self,
        detector,
        params,
        detection_cache,
        start_frame,
        end_frame,
    ):
        """
        Precompute and cache per-detection individual properties for filtered detections.

        Returns:
            (cache_path, cache_hit)
        """
        from multi_tracker.core.identity.properties_cache import (
            IndividualPropertiesCache,
            compute_detection_hash,
            compute_extractor_hash,
            compute_filter_settings_hash,
            compute_individual_properties_id,
        )

        pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
        if not pose_enabled:
            logger.info(
                "Individual pipeline enabled but no extractor active; skipping properties precompute."
            )
            return None, True

        if params.get("DETECTION_METHOD", "background_subtraction") != "yolo_obb":
            raise RuntimeError(
                "Individual properties precompute requires YOLO OBB detection mode."
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
        cache_path = self._build_individual_properties_cache_path(
            properties_id, start_frame, end_frame
        )
        self.individual_properties_id = str(properties_id)
        self.individual_properties_cache_path = str(cache_path)
        params["INDIVIDUAL_PROPERTIES_ID"] = properties_id
        params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(cache_path)
        if self._stop_requested:
            return "", False

        if cache_path.exists():
            existing = IndividualPropertiesCache(str(cache_path), mode="r")
            try:
                if existing.is_compatible():
                    meta = existing.metadata or {}
                    if (
                        str(meta.get("individual_properties_id", "")) == properties_id
                        and int(meta.get("start_frame", -1)) == int(start_frame)
                        and int(meta.get("end_frame", -1)) == int(end_frame)
                    ):
                        logger.info(
                            "Individual properties cache hit: %s", str(cache_path)
                        )
                        self.progress_signal.emit(
                            100,
                            "Individual precompute cache hit: reusing existing properties.",
                        )
                        return str(cache_path), True
            finally:
                existing.close()

        logger.info("=" * 80)
        logger.info("PRECOMPUTE: Individual properties for filtered detections")
        logger.info("=" * 80)
        self.progress_signal.emit(
            0,
            "Precompute: initializing pose runtime and validating model...",
        )

        pose_model_path = str(params.get("POSE_MODEL_DIR", "")).strip()
        if not pose_model_path:
            raise RuntimeError("Pose extractor is enabled but model path is empty.")
        pose_model_file = Path(pose_model_path).expanduser().resolve()
        if not pose_model_file.exists():
            raise RuntimeError(f"Pose model not found: {pose_model_file}")

        from multi_tracker.core.identity.runtime_api import (
            build_runtime_config,
            create_pose_backend_from_config,
        )

        runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "auto")).strip().lower()
        backend_family = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
        runtime_device = (
            str(params.get("POSE_SLEAP_DEVICE", "auto")).strip()
            if backend_family == "sleap"
            else str(params.get("YOLO_DEVICE", "auto")).strip()
        )
        self.progress_signal.emit(
            0,
            "Precompute: loading pose backend "
            f"({backend_family}/{runtime_flavor} on {runtime_device})...",
        )
        # Use individual dataset output directory if available, otherwise fall back to cache directory
        pose_out_root = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
        if not pose_out_root:
            pose_out_root = str(cache_path.parent)
        pose_config = build_runtime_config(params, out_root=pose_out_root)
        pose_backend = create_pose_backend_from_config(pose_config)
        pose_backend.warmup()
        if runtime_flavor.startswith("onnx") or runtime_flavor.startswith("tensorrt"):
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
        self.progress_signal.emit(
            1,
            "Precompute: pose backend ready, reading frames...",
        )

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

        pose_cap = cv2.VideoCapture(self.video_path)
        if not pose_cap.isOpened():
            raise RuntimeError(
                f"Failed to open video for pose precompute: {self.video_path}"
            )
        if start_frame > 0:
            pose_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        cache_writer = IndividualPropertiesCache(str(cache_path), mode="w")
        total_frames = max(1, end_frame - start_frame + 1)
        precompute_start_ts = time.time()
        cancelled = False
        self.progress_signal.emit(
            1,
            f"Precompute: processing {total_frames} frame(s) of filtered detections...",
        )
        try:
            for rel_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if self._stop_requested:
                    cancelled = True
                    break
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                ) = detection_cache.get_frame(frame_idx)

                (
                    meas,
                    _sizes,
                    _shapes,
                    _confs,
                    filtered_obb_corners,
                    detection_ids,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=roi_mask,
                    detection_ids=raw_detection_ids,
                )

                ret, frame = pose_cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Failed to read frame {frame_idx} for individual precompute."
                    )
                if resize_f < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                pose_keypoints = []

                if meas and filtered_obb_corners:
                    crops = []
                    crop_to_det = []
                    pose_outputs = [{} for _ in range(len(meas))]
                    crop_offsets = {}
                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            crops.append(crop)
                            crop_to_det.append(det_idx)
                            crop_offsets[det_idx] = crop_offset

                    if crops:
                        if self._stop_requested:
                            cancelled = True
                            break
                        pred_outputs = pose_backend.predict_batch(crops)
                        for crop_idx, det_idx in enumerate(crop_to_det):
                            if crop_idx < len(pred_outputs):
                                out = pred_outputs[crop_idx]
                                crop_offset = crop_offsets.get(det_idx)
                                kpts = out.keypoints
                                if (
                                    kpts is not None
                                    and crop_offset is not None
                                    and len(kpts) > 0
                                ):
                                    x0, y0 = crop_offset
                                    gkpts = np.asarray(kpts, dtype=np.float32).copy()
                                    gkpts[:, 0] += float(x0)
                                    gkpts[:, 1] += float(y0)
                                else:
                                    gkpts = kpts
                                pose_outputs[det_idx] = {
                                    "keypoints": gkpts,
                                }

                    for det_idx in range(len(meas)):
                        out = pose_outputs[det_idx]
                        pose_keypoints.append(out.get("keypoints", None))

                # Only store raw keypoints; summary stats computed on-demand when reading
                cache_writer.add_frame(
                    frame_idx,
                    detection_ids,
                    pose_keypoints=pose_keypoints,
                )

                processed_count = rel_idx + 1
                if rel_idx % 10 == 0 or rel_idx == total_frames - 1:
                    elapsed = max(1e-6, time.time() - precompute_start_ts)
                    rate_fps = processed_count / elapsed
                    remaining = max(0, total_frames - processed_count)
                    eta = (remaining / rate_fps) if rate_fps > 1e-9 else 0.0
                    pct = int((processed_count * 100) / total_frames)
                    self.progress_signal.emit(
                        pct,
                        "Precompute individual properties: "
                        f"{processed_count}/{total_frames} ",
                    )
                    self.stats_signal.emit(
                        {
                            "phase": "individual_precompute",
                            "fps": rate_fps,
                            "elapsed": elapsed,
                            "eta": eta,
                        }
                    )

            if cancelled or self._stop_requested:
                logger.info("Individual properties precompute cancelled.")
                self.progress_signal.emit(0, "Precompute cancelled.")
                return "", False

            cache_writer.save(
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
            logger.info("Individual properties cache saved: %s", str(cache_path))
            self.progress_signal.emit(
                100,
                "Precompute complete: individual properties cache saved.",
            )
        finally:
            cache_writer.close()
            pose_cap.release()
            try:
                pose_backend.close()
            except Exception:
                pass

        return str(cache_path), False

    def _precompute_appearance_embeddings(
        self,
        detector,
        params,
        detection_cache,
        start_frame,
        end_frame,
    ):
        """
        Precompute and cache appearance embeddings for filtered detections.

        Returns:
            (cache_path, cache_hit)
        """
        from multi_tracker.core.identity.appearance_cache import (
            AppearanceEmbeddingCache,
            build_appearance_cache_path,
            compute_appearance_embedding_id,
            compute_appearance_extractor_hash,
        )
        from multi_tracker.core.identity.properties_cache import (
            compute_detection_hash,
            compute_filter_settings_hash,
        )
        from multi_tracker.core.identity.runtime_api import (
            AppearanceRuntimeConfig,
            create_appearance_backend_from_config,
        )

        appearance_enabled = bool(params.get("APPEARANCE_ENABLED", False))
        if not appearance_enabled:
            logger.info("Appearance embedding extraction is disabled; skipping.")
            return None, True

        if params.get("DETECTION_METHOD", "background_subtraction") != "yolo_obb":
            logger.warning(
                "Appearance embedding extraction requires YOLO OBB detection mode; skipping."
            )
            return None, True

        # Compute cache ID
        detection_hash = compute_detection_hash(
            params.get("INFERENCE_MODEL_ID", ""),
            self.video_path,
            start_frame,
            end_frame,
            detection_cache_version="2.0",
        )
        filter_hash = compute_filter_settings_hash(params)
        appearance_hash = compute_appearance_extractor_hash(params)
        embedding_id = compute_appearance_embedding_id(
            detection_hash, filter_hash, appearance_hash
        )

        video_stem = Path(self.video_path).stem
        output_dir = params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "").strip()
        if not output_dir:
            output_dir = str(Path(self.video_path).parent / "appearance_embeddings")

        cache_path = build_appearance_cache_path(
            output_dir, video_stem, embedding_id, start_frame, end_frame
        )

        if self._stop_requested:
            return "", False

        # Check for existing cache
        if cache_path.exists():
            existing = AppearanceEmbeddingCache(str(cache_path), mode="r")
            try:
                if existing.is_compatible():
                    meta = existing.metadata or {}
                    if (
                        str(meta.get("model_name", ""))
                        == params.get("APPEARANCE_MODEL_NAME", "")
                        and int(meta.get("start_frame", -1)) == int(start_frame)
                        and int(meta.get("end_frame", -1)) == int(end_frame)
                    ):
                        logger.info(
                            "Appearance embedding cache hit: %s", str(cache_path)
                        )
                        self.progress_signal.emit(
                            100,
                            "Appearance embedding cache hit: reusing existing embeddings.",
                        )
                        return str(cache_path), True
            finally:
                existing.close()

        logger.info("=" * 80)
        logger.info("PRECOMPUTE: Appearance embeddings for filtered detections")
        logger.info("=" * 80)
        self.progress_signal.emit(
            0,
            "Precompute: initializing appearance embedding runtime...",
        )

        # Create appearance runtime config
        runtime_config = AppearanceRuntimeConfig(
            model_name=params.get(
                "APPEARANCE_MODEL_NAME", "timm/vit_base_patch14_dinov2.lvd142m"
            ),
            runtime_flavor=params.get("APPEARANCE_RUNTIME_FLAVOR", "auto"),
            batch_size=params.get("APPEARANCE_BATCH_SIZE", 32),
            max_image_side=params.get("APPEARANCE_MAX_IMAGE_SIDE", 512),
            use_clahe=params.get("APPEARANCE_USE_CLAHE", False),
            normalize_embeddings=params.get("APPEARANCE_NORMALIZE", True),
            compute_runtime=str(
                params.get("COMPUTE_RUNTIME", params.get("compute_runtime", ""))
            ),
        )

        self.progress_signal.emit(
            0,
            f"Precompute: loading appearance model {runtime_config.model_name}...",
        )
        backend = create_appearance_backend_from_config(runtime_config)
        backend.warmup()

        embedding_dim = backend.output_dimension
        logger.info(f"Appearance model loaded. Embedding dimension: {embedding_dim}")

        self.progress_signal.emit(
            1,
            "Precompute: appearance backend ready, reading frames...",
        )

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

        app_cap = cv2.VideoCapture(self.video_path)
        if not app_cap.isOpened():
            raise RuntimeError(
                f"Failed to open video for appearance precompute: {self.video_path}"
            )
        if start_frame > 0:
            app_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        cache_writer = AppearanceEmbeddingCache(str(cache_path), mode="w")
        total_frames = max(1, end_frame - start_frame + 1)
        precompute_start_ts = time.time()
        total_detections = 0
        total_embeddings_computed = 0
        cancelled = False
        self.progress_signal.emit(
            1,
            f"Precompute: processing {total_frames} frame(s) for appearance embeddings...",
        )

        try:
            for rel_idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
                if self._stop_requested:
                    cancelled = True
                    break

                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                ) = detection_cache.get_frame(frame_idx)

                (
                    meas,
                    _sizes,
                    _shapes,
                    _confs,
                    filtered_obb_corners,
                    detection_ids,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=roi_mask,
                    detection_ids=raw_detection_ids,
                )

                ret, frame = app_cap.read()
                if not ret:
                    logger.warning(
                        f"Failed to read frame {frame_idx} for appearance precompute, adding empty frame"
                    )
                    cache_writer.add_frame(
                        frame_idx, detection_ids or [], embeddings=[]
                    )
                    continue

                if resize_f < 1.0:
                    frame = cv2.resize(
                        frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                embeddings_list = []

                if meas and filtered_obb_corners:
                    total_detections += len(meas)
                    crops = []
                    crop_to_det = []

                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            crops.append(crop)
                            crop_to_det.append(det_idx)

                    embeddings_list = [None] * len(meas)

                    if crops:
                        if self._stop_requested:
                            cancelled = True
                            break

                        total_embeddings_computed += len(crops)
                        appearance_results = backend.predict_crops(crops)

                        for crop_idx, det_idx in enumerate(crop_to_det):
                            if crop_idx < len(appearance_results):
                                result = appearance_results[crop_idx]
                                embeddings_list[det_idx] = result.embedding

                cache_writer.add_frame(
                    frame_idx, detection_ids or [], embeddings=embeddings_list
                )

                processed_count = rel_idx + 1
                if rel_idx % 10 == 0 or rel_idx == total_frames - 1:
                    elapsed = max(1e-6, time.time() - precompute_start_ts)
                    rate_fps = processed_count / elapsed
                    remaining = max(0, total_frames - processed_count)
                    eta = (remaining / rate_fps) if rate_fps > 1e-9 else 0.0
                    pct = int((processed_count * 100) / total_frames)
                    self.progress_signal.emit(
                        pct,
                        f"Precompute appearance embeddings: {processed_count}/{total_frames}",
                    )
                    self.stats_signal.emit(
                        {
                            "phase": "appearance_precompute",
                            "fps": rate_fps,
                            "elapsed": elapsed,
                            "eta": eta,
                        }
                    )

            if cancelled or self._stop_requested:
                logger.info("Appearance embedding precompute cancelled.")
                self.progress_signal.emit(0, "Appearance precompute cancelled.")
                return "", False

            cache_writer.save(
                metadata={
                    "model_name": runtime_config.model_name,
                    "embedding_dimension": embedding_dim,
                    "device": runtime_config.device,
                    "batch_size": runtime_config.batch_size,
                    "max_image_side": runtime_config.max_image_side,
                    "use_clahe": runtime_config.use_clahe,
                    "normalize_embeddings": runtime_config.normalize_embeddings,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "video_path": str(Path(self.video_path).expanduser().resolve()),
                    "detection_cache_path": str(self.detection_cache_path),
                }
            )
            logger.info("Appearance embedding cache saved: %s", str(cache_path))
            self.progress_signal.emit(
                100,
                "Precompute complete: appearance embedding cache saved.",
            )
        finally:
            cache_writer.close()
            app_cap.release()
            try:
                backend.close()
            except Exception:
                pass

        return str(cache_path), False

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

        individual_precompute_enabled = bool(
            individual_pipeline_enabled
            and not self.backward_mode
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and p.get("ENABLE_POSE_EXTRACTOR", False)
        )
        if individual_precompute_enabled and not self.detection_cache_path:
            logger.error(
                "Individual precompute requires detection caching, but no detection cache path is configured."
            )
            cap.release()
            self.finished_signal.emit(False, [], [])
            return

        # Individual precompute needs full raw detections before tracking starts.
        # Force two-phase detection in YOLO mode so precompute can run on cached detections.
        if individual_precompute_enabled and not use_batched_detection:
            use_batched_detection = True
            logger.info(
                "Enabling batched YOLO prepass for individual-properties precompute."
            )

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
        track_states, missed_frames = ["active"] * N, [0] * N
        self.trajectories_full = [[] for _ in range(N)]
        trajectories_pruned = [[] for _ in range(N)]
        position_deques = [deque(maxlen=2) for _ in range(N)]
        orientation_last, last_shape_info = [None] * N, [None] * N
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
                (self.use_cached_detections or individual_precompute_enabled)
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

        if individual_precompute_enabled:
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

            # Unified precompute for both pose and appearance in a single video pass
            try:
                (
                    props_path,
                    props_cache_hit,
                    app_path,
                    app_cache_hit,
                ) = self._precompute_individual_data_unified(
                    detector, p, detection_cache, start_frame, end_frame
                )

                # Log results
                if props_path:
                    state = "hit" if props_cache_hit else "miss"
                    logger.info("Individual properties cache %s: %s", state, props_path)
                if app_path:
                    state = "hit" if app_cache_hit else "miss"
                    logger.info("Appearance embedding cache %s: %s", state, app_path)

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
                pose_direction_anterior_indices = self._resolve_pose_group_indices(
                    p.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), pose_keypoint_names
                )
                pose_direction_posterior_indices = self._resolve_pose_group_indices(
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
            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
                [],
                [],
                [],
                [],
                [],
            )
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
                ) = detection_cache.get_frame(actual_frame_index)

                if detection_method == "yolo_obb":
                    (
                        meas,
                        sizes,
                        shapes,
                        detection_confidences,
                        filtered_obb_corners,
                        detection_ids,
                    ) = detector.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confidences,
                        raw_obb_corners,
                        roi_mask=ROI_mask_current,
                        detection_ids=raw_detection_ids,
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
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=ROI_mask_current,
                    detection_ids=raw_detection_ids,
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
                )
                cached_frame_indices.add(actual_frame_index)

            profile_times["detection"] += time.time() - detect_start

            # Optional pose-based direction override on filtered detections.
            if pose_direction_enabled and meas and detection_ids:
                if pose_frame_keypoints_map_frame != actual_frame_index:
                    pose_frame_keypoints_map = self._build_pose_detection_keypoint_map(
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
                    pose_theta = self._compute_pose_heading_from_keypoints(
                        keypoints,
                        pose_direction_anterior_indices,
                        pose_direction_posterior_indices,
                        pose_min_valid_conf,
                    )
                    if pose_theta is None:
                        continue
                    m = np.asarray(meas[det_idx], dtype=np.float32).copy()
                    if len(m) < 3:
                        continue
                    m[2] = np.float32(pose_theta)
                    meas[det_idx] = m
                    pose_directed_mask[det_idx] = 1

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

            hist_velocities = []
            hist_sizes = []
            hist_orientations = []
            hist_costs = []

            if detection_initialized and meas:
                # --- Assignment ---
                assign_start = time.time()
                preds = self.kf_manager.get_predictions()

                # The Assigner now takes the kf_manager directly to access X and S_inv
                cost, spatial_candidates = assigner.compute_cost_matrix(
                    N,
                    meas,
                    preds,
                    shapes,
                    self.kf_manager,
                    last_shape_info,
                    meas_ori_directed=(
                        pose_directed_mask
                        if len(pose_directed_mask) == len(meas)
                        else None
                    ),
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
                    )
                )
                next_trajectory_id = next_id

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
                avg_cost = 0.0
                for r, c in zip(rows, cols):
                    x = float(meas[c][0])
                    y = float(meas[c][1])
                    measured_theta = float(meas[c][2])
                    pose_directed = bool(
                        c < len(pose_directed_mask) and pose_directed_mask[c] == 1
                    )

                    if pose_directed:
                        theta_for_tracking = self._normalize_theta(measured_theta)
                        pose_direction_applied_count += 1
                    else:
                        reference_theta = orientation_last[r]
                        if reference_theta is None:
                            try:
                                reference_theta = float(preds[r, 2])
                            except Exception:
                                reference_theta = None
                        theta_for_tracking = self._collapse_obb_axis_theta(
                            measured_theta, reference_theta
                        )
                        pose_direction_fallback_count += 1

                    # Vectorized manager handles the reshaping and indexing internally
                    corrected_meas = np.asarray(
                        [x, y, theta_for_tracking], dtype=np.float32
                    )
                    self.kf_manager.correct(r, corrected_meas)

                    tracking_continuity[r] += 1
                    position_deques[r].append((x, y))
                    speed = (
                        math.hypot(
                            position_deques[r][1][0] - position_deques[r][0][0],
                            position_deques[r][1][1] - position_deques[r][0][1],
                        )
                        if len(position_deques[r]) == 2
                        else 0
                    )
                    orientation_last[r] = self._smooth_orientation(
                        r,
                        theta_for_tracking,
                        speed,
                        params,
                        orientation_last,
                        position_deques,
                    )
                    last_shape_info[r] = shapes[c]

                    theta_out = orientation_last[r]
                    # Backward fallback orientation historically needs a 180-degree correction.
                    if self.backward_mode and not pose_directed:
                        theta_out = (theta_out + np.pi) % (2 * np.pi)

                    # Update trajectory with actual frame index
                    pt = (int(x), int(y), theta_out, actual_frame_index)
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
                    avg_cost += current_cost / N

                    # Populate histogram lists (this part is correct)
                    hist_velocities.append(speed)
                    hist_sizes.append(sizes[c])
                    hist_orientations.append(theta_out)
                    hist_costs.append(current_cost)

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
                            self.kf_manager.initialize_filter(
                                track_idx,
                                np.array(
                                    [
                                        meas[d_idx][0],
                                        meas[d_idx][1],
                                        meas[d_idx][2],
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
                            trajectory_ids[track_idx] = next_trajectory_id
                            next_trajectory_id += 1
                            break

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
                    and "cols" in dir()
                    and "rows" in dir()
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

            # Only emit histogram data if the feature is enabled in the GUI
            if params.get("ENABLE_HISTOGRAMS", False):
                histogram_payload = {
                    "velocities": hist_velocities,
                    "sizes": hist_sizes,
                    "orientations": hist_orientations,
                    "assignment_costs": hist_costs,
                }
                self.histogram_data_signal.emit(histogram_payload)

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

        if pose_direction_enabled:
            logger.info(
                "Pose direction summary: applied=%d, fallback=%d",
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
            (x1, y1), (x2, y2) = position_deques[r]
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
