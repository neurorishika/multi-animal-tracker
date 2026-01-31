"""
Core tracking engine running in separate thread for real-time performance.
This is the main orchestrator, functionally identical to the original.
"""

import sys, time, gc, math, logging, os, random
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from collections import deque
from PySide2.QtCore import QThread, Signal, QMutex, Slot

from ..utils.image_processing import apply_image_adjustments, stabilize_lighting
from ..utils.geometry import wrap_angle_degs
from ..utils.detection_cache import DetectionCache
from ..utils.batch_optimizer import BatchOptimizer
from ..utils.frame_prefetcher import FramePrefetcher
from .kalman_filters import KalmanFilterManager
from .background_models import BackgroundModel
from .detection import create_detector
from .assignment import TrackAssigner
from .individual_analysis import IndividualDatasetGenerator

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

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        detection_cache_path=None,
        preview_mode=False,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.video_output_path = video_output_path
        self.backward_mode = backward_mode
        self.detection_cache_path = detection_cache_path
        self.preview_mode = preview_mode
        self.video_writer = None
        self.params_mutex = QMutex()
        self.parameters = {}

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

    def set_parameters(self, p: dict):
        self.params_mutex.lock()
        self.parameters = p
        self.params_mutex.unlock()

    @Slot(dict)
    def update_parameters(self, new_params: dict):
        """Slot to safely update parameters from the GUI thread."""
        self.params_mutex.lock()
        self.parameters = new_params
        self.params_mutex.unlock()
        logger.info("Tracking worker parameters updated.")

    def get_current_params(self):
        self.params_mutex.lock()
        p = dict(self.parameters)
        self.params_mutex.unlock()
        return p

    def stop(self):
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

    def _cached_detection_iterator(self, total_frames):
        """Iterate through frame indices for cached detection mode (no actual frames needed)."""
        # In backward mode with cached detections, we don't need frames at all
        # Just iterate through frame indices in reverse
        for frame_idx in range(total_frames - 1, -1, -1):
            if self._stop_requested:
                break
            yield None, total_frames - frame_idx  # Return None for frame, 1-indexed frame number

    def emit_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb)

    def _run_batched_detection_phase(self, cap, detection_cache, detector, params):
        """
        Phase 1: Run batched YOLO detection on entire video and cache results.

        Args:
            cap: OpenCV VideoCapture object
            detection_cache: DetectionCache instance for writing
            detector: YOLOOBBDetector instance
            params: Configuration parameters

        Returns:
            int: Total frames processed
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: Batched YOLO Detection")
        logger.info("=" * 80)

        # Get batch size using advanced config
        advanced_config = params.get("ADVANCED_CONFIG", {})
        batch_optimizer = BatchOptimizer(advanced_config)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Estimate optimal batch size
        model_name = params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt")
        batch_size = batch_optimizer.estimate_batch_size(
            frame_width, frame_height, model_name
        )

        logger.info(f"Video: {frame_width}x{frame_height}, {total_frames} frames")
        logger.info(f"Batch size: {batch_size}")

        # Process video in batches
        frame_idx = 0
        batch_count = 0
        total_batches = (total_frames + batch_size - 1) // batch_size

        resize_factor = params.get("RESIZE_FACTOR", 1.0)

        while not self._stop_requested:
            # Read a batch of frames
            batch_frames = []
            batch_start_idx = frame_idx

            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
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

            # Run batched detection
            batch_count += 1
            logger.info(
                f"Processing batch {batch_count}/{total_batches} ({len(batch_frames)} frames)"
            )

            # Progress callback for within-batch updates
            def progress_cb(current, total, msg):
                pass  # Could add finer-grained progress here if needed

            batch_results = detector.detect_objects_batched(
                batch_frames, batch_start_idx, progress_cb
            )

            # Cache each frame's detections
            for local_idx, (meas, sizes, shapes, confidences, obb_corners) in enumerate(
                batch_results
            ):
                global_idx = batch_start_idx + local_idx
                detection_cache.add_frame(
                    global_idx, meas, sizes, shapes, confidences, obb_corners
                )

            # Emit progress
            percentage = (
                int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
            )
            status_text = f"Detecting objects: batch {batch_count}/{total_batches} ({percentage}%)"
            self.progress_signal.emit(percentage, status_text)

        logger.info(
            f"Detection phase complete: {frame_idx} frames processed in {batch_count} batches"
        )
        return frame_idx

    def run(self):
        # === 1. INITIALIZATION (Identical to Original) ===
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            self.finished_signal.emit(True, [], [])
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None

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

        # Initialize detector using factory function
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

        # Initialize individual dataset generator (supports both YOLO OBB and BG subtraction)
        individual_generator = None
        if (
            p.get("ENABLE_INDIVIDUAL_DATASET", False)
            and not self.backward_mode  # Only generate dataset in forward pass
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
        assigner = TrackAssigner(p)

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
        if self.detection_cache_path:
            if self.backward_mode:
                # Backward pass: load cached detections
                detection_cache = DetectionCache(self.detection_cache_path, mode="r")
                total_frames = detection_cache.get_total_frames()
                use_cached_detections = True
                logger.info(
                    f"Backward pass using cached detections ({total_frames} frames)"
                )
            else:
                # Forward pass: create cache for writing
                detection_cache = DetectionCache(self.detection_cache_path, mode="w")
                logger.info("Forward pass caching detections")

        # === RUN BATCHED DETECTION PHASE (if applicable) ===
        if use_batched_detection:
            # Phase 1: Batched YOLO detection
            frames_processed = self._run_batched_detection_phase(
                cap, detection_cache, detector, p
            )

            # Save detection cache after phase 1
            detection_cache.save()
            logger.info("Detection cache saved after batched phase")

            # Reopen cache in read mode for phase 2
            detection_cache.close()
            detection_cache = DetectionCache(self.detection_cache_path, mode="r")
            total_frames = frames_processed
            use_cached_detections = True  # Phase 2 uses cached detections

            # Reset video capture for phase 2 (tracking + visualization)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            logger.info("=" * 80)
            logger.info("PHASE 2: Tracking and Visualization")
            logger.info("=" * 80)

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
            if use_batched_detection:
                # Phase 2 of batched detection: use forward iterator for visualization
                # but load detections from cache
                frame_iterator = self._forward_frame_iterator(
                    cap, use_prefetcher=use_prefetcher
                )
                skip_visualization = False  # Show visualization in phase 2
                logger.info("Phase 2: Using cached detections with visualization")
            else:
                # Backward pass: no frames needed, skip visualization
                frame_iterator = self._cached_detection_iterator(total_frames)
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
            ROI_mask_3ch = None
            mask_inv_3ch = None

            if ROI_mask is not None and frame is not None:
                ROI_mask_current = (
                    cv2.resize(
                        ROI_mask, (frame.shape[1], frame.shape[0]), cv2.INTER_NEAREST
                    )
                    if resize_f != 1.0
                    else ROI_mask
                )
                # Calculate fill color on first frame if not yet done
                if roi_fill_color is None:
                    mask_inv = ROI_mask_current == 0
                    outside_pixels = frame[mask_inv]
                    if len(outside_pixels) > 0:
                        roi_fill_color = np.mean(outside_pixels, axis=0).astype(
                            np.uint8
                        )
                    else:
                        roi_fill_color = np.array([0, 0, 0], dtype=np.uint8)

                # Pre-compute 3-channel masks for reuse
                ROI_mask_3ch = cv2.cvtColor(ROI_mask_current, cv2.COLOR_GRAY2BGR)
                mask_inv_3ch = cv2.bitwise_not(ROI_mask_3ch)

            detect_start = time.time()

            # Get detections either from cache or by detection
            if use_cached_detections:
                # Load cached detections (backward pass or phase 2 of batched detection)
                # Frame index is 0-based, but self.frame_count is 1-based during iteration
                if use_batched_detection:
                    # Phase 2: forward iteration, so use frame_count - 1 directly
                    cache_frame_idx = self.frame_count - 1
                else:
                    # Backward pass: reverse mapping
                    cache_frame_idx = total_frames - self.frame_count

                meas, sizes, shapes, detection_confidences, filtered_obb_corners = (
                    detection_cache.get_frame(cache_frame_idx)
                )
                # No yolo_results object in cached mode
                yolo_results = None
                fg_mask = None
                bg_u8 = None

            elif detection_method == "background_subtraction" and frame is not None:
                # Background subtraction detection pipeline
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = apply_image_adjustments(
                    gray, params["BRIGHTNESS"], params["CONTRAST"], params["GAMMA"]
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
                    detector.detect_objects(fg_mask, self.frame_count)
                )
                # No OBB corners for background subtraction
                filtered_obb_corners = []

            elif (
                detection_method == "yolo_obb" and frame is not None
            ):  # YOLO OBB detection
                # YOLO uses the original BGR frame directly without masking
                # This preserves natural image context for better confidence estimates
                yolo_frame = frame.copy()

                # No foreground mask or background for YOLO
                fg_mask = None
                bg_u8 = None
                meas, sizes, shapes, yolo_results, detection_confidences = (
                    detector.detect_objects(yolo_frame, self.frame_count)
                )

                # Get filtered OBB corners from detector (already filtered by size)
                filtered_obb_corners = (
                    getattr(yolo_results, "_filtered_obb_corners", [])
                    if yolo_results
                    else []
                )

                # Filter detections by ROI mask AFTER detection (vectorized)
                # This is better than masking the image which reduces YOLO confidence
                if ROI_mask_current is not None and len(meas) > 0:
                    # Vectorized filtering using NumPy for efficiency with large n
                    meas_arr = np.array(meas)
                    cx_arr = meas_arr[:, 0].astype(np.int32)
                    cy_arr = meas_arr[:, 1].astype(np.int32)

                    # Bounds check
                    h, w = ROI_mask_current.shape[:2]
                    in_bounds = (
                        (cy_arr >= 0) & (cy_arr < h) & (cx_arr >= 0) & (cx_arr < w)
                    )

                    # ROI check (clip to bounds for safe indexing, then apply bounds mask)
                    cy_safe = np.clip(cy_arr, 0, h - 1)
                    cx_safe = np.clip(cx_arr, 0, w - 1)
                    in_roi = ROI_mask_current[cy_safe, cx_safe] > 0

                    # Combined mask
                    keep_mask = in_bounds & in_roi
                    keep_indices = np.where(keep_mask)[0]

                    # Apply filter using boolean indexing
                    # Keep meas as list of numpy arrays (required by Kalman filter)
                    meas = [meas_arr[i] for i in keep_indices]
                    sizes = np.array(sizes)[keep_mask].tolist()
                    shapes = np.array(shapes)[keep_mask].tolist()
                    detection_confidences = np.array(detection_confidences)[
                        keep_mask
                    ].tolist()
                    # Also filter OBB corners
                    filtered_obb_corners = [
                        filtered_obb_corners[i] for i in keep_indices
                    ]

            else:
                # No frame and no cached detections - skip this iteration
                if not use_cached_detections:
                    logger.warning(
                        f"Frame {self.frame_count}: No frame available and no cached detections"
                    )
                    continue

            # Cache detections during forward pass (only when actively detecting, not when loading from cache)
            if detection_cache and not self.backward_mode and not use_cached_detections:
                # Frame index is 0-based (self.frame_count - 1)
                detection_cache.add_frame(
                    self.frame_count - 1,
                    meas,
                    sizes,
                    shapes,
                    detection_confidences,
                    filtered_obb_corners if filtered_obb_corners else None,
                )

            profile_times["detection"] += time.time() - detect_start

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
                    N, meas, preds, shapes, self.kf_manager, last_shape_info
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
                    # Vectorized manager handles the reshaping and indexing internally
                    self.kf_manager.correct(r, meas[c])

                    x, y, theta = meas[c]
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
                        r, theta, speed, params, orientation_last, position_deques
                    )
                    last_shape_info[r] = shapes[c]

                    pt = (int(x), int(y), orientation_last[r], self.frame_count)
                    self.trajectories_full[r].append(pt)
                    trajectories_pruned[r].append(pt)

                    if self.csv_writer_thread:
                        # Build base data row
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

                        self.csv_writer_thread.enqueue(row_data)
                        local_counts[r] += 1
                    current_cost = cost[r, c]
                    avg_cost += current_cost / N

                    # Populate histogram lists (this part is correct)
                    hist_velocities.append(speed)
                    hist_sizes.append(sizes[c])
                    hist_orientations.append(orientation_last[r])
                    hist_costs.append(current_cost)

                # --- CSV for Unmatched & Final Respawn (Identical to Original) ---
                if self.csv_writer_thread:
                    for r in unmatched:
                        last_pos = (
                            self.trajectories_full[r][-1]
                            if self.trajectories_full[r]
                            else (float("nan"),) * 4
                        )
                        # Build base data row
                        row_data = [
                            r,
                            trajectory_ids[r],
                            local_counts[r],
                            last_pos[0],
                            last_pos[1],
                            last_pos[2],
                            self.frame_count,
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

                # Determine which data source to use
                # Use original_frame (full resolution) and coord_scale_factor (1/resize_f)
                coord_scale_factor = 1.0 / resize_f

                if filtered_obb_corners:
                    # YOLO OBB detection - use OBB corners directly
                    individual_generator.process_frame(
                        frame=original_frame,
                        frame_id=self.frame_count,
                        meas=meas,
                        obb_corners=filtered_obb_corners,
                        ellipse_params=None,
                        confidences=(
                            detection_confidences if detection_confidences else None
                        ),
                        track_ids=matched_track_ids if matched_track_ids else None,
                        trajectory_ids=matched_traj_ids if matched_traj_ids else None,
                        coord_scale_factor=coord_scale_factor,
                    )
                elif shapes:
                    # Background subtraction - compute ellipse params from shapes
                    # shapes contains (area, aspect_ratio) tuples
                    # From: area = π * ax1 * ax2 / 4, aspect_ratio = ax1 / ax2
                    # Solve: ax2 = sqrt(4 * area / (π * aspect_ratio))
                    #        ax1 = aspect_ratio * ax2
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
                        frame_id=self.frame_count,
                        meas=meas,
                        obb_corners=None,
                        ellipse_params=ellipse_params,
                        confidences=(
                            detection_confidences if detection_confidences else None
                        ),
                        track_ids=matched_track_ids if matched_track_ids else None,
                        trajectory_ids=matched_traj_ids if matched_traj_ids else None,
                        coord_scale_factor=coord_scale_factor,
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
                # Emit every 100 frames or so to keep GUI responsive
                if self.frame_count % 100 == 0:
                    percentage = int((self.frame_count * 100) / total_frames)
                    status_text = (
                        f"Processing Frame {self.frame_count} / {total_frames}"
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

        # === 3. CLEANUP (Identical to Original) ===
        # Stop frame prefetcher if still running
        if self.frame_prefetcher is not None:
            self.frame_prefetcher.stop()
            self.frame_prefetcher = None

        cap.release()
        if self.video_writer:
            self.video_writer.release()

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
