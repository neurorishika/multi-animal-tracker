"""QThread workers for the parameter optimizer UI.

DetectionCacheBuilderWorker — builds a detection cache for a frame range.
TrackingPreviewWorker — emits preview frames using cached detections.
"""

import logging
import math
from collections import deque
from typing import Any, Dict

import cv2
import numpy as np
from multi_tracker.core.assigners.hungarian import TrackAssigner
from multi_tracker.core.detectors import DetectionFilter
from multi_tracker.core.filters.kalman import KalmanFilterManager
from multi_tracker.core.identity.geometry import (
    build_detection_direction_overrides as _pf_build_direction_overrides,
)
from multi_tracker.core.identity.geometry import normalize_theta as _pf_normalize_theta
from multi_tracker.core.identity.geometry import (
    resolve_detection_tracking_theta as _pf_resolve_detection_tracking_theta,
)
from multi_tracker.core.identity.pose.features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from multi_tracker.core.identity.pose.features import (
    compute_detection_pose_features as _pf_compute_det_features,
)
from multi_tracker.core.identity.pose.features import (
    load_pose_context_from_params as _pf_load_pose_context,
)
from multi_tracker.data.detection_cache import DetectionCache
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class DetectionCacheBuilderWorker(QThread):
    """
    Phase-1-only worker: runs YOLO detection on a frame range and writes a
    DetectionCache.  Does NOT run Kalman tracking, pose precompute, CSV
    writing, interpolation, or any other production-pipeline stage.

    Used by the Parameter Helper to build a minimal detection cache so the
    Bayesian optimizer can run without triggering the full tracking pipeline.
    """

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)  # (success, cache_path)

    def __init__(
        self,
        video_path: str,
        cache_path: str,
        params: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_path = cache_path
        self.params = params.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        import time
        from collections import deque

        from multi_tracker.core.detectors import create_detector
        from multi_tracker.utils.batch_optimizer import BatchOptimizer

        # --- Load detector (YOLO model) ---
        try:
            detector = create_detector(self.params)
        except Exception as e:
            logger.error("DetectionCacheBuilder: could not create detector: %s", e)
            self.finished_signal.emit(False, "")
            return

        cap = cv2.VideoCapture(self.video_path)
        cache = None
        try:
            if not cap.isOpened():
                logger.error(
                    "DetectionCacheBuilder: could not open video: %s", self.video_path
                )
                self.finished_signal.emit(False, "")
                return

            resize_f = self.params.get("RESIZE_FACTOR", 1.0)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_f)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_f)

            advanced = self.params.get("ADVANCED_CONFIG", {}).copy()
            advanced["enable_tensorrt"] = self.params.get("ENABLE_TENSORRT", False)
            advanced["tensorrt_max_batch_size"] = self.params.get(
                "TENSORRT_MAX_BATCH_SIZE", 16
            )
            batch_size = BatchOptimizer(advanced).estimate_batch_size(
                fw, fh, self.params.get("YOLO_MODEL_PATH", "")
            )

            cache = DetectionCache(
                self.cache_path,
                mode="w",
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            total_frames = self.end_frame - self.start_frame + 1
            rel_idx = 0  # 0-based position relative to start_frame
            batch_times: deque = deque(maxlen=30)

            while rel_idx < total_frames and not self._stop_requested:
                batch_start = rel_idx
                batch_frames = []
                while len(batch_frames) < batch_size and rel_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if resize_f != 1.0:
                        frame = cv2.resize(
                            frame,
                            (0, 0),
                            fx=resize_f,
                            fy=resize_f,
                            interpolation=cv2.INTER_AREA,
                        )
                    batch_frames.append(frame)
                    rel_idx += 1

                if not batch_frames:
                    break

                bt0 = time.time()
                results = detector.detect_objects_batched(
                    batch_frames, batch_start, None, return_raw=True
                )
                batch_times.append(time.time() - bt0)

                for local_i, (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_hints,
                    raw_directed,
                ) in enumerate(results):
                    actual_f = self.start_frame + batch_start + local_i
                    det_ids = [actual_f * 10000 + i for i in range(len(raw_meas))]
                    cache.add_frame(
                        actual_f,
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        det_ids,
                        raw_hints,
                        raw_directed,
                    )

                pct = int(rel_idx * 100 / total_frames)
                avg = sum(batch_times) / len(batch_times) if batch_times else 0
                fps = (len(batch_frames) / avg) if avg > 0 else 0
                eta_s = int((total_frames - rel_idx) / fps) if fps > 0 else -1
                eta_str = f"  ETA {eta_s}s" if eta_s >= 0 else ""
                self.progress_signal.emit(
                    pct, f"Building detection cache: {pct}%{eta_str}"
                )

            if self._stop_requested:
                self.progress_signal.emit(0, "Cancelled.")
                self.finished_signal.emit(False, "")
                return

            cache.save()
            cache.close()
            cache = None
            logger.info("DetectionCacheBuilder: cache saved to %s", self.cache_path)
            self.finished_signal.emit(True, self.cache_path)

        except Exception:
            logger.exception("DetectionCacheBuilder error")
            self.finished_signal.emit(False, "")
        finally:
            cap.release()
            if cache is not None:
                try:
                    cache.close()
                except Exception:
                    pass


class TrackingPreviewWorker(QThread):
    """
    Emits visualization frames for previewing optimization results.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cache = DetectionCache(self.detection_cache_path, mode="r")
        try:
            if not cap.isOpened():
                logger.error("PreviewWorker: could not open video: %s", self.video_path)
                return
            if not cache.is_compatible():
                logger.error("PreviewWorker: incompatible detection cache.")
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            kf_manager = KalmanFilterManager(self.params["MAX_TARGETS"], self.params)
            assigner = TrackAssigner(self.params)
            det_filter = DetectionFilter(self.params)
            _roi_mask = self.params.get("ROI_MASK", None)

            N = self.params["MAX_TARGETS"]

            # --- Pose context ---
            (
                _pose_cache,
                _pose_anterior,
                _pose_posterior,
                _pose_ignore,
                _pose_enabled,
            ) = _pf_load_pose_context(self.params)
            _pose_min_conf = float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
            _pose_kpt_map: dict = {}
            _pose_kpt_map_frame = None
            track_pose_prototypes: list = [None] * N
            track_states, tracking_continuity = ["lost"] * N, [0] * N
            missed_frames = [0] * N
            trajectory_ids, next_trajectory_id = list(range(N)), N
            orientation_last: list = [None] * N
            last_shape_info = [None] * N
            lost_threshold = self.params.get("LOST_THRESHOLD_FRAMES", 5)

            resize_f = self.params.get("RESIZE_FACTOR", 1.0)

            traj_colors = self.params.get("TRAJECTORY_COLORS", [])
            if not traj_colors:
                np.random.seed(42)
                traj_colors = [
                    tuple(int(c) for c in row)
                    for row in np.random.randint(0, 255, (max(N, 32), 3))
                ]

            _TRAIL_LEN = int(self.params.get("TRAJECTORY_HISTORY_SECONDS", 5))
            trail: list[deque] = [deque(maxlen=max(_TRAIL_LEN, 10)) for _ in range(N)]

            show_circles = self.params.get("SHOW_CIRCLES", True)
            show_orientation = self.params.get("SHOW_ORIENTATION", True)
            show_trails = self.params.get("SHOW_TRAJECTORIES", True)
            show_labels = self.params.get("SHOW_LABELS", True)

            for f_idx in range(self.start_frame, self.end_frame + 1):
                if self._stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_det_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                    _raw_canonical_affines,
                    _raw_canvas_dims,
                    _raw_M_inverse,
                ) = cache.get_frame(f_idx)
                if raw_heading_hints:
                    filtered = det_filter.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        roi_mask=_roi_mask,
                        detection_ids=raw_det_ids,
                        heading_hints=raw_heading_hints,
                        directed_mask=raw_directed_mask,
                    )
                    (
                        meas,
                        _,
                        shapes,
                        _confs,
                        _obb_out,
                        detection_ids,
                        _headtail_hints,
                        _headtail_directed,
                    ) = filtered
                else:
                    filtered = det_filter.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        roi_mask=_roi_mask,
                        detection_ids=raw_det_ids,
                    )
                    meas, _, shapes, _confs, _obb_out, detection_ids = filtered
                    _headtail_hints, _headtail_directed = [], []

                kf_manager.predict()

                # --- Per-frame pose features ---
                _det_pose_kpts: list = [None] * len(meas)
                _det_pose_vis = np.zeros(len(meas), dtype=np.float32)
                _det_pose_headings: list = [None] * len(meas)
                if _pose_enabled and meas and detection_ids:
                    if _pose_kpt_map_frame != f_idx:
                        _pose_kpt_map = _pf_build_keypoint_map(_pose_cache, f_idx)
                        _pose_kpt_map_frame = f_idx
                    _det_pose_kpts, _det_pose_vis, _det_pose_headings = (
                        _pf_compute_det_features(
                            [int(d) for d in detection_ids],
                            _pose_kpt_map,
                            _pose_anterior,
                            _pose_posterior,
                            _pose_ignore,
                            _pose_min_conf,
                            return_headings=True,
                        )
                    )
                detection_directed_heading, detection_directed_mask = (
                    _pf_build_direction_overrides(
                        len(meas),
                        _det_pose_headings,
                        [heading is not None for heading in _det_pose_headings],
                        _headtail_hints,
                        _headtail_directed,
                        pose_overrides_headtail=bool(
                            self.params.get("POSE_OVERRIDES_HEADTAIL", True)
                        ),
                    )
                )
                _association_data: dict = {
                    "detection_pose_heading": detection_directed_heading,
                    "detection_pose_keypoints": _det_pose_kpts,
                    "detection_pose_visibility": _det_pose_vis,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": np.zeros(N, dtype=np.float32),
                }

                if meas:
                    cost, _ = assigner.compute_cost_matrix(
                        N,
                        meas,
                        kf_manager.X,
                        shapes,
                        kf_manager,
                        last_shape_info,
                        meas_ori_directed=(
                            detection_directed_mask
                            if len(detection_directed_mask) == len(meas)
                            else None
                        ),
                        association_data=_association_data,
                    )
                    matched_r, matched_c, free_dets, next_trajectory_id, _ = (
                        assigner.assign_tracks(
                            cost,
                            N,
                            len(meas),
                            meas,
                            track_states,
                            tracking_continuity,
                            kf_manager,
                            trajectory_ids,
                            next_trajectory_id,
                        )
                    )
                    for r, c in zip(matched_r, matched_c):
                        m = np.asarray(meas[c], dtype=np.float32)
                        _pose_d = (
                            bool(detection_directed_mask[c])
                            if c < len(detection_directed_mask)
                            else False
                        )
                        theta_cor = _pf_resolve_detection_tracking_theta(
                            r,
                            float(m[2]),
                            (
                                detection_directed_heading[c]
                                if c < len(detection_directed_heading)
                                else math.nan
                            ),
                            _pose_d,
                            orientation_last,
                        )
                        m_cor = np.array([m[0], m[1], theta_cor], dtype=np.float32)
                        if track_states[r] == "lost":
                            trail[r].clear()
                            kf_manager.initialize_filter(
                                r,
                                np.array(
                                    [m_cor[0], m_cor[1], theta_cor, 0.0, 0.0],
                                    dtype=np.float32,
                                ),
                            )
                        kf_manager.correct(r, m_cor)
                        orientation_last[r] = _pf_normalize_theta(
                            float(kf_manager.X[r, 2])
                        )

                    for r, c in zip(matched_r, matched_c):
                        proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
                        if proto is not None:
                            track_pose_prototypes[r] = np.asarray(
                                proto, dtype=np.float32
                            ).copy()

                    newly_initialized: set = set()
                    existing_matched = set(matched_r)
                    for d_idx in free_dets:
                        for r in range(N):
                            if (
                                r not in existing_matched | newly_initialized
                                and track_states[r] == "lost"
                            ):
                                m = np.asarray(meas[d_idx], dtype=np.float32)
                                _pose_d = (
                                    bool(detection_directed_mask[d_idx])
                                    if d_idx < len(detection_directed_mask)
                                    else False
                                )
                                theta_cor = _pf_resolve_detection_tracking_theta(
                                    r,
                                    float(m[2]),
                                    (
                                        detection_directed_heading[d_idx]
                                        if d_idx < len(detection_directed_heading)
                                        else math.nan
                                    ),
                                    _pose_d,
                                    orientation_last,
                                )
                                kf_manager.initialize_filter(
                                    r,
                                    np.array(
                                        [m[0], m[1], theta_cor, 0.0, 0.0],
                                        dtype=np.float32,
                                    ),
                                )
                                trail[r].clear()
                                orientation_last[r] = _pf_normalize_theta(theta_cor)
                                track_states[r] = "active"
                                missed_frames[r] = 0
                                tracking_continuity[r] = 0
                                trajectory_ids[r] = next_trajectory_id
                                next_trajectory_id += 1
                                newly_initialized.add(r)
                                proto = (
                                    _det_pose_kpts[d_idx]
                                    if d_idx < len(_det_pose_kpts)
                                    else None
                                )
                                if proto is not None:
                                    track_pose_prototypes[r] = np.asarray(
                                        proto, dtype=np.float32
                                    ).copy()
                                break
                else:
                    matched_r, matched_c, newly_initialized = [], [], set()

                # --- State management ---
                matched_r_set = set(matched_r) | newly_initialized
                for r in matched_r:
                    missed_frames[r] = 0
                    track_states[r] = "active"
                    tracking_continuity[r] += 1
                for r in range(N):
                    if r not in matched_r_set and track_states[r] != "lost":
                        missed_frames[r] += 1
                        if missed_frames[r] >= lost_threshold:
                            track_states[r] = "lost"
                            tracking_continuity[r] = 0
                        else:
                            track_states[r] = "occluded"
                for r, c in zip(matched_r, matched_c):
                    last_shape_info[r] = shapes[c]

                # Update trails
                for r in range(N):
                    if track_states[r] != "lost":
                        x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
                        if math.isfinite(x) and math.isfinite(y):
                            trail[r].append((int(x), int(y)))
                    else:
                        trail[r].clear()

                display = cv2.resize(frame, (0, 0), fx=resize_f, fy=resize_f)

                for r in range(N):
                    if track_states[r] == "lost":
                        continue
                    col = traj_colors[r % len(traj_colors)]

                    x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
                    theta = float(kf_manager.X[r, 2])
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue

                    pt = (int(x), int(y))

                    if show_trails and len(trail[r]) > 1:
                        pts = np.array(list(trail[r]), dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(
                            display, [pts], isClosed=False, color=col, thickness=2
                        )

                    if show_circles:
                        cv2.circle(display, pt, 7, col, -1)

                    if show_orientation:
                        ex = int(x + 18 * math.cos(theta))
                        ey = int(y + 18 * math.sin(theta))
                        cv2.arrowedLine(display, pt, (ex, ey), col, 2, tipLength=0.4)

                    if show_labels:
                        state_tag = (
                            ""
                            if track_states[r] == "active"
                            else f" ({track_states[r]})"
                        )
                        cv2.putText(
                            display,
                            f"T{trajectory_ids[r]}{state_tag}",
                            (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            col,
                            1,
                            cv2.LINE_AA,
                        )

                self.frame_signal.emit(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                self.msleep(20)
        except Exception:
            logger.exception("PreviewWorker encountered an error.")
        finally:
            cap.release()
            cache.close()
            if _pose_cache is not None:
                try:
                    _pose_cache.close()
                except Exception:
                    pass
            self.finished_signal.emit()
