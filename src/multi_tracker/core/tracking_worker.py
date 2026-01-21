"""
Core tracking engine running in separate thread for real-time performance.
This is the main orchestrator, functionally identical to the original.
"""

import sys, time, gc, math, logging, os, random
import numpy as np
import cv2
from collections import deque
from PySide2.QtCore import QThread, Signal, QMutex, Slot

from ..utils.image_processing import apply_image_adjustments, stabilize_lighting
from ..utils.geometry import wrap_angle_degs
from .kalman_filters import KalmanFilterManager
from .background_models import BackgroundModel
from .detection import create_detector
from .assignment import TrackAssigner

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

    def __init__(
        self,
        video_path,
        csv_writer_thread=None,
        video_output_path=None,
        backward_mode=False,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.csv_writer_thread = csv_writer_thread
        self.video_output_path = video_output_path
        self.backward_mode = backward_mode
        self.video_writer = None
        self.params_mutex = QMutex()
        self.parameters = {}
        self._stop_requested = False

        # Internal state variables that helper methods depend on
        self.frame_count = 0
        self.trajectories_full = []

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

    def _forward_frame_iterator(self, cap):
        frame_num = 0
        while not self._stop_requested:
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            yield frame, frame_num

    def emit_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.frame_signal.emit(rgb)

    def run(self):
        # === 1. INITIALIZATION (Identical to Original) ===
        gc.collect()
        self._stop_requested = False
        p = self.get_current_params()

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
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

        # Initialize background model only if using background subtraction
        bg_model = None
        detection_method = p.get("DETECTION_METHOD", "background_subtraction")
        if detection_method == "background_subtraction":
            bg_model = BackgroundModel(p)
            bg_model.prime_background(cap)

        kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)
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

        # === 2. FRAME PROCESSING LOOP (Identical Flow to Original) ===
        for frame, _ in self._forward_frame_iterator(cap):
            params = self.get_current_params()
            self.frame_count += 1

            # --- Preprocessing & Detection ---
            resize_f = params["RESIZE_FACTOR"]
            if resize_f < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_f,
                    fy=resize_f,
                    interpolation=cv2.INTER_AREA,
                )

            detection_method = params.get("DETECTION_METHOD", "background_subtraction")

            if detection_method == "background_subtraction":
                # Background subtraction detection pipeline
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = apply_image_adjustments(
                    gray, params["BRIGHTNESS"], params["CONTRAST"], params["GAMMA"]
                )

                ROI_mask = params.get("ROI_MASK", None)
                ROI_mask_current = None
                if ROI_mask is not None:
                    ROI_mask_current = (
                        cv2.resize(
                            ROI_mask, (gray.shape[1], gray.shape[0]), cv2.INTER_NEAREST
                        )
                        if resize_f != 1.0
                        else ROI_mask
                    )
                    gray = cv2.bitwise_and(gray, gray, mask=ROI_mask_current)

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
                    self.emit_frame(frame)
                    continue

                fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)
                if detection_initialized and params.get(
                    "ENABLE_CONSERVATIVE_SPLIT", True
                ):
                    fg_mask = detector.apply_conservative_split(fg_mask)
                meas, sizes, shapes = detector.detect_objects(fg_mask, self.frame_count)

            else:  # YOLO OBB detection
                # YOLO uses the original BGR frame directly
                # Apply ROI mask to frame before YOLO detection (mask out areas outside ROI)
                ROI_mask = params.get("ROI_MASK", None)
                yolo_frame = frame.copy()
                if ROI_mask is not None:
                    ROI_mask_current = (
                        cv2.resize(
                            ROI_mask,
                            (frame.shape[1], frame.shape[0]),
                            cv2.INTER_NEAREST,
                        )
                        if resize_f != 1.0
                        else ROI_mask
                    )
                    # Create a 3-channel mask for BGR frame
                    ROI_mask_3ch = cv2.cvtColor(ROI_mask_current, cv2.COLOR_GRAY2BGR)
                    # Apply mask: areas outside ROI become black
                    yolo_frame = cv2.bitwise_and(yolo_frame, ROI_mask_3ch)

                # No foreground mask or background for YOLO
                fg_mask = None
                bg_u8 = None
                meas, sizes, shapes = detector.detect_objects(
                    yolo_frame, self.frame_count
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

            overlay = frame.copy()

            hist_velocities = []
            hist_sizes = []
            hist_orientations = []
            hist_costs = []

            if detection_initialized and meas:
                # --- Assignment ---
                preds = kf_manager.get_predictions()
                cost = assigner.compute_cost_matrix(
                    N, meas, preds, shapes, kf_manager.filters, last_shape_info
                )
                rows, cols, free_dets, next_id, high_cost_tracks = (
                    assigner.assign_tracks(
                        cost,
                        N,
                        len(meas),
                        meas,
                        track_states,
                        tracking_continuity,
                        kf_manager.filters,
                        trajectory_ids,
                        next_trajectory_id,
                    )
                )
                next_trajectory_id = next_id

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
                    kf_manager.filters[r].correct(meas[c].reshape(3, 1))
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
                        self.csv_writer_thread.enqueue(
                            [
                                r,
                                trajectory_ids[r],
                                local_counts[r],
                                pt[0],
                                pt[1],
                                pt[2],
                                pt[3],
                                track_states[r],
                            ]
                        )
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
                        self.csv_writer_thread.enqueue(
                            [
                                r,
                                trajectory_ids[r],
                                local_counts[r],
                                last_pos[0],
                                last_pos[1],
                                last_pos[2],
                                self.frame_count,
                                track_states[r],
                            ]
                        )
                        local_counts[r] += 1

                for d_idx in free_dets:
                    for track_idx in range(N):
                        if track_states[track_idx] == "lost":
                            kf_manager.initialize_filter(
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
            trajectories_pruned = [
                [
                    pt
                    for pt in tr
                    if self.frame_count - pt[3] <= params["TRAJECTORY_HISTORY_SECONDS"]
                ]
                for tr in trajectories_pruned
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
            )
            if self.video_writer:
                self.video_writer.write(overlay)
            self.emit_frame(overlay)

            elapsed = time.time() - start_time
            if elapsed > 0:
                fps_list.append(self.frame_count / elapsed)

        # === 3. CLEANUP (Identical to Original) ===
        cap.release()
        if self.video_writer:
            self.video_writer.release()

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

    def _draw_overlays(
        self, overlay, p, trajectories, track_states, ids, continuity, fg, bg
    ):
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
