#!/usr/bin/env python3
"""
Main application window for the Multi-Animal-Tracker.

Refactored for improved UX with Tabbed interface and logical grouping.
"""

import csv
import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.identity.analysis import IndividualDatasetGenerator
from ..core.identity.properties_export import (
    POSE_SUMMARY_COLUMNS,
    augment_trajectories_with_pose_cache,
    build_pose_keypoint_labels,
    flatten_pose_keypoints_row,
    merge_interpolated_pose_df,
    pose_wide_columns_for_labels,
)
from ..core.post.processing import (
    interpolate_trajectories,
    process_trajectories,
    resolve_trajectories,
)
from ..core.runtime.compute_runtime import (
    CANONICAL_RUNTIMES,
    allowed_runtimes_for_pipelines,
    derive_detection_runtime_settings,
    derive_pose_runtime_settings,
    infer_compute_runtime_from_legacy,
    runtime_label,
)
from ..core.tracking.worker import TrackingWorker
from ..data.csv_writer import CSVWriterThread
from ..data.detection_cache import DetectionCache
from ..utils.geometry import fit_circle_to_points, wrap_angle_degs
from ..utils.gpu_utils import (
    MPS_AVAILABLE,
    ROCM_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)
from .dialogs.train_yolo_dialog import TrainYoloDialog
from .widgets.histograms import HistogramPanel

# Configuration file for saving/loading tracking parameters
CONFIG_FILENAME = "tracking_config.json"  # Fallback for manual load/save


class MergeWorker(QThread):
    """Worker thread for merging trajectories without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(object)  # merged trajectories
    error_signal = Signal(str)  # error message

    def __init__(
        self,
        forward_trajs,
        backward_trajs,
        total_frames,
        params,
        resize_factor,
        interp_method,
        max_gap,
    ):
        super().__init__()
        self.forward_trajs = forward_trajs
        self.backward_trajs = backward_trajs
        self.total_frames = total_frames
        self.params = params
        self.resize_factor = resize_factor
        self.interp_method = interp_method
        self.max_gap = max_gap
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def run(self: object) -> object:
        """run method documentation."""
        # pose_backend = None
        try:
            if self._should_stop():
                return
            self.progress_signal.emit(10, "Preparing trajectories...")

            # Convert DataFrames to list of DataFrames (one per trajectory)
            def prepare_trajs_for_merge(trajs):
                if isinstance(trajs, pd.DataFrame):
                    return [group for _, group in trajs.groupby("TrajectoryID")]
                else:
                    return trajs

            forward_prepared = prepare_trajs_for_merge(self.forward_trajs)
            backward_prepared = prepare_trajs_for_merge(self.backward_trajs)

            if self._should_stop():
                return
            self.progress_signal.emit(30, "Resolving trajectory conflicts...")

            resolved_trajectories = resolve_trajectories(
                forward_prepared,
                backward_prepared,
                params=self.params,
            )

            if self._should_stop():
                return
            self.progress_signal.emit(60, "Converting to DataFrame...")

            # Convert resolved trajectories to DataFrame
            if resolved_trajectories and isinstance(resolved_trajectories, list):
                if isinstance(resolved_trajectories[0], pd.DataFrame):
                    # Reassign TrajectoryID to ensure unique IDs
                    for new_id, traj_df in enumerate(resolved_trajectories):
                        traj_df["TrajectoryID"] = new_id
                    resolved_trajectories = pd.concat(
                        resolved_trajectories, ignore_index=True
                    )
                else:
                    # Fallback for old tuple format
                    logger.warning(
                        "Received tuple format from resolve_trajectories, converting..."
                    )
                    all_data = []
                    for traj_id, traj in enumerate(resolved_trajectories):
                        for x, y, theta, frame in traj:
                            all_data.append(
                                {
                                    "TrajectoryID": traj_id,
                                    "X": x,
                                    "Y": y,
                                    "Theta": theta,
                                    "FrameID": frame,
                                }
                            )
                    if all_data:
                        resolved_trajectories = pd.DataFrame(all_data)
                    else:
                        resolved_trajectories = []

            self.progress_signal.emit(75, "Applying interpolation...")

            # Apply interpolation if enabled
            if isinstance(resolved_trajectories, pd.DataFrame):
                if self.interp_method != "none":
                    resolved_trajectories = interpolate_trajectories(
                        resolved_trajectories,
                        method=self.interp_method,
                        max_gap=self.max_gap,
                    )

            if self._should_stop():
                return
            self.progress_signal.emit(90, "Scaling to original space...")

            # Scale coordinates back to original video space
            if isinstance(resolved_trajectories, pd.DataFrame):
                # Log pre-scaling ranges for debugging
                logger.info(
                    f"Pre-scaling (resize_factor={self.resize_factor:.3f}): "
                    f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
                    f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
                )

                resolved_trajectories[["X", "Y"]] = (
                    resolved_trajectories[["X", "Y"]] / self.resize_factor
                )
                if "Width" in resolved_trajectories.columns:
                    resolved_trajectories["Width"] /= self.resize_factor
                if "Height" in resolved_trajectories.columns:
                    resolved_trajectories["Height"] /= self.resize_factor

                # Log post-scaling ranges for debugging
                logger.info(
                    f"Post-scaling: "
                    f"X range [{resolved_trajectories['X'].min():.1f}, {resolved_trajectories['X'].max():.1f}], "
                    f"Y range [{resolved_trajectories['Y'].min():.1f}, {resolved_trajectories['Y'].max():.1f}]"
                )

            if not self._should_stop():
                self.progress_signal.emit(100, "Merge complete!")
                self.finished_signal.emit(resolved_trajectories)

        except Exception as e:
            logger.exception("Error during trajectory merging")
            self.error_signal.emit(str(e))


class InterpolatedCropsWorker(QThread):
    """Worker thread for interpolating occluded crops without blocking the UI."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(dict)

    def __init__(self, csv_path, video_path, detection_cache_path, params):
        super().__init__()
        self.csv_path = csv_path
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.params = params
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    @staticmethod
    def _interp_angle(theta_start, theta_end, t):
        deg0 = math.degrees(theta_start)
        deg1 = math.degrees(theta_end)
        candidates = (deg1, deg1 + 180.0, deg1 - 180.0)
        best_delta = None
        for cand in candidates:
            delta = wrap_angle_degs(cand - deg0)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        return math.radians(deg0 + (best_delta or 0.0) * t)

    @staticmethod
    def _get_detection_size(detection_cache, frame_id, detection_id):
        if detection_cache is None or detection_id is None or pd.isna(detection_id):
            return None, None
        try:
            _, _, shapes, _, obb_corners, detection_ids = detection_cache.get_frame(
                int(frame_id)
            )
        except Exception:
            return None, None

        idx = None
        try:
            for i, did in enumerate(detection_ids):
                if int(did) == int(detection_id):
                    idx = i
                    break
        except Exception:
            idx = None

        if idx is None:
            return None, None

        if obb_corners and idx < len(obb_corners):
            c = np.asarray(obb_corners[idx], dtype=np.float32)
            if c.shape[0] >= 4:
                w = float(np.linalg.norm(c[1] - c[0]))
                h = float(np.linalg.norm(c[2] - c[1]))
                if w < h:
                    w, h = h, w
                return w, h

        if shapes and idx < len(shapes):
            area, aspect_ratio = shapes[idx][0], shapes[idx][1]
            if aspect_ratio > 0 and area > 0:
                ax2 = math.sqrt(4 * area / (math.pi * aspect_ratio))
                ax1 = aspect_ratio * ax2
                return ax1, ax2

        return None, None

    def run(self: object) -> object:
        """run method documentation."""
        pose_backend = None
        detection_cache = None
        cap = None
        try:
            if self._should_stop():
                return
            if not self.csv_path or not os.path.exists(self.csv_path):
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return
            if not self.video_path or not os.path.exists(self.video_path):
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return

            output_dir = self.params.get("INDIVIDUAL_DATASET_OUTPUT_DIR")
            if not output_dir:
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return

            df = pd.read_csv(self.csv_path)
            if "FrameID" not in df.columns and "Frame" in df.columns:
                df = df.rename(columns={"Frame": "FrameID"})
            if "TrajectoryID" not in df.columns and "Trajectory" in df.columns:
                df = df.rename(columns={"Trajectory": "TrajectoryID"})
            if df.empty or "FrameID" not in df.columns or "State" not in df.columns:
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return
            # Normalize numeric columns
            for col in ("FrameID", "X", "Y", "Theta"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            resize_factor = self.params.get("RESIZE_FACTOR", 1.0)
            position_scale = 1.0
            size_scale = 1.0 / resize_factor if resize_factor else 1.0

            if self.detection_cache_path and os.path.exists(self.detection_cache_path):
                detection_cache = DetectionCache(self.detection_cache_path, mode="r")

            # Respect runtime save toggle: if disabled, run interpolation/pose without writing
            # interpolated crops or related metadata artifacts to disk.
            save_interpolated_outputs = bool(
                self.params.get("ENABLE_INDIVIDUAL_IMAGE_SAVE", False)
            )
            gen_params = dict(self.params or {})
            gen_params["ENABLE_INDIVIDUAL_DATASET"] = save_interpolated_outputs
            gen_params["ENABLE_INDIVIDUAL_IMAGE_SAVE"] = save_interpolated_outputs

            gen = IndividualDatasetGenerator(
                gen_params,
                output_dir,
                Path(self.video_path).stem,
                self.params.get("INDIVIDUAL_DATASET_NAME", "individual_dataset"),
            )
            gen.enabled = save_interpolated_outputs

            pose_enabled = bool(self.params.get("ENABLE_POSE_EXTRACTOR", False))
            pose_kpt_source_names = []
            pose_kpt_labels = []
            if pose_enabled:
                from ..core.identity.feature_runtime import create_pose_backend

                try:
                    pose_backend = create_pose_backend(
                        self.params, out_root=str(Path(output_dir).expanduser())
                    )
                    pose_kpt_source_names = list(
                        getattr(pose_backend, "output_keypoint_names", []) or []
                    )
                    pose_kpt_labels = build_pose_keypoint_labels(
                        pose_kpt_source_names, len(pose_kpt_source_names)
                    )
                except Exception as exc:
                    logger.warning(
                        "Interpolated pose analysis disabled (backend init failed): %s",
                        exc,
                    )
                    pose_backend = None

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            try:
                max_x = df["X"].dropna().max()
                max_y = df["Y"].dropna().max()
                if (
                    resize_factor
                    and resize_factor < 1.0
                    and max_x <= frame_width * resize_factor * 1.05
                    and max_y <= frame_height * resize_factor * 1.05
                ):
                    position_scale = 1.0 / resize_factor
            except Exception:
                position_scale = 1.0

            interp_saved = 0
            interp_gaps = 0
            occluded_rows = 0
            interp_runs = 0
            interp_rows = []
            interp_pose_rows = []
            roi_rows = []
            roi_corners = []
            frame_tasks = defaultdict(list)
            if "TrajectoryID" not in df.columns:
                logger.warning("Interpolated crops skipped: CSV missing TrajectoryID.")
                self.finished_signal.emit({"saved": 0, "gaps": 0})
                return

            for traj_id, group in df.groupby("TrajectoryID"):
                if self._should_stop():
                    return
                group = group.sort_values("FrameID").reset_index(drop=True)
                states = group["State"].astype(str).str.strip().str.lower()
                # Treat any value containing 'occluded' as occluded
                states = states.where(
                    ~states.str.contains("occluded", na=False), "occluded"
                )
                occluded_rows += int((states == "occluded").sum())

                last_valid_idx = None
                i = 0
                while i < len(group):
                    if self._should_stop():
                        return
                    if states[i] != "occluded":
                        if not pd.isna(group.at[i, "X"]) and not pd.isna(
                            group.at[i, "Y"]
                        ):
                            last_valid_idx = i
                        i += 1
                        continue

                    if last_valid_idx is None:
                        i += 1
                        continue

                    j = i
                    while j < len(group) and states[j] == "occluded":
                        j += 1
                    if j >= len(group):
                        break

                    prev_row = group.iloc[last_valid_idx]
                    next_row = group.iloc[j]
                    if (
                        pd.isna(prev_row["X"])
                        or pd.isna(prev_row["Y"])
                        or pd.isna(next_row["X"])
                        or pd.isna(next_row["Y"])
                    ):
                        i = j
                        continue

                    f0 = int(prev_row["FrameID"])
                    f1 = int(next_row["FrameID"])
                    if f1 - f0 <= 1:
                        i = j
                        continue
                    interp_runs += 1

                    interp_total = max(0, f1 - f0 - 1)
                    interp_gaps += interp_total

                    det_id_prev = (
                        prev_row["DetectionID"]
                        if "DetectionID" in group.columns
                        else None
                    )
                    det_id_next = (
                        next_row["DetectionID"]
                        if "DetectionID" in group.columns
                        else None
                    )

                    w0, h0 = self._get_detection_size(detection_cache, f0, det_id_prev)
                    w1, h1 = self._get_detection_size(detection_cache, f1, det_id_next)

                    if w0 is None or h0 is None or w1 is None or h1 is None:
                        ref_size = self.params.get("REFERENCE_BODY_SIZE", 20.0)
                        w0 = w0 or ref_size * 2.2
                        h0 = h0 or ref_size * 0.8
                        w1 = w1 or ref_size * 2.2
                        h1 = h1 or ref_size * 0.8

                    for k in range(i, j):
                        if self._should_stop():
                            return
                        row = group.iloc[k]
                        f = int(row["FrameID"])
                        t = (f - f0) / (f1 - f0)
                        cx = float(prev_row["X"]) + t * (
                            float(next_row["X"]) - float(prev_row["X"])
                        )
                        cy = float(prev_row["Y"]) + t * (
                            float(next_row["Y"]) - float(prev_row["Y"])
                        )
                        theta = self._interp_angle(
                            float(prev_row["Theta"]), float(next_row["Theta"]), t
                        )
                        w = w0 + t * (w1 - w0)
                        h = h0 + t * (h1 - h0)

                        interp_index = max(1, f - f0)

                        frame_tasks[f].append(
                            {
                                "frame_id": f,
                                "cx": cx * position_scale,
                                "cy": cy * position_scale,
                                "w": w * size_scale,
                                "h": h * size_scale,
                                "theta": theta,
                                "traj_id": traj_id,
                                "interp_from": (f0, f1),
                                "interp_index": interp_index,
                                "interp_total": interp_total,
                            }
                        )

                    i = j
                    continue

            logger.info(
                f"Interpolated occlusion rows: {occluded_rows} "
                f"(runs: {interp_runs}, gaps: {interp_gaps})"
            )
            del df
            gc.collect()

            if frame_tasks:
                needed_frames = sorted(frame_tasks.keys())
                total_frames = len(needed_frames)
                current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                for idx, f in enumerate(needed_frames, start=1):
                    if self._should_stop():
                        return
                    if f != current_pos:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                    ret, frame = cap.read()
                    current_pos = f + 1
                    if not ret or frame is None:
                        continue
                    frame_pose_tasks = []
                    for task in frame_tasks[f]:
                        filename = ""
                        corners = gen.ellipse_to_obb_corners(
                            task["cx"],
                            task["cy"],
                            task["w"],
                            task["h"],
                            task["theta"],
                        )
                        if save_interpolated_outputs:
                            filename = gen.save_interpolated_crop(
                                frame=frame,
                                frame_id=task["frame_id"],
                                cx=task["cx"],
                                cy=task["cy"],
                                w=task["w"],
                                h=task["h"],
                                theta=task["theta"],
                                traj_id=task["traj_id"],
                                interp_from=task["interp_from"],
                                interp_index=task["interp_index"],
                                interp_total=task["interp_total"],
                            )
                        if save_interpolated_outputs and filename:
                            interp_saved += 1
                            interp_rows.append(
                                {
                                    "frame_id": int(task["frame_id"]),
                                    "trajectory_id": int(task["traj_id"]),
                                    "filename": filename,
                                    "interp_from_start": int(task["interp_from"][0]),
                                    "interp_from_end": int(task["interp_from"][1]),
                                    "interp_index": int(task["interp_index"]),
                                    "interp_total": int(task["interp_total"]),
                                }
                            )
                            roi_rows.append(
                                {
                                    "frame_id": int(task["frame_id"]),
                                    "trajectory_id": int(task["traj_id"]),
                                    "filename": filename,
                                    "cx": float(task["cx"]),
                                    "cy": float(task["cy"]),
                                    "w": float(task["w"]),
                                    "h": float(task["h"]),
                                    "theta": float(task["theta"]),
                                    "interp_from_start": int(task["interp_from"][0]),
                                    "interp_from_end": int(task["interp_from"][1]),
                                    "interp_index": int(task["interp_index"]),
                                    "interp_total": int(task["interp_total"]),
                                }
                            )
                            roi_corners.append(corners)
                        if pose_backend is not None:
                            pose_crop = None
                            pose_crop_info = None
                            try:
                                pose_crop, pose_crop_info = (
                                    gen._extract_obb_masked_crop(
                                        frame,
                                        corners,
                                        frame.shape[0],
                                        frame.shape[1],
                                    )
                                )
                            except Exception:
                                pose_crop = None
                                pose_crop_info = None
                            if pose_crop is not None and pose_crop.size > 0:
                                frame_pose_tasks.append(
                                    {
                                        "task": task,
                                        "filename": filename,
                                        "crop": pose_crop,
                                        "crop_info": pose_crop_info,
                                    }
                                )

                    if pose_backend is not None and frame_pose_tasks:
                        if self._should_stop():
                            return
                        pose_results = pose_backend.predict_crops(
                            [entry["crop"] for entry in frame_pose_tasks]
                        )
                        for pidx, entry in enumerate(frame_pose_tasks):
                            pose_out = (
                                pose_results[pidx] if pidx < len(pose_results) else None
                            )
                            pose_mean_conf = 0.0
                            pose_valid_fraction = 0.0
                            pose_num_valid = 0
                            pose_num_keypoints = 0
                            pose_wide = {}
                            if pose_out is not None:
                                pose_mean_conf = float(
                                    getattr(pose_out, "mean_conf", 0.0)
                                )
                                pose_valid_fraction = float(
                                    getattr(pose_out, "valid_fraction", 0.0)
                                )
                                pose_num_valid = int(getattr(pose_out, "num_valid", 0))
                                pose_num_keypoints = int(
                                    getattr(pose_out, "num_keypoints", 0)
                                )
                                keypoints = getattr(pose_out, "keypoints", None)
                                crop_info = entry.get("crop_info") or {}
                                crop_bbox = crop_info.get("crop_bbox")
                                if (
                                    keypoints is not None
                                    and crop_bbox is not None
                                    and len(crop_bbox) >= 2
                                    and len(keypoints) > 0
                                ):
                                    x0 = float(crop_bbox[0])
                                    y0 = float(crop_bbox[1])
                                    gkpts = np.asarray(
                                        keypoints, dtype=np.float32
                                    ).copy()
                                    gkpts[:, 0] += x0
                                    gkpts[:, 1] += y0
                                    if len(gkpts) > len(pose_kpt_labels):
                                        pose_kpt_labels = build_pose_keypoint_labels(
                                            pose_kpt_source_names, len(gkpts)
                                        )
                                    pose_wide = flatten_pose_keypoints_row(
                                        gkpts, pose_kpt_labels
                                    )

                            pose_row = {
                                "frame_id": int(entry["task"]["frame_id"]),
                                "trajectory_id": int(entry["task"]["traj_id"]),
                                "filename": entry["filename"],
                                "PoseMeanConf": pose_mean_conf,
                                "PoseValidFraction": pose_valid_fraction,
                                "PoseNumValid": pose_num_valid,
                                "PoseNumKeypoints": pose_num_keypoints,
                            }
                            pose_row.update(pose_wide)
                            interp_pose_rows.append(pose_row)

                    if idx % 25 == 0 or idx == total_frames:
                        progress = int((idx / total_frames) * 100)
                        self.progress_signal.emit(
                            progress,
                            f"Interpolating occlusions... {idx}/{total_frames}",
                        )
                        del frame
                        gc.collect()

            mapping_path = None
            roi_csv_path = None
            roi_npz_path = None
            pose_csv_path = None
            if save_interpolated_outputs and interp_rows and gen.crops_dir is not None:
                mapping_path = gen.crops_dir.parent / "interpolated_mapping.csv"
                try:
                    with open(mapping_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "frame_id",
                                "trajectory_id",
                                "filename",
                                "interp_from_start",
                                "interp_from_end",
                                "interp_index",
                                "interp_total",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(interp_rows)
                except Exception:
                    pass
            if save_interpolated_outputs and roi_rows and gen.crops_dir is not None:
                roi_csv_path = gen.crops_dir.parent / "interpolated_rois.csv"
                try:
                    with open(roi_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=[
                                "frame_id",
                                "trajectory_id",
                                "filename",
                                "cx",
                                "cy",
                                "w",
                                "h",
                                "theta",
                                "interp_from_start",
                                "interp_from_end",
                                "interp_index",
                                "interp_total",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(roi_rows)
                except Exception:
                    pass
                roi_npz_path = gen.crops_dir.parent / "interpolated_rois.npz"
                try:
                    np.savez_compressed(
                        str(roi_npz_path),
                        frame_id=np.array(
                            [r["frame_id"] for r in roi_rows], dtype=np.int64
                        ),
                        trajectory_id=np.array(
                            [r["trajectory_id"] for r in roi_rows], dtype=np.int64
                        ),
                        filename=np.array(
                            [r["filename"] for r in roi_rows], dtype=object
                        ),
                        cx=np.array([r["cx"] for r in roi_rows], dtype=np.float32),
                        cy=np.array([r["cy"] for r in roi_rows], dtype=np.float32),
                        w=np.array([r["w"] for r in roi_rows], dtype=np.float32),
                        h=np.array([r["h"] for r in roi_rows], dtype=np.float32),
                        theta=np.array(
                            [r["theta"] for r in roi_rows], dtype=np.float32
                        ),
                        interp_from_start=np.array(
                            [r["interp_from_start"] for r in roi_rows], dtype=np.int64
                        ),
                        interp_from_end=np.array(
                            [r["interp_from_end"] for r in roi_rows], dtype=np.int64
                        ),
                        interp_index=np.array(
                            [r["interp_index"] for r in roi_rows], dtype=np.int64
                        ),
                        interp_total=np.array(
                            [r["interp_total"] for r in roi_rows], dtype=np.int64
                        ),
                        obb_corners=(
                            np.stack(roi_corners).astype(np.float32)
                            if roi_corners
                            else np.zeros((0, 4, 2), dtype=np.float32)
                        ),
                    )
                except Exception:
                    pass
            if (
                save_interpolated_outputs
                and interp_pose_rows
                and gen.crops_dir is not None
            ):
                pose_csv_path = gen.crops_dir.parent / "interpolated_pose.csv"
                try:
                    pose_fieldnames = [
                        "frame_id",
                        "trajectory_id",
                        "filename",
                        *POSE_SUMMARY_COLUMNS,
                        *pose_wide_columns_for_labels(pose_kpt_labels),
                    ]
                    with open(pose_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(
                            f,
                            fieldnames=pose_fieldnames,
                        )
                        writer.writeheader()
                        writer.writerows(interp_pose_rows)
                except Exception:
                    pose_csv_path = None
            if save_interpolated_outputs:
                gen.finalize()
            if not self._should_stop():
                self.finished_signal.emit(
                    {
                        "saved": interp_saved,
                        "gaps": interp_gaps,
                        "mapping_path": str(mapping_path) if mapping_path else None,
                        "roi_csv_path": str(roi_csv_path) if roi_csv_path else None,
                        "roi_npz_path": str(roi_npz_path) if roi_npz_path else None,
                        "pose_csv_path": str(pose_csv_path) if pose_csv_path else None,
                        "pose_rows": (
                            interp_pose_rows
                            if (interp_pose_rows and not save_interpolated_outputs)
                            else None
                        ),
                    }
                )
        except Exception:
            self.finished_signal.emit({"saved": 0, "gaps": 0})
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            if detection_cache is not None:
                try:
                    detection_cache.close()
                except Exception:
                    pass
            if pose_backend is not None:
                try:
                    pose_backend.close()
                except Exception:
                    pass


class DatasetGenerationWorker(QThread):
    """Worker thread for generating training datasets without blocking the UI."""

    progress_signal = Signal(int, str)  # progress value, status message
    finished_signal = Signal(str, int)  # dataset_dir, num_frames
    error_signal = Signal(str)  # error message

    def __init__(
        self,
        video_path,
        csv_path,
        detection_cache_path,
        output_dir,
        dataset_name,
        class_name,
        params,
        max_frames,
        diversity_window,
        include_context,
        probabilistic,
    ):
        super().__init__()
        self.video_path = video_path
        self.csv_path = csv_path
        self.detection_cache_path = detection_cache_path
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.class_name = class_name
        self.params = params
        self.max_frames = max_frames
        self.diversity_window = diversity_window
        self.include_context = include_context
        self.probabilistic = probabilistic
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def run(self: object) -> object:
        """run method documentation."""
        detection_cache = None
        try:
            from ..data.dataset_generation import FrameQualityScorer, export_dataset

            if self._should_stop():
                return
            self.progress_signal.emit(5, "Initializing dataset generation...")

            # Load tracking CSV to compute quality scores
            self.progress_signal.emit(10, "Loading tracking data...")
            df = pd.read_csv(self.csv_path)

            # Initialize quality scorer
            self.progress_signal.emit(15, "Initializing quality scorer...")
            scorer = FrameQualityScorer(self.params)
            if self.detection_cache_path and os.path.exists(self.detection_cache_path):
                try:
                    detection_cache = DetectionCache(
                        self.detection_cache_path, mode="r"
                    )
                    if not detection_cache.is_compatible():
                        detection_cache.close()
                        detection_cache = None
                except Exception:
                    detection_cache = None

            # Score each frame
            self.progress_signal.emit(20, "Scoring frames...")
            unique_frames = df["FrameID"].unique()
            total_unique = len(unique_frames)

            for idx, frame_id in enumerate(unique_frames):
                if self._should_stop():
                    return
                if idx % 100 == 0:  # Update progress every 100 frames
                    progress = 20 + int((idx / total_unique) * 30)
                    self.progress_signal.emit(
                        progress, f"Scoring frames ({idx}/{total_unique})..."
                    )

                frame_data = df[df["FrameID"] == frame_id]
                raw_confidences = []
                if detection_cache is not None:
                    try:
                        _, _, _, raw_confidences, _, _ = detection_cache.get_frame(
                            int(frame_id)
                        )
                    except Exception:
                        raw_confidences = []

                # Detection data
                detection_data = {
                    "confidences": (
                        raw_confidences
                        if raw_confidences
                        else (
                            frame_data["DetectionConfidence"].tolist()
                            if "DetectionConfidence" in frame_data.columns
                            else []
                        )
                    ),
                    "count": len(frame_data),
                }

                # Tracking data
                tracking_data = {
                    "lost_tracks": int((frame_data["State"] == "lost").sum()),
                    "uncertainties": (
                        frame_data["PositionUncertainty"].tolist()
                        if "PositionUncertainty" in frame_data.columns
                        else []
                    ),
                }

                scorer.score_frame(frame_id, detection_data, tracking_data)

            if self._should_stop():
                return
            # Select worst frames with diversity
            self.progress_signal.emit(50, "Selecting challenging frames...")
            selected_frames = scorer.get_worst_frames(
                self.max_frames, self.diversity_window, probabilistic=self.probabilistic
            )

            if not selected_frames:
                self.error_signal.emit("No frames met the quality criteria for export.")
                return

            # Export dataset
            self.progress_signal.emit(60, f"Exporting {len(selected_frames)} frames...")
            if self._should_stop():
                return
            dataset_dir = export_dataset(
                video_path=self.video_path,
                csv_path=self.csv_path,
                frame_ids=selected_frames,
                output_dir=self.output_dir,
                dataset_name=self.dataset_name,
                class_name=self.class_name,
                params=self.params,
                include_context=self.include_context,
            )

            if not self._should_stop():
                self.progress_signal.emit(100, "Dataset generation complete!")
                self.finished_signal.emit(dataset_dir, len(selected_frames))

        except Exception as e:
            logger.exception("Error during dataset generation")
            self.error_signal.emit(str(e))
        finally:
            if detection_cache is not None:
                try:
                    detection_cache.close()
                except Exception:
                    pass


def get_video_config_path(video_path: object) -> object:
    """Get the config file path for a given video file."""
    if not video_path:
        return None
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(video_dir, f"{video_name}_config.json")


def get_models_directory() -> object:
    """
    Get the path to the local models directory.

    Returns the models/YOLO-obb directory for OBB detection models.
    Creates the directory if it doesn't exist.
    """
    # Get project root directory (multi-animal-tracker/)
    # __file__ is: .../multi-animal-tracker/src/multi_tracker/gui/main_window.py
    # Need to go up 4 levels: gui -> multi_tracker -> src -> multi-animal-tracker
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    models_dir = os.path.join(project_root, "models", "YOLO-obb")

    # Create directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    return models_dir


def get_pose_models_directory(backend: str | None = None) -> object:
    """
    Get the local pose-model repository directory.

    Layout:
      models/YOLO-pose/
      models/SLEAP/
    """
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    models_root = os.path.join(project_root, "models")
    os.makedirs(models_root, exist_ok=True)
    if not backend:
        return models_root
    key = str(backend or "").strip().lower()
    backend_dirname = "SLEAP" if key == "sleap" else "YOLO-pose"
    backend_dir = os.path.join(models_root, backend_dirname)
    os.makedirs(backend_dir, exist_ok=True)
    return backend_dir


def resolve_pose_model_path(model_path: object, backend: str | None = None) -> object:
    """Resolve a pose model path (relative or absolute) to an absolute path when possible."""
    if not model_path:
        return model_path

    path_str = str(model_path).strip()
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str

    candidates = []
    models_root = get_pose_models_directory()
    candidates.append(os.path.join(models_root, path_str))
    if backend:
        candidates.append(os.path.join(get_pose_models_directory(backend), path_str))
    # Backward compatibility with older models/pose/{yolo,sleap} layout.
    legacy_pose_root = os.path.join(models_root, "pose")
    candidates.append(os.path.join(legacy_pose_root, path_str))
    if backend:
        legacy_backend = "sleap" if str(backend).strip().lower() == "sleap" else "yolo"
        candidates.append(os.path.join(legacy_pose_root, legacy_backend, path_str))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    return path_str


def make_pose_model_path_relative(model_path: object) -> object:
    """Convert absolute pose-model paths under models/ into relative paths."""
    if not model_path or not os.path.isabs(str(model_path)):
        return model_path
    pose_root = get_pose_models_directory()
    try:
        rel_path = os.path.relpath(str(model_path), pose_root)
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass
    return model_path


def resolve_model_path(model_path: object) -> object:
    """
    Resolve a model path to an absolute path.

    If the path is relative, look for it in the models directory.
    If absolute and exists, return as-is.

    Args:
        model_path: Relative or absolute model path

    Returns:
        Absolute path to the model file, or original path if not found
    """
    if not model_path:
        return model_path

    # If already absolute and exists, return it
    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    # Try to resolve relative to models directory
    models_dir = get_models_directory()
    resolved_path = os.path.join(models_dir, model_path)

    if os.path.exists(resolved_path):
        return resolved_path

    # If relative path doesn't exist in models dir, try as-is
    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    # Return original if nothing works (will fail later with clear error)
    return model_path


def make_model_path_relative(model_path: object) -> object:
    """
    Convert an absolute model path to relative if it's in the models directory.

    This allows presets to be portable across devices.

    Args:
        model_path: Absolute or relative model path

    Returns:
        Relative path if model is in archive, otherwise absolute path
    """
    if not model_path or not os.path.isabs(model_path):
        return model_path

    models_dir = get_models_directory()

    # Check if model is inside the models directory
    try:
        rel_path = os.path.relpath(model_path, models_dir)
        # If relpath doesn't start with .., it's inside models_dir
        if not rel_path.startswith(".."):
            return rel_path
    except (ValueError, TypeError):
        pass

    # Return absolute path if not in models directory
    return model_path


def get_yolo_model_registry_path() -> object:
    """Return path to the local YOLO model metadata registry JSON."""
    return os.path.join(get_models_directory(), "model_registry.json")


def _sanitize_model_token(text: object) -> object:
    """Sanitize a species/info token for filenames and metadata."""
    raw = str(text or "").strip()
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in raw)
    return cleaned.strip("_")


def _normalize_yolo_model_metadata(metadata: object) -> object:
    """Normalize legacy model metadata to species + model_info schema."""
    if not isinstance(metadata, dict):
        return {}

    normalized = dict(metadata)
    species = _sanitize_model_token(normalized.get("species", ""))
    model_info = _sanitize_model_token(normalized.get("model_info", ""))

    # Legacy migration path: identifier/id -> species + model_info
    legacy_identifier = normalized.get("identifier") or normalized.get("id") or ""
    if (not species or not model_info) and legacy_identifier:
        legacy_identifier = _sanitize_model_token(legacy_identifier)
        parts = [p for p in legacy_identifier.split("_") if p]
        if not species and parts:
            species = parts[0]
        if not model_info:
            model_info = "_".join(parts[1:]) if len(parts) > 1 else "model"

    if species:
        normalized["species"] = species
    if model_info:
        normalized["model_info"] = model_info

    # Drop deprecated fields
    normalized.pop("identifier", None)
    normalized.pop("id", None)
    return normalized


def load_yolo_model_registry() -> object:
    """Load YOLO model metadata registry (path -> metadata)."""
    registry_path = get_yolo_model_registry_path()
    if not os.path.exists(registry_path):
        return {}
    try:
        with open(registry_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}

        migrated = {}
        changed = False
        for k, v in data.items():
            key = str(k)
            norm = _normalize_yolo_model_metadata(v)
            migrated[key] = norm
            if norm != v:
                changed = True

        if changed:
            save_yolo_model_registry(migrated)
            logger.info("Migrated YOLO model registry to species/model_info schema")

        return migrated
    except Exception as e:
        logger.warning(f"Failed to load YOLO model registry: {e}")
        return {}


def save_yolo_model_registry(registry: object) -> object:
    """Persist YOLO model metadata registry JSON."""
    registry_path = get_yolo_model_registry_path()
    try:
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save YOLO model registry: {e}")


def get_yolo_model_metadata(model_path: object) -> object:
    """Get metadata for a model path if registered."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    if rel_path in registry:
        return _normalize_yolo_model_metadata(registry[rel_path])
    # Backward compatibility: if absolute path was stored in older versions
    abs_path = resolve_model_path(model_path)
    return _normalize_yolo_model_metadata(registry.get(abs_path, {}))


def register_yolo_model(model_path: object, metadata: object) -> object:
    """Register/overwrite metadata entry for a model path."""
    rel_path = make_model_path_relative(model_path)
    registry = load_yolo_model_registry()
    registry[rel_path] = _normalize_yolo_model_metadata(metadata)
    save_yolo_model_registry(registry)


logger = logging.getLogger(__name__)


class CollapsibleGroupBox(QWidget):
    """
    A collapsible group box widget that can expand/collapse its content.
    Used for advanced settings that don't need to be visible all the time.
    """

    toggled = Signal(bool)  # Emitted when expanded/collapsed

    def __init__(self, title: str, parent=None, initially_expanded: bool = False):
        super().__init__(parent)
        self._is_expanded = initially_expanded
        self._title = title
        self._content_widget = None
        self._accordion_group = None  # Reference to accordion container

        # Main layout
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(0, 0, 0, 0)
        self._main_layout.setSpacing(0)

        # Header button (acts as toggle)
        self._header_button = QToolButton()
        self._header_button.setStyleSheet("""
            QToolButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 12px;
                font-weight: bold;
                font-size: 12px;
                color: #4a9eff;
                text-align: left;
            }
            QToolButton:hover {
                background-color: #454545;
                border-color: #666;
            }
            QToolButton:checked {
                background-color: #404040;
                border-color: #4a9eff;
            }
        """)
        self._header_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header_button.setArrowType(
            Qt.RightArrow if not initially_expanded else Qt.DownArrow
        )
        self._header_button.setText(title)
        self._header_button.setCheckable(True)
        self._header_button.setChecked(initially_expanded)
        self._header_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header_button.clicked.connect(self._on_header_clicked)

        self._main_layout.addWidget(self._header_button)

        # Content container
        self._content_container = QWidget()
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(0, 5, 0, 5)
        self._content_container.setVisible(initially_expanded)

        self._main_layout.addWidget(self._content_container)

    def setContentLayout(self: object, layout: object) -> object:
        """Set the content layout for the collapsible section."""
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Add new content as a widget
        content_widget = QWidget()
        content_widget.setLayout(layout)
        self._content_layout.addWidget(content_widget)
        self._content_widget = content_widget

    def addWidget(self: object, widget: object) -> object:
        """Add a widget to the content area."""
        self._content_layout.addWidget(widget)

    def addLayout(self: object, layout: object) -> object:
        """Add a layout to the content area."""
        self._content_layout.addLayout(layout)

    def setAccordionGroup(self: object, accordion: object) -> object:
        """Set the accordion group this collapsible belongs to."""
        self._accordion_group = accordion

    def _on_header_clicked(self, checked):
        """Handle header button click."""
        if checked:
            # Notify accordion to collapse others
            if self._accordion_group:
                self._accordion_group.collapseAllExcept(self)
        self.setExpanded(checked)

    def setExpanded(self: object, expanded: bool) -> object:
        """Set the expanded state of the collapsible."""
        self._is_expanded = expanded
        self._header_button.setChecked(expanded)
        self._header_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._content_container.setVisible(expanded)
        self.toggled.emit(expanded)

    def isExpanded(self) -> bool:
        """Check if the collapsible is expanded."""
        return self._is_expanded

    def title(self) -> str:
        """Get the title of the collapsible."""
        return self._title


class AccordionContainer:
    """
    Manages a group of CollapsibleGroupBox widgets to ensure only one is expanded at a time.
    """

    def __init__(self):
        self._collapsibles = []

    def addCollapsible(self: object, collapsible: CollapsibleGroupBox) -> object:
        """Add a collapsible to this accordion group."""
        collapsible.setAccordionGroup(self)
        self._collapsibles.append(collapsible)

    def collapseAllExcept(self: object, keep_expanded: CollapsibleGroupBox) -> object:
        """Collapse all collapsibles except the specified one."""
        for collapsible in self._collapsibles:
            if collapsible is not keep_expanded and collapsible.isExpanded():
                collapsible.setExpanded(False)

    def collapseAll(self: object) -> object:
        """Collapse all collapsibles."""
        for collapsible in self._collapsibles:
            collapsible.setExpanded(False)

    def expandFirst(self: object) -> object:
        """Expand the first collapsible (if any)."""
        if self._collapsibles:
            self._collapsibles[0].setExpanded(True)


class MainWindow(QMainWindow):
    """
    Main application window providing GUI interface for multi-animal tracking.
    """

    parameters_changed = Signal(dict)

    def __init__(self):
        """Initialize the main application window and UI components."""
        super().__init__()
        self.setWindowTitle("Multi-Animal-Tracker")
        self.resize(1360, 850)

        # Set comprehensive dark mode styling
        self.setStyleSheet("""
            /* Main window and widgets */
            QMainWindow, QWidget { background-color: #2b2b2b; color: #ffffff; font-family: -apple-system, system-ui, sans-serif; }

            /* Tabs */
            QTabWidget::pane { border: 1px solid #444; top: -1px; }
            QTabBar::tab {
                background: #353535; color: #aaa; padding: 8px 12px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #4a9eff; color: white; font-weight: bold; }
            QTabBar::tab:hover { background: #404040; }

            /* Group boxes */
            QGroupBox {
                font-weight: bold; border: 1px solid #555; border-radius: 6px;
                margin-top: 20px; padding-top: 10px; background-color: #323232;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #4a9eff;
            }

            /* Buttons */
            QPushButton {
                background-color: #444; border: 1px solid #555; color: #fff;
                padding: 6px 12px; border-radius: 4px; min-height: 25px;
            }
            QPushButton:hover { background-color: #555; border-color: #666; }
            QPushButton:pressed { background-color: #2a75c4; }
            QPushButton:checked { background-color: #2a75c4; border: 1px solid #4a9eff; }
            QPushButton:disabled { background-color: #333; color: #666; border-color: #333; }

            /* Specific Action Buttons */
            QPushButton#ActionBtn { background-color: #4a9eff; font-weight: bold; font-size: 13px; }
            QPushButton#ActionBtn:hover { background-color: #3d8bdb; }
            QPushButton#StopBtn { background-color: #d9534f; font-weight: bold; }
            QPushButton#StopBtn:hover { background-color: #c9302c; }

            /* Inputs */
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #222; border: 1px solid #555; border-radius: 3px;
                padding: 4px; color: #fff; selection-background-color: #4a9eff;
                min-width: 120px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 1px solid #4a9eff; }

            /* ComboBox dropdown */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555;
                background-color: #555;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox::drop-down:hover { background-color: #666; }
            QComboBox::down-arrow {
                image: none;
                border: 2px solid #fff;
                width: 6px;
                height: 6px;
                border-top: none;
                border-right: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                border: 1px solid #555;
                selection-background-color: #4a9eff;
                selection-color: #fff;
                color: #fff;
                padding: 4px;
                min-width: 200px;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 8px;
                min-height: 24px;
                border: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #3d8bdb;
                color: #fff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #4a9eff;
                color: #fff;
            }

            /* SpinBox arrows */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 18px;
                border-left: 1px solid #555;
                background-color: #555;
                border-top-right-radius: 3px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #666;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #4a9eff;
            }

            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 18px;
                border-left: 1px solid #555;
                background-color: #555;
                border-bottom-right-radius: 3px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #666;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #4a9eff;
            }

            /* Scrollbars */
            QScrollBar:vertical { background: #2b2b2b; width: 12px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 6px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #666; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

            /* Progress Bar */
            QProgressBar {
                border: 1px solid #555; border-radius: 4px; text-align: center; background: #222;
            }
            QProgressBar::chunk { background-color: #4a9eff; width: 10px; margin: 0.5px; }

            QSplitter::handle { background-color: #444; }
            """)

        # === STATE VARIABLES ===
        self.roi_base_frame = None
        self.roi_points = []
        self.roi_mask = None
        self.roi_selection_active = False
        self.roi_fitted_circle = None
        # Enhanced ROI support: multiple shapes with inclusion/exclusion
        self.roi_shapes = (
            []
        )  # List of dicts: {'type': 'circle'/'polygon', 'params': ..., 'mode': 'include'/'exclude'}
        self.roi_current_mode = "circle"  # 'circle' or 'polygon'
        self.roi_current_zone_type = "include"  # 'include' or 'exclude'

        self.histogram_panel = None
        self.histogram_window = None
        self.current_worker = None

        self.tracking_worker = None
        self.merge_worker = None
        self.csv_writer_thread = None
        self.dataset_worker = None
        self.interp_worker = None
        self.reversal_worker = None
        self.final_full_trajs = []
        self.temporary_files = []  # Track temporary files for cleanup
        self.session_log_handler = None  # Track current session log file handler
        self._individual_dataset_run_id = None
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None
        self.current_interpolated_pose_csv_path = None
        self.current_interpolated_pose_df = None
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False

        # Preview frame for live image adjustments
        self.preview_frame_original = None  # Original frame without adjustments
        self.detection_test_result = None  # Store detection test result
        self.current_video_path = None
        self.detected_sizes = None  # Store detected object sizes for statistics

        # ROI optimization tracking
        self.roi_crop_warning_shown = (
            False  # Track if we've warned about cropping this session
        )

        # ROI display caching (for performance)
        self._roi_masked_cache = {}  # Cache: {(frame_id, roi_hash): masked_image}
        self._roi_hash = None  # Hash of current ROI configuration

        # Interactive pan/zoom state
        self._is_panning = False
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0
        self._pan_start_pos = None
        self._scroll_start_h = 0
        self._scroll_start_v = 0

        # Track first frame for auto-fit during tracking
        self._tracking_first_frame = True
        self._tracking_frame_size = None  # (width, height) of resized tracking frames

        # UI interaction state
        self._video_interactions_enabled = True
        self._ui_state = "idle"
        self._saved_widget_enabled_states = {}
        self._pending_finish_after_interp = False
        self._stop_all_requested = False

        # Advanced configuration (for power users)
        self.advanced_config = self._load_advanced_config()

        # Video player state
        self.video_cap = None  # cv2.VideoCapture for video playback
        self.video_total_frames = 0
        self.video_current_frame_idx = 0
        self.last_read_frame_idx = (
            -1
        )  # Track last frame read for sequential optimization
        self.is_playing = False
        self.playback_timer = None  # QTimer for playback

        # === UI CONSTRUCTION ===
        self.init_ui()

        # === POST-INIT ===
        # Disable wheel events on all spinboxes to prevent accidental value changes
        self._disable_spinbox_wheel_events()

        # Config is now loaded automatically when a video is selected
        # instead of at startup
        self._connect_parameter_signals()

        # Cache preview-related controls for UI state transitions
        self._preview_controls = self._collect_preview_controls()

        # Default to "no video loaded" state
        self._apply_ui_state("no_video")

    def init_ui(self: object) -> object:
        """Build the structured UI using Splitter and Tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout is a horizontal splitter (Video Left | Controls Right)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Horizontal)

        # --- LEFT PANEL: Video & ROI ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        # Video Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #000; border: none;")
        self.scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label = QLabel("")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("color: #666; font-size: 16px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll.setWidget(self.video_label)
        self._show_video_logo_placeholder()

        # Enable mouse tracking and events for interactive pan/zoom
        self.video_label.setMouseTracking(True)
        self.video_label.mousePressEvent = self._handle_video_mouse_press
        self.video_label.mouseMoveEvent = self._handle_video_mouse_move
        self.video_label.mouseReleaseEvent = self._handle_video_mouse_release
        self.video_label.mouseDoubleClickEvent = self._handle_video_double_click
        self.video_label.wheelEvent = self._handle_video_wheel

        # Enable pinch-to-zoom gesture
        self.video_label.setAttribute(Qt.WA_AcceptTouchEvents, True)
        self.video_label.grabGesture(Qt.PinchGesture)
        self.video_label.event = self._handle_video_event

        # ROI Toolbar (Contextual to video)
        roi_frame = QFrame()
        roi_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        roi_main_layout = QVBoxLayout(roi_frame)
        roi_main_layout.setContentsMargins(10, 5, 10, 5)

        # Top row: mode selection and controls
        roi_layout = QHBoxLayout()
        roi_label = QLabel("ROI controls")
        roi_label.setStyleSheet("font-weight: bold; color: #bbb;")

        # Mode selector
        self.combo_roi_mode = QComboBox()
        self.combo_roi_mode.addItems(["Circle", "Polygon"])
        self.combo_roi_mode.setToolTip("Select shape type for ROI")
        self.combo_roi_mode.currentIndexChanged.connect(self._on_roi_mode_changed)

        # Zone type selector (Include/Exclude)
        self.combo_roi_zone = QComboBox()
        self.combo_roi_zone.addItems(["Include Zone", "Exclude Zone"])
        self.combo_roi_zone.setToolTip(
            "Include: Area where tracking is active\n"
            "Exclude: Area to remove from tracking (applied after inclusions)"
        )
        self.combo_roi_zone.currentIndexChanged.connect(self._on_roi_zone_changed)

        self.btn_start_roi = QPushButton("Add Shape")
        self.btn_start_roi.clicked.connect(self.start_roi_selection)
        self.btn_start_roi.setShortcut("Ctrl+R")
        self.btn_start_roi.setToolTip("Start adding ROI shape (Ctrl+R)")

        self.btn_finish_roi = QPushButton("Confirm Shape")
        self.btn_finish_roi.clicked.connect(self.finish_roi_selection)
        self.btn_finish_roi.setEnabled(False)
        self.btn_finish_roi.setShortcut("Ctrl+F")
        self.btn_finish_roi.setToolTip("Finish current shape (Ctrl+F)")

        self.btn_undo_roi = QPushButton("Undo Last")
        self.btn_undo_roi.clicked.connect(self.undo_last_roi_shape)
        self.btn_undo_roi.setEnabled(False)
        self.btn_undo_roi.setToolTip("Remove last added shape")

        self.btn_clear_roi = QPushButton("Clear All")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setShortcut("Ctrl+C")
        self.btn_clear_roi.setToolTip("Clear all ROI shapes (Ctrl+C)")

        self.btn_crop_video = QPushButton("Crop Video to ROI")
        self.btn_crop_video.clicked.connect(self.crop_video_to_roi)
        self.btn_crop_video.setEnabled(False)
        self.btn_crop_video.setToolTip(
            "Generate a cropped video containing only the ROI area\n"
            "This can significantly improve tracking performance"
        )
        self.btn_crop_video.setStyleSheet(
            "QPushButton { background-color: #2d7a3e; }"
            "QPushButton:hover { background-color: #3a9150; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )

        self.roi_status_label = QLabel("No ROI")
        self.roi_status_label.setStyleSheet("color: #888; margin-left: 10px;")

        roi_layout.addWidget(roi_label)
        roi_layout.addWidget(self.combo_roi_mode)
        roi_layout.addWidget(self.combo_roi_zone)
        roi_layout.addWidget(self.btn_start_roi)
        roi_layout.addWidget(self.btn_finish_roi)
        roi_layout.addWidget(self.btn_undo_roi)
        roi_layout.addWidget(self.btn_clear_roi)
        roi_layout.addWidget(self.btn_crop_video)
        roi_layout.addStretch()

        roi_main_layout.addLayout(roi_layout)

        # Second row: status and optimization info
        roi_status_layout = QHBoxLayout()
        roi_status_layout.addWidget(self.roi_status_label)
        self.roi_optimization_label = QLabel("")
        self.roi_optimization_label.setStyleSheet(
            "color: #f0ad4e; margin-left: 10px; font-weight: bold;"
        )
        roi_status_layout.addWidget(self.roi_optimization_label)
        roi_main_layout.addLayout(roi_status_layout)

        # Instructions (Hidden unless active)
        self.roi_instructions = QLabel("")
        self.roi_instructions.setWordWrap(True)
        self.roi_instructions.setStyleSheet(
            "color: #4a9eff; font-size: 11px; font-weight: bold; "
            "padding: 6px; background-color: #1a3a5a; border-radius: 4px;"
        )
        roi_main_layout.addWidget(self.roi_instructions)

        left_layout.addWidget(self.scroll, stretch=1)

        # Interactive instructions
        self.interaction_help = QLabel(
            "Double-click: Fit to screen  •  Drag: Pan  •  Ctrl+Scroll/Pinch: Zoom"
        )
        self.interaction_help.setAlignment(Qt.AlignCenter)
        self.interaction_help.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic; "
            "padding: 4px; background-color: #1a1a1a; border-radius: 3px;"
        )
        left_layout.addWidget(self.interaction_help)

        left_layout.addWidget(roi_frame)

        # Zoom control under video
        zoom_frame = QFrame()
        zoom_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(10, 5, 10, 5)

        zoom_label = QLabel("Zoom")
        zoom_label.setStyleSheet("font-weight: bold; color: #bbb;")

        self.slider_zoom = QSlider(Qt.Horizontal)
        self.slider_zoom.setRange(10, 500)  # 0.1x to 5.0x, scaled by 100
        self.slider_zoom.setValue(100)  # 1.0x
        self.slider_zoom.setTickPosition(QSlider.TicksBelow)
        self.slider_zoom.setTickInterval(50)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)

        self.label_zoom_val = QLabel("1.00x")
        self.label_zoom_val.setStyleSheet(
            "color: #4a9eff; font-weight: bold; min-width: 50px;"
        )

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.slider_zoom, stretch=1)
        zoom_layout.addWidget(self.label_zoom_val)

        left_layout.addWidget(zoom_frame)

        # Preview detection button (uses current player frame)
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #323232; border-radius: 6px;")
        preview_layout = QHBoxLayout(preview_frame)
        preview_layout.setContentsMargins(10, 5, 10, 5)

        self.btn_test_detection = QPushButton("Test Detection on Preview")
        self.btn_test_detection.clicked.connect(self._test_detection_on_preview)
        self.btn_test_detection.setEnabled(False)
        self.btn_test_detection.setStyleSheet(
            "background-color: #4a9eff; color: white; font-weight: bold;"
        )
        preview_layout.addWidget(self.btn_test_detection)

        left_layout.addWidget(preview_frame)

        # --- RIGHT PANEL: Configuration Tabs & Actions ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # Tab Widget
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.setUsesScrollButtons(
            True
        )  # Enable scroll buttons when tabs don't fit
        self.tabs.setElideMode(Qt.ElideNone)  # Don't truncate tab text

        # Tab 1: Setup (Files & Performance)
        self.tab_setup = QWidget()
        self.setup_setup_ui()
        self.tabs.addTab(self.tab_setup, "Get Started")

        # Tab 2: Detection (Image, Method, Params)
        self.tab_detection = QWidget()
        self.setup_detection_ui()
        self.tabs.addTab(self.tab_detection, "Find Animals")

        # Tab 3: Tracking (Kalman, Logic, Lifecycle)
        self.tab_tracking = QWidget()
        self.setup_tracking_ui()
        self.tabs.addTab(self.tab_tracking, "Track Movement")

        # Tab 4: Data (Post-proc, Histograms)
        self.tab_data = QWidget()
        self.setup_data_ui()
        self.tabs.addTab(self.tab_data, "Clean Results")

        # Tab 5: Dataset Generation (Active Learning)
        self.tab_dataset = QWidget()
        self.setup_dataset_ui()
        self.tabs.addTab(self.tab_dataset, "Build Dataset")

        # Tab 6: Individual Analysis (Identity)
        self.tab_individual = QWidget()
        self.setup_individual_analysis_ui()
        self.tabs.addTab(self.tab_individual, "Analyze Individuals")

        right_layout.addWidget(self.tabs, stretch=1)

        # Persistent Action Panel (Bottom Right)
        action_frame = QFrame()
        action_frame.setStyleSheet(
            "background-color: #252525; border-top: 1px solid #444; border-radius: 0px;"
        )
        action_layout = QVBoxLayout(action_frame)

        # Progress Info
        prog_layout = QHBoxLayout()
        self.progress_label = QLabel("Ready")
        self.progress_label.setVisible(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)

        # Real-time stats display
        stats_layout = QHBoxLayout()
        stats_layout.setContentsMargins(5, 5, 5, 5)

        self.label_current_fps = QLabel("FPS: --")
        self.label_current_fps.setStyleSheet(
            "color: #4a9eff; font-weight: bold; font-size: 11px;"
        )
        self.label_current_fps.setVisible(False)
        stats_layout.addWidget(self.label_current_fps)

        self.label_elapsed_time = QLabel("Elapsed: --")
        self.label_elapsed_time.setStyleSheet("color: #888; font-size: 11px;")
        self.label_elapsed_time.setVisible(False)
        stats_layout.addWidget(self.label_elapsed_time)

        self.label_eta = QLabel("ETA: --")
        self.label_eta.setStyleSheet("color: #888; font-size: 11px;")
        self.label_eta.setVisible(False)
        stats_layout.addWidget(self.label_eta)

        stats_layout.addStretch()

        # Buttons
        btn_layout = QHBoxLayout()

        self.btn_preview = QPushButton("Preview Mode")
        self.btn_preview.setCheckable(True)
        self.btn_preview.clicked.connect(lambda ch: self.toggle_preview(ch))
        self.btn_preview.setMinimumHeight(40)

        self.btn_start = QPushButton("Start Full Tracking")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.setCheckable(True)
        self.btn_start.clicked.connect(lambda ch: self.toggle_tracking(ch))
        self.btn_start.setMinimumHeight(40)

        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_start)

        action_layout.addLayout(prog_layout)
        action_layout.addLayout(stats_layout)
        action_layout.addLayout(btn_layout)

        right_layout.addWidget(action_frame)

        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(right_panel)

        # Set initial splitter ratio (60% Video, 40% Controls) and minimum sizes
        total_width = 1360  # Default window width
        self.splitter.setSizes([int(total_width * 0.6), int(total_width * 0.4)])
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)

        main_layout.addWidget(self.splitter)

        # =====================================================================
        # INITIALIZE PRESETS
        # =====================================================================
        # Populate preset combo box with available presets
        self._populate_preset_combo()

        # Load default preset (custom if available, otherwise default.json)
        self._load_default_preset_on_startup()

    # =========================================================================
    # TAB UI BUILDERS
    # =========================================================================

    def setup_setup_ui(self: object) -> object:
        """Tab 1: Setup - Files, Video, Display & Debug."""
        layout = QVBoxLayout(self.tab_setup)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)

        # ============================================================
        # Preset Selector
        # ============================================================
        g_presets = QGroupBox("How do you want to start?")
        vl_presets = QVBoxLayout(g_presets)
        vl_presets.addWidget(
            self._create_help_label(
                "Load optimized default values for different model organisms. Video-specific configs override presets."
            )
        )

        preset_layout = QHBoxLayout()
        preset_label = QLabel("Organism preset")
        preset_label.setStyleSheet("font-weight: bold;")

        self.combo_presets = QComboBox()
        self.combo_presets.setToolTip(
            "Select preset optimized for your organism.\n"
            "Custom: Your personal saved defaults (if exists)"
        )
        self._populate_preset_combo()

        self.btn_load_preset = QPushButton("Load Preset")
        self.btn_load_preset.clicked.connect(self._load_selected_preset)
        self.btn_load_preset.setToolTip("Apply selected preset to all parameters")

        self.btn_save_custom = QPushButton("Save as Custom")
        self.btn_save_custom.clicked.connect(self._save_custom_preset)
        self.btn_save_custom.setToolTip("Save current settings as your custom defaults")

        self.preset_status_label = QLabel("")
        self.preset_status_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 10px;"
        )

        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self.combo_presets, stretch=1)
        preset_layout.addWidget(self.btn_load_preset)
        preset_layout.addWidget(self.btn_save_custom)
        preset_layout.addWidget(self.preset_status_label, stretch=1)

        vl_presets.addLayout(preset_layout)

        # Description display
        self.preset_description_label = QLabel("")
        self.preset_description_label.setWordWrap(True)
        self.preset_description_label.setStyleSheet(
            "color: #bbb; font-style: italic; font-size: 10px; padding: 5px; "
            "background-color: #1a1a1a; border-radius: 3px;"
        )
        self.preset_description_label.setVisible(False)
        vl_presets.addWidget(self.preset_description_label)

        # Connect combo box to show description
        self.combo_presets.currentIndexChanged.connect(
            self._on_preset_selection_changed
        )

        form.addWidget(g_presets)

        # ============================================================
        # Video Setup (File Management + Frame Rate)
        # ============================================================
        g_files = QGroupBox("What video are you analyzing?")
        vl_files = QVBoxLayout(g_files)
        vl_files.addWidget(
            self._create_help_label(
                "Select your input video and output locations. Configuration is auto-saved per video - "
                "next time you load the same video, your settings will be restored automatically."
            )
        )
        fl = QFormLayout(None)
        fl.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.btn_file = QPushButton("Select Input Video...")
        self.btn_file.clicked.connect(self.select_file)
        self.file_line = QLineEdit()
        self.file_line.setPlaceholderText("path/to/video.mp4")
        self.file_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl.addRow(self.btn_file, self.file_line)

        # FPS with detect button (moved here from Reference Parameters)
        fps_layout = QHBoxLayout()
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1.0, 240.0)
        self.spin_fps.setSingleStep(1.0)
        self.spin_fps.setValue(30.0)
        self.spin_fps.setDecimals(2)
        self.spin_fps.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_fps.setToolTip(
            "Acquisition frame rate (frames per second) at which the video was recorded.\n"
            "NOTE: This may differ from the video file's playback framerate.\n"
            "Use 'Detect from Video' to read from file metadata as a starting point.\n"
            "Time-dependent parameters (velocity, durations) scale with this.\n"
            "Affects: motion prediction, track lifecycle, velocity thresholds."
        )
        self.spin_fps.valueChanged.connect(self._update_fps_info)
        fps_layout.addWidget(self.spin_fps)

        self.btn_detect_fps = QPushButton("Detect from Video")
        self.btn_detect_fps.clicked.connect(self._detect_fps_from_current_video)
        self.btn_detect_fps.setEnabled(False)
        self.btn_detect_fps.setToolTip(
            "Auto-detect frame rate from video metadata (may differ from actual acquisition rate)"
        )
        fps_layout.addWidget(self.btn_detect_fps)
        fl.addRow("What frame rate was the video acquired at (FPS)?", fps_layout)

        # FPS info label
        self.label_fps_info = QLabel()
        self.label_fps_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        fl.addRow("", self.label_fps_info)

        vl_files.addLayout(fl)
        form.addWidget(g_files)

        # ============================================================
        # Video Player & Frame Range
        # ============================================================
        self.g_video_player = QGroupBox("Which part of the video should be used?")
        vl_player = QVBoxLayout(self.g_video_player)
        vl_player.addWidget(
            self._create_help_label(
                "Preview video and select frame range for tracking. Use the slider to seek through the video."
            )
        )

        # Video info label
        self.lbl_video_info = QLabel("No video loaded")
        self.lbl_video_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic; padding: 5px;"
        )
        vl_player.addWidget(self.lbl_video_info)

        # Timeline slider
        timeline_layout = QVBoxLayout()
        self.lbl_current_frame = QLabel("Frame: -")
        self.lbl_current_frame.setStyleSheet("font-size: 10px; color: #aaa;")
        timeline_layout.addWidget(self.lbl_current_frame)

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setMinimum(0)
        self.slider_timeline.setMaximum(0)
        self.slider_timeline.setValue(0)
        self.slider_timeline.setEnabled(False)
        self.slider_timeline.setToolTip("Seek through video frames")
        self.slider_timeline.valueChanged.connect(self._on_timeline_changed)
        timeline_layout.addWidget(self.slider_timeline)
        vl_player.addLayout(timeline_layout)

        # Playback controls
        controls_layout = QHBoxLayout()

        self.btn_first_frame = QPushButton("⏮ First")
        self.btn_first_frame.setEnabled(False)
        self.btn_first_frame.clicked.connect(self._goto_first_frame)
        self.btn_first_frame.setToolTip("Go to first frame")
        controls_layout.addWidget(self.btn_first_frame)

        self.btn_prev_frame = QPushButton("◀ Prev")
        self.btn_prev_frame.setEnabled(False)
        self.btn_prev_frame.clicked.connect(self._goto_prev_frame)
        self.btn_prev_frame.setToolTip("Previous frame")
        controls_layout.addWidget(self.btn_prev_frame)

        self.btn_play_pause = QPushButton("▶ Play")
        self.btn_play_pause.setEnabled(False)
        self.btn_play_pause.clicked.connect(self._toggle_playback)
        self.btn_play_pause.setToolTip("Play/pause video")
        controls_layout.addWidget(self.btn_play_pause)

        self.btn_next_frame = QPushButton("Next ▶")
        self.btn_next_frame.setEnabled(False)
        self.btn_next_frame.clicked.connect(self._goto_next_frame)
        self.btn_next_frame.setToolTip("Next frame")
        controls_layout.addWidget(self.btn_next_frame)

        self.btn_last_frame = QPushButton("Last ⏭")
        self.btn_last_frame.setEnabled(False)
        self.btn_last_frame.clicked.connect(self._goto_last_frame)
        self.btn_last_frame.setToolTip("Go to last frame")
        controls_layout.addWidget(self.btn_last_frame)

        self.btn_random_seek = QPushButton("🎲 Random")
        self.btn_random_seek.setEnabled(False)
        self.btn_random_seek.clicked.connect(self._goto_random_frame)
        self.btn_random_seek.setToolTip("Jump to a random frame")
        controls_layout.addWidget(self.btn_random_seek)

        controls_layout.addStretch()

        # Playback speed control
        controls_layout.addWidget(QLabel("Playback speed"))
        self.combo_playback_speed = QComboBox()
        self.combo_playback_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.combo_playback_speed.setCurrentText("1x")
        self.combo_playback_speed.setEnabled(False)
        self.combo_playback_speed.setToolTip("Playback speed")
        controls_layout.addWidget(self.combo_playback_speed)

        vl_player.addLayout(controls_layout)

        # Frame range selection
        range_group = QGroupBox("What frame range should be tracked?")
        range_layout = QFormLayout(range_group)
        range_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Start frame
        start_layout = QHBoxLayout()
        self.spin_start_frame = QSpinBox()
        self.spin_start_frame.setMinimum(0)
        self.spin_start_frame.setMaximum(0)
        self.spin_start_frame.setValue(0)
        self.spin_start_frame.setEnabled(False)
        self.spin_start_frame.setToolTip("First frame to track (0-based index)")
        self.spin_start_frame.valueChanged.connect(self._on_frame_range_changed)
        start_layout.addWidget(self.spin_start_frame)

        self.btn_set_start_current = QPushButton("Set to Current")
        self.btn_set_start_current.setEnabled(False)
        self.btn_set_start_current.clicked.connect(self._set_start_to_current)
        self.btn_set_start_current.setToolTip("Set start frame to current frame")
        start_layout.addWidget(self.btn_set_start_current)
        range_layout.addRow("Which frame should tracking start from?", start_layout)

        # End frame
        end_layout = QHBoxLayout()
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setMinimum(0)
        self.spin_end_frame.setMaximum(0)
        self.spin_end_frame.setValue(0)
        self.spin_end_frame.setEnabled(False)
        self.spin_end_frame.setToolTip("Last frame to track (0-based index, inclusive)")
        self.spin_end_frame.valueChanged.connect(self._on_frame_range_changed)
        end_layout.addWidget(self.spin_end_frame)

        self.btn_set_end_current = QPushButton("Set to Current")
        self.btn_set_end_current.setEnabled(False)
        self.btn_set_end_current.clicked.connect(self._set_end_to_current)
        self.btn_set_end_current.setToolTip("Set end frame to current frame")
        end_layout.addWidget(self.btn_set_end_current)
        range_layout.addRow("Which frame should tracking stop at?", end_layout)

        # Range info
        self.lbl_range_info = QLabel()
        self.lbl_range_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic; padding: 5px;"
        )
        range_layout.addRow("", self.lbl_range_info)

        # Reset to full range button
        self.btn_reset_range = QPushButton("Reset to Full Video")
        self.btn_reset_range.setEnabled(False)
        self.btn_reset_range.clicked.connect(self._reset_frame_range)
        self.btn_reset_range.setToolTip("Reset to track entire video")
        range_layout.addRow("", self.btn_reset_range)

        vl_player.addWidget(range_group)
        form.addWidget(self.g_video_player)

        # Initially hide video player (shown when video is loaded)
        self.g_video_player.setVisible(False)

        # ============================================================
        # Output Files
        # ============================================================
        g_output = QGroupBox("Where should results be saved?")
        vl_output = QVBoxLayout(g_output)
        fl_output = QFormLayout(None)
        fl_output.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.btn_csv = QPushButton("Select CSV Output...")
        self.btn_csv.clicked.connect(self.select_csv)
        self.csv_line = QLineEdit()
        self.csv_line.setPlaceholderText("path/to/output.csv")
        self.csv_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl_output.addRow(self.btn_csv, self.csv_line)

        # Config Management
        config_layout = QHBoxLayout()
        self.btn_load_config = QPushButton("Load Config...")
        self.btn_load_config.clicked.connect(self.load_config)
        self.btn_load_config.setToolTip("Manually load configuration from a JSON file")
        config_layout.addWidget(self.btn_load_config)

        self.btn_save_config = QPushButton("Save Config...")
        self.btn_save_config.clicked.connect(self.save_config)
        self.btn_save_config.setToolTip("Save current settings to a JSON file")
        config_layout.addWidget(self.btn_save_config)

        self.btn_show_gpu_info = QPushButton("GPU Info")
        self.btn_show_gpu_info.clicked.connect(self.show_gpu_info)
        self.btn_show_gpu_info.setToolTip(
            "Show available GPU and acceleration information"
        )
        config_layout.addWidget(self.btn_show_gpu_info)

        config_layout.addStretch()
        fl_output.addRow("Save configuration snapshot?", config_layout)

        # Config status label
        self.config_status_label = QLabel("No config loaded (using defaults)")
        self.config_status_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 10px;"
        )
        fl_output.addRow("", self.config_status_label)
        vl_output.addLayout(fl_output)

        form.addWidget(g_output)

        # ============================================================
        # System Performance
        # ============================================================
        g_sys = QGroupBox("How should performance be balanced?")
        vl_sys = QVBoxLayout(g_sys)
        vl_sys.addWidget(
            self._create_help_label(
                "Resize factor reduces computational cost by downscaling frames. "
                "Lower values speed up processing but reduce spatial accuracy."
            )
        )
        fl_sys = QFormLayout(None)
        fl_sys.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_resize = QDoubleSpinBox()
        self.spin_resize.setRange(0.1, 1.0)
        self.spin_resize.setSingleStep(0.1)
        self.spin_resize.setValue(1.0)
        self.spin_resize.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.spin_resize.setToolTip(
            "Downscale video frames for faster processing.\n"
            "1.0 = full resolution, 0.5 = half resolution (4× faster).\n"
            "All body-size-based parameters auto-scale with this value."
        )
        fl_sys.addRow(
            "How much should frames be downscaled before processing?", self.spin_resize
        )

        self.combo_compute_runtime = QComboBox()
        self.combo_compute_runtime.setToolTip(
            "Global compute runtime for detection and pose.\n"
            "Only runtimes compatible with all enabled pipelines are shown."
        )
        self.combo_compute_runtime.currentIndexChanged.connect(
            self._on_runtime_context_changed
        )
        fl_sys.addRow(
            "Which compute runtime should be used globally?",
            self.combo_compute_runtime,
        )

        self.check_save_confidence = QCheckBox("Save confidence metrics (slower)")
        self.check_save_confidence.setChecked(True)
        self.check_save_confidence.setToolTip(
            "Save detection, assignment, and position uncertainty metrics to CSV.\n"
            "Useful for post-hoc quality control but adds ~10-20% processing time.\n"
            "Disable for maximum tracking speed."
        )
        fl_sys.addRow("", self.check_save_confidence)

        # Use Cached Detections
        self.chk_use_cached_detections = QCheckBox(
            "Reuse Cached Detections When Available"
        )
        self.chk_use_cached_detections.setChecked(True)
        self.chk_use_cached_detections.setToolTip(
            "Automatically reuse detections from previous runs if available.\n"
            "Cache is model-specific: only reused if detection method/model hasn't changed.\n"
            "Massive speedup for re-processing with different tracking parameters.\n"
            "Disable to force fresh detection on every run."
        )
        fl_sys.addRow("", self.chk_use_cached_detections)

        # Visualization-Free Mode
        self.chk_visualization_free = QCheckBox(
            "Enable Visualization-Free Mode (Maximum Speed)"
        )
        self.chk_visualization_free.setChecked(False)
        self.chk_visualization_free.setToolTip(
            "Skip all frame visualization and rendering.\n"
            "Significantly faster processing (2-4× speedup).\n"
            "Real-time FPS/ETA stats still shown in UI.\n"
            "Recommended for large batch processing."
        )
        self.chk_visualization_free.stateChanged.connect(
            self._on_visualization_mode_changed
        )
        fl_sys.addRow("", self.chk_visualization_free)

        vl_sys.addLayout(fl_sys)
        form.addWidget(g_sys)

        # ============================================================
        # Display Settings (moved from Visuals tab)
        # ============================================================
        self.g_display = QGroupBox("How should the preview look?")
        vl_display = QVBoxLayout(self.g_display)
        vl_display.addWidget(
            self._create_help_label(
                "Configure visual overlays shown during tracking. These settings affect "
                "both the live preview and exported video output."
            )
        )

        # Common overlays
        self.chk_show_circles = QCheckBox("Show Track Markers (Circles)")
        self.chk_show_circles.setChecked(True)
        self.chk_show_circles.setToolTip("Draw circles around tracked animals.")
        vl_display.addWidget(self.chk_show_circles)

        self.chk_show_orientation = QCheckBox("Show Orientation Lines")
        self.chk_show_orientation.setChecked(True)
        self.chk_show_orientation.setToolTip("Draw lines showing heading direction.")
        vl_display.addWidget(self.chk_show_orientation)

        self.chk_show_trajectories = QCheckBox("Show Trajectory Trails")
        self.chk_show_trajectories.setChecked(True)
        self.chk_show_trajectories.setToolTip(
            "Draw recent path history for each track."
        )
        vl_display.addWidget(self.chk_show_trajectories)

        self.chk_show_labels = QCheckBox("Show ID Labels")
        self.chk_show_labels.setChecked(True)
        self.chk_show_labels.setToolTip("Display unique track IDs on each animal.")
        vl_display.addWidget(self.chk_show_labels)

        self.chk_show_state = QCheckBox("Show State Text")
        self.chk_show_state.setChecked(True)
        self.chk_show_state.setToolTip(
            "Display tracking state (ACTIVE, PREDICTED, etc.)."
        )
        vl_display.addWidget(self.chk_show_state)

        self.chk_show_kalman_uncertainty = QCheckBox("Show prediction uncertainty")
        self.chk_show_kalman_uncertainty.setChecked(False)
        self.chk_show_kalman_uncertainty.setToolTip(
            "Draw ellipses showing Kalman filter position uncertainty.\n"
            "Larger ellipse = more uncertainty in predicted position.\n"
            "Useful for debugging tracking quality and filter convergence."
        )
        vl_display.addWidget(self.chk_show_kalman_uncertainty)

        # Trail length
        f_trail = QFormLayout(None)
        self.spin_traj_hist = QSpinBox()
        self.spin_traj_hist.setRange(1, 60)
        self.spin_traj_hist.setValue(5)
        self.spin_traj_hist.setToolTip(
            "Length of trajectory trails to display (1-60 seconds).\n"
            "Longer = more visible path history but more cluttered.\n"
            "Recommended: 3-10 seconds."
        )
        f_trail.addRow(
            "How many seconds of trail history should be shown?", self.spin_traj_hist
        )
        vl_display.addLayout(f_trail)

        form.addWidget(self.g_display)

        # ============================================================
        # Advanced / Debug (moved from Visuals tab)
        # ============================================================
        g_debug = QGroupBox("Need advanced troubleshooting options?")
        v_dbg = QVBoxLayout(g_debug)
        v_dbg.addWidget(
            self._create_help_label(
                "Enable verbose logging to see detailed tracking decisions. Useful for troubleshooting "
                "but generates large log files. Disable for production runs."
            )
        )
        self.chk_debug_logging = QCheckBox("Enable detailed debug logging")
        self.chk_debug_logging.stateChanged.connect(self.toggle_debug_logging)
        v_dbg.addWidget(self.chk_debug_logging)
        form.addWidget(g_debug)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)
        self._populate_compute_runtime_options(preferred="cpu")
        self._on_runtime_context_changed()

    def setup_detection_ui(self: object) -> object:
        """Tab 2: Detection - Method, Image Proc, Algo specific."""
        layout = QVBoxLayout(self.tab_detection)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # ============================================================
        # 1. Detection Method Selector
        # ============================================================
        g_method = QGroupBox("How should animals be detected?")
        l_method_outer = QVBoxLayout(g_method)
        l_method_outer.addWidget(
            self._create_help_label(
                "Choose how to detect animals in each frame. Background Subtraction works by modeling "
                "the static background and finding moving objects. YOLO uses deep learning to detect animals directly."
            )
        )
        f_method = QFormLayout(None)
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.currentIndexChanged.connect(
            self._on_detection_method_changed_ui
        )
        f_method.addRow(
            "Which detection method should be used?", self.combo_detection_method
        )

        # Legacy device selection (hidden; derived from canonical runtime).
        self.combo_device = QComboBox()
        device_options = ["auto", "cpu"]
        device_tooltip_parts = [
            "Select compute device for detection:",
            "  • auto - Automatically select best available device",
            "  • cpu - CPU-only mode",
        ]

        if TORCH_CUDA_AVAILABLE:
            device_options.append("cuda:0")
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU (not available)")

        if MPS_AVAILABLE:
            device_options.append("mps")
            device_tooltip_parts.append("  • mps - Apple Silicon GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • mps - Apple Silicon GPU (not available)")

        device_tooltip_parts.append(
            "\nApplies to both YOLO and Background Subtraction GPU acceleration."
        )

        self.combo_device.addItems(device_options)
        self.combo_device.setToolTip("\n".join(device_tooltip_parts))
        f_method.addRow("Which compute device should run detection?", self.combo_device)
        device_label = f_method.labelForField(self.combo_device)
        if device_label is not None:
            device_label.setVisible(False)
        self.combo_device.setVisible(False)

        l_method_outer.addLayout(f_method)
        vbox.addWidget(g_method)

        # Stacked Widget for Method Specific Params
        self.stack_detection = QStackedWidget()

        # --- Page 0: Background Subtraction Params ---
        page_bg = QWidget()
        l_bg = QVBoxLayout(page_bg)
        l_bg.setContentsMargins(0, 0, 0, 0)
        l_bg.addWidget(
            self._create_help_label(
                "Background subtraction identifies moving animals by comparing each frame to a learned background model. "
                "Start with defaults and increase threshold if you see too much noise, decrease if animals are missed."
            )
        )

        # Create accordion for BG subtraction settings
        self.bg_accordion = AccordionContainer()

        # Image Enhancement (Pre-processing)
        self.g_img = CollapsibleGroupBox("Brightness / Contrast / Gamma")
        self.bg_accordion.addCollapsible(self.g_img)
        vl_img = QVBoxLayout()
        vl_img.addWidget(
            self._create_help_label(
                "Adjust image properties before detection to improve contrast between animals and background. "
                "Start with default values and adjust only if animals are hard to distinguish."
            )
        )

        # Brightness slider
        bright_layout = QVBoxLayout()
        bright_label_row = QHBoxLayout()
        bright_label_row.addWidget(QLabel("Brightness"))
        self.label_brightness_val = QLabel("0")
        self.label_brightness_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        bright_label_row.addWidget(self.label_brightness_val)
        bright_label_row.addStretch()
        bright_layout.addLayout(bright_label_row)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-255, 255)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTickPosition(QSlider.TicksBelow)
        self.slider_brightness.setTickInterval(50)
        self.slider_brightness.valueChanged.connect(self._on_brightness_changed)
        self.slider_brightness.setToolTip(
            "Adjust overall image brightness.\n"
            "Positive = lighter, Negative = darker.\n"
            "Use to improve contrast between animals and background."
        )
        bright_layout.addWidget(self.slider_brightness)
        vl_img.addLayout(bright_layout)

        # Contrast slider
        contrast_layout = QVBoxLayout()
        contrast_label_row = QHBoxLayout()
        contrast_label_row.addWidget(QLabel("Contrast"))
        self.label_contrast_val = QLabel("1.0")
        self.label_contrast_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        contrast_label_row.addWidget(self.label_contrast_val)
        contrast_label_row.addStretch()
        contrast_layout.addLayout(contrast_label_row)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 300)  # 0.0 to 3.0, scaled by 100
        self.slider_contrast.setValue(100)  # 1.0
        self.slider_contrast.setTickPosition(QSlider.TicksBelow)
        self.slider_contrast.setTickInterval(50)
        self.slider_contrast.valueChanged.connect(self._on_contrast_changed)
        self.slider_contrast.setToolTip(
            "Adjust image contrast (difference between light and dark).\n"
            "1.0 = original, >1.0 = more contrast, <1.0 = less contrast.\n"
            "Increase to make animals stand out from background."
        )
        contrast_layout.addWidget(self.slider_contrast)
        vl_img.addLayout(contrast_layout)

        # Gamma slider
        gamma_layout = QVBoxLayout()
        gamma_label_row = QHBoxLayout()
        gamma_label_row.addWidget(QLabel("Gamma"))
        self.label_gamma_val = QLabel("1.0")
        self.label_gamma_val.setStyleSheet("color: #4a9eff; font-weight: bold;")
        gamma_label_row.addWidget(self.label_gamma_val)
        gamma_label_row.addStretch()
        gamma_layout.addLayout(gamma_label_row)

        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 300)  # 0.1 to 3.0, scaled by 100
        self.slider_gamma.setValue(100)  # 1.0
        self.slider_gamma.setTickPosition(QSlider.TicksBelow)
        self.slider_gamma.setTickInterval(50)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        self.slider_gamma.setToolTip(
            "Adjust gamma correction (mid-tone brightness).\n"
            "1.0 = original, >1.0 = brighter mid-tones, <1.0 = darker mid-tones.\n"
            "Use to enhance detail in shadowed or bright areas."
        )
        gamma_layout.addWidget(self.slider_gamma)
        vl_img.addLayout(gamma_layout)

        # Dark on light checkbox
        self.chk_dark_on_light = QCheckBox("Animals are darker than background")
        self.chk_dark_on_light.setChecked(True)
        self.chk_dark_on_light.setToolTip(
            "Check if animals are darker than background (most common).\n"
            "Uncheck if animals are lighter than background.\n"
            "This inverts the foreground detection."
        )
        vl_img.addWidget(self.chk_dark_on_light)
        self.g_img.setContentLayout(vl_img)
        l_bg.addWidget(self.g_img)

        # Background Model
        g_bg_model = CollapsibleGroupBox("Background Estimation")
        self.bg_accordion.addCollapsible(g_bg_model)
        vl_bg_model = QVBoxLayout()
        vl_bg_model.addWidget(
            self._create_help_label(
                "Build a model of the static background. Priming frames establish initial model, learning rate "
                "controls adaptation speed, threshold sets sensitivity. Lower threshold = more sensitive detection."
            )
        )
        f_bg = QFormLayout(None)
        f_bg.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_bg_prime = QSpinBox()
        self.spin_bg_prime.setRange(0, 5000)
        self.spin_bg_prime.setValue(10)
        self.spin_bg_prime.setToolTip(
            "Number of initial frames to build background model.\n"
            "Recommended: 10-100 frames.\n"
            "Use more if background varies or animals are present initially."
        )
        f_bg.addRow(
            "How many startup frames should build the background?", self.spin_bg_prime
        )

        self.chk_adaptive_bg = QCheckBox("Continuously update background model")
        self.chk_adaptive_bg.setChecked(True)
        self.chk_adaptive_bg.setToolTip(
            "Continuously update background model during tracking.\n"
            "Recommended: Enable for videos with changing lighting.\n"
            "Disable for static background to improve performance."
        )
        f_bg.addRow(self.chk_adaptive_bg)

        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setValue(0.001)
        self.spin_bg_learning.setToolTip(
            "How quickly background adapts to changes (0.0001-0.1).\n"
            "Lower = slower adaptation (stable, good for mostly static background).\n"
            "Higher = faster adaptation (use for variable lighting/shadows)."
        )
        f_bg.addRow("How quickly should the background adapt?", self.spin_bg_learning)

        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setToolTip(
            "Pixel intensity difference to detect foreground (0-255).\n"
            "Lower = more sensitive (detects subtle animals, more noise).\n"
            "Higher = less sensitive (cleaner, may miss animals).\n"
            "Recommended: 30-70 depending on contrast."
        )
        f_bg.addRow("What subtraction threshold should be used?", self.spin_threshold)
        vl_bg_model.addLayout(f_bg)
        g_bg_model.setContentLayout(vl_bg_model)
        l_bg.addWidget(g_bg_model)

        # Lighting Stab
        g_light = CollapsibleGroupBox("Scene Lighting Stabilization")
        self.bg_accordion.addCollapsible(g_light)
        vl_light = QVBoxLayout()
        vl_light.addWidget(
            self._create_help_label(
                "Compensate for gradual lighting changes (clouds, time of day). Smoothing factor controls "
                "adaptation speed - higher = slower/more stable. Enable for outdoor or variable-light videos."
            )
        )
        f_light = QFormLayout(None)
        f_light.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.chk_lighting_stab = QCheckBox("Enable Stabilization")
        self.chk_lighting_stab.setChecked(True)
        self.chk_lighting_stab.setToolTip(
            "Compensate for gradual lighting changes over time.\n"
            "Recommended: Enable for videos with variable lighting.\n"
            "Disable for consistent illumination to improve speed."
        )
        f_light.addRow(self.chk_lighting_stab)

        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setValue(0.95)
        self.spin_lighting_smooth.setToolTip(
            "Temporal smoothing factor for lighting correction (0.8-0.999).\n"
            "Higher = smoother, slower adaptation to lighting changes.\n"
            "Lower = faster response to sudden lighting shifts.\n"
            "Recommended: 0.9-0.98"
        )
        f_light.addRow(
            "How much lighting smoothing should be applied?", self.spin_lighting_smooth
        )

        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setSingleStep(2)
        self.spin_lighting_median.setValue(5)
        self.spin_lighting_median.setToolTip(
            "Median filter window size (odd number, 3-15).\n"
            "Larger window = smoother lighting estimate, slower response.\n"
            "Smaller window = faster response, less smoothing.\n"
            "Recommended: 5-9"
        )
        f_light.addRow(
            "How many frames for the lighting median window?", self.spin_lighting_median
        )
        vl_light.addLayout(f_light)
        g_light.setContentLayout(vl_light)
        l_bg.addWidget(g_light)

        # Morphology (Standard)
        g_morph = CollapsibleGroupBox("Noise Removal and Morphology")
        self.bg_accordion.addCollapsible(g_morph)
        vl_morph = QVBoxLayout()
        vl_morph.addWidget(
            self._create_help_label(
                "Clean up detected blobs using morphological operations. Closing fills small holes, opening removes "
                "small noise. Larger kernels = stronger effect but may distort shape. Use odd numbers only."
            )
        )
        f_morph = QFormLayout(None)
        f_morph.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 25)
        self.spin_morph_size.setSingleStep(2)
        self.spin_morph_size.setValue(5)
        self.spin_morph_size.setToolTip(
            "Morphological operation kernel size (odd number, 1-25).\n"
            "Larger = more aggressive noise removal, may merge nearby animals.\n"
            "Smaller = preserves detail, may leave noise.\n"
            "Recommended: 3-7 for typical tracking scenarios."
        )
        f_morph.addRow("What main kernel size should be used?", self.spin_morph_size)

        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        self.spin_min_contour.setToolTip(
            "Minimum contour area in pixels² to keep.\n"
            "Filters out small noise blobs after morphology.\n"
            "Recommended: 20-100 depending on animal size and zoom.\n"
            "Note: Similar to min object size but in absolute pixels."
        )
        f_morph.addRow(
            "What is the smallest contour area to keep (px^2)?", self.spin_min_contour
        )

        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        self.spin_max_contour_multiplier.setToolTip(
            "Maximum contour area as multiplier of minimum (5-100).\n"
            "Max area = min_contour × this multiplier.\n"
            "Filters out very large blobs (clusters, shadows, artifacts).\n"
            "Recommended: 10-30"
        )
        f_morph.addRow("Maximum contour multiplier", self.spin_max_contour_multiplier)
        vl_morph.addLayout(f_morph)
        g_morph.setContentLayout(vl_morph)
        l_bg.addWidget(g_morph)

        # Morphology (Advanced/Splitting)
        g_split = CollapsibleGroupBox("Split Touching Animals")
        self.bg_accordion.addCollapsible(g_split)
        vl_split = QVBoxLayout()
        vl_split.addWidget(
            self._create_help_label(
                "Split touching animals using erosion/dilation. Conservative split uses watershed, aggressive uses "
                "multi-stage erosion/dilation. Enable only if animals frequently touch."
            )
        )
        f_split = QFormLayout(None)
        self.chk_conservative_split = QCheckBox("Use conservative split (erosion)")
        self.chk_conservative_split.setChecked(True)
        self.chk_conservative_split.setToolTip(
            "Use erosion to separate touching animals more conservatively.\n"
            "Recommended: Enable to avoid over-splitting single animals.\n"
            "Disable for aggressive separation of tightly clustered animals."
        )
        f_split.addRow(self.chk_conservative_split)

        h_split = QHBoxLayout()
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setRange(1, 15)
        self.spin_conservative_kernel.setSingleStep(2)
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_kernel.setToolTip(
            "Erosion kernel size (odd number, 1-15).\n"
            "Larger = more aggressive separation.\n"
            "Recommended: 3-5"
        )
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setRange(1, 10)
        self.spin_conservative_erode.setValue(1)
        self.spin_conservative_erode.setToolTip(
            "Number of erosion iterations (1-10).\n"
            "More iterations = stronger separation effect.\n"
            "Recommended: 1-2"
        )
        h_split.addWidget(QLabel("Kernel size"))
        h_split.addWidget(self.spin_conservative_kernel)
        h_split.addWidget(QLabel("Iterations"))
        h_split.addWidget(self.spin_conservative_erode)
        f_split.addRow(h_split)

        self.spin_merge_threshold = QSpinBox()
        self.spin_merge_threshold.setRange(100, 10000)
        self.spin_merge_threshold.setValue(1000)
        self.spin_merge_threshold.setToolTip(
            "Maximum area (px²) of small blobs to merge with nearby animals.\n"
            "Helps reconnect fragmented detections.\n"
            "Lower = merge more aggressively, Higher = keep fragments separate.\n"
            "Recommended: 500-2000"
        )
        f_split.addRow(
            "What merge area threshold should be used?", self.spin_merge_threshold
        )

        self.chk_additional_dilation = QCheckBox("Reconnect thin parts (dilation)")
        self.chk_additional_dilation.setToolTip(
            "Use dilation to reconnect thin parts (e.g., legs, antennae).\n"
            "Recommended: Enable if animals have thin appendages.\n"
            "Disable to maintain accurate body shape."
        )
        f_split.addRow(self.chk_additional_dilation)

        h_dil = QHBoxLayout()
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setRange(1, 15)
        self.spin_dilation_kernel_size.setSingleStep(2)
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_kernel_size.setToolTip(
            "Dilation kernel size (odd number, 1-15).\n"
            "Larger = thicker reconnection.\n"
            "Recommended: 3-5"
        )
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setRange(1, 10)
        self.spin_dilation_iterations.setValue(2)
        self.spin_dilation_iterations.setToolTip(
            "Number of dilation iterations (1-10).\n"
            "More iterations = thicker result.\n"
            "Recommended: 1-3"
        )
        h_dil.addWidget(QLabel("Kernel size"))
        h_dil.addWidget(self.spin_dilation_kernel_size)
        h_dil.addWidget(QLabel("Iterations"))
        h_dil.addWidget(self.spin_dilation_iterations)
        f_split.addRow(h_dil)
        vl_split.addLayout(f_split)
        g_split.setContentLayout(vl_split)

        l_bg.addWidget(g_split)

        # --- Page 1: YOLO Params ---
        page_yolo = QWidget()
        l_yolo = QVBoxLayout(page_yolo)
        l_yolo.setContentsMargins(0, 0, 0, 0)
        l_yolo.addWidget(
            self._create_help_label(
                "YOLO uses a trained neural network to detect animals. Choose your model file and adjust confidence "
                "threshold to balance detection sensitivity vs false positives. Higher confidence = fewer false detections."
            )
        )

        self.yolo_group = QGroupBox("How should YOLO detection be configured?")
        self.yolo_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        f_yolo = QFormLayout(self.yolo_group)

        self.combo_yolo_model = QComboBox()
        self._refresh_yolo_model_combo()
        self.combo_yolo_model.currentIndexChanged.connect(self.on_yolo_model_changed)
        self.combo_yolo_model.setToolTip(
            "YOLO model for oriented bounding box detection.\n"
            "yolo26s = balanced speed/accuracy, yolo26n = fastest.\n"
            "Select 'Custom Model...' to use your own trained model."
        )
        f_yolo.addRow("Which YOLO model should be used?", self.combo_yolo_model)

        # Custom model container (hidden by default)
        self.yolo_custom_model_widget = QWidget()
        h_cust = QHBoxLayout(self.yolo_custom_model_widget)
        h_cust.setContentsMargins(0, 0, 0, 0)
        self.yolo_custom_model_line = QLineEdit()
        self.btn_yolo_custom_model = QPushButton("...")
        self.btn_yolo_custom_model.clicked.connect(self.select_yolo_custom_model)
        h_cust.addWidget(self.yolo_custom_model_line)
        h_cust.addWidget(self.btn_yolo_custom_model)
        f_yolo.addRow(
            "Which custom YOLO model file should be used?",
            self.yolo_custom_model_widget,
        )
        self.yolo_custom_model_widget.setVisible(False)

        self.spin_yolo_confidence = QDoubleSpinBox()
        self.spin_yolo_confidence.setRange(0.01, 1.0)
        self.spin_yolo_confidence.setValue(0.25)
        self.spin_yolo_confidence.setToolTip(
            "Minimum confidence score for YOLO detections (0.01-1.0).\n"
            "Lower = more detections (more false positives).\n"
            "Higher = fewer detections (may miss animals).\n"
            "Recommended: 0.2-0.4"
        )
        f_yolo.addRow(
            "What minimum YOLO confidence should be accepted?",
            self.spin_yolo_confidence,
        )

        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        self.spin_yolo_iou.setToolTip(
            "Intersection-over-Union threshold for non-max suppression (0.01-1.0).\n"
            "Lower = more aggressive duplicate removal.\n"
            "Higher = keep more overlapping detections.\n"
            "Recommended: 0.5-0.8"
        )
        f_yolo.addRow("What IOU overlap threshold should YOLO use?", self.spin_yolo_iou)

        self.chk_use_custom_obb_iou = QCheckBox("Use custom OBB overlap filtering")
        self.chk_use_custom_obb_iou.setChecked(True)
        self.chk_use_custom_obb_iou.setEnabled(False)
        self.chk_use_custom_obb_iou.setToolTip(
            "Custom polygon-based OBB IOU filtering is always enabled.\n"
            "This improves overlap handling consistency across cached and live detections."
        )
        self.chk_use_custom_obb_iou.setVisible(False)

        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setPlaceholderText("e.g. 15, 16 (Empty for all)")
        self.line_yolo_classes.setToolTip(
            "Comma-separated class IDs to detect (leave empty for all classes).\n"
            "Example: '0,1,2' to detect only classes 0, 1, and 2.\n"
            "Refer to your model's class definitions."
        )
        f_yolo.addRow("Which classes should be detected?", self.line_yolo_classes)

        l_yolo.addWidget(self.yolo_group)

        # ============================================================
        # GPU Acceleration Settings (TensorRT + Batching)
        # ============================================================
        self.g_gpu_accel = QGroupBox("Which hardware should be used?")
        vl_gpu = QVBoxLayout(self.g_gpu_accel)
        vl_gpu.addWidget(
            self._create_help_label(
                "Optimize YOLO inference speed using GPU acceleration. Only applies to YOLO detection on GPU devices."
            )
        )

        f_gpu = QFormLayout(None)
        f_gpu.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # TensorRT Optimization
        self.chk_enable_tensorrt = QCheckBox("Enable TensorRT (NVIDIA Only)")
        self.chk_enable_tensorrt.setChecked(False)
        self.chk_enable_tensorrt.setEnabled(TENSORRT_AVAILABLE)

        tensorrt_tooltip = (
            "Enable NVIDIA TensorRT for 2-5× faster YOLO inference.\n"
            "Requires NVIDIA GPU with CUDA.\n"
            "First run will export model (1-5 min), then cached for future use.\n"
            "Uses FP16 precision for maximum speed.\n"
        )
        if TENSORRT_AVAILABLE:
            tensorrt_tooltip += "\n✓ TensorRT is available on this system"
        else:
            tensorrt_tooltip += (
                "\n✗ TensorRT not available (requires NVIDIA GPU + tensorrt package)"
            )

        self.chk_enable_tensorrt.setToolTip(tensorrt_tooltip)
        self.chk_enable_tensorrt.stateChanged.connect(self._on_tensorrt_toggled)
        f_gpu.addRow("", self.chk_enable_tensorrt)

        self.spin_tensorrt_batch = QSpinBox()
        self.spin_tensorrt_batch.setRange(1, 64)
        self.spin_tensorrt_batch.setValue(
            self.advanced_config.get("tensorrt_max_batch_size", 16)
        )
        self.spin_tensorrt_batch.setToolTip(
            "Maximum batch size for TensorRT engine.\n"
            "Higher = potentially faster, Lower = more stable.\n"
            "Reduce if build fails (try 8, 4, or 1).\n"
            "Typical: 16-32 (high-end), 8-16 (mid-range), 1-8 (low VRAM)"
        )
        self.lbl_tensorrt_batch = QLabel("Maximum batch size")
        f_gpu.addRow(self.lbl_tensorrt_batch, self.spin_tensorrt_batch)

        # GPU Batching
        f_gpu.addRow(QLabel(""))  # Spacer

        self.chk_enable_yolo_batching = QCheckBox(
            "Enable Batched Detection (Full Tracking Only)"
        )
        self.chk_enable_yolo_batching.setChecked(
            self.advanced_config.get("enable_yolo_batching", True)
        )
        self.chk_enable_yolo_batching.setToolTip(
            "Process frames in batches on GPU for 2-5× faster detection.\n"
            "Only works in full tracking mode (not preview)."
        )
        self.chk_enable_yolo_batching.stateChanged.connect(
            self._on_yolo_batching_toggled
        )
        f_gpu.addRow("", self.chk_enable_yolo_batching)

        self.combo_yolo_batch_mode = QComboBox()
        self.combo_yolo_batch_mode.addItems(["Auto", "Manual"])
        self.combo_yolo_batch_mode.setToolTip(
            "Auto: Automatically estimate batch size based on GPU memory.\n"
            "Manual: Specify a fixed batch size."
        )
        self.combo_yolo_batch_mode.currentIndexChanged.connect(
            self._on_yolo_batch_mode_changed
        )
        self.lbl_yolo_batch_mode = QLabel("Batch size mode")
        f_gpu.addRow(self.lbl_yolo_batch_mode, self.combo_yolo_batch_mode)

        self.spin_yolo_batch_size = QSpinBox()
        self.spin_yolo_batch_size.setRange(1, 64)
        self.spin_yolo_batch_size.setValue(
            self.advanced_config.get("yolo_manual_batch_size", 16)
        )
        self.spin_yolo_batch_size.setToolTip(
            "Manual batch size (only used when mode is Manual).\n"
            "Larger = faster but uses more GPU memory.\n"
            "Typical values: 8-32 depending on GPU."
        )
        self.spin_yolo_batch_size.valueChanged.connect(
            self._on_yolo_manual_batch_size_changed
        )
        self.lbl_yolo_batch_size = QLabel("Manual batch size")
        f_gpu.addRow(self.lbl_yolo_batch_size, self.spin_yolo_batch_size)

        vl_gpu.addLayout(f_gpu)
        l_yolo.addWidget(self.g_gpu_accel)

        # Set initial visibility for TensorRT widgets
        self.chk_enable_tensorrt.setVisible(False)
        self.spin_tensorrt_batch.setVisible(False)
        self.lbl_tensorrt_batch.setVisible(False)

        # Set initial visibility for batching widgets
        initial_batching_enabled = self.chk_enable_yolo_batching.isChecked()
        self.combo_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.spin_yolo_batch_size.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_size.setVisible(initial_batching_enabled)
        self.combo_yolo_batch_mode.setEnabled(initial_batching_enabled)
        self.spin_yolo_batch_size.setEnabled(False)  # Auto mode by default
        l_yolo.addStretch()  # Push YOLO config to top

        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
        self.stack_detection.addWidget(page_yolo)

        vbox.addWidget(self.stack_detection)

        # ============================================================
        # Detection Overlays (moved from Visuals tab)
        # ============================================================
        # Background Subtraction specific overlays
        self.g_overlays_bg = QGroupBox("Which background diagnostics should be shown?")
        v_ov_bg = QVBoxLayout(self.g_overlays_bg)
        v_ov_bg.addWidget(
            self._create_help_label(
                "Debug background subtraction by viewing the foreground mask (detected movement) "
                "and background model (learned static image)."
            )
        )

        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        v_ov_bg.addWidget(self.chk_show_fg)

        self.chk_show_bg = QCheckBox("Show Background Model")
        self.chk_show_bg.setChecked(True)
        v_ov_bg.addWidget(self.chk_show_bg)

        vbox.addWidget(self.g_overlays_bg)

        # YOLO specific overlays
        self.g_overlays_yolo = QGroupBox("Which YOLO diagnostics should be shown?")
        v_ov_yolo = QVBoxLayout(self.g_overlays_yolo)
        v_ov_yolo.addWidget(
            self._create_help_label(
                "Show oriented bounding boxes from YOLO detection. Useful for debugging detection quality "
                "and verifying model performance."
            )
        )

        self.chk_show_yolo_obb = QCheckBox("Show YOLO OBB Detection Boxes")
        self.chk_show_yolo_obb.setChecked(False)
        v_ov_yolo.addWidget(self.chk_show_yolo_obb)

        vbox.addWidget(self.g_overlays_yolo)

        # Initially show/hide based on detection method (will be set properly by combo)
        self.g_overlays_bg.setVisible(True)
        self.g_overlays_yolo.setVisible(False)

        # ============================================================
        # Reference Body Size (Spatial Scale)
        # ============================================================
        g_body_size = QGroupBox("What is the expected animal size?")
        vl_body_size = QVBoxLayout(g_body_size)
        vl_body_size.addWidget(
            self._create_help_label(
                "Define the spatial scale for tracking. This reference size makes all distance/size "
                "parameters portable across videos. Set this BEFORE configuring tracking parameters."
            )
        )
        fl_body = QFormLayout(None)
        fl_body.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_reference_body_size = QDoubleSpinBox()
        self.spin_reference_body_size.setRange(1.0, 500.0)
        self.spin_reference_body_size.setSingleStep(1.0)
        self.spin_reference_body_size.setValue(20.0)
        self.spin_reference_body_size.setDecimals(2)
        self.spin_reference_body_size.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.spin_reference_body_size.setToolTip(
            "Reference animal body diameter in pixels (at resize=1.0).\n"
            "All distance/size parameters are scaled relative to this value."
        )
        self.spin_reference_body_size.valueChanged.connect(self._update_body_size_info)
        fl_body.addRow(
            "What reference body size (px) should be used?",
            self.spin_reference_body_size,
        )

        # Info label showing calculated area
        self.label_body_size_info = QLabel()
        self.label_body_size_info.setStyleSheet(
            "color: #888; font-size: 10px; font-style: italic;"
        )
        fl_body.addRow("", self.label_body_size_info)
        vl_body_size.addLayout(fl_body)
        vbox.addWidget(g_body_size)

        # ============================================================
        # Detection size statistics panel
        # ============================================================
        g_detect_stats = QGroupBox("Should size be estimated from detections?")
        vl_stats = QVBoxLayout(g_detect_stats)
        vl_stats.addWidget(
            self._create_help_label(
                "Workflow for accurate size estimation:\n"
                "1. Configure your detection method above\n"
                "2. Click 'Load Random Frame for Preview' (bottom of page)\n"
                "3. Choose a frame with many animals well-separated\n"
                "4. Click 'Test Detection' to analyze sizes\n"
                "5. Use 'Auto-Set' to apply the recommended body size"
            )
        )

        self.label_detection_stats = QLabel(
            "No detection data yet.\nRun 'Test Detection' to estimate sizes."
        )
        self.label_detection_stats.setStyleSheet(
            "color: #aaa; font-size: 11px; padding: 8px; "
            "background-color: #2a2a2a; border-radius: 4px;"
        )
        self.label_detection_stats.setWordWrap(True)
        vl_stats.addWidget(self.label_detection_stats)

        # Auto-set button
        btn_layout = QHBoxLayout()
        self.btn_auto_set_body_size = QPushButton("Auto-Set Body Size from Median")
        self.btn_auto_set_body_size.clicked.connect(
            self._auto_set_body_size_from_detection
        )
        self.btn_auto_set_body_size.setEnabled(False)
        self.btn_auto_set_body_size.setToolTip(
            "Automatically set reference body size to the median detected diameter"
        )
        btn_layout.addWidget(self.btn_auto_set_body_size)
        btn_layout.addStretch()
        vl_stats.addLayout(btn_layout)

        vbox.addWidget(g_detect_stats)

        # ============================================================
        # Size Filtering
        # ============================================================
        g_size = QGroupBox("Which detections should be kept by size?")
        vl_size = QVBoxLayout(g_size)
        vl_size.addWidget(
            self._create_help_label(
                "Filter detections by size relative to your reference body size. This removes noise (too small) "
                "and erroneous clusters (too large). Most effective when animals are similar size."
            )
        )
        f_size = QFormLayout(None)
        self.chk_size_filtering = QCheckBox("Filter detections by size")
        self.chk_size_filtering.setToolTip(
            "Filter detected objects by area to remove noise and artifacts.\n"
            "Recommended: Enable for cleaner tracking."
        )
        f_size.addRow(self.chk_size_filtering)

        h_sf = QHBoxLayout()
        self.spin_min_object_size = QDoubleSpinBox()
        self.spin_min_object_size.setRange(0.1, 5.0)
        self.spin_min_object_size.setSingleStep(0.1)
        self.spin_min_object_size.setDecimals(2)
        self.spin_min_object_size.setValue(0.3)
        self.spin_min_object_size.setToolTip(
            "Minimum object area as multiple of reference body area.\n"
            "Filters out small noise/artifacts.\n"
            "Recommended: 0.2-0.5× (allows partial occlusion)"
        )
        self.spin_max_object_size = QDoubleSpinBox()
        self.spin_max_object_size.setRange(0.5, 10.0)
        self.spin_max_object_size.setSingleStep(0.1)
        self.spin_max_object_size.setDecimals(2)
        self.spin_max_object_size.setValue(3.0)
        self.spin_max_object_size.setToolTip(
            "Maximum object area as multiple of reference body area.\n"
            "Filters out large clusters or artifacts.\n"
            "Recommended: 2-4× (handles overlapping animals)"
        )
        h_sf.addWidget(QLabel("Min size (body lengths)"))
        h_sf.addWidget(self.spin_min_object_size)
        h_sf.addWidget(QLabel("Max size (body lengths)"))
        h_sf.addWidget(self.spin_max_object_size)
        f_size.addRow(h_sf)
        vl_size.addLayout(f_size)
        vbox.addWidget(g_size)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_tracking_ui(self: object) -> object:
        """Tab 3: Tracking Logic."""
        layout = QVBoxLayout(self.tab_tracking)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # Core Params
        g_core = QGroupBox("How should track continuity be handled?")
        vl_core = QVBoxLayout(g_core)
        vl_core.addWidget(
            self._create_help_label(
                "These control basic track-to-detection matching. Max assignment distance sets how far an animal can "
                "move between frames. Recovery search distance helps reconnect lost tracks."
            )
        )
        f_core = QFormLayout(None)
        f_core.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_max_targets = QSpinBox()
        self.spin_max_targets.setRange(1, 200)
        self.spin_max_targets.setValue(4)
        self.spin_max_targets.setToolTip(
            "Maximum number of animals to track simultaneously (1-200).\n"
            "Set this to the expected number of animals in your video.\n"
            "Higher values use more memory and may slow down processing."
        )
        f_core.addRow("How many animals should be tracked?", self.spin_max_targets)

        self.spin_max_dist = QDoubleSpinBox()
        self.spin_max_dist.setRange(0.1, 20.0)
        self.spin_max_dist.setSingleStep(0.1)
        self.spin_max_dist.setDecimals(2)
        self.spin_max_dist.setValue(1.5)
        self.spin_max_dist.setToolTip(
            "Maximum distance for track-to-detection assignment (×body size).\n"
            "Animals can move at most this distance between frames.\n"
            "Too low = tracks break frequently, Too high = identity swaps.\n"
            "Recommended: 1-2× for normal motion, 3-5× for fast motion."
        )
        f_core.addRow(
            "How far can an animal move between frames (body lengths)?",
            self.spin_max_dist,
        )

        self.spin_continuity_thresh = QDoubleSpinBox()
        self.spin_continuity_thresh.setRange(0.1, 10.0)
        self.spin_continuity_thresh.setSingleStep(0.1)
        self.spin_continuity_thresh.setDecimals(2)
        self.spin_continuity_thresh.setValue(0.5)
        self.spin_continuity_thresh.setToolTip(
            "Search radius for recovering lost tracks (×body size).\n"
            "When a track is lost, looks backward within this distance.\n"
            "Smaller = more conservative recovery (fewer false merges).\n"
            "Recommended: 0.3-1.0×"
        )
        f_core.addRow(
            "Recovery search distance (body lengths)",
            self.spin_continuity_thresh,
        )

        self.chk_enable_backward = QCheckBox("Run reverse pass for better accuracy")
        self.chk_enable_backward.setChecked(True)
        self.chk_enable_backward.setToolTip(
            "Run tracking in reverse (using cached detections) after forward pass to improve accuracy.\n"
            "Forward detections are cached (~10MB/10k frames), then tracking runs backward.\n"
            "No video reversal needed - RAM efficient and faster.\n"
            "Recommended: Enable for best results (takes ~2× time).\n"
            "Disable for faster processing if accuracy is sufficient."
        )
        f_core.addRow("", self.chk_enable_backward)
        vl_core.addLayout(f_core)
        vbox.addWidget(g_core)

        # Create accordion for advanced tracking settings
        self.tracking_accordion = AccordionContainer()

        # Kalman
        g_kf = CollapsibleGroupBox("How should motion prediction behave?")
        self.tracking_accordion.addCollapsible(g_kf)
        vl_kf = QVBoxLayout()
        vl_kf.addWidget(
            self._create_help_label(
                "Kalman filter predicts animal positions using motion history. Process noise controls smoothing, "
                "measurement noise controls responsiveness. Age-dependent damping helps stabilize newly initialized tracks."
            )
        )
        f_kf = QFormLayout(None)
        f_kf.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.spin_kalman_noise = QDoubleSpinBox()
        self.spin_kalman_noise.setRange(0.0, 1.0)
        self.spin_kalman_noise.setValue(0.03)
        self.spin_kalman_noise.setToolTip(
            "Process noise covariance (0.0-1.0) for motion prediction.\n"
            "Lower = trust motion model more (smooth, may lag).\n"
            "Higher = trust measurements more (responsive, less smooth).\n"
            "Note: Optimal value depends on frame rate (time step).\n"
            "Recommended: 0.01-0.05 for predictable motion."
        )
        f_kf.addRow("How smooth should motion prediction be?", self.spin_kalman_noise)

        self.spin_kalman_meas = QDoubleSpinBox()
        self.spin_kalman_meas.setRange(0.0, 1.0)
        self.spin_kalman_meas.setValue(0.1)
        self.spin_kalman_meas.setToolTip(
            "Measurement noise covariance (0.0-1.0).\n"
            "Lower = trust detections more (accurate, may be jittery).\n"
            "Higher = trust predictions more (smooth, may drift).\n"
            "Recommended: 0.05-0.15"
        )
        f_kf.addRow(
            "How strongly should detections override prediction?", self.spin_kalman_meas
        )

        self.spin_kalman_damping = QDoubleSpinBox()
        self.spin_kalman_damping.setRange(0.5, 0.99)
        self.spin_kalman_damping.setSingleStep(0.01)
        self.spin_kalman_damping.setDecimals(2)
        self.spin_kalman_damping.setValue(0.95)
        self.spin_kalman_damping.setToolTip(
            "Velocity damping coefficient (0.5-0.99).\n"
            "Controls how quickly velocity decays each frame.\n"
            "Lower = faster decay (better for stop-and-go behavior).\n"
            "Higher = slower decay (better for continuous motion).\n"
            "Recommended: 0.90-0.95"
        )
        f_kf.addRow(
            "How quickly should estimated speed decay?", self.spin_kalman_damping
        )

        # Age-dependent velocity damping
        age_label = QLabel("How conservative should new tracks be?")
        age_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(age_label)

        self.spin_kalman_maturity_age = QSpinBox()
        self.spin_kalman_maturity_age.setRange(1, 30)
        self.spin_kalman_maturity_age.setValue(5)
        self.spin_kalman_maturity_age.setToolTip(
            "Number of frames for a track to reach maturity (1-30).\n"
            "Young tracks use conservative velocity estimates.\n"
            "After this many successful updates, tracks use full dynamics.\n"
            "Lower = faster adaptation, Higher = more conservative.\n"
            "Recommended: 3-10 frames"
        )
        f_kf.addRow(
            "How many frames until a track is trusted?", self.spin_kalman_maturity_age
        )

        self.spin_kalman_initial_velocity_retention = QDoubleSpinBox()
        self.spin_kalman_initial_velocity_retention.setRange(0.0, 1.0)
        self.spin_kalman_initial_velocity_retention.setSingleStep(0.05)
        self.spin_kalman_initial_velocity_retention.setDecimals(2)
        self.spin_kalman_initial_velocity_retention.setValue(0.2)
        self.spin_kalman_initial_velocity_retention.setToolTip(
            "Initial velocity retention for brand new tracks (0.0-1.0).\n"
            "0.0 = assume stationary (no velocity)\n"
            "1.0 = use full velocity estimate\n"
            "Gradually increases to 1.0 as track ages to maturity.\n"
            "Lower = more conservative (prevents wild predictions).\n"
            "Recommended: 0.1-0.3"
        )
        f_kf.addRow(
            "How much initial speed should new tracks keep?",
            self.spin_kalman_initial_velocity_retention,
        )

        self.spin_kalman_max_velocity = QDoubleSpinBox()
        self.spin_kalman_max_velocity.setRange(0.5, 10.0)
        self.spin_kalman_max_velocity.setSingleStep(0.1)
        self.spin_kalman_max_velocity.setDecimals(1)
        self.spin_kalman_max_velocity.setValue(2.0)
        self.spin_kalman_max_velocity.setToolTip(
            "Maximum velocity constraint (body size multiplier).\n"
            "Prevents unrealistic predictions during occlusions.\n"
            "velocity_max = this_value × reference_body_size (pixels/frame)\n"
            "Lower = more conservative, Higher = allows faster movement.\n"
            "Recommended: 1.5-3.0 depending on animal speed"
        )
        f_kf.addRow(
            "What maximum speed should prediction allow (body lengths/frame)?",
            self.spin_kalman_max_velocity,
        )

        # Anisotropic process noise
        aniso_label = QLabel("Should forward and sideways uncertainty differ?")
        aniso_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        f_kf.addRow(aniso_label)

        self.spin_kalman_longitudinal_noise = QDoubleSpinBox()
        self.spin_kalman_longitudinal_noise.setRange(0.1, 20.0)
        self.spin_kalman_longitudinal_noise.setSingleStep(0.5)
        self.spin_kalman_longitudinal_noise.setDecimals(1)
        self.spin_kalman_longitudinal_noise.setValue(5.0)
        self.spin_kalman_longitudinal_noise.setToolTip(
            "Forward/longitudinal noise multiplier (0.1-20.0).\n"
            "Controls uncertainty in direction of movement.\n"
            "Higher = more uncertainty forward (smoother forward motion).\n"
            "Multiplies base process noise for forward direction.\n"
            "Recommended: 3.0-7.0"
        )
        f_kf.addRow(
            "How much uncertainty along movement direction?",
            self.spin_kalman_longitudinal_noise,
        )

        self.spin_kalman_lateral_noise = QDoubleSpinBox()
        self.spin_kalman_lateral_noise.setRange(0.01, 5.0)
        self.spin_kalman_lateral_noise.setSingleStep(0.05)
        self.spin_kalman_lateral_noise.setDecimals(2)
        self.spin_kalman_lateral_noise.setValue(0.1)
        self.spin_kalman_lateral_noise.setToolTip(
            "Sideways/lateral noise multiplier (0.01-5.0).\n"
            "Controls uncertainty perpendicular to movement.\n"
            "Lower = less uncertainty sideways (constrains lateral drift).\n"
            "Multiplies base process noise for sideways direction.\n"
            "Recommended: 0.05-0.2"
        )
        f_kf.addRow("How much uncertainty sideways?", self.spin_kalman_lateral_noise)

        vl_kf.addLayout(f_kf)
        g_kf.setContentLayout(vl_kf)
        vbox.addWidget(g_kf)

        # Weights
        g_weights = CollapsibleGroupBox("What should matching prioritize?")
        self.tracking_accordion.addCollapsible(g_weights)
        l_weights = QVBoxLayout()
        l_weights.addWidget(
            self._create_help_label(
                "Control how different factors influence track-to-detection matching. Position is primary; orientation, "
                "area, and aspect help resolve ambiguities. Increase Mahalanobis to trust Kalman predictions more."
            )
        )

        row1 = QHBoxLayout()
        self.spin_Wp = QDoubleSpinBox()
        self.spin_Wp.setRange(0.0, 10.0)
        self.spin_Wp.setValue(1.0)
        self.spin_Wp.setToolTip(
            "Weight for position distance in assignment cost.\n"
            "Higher = prioritize spatial proximity.\n"
            "Recommended: 1.0 (primary factor)"
        )
        row1.addWidget(QLabel("Position importance"))
        row1.addWidget(self.spin_Wp)

        self.spin_Wo = QDoubleSpinBox()
        self.spin_Wo.setRange(0.0, 10.0)
        self.spin_Wo.setValue(1.0)
        self.spin_Wo.setToolTip(
            "Weight for orientation difference in assignment cost.\n"
            "Higher = penalize large orientation changes.\n"
            "Recommended: 0.5-2.0 (helps maintain correct identity)"
        )
        row1.addWidget(QLabel("Direction importance"))
        row1.addWidget(self.spin_Wo)
        l_weights.addLayout(row1)

        row2 = QHBoxLayout()
        self.spin_Wa = QDoubleSpinBox()
        self.spin_Wa.setRange(0.0, 1.0)
        self.spin_Wa.setSingleStep(0.001)
        self.spin_Wa.setDecimals(4)
        self.spin_Wa.setValue(0.001)
        self.spin_Wa.setToolTip(
            "Weight for area difference in assignment cost.\n"
            "Higher = penalize size changes.\n"
            "Recommended: 0.001-0.01 (prevents size-based swaps)"
        )
        row2.addWidget(QLabel("Size importance"))
        row2.addWidget(self.spin_Wa)

        self.spin_Wasp = QDoubleSpinBox()
        self.spin_Wasp.setRange(0.0, 10.0)
        self.spin_Wasp.setValue(0.1)
        self.spin_Wasp.setToolTip(
            "Weight for aspect ratio difference in assignment cost.\n"
            "Higher = penalize shape changes.\n"
            "Recommended: 0.05-0.2 (helps with occlusions)"
        )
        row2.addWidget(QLabel("Shape importance"))
        row2.addWidget(self.spin_Wasp)
        l_weights.addLayout(row2)

        self.chk_use_mahal = QCheckBox("Use motion-aware distance")
        self.chk_use_mahal.setChecked(True)
        self.chk_use_mahal.setToolTip(
            "Use Mahalanobis distance instead of Euclidean for position.\n"
            "Accounts for velocity and uncertainty in motion prediction.\n"
            "Recommended: Enable for better handling of motion variability."
        )
        l_weights.addWidget(self.chk_use_mahal)
        g_weights.setContentLayout(l_weights)
        vbox.addWidget(g_weights)

        # Assignment Algorithm (for large N optimization)
        g_assign = CollapsibleGroupBox("How should matches be computed at scale?")
        self.tracking_accordion.addCollapsible(g_assign)
        vl_assign = QVBoxLayout()
        vl_assign.addWidget(
            self._create_help_label(
                "Choose matching algorithm. Hungarian is optimal but slow for many animals (N>100). "
                "Greedy approximation is faster but may produce suboptimal assignments."
            )
        )
        f_assign = QFormLayout(None)

        self.combo_assignment_method = QComboBox()
        self.combo_assignment_method.addItems(
            ["Most accurate (slower)", "Fast approximate (large groups)"]
        )
        self.combo_assignment_method.setCurrentIndex(0)
        self.combo_assignment_method.setToolTip(
            "Hungarian: Optimal global assignment (slow for N>100)\n"
            "Greedy: Fast approximation for large N (200+)"
        )
        f_assign.addRow("Which method should be used?", self.combo_assignment_method)

        self.chk_spatial_optimization = QCheckBox("Speed up matching for many animals")
        self.chk_spatial_optimization.setChecked(False)
        self.chk_spatial_optimization.setToolTip(
            "Uses KD-tree to reduce comparisons for large N (50+).\n"
            "Disable for small N (8-50) to reduce overhead."
        )
        f_assign.addRow(self.chk_spatial_optimization)

        vl_assign.addLayout(f_assign)
        g_assign.setContentLayout(vl_assign)
        vbox.addWidget(g_assign)

        # Orientation & Lifecycle
        g_misc = CollapsibleGroupBox("How should direction changes be handled?")
        self.tracking_accordion.addCollapsible(g_misc)
        vl_misc = QVBoxLayout()
        vl_misc.addWidget(
            self._create_help_label(
                "Control how orientation is calculated based on movement. Moving animals can flip orientation instantly, "
                "stationary animals change orientation gradually within max angle limit."
            )
        )
        f_misc = QFormLayout(None)

        self.spin_velocity = QDoubleSpinBox()
        self.spin_velocity.setRange(0.1, 100.0)
        self.spin_velocity.setSingleStep(0.5)
        self.spin_velocity.setDecimals(2)
        self.spin_velocity.setValue(5.0)
        self.spin_velocity.setToolTip(
            "Velocity threshold (body-sizes/second) to classify as 'moving'.\n"
            "Below this = stationary (allows larger orientation changes).\n"
            "Above this = moving (instant orientation flip possible).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 2-10 body-sizes/s depending on animal speed."
        )
        f_misc.addRow("Moving-speed threshold (body lengths/sec)", self.spin_velocity)

        self.chk_instant_flip = QCheckBox(
            "Allow instant direction flips when moving fast"
        )
        self.chk_instant_flip.setChecked(True)
        self.chk_instant_flip.setToolTip(
            "Allow instant 180° orientation flip when moving quickly.\n"
            "Recommended: Enable for animals that can turn rapidly.\n"
            "Disable for slowly rotating animals."
        )
        f_misc.addRow(self.chk_instant_flip)

        self.spin_max_orient = QDoubleSpinBox()
        self.spin_max_orient.setRange(1, 180)
        self.spin_max_orient.setValue(30)
        self.spin_max_orient.setToolTip(
            "Maximum orientation change (degrees) when stationary (1-180).\n"
            "Larger = allow more rotation while stopped.\n"
            "Recommended: 20-45° (prevents orientation jitter)."
        )
        f_misc.addRow(
            "Max direction change while stopped (degrees)", self.spin_max_orient
        )
        vl_misc.addLayout(f_misc)
        g_misc.setContentLayout(vl_misc)
        vbox.addWidget(g_misc)

        # Track Lifecycle
        g_lifecycle = CollapsibleGroupBox("When should tracks start and end?")
        self.tracking_accordion.addCollapsible(g_lifecycle)
        vl_lifecycle = QVBoxLayout()
        vl_lifecycle.addWidget(
            self._create_help_label(
                "Control when tracks start and end. Lost frames determines how long to wait before terminating a track. "
                "Min respawn distance prevents creating duplicate IDs near existing animals."
            )
        )
        f_lifecycle = QFormLayout(None)

        self.spin_lost_thresh = QSpinBox()
        self.spin_lost_thresh.setRange(1, 100)
        self.spin_lost_thresh.setValue(10)
        self.spin_lost_thresh.setToolTip(
            "Number of frames without detection before track is terminated (1-100).\n"
            "Higher = tracks persist longer during occlusions.\n"
            "Lower = tracks end quickly, creating fragments.\n"
            "Recommended: 5-20 frames."
        )
        f_lifecycle.addRow(
            "How long to keep a track without detections (frames)?",
            self.spin_lost_thresh,
        )

        self.spin_min_respawn_distance = QDoubleSpinBox()
        self.spin_min_respawn_distance.setRange(0.0, 20.0)
        self.spin_min_respawn_distance.setSingleStep(0.5)
        self.spin_min_respawn_distance.setDecimals(2)
        self.spin_min_respawn_distance.setValue(2.5)
        self.spin_min_respawn_distance.setToolTip(
            "Minimum distance from existing tracks to spawn new track (×body size).\n"
            "Prevents creating duplicate tracks near existing animals.\n"
            "Recommended: 2-4× body size."
        )
        f_lifecycle.addRow(
            "How far from existing tracks to start a new one (body lengths)?",
            self.spin_min_respawn_distance,
        )
        vl_lifecycle.addLayout(f_lifecycle)
        g_lifecycle.setContentLayout(vl_lifecycle)
        vbox.addWidget(g_lifecycle)

        # Stability
        g_stab = CollapsibleGroupBox("How strict should new track validation be?")
        self.tracking_accordion.addCollapsible(g_stab)
        vl_stab = QVBoxLayout()
        vl_stab.addWidget(
            self._create_help_label(
                "Filter out unreliable tracks. Min detections to start prevents creating tracks from noise. "
                "Min detect/tracking frames removes short-lived false tracks in post-processing."
            )
        )
        f_stab = QFormLayout(None)
        self.spin_min_detections_to_start = QSpinBox()
        self.spin_min_detections_to_start.setRange(1, 50)
        self.spin_min_detections_to_start.setValue(1)
        self.spin_min_detections_to_start.setToolTip(
            "Minimum consecutive detections before starting a new track (1-50).\n"
            "Higher = fewer false tracks from noise, slower to start tracking.\n"
            "Lower = faster tracking startup, more noise-based tracks.\n"
            "Recommended: 1-3"
        )
        f_stab.addRow(
            "How many detections before starting a new track?",
            self.spin_min_detections_to_start,
        )

        self.spin_min_detect = QSpinBox()
        self.spin_min_detect.setRange(1, 500)
        self.spin_min_detect.setValue(10)
        self.spin_min_detect.setToolTip(
            "Minimum total detection frames to keep a track (1-500).\n"
            "Filters out short-lived false tracks in post-processing.\n"
            "Recommended: 5-20 frames."
        )
        f_stab.addRow("Minimum detection frames to keep a track", self.spin_min_detect)

        self.spin_min_track = QSpinBox()
        self.spin_min_track.setRange(1, 500)
        self.spin_min_track.setValue(10)
        self.spin_min_track.setToolTip(
            "Minimum tracking frames (including predicted) to keep (1-500).\n"
            "Filters out tracks with too many gaps/predictions.\n"
            "Recommended: Similar to min detect frames."
        )
        f_stab.addRow("Minimum total frames to keep a track", self.spin_min_track)
        vl_stab.addLayout(f_stab)
        g_stab.setContentLayout(vl_stab)
        vbox.addWidget(g_stab)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_data_ui(self: object) -> object:
        """Tab 4: Post-Processing."""
        layout = QVBoxLayout(self.tab_data)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)

        # Post-Processing
        g_pp = QGroupBox("How should tracks be cleaned after tracking?")
        vl_pp = QVBoxLayout(g_pp)
        vl_pp.addWidget(
            self._create_help_label(
                "Clean trajectories after tracking by removing outliers and splitting at identity swaps. "
                "Velocity/distance breaks detect unrealistic jumps that indicate ID switching."
            )
        )
        f_pp = QFormLayout(None)
        f_pp.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.enable_postprocessing = QCheckBox("Auto-clean trajectories")
        self.enable_postprocessing.setChecked(True)
        self.enable_postprocessing.setToolTip(
            "Automatically clean trajectories by removing outliers and fragments.\n"
            "Uses velocity and distance thresholds to detect anomalies.\n"
            "Recommended: Enable for cleaner data output."
        )
        self.enable_postprocessing.stateChanged.connect(self._on_cleaning_toggled)
        f_pp.addRow(self.enable_postprocessing)

        self.spin_min_trajectory_length = QSpinBox()
        self.spin_min_trajectory_length.setRange(1, 1000)
        self.spin_min_trajectory_length.setValue(10)
        self.spin_min_trajectory_length.setToolTip(
            "Remove trajectories shorter than this (1-1000 frames).\n"
            "Filters out brief false detections and transient tracks.\n"
            "Recommended: 5-30 frames depending on video length."
        )
        self.lbl_min_trajectory_length = QLabel("Minimum trajectory length (frames)")
        f_pp.addRow(self.lbl_min_trajectory_length, self.spin_min_trajectory_length)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1.0, 500.0)
        self.spin_max_velocity_break.setSingleStep(5.0)
        self.spin_max_velocity_break.setDecimals(1)
        self.spin_max_velocity_break.setValue(50.0)
        self.spin_max_velocity_break.setToolTip(
            "Maximum velocity (body-sizes/second) before breaking trajectory.\n"
            "Splits tracks at unrealistic speed jumps (likely identity swaps).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 30-100 body-sizes/s for typical animal motion."
        )
        self.lbl_max_velocity_break = QLabel(
            "Break trajectory above speed (body lengths/sec)"
        )
        f_pp.addRow(self.lbl_max_velocity_break, self.spin_max_velocity_break)

        self.spin_max_occlusion_gap = QSpinBox()
        self.spin_max_occlusion_gap.setRange(0, 200)
        self.spin_max_occlusion_gap.setValue(30)
        self.spin_max_occlusion_gap.setToolTip(
            "Maximum consecutive occluded/lost frames before splitting trajectory (0-200).\n"
            "Prevents unreliable interpolation across long gaps.\n"
            "Set to 0 to disable occlusion-based splitting.\n"
            "Recommended: 20-50 frames for typical tracking scenarios."
        )
        self.lbl_max_occlusion_gap = QLabel("Maximum occlusion gap (frames)")
        f_pp.addRow(self.lbl_max_occlusion_gap, self.spin_max_occlusion_gap)

        # Z-score based velocity breaking
        self.spin_max_velocity_zscore = QDoubleSpinBox()
        self.spin_max_velocity_zscore.setRange(0.0, 10.0)
        self.spin_max_velocity_zscore.setSingleStep(0.5)
        self.spin_max_velocity_zscore.setDecimals(1)
        self.spin_max_velocity_zscore.setValue(0.0)  # 0 = disabled
        self.spin_max_velocity_zscore.setToolTip(
            "Z-score threshold for velocity-based trajectory breaking (0 = disabled).\n"
            "Detects sudden, statistically anomalous velocity changes that often\n"
            "indicate identity swaps or tracking errors.\n\n"
            "Safeguards prevent false breaks when animals transition from rest to movement:\n"
            "• Only triggers on substantial velocities (>2 px/frame)\n"
            "• Uses regularized statistics to handle low-variability periods\n"
            "• Filters out stationary noise from baseline calculations\n\n"
            "Recommended: 3.0-5.0 for sensitive detection, 0 to disable."
        )
        self.lbl_max_velocity_zscore = QLabel("Velocity z-score threshold")
        f_pp.addRow(self.lbl_max_velocity_zscore, self.spin_max_velocity_zscore)

        self.spin_velocity_zscore_window = QSpinBox()
        self.spin_velocity_zscore_window.setRange(5, 50)
        self.spin_velocity_zscore_window.setValue(10)
        self.spin_velocity_zscore_window.setToolTip(
            "Number of past velocities to use for z-score calculation (5-50 frames).\n"
            "Larger windows = more stable statistics but less responsive to changes.\n"
            "Smaller windows = more sensitive but may be noisy.\n"
            "Recommended: 10-20 frames."
        )
        self.lbl_velocity_zscore_window = QLabel("Z-score window (frames)")
        f_pp.addRow(self.lbl_velocity_zscore_window, self.spin_velocity_zscore_window)

        self.spin_velocity_zscore_min_vel = QDoubleSpinBox()
        self.spin_velocity_zscore_min_vel.setRange(0.1, 50.0)
        self.spin_velocity_zscore_min_vel.setSingleStep(0.5)
        self.spin_velocity_zscore_min_vel.setDecimals(1)
        self.spin_velocity_zscore_min_vel.setValue(2.0)
        self.spin_velocity_zscore_min_vel.setToolTip(
            "Minimum velocity for z-score breaking (body-sizes/second).\n"
            "Prevents false breaks when animal starts moving from stationary state.\n"
            "Z-score analysis only triggers when velocity exceeds this threshold.\n"
            "Automatically scaled by body size and frame rate.\n"
            "Recommended: 1.0-3.0 body-sizes/s depending on animal locomotion speed."
        )
        self.lbl_velocity_zscore_min_vel = QLabel(
            "Minimum speed for z-score check (body lengths/sec)"
        )
        f_pp.addRow(self.lbl_velocity_zscore_min_vel, self.spin_velocity_zscore_min_vel)

        # Interpolation settings
        self.combo_interpolation_method = QComboBox()
        self.combo_interpolation_method.addItems(["None", "Linear", "Cubic", "Spline"])
        self.combo_interpolation_method.setCurrentText("None")
        self.combo_interpolation_method.setToolTip(
            "Interpolation method for filling gaps in trajectories:\n"
            "• None: No interpolation (keep NaN values)\n"
            "• Linear: Simple linear interpolation\n"
            "• Cubic: Smooth cubic spline interpolation\n"
            "• Spline: Smoothing spline with automatic smoothing\n"
            "Applied to X, Y positions and heading (circular interpolation)."
        )
        self.lbl_interpolation_method = QLabel(
            "Which interpolation method should be used?"
        )
        f_pp.addRow(self.lbl_interpolation_method, self.combo_interpolation_method)

        self.spin_interpolation_max_gap = QSpinBox()
        self.spin_interpolation_max_gap.setRange(1, 100)
        self.spin_interpolation_max_gap.setValue(10)
        self.spin_interpolation_max_gap.setToolTip(
            "Maximum gap size to interpolate (1-100 frames).\n"
            "Gaps larger than this will remain as NaN.\n"
            "Prevents interpolation across large occlusions.\n"
            "Recommended: 5-15 frames."
        )
        self.lbl_interpolation_max_gap = QLabel("Maximum interpolation gap (frames)")
        f_pp.addRow(self.lbl_interpolation_max_gap, self.spin_interpolation_max_gap)

        # Trajectory Merging Settings (Conservative Strategy)
        self.spin_merge_overlap_multiplier = QDoubleSpinBox()
        self.spin_merge_overlap_multiplier.setRange(0.1, 10.0)
        self.spin_merge_overlap_multiplier.setSingleStep(0.1)
        self.spin_merge_overlap_multiplier.setDecimals(2)
        self.spin_merge_overlap_multiplier.setValue(0.5)
        self.spin_merge_overlap_multiplier.setToolTip(
            "Agreement distance for merging forward/backward trajectories (×body size).\n"
            "Frames where both trajectories are within this distance are considered 'agreeing'.\n"
            "Disagreeing frames cause trajectory splits for conservative identity handling.\n"
            "Recommended: 0.3-0.7× body size."
        )
        self.lbl_merge_overlap_multiplier = QLabel(
            "Merge agreement distance (body lengths)"
        )
        f_pp.addRow(
            self.lbl_merge_overlap_multiplier, self.spin_merge_overlap_multiplier
        )

        self.spin_min_overlap_frames = QSpinBox()
        self.spin_min_overlap_frames.setRange(1, 100)
        self.spin_min_overlap_frames.setValue(5)
        self.spin_min_overlap_frames.setToolTip(
            "Minimum agreeing frames required to consider trajectories as merge candidates.\n"
            "Forward/backward trajectory pairs need at least this many frames within\n"
            "the agreement distance to be merged. Higher = more conservative.\n"
            "Recommended: 5-15 frames."
        )
        self.lbl_min_overlap_frames = QLabel("Minimum overlap frames")
        f_pp.addRow(self.lbl_min_overlap_frames, self.spin_min_overlap_frames)

        # Cleanup option
        self.chk_cleanup_temp_files = QCheckBox("Auto-cleanup temporary files")
        self.chk_cleanup_temp_files.setChecked(True)
        self.chk_cleanup_temp_files.setToolTip(
            "Automatically delete temporary files after successful tracking:\n"
            "• Intermediate CSV files (*_forward.csv, *_backward.csv)\n"
            "• Pose inference cache (posekit/ directory)\n"
            "Keeps only final merged/processed output files."
        )
        f_pp.addRow("", self.chk_cleanup_temp_files)

        vl_pp.addLayout(f_pp)
        vbox.addWidget(g_pp)

        # Video Export (from post-processed trajectories)
        g_video = QGroupBox("What export video should be created?")
        vl_video = QVBoxLayout(g_video)
        vl_video.addWidget(
            self._create_help_label(
                "Generate annotated video from final post-processed trajectories. "
                "Video is created AFTER merging and interpolation, showing clean tracks with stable IDs. "
                "This is recommended over real-time video output during tracking."
            )
        )
        f_video = QFormLayout(None)
        f_video.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.check_video_output = QCheckBox("Export trajectory video")
        self.check_video_output.setChecked(False)
        self.check_video_output.toggled.connect(self._on_video_output_toggled)
        self.check_video_output.setToolTip(
            "Generate annotated video showing post-processed trajectories.\n"
            "Video is created from merged/interpolated tracks, not raw tracking.\n"
            "Shows clean, stable trajectories with final IDs.\n"
            "Recommended for publication and visualization."
        )
        f_video.addRow("", self.check_video_output)

        self.btn_video_out = QPushButton("Select Video Output...")
        self.btn_video_out.clicked.connect(self.select_video_output)
        self.btn_video_out.setEnabled(False)
        self.video_out_line = QLineEdit()
        self.video_out_line.setPlaceholderText("Path for annotated video output")
        self.video_out_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_out_line.setEnabled(False)
        self.lbl_video_path = QLabel("")
        f_video.addRow(self.lbl_video_path, self.btn_video_out)
        f_video.addRow("", self.video_out_line)

        # Video Visualization Settings
        f_video.addRow(QLabel(""))  # Spacer
        self.lbl_video_viz_settings = QLabel("<b>Visualization Settings</b>")
        f_video.addRow(self.lbl_video_viz_settings)

        self.check_show_labels = QCheckBox("Show Track IDs")
        self.check_show_labels.setChecked(True)
        self.check_show_labels.setToolTip(
            "Display trajectory ID labels next to each tracked animal."
        )
        f_video.addRow("", self.check_show_labels)

        self.check_show_orientation = QCheckBox("Show Orientation Arrows")
        self.check_show_orientation.setChecked(True)
        self.check_show_orientation.setToolTip(
            "Display arrows indicating heading direction."
        )
        f_video.addRow("", self.check_show_orientation)

        self.check_show_trails = QCheckBox("Show Trajectory Trails")
        self.check_show_trails.setChecked(False)
        self.check_show_trails.setToolTip(
            "Display past trajectory path as a fading trail."
        )
        f_video.addRow("", self.check_show_trails)

        self.spin_trail_duration = QDoubleSpinBox()
        self.spin_trail_duration.setRange(0.1, 10.0)
        self.spin_trail_duration.setSingleStep(0.5)
        self.spin_trail_duration.setDecimals(1)
        self.spin_trail_duration.setValue(1.0)
        self.spin_trail_duration.setToolTip(
            "Duration of trail history in seconds (0.1-10.0).\n"
            "Longer trails show more movement history.\n"
            "Automatically converted to frames using video FPS."
        )
        self.lbl_trail_duration = QLabel("Trail duration (seconds)")
        f_video.addRow(self.lbl_trail_duration, self.spin_trail_duration)

        self.spin_marker_size = QDoubleSpinBox()
        self.spin_marker_size.setRange(0.1, 300.0)
        self.spin_marker_size.setSingleStep(0.1)
        self.spin_marker_size.setDecimals(1)
        self.spin_marker_size.setValue(0.3)
        self.spin_marker_size.setToolTip(
            "Size of position marker (0.1-5.0 × body size).\n"
            "Scaled by reference body size for consistency."
        )
        self.lbl_marker_size = QLabel("Marker size (body lengths)")
        f_video.addRow(self.lbl_marker_size, self.spin_marker_size)

        self.spin_text_scale = QDoubleSpinBox()
        self.spin_text_scale.setRange(0.3, 3.0)
        self.spin_text_scale.setSingleStep(0.1)
        self.spin_text_scale.setDecimals(1)
        self.spin_text_scale.setValue(0.5)
        self.spin_text_scale.setToolTip(
            "Scale factor for ID labels (0.3-3.0).\n" "Larger values = bigger text."
        )
        self.lbl_text_scale = QLabel("Text scale")
        f_video.addRow(self.lbl_text_scale, self.spin_text_scale)

        self.spin_arrow_length = QDoubleSpinBox()
        self.spin_arrow_length.setRange(0.5, 10.0)
        self.spin_arrow_length.setSingleStep(0.5)
        self.spin_arrow_length.setDecimals(1)
        self.spin_arrow_length.setValue(0.7)
        self.spin_arrow_length.setToolTip(
            "Length of orientation arrow (0.5-10.0 × body size).\n"
            "Scaled by reference body size."
        )
        self.lbl_arrow_length = QLabel("Arrow length (body lengths)")
        f_video.addRow(self.lbl_arrow_length, self.spin_arrow_length)

        f_video.addRow(QLabel(""))  # Spacer
        self.lbl_video_pose_settings = QLabel("<b>Pose Overlay Settings</b>")
        f_video.addRow(self.lbl_video_pose_settings)

        self.check_video_show_pose = QCheckBox("Show Pose Keypoints/Skeleton")
        self.check_video_show_pose.setChecked(
            bool(self.advanced_config.get("video_show_pose", True))
        )
        self.check_video_show_pose.setToolTip(
            "Overlay pose keypoints/skeleton in exported video.\n"
            "Requires pose inference to be enabled in Analyze Individuals."
        )
        self.check_video_show_pose.toggled.connect(
            self._sync_video_pose_overlay_controls
        )
        f_video.addRow("", self.check_video_show_pose)

        self.combo_video_pose_color_mode = QComboBox()
        self.combo_video_pose_color_mode.addItems(["Track Color", "Fixed Color"])
        color_mode = str(
            self.advanced_config.get("video_pose_color_mode", "track")
        ).strip()
        self.combo_video_pose_color_mode.setCurrentIndex(
            0 if color_mode == "track" else 1
        )
        self.combo_video_pose_color_mode.setToolTip(
            "Pose color source for video overlay."
        )
        self.combo_video_pose_color_mode.currentIndexChanged.connect(
            self._sync_video_pose_overlay_controls
        )
        self.lbl_video_pose_color_mode = QLabel("Pose color mode")
        f_video.addRow(self.lbl_video_pose_color_mode, self.combo_video_pose_color_mode)

        pose_color_row = QHBoxLayout()
        self.btn_video_pose_color = QPushButton()
        self.btn_video_pose_color.setMaximumWidth(60)
        self.btn_video_pose_color.setMinimumHeight(28)
        self.btn_video_pose_color.clicked.connect(self._select_video_pose_color)
        self.lbl_video_pose_color = QLabel("")
        pose_color_row.addWidget(self.btn_video_pose_color)
        pose_color_row.addWidget(self.lbl_video_pose_color)
        pose_color_row.addStretch()
        pose_color_cfg = self.advanced_config.get("video_pose_color", [255, 255, 255])
        if isinstance(pose_color_cfg, (list, tuple)) and len(pose_color_cfg) == 3:
            self._video_pose_color = tuple(
                int(max(0, min(255, float(v)))) for v in pose_color_cfg
            )
        else:
            self._video_pose_color = (255, 255, 255)
        self._update_video_pose_color_button()
        self.lbl_video_pose_color_label = QLabel("Fixed pose color (BGR)")
        f_video.addRow(self.lbl_video_pose_color_label, pose_color_row)

        self.spin_video_pose_point_radius = QSpinBox()
        self.spin_video_pose_point_radius.setRange(1, 20)
        self.spin_video_pose_point_radius.setValue(
            int(self.advanced_config.get("video_pose_point_radius", 3))
        )
        self.spin_video_pose_point_radius.setToolTip(
            "Radius of rendered pose keypoints in pixels."
        )
        self.lbl_video_pose_point_radius = QLabel("Pose keypoint radius (px)")
        f_video.addRow(
            self.lbl_video_pose_point_radius, self.spin_video_pose_point_radius
        )

        self.spin_video_pose_point_thickness = QSpinBox()
        self.spin_video_pose_point_thickness.setRange(-1, 10)
        self.spin_video_pose_point_thickness.setValue(
            int(self.advanced_config.get("video_pose_point_thickness", -1))
        )
        self.spin_video_pose_point_thickness.setToolTip(
            "Keypoint circle thickness (-1 fills circles)."
        )
        self.lbl_video_pose_point_thickness = QLabel("Pose keypoint thickness")
        f_video.addRow(
            self.lbl_video_pose_point_thickness, self.spin_video_pose_point_thickness
        )

        self.spin_video_pose_line_thickness = QSpinBox()
        self.spin_video_pose_line_thickness.setRange(1, 12)
        self.spin_video_pose_line_thickness.setValue(
            int(self.advanced_config.get("video_pose_line_thickness", 2))
        )
        self.spin_video_pose_line_thickness.setToolTip(
            "Skeleton edge line thickness in pixels."
        )
        self.lbl_video_pose_line_thickness = QLabel("Pose skeleton thickness (px)")
        f_video.addRow(
            self.lbl_video_pose_line_thickness, self.spin_video_pose_line_thickness
        )

        self.lbl_video_pose_disabled_hint = self._create_help_label(
            "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings."
        )
        f_video.addRow("", self.lbl_video_pose_disabled_hint)

        vl_video.addLayout(f_video)
        vbox.addWidget(g_video)

        # Set initial visibility for video export widgets (hidden since checkbox starts unchecked)
        self.btn_video_out.setVisible(False)
        self.video_out_line.setVisible(False)
        self.lbl_video_path.setVisible(False)
        self.lbl_video_viz_settings.setVisible(False)
        self.check_show_labels.setVisible(False)
        self.check_show_orientation.setVisible(False)
        self.check_show_trails.setVisible(False)
        self.spin_trail_duration.setVisible(False)
        self.lbl_trail_duration.setVisible(False)
        self.spin_marker_size.setVisible(False)
        self.lbl_marker_size.setVisible(False)
        self.spin_text_scale.setVisible(False)
        self.lbl_text_scale.setVisible(False)
        self.spin_arrow_length.setVisible(False)
        self.lbl_arrow_length.setVisible(False)
        self.lbl_video_pose_settings.setVisible(False)
        self.check_video_show_pose.setVisible(False)
        self.lbl_video_pose_color_mode.setVisible(False)
        self.combo_video_pose_color_mode.setVisible(False)
        self.lbl_video_pose_color_label.setVisible(False)
        self.btn_video_pose_color.setVisible(False)
        self.lbl_video_pose_color.setVisible(False)
        self.lbl_video_pose_point_radius.setVisible(False)
        self.spin_video_pose_point_radius.setVisible(False)
        self.lbl_video_pose_point_thickness.setVisible(False)
        self.spin_video_pose_point_thickness.setVisible(False)
        self.lbl_video_pose_line_thickness.setVisible(False)
        self.spin_video_pose_line_thickness.setVisible(False)
        self.lbl_video_pose_disabled_hint.setVisible(False)

        # Histograms
        g_hist = QGroupBox("Which live metrics should be displayed?")
        vl_hist = QVBoxLayout(g_hist)
        vl_hist.addWidget(
            self._create_help_label(
                "Collect and visualize statistics during tracking. Useful for monitoring behavior patterns in real-time. "
                "History window controls how many recent frames to include in the analysis."
            )
        )
        f_hist = QFormLayout(None)
        self.enable_histograms = QCheckBox("Collect live analytics")
        self.enable_histograms.setToolTip(
            "Collect real-time statistics during tracking.\n"
            "Tracks speed, direction, and spatial distributions.\n"
            "Slight performance overhead but useful for monitoring."
        )
        f_hist.addRow(self.enable_histograms)

        self.spin_histogram_history = QSpinBox()
        self.spin_histogram_history.setRange(50, 5000)
        self.spin_histogram_history.setValue(300)
        self.spin_histogram_history.setToolTip(
            "Number of frames to include in rolling statistics (50-5000).\n"
            "Larger window = smoother trends but slower response.\n"
            "Recommended: 100-500 frames for most videos."
        )
        f_hist.addRow(
            "How many frames in live analytics history?", self.spin_histogram_history
        )

        self.btn_show_histograms = QPushButton("Open Plot Window")
        self.btn_show_histograms.setCheckable(True)
        self.btn_show_histograms.clicked.connect(self.toggle_histogram_window)
        f_hist.addRow(self.btn_show_histograms)
        vl_hist.addLayout(f_hist)
        vbox.addWidget(g_hist)

        vbox.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_dataset_ui(self: object) -> object:
        """Tab 6: Dataset Generation for Active Learning."""
        layout = QVBoxLayout(self.tab_dataset)
        layout.setContentsMargins(0, 0, 0, 0)

        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(10, 10, 10, 10)

        # ============================================================
        # Active Learning Dataset Section
        # ============================================================
        self.g_active_learning = QGroupBox(
            "Do you want to generate a detection dataset?"
        )
        vl_active = QVBoxLayout(self.g_active_learning)
        vl_active.addWidget(
            self._create_help_label(
                "Automatically identify challenging frames during tracking and export them for annotation.\n\n"
                "Workflow: Run tracking → Review/correct in X-AnyLabeling → Train improved YOLO model"
            )
        )

        # Enable checkbox
        self.chk_enable_dataset_gen = QCheckBox(
            "Enable Dataset Generation for Active Learning"
        )
        self.chk_enable_dataset_gen.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4a9eff;"
        )
        self.chk_enable_dataset_gen.setChecked(False)
        self.chk_enable_dataset_gen.toggled.connect(self._on_dataset_generation_toggled)
        vl_active.addWidget(self.chk_enable_dataset_gen)

        # Content container for all configuration options
        self.active_learning_content = QWidget()
        vl_content = QVBoxLayout(self.active_learning_content)

        # Dataset configuration
        self.g_dataset_config = QGroupBox("How should the dataset be configured?")
        f_config = QFormLayout(self.g_dataset_config)
        f_config.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Dataset name
        self.line_dataset_name = QLineEdit()
        self.line_dataset_name.setPlaceholderText("e.g., my_dataset_v1")
        self.line_dataset_name.setToolTip(
            "Name for the dataset (used for folder and zip file naming)."
        )
        f_config.addRow("Dataset name", self.line_dataset_name)

        # Class name
        self.line_dataset_class_name = QLineEdit()
        self.line_dataset_class_name.setPlaceholderText("e.g., ant")
        self.line_dataset_class_name.setText("object")
        self.line_dataset_class_name.setToolTip(
            "Name of the object class being tracked.\n"
            "This will be used in the classes.txt file for YOLO training.\n"
            "Examples: ant, bee, mouse, fish, etc."
        )
        f_config.addRow(
            "What class label should be used?", self.line_dataset_class_name
        )

        # Output directory
        h_output = QHBoxLayout()
        self.line_dataset_output = QLineEdit()
        self.line_dataset_output.setPlaceholderText("Select output directory...")
        self.line_dataset_output.setToolTip(
            "Directory where the dataset will be saved."
        )
        self.btn_browse_output = QPushButton("Browse...")
        self.btn_browse_output.clicked.connect(self._select_dataset_output_dir)
        h_output.addWidget(self.line_dataset_output)
        h_output.addWidget(self.btn_browse_output)
        f_config.addRow("Output directory", h_output)

        vl_content.addWidget(self.g_dataset_config)

        # Frame selection parameters
        self.g_frame_selection = QGroupBox("How should frames be selected?")
        f_selection = QFormLayout(self.g_frame_selection)
        f_selection.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Number of frames to export
        self.spin_dataset_max_frames = QSpinBox()
        self.spin_dataset_max_frames.setRange(10, 1000)
        self.spin_dataset_max_frames.setValue(100)
        self.spin_dataset_max_frames.setToolTip(
            "Maximum number of frames to export (10-1000).\n"
            "Higher values provide more training data but increase annotation time.\n"
            "Recommended: 50-200 frames for initial improvement."
        )
        f_selection.addRow("Maximum frames to export", self.spin_dataset_max_frames)

        # Frame quality scoring threshold (used DURING TRACKING to identify problematic frames)
        self.spin_dataset_conf_threshold = QDoubleSpinBox()
        self.spin_dataset_conf_threshold.setRange(0.0, 1.0)
        self.spin_dataset_conf_threshold.setSingleStep(0.05)
        self.spin_dataset_conf_threshold.setDecimals(2)
        self.spin_dataset_conf_threshold.setValue(0.5)
        self.spin_dataset_conf_threshold.setToolTip(
            "[FRAME SELECTION] Confidence threshold for identifying problematic frames DURING TRACKING.\n\n"
            "Frames with detections below this confidence are scored as 'challenging'\n"
            "and prioritized for export to improve training data.\n\n"
            "• Lower (0.3-0.4): Only flag very uncertain detections\n"
            "• Higher (0.5-0.7): Flag moderately uncertain detections too\n\n"
            "Recommended: 0.5 (default) - captures frames that need model improvement"
        )
        f_selection.addRow(
            "What frame quality threshold should be used?",
            self.spin_dataset_conf_threshold,
        )

        # Add help label explaining advanced options
        advanced_help = self._create_help_label(
            "Note: YOLO detection sensitivity for export (confidence=0.05, IOU=0.5) can be "
            "customized in advanced_config.json. These are separate from tracking parameters and "
            "optimized for annotation (detect everything, manual review corrects errors)."
        )
        f_selection.addRow(advanced_help)

        # Visual diversity window
        self.spin_dataset_diversity_window = QSpinBox()
        self.spin_dataset_diversity_window.setRange(10, 500)
        self.spin_dataset_diversity_window.setValue(30)
        self.spin_dataset_diversity_window.setToolTip(
            "Minimum frame separation for visual diversity (10-500 frames).\n"
            "Prevents selecting too many consecutive similar frames.\n"
            "Higher = more spread out frames, more visual variety.\n"
            "Recommended: 20-50 frames (depends on video frame rate)."
        )
        f_selection.addRow(
            "Diversity window (frames)",
            self.spin_dataset_diversity_window,
        )

        # Include context frames
        self.chk_dataset_include_context = QCheckBox(
            "Include neighboring frames (+/-1)"
        )
        self.chk_dataset_include_context.setChecked(True)
        self.chk_dataset_include_context.setToolTip(
            "Export the frame before and after each selected frame.\n"
            "Provides temporal context which can improve annotation quality.\n"
            "Increases dataset size by 3x."
        )
        f_selection.addRow(
            "Include neighboring frames", self.chk_dataset_include_context
        )

        self.chk_dataset_probabilistic = QCheckBox("Probabilistic Sampling")
        self.chk_dataset_probabilistic.setChecked(True)
        self.chk_dataset_probabilistic.setToolTip(
            "Use rank-based probabilistic sampling instead of greedy selection.\n"
            "Probabilistic: Higher quality scores = higher probability (more variety).\n"
            "Greedy: Always select absolute worst frames first (may be too extreme).\n"
            "Recommended: Enabled for better training data diversity."
        )
        f_selection.addRow("Use probabilistic sampling", self.chk_dataset_probabilistic)

        vl_content.addWidget(self.g_frame_selection)

        # Quality metrics
        self.g_quality_metrics = QGroupBox("Which quality checks should be applied?")
        v_metrics = QVBoxLayout(self.g_quality_metrics)

        self.chk_metric_low_confidence = QCheckBox("Flag low detection confidence")
        self.chk_metric_low_confidence.setChecked(True)
        self.chk_metric_low_confidence.setToolTip(
            "Flag frames where YOLO confidence is below threshold."
        )
        v_metrics.addWidget(self.chk_metric_low_confidence)

        self.chk_metric_count_mismatch = QCheckBox("Flag detection count mismatch")
        self.chk_metric_count_mismatch.setChecked(True)
        self.chk_metric_count_mismatch.setToolTip(
            "Flag frames where detected count doesn't match expected number of animals."
        )
        v_metrics.addWidget(self.chk_metric_count_mismatch)

        self.chk_metric_high_assignment_cost = QCheckBox(
            "Flag uncertain track assignment"
        )
        self.chk_metric_high_assignment_cost.setChecked(True)
        self.chk_metric_high_assignment_cost.setToolTip(
            "Flag frames where tracker struggles to match detections to tracks."
        )
        v_metrics.addWidget(self.chk_metric_high_assignment_cost)

        self.chk_metric_track_loss = QCheckBox("Flag frequent track loss")
        self.chk_metric_track_loss.setChecked(True)
        self.chk_metric_track_loss.setToolTip(
            "Flag frames where tracks are frequently lost."
        )
        v_metrics.addWidget(self.chk_metric_track_loss)

        self.chk_metric_high_uncertainty = QCheckBox("Flag high position uncertainty")
        self.chk_metric_high_uncertainty.setChecked(False)
        self.chk_metric_high_uncertainty.setToolTip(
            "Flag frames where Kalman filter is very uncertain about positions."
        )
        v_metrics.addWidget(self.chk_metric_high_uncertainty)

        vl_content.addWidget(self.g_quality_metrics)

        # X-AnyLabeling Integration
        self.g_xanylabeling = QGroupBox("How should X-AnyLabeling be integrated?")
        vl_xany = QVBoxLayout(self.g_xanylabeling)

        # Conda environment selection
        h_env = QHBoxLayout()
        h_env.addWidget(QLabel("Conda environment"))
        self.combo_xanylabeling_env = QComboBox()
        self.combo_xanylabeling_env.setToolTip(
            "Select a conda environment with X-AnyLabeling installed.\n"
            "Environment names should start with 'x-anylabeling-' to be detected."
        )
        h_env.addWidget(self.combo_xanylabeling_env, 1)
        self.btn_refresh_envs = QPushButton("🔄")
        self.btn_refresh_envs.setMaximumWidth(40)
        self.btn_refresh_envs.setToolTip("Refresh conda environments list")
        self.btn_refresh_envs.clicked.connect(self._refresh_xanylabeling_envs)
        h_env.addWidget(self.btn_refresh_envs)
        vl_xany.addLayout(h_env)

        # Open in X-AnyLabeling button
        self.btn_open_xanylabeling = QPushButton(
            "Open Active Learning Dataset in X-AnyLabeling"
        )
        self.btn_open_xanylabeling.setToolTip(
            "Browse for a dataset directory and open it in X-AnyLabeling.\n"
            "Directory must contain: classes.txt, images/, and labels/"
        )
        self.btn_open_xanylabeling.clicked.connect(self._open_in_xanylabeling)
        self.btn_open_xanylabeling.setEnabled(False)
        vl_xany.addWidget(self.btn_open_xanylabeling)

        # X-AnyLabeling integration is now separate from Active Learning content

        # Add content to main group box
        vl_active.addWidget(self.active_learning_content)

        # Add main group box to form
        form.addWidget(self.g_active_learning)

        # ============================================================
        # X-AnyLabeling Integration (separate section)
        # ============================================================
        form.addWidget(self.g_xanylabeling)

        # ============================================================
        # YOLO-OBB Training (separate section)
        # ============================================================
        self.g_yolo_training = QGroupBox("Do you want to train a YOLO-OBB model?")
        vl_yolo_train = QVBoxLayout(self.g_yolo_training)
        self.btn_open_training_dialog = QPushButton("Train YOLO-OBB Model...")
        self.btn_open_training_dialog.setToolTip(
            "Open training dialog to merge datasets and train a YOLO-OBB model."
        )
        self.btn_open_training_dialog.clicked.connect(self._open_training_dialog)
        vl_yolo_train.addWidget(self.btn_open_training_dialog)
        form.addWidget(self.g_yolo_training)

        # Populate conda environments on startup
        self._refresh_xanylabeling_envs()

        # Initially hide content (checkbox starts unchecked)
        self.active_learning_content.setVisible(False)

        # ============================================================
        # Individual Dataset Generator Section (Real-time OBB crops)
        # ============================================================
        self.g_individual_dataset = QGroupBox(
            "Should individual crops be collected in real time?"
        )
        vl_ind_dataset = QVBoxLayout(self.g_individual_dataset)
        vl_ind_dataset.addWidget(
            self._create_help_label(
                "Persist individual-analysis crop images to disk.\n\n"
                "• This save option depends on Analyze Individuals pipeline being enabled\n"
                "• Crops contain only the detected animal (OBB-masked)\n"
                "• Intended for downstream labeling/training workflows\n\n"
                "Note: Available only in YOLO OBB mode."
            )
        )

        self.chk_enable_individual_dataset = QCheckBox(
            "Save Individual Analysis Images to Disk"
        )
        self.chk_enable_individual_dataset.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4a9eff;"
        )
        self.chk_enable_individual_dataset.toggled.connect(
            self._on_individual_dataset_toggled
        )
        vl_ind_dataset.addWidget(self.chk_enable_individual_dataset)

        # Output Configuration
        self.ind_output_group = QGroupBox(
            "Where should individual-analysis outputs go?"
        )
        ind_output_layout = QFormLayout(self.ind_output_group)
        ind_output_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Dataset name
        self.line_individual_dataset_name = QLineEdit()
        self.line_individual_dataset_name.setPlaceholderText("individual_dataset")
        self.line_individual_dataset_name.setToolTip(
            "Name for this dataset. Timestamp will be appended automatically."
        )
        ind_output_layout.addRow("Dataset name", self.line_individual_dataset_name)

        # Output directory
        h_ind_output = QHBoxLayout()
        self.line_individual_output = QLineEdit()
        self.line_individual_output.setPlaceholderText("Select output directory...")
        self.line_individual_output.setToolTip(
            "Directory where individual crops will be saved."
        )
        self.btn_browse_ind_output = QPushButton("Browse...")
        self.btn_browse_ind_output.clicked.connect(self._select_individual_output_dir)
        h_ind_output.addWidget(self.line_individual_output)
        h_ind_output.addWidget(self.btn_browse_ind_output)
        ind_output_layout.addRow("Output directory", h_ind_output)

        # Output format
        self.combo_individual_format = QComboBox()
        self.combo_individual_format.addItems(["PNG", "JPEG"])
        self.combo_individual_format.setCurrentText("PNG")
        self.combo_individual_format.setToolTip(
            "PNG: Lossless, larger files\nJPEG: Smaller files, slight quality loss"
        )
        ind_output_layout.addRow("Image format", self.combo_individual_format)

        # Save interval
        self.spin_individual_interval = QSpinBox()
        self.spin_individual_interval.setRange(1, 100)
        self.spin_individual_interval.setValue(1)
        self.spin_individual_interval.setSingleStep(1)
        self.spin_individual_interval.setToolTip(
            "Save crops every N frames.\n"
            "1 = every frame, 10 = every 10th frame, etc."
        )
        ind_output_layout.addRow("Save every N frames", self.spin_individual_interval)

        vl_ind_dataset.addWidget(
            self._create_help_label(
                "Interpolation, padding, and crop background settings are configured in:\n"
                "Analyze Individuals -> Individual Analysis Pipeline Settings"
            )
        )

        vl_ind_dataset.addWidget(self.ind_output_group)

        # Info label about filtering
        self.lbl_individual_info = self._create_help_label(
            "Note: Crops use detections already filtered by ROI and size settings.\n"
            "No additional filtering parameters needed."
        )
        vl_ind_dataset.addWidget(self.lbl_individual_info)

        form.addWidget(self.g_individual_dataset)

        # ============================================================
        # Pose Label UI Integration (Top-level section)
        # ============================================================
        self.g_pose_label = QGroupBox("Do you want to launch PoseKit labeler?")
        vl_pose = QVBoxLayout(self.g_pose_label)
        vl_pose.addWidget(
            self._create_help_label(
                "Open an individual dataset folder in PoseKit Labeler for keypoint annotation.\n"
                "Select a dataset root that contains an `images/` directory."
            )
        )

        # Open in Pose Label button
        self.btn_open_pose_label = QPushButton(
            "Open Individual Dataset in PoseKit Labeler"
        )
        self.btn_open_pose_label.setToolTip(
            "Browse for a dataset directory and open it in PoseKit Labeler.\n"
            "Directory must contain: images/\n"
            "PoseKit project data will be stored in: posekit_project/"
        )
        self.btn_open_pose_label.clicked.connect(self._open_pose_label_ui)
        self.btn_open_pose_label.setEnabled(True)
        vl_pose.addWidget(self.btn_open_pose_label)

        form.addWidget(self.g_pose_label)

        # Initially hide individual dataset widgets (checkbox starts unchecked)
        self.ind_output_group.setVisible(False)
        self.lbl_individual_info.setVisible(False)

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def setup_individual_analysis_ui(self: object) -> object:
        """Tab 7: Individual Analysis - Real-time Identity & Post-hoc Pose Analysis."""
        layout = QVBoxLayout(self.tab_individual)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)

        # Info box
        info_box = QGroupBox("How should individuals be processed?")
        info_layout = QVBoxLayout(info_box)
        info_layout.addWidget(
            self._create_help_label(
                "Run the individual-analysis pipeline during tracking.\n\n"
                "• Computes per-detection individual properties for filtered detections\n"
                "• Supports pose extraction (YOLO/SLEAP) with reusable caches\n"
                "• Individual analysis is available only in YOLO OBB mode\n"
                "• Background-subtraction mode is intended for YOLO bootstrap dataset creation"
            )
        )
        form.addWidget(info_box)

        # Main Enable Checkbox
        self.chk_enable_individual_analysis = QCheckBox(
            "Enable Individual Analysis Pipeline (YOLO OBB only)"
        )
        self.chk_enable_individual_analysis.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #4a9eff;"
        )
        self.chk_enable_individual_analysis.toggled.connect(
            self._on_individual_analysis_toggled
        )
        form.addWidget(self.chk_enable_individual_analysis)

        self.lbl_individual_yolo_only_notice = self._create_help_label(
            "Individual analysis requires YOLO OBB mode.\n"
            "Switch detection method to YOLO OBB to enable this pipeline."
        )
        self.lbl_individual_yolo_only_notice.setVisible(False)
        form.addWidget(self.lbl_individual_yolo_only_notice)

        # Identity Classification Section
        self.g_identity = QGroupBox("Should identity classification run in real time?")
        vl_identity = QVBoxLayout(self.g_identity)
        vl_identity.addWidget(
            self._create_help_label(
                "Classify individual identity during tracking. Extracts crops around each detection "
                "and processes them with the selected method."
            )
        )

        fl_identity = QFormLayout(None)
        fl_identity.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Identity Method
        self.combo_identity_method = QComboBox()
        self.combo_identity_method.addItems(
            ["None (Disabled)", "Color Tags (YOLO)", "AprilTags", "Custom"]
        )
        self.combo_identity_method.setToolTip(
            "Select method for identifying individual animals:\n"
            "• Color Tags: Detect color markers using YOLO model\n"
            "• AprilTags: Detect fiducial markers\n"
            "• Custom: Implement your own classifier"
        )
        self.combo_identity_method.currentIndexChanged.connect(
            self._on_identity_method_changed
        )
        fl_identity.addRow(
            "Which identity method should be used?", self.combo_identity_method
        )

        # Model/Config for identity (stacked widget for different methods)
        self.identity_config_stack = QStackedWidget()

        # Page 0: None
        none_widget = QWidget()
        none_layout = QVBoxLayout(none_widget)
        none_layout.addWidget(
            self._create_help_label("Identity classification is disabled.")
        )
        self.identity_config_stack.addWidget(none_widget)

        # Page 1: Color Tags (YOLO)
        color_widget = QWidget()
        color_layout = QFormLayout(color_widget)
        self.line_color_tag_model = QLineEdit()
        self.line_color_tag_model.setPlaceholderText("path/to/color_tag_model.pt")
        btn_select_color_model = QPushButton("Browse...")
        btn_select_color_model.clicked.connect(self._select_color_tag_model)
        color_model_layout = QHBoxLayout()
        color_model_layout.addWidget(self.line_color_tag_model)
        color_model_layout.addWidget(btn_select_color_model)
        color_layout.addRow(
            "Which identity model file should be used?", color_model_layout
        )
        self.spin_color_tag_conf = QDoubleSpinBox()
        self.spin_color_tag_conf.setRange(0.01, 1.0)
        self.spin_color_tag_conf.setValue(0.5)
        self.spin_color_tag_conf.setSingleStep(0.05)
        self.spin_color_tag_conf.setToolTip(
            "Minimum confidence for color tag detection"
        )
        color_layout.addRow(
            "What confidence threshold should identity use?", self.spin_color_tag_conf
        )
        self.identity_config_stack.addWidget(color_widget)

        # Page 2: AprilTags
        apriltag_widget = QWidget()
        apriltag_layout = QFormLayout(apriltag_widget)
        self.combo_apriltag_family = QComboBox()
        self.combo_apriltag_family.addItems(
            ["tag36h11", "tag25h9", "tag16h5", "tagCircle21h7", "tagStandard41h12"]
        )
        self.combo_apriltag_family.setToolTip("AprilTag family to detect")
        apriltag_layout.addRow(
            "Which AprilTag family should be used?", self.combo_apriltag_family
        )
        self.spin_apriltag_decimate = QDoubleSpinBox()
        self.spin_apriltag_decimate.setRange(1.0, 4.0)
        self.spin_apriltag_decimate.setValue(1.0)
        self.spin_apriltag_decimate.setSingleStep(0.5)
        self.spin_apriltag_decimate.setToolTip(
            "Decimation factor for faster detection (higher = faster but less accurate)"
        )
        apriltag_layout.addRow(
            "How much AprilTag downsampling should be used?",
            self.spin_apriltag_decimate,
        )
        self.identity_config_stack.addWidget(apriltag_widget)

        # Page 3: Custom
        custom_widget = QWidget()
        custom_layout = QVBoxLayout(custom_widget)
        custom_layout.addWidget(
            self._create_help_label(
                "Implement custom identity classifier in:\n"
                "src/multi_tracker/core/identity/analysis.py"
            )
        )
        self.identity_config_stack.addWidget(custom_widget)

        vl_identity.addLayout(fl_identity)
        vl_identity.addWidget(self.identity_config_stack)

        # Crop Parameters
        crop_group = QGroupBox("How should identity crops be extracted?")
        crop_layout = QFormLayout(crop_group)

        self.spin_identity_crop_multiplier = QDoubleSpinBox()
        self.spin_identity_crop_multiplier.setRange(1.0, 10.0)
        self.spin_identity_crop_multiplier.setValue(3.0)
        self.spin_identity_crop_multiplier.setSingleStep(0.5)
        self.spin_identity_crop_multiplier.setDecimals(1)
        self.spin_identity_crop_multiplier.setToolTip(
            "Crop size = body_size × multiplier\n"
            "Larger values include more context, smaller values focus on the animal"
        )
        crop_layout.addRow("Crop size multiplier", self.spin_identity_crop_multiplier)

        self.spin_identity_crop_min = QSpinBox()
        self.spin_identity_crop_min.setRange(32, 512)
        self.spin_identity_crop_min.setValue(64)
        self.spin_identity_crop_min.setSingleStep(16)
        self.spin_identity_crop_min.setToolTip("Minimum crop size in pixels")
        crop_layout.addRow("Minimum crop size (px)", self.spin_identity_crop_min)

        self.spin_identity_crop_max = QSpinBox()
        self.spin_identity_crop_max.setRange(64, 1024)
        self.spin_identity_crop_max.setValue(256)
        self.spin_identity_crop_max.setSingleStep(16)
        self.spin_identity_crop_max.setToolTip("Maximum crop size in pixels")
        crop_layout.addRow("Maximum crop size (px)", self.spin_identity_crop_max)

        vl_identity.addWidget(crop_group)

        form.addWidget(self.g_identity)

        # Hide legacy identity skeleton UI while keeping controls available for
        # backward-compatible config load/save paths.
        self.g_identity.setVisible(False)

        self.g_individual_pipeline_common = QGroupBox(
            "Individual Analysis Pipeline Settings"
        )
        fl_common = QFormLayout(self.g_individual_pipeline_common)

        self.chk_individual_interpolate = QCheckBox(
            "Interpolate Occluded Frames After Tracking"
        )
        self.chk_individual_interpolate.setChecked(True)
        self.chk_individual_interpolate.setToolTip(
            "After tracking completes, fill occluded gaps by interpolating center/size/angle\n"
            "and generate additional masked crops. Interpolated crops are prefixed with 'interp_'."
        )
        fl_common.addRow("Interpolate occluded frames", self.chk_individual_interpolate)

        self.spin_individual_padding = QDoubleSpinBox()
        self.spin_individual_padding.setRange(0.0, 0.5)
        self.spin_individual_padding.setValue(0.1)
        self.spin_individual_padding.setSingleStep(0.05)
        self.spin_individual_padding.setDecimals(2)
        self.spin_individual_padding.setToolTip(
            "Padding around OBB bounding box as fraction of size.\n"
            "0.1 = 10% padding on each side."
        )
        fl_common.addRow("Crop padding fraction", self.spin_individual_padding)

        bg_color_layout = QHBoxLayout()
        self.btn_background_color = QPushButton()
        self.btn_background_color.setMaximumWidth(60)
        self.btn_background_color.setMinimumHeight(30)
        self.btn_background_color.setToolTip("Click to choose background color")
        self.btn_background_color.clicked.connect(
            self._select_individual_background_color
        )
        self._background_color = (0, 0, 0)  # BGR
        bg_color_layout.addWidget(self.btn_background_color)

        self.btn_median_color = QPushButton("Use Median from Frame")
        self.btn_median_color.setToolTip(
            "Compute median color from the preview frame and use as background"
        )
        self.btn_median_color.clicked.connect(self._compute_median_background_color)
        bg_color_layout.addWidget(self.btn_median_color)
        bg_color_layout.addStretch()

        self.lbl_background_color = QLabel("(0, 0, 0)")
        self.lbl_background_color.setToolTip("Current background color in BGR format")
        bg_color_layout.addWidget(self.lbl_background_color)
        self._update_background_color_button()
        fl_common.addRow("Crop background color", bg_color_layout)

        form.addWidget(self.g_individual_pipeline_common)

        self.g_pose_runtime = QGroupBox("Pose Extraction Settings")
        vl_pose = QVBoxLayout(self.g_pose_runtime)
        vl_pose.addWidget(
            self._create_help_label(
                "Minimum runtime pose settings used by the individual-analysis pipeline."
            )
        )
        fl_pose = QFormLayout(None)
        fl_pose.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form_pose_runtime = fl_pose

        self.chk_enable_pose_extractor = QCheckBox("Enable Pose Extraction")
        self.chk_enable_pose_extractor.setChecked(False)
        self.chk_enable_pose_extractor.toggled.connect(
            self._sync_video_pose_overlay_controls
        )
        self.chk_enable_pose_extractor.toggled.connect(self._on_runtime_context_changed)
        fl_pose.addRow(self.chk_enable_pose_extractor)

        self.combo_pose_model_type = QComboBox()
        self.combo_pose_model_type.addItems(["YOLO", "SLEAP"])
        self.combo_pose_model_type.setToolTip("Pose backend for individual analysis.")
        self.combo_pose_model_type.currentIndexChanged.connect(
            self._sync_pose_backend_ui
        )
        fl_pose.addRow("Pose model type", self.combo_pose_model_type)

        self.combo_pose_runtime_flavor = QComboBox()
        self.combo_pose_runtime_flavor.setToolTip(
            "Pose runtime implementation.\n"
            "Auto/Native uses default backend runtime.\n"
            "ONNX/TensorRT artifacts are exported and reused automatically."
        )
        self._populate_pose_runtime_flavor_options(backend="yolo")
        self.combo_pose_runtime_flavor.currentIndexChanged.connect(
            self._sync_pose_backend_ui
        )
        fl_pose.addRow("Pose runtime", self.combo_pose_runtime_flavor)
        self._set_form_row_visible(fl_pose, self.combo_pose_runtime_flavor, False)

        h_pose_model = QHBoxLayout()
        self.line_pose_model_dir = QLineEdit()
        self.line_pose_model_dir.setPlaceholderText("Select pose model path...")
        self.line_pose_model_dir.textChanged.connect(
            self._on_pose_model_dir_text_changed
        )
        self.btn_browse_pose_model_dir = QPushButton("Browse...")
        self.btn_browse_pose_model_dir.clicked.connect(self._select_pose_model_dir)
        h_pose_model.addWidget(self.line_pose_model_dir)
        h_pose_model.addWidget(self.btn_browse_pose_model_dir)
        fl_pose.addRow("Pose model path", h_pose_model)

        self.spin_pose_min_kpt_conf_valid = QDoubleSpinBox()
        self.spin_pose_min_kpt_conf_valid.setRange(0.0, 1.0)
        self.spin_pose_min_kpt_conf_valid.setSingleStep(0.05)
        self.spin_pose_min_kpt_conf_valid.setDecimals(2)
        self.spin_pose_min_kpt_conf_valid.setValue(0.2)
        self.spin_pose_min_kpt_conf_valid.setToolTip(
            "Minimum per-keypoint confidence to consider a keypoint valid."
        )
        fl_pose.addRow("Min keypoint confidence", self.spin_pose_min_kpt_conf_valid)

        self.spin_pose_batch = QSpinBox()
        self.spin_pose_batch.setRange(1, 256)
        self.spin_pose_batch.setValue(
            int(self.advanced_config.get("pose_batch_size", 4))
        )
        self.spin_pose_batch.setToolTip(
            "Shared batch size for pose inference across YOLO and SLEAP backends."
        )
        fl_pose.addRow("Pose batch size", self.spin_pose_batch)

        h_pose_skeleton = QHBoxLayout()
        self.line_pose_skeleton_file = QLineEdit()
        self.line_pose_skeleton_file.setPlaceholderText(
            "Select skeleton JSON (keypoint_names + skeleton_edges)..."
        )
        default_skeleton = str(
            self.advanced_config.get("pose_skeleton_file", "")
        ).strip()
        if not default_skeleton:
            candidate = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "skeletons"
                / "ooceraea_biroi.json"
            )
            if candidate.exists():
                default_skeleton = str(candidate)
        if default_skeleton:
            self.line_pose_skeleton_file.setText(default_skeleton)
        self.btn_browse_pose_skeleton_file = QPushButton("Browse...")
        self.btn_browse_pose_skeleton_file.clicked.connect(
            self._select_pose_skeleton_file
        )
        self.line_pose_skeleton_file.textChanged.connect(
            self._refresh_pose_direction_keypoint_lists
        )
        h_pose_skeleton.addWidget(self.line_pose_skeleton_file)
        h_pose_skeleton.addWidget(self.btn_browse_pose_skeleton_file)
        fl_pose.addRow("Skeleton file", h_pose_skeleton)

        self.list_pose_ignore_keypoints = QListWidget()
        self.list_pose_ignore_keypoints.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_ignore_keypoints.setMinimumHeight(110)
        self.list_pose_ignore_keypoints.setMaximumHeight(140)
        self.list_pose_ignore_keypoints.setToolTip(
            "Select keypoints to ignore in pose export and orientation logic."
        )

        self.list_pose_direction_anterior = QListWidget()
        self.list_pose_direction_anterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_anterior.setMinimumHeight(110)
        self.list_pose_direction_anterior.setMaximumHeight(140)
        self.list_pose_direction_anterior.setToolTip(
            "Select anterior keypoints from skeleton keypoint list."
        )

        self.list_pose_direction_posterior = QListWidget()
        self.list_pose_direction_posterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_posterior.setMinimumHeight(110)
        self.list_pose_direction_posterior.setMaximumHeight(140)
        self.list_pose_direction_posterior.setToolTip(
            "Select posterior keypoints from skeleton keypoint list."
        )

        pose_kpt_groups_widget = QWidget()
        pose_kpt_groups_layout = QHBoxLayout(pose_kpt_groups_widget)
        pose_kpt_groups_layout.setContentsMargins(0, 0, 0, 0)
        pose_kpt_groups_layout.setSpacing(8)

        ignore_col = QVBoxLayout()
        ignore_label = QLabel("Ignore")
        ignore_label.setStyleSheet("font-weight: bold;")
        ignore_col.addWidget(ignore_label)
        ignore_col.addWidget(self.list_pose_ignore_keypoints)
        pose_kpt_groups_layout.addLayout(ignore_col, 1)

        anterior_col = QVBoxLayout()
        anterior_label = QLabel("Anterior")
        anterior_label.setStyleSheet("font-weight: bold;")
        anterior_col.addWidget(anterior_label)
        anterior_col.addWidget(self.list_pose_direction_anterior)
        pose_kpt_groups_layout.addLayout(anterior_col, 1)

        posterior_col = QVBoxLayout()
        posterior_label = QLabel("Posterior")
        posterior_label.setStyleSheet("font-weight: bold;")
        posterior_col.addWidget(posterior_label)
        posterior_col.addWidget(self.list_pose_direction_posterior)
        pose_kpt_groups_layout.addLayout(posterior_col, 1)

        fl_pose.addRow("Keypoint groups", pose_kpt_groups_widget)
        self.list_pose_ignore_keypoints.itemSelectionChanged.connect(
            lambda: self._on_pose_keypoint_group_changed("ignore")
        )
        self.list_pose_direction_anterior.itemSelectionChanged.connect(
            lambda: self._on_pose_keypoint_group_changed("anterior")
        )
        self.list_pose_direction_posterior.itemSelectionChanged.connect(
            lambda: self._on_pose_keypoint_group_changed("posterior")
        )

        h_sleap_env = QHBoxLayout()
        self.combo_pose_sleap_env = QComboBox()
        self.combo_pose_sleap_env.setToolTip(
            "Conda environment name must start with 'sleap'."
        )
        h_sleap_env.addWidget(self.combo_pose_sleap_env, 1)
        self.btn_refresh_pose_sleap_envs = QPushButton("Refresh")
        self.btn_refresh_pose_sleap_envs.setToolTip("Refresh SLEAP conda envs list")
        self.btn_refresh_pose_sleap_envs.clicked.connect(self._refresh_pose_sleap_envs)
        h_sleap_env.addWidget(self.btn_refresh_pose_sleap_envs)
        self.pose_sleap_env_row_widget = QWidget()
        self.pose_sleap_env_row_widget.setLayout(h_sleap_env)
        fl_pose.addRow("SLEAP env", self.pose_sleap_env_row_widget)

        self.chk_sleap_experimental_features = QCheckBox("Allow experimental features")
        self.chk_sleap_experimental_features.setChecked(False)
        self.chk_sleap_experimental_features.setToolTip(
            "Enable experimental SLEAP features (ONNX/TensorRT runtimes).\n"
            "When disabled, ONNX/TensorRT runtime selections will revert to native runtime.\n"
            "Experimental features may have stability or accuracy issues."
        )
        self.chk_sleap_experimental_features.stateChanged.connect(
            self._on_sleap_experimental_toggled
        )
        self.pose_sleap_experimental_row_widget = QWidget()
        sleap_exp_layout = QHBoxLayout()
        sleap_exp_layout.setContentsMargins(0, 0, 0, 0)
        sleap_exp_layout.addWidget(self.chk_sleap_experimental_features)
        sleap_exp_layout.addStretch()
        self.pose_sleap_experimental_row_widget.setLayout(sleap_exp_layout)
        fl_pose.addRow("", self.pose_sleap_experimental_row_widget)

        vl_pose.addLayout(fl_pose)
        form.addWidget(self.g_pose_runtime)

        self._refresh_pose_sleap_envs()
        self._set_form_row_visible(
            fl_pose, self.pose_sleap_experimental_row_widget, False
        )

        form.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Initially disable all controls
        self.g_identity.setEnabled(False)
        self.g_pose_runtime.setEnabled(False)
        self._refresh_pose_direction_keypoint_lists()
        self._sync_pose_backend_ui()
        self._sync_individual_analysis_mode_ui()

    def _on_dataset_generation_toggled(self, enabled):
        """Enable/disable dataset generation controls."""
        # Hide/show entire content container
        self.active_learning_content.setVisible(enabled)

    def _select_dataset_output_dir(self):
        """Browse for dataset output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Dataset Output Directory"
        )
        if directory:
            self.line_dataset_output.setText(directory)

    def _open_training_dialog(self):
        class_name = self.line_dataset_class_name.text().strip() or "object"
        envs = []
        for i in range(self.combo_xanylabeling_env.count()):
            name = self.combo_xanylabeling_env.itemText(i)
            if "No X-AnyLabeling" in name or "Conda not available" in name:
                continue
            envs.append(name)
        dialog = TrainYoloDialog(self, class_name=class_name, conda_envs=envs)
        dialog.exec()

    def _refresh_xanylabeling_envs(self):
        """Scan for conda environments starting with 'x-anylabeling-'."""
        self.combo_xanylabeling_env.clear()

        try:
            import subprocess

            # Get list of conda environments
            result = subprocess.run(
                ["conda", "env", "list"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                envs = []
                for line in result.stdout.split("\n"):
                    # Skip comments and empty lines
                    if line.strip() and not line.startswith("#"):
                        # Parse environment name (first column)
                        parts = line.split()
                        if parts:
                            env_name = parts[0]
                            # Check if it starts with 'x-anylabeling-'
                            if env_name.startswith("x-anylabeling-"):
                                envs.append(env_name)

                if envs:
                    self.combo_xanylabeling_env.addItems(envs)
                    self.btn_open_xanylabeling.setEnabled(True)
                    logger.info(f"Found {len(envs)} X-AnyLabeling conda environment(s)")
                else:
                    self.combo_xanylabeling_env.addItem("No X-AnyLabeling envs found")
                    self.btn_open_xanylabeling.setEnabled(False)
                    logger.warning(
                        "No conda environments starting with 'x-anylabeling-' found. "
                        "Create one with: conda create -n x-anylabeling-env python=3.10 && "
                        "conda activate x-anylabeling-env && pip install x-anylabeling"
                    )
            else:
                self.combo_xanylabeling_env.addItem("Conda not available")
                self.btn_open_xanylabeling.setEnabled(False)
                logger.warning("Could not detect conda environments")

        except FileNotFoundError:
            self.combo_xanylabeling_env.addItem("Conda not installed")
            self.btn_open_xanylabeling.setEnabled(False)
            logger.warning("Conda not found in PATH")
        except Exception as e:
            self.combo_xanylabeling_env.addItem("Error detecting envs")
            self.btn_open_xanylabeling.setEnabled(False)
            logger.error(f"Error detecting conda environments: {e}")

    def _open_in_xanylabeling(self):
        """Open a dataset directory in X-AnyLabeling."""
        # Get selected conda environment
        env_name = self.combo_xanylabeling_env.currentText()
        if (
            not env_name
            or env_name.startswith("No ")
            or env_name.startswith("Conda ")
            or env_name.startswith("Error")
        ):
            QMessageBox.warning(
                self,
                "No Environment",
                "Please select a valid conda environment with X-AnyLabeling installed.",
            )
            return

        # Browse for dataset directory
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory",
            self.line_dataset_output.text() if self.line_dataset_output.text() else "",
        )

        if not directory:
            return

        dataset_path = Path(directory)

        # Validate directory structure
        classes_file = dataset_path / "classes.txt"
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        missing = []
        if not classes_file.exists():
            missing.append("classes.txt")
        if not images_dir.exists() or not images_dir.is_dir():
            missing.append("images/")
        if not labels_dir.exists() or not labels_dir.is_dir():
            missing.append("labels/")

        if missing:
            QMessageBox.warning(
                self,
                "Invalid Dataset",
                f"Dataset directory is missing required items:\n{', '.join(missing)}\n\n"
                f"A valid dataset must contain:\n"
                f"- classes.txt\n"
                f"- images/ (directory)\n"
                f"- labels/ (directory)",
            )
            return

        # Determine shell command based on OS
        import platform
        import subprocess

        system = platform.system()

        try:
            if system == "Darwin":  # macOS
                # Create an AppleScript to open Terminal and run commands
                # Source conda.sh directly to initialize conda in the session
                script = f"""
                tell application "Terminal"
                    activate
                    do script "source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd '{dataset_path}' && xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels --output ./images --classes classes.txt && xanylabeling --filename ./images"
                end tell
                """
                subprocess.Popen(["osascript", "-e", script])

            elif system == "Windows":
                # Use cmd.exe with conda activation
                cmd = f'start cmd /k "conda activate {env_name} && cd /d {dataset_path} && xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels --output ./images --classes classes.txt && xanylabeling --filename ./images"'
                subprocess.Popen(cmd, shell=True)

            else:  # Linux
                # Try common terminal emulators
                terminals = ["gnome-terminal", "konsole", "xterm"]
                terminal_found = False

                for terminal in terminals:
                    try:
                        if terminal == "gnome-terminal":
                            subprocess.Popen(
                                [
                                    terminal,
                                    "--",
                                    "bash",
                                    "-c",
                                    f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd '{dataset_path}' && xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels --output ./images --classes classes.txt && xanylabeling --filename ./images; exec bash",
                                ]
                            )
                        elif terminal == "konsole":
                            subprocess.Popen(
                                [
                                    terminal,
                                    "-e",
                                    "bash",
                                    "-c",
                                    f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd '{dataset_path}' && xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels --output ./images --classes classes.txt && xanylabeling --filename ./images; exec bash",
                                ]
                            )
                        else:  # xterm
                            subprocess.Popen(
                                [
                                    terminal,
                                    "-e",
                                    "bash",
                                    "-c",
                                    f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {env_name} && cd '{dataset_path}' && xanylabeling convert --task yolo2xlabel --mode obb --images ./images --labels ./labels --output ./images --classes classes.txt && xanylabeling --filename ./images; exec bash",
                                ]
                            )
                        terminal_found = True
                        break
                    except FileNotFoundError:
                        continue

                if not terminal_found:
                    QMessageBox.warning(
                        self,
                        "No Terminal Found",
                        "Could not find a supported terminal emulator (gnome-terminal, konsole, or xterm).",
                    )
                    return

            logger.info(f"Opened X-AnyLabeling for dataset: {dataset_path}")
            QMessageBox.information(
                self,
                "X-AnyLabeling Launched",
                f"X-AnyLabeling is starting in environment: {env_name}\n\n"
                f"Commands being executed:\n"
                f"1. Convert YOLO to X-Label format\n"
                f"2. Open X-AnyLabeling with images\n\n"
                f"Dataset: {dataset_path}",
            )

        except Exception as e:
            logger.error(f"Failed to open X-AnyLabeling: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Launch Error", f"Failed to open X-AnyLabeling:\n{str(e)}"
            )

    def _open_pose_label_ui(self):
        """Open a dataset folder in PoseKit Labeler."""
        start_dir = (
            self.line_individual_output.text().strip()
            or self.line_dataset_output.text().strip()
            or str(Path.home())
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Pose Dataset Folder",
            start_dir,
        )
        if not directory:
            return

        dataset_root = Path(directory).expanduser().resolve()
        images_dir = dataset_root / "images"
        if not images_dir.exists() or not images_dir.is_dir():
            QMessageBox.warning(
                self,
                "Invalid Dataset",
                "Selected folder does not contain an `images/` directory.\n\n"
                "Please select a dataset root with this structure:\n"
                "dataset_root/\n"
                "  images/\n",
            )
            return

        # Check if images directory has any images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in images_dir.rglob("*")
            if f.is_file()
        )

        if not has_images:
            QMessageBox.warning(
                self,
                "No Images Found",
                f"No image files found in:\n{images_dir}\n\n"
                f"Supported formats: {', '.join(image_extensions)}",
            )
            return

        # Launch pose_label.py using the current Python interpreter
        import subprocess
        import sys

        try:
            # The labeler lives in the top-level posekit package.
            gui_dir = Path(__file__).parent
            package_root = gui_dir.parent
            labeler_dir = package_root / "posekit"
            pose_label_script = labeler_dir / "pose_label.py"

            if not pose_label_script.exists():
                QMessageBox.critical(
                    self,
                    "Script Not Found",
                    f"Could not find pose_label.py at:\n{pose_label_script}",
                )
                return

            # Use the current Python executable (same environment as mat)
            # This avoids conda activation and terminal detection complexity
            subprocess.Popen(
                [sys.executable, str(pose_label_script), str(dataset_root)],
                cwd=str(labeler_dir),
            )

            logger.info(f"Opened PoseKit Labeler for dataset: {dataset_root}")
            QMessageBox.information(
                self,
                "PoseKit Labeler Launched",
                f"PoseKit Labeler is starting...\n\n"
                f"Dataset: {dataset_root}\n"
                f"Images: {images_dir}\n\n"
                f"Project data will be created/loaded at:\n"
                f"{dataset_root / 'posekit_project'}",
            )

        except Exception as e:
            logger.error(f"Failed to open Pose Label UI: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Launch Error", f"Failed to open Pose Label UI:\n{str(e)}"
            )

    def _on_individual_analysis_toggled(self, state):
        """Enable/disable individual analysis controls."""
        self._sync_individual_analysis_mode_ui()

    def _on_identity_method_changed(self, index):
        """Update identity configuration stack when method changes."""
        self.identity_config_stack.setCurrentIndex(index)

    def _select_color_tag_model(self):
        """Browse for color tag YOLO model."""
        # Default to models directory
        start_dir = get_models_directory()
        if self.line_color_tag_model.text():
            current_path = resolve_model_path(self.line_color_tag_model.text())
            if os.path.exists(current_path):
                start_dir = os.path.dirname(current_path)

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Color Tag YOLO Model", start_dir, "YOLO Models (*.pt *.onnx)"
        )
        if filepath:
            # Check if model is outside the archive
            models_dir = get_models_directory()
            try:
                rel_path = os.path.relpath(filepath, models_dir)
                is_in_archive = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_archive = False

            if not is_in_archive:
                # Ask user if they want to copy to archive
                reply = QMessageBox.question(
                    self,
                    "Copy Model to Archive?",
                    f"The selected model is outside the local model archive.\n\n"
                    f"Would you like to copy it to the archive at:\n{models_dir}\n\n"
                    f"This makes presets portable across devices.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    import shutil

                    filename = os.path.basename(filepath)
                    dest_path = os.path.join(models_dir, filename)

                    # Handle duplicate filenames
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(filename)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(
                                models_dir, f"{base}_{counter}{ext}"
                            )
                            counter += 1

                    try:
                        shutil.copy2(filepath, dest_path)
                        filepath = dest_path
                        logger.info(f"Copied color tag model to archive: {dest_path}")
                        QMessageBox.information(
                            self,
                            "Model Copied",
                            f"Model copied to archive as:\n{os.path.basename(dest_path)}",
                        )
                    except Exception as e:
                        logger.error(f"Failed to copy model: {e}")
                        QMessageBox.warning(
                            self,
                            "Copy Failed",
                            f"Could not copy model to archive:\n{e}\n\nUsing original path.",
                        )

            self.line_color_tag_model.setText(filepath)

    def _on_individual_dataset_toggled(self, enabled):
        """Enable/disable individual dataset generation controls."""
        self._sync_individual_analysis_mode_ui()

    def _ensure_pose_model_path_store(self):
        if not hasattr(self, "_pose_model_path_by_backend"):
            self._pose_model_path_by_backend = {"yolo": "", "sleap": ""}

    def _current_pose_backend_key(self):
        if not hasattr(self, "combo_pose_model_type"):
            return "yolo"
        backend = self.combo_pose_model_type.currentText().strip().lower()
        return "sleap" if backend == "sleap" else "yolo"

    def _pose_model_path_for_backend(self, backend=None):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        key = "sleap" if key == "sleap" else "yolo"
        return str(self._pose_model_path_by_backend.get(key, "")).strip()

    def _set_pose_model_path_for_backend(self, path, backend=None, update_line=False):
        self._ensure_pose_model_path_store()
        key = (backend or self._current_pose_backend_key()).strip().lower()
        key = "sleap" if key == "sleap" else "yolo"
        value = str(path or "").strip()
        if value:
            resolved = str(resolve_pose_model_path(value, backend=key)).strip()
            if resolved and os.path.exists(resolved):
                value = str(make_pose_model_path_relative(os.path.abspath(resolved)))
        self._pose_model_path_by_backend[key] = value
        if update_line and hasattr(self, "line_pose_model_dir"):
            self.line_pose_model_dir.blockSignals(True)
            self.line_pose_model_dir.setText(value)
            self.line_pose_model_dir.blockSignals(False)

    def _on_pose_model_dir_text_changed(self, text):
        self._set_pose_model_path_for_backend(
            text, backend=self._current_pose_backend_key()
        )

    def _select_pose_model_dir(self):
        """Select a pose model file/directory depending on backend."""
        backend = self.combo_pose_model_type.currentText().strip().lower()
        backend_key = "sleap" if backend == "sleap" else "yolo"
        current = self._pose_model_path_for_backend(backend)
        if current:
            resolved_current = str(resolve_pose_model_path(current, backend=backend))
            if os.path.isdir(resolved_current):
                start = resolved_current
            else:
                start = os.path.dirname(resolved_current) or str(Path.home())
        else:
            start = get_pose_models_directory(backend_key)

        if backend == "sleap":
            selected = QFileDialog.getExistingDirectory(
                self, "Select SLEAP Model Directory", start
            )
            if selected:
                selected_abs = os.path.abspath(selected)
                pose_root = get_pose_models_directory(backend_key)
                try:
                    rel_path = os.path.relpath(selected_abs, pose_root)
                    is_in_repo = not rel_path.startswith("..")
                except (ValueError, TypeError):
                    is_in_repo = False

                if is_in_repo:
                    final_path = make_pose_model_path_relative(selected_abs)
                else:
                    final_path = self._import_pose_model_to_repository(
                        selected_abs, backend=backend_key
                    )
                    if not final_path:
                        return
                    QMessageBox.information(
                        self,
                        "Model Imported",
                        f"SLEAP model imported to repository as:\n{final_path}",
                    )
                self._set_pose_model_path_for_backend(
                    final_path, backend=backend, update_line=True
                )
            return

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Pose Weights",
            start,
            "PyTorch Weights (*.pt);;All Files (*)",
        )
        if selected:
            selected_abs = os.path.abspath(selected)
            pose_root = get_pose_models_directory(backend_key)
            try:
                rel_path = os.path.relpath(selected_abs, pose_root)
                is_in_repo = not rel_path.startswith("..")
            except (ValueError, TypeError):
                is_in_repo = False

            if is_in_repo:
                final_path = make_pose_model_path_relative(selected_abs)
            else:
                final_path = self._import_pose_model_to_repository(
                    selected_abs, backend=backend_key
                )
                if not final_path:
                    return
                QMessageBox.information(
                    self,
                    "Model Imported",
                    f"Pose model imported to repository as:\n{final_path}",
                )
            self._set_pose_model_path_for_backend(
                final_path, backend=backend, update_line=True
            )

    def _import_pose_model_to_repository(self, source_path, backend="yolo"):
        """Copy a selected pose model into models/{YOLO-pose|SLEAP} and return relative path."""
        src = str(source_path or "").strip()
        if not src or not os.path.exists(src):
            return None

        backend_key = "sleap" if str(backend).strip().lower() == "sleap" else "yolo"
        dest_dir = get_pose_models_directory(backend_key)

        try:
            src_path = Path(src).expanduser().resolve()
        except Exception:
            src_path = Path(src)

        # Metadata collection (same style as YOLO OBB import dialog).
        now_preview = datetime.now()
        dlg = QDialog(self)
        dlg.setWindowTitle("Pose Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        stem_tokens = [t for t in src_path.stem.replace("-", "_").split("_") if t]
        default_species = (
            self._sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            self._sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        size_combo = None
        type_line = None
        if backend_key == "yolo":
            size_combo = QComboBox(dlg)
            size_combo.addItems(
                ["26n", "26s", "26m", "26l", "26x", "custom", "unknown"]
            )
            size_combo.setCurrentText("26s")
            dlg_form.addRow("YOLO model size:", size_combo)
        else:
            default_type = self._sanitize_model_token(src_path.name) or "sleap_model"
            type_line = QLineEdit(default_type, dlg)
            type_line.setPlaceholderText("model-type")
            dlg_form.addRow("Model type:", type_line)

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_species = self._sanitize_model_token(species_line.text())
        model_info = self._sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        if backend_key == "sleap":
            model_type = (
                self._sanitize_model_token(type_line.text()) if type_line else ""
            )
            if not model_type:
                QMessageBox.warning(
                    self,
                    "Invalid Metadata",
                    "SLEAP model type must be provided.",
                )
                return None
            target_name = f"{timestamp}_{model_type}_{model_species}_{model_info}"
            dest_path = Path(dest_dir) / target_name
            counter = 1
            while dest_path.exists():
                dest_path = Path(dest_dir) / f"{target_name}_{counter}"
                counter += 1
            try:
                shutil.copytree(src_path, dest_path)
            except Exception as exc:
                logger.error("Failed to copy SLEAP model directory: %s", exc)
                QMessageBox.warning(
                    self,
                    "Import Failed",
                    f"Could not import SLEAP model directory:\n{exc}",
                )
                return None
            return make_pose_model_path_relative(str(dest_path))

        model_size = size_combo.currentText().strip() if size_combo else "unknown"
        model_size = self._sanitize_model_token(model_size) or "unknown"
        ext = src_path.suffix or ".pt"
        target_name = f"{timestamp}_{model_size}_{model_species}_{model_info}{ext}"
        dest_path = Path(dest_dir) / target_name
        counter = 1
        while dest_path.exists():
            dest_path = (
                Path(dest_dir)
                / f"{timestamp}_{model_size}_{model_species}_{model_info}_{counter}{ext}"
            )
            counter += 1
        try:
            shutil.copy2(src_path, dest_path)
        except Exception as exc:
            logger.error("Failed to copy pose model: %s", exc)
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Could not import pose model:\n{exc}",
            )
            return None
        return make_pose_model_path_relative(str(dest_path))

    def _select_pose_skeleton_file(self):
        """Select pose skeleton JSON file."""
        start = self.line_pose_skeleton_file.text().strip() or str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pose Skeleton JSON",
            start,
            "JSON Files (*.json);;All Files (*)",
        )
        if selected:
            self.line_pose_skeleton_file.setText(selected)

    def _load_pose_skeleton_keypoint_names(self):
        """Load keypoint names from selected skeleton JSON."""
        skeleton_file = self.line_pose_skeleton_file.text().strip()
        if not skeleton_file:
            return []
        try:
            path = Path(skeleton_file).expanduser().resolve()
            if not path.exists():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get("keypoint_names", data.get("keypoints", []))
            names = [str(v).strip() for v in raw if str(v).strip()]
            return names
        except Exception:
            return []

    def _selected_pose_group_keypoints(self, list_widget):
        """Return selected keypoint names for a pose direction list widget."""
        if list_widget is None:
            return []
        return [
            item.text().strip()
            for item in list_widget.selectedItems()
            if item.text().strip()
        ]

    def _set_pose_group_selection(self, list_widget, values):
        """Select keypoints in list widget from config-provided values."""
        if list_widget is None:
            return
        tokens = self._parse_pose_keypoint_tokens(values)
        if not tokens:
            return
        items_by_name = {}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            items_by_name[item.text().strip().lower()] = item
        for tok in tokens:
            if isinstance(tok, int):
                if 0 <= tok < list_widget.count():
                    list_widget.item(tok).setSelected(True)
                continue
            item = items_by_name.get(str(tok).strip().lower())
            if item is not None:
                item.setSelected(True)

    def _apply_pose_keypoint_selection_set(self, list_widget, selected_names):
        """Set list selection to exact keypoint-name set."""
        if list_widget is None:
            return
        target = {str(v).strip().lower() for v in selected_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            item.setSelected(item_name in target)

    def _set_pose_keypoints_enabled(self, list_widget, disabled_names):
        """Disable keypoints in a list by name while preserving others."""
        if list_widget is None:
            return
        disabled = {str(v).strip().lower() for v in disabled_names if str(v).strip()}
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_name = item.text().strip().lower()
            flags = item.flags()
            if item_name in disabled:
                item.setSelected(False)
                item.setFlags(flags & ~Qt.ItemIsEnabled)
            else:
                item.setFlags(flags | Qt.ItemIsEnabled)

    def _apply_pose_keypoint_selection_constraints(self, changed_group="ignore"):
        """Enforce exclusivity across ignore/anterior/posterior keypoint selections."""
        ignore_list = getattr(self, "list_pose_ignore_keypoints", None)
        ant_list = getattr(self, "list_pose_direction_anterior", None)
        post_list = getattr(self, "list_pose_direction_posterior", None)
        if ignore_list is None or ant_list is None or post_list is None:
            return

        ignore = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ignore_list)
            if k.strip()
        }
        anterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(ant_list)
            if k.strip()
        }
        posterior = {
            k.lower()
            for k in self._selected_pose_group_keypoints(post_list)
            if k.strip()
        }

        # Ignore selection always wins over directional groups.
        anterior -= ignore
        posterior -= ignore

        # Anterior/posterior remain mutually exclusive. The changed group wins.
        if changed_group == "posterior":
            anterior -= posterior
        else:
            posterior -= anterior

        ant_list.blockSignals(True)
        post_list.blockSignals(True)
        self._apply_pose_keypoint_selection_set(ant_list, anterior)
        self._apply_pose_keypoint_selection_set(post_list, posterior)
        self._set_pose_keypoints_enabled(ant_list, ignore)
        self._set_pose_keypoints_enabled(post_list, ignore)
        ant_list.blockSignals(False)
        post_list.blockSignals(False)

    def _on_pose_keypoint_group_changed(self, changed_group):
        """Handle keypoint group updates and keep list constraints synchronized."""
        self._apply_pose_keypoint_selection_constraints(changed_group)

    def _refresh_pose_direction_keypoint_lists(self):
        """Populate ignore/anterior/posterior keypoint pickers from skeleton file."""
        if (
            not hasattr(self, "list_pose_ignore_keypoints")
            or not hasattr(self, "list_pose_direction_anterior")
            or not hasattr(self, "list_pose_direction_posterior")
        ):
            return

        prev_ignore = self._selected_pose_group_keypoints(
            self.list_pose_ignore_keypoints
        )
        prev_anterior = self._selected_pose_group_keypoints(
            self.list_pose_direction_anterior
        )
        prev_posterior = self._selected_pose_group_keypoints(
            self.list_pose_direction_posterior
        )
        names = self._load_pose_skeleton_keypoint_names()

        self.list_pose_ignore_keypoints.blockSignals(True)
        self.list_pose_direction_anterior.blockSignals(True)
        self.list_pose_direction_posterior.blockSignals(True)
        self.list_pose_ignore_keypoints.clear()
        self.list_pose_direction_anterior.clear()
        self.list_pose_direction_posterior.clear()
        self.list_pose_ignore_keypoints.addItems(names)
        self.list_pose_direction_anterior.addItems(names)
        self.list_pose_direction_posterior.addItems(names)
        self._set_pose_group_selection(self.list_pose_ignore_keypoints, prev_ignore)
        self._set_pose_group_selection(self.list_pose_direction_anterior, prev_anterior)
        self._set_pose_group_selection(
            self.list_pose_direction_posterior, prev_posterior
        )
        enabled = len(names) > 0
        self.list_pose_ignore_keypoints.setEnabled(enabled)
        self.list_pose_direction_anterior.setEnabled(enabled)
        self.list_pose_direction_posterior.setEnabled(enabled)
        self.list_pose_ignore_keypoints.blockSignals(False)
        self.list_pose_direction_anterior.blockSignals(False)
        self.list_pose_direction_posterior.blockSignals(False)
        self._apply_pose_keypoint_selection_constraints("ignore")

    def _parse_pose_keypoint_tokens(self, raw):
        """Parse comma-separated keypoint names and/or indices."""
        if not raw:
            return []
        if isinstance(raw, (list, tuple, set)):
            tokens = []
            for value in raw:
                if value is None:
                    continue
                text = str(value).strip()
                if not text:
                    continue
                try:
                    tokens.append(int(text))
                except ValueError:
                    tokens.append(text)
            return tokens
        out = []
        for token in str(raw).split(","):
            t = token.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                out.append(t)
        return out

    def _parse_pose_ignore_keypoints(self):
        """Parse keypoints to ignore from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_ignore_keypoints", None)
        )

    def _parse_pose_direction_anterior_keypoints(self):
        """Parse anterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_direction_anterior", None)
        )

    def _parse_pose_direction_posterior_keypoints(self):
        """Parse posterior keypoint group from list selection."""
        return self._selected_pose_group_keypoints(
            getattr(self, "list_pose_direction_posterior", None)
        )

    def _refresh_pose_sleap_envs(self):
        """Refresh conda environments starting with 'sleap'."""
        if not hasattr(self, "combo_pose_sleap_env"):
            return

        self.combo_pose_sleap_env.clear()
        self.combo_pose_sleap_env.setEnabled(True)
        envs = []
        preferred = str(self.advanced_config.get("pose_sleap_env", "sleap")).strip()
        try:
            import subprocess

            res = subprocess.run(
                ["conda", "env", "list"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    name = parts[0].strip()
                    if name.lower().startswith("sleap"):
                        envs.append(name)
        except Exception:
            envs = []

        if not envs:
            self.combo_pose_sleap_env.addItem("No sleap envs found")
            self.combo_pose_sleap_env.setEnabled(False)
            return

        self.combo_pose_sleap_env.addItems(envs)
        if preferred and preferred in envs:
            self.combo_pose_sleap_env.setCurrentText(preferred)

    def _selected_pose_sleap_env(self):
        """Return valid selected SLEAP env name or default."""
        if not hasattr(self, "combo_pose_sleap_env"):
            return "sleap"
        txt = self.combo_pose_sleap_env.currentText().strip()
        if not txt or txt.lower().startswith("no sleap envs"):
            return "sleap"
        return txt

    def _sleap_experimental_features_enabled(self):
        """Return True if SLEAP experimental features (ONNX/TensorRT) are allowed."""
        if not hasattr(self, "chk_sleap_experimental_features"):
            return False
        return self.chk_sleap_experimental_features.isChecked()

    def _on_sleap_experimental_toggled(self):
        """Handle experimental features checkbox toggle."""
        if not hasattr(self, "combo_pose_runtime_flavor"):
            return
        # Refresh runtime options to show warning if needed
        backend = (
            self.combo_pose_model_type.currentText().strip().lower()
            if hasattr(self, "combo_pose_model_type")
            else "yolo"
        )
        if backend == "sleap":
            current_flavor = self._selected_pose_runtime_flavor()
            if (
                current_flavor in ("onnx", "tensorrt")
                and not self._sleap_experimental_features_enabled()
            ):
                # Show warning that runtime will revert to native
                from PyQt5.QtWidgets import QMessageBox

                QMessageBox.warning(
                    self,
                    "Experimental Features Disabled",
                    f"SLEAP {current_flavor.upper()} runtime is experimental.\\n"
                    "With experimental features disabled, the runtime will revert to native.\\n\\n"
                    "To use ONNX/TensorRT for SLEAP, enable experimental features.",
                )

    def _populate_pose_sleap_device_options(self):
        """Populate SLEAP device options using gpu_utils availability flags."""
        if not hasattr(self, "combo_pose_sleap_device"):
            return
        self.combo_pose_sleap_device.clear()
        opts = ["auto", "cpu"]
        if MPS_AVAILABLE:
            opts.append("mps")
        if TORCH_CUDA_AVAILABLE or ROCM_AVAILABLE:
            opts.extend(["cuda", "cuda:0"])
        self.combo_pose_sleap_device.addItems(opts)
        default_sleap_device = str(
            self.advanced_config.get("pose_sleap_device", "auto")
        ).strip()
        idx = self.combo_pose_sleap_device.findText(default_sleap_device)
        if idx >= 0:
            self.combo_pose_sleap_device.setCurrentIndex(idx)

    def _runtime_pipelines_for_current_ui(self):
        pipelines = []
        if self._is_yolo_detection_mode():
            pipelines.append("yolo_obb_detection")
        if self._is_pose_inference_enabled():
            backend = self.combo_pose_model_type.currentText().strip().lower()
            if backend == "sleap":
                pipelines.append("sleap_pose")
            else:
                pipelines.append("yolo_pose")
        return pipelines

    def _compute_runtime_options_for_current_ui(self):
        allowed = allowed_runtimes_for_pipelines(
            self._runtime_pipelines_for_current_ui()
        )
        if not allowed:
            allowed = ["cpu"]
        return [(runtime_label(rt), rt) for rt in allowed if rt in CANONICAL_RUNTIMES]

    def _populate_compute_runtime_options(self, preferred: str | None = None):
        if not hasattr(self, "combo_compute_runtime"):
            return
        combo = self.combo_compute_runtime
        selected = (
            str(preferred or self._selected_compute_runtime() or "cpu").strip().lower()
        )
        options = self._compute_runtime_options_for_current_ui()
        values = [value for _label, value in options]
        if selected not in values:
            selected = values[0] if values else "cpu"
        combo.blockSignals(True)
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _selected_compute_runtime(self) -> str:
        if not hasattr(self, "combo_compute_runtime"):
            return "cpu"
        data = self.combo_compute_runtime.currentData()
        if data:
            return str(data).strip().lower()
        txt = self.combo_compute_runtime.currentText().strip().lower()
        if txt in CANONICAL_RUNTIMES:
            return txt
        return "cpu"

    def _runtime_requires_fixed_yolo_batch(self, runtime: str | None = None) -> bool:
        rt = str(runtime or self._selected_compute_runtime() or "").strip().lower()
        return rt == "tensorrt" or rt.startswith("onnx")

    def _on_runtime_context_changed(self, *_args):
        previous = self._selected_compute_runtime()
        self._populate_compute_runtime_options(preferred=previous)
        selected_runtime = self._selected_compute_runtime()
        # Keep hidden legacy controls synchronized for compatibility paths.
        derived = derive_detection_runtime_settings(selected_runtime)
        if hasattr(self, "combo_device"):
            idx = self.combo_device.findText(
                str(derived.get("yolo_device", "cpu")), Qt.MatchStartsWith
            )
            if idx >= 0:
                self.combo_device.setCurrentIndex(idx)
        if hasattr(self, "chk_enable_tensorrt"):
            self.chk_enable_tensorrt.setChecked(
                bool(derived.get("enable_tensorrt", False))
            )
        if (
            self._runtime_requires_fixed_yolo_batch(selected_runtime)
            and hasattr(self, "combo_yolo_batch_mode")
            and hasattr(self, "spin_yolo_batch_size")
            and hasattr(self, "chk_enable_yolo_batching")
        ):
            self.chk_enable_yolo_batching.setChecked(True)
            self.chk_enable_yolo_batching.setEnabled(False)
            self.combo_yolo_batch_mode.setCurrentIndex(1)  # Manual
            self.combo_yolo_batch_mode.setEnabled(False)
            self.spin_yolo_batch_size.setEnabled(True)
            if hasattr(self, "spin_tensorrt_batch"):
                self.spin_tensorrt_batch.setValue(self.spin_yolo_batch_size.value())
        elif hasattr(self, "combo_yolo_batch_mode") and hasattr(
            self, "chk_enable_yolo_batching"
        ):
            self.chk_enable_yolo_batching.setEnabled(True)
            self.combo_yolo_batch_mode.setEnabled(
                self.chk_enable_yolo_batching.isChecked()
            )
            self._on_yolo_batch_mode_changed(self.combo_yolo_batch_mode.currentIndex())
        if hasattr(self, "combo_pose_model_type"):
            self._populate_pose_runtime_flavor_options(
                backend=self.combo_pose_model_type.currentText().strip().lower(),
                preferred=self._selected_pose_runtime_flavor(),
            )

    def _pose_runtime_options_for_backend(self, backend: str):
        derived = derive_pose_runtime_settings(
            self._selected_compute_runtime(), backend_family=backend
        )
        flavor = str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()
        return [(runtime_label(self._selected_compute_runtime()), flavor)]

    def _populate_pose_runtime_flavor_options(
        self, backend: str, preferred: str | None = None
    ):
        if not hasattr(self, "combo_pose_runtime_flavor"):
            return
        combo = self.combo_pose_runtime_flavor
        selected = (
            str(preferred or self._selected_pose_runtime_flavor() or "auto")
            .strip()
            .lower()
        )
        options = self._pose_runtime_options_for_backend(backend)
        values = [value for _label, value in options]
        if selected not in values:
            selected = values[0] if values else "cpu"
        combo.blockSignals(True)
        combo.clear()
        for label, value in options:
            combo.addItem(label, value)
        idx = combo.findData(selected)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _selected_pose_runtime_flavor(self) -> str:
        backend = (
            self.combo_pose_model_type.currentText().strip().lower()
            if hasattr(self, "combo_pose_model_type")
            else "yolo"
        )
        derived = derive_pose_runtime_settings(
            self._selected_compute_runtime(), backend_family=backend
        )
        return str(derived.get("pose_runtime_flavor", "cpu")).strip().lower()

    def _set_form_row_visible(self, form_layout, field_widget, visible: bool):
        """Show/hide a QFormLayout row by field widget."""
        if form_layout is None or field_widget is None:
            return
        label = form_layout.labelForField(field_widget)
        if label is not None:
            label.setVisible(bool(visible))
        field_widget.setVisible(bool(visible))

    def _sync_pose_backend_ui(self):
        """Show/hide backend-specific pose controls."""
        if not hasattr(self, "combo_pose_model_type"):
            return
        backend = self.combo_pose_model_type.currentText().strip().lower()
        self._populate_pose_runtime_flavor_options(backend=backend)
        is_sleap = backend == "sleap"
        if hasattr(self, "form_pose_runtime") and hasattr(
            self, "pose_sleap_env_row_widget"
        ):
            self._set_form_row_visible(
                self.form_pose_runtime, self.pose_sleap_env_row_widget, is_sleap
            )
        if hasattr(self, "form_pose_runtime") and hasattr(
            self, "pose_sleap_experimental_row_widget"
        ):
            self._set_form_row_visible(
                self.form_pose_runtime,
                self.pose_sleap_experimental_row_widget,
                is_sleap,
            )
        if hasattr(self, "line_pose_model_dir"):
            if is_sleap:
                self.line_pose_model_dir.setPlaceholderText(
                    "Select SLEAP model directory (copied into models/SLEAP)..."
                )
            else:
                self.line_pose_model_dir.setPlaceholderText(
                    "Select YOLO pose weights (.pt, copied into models/YOLO-pose)..."
                )
            # Keep backend-specific pose paths independent.
            self._set_pose_model_path_for_backend(
                self._pose_model_path_for_backend(backend),
                backend=backend,
                update_line=True,
            )
        self._on_runtime_context_changed()

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        return bool(
            self._is_individual_pipeline_enabled()
            and hasattr(self, "chk_enable_pose_extractor")
            and self.chk_enable_pose_extractor.isChecked()
        )

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        has_controls = hasattr(self, "check_video_show_pose") and hasattr(
            self, "combo_video_pose_color_mode"
        )
        if not has_controls:
            return

        video_visible = bool(
            hasattr(self, "check_video_output") and self.check_video_output.isChecked()
        )
        pose_enabled = self._is_pose_inference_enabled()
        enabled = bool(video_visible and pose_enabled)

        self.check_video_show_pose.setEnabled(enabled)
        show_pose = bool(enabled and self.check_video_show_pose.isChecked())
        fixed_color_mode = self.combo_video_pose_color_mode.currentIndex() == 1

        # Show detailed controls only when pose overlay is on.
        self.lbl_video_pose_color_mode.setVisible(show_pose)
        self.combo_video_pose_color_mode.setVisible(show_pose)
        self.lbl_video_pose_point_radius.setVisible(show_pose)
        self.spin_video_pose_point_radius.setVisible(show_pose)
        self.lbl_video_pose_point_thickness.setVisible(show_pose)
        self.spin_video_pose_point_thickness.setVisible(show_pose)
        self.lbl_video_pose_line_thickness.setVisible(show_pose)
        self.spin_video_pose_line_thickness.setVisible(show_pose)

        show_fixed_color = bool(show_pose and fixed_color_mode)
        self.lbl_video_pose_color_label.setVisible(show_fixed_color)
        self.btn_video_pose_color.setVisible(show_fixed_color)
        self.lbl_video_pose_color.setVisible(show_fixed_color)

        self.combo_video_pose_color_mode.setEnabled(show_pose)
        self.spin_video_pose_point_radius.setEnabled(show_pose)
        self.spin_video_pose_point_thickness.setEnabled(show_pose)
        self.spin_video_pose_line_thickness.setEnabled(show_pose)
        self.btn_video_pose_color.setEnabled(show_fixed_color)

        self.lbl_video_pose_disabled_hint.setVisible(video_visible)
        if enabled:
            self.lbl_video_pose_disabled_hint.setText(
                "Pose overlay will use keypoints from pose-augmented tracking output."
            )
        else:
            self.lbl_video_pose_disabled_hint.setText(
                "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings."
            )

    def _is_yolo_detection_mode(self) -> bool:
        """Return True when current detection mode is YOLO OBB."""
        if not hasattr(self, "combo_detection_method"):
            return False
        return self.combo_detection_method.currentIndex() == 1

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline."""
        if not hasattr(self, "chk_enable_individual_analysis"):
            return False
        return bool(
            self.chk_enable_individual_analysis.isChecked()
            and self._is_yolo_detection_mode()
        )

    def _is_individual_image_save_enabled(self) -> bool:
        """Return effective runtime state for saving individual crops."""
        if not hasattr(self, "chk_enable_individual_dataset"):
            return False
        return bool(
            self.chk_enable_individual_dataset.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _should_run_interpolated_postpass(self) -> bool:
        """
        Return True when interpolated post-pass should run.

        We run this pass when interpolation is enabled and either:
        - individual crop saving is enabled, or
        - pose export is enabled (to fill occluded-frame pose rows in final CSV).
        """
        if not hasattr(self, "chk_individual_interpolate"):
            return False
        if not self.chk_individual_interpolate.isChecked():
            return False
        if not self._is_individual_pipeline_enabled():
            return False
        return bool(
            self._is_individual_image_save_enabled() or self._is_pose_export_enabled()
        )

    def _sync_individual_analysis_mode_ui(self):
        """Enforce YOLO-only pipeline and run/save dependency in UI."""
        has_analyze_toggle = hasattr(self, "chk_enable_individual_analysis")
        has_save_toggle = hasattr(self, "chk_enable_individual_dataset")
        is_yolo = self._is_yolo_detection_mode()

        if has_analyze_toggle:
            # Pipeline can only be enabled in YOLO mode.
            if not is_yolo and self.chk_enable_individual_analysis.isChecked():
                self.chk_enable_individual_analysis.blockSignals(True)
                self.chk_enable_individual_analysis.setChecked(False)
                self.chk_enable_individual_analysis.blockSignals(False)
            self.chk_enable_individual_analysis.setEnabled(is_yolo)

        pipeline_enabled = self._is_individual_pipeline_enabled()

        if hasattr(self, "lbl_individual_yolo_only_notice"):
            self.lbl_individual_yolo_only_notice.setVisible(not is_yolo)

        if hasattr(self, "g_pose_runtime"):
            self.g_pose_runtime.setEnabled(pipeline_enabled)
        if hasattr(self, "g_individual_pipeline_common"):
            self.g_individual_pipeline_common.setEnabled(pipeline_enabled)
        self._sync_pose_backend_ui()

        if has_save_toggle:
            if not pipeline_enabled and self.chk_enable_individual_dataset.isChecked():
                self.chk_enable_individual_dataset.blockSignals(True)
                self.chk_enable_individual_dataset.setChecked(False)
                self.chk_enable_individual_dataset.blockSignals(False)
            self.chk_enable_individual_dataset.setEnabled(pipeline_enabled)

        save_enabled = self._is_individual_image_save_enabled()
        if hasattr(self, "ind_output_group"):
            self.ind_output_group.setVisible(save_enabled)
            self.ind_output_group.setEnabled(save_enabled)
        if hasattr(self, "lbl_individual_info"):
            self.lbl_individual_info.setVisible(save_enabled)
        self._sync_video_pose_overlay_controls()
        self._on_runtime_context_changed()

    def _select_individual_background_color(self):
        """Open color picker for individual dataset background color."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        # Convert current BGR to RGB for QColorDialog
        b, g, r = self._background_color
        initial_color = QColor(r, g, b)

        color = QColorDialog.getColor(initial_color, self, "Choose Background Color")
        if color.isValid():
            # Convert RGB back to BGR for OpenCV
            self._background_color = (color.blue(), color.green(), color.red())
            self._update_background_color_button()

    def _select_video_pose_color(self):
        """Open color picker for fixed pose overlay color (BGR)."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        b, g, r = self._video_pose_color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(initial_color, self, "Choose Pose Overlay Color")
        if color.isValid():
            self._video_pose_color = (color.blue(), color.green(), color.red())
            self._update_video_pose_color_button()

    def _update_video_pose_color_button(self):
        """Update fixed pose-color preview button and text label."""
        b, g, r = self._video_pose_color
        self.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self.lbl_video_pose_color.setText(f"{self._video_pose_color}")

    def _update_background_color_button(self):
        """Update the color button display and label."""
        b, g, r = self._background_color
        # Set button color
        self.btn_background_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        # Update label with BGR values
        self.lbl_background_color.setText(f"{self._background_color}")

    def _compute_median_background_color(self):
        """Compute median color from current preview frame or load from video."""
        frame = None

        # Try to use preview frame first
        if (
            hasattr(self, "preview_frame_original")
            and self.preview_frame_original is not None
        ):
            # preview_frame_original is in RGB, convert to BGR for processing
            frame = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2BGR)
        # Otherwise, try to load from video if available
        elif self.current_video_path:
            cap = cv2.VideoCapture(self.current_video_path)
            if cap.isOpened():
                ret, frame_bgr = cap.read()
                cap.release()
                if ret:
                    frame = frame_bgr

        if frame is None:
            QMessageBox.warning(
                self, "No Frame", "Please load a video first to compute median color."
            )
            return

        try:
            from ..utils.image_processing import compute_median_color_from_frame

            # Compute median color
            median_color = compute_median_color_from_frame(frame)
            # Convert numpy.uint8 to regular int for JSON serialization
            self._background_color = tuple(int(c) for c in median_color)
            self._update_background_color_button()

            QMessageBox.information(
                self,
                "Median Color Computed",
                f"Background color set to median:\nBGR: {median_color}",
            )
        except Exception as e:
            logger.error(f"Failed to compute median color: {e}")
            QMessageBox.warning(self, "Error", f"Failed to compute median color:\n{e}")

    def _select_individual_output_dir(self):
        """Browse for individual dataset output directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Individual Dataset Output Directory"
        )
        if directory:
            self.line_individual_output.setText(directory)

    def _on_yolo_batching_toggled(self, state):
        """Enable/disable YOLO batching controls based on checkbox."""
        if self._runtime_requires_fixed_yolo_batch():
            # TensorRT/ONNX runtimes require explicit fixed batch size.
            if not self.chk_enable_yolo_batching.isChecked():
                self.chk_enable_yolo_batching.setChecked(True)
            self.chk_enable_yolo_batching.setEnabled(False)
            self.combo_yolo_batch_mode.setVisible(True)
            self.lbl_yolo_batch_mode.setVisible(True)
            self.spin_yolo_batch_size.setVisible(True)
            self.lbl_yolo_batch_size.setVisible(True)
            self.combo_yolo_batch_mode.setCurrentIndex(1)
            self.combo_yolo_batch_mode.setEnabled(False)
            self.spin_yolo_batch_size.setEnabled(True)
            return

        # Directly check checkbox state for reliability
        enabled = self.chk_enable_yolo_batching.isChecked()

        # Hide/show batching widgets
        self.combo_yolo_batch_mode.setVisible(enabled)
        self.lbl_yolo_batch_mode.setVisible(enabled)
        self.spin_yolo_batch_size.setVisible(enabled)
        self.lbl_yolo_batch_size.setVisible(enabled)

        # Also control enable state
        self.combo_yolo_batch_mode.setEnabled(enabled)
        # Manual batch size only enabled if batching is on AND mode is Manual
        manual_mode = self.combo_yolo_batch_mode.currentIndex() == 1
        self.spin_yolo_batch_size.setEnabled(enabled and manual_mode)

    def _on_yolo_manual_batch_size_changed(self, value: int):
        """Keep legacy fixed-batch field synchronized for fixed runtimes."""
        if self._runtime_requires_fixed_yolo_batch() and hasattr(
            self, "spin_tensorrt_batch"
        ):
            self.spin_tensorrt_batch.setValue(int(value))

    def _on_yolo_batch_mode_changed(self, index):
        """Show/hide manual batch size based on selected mode."""
        if self._runtime_requires_fixed_yolo_batch():
            # TensorRT/ONNX runtimes require explicit fixed batch size.
            if self.combo_yolo_batch_mode.currentIndex() != 1:
                self.combo_yolo_batch_mode.setCurrentIndex(1)
            self.spin_yolo_batch_size.setEnabled(True)
            return
        # index 0 = Auto, index 1 = Manual
        is_manual = index == 1
        batching_enabled = self.chk_enable_yolo_batching.isChecked()
        self.spin_yolo_batch_size.setEnabled(batching_enabled and is_manual)

    def _on_tensorrt_toggled(self, state):
        """Enable/disable TensorRT batch size control based on checkbox."""
        # TensorRT toggles are now derived from canonical compute runtime.
        # Keep legacy widgets hidden from UI.
        if not self.chk_enable_tensorrt.isVisible():
            self.spin_tensorrt_batch.setVisible(False)
            self.lbl_tensorrt_batch.setVisible(False)
            return

        # Directly check checkbox state for reliability
        enabled = self.chk_enable_tensorrt.isChecked()

        # Hide/show TensorRT batch size widgets
        self.spin_tensorrt_batch.setVisible(enabled)
        self.lbl_tensorrt_batch.setVisible(enabled)

        # Also control enable state
        self.spin_tensorrt_batch.setEnabled(enabled)
        self.lbl_tensorrt_batch.setEnabled(enabled)

    def _on_cleaning_toggled(self, state):
        """Enable/disable trajectory cleaning controls based on checkbox."""
        enabled = self.enable_postprocessing.isChecked()

        # Hide/show all cleaning parameter widgets
        self.spin_min_trajectory_length.setVisible(enabled)
        self.lbl_min_trajectory_length.setVisible(enabled)
        self.spin_max_velocity_break.setVisible(enabled)
        self.lbl_max_velocity_break.setVisible(enabled)
        self.spin_max_occlusion_gap.setVisible(enabled)
        self.lbl_max_occlusion_gap.setVisible(enabled)
        self.spin_max_velocity_zscore.setVisible(enabled)
        self.lbl_max_velocity_zscore.setVisible(enabled)
        self.spin_velocity_zscore_window.setVisible(enabled)
        self.lbl_velocity_zscore_window.setVisible(enabled)
        self.spin_velocity_zscore_min_vel.setVisible(enabled)
        self.lbl_velocity_zscore_min_vel.setVisible(enabled)
        self.combo_interpolation_method.setVisible(enabled)
        self.lbl_interpolation_method.setVisible(enabled)
        self.spin_interpolation_max_gap.setVisible(enabled)
        self.lbl_interpolation_max_gap.setVisible(enabled)
        self.spin_merge_overlap_multiplier.setVisible(enabled)
        self.lbl_merge_overlap_multiplier.setVisible(enabled)
        self.spin_min_overlap_frames.setVisible(enabled)
        self.lbl_min_overlap_frames.setVisible(enabled)
        self.chk_cleanup_temp_files.setVisible(enabled)

        # Also control enable state
        self.spin_min_trajectory_length.setEnabled(enabled)
        self.spin_max_velocity_break.setEnabled(enabled)
        self.spin_max_occlusion_gap.setEnabled(enabled)
        self.spin_max_velocity_zscore.setEnabled(enabled)
        self.spin_velocity_zscore_window.setEnabled(enabled)
        self.spin_velocity_zscore_min_vel.setEnabled(enabled)
        self.combo_interpolation_method.setEnabled(enabled)
        self.spin_interpolation_max_gap.setEnabled(enabled)
        self.spin_merge_overlap_multiplier.setEnabled(enabled)
        self.spin_min_overlap_frames.setEnabled(enabled)
        self.chk_cleanup_temp_files.setEnabled(enabled)

    # =========================================================================
    # EVENT HANDLERS (Identical Logic to Original)
    # =========================================================================

    def _on_detection_method_changed_ui(self, index):
        """Update stack widget when detection method changes."""
        self.stack_detection.setCurrentIndex(index)
        # Show image adjustments only for Background Subtraction (index 0)
        is_background_subtraction = index == 0
        self.g_img.setVisible(is_background_subtraction)
        # Show/hide method-specific overlay groups
        self.g_overlays_bg.setVisible(is_background_subtraction)
        self.g_overlays_yolo.setVisible(not is_background_subtraction)
        # Refresh preview to show correct mode
        self._update_preview_display()
        self.on_detection_method_changed(index)
        self._on_runtime_context_changed()

    def select_file(self: object) -> object:
        """Select video file via file dialog."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if fp:
            self._setup_video_file(fp)

    def _setup_video_file(self, fp, skip_config_load=False):
        """
        Setup a video file for tracking.

        Args:
            fp: Path to the video file
            skip_config_load: If True, skip auto-loading config (used when loading config itself)
        """
        self.file_line.setText(fp)
        self.current_video_path = fp
        if self.roi_selection_active:
            self.clear_roi()

        # Auto-generate output paths based on video name
        video_dir = os.path.dirname(fp)
        video_name = os.path.splitext(os.path.basename(fp))[0]

        # Auto-populate CSV output
        csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
        self.csv_line.setText(csv_path)

        # Auto-populate video output and enable it
        video_out_path = os.path.join(video_dir, f"{video_name}_tracking.mp4")
        self.video_out_line.setText(video_out_path)
        self.check_video_output.setChecked(True)

        # Enable preview detection button
        self.btn_test_detection.setEnabled(True)
        self.btn_detect_fps.setEnabled(True)

        # Initialize video player
        self._init_video_player(fp)

        # Auto-load config if it exists for this video (unless explicitly skipped)
        if not skip_config_load:
            config_path = get_video_config_path(fp)
            if config_path and os.path.isfile(config_path):
                self._load_config_from_file(config_path)
                self.config_status_label.setText(
                    f"✓ Loaded: {os.path.basename(config_path)}"
                )
                self.config_status_label.setStyleSheet(
                    "color: #4a9eff; font-style: italic; font-size: 10px;"
                )
                logger.info(
                    f"Video selected: {fp} (auto-loaded config from {config_path})"
                )
            else:
                self.config_status_label.setText(
                    "No config found (using current settings)"
                )
                self.config_status_label.setStyleSheet(
                    "color: #f39c12; font-style: italic; font-size: 10px;"
                )
                logger.info(
                    f"Video selected: {fp} (no config found, using current settings)"
                )

        # Enable full UI now that a video is loaded
        self._apply_ui_state("idle")

    def select_csv(self: object) -> object:
        """select_csv method documentation."""
        fp, _ = QFileDialog.getSaveFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if fp:
            self.csv_line.setText(fp)

    def select_video_output(self: object) -> object:
        """select_video_output method documentation."""
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self.video_out_line.setText(fp)

    def _load_preview_frame(self):
        """Load a random frame from the video for live preview."""
        if not self.current_video_path:
            return

        cap = cv2.VideoCapture(self.current_video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for preview")
            return

        # Get total frames and pick a random one
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            random_frame_idx = np.random.randint(0, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)

        ret, frame = cap.read()
        cap.release()

        if ret:
            # Store original frame for adjustments
            self.preview_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Clear any previous detection test result
            self.detection_test_result = None
            self._update_preview_display()
            # Auto-fit to screen - use QTimer to ensure display is updated first
            from PySide6.QtCore import QTimer

            QTimer.singleShot(10, self._fit_image_to_screen)
            logger.info(f"Loaded preview frame {random_frame_idx}/{total_frames}")
        else:
            logger.warning("Failed to read preview frame")

    # =========================================================================
    # VIDEO PLAYER FUNCTIONS
    # =========================================================================

    def _init_video_player(self, video_path):
        """Initialize video player with the loaded video."""
        # Release any existing video capture
        if self.video_cap is not None:
            self.video_cap.release()

        # Stop any active playback
        if self.playback_timer:
            self.playback_timer.stop()
            self.playback_timer = None
        self.is_playing = False

        # Open video
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        # Get video properties
        self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update UI
        self.lbl_video_info.setText(
            f"Video: {self.video_total_frames} frames, {width}x{height}, {fps:.2f} FPS"
        )

        # Enable controls
        self.slider_timeline.setMaximum(self.video_total_frames - 1)
        self.slider_timeline.setEnabled(True)
        self.btn_first_frame.setEnabled(True)
        self.btn_prev_frame.setEnabled(True)
        self.btn_play_pause.setEnabled(True)
        self.btn_next_frame.setEnabled(True)
        self.btn_last_frame.setEnabled(True)
        self.btn_random_seek.setEnabled(True)
        self.combo_playback_speed.setEnabled(True)

        # Enable frame range controls
        self.spin_start_frame.setMaximum(self.video_total_frames - 1)
        self.spin_start_frame.setEnabled(True)
        self.spin_end_frame.setMaximum(self.video_total_frames - 1)
        self.spin_end_frame.setValue(self.video_total_frames - 1)
        self.spin_end_frame.setEnabled(True)
        self.btn_set_start_current.setEnabled(True)
        self.btn_set_end_current.setEnabled(True)
        self.btn_reset_range.setEnabled(True)

        # Show video player group
        self.g_video_player.setVisible(True)

        # Go to first frame
        self.video_current_frame_idx = 0
        self._display_current_frame()
        self._update_range_info()

        logger.info(f"Video player initialized: {self.video_total_frames} frames")

    def _display_current_frame(self):
        """Display the current frame in the video label."""
        if self.video_cap is None:
            return

        # Only seek if not reading sequentially (seeking is slow)
        if self.last_read_frame_idx != self.video_current_frame_idx - 1:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.video_current_frame_idx)

        ret, frame = self.video_cap.read()

        if not ret:
            return

        self.last_read_frame_idx = self.video_current_frame_idx

        # Convert to RGB and update preview
        self.preview_frame_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.detection_test_result = None  # Clear any detection overlay
        self._update_preview_display()

        # Update UI
        self.lbl_current_frame.setText(
            f"Frame: {self.video_current_frame_idx}/{self.video_total_frames - 1}"
        )
        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(self.video_current_frame_idx)
        self.slider_timeline.blockSignals(False)

    def _on_timeline_changed(self, value):
        """Handle timeline slider change."""
        # Only stop playback if this is a manual user change (not from playback itself)
        if self.is_playing and not self.slider_timeline.signalsBlocked():
            self._stop_playback()

        self.video_current_frame_idx = value
        self._display_current_frame()

    def _goto_first_frame(self):
        """Go to the first frame."""
        if self.is_playing:
            self._stop_playback()
        self.video_current_frame_idx = 0
        self.slider_timeline.setValue(0)
        self._display_current_frame()

    def _goto_prev_frame(self):
        """Go to the previous frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_current_frame_idx > 0:
            self.video_current_frame_idx -= 1
            self.slider_timeline.setValue(self.video_current_frame_idx)
            self._display_current_frame()

    def _goto_next_frame(self):
        """Go to the next frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_current_frame_idx < self.video_total_frames - 1:
            self.video_current_frame_idx += 1
            self.slider_timeline.setValue(self.video_current_frame_idx)
            self._display_current_frame()

    def _goto_last_frame(self):
        """Go to the last frame."""
        if self.is_playing:
            self._stop_playback()
        self.video_current_frame_idx = self.video_total_frames - 1
        self.slider_timeline.setValue(self.video_current_frame_idx)
        self._display_current_frame()

    def _goto_random_frame(self):
        """Jump to a random frame."""
        if self.is_playing:
            self._stop_playback()
        if self.video_total_frames <= 0:
            return
        self.video_current_frame_idx = np.random.randint(0, self.video_total_frames)
        self.slider_timeline.setValue(self.video_current_frame_idx)
        self._display_current_frame()

    def _toggle_playback(self):
        """Toggle play/pause."""
        if self.is_playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        """Start video playback."""
        if self.video_cap is None or self.is_playing:
            return

        self.is_playing = True
        self.btn_play_pause.setText("⏸ Pause")

        # Get playback speed
        speed_text = self.combo_playback_speed.currentText()
        speed = float(speed_text.replace("x", ""))

        # Calculate interval based on FPS and speed
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Default

        interval_ms = max(1, int((1000.0 / fps) / speed))

        # Create timer if needed (use as single-shot timer)
        if self.playback_timer is None:
            self.playback_timer = QTimer(self)

        # Start first frame with single-shot
        self.playback_timer.singleShot(interval_ms, self._playback_step)
        logger.debug(f"Started playback at {speed}x speed ({interval_ms}ms interval)")

    def _stop_playback(self):
        """Stop video playback."""
        if not self.is_playing:
            return

        self.is_playing = False
        self.btn_play_pause.setText("▶ Play")

        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()

        logger.debug("Stopped playback")

    def _playback_step(self):
        """Advance one frame during playback."""
        # Stop timer first to prevent event queueing
        if self.playback_timer and self.playback_timer.isActive():
            self.playback_timer.stop()

        # Check if still playing (user might have stopped it)
        if not self.is_playing:
            return

        if self.video_current_frame_idx < self.video_total_frames - 1:
            self.video_current_frame_idx += 1
            self._display_current_frame()

            # Process events to keep UI responsive
            from PySide6.QtWidgets import QApplication

            QApplication.processEvents()

            # Re-check if still playing after processing events
            if self.is_playing and self.playback_timer:
                # Calculate next interval
                speed_text = self.combo_playback_speed.currentText()
                speed = float(speed_text.replace("x", ""))
                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                interval_ms = max(1, int((1000.0 / fps) / speed))

                # Schedule next frame
                self.playback_timer.singleShot(interval_ms, self._playback_step)
        else:
            # Reached end of video
            self._stop_playback()

    def _on_frame_range_changed(self):
        """Handle frame range spinbox changes."""
        # Ensure start <= end
        if self.spin_start_frame.value() > self.spin_end_frame.value():
            self.spin_end_frame.setValue(self.spin_start_frame.value())

        self._update_range_info()

    def _update_range_info(self):
        """Update the frame range info label."""
        start = self.spin_start_frame.value()
        end = self.spin_end_frame.value()
        num_frames = end - start + 1

        fps = self.spin_fps.value()
        duration_sec = num_frames / fps if fps > 0 else 0

        self.lbl_range_info.setText(
            f"Tracking {num_frames} frames ({duration_sec:.2f} seconds)"
        )

    def _set_start_to_current(self):
        """Set start frame to current frame."""
        self.spin_start_frame.setValue(self.video_current_frame_idx)

    def _set_end_to_current(self):
        """Set end frame to current frame."""
        self.spin_end_frame.setValue(self.video_current_frame_idx)

    def _reset_frame_range(self):
        """Reset frame range to full video."""
        self.spin_start_frame.setValue(0)
        self.spin_end_frame.setValue(self.video_total_frames - 1)

    def _on_brightness_changed(self, value):
        """Handle brightness slider change."""
        self.label_brightness_val.setText(str(value))
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_contrast_changed(self, value):
        """Handle contrast slider change."""
        contrast_val = value / 100.0
        self.label_contrast_val.setText(f"{contrast_val:.2f}")
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_gamma_changed(self, value):
        """Handle gamma slider change."""
        gamma_val = value / 100.0
        self.label_gamma_val.setText(f"{gamma_val:.2f}")
        self.detection_test_result = None  # Clear test result
        self._update_preview_display()

    def _on_zoom_changed(self, value):
        """Handle zoom slider change."""
        zoom_val = value / 100.0
        self.label_zoom_val.setText(f"{zoom_val:.2f}x")
        # If detection test result exists, redisplay it; otherwise show preview
        if self.detection_test_result is not None:
            self._redisplay_detection_test()
        elif self.roi_base_frame is not None and self.roi_shapes:
            # If ROI is active but no preview frame, show ROI base frame with mask
            self._display_roi_with_zoom()
        else:
            self._update_preview_display()

    def _update_body_size_info(self):
        """Update the info label showing calculated body area."""
        import math

        body_size = self.spin_reference_body_size.value()
        body_area = math.pi * (body_size / 2.0) ** 2
        self.label_body_size_info.setText(
            f"≈ {body_area:.1f} px² area (all size/distance params scale with this)"
        )

    def _update_fps_info(self):
        """Update the FPS info label with time per frame."""
        fps = self.spin_fps.value()
        time_per_frame = 1000.0 / fps  # milliseconds
        self.label_fps_info.setText(f"= {time_per_frame:.2f} ms per frame")

    def _detect_fps_from_current_video(self):
        """Detect and set FPS from the currently loaded video."""
        if not self.current_video_path:
            QMessageBox.warning(
                self, "No Video Loaded", "Please load a video file first."
            )
            return

        detected_fps = self._auto_detect_fps(self.current_video_path)
        if detected_fps is not None:
            self.spin_fps.setValue(detected_fps)
            QMessageBox.information(
                self,
                "FPS Detected",
                f"Frame rate detected: {detected_fps:.2f} FPS\n\n"
                f"Time per frame: {1000.0 / detected_fps:.2f} ms",
            )

    def _auto_detect_fps(self, video_path):
        """Auto-detect FPS from video metadata and return the value."""
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if fps > 0:
                    logger.info(f"Detected FPS from video: {fps:.2f}")
                    return fps
                else:
                    logger.warning("Could not detect FPS from video metadata")
                    return None
            else:
                logger.warning("Could not open video for FPS detection")
                return None
        except Exception as e:
            logger.error(f"Error detecting FPS: {e}")

    def _update_detection_stats(self, detected_dimensions, resize_factor=1.0):
        """Update detection statistics display.

        Args:
            detected_dimensions: List of (major_axis, minor_axis) tuples in pixels
            resize_factor: Factor by which frame was resized (to scale back to original)
        """
        if not detected_dimensions or len(detected_dimensions) == 0:
            self.label_detection_stats.setText(
                "No detections found.\nAdjust parameters and try again."
            )
            self.btn_auto_set_body_size.setEnabled(False)
            self.detected_sizes = None
            return

        # Scale dimensions back to original resolution (resize_factor < 1 means frame was downscaled)
        # Linear dimensions scale with resize_factor
        scale_factor = 1.0 / resize_factor
        major_axes = [dims[0] * scale_factor for dims in detected_dimensions]
        minor_axes = [dims[1] * scale_factor for dims in detected_dimensions]

        # Calculate aspect ratios
        aspect_ratios = [
            major / minor if minor > 0 else 1.0
            for major, minor in zip(major_axes, minor_axes)
        ]

        # Calculate geometric mean as representative body size (better than assuming circular)
        geometric_means = [
            math.sqrt(major * minor) for major, minor in zip(major_axes, minor_axes)
        ]

        # Calculate statistics for each dimension
        stats = {
            "major": {
                "mean": np.mean(major_axes),
                "median": np.median(major_axes),
                "std": np.std(major_axes),
                "min": np.min(major_axes),
                "max": np.max(major_axes),
            },
            "minor": {
                "mean": np.mean(minor_axes),
                "median": np.median(minor_axes),
                "std": np.std(minor_axes),
                "min": np.min(minor_axes),
                "max": np.max(minor_axes),
            },
            "aspect_ratio": {
                "mean": np.mean(aspect_ratios),
                "median": np.median(aspect_ratios),
                "std": np.std(aspect_ratios),
            },
            "geometric_mean": {
                "mean": np.mean(geometric_means),
                "median": np.median(geometric_means),
                "std": np.std(geometric_means),
            },
        }

        # Store for auto-set
        self.detected_sizes = {
            "major_axes": major_axes,
            "minor_axes": minor_axes,
            "aspect_ratios": aspect_ratios,
            "geometric_means": geometric_means,
            "stats": stats,
            "count": len(detected_dimensions),
            "resize_factor": resize_factor,
            "recommended_body_size": stats["geometric_mean"][
                "median"
            ],  # Use geometric mean median
        }

        # Update label with comprehensive statistics
        stats_text = (
            f"Analyzed {len(detected_dimensions)} detections:\n\n"
            f"Major Axis (length):\n"
            f"  • Median: {stats['major']['median']:.1f} px  (range: {stats['major']['min']:.1f} - {stats['major']['max']:.1f})\n"
            f"  • Mean: {stats['major']['mean']:.1f} ± {stats['major']['std']:.1f} px\n\n"
            f"Minor Axis (width):\n"
            f"  • Median: {stats['minor']['median']:.1f} px  (range: {stats['minor']['min']:.1f} - {stats['minor']['max']:.1f})\n"
            f"  • Mean: {stats['minor']['mean']:.1f} ± {stats['minor']['std']:.1f} px\n\n"
            f"Aspect Ratio (length/width):\n"
            f"  • Median: {stats['aspect_ratio']['median']:.2f}  Mean: {stats['aspect_ratio']['mean']:.2f} ± {stats['aspect_ratio']['std']:.2f}\n\n"
            f"Recommended Body Size: {stats['geometric_mean']['median']:.1f} px\n"
            f"  (geometric mean of dimensions)"
        )
        self.label_detection_stats.setText(stats_text)
        self.btn_auto_set_body_size.setEnabled(True)

    def _auto_set_body_size_from_detection(self):
        """Auto-set reference body size from detected geometric mean."""
        if self.detected_sizes is None:
            return

        recommended_size = self.detected_sizes["recommended_body_size"]
        stats = self.detected_sizes["stats"]
        self.spin_reference_body_size.setValue(recommended_size)

        # Show confirmation with aspect ratio info
        QMessageBox.information(
            self,
            "Body Size Updated",
            f"Reference body size set to {recommended_size:.1f} px\n"
            f"(geometric mean of {self.detected_sizes['count']} detections)\n\n"
            f"Detected dimensions:\n"
            f"  • Major axis: {stats['major']['median']:.1f} px\n"
            f"  • Minor axis: {stats['minor']['median']:.1f} px\n"
            f"  • Aspect ratio: {stats['aspect_ratio']['median']:.2f}\n\n"
            f"All distance/size parameters will now scale relative to this value.",
        )

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        # Hide/show all video output widgets
        self.btn_video_out.setVisible(checked)
        self.video_out_line.setVisible(checked)
        self.lbl_video_path.setVisible(checked)
        self.lbl_video_viz_settings.setVisible(checked)
        self.check_show_labels.setVisible(checked)
        self.check_show_orientation.setVisible(checked)
        self.check_show_trails.setVisible(checked)
        self.spin_trail_duration.setVisible(checked)
        self.lbl_trail_duration.setVisible(checked)
        self.spin_marker_size.setVisible(checked)
        self.lbl_marker_size.setVisible(checked)
        self.spin_text_scale.setVisible(checked)
        self.lbl_text_scale.setVisible(checked)
        self.spin_arrow_length.setVisible(checked)
        self.lbl_arrow_length.setVisible(checked)
        self.lbl_video_pose_settings.setVisible(checked)
        self.check_video_show_pose.setVisible(checked)
        self.lbl_video_pose_color_mode.setVisible(checked)
        self.combo_video_pose_color_mode.setVisible(checked)
        self.lbl_video_pose_color_label.setVisible(checked)
        self.btn_video_pose_color.setVisible(checked)
        self.lbl_video_pose_color.setVisible(checked)
        self.lbl_video_pose_point_radius.setVisible(checked)
        self.spin_video_pose_point_radius.setVisible(checked)
        self.lbl_video_pose_point_thickness.setVisible(checked)
        self.spin_video_pose_point_thickness.setVisible(checked)
        self.lbl_video_pose_line_thickness.setVisible(checked)
        self.spin_video_pose_line_thickness.setVisible(checked)
        self.lbl_video_pose_disabled_hint.setVisible(checked)

        # Also control enable state
        self.btn_video_out.setEnabled(checked)
        self.video_out_line.setEnabled(checked)
        self._sync_video_pose_overlay_controls()

    def _update_preview_display(self):
        """Update the video display with current brightness/contrast/gamma settings."""
        if self.preview_frame_original is None:
            return

        # If we have a detection test result, redisplay it with the new zoom
        if self.detection_test_result is not None:
            self._redisplay_detection_test()
            return

        # Get current adjustment values
        brightness = self.slider_brightness.value()
        contrast = self.slider_contrast.value() / 100.0
        gamma = self.slider_gamma.value() / 100.0

        # Get detection method
        detection_method = self.combo_detection_method.currentText()
        is_background_subtraction = detection_method == "Background Subtraction"

        # Apply adjustments
        from ..utils.image_processing import apply_image_adjustments

        if is_background_subtraction:
            # Background subtraction uses grayscale with adjustments
            gray = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2GRAY)
            # GPU not needed for preview - single frame
            adjusted = apply_image_adjustments(
                gray, brightness, contrast, gamma, use_gpu=False
            )
            # Convert back to RGB for display
            adjusted_rgb = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2RGB)
        else:
            # YOLO uses color frames directly without brightness/contrast/gamma adjustments
            adjusted_rgb = self.preview_frame_original

        # Display the adjusted frame
        h, w, ch = adjusted_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(adjusted_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Apply ROI mask if exists
        if self.roi_mask is not None:
            qimg = self._apply_roi_mask_to_image(qimg)

        # Apply zoom (always use fast transformation for responsive UI)
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _redisplay_detection_test(self):
        """Redisplay the stored detection test result with current zoom."""
        if self.detection_test_result is None:
            return

        test_frame_rgb, resize_f = self.detection_test_result
        h, w, ch = test_frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Apply zoom (always use fast transformation for responsive UI)
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        effective_scale = zoom_val * resize_f

        if effective_scale != 1.0:
            orig_h, orig_w = self.preview_frame_original.shape[:2]
            scaled_w = int(orig_w * effective_scale)
            scaled_h = int(orig_h * effective_scale)
            qimg = qimg.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.FastTransformation
            )

        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap)

    def _test_detection_on_preview(self):
        """Test detection algorithm on the current preview frame."""
        if self.preview_frame_original is None:
            logger.warning("No preview frame loaded")
            return

        # If size filtering is enabled, ask user whether to use it for the test
        use_size_filtering = False
        if self.chk_size_filtering.isChecked():
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Size Filtering Options")
            msg.setText("Size filtering is currently enabled!")
            msg.setInformativeText(
                "For accurate size estimation, it's recommended to run detection\n"
                "WITHOUT size constraints. However, you can test with constraints\n"
                "if you want to see how filtering affects the results.\n\n"
                "How would you like to proceed?"
            )

            btn_without = msg.addButton(
                "NO Size Filtering (Recommended)", QMessageBox.AcceptRole
            )
            btn_with = msg.addButton("WITH Size Filtering", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_without)

            msg.exec()
            clicked = msg.clickedButton()

            if clicked == btn_cancel:
                return
            elif clicked == btn_with:
                use_size_filtering = True
                logger.info("Running detection test WITH size filtering enabled")
            else:  # btn_without
                use_size_filtering = False
                logger.info(
                    "Running detection test WITHOUT size filtering (recommended for size estimation)"
                )

        from ..core.background.model import BackgroundModel
        from ..core.detectors.engine import YOLOOBBDetector
        from ..utils.image_processing import apply_image_adjustments

        # Convert RGB preview to BGR for OpenCV
        frame_bgr = cv2.cvtColor(self.preview_frame_original, cv2.COLOR_RGB2BGR)

        # Get current parameters
        detection_method = self.combo_detection_method.currentIndex()
        is_background_subtraction = detection_method == 0

        # Create a copy for visualization
        test_frame = frame_bgr.copy()

        try:
            if is_background_subtraction:
                # Build actual background model using priming frames
                logger.info("Building background model for test detection...")

                # Open video to sample priming frames
                video_path = self.file_line.text()
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error("Cannot open video for background priming")
                    return

                # Build parameters dict for BackgroundModel
                bg_params = {
                    "BACKGROUND_PRIME_FRAMES": self.spin_bg_prime.value(),
                    "BRIGHTNESS": self.slider_brightness.value(),
                    "CONTRAST": self.slider_contrast.value() / 100.0,
                    "GAMMA": self.slider_gamma.value() / 100.0,
                    "ROI_MASK": self.roi_mask,
                    "RESIZE_FACTOR": self.spin_resize.value(),
                    "DARK_ON_LIGHT_BACKGROUND": self.chk_dark_on_light.isChecked(),
                    "THRESHOLD_VALUE": self.spin_threshold.value(),
                    "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
                    "ENABLE_ADDITIONAL_DILATION": self.chk_additional_dilation.isChecked(),
                    "DILATION_KERNEL_SIZE": self.spin_dilation_kernel_size.value(),
                    "DILATION_ITERATIONS": self.spin_dilation_iterations.value(),
                }

                # Create and prime background model
                bg_model = BackgroundModel(bg_params)
                bg_model.prime_background(cap)

                if bg_model.lightest_background is None:
                    logger.error("Failed to build background model")
                    cap.release()
                    return

                # Now process the preview frame with the primed background
                # Need to resize frame to match background dimensions if resize factor is set
                resize_f = bg_params["RESIZE_FACTOR"]
                frame_to_process = frame_bgr.copy()
                if resize_f < 1.0:
                    frame_to_process = cv2.resize(
                        frame_to_process,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )
                    # Also resize the test_frame for visualization
                    test_frame = cv2.resize(
                        test_frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                gray = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2GRAY)
                # GPU not needed for single-frame test
                gray = apply_image_adjustments(
                    gray,
                    bg_params["BRIGHTNESS"],
                    bg_params["CONTRAST"],
                    bg_params["GAMMA"],
                    use_gpu=False,
                )

                # Apply ROI mask if exists (resize it too if needed)
                roi_for_test = self.roi_mask
                if self.roi_mask is not None:
                    if resize_f < 1.0:
                        roi_for_test = cv2.resize(
                            self.roi_mask,
                            (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    gray = cv2.bitwise_and(gray, gray, mask=roi_for_test)

                # Get background (use lightest_background as starting point)
                bg_u8 = bg_model.lightest_background.astype(np.uint8)

                # Generate foreground mask (includes morphology operations)
                fg_mask = bg_model.generate_foreground_mask(gray, bg_u8)

                # Find contours
                cnts, _ = cv2.findContours(
                    fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                min_contour = self.spin_min_contour.value()
                detections = []
                detected_dimensions = (
                    []
                )  # Collect (major, minor) axis pairs for statistics

                # Size filtering is based on body area multipliers, not mm²
                # Calculate pixel areas from reference body size
                reference_body_size = self.spin_reference_body_size.value()
                reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
                scaled_body_area = reference_body_area * (resize_f**2)

                if use_size_filtering:
                    min_size_multiplier = self.spin_min_object_size.value()
                    max_size_multiplier = self.spin_max_object_size.value()
                    min_size_px2 = min_size_multiplier * scaled_body_area
                    max_size_px2 = max_size_multiplier * scaled_body_area
                    logger.info("Background subtraction size filtering ENABLED:")
                    logger.info(f"  Resize factor: {resize_f:.2f}")
                    logger.info(f"  Reference body size: {reference_body_size:.1f} px")
                    logger.info(
                        f"  Reference body area (original): {reference_body_area:.1f} px²"
                    )
                    logger.info(
                        f"  Reference body area (resized): {scaled_body_area:.1f} px²"
                    )
                    logger.info(
                        f"  Min size multiplier: {min_size_multiplier:.2f}× → {min_size_px2:.1f} px²"
                    )
                    logger.info(
                        f"  Max size multiplier: {max_size_multiplier:.2f}× → {max_size_px2:.1f} px²"
                    )

                filtered_count = 0
                detection_num = 0
                for c in cnts:
                    area = cv2.contourArea(c)
                    if area < min_contour or len(c) < 5:
                        continue

                    # Apply size filtering based on user choice
                    if use_size_filtering:
                        # Compare pixel area to size thresholds (already in pixels)
                        passes_filter = min_size_px2 <= area <= max_size_px2

                        if detection_num < 5:  # Log first 5 detections for debugging
                            logger.info(
                                f"  Detection {detection_num + 1}: {area:.1f} px² (range: {min_size_px2:.1f}-{max_size_px2:.1f}) - {'PASS' if passes_filter else 'FILTERED OUT'}"
                            )

                        detection_num += 1

                        if not passes_filter:
                            filtered_count += 1
                            continue

                    # Fit ellipse
                    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
                    detections.append(((cx, cy), (ax1, ax2), ang, area))
                    # Store major and minor axes (fitEllipse returns full axes, not semi-axes)
                    major_axis = max(ax1, ax2)
                    minor_axis = min(ax1, ax2)
                    detected_dimensions.append((major_axis, minor_axis))

                    # Draw ellipse
                    cv2.ellipse(
                        test_frame,
                        ((int(cx), int(cy)), (int(ax1), int(ax2)), ang),
                        (0, 255, 0),
                        2,
                    )
                    cv2.circle(test_frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)

                # Show foreground mask in corner
                small_fg = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
                test_frame[0 : small_fg.shape[0], 0 : small_fg.shape[1]] = cv2.cvtColor(
                    small_fg, cv2.COLOR_GRAY2BGR
                )

                # Show estimated background in opposite corner
                small_bg = cv2.resize(bg_u8, (0, 0), fx=0.3, fy=0.3)
                bg_bgr = cv2.cvtColor(small_bg, cv2.COLOR_GRAY2BGR)
                test_frame[0 : bg_bgr.shape[0], -bg_bgr.shape[1] :] = bg_bgr

                # Add detection count and note
                cv2.putText(
                    test_frame,
                    f"Detections: {len(detections)} (BG from {self.spin_bg_prime.value()} frames)",
                    (10, test_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                cap.release()

                if use_size_filtering:
                    logger.info(
                        f"Background subtraction test complete: {len(detections)} detections passed size filter, {filtered_count} filtered out"
                    )
                else:
                    logger.info(
                        f"Background subtraction test complete: {len(detections)} detections"
                    )

                # Update detection statistics (scale dimensions back to original resolution)
                self._update_detection_stats(detected_dimensions, resize_f)
            else:
                # YOLO Detection
                # Apply resize factor (same as tracking does)
                resize_f = self.spin_resize.value()
                frame_to_process = frame_bgr.copy()
                if resize_f < 1.0:
                    frame_to_process = cv2.resize(
                        frame_to_process,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )
                    # Also resize the test_frame for visualization
                    test_frame = cv2.resize(
                        test_frame,
                        (0, 0),
                        fx=resize_f,
                        fy=resize_f,
                        interpolation=cv2.INTER_AREA,
                    )

                # Build parameters for YOLO
                # Convert size multipliers to pixel areas for detector (same as full tracking run)
                reference_body_size = self.spin_reference_body_size.value()
                reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
                scaled_body_area = reference_body_area * (resize_f**2)

                if use_size_filtering:
                    min_size_px2 = int(
                        self.spin_min_object_size.value() * scaled_body_area
                    )
                    max_size_px2 = int(
                        self.spin_max_object_size.value() * scaled_body_area
                    )
                else:
                    min_size_px2 = 0
                    max_size_px2 = float("inf")

                runtime_detection = derive_detection_runtime_settings(
                    self._selected_compute_runtime()
                )
                trt_batch_size = (
                    self.spin_yolo_batch_size.value()
                    if self._runtime_requires_fixed_yolo_batch(
                        self._selected_compute_runtime()
                    )
                    else self.spin_tensorrt_batch.value()
                )
                yolo_params = {
                    "YOLO_MODEL_PATH": resolve_model_path(
                        self._get_selected_yolo_model_path()
                    ),
                    "YOLO_CONFIDENCE_THRESHOLD": self.spin_yolo_confidence.value(),
                    "YOLO_IOU_THRESHOLD": self.spin_yolo_iou.value(),
                    "USE_CUSTOM_OBB_IOU_FILTERING": True,
                    "YOLO_TARGET_CLASSES": (
                        [
                            int(x.strip())
                            for x in self.line_yolo_classes.text().split(",")
                        ]
                        if self.line_yolo_classes.text().strip()
                        else None
                    ),
                    "YOLO_DEVICE": runtime_detection["yolo_device"],
                    "ENABLE_GPU_BACKGROUND": runtime_detection["enable_gpu_background"],
                    "ENABLE_TENSORRT": runtime_detection["enable_tensorrt"],
                    "ENABLE_ONNX_RUNTIME": runtime_detection["enable_onnx_runtime"],
                    "TENSORRT_MAX_BATCH_SIZE": trt_batch_size,
                    "MAX_TARGETS": self.spin_max_targets.value(),
                    "MAX_CONTOUR_MULTIPLIER": self.spin_max_contour_multiplier.value(),
                    "ENABLE_SIZE_FILTERING": use_size_filtering,  # Use the user's choice
                    "MIN_OBJECT_SIZE": min_size_px2,  # Already converted to pixels
                    "MAX_OBJECT_SIZE": max_size_px2,  # Already converted to pixels
                }

                # Prepare ROI mask for filtering (resize if needed)
                roi_for_yolo = None
                if self.roi_mask is not None:
                    roi_for_yolo = self.roi_mask
                    if resize_f < 1.0:
                        roi_for_yolo = cv2.resize(
                            self.roi_mask,
                            (frame_to_process.shape[1], frame_to_process.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )

                # Log size filtering parameters if enabled
                if use_size_filtering:
                    # Logging parameters (already calculated above when building yolo_params)
                    min_size_multiplier = self.spin_min_object_size.value()
                    max_size_multiplier = self.spin_max_object_size.value()
                    logger.info("YOLO size filtering ENABLED:")
                    logger.info(f"  Resize factor: {resize_f:.2f}")
                    logger.info(f"  Reference body size: {reference_body_size:.1f} px")
                    logger.info(
                        f"  Reference body area (original): {reference_body_area:.1f} px²"
                    )
                    logger.info(
                        f"  Reference body area (resized): {scaled_body_area:.1f} px²"
                    )
                    logger.info(
                        f"  Min size multiplier: {min_size_multiplier:.2f}× → {min_size_px2:.1f} px²"
                    )
                    logger.info(
                        f"  Max size multiplier: {max_size_multiplier:.2f}× → {max_size_px2:.1f} px²"
                    )

                # Create detector and run detection on FULL frame (no masking)
                # This preserves natural image context for better YOLO confidence
                logger.info(
                    f"Running YOLO detection (conf={yolo_params['YOLO_CONFIDENCE_THRESHOLD']:.2f}, "
                    f"iou={yolo_params['YOLO_IOU_THRESHOLD']:.2f})"
                )
                detector = YOLOOBBDetector(yolo_params)
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    yolo_results,
                    raw_confidences,
                    raw_obb_corners,
                ) = detector.detect_objects(frame_to_process, 0, return_raw=True)
                (
                    meas,
                    sizes,
                    shapes,
                    detection_confidences,
                    filtered_obb_corners,
                    _,
                ) = detector.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=roi_for_yolo,
                    detection_ids=None,
                )

                if use_size_filtering:
                    logger.info(f"YOLO detected {len(meas)} objects after filtering")
                    if len(sizes) > 0:
                        logger.info(
                            f"  Size range: {min(sizes):.1f} - {max(sizes):.1f} px²"
                        )
                        logger.info(
                            f"  Filtering range: {min_size_px2:.1f} - {max_size_px2:.1f} px²"
                        )
                        # Show first few detections
                        for i in range(min(5, len(sizes))):
                            passes_filter = min_size_px2 <= sizes[i] <= max_size_px2
                            logger.info(
                                f"  Detection {i + 1}: {sizes[i]:.1f} px² - {'PASS' if passes_filter else 'FILTERED OUT'}"
                            )

                # Collect detected dimensions for statistics (only for filtered detections)
                detected_dimensions = []
                for i, corners in enumerate(filtered_obb_corners):
                    corners = np.asarray(corners, dtype=np.float32)
                    major_axis = float(np.linalg.norm(corners[1] - corners[0]))
                    minor_axis = float(np.linalg.norm(corners[2] - corners[1]))
                    if major_axis < minor_axis:
                        major_axis, minor_axis = minor_axis, major_axis
                    detected_dimensions.append((major_axis, minor_axis))

                    corners_int = corners.astype(np.int32)
                    cv2.polylines(
                        test_frame,
                        [corners_int],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                    )

                    conf = (
                        detection_confidences[i]
                        if i < len(detection_confidences)
                        else float("nan")
                    )
                    if not np.isnan(conf):
                        cx = int(corners[:, 0].mean())
                        cy = int(corners[:, 1].mean())
                        cv2.putText(
                            test_frame,
                            f"{conf:.2f}",
                            (cx - 15, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            2,
                        )

                # Draw detection centers and orientations
                for i, m in enumerate(meas):
                    cx, cy, angle_rad = m
                    cv2.circle(test_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
                    # Draw orientation
                    ex = int(cx + 30 * math.cos(angle_rad))
                    ey = int(cy + 30 * math.sin(angle_rad))
                    cv2.line(test_frame, (int(cx), int(cy)), (ex, ey), (0, 255, 0), 2)

                # Add detection count and parameters
                cv2.putText(
                    test_frame,
                    f"Detections: {len(meas)} (IOU={yolo_params['YOLO_IOU_THRESHOLD']:.2f})",
                    (10, test_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # Update detection statistics (scale dimensions back to original resolution)
                self._update_detection_stats(detected_dimensions, resize_f)

            # Convert BGR to RGB for Qt display
            test_frame_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)

            # Store the detection test result for redisplay when zoom changes
            self.detection_test_result = (test_frame_rgb.copy(), resize_f)

            h, w, ch = test_frame_rgb.shape
            bytes_per_line = ch * w

            qimg = QImage(
                test_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            )

            # Apply zoom (zoom is applied after any resize_factor processing)
            # The zoom should be relative to the original frame size, not the resized one
            zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
            effective_scale = zoom_val * resize_f

            if effective_scale != 1.0:
                # Get original dimensions
                orig_h, orig_w = self.preview_frame_original.shape[:2]
                scaled_w = int(orig_w * effective_scale)
                scaled_h = int(orig_h * effective_scale)
                qimg = qimg.scaled(
                    scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

            pixmap = QPixmap.fromImage(qimg)
            self.video_label.setPixmap(pixmap)

            # Auto-fit to screen after detection
            self._fit_image_to_screen()

            logger.info("Detection test completed on preview frame")

        except Exception as e:
            logger.error(f"Detection test failed: {e}")
            import traceback

            traceback.print_exc()

    def _on_roi_mode_changed(self, index):
        """Handle ROI mode selection change."""
        self.roi_current_mode = "circle" if index == 0 else "polygon"
        if self.roi_selection_active:
            # If actively selecting, update instructions
            if self.roi_current_mode == "circle":
                self.roi_instructions.setText(
                    "Circle: Left-click 3+ points on boundary  •  Right-click to undo  •  ESC to cancel"
                )
            else:
                self.roi_instructions.setText(
                    "Polygon: Left-click vertices  •  Right-click to undo  •  Double-click to finish  •  ESC to cancel"
                )

    def _on_roi_zone_changed(self, index):
        """Handle ROI zone type selection change."""
        self.roi_current_zone_type = "include" if index == 0 else "exclude"

    def _handle_video_mouse_press(self, evt):
        """Handle mouse press on video - either ROI selection or pan/zoom."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        # ROI selection takes priority
        if self.roi_selection_active:
            self.record_roi_click(evt)
            return

        # Pan mode: Left button or Middle button
        from PySide6.QtCore import Qt

        if evt.button() == Qt.LeftButton or evt.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start_pos = evt.globalPosition().toPoint()
            self._scroll_start_h = self.scroll.horizontalScrollBar().value()
            self._scroll_start_v = self.scroll.verticalScrollBar().value()
            self.video_label.setCursor(Qt.ClosedHandCursor)
            evt.accept()

    def _handle_video_mouse_move(self, evt):
        """Handle mouse move - update pan if active."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        if self._is_panning and self._pan_start_pos:
            from PySide6.QtCore import Qt

            delta = evt.globalPosition().toPoint() - self._pan_start_pos
            self.scroll.horizontalScrollBar().setValue(self._scroll_start_h - delta.x())
            self.scroll.verticalScrollBar().setValue(self._scroll_start_v - delta.y())
            evt.accept()
        elif not self.roi_selection_active:
            # Show open hand cursor to indicate draggable
            from PySide6.QtCore import Qt

            self.video_label.setCursor(Qt.OpenHandCursor)

    def _handle_video_mouse_release(self, evt):
        """Handle mouse release - end pan."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        from PySide6.QtCore import Qt

        if self._is_panning:
            self._is_panning = False
            self._pan_start_pos = None
            # Return to open hand (still draggable) if not in ROI mode
            if not self.roi_selection_active:
                self.video_label.setCursor(Qt.OpenHandCursor)
            else:
                self.video_label.setCursor(Qt.ArrowCursor)
            evt.accept()

    def _handle_video_double_click(self, evt):
        """Handle double-click on video to fit to screen."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        if evt.button() == Qt.LeftButton:
            self._fit_image_to_screen()

    def _handle_video_wheel(self, evt):
        """Handle mouse wheel - zoom in/out."""
        if not self._video_interactions_enabled:
            evt.ignore()
            return
        from PySide6.QtCore import Qt

        # Block zoom during ROI selection
        if self.roi_selection_active:
            evt.ignore()
            return

        # Ctrl+Wheel for zoom
        if evt.modifiers() == Qt.ControlModifier:
            delta = evt.angleDelta().y()

            # Get current zoom
            current_zoom = self.slider_zoom.value()

            # Calculate new zoom (10% per wheel step)
            zoom_change = 10 if delta > 0 else -10
            new_zoom = max(10, min(400, current_zoom + zoom_change))

            # Update zoom slider (will trigger zoom change)
            self.slider_zoom.setValue(new_zoom)
            evt.accept()
        else:
            # Normal scroll - pass to scroll area
            evt.ignore()

    def _handle_video_event(self, evt):
        """Handle video events including pinch gestures."""
        from PySide6.QtCore import QEvent

        if evt.type() == QEvent.Gesture:
            if not self._video_interactions_enabled:
                evt.ignore()
                return False
            return self._handle_gesture_event(evt)

        # Always pass non-gesture events through so paint/layout still work
        # even when interactions are disabled (no-video placeholder state).
        return QLabel.event(self.video_label, evt)

    def _handle_gesture_event(self, evt):
        """Handle pinch-to-zoom gesture."""
        if not self._video_interactions_enabled:
            return False
        from PySide6.QtCore import Qt

        # Block gestures during ROI selection
        if self.roi_selection_active:
            return False

        gesture = evt.gesture(Qt.PinchGesture)
        if gesture:

            if gesture.state() == Qt.GestureUpdated:
                # Get scale factor
                scale_factor = gesture.scaleFactor()

                # Get current zoom
                current_zoom = self.slider_zoom.value()

                # Calculate new zoom based on pinch scale
                # Scale factor > 1 = zoom in, < 1 = zoom out
                zoom_delta = int((scale_factor - 1.0) * 50)  # Sensitivity adjustment
                new_zoom = max(10, min(400, current_zoom + zoom_delta))

                # Update zoom slider
                self.slider_zoom.setValue(new_zoom)

            return True

        return False

    def _display_roi_with_zoom(self):
        """Display the ROI base frame with mask and current zoom applied."""
        if self.roi_base_frame is None or not self.roi_shapes:
            return

        # Apply ROI mask to base frame
        qimg_masked = self._apply_roi_mask_to_image(self.roi_base_frame)

        # Apply zoom
        zoom_val = max(self.slider_zoom.value() / 100.0, 0.1)
        if zoom_val != 1.0:
            w = qimg_masked.width()
            h = qimg_masked.height()
            scaled_w = int(w * zoom_val)
            scaled_h = int(h * zoom_val)
            qimg_masked = qimg_masked.scaled(
                scaled_w, scaled_h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

        pixmap = QPixmap.fromImage(qimg_masked)
        self.video_label.setPixmap(pixmap)

    def _fit_image_to_screen(self):
        """Fit the image to the available screen space."""
        # Determine which frame to use for sizing and whether resize factor applies
        # Resize factor applies differently depending on display mode:
        # - ROI selection / preview display: full resolution, resize factor NOT yet applied
        # - Detection test / tracking preview: resize factor already applied to displayed frame

        # Check if tracking worker is running (frames are already resized)
        tracking_active = (
            self.tracking_worker is not None and self.tracking_worker.isRunning()
        )

        if tracking_active and self._tracking_frame_size is not None:
            # During tracking/preview, use the actual frame size from the worker
            # These frames are already resized, so use dimensions directly
            effective_width, effective_height = self._tracking_frame_size
        elif self.detection_test_result is not None:
            # Detection test shows resized content
            if self.preview_frame_original is not None:
                h, w = self.preview_frame_original.shape[:2]
                resize_factor = self.spin_resize.value()
                effective_width = int(w * resize_factor)
                effective_height = int(h * resize_factor)
            else:
                return
        elif self.preview_frame_original is not None:
            # Preview frame at original resolution (pre-test detection)
            h, w = self.preview_frame_original.shape[:2]
            effective_width = w
            effective_height = h
        elif self.roi_base_frame is not None:
            # ROI base frame is always full resolution
            effective_width = self.roi_base_frame.width()
            effective_height = self.roi_base_frame.height()
        else:
            return

        # Get the scroll area viewport size
        viewport_width = self.scroll.viewport().width()
        viewport_height = self.scroll.viewport().height()

        # Calculate zoom to fit the effective dimensions
        zoom_w = viewport_width / effective_width
        zoom_h = viewport_height / effective_height
        zoom_fit = min(zoom_w, zoom_h) * 0.95  # 95% to leave some margin

        # Clamp to valid range
        zoom_fit = max(0.1, min(5.0, zoom_fit))

        # Set the zoom slider
        self.slider_zoom.setValue(int(zoom_fit * 100))

        # Reset scroll position to center
        self.scroll.horizontalScrollBar().setValue(0)
        self.scroll.verticalScrollBar().setValue(0)

    def record_roi_click(self: object, evt: object) -> object:
        """record_roi_click method documentation."""
        if not self.roi_selection_active or self.roi_base_frame is None:
            return

        # Right-click to undo last point
        if evt.button() == Qt.RightButton:
            if len(self.roi_points) > 0:
                removed = self.roi_points.pop()
                logger.info(f"Undid last ROI point: ({removed[0]}, {removed[1]})")
                self.update_roi_preview()
            return

        # Left-click to add point
        if evt.button() != Qt.LeftButton:
            return

        pos = evt.position().toPoint()
        x, y = pos.x(), pos.y()

        # Double-click detection for polygon closing
        if self.roi_current_mode == "polygon" and len(self.roi_points) >= 3:
            if hasattr(self, "_last_click_pos") and hasattr(self, "_last_click_time"):
                import time

                current_time = time.time()
                last_x, last_y = self._last_click_pos
                # Check if double-click (within 0.5s and 10 pixels)
                if (
                    current_time - self._last_click_time < 0.5
                    and abs(x - last_x) < 10
                    and abs(y - last_y) < 10
                ):
                    self.finish_roi_selection()
                    return
            import time

            self._last_click_pos = (x, y)
            self._last_click_time = time.time()

        self.roi_points.append((x, y))
        self.update_roi_preview()

    def update_roi_preview(self: object) -> object:
        """update_roi_preview method documentation."""
        if self.roi_base_frame is None:
            return
        pix = QPixmap.fromImage(self.roi_base_frame).toImage().copy()
        painter = QPainter(pix)

        # Draw existing shapes first (color-coded by mode)
        for shape in self.roi_shapes:
            # Choose color based on inclusion/exclusion
            is_include = shape.get("mode", "include") == "include"
            color = Qt.cyan if is_include else Qt.red

            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.setPen(QPen(color, 2))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
            elif shape["type"] == "polygon":
                from PySide6.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(color, 2))
                painter.drawPolygon(points)

        # Draw current selection
        painter.setPen(QPen(Qt.red, 6))
        for i, (px, py) in enumerate(self.roi_points):
            painter.drawPoint(px, py)
            painter.setPen(QPen(Qt.black, 3))
            painter.drawText(px + 12, py - 12, str(i + 1))
            painter.setPen(QPen(Qt.white, 2))
            painter.drawText(px + 10, py - 10, str(i + 1))
            painter.setPen(QPen(Qt.red, 6))

        can_finish = False
        # Color for current shape preview (lighter version of final color)
        preview_color = (
            Qt.green if self.roi_current_zone_type == "include" else QColor(255, 165, 0)
        )  # Orange for exclude

        if self.roi_current_mode == "circle" and len(self.roi_points) >= 3:
            circle_fit = fit_circle_to_points(self.roi_points)
            if circle_fit:
                cx, cy, radius = circle_fit
                self.roi_fitted_circle = circle_fit
                painter.setPen(QPen(preview_color, 3))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
                painter.setPen(QPen(Qt.blue, 8))
                painter.drawPoint(int(cx), int(cy))
                zone_type = (
                    "Include" if self.roi_current_zone_type == "include" else "Exclude"
                )
                self.roi_status_label.setText(
                    f"Preview {zone_type} Circle: R={radius:.1f}px"
                )
                can_finish = True
            else:
                self.roi_status_label.setText("Invalid circle fit")
        elif self.roi_current_mode == "polygon" and len(self.roi_points) >= 3:
            # Draw preview polygon
            from PySide6.QtCore import QPoint

            points = [QPoint(int(x), int(y)) for x, y in self.roi_points]
            painter.setPen(QPen(preview_color, 3))
            painter.drawPolygon(points)
            zone_type = (
                "Include" if self.roi_current_zone_type == "include" else "Exclude"
            )
            self.roi_status_label.setText(
                f"Preview {zone_type} Polygon: {len(self.roi_points)} vertices"
            )
            can_finish = True
        else:
            min_pts = 3
            self.roi_status_label.setText(
                f"Points: {len(self.roi_points)} (Need {min_pts}+)"
            )

        self.btn_finish_roi.setEnabled(can_finish)
        painter.end()
        self.video_label.setPixmap(QPixmap.fromImage(pix))

    def start_roi_selection(self: object) -> object:
        """start_roi_selection method documentation."""
        if not self.file_line.text():
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        # Load base frame if not already loaded
        if self.roi_base_frame is None:
            cap = cv2.VideoCapture(self.file_line.text())
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Cannot open video file.")
                return
            ret, frame = cap.read()
            cap.release()
            if not ret:
                QMessageBox.warning(self, "Error", "Cannot read video frame.")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.roi_base_frame = qt_image

        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = True
        self.btn_start_roi.setEnabled(False)
        self.btn_finish_roi.setEnabled(False)
        self.combo_roi_mode.setEnabled(False)
        self.combo_roi_zone.setEnabled(False)

        # Disable zoom slider during ROI selection
        self.slider_zoom.setEnabled(False)

        # Set crosshair cursor for precise ROI selection
        self.video_label.setCursor(Qt.CrossCursor)

        zone_type = (
            "INCLUSION" if self.roi_current_zone_type == "include" else "EXCLUSION"
        )
        if self.roi_current_mode == "circle":
            self.roi_status_label.setText(
                f"Click points on {zone_type.lower()} circle boundary"
            )
            self.roi_instructions.setText(
                f"{zone_type} Circle: Left-click 3+ points on boundary  •  Right-click to undo  •  ESC to cancel"
            )
        else:
            self.roi_status_label.setText(f"Click {zone_type.lower()} polygon vertices")
            self.roi_instructions.setText(
                f"{zone_type} Polygon: Left-click vertices  •  Right-click to undo  •  Double-click to finish  •  ESC to cancel"
            )

        self.update_roi_preview()

    def finish_roi_selection(self: object) -> object:
        """finish_roi_selection method documentation."""
        if not self.roi_base_frame:
            return

        fh, fw = self.roi_base_frame.height(), self.roi_base_frame.width()

        if self.roi_current_mode == "circle":
            if not self.roi_fitted_circle:
                QMessageBox.warning(self, "No ROI", "No valid circle fit available.")
                return
            cx, cy, radius = self.roi_fitted_circle
            self.roi_shapes.append(
                {
                    "type": "circle",
                    "params": (cx, cy, radius),
                    "mode": self.roi_current_zone_type,
                }
            )
            zone_type = (
                "inclusion" if self.roi_current_zone_type == "include" else "exclusion"
            )
            logger.info(
                f"Added circle {zone_type} zone: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}"
            )

        elif self.roi_current_mode == "polygon":
            if len(self.roi_points) < 3:
                QMessageBox.warning(
                    self, "No ROI", "Need at least 3 points for polygon."
                )
                return
            self.roi_shapes.append(
                {
                    "type": "polygon",
                    "params": list(self.roi_points),
                    "mode": self.roi_current_zone_type,
                }
            )
            zone_type = (
                "inclusion" if self.roi_current_zone_type == "include" else "exclusion"
            )
            logger.info(
                f"Added polygon {zone_type} zone with {len(self.roi_points)} vertices"
            )

        # Generate combined mask from all shapes
        self._generate_combined_roi_mask(fh, fw)

        # Reset for next shape
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_selection_active = False
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        self.combo_roi_mode.setEnabled(True)
        self.combo_roi_zone.setEnabled(True)
        self.roi_instructions.setText("")

        # Re-enable zoom slider
        self.slider_zoom.setEnabled(True)

        # Restore open hand cursor (for pan/zoom)
        if hasattr(Qt, "OpenHandCursor"):
            self.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self.video_label.unsetCursor()

        # Update status to show inclusion/exclusion counts
        include_count = sum(
            1 for s in self.roi_shapes if s.get("mode", "include") == "include"
        )
        exclude_count = sum(
            1 for s in self.roi_shapes if s.get("mode", "include") == "exclude"
        )
        self.roi_status_label.setText(
            f"Active ROI: {include_count} inclusion, {exclude_count} exclusion zone(s)"
        )

        # Enable crop button and show optimization info
        self.btn_crop_video.setEnabled(True)
        self._update_roi_optimization_info()

        # Auto-fit to screen after ROI change - use QTimer to ensure proper sequencing
        if self.roi_base_frame:
            from PySide6.QtCore import QTimer

            # First fit the screen (sets zoom slider value)
            QTimer.singleShot(10, self._fit_image_to_screen)
            # Then display with the new zoom applied
            QTimer.singleShot(50, self._display_roi_with_zoom)

    def _generate_combined_roi_mask(self, height, width):
        """Generate a combined mask from all ROI shapes with inclusion/exclusion support."""
        if not self.roi_shapes:
            self.roi_mask = None
            return

        # Create blank mask
        combined_mask = np.zeros((height, width), np.uint8)

        # First pass: apply all inclusion zones (OR operation)
        for shape in self.roi_shapes:
            if shape.get("mode", "include") == "include":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 255, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 255)

        # Second pass: subtract all exclusion zones (AND NOT operation)
        for shape in self.roi_shapes:
            if shape.get("mode", "include") == "exclude":
                if shape["type"] == "circle":
                    cx, cy, radius = shape["params"]
                    cv2.circle(combined_mask, (int(cx), int(cy)), int(radius), 0, -1)
                elif shape["type"] == "polygon":
                    pts = np.array(shape["params"], dtype=np.int32)
                    cv2.fillPoly(combined_mask, [pts], 0)

        self.roi_mask = combined_mask
        logger.info(f"Generated combined ROI mask from {len(self.roi_shapes)} shape(s)")

        # Invalidate cache when ROI changes
        self._invalidate_roi_cache()

    def undo_last_roi_shape(self: object) -> object:
        """Remove the last added ROI shape."""
        if not self.roi_shapes:
            return

        removed = self.roi_shapes.pop()
        logger.info(f"Removed last ROI shape: {removed['type']}")

        if self.roi_base_frame:
            fh, fw = self.roi_base_frame.height(), self.roi_base_frame.width()
            self._generate_combined_roi_mask(fh, fw)
        else:
            self.roi_mask = None

        # Update UI
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        if self.roi_shapes:
            num_shapes = len(self.roi_shapes)
            shape_summary = ", ".join([s["type"] for s in self.roi_shapes])
            self.roi_status_label.setText(
                f"Active ROI: {num_shapes} shape(s) ({shape_summary})"
            )
            # Show the updated masked result
            if self.roi_base_frame:
                qimg_masked = self._apply_roi_mask_to_image(self.roi_base_frame)
                self.video_label.setPixmap(QPixmap.fromImage(qimg_masked))
        else:
            self.roi_status_label.setText("No ROI")
            # Show original frame without masking
            if self.roi_base_frame:
                self.video_label.setPixmap(QPixmap.fromImage(self.roi_base_frame))

        self.update_roi_preview()

    def clear_roi(self: object) -> object:
        """clear_roi method documentation."""
        self.roi_mask = None
        self.roi_points = []
        self.roi_fitted_circle = None
        self.roi_shapes = []
        self.roi_selection_active = False
        self.roi_base_frame = None
        self.btn_start_roi.setEnabled(True)
        self.btn_finish_roi.setEnabled(False)
        self.btn_undo_roi.setEnabled(False)
        self.combo_roi_mode.setEnabled(True)
        self.roi_status_label.setText("No ROI")
        self.roi_instructions.setText("")
        self.video_label.setText("ROI Cleared.")

        # Re-enable zoom slider
        self.slider_zoom.setEnabled(True)

        # Restore open hand cursor
        if hasattr(Qt, "OpenHandCursor"):
            self.video_label.setCursor(Qt.OpenHandCursor)
        else:
            self.video_label.unsetCursor()

        logger.info("All ROI shapes cleared")

    def keyPressEvent(self: object, event: object) -> object:
        """keyPressEvent method documentation."""
        if event.key() == Qt.Key_Escape and self.roi_selection_active:
            self.clear_roi()
        else:
            super().keyPressEvent(event)

    def on_detection_method_changed(self: object, index: object) -> object:
        """on_detection_method_changed method documentation."""
        # Keep compatibility hook and synchronize YOLO-only individual-analysis controls.
        self._sync_individual_analysis_mode_ui()

    def _sanitize_model_token(self, text: object) -> object:
        """Sanitize model metadata token for safe filename use."""
        return _sanitize_model_token(text)

    def _format_yolo_model_label(self, model_path: object) -> object:
        """Build combo-box label for a model path, including metadata if available."""
        rel_path = make_model_path_relative(model_path)
        filename = os.path.basename(rel_path)
        metadata = get_yolo_model_metadata(rel_path) or {}

        size = metadata.get("size") or metadata.get("model_size")
        species = metadata.get("species")
        model_info = metadata.get("model_info")
        model_id = None
        if species and model_info:
            model_id = f"{species}_{model_info}"
        elif species:
            model_id = species
        if size and model_id:
            return f"{filename} ({size}, {model_id})"

        builtins = {
            "yolo26s-obb.pt": "Balanced",
            "yolo26n-obb.pt": "Fastest",
            "yolov11s-obb.pt": "Legacy",
        }
        if filename in builtins:
            return f"{filename} ({builtins[filename]})"
        return filename

    def _refresh_yolo_model_combo(self, preferred_model_path: object = None) -> object:
        """Populate YOLO model combo from built-ins + repository models."""
        selected_path = preferred_model_path
        if selected_path is None and hasattr(self, "combo_yolo_model"):
            selected_path = self._get_selected_yolo_model_path()

        entries = {}
        for built_in in ("yolo26s-obb.pt", "yolo26n-obb.pt", "yolov11s-obb.pt"):
            entries[built_in] = self._format_yolo_model_label(built_in)

        models_dir = get_models_directory()
        try:
            local_models = sorted(
                f
                for f in os.listdir(models_dir)
                if os.path.splitext(f)[1].lower() in (".pt", ".pth")
            )
        except Exception as e:
            logger.warning(f"Failed to list YOLO model directory '{models_dir}': {e}")
            local_models = []

        for filename in local_models:
            rel_path = make_model_path_relative(os.path.join(models_dir, filename))
            entries[rel_path] = self._format_yolo_model_label(rel_path)

        self.combo_yolo_model.blockSignals(True)
        self.combo_yolo_model.clear()
        for model_path, label in entries.items():
            self.combo_yolo_model.addItem(label, model_path)
        self.combo_yolo_model.addItem("Custom Model...", "__custom__")
        self.combo_yolo_model.blockSignals(False)

        self._set_yolo_model_selection(selected_path)

    def _get_selected_yolo_model_path(self) -> object:
        """Return currently selected YOLO model path (relative when available)."""
        if not hasattr(self, "combo_yolo_model"):
            return "yolo26s-obb.pt"
        selected_data = self.combo_yolo_model.currentData(Qt.UserRole)
        if selected_data and selected_data != "__custom__":
            return str(selected_data)
        if (
            hasattr(self, "yolo_custom_model_line")
            and self.yolo_custom_model_line.text().strip()
        ):
            return make_model_path_relative(self.yolo_custom_model_line.text().strip())
        return "yolo26s-obb.pt"

    def _set_yolo_model_selection(self, model_path: object) -> object:
        """Set combo/custom selection from a model path."""
        target_path = make_model_path_relative(model_path or "")
        if not target_path:
            target_path = "yolo26s-obb.pt"

        for i in range(self.combo_yolo_model.count()):
            item_data = self.combo_yolo_model.itemData(i, Qt.UserRole)
            if item_data == target_path:
                self.combo_yolo_model.setCurrentIndex(i)
                if hasattr(self, "yolo_custom_model_line"):
                    self.yolo_custom_model_line.setText("")
                return

        custom_idx = self.combo_yolo_model.findData("__custom__", Qt.UserRole)
        if custom_idx < 0:
            custom_idx = self.combo_yolo_model.count() - 1
        self.combo_yolo_model.setCurrentIndex(custom_idx)
        if hasattr(self, "yolo_custom_model_line"):
            self.yolo_custom_model_line.setText(target_path)

    def _import_yolo_model_to_repository(self, source_path: object) -> object:
        """Import a model file into models/YOLO-obb using metadata-based naming."""
        src = str(source_path or "")
        if not src or not os.path.exists(src):
            return None

        # Single dialog for metadata collection (size + species + model info + timestamp preview)
        now_preview = datetime.now()
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Metadata")
        dlg_layout = QVBoxLayout(dlg)
        dlg_form = QFormLayout()

        size_combo = QComboBox(dlg)
        size_combo.addItems(["26n", "26s", "26m", "26l", "26x", "custom", "unknown"])
        size_combo.setCurrentText("26s")
        dlg_form.addRow("YOLO model size:", size_combo)

        stem_tokens = [t for t in Path(src).stem.replace("-", "_").split("_") if t]
        default_species = (
            self._sanitize_model_token(stem_tokens[0]) if stem_tokens else "species"
        )
        default_info = (
            self._sanitize_model_token("_".join(stem_tokens[1:]))
            if len(stem_tokens) > 1
            else "model"
        )

        species_line = QLineEdit(default_species, dlg)
        species_line.setPlaceholderText("species")
        dlg_form.addRow("Model species:", species_line)

        info_line = QLineEdit(default_info, dlg)
        info_line.setPlaceholderText("model-info")
        dlg_form.addRow("Model info:", info_line)

        ts_label = QLabel(now_preview.isoformat(timespec="seconds"), dlg)
        ts_label.setToolTip("Timestamp applied when model is added to repository")
        dlg_form.addRow("Added timestamp:", ts_label)

        dlg_layout.addLayout(dlg_form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        dlg_layout.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return None

        model_size = size_combo.currentText().strip() or "unknown"
        model_species = self._sanitize_model_token(species_line.text())
        model_info = self._sanitize_model_token(info_line.text())
        if not model_species or not model_info:
            QMessageBox.warning(
                self,
                "Invalid Metadata",
                "Species and model info must both be provided.",
            )
            return None

        now = datetime.now()
        timestamp_token = now.strftime("%Y%m%d-%H%M%S")
        added_at = now.isoformat(timespec="seconds")
        ext = os.path.splitext(src)[1].lower() or ".pt"
        models_dir = get_models_directory()

        model_slug = f"{model_species}_{model_info}"
        base_name = f"{timestamp_token}_{model_size}_{model_slug}"
        dest_path = os.path.join(models_dir, f"{base_name}{ext}")
        counter = 1
        while os.path.exists(dest_path):
            dest_path = os.path.join(models_dir, f"{base_name}_{counter}{ext}")
            counter += 1

        try:
            shutil.copy2(src, dest_path)
        except Exception as e:
            logger.error(f"Failed to copy model to repository: {e}")
            QMessageBox.warning(
                self,
                "Import Failed",
                f"Could not import model into repository:\n{e}",
            )
            return None

        rel_path = make_model_path_relative(dest_path)
        metadata = {
            "size": model_size,
            "species": model_species,
            "model_info": model_info,
            "added_at": added_at,
            "source_path": src,
            "stored_filename": os.path.basename(dest_path),
        }
        register_yolo_model(rel_path, metadata)
        logger.info(f"Imported model to repository: {dest_path}")
        return rel_path

    def on_yolo_model_changed(self: object, index: object) -> object:
        """on_yolo_model_changed method documentation."""
        is_custom = self.combo_yolo_model.currentData(Qt.UserRole) == "__custom__"
        self.yolo_custom_model_widget.setVisible(is_custom)

    def select_yolo_custom_model(self: object) -> object:
        """select_yolo_custom_model method documentation."""
        start_dir = get_models_directory()

        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO Model",
            start_dir,
            "PyTorch Model Files (*.pt *.pth);;All Files (*)",
        )
        if not fp:
            return

        models_dir = get_models_directory()
        selected_abs = os.path.abspath(fp)
        try:
            rel_path = os.path.relpath(selected_abs, models_dir)
            is_in_repo = not rel_path.startswith("..")
        except (ValueError, TypeError):
            is_in_repo = False

        if is_in_repo:
            final_model_path = make_model_path_relative(selected_abs)
        else:
            final_model_path = self._import_yolo_model_to_repository(selected_abs)
            if not final_model_path:
                return
            QMessageBox.information(
                self,
                "Model Imported",
                f"Model imported to repository as:\n{os.path.basename(final_model_path)}",
            )

        self._refresh_yolo_model_combo(preferred_model_path=final_model_path)
        self._set_yolo_model_selection(final_model_path)

    def toggle_histogram_window(self: object) -> object:
        """toggle_histogram_window method documentation."""
        if self.histogram_window is None:
            if self.histogram_panel is None:
                self.histogram_panel = HistogramPanel(
                    history_frames=self.spin_histogram_history.value()
                )
            self.histogram_window = QMainWindow()
            self.histogram_window.setWindowTitle("Real-Time Parameter Histograms")
            self.histogram_window.setCentralWidget(self.histogram_panel)
            self.histogram_window.resize(900, 700)
            self.histogram_window.setStyleSheet(self.styleSheet())

            def on_close():
                self.btn_show_histograms.setChecked(False)
                self.histogram_window.hide()

            self.histogram_window.closeEvent = lambda event: (
                on_close(),
                event.accept(),
            )

        if self.btn_show_histograms.isChecked():
            self.histogram_window.show()
            self.histogram_window.raise_()
            self.histogram_window.activateWindow()
        else:
            self.histogram_window.hide()

    def toggle_preview(self: object, checked: object) -> object:
        """toggle_preview method documentation."""
        if checked:
            # Warn user that preview doesn't save config
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Preview Mode")
            msg.setText(
                "Preview mode will run forward tracking only without saving configuration."
            )
            msg.setInformativeText(
                "Preview features:\n"
                "• Forward pass only (no backward tracking)\n"
                "• Configuration is NOT saved\n"
                "• No CSV output\n\n"
                "Use 'Run Full Tracking' to save results and config."
            )
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msg.setDefaultButton(QMessageBox.Ok)

            if msg.exec() == QMessageBox.Ok:
                self.start_tracking(preview_mode=True)
                self.btn_preview.setText("Stop Preview")
                self.btn_start.setEnabled(False)
            else:
                self.btn_preview.setChecked(False)
        else:
            self.stop_tracking()
            self.btn_preview.setText("Preview Mode")
            self.btn_start.setEnabled(True)

    def toggle_tracking(self: object, checked: object) -> object:
        """toggle_tracking method documentation."""
        if checked:
            # If preview is active, stop it first
            if self.btn_preview.isChecked():
                self.btn_preview.setChecked(False)
                self.btn_preview.setText("Preview Mode")
                self.stop_tracking()

            self.btn_start.setText("Stop Tracking")
            self.btn_preview.setEnabled(False)
            self.start_full()

            if not (self.tracking_worker and self.tracking_worker.isRunning()):
                # Tracking did not start (e.g., no video or config save cancelled)
                self.btn_start.blockSignals(True)
                self.btn_start.setChecked(False)
                self.btn_start.blockSignals(False)
                self.btn_start.setText("Start Full Tracking")
                self.btn_preview.setEnabled(True)
        else:
            self.stop_tracking()

    def toggle_debug_logging(self: object, checked: object) -> object:
        """toggle_debug_logging method documentation."""
        if checked:
            logger.setLevel(logging.DEBUG)
            logger.info("Debug logging enabled")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug logging disabled")

    def _on_visualization_mode_changed(self, state):
        """Handle visualization-free mode toggle."""
        is_viz_free = self.chk_visualization_free.isChecked()
        is_preview_active = self.btn_preview.isChecked()
        is_tracking_active = self.tracking_worker and self.tracking_worker.isRunning()

        # Keep display settings visible; only gate their effect at runtime
        self.g_display.setVisible(True)

        # Keep individual checkboxes enabled for pre-configuration
        self.chk_show_circles.setEnabled(True)
        self.chk_show_orientation.setEnabled(True)
        self.chk_show_trajectories.setEnabled(True)
        self.chk_show_labels.setEnabled(True)
        self.chk_show_state.setEnabled(True)
        self.chk_show_kalman_uncertainty.setEnabled(True)
        self.chk_show_fg.setEnabled(True)
        self.chk_show_bg.setEnabled(True)
        self.chk_show_yolo_obb.setEnabled(True)

        # Only affect display during active tracking (not setup/preview)
        if is_tracking_active and is_viz_free and not is_preview_active:
            # Store current preview if any, then show placeholder
            self._stored_preview_text = (
                self.video_label.text() if not self.video_label.pixmap() else None
            )
            self.video_label.clear()
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #888; font-size: 14px;")
            logger.info("Visualization-Free Mode enabled - Maximum speed processing")
        elif is_tracking_active and not is_viz_free:
            # Restore previous state or default message
            if hasattr(self, "_stored_preview_text") and self._stored_preview_text:
                self.video_label.setText(self._stored_preview_text)
            elif not self.video_label.pixmap():
                self._show_video_logo_placeholder()
            self.video_label.setStyleSheet("color: #666; font-size: 16px;")

    def start_full(self: object) -> object:
        """start_full method documentation."""
        if self.btn_preview.isChecked():
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            self.stop_tracking()

        # Set up comprehensive session logging once for entire tracking session
        video_path = self.file_line.text()
        if video_path:
            self._setup_session_logging(video_path, backward_mode=False)
            from datetime import datetime

            self._individual_dataset_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_detection_cache_path = None
            self.current_individual_properties_cache_path = None
            self.current_interpolated_pose_csv_path = None
            self.current_interpolated_pose_df = None
            self._pending_pose_export_csv_path = None
            self._pending_video_csv_path = None
            self._pending_video_generation = False

        self.start_tracking(preview_mode=False)

    def _request_qthread_stop(
        self,
        worker,
        worker_name: str,
        *,
        timeout_ms: int = 1500,
        force_terminate: bool = True,
    ) -> None:
        """Stop a QThread cooperatively, then force terminate if needed."""
        if worker is None:
            return
        try:
            if not worker.isRunning():
                return
        except Exception:
            return

        try:
            if hasattr(worker, "stop"):
                worker.stop()
        except Exception:
            logger.debug("Failed to call stop() on %s", worker_name, exc_info=True)

        try:
            worker.requestInterruption()
        except Exception:
            pass

        stopped = False
        try:
            stopped = bool(worker.wait(int(timeout_ms)))
        except Exception:
            stopped = False

        if stopped:
            logger.info("%s stopped.", worker_name)
            return

        if not force_terminate:
            logger.warning(
                "%s did not stop within %d ms (cooperative stop only).",
                worker_name,
                int(timeout_ms),
            )
            return

        logger.warning(
            "%s did not stop cooperatively; forcing terminate().", worker_name
        )
        try:
            worker.terminate()
        except Exception:
            logger.debug("terminate() failed for %s", worker_name, exc_info=True)
        try:
            worker.wait(max(500, int(timeout_ms)))
        except Exception:
            pass

    def _stop_csv_writer(self, timeout_sec: float = 2.0) -> None:
        """Stop background CSV writer thread safely without indefinite blocking."""
        writer = self.csv_writer_thread
        if writer is None:
            return
        try:
            writer.stop()
        except Exception:
            logger.debug("Failed to request CSV writer stop.", exc_info=True)
        try:
            if writer.is_alive():
                writer.join(timeout=timeout_sec)
                if writer.is_alive():
                    logger.warning("CSV writer did not stop within %.1fs.", timeout_sec)
        except Exception:
            logger.debug("Failed to join CSV writer thread.", exc_info=True)
        finally:
            self.csv_writer_thread = None

    def _cleanup_thread_reference(self, attr_name: str) -> None:
        """Delete finished QThread references safely."""
        worker = getattr(self, attr_name, None)
        if worker is None:
            return
        try:
            running = bool(worker.isRunning())
        except Exception:
            running = False
        if not running:
            try:
                worker.deleteLater()
            except Exception:
                pass
            setattr(self, attr_name, None)

    def stop_tracking(self: object) -> object:
        """stop_tracking method documentation."""
        self._stop_all_requested = True
        self._pending_finish_after_interp = False
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False

        # Stop all active workers and subprocess-like threads.
        self._request_qthread_stop(
            getattr(self, "merge_worker", None), "MergeWorker", timeout_ms=1200
        )
        self._request_qthread_stop(self.dataset_worker, "DatasetGenerationWorker")
        self._request_qthread_stop(self.interp_worker, "InterpolatedCropsWorker")
        self._request_qthread_stop(self.tracking_worker, "TrackingWorker")
        self._stop_csv_writer()

        self._cleanup_thread_reference("merge_worker")
        self._cleanup_thread_reference("dataset_worker")
        self._cleanup_thread_reference("interp_worker")

        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        self._set_ui_controls_enabled(True)
        # Ensure UI state is restored after stopping
        if self.current_video_path:
            self._apply_ui_state("idle")
        else:
            self._apply_ui_state("no_video")
        self.btn_preview.setChecked(False)
        self.btn_preview.setText("Preview Mode")
        self.btn_start.blockSignals(True)
        self.btn_start.setChecked(False)
        self.btn_start.blockSignals(False)
        self.btn_start.setText("Start Full Tracking")
        self.btn_start.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self._individual_dataset_run_id = None
        self.current_detection_cache_path = None
        self.current_individual_properties_cache_path = None
        self.current_interpolated_pose_csv_path = None
        self.current_interpolated_pose_df = None

        # Hide stats labels when tracking stops
        self.label_current_fps.setVisible(False)
        self.label_elapsed_time.setVisible(False)
        self.label_eta.setVisible(False)

        # Reset tracking frame size
        self._tracking_frame_size = None
        self._cleanup_session_logging()

    def _set_ui_controls_enabled(self, enabled: bool):
        if enabled:
            if self.current_video_path:
                self._apply_ui_state("idle")
            else:
                self._apply_ui_state("no_video")
            return

        # Disabled state - choose mode based on tracking/preview status
        if self.tracking_worker and self.tracking_worker.isRunning():
            if self.btn_preview.isChecked():
                self._apply_ui_state("preview")
            else:
                self._apply_ui_state("tracking")
        else:
            self._apply_ui_state("locked")

    def _collect_preview_controls(self):
        return [
            self.btn_test_detection,
            self.slider_timeline,
            self.btn_first_frame,
            self.btn_prev_frame,
            self.btn_play_pause,
            self.btn_next_frame,
            self.btn_last_frame,
            self.btn_random_seek,
            self.combo_playback_speed,
            self.spin_start_frame,
            self.spin_end_frame,
            self.btn_set_start_current,
            self.btn_set_end_current,
            self.btn_reset_range,
        ]

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        allow = set(allowlist or [])
        block = set(blocklist or [])
        interactive_types = (
            QAbstractButton,
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QSlider,
        )
        widgets = []
        for widget_type in interactive_types:
            widgets.extend(self.findChildren(widget_type))

        if enabled and remember_state and self._saved_widget_enabled_states:
            for widget in widgets:
                if widget in block:
                    widget.setEnabled(False)
                elif widget in allow:
                    widget.setEnabled(True)
                elif widget in self._saved_widget_enabled_states:
                    widget.setEnabled(self._saved_widget_enabled_states[widget])
            self._saved_widget_enabled_states = {}
            return

        if not enabled and remember_state:
            for widget in widgets:
                if widget in block or widget in allow:
                    continue
                self._saved_widget_enabled_states[widget] = widget.isEnabled()

        for widget in widgets:
            if widget in block:
                widget.setEnabled(False)
            elif widget in allow:
                widget.setEnabled(True)
            else:
                widget.setEnabled(enabled)

    def _set_video_interaction_enabled(self, enabled: bool):
        self._video_interactions_enabled = enabled
        self.slider_zoom.setEnabled(enabled)
        # Keep the viewport enabled so placeholder/logo rendering is not dimmed
        # by disabled-widget styling (notably on macOS).
        self.scroll.setEnabled(True)
        if not enabled:
            self.video_label.unsetCursor()

    def _prepare_tracking_display(self):
        """Clear any stale frame before tracking starts."""
        self.video_label.clear()
        if self._is_visualization_enabled():
            self.video_label.setText("")
            self.video_label.setStyleSheet("color: #666; font-size: 16px;")
        else:
            self.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self.video_label.setStyleSheet("color: #888; font-size: 14px;")

    def _show_video_logo_placeholder(self):
        """Show MAT logo in the video panel when no video is loaded."""
        try:
            project_root = Path(__file__).resolve().parents[3]
            logo_path = project_root / "brand" / "multianimaltracker.svg"
            vw = max(640, self.scroll.viewport().width())
            vh = max(420, self.scroll.viewport().height())
            canvas = QPixmap(vw, vh)
            canvas.fill(QColor(0, 0, 0, 0))

            renderer = QSvgRenderer(str(logo_path))
            if renderer.isValid():
                view_box = renderer.viewBoxF()
                if view_box.isEmpty():
                    default_size = renderer.defaultSize()
                    view_box = QRectF(
                        0,
                        0,
                        max(1, default_size.width()),
                        max(1, default_size.height()),
                    )

                # Preserve source aspect ratio and size it prominently.
                max_w = max(1, int(vw * 0.9))
                max_h = max(1, int(vh * 0.8))
                scale = min(max_w / view_box.width(), max_h / view_box.height())
                logo_w = max(1, int(view_box.width() * scale))
                logo_h = max(1, int(view_box.height() * scale))
                x = (vw - logo_w) // 2
                y = (vh - logo_h) // 2

                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(x, y, logo_w, logo_h))
                painter.end()
                self.video_label.setPixmap(canvas)
                self.video_label.setText("")
                return
        except Exception:
            pass
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Multi-Animal-Tracker\n\nLoad a video to begin...")

    def _is_visualization_enabled(self) -> bool:
        # Preview should always render frames regardless of visualization-free toggle
        return (
            not self.chk_visualization_free.isChecked() or self.btn_preview.isChecked()
        )

    def _sync_contextual_controls(self):
        # ROI
        self.btn_finish_roi.setEnabled(self.roi_selection_active)
        self.btn_undo_roi.setEnabled(len(self.roi_shapes) > 0)
        self.btn_clear_roi.setEnabled(
            len(self.roi_shapes) > 0 or self.roi_selection_active
        )

        # Crop video only if ROI exists and video loaded
        if hasattr(self, "btn_crop_video"):
            self.btn_crop_video.setEnabled(
                bool(self.roi_shapes) and bool(self.current_video_path)
            )

    def _apply_ui_state(self, state: str):
        self._ui_state = state

        if state == "no_video":
            extra_allowed = [
                self.combo_xanylabeling_env,
                self.btn_refresh_envs,
                self.btn_open_xanylabeling,
                self.btn_open_training_dialog,
                self.btn_open_pose_label,
            ]
            self._set_interactive_widgets_enabled(
                False,
                allowlist=[self.btn_file, self.btn_load_config] + extra_allowed,
                remember_state=False,
            )
            self.btn_start.setEnabled(False)
            self._set_video_interaction_enabled(False)
            self.g_video_player.setVisible(False)
            self._show_video_logo_placeholder()
            return

        if state == "idle":
            self._set_interactive_widgets_enabled(True)
            self.btn_start.setEnabled(True)
            self._set_video_interaction_enabled(True)
            self._sync_contextual_controls()
            return

        if state == "tracking":
            allow = [self.btn_start]
            if self._is_visualization_enabled():
                allow.append(self.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self.btn_start.setEnabled(True)
            self._set_video_interaction_enabled(self._is_visualization_enabled())
            return

        if state == "preview":
            allow = [self.btn_preview] + list(self._preview_controls)
            if self._is_visualization_enabled():
                allow.append(self.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self.btn_preview.setEnabled(True)
            self._set_video_interaction_enabled(self._is_visualization_enabled())
            return

        # Locked (non-tracking) state: disable all interactive widgets
        if state == "locked":
            self._set_interactive_widgets_enabled(False)
            self._set_video_interaction_enabled(False)
            return

    def _draw_roi_overlay(self, qimage):
        """Draw ROI shapes overlay on a QImage."""
        if not self.roi_shapes:
            return qimage

        # Create a copy to draw on
        pix = QPixmap.fromImage(qimage).copy()
        painter = QPainter(pix)

        # Draw all ROI shapes
        for shape in self.roi_shapes:
            if shape["type"] == "circle":
                cx, cy, radius = shape["params"]
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawEllipse(
                    int(cx - radius), int(cy - radius), int(2 * radius), int(2 * radius)
                )
                # Draw center point
                painter.setPen(QPen(Qt.cyan, 6))
                painter.drawPoint(int(cx), int(cy))
            elif shape["type"] == "polygon":
                from PySide6.QtCore import QPoint

                points = [QPoint(int(x), int(y)) for x, y in shape["params"]]
                painter.setPen(QPen(Qt.cyan, 2, Qt.DashLine))
                painter.drawPolygon(points)

        painter.end()
        return pix.toImage()

    def _apply_roi_mask_to_image(self, qimage):
        """Apply ROI visualization - draw boundary overlay for all detection methods.

        Both YOLO and Background Subtraction now show the same cyan boundary overlay
        for consistent UI experience. The actual masking happens in the tracking pipeline.
        """
        if self.roi_mask is None or not self.roi_shapes:
            return qimage

        # Use boundary overlay for all detection methods
        return self._draw_roi_overlay(qimage)

    def _apply_roi_mask_darkening(self, qimage):
        """Apply ROI mask to darken areas outside the ROI (with caching).
        Used for background subtraction where the image is actually masked.
        """
        if self.roi_mask is None or not self.roi_shapes:
            return qimage

        # Generate cache key from image pointer and ROI hash
        frame_id = id(qimage)
        roi_hash = self._get_roi_hash()
        cache_key = (frame_id, roi_hash)

        # Return cached result if available
        if cache_key in self._roi_masked_cache:
            return self._roi_masked_cache[cache_key]

        # Convert QImage to numpy array
        width = qimage.width()
        height = qimage.height()

        # Ensure image is in RGB888 format
        if qimage.format() != QImage.Format_RGB888:
            qimage = qimage.convertToFormat(QImage.Format_RGB888)

        # Convert to numpy array using buffer protocol
        ptr = qimage.bits()
        if hasattr(ptr, "setsize"):
            # Older PySide versions (sip.voidptr)
            ptr.setsize(height * width * 3)
            arr = np.array(ptr).reshape(height, width, 3)
        else:
            # PySide6 and newer versions (memoryview)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 3)

        # Create a copy to modify
        arr_copy = arr.copy()

        # Resize ROI mask to match image dimensions if needed
        if self.roi_mask.shape != (height, width):
            roi_resized = cv2.resize(
                self.roi_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
        else:
            roi_resized = self.roi_mask

        # Darken areas outside ROI (multiply by 0.3 for 70% darkening)
        mask_inv = roi_resized == 0
        arr_copy[mask_inv] = (arr_copy[mask_inv] * 0.3).astype(np.uint8)

        # Create new QImage from modified array
        result = QImage(arr_copy.data, width, height, width * 3, QImage.Format_RGB888)
        # Make a copy to ensure data persistence
        result_copy = result.copy()

        # Cache the result (limit cache size to prevent memory bloat)
        if len(self._roi_masked_cache) > 50:
            # Remove oldest entries
            self._roi_masked_cache.clear()
        self._roi_masked_cache[cache_key] = result_copy

        return result_copy

    @Slot(int, str)
    def on_progress_update(
        self: object, percentage: object, status_text: object
    ) -> object:
        """on_progress_update method documentation."""
        if self._stop_all_requested:
            return
        self.progress_bar.setValue(percentage)
        self.progress_label.setText(status_text)

    @Slot(str)
    def on_pose_exported_model_resolved(self, artifact_path: str) -> None:
        """Update pose exported-model UI/config when runtime resolves an artifact path."""
        if self._stop_all_requested:
            return
        path = str(artifact_path or "").strip()
        if not path:
            return
        logger.info("Pose runtime resolved exported model artifact: %s", path)
        try:
            # Persist run metadata immediately.
            self.save_config(prompt_if_exists=False)
        except Exception:
            logger.debug(
                "Failed to persist resolved pose runtime artifact metadata.",
                exc_info=True,
            )

    @Slot(str, str)
    def on_tracking_warning(self: object, title: object, message: object) -> object:
        """Display tracking warnings in the UI."""
        if self._stop_all_requested:
            return
        QMessageBox.information(self, title, message)

    def show_gpu_info(self: object) -> object:
        """Display GPU and acceleration information dialog."""
        from ..utils.gpu_utils import get_device_info

        info = get_device_info()

        # Build formatted message
        lines = ["<b>GPU & Acceleration Status</b><br>"]

        # CUDA
        cuda_status = "✓ Available" if info["cuda_available"] else "✗ Not Available"
        lines.append(f"<br><b>NVIDIA CUDA:</b> {cuda_status}")
        if info["cuda_available"] and info.get("cuda_device_count", 0) > 0:
            lines.append(f"&nbsp;&nbsp;• Devices: {info['cuda_device_count']}")
            if "cupy_version" in info:
                lines.append(f"&nbsp;&nbsp;• CuPy: {info['cupy_version']}")

        # TensorRT
        tensorrt_status = (
            "✓ Available"
            if info.get("tensorrt_available", False)
            else "✗ Not Available"
        )
        lines.append(f"<br><b>NVIDIA TensorRT:</b> {tensorrt_status}")
        if info.get("tensorrt_available", False):
            lines.append("&nbsp;&nbsp;• 2-5× faster YOLO inference")

        # MPS (Apple Silicon)
        mps_status = "✓ Available" if info["mps_available"] else "✗ Not Available"
        lines.append(f"<br><b>Apple MPS:</b> {mps_status}")
        if info.get("torch_available", False) and "torch_version" in info:
            lines.append(f"&nbsp;&nbsp;• PyTorch: {info['torch_version']}")

        # CPU Acceleration
        numba_status = "✓ Available" if info["numba_available"] else "✗ Not Available"
        lines.append(f"<br><b>CPU JIT (Numba):</b> {numba_status}")
        if info["numba_available"] and "numba_version" in info:
            lines.append(f"&nbsp;&nbsp;• Version: {info['numba_version']}")

        # Overall status
        lines.append("<br><b>Overall Status:</b>")
        if info["cuda_available"]:
            lines.append("&nbsp;&nbsp;• Using NVIDIA GPU acceleration")
        elif info["mps_available"]:
            lines.append("&nbsp;&nbsp;• Using Apple Silicon GPU acceleration")
        elif info["numba_available"]:
            lines.append("&nbsp;&nbsp;• Using CPU JIT compilation")
        else:
            lines.append("&nbsp;&nbsp;• Using NumPy (no acceleration)")

        message = "<br>".join(lines)

        # Create message box with rich text
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("GPU & Acceleration Info")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec()

    @Slot(dict)
    def on_stats_update(self: object, stats: object) -> object:
        """Update real-time tracking statistics."""
        if self._stop_all_requested:
            return
        phase = str(stats.get("phase", "tracking"))
        is_precompute = phase == "individual_precompute"

        # Update FPS
        if "fps" in stats:
            if is_precompute:
                self.label_current_fps.setText(f"Precompute Rate: {stats['fps']:.1f}/s")
            else:
                self.label_current_fps.setText(f"FPS: {stats['fps']:.1f}")
            self.label_current_fps.setVisible(True)

        # Update elapsed time
        if "elapsed" in stats:
            elapsed_sec = stats["elapsed"]
            hours = int(elapsed_sec // 3600)
            minutes = int((elapsed_sec % 3600) // 60)
            seconds = int(elapsed_sec % 60)
            if hours > 0:
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                elapsed_str = f"{minutes:02d}:{seconds:02d}"
            if is_precompute:
                self.label_elapsed_time.setText(f"Precompute Elapsed: {elapsed_str}")
            else:
                self.label_elapsed_time.setText(f"Elapsed: {elapsed_str}")
            self.label_elapsed_time.setVisible(True)

        # Update ETA
        if "eta" in stats:
            eta_sec = stats["eta"]
            if eta_sec > 0:
                hours = int(eta_sec // 3600)
                minutes = int((eta_sec % 3600) // 60)
                seconds = int(eta_sec % 60)
                if hours > 0:
                    eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    eta_str = f"{minutes:02d}:{seconds:02d}"
                if is_precompute:
                    self.label_eta.setText(f"Precompute ETA: {eta_str}")
                else:
                    self.label_eta.setText(f"ETA: {eta_str}")
            else:
                if is_precompute:
                    self.label_eta.setText("Precompute ETA: calculating...")
                else:
                    self.label_eta.setText("ETA: calculating...")
            self.label_eta.setVisible(True)

    @Slot(np.ndarray)
    def on_new_frame(self: object, rgb: object) -> object:
        """on_new_frame method documentation."""
        z = max(self.slider_zoom.value() / 100.0, 0.1)
        h, w, _ = rgb.shape

        # Store tracking frame size for fit-to-screen calculation
        self._tracking_frame_size = (w, h)

        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)

        # ROI masking is now done in tracking worker - no need to duplicate here
        scaled = qimg.scaled(
            int(w * z), int(h * z), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled))

        # Auto-fit to screen on first frame of tracking
        if self._tracking_first_frame:
            self._tracking_first_frame = False
            # Use QTimer to ensure frame is displayed first
            from PySide6.QtCore import QTimer

            QTimer.singleShot(50, self._fit_image_to_screen)

    def _scale_trajectories_to_original_space(self, trajectories_df, resize_factor):
        """Scale trajectory coordinates from resized space back to original video space."""
        if trajectories_df is None or trajectories_df.empty:
            return trajectories_df

        if resize_factor == 1.0:
            return trajectories_df  # No scaling needed

        # Scale factor to go from resized -> original is 1/resize_factor
        scale_factor = 1.0 / resize_factor

        logger.info(
            f"Scaling trajectories to original video space (resize_factor={resize_factor:.3f}, scale_factor={scale_factor:.3f})"
        )

        result_df = trajectories_df.copy()

        # Scale X, Y coordinates
        result_df["X"] = result_df["X"] * scale_factor
        result_df["Y"] = result_df["Y"] * scale_factor

        # Theta doesn't need scaling (it's an angle)
        # FrameID doesn't need scaling

        logger.info(
            f"Scaled {len(result_df)} trajectory points to original video coordinates"
        )
        return result_df

    def save_trajectories_to_csv(
        self: object, trajectories: object, output_path: object
    ) -> object:
        """Save processed trajectories to CSV.

        Args:
            trajectories: Either list of tuples (old format) or pandas DataFrame (new format with confidence)
            output_path: Path to save CSV file
        """
        if trajectories is None:
            logger.warning("No post-processed trajectories to save (None).")
            return False

        # Check if input is a DataFrame (new format with confidence)
        if isinstance(trajectories, pd.DataFrame):
            if trajectories.empty:
                logger.warning(
                    "No post-processed trajectories to save (empty DataFrame)."
                )
                return False
            try:
                # DataFrame already has all columns including confidence metrics
                # Convert X and Y to integers where possible (non-NaN values)
                df_to_save = trajectories.copy()
                for col in ["X", "Y", "FrameID"]:
                    if col in df_to_save.columns:
                        # Convert to float first to handle any issues, then to Int64 (nullable integer)
                        df_to_save[col] = pd.to_numeric(
                            df_to_save[col], errors="coerce"
                        )
                        # Use Int64 dtype which supports NaN values
                        df_to_save[col] = df_to_save[col].round().astype("Int64")

                # Drop unwanted columns from raw tracking data
                unwanted_cols = ["TrackID", "Index"]
                df_to_save = df_to_save.drop(
                    columns=[col for col in unwanted_cols if col in df_to_save.columns],
                    errors="ignore",
                )

                # Reorder columns to put basic trajectory info first
                base_cols = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
                other_cols = [col for col in df_to_save.columns if col not in base_cols]
                ordered_cols = base_cols + other_cols
                df_to_save[ordered_cols].to_csv(output_path, index=False)
                logger.info(
                    f"Successfully saved {df_to_save['TrajectoryID'].nunique()} post-processed trajectories "
                    f"({len(df_to_save)} rows) with {len(ordered_cols)} columns to {output_path}"
                )
                return True
            except Exception as e:
                logger.error(
                    f"Failed to save processed trajectories to {output_path}: {e}"
                )
                return False

        # Old format (list of tuples) - for backward compatibility
        if not trajectories:
            logger.warning("No post-processed trajectories to save.")
            return False
        header = ["TrajectoryID", "X", "Y", "Theta", "FrameID"]
        try:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for trajectory_id, segment in enumerate(trajectories):
                    for x, y, theta, frame_id in segment:
                        # Handle NaN values - write as empty string or keep as float
                        x_val = int(x) if not np.isnan(x) else ""
                        y_val = int(y) if not np.isnan(y) else ""
                        frame_val = int(frame_id) if not np.isnan(frame_id) else ""
                        writer.writerow([trajectory_id, x_val, y_val, theta, frame_val])
            logger.info(
                f"Successfully saved {len(trajectories)} post-processed trajectories to {output_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save processed trajectories to {output_path}: {e}")
            return False

    def merge_and_save_trajectories(self: object) -> object:
        """merge_and_save_trajectories method documentation."""
        if self._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting trajectory merging process...")
        logger.info("=" * 80)

        forward_trajs = getattr(self, "forward_processed_trajs", None)
        backward_trajs = getattr(self, "backward_processed_trajs", None)

        # Check if trajectories exist and are not empty (handle both DataFrame and list)
        forward_empty = (
            forward_trajs is None
            or (isinstance(forward_trajs, pd.DataFrame) and forward_trajs.empty)
            or (isinstance(forward_trajs, list) and len(forward_trajs) == 0)
        )
        backward_empty = (
            backward_trajs is None
            or (isinstance(backward_trajs, pd.DataFrame) and backward_trajs.empty)
            or (isinstance(backward_trajs, list) and len(backward_trajs) == 0)
        )

        if forward_empty or backward_empty:
            QMessageBox.warning(
                self,
                "No Trajectories",
                "No forward or backward trajectories available to merge.",
            )
            return

        video_fp = self.file_line.text()
        if not video_fp:
            return
        cap = cv2.VideoCapture(video_fp)
        if not cap.isOpened():
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        current_params = self.get_parameters_dict()
        resize_factor = self.spin_resize.value()
        interp_method = self.combo_interpolation_method.currentText().lower()
        max_gap = self.spin_interpolation_max_gap.value()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Merging trajectories...")

        # Create and start merge worker thread
        self.merge_worker = MergeWorker(
            forward_trajs,
            backward_trajs,
            total_frames,
            current_params,
            resize_factor,
            interp_method,
            max_gap,
        )
        self.merge_worker.progress_signal.connect(self.on_merge_progress)
        self.merge_worker.finished_signal.connect(self.on_merge_finished)
        self.merge_worker.error_signal.connect(self.on_merge_error)
        self.merge_worker.start()

    def on_merge_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during merge."""
        if self._stop_all_requested:
            return
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_interpolated_crops_finished(self, result):
        sender = self.sender()
        if (
            sender is not None
            and self.interp_worker is not None
            and sender is not self.interp_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("interp_worker")
            self._refresh_progress_visibility()
            return
        saved = 0
        gaps = 0
        mapping_path = None
        roi_csv_path = None
        roi_npz_path = None
        pose_csv_path = None
        pose_rows = None
        try:
            saved = int(result.get("saved", 0))
            gaps = int(result.get("gaps", 0))
            mapping_path = result.get("mapping_path")
            roi_csv_path = result.get("roi_csv_path")
            roi_npz_path = result.get("roi_npz_path")
            pose_csv_path = result.get("pose_csv_path")
            pose_rows = result.get("pose_rows")
        except Exception:
            pass
        self._refresh_progress_visibility()
        logger.info(f"Interpolated individual crops saved: {saved} (gaps: {gaps})")
        if mapping_path:
            logger.info(f"Interpolated mapping saved: {mapping_path}")
        if roi_csv_path:
            logger.info(f"Interpolated ROIs CSV saved: {roi_csv_path}")
        if roi_npz_path:
            logger.info(f"Interpolated ROIs cache saved: {roi_npz_path}")
        if pose_csv_path:
            self.current_interpolated_pose_csv_path = pose_csv_path
            self.current_interpolated_pose_df = None
            logger.info(f"Interpolated pose CSV saved: {pose_csv_path}")
        elif pose_rows:
            try:
                self.current_interpolated_pose_df = pd.DataFrame(pose_rows)
                self.current_interpolated_pose_csv_path = None
                logger.info(
                    "Interpolated pose rows kept in-memory: %d",
                    len(self.current_interpolated_pose_df),
                )
            except Exception:
                self.current_interpolated_pose_df = None

        self._cleanup_thread_reference("interp_worker")
        self._refresh_progress_visibility()

        if self._pending_pose_export_csv_path:
            self._export_pose_augmented_csv(self._pending_pose_export_csv_path)

        if self._pending_finish_after_interp:
            self._pending_finish_after_interp = False
            self._run_pending_video_generation_or_finalize()

    def on_merge_error(self: object, error_message: object) -> object:
        """Handle merge errors."""
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._stop_all_requested:
            self._refresh_progress_visibility()
            return
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.critical(
            self, "Merge Error", f"Error during trajectory merging:\n{error_message}"
        )
        logger.error(f"Trajectory merge error: {error_message}")

    def on_merge_finished(self: object, resolved_trajectories: object) -> object:
        """Handle completion of trajectory merging."""
        sender = self.sender()
        if (
            sender is not None
            and self.merge_worker is not None
            and sender is not self.merge_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("merge_worker")
        if self._stop_all_requested:
            self._refresh_progress_visibility()
            return
        self.progress_label.setText("Saving merged trajectories...")

        raw_csv_path = self.csv_line.text()
        merged_csv_path = None
        if raw_csv_path:
            base, ext = os.path.splitext(raw_csv_path)
            merged_csv_path = f"{base}_final.csv"
            if self.save_trajectories_to_csv(resolved_trajectories, merged_csv_path):
                # Track initial tracking CSV as temporary (only if cleanup enabled)
                if (
                    self.chk_cleanup_temp_files.isChecked()
                    and raw_csv_path not in self.temporary_files
                ):
                    self.temporary_files.append(raw_csv_path)
                logger.info(f"✓ Merged trajectory data saved to: {merged_csv_path}")

        # Complete session pipeline. Video generation is deferred to the very end
        # after pose export and interpolated individual analysis complete.
        self._finish_tracking_session(final_csv_path=merged_csv_path)

    def _generate_video_from_trajectories(
        self, trajectories_df, csv_path=None, finalize_on_complete=True
    ):
        """
        Generate annotated video from post-processed trajectories.

        Args:
            trajectories_df: DataFrame with merged/interpolated trajectories
            csv_path: Path to the CSV file (optional, for logging)
            finalize_on_complete: If True, continue full finish pipeline after render.
        """
        logger.info("=" * 80)
        logger.info("Generating video from post-processed trajectories...")
        logger.info("=" * 80)

        # Ensure progress UI is visible for video generation
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Generating video...")
        QApplication.processEvents()

        video_path = self.file_line.text()
        output_path = self.video_out_line.text()

        def _complete_after_video():
            if finalize_on_complete:
                self._finish_tracking_session(final_csv_path=csv_path)
            else:
                self._finalize_tracking_session_ui()

        if not video_path or not output_path:
            logger.error("Video input or output path not specified")
            _complete_after_video()
            return

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            _complete_after_video()
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            cap.release()
            _complete_after_video()
            return

        logger.info(f"Writing video: {frame_width}x{frame_height} @ {fps} FPS")

        # Get visualization parameters
        params = self.get_parameters_dict()
        start_frame = int(params.get("START_FRAME", 0) or 0)
        end_frame = params.get("END_FRAME", None)
        if end_frame is None:
            end_frame = total_video_frames - 1 if total_video_frames > 0 else 0
        end_frame = int(end_frame)

        # Clamp to video bounds and export only the tracked subset.
        if total_video_frames > 0:
            start_frame = max(0, min(start_frame, total_video_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_video_frames - 1))
        total_frames = max(0, end_frame - start_frame + 1)
        logger.info(
            f"Exporting tracked frame range: {start_frame}-{end_frame} ({total_frames} frames)"
        )

        if total_frames <= 0:
            logger.error("Invalid frame range for video generation.")
            cap.release()
            out.release()
            _complete_after_video()
            return

        # Seek once to the tracked start frame so we don't write unrelated frames.
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        colors = params.get("TRAJECTORY_COLORS", [])
        reference_body_size = params.get("REFERENCE_BODY_SIZE", 30.0)

        # Get video visualization settings
        show_labels = self.check_show_labels.isChecked()
        show_orientation = self.check_show_orientation.isChecked()
        show_trails = self.check_show_trails.isChecked()
        trail_duration_sec = self.spin_trail_duration.value()
        trail_duration_frames = int(
            trail_duration_sec * fps
        )  # Convert seconds to frames
        marker_size = self.spin_marker_size.value()
        text_scale = self.spin_text_scale.value()
        arrow_length = self.spin_arrow_length.value()
        advanced_config = params.get("ADVANCED_CONFIG", {})

        # NOTE: Trajectories in merged CSV are already scaled to original coordinates
        # (see MergeWorker line ~173: coordinates divided by resize_factor)
        # So we use them directly without additional scaling

        # Scale drawing parameters by body size
        # reference_body_size is in original (unresized) coordinates
        # Video is at original resolution, so use body size directly
        marker_radius = int(marker_size * reference_body_size)
        arrow_len = int(arrow_length * reference_body_size)
        text_size = 0.5 * text_scale
        marker_thickness = max(2, int(0.15 * reference_body_size))
        pose_point_radius = int(
            max(
                1,
                advanced_config.get(
                    "video_pose_point_radius", max(2, marker_radius // 3)
                ),
            )
        )
        pose_point_thickness = int(
            advanced_config.get("video_pose_point_thickness", -1)
        )
        pose_line_thickness = int(
            max(1, advanced_config.get("video_pose_line_thickness", 2))
        )
        pose_color_mode = (
            str(advanced_config.get("video_pose_color_mode", "track")).strip().lower()
        )
        pose_fixed_color_raw = advanced_config.get("video_pose_color", [255, 255, 255])
        if (
            isinstance(pose_fixed_color_raw, (list, tuple))
            and len(pose_fixed_color_raw) == 3
        ):
            try:
                pose_fixed_color = tuple(
                    int(max(0, min(255, float(v)))) for v in pose_fixed_color_raw
                )
            except Exception:
                pose_fixed_color = (255, 255, 255)
        else:
            pose_fixed_color = (255, 255, 255)
        pose_min_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        pose_edges = []
        pose_column_triplets = []
        show_pose = bool(advanced_config.get("video_show_pose", True))
        pose_col_pattern = re.compile(r"^PoseKpt_(.+)_(X|Y|Conf)$")
        pose_labels_available = {}
        for col in trajectories_df.columns:
            m = pose_col_pattern.match(str(col))
            if m is None:
                continue
            label = m.group(1)
            axis = m.group(2)
            pose_labels_available.setdefault(label, set()).add(axis)
        if not pose_labels_available:
            show_pose = False
        if show_pose:
            skeleton_names = []
            skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
            if skeleton_file and os.path.exists(skeleton_file):
                try:
                    with open(skeleton_file, "r", encoding="utf-8") as f:
                        skeleton_data = json.load(f)
                    names_raw = skeleton_data.get(
                        "keypoint_names", skeleton_data.get("keypoints", [])
                    )
                    skeleton_names = [str(n) for n in names_raw]
                    raw_edges = skeleton_data.get(
                        "skeleton_edges", skeleton_data.get("edges", [])
                    )
                    for edge in raw_edges:
                        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                            try:
                                pose_edges.append((int(edge[0]), int(edge[1])))
                            except Exception:
                                continue
                except Exception:
                    pose_edges = []
            ordered_labels = build_pose_keypoint_labels(
                skeleton_names, len(skeleton_names)
            )
            # Add any extra labels present in CSV but absent from skeleton.
            extras = sorted(
                [l for l in pose_labels_available.keys() if l not in ordered_labels]
            )
            ordered_labels.extend(extras)
            for label in ordered_labels:
                axes = pose_labels_available.get(label, set())
                if {"X", "Y", "Conf"}.issubset(axes):
                    pose_column_triplets.append(
                        (
                            f"PoseKpt_{label}_X",
                            f"PoseKpt_{label}_Y",
                            f"PoseKpt_{label}_Conf",
                        )
                    )
            if not pose_column_triplets:
                show_pose = False

        # Build lookup for trajectories by frame and track
        traj_by_frame = {}
        traj_by_track = {}  # For trails
        for _, row in trajectories_df.iterrows():
            frame_num = int(row["FrameID"])
            track_id = int(row["TrajectoryID"])

            if frame_num not in traj_by_frame:
                traj_by_frame[frame_num] = []
            traj_by_frame[frame_num].append(row)

            if track_id not in traj_by_track:
                traj_by_track[track_id] = []
            traj_by_track[track_id].append(row)

        # Process only the tracked frame range.
        for rel_idx in range(total_frames):
            frame_idx = start_frame + rel_idx
            ret, frame = cap.read()
            if not ret:
                break

            # Get trajectories for this frame
            frame_trajs = traj_by_frame.get(frame_idx, [])

            # Draw trails first (underneath current positions)
            if show_trails:
                for traj in frame_trajs:
                    track_id = int(traj["TrajectoryID"])

                    # Get color for this track
                    if colors and track_id < len(colors):
                        color = colors[track_id]
                    else:
                        # Use matplotlib's category20 colormap (BGR format for OpenCV)
                        category20_colors = [
                            (127, 127, 31),
                            (188, 189, 34),
                            (140, 86, 75),
                            (255, 127, 14),
                            (214, 39, 40),
                            (255, 152, 150),
                            (197, 176, 213),
                            (148, 103, 189),
                            (196, 156, 148),
                            (227, 119, 194),
                            (199, 199, 199),
                            (140, 140, 140),
                            (23, 190, 207),
                            (158, 218, 229),
                            (57, 59, 121),
                            (82, 84, 163),
                            (107, 110, 207),
                            (156, 158, 222),
                            (99, 121, 57),
                            (140, 162, 82),
                        ]
                        color = category20_colors[track_id % len(category20_colors)]

                    # Get trail points (past N frames based on duration in seconds)
                    trail_points = []
                    if track_id in traj_by_track:
                        for past_row in traj_by_track[track_id]:
                            past_frame = int(past_row["FrameID"])
                            if (
                                frame_idx - trail_duration_frames
                                <= past_frame
                                < frame_idx
                            ):
                                px, py = past_row["X"], past_row["Y"]
                                if not pd.isna(px) and not pd.isna(py):
                                    # Coordinates already in original space from merged CSV
                                    trail_points.append((int(px), int(py), past_frame))

                    # Draw trail as fading line segments
                    if len(trail_points) > 1:
                        trail_points.sort(key=lambda p: p[2])  # Sort by frame
                        for i in range(len(trail_points) - 1):
                            pt1 = (trail_points[i][0], trail_points[i][1])
                            pt2 = (trail_points[i + 1][0], trail_points[i + 1][1])

                            # Calculate opacity based on age
                            age = frame_idx - trail_points[i][2]
                            alpha = 1.0 - (age / trail_duration_frames)
                            faded_color = tuple(int(c * alpha) for c in color)

                            cv2.line(
                                frame,
                                pt1,
                                pt2,
                                faded_color,
                                max(1, marker_thickness // 2),
                            )

            # Draw current positions
            for traj in frame_trajs:
                track_id = int(traj["TrajectoryID"])
                cx, cy = traj["X"], traj["Y"]

                # Skip if NaN
                if pd.isna(cx) or pd.isna(cy):
                    continue

                # Coordinates already in original space from merged CSV
                cx, cy = int(cx), int(cy)

                # Get color for this track
                if colors and track_id < len(colors):
                    color = colors[track_id]
                else:
                    # Use matplotlib's category20 colormap (BGR format for OpenCV)
                    category20_colors = [
                        (127, 127, 31),
                        (188, 189, 34),
                        (140, 86, 75),
                        (255, 127, 14),
                        (214, 39, 40),
                        (255, 152, 150),
                        (197, 176, 213),
                        (148, 103, 189),
                        (196, 156, 148),
                        (227, 119, 194),
                        (199, 199, 199),
                        (140, 140, 140),
                        (23, 190, 207),
                        (158, 218, 229),
                        (57, 59, 121),
                        (82, 84, 163),
                        (107, 110, 207),
                        (156, 158, 222),
                        (99, 121, 57),
                        (140, 162, 82),
                    ]
                    color = category20_colors[track_id % len(category20_colors)]

                # Draw circle at position
                cv2.circle(frame, (cx, cy), marker_radius, color, marker_thickness)

                # Draw label
                if show_labels:
                    label = f"ID{track_id}"
                    label_offset = int(marker_radius + 5)
                    cv2.putText(
                        frame,
                        label,
                        (cx + label_offset, cy - label_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        text_size,
                        color,
                        max(1, int(text_scale * 2)),
                    )

                # Draw orientation if available
                if show_orientation and "Theta" in traj and not pd.isna(traj["Theta"]):
                    heading = traj["Theta"]
                    end_x = int(cx + arrow_len * np.cos(heading))
                    end_y = int(cy + arrow_len * np.sin(heading))
                    cv2.arrowedLine(
                        frame,
                        (cx, cy),
                        (end_x, end_y),
                        color,
                        marker_thickness,
                        tipLength=0.3,
                    )

                # Draw pose keypoints/skeleton from pose-augmented CSV (global coords).
                if show_pose and pose_column_triplets:
                    kpts_arr = np.full(
                        (len(pose_column_triplets), 3), np.nan, dtype=np.float32
                    )
                    for k_idx, (x_col, y_col, c_col) in enumerate(pose_column_triplets):
                        try:
                            x_kp = float(traj.get(x_col))
                            y_kp = float(traj.get(y_col))
                            c_kp = float(traj.get(c_col))
                        except Exception:
                            continue
                        if np.isnan(x_kp) or np.isnan(y_kp) or np.isnan(c_kp):
                            continue
                        kpts_arr[k_idx, 0] = x_kp
                        kpts_arr[k_idx, 1] = y_kp
                        kpts_arr[k_idx, 2] = c_kp

                    valid_mask = ~np.isnan(kpts_arr[:, 2])
                    if np.any(valid_mask):
                        pose_color = (
                            color if pose_color_mode == "track" else pose_fixed_color
                        )
                        if pose_edges:
                            for e0, e1 in pose_edges:
                                if e0 < 0 or e1 < 0:
                                    continue
                                if e0 >= len(kpts_arr) or e1 >= len(kpts_arr):
                                    continue
                                if (
                                    np.isnan(kpts_arr[e0, 2])
                                    or np.isnan(kpts_arr[e1, 2])
                                    or np.isnan(kpts_arr[e0, 0])
                                    or np.isnan(kpts_arr[e0, 1])
                                    or np.isnan(kpts_arr[e1, 0])
                                    or np.isnan(kpts_arr[e1, 1])
                                ):
                                    continue
                                if (
                                    float(kpts_arr[e0, 2]) < pose_min_conf
                                    or float(kpts_arr[e1, 2]) < pose_min_conf
                                ):
                                    continue
                                p0 = (
                                    int(round(float(kpts_arr[e0, 0]))),
                                    int(round(float(kpts_arr[e0, 1]))),
                                )
                                p1 = (
                                    int(round(float(kpts_arr[e1, 0]))),
                                    int(round(float(kpts_arr[e1, 1]))),
                                )
                                cv2.line(
                                    frame,
                                    p0,
                                    p1,
                                    pose_color,
                                    pose_line_thickness,
                                )
                        for x_kp, y_kp, c_kp in kpts_arr:
                            if (
                                np.isnan(x_kp)
                                or np.isnan(y_kp)
                                or np.isnan(c_kp)
                                or float(c_kp) < pose_min_conf
                            ):
                                continue
                            cv2.circle(
                                frame,
                                (int(round(float(x_kp))), int(round(float(y_kp)))),
                                pose_point_radius,
                                pose_color,
                                pose_point_thickness,
                            )

            # Write frame
            out.write(frame)

            # Update progress every 30 frames
            if rel_idx % 30 == 0:
                progress = int(((rel_idx + 1) / total_frames) * 100)
                self.progress_bar.setValue(progress)
                QApplication.processEvents()

        # Cleanup
        cap.release()
        out.release()

        logger.info(f"✓ Video saved to: {output_path}")
        logger.info("=" * 80)

        _complete_after_video()

    @Slot(bool, list, list)
    def on_tracking_finished(
        self: object, finished_normally: object, fps_list: object, full_traj: object
    ) -> object:
        """on_tracking_finished method documentation."""
        sender = self.sender()
        if (
            sender is not None
            and self.tracking_worker is not None
            and sender is not self.tracking_worker
        ):
            logger.debug(
                "Ignoring stale tracking finished signal from previous worker."
            )
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        self._stop_csv_writer()

        if self._stop_all_requested:
            logger.info("Tracking stop requested; skipping post-processing pipeline.")
            self._cleanup_thread_reference("tracking_worker")
            self._refresh_progress_visibility()
            gc.collect()
            return

        # Check if this was preview mode
        was_preview_mode = self.btn_preview.isChecked()

        if was_preview_mode:
            self.btn_preview.setChecked(False)
            self.btn_preview.setText("Preview Mode")
            # Hide stats labels and re-enable UI for preview mode
            self.label_current_fps.setVisible(False)
            self.label_elapsed_time.setVisible(False)
            self.label_eta.setVisible(False)
            self._set_ui_controls_enabled(True)
            self.btn_start.blockSignals(True)
            self.btn_start.setChecked(False)
            self.btn_start.blockSignals(False)
            self.btn_start.setText("Start Full Tracking")
            self._apply_ui_state("idle" if self.current_video_path else "no_video")
            if finished_normally:
                logger.info("Preview completed.")
            else:
                QMessageBox.warning(
                    self,
                    "Preview Interrupted",
                    "Preview was stopped or encountered an error.",
                )
            gc.collect()
            return  # Exit early - no post-processing, no backward tracking for preview

        worker_props_path = ""
        if self.tracking_worker is not None:
            worker_props_path = str(
                getattr(self.tracking_worker, "individual_properties_cache_path", "")
                or ""
            ).strip()
        if worker_props_path:
            self.current_individual_properties_cache_path = worker_props_path
            logger.info(
                "Using individual properties cache for export: %s",
                worker_props_path,
            )

        if finished_normally:
            logger.info("Tracking completed successfully.")
            is_backward_mode = (
                hasattr(self.tracking_worker, "backward_mode")
                and self.tracking_worker.backward_mode
            )
            is_backward_enabled = self.chk_enable_backward.isChecked()

            processed_trajectories = full_traj
            if self.enable_postprocessing.isChecked():
                params = self.get_parameters_dict()
                raw_csv_path = self.csv_line.text()

                if is_backward_mode and raw_csv_path:
                    # Use backward CSV for processing
                    base, ext = os.path.splitext(raw_csv_path)
                    csv_to_process = f"{base}_backward{ext}"
                elif is_backward_enabled and raw_csv_path:
                    # Forward mode with backward enabled: use _forward.csv
                    # (tracking writes to _forward.csv when backward is enabled)
                    base, ext = os.path.splitext(raw_csv_path)
                    csv_to_process = f"{base}_forward{ext}"
                else:
                    # Forward-only mode: use base path
                    csv_to_process = raw_csv_path

                if csv_to_process and os.path.exists(csv_to_process):
                    # Use CSV-based processing to preserve confidence columns
                    from ..core.post.processing import (
                        interpolate_trajectories,
                        process_trajectories_from_csv,
                    )

                    processed_trajectories, stats = process_trajectories_from_csv(
                        csv_to_process, params
                    )
                    logger.info(f"Post-processing stats: {stats}")

                    # NOTE: Do NOT apply interpolation here if backward tracking will follow
                    # Interpolation should only be applied AFTER merging (in MergeWorker)
                    # or in forward-only mode (below). Pre-merge interpolation can affect
                    # merge candidate detection by filling gaps that should remain as gaps.

                    # NOTE: Do NOT scale to original space yet if backward tracking will happen
                    # Scaling will be done after merging or in forward-only mode
                else:
                    # Fallback to old method if CSV not available
                    processed_trajectories, stats = process_trajectories(
                        full_traj, params
                    )
                    logger.info(f"Post-processing stats (fallback): {stats}")

            if not is_backward_mode:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate forward CSV as temporary (only if cleanup enabled)
                    forward_csv = f"{base}_forward{ext}"
                    if (
                        self.chk_cleanup_temp_files.isChecked()
                        and forward_csv not in self.temporary_files
                    ):
                        self.temporary_files.append(forward_csv)

                    processed_csv_path = f"{base}_forward_processed{ext}"
                    # Only track processed CSV as temporary if backward tracking will run
                    # and cleanup is enabled (it will be merged into final file).
                    # Otherwise, this IS the final file.
                    if (
                        is_backward_enabled
                        and self.chk_cleanup_temp_files.isChecked()
                        and processed_csv_path not in self.temporary_files
                    ):
                        self.temporary_files.append(processed_csv_path)

                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )

                if is_backward_enabled:
                    self.forward_processed_trajs = processed_trajectories
                    # Log coordinate ranges for debugging
                    if (
                        isinstance(processed_trajectories, pd.DataFrame)
                        and not processed_trajectories.empty
                    ):
                        logger.info(
                            f"Forward trajectories stored for merge: "
                            f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                            f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
                        )
                    self.start_backward_tracking()
                else:
                    # Forward-only mode: Apply interpolation here (no merge step)
                    from ..core.post.processing import interpolate_trajectories

                    interp_method = (
                        self.combo_interpolation_method.currentText().lower()
                    )
                    if interp_method != "none":
                        max_gap = self.spin_interpolation_max_gap.value()
                        processed_trajectories = interpolate_trajectories(
                            processed_trajectories,
                            method=interp_method,
                            max_gap=max_gap,
                        )

                    # Scale coordinates to original video space (forward-only mode)
                    resize_factor = self.spin_resize.value()
                    processed_trajectories = self._scale_trajectories_to_original_space(
                        processed_trajectories, resize_factor
                    )

                    # Re-save the scaled trajectories
                    final_csv_path = None
                    if raw_csv_path:
                        base, ext = os.path.splitext(raw_csv_path)
                        # In forward-only mode, the final output is _forward_processed.csv
                        final_csv_path = f"{base}_forward_processed{ext}"
                        self.save_trajectories_to_csv(
                            processed_trajectories, final_csv_path
                        )
                        # Track initial tracking CSV as temporary (only if cleanup enabled)
                        if (
                            self.chk_cleanup_temp_files.isChecked()
                            and raw_csv_path not in self.temporary_files
                        ):
                            self.temporary_files.append(raw_csv_path)

                    # Complete session pipeline. Video generation is deferred to
                    # the final step after pose export + interpolation.
                    self._finish_tracking_session(final_csv_path=final_csv_path)
                    return
            else:
                raw_csv_path = self.csv_line.text()
                if raw_csv_path:
                    base, ext = os.path.splitext(raw_csv_path)
                    # Track intermediate backward CSV as temporary (only if cleanup enabled)
                    backward_csv = f"{base}_backward{ext}"
                    if (
                        self.chk_cleanup_temp_files.isChecked()
                        and backward_csv not in self.temporary_files
                    ):
                        self.temporary_files.append(backward_csv)

                    processed_csv_path = f"{base}_backward_processed{ext}"
                    # Track processed CSV as temporary (only if cleanup enabled)
                    if (
                        self.chk_cleanup_temp_files.isChecked()
                        and processed_csv_path not in self.temporary_files
                    ):
                        self.temporary_files.append(processed_csv_path)
                    self.save_trajectories_to_csv(
                        processed_trajectories, processed_csv_path
                    )
                self.backward_processed_trajs = processed_trajectories
                # Log coordinate ranges for debugging
                if (
                    isinstance(processed_trajectories, pd.DataFrame)
                    and not processed_trajectories.empty
                ):
                    logger.info(
                        f"Backward trajectories stored for merge: "
                        f"X range [{processed_trajectories['X'].min():.1f}, {processed_trajectories['X'].max():.1f}], "
                        f"Y range [{processed_trajectories['Y'].min():.1f}, {processed_trajectories['Y'].max():.1f}]"
                    )

                # Check if both forward and backward trajectories exist for merging
                has_forward = self.forward_processed_trajs is not None and (
                    isinstance(self.forward_processed_trajs, pd.DataFrame)
                    and not self.forward_processed_trajs.empty
                    or isinstance(self.forward_processed_trajs, list)
                    and len(self.forward_processed_trajs) > 0
                )
                has_backward = self.backward_processed_trajs is not None and (
                    isinstance(self.backward_processed_trajs, pd.DataFrame)
                    and not self.backward_processed_trajs.empty
                    or isinstance(self.backward_processed_trajs, list)
                    and len(self.backward_processed_trajs) > 0
                )

                if has_forward and has_backward:
                    # Start merge in background thread (will handle cleanup when done)
                    self.merge_and_save_trajectories()
                else:
                    # No merge needed, do cleanup now
                    # Pass the correct CSV path based on what we processed
                    self._finish_tracking_session(final_csv_path=processed_csv_path)

    def _is_pose_export_enabled(self) -> bool:
        """Return True when pose extraction export should be produced."""
        return bool(
            self._is_individual_pipeline_enabled()
            and hasattr(self, "chk_enable_pose_extractor")
            and self.chk_enable_pose_extractor.isChecked()
            and self._is_yolo_detection_mode()
        )

    def _export_pose_augmented_csv(self, final_csv_path):
        """Write a pose-augmented trajectories CSV next to the final CSV."""
        if not final_csv_path or not os.path.exists(final_csv_path):
            return None
        if not self._is_pose_export_enabled():
            return None

        cache_path = str(self.current_individual_properties_cache_path or "").strip()
        cache_available = bool(cache_path and os.path.exists(cache_path))
        interp_pose_path = str(self.current_interpolated_pose_csv_path or "").strip()
        interp_available = bool(interp_pose_path and os.path.exists(interp_pose_path))
        interp_pose_df_mem = getattr(self, "current_interpolated_pose_df", None)
        interp_mem_available = (
            isinstance(interp_pose_df_mem, pd.DataFrame)
            and not interp_pose_df_mem.empty
        )
        if not cache_available and not interp_available and not interp_mem_available:
            logger.warning(
                "Pose export skipped: no pose sources found (cache=%s, interpolated=%s, in_memory=%s).",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
                bool(interp_mem_available),
            )
            return None

        try:
            trajectories_df = pd.read_csv(final_csv_path)
        except Exception:
            logger.exception(
                "Pose export skipped: failed to load trajectories CSV: %s",
                final_csv_path,
            )
            return None

        try:
            with_pose_df = trajectories_df
            if cache_available:
                min_valid_conf = float(self.spin_pose_min_kpt_conf_valid.value())
                with_pose_df = augment_trajectories_with_pose_cache(
                    with_pose_df,
                    cache_path,
                    ignore_keypoints=self._parse_pose_ignore_keypoints(),
                    min_valid_conf=min_valid_conf,
                )
            if interp_available:
                interp_pose_df = pd.read_csv(interp_pose_path)
                with_pose_df = merge_interpolated_pose_df(with_pose_df, interp_pose_df)
            elif interp_mem_available:
                with_pose_df = merge_interpolated_pose_df(
                    with_pose_df, interp_pose_df_mem
                )
        except Exception:
            logger.exception(
                "Pose export skipped: failed while merging pose sources (cache=%s, interpolated=%s)",
                cache_path or "<empty>",
                interp_pose_path or "<empty>",
            )
            return None

        if with_pose_df is None or with_pose_df.empty:
            logger.warning(
                "Pose export skipped: merged pose dataframe is empty for %s",
                final_csv_path,
            )
            return None

        base, ext = os.path.splitext(final_csv_path)
        with_pose_path = f"{base}_with_pose{ext or '.csv'}"
        try:
            with_pose_df.to_csv(with_pose_path, index=False)
        except Exception:
            logger.exception("Failed to save pose-augmented CSV to: %s", with_pose_path)
            return None

        logger.info("Pose-augmented trajectories saved to: %s", with_pose_path)
        return with_pose_path

    def _load_video_trajectories(self, final_csv_path):
        """Load best available trajectories for video generation (prefers pose-augmented CSV)."""
        if not final_csv_path:
            return None, None
        base, ext = os.path.splitext(final_csv_path)
        with_pose_path = f"{base}_with_pose{ext or '.csv'}"
        candidate = with_pose_path if os.path.exists(with_pose_path) else final_csv_path
        if not os.path.exists(candidate):
            return None, None
        try:
            return pd.read_csv(candidate), candidate
        except Exception:
            logger.exception("Failed to load video trajectories from: %s", candidate)
            return None, None

    def _run_pending_video_generation_or_finalize(self):
        """Run video generation if queued; otherwise finalize UI/session cleanup."""
        if self._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        csv_path = self._pending_video_csv_path
        should_render_video = bool(self._pending_video_generation and csv_path)
        self._pending_video_generation = False
        self._pending_video_csv_path = None
        self._pending_pose_export_csv_path = None

        if should_render_video:
            trajectories_df, loaded_path = self._load_video_trajectories(csv_path)
            if trajectories_df is None or trajectories_df.empty:
                logger.warning(
                    "Skipping final video generation: no trajectories loaded from %s",
                    csv_path,
                )
                self._finalize_tracking_session_ui()
                return
            logger.info(
                "Final video rendering uses trajectories from: %s",
                loaded_path or csv_path,
            )
            self._generate_video_from_trajectories(
                trajectories_df,
                csv_path=csv_path,
                finalize_on_complete=False,
            )
            return

        self._finalize_tracking_session_ui()

    def _finish_tracking_session(self, final_csv_path=None):
        """Complete tracking session cleanup and UI updates."""
        if self._stop_all_requested:
            self._finalize_tracking_session_ui()
            return
        # Hide progress elements
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

        if final_csv_path:
            self._pending_pose_export_csv_path = final_csv_path
            self._export_pose_augmented_csv(final_csv_path)

        self._pending_video_csv_path = final_csv_path
        self._pending_video_generation = bool(
            final_csv_path
            and self.check_video_output.isChecked()
            and self.video_out_line.text().strip()
        )

        # Generate dataset if enabled (BEFORE cleanup so files are still available)
        if self.chk_enable_dataset_gen.isChecked():
            self._generate_training_dataset(override_csv_path=final_csv_path)

        # Interpolate occlusions for individual analysis (post-pass).
        # This also powers pose enrichment on occluded frames in final CSV.
        if self._should_run_interpolated_postpass():
            started = self._generate_interpolated_individual_crops(final_csv_path)
            if started:
                # Hold final UI/session completion until interpolation finishes.
                self._pending_finish_after_interp = True
                return

        self._run_pending_video_generation_or_finalize()

    def _finalize_tracking_session_ui(self):
        """Finalize session cleanup and return UI to idle state."""
        self._pending_pose_export_csv_path = None
        self._pending_video_csv_path = None
        self._pending_video_generation = False
        self.current_interpolated_pose_df = None
        # Force-clear progress UI at terminal session state.
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")
        # Clean up session logging
        self._cleanup_session_logging()
        self._cleanup_temporary_files()

        # Hide stats labels
        self.label_current_fps.setVisible(False)
        self.label_elapsed_time.setVisible(False)
        self.label_eta.setVisible(False)
        self._set_ui_controls_enabled(True)
        self.btn_start.blockSignals(True)
        self.btn_start.setChecked(False)
        self.btn_start.blockSignals(False)
        self.btn_start.setText("Start Full Tracking")
        self._apply_ui_state("idle" if self.current_video_path else "no_video")
        logger.info("✓ Tracking session complete.")

    def _generate_interpolated_individual_crops(self, csv_path):
        """Post-pass interpolation for occluded segments in individual dataset."""
        try:
            if self._stop_all_requested:
                return False
            if not self.chk_individual_interpolate.isChecked():
                return False

            target_csv = None
            if csv_path and os.path.exists(csv_path):
                target_csv = csv_path
            elif self.csv_line.text() and os.path.exists(self.csv_line.text()):
                target_csv = self.csv_line.text()
            if not target_csv or not os.path.exists(target_csv):
                return False

            video_path = self.file_line.text()
            if not video_path or not os.path.exists(video_path):
                return False

            params = self.get_parameters_dict()
            output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
            if not output_dir:
                # Keep interpolated analysis available even when image-save toggle is off.
                csv_dir = os.path.dirname(target_csv) if target_csv else ""
                fallback_output = (
                    os.path.join(csv_dir, "training_data") if csv_dir else ""
                )
                if fallback_output:
                    try:
                        os.makedirs(fallback_output, exist_ok=True)
                    except Exception:
                        pass
                    params["INDIVIDUAL_DATASET_OUTPUT_DIR"] = fallback_output
                    logger.info(
                        "Interpolated analysis output dir not set; using fallback: %s",
                        fallback_output,
                    )

            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Interpolating occluded crops...")

            if self.interp_worker is not None and self.interp_worker.isRunning():
                logger.warning(
                    "Interpolated crop generation already in progress; skipping duplicate request."
                )
                return True
            if self.interp_worker is not None and not self.interp_worker.isRunning():
                self.interp_worker.deleteLater()
                self.interp_worker = None

            self.current_interpolated_pose_csv_path = None
            self.current_interpolated_pose_df = None
            self.interp_worker = InterpolatedCropsWorker(
                target_csv,
                video_path,
                self.current_detection_cache_path,
                params,
            )
            self.interp_worker.progress_signal.connect(self.on_progress_update)
            self.interp_worker.finished_signal.connect(
                self._on_interpolated_crops_finished
            )
            self.interp_worker.start()
            return True
        except Exception as e:
            logger.warning(f"Interpolated individual crops failed: {e}")
            return False

    def _interp_angle(self, theta_start, theta_end, t):
        deg0 = math.degrees(theta_start)
        deg1 = math.degrees(theta_end)
        # Treat OBB angles as 180-degree periodic; pick shortest path.
        candidates = (deg1, deg1 + 180.0, deg1 - 180.0)
        best_delta = None
        for cand in candidates:
            delta = wrap_angle_degs(cand - deg0)
            if best_delta is None or abs(delta) < abs(best_delta):
                best_delta = delta
        return math.radians(deg0 + (best_delta or 0.0) * t)

    def _get_detection_size(self, detection_cache, frame_id, detection_id, params):
        if detection_cache is None or detection_id is None or pd.isna(detection_id):
            return None, None
        try:
            _, _, shapes, _, obb_corners, detection_ids = detection_cache.get_frame(
                int(frame_id)
            )
        except Exception:
            return None, None

        idx = None
        try:
            for i, did in enumerate(detection_ids):
                if int(did) == int(detection_id):
                    idx = i
                    break
        except Exception:
            idx = None

        if idx is None:
            return None, None

        # YOLO OBB corners
        if obb_corners and idx < len(obb_corners):
            c = np.asarray(obb_corners[idx], dtype=np.float32)
            if c.shape[0] >= 4:
                w = float(np.linalg.norm(c[1] - c[0]))
                h = float(np.linalg.norm(c[2] - c[1]))
                if w < h:
                    w, h = h, w
                return w, h

        # Background subtraction shapes
        if shapes and idx < len(shapes):
            area, aspect_ratio = shapes[idx][0], shapes[idx][1]
            if aspect_ratio > 0 and area > 0:
                ax2 = math.sqrt(4 * area / (math.pi * aspect_ratio))
                ax1 = aspect_ratio * ax2
                return ax1, ax2

        return None, None

    @Slot(dict)
    def on_histogram_data(self: object, histogram_data: object) -> object:
        """on_histogram_data method documentation."""
        if (
            self.enable_histograms.isChecked()
            and self.histogram_window is not None
            and self.histogram_window.isVisible()
        ):

            current_history = self.spin_histogram_history.value()
            if self.histogram_panel.history_frames != current_history:
                self.histogram_panel.set_history_frames(current_history)

            if "velocities" in histogram_data:
                self.histogram_panel.update_velocity_data(histogram_data["velocities"])
            if "sizes" in histogram_data:
                self.histogram_panel.update_size_data(histogram_data["sizes"])
            if "orientations" in histogram_data:
                self.histogram_panel.update_orientation_data(
                    histogram_data["orientations"]
                )
            if "assignment_costs" in histogram_data:
                self.histogram_panel.update_assignment_cost_data(
                    histogram_data["assignment_costs"]
                )

    def start_backward_tracking(self: object) -> object:
        """start_backward_tracking method documentation."""
        if self._stop_all_requested:
            return
        logger.info("=" * 80)
        logger.info("Starting backward tracking pass (using cached detections)...")
        logger.info("=" * 80)

        video_fp = self.file_line.text()
        if not video_fp:
            return

        # Use original video (no reversal needed with detection caching)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Starting backward tracking (using cached detections)..."
        )
        QApplication.processEvents()

        # Start backward tracking directly on original video with cached detections
        self.start_tracking_on_video(video_fp, backward_mode=True)

    def start_tracking(
        self: object, preview_mode: bool, backward_mode: bool = False
    ) -> object:
        """start_tracking method documentation."""
        if not preview_mode:
            if not self.save_config():
                # User cancelled config save, abort tracking
                return
        video_fp = self.file_line.text()
        if not video_fp:
            QMessageBox.warning(self, "No video", "Please select a video file first.")
            return
        if preview_mode:
            self.start_preview_on_video(video_fp)
        else:
            self.start_tracking_on_video(video_fp, backward_mode=False)

    def start_preview_on_video(self: object, video_path: object) -> object:
        """start_preview_on_video method documentation."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return
        self._stop_all_requested = False
        self._pending_finish_after_interp = False

        # Stop video playback if active
        if self.is_playing:
            self._stop_playback()

        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True
        self.csv_writer_thread = None
        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=None,
            video_output_path=None,
            backward_mode=False,
            detection_cache_path=None,  # No caching in preview mode
            preview_mode=True,  # Preview mode - frame-by-frame only
        )
        params = self.get_parameters_dict()
        # Preview should always render frames regardless of visualization-free toggle
        params["VISUALIZATION_FREE_MODE"] = False
        self.tracking_worker.set_parameters(params)
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)
        self.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Preview Mode Active")

        self._prepare_tracking_display()
        self._apply_ui_state("preview")
        self.tracking_worker.start()

    def start_tracking_on_video(
        self: object, video_path: object, backward_mode: object = False
    ) -> object:
        """start_tracking_on_video method documentation."""
        if self.tracking_worker and self.tracking_worker.isRunning():
            return
        self._stop_all_requested = False
        self._pending_finish_after_interp = False

        # Stop video playback if active
        if self.is_playing:
            self._stop_playback()

        # Reset first frame flag for auto-fit
        self._tracking_first_frame = True

        # Session logging is already set up in start_full() - don't duplicate here
        # For backward mode, we reuse the same session log

        self.csv_writer_thread = None
        if self.csv_line.text():
            # Determine header based on confidence tracking setting
            save_confidence = self.check_save_confidence.isChecked()
            if save_confidence:
                hdr = [
                    "TrackID",
                    "TrajectoryID",
                    "Index",
                    "X",
                    "Y",
                    "Theta",
                    "FrameID",
                    "State",
                    "DetectionConfidence",
                    "AssignmentConfidence",
                    "PositionUncertainty",
                    "DetectionID",
                ]
            else:
                hdr = [
                    "TrackID",
                    "TrajectoryID",
                    "Index",
                    "X",
                    "Y",
                    "Theta",
                    "FrameID",
                    "State",
                    "DetectionID",
                ]
            csv_path = self.csv_line.text()
            base, ext = os.path.splitext(csv_path)
            if backward_mode:
                csv_path = f"{base}_backward{ext}"
            elif self.chk_enable_backward.isChecked():
                # Forward mode with backward tracking enabled - save as _forward.csv
                csv_path = f"{base}_forward{ext}"
            self.csv_writer_thread = CSVWriterThread(csv_path, header=hdr)
            self.csv_writer_thread.start()

        # Video output is no longer generated during tracking
        # Instead, it's generated from post-processed trajectories after merging
        # This ensures the video shows clean, merged trajectories with stable IDs
        video_output_path = None

        # Generate detection cache path based on video and detection method
        # Cache is always created for forward tracking to allow reuse on reruns
        detection_cache_path = None
        params = self.get_parameters_dict()
        logger.info(
            f"Launching {'backward' if backward_mode else 'forward'} tracking for frame range "
            f"{params.get('START_FRAME')}..{params.get('END_FRAME')}"
        )
        detection_method = params.get("DETECTION_METHOD", "background_subtraction")
        use_cached_detections = self.chk_use_cached_detections.isChecked()

        # Generate model-specific cache name
        def get_inference_model_id() -> object:
            """Generate an inference identity key shared by raw detection and TensorRT cache."""
            # Include resize factor in cache ID since detections are scale-dependent
            resize_factor = params.get("RESIZE_FACTOR", 1.0)
            resize_str = f"r{int(resize_factor * 100)}"

            def normalize_for_hash(value: object) -> object:
                """Convert values to deterministic, JSON-safe forms for hashing."""
                if isinstance(value, np.ndarray):
                    arr = np.ascontiguousarray(value)
                    return {
                        "type": "ndarray",
                        "dtype": str(arr.dtype),
                        "shape": list(arr.shape),
                        "digest": hashlib.md5(arr.tobytes()).hexdigest(),
                    }
                if isinstance(value, np.integer):
                    return int(value)
                if isinstance(value, np.floating):
                    if np.isnan(value):
                        return "NaN"
                    if np.isinf(value):
                        return "Infinity" if value > 0 else "-Infinity"
                    return float(value)
                if isinstance(value, np.bool_):
                    return bool(value)
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, dict):
                    return {
                        str(k): normalize_for_hash(v)
                        for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                    }
                if isinstance(value, (list, tuple)):
                    return [normalize_for_hash(v) for v in value]
                return value

            def extract_hash_params(keys: object) -> object:
                return {k: normalize_for_hash(params.get(k)) for k in keys}

            def get_model_fingerprint(model_path: object) -> object:
                configured = str(model_path or "")
                resolved = str(resolve_model_path(configured))
                fingerprint = {
                    "configured_path": configured,
                    "resolved_path": resolved,
                }
                if resolved and os.path.exists(resolved):
                    try:
                        stat = os.stat(resolved)
                        fingerprint["size_bytes"] = stat.st_size
                        fingerprint["mtime_ns"] = stat.st_mtime_ns
                    except OSError:
                        fingerprint["size_bytes"] = None
                        fingerprint["mtime_ns"] = None
                else:
                    fingerprint["size_bytes"] = None
                    fingerprint["mtime_ns"] = None
                return fingerprint

            # Common inference settings that affect detections for both methods.
            common_detection_keys = (
                "DETECTION_METHOD",
                "RESIZE_FACTOR",
                "MAX_TARGETS",
                "COMPUTE_RUNTIME",
            )

            if detection_method == "yolo_obb":
                yolo_model = params.get("YOLO_MODEL_PATH", "best.pt")
                model_fingerprint = get_model_fingerprint(yolo_model)
                model_name = os.path.basename(
                    model_fingerprint["resolved_path"]
                    or model_fingerprint["configured_path"]
                )
                model_stem = os.path.splitext(model_name)[0] or "model"
                safe_model_stem = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_" for c in model_stem
                )

                yolo_inference_keys = (
                    "YOLO_TARGET_CLASSES",
                    "YOLO_DEVICE",
                    "ENABLE_TENSORRT",
                    "TENSORRT_MAX_BATCH_SIZE",
                )
                cache_params = {
                    "common": extract_hash_params(common_detection_keys),
                    "yolo": extract_hash_params(yolo_inference_keys),
                    "model": normalize_for_hash(model_fingerprint),
                    # Bump when raw detection extraction/filtering semantics change.
                    "raw_detection_cache_version": 3,
                }
                # Class order should not change cache identity.
                classes = cache_params["yolo"].get("YOLO_TARGET_CLASSES")
                if classes is not None:
                    if isinstance(classes, str):
                        raw_classes = [
                            c.strip() for c in classes.split(",") if c.strip()
                        ]
                    elif isinstance(classes, (list, tuple)):
                        raw_classes = list(classes)
                    else:
                        raw_classes = [classes]
                    try:
                        cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                            int(c) for c in raw_classes
                        )
                    except (TypeError, ValueError):
                        cache_params["yolo"]["YOLO_TARGET_CLASSES"] = sorted(
                            str(c) for c in raw_classes
                        )

                h = hashlib.md5(
                    json.dumps(cache_params, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
                return f"yolo_{safe_model_stem}_{resize_str}_{h}"

            bg_detection_keys = (
                "MAX_CONTOUR_MULTIPLIER",
                "ENABLE_SIZE_FILTERING",
                "MIN_OBJECT_SIZE",
                "MAX_OBJECT_SIZE",
                "ROI_MASK",
                "BACKGROUND_PRIME_FRAMES",
                "ENABLE_ADAPTIVE_BACKGROUND",
                "BACKGROUND_LEARNING_RATE",
                "ENABLE_GPU_BACKGROUND",
                "GPU_DEVICE_ID",
                "THRESHOLD_VALUE",
                "MORPH_KERNEL_SIZE",
                "ENABLE_ADDITIONAL_DILATION",
                "DILATION_ITERATIONS",
                "DILATION_KERNEL_SIZE",
                "BRIGHTNESS",
                "CONTRAST",
                "GAMMA",
                "DARK_ON_LIGHT_BACKGROUND",
                "ENABLE_LIGHTING_STABILIZATION",
                "LIGHTING_SMOOTH_FACTOR",
                "LIGHTING_MEDIAN_WINDOW",
                "ENABLE_CONSERVATIVE_SPLIT",
                "MERGE_AREA_THRESHOLD",
                "CONSERVATIVE_KERNEL_SIZE",
                "CONSERVATIVE_ERODE_ITER",
                "MIN_CONTOUR_AREA",
                "MIN_DETECTIONS_TO_START",
                "MIN_DETECTION_COUNTS",
            )
            cache_params = {
                "common": extract_hash_params(common_detection_keys),
                "background_subtraction": extract_hash_params(bg_detection_keys),
            }
            h = hashlib.md5(
                json.dumps(cache_params, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            return f"bgsub_{resize_str}_{h}"

        base_name = os.path.splitext(video_path)[0]
        model_id = get_inference_model_id()
        # Share the exact inference hash with downstream detector code so
        # TensorRT engine cache and raw detection cache are invalidated together.
        params["INFERENCE_MODEL_ID"] = model_id
        base_prefix = os.path.basename(base_name) + "_detection_cache_"

        # Choose a writable directory for cache (prefer video dir, then CSV dir, else temp)
        cache_dir = os.path.dirname(video_path)
        if not os.access(cache_dir, os.W_OK):
            csv_dir = (
                os.path.dirname(self.csv_line.text()) if self.csv_line.text() else ""
            )
            if csv_dir and os.access(csv_dir, os.W_OK):
                cache_dir = csv_dir
            else:
                cache_dir = tempfile.gettempdir()
                logger.warning(
                    f"Video directory not writable; using temp cache dir: {cache_dir}"
                )

        detection_cache_path = os.path.join(cache_dir, f"{base_prefix}{model_id}.npz")

        # Do NOT delete old detection caches; keep all for reuse
        self.current_detection_cache_path = detection_cache_path

        self.tracking_worker = TrackingWorker(
            video_path,
            csv_writer_thread=self.csv_writer_thread,
            video_output_path=video_output_path,
            backward_mode=backward_mode,
            detection_cache_path=detection_cache_path,
            preview_mode=False,  # Full tracking mode - batching enabled if applicable
            use_cached_detections=use_cached_detections,
        )
        self.tracking_worker.set_parameters(params)
        self.parameters_changed.connect(self.tracking_worker.update_parameters)
        self.tracking_worker.frame_signal.connect(self.on_new_frame)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.progress_signal.connect(self.on_progress_update)
        self.tracking_worker.histogram_data_signal.connect(self.on_histogram_data)
        self.tracking_worker.stats_signal.connect(self.on_stats_update)
        self.tracking_worker.warning_signal.connect(self.on_tracking_warning)
        self.tracking_worker.pose_exported_model_resolved_signal.connect(
            self.on_pose_exported_model_resolved
        )

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText(
            "Backward Tracking..." if backward_mode else "Forward Tracking..."
        )

        self._prepare_tracking_display()
        self._apply_ui_state("tracking")
        self.tracking_worker.start()

    def get_parameters_dict(self: object) -> object:
        """get_parameters_dict method documentation."""
        N = self.spin_max_targets.value()
        np.random.seed(42)
        colors = [tuple(c.tolist()) for c in np.random.randint(0, 255, (N, 3))]

        det_method = (
            "background_subtraction"
            if self.combo_detection_method.currentIndex() == 0
            else "yolo_obb"
        )

        yolo_path = resolve_model_path(
            self._get_selected_yolo_model_path() or "yolo26s-obb.pt"
        )

        yolo_cls = None
        if self.line_yolo_classes.text().strip():
            try:
                yolo_cls = [
                    int(x.strip()) for x in self.line_yolo_classes.text().split(",")
                ]
            except ValueError:
                pass

        # Calculate actual pixel values from body-size multipliers
        reference_body_size = self.spin_reference_body_size.value()
        resize_factor = self.spin_resize.value()
        scaled_body_size = reference_body_size * resize_factor

        # Area is π * (diameter/2)^2
        import math

        reference_body_area = math.pi * (reference_body_size / 2.0) ** 2
        scaled_body_area = reference_body_area * (resize_factor**2)

        # Convert multipliers to actual pixels
        min_object_size_pixels = int(
            self.spin_min_object_size.value() * scaled_body_area
        )
        max_object_size_pixels = int(
            self.spin_max_object_size.value() * scaled_body_area
        )
        max_distance_pixels = self.spin_max_dist.value() * scaled_body_size
        recovery_search_distance_pixels = (
            self.spin_continuity_thresh.value() * scaled_body_size
        )
        min_respawn_distance_pixels = (
            self.spin_min_respawn_distance.value() * scaled_body_size
        )

        # Convert time-based velocities to frame-based for tracking
        fps = self.spin_fps.value()
        velocity_threshold_pixels_per_frame = (
            self.spin_velocity.value() * scaled_body_size / fps
        )
        max_velocity_break_pixels_per_frame = (
            self.spin_max_velocity_break.value() * scaled_body_size / fps
        )

        # YOLO Batching settings from UI (overrides advanced_config defaults)
        advanced_config = self.advanced_config.copy()
        advanced_config["enable_yolo_batching"] = (
            self.chk_enable_yolo_batching.isChecked()
        )
        advanced_config["yolo_batch_size_mode"] = (
            "auto" if self.combo_yolo_batch_mode.currentIndex() == 0 else "manual"
        )
        advanced_config["yolo_manual_batch_size"] = self.spin_yolo_batch_size.value()
        advanced_config["video_show_pose"] = self.check_video_show_pose.isChecked()
        advanced_config["video_pose_point_radius"] = (
            self.spin_video_pose_point_radius.value()
        )
        advanced_config["video_pose_point_thickness"] = (
            self.spin_video_pose_point_thickness.value()
        )
        advanced_config["video_pose_line_thickness"] = (
            self.spin_video_pose_line_thickness.value()
        )
        advanced_config["video_pose_color_mode"] = (
            "track" if self.combo_video_pose_color_mode.currentIndex() == 0 else "fixed"
        )
        advanced_config["video_pose_color"] = [
            int(self._video_pose_color[0]),
            int(self._video_pose_color[1]),
            int(self._video_pose_color[2]),
        ]

        individual_pipeline_enabled = self._is_individual_pipeline_enabled()
        individual_image_save_enabled = self._is_individual_image_save_enabled()
        compute_runtime = self._selected_compute_runtime()
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        trt_batch_size = (
            self.spin_yolo_batch_size.value()
            if self._runtime_requires_fixed_yolo_batch(compute_runtime)
            else self.spin_tensorrt_batch.value()
        )
        pose_backend_family = self.combo_pose_model_type.currentText().strip().lower()
        runtime_pose = derive_pose_runtime_settings(
            compute_runtime, backend_family=pose_backend_family
        )

        return {
            "ADVANCED_CONFIG": advanced_config,  # Include advanced config for batch optimization
            "DETECTION_METHOD": det_method,
            "FPS": fps,  # Acquisition frame rate
            # Keep selected frame range stable even when controls are disabled
            # during tracking/backward pass.
            "START_FRAME": self.spin_start_frame.value(),
            "END_FRAME": self.spin_end_frame.value(),
            "YOLO_MODEL_PATH": yolo_path,
            "YOLO_CONFIDENCE_THRESHOLD": self.spin_yolo_confidence.value(),
            "YOLO_IOU_THRESHOLD": self.spin_yolo_iou.value(),
            "USE_CUSTOM_OBB_IOU_FILTERING": True,
            "YOLO_TARGET_CLASSES": yolo_cls,
            "COMPUTE_RUNTIME": compute_runtime,
            "YOLO_DEVICE": runtime_detection["yolo_device"],
            "ENABLE_GPU_BACKGROUND": runtime_detection["enable_gpu_background"],
            "ENABLE_TENSORRT": runtime_detection["enable_tensorrt"],
            "ENABLE_ONNX_RUNTIME": runtime_detection["enable_onnx_runtime"],
            "TENSORRT_MAX_BATCH_SIZE": trt_batch_size,
            "MAX_TARGETS": N,
            "THRESHOLD_VALUE": self.spin_threshold.value(),
            "MORPH_KERNEL_SIZE": self.spin_morph_size.value(),
            "MIN_CONTOUR_AREA": self.spin_min_contour.value(),
            "ENABLE_SIZE_FILTERING": self.chk_size_filtering.isChecked(),
            "MIN_OBJECT_SIZE": min_object_size_pixels,
            "MAX_OBJECT_SIZE": max_object_size_pixels,
            "MAX_CONTOUR_MULTIPLIER": self.spin_max_contour_multiplier.value(),
            "MAX_DISTANCE_THRESHOLD": max_distance_pixels,
            "ENABLE_POSTPROCESSING": self.enable_postprocessing.isChecked(),
            "MIN_TRAJECTORY_LENGTH": self.spin_min_trajectory_length.value(),
            "MAX_VELOCITY_BREAK": max_velocity_break_pixels_per_frame,
            "MAX_OCCLUSION_GAP": self.spin_max_occlusion_gap.value(),
            "MAX_VELOCITY_ZSCORE": self.spin_max_velocity_zscore.value(),
            "VELOCITY_ZSCORE_WINDOW": self.spin_velocity_zscore_window.value(),
            "VELOCITY_ZSCORE_MIN_VELOCITY": self.spin_velocity_zscore_min_vel.value()
            * scaled_body_size
            / fps,
            "CONTINUITY_THRESHOLD": recovery_search_distance_pixels,
            "MIN_RESPAWN_DISTANCE": min_respawn_distance_pixels,
            "MIN_DETECTION_COUNTS": self.spin_min_detect.value(),
            "MIN_DETECTIONS_TO_START": self.spin_min_detections_to_start.value(),
            "MIN_TRACKING_COUNTS": self.spin_min_track.value(),
            "TRAJECTORY_HISTORY_SECONDS": self.spin_traj_hist.value(),
            "BACKGROUND_PRIME_FRAMES": self.spin_bg_prime.value(),
            "ENABLE_LIGHTING_STABILIZATION": self.chk_lighting_stab.isChecked(),
            "ENABLE_ADAPTIVE_BACKGROUND": self.chk_adaptive_bg.isChecked(),
            "BACKGROUND_LEARNING_RATE": self.spin_bg_learning.value(),
            "LIGHTING_SMOOTH_FACTOR": self.spin_lighting_smooth.value(),
            "LIGHTING_MEDIAN_WINDOW": self.spin_lighting_median.value(),
            "KALMAN_NOISE_COVARIANCE": self.spin_kalman_noise.value(),
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": self.spin_kalman_meas.value(),
            "KALMAN_DAMPING": self.spin_kalman_damping.value(),
            "KALMAN_MATURITY_AGE": self.spin_kalman_maturity_age.value(),
            "KALMAN_INITIAL_VELOCITY_RETENTION": self.spin_kalman_initial_velocity_retention.value(),
            "KALMAN_MAX_VELOCITY_MULTIPLIER": self.spin_kalman_max_velocity.value(),
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": self.spin_kalman_longitudinal_noise.value(),
            "KALMAN_LATERAL_NOISE_MULTIPLIER": self.spin_kalman_lateral_noise.value(),
            "RESIZE_FACTOR": self.spin_resize.value(),
            "ENABLE_CONSERVATIVE_SPLIT": self.chk_conservative_split.isChecked(),
            "MERGE_AREA_THRESHOLD": self.spin_merge_threshold.value(),
            "CONSERVATIVE_KERNEL_SIZE": self.spin_conservative_kernel.value(),
            "CONSERVATIVE_ERODE_ITER": self.spin_conservative_erode.value(),
            "ENABLE_ADDITIONAL_DILATION": self.chk_additional_dilation.isChecked(),
            "DILATION_ITERATIONS": self.spin_dilation_iterations.value(),
            "DILATION_KERNEL_SIZE": self.spin_dilation_kernel_size.value(),
            "BRIGHTNESS": self.slider_brightness.value(),
            "CONTRAST": self.slider_contrast.value() / 100.0,
            "GAMMA": self.slider_gamma.value() / 100.0,
            "DARK_ON_LIGHT_BACKGROUND": self.chk_dark_on_light.isChecked(),
            "VELOCITY_THRESHOLD": velocity_threshold_pixels_per_frame,
            "INSTANT_FLIP_ORIENTATION": self.chk_instant_flip.isChecked(),
            "MAX_ORIENT_DELTA_STOPPED": self.spin_max_orient.value(),
            "LOST_THRESHOLD_FRAMES": self.spin_lost_thresh.value(),
            "W_POSITION": self.spin_Wp.value(),
            "W_ORIENTATION": self.spin_Wo.value(),
            "W_AREA": self.spin_Wa.value(),
            "W_ASPECT": self.spin_Wasp.value(),
            "USE_MAHALANOBIS": self.chk_use_mahal.isChecked(),
            "ENABLE_GREEDY_ASSIGNMENT": self.combo_assignment_method.currentIndex()
            == 1,
            "ENABLE_SPATIAL_OPTIMIZATION": self.chk_spatial_optimization.isChecked(),
            "TRAJECTORY_COLORS": colors,
            "SHOW_FG": self.chk_show_fg.isChecked(),
            "SHOW_BG": self.chk_show_bg.isChecked(),
            "SHOW_CIRCLES": self.chk_show_circles.isChecked(),
            "SHOW_ORIENTATION": self.chk_show_orientation.isChecked(),
            "SHOW_YOLO_OBB": self.chk_show_yolo_obb.isChecked(),
            "SHOW_TRAJECTORIES": self.chk_show_trajectories.isChecked(),
            "SHOW_LABELS": self.chk_show_labels.isChecked(),
            "SHOW_STATE": self.chk_show_state.isChecked(),
            "SHOW_KALMAN_UNCERTAINTY": self.chk_show_kalman_uncertainty.isChecked(),
            "VISUALIZATION_FREE_MODE": self.chk_visualization_free.isChecked(),
            "zoom_factor": self.slider_zoom.value() / 100.0,
            "ENABLE_HISTOGRAMS": self.enable_histograms.isChecked(),
            "HISTOGRAM_HISTORY_FRAMES": self.spin_histogram_history.value(),
            "ROI_MASK": self.roi_mask,
            "REFERENCE_BODY_SIZE": reference_body_size,
            # Conservative trajectory merging parameters (in resized coordinate space)
            # These are used in resolve_trajectories() for bidirectional merging
            # AGREEMENT_DISTANCE: max distance for frames to be considered "agreeing"
            # MIN_OVERLAP_FRAMES: minimum agreeing frames to consider merge candidates
            "AGREEMENT_DISTANCE": self.spin_merge_overlap_multiplier.value()
            * scaled_body_size,
            "MIN_OVERLAP_FRAMES": self.spin_min_overlap_frames.value(),
            # Dataset generation parameters
            "ENABLE_DATASET_GENERATION": self.chk_enable_dataset_gen.isChecked(),
            "DATASET_NAME": self.line_dataset_name.text(),
            "DATASET_CLASS_NAME": self.line_dataset_class_name.text(),
            "DATASET_OUTPUT_DIR": self.line_dataset_output.text(),
            "DATASET_MAX_FRAMES": self.spin_dataset_max_frames.value(),
            "DATASET_CONF_THRESHOLD": self.spin_dataset_conf_threshold.value(),
            # Dataset-specific YOLO parameters from advanced config (for export, not tracking)
            "DATASET_YOLO_CONFIDENCE_THRESHOLD": self.advanced_config.get(
                "dataset_yolo_confidence_threshold", 0.05
            ),
            "DATASET_YOLO_IOU_THRESHOLD": self.advanced_config.get(
                "dataset_yolo_iou_threshold", 0.5
            ),
            "DATASET_DIVERSITY_WINDOW": self.spin_dataset_diversity_window.value(),
            "DATASET_INCLUDE_CONTEXT": self.chk_dataset_include_context.isChecked(),
            "DATASET_PROBABILISTIC_SAMPLING": self.chk_dataset_probabilistic.isChecked(),
            "METRIC_LOW_CONFIDENCE": self.chk_metric_low_confidence.isChecked(),
            "METRIC_COUNT_MISMATCH": self.chk_metric_count_mismatch.isChecked(),
            "METRIC_HIGH_ASSIGNMENT_COST": self.chk_metric_high_assignment_cost.isChecked(),
            "METRIC_TRACK_LOSS": self.chk_metric_track_loss.isChecked(),
            "METRIC_HIGH_UNCERTAINTY": self.chk_metric_high_uncertainty.isChecked(),
            # Individual analysis parameters
            "ENABLE_IDENTITY_ANALYSIS": individual_pipeline_enabled,
            "ENABLE_INDIVIDUAL_PIPELINE": individual_pipeline_enabled,
            "IDENTITY_METHOD": self.combo_identity_method.currentText()
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""),
            "IDENTITY_CROP_SIZE_MULTIPLIER": self.spin_identity_crop_multiplier.value(),
            "IDENTITY_CROP_MIN_SIZE": self.spin_identity_crop_min.value(),
            "IDENTITY_CROP_MAX_SIZE": self.spin_identity_crop_max.value(),
            "COLOR_TAG_MODEL_PATH": self.line_color_tag_model.text(),
            "COLOR_TAG_CONFIDENCE": self.spin_color_tag_conf.value(),
            "APRILTAG_FAMILY": self.combo_apriltag_family.currentText(),
            "APRILTAG_DECIMATE": self.spin_apriltag_decimate.value(),
            "ENABLE_POSE_EXTRACTOR": self.chk_enable_pose_extractor.isChecked(),
            "POSE_MODEL_TYPE": self.combo_pose_model_type.currentText().strip().lower(),
            "POSE_MODEL_DIR": resolve_pose_model_path(
                self._pose_model_path_for_backend(
                    self.combo_pose_model_type.currentText().strip().lower()
                ),
                backend=self.combo_pose_model_type.currentText().strip().lower(),
            ),
            "POSE_RUNTIME_FLAVOR": runtime_pose["pose_runtime_flavor"],
            "POSE_EXPORTED_MODEL_PATH": "",
            "POSE_MIN_KPT_CONF_VALID": self.spin_pose_min_kpt_conf_valid.value(),
            "POSE_SKELETON_FILE": self.line_pose_skeleton_file.text().strip(),
            "POSE_IGNORE_KEYPOINTS": self._parse_pose_ignore_keypoints(),
            "POSE_DIRECTION_ANTERIOR_KEYPOINTS": self._parse_pose_direction_anterior_keypoints(),
            "POSE_DIRECTION_POSTERIOR_KEYPOINTS": self._parse_pose_direction_posterior_keypoints(),
            "POSE_YOLO_BATCH": self.spin_pose_batch.value(),
            "POSE_BATCH_SIZE": self.spin_pose_batch.value(),
            "POSE_SLEAP_ENV": self._selected_pose_sleap_env(),
            "POSE_SLEAP_DEVICE": runtime_pose["pose_sleap_device"],
            "POSE_SLEAP_BATCH": self.spin_pose_batch.value(),
            "POSE_SLEAP_MAX_INSTANCES": 1,
            "POSE_SLEAP_EXPERIMENTAL_FEATURES": self._sleap_experimental_features_enabled(),
            "INDIVIDUAL_PROPERTIES_CACHE_PATH": str(
                self.current_individual_properties_cache_path or ""
            ).strip(),
            # Real-time Individual Dataset Generation parameters
            "ENABLE_INDIVIDUAL_DATASET": individual_image_save_enabled,
            "ENABLE_INDIVIDUAL_IMAGE_SAVE": individual_image_save_enabled,
            "INDIVIDUAL_DATASET_NAME": self.line_individual_dataset_name.text().strip()
            or "individual_dataset",
            "INDIVIDUAL_DATASET_OUTPUT_DIR": self.line_individual_output.text(),
            "INDIVIDUAL_OUTPUT_FORMAT": self.combo_individual_format.currentText().lower(),
            "INDIVIDUAL_SAVE_INTERVAL": self.spin_individual_interval.value(),
            "INDIVIDUAL_INTERPOLATE_OCCLUSIONS": self.chk_individual_interpolate.isChecked(),
            "INDIVIDUAL_CROP_PADDING": self.spin_individual_padding.value(),
            "INDIVIDUAL_BACKGROUND_COLOR": [
                int(c) for c in self._background_color
            ],  # Ensure JSON serializable
            "INDIVIDUAL_DATASET_RUN_ID": self._individual_dataset_run_id,
        }

    def load_config(self: object) -> object:
        """Manually load config from file dialog."""
        config_path, _ = QFileDialog.getOpenFileName(
            self, "Load Configuration", "", "JSON Files (*.json)"
        )
        if config_path:
            self._load_config_from_file(config_path)
            self.config_status_label.setText(
                f"✓ Loaded: {os.path.basename(config_path)}"
            )
            self.config_status_label.setStyleSheet(
                "color: #4a9eff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Configuration loaded from {config_path}")

    def _load_config_from_file(self, config_path, preset_mode=False):
        """Internal method to load config from a specific file path.

        This method supports both new standardized key names and legacy key names
        for backward compatibility with older config files.

        Args:
            config_path: Path to the config/preset file
            preset_mode: If True, skip loading video paths and ROI data (for organism presets)
        """
        if not os.path.isfile(config_path):
            return

        def get_cfg(
            new_key: object, *legacy_keys: object, default: object = None
        ) -> object:
            """Helper to get config value with fallback to legacy keys."""
            if new_key in cfg:
                return cfg[new_key]
            for key in legacy_keys:
                if key in cfg:
                    return cfg[key]
            return default

        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # === FILE MANAGEMENT ===
            # Skip file paths when loading presets (only load for full configs)
            if not preset_mode:
                # Only set paths if they're currently empty (preserve existing paths)
                if not self.file_line.text().strip():
                    video_path = get_cfg("file_path", default="")
                    if video_path:
                        # Use the same setup logic as browsing for a file
                        self._setup_video_file(video_path, skip_config_load=True)
                if not self.csv_line.text().strip():
                    self.csv_line.setText(get_cfg("csv_path", default=""))
                self.check_video_output.setChecked(
                    get_cfg("video_output_enabled", default=False)
                )
                saved_video_path = get_cfg("video_output_path", default="")
                if saved_video_path and not self.video_out_line.text().strip():
                    self.video_out_line.setText(saved_video_path)

            # === REFERENCE PARAMETERS ===
            # Only load video-specific reference parameters from configs (not presets)
            if not preset_mode:
                # Load FPS if saved in config
                saved_fps = get_cfg("fps", default=None)
                if saved_fps is not None:
                    self.spin_fps.setValue(saved_fps)

                # Load reference body size if saved in config
                saved_body_size = get_cfg("reference_body_size", default=None)
                if saved_body_size is not None:
                    self.spin_reference_body_size.setValue(saved_body_size)

                # Load frame range if saved in config
                saved_start_frame = get_cfg("start_frame", default=None)
                if saved_start_frame is not None and self.spin_start_frame.isEnabled():
                    self.spin_start_frame.setValue(saved_start_frame)

                saved_end_frame = get_cfg("end_frame", default=None)
                if saved_end_frame is not None and self.spin_end_frame.isEnabled():
                    self.spin_end_frame.setValue(saved_end_frame)

            # === SYSTEM PERFORMANCE ===
            self.spin_resize.setValue(get_cfg("resize_factor", default=1.0))
            self.check_save_confidence.setChecked(
                get_cfg("save_confidence_metrics", default=True)
            )
            self.chk_use_cached_detections.setChecked(
                get_cfg("use_cached_detections", default=True)
            )
            self.chk_visualization_free.setChecked(
                get_cfg("visualization_free_mode", default=False)
            )

            # === DETECTION STRATEGY ===
            det_method = get_cfg("detection_method", default="background_subtraction")
            self.combo_detection_method.setCurrentIndex(
                0 if det_method == "background_subtraction" else 1
            )

            # === SIZE FILTERING ===
            self.chk_size_filtering.setChecked(
                get_cfg("enable_size_filtering", default=False)
            )
            self.spin_min_object_size.setValue(
                get_cfg("min_object_size_multiplier", default=0.3)
            )
            self.spin_max_object_size.setValue(
                get_cfg("max_object_size_multiplier", default=3.0)
            )

            # === IMAGE ENHANCEMENT ===
            self.slider_brightness.setValue(int(get_cfg("brightness", default=0.0)))
            self.slider_contrast.setValue(int(get_cfg("contrast", default=1.0) * 100))
            self.slider_gamma.setValue(int(get_cfg("gamma", default=1.0) * 100))
            self.chk_dark_on_light.setChecked(
                get_cfg("dark_on_light_background", default=True)
            )

            # === BACKGROUND SUBTRACTION ===
            self.spin_bg_prime.setValue(
                get_cfg("background_prime_frames", "bg_prime_frames", default=10)
            )
            self.chk_adaptive_bg.setChecked(
                get_cfg(
                    "enable_adaptive_background", "adaptive_background", default=True
                )
            )
            self.spin_bg_learning.setValue(
                get_cfg("background_learning_rate", default=0.001)
            )
            self.spin_threshold.setValue(
                get_cfg("subtraction_threshold", "threshold_value", default=50)
            )

            # === LIGHTING STABILIZATION ===
            self.chk_lighting_stab.setChecked(
                get_cfg(
                    "enable_lighting_stabilization",
                    "lighting_stabilization",
                    default=True,
                )
            )
            self.spin_lighting_smooth.setValue(
                get_cfg("lighting_smooth_factor", default=0.95)
            )
            self.spin_lighting_median.setValue(
                get_cfg("lighting_median_window", default=5)
            )

            # === MORPHOLOGY & NOISE ===
            self.spin_morph_size.setValue(get_cfg("morph_kernel_size", default=5))
            self.spin_min_contour.setValue(get_cfg("min_contour_area", default=50))
            self.spin_max_contour_multiplier.setValue(
                get_cfg("max_contour_multiplier", default=20)
            )

            # === ADVANCED SEPARATION ===
            self.chk_conservative_split.setChecked(
                get_cfg("enable_conservative_split", default=True)
            )
            self.spin_conservative_kernel.setValue(
                get_cfg("conservative_kernel_size", default=3)
            )
            self.spin_conservative_erode.setValue(
                get_cfg(
                    "conservative_erode_iterations",
                    "conservative_erode_iter",
                    default=1,
                )
            )
            self.spin_merge_threshold.setValue(
                get_cfg("merge_area_threshold", default=1000)
            )
            self.chk_additional_dilation.setChecked(
                get_cfg("enable_additional_dilation", default=False)
            )
            self.spin_dilation_kernel_size.setValue(
                get_cfg("dilation_kernel_size", default=3)
            )
            self.spin_dilation_iterations.setValue(
                get_cfg("dilation_iterations", default=2)
            )

            # === YOLO CONFIGURATION ===
            yolo_model = get_cfg("yolo_model_path", default="yolo26s-obb.pt")

            # Resolve model path (handles both relative and absolute)
            resolved_yolo_model = resolve_model_path(yolo_model)

            # Validate model exists if it's a relative path from preset
            if preset_mode and not os.path.isabs(yolo_model):
                if not os.path.exists(resolved_yolo_model):
                    logger.warning(
                        f"Preset references non-existent model: {yolo_model}\n"
                        f"Expected location: {resolved_yolo_model}"
                    )
                    QMessageBox.warning(
                        self,
                        "Model Not Found",
                        f"The preset references a model that doesn't exist:\n\n"
                        f"Model: {yolo_model}\n"
                        f"Expected at: {resolved_yolo_model}\n\n"
                        f"Please ensure the model is in your local models archive:\n"
                        f"{get_models_directory()}",
                    )

            # Refresh model combo from repository and apply selection
            self._refresh_yolo_model_combo(preferred_model_path=yolo_model)
            if self._get_selected_yolo_model_path() != make_model_path_relative(
                yolo_model
            ):
                # Fall back to resolved path when the configured relative path cannot be matched
                self._set_yolo_model_selection(resolved_yolo_model)

            self.spin_yolo_confidence.setValue(
                get_cfg("yolo_confidence_threshold", default=0.25)
            )
            self.spin_yolo_iou.setValue(get_cfg("yolo_iou_threshold", default=0.7))
            self.chk_use_custom_obb_iou.setChecked(True)
            yolo_cls = get_cfg("yolo_target_classes", default=None)
            if yolo_cls:
                self.line_yolo_classes.setText(",".join(map(str, yolo_cls)))

            compute_runtime_cfg = (
                str(
                    get_cfg(
                        "compute_runtime",
                        default=infer_compute_runtime_from_legacy(
                            yolo_device=str(get_cfg("yolo_device", default="auto")),
                            enable_tensorrt=bool(
                                get_cfg("enable_tensorrt", default=False)
                            ),
                            pose_runtime_flavor=str(
                                get_cfg("pose_runtime_flavor", default="auto")
                            ),
                        ),
                    )
                )
                .strip()
                .lower()
            )
            self._populate_compute_runtime_options(preferred=compute_runtime_cfg)
            self._on_runtime_context_changed()

            # TensorRT batch size is still configurable (runtime-derived usage).
            self.spin_tensorrt_batch.setValue(
                get_cfg("tensorrt_max_batch_size", default=16)
            )
            self.spin_tensorrt_batch.setEnabled(
                bool(
                    derive_detection_runtime_settings(self._selected_compute_runtime())[
                        "enable_tensorrt"
                    ]
                )
            )
            self.lbl_tensorrt_batch.setEnabled(
                bool(
                    derive_detection_runtime_settings(self._selected_compute_runtime())[
                        "enable_tensorrt"
                    ]
                )
            )

            # YOLO Batching settings
            self.chk_enable_yolo_batching.setChecked(
                get_cfg("enable_yolo_batching", default=True)
            )
            batch_mode = get_cfg("yolo_batch_size_mode", default="auto")
            self.combo_yolo_batch_mode.setCurrentIndex(0 if batch_mode == "auto" else 1)
            self.spin_yolo_batch_size.setValue(
                get_cfg("yolo_manual_batch_size", default=16)
            )
            # Re-apply runtime-derived constraints (e.g., TensorRT => manual batch mode).
            self._on_runtime_context_changed()

            # === CORE TRACKING ===
            self.spin_max_targets.setValue(get_cfg("max_targets", default=4))
            self.spin_max_dist.setValue(
                get_cfg(
                    "max_assignment_distance_multiplier",
                    "max_dist_multiplier",
                    default=1.5,
                )
            )
            self.spin_continuity_thresh.setValue(
                get_cfg(
                    "recovery_search_distance_multiplier",
                    "continuity_threshold_multiplier",
                    default=0.5,
                )
            )
            self.chk_enable_backward.setChecked(
                get_cfg("enable_backward_tracking", default=True)
            )

            # === KALMAN FILTER ===
            self.spin_kalman_noise.setValue(
                get_cfg("kalman_process_noise", "kalman_noise", default=0.03)
            )
            self.spin_kalman_meas.setValue(
                get_cfg("kalman_measurement_noise", "kalman_meas_noise", default=0.1)
            )
            self.spin_kalman_damping.setValue(
                get_cfg("kalman_velocity_damping", "kalman_damping", default=0.95)
            )
            self.spin_kalman_maturity_age.setValue(
                get_cfg("kalman_maturity_age", default=5)
            )
            self.spin_kalman_initial_velocity_retention.setValue(
                get_cfg("kalman_initial_velocity_retention", default=0.2)
            )
            self.spin_kalman_max_velocity.setValue(
                get_cfg("kalman_max_velocity_multiplier", default=2.0)
            )
            self.spin_kalman_longitudinal_noise.setValue(
                get_cfg("kalman_longitudinal_noise_multiplier", default=5.0)
            )
            self.spin_kalman_lateral_noise.setValue(
                get_cfg("kalman_lateral_noise_multiplier", default=0.1)
            )

            # === COST FUNCTION WEIGHTS ===
            self.spin_Wp.setValue(get_cfg("weight_position", "W_POSITION", default=1.0))
            self.spin_Wo.setValue(
                get_cfg("weight_orientation", "W_ORIENTATION", default=1.0)
            )
            self.spin_Wa.setValue(get_cfg("weight_area", "W_AREA", default=0.001))
            self.spin_Wasp.setValue(
                get_cfg("weight_aspect_ratio", "W_ASPECT", default=0.1)
            )
            self.chk_use_mahal.setChecked(
                get_cfg("use_mahalanobis_distance", "USE_MAHALANOBIS", default=True)
            )

            # === ASSIGNMENT ALGORITHM ===
            self.combo_assignment_method.setCurrentIndex(
                1 if get_cfg("enable_greedy_assignment", default=False) else 0
            )
            self.chk_spatial_optimization.setChecked(
                get_cfg("enable_spatial_optimization", default=False)
            )

            # === ORIENTATION & MOTION ===
            self.spin_velocity.setValue(get_cfg("velocity_threshold", default=5.0))
            self.chk_instant_flip.setChecked(
                get_cfg("enable_instant_flip", "instant_flip", default=True)
            )
            self.spin_max_orient.setValue(
                get_cfg(
                    "max_orientation_delta_stopped",
                    "max_orient_delta_stopped",
                    default=30.0,
                )
            )

            # === TRACK LIFECYCLE ===
            self.spin_lost_thresh.setValue(
                get_cfg("lost_frames_threshold", "lost_threshold_frames", default=10)
            )
            self.spin_min_respawn_distance.setValue(
                get_cfg("min_respawn_distance_multiplier", default=2.5)
            )
            self.spin_min_detections_to_start.setValue(
                get_cfg("min_detections_to_start", default=1)
            )
            self.spin_min_detect.setValue(
                get_cfg("min_detect_frames", "min_detect_counts", default=10)
            )
            self.spin_min_track.setValue(
                get_cfg("min_track_frames", "min_track_counts", default=10)
            )

            # === POST-PROCESSING ===
            self.enable_postprocessing.setChecked(
                get_cfg("enable_postprocessing", default=True)
            )
            self.spin_min_trajectory_length.setValue(
                get_cfg("min_trajectory_length", default=10)
            )
            self.spin_max_velocity_break.setValue(
                get_cfg("max_velocity_break", default=50.0)
            )
            self.spin_max_occlusion_gap.setValue(
                get_cfg("max_occlusion_gap", default=30)
            )
            self.spin_max_velocity_zscore.setValue(
                get_cfg("max_velocity_zscore", default=0.0)
            )
            self.spin_velocity_zscore_window.setValue(
                get_cfg("velocity_zscore_window", default=10)
            )
            self.spin_velocity_zscore_min_vel.setValue(
                get_cfg("velocity_zscore_min_velocity", default=2.0)
            )
            interp_method = get_cfg("interpolation_method", default="None")
            idx = self.combo_interpolation_method.findText(
                interp_method, Qt.MatchFixedString
            )
            if idx >= 0:
                self.combo_interpolation_method.setCurrentIndex(idx)
            self.spin_interpolation_max_gap.setValue(
                get_cfg("interpolation_max_gap", default=10)
            )
            self.chk_cleanup_temp_files.setChecked(
                get_cfg("cleanup_temp_files", default=True)
            )

            # === TRAJECTORY MERGING (Conservative Strategy) ===
            # Agreement distance and min overlap frames for conservative merging
            self.spin_merge_overlap_multiplier.setValue(
                get_cfg("merge_agreement_distance_multiplier", default=0.5)
            )
            self.spin_min_overlap_frames.setValue(
                get_cfg("min_overlap_frames", default=5)
            )

            # === VIDEO VISUALIZATION ===
            self.check_show_labels.setChecked(
                get_cfg("video_show_labels", default=True)
            )
            self.check_show_orientation.setChecked(
                get_cfg("video_show_orientation", default=True)
            )
            self.check_show_trails.setChecked(
                get_cfg("video_show_trails", default=False)
            )
            self.spin_trail_duration.setValue(
                get_cfg("video_trail_duration", default=1.0)
            )
            self.spin_marker_size.setValue(get_cfg("video_marker_size", default=0.3))
            self.spin_text_scale.setValue(get_cfg("video_text_scale", default=0.5))
            self.spin_arrow_length.setValue(get_cfg("video_arrow_length", default=0.7))
            self.check_video_show_pose.setChecked(
                get_cfg(
                    "video_show_pose",
                    default=bool(self.advanced_config.get("video_show_pose", True)),
                )
            )
            pose_color_mode = str(
                get_cfg(
                    "video_pose_color_mode",
                    default=self.advanced_config.get("video_pose_color_mode", "track"),
                )
            ).strip()
            self.combo_video_pose_color_mode.setCurrentIndex(
                0 if pose_color_mode == "track" else 1
            )
            self.spin_video_pose_point_radius.setValue(
                int(
                    get_cfg(
                        "video_pose_point_radius",
                        default=self.advanced_config.get("video_pose_point_radius", 3),
                    )
                )
            )
            self.spin_video_pose_point_thickness.setValue(
                int(
                    get_cfg(
                        "video_pose_point_thickness",
                        default=self.advanced_config.get(
                            "video_pose_point_thickness", -1
                        ),
                    )
                )
            )
            self.spin_video_pose_line_thickness.setValue(
                int(
                    get_cfg(
                        "video_pose_line_thickness",
                        default=self.advanced_config.get(
                            "video_pose_line_thickness", 2
                        ),
                    )
                )
            )
            pose_color = get_cfg(
                "video_pose_color",
                default=self.advanced_config.get("video_pose_color", [255, 255, 255]),
            )
            if isinstance(pose_color, (list, tuple)) and len(pose_color) == 3:
                self._video_pose_color = tuple(
                    int(max(0, min(255, float(v)))) for v in pose_color
                )
                self._update_video_pose_color_button()
            self._sync_video_pose_overlay_controls()

            # === REAL-TIME ANALYTICS ===
            self.enable_histograms.setChecked(
                get_cfg("enable_histograms", default=False)
            )
            self.spin_histogram_history.setValue(
                get_cfg("histogram_history_frames", default=300)
            )

            # === VISUALIZATION OVERLAYS ===
            self.chk_show_circles.setChecked(
                get_cfg("show_track_markers", "show_circles", default=True)
            )
            self.chk_show_orientation.setChecked(
                get_cfg("show_orientation_lines", "show_orientation", default=True)
            )
            self.chk_show_trajectories.setChecked(
                get_cfg("show_trajectory_trails", "show_trajectories", default=True)
            )
            self.chk_show_labels.setChecked(
                get_cfg("show_id_labels", "show_labels", default=True)
            )
            self.chk_show_state.setChecked(
                get_cfg("show_state_text", "show_state", default=True)
            )
            self.chk_show_kalman_uncertainty.setChecked(
                get_cfg("show_kalman_uncertainty", default=False)
            )
            self.chk_show_fg.setChecked(
                get_cfg("show_foreground_mask", "show_fg", default=True)
            )
            self.chk_show_bg.setChecked(
                get_cfg("show_background_model", "show_bg", default=True)
            )
            self.chk_show_yolo_obb.setChecked(get_cfg("show_yolo_obb", default=False))
            self.spin_traj_hist.setValue(
                get_cfg("trajectory_history_seconds", "traj_history", default=5)
            )
            self.chk_debug_logging.setChecked(get_cfg("debug_logging", default=False))
            self.slider_zoom.setValue(int(get_cfg("zoom_factor", default=1.0) * 100))

            # === DATASET GENERATION ===
            self.chk_enable_dataset_gen.setChecked(
                get_cfg("enable_dataset_generation", default=False)
            )
            self.line_dataset_name.setText(get_cfg("dataset_name", default=""))
            self.line_dataset_class_name.setText(
                get_cfg("dataset_class_name", default="object")
            )
            # Skip output directory in preset mode
            if not preset_mode:
                self.line_dataset_output.setText(
                    get_cfg("dataset_output_dir", default="")
                )
            self.spin_dataset_max_frames.setValue(
                get_cfg("dataset_max_frames", default=100)
            )
            self.spin_dataset_conf_threshold.setValue(
                get_cfg(
                    "dataset_confidence_threshold",
                    "dataset_conf_threshold",
                    default=0.5,
                )
            )
            # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
            self.spin_dataset_diversity_window.setValue(
                get_cfg("dataset_diversity_window", default=30)
            )
            self.chk_dataset_include_context.setChecked(
                get_cfg("dataset_include_context", default=True)
            )
            self.chk_dataset_probabilistic.setChecked(
                get_cfg("dataset_probabilistic_sampling", default=True)
            )
            self.chk_metric_low_confidence.setChecked(
                get_cfg("metric_low_confidence", default=True)
            )
            self.chk_metric_count_mismatch.setChecked(
                get_cfg("metric_count_mismatch", default=True)
            )
            self.chk_metric_high_assignment_cost.setChecked(
                get_cfg("metric_high_assignment_cost", default=True)
            )
            self.chk_metric_track_loss.setChecked(
                get_cfg("metric_track_loss", default=True)
            )
            self.chk_metric_high_uncertainty.setChecked(
                get_cfg("metric_high_uncertainty", default=False)
            )

            # === INDIVIDUAL ANALYSIS ===
            pipeline_enabled = get_cfg(
                "enable_individual_pipeline",
                "enable_identity_analysis",
                default=False,
            )
            self.chk_enable_individual_analysis.setChecked(bool(pipeline_enabled))
            method_map = {
                "none_disabled": 0,
                "color_tags_yolo": 1,
                "apriltags": 2,
                "custom": 3,
            }
            identity_method = get_cfg("identity_method", default="none_disabled")
            self.combo_identity_method.setCurrentIndex(
                method_map.get(identity_method, 0)
            )
            self.spin_identity_crop_multiplier.setValue(
                get_cfg("identity_crop_size_multiplier", default=3.0)
            )
            self.spin_identity_crop_min.setValue(
                get_cfg("identity_crop_min_size", default=64)
            )
            self.spin_identity_crop_max.setValue(
                get_cfg("identity_crop_max_size", default=256)
            )
            # Skip model path in preset mode
            if not preset_mode and not self.line_color_tag_model.text().strip():
                color_tag_model = get_cfg("color_tag_model_path", default="")
                if color_tag_model:
                    # Resolve model path
                    resolved_path = resolve_model_path(color_tag_model)
                    self.line_color_tag_model.setText(resolved_path)
            elif preset_mode:
                # For presets, resolve but also validate
                color_tag_model = get_cfg("color_tag_model_path", default="")
                if color_tag_model:
                    resolved_path = resolve_model_path(color_tag_model)

                    # Validate if it's a relative path
                    if not os.path.isabs(color_tag_model) and not os.path.exists(
                        resolved_path
                    ):
                        logger.warning(
                            f"Preset references non-existent color tag model: {color_tag_model}\n"
                            f"Expected location: {resolved_path}"
                        )
                        # Don't show dialog for color tag model since it's optional
                        # Just log the warning
                    elif not self.line_color_tag_model.text().strip():
                        self.line_color_tag_model.setText(resolved_path)
            self.spin_color_tag_conf.setValue(
                get_cfg("color_tag_confidence", default=0.5)
            )
            apriltag_family = get_cfg("apriltag_family", default="tag36h11")
            families = [
                "tag36h11",
                "tag25h9",
                "tag16h5",
                "tagCircle21h7",
                "tagStandard41h12",
            ]
            if apriltag_family in families:
                self.combo_apriltag_family.setCurrentIndex(
                    families.index(apriltag_family)
                )
            self.spin_apriltag_decimate.setValue(
                get_cfg("apriltag_decimate", default=1.0)
            )

            self.chk_enable_pose_extractor.setChecked(
                get_cfg("enable_pose_extractor", default=False)
            )
            pose_backend = (
                str(get_cfg("pose_model_type", default="yolo")).strip().upper()
            )
            pose_backend_idx = self.combo_pose_model_type.findText(pose_backend)
            if pose_backend_idx >= 0:
                self.combo_pose_model_type.setCurrentIndex(pose_backend_idx)
            yolo_pose_model = str(get_cfg("pose_yolo_model_dir", default="")).strip()
            sleap_pose_model = str(get_cfg("pose_sleap_model_dir", default="")).strip()
            legacy_pose_model = str(get_cfg("pose_model_dir", default="")).strip()
            if not yolo_pose_model and pose_backend.lower() == "yolo":
                yolo_pose_model = legacy_pose_model
            if not sleap_pose_model and pose_backend.lower() == "sleap":
                sleap_pose_model = legacy_pose_model
            self._set_pose_model_path_for_backend(yolo_pose_model, backend="yolo")
            self._set_pose_model_path_for_backend(sleap_pose_model, backend="sleap")
            self._set_pose_model_path_for_backend(
                self._pose_model_path_for_backend(
                    self.combo_pose_model_type.currentText().strip().lower()
                ),
                backend=self.combo_pose_model_type.currentText().strip().lower(),
                update_line=True,
            )
            pose_runtime_flavor = derive_pose_runtime_settings(
                self._selected_compute_runtime(),
                backend_family=self.combo_pose_model_type.currentText().strip().lower(),
            )["pose_runtime_flavor"]
            self._populate_pose_runtime_flavor_options(
                backend=self.combo_pose_model_type.currentText().strip().lower(),
                preferred=pose_runtime_flavor,
            )
            self.spin_pose_min_kpt_conf_valid.setValue(
                get_cfg("pose_min_kpt_conf_valid", default=0.2)
            )
            self.line_pose_skeleton_file.setText(
                get_cfg("pose_skeleton_file", default="")
            )
            self._refresh_pose_direction_keypoint_lists()
            ignore_kpts = get_cfg("pose_ignore_keypoints", default=[])
            self._set_pose_group_selection(self.list_pose_ignore_keypoints, ignore_kpts)
            ant_kpts = get_cfg("pose_direction_anterior_keypoints", default=[])
            self._set_pose_group_selection(self.list_pose_direction_anterior, ant_kpts)
            post_kpts = get_cfg("pose_direction_posterior_keypoints", default=[])
            self._set_pose_group_selection(
                self.list_pose_direction_posterior, post_kpts
            )
            self._apply_pose_keypoint_selection_constraints("ignore")
            self.advanced_config["pose_sleap_env"] = str(
                get_cfg("pose_sleap_env", default="sleap")
            )
            self._refresh_pose_sleap_envs()
            if hasattr(self, "chk_sleap_experimental_features"):
                self.chk_sleap_experimental_features.setChecked(
                    get_cfg("pose_sleap_experimental_features", default=False)
                )
            shared_pose_batch = int(
                get_cfg(
                    "pose_batch_size",
                    default=get_cfg(
                        "pose_yolo_batch",
                        default=get_cfg("pose_sleap_batch", default=4),
                    ),
                )
            )
            self.spin_pose_batch.setValue(shared_pose_batch)

            # === REAL-TIME INDIVIDUAL DATASET ===
            self.chk_enable_individual_dataset.setChecked(
                get_cfg(
                    "enable_individual_image_save",
                    "enable_individual_dataset",
                    default=False,
                )
            )
            self.line_individual_dataset_name.setText(
                get_cfg("individual_dataset_name", default="individual_dataset")
            )
            # Skip output directory in preset mode
            if not preset_mode:
                self.line_individual_output.setText(
                    get_cfg("individual_dataset_output_dir", default="")
                )
            format_text = get_cfg("individual_output_format", default="png").upper()
            format_idx = self.combo_individual_format.findText(format_text)
            if format_idx >= 0:
                self.combo_individual_format.setCurrentIndex(format_idx)
            self.spin_individual_interval.setValue(
                get_cfg("individual_save_interval", default=1)
            )
            self.chk_individual_interpolate.setChecked(
                get_cfg("individual_interpolate_occlusions", default=True)
            )
            self.spin_individual_padding.setValue(
                get_cfg("individual_crop_padding", default=0.1)
            )
            # Load background color
            bg_color = get_cfg("individual_background_color", default=[0, 0, 0])
            if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
                self._background_color = tuple(bg_color)
            self._update_background_color_button()
            self._sync_individual_analysis_mode_ui()

            # === ROI ===
            self.roi_shapes = cfg.get("roi_shapes", [])
            if self.roi_shapes:
                # Regenerate the combined mask from loaded shapes
                # Need to get video frame dimensions first
                video_path = cfg.get("file_path", "")
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            fh, fw = frame.shape[:2]
                            self._generate_combined_roi_mask(fh, fw)
                            num_shapes = len(self.roi_shapes)
                            shape_summary = ", ".join(
                                [s["type"] for s in self.roi_shapes]
                            )
                            self.roi_status_label.setText(
                                f"Loaded ROI: {num_shapes} shape(s) ({shape_summary})"
                            )
                            self.btn_undo_roi.setEnabled(True)
                            logger.info(f"Loaded {num_shapes} ROI shapes from config")
                        cap.release()
                else:
                    # Video not available yet, just store shapes for later
                    logger.info(
                        f"Loaded {len(self.roi_shapes)} ROI shapes (mask will be generated when video loads)"
                    )
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")

    def save_config(
        self: object,
        preset_mode: object = False,
        preset_path: object = None,
        preset_name: object = None,
        preset_description: object = None,
        prompt_if_exists: bool = True,
    ) -> object:
        """Save current configuration to JSON file.

        Args:
            preset_mode: If True, skip video paths, device settings, and ROI data (for organism presets)
            preset_path: If provided, save directly to this path without prompting
            preset_name: Name for the preset (only used in preset_mode)
            preset_description: Description for the preset (only used in preset_mode)
            prompt_if_exists: If False, overwrite default config path without interactive replace dialog.

        Returns:
            bool: True if config was saved successfully, False if cancelled or failed
        """
        yolo_path = self._get_selected_yolo_model_path()
        yolo_cls = (
            [int(x.strip()) for x in self.line_yolo_classes.text().split(",")]
            if self.line_yolo_classes.text().strip()
            else None
        )

        cfg = {}

        # === PRESET METADATA ===
        # Add name and description when saving as preset
        if preset_mode:
            cfg.update(
                {
                    "preset_name": preset_name or "Custom",
                    "description": preset_description or "User-defined custom preset",
                }
            )

        # === FILE MANAGEMENT ===
        # Skip file paths when saving as preset
        if not preset_mode:
            cfg.update(
                {
                    "file_path": self.file_line.text(),
                    "csv_path": self.csv_line.text(),
                    "video_output_enabled": self.check_video_output.isChecked(),
                    "video_output_path": self.video_out_line.text(),
                    # Video-specific reference parameters
                    "fps": self.spin_fps.value(),
                    "reference_body_size": self.spin_reference_body_size.value(),
                    # Frame range
                    "start_frame": (
                        self.spin_start_frame.value()
                        if self.spin_start_frame.isEnabled()
                        else 0
                    ),
                    "end_frame": (
                        self.spin_end_frame.value()
                        if self.spin_end_frame.isEnabled()
                        else None
                    ),
                }
            )

        compute_runtime = self._selected_compute_runtime()
        pose_runtime_derived = derive_pose_runtime_settings(
            compute_runtime,
            backend_family=self.combo_pose_model_type.currentText().strip().lower(),
        )

        cfg.update(
            {
                # === SYSTEM PERFORMANCE ===
                "resize_factor": self.spin_resize.value(),
                "save_confidence_metrics": self.check_save_confidence.isChecked(),
                "use_cached_detections": self.chk_use_cached_detections.isChecked(),
                "visualization_free_mode": self.chk_visualization_free.isChecked(),
                # === DETECTION STRATEGY ===
                "detection_method": (
                    "background_subtraction"
                    if self.combo_detection_method.currentIndex() == 0
                    else "yolo_obb"
                ),
                # === SIZE FILTERING ===
                "enable_size_filtering": self.chk_size_filtering.isChecked(),
                "min_object_size_multiplier": self.spin_min_object_size.value(),
                "max_object_size_multiplier": self.spin_max_object_size.value(),
                # === IMAGE ENHANCEMENT ===
                "brightness": self.slider_brightness.value(),
                "contrast": self.slider_contrast.value() / 100.0,
                "gamma": self.slider_gamma.value() / 100.0,
                "dark_on_light_background": self.chk_dark_on_light.isChecked(),
                # === BACKGROUND SUBTRACTION ===
                "background_prime_frames": self.spin_bg_prime.value(),
                "enable_adaptive_background": self.chk_adaptive_bg.isChecked(),
                "background_learning_rate": self.spin_bg_learning.value(),
                "subtraction_threshold": self.spin_threshold.value(),
                # === LIGHTING STABILIZATION ===
                "enable_lighting_stabilization": self.chk_lighting_stab.isChecked(),
                "lighting_smooth_factor": self.spin_lighting_smooth.value(),
                "lighting_median_window": self.spin_lighting_median.value(),
                # === MORPHOLOGY & NOISE ===
                "morph_kernel_size": self.spin_morph_size.value(),
                "min_contour_area": self.spin_min_contour.value(),
                "max_contour_multiplier": self.spin_max_contour_multiplier.value(),
                # === ADVANCED SEPARATION ===
                "enable_conservative_split": self.chk_conservative_split.isChecked(),
                "conservative_kernel_size": self.spin_conservative_kernel.value(),
                "conservative_erode_iterations": self.spin_conservative_erode.value(),
                "merge_area_threshold": self.spin_merge_threshold.value(),
                "enable_additional_dilation": self.chk_additional_dilation.isChecked(),
                "dilation_kernel_size": self.spin_dilation_kernel_size.value(),
                "dilation_iterations": self.spin_dilation_iterations.value(),
                # === YOLO CONFIGURATION ===
                # Store relative path if model is in archive, otherwise absolute
                "yolo_model_path": make_model_path_relative(yolo_path),
                "yolo_confidence_threshold": self.spin_yolo_confidence.value(),
                "yolo_iou_threshold": self.spin_yolo_iou.value(),
                "use_custom_obb_iou_filtering": self.chk_use_custom_obb_iou.isChecked(),
                "yolo_target_classes": yolo_cls,
            }
        )
        yolo_meta = get_yolo_model_metadata(yolo_path) or {}
        if yolo_meta:
            cfg["yolo_model_size"] = yolo_meta.get("size", "")
            cfg["yolo_model_species"] = yolo_meta.get("species", "")
            cfg["yolo_model_info"] = yolo_meta.get("model_info", "")
            cfg["yolo_model_added_at"] = yolo_meta.get("added_at", "")

        # === COMPUTE RUNTIME ===
        runtime_detection = derive_detection_runtime_settings(compute_runtime)
        cfg["compute_runtime"] = compute_runtime
        # Keep legacy fields writable for backward compatibility.
        if not preset_mode:
            cfg["yolo_device"] = runtime_detection["yolo_device"]

        cfg.update(
            {
                # TensorRT
                "enable_tensorrt": runtime_detection["enable_tensorrt"],
                "tensorrt_max_batch_size": (
                    self.spin_yolo_batch_size.value()
                    if self._runtime_requires_fixed_yolo_batch(compute_runtime)
                    else self.spin_tensorrt_batch.value()
                ),
                # YOLO Batching
                "enable_yolo_batching": self.chk_enable_yolo_batching.isChecked(),
                "yolo_batch_size_mode": (
                    "auto"
                    if self.combo_yolo_batch_mode.currentIndex() == 0
                    else "manual"
                ),
                "yolo_manual_batch_size": self.spin_yolo_batch_size.value(),
                # === CORE TRACKING ===
                "max_targets": self.spin_max_targets.value(),
                "max_assignment_distance_multiplier": self.spin_max_dist.value(),
                "recovery_search_distance_multiplier": self.spin_continuity_thresh.value(),
                "enable_backward_tracking": self.chk_enable_backward.isChecked(),
                # === KALMAN FILTER ===
                "kalman_process_noise": self.spin_kalman_noise.value(),
                "kalman_measurement_noise": self.spin_kalman_meas.value(),
                "kalman_velocity_damping": self.spin_kalman_damping.value(),
                "kalman_maturity_age": self.spin_kalman_maturity_age.value(),
                "kalman_initial_velocity_retention": self.spin_kalman_initial_velocity_retention.value(),
                "kalman_max_velocity_multiplier": self.spin_kalman_max_velocity.value(),
                "kalman_longitudinal_noise_multiplier": self.spin_kalman_longitudinal_noise.value(),
                "kalman_lateral_noise_multiplier": self.spin_kalman_lateral_noise.value(),
                # === COST FUNCTION WEIGHTS ===
                "weight_position": self.spin_Wp.value(),
                "weight_orientation": self.spin_Wo.value(),
                "weight_area": self.spin_Wa.value(),
                "weight_aspect_ratio": self.spin_Wasp.value(),
                "use_mahalanobis_distance": self.chk_use_mahal.isChecked(),
                # === ASSIGNMENT ALGORITHM ===
                "enable_greedy_assignment": self.combo_assignment_method.currentIndex()
                == 1,
                "enable_spatial_optimization": self.chk_spatial_optimization.isChecked(),
                # === ORIENTATION & MOTION ===
                "velocity_threshold": self.spin_velocity.value(),
                "enable_instant_flip": self.chk_instant_flip.isChecked(),
                "max_orientation_delta_stopped": self.spin_max_orient.value(),
                # === TRACK LIFECYCLE ===
                "lost_frames_threshold": self.spin_lost_thresh.value(),
                "min_respawn_distance_multiplier": self.spin_min_respawn_distance.value(),
                "min_detections_to_start": self.spin_min_detections_to_start.value(),
                "min_detect_frames": self.spin_min_detect.value(),
                "min_track_frames": self.spin_min_track.value(),
                # === POST-PROCESSING ===
                "enable_postprocessing": self.enable_postprocessing.isChecked(),
                "min_trajectory_length": self.spin_min_trajectory_length.value(),
                "max_velocity_break": self.spin_max_velocity_break.value(),
                "max_occlusion_gap": self.spin_max_occlusion_gap.value(),
                "max_velocity_zscore": self.spin_max_velocity_zscore.value(),
                "velocity_zscore_window": self.spin_velocity_zscore_window.value(),
                "velocity_zscore_min_velocity": self.spin_velocity_zscore_min_vel.value(),
                "interpolation_method": self.combo_interpolation_method.currentText(),
                "interpolation_max_gap": self.spin_interpolation_max_gap.value(),
                "cleanup_temp_files": self.chk_cleanup_temp_files.isChecked(),
                # === TRAJECTORY MERGING (Conservative Strategy) ===
                # Agreement distance and min overlap frames for conservative merging
                "merge_agreement_distance_multiplier": self.spin_merge_overlap_multiplier.value(),
                "min_overlap_frames": self.spin_min_overlap_frames.value(),
                # === VIDEO VISUALIZATION ===
                "video_show_labels": self.check_show_labels.isChecked(),
                "video_show_orientation": self.check_show_orientation.isChecked(),
                "video_show_trails": self.check_show_trails.isChecked(),
                "video_trail_duration": self.spin_trail_duration.value(),
                "video_marker_size": self.spin_marker_size.value(),
                "video_text_scale": self.spin_text_scale.value(),
                "video_arrow_length": self.spin_arrow_length.value(),
                "video_show_pose": self.check_video_show_pose.isChecked(),
                "video_pose_color_mode": (
                    "track"
                    if self.combo_video_pose_color_mode.currentIndex() == 0
                    else "fixed"
                ),
                "video_pose_color": [
                    int(self._video_pose_color[0]),
                    int(self._video_pose_color[1]),
                    int(self._video_pose_color[2]),
                ],
                "video_pose_point_radius": self.spin_video_pose_point_radius.value(),
                "video_pose_point_thickness": self.spin_video_pose_point_thickness.value(),
                "video_pose_line_thickness": self.spin_video_pose_line_thickness.value(),
                # === REAL-TIME ANALYTICS ===
                "enable_histograms": self.enable_histograms.isChecked(),
                "histogram_history_frames": self.spin_histogram_history.value(),
                # === VISUALIZATION OVERLAYS ===
                "show_track_markers": self.chk_show_circles.isChecked(),
                "show_orientation_lines": self.chk_show_orientation.isChecked(),
                "show_trajectory_trails": self.chk_show_trajectories.isChecked(),
                "show_id_labels": self.chk_show_labels.isChecked(),
                "show_state_text": self.chk_show_state.isChecked(),
                "show_kalman_uncertainty": self.chk_show_kalman_uncertainty.isChecked(),
                "show_foreground_mask": self.chk_show_fg.isChecked(),
                "show_background_model": self.chk_show_bg.isChecked(),
                "show_yolo_obb": self.chk_show_yolo_obb.isChecked(),
                "trajectory_history_seconds": self.spin_traj_hist.value(),
                "debug_logging": self.chk_debug_logging.isChecked(),
                "zoom_factor": self.slider_zoom.value() / 100.0,
            }
        )

        # === ROI ===
        # Skip ROI when saving as preset
        if not preset_mode:
            cfg["roi_shapes"] = self.roi_shapes

        cfg.update(
            {
                # === DATASET GENERATION ===
                "enable_dataset_generation": self.chk_enable_dataset_gen.isChecked(),
                "dataset_name": self.line_dataset_name.text(),
                "dataset_class_name": self.line_dataset_class_name.text(),
                "dataset_max_frames": self.spin_dataset_max_frames.value(),
            }
        )

        # Dataset output directory (device-specific)
        if not preset_mode:
            cfg["dataset_output_dir"] = self.line_dataset_output.text()

        cfg.update(
            {
                "dataset_confidence_threshold": self.spin_dataset_conf_threshold.value(),
                # Note: dataset YOLO conf/IOU now in advanced_config.json, not per-video config
                "dataset_diversity_window": self.spin_dataset_diversity_window.value(),
                "dataset_include_context": self.chk_dataset_include_context.isChecked(),
                "dataset_probabilistic_sampling": self.chk_dataset_probabilistic.isChecked(),
                "metric_low_confidence": self.chk_metric_low_confidence.isChecked(),
                "metric_count_mismatch": self.chk_metric_count_mismatch.isChecked(),
                "metric_high_assignment_cost": self.chk_metric_high_assignment_cost.isChecked(),
                "metric_track_loss": self.chk_metric_track_loss.isChecked(),
                "metric_high_uncertainty": self.chk_metric_high_uncertainty.isChecked(),
                # === INDIVIDUAL ANALYSIS ===
                "enable_identity_analysis": self._is_individual_pipeline_enabled(),
                "enable_individual_pipeline": self._is_individual_pipeline_enabled(),
                "identity_method": self.combo_identity_method.currentText()
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", ""),
                "identity_crop_size_multiplier": self.spin_identity_crop_multiplier.value(),
                "identity_crop_min_size": self.spin_identity_crop_min.value(),
                "identity_crop_max_size": self.spin_identity_crop_max.value(),
                "color_tag_confidence": self.spin_color_tag_conf.value(),
            }
        )

        # Model paths - save differently for configs vs presets
        color_tag_path = self.line_color_tag_model.text()
        if preset_mode:
            # For presets: always try to make relative if in archive
            cfg["color_tag_model_path"] = (
                make_model_path_relative(color_tag_path) if color_tag_path else ""
            )
        else:
            # For configs: store relative if in archive, absolute otherwise
            cfg["color_tag_model_path"] = (
                make_model_path_relative(color_tag_path) if color_tag_path else ""
            )

        cfg.update(
            {
                "apriltag_family": self.combo_apriltag_family.currentText(),
                "apriltag_decimate": self.spin_apriltag_decimate.value(),
                "enable_pose_extractor": self.chk_enable_pose_extractor.isChecked(),
                "pose_model_type": self.combo_pose_model_type.currentText()
                .strip()
                .lower(),
                "pose_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend(
                        self.combo_pose_model_type.currentText().strip().lower()
                    )
                ),
                "pose_yolo_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend("yolo")
                ),
                "pose_sleap_model_dir": make_pose_model_path_relative(
                    self._pose_model_path_for_backend("sleap")
                ),
                "pose_runtime_flavor": pose_runtime_derived["pose_runtime_flavor"],
                "pose_exported_model_path": "",
                "pose_min_kpt_conf_valid": self.spin_pose_min_kpt_conf_valid.value(),
                "pose_skeleton_file": self.line_pose_skeleton_file.text().strip(),
                "pose_ignore_keypoints": self._parse_pose_ignore_keypoints(),
                "pose_direction_anterior_keypoints": self._parse_pose_direction_anterior_keypoints(),
                "pose_direction_posterior_keypoints": self._parse_pose_direction_posterior_keypoints(),
                "pose_batch_size": self.spin_pose_batch.value(),
                "pose_yolo_batch": self.spin_pose_batch.value(),
                "pose_sleap_env": self._selected_pose_sleap_env(),
                "pose_sleap_device": pose_runtime_derived["pose_sleap_device"],
                "pose_sleap_batch": self.spin_pose_batch.value(),
                "pose_sleap_max_instances": 1,
                "pose_sleap_experimental_features": self._sleap_experimental_features_enabled(),
                # === REAL-TIME INDIVIDUAL DATASET ===
                "enable_individual_dataset": self._is_individual_image_save_enabled(),
                "enable_individual_image_save": self._is_individual_image_save_enabled(),
                "individual_dataset_name": self.line_individual_dataset_name.text().strip()
                or "individual_dataset",
                "individual_output_format": self.combo_individual_format.currentText().lower(),
            }
        )

        # Individual output directory (device-specific)
        if not preset_mode:
            cfg["individual_dataset_output_dir"] = self.line_individual_output.text()

        cfg.update(
            {
                "individual_save_interval": self.spin_individual_interval.value(),
                "individual_interpolate_occlusions": self.chk_individual_interpolate.isChecked(),
                "individual_crop_padding": self.spin_individual_padding.value(),
                "individual_background_color": [
                    int(c) for c in self._background_color
                ],  # Ensure JSON serializable
            }
        )

        # If preset mode with path provided, save directly
        if preset_mode and preset_path:
            try:
                import tempfile

                os.makedirs(os.path.dirname(preset_path), exist_ok=True)
                # Write to temp file first, then rename (atomic on most filesystems)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=os.path.dirname(preset_path),
                    delete=False,
                    suffix=".tmp",
                ) as tmp:
                    json.dump(cfg, tmp, indent=2)
                    tmp_path = tmp.name
                os.replace(tmp_path, preset_path)  # Atomic rename
                logger.info(f"Saved preset to {preset_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to save preset: {e}")
                QMessageBox.critical(self, "Save Error", f"Failed to save preset:\n{e}")
                return False

        # Determine save path: video-based if video selected, otherwise ask user
        video_path = self.file_line.text()
        if video_path:
            default_path = get_video_config_path(video_path)
        else:
            default_path = CONFIG_FILENAME

        config_path = None

        # If default path exists, ask user whether to replace or save elsewhere
        if default_path and os.path.exists(default_path) and prompt_if_exists:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Configuration File Exists")
            msg.setText(
                f"A configuration file already exists:\n{os.path.basename(default_path)}"
            )
            msg.setInformativeText(
                "Do you want to replace it or save to a different location?"
            )

            replace_btn = msg.addButton("Replace Existing", QMessageBox.AcceptRole)
            save_as_btn = msg.addButton("Save As...", QMessageBox.ActionRole)
            cancel_btn = msg.addButton(QMessageBox.Cancel)
            msg.setDefaultButton(replace_btn)

            result = msg.exec()
            clicked = msg.clickedButton()

            if clicked == replace_btn:
                config_path = default_path
            elif clicked == save_as_btn:
                config_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration As", default_path, "JSON Files (*.json)"
                )
            else:
                # User clicked Cancel or closed dialog - return False to cancel operation
                return False
        else:
            # No existing file, save directly to default path if available
            if default_path:
                config_path = default_path
            else:
                # No video selected, ask user where to save
                config_path, _ = QFileDialog.getSaveFileName(
                    self, "Save Configuration", CONFIG_FILENAME, "JSON Files (*.json)"
                )

        if config_path:
            try:
                import tempfile

                # Write to temp file first, then rename (atomic on most filesystems)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    dir=os.path.dirname(config_path),
                    delete=False,
                    suffix=".tmp",
                ) as tmp:
                    json.dump(cfg, tmp, indent=2)
                    tmp_path = tmp.name
                os.replace(tmp_path, config_path)  # Atomic rename
                logger.info(
                    f"Configuration saved to {config_path} (including ROI shapes)"
                )
                return True
            except Exception as e:
                logger.warning(f"Failed to save configuration: {e}")
                # Clean up temp file if save failed
                try:
                    if "tmp_path" in locals() and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                return False
        else:
            # User cancelled file dialog
            return False

    def _setup_session_logging(self, video_path, backward_mode=False):
        """Set up comprehensive logging for the entire tracking session."""
        from datetime import datetime
        from pathlib import Path

        # Close existing session log if any
        self._cleanup_session_logging()

        # Only set up logging if not already set up
        if self.session_log_handler is not None:
            logger.info("=" * 80)
            logger.info("Session log already active, continuing...")
            logger.info("=" * 80)
            return

        # Create log file next to the video
        video_path = Path(video_path)
        log_dir = video_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{video_path.stem}_tracking_{timestamp}.log"
        log_path = log_dir / log_filename

        # Create file handler for session
        self.session_log_handler = logging.FileHandler(log_path, mode="w")
        self.session_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.session_log_handler.setFormatter(formatter)

        # Add to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self.session_log_handler)

        logger.info("=" * 80)
        logger.info("TRACKING SESSION STARTED")
        logger.info(f"Session log: {log_path}")
        logger.info(f"Video: {video_path}")
        logger.info("=" * 80)

    def _cleanup_session_logging(self):
        """Remove session log handler from root logger."""
        if self.session_log_handler:
            logger.info("=" * 80)
            logger.info("Tracking session completed")
            logger.info("=" * 80)

            root_logger = logging.getLogger()
            root_logger.removeHandler(self.session_log_handler)
            self.session_log_handler.close()
            self.session_log_handler = None

    def _generate_training_dataset(self, override_csv_path=None):
        """Generate training dataset from tracking results for active learning."""
        try:
            if self._stop_all_requested:
                return
            logger.info("Starting training dataset generation...")

            # Prevent launching overlapping dataset threads; this can lead to
            # QThread destruction while still running if references are replaced.
            if self.dataset_worker is not None and self.dataset_worker.isRunning():
                logger.warning(
                    "Dataset generation already in progress; skipping duplicate request."
                )
                return
            if self.dataset_worker is not None and not self.dataset_worker.isRunning():
                self.dataset_worker.deleteLater()
                self.dataset_worker = None

            # Validate parameters
            dataset_name = self.line_dataset_name.text().strip()
            output_dir = self.line_dataset_output.text().strip()

            if not dataset_name:
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Please enter a dataset name."
                )
                return

            if not output_dir:
                QMessageBox.warning(
                    self,
                    "Dataset Generation Error",
                    "Please select an output directory.",
                )
                return

            if not os.path.exists(output_dir):
                QMessageBox.warning(
                    self,
                    "Dataset Generation Error",
                    f"Output directory does not exist: {output_dir}",
                )
                return

            video_path = self.file_line.text()
            # Use override path if provided (e.g. valid processed CSV), otherwise fallback to UI field
            csv_path = override_csv_path if override_csv_path else self.csv_line.text()

            if not video_path or not os.path.exists(video_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Source video file not found."
                )
                return

            if not csv_path or not os.path.exists(csv_path):
                QMessageBox.warning(
                    self, "Dataset Generation Error", "Tracking CSV file not found."
                )
                return

            # Get parameters
            params = self.get_parameters_dict()
            max_frames = self.spin_dataset_max_frames.value()
            diversity_window = self.spin_dataset_diversity_window.value()
            include_context = self.chk_dataset_include_context.isChecked()
            probabilistic = self.chk_dataset_probabilistic.isChecked()

            # Get class name
            class_name = self.line_dataset_class_name.text().strip()
            if not class_name:
                class_name = "object"

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Preparing dataset generation...")

            # Create and start dataset generation worker thread
            self.dataset_worker = DatasetGenerationWorker(
                video_path=video_path,
                csv_path=csv_path,
                detection_cache_path=self.current_detection_cache_path,
                output_dir=output_dir,
                dataset_name=dataset_name,
                class_name=class_name,
                params=params,
                max_frames=max_frames,
                diversity_window=diversity_window,
                include_context=include_context,
                probabilistic=probabilistic,
            )
            self.dataset_worker.progress_signal.connect(self.on_dataset_progress)
            self.dataset_worker.finished_signal.connect(self.on_dataset_finished)
            self.dataset_worker.error_signal.connect(self.on_dataset_error)
            self.dataset_worker.finished.connect(
                self._on_dataset_worker_thread_finished
            )
            self.dataset_worker.start()

        except Exception as e:
            logger.error(f"Dataset generation failed: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Dataset Generation Error",
                f"Failed to generate dataset:\n{str(e)}",
            )

    def on_dataset_progress(self: object, value: object, message: object) -> object:
        """Update progress bar during dataset generation."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            return
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def on_dataset_finished(
        self: object, dataset_dir: object, num_frames: object
    ) -> object:
        """Handle dataset generation completion."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._refresh_progress_visibility()
            return
        self._refresh_progress_visibility()

        logger.info(f"Dataset generation complete: {dataset_dir}")
        logger.info(f"Frames exported: {num_frames}")
        logger.info(
            "Use 'Open Dataset in X-AnyLabeling' button to review/correct annotations"
        )

        # Optional: Show success message
        QMessageBox.information(
            self,
            "Dataset Generation Complete",
            f"Successfully generated dataset with {num_frames} frames.\n\n"
            f"Location: {dataset_dir}\n\n"
            "Use 'Open Dataset in X-AnyLabeling' to review annotations.",
        )

    def on_dataset_error(self: object, error_message: object) -> object:
        """Handle dataset generation errors."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        if self._stop_all_requested:
            self._cleanup_thread_reference("dataset_worker")
            self._refresh_progress_visibility()
            return
        self._refresh_progress_visibility()

        logger.error(f"Dataset generation error: {error_message}")
        QMessageBox.critical(
            self,
            "Dataset Generation Error",
            f"Failed to generate dataset:\n{error_message}",
        )

    def _on_dataset_worker_thread_finished(self):
        """Release completed dataset worker safely."""
        sender = self.sender()
        if (
            sender is not None
            and self.dataset_worker is not None
            and sender is not self.dataset_worker
        ):
            try:
                sender.deleteLater()
            except Exception:
                pass
            return
        self._cleanup_thread_reference("dataset_worker")
        self._refresh_progress_visibility()

    def _is_worker_running(self, worker):
        """Safely check whether a worker thread-like object is running."""
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    def _has_active_progress_task(self) -> bool:
        """Return True if any async task that owns progress UI is still active."""
        return any(
            [
                self._is_worker_running(self.tracking_worker),
                self._is_worker_running(getattr(self, "merge_worker", None)),
                self._is_worker_running(self.dataset_worker),
                self._is_worker_running(self.interp_worker),
            ]
        )

    def _refresh_progress_visibility(self):
        """Keep progress UI visible while any async tracking task is still running."""
        has_active_task = self._has_active_progress_task()
        self.progress_bar.setVisible(has_active_task)
        self.progress_label.setVisible(has_active_task)

    def _cleanup_temporary_files(self):
        """Remove temporary files if cleanup is enabled."""
        if not self.chk_cleanup_temp_files.isChecked():
            logger.info("Temporary file cleanup disabled, keeping intermediate files.")
            return

        if not self.temporary_files:
            logger.info("No temporary files to clean up.")
            return

        cleaned = []
        failed = []
        for temp_file in self.temporary_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    cleaned.append(os.path.basename(temp_file))
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    failed.append(os.path.basename(temp_file))
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        # Clear the list after cleanup attempt
        self.temporary_files.clear()

        # Also clean up posekit directories if they exist
        params = self.get_parameters_dict()
        output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
        if output_dir and os.path.exists(output_dir):
            posekit_dir = os.path.join(output_dir, "posekit")
            if os.path.exists(posekit_dir) and os.path.isdir(posekit_dir):
                try:
                    import shutil

                    shutil.rmtree(posekit_dir)
                    logger.info(f"Removed posekit directory: {posekit_dir}")
                    cleaned.append("posekit/")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove posekit directory {posekit_dir}: {e}"
                    )
                    failed.append("posekit/")

        if cleaned:
            logger.info(
                f"Cleaned up {len(cleaned)} temporary file(s): {', '.join(cleaned)}"
            )
        if failed:
            logger.warning(
                f"Failed to clean {len(failed)} file(s): {', '.join(failed)}"
            )

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _connect_parameter_signals(self):
        widgets_to_connect = (
            self.findChildren(QSpinBox)
            + self.findChildren(QDoubleSpinBox)
            + self.findChildren(QCheckBox)
        )
        for widget in widgets_to_connect:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._on_parameter_changed)
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._on_parameter_changed)

    @Slot()
    def _on_parameter_changed(self):
        params = self.get_parameters_dict()
        self.parameters_changed.emit(params)

    def _create_help_label(self, text):
        """Create a styled help label for section guidance."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(
            "color: #aaa; font-size: 11px; font-weight: normal; "
            "font-style: italic; padding: 4px 2px; margin: 2px 0px;"
        )
        return label

    def _get_roi_hash(self):
        """Generate a hash of current ROI configuration for caching."""
        if not self.roi_shapes:
            return None

        # Create a simple hash from ROI shapes
        roi_str = str(
            [
                (
                    s["type"],
                    (
                        tuple(s["params"])
                        if isinstance(s["params"], list)
                        else s["params"]
                    ),
                    s.get("mode", "include"),
                )
                for s in self.roi_shapes
            ]
        )
        return hash(roi_str)

    def _invalidate_roi_cache(self):
        """Invalidate ROI display cache when ROI changes."""
        self._roi_masked_cache.clear()
        self._roi_hash = self._get_roi_hash()

    # =========================================================================
    # PRESET MANAGEMENT
    # =========================================================================

    def _get_presets_dir(self):
        """Get the presets directory path."""
        # Get the repo root (3 levels up from this file)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        presets_dir = os.path.join(repo_root, "configs")
        return presets_dir

    def _on_preset_selection_changed(self, index):
        """Update description label when preset selection changes."""
        filepath = self.combo_presets.currentData()
        if not filepath or not os.path.exists(filepath):
            self.preset_description_label.setVisible(False)
            return

        try:
            with open(filepath, "r") as f:
                cfg = json.load(f)

            description = cfg.get("description", "")
            if description:
                self.preset_description_label.setText(f"📋 {description}")
                self.preset_description_label.setVisible(True)
            else:
                self.preset_description_label.setVisible(False)
        except (OSError, json.JSONDecodeError):
            self.preset_description_label.setVisible(False)

    def _populate_preset_combo(self):
        """Populate the preset combo box by auto-scanning configs folder."""
        presets_dir = self._get_presets_dir()

        if not os.path.exists(presets_dir):
            return

        presets = []
        custom_preset = None

        # Scan all JSON files in configs directory
        for filename in sorted(os.listdir(presets_dir)):
            if not filename.endswith(".json"):
                continue

            filepath = os.path.join(presets_dir, filename)

            try:
                with open(filepath, "r") as f:
                    cfg = json.load(f)

                # Get preset name from file, fallback to filename
                preset_name = cfg.get(
                    "preset_name",
                    filename.replace(".json", "").replace("_", " ").title(),
                )

                # Custom preset gets special treatment (star marker, shown first)
                if filename == "custom.json":
                    custom_preset = (f"{preset_name} ★", filepath)
                else:
                    presets.append((preset_name, filepath))
            except Exception as e:
                logger.warning(f"Failed to load preset {filename}: {e}")
                continue

        # Populate combo box (custom first, then others alphabetically)
        self.combo_presets.clear()
        if custom_preset:
            self.combo_presets.addItem(custom_preset[0], custom_preset[1])
        for name, filepath in presets:
            self.combo_presets.addItem(name, filepath)

    def _load_selected_preset(self):
        """Load the currently selected preset."""
        filepath = self.combo_presets.currentData()
        if not filepath or not os.path.exists(filepath):
            QMessageBox.warning(
                self, "Preset Not Found", f"Preset file not found: {filepath}"
            )
            return

        # Confirm if current settings differ significantly
        reply = QMessageBox.question(
            self,
            "Load Preset",
            f"Load preset: {self.combo_presets.currentText()}?\n\n"
            "This will replace your current parameter values.\n"
            "(Video-specific configs will still override presets when loading videos)",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Yes:
            # Use existing config loader in preset mode
            self._load_config_from_file(filepath, preset_mode=True)

            # Update status
            preset_name = self.combo_presets.currentText()
            self.preset_status_label.setText(f"✓ Loaded: {preset_name}")
            self.preset_status_label.setStyleSheet(
                "color: #4a9eff; font-style: italic; font-size: 10px;"
            )
            logger.info(f"Loaded preset: {preset_name} from {filepath}")

    def _save_custom_preset(self):
        """Save current settings as custom preset with user-defined name and description."""
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QTextEdit

        # Create dialog for preset metadata
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Preset")
        dialog.setModal(True)
        dialog_layout = QVBoxLayout(dialog)

        # Name input
        name_label = QLabel("Preset name (e.g., danio rerio / zebrafish)")
        name_label.setStyleSheet("color: #fff; font-weight: bold;")
        name_input = QLineEdit()
        name_input.setPlaceholderText("Scientific Name (Common Name)")
        name_input.setText("Custom")

        # Description input
        desc_label = QLabel("Description (optional)")
        desc_label.setStyleSheet("color: #fff; font-weight: bold; margin-top: 10px;")
        desc_input = QTextEdit()
        desc_input.setPlaceholderText(
            "Describe the optimizations or use case for this preset..."
        )
        desc_input.setMaximumHeight(80)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        dialog_layout.addWidget(name_label)
        dialog_layout.addWidget(name_input)
        dialog_layout.addWidget(desc_label)
        dialog_layout.addWidget(desc_input)
        dialog_layout.addWidget(button_box)

        if dialog.exec() != QDialog.Accepted:
            return

        preset_name = name_input.text().strip() or "Custom"
        description = desc_input.toPlainText().strip() or "User-defined custom preset"

        # Ask user for filename
        presets_dir = self._get_presets_dir()
        os.makedirs(presets_dir, exist_ok=True)

        # Generate suggested filename from preset name
        suggested_filename = preset_name.lower()
        suggested_filename = suggested_filename.split("(")[
            0
        ].strip()  # Remove common name part
        suggested_filename = suggested_filename.replace(" ", "_").replace(".", "")
        suggested_filename = "".join(
            c for c in suggested_filename if c.isalnum() or c == "_"
        )
        suggested_filename = suggested_filename or "custom"
        suggested_path = os.path.join(presets_dir, f"{suggested_filename}.json")

        preset_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset As", suggested_path, "JSON Files (*.json)"
        )

        if not preset_path:
            return

        custom_path = preset_path

        # Use existing config saver in preset mode with custom metadata
        success = self.save_config(
            preset_mode=True,
            preset_path=custom_path,
            preset_name=preset_name,
            preset_description=description,
        )

        if success:
            # Refresh combo box to show new preset
            self._populate_preset_combo()
            # Select the newly saved preset
            for i in range(self.combo_presets.count()):
                item_path = self.combo_presets.itemData(i)
                if item_path == custom_path:
                    self.combo_presets.setCurrentIndex(i)
                    break

            self.preset_status_label.setText(f"✓ Saved: {preset_name}")
            self.preset_status_label.setStyleSheet(
                "color: #4a9eff; font-style: italic; font-size: 10px;"
            )

            filename = os.path.basename(custom_path)
            is_custom = filename == "custom.json"

            QMessageBox.information(
                self,
                "Preset Saved",
                f"Your settings have been saved as:\n{preset_name}\n\n"
                f"Location: {custom_path}\n\n"
                + (
                    "This preset will be loaded automatically on startup and will appear\n"
                    "at the top of the preset selector with a ★ indicator."
                    if is_custom
                    else "This preset is now available in the preset selector."
                ),
            )

    def _load_default_preset_on_startup(self):
        """Load default preset on application startup."""
        presets_dir = self._get_presets_dir()

        # Try custom preset first
        custom_path = os.path.join(presets_dir, "custom.json")
        if os.path.exists(custom_path):
            logger.info("Loading custom preset on startup")
            self._load_config_from_file(custom_path, preset_mode=True)
            return

        # Fall back to default preset
        default_path = os.path.join(presets_dir, "default.json")
        if os.path.exists(default_path):
            logger.info("Loading default preset on startup")
            self._load_config_from_file(default_path, preset_mode=True)
            return

        logger.info("No preset found, using hardcoded defaults")

    # =========================================================================
    # ROI OPTIMIZATION AND VIDEO CROPPING
    # =========================================================================

    def _load_advanced_config(self):
        """Load advanced configuration for power users."""
        # Store config in the package directory (where this file is located)
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_dir, "advanced_config.json")

        default_config = {
            "roi_crop_warning_threshold": 0.6,  # Warn if ROI is <60% of frame
            "roi_crop_auto_suggest": True,  # Auto-suggest cropping
            "roi_crop_remind_every_session": False,  # Remind every time or once
            "roi_crop_padding_fraction": 0.05,  # Padding as fraction of min(width, height) - typically 5%
            "video_crop_codec": "libx264",  # Codec for cropped videos (libx264 for quality)
            "video_crop_crf": 18,  # CRF quality (lower = better, 18 = visually lossless)
            "video_crop_preset": "medium",  # ffmpeg preset (ultrafast, fast, medium, slow, veryslow)
            # YOLO Batching - Memory Fractions (device-specific optimization)
            "mps_memory_fraction": 0.3,  # Conservative 30% of unified memory for MPS (Apple Silicon)
            "cuda_memory_fraction": 0.7,  # 70% of VRAM for CUDA (NVIDIA GPUs)
            # Dataset Generation - YOLO Detection Parameters (separate from tracking)
            "dataset_yolo_confidence_threshold": 0.05,  # Very low - detect all animals including uncertain ones for annotation
            "dataset_yolo_iou_threshold": 0.5,  # Moderate - remove obvious duplicates but keep borderline cases for manual review
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded advanced config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load advanced config: {e}")
        else:
            # Auto-create advanced_config.json with defaults on first run
            # This makes the file discoverable for power users who want to customize
            try:
                config_dir = os.path.dirname(config_path)
                os.makedirs(config_dir, exist_ok=True)
                with open(config_path, "w") as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"Created default advanced config at {config_path}")
            except Exception as e:
                logger.warning(f"Could not create advanced config file: {e}")

        return default_config

    def _save_advanced_config(self):
        """Save advanced configuration."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "advanced_config.json",
        )
        try:
            import tempfile

            # Write to temp file first, then rename (atomic on most filesystems)
            config_dir = os.path.dirname(config_path)
            os.makedirs(config_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=config_dir, delete=False, suffix=".tmp"
            ) as tmp:
                json.dump(self.advanced_config, tmp, indent=2)
                tmp_path = tmp.name
            os.replace(tmp_path, config_path)  # Atomic rename
            logger.info(f"Saved advanced config to {config_path}")
        except Exception as e:
            logger.error(f"Could not save advanced config: {e}")

    def _calculate_roi_bounding_box(self, padding=None):
        """Calculate the bounding box of the current ROI mask with optional padding.

        Args:
            padding: Fraction of min(width, height) to add as padding (e.g., 0.05 = 5%).
                    If None, uses value from advanced config.

        Returns:
            Tuple (x, y, w, h) or None if no ROI
        """
        if self.roi_mask is None:
            return None

        # Find all non-zero points in the mask
        points = cv2.findNonZero(self.roi_mask)
        if points is None or len(points) == 0:
            return None

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(points)

        # Apply padding if requested (helps with detection by providing context)
        if padding is None:
            padding = self.advanced_config.get("roi_crop_padding_fraction", 0.05)

        if padding > 0:
            # Get frame dimensions
            frame_h, frame_w = self.roi_mask.shape[:2]

            # Calculate padding in pixels based on smaller dimension
            min_dim = min(frame_w, frame_h)
            padding_pixels = int(min_dim * padding)

            # Add padding while staying within frame bounds
            x = max(0, x - padding_pixels)
            y = max(0, y - padding_pixels)
            w = min(frame_w - x, w + 2 * padding_pixels)
            h = min(frame_h - y, h + 2 * padding_pixels)

        return (x, y, w, h)

    def _estimate_roi_efficiency(self):
        """Estimate the efficiency gain from cropping to ROI.

        Returns:
            tuple: (roi_coverage_percent, potential_speedup_factor) or (None, None)
        """
        if self.roi_mask is None or self.preview_frame_original is None:
            return (None, None)

        bbox = self._calculate_roi_bounding_box()
        if bbox is None:
            return (None, None)

        x, y, w, h = bbox
        frame_h, frame_w = self.roi_mask.shape

        frame_area = frame_w * frame_h
        roi_area = w * h

        roi_coverage = roi_area / frame_area
        # Speedup is roughly inverse of area ratio (simplification, but good estimate)
        potential_speedup = 1.0 / roi_coverage if roi_coverage > 0 else 1.0

        return (roi_coverage * 100, potential_speedup)

    def _update_roi_optimization_info(self):
        """Update the ROI optimization label with efficiency information."""
        coverage, speedup = self._estimate_roi_efficiency()

        if coverage is None:
            if hasattr(self, "roi_optimization_label"):
                self.roi_optimization_label.setText("")
            return

        threshold = self.advanced_config.get("roi_crop_warning_threshold", 0.6) * 100

        if coverage < threshold and hasattr(self, "roi_optimization_label"):
            self.roi_optimization_label.setText(
                f"⚡ ROI is {coverage:.1f}% of frame - up to {speedup:.1f}x faster if cropped!"
            )
        elif hasattr(self, "roi_optimization_label"):
            self.roi_optimization_label.setText("")

    def _check_roi_optimization_warning(self):
        """Check if we should warn the user about ROI optimization."""
        if not self.advanced_config.get("roi_crop_auto_suggest", True):
            return

        # Don't warn if already shown this session (unless configured otherwise)
        if self.roi_crop_warning_shown and not self.advanced_config.get(
            "roi_crop_remind_every_session", False
        ):
            return

        coverage, speedup = self._estimate_roi_efficiency()
        if coverage is None:
            return

        threshold = self.advanced_config.get("roi_crop_warning_threshold", 0.6) * 100

        if coverage < threshold:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("ROI Optimization Opportunity")
            msg.setText("⚡ Performance Optimization Available")
            msg.setInformativeText(
                f"Your ROI covers only {coverage:.1f}% of the video frame.\\n\\n"
                f"Cropping the video to the ROI bounding box could provide\\n"
                f"up to {speedup:.1f}x speedup in tracking performance!\\n\\n"
                f"Would you like to:"
            )

            btn_crop_now = msg.addButton("Crop Video Now", QMessageBox.AcceptRole)
            btn_remind_later = msg.addButton(
                "Remind Me When Tracking", QMessageBox.ActionRole
            )
            btn_dont_show = msg.addButton("Don't Show Again", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_crop_now)

            msg.exec()
            clicked = msg.clickedButton()

            if clicked == btn_crop_now:
                self.crop_video_to_roi()
            elif clicked == btn_remind_later:
                self.roi_crop_warning_shown = True
            elif clicked == btn_dont_show:
                self.advanced_config["roi_crop_auto_suggest"] = False
                self._save_advanced_config()
                self.roi_crop_warning_shown = True

    def crop_video_to_roi(self: object) -> object:
        """Crop the video to the ROI bounding box and save as new file."""
        if self.roi_mask is None:
            QMessageBox.warning(self, "No ROI", "Please define an ROI before cropping.")
            return

        video_path = self.file_line.text()
        if not video_path or not os.path.exists(video_path):
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return

        # Get padding fraction from advanced config (default 5% of min dimension)
        padding_fraction = self.advanced_config.get("roi_crop_padding_fraction", 0.05)

        bbox = self._calculate_roi_bounding_box(padding=padding_fraction)
        if bbox is None:
            QMessageBox.warning(
                self, "Invalid ROI", "Could not calculate ROI bounding box."
            )
            return

        x, y, w, h = bbox
        padding_percent = padding_fraction * 100
        logger.info(
            f"ROI crop with {padding_percent:.1f}% padding: x={x}, y={y}, w={w}, h={h}"
        )

        # Suggest output filename
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        suggested_name = f"{video_name}_cropped_roi.mp4"
        suggested_path = os.path.join(video_dir, suggested_name)

        # Ask user for output location
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Cropped Video",
            suggested_path,
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)",
        )

        if not output_path:
            return  # User cancelled

        try:
            # Use ffmpeg for high-quality cropping that preserves video properties
            # This is much faster and maintains quality better than re-encoding with OpenCV
            import subprocess

            # Get video properties for the success message
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open source video")

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Build ffmpeg command for high-quality cropping
            # Get settings from advanced config
            codec = self.advanced_config.get("video_crop_codec", "libx264")
            crf = str(self.advanced_config.get("video_crop_crf", 18))
            preset = self.advanced_config.get("video_crop_preset", "medium")

            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter:v",
                f"crop={w}:{h}:{x}:{y}",  # Crop to ROI bounding box
                "-c:v",
                codec,  # Video codec from config
                "-crf",
                crf,  # Quality (lower = better, 18 is visually lossless)
                "-preset",
                preset,  # Encoding speed preset
                "-c:a",
                "copy",  # Copy audio without re-encoding
                "-movflags",
                "+faststart",  # Enable streaming/fast preview
                "-y",  # Overwrite output file
                output_path,
            ]

            # Log the ffmpeg command for debugging
            logger.info(
                f"Starting video crop: {frame_w}x{frame_h} -> {w}x{h} (padding: {padding_percent:.1f}%)"
            )
            logger.info(f"ffmpeg command: {' '.join(ffmpeg_cmd)}")

            # Show non-blocking message
            QMessageBox.information(
                self,
                "Cropping Video",
                f"Video cropping has started in the background.\n\n"
                f"Original: {frame_w}x{frame_h}\n"
                f"Cropped: {w}x{h}\n"
                f"Padding: {padding_percent:.1f}% of frame (improves detection)\n\n"
                f"Note: Padding provides visual context for better YOLO confidence.\n"
                f"Adjust 'roi_crop_padding_fraction' in advanced_config.json if needed.\n\n"
                f"This may take a few minutes. The application will remain responsive.\n"
                f"You'll be notified when cropping is complete.",
            )

            # Run ffmpeg in background with progress logging

            # Run ffmpeg in background with progress logging
            # Capture stderr for progress tracking but don't block UI
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                start_new_session=True,  # Detach from parent process
            )

            # Get total frames for progress tracking
            cap_temp = cv2.VideoCapture(video_path)
            total_frames = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_temp.release()

            # Store process info for potential future use
            self._crop_process = {
                "process": process,
                "output_path": output_path,
                "original_size": (frame_w, frame_h),
                "cropped_size": (w, h),
                "total_frames": total_frames,
                "last_logged_progress": 0,
            }

            # Set up a timer to check when process completes
            from PySide6.QtCore import QTimer

            self._crop_check_timer = QTimer()
            self._crop_check_timer.timeout.connect(self._check_crop_completion)
            self._crop_check_timer.start(2000)  # Check every 2 seconds

            # Disable UI controls while cropping is in progress
            self._set_ui_controls_enabled(False)
            # Also disable crop button specifically
            if hasattr(self, "btn_crop_video"):
                self.btn_crop_video.setText("Cropping...")
                self.btn_crop_video.setEnabled(False)

            logger.info(f"Started background video crop: {video_path} -> {output_path}")

        except Exception as e:
            # Re-enable UI if crop failed to start
            self._set_ui_controls_enabled(True)
            if hasattr(self, "btn_crop_video"):
                self.btn_crop_video.setText("Crop Video to ROI")
                self.btn_crop_video.setEnabled(True)

            QMessageBox.critical(
                self,
                "Crop Failed",
                f"Failed to start video crop:\n{str(e)}",
            )
            logger.error(f"Video crop failed: {e}")
            import traceback

            traceback.print_exc()

    def _check_crop_completion(self):
        """Check if background crop process has completed."""
        if not hasattr(self, "_crop_process"):
            if hasattr(self, "_crop_check_timer"):
                self._crop_check_timer.stop()
            return

        process = self._crop_process["process"]

        # Read and log any new stderr output (ffmpeg progress)
        try:
            # Read available lines without blocking (non-blocking I/O)
            if process.stderr:
                import fcntl
                import os as os_module

                # Set stderr to non-blocking mode
                fd = process.stderr.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os_module.O_NONBLOCK)

                try:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break

                        # Parse progress from ffmpeg output
                        if "frame=" in line:
                            try:
                                frame_str = line.split("frame=")[1].split()[0]
                                current_frame = int(frame_str)
                                total_frames = self._crop_process.get("total_frames", 0)

                                # Log every 10% of progress
                                if total_frames > 0:
                                    progress_pct = int(
                                        (current_frame / total_frames) * 100
                                    )
                                    last_logged = self._crop_process.get(
                                        "last_logged_progress", 0
                                    )

                                    if progress_pct >= last_logged + 10:
                                        logger.info(
                                            f"Video crop progress: {progress_pct}% ({current_frame}/{total_frames} frames)"
                                        )
                                        self._crop_process["last_logged_progress"] = (
                                            progress_pct
                                        )
                            except (ValueError, IndexError):
                                pass
                except (IOError, OSError):
                    # No data available right now
                    pass
        except Exception:
            # Don't let logging errors break the process
            pass

        return_code = process.poll()  # Non-blocking check

        if return_code is not None:  # Process has finished
            self._crop_check_timer.stop()
            output_path = self._crop_process["output_path"]
            orig_w, orig_h = self._crop_process["original_size"]
            crop_w, crop_h = self._crop_process["cropped_size"]

            if return_code == 0 and os.path.exists(output_path):
                # Success - ask if user wants to load the cropped video
                reply = QMessageBox.question(
                    self,
                    "Crop Complete",
                    f"Video successfully cropped to ROI!\n\n"
                    f"Original: {orig_w}x{orig_h}\n"
                    f"Cropped: {crop_w}x{crop_h}\n"
                    f"Saved to: {os.path.basename(output_path)}\n\n"
                    f"Would you like to load the cropped video now?\n"
                    f"(Note: ROI will be cleared since the video is already cropped)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    # Load the cropped video with full initialization (same as select_file)
                    self.file_line.setText(output_path)
                    self.current_video_path = output_path
                    self.clear_roi()  # Clear ROI since we're loading the cropped version

                    # Auto-generate output paths based on cropped video name
                    video_dir = os.path.dirname(output_path)
                    video_name = os.path.splitext(os.path.basename(output_path))[0]

                    # Auto-populate CSV output
                    csv_path = os.path.join(video_dir, f"{video_name}_tracking.csv")
                    self.csv_line.setText(csv_path)

                    # Auto-populate video output and enable it
                    video_out_path = os.path.join(
                        video_dir, f"{video_name}_tracking.mp4"
                    )
                    self.video_out_line.setText(video_out_path)
                    self.check_video_output.setChecked(True)

                    # Enable preview detection button
                    self.btn_test_detection.setEnabled(True)
                    self.btn_detect_fps.setEnabled(True)

                    # Disable crop button and clear optimization info (no ROI anymore)
                    self.btn_crop_video.setEnabled(False)
                    if hasattr(self, "roi_optimization_label"):
                        self.roi_optimization_label.setText("")
                    self.roi_crop_warning_shown = False

                    # Auto-load config if it exists
                    config_path = get_video_config_path(output_path)
                    if config_path and os.path.isfile(config_path):
                        self._load_config_from_file(config_path)
                        self.config_status_label.setText(
                            f"✓ Loaded: {os.path.basename(config_path)}"
                        )
                        self.config_status_label.setStyleSheet(
                            "color: #4a9eff; font-style: italic; font-size: 10px;"
                        )
                        logger.info(
                            f"Cropped video loaded: {output_path} (auto-loaded config)"
                        )
                    else:
                        self.config_status_label.setText(
                            "No config found (using current settings)"
                        )
                        self.config_status_label.setStyleSheet(
                            "color: #f39c12; font-style: italic; font-size: 10px;"
                        )
                        logger.info(
                            f"Cropped video loaded: {output_path} (no config found)"
                        )

                # Re-enable UI controls after successful crop
                self._set_ui_controls_enabled(True)
                if hasattr(self, "btn_crop_video"):
                    self.btn_crop_video.setText("Crop Video to ROI")

                logger.info(f"Successfully cropped video to {output_path}")
            else:
                # Process failed - re-enable UI
                self._set_ui_controls_enabled(True)
                if hasattr(self, "btn_crop_video"):
                    self.btn_crop_video.setText("Crop Video to ROI")
                    self.btn_crop_video.setEnabled(True)

                logger.error(f"Video crop failed with return code {return_code}")
                QMessageBox.critical(
                    self,
                    "Crop Failed",
                    f"Video cropping failed (return code: {return_code})\n\n"
                    f"Check that ffmpeg is installed and the video is valid.",
                )

            # Clean up
            del self._crop_process

    def plot_fps(self: object, fps_list: object) -> object:
        """plot_fps method documentation."""
        if len(fps_list) < 2:
            return
        plt.figure()
        plt.plot(fps_list)
        plt.xlabel("Frame Index")
        plt.ylabel("FPS")
        plt.title("Tracking FPS Over Time")
        plt.show()
