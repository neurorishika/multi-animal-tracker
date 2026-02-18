"""
Worker for extracting appearance embeddings from detection crops.

Independent from pose extraction, allows on-demand computation of appearance
embeddings for re-identification and similarity analysis.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class AppearanceExtractionWorker(QThread):
    """QThread worker for extracting appearance embeddings from detection crops."""

    progress_signal = pyqtSignal(int, str)  # (percent, message)
    stats_signal = pyqtSignal(dict)  # Runtime statistics
    completed_signal = pyqtSignal(str, dict)  # (cache_path, stats)
    error_signal = pyqtSignal(str)  # Error message

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        params: Dict,
        appearance_config: Dict,
        output_cache_path: str,
    ):
        super().__init__()
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self.appearance_config = appearance_config
        self.output_cache_path = output_cache_path
        self._stop_requested = False

    def request_stop(self):
        """Request worker to stop processing."""
        self._stop_requested = True

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

    def run(self):
        """Main worker execution."""
        try:
            self._extract_embeddings()
        except Exception as e:
            logger.exception("Appearance extraction worker failed")
            self.error_signal.emit(f"Extraction failed: {str(e)}")

    def _extract_embeddings(self):
        """Extract appearance embeddings and save to cache."""
        from multi_tracker.core.detection.detection_cache import DetectionCache
        from multi_tracker.core.identity.appearance_cache import (
            AppearanceEmbeddingCache,
        )
        from multi_tracker.core.identity.runtime_api import (
            AppearanceRuntimeConfig,
            create_appearance_backend_from_config,
        )

        logger.info("=" * 80)
        logger.info("APPEARANCE EXTRACTION: Computing embeddings for detections")
        logger.info("=" * 80)
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Frame range: {self.start_frame} - {self.end_frame}")
        logger.info(f"Model: {self.appearance_config['model_name']}")
        logger.info(f"Device: {self.appearance_config['device']}")
        logger.info(f"Output cache: {self.output_cache_path}")

        self.progress_signal.emit(0, "Initializing appearance backend...")

        # Create appearance runtime config
        runtime_config = AppearanceRuntimeConfig(
            model_name=self.appearance_config["model_name"],
            runtime_flavor=self.appearance_config.get("runtime_flavor", "auto"),
            batch_size=self.appearance_config["batch_size"],
            max_image_side=self.appearance_config["max_image_side"],
            use_clahe=self.appearance_config["use_clahe"],
            normalize_embeddings=self.appearance_config.get("normalize", True),
        )

        # Create backend
        backend = create_appearance_backend_from_config(runtime_config)

        self.progress_signal.emit(5, "Loading appearance model...")
        backend.warmup()

        embedding_dim = backend.output_dimension
        logger.info(f"Appearance model loaded. Embedding dimension: {embedding_dim}")

        self.progress_signal.emit(10, "Loading detection cache...")

        # Load detection cache
        detection_cache = DetectionCache(self.detection_cache_path, mode="r")
        if not detection_cache.is_compatible():
            raise RuntimeError(
                f"Detection cache is incompatible: {self.detection_cache_path}"
            )

        self.progress_signal.emit(15, "Opening video and preparing for extraction...")

        # Open video
        resize_f = float(self.params.get("RESIZE_FACTOR", 1.0))
        padding_fraction = float(self.params.get("INDIVIDUAL_CROP_PADDING", 0.1))
        roi_mask = self.params.get("ROI_MASK", None)

        if roi_mask is not None:
            dims_cap = cv2.VideoCapture(self.video_path)
            base_h = int(dims_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            base_w = int(dims_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            dims_cap.release()
            target_w = max(1, int(base_w * resize_f))
            target_h = max(1, int(base_h * resize_f))
            if roi_mask.shape[1] != target_w or roi_mask.shape[0] != target_h:
                roi_mask = cv2.resize(roi_mask, (target_w, target_h), cv2.INTER_NEAREST)

        video_cap = cv2.VideoCapture(self.video_path)
        if not video_cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        if self.start_frame > 0:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        # Create cache writer
        cache_writer = AppearanceEmbeddingCache(self.output_cache_path, mode="w")
        total_frames = max(1, self.end_frame - self.start_frame + 1)
        extraction_start_ts = time.time()
        total_detections = 0
        total_embeddings_computed = 0
        cancelled = False

        self.progress_signal.emit(
            15, f"Processing {total_frames} frame(s) for appearance extraction..."
        )

        # Load detector for filtering if needed
        detector = None
        if self.params.get("DETECTION_METHOD") == "yolo_obb":
            from multi_tracker.core.detection.yolo_detector import YoloDetector

            detector = YoloDetector(self.params)

        try:
            for rel_idx, frame_idx in enumerate(
                range(self.start_frame, self.end_frame + 1)
            ):
                if self._stop_requested:
                    cancelled = True
                    break

                # Get detections from cache
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    raw_detection_ids,
                ) = detection_cache.get_frame(frame_idx)

                # Apply filtering if detector available
                if detector and raw_meas:
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
                else:
                    meas = raw_meas
                    filtered_obb_corners = raw_obb_corners
                    detection_ids = raw_detection_ids

                # Read frame
                ret, frame = video_cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame {frame_idx}, skipping")
                    # Add empty frame to cache
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

                    # Extract crops
                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            crops.append(crop)
                            crop_to_det.append(det_idx)

                    # Initialize embeddings list with None
                    embeddings_list = [None] * len(meas)

                    # Compute embeddings if we have crops
                    if crops:
                        if self._stop_requested:
                            cancelled = True
                            break

                        total_embeddings_computed += len(crops)
                        appearance_results = backend.predict_crops(crops)

                        # Map results back to detections
                        for crop_idx, det_idx in enumerate(crop_to_det):
                            if crop_idx < len(appearance_results):
                                result = appearance_results[crop_idx]
                                embeddings_list[det_idx] = result.embedding

                # Add frame to cache
                cache_writer.add_frame(
                    frame_idx, detection_ids or [], embeddings=embeddings_list
                )

                # Update progress
                processed_count = rel_idx + 1
                if rel_idx % 10 == 0 or rel_idx == total_frames - 1:
                    elapsed = max(1e-6, time.time() - extraction_start_ts)
                    rate_fps = processed_count / elapsed
                    remaining = max(0, total_frames - processed_count)
                    eta = (remaining / rate_fps) if rate_fps > 1e-9 else 0.0
                    pct = 15 + int((processed_count * 80) / total_frames)
                    self.progress_signal.emit(
                        pct,
                        f"Extracting embeddings: {processed_count}/{total_frames} frames",
                    )
                    self.stats_signal.emit(
                        {
                            "phase": "appearance_extraction",
                            "fps": rate_fps,
                            "elapsed": elapsed,
                            "eta": eta,
                            "detections": total_detections,
                            "embeddings": total_embeddings_computed,
                        }
                    )

            if cancelled or self._stop_requested:
                logger.info("Appearance extraction cancelled.")
                self.progress_signal.emit(0, "Extraction cancelled.")
                self.error_signal.emit("Extraction cancelled by user.")
                return

            # Save cache
            self.progress_signal.emit(95, "Saving appearance embedding cache...")

            cache_writer.save(
                metadata={
                    "model_name": self.appearance_config["model_name"],
                    "embedding_dimension": embedding_dim,
                    "device": self.appearance_config["device"],
                    "batch_size": self.appearance_config["batch_size"],
                    "max_image_side": self.appearance_config["max_image_side"],
                    "use_clahe": self.appearance_config["use_clahe"],
                    "normalize_embeddings": self.appearance_config.get(
                        "normalize", True
                    ),
                    "start_frame": self.start_frame,
                    "end_frame": self.end_frame,
                    "video_path": str(Path(self.video_path).expanduser().resolve()),
                    "detection_cache_path": str(self.detection_cache_path),
                }
            )

            logger.info(f"Appearance cache saved: {self.output_cache_path}")

            elapsed = time.time() - extraction_start_ts
            final_stats = {
                "total_frames": total_frames,
                "total_detections": total_detections,
                "total_embeddings": total_embeddings_computed,
                "embedding_dimension": embedding_dim,
                "elapsed_seconds": elapsed,
                "fps": total_frames / elapsed if elapsed > 0 else 0,
            }

            self.progress_signal.emit(100, "Appearance extraction complete!")
            self.completed_signal.emit(self.output_cache_path, final_stats)

            logger.info("=" * 80)
            logger.info("APPEARANCE EXTRACTION COMPLETE")
            logger.info(f"Total frames: {total_frames}")
            logger.info(f"Total detections: {total_detections}")
            logger.info(f"Embeddings computed: {total_embeddings_computed}")
            logger.info(f"Embedding dimension: {embedding_dim}")
            logger.info(f"Elapsed time: {elapsed:.2f}s")
            logger.info("=" * 80)

        finally:
            cache_writer.close()
            video_cap.release()
            detection_cache.close()
            try:
                backend.close()
            except Exception:
                pass
