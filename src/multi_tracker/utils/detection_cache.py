"""
Detection caching for efficient bidirectional tracking.

This module provides a memory-efficient way to cache detection data from the forward
tracking pass and reuse it during the backward pass, eliminating the need for
RAM-intensive video reversal.
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DetectionCache:
    """
    Efficient frame-by-frame detection cache using NPZ format.

    This class provides streaming write during forward pass and streaming read
    during backward pass, minimizing memory footprint while enabling detection reuse.
    """

    def __init__(self, cache_path, mode="w"):
        """
        Initialize detection cache.

        Args:
            cache_path: Path to the .npz cache file
            mode: 'w' for writing (forward pass), 'r' for reading (backward pass)
        """
        self.cache_path = Path(cache_path)
        self.mode = mode
        self._data = {}  # Temporary storage during writing
        self._loaded_data = None  # Loaded data during reading
        self._total_frames = 0

        if mode == "r" and self.cache_path.exists():
            logger.info(f"Loading detection cache from {self.cache_path}")
            self._loaded_data = np.load(str(self.cache_path), allow_pickle=True)
            # Extract total frames from metadata
            if "metadata" in self._loaded_data:
                metadata = self._loaded_data["metadata"].item()
                self._total_frames = metadata.get("total_frames", 0)
            logger.info(f"Cache loaded: {self._total_frames} frames")

    def add_frame(
        self,
        frame_idx,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners=None,
        detection_ids=None,
    ):
        """
        Add detection data for a single frame (forward pass).

        Args:
            frame_idx: Frame number (0-based)
            meas: List of numpy arrays, each [cx, cy, theta]
            sizes: List of detection areas
            shapes: List of tuples (ellipse_area, aspect_ratio)
            confidences: List of confidence scores (float or nan)
            obb_corners: Optional list of OBB corner arrays for YOLO
            detection_ids: Optional list of detection IDs (FrameID * 10000 + detection_index)
        """
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write")

        # Convert to numpy arrays for efficient storage
        if len(meas) > 0:
            meas_arr = np.array(meas, dtype=np.float32)  # Shape: (n, 3)
            sizes_arr = np.array(sizes, dtype=np.float32)
            shapes_arr = np.array(shapes, dtype=np.float32)  # Shape: (n, 2)
            confidences_arr = np.array(confidences, dtype=np.float32)

            # Store detection IDs if provided
            if detection_ids and len(detection_ids) > 0:
                detection_ids_arr = np.array(detection_ids, dtype=np.float64)
            else:
                detection_ids_arr = np.array([], dtype=np.float64)

            # Store OBB corners if provided (YOLO detection)
            if obb_corners and len(obb_corners) > 0:
                obb_arr = np.array(obb_corners, dtype=np.float32)
            else:
                obb_arr = np.array([], dtype=np.float32)
        else:
            # Empty frame (no detections)
            meas_arr = np.array([], dtype=np.float32).reshape(0, 3)
            sizes_arr = np.array([], dtype=np.float32)
            shapes_arr = np.array([], dtype=np.float32).reshape(0, 2)
            confidences_arr = np.array([], dtype=np.float32)
            detection_ids_arr = np.array([], dtype=np.float64)
            obb_arr = np.array([], dtype=np.float32)

        # Store with frame index as key
        frame_key = f"frame_{frame_idx:06d}"
        self._data[f"{frame_key}_meas"] = meas_arr
        self._data[f"{frame_key}_sizes"] = sizes_arr
        self._data[f"{frame_key}_shapes"] = shapes_arr
        self._data[f"{frame_key}_confidences"] = confidences_arr
        self._data[f"{frame_key}_detection_ids"] = detection_ids_arr
        self._data[f"{frame_key}_obb"] = obb_arr

        self._total_frames = max(self._total_frames, frame_idx + 1)

    def save(self):
        """Save cached detections to disk (call at end of forward pass)."""
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save")

        # Add metadata
        self._data["metadata"] = np.array(
            {"total_frames": self._total_frames, "version": "1.0"}
        )

        logger.info(
            f"Saving detection cache to {self.cache_path} ({self._total_frames} frames)"
        )

        # Use compressed format for smaller file size
        np.savez_compressed(str(self.cache_path), **self._data)

        # Calculate file size for logging
        file_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Detection cache saved: {file_size_mb:.2f} MB")

        # Clear memory
        self._data.clear()

    def get_frame(self, frame_idx):
        """
        Get detection data for a single frame (backward pass).

        Args:
            frame_idx: Frame number (0-based)

        Returns:
            Tuple of (meas, sizes, shapes, confidences, obb_corners, detection_ids)
            where meas is a list of numpy arrays to match the tracking worker API
        """
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read")

        if self._loaded_data is None:
            raise RuntimeError("No cache data loaded")

        frame_key = f"frame_{frame_idx:06d}"

        # Load arrays
        meas_arr = self._loaded_data.get(
            f"{frame_key}_meas", np.array([], dtype=np.float32).reshape(0, 3)
        )
        sizes_arr = self._loaded_data.get(
            f"{frame_key}_sizes", np.array([], dtype=np.float32)
        )
        shapes_arr = self._loaded_data.get(
            f"{frame_key}_shapes", np.array([], dtype=np.float32).reshape(0, 2)
        )
        confidences_arr = self._loaded_data.get(
            f"{frame_key}_confidences", np.array([], dtype=np.float32)
        )
        detection_ids_arr = self._loaded_data.get(
            f"{frame_key}_detection_ids", np.array([], dtype=np.float64)
        )
        obb_arr = self._loaded_data.get(
            f"{frame_key}_obb", np.array([], dtype=np.float32)
        )

        # Convert back to lists matching the original format
        # meas should be list of numpy arrays (one per detection)
        meas = [meas_arr[i] for i in range(len(meas_arr))] if len(meas_arr) > 0 else []
        sizes = sizes_arr.tolist()
        shapes = (
            [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
            if len(shapes_arr) > 0
            else []
        )
        confidences = confidences_arr.tolist()
        detection_ids = detection_ids_arr.tolist()

        # OBB corners (list of arrays)
        if len(obb_arr) > 0 and obb_arr.ndim > 1:
            obb_corners = [obb_arr[i] for i in range(len(obb_arr))]
        else:
            obb_corners = []

        return meas, sizes, shapes, confidences, obb_corners, detection_ids

    def get_total_frames(self):
        """Get total number of frames in cache."""
        return self._total_frames

    def close(self):
        """Close and cleanup cache resources."""
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None
        logger.info(f"Detection cache closed: {self.cache_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
