"""
Detection caching for efficient bidirectional tracking.

This module provides a memory-efficient way to cache detection data from the forward
tracking pass and reuse it during the backward pass, eliminating the need for
RAM-intensive video reversal.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_detection_ids(detection_ids):
    """Return integer detection IDs from persisted cache values.

    Legacy caches stored IDs as float64. Accept those values when they are
    finite whole numbers so existing caches remain readable.
    """
    if detection_ids is None:
        return []

    arr = np.asarray(detection_ids)
    if arr.size == 0:
        return []

    normalized = []
    for raw_value in arr.reshape(-1).tolist():
        if isinstance(raw_value, (np.integer, int)):
            normalized.append(int(raw_value))
            continue

        try:
            float_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid detection ID value: {raw_value!r}") from exc

        if not np.isfinite(float_value) or not float_value.is_integer():
            raise ValueError(
                f"Detection ID must be a finite whole number, got {raw_value!r}"
            )
        normalized.append(int(float_value))

    return normalized


class DetectionCache:
    """
    Efficient frame-by-frame detection cache using NPZ format.

    This class provides streaming write during forward pass and streaming read
    during backward pass, minimizing memory footprint while enabling detection reuse.
    """

    def __init__(self, cache_path, mode="w", start_frame=0, end_frame=None):
        """
        Initialize detection cache.

        Args:
            cache_path: Path to the .npz cache file
            mode: 'w' for writing (forward pass), 'r' for reading (backward pass)
            start_frame: Starting frame index for writing
            end_frame: Ending frame index for writing
        """
        self.cache_path = Path(cache_path)
        self.mode = mode
        self._data = {}  # Temporary storage during writing
        self._loaded_data = None  # Loaded data during reading
        self._total_frames = 0
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._cached_frames = None
        self._compatible = True

        if mode == "r" and self.cache_path.exists():
            logger.info(f"Loading detection cache from {self.cache_path}")
            self._loaded_data = np.load(str(self.cache_path), allow_pickle=True)
            # Extract metadata
            if "metadata" in self._loaded_data:
                metadata = self._loaded_data["metadata"].item()
                cache_version = str(metadata.get("version", ""))
                if cache_version not in {"2.0", "2.1", "2.2", "2.3", "2.4"}:
                    logger.warning(
                        f"Incompatible detection cache version '{cache_version}' "
                        f"(expected '2.0', '2.1', '2.2', '2.3', or '2.4'). Cache will be regenerated."
                    )
                    self._compatible = False
                    self._loaded_data.close()
                    self._loaded_data = None
                    self._cached_frames = set()
                    return
                self._total_frames = metadata.get("total_frames", 0)
                self._start_frame = metadata.get("start_frame", 0)
                self._end_frame = metadata.get("end_frame", self._total_frames - 1)
            else:
                logger.warning(
                    "Detection cache missing metadata. Cache will be regenerated."
                )
                self._compatible = False
                self._loaded_data.close()
                self._loaded_data = None
                self._cached_frames = set()
                return
            logger.info(
                f"Cache loaded: {self._total_frames} frames (range: {self._start_frame}-{self._end_frame})"
            )
            self._cached_frames = self._extract_cached_frames()

    def _extract_cached_frames(self):
        """Extract available frame indices from loaded cache keys."""
        if self._loaded_data is None:
            return set()

        cached = set()
        for key in self._loaded_data.files:
            if not key.startswith("frame_") or not key.endswith("_meas"):
                continue
            try:
                frame_str = key.split("_")[1]
                cached.add(int(frame_str))
            except (IndexError, ValueError):
                continue
        return cached

    @staticmethod
    def _to_optional_arr(data, dtype, filter_none=False):
        """Convert optional list to numpy array, returning empty array if absent."""
        if not data or len(data) == 0:
            return np.array([], dtype=dtype)
        if filter_none:
            data = [v for v in data if v is not None]
            if not data:
                return np.array([], dtype=dtype)
        return np.array(data, dtype=dtype)

    def add_frame(
        self: object,
        frame_idx: object,
        meas: object,
        sizes: object,
        shapes: object,
        confidences: object,
        obb_corners: object = None,
        detection_ids: object = None,
        heading_hints: object = None,
        heading_confidences: object = None,
        directed_mask: object = None,
        canonical_affines: object = None,
        canonical_canvas_dims: object = None,
        canonical_M_inverse: object = None,
    ) -> object:
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
            heading_hints: Optional list of directed heading hints (radians).
            heading_confidences: Optional list of head-tail confidence scores.
            directed_mask: Optional list indicating whether heading_hints are directed.
            canonical_affines: Optional list of (2, 3) affine matrices (M_align per detection).
            canonical_canvas_dims: Optional list of (width, height) int tuples per detection.
            canonical_M_inverse: Optional list of (2, 3) inverse affine matrices per detection.
        """
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write")

        if len(meas) > 0:
            meas_arr = np.array(meas, dtype=np.float32)
            sizes_arr = np.array(sizes, dtype=np.float32)
            shapes_arr = np.array(shapes, dtype=np.float32)
            confidences_arr = np.array(confidences, dtype=np.float32)

            if detection_ids and len(detection_ids) > 0:
                detection_ids_arr = np.array(
                    _normalize_detection_ids(detection_ids), dtype=np.int64
                )
            else:
                detection_ids_arr = np.array([], dtype=np.int64)

            obb_arr = self._to_optional_arr(obb_corners, np.float32)
            heading_hints_arr = self._to_optional_arr(heading_hints, np.float32)
            heading_confidences_arr = self._to_optional_arr(
                heading_confidences, np.float32
            )
            directed_mask_arr = self._to_optional_arr(directed_mask, np.uint8)
            canonical_affines_arr = self._to_optional_arr(
                canonical_affines, np.float32, filter_none=True
            )
            canvas_dims_arr = self._to_optional_arr(
                canonical_canvas_dims, np.int32, filter_none=True
            )
            M_inverse_arr = self._to_optional_arr(
                canonical_M_inverse, np.float32, filter_none=True
            )
        else:
            meas_arr = np.array([], dtype=np.float32).reshape(0, 3)
            sizes_arr = np.array([], dtype=np.float32)
            shapes_arr = np.array([], dtype=np.float32).reshape(0, 2)
            confidences_arr = np.array([], dtype=np.float32)
            detection_ids_arr = np.array([], dtype=np.int64)
            obb_arr = np.array([], dtype=np.float32)
            heading_hints_arr = np.array([], dtype=np.float32)
            heading_confidences_arr = np.array([], dtype=np.float32)
            directed_mask_arr = np.array([], dtype=np.uint8)
            canonical_affines_arr = np.array([], dtype=np.float32)
            canvas_dims_arr = np.array([], dtype=np.int32)
            M_inverse_arr = np.array([], dtype=np.float32)

        frame_key = f"frame_{frame_idx:06d}"
        self._data[f"{frame_key}_meas"] = meas_arr
        self._data[f"{frame_key}_sizes"] = sizes_arr
        self._data[f"{frame_key}_shapes"] = shapes_arr
        self._data[f"{frame_key}_confidences"] = confidences_arr
        self._data[f"{frame_key}_detection_ids"] = detection_ids_arr
        self._data[f"{frame_key}_obb"] = obb_arr
        self._data[f"{frame_key}_heading_hints"] = heading_hints_arr
        self._data[f"{frame_key}_heading_confidences"] = heading_confidences_arr
        self._data[f"{frame_key}_directed_mask"] = directed_mask_arr
        self._data[f"{frame_key}_canonical_affine"] = canonical_affines_arr
        self._data[f"{frame_key}_canvas_dims"] = canvas_dims_arr
        self._data[f"{frame_key}_M_inverse"] = M_inverse_arr

        self._total_frames = max(self._total_frames, frame_idx + 1)

    def save(self: object) -> object:
        """Save cached detections to disk (call at end of forward pass)."""
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save")

        # Add metadata including frame range
        self._data["metadata"] = np.array(
            {
                "total_frames": self._total_frames,
                "start_frame": self._start_frame,
                "end_frame": self._end_frame,
                "version": "2.4",
                "format": "raw_detections",
            }
        )

        logger.info(
            f"Saving detection cache to {self.cache_path} ({self._total_frames} frames)"
        )

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Use compressed format for smaller file size
        np.savez_compressed(str(self.cache_path), **self._data)

        # Calculate file size for logging
        file_size_mb = self.cache_path.stat().st_size / (1024 * 1024)
        logger.info(f"Detection cache saved: {file_size_mb:.2f} MB")

        # Clear memory
        self._data.clear()

    def get_frame(self: object, frame_idx: object) -> object:
        """
        Get detection data for a single frame (backward pass).

        Args:
            frame_idx: Frame number (0-based)

        Returns:
            Tuple of (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners,
                detection_ids,
                heading_hints,
                heading_confidences,
                directed_mask,
            )
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
            f"{frame_key}_detection_ids", np.array([], dtype=np.int64)
        )
        obb_arr = self._loaded_data.get(
            f"{frame_key}_obb", np.array([], dtype=np.float32)
        )
        heading_hints_arr = self._loaded_data.get(
            f"{frame_key}_heading_hints", np.array([], dtype=np.float32)
        )
        heading_confidences_arr = self._loaded_data.get(
            f"{frame_key}_heading_confidences", np.array([], dtype=np.float32)
        )
        directed_mask_arr = self._loaded_data.get(
            f"{frame_key}_directed_mask", np.array([], dtype=np.uint8)
        )
        canonical_affines_arr = self._loaded_data.get(
            f"{frame_key}_canonical_affine", np.array([], dtype=np.float32)
        )
        canvas_dims_arr = self._loaded_data.get(
            f"{frame_key}_canvas_dims", np.array([], dtype=np.int32)
        )
        M_inverse_arr = self._loaded_data.get(
            f"{frame_key}_M_inverse", np.array([], dtype=np.float32)
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
        detection_ids = _normalize_detection_ids(detection_ids_arr)

        # OBB corners (list of arrays)
        if len(obb_arr) > 0 and obb_arr.ndim > 1:
            obb_corners = [obb_arr[i] for i in range(len(obb_arr))]
        else:
            obb_corners = []
        heading_hints = heading_hints_arr.tolist()
        heading_confidences = heading_confidences_arr.tolist()
        directed_mask = directed_mask_arr.tolist()

        # Canonical affines: (N, 2, 3) or empty
        if len(canonical_affines_arr) > 0 and canonical_affines_arr.ndim == 3:
            canonical_affines = [
                canonical_affines_arr[i] for i in range(len(canonical_affines_arr))
            ]
        else:
            canonical_affines = None

        # Canvas dims: (N, 2) int32 or empty
        if len(canvas_dims_arr) > 0 and canvas_dims_arr.ndim == 2:
            canonical_canvas_dims = [
                tuple(canvas_dims_arr[i]) for i in range(len(canvas_dims_arr))
            ]
        else:
            canonical_canvas_dims = None

        # M_inverse: (N, 2, 3) float32 or empty
        if len(M_inverse_arr) > 0 and M_inverse_arr.ndim == 3:
            canonical_M_inverse = [M_inverse_arr[i] for i in range(len(M_inverse_arr))]
        else:
            canonical_M_inverse = None

        return (
            meas,
            sizes,
            shapes,
            confidences,
            obb_corners,
            detection_ids,
            heading_hints,
            heading_confidences,
            directed_mask,
            canonical_affines,
            canonical_canvas_dims,
            canonical_M_inverse,
        )

    def get_total_frames(self: object) -> object:
        """Get total number of frames in cache."""
        return self._total_frames

    def is_compatible(self):
        """Return whether the loaded cache format is supported by current code."""
        return self._compatible

    def get_frame_range(self: object) -> object:
        """Get the frame range stored in cache."""
        return self._start_frame, self._end_frame

    def covers_frame_range(
        self: object, start_frame: object, end_frame: object
    ) -> object:
        """Check if cache fully covers the requested frame range."""
        if self._loaded_data is None:
            return False
        if self._start_frame > start_frame or self._end_frame < end_frame:
            return False
        if self._cached_frames is None:
            return False
        return all(
            frame_idx in self._cached_frames
            for frame_idx in range(start_frame, end_frame + 1)
        )

    def matches_frame_range(self, start_frame, end_frame):
        """Return whether the cache metadata exactly matches the requested range."""
        return self._start_frame == start_frame and self._end_frame == end_frame

    def get_missing_frames(
        self: object, start_frame: object, end_frame: object, max_report: object = 10
    ) -> object:
        """Return a list of missing frame indices (up to max_report)."""
        if self._cached_frames is None:
            return []
        missing = [
            frame_idx
            for frame_idx in range(start_frame, end_frame + 1)
            if frame_idx not in self._cached_frames
        ]
        return missing[:max_report]

    def close(self: object) -> object:
        """Close and cleanup cache resources."""
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None
        logger.info(f"Detection cache closed: {self.cache_path}")

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
