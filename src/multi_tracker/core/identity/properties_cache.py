"""
Caching utilities for per-detection individual properties.

This cache is keyed by:
  - detection hash (inference identity + video/range fingerprint)
  - filter settings hash
  - extractor settings hash
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.2"


def _normalize_for_hash(value: Any) -> Any:
    """Convert values into deterministic JSON-serializable structures."""
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
            str(k): _normalize_for_hash(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    return value


def _hash_payload(payload: Dict[str, Any], length: int = 16) -> str:
    blob = json.dumps(_normalize_for_hash(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def _file_fingerprint(path_str: Any) -> Dict[str, Any]:
    configured = str(path_str or "")
    resolved = ""
    if configured:
        try:
            resolved = str(Path(configured).expanduser().resolve())
        except Exception:
            resolved = configured
    fp = {
        "configured_path": configured,
        "resolved_path": resolved,
        "exists": False,
        "size_bytes": None,
        "mtime_ns": None,
    }
    if resolved and Path(resolved).exists():
        try:
            stat = Path(resolved).stat()
            fp["exists"] = True
            fp["size_bytes"] = int(stat.st_size)
            fp["mtime_ns"] = int(stat.st_mtime_ns)
        except OSError:
            pass
    return fp


def compute_detection_hash(
    inference_model_id: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    detection_cache_version: str = "2.0",
) -> str:
    """Hash identity for raw detections over a specific video frame range."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "inference_model_id": str(inference_model_id or ""),
        "video": _file_fingerprint(video_path),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "detection_cache_version": str(detection_cache_version),
    }
    return _hash_payload(payload)


def compute_filter_settings_hash(params: Dict[str, Any]) -> str:
    """Hash filtering settings that determine filtered detections."""
    roi_mask = params.get("ROI_MASK", None)
    roi_digest = None
    if isinstance(roi_mask, np.ndarray):
        roi_arr = np.ascontiguousarray(roi_mask)
        roi_digest = hashlib.md5(roi_arr.tobytes()).hexdigest()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection_method": params.get("DETECTION_METHOD", "background_subtraction"),
        "yolo_confidence_threshold": params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25),
        "yolo_iou_threshold": params.get("YOLO_IOU_THRESHOLD", 0.7),
        "enable_size_filtering": bool(params.get("ENABLE_SIZE_FILTERING", False)),
        "min_object_size": params.get("MIN_OBJECT_SIZE", 0),
        "max_object_size": params.get("MAX_OBJECT_SIZE", None),
        "roi_mask_digest": roi_digest,
    }
    return _hash_payload(payload)


def compute_extractor_hash(params: Dict[str, Any]) -> str:
    """Hash extractor settings that shape individual-property outputs."""
    pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
    pose_model_type = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
    pose_model_dir = str(params.get("POSE_MODEL_DIR", "")).strip()
    pose_skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "enable_pose_extractor": pose_enabled,
        "pose_model_type": pose_model_type,
        "pose_min_kpt_conf_valid": params.get("POSE_MIN_KPT_CONF_VALID", 0.2),
        "pose_skeleton_file": (
            _file_fingerprint(pose_skeleton_file) if pose_skeleton_file else None
        ),
    }
    if pose_model_type == "sleap":
        payload["pose_sleap_env"] = params.get("POSE_SLEAP_ENV", "sleap")
        payload["pose_sleap_device"] = params.get("POSE_SLEAP_DEVICE", "auto")
        payload["pose_sleap_batch"] = params.get("POSE_SLEAP_BATCH", 4)
        payload["pose_sleap_max_instances"] = 1
    if pose_enabled:
        payload["pose_model"] = _file_fingerprint(pose_model_dir)
    return _hash_payload(payload)


def compute_individual_properties_id(
    detection_hash: str,
    filter_settings_hash: str,
    extractor_hash: str,
) -> str:
    """Canonical identity key for individual-properties cache artifacts."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection_hash": str(detection_hash),
        "filter_settings_hash": str(filter_settings_hash),
        "extractor_hash": str(extractor_hash),
    }
    return _hash_payload(payload)


class IndividualPropertiesCache:
    """NPZ-backed cache for per-detection properties keyed by frame and detection ID."""

    CACHE_VERSION = "1.0"

    def __init__(
        self,
        cache_path: str,
        mode: str = "w",
    ):
        self.cache_path = Path(cache_path)
        self.mode = mode
        self._data: Dict[str, np.ndarray] = {}
        self._loaded_data = None
        self._compatible = True
        self._cached_frames = set()
        self.metadata: Dict[str, Any] = {}

        if self.mode == "r" and self.cache_path.exists():
            self._loaded_data = np.load(str(self.cache_path), allow_pickle=True)
            meta = self._loaded_data.get("metadata", None)
            if meta is None:
                self._compatible = False
                self._cached_frames = set()
                return
            try:
                self.metadata = dict(meta.item())
            except Exception:
                self.metadata = {}
            if str(self.metadata.get("version", "")) != self.CACHE_VERSION:
                self._compatible = False
                self._loaded_data.close()
                self._loaded_data = None
                return
            self._cached_frames = self._extract_cached_frames()

    def _extract_cached_frames(self) -> set:
        if self._loaded_data is None:
            return set()
        frames = set()
        for key in self._loaded_data.files:
            if key.startswith("frame_") and key.endswith("_detection_ids"):
                try:
                    frames.add(int(key.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return frames

    def is_compatible(self) -> bool:
        return self._compatible

    def has_frame(self, frame_idx: int) -> bool:
        return int(frame_idx) in self._cached_frames

    def get_cached_frames(self) -> Iterable[int]:
        return sorted(self._cached_frames)

    def add_frame(
        self,
        frame_idx: int,
        detection_ids: List[float],
        pose_mean_conf: Optional[List[float]] = None,
        pose_valid_fraction: Optional[List[float]] = None,
        pose_num_valid: Optional[List[int]] = None,
        pose_num_keypoints: Optional[List[int]] = None,
        pose_keypoints: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write.")

        n = len(detection_ids)
        ids_arr = np.asarray(detection_ids, dtype=np.float64)
        mean_conf_arr = np.asarray(
            pose_mean_conf if pose_mean_conf is not None else [0.0] * n,
            dtype=np.float32,
        )
        valid_fraction_arr = np.asarray(
            pose_valid_fraction if pose_valid_fraction is not None else [0.0] * n,
            dtype=np.float32,
        )
        num_valid_arr = np.asarray(
            pose_num_valid if pose_num_valid is not None else [0] * n,
            dtype=np.int16,
        )
        num_kpts_arr = np.asarray(
            pose_num_keypoints if pose_num_keypoints is not None else [0] * n,
            dtype=np.int16,
        )

        if pose_keypoints is None:
            pose_keypoints = [None] * n
        pose_keypoints_arr = np.empty((n,), dtype=object)
        for i in range(n):
            kpts = pose_keypoints[i] if i < len(pose_keypoints) else None
            if kpts is None:
                pose_keypoints_arr[i] = None
            else:
                arr = np.asarray(kpts, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(
                        "pose_keypoints entries must be arrays of shape (K, 3)"
                    )
                pose_keypoints_arr[i] = arr

        frame_key = f"frame_{int(frame_idx):06d}"
        self._data[f"{frame_key}_detection_ids"] = ids_arr
        self._data[f"{frame_key}_pose_mean_conf"] = mean_conf_arr
        self._data[f"{frame_key}_pose_valid_fraction"] = valid_fraction_arr
        self._data[f"{frame_key}_pose_num_valid"] = num_valid_arr
        self._data[f"{frame_key}_pose_num_keypoints"] = num_kpts_arr
        self._data[f"{frame_key}_pose_keypoints"] = pose_keypoints_arr

    def save(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot save.")
        meta = {
            "version": self.CACHE_VERSION,
            "schema_version": SCHEMA_VERSION,
        }
        if metadata:
            meta.update(metadata)
        self._data["metadata"] = np.array(meta, dtype=object)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self.cache_path), **self._data)
        logger.info("Saved individual properties cache: %s", self.cache_path)
        self._data.clear()

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read.")
        if self._loaded_data is None:
            raise RuntimeError("No cache data loaded.")

        frame_key = f"frame_{int(frame_idx):06d}"
        ids_arr = self._loaded_data.get(f"{frame_key}_detection_ids", np.array([]))
        mean_arr = self._loaded_data.get(
            f"{frame_key}_pose_mean_conf", np.array([], dtype=np.float32)
        )
        frac_arr = self._loaded_data.get(
            f"{frame_key}_pose_valid_fraction", np.array([], dtype=np.float32)
        )
        num_valid_arr = self._loaded_data.get(
            f"{frame_key}_pose_num_valid", np.array([], dtype=np.int16)
        )
        num_kpts_arr = self._loaded_data.get(
            f"{frame_key}_pose_num_keypoints", np.array([], dtype=np.int16)
        )
        pose_arr = self._loaded_data.get(
            f"{frame_key}_pose_keypoints", np.array([], dtype=object)
        )

        return {
            "detection_ids": ids_arr.tolist(),
            "pose_mean_conf": mean_arr.tolist(),
            "pose_valid_fraction": frac_arr.tolist(),
            "pose_num_valid": num_valid_arr.tolist(),
            "pose_num_keypoints": num_kpts_arr.tolist(),
            "pose_keypoints": list(pose_arr.tolist()),
        }

    def get_detection(
        self, frame_idx: int, detection_id: float
    ) -> Optional[Dict[str, Any]]:
        frame = self.get_frame(frame_idx)
        ids = frame.get("detection_ids", [])
        try:
            target = int(detection_id)
        except Exception:
            return None
        for i, did in enumerate(ids):
            try:
                if int(did) != target:
                    continue
            except Exception:
                continue
            return {
                "detection_id": did,
                "pose_mean_conf": frame["pose_mean_conf"][i],
                "pose_valid_fraction": frame["pose_valid_fraction"][i],
                "pose_num_valid": frame["pose_num_valid"][i],
                "pose_num_keypoints": frame["pose_num_keypoints"][i],
                "pose_keypoints": frame["pose_keypoints"][i],
            }
        return None

    def close(self) -> None:
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None
