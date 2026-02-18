"""
Caching utilities for per-detection appearance embeddings.

This cache is keyed by:
  - detection hash (inference identity + video/range fingerprint)
  - filter settings hash
  - appearance extractor settings hash
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"

# Re-use normalization and hashing functions from properties_cache
from .properties_cache import _hash_payload


def compute_appearance_extractor_hash(params: Dict[str, Any]) -> str:
    """Hash appearance extractor settings.

    Args:
        params: Dictionary containing:
            - APPEARANCE_ENABLED: bool
            - APPEARANCE_MODEL_NAME: str (e.g., "timm/vit_base_patch14_dinov2.lvd142m")
            - APPEARANCE_RUNTIME_FLAVOR: str (auto/native/onnx/tensorrt)
            - APPEARANCE_BATCH_SIZE: int
            - APPEARANCE_MAX_IMAGE_SIDE: int
            - APPEARANCE_USE_CLAHE: bool
            - APPEARANCE_NORMALIZE: bool (optional, default True)
    """
    appearance_enabled = bool(params.get("APPEARANCE_ENABLED", False))
    appearance_model_name = str(params.get("APPEARANCE_MODEL_NAME", "")).strip()
    appearance_runtime_flavor = (
        str(params.get("APPEARANCE_RUNTIME_FLAVOR", "auto")).strip().lower()
    )
    appearance_batch_size = int(params.get("APPEARANCE_BATCH_SIZE", 32))
    appearance_max_image_side = int(params.get("APPEARANCE_MAX_IMAGE_SIDE", 512))
    appearance_use_clahe = bool(params.get("APPEARANCE_USE_CLAHE", False))
    appearance_normalize = bool(params.get("APPEARANCE_NORMALIZE", True))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "appearance_enabled": appearance_enabled,
        "appearance_model_name": appearance_model_name,
        "appearance_runtime_flavor": appearance_runtime_flavor,
        "appearance_batch_size": appearance_batch_size,
        "appearance_max_image_side": appearance_max_image_side,
        "appearance_use_clahe": appearance_use_clahe,
        "appearance_normalize": appearance_normalize,
    }

    return _hash_payload(payload)


def compute_appearance_embedding_id(
    detection_hash: str,
    filter_settings_hash: str,
    appearance_extractor_hash: str,
) -> str:
    """Canonical identity key for appearance embedding cache artifacts."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "detection_hash": str(detection_hash),
        "filter_settings_hash": str(filter_settings_hash),
        "appearance_extractor_hash": str(appearance_extractor_hash),
    }
    return _hash_payload(payload)


class AppearanceEmbeddingCache:
    """NPZ-backed cache for per-detection appearance embeddings."""

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
        embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Add a frame of appearance embeddings to cache.

        Args:
            frame_idx: Frame index
            detection_ids: List of detection IDs for this frame
            embeddings: List of embedding vectors (1D arrays) or None for failed extractions
        """
        if self.mode != "w":
            raise RuntimeError("Cache opened in read mode, cannot write.")

        n = len(detection_ids)
        ids_arr = np.asarray(detection_ids, dtype=np.float64)

        if embeddings is None:
            embeddings = [None] * n

        # Store embeddings as object array to handle None values
        embeddings_arr = np.empty((n,), dtype=object)
        for i in range(n):
            emb = embeddings[i] if i < len(embeddings) else None
            if emb is None:
                embeddings_arr[i] = None
            else:
                arr = np.asarray(emb, dtype=np.float32)
                if arr.ndim != 1:
                    raise ValueError(
                        f"Embedding must be 1D array, got shape {arr.shape}"
                    )
                embeddings_arr[i] = arr

        frame_key = f"frame_{int(frame_idx):06d}"
        self._data[f"{frame_key}_detection_ids"] = ids_arr
        self._data[f"{frame_key}_embeddings"] = embeddings_arr

    def save(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save cache to disk with metadata."""
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
        logger.info("Saved appearance embedding cache: %s", self.cache_path)
        self._data.clear()

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """Get frame data with embeddings.

        Args:
            frame_idx: Frame index to retrieve

        Returns:
            Dict with detection_ids and embeddings (list of 1D arrays or None)
        """
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read.")
        if self._loaded_data is None:
            raise RuntimeError("No cache data loaded.")

        frame_key = f"frame_{int(frame_idx):06d}"
        ids_arr = self._loaded_data.get(f"{frame_key}_detection_ids", np.array([]))
        embeddings_arr = self._loaded_data.get(
            f"{frame_key}_embeddings", np.array([], dtype=object)
        )

        return {
            "detection_ids": ids_arr.tolist(),
            "embeddings": list(embeddings_arr.tolist()),
        }

    def get_embeddings(
        self, frame_idx: int, detection_id: float
    ) -> Optional[np.ndarray]:
        """Get embedding for a specific detection.

        Args:
            frame_idx: Frame index
            detection_id: Detection ID

        Returns:
            Embedding array (1D) or None if not found or failed extraction
        """
        frame = self.get_frame(frame_idx)
        ids = frame.get("detection_ids", [])
        embeddings = frame.get("embeddings", [])

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
            if i < len(embeddings):
                return embeddings[i]
            return None

        return None

    def get_all_embeddings(
        self, exclude_none: bool = True
    ) -> tuple[np.ndarray, List[tuple[int, float]]]:
        """Get all embeddings from cache.

        Args:
            exclude_none: If True, skip None embeddings

        Returns:
            Tuple of (embeddings_matrix, metadata_list) where:
                - embeddings_matrix: (N, D) array of embeddings
                - metadata_list: List of (frame_idx, detection_id) tuples
        """
        if self.mode != "r":
            raise RuntimeError("Cache opened in write mode, cannot read.")
        if self._loaded_data is None:
            raise RuntimeError("No cache data loaded.")

        all_embeddings = []
        all_metadata = []

        for frame_idx in self.get_cached_frames():
            frame_data = self.get_frame(frame_idx)
            detection_ids = frame_data["detection_ids"]
            embeddings = frame_data["embeddings"]

            for det_id, emb in zip(detection_ids, embeddings):
                if emb is None and exclude_none:
                    continue
                if emb is not None:
                    all_embeddings.append(np.asarray(emb, dtype=np.float32))
                    all_metadata.append((int(frame_idx), float(det_id)))

        if not all_embeddings:
            embedding_dim = int(self.metadata.get("embedding_dimension", 0))
            return np.zeros((0, embedding_dim), dtype=np.float32), []

        embeddings_matrix = np.stack(all_embeddings, axis=0)
        return embeddings_matrix, all_metadata

    def close(self) -> None:
        """Close the cache file."""
        if self._loaded_data is not None:
            self._loaded_data.close()
            self._loaded_data = None


def build_appearance_cache_path(
    out_root: str,
    video_stem: str,
    embedding_id: str,
    start_frame: int,
    end_frame: int,
) -> Path:
    """Build deterministic cache path for appearance embeddings.

    Args:
        out_root: Output directory root
        video_stem: Video filename stem (without extension)
        embedding_id: Combined hash of detection/filter/appearance settings
        start_frame: Start frame index
        end_frame: End frame index

    Returns:
        Path to cache file
    """
    cache_dir = Path(out_root) / "appearance_cache"
    cache_filename = (
        f"{video_stem}_appearance_{embedding_id}_{start_frame}_{end_frame}.npz"
    )
    return cache_dir / cache_filename
