"""Unified precompute pipeline.

All phases (pose, AprilTag, CNN identity, ...) implement PrecomputePhase and
receive identical pre-extracted crops each frame via process_frame().

Single video read per tracking run regardless of how many phases are enabled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from multi_tracker.core.identity.apriltag_detector import AprilTagDetector
from multi_tracker.core.identity.cnn_identity import (
    ClassPrediction,
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
)
from multi_tracker.core.tracking.pose_pipeline import extract_one_crop
from multi_tracker.data.tag_observation_cache import TagObservationCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PrecomputePhase:
    """Protocol for a precompute phase.

    Implement all four methods plus ``name`` and ``is_fatal`` attributes.
    """

    name: str  # e.g. "pose", "apriltag", "cnn_identity"
    is_fatal: bool  # True → failure in finalize() aborts tracking

    def has_cache_hit(self) -> bool:
        """Return True if a valid existing cache covers this run.

        Called before the frame loop. If all phases return True, the video
        read is skipped entirely.
        """
        raise NotImplementedError

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        """Process one frame.

        Called with empty lists when no detections exist for the frame.
        Must be a no-op when has_cache_hit() returned True.

        Note: len(crops) <= len(all_obb). Not every OBB produces a crop (zero-size
        or invalid bounding boxes yield None from extract_one_crop and are skipped).
        detection_ids gives the full detection-ID list for the frame, aligned to
        all_obb. crop_det_indices[i] gives the detection-slot index into all_obb
        for crops[i].

        canonical_affines: When canonical crops are enabled, parallel list of
        M_inverse (2x3 float32) arrays for each crop, or None per entry if
        unavailable.  Phases should use ``invert_keypoints(kpts, M_inverse)``
        instead of offset addition when an affine is present.
        """
        raise NotImplementedError

    def finalize(self) -> Optional[str]:
        """Flush caches and return artifact path, or None on failure.

        Runs to completion — stop_check is NOT called inside finalize().
        Non-fatal phases should catch their own exceptions and return None.
        Fatal phases should re-raise; UnifiedPrecompute propagates the exception.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Release backend/model resources.

        Called after finalize(), or after cancellation (no finalize() in that case).
        Must be idempotent.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CropConfig
# ---------------------------------------------------------------------------


@dataclass
class CropConfig:
    """Controls shared crop extraction in UnifiedPrecompute."""

    padding_fraction: float = 0.1  # maps to INDIVIDUAL_CROP_PADDING
    suppress_foreign: bool = True  # maps to SUPPRESS_FOREIGN_OBB_REGIONS
    bg_color: Tuple[int, int, int] = (0, 0, 0)  # maps to INDIVIDUAL_BACKGROUND_COLOR
    canonical_crop_width: int = 0  # 0 = disabled (legacy AABB path)
    canonical_crop_height: int = 0  # derived from reference_aspect_ratio


# ---------------------------------------------------------------------------
# UnifiedPrecompute
# ---------------------------------------------------------------------------


class UnifiedPrecompute:
    """Single-pass precompute orchestrator.

    Reads each video frame once, extracts crops once, and dispatches to all
    registered phases. Phases are responsible for their own batching, cache
    writes, and resource lifecycle.
    """

    def __init__(
        self,
        phases: List[PrecomputePhase],
        crop_config: CropConfig,
    ) -> None:
        self._phases = phases
        self._crop_config = crop_config

    def run(
        self,
        cap: cv2.VideoCapture,
        detection_cache,
        detector,
        start_frame: int,
        end_frame: int,
        resize_factor: float,
        roi_mask,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        warning_cb: Optional[Callable[[str, str], None]] = None,
    ) -> Dict[str, Optional[str]]:
        """Run all phases over [start_frame, end_frame].

        Returns {phase.name: cache_path_or_none} for every phase.
        """
        if not self._phases:
            return {}

        # --- cache-hit short-circuit ---
        hits = [p.has_cache_hit() for p in self._phases]
        if all(hits):
            results: Dict[str, Optional[str]] = {}
            try:
                for p in self._phases:
                    try:
                        results[p.name] = p.finalize()
                    except Exception as exc:
                        if p.is_fatal:
                            raise
                        logger.warning(
                            "Precompute phase '%s' finalize failed (cache-hit path): %s",
                            p.name,
                            exc,
                        )
                        if warning_cb:
                            warning_cb(f"Precompute Warning ({p.name})", str(exc))
                        results[p.name] = None
            finally:
                for p in self._phases:
                    try:
                        p.close()
                    except Exception:
                        pass
            return results

        total = max(1, end_frame - start_frame + 1)
        cfg = self._crop_config

        # --- frame loop ---
        cancelled = False
        for rel_idx in range(total):
            frame_idx = start_frame + rel_idx

            # read + optional resize
            ret, frame = cap.read()
            if not ret:
                logger.warning(
                    "cap.read() failed at frame %d (frame %d/%d) — stopping precompute early",
                    frame_idx,
                    rel_idx + 1,
                    total,
                )
                break
            if resize_factor < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_factor,
                    fy=resize_factor,
                    interpolation=cv2.INTER_AREA,
                )

            # get raw detections
            try:
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_ids,
                    raw_headings,
                    raw_directed,
                    raw_canonical_affines,
                ) = detection_cache.get_frame(frame_idx)
            except Exception:
                raw_meas = raw_sizes = raw_shapes = raw_confs = []
                raw_obb = raw_ids = raw_headings = raw_directed = []
                raw_canonical_affines = None

            # filter detections (ROI mask, size/confidence gates)
            (
                _meas,
                _sz,
                _sh,
                _cf,
                filt_obb,
                det_ids,
                _hd,
                _dm,
            ) = detector.filter_raw_detections(
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confs,
                raw_obb,
                roi_mask=roi_mask,
                detection_ids=raw_ids,
                heading_hints=raw_headings,
                directed_mask=raw_directed,
            )

            all_obb = [np.asarray(c, dtype=np.float32) for c in (filt_obb or [])]

            # --- Determine canonical extraction availability ---
            use_canonical = (
                cfg.canonical_crop_width > 0
                and cfg.canonical_crop_height > 0
                and raw_canonical_affines is not None
            )

            # Map raw detection IDs → raw indices so we can filter affines
            filtered_affines: Optional[List[Optional[np.ndarray]]] = None
            if use_canonical and raw_ids:
                raw_id_to_idx: Dict[int, int] = {}
                for idx, rid in enumerate(raw_ids):
                    raw_id_to_idx[int(rid)] = idx
                filtered_affines = []
                for did in det_ids:
                    raw_idx = raw_id_to_idx.get(int(did))
                    if (
                        raw_idx is not None
                        and raw_idx < len(raw_canonical_affines)
                        and raw_canonical_affines[raw_idx] is not None
                    ):
                        filtered_affines.append(raw_canonical_affines[raw_idx])
                    else:
                        filtered_affines.append(None)

            # --- Extract crops ---
            aabb_crops: List[np.ndarray] = []
            aabb_offsets: List[Tuple[int, int]] = []
            canonical_crops: List[np.ndarray] = []
            canonical_M_inv: List[Optional[np.ndarray]] = []
            crop_det_indices: List[int] = []

            if all_obb:
                for di, corners in enumerate(all_obb):
                    # Always extract AABB crop (for AprilTag + fallback)
                    aabb_result = extract_one_crop(
                        frame,
                        corners,
                        di,
                        cfg.padding_fraction,
                        all_obb,
                        cfg.suppress_foreign,
                        cfg.bg_color,
                    )
                    if aabb_result is None:
                        continue

                    aabb_crop, offset, _ = aabb_result
                    aabb_crops.append(aabb_crop)
                    aabb_offsets.append((int(offset[0]), int(offset[1])))
                    crop_det_indices.append(di)

                    # Extract canonical crop when affine is available
                    M_can = (
                        filtered_affines[di]
                        if (filtered_affines and di < len(filtered_affines))
                        else None
                    )
                    if use_canonical and M_can is not None:
                        from multi_tracker.core.tracking.canonical_crop import (
                            extract_canonical_crop,
                        )

                        c_crop = extract_canonical_crop(
                            frame,
                            M_can,
                            cfg.canonical_crop_width,
                            cfg.canonical_crop_height,
                            bg_color=cfg.bg_color,
                        )
                        M_inv = cv2.invertAffineTransform(
                            np.asarray(M_can, dtype=np.float64)
                        )
                        canonical_crops.append(c_crop)
                        canonical_M_inv.append(M_inv.astype(np.float32))
                    else:
                        # Fallback: use AABB crop, no affine
                        canonical_crops.append(aabb_crop)
                        canonical_M_inv.append(None)

            frame_detection_ids = [
                int(det_ids[i]) if det_ids and i < len(det_ids) else i
                for i in range(len(all_obb))
            ]

            # --- Fan-out to all phases ---
            # Phases that prefer AABB crops (e.g. AprilTag) get aabb_crops + offsets.
            # Other phases get canonical crops + M_inverse affines.
            for phase in self._phases:
                if getattr(phase, "_prefer_aabb_crops", False):
                    phase.process_frame(
                        frame_idx,
                        aabb_crops,
                        frame_detection_ids,
                        crop_det_indices,
                        all_obb,
                        aabb_offsets,
                    )
                else:
                    phase.process_frame(
                        frame_idx,
                        canonical_crops if use_canonical else aabb_crops,
                        frame_detection_ids,
                        crop_det_indices,
                        all_obb,
                        aabb_offsets,
                        canonical_affines=(canonical_M_inv if use_canonical else None),
                    )

            # cancellation check — AFTER processing the frame
            if stop_check and stop_check():
                cancelled = True
                break

            # progress
            if progress_cb and (rel_idx % 50 == 0 or rel_idx == total - 1):
                pct = int((rel_idx + 1) * 100 / total)
                progress_cb(pct, f"Precompute: {rel_idx + 1}/{total} frames")

        # --- cancellation: skip finalize, just close ---
        if cancelled:
            for p in self._phases:
                try:
                    p.close()
                except Exception:
                    pass
            return {p.name: None for p in self._phases}

        # --- finalize all phases ---
        results = {}
        try:
            for p in self._phases:
                try:
                    results[p.name] = p.finalize()
                except Exception as exc:
                    if p.is_fatal:
                        raise
                    logger.warning(
                        "Precompute phase '%s' finalize failed: %s", p.name, exc
                    )
                    if warning_cb:
                        warning_cb(
                            f"Precompute Warning ({p.name})",
                            str(exc),
                        )
                    results[p.name] = None
        finally:
            for p in self._phases:
                try:
                    p.close()
                except Exception:
                    pass

        return results


# ---------------------------------------------------------------------------
# AprilTagPrecomputePhase
# ---------------------------------------------------------------------------


class AprilTagPrecomputePhase(PrecomputePhase):
    """Precompute phase that runs AprilTag detection on per-frame crops.

    Non-fatal: a detection failure will not abort the tracking run.
    If a compatible cache already exists for the requested frame range, the
    detector is never created and the video read is skipped.
    """

    name = "apriltag"
    is_fatal = False
    _prefer_aabb_crops = True  # AprilTag needs AABB crops for precise edge detection

    def __init__(
        self,
        detector_config,
        cache_path,
        start_frame: int,
        end_frame: int,
        video_path: str = "",
    ) -> None:
        self._cache_path = Path(cache_path)
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._video_path = video_path
        self._hit = False
        self._detector: Optional[AprilTagDetector] = None
        self._tag_cache: Optional[TagObservationCache] = None

        # Check for a compatible existing cache.
        if self._cache_path.exists():
            probe = TagObservationCache(
                self._cache_path, mode="r", start_frame=start_frame, end_frame=end_frame
            )
            try:
                if probe.is_compatible() and probe.covers_frame_range(
                    start_frame, end_frame
                ):
                    self._hit = True
                    logger.info(
                        "AprilTag cache hit: %s covers frames %d-%d",
                        self._cache_path,
                        start_frame,
                        end_frame,
                    )
                    return
                # fall through to create detector + write-mode cache
                logger.info(
                    "AprilTag cache miss or incompatible at %s — will regenerate",
                    self._cache_path,
                )
            finally:
                probe.close()

        # Cache miss: create detector and write-mode cache.
        self._detector = AprilTagDetector(detector_config)
        self._tag_cache = TagObservationCache(
            self._cache_path, mode="w", start_frame=start_frame, end_frame=end_frame
        )

    # ------------------------------------------------------------------
    # PrecomputePhase interface
    # ------------------------------------------------------------------

    def has_cache_hit(self) -> bool:
        return self._hit

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        if self._hit:
            return
        if self._tag_cache is None:
            return

        if not crops:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[],
                centers_xy=[],
                corners=[],
                det_indices=[],
                hammings=[],
            )
            return

        observations = self._detector.detect_in_crops(
            crops, crop_offsets, det_indices=crop_det_indices
        )

        if not observations:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[],
                centers_xy=[],
                corners=[],
                det_indices=[],
                hammings=[],
            )
        else:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[obs.tag_id for obs in observations],
                centers_xy=[obs.center_xy for obs in observations],
                corners=[obs.corners for obs in observations],
                det_indices=[obs.det_index for obs in observations],
                hammings=[obs.hamming for obs in observations],
            )

    def finalize(self) -> Optional[str]:
        if self._hit:
            return str(self._cache_path)
        if self._tag_cache is None:
            return None
        try:
            self._tag_cache.save(
                metadata={
                    "start_frame": self._start_frame,
                    "end_frame": self._end_frame,
                    "video_path": self._video_path,
                }
            )
            self._tag_cache.close()
            self._tag_cache = None
            return str(self._cache_path)
        except Exception:
            logger.exception("AprilTagPrecomputePhase.finalize() failed")
            return None

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._tag_cache is not None:
            try:
                self._tag_cache.close()
            except Exception:
                pass
            self._tag_cache = None


# ---------------------------------------------------------------------------
# CNNPrecomputePhase
# ---------------------------------------------------------------------------


class CNNPrecomputePhase(PrecomputePhase):
    """Precompute phase that runs CNN identity classification on OBB crops.

    Supports multiple instances with different model configs and names (e.g.
    one for individual identity, another for age classification).
    """

    is_fatal = False

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str,
        cache_path,
        compute_runtime: str = "cpu",
        name: str = "cnn_identity",
    ) -> None:
        self.name = name
        self._cache_path = Path(cache_path)
        self._cfg = config
        self._hit = self._cache_path.exists()
        self._closed = False

        # accumulator for batching
        self._pending_crops: List[np.ndarray] = []
        self._pending_frame_idx: List[int] = []
        self._pending_det_ids: List[int] = []

        if not self._hit:
            self._backend = CNNIdentityBackend(
                config, model_path=model_path, compute_runtime=compute_runtime
            )
            self._cache = CNNIdentityCache(str(self._cache_path))
        else:
            self._backend = None
            self._cache = None

    def has_cache_hit(self) -> bool:
        return self._hit

    def process_frame(
        self,
        frame_idx: int,
        crops: List[np.ndarray],
        detection_ids: List[int],
        crop_det_indices: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
        *,
        canonical_affines: Optional[List[Optional[np.ndarray]]] = None,
    ) -> None:
        if self._hit or self._cache is None:
            return
        if not crops:
            self._cache.save(frame_idx, [])
            return
        for crop, det_idx in zip(crops, crop_det_indices):
            self._pending_crops.append(crop)
            self._pending_frame_idx.append(frame_idx)
            self._pending_det_ids.append(det_idx)

        if len(self._pending_crops) >= self._cfg.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        if not self._pending_crops or self._backend is None or self._cache is None:
            return
        preds = self._backend.predict_batch(self._pending_crops)
        # group by frame
        frame_preds: Dict[int, List[ClassPrediction]] = {}
        for pred, frame_idx, det_id in zip(
            preds, self._pending_frame_idx, self._pending_det_ids
        ):
            pred.det_index = det_id
            frame_preds.setdefault(frame_idx, []).append(pred)
        for fid, fps in frame_preds.items():
            self._cache.save(fid, fps)
        self._pending_crops.clear()
        self._pending_frame_idx.clear()
        self._pending_det_ids.clear()

    def finalize(self) -> Optional[str]:
        if self._hit:
            return str(self._cache_path)
        if self._backend is None or self._cache is None:
            return None
        self._flush_batch()  # flush any remaining partial batch
        self._cache.flush()  # single np.savez_compressed write
        logger.info("CNN identity cache written: %s", self._cache_path)
        return str(self._cache_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                pass
            self._backend = None
