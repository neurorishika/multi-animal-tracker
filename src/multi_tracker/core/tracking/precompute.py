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
        det_ids: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
    ) -> None:
        """Process one frame.

        Called with empty lists when no detections exist for the frame.
        Must be a no-op when has_cache_hit() returned True.

        Note: len(crops) <= len(all_obb). Not every OBB produces a crop (zero-size
        or invalid bounding boxes yield None from extract_one_crop and are skipped).
        crop_det_ids[i] gives the detection index into all_obb for crops[i].
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
                ) = detection_cache.get_frame(frame_idx)
            except Exception:
                raw_meas = raw_sizes = raw_shapes = raw_confs = []
                raw_obb = raw_ids = raw_headings = raw_directed = []

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

            # extract crops ONCE — shared across all phases
            crops: List[np.ndarray] = []
            crop_det_ids: List[int] = []
            crop_offsets: List[Tuple[int, int]] = []

            if all_obb:
                for di, corners in enumerate(all_obb):
                    result = extract_one_crop(
                        frame,
                        corners,
                        di,
                        cfg.padding_fraction,
                        all_obb,
                        cfg.suppress_foreign,
                        cfg.bg_color,
                    )
                    if result is not None:
                        crop, offset, _ = result
                        crops.append(crop)
                        crop_det_ids.append(
                            det_ids[di] if det_ids and di < len(det_ids) else di
                        )
                        crop_offsets.append((int(offset[0]), int(offset[1])))

            # fan-out to all phases
            for phase in self._phases:
                phase.process_frame(
                    frame_idx, crops, crop_det_ids, all_obb, crop_offsets
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
            if probe.is_compatible() and probe.covers_frame_range(
                start_frame, end_frame
            ):
                self._hit = True
                probe.close()
                logger.info(
                    "AprilTag cache hit: %s covers frames %d-%d",
                    self._cache_path,
                    start_frame,
                    end_frame,
                )
                return
            probe.close()
            logger.info(
                "AprilTag cache miss or incompatible at %s — will regenerate",
                self._cache_path,
            )

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
        det_ids: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
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
            crops, crop_offsets, det_indices=det_ids
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
