# Unified Precompute Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 3 sequential full-video-read precompute phases (pose, AprilTag, CNN identity) with a single video pass that reads each frame once, extracts crops once via `extract_one_crop()`, and fans out to all enabled phases via the `PrecomputePhase` protocol.

**Architecture:** New `core/tracking/precompute.py` defines a `PrecomputePhase` protocol, `CropConfig` dataclass, `UnifiedPrecompute` orchestrator, `AprilTagPrecomputePhase`, and `CNNPrecomputePhase`. `PosePipeline` gains `has_cache_hit()`, `process_frame()`, `finalize()` methods and a guarded `close()`. `worker.py` replaces 3 precompute blocks with `_build_precompute_phases()` + `UnifiedPrecompute.run()`.

**Tech Stack:** Python 3.10+, OpenCV, NumPy, PyTorch (pose backend), PyQt5, pytest/unittest.mock

---

## File Structure

| File | Role |
|---|---|
| `src/multi_tracker/core/tracking/precompute.py` | **New** — `PrecomputePhase` protocol, `CropConfig`, `UnifiedPrecompute`, `AprilTagPrecomputePhase`, `CNNPrecomputePhase` |
| `src/multi_tracker/core/tracking/pose_pipeline.py` | Add `has_cache_hit()`, `process_frame()`, `finalize()` methods; update `close()` with idempotency guard |
| `src/multi_tracker/core/tracking/worker.py` | Delete 3 precompute methods + 3 `_should_precompute_*` helpers; add `_build_precompute_phases()`; wire `UnifiedPrecompute` |
| `src/multi_tracker/core/identity/cnn_identity.py` | Remove `crop_padding` field from `CNNIdentityConfig` |
| `configs/default.json` | Remove `cnn_classifier_crop_padding` key |
| `src/multi_tracker/gui/main_window.py` | Remove `spin_cnn_crop_padding` widget + wiring; update `INDIVIDUAL_CROP_PADDING` label |
| `tests/test_unified_precompute.py` | **New** — all unified precompute unit tests |
| `tests/test_mat_cnn_identity.py` | Remove `crop_padding` assertion from config defaults test |

---

## Task 1: Remove `crop_padding` from `CNNIdentityConfig`

**Files:**
- Modify: `src/multi_tracker/core/identity/cnn_identity.py:26-36`
- Modify: `tests/test_mat_cnn_identity.py`

- [ ] **Step 1: Update the test to remove the `crop_padding` assertion**

In `tests/test_mat_cnn_identity.py`, find the test that checks `cfg.crop_padding == 0.1` and remove that line. Run all CNN identity tests to confirm they still pass:

```bash
python -m pytest tests/test_mat_cnn_identity.py -v
```

Expected: all tests pass (one assertion removed, no new failures).

- [ ] **Step 2: Remove `crop_padding` from `CNNIdentityConfig`**

In `src/multi_tracker/core/identity/cnn_identity.py`, remove this line from the dataclass:

```python
crop_padding: float = 0.1
```

The dataclass after the change:

```python
@dataclass
class CNNIdentityConfig:
    """Configuration for CNN Classifier identity method."""

    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    match_bonus: float = 20.0
    mismatch_penalty: float = 50.0
    window: int = 10
```

- [ ] **Step 3: Run tests to confirm nothing broke**

```bash
python -m pytest tests/test_mat_cnn_identity.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/multi_tracker/core/identity/cnn_identity.py tests/test_mat_cnn_identity.py
git commit -m "refactor: remove crop_padding from CNNIdentityConfig — centralized in UnifiedPrecompute"
```

---

## Task 2: Create `precompute.py` — Protocol, CropConfig, UnifiedPrecompute

**Files:**
- Create: `src/multi_tracker/core/tracking/precompute.py`
- Create: `tests/test_unified_precompute.py`

- [ ] **Step 1: Write failing tests for `CropConfig` and `UnifiedPrecompute`**

Create `tests/test_unified_precompute.py`:

```python
"""Tests for the unified precompute pipeline."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, Mock, call, patch

from multi_tracker.core.tracking.precompute import CropConfig, UnifiedPrecompute


# ---------------------------------------------------------------------------
# CropConfig
# ---------------------------------------------------------------------------


def test_crop_config_defaults():
    cfg = CropConfig()
    assert cfg.padding_fraction == 0.1
    assert cfg.suppress_foreign is True
    assert cfg.bg_color == (0, 0, 0)


def test_crop_config_custom():
    cfg = CropConfig(padding_fraction=0.2, suppress_foreign=False, bg_color=(255, 0, 0))
    assert cfg.padding_fraction == 0.2
    assert cfg.suppress_foreign is False
    assert cfg.bg_color == (255, 0, 0)


# ---------------------------------------------------------------------------
# UnifiedPrecompute helpers
# ---------------------------------------------------------------------------


def _make_phase(name, is_fatal=False, cache_hit=False, finalize_return=None):
    """Build a mock PrecomputePhase."""
    p = Mock()
    p.name = name
    p.is_fatal = is_fatal
    p.has_cache_hit.return_value = cache_hit
    p.finalize.return_value = finalize_return
    return p


def _make_cap(frame=None):
    cap = Mock()
    if frame is None:
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


def _make_det_cache():
    dc = Mock()
    dc.get_frame.return_value = ([], [], [], [], [], [], [], [])
    return dc


def _make_detector():
    det = Mock()
    det.filter_raw_detections.return_value = ([], [], [], [], [], [], [], [])
    return det


# ---------------------------------------------------------------------------
# Empty phase list
# ---------------------------------------------------------------------------


def test_empty_phases_returns_empty_dict_without_reading_cap():
    cap = _make_cap()
    up = UnifiedPrecompute([], CropConfig())
    result = up.run(
        cap=cap,
        detection_cache=_make_det_cache(),
        detector=_make_detector(),
        start_frame=0,
        end_frame=5,
        resize_factor=1.0,
        roi_mask=None,
    )
    assert result == {}
    cap.read.assert_not_called()


# ---------------------------------------------------------------------------
# Dispatch — process_frame called once per frame per phase
# ---------------------------------------------------------------------------


def test_process_frame_called_once_per_frame_per_phase():
    p1 = _make_phase("a", finalize_return="/a")
    p2 = _make_phase("b", finalize_return="/b")

    up = UnifiedPrecompute([p1, p2], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 2, 1.0, None)

    assert p1.process_frame.call_count == 3  # frames 0, 1, 2
    assert p2.process_frame.call_count == 3


# ---------------------------------------------------------------------------
# Empty detections → process_frame called with empty lists
# ---------------------------------------------------------------------------


def test_process_frame_called_with_empty_crops_when_no_detections():
    p = _make_phase("x", finalize_return=None)
    up = UnifiedPrecompute([p], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    args = p.process_frame.call_args[0]  # positional args
    # args: (frame_idx, crops, det_ids, all_obb, crop_offsets)
    frame_idx, crops, det_ids, all_obb, crop_offsets = args
    assert frame_idx == 0
    assert crops == []
    assert det_ids == []
    assert all_obb == []
    assert crop_offsets == []


# ---------------------------------------------------------------------------
# All-cache-hit: video read skipped
# ---------------------------------------------------------------------------


def test_all_cache_hit_skips_video_read():
    p = _make_phase("x", cache_hit=True, finalize_return="/cached")
    cap = _make_cap()
    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(cap, _make_det_cache(), _make_detector(), 0, 5, 1.0, None)
    cap.read.assert_not_called()
    assert result == {"x": "/cached"}


# ---------------------------------------------------------------------------
# Partial cache hit: video runs; hit phase process_frame no-op, miss phase runs
# ---------------------------------------------------------------------------


def test_partial_cache_hit_video_runs_miss_phase_gets_frames():
    hit = _make_phase("hit", cache_hit=True, finalize_return="/old")
    miss = _make_phase("miss", cache_hit=False, finalize_return="/new")

    up = UnifiedPrecompute([hit, miss], CropConfig())
    result = up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 1, 1.0, None)

    assert miss.process_frame.call_count == 2  # both frames processed
    assert result == {"hit": "/old", "miss": "/new"}


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_stop_check_exits_loop_calls_close_not_finalize():
    p = _make_phase("y")
    cap = _make_cap()

    call_count = [0]

    def stop_after_first():
        call_count[0] += 1
        return call_count[0] > 1

    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(
        cap, _make_det_cache(), _make_detector(), 0, 10, 1.0, None,
        stop_check=stop_after_first,
    )
    p.finalize.assert_not_called()
    p.close.assert_called_once()
    assert result == {"y": None}


# ---------------------------------------------------------------------------
# Fatal phase raises in finalize → propagates; close still called
# ---------------------------------------------------------------------------


def test_fatal_phase_finalize_raises_propagates_and_close_called():
    p = _make_phase("pose", is_fatal=True)
    p.finalize.side_effect = RuntimeError("inference failed")

    up = UnifiedPrecompute([p], CropConfig())
    with pytest.raises(RuntimeError, match="inference failed"):
        up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    p.close.assert_called_once()


# ---------------------------------------------------------------------------
# Non-fatal phase raises in finalize → None in result; warning_cb called; close called
# ---------------------------------------------------------------------------


def test_nonfatal_phase_finalize_raises_returns_none_and_warns():
    p = _make_phase("apriltag", is_fatal=False)
    p.finalize.side_effect = RuntimeError("at failed")
    warnings = []

    up = UnifiedPrecompute([p], CropConfig())
    result = up.run(
        _make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None,
        warning_cb=lambda t, m: warnings.append((t, m)),
    )
    assert result == {"apriltag": None}
    assert len(warnings) == 1
    p.close.assert_called_once()


# ---------------------------------------------------------------------------
# finalize + close called on ALL phases even when one raises
# ---------------------------------------------------------------------------


def test_close_called_on_all_phases_when_one_finalize_raises():
    p1 = _make_phase("ok", finalize_return="/ok")
    p2 = _make_phase("bad", is_fatal=False)
    p2.finalize.side_effect = RuntimeError("oops")

    up = UnifiedPrecompute([p1, p2], CropConfig())
    up.run(_make_cap(), _make_det_cache(), _make_detector(), 0, 0, 1.0, None)

    p1.close.assert_called_once()
    p2.close.assert_called_once()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_unified_precompute.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` for `precompute`.

- [ ] **Step 3: Implement `precompute.py`**

Create `src/multi_tracker/core/tracking/precompute.py`:

```python
"""Unified precompute pipeline.

All phases (pose, AprilTag, CNN identity, ...) implement PrecomputePhase and
receive identical pre-extracted crops each frame via process_frame().

Single video read per tracking run regardless of how many phases are enabled.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from multi_tracker.core.tracking.pose_pipeline import extract_one_crop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PrecomputePhase:
    """Protocol for a precompute phase.

    Implement all four methods plus ``name`` and ``is_fatal`` attributes.
    """

    name: str        # e.g. "pose", "apriltag", "cnn_identity"
    is_fatal: bool   # True → failure in finalize() aborts tracking

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

    padding_fraction: float = 0.1            # maps to INDIVIDUAL_CROP_PADDING
    suppress_foreign: bool = True            # maps to SUPPRESS_FOREIGN_OBB_REGIONS
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
                    results[p.name] = p.finalize()
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

            if stop_check and stop_check():
                cancelled = True
                break

            # read + optional resize
            ret, frame = cap.read()
            if ret and resize_factor < 1.0:
                frame = cv2.resize(
                    frame, (0, 0),
                    fx=resize_factor, fy=resize_factor,
                    interpolation=cv2.INTER_AREA,
                )

            # get raw detections
            try:
                (
                    raw_meas, raw_sizes, raw_shapes, raw_confs,
                    raw_obb, raw_ids, raw_headings, raw_directed,
                ) = detection_cache.get_frame(frame_idx)
            except Exception:
                raw_meas = raw_sizes = raw_shapes = raw_confs = []
                raw_obb = raw_ids = raw_headings = raw_directed = []

            # filter detections (ROI mask, size/confidence gates)
            (
                _meas, _sz, _sh, _cf,
                filt_obb, det_ids, _hd, _dm,
            ) = detector.filter_raw_detections(
                raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb,
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

            if ret and frame is not None and all_obb:
                for di, corners in enumerate(all_obb):
                    result = extract_one_crop(
                        frame, corners, di,
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
                        crop_offsets.append(offset)

            # fan-out to all phases
            for phase in self._phases:
                phase.process_frame(
                    frame_idx, crops, crop_det_ids, all_obb, crop_offsets
                )

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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_unified_precompute.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/precompute.py tests/test_unified_precompute.py
git commit -m "feat: add PrecomputePhase protocol, CropConfig, and UnifiedPrecompute orchestrator"
```

---

## Task 3: `AprilTagPrecomputePhase`

**Files:**
- Modify: `src/multi_tracker/core/tracking/precompute.py` (append class)
- Modify: `tests/test_unified_precompute.py` (append tests)

- [ ] **Step 1: Write failing tests for `AprilTagPrecomputePhase`**

Append to `tests/test_unified_precompute.py`:

```python
# ---------------------------------------------------------------------------
# AprilTagPrecomputePhase
# ---------------------------------------------------------------------------

from pathlib import Path


def test_apriltag_phase_cache_hit_returns_true_when_cache_valid(tmp_path):
    """has_cache_hit() returns True when a compatible cache file exists."""
    from multi_tracker.core.tracking.precompute import AprilTagPrecomputePhase

    cache_path = tmp_path / "tags_0_9.npz"

    # Build a real (minimal) compatible TagObservationCache so is_compatible() passes
    from multi_tracker.data.tag_observation_cache import TagObservationCache
    writer = TagObservationCache(str(cache_path), mode="w", start_frame=0, end_frame=9)
    for fid in range(10):
        writer.add_frame(fid, [], [], [], [])
    writer.save(metadata={"family": "tag36h11", "start_frame": 0, "end_frame": 9,
                          "video_path": "", "detection_cache_hash": ""})
    writer.close()

    with patch(
        "multi_tracker.core.tracking.precompute.AprilTagDetector"
    ) as MockDet:
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=9,
            video_path="",
        )
        assert phase.has_cache_hit() is True
        MockDet.assert_not_called()  # detector not created on cache hit


def test_apriltag_phase_cache_miss_creates_detector(tmp_path):
    cache_path = tmp_path / "tags_0_9.npz"  # does not exist
    with patch(
        "multi_tracker.core.tracking.precompute.AprilTagDetector"
    ) as MockDet:
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=9,
            video_path="",
        )
        assert phase.has_cache_hit() is False
        MockDet.assert_called_once()


def test_apriltag_phase_process_frame_empty_crops_adds_empty_frame(tmp_path):
    cache_path = tmp_path / "tags_0_9.npz"
    with patch("multi_tracker.core.tracking.precompute.AprilTagDetector") as MockDet:
        mock_detector = MockDet.return_value
        mock_detector.detect_in_crops.return_value = []

        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=0,
            video_path="",
        )
        phase.process_frame(0, [], [], [], [])
        mock_detector.detect_in_crops.assert_not_called()
        # finalize should still succeed (empty frame was recorded)
        path = phase.finalize()
        assert path == str(cache_path)


def test_apriltag_phase_process_frame_calls_detect_in_crops(tmp_path):
    cache_path = tmp_path / "tags_0_0.npz"
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    with patch("multi_tracker.core.tracking.precompute.AprilTagDetector") as MockDet:
        mock_detector = MockDet.return_value
        mock_detector.detect_in_crops.return_value = []

        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=0,
            video_path="",
        )
        phase.process_frame(0, [crop], [0], [], [(5, 10)])
        mock_detector.detect_in_crops.assert_called_once_with(
            [crop], [(5, 10)], det_indices=[0]
        )
        phase.finalize()


def test_apriltag_phase_finalize_returns_existing_path_on_hit(tmp_path):
    from multi_tracker.data.tag_observation_cache import TagObservationCache
    cache_path = tmp_path / "tags_0_2.npz"
    writer = TagObservationCache(str(cache_path), mode="w", start_frame=0, end_frame=2)
    for fid in range(3):
        writer.add_frame(fid, [], [], [], [])
    writer.save(metadata={"family": "tag36h11", "start_frame": 0, "end_frame": 2,
                          "video_path": "", "detection_cache_hash": ""})
    writer.close()

    with patch("multi_tracker.core.tracking.precompute.AprilTagDetector"):
        phase = AprilTagPrecomputePhase(
            detector_config=Mock(),
            cache_path=cache_path,
            start_frame=0,
            end_frame=2,
            video_path="",
        )
        assert phase.has_cache_hit() is True
        result = phase.finalize()
        assert result == str(cache_path)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_unified_precompute.py::test_apriltag_phase_cache_hit_returns_true_when_cache_valid -v 2>&1 | head -20
```

Expected: `ImportError` for `AprilTagPrecomputePhase`.

- [ ] **Step 3: Implement `AprilTagPrecomputePhase` in `precompute.py`**

Append to `src/multi_tracker/core/tracking/precompute.py`, after `UnifiedPrecompute`:

```python
# ---------------------------------------------------------------------------
# AprilTagPrecomputePhase
# ---------------------------------------------------------------------------


class AprilTagPrecomputePhase:
    """Precompute phase that runs AprilTag detection on OBB crops."""

    name = "apriltag"
    is_fatal = False

    def __init__(
        self,
        detector_config,           # AprilTagConfig instance
        cache_path,                # str or Path
        start_frame: int,
        end_frame: int,
        video_path: str = "",
    ) -> None:
        from pathlib import Path as _Path

        from multi_tracker.core.identity.apriltag_detector import AprilTagDetector
        from multi_tracker.data.tag_observation_cache import TagObservationCache

        self._cache_path = _Path(cache_path)
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._video_path = video_path
        self._cfg = detector_config
        self._detector = None
        self._tag_cache = None
        self._hit = False
        self._closed = False

        # Cache-hit check
        if self._cache_path.exists():
            probe = TagObservationCache(str(self._cache_path), mode="r")
            if probe.is_compatible() and probe.covers_frame_range(start_frame, end_frame):
                self._hit = True
            probe.close()

        if not self._hit:
            self._detector = AprilTagDetector(detector_config)
            self._tag_cache = TagObservationCache(
                str(self._cache_path), mode="w",
                start_frame=start_frame, end_frame=end_frame,
            )

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
        if self._hit or self._tag_cache is None or self._detector is None:
            return
        if not crops:
            self._tag_cache.add_frame(frame_idx, [], [], [], [], hammings=[])
            return
        observations = self._detector.detect_in_crops(
            crops, crop_offsets, det_indices=det_ids
        )
        if observations:
            self._tag_cache.add_frame(
                frame_idx,
                tag_ids=[o.tag_id for o in observations],
                centers_xy=[o.center_xy for o in observations],
                corners=[o.corners for o in observations],
                det_indices=[o.det_index for o in observations],
                hammings=[o.hamming for o in observations],
            )
        else:
            self._tag_cache.add_frame(frame_idx, [], [], [], [])

    def finalize(self) -> Optional[str]:
        if self._hit:
            return str(self._cache_path)
        if self._tag_cache is None:
            return None
        from multi_tracker.data.tag_observation_cache import detection_cache_hash

        det_hash = ""
        self._tag_cache.save(
            metadata={
                "family": getattr(self._cfg, "family", ""),
                "detection_cache_hash": det_hash,
                "video_path": self._video_path,
                "start_frame": self._start_frame,
                "end_frame": self._end_frame,
            }
        )
        self._tag_cache.close()
        return str(self._cache_path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_unified_precompute.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/precompute.py tests/test_unified_precompute.py
git commit -m "feat: add AprilTagPrecomputePhase to unified precompute"
```

---

## Task 4: `CNNPrecomputePhase`

**Files:**
- Modify: `src/multi_tracker/core/tracking/precompute.py` (append class)
- Modify: `tests/test_unified_precompute.py` (append tests)

- [ ] **Step 1: Write failing tests for `CNNPrecomputePhase`**

Append to `tests/test_unified_precompute.py`:

```python
# ---------------------------------------------------------------------------
# CNNPrecomputePhase
# ---------------------------------------------------------------------------


def test_cnn_phase_has_cache_hit_false_when_no_file(tmp_path):
    from multi_tracker.core.tracking.precompute import CNNPrecomputePhase
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig

    cache_path = tmp_path / "cnn_0_9.npz"
    with patch("multi_tracker.core.tracking.precompute.CNNIdentityBackend"):
        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake/model.pth"),
            model_path="/fake/model.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )
        assert phase.has_cache_hit() is False


def test_cnn_phase_process_frame_batches_crops(tmp_path):
    from multi_tracker.core.tracking.precompute import CNNPrecomputePhase
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig, ClassPrediction

    cache_path = tmp_path / "cnn.npz"
    with patch("multi_tracker.core.tracking.precompute.CNNIdentityBackend") as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.predict_batch.return_value = [
            ClassPrediction(class_name="tag_0", confidence=0.9, det_index=0)
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=2),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        # Two frames each with one crop — batch_size=2, so flush happens after frame 1
        phase.process_frame(0, [crop], [0], [], [(0, 0)])
        phase.process_frame(1, [crop], [0], [], [(0, 0)])
        # One batch of 2 should have been flushed
        assert mock_backend.predict_batch.call_count == 1

        phase.finalize()
        assert cache_path.exists()


def test_cnn_phase_finalize_flushes_partial_batch(tmp_path):
    from multi_tracker.core.tracking.precompute import CNNPrecomputePhase
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig, ClassPrediction

    cache_path = tmp_path / "cnn_partial.npz"
    with patch("multi_tracker.core.tracking.precompute.CNNIdentityBackend") as MockBackend:
        mock_backend = MockBackend.return_value
        mock_backend.predict_batch.return_value = [
            ClassPrediction(class_name="tag_1", confidence=0.8, det_index=0)
        ]

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=10),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )

        crop = np.zeros((20, 20, 3), dtype=np.uint8)
        phase.process_frame(0, [crop], [0], [], [(0, 0)])  # 1 crop, batch=10 → not flushed yet
        assert mock_backend.predict_batch.call_count == 0

        phase.finalize()
        assert mock_backend.predict_batch.call_count == 1  # flushed in finalize
        assert cache_path.exists()


def test_cnn_phase_process_frame_empty_crops_does_not_add_to_batch(tmp_path):
    from multi_tracker.core.tracking.precompute import CNNPrecomputePhase
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig

    cache_path = tmp_path / "cnn_empty.npz"
    with patch("multi_tracker.core.tracking.precompute.CNNIdentityBackend") as MockBackend:
        mock_backend = MockBackend.return_value

        phase = CNNPrecomputePhase(
            config=CNNIdentityConfig(model_path="/fake.pth", batch_size=2),
            model_path="/fake.pth",
            cache_path=cache_path,
            name="cnn_identity",
        )
        phase.process_frame(0, [], [], [], [])
        mock_backend.predict_batch.assert_not_called()
        phase.finalize()
        # cache still flushed (empty frame recorded)
        assert cache_path.exists()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_unified_precompute.py::test_cnn_phase_has_cache_hit_false_when_no_file -v 2>&1 | head -15
```

Expected: `ImportError` for `CNNPrecomputePhase`.

- [ ] **Step 3: Implement `CNNPrecomputePhase` in `precompute.py`**

Add this import at the top of `precompute.py` (alongside existing imports):

```python
from multi_tracker.core.identity.cnn_identity import (
    CNNIdentityBackend,
    CNNIdentityCache,
    CNNIdentityConfig,
    ClassPrediction,
)
```

Then append `CNNPrecomputePhase` class to `precompute.py`:

```python
# ---------------------------------------------------------------------------
# CNNPrecomputePhase
# ---------------------------------------------------------------------------


class CNNPrecomputePhase:
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
        from pathlib import Path as _Path

        self.name = name
        self._cache_path = _Path(cache_path)
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
        det_ids: List[int],
        all_obb: List[np.ndarray],
        crop_offsets: List[Tuple[int, int]],
    ) -> None:
        if self._hit or self._cache is None:
            return
        if not crops:
            self._cache.save(frame_idx, [])
            return
        for crop, det_id in zip(crops, det_ids):
            self._pending_crops.append(crop)
            self._pending_frame_idx.append(frame_idx)
            self._pending_det_ids.append(det_id)

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
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_unified_precompute.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/precompute.py tests/test_unified_precompute.py
git commit -m "feat: add CNNPrecomputePhase to unified precompute"
```

---

## Task 5: `PosePipeline` implements `PrecomputePhase`

**Files:**
- Modify: `src/multi_tracker/core/tracking/pose_pipeline.py`

No new test file — the `PosePipeline` protocol methods are integration-tested via `worker.py` in the existing tracking test suite. Unit tests for the new methods are added inline.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_unified_precompute.py`:

```python
# ---------------------------------------------------------------------------
# PosePipeline as PrecomputePhase
# ---------------------------------------------------------------------------


def test_pose_pipeline_has_cache_hit_false_when_not_set():
    from multi_tracker.core.tracking.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=False,
        cache_path="/some/path.hdf5",
        finalize_metadata={},
    )
    assert pipeline.has_cache_hit() is False


def test_pose_pipeline_has_cache_hit_true_when_set():
    from multi_tracker.core.tracking.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/some/path.hdf5",
        finalize_metadata={},
    )
    assert pipeline.has_cache_hit() is True


def test_pose_pipeline_finalize_returns_cache_path_on_hit():
    from multi_tracker.core.tracking.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/cached/path.hdf5",
        finalize_metadata={},
    )
    result = pipeline.finalize()
    assert result == "/cached/path.hdf5"


def test_pose_pipeline_process_frame_noop_on_hit():
    from multi_tracker.core.tracking.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/hit.hdf5",
        finalize_metadata={},
    )
    # Should not raise even with no backend
    pipeline.process_frame(0, [], [], [], [])


def test_pose_pipeline_close_idempotent():
    from multi_tracker.core.tracking.pose_pipeline import PosePipeline

    pipeline = PosePipeline(
        pose_backend=None,
        cache_writer=None,
        cache_hit=True,
        cache_path="/hit.hdf5",
        finalize_metadata={},
    )
    pipeline.close()
    pipeline.close()  # must not raise
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_unified_precompute.py::test_pose_pipeline_has_cache_hit_false_when_not_set -v 2>&1 | head -15
```

Expected: `TypeError` (unexpected keyword arguments `cache_hit`, `cache_path`, `finalize_metadata`).

- [ ] **Step 3: Update `PosePipeline.__init__` to accept PrecomputePhase params**

In `src/multi_tracker/core/tracking/pose_pipeline.py`, update the constructor of `PosePipeline`:

```python
def __init__(
    self,
    pose_backend,
    cache_writer,
    *,
    cross_frame_batch: int = 64,
    crop_workers: int = 4,
    pre_resize_target: int = 0,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    suppress_foreign_obb: bool = True,
    padding_fraction: float = 0.1,
    # PrecomputePhase protocol support:
    cache_hit: bool = False,
    cache_path: Optional[str] = None,
    finalize_metadata: Optional[dict] = None,
):
    self._backend = pose_backend
    self._batch_size = max(1, cross_frame_batch)
    self._pre_resize = max(0, pre_resize_target)
    self._bg_color = bg_color
    self._suppress_foreign = suppress_foreign_obb
    self._padding = padding_fraction

    self._crop_pool = ThreadPoolExecutor(
        max_workers=max(1, crop_workers),
        thread_name_prefix="pose-crop",
    )
    self._infer_pool = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="pose-infer"
    )
    self._async_cache = AsyncCacheWriter(cache_writer) if cache_writer else None
    self._cache_writer_raw = cache_writer

    # Batch accumulators
    self._pending: List[FrameCropResult] = []
    self._flat_crops: List[np.ndarray] = []
    self._inflight: Optional[Future] = None

    # PrecomputePhase state
    self._cache_hit = cache_hit
    self._cache_path = cache_path
    self._finalize_metadata = finalize_metadata or {}
    self._closed = False
    self._async_cache_closed = False
```

- [ ] **Step 4: Add `name`, `is_fatal`, `has_cache_hit()`, `process_frame()`, `finalize()` and update `close()`**

After the existing `run()` method, before `close()`, add:

```python
# ------------------------------------------------------------------ #
# PrecomputePhase protocol                                            #
# ------------------------------------------------------------------ #

name = "pose"
is_fatal = True

def has_cache_hit(self) -> bool:
    """Return True if the cache was pre-built and the loop can be skipped."""
    return self._cache_hit

def process_frame(
    self,
    frame_idx: int,
    crops: List[np.ndarray],
    det_ids: List[int],
    all_obb: List[np.ndarray],
    crop_offsets: List[Tuple[int, int]],
) -> None:
    """Accept pre-extracted crops and feed them into the inference pipeline.

    Letterboxing is applied here if pre_resize_target > 0 (backend detail).
    """
    if self._cache_hit:
        return
    if not crops:
        # No detections this frame — nothing to infer. The pose cache is sparse;
        # frames with no detections are simply absent from the cache.
        return

    fcr = FrameCropResult(
        frame_idx=frame_idx,
        det_ids=list(det_ids),
        n_dets=len(all_obb),
        crops=[],
        crop_to_det=[],
        crop_offsets={},
        all_obb_corners=list(all_obb),
        crop_transforms={},
    )

    for crop, offset, det_id in zip(crops, crop_offsets, det_ids):
        processed = crop
        if self._pre_resize > 0:
            processed, transform = letterbox_crop(
                processed, self._pre_resize, self._bg_color
            )
            fcr.crop_transforms[det_id] = transform
        fcr.crops.append(processed)
        fcr.crop_to_det.append(det_id)
        fcr.crop_offsets[det_id] = offset

    self._pending.append(fcr)
    self._flat_crops.extend(fcr.crops)

    if len(self._flat_crops) >= self._batch_size:
        self._flush()

def finalize(self) -> Optional[str]:
    """Flush in-flight inference, write cache, return path."""
    if self._cache_hit:
        return self._cache_path
    self._wait_inflight()
    self._close_async_cache()
    if self._cache_writer_raw is not None and self._finalize_metadata:
        self._cache_writer_raw.save(metadata=self._finalize_metadata)
        self._cache_writer_raw.close()
    logger.info("Pose properties cache saved: %s", self._cache_path)
    return self._cache_path

def _close_async_cache(self) -> None:
    if self._async_cache and not self._async_cache_closed:
        self._async_cache_closed = True
        try:
            self._async_cache.flush_and_close()
        except Exception:
            pass
```

Replace the existing `close()` method with:

```python
def close(self) -> None:
    """Shut down thread pools and release backend. Idempotent."""
    if self._closed:
        return
    self._closed = True
    self._wait_inflight()
    self._close_async_cache()
    self._crop_pool.shutdown(wait=False)
    self._infer_pool.shutdown(wait=False)
    if self._backend is not None:
        try:
            self._backend.close()
        except Exception:
            pass
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_unified_precompute.py -v
```

Expected: all pass.

- [ ] **Step 6: Run the full test suite to check nothing regressed**

```bash
python -m pytest tests/ -v --ignore=tests/benchmarks -x -q 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/multi_tracker/core/tracking/pose_pipeline.py tests/test_unified_precompute.py
git commit -m "feat: PosePipeline implements PrecomputePhase protocol (process_frame, finalize, close)"
```

---

## Task 6: Wire `worker.py` — Replace Precompute Blocks

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py`

This is the biggest refactor. Read `worker.py` carefully before editing.

- [ ] **Step 1: Add import of `precompute` module at top of `worker.py`**

Find the imports section. Add (near other core identity imports):

```python
from multi_tracker.core.tracking.precompute import (
    AprilTagPrecomputePhase,
    CNNPrecomputePhase,
    CropConfig,
    UnifiedPrecompute,
)
```

- [ ] **Step 2: Delete the six old precompute runner/flag helpers**

Delete these six methods entirely from `worker.py` (do NOT delete `_build_tag_cache_path` or `_build_cnn_identity_cache_path` — those are still used by `_build_precompute_phases()`):

1. `_should_precompute_individual_data()` (lines ~379–387)
2. `_precompute_pose_data()` (lines ~415–671)
3. `_should_precompute_apriltag_data()` (lines ~677–685)
4. `_should_precompute_cnn_identity_data()` (lines ~691–699)
5. `_run_cnn_identity_precompute()` (lines ~710–830)
6. `_run_apriltag_precompute()` (lines ~841–1051)

After deleting, run tests to confirm nothing else imports these methods:

```bash
python -m pytest tests/ -x -q 2>&1 | tail -20
```

- [ ] **Step 3: Add `_build_precompute_phases()`**

Add this new method in place of the deleted helpers (keep the section comment `# Precompute`):

```python
def _build_precompute_phases(
    self,
    params: dict,
    detection_method: str,
    detection_cache,
    start_frame: int,
    end_frame: int,
) -> list:
    """Build the list of enabled precompute phases for a tracking run.

    Returns [] when precompute should be skipped entirely (backward mode,
    preview mode, wrong detection method, or no detection cache).
    """
    if detection_method != "yolo_obb":
        return []
    if self.backward_mode or self.preview_mode:
        return []
    if detection_cache is None:
        return []

    phases = []

    # --- Pose ---
    pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
    if pose_enabled:
        from multi_tracker.core.identity.properties_cache import (
            IndividualPropertiesCache,
            compute_detection_hash,
            compute_extractor_hash,
            compute_filter_settings_hash,
            compute_individual_properties_id,
        )
        from multi_tracker.core.identity.runtime_api import (
            build_runtime_config,
            create_pose_backend_from_config,
        )

        detection_hash = compute_detection_hash(
            params.get("INFERENCE_MODEL_ID", ""),
            self.video_path,
            start_frame,
            end_frame,
            detection_cache_version="2.0",
        )
        filter_hash = compute_filter_settings_hash(params)
        extractor_hash = compute_extractor_hash(params)
        properties_id = compute_individual_properties_id(
            detection_hash, filter_hash, extractor_hash
        )
        pose_cache_path = self._build_individual_properties_cache_path(
            properties_id, start_frame, end_frame
        )
        self.individual_properties_cache_path = str(pose_cache_path)
        params["INDIVIDUAL_PROPERTIES_ID"] = properties_id
        params["INDIVIDUAL_PROPERTIES_CACHE_PATH"] = str(pose_cache_path)

        # Check cache hit
        pose_cache_hit = False
        if pose_cache_path.exists():
            existing = IndividualPropertiesCache(str(pose_cache_path), mode="r")
            try:
                pose_cache_hit = existing.is_compatible()
            finally:
                existing.close()

        pose_backend = None
        pose_cache_writer = None
        finalize_metadata = {}

        if not pose_cache_hit:
            pose_out_root = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
            if not pose_out_root:
                pose_out_root = str(pose_cache_path.parent)

            pose_config = build_runtime_config(params, out_root=pose_out_root)
            pose_backend = create_pose_backend_from_config(pose_config)
            pose_backend.warmup()

            runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "")).lower()
            if runtime_flavor.startswith("onnx") or runtime_flavor.startswith("tensorrt"):
                try:
                    resolved = str(
                        getattr(pose_backend, "exported_model_path", "")
                        or getattr(pose_backend, "model_path", "")
                    ).strip()
                except Exception:
                    resolved = ""
                if resolved:
                    params["POSE_EXPORTED_MODEL_PATH"] = resolved
                    self.pose_exported_model_resolved_signal.emit(resolved)

            pose_cache_writer = IndividualPropertiesCache(str(pose_cache_path), mode="w")
            keypoint_names = list(
                getattr(pose_backend, "output_keypoint_names", []) or []
            )
            finalize_metadata = {
                "individual_properties_id": properties_id,
                "detection_hash": detection_hash,
                "filter_settings_hash": filter_hash,
                "extractor_hash": extractor_hash,
                "pose_keypoint_names": keypoint_names,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "video_path": str(Path(self.video_path).expanduser().resolve()),
            }

        _POSE_CROSS_FRAME_BATCH = int(params.get("POSE_PRECOMPUTE_BATCH_SIZE", 64))
        _bg_raw = params.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])
        _pose_bg_color = (
            tuple(int(c) for c in _bg_raw)
            if isinstance(_bg_raw, (list, tuple)) and len(_bg_raw) == 3
            else (0, 0, 0)
        )
        _suppress_foreign_obb = bool(params.get("SUPPRESS_FOREIGN_OBB_REGIONS", True))
        _crop_workers = int(params.get("POSE_PIPELINE_CROP_WORKERS", 4))
        _pre_resize = int(params.get("POSE_PIPELINE_PRE_RESIZE", 0))
        if _pre_resize <= 0 and pose_backend is not None:
            _pre_resize = int(getattr(pose_backend, "preferred_input_size", 0) or 0)

        from multi_tracker.core.tracking.pose_pipeline import PosePipeline

        pipeline = PosePipeline(
            pose_backend,
            pose_cache_writer,
            cross_frame_batch=_POSE_CROSS_FRAME_BATCH,
            crop_workers=_crop_workers,
            pre_resize_target=_pre_resize,
            bg_color=_pose_bg_color,
            suppress_foreign_obb=_suppress_foreign_obb,
            padding_fraction=float(params.get("INDIVIDUAL_CROP_PADDING", 0.1)),
            cache_hit=pose_cache_hit,
            cache_path=str(pose_cache_path),
            finalize_metadata=finalize_metadata,
        )
        phases.append(pipeline)

    # --- AprilTag ---
    identity_method = str(params.get("IDENTITY_METHOD", "none_disabled")).lower()
    if identity_method == "apriltags":
        from multi_tracker.core.identity.apriltag_detector import AprilTagConfig

        cfg = AprilTagConfig.from_params(params)
        tag_cache_path = self._build_tag_cache_path(start_frame, end_frame)
        if tag_cache_path is not None:
            try:
                phase = AprilTagPrecomputePhase(
                    detector_config=cfg,
                    cache_path=tag_cache_path,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    video_path=str(Path(self.video_path).expanduser().resolve()),
                )
                phases.append(phase)
            except ImportError as exc:
                logger.warning("AprilTag precompute skipped: %s", exc)
                self.warning_signal.emit("AprilTag Unavailable", str(exc))

    # --- CNN Identity ---
    if identity_method == "cnn_classifier":
        model_path = str(params.get("CNN_CLASSIFIER_MODEL_PATH", ""))
        if model_path and os.path.exists(model_path):
            from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig

            cnn_cfg = CNNIdentityConfig(
                model_path=model_path,
                confidence=float(params.get("CNN_CLASSIFIER_CONFIDENCE", 0.5)),
                batch_size=int(params.get("CNN_CLASSIFIER_BATCH_SIZE", 64)),
            )
            cnn_cache_path = self._build_cnn_identity_cache_path(start_frame, end_frame)
            if cnn_cache_path:
                phase = CNNPrecomputePhase(
                    config=cnn_cfg,
                    model_path=model_path,
                    cache_path=cnn_cache_path,
                    compute_runtime=str(params.get("COMPUTE_RUNTIME", "cpu")),
                    name="cnn_identity",
                )
                phases.append(phase)
        else:
            logger.warning(
                "CNN identity precompute skipped: model_path not found: %s", model_path
            )

    return phases
```

**Note:** `_build_tag_cache_path()` and `_build_cnn_identity_cache_path()` are intentionally kept — they are still used by `_build_precompute_phases()`. Only the precompute runner and flag methods are deleted.

- [ ] **Step 4: Replace the three precompute blocks in `run()` with the unified block**

Find the three sequential blocks in `run()` (around lines 1797–1878):

```python
if individual_data_precompute_enabled:
    ...
    props_path, props_cache_hit = self._precompute_pose_data(...)
    ...

apriltag_precompute_enabled = ...
if (apriltag_precompute_enabled and ...):
    tag_observation_cache_path = self._run_apriltag_precompute(...)

cnn_identity_precompute_enabled = ...
if (cnn_identity_precompute_enabled and ...):
    cnn_identity_cache_path = self._run_cnn_identity_precompute(...)
```

Also delete the `individual_data_precompute_enabled = self._should_precompute_individual_data(...)` line that precedes them.

Replace ALL of the above with:

```python
# === UNIFIED PRECOMPUTE ===
props_path = None
tag_observation_cache_path = None
cnn_identity_cache_path = None

phases = self._build_precompute_phases(
    p, detection_method, detection_cache, start_frame, end_frame
)
if phases:
    _bg_raw = p.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])
    crop_config = CropConfig(
        padding_fraction=float(p.get("INDIVIDUAL_CROP_PADDING", 0.1)),
        suppress_foreign=bool(p.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)),
        bg_color=(
            tuple(int(c) for c in _bg_raw)
            if isinstance(_bg_raw, (list, tuple)) and len(_bg_raw) == 3
            else (0, 0, 0)
        ),
    )
    precompute = UnifiedPrecompute(phases, crop_config)
    try:
        results = precompute.run(
            cap,
            detection_cache,
            detector,
            start_frame,
            end_frame,
            float(p.get("RESIZE_FACTOR", 1.0)),
            p.get("ROI_MASK", None),
            progress_cb=lambda pct, msg: self.progress_signal.emit(pct, msg),
            stop_check=lambda: self._stop_requested,
            warning_cb=lambda title, msg: self.warning_signal.emit(title, msg),
        )
    except Exception as exc:
        logger.exception("Unified precompute failed (fatal phase).")
        self.warning_signal.emit(
            "Precompute Failed",
            f"Tracking aborted because precompute failed:\n{exc}",
        )
        if detection_cache:
            detection_cache.close()
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.finished_signal.emit(False, [], [])
        return

    props_path = results.get("pose")
    tag_observation_cache_path = results.get("apriltag")
    cnn_identity_cache_path = results.get("cnn_identity")

    if props_path:
        logger.info("Individual properties cache: %s", props_path)
```

- [ ] **Step 5: Remove the `individual_data_precompute_enabled` variable wherever it's used to gate downstream logic**

Search for `individual_data_precompute_enabled` in `worker.py` and remove any remaining references. The `props_path` variable now serves the same purpose (it is `None` when pose precompute didn't run).

```bash
grep -n "individual_data_precompute_enabled" src/multi_tracker/core/tracking/worker.py
```

Replace each remaining use: if code says `if individual_data_precompute_enabled and props_path:` change to `if props_path:`.

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/multi_tracker/core/tracking/worker.py
git commit -m "refactor: replace 3 precompute video reads with UnifiedPrecompute single-pass"
```

---

## Task 7: Config and GUI Cleanup

**Files:**
- Modify: `configs/default.json`
- Modify: `src/multi_tracker/gui/main_window.py`

- [ ] **Step 1: Remove `cnn_classifier_crop_padding` from `configs/default.json`**

In `configs/default.json`, delete this line (around line 129):

```json
  "cnn_classifier_crop_padding": 0.1,
```

- [ ] **Step 2: Remove `spin_cnn_crop_padding` from `main_window.py`**

In `src/multi_tracker/gui/main_window.py`, find and delete:

1. The widget creation block (around line 6157):
```python
self.spin_cnn_crop_padding = QDoubleSpinBox()
self.spin_cnn_crop_padding.setRange(0.0, 1.0)
self.spin_cnn_crop_padding.setSingleStep(0.05)
self.spin_cnn_crop_padding.setValue(0.1)
color_layout.addRow("CNN crop padding fraction", self.spin_cnn_crop_padding)
```

2. The params dict entry (around line 14544):
```python
"CNN_CLASSIFIER_CROP_PADDING": self.spin_cnn_crop_padding.value(),
```

3. The config load block (around line 15419):
```python
self.spin_cnn_crop_padding.setValue(
    float(get_cfg("cnn_classifier_crop_padding", default=0.1))
)
```

4. The config save block (around line 15921):
```python
"cnn_classifier_crop_padding": self.spin_cnn_crop_padding.value(),
```

- [ ] **Step 3: Add migration warning for non-default `cnn_classifier_crop_padding`**

In the config load section, where the CNN classifier config is loaded from file, add a one-time logger warning. Find where `get_cfg("cnn_classifier_confidence", ...)` is loaded (around the same block) and add directly before that block:

```python
# Warn users who had a non-default cnn_classifier_crop_padding in their config
_legacy_crop_padding = get_cfg("cnn_classifier_crop_padding", default=None)
if _legacy_crop_padding is not None and float(_legacy_crop_padding) != 0.1:
    logger.warning(
        "Config key 'cnn_classifier_crop_padding' (value=%.2f) is no longer used. "
        "All precompute phases now use 'individual_crop_padding'. "
        "Update your crop padding setting in the Individual Analysis panel.",
        float(_legacy_crop_padding),
    )
```

- [ ] **Step 4: Update the `INDIVIDUAL_CROP_PADDING` spinbox label and tooltip**

Find the `spin_individual_padding` creation block (~line 6209):

```python
fl_common.addRow("Crop padding fraction", self.spin_individual_padding)
```

Change to:

```python
fl_common.addRow("Crop padding fraction (all phases)", self.spin_individual_padding)
```

And update the tooltip from:

```python
self.spin_individual_padding.setToolTip(
    "Padding around OBB bounding box as fraction of size.\n"
    "0.1 = 10% padding on each side."
)
```

To:

```python
self.spin_individual_padding.setToolTip(
    "Padding around OBB bounding box as fraction of size.\n"
    "0.1 = 10% padding on each side.\n"
    "Applies to all precompute phases: pose, AprilTag, and CNN identity."
)
```

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add configs/default.json src/multi_tracker/gui/main_window.py
git commit -m "feat: remove cnn_classifier_crop_padding — all phases use INDIVIDUAL_CROP_PADDING"
```

---

## Final Verification

- [ ] Run the full test suite one last time:

```bash
python -m pytest tests/ -q 2>&1 | tail -10
```

- [ ] Confirm `cnn_classifier_crop_padding` is fully gone:

```bash
grep -r "cnn_classifier_crop_padding\|spin_cnn_crop_padding\|CNN_CLASSIFIER_CROP_PADDING" src/ configs/ tests/
```

Expected: no output.

- [ ] Confirm `crop_padding` is gone from `CNNIdentityConfig`:

```bash
grep -n "crop_padding" src/multi_tracker/core/identity/cnn_identity.py
```

Expected: no output.
