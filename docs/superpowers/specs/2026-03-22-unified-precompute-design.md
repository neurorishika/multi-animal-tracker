# MAT Unified Precompute Pipeline — Design Spec

**Date:** 2026-03-22
**Status:** Approved

---

## Overview

Replace the three sequential precompute phases (pose, AprilTag, CNN identity) with a single video pass that reads each frame once, extracts crops once, and fans out to all enabled phases simultaneously. Introduces a `PrecomputePhase` protocol so new phases (additional CNN classifiers, future identity methods) can be added without touching the core loop.

---

## Goals

- Single video read per tracking run, regardless of how many precompute phases are enabled.
- Identical crop extraction for all phases: same padding, same foreign-OBB suppression, same filtered detection set.
- Protocol-based phase registry: adding a new phase requires implementing three methods, not modifying `worker.py`.
- Extensible to future simultaneous phases (e.g. AprilTag + CNN identity + CNN age classifier running together).
- Designed so real-time per-frame inference (no pre-pass) is a future drop-in: phases implement the same `process_frame()` interface whether called in a pre-pass or inline during tracking.

---

## Non-Goals

- No changes to AprilTag detection logic, CNN inference logic, or pose inference logic.
- No ONNX/TensorRT integration for pose (separate spec).
- No real-time tracking mode (future work; protocol is designed to support it without structural changes).
- No changes to cache file formats — existing pose, AprilTag, and CNN caches remain valid.

---

## Architecture

### `PrecomputePhase` Protocol

Defined in `core/tracking/precompute.py`. Any object implementing this protocol can participate in the unified precompute loop.

```python
class PrecomputePhase(Protocol):
    name: str  # e.g. "pose", "apriltag", "cnn_identity"

    def process_frame(
        self,
        frame_idx: int,
        crops: list[np.ndarray],      # pre-extracted, one per detection
        det_ids: list[int],           # detection index per crop
        all_obb: list[np.ndarray],    # all filtered OBB corners this frame
    ) -> None: ...

    def finalize(self) -> str | None: ...   # flush caches, return artifact path or None
    def close(self) -> None: ...            # release backend/model resources
```

All phases receive **identical** crops — the only difference between phases is what they do with those crops internally (AprilTag detection, CNN inference, pose inference with optional letterboxing).

### `CropConfig` Dataclass

Controls the shared crop extraction step in `UnifiedPrecompute`. All phases use this config.

```python
@dataclass
class CropConfig:
    padding_fraction: float = 0.1           # maps to INDIVIDUAL_CROP_PADDING
    suppress_foreign: bool = True           # maps to SUPPRESS_FOREIGN_OBB_REGIONS
    bg_color: tuple[int, int, int] = (0, 0, 0)  # maps to INDIVIDUAL_BACKGROUND_COLOR
```

`CNN_CLASSIFIER_CROP_PADDING` is removed. `INDIVIDUAL_CROP_PADDING` now governs all phases.

### `UnifiedPrecompute`

```python
class UnifiedPrecompute:
    def __init__(self, phases: list[PrecomputePhase], crop_config: CropConfig): ...

    def run(
        self,
        cap: cv2.VideoCapture,
        detection_cache,
        detector,
        start_frame: int,
        end_frame: int,
        resize_factor: float,
        roi_mask,
        progress_cb: Callable[[int, str], None] | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> dict[str, str | None]:
        ...
```

**Per-frame loop:**

1. `cap.read()` + optional resize — one read, shared across all phases.
2. `detection_cache.get_frame(frame_idx)` → raw detections.
3. `detector.filter_raw_detections(...)` with `roi_mask` → filtered OBBs and det IDs.
4. `extract_one_crop()` per detection — shared crop extraction, once per frame.
5. `phase.process_frame(frame_idx, crops, det_ids, all_obb)` for every phase in the list.

**After loop:**

6. `phase.finalize()` for each phase → collect `dict[phase.name → path_or_none]`.
7. `phase.close()` for each phase.
8. Return the results dict.

**Cache-hit short-circuit:** Before the loop begins, `UnifiedPrecompute.run()` calls a `has_cache_hit()` check on each phase. If all phases report a hit, the video read is skipped entirely and existing cache paths are returned immediately.

**Progress reporting:** `UnifiedPrecompute` emits a single unified progress percentage based on frames completed. Each phase may optionally provide a status string via an internal `status()` method; the unified loop forwards the active phase statuses to `progress_cb` alongside the frame count.

**Cancellation:** After each frame, `stop_check()` is called. If it returns `True`, the loop exits, `close()` is called on all phases (no `finalize()`), and the results dict has `None` for all phase keys.

---

## Phase Implementations

### `AprilTagPrecomputePhase`

Thin wrapper around the existing `AprilTagDetector` and `TagObservationCache`.

- Constructor: checks for existing compatible cache → sets `_cache_hit`. If hit, `process_frame()` is a no-op and `finalize()` returns the existing path.
- `process_frame()`: calls `at_detector.detect_in_crops(crops, offsets=None, det_indices=det_ids)` — crops are already extracted, offsets are not needed for classification purposes. Accumulates `TagObservation` results into the cache per frame.
- `finalize()`: saves `TagObservationCache` with metadata, closes detector, returns path.
- `close()`: calls `at_detector.close()` if not already closed.

### `CNNPrecomputePhase`

Wraps `CNNIdentityBackend` and `CNNIdentityCache`.

- Constructor: builds `CNNIdentityConfig` (without `crop_padding` — removed), checks for existing cache → sets `_cache_hit`.
- `process_frame()`: accumulates crops into an internal buffer. When buffer reaches `batch_size`, calls `backend.predict_batch()` and writes results to cache (in-memory via `cache.save()`).
- `finalize()`: flushes any remaining partial batch through `backend.predict_batch()`, calls `cache.flush()` (single `np.savez_compressed` write), returns path.
- `close()`: calls `backend.close()`.

**Note:** `crop_padding` is removed from `CNNIdentityConfig`. Crop extraction is fully delegated to `UnifiedPrecompute`.

### `PosePipeline` as a `PrecomputePhase`

`PosePipeline` (in `pose_pipeline.py`) gains three new methods to implement the protocol:

```python
def process_frame(
    self,
    frame_idx: int,
    crops: list[np.ndarray],      # pre-extracted base crops (no letterboxing yet)
    det_ids: list[int],
    all_obb: list[np.ndarray],
) -> None:
    # apply letterboxing internally if pre_resize_target > 0
    # accumulate into self._pending / self._flat_crops
    # flush inference batch if full (same logic as current run() loop)
```

```python
def finalize(self) -> str | None:
    # wait for in-flight inference (self._wait_inflight())
    # flush async cache writer (self._async_cache.flush_and_close())
    # save IndividualPropertiesCache with metadata
    # return cache path
```

```python
def close(self) -> None:
    # shut down thread pools
    # release pose backend
```

**Letterboxing is a backend detail:** It lives inside `process_frame()`, invisible to `UnifiedPrecompute`. All phases receive identical raw crops; pose applies letterboxing as its own backend-specific preprocessing step before inference.

**Standalone `PosePipeline.run()` is kept:** The existing `run()` method remains unchanged as a convenience wrapper for standalone pose-only usage. It calls `detector.filter_raw_detections()` internally (as it does now) and calls `process_frame()` per frame. No callers outside `worker.py` break.

---

## `worker.py` Orchestration

The three separate precompute blocks and six `_should_precompute_*` / `_run_*_precompute` helper methods are replaced with:

**`_build_precompute_phases()`** — new private helper:

```python
def _build_precompute_phases(
    self, params, detection_method, detection_cache, start_frame, end_frame
) -> list[PrecomputePhase]:
```

- Returns `[]` if `detection_method != "yolo_obb"` or `backward_mode` or `preview_mode`.
- Appends `PosePipeline` instance if `ENABLE_POSE_EXTRACTOR` is True.
- Appends `AprilTagPrecomputePhase` if `IDENTITY_METHOD == "apriltags"`.
- Appends `CNNPrecomputePhase` if `IDENTITY_METHOD == "cnn_classifier"`.
- Future phases (second CNN classifier, etc.) are added here without changing any other code.

**Unified precompute block in `run()`:**

```python
phases = self._build_precompute_phases(p, detection_method, detection_cache,
                                       start_frame, end_frame)
if phases:
    crop_config = CropConfig(
        padding_fraction=float(p.get("INDIVIDUAL_CROP_PADDING", 0.1)),
        suppress_foreign=bool(p.get("SUPPRESS_FOREIGN_OBB_REGIONS", True)),
        bg_color=tuple(p.get("INDIVIDUAL_BACKGROUND_COLOR", [0, 0, 0])),
    )
    precompute = UnifiedPrecompute(phases, crop_config)
    results = precompute.run(
        cap, detection_cache, detector, start_frame, end_frame,
        resize_factor, roi_mask,
        progress_cb=lambda pct, msg: self.progress_signal.emit(pct, msg),
        stop_check=lambda: self._stop_requested,
    )
    props_path             = results.get("pose")
    tag_observation_path   = results.get("apriltag")
    cnn_identity_path      = results.get("cnn_identity")
```

**Error handling** mirrors current behaviour:

- Pose failure → fatal: emit warning, release resources, abort tracking.
- AprilTag failure → non-fatal: emit warning, tracking continues without tag identity.
- CNN failure → non-fatal: emit warning, tracking continues without CNN identity.

Each phase catches its own errors in `finalize()` and returns `None` on failure (non-fatal phases) or re-raises (fatal phases).

---

## Crop Extraction Policy

The unified loop applies `extract_one_crop()` (from `pose_pipeline.py`) with:

- Padding: `INDIVIDUAL_CROP_PADDING` (same for all phases).
- Foreign suppression: `SUPPRESS_FOREIGN_OBB_REGIONS` (same for all phases).
- Background fill: `INDIVIDUAL_BACKGROUND_COLOR` (same for all phases).
- Detection set: filtered OBBs after `detector.filter_raw_detections()` with `roi_mask` applied.

**Behavioral changes vs. current:**

| Phase | Before | After |
|---|---|---|
| Pose | filtered OBBs, foreign suppressed, `INDIVIDUAL_CROP_PADDING` | unchanged |
| AprilTag | raw OBBs, optional foreign suppression, `INDIVIDUAL_CROP_PADDING` | now uses filtered OBBs (ROI mask applied) |
| CNN | raw OBBs, no foreign suppression, `CNN_CLASSIFIER_CROP_PADDING` | now uses filtered OBBs, foreign suppressed, `INDIVIDUAL_CROP_PADDING` |

AprilTag and CNN both receive strictly more correct crops under the new scheme (ROI mask respected, neighboring animals masked out).

---

## Config Changes

| Key | Change |
|---|---|
| `cnn_classifier_crop_padding` | **Removed** from `configs/default.json` |
| `INDIVIDUAL_CROP_PADDING` | Now governs all phases — UI label updated to "Crop Padding (all phases)" |

Existing saved configs containing `cnn_classifier_crop_padding` are silently ignored on load (unknown keys already dropped during config parsing).

---

## GUI Changes (`gui/main_window.py`)

- Remove `spin_cnn_crop_padding` widget, its label, and its wiring in config save/load.
- Update `INDIVIDUAL_CROP_PADDING` spinbox label to "Crop Padding (all phases)".

---

## `CNNIdentityConfig` Changes (`core/identity/cnn_identity.py`)

- Remove `crop_padding: float` field from the dataclass.
- Remove any reference to `crop_padding` in `CNNIdentityBackend` (it was only used during precompute, which is now handled by `UnifiedPrecompute`).

---

## Future Extension Points

**Adding a new phase** (e.g. CNN age classifier):

1. Implement `PrecomputePhase` (e.g. `CNNAgePrecomputePhase` in `precompute.py`).
2. Add one `if` block in `_build_precompute_phases()`.
3. No changes to `UnifiedPrecompute`, `PosePipeline`, or the tracking loop.

**Real-time mode** (future):

- The tracking loop calls `UnifiedPrecompute.process_one_frame(frame_idx, frame)` per frame instead of running a pre-pass.
- Each phase's `process_frame()` is called inline; phases with batch accumulation flush immediately (batch size 1) or return buffered results.
- No protocol changes needed.

---

## Testing (`tests/test_unified_precompute.py`)

Unit tests — all phases mocked, no real video or GPU required.

| Test | What it covers |
|---|---|
| `CropConfig` defaults and field types | Dataclass correctness |
| `UnifiedPrecompute` with empty phase list skips video read | No-op path |
| `process_frame` called once per frame per phase | Dispatch correctness |
| `finalize` and `close` called on all phases after loop | Lifecycle correctness |
| `stop_check` returning True exits loop early, calls `close` not `finalize` | Cancellation path |
| All phases cache-hit → video read skipped, existing paths returned | Cache-hit short-circuit |
| `AprilTagPrecomputePhase` returns existing path on cache hit | AT cache hit |
| `CNNPrecomputePhase.process_frame` batches and flushes at batch boundary | Batch accumulation |
| `CNNPrecomputePhase.finalize` flushes remaining partial batch | Partial batch at end |
| Phase `finalize` error returns None without preventing other phases from closing | Error isolation |

`tests/test_mat_cnn_identity.py` updated to remove `crop_padding` from `CNNIdentityConfig` test fixtures.

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/multi_tracker/core/tracking/precompute.py` | **New** — `PrecomputePhase` protocol, `CropConfig`, `UnifiedPrecompute`, `AprilTagPrecomputePhase`, `CNNPrecomputePhase` |
| `src/multi_tracker/core/tracking/pose_pipeline.py` | Add `process_frame()`, `finalize()`, `close()` — `PosePipeline` implements `PrecomputePhase` |
| `src/multi_tracker/core/tracking/worker.py` | Replace 3 precompute blocks + 6 helpers with `_build_precompute_phases()` + `UnifiedPrecompute` |
| `src/multi_tracker/core/identity/cnn_identity.py` | Remove `crop_padding` from `CNNIdentityConfig` |
| `configs/default.json` | Remove `cnn_classifier_crop_padding` |
| `src/multi_tracker/gui/main_window.py` | Remove `spin_cnn_crop_padding`; update `INDIVIDUAL_CROP_PADDING` label |
| `tests/test_unified_precompute.py` | **New** — unified precompute unit tests |
| `tests/test_mat_cnn_identity.py` | Remove `crop_padding` from `CNNIdentityConfig` fixtures |
