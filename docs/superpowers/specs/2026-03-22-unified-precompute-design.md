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
- Protocol-based phase registry: adding a new phase requires implementing the protocol, not modifying `worker.py`.
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
    name: str        # e.g. "pose", "apriltag", "cnn_identity"
    is_fatal: bool   # True → failure in finalize() aborts tracking; False → non-fatal warning

    def has_cache_hit(self) -> bool:
        """Return True if a valid existing cache covers this run.
        Called before the frame loop. If all phases return True, the video
        read is skipped entirely."""
        ...

    def process_frame(
        self,
        frame_idx: int,
        crops: list[np.ndarray],              # pre-extracted, one per detection
        det_ids: list[int],                   # detection index per crop
        all_obb: list[np.ndarray],            # all filtered OBB corners this frame (informational)
        crop_offsets: list[tuple[int, int]],  # (x0, y0) per crop for coord remapping
    ) -> None:
        """Process one frame. Called with empty lists when no detections exist.
        Must be a no-op when has_cache_hit() returned True."""
        ...

    def finalize(self) -> str | None:
        """Flush caches, return artifact path or None on failure.
        Runs to completion — stop_check is not called inside finalize().
        Non-fatal phases catch their own exceptions and return None.
        Fatal phases re-raise; UnifiedPrecompute.run() propagates the exception."""
        ...

    def close(self) -> None:
        """Release backend/model resources. Called after finalize(), or after
        cancellation (in which case finalize() is NOT called)."""
        ...
```

**`all_obb` in `process_frame()`:** Foreign suppression is already applied during crop extraction, so phases do not need `all_obb` for suppression. It is provided for phases that require spatial context beyond the pre-extracted crops (e.g. a future phase computing inter-animal distances). Current phases (AprilTag, CNN, Pose) treat it as informational only.

**`crop_offsets` in `process_frame()`:** Each element is the `(x0, y0)` origin of the corresponding crop in frame-space, produced by the AABB crop extraction step. AprilTag uses these to remap tag corner coordinates from crop-relative to frame-relative space before storing them in `TagObservationCache`. Pose uses them for keypoint coordinate remapping. CNN identity ignores them.

All phases receive **identical** crops — the only difference between phases is what they do with those crops internally.

### `CropConfig` Dataclass

Controls the shared crop extraction step. All phases use this config.

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
        warning_cb: Callable[[str, str], None] | None = None,
    ) -> dict[str, str | None]:
        ...
```

**Empty phase list:** If `phases` is empty, `run()` returns an empty dict immediately without reading from `cap`.

**Cache-hit short-circuit:**
- Before the frame loop, call `phase.has_cache_hit()` on each phase.
- If **all** phases return `True` → return existing paths immediately, no video read.
- If **some** phases hit and others miss → the video read proceeds. Cache-hit phases have `process_frame()` as a no-op and `finalize()` returns the existing path. Cache-miss phases run normally.

**Per-frame loop:**

1. `cap.read()` + optional resize — one read, shared across all phases.
2. `detection_cache.get_frame(frame_idx)` → raw detections.
3. `detector.filter_raw_detections(...)` with `roi_mask` → filtered OBBs and det IDs.
4. `extract_one_crop()` per detection → `(crop, (x0, y0))` pairs — once per frame, shared.
5. `phase.process_frame(frame_idx, crops, det_ids, all_obb, crop_offsets)` for every phase.
   - Step 5 is called even when `crops` is empty (zero detections in this frame). Each phase must handle empty lists gracefully.
6. After each frame, check `stop_check()`. If `True`: call `phase.close()` on all phases (no `finalize()`), return `{phase.name: None for phase in phases}`.

**Precondition:** `_build_precompute_phases()` returns `[]` if `detection_cache is None` (see below), so `UnifiedPrecompute.run()` is never called without a valid cache.

**After loop:**

Steps 7 and 8 are wrapped in a `try/finally` block so that `phase.close()` on all phases is guaranteed even when a fatal phase re-raises.

7. For each phase: call `phase.finalize()` inside a try/except.
   - If `phase.is_fatal` and `finalize()` raises → re-raise (caught by the outer `try/finally` which still runs step 8).
   - If `not phase.is_fatal` and `finalize()` raises → call `warning_cb(title, message)` if provided, log warning, record `None` for that phase. The phase does not emit Qt signals directly — `UnifiedPrecompute` is a plain Python object and surfaces warnings through `warning_cb`.
8. Call `phase.close()` on all phases (guaranteed via `finally`, regardless of `finalize()` outcome).
9. Return `{phase.name: path_or_none}`.

**`stop_check` during finalization:** `stop_check()` is not called inside `finalize()`. Each `finalize()` runs to completion. For `PosePipeline`, this means waiting for in-flight inference and async cache writes — bounded time, not indefinite.

**Progress reporting:** `UnifiedPrecompute` emits a single unified progress percentage based on frames completed. Progress messages include the frame count and any active phase status string.

---

## Phase Implementations

### `AprilTagPrecomputePhase`

Thin wrapper around the existing `AprilTagDetector` and `TagObservationCache`.

- `name = "apriltag"`, `is_fatal = False`.
- Constructor: checks for existing compatible cache with `probe.is_compatible() and probe.covers_frame_range(...)` → sets `_hit = True`. If hit, `process_frame()` is a no-op.
- `has_cache_hit()` → `self._hit`.
- `process_frame()`: if `crops` is empty, records an empty frame in the cache. Otherwise calls `at_detector.detect_in_crops(crops, crop_offsets, det_indices=det_ids)` using the passed `crop_offsets` for frame-relative coordinate remapping. Accumulates `TagObservation` results into `TagObservationCache` per frame.
- `finalize()`: saves `TagObservationCache` with metadata, returns path. Returns existing path if cache hit.
- `close()`: calls `at_detector.close()` if not already closed.

### `CNNPrecomputePhase`

Wraps `CNNIdentityBackend` and `CNNIdentityCache`.

- `name = "cnn_identity"` (or a user-supplied name for multi-classifier future use), `is_fatal = False`.
- Constructor: builds `CNNIdentityConfig` (without `crop_padding` — removed), checks for existing cache → sets `_hit = True`.
- `has_cache_hit()` → `self._hit`.
- `process_frame()`: if `crops` is empty, records an empty prediction list in memory. Otherwise accumulates crops into an internal buffer. When buffer reaches `batch_size`, calls `backend.predict_batch()` and records results in `CNNIdentityCache` in memory via `cache.save()` (in-memory accumulation — no disk write yet).
- `finalize()`: flushes any remaining partial batch through `backend.predict_batch()`, calls `cache.flush()` (single `np.savez_compressed` write to disk), returns path.
- `close()`: calls `backend.close()`.

**`CNNIdentityCache` two-phase semantics (already implemented):** `cache.save(frame_idx, predictions)` accumulates data in `self._data` dict in memory. `cache.flush()` writes the full `self._data` dict to disk via `np.savez_compressed` in a single call. `cache.exists()` checks disk only. This is the existing behavior — no changes to `CNNIdentityCache` are needed.

**`crop_padding` removed:** `CNNIdentityConfig.crop_padding` is removed. Crop extraction is fully delegated to `UnifiedPrecompute`. The `CNNPrecomputePhase` constructor takes `CNNIdentityConfig` (minus `crop_padding`) and a `cache_path`.

### `PosePipeline` as a `PrecomputePhase`

`PosePipeline` (in `pose_pipeline.py`) gains four new methods to implement the protocol:

- `name = "pose"`, `is_fatal = True`.

```python
def has_cache_hit(self) -> bool:
    """Return True if an existing compatible IndividualPropertiesCache exists."""
    ...

def process_frame(
    self,
    frame_idx: int,
    crops: list[np.ndarray],      # pre-extracted base crops (no letterboxing yet)
    det_ids: list[int],
    all_obb: list[np.ndarray],    # informational only
    crop_offsets: list[tuple[int, int]],
) -> None:
    # if crops is empty: record empty frame in async cache writer, return
    # apply letterboxing to each crop if pre_resize_target > 0
    # accumulate into self._pending / self._flat_crops
    # flush inference batch if full (same logic as current run() loop)
```

```python
def finalize(self) -> str | None:
    # wait for in-flight inference (self._wait_inflight())
    # flush async cache writer (self._async_cache.flush_and_close())
    # save IndividualPropertiesCache with metadata
    # return cache path
    # raises on failure (is_fatal = True)
```

```python
def close(self) -> None:
    # shut down crop_pool and infer_pool thread pools
    # release pose backend
```

**Letterboxing is a backend detail:** Lives inside `process_frame()`, invisible to `UnifiedPrecompute`. All phases receive identical raw crops; pose applies letterboxing as its own backend-specific preprocessing before inference.

**Standalone `PosePipeline.run()` is kept unchanged:** The existing `run(video_cap, detection_cache, detector, ...)` method remains as a convenience wrapper for standalone pose-only usage. It calls `detector.filter_raw_detections()` and crop extraction internally per frame, then calls `process_frame()`. The standalone path has independent test coverage so the two code paths remain consistent.

**Duplication note:** `PosePipeline.run()` calls `detector.filter_raw_detections()` internally. When called via `UnifiedPrecompute`, filtering is done externally before `process_frame()`. These two paths must remain consistent — if filter logic changes, both must be updated. This is flagged here as an intentional duplication to preserve backward compatibility of the standalone `run()` API.

---

## `worker.py` Orchestration

The three separate precompute blocks and six `_should_precompute_*` / `_run_*_precompute` helper methods are deleted and replaced with:

**`_build_precompute_phases()`** — new private helper:

```python
def _build_precompute_phases(
    self, params, detection_method, detection_cache, start_frame, end_frame
) -> list[PrecomputePhase]:
```

Returns `[]` (no precompute) if any of:
- `detection_method != "yolo_obb"`
- `self.backward_mode` is True
- `self.preview_mode` is True
- `detection_cache is None` (precondition: cache must exist for precompute)

Otherwise appends phases in this order:
1. If `ENABLE_POSE_EXTRACTOR` is True → add `PosePipeline` instance (constructed with backend, cache writer, etc.)
2. If `IDENTITY_METHOD == "apriltags"` → add `AprilTagPrecomputePhase`
3. If `IDENTITY_METHOD == "cnn_classifier"` → add `CNNPrecomputePhase`

Future phases (second CNN classifier, etc.) are added here without changing any other code.

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
    try:
        results = precompute.run(
            cap, detection_cache, detector, start_frame, end_frame,
            resize_factor, roi_mask,
            progress_cb=lambda pct, msg: self.progress_signal.emit(pct, msg),
            stop_check=lambda: self._stop_requested,
            warning_cb=lambda title, msg: self.warning_signal.emit(title, msg),
        )
    except Exception as exc:
        # A fatal phase (pose) raised — abort tracking
        logger.exception("Unified precompute failed (fatal phase).")
        self.warning_signal.emit("Precompute Failed", str(exc))
        cap.release(); detection_cache.close(); self.finished_signal.emit(False, [], [])
        return
    props_path             = results.get("pose")
    tag_observation_path   = results.get("apriltag")
    cnn_identity_path      = results.get("cnn_identity")
```

Non-fatal phase failures (AprilTag, CNN) are recorded as `None` in `results` by `UnifiedPrecompute.run()`. Warnings are surfaced via `warning_cb` — `UnifiedPrecompute` calls it after catching a non-fatal exception from `finalize()`. Phases themselves do not emit Qt signals.

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
| AprilTag | raw OBBs, optional foreign suppression, `INDIVIDUAL_CROP_PADDING` | filtered OBBs (ROI mask applied), always foreign suppressed |
| CNN | raw OBBs, no foreign suppression, `CNN_CLASSIFIER_CROP_PADDING` | filtered OBBs, foreign suppressed, `INDIVIDUAL_CROP_PADDING` |

AprilTag and CNN both receive strictly more correct crops (ROI mask respected, neighboring animals masked out).

---

## Config Changes

| Key | Change |
|---|---|
| `cnn_classifier_crop_padding` | **Removed** from `configs/default.json` |
| `INDIVIDUAL_CROP_PADDING` | Now governs all phases — UI label updated to "Crop Padding (all phases)" |

**Backward compatibility:** Existing saved configs containing `cnn_classifier_crop_padding` are silently ignored on load (unknown keys already dropped during config parsing). If the saved value differed from the default (i.e. the user had tuned it), a one-time `logger.warning()` is emitted on config load noting that `cnn_classifier_crop_padding` is no longer used and `INDIVIDUAL_CROP_PADDING` governs all phases.

---

## GUI Changes (`gui/main_window.py`)

- Remove `spin_cnn_crop_padding` widget, its label, its tooltip, and its wiring in config save/load.
- Update `INDIVIDUAL_CROP_PADDING` spinbox label to "Crop Padding (all phases)" and update its tooltip to reflect that pose, AprilTag, and CNN identity all use this value.

---

## `CNNIdentityConfig` Changes (`core/identity/cnn_identity.py`)

- Remove `crop_padding: float` field from `CNNIdentityConfig`.
- Any reference to `crop_padding` in `CNNIdentityBackend` or the precompute helpers is deleted.

---

## Future Extension Points

**Adding a new phase** (e.g. CNN age classifier):

1. Implement `PrecomputePhase` (e.g. `CNNAgePrecomputePhase` in `precompute.py`).
2. Add one `if` block in `_build_precompute_phases()`.
3. No changes to `UnifiedPrecompute`, `PosePipeline`, or the tracking loop.
4. The phase's `name` becomes its key in the results dict; wire the returned path into whatever downstream consumer needs it.

**Real-time mode** (future):

- The tracking loop calls `precompute.process_one_frame(frame_idx, frame)` per frame instead of running a full pre-pass.
- Each phase's `process_frame()` is called inline; phases with batch accumulation use batch size 1 or return buffered results.
- No protocol changes needed — `process_frame()` already operates frame-by-frame.

---

## Testing (`tests/test_unified_precompute.py`)

Unit tests — all phases mocked, no real video or GPU required.

| Test | What it covers |
|---|---|
| `CropConfig` defaults and field types | Dataclass correctness |
| `UnifiedPrecompute` with empty phase list returns empty dict, no video read | No-op path |
| `process_frame` called once per frame per phase | Dispatch correctness |
| `process_frame` called with empty crop list when frame has no detections | Empty-detection handling |
| `finalize` and `close` called on all phases after loop | Lifecycle correctness |
| `stop_check` returning True exits loop early, calls `close` not `finalize` | Cancellation path |
| All phases `has_cache_hit() = True` → video read skipped, existing paths returned | All-hit short-circuit |
| Mixed cache hits: hit phase is no-op, miss phase runs normally | Partial-hit path |
| Fatal phase raises in `finalize` → exception propagates from `run()` | Fatal error path |
| Non-fatal phase raises in `finalize` → `None` returned, other phases unaffected | Non-fatal error isolation |
| `close` called on all phases even when `finalize` raises | Cleanup correctness |
| `AprilTagPrecomputePhase.has_cache_hit()` returns True for existing compatible cache | AT cache hit |
| `CNNPrecomputePhase.process_frame` batches crops and flushes at batch boundary | Batch accumulation |
| `CNNPrecomputePhase.finalize` flushes remaining partial batch | Partial batch at end |
| `CNNPrecomputePhase.process_frame` with empty crops list does not add to batch | Empty frame no-op |

`tests/test_mat_cnn_identity.py` updated to remove `crop_padding` from `CNNIdentityConfig` test fixtures.

---

## Files Changed / Created

| File | Change |
|---|---|
| `src/multi_tracker/core/tracking/precompute.py` | **New** — `PrecomputePhase` protocol, `CropConfig`, `UnifiedPrecompute`, `AprilTagPrecomputePhase`, `CNNPrecomputePhase` |
| `src/multi_tracker/core/tracking/pose_pipeline.py` | Add `has_cache_hit()`, `process_frame()`, `finalize()`, `close()` — `PosePipeline` implements `PrecomputePhase` |
| `src/multi_tracker/core/tracking/worker.py` | Delete 3 precompute blocks + 6 helpers; add `_build_precompute_phases()` + `UnifiedPrecompute` wiring |
| `src/multi_tracker/core/identity/cnn_identity.py` | Remove `crop_padding` from `CNNIdentityConfig` |
| `configs/default.json` | Remove `cnn_classifier_crop_padding` |
| `src/multi_tracker/gui/main_window.py` | Remove `spin_cnn_crop_padding`; update `INDIVIDUAL_CROP_PADDING` label and tooltip |
| `tests/test_unified_precompute.py` | **New** — unified precompute unit tests |
| `tests/test_mat_cnn_identity.py` | Remove `crop_padding` from `CNNIdentityConfig` fixtures |
| `vulture_whitelist.py` | Add entries for `has_cache_hit`, `process_frame`, `finalize`, `close` on `PosePipeline` (protocol methods, not called by name) |
