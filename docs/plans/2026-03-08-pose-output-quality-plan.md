# Pose Output Quality Improvement Plan

**Date:** 2026-03-08
**Branch:** `mat-pose-integration`
**Status:** Proposed - ready for phased implementation

---

## Goal

Ensure that pose data written into final MAT results is quality-controlled enough to be trusted for:

- downstream quantitative analysis,
- pose-aware relinking and false-merge reduction,
- debugging difficult videos without mistaking raw network noise for valid anatomy.

The core policy should be conservative:

- keep the raw model output available for debugging,
- but make the default final pose export reflect cleaned, validated pose,
- and blank or downweight dubious pose instead of preserving plausible-looking garbage.

---

## What The Current Code Does

### Inference layer

- `src/multi_tracker/core/identity/pose/yolo_backend.py`
  - selects the best pose instance by mean keypoint confidence,
  - clamps/conf-normalizes keypoint confidence values,
  - returns `PoseResult(keypoints, mean_conf, valid_fraction, num_valid, num_keypoints)`.
- `src/multi_tracker/core/identity/pose/sleap_backend.py`
  - returns the same `PoseResult` shape through `_summarize_keypoints(...)`.
- `src/multi_tracker/core/identity/runtime_utils.py`
  - computes summary fields from raw keypoints,
  - but does not do anatomical validation, temporal consistency checks, or output masking.

### Cache layer

- `src/multi_tracker/core/tracking/worker.py`
  - writes raw global keypoints into `IndividualPropertiesCache` during pose precompute.
- `src/multi_tracker/core/identity/properties_cache.py`
  - stores raw `pose_keypoints` only,
  - recomputes `PoseMeanConf`, `PoseValidFraction`, `PoseNumValid`, and `PoseNumKeypoints` on read.

### Final export layer

- `src/multi_tracker/core/identity/properties_export.py`
  - merges cached pose into the final trajectories CSV,
  - applies `POSE_IGNORE_KEYPOINTS` to omitted landmarks,
  - computes summary statistics using `POSE_MIN_KPT_CONF_VALID`,
  - but still exports low-confidence keypoint coordinates unless they are explicitly ignored.
- `src/multi_tracker/gui/main_window.py`
  - builds the pose-augmented DataFrame,
  - merges interpolated pose rows,
  - rewrites `_with_pose.csv`,
  - and runs `relink_trajectories_with_pose(...)` on that merged result.

### Post-processing / relink layer

- `src/multi_tracker/core/post/processing.py`
  - uses normalized pose shape for fragment relinking,
  - honors `POSE_MIN_KPT_CONF_VALID` while building relink summaries,
  - but does not apply a dedicated pose-quality pass to the final per-frame outputs.

### Existing evaluation tooling

- `src/multi_tracker/posekit/ui/dialogs/evaluation.py`
  - already computes PCK, OKS, mean error, mean confidence, and worst frames,
  - which is useful for calibrating thresholds and validating changes,
  - but is not currently connected to MAT final-output QA.

---

## Findings

### 1. The system summarizes pose quality, but mostly does not enforce it

The current stack tracks pose confidence summaries well enough for diagnostics, but it does not consistently convert those diagnostics into output gating. `POSE_MIN_KPT_CONF_VALID` affects summary fields and some pose-aware association logic, yet the exported `PoseKpt_*` coordinates remain raw by default.

### 2. There is no explicit distinction between raw pose and trusted pose

The cache stores raw keypoints. The final export merges raw keypoints. The relinker consumes pose-shaped data from the merged output. That means a downstream consumer cannot easily tell whether a keypoint was:

- confidently visible,
- barely above threshold,
- anatomically implausible,
- temporally inconsistent,
- or filled by interpolation.

### 3. Final output quality is missing a dedicated pose post-processing stage

There is already strong post-processing for trajectories, but not an equivalent pass for pose itself. This is the main gap. A final pose pass should exist between raw pose merge and final CSV write.

### 4. Relinking uses pose continuity, but only weakly conditioned on pose trustworthiness

The relinker already rejects some bad joins using normalized pose distance. That is useful, but it currently depends on whether coordinates happen to be finite, not on a richer notion of pose quality.

### 5. A correctness bug exists in the interpolated pose export path

In `src/multi_tracker/gui/main_window.py`, the interpolated pose batching code appends crops to `_pending_crops`, but later calls:

```python
pose_backend.predict_batch([e["crop"] for e in _pending_entries])
```

`_pending_entries` does not currently store a `crop` field. That should be fixed immediately because it can break the auxiliary interpolated pose export path and obscure later QA work.

---

## Desired End State

By default, the final pose-augmented output should contain:

- pose rows that have been filtered through a shared quality module,
- per-keypoint invalidation for obviously bad landmarks,
- row-level quality scores and reason flags,
- explicit distinction between raw, cleaned, and interpolated pose,
- relinking that uses pose only when pose quality is actually strong enough.

The raw cache should remain available for reproducibility and debugging.

---

## Design Principles

1. Raw inference and trusted export are not the same thing.
2. Invalid pose should be blanked or downweighted, not cosmetically preserved.
3. Confidence alone is insufficient; use anatomy, crop quality, and temporal continuity too.
4. The quality logic must be shared across export, relink, and later tracking-time safeguards.
5. Keep the first UI surface small. Most thresholds can stay in config until behavior stabilizes.

---

## Proposed Phases

## Phase 0: Fix Immediate Correctness And Add Observability

### Phase 0 Objective

Repair the broken interpolated pose batching path and add the debug fields needed to evaluate pose quality decisions.

### Phase 0 Changes

- Fix the `_pending_entries` / `_pending_crops` bug in `src/multi_tracker/gui/main_window.py`.
- Add a lightweight shared helper for pose quality reporting, even before full filtering.
- Persist or derive enough context for later QA decisions:
  - detection confidence,
  - detection crop quality,
  - pose source (`cache`, `interpolated`, `cleaned`),
  - whether a row was altered by the quality pass.
- Add tests that cover:
  - interpolated pose batch execution,
  - export behavior when pose rows are missing,
  - relink input construction with and without pose.

### Phase 0 Primary Files

- `src/multi_tracker/gui/main_window.py`
- `src/multi_tracker/core/identity/properties_export.py`
- `tests/` for export-path coverage

### Phase 0 Deliverable

A stable pose export path with enough instrumentation to compare raw versus cleaned outputs.

---

## Phase 1: Introduce Shared Per-Frame Pose Quality Gating

### Phase 1 Objective

Create a single shared quality module that converts raw pose into cleaned pose plus explicit QA metadata.

### Phase 1 Proposed New Module

- `src/multi_tracker/core/identity/pose_quality.py`

### Phase 1 Responsibilities

- Accept raw keypoints and context:
  - keypoints `[K, 3]`,
  - detection size / shape,
  - detection crop quality,
  - detection confidence,
  - optional skeleton edges and keypoint names,
  - existing params like `POSE_MIN_KPT_CONF_VALID` and `POSE_IGNORE_KEYPOINTS`.
- Return:
  - cleaned keypoints,
  - per-keypoint validity mask,
  - row-level quality score,
  - failure reasons / flags.

### Phase 1 Minimum Checks For V1

- Confidence checks:
  - keypoints below `POSE_MIN_KPT_CONF_VALID` should not remain exported as trusted coordinates.
  - rows with too few valid landmarks should be rejected.
- Geometry checks:
  - reject NaN / inf / collapsed keypoint layouts,
  - reject coordinates grossly outside the crop or unreasonable relative to body size,
  - reject rows where required orientation anchors are absent when pose direction is requested.
- Skeleton checks when available:
  - use skeleton edges to reject extreme segment-length distortions,
  - estimate length priors from high-confidence frames within the same run rather than hard-coding species-specific values.
- Crop-context checks:
  - downweight pose from low-quality or tiny detections,
  - optionally reject rows when crop quality is below a floor.

### Phase 1 Export Policy

- Keep the cache raw.
- Clean pose on read/export.
- In the default `_with_pose.csv`:
  - invalid keypoints become `NaN` for `X`/`Y` and `0` or `NaN` confidence according to the chosen schema,
  - low-quality rows keep summary/debug fields but not trusted coordinates.
- Optionally add a debug artifact such as `_with_pose_raw.csv` or `pose_quality_report.csv` if needed.

### Phase 1 Suggested New Output Columns

- `PoseQualityScore`
- `PoseQualityState`
- `PoseQualityFlags`
- `PoseSource`
- `PoseWasCleaned`

### Phase 1 Primary Files

- `src/multi_tracker/core/identity/pose_quality.py`
- `src/multi_tracker/core/identity/properties_export.py`
- `src/multi_tracker/gui/main_window.py`

### Phase 1 Deliverable

A reusable quality gate that makes final pose outputs conservative by default.

---

## Phase 2: Add Temporal And Anatomical Pose Post-Processing

### Phase 2 Objective

Use trajectory context to improve pose stability after per-frame gating.

### Phase 2 Rationale

Per-frame confidence filtering is necessary but not sufficient. Many bad pose outputs look locally plausible but are temporally inconsistent, especially near contacts, occlusions, and fragmented tracks.

### Phase 2 Proposed Operations

- Gap-limited per-keypoint interpolation:
  - fill only short gaps,
  - only when neighboring frames are high-confidence and anatomically consistent.
- Temporal outlier rejection:
  - flag keypoints with implausible frame-to-frame jumps,
  - flag sudden body-length explosions or collapses,
  - flag orientation flips unsupported by motion or neighboring frames.
- Short-window smoothing:
  - use a conservative median or Savitzky-Golay style smoother on confident sequences,
  - never smooth across occlusion boundaries or long gaps.
- Multi-keypoint consistency:
  - reject or blank isolated landmarks that disagree strongly with the rest of the body.

### Phase 2 Integration Point

Run this after pose has been merged into the final DataFrame and before:

- `_with_pose.csv` is written,
- `relink_trajectories_with_pose(...)` is called.

That means the natural insertion point is the final export/relink flow in `src/multi_tracker/gui/main_window.py`.

### Phase 2 Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/core/identity/pose_quality.py`
- `src/multi_tracker/gui/main_window.py`

### Phase 2 Deliverable

A final pose pass that removes anatomically or temporally implausible output before the CSV becomes authoritative.

---

## Phase 3: Make Relinking And Pose-Aware Tracking Use Quality-Controlled Pose

### Phase 3 Objective

Stop using pose merely because it exists. Use it only when it is trustworthy.

### Phase 3 Relink Changes

- In `src/multi_tracker/core/post/processing.py`:
  - require minimum pose quality and visibility before pose contributes to fragment joins,
  - ignore pose distance when either endpoint is low-quality,
  - optionally penalize relink candidates that cross a local pose-quality collapse.

### Phase 3 Tracking / Association Follow-Up

- Later, reuse the same quality module in `src/multi_tracker/core/tracking/worker.py` so that:
  - pose-aware assignment rejection sees cleaned pose features,
  - noisy pose does not veto valid matches,
  - directional overrides use trustworthy landmarks only.

This should be a follow-up after export-time QA is stable, not the first step.

### Phase 3 Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/core/tracking/worker.py`
- `src/multi_tracker/core/tracking/pose_features.py`

### Phase 3 Deliverable

Pose influences tracking decisions only when pose quality is demonstrably good.

---

## Phase 4: Validation, Threshold Calibration, And Regression Coverage

### Phase 4 Objective

Validate that the new quality pass improves final outputs without silently over-blanking useful pose.

### Phase 4 Validation Strategy

- Use existing PoseKit evaluation tooling in `src/multi_tracker/posekit/ui/dialogs/evaluation.py` to measure:
  - PCK,
  - OKS,
  - mean keypoint error,
  - mean confidence,
  - worst-frame lists.
- Build a small MAT-specific validation set of clips with:
  - clean isolated animals,
  - close interactions,
  - crossings,
  - occlusions,
  - small / low-quality detections.
- Compare three modes:
  - raw export,
  - per-frame gated export,
  - gated + temporal/anatomical post-processed export.

### Phase 4 Success Metrics

- Fewer clearly implausible landmarks in final CSV outputs.
- Lower relink false positives in crowded clips.
- No significant degradation in good-frame pose coverage.
- Worst-frame tables become more concentrated on genuinely hard cases rather than obvious noise.

### Phase 4 Test Coverage

- Unit tests for the quality module:
  - low-confidence masking,
  - anchor-missing rejection,
  - segment-length outlier rejection,
  - temporal outlier suppression.
- Export tests:
  - cleaned columns written correctly,
  - raw cache unchanged,
  - interpolated pose merge respects quality fields.
- Relink tests:
  - low-quality pose does not drive joins,
  - high-quality pose still helps reject incompatible joins.

### Phase 4 Deliverable

A calibrated quality pass with measurable benefit and regression protection.

---

## Recommended Minimal Parameter Surface

Do not expose a large new UI surface initially. Start with a minimal stable subset.

### Reuse Existing Parameters

- `POSE_MIN_KPT_CONF_VALID`
- `POSE_IGNORE_KEYPOINTS`
- `POSE_DIRECTION_ANTERIOR_KEYPOINTS`
- `POSE_DIRECTION_POSTERIOR_KEYPOINTS`
- `RELINK_POSE_MAX_DISTANCE`

### Add Only A Few New Ones At First

- `POSE_EXPORT_MIN_VALID_FRACTION`
- `POSE_EXPORT_MIN_VALID_KEYPOINTS`
- `POSE_POSTPROC_MAX_GAP`
- `RELINK_MIN_POSE_QUALITY`

Everything else should remain config-only until the behavior is well understood.

---

## Suggested Implementation Order

1. Fix the interpolated pose batching bug in `src/multi_tracker/gui/main_window.py`.
2. Add `pose_quality.py` with per-frame gating only.
3. Route `_build_pose_augmented_dataframe(...)` through the quality gate.
4. Add QA columns to `_with_pose.csv`.
5. Make `relink_trajectories_with_pose(...)` require minimum pose quality.
6. Add temporal/anatomical post-processing.
7. Reuse the same logic in tracking-time pose features only after export-time behavior is validated.

---

## Acceptance Criteria

- Final pose CSV exports no longer present low-confidence or implausible keypoints as if they were trusted.
- Every exported pose row includes enough metadata to explain why it was accepted, modified, interpolated, or rejected.
- Pose-aware relinking uses only quality-controlled pose.
- The raw cache remains reproducible and debuggable.
- The interpolated pose export path runs without the current batching bug.
- Validation with PoseKit evaluation and worst-frame review shows improved output quality on hard clips.

---

## Summary

The repository already has most of the pieces needed for good pose quality control: normalized runtime outputs, raw pose caching, pose-aware association, pose-aware relinking, and a solid evaluation dashboard. What is missing is the explicit middle layer that turns raw pose into trusted pose before final export. The fastest path forward is therefore not a new model, but a shared pose-quality module plus a final post-processing pass that the export and relink pipeline both respect.
