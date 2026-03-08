# False-Merge Reduction Implementation Plan

**Date:** 2026-03-07
**Branch:** `mat-pose-integration`
**Status:** Proposed - ready for phased implementation

---

## Goal

Reduce false trajectory merges as aggressively as practical, even if that increases fragmentation. The system should prefer splitting an uncertain continuity link over preserving a long but incorrect identity chain.

This plan assumes a **split-first, relink-later** philosophy:

- False merges are more harmful than extra fragments.
- Relinking may remain conservative and incomplete.
- Stronger re-identification methods can be added later, but they are not required for this phase.

---

## Non-Goals

- Do not make merge or relink decisions based on appearance embeddings as positive identity evidence.
- Do not attempt full re-identification across the entire video.
- Do not optimize for the longest possible trajectories.
- Do not make risky joins simply because they are plausible.

Appearance features may be explored later for manual review or offline analysis. If appearance enters this plan at all, it should do so only as **split-only negative evidence** through change-point detection, never as positive evidence for merging or relinking.

---

## Current Relevant Pipeline

The current stack already includes several conservative pieces:

- Velocity and z-score based splitting in `src/multi_tracker/core/post/processing.py`
- Conservative forward/backward resolution in `src/multi_tracker/core/post/processing.py`
- Spatial redundancy removal and overlap-aware merging in `src/multi_tracker/core/post/processing.py`
- Greedy motion/pose-based fragment relinking in `src/multi_tracker/core/post/processing.py`
- Confidence and uncertainty persistence from `src/multi_tracker/core/tracking/worker.py`
- Forward/backward consistency scoring in `src/multi_tracker/core/tracking/optimizer.py`

This plan builds on that foundation by adding stronger negative evidence, more explicit ambiguity modeling, and stricter merge/relink gating.

---

## Design Principles

1. **Continuity must be earned.** A trajectory should continue only when multiple cues support continuity.
2. **Ambiguous windows are hostile territory.** Crossings, contacts, and short occlusions should be treated as likely failure zones.
3. **Negative evidence matters more than positive evidence.** Motion jumps, uncertainty spikes, pose discontinuities, and ambiguous assignments should veto continuity.
4. **Relinking should reject by default.** If a candidate join is not clearly correct, it should remain split.
5. **DetectionID is not a temporal identity signal.** It is frame-local and should only be used for overlap reconciliation, not long-term identity continuity.
6. **Appearance may veto continuity, but should not create it.** A temporally weird crop jump can justify a split, but should not justify a merge.

---

## Signals To Use

### Already Available

- `X`, `Y`, `Theta`, `FrameID`, `State`
- `DetectionConfidence`
- `AssignmentConfidence`
- `PositionUncertainty`
- `DetectionID` (frame-local only)
- Pose keypoints and pose confidence in pose-augmented CSVs

### Recommended New Signals

- Normalized assignment cost per matched track
- Kalman innovation magnitude per update
- Per-frame nearest-neighbor distance between active tracks
- Per-frame OBB overlap or center-distance crowding score
- Optional assignment margin: best cost vs second-best cost
- Optional split-only appearance change metrics:
  - pHash or dHash jump magnitude on canonicalized OBB crops
  - grayscale or HSV histogram distance jump
  - cheap low-dimensional crop embedding distance if it remains dataset-agnostic

These signals are useful because false merges usually happen around ambiguous assignments, not during stable isolated motion.

The appearance signals are intentionally listed as optional and **negative-only**. They are not meant to prove identity, only to flag temporal discontinuities that are suspicious enough to split.

---

## Proposed Phases

## Phase 0: Instrumentation and Diagnostics

### Objective

Add the missing tracking diagnostics needed to identify high-risk continuity failures.

### Changes

- Persist normalized assignment cost to tracking CSV output.
- Persist Kalman innovation magnitude or a closely related residual metric.
- Persist crowding features when possible:
  - nearest-neighbor distance
  - optional OBB overlap fraction
  - optional assignment margin
- Extend post-processing stats so new split causes are counted separately.

### Primary Files

- `src/multi_tracker/core/tracking/worker.py`
- `src/multi_tracker/core/assigners/hungarian.py`
- `src/multi_tracker/core/post/processing.py`
- `docs/developer-guide/confidence-metrics.md`

### Deliverable

A richer per-frame CSV that makes risky handoffs measurable instead of implicit.

---

## Phase 1: Ambiguity Window Detection

### Objective

Detect windows where identity continuity is inherently unreliable and treat them as high-risk for false merges.

### Rationale

Most false merges happen during close approaches, body contacts, short occlusions, or low-margin assignments. These windows should be marked explicitly and used to raise split sensitivity.

### Candidate Triggers

- Two active tracks move within a distance threshold based on body size.
- OBBs overlap beyond a threshold.
- Assignment confidence drops below a threshold.
- Assignment margin becomes small.
- Position uncertainty spikes.
- State transitions into or out of `occluded` or `lost` near another animal.

### Implementation Sketch

Add a helper that produces per-frame or per-row risk flags, for example:

- `RiskCrowding`
- `RiskLowAssignmentConfidence`
- `RiskHighUncertainty`
- `RiskOcclusionBoundary`
- `RiskAmbiguousWindow`

These flags can be computed either during tracking output or in post-processing from the final CSV and optional pose-augmented CSV.

### Primary Files

- `src/multi_tracker/core/tracking/worker.py`
- `src/multi_tracker/core/post/processing.py`

### Deliverable

A reusable ambiguity-window signal that later phases can use for split and relink gating.

---

## Phase 2: Multi-Cue False-Merge Break Scoring

### Objective

Replace single-cue break logic with a multi-cue score that prefers splitting when continuity becomes untrustworthy.

### Rationale

Velocity alone is too brittle. False merges often show up as a combination of moderate anomalies rather than one extreme jump.

### Candidate Cues

- Velocity z-score
- Heading jump
- Acceleration or jerk spike
- Curvature spike
- Assignment confidence drop
- Assignment cost spike
- Position uncertainty spike
- Ambiguity-window membership
- Pose discontinuity when pose data exists

### Decision Rule

Use a weighted score with a multi-cue requirement. Example:

`break_score = motion + geometry + uncertainty + assignment + pose + ambiguity`

Recommended policy:

- Split only if the total score exceeds a threshold.
- Also require either:
  - at least two independent cues to fire, or
  - a single extreme anomaly inside an ambiguity window.

This avoids overfragmenting clean trajectories while still being much more aggressive near risky events.

### Implementation Sketch

Add a new helper such as:

- `_compute_false_merge_breaks(...)`

Then integrate it into:

- `process_trajectories_from_csv(...)`
- `process_trajectories(...)`

Track new counters in `stats`, for example:

- `broken_ambiguity`
- `broken_assignment_confidence`
- `broken_uncertainty`
- `broken_pose_discontinuity`
- `broken_multicue`

### Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

### Deliverable

A new break pass that is more conservative than the current velocity-only approach and explicitly optimized for reducing false merges.

---

## Phase 2A: Optional Appearance Change-Point Detection

### Objective

Add a cheap image-based change-point detector that can split trajectories when the visual content of the tracked OBB crop changes too abruptly over time.

### Rationale

This is useful for false-merge prevention because a bad swap often produces a sudden crop discontinuity even when motion alone remains plausible. The key constraint is that appearance must be used only as **negative evidence** for splitting, not as positive evidence for merging.

### Best Placement In The Pipeline

The best insertion point is:

- **after interpolation**
- **after forward/backward resolution if backward tracking is enabled**
- **before conservative relinking**

Concretely, the easiest place to integrate this in the current application flow is alongside the final pose-augmented relink pipeline in `src/multi_tracker/gui/main_window.py`, immediately before `relink_trajectories_with_pose(...)` runs.

Recommended flow:

1. build final merged trajectory DataFrame
2. interpolate trajectories
3. build pose-augmented DataFrame if available
4. run appearance change-point split pass
5. run conservative relinking on the already-split result

This avoids two bad outcomes:

- running too early, before interpolation has produced the temporal support needed for continuity checks
- running too late, after relinking has already stitched a bad merge back together

### Data Source Strategy

The change-point pass should extract crops from raw video frames using per-row geometry from the final trajectory table.

Preferred geometry source order:

1. exact OBB geometry from detection cache when `DetectionID` is present
2. interpolated width, height, and angle across gaps when exact detections are missing
3. fallback to track-local median size if width and height cannot be reconstructed exactly

This means interpolation remains useful: it provides a dense temporal path for crop extraction, even across short gaps.

### Crop Strategy

- use masked OBB crops, not loose axis-aligned crops
- canonicalize orientation so the animal is compared in a consistent frame
- keep the crop cheap and deterministic
- ignore frames where crop extraction is too unstable or too small

Existing repository helpers already support much of this:

- OBB masked crop extraction in `src/multi_tracker/core/identity/analysis.py`
- lightweight perceptual hashing in `src/multi_tracker/tools/data_sieve/core.py`

### Recommended Methods

Start with the cheapest robust options first:

1. pHash over canonicalized masked OBB crops
2. dHash as a simpler gradient-sensitive fallback
3. grayscale or HSV histogram distance as a secondary cue

If a learned embedding is later tested, it should remain optional and off by default unless it proves stable on real data.

### Change-Point Logic

Per trajectory:

- compute a crop signature per valid frame
- compute frame-to-frame or window-to-window signature distance
- smooth distances with a short robust window
- flag change-points when the appearance jump is large relative to the local baseline

Recommended policy:

- do not split on appearance alone in normal frames
- allow appearance to trigger a split only when one of the following is also true:
  - the frame lies inside an ambiguity window
  - motion or uncertainty cues are also abnormal
  - the jump is extreme and persists for multiple frames

This keeps the method aligned with the overall split-first but artifact-aware design.

### Suggested Parameters

- `ENABLE_APPEARANCE_CHANGEPOINT_SPLITTING`
- `APPEARANCE_CHANGEPOINT_METHOD`
- `APPEARANCE_CHANGEPOINT_THRESHOLD`
- `APPEARANCE_CHANGEPOINT_WINDOW`
- `APPEARANCE_CHANGEPOINT_REQUIRE_AMBIGUITY`
- `APPEARANCE_CHANGEPOINT_MIN_VALID_CROPS`

These should remain advanced or config-only at first.

### Primary Files

- `src/multi_tracker/gui/main_window.py`
- `src/multi_tracker/core/post/processing.py`
- optional new helper module under `src/multi_tracker/core/post/`
- `src/multi_tracker/core/identity/analysis.py`
- `src/multi_tracker/tools/data_sieve/core.py`

### Deliverable

An optional split-only appearance change-point pass that catches temporally weird crop jumps before relinking.

---

## Phase 2A Implementation Checklist

The checklist below is intentionally ordered so the first slice can land without forcing a full architecture refactor.

### Recommended First Implementation Boundary

Ship the first version with these constraints:

- use pHash only
- use appearance only as split veto, never as merge or relink evidence
- run only in the final CSV path before relinking
- require either ambiguity-window support or another anomaly cue before splitting
- skip UI controls initially and keep the feature config-only

That keeps the first implementation small, measurable, and easy to disable.

### 1. Shared Signature Utilities

- [ ] Create a lightweight shared signature utility instead of importing `DataSieveCore` directly into post-processing.
- [ ] Move or duplicate only the small hash helpers needed from `src/multi_tracker/tools/data_sieve/core.py`.
- [ ] Recommended destination:
  - `src/multi_tracker/core/post/appearance_signatures.py`
  - or `src/multi_tracker/utils/image_signatures.py`
- [ ] Implement these helpers as pure functions:
  - `compute_phash(image, hash_size=8)`
  - `compute_dhash(image, hash_size=8)`
  - `compute_hist_signature(image, bins=32)`
  - `signature_distance(sig1, sig2, method="phash")`
  - `hamming_distance(hash1, hash2)`
- [ ] Keep the implementation free of heavyweight DataSieve-only dependencies.
- [ ] If `DataSieveCore` continues to expose these methods, make it delegate to the shared helpers to avoid logic drift.

### 2. Appearance Change-Point Module

- [ ] Create a dedicated helper module rather than bloating `processing.py` further.
- [ ] Recommended file:
  - `src/multi_tracker/core/post/appearance_changepoints.py`
- [ ] Add one public entry point:
  - `split_trajectories_with_appearance_changepoints(...)`
- [ ] Add private helpers with narrow responsibilities:
  - `_resolve_row_geometry(...)`
  - `_extract_canonical_crop(...)`
  - `_compute_row_signature(...)`
  - `_compute_trajectory_signature_series(...)`
  - `_detect_signature_changepoints(...)`
  - `_split_trajectory_at_frames(...)`
- [ ] Keep the module DataFrame-in/DataFrame-out so it composes cleanly with the existing post-processing path.

### 3. Geometry Source Decision

- [ ] Decide whether to persist `Width` and `Height` all the way through the final CSV path.
- [ ] If yes, add them to the tracked trajectory DataFrame whenever available and preserve them through interpolation and save/load.
- [ ] If no, implement lazy reconstruction using this priority order:
  - exact OBB corners from detection cache via `DetectionID`
  - size reconstruction via detection-cache shape info
  - interpolated or track-median width and height fallback
- [ ] Reuse or adapt the existing detection-size lookup logic in `src/multi_tracker/gui/main_window.py` rather than inventing a second incompatible path.
- [ ] Add a guard for rows where geometry cannot be resolved reliably.

### 4. Crop Extraction Path

- [ ] Reuse existing OBB masked crop behavior from `src/multi_tracker/core/identity/analysis.py` instead of inventing a new crop definition.
- [ ] Extract common crop code into a shared helper if duplication becomes awkward.
- [ ] Ensure crops are:
  - masked to the OBB region
  - padded consistently
  - canonicalized by angle before hashing
  - skipped if too small or mostly invalid
- [ ] Add hard guards for unstable crops:
  - minimum crop width and height
  - finite center and angle
  - finite width and height
  - non-empty masked region
- [ ] Keep crop extraction deterministic so the same trajectory produces the same signatures across runs.

### 5. Video and Cache Access Strategy

- [ ] Do not random-seek the video for every row.
- [ ] Iterate frames in ascending `FrameID` order and process all trajectory rows belonging to the current frame.
- [ ] Open the video once for the whole pass.
- [ ] Open the detection cache once for the whole pass if it is available.
- [ ] Gracefully skip the appearance pass if:
  - the video path is missing
  - the video cannot be opened
  - the detection cache is unavailable and fallback geometry is insufficient
- [ ] Log whether the pass ran, skipped, or partially degraded.

### 6. Signature Series Construction

- [ ] Restrict signature computation to rows with usable geometry and valid crops.
- [ ] Prefer `State == "active"` rows for baseline continuity.
- [ ] Allow interpolated rows to contribute only if crop geometry is credible.
- [ ] Store per-row debugging fields in-memory while computing:
  - `AppearanceSignatureValid`
  - `AppearanceDistancePrev`
  - `AppearanceJumpScore`
- [ ] Do not persist raw hash values to the final CSV unless there is a strong debugging need.

### 7. Change-Point Detection Policy

- [ ] Start with consecutive-frame signature distances.
- [ ] Smooth distances with a short median or robust rolling window.
- [ ] Compute a local baseline and detect unusually large positive jumps.
- [ ] Require one of the following before splitting:
  - ambiguity-window membership
  - co-occurring motion or uncertainty anomaly
  - extreme jump persisted for multiple valid frames
- [ ] Never allow appearance alone in a normal low-risk region to create a split in the first version.
- [ ] Emit candidate split frame IDs, not row offsets, so the splitter stays compatible with current DataFrame logic.

### 8. Integration Point In MainWindow

- [ ] Integrate the new pass inside `MainWindow._relink_final_pose_augmented_csv(...)`.
- [ ] Insert it after the final `relink_input_df` has been built and before `relink_trajectories_with_pose(...)` is called.
- [ ] Recommended flow inside that method:
  - load base final CSV
  - build pose-augmented DataFrame if available
  - run appearance change-point splitting on `relink_input_df`
  - feed the split result into `relink_trajectories_with_pose(...)`
  - rewrite final CSV and `_with_pose.csv` from the relinked result
- [ ] Keep the integration behind a parameter gate so the behavior can be disabled instantly.
- [ ] Ensure the pass runs for both forward-only and forward/backward workflows, since both converge on this method.

### 9. Parameter Wiring

- [ ] Add config-only parameters to the parameter dictionary path first.
- [ ] Recommended initial parameters:
  - `ENABLE_APPEARANCE_CHANGEPOINT_SPLITTING`
  - `APPEARANCE_CHANGEPOINT_METHOD`
  - `APPEARANCE_CHANGEPOINT_THRESHOLD`
  - `APPEARANCE_CHANGEPOINT_WINDOW`
  - `APPEARANCE_CHANGEPOINT_REQUIRE_AMBIGUITY`
  - `APPEARANCE_CHANGEPOINT_MIN_VALID_CROPS`
  - `APPEARANCE_CHANGEPOINT_MIN_CROP_SIZE`
- [ ] Thread the parameters through `MainWindow.get_parameters_dict()` and config load/save.
- [ ] Do not add UI controls in the first pass unless the feature stabilizes quickly.
- [ ] If a UI surface is later needed, put it under advanced post-processing settings.

### 10. Stats and Logging

- [ ] Add explicit counters for the new pass, for example:
  - `appearance_rows_considered`
  - `appearance_valid_crops`
  - `appearance_split_candidates`
  - `broken_appearance_changepoint`
- [ ] Log how many trajectories were examined and how many were split.
- [ ] Log the degrade mode when geometry or cache information is missing.
- [ ] Keep logs concise enough for long jobs.

### 11. Testing Checklist

- [ ] Add `tests/test_post_processing_appearance_changepoints.py`.
- [ ] Unit-test the shared signature helpers.
- [ ] Test that stable synthetic crops do not trigger splits.
- [ ] Test that an abrupt crop swap does trigger a split candidate.
- [ ] Test that tiny or invalid crops are skipped cleanly.
- [ ] Test that the pass is a no-op when disabled.
- [ ] Test that the pass does not crash when video or cache inputs are unavailable.
- [ ] Test that relinking receives the split result and not the original unsplit DataFrame.

### 12. Documentation Checklist

- [ ] Update this plan if helper names or integration points change during implementation.
- [ ] Add a short developer note after implementation explaining:
  - why appearance is split-only here
  - why the pass runs before relinking
  - which crop geometry fallbacks are used
- [ ] Update `docs/developer-guide/confidence-metrics.md` only if new persisted debug fields become user-visible.

### 13. Exact First Slice To Implement

- [ ] Create `appearance_signatures.py` with pHash and Hamming distance only.
- [ ] Create `appearance_changepoints.py` with one public `split_trajectories_with_appearance_changepoints(...)` function.
- [ ] Reuse the existing masked OBB crop behavior.
- [ ] Use only pHash in the first pass.
- [ ] Require ambiguity support or another anomaly cue before splitting.
- [ ] Call the new function from `MainWindow._relink_final_pose_augmented_csv(...)` before `relink_trajectories_with_pose(...)`.
- [ ] Add one focused test file covering stable crops, abrupt swaps, and disabled-mode no-op behavior.

If this first slice works well, then add dHash or histogram distance, stronger geometry persistence, and more detailed debug outputs in follow-up patches.

---

## Phase 3: Hardening Forward/Backward Merge Resolution

### Objective

Make forward/backward consensus merging much stricter so ambiguous overlaps do not collapse into false continuity.

### Rationale

The current resolver already emphasizes agreement, but false merges can still survive if overlap evidence is sparse or fragmented.

### New Gating Rules

- Require both absolute agreement count and agreement ratio.
- Require a minimum contiguous run of agreeing frames, not just scattered agreement.
- Require agreement near overlap boundaries when possible.
- Penalize alternating agree/disagree patterns.
- Reject merges if overlap occurs mostly inside a flagged ambiguity window.
- Prefer splitting when disagreement density is high, even if total agreement count passes threshold.

### Suggested New Parameters

- `MIN_OVERLAP_AGREEMENT_RATIO`
- `MIN_CONTIGUOUS_AGREEMENT_FRAMES`
- `MAX_DISAGREEMENT_SWITCHES`
- `MERGE_REJECT_IN_AMBIGUOUS_WINDOWS`

### Primary Files

- `src/multi_tracker/core/post/processing.py`

### Deliverable

A forward/backward merge stage that is less willing to average through ambiguous overlaps and more willing to preserve fragments.

---

## Phase 4: Conservative Relinking With Strong Reject Option

### Objective

Keep relinking conservative and explicitly biased toward leaving fragments unmatched unless continuity is very clear.

### Rationale

If the splitter becomes stronger, relinking must not undo those gains by greedily reconnecting risky fragments.

### Changes

- Keep motion and pose only; do not introduce appearance.
- Replace greedy pairing with a global matching step when feasible.
- Add a hard reject threshold so unmatched fragments remain unmatched by default.
- Prevent relinking across strong ambiguity windows unless the motion and pose score is exceptionally good.
- Penalize relinking across uncertainty spikes or low-confidence regions.

### Candidate Upgrade Path

1. Tighten current greedy relinking thresholds.
2. Add explicit rejection criteria.
3. Replace greedy chain-building with Hungarian or min-cost flow over fragment endpoints.

### Suggested New Parameters

- `RELINK_MAX_SCORE`
- `RELINK_REJECT_IN_AMBIGUOUS_WINDOWS`
- `RELINK_MAX_UNCERTAINTY`
- `RELINK_MIN_ASSIGNMENT_CONFIDENCE`

### Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

### Deliverable

A relinking stage that preserves precision even if recall is modest.

---

## Phase 5: Local Replay Around Suspect Events

### Objective

Use the detection cache to re-evaluate short windows around suspected merge points with stricter hypotheses.

### Rationale

False merges are often local mistakes. A short replay window can compare competing explanations without rerunning the entire video.

### Hypotheses To Compare

- `H0`: continuity is correct; keep one chain
- `H1`: continuity failed; split into two fragments

### Scoring Terms

- cumulative assignment cost
- uncertainty growth
- number of occlusion frames
- motion smoothness
- pose continuity if available
- penalty for passing through ambiguity windows as one identity

### Recommended Scope

- Start with short windows around only the highest-confidence suspect events.
- Keep this phase optional behind a parameter flag.

### Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/core/tracking/optimizer.py`
- optional new helper module under `src/multi_tracker/core/post/`

### Deliverable

A principled tie-breaker for difficult local cases without requiring full global replay.

---

## Recommended Implementation Order

1. Phase 0 instrumentation
2. Phase 1 ambiguity windows
3. Phase 2 multi-cue split scoring
4. Phase 2A optional appearance change-point splitting
5. Phase 3 stricter forward/backward merge gating
6. Phase 4 conservative relinking upgrades
7. Phase 5 optional local replay

This order is intentional:

- It improves observability first.
- It reduces false merges before relinking changes.
- It keeps higher-cost replay work until the core heuristics are validated.

---

## Proposed Parameter Surface

The following parameters should be added only as needed and kept hidden or advanced by default until stabilized:

- `ENABLE_FALSE_MERGE_GUARD`
- `FALSE_MERGE_REQUIRE_MULTI_CUE`
- `FALSE_MERGE_BREAK_SCORE_THRESHOLD`
- `FALSE_MERGE_AMBIGUITY_DISTANCE_MULTIPLIER`
- `FALSE_MERGE_UNCERTAINTY_ZSCORE_THRESHOLD`
- `FALSE_MERGE_ASSIGNMENT_COST_ZSCORE_THRESHOLD`
- `ENABLE_APPEARANCE_CHANGEPOINT_SPLITTING`
- `APPEARANCE_CHANGEPOINT_METHOD`
- `APPEARANCE_CHANGEPOINT_THRESHOLD`
- `APPEARANCE_CHANGEPOINT_WINDOW`
- `APPEARANCE_CHANGEPOINT_REQUIRE_AMBIGUITY`
- `MIN_OVERLAP_AGREEMENT_RATIO`
- `MIN_CONTIGUOUS_AGREEMENT_FRAMES`
- `MAX_DISAGREEMENT_SWITCHES`
- `RELINK_MAX_SCORE`
- `RELINK_REJECT_IN_AMBIGUOUS_WINDOWS`
- `ENABLE_LOCAL_REPLAY_ON_SUSPECT_WINDOWS`

The UI should expose only the minimal stable subset at first. The rest can stay in config or experimental settings until the behavior is well understood.

---

## Validation Plan

## Test Assets

Build a small benchmark set of short clips with known failure modes:

- simple isolated motion
- crossings without contact
- crossings with contact
- short occlusion and re-emergence
- dense crowding
- turning events without identity swaps

### Metrics

- false merges per 1000 frames
- number of split points near manually marked swap events
- fragment count increase relative to baseline
- relink precision on benchmark fragments
- forward/backward consistency improvement
- regression rate on clean isolated clips

### Acceptance Criteria

- false merges decrease substantially on crowded and crossing clips
- overfragmentation increase remains acceptable on clean clips
- relinking precision does not fall below the current conservative baseline
- no major regression in processing time for default settings

---

## Testing and Documentation

### New Tests

- `tests/test_post_processing_false_merge_breaks.py`
- `tests/test_post_processing_appearance_changepoints.py`
- `tests/test_post_processing_merge_gating.py`
- `tests/test_tracklet_relinking_conservative.py`
- optional replay-focused tests if Phase 5 lands

### Documentation Updates After Implementation

- update `docs/developer-guide/confidence-metrics.md`
- update `docs/developer-guide/tracking-optimization.md` if new metrics are surfaced in tuning
- add a developer note describing ambiguity windows and the split-first philosophy

---

## Concrete First Slice

The most practical first implementation slice is:

1. Log one or two new diagnostics from tracking:
   - normalized assignment cost
   - nearest-neighbor distance or crowding score
2. Add ambiguity-window detection in post-processing.
3. Add a multi-cue break helper that uses:
   - velocity z-score
   - assignment confidence or cost anomaly
   - uncertainty spike
   - ambiguity-window membership
4. Optionally add a pHash-based change-point veto before relinking.
5. Tighten merge gating to require agreement ratio plus contiguous agreement.
6. Leave relinking conservative and mostly unchanged for the first pass.

This slice should deliver the highest reduction in false merges with the lowest architectural disruption.

---

## Open Questions

- Should nearest-neighbor distance be computed during tracking or reconstructed later from the final CSV?
- Is assignment margin available cheaply enough to log without materially slowing tracking?
- Should ambiguity windows be computed at the frame level, the trajectory-row level, or both?
- Should local replay reuse existing optimizer code paths, or remain isolated in post-processing?
- Which advanced thresholds deserve UI exposure versus remaining config-only?

---

## Summary

The core strategy is to stop asking post-processing to prove identity and instead ask it to prove that continuity is still trustworthy. If continuity cannot be defended with motion, geometry, uncertainty, assignment quality, and optional pose continuity, the system should split.

That bias is the right one for this repository because conservative relinking already exists, stronger re-identification can be added later, and false merges are more damaging than extra fragments.
