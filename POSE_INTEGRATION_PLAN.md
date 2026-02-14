# Unified Individual-Properties Pipeline (YOLO-Only Analysis, BG Bootstrap Mode)

## Summary
Implement a single, cleaned-up **Analyze Individuals** pipeline that computes and caches per-detection properties for **all filtered detections** (not track-assigned), keyed by:

`INDIVIDUAL_PROPERTIES_ID = hash(DETECTION_HASH + FILTER_SETTINGS_HASH + EXTRACTOR_SETTINGS_HASH + SCHEMA_VERSION)`

This allows reuse when users change tracking-only settings.
Individual analysis (pose and future features) is **YOLO detection mode only**.
Background subtraction mode is explicitly treated as **bootstrap mode** for creating initial datasets used to train YOLO models.

Hard-fail policy: if precompute is entered and fails, abort run.

## Public Interfaces / Types Changes
1. Add pipeline manager and cache contract in:
`/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/identity/feature_runtime.py`
with:
`FeatureObservation`, `FeatureResult`, `ExtractorConfig`, `PoseExtractor`, `FeatureRuntimeManager`.

2. Add individual-properties cache API in:
`/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/identity/properties_cache.py`
with:
`compute_detection_hash(...)`, `compute_filter_settings_hash(...)`, `compute_extractor_hash(...)`, `compute_individual_properties_id(...)`, `load/save/get_frame/get_detection`.

3. Extend tracking params in:
`/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/gui/main_window.py`
new keys:
`ENABLE_INDIVIDUAL_PIPELINE`, `ENABLE_INDIVIDUAL_IMAGE_SAVE`,
`ENABLE_POSE_EXTRACTOR`, `POSE_MODEL_TYPE`, `POSE_MODEL_DIR`, `POSE_MIN_KPT_CONF_VALID`.
Add explicit scope guard:
`INDIVIDUAL_ANALYSIS_YOLO_ONLY = true` (constant/behavioral policy).

4. Config migration in:
`/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/gui/main_window.py`
- Map legacy `enable_individual_dataset` to new run/save split.
- Keep read compatibility, write only new keys.

5. Keep existing pose inference backend service usage through:
`/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/posekit/pose_inference.py`
with minimal MAT controls only:
model type (`yolo`/`sleap`), model dir/path, min valid keypoint confidence.

## Cache Identity Contract
1. `DETECTION_HASH` inputs:
`INFERENCE_MODEL_ID`, video fingerprint (absolute path + mtime + size), `START_FRAME`, `END_FRAME`, detection cache schema version.

2. `FILTER_SETTINGS_HASH` inputs:
`YOLO_CONFIDENCE_THRESHOLD`, `YOLO_IOU_THRESHOLD`, `ENABLE_SIZE_FILTERING`, `MIN_OBJECT_SIZE`, `MAX_OBJECT_SIZE`, ROI-mask digest, `DETECTION_METHOD`.
Enforcement: cache/use of individual properties only when `DETECTION_METHOD == yolo_obb`.

3. `EXTRACTOR_SETTINGS_HASH` (pose now):
`POSE_MODEL_TYPE`, pose model fingerprint (path + mtime + size), keypoint schema signature, `POSE_MIN_KPT_CONF_VALID`, extractor schema version.

4. Cache content:
- Per filtered detection keyed by `DetectionID`.
- Includes frame id, geometry refs, extractor outputs (pose keypoints/confidence summary).
- Optional side index per frame for fast retrieval.
- Unmatched detections are preserved in cache artifacts.

## Execution Flow
1. In `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/core/tracking/worker.py`, before assignment loop starts:
- If `ENABLE_INDIVIDUAL_PIPELINE`:
  - Require YOLO mode (`DETECTION_METHOD == yolo_obb`), else short-circuit with explicit reason.
  - Resolve `INDIVIDUAL_PROPERTIES_ID`.
  - Try cache hit.
  - On miss, run YOLO precompute.
  - Hard fail run on precompute execution error.

2. YOLO precompute stage:
- Iterate raw detections from detection cache over requested subset.
- Apply the same filtering function used at runtime.
- Generate per-detection crops once.
- Run pose extractor in batched mode.
- Persist properties cache.
- If `ENABLE_INDIVIDUAL_IMAGE_SAVE` is ON, persist `images/` artifacts; else use temp run storage only.

3. Tracking phase:
- Continue using existing tracking logic (no behavior change).
- Individual properties are read-only side data for export/future assignment-cost work.

4. Interpolated rows:
- Keep current interpolated crop flow.
- Add optional interpolated feature pass after interpolation stage, keyed separately by `(TrajectoryID, FrameID)` namespace.
- Join into `_with_pose` export where available.

## UI Cleanup (Replace Skeleton Inputs)
1. In `/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/gui/main_window.py`:
- Remove placeholder identity-method skeleton controls (color/apriltag/custom dummy classifier UI).
- Keep Analyze Individuals as the runtime pipeline tab.
- Keep Build Dataset tab as persistence/output controls only.
- Enforce dependency: save-images control disabled unless pipeline master toggle is ON.
- When detection method is background subtraction:
  - Disable individual-analysis controls.
  - Show guidance text: "Individual analysis requires YOLO mode. Use BG mode to bootstrap training data."

2. Minimal pose UI in Analyze Individuals:
- `Enable Pose Extractor`
- `Model Type: YOLO/SLEAP`
- `Model Dir/Path`
- `Min valid keypoint confidence`
- Status panel: cache hit/miss, precompute progress, artifact location.

## Scope Decision: YOLO-Only Analysis
1. Individual analysis pipeline (pose/tags/embeddings/future extractors) runs only in YOLO mode.
2. Background subtraction remains available for tracking and bootstrap dataset generation, but does not run individual-analysis extractors.
3. No BG pre-tracking individual-analysis prepass is implemented in this phase.

## Test Cases and Scenarios
1. Hash correctness:
- Tracking-only parameter changes do not invalidate individual-properties cache.
- Filter-setting or detection-hash changes do invalidate cache.
- Switching `DETECTION_METHOD` between YOLO and BG invalidates/segregates analysis usage.

2. Coverage of “all filtered detections”:
- Cached property count equals filtered detections count per frame.
- No dependence on assignment/matched tracks.

3. Hard-fail policy:
- Corrupt pose model path/backend crash during precompute aborts run with clear error.

4. Reuse behavior:
- Second run with same detection+filter+extractor hashes skips precompute and reuses cache.

5. UI/config:
- Legacy config loads correctly.
- New Analyze/Build split persists and restores correctly.
- Save-images toggle is contingent on pipeline toggle.
- BG mode disables Analyze Individuals execution controls with clear UX messaging.

6. Regression safety:
- Tracking outputs unchanged when pipeline OFF.
- Existing post-processing equivalence suite remains green.
- Existing detector/tracking tests remain green.

## Assumptions and Defaults
1. Cache backward compatibility for old individual-property artifacts is not required; bump schema and ignore old artifacts.
2. Default `POSE_MIN_KPT_CONF_VALID` is strict enough for usable outputs (e.g., 0.2) and configurable.
3. `_tracking_final_with_pose.csv` remains row-aligned to final tracking rows; unmatched detections stay in properties cache artifacts.
4. Product policy for this phase: YOLO is the only supported detection mode for individual analysis.
5. BG mode is intentionally positioned as a bootstrap workflow for collecting initial data to train YOLO models.
