# Multi-Animal Pose Contamination Mitigation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent single-animal pose estimation from being corrupted by overlapping animals in the crop field of view.

**Architecture:** Two complementary surgical fixes: (1) mask out other animals' OBB regions from each crop *before* inference so the pose model never sees them, and (2) zero confidence on any keypoint that, after back-projection to global frame coordinates, lands inside another animal's OBB *after* inference. Shared pure utility functions are added to `pose_features.py`; integration touches `worker.py` (primary precompute path) and `analysis.py` + `main_window.py` (interpolated path). Both options are guarded by config flags and have targeted unit tests.

**Tech Stack:** NumPy, OpenCV (`cv2.fillPoly`, `cv2.pointPolygonTest`)

---

## Background: Two Crop Extraction Code Paths

The codebase has two distinct places where crops are extracted and pose is run:

1. **Precompute path** (`src/multi_tracker/core/tracking/worker.py`, lines 858–876):
   - Iterates every frame in the detection cache.
   - For each detection, calls `_extract_expanded_obb_crop(frame, corners, padding_fraction)` — a *raw axis-aligned* crop with no OBB masking.
   - Has `filtered_obb_corners` (all detection corners for the frame) in scope.
   - Batch inference happens in `_flush_pose_batch()`, which back-projects crop-local keypoints to global frame coordinates.

2. **Interpolated-pose path** (`src/multi_tracker/core/identity/analysis.py`, line 322 + `src/multi_tracker/gui/main_window.py`, line 656):
   - Called for occluded frames during dataset generation.
   - Uses `gen._extract_obb_masked_crop(frame, corners, frame_h, frame_w)` — already masks pixels outside the target OBB but does not suppress other animals.

Both paths share the same problem but the precompute path is higher priority (more frames, more pose data).

---

## Option Inventory

### Option 1 — Cross-animal OBB exclusion mask (IMPLEMENT THIS SESSION)
Before passing a crop to the pose model, fill every pixel belonging to another detected animal's OBB with the background colour. The pose model never sees the contaminating animal.

- **Pros:** Root cause fix; works at inference time; no changes to model or training.
- **Cons:** Requires all OBBs per frame to be in scope at crop-extraction time.
- **Files:** `pose_features.py` (utility), `worker.py` (precompute), `analysis.py` (interpolated).

### Option 2 — Post-estimation foreign-OBB keypoint rejection (IMPLEMENT THIS SESSION)
After back-projecting pose keypoints to global frame coordinates, zero the confidence of any keypoint that falls geometrically inside another detected animal's OBB polygon.

- **Pros:** Works even when Option 1 is imperfect (OBB expansion may clip the mask); catches any remaining contamination; pure keypoint math.
- **Cons:** Requires storing per-frame OBB corners in the precompute batch accumulator.
- **Files:** `pose_features.py` (utility), `worker.py` (flush function).

### Option 3 — OBB overlap contamination quality score
In `assess_pose_row` (or `apply_quality_to_dataframe`), compute the IoU between the target OBB and the nearest other OBB. Propagate a `"overlap_contamination"` flag and degrade `PoseQualityScore` proportionally when IoU is high. Downstream relinking automatically respects the lowered score.

- **Pros:** Adds explainability; requires no re-running of inference.
- **Cons:** Needs OBB context passed alongside keypoints at export time — currently not stored in the cache.
- **Status:** Future work. Requires cache schema extension.

### Option 4 — Body-shape bimodality check
After pose estimation, check whether keypoints cluster into one body-sized blob or two. High inter-keypoint spread relative to the expected body size is a signal of two-animal contamination. Flag as `"bimodal_keypoints"`.

- **Pros:** Detects contamination even when OBBs are not available.
- **Cons:** Requires a reliable body-length prior (now available via `calibrate_body_length_prior`); threshold tuning needed.
- **Status:** Future work. Natural addition to `assess_pose_row`.

### Option 5 — Temporal vote for contaminated frames
When a frame has `"overlap_contamination"` or `"body_length_outlier"` flags, use the two neighbouring clean frames' normalised pose as candidates and pick the subset of keypoints geometrically closest to the prior-frame pose.

- **Pros:** Recovers pose information instead of just blanking it.
- **Cons:** Requires two valid neighbouring frames; adds complexity to temporal post-processing.
- **Status:** Future work. Extend `apply_temporal_pose_postprocessing`.

### Option 6 — Repulsive background inpainting
Instead of filling other-animal pixels with a constant background colour (Option 1), fill them with the maintained background model (`background_image` from the detector). The pose model sees a natural image with the contaminating animal erased.

- **Pros:** Removes the hard edge artifact that a solid colour introduces at OBB boundaries.
- **Cons:** Requires passing the background model to the crop extractor; slightly more memory.
- **Status:** Future work. Upgrade to Option 1 once Option 1 is validated.

### Option 7 — Kalman-state-conditioned keypoint weighting
Use the Kalman-predicted centroid and velocity to define a spatial prior. Weight keypoints by their distance from the predicted body centre — far-away keypoints are more likely to be contamination. Soft-gate rather than hard zero.

- **Pros:** Uses existing track state; no OBB geometry needed.
- **Cons:** Only works during tracking (not precompute); complex to tune.
- **Status:** Future work. Could extend `compute_detection_pose_features` in `pose_features.py`.

---

## Implementation Plan: Options 1 and 2

### Task 1: Add `apply_foreign_obb_mask` utility to `pose_features.py`

**Files:**
- Modify: `src/multi_tracker/core/tracking/pose_features.py`
- Test: `tests/test_pose_features.py`

**What it does:** Given a crop image and a list of other animals' OBB corner arrays (in *frame* coordinates), shifts them into crop-local coordinates and fills each polygon with a constant background colour using `cv2.fillPoly`.

**Step 1: Write the failing test**

Add to `tests/test_pose_features.py`:

```python
def test_apply_foreign_obb_mask_blanks_other_animal():
    import cv2
    from multi_tracker.core.tracking.pose_features import apply_foreign_obb_mask

    # 100×100 white frame crop placed at offset (10, 20) in the frame
    crop = np.full((100, 100, 3), 200, dtype=np.uint8)
    x_offset, y_offset = 10, 20

    # Another animal whose OBB (in frame coords) covers the top-left of the crop
    # OBB corners at frame (10,20), (50,20), (50,60), (10,60) → crop (0,0)–(40,40)
    other_corners = np.array([[10,20],[50,20],[50,60],[10,60]], dtype=np.float32)

    result = apply_foreign_obb_mask(crop, x_offset, y_offset, [other_corners], background_color=0)

    # Top-left corner (0,0) should be zeroed
    assert result[5, 5].sum() == 0, "pixel inside foreign OBB should be zeroed"
    # Bottom-right corner (80,80) should be unchanged
    assert result[80, 80].sum() > 0, "pixel outside foreign OBB should be untouched"


def test_apply_foreign_obb_mask_empty_list_returns_unchanged():
    from multi_tracker.core.tracking.pose_features import apply_foreign_obb_mask

    crop = np.full((50, 50, 3), 128, dtype=np.uint8)
    result = apply_foreign_obb_mask(crop, 0, 0, [], background_color=0)
    np.testing.assert_array_equal(result, crop)
```

**Step 2: Run to see failure**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py::test_apply_foreign_obb_mask_blanks_other_animal tests/test_pose_features.py::test_apply_foreign_obb_mask_empty_list_returns_unchanged -v
```
Expected: `ImportError` or `AttributeError` — function does not exist yet.

**Step 3: Implement in `pose_features.py`**

Add after the existing imports (add `import cv2` alongside existing imports):

```python
def apply_foreign_obb_mask(
    crop: np.ndarray,
    x_offset: int,
    y_offset: int,
    other_corners_list,
    background_color: int = 128,
) -> np.ndarray:
    """Zero out pixels in *crop* that belong to other animals' OBB regions.

    Args:
        crop: BGR image crop extracted from the full frame.
        x_offset: Horizontal offset of the crop's top-left corner in frame coords.
        y_offset: Vertical offset of the crop's top-left corner in frame coords.
        other_corners_list: Sequence of (4, 2) float32 arrays of OBB corners in
            *frame* coordinates for every other detected animal.
        background_color: Scalar fill value for suppressed regions (0–255).

    Returns:
        Modified copy of *crop* with other-animal regions filled.
    """
    if crop is None or not other_corners_list:
        return crop

    import cv2  # local import to avoid hard dependency at module level

    out = crop.copy()
    crop_h, crop_w = out.shape[:2]
    fill = int(np.clip(background_color, 0, 255))

    for corners in other_corners_list:
        try:
            arr = np.asarray(corners, dtype=np.float32)
            if arr.shape != (4, 2):
                continue
            # Shift from frame coordinates to crop-local coordinates
            local = arr.copy()
            local[:, 0] -= float(x_offset)
            local[:, 1] -= float(y_offset)
            # Clip to crop bounds before drawing
            local[:, 0] = np.clip(local[:, 0], 0, crop_w - 1)
            local[:, 1] = np.clip(local[:, 1], 0, crop_h - 1)
            poly = local.astype(np.int32)
            cv2.fillPoly(out, [poly], (fill, fill, fill))
        except Exception:
            continue

    return out
```

**Step 4: Run tests**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py::test_apply_foreign_obb_mask_blanks_other_animal tests/test_pose_features.py::test_apply_foreign_obb_mask_empty_list_returns_unchanged -v
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/pose_features.py tests/test_pose_features.py
git commit -m "feat(pose): add apply_foreign_obb_mask utility for cross-animal crop suppression"
```

---

### Task 2: Add `filter_keypoints_by_foreign_obbs` utility to `pose_features.py`

**Files:**
- Modify: `src/multi_tracker/core/tracking/pose_features.py`
- Test: `tests/test_pose_features.py`

**What it does:** Given global-frame keypoints `[K, 3]` for one detection and the full list of OBB corner arrays for all detections in the same frame, zero the confidence of any keypoint that lies geometrically inside another animal's OBB polygon.

**Step 1: Write the failing test**

Add to `tests/test_pose_features.py`:

```python
def test_filter_keypoints_by_foreign_obbs_zeros_contaminated():
    from multi_tracker.core.tracking.pose_features import filter_keypoints_by_foreign_obbs

    # Target is detection index 0.  Detection index 1 occupies x=[50,150], y=[50,150].
    all_corners = [
        np.array([[0,0],[30,0],[30,30],[0,30]], dtype=np.float32),   # target (idx 0)
        np.array([[50,50],[150,50],[150,150],[50,150]], dtype=np.float32),  # other (idx 1)
    ]
    # Two keypoints: one inside the other animal's OBB, one outside
    keypoints = np.array([[100.0, 100.0, 0.9],   # inside OBB 1 → should be zeroed
                          [10.0,  10.0,  0.8]], dtype=np.float32)  # outside → unchanged

    result = filter_keypoints_by_foreign_obbs(keypoints, all_corners, target_idx=0)

    assert result[0, 2] == 0.0, "keypoint inside foreign OBB should have conf zeroed"
    assert result[0, 0] == 100.0, "X coordinate should be preserved"
    assert abs(result[1, 2] - 0.8) < 1e-5, "keypoint outside foreign OBB should be unchanged"


def test_filter_keypoints_by_foreign_obbs_single_detection_unchanged():
    from multi_tracker.core.tracking.pose_features import filter_keypoints_by_foreign_obbs

    all_corners = [np.array([[0,0],[30,0],[30,30],[0,30]], dtype=np.float32)]
    keypoints = np.array([[15.0, 15.0, 0.9]], dtype=np.float32)

    result = filter_keypoints_by_foreign_obbs(keypoints, all_corners, target_idx=0)
    np.testing.assert_allclose(result, keypoints)
```

**Step 2: Run to see failure**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py::test_filter_keypoints_by_foreign_obbs_zeros_contaminated tests/test_pose_features.py::test_filter_keypoints_by_foreign_obbs_single_detection_unchanged -v
```
Expected: `ImportError` — function does not exist yet.

**Step 3: Implement in `pose_features.py`**

Add after `apply_foreign_obb_mask`:

```python
def filter_keypoints_by_foreign_obbs(
    keypoints,
    all_corners_list,
    target_idx: int,
) -> np.ndarray:
    """Zero confidence of keypoints that fall inside another animal's OBB.

    Operates on *global frame coordinates* (after crop back-projection).

    Args:
        keypoints: [K, 3] float32 array of (x, y, conf) in frame coordinates.
        all_corners_list: List of (4, 2) float32 OBB corner arrays for every
            detection in the frame (including the target).
        target_idx: Index into *all_corners_list* that identifies the current
            animal (its own OBB is skipped).

    Returns:
        Modified copy of *keypoints* with contaminated entries having conf=0.
    """
    if keypoints is None:
        return keypoints

    import cv2  # local import

    arr = np.asarray(keypoints, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return arr
    if not all_corners_list:
        return arr

    for j, corners in enumerate(all_corners_list):
        if j == target_idx:
            continue
        try:
            poly = np.asarray(corners, dtype=np.float32)
            if poly.shape != (4, 2):
                continue
            for k in range(len(arr)):
                if arr[k, 2] <= 0.0:
                    continue  # already zeroed
                x, y = float(arr[k, 0]), float(arr[k, 1])
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                dist = cv2.pointPolygonTest(poly, (x, y), measureDist=False)
                if dist >= 0.0:  # inside or on the boundary
                    arr[k, 2] = 0.0
        except Exception:
            continue

    return arr
```

**Step 4: Run tests**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py::test_filter_keypoints_by_foreign_obbs_zeros_contaminated tests/test_pose_features.py::test_filter_keypoints_by_foreign_obbs_single_detection_unchanged -v
```
Expected: both PASS.

**Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/pose_features.py tests/test_pose_features.py
git commit -m "feat(pose): add filter_keypoints_by_foreign_obbs utility for post-estimation contamination rejection"
```

---

### Task 3: Wire Option 1 into the precompute loop in `worker.py`

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py` (lines 858–880)

**Context:** The precompute loop already has `filtered_obb_corners` (all OBBs for the current frame). The crop extraction call is at line 870:

```python
crop, crop_offset = self._extract_expanded_obb_crop(
    frame, corners_arr, padding_fraction
)
```

After this call, `crop_offset` is `(x_min, y_min)`. The other animals' corners are `[filtered_obb_corners[j] for j in range(len(filtered_obb_corners)) if j != det_idx]`.

**Step 1: No new test needed** — the utility is tested in Task 1. The integration is verified by the full test suite passing.

**Step 2: Edit the crop loop** (lines 867–876)

Change from:
```python
                if ret and meas and filtered_obb_corners:
                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            pf["crops"].append(crop)
                            pf["crop_to_det"].append(det_idx)
                            pf["crop_offsets"][det_idx] = crop_offset
```

To:
```python
                if ret and meas and filtered_obb_corners:
                    for det_idx, corners in enumerate(filtered_obb_corners):
                        corners_arr = np.asarray(corners, dtype=np.float32)
                        crop, crop_offset = self._extract_expanded_obb_crop(
                            frame, corners_arr, padding_fraction
                        )
                        if crop is not None and crop.size > 0:
                            if len(filtered_obb_corners) > 1:
                                other_corners = [
                                    np.asarray(filtered_obb_corners[j], dtype=np.float32)
                                    for j in range(len(filtered_obb_corners))
                                    if j != det_idx
                                ]
                                crop = apply_foreign_obb_mask(
                                    crop,
                                    crop_offset[0],
                                    crop_offset[1],
                                    other_corners,
                                    background_color=pose_background_color,
                                )
                            pf["crops"].append(crop)
                            pf["crop_to_det"].append(det_idx)
                            pf["crop_offsets"][det_idx] = crop_offset
```

Where `pose_background_color` comes from `params.get("POSE_BACKGROUND_COLOR", 128)`.

Also add at the top of the precompute section (after `pose_enabled` is established):
```python
        pose_background_color = int(params.get("POSE_BACKGROUND_COLOR", 128))
```

And add to imports at the top of the precompute try-block or near the pose imports:
```python
        from multi_tracker.core.tracking.pose_features import apply_foreign_obb_mask
```

**Step 3: Run the full test suite**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -q --ignore=tests/test_pose_inference_sleap_service.py --ignore=tests/test_runtime_api_sleap_export.py --ignore=tests/test_sleap_export_predict_worker.py
```
Expected: all passing (same count as before).

**Step 4: Commit**

```bash
git add src/multi_tracker/core/tracking/worker.py
git commit -m "feat(pose): apply cross-animal OBB exclusion mask before pose inference (Option 1, precompute path)"
```

---

### Task 4: Wire Option 2 into `_flush_pose_batch` in `worker.py`

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py` (the `_flush_pose_batch` closure and the per-frame record `pf`)

**Context:** `_flush_pose_batch` already back-projects keypoints to global frame coordinates (`gkpts`). It needs access to all OBB corners for each frame to call `filter_keypoints_by_foreign_obbs`. The per-frame accumulator `pf` is the right place to store them.

**Step 1: Add `all_obb_corners` to the `pf` dict**

In the section that builds `pf` (around line 859), change:
```python
                pf: dict = {
                    "frame_idx": frame_idx,
                    "det_ids": detection_ids,
                    "n_dets": len(meas),
                    "crops": [],
                    "crop_to_det": [],
                    "crop_offsets": {},
                }
```
To:
```python
                pf: dict = {
                    "frame_idx": frame_idx,
                    "det_ids": detection_ids,
                    "n_dets": len(meas),
                    "crops": [],
                    "crop_to_det": [],
                    "crop_offsets": {},
                    "all_obb_corners": [
                        np.asarray(c, dtype=np.float32)
                        for c in (filtered_obb_corners or [])
                    ],
                }
```

**Step 2: Call `filter_keypoints_by_foreign_obbs` in `_flush_pose_batch`**

In `_flush_pose_batch`, after the back-projection of keypoints (after `gkpts[:, 0] += float(x0)` / `gkpts[:, 1] += float(y0)` lines, around line 793–796), add:

```python
                    if kpts is not None and crop_offset is not None and len(kpts) > 0:
                        x0, y0 = crop_offset
                        gkpts = np.asarray(kpts, dtype=np.float32).copy()
                        gkpts[:, 0] += float(x0)
                        gkpts[:, 1] += float(y0)
                        # Option 2: reject keypoints that fall inside other animals' OBBs
                        all_obbs = pf.get("all_obb_corners", [])
                        if len(all_obbs) > 1:
                            gkpts = filter_keypoints_by_foreign_obbs(
                                gkpts, all_obbs, target_idx=det_idx
                            )
```

Note: `det_idx` here is `pf["crop_to_det"][ci]`, which maps crop index back to detection index.

The full corrected block (replace lines ~789–797):

```python
                    if kpts is not None and crop_offset is not None and len(kpts) > 0:
                        x0, y0 = crop_offset
                        gkpts = np.asarray(kpts, dtype=np.float32).copy()
                        gkpts[:, 0] += float(x0)
                        gkpts[:, 1] += float(y0)
                        all_obbs = pf.get("all_obb_corners", [])
                        if len(all_obbs) > 1:
                            from multi_tracker.core.tracking.pose_features import (
                                filter_keypoints_by_foreign_obbs,
                            )
                            gkpts = filter_keypoints_by_foreign_obbs(
                                gkpts, all_obbs, target_idx=det_idx
                            )
                    else:
                        gkpts = kpts
```

Also add `det_idx = pf["crop_to_det"][ci]` before the `if kpts is not None` check (replacing or alongside the existing reference). Check the exact current code at lines 784–797 before editing to get the variable name right.

**Step 3: Run the full test suite**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -q --ignore=tests/test_pose_inference_sleap_service.py --ignore=tests/test_runtime_api_sleap_export.py --ignore=tests/test_sleap_export_predict_worker.py
```
Expected: all passing.

**Step 4: Commit**

```bash
git add src/multi_tracker/core/tracking/worker.py
git commit -m "feat(pose): reject keypoints inside foreign OBBs after pose inference (Option 2, precompute path)"
```

---

### Task 5: Wire Option 1 into the interpolated-pose path in `analysis.py`

**Files:**
- Modify: `src/multi_tracker/core/identity/analysis.py` — `_extract_obb_masked_crop` signature + call sites
- Modify: `src/multi_tracker/gui/main_window.py` — `gen._extract_obb_masked_crop(...)` call at line 656

**Context:** `_extract_obb_masked_crop` already masks pixels outside the target OBB. The new `other_corners_list` parameter adds a second masking pass for other animals' OBBs.

**Step 1: Extend `_extract_obb_masked_crop` signature** (line 446 in `analysis.py`)

Change:
```python
    def _extract_obb_masked_crop(self, frame, corners, frame_h, frame_w):
```
To:
```python
    def _extract_obb_masked_crop(self, frame, corners, frame_h, frame_w, other_corners_list=None):
```

At the end of the method, just before `return masked_crop, crop_info` (after line 512), add:

```python
        # Option 1: suppress other animals' OBB regions
        if other_corners_list:
            from multi_tracker.core.tracking.pose_features import apply_foreign_obb_mask
            masked_crop = apply_foreign_obb_mask(
                masked_crop, x_min, y_min, other_corners_list,
                background_color=int(self.background_color)
                if hasattr(self.background_color, "__int__") else 128,
            )
```

**Step 2: Update the call in `main_window.py`** (line 656)

Current call:
```python
                                    gen._extract_obb_masked_crop(
                                        frame, corners, h, w
                                    )
```

To pass other corners, we need all corners for the frame. In the interpolated-pose path, the other detections' corners should be obtainable from the detection cache for that frame. Look at the surrounding code (lines 640–680) to find `frame_idx` and the detection cache, then build:

```python
                                    other_corners = []
                                    if detection_cache is not None:
                                        try:
                                            (_, _, _, _, all_frame_corners, *_) = detection_cache.get_frame(frame_idx)
                                            if all_frame_corners:
                                                other_corners = [
                                                    np.asarray(c, dtype=np.float32)
                                                    for j, c in enumerate(all_frame_corners)
                                                    if j != det_idx_in_frame
                                                ]
                                        except Exception:
                                            pass
                                    gen._extract_obb_masked_crop(
                                        frame, corners, h, w,
                                        other_corners_list=other_corners,
                                    )
```

**Note:** Read lines 630–680 of `main_window.py` carefully before implementing this step, as the variable names for frame index and detection index may differ. The exact wiring depends on the loop structure. If `detection_cache` or `frame_idx` are not in scope at that point, this step can be deferred — Option 1 is already active in the more important precompute path from Task 3.

**Step 3: Run the full test suite**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -q --ignore=tests/test_pose_inference_sleap_service.py --ignore=tests/test_runtime_api_sleap_export.py --ignore=tests/test_sleap_export_predict_worker.py
```
Expected: all passing.

**Step 4: Commit**

```bash
git add src/multi_tracker/core/identity/analysis.py src/multi_tracker/gui/main_window.py
git commit -m "feat(pose): apply foreign OBB masking in interpolated pose path (Option 1, analysis path)"
```

---

### Task 6: Add config keys and `default.json` defaults

**Files:**
- Modify: `configs/default.json`

**Step 1: Add to `configs/default.json`** (before the final `}`)

```json
  "pose_background_color": 128
```

This is the fill colour for suppressed OBB regions. 128 (mid-grey) is a neutral choice that most pose models handle better than pure black (0) or pure white (255). Users with specific model training backgrounds can override this.

**Step 2: Run full tests**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -q --ignore=tests/test_pose_inference_sleap_service.py --ignore=tests/test_runtime_api_sleap_export.py --ignore=tests/test_sleap_export_predict_worker.py
```

**Step 3: Commit**

```bash
git add configs/default.json
git commit -m "feat(pose): add pose_background_color config default for foreign OBB masking"
```

---

### Task 7: Final verification

**Step 1: Run the complete targeted test set**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py tests/test_pose_quality.py tests/test_post_tracklet_relinking.py tests/test_track_assigner.py -v
```
Expected: all PASS.

**Step 2: Lint check**

```bash
conda run -n multi-animal-tracker-mps make lint
```

**Step 3: Format**

```bash
conda run -n multi-animal-tracker-mps make format
```

**Step 4: Final commit if formatting changed anything**

```bash
git add -p
git commit -m "style: apply formatter to pose contamination changes"
```

---

## Summary of Files Changed

| File | Change |
|---|---|
| `src/multi_tracker/core/tracking/pose_features.py` | +`apply_foreign_obb_mask`, +`filter_keypoints_by_foreign_obbs` |
| `src/multi_tracker/core/tracking/worker.py` | Apply Option 1 mask after crop extraction; store OBB corners in `pf`; apply Option 2 rejection in `_flush_pose_batch` |
| `src/multi_tracker/core/identity/analysis.py` | Extend `_extract_obb_masked_crop` with optional `other_corners_list` |
| `src/multi_tracker/gui/main_window.py` | Pass other corners to `_extract_obb_masked_crop` in interpolated path |
| `configs/default.json` | Add `pose_background_color` key |
| `tests/test_pose_features.py` | 4 new tests for the two utility functions |
