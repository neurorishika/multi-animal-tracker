# Pose-Aware Tracking Optimizer Implementation Plan

**Goal:** Make `TrackingOptimizer` and `TrackingPreviewWorker` use real pose keypoints when a pose properties cache is available, so parameters are tuned against the true tracking algorithm (advanced cost matrix with pose rejection) instead of the simplified one.

**Architecture:** Extract all pose utility functions from `TrackingWorker` into a pure shared module (`pose_features.py`) that has no Qt/thread dependencies. `TrackingWorker` delegates to those functions unchanged. `TrackingOptimizer._run_tracking_loop` and `TrackingPreviewWorker.run` import the same module to load the pose cache and build per-frame `association_data`, enabling the advanced `_compute_advanced_cost_matrix` path automatically when pose is present.

**Tech Stack:** Python, NumPy, `IndividualPropertiesCache` (existing NPZ-backed cache), `DetectionFilter` (existing), `TrackAssigner` (existing).

---

## Context: How the Pose Pipeline Works

1. During a full tracking run with `ENABLE_POSE_EXTRACTOR=True`, a **pose properties cache** (`.npz`) is written alongside the detection cache. Its path is stored in `params["INDIVIDUAL_PROPERTIES_CACHE_PATH"]`.
2. For each frame the cache stores: `detection_ids` (matching the detection cache IDs: `frame_idx * 10000 + i`) → `pose_keypoints` (shape `[K, 3]` = x, y, conf).
3. At association time the worker: (a) looks up detection IDs in the pose cache, (b) computes visibility + normalized keypoints per detection, (c) populates `association_data` which causes `compute_cost_matrix` to use `_compute_advanced_cost_matrix` (Stage-1 gate + pose rejection).
4. After fixing `_advanced_association_enabled` to require actual non-None data, the advanced path is only active when real keypoints exist — so the optimizer must supply them too.

## Key Files

| File | Role |
|------|------|
| `src/multi_tracker/core/tracking/worker.py` | Contains pose methods to be extracted; updated to import from shared module |
| `src/multi_tracker/core/tracking/pose_features.py` | **New**: shared pure-function module |
| `src/multi_tracker/core/tracking/optimizer.py` | Updated: `_run_tracking_loop` and `TrackingPreviewWorker.run` add pose support |
| `src/multi_tracker/core/identity/properties_cache.py` | Existing `IndividualPropertiesCache` — read-only use |
| `src/multi_tracker/core/detectors/engine.py` | `DetectionFilter.filter_raw_detections` — needs `detection_ids` arg passed |
| `tests/test_pose_features.py` | **New**: unit tests for extracted functions |
| `tests/test_track_assigner.py` | Add test that advanced path is triggered when pose data is non-None |

---

## Task 1: Create `pose_features.py` with Extracted Pure Functions

**Files:**
- Create: `src/multi_tracker/core/tracking/pose_features.py`

These functions are copied verbatim from `TrackingWorker` and converted from instance methods to module-level functions (drop `self`, use `math` directly instead of `self._normalize_theta`).

**Step 1: Write the new module**

```python
# src/multi_tracker/core/tracking/pose_features.py
"""
Shared pose-feature utilities used by TrackingWorker, TrackingOptimizer, and
TrackingPreviewWorker.  Pure functions — no Qt, no threads, no side effects.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_theta(theta: float) -> float:
    """Normalize radians to [0, 2*pi)."""
    try:
        value = float(theta)
    except Exception:
        value = 0.0
    return value % (2 * math.pi)


def parse_pose_group_tokens(raw_spec) -> List:
    """Parse keypoint group spec from list/tuple/string into tokens."""
    if raw_spec is None:
        return []
    if isinstance(raw_spec, str):
        raw_tokens = raw_spec.split(",")
    elif isinstance(raw_spec, (list, tuple)):
        raw_tokens = list(raw_spec)
    else:
        raw_tokens = [raw_spec]
    tokens = []
    for token in raw_tokens:
        t = str(token).strip()
        if not t:
            continue
        try:
            tokens.append(int(t))
        except Exception:
            tokens.append(t)
    return tokens


def resolve_pose_group_indices(raw_spec, keypoint_names: List[str]) -> List[int]:
    """Resolve keypoint group names/indices to a deduplicated index list."""
    tokens = parse_pose_group_tokens(raw_spec)
    indices = []
    seen = set()
    for token in tokens:
        if isinstance(token, int):
            if 0 <= token < len(keypoint_names) and token not in seen:
                indices.append(token)
                seen.add(token)
        else:
            name = str(token).strip().lower()
            for i, kn in enumerate(keypoint_names):
                if kn.strip().lower() == name and i not in seen:
                    indices.append(i)
                    seen.add(i)
    return indices


def build_pose_detection_keypoint_map(pose_props_cache, frame_idx: int) -> Dict[int, Any]:
    """Return {detection_id: keypoints_array} for one frame from the pose cache."""
    if pose_props_cache is None:
        return {}
    try:
        frame = pose_props_cache.get_frame(int(frame_idx))
    except Exception:
        return {}
    ids = frame.get("detection_ids", [])
    keypoints = frame.get("pose_keypoints", [])
    n = min(len(ids), len(keypoints))
    out: Dict[int, Any] = {}
    for i in range(n):
        try:
            det_id = int(ids[i])
        except Exception:
            continue
        out[det_id] = keypoints[i]
    return out


def compute_pose_geometry_from_keypoints(
    keypoints,
    anterior_indices: List[int],
    posterior_indices: List[int],
    min_valid_conf: float,
    ignore_indices: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract heading, body length, and visibility from pose keypoints."""
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return None

    ignore_set = {int(idx) for idx in (ignore_indices or [])}

    def weighted_centroid(indices):
        pts, weights = [], []
        for idx in indices:
            if idx in ignore_set or idx < 0 or idx >= len(arr):
                continue
            x, y, conf = arr[idx]
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)):
                continue
            if float(conf) < float(min_valid_conf):
                continue
            pts.append((float(x), float(y)))
            weights.append(max(1e-6, float(conf)))
        if not pts:
            return None
        pts_a = np.asarray(pts, dtype=np.float64)
        w_a = np.asarray(weights, dtype=np.float64)
        return float(np.average(pts_a[:, 0], weights=w_a)), float(np.average(pts_a[:, 1], weights=w_a))

    valid_total = visible_total = 0
    for idx in range(len(arr)):
        if idx in ignore_set:
            continue
        valid_total += 1
        conf = arr[idx, 2]
        if np.isfinite(conf) and float(conf) >= float(min_valid_conf):
            visible_total += 1
    visibility = float(visible_total) / float(valid_total) if valid_total > 0 else 0.0

    ant = weighted_centroid(anterior_indices)
    post = weighted_centroid(posterior_indices)
    if ant is None or post is None:
        return {"heading": None, "body_length": None, "visibility": float(np.clip(visibility, 0.0, 1.0))}
    dx, dy = ant[0] - post[0], ant[1] - post[1]
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return {"heading": None, "body_length": None, "visibility": float(np.clip(visibility, 0.0, 1.0))}
    return {
        "heading": normalize_theta(math.atan2(dy, dx)),
        "body_length": float(math.hypot(dx, dy)),
        "visibility": float(np.clip(visibility, 0.0, 1.0)),
    }


def normalize_pose_keypoints(
    keypoints,
    min_valid_conf: float,
    ignore_indices: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Center and scale pose keypoints for shape comparison."""
    if keypoints is None:
        return None
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return None
    ignore_set = {int(idx) for idx in (ignore_indices or [])}
    valid = np.zeros(len(arr), dtype=bool)
    valid_points, valid_weights = [], []
    for idx in range(len(arr)):
        if idx in ignore_set:
            continue
        x, y, conf = arr[idx]
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(conf) and float(conf) >= float(min_valid_conf):
            valid[idx] = True
            valid_points.append((float(x), float(y)))
            valid_weights.append(max(1e-6, float(conf)))
    if not valid_points:
        return None
    pts_a = np.asarray(valid_points, dtype=np.float64)
    w_a = np.asarray(valid_weights, dtype=np.float64)
    cx = float(np.average(pts_a[:, 0], weights=w_a))
    cy = float(np.average(pts_a[:, 1], weights=w_a))
    centered = pts_a - np.array([[cx, cy]], dtype=np.float64)
    radii = np.sqrt(np.sum(centered**2, axis=1))
    scale = float(np.median(radii[radii > 1e-6])) if np.any(radii > 1e-6) else 1.0
    scale = max(scale, 1.0)
    out = np.full((len(arr), 3), np.nan, dtype=np.float32)
    out[:, 2] = 0.0
    for src_idx, kp_idx in enumerate(np.where(valid)[0]):
        out[kp_idx, 0] = np.float32(centered[src_idx, 0] / scale)
        out[kp_idx, 1] = np.float32(centered[src_idx, 1] / scale)
        out[kp_idx, 2] = np.float32(arr[kp_idx, 2])
    return out


def load_pose_context_from_params(
    params: Dict[str, Any],
) -> Tuple[Any, List[int], List[int], List[int], bool]:
    """
    Open the pose properties cache and resolve keypoint group indices.

    Returns
    -------
    pose_props_cache : IndividualPropertiesCache | None
        Opened in read mode, or None if pose is disabled / cache missing.
    anterior_indices : list[int]
    posterior_indices : list[int]
    ignore_indices : list[int]
    pose_direction_enabled : bool
        True only when both anterior and posterior groups are non-empty.
    """
    pose_enabled = bool(params.get("ENABLE_POSE_EXTRACTOR", False))
    cache_path = str(params.get("INDIVIDUAL_PROPERTIES_CACHE_PATH", "") or "").strip()
    if not pose_enabled or not cache_path or not os.path.exists(cache_path):
        return None, [], [], [], False

    from multi_tracker.core.identity.properties_cache import IndividualPropertiesCache

    pose_props_cache = IndividualPropertiesCache(cache_path, mode="r")
    if not pose_props_cache.is_compatible():
        logger.warning("Pose cache incompatible, pose direction disabled: %s", cache_path)
        pose_props_cache.close()
        return None, [], [], [], False

    names_raw = pose_props_cache.metadata.get("pose_keypoint_names", [])
    keypoint_names = [str(v) for v in names_raw] if isinstance(names_raw, (list, tuple)) else []

    ignore_indices = resolve_pose_group_indices(params.get("POSE_IGNORE_KEYPOINTS", []), keypoint_names)
    anterior_indices = resolve_pose_group_indices(params.get("POSE_DIRECTION_ANTERIOR_KEYPOINTS", []), keypoint_names)
    posterior_indices = resolve_pose_group_indices(params.get("POSE_DIRECTION_POSTERIOR_KEYPOINTS", []), keypoint_names)

    pose_direction_enabled = bool(anterior_indices and posterior_indices)
    if not pose_direction_enabled:
        logger.info("Pose direction disabled: define both anterior/posterior keypoint groups.")
    return pose_props_cache, anterior_indices, posterior_indices, ignore_indices, pose_direction_enabled


def compute_detection_pose_features(
    detection_ids: List[int],
    pose_keypoint_map: Dict[int, Any],
    anterior_indices: List[int],
    posterior_indices: List[int],
    ignore_indices: List[int],
    min_valid_conf: float,
) -> Tuple[List, np.ndarray]:
    """
    For each detection, look up its pose keypoints and compute:
    - normalized keypoints (for shape-based matching)
    - visibility score

    Returns
    -------
    detection_pose_keypoints : list[ndarray | None]  length == len(detection_ids)
    detection_pose_visibility : ndarray float32       length == len(detection_ids)
    """
    n = len(detection_ids)
    detection_pose_keypoints = [None] * n
    detection_pose_visibility = np.zeros(n, dtype=np.float32)

    for det_idx in range(n):
        try:
            det_id = int(detection_ids[det_idx])
        except Exception:
            continue
        keypoints = pose_keypoint_map.get(det_id)
        features = compute_pose_geometry_from_keypoints(
            keypoints, anterior_indices, posterior_indices, min_valid_conf, ignore_indices
        )
        if features is None:
            continue
        detection_pose_visibility[det_idx] = float(features.get("visibility", 0.0) or 0.0)
        detection_pose_keypoints[det_idx] = normalize_pose_keypoints(
            keypoints, min_valid_conf, ignore_indices=ignore_indices
        )
    return detection_pose_keypoints, detection_pose_visibility
```

**Step 2: Verify module is importable**

```bash
cd "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker"
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.core.tracking.pose_features import load_pose_context_from_params; print('OK')"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add src/multi_tracker/core/tracking/pose_features.py
git commit -m "feat(pose): extract pose utility functions into shared pose_features module"
```

---

## Task 2: Unit Tests for `pose_features.py`

**Files:**
- Create: `tests/test_pose_features.py`

**Step 1: Write tests**

```python
# tests/test_pose_features.py
"""Unit tests for the shared pose_features module."""
import math
import numpy as np
import pytest
from multi_tracker.core.tracking.pose_features import (
    normalize_theta,
    parse_pose_group_tokens,
    resolve_pose_group_indices,
    build_pose_detection_keypoint_map,
    compute_pose_geometry_from_keypoints,
    normalize_pose_keypoints,
    compute_detection_pose_features,
)


def test_normalize_theta_wraps():
    assert abs(normalize_theta(0.0)) < 1e-6
    assert abs(normalize_theta(2 * math.pi)) < 1e-6
    assert abs(normalize_theta(-math.pi) - math.pi) < 1e-6


def test_parse_pose_group_tokens_string():
    assert parse_pose_group_tokens("0, head, 2") == [0, "head", 2]


def test_resolve_pose_group_indices_by_name():
    names = ["thorax", "head", "abdomen"]
    result = resolve_pose_group_indices(["head", "thorax"], names)
    assert set(result) == {0, 1}


def test_resolve_pose_group_indices_by_int():
    names = ["a", "b", "c"]
    assert resolve_pose_group_indices([0, 2], names) == [0, 2]


def test_resolve_pose_group_indices_deduplicates():
    names = ["a", "b"]
    assert resolve_pose_group_indices([0, 0, 1], names) == [0, 1]


def test_build_pose_detection_keypoint_map_none_cache():
    assert build_pose_detection_keypoint_map(None, 0) == {}


def test_compute_pose_geometry_from_keypoints_basic():
    # Two keypoints: anterior at (10,0), posterior at (0,0)
    kpts = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["heading"] is not None
    assert abs(result["heading"]) < 0.1 or abs(result["heading"] - 2 * math.pi) < 0.1
    assert result["body_length"] == pytest.approx(10.0, abs=0.1)
    assert result["visibility"] == pytest.approx(1.0, abs=0.01)


def test_compute_pose_geometry_from_keypoints_low_conf_returns_none_heading():
    kpts = np.array([[10.0, 0.0, 0.05], [0.0, 0.0, 0.05]], dtype=np.float32)
    result = compute_pose_geometry_from_keypoints(kpts, [0], [1], min_valid_conf=0.1)
    assert result is not None
    assert result["heading"] is None


def test_normalize_pose_keypoints_centered():
    kpts = np.array([[2.0, 0.0, 0.9], [-2.0, 0.0, 0.9]], dtype=np.float32)
    out = normalize_pose_keypoints(kpts, min_valid_conf=0.1)
    assert out is not None
    # Center should be at (0,0) after normalization
    assert abs(float(out[0, 0]) + float(out[1, 0])) < 1e-5


def test_normalize_pose_keypoints_none_input():
    assert normalize_pose_keypoints(None, 0.2) is None


def test_compute_detection_pose_features_no_match():
    kpt_map = {}
    kpts, vis = compute_detection_pose_features([12345], kpt_map, [0], [1], [], 0.2)
    assert kpts == [None]
    assert vis[0] == pytest.approx(0.0)


def test_compute_detection_pose_features_with_match():
    det_id = 42
    kpts_raw = np.array([[10.0, 0.0, 0.9], [0.0, 0.0, 0.9]], dtype=np.float32)
    kpt_map = {det_id: kpts_raw}
    kpts, vis = compute_detection_pose_features([det_id], kpt_map, [0], [1], [], 0.2)
    assert kpts[0] is not None
    assert vis[0] > 0.0
```

**Step 2: Run tests to verify they pass**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_pose_features.py -v
```
Expected: all tests PASS

**Step 3: Commit**

```bash
git add tests/test_pose_features.py
git commit -m "test(pose): unit tests for shared pose_features module"
```

---

## Task 3: Update `TrackingWorker` to Delegate to `pose_features`

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py`

Replace all `self._build_pose_detection_keypoint_map`, `self._compute_pose_geometry_from_keypoints`, `self._normalize_pose_keypoints`, `self._normalize_theta`, `self._resolve_pose_group_indices`, `self._parse_pose_group_tokens` call sites with imports from `pose_features`. Keep the method definitions temporarily as thin wrappers (or delete them); the bodies become single-line delegations to the module. **Do not change any other logic.**

**Step 1: Add import at top of worker.py** (after existing imports)

```python
from multi_tracker.core.tracking.pose_features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
    compute_pose_geometry_from_keypoints as _pf_compute_geometry,
    normalize_pose_keypoints as _pf_normalize_keypoints,
    normalize_theta as _pf_normalize_theta,
    resolve_pose_group_indices as _pf_resolve_indices,
    load_pose_context_from_params as _pf_load_pose_context,
    compute_detection_pose_features as _pf_compute_det_features,
)
```

**Step 2: Replace pose setup block in `run()` (lines ~1498–1561)**

The existing block that opens `IndividualPropertiesCache`, reads keypoint names, resolves indices, and sets `pose_direction_enabled` is replaced with a single call:

```python
# Replace everything from "pose_props_cache = None" to "else: logger.info(..."
(
    pose_props_cache,
    pose_direction_anterior_indices,
    pose_direction_posterior_indices,
    pose_ignore_indices,
    pose_direction_enabled,
) = _pf_load_pose_context(p)
if pose_direction_enabled:
    logger.info(
        "Pose direction override enabled: anterior=%s, posterior=%s",
        pose_direction_anterior_indices,
        pose_direction_posterior_indices,
    )
```

**Step 3: Replace per-frame pose block (~lines 1938–1973)**

Replace the inner `if pose_direction_enabled and meas and detection_ids:` block with:

```python
if pose_direction_enabled and meas and detection_ids:
    if pose_frame_keypoints_map_frame != actual_frame_index:
        pose_frame_keypoints_map = _pf_build_keypoint_map(
            pose_props_cache, actual_frame_index
        )
        pose_frame_keypoints_map_frame = actual_frame_index

    detection_pose_keypoints, detection_pose_visibility = _pf_compute_det_features(
        detection_ids,
        pose_frame_keypoints_map,
        pose_direction_anterior_indices,
        pose_direction_posterior_indices,
        pose_ignore_indices,
        pose_min_valid_conf,
    )
    # --- heading override (kept as-is, uses detection_pose_heading separately) ---
    # NOTE: _pf_compute_det_features does NOT fill detection_pose_heading.
    # The heading override loop below this block still reads it from the
    # original per-detection code — leave lines 1975-2005 unchanged.
```

Wait — `detection_pose_heading` is filled separately in the original code for the heading override. `compute_detection_pose_features` only computes keypoints and visibility (for association). The heading for the KF correction comes from `_compute_pose_geometry_from_keypoints` called inline. Keep that inline block, but extract only the keypoints/visibility calls to use shared functions.

**Revised Step 3 approach** — minimize diff, only replace the keypoints + visibility sub-computation:

Inside the existing `for det_idx in range(n_det):` loop, replace:

```python
# OLD:
detection_pose_keypoints[det_idx] = self._normalize_pose_keypoints(
    keypoints, pose_min_valid_conf, ignore_indices=pose_ignore_indices,
)
```

with:

```python
detection_pose_keypoints[det_idx] = _pf_normalize_keypoints(
    keypoints, pose_min_valid_conf, ignore_indices=pose_ignore_indices,
)
```

And replace the calls to `self._compute_pose_geometry_from_keypoints` with `_pf_compute_geometry`, `self._normalize_theta` with `_pf_normalize_theta`, etc.

**Step 4: Run full test suite**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -x -q --tb=short --ignore=tests/test_pose_inference_sleap_service.py
```
Expected: all existing tests pass (pre-existing sleap test excluded — missing file unrelated to this work)

**Step 5: Commit**

```bash
git add src/multi_tracker/core/tracking/worker.py
git commit -m "refactor(worker): delegate pose utilities to shared pose_features module"
```

---

## Task 4: Add Pose Support to `_run_tracking_loop` in `optimizer.py`

This is the core fix. The optimizer's inner loop gains pose-aware `association_data` building.

**Files:**
- Modify: `src/multi_tracker/core/tracking/optimizer.py`

**Step 1: Add import at top of optimizer.py**

```python
from multi_tracker.core.tracking.pose_features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
    compute_detection_pose_features as _pf_compute_det_features,
    load_pose_context_from_params as _pf_load_pose_context,
)
```

**Step 2: Update `_run_tracking_loop` setup (after `_roi_mask = ...`)**

Add pose setup after line `_roi_mask = params.get("ROI_MASK", None)`:

```python
# Load pose context — mirrors TrackingWorker behaviour.
(
    _pose_cache,
    _pose_anterior,
    _pose_posterior,
    _pose_ignore,
    _pose_enabled,
) = _pf_load_pose_context(params)
_pose_min_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
_pose_kpt_map: Dict[int, Any] = {}
_pose_kpt_map_frame: int | None = None
track_pose_prototypes: list = [None] * N  # updated after each match
```

**Step 3: Update cache unpacking in the frame loop**

Change:
```python
raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb, *_ = (
    self.cache.get_frame(f_idx)
)
```
To:
```python
raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb, raw_det_ids, *_ = (
    self.cache.get_frame(f_idx)
)
```

**Step 4: Update `filter_raw_detections` call to capture `detection_ids`**

Change:
```python
filtered = det_filter.filter_raw_detections(
    raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb,
    roi_mask=_roi_mask,
)
meas, _, shapes, _confs, _obb_out, *_ = filtered
```
To:
```python
filtered = det_filter.filter_raw_detections(
    raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb,
    roi_mask=_roi_mask,
    detection_ids=raw_det_ids,
)
meas, _, shapes, _confs, _obb_out, detection_ids, *_ = filtered
```

**Step 5: Compute pose features per frame and build `association_data`**

After `kf_manager.predict()` and before cost matrix (insert between predict and the `if meas:` check):

```python
# --- Pose features for this frame ---
_det_pose_kpts = [None] * len(meas)
_det_pose_vis = np.zeros(len(meas), dtype=np.float32)
if _pose_enabled and meas and detection_ids:
    if _pose_kpt_map_frame != f_idx:
        _pose_kpt_map = _pf_build_keypoint_map(_pose_cache, f_idx)
        _pose_kpt_map_frame = f_idx
    _det_pose_kpts, _det_pose_vis = _pf_compute_det_features(
        [int(d) for d in detection_ids],
        _pose_kpt_map,
        _pose_anterior,
        _pose_posterior,
        _pose_ignore,
        _pose_min_conf,
    )

_association_data: Dict[str, Any] = {
    "detection_pose_keypoints": _det_pose_kpts,
    "detection_pose_visibility": _det_pose_vis,
    "track_pose_prototypes": track_pose_prototypes,
    "track_avg_step": np.zeros(N, dtype=np.float32),
}
```

**Step 6: Pass `association_data` to cost matrix call**

Change:
```python
cost, _ = assigner.compute_cost_matrix(
    N, meas, kf_manager.X, shapes, kf_manager, last_shape_info
)
```
To:
```python
cost, _ = assigner.compute_cost_matrix(
    N, meas, kf_manager.X, shapes, kf_manager, last_shape_info,
    association_data=_association_data,
)
```

**Step 7: Update `track_pose_prototypes` after matches**

After the `for r, c in zip(matched_r, matched_c): kf_manager.correct(r, m)` block, add:

```python
# Keep per-track pose prototypes current for next frame's association.
for r, c in zip(matched_r, matched_c):
    proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
    if proto is not None:
        track_pose_prototypes[r] = np.asarray(proto, dtype=np.float32).copy()
```

Also initialize prototypes in `free_dets` (first-frame bootstrap), after setting `track_states[r] = "active"`:
```python
track_pose_prototypes[r] = (
    np.asarray(_det_pose_kpts[c], dtype=np.float32).copy()
    if c < len(_det_pose_kpts) and _det_pose_kpts[c] is not None
    else None
)
```

**Step 8: Close pose cache at end of `_run_tracking_loop`**

In the `finally:` block (or after the loop), add:
```python
if _pose_cache is not None:
    try:
        _pose_cache.close()
    except Exception:
        pass
```

**Step 9: Run tests**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -x -q --tb=short --ignore=tests/test_pose_inference_sleap_service.py
```
Expected: all pass

**Step 10: Commit**

```bash
git add src/multi_tracker/core/tracking/optimizer.py
git commit -m "feat(optimizer): feed real pose keypoints into _run_tracking_loop for true advanced-cost-matrix optimization"
```

---

## Task 5: Add Pose Support to `TrackingPreviewWorker.run()`

**Files:**
- Modify: `src/multi_tracker/core/tracking/optimizer.py` (`TrackingPreviewWorker` class)

**Step 1: Add pose setup after `det_filter = DetectionFilter(self.params)` line**

```python
(
    _pose_cache,
    _pose_anterior,
    _pose_posterior,
    _pose_ignore,
    _pose_enabled,
) = _pf_load_pose_context(self.params)
_pose_min_conf = float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
_pose_kpt_map: dict = {}
_pose_kpt_map_frame = None
track_pose_prototypes = [None] * N
```

**Step 2: Update `cache.get_frame` unpacking**

```python
raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb, raw_det_ids, *_ = (
    cache.get_frame(f_idx)
)
```

**Step 3: Update `filter_raw_detections` call**

```python
filtered = det_filter.filter_raw_detections(
    raw_meas, raw_sizes, raw_shapes, raw_confs, raw_obb,
    roi_mask=_roi_mask,
    detection_ids=raw_det_ids,
)
meas, _, shapes, _confs, _obb_out, detection_ids, *_ = filtered
```

**Step 4: Compute pose features and build `association_data`**

After `kf_manager.predict()`, before `if meas:`:

```python
_det_pose_kpts = [None] * len(meas)
_det_pose_vis = np.zeros(len(meas), dtype=np.float32)
if _pose_enabled and meas and detection_ids:
    if _pose_kpt_map_frame != f_idx:
        _pose_kpt_map = _pf_build_keypoint_map(_pose_cache, f_idx)
        _pose_kpt_map_frame = f_idx
    _det_pose_kpts, _det_pose_vis = _pf_compute_det_features(
        [int(d) for d in detection_ids],
        _pose_kpt_map,
        _pose_anterior,
        _pose_posterior,
        _pose_ignore,
        _pose_min_conf,
    )
_association_data = {
    "detection_pose_keypoints": _det_pose_kpts,
    "detection_pose_visibility": _det_pose_vis,
    "track_pose_prototypes": track_pose_prototypes,
    "track_avg_step": np.zeros(N, dtype=np.float32),
}
```

**Step 5: Pass `association_data` to cost matrix**

```python
cost, _ = assigner.compute_cost_matrix(
    N, meas, kf_manager.X, shapes, kf_manager, last_shape_info,
    association_data=_association_data,
)
```

**Step 6: Update `track_pose_prototypes` after matches and in free_dets**

Same pattern as Task 4 Step 7.

**Step 7: Close pose cache in `finally:`**

```python
if _pose_cache is not None:
    try:
        _pose_cache.close()
    except Exception:
        pass
```

**Step 8: Run tests**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -x -q --tb=short --ignore=tests/test_pose_inference_sleap_service.py
```

**Step 9: Commit**

```bash
git add src/multi_tracker/core/tracking/optimizer.py
git commit -m "feat(preview): feed real pose keypoints into TrackingPreviewWorker for optimizer-consistent visualization"
```

---

## Task 6: Integration Test — Advanced Path Parity

**Files:**
- Modify: `tests/test_track_assigner.py`

Add tests verifying the `_advanced_association_enabled` guard works correctly:

```python
def test_advanced_association_disabled_when_all_keypoints_none():
    """All-None keypoint list must NOT trigger the advanced cost matrix path."""
    from multi_tracker.core.assigners.hungarian import TrackAssigner
    params = {"MAX_DISTANCE_THRESHOLD": 200.0, "W_POSITION": 1.0,
              "W_ORIENTATION": 0.1, "W_AREA": 0.0, "W_ASPECT": 0.0,
              "USE_MAHALANOBIS": False, "ENABLE_SPATIAL_OPTIMIZATION": False,
              "MAX_TARGETS": 2}
    assigner = TrackAssigner(params)
    data_all_none = {
        "detection_pose_keypoints": [None, None],
        "track_pose_prototypes": [None, None],
    }
    assert not assigner._advanced_association_enabled(data_all_none)


def test_advanced_association_enabled_when_keypoint_present():
    """Non-None keypoint must trigger the advanced cost matrix path."""
    import numpy as np
    from multi_tracker.core.assigners.hungarian import TrackAssigner
    params = {"MAX_DISTANCE_THRESHOLD": 200.0, "W_POSITION": 1.0,
              "W_ORIENTATION": 0.1, "W_AREA": 0.0, "W_ASPECT": 0.0,
              "USE_MAHALANOBIS": False, "ENABLE_SPATIAL_OPTIMIZATION": False,
              "MAX_TARGETS": 2}
    assigner = TrackAssigner(params)
    kpt = np.zeros((4, 3), dtype=np.float32)
    data_with_kpt = {
        "detection_pose_keypoints": [kpt, None],
        "track_pose_prototypes": [None, None],
    }
    assert assigner._advanced_association_enabled(data_with_kpt)
```

**Step 1: Add the two tests above to `tests/test_track_assigner.py`**

**Step 2: Run**

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_track_assigner.py -v
```
Expected: all pass (including the two new ones)

**Step 3: Commit**

```bash
git add tests/test_track_assigner.py
git commit -m "test(assigner): verify advanced association path guards with real vs all-None pose data"
```

---

## Task 7: Full Suite Validation

```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/ -q --tb=short --ignore=tests/test_pose_inference_sleap_service.py
```

Expected: all tests pass. Verify the count matches or exceeds the pre-change count.

---

## Summary of Changes

| Component | Before | After |
|-----------|--------|-------|
| `_advanced_association_enabled` | True whenever key exists (even all-None list) | True only when ≥1 non-None keypoint/prototype |
| `TrackingWorker` | Pose methods as `self.*` instance methods | Delegates to `pose_features` module |
| `TrackingOptimizer._run_tracking_loop` | No pose data, always uses simple Numba matrix | Loads pose cache, builds `association_data`, uses advanced matrix when pose present |
| `TrackingPreviewWorker.run` | No pose data, always uses simple matrix | Same as optimizer above |
| `pose_features.py` | Did not exist | Shared, pure-function module; imported by all three |
