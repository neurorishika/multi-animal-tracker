# True Canonicalization Plan

**Date:** 2026-03-26  
**Status:** Proposed — ready for phased implementation

---

## Goal

Unify all per-detection crop extraction into a single **canonical crop** that:

1. Is **affine-warped** so the animal's major axis is horizontal.
2. Has **head-tail orientation applied as a rotation** (not a flip) so the
   head always faces **right** in the output image — preserving chirality.
3. Applies **individual-analysis settings**: configurable padding, background
   colour, and foreign-OBB suppression — once, at extraction time.
4. Serves **every downstream model** (head-tail classifier, pose estimator,
   CNN identity, dataset export) from a single crop.
5. Carries a stored **affine matrix** `M_canonical` that can be inverted to
   map any prediction (keypoints, bounding-box corners) back to original
   frame coordinates.
6. Works in both **batch/precompute mode** (offline, multi-frame) and
   **real-time / preview mode** (frame-by-frame, no cache).

---

## Definitions

| Term | Meaning |
|---|---|
| **OBB** | Oriented bounding box from YOLO, 4 corners in frame coordinates |
| **M_align** | 2 × 3 affine matrix: frame → rotation-normalised (major axis horizontal, centroid centred) |
| **M_orient** | 2 × 3 rotation matrix applied after head-tail prediction (0 °/ 90 °/ 180 °/ 270 °) |
| **M_canonical** | Composite affine `M_orient ∘ M_align`: frame → final head-right canonical space |
| **M_inverse** | Pseudo-inverse of `M_canonical`: canonical → frame |
| **Canonical crop** | The image produced by `cv2.warpAffine(frame, M_canonical, (W, H))` |

---

## What the Current Code Does

There are **three independent crop-extraction paths**, each with its own
geometry, settings, and call site:

### 1. Head-tail canonical crop (engine.py — Phase 1)

- `_canonicalize_obb_for_headtail(frame, corners)`
- Affine warp, major-axis horizontal, fixed 128 × 64, margin 1.3×.
- No padding, no foreign-OBB suppression, no configurable background.
- Fed to `_predict_headtail_results()` → left/right/up/down/unknown label.
- Runs during **detection** (Phase 1 or live per-frame).

### 2. Precompute AABB crop (precompute.py + pose_pipeline.py)

- `extract_one_crop(frame, corners, ...)` → axis-aligned bounding box.
- Expands OBB corners outward by `padding_fraction`, clips to frame.
- Foreign OBB suppression via `apply_foreign_obb_mask`.
- Variable output size (depends on animal size and padding).
- Letterboxed to square for pose models; raw for CNN identity.
- Runs during **unified precompute** (between Phase 1 and Phase 2).

### 3. Individual dataset crop (analysis.py — Phase 2)

- `_extract_obb_masked_crop(frame, corners, ...)`
- AABB of expanded corners, but also applies OBB polygon mask
  (background outside the expanded OBB, not just outside the AABB).
- Foreign OBB suppression.
- Saved to disk with `canonicalization` metadata block.
- Runs during **tracking loop** (Phase 2), needs track IDs.

### Problems

| Issue | Impact |
|---|---|
| Crops are extracted **3 separate times** from the same frame data | Wasted CPU cycles (affine warp / copy / mask per extraction) |
| Head-tail crop has **no foreign suppression** | Other animals bleed into the classification crop |
| Precompute crops are **axis-aligned** (not rotation-normalised) | Pose and CNN models must learn rotational equivariance |
| Dataset crops use a **different geometry** (OBB polygon mask) from precompute crops | Downstream tools (DataSieve, PoseKit, ClassKit) operate on a different crop style than what models were trained on |
| Orientation normalisation happens **inconsistently** | Head-tail resolves to heading angle stored as a scalar; individual dataset stores metadata but doesn't rotate the crop; pose model never sees a rotation-normalised input |

---

## Proposed Architecture

### The Canonical Crop

Every detection produces **one** canonical crop with the following properties:

```
┌───────────────────────────────────┐
│                                   │
│   HEAD →                     ←    │  Fixed size: W_CANON × H_CANON
│   (right)    Animal body          │  (default: 256 × 128)
│                                   │
│   Background: config colour       │  Foreign OBBs masked
│   Padding: config fraction        │
└───────────────────────────────────┘
```

- **Major axis horizontal, head facing right** (after head-tail rotation).
- **Aspect ratio preserved** within the fixed canvas via letterbox-style
  scaling: the expanded OBB region (with padding) is uniformly scaled so
  the longest dimension fits, and the shorter dimension is centred with
  background fill.
- **Foreign OBB regions filled** with background colour (transformed into
  canonical space via `M_align`).
- Configurable: `CANONICAL_CROP_WIDTH`, `CANONICAL_CROP_HEIGHT`,
  `INDIVIDUAL_CROP_PADDING`, `INDIVIDUAL_BACKGROUND_COLOR`,
  `SUPPRESS_FOREIGN_OBB_REGIONS`.

### The Affine Chain

For a detection with OBB corners $c_0 \ldots c_3$:

1. **Compute OBB geometry:**
   - Centroid: $(c_x, c_y) = \operatorname{mean}(c_i)$
   - Major axis vector: longer edge → angle $\theta$
   - Major / minor lengths: $a$, $b$
   - Expanded dimensions: $a' = a \cdot (1 + 2 \cdot \text{padding})$,
     $b' = b \cdot (1 + 2 \cdot \text{padding})$

2. **Build M_align** (frame → rotation-normalised, centred, scaled):
   - Source: 3 corners of the expanded OBB rectangle in frame coords
   - Destination: 3 corners of the output canvas
   - Uniform scale so that the expanded OBB fills the canvas with
     letterbox padding on the short axis
   - One `cv2.getAffineTransform` call

3. **Apply M_align:**
   - `cv2.warpAffine(frame, M_align, (W, H), borderMode=BORDER_REPLICATE)`
   - Result: rotation-normalised crop with major axis horizontal

4. **Foreign OBB suppression** (in canonical space):
   - Transform each foreign OBB's 4 corners via M_align (matrix multiply)
   - `cv2.fillPoly` with background colour

5. **Head-tail classification** on the rotation-normalised crop:
   - The existing model expects 128 × 64: `cv2.resize` the canonical crop
     down for inference (or retrain on the new canonical size)
   - Prediction: `left` / `right` / `up` / `down` / `unknown`

6. **Compute M_orient** — a simple rotation about the canvas centre:

   | Head-tail label | Rotation angle | cv2 constant |
   |---|---|---|
   | `right` | 0° | — (identity) |
   | `left` | 180° | `ROTATE_180` |
   | `up` | 90° CW | `ROTATE_90_CLOCKWISE` |
   | `down` | 90° CCW | `ROTATE_90_COUNTERCLOCKWISE` |
   | `unknown` | 0° (keep axis-aligned) | — |

   **Note:** Rotation, NOT flip.  Flipping would mirror the animal,
   swapping left/right antenna.  A 180° rotation preserves chirality.

   **Aspect ratio note:** For left/right (the common case for elongated
   animals), the output stays W × H.  For up/down (rare — perpendicular
   head), the output becomes H × W.  Downstream models should handle
   both via letterboxing.  Alternatively, `up`/`down` can be treated as
   `unknown` for practical purposes.

7. **Composite M_canonical = M_orient ∘ M_align:**
   - Stored as 2 × 3 float32 (6 values per detection)
   - Inverse: `cv2.invertAffineTransform(M_canonical)` → `M_inverse`

### Inverse Mapping (Canonical → Frame)

To map predictions back to frame coordinates:

$$
\begin{bmatrix} x_{\text{frame}} \\ y_{\text{frame}} \end{bmatrix}
= M_{\text{inverse}} \cdot
\begin{bmatrix} x_{\text{canon}} \\ y_{\text{canon}} \\ 1 \end{bmatrix}
$$

For **pose keypoints**: apply $M_{\text{inverse}}$ to each $(x, y)$ pair.
Confidence values pass through unchanged.

---

## Pipeline Redesign

### Batch / Offline Mode (Two-Phase)

```
Phase 1: Detection + Canonicalization
├── Read frames sequentially
├── OBB inference (cross-frame batched)
├── Per detection: compute M_align, apply warpAffine, apply foreign mask
├── Head-tail classification (cross-frame batched on canonical crops)
├── Per detection: compute M_orient, apply rotation → final canonical crop
│   (stored only transiently — M_canonical is what gets cached)
├── Cache: corners, M_canonical (2×3), heading, directed_mask
│
│ [Video seek to start_frame]
│
Density Map (optional, unchanged)
│
│ [Video seek to start_frame]
│
Unified Precompute (replaces current AABB cropping)
├── Read frames sequentially
├── Per frame: read cached M_canonical per detection
├── Apply warpAffine(frame, M_canonical, (W, H)) → canonical crop
│   (this replaces extract_one_crop entirely)
├── Foreign OBB suppression in canonical space
│   (transform foreign corners via M_align, fillPoly)
├── Fan out canonical crops to all phases:
│   ├── Pose: letterbox to model input → predict → undo letterbox
│   │         → apply M_inverse → frame-coordinate keypoints
│   ├── CNN Identity: resize to model input → predict → class + conf
│   ├── AprilTag: optionally use AABB crop instead (configurable)
│   └── Individual Dataset: save canonical crop + metadata to disk
│       (this replaces _extract_obb_masked_crop during Phase 2)
│
│ [Video seek to start_frame]
│
Phase 2: Tracking + Visualization
├── Read frames sequentially
├── Cached detections + cached heading/pose → assignment → KF update
├── Individual dataset: crops already saved by precompute
│   (only need to write track/trajectory ID mapping, not re-extract)
├── Visualization, video output
```

**Key changes vs current:**

| Aspect | Current | Proposed |
|---|---|---|
| Crop extraction in Phase 1 | 128 × 64 affine for head-tail only | Full canonical crop (configurable size) for head-tail + cache M |
| Crop extraction in precompute | AABB via `extract_one_crop` | `warpAffine(frame, M_canonical)` — rotation-normalised |
| Crop extraction in Phase 2 | `_extract_obb_masked_crop` per detection | None — crops saved during precompute |
| Foreign suppression | AABB-space polygon fill | Canonical-space polygon fill (foreign corners transformed via M) |
| Pose keypoint mapping | crop_offset (x, y) addition | `M_inverse` matrix multiplication |
| Head-tail as rotation | Not applied (stored as angle scalar) | Applied to image via `cv2.rotate` |
| Dataset crops vs model crops | Different extraction methods | Identical — same canonical crop |

### Real-Time / Preview Mode (Single-Frame)

```
Per frame:
├── OBB detection (single frame)
├── Per detection: compute M_align → warpAffine → canonical crop
├── Foreign OBB suppression in canonical space
├── Head-tail classification (on canonical crops, single-frame batch)
├── M_orient rotation → final canonical crop, M_canonical
├── Feed to enabled real-time models:
│   ├── Pose → M_inverse → frame keypoints
│   ├── CNN identity → class label
│   └── (no dataset save in preview mode)
├── Tracking: assignment + KF update (as today)
```

No cache needed.  The canonical crop is computed once and passed to
all consumers within the same frame iteration.  This is a strict
superset of the current single-frame path.

### Parallel Processing Strategy

The canonical crop architecture enables three levels of parallelism:

#### 1. Cross-frame batching (existing, improved)

Collect canonical crops from N frames → single GPU call per model.
Already implemented for head-tail; now extends to pose and CNN identity
in precompute.

Batch dimensions for 16 frames × ~20 detections:

| Model | Batch size | Current | Proposed |
|---|---|---|---|
| Head-tail | ~320 crops | 1 GPU call (already batched) | Same |
| Pose | ~320 crops | 1 GPU call per flush | Same, but on canonical crops |
| CNN Identity | ~320 crops | 1 GPU call per flush | Same, but on canonical crops |

#### 2. Inter-model pipeline parallelism (new opportunity)

Once canonical crops are extracted, pose and CNN identity have **no data
dependency** on each other.  They can run concurrently:

```
Thread/Stream 1: Pose model (GPU)      ─────────►
Thread/Stream 2: CNN Identity (GPU)    ─────────►
Main thread:     Crop extraction (CPU) ─────────► (next batch)
```

This requires separate CUDA streams per model.  Not a first-phase
priority, but the architecture enables it cleanly.

#### 3. CPU/GPU overlap (existing, improved)

Crop extraction (warpAffine) is CPU-bound.  While the GPU processes
batch N, the CPU can extract crops for batch N+1.  The precompute
orchestrator already reads ahead in the frame loop; with canonical
crops this pattern continues naturally.

---

## Cache Schema Changes

### Detection Cache (version 2.1 → 2.2)

New per-frame arrays:

| Key suffix | dtype | Shape | Description |
|---|---|---|---|
| `_canonical_affine` | float32 | `(N, 2, 3)` | `M_canonical` per detection |
| `_canonical_crop_size` | int32 | `(2,)` | `(W, H)` of canonical crop (frame-level, same for all detections) |

**Backward compatibility:** Version 2.1 caches lacking
`_canonical_affine` trigger re-detection (same as any schema mismatch).
The `canonical_crop_size` is frame-level metadata (all detections in a
frame share the same canvas size).

**Storage overhead:** 6 floats × N detections × 4 bytes = 24N bytes/frame.
For 20 detections × 50,000 frames = 24 MB.  Negligible.

### Individual Properties Cache (unchanged schema)

Pose keypoints continue to be stored in **frame coordinates** (after
`M_inverse` mapping).  This ensures backward compatibility with
downstream consumers that expect frame-space keypoints.

### Individual Dataset Metadata

The `canonicalization` block in each crop's metadata gains:

```json
{
  "canonicalization": {
    "M_canonical": [[m00, m01, m02], [m10, m11, m12]],
    "M_inverse": [[m00, m01, m02], [m10, m11, m12]],
    "crop_size_px": [256, 128],
    "heading_angle_rad": 1.57,
    "headtail_label": "right",
    "headtail_confidence": 0.93,
    "directed": true,
    "orientation_source": "head_tail_model"
  }
}
```

This replaces the current `center_px`, `size_px`, `angle_rad`,
`major_axis_theta_rad`, `minor_axis_theta_rad` fields.  The affine
matrices are strictly more informative and can reconstruct all prior
fields.

---

## New Module: `canonical_crop.py`

A single module owns all crop geometry.  Proposed location:
`src/multi_tracker/core/tracking/canonical_crop.py`

### Public API

```python
@dataclass
class CanonicalCropResult:
    """Result of canonical crop extraction for one detection."""
    crop: np.ndarray              # (H, W, 3) uint8 BGR — final head-right crop
    M_canonical: np.ndarray       # (2, 3) float32 — frame → canonical
    M_inverse: np.ndarray         # (2, 3) float32 — canonical → frame
    aligned_crop: np.ndarray      # (H, W, 3) uint8 BGR — pre-rotation (for head-tail)
    headtail_label: str           # "left"/"right"/"up"/"down"/"unknown"
    headtail_confidence: float    # classifier confidence
    heading_angle: float          # resolved heading in frame coords (radians)
    directed: bool                # True if heading is reliable

def compute_alignment_affine(
    corners: np.ndarray,
    crop_width: int,
    crop_height: int,
    padding_fraction: float,
) -> Tuple[np.ndarray, float]:
    """Compute M_align from OBB corners.

    Returns (M_align, major_axis_theta).
    Letterbox-scales the expanded OBB into (crop_width, crop_height).
    """

def extract_canonical_crop(
    frame: np.ndarray,
    corners: np.ndarray,
    M_align: np.ndarray,
    crop_size: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    foreign_corners: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Apply M_align to extract a rotation-normalised crop.

    Applies foreign OBB suppression in canonical space.
    """

def apply_headtail_rotation(
    crop: np.ndarray,
    M_align: np.ndarray,
    label: str,
    major_axis_theta: float,
    crop_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Rotate crop so head faces right.

    Returns (rotated_crop, M_canonical, M_inverse, heading_angle_frame).
    For left → 180° rotation.
    For up   → 90° CW rotation.
    For down → 90° CCW rotation.
    """

def invert_keypoints(
    keypoints: np.ndarray,
    M_inverse: np.ndarray,
) -> np.ndarray:
    """Map (K, 2) or (K, 3) keypoints from canonical to frame coordinates.

    Confidence column (if present) passes through unchanged.
    """

def extract_and_classify_batch(
    frames: List[np.ndarray],
    per_frame_corners: List[List[np.ndarray]],
    headtail_model,
    headtail_backend: str,
    crop_width: int,
    crop_height: int,
    padding_fraction: float,
    bg_color: Tuple[int, int, int],
    suppress_foreign: bool,
    headtail_conf_threshold: float,
    per_frame_all_corners: Optional[List[List[np.ndarray]]] = None,
) -> List[List[CanonicalCropResult]]:
    """Full canonical pipeline for a batch of frames.

    1. Compute M_align per detection
    2. Extract aligned crops (CPU, parallelisable)
    3. Head-tail classify all crops in one GPU call
    4. Apply orientation rotation → M_canonical
    5. Return per-frame lists of CanonicalCropResult
    """
```

### Key Implementation Details

**Letterbox scaling within the affine:**  Rather than warp to one size
and then letterbox, we bake the letterbox into `M_align` itself.  The
uniform scale factor is:

$$s = \min\!\left(\frac{W_{\text{out}}}{a'}, \frac{H_{\text{out}}}{b'}\right)$$

where $a' = a \cdot (1 + 2p)$ and $b' = b \cdot (1 + 2p)$ are the
padded major/minor lengths.  The translation centres the scaled animal
on the canvas.  This gives us a single `warpAffine` that does rotation +
scale + translate + letterbox in one pass.

**Foreign OBB suppression in canonical space:**  Transform each foreign
OBB's corners via `M_align`:

```python
foreign_canon = (M_align[:, :2] @ foreign_corners.T + M_align[:, 2:3]).T
cv2.fillPoly(crop, [foreign_canon.astype(np.int32)], bg_color)
```

This correctly handles the rotation — foreign regions are masked in the
correct location even though the image is rotated.

**90° rotations for up/down:**  The output canvas flips from (W, H) to
(H, W).  The composite `M_canonical` accounts for this by including the
90° rotation matrix.  Downstream consumers that expect (W, H) should
letterbox the (H, W) result — or, pragmatically, up/down predictions
can be treated as `unknown` (no rotation) since they're rare for
elongated animals.

---

## Migration: What Changes in Each Module

### engine.py (YOLOOBBDetector)

| Method | Change |
|---|---|
| `_canonicalize_obb_for_headtail` | Replace with call to `compute_alignment_affine` + `extract_canonical_crop` (resize to 128 × 64 for inference) |
| `_compute_headtail_hints` | Refactor: use `canonical_crop.extract_and_classify_batch` for the full pipeline.  Returns heading hints + stores `M_canonical` per detection. |
| `_compute_headtail_hints_cross_frame` | Replace with `extract_and_classify_batch`. |
| `detect_objects` (single-frame) | Now also computes and returns `M_canonical` per detection.  New return element in the raw tuple. |
| `detect_objects_batched` | Post-processing stores `M_canonical` arrays per frame. |

### detection_cache.py

| Change | Detail |
|---|---|
| Schema version | `"2.1"` → `"2.2"` |
| `add_frame` | New parameter: `canonical_affines` (N, 2, 3) |
| `get_frame` | Returns additional `canonical_affines` array |
| Backward compat | Missing `_canonical_affine` key → returns `None`; caller falls back to re-computing from corners |

### precompute.py (UnifiedPrecompute)

| Change | Detail |
|---|---|
| Crop extraction | Replace `extract_one_crop` call with `extract_canonical_crop(frame, corners, M_canonical, ...)` using cached `M_canonical` from detection cache |
| `CropConfig` | Add `canonical_crop_width`, `canonical_crop_height` fields |
| Phase dispatch | Phases receive canonical crops (rotation-normalised, head-right) instead of AABB crops |
| Crop offsets | **Removed.** Phases receive `M_inverse` instead of `(x0, y0)` integer offsets |

### pose_pipeline.py

| Method | Change |
|---|---|
| `extract_one_crop` | Still available as legacy fallback; not called from precompute |
| `letterbox_crop` | Unchanged — still used for pose model input |
| `invert_letterbox_keypoints` | Unchanged — still used for undo letterbox |
| Keypoint back-mapping | After undo-letterbox: `canonical_crop.invert_keypoints(kpts, M_inverse)` replaces the current `kpts[:, 0] += x0; kpts[:, 1] += y0` offset addition |
| `filter_keypoints_by_foreign_obbs` | Unchanged — operates on frame-global keypoints |

### analysis.py (IndividualDatasetGenerator)

| Method | Change |
|---|---|
| `_extract_obb_masked_crop` | **Deprecated.** Replaced by reading the canonical crop directly. |
| `process_frame` | Receives canonical crops (or `M_canonical` + frame) instead of extracting its own crops.  If precompute already saved crops to disk, just writes track-ID mapping. |
| Metadata block | Uses new `canonicalization` schema with `M_canonical` / `M_inverse` |

### cnn_identity.py (CNNPrecomputePhase)

| Change | Detail |
|---|---|
| Crop input | Receives canonical crops (head-right, rotation-normalised) instead of AABB crops.  The `resize → normalize → forward` pipeline is unchanged. |
| Training implication | New ClassKit models trained on canonical crops generalise better (no rotational variance in training data). Existing models work but may need fine-tuning on the new crop style. |

### AprilTag Phase

| Change | Detail |
|---|---|
| Default | **Keep AABB crops** for AprilTag detection (configurable).  Tag detection relies on precise edge geometry; affine interpolation may degrade marginal tags. |
| Optional | Add config flag `APRILTAG_USE_CANONICAL_CROP` (default `false`) for users who want to try it. |

---

## Real-Time Tracking Compatibility

### Current real-time path (preview mode)

```python
# Per frame:
detect_objects(frame, frame_count)  →  (meas, sizes, shapes, ...)
# Head-tail runs inside detect_objects via _compute_headtail_hints
# No precompute, no pose, no CNN identity
```

### Proposed real-time path

```python
# Per frame:
detect_objects(frame, frame_count, return_canonical=True)
    →  (meas, sizes, shapes, ..., canonical_results)
# canonical_results: List[CanonicalCropResult] per detection

# If real-time pose is enabled:
for cr in canonical_results:
    letterboxed, transform = letterbox_crop(cr.crop, pose_input_size)
    kpts_canon = pose_model.predict(letterboxed)
    kpts_canon = invert_letterbox_keypoints(kpts_canon, transform)
    kpts_frame = invert_keypoints(kpts_canon, cr.M_inverse)

# If real-time CNN identity is enabled:
crops = [cr.crop for cr in canonical_results]
labels = cnn_model.predict_batch(crops)
```

**No architectural change needed** — the canonical crop is computed
as part of detection, and downstream models consume it immediately.
The only difference from offline mode is that crops are not cached
to disk.

### Future: Real-Time Classification in Tracking Loop

The canonical crop architecture enables real-time CNN identity during
Phase 2 tracking (not just precompute):

```python
# In tracking loop, after detection:
for det_idx, cr in enumerate(canonical_results):
    label, conf = cnn_classifier.predict_single(cr.crop)
    # Use label in Hungarian assignment cost matrix
```

This is currently not possible because precompute crops are not
available during Phase 2.  With canonical crops computed at detection
time, the crops are immediately available.

---

## Configuration

New parameters (in `ADVANCED_CONFIG` or top-level):

```json
{
  "canonical_crop_width": 256,
  "canonical_crop_height": 128,
  "canonical_headtail_inference_width": 128,
  "canonical_headtail_inference_height": 64,
  "canonical_treat_updown_as_unknown": true,
  "apriltag_use_canonical_crop": false
}
```

Existing parameters that now apply to canonical crops (semantics
unchanged):

```json
{
  "individual_crop_padding": 0.5,
  "individual_background_color": [233, 232, 231],
  "suppress_foreign_obb_regions": true
}
```

---

## Implementation Phases

### Phase A: Core Module + Detection Integration

**Files:** New `canonical_crop.py`, modified `engine.py`, modified
`detection_cache.py`.

1. Implement `canonical_crop.py` with full API.
2. Refactor `_canonicalize_obb_for_headtail` → use `compute_alignment_affine` + `extract_canonical_crop` + resize for inference.
3. Refactor `_compute_headtail_hints_cross_frame` → use `extract_and_classify_batch`.
4. Store `M_canonical` in detection cache (version 2.2).
5. Single-frame `detect_objects` returns `CanonicalCropResult` list.
6. All existing head-tail tests pass.

### Phase B: Precompute Migration

**Files:** Modified `precompute.py`, modified `pose_pipeline.py`.

1. `UnifiedPrecompute.run()` uses `M_canonical` from cache + `extract_canonical_crop` instead of `extract_one_crop`.
2. Pose phase receives canonical crops; keypoint back-mapping uses `M_inverse` instead of offset addition.
3. CNN identity phase receives canonical crops.
4. AprilTag phase: keep AABB by default, add canonical option.
5. All precompute + pose tests pass.

### Phase C: Individual Dataset Migration

**Files:** Modified `analysis.py`, modified `worker.py`.

1. Individual dataset crops saved during precompute (not Phase 2).
2. Phase 2 `process_frame` writes track/trajectory ID mappings
   referencing precompute-saved crops (or re-extracts via `M_canonical`
   from frame if precompute didn't save images).
3. Metadata block updated to new schema.
4. DataSieve / PoseKit / ClassKit consume head-right canonical crops
   without needing their own orientation logic.

### Phase D: Real-Time Path

**Files:** Modified `worker.py` (preview mode), potentially new
real-time pose/CNN hooks.

1. Preview mode computes canonical crops at detection time.
2. Optional real-time pose: canonical crop → pose model → `M_inverse`.
3. Optional real-time CNN identity: canonical crop → classifier.
4. All preview-mode tests pass.

---

## Testing Strategy

| Test | Validates |
|---|---|
| Unit: `M_canonical` round-trip | `invert_keypoints(M_inverse, point) ≈ original` for random OBBs |
| Unit: 180° rotation correctness | Crop rotated 180° has same pixels as re-extraction with θ + π |
| Unit: foreign OBB suppression in canonical space | Transformed foreign polygon covers correct pixels |
| Integration: detection cache v2.2 | Write/read `M_canonical` arrays; backward compat with v2.1 |
| Integration: precompute with canonical crops | Pose keypoints in frame coords match baseline (within 2 px) |
| Integration: individual dataset metadata | `M_canonical` in metadata reconstructs frame-coordinate crops |
| End-to-end: full pipeline | Tracking results (trajectories, IDs) unchanged within tolerance |
| Benchmark: crop extraction speed | Canonical warpAffine vs current AABB extraction (expect similar) |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Existing head-tail model degrades on larger canonical crop (resized down to 128 × 64 for inference) | Low — bilinear downscale from 256 × 128 is equivalent to current warp at 128 × 64 | Monitor accuracy; retrain if needed |
| Pose model accuracy changes on rotation-normalised input vs AABB | Medium — model was trained on AABB crops | Retrain pose model on canonical crops (Phase B follow-up) |
| CNN identity accuracy changes | Low — rotation normalisation should help, not hurt | Validate on held-out set; retrain via ClassKit if needed |
| AprilTag detection degrades on canonical crops | Medium — affine interpolation blurs edges | Keep AABB as default for AprilTags |
| 90° rotation (up/down head-tail) produces portrait aspect ratio | Low — rare for elongated animals | `canonical_treat_updown_as_unknown: true` by default |
| Detection cache v2.1 → v2.2 migration | Low — schema mismatch triggers re-detection | Ensure clear version check and user messaging |
