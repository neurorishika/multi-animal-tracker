# MAT-afterhours Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the MAT-afterhours standalone application for interactive post-hoc identity swap proofreading, along with upstream confidence density map computation integrated into the MAT tracking pipeline.

**Architecture:** Three coordinated additions: (A) a confidence density map computed after the batched detection phase and stored as a sidecar artifact, (B) density-aware backward-pass assignment that prefers safe fragments over risky identity maintenance in ambiguous zones, and (C) the MAT-afterhours Qt application — a fifth standalone app following the ClassKit bootstrap pattern — that scores and presents suspected swap events as a ranked queue for user review, with frame-picker + before/after thumbnail identity assignment and atomic CSV correction.

**Tech Stack:** Python 3.11+, PySide6, pandas, numpy, scipy (ndimage for 3D connected components), OpenCV (frame reading), existing `multi_tracker.core.*` and `multi_tracker.data.*` modules.

**Design doc:** `docs/plans/2026-03-09-mat-afterhours-design.md`

---

## Part A — Confidence Density Map (MAT pipeline addition)

---

### Task 1: Confidence density core module

**Files:**
- Create: `src/multi_tracker/afterhours/__init__.py`
- Create: `src/multi_tracker/afterhours/core/__init__.py`
- Create: `src/multi_tracker/afterhours/core/confidence_density.py`
- Create: `tests/test_confidence_density.py`

**Context:**
The confidence density map accumulates `(1 - confidence)` Gaussians per detection over a 2D float32 grid, smooths temporally, binarizes, then finds 3D connected components (x, y, t) to define "low-confidence high-density" regions. Each region gets a label (`region-N`) and a bounding box.

The Gaussian sigma is proportional to `bbox_diagonal` (body-size-scaled). Detections with low confidence contribute strongly; empty space contributes zero.

---

**Step 1: Write failing tests**

```python
# tests/test_confidence_density.py
import numpy as np
import pytest
from multi_tracker.afterhours.core.confidence_density import (
    ConfidenceDensityMap,
    DensityRegion,
    accumulate_frame,
    smooth_and_binarize,
    find_regions,
    tag_detections,
)


def _make_detections(n, cx, cy, conf, bbox_diag=30.0):
    """Helper: n detections at (cx, cy) with given confidence."""
    meas = np.array([[cx, cy, 0.0]] * n, dtype=np.float32)
    confidences = np.array([conf] * n, dtype=np.float32)
    sizes = np.array([bbox_diag**2] * n, dtype=np.float32)
    return meas, confidences, sizes


def test_accumulate_frame_empty():
    """Empty frame contributes zero density."""
    grid = np.zeros((64, 64), dtype=np.float32)
    meas = np.zeros((0, 3), dtype=np.float32)
    confs = np.zeros(0, dtype=np.float32)
    sizes = np.zeros(0, dtype=np.float32)
    result = accumulate_frame(grid, meas, confs, sizes, sigma_scale=1.0)
    assert result.max() == 0.0


def test_accumulate_frame_low_confidence_is_strong():
    """Low confidence detection gives higher density than high confidence."""
    h, w = 64, 64
    cx, cy = 32, 32
    meas_hi = np.array([[cx, cy, 0.0]], dtype=np.float32)
    meas_lo = np.array([[cx, cy, 0.0]], dtype=np.float32)
    confs_hi = np.array([0.95], dtype=np.float32)
    confs_lo = np.array([0.10], dtype=np.float32)
    sizes = np.array([900.0], dtype=np.float32)  # bbox_diag=30

    grid_hi = accumulate_frame(np.zeros((h, w), dtype=np.float32), meas_hi, confs_hi, sizes, sigma_scale=1.0)
    grid_lo = accumulate_frame(np.zeros((h, w), dtype=np.float32), meas_lo, confs_lo, sizes, sigma_scale=1.0)

    assert grid_lo[cy, cx] > grid_hi[cy, cx]


def test_accumulate_frame_high_confidence_near_zero():
    """High confidence detection contributes near-zero density."""
    h, w = 64, 64
    grid = np.zeros((h, w), dtype=np.float32)
    meas = np.array([[32, 32, 0.0]], dtype=np.float32)
    confs = np.array([0.99], dtype=np.float32)
    sizes = np.array([900.0], dtype=np.float32)
    result = accumulate_frame(grid, meas, confs, sizes, sigma_scale=1.0)
    assert result.max() < 0.05


def test_smooth_and_binarize_returns_binary():
    """Output of smooth_and_binarize is strictly 0 or 1."""
    frames = np.random.rand(10, 32, 32).astype(np.float32)
    binary = smooth_and_binarize(frames, temporal_sigma=1.0, threshold=0.3)
    assert binary.dtype == np.uint8
    assert set(np.unique(binary)).issubset({0, 1})


def test_smooth_and_binarize_shape():
    """Output shape matches input (T, H, W)."""
    frames = np.zeros((20, 48, 64), dtype=np.float32)
    binary = smooth_and_binarize(frames, temporal_sigma=2.0, threshold=0.3)
    assert binary.shape == (20, 48, 64)


def test_find_regions_empty():
    """All-zero binary volume yields no regions."""
    binary = np.zeros((10, 32, 32), dtype=np.uint8)
    regions = find_regions(binary, frame_h=32, frame_w=32)
    assert regions == []


def test_find_regions_single_blob():
    """A single hot blob produces one DensityRegion."""
    binary = np.zeros((10, 64, 64), dtype=np.uint8)
    binary[2:5, 10:20, 10:20] = 1
    regions = find_regions(binary, frame_h=64, frame_w=64)
    assert len(regions) == 1
    r = regions[0]
    assert r.label == "region-1"
    assert r.frame_start <= 2
    assert r.frame_end >= 4
    assert isinstance(r.pixel_bbox, tuple) and len(r.pixel_bbox) == 4


def test_find_regions_two_blobs():
    """Two separated blobs yield two DensityRegions."""
    binary = np.zeros((20, 64, 64), dtype=np.uint8)
    binary[0:3, 0:10, 0:10] = 1
    binary[15:18, 50:64, 50:64] = 1
    regions = find_regions(binary, frame_h=64, frame_w=64)
    assert len(regions) == 2


def test_tag_detections_labels_correctly():
    """Detections inside a region get the region label, others get open_field."""
    regions = [
        DensityRegion(
            label="region-1",
            frame_start=5,
            frame_end=15,
            pixel_bbox=(10, 10, 50, 50),  # x1, y1, x2, y2
        )
    ]
    # Inside region
    inside = {"frame": 10, "cx": 30.0, "cy": 30.0}
    # Outside region (wrong frame)
    outside_frame = {"frame": 20, "cx": 30.0, "cy": 30.0}
    # Outside region (wrong position)
    outside_pos = {"frame": 10, "cx": 5.0, "cy": 5.0}

    assert tag_detections([inside], regions)[0]["region_label"] == "region-1"
    assert tag_detections([outside_frame], regions)[0]["region_label"] == "open_field"
    assert tag_detections([outside_pos], regions)[0]["region_label"] == "open_field"


def test_density_region_is_boundary():
    """Regions expose a helper to check if a frame is near the boundary."""
    r = DensityRegion(
        label="region-1",
        frame_start=10,
        frame_end=20,
        pixel_bbox=(0, 0, 100, 100),
    )
    assert r.is_boundary_frame(10, margin=2)   # at start
    assert r.is_boundary_frame(20, margin=2)   # at end
    assert not r.is_boundary_frame(15, margin=2)  # interior
```

**Step 2: Run test to verify it fails**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_confidence_density.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError` or `ImportError` — module does not exist yet.

---

**Step 3: Implement the module**

Create `src/multi_tracker/afterhours/__init__.py` (empty).

Create `src/multi_tracker/afterhours/core/__init__.py` (empty).

Create `src/multi_tracker/afterhours/core/confidence_density.py`:

```python
"""
Confidence density map: identifies spatiotemporal regions where detections
are numerous but low-confidence (crowding / occlusion zones).

Accumulation formula: (1 - confidence) * G_normalised(x, y)
  - low confidence → strong signal
  - high confidence → near-zero
  - empty space → zero

Produces DensityRegion objects covering 3D connected components in (x, y, t).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label

logger = logging.getLogger(__name__)


@dataclass
class DensityRegion:
    label: str                  # 'region-1', 'region-2', ...
    frame_start: int
    frame_end: int
    pixel_bbox: tuple           # (x1, y1, x2, y2) in original pixel coords

    def contains(self, frame: int, cx: float, cy: float) -> bool:
        """True if (frame, cx, cy) falls inside this region."""
        if not (self.frame_start <= frame <= self.frame_end):
            return False
        x1, y1, x2, y2 = self.pixel_bbox
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def is_boundary_frame(self, frame: int, margin: int = 3) -> bool:
        """True if frame is within margin frames of the region boundary."""
        return (
            self.frame_start <= frame <= self.frame_start + margin
            or self.frame_end - margin <= frame <= self.frame_end
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "pixel_bbox": list(self.pixel_bbox),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DensityRegion":
        return cls(
            label=d["label"],
            frame_start=d["frame_start"],
            frame_end=d["frame_end"],
            pixel_bbox=tuple(d["pixel_bbox"]),
        )


@dataclass
class ConfidenceDensityMap:
    """Accumulated per-frame density grids and derived regions."""
    frame_grids: list[np.ndarray] = field(default_factory=list)  # list of (H, W) float32
    regions: list[DensityRegion] = field(default_factory=list)
    frame_h: int = 0
    frame_w: int = 0


def accumulate_frame(
    grid: np.ndarray,
    meas: np.ndarray,        # (N, 3) [cx, cy, theta]
    confidences: np.ndarray, # (N,) float32
    sizes: np.ndarray,       # (N,) detection areas (px²); sqrt = bbox_diagonal proxy
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """
    Add (1 - confidence) Gaussian contributions for each detection to grid.

    grid: (H, W) float32 accumulator — modified in place and returned.
    sigma = sigma_scale * sqrt(size) / 2  (half bbox diagonal)
    """
    if len(meas) == 0:
        return grid
    h, w = grid.shape
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    for i in range(len(meas)):
        cx, cy = float(meas[i, 0]), float(meas[i, 1])
        conf = float(confidences[i])
        sigma = sigma_scale * float(np.sqrt(max(sizes[i], 1.0))) / 2.0
        amplitude = max(0.0, 1.0 - conf)
        if amplitude < 1e-6:
            continue
        # Normalised Gaussian (peak = amplitude / (2π σ²), but we keep peak = amplitude)
        g = amplitude * np.exp(-((xg - cx) ** 2 + (yg - cy) ** 2) / (2.0 * sigma ** 2))
        grid += g

    return grid


def smooth_and_binarize(
    frames: np.ndarray,       # (T, H, W) float32 accumulated grids
    temporal_sigma: float,
    threshold: float,
) -> np.ndarray:
    """
    Apply temporal Gaussian smoothing then binarize.

    Returns uint8 array of shape (T, H, W) with values 0 or 1.
    """
    smoothed = gaussian_filter(frames.astype(np.float32), sigma=(temporal_sigma, 0, 0))
    # Normalise per-frame to [0, 1] before threshold to handle varying detection counts
    max_val = smoothed.max()
    if max_val > 0:
        smoothed = smoothed / max_val
    binary = (smoothed >= threshold).astype(np.uint8)
    return binary


def find_regions(
    binary: np.ndarray,  # (T, H, W) uint8
    frame_h: int,
    frame_w: int,
) -> list[DensityRegion]:
    """
    Find 3D connected components in binary volume and return DensityRegion list.
    Axes: (T, H, W) → components are blobs in time and space.
    """
    if binary.max() == 0:
        return []

    labeled, n_components = label(binary)
    regions = []

    for comp_id in range(1, n_components + 1):
        mask = labeled == comp_id
        t_coords, y_coords, x_coords = np.where(mask)

        frame_start = int(t_coords.min())
        frame_end = int(t_coords.max())
        # Pixel bbox in original image coords (H→y, W→x)
        x1, y1 = int(x_coords.min()), int(y_coords.min())
        x2, y2 = int(x_coords.max()), int(y_coords.max())

        regions.append(DensityRegion(
            label=f"region-{comp_id}",
            frame_start=frame_start,
            frame_end=frame_end,
            pixel_bbox=(x1, y1, x2, y2),
        ))

    return regions


def tag_detections(
    detections: list[dict[str, Any]],
    regions: list[DensityRegion],
) -> list[dict[str, Any]]:
    """
    Tag each detection dict with 'region_label' ('open_field' or 'region-N')
    and 'region_boundary' (bool). Dicts must have 'frame', 'cx', 'cy' keys.
    Returns the same list with tags added in place.
    """
    for det in detections:
        frame = int(det["frame"])
        cx = float(det["cx"])
        cy = float(det["cy"])
        det["region_label"] = "open_field"
        det["region_boundary"] = False
        for r in regions:
            if r.contains(frame, cx, cy):
                det["region_label"] = r.label
                det["region_boundary"] = r.is_boundary_frame(frame)
                break
    return detections


# ── Serialisation helpers ────────────────────────────────────────────────────

def save_regions(regions: list[DensityRegion], path: Path) -> None:
    """Write regions to <name>_confidence_regions.json."""
    data = {"regions": [r.to_dict() for r in regions]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_regions(path: Path) -> list[DensityRegion]:
    """Load regions from <name>_confidence_regions.json."""
    with open(path) as f:
        data = json.load(f)
    return [DensityRegion.from_dict(d) for d in data.get("regions", [])]


# ── Pipeline entry point ─────────────────────────────────────────────────────

def compute_density_map_from_cache(
    detection_cache,          # DetectionCache instance (read mode)
    frame_h: int,
    frame_w: int,
    sigma_scale: float = 1.0,
    temporal_sigma: float = 2.0,
    threshold: float = 0.3,
) -> tuple[ConfidenceDensityMap, list[np.ndarray]]:
    """
    Full pipeline: load cache → accumulate grids → smooth → binarize → find regions.

    Returns (ConfidenceDensityMap, per_frame_grids) where per_frame_grids is the
    raw accumulated float32 grids for diagnostic video rendering.
    """
    frame_range = detection_cache.get_frame_range()
    if frame_range is None:
        return ConfidenceDensityMap(frame_h=frame_h, frame_w=frame_w), []

    start, end = frame_range
    n_frames = end - start + 1
    grids = np.zeros((n_frames, frame_h, frame_w), dtype=np.float32)

    for frame_idx in range(start, end + 1):
        result = detection_cache.get_frame(frame_idx)
        if result is None:
            continue
        meas, sizes, shapes, confidences, *_ = result
        if len(meas) == 0:
            continue
        t = frame_idx - start
        accumulate_frame(grids[t], meas, confidences, sizes, sigma_scale=sigma_scale)

    binary = smooth_and_binarize(grids, temporal_sigma=temporal_sigma, threshold=threshold)
    regions = find_regions(binary, frame_h=frame_h, frame_w=frame_w)

    dm = ConfidenceDensityMap(
        frame_grids=list(grids),
        regions=regions,
        frame_h=frame_h,
        frame_w=frame_w,
    )
    return dm, list(grids)
```

**Step 4: Run tests**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_confidence_density.py -v
```

Expected: All tests pass.

**Step 5: Commit**

```bash
git add src/multi_tracker/afterhours/ tests/test_confidence_density.py
git commit -m "feat(afterhours): add confidence density map core module with tests"
```

---

### Task 2: Diagnostic video export (MAT artifact)

**Files:**
- Modify: `src/multi_tracker/afterhours/core/confidence_density.py` (add `export_diagnostic_video`)
- Create: `tests/test_confidence_density_video.py`

**Context:**
After every MAT tracking run, generate `<name>_confidence_map.mp4` with the original video frames overlaid by a semi-transparent red heatmap on low-confidence zones and animal ID annotations. This is a read-only artifact consumed by MAT-afterhours.

---

**Step 1: Write failing test**

```python
# tests/test_confidence_density_video.py
import numpy as np
import tempfile
from pathlib import Path
from multi_tracker.afterhours.core.confidence_density import export_diagnostic_video, DensityRegion


def _fake_frame_reader(n_frames, h=64, w=64):
    """Returns a callable that yields (H, W, 3) uint8 frames."""
    def reader(frame_idx):
        if frame_idx >= n_frames:
            return None
        return np.full((h, w, 3), 128, dtype=np.uint8)
    return reader, n_frames, h, w


def test_export_diagnostic_video_creates_file():
    """export_diagnostic_video writes an mp4 file."""
    reader, n_frames, h, w = _fake_frame_reader(10)
    grids = [np.zeros((h, w), dtype=np.float32) for _ in range(n_frames)]
    grids[3][10:20, 10:20] = 1.0  # hot zone in frame 3
    regions = [DensityRegion("region-1", 3, 3, (10, 10, 20, 20))]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test_confidence_map.mp4"
        export_diagnostic_video(
            frame_reader=reader,
            n_frames=n_frames,
            frame_h=h,
            frame_w=w,
            density_grids=grids,
            regions=regions,
            output_path=out,
            fps=5,
        )
        assert out.exists()
        assert out.stat().st_size > 0
```

**Step 2: Run test to verify it fails**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_confidence_density_video.py -v
```

Expected: `ImportError` on `export_diagnostic_video`.

---

**Step 3: Implement `export_diagnostic_video`**

Add to `confidence_density.py`:

```python
def export_diagnostic_video(
    frame_reader,           # callable: frame_idx -> np.ndarray (H,W,3) uint8 or None
    n_frames: int,
    frame_h: int,
    frame_w: int,
    density_grids: list[np.ndarray],  # per-frame (H, W) float32 raw grids
    regions: list[DensityRegion],
    output_path: Path,
    fps: float = 25.0,
    heatmap_alpha: float = 0.35,
) -> None:
    """
    Write diagnostic video with red heatmap overlay on low-confidence zones.
    """
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))

    # Normalise grids globally for consistent colour scale
    all_vals = np.concatenate([g.ravel() for g in density_grids]) if density_grids else np.array([0.0])
    global_max = float(all_vals.max()) if all_vals.max() > 0 else 1.0

    for frame_idx in range(n_frames):
        frame = frame_reader(frame_idx)
        if frame is None:
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        frame = frame.copy()

        # Draw heatmap overlay
        if frame_idx < len(density_grids):
            norm = (density_grids[frame_idx] / global_max).clip(0, 1)
            red_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            red_mask[:, :, 2] = (norm * 255).astype(np.uint8)  # OpenCV BGR → red=channel2
            frame = cv2.addWeighted(frame, 1 - heatmap_alpha, red_mask, heatmap_alpha, 0)

        # Annotate region labels
        for r in regions:
            if r.frame_start <= frame_idx <= r.frame_end:
                x1, y1, x2, y2 = r.pixel_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                label_text = f"{r.label} [{r.frame_start}-{r.frame_end}]"
                cv2.putText(frame, label_text, (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

        writer.write(frame)

    writer.release()
```

**Step 4: Run test**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_confidence_density_video.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/multi_tracker/afterhours/core/confidence_density.py tests/test_confidence_density_video.py
git commit -m "feat(afterhours): add diagnostic video export for confidence density map"
```

---

### Task 3: Wire density map into MAT post-processing pipeline

**Files:**
- Modify: `src/multi_tracker/gui/main_window.py` (find `_run_post_processing` or equivalent — search for where CSV is written after tracking)
- Modify: `configs/default.json` (add 3 new keys)

**Context:**
After the batched detection phase writes the `.npz` cache, call `compute_density_map_from_cache` and write `<name>_confidence_regions.json` and `<name>_confidence_map.mp4` alongside the existing output files. This runs unconditionally when a detection cache exists.

---

**Step 1: Find the output path construction pattern**

```bash
grep -n "confidence_regions\|_with_pose\|output_path\|csv_path\|save_path" \
  src/multi_tracker/gui/main_window.py | head -30
```

Use the same path-construction pattern as the existing `_with_pose.csv`.

**Step 2: Add config keys to `configs/default.json`**

Open `configs/default.json` and add inside the top-level object:

```json
"density_gaussian_sigma_scale": 1.0,
"density_temporal_sigma": 2.0,
"density_binarize_threshold": 0.3
```

**Step 3: Add density computation call in `main_window.py`**

Find the method that runs after post-processing completes and the output CSV path is known. After the existing CSV-writing logic, add:

```python
# --- Confidence density map ---
detection_cache_path = self._get_detection_cache_path()  # use existing helper
if detection_cache_path and Path(detection_cache_path).exists():
    try:
        from multi_tracker.afterhours.core.confidence_density import (
            compute_density_map_from_cache,
            save_regions,
            export_diagnostic_video,
        )
        from multi_tracker.data.detection_cache import DetectionCache

        cache = DetectionCache(detection_cache_path, mode="r")
        dm, raw_grids = compute_density_map_from_cache(
            detection_cache=cache,
            frame_h=self._video_height,
            frame_w=self._video_width,
            sigma_scale=params.get("density_gaussian_sigma_scale", 1.0),
            temporal_sigma=params.get("density_temporal_sigma", 2.0),
            threshold=params.get("density_binarize_threshold", 0.3),
        )
        cache.close()

        regions_path = Path(output_csv_path).with_name(
            Path(output_csv_path).stem + "_confidence_regions.json"
        )
        save_regions(dm.regions, regions_path)

        diag_path = Path(output_csv_path).with_name(
            Path(output_csv_path).stem + "_confidence_map.mp4"
        )
        # Build a lightweight frame reader from the video path
        cap_diag = cv2.VideoCapture(str(self._video_path))

        def _diag_reader(fidx):
            cap_diag.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, fr = cap_diag.read()
            return fr if ok else None

        export_diagnostic_video(
            frame_reader=_diag_reader,
            n_frames=int(cap_diag.get(cv2.CAP_PROP_FRAME_COUNT)),
            frame_h=self._video_height,
            frame_w=self._video_width,
            density_grids=raw_grids,
            regions=dm.regions,
            output_path=diag_path,
            fps=self._video_fps,
        )
        cap_diag.release()
        logger.info(f"Confidence map written: {diag_path}")
    except Exception:
        logger.exception("Confidence density map generation failed (non-fatal)")
```

Note: wrap in try/except so a failure here never blocks the tracking result.

**Step 4: Smoke-test manually** (no automated test — requires real video)

Run MAT on a short test video. Confirm `_confidence_regions.json` and `_confidence_map.mp4` appear alongside the output CSV.

**Step 5: Commit**

```bash
git add src/multi_tracker/gui/main_window.py configs/default.json
git commit -m "feat(mat): generate confidence density map artifacts after tracking"
```

---

### Task 4: Density-aware backward-pass assignment

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py`
- Create: `tests/test_density_aware_assignment.py`

**Context:**
In flagged regions (`region-N` tagged detections), the backward pass uses tighter assignment parameters: smaller max-distance and a stronger preference for "no assignment" (spawn a new fragment) over a risky long-distance match. The density regions JSON is loaded once before the backward pass begins.

---

**Step 1: Write failing test**

```python
# tests/test_density_aware_assignment.py
import numpy as np
from multi_tracker.afterhours.core.confidence_density import DensityRegion
from multi_tracker.core.tracking.worker import get_assignment_params_for_region


def test_open_field_returns_default_params():
    """Detections in open field use standard assignment distance."""
    params = get_assignment_params_for_region(
        region_label="open_field",
        base_max_distance=50.0,
        conservative_factor=0.5,
    )
    assert params["max_distance"] == 50.0


def test_flagged_region_tightens_distance():
    """Detections in a flagged region get reduced max assignment distance."""
    params = get_assignment_params_for_region(
        region_label="region-1",
        base_max_distance=50.0,
        conservative_factor=0.5,
    )
    assert params["max_distance"] < 50.0
    assert params["max_distance"] == pytest.approx(25.0)


def test_conservative_factor_zero_means_no_change():
    """conservative_factor=0 disables density-aware adjustment."""
    params = get_assignment_params_for_region(
        region_label="region-1",
        base_max_distance=50.0,
        conservative_factor=0.0,
    )
    assert params["max_distance"] == 50.0
```

Add `import pytest` to the test file.

**Step 2: Run test to verify it fails**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_density_aware_assignment.py -v
```

Expected: `ImportError`.

---

**Step 3: Add `get_assignment_params_for_region` to worker.py**

Find the top of `worker.py` (before the class definition) and add:

```python
def get_assignment_params_for_region(
    region_label: str,
    base_max_distance: float,
    conservative_factor: float = 0.5,
) -> dict:
    """
    Return assignment parameters adjusted for the detection region.

    In flagged regions the max assignment distance is reduced by
    (1 - conservative_factor) to prefer fragment spawning over
    long-distance risky matches.
    """
    if region_label == "open_field" or conservative_factor == 0.0:
        return {"max_distance": base_max_distance}
    reduced = base_max_distance * (1.0 - conservative_factor)
    return {"max_distance": reduced}
```

**Step 4: Load regions before backward pass and apply per-detection**

In the backward pass section of `worker.py`, after the detection cache is opened:

```python
# Load density regions if available
_density_regions = []
if self.detection_cache_path:
    from pathlib import Path as _Path
    from multi_tracker.afterhours.core.confidence_density import load_regions as _load_regions
    _regions_path = _Path(self.detection_cache_path).with_suffix("").with_name(
        _Path(self.detection_cache_path).stem.replace("_cache", "") + "_confidence_regions.json"
    )
    if _regions_path.exists():
        try:
            _density_regions = _load_regions(_regions_path)
        except Exception:
            _density_regions = []
```

Then, inside the backward-pass assignment loop when computing `max_distance` for each frame's detections, call `get_assignment_params_for_region` using the detection's region label (tag detections on the fly using `tag_detections` helper).

**Step 5: Run tests**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_density_aware_assignment.py -v
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/multi_tracker/core/tracking/worker.py tests/test_density_aware_assignment.py
git commit -m "feat(tracking): density-aware backward-pass assignment in flagged regions"
```

---

## Part B — Swap Suspicion Scorer

---

### Task 5: SwapSuspicionEvent dataclass + scorer skeleton

**Files:**
- Create: `src/multi_tracker/afterhours/core/swap_scorer.py`
- Create: `tests/test_swap_scorer.py`

---

**Step 1: Write failing tests**

```python
# tests/test_swap_scorer.py
import pandas as pd
import numpy as np
import pytest
from multi_tracker.afterhours.core.swap_scorer import (
    SwapSuspicionEvent,
    SwapScorer,
)


def _make_df(track_id, frames, xs, ys, thetas=None, pose_quality=None):
    """Build a minimal trajectory DataFrame."""
    n = len(frames)
    d = {
        "TrajectoryID": track_id,
        "FrameID": frames,
        "X": xs,
        "Y": ys,
        "Theta": thetas if thetas is not None else [0.0] * n,
        "State": ["active"] * n,
    }
    if pose_quality is not None:
        d["PoseQualityScore"] = pose_quality
    return pd.DataFrame(d)


def test_swap_suspicion_event_fields():
    """SwapSuspicionEvent has required fields and sensible defaults."""
    ev = SwapSuspicionEvent(
        track_a=1,
        track_b=2,
        frame_peak=100,
        frame_range=(90, 110),
        score=0.75,
        signals=["Cr", "Hd"],
        region_label="open_field",
        region_boundary=False,
    )
    assert ev.score == pytest.approx(0.75)
    assert "Cr" in ev.signals


def test_scorer_returns_list():
    """SwapScorer.score returns a list (possibly empty) of events."""
    df = pd.concat([
        _make_df(1, list(range(10)), list(range(10)), [0.0] * 10),
        _make_df(2, list(range(10)), [50.0] * 10, [0.0] * 10),
    ])
    scorer = SwapScorer(regions=[])
    events = scorer.score(df)
    assert isinstance(events, list)


def test_scorer_ranks_by_score_descending():
    """Events are returned sorted highest score first."""
    df = pd.concat([
        _make_df(1, list(range(20)), list(range(20)), [0.0] * 20),
        _make_df(2, list(range(20)), list(reversed(range(20))), [0.0] * 20),
    ])
    scorer = SwapScorer(regions=[])
    events = scorer.score(df)
    if len(events) >= 2:
        scores = [e.score for e in events]
        assert scores == sorted(scores, reverse=True)


def test_scorer_filters_below_threshold():
    """Events below min_score are excluded from output."""
    df = pd.concat([
        _make_df(1, list(range(10)), list(range(10)), [0.0] * 10),
        _make_df(2, list(range(10)), [500.0] * 10, [0.0] * 10),
    ])
    scorer = SwapScorer(regions=[], min_score=0.99)
    events = scorer.score(df)
    assert all(e.score >= 0.99 for e in events)
```

**Step 2: Run tests to verify they fail**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_swap_scorer.py -v
```

Expected: `ImportError`.

---

**Step 3: Implement skeleton**

Create `src/multi_tracker/afterhours/core/swap_scorer.py`:

```python
"""
Swap suspicion scorer.

Scores pairs of trajectories for the likelihood of an identity swap.
Signals:
  Cr  — crossing / position exchange (strong)
  Pr  — close approach + separation (moderate)
  Hd  — heading discontinuity (medium)
  Pq  — pose quality drop (amplifier, requires pose)
  Region context — discount in ambiguous zones, bonus at boundaries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from multi_tracker.afterhours.core.confidence_density import DensityRegion

logger = logging.getLogger(__name__)

# Signal weights
W_CROSSING = 1.0
W_PROXIMITY = 0.45
W_HEADING = 0.5
W_POSE_DROP = 0.3       # amplifier only — added when Cr or Pr also fires
W_REGION_DISCOUNT = 0.4
W_BOUNDARY_BONUS = 0.25


@dataclass
class SwapSuspicionEvent:
    track_a: int
    track_b: int | None
    frame_peak: int
    frame_range: tuple[int, int]
    score: float
    signals: list[str] = field(default_factory=list)
    region_label: str = "open_field"
    region_boundary: bool = False


class SwapScorer:
    """
    Score all trajectory pairs in a DataFrame for swap suspicion.

    Parameters
    ----------
    regions : list of DensityRegion
    min_score : float
        Events below this threshold are excluded from output by default.
        Lower this via get_more() to surface weaker events.
    approach_distance : float
        Max distance (px) for two tracks to be considered "approaching".
    crossing_window : int
        Frames within which a position exchange must occur to count as Cr.
    heading_reversal_deg : float
        Minimum heading change (degrees) to flag as discontinuity.
    """

    def __init__(
        self,
        regions: list[DensityRegion],
        min_score: float = 0.15,
        approach_distance: float = 60.0,
        crossing_window: int = 15,
        heading_reversal_deg: float = 120.0,
    ):
        self.regions = regions
        self.min_score = min_score
        self.approach_distance = approach_distance
        self.crossing_window = crossing_window
        self.heading_reversal_rad = np.deg2rad(heading_reversal_deg)

    def score(self, df: pd.DataFrame, min_score: float | None = None) -> list[SwapSuspicionEvent]:
        """
        Score all pairs and return events above min_score, sorted descending.
        """
        threshold = min_score if min_score is not None else self.min_score
        track_ids = sorted(df["TrajectoryID"].unique())
        events: list[SwapSuspicionEvent] = []

        for i, tid_a in enumerate(track_ids):
            for tid_b in track_ids[i + 1:]:
                df_a = df[df["TrajectoryID"] == tid_a].sort_values("FrameID")
                df_b = df[df["TrajectoryID"] == tid_b].sort_values("FrameID")
                ev = self._score_pair(df_a, df_b)
                if ev is not None and ev.score >= threshold:
                    events.append(ev)

        events.sort(key=lambda e: e.score, reverse=True)
        return events

    def _score_pair(self, df_a: pd.DataFrame, df_b: pd.DataFrame) -> SwapSuspicionEvent | None:
        """Score a single pair of trajectories."""
        common = np.intersect1d(df_a["FrameID"].values, df_b["FrameID"].values)
        if len(common) < 3:
            return None

        a = df_a.set_index("FrameID").loc[common]
        b = df_b.set_index("FrameID").loc[common]

        distances = np.sqrt((a["X"].values - b["X"].values) ** 2 +
                            (a["Y"].values - b["Y"].values) ** 2)

        cr_score, cr_frame = self._crossing_signal(a, b, common, distances)
        pr_score, pr_frame = self._proximity_signal(distances, common)
        hd_a_score, hd_a_frame = self._heading_signal(df_a)
        hd_b_score, hd_b_frame = self._heading_signal(df_b)
        hd_score = max(hd_a_score, hd_b_score)
        peak_frame = cr_frame or pr_frame or hd_a_frame or hd_b_frame
        if peak_frame is None:
            return None

        pq_score = self._pose_quality_signal(df_a, df_b, peak_frame)

        # Region context
        region_label, region_boundary = self._get_region_context(
            float(df_a[df_a["FrameID"] == peak_frame]["X"].iloc[0]) if not df_a[df_a["FrameID"] == peak_frame].empty else 0.0,
            float(df_a[df_a["FrameID"] == peak_frame]["Y"].iloc[0]) if not df_a[df_a["FrameID"] == peak_frame].empty else 0.0,
            peak_frame,
        )
        region_discount = W_REGION_DISCOUNT if region_label != "open_field" and not region_boundary else 0.0
        boundary_bonus = W_BOUNDARY_BONUS if region_boundary else 0.0

        # Pose amplifier only fires alongside Cr or Pr
        pq_term = W_POSE_DROP * pq_score if (cr_score > 0 or pr_score > 0) else 0.0

        raw = (
            W_CROSSING * cr_score
            + W_PROXIMITY * pr_score
            + W_HEADING * hd_score
            + pq_term
            - region_discount
            + boundary_bonus
        )
        final_score = float(np.clip(raw, 0.0, 1.0))

        signals = []
        if cr_score > 0:
            signals.append("Cr")
        if pr_score > 0:
            signals.append("Pr")
        if hd_score > 0:
            signals.append("Hd")
        if pq_term > 0:
            signals.append("Pq")

        if not signals:
            return None

        margin = self.crossing_window
        frame_range = (max(0, peak_frame - margin), peak_frame + margin)

        return SwapSuspicionEvent(
            track_a=int(df_a["TrajectoryID"].iloc[0]),
            track_b=int(df_b["TrajectoryID"].iloc[0]),
            frame_peak=peak_frame,
            frame_range=frame_range,
            score=final_score,
            signals=signals,
            region_label=region_label,
            region_boundary=region_boundary,
        )

    # ── Individual signal methods ────────────────────────────────────────────

    def _crossing_signal(self, a, b, common, distances):
        """Detect position exchange: tracks swap relative position within window."""
        close_mask = distances < self.approach_distance
        if not close_mask.any():
            return 0.0, None

        # Relative x-position: sign(A.x - B.x) before vs after approach
        close_frames = common[close_mask]
        peak_frame = int(close_frames[len(close_frames) // 2])

        # Look for sign flip across the window
        w = self.crossing_window
        idx_peak = np.searchsorted(common, peak_frame)
        before = common[max(0, idx_peak - w): idx_peak]
        after = common[idx_peak + 1: min(len(common), idx_peak + w + 1)]

        if len(before) == 0 or len(after) == 0:
            return 0.0, None

        sign_before = np.sign(a.loc[before, "X"].values - b.loc[before, "X"].values)
        sign_after = np.sign(a.loc[after, "X"].values - b.loc[after, "X"].values)

        if len(sign_before) == 0 or len(sign_after) == 0:
            return 0.0, None

        flipped = (np.median(sign_before) * np.median(sign_after)) < 0
        if not flipped:
            return 0.0, None

        min_dist = float(distances[close_mask].min())
        strength = 1.0 - min(min_dist / self.approach_distance, 1.0)
        return float(strength), peak_frame

    def _proximity_signal(self, distances, common):
        """Close approach + separation without position exchange."""
        close_mask = distances < self.approach_distance
        if not close_mask.any():
            return 0.0, None
        min_dist = float(distances[close_mask].min())
        strength = 1.0 - min(min_dist / self.approach_distance, 1.0)
        peak_idx = int(np.argmin(distances))
        return float(strength * 0.6), int(common[peak_idx])  # 60% of full strength

    def _heading_signal(self, df: pd.DataFrame):
        """Detect sudden heading reversal."""
        if len(df) < 3:
            return 0.0, None
        df_s = df.sort_values("FrameID")
        thetas = df_s["Theta"].values
        # Circular difference
        diffs = np.abs(np.diff(thetas))
        diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
        if diffs.max() < self.heading_reversal_rad:
            return 0.0, None
        idx = int(np.argmax(diffs))
        strength = min(float(diffs[idx]) / np.pi, 1.0)
        frame = int(df_s["FrameID"].values[idx + 1])
        return strength, frame

    def _pose_quality_signal(self, df_a, df_b, peak_frame, window=10, zscore_thresh=2.5):
        """Detect pose quality drop near peak_frame."""
        if "PoseQualityScore" not in df_a.columns:
            return 0.0
        scores = []
        for df in (df_a, df_b):
            near = df[abs(df["FrameID"] - peak_frame) <= window]["PoseQualityScore"]
            far = df["PoseQualityScore"]
            if len(near) == 0 or len(far) < 3:
                continue
            std = far.std()
            if std < 1e-6:
                continue
            drop = (far.mean() - near.mean()) / std
            scores.append(max(0.0, float(drop)))
        if not scores:
            return 0.0
        return min(max(scores) / zscore_thresh, 1.0)

    def _get_region_context(self, cx, cy, frame):
        """Return (region_label, is_boundary) for a given position and frame."""
        for r in self.regions:
            if r.contains(frame, cx, cy):
                return r.label, r.is_boundary_frame(frame)
        return "open_field", False
```

**Step 4: Run tests**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_swap_scorer.py -v
```

Expected: All pass.

**Step 5: Commit**

```bash
git add src/multi_tracker/afterhours/core/swap_scorer.py tests/test_swap_scorer.py
git commit -m "feat(afterhours): swap suspicion scorer with five signals"
```

---

### Task 6: Correction writer

**Files:**
- Create: `src/multi_tracker/afterhours/core/correction_writer.py`
- Create: `tests/test_correction_writer.py`

**Context:**
Takes a `_proofread.csv`, a split frame, two TrajectoryIDs, and an identity mapping, then performs:
1. Break track A at frame N → two new trajectory ID segments
2. Break track B at frame N → two new trajectory ID segments
3. Swap post-split segment IDs per user mapping
4. Write atomically (`.tmp` rename)

---

**Step 1: Write failing tests**

```python
# tests/test_correction_writer.py
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from multi_tracker.afterhours.core.correction_writer import (
    CorrectionWriter,
    apply_split_and_swap,
)


def _make_two_track_df():
    frames = list(range(1, 21))
    df = pd.DataFrame({
        "TrajectoryID": [1] * 20 + [2] * 20,
        "FrameID": frames + frames,
        "X": list(range(20)) + list(range(100, 120)),
        "Y": [0.0] * 40,
        "Theta": [0.0] * 40,
        "State": ["active"] * 40,
    })
    return df


def test_apply_split_creates_new_segments():
    """Split at frame 10 creates 4 trajectory segments from 2."""
    df = _make_two_track_df()
    result = apply_split_and_swap(
        df=df,
        track_a=1,
        track_b=2,
        split_frame=10,
        swap_post=True,    # swap identities after split
    )
    tids = sorted(result["TrajectoryID"].unique())
    # Original 1 and 2 each split into two segments → 4 unique IDs
    assert len(tids) == 4


def test_apply_split_no_row_loss():
    """All rows are preserved after split."""
    df = _make_two_track_df()
    result = apply_split_and_swap(df, 1, 2, split_frame=10, swap_post=True)
    assert len(result) == len(df)


def test_apply_split_no_swap():
    """With swap_post=False identities are not exchanged."""
    df = _make_two_track_df()
    result = apply_split_and_swap(df, 1, 2, split_frame=10, swap_post=False)
    # Post-split rows for track 1 still belong to a segment derived from 1
    pre_1 = result[(result["FrameID"] < 10)]
    post_1 = result[(result["FrameID"] >= 10)]
    assert len(pre_1) > 0
    assert len(post_1) > 0


def test_correction_writer_creates_proofread_copy():
    """CorrectionWriter.open creates _proofread.csv if it doesn't exist."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        assert writer.proofread_path.exists()
        writer.close()


def test_correction_writer_does_not_overwrite_existing():
    """If _proofread.csv already exists it is NOT overwritten on open."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        proofread = Path(tmp) / "test_tracked_proofread.csv"
        proofread.write_text("existing content")
        writer = CorrectionWriter(src)
        writer.open()
        assert proofread.read_text() == "existing content"
        writer.close()


def test_correction_writer_apply_correction_writes_atomically():
    """apply_correction writes .tmp then renames — original survives a crash."""
    df = _make_two_track_df()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "test_tracked.csv"
        df.to_csv(src, index=False)
        writer = CorrectionWriter(src)
        writer.open()
        writer.apply_correction(track_a=1, track_b=2, split_frame=10, swap_post=True)
        result = pd.read_csv(writer.proofread_path)
        assert len(result["TrajectoryID"].unique()) == 4
        writer.close()
```

**Step 2: Run tests to verify failure**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_correction_writer.py -v
```

Expected: `ImportError`.

---

**Step 3: Implement**

Create `src/multi_tracker/afterhours/core/correction_writer.py`:

```python
"""
Atomic correction writer for _proofread.csv.

Applies split + identity swap corrections to the proofread copy of the
trajectory CSV. Never touches the original CSV.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

_NEW_ID_OFFSET = 100_000  # Added to base IDs to create post-split segment IDs


def apply_split_and_swap(
    df: pd.DataFrame,
    track_a: int,
    track_b: int,
    split_frame: int,
    swap_post: bool,
) -> pd.DataFrame:
    """
    Split track_a and track_b at split_frame, optionally swapping post-split IDs.

    New segment IDs:
      track_a, pre-split  → track_a  (unchanged)
      track_a, post-split → track_a + _NEW_ID_OFFSET   (or track_b + _NEW_ID_OFFSET if swapped)
      track_b, pre-split  → track_b  (unchanged)
      track_b, post-split → track_b + _NEW_ID_OFFSET   (or track_a + _NEW_ID_OFFSET if swapped)
    """
    df = df.copy()

    mask_a_post = (df["TrajectoryID"] == track_a) & (df["FrameID"] >= split_frame)
    mask_b_post = (df["TrajectoryID"] == track_b) & (df["FrameID"] >= split_frame)

    if swap_post:
        # A's post-split gets B's post-split ID family and vice versa
        df.loc[mask_a_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
    else:
        df.loc[mask_a_post, "TrajectoryID"] = track_a + _NEW_ID_OFFSET
        df.loc[mask_b_post, "TrajectoryID"] = track_b + _NEW_ID_OFFSET

    return df


class CorrectionWriter:
    """
    Manages the _proofread.csv lifecycle: open → apply corrections → close.

    The proofread copy is created once from the original CSV. Subsequent
    opens load the existing proofread copy without overwriting it.
    """

    def __init__(self, source_csv: Path | str):
        self.source_csv = Path(source_csv)
        stem = self.source_csv.stem
        self.proofread_path = self.source_csv.with_name(f"{stem}_proofread.csv")
        self._df: pd.DataFrame | None = None

    def open(self) -> None:
        """Create proofread copy if absent; load into memory."""
        if not self.proofread_path.exists():
            shutil.copy2(self.source_csv, self.proofread_path)
            logger.info(f"Created proofread copy: {self.proofread_path}")
        self._df = pd.read_csv(self.proofread_path)

    def apply_correction(
        self,
        track_a: int,
        track_b: int,
        split_frame: int,
        swap_post: bool,
    ) -> None:
        """Apply split + swap correction and write atomically."""
        if self._df is None:
            raise RuntimeError("Call open() before apply_correction()")
        self._df = apply_split_and_swap(self._df, track_a, track_b, split_frame, swap_post)
        self._write_atomic()

    def _write_atomic(self) -> None:
        tmp = self.proofread_path.with_suffix(".tmp")
        self._df.to_csv(tmp, index=False)
        os.replace(tmp, self.proofread_path)

    def close(self) -> None:
        self._df = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call open() first")
        return self._df
```

**Step 4: Run tests**

```bash
conda run -n multi-animal-tracker python -m pytest tests/test_correction_writer.py -v
```

Expected: All pass.

**Step 5: Commit**

```bash
git add src/multi_tracker/afterhours/core/correction_writer.py tests/test_correction_writer.py
git commit -m "feat(afterhours): atomic correction writer for _proofread.csv"
```

---

## Part C — MAT-afterhours Application

---

### Task 7: Application entry point + pyproject.toml

**Files:**
- Create: `src/multi_tracker/afterhours/app.py`
- Modify: `pyproject.toml`
- Create: `brand/matafterhours.svg` (placeholder — 400×400 SVG)

---

**Step 1: Add entry points to pyproject.toml**

Open `pyproject.toml`. Find the `[project.scripts]` section and add:

```toml
mat-afterhours = "multi_tracker.afterhours.app:main"
afterhours = "multi_tracker.afterhours.app:main"
```

**Step 2: Create `app.py`** (follows ClassKit pattern exactly)

```python
# src/multi_tracker/afterhours/app.py
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from .gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MATAfterhours")
    app.setApplicationDisplayName("MAT Afterhours")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("mat-afterhours")

    try:
        icon_path = Path(__file__).resolve().parents[3] / "brand" / "matafterhours.svg"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

**Step 3: Create placeholder icon**

```
brand/matafterhours.svg
```

Minimal SVG:
```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">
  <rect width="400" height="400" rx="60" fill="#1a1a2e"/>
  <text x="200" y="230" font-size="120" text-anchor="middle" fill="#e94560" font-family="monospace">AH</text>
</svg>
```

**Step 4: Reinstall package so entry point is registered**

```bash
conda run -n multi-animal-tracker pip install -e . --no-build-isolation
```

**Step 5: Verify entry point exists**

```bash
conda run -n multi-animal-tracker which afterhours
```

Expected: path to the installed script.

**Step 6: Commit**

```bash
git add pyproject.toml src/multi_tracker/afterhours/app.py brand/matafterhours.svg
git commit -m "feat(afterhours): register mat-afterhours entry point and app bootstrap"
```

---

### Task 8: MainWindow skeleton — tabbed layout + session navigator

**Files:**
- Create: `src/multi_tracker/afterhours/gui/__init__.py`
- Create: `src/multi_tracker/afterhours/gui/main_window.py`

**Context:**
The window has: a top session navigator bar (prev/next, counter, load button), a tab bar (Swap Review | Merge Review | Manual Edit), and below that a splitter between the left panel (tab-specific) and a right side containing the video player above and timeline panel below.

---

**Step 1: Create `__init__.py`**

Empty file: `src/multi_tracker/afterhours/gui/__init__.py`

**Step 2: Create `main_window.py`**

```python
# src/multi_tracker/afterhours/gui/main_window.py
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.correction_writer import CorrectionWriter
from multi_tracker.afterhours.gui.widgets.suspicion_queue import SuspicionQueueWidget
from multi_tracker.afterhours.gui.widgets.video_player import VideoPlayerWidget
from multi_tracker.afterhours.gui.widgets.timeline_panel import TimelinePanelWidget

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MAT Afterhours")

        # Session state
        self._video_paths: list[Path] = []
        self._session_index: int = 0
        self._writer: CorrectionWriter | None = None

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        root_layout.addWidget(self._build_session_bar())
        root_layout.addWidget(self._build_content_area(), stretch=1)

    def _build_session_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)

        self._btn_prev = QPushButton("◀ Prev")
        self._btn_prev.clicked.connect(self._prev_session)
        self._btn_prev.setEnabled(False)

        self._lbl_session = QLabel("No session loaded")
        self._lbl_session.setAlignment(Qt.AlignCenter)

        self._btn_next = QPushButton("Next ▶")
        self._btn_next.clicked.connect(self._next_session)
        self._btn_next.setEnabled(False)

        self._btn_load = QPushButton("⊕ Load")
        self._btn_load.clicked.connect(self._load_session)

        layout.addWidget(self._btn_prev)
        layout.addWidget(self._lbl_session, stretch=1)
        layout.addWidget(self._btn_next)
        layout.addStretch()
        layout.addWidget(self._btn_load)
        return bar

    def _build_content_area(self) -> QWidget:
        # Tabs
        self._tabs = QTabWidget()

        # Swap Review tab
        swap_widget = self._build_swap_review_tab()
        self._tabs.addTab(swap_widget, "Swap Review")

        # Placeholder tabs
        for label in ("Merge Review", "Manual Edit"):
            placeholder = QLabel(f"{label} — coming soon")
            placeholder.setAlignment(Qt.AlignCenter)
            self._tabs.addTab(placeholder, label)

        return self._tabs

    def _build_swap_review_tab(self) -> QWidget:
        splitter_h = QSplitter(Qt.Horizontal)

        # Left: suspicion queue
        self._queue_widget = SuspicionQueueWidget()
        self._queue_widget.event_selected.connect(self._on_event_selected)
        splitter_h.addWidget(self._queue_widget)

        # Right: video + timeline stacked vertically
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self._video_player = VideoPlayerWidget()
        self._timeline = TimelinePanelWidget()
        self._timeline.split_requested.connect(self._on_manual_split)

        splitter_v = QSplitter(Qt.Vertical)
        splitter_v.addWidget(self._video_player)
        splitter_v.addWidget(self._timeline)
        splitter_v.setSizes([600, 200])

        right_layout.addWidget(splitter_v)
        splitter_h.addWidget(right)
        splitter_h.setSizes([280, 1000])

        return splitter_h

    # ── Session management ───────────────────────────────────────────────────

    def _load_session(self):
        choice = QMessageBox.question(
            self,
            "Load session",
            "Load a single video or a .txt list of videos?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._load_single_video()
        elif choice == QMessageBox.StandardButton.No:
            self._load_video_list()

    def _load_single_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Video files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self._video_paths = [Path(path)]
            self._session_index = 0
            self._open_current_session()

    def _load_video_list(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video list", "", "Text files (*.txt)"
        )
        if path:
            lines = Path(path).read_text().splitlines()
            self._video_paths = [Path(l.strip()) for l in lines if l.strip()]
            self._session_index = 0
            self._open_current_session()

    def _open_current_session(self):
        if not self._video_paths:
            return
        video_path = self._video_paths[self._session_index]
        csv_path = self._discover_csv(video_path)

        if self._writer:
            self._writer.close()

        if csv_path is None:
            QMessageBox.warning(self, "No CSV found",
                                f"Could not find a processed CSV alongside:\n{video_path}")
            return

        self._writer = CorrectionWriter(csv_path)
        self._writer.open()

        self._video_player.load_video(video_path)
        self._timeline.load_trajectories(self._writer.df)
        self._queue_widget.clear()

        self._update_session_label()
        self._run_scorer()

    def _discover_csv(self, video_path: Path) -> Path | None:
        """Find _with_pose.csv or standard tracked CSV alongside the video."""
        stem = video_path.stem
        parent = video_path.parent
        for suffix in ("_with_pose.csv", "_tracked.csv", ".csv"):
            candidate = parent / (stem + suffix)
            if candidate.exists():
                return candidate
        return None

    def _update_session_label(self):
        n = len(self._video_paths)
        idx = self._session_index + 1
        name = self._video_paths[self._session_index].stem if self._video_paths else ""
        self._lbl_session.setText(f"{name}  ({idx}/{n})")
        self._btn_prev.setEnabled(self._session_index > 0)
        self._btn_next.setEnabled(self._session_index < n - 1)

    def _prev_session(self):
        if self._session_index > 0:
            self._session_index -= 1
            self._open_current_session()

    def _next_session(self):
        if self._session_index < len(self._video_paths) - 1:
            self._session_index += 1
            self._open_current_session()

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _run_scorer(self):
        if self._writer is None:
            return
        from multi_tracker.afterhours.core.swap_scorer import SwapScorer
        from multi_tracker.afterhours.core.confidence_density import load_regions

        regions = []
        video_path = self._video_paths[self._session_index]
        regions_path = video_path.parent / (video_path.stem + "_confidence_regions.json")
        if regions_path.exists():
            try:
                regions = load_regions(regions_path)
            except Exception:
                pass

        scorer = SwapScorer(regions=regions)
        events = scorer.score(self._writer.df)
        self._queue_widget.populate(events)

    # ── Event handling ───────────────────────────────────────────────────────

    def _on_event_selected(self, event):
        self._video_player.seek_to(event.frame_peak)
        self._video_player.highlight_tracks([event.track_a, event.track_b])
        self._timeline.highlight_event(event)
        self._show_review_dialogs(event)

    def _show_review_dialogs(self, event):
        from multi_tracker.afterhours.gui.dialogs.frame_picker import FramePickerDialog
        dlg = FramePickerDialog(
            video_path=self._video_paths[self._session_index],
            frame_range=event.frame_range,
            track_ids=[event.track_a, event.track_b],
            df=self._writer.df,
            parent=self,
        )
        if dlg.exec() != dlg.Accepted:
            return

        split_frame = dlg.selected_frame()
        from multi_tracker.afterhours.gui.dialogs.identity_assignment import IdentityAssignmentDialog
        dlg2 = IdentityAssignmentDialog(
            video_path=self._video_paths[self._session_index],
            split_frame=split_frame,
            track_a=event.track_a,
            track_b=event.track_b,
            df=self._writer.df,
            parent=self,
        )
        if dlg2.exec() != dlg2.Accepted:
            return

        swap_post = dlg2.should_swap()
        self._writer.apply_correction(event.track_a, event.track_b, split_frame, swap_post)
        self._timeline.load_trajectories(self._writer.df)
        self._queue_widget.mark_resolved(event)

    def _on_manual_split(self, track_id: int, frame: int):
        """Called when user clicks the timeline to force a split."""
        from multi_tracker.afterhours.gui.dialogs.identity_assignment import IdentityAssignmentDialog
        dlg = IdentityAssignmentDialog(
            video_path=self._video_paths[self._session_index],
            split_frame=frame,
            track_a=track_id,
            track_b=None,
            df=self._writer.df,
            parent=self,
        )
        if dlg.exec() == dlg.Accepted:
            self._writer.apply_correction(track_id, track_id, frame, swap_post=False)
            self._timeline.load_trajectories(self._writer.df)

    def closeEvent(self, event):
        if self._writer:
            self._writer.close()
        super().closeEvent(event)
```

**Step 3: Verify the app opens (smoke test)**

```bash
conda run -n multi-animal-tracker python -c "
from multi_tracker.afterhours.gui.main_window import MainWindow
print('Import OK')
"
```

Expected: `Import OK` (widgets don't exist yet, so this will fail — that's the cue for the next tasks).

**Step 4: Commit scaffold**

```bash
git add src/multi_tracker/afterhours/gui/
git commit -m "feat(afterhours): MainWindow skeleton with tabs and session navigator"
```

---

### Task 9: SuspicionQueueWidget

**Files:**
- Create: `src/multi_tracker/afterhours/gui/widgets/__init__.py`
- Create: `src/multi_tracker/afterhours/gui/widgets/suspicion_queue.py`

---

**Step 1: Implement**

```python
# src/multi_tracker/afterhours/gui/widgets/suspicion_queue.py
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.swap_scorer import SwapSuspicionEvent

_TIER_THRESHOLDS = [0.6, 0.4, 0.25, 0.15, 0.0]  # each "show more" reveals next tier


class _EventCard(QFrame):
    clicked = Signal(object)   # emits the SwapSuspicionEvent

    def __init__(self, event: SwapSuspicionEvent, parent=None):
        super().__init__(parent)
        self.event = event
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self._resolved = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        signals_str = "+".join(self.event.signals) if self.event.signals else "—"
        top = QLabel(f"● {self.event.score:.2f}  {signals_str}")
        top.setStyleSheet("font-weight: bold;")

        frame_lbl = QLabel(f"T{self.event.frame_range[0]}–T{self.event.frame_range[1]}")
        tracks_lbl = QLabel(f"tracks {self.event.track_a} & {self.event.track_b}")
        region_lbl = QLabel(self.event.region_label)
        region_lbl.setStyleSheet("color: #888; font-size: 10px;")

        for w in (top, frame_lbl, tracks_lbl, region_lbl):
            layout.addWidget(w)

    def mark_resolved(self):
        self._resolved = True
        self.setStyleSheet("background: #1a3a1a;")

    def mousePressEvent(self, ev):
        if not self._resolved:
            self.clicked.emit(self.event)
        super().mousePressEvent(ev)


class SuspicionQueueWidget(QWidget):
    event_selected = Signal(object)  # SwapSuspicionEvent

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events: list[SwapSuspicionEvent] = []
        self._shown_tier = 0
        self._cards: list[_EventCard] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        header = QLabel("Suspicion Queue")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(3)
        self._list_layout.addStretch()

        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll, stretch=1)

        self._btn_more = QPushButton("Show more ↓")
        self._btn_more.clicked.connect(self._show_more)
        self._btn_more.setEnabled(False)
        layout.addWidget(self._btn_more)

    def populate(self, events: list[SwapSuspicionEvent]):
        self.clear()
        self._events = events
        self._shown_tier = 0
        self._render_tier()

    def clear(self):
        self._events = []
        self._cards = []
        self._shown_tier = 0
        # Remove all cards from layout
        while self._list_layout.count() > 1:  # keep the stretch
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._btn_more.setEnabled(False)

    def mark_resolved(self, event: SwapSuspicionEvent):
        for card in self._cards:
            if card.event is event:
                card.mark_resolved()
                break

    def _render_tier(self):
        threshold = _TIER_THRESHOLDS[min(self._shown_tier, len(_TIER_THRESHOLDS) - 1)]
        for ev in self._events:
            if ev.score >= threshold:
                if not any(c.event is ev for c in self._cards):
                    card = _EventCard(ev)
                    card.clicked.connect(self.event_selected)
                    self._cards.append(card)
                    self._list_layout.insertWidget(self._list_layout.count() - 1, card)

        has_more = self._shown_tier < len(_TIER_THRESHOLDS) - 1
        self._btn_more.setEnabled(has_more)

    def _show_more(self):
        self._shown_tier = min(self._shown_tier + 1, len(_TIER_THRESHOLDS) - 1)
        self._render_tier()
```

**Step 2: Commit**

```bash
git add src/multi_tracker/afterhours/gui/widgets/
git commit -m "feat(afterhours): SuspicionQueueWidget with tiered show-more"
```

---

### Task 10: VideoPlayerWidget

**Files:**
- Create: `src/multi_tracker/afterhours/gui/widgets/video_player.py`

**Context:**
Loads video frames on demand into a RAM cache (decoded numpy arrays). Renders the current frame as a QPixmap with trajectory overlay (colored dots + ID labels). Highlighted tracks get a pulsing ring. Scrub bar + frame counter below the canvas.

---

**Step 1: Implement**

```python
# src/multi_tracker/afterhours/gui/widgets/video_player.py
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

_TRACK_COLORS = [
    "#e94560", "#00b4d8", "#06d6a0", "#ffd166", "#ef476f",
    "#118ab2", "#073b4c", "#f77f00", "#a8dadc", "#457b9d",
]


class VideoPlayerWidget(QWidget):
    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._n_frames = 0
        self._current_frame = 0
        self._frame_cache: dict[int, np.ndarray] = {}
        self._max_cache = 200
        self._df: pd.DataFrame | None = None
        self._highlighted: list[int] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setStyleSheet("background: black;")
        layout.addWidget(self._canvas, stretch=1)

        controls = QWidget()
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(4, 0, 4, 0)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.valueChanged.connect(self._on_slider)

        self._lbl_frame = QLabel("0 / 0")
        self._lbl_frame.setFixedWidth(90)
        self._lbl_frame.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        ctrl_layout.addWidget(self._slider, stretch=1)
        ctrl_layout.addWidget(self._lbl_frame)
        layout.addWidget(controls)

    def load_video(self, path: Path):
        if self._cap:
            self._cap.release()
        self._frame_cache.clear()
        self._cap = cv2.VideoCapture(str(path))
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._slider.setMaximum(max(0, self._n_frames - 1))
        self.seek_to(0)

    def load_trajectories(self, df: pd.DataFrame):
        self._df = df
        self._refresh_display()

    def seek_to(self, frame: int):
        self._current_frame = max(0, min(frame, self._n_frames - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(self._current_frame)
        self._slider.blockSignals(False)
        self._refresh_display()
        self.frame_changed.emit(self._current_frame)

    def highlight_tracks(self, track_ids: list[int]):
        self._highlighted = [t for t in track_ids if t is not None]
        self._refresh_display()

    def _on_slider(self, value: int):
        self._current_frame = value
        self._refresh_display()
        self.frame_changed.emit(value)

    def _get_frame(self, idx: int) -> np.ndarray | None:
        if idx in self._frame_cache:
            return self._frame_cache[idx]
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        if len(self._frame_cache) >= self._max_cache:
            oldest = next(iter(self._frame_cache))
            del self._frame_cache[oldest]
        self._frame_cache[idx] = frame
        return frame

    def _refresh_display(self):
        frame = self._get_frame(self._current_frame)
        if frame is None:
            self._lbl_frame.setText(f"{self._current_frame} / {self._n_frames}")
            return

        overlay = frame.copy()
        self._draw_trajectories(overlay, self._current_frame)

        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._canvas.setPixmap(pix)
        self._lbl_frame.setText(f"{self._current_frame} / {self._n_frames}")

    def _draw_trajectories(self, frame: np.ndarray, current_frame: int):
        if self._df is None:
            return
        at_frame = self._df[self._df["FrameID"] == current_frame]
        for _, row in at_frame.iterrows():
            tid = int(row["TrajectoryID"])
            cx, cy = int(row["X"]), int(row["Y"])
            color_hex = _TRACK_COLORS[tid % len(_TRACK_COLORS)]
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            color_bgr = (b, g, r)
            radius = 8
            if tid in self._highlighted:
                cv2.circle(frame, (cx, cy), radius + 4, color_bgr, 2)
            cv2.circle(frame, (cx, cy), radius, color_bgr, -1)
            cv2.putText(frame, str(tid), (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)

    def resizeEvent(self, event):
        self._refresh_display()
        super().resizeEvent(event)
```

**Step 2: Commit**

```bash
git add src/multi_tracker/afterhours/gui/widgets/video_player.py
git commit -m "feat(afterhours): VideoPlayerWidget with in-memory frame cache and overlay"
```

---

### Task 11: TimelinePanelWidget

**Files:**
- Create: `src/multi_tracker/afterhours/gui/widgets/timeline_panel.py`

---

**Step 1: Implement**

```python
# src/multi_tracker/afterhours/gui/widgets/timeline_panel.py
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import QAbstractScrollArea, QScrollArea, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)

_ROW_H = 18
_LABEL_W = 60
_COLORS = [
    "#e94560", "#00b4d8", "#06d6a0", "#ffd166", "#ef476f",
    "#118ab2", "#073b4c", "#f77f00", "#a8dadc", "#457b9d",
]


class _TimelineCanvas(QWidget):
    split_at = Signal(int, int)  # (track_id, frame)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._track_ids: list[int] = []
        self._min_frame = 0
        self._max_frame = 1
        self._highlighted_event = None
        self.setMinimumHeight(40)

    def load_trajectories(self, df: pd.DataFrame):
        self._df = df
        self._track_ids = sorted(df["TrajectoryID"].unique().tolist())
        self._min_frame = int(df["FrameID"].min()) if len(df) else 0
        self._max_frame = int(df["FrameID"].max()) if len(df) else 1
        h = max(60, len(self._track_ids) * _ROW_H + 20)
        self.setMinimumHeight(h)
        self.setFixedHeight(h)
        self.update()

    def highlight_event(self, event):
        self._highlighted_event = event
        self.update()

    def paintEvent(self, ev):
        if self._df is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        total_frames = max(1, self._max_frame - self._min_frame)
        canvas_w = self.width() - _LABEL_W

        for row_idx, tid in enumerate(self._track_ids):
            y = row_idx * _ROW_H + 10
            color = QColor(_COLORS[tid % len(_COLORS)])

            # Label
            p.setPen(QPen(Qt.white))
            p.setFont(QFont("monospace", 9))
            p.drawText(0, y, _LABEL_W - 4, _ROW_H, Qt.AlignRight | Qt.AlignVCenter, str(tid))

            # Track bar
            tdf = self._df[self._df["TrajectoryID"] == tid].sort_values("FrameID")
            if tdf.empty:
                continue
            f_min = int(tdf["FrameID"].min())
            f_max = int(tdf["FrameID"].max())
            x1 = _LABEL_W + int((f_min - self._min_frame) / total_frames * canvas_w)
            x2 = _LABEL_W + int((f_max - self._min_frame) / total_frames * canvas_w)
            p.fillRect(x1, y + 3, max(2, x2 - x1), _ROW_H - 8, color)

        p.end()

    def mousePressEvent(self, ev):
        if self._df is None or ev.button() != Qt.LeftButton:
            return
        total_frames = max(1, self._max_frame - self._min_frame)
        canvas_w = self.width() - _LABEL_W
        x = ev.position().x() - _LABEL_W
        frame = self._min_frame + int(x / canvas_w * total_frames)

        y = ev.position().y()
        row = int((y - 10) / _ROW_H)
        if 0 <= row < len(self._track_ids):
            tid = self._track_ids[row]
            self.split_at.emit(tid, frame)


class TimelinePanelWidget(QWidget):
    split_requested = Signal(int, int)  # (track_id, frame)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._canvas = _TimelineCanvas()
        self._canvas.split_at.connect(self.split_requested)
        scroll.setWidget(self._canvas)
        layout.addWidget(scroll)

    def load_trajectories(self, df: pd.DataFrame):
        self._canvas.load_trajectories(df)

    def highlight_event(self, event):
        self._canvas.highlight_event(event)
```

**Step 2: Commit**

```bash
git add src/multi_tracker/afterhours/gui/widgets/timeline_panel.py
git commit -m "feat(afterhours): TimelinePanelWidget with click-to-split"
```

---

### Task 12: FramePickerDialog and IdentityAssignmentDialog

**Files:**
- Create: `src/multi_tracker/afterhours/gui/dialogs/__init__.py`
- Create: `src/multi_tracker/afterhours/gui/dialogs/frame_picker.py`
- Create: `src/multi_tracker/afterhours/gui/dialogs/identity_assignment.py`

---

**Step 1: Implement FramePickerDialog**

```python
# src/multi_tracker/afterhours/gui/dialogs/frame_picker.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QSlider,
    QVBoxLayout,
)


class FramePickerDialog(QDialog):
    """
    Shows a cropped video clip of a suspicious event.
    User scrubs to the exact split frame and clicks OK.
    """

    def __init__(
        self,
        video_path: Path,
        frame_range: tuple[int, int],
        track_ids: list[int],
        df: pd.DataFrame,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Split Frame")
        self.resize(600, 500)

        self._video_path = video_path
        self._frame_range = frame_range
        self._track_ids = track_ids
        self._df = df
        self._selected_frame = frame_range[0] + (frame_range[1] - frame_range[0]) // 2
        self._frame_cache: dict[int, np.ndarray] = {}

        self._cap = cv2.VideoCapture(str(video_path))
        self._build()
        self._update_display()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        instructions = QLabel("Scrub to the frame where the identity swap occurs, then click OK.")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setStyleSheet("background: black;")
        self._canvas.setMinimumHeight(300)
        layout.addWidget(self._canvas, stretch=1)

        self._lbl_frame = QLabel()
        self._lbl_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._lbl_frame)

        self._slider = QSlider(Qt.Horizontal)
        start, end = self._frame_range
        self._slider.setMinimum(start)
        self._slider.setMaximum(end)
        self._slider.setValue(self._selected_frame)
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_slider(self, value: int):
        self._selected_frame = value
        self._update_display()

    def _get_frame(self, idx: int) -> np.ndarray | None:
        if idx in self._frame_cache:
            return self._frame_cache[idx]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_cache[idx] = frame
        return frame

    def _update_display(self):
        frame = self._get_frame(self._selected_frame)
        if frame is None:
            return

        # Crop around the tracks
        frame = self._crop_to_tracks(frame, self._selected_frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self._canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._canvas.setPixmap(pix)
        self._lbl_frame.setText(f"Frame {self._selected_frame}")

    def _crop_to_tracks(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Crop frame to bounding box of the two tracks ± margin."""
        if self._df is None or not self._track_ids:
            return frame
        at = self._df[
            (self._df["FrameID"] == frame_idx) &
            (self._df["TrajectoryID"].isin(self._track_ids))
        ]
        if at.empty:
            return frame
        margin = 80
        h, w = frame.shape[:2]
        x1 = max(0, int(at["X"].min()) - margin)
        x2 = min(w, int(at["X"].max()) + margin)
        y1 = max(0, int(at["Y"].min()) - margin)
        y2 = min(h, int(at["Y"].max()) + margin)
        return frame[y1:y2, x1:x2]

    def selected_frame(self) -> int:
        return self._selected_frame

    def closeEvent(self, ev):
        self._cap.release()
        super().closeEvent(ev)
```

**Step 2: Implement IdentityAssignmentDialog**

```python
# src/multi_tracker/afterhours/gui/dialogs/identity_assignment.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

_THUMB_WINDOW = 5   # frames before/after split to show


class IdentityAssignmentDialog(QDialog):
    """
    Shows before/after thumbnail grids for two tracks around a split frame.
    User clicks a radio button to assign identity:
      "Keep order" — post-split identities stay the same
      "Swap"        — post-split identities are exchanged
    """

    def __init__(
        self,
        video_path: Path,
        split_frame: int,
        track_a: int,
        track_b: int | None,
        df: pd.DataFrame,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Assign Identity After Split")
        self.resize(700, 400)

        self._video_path = video_path
        self._split_frame = split_frame
        self._track_a = track_a
        self._track_b = track_b
        self._df = df
        self._cap = cv2.VideoCapture(str(video_path))
        self._swap = False

        self._build()

    def _build(self):
        layout = QVBoxLayout(self)

        instructions = QLabel(
            f"Split at frame {self._split_frame}. "
            "Should the identities be swapped after the split?"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        thumbs_layout = QHBoxLayout()

        before_box = QGroupBox("Before split")
        before_layout = QVBoxLayout(before_box)
        self._before_canvas = QLabel()
        self._before_canvas.setAlignment(Qt.AlignCenter)
        before_layout.addWidget(self._before_canvas)
        thumbs_layout.addWidget(before_box)

        after_box = QGroupBox("After split")
        after_layout = QVBoxLayout(after_box)
        self._after_canvas = QLabel()
        self._after_canvas.setAlignment(Qt.AlignCenter)
        after_layout.addWidget(self._after_canvas)
        thumbs_layout.addWidget(after_box)

        layout.addLayout(thumbs_layout, stretch=1)

        self._rb_keep = QRadioButton("Keep order (no swap)")
        self._rb_swap = QRadioButton("Swap identities")
        self._rb_keep.setChecked(True)
        self._rb_keep.toggled.connect(lambda checked: setattr(self, "_swap", not checked))

        layout.addWidget(self._rb_keep)
        layout.addWidget(self._rb_swap)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_thumbnails()

    def _load_thumbnails(self):
        before_frame = max(0, self._split_frame - _THUMB_WINDOW)
        after_frame = self._split_frame + _THUMB_WINDOW

        before = self._get_frame_with_overlay(before_frame)
        after = self._get_frame_with_overlay(after_frame)

        for canvas, img in ((self._before_canvas, before), (self._after_canvas, after)):
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pix = QPixmap.fromImage(qimg).scaled(300, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                canvas.setPixmap(pix)

    def _get_frame_with_overlay(self, frame_idx: int) -> np.ndarray | None:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        track_ids = [t for t in (self._track_a, self._track_b) if t is not None]
        at = self._df[
            (self._df["FrameID"] == frame_idx) &
            (self._df["TrajectoryID"].isin(track_ids))
        ]
        colors = [(0, 80, 200), (0, 180, 60)]
        for i, (_, row) in enumerate(at.iterrows()):
            cx, cy = int(row["X"]), int(row["Y"])
            c = colors[i % len(colors)]
            cv2.circle(frame, (cx, cy), 10, c, -1)
            cv2.putText(frame, str(int(row["TrajectoryID"])),
                        (cx + 12, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        # Crop around tracks
        margin = 80
        h, w = frame.shape[:2]
        if not at.empty:
            x1 = max(0, int(at["X"].min()) - margin)
            x2 = min(w, int(at["X"].max()) + margin)
            y1 = max(0, int(at["Y"].min()) - margin)
            y2 = min(h, int(at["Y"].max()) + margin)
            frame = frame[y1:y2, x1:x2]
        return frame

    def should_swap(self) -> bool:
        return self._swap

    def closeEvent(self, ev):
        self._cap.release()
        super().closeEvent(ev)
```

**Step 3: Commit**

```bash
git add src/multi_tracker/afterhours/gui/dialogs/
git commit -m "feat(afterhours): FramePickerDialog and IdentityAssignmentDialog"
```

---

## Part D — MAT Integration

---

### Task 13: MAT post-processing tab — "Open in MAT-afterhours" button

**Files:**
- Modify: `src/multi_tracker/gui/main_window.py`

**Context:**
After post-processing completes and an output CSV is written, show a button "Open in MAT-afterhours" in the post-processing tab. Also show a `QMessageBox` prompt at the end of a full tracking run.

---

**Step 1: Find the post-processing completion callback**

```bash
grep -n "post_process\|postprocess\|_on_post\|post_done" \
  src/multi_tracker/gui/main_window.py | head -20
```

**Step 2: Add the button to the post-processing tab**

In the UI builder for the post-processing tab, add at the bottom:

```python
self._btn_open_afterhours = QPushButton("Open in MAT-afterhours")
self._btn_open_afterhours.setEnabled(False)
self._btn_open_afterhours.clicked.connect(self._open_afterhours)
post_proc_layout.addWidget(self._btn_open_afterhours)
```

**Step 3: Enable the button and show prompt after tracking/post-processing completes**

In the post-processing done handler:

```python
def _on_postprocessing_done(self, output_csv_path):
    # ... existing code ...
    self._last_output_csv = output_csv_path
    self._btn_open_afterhours.setEnabled(True)

    reply = QMessageBox.question(
        self,
        "Open MAT-afterhours?",
        "Post-processing complete. Open in MAT-afterhours for interactive proofreading?",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    if reply == QMessageBox.Yes:
        self._open_afterhours()
```

**Step 4: Implement `_open_afterhours`**

```python
def _open_afterhours(self):
    """Launch mat-afterhours as a subprocess with the current output CSV's video."""
    import subprocess
    import sys
    video_path = str(self._video_path)
    subprocess.Popen([sys.executable, "-m", "multi_tracker.afterhours.app", video_path])
```

Update `app.py` to accept an optional video path argument:

```python
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=None)
    args = ap.parse_args()

    app = QApplication(sys.argv)
    # ... existing setup ...
    window = MainWindow()
    if args.video:
        from pathlib import Path
        window._video_paths = [Path(args.video)]
        window._session_index = 0
        window._open_current_session()
    window.showMaximized()
    sys.exit(app.exec())
```

**Step 5: Commit**

```bash
git add src/multi_tracker/gui/main_window.py src/multi_tracker/afterhours/app.py
git commit -m "feat(mat): add Open in MAT-afterhours button and post-tracking prompt"
```

---

### Task 14: Smoke test + format + lint

**Step 1: Format**

```bash
conda run -n multi-animal-tracker make format
```

**Step 2: Lint**

```bash
conda run -n multi-animal-tracker make lint
```

Fix any issues reported.

**Step 3: Run full test suite**

```bash
conda run -n multi-animal-tracker make pytest
```

Expected: All existing tests pass plus new tests for confidence_density, swap_scorer, correction_writer, density_aware_assignment.

**Step 4: Smoke test the application**

```bash
conda run -n multi-animal-tracker afterhours
```

Expected: MAT-afterhours window opens maximized with no errors.

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore(afterhours): format, lint, smoke test complete"
```

---

## Summary of New Files

```
src/multi_tracker/afterhours/
├── __init__.py
├── app.py
├── core/
│   ├── __init__.py
│   ├── confidence_density.py
│   ├── swap_scorer.py
│   └── correction_writer.py
└── gui/
    ├── __init__.py
    ├── main_window.py
    ├── widgets/
    │   ├── __init__.py
    │   ├── suspicion_queue.py
    │   ├── video_player.py
    │   └── timeline_panel.py
    └── dialogs/
        ├── __init__.py
        ├── frame_picker.py
        └── identity_assignment.py

tests/
├── test_confidence_density.py
├── test_confidence_density_video.py
├── test_swap_scorer.py
├── test_correction_writer.py
└── test_density_aware_assignment.py

brand/matafterhours.svg
docs/plans/2026-03-09-mat-afterhours-design.md  (already committed)
```

## New config keys in `configs/default.json`

| Key | Default | Description |
|---|---|---|
| `density_gaussian_sigma_scale` | 1.0 | Gaussian sigma as multiple of bbox_diagonal/2 |
| `density_temporal_sigma` | 2.0 | Temporal Gaussian smoothing (frames) |
| `density_binarize_threshold` | 0.3 | Binarization threshold (after normalisation) |
