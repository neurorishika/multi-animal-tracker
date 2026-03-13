# MAT-afterhours: Interactive Identity Proofreading — Design Document

**Date:** 2026-03-09
**Status:** Approved
**Branch:** mat-pose-integration

---

## Problem

Identity swaps occur when social animals cluster together. The tracker already employs conservative merging, velocity z-score breaks, pose quality gating, and DetectionID confirmation — but swaps still occur in high-density, low-confidence regions. The key failure modes are:

- Animals coming close and exchanging identity post-separation
- Detection flickering in crowds causing jumps between animals
- No structured way for users to find, review, or correct suspected swaps

---

## Solution Overview

Three coordinated components:

1. **Confidence density map** — computed during pre-tracking detection, characterises where tracking is hard
2. **Density-aware assignment** — forward and backward passes use the density map to prefer fragmentation over risky identity maintenance in ambiguous zones
3. **MAT-afterhours** — a standalone Qt application (fifth app in the suite) for interactive post-hoc proofreading of swap events

---

## Component 1: Confidence Density Map

### Location
`src/multi_tracker/afterhours/core/confidence_density.py`
Called from the pre-tracking detection phase (made default for all runs).

### Algorithm

**Accumulation (per frame, during pre-detection):**

For each detection at position `(x, y)` with confidence `c`:
- Place a 2D Gaussian at `(x, y)` with sigma proportional to `bbox_diagonal` (body-size-scaled)
- Gaussian is normalised; amplitude = `(1 - c)`
- Low confidence → strong signal; high confidence → near-zero; empty space → zero
- Accumulated into a per-frame float32 density frame stored in the detection cache `.npz`

**Post-detection processing:**

1. Load all per-frame density frames from cache
2. Apply temporal Gaussian smoothing across frames
3. Binarize at a configurable threshold
4. Run 3D connected-components (`scipy.ndimage.label`) on the `(H, W, T)` volume
5. Extract each component's bounding box (pixel coords + frame range) → `Region-N`
6. Tag each detection in cache with its region label (`open_field` or `region-N`)
7. Write `<name>_confidence_regions.json`

**Diagnostic video export (`<name>_confidence_map.mp4`):**

Generated automatically as part of the standard MAT post-processing artifact set:
- Original video frames with semi-transparent red heatmap overlay on low-confidence zones
- Animal IDs drawn in warning colour inside flagged regions
- Region labels (`Region-1 [frames 240–310]`) annotated in frame corners
- Exported at original resolution

### Config keys (new)

| Key | Default | Description |
|---|---|---|
| `density_gaussian_sigma_scale` | 1.0 | Gaussian sigma as multiple of bbox_diagonal |
| `density_temporal_sigma` | 2.0 | Frames for temporal Gaussian smoothing |
| `density_binarize_threshold` | 0.3 | Threshold after normalisation |

---

## Component 2: Density-Aware Assignment

### Insertion point

Pre-tracking detection is made the **default** for all runs (currently default only for batched runs). This ensures the density map is always available before both the forward and backward tracking passes.

### Mechanism

During assignment in flagged regions (`region-N` detections):
- **Tighter max assignment distance** — search radius reduced to discourage long-distance matches
- **Lower acceptance threshold** — prefer "no assignment → new fragment" over risky match
- **Occlusion preference** — in flagged zones, prefer marking as occluded over forcing an assignment through a crowd

This shifts the outcome in hard zones from *noisy identity maintenance* to *clean fragments awaiting proofreading* — exactly what MAT-afterhours is designed to handle.

---

## Component 3: MAT-afterhours Application

### Application structure

Follows the ClassKit pattern exactly:

```
src/multi_tracker/
└── afterhours/
    ├── __init__.py
    ├── app.py                          # main() entry point
    ├── core/
    │   ├── __init__.py
    │   ├── confidence_density.py       # Density map + diagnostic video
    │   ├── swap_scorer.py              # Suspicion scoring engine
    │   └── correction_writer.py        # Edits _proofread.csv atomically
    └── gui/
        ├── __init__.py
        ├── main_window.py              # MainWindow(QMainWindow)
        ├── widgets/
        │   ├── suspicion_queue.py      # Ranked queue panel
        │   ├── video_player.py         # Video + trajectory overlay
        │   └── timeline_panel.py       # Per-animal horizontal timeline
        └── dialogs/
            ├── frame_picker.py         # Step 1: pick split frame
            └── identity_assignment.py  # Step 2: before/after thumbnail reassignment
```

**Entry points (pyproject.toml):**
```toml
mat-afterhours = "multi_tracker.afterhours.app:main"
afterhours = "multi_tracker.afterhours.app:main"
```

**MAT integration:**
- Post-processing tab gets an "Open in MAT-afterhours" button
- After a full tracking run, a `QMessageBox` prompts "Open in MAT-afterhours?"

---

### UI Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  MAT-afterhours  |  [◀ prev]  fly_run_01  (2/8)  [next ▶]  [⊕ load] │
├──────────────────────────────────────────────────────────────────────┤
│  [Swap Review] [Merge Review] [Manual Edit] [...]                    │
├──────────────────┬───────────────────────────────────────────────────┤
│  SUSPICION QUEUE │  VIDEO PLAYER + OVERLAY                          │
│                  │                                                   │
│  ● 0.91  Cr+Hd   │  [video frame with trajectory overlays]          │
│    T214–T231     │  colored dots + ID labels                        │
│    tracks 2 & 5  │  flagged tracks: pulsing highlight ring          │
│    open_field    │  low-conf zones: semi-transparent red tint        │
│                  │                                                   │
│  ● 0.78  Pr+Hd   │  [scrub bar]  ◀ ▶  frame 214 / 3200             │
│    T445–T461     │                                                   │
│    tracks 1 & 3  ├───────────────────────────────────────────────────┤
│    region-2      │  TIMELINE PANEL                                   │
│                  │                                                   │
│  ○ 0.61  Pr      │  track 2  ━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━      │
│    T892–T910     │  track 5  ━━━━━━╋━━━━━━━━━━━━━━━━━━━━━━━━━      │
│    tracks 4 & 2  │           ↑ flagged zone highlighted             │
│                  │  [all tracks, click to split, drag to reassign]  │
│  [show more ↓]   │                                                   │
└──────────────────┴───────────────────────────────────────────────────┘
```

**Top bar — Session navigator:**
- `[⊕ load]`: pick single video+CSV pair, or load a `.txt` file (one video path per line; matching CSV auto-discovered)
- `[◀ prev]` / `[next ▶]`: step through loaded list
- Session counter `(2/8)` shows position
- Unsaved changes prompt on navigation

**Content tabs (extensible):**
- `Swap Review` — ranked queue + frame-picker + identity assignment (v1)
- `Merge Review` — future: review over-split fragments
- `Manual Edit` — future: free-form editing without a queue
- `[...]` — additional tabs without window restructuring

Video player and timeline panel are **shared across all tabs**.

---

### Swap Suspicion Scoring

**`SwapSuspicionEvent` dataclass:**
```python
@dataclass
class SwapSuspicionEvent:
    track_a: int           # TrajectoryID
    track_b: int           # TrajectoryID (None for single-track anomaly)
    frame_peak: int        # Frame of highest suspicion
    frame_range: tuple     # (start, end) for clip extraction
    score: float           # 0–1
    signals: list[str]     # signals fired: 'Cr', 'Pr', 'Hd', 'Pq'
    region_label: str      # 'open_field' or 'region-N'
    region_boundary: bool  # True if event is at region edge
```

**Signals:**

| Signal | Abbrev | Description | Weight |
|---|---|---|---|
| Crossing / position exchange | `Cr` | Tracks swap relative position within T frames | High |
| Close approach + separation | `Pr` | Tracks come within D_close px then diverge, no exchange required | ~40–50% of Cr |
| Heading discontinuity | `Hd` | Sudden heading reversal (>120°) coinciding with nearby track | Medium |
| Pose quality drop | `Pq` | PoseQualityScore z-score drop near crossing event; gated on pose availability | Amplifier |
| Confidence region discount | — | Event inside low-confidence zone → score reduced | Negative |
| Region boundary bonus | — | Event at zone boundary (entering/leaving crowd) → score increased | Positive |

**Combined score:**
```
score = w1a * crossing_signal
      + w1b * proximity_signal
      + w2  * heading_discontinuity
      + w3  * pose_quality_drop        # 0 if no pose
      - w4  * confidence_region_discount
      + w5  * region_boundary_bonus
```

Normalised 0–1. Events below ~0.15 hidden by default, surfaced via "Show more."

**Queue behaviour:**
- Scored events shown highest-first (most obvious swaps)
- "Show more" button at bottom expands with next threshold tier
- User works until satisfied and stops — no forced completion
- Scoring runs in background `QThread`, queue populates progressively

---

### Review Workflow (Swap Review tab)

**Per-event flow:**

```
Click queue card
      ↓
Video seeks to frame_peak; flagged tracks highlighted
      ↓
User: [Review]  [Skip]  [Mark OK]
      ↓ Review
FramePickerDialog
  Cropped clip of the two tracks ±margin, in-memory frame cache
  User scrubs, clicks exact split frame N
      ↓
IdentityAssignmentDialog
  Before: thumbnail grid of frames N-T ... N
  After:  thumbnail grid of frames N ... N+T
  User clicks to assign "track A before = track B after"
      ↓
correction_writer.py:
  1. Break track A at frame N → two new TrajectoryID segments
  2. Break track B at frame N → two new TrajectoryID segments
  3. Swap post-split segment IDs per user assignment
  4. Re-interpolate any gaps created
  5. Atomic write (_proofread.csv.tmp → rename)
      ↓
Queue card checkmarked; timeline panel refreshes
```

**Skip:** card stays in queue unresolved
**Mark OK:** written to `_proofread_dismissed.json`, never re-surfaced

**Manual timeline edits:**
- Click track bar at frame → Force split (skips FramePickerDialog, frame already known) → IdentityAssignmentDialog
- Drag segment to another track row → reassign identity
- Right-click → context menu: "Force split here", "Mark as reviewed", "Ignore segment"

---

### Session Persistence

| File | Contents | Mutability |
|---|---|---|
| `<name>_proofread.csv` | Corrected trajectory data | Edited in place by MAT-afterhours |
| `<name>_proofread_dismissed.json` | Events marked OK | Written by MAT-afterhours |
| `<name>_confidence_regions.json` | Density region definitions | Read-only (from MAT) |
| `<name>_confidence_map.mp4` | Diagnostic overlay video | Read-only (from MAT) |

On session re-open:
- Dismissed events excluded from queue
- Already-corrected segments shown in distinct timeline colour
- Scoring re-runs only on uncorrected regions

---

## Architecture Boundaries

- `afterhours/core/` modules are pure Python — no Qt dependencies, fully testable
- `afterhours/gui/` imports from `afterhours/core/` and `multi_tracker.core.*`
- `confidence_density.py` is called from MAT's existing post-processing pipeline; `afterhours/` does not import from `gui/`
- Correction writer edits only `_proofread.csv` — never the original processed CSV
