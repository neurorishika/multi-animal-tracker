# Fragment Merge Proofreading — Two-Phase Afterhours Plan

**Date:** 2026-03-11
**Branch:** `mat-pose-integration`
**Author:** Claude Code

---

## Critical Assessment

The core intuition is **correct**: the current afterhours pipeline chokes when
fragmentation creates dozens of short-lived track IDs. The swap detector produces
O(n²) pairwise events over n tracks in a temporal window, so 30 fragments in a
crowded region can produce ~400+ spurious events that drown real swaps. Merging
fragments first reduces n and dramatically improves the signal-to-noise ratio
of the swap detector.

### Why this works

1. **Fragment merging is a much easier decision** than swap resolution. A human
   can tell whether two adjacent-in-time trajectory segments belong to the same
   animal from a short clip of each. The merge decision is visually binary:
   "same animal" or "different animal." Swap decisions require understanding
   the *crossing dynamics* of two animals simultaneously.

2. **Merge decisions are composable.** Merging A→B then B→C is equivalent to
   merging {A,B,C} atomically. This means greedy sequential merging is safe —
   no need for global optimization.

3. **The existing infrastructure supports it.** `CorrectionWriter.merge_fragments()`
   already relabels IDs atomically, the `TrackEditorModel` handles fragment
   reassignment, and the `EventScorer._detect_fragmentation()` already computes
   the candidate graph with distance + gap scoring.

### One concern addressed: "recent past" double-detection overlap

The proposal mentions considering fragments born in the "recent past" (before a
track dies) to handle double detections. This is important because a common
failure mode is:

```
Track A: ───────────────────────────┤ (dies frame 500)
Track B:                        ├────────────── (born frame 495, overlaps 5 frames)
```

The current fragmentation detector requires `gap > 0` (strictly sequential). We
need to relax this to allow **small overlaps** (negative gap, e.g., -10 to 0
frames). The scoring should penalize overlapping hypotheses since they require
the merge to also delete the overlapping segment of the less-confident track.
This is a real but tractable extension.

---

## Architecture Overview

### Current Pipeline (Phase 2 only)

```
Load video + CSV
        │
        ▼
EventScorer.score_all()  →  Suspicion Queue (all 6 detectors)
        │
        ▼
User clicks event → TrackEditorDialog → Apply → local rescore
```

### Proposed Pipeline (Phase 1 + Phase 2)

```
Load video + CSV
        │
        ▼
┌──────────────────────────────────────────────────┐
│  PHASE 1: Fragment Merge Wizard                  │
│                                                  │
│  Build merge candidate graph (born-after-death)  │
│  For each dying track, rank merge hypotheses     │
│  Present synchronized video comparison           │
│  User picks: merge / next hypothesis / skip      │
│  Apply merges incrementally                      │
│  Repeat until no more mergeable fragments        │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  PHASE 2: Swap Proofreading (unchanged)          │
│                                                  │
│  EventScorer.score_all() on merged data          │
│  Suspicion queue → TrackEditorDialog             │
└──────────────────────────────────────────────────┘
```

---

## Data Model: Merge Candidate Graph

### Node definition

Each node is a trajectory segment (track ID) with its lifespan:

```python
@dataclass
class TrackSegment:
    track_id: int
    frame_birth: int        # First active frame
    frame_death: int        # Last active frame
    pos_birth: Tuple[float, float]   # (x, y) at birth
    pos_death: Tuple[float, float]   # (x, y) at death
    heading_birth: float    # θ at birth (from Kalman or finite diff)
    heading_death: float    # θ at death
    vel_death: Tuple[float, float]   # (vx, vy) at death (for prediction)
    n_active_frames: int    # Number of non-NaN frames
    is_alive_at_end: bool   # True if track extends to last video frame
```

### Edge definition

A directed edge A→B means "B is a candidate continuation of A":

```python
@dataclass
class MergeCandidate:
    source_id: int          # Dying track
    target_id: int          # Candidate continuation
    gap_frames: int         # target.birth - source.death (can be negative for overlap)
    spatial_dist: float     # Euclidean distance at junction
    predicted_dist: float   # Distance from Kalman-predicted position to target birth
    heading_agreement: float  # cos(θ_death_A - θ_birth_B), [−1, 1]
    overlap_frames: int     # max(0, -gap_frames)
    score: float            # Composite feasibility score ∈ [0, 1]
```

### Scoring formula

```
score = w_dist * (1 - min(spatial_dist / MAX_DIST, 1))
      + w_pred * (1 - min(predicted_dist / MAX_PRED_DIST, 1))
      + w_gap  * (1 - min(abs(gap_frames) / MAX_GAP, 1))
      + w_head * max(0, heading_agreement)
      - w_overlap * (overlap_frames / MAX_OVERLAP)  # penalty for overlaps
```

Default weights (tunable):
```python
w_dist = 0.35
w_pred = 0.25    # Kalman extrapolation match — strong signal
w_gap  = 0.15
w_head = 0.15
w_overlap = 0.10
MAX_DIST = 100.0       # px
MAX_PRED_DIST = 60.0   # px (tighter — prediction should be accurate)
MAX_GAP = 30           # frames
MAX_OVERLAP = 10       # frames (overlap beyond this is rejected)
```

### Candidate generation algorithm

```
For each track A that dies before the last frame:
    For each track B born within [A.death - MAX_OVERLAP, A.death + MAX_GAP]:
        if B == A: skip
        if B.birth > A.death + MAX_GAP: skip
        if B is already merged into A's chain: skip
        Compute spatial_dist, predicted_dist, heading_agreement
        if spatial_dist > MAX_DIST: skip
        Compute score
        if score > MIN_THRESHOLD: add edge A→B
    Sort candidates by score descending
```

The candidates for each dying track form a ranked list. The user sees #1 first
and can request the next hypothesis.

---

## UI: Merge Comparison Viewer

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Fragment Merge Wizard          Track 7 → ? (3 of 41)       │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  HYPOTHESIS A     │  │  HYPOTHESIS B     │                │
│  │  Merge T7 → T12   │  │  Merge T7 → T19   │                │
│  │  Score: 0.87       │  │  Score: 0.64       │                │
│  │                    │  │                    │                │
│  │  [Video crop]      │  │  [Video crop]      │                │
│  │  T7 trail shown    │  │  T7 trail shown    │                │
│  │  T12 trail shown   │  │  T19 trail shown   │                │
│  │                    │  │                    │                │
│  │  gap: 3 frames     │  │  gap: 8 frames     │                │
│  │  dist: 12.4 px     │  │  dist: 34.1 px     │                │
│  └──────────────────┘  └──────────────────┘                 │
│                                                              │
│  Playback: ◄ [|◄] [▶/❚❚] [►|] ►    Frame: 497 / 505       │
│            ═══════════●════════                              │
│                                                              │
│  Timeline:  T7  ━━━━━━━━━━━━━━━━━┤                          │
│             T12            ├━━━━━━━━━━━━━━━                  │
│             T19                 ├━━━━━━━━━━                  │
│                                                              │
│  [← Prev Hypothesis]  [Accept A]  [Accept B]  [Next →]      │
│                        [No Good Merge — Skip Track]          │
│  ─────────────────────────────────────────────────            │
│  Progress: ████████░░░░░░░░░░  12 / 41 tracks reviewed      │
└─────────────────────────────────────────────────────────────┘
```

### Key UI behaviors

1. **Synchronized playback.** All hypothesis panels share a single transport
   (play/pause/seek/scroll). Frame position is synced. Each panel shows the
   zoomed crop around the junction point (death of source track ± context).

2. **Shared zoom and pan.** Mouse wheel or pinch zooms all panels identically.
   Pan drags are mirrored. This ensures fair visual comparison.

3. **Trajectory overlay.** The source track is drawn in one color, each candidate
   continuation in a distinct color. A dashed line or arrow highlights the
   "gap" segment (extrapolated position).

4. **Variable hypothesis count.** Show up to 3 hypotheses side-by-side (the
   top-scored candidates). "Next →" loads the next batch if more exist.
   "← Prev" goes back. If only 1 candidate exists, show single panel.

5. **Keyboard shortcuts.** `1`/`2`/`3` accept hypothesis A/B/C. `N` = next batch.
   `S` = skip. `Space` = play/pause. `←`/`→` = frame step.

6. **Progressive application.** Each accepted merge is applied immediately via
   `CorrectionWriter.merge_fragments()`. The candidate graph is updated: the
   merged track's death position becomes the continuation's death, and new
   candidates from the continuation's death are added.

7. **Undo.** Each merge is pushed onto an undo stack scoped to the wizard
   session. `Ctrl+Z` reverts the last merge (re-splits the track).

### Frame window for crop

```
crop_start = source.frame_death - CONTEXT_BEFORE   # default 30 frames
crop_end   = target.frame_birth + CONTEXT_AFTER     # default 30 frames
```

The crop region is the bounding box of source + target positions within this
window, expanded by 80px margin (same logic as `TrackEditorDialog._compute_crop`).

---

## Implementation Plan

### Task 1: Merge candidate scoring module (core, no Qt)

**Files:**
- Create: `src/multi_tracker/afterhours/core/merge_candidates.py`
- Create: `tests/test_merge_candidates.py`

**Implement:**

```python
# merge_candidates.py

@dataclass
class TrackSegment:
    """Trajectory endpoint summary for merge candidate ranking."""
    track_id: int
    frame_birth: int
    frame_death: int
    pos_birth: Tuple[float, float]
    pos_death: Tuple[float, float]
    heading_birth: float
    heading_death: float
    vel_death: Tuple[float, float]
    n_active_frames: int
    is_alive_at_end: bool

@dataclass
class MergeCandidate:
    """Directed merge hypothesis: source → target."""
    source_id: int
    target_id: int
    gap_frames: int
    spatial_dist: float
    predicted_dist: float
    heading_agreement: float
    overlap_frames: int
    score: float

def extract_segments(df: pd.DataFrame, last_frame: int) -> List[TrackSegment]:
    """Build TrackSegment list from trajectory DataFrame."""
    ...

def build_candidates(
    segments: List[TrackSegment],
    *,
    max_gap: int = 30,
    max_overlap: int = 10,
    max_dist: float = 100.0,
    min_score: float = 0.15,
) -> Dict[int, List[MergeCandidate]]:
    """For each dying track, return ranked list of merge candidates."""
    ...

def update_after_merge(
    segments: List[TrackSegment],
    candidates: Dict[int, List[MergeCandidate]],
    source_id: int,
    target_id: int,
) -> Tuple[List[TrackSegment], Dict[int, List[MergeCandidate]]]:
    """Update segment list + candidate graph after accepting a merge."""
    ...
```

**Tests:**
- Fragmentation with 0-frame gap, 5-frame gap, 20-frame gap → correct scores
- Overlap case (gap = -3) → candidate generated with penalty
- Multiple candidates ranked by score descending
- update_after_merge collapses segments correctly
- Track alive at video end → no merge candidates generated for it
- Heading agreement signal: same direction vs opposite
- Kalman-predicted distance: close prediction → higher score

---

### Task 2: Kalman extrapolation utility — with sidecar upgrade path

**Files:**
- Modify: `src/multi_tracker/afterhours/core/merge_candidates.py`

The merge scorer needs to extrapolate "where would the dying track be N frames later?"

**Tier 1 — plain CV fallback** (always available, no sidecar):

```python
def predict_position(
    pos: Tuple[float, float],
    vel: Tuple[float, float],
    n_frames: int,
    damping: float = 0.95,
) -> Tuple[float, float]:
    """Fallback: constant-velocity extrapolation with damping."""
    x, y = pos
    vx, vy = vel
    for _ in range(n_frames):
        x += vx; y += vy
        vx *= damping; vy *= damping
    return (x, y)
```

**Tier 2 — proper Kalman extrapolation** (when terminal-state sidecar exists):

The tracking worker (see Phase 8 in the EM/IMM plan,
`docs/plans/2026-03-11-em-imm-motion-model-plan.md`) saves a
`{stem}_kalman_terminal_states.npz` sidecar alongside the output CSV. This
contains `X` (5D state) and `P` (5×5 covariance) for every dying track,
keyed by `TrajectoryID`.

When the sidecar is available:
```python
def kalman_predicted_position(
    state: TrackTerminalState,
    n_steps: int,
    kalman_params: Dict[str, Any],
) -> Tuple[Tuple[float, float], np.ndarray]:
    """Exact N-step Kalman extrapolation → (predicted_xy, P_2x2)."""
    ...
```

The 2D positional submatrix `P_pred[0:2, 0:2]` replaces the hard pixel
threshold with a **Mahalanobis distance gate**:

```
d_maha = sqrt(Δp^T @ P_pred_2x2^{-1} @ Δp)  <  3.0  (3-sigma gate)
```

This automatically accounts for: track speed (high vx/vy → wide forward
   uncertainty), track age (high P → more tolerance), and anisotropy from
   the Kalman noise model (longitudinal vs lateral). When EM/IMM improves
   the filter parameters, the cached states improve automatically — no
   changes needed in the merge scorer.

---

### Task 3: MergeWizardDialog — comparison viewer

**Files:**
- Create: `src/multi_tracker/afterhours/gui/dialogs/merge_wizard.py`
- Create: `tests/test_merge_wizard_model.py` (model-only, no Qt)

**Implement a `MergeWizardDialog(QDialog)` with:**

1. **Top status bar** — current track index, total remaining, progress
2. **Hypothesis panels** — up to 3 `InteractiveCanvas` widgets side-by-side,
   each rendering cropped video + trajectory overlay for one hypothesis
3. **Shared transport** — single slider + play/pause controlling all panels
4. **Stat badges** — per-hypothesis: score, gap, distance, heading
5. **Timeline strip** — miniature fragment timeline showing source + candidates
6. **Action buttons**: Accept 1/2/3, Next page, Skip, Undo
7. **FrameLoader** — reuse the existing `_FrameLoader` pattern from
   `track_editor_dialog.py`, but decode crops for multiple hypotheses

**Internal model:**
```python
class MergeWizardModel:
    """Non-Qt model holding wizard state."""

    def __init__(self, df, segments, candidates, writer):
        self._df = df
        self._segments = segments
        self._candidates = candidates  # Dict[source_id, List[MergeCandidate]]
        self._writer = writer
        self._merge_order: List[int] = []  # source_ids sorted by priority
        self._current_idx: int = 0
        self._hypothesis_page: int = 0
        self._undo_stack: List[Tuple[int, int, pd.DataFrame]] = []

    @property
    def current_source(self) -> TrackSegment: ...

    @property
    def current_hypotheses(self) -> List[MergeCandidate]: ...

    def accept(self, hypothesis_idx: int) -> None:
        """Apply merge, update graph, advance."""
        ...

    def skip(self) -> None:
        """Mark track as unresolvable, advance."""
        ...

    def undo(self) -> None:
        """Revert last merge, go back."""
        ...

    @property
    def is_finished(self) -> bool: ...

    @property
    def progress(self) -> Tuple[int, int]: ...
```

**Priority ordering:** Merge candidates are processed starting from the
**most confident** merge (highest top-candidate score per dying track). This
ensures the easiest decisions come first and each merge can simplify later ones.

---

### Task 4: Integrate Phase 1 into MainWindow

**Files:**
- Modify: `src/multi_tracker/afterhours/gui/main_window.py`

**Changes to `_open_current_session()`:**

After loading the CSV and before running the swap scorer, check whether
fragment merging should be offered:

```python
def _open_current_session(self) -> None:
    # ... existing CSV load ...

    segments = extract_segments(self._df, last_frame)
    candidates = build_candidates(segments)

    dying_count = sum(1 for s in segments if not s.is_alive_at_end)
    mergeable_count = sum(1 for cands in candidates.values() if cands)

    if mergeable_count > 0:
        reply = QMessageBox.question(
            self,
            "Fragment Merge Wizard",
            f"Found {mergeable_count} tracks with merge candidates "
            f"(out of {dying_count} ending tracks).\n\n"
            f"Run the merge wizard before swap proofreading?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._run_merge_wizard(segments, candidates)

    # ... existing scorer + queue logic ...
```

**New method:**

```python
def _run_merge_wizard(self, segments, candidates) -> None:
    dlg = MergeWizardDialog(
        video_path=self._video_path,
        df=self._df,
        segments=segments,
        candidates=candidates,
        writer=self._writer,
        parent=self,
    )
    dlg.exec()

    if dlg.merges_applied > 0:
        # Refresh DataFrame after merges
        self._df = self._writer.df
        self._player.load_trajectories(self._df)
        self._timeline.load_trajectories(self._df)
        self.statusBar().showMessage(
            f"Merge wizard: {dlg.merges_applied} fragments merged", 5000
        )
```

**Also add a toolbar/menu action** to re-run the merge wizard at any point
(in case the user wants to do another pass after some swap corrections create
new fragments).

---

### Task 5: Extend `_detect_fragmentation` to handle overlaps

**Files:**
- Modify: `src/multi_tracker/afterhours/core/event_scorer.py`
- Modify: `tests/test_event_scorer.py` (add overlap test cases)

The existing fragmentation detector requires `gap > 0`. Extend to allow
small negative gaps (overlaps):

```python
# In _detect_fragmentation:
gap = b_start - a_end
if -_FRAG_MAX_OVERLAP <= gap <= _FRAG_MAX_GAP:
    # For overlaps (gap < 0), use slightly different scoring
    overlap = max(0, -gap)
    effective_gap = abs(gap)
    ...
```

This also benefits the existing Phase 2 pipeline by surfacing double-detection
fragments that were previously invisible.

---

### Task 6: Shared video transport widget

**Files:**
- Create: `src/multi_tracker/afterhours/gui/widgets/synced_video_grid.py`

A reusable widget for synchronized multi-video comparison:

```python
class SyncedVideoGrid(QWidget):
    """Grid of InteractiveCanvas panels with shared playback controls."""

    def __init__(self, n_panels: int = 3, parent=None): ...

    def set_panel_data(self, idx: int, frames: np.ndarray,
                       overlay_fn: Callable): ...

    def seek(self, frame: int): ...

    # Signals
    frame_changed = Signal(int)
```

This isolates the synchronized playback logic so it can be reused in
future tools (e.g., multi-hypothesis identity comparison).

---

### Task 7: Re-run merge wizard button in queue panel

**Files:**
- Modify: `src/multi_tracker/afterhours/gui/widgets/suspicion_queue.py`
- Modify: `src/multi_tracker/afterhours/gui/main_window.py`

Add a "Merge Wizard" button to the suspicion queue header, adjacent to the
existing "Rescore All" button. This allows the user to re-enter Phase 1 after
performing some Phase 2 corrections (which may have created new fragments).

---

## Ordering and Dependencies

```
Task 1: merge_candidates.py (core model)
  │
  ├──► Task 2: Kalman extrapolation utility (extends Task 1)
  │
  ├──► Task 5: Extend fragmentation detector for overlaps
  │
  └──► Task 3: MergeWizardDialog (depends on Task 1 + Task 6)
         │
         └──► Task 6: SyncedVideoGrid widget (can be built in parallel)
                │
                └──► Task 4: MainWindow integration (depends on Task 3)
                       │
                       └──► Task 7: Re-run button (depends on Task 4)
```

**Suggested build order:** 1 → 2 → 5 (parallel with 6) → 3 → 4 → 7

---

## Edge Cases and Failure Modes

### 1. Merge chains that create impossibly long tracks

If an animal has 10 fragments across the video, merging all 10 sequentially is
correct. But the wizard should show the *accumulated* merged trajectory at each
step so the user can see the full history, not just the junction.

### 2. Branching: one track has multiple valid continuations

This genuinely happens when two animals are near each other and both create
fragments. The scoring uses heading + Kalman prediction to disambiguate,
and the human makes the final call. The side-by-side comparison makes this
easy.

### 3. Circular merges

A→B→A would create a loop. The candidate builder must exclude any track already
in the current merge chain. `update_after_merge` collapses A and B into a single
segment, preventing A from appearing as a candidate for itself.

### 4. Very long videos with thousands of fragments

Pre-compute all segments and candidates in a background thread (same
`_ScorerWorker` pattern). Show a progress bar. The O(n²) candidate generation
should be fast — spatial indexing (KD-tree on death/birth positions) can reduce
this to O(n log n) if needed, but unlikely necessary for < 10,000 tracks.

### 5. User fatigue

The wizard should show an estimated remaining count and allow the user to
"Skip All Remaining" to proceed directly to Phase 2. Any unskipped fragments
will appear as FRAGMENTATION events in the suspicion queue.

---

## Metrics and Success Criteria

1. **Fragment reduction ratio**: After Phase 1, measure
   `n_tracks_after / n_tracks_before`. Target: >30% reduction on fragmented
   videos (clips with >2× expected animal count in track IDs).

2. **Phase 2 event count reduction**: The swap scorer should produce
   significantly fewer events after Phase 1 merging. Target: >50% fewer events
   on fragmented videos.

3. **Decision speed**: Average merge decision should take <5 seconds per
   hypothesis (visual comparison is fast when crops are aligned and synced).

4. **Correctness**: Merged trajectories should not introduce position
   discontinuities > 2× the merge distance threshold.

---

## Not in Scope (Future Work)

- **Automatic merge without human review**: Could be done for very high-scoring
  candidates (>0.95), but violates the proofreading philosophy. Consider as a
  "trust threshold" option in a later version.

- **Appearance-based matching**: Using classification embeddings to verify
  identity continuity at merge junctions. Would boost accuracy but requires
  ClassKit integration. Natural follow-up once ClassKit embedding extraction
  is stable.

- **Pose-based matching**: Using skeleton continuity to validate merges.
  Requires the pose pipeline to be integrated. Natural follow-up after
  pose output is stabilized.
