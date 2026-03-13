# Swap-Resistant Continuity Plan

**Date:** 2026-03-09
**Branch:** `mat-pose-integration`
**Status:** Proposed - replaces the earlier broad false-merge plan

---

## Why This Plan Exists

The current tracker is already reasonably good. The main problem is not general tracking collapse; it is **occasional identity swaps** that appear during close interactions, especially when:

- two animals walk together in a stable tandem-like configuration
- several animals cluster in a small region
- detections remain available but individual visibility flickers
- short occlusions or contact events create ambiguous continuity

That matters because the right fix is not to globally split more aggressively. A naive split-first policy would reduce some false merges, but it would also overfragment legitimate tandem walking and normal social clustering.

This rewrite therefore changes the emphasis:

- target **swap seams**, not all close-contact frames
- distinguish **structured proximity** from **identity ambiguity**
- treat cluster interiors as low-information regions where identity should often be deferred
- make re-assignment decisions mainly at **entry and exit boundaries**
- keep appearance as optional negative-only evidence, not a primary solution

---

## Executive Summary

The new plan is to make the pipeline **interaction-aware**.

Instead of treating all crowding as one generic risk signal, the post-processing path should classify each trajectory row into one of four regimes:

1. **Isolated**: continuity is trustworthy; normal motion-based tracking rules apply.
2. **Tandem**: two animals are close but moving coherently; continuity should usually be preserved, not split.
3. **Cluster**: local identity is genuinely ambiguous due to crowding, overlap, or visibility flicker; do not trust interior continuity too much.
4. **Recovery**: the first frames after a tandem or cluster dissolves; this is the highest-risk swap boundary and should receive the strongest checks.

This is the central design decision behind the entire plan.

---

## Audit Of The Previous Plan

The previous document contained several correct instincts, but it was too broad for the actual problem now facing this repository.

### What Was Right

- It correctly prioritised false merges over trajectory length.
- It correctly treated appearance as dangerous if used as positive merge evidence.
- It correctly identified ambiguity, uncertainty, and assignment quality as the signals that matter most.
- It correctly aimed to harden relinking instead of trusting it to solve everything.

### What Was Missing

The previous plan did not clearly separate three distinct cases:

1. **Animals are close, but identity is still stable.**
   Tandem walking often falls into this category.

2. **Animals are close, and identity is temporarily unknowable.**
   Dense clusters with flickering visibility fall into this category.

3. **Animals are leaving a close interaction, and continuity must be re-earned.**
   This is where many swaps actually happen.

Without that separation, the old plan risked turning all close contact into generic danger. That would be too blunt for ants and similar species, where proximity is normal biology.

### Main Conclusion From The Audit

The codebase should not ask, "Are animals close?" and then split more.

It should ask:

- "Is this a stable tandem?"
- "Is this a genuinely ambiguous cluster?"
- "Are we at the boundary where identities can swap?"

That is the basis of the new plan.

---

## Core Strategy

The repository should move to a **boundary-focused, interaction-aware continuity model**.

The practical policy is:

- preserve continuity in isolated motion
- preserve continuity in stable tandem motion unless there is strong contradictory evidence
- avoid making strong identity claims inside ambiguous clusters
- apply the strongest swap checks at cluster or tandem exit boundaries
- relink only when continuity is clearly re-established

This plan therefore aims to reduce ID swaps by changing **where** we spend conservatism:

- less aggression inside normal social contact
- much more aggression at the boundaries where identities separate again

---

## Design Principles

1. **Close proximity is not itself an error.**
   For social insects, proximity is common. The pipeline must model it, not penalise it blindly.

2. **Continuity inside an ambiguous cluster is provisional.**
   When visibility flickers and multiple animals overlap, the tracker should avoid overcommitting to identity.

3. **Swap prevention is a boundary problem first.**
   Most damaging merges happen when animals enter, cross within, or exit an interaction zone.

4. **Tandem and cluster are different states.**
   A two-animal coherent pair should not be treated the same as a multi-animal unstable cluster.

5. **Relinking must be precision-first.**
   Unmatched fragments are acceptable. A confident but wrong join is not.

6. **Appearance stays optional and negative-only.**
   If used at all, it should support vetoing suspicious continuity, never creating it.

---

## Current Code Reality

This plan is designed around the current architecture rather than an imagined rewrite.

Relevant existing hooks already exist in these files:

- `src/multi_tracker/core/tracking/worker.py`
- `src/multi_tracker/core/assigners/hungarian.py`
- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

Important current facts:

- final post-processing already flows through `process_trajectories(...)` and `process_trajectories_from_csv(...)`
- final pose-aware relinking is centralised in `MainWindow._relink_final_pose_augmented_csv(...)`
- relinking is currently greedy and based on motion plus optional pose in `relink_trajectories_with_pose(...)`
- confidence-style outputs already exist, including `AssignmentConfidence` and `PositionUncertainty`
- the optimiser already acknowledges that generic crowding penalties are often unhelpful for ants and similar datasets

This means the first implementation should extend the existing post-processing and relink path, not invent a parallel pipeline.

---

## Failure Model We Should Optimise For

We should explicitly optimise for this narrow failure mode:

- tracking is mostly correct during isolated motion
- the tracker enters a close-interaction window
- detections continue, but visibility flickers or assignments become low-margin
- continuity through that window is plausible but not trustworthy
- when animals separate, a wrong identity chain is preserved

That is the failure mode this plan is built to catch.

The plan is **not** designed to solve full-video re-identification, nor to prove identity across long unobserved gaps.

---

## New Operational Model

### Regime Labels

Each trajectory row should be annotated with an interaction regime derived in post-processing:

- `isolated`
- `tandem`
- `cluster`
- `recovery`

These labels should be pipeline-internal at first. They can be persisted later if they prove useful.

### Regime Definitions

#### `isolated`

Use when the animal is well-separated and assignment quality is normal.

Policy:

- normal motion and uncertainty checks apply
- false-merge scoring can be active
- continuity may be trusted if multiple signals agree

#### `tandem`

Use when two animals remain close over multiple frames but show structured, coherent movement rather than unstable mixing.

Typical clues:

- only two animals involved
- distance remains low but stable
- headings are broadly aligned
- both remain detectable most frames
- assignment confidence may soften slightly, but not collapse

Policy:

- do not split merely because the pair is close
- suppress generic crowding-based split pressure
- raise suspicion only if one of the animals flickers, swaps heading abruptly, or exits the pair with contradictory motion

#### `cluster`

Use when local identity is not reliable enough to defend frame-by-frame continuity.

Typical clues:

- three or more animals in the local component, or two animals with repeated visibility flicker and overlap
- low assignment margin or unstable assignment confidence
- repeated transitions among `active`, `occluded`, and `lost`
- uncertainty spikes during continued close contact

Policy:

- avoid using cluster interior motion anomalies as direct split triggers
- treat continuity through the interior as provisional
- shift decision weight toward cluster entry and exit boundaries

#### `recovery`

Use for the first short window after a tandem or cluster state ends.

Policy:

- this is the strongest anti-swap check region
- require continuity to be re-earned using position, velocity, heading, pose, and assignment evidence
- lower the threshold for boundary-focused split decisions here
- do not let generic relinking override a rejected recovery boundary

---

## Signals To Use

### Existing Signals We Should Leverage Better

- `X`, `Y`, `Theta`, `FrameID`, `State`
- `DetectionConfidence`
- `AssignmentConfidence`
- `PositionUncertainty`
- `DetectionID` for frame-local reconciliation only
- pose keypoints and pose quality when present

### New Signals Worth Adding First

- normalized assignment cost
- assignment margin where available
- Kalman innovation magnitude or a similar residual
- nearest-neighbor distance among active tracks
- local visibility flicker score over a short window
- interaction component size per frame

### Signals To Avoid Over-Trusting

- raw proximity alone
- motion spikes inside stable tandem movement
- motion spikes inside dense cluster interiors
- appearance similarity as positive identity evidence

---

## Proposed Implementation Plan

## Phase 0: Instrument The Real Swap Signals

### Phase 0 Objective

Persist the minimum additional diagnostics needed to tell normal close contact apart from actual identity danger.

### Phase 0 Changes

- Persist normalized assignment cost.
- Persist assignment margin if it is cheap to extract.
- Persist Kalman innovation magnitude or an equivalent update residual.
- Persist nearest-neighbor distance for active tracks.
- Add a lightweight visibility flicker metric computed over a short trailing window.

### Phase 0 Rationale

Today the pipeline has confidence-like fields, but not enough direct evidence about ambiguous handoffs. Without that, later split logic will remain heuristic in the wrong way.

### Phase 0 Primary Files

- `src/multi_tracker/core/tracking/worker.py`
- `src/multi_tracker/core/assigners/hungarian.py`
- `src/multi_tracker/core/post/processing.py`
- `docs/developer-guide/confidence-metrics.md`

### Phase 0 Deliverable

A CSV and post-processing input set that can measure instability explicitly instead of inferring it indirectly.

---

## Phase 1: Detect Interaction Regimes Instead Of Generic Crowding

### Phase 1 Objective

Add a dedicated regime detector that classifies rows into `isolated`, `tandem`, `cluster`, or `recovery`.

### Phase 1 Rationale

This is the most important architectural change in the plan. If the pipeline cannot distinguish tandem from cluster, the rest of the logic will stay too blunt.

### Phase 1 Implementation Sketch

Create a helper module that works on the final trajectory table and returns:

- a row-level `InteractionRegime`
- a per-frame interaction component identifier
- regime transition markers for entry and exit boundaries

Recommended helper file:

- `src/multi_tracker/core/post/interaction_events.py`

Recommended public entry point:

- `annotate_interaction_regimes(df, params)`

### Phase 1 Initial Heuristics

Classify a component as likely `tandem` when:

- exactly two tracks participate
- inter-animal distance stays below a threshold for several frames
- motion direction remains broadly aligned
- visibility remains mostly stable
- there is no repeated identity-style flicker

Classify a component as likely `cluster` when one or more of the following hold:

- three or more tracks are involved
- repeated visibility flicker occurs during continued proximity
- assignment confidence and margin are unstable for several frames
- uncertainty remains elevated inside the component

Classify `recovery` for the first short post-interaction window after `tandem` or `cluster` ends.

### Phase 1 Primary Files

- new: `src/multi_tracker/core/post/interaction_events.py`
- `src/multi_tracker/core/post/processing.py`

### Phase 1 Deliverable

A reusable state annotation layer that later split and relink logic can use.

---

## Phase 2: Replace Global Split Pressure With Boundary-Focused Break Logic

### Phase 2 Objective

Stop treating all ambiguous frames the same. Split mainly at the seams where an identity chain becomes undefendable.

### Phase 2 Rationale

The current risk is not that cluster interiors are always wrong. The risk is that the pipeline silently carries one identity through an ambiguous interaction and then keeps it after separation.

### Phase 2 Decision Policy By Regime

#### `isolated` Policy

Use the normal multi-cue break score.

Candidate cues:

- velocity z-score
- heading jump
- uncertainty spike
- assignment cost spike
- assignment margin collapse

#### `tandem` Policy

Reduce split sensitivity.

Only split if there is strong contradictory evidence such as:

- repeated visibility flicker affecting only one participant
- a sharp recovery-boundary mismatch when the pair separates
- a simultaneous motion and assignment anomaly that persists

#### `cluster` Policy

Suppress most interior split triggers.

Specifically:

- do not split on velocity anomaly alone
- do not split on generic proximity alone
- do not split on uncertainty alone if the component is still in active cluster state

Instead, record candidate concern points and defer actual identity decisions to entry and exit boundaries.

#### `recovery` Policy

Increase split sensitivity.

This is where a bad continuity chain should be broken if it cannot be defended.

### Phase 2 Recommended Helper

- `_compute_boundary_break_candidates(...)`

This helper should accept regime labels and use different thresholds by regime.

### Phase 2 Primary Files

- `src/multi_tracker/core/post/processing.py`
- optional new helper: `src/multi_tracker/core/post/boundary_breaks.py`

### Phase 2 Deliverable

A split pass that is less trigger-happy during normal social contact but much stricter at post-contact swap seams.

---

## Phase 3: Add Boundary Re-Anchoring For Interaction Exits

### Phase 3 Objective

When animals leave a tandem or cluster event, explicitly solve a small local continuity problem instead of relying on naive greedy fragment linking.

### Phase 3 Rationale

The current relinker in `processing.py` is greedy and works best when the fragments it sees are already clean. It should not bear the entire burden of disambiguating cluster exits.

### Phase 3 Strategy

For each interaction event boundary:

- collect short pre-boundary anchor summaries
- collect short post-boundary candidate summaries
- score plausible continuations using motion, heading, uncertainty, and pose if available
- accept only low-cost matches
- leave uncertain exits unmatched instead of forcing continuity

This should be a local mini-assignment, not a full-video identity solver.

### Phase 3 Recommended Module

- `src/multi_tracker/core/post/interaction_anchoring.py`

Recommended public entry point:

- `resolve_interaction_boundaries(df, regime_table, params)`

### Phase 3 Important Constraint

Do not allow the later general relinker to override an explicit rejection made here.

### Phase 3 Primary Files

- new: `src/multi_tracker/core/post/interaction_anchoring.py`
- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

### Phase 3 Deliverable

A precise, boundary-local defence against swaps at the exact point where animals separate.

---

## Phase 4: Harden General Relinking So It Cannot Undo Good Splits

### Phase 4 Objective

Keep the existing relinker conservative and make it subordinate to the new boundary logic.

### Phase 4 Changes

- add a hard reject threshold for motion-plus-pose relinking
- prevent relinking across recently rejected recovery boundaries
- penalise relinking over low-margin or high-uncertainty gaps
- prefer leaving fragments unmatched rather than greedily chaining them
- consider replacing greedy pairing with Hungarian matching only if the first hard-reject version is still too permissive

### Phase 4 Rationale

Boundary anchoring and general fragment relinking are different jobs. The first handles known high-risk seams. The second cleans up easy residual fragments.

### Phase 4 Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

### Phase 4 Deliverable

A relinker that preserves the gains from earlier split decisions instead of smoothing them away.

---

## Phase 5: Add Optional Population Guardrails

### Phase 5 Objective

Use known animal count as a validator when the experiment provides it.

### Phase 5 Rationale

For many colony or arena experiments, the number of animals is known. That is valuable as a guardrail, but it should not be the main anti-swap mechanism.

### Phase 5 Policy

- if `MAX_EXPECTED_ANIMALS` is configured, use it to flag impossible post-processing states
- use it after split, anchoring, and relink passes
- default to warning or flagging before any automatic enforcement

### Phase 5 Important Limitation

Population count does not tell us who is who. It only helps catch obviously invalid outcomes.

### Phase 5 Primary Files

- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/gui/main_window.py`

### Phase 5 Deliverable

A sanity layer that can suppress the most obvious oversplitting mistakes in count-known experiments.

---

## Phase 6: Optional Appearance Veto Near Recovery Boundaries

### Phase 6 Objective

Add appearance only as a narrow, negative-only cue near recovery seams after the structural fixes above are in place.

### Phase 6 Rationale

Appearance is still useful, but it should not be one of the first changes. The core problem here is state modelling, not missing embeddings.

### Phase 6 Recommended Scope

- run only near `recovery` windows or explicitly flagged boundary candidates
- use only as a veto for suspicious continuity
- never use to create a positive merge or relink decision
- keep it config-only and off by default in the first implementation

### Phase 6 Suggested Helpers

- `src/multi_tracker/core/post/appearance_signatures.py`
- `src/multi_tracker/core/post/appearance_changepoints.py`

### Phase 6 Deliverable

An optional last-layer veto for stubborn swap cases, without turning the tracker into an appearance-driven system.

---

## Recommended Implementation Order

1. Phase 0: instrumentation
2. Phase 1: interaction regime detection
3. Phase 2: boundary-focused break logic
4. Phase 3: interaction boundary anchoring
5. Phase 4: conservative relinking hardening
6. Phase 5: optional population guardrails
7. Phase 6: optional appearance veto

This order is intentional.

- It first improves observability.
- Then it adds the missing biological state model.
- Then it changes split logic.
- Only after that does it touch relinking and appearance.

---

## Concrete First Slice

The first implementation should stay narrow and measurable.

### First Slice Contents

1. Persist two or three missing diagnostics:
   - normalized assignment cost
   - assignment margin if cheap
   - nearest-neighbor distance
2. Add row-level interaction regime detection:
   - `isolated`
   - `tandem`
   - `cluster`
   - `recovery`
3. Add a boundary-focused split helper that:
   - is conservative in `tandem`
   - suppresses cluster-interior motion splits
   - is aggressive in `recovery`
4. Prevent general relinking from reconnecting fragments across a rejected recovery boundary.

### Why This Is The Right First Slice

It directly addresses the real failure mode without committing the project to appearance pipelines, global re-identification, or an oversized architecture refactor.

---

## Suggested Parameter Surface

These should remain config-only or advanced at first.

- `ENABLE_SWAP_RESISTANT_CONTINUITY`
- `INTERACTION_TANDEM_MAX_DISTANCE_MULTIPLIER`
- `INTERACTION_TANDEM_MIN_DURATION_FRAMES`
- `INTERACTION_CLUSTER_MIN_SIZE`
- `INTERACTION_CLUSTER_MIN_DURATION_FRAMES`
- `INTERACTION_RECOVERY_WINDOW_FRAMES`
- `BOUNDARY_BREAK_SCORE_THRESHOLD`
- `BOUNDARY_BREAK_RECOVERY_MULTIPLIER`
- `BOUNDARY_BREAK_TANDEM_MULTIPLIER`
- `BOUNDARY_BREAK_CLUSTER_INTERIOR_SUPPRESS`
- `ENABLE_INTERACTION_BOUNDARY_ANCHORING`
- `INTERACTION_ANCHOR_LOOKBACK_FRAMES`
- `INTERACTION_ANCHOR_LOOKAHEAD_FRAMES`
- `INTERACTION_ANCHOR_MAX_COST`
- `RELINK_RESPECT_REJECTED_BOUNDARIES`
- `MAX_EXPECTED_ANIMALS`
- `POPULATION_CEILING_ENFORCE_MODE`
- `ENABLE_APPEARANCE_BOUNDARY_VETO`

---

## Testing Plan

### New Tests

- `tests/test_post_processing_interaction_regimes.py`
- `tests/test_post_processing_boundary_breaks.py`
- `tests/test_post_processing_interaction_anchoring.py`
- `tests/test_tracklet_relinking_conservative.py`
- optional later: `tests/test_post_processing_appearance_changepoints.py`

### Required Test Cases

1. Stable isolated motion does not fragment.
2. Stable tandem walking does not split purely because of proximity.
3. A two-animal tandem that separates cleanly preserves identity through the boundary.
4. A dense cluster with visibility flicker does not create interior motion-based split spam.
5. A bad post-cluster continuity chain is split at recovery.
6. General relinking cannot rejoin fragments across an explicitly rejected recovery seam.
7. Population guardrails warn or flag impossible active counts when configured.

---

## Validation Metrics

The previous plan measured false merges broadly. This plan should measure boundary performance specifically.

Recommended metrics:

- ID swaps per 1000 frames
- swap rate at interaction exits
- fragmentation increase outside interaction windows
- fragmentation increase inside tandem windows
- relink precision on post-interaction fragments
- number of rejected recovery joins left unmatched
- regression rate on clean isolated clips

The critical acceptance criterion is:

**swap reduction must improve materially without causing tandem walking to explode into fragments.**

---

## Documentation Updates After Implementation

- update `docs/developer-guide/confidence-metrics.md` for any new persisted diagnostics
- add a short developer note describing the four interaction regimes
- document that crowding is no longer treated as a single generic post-processing concept
- document that appearance, if enabled, is a recovery-boundary veto only

---

## Open Questions

1. Should `InteractionRegime` remain pipeline-internal, or be persisted to the final CSV for downstream review?
2. What is the cleanest heuristic boundary between `tandem` and small unstable two-animal `cluster` events?
3. Is assignment margin cheap enough to log on all paths, or should it remain optional?
4. Should interaction boundary anchoring live as a separate module, or as a tightly gated mode of post-processing relinking?
5. For species with strong pose signals, should pose be mandatory at recovery boundaries when available?

---

## Final Recommendation

The repository does not need a broad "more splitting everywhere" strategy.

It needs a **swap-resistant continuity strategy** built around the fact that close contact has multiple meanings:

- isolated motion is easy
- tandem motion is structured and often legitimate
- cluster interiors are ambiguous and should be treated cautiously
- exit boundaries are where identity must be re-earned

That is the right abstraction for this codebase, for this species mix, and for the specific problem of occasional ID swaps in otherwise decent tracking.
