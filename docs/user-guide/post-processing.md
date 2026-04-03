# Post-processing

Post-processing is where HYDRA Suite turns raw online trajectories into something you can trust for analysis.

The current implementation is deliberately conservative: it would rather keep two defensible fragments than one polished but incorrect identity trace.

## Why Post-processing Exists

Online tracking has to make decisions immediately. That is useful for visualization and throughput, but it also means difficult events can slip through:

- crossings,
- temporary occlusions,
- missing detections,
- abrupt detector failures,
- and short identity swaps.

Post-processing gets a second chance to inspect those events with more context.

## Current Processing Stages

### 1. Remove weak fragments

Very short trajectories are dropped using `MIN_TRAJECTORY_LENGTH`.

This removes:

- detector noise,
- unstable startup segments,
- and partial fragments that are too small to support reliable identity claims.

### 2. Split implausible motion

The pipeline can break a trajectory when motion becomes physically implausible.

Controls:

- `MAX_VELOCITY_BREAK`
- `MAX_VELOCITY_ZSCORE`
- `VELOCITY_ZSCORE_WINDOW`
- `VELOCITY_ZSCORE_MIN_VELOCITY`

Use the fixed threshold when you know the approximate speed envelope. Use the z-score breaker when failures look like sudden outliers relative to each trajectory's own history.

### 3. Split long occlusion runs

If a trajectory stays in `occluded` state for too long, the current implementation can split at that gap rather than pretending continuity survived a long missing period.

Control:

- `MAX_OCCLUSION_GAP`

### 4. Merge forward and backward hypotheses

If backward tracking was enabled, the post-processor compares forward and backward outputs and merges only the segments that genuinely agree.

Controls:

- `AGREEMENT_DISTANCE`
- `MIN_OVERLAP_FRAMES`

### 5. Stitch broken fragments

After merge resolution, nearby fragments can be stitched across short gaps when the geometry still makes sense.

This is a pragmatic recovery step for tracks broken by:

- quick turns,
- short detector dropouts,
- or ambiguity during close interactions.

### 6. Relink with motion and optional pose continuity

The current relinking step is stronger than simple gap stitching. It summarizes fragment endpoints and asks whether the later fragment could plausibly be the continuation of the earlier one.

It uses:

- motion extrapolation,
- heading compatibility,
- and pose-shape compatibility when pose data exists.

Controls:

- `ENABLE_TRACKLET_RELINKING`
- `RELINK_POSE_MAX_DISTANCE`
- `MAX_OCCLUSION_GAP`
- `MAX_VELOCITY_BREAK`

### 7. Interpolate short gaps

Interpolation is the last stage. It is not intended to rescue fundamentally bad tracking.

Controls:

- `INTERPOLATION_METHOD`
- `INTERPOLATION_MAX_GAP`

Available modes:

- `none`
- `linear`
- `cubic`
- `spline`

## Recommended Defaults

For most experiments, a safe default posture is:

- enable post-processing,
- keep merge thresholds conservative,
- enable relinking only when fragmenting is a real problem,
- use linear interpolation with a small max gap,
- and visually spot-check crossings before trusting derived statistics.

## When to Be More Conservative

Tighten the post-processing stack if:

- animals are visually similar and identity errors are costly,
- social interactions create frequent overlaps,
- or downstream metrics depend heavily on exact individual continuity.

## When to Be More Permissive

Relax it only when:

- detections are clean,
- crossings are rare,
- and fragmentation is the main failure mode.

## Related Reading

- [Tracking, Identity Continuity, and Merging](tracking-and-merging.md)
- [Tracking Algorithm Deep Dive](../developer-guide/tracking-algorithm-deep-dive.md)
