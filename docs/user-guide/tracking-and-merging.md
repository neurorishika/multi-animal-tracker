# Tracking and Merging

## Core Components

- Kalman prediction (`multi_tracker.core.filters.kalman`)
- Assignment (`multi_tracker.core.assigners.hungarian`)
- Worker orchestration (`multi_tracker.core.tracking.worker`)

## What the Main Tracking Controls Mean

- `MAX_TARGETS`: hard ceiling on concurrently tracked animals.
- `MAX_ASSIGNMENT_DISTANCE_MULTIPLIER`: geometric gate for detection-to-track pairing.
- `LOST_FRAMES_THRESHOLD`: tolerance before a track is terminated.
- `MIN_RESPAWN_DISTANCE_MULTIPLIER`: avoids immediate duplicate respawns near active tracks.

## Bidirectional Tracking and Merge

### Why It Exists

Forward-only runs can produce identity swaps in crossings/occlusions. Backward pass provides a second trajectory hypothesis.

### Merge Logic (User View)

- Compare forward and backward trajectories.
- Resolve conflicts using continuity and agreement metrics.
- Keep consistent segments and flag weak links.

### Tradeoff

- Better identity continuity at the cost of longer runtime.

## Typical Tuning Path

1. Set realistic `reference_body_size`.
2. Start with default Kalman/process/measurement noise.
3. Increase assignment gate only if consistent misses occur.
4. Adjust lifecycle thresholds to reduce fragmentation without over-persisting bad tracks.
