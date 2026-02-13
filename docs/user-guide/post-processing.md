# Post-processing

Post-processing refines raw trajectories after tracking.

## Main Operations

- Remove short/noisy fragments.
- Split trajectories at implausible jumps.
- Resolve forward/backward trajectory conflicts.
- Interpolate short gaps.

## Key Parameters and Meaning

- `ENABLE_POSTPROCESSING`: master switch.
- `MIN_TRAJECTORY_LENGTH`: removes short, low-value fragments.
- `MAX_VELOCITY_BREAK`: controls where a trajectory should be broken.
- `MAX_OCCLUSION_GAP`: limit for tolerated missing spans.
- `INTERPOLATION_METHOD`, `INTERPOLATION_MAX_GAP`: gap-fill strategy and scope.

## Tradeoffs

- Aggressive cleanup can remove valid but unusual behavior.
- Too permissive cleanup can retain identity switches and noise.
- Interpolation improves continuity but should not hide long detector failures.

## Recommended Default Strategy

- Keep post-processing enabled.
- Use conservative interpolation (`Linear`, small max gap).
- Validate final CSV with quick visual spot-check on crossings and dense regions.
