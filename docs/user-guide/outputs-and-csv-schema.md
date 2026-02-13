# Outputs and CSV Schema

## Core Files

- **Primary CSV**: final trajectories for downstream analysis.
- **Intermediate CSVs**: forward/backward passes when enabled.
- **Detection cache** (`.npz`): optional reuse for backward/iterative runs.
- **Rendered video**: optional visualization output.

## Common CSV Columns

The exact schema can vary by enabled features, but includes:

- `TrackID`: active slot identifier.
- `TrajectoryID`: persistent trajectory identity.
- `FrameID`: source frame index.
- `X`, `Y`: position in pixel coordinates.
- `Theta`: orientation (radians).
- `State`: active/occluded/lost lifecycle state.

Optional confidence-related fields (if enabled):

- `DetectionConfidence`
- `AssignmentConfidence`
- `PositionUncertainty`

## Interpreting Output Quality

- Frequent short trajectories usually indicate weak detection or overly strict lifecycle limits.
- Large coordinate jumps suggest assignment gate or detector noise issues.
- High uncertainty with low confidence signals difficult frames worth relabeling or model retraining.

## Best Practices

- Keep raw/intermediate CSVs for debugging if experimenting with thresholds.
- Use final merged CSV for analysis.
- Version output folders by date/model/parameter preset for reproducibility.
