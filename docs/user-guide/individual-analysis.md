# Individual Analysis

Individual analysis tools extract per-track crops and metadata for identity-focused workflows.

## What It Provides

- Crop extraction around detections/trajectories.
- Identity-oriented dataset output structure.
- Optional color/marker-based workflows depending on method configuration.

## Main Controls

- `ENABLE_IDENTITY_ANALYSIS`
- `IDENTITY_METHOD`
- Crop size/padding constraints
- Output format and destination options

## Feature Meaning

- Larger crop padding improves context but increases storage and may include neighbors.
- Smaller crops are efficient but risk truncation near boundaries.
- Method choice should reflect your marker protocol (none/color/apriltag/custom).

## Typical Uses

- Build identity classifier training sets.
- Diagnose identity-switch failure regions.
- Export curated individual-level clips/crops.
