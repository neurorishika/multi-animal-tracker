# Extending Identity

## Main Extension Surface

`multi_tracker.core.identity.analysis`

## Recommended Pattern

1. Extend identity processor logic (classification or marker interpretation).
2. Keep crop extraction metadata consistent.
3. Preserve output contract expected by GUI and data export routines.

## Validation Checklist

- Robust near frame boundaries.
- Handles missing/low-quality detections.
- Produces deterministic outputs for reproducibility-sensitive workflows.
