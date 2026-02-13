# Post-Processing Fixture Schema

This directory contains deterministic fixture inputs and golden outputs for strict
equivalence testing of trajectory post-processing behavior.

## Layout

- `resolve/<case_name>/`
- `interpolate/<case_name>/`
- `benchmarks/baseline_runtime_ms.json`

Each `resolve` case contains:

- `input_forward.csv`: forward trajectories
- `input_backward.csv`: backward trajectories
- `params.json`: resolve parameters
- `expected.csv`: golden resolved output

Each `interpolate` case contains:

- `input.csv`: trajectory rows before interpolation
- `params.json`: interpolation params
- `expected.csv`: golden interpolated output

## Required Columns

Core columns:

- `TrajectoryID` (int)
- `FrameID` (int)
- `X` (float)
- `Y` (float)
- `Theta` (float, radians)

Optional columns preserved when present:

- `State` (`active`/`occluded`/`lost`)
- `DetectionID` (int)
- `DetectionConfidence` (float)
- `AssignmentConfidence` (float)
- `PositionUncertainty` (float)

## Canonical Ordering Contract

Golden files are canonicalized before comparison:

- Rows sorted by `TrajectoryID`, then `FrameID` (stable sort)
- Integer columns normalized to nullable integer type
- Float columns compared with strict tolerance (`abs=1e-9`, `rel=1e-9`)
- Column order normalized with core columns first

## Regeneration

Regenerate all fixture goldens with:

```bash
PYTHONPATH=src python tests/fixtures/postproc/generate_baselines.py
```

This should only be done intentionally when behavior changes are explicitly approved.
