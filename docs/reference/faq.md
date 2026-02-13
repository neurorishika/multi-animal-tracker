# FAQ

## Which command should I use for tracking?

Use `mat` (shortcut) or `multi-animal-tracker`.

## Which command should I use for pose labeling?

Use `posekit-labeler` (canonical) or `pose`.

## Should I use background subtraction or YOLO OBB?

- Background subtraction for stable scenes with clear motion contrast.
- YOLO OBB for cluttered scenes or stationary targets.

## Why are tracks fragmented?

Common causes are strict assignment/lifecycle thresholds, low detector recall, or calibration mismatch in `reference_body_size`.

## How do I improve runtime speed?

Lower `resize_factor`, reduce non-essential visualization, and verify GPU backend/device selection.
