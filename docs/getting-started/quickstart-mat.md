# Quickstart (MAT)

## Launch

```bash
mat
```

The main window opens maximized and exposes the tabbed workflow.

## Minimal End-to-End Run

1. **Setup tab**:
- Select input video.
- Select CSV output path.
- Optionally choose video output path.
2. **Detection tab**:
- Choose `background_subtraction` for moving animals on stable backgrounds.
- Choose `yolo_obb` for harder scenes or stationary animals.
3. **Tracking tab**:
- Set `MAX_TARGETS`.
- Keep defaults first for Kalman and assignment.
4. **Processing tab**:
- Keep post-processing enabled.
- Choose interpolation only if short gaps are expected.
5. Click **Start Tracking**.

## Important First-Run Checks

- ROI is set correctly.
- `reference_body_size` is realistic.
- Detection preview shows one detection per target on representative frames.

## Outputs

- Main CSV trajectory file
- Optional forward/backward/intermediate outputs
- Optional rendered output video
- Optional detection cache (`.npz`) when caching is enabled
