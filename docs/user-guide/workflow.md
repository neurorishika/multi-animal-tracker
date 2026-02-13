# End-to-End Workflow

## MAT Workflow

1. **Load video and outputs**
- Set input video, CSV output, optional rendered video output.
2. **Calibrate detection**
- Pick detection mode.
- Use preview/test detection before full run.
3. **Configure tracking**
- Set `MAX_TARGETS`, assignment distance, track lifecycle thresholds.
4. **Run forward/backward tracking**
- Enable backward pass for better conflict resolution.
5. **Post-process and export**
- Resolve identities, interpolate gaps as needed.
- Save final CSV and optional diagnostics.

## PoseKit Workflow

1. Load image set and project settings.
2. Label or refine keypoints frame-by-frame.
3. Use tools (smart select, metadata tags, split generation).
4. Export/prepare training-ready datasets.

## Decision Points That Matter Most

- **Detection mode** affects raw input quality to tracker.
- **Reference body size** scales multiple heuristics.
- **Post-processing** can fix or amplify detection mistakes depending on thresholds.

## Failure Pattern Checklist

- If targets merge often: tighten morphology and assignment distance.
- If tracks fragment: increase recovery/lost frame thresholds and validate detection confidence.
- If runtime is slow: reduce resize factor, disable non-critical overlays/histograms, verify GPU backend.
