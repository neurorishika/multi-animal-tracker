# End-to-End Workflow

## TrackerKit Workflow

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

## ClassKit Workflow

1. Create or open a project with source image directories.
2. Ingest and embed crops using a backbone model.
3. Cluster embeddings and visualize with UMAP.
4. Label identity classes manually or via AprilTag auto-labeling.
5. Train a classification head and evaluate results.
6. Export labeled datasets for downstream use.

## DetectKit Workflow

1. Curate detection training datasets from TrackerKit exports.
2. Configure YOLO training parameters.
3. Launch training and monitor loss curves.
4. Evaluate model performance on validation sets.

## FilterKit Workflow

1. Load a dataset directory.
2. Apply filtering criteria (quality, diversity, metadata).
3. Export the filtered subset for training or analysis.

## RefineKit Workflow

1. Load tracked trajectories and source video.
2. Review flagged suspicious segments in the suspicion queue.
3. Correct identity assignments using the interactive canvas.
4. Export refined trajectories.
