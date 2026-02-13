# Data Flow

## MAT Pipeline

1. Video frames are read (optionally prefetched).
2. Detector generates per-frame measurements.
3. Kalman prediction and assignment update track state.
4. Worker emits frame/status/metrics to GUI.
5. Trajectories are written to CSV.
6. Optional backward pass reuses detection cache.
7. Post-processing resolves and interpolates trajectories.

## Key Data Artifacts

- **Measurements**: detector outputs including center, orientation, size cues.
- **Track State**: predicted/corrected state vectors and covariance.
- **Cache**: `.npz` detection cache for repeatability and performance.
- **CSV**: final analysis artifact for downstream pipelines.

## PoseKit Pipeline

1. Image set + project metadata loaded.
2. Annotation state edited in UI.
3. Labels persisted to YOLO pose format.
4. Optional model-assisted inference and split-generation steps create derived artifacts.
