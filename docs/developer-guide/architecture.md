# Architecture

## System Layers

- `multi_tracker.app`: launcher/CLI/bootstrap.
- `multi_tracker.gui`: tracking GUI, dialogs, widgets.
- `multi_tracker.core`: detection, filtering, assignment, post-processing, worker orchestration.
- `multi_tracker.data`: CSV/cache/dataset generation and merging.
- `multi_tracker.posekit`: pose-labeling application and related dialogs/inference flows.
- `multi_tracker.utils`: shared helpers (GPU detection, geometry, image processing, batching, prefetch).

## Design Intent

- Keep high-throughput frame processing in worker-style components.
- Keep GUI orchestration separated from algorithm implementations.
- Keep data export/cache utilities outside core tracking logic.

## Runtime Entry Paths

- MAT: `multi_tracker.app.launcher:main`
- PoseKit: `multi_tracker.posekit.pose_label:main`

## Key Operational Boundaries

- Core tracking should not depend on GUI widget internals.
- Data layer should stay reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow but shared environment/tooling.
