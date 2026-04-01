# Architecture

## System Layers

- `multi_tracker.mat`: MAT launcher, GUI, dialogs, widgets.
- `multi_tracker.posekit`: pose-labeling application and related dialogs/inference flows.
- `multi_tracker.classkit`: classification/embedding toolkit.
- `multi_tracker.afterhours`: interactive proofreading.
- `multi_tracker.datasieve`: data sieve tool.
- `multi_tracker.integrations`: external tool bridges (SLEAP, X-AnyLabeling).
- `multi_tracker.core`: detection, filtering, assignment, post-processing, worker orchestration.
- `multi_tracker.runtime`: compute runtime selection and GPU utilities.
- `multi_tracker.data`: CSV/cache/dataset generation and merging.
- `multi_tracker.training`: dataset builders, training runner, registry, model publishing.
- `multi_tracker.utils`: shared helpers (GPU detection, geometry, image processing, batching, prefetch).
- `multi_tracker.paths`: central path resolution — bundled assets via `importlib.resources`, user dirs via `platformdirs`.
- `multi_tracker.resources`: bundled read-only assets (brand icons, default configs, skeletons).

## Design Intent

- Keep high-throughput frame processing in worker-style components.
- Keep GUI orchestration separated from algorithm implementations.
- Keep data export/cache utilities outside core tracking logic.

## Runtime Entry Paths

- MAT: `multi_tracker.mat.app.launcher:main`
- PoseKit: `multi_tracker.posekit.ui.main:main`
- DataSieve: `multi_tracker.datasieve.gui:main`
- ClassKit: `multi_tracker.classkit.app:main`
- Afterhours: `multi_tracker.afterhours.app:main`

## Key Operational Boundaries

- Core tracking should not depend on GUI widget internals.
- Data layer should stay reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow but shared environment/tooling.
- All path resolution goes through `multi_tracker.paths`. No module uses `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are in `multi_tracker.resources` and accessed via `importlib.resources`.
- User-writable data (models, training runs, config) lives in platform directories, never inside the installed package.
