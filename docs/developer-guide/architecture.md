# Architecture

## System Layers

- `multi_tracker.mat`: MAT launcher, GUI, dialogs, widgets.
- `multi_tracker.posekit`: pose-labeling application and related dialogs/inference flows.
- `multi_tracker.classkit`: classification/embedding toolkit.
- `multi_tracker.refinekit`: interactive proofreading.
- `multi_tracker.filterkit`: data sieve tool.
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
- FilterKit: `multi_tracker.filterkit.gui:main`
- ClassKit: `multi_tracker.classkit.app:main`
- RefineKit: `multi_tracker.refinekit.app:main`

## Key Operational Boundaries

- Core tracking should not depend on GUI widget internals.
- Data layer should stay reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow but shared environment/tooling.
- All path resolution goes through `multi_tracker.paths`. No module uses `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are in `multi_tracker.resources` and accessed via `importlib.resources`.
- User-writable data (models, training runs, config) lives in platform directories, never inside the installed package.
- Users can override directories with `MAT_DATA_DIR` and `MAT_CONFIG_DIR` environment variables.

## Data and Config Directories

All apps share paths via `multi_tracker.paths`. This module resolves user-writable directories (models, training, presets, skeletons) and bundled read-only assets (brand icons, default configs).

**Key functions:**

| Function | Returns |
|----------|---------|
| `get_models_dir()` | `<data>/models/` — trained model storage |
| `get_training_runs_dir()` | `<data>/training/runs/` — training run registry |
| `get_training_workspace_dir(subdir)` | `<data>/training/<subdir>/` — training workspace |
| `get_presets_dir()` | `<config>/presets/` — config presets (seeded from bundled defaults) |
| `get_skeleton_dir()` | `<config>/skeletons/` — skeleton definitions (seeded from bundled defaults) |
| `get_advanced_config_path()` | `<config>/advanced_config.json` |
| `get_brand_icon_bytes(name)` | Bytes of a bundled SVG/PNG icon |
| `get_brand_qicon(name)` | `QIcon` from bundled icon (lazy Qt import) |
| `print_paths()` | Prints all resolved paths to stdout |

**Environment variable overrides:**

| Variable | Overrides | Default |
|----------|-----------|---------|
| `MAT_DATA_DIR` | Models, training runs | `platformdirs.user_data_dir()` |
| `MAT_CONFIG_DIR` | Presets, skeletons, advanced config | `platformdirs.user_config_dir()` |

When adding a new module that needs to read models, configs, or assets, always import from `multi_tracker.paths` — never construct paths relative to `__file__`.
