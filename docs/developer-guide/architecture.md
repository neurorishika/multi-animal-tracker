# Architecture

## System Layers

- `hydra_suite.tracker`: MAT launcher, GUI, dialogs, widgets.
- `hydra_suite.posekit`: pose-labeling application and related dialogs/inference flows.
- `hydra_suite.classkit`: classification/embedding toolkit.
- `hydra_suite.refinekit`: interactive proofreading.
- `hydra_suite.filterkit`: data sieve tool.
- `hydra_suite.integrations`: external tool bridges (SLEAP, X-AnyLabeling).
- `hydra_suite.core`: detection, filtering, assignment, post-processing, worker orchestration.
- `hydra_suite.runtime`: compute runtime selection and GPU utilities.
- `hydra_suite.data`: CSV/cache/dataset generation and merging.
- `hydra_suite.training`: dataset builders, training runner, registry, model publishing.
- `hydra_suite.utils`: shared helpers (GPU detection, geometry, image processing, batching, prefetch).
- `hydra_suite.paths`: central path resolution — bundled assets via `importlib.resources`, user dirs via `platformdirs`.
- `hydra_suite.resources`: bundled read-only assets (brand icons, default configs, skeletons).

## Design Intent

- Keep high-throughput frame processing in worker-style components.
- Keep GUI orchestration separated from algorithm implementations.
- Keep data export/cache utilities outside core tracking logic.

## Runtime Entry Paths

- MAT: `hydra_suite.tracker.app.launcher:main`
- PoseKit: `hydra_suite.posekit.ui.main:main`
- FilterKit: `hydra_suite.filterkit.gui:main`
- ClassKit: `hydra_suite.classkit.app:main`
- RefineKit: `hydra_suite.refinekit.app:main`

## Key Operational Boundaries

- Core tracking should not depend on GUI widget internals.
- Data layer should stay reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow but shared environment/tooling.
- All path resolution goes through `hydra_suite.paths`. No module uses `Path(__file__).parents[N]` to navigate to repo root.
- Bundled read-only assets are in `hydra_suite.resources` and accessed via `importlib.resources`.
- User-writable data (models, training runs, config) lives in platform directories, never inside the installed package.
- Users can override directories with `HYDRA_DATA_DIR` and `HYDRA_CONFIG_DIR` environment variables.

## Data and Config Directories

All apps share paths via `hydra_suite.paths`. This module resolves user-writable directories (models, training, presets, skeletons) and bundled read-only assets (brand icons, default configs).

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
| `HYDRA_DATA_DIR` | Models, training runs | `platformdirs.user_data_dir()` |
| `HYDRA_CONFIG_DIR` | Presets, skeletons, advanced config | `platformdirs.user_config_dir()` |

When adding a new module that needs to read models, configs, or assets, always import from `hydra_suite.paths` — never construct paths relative to `__file__`.
