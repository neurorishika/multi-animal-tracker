# Module Map

## Entrypoints

- `hydra_suite.tracker.app.launcher`
- `hydra_suite.posekit.ui.main`
- `hydra_suite.filterkit.gui`
- `hydra_suite.classkit.app`
- `hydra_suite.refinekit.app`

## Core

- `hydra_suite.core.tracking.worker`
- `hydra_suite.core.detectors.engine`
- `hydra_suite.core.background.model`
- `hydra_suite.core.filters.kalman`
- `hydra_suite.core.assigners.hungarian`
- `hydra_suite.core.post.processing`
- `hydra_suite.core.identity.analysis`

## Data

- `hydra_suite.data.csv_writer`
- `hydra_suite.data.detection_cache`
- `hydra_suite.data.dataset_generation`
- `hydra_suite.data.dataset_merge`

## Runtime

- `hydra_suite.runtime.compute_runtime`

## MAT GUI

- `hydra_suite.tracker.gui.main_window`
- `hydra_suite.tracker.gui.dialogs.train_yolo_dialog`
- `hydra_suite.tracker.gui.widgets.histograms`

## PoseKit

- `hydra_suite.posekit.ui.main`
- `hydra_suite.posekit.ui.main_window`
- `hydra_suite.posekit.core.extensions`
- `hydra_suite.posekit.inference`

## Integrations and Utils

- `hydra_suite.integrations.sleap.service`
- `hydra_suite.integrations.xanylabeling.cli`
- `hydra_suite.utils.*`

## Paths and Resources

- `hydra_suite.paths` — central path resolver; all apps import from here for models, configs, assets
  - `get_models_dir()`, `get_presets_dir()`, `get_skeleton_dir()`, `get_training_runs_dir()`
  - `get_brand_icon_bytes(name)`, `get_brand_qicon(name)`
  - `print_paths()` — debug helper to show all resolved paths
  - Respects `MAT_DATA_DIR` and `MAT_CONFIG_DIR` environment variable overrides
- `hydra_suite.paths_migrate` — one-time migration helper for repo-to-user-dir data
- `hydra_suite.resources` — bundled read-only assets package
- `hydra_suite.resources.brand` — SVG/PNG brand icons
- `hydra_suite.resources.configs` — default config presets
- `hydra_suite.resources.configs.skeletons` — skeleton keypoint definitions
