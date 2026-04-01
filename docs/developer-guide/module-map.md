# Module Map

## Entrypoints

- `multi_tracker.mat.app.launcher`
- `multi_tracker.posekit.ui.main`
- `multi_tracker.datasieve.gui`
- `multi_tracker.classkit.app`
- `multi_tracker.afterhours.app`

## Core

- `multi_tracker.core.tracking.worker`
- `multi_tracker.core.detectors.engine`
- `multi_tracker.core.background.model`
- `multi_tracker.core.filters.kalman`
- `multi_tracker.core.assigners.hungarian`
- `multi_tracker.core.post.processing`
- `multi_tracker.core.identity.analysis`

## Data

- `multi_tracker.data.csv_writer`
- `multi_tracker.data.detection_cache`
- `multi_tracker.data.dataset_generation`
- `multi_tracker.data.dataset_merge`

## Runtime

- `multi_tracker.runtime.compute_runtime`

## MAT GUI

- `multi_tracker.mat.gui.main_window`
- `multi_tracker.mat.gui.dialogs.train_yolo_dialog`
- `multi_tracker.mat.gui.widgets.histograms`

## PoseKit

- `multi_tracker.posekit.ui.main`
- `multi_tracker.posekit.ui.main_window`
- `multi_tracker.posekit.core.extensions`
- `multi_tracker.posekit.inference`

## Integrations and Utils

- `multi_tracker.integrations.sleap.service`
- `multi_tracker.integrations.xanylabeling.cli`
- `multi_tracker.utils.*`

## Paths and Resources

- `multi_tracker.paths` — central path resolver (brand icons, configs, models dir, training dir)
- `multi_tracker.paths_migrate` — one-time migration helper for repo-to-user-dir data
- `multi_tracker.resources` — bundled read-only assets package
- `multi_tracker.resources.brand` — SVG/PNG brand icons
- `multi_tracker.resources.configs` — default config presets
- `multi_tracker.resources.configs.skeletons` — skeleton keypoint definitions
