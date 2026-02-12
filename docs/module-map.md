# Module Map

This file tracks the post-reorganization Python package layout.

## Entry Points

- `multi_tracker.app.launcher`
- `multi_tracker.posekit.pose_label`

## Core Tracking

- `multi_tracker.core.tracking.worker`
- `multi_tracker.core.detectors.engine`
- `multi_tracker.core.background.model`
- `multi_tracker.core.filters.kalman`
- `multi_tracker.core.assigners.hungarian`
- `multi_tracker.core.post.processing`
- `multi_tracker.core.identity.analysis`

## Data and Integrations

- `multi_tracker.data.csv_writer`
- `multi_tracker.data.detection_cache`
- `multi_tracker.data.dataset_generation`
- `multi_tracker.data.dataset_merge`
- `multi_tracker.integrations.xanylabeling_cli`

## GUI

- `multi_tracker.gui.main_window`
- `multi_tracker.gui.widgets.histograms`
- `multi_tracker.gui.dialogs.train_yolo_dialog`
- `multi_tracker.posekit.pose_label`
- `multi_tracker.posekit.pose_label_dialogs`
- `multi_tracker.posekit.pose_label_extensions`
- `multi_tracker.posekit.pose_inference`

## Legacy

Legacy reference files were moved to `legacy/core/` and are not runtime code.
