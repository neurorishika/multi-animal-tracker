# Tracking Optimization System (Auto-Tuner)

This document details the architecture and implementation of the Bayesian Tracking Optimization system in TrackerKit.

## Overview

The Tracking Optimization system (Auto-Tuner) is designed to solve the "parameter-tuning headache" in multi-animal tracking. Instead of manually adjusting sliders for Kalman filters, association thresholds, and YOLO confidences, users can select a short representative slice of their video and let an automated agent find the mathematically optimal settings.

## Core Architecture

The system follows a **Hybrid Optimization** approach, separating heavy compute (AI detection) from lightweight logic (tracking association).

### 1. Phase 1: Detection Caching
Before optimization begins, the system requires a `DetectionCache` (`.npz`).
- **Efficiency:** By caching the raw YOLO detections (bounding boxes, confidences, and OBB corners), the optimizer can run hundreds of tracking "trials" without ever re-running the GPU-heavy YOLO model.
- **Reuse:** The system reuses the same cache used for standard forward/backward tracking passes.

### 2. Phase 2: Bayesian Optimization (Optuna)
The "brain" of the tuner is powered by **Optuna**.
- **Strategy:** Unlike a "Grid Search" which tries every combination blindly, the tuner uses a **Tree-structured Parzen Estimator (TPE)**. It analyzes the scores of previous trials to predict which parameter ranges are most likely to yield a better result.
- **Search Space:** It dynamically constructs a search space based on user-selected checkboxes in the UI.

### 3. Phase 3: Quality Scoring
Each trial is evaluated using the `FrameQualityScorer`. The "Best" result is determined by minimizing a multi-factor error cost:
- **Lost Tracks:** Penalizes parameters that cause tracks to terminate prematurely.
- **Uncertainty:** Penalizes high Kalman filter variance (indicates poor motion prediction).
- **Assignment Cost:** Penalizes large jumps or poor spatial/orientation matches.
- **Detection Count Mismatch:** Penalizes YOLO thresholds that result in inconsistent animal counts.

## Tunable Parameters

The system provides granular control over which aspects of the pipeline are optimized:

| Category | Parameter | Description |
| :--- | :--- | :--- |
| **Detection** | `YOLO_CONFIDENCE_THRESHOLD` | Sensitivity of animal detection. |
| | `YOLO_IOU_THRESHOLD` | NMS overlap filtering. |
| **Movement** | `MAX_DISTANCE_MULTIPLIER` | Max distance an animal can travel per frame. |
| **Kalman** | `KALMAN_NOISE_COVARIANCE` | Motion smoothing (Process Noise). |
| | `KALMAN_MEASUREMENT_COVARIANCE` | Responsiveness to detections (Meas Noise). |
| **Weights** | `W_POSITION`, `W_ORIENTATION` | Relative importance of distance vs direction. |
| | `W_AREA`, `W_ASPECT` | Relative importance of size/shape consistency. |

## Key Components

### `TrackingOptimizer` (Threaded)
Located in `src/hydra_suite/core/tracking/optimizer.py`.
- Manages the Optuna `study`.
- Orchestrates the `_run_tracking_pass_cached` loop.
- Emits progress and results back to the UI.

### `ParameterHelperDialog`
Located in `src/hydra_suite/trackerkit/gui/dialogs/parameter_helper.py`.
- Provides the selection grid for parameters.
- Displays a ranked table of the top trials.
- Handles the "Preview" logic.

### `TrackingPreviewWorker`
- Allows the user to select any result from the optimization table and watch it run in the main TrackerKit window.
- This provides "human-in-the-loop" verification to ensure the mathematical optimum also looks visually stable.

## Workflow for Users

1.  **Select Range:** Use the `Start Frame` and `End Frame` boxes in the main TrackerKit UI to select a challenging slice (e.g., 200 frames with occlusions).
2.  **Open Tuner:** Click **"Auto-Tune Tracking Parameters..."** in the Tracking tab.
3.  **Configure:** Check the parameters that seem problematic (e.g., if tracks are swapping, check `W_POSITION` and `MAX_DISTANCE`).
4.  **Optimize:** Run the Bayesian search (usually 50 trials take ~10-20 seconds).
5.  **Preview:** Select the top-ranked result and click **"Preview Selected"**.
6.  **Apply:** Click **"Apply Best to TrackerKit"** to update all main UI sliders and spin-boxes automatically.

## Integration Details

The system is tightly integrated into `MainWindow` via the `_open_parameter_helper` method. It ensures that when a user applies a result, `_update_parameters_from_ui()` is called to sync the internal tracking engine with the new visual settings.
