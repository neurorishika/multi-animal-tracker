# User Guide Overview

## Product Surface

The HYDRA Suite provides seven tools:

- **HYDRA** (`hydra`): launcher and tool selector.
- **TrackerKit** (`trackerkit`): multi-animal tracking from video to cleaned trajectories.
- **PoseKit** (`posekit`): pose annotation and dataset workflow.
- **ClassKit** (`classkit`): classification and embedding toolkit for identity analysis.
- **DetectKit** (`detectkit`): detection model training and dataset management.
- **FilterKit** (`filterkit`): dataset filtering and curation.
- **RefineKit** (`refinekit`): interactive trajectory proofreading.

## High-Level Flow

1. Configure environment and launch app.
2. Run detection/tracking or annotation workflow.
3. Validate outputs and quality metrics.
4. Export final files for downstream training/analysis.

## Choose the Right Tool

- Use **TrackerKit** for position/orientation tracking and trajectory analysis.
- Use **PoseKit** for keypoint annotation, pose inference loops, and pose dataset prep.
- Use **ClassKit** for identity classification, embedding visualization, and active learning.
- Use **DetectKit** for training and evaluating detection models.
- Use **FilterKit** for curating and filtering datasets.
- Use **RefineKit** for interactive proofreading and correction of tracked trajectories.

## Reading This Guide

- If you are new: start with `workflow.md`.
- If you are tuning quality/speed: read `detection-modes.md`, `tracking-and-merging.md`, and `post-processing.md`.
- If you need exact parameter meaning: use `configuration-reference.md`.
