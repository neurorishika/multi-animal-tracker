# User Guide Overview

## Product Surface

The repository provides two user-facing UIs:

- **MAT GUI** (`mat`): multi-animal tracking from video to cleaned trajectories.
- **PoseKit Labeler** (`posekit-labeler`): pose annotation and dataset workflow.

## High-Level Flow

1. Configure environment and launch app.
2. Run detection/tracking or annotation workflow.
3. Validate outputs and quality metrics.
4. Export final files for downstream training/analysis.

## Choose the Right Tool

- Use **MAT** for position/orientation tracking and trajectory analysis.
- Use **PoseKit** for keypoint annotation, pose inference loops, and pose dataset prep.

## Reading This Guide

- If you are new: start with `workflow.md`.
- If you are tuning quality/speed: read `detection-modes.md`, `tracking-and-merging.md`, and `post-processing.md`.
- If you need exact parameter meaning: use `configuration-reference.md`.
