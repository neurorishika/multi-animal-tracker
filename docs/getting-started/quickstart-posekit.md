# Quickstart (PoseKit)

## Launch

```bash
posekit-labeler
# or
pkl
```

## Basic Project Setup

1. Select images folder.
2. Create or load project metadata.
3. Define classes, keypoints, and skeleton edges.
4. Start labeling frames.

## Core Labeling Loop

1. Select frame.
2. Place keypoints in order.
3. Save (`Ctrl+S`) or rely on autosave.
4. Move to next frame.

## Built-In Advanced Workflow

- Smart frame selection
- Project settings and keypoint migration
- Dataset split generation
- Training/evaluation dialogs
- Metadata tagging and active learning hooks

## Output Artifacts

- YOLO pose labels (`.txt`)
- Project config metadata
- Optional split files and evaluation artifacts
- PoseKit working folders under `posekit/` in project output
