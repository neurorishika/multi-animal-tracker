# PoseKit Labeler

PoseKit is now organized under `multi_tracker.posekit` and launched via `posekit-labeler`.

## Full UI Control Reference

For complete PoseKit interface documentation (every form control, question labels, defaults, value-selection guidance, and failure modes), see:

- `/reference/ui-components-posekit/`

## Capabilities

- Project setup wizard (classes/keypoints/skeleton)
- Frame-by-frame keypoint annotation
- Autosave with safer write semantics
- Metadata/tagging and smart frame selection
- Split generation and training/evaluation dialogs

## SLEAP Backend Setup

For SLEAP env creation and ONNX/TensorRT integration setup, see:

- `/getting-started/integrations/`

## Key UX Concepts

- Keypoint order is semantic and must remain stable.
- Visibility flags encode present/occluded/missing states.
- Project settings changes can require label migration.

## Core Hotkeys

- `A/D`: previous/next frame
- `Q/E`: previous/next keypoint
- `Ctrl+S`: save
- `V/O/N`: visible/occluded/missing mode

## Output Expectations

- YOLO pose labels are normalized text files per image.
- Project metadata and optional posekit artifacts are stored in project output directories.

## Recommended Workflow

1. Finalize keypoint spec before large-scale labeling.
2. Label a pilot subset and run sanity checks.
3. Generate split files and train a baseline.
4. Use model-assisted passes and active learning to iterate.
