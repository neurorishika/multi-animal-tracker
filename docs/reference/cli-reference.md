# CLI Reference

## Tracking App

Entry: `multi_tracker.app.launcher:main`

### Commands

- `multianimaltracker`
- `mat`

### Flags

- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`
- `--no-file-log`
- `--log-dir <path>`
- `--version`

## PoseKit Labeler

Entry: `multi_tracker.posekit.pose_label:main`

### Commands

- `posekit-labeler`
- `pose`

### Arguments

- `images` (optional positional): root folder of images
- `--out <path>`
- `--project <path>`
- `--new`

## Useful Examples

```bash
mat --log-level DEBUG
posekit-labeler /path/to/images --out /path/to/project
pose /path/to/images --new
```
