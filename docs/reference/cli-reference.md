# CLI Reference

## Tracking App

Entry: `hydra_suite.tracker.app.launcher:main`

### Commands

- `hydra`
- `hydra`

### Flags

- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`
- `--no-file-log`
- `--log-dir <path>`
- `--version`

## PoseKit Labeler

Entry: `hydra_suite.posekit.ui.main:main`

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
hydra --log-level DEBUG
posekit-labeler /path/to/images --out /path/to/project
pose /path/to/images --new
```
