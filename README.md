# Multi-Animal Tracker

High-performance multi-animal tracking and pose-labeling toolkit.

- Tracking GUI: `mat` / `multianimaltracker`
- Pose labeling UI: `posekit-labeler` / `pkl`

## Quick Install

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

## Launch

```bash
# tracking GUI
mat

# pose labeling UI
posekit-labeler
```

## Documentation

This repository now uses MkDocs Material for full system documentation.

- Home: `docs/index.md`
- Getting Started: `docs/getting-started/`
- User Guide: `docs/user-guide/`
- Developer Guide: `docs/developer-guide/`
- API + CLI Reference: `docs/reference/`

Build docs locally:

```bash
make docs-install
make docs-serve
# strict static build
make docs-build
```

Documentation quality audit:

```bash
make docs-quality
make docs-check
```

## Common Commands

```bash
# runtime deps
make setup
make install

# docs
make docs-install
make docs-quality
make docs-check

# cleanup
make clean
```

## Where to Find What

- App bootstrap: `src/multi_tracker/app/launcher.py`
- Tracking core: `src/multi_tracker/core/`
- Data/export pipeline: `src/multi_tracker/data/`
- Tracker GUI: `src/multi_tracker/gui/`
- Pose labeler app: `src/multi_tracker/posekit/`
- Utilities: `src/multi_tracker/utils/`
