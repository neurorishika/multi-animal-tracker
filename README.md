# Multi-Animal-Tracker

<div align="center">
  <img src="brand/banner.png" alt="Multi-Animal-Tracker Banner" width="100%" />
</div>

<p align="center"><strong>The primary documentation lives here:</strong><br><a href="https://neurorishika.github.io/multi-animal-tracker/">https://neurorishika.github.io/multi-animal-tracker/</a></p>

## Start Here

- User docs: <https://neurorishika.github.io/multi-animal-tracker/>
- Getting Started: <https://neurorishika.github.io/multi-animal-tracker/getting-started/installation/>
- User Guide: <https://neurorishika.github.io/multi-animal-tracker/user-guide/overview/>
- Developer Guide: <https://neurorishika.github.io/multi-animal-tracker/developer-guide/architecture/>
- API + CLI Reference: <https://neurorishika.github.io/multi-animal-tracker/reference/api-index/>

## Install (Quick)

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-mps  # or your platform env
uv pip install -r requirements.txt
```

Platform-specific environments are documented in `ENVIRONMENTS.md` and in the online docs.

## Launch

```bash
# Multi-Animal-Tracker GUI
mat
# or
multianimaltracker

# PoseKit labeler
posekit-labeler
# or
pose
```

## Common Commands

```bash
# Docs
make docs-install
make docs-serve
make docs-build

# Lint / format
make lint-autofix
make lint-moderate
make lint-strict
```

## Project Links

- Docs site: <https://neurorishika.github.io/multi-animal-tracker/>
- Source: <https://github.com/neurorishika/multi-animal-tracker>
- License: `LICENSE`
