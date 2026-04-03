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

### Option A: pip (CPU, simplest)

```bash
pip install multi-animal-tracker
```

For GPU (NVIDIA), install PyTorch first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install multi-animal-tracker[cuda]
```

### Option B: Full environment (GPU, recommended for development)

```bash
mamba env create -f environment.yml          # or environment-mps.yml / environment-cuda.yml
conda activate multi-animal-tracker          # or -mps / -cuda
uv pip install -r requirements.txt           # or requirements-mps.txt / requirements-cuda13.txt
```

Platform-specific environments are documented in the [online docs](https://neurorishika.github.io/multi-animal-tracker/getting-started/environments/).

## Launch

```bash
# Multi-Animal-Tracker GUI
mat

# PoseKit labeler
posekit-labeler

# Other tools
filterkit
classkit
refinekit
```

## Common Commands

```bash
# Docs
make docs-install
make docs-serve
make docs-build
make techref-build

# Lint / format
make lint-fix
make lint
make lint-strict
```

## Publication-Style Reference

The repository also includes a standalone LaTeX technical reference for the current tracking algorithm in `technical-reference/`.

## Project Links

- Docs site: <https://neurorishika.github.io/multi-animal-tracker/>
- Source: <https://github.com/neurorishika/multi-animal-tracker>
- License: `LICENSE`
