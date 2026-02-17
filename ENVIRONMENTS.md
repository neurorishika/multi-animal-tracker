# Environment Files Guide

The full environment and Makefile setup guide now lives in MkDocs:

- `docs/getting-started/environments.md`

Use this file as a quick local shortcut.

## Quick Commands

### CPU

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

### NVIDIA CUDA

```bash
mamba env create -f environment-cuda.yml
conda activate multi-animal-tracker-cuda
uv pip install -v -r requirements-cuda13.txt   # CUDA 13.x
# uv pip install -v -r requirements-cuda12.txt # CUDA 12.x
```

### Makefile discovery

```bash
make help
```
