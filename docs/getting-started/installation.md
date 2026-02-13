# Installation

## Prerequisites

- Python 3.11+
- Conda or Mamba recommended
- Optional GPU stack depending on platform (CUDA, MPS, ROCm)

## Base Environment (Recommended)

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

## Platform-Specific Environments

```bash
# NVIDIA
mamba env create -f environment-cuda.yml
conda activate multi-animal-tracker-gpu
uv pip install -v -r requirements-gpu.txt

# Apple Silicon
mamba env create -f environment-mps.yml
conda activate multi-animal-tracker-mps
uv pip install -v -r requirements-mps.txt

# AMD ROCm
mamba env create -f environment-rocm.yml
conda activate multi-animal-tracker-rocm
uv pip install -v -r requirements-rocm.txt
```

## Verify Installation

```bash
python -c "from multi_tracker.app.launcher import parse_arguments; print('ok')"
python -c "from multi_tracker.utils.gpu_utils import log_device_info; log_device_info()"
```

## Docs Tooling

```bash
uv pip install -r requirements-docs.txt
mkdocs build --strict
```
