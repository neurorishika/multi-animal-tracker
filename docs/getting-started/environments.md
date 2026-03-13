# Environment Setup and Makefile Options

This page is the full environment and installation matrix for Multi-Animal-Tracker.

## Quick Start

### CPU (default)

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

## Environment Files

### `environment.yml` + `requirements.txt` (CPU)

Recommended for:

- General use and development
- Portable installs across all platforms
- CPU-only systems

### `environment-cuda.yml` + `requirements-cuda12/13.txt` (NVIDIA)

Recommended for:

- NVIDIA GPUs with CUDA support
- High-throughput detection and inference
- Users who need TensorRT and CuPy acceleration

CUDA requirements profiles:

- `requirements-cuda13.txt` for CUDA 13.x
- `requirements-cuda12.txt` for CUDA 12.x

Important ONNX Runtime note:

- `onnxruntime-gpu==1.24.1` currently links against CUDA 12 user-space libraries.
- `environment-cuda.yml` includes compatible CUDA 12 runtime libs for reliable loading.
- `make install-gpu` writes conda activate/deactivate hooks to set runtime linker paths.

Ultralytics ONNX note:

- MAT disables Ultralytics auto-install checks at runtime to prevent automatic installation of CPU `onnxruntime` in CUDA environments.
- This avoids accidental replacement of GPU-capable ONNX Runtime behavior.

FAISS note:

- `requirements-cuda12.txt` installs `faiss-gpu` by default.
- `requirements-cuda13.txt` installs `faiss-cpu` by default for reliable installation on CUDA 13 + Python 3.13.
- If you have a compatible CUDA 13 FAISS GPU build, you can replace/install `faiss-gpu` manually.

### `environment-mps.yml` + `requirements-mps.txt` (Apple Silicon)

Recommended for:

- macOS on M1/M2/M3/M4 chips
- PyTorch MPS acceleration

### `environment-rocm.yml` + `requirements-rocm.txt` (AMD ROCm)

Recommended for:

- Linux systems with supported AMD GPUs
- ROCm 6.0+ system installation

See `ROCM_SETUP.md` for system prerequisites.

## Installation by Platform

### CPU

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

### NVIDIA (CUDA)

```bash
nvidia-smi
mamba env create -f environment-cuda.yml
conda activate multi-animal-tracker-cuda
uv pip install -v -r requirements-cuda13.txt   # CUDA 13.x
# uv pip install -v -r requirements-cuda12.txt # CUDA 12.x
```

### Apple Silicon (MPS)

```bash
mamba env create -f environment-mps.yml
conda activate multi-animal-tracker-mps
uv pip install -v -r requirements-mps.txt
```

### AMD (ROCm)

```bash
mamba env create -f environment-rocm.yml
conda activate multi-animal-tracker-rocm
uv pip install -v -r requirements-rocm.txt
```

## Makefile Workflow and Options

The project Makefile provides one-command setup/update flows.

### Most-used setup commands

```bash
make setup            # CPU environment bootstrap
make setup-gpu        # CUDA environment bootstrap
make setup-mps        # Apple Silicon bootstrap
make setup-rocm       # AMD ROCm bootstrap
```

### Install package sets

```bash
make install
make install-gpu CUDA_MAJOR=13
# make install-gpu CUDA_MAJOR=12
make install-mps
make install-rocm
```

### Update environments

```bash
make env-update
make env-update-gpu CUDA_MAJOR=13
# make env-update-gpu CUDA_MAJOR=12
make env-update-mps
make env-update-rocm
```

### Remove environments

```bash
make env-remove
make env-remove-gpu
make env-remove-mps
make env-remove-rocm
```

### Important Make variables

- `CUDA_MAJOR` controls CUDA profile selection for GPU install/update targets.
- Supported values: `12`, `13`.
- Default: `13`.

### Discover all targets

```bash
make help
```

## Verify Installation

Run in the activated environment:

```bash
python -c "from multi_tracker.app.launcher import parse_arguments; print('✅ Core import OK')"
python -c "from multi_tracker.utils.gpu_utils import log_device_info; log_device_info()"
```

For CUDA installs:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected: `CUDAExecutionProvider` appears in providers list for ONNX CUDA setups.
