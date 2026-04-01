# Installation

This guide walks you through a complete installation of Multi-Animal-Tracker for CPU, NVIDIA CUDA, Apple Silicon (MPS), and AMD ROCm setups.

## Before You Start

### System requirements

- Python 3.11+
- Linux/macOS/Windows supported (ROCm is Linux-only)

### Recommended tooling (for full environment setup)

- `mamba` for fast environment solving
- `uv` for fast pip installs (already included in project environment files)

## Quick Install (pip)

If you want the simplest install without GPU acceleration:

```bash
pip install multi-animal-tracker
```

This installs all dependencies and bundled assets. Launch with `mat` or `posekit-labeler`.

### GPU via pip

Install the correct PyTorch variant first, then install the package with GPU extras:

```bash
# NVIDIA CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install multi-animal-tracker[cuda]

# NVIDIA CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install multi-animal-tracker[cuda]

# Apple Silicon (MPS) — torch includes MPS by default
pip install torch torchvision
pip install multi-animal-tracker[mps]

# AMD ROCm
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install multi-animal-tracker[rocm]
```

> **Note:** PyTorch GPU builds are not hosted on PyPI. You must install torch from PyTorch's own index first. The `pip install multi-animal-tracker` step will not overwrite your existing torch installation.

### Data directories

After installation, user data is stored in platform-appropriate locations:

| Data | macOS | Linux |
|------|-------|-------|
| Config | `~/Library/Application Support/multi-animal-tracker/` | `~/.config/multi-animal-tracker/` |
| Models | `~/Library/Application Support/multi-animal-tracker/models/` | `~/.local/share/multi-animal-tracker/models/` |
| Training | `~/Library/Application Support/multi-animal-tracker/training/` | `~/.local/share/multi-animal-tracker/training/` |

Default config presets and skeleton definitions are bundled with the package and automatically seeded to your config directory on first run.

### Migrating from a repo checkout

If you previously used a cloned repo with models in `<repo>/models/` and training data in `<repo>/training/`, migrate with:

```bash
python -m multi_tracker.paths_migrate /path/to/multi-animal-tracker --dry-run  # preview
python -m multi_tracker.paths_migrate /path/to/multi-animal-tracker            # copy
```

---

## Full Environment Setup (recommended for GPU and development)

The full environment setup uses conda/mamba for system libraries and pip for Python packages. This is recommended for GPU users and developers.

### Clone the repository

```bash
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker
```

### Choose Your Installation Path

- **CPU-only (most portable):** `environment.yml` + `requirements.txt`
- **NVIDIA GPU:** `environment-cuda.yml` + `requirements-cuda12.txt` or `requirements-cuda13.txt`
- **Apple Silicon (M1/M2/M3/M4):** `environment-mps.yml` + `requirements-mps.txt`
- **AMD GPU (ROCm):** `environment-rocm.yml` + `requirements-rocm.txt`

---

## 1) CPU Installation (Default)

Use this if you do not need GPU acceleration.

```bash
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

---

## 2) NVIDIA GPU Installation (CUDA)

Use this for CUDA-enabled systems.

### Step A: Confirm CUDA major version

```bash
nvidia-smi
```

Pick one requirements profile:

- CUDA 13.x: `requirements-cuda13.txt`
- CUDA 12.x: `requirements-cuda12.txt`

### Step B: Create and activate CUDA environment

```bash
mamba env create -f environment-cuda.yml
conda activate multi-animal-tracker-cuda
```

### Step C: Install Python packages by CUDA version

```bash
# Optional explicit shared CUDA layer install (common deps used by both profiles)
uv pip install -v -r requirements-cuda.txt

# CUDA 13.x
uv pip install -v -r requirements-cuda13.txt

# CUDA 12.x
# uv pip install -v -r requirements-cuda12.txt
```

Notes:

- `requirements-cuda13.txt` and `requirements-cuda12.txt` already include `-r requirements-cuda.txt`.
- Running the shared install command explicitly is optional, but can make the install flow clearer when debugging package conflicts.

### Optional: Makefile helper

```bash
# CUDA 13.x
make install-cuda CUDA_MAJOR=13

# CUDA 12.x
# make install-cuda CUDA_MAJOR=12
```

### Important ONNX Runtime note (CUDA users)

`onnxruntime-gpu==1.24.1` currently expects CUDA 12 user-space runtime libraries (for example `libcublasLt.so.12`).

This project handles that by:

- Installing compatible CUDA 12 runtime libs in `environment-cuda.yml`
- Configuring `LD_LIBRARY_PATH` hooks during `make install-cuda`
- Disabling Ultralytics auto-requirement installs to avoid pulling CPU `onnxruntime`

This is expected behavior even on CUDA 13 systems.

### ONNX CPU option behavior in MAT

When selecting ONNX CPU in MAT, the app still uses the ONNX Runtime module provided by the active environment.

- You do **not** need a separate side-by-side `onnxruntime` CPU wheel in CUDA environments.
- Auto-install is disabled to prevent accidental replacement of GPU-capable runtime behavior.
- If ONNX CPU is selected, inference runs on CPU provider from the same ONNX Runtime install.

---

## 3) Apple Silicon Installation (MPS)

```bash
mamba env create -f environment-mps.yml
conda activate multi-animal-tracker-mps
uv pip install -v -r requirements-mps.txt
```

---

## 4) AMD GPU Installation (ROCm)

Install ROCm system packages first, then create the Python environment.

```bash
mamba env create -f environment-rocm.yml
conda activate multi-animal-tracker-rocm
uv pip install -v -r requirements-rocm.txt
```

For ROCm system prerequisites and troubleshooting, see `ROCM_SETUP.md`.

---

## Verify Your Installation

Run these checks in the activated environment:

```bash
python -c "from multi_tracker.mat.app.launcher import parse_arguments; print('✅ Core import OK')"
python -c "from multi_tracker.utils.gpu_utils import log_device_info; log_device_info()"
```

For CUDA environments, also verify ONNX Runtime providers:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

You should see `CUDAExecutionProvider` in the output for a working CUDA setup.

---

## Launch the Application

```bash
mat
```

---

## Makefile Workflow and Options

The Makefile provides one-command setup, install, update, and maintenance flows.

### Core setup targets

```bash
make setup
make setup-cuda
make setup-mps
make setup-rocm
```

### Install targets

```bash
make install
make install-cuda CUDA_MAJOR=13
# make install-cuda CUDA_MAJOR=12
make install-mps
make install-rocm
```

### Update targets

```bash
make env-update
make env-update-cuda CUDA_MAJOR=13
# make env-update-cuda CUDA_MAJOR=12
make env-update-mps
make env-update-rocm
```

### Useful options

- `CUDA_MAJOR` selects CUDA profile for GPU install/update (`12` or `13`)
- default `CUDA_MAJOR` is `13`
- run `make help` for the full command catalog

---

## Update an Existing Environment

After activating the target environment:

```bash
# CPU
uv pip install -v -r requirements.txt --upgrade

# CUDA 13
uv pip install -v -r requirements-cuda.txt --upgrade
uv pip install -v -r requirements-cuda13.txt --upgrade

# CUDA 12
uv pip install -v -r requirements-cuda.txt --upgrade
# uv pip install -v -r requirements-cuda12.txt --upgrade
```

If conda dependencies changed, update environment packages too:

```bash
mamba env update -f environment.yml --prune
```

Use the matching environment file for MPS/ROCm/CUDA updates.

---

## Troubleshooting

### `CUDAExecutionProvider` missing from ONNX Runtime

- Ensure you installed `requirements-cuda12.txt` or `requirements-cuda13.txt` (not CPU requirements)
- Ensure you are in `multi-animal-tracker-cuda`
- Re-run `make install-cuda CUDA_MAJOR=13` (or `12`) to refresh linker hooks

### Library error like `libcublasLt.so.12: cannot open shared object file`

- This indicates CUDA 12 user-space libs are not visible to the runtime
- Re-activate the conda environment after running the install helper
- Confirm provider list with:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

### Fresh reinstall (clean slate)

```bash
conda env remove -n multi-animal-tracker-cuda
mamba env create -f environment-cuda.yml
conda activate multi-animal-tracker-cuda
uv pip install -v -r requirements-cuda13.txt
```

---

## Related Setup Docs

- Integrations: [Integrations](integrations.md)
- Full environment matrix: [Environments](environments.md)
- ROCm setup details: `ROCM_SETUP.md`

## Docs Tooling (Optional)

```bash
uv pip install -r requirements-docs.txt
mkdocs build --strict
```
