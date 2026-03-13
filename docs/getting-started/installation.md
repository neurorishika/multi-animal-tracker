# Installation

This guide walks you through a complete installation of Multi-Animal-Tracker for CPU, NVIDIA CUDA, Apple Silicon (MPS), and AMD ROCm setups.

## Before You Start

### System requirements

- Python 3.11+
- Conda or Mamba (Mamba recommended for speed)
- Linux/macOS/Windows supported (ROCm is Linux-only)

### Recommended tooling

- `mamba` for fast environment solving
- `uv` for fast pip installs (already included in project environment files)

### Clone the repository

```bash
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker
```

## Choose Your Installation Path

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
make install-gpu CUDA_MAJOR=13

# CUDA 12.x
# make install-gpu CUDA_MAJOR=12
```

### Important ONNX Runtime note (CUDA users)

`onnxruntime-gpu==1.24.1` currently expects CUDA 12 user-space runtime libraries (for example `libcublasLt.so.12`).

This project handles that by:

- Installing compatible CUDA 12 runtime libs in `environment-cuda.yml`
- Configuring `LD_LIBRARY_PATH` hooks during `make install-gpu`
- Disabling Ultralytics auto-requirement installs to avoid pulling CPU `onnxruntime`

This is expected behavior even on CUDA 13 systems.

### ONNX CPU option behavior in MAT

When selecting ONNX CPU in MAT, the app still uses the ONNX Runtime module provided by the active environment.

- You do **not** need a separate side-by-side `onnxruntime` CPU wheel in CUDA environments.
- Auto-install is disabled to prevent accidental replacement of GPU-capable runtime behavior.
- If ONNX CPU is selected, inference runs on CPU provider from the same ONNX Runtime install.

### FAISS note (CUDA 13)

- `faiss-gpu` currently has limited wheel availability for CUDA 13 + Python 3.13.
- `requirements-cuda13.txt` uses `faiss-cpu` for reliable installation.
- `requirements-cuda12.txt` includes `faiss-gpu` by default.
- If you have a compatible FAISS GPU build (commonly CUDA 12-focused), install it manually.

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
python -c "from multi_tracker.app.launcher import parse_arguments; print('✅ Core import OK')"
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
# or
multianimaltracker
```

---

## Makefile Workflow and Options

The Makefile provides one-command setup, install, update, and maintenance flows.

### Core setup targets

```bash
make setup
make setup-gpu
make setup-mps
make setup-rocm
```

### Install targets

```bash
make install
make install-gpu CUDA_MAJOR=13
# make install-gpu CUDA_MAJOR=12
make install-mps
make install-rocm
```

### Update targets

```bash
make env-update
make env-update-gpu CUDA_MAJOR=13
# make env-update-gpu CUDA_MAJOR=12
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
- Re-run `make install-gpu CUDA_MAJOR=13` (or `12`) to refresh linker hooks

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
- Full environment matrix: `ENVIRONMENTS.md`
- ROCm setup details: `ROCM_SETUP.md`

## Docs Tooling (Optional)

```bash
uv pip install -r requirements-docs.txt
mkdocs build --strict
```
