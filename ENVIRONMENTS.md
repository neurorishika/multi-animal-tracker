# Environment Files Guide

This directory contains multiple environment configuration files for different use cases. Choose the one that best fits your needs.

## Quick Start

All environments use a **two-step installation** for maximum speed:

```bash
# Step 1: Create conda environment with mamba (fast)
mamba env create -f environment.yml

# Step 2: Activate and install pip packages with uv (very fast)
mamba activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

> **Why two steps?** Mamba handles conda packages 10-100x faster than conda. UV handles pip packages 10-100x faster than pip. Combined, installation that took 30+ minutes now takes ~3 minutes.

---

## Available Environments

### 1. `environment.yml` + `requirements.txt` - Standard Full Installation
**Recommended for**: Most users

- Full feature set with all analysis tools
- CPU-optimized performance (Numba JIT)
- Jupyter notebook support for analysis
- All visualization tools

**Installation**:
```bash
mamba env create -f environment.yml
mamba activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

---

### 2. `environment-gpu.yml` + `requirements-gpu.txt` - NVIDIA GPU Accelerated
**Recommended for**: Users with NVIDIA GPUs, high-performance requirements

- Includes CuPy for GPU-accelerated background processing (8-30x speedup)
- TensorRT for accelerated YOLO inference (2-5x speedup)
- CUDA toolkit and cuDNN
- All features from standard environment
- Automatic CPU fallback if GPU unavailable

**Requirements**:
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.x, 12.x, or 13.x installed

**Installation**:
```bash
mamba env create -f environment-gpu.yml
mamba activate multi-animal-tracker-gpu
uv pip install -v -r requirements-gpu.txt
```

**CUDA Version Configuration**:
Edit `requirements-gpu.txt` and uncomment the appropriate lines:
- CUDA 13.x: `cu130` (default)
- CUDA 12.x: `cu126` or `cu128`
- CUDA 11.x: `cu118`

**Enable GPU in config**:
```json
{
  "ENABLE_GPU_BACKGROUND": true,
  "GPU_DEVICE_ID": 0,
  "ENABLE_TENSORRT": true,
  "TENSORRT_MAX_BATCH_SIZE": 16
}
```

---

### 4. `environment-mps.yml` + `requirements-mps.txt` - Apple Silicon (M1/M2/M3/M4)
**Recommended for**: macOS with Apple Silicon chips

- Metal Performance Shaders (MPS) GPU acceleration via PyTorch
- Optimized for ARM64 architecture      # NVIDIA
mamba env create -f environment-mps.yml      # Apple Silicon
mamba env create -f environment-rocm.yml     # AMD

# Install pip packages for each (after activating)
mamba activate multi-animal-tracker-base && uv pip install -v -r requirements.txt
mamba activate multi-animal-tracker-gpu && uv pip install -v -r requirements-gpu.txt
mamba activate multi-animal-tracker-mps && uv pip install -v -r requirements-mps.txt
mamba activate multi-animal-tracker-rocm && uv pip install -v -r requirements-rocm.txt

# Switch between them
mamba activate multi-animal-tracker-base     # Standard (CPU)
mamba activate multi-animal-tracker-gpu      # NVIDIA GPU
mamba activate multi-animal-tracker-mps      # Apple Silicon
mamba activate multi-animal-tracker-rocm     # AMD
mamba env create -f environment-mps.yml
mamba activate multi-animal-tracker-mps
uv pip install -v -r requirements-mps.txt
```

**Performance**:
- YOLO inference: ~2-3× faster than CPU (~30 FPS on M1 Pro)
- Background subtraction: CPU-based (~60 FPS with Numba)

**Enable MPS**:
MPS is automatically detected and used by PyTorch. No configuration needed.

---

### 5. `environment-rocm.yml` + `requirements-rocm.txt` - AMD GPU (ROCm)
**Recommended for**: Users with AMD Radeon/Instinct GPUs on Linux

- ROCm GPU acceleration for YOLO inference
- CuPy-ROCm for GPU-accelerated background processing (experimental)
- All features from standard environment
- Linux only (Ubuntu 22.04/24.04, RHEL 8/9, SLES 15)

**Requirements**:
- AMD GPU: Radeon RX 5000+, Radeon Pro, Instinct MI series
- ROCm 6.0+ installed system-wide
- Linux only

**⚠️ IMPORTANT: ROCm System Installation Required First**

ROCm is NOT a Python package - it requires system-level installation before Python packages will work.

**See detailed guide: [ROCM_SETUP.md](ROCM_SETUP.md)**

**Quick Pre-Installation** (Ubuntu 22.04/24.04):
```bash
# Install ROCm system packages (required before Python packages)
sudo apt install rocm-hip-runtime rocm-hip-sdk rocm-smi-lib
sudo apt install rocm-dev rocrand rocblas rocsparse rocfft hipsparse

# Add user to required groups
sudo usermod -a -G video,render $USER
# Log out and back in for group changes to take effect

# Verify ROCm installation
rocm-smi  # Should show your GPU
```

For other distributions (RHEL, SLES) or troubleshooting, see [ROCM_SETUP.md](ROCM_SETUP.md).

**Python Environment Installation** (after ROCm system install):
```bash
mamba env create -f environment-rocm.yml
mamba activate multi-animal-tracker-rocm
uv pip install -v -r requirements-rocm.txt  # May take 5-10 min (CuPy compilation)
```

**Verify Installation**:
```bash
# Test PyTorch ROCm
python -c "import torch; print('ROCm available:', torch.cuda.is_available())"

# Test CuPy-ROCm (may take a minute on first run)
python -c "import cupy as cp; print('CuPy device:', cp.cuda.Device(0))"
```

**ROCm Version Configuration**:
Edit `requirements-rocm.txt` and uncomment the appropriate lines:
- ROCm 6.2: `rocm6.2` (default, latest)
- ROCm 6.1: `rocm6.1`
- ROCm 6.0: `rocm6.0`
- ROCm 5.7: `rocm5.7` (older)

**Enable GPU in config**:
```json
{
  "ENABLE_GPU_BACKGROUND": true,
  "GPU_DEVICE_ID": 0,
  "YOLO_DEVICE": "cuda:0"
}
```

**Performance**:
- YOLO inference: ~40-60 FPS (close to CUDA performance)
- Background subtraction: ~200-300 FPS with CuPy-ROCm (experimental)

**Note**: If CuPy-ROCm fails to install or compile, the tracker will automatically fall back to CPU for background subtraction (still ~60 FPS with Numba). See [ROCM_SETUP.md](ROCM_SETUP.md) for troubleshooting.
**Note**: CuPy-ROCm is experimental and may have compatibility issues. Falls back to CPU if errors occur.

---

## Environment Comparison Table

| Environment | Size | Platform | YOLO FPS | BG Sub FPS | Use Case |
|-------------|------|----------|----------|------------|----------|
| **standard** | ~5 GB | All | 4 | 60 | General use, development |
| **gpu** | ~8 GB | NVIDIA + Linux/Win | 85 | 400 | High performance NVIDIA |
| **mps** | ~5 GB | Apple Silicon + macOS | 30 | 60 | Apple M1/M2/M3/M4 |
| **rocm** | ~7 GB | AMD + Linux | 40-60 | 200-300 | AMD Radeon/Instinct |

## Switching Environments

You can have multiple environments installed simultaneously:

```bash
# Create all environments
mamba env create -f environment.yml
mamba env create -f environment-gpu.yml

# Install pip packages for each
mamba activate multi-animal-tracker-base && uv pip install -v -r requirements.txt
mamba activate multi-animal-tracker-gpu && uv pip install -v -r requirements-gpu.txt

# Switch between them
mamba activate multi-animal-tracker-base     # Standard
mamba activate multi-animal-tracker-gpu      # GPU

# List installed environments
conda env list
```

---

## Updating Environments

### Update pip packages only (fast)
```bash
mamba activate multi-animal-tracker-base
uv pip install -v -r requirements.txt --upgrade
```

### Update conda packages
```bash
mamba activate multi-animal-tracker-base
mamba update --all
```

### Recreate from scratch
```bash
# Remove old environment
conda env remove -n multi-animal-tracker-base

# Create new one
mamba env create -f environment.yml
mamba activate multi-animal-tracker-base
uv pip install -v -r requirements.txt
```

---

## Installation Without Mamba/UV (Slower Alternative)

If you don't have mamba or uv installed:

```bash
# Standard conda (slower)
conda env create -f environment.yml
mamba activate multi-animal-tracker-base
pip install -r requirements.txt
```

To install mamba (recommended):
```bash
conda install -c conda-forge mamba
```

UV is aut
**Apple Silicon (M1/M2/M3/M4)**:
- ✅ Use `environment-mps.yml` for GPU-accelerated YOLO
- ✅ MPS backend provides 2-3× YOLO speedup
- ❌ No CUDA/CuPy (background subtraction uses CPU)
- Automatic ARM64 optimization via conda

**Intel Macs**:
- Use `environment.yml`
- CPU-only acceleration via Numba

### Linux
**NVIDIA GPU**:
- ✅ Use `environment-gpu.yml` (best performance)
- Install NVIDIA drivers first:
  ```bash
  sudo apt-get install nvidia-driver-535  # Or latest
  ```
- Supports CUDA 11.x, 12.x, 13.x

**AMD GPU**:
- ✅ Use `environment-rocm.yml` (ROCm support)
- Install ROCm 6.0+ first:
  ```bash
  # See: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/
  sudo apt-get install rocm-hip-runtime rocm-smi-lib
  ```
- Ubuntu 22.04/24.04, RHEL 8/9, SLES 15 supported
- Supported GPUs: Radeon RX 5000+, Instinct MI series

**Intel/AMD CPU**:
- Use `environment.yml`
- Numba JIT provides good CPU performance

### Windows
**NVIDIA GPU**:
- ✅ Use `environment-gpu.yml`
- Install CUDA Toolkit from NVIDIA first
- Use Anaconda Prompt (not PowerShell/CMD)

**AMD GPU**:
- ❌ ROCm not officially supported on Windows
- Use `environment.yml` (CPU-only)

1. **NVIDIA GPU (Linux/Windows)**: Use `environment-gpu.yml` for 8-30× speedup
2. **Apple Silicon (macOS)**: Use `environment-mps.yml` for 2-3× YOLO speedup
3. **AMD GPU (Linux)**: Use `environment-rocm.yml` for GPU acceleration
4. **CPU-only systems**: Use `environment.yml` (standard) with Numba optimization
5. **Development**: Use `environment.yml` + install dev dependencies

### Which Environment Should I Choose?

**Decision Tree:**
1. Do you have a GPU?
   - **NVIDIA** → `environment-gpu.yml` (best performance)
   - **AMD** → `environment-rocm.yml` (Linux only)
   - **Apple Silicon** → `environment-mps.yml` (macOS only)
   - **No GPU/Intel GPU** → `environment.yml`

2. Do you need all features?
   - **Yes** → Use appropriate GPU environment or `environment.yml`
   - **No** → `environment.yml` (standard, smaller footprint)

3. Are you developing/analyzing?
   - **Yes** → `environment.yml` or `environment-gpu.yml`
   - **No** → `environment.yml`

### Windows
- All environments supported
- Use Anaconda Prompt (not PowerShell/CMD)
- For GPU: Install CUDA Toolkit from NVIDIA first

---

## Troubleshooting

### Conda is slow
```bash
# Use mamba (faster conda alternative)
conda install -c conda-forge mamba
mamba env create -f environment.yml
```

### Environment conflicts
```bash
# Start fresh
conda clean --all
conda env create -f environment.yml
```
---

## Performance Tips

1. **CPU-only systems**: Use `environment.yml` (standard)
2. **NVIDIA GPU available**: Use `environment-gpu.yml` for 8-30x speedup
3. **Development**: Use `environment.yml` + install dev dependencies
