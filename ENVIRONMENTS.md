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

### 2. `environment-minimal.yml` + `requirements-minimal.txt` - Lightweight Installation
**Recommended for**: First-time users, limited disk space, production deployments

- Minimal dependencies
- Faster installation
- Smaller environment size (~2GB vs ~5GB)
- No development tools or Jupyter

**Installation**:
```bash
mamba env create -f environment-minimal.yml
mamba activate multi-animal-tracker-minimal
uv pip install -v -r requirements-minimal.txt
```

---

### 3. `environment-gpu.yml` + `requirements-gpu.txt` - GPU-Accelerated
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
  "GPU_DEVICE_ID": 0
}
```

## Switching Environments

You can have multiple environments installed simultaneously:

```bash
# Create all environments
mamba env create -f environment.yml
mamba env create -f environment-minimal.yml
mamba env create -f environment-gpu.yml

# Install pip packages for each
mamba activate multi-animal-tracker-base && uv pip install -v -r requirements.txt
mamba activate multi-animal-tracker-minimal && uv pip install -v -r requirements-minimal.txt
mamba activate multi-animal-tracker-gpu && uv pip install -v -r requirements-gpu.txt

# Switch between them
mamba activate multi-animal-tracker-base     # Standard
mamba activate multi-animal-tracker-minimal  # Minimal
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

UV is automatically installed as part of each environment.

---

## Platform-Specific Notes

### macOS (Apple Silicon)
- GPU environment NOT supported (no CUDA on M1/M2/M3)
- Use `environment.yml` or `environment-minimal.yml`
- Automatic ARM64 optimization via conda

### Linux
- All environments fully supported
- For GPU: Install NVIDIA drivers first
  ```bash
  sudo apt-get install nvidia-driver-535  # Or latest
  ```

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
3. **Limited resources**: Use `environment-minimal.yml`
4. **Development**: Use `environment.yml` + install dev dependencies
