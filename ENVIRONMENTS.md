# Environment Files Guide

This directory contains multiple environment configuration files for different use cases. Choose the one that best fits your needs.

## Available Environments

### 1. `environment.yml` - Standard Full Installation
**Recommended for**: Most users

- Full feature set with all analysis tools
- CPU-optimized performance (Numba JIT)
- Jupyter notebook support for analysis
- All visualization tools

**Installation**:
```bash
conda env create -f environment.yml
conda activate multi-animal-tracker
```

---

### 2. `environment-minimal.yml` - Lightweight Installation
**Recommended for**: First-time users, limited disk space, production deployments

- Minimal dependencies
- Faster installation
- Smaller environment size (~2GB vs ~5GB)
- No development tools or Jupyter

**Installation**:
```bash
conda env create -f environment-minimal.yml
conda activate multi-animal-tracker-minimal
```

---

### 3. `environment-gpu.yml` - GPU-Accelerated
**Recommended for**: Users with NVIDIA GPUs, high-performance requirements

- Includes CuPy for GPU-accelerated background processing (8-30x speedup)
- CUDA toolkit and cuDNN
- All features from standard environment
- Automatic CPU fallback if GPU unavailable

**Requirements**:
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.8+ installed

**Installation**:
```bash
conda env create -f environment-gpu.yml
conda activate multi-animal-tracker-gpu
```

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
conda env create -f environment.yml
conda env create -f environment-minimal.yml
conda env create -f environment-gpu.yml

# Switch between them
conda activate multi-animal-tracker          # Standard
conda activate multi-animal-tracker-minimal  # Minimal
conda activate multi-animal-tracker-gpu      # GPU

# List installed environments
conda env list
```

---

## Updating Environments

### Update existing environment
```bash
# Activate environment
conda activate multi-animal-tracker

# Update from file
conda env update -f environment.yml --prune

# Reinstall package
pip install -e . --upgrade
```

### Recreate from scratch
```bash
# Remove old environment
conda env remove -n multi-animal-tracker

# Create new one
conda env create -f environment.yml
```

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
