# ROCm Setup Guide for Multi-Animal Tracker

This guide provides detailed instructions for setting up AMD GPU acceleration with ROCm.

## Prerequisites

### Hardware Requirements
- **AMD GPU**: Radeon RX 5000+, Radeon Pro (Vega/RDNA/RDNA2/RDNA3), or Instinct MI series
- **Supported architectures**: GCN 3.0+, RDNA, CDNA
- Check compatibility: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html

### Software Requirements
- **Linux only**: Ubuntu 22.04/24.04, RHEL 8/9, SLES 15 SP4+
- **Kernel**: Linux 5.15+ recommended
- **Python**: 3.10-3.13

### Not Supported
- ❌ Windows (ROCm is Linux-only)
- ❌ macOS (use MPS environment instead)
- ❌ Older AMD GPUs (pre-GCN 3.0)

---

## Step-by-Step Installation

### 1. Verify GPU Compatibility

```bash
# Check your GPU
lspci | grep -i vga

# Example output:
# 03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 23 [Radeon RX 6600/6600 XT/6600M]
```

### 2. Install ROCm System Packages

#### Ubuntu 22.04 / 24.04

```bash
# Add ROCm repository
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

# Ubuntu 22.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

# Ubuntu 24.04
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2.4 noble main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list

# Update and install
sudo apt update
sudo apt install rocm-hip-runtime rocm-hip-sdk rocm-smi-lib

# Install additional development libraries for CuPy-ROCm
sudo apt install rocm-dev rocrand rocblas rocsparse rocfft hipsparse
```

#### RHEL 8 / 9

```bash
# Add ROCm repository
sudo tee /etc/yum.repos.d/rocm.repo <<EOF
[ROCm]
name=ROCm
baseurl=https://repo.radeon.com/rocm/rhel8/6.2.4/main
enabled=1
gpgcheck=1
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key
EOF

# Install packages
sudo yum install rocm-hip-runtime rocm-hip-sdk rocm-smi-lib
sudo yum install rocm-dev rocrand rocblas rocsparse rocfft hipsparse
```

### 3. Add User to ROCm Groups

```bash
# Add your user to video and render groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Log out and log back in for group changes to take effect
# Or use: newgrp video && newgrp render
```

### 4. Verify ROCm Installation

```bash
# Check ROCm installation
/opt/rocm/bin/rocm-smi

# Example output should show your GPU(s):
# ========================= ROCm System Management Interface =========================
# ================================ GPU 0 ================================
# GPU[0]          : card0
# GPU[0]          : Unique ID: 0x1234567890abcdef
# ...

# Check HIP (ROCm's CUDA equivalent)
/opt/rocm/bin/hipconfig --version
```

### 5. Set Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# ROCm environment
export ROCM_HOME=/opt/rocm
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0  # Use first GPU (change if needed)

# For CuPy-ROCm
export ROCM_PATH=$ROCM_HOME
export HIP_PATH=$ROCM_HOME/hip
```

Then reload:
```bash
source ~/.bashrc
```

### 6. Install Multi-Animal Tracker with ROCm

Now install the tracker environment:

```bash
# Create conda environment
mamba env create -f environment-rocm.yml
conda activate multi-animal-tracker-rocm

# Install Python packages (will take a few minutes - CuPy needs to compile)
uv pip install -v -r requirements-rocm.txt
```

### 7. Verify Python ROCm Integration

```bash
# Test PyTorch ROCm
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# Expected output:
# CUDA available: True
# Device: AMD Radeon RX 6600 XT

# Test CuPy-ROCm (may take a minute to compile kernels on first run)
python -c "import cupy as cp; print('CuPy device:', cp.cuda.Device(0).compute_capability)"

# Test ROCm version detection
python -c "import torch; print('ROCm version:', torch.version.hip if hasattr(torch.version, 'hip') else 'Not detected')"
```

---

## Troubleshooting

### GPU Not Detected

**Problem**: `rocm-smi` doesn't show GPU or shows "No GPU detected"

**Solutions**:
1. Check kernel compatibility:
   ```bash
   uname -r  # Should be 5.15 or newer
   ```

2. Verify GPU is recognized:
   ```bash
   lspci | grep -i amd
   dmesg | grep -i amdgpu
   ```

3. Ensure AMDGPU driver is loaded:
   ```bash
   lsmod | grep amdgpu
   # If not loaded:
   sudo modprobe amdgpu
   ```

4. Check user groups:
   ```bash
   groups  # Should include "video" and "render"
   ```

### PyTorch Not Using ROCm

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
1. Check PyTorch ROCm installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   # Should show something like: 2.5.0+rocm6.2
   ```

2. If it shows CPU version, reinstall:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install -v -r requirements-rocm.txt
   ```

3. Verify environment variables:
   ```bash
   echo $ROCM_HOME  # Should be /opt/rocm
   echo $PATH       # Should include /opt/rocm/bin
   ```

### CuPy-ROCm Compilation Errors

**Problem**: CuPy installation fails with compilation errors

**Solutions**:
1. Install all ROCm development libraries:
   ```bash
   sudo apt install rocm-dev rocrand rocblas rocsparse rocfft hipsparse hiprand hipfft
   ```

2. Set compilation flags:
   ```bash
   export CUPY_INSTALL_USE_HIP=1
   export ROCM_HOME=/opt/rocm
   uv pip install -v cupy-rocm-6-0 --no-cache-dir
   ```

3. If still failing, try without CuPy (will fall back to CPU for background subtraction):
   ```bash
   # Edit requirements-rocm.txt and comment out:
   # cupy-rocm-6-0
   ```

### Performance Issues

**Problem**: ROCm performance is much slower than expected

**Solutions**:
1. Check GPU clock speeds:
   ```bash
   rocm-smi --showclocks
   ```

2. Set performance mode:
   ```bash
   sudo rocm-smi --setperflevel high
   ```

3. Monitor GPU usage:
   ```bash
   watch -n 1 rocm-smi
   ```

4. Check VRAM usage:
   ```bash
   rocm-smi --showmeminfo vram
   ```

### Memory Errors

**Problem**: Out of memory errors during tracking

**Solutions**:
1. Reduce batch size in config:
   ```json
   {
     "TENSORRT_MAX_BATCH_SIZE": 8,  // Lower from default 16
     "RESIZE_FACTOR": 0.75           // Reduce resolution
   }
   ```

2. Monitor VRAM:
   ```bash
   rocm-smi --showmeminfo vram
   ```

3. Close other GPU applications

---

## ROCm Version Compatibility

| ROCm Version | Ubuntu | PyTorch | CuPy | Notes |
|--------------|--------|---------|------|-------|
| **6.2** | 22.04, 24.04 | ✅ 2.5+ | ✅ Experimental | Latest, recommended |
| **6.1** | 22.04, 24.04 | ✅ 2.4+ | ✅ Experimental | Stable |
| **6.0** | 22.04, 24.04 | ✅ 2.3+ | ✅ Experimental | Stable |
| **5.7** | 20.04, 22.04 | ✅ 2.1+ | ⚠️ Limited | Older, not recommended |

### Changing ROCm Version

If you need a different ROCm version:

1. **Edit `requirements-rocm.txt`**:
   ```bash
   # Change from:
   --extra-index-url https://download.pytorch.org/whl/rocm6.2
   cupy-rocm-6-0
   
   # To (for ROCm 6.1):
   --extra-index-url https://download.pytorch.org/whl/rocm6.1
   cupy-rocm-6-0  # CuPy-ROCm 6.0 works with ROCm 6.x
   ```

2. **Reinstall Python packages**:
   ```bash
   conda activate multi-animal-tracker-rocm
   uv pip install -v -r requirements-rocm.txt --force-reinstall
   ```

---

## Supported GPU Architectures

| Architecture | Example GPUs | Support Level | Performance |
|--------------|--------------|---------------|-------------|
| **RDNA 3** | RX 7000 series | ✅ Excellent | Best |
| **RDNA 2** | RX 6000 series | ✅ Excellent | Very good |
| **RDNA** | RX 5000 series | ✅ Good | Good |
| **Vega** | Vega 56/64, VII | ✅ Good | Good |
| **Polaris** | RX 400/500 series | ⚠️ Limited | Basic |
| **CDNA** | Instinct MI series | ✅ Excellent | Best (datacenter) |
| **GCN 3.0-5.0** | R9 Fury, RX 400 | ⚠️ Partial | Limited |

### Check Your GPU Architecture

```bash
# Get GPU details
rocminfo | grep "Name:"

# Or
lspci -vnn | grep -i vga
```

---

## Performance Benchmarks (ROCm vs CUDA)

Based on Radeon RX 6800 XT vs RTX 3080:

| Task | CUDA (RTX 3080) | ROCm (RX 6800 XT) | Ratio |
|------|-----------------|-------------------|-------|
| YOLO Inference (PyTorch) | 85 FPS (TensorRT) / 30 FPS (PyTorch) | 50-60 FPS (PyTorch) | ~0.7x |
| Background Subtraction (CuPy) | 400 FPS | 200-300 FPS | ~0.6x |
| CPU (Numba) | 60 FPS | 60 FPS | 1.0x |

**Note**: ROCm performance is typically 60-80% of equivalent CUDA performance, which is still **significantly faster than CPU-only** (3-10× speedup).

---

## Additional Resources

- **Official ROCm Docs**: https://rocm.docs.amd.com/
- **ROCm Quick Start**: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
- **PyTorch ROCm**: https://pytorch.org/get-started/locally/ (select ROCm)
- **CuPy ROCm Install**: https://docs.cupy.dev/en/stable/install.html#using-cupy-on-amd-gpu-experimental
- **ROCm GitHub**: https://github.com/ROCm/ROCm
- **GPU Compatibility**: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html

---

## Getting Help

If you encounter issues:

1. Check [Troubleshooting](#troubleshooting) section above
2. Verify ROCm installation: `rocm-smi --showproductname`
3. Test PyTorch ROCm: Run verification scripts above
4. Check [GitHub Issues](https://github.com/neurorishika/multi-animal-tracker/issues)
5. Include in bug reports:
   - GPU model (`lspci | grep VGA`)
   - ROCm version (`rocm-smi --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Error messages and full traceback

---

**Note**: ROCm support is **experimental** but generally works well. PyTorch ROCm is stable and well-supported. CuPy-ROCm is more experimental and may require troubleshooting.
