# Installation

## Which method should I use?

| | pip | conda + pip |
|---|---|---|
| **Best for** | End users, quick start, CPU use | Developers, full GPU acceleration |
| **GPU inference** (YOLO, pose) | Yes | Yes |
| **TensorRT acceleration** | No | Yes |
| **ONNX Runtime GPU** | Yes, but user must have CUDA toolkit installed | Yes, conda provides CUDA runtime libs |
| **CuPy GPU background subtraction** | Yes, but user must have CUDA toolkit installed | Yes |
| **LD_LIBRARY_PATH management** | Manual | Automatic (`make install-cuda` sets hooks) |
| **Requires git clone** | No | Yes |
| **Requires conda/mamba** | No | Yes |
| **Editable install for development** | With `pip install -e .` | Yes (via `make install-*`) |

**In short:** pip gets you GPU inference via PyTorch (YOLO detection, pose estimation, classification). conda + pip adds TensorRT optimization and handles CUDA runtime library paths automatically. If you're not sure, start with pip — you can switch to conda later.

---

## pip install

Requires only Python 3.11+.

### CPU

```bash
pip install multi-animal-tracker
```

### Apple Silicon (MPS)

```bash
pip install torch torchvision
pip install "multi-animal-tracker[mps]"
```

MPS support is built into the standard PyTorch macOS wheel. GPU acceleration works for YOLO inference and neural network operations. Background subtraction remains CPU (Numba-optimized).

### NVIDIA GPU (CUDA)

```bash
# Step 1: Install PyTorch from PyTorch's index (not PyPI)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install multi-animal-tracker with CUDA extras
pip install "multi-animal-tracker[cuda]"
```

Replace `cu128` with your CUDA version: `cu126`, `cu128`, or `cu130`.

**What works:** PyTorch GPU inference (YOLO detection, pose, classification), ONNX Runtime GPU, CuPy GPU background subtraction — provided you have the CUDA toolkit installed system-wide.

**What doesn't work:** TensorRT acceleration (requires conda path). If `onnxruntime-gpu` fails to find CUDA libraries, you need either the CUDA toolkit installed or the conda path which manages this automatically.

### AMD GPU (ROCm)

Requires [ROCm 6.0+](https://rocm.docs.amd.com/) installed system-wide (Linux only).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install "multi-animal-tracker[rocm]"
```

**What works:** PyTorch GPU inference, CuPy-ROCm background subtraction.

**What doesn't work:** TensorRT (NVIDIA only), ONNX Runtime GPU (no maintained ROCm provider — falls back to CPU).

### Why is GPU install two commands?

PyTorch GPU builds are hosted on PyTorch's own server, not PyPI. Python packaging (PEP 621) has no way to specify per-dependency index URLs. Every ML project handles this identically — you install torch separately, then install the package. The second command will **not** overwrite your existing torch.

---

## conda + pip install (developer / full GPU)

This method uses conda for system libraries (Qt, CUDA toolkit, runtime libs) and pip for Python packages. It provides:

- Automatic CUDA runtime library management (no system CUDA toolkit needed)
- `LD_LIBRARY_PATH` hooks for ONNX Runtime GPU compatibility
- TensorRT support
- Editable install for development

### Step 1: Clone

```bash
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker
```

### Step 2: Create environment and install

Pick your platform:

```bash
# CPU
make setup
conda activate multi-animal-tracker
make install

# Apple Silicon (MPS)
make setup-mps
conda activate multi-animal-tracker-mps
make install-mps

# NVIDIA GPU (CUDA)
make setup-cuda
conda activate multi-animal-tracker-cuda
make install-cuda CUDA_MAJOR=13     # or CUDA_MAJOR=12

# AMD GPU (ROCm — requires ROCm 6.0+ installed system-wide)
make setup-rocm
conda activate multi-animal-tracker-rocm
make install-rocm
```

### What conda adds beyond pip

The conda environment files (`environment-cuda.yml`, etc.) install:

- **CUDA 12 runtime libraries** (`libcublas`, `libcudnn`, `libcufft`) — needed by `onnxruntime-gpu` even on CUDA 13 systems
- **PySide6 / Qt6** system dependencies — avoids `libGL` / `libxcb` errors on Linux
- **Python** itself — pinned to a known-good version

The `make install-cuda` target additionally:

- Installs **TensorRT** (NVIDIA inference optimizer)
- Configures **`LD_LIBRARY_PATH` activation hooks** so ONNX Runtime finds the conda-provided CUDA libs
- Installs **ONNX export tools** (`onnxscript`, `onnxslim`)

### How dependencies stay in sync

```
pyproject.toml [dependencies]          ← single source of truth for base deps
                                         (numpy, scipy, PySide6, ultralytics, ...)

requirements-*.txt                     ← adds ONLY what pyproject.toml can't express:
  torch + --extra-index-url               - torch (needs custom index URL)
  tensorrt, onnxruntime-gpu               - GPU-specific packages
  -e .                                    - pulls all pyproject.toml deps
```

Adding a new base dependency means editing **one place**: `pyproject.toml`. The `requirements-*.txt` files inherit it via `-e .`.

### ONNX Runtime note (CUDA)

`onnxruntime-gpu` links against CUDA 12 user-space libraries. On CUDA 13 systems, the conda environment provides compatible CUDA 12 libs, and `make install-cuda` sets `LD_LIBRARY_PATH` activation hooks. This is expected and handled automatically.

---

## After installation

### Verify

```bash
mat --help
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| MPS:', torch.backends.mps.is_available())"
```

### Launch

```bash
mat                # Multi-Animal-Tracker GUI
posekit-labeler    # PoseKit pose-labeling GUI
datasieve          # DataSieve tool
classkit           # ClassKit labeler
afterhours         # Afterhours proofreading
```

### Data directories

User data is stored in platform-appropriate locations (not inside the repo or package):

| Data | macOS | Linux |
|------|-------|-------|
| Config / presets | `~/Library/Application Support/multi-animal-tracker/` | `~/.config/multi-animal-tracker/` |
| Models | `~/Library/Application Support/multi-animal-tracker/models/` | `~/.local/share/multi-animal-tracker/models/` |
| Training runs | `~/Library/Application Support/multi-animal-tracker/training/` | `~/.local/share/multi-animal-tracker/training/` |

Default config presets and skeleton definitions are bundled with the package and seeded on first run.

### Migrating from an older repo checkout

If you had models in `<repo>/models/` or training data in `<repo>/training/`:

```bash
python -m multi_tracker.paths_migrate /path/to/repo --dry-run  # preview
python -m multi_tracker.paths_migrate /path/to/repo            # copy
```

---

## Updating

```bash
# pip users
pip install --upgrade multi-animal-tracker

# conda + pip users
conda activate multi-animal-tracker-mps  # or your env
git pull
mamba env update -f environment-mps.yml --prune
make install-mps
```

---

## Troubleshooting

### `CUDAExecutionProvider` missing from ONNX Runtime

- **pip path:** You need the CUDA toolkit installed system-wide, or switch to the conda path
- **conda path:** Re-run `make install-cuda CUDA_MAJOR=13` to refresh LD_LIBRARY_PATH hooks
- Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`

### `libcublasLt.so.12: cannot open shared object file`

CUDA 12 runtime libs not found. **conda path:** re-activate the environment (hooks set `LD_LIBRARY_PATH`). **pip path:** install the CUDA toolkit or switch to conda.

### PySide6 / Qt errors on Linux (pip path only)

```bash
sudo apt install libgl1-mesa-glx libegl1 libxcb-xinerama0  # Ubuntu/Debian
```

The conda path handles this automatically.

### Fresh reinstall

```bash
# pip
pip uninstall multi-animal-tracker && pip install multi-animal-tracker

# conda
conda env remove -n multi-animal-tracker-cuda
make setup-cuda
conda activate multi-animal-tracker-cuda
make install-cuda CUDA_MAJOR=13
```

---

## Related docs

- [Environments](environments.md) — full environment matrix
- [Integrations](integrations.md) — SLEAP, X-AnyLabeling setup
- `ROCM_SETUP.md` — ROCm system prerequisites
- [Publishing to PyPI](../developer-guide/publishing.md) — releasing new versions
