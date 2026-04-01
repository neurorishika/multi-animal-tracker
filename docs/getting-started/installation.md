# Installation

## Which method should I use?

| | pip | conda + pip |
|---|---|---|
| **Best for** | End users, deployment, CI | Developers, reproducible environments |
| **GPU inference** (YOLO, pose) | Yes | Yes |
| **TensorRT acceleration** | Yes | Yes |
| **ONNX Runtime GPU** | Yes | Yes |
| **CuPy GPU background subtraction** | Yes | Yes |
| **Requires git clone** | No | Yes |
| **Requires conda/mamba** | No | Yes |
| **CUDA runtime libs** | PyTorch pip wheel provides them | conda provides them |
| **LD_LIBRARY_PATH setup** | Not needed (PyTorch preloads CUDA libs) | Automatic via activation hooks |

Both methods are **functionally equivalent** for all GPU features. The conda path additionally manages system-level libraries (Qt, OpenGL) and provides reproducible environments for development.

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
# Step 1: Install PyTorch with CUDA from PyTorch's index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Step 2: Install multi-animal-tracker with CUDA extras
pip install "multi-animal-tracker[cuda]"
```

This installs everything: ONNX Runtime GPU, TensorRT, CuPy, and ONNX export tools.

**CUDA version variants:**

| Your CUDA | PyTorch index | Package extra |
|-----------|---------------|---------------|
| 12.6 | `cu126` | `[cuda]` or `[cuda12]` |
| 12.8 | `cu128` | `[cuda]` or `[cuda12]` |
| 13.0 | `cu130` | `[cuda13]` |

`[cuda]` is an alias for `[cuda12]`. Use `[cuda13]` explicitly for CUDA 13 systems:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install "multi-animal-tracker[cuda13]"
```

**Why does this work without system CUDA?** PyTorch's pip wheel bundles NVIDIA CUDA runtime libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) as pip dependencies. PyTorch preloads them at import time via `ctypes`, so ONNX Runtime and other CUDA consumers find them automatically in the same process — no `LD_LIBRARY_PATH` needed.

### AMD GPU (ROCm, Linux only)

Requires [ROCm 6.0+](https://rocm.docs.amd.com/) installed system-wide.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install "multi-animal-tracker[rocm]"
```

Note: TensorRT is NVIDIA-only and not included in the ROCm extra.

### Why is GPU install two commands?

PyTorch GPU builds are hosted on PyTorch's server, not PyPI. Python packaging (PEP 621) has no way to specify per-dependency index URLs. Every ML project handles this identically: install torch separately, then install the package. The second command will **not** overwrite your existing torch.

---

## conda + pip install (developer workflow)

This method uses conda for system libraries (Qt, OpenGL, CUDA toolkit) and pip for Python packages. Recommended for development because it provides:

- Reproducible environments with pinned system-level dependencies
- Editable install (`-e .`) for live code changes
- Makefile targets for common operations

### Step 1: Clone

```bash
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker
```

### Step 2: Create environment and install

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

# AMD GPU (ROCm — requires system-wide ROCm 6.0+)
make setup-rocm
conda activate multi-animal-tracker-rocm
make install-rocm
```

### How dependencies stay in sync

```
pyproject.toml [dependencies]          ← single source of truth for base deps
                                         (numpy, scipy, PySide6, ultralytics, ...)

pyproject.toml [optional-dependencies] ← GPU extras (onnxruntime-gpu, tensorrt, cupy, ...)
                                         Used by: pip install "multi-animal-tracker[cuda]"

requirements-*.txt                     ← adds ONLY what pyproject.toml can't:
  torch + --extra-index-url               - torch (needs custom index URL for GPU)
  -e .                                    - pulls all pyproject.toml deps
```

Adding a new base dependency means editing **one place**: `pyproject.toml`. The `requirements-*.txt` files inherit it via `-e .`.

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

User data is stored in platform-appropriate locations:

| Data | macOS | Linux |
|------|-------|-------|
| Config / presets | `~/Library/Application Support/multi-animal-tracker/` | `~/.config/multi-animal-tracker/` |
| Models | `~/Library/Application Support/multi-animal-tracker/models/` | `~/.local/share/multi-animal-tracker/models/` |
| Training runs | `~/Library/Application Support/multi-animal-tracker/training/` | `~/.local/share/multi-animal-tracker/training/` |

Default presets and skeleton definitions are bundled with the package and seeded on first run.

### Migrating from an older repo checkout

If you had models in `<repo>/models/` or training data in `<repo>/training/`:

```bash
python -m multi_tracker.paths_migrate /path/to/repo --dry-run  # preview
python -m multi_tracker.paths_migrate /path/to/repo            # copy
```

---

## Updating

```bash
# pip
pip install --upgrade multi-animal-tracker

# conda + pip
conda activate multi-animal-tracker-mps  # or your env
git pull
mamba env update -f environment-mps.yml --prune
make install-mps
```

---

## Troubleshooting

### `CUDAExecutionProvider` missing from ONNX Runtime

- Ensure you installed with `[cuda]` or `[cuda13]` extra (pip), or used `requirements-cuda*.txt` (conda)
- Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`
- **conda path:** re-run `make install-cuda CUDA_MAJOR=13` to refresh hooks

### PySide6 / Qt errors on Linux (pip only)

```bash
sudo apt install libgl1-mesa-glx libegl1 libxcb-xinerama0  # Ubuntu/Debian
```

The conda path handles this automatically.

### Fresh reinstall

```bash
# pip
pip uninstall multi-animal-tracker && pip install "multi-animal-tracker[cuda]"

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
