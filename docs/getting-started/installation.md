# Installation

Multi-Animal-Tracker supports two installation methods:

| Method | Best for | GPU support | System lib setup |
|--------|----------|-------------|-----------------|
| **pip** | End users, quick start | Yes (manual torch step) | User handles |
| **conda + pip** | Developers, full GPU stack | Yes (automatic) | conda handles |

Both methods produce the same working application. Choose one — don't mix them.

---

## Method 1: pip install

Requires only Python 3.11+ — no conda, no git clone.

### CPU (one command)

```bash
pip install multi-animal-tracker
```

### Apple Silicon / MPS (macOS M1-M4)

```bash
pip install torch torchvision
pip install "multi-animal-tracker[mps]"
```

MPS support is built into the standard PyTorch macOS wheel.

### NVIDIA GPU (CUDA)

```bash
# Step 1: Install PyTorch with CUDA from PyTorch's index
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130  # CUDA 13.0

# Step 2: Install multi-animal-tracker with CUDA extras
pip install "multi-animal-tracker[cuda]"
```

### AMD GPU (ROCm, Linux only)

Requires [ROCm 6.0+](https://rocm.docs.amd.com/) installed system-wide first.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install "multi-animal-tracker[rocm]"
```

### Why two commands for GPU?

PyTorch GPU builds are hosted on PyTorch's own server, not PyPI. Python packaging standards (PEP 621) have no way to specify per-dependency index URLs, so every ML project handles this the same way: you install torch separately, then install the package. The second `pip install` step will **not** overwrite your existing torch.

### Verify

```bash
mat --help                        # Should print usage
python -c "import torch; print(torch.cuda.is_available())"   # GPU check
```

### Launch

```bash
mat                # Multi-Animal-Tracker GUI
posekit-labeler    # PoseKit pose-labeling GUI
datasieve          # DataSieve tool
classkit           # ClassKit labeler
afterhours         # Afterhours proofreading
```

---

## Method 2: conda + pip (developer / full GPU)

This method uses conda for system libraries (Qt, OpenGL, CUDA toolkit) and pip for Python packages. It gives you:

- Automatic CUDA/ROCm library management
- `LD_LIBRARY_PATH` hooks for ONNX Runtime compatibility
- TensorRT support
- Editable install for development

### Step 1: Clone

```bash
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker
```

### Step 2: Create conda environment

Pick your platform:

```bash
make setup            # CPU
make setup-mps        # Apple Silicon
make setup-cuda       # NVIDIA GPU
make setup-rocm       # AMD GPU (ROCm 6.0+ must be installed system-wide first)
```

### Step 3: Activate and install

```bash
# CPU
conda activate multi-animal-tracker
make install

# Apple Silicon
conda activate multi-animal-tracker-mps
make install-mps

# NVIDIA GPU
conda activate multi-animal-tracker-cuda
make install-cuda CUDA_MAJOR=13     # or CUDA_MAJOR=12

# AMD GPU
conda activate multi-animal-tracker-rocm
make install-rocm
```

### How it works

The conda environment provides system libraries and Python. The `make install-*` targets run `uv pip install -r requirements-*.txt`, which installs:

1. **torch** with the correct GPU variant and index URL
2. **GPU-specific packages** (TensorRT, CuPy, ONNX Runtime GPU)
3. **`-e .`** — an editable install of multi-animal-tracker itself, which pulls all base dependencies from `pyproject.toml`

Base dependencies (numpy, scipy, PySide6, ultralytics, etc.) are declared once in `pyproject.toml`. The requirements files only add what `pyproject.toml` cannot express (torch index URLs, GPU-specific wheels). This means there is **one source of truth** for dependencies — `pyproject.toml` — with platform-specific overlays in `requirements-*.txt`.

### ONNX Runtime note (CUDA)

`onnxruntime-gpu` currently links against CUDA 12 user-space libraries. On CUDA 13 systems, the conda environment provides compatible CUDA 12 libs, and `make install-cuda` configures `LD_LIBRARY_PATH` activation hooks. This is expected and handled automatically.

---

## After installation

### Data directories

User data is stored in platform-appropriate locations (not inside the repo or package):

| Data | macOS | Linux |
|------|-------|-------|
| Config / presets | `~/Library/Application Support/multi-animal-tracker/` | `~/.config/multi-animal-tracker/` |
| Models | `~/Library/Application Support/multi-animal-tracker/models/` | `~/.local/share/multi-animal-tracker/models/` |
| Training runs | `~/Library/Application Support/multi-animal-tracker/training/` | `~/.local/share/multi-animal-tracker/training/` |

Default config presets and skeleton definitions are bundled with the package and seeded to your config directory on first run.

### Migrating from an older repo checkout

If you previously used a cloned repo with models in `<repo>/models/` and training data in `<repo>/training/`:

```bash
python -m multi_tracker.paths_migrate /path/to/multi-animal-tracker --dry-run  # preview
python -m multi_tracker.paths_migrate /path/to/multi-animal-tracker            # copy
```

---

## Updating

### pip users

```bash
pip install --upgrade multi-animal-tracker
```

### conda + pip users

```bash
conda activate multi-animal-tracker-mps  # or your env
git pull
mamba env update -f environment-mps.yml --prune
make install-mps                         # or install / install-cuda / install-rocm
```

---

## Troubleshooting

### `CUDAExecutionProvider` missing from ONNX Runtime

- Ensure you used `requirements-cuda12.txt` or `requirements-cuda13.txt` (not CPU)
- Ensure you are in the `multi-animal-tracker-cuda` conda environment
- Re-run `make install-cuda CUDA_MAJOR=13` (or `12`) to refresh linker hooks
- Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`

### `libcublasLt.so.12: cannot open shared object file`

CUDA 12 user-space libs are not visible. Re-activate the conda environment (the activation hook sets `LD_LIBRARY_PATH`).

### PySide6 / Qt errors on Linux

pip-installed PySide6 may need system libraries:

```bash
sudo apt install libgl1-mesa-glx libegl1 libxcb-xinerama0  # Ubuntu/Debian
```

The conda install method handles this automatically.

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
- [Publishing to PyPI](../developer-guide/publishing.md) — how to release new versions
