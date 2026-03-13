# FAQ

## Commands

### Which command should I use for tracking?

Use `mat` (shortcut) or `multi-animal-tracker`.

### Which command should I use for pose labeling?

Use `posekit-labeler` (canonical) or `pose`.

## Tracking and Detection

### Should I use background subtraction or YOLO OBB?

- Background subtraction for stable scenes with clear motion contrast.
- YOLO OBB for cluttered scenes or stationary targets.

### Why are tracks fragmented?

Common causes are strict assignment/lifecycle thresholds, low detector recall, or calibration mismatch in `reference_body_size`.

### How do I improve runtime speed?

Lower `resize_factor`, reduce non-essential visualization, and verify GPU backend/device selection.

## PoseKit + SLEAP

### MAT SLEAP works, but PoseKit falls back to native runtime. Why?

PoseKit keeps SLEAP ONNX/TensorRT behind an explicit opt-in toggle.

Enable:

- `Inference -> SLEAP -> Allow experimental SLEAP runtimes`

Then rerun prediction. If disabled, PoseKit intentionally reverts to native SLEAP runtime.

### Where do I set the SLEAP environment?

- MAT: `Analyze Individuals -> Pose Extraction -> SLEAP env`
- PoseKit: `Inference -> SLEAP -> Conda environment`

Use a dedicated SLEAP env (typically named with prefix `sleap`).

### How do I verify my SLEAP env quickly?

```bash
conda run -n sleap python -c "import importlib.util as u; print('sleap_nn', bool(u.find_spec('sleap_nn'))); print('onnx', bool(u.find_spec('onnx'))); print('onnxruntime', bool(u.find_spec('onnxruntime')))"
conda run -n sleap python -c "import torch, torchvision; print('torch', torch.__version__); print('torchvision', torchvision.__version__)"
```

### I get `operator torchvision::nms does not exist`. How do I fix it?

`torch` and `torchvision` are mismatched. Reinstall a matching pair from one channel/index.

```bash
conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio

# CPU-only
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# or CUDA 13.0
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
```

### I get `libtorch_cuda.so: undefined symbol: ncclAlltoAll`. How do I fix it?

This is usually a CUDA/NCCL mismatch.

```bash
conda run -n sleap python -m pip uninstall -y torch torchvision torchaudio nvidia-nccl-cu12 nvidia-nccl-cu13
conda run -n sleap python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision
conda run -n sleap python -m pip install --upgrade nvidia-nccl-cu13
```

If import only works when unsetting `LD_LIBRARY_PATH`, your shell is injecting incompatible CUDA/NCCL libs:

```bash
env -u LD_LIBRARY_PATH conda run -n sleap python -c "import torch; print(torch.cuda.is_available())"
```

### Where is the full SLEAP integration guide?

See [SLEAP Integration](../getting-started/integrations.md).
