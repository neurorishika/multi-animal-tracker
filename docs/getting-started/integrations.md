# Integrations

## SLEAP Integration (MAT + PoseKit)

SLEAP inference is executed from the **SLEAP conda/mamba environment selected in the UI**.

### Why this env is separate

Do not rely on the main MAT environment for SLEAP export/runtime dependencies.
In practice, the `uv`-based MAT install path does not guarantee the SLEAP runtime/export modules needed for all SLEAP backends.

Create a dedicated SLEAP environment and select it from:

- MAT: `Analyze Individuals -> Pose Extraction -> SLEAP env`
- PoseKit: `Inference -> SLEAP -> Conda environment`

### Environment Setup

```bash
mamba create -n sleap python=3.13 -y
conda activate sleap
```

Choose one install profile:

```bash
# CPU + ONNX CPU export/runtime support
pip install "sleap[nn,nn-export]"

# GPU + ONNX GPU export/runtime support
pip install "sleap[nn,nn-export-gpu]"

# GPU + TensorRT export/runtime support
pip install "sleap[nn,nn-export-gpu,nn-tensorrt]"
```

### Compatibility Matrix

- **macOS (Apple Silicon)**:
  - Supported: SLEAP native (`mps`/`cpu`), ONNX CPU.
  - Not supported: TensorRT.
- **NVIDIA CUDA systems**:
  - Supported: SLEAP native CUDA, ONNX GPU, TensorRT (with `nn-tensorrt` extras and system CUDA/TensorRT compatibility).
- **CPU-only systems**:
  - Supported: SLEAP native CPU, ONNX CPU.

### Verify SLEAP Integration

```bash
conda run -n sleap python -c "import importlib.util as u; print('sleap_nn', bool(u.find_spec('sleap_nn'))); print('onnx', bool(u.find_spec('onnx'))); print('onnxruntime', bool(u.find_spec('onnxruntime')))"
conda run -n sleap sleap-nn export --help
```

For ONNX export smoke test:

```bash
conda run -n sleap sleap-nn export "/path/to/sleap_model_dir" --output "/tmp/sleap_export_test_onnx" --format onnx --device cpu --input-height 224 --input-width 224
```

For ONNX predictor smoke test:

```bash
conda run -n sleap python -c "from sleap_nn.export.predictors import load_exported_model as L; p=L('/tmp/sleap_export_test_onnx/model.onnx', runtime='onnx', providers=['CPUExecutionProvider']); print(type(p).__name__)"
```

See also:

- [Compute Runtimes (User Guide)](../user-guide/compute-runtimes.md)
- [Runtime Integration (Developer Guide)](../developer-guide/runtime-integration.md)
