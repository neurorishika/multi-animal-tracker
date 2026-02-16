# Compute Runtimes

This page explains how global runtime selection works in MAT and PoseKit.

## One Global Runtime

Runtime is selected once via `compute_runtime`.

- MAT: `Get Started -> Performance -> Compute runtime`
- PoseKit: `Inference -> Runtime`

The same runtime drives:

- YOLO-OBB detection
- YOLO-pose inference
- SLEAP inference

## Runtime Options

- `cpu`
- `mps`
- `cuda`
- `rocm`
- `onnx_cpu`
- `onnx_cuda`
- `onnx_rocm`
- `tensorrt`

The UI only shows options valid for the currently enabled pipelines.

## Intersection Gating (Important)

Available runtimes are the **intersection** of support across enabled pipelines.

Examples:

- If tracking uses YOLO-OBB + YOLO-pose, runtime must be valid for both.
- If pose backend is switched to SLEAP, runtime options are recomputed immediately.
- Invalid saved runtime is automatically reset to a valid one.

## What Each Runtime Means

- `cpu`: CPU inference paths.
- `mps`: Apple Metal (PyTorch-native paths).
- `cuda`: PyTorch CUDA native paths.
- `rocm`: PyTorch ROCm native paths.
- `onnx_cpu`: ONNX Runtime with CPU provider.
- `onnx_cuda`: ONNX Runtime with CUDA provider.
- `onnx_rocm`: ONNX Runtime with ROCm provider.
- `tensorrt`: TensorRT engine/runtime on NVIDIA CUDA.

## Auto Export and Artifact Location

Export artifacts are generated automatically when needed.

- YOLO model `/path/model.pt`:
  - ONNX: `/path/model.onnx`
  - TensorRT: `/path/model.engine`
- SLEAP model directory `/path/sleap_model_dir`:
  - ONNX export directory: `/path/sleap_model_dir.onnx`
  - TensorRT export directory: `/path/sleap_model_dir.tensorrt`

No manual exported-model-path entry is required.

## Platform Expectations

- Apple Silicon:
  - Typically `mps` and `onnx_cpu` (no TensorRT).
- NVIDIA CUDA:
  - `cuda`, `onnx_cuda`, and `tensorrt` when installed correctly.
- ROCm:
  - `rocm` and `onnx_rocm` when providers are available.

## Behavior During Runs

Runtime controls are locked during long-running prediction/tracking tasks to prevent backend switching crashes.

## Troubleshooting

- Runtime disappeared from dropdown:
  - A selected backend/pipeline no longer supports it.
- ONNX shown but fails at runtime:
  - Check provider availability and model export compatibility.
- SLEAP ONNX/TensorRT unavailable:
  - Verify selected SLEAP environment and export dependencies.

See also: [Integrations](../getting-started/integrations.md), [Troubleshooting](troubleshooting.md)
