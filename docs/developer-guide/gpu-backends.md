# GPU Backends

## Backend Detection

`hydra_suite.utils.gpu_utils` centralizes runtime capability checks for:

- CUDA (CuPy/PyTorch)
- MPS (PyTorch)
- ROCm (PyTorch/CuPy variants where configured)

## Where GPU Is Used

- YOLO inference paths (`hydra_suite.core.detectors.engine`)
- Background operations in supported acceleration paths
- Batch sizing heuristics (`hydra_suite.utils.batch_optimizer`)

## Operational Notes

- Keep device selection explicit in reproducibility-critical runs.
- Use `auto` for general user workflows.
- TensorRT is optional and environment-sensitive.
