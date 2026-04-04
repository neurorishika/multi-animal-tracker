"""Canonical compute runtime capability and translation helpers.

This module defines one user-facing runtime enum and translates it into
backend-specific settings for detection and pose inference.
"""

from __future__ import annotations

import shutil
from typing import Iterable, List

from hydra_suite.utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ONNXRUNTIME_CPU_AVAILABLE,
    ONNXRUNTIME_CUDA_AVAILABLE,
    ONNXRUNTIME_ROCM_AVAILABLE,
    ROCM_AVAILABLE,
    SLEAP_RUNTIME_TENSORRT_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

CANONICAL_RUNTIMES: List[str] = [
    "cpu",
    "mps",
    "cuda",
    "rocm",
    "onnx_cpu",
    "onnx_cuda",
    "onnx_rocm",
    "tensorrt",
]


def _best_explicit_onnx_runtime() -> str:
    if ONNXRUNTIME_ROCM_AVAILABLE and ROCM_AVAILABLE:
        return "onnx_rocm"
    if ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available() and not ROCM_AVAILABLE:
        return "onnx_cuda"
    if ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE:
        return "onnx_cpu"
    return "onnx_cpu"


def _normalize_runtime(runtime: str) -> str:
    rt = str(runtime or "cpu").strip().lower().replace("-", "_")
    if rt in {"", "auto"}:
        # Canonical runtime set intentionally excludes auto.
        # Default to CPU for deterministic fallback.
        return "cpu"
    if rt == "onnxruntime":
        return _best_explicit_onnx_runtime()
    if rt in {"trt", "tensor_rt"}:
        return "tensorrt"
    if rt == "onnx":
        return _best_explicit_onnx_runtime()
    if rt == "onnx_gpu":
        if ONNXRUNTIME_ROCM_AVAILABLE and ROCM_AVAILABLE:
            return "onnx_rocm"
        return "onnx_cuda"
    if rt in {"onnx_cpu", "onnx_cuda", "onnx_rocm"}:
        return rt
    if rt.startswith("tensorrt"):
        return "tensorrt"
    if rt.startswith("cuda"):
        return "cuda"
    if rt.startswith("rocm"):
        return "rocm"
    return rt if rt in CANONICAL_RUNTIMES else "cpu"


def runtime_label(runtime: str) -> str:
    rt = _normalize_runtime(runtime)
    return {
        "cpu": "CPU",
        "mps": "MPS",
        "cuda": "CUDA",
        "rocm": "ROCm",
        "onnx_cpu": "ONNX (CPU)",
        "onnx_cuda": "ONNX (CUDA)",
        "onnx_rocm": "ONNX (ROCm)",
        "tensorrt": "TensorRT",
    }[rt]


def _cuda_like_available() -> bool:
    return bool(CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE)


def _onnx_available(rt: str) -> bool:
    """Check local ONNX runtime availability for the given canonical ONNX runtime."""
    if rt == "onnx_cpu":
        return bool(ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE)
    if rt == "onnx_cuda":
        return bool(
            ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available() and not ROCM_AVAILABLE
        )
    if rt == "onnx_rocm":
        return bool(ONNXRUNTIME_ROCM_AVAILABLE and ROCM_AVAILABLE)
    return False


def _tensorrt_available() -> bool:
    return bool(TENSORRT_AVAILABLE and _cuda_like_available() and not ROCM_AVAILABLE)


def _sleap_onnx_available(rt: str) -> bool:
    """SLEAP ONNX runs in the selected SLEAP conda env, not the MAT env."""
    if not shutil.which("conda"):
        return False
    if rt == "onnx_cpu":
        return True
    if rt == "onnx_cuda":
        return bool(_cuda_like_available() and not ROCM_AVAILABLE)
    if rt == "onnx_rocm":
        return bool(ROCM_AVAILABLE)
    return False


def _pipeline_supports_runtime(pipeline: str, runtime: str) -> bool:
    p = str(pipeline or "").strip().lower()
    rt = _normalize_runtime(runtime)

    # Baseline runtime support independent of pipeline.
    if rt == "cpu":
        return True
    if rt == "mps":
        return bool(MPS_AVAILABLE)
    if rt == "cuda":
        return bool(_cuda_like_available() and not ROCM_AVAILABLE)
    if rt == "rocm":
        return bool(ROCM_AVAILABLE)

    # SLEAP has its own ONNX/TRT availability logic.
    if p == "sleap_pose":
        if rt in {"onnx_cpu", "onnx_cuda", "onnx_rocm"}:
            return _sleap_onnx_available(rt)
        if rt == "tensorrt":
            return bool(
                (SLEAP_RUNTIME_TENSORRT_AVAILABLE or TENSORRT_AVAILABLE)
                and _cuda_like_available()
                and not ROCM_AVAILABLE
            )
        return True

    # All other pipelines use the standard ONNX/TRT availability checks.
    if rt in {"onnx_cpu", "onnx_cuda", "onnx_rocm"}:
        return _onnx_available(rt)
    if rt == "tensorrt":
        return _tensorrt_available()
    return True


def supported_runtimes_for_pipeline(pipeline: str) -> List[str]:
    """Return canonical runtimes supported for a single pipeline."""
    return [rt for rt in CANONICAL_RUNTIMES if _pipeline_supports_runtime(pipeline, rt)]


def allowed_runtimes_for_pipelines(pipelines: Iterable[str]) -> List[str]:
    """Return canonical runtimes allowed for all provided pipelines.

    If no pipelines are provided, returns host-capable runtimes from canonical set.
    """
    pls = [str(p).strip().lower() for p in pipelines if str(p).strip()]
    if not pls:
        return [
            rt for rt in CANONICAL_RUNTIMES if _pipeline_supports_runtime("generic", rt)
        ]

    allowed = []
    for rt in CANONICAL_RUNTIMES:
        if all(_pipeline_supports_runtime(p, rt) for p in pls):
            allowed.append(rt)
    return allowed


def _best_auto_runtime() -> str:
    """Pick the best available canonical runtime for auto-detection."""
    if _tensorrt_available():
        return "tensorrt"
    if MPS_AVAILABLE:
        return "mps"
    if ROCM_AVAILABLE:
        return "rocm"
    if _cuda_like_available():
        return "cuda"
    if ONNXRUNTIME_ROCM_AVAILABLE and ROCM_AVAILABLE:
        return "onnx_rocm"
    if ONNXRUNTIME_CUDA_AVAILABLE and _cuda_like_available() and not ROCM_AVAILABLE:
        return "onnx_cuda"
    if ONNXRUNTIME_CPU_AVAILABLE or ONNXRUNTIME_AVAILABLE:
        return "onnx_cpu"
    return "cpu"


def _runtime_from_pose_flavor(pose_runtime_flavor: str) -> str | None:
    """Map legacy pose_runtime_flavor to canonical runtime, or None."""
    pr = str(pose_runtime_flavor or "").strip().lower()
    if pr.startswith("tensorrt"):
        return "tensorrt"
    if pr.startswith("onnx_rocm"):
        return "onnx_rocm"
    if pr.startswith("onnx_cuda"):
        return "onnx_cuda"
    if pr.startswith("onnx"):
        return "onnx_cpu"
    return None


_DEVICE_MAP = {
    "mps": "mps",
    "rocm": "rocm",
    "cpu": "cpu",
}
_CUDA_DEVICES = {"cuda", "cuda:0", "gpu"}


def infer_compute_runtime_from_legacy(
    yolo_device: str,
    enable_tensorrt: bool,
    pose_runtime_flavor: str,
) -> str:
    """Infer canonical runtime from legacy config fields."""
    if bool(enable_tensorrt):
        return "tensorrt"

    from_pose = _runtime_from_pose_flavor(pose_runtime_flavor)
    if from_pose is not None:
        return from_pose

    dev = str(yolo_device or "auto").strip().lower()
    if dev in _DEVICE_MAP:
        return _DEVICE_MAP[dev]
    if dev in _CUDA_DEVICES:
        return "rocm" if ROCM_AVAILABLE else "cuda"

    return _best_auto_runtime()


def derive_detection_runtime_settings(compute_runtime: str) -> dict:
    """Map canonical runtime to OBB detection legacy settings."""
    rt = _normalize_runtime(compute_runtime)

    yolo_device = "cpu"
    enable_tensorrt = False
    enable_onnx_runtime = False

    if rt == "mps":
        yolo_device = "mps"
    elif rt in {"cuda", "rocm"}:
        yolo_device = "cuda:0"
    elif rt == "tensorrt":
        yolo_device = "cuda:0"
        enable_tensorrt = True
    elif rt == "onnx_cpu":
        yolo_device = "cpu"
        enable_onnx_runtime = True
    elif rt in {"onnx_cuda", "onnx_rocm"}:
        yolo_device = "cuda:0"
        enable_onnx_runtime = True

    return {
        "yolo_device": yolo_device,
        "enable_tensorrt": bool(enable_tensorrt),
        "enable_onnx_runtime": bool(enable_onnx_runtime),
        "enable_gpu_background": yolo_device != "cpu",
    }


def derive_pose_runtime_settings(compute_runtime: str, backend_family: str) -> dict:
    """Map canonical runtime to pose runtime legacy settings consumed by runtime_api."""
    rt = _normalize_runtime(compute_runtime)

    if rt == "cpu":
        return {"pose_runtime_flavor": "cpu", "pose_sleap_device": "cpu"}
    if rt == "mps":
        return {"pose_runtime_flavor": "mps", "pose_sleap_device": "mps"}
    if rt == "cuda":
        return {"pose_runtime_flavor": "cuda", "pose_sleap_device": "cuda:0"}
    if rt == "rocm":
        return {"pose_runtime_flavor": "rocm", "pose_sleap_device": "cuda:0"}
    if rt == "tensorrt":
        return {
            "pose_runtime_flavor": "tensorrt_cuda",
            "pose_sleap_device": "cuda:0",
        }

    if rt == "onnx_cpu":
        return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
    if rt in {"onnx_cuda", "onnx_rocm"}:
        return {"pose_runtime_flavor": rt, "pose_sleap_device": "cuda:0"}

    # Legacy alias fallback (e.g. compute_runtime="onnx").
    resolved = _best_explicit_onnx_runtime()
    if resolved == "onnx_cpu":
        return {"pose_runtime_flavor": "onnx_cpu", "pose_sleap_device": "cpu"}
    return {"pose_runtime_flavor": resolved, "pose_sleap_device": "cuda:0"}
