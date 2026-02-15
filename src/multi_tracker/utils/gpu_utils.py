"""
GPU utilities and device detection for the multi-animal tracker.

This module provides centralized GPU availability detection and utilities
that can be used throughout the codebase. Supports:
  - CUDA (NVIDIA GPUs via CuPy)
  - MPS (Apple Silicon via PyTorch)
  - Automatic fallback to CPU

Import this module to check GPU availability:
    from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, MPS_AVAILABLE, GPU_AVAILABLE
"""

import logging
import sys
from importlib.util import find_spec

logger = logging.getLogger(__name__)

# CuPy for CUDA GPU acceleration (NVIDIA GPUs)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    CUPY_AVAILABLE = True
    # Test if CUDA is actually available (not just installed)
    try:
        device = cp.cuda.Device(0)
        CUDA_AVAILABLE = True
    except (cp.cuda.runtime.CUDARuntimeError, Exception):
        CUDA_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False
    cp = None
    cupy_ndimage = None

# PyTorch for MPS GPU acceleration (Apple Silicon) and CUDA/ROCm
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()

    # Detect ROCm backend (AMD GPUs)
    ROCM_AVAILABLE = False
    if TORCH_CUDA_AVAILABLE:
        try:
            # ROCm identifies itself in torch.version.hip or via device name
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                ROCM_AVAILABLE = True
            elif "gfx" in torch.cuda.get_device_name(0).lower():  # AMD GCN architecture
                ROCM_AVAILABLE = True
        except Exception:
            pass

except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False
    ROCM_AVAILABLE = False
    torch = None
    F = None

# Numba for CPU JIT acceleration
try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    njit = None
    prange = None

# TensorRT for NVIDIA GPU inference optimization
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = CUDA_AVAILABLE  # TensorRT requires CUDA
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None

# ONNX Runtime availability and provider capabilities.
try:
    import onnxruntime as ort

    ONNXRUNTIME_AVAILABLE = True
    try:
        ONNXRUNTIME_PROVIDERS = list(ort.get_available_providers() or [])
    except Exception:
        ONNXRUNTIME_PROVIDERS = []
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ONNXRUNTIME_PROVIDERS = []
    ort = None

ONNXRUNTIME_CPU_AVAILABLE = ONNXRUNTIME_AVAILABLE and (
    not ONNXRUNTIME_PROVIDERS or "CPUExecutionProvider" in ONNXRUNTIME_PROVIDERS
)
ONNXRUNTIME_CUDA_AVAILABLE = ONNXRUNTIME_AVAILABLE and any(
    p in ONNXRUNTIME_PROVIDERS
    for p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
)
ONNXRUNTIME_ROCM_AVAILABLE = ONNXRUNTIME_AVAILABLE and (
    "ROCMExecutionProvider" in ONNXRUNTIME_PROVIDERS
)


def _module_exists(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except Exception:
        return False


# SLEAP exported-runtime capabilities.
SLEAP_NN_EXPORT_AVAILABLE = bool(
    _module_exists("sleap_nn.export.predictors")
    and _module_exists("sleap_nn.export.metadata")
)
SLEAP_RUNTIME_ONNX_AVAILABLE = SLEAP_NN_EXPORT_AVAILABLE and ONNXRUNTIME_AVAILABLE
SLEAP_RUNTIME_TENSORRT_AVAILABLE = SLEAP_NN_EXPORT_AVAILABLE and TENSORRT_AVAILABLE

# Summary flags
GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE
ANY_ACCELERATION = GPU_AVAILABLE or NUMBA_AVAILABLE


def get_device_info() -> object:
    """
    Get information about available compute devices.

    Returns:
        dict: Device availability information
    """
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "mps_available": MPS_AVAILABLE,
        "rocm_available": ROCM_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "torch_cuda_available": TORCH_CUDA_AVAILABLE,
        "tensorrt_available": TENSORRT_AVAILABLE,
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "onnxruntime_providers": list(ONNXRUNTIME_PROVIDERS),
        "onnxruntime_cpu_available": ONNXRUNTIME_CPU_AVAILABLE,
        "onnxruntime_cuda_available": ONNXRUNTIME_CUDA_AVAILABLE,
        "onnxruntime_rocm_available": ONNXRUNTIME_ROCM_AVAILABLE,
        "sleap_nn_export_available": SLEAP_NN_EXPORT_AVAILABLE,
        "sleap_runtime_onnx_available": SLEAP_RUNTIME_ONNX_AVAILABLE,
        "sleap_runtime_tensorrt_available": SLEAP_RUNTIME_TENSORRT_AVAILABLE,
        "numba_available": NUMBA_AVAILABLE,
        "gpu_available": GPU_AVAILABLE,
        "any_acceleration": ANY_ACCELERATION,
    }

    # Add version information
    if CUPY_AVAILABLE:
        try:
            info["cupy_version"] = cp.__version__
        except Exception:
            pass

    if TORCH_AVAILABLE:
        try:
            info["torch_version"] = torch.__version__
        except Exception:
            pass

    if NUMBA_AVAILABLE:
        try:
            import numba

            info["numba_version"] = numba.__version__
        except Exception:
            pass

    if TENSORRT_AVAILABLE and trt is not None:
        try:
            info["tensorrt_version"] = trt.__version__
        except Exception:
            pass

    if ONNXRUNTIME_AVAILABLE and ort is not None:
        try:
            info["onnxruntime_version"] = ort.__version__
        except Exception:
            pass

    # Add device details if available
    if CUDA_AVAILABLE:
        try:
            info["cuda_device_count"] = cp.cuda.runtime.getDeviceCount()
            info["cuda_device_name"] = cp.cuda.Device(0).compute_capability
        except Exception:
            pass

    if MPS_AVAILABLE:
        info["mps_device"] = "Apple Silicon (Metal)"

    if TORCH_CUDA_AVAILABLE:
        try:
            info["torch_cuda_device_count"] = torch.cuda.device_count()
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(0)

            # Add ROCm-specific info
            if ROCM_AVAILABLE:
                if hasattr(torch.version, "hip"):
                    info["rocm_version"] = torch.version.hip
                info["backend"] = "ROCm (AMD GPU)"
            else:
                info["backend"] = "CUDA (NVIDIA GPU)"
        except Exception:
            pass

    return info


def log_device_info() -> object:
    """Log available compute devices to help with debugging."""
    info = get_device_info()

    logger.info("=" * 60)
    logger.info("Available Compute Devices:")
    logger.info("-" * 60)

    if info["cuda_available"]:
        logger.info("✓ CUDA (CuPy): Available")
        if "cuda_device_count" in info:
            logger.info(f"  Devices: {info['cuda_device_count']}")
    else:
        logger.info("✗ CUDA (CuPy): Not available")

    if info["mps_available"]:
        logger.info("✓ MPS (Apple Silicon): Available")
    else:
        logger.info("✗ MPS (Apple Silicon): Not available")

    if info["torch_cuda_available"]:
        logger.info("✓ CUDA (PyTorch): Available")
        if "torch_cuda_device_name" in info:
            logger.info(f"  Device: {info['torch_cuda_device_name']}")
        if info.get("rocm_available"):
            logger.info("  Backend: ROCm (AMD GPU)")
            if "rocm_version" in info:
                logger.info(f"  ROCm Version: {info['rocm_version']}")
        else:
            logger.info("  Backend: CUDA (NVIDIA GPU)")
    else:
        logger.info("✗ CUDA (PyTorch): Not available")

    if info["tensorrt_available"]:
        logger.info("✓ TensorRT (NVIDIA): Available")
        if trt is not None:
            logger.info(f"  Version: {trt.__version__}")
    else:
        logger.info("✗ TensorRT (NVIDIA): Not available")

    if info["onnxruntime_available"]:
        providers = ", ".join(info.get("onnxruntime_providers", [])) or "unknown"
        logger.info("✓ ONNX Runtime: Available")
        logger.info(f"  Providers: {providers}")
    else:
        logger.info("✗ ONNX Runtime: Not available")

    if info["sleap_nn_export_available"]:
        logger.info("✓ SLEAP export predictors: Available")
    else:
        logger.info("✗ SLEAP export predictors: Not available")

    if info["numba_available"]:
        logger.info("✓ Numba JIT: Available")
    else:
        logger.info("✗ Numba JIT: Not available")

    logger.info("-" * 60)
    if info["gpu_available"]:
        logger.info("GPU acceleration: ENABLED")
    elif info["numba_available"]:
        logger.info("CPU acceleration: Numba JIT")
    else:
        logger.info("CPU acceleration: NumPy only (slow)")
    logger.info("=" * 60)


def get_optimal_device(enable_gpu: object = True, prefer_cuda: object = True) -> object:
    """
    Get the optimal compute device based on availability.

    Args:
        enable_gpu: Whether to use GPU if available
        prefer_cuda: Prefer CUDA over MPS if both available

    Returns:
        tuple: (device_type, device_object)
            device_type: 'cuda', 'mps', or 'cpu'
            device_object: GPU device object or None for CPU
    """
    if not enable_gpu:
        return "cpu", None

    # Prefer CUDA if requested and available
    if prefer_cuda and CUDA_AVAILABLE:
        return "cuda", cp.cuda.Device(0)

    # Then try MPS
    if MPS_AVAILABLE:
        return "mps", torch.device("mps")

    # Then CUDA if not already tried
    if not prefer_cuda and CUDA_AVAILABLE:
        return "cuda", cp.cuda.Device(0)

    # Fallback to CPU
    return "cpu", None


def get_pose_runtime_options(backend_family: str = "yolo"):
    """
    Return runtime options for pose inference as list[(label, value)].

    Values are normalized ids consumed by runtime_api, e.g.:
    - auto
    - cpu / mps / cuda / rocm
    - onnx_cpu / onnx_cuda
    - tensorrt_cuda
    """
    backend = str(backend_family or "yolo").strip().lower()
    is_mac = sys.platform == "darwin"
    options = [("Auto", "auto")]

    supports_onnx = (
        ONNXRUNTIME_AVAILABLE if backend != "sleap" else SLEAP_RUNTIME_ONNX_AVAILABLE
    )
    supports_tensorrt = (
        TENSORRT_AVAILABLE if backend != "sleap" else SLEAP_RUNTIME_TENSORRT_AVAILABLE
    )
    cuda_like = CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE

    if is_mac:
        # macOS: expose CPU/MPS native options and ONNX CPU only.
        if MPS_AVAILABLE:
            options.append(("MPS", "mps"))
        options.append(("CPU", "cpu"))
        if supports_onnx:
            options.append(("ONNX (CPU)", "onnx_cpu"))
    else:
        options.append(("CPU", "cpu"))
        if TORCH_CUDA_AVAILABLE and not ROCM_AVAILABLE:
            options.append(("CUDA", "cuda"))
        if ROCM_AVAILABLE:
            options.append(("ROCm", "rocm"))

        if supports_onnx:
            options.append(("ONNX (CPU)", "onnx_cpu"))
            if (
                ONNXRUNTIME_CUDA_AVAILABLE
                and TORCH_CUDA_AVAILABLE
                and not ROCM_AVAILABLE
            ):
                options.append(("ONNX (CUDA)", "onnx_cuda"))
            if ONNXRUNTIME_ROCM_AVAILABLE and ROCM_AVAILABLE:
                options.append(("ONNX (ROCm)", "onnx_rocm"))

        if supports_tensorrt and cuda_like:
            options.append(("TensorRT (CUDA)", "tensorrt_cuda"))

    # Deduplicate by value while preserving order.
    seen = set()
    deduped = []
    for label, value in options:
        if value in seen:
            continue
        seen.add(value)
        deduped.append((label, value))
    return deduped


# Export all availability flags and utilities
__all__ = [
    # Flags
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "ROCM_AVAILABLE",
    "TORCH_CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "TENSORRT_AVAILABLE",
    "ONNXRUNTIME_AVAILABLE",
    "ONNXRUNTIME_PROVIDERS",
    "ONNXRUNTIME_CPU_AVAILABLE",
    "ONNXRUNTIME_CUDA_AVAILABLE",
    "ONNXRUNTIME_ROCM_AVAILABLE",
    "SLEAP_NN_EXPORT_AVAILABLE",
    "SLEAP_RUNTIME_ONNX_AVAILABLE",
    "SLEAP_RUNTIME_TENSORRT_AVAILABLE",
    "NUMBA_AVAILABLE",
    "GPU_AVAILABLE",
    "ANY_ACCELERATION",
    # Modules (may be None)
    "cp",
    "cupy_ndimage",
    "torch",
    "F",
    "njit",
    "prange",
    "trt",
    "ort",
    # Functions
    "get_device_info",
    "log_device_info",
    "get_optimal_device",
    "get_pose_runtime_options",
]
