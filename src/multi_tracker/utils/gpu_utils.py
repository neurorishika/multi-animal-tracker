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

logger = logging.getLogger(__name__)

# CuPy for CUDA GPU acceleration (NVIDIA GPUs)
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cupy_ndimage

    CUPY_AVAILABLE = True
    # Test if CUDA is actually available (not just installed)
    try:
        _ = cp.cuda.Device(0)
        CUDA_AVAILABLE = True
    except (cp.cuda.runtime.CUDARuntimeError, Exception):
        CUDA_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False
    CUDA_AVAILABLE = False
    cp = None
    cupy_ndimage = None

# PyTorch for MPS GPU acceleration (Apple Silicon) and CUDA
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False
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

# Summary flags
GPU_AVAILABLE = CUDA_AVAILABLE or MPS_AVAILABLE
ANY_ACCELERATION = GPU_AVAILABLE or NUMBA_AVAILABLE


def get_device_info():
    """
    Get information about available compute devices.

    Returns:
        dict: Device availability information
    """
    info = {
        "cuda_available": CUDA_AVAILABLE,
        "mps_available": MPS_AVAILABLE,
        "torch_cuda_available": TORCH_CUDA_AVAILABLE,
        "numba_available": NUMBA_AVAILABLE,
        "gpu_available": GPU_AVAILABLE,
        "any_acceleration": ANY_ACCELERATION,
    }

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
        except Exception:
            pass

    return info


def log_device_info():
    """Log available compute devices to help with debugging."""
    info = get_device_info()

    logger.info("=" * 60)
    logger.info("Available Compute Devices:")
    logger.info("-" * 60)

    if info["cuda_available"]:
        logger.info(f"✓ CUDA (CuPy): Available")
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
    else:
        logger.info("✗ CUDA (PyTorch): Not available")

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


def get_optimal_device(enable_gpu=True, prefer_cuda=True):
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


# Export all availability flags and utilities
__all__ = [
    # Flags
    "CUDA_AVAILABLE",
    "MPS_AVAILABLE",
    "TORCH_CUDA_AVAILABLE",
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
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
    # Functions
    "get_device_info",
    "log_device_info",
    "get_optimal_device",
]
