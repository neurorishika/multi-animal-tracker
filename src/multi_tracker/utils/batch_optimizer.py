"""
Batch size optimizer for YOLO detection based on available device memory.
"""

import logging
import numpy as np
from .gpu_utils import TORCH_CUDA_AVAILABLE, MPS_AVAILABLE, torch

logger = logging.getLogger(__name__)


class BatchOptimizer:
    """Optimize batch size for YOLO inference based on device capabilities."""

    def __init__(self, advanced_config=None):
        """
        Initialize batch optimizer.

        Args:
            advanced_config: Dictionary with memory allocation settings
        """
        self.advanced_config = advanced_config or {}
        self.device_type = None
        self.device_name = None
        self.available_memory = None

    def detect_device(self):
        """
        Detect available compute device and its memory.

        Returns:
            tuple: (device_type, device_name, available_memory_mb)
        """
        if torch is None:
            self.device_type = "cpu"
            self.device_name = "CPU (PyTorch not available)"
            self.available_memory = 0
            logger.warning("PyTorch not available - batching disabled")
            return

        if TORCH_CUDA_AVAILABLE:
            self.device_type = "cuda"
            self.device_name = torch.cuda.get_device_name(0)
            # Get available memory (free memory, not total)
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            self.available_memory = free_memory / (1024**2)  # Convert to MB
            logger.info(f"CUDA device detected: {self.device_name}")
            logger.info(
                f"Available VRAM: {self.available_memory:.0f} MB / {total_memory / (1024 ** 2):.0f} MB"
            )

        elif MPS_AVAILABLE:
            self.device_type = "mps"
            self.device_name = "Apple Silicon (MPS)"
            # MPS uses unified memory - estimate conservatively
            # Get system memory as approximation
            try:
                import psutil

                available_memory = psutil.virtual_memory().available / (1024**2)
                # Use only a conservative fraction for MPS (30% default, configurable)
                mps_fraction = self.advanced_config.get("mps_memory_fraction", 0.3)
                self.available_memory = available_memory * mps_fraction
                logger.info(f"MPS device detected: {self.device_name}")
                logger.info(
                    f"Available unified memory (conservative): {self.available_memory:.0f} MB ({mps_fraction*100:.0f}% of {available_memory:.0f} MB)"
                )
            except ImportError:
                # Fallback if psutil not available
                self.available_memory = 2048  # Conservative 2GB default
                logger.warning(
                    "psutil not available, using conservative 2GB estimate for MPS"
                )

        else:
            self.device_type = "cpu"
            self.device_name = "CPU"
            self.available_memory = 0  # CPU doesn't benefit from batching
            logger.info("CPU device detected - batching disabled")

        return (self.device_type, self.device_name, self.available_memory)

    def estimate_batch_size(
        self, frame_width, frame_height, model_name="yolo26s-obb.pt"
    ):
        """
        Estimate optimal batch size for YOLO inference.

        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            model_name: YOLO model name (for memory estimation)

        Returns:
            int: Recommended batch size (1 if batching not recommended)
        """
        # Auto-detect device if not already done
        if self.device_type is None:
            self.detect_device()

        # CPU never batches
        if self.device_type == "cpu":
            logger.info("CPU device: batch size = 1 (no batching)")
            return 1

        # Check if batching is enabled in config
        if not self.advanced_config.get("enable_yolo_batching", True):
            logger.info("YOLO batching disabled in config: batch size = 1")
            return 1

        # Check for manual override
        batch_mode = self.advanced_config.get("yolo_batch_size_mode", "auto")
        if batch_mode == "manual":
            manual_size = self.advanced_config.get("yolo_manual_batch_size", 16)
            logger.info(f"Manual batch size override: {manual_size}")
            return max(1, min(manual_size, 64))  # Clamp to 1-64

        # Estimate memory per frame
        # Formula: width × height × 3 (RGB) × 4 (float32) + model overhead
        frame_memory_mb = (frame_width * frame_height * 3 * 4) / (1024**2)

        # Model memory overhead (rough estimates based on model size)
        model_overhead = {
            "yolo26n": 20,  # Nano: ~20MB
            "yolo26s": 50,  # Small: ~50MB
            "yolo26m": 120,  # Medium: ~120MB
            "yolo26l": 200,  # Large: ~200MB
            "yolo11n": 20,
            "yolo11s": 50,
            "yolo11m": 120,
        }

        # Extract model size from name
        overhead_mb = 50  # Default
        for key, value in model_overhead.items():
            if key in model_name.lower():
                overhead_mb = value
                break

        # Get memory fraction based on device
        if self.device_type == "cuda":
            memory_fraction = self.advanced_config.get("cuda_memory_fraction", 0.7)
        else:  # MPS
            memory_fraction = self.advanced_config.get("mps_memory_fraction", 0.3)

        # Calculate usable memory
        usable_memory = self.available_memory * memory_fraction

        # Estimate batch size with safety margin
        memory_per_frame_total = (
            frame_memory_mb * 2.5
        )  # 2.5x for YOLO processing overhead
        available_for_batch = usable_memory - overhead_mb

        if available_for_batch <= 0:
            logger.warning(
                f"Insufficient memory for batching: {usable_memory:.0f} MB available, {overhead_mb} MB model overhead"
            )
            return 1

        estimated_batch = int(available_for_batch / memory_per_frame_total)

        # Apply safety margin (0.8x) and clamp to reasonable range
        safe_batch = int(estimated_batch * 0.8)
        batch_size = max(1, min(safe_batch, 64))

        logger.info(f"Batch size estimation:")
        logger.info(f"  Device: {self.device_type.upper()} ({self.device_name})")
        logger.info(f"  Frame size: {frame_width}×{frame_height}")
        logger.info(f"  Available memory: {self.available_memory:.0f} MB")
        logger.info(f"  Memory fraction: {memory_fraction*100:.0f}%")
        logger.info(f"  Usable memory: {usable_memory:.0f} MB")
        logger.info(f"  Model overhead: {overhead_mb} MB")
        logger.info(f"  Memory per frame: {memory_per_frame_total:.1f} MB")
        logger.info(f"  Estimated batch size: {estimated_batch}")
        logger.info(f"  Safe batch size: {batch_size}")

        return batch_size

    def get_device_info(self):
        """
        Get human-readable device information.

        Returns:
            dict: Device information for display
        """
        if self.device_type is None:
            self.detect_device()

        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "available_memory_mb": self.available_memory,
            "batching_supported": self.device_type in ["cuda", "mps"],
        }
