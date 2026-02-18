"""
Core data types and interfaces for identity runtime backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np


@dataclass
class PoseResult:
    """Canonical pose output for one crop."""

    keypoints: Optional[np.ndarray]
    mean_conf: float
    valid_fraction: float
    num_valid: int
    num_keypoints: int


@dataclass
class AppearanceResult:
    """Canonical appearance embedding output for one crop."""

    embedding: Optional[np.ndarray]  # 1D feature vector
    dimension: int  # Embedding dimensionality
    model_name: str  # Model identifier


@dataclass
class PoseRuntimeConfig:
    """Configuration for pose runtime backend selection."""

    backend_family: str  # yolo | sleap
    runtime_flavor: str = "auto"  # native | onnx | tensorrt | auto
    device: str = "auto"  # auto | cpu | cuda | mps
    batch_size: int = 4
    model_path: str = ""
    exported_model_path: str = ""
    out_root: str = "."
    min_valid_conf: float = 0.2
    yolo_conf: float = 1e-4
    yolo_iou: float = 0.7
    yolo_max_det: int = 1
    yolo_batch: int = 4
    sleap_env: str = "sleap"
    sleap_device: str = "auto"
    sleap_batch: int = 4
    sleap_max_instances: int = 1
    sleap_export_input_hw: Optional[Tuple[int, int]] = None
    sleap_experimental_features: bool = False
    keypoint_names: List[str] = field(default_factory=list)
    skeleton_edges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class AppearanceRuntimeConfig:
    """Configuration for appearance embedding runtime."""

    model_name: str = "timm/vit_base_patch14_dinov2.lvd142m"
    runtime_flavor: str = "auto"  # native | onnx | tensorrt | auto
    device: str = "auto"  # auto | cpu | cuda | mps (resolved via global runtime)
    batch_size: int = 32
    max_image_side: int = 512  # Max side length for preprocessing
    use_clahe: bool = False  # Apply CLAHE enhancement
    normalize_embeddings: bool = True  # L2 normalize output vectors
    exported_model_path: str = ""  # Path to exported ONNX/TensorRT model
    out_root: str = "."  # Root directory for exports
    compute_runtime: str = ""  # Canonical compute runtime (e.g. mps, cuda, onnx_cpu)


class PoseInferenceBackend(Protocol):
    """Protocol for all runtime backends."""

    output_keypoint_names: List[str]

    def warmup(self) -> None:
        """Warm runtime (optional no-op for unsupported backends)."""

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        """Run inference for a list of crops."""

    def close(self) -> None:
        """Release backend resources."""

    def benchmark(self, crops: Sequence[np.ndarray], runs: int = 3) -> Dict[str, float]:
        """Benchmark inference speed on provided crops."""


class AppearanceInferenceBackend(Protocol):
    """Protocol for appearance embedding backends."""

    output_dimension: int
    model_name: str

    def warmup(self) -> None:
        """Warm runtime (load model if needed)."""

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[AppearanceResult]:
        """Run inference for a list of crops (HxWx3 BGR uint8 arrays)."""

    def close(self) -> None:
        """Release backend resources."""
