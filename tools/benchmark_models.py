#!/home/tracking/miniforge3/envs/multi-animal-tracker-cuda/bin/python
"""Benchmark MAT model inference across different runtimes.

Measures latency and throughput for each model type (OBB detection, pose
estimation, classification) under every available compute runtime. Produces
a summary table and optional JSON/CSV reports.

Usage
-----
    # Auto-discover models in the default registry and benchmark everything:
    python tools/benchmark_models.py

    # Specify particular model paths:
    python tools/benchmark_models.py \
        --obb-model models/obb/my_obb.pt \
        --pose-model models/pose/YOLO/my_pose.pt \
        --classify-model models/classification/orientation/tiny/my_cls.pth

    # Restrict runtimes:
    python tools/benchmark_models.py --runtimes cpu cuda tensorrt

    # Adjust warmup/iterations:
    python tools/benchmark_models.py --warmup 10 --iterations 100

    # Measure TensorRT/ONNX export time instead of inference time:
    python tools/benchmark_models.py --compile-benchmark --runtimes tensorrt onnx_cuda

    # Export results:
    python tools/benchmark_models.py --output-json results.json --output-csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the MAT package is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

from multi_tracker.runtime.compute_runtime import (
    CANONICAL_RUNTIMES,
    _normalize_runtime,
    allowed_runtimes_for_pipelines,
    runtime_label,
    supported_runtimes_for_pipeline,
)
from multi_tracker.utils.gpu_utils import get_device_info

logger = logging.getLogger("mat_benchmark")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

PIPELINE_NAMES = {
    "obb": "yolo_obb_detection",
    "detect": "yolo_obb_detection",
    "pose": "yolo_pose",
    "classify": "tiny_classify",
}


@dataclass
class BenchmarkResult:
    """Result for a single (model_type, runtime, batch_size) configuration."""

    model_type: str  # obb | pose | classify
    model_path: str
    runtime: str
    runtime_label: str
    batch_size: int
    input_shape: tuple
    warmup_iters: int
    bench_iters: int
    # Timing (seconds)
    latencies_ms: list[float] = field(default_factory=list)
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_fps: float = 0.0
    # Status
    success: bool = True
    error: str = ""

    def compute_stats(self) -> None:
        if not self.latencies_ms:
            return
        self.mean_ms = statistics.mean(self.latencies_ms)
        self.median_ms = statistics.median(self.latencies_ms)
        self.std_ms = (
            statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0
        )
        self.min_ms = min(self.latencies_ms)
        self.max_ms = max(self.latencies_ms)
        sorted_lat = sorted(self.latencies_ms)
        idx_95 = min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)
        idx_99 = min(int(len(sorted_lat) * 0.99), len(sorted_lat) - 1)
        self.p95_ms = sorted_lat[idx_95]
        self.p99_ms = sorted_lat[idx_99]
        if self.mean_ms > 0:
            self.throughput_fps = (self.batch_size / self.mean_ms) * 1000.0


@dataclass
class CompileBenchmarkResult:
    """Result for a single compile/export benchmark configuration."""

    model_type: str
    model_path: str
    runtime: str
    runtime_label: str
    batch_size: int
    artifact_path: str = ""
    compile_ms: float = 0.0
    reused_existing: bool = False
    success: bool = True
    error: str = ""


def _artifact_meta_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.runtime_meta.json")


def _remove_artifact_pair(path: Path) -> None:
    path.unlink(missing_ok=True)
    _artifact_meta_path(path).unlink(missing_ok=True)


def _resolve_obb_artifact_path(
    model_path: str, runtime: str, batch_size: int
) -> Path | None:
    resolved_model = Path(model_path).expanduser().resolve()
    rt = _normalize_runtime(runtime)
    if rt == "tensorrt":
        return resolved_model.with_name(
            f"{resolved_model.stem}_b{int(batch_size)}.engine"
        )
    if rt.startswith("onnx"):
        return resolved_model.with_name(
            f"{resolved_model.stem}_b{int(batch_size)}.onnx"
        )
    return None


def _resolve_pose_artifact_path(model_path: str, runtime: str) -> Path | None:
    resolved_model = Path(model_path).expanduser().resolve()
    rt = _normalize_runtime(runtime)
    if rt == "tensorrt":
        return resolved_model.with_suffix(".engine")
    if rt.startswith("onnx"):
        return resolved_model.with_suffix(".onnx")
    return None


# ---------------------------------------------------------------------------
# Synthetic input generation
# ---------------------------------------------------------------------------


def make_synthetic_frame(height: int = 640, width: int = 640) -> np.ndarray:
    """Generate a random BGR frame resembling a typical tracking scene."""
    rng = np.random.default_rng(42)
    frame = rng.integers(100, 200, (height, width, 3), dtype=np.uint8)
    # Add some blobs to simulate objects
    for _ in range(5):
        cx, cy = rng.integers(50, width - 50), rng.integers(50, height - 50)
        r = rng.integers(10, 40)
        cv2.circle(frame, (int(cx), int(cy)), int(r), (50, 50, 50), -1)
    return frame


def make_synthetic_crops(
    n: int = 16, height: int = 128, width: int = 128
) -> list[np.ndarray]:
    """Generate a batch of synthetic crops for pose/classification."""
    rng = np.random.default_rng(42)
    crops = []
    for _ in range(n):
        crop = rng.integers(80, 220, (height, width, 3), dtype=np.uint8)
        # Add a centred ellipse to simulate an animal
        cv2.ellipse(
            crop,
            (width // 2, height // 2),
            (width // 4, height // 6),
            rng.integers(0, 180),
            0,
            360,
            (40, 40, 40),
            -1,
        )
        crops.append(crop)
    return crops


# ---------------------------------------------------------------------------
# OBB detection benchmark
# ---------------------------------------------------------------------------


def _is_cropped_model(model_path: str) -> bool:
    """Return True when the model lives under an obb/cropped directory.

    These are stage-2 models that receive pre-cropped individual detections
    rather than full frames, so they should be benchmarked with a smaller
    synthetic input.
    """
    parts = Path(model_path).resolve().parts
    # Match any path segment called 'cropped' that follows 'obb'
    return any(
        part == "obb" and i + 1 < len(parts) and parts[i + 1] == "cropped"
        for i, part in enumerate(parts)
    )


def _runtime_to_obb_params(
    runtime: str,
    model_path: str,
    imgsz: int = 640,
    batch_size: int = 1,
    trt_workspace_gb: float = 4.0,
    trt_build_batch_size: int | None = None,
) -> dict[str, Any]:
    """Build a minimal params dict for YOLOOBBDetector from a canonical runtime."""
    rt = _normalize_runtime(runtime)
    device = "cpu"
    enable_trt = False
    enable_onnx = False
    if rt == "mps":
        device = "mps"
    elif rt in ("cuda", "rocm"):
        device = "cuda:0"
    elif rt == "tensorrt":
        device = "cuda:0"
        enable_trt = True
    elif rt in ("onnx_cpu",):
        device = "cpu"
        enable_onnx = True
    elif rt in ("onnx_cuda", "onnx_rocm"):
        device = "cuda:0"
        enable_onnx = True

    return {
        "YOLO_MODEL_PATH": model_path,
        "YOLO_OBB_MODE": "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path,
        "YOLO_DETECT_MODEL_PATH": "",
        "YOLO_CROP_OBB_MODEL_PATH": model_path,
        "YOLO_HEADTAIL_MODEL_PATH": "",
        "YOLO_DEVICE": device,
        "ENABLE_TENSORRT": enable_trt,
        "ENABLE_ONNX_RUNTIME": enable_onnx,
        "ENABLE_YOLO_BATCHING": batch_size > 1,
        "YOLO_BATCH_SIZE_MODE": "manual",
        "YOLO_MANUAL_BATCH_SIZE": batch_size,
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.7,
        "TENSORRT_MAX_BATCH_SIZE": batch_size,
        "TENSORRT_BUILD_WORKSPACE_GB": trt_workspace_gb,
        "TENSORRT_BUILD_BATCH_SIZE": trt_build_batch_size,
        "MAX_TARGETS": 25,
        "YOLO_IMGSZ": imgsz,
    }


def bench_obb(
    model_path: str,
    runtime: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    frame_size: tuple[int, int],
    imgsz: int | None = None,
    model_type: str = "obb",
    trt_workspace_gb: float = 4.0,
    trt_build_batch_size: int | None = None,
) -> BenchmarkResult:
    """Benchmark OBB/detection inference for a single runtime."""
    # imgsz defaults to the larger of the two frame dimensions when not set explicitly
    effective_imgsz = imgsz if imgsz is not None else max(frame_size)
    result = BenchmarkResult(
        model_type=model_type,
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=batch_size,
        input_shape=frame_size,
        warmup_iters=warmup,
        bench_iters=iterations,
    )

    try:
        from multi_tracker.core.detectors import YOLOOBBDetector

        params = _runtime_to_obb_params(
            runtime,
            model_path,
            imgsz=effective_imgsz,
            batch_size=batch_size,
            trt_workspace_gb=trt_workspace_gb,
            trt_build_batch_size=trt_build_batch_size,
        )
        detector = YOLOOBBDetector(params)

        frames = [make_synthetic_frame(*frame_size) for _ in range(batch_size)]

        # Warmup
        for _ in range(warmup):
            if batch_size == 1:
                detector.detect_objects(frames[0], frame_count=0)
            else:
                detector.detect_objects_batched(frames, start_frame_idx=0)

        # Timed iterations
        for _ in range(iterations):
            gc.disable()
            t0 = time.perf_counter()
            if batch_size == 1:
                detector.detect_objects(frames[0], frame_count=0)
            else:
                detector.detect_objects_batched(frames, start_frame_idx=0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            gc.enable()
            result.latencies_ms.append(elapsed_ms)

        result.compute_stats()

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("OBB benchmark failed [%s]: %s", runtime, exc)

    return result


def bench_obb_compile(
    model_path: str,
    runtime: str,
    batch_size: int,
    frame_size: tuple[int, int],
    imgsz: int | None = None,
    model_type: str = "obb",
    *,
    force_rebuild: bool = True,
    trt_workspace_gb: float = 4.0,
    trt_build_batch_size: int | None = None,
) -> CompileBenchmarkResult:
    """Benchmark export/build time for OBB TensorRT or ONNX artifacts."""
    rt = _normalize_runtime(runtime)
    result = CompileBenchmarkResult(
        model_type=model_type,
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=int(trt_build_batch_size or batch_size),
    )
    if rt not in {"tensorrt", "onnx_cpu", "onnx_cuda", "onnx_rocm"}:
        result.success = False
        result.error = "compile benchmark only supports TensorRT/ONNX runtimes"
        return result

    artifact_path = _resolve_obb_artifact_path(
        model_path, runtime, int(trt_build_batch_size or batch_size)
    )
    if artifact_path is not None:
        result.artifact_path = str(artifact_path)
        result.reused_existing = artifact_path.exists()
        if force_rebuild:
            _remove_artifact_pair(artifact_path)

    effective_imgsz = imgsz if imgsz is not None else max(frame_size)

    try:
        from multi_tracker.core.detectors import YOLOOBBDetector

        params = _runtime_to_obb_params(
            runtime,
            model_path,
            imgsz=effective_imgsz,
            batch_size=batch_size,
            trt_workspace_gb=trt_workspace_gb,
            trt_build_batch_size=trt_build_batch_size,
        )
        t0 = time.perf_counter()
        YOLOOBBDetector(params)
        result.compile_ms = (time.perf_counter() - t0) * 1000.0
        if artifact_path is not None and artifact_path.exists():
            result.artifact_path = str(artifact_path)
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("OBB compile benchmark failed [%s]: %s", runtime, exc)

    return result


# ---------------------------------------------------------------------------
# Pose estimation benchmark
# ---------------------------------------------------------------------------


def _runtime_to_pose_flavor(runtime: str) -> tuple[str, str]:
    """Map canonical runtime to (runtime_flavor, device) for YOLO pose."""
    rt = _normalize_runtime(runtime)
    mapping = {
        "cpu": ("native", "cpu"),
        "mps": ("native", "mps"),
        "cuda": ("native", "cuda:0"),
        "rocm": ("native", "cuda:0"),
        "onnx_cpu": ("onnx", "cpu"),
        "onnx_cuda": ("onnx", "cuda:0"),
        "onnx_rocm": ("onnx", "cuda:0"),
        "tensorrt": ("tensorrt", "cuda:0"),
    }
    return mapping.get(rt, ("native", "cpu"))


def bench_pose(
    model_path: str,
    runtime: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    crop_size: int,
) -> BenchmarkResult:
    """Benchmark pose estimation for a single runtime."""
    result = BenchmarkResult(
        model_type="pose",
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=batch_size,
        input_shape=(crop_size, crop_size),
        warmup_iters=warmup,
        bench_iters=iterations,
    )

    try:
        from multi_tracker.core.identity.pose.backends.yolo import (
            YoloNativeBackend,
            auto_export_yolo_model,
        )
        from multi_tracker.core.identity.pose.types import PoseRuntimeConfig

        flavor, device = _runtime_to_pose_flavor(runtime)

        actual_model = model_path
        if flavor in ("onnx", "tensorrt"):
            config = PoseRuntimeConfig(
                backend_family="yolo",
                runtime_flavor=flavor,
                device=device,
                model_path=model_path,
                yolo_batch=batch_size,
            )
            try:
                actual_model = auto_export_yolo_model(
                    config, flavor, runtime_device=device
                )
                logger.info("Exported pose model for %s: %s", flavor, actual_model)
            except Exception as exc:
                logger.warning(
                    "Pose export to %s failed (%s), falling back to native", flavor, exc
                )
                actual_model = model_path

        backend = YoloNativeBackend(
            model_path=actual_model,
            device=device,
            batch_size=batch_size,
        )

        crops = make_synthetic_crops(batch_size, crop_size, crop_size)

        # Warmup
        for _ in range(warmup):
            backend.predict_batch(crops)

        # Timed iterations
        for _ in range(iterations):
            gc.disable()
            t0 = time.perf_counter()
            backend.predict_batch(crops)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            gc.enable()
            result.latencies_ms.append(elapsed_ms)

        result.compute_stats()
        backend.close()

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Pose benchmark failed [%s]: %s", runtime, exc)

    return result


def bench_pose_compile(
    model_path: str,
    runtime: str,
    batch_size: int,
    *,
    force_rebuild: bool = True,
) -> CompileBenchmarkResult:
    """Benchmark export/build time for pose TensorRT or ONNX artifacts."""
    rt = _normalize_runtime(runtime)
    result = CompileBenchmarkResult(
        model_type="pose",
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=int(batch_size),
    )
    if rt not in {"tensorrt", "onnx_cpu", "onnx_cuda", "onnx_rocm"}:
        result.success = False
        result.error = "compile benchmark only supports TensorRT/ONNX runtimes"
        return result

    try:
        from multi_tracker.core.identity.pose.backends.yolo import (
            auto_export_yolo_model,
        )
        from multi_tracker.core.identity.pose.types import PoseRuntimeConfig

        flavor, device = _runtime_to_pose_flavor(runtime)
        artifact_path = _resolve_pose_artifact_path(model_path, runtime)
        if artifact_path is not None:
            result.artifact_path = str(artifact_path)
            result.reused_existing = artifact_path.exists()
            if force_rebuild:
                _remove_artifact_pair(artifact_path)

        config = PoseRuntimeConfig(
            backend_family="yolo",
            runtime_flavor=flavor,
            device=device,
            model_path=model_path,
            yolo_batch=batch_size,
        )
        t0 = time.perf_counter()
        exported_path = auto_export_yolo_model(config, flavor, runtime_device=device)
        result.compile_ms = (time.perf_counter() - t0) * 1000.0
        result.artifact_path = str(exported_path)
    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Pose compile benchmark failed [%s]: %s", runtime, exc)

    return result


# ---------------------------------------------------------------------------
# Classification benchmark
# ---------------------------------------------------------------------------


def bench_classify(
    model_path: str,
    runtime: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    crop_size: int,
) -> BenchmarkResult:
    """Benchmark classification for a single runtime."""
    result = BenchmarkResult(
        model_type="classify",
        model_path=model_path,
        runtime=runtime,
        runtime_label=runtime_label(runtime),
        batch_size=batch_size,
        input_shape=(crop_size, crop_size),
        warmup_iters=warmup,
        bench_iters=iterations,
    )

    try:
        from multi_tracker.core.identity.classification.cnn import (
            CNNIdentityBackend,
            CNNIdentityConfig,
        )

        config = CNNIdentityConfig(
            model_path=model_path,
            confidence=0.5,
            batch_size=batch_size,
        )
        backend = CNNIdentityBackend(
            config=config,
            model_path=model_path,
            compute_runtime=runtime,
        )

        crops = make_synthetic_crops(batch_size, crop_size, crop_size)

        # Warmup (also triggers lazy model loading)
        for _ in range(warmup):
            backend.predict_batch(crops)

        # Timed iterations
        for _ in range(iterations):
            gc.disable()
            t0 = time.perf_counter()
            backend.predict_batch(crops)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            gc.enable()
            result.latencies_ms.append(elapsed_ms)

        result.compute_stats()
        backend.close()

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        logger.warning("Classify benchmark failed [%s]: %s", runtime, exc)

    return result


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_MODELS_DIR = _REPO_ROOT / "models"


def _find_models_in_registry() -> dict[str, list[str]]:
    """Scan the model registry for available models by task family."""
    registry_path = _MODELS_DIR / "model_registry.json"
    found: dict[str, list[str]] = {"obb": [], "detect": [], "pose": [], "classify": []}

    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
            for rel_path, meta in registry.items():
                full_path = str((_MODELS_DIR / rel_path).resolve())
                if not Path(full_path).exists():
                    continue
                family = str(meta.get("task_family", "")).strip().lower()
                if family == "obb":
                    found["obb"].append(full_path)
                elif family == "detect":
                    # Regular YOLO detection models (first-stage in sequential pipeline)
                    found["detect"].append(full_path)
                elif family == "classify":
                    found["classify"].append(full_path)
        except Exception as exc:
            logger.warning("Failed to parse model_registry.json: %s", exc)

    # Scan filesystem for .pt files not in the registry
    for subdir, task_key in [
        ("obb", "obb"),
        ("detection", "detect"),
        ("pose/YOLO", "pose"),
    ]:
        scan_dir = _MODELS_DIR / subdir
        if scan_dir.is_dir():
            for p in scan_dir.rglob("*.pt"):
                full = str(p.resolve())
                if full not in found[task_key]:
                    found[task_key].append(full)

    # Scan for classification models (.pth and .pt)
    cls_dir = _MODELS_DIR / "classification"
    if cls_dir.is_dir():
        for ext in ("*.pth", "*.pt"):
            for p in cls_dir.rglob(ext):
                full = str(p.resolve())
                if full not in found["classify"]:
                    found["classify"].append(full)

    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "type": 10,
    "runtime": 15,
    "batch": 6,
    "mean": 10,
    "median": 10,
    "std": 10,
    "min": 10,
    "max": 10,
    "p95": 10,
    "p99": 10,
    "fps": 10,
    "status": 8,
}


def _print_header() -> None:
    hdr = (
        f"{'Model':10s} │ {'Runtime':15s} │ {'Batch':>6s} │ "
        f"{'Mean(ms)':>10s} │ {'Med(ms)':>10s} │ {'Std(ms)':>10s} │ "
        f"{'Min(ms)':>10s} │ {'Max(ms)':>10s} │ {'P95(ms)':>10s} │ "
        f"{'P99(ms)':>10s} │ {'FPS':>10s} │ {'Status':>8s}"
    )
    print("─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))


def _print_result(r: BenchmarkResult) -> None:
    status = "✓ OK" if r.success else "✗ FAIL"
    if r.success:
        print(
            f"{r.model_type:10s} │ {r.runtime_label:15s} │ {r.batch_size:6d} │ "
            f"{r.mean_ms:10.2f} │ {r.median_ms:10.2f} │ {r.std_ms:10.2f} │ "
            f"{r.min_ms:10.2f} │ {r.max_ms:10.2f} │ {r.p95_ms:10.2f} │ "
            f"{r.p99_ms:10.2f} │ {r.throughput_fps:10.1f} │ {status:>8s}"
        )
    else:
        err_short = r.error[:40] if r.error else "unknown"
        print(
            f"{r.model_type:10s} │ {r.runtime_label:15s} │ {r.batch_size:6d} │ "
            f"{'—':>10s} │ {'—':>10s} │ {'—':>10s} │ "
            f"{'—':>10s} │ {'—':>10s} │ {'—':>10s} │ "
            f"{'—':>10s} │ {'—':>10s} │ {status:>8s}  {err_short}"
        )


def _print_footer(n_total: int, n_success: int) -> None:
    print("─" * 150)
    print(f"Completed {n_success}/{n_total} benchmarks successfully.")


def _print_compile_header() -> None:
    hdr = (
        f"{'Model':10s} │ {'Runtime':15s} │ {'Batch':>6s} │ {'Compile(ms)':>12s} │ "
        f"{'Artifact':45s} │ {'Cache':>8s} │ {'Status':>8s}"
    )
    print("─" * len(hdr))
    print(hdr)
    print("─" * len(hdr))


def _print_compile_result(r: CompileBenchmarkResult) -> None:
    status = "✓ OK" if r.success else "✗ FAIL"
    cache_mode = "warm" if r.reused_existing else "cold"
    artifact = Path(r.artifact_path).name if r.artifact_path else "—"
    artifact = artifact[:45]
    if r.success:
        print(
            f"{r.model_type:10s} │ {r.runtime_label:15s} │ {r.batch_size:6d} │ "
            f"{r.compile_ms:12.2f} │ {artifact:45s} │ {cache_mode:>8s} │ {status:>8s}"
        )
    else:
        err_short = r.error[:40] if r.error else "unknown"
        print(
            f"{r.model_type:10s} │ {r.runtime_label:15s} │ {r.batch_size:6d} │ "
            f"{'—':>12s} │ {artifact:45s} │ {cache_mode:>8s} │ {status:>8s}  {err_short}"
        )


def _save_compile_json(results: list[CompileBenchmarkResult], path: str) -> None:
    payload = [asdict(r) for r in results]
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"\nJSON results saved to {path}")


def _save_compile_csv(results: list[CompileBenchmarkResult], path: str) -> None:
    fieldnames = [
        "model_type",
        "model_path",
        "runtime",
        "runtime_label",
        "batch_size",
        "artifact_path",
        "compile_ms",
        "reused_existing",
        "success",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"CSV results saved to {path}")


def _print_results_summary(results: list["BenchmarkResult"]) -> None:
    """Re-print all results in a clean grouped table, free of log noise."""
    if not results:
        return

    print("\n")
    print("╔" + "═" * 148 + "╗")
    print("║" + " FINAL RESULTS SUMMARY ".center(148) + "║")
    print("╚" + "═" * 148 + "╝")

    section_order = ["obb", "detect", "pose", "classify"]
    section_labels = {
        "obb": "OBB Detection",
        "detect": "Detection (first-stage)",
        "pose": "Pose Estimation",
        "classify": "Classification",
    }

    for model_type in section_order:
        group = [r for r in results if r.model_type == model_type]
        if not group:
            continue
        n_ok = sum(1 for r in group if r.success)
        print(f"\n{'═' * 60}")
        print(f"  {section_labels[model_type]}")
        print(f"{'═' * 60}")
        _print_header()
        for r in group:
            _print_result(r)
        _print_footer(len(group), n_ok)


def _print_device_info() -> None:
    """Print a summary of available compute capabilities."""
    info = get_device_info()
    print("\n╔══════════════════════════════════════════╗")
    print("║       MAT Model Benchmark Suite          ║")
    print("╚══════════════════════════════════════════╝\n")
    print("Compute Environment:")
    print(f"  PyTorch:        {'✓' if info.get('torch_available') else '✗'}", end="")
    if info.get("torch_version"):
        print(f"  (v{info['torch_version']})")
    else:
        print()
    print(f"  CUDA:           {'✓' if info.get('cuda_available') else '✗'}")
    print(f"  Torch CUDA:     {'✓' if info.get('torch_cuda_available') else '✗'}")
    print(f"  ROCm:           {'✓' if info.get('rocm_available') else '✗'}")
    print(f"  MPS:            {'✓' if info.get('mps_available') else '✗'}")
    print(f"  TensorRT:       {'✓' if info.get('tensorrt_available') else '✗'}", end="")
    if info.get("tensorrt_version"):
        print(f"  (v{info['tensorrt_version']})")
    else:
        print()
    print(
        f"  ONNXRuntime:    {'✓' if info.get('onnxruntime_available') else '✗'}", end=""
    )
    if info.get("onnxruntime_version"):
        print(f"  (v{info['onnxruntime_version']})")
    else:
        print()
    if info.get("onnxruntime_available"):
        print(f"    Providers:    {', '.join(info.get('onnxruntime_providers', []))}")
    print(f"  Numba:          {'✓' if info.get('numba_available') else '✗'}")

    supported = allowed_runtimes_for_pipelines([])
    print(f"\n  Available runtimes: {', '.join(runtime_label(r) for r in supported)}")
    print()


def _save_json(results: list[BenchmarkResult], path: str) -> None:
    """Save results to a JSON file."""
    out = []
    for r in results:
        d = asdict(r)
        d.pop("latencies_ms", None)  # omit raw latencies to keep the file compact
        d["input_shape"] = list(r.input_shape)
        out.append(d)
    Path(path).write_text(json.dumps(out, indent=2))
    print(f"\nJSON results saved to {path}")


def _save_csv(results: list[BenchmarkResult], path: str) -> None:
    """Save results to a CSV file."""
    fieldnames = [
        "model_type",
        "model_path",
        "runtime",
        "runtime_label",
        "batch_size",
        "input_shape",
        "warmup_iters",
        "bench_iters",
        "mean_ms",
        "median_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "p95_ms",
        "p99_ms",
        "throughput_fps",
        "success",
        "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fieldnames}
            row["input_shape"] = "x".join(str(x) for x in r.input_shape)
            writer.writerow(row)
    print(f"CSV results saved to {path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark MAT models across different runtimes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--obb-model",
        type=str,
        default=None,
        help="Path to an OBB detection model (.pt). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--detect-model",
        type=str,
        default=None,
        help="Path to a YOLO detection model (.pt). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--pose-model",
        type=str,
        default=None,
        help="Path to a YOLO pose model (.pt). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--classify-model",
        type=str,
        default=None,
        help="Path to a classification model (.pth/.pt). Auto-discovered if omitted.",
    )
    p.add_argument(
        "--runtimes",
        nargs="*",
        default=None,
        help=(
            "Canonical runtimes to test. Default: all available. "
            f"Choices: {', '.join(CANONICAL_RUNTIMES)}"
        ),
    )
    p.add_argument(
        "--batch-sizes",
        nargs="*",
        type=int,
        default=[1, 8, 32],
        help="Batch sizes to benchmark (default: 1 8 32).",
    )
    p.add_argument(
        "--compile-benchmark",
        action="store_true",
        help="Measure export/build time instead of steady-state inference latency.",
    )
    p.add_argument(
        "--keep-existing-artifacts",
        action="store_true",
        help="Reuse existing exported artifacts during compile benchmarking instead of deleting them first.",
    )
    p.add_argument(
        "--tensorrt-workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT builder workspace limit in GB for OBB compile/inference benchmarks (default: 4.0).",
    )
    p.add_argument(
        "--tensorrt-build-batch-size",
        type=int,
        default=None,
        help="Optional fixed TensorRT engine build batch size override for OBB benchmarks.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of timed iterations (default: 50).",
    )
    p.add_argument(
        "--frame-size",
        type=int,
        nargs=2,
        default=[4000, 4000],
        metavar=("H", "W"),
        help="Input frame size for full-frame OBB/detection models (default: 640 640).",
    )
    p.add_argument(
        "--crop-size",
        type=int,
        default=300,
        help=(
            "Square crop size (pixels) used for obb/cropped stage-2 models, "
            "pose estimation, and classification (default: 160)."
        ),
    )
    p.add_argument(
        "--skip-obb",
        action="store_true",
        help="Skip OBB detection benchmarks.",
    )
    p.add_argument(
        "--skip-detect",
        action="store_true",
        help="Skip YOLO detection model benchmarks.",
    )
    p.add_argument(
        "--skip-pose",
        action="store_true",
        help="Skip pose estimation benchmarks.",
    )
    p.add_argument(
        "--skip-classify",
        action="store_true",
        help="Skip classification benchmarks.",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save results to a JSON file.",
    )
    p.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Save results to a CSV file.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    _print_device_info()

    # ----- Discover models -----
    registry = _find_models_in_registry()

    obb_models: list[str] = []
    if not args.skip_obb:
        if args.obb_model:
            obb_models = [args.obb_model]
        else:
            obb_models = registry.get("obb", [])
        if not obb_models:
            logger.info("No OBB models found — skipping OBB benchmarks.")

    detect_models: list[str] = []
    if not args.skip_detect:
        if args.detect_model:
            detect_models = [args.detect_model]
        else:
            detect_models = registry.get("detect", [])
        if not detect_models:
            logger.info("No detection models found — skipping detection benchmarks.")

    pose_models: list[str] = []
    if not args.skip_pose:
        if args.pose_model:
            pose_models = [args.pose_model]
        else:
            pose_models = registry.get("pose", [])
        if not pose_models:
            logger.info("No pose models found — skipping pose benchmarks.")

    classify_models: list[str] = []
    if not args.skip_classify:
        if args.classify_model:
            classify_models = [args.classify_model]
        else:
            classify_models = registry.get("classify", [])
        if not classify_models:
            logger.info(
                "No classification models found — skipping classify benchmarks."
            )

    # ----- Determine runtimes -----
    if args.runtimes:
        runtimes = [_normalize_runtime(r) for r in args.runtimes]
    else:
        runtimes = allowed_runtimes_for_pipelines([])

    # ----- Run benchmarks -----
    frame_size = tuple(args.frame_size)
    cropped_frame_size = (args.crop_size, args.crop_size)

    if args.compile_benchmark:
        compile_results: list[CompileBenchmarkResult] = []
        force_rebuild = not args.keep_existing_artifacts
        compile_runtimes = {"tensorrt", "onnx_cpu", "onnx_cuda", "onnx_rocm"}

        if obb_models:
            start_idx = len(compile_results)
            obb_runtimes = [
                r
                for r in runtimes
                if r in supported_runtimes_for_pipeline("yolo_obb_detection")
                and r in compile_runtimes
            ]
            print(f"\n{'═' * 60}")
            print("  OBB Detection Compile Benchmarks")
            print(
                f"  Cache policy: {'cold rebuild' if force_rebuild else 'reuse existing artifacts'}"
            )
            print(f"{'═' * 60}")
            _print_compile_header()
            for model in obb_models:
                is_crop = _is_cropped_model(model)
                fsize = cropped_frame_size if is_crop else frame_size
                for rt in obb_runtimes:
                    for bs in args.batch_sizes:
                        r = bench_obb_compile(
                            model,
                            rt,
                            bs,
                            fsize,
                            imgsz=max(fsize),
                            force_rebuild=force_rebuild,
                            trt_workspace_gb=args.tensorrt_workspace_gb,
                            trt_build_batch_size=args.tensorrt_build_batch_size,
                        )
                        compile_results.append(r)
                        _print_compile_result(r)
            group = compile_results[start_idx:]
            _print_footer(len(group), sum(1 for r in group if r.success))

        if detect_models:
            det_runtimes = [
                r
                for r in runtimes
                if r in supported_runtimes_for_pipeline("yolo_obb_detection")
                and r in compile_runtimes
            ]
            print(f"\n{'═' * 60}")
            print("  Detection Model Compile Benchmarks")
            print(f"{'═' * 60}")
            _print_compile_header()
            start_idx = len(compile_results)
            for model in detect_models:
                for rt in det_runtimes:
                    for bs in args.batch_sizes:
                        r = bench_obb_compile(
                            model,
                            rt,
                            bs,
                            frame_size,
                            imgsz=max(frame_size),
                            model_type="detect",
                            force_rebuild=force_rebuild,
                            trt_workspace_gb=args.tensorrt_workspace_gb,
                            trt_build_batch_size=args.tensorrt_build_batch_size,
                        )
                        compile_results.append(r)
                        _print_compile_result(r)
            group = compile_results[start_idx:]
            _print_footer(len(group), sum(1 for r in group if r.success))

        if pose_models:
            pose_runtimes = [
                r
                for r in runtimes
                if r in supported_runtimes_for_pipeline("yolo_pose")
                and r in compile_runtimes
            ]
            print(f"\n{'═' * 60}")
            print("  Pose Compile Benchmarks")
            print(f"{'═' * 60}")
            _print_compile_header()
            start_idx = len(compile_results)
            for model in pose_models:
                for rt in pose_runtimes:
                    for bs in args.batch_sizes:
                        r = bench_pose_compile(
                            model,
                            rt,
                            bs,
                            force_rebuild=force_rebuild,
                        )
                        compile_results.append(r)
                        _print_compile_result(r)
            group = compile_results[start_idx:]
            _print_footer(len(group), sum(1 for r in group if r.success))

        if classify_models:
            logger.info(
                "Classification compile benchmark is not implemented; skipping classification models."
            )

        n_total = len(compile_results)
        n_ok = sum(1 for r in compile_results if r.success)
        n_fail = n_total - n_ok
        print(f"\n{'═' * 60}")
        print(f"  Overall compile results: {n_ok}/{n_total} passed, {n_fail} failed")
        print(f"{'═' * 60}\n")

        if args.output_json:
            _save_compile_json(compile_results, args.output_json)
        if args.output_csv:
            _save_compile_csv(compile_results, args.output_csv)
        return

    all_results: list[BenchmarkResult] = []

    # OBB Detection
    if obb_models:
        obb_runtimes = [
            r
            for r in runtimes
            if r in supported_runtimes_for_pipeline("yolo_obb_detection")
        ]
        full_models = [m for m in obb_models if not _is_cropped_model(m)]
        crop_models = [m for m in obb_models if _is_cropped_model(m)]
        print(f"\n{'═' * 60}")
        print("  OBB Detection Benchmarks")
        print(
            f"  Models: {len(obb_models)} ({len(full_models)} full @ {frame_size[0]}x{frame_size[1]}, "
            f"{len(crop_models)} cropped @ {args.crop_size}x{args.crop_size})"
        )
        print(f"  Runtimes: {len(obb_runtimes)} │ Batch sizes: {args.batch_sizes}")
        print(f"{'═' * 60}")
        _print_header()
        for model in obb_models:
            is_crop = _is_cropped_model(model)
            fsize = cropped_frame_size if is_crop else frame_size
            label_tag = "[cropped]" if is_crop else "[full]"
            logger.info("Benchmarking OBB model %s: %s", label_tag, Path(model).name)
            for rt in obb_runtimes:
                for bs in args.batch_sizes:
                    r = bench_obb(
                        model,
                        rt,
                        args.warmup,
                        args.iterations,
                        bs,
                        fsize,
                        imgsz=max(fsize),
                        trt_workspace_gb=args.tensorrt_workspace_gb,
                        trt_build_batch_size=args.tensorrt_build_batch_size,
                    )
                    all_results.append(r)
                    _print_result(r)
        _print_footer(
            sum(1 for r in all_results if r.model_type == "obb"),
            sum(1 for r in all_results if r.model_type == "obb" and r.success),
        )

    # Detection (first-stage detect models)
    if detect_models:
        det_runtimes = [
            r
            for r in runtimes
            if r in supported_runtimes_for_pipeline("yolo_obb_detection")
        ]
        print(f"\n{'═' * 60}")
        print("  Detection Model Benchmarks  (task=detect, full-frame)")
        print(f"  Models: {len(detect_models)} @ {frame_size[0]}x{frame_size[1]}")
        print(f"  Runtimes: {len(det_runtimes)} │ Batch sizes: {args.batch_sizes}")
        print(f"{'═' * 60}")
        _print_header()
        for model in detect_models:
            logger.info("Benchmarking detect model: %s", Path(model).name)
            for rt in det_runtimes:
                for bs in args.batch_sizes:
                    r = bench_obb(
                        model,
                        rt,
                        args.warmup,
                        args.iterations,
                        bs,
                        frame_size,
                        imgsz=max(frame_size),
                        model_type="detect",
                        trt_workspace_gb=args.tensorrt_workspace_gb,
                        trt_build_batch_size=args.tensorrt_build_batch_size,
                    )
                    all_results.append(r)
                    _print_result(r)
        _print_footer(
            sum(1 for r in all_results if r.model_type == "detect"),
            sum(1 for r in all_results if r.model_type == "detect" and r.success),
        )

    # Pose Estimation
    if pose_models:
        pose_runtimes = [
            r for r in runtimes if r in supported_runtimes_for_pipeline("yolo_pose")
        ]
        print(f"\n{'═' * 60}")
        print("  Pose Estimation Benchmarks")
        print(
            f"  Models: {len(pose_models)} │ Runtimes: {len(pose_runtimes)} │ Batch sizes: {args.batch_sizes}"
        )
        print(f"{'═' * 60}")
        _print_header()
        for model in pose_models:
            logger.info("Benchmarking pose model: %s", Path(model).name)
            for rt in pose_runtimes:
                for bs in args.batch_sizes:
                    r = bench_pose(
                        model, rt, args.warmup, args.iterations, bs, args.crop_size
                    )
                    all_results.append(r)
                    _print_result(r)
        _print_footer(
            sum(1 for r in all_results if r.model_type == "pose"),
            sum(1 for r in all_results if r.model_type == "pose" and r.success),
        )

    # Classification
    if classify_models:
        cls_runtimes = [
            r for r in runtimes if r in supported_runtimes_for_pipeline("tiny_classify")
        ]
        print(f"\n{'═' * 60}")
        print("  Classification Benchmarks")
        print(
            f"  Models: {len(classify_models)} │ Runtimes: {len(cls_runtimes)} │ Batch sizes: {args.batch_sizes}"
        )
        print(f"{'═' * 60}")
        _print_header()
        for model in classify_models:
            logger.info("Benchmarking classification model: %s", Path(model).name)
            for rt in cls_runtimes:
                for bs in args.batch_sizes:
                    r = bench_classify(
                        model, rt, args.warmup, args.iterations, bs, args.crop_size
                    )
                    all_results.append(r)
                    _print_result(r)
        _print_footer(
            sum(1 for r in all_results if r.model_type == "classify"),
            sum(1 for r in all_results if r.model_type == "classify" and r.success),
        )

    # ----- Summary -----
    n_total = len(all_results)  # already includes obb + detect + pose + classify
    n_ok = sum(1 for r in all_results if r.success)
    n_fail = n_total - n_ok

    print(f"\n{'═' * 60}")
    print(f"  Overall: {n_ok}/{n_total} passed, {n_fail} failed")
    print(f"{'═' * 60}\n")

    if n_ok > 0:
        print("Top 5 fastest configurations (by median latency):\n")
        ok_results = sorted(
            [r for r in all_results if r.success], key=lambda r: r.median_ms
        )
        for i, r in enumerate(ok_results[:5], 1):
            print(
                f"  {i}. {r.model_type:10s} │ {r.runtime_label:15s} │ "
                f"batch={r.batch_size:<3d} │ {r.median_ms:.2f}ms │ "
                f"{r.throughput_fps:.1f} FPS"
            )
        print()

    # ----- Clean summary reprint (no log noise) -----
    _print_results_summary(all_results)

    # ----- Export -----
    if args.output_json:
        _save_json(all_results, args.output_json)
    if args.output_csv:
        _save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
