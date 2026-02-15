"""
Shared pose inference runtime API for MAT + PoseKit.

This module centralizes backend selection and runtime behavior while keeping
the calling surface small and stable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from multi_tracker.utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    ONNXRUNTIME_AVAILABLE,
    ROCM_AVAILABLE,
    SLEAP_RUNTIME_ONNX_AVAILABLE,
    SLEAP_RUNTIME_TENSORRT_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Canonical pose output for one crop."""

    keypoints: Optional[np.ndarray]
    mean_conf: float
    valid_fraction: float
    num_valid: int
    num_keypoints: int


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
    keypoint_names: List[str] = field(default_factory=list)
    skeleton_edges: List[Tuple[int, int]] = field(default_factory=list)


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


def _empty_pose_result() -> PoseResult:
    return PoseResult(
        keypoints=None,
        mean_conf=0.0,
        valid_fraction=0.0,
        num_valid=0,
        num_keypoints=0,
    )


def _summarize_keypoints(
    keypoints: Optional[np.ndarray], min_valid_conf: float
) -> PoseResult:
    if keypoints is None:
        return _empty_pose_result()
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return _empty_pose_result()
    valid_mask = arr[:, 2] >= float(min_valid_conf)
    return PoseResult(
        keypoints=arr,
        mean_conf=float(np.nanmean(arr[:, 2])),
        valid_fraction=float(np.mean(valid_mask)),
        num_valid=int(np.sum(valid_mask)),
        num_keypoints=int(len(arr)),
    )


def _resolve_device(requested: str, backend_family: str) -> str:
    req = str(requested or "auto").strip().lower()
    cuda_like_available = CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE or ROCM_AVAILABLE
    if req in ("", "auto"):
        if cuda_like_available:
            return "cuda:0"
        if MPS_AVAILABLE:
            return "mps"
        return "cpu"

    if req in ("cuda", "gpu", "rocm"):
        req = "cuda:0"
    if req.startswith("cuda") and not cuda_like_available:
        logger.warning(
            "Requested CUDA/ROCm for %s pose runtime but no compatible torch CUDA backend is available. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    if req == "mps" and not MPS_AVAILABLE:
        logger.warning(
            "Requested MPS for %s pose runtime but MPS is unavailable. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    return req


def _load_skeleton_from_json(path_str: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    if not path_str:
        return [], []
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Skeleton file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    names_raw = data.get("keypoint_names", data.get("keypoints", []))
    edges_raw = data.get("skeleton_edges", data.get("edges", []))

    names = [str(v).strip() for v in names_raw if str(v).strip()]
    edges: List[Tuple[int, int]] = []
    for edge in edges_raw:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            edges.append((int(edge[0]), int(edge[1])))
        except Exception:
            continue
    return names, edges


def _resize_crop(crop: np.ndarray, hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if hw is None:
        return crop
    h, w = int(hw[0]), int(hw[1])
    if h <= 0 or w <= 0:
        return crop
    if crop.shape[0] == h and crop.shape[1] == w:
        return crop
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _extract_metadata_attr(meta: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(meta, dict) and name in meta:
            return meta.get(name)
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            return value.cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(value)
    except Exception:
        return None


def _dict_first_present(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _pick_best_instance(
    xy: np.ndarray, conf: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    if xy.ndim == 2 and xy.shape[1] == 2:
        if conf is None:
            conf = np.zeros((xy.shape[0],), dtype=np.float32)
        conf_vec = conf.reshape((-1,)).astype(np.float32)
        return np.column_stack((xy.astype(np.float32), conf_vec))

    if xy.ndim == 3 and xy.shape[-1] == 2:
        n_instances = int(xy.shape[0])
        if n_instances <= 0:
            return None
        if conf is None:
            conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)
        if conf.ndim == 1:
            conf = np.tile(conf[None, :], (n_instances, 1))
        mean_scores = np.nanmean(conf, axis=1)
        idx = int(np.nanargmax(mean_scores)) if len(mean_scores) else 0
        xy_i = np.asarray(xy[idx], dtype=np.float32)
        conf_i = np.asarray(conf[idx], dtype=np.float32)
        return np.column_stack((xy_i, conf_i))
    return None


def _coerce_prediction_batch(
    pred_out: Any, batch_size: int
) -> List[Optional[np.ndarray]]:
    """
    Convert predictor output into per-crop Nx3 arrays where columns are x,y,conf.
    """
    empty = [None] * int(max(0, batch_size))
    if pred_out is None or batch_size <= 0:
        return empty

    # Common dict output: instance_peaks + instance_peak_vals.
    if isinstance(pred_out, dict):
        xy = _as_array(
            _dict_first_present(
                pred_out,
                ["instance_peaks", "pred_instance_peaks", "peaks", "keypoints"],
            )
        )
        conf = _as_array(
            _dict_first_present(
                pred_out,
                [
                    "instance_peak_vals",
                    "pred_instance_peak_vals",
                    "peak_vals",
                    "scores",
                    "confidences",
                ],
            )
        )
        if xy is None:
            return empty

        # Normalize to [B, ...].
        if xy.ndim == 2 and xy.shape[1] == 2:
            xy = xy[None, :, :]
            if conf is not None and conf.ndim == 1:
                conf = conf[None, :]
        elif xy.ndim == 3 and xy.shape[-1] == 2:
            # Could be [B,K,2] or [I,K,2] for single image. Use batch size to infer.
            if batch_size == 1:
                xy = xy[None, :, :, :]
                if conf is not None and conf.ndim == 2:
                    conf = conf[None, :, :]
        elif xy.ndim == 4 and xy.shape[-1] == 2:
            pass
        else:
            return empty

        out: List[Optional[np.ndarray]] = []
        if xy.ndim == 3:
            # [B,K,2]
            for b in range(min(batch_size, xy.shape[0])):
                conf_b = conf[b] if conf is not None and conf.ndim >= 2 else None
                out.append(_pick_best_instance(xy[b], conf_b))
        else:
            # [B,I,K,2]
            for b in range(min(batch_size, xy.shape[0])):
                conf_b = conf[b] if conf is not None and conf.ndim >= 3 else None
                out.append(_pick_best_instance(xy[b], conf_b))
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]

    # List output, one entry per crop.
    if isinstance(pred_out, (list, tuple)):
        out: List[Optional[np.ndarray]] = []
        for item in list(pred_out)[:batch_size]:
            if isinstance(item, dict):
                parsed = _coerce_prediction_batch(item, 1)
                out.append(parsed[0] if parsed else None)
                continue
            arr = _as_array(item)
            if arr is None:
                out.append(None)
                continue
            if arr.ndim == 2 and arr.shape[1] == 3:
                out.append(np.asarray(arr, dtype=np.float32))
            elif arr.ndim == 2 and arr.shape[1] == 2:
                conf = np.zeros((arr.shape[0],), dtype=np.float32)
                out.append(np.column_stack((arr.astype(np.float32), conf)))
            else:
                out.append(None)
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]

    # Raw ndarray output.
    arr = _as_array(pred_out)
    if arr is None:
        return empty
    if arr.ndim == 3 and arr.shape[-1] == 3:
        out = [
            np.asarray(arr[i], dtype=np.float32)
            for i in range(min(batch_size, arr.shape[0]))
        ]
        if len(out) < batch_size:
            out.extend([None] * (batch_size - len(out)))
        return out[:batch_size]
    if arr.ndim == 2 and arr.shape[1] == 3 and batch_size == 1:
        return [np.asarray(arr, dtype=np.float32)]
    return empty


def _path_fingerprint_token(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        return f"{p}|missing"
    if p.is_file():
        stat = p.stat()
        return f"{p}|f|{stat.st_mtime_ns}|{stat.st_size}"

    # Directory fingerprint: root stat + key files + shallow recursive fallback.
    parts: List[str] = []
    try:
        stat = p.stat()
        parts.append(f"{p}|d|{stat.st_mtime_ns}")
    except OSError:
        parts.append(f"{p}|d|unknown")

    key_names = {
        "best.ckpt",
        "training_config.yaml",
        "training_config.json",
        "export_metadata.json",
        "metadata.json",
    }
    file_count = 0
    for child in sorted(p.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(p)
        suffix = child.suffix.lower()
        if (
            child.name in key_names
            or suffix
            in {".pt", ".ckpt", ".onnx", ".engine", ".trt", ".json", ".yaml", ".yml"}
            or file_count < 64
        ):
            try:
                st = child.stat()
                parts.append(f"{rel}|{st.st_mtime_ns}|{st.st_size}")
            except OSError:
                parts.append(f"{rel}|unknown")
            file_count += 1
        if file_count >= 256:
            break
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:24]


def _runtime_export_cache_root(out_root: str) -> Path:
    base = Path(out_root).expanduser().resolve()
    root = base / "posekit" / "runtime_exports"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_runtime_request(requested: str) -> Tuple[str, Optional[str]]:
    req = str(requested or "auto").strip().lower()
    req = req.replace("-", "_").replace(" ", "_")
    if req in {"", "auto"}:
        return "auto", None

    if req in {"native", "onnx", "tensorrt"}:
        return req, None

    for prefix in ("native", "onnx", "tensorrt"):
        token = f"{prefix}_"
        if req.startswith(token):
            suffix = req[len(token) :]
            if suffix in {"cpu", "mps"}:
                return prefix, suffix
            if suffix in {"cuda", "rocm"}:
                return prefix, "cuda:0"
            return prefix, None

    return "native", None


def _normalize_runtime_flavor(backend_family: str, requested: str) -> str:
    req, _device = _parse_runtime_request(requested)
    backend = str(backend_family or "yolo").strip().lower()
    if req in {"native", "onnx", "tensorrt"}:
        return req
    if req != "auto":
        return "native"
    if backend == "sleap":
        if SLEAP_RUNTIME_TENSORRT_AVAILABLE and CUDA_AVAILABLE:
            return "tensorrt"
        if SLEAP_RUNTIME_ONNX_AVAILABLE:
            return "onnx"
        return "native"
    if TENSORRT_AVAILABLE and CUDA_AVAILABLE:
        return "tensorrt"
    if ONNXRUNTIME_AVAILABLE:
        return "onnx"
    return "native"


def _looks_like_sleap_export_path(path_str: str, runtime_flavor: str) -> bool:
    path = Path(path_str).expanduser().resolve()
    runtime = str(runtime_flavor or "").strip().lower()
    if not path.exists():
        return False

    engine_suffixes = {".engine", ".trt"}
    if path.is_file():
        if runtime == "onnx":
            return path.suffix.lower() == ".onnx"
        if runtime == "tensorrt":
            return path.suffix.lower() in engine_suffixes
        return path.suffix.lower() in {".onnx", *engine_suffixes}

    # Directory: accept metadata + model artifact, or direct artifact presence.
    has_meta = (path / "metadata.json").exists() or (
        path / "export_metadata.json"
    ).exists()
    if runtime == "onnx":
        has_artifact = any(path.rglob("*.onnx"))
    elif runtime == "tensorrt":
        has_artifact = any(path.rglob("*.engine")) or any(path.rglob("*.trt"))
    else:
        has_artifact = (
            any(path.rglob("*.onnx"))
            or any(path.rglob("*.engine"))
            or any(path.rglob("*.trt"))
        )
    return bool(has_meta or has_artifact)


def _normalize_export_result_path(
    export_result: Any, expected_suffix: str
) -> Optional[Path]:
    candidates: List[Path] = []
    if isinstance(export_result, (str, Path)):
        candidates.append(Path(export_result))
    elif isinstance(export_result, (list, tuple)):
        for item in export_result:
            if isinstance(item, (str, Path)):
                candidates.append(Path(item))
    for p in candidates:
        p = p.expanduser().resolve()
        if p.exists() and p.is_file():
            if not expected_suffix or p.suffix.lower() == expected_suffix.lower():
                return p
    for p in candidates:
        parent = p.expanduser().resolve().parent
        if not parent.exists():
            continue
        matches = sorted(parent.glob(f"*{expected_suffix}"))
        if matches:
            return matches[-1].resolve()
    return None


def _auto_export_yolo_model(
    config: PoseRuntimeConfig, runtime_flavor: str, runtime_device: Optional[str] = None
) -> str:
    runtime = str(runtime_flavor or "native").strip().lower()
    model_path = Path(str(config.model_path).strip()).expanduser().resolve()
    if not model_path.exists():
        raise RuntimeError(f"YOLO pose model not found: {model_path}")
    if runtime == "onnx" and model_path.suffix.lower() == ".onnx":
        return str(model_path)
    if runtime == "tensorrt" and model_path.suffix.lower() == ".engine":
        return str(model_path)
    if model_path.suffix.lower() != ".pt":
        raise RuntimeError(
            "YOLO ONNX/TensorRT runtime requires .pt source model or matching exported artifact."
        )

    if runtime == "tensorrt" and not (TENSORRT_AVAILABLE and CUDA_AVAILABLE):
        raise RuntimeError(
            "TensorRT runtime requested but TensorRT/CUDA is unavailable."
        )

    ext = ".onnx" if runtime == "onnx" else ".engine"
    export_device = (
        str(runtime_device or config.device or "auto").strip().lower() or "auto"
    )
    sig_blob = (
        f"{_path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.yolo_batch)}|device={export_device}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    cache_dir = _runtime_export_cache_root(config.out_root) / "yolo" / runtime
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_path.stem}_{sig}{ext}"
    if cache_path.exists():
        return str(cache_path.resolve())

    from ultralytics import YOLO

    try:
        model = YOLO(str(model_path), task="pose")
    except TypeError:
        model = YOLO(str(model_path))
    export_kwargs: Dict[str, Any] = {
        "format": "onnx" if runtime == "onnx" else "engine",
        "imgsz": 640,
        "verbose": False,
        "project": str(cache_dir),
        "name": cache_path.stem,
        "exist_ok": True,
    }
    if runtime == "onnx":
        # torch.onnx.export in current environment supports up to opset 20.
        export_kwargs.update({"dynamic": True, "simplify": True, "opset": 20})
    else:
        export_kwargs.update(
            {
                "device": "cuda:0",
                "batch": int(max(1, config.yolo_batch)),
                "dynamic": True,
            }
        )

    logger.info(
        "Exporting YOLO pose model for %s runtime: %s -> %s",
        runtime,
        model_path,
        cache_path,
    )
    export_out = model.export(**export_kwargs)
    out_path = _normalize_export_result_path(export_out, expected_suffix=ext)
    if out_path is None or not out_path.exists():
        raise RuntimeError(
            f"YOLO export did not produce expected {ext} artifact (result={export_out})."
        )
    if out_path.resolve() != cache_path.resolve():
        shutil.copy2(out_path, cache_path)
    return str(cache_path.resolve())


def _run_cli_command(cmd: List[str], timeout_sec: int = 1800) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout or "").strip()
    return False, err[-1500:]


def _attempt_sleap_python_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    batch_size: int,
    max_instances: int,
) -> Tuple[bool, str]:
    try:
        import importlib
        import inspect
    except Exception as exc:
        return False, str(exc)

    runtime = str(runtime_flavor).strip().lower()
    module_candidates = ["sleap_nn.export.exporters", "sleap_nn.export"]
    func_candidates = ["export_model", "export"]
    if runtime == "onnx":
        func_candidates.insert(0, "export_to_onnx")
    elif runtime == "tensorrt":
        func_candidates.insert(0, "export_to_tensorrt")

    last_err = "No compatible SLEAP Python export API found."
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            last_err = str(exc)
            continue
        for fn_name in func_candidates:
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                kwargs: Dict[str, Any] = {}
                if "model_dir" in params:
                    kwargs["model_dir"] = str(model_dir)
                elif "model_path" in params:
                    kwargs["model_path"] = str(model_dir)
                elif "trained_model_path" in params:
                    kwargs["trained_model_path"] = str(model_dir)
                if "output_dir" in params:
                    kwargs["output_dir"] = str(export_dir)
                elif "export_dir" in params:
                    kwargs["export_dir"] = str(export_dir)
                elif "save_dir" in params:
                    kwargs["save_dir"] = str(export_dir)
                if "runtime" in params:
                    kwargs["runtime"] = runtime
                if "model_type" in params:
                    kwargs["model_type"] = runtime
                if "format" in params:
                    kwargs["format"] = runtime
                if "batch_size" in params:
                    kwargs["batch_size"] = int(max(1, batch_size))
                if "max_instances" in params:
                    kwargs["max_instances"] = int(max(1, max_instances))
                if kwargs:
                    fn(**kwargs)
                else:
                    fn(str(model_dir), str(export_dir))
                if _looks_like_sleap_export_path(str(export_dir), runtime):
                    return True, ""
            except Exception as exc:
                last_err = str(exc)
                continue
    return False, last_err


def _attempt_sleap_cli_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    sleap_env: str,
) -> Tuple[bool, str]:
    runtime = str(runtime_flavor).strip().lower()
    runtime_tokens = [runtime]
    if runtime == "tensorrt":
        runtime_tokens.append("trt")

    command_variants: List[List[str]] = []
    for token in runtime_tokens:
        command_variants.extend(
            [
                [
                    "sleap-nn",
                    "export",
                    str(model_dir),
                    "--output",
                    str(export_dir),
                    "--format",
                    token,
                ],
                [
                    "sleap-nn",
                    "export",
                    "--model",
                    str(model_dir),
                    "--output",
                    str(export_dir),
                    "--format",
                    token,
                ],
                [
                    "python",
                    "-m",
                    "sleap_nn.export",
                    "--model",
                    str(model_dir),
                    "--output",
                    str(export_dir),
                    "--format",
                    token,
                ],
            ]
        )

    if shutil.which("conda") and sleap_env:
        conda_wrapped = []
        for cmd in command_variants:
            conda_wrapped.append(["conda", "run", "-n", sleap_env, *cmd])
        command_variants = conda_wrapped + command_variants

    last_err = "No SLEAP export CLI command succeeded."
    for cmd in command_variants:
        ok, err = _run_cli_command(cmd)
        if ok and _looks_like_sleap_export_path(str(export_dir), runtime):
            return True, ""
        if err:
            last_err = err
    return False, last_err


def _auto_export_sleap_model(config: PoseRuntimeConfig, runtime_flavor: str) -> str:
    runtime = str(runtime_flavor or "native").strip().lower()
    if runtime not in {"onnx", "tensorrt"}:
        raise RuntimeError(f"Unsupported SLEAP auto-export runtime: {runtime}")

    explicit = str(config.exported_model_path or "").strip()
    if explicit:
        explicit_path = Path(explicit).expanduser().resolve()
        if explicit_path.exists():
            return str(explicit_path)

    model_path = Path(str(config.model_path or "")).expanduser().resolve()
    if _looks_like_sleap_export_path(str(model_path), runtime):
        return str(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise RuntimeError(
            f"SLEAP model path does not exist or is not a directory: {model_path}"
        )

    sig_blob = (
        f"{_path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.sleap_batch)}|max_instances={int(config.sleap_max_instances)}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    export_dir = (
        _runtime_export_cache_root(config.out_root)
        / "sleap"
        / runtime
        / f"{model_path.name}_{sig}"
    )
    export_dir.mkdir(parents=True, exist_ok=True)
    if _looks_like_sleap_export_path(str(export_dir), runtime):
        return str(export_dir.resolve())

    logger.info(
        "Exporting SLEAP model for %s runtime: %s -> %s",
        runtime,
        model_path,
        export_dir,
    )
    ok, err = _attempt_sleap_python_export(
        model_dir=model_path,
        export_dir=export_dir,
        runtime_flavor=runtime,
        batch_size=int(max(1, config.sleap_batch)),
        max_instances=int(max(1, config.sleap_max_instances)),
    )
    if not ok:
        ok, err = _attempt_sleap_cli_export(
            model_dir=model_path,
            export_dir=export_dir,
            runtime_flavor=runtime,
            sleap_env=str(config.sleap_env or "").strip(),
        )
    if not ok or not _looks_like_sleap_export_path(str(export_dir), runtime):
        raise RuntimeError(f"SLEAP auto-export failed for runtime '{runtime}'. {err}")
    return str(export_dir.resolve())


class YoloNativeBackend:
    """Ultralytics-native YOLO pose runtime (.pt/.onnx/.engine)."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        min_valid_conf: float = 0.2,
        keypoint_names: Optional[Sequence[str]] = None,
        conf: float = 1e-4,
        iou: float = 0.7,
        max_det: int = 1,
        batch_size: int = 4,
    ):
        from ultralytics import YOLO

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = _resolve_device(device, "yolo")
        self.is_onnx_model = self.model_path.lower().endswith(".onnx")
        if self.is_onnx_model and self.device == "mps":
            # Ultralytics maps MPS to CoreMLExecutionProvider for ORT; this is unstable
            # on some large pose models. Use CPU ORT provider for reliability.
            logger.info(
                "YOLO ONNX runtime requested with mps device; forcing cpu to avoid CoreMLExecutionProvider instability."
            )
            self.device = "cpu"
        self.min_valid_conf = float(min_valid_conf)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.batch_size = max(1, int(batch_size))
        self.output_keypoint_names = [str(v) for v in (keypoint_names or [])]
        self._warned_conf_out_of_range = False

        try:
            self.model = YOLO(self.model_path, task="pose")
        except TypeError:
            # Backward-compatible with older Ultralytics constructors.
            self.model = YOLO(self.model_path)
        # .to() is meaningful for torch-based weights; harmless to ignore failures.
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def warmup(self) -> None:
        try:
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            self.predict_batch([dummy])
        except Exception:
            # Warmup failure should not hard-fail runtime creation.
            logger.debug("YOLO warmup skipped.", exc_info=True)

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []

        t0 = time.perf_counter()
        try:
            results = self.model.predict(
                source=list(crops),
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                batch=self.batch_size,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            msg = str(exc)
            coreml_failure = self.is_onnx_model and (
                "CoreMLExecutionProvider" in msg
                or "Unable to compute the prediction using a neural network model"
                in msg
            )
            if coreml_failure and self.device != "cpu":
                logger.warning(
                    "YOLO ONNX inference failed on %s (CoreML path). Retrying on CPU ORT provider.",
                    self.device,
                )
                self.device = "cpu"
                try:
                    # Reset Ultralytics predictor so provider selection is rebuilt.
                    if hasattr(self.model, "predictor"):
                        self.model.predictor = None
                except Exception:
                    pass
                results = self.model.predict(
                    source=list(crops),
                    conf=self.conf,
                    iou=self.iou,
                    max_det=self.max_det,
                    batch=self.batch_size,
                    verbose=False,
                    device=self.device,
                )
            else:
                raise
        infer_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "YOLO pose runtime: %d crops in %.2f ms (%.2f ms/crop)",
            len(crops),
            infer_ms,
            infer_ms / max(1, len(crops)),
        )

        outputs: List[PoseResult] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                outputs.append(_empty_pose_result())
                continue
            try:
                xy = keypoints.xy
                xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
                conf = getattr(keypoints, "conf", None)
                if conf is not None:
                    conf = (
                        conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
                    )
            except Exception:
                outputs.append(_empty_pose_result())
                continue

            if xy.ndim == 2:
                xy = xy[None, :, :]
            if conf is not None and conf.ndim == 1:
                conf = conf[None, :]
            if xy.size == 0:
                outputs.append(_empty_pose_result())
                continue

            if conf is None:
                conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)
            mean_per_instance = np.nanmean(conf, axis=1)
            best_idx = (
                int(np.nanargmax(mean_per_instance)) if len(mean_per_instance) else 0
            )
            pred_xy = np.asarray(xy[best_idx], dtype=np.float32)
            pred_conf = np.asarray(conf[best_idx], dtype=np.float32)
            # Confidence should be in [0,1]. Log once if backend returns out-of-range
            # values, then clamp for downstream stability.
            if pred_conf.size:
                bad_mask = (
                    (pred_conf < 0.0) | (pred_conf > 1.0) | ~np.isfinite(pred_conf)
                )
                if np.any(bad_mask) and not self._warned_conf_out_of_range:
                    bad_vals = pred_conf[bad_mask]
                    logger.warning(
                        "YOLO pose keypoint confidence out-of-range detected "
                        "(count=%d/%d, min=%s, max=%s, model=%s, device=%s). "
                        "Values will be clamped to [0,1].",
                        int(np.sum(bad_mask)),
                        int(pred_conf.size),
                        float(np.nanmin(bad_vals)) if bad_vals.size else "nan",
                        float(np.nanmax(bad_vals)) if bad_vals.size else "nan",
                        self.model_path,
                        self.device,
                    )
                    self._warned_conf_out_of_range = True
            pred_conf = np.nan_to_num(pred_conf, nan=0.0, posinf=1.0, neginf=0.0)
            pred_conf = np.clip(pred_conf, 0.0, 1.0)
            if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
                outputs.append(_empty_pose_result())
                continue

            kpts = np.column_stack((pred_xy, pred_conf)).astype(np.float32)
            outputs.append(_summarize_keypoints(kpts, self.min_valid_conf))

        return outputs

    def benchmark(self, crops: Sequence[np.ndarray], runs: int = 3) -> Dict[str, float]:
        if not crops:
            return {"runs": 0.0, "total_ms": 0.0, "ms_per_run": 0.0, "fps": 0.0}
        total = 0.0
        runs = max(1, int(runs))
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict_batch(crops)
            total += (time.perf_counter() - t0) * 1000.0
        ms_per_run = total / runs
        return {
            "runs": float(runs),
            "total_ms": float(total),
            "ms_per_run": float(ms_per_run),
            "fps": float((len(crops) * 1000.0) / max(1e-6, ms_per_run)),
        }

    def close(self) -> None:
        return None


class SleapServiceBackend:
    """SLEAP runtime adapter via PoseInferenceService HTTP service."""

    def __init__(
        self,
        model_dir: str,
        out_root: str,
        keypoint_names: Sequence[str],
        min_valid_conf: float = 0.2,
        sleap_env: str = "sleap",
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        skeleton_edges: Optional[Sequence[Sequence[int]]] = None,
    ):
        from multi_tracker.posekit.pose_inference import PoseInferenceService

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.output_keypoint_names = [str(v) for v in keypoint_names]
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_env = str(sleap_env or "sleap")
        self.sleap_device = str(sleap_device or "auto")
        self.sleap_batch = max(1, int(sleap_batch))
        self.sleap_max_instances = max(1, int(sleap_max_instances))
        self.skeleton_edges = (
            [tuple(int(v) for v in e[:2]) for e in (skeleton_edges or [])]
            if skeleton_edges
            else []
        )
        self._infer = PoseInferenceService(
            self.out_root, self.output_keypoint_names, self.skeleton_edges
        )
        self._service_started_here = False
        self._tmp_root = (
            self.out_root / "posekit" / "tmp" / f"runtime_{uuid.uuid4().hex}"
        )
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def warmup(self) -> None:
        # Explicitly start/validate service when env is configured.
        try:
            was_running = self._infer.sleap_service_running()
            ok, err, _ = self._infer.start_sleap_service(self.sleap_env, self.out_root)
            if not ok:
                raise RuntimeError(err or "Failed to start SLEAP service.")
            self._service_started_here = (
                not was_running
            ) and self._infer.sleap_service_running()
        except Exception as exc:
            logger.warning("SLEAP service warmup failed: %s", exc)

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []

        paths: List[Path] = []
        for i, crop in enumerate(crops):
            p = self._tmp_root / f"crop_{i:06d}.png"
            ok = cv2.imwrite(str(p), crop)
            if not ok:
                paths.append(Path("__invalid__"))
            else:
                paths.append(p)

        valid_paths = [p for p in paths if p.exists()]
        preds: Dict[str, List[Any]] = {}
        if valid_paths:
            pred_map, err = self._infer.predict(
                model_path=self.model_dir,
                image_paths=valid_paths,
                device="auto",
                imgsz=640,
                conf=1e-4,
                batch=self.sleap_batch,
                backend="sleap",
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
            )
            if pred_map is None:
                raise RuntimeError(err or "SLEAP inference failed.")
            preds = pred_map

        outputs: List[PoseResult] = []
        for p in paths:
            if not p.exists():
                outputs.append(_empty_pose_result())
                continue
            pred = preds.get(str(p))
            if pred is None:
                pred = preds.get(str(p.resolve()))
            if not pred:
                outputs.append(_empty_pose_result())
                continue
            arr = np.asarray(pred, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                outputs.append(_empty_pose_result())
                continue
            outputs.append(_summarize_keypoints(arr, self.min_valid_conf))

        return outputs

    def benchmark(self, crops: Sequence[np.ndarray], runs: int = 3) -> Dict[str, float]:
        if not crops:
            return {"runs": 0.0, "total_ms": 0.0, "ms_per_run": 0.0, "fps": 0.0}
        total = 0.0
        runs = max(1, int(runs))
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict_batch(crops)
            total += (time.perf_counter() - t0) * 1000.0
        ms_per_run = total / runs
        return {
            "runs": float(runs),
            "total_ms": float(total),
            "ms_per_run": float(ms_per_run),
            "fps": float((len(crops) * 1000.0) / max(1e-6, ms_per_run)),
        }

    def close(self) -> None:
        if self._service_started_here:
            try:
                self._infer.shutdown_sleap_service()
            except Exception:
                logger.debug(
                    "Failed to stop SLEAP service from backend close.", exc_info=True
                )
            self._service_started_here = False
        if self._tmp_root.exists():
            try:
                for p in self._tmp_root.glob("*.png"):
                    p.unlink(missing_ok=True)
                self._tmp_root.rmdir()
            except Exception:
                pass


class SleapExportBackend:
    """SLEAP exported-model runtime using sleap-nn predictors."""

    def __init__(
        self,
        exported_model_path: str,
        runtime_flavor: str,
        device: str,
        min_valid_conf: float,
        keypoint_names: Sequence[str],
        sleap_batch: int,
        sleap_max_instances: int,
    ):
        self.runtime_flavor = str(runtime_flavor or "auto").strip().lower()
        if self.runtime_flavor not in {"onnx", "tensorrt"}:
            raise RuntimeError(
                f"SLEAP export backend requires onnx/tensorrt runtime, got: {self.runtime_flavor}"
            )
        self.exported_model_path = str(Path(exported_model_path).expanduser().resolve())
        self.device = _resolve_device(device, "sleap")
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_batch = max(1, int(sleap_batch))
        self.sleap_max_instances = max(1, int(sleap_max_instances))
        self.output_keypoint_names = [str(v) for v in (keypoint_names or [])]
        self._predictor: Any = None
        self._metadata: Any = None
        self._input_hw: Optional[Tuple[int, int]] = None
        self._input_channels: Optional[int] = None

        self._init_predictor()

    def _build_predictor_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.runtime_flavor == "onnx":
            if self.device.startswith("cuda"):
                kwargs["provider"] = "cuda"
            elif self.device == "cpu":
                kwargs["provider"] = "cpu"
            elif self.device == "mps":
                # ONNXRuntime does not use MPS execution provider.
                kwargs["provider"] = "cpu"
            else:
                kwargs["provider"] = "cpu"
        else:
            if not self.device.startswith("cuda"):
                raise RuntimeError(
                    "SLEAP TensorRT runtime requires CUDA device. "
                    f"Configured device: {self.device}"
                )
            kwargs["device"] = self.device
        kwargs["batch_size"] = self.sleap_batch
        kwargs["max_instances"] = self.sleap_max_instances
        return kwargs

    def _detect_input_spec(self) -> None:
        meta = self._metadata
        if meta is None:
            return
        hw = _extract_metadata_attr(
            meta,
            names=[
                "crop_size",
                "crop_hw",
                "input_hw",
                "image_hw",
            ],
            default=None,
        )
        if isinstance(hw, (list, tuple)) and len(hw) >= 2:
            try:
                self._input_hw = (int(hw[0]), int(hw[1]))
            except Exception:
                self._input_hw = None
        input_shape = _extract_metadata_attr(
            meta,
            names=[
                "input_image_shape",
                "input_shape",
            ],
            default=None,
        )
        if input_shape is not None:
            arr = np.asarray(input_shape).reshape((-1,))
            if arr.size >= 4:
                # Supports [B,H,W,C] and [B,C,H,W] style shapes.
                if int(arr[-1]) in (1, 3):
                    self._input_channels = int(arr[-1])
                    self._input_hw = (int(arr[-3]), int(arr[-2]))
                elif int(arr[1]) in (1, 3):
                    self._input_channels = int(arr[1])
                    self._input_hw = (int(arr[-2]), int(arr[-1]))

        channels = _extract_metadata_attr(
            meta,
            names=[
                "input_channels",
                "channels",
                "num_channels",
            ],
            default=None,
        )
        if channels is not None:
            try:
                self._input_channels = int(channels)
            except Exception:
                pass

        node_names = _extract_metadata_attr(
            meta,
            names=["node_names", "keypoint_names"],
            default=None,
        )
        if node_names:
            try:
                nodes = [str(v) for v in list(node_names)]
                if nodes:
                    self.output_keypoint_names = nodes
            except Exception:
                pass

    def _init_predictor(self) -> None:
        try:
            from sleap_nn.export.metadata import load_metadata
        except Exception:
            load_metadata = None

        try:
            from sleap_nn.export.predictors import load_exported_model
        except Exception as exc:
            raise RuntimeError(
                "sleap-nn exported predictors are unavailable. "
                "Install sleap-nn export dependencies to use ONNX/TensorRT runtime."
            ) from exc

        if load_metadata is not None:
            try:
                self._metadata = load_metadata(self.exported_model_path)
            except Exception:
                self._metadata = None
        self._detect_input_spec()

        loader_attempts = [
            {"runtime": self.runtime_flavor, **self._build_predictor_kwargs()},
            {"inference_model": self.runtime_flavor, **self._build_predictor_kwargs()},
            {"model_type": self.runtime_flavor, **self._build_predictor_kwargs()},
            {"runtime": self.runtime_flavor},
            {},
        ]
        last_err: Optional[Exception] = None
        for kwargs in loader_attempts:
            try:
                self._predictor = load_exported_model(
                    self.exported_model_path, **kwargs
                )
                break
            except TypeError as exc:
                # Signature mismatch across versions; keep trying with fewer kwargs.
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                break

        if self._predictor is None:
            raise RuntimeError(
                f"Failed to initialize SLEAP exported runtime from {self.exported_model_path}: {last_err}"
            )

        cls_name = type(self._predictor).__name__.lower()
        if self.runtime_flavor == "onnx" and "trt" in cls_name:
            raise RuntimeError(
                "Requested ONNX runtime but TensorRT predictor was loaded."
            )
        if self.runtime_flavor == "tensorrt" and "onnx" in cls_name:
            raise RuntimeError(
                "Requested TensorRT runtime but ONNX predictor was loaded."
            )

        logger.info(
            "Initialized SLEAP exported runtime (%s) from %s with predictor=%s",
            self.runtime_flavor,
            self.exported_model_path,
            type(self._predictor).__name__,
        )

    def warmup(self) -> None:
        try:
            dummy_h, dummy_w = self._input_hw if self._input_hw else (64, 64)
            channels = self._input_channels or 3
            if channels == 1:
                dummy = np.zeros((dummy_h, dummy_w), dtype=np.uint8)
            else:
                dummy = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            self.predict_batch([dummy])
        except Exception:
            logger.debug("SLEAP export warmup skipped.", exc_info=True)

    def _prepare_inputs_uint8(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        processed: List[np.ndarray] = []
        channels = self._input_channels or 3
        for crop in crops:
            arr = np.asarray(crop)
            arr = _resize_crop(arr, self._input_hw)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[-1] >= 3 and channels == 1:
                arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)[:, :, None]
            elif arr.shape[-1] == 1 and channels == 3:
                arr = np.repeat(arr, 3, axis=2)
            elif arr.shape[-1] > channels:
                arr = arr[:, :, :channels]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            processed.append(arr)
        return np.stack(processed, axis=0)

    def _predict_raw(self, crops: Sequence[np.ndarray]) -> Any:
        if self._predictor is None:
            raise RuntimeError("SLEAP export predictor is not initialized.")
        # Try common predictor input conventions in descending likelihood.
        last_err: Optional[Exception] = None
        attempts: List[Any] = []
        attempts.append(list(crops))
        try:
            batch_uint8 = self._prepare_inputs_uint8(crops)
            attempts.append(batch_uint8)
            attempts.append(batch_uint8.astype(np.float32) / 255.0)
            if batch_uint8.ndim == 4:
                nchw = np.transpose(
                    batch_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2)
                )
                attempts.append(nchw)
        except Exception as exc:
            last_err = exc

        for inp in attempts:
            try:
                return self._predictor.predict(inp)
            except Exception as exc:
                last_err = exc
                continue
        raise RuntimeError(
            f"SLEAP exported predictor failed to run inference: {last_err}"
        )

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []
        raw_out = self._predict_raw(crops)
        kpt_batch = _coerce_prediction_batch(raw_out, batch_size=len(crops))
        outputs: List[PoseResult] = []
        for kpts in kpt_batch:
            outputs.append(_summarize_keypoints(kpts, self.min_valid_conf))
        return outputs

    def benchmark(self, crops: Sequence[np.ndarray], runs: int = 3) -> Dict[str, float]:
        if not crops:
            return {"runs": 0.0, "total_ms": 0.0, "ms_per_run": 0.0, "fps": 0.0}
        total = 0.0
        runs = max(1, int(runs))
        for _ in range(runs):
            t0 = time.perf_counter()
            self.predict_batch(crops)
            total += (time.perf_counter() - t0) * 1000.0
        ms_per_run = total / runs
        return {
            "runs": float(runs),
            "total_ms": float(total),
            "ms_per_run": float(ms_per_run),
            "fps": float((len(crops) * 1000.0) / max(1e-6, ms_per_run)),
        }

    def close(self) -> None:
        if self._predictor is not None and hasattr(self._predictor, "close"):
            try:
                self._predictor.close()
            except Exception:
                logger.debug("SLEAP exported predictor close failed.", exc_info=True)
        self._predictor = None


def build_runtime_config(
    params: Dict[str, Any],
    out_root: str,
    keypoint_names_override: Optional[Sequence[str]] = None,
    skeleton_edges_override: Optional[Sequence[Sequence[int]]] = None,
) -> PoseRuntimeConfig:
    backend_family = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
    runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "auto")).strip().lower()
    model_path = str(params.get("POSE_MODEL_DIR", "")).strip()
    exported_model_path = str(params.get("POSE_EXPORTED_MODEL_PATH", "")).strip()

    skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
    skeleton_names, skeleton_edges = _load_skeleton_from_json(skeleton_file)

    if keypoint_names_override:
        skeleton_names = [str(v) for v in keypoint_names_override]
    if skeleton_edges_override:
        skeleton_edges = [
            (int(edge[0]), int(edge[1]))
            for edge in skeleton_edges_override
            if isinstance(edge, (list, tuple)) and len(edge) >= 2
        ]

    # Keep YOLO and SLEAP device controls explicit.
    if backend_family == "sleap":
        device = str(params.get("POSE_SLEAP_DEVICE", "auto")).strip() or "auto"
        batch_size = int(params.get("POSE_SLEAP_BATCH", 4))
    else:
        device = (
            str(params.get("YOLO_DEVICE", params.get("POSE_DEVICE", "auto"))).strip()
            or "auto"
        )
        batch_size = int(
            params.get(
                "POSE_YOLO_BATCH",
                params.get("POSE_BATCH_SIZE", 4),
            )
        )

    return PoseRuntimeConfig(
        backend_family=backend_family,
        runtime_flavor=runtime_flavor,
        device=device,
        batch_size=max(1, batch_size),
        model_path=model_path,
        exported_model_path=exported_model_path,
        out_root=str(out_root),
        min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
        yolo_conf=float(params.get("POSE_YOLO_CONF", 1e-4)),
        yolo_iou=float(params.get("POSE_YOLO_IOU", 0.7)),
        yolo_max_det=int(params.get("POSE_YOLO_MAX_DET", 1)),
        yolo_batch=max(1, int(batch_size)),
        sleap_env=str(params.get("POSE_SLEAP_ENV", "sleap")),
        sleap_device=str(params.get("POSE_SLEAP_DEVICE", "auto")),
        sleap_batch=int(params.get("POSE_SLEAP_BATCH", 4)),
        sleap_max_instances=int(params.get("POSE_SLEAP_MAX_INSTANCES", 1)),
        keypoint_names=skeleton_names,
        skeleton_edges=skeleton_edges,
    )


def create_pose_backend_from_config(config: PoseRuntimeConfig) -> PoseInferenceBackend:
    backend_family = str(config.backend_family or "yolo").strip().lower()
    requested_runtime = str(config.runtime_flavor or "auto").strip().lower()
    parsed_runtime, parsed_device = _parse_runtime_request(requested_runtime)
    runtime_flavor = _normalize_runtime_flavor(backend_family, requested_runtime)
    effective_device = (
        parsed_device
        if parsed_device
        else (
            str(config.sleap_device or "auto")
            if backend_family == "sleap"
            else str(config.device or "auto")
        )
    )
    if parsed_runtime == "auto":
        logger.info(
            "Pose runtime auto-selected for %s backend: %s",
            backend_family,
            runtime_flavor,
        )

    if backend_family == "yolo":
        model_candidate = str(config.model_path).strip()
        model_candidate_path = (
            Path(model_candidate).expanduser().resolve() if model_candidate else None
        )
        if model_candidate_path is not None and model_candidate_path.exists():
            if model_candidate_path.is_dir():
                raise RuntimeError(
                    "POSE_MODEL_TYPE is set to 'yolo' but POSE_MODEL_DIR points to a "
                    "directory. This looks like a SLEAP model directory. "
                    "Set POSE_MODEL_TYPE='sleap' for this model, or select a YOLO model "
                    "file (.pt/.onnx/.engine)."
                )
            valid_yolo_suffixes = {".pt", ".onnx", ".engine", ".trt"}
            if model_candidate_path.suffix.lower() not in valid_yolo_suffixes:
                raise RuntimeError(
                    "Unsupported YOLO model path for pose inference: "
                    f"{model_candidate_path}. Expected one of: "
                    ".pt, .onnx, .engine, .trt"
                )
        if runtime_flavor in ("onnx", "tensorrt"):
            exported = str(config.exported_model_path).strip()
            try:
                if exported:
                    model_candidate = str(Path(exported).expanduser().resolve())
                else:
                    model_candidate = _auto_export_yolo_model(
                        config, runtime_flavor, runtime_device=effective_device
                    )
            except Exception as exc:
                logger.warning(
                    "YOLO %s runtime initialization failed (%s). Falling back to native runtime.",
                    runtime_flavor,
                    exc,
                )
                runtime_flavor = "native"
                model_candidate = str(config.model_path).strip()
        if not model_candidate:
            raise RuntimeError("Pose model path is empty.")
        return YoloNativeBackend(
            model_path=model_candidate,
            device=effective_device,
            min_valid_conf=config.min_valid_conf,
            keypoint_names=config.keypoint_names,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            max_det=config.yolo_max_det,
            batch_size=config.yolo_batch,
        )

    if backend_family == "sleap":
        if not config.keypoint_names:
            raise RuntimeError(
                "SLEAP backend requires keypoint_names (from skeleton JSON or override)."
            )
        if runtime_flavor in ("onnx", "tensorrt"):
            try:
                export_candidate = _auto_export_sleap_model(config, runtime_flavor)
                return SleapExportBackend(
                    exported_model_path=export_candidate,
                    runtime_flavor=runtime_flavor,
                    device=effective_device,
                    min_valid_conf=config.min_valid_conf,
                    keypoint_names=config.keypoint_names,
                    sleap_batch=config.sleap_batch,
                    sleap_max_instances=max(1, int(config.sleap_max_instances)),
                )
            except Exception as exc:
                logger.warning(
                    "SLEAP exported runtime (%s) initialization failed: %s. "
                    "Falling back to SLEAP service backend.",
                    runtime_flavor,
                    exc,
                )

        if not config.model_path:
            raise RuntimeError(
                "SLEAP model path is empty. Provide POSE_MODEL_DIR for service fallback."
            )
        service_backend = SleapServiceBackend(
            model_dir=config.model_path,
            out_root=config.out_root,
            keypoint_names=config.keypoint_names,
            min_valid_conf=config.min_valid_conf,
            sleap_env=config.sleap_env,
            sleap_device=effective_device,
            sleap_batch=config.sleap_batch,
            sleap_max_instances=max(1, int(config.sleap_max_instances)),
            skeleton_edges=config.skeleton_edges,
        )
        return service_backend

    raise RuntimeError(f"Unsupported pose backend family: {backend_family}")
