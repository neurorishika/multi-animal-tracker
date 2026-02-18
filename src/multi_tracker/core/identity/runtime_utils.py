"""
Shared utility functions for pose and appearance runtime backends.

This module contains device resolution, artifact caching, skeleton parsing,
and other helper functions used across all runtime implementations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from multi_tracker.core.identity.runtime_types import PoseResult
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


# ==============================================================================
# DEVICE RESOLUTION
# ==============================================================================


def _resolve_device(requested: str, backend_family: str) -> str:
    """Resolve device string to actual device identifier."""
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
            "Requested CUDA/ROCm for %s runtime but no compatible torch CUDA backend is available. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    if req == "mps" and not MPS_AVAILABLE:
        logger.warning(
            "Requested MPS for %s runtime but MPS is unavailable. Falling back to CPU.",
            backend_family,
        )
        return "cpu"
    return req


# ==============================================================================
# SKELETON AND KEYPOINT PARSING
# ==============================================================================


def _load_skeleton_from_json(path_str: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Load keypoint names and skeleton edges from JSON file."""
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


def _normalize_conf_values(conf: Any) -> np.ndarray:
    """Normalize confidence values to [0, 1] range."""
    arr = np.asarray(conf, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if np.any((arr < 0.0) | (arr > 1.0)):
        # Some exported ONNX predictors can emit logits instead of probabilities.
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)


# ==============================================================================
# IMAGE PROCESSING
# ==============================================================================


def _resize_crop(crop: np.ndarray, hw: Optional[Tuple[int, int]]) -> np.ndarray:
    """Resize crop to specified height and width."""
    if hw is None:
        return crop
    h, w = int(hw[0]), int(hw[1])
    if h <= 0 or w <= 0:
        return crop
    if crop.shape[0] == h and crop.shape[1] == w:
        return crop
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# ==============================================================================
# PREDICTION PARSING
# ==============================================================================


def _extract_metadata_attr(meta: Any, names: Sequence[str], default: Any = None) -> Any:
    """Extract attribute from metadata object by trying sequence of names."""
    for name in names:
        if isinstance(meta, dict) and name in meta:
            return meta.get(name)
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def _as_array(value: Any) -> Optional[np.ndarray]:
    """Convert value to numpy array, handling torch/tf tensors."""
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
    """Return first key present in mapping."""
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _pick_best_instance(
    xy: np.ndarray, conf: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Pick best instance from multi-instance predictions."""
    if xy.ndim == 2 and xy.shape[1] == 2:
        n_kpts = int(xy.shape[0])
        if conf is None:
            conf_vec = np.zeros((n_kpts,), dtype=np.float32)
        else:
            conf_vec = _normalize_conf_values(conf).reshape((-1,))
            if conf_vec.size < n_kpts:
                conf_vec = np.pad(conf_vec, (0, n_kpts - conf_vec.size))
            elif conf_vec.size > n_kpts:
                conf_vec = conf_vec[:n_kpts]
        return np.column_stack((xy.astype(np.float32), conf_vec))

    if xy.ndim == 3 and xy.shape[-1] == 2:
        n_instances = int(xy.shape[0])
        n_kpts = int(xy.shape[1])
        if n_instances <= 0:
            return None
        if conf is None:
            conf = np.zeros((n_instances, n_kpts), dtype=np.float32)
        else:
            conf = _normalize_conf_values(conf)
            if conf.ndim == 1:
                conf = np.tile(conf[None, :], (n_instances, 1))
            elif conf.ndim > 2:
                conf = conf.reshape((n_instances, -1))
            if conf.shape[0] < n_instances:
                pad_rows = n_instances - conf.shape[0]
                conf = np.pad(conf, ((0, pad_rows), (0, 0)))
            elif conf.shape[0] > n_instances:
                conf = conf[:n_instances, :]
            if conf.shape[1] < n_kpts:
                pad_cols = n_kpts - conf.shape[1]
                conf = np.pad(conf, ((0, 0), (0, pad_cols)))
            elif conf.shape[1] > n_kpts:
                conf = conf[:, :n_kpts]
        mean_scores = np.nanmean(conf, axis=1)
        idx = int(np.nanargmax(mean_scores)) if len(mean_scores) else 0
        xy_i = np.asarray(xy[idx], dtype=np.float32)
        conf_i = _normalize_conf_values(conf[idx])
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
            if int(xy.shape[0]) != int(batch_size):
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


# ==============================================================================
# PATH FINGERPRINTING AND ARTIFACT CACHING
# ==============================================================================


def _path_fingerprint_token(path_str: str) -> str:
    """Generate fingerprint token for path (file or directory)."""
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


def _artifact_meta_path(path: Path) -> Path:
    """Get metadata file path for artifact."""
    p = path.expanduser().resolve()
    if p.exists() and p.is_dir():
        return p / ".runtime_meta.json"
    if p.suffix:
        return p.with_suffix(f"{p.suffix}.runtime_meta.json")
    return p / ".runtime_meta.json"


def _artifact_meta_matches(path: Path, signature: str) -> bool:
    """Check if artifact metadata matches expected signature."""
    p = path.expanduser().resolve()
    if not p.exists():
        return False
    meta = _artifact_meta_path(p)
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(data.get("signature", "")) == str(signature)


def _write_artifact_meta(path: Path, signature: str) -> None:
    """Write artifact metadata with signature."""
    meta = _artifact_meta_path(path.expanduser().resolve())
    payload = {"signature": str(signature)}
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ==============================================================================
# RUNTIME FLAVOR PARSING
# ==============================================================================


def _parse_runtime_request(requested: str) -> Tuple[str, Optional[str]]:
    """Parse runtime request string into (flavor, device)."""
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
    """Normalize runtime flavor request to actual available runtime."""
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


# ==============================================================================
# GENERIC HELPERS
# ==============================================================================


def _nested_get(mapping: Dict[str, Any], path: Sequence[str]) -> Any:
    """Get nested value from dict using path sequence."""
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _safe_pos_int(value: Any) -> Optional[int]:
    """Convert value to positive integer or return None."""
    try:
        v = int(value)
    except Exception:
        return None
    return v if v > 0 else None


def _align_up(value: int, stride: int) -> int:
    """Align value up to stride."""
    s = max(1, int(stride))
    return int(((int(value) + s - 1) // s) * s)


def _align_hw_to_stride(
    hw: Tuple[int, int], stride: int, min_size: int = 64, max_size: int = 1024
) -> Tuple[int, int]:
    """Align height and width to stride with min/max constraints."""
    h = max(1, int(hw[0]))
    w = max(1, int(hw[1]))
    h = _align_up(h, stride)
    w = _align_up(w, stride)
    h = max(int(min_size), min(int(max_size), h))
    w = max(int(min_size), min(int(max_size), w))
    return int(h), int(w)


def _load_structured_config(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON or YAML config file."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            data = json.loads(text)
            return data if isinstance(data, dict) else None
    except Exception:
        return None

    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ==============================================================================
# SLEAP-SPECIFIC UTILITIES
# ==============================================================================


def _extract_hw_from_sleap_config(
    cfg: Dict[str, Any],
) -> Tuple[Optional[Tuple[int, int]], Optional[int]]:
    """Extract height/width and stride from SLEAP config."""
    prep = _nested_get(cfg, ["data_config", "preprocessing"])
    hw: Optional[Tuple[int, int]] = None
    if isinstance(prep, dict):
        crop_size = prep.get("crop_size")
        if isinstance(crop_size, (list, tuple)) and len(crop_size) >= 2:
            ch = _safe_pos_int(crop_size[0])
            cw = _safe_pos_int(crop_size[1])
            if ch and cw:
                hw = (int(ch), int(cw))
        elif isinstance(crop_size, dict):
            ch = _safe_pos_int(crop_size.get("height", crop_size.get("h")))
            cw = _safe_pos_int(crop_size.get("width", crop_size.get("w")))
            if ch and cw:
                hw = (int(ch), int(cw))
        else:
            c = _safe_pos_int(crop_size)
            if c:
                hw = (int(c), int(c))

        if hw is None:
            mh = _safe_pos_int(prep.get("max_height"))
            mw = _safe_pos_int(prep.get("max_width"))
            if mh and mw:
                hw = (int(mh), int(mw))
            elif mh:
                hw = (int(mh), int(mh))
            elif mw:
                hw = (int(mw), int(mw))

    stride = _safe_pos_int(
        _nested_get(cfg, ["model_config", "backbone_config", "unet", "max_stride"])
    )
    if stride is None:
        stride = _safe_pos_int(
            _nested_get(
                cfg, ["model_config", "backbone_config", "unet", "output_stride"]
            )
        )
    return hw, stride


def _derive_sleap_export_input_hw(model_path_str: str) -> Optional[Tuple[int, int]]:
    """Derive input height/width for SLEAP model export from training config."""
    model_path = Path(str(model_path_str or "")).expanduser().resolve()
    model_dir = model_path if model_path.is_dir() else model_path.parent
    if not model_dir.exists() or not model_dir.is_dir():
        return None

    candidates = [
        model_dir / "training_config.yaml",
        model_dir / "training_config.yml",
        model_dir / "training_config.json",
        model_dir / "initial_config.yaml",
        model_dir / "initial_config.yml",
        model_dir / "initial_config.json",
    ]
    chosen_hw: Optional[Tuple[int, int]] = None
    stride = 32
    for cfg_path in candidates:
        if not cfg_path.exists() or not cfg_path.is_file():
            continue
        cfg = _load_structured_config(cfg_path)
        if not isinstance(cfg, dict):
            continue
        hw, stride_candidate = _extract_hw_from_sleap_config(cfg)
        if stride_candidate is not None and stride_candidate > 0:
            stride = int(stride_candidate)
        if hw is not None and chosen_hw is None:
            chosen_hw = hw
        if chosen_hw is not None and stride is not None:
            # training_config generally contains both; stop early once found.
            break

    if chosen_hw is None:
        return None
    return _align_hw_to_stride(chosen_hw, stride=max(1, int(stride)))


def _looks_like_sleap_export_path(path_str: str, runtime_flavor: str) -> bool:
    """Check if path looks like a SLEAP export directory/file."""
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
    """Normalize export result to Path with expected suffix."""
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


def _run_cli_command(cmd: List[str], timeout_sec: int = 1800) -> Tuple[bool, str]:
    """Run CLI command and return (success, output)."""
    import subprocess

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
        return True, proc.stdout or ""
    err = (proc.stderr or proc.stdout or "").strip()
    return False, err


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
    arr = arr.copy()
    arr[:, 2] = _normalize_conf_values(arr[:, 2])
    valid_mask = arr[:, 2] >= float(min_valid_conf)
    return PoseResult(
        keypoints=arr,
        mean_conf=float(np.nanmean(arr[:, 2])),
        valid_fraction=float(np.mean(valid_mask)),
        num_valid=int(np.sum(valid_mask)),
        num_keypoints=int(len(arr)),
    )
