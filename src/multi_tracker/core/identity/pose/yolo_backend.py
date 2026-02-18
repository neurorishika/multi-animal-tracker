"""
YOLO pose backend implementation.

Provides native Ultralytics YOLO pose inference and ONNX/TensorRT export.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Import PoseResult and PoseRuntimeConfig from runtime_types
from multi_tracker.core.identity.runtime_types import PoseResult, PoseRuntimeConfig
from multi_tracker.core.identity.runtime_utils import (
    _artifact_meta_matches,
    _normalize_export_result_path,
    _path_fingerprint_token,
    _resolve_device,
    _summarize_keypoints,
    _write_artifact_meta,
)
from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, TENSORRT_AVAILABLE

logger = logging.getLogger(__name__)


def _empty_pose_result() -> PoseResult:
    return PoseResult(
        keypoints=None,
        mean_conf=0.0,
        valid_fraction=0.0,
        num_valid=0,
        num_keypoints=0,
    )


def _auto_export_yolo_model(
    config: PoseRuntimeConfig, runtime_flavor: str, runtime_device: Optional[str] = None
) -> str:
    """
    Export YOLO pose model to ONNX or TensorRT format.

    Returns:
        Path to exported model
    """
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
    target_path = model_path.with_suffix(ext)
    export_device = (
        str(runtime_device or config.device or "auto").strip().lower() or "auto"
    )
    sig_blob = (
        f"{_path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.yolo_batch)}|device={export_device}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    if _artifact_meta_matches(target_path, sig):
        return str(target_path.resolve())
    target_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    try:
        model = YOLO(str(model_path), task="pose")
    except TypeError:
        model = YOLO(str(model_path))
    export_kwargs: Dict[str, Any] = {
        "format": "onnx" if runtime == "onnx" else "engine",
        "imgsz": 640,
        "verbose": False,
        "project": str(target_path.parent),
        "name": model_path.stem,
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
        target_path,
    )
    export_out = model.export(**export_kwargs)
    out_path = _normalize_export_result_path(export_out, expected_suffix=ext)
    if out_path is None or not out_path.exists():
        raise RuntimeError(
            f"YOLO export did not produce expected {ext} artifact (result={export_out})."
        )
    if out_path.resolve() != target_path.resolve():
        shutil.copy2(out_path, target_path)
    _write_artifact_meta(target_path, sig)
    return str(target_path.resolve())


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
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
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
