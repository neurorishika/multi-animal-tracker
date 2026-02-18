"""
SLEAP pose backend implementation.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from multi_tracker.core.identity.runtime_types import PoseResult, PoseRuntimeConfig
from multi_tracker.core.identity.runtime_utils import (
    _artifact_meta_matches,
    _coerce_prediction_batch,
    _empty_pose_result,
    _extract_metadata_attr,
    _looks_like_sleap_export_path,
    _path_fingerprint_token,
    _resize_crop,
    _resolve_device,
    _run_cli_command,
    _summarize_keypoints,
    _write_artifact_meta,
)

logger = logging.getLogger(__name__)


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
    input_hw: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
) -> Tuple[bool, str]:
    runtime = str(runtime_flavor).strip().lower()
    sleap_env = str(sleap_env or "").strip()
    batch_size = int(max(1, int(batch_size)))
    runtime_tokens = [runtime]
    if runtime == "tensorrt":
        runtime_tokens.append("trt")

    size_candidates: List[Tuple[int, int]] = []
    if (
        isinstance(input_hw, (tuple, list))
        and len(input_hw) >= 2
        and int(input_hw[0]) > 0
        and int(input_hw[1]) > 0
    ):
        size_candidates.append((int(input_hw[0]), int(input_hw[1])))
    # Fallback sizes that are generally safe for encoder/decoder stride alignment.
    size_candidates.extend([(224, 224), (256, 256)])
    # Deduplicate while preserving order.
    deduped_sizes: List[Tuple[int, int]] = []
    for s in size_candidates:
        if s not in deduped_sizes:
            deduped_sizes.append(s)
    size_candidates = deduped_sizes

    command_variants: List[List[str]] = []
    for token in runtime_tokens:
        profile_variants = [
            ["--batch-size", str(batch_size)],
            ["--batch", str(batch_size)],
            [],
        ]
        base_variants = [
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
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                str(model_dir),
                "--output",
                str(export_dir),
                "--format",
                token,
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
        ]
        for h, w in size_candidates:
            for base in base_variants:
                for prof in profile_variants:
                    command_variants.append(
                        [
                            *base,
                            "--input-height",
                            str(int(h)),
                            "--input-width",
                            str(int(w)),
                            *prof,
                        ]
                    )
        # Keep no-size variants as last resort for CLI versions/models that infer size.
        for base in base_variants:
            for prof in profile_variants:
                command_variants.append([*base, *prof])

    if shutil.which("conda") and sleap_env:
        conda_wrapped = []
        for cmd in command_variants:
            conda_wrapped.append(["conda", "run", "-n", sleap_env, *cmd])
        # When a SLEAP env is explicitly selected, keep export execution in that env.
        # Do not fall back to the MAT process env.
        command_variants = conda_wrapped

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

    model_path = Path(str(config.model_path or "")).expanduser().resolve()
    if _looks_like_sleap_export_path(str(model_path), runtime):
        return str(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise RuntimeError(
            f"SLEAP model path does not exist or is not a directory: {model_path}"
        )

    input_hw = (
        tuple(int(v) for v in config.sleap_export_input_hw)
        if config.sleap_export_input_hw is not None
        else None
    )
    sig_blob = (
        f"{_path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.sleap_batch)}|max_instances={int(config.sleap_max_instances)}|"
        f"input_hw={input_hw}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    export_dir = model_path.parent / f"{model_path.name}.{runtime}"
    if _looks_like_sleap_export_path(
        str(export_dir), runtime
    ) and _artifact_meta_matches(export_dir, sig):
        return str(export_dir.resolve())
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SLEAP model for %s runtime: %s -> %s",
        runtime,
        model_path,
        export_dir,
    )
    if input_hw is not None:
        logger.info(
            "SLEAP export input size hint: %dx%d",
            int(input_hw[0]),
            int(input_hw[1]),
        )
    sleap_env = str(config.sleap_env or "").strip()
    # Preferred/export-stable path: run export through selected SLEAP env.
    ok, err = _attempt_sleap_cli_export(
        model_dir=model_path,
        export_dir=export_dir,
        runtime_flavor=runtime,
        sleap_env=sleap_env,
        input_hw=input_hw,
        batch_size=int(max(1, config.sleap_batch)),
    )
    if not ok and not sleap_env:
        # Dev fallback when no env is configured.
        ok, err = _attempt_sleap_python_export(
            model_dir=model_path,
            export_dir=export_dir,
            runtime_flavor=runtime,
            batch_size=int(max(1, config.sleap_batch)),
            max_instances=int(max(1, config.sleap_max_instances)),
        )
    if not ok or not _looks_like_sleap_export_path(str(export_dir), runtime):
        raise RuntimeError(f"SLEAP auto-export failed for runtime '{runtime}'. {err}")
    _write_artifact_meta(export_dir, sig)
    return str(export_dir.resolve())


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
        runtime_flavor: str = "native",
        exported_model_path: str = "",
        export_input_hw: Optional[Tuple[int, int]] = None,
    ):
        try:
            from multi_tracker.posekit.inference.service import PoseInferenceService
        except ImportError:
            from multi_tracker.posekit_old.pose_inference import PoseInferenceService

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.output_keypoint_names = [str(v) for v in keypoint_names]
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_env = str(sleap_env or "sleap").strip() or "sleap"
        self.sleap_device = str(sleap_device or "auto")
        self.sleap_batch = max(1, int(sleap_batch))
        self.sleap_max_instances = max(1, int(sleap_max_instances))
        self.runtime_flavor = str(runtime_flavor or "native").strip().lower()
        self.exported_model_path = str(exported_model_path or "").strip()
        self.export_input_hw = (
            (int(export_input_hw[0]), int(export_input_hw[1]))
            if isinstance(export_input_hw, (tuple, list))
            and len(export_input_hw) >= 2
            and int(export_input_hw[0]) > 0
            and int(export_input_hw[1]) > 0
            else None
        )
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
        # All SLEAP runtime flavors now execute through the persistent service.
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
                sleap_runtime_flavor=self.runtime_flavor,
                sleap_exported_model_path=self.exported_model_path,
                sleap_export_input_hw=self.export_input_hw,
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
                kwargs["providers"] = ["CUDAExecutionProvider"]
            elif self.device == "cpu":
                kwargs["providers"] = ["CPUExecutionProvider"]
            elif self.device == "mps":
                # ONNXRuntime does not use MPS execution provider.
                kwargs["providers"] = ["CPUExecutionProvider"]
            else:
                kwargs["providers"] = ["CPUExecutionProvider"]
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
        if meta is not None:
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

        # FALLBACK: If input_hw is still None after checking metadata, infer from training config or use safe default
        if self._input_hw is None and meta is not None:
            # Check for max_height/max_width in training config
            max_h = _extract_metadata_attr(meta, names=["max_height"], default=None)
            max_w = _extract_metadata_attr(meta, names=["max_width"], default=None)
            min_crop = _extract_metadata_attr(
                meta, names=["min_crop_size"], default=None
            )

            # For U-Net with dynamic sizes, use a safe default based on training data
            # Round up to next multiple of 32 (common U-Net stride for encoder/decoder matching)
            if max_h is not None and max_w is not None:
                try:
                    target_size = max(int(max_h), int(max_w))
                    # Round up to multiple of 32
                    target_size = ((target_size + 31) // 32) * 32
                    self._input_hw = (target_size, target_size)
                    logger.info(
                        f"SLEAP export: detected dynamic input, using square resize to {target_size}x{target_size}"
                    )
                except Exception:
                    pass

            # Last resort: use minimum crop size rounded to multiple of 32
            if self._input_hw is None and min_crop is not None:
                try:
                    target_size = ((int(min_crop) + 31) // 32) * 32
                    self._input_hw = (target_size, target_size)
                    logger.info(
                        f"SLEAP export: using default square resize to {target_size}x{target_size}"
                    )
                except Exception:
                    pass

        # Final fallback for U-Net models - use safe default divisible by 32
        if self._input_hw is None:
            self._input_hw = (224, 224)
            logger.warning(
                "SLEAP export: no input size in metadata, using default 224x224 for U-Net compatibility."
            )

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

        export_path = Path(self.exported_model_path).expanduser().resolve()
        if export_path.is_dir():
            if self.runtime_flavor == "onnx":
                onnx_files = sorted(export_path.rglob("*.onnx"))
                if onnx_files:
                    export_path = onnx_files[0].resolve()
            elif self.runtime_flavor == "tensorrt":
                engine_files = sorted(
                    list(export_path.rglob("*.engine"))
                    + list(export_path.rglob("*.trt"))
                )
                if engine_files:
                    export_path = engine_files[0].resolve()
        self.exported_model_path = str(export_path)

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

    def _detect_model_min_batch(self) -> Optional[int]:
        """Detect the minimum required batch from the model session/engine."""
        session = None
        for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
            cand = getattr(self._predictor, attr, None)
            if cand is not None and hasattr(cand, "get_inputs"):
                session = cand
                break
        if session is not None:
            try:
                inputs = session.get_inputs()
                if inputs:
                    shape = getattr(inputs[0], "shape", [])
                    if shape:
                        try:
                            b = int(shape[0])
                            if b > 0:
                                return b
                        except (TypeError, ValueError):
                            pass
            except Exception:
                pass
        for attr in ("engine", "_engine", "trt_engine"):
            engine = getattr(self._predictor, attr, None)
            if engine is None:
                continue
            if hasattr(engine, "get_profile_shape"):
                try:
                    min_s, _, _ = engine.get_profile_shape(0, 0)
                    if min_s and int(min_s[0]) > 0:
                        return int(min_s[0])
                except Exception:
                    pass
            if hasattr(engine, "get_binding_shape"):
                try:
                    shape = list(engine.get_binding_shape(0))
                    if shape and int(shape[0]) > 0:
                        return int(shape[0])
                except Exception:
                    pass
        return None

    def _detect_input_format(self) -> Optional[Dict[str, Any]]:
        """Detect expected dtype and layout (NCHW/NHWC) from ONNX session."""
        session = None
        for attr in ("session", "_session", "ort_session", "_ort_session", "sess"):
            cand = getattr(self._predictor, attr, None)
            if cand is not None and hasattr(cand, "get_inputs"):
                session = cand
                break
        if session is None:
            return None
        try:
            inputs = session.get_inputs()
            if not inputs:
                return None
            inp = inputs[0]
            raw_type = str(getattr(inp, "type", "")).lower()
            shape = list(getattr(inp, "shape", []) or [])
        except Exception:
            return None
        is_float = "float" in raw_type
        dims: List[int] = []
        for d in shape:
            try:
                dims.append(int(d))
            except (TypeError, ValueError):
                dims.append(-1)
        layout = "nhwc"
        if len(dims) >= 4:
            if dims[1] in (1, 3):
                layout = "nchw"
            elif dims[-1] in (1, 3):
                layout = "nhwc"
        return {"is_float": is_float, "layout": layout}

    def _predict_raw(self, crops: Sequence[np.ndarray]) -> Any:
        if self._predictor is None:
            raise RuntimeError("SLEAP export predictor is not initialized.")

        # Prepare uint8 NHWC batch – always needed as a base.
        last_err: Optional[Exception] = None
        batch_uint8: Optional[np.ndarray] = None
        try:
            batch_uint8 = self._prepare_inputs_uint8(crops)
        except Exception as exc:
            last_err = exc

        # Detected format from ONNX session (no guessing).
        fmt = self._detect_input_format()
        logger.debug(f"SLEAP export: detected format = {fmt}")
        if fmt is not None and batch_uint8 is not None:
            inp = (
                batch_uint8.astype(np.float32) / 255.0
                if fmt.get("is_float", True)
                else batch_uint8.copy()
            )
            logger.debug(
                f"SLEAP export: before transpose, inp.shape = {inp.shape}, layout = {fmt.get('layout')}"
            )
            if fmt.get("layout") == "nchw" and inp.ndim == 4:
                inp = np.transpose(inp, (0, 3, 1, 2))
                logger.debug(
                    f"SLEAP export: after transpose to NCHW, inp.shape = {inp.shape}"
                )
            try:
                result = self._predictor.predict(inp)
                logger.debug("SLEAP export: prediction succeeded with detected format")
                return result
            except Exception as exc:
                logger.debug(f"SLEAP export: detected format failed: {exc}")
                last_err = exc

        # Fallback: try numpy formats first, list(crops) last.
        attempts: List[Any] = []
        if batch_uint8 is not None:
            if batch_uint8.ndim == 4:
                nchw = np.transpose(
                    batch_uint8.astype(np.float32) / 255.0, (0, 3, 1, 2)
                )
                attempts.append(nchw)  # NCHW float
            attempts.append(batch_uint8.astype(np.float32) / 255.0)  # NHWC float
            attempts.append(batch_uint8)  # NHWC uint8
        attempts.append(list(crops))  # raw list – last resort

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
        actual_count = len(crops)

        # Store original crop dimensions for coordinate rescaling
        original_sizes = []
        for crop in crops:
            arr = np.asarray(crop)
            h, w = arr.shape[:2]
            original_sizes.append((h, w))

        # Pad to model's minimum batch if needed.
        min_b = self._detect_model_min_batch()
        padded_crops: Sequence[np.ndarray] = crops
        if min_b is not None and actual_count < min_b:
            pad_list = list(crops)
            pad_list.extend([crops[-1]] * (min_b - actual_count))
            padded_crops = pad_list
            # Extend original_sizes for padded crops
            for _ in range(min_b - actual_count):
                original_sizes.append(original_sizes[-1])

        raw_out = self._predict_raw(padded_crops)
        kpt_batch = _coerce_prediction_batch(raw_out, batch_size=len(padded_crops))
        # Trim back to actual crop count.
        kpt_batch = kpt_batch[:actual_count]
        original_sizes = original_sizes[:actual_count]

        # Rescale keypoint coordinates from resized space back to original crop space
        if self._input_hw is not None:
            model_h, model_w = self._input_hw
            for i, kpts in enumerate(kpt_batch):
                if kpts is not None and len(kpts) > 0:
                    orig_h, orig_w = original_sizes[i]
                    scale_x = orig_w / model_w
                    scale_y = orig_h / model_h
                    # Scale x, y coordinates (columns 0, 1), leave confidence (column 2) unchanged
                    kpts[:, 0] *= scale_x
                    kpts[:, 1] *= scale_y

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
