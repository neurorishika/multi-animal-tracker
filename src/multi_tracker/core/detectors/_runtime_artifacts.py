"""ONNX and TensorRT runtime artifact management for YOLO detectors.

Provided as a mixin class so that ``YOLOOBBDetector`` can inherit these methods
cleanly. All methods depend only on ``self.params``, ``self.device``, and each
other — no YOLO model state is accessed.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class RuntimeArtifactMixin:
    """Mixin supplying ONNX/TensorRT artifact caching and export helpers."""

    # ------------------------------------------------------------------
    # ONNX resolution
    # ------------------------------------------------------------------

    def _resolve_onnx_imgsz(self, model_path: Path | None = None) -> int:
        """Resolve ONNX export/inference image size.

        Priority:
        1) Explicit `YOLO_ONNX_IMGSZ`
        2) Explicit `YOLO_IMGSZ`
        3) Model metadata from source .pt (`model.overrides['imgsz']` / `model.args['imgsz']`)
        4) Fallback 640
        """
        raw = self.params.get("YOLO_ONNX_IMGSZ", None)
        if raw is None and "YOLO_IMGSZ" in self.params:
            raw = self.params.get("YOLO_IMGSZ")

        imgsz = None
        if raw is not None:
            try:
                imgsz = int(raw)
            except Exception:
                imgsz = None

        if imgsz is None and model_path is not None and model_path.exists():
            try:
                from ultralytics import YOLO

                model = YOLO(str(model_path), task="obb")
                ov = getattr(model, "overrides", {}) or {}
                arg_imgsz = None
                try:
                    arg_imgsz = ov.get("imgsz")
                except Exception:
                    arg_imgsz = None
                if arg_imgsz is None:
                    margs = getattr(getattr(model, "model", None), "args", {}) or {}
                    if isinstance(margs, dict):
                        arg_imgsz = margs.get("imgsz")
                if arg_imgsz is not None:
                    imgsz = int(arg_imgsz)
            except Exception:
                imgsz = None

        if imgsz is None:
            imgsz = 640
        # Keep this aligned with practical YOLO defaults and export constraints.
        imgsz = max(64, min(4096, int(imgsz)))
        return imgsz

    # ------------------------------------------------------------------
    # TensorRT resolution helpers
    # ------------------------------------------------------------------

    def _resolve_tensorrt_build_batch_size(
        self, requested_batch_size: int | None = None
    ) -> int:
        """Resolve the fixed TensorRT engine batch size."""
        default_batch = requested_batch_size
        if default_batch is None:
            default_batch = self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)
        try:
            default_batch = max(1, int(default_batch or 1))
        except (TypeError, ValueError):
            default_batch = 1

        raw_override = self.params.get("TENSORRT_BUILD_BATCH_SIZE", None)
        if raw_override in (None, ""):
            from ._utils import _advanced_config_value

            raw_override = _advanced_config_value(
                self.params, "tensorrt_build_batch_size", None
            )
        if raw_override in (None, "", 0, "0"):
            return default_batch
        try:
            return max(1, int(raw_override))
        except (TypeError, ValueError):
            return default_batch

    def _resolve_tensorrt_workspace_gb(self) -> float:
        """Resolve TensorRT builder workspace limit in GB."""
        raw_value = self.params.get("TENSORRT_BUILD_WORKSPACE_GB", None)
        if raw_value in (None, ""):
            from ._utils import _advanced_config_value

            raw_value = _advanced_config_value(
                self.params, "tensorrt_build_workspace_gb", 4.0
            )
        try:
            return max(0.5, float(raw_value))
        except (TypeError, ValueError):
            return 4.0

    def _get_tensorrt_build_context(self) -> dict[str, str]:
        """Summarize device/runtime details for TensorRT build logging."""
        from multi_tracker.utils.gpu_utils import get_device_info

        gpu_name = str(self.device)
        cuda_version = "unknown"
        try:
            info = get_device_info()
        except Exception:
            info = {}
        if isinstance(info, dict):
            gpu_name = str(
                info.get("torch_cuda_device_name")
                or info.get("cuda_device_name")
                or gpu_name
            )
        try:
            import torch

            cuda_version = str(
                getattr(getattr(torch, "version", None), "cuda", None) or "unknown"
            )
        except Exception:
            cuda_version = "unknown"
        return {"gpu_name": gpu_name, "cuda_version": cuda_version}

    # ------------------------------------------------------------------
    # Artifact signature / metadata
    # ------------------------------------------------------------------

    def _artifact_signature(
        self, runtime: str, batch_size: int = 1, onnx_imgsz: int | None = None
    ) -> str:
        engine_model_id = self.params.get("ENGINE_MODEL_ID")
        inference_model_id = self.params.get("INFERENCE_MODEL_ID")
        if str(runtime) == "tensorrt" and engine_model_id:
            token = str(engine_model_id)
        elif inference_model_id:
            token = str(inference_model_id)
        else:
            token = str(
                self.params.get(
                    "YOLO_MODEL_PATH", getattr(self, "active_obb_model_path", "")
                )
            )
        runtime_profile = str(runtime)
        if str(runtime) == "onnx":
            # Keep ONNX export profile explicit in cache signature so profile changes
            # always trigger a rebuild of potentially incompatible artifacts.
            resolved_imgsz = int(onnx_imgsz or self._resolve_onnx_imgsz())
            runtime_profile = f"onnx_v3_static_imgsz{resolved_imgsz}_opset17_nosimplify"
        return hashlib.sha1(
            f"{token}|runtime={runtime_profile}|batch={int(batch_size)}".encode("utf-8")
        ).hexdigest()[:16]

    def _artifact_meta_path(self, artifact_path: Path) -> Path:
        return artifact_path.with_suffix(f"{artifact_path.suffix}.runtime_meta.json")

    def _artifact_is_fresh(self, artifact_path: Path, signature: str) -> bool:
        if not artifact_path.exists():
            return False
        data = self._read_artifact_meta(artifact_path)
        if not data:
            return False
        return str(data.get("signature", "")) == str(signature)

    def _read_artifact_meta(self, artifact_path: Path) -> dict:
        meta_path = self._artifact_meta_path(artifact_path)
        if not meta_path.exists():
            return {}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_artifact_meta(
        self, artifact_path: Path, signature: str, **extra_meta
    ) -> None:
        meta_path = self._artifact_meta_path(artifact_path)
        payload = {"signature": str(signature)}
        payload.update({str(k): v for k, v in extra_meta.items()})
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # ONNX model loading / export
    # ------------------------------------------------------------------

    def _try_load_onnx_model(self, model_path_str):
        """Try to load or export ONNX model for CPU runtime."""
        try:
            from ultralytics import YOLO

            resolved_model = Path(model_path_str).expanduser().resolve()
            onnx_batch_size = max(1, int(self.params.get("TENSORRT_MAX_BATCH_SIZE", 1)))
            if resolved_model.suffix.lower() == ".onnx":
                # User supplied explicit ONNX artifact path.
                onnx_path = resolved_model
                if not onnx_path.exists():
                    raise RuntimeError(f"ONNX model path not found: {onnx_path}")
                meta = self._read_artifact_meta(onnx_path)
                try:
                    meta_imgsz = int(meta.get("imgsz", 0))
                except Exception:
                    meta_imgsz = 0
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    onnx_batch_size = meta_batch
                else:
                    # Unknown exported batch dimension for user-supplied ONNX.
                    # Keep conservative default to avoid invalid-batch errors.
                    onnx_batch_size = 1
                onnx_imgsz = (
                    meta_imgsz
                    if meta_imgsz > 0
                    else self._resolve_onnx_imgsz(model_path=resolved_model)
                )
                logger.info(f"Loading ONNX model from: {onnx_path}")
                self.model = YOLO(str(onnx_path), task="obb")
                self.use_onnx = True
                self.onnx_model_path = str(onnx_path)
                self.onnx_imgsz = int(onnx_imgsz)
                self.onnx_batch_size = int(onnx_batch_size)
                return
            else:
                onnx_path = resolved_model.with_name(
                    f"{resolved_model.stem}_b{onnx_batch_size}.onnx"
                )
            onnx_imgsz = self._resolve_onnx_imgsz(model_path=resolved_model)
            signature = self._artifact_signature(
                runtime="onnx",
                batch_size=int(onnx_batch_size),
                onnx_imgsz=onnx_imgsz,
            )

            if self._artifact_is_fresh(onnx_path, signature):
                meta = self._read_artifact_meta(onnx_path)
                try:
                    meta_imgsz = int(meta.get("imgsz", 0))
                except Exception:
                    meta_imgsz = 0
                if meta_imgsz > 0:
                    onnx_imgsz = meta_imgsz
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    onnx_batch_size = meta_batch
                logger.info(f"Loading ONNX model from: {onnx_path}")
                self.model = YOLO(str(onnx_path), task="obb")
                self.use_onnx = True
                self.onnx_model_path = str(onnx_path)
                self.onnx_imgsz = onnx_imgsz
                self.onnx_batch_size = int(onnx_batch_size)
                return

            logger.info("Exporting YOLO OBB model to ONNX runtime artifact...")
            base_model = YOLO(str(resolved_model), task="obb")
            export_path = base_model.export(
                format="onnx",
                imgsz=onnx_imgsz,
                dynamic=False,
                simplify=False,
                nms=False,
                opset=17,
                batch=int(onnx_batch_size),
                verbose=False,
            )
            out_path = Path(export_path).expanduser().resolve()
            if not out_path.exists():
                raise RuntimeError(f"ONNX export output missing: {out_path}")
            if out_path != onnx_path:
                shutil.copy2(str(out_path), str(onnx_path))
            self._write_artifact_meta(
                onnx_path,
                signature,
                imgsz=int(onnx_imgsz),
                batch_size=int(onnx_batch_size),
            )
            self.model = YOLO(str(onnx_path), task="obb")
            self.use_onnx = True
            self.onnx_model_path = str(onnx_path)
            self.onnx_imgsz = onnx_imgsz
            self.onnx_batch_size = int(onnx_batch_size)
            logger.info(f"ONNX model ready: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX runtime optimization failed: {e}")
            self.use_onnx = False

    # ------------------------------------------------------------------
    # TensorRT model loading / export
    # ------------------------------------------------------------------

    def _try_load_tensorrt_model(self, model_path_str):
        """Try to load or export TensorRT model for faster inference."""
        try:
            from ultralytics import YOLO

            requested_batch_size = self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)
            build_batch_size = self._resolve_tensorrt_build_batch_size(
                requested_batch_size
            )
            build_workspace_gb = self._resolve_tensorrt_workspace_gb()
            build_context = self._get_tensorrt_build_context()

            resolved_model = Path(model_path_str).expanduser().resolve()
            if resolved_model.suffix.lower() in {".engine", ".trt"}:
                engine_path = resolved_model
                meta = self._read_artifact_meta(engine_path)
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    build_batch_size = meta_batch
            else:
                engine_path = resolved_model.with_name(
                    f"{resolved_model.stem}_b{int(build_batch_size)}.engine"
                )
            signature = self._artifact_signature(
                runtime="tensorrt", batch_size=int(build_batch_size)
            )

            logger.info(
                "TensorRT engine cache check: path=%s | signature=%s | build_batch=%d | workspace=%.1f GB | gpu=%s | cuda=%s",
                engine_path,
                signature,
                int(build_batch_size),
                float(build_workspace_gb),
                build_context["gpu_name"],
                build_context["cuda_version"],
            )

            # Check if TensorRT engine already exists and matches current inference signature
            if self._artifact_is_fresh(engine_path, signature):
                logger.info("TensorRT engine cache hit: reusing cached engine.")
                try:
                    self.model = YOLO(str(engine_path), task="obb")
                    self.use_tensorrt = True
                    self.tensorrt_model_path = str(engine_path)
                    self.tensorrt_batch_size = (
                        build_batch_size  # Store batch size for chunking
                    )
                    logger.info(
                        "TensorRT engine reused successfully: path=%s | build_batch=%d",
                        engine_path,
                        int(build_batch_size),
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "Failed to load cached TensorRT engine %s: %s",
                        engine_path,
                        e,
                    )
                    engine_path.unlink(missing_ok=True)

            logger.info(
                "TensorRT engine cache miss or stale metadata detected; rebuilding."
            )

            # Export to TensorRT
            logger.info("=" * 60)
            logger.info("BUILDING TENSORRT ENGINE - This is a one-time optimization")
            logger.info("This may take 1-5 minutes. Please wait...")
            logger.info("Engine path: %s", engine_path)
            logger.info("Engine signature: %s", signature)
            logger.info(
                "Build settings: batch=%d | workspace=%.1f GB | gpu=%s | cuda=%s",
                int(build_batch_size),
                float(build_workspace_gb),
                build_context["gpu_name"],
                build_context["cuda_version"],
            )
            logger.info("=" * 60)
            base_model = YOLO(model_path_str)
            base_model.to(self.device)

            # Try dynamic batching first, fall back to static if it fails
            logger.info(
                "Building TensorRT engine (batch size: %d, workspace: %.1f GB)...",
                int(build_batch_size),
                float(build_workspace_gb),
            )

            # Export to TensorRT engine format
            # Note: dynamic=False uses fixed batch size which is more compatible
            # but requires batches to exactly match max_batch_size
            export_path = base_model.export(
                format="engine",
                device=self.device,
                half=True,  # Use FP16 for faster inference
                workspace=float(build_workspace_gb),
                dynamic=False,  # Static shapes (more compatible)
                batch=int(build_batch_size),  # Fixed batch size
                verbose=False,
            )

            # Move exported engine to cache directory
            if Path(export_path).exists():
                exported_path = Path(export_path).expanduser().resolve()
                if exported_path != engine_path:
                    shutil.copy2(str(exported_path), str(engine_path))
                self._write_artifact_meta(
                    engine_path,
                    signature,
                    batch_size=int(build_batch_size),
                    workspace_gb=float(build_workspace_gb),
                    gpu_name=build_context["gpu_name"],
                    cuda_version=build_context["cuda_version"],
                )
                logger.info(
                    "TensorRT engine rebuilt and cached: path=%s | signature=%s",
                    engine_path,
                    signature,
                )

                # Load the TensorRT model
                self.model = YOLO(str(engine_path), task="obb")
                self.use_tensorrt = True
                self.tensorrt_model_path = str(engine_path)
                self.tensorrt_batch_size = build_batch_size  # Store for batching logic
                logger.info("=" * 60)
                logger.info(
                    "TENSORRT OPTIMIZATION COMPLETE (batch=%d, workspace=%.1f GB)",
                    int(build_batch_size),
                    float(build_workspace_gb),
                )
                logger.info("=" * 60)
            else:
                logger.warning("TensorRT export failed - exported file not found")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"TensorRT optimization failed: {error_msg}")

            # Provide helpful suggestions based on error type
            build_batch_size = self._resolve_tensorrt_build_batch_size(
                self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)
            )
            if "memory" in error_msg.lower() or "allocate" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT build ran out of GPU memory (build batch = {build_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning(f"Try: 8, 4, or 1 instead of {build_batch_size}")
                logger.warning("=" * 60)
            elif "engine build failed" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT engine build failed (build batch = {build_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning("=" * 60)

            logger.info("Continuing with standard PyTorch inference (still uses GPU)")
            self.use_tensorrt = False

    # ------------------------------------------------------------------
    # Ultralytics logging suppression
    # ------------------------------------------------------------------

    def _configure_ultralytics_logging(self):
        """Reduce per-frame Ultralytics runtime banners unless explicitly requested."""
        if bool(self.params.get("YOLO_VERBOSE_ULTRALYTICS", False)):
            return
        try:
            from ultralytics.utils import LOGGER as ULTRA_LOGGER

            ULTRA_LOGGER.setLevel(logging.WARNING)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Auxiliary task artifact preparation
    # ------------------------------------------------------------------

    def _prepare_runtime_artifact_for_task(self, model_path_str: str, task: str) -> str:
        """Resolve/export runtime artifact for auxiliary YOLO tasks.

        This keeps sequential stage-1 detect/classify aligned with the selected
        MAT compute runtime. Explicit runtime artifacts (.onnx/.engine/.trt)
        are used as-is. For local .pt checkpoints, ONNX/TensorRT artifacts are
        exported lazily when requested by runtime flags.
        """
        if not model_path_str:
            return model_path_str

        # Built-in model aliases are loaded directly by ultralytics.
        if str(model_path_str).startswith(("yolo26", "yolov8", "yolov11")):
            return model_path_str

        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists() or not model_path.is_file():
            return model_path_str

        suffix = model_path.suffix.lower()
        if suffix in {".onnx", ".engine", ".trt"}:
            return str(model_path)
        if suffix != ".pt":
            return model_path_str

        try:
            from ultralytics import YOLO

            from multi_tracker.utils.gpu_utils import (
                ONNXRUNTIME_AVAILABLE,
                TENSORRT_AVAILABLE,
            )

            enable_onnx_runtime = bool(self.params.get("ENABLE_ONNX_RUNTIME", False))
            enable_tensorrt = bool(self.params.get("ENABLE_TENSORRT", False)) and str(
                self.device
            ).startswith("cuda")

            # Make stage artifact names unambiguous when direct/crop/detect models differ.
            task_tag = str(task or "task").strip().lower().replace(" ", "_")

            # Prefer ONNX when requested, matching primary OBB runtime preference.
            if enable_onnx_runtime and ONNXRUNTIME_AVAILABLE:
                onnx_path = model_path.with_name(
                    f"{model_path.stem}_{task_tag}_b1.onnx"
                )
                needs_build = (not onnx_path.exists()) or (
                    onnx_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
                )
                if needs_build:
                    logger.info("Exporting %s model to ONNX runtime artifact...", task)
                    if task == "detect":
                        seq_detect_imgsz = int(
                            self.params.get("YOLO_SEQ_DETECT_IMGSZ", 0)
                        )
                        onnx_imgsz = (
                            seq_detect_imgsz
                            if seq_detect_imgsz > 0
                            else self._resolve_onnx_imgsz(model_path=model_path)
                        )
                    else:
                        onnx_imgsz = self._resolve_onnx_imgsz(model_path=model_path)
                    base_model = YOLO(str(model_path), task=task)
                    export_path = base_model.export(
                        format="onnx",
                        imgsz=int(onnx_imgsz),
                        dynamic=False,
                        simplify=False,
                        nms=False,
                        opset=17,
                        batch=1,
                        verbose=False,
                    )
                    out_path = Path(export_path).expanduser().resolve()
                    if out_path.exists() and out_path != onnx_path:
                        shutil.copy2(str(out_path), str(onnx_path))
                if onnx_path.exists():
                    return str(onnx_path)

            if enable_tensorrt and TENSORRT_AVAILABLE:
                engine_path = model_path.with_name(
                    f"{model_path.stem}_{task_tag}_b1.engine"
                )
                needs_build = (not engine_path.exists()) or (
                    engine_path.stat().st_mtime_ns < model_path.stat().st_mtime_ns
                )
                if needs_build:
                    build_workspace_gb = self._resolve_tensorrt_workspace_gb()
                    logger.info(
                        "Building TensorRT runtime artifact for %s model (workspace=%.1f GB)...",
                        task,
                        float(build_workspace_gb),
                    )
                    base_model = YOLO(str(model_path), task=task)
                    base_model.to(self.device)
                    export_path = base_model.export(
                        format="engine",
                        device=self.device,
                        half=True,
                        workspace=float(build_workspace_gb),
                        dynamic=False,
                        batch=1,
                        verbose=False,
                    )
                    out_path = Path(export_path).expanduser().resolve()
                    if out_path.exists() and out_path != engine_path:
                        shutil.copy2(str(out_path), str(engine_path))
                if engine_path.exists():
                    return str(engine_path)
        except Exception as exc:
            logger.warning(
                "Aux runtime artifact preparation failed for %s model (%s). Using source checkpoint.",
                task,
                exc,
            )

        return model_path_str
