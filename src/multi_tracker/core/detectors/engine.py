"""
Object detection utilities for multi-object tracking.
Supports both background subtraction and YOLO OBB detection methods.
"""

import hashlib
import json
import logging
import math
import shutil
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _normalize_detection_ids(detection_ids):
    """Normalize detection IDs to runtime integers.

    Cached raw detections may come back as float-backed arrays from older NPZ
    files. Accept finite whole-number values to preserve compatibility.
    """
    if detection_ids is None:
        return None

    arr = np.asarray(detection_ids)
    if arr.size == 0:
        return []

    normalized = []
    for raw_value in arr.reshape(-1).tolist():
        if isinstance(raw_value, (np.integer, int)):
            normalized.append(int(raw_value))
            continue

        try:
            float_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid detection ID value: {raw_value!r}") from exc

        if not np.isfinite(float_value) or not float_value.is_integer():
            raise ValueError(
                f"Detection ID must be a finite whole number, got {raw_value!r}"
            )
        normalized.append(int(float_value))

    return normalized


_HEADTAIL_DIRECTIONAL_CLASS_SET = frozenset({"left", "right"})
_HEADTAIL_FIVE_CLASS_SET = frozenset({"up", "down", "left", "right", "unknown"})


class ObjectDetector:
    """
    Detects objects in foreground masks and extracts measurements.
    """

    def __init__(self, params):
        self.params = params

    def _local_conservative_split(self, sub):
        """Applies conservative morphological operations to split merged objects."""
        k = self.params["CONSERVATIVE_KERNEL_SIZE"]
        it = self.params["CONSERVATIVE_ERODE_ITER"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.erode(sub, kernel, iterations=it)
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    def apply_conservative_split(self: object, fg_mask: object) -> object:
        """Attempts to split merged objects in the foreground mask."""
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        N = self.params["MAX_TARGETS"]

        suspicious = [
            cv2.boundingRect(c)
            for c in cnts
            if cv2.contourArea(c) > self.params["MERGE_AREA_THRESHOLD"]
            or sum(1 for cc in cnts if cv2.contourArea(cc) > 0) < N
        ]

        for bx, by, bw, bh in suspicious:
            sub = fg_mask[by : by + bh, bx : bx + bw]
            fg_mask[by : by + bh, bx : bx + bw] = self._local_conservative_split(sub)

        return fg_mask

    def detect_objects(self: object, fg_mask: object, frame_count: object) -> object:
        """Detects and measures objects from the final foreground mask.

        Returns:
            meas: List of measurements [cx, cy, angle]
            sizes: List of detection areas
            shapes: List of (area, aspect_ratio) tuples
            yolo_results: None (for compatibility with YOLO detector)
            confidences: List of detection confidence scores (0-1)
        """
        p = self.params
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        N = p["MAX_TARGETS"]
        max_allowed_contours = N * p.get("MAX_CONTOUR_MULTIPLIER", 20)

        if len(cnts) > max_allowed_contours:
            logger.debug(
                f"Frame {frame_count}: Too many contours ({len(cnts)}), skipping."
            )
            return [], [], [], None, []

        meas, sizes, shapes, confidences = [], [], [], []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p["MIN_CONTOUR_AREA"] or len(c) < 5:
                continue

            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)

            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                ang = (ang + 90) % 180

            # Detection confidence not feasible for background subtraction
            # Quality is too context-specific (lighting, camera, species, etc.)
            confidence = np.nan

            meas.append(np.array([cx, cy, np.deg2rad(ang)], np.float32))
            sizes.append(area)
            shapes.append((np.pi * (ax1 / 2) * (ax2 / 2), ax1 / ax2 if ax2 > 0 else 0))
            confidences.append(confidence)

        if meas and p.get("ENABLE_SIZE_FILTERING", False):
            min_size = p.get("MIN_OBJECT_SIZE", 0)
            max_size = p.get("MAX_OBJECT_SIZE", float("inf"))

            original_count = len(meas)
            filtered = [
                (m, s, sh, conf)
                for m, s, sh, conf in zip(meas, sizes, shapes, confidences)
                if min_size <= s <= max_size
            ]

            if filtered:
                meas, sizes, shapes, confidences = zip(*filtered)
                meas, sizes, shapes, confidences = (
                    list(meas),
                    list(sizes),
                    list(shapes),
                    list(confidences),
                )
            else:
                meas, sizes, shapes, confidences = [], [], [], []

            if len(meas) != original_count:
                logger.debug(
                    f"Size filtering: {original_count} -> {len(meas)} detections"
                )

        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]
            confidences = [confidences[i] for i in idxs]

        return meas, sizes, shapes, None, confidences


class YOLOOBBDetector:
    """
    Detects objects using a pretrained YOLO OBB (Oriented Bounding Box) model.
    Compatible interface with ObjectDetector for seamless integration.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.detect_model = None
        self.headtail_model = None
        self.headtail_backend = "none"
        self.headtail_class_names = None  # populated for classkit_tiny N-class models
        self.headtail_input_size = None  # (w, h) used during classkit_tiny training
        self.obb_predict_device = None
        self.detect_predict_device = None
        self.headtail_predict_device = None
        self.device = self._detect_device()
        self.use_tensorrt = False
        self.use_onnx = False
        self.onnx_imgsz = None
        self.onnx_batch_size = 1
        self.tensorrt_batch_size = 1
        self.obb_mode = str(self.params.get("YOLO_OBB_MODE", "direct")).strip().lower()
        if self.obb_mode not in {"direct", "sequential"}:
            self.obb_mode = "direct"
        self.direct_model_path = str(
            self.params.get(
                "YOLO_OBB_DIRECT_MODEL_PATH",
                self.params.get("YOLO_MODEL_PATH", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        self.detect_model_path = str(
            self.params.get("YOLO_DETECT_MODEL_PATH", "") or ""
        ).strip()
        self.crop_obb_model_path = str(
            self.params.get("YOLO_CROP_OBB_MODEL_PATH", "") or ""
        ).strip()
        self.headtail_model_path = str(
            self.params.get("YOLO_HEADTAIL_MODEL_PATH", "") or ""
        ).strip()
        self.active_obb_model_path = (
            self.direct_model_path
            if self.obb_mode == "direct"
            else (self.crop_obb_model_path or self.direct_model_path)
        )
        # Keep legacy field in sync for downstream code that still reads YOLO_MODEL_PATH.
        self.params["YOLO_MODEL_PATH"] = self.active_obb_model_path
        self._load_model()
        self._load_aux_models()

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

    def _artifact_signature(
        self, runtime: str, batch_size: int = 1, onnx_imgsz: int | None = None
    ) -> str:
        inference_model_id = self.params.get("INFERENCE_MODEL_ID")
        if inference_model_id:
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
            self.onnx_imgsz = onnx_imgsz
            self.onnx_batch_size = int(onnx_batch_size)
            logger.info(f"ONNX model ready: {onnx_path}")
        except Exception as e:
            logger.warning(f"ONNX runtime optimization failed: {e}")
            self.use_onnx = False

    def _try_load_tensorrt_model(self, model_path_str):
        """Try to load or export TensorRT model for faster inference."""
        try:
            from ultralytics import YOLO

            # Get max batch size from UI parameter
            max_batch_size = self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)

            resolved_model = Path(model_path_str).expanduser().resolve()
            if resolved_model.suffix.lower() in {".engine", ".trt"}:
                engine_path = resolved_model
                meta = self._read_artifact_meta(engine_path)
                try:
                    meta_batch = int(meta.get("batch_size", 0))
                except Exception:
                    meta_batch = 0
                if meta_batch > 0:
                    max_batch_size = meta_batch
                else:
                    max_batch_size = 1
            else:
                engine_path = resolved_model.with_name(
                    f"{resolved_model.stem}_b{int(max_batch_size)}.engine"
                )
            signature = self._artifact_signature(
                runtime="tensorrt", batch_size=int(max_batch_size)
            )

            # Check if TensorRT engine already exists and matches current inference signature
            if self._artifact_is_fresh(engine_path, signature):
                logger.info(f"Loading cached TensorRT engine: {engine_path}")
                try:
                    self.model = YOLO(str(engine_path), task="obb")
                    self.use_tensorrt = True
                    self.tensorrt_batch_size = (
                        max_batch_size  # Store batch size for chunking
                    )
                    logger.info(
                        f"TensorRT model loaded successfully (max batch={max_batch_size}, 2-5x faster inference expected)"
                    )
                    return
                except Exception as e:
                    logger.warning(f"Failed to load cached TensorRT engine: {e}")
                    engine_path.unlink(missing_ok=True)

            # Export to TensorRT
            logger.info("=" * 60)
            logger.info("BUILDING TENSORRT ENGINE - This is a one-time optimization")
            logger.info("This may take 1-5 minutes. Please wait...")
            logger.info("The engine will be stored next to the source model.")
            logger.info("=" * 60)
            base_model = YOLO(model_path_str)
            base_model.to(self.device)

            # Try dynamic batching first, fall back to static if it fails
            logger.info(f"Building TensorRT engine (batch size: {max_batch_size})...")

            # Export to TensorRT engine format
            # Note: dynamic=False uses fixed batch size which is more compatible
            # but requires batches to exactly match max_batch_size
            export_path = base_model.export(
                format="engine",
                device=self.device,
                half=True,  # Use FP16 for faster inference
                workspace=4,  # 4GB workspace
                dynamic=False,  # Static shapes (more compatible)
                batch=max_batch_size,  # Fixed batch size
                verbose=False,
            )

            # Move exported engine to cache directory
            if Path(export_path).exists():
                exported_path = Path(export_path).expanduser().resolve()
                if exported_path != engine_path:
                    shutil.copy2(str(exported_path), str(engine_path))
                self._write_artifact_meta(
                    engine_path, signature, batch_size=int(max_batch_size)
                )
                logger.info(f"TensorRT engine exported and cached: {engine_path}")

                # Load the TensorRT model
                self.model = YOLO(str(engine_path), task="obb")
                self.use_tensorrt = True
                self.tensorrt_batch_size = max_batch_size  # Store for batching logic
                logger.info("=" * 60)
                logger.info(f"TENSORRT OPTIMIZATION COMPLETE (batch={max_batch_size})")
                logger.info("=" * 60)
            else:
                logger.warning("TensorRT export failed - exported file not found")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"TensorRT optimization failed: {error_msg}")

            # Provide helpful suggestions based on error type
            max_batch_size = self.params.get("TENSORRT_MAX_BATCH_SIZE", 16)
            if "memory" in error_msg.lower() or "allocate" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT build ran out of GPU memory (max batch = {max_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning(f"Try: 8, 4, or 1 instead of {max_batch_size}")
                logger.warning("=" * 60)
            elif "engine build failed" in error_msg.lower():
                logger.warning("=" * 60)
                logger.warning(
                    f"TensorRT engine build failed (max batch = {max_batch_size})."
                )
                logger.warning(
                    "FIX: Reduce 'TensorRT Max Batch Size' in YOLO settings."
                )
                logger.warning("=" * 60)

            logger.info("Continuing with standard PyTorch inference (still uses GPU)")
            self.use_tensorrt = False

    def _detect_device(self):
        """Detect and configure the optimal device for inference."""
        from multi_tracker.utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE

        # Check user preference
        device_preference = self.params.get("YOLO_DEVICE", "auto")

        if device_preference != "auto":
            logger.info(f"Using user-specified device: {device_preference}")
            return device_preference

        # Auto-detect best available device using centralized gpu_utils
        if TORCH_CUDA_AVAILABLE:
            device = "cuda:0"
            logger.info(f"CUDA GPU detected, using {device}")
        elif MPS_AVAILABLE:
            device = "mps"  # Apple Silicon GPU
            logger.info("Apple Metal Performance Shaders (MPS) detected, using mps")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")

        return device

    def _configure_ultralytics_logging(self):
        """Reduce per-frame Ultralytics runtime banners unless explicitly requested."""
        if bool(self.params.get("YOLO_VERBOSE_ULTRALYTICS", False)):
            return
        try:
            from ultralytics.utils import LOGGER as ULTRA_LOGGER

            ULTRA_LOGGER.setLevel(logging.WARNING)
        except Exception:
            pass

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
                    logger.info(
                        "Building TensorRT runtime artifact for %s model...", task
                    )
                    base_model = YOLO(str(model_path), task=task)
                    base_model.to(self.device)
                    export_path = base_model.export(
                        format="engine",
                        device=self.device,
                        half=True,
                        workspace=4,
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

    def _load_model_for_task(self, model_path_str: str, task: str):
        """Load an auxiliary YOLO model for detect/classify tasks."""
        if not model_path_str:
            return None, None
        from ultralytics import YOLO

        self._configure_ultralytics_logging()

        runtime_model_path_str = self._prepare_runtime_artifact_for_task(
            model_path_str, task
        )

        model_path = Path(runtime_model_path_str).expanduser().resolve()
        use_builtin = runtime_model_path_str.startswith(("yolo26", "yolov8", "yolov11"))
        if use_builtin:
            model = YOLO(runtime_model_path_str, task=task)
        else:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLO {task} model file not found: {runtime_model_path_str}"
                )
            model = YOLO(str(model_path), task=task)
        is_pytorch_checkpoint = use_builtin or model_path.suffix.lower() == ".pt"
        predict_device = self.device
        if is_pytorch_checkpoint:
            try:
                model.to(self.device)
                # Avoid passing device per inference call when model is already placed.
                # This prevents repeated select_device() logs in preview loops.
                predict_device = None
            except Exception:
                # Fallback to per-call device argument for compatibility.
                predict_device = self.device
        return model, predict_device

    def _build_tiny_head_classifier(self, input_size=(128, 64)):
        """Build notebook-compatible tiny head direction classifier."""
        import torch.nn as nn

        class _TinyHeadClassifier(nn.Module):
            def __init__(self, input_size=(128, 64)):
                super().__init__()
                self.input_size = tuple(input_size)
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )

            def forward(self, x):  # noqa: DC04  (nn.Module; called via __call__)
                x = self.features(x)
                return self.classifier(x)

        return _TinyHeadClassifier(input_size=input_size)

    def _try_load_tiny_head_classifier(self, model_path_str: str):
        """Load a tiny head-tail classifier (.pth checkpoint).

        Supports two checkpoint formats:
                * **Notebook/legacy binary** – older single-output checkpoints.
          Classifier has a single output (sigmoid). Returns ``(model, None, input_size)``.
        * **ClassKit N-class** – produced by ``_train_tiny_classify`` in runner.py.
          Classifier has N outputs (softmax). Stores ``class_names`` alongside the model.
          Returns ``(model, class_names, input_size)``.

        Returns ``None`` when the file is not a recognised tiny checkpoint.
        """
        import torch

        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            return None
        if model_path.suffix.lower() not in {".pth", ".pt"}:
            return None

        try:
            checkpoint = torch.load(
                str(model_path), map_location="cpu", weights_only=False
            )
        except Exception:
            return None

        state_dict = None
        input_size = (128, 64)
        class_names = None

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint.get("model_state_dict")
            maybe_size = checkpoint.get("input_size")
            if isinstance(maybe_size, (list, tuple)) and len(maybe_size) == 2:
                input_size = (int(maybe_size[0]), int(maybe_size[1]))
            # ClassKit checkpoints include class_names
            raw_names = checkpoint.get("class_names")
            if isinstance(raw_names, (list, tuple)) and raw_names:
                class_names = [str(n) for n in raw_names]
        elif isinstance(checkpoint, (dict, OrderedDict)):
            # Raw state-dict save path used in old HeadTail notebooks
            state_dict = checkpoint
        else:
            return None

        if not isinstance(state_dict, (dict, OrderedDict)):
            return None
        keys = list(state_dict.keys())
        if not keys:
            return None
        if not any(str(k).startswith("features.") for k in keys):
            return None

        # Detect N-class vs binary by inspecting the last Linear output size.
        linear_classifier_keys = sorted(
            [k for k in keys if k.startswith("classifier.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        if not linear_classifier_keys:
            return None
        last_weight = state_dict[linear_classifier_keys[-1]]
        n_out = int(last_weight.shape[0])

        if n_out == 1:
            # Binary notebook-format model – use the local minimal architecture.
            try:
                model = self._build_tiny_head_classifier(input_size=input_size)
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                return None
        else:
            # ClassKit N-class model – reconstruct via training.tiny_model.
            try:
                from multi_tracker.training.tiny_model import rebuild_from_checkpoint

                model = rebuild_from_checkpoint({"model_state_dict": state_dict})
            except Exception as exc:
                logger.warning(
                    "Failed to load ClassKit tiny head-tail classifier: %s", exc
                )
                return None

        model.to(self.device)
        model.eval()
        return model, class_names, input_size

    def _load_headtail_model(self, model_path_str: str):
        """Load optional head-tail model (tiny .pth or YOLO classify)."""
        tiny_result = self._try_load_tiny_head_classifier(model_path_str)
        if tiny_result is not None:
            tiny_model, class_names, input_size = tiny_result
            self.headtail_class_names = (
                self._validate_headtail_class_names(
                    class_names,
                    source=f"head-tail checkpoint {Path(model_path_str).name}",
                )
                if class_names is not None
                else None
            )
            self.headtail_input_size = input_size
            self.headtail_predict_device = None
            if class_names is not None:
                self.headtail_backend = "classkit_tiny"
                self.headtail_model = tiny_model
                logger.info(
                    "Loaded ClassKit tiny head-tail classifier (%d classes: %s).",
                    len(self.headtail_class_names),
                    ", ".join(self.headtail_class_names[:8]),
                )
            else:
                self.headtail_backend = "tiny"
                self.headtail_model = tiny_model
                logger.info("Loaded notebook tiny head-tail classifier.")
            return

        model, predict_device = self._load_model_for_task(
            model_path_str, task="classify"
        )
        model_names = getattr(model, "names", None)
        if model_names is None:
            model_names = getattr(getattr(model, "model", None), "names", None)
        self.headtail_class_names = self._validate_headtail_class_names(
            model_names,
            source=f"head-tail model {Path(model_path_str).name}",
        )
        self.headtail_backend = "yolo"
        self.headtail_model = model
        self.headtail_predict_device = predict_device

    def _load_aux_models(self):
        """Load optional sequential + head-tail models."""
        if self.obb_mode == "sequential":
            if not self.detect_model_path:
                raise ValueError(
                    "Sequential YOLO OBB mode requires YOLO_DETECT_MODEL_PATH."
                )
            self.detect_model, self.detect_predict_device = self._load_model_for_task(
                self.detect_model_path, task="detect"
            )
            logger.info("YOLO detect model loaded for sequential mode.")

        if self.headtail_model_path:
            self._load_headtail_model(self.headtail_model_path)
            if self.headtail_backend not in ("yolo", "none"):
                logger.info(
                    "Head-tail tiny classifier model loaded (%s).",
                    self.headtail_backend,
                )
            else:
                logger.info("YOLO head-tail classify model loaded.")

    def _runtime_fixed_batch_size(self) -> int:
        """Return fixed runtime batch size when backend enforces static batch dims."""
        if self.use_tensorrt and int(getattr(self, "tensorrt_batch_size", 1)) > 1:
            return int(self.tensorrt_batch_size)
        if self.use_onnx and int(getattr(self, "onnx_batch_size", 1)) > 1:
            return int(self.onnx_batch_size)
        return 1

    def _predict_obb_results(
        self, source, target_classes, raw_conf_floor, max_det, imgsz=None
    ):
        """Run OBB model prediction with backend-specific constraints."""
        fixed_batch = self._runtime_fixed_batch_size()
        # obb_predict_device is None when the model was placed via .to(device).
        # Always fall back to self.device so the explicit device= argument is always
        # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
        predict_device = getattr(self, "obb_predict_device", None) or self.device

        if isinstance(source, list):
            if len(source) == 0:
                return []

            # Static-batch runtimes require exact batch size. Chunk and pad to avoid
            # invalid-batch errors while still leveraging one predict() call per chunk.
            if fixed_batch > 1:
                all_results = []
                for chunk_start in range(0, len(source), fixed_batch):
                    chunk = list(source[chunk_start : chunk_start + fixed_batch])
                    actual_chunk = len(chunk)
                    if actual_chunk < fixed_batch:
                        chunk.extend([chunk[0]] * (fixed_batch - actual_chunk))
                    predict_kwargs = dict(
                        source=chunk,
                        conf=raw_conf_floor,
                        iou=1.0,  # Always use custom OBB IOU filtering after inference
                        classes=target_classes,
                        max_det=max_det,
                        verbose=False,
                    )
                    if predict_device is not None:
                        predict_kwargs["device"] = predict_device
                    if self.use_onnx and self.onnx_imgsz:
                        predict_kwargs["imgsz"] = int(self.onnx_imgsz)
                    elif imgsz is not None:
                        predict_kwargs["imgsz"] = imgsz
                    chunk_results = self.model.predict(**predict_kwargs)
                    all_results.extend(chunk_results[:actual_chunk])
                return all_results

            source_input = source
        elif fixed_batch > 1:
            source_input = [source] * fixed_batch
        else:
            source_input = source

        predict_kwargs = dict(
            source=source_input,
            conf=raw_conf_floor,
            iou=1.0,  # Always use custom OBB IOU filtering after inference
            classes=target_classes,
            max_det=max_det,
            verbose=False,
        )
        if predict_device is not None:
            predict_kwargs["device"] = predict_device
        if self.use_onnx and self.onnx_imgsz:
            predict_kwargs["imgsz"] = int(self.onnx_imgsz)
        elif imgsz is not None:
            predict_kwargs["imgsz"] = imgsz
        results = self.model.predict(**predict_kwargs)
        if not isinstance(source, list) and fixed_batch > 1:
            results = results[:1]
        return results

    def _crops_to_tensor(self, source_crops, target_hw=None):
        """Convert a list of BGR ndarray crops to a float32 RGB tensor [N,3,H,W]."""
        import torch

        tensors = []
        for crop in source_crops:
            c = np.asarray(crop)
            if c.ndim == 2:
                c = np.stack([c, c, c], axis=-1)
            # Tracking frames are BGR; tiny models are trained on RGB crops.
            if c.ndim == 3 and c.shape[2] == 3:
                c = c[:, :, ::-1].copy()
            if target_hw is not None:
                import cv2

                w, h = int(target_hw[0]), int(target_hw[1])
                if c.shape[1] != w or c.shape[0] != h:
                    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    def _predict_headtail_results(self, source_crops):
        """Run head-tail classification in batches when possible."""
        if self.headtail_model is None or not source_crops:
            return []

        backend = getattr(self, "headtail_backend", "yolo")

        if backend == "tiny":
            import torch

            batch = self._crops_to_tensor(source_crops).to(self.device)
            with torch.inference_mode():
                logits = self.headtail_model(batch)
                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            return probs

        if backend == "classkit_tiny":
            import torch
            import torch.nn.functional as F

            target_hw = getattr(self, "headtail_input_size", None)
            batch = self._crops_to_tensor(source_crops, target_hw=target_hw).to(
                self.device
            )
            class_names = getattr(self, "headtail_class_names", None) or []
            with torch.inference_mode():
                logits = self.headtail_model(batch)  # [B, n_classes]
                softmax = F.softmax(logits, dim=1)  # [B, n_classes]
                top1_conf, top1_idx = softmax.max(dim=1)  # [B]
                top1_conf = top1_conf.detach().cpu().numpy()
                top1_idx = top1_idx.detach().cpu().numpy()

            classified = []
            for cls_idx, conf in zip(top1_idx, top1_conf):
                label = self._label_from_top1(int(cls_idx), class_names)
                direction = self._headtail_class_to_direction(
                    label, cls_idx=int(cls_idx), names=class_names
                )
                classified.append((direction, float(conf)))
            return classified

        # headtail_predict_device is None when the model was placed via .to(device).
        # Always fall back to self.device so the explicit device= argument is always
        # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
        predict_device = getattr(self, "headtail_predict_device", None) or self.device
        try:
            kwargs = dict(
                source=source_crops,
                conf=0.0,
                verbose=False,
            )
            if predict_device is not None:
                kwargs["device"] = predict_device
            return self.headtail_model.predict(**kwargs)
        except Exception:
            # Backend/model combinations can reject list sources.
            # Fall back to per-crop inference for compatibility.
            outputs = []
            for crop in source_crops:
                try:
                    kwargs = dict(
                        source=crop,
                        conf=0.0,
                        verbose=False,
                    )
                    if predict_device is not None:
                        kwargs["device"] = predict_device
                    one = self.headtail_model.predict(**kwargs)
                    outputs.append(one[0] if one else None)
                except Exception:
                    outputs.append(None)
            return outputs

    def _clip_crop_box(self, x1, y1, x2, y2, frame_w, frame_h):
        xi1 = int(np.floor(max(0.0, x1)))
        yi1 = int(np.floor(max(0.0, y1)))
        xi2 = int(np.ceil(min(float(frame_w), x2)))
        yi2 = int(np.ceil(min(float(frame_h), y2)))
        if xi2 <= xi1 or yi2 <= yi1:
            return None
        return xi1, yi1, xi2, yi2

    def _build_sequential_crop(self, frame, bbox_xyxy):
        """Create padded crop from stage-1 detection bbox."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        pad_ratio = float(self.params.get("YOLO_SEQ_CROP_PAD_RATIO", 0.15))
        min_crop_size = float(self.params.get("YOLO_SEQ_MIN_CROP_SIZE_PX", 64))
        enforce_square = bool(self.params.get("YOLO_SEQ_ENFORCE_SQUARE_CROP", True))

        crop_w = bw * (1.0 + 2.0 * max(0.0, pad_ratio))
        crop_h = bh * (1.0 + 2.0 * max(0.0, pad_ratio))
        if enforce_square:
            side = max(crop_w, crop_h)
            crop_w = side
            crop_h = side
        crop_w = max(min_crop_size, crop_w)
        crop_h = max(min_crop_size, crop_h)

        xx1 = cx - crop_w * 0.5
        yy1 = cy - crop_h * 0.5
        xx2 = cx + crop_w * 0.5
        yy2 = cy + crop_h * 0.5

        clipped = self._clip_crop_box(xx1, yy1, xx2, yy2, w, h)
        if clipped is None:
            return None, None
        xi1, yi1, xi2, yi2 = clipped
        crop = frame[yi1:yi2, xi1:xi2]
        if crop is None or crop.size == 0:
            return None, None
        return crop, (float(xi1), float(yi1))

    def _label_from_top1(self, cls_idx, names):
        if names is None:
            return ""
        if isinstance(names, dict):
            return str(names.get(int(cls_idx), "")).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= int(cls_idx) < len(names):
            return str(names[int(cls_idx)]).strip().lower()
        return ""

    def _ordered_headtail_class_names(self, names):
        if isinstance(names, dict):
            try:
                ordered_items = sorted(names.items(), key=lambda kv: int(kv[0]))
            except Exception:
                ordered_items = list(names.items())
            return [str(v) for _, v in ordered_items]
        if isinstance(names, (list, tuple)):
            return [str(v) for v in names]
        return []

    def _canonicalize_headtail_class_label(self, label: str):
        text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "left": "left",
            "head_left": "left",
            "right": "right",
            "head_right": "right",
            "up": "up",
            "head_up": "up",
            "down": "down",
            "head_down": "down",
            "unknown": "unknown",
            "head_unknown": "unknown",
        }
        return aliases.get(text)

    def _validate_headtail_class_names(self, class_names, *, source: str = "model"):
        ordered = self._ordered_headtail_class_names(class_names)
        if not ordered:
            raise ValueError(
                f"{source} is missing class names. Expected exactly left/right or up/down/left/right/unknown."
            )

        normalized = []
        for raw_name in ordered:
            token = self._canonicalize_headtail_class_label(raw_name)
            if token is None:
                raise ValueError(
                    f"Unsupported head-tail class label {raw_name!r} in {source}. "
                    "Expected exactly left/right or up/down/left/right/unknown."
                )
            normalized.append(token)

        normalized_set = frozenset(normalized)
        if len(normalized_set) != len(normalized):
            raise ValueError(
                f"Duplicate or aliased head-tail labels in {source}: {ordered}."
            )
        if normalized_set not in (
            _HEADTAIL_DIRECTIONAL_CLASS_SET,
            _HEADTAIL_FIVE_CLASS_SET,
        ):
            raise ValueError(
                f"Unsupported head-tail class schema in {source}: {ordered}. "
                "Expected exactly left/right or up/down/left/right/unknown."
            )
        return normalized

    def _headtail_class_to_direction(self, label: str, cls_idx=None, names=None):
        text = self._canonicalize_headtail_class_label(label)
        if text == "left":
            return "left"
        if text == "right":
            return "right"
        if text in {"up", "down", "unknown"}:
            return None

        # Fallback for unnamed binary classifiers.
        if names is not None:
            ordered = self._ordered_headtail_class_names(names)
            if len(ordered) == 2:
                if cls_idx is not None:
                    return "right" if int(cls_idx) == 1 else "left"
        return None

    def _canonicalize_obb_for_headtail(self, frame, corners):
        """
        Notebook-aligned affine canonicalization for head-tail inference.
        Returns (canonical_crop, major_axis_theta).
        """
        c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        e01 = float(np.linalg.norm(c[1] - c[0]))
        e12 = float(np.linalg.norm(c[2] - c[1]))
        if e01 < 1e-3 or e12 < 1e-3:
            return None, None

        if e01 >= e12:
            major = e01
            minor = e12
            major_vec = c[1] - c[0]
        else:
            major = e12
            minor = e01
            major_vec = c[2] - c[1]

        cx = float(np.mean(c[:, 0]))
        cy = float(np.mean(c[:, 1]))
        angle = float(math.atan2(float(major_vec[1]), float(major_vec[0])))

        margin = float(self.params.get("YOLO_HEADTAIL_CANONICAL_MARGIN", 1.3))
        out_w = int(self.params.get("YOLO_HEADTAIL_CANONICAL_WIDTH", 128))
        out_h = int(self.params.get("YOLO_HEADTAIL_CANONICAL_HEIGHT", 64))
        out_w = max(8, out_w)
        out_h = max(8, out_h)

        w_exp = float(major) * margin
        h_exp = float(minor) * margin
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        hw = w_exp * 0.5
        hh = h_exp * 0.5

        src_pts = np.array(
            [
                [
                    cx - hw * cos_a + hh * sin_a,
                    cy - hw * sin_a - hh * cos_a,
                ],  # top-left
                [
                    cx + hw * cos_a + hh * sin_a,
                    cy + hw * sin_a - hh * cos_a,
                ],  # top-right
                [
                    cx - hw * cos_a - hh * sin_a,
                    cy - hw * sin_a + hh * cos_a,
                ],  # bottom-left
            ],
            dtype=np.float32,
        )
        dst_pts = np.array([[0, 0], [out_w, 0], [0, out_h]], dtype=np.float32)
        M = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(
            frame,
            M,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped is None or warped.size == 0:
            return None, None
        return warped, angle

    def _compute_headtail_hints(self, frame, obb_corners_list):
        """Infer directed heading hints from optional head-tail classifier."""
        n = len(obb_corners_list)
        heading_hints = [float("nan")] * n
        directed_mask = [0] * n
        if self.headtail_model is None or n == 0:
            return heading_hints, directed_mask

        conf_threshold = float(self.params.get("YOLO_HEADTAIL_CONF_THRESHOLD", 0.50))
        canonical_crops = []
        canonical_meta = []
        for i, corners in enumerate(obb_corners_list):
            try:
                canonical, axis_theta = self._canonicalize_obb_for_headtail(
                    frame, corners
                )
                if canonical is None or axis_theta is None:
                    continue
                canonical_crops.append(canonical)
                canonical_meta.append((i, float(axis_theta)))
            except Exception:
                continue

        if not canonical_crops:
            return heading_hints, directed_mask

        cls_results = self._predict_headtail_results(canonical_crops)
        if cls_results is None or len(cls_results) == 0:
            return heading_hints, directed_mask

        backend = getattr(self, "headtail_backend", "yolo")
        if backend == "tiny":
            probs = np.asarray(cls_results, dtype=np.float32).reshape(-1)
            n_eval = min(len(canonical_meta), len(probs))
            for j in range(n_eval):
                i, axis_theta = canonical_meta[j]
                p_right = float(probs[j])
                conf = max(p_right, 1.0 - p_right)
                if conf < conf_threshold:
                    continue
                theta = axis_theta if p_right >= 0.5 else (axis_theta + np.pi)
                heading_hints[i] = float(theta % (2.0 * np.pi))
                directed_mask[i] = 1
        elif backend == "classkit_tiny":
            n_eval = min(len(canonical_meta), len(cls_results))
            for j in range(n_eval):
                i, axis_theta = canonical_meta[j]
                try:
                    direction, conf = cls_results[j]
                except Exception:
                    continue
                if direction not in {"left", "right"}:
                    continue
                if float(conf) < conf_threshold:
                    continue
                theta = axis_theta if direction == "right" else (axis_theta + np.pi)
                heading_hints[i] = float(theta % (2.0 * np.pi))
                directed_mask[i] = 1
        else:
            n_eval = min(len(canonical_meta), len(cls_results))
            for j in range(n_eval):
                i, axis_theta = canonical_meta[j]
                try:
                    result = cls_results[j]
                    if result is None:
                        continue
                    probs = getattr(result, "probs", None)
                    if probs is None:
                        continue
                    top1 = int(getattr(probs, "top1", -1))
                    top1_conf = float(getattr(probs, "top1conf", 0.0))
                    if top1 < 0 or top1_conf < conf_threshold:
                        continue
                    names = getattr(self, "headtail_class_names", None) or getattr(
                        result, "names", None
                    )
                    label = self._label_from_top1(top1, names)
                    direction = self._headtail_class_to_direction(
                        label, cls_idx=top1, names=names
                    )
                    if direction is None:
                        continue
                    theta = axis_theta if direction == "right" else (axis_theta + np.pi)
                    heading_hints[i] = float(theta % (2.0 * np.pi))
                    directed_mask[i] = 1
                except Exception:
                    continue

        return heading_hints, directed_mask

    def _run_direct_raw_detection(self, frame, target_classes, raw_conf_floor, max_det):
        results = self._predict_obb_results(
            frame, target_classes, raw_conf_floor, max_det
        )
        if len(results) == 0:
            return [], [], [], [], [], None
        result0 = results[0]
        if result0.obb is None or len(result0.obb) == 0:
            return [], [], [], [], [], result0
        raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
            self._extract_raw_detections(result0.obb)
        )
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            result0,
        )

    def _run_sequential_raw_detection(
        self, frame, target_classes, raw_conf_floor, max_det
    ):
        if self.detect_model is None:
            return [], [], [], [], [], None
        # detect_predict_device is None when the model was placed via .to(device).
        # Always fall back to self.device so the explicit device= argument is always
        # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
        detect_predict_device = (
            getattr(self, "detect_predict_device", None) or self.device
        )
        detect_target_classes = self.params.get(
            "YOLO_DETECT_TARGET_CLASSES", target_classes
        )
        try:
            # YOLO_SEQ_DETECT_CONF_THRESHOLD lets users tune stage-1 sensitivity
            # independently of the stage-2 OBB confidence threshold.
            # Default falls back to raw_conf_floor (the global minimum floor).
            seq_detect_conf = float(
                self.params.get("YOLO_SEQ_DETECT_CONF_THRESHOLD", raw_conf_floor)
            )
            seq_detect_conf = max(1e-4, seq_detect_conf)
            detect_kwargs = dict(
                source=frame,
                conf=seq_detect_conf,
                iou=1.0,
                classes=detect_target_classes,
                max_det=max_det,
                verbose=False,
            )
            if detect_predict_device is not None:
                detect_kwargs["device"] = detect_predict_device
            # Allow forcing a smaller imgsz for the detect stage on devices (e.g. MPS)
            # where inference time scales strongly with input resolution.
            # YOLO_SEQ_DETECT_IMGSZ=0 → use the model's native resolution (default).
            seq_detect_imgsz = int(self.params.get("YOLO_SEQ_DETECT_IMGSZ", 0))
            if seq_detect_imgsz > 0:
                detect_kwargs["imgsz"] = seq_detect_imgsz
            detect_results = self.detect_model.predict(**detect_kwargs)
        except Exception as exc:
            logger.error("YOLO sequential detect stage failed: %s", exc)
            return [], [], [], [], [], None

        if not detect_results:
            return [], [], [], [], [], None
        det0 = detect_results[0]
        boxes = getattr(det0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], [], [], [], [], det0

        xyxy = np.ascontiguousarray(boxes.xyxy.cpu().numpy(), dtype=np.float32)
        det_conf = np.ascontiguousarray(boxes.conf.cpu().numpy(), dtype=np.float32)
        order = np.argsort(det_conf)[::-1]
        if len(order) > max_det:
            order = order[:max_det]

        merged_meas = []
        merged_sizes = []
        merged_shapes = []
        merged_conf = []
        merged_corners = []
        crops = []
        crop_offsets = []

        crop_original_sizes = []  # (w, h) of each original crop before any resize
        for idx in order:
            crop, offset = self._build_sequential_crop(frame, xyxy[idx])
            if crop is None or offset is None:
                continue
            crop_original_sizes.append((crop.shape[1], crop.shape[0]))  # (w, h)
            crops.append(crop)
            crop_offsets.append(offset)

        if not crops:
            return [], [], [], [], [], det0

        # On MPS (and any device where batch-inference overhead is latency-bound rather
        # than throughput-bound) variable-size crops cause Metal shader retracing per
        # unique shape.  Pre-resize every crop to the same square pixel size so that
        # (a) ultralytics sees a uniform-shape batch, and (b) imgsz is pinned to one
        # value, preventing repeated graph recompilation.
        # YOLO_SEQ_STAGE2_IMGSZ=0 disables this (pass raw crops as before).
        # Default 160 = native training imgsz of the crop OBB model; DO NOT use 256+
        # as that puts the model out-of-distribution and kills detection confidence.
        stage2_imgsz = int(self.params.get("YOLO_SEQ_STAGE2_IMGSZ", 160))
        if stage2_imgsz > 0:
            # Some unit-test cv2 stubs only expose a subset of interpolation enums.
            resize_interp = getattr(
                cv2,
                "INTER_LINEAR",
                getattr(cv2, "INTER_AREA", 1),
            )
            resized_crops = []
            for crop in crops:
                h_c, w_c = crop.shape[:2]
                if h_c != stage2_imgsz or w_c != stage2_imgsz:
                    resized_crops.append(
                        cv2.resize(
                            crop,
                            (stage2_imgsz, stage2_imgsz),
                            interpolation=resize_interp,
                        )
                    )
                else:
                    resized_crops.append(crop)
            crops_for_stage2 = resized_crops
            predict_imgsz = stage2_imgsz
        else:
            crops_for_stage2 = crops
            predict_imgsz = None

        # On MPS, Metal recompiles shaders for each unique batch size.
        # Padding the crop list to the next power-of-2 count ensures that
        # only O(log N) distinct batch sizes are ever compiled (1,2,4,8,…),
        # which is the primary reason sequential is slow on Apple Silicon.
        # YOLO_SEQ_STAGE2_POW2_PAD=0 disables this.
        n_real_crops = len(crops_for_stage2)
        pow2_pad = self.params.get("YOLO_SEQ_STAGE2_POW2_PAD", 0)
        pad_img = None
        if pow2_pad and predict_imgsz is not None and n_real_crops > 0:
            p2 = 1
            while p2 < n_real_crops:
                p2 *= 2
            if p2 > n_real_crops:
                pad_img = np.zeros((predict_imgsz, predict_imgsz, 3), dtype=np.uint8)
                crops_for_stage2 = list(crops_for_stage2) + [pad_img] * (
                    p2 - n_real_crops
                )

        try:
            stage2_results = self._predict_obb_results(
                crops_for_stage2,
                target_classes=target_classes,
                raw_conf_floor=raw_conf_floor,
                max_det=max_det,
                imgsz=predict_imgsz,
            )
        except TypeError as exc:
            # Backward-compat for monkeypatched test doubles without imgsz kwarg.
            if "imgsz" not in str(exc):
                raise
            stage2_results = self._predict_obb_results(
                crops_for_stage2,
                target_classes=target_classes,
                raw_conf_floor=raw_conf_floor,
                max_det=max_det,
            )
        n_stage2 = min(len(stage2_results), len(crop_offsets), n_real_crops)
        for i in range(n_stage2):
            result = stage2_results[i]
            x0, y0 = crop_offsets[i]
            if result is None or result.obb is None or len(result.obb) == 0:
                continue
            (
                crop_meas,
                crop_sizes,
                crop_shapes,
                crop_conf,
                crop_corners,
            ) = self._extract_raw_detections(result.obb)
            if not crop_meas:
                continue
            # If crops were pre-resized, scale detected coordinates back to the
            # original crop pixel space before applying the global frame offset.
            if predict_imgsz is not None and i < len(crop_original_sizes):
                orig_w, orig_h = crop_original_sizes[i]
                sx = orig_w / float(predict_imgsz)
                sy = orig_h / float(predict_imgsz)
            else:
                sx, sy = 1.0, 1.0
            for j in range(len(crop_meas)):
                m = np.asarray(crop_meas[j], dtype=np.float32).copy()
                m[0] = m[0] * np.float32(sx) + np.float32(x0)
                m[1] = m[1] * np.float32(sy) + np.float32(y0)
                c = np.asarray(crop_corners[j], dtype=np.float32).copy()
                c[:, 0] = c[:, 0] * np.float32(sx) + np.float32(x0)
                c[:, 1] = c[:, 1] * np.float32(sy) + np.float32(y0)
                merged_meas.append(m)
                # Scale area back to original-frame pixel space (area scales by sx*sy)
                merged_sizes.append(float(crop_sizes[j]) * sx * sy)
                merged_shapes.append(tuple(crop_shapes[j]))
                merged_conf.append(float(crop_conf[j]))
                merged_corners.append(c)

        if not merged_meas:
            return [], [], [], [], [], det0

        conf_arr = np.asarray(merged_conf, dtype=np.float32)
        order_final = np.argsort(conf_arr)[::-1]
        if len(order_final) > max_det:
            order_final = order_final[:max_det]

        raw_meas = [merged_meas[i] for i in order_final]
        raw_sizes = [merged_sizes[i] for i in order_final]
        raw_shapes = [merged_shapes[i] for i in order_final]
        raw_confidences = [merged_conf[i] for i in order_final]
        raw_obb_corners = [merged_corners[i] for i in order_final]
        return (
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            det0,
        )

    def _load_model(self):
        """Load the YOLO OBB model with optional TensorRT optimization."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error(
                "ultralytics package not found. Please install it: pip install ultralytics"
            )
            raise ImportError(
                "ultralytics package required for YOLO detection. Install with: pip install ultralytics"
            )
        self._configure_ultralytics_logging()

        model_path_str = str(
            self.params.get(
                "YOLO_MODEL_PATH",
                getattr(self, "active_obb_model_path", "yolo26s-obb.pt"),
            )
            or "yolo26s-obb.pt"
        )
        enable_tensorrt = self.params.get("ENABLE_TENSORRT", False)
        enable_onnx_runtime = self.params.get("ENABLE_ONNX_RUNTIME", False)
        model_path = Path(model_path_str).expanduser().resolve()
        local_model_file = model_path.exists() and model_path.is_file()

        # Check if TensorRT is requested and available
        from multi_tracker.utils.gpu_utils import (
            ONNXRUNTIME_AVAILABLE,
            TENSORRT_AVAILABLE,
        )

        if enable_onnx_runtime and ONNXRUNTIME_AVAILABLE and local_model_file:
            self._try_load_onnx_model(model_path_str)
            if self.use_onnx:
                self.obb_predict_device = self.device
                return

        if (
            enable_tensorrt
            and TENSORRT_AVAILABLE
            and self.device.startswith("cuda")
            and local_model_file
        ):
            self._try_load_tensorrt_model(model_path_str)
            if self.use_tensorrt:
                self.obb_predict_device = self.device
                return
            else:
                logger.info("Falling back to standard PyTorch inference")

        # For pretrained model names (yolo26s-obb.pt, etc.), pass directly to YOLO
        # These will be auto-downloaded by ultralytics
        if model_path_str.startswith(("yolov8", "yolov11", "yolo26")):
            try:
                self.model = YOLO(model_path_str)
                # Move model to the appropriate device
                self.model.to(self.device)
                self.obb_predict_device = None
                logger.info(
                    f"YOLO OBB model loaded successfully: {model_path_str} on device: {self.device}"
                )
                return
            except Exception as e:
                logger.error(f"Failed to load YOLO model '{model_path_str}': {e}")
                raise

        # Check if the file exists
        if not model_path.exists():
            logger.error(
                f"YOLO model file not found: {model_path}\n"
                f"Original path: {model_path_str}\n"
                f"Working directory: {Path.cwd()}"
            )
            raise FileNotFoundError(
                f"YOLO model file not found: {model_path}. "
                f"Please check the path and ensure the file exists."
            )

        try:
            # Use the resolved absolute path as a string
            self.model = YOLO(str(model_path))
            # Move model to the appropriate device
            self.model.to(self.device)
            self.obb_predict_device = None
            logger.info(
                f"YOLO OBB model loaded successfully from {model_path} on device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

    def _compute_obb_iou_batch(self, corners1, corners_list, indices):
        """
        Batch compute IOU between one OBB and multiple OBBs efficiently.

        Args:
            corners1: (4, 2) array for reference OBB
            corners_list: List of all corner arrays
            indices: List of indices to compute IOU for

        Returns:
            Array of IOU values
        """
        if len(indices) == 0:
            return np.array([])

        p1 = cv2.convexHull(np.asarray(corners1, dtype=np.float32)).reshape(-1, 2)
        area1 = float(abs(cv2.contourArea(p1)))
        if area1 <= 1e-9:
            return np.zeros(len(indices), dtype=np.float32)

        ious = np.zeros(len(indices), dtype=np.float32)
        for i, idx in enumerate(indices):
            p2 = cv2.convexHull(
                np.asarray(corners_list[idx], dtype=np.float32)
            ).reshape(-1, 2)
            area2 = float(abs(cv2.contourArea(p2)))
            if area2 <= 1e-9:
                continue
            try:
                inter_area, _ = cv2.intersectConvexConvex(p1, p2)
                inter_area = float(max(0.0, inter_area))
            except Exception:
                inter_area = 0.0
            union = area1 + area2 - inter_area
            if union > 1e-9:
                ious[i] = inter_area / union

        return ious

    def _compute_obb_iou(self, corners1, corners2):
        """
        Compute IOU between two oriented bounding boxes efficiently.

        Args:
            corners1: (4, 2) array of corner points for first OBB
            corners2: (4, 2) array of corner points for second OBB

        Returns:
            IOU value (0-1)
        """
        p1 = cv2.convexHull(np.asarray(corners1, dtype=np.float32)).reshape(-1, 2)
        p2 = cv2.convexHull(np.asarray(corners2, dtype=np.float32)).reshape(-1, 2)
        area1 = float(abs(cv2.contourArea(p1)))
        area2 = float(abs(cv2.contourArea(p2)))
        if area1 <= 1e-9 or area2 <= 1e-9:
            return 0.0
        try:
            inter_area, _ = cv2.intersectConvexConvex(p1, p2)
            inter_area = float(max(0.0, inter_area))
        except Exception:
            inter_area = 0.0
        union = area1 + area2 - inter_area
        if union <= 1e-9:
            return 0.0
        return inter_area / union

    def _filter_overlapping_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners_list,
        iou_threshold,
        detection_ids=None,
        heading_hints=None,
        directed_mask=None,
    ):
        """
        Filter overlapping detections using spatially-optimized IOU-based NMS for OBB.
        Keeps highest confidence detections and removes overlapping ones.
        Optimized for high detection counts (25-200 animals).

        Args:
            meas: List of measurements [cx, cy, angle]
            sizes: List of detection areas
            shapes: List of (area, aspect_ratio) tuples
            confidences: List of confidence scores (0-1)
            obb_corners_list: List of corner arrays (4, 2) for each detection
            iou_threshold: IOU threshold for considering detections as overlapping

        Returns:
            Filtered versions of all input lists. If detection_ids is provided,
            it is filtered identically and returned as the final element.
        """
        if len(meas) <= 1:
            if detection_ids is None and heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list
            if detection_ids is not None and heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list, detection_ids
            if detection_ids is None:
                return (
                    meas,
                    sizes,
                    shapes,
                    confidences,
                    obb_corners_list,
                    heading_hints,
                    directed_mask,
                )
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                detection_ids,
                heading_hints,
                directed_mask,
            )

        n_detections = len(meas)

        # Convert inputs to numpy arrays for vectorized operations
        confidences_arr = np.array(confidences)

        # Pre-compute axis-aligned bounding boxes (fully vectorized)
        corners_array = np.array(obb_corners_list)  # (n, 4, 2)
        bbox_mins = corners_array.min(axis=1)  # (n, 2)
        bbox_maxs = corners_array.max(axis=1)  # (n, 2)

        # Sort indices by confidence (highest first)
        sorted_indices = np.argsort(confidences_arr)[::-1]

        keep_mask = np.zeros(n_detections, dtype=bool)

        idx = 0
        while idx < len(sorted_indices):
            # Keep the detection with highest remaining confidence
            current_idx = sorted_indices[idx]
            keep_mask[current_idx] = True

            if idx == len(sorted_indices) - 1:
                break

            # Get current box bounding box
            curr_min = bbox_mins[current_idx]
            curr_max = bbox_maxs[current_idx]

            # Get remaining candidates
            remaining_indices = sorted_indices[idx + 1 :]
            rem_mins = bbox_mins[remaining_indices]
            rem_maxs = bbox_maxs[remaining_indices]

            # Vectorized axis-aligned bbox overlap check
            inter_mins = np.maximum(curr_min, rem_mins)
            inter_maxs = np.minimum(curr_max, rem_maxs)

            # Check if boxes overlap (width and height both positive)
            inter_wh = inter_maxs - inter_mins
            overlaps = (inter_wh[:, 0] > 0) & (inter_wh[:, 1] > 0)

            # For non-overlapping boxes, skip IOU calculation
            keep_remaining = ~overlaps

            # For overlapping boxes, compute IOU
            if overlaps.any():
                # Initial state: keep overlapping candidates unless precise OBB IOU says suppress.
                overlapping_local = np.where(overlaps)[0]
                keep_remaining[overlapping_local] = True

                # Run precise polygon IOU for all AABB-overlapping candidates.
                # This matches direct-mode manual OBB suppression behavior.
                precise_check_global = remaining_indices[overlapping_local]
                precise_ious = self._compute_obb_iou_batch(
                    obb_corners_list[current_idx],
                    obb_corners_list,
                    precise_check_global,
                )

                suppress = precise_ious >= iou_threshold
                keep_remaining[overlapping_local] = ~suppress

            # Update sorted indices to keep only non-suppressed detections
            sorted_indices = np.concatenate(
                [sorted_indices[: idx + 1], remaining_indices[keep_remaining]]
            )

            idx += 1

        # Use numpy indexing for final filtering (much faster than list comprehension)
        keep_indices = np.where(keep_mask)[0]

        # Convert back to lists with proper indexing
        meas = [meas[i] for i in keep_indices]
        sizes = [sizes[i] for i in keep_indices]
        shapes = [shapes[i] for i in keep_indices]
        confidences = [confidences[i] for i in keep_indices]
        obb_corners_list = [obb_corners_list[i] for i in keep_indices]
        if heading_hints is not None:
            heading_hints = [heading_hints[i] for i in keep_indices]
            if directed_mask is None:
                directed_mask = [0] * len(heading_hints)
            else:
                directed_mask = [directed_mask[i] for i in keep_indices]
        if detection_ids is not None:
            detection_ids = [detection_ids[i] for i in keep_indices]
            if heading_hints is None:
                return meas, sizes, shapes, confidences, obb_corners_list, detection_ids
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                detection_ids,
                heading_hints,
                directed_mask,
            )
        if heading_hints is not None:
            return (
                meas,
                sizes,
                shapes,
                confidences,
                obb_corners_list,
                heading_hints,
                directed_mask,
            )
        return meas, sizes, shapes, confidences, obb_corners_list

    def _raw_detection_cap(self) -> int:
        """Cap raw detections to 2x MAX_TARGETS to bound cache size and filtering cost."""
        max_targets = max(1, int(self.params.get("MAX_TARGETS", 8)))
        return max_targets * 2

    def _extract_raw_detections(self, obb_data):
        """Extract raw OBB detections, sorted by confidence and capped by policy."""
        if obb_data is None or len(obb_data) == 0:
            return [], [], [], [], []

        xywhr = np.ascontiguousarray(obb_data.xywhr.cpu().numpy(), dtype=np.float32)
        conf_scores = np.ascontiguousarray(
            obb_data.conf.cpu().numpy(), dtype=np.float32
        )
        corners = np.ascontiguousarray(
            obb_data.xyxyxyxy.cpu().numpy(), dtype=np.float32
        )

        if xywhr.size == 0 or conf_scores.size == 0:
            return [], [], [], [], []

        cx = xywhr[:, 0]
        cy = xywhr[:, 1]
        w = xywhr[:, 2]
        h = xywhr[:, 3]
        angle_raw = xywhr[:, 4]
        # Runtime parity guard:
        # Some exported backends may report theta in degrees instead of radians.
        if np.nanmax(np.abs(angle_raw)) > (2.0 * np.pi + 1e-3):
            angle_rad = np.deg2rad(angle_raw)
        else:
            angle_rad = angle_raw

        angle_deg = np.rad2deg(angle_rad) % 180.0
        swap_mask = w < h
        major = np.where(swap_mask, h, w)
        minor = np.where(swap_mask, w, h)
        angle_deg = np.where(swap_mask, (angle_deg + 90.0) % 180.0, angle_deg)
        angle_rad_fixed = np.deg2rad(angle_deg).astype(np.float32)

        sizes = (major * minor).astype(np.float32)
        ellipse_area = (np.pi * (major / 2.0) * (minor / 2.0)).astype(np.float32)
        aspect_ratio = np.divide(
            major,
            minor,
            out=np.zeros_like(major, dtype=np.float32),
            where=minor > 0,
        )

        meas_arr = np.column_stack((cx, cy, angle_rad_fixed)).astype(np.float32)
        shapes_arr = np.column_stack((ellipse_area, aspect_ratio)).astype(np.float32)

        # Build OBB corners from xywhr directly for stable geometry across runtimes
        # (ONNX/TensorRT can disagree on provided xyxyxyxy corner ordering/decoding).
        half_w = major / 2.0
        half_h = minor / 2.0
        x_offsets = np.stack((-half_w, half_w, half_w, -half_w), axis=1)
        y_offsets = np.stack((-half_h, -half_h, half_h, half_h), axis=1)
        cos_t = np.cos(angle_rad_fixed)
        sin_t = np.sin(angle_rad_fixed)
        x_coords = cx[:, None] + x_offsets * cos_t[:, None] - y_offsets * sin_t[:, None]
        y_coords = cy[:, None] + x_offsets * sin_t[:, None] + y_offsets * cos_t[:, None]
        corners = np.stack((x_coords, y_coords), axis=2).astype(np.float32, copy=False)

        cap = self._raw_detection_cap()
        order = np.argsort(conf_scores)[::-1]
        if len(order) > cap:
            order = order[:cap]

        meas_arr = np.ascontiguousarray(meas_arr[order], dtype=np.float32)
        sizes = np.ascontiguousarray(sizes[order], dtype=np.float32)
        shapes_arr = np.ascontiguousarray(shapes_arr[order], dtype=np.float32)
        conf_scores = np.ascontiguousarray(conf_scores[order], dtype=np.float32)
        corners = np.ascontiguousarray(corners[order], dtype=np.float32)

        meas = [meas_arr[i] for i in range(len(meas_arr))]
        sizes_list = sizes.tolist()
        shapes = [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
        confidences = conf_scores.tolist()
        obb_corners_list = [corners[i] for i in range(len(corners))]

        return meas, sizes_list, shapes, confidences, obb_corners_list

    def filter_raw_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners_list,
        roi_mask=None,
        detection_ids=None,
        heading_hints=None,
        directed_mask=None,
    ):
        """
        Apply vectorized confidence/size/ROI filtering, then custom OBB IOU suppression.
        This is shared by live detection and cached-raw detection paths.
        """
        if not meas:
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], []

        conf_threshold = float(self.params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
        iou_threshold = float(self.params.get("YOLO_IOU_THRESHOLD", 0.7))
        max_targets = max(1, int(self.params.get("MAX_TARGETS", 8)))

        meas_arr = np.ascontiguousarray(np.asarray(meas, dtype=np.float32))
        sizes_arr = np.ascontiguousarray(np.asarray(sizes, dtype=np.float32))
        shapes_arr = np.ascontiguousarray(np.asarray(shapes, dtype=np.float32))
        conf_arr = np.ascontiguousarray(np.asarray(confidences, dtype=np.float32))

        if detection_ids is None:
            ids_arr = np.arange(len(meas_arr), dtype=np.int64)
        else:
            ids_arr = np.ascontiguousarray(
                np.asarray(_normalize_detection_ids(detection_ids), dtype=np.int64)
            )

        n = min(
            len(meas_arr), len(sizes_arr), len(shapes_arr), len(conf_arr), len(ids_arr)
        )
        if obb_corners_list:
            n = min(n, len(obb_corners_list))
        if heading_hints is not None:
            n = min(n, len(heading_hints))
            if directed_mask is not None:
                n = min(n, len(directed_mask))
        if n == 0:
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], []

        meas_arr = meas_arr[:n]
        sizes_arr = sizes_arr[:n]
        shapes_arr = shapes_arr[:n]
        conf_arr = conf_arr[:n]
        ids_arr = ids_arr[:n]

        if obb_corners_list:
            obb_arr = np.ascontiguousarray(
                np.asarray(obb_corners_list, dtype=np.float32)
            )
            obb_arr = obb_arr[:n]
        else:
            obb_arr = np.empty((n, 4, 2), dtype=np.float32)

        if heading_hints is not None:
            heading_arr = np.ascontiguousarray(
                np.asarray(heading_hints, dtype=np.float32)
            )[:n]
            if directed_mask is None:
                directed_arr = np.zeros(n, dtype=np.uint8)
            else:
                directed_arr = np.ascontiguousarray(
                    np.asarray(directed_mask, dtype=np.uint8)
                )[:n]
        else:
            heading_arr = None
            directed_arr = None

        keep_mask = conf_arr >= conf_threshold

        if self.params.get("ENABLE_SIZE_FILTERING", False):
            min_size = float(self.params.get("MIN_OBJECT_SIZE", 0))
            max_size = float(self.params.get("MAX_OBJECT_SIZE", float("inf")))
            keep_mask &= (sizes_arr >= min_size) & (sizes_arr <= max_size)

        if roi_mask is not None and len(meas_arr) > 0:
            h, w = roi_mask.shape[:2]
            cx = meas_arr[:, 0].astype(np.int32)
            cy = meas_arr[:, 1].astype(np.int32)
            in_bounds = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
            cx_safe = np.clip(cx, 0, max(0, w - 1))
            cy_safe = np.clip(cy, 0, max(0, h - 1))
            in_roi = roi_mask[cy_safe, cx_safe] > 0
            keep_mask &= in_bounds & in_roi

        if not np.any(keep_mask):
            if heading_hints is None:
                return [], [], [], [], [], []
            return [], [], [], [], [], [], [], []

        meas_arr = meas_arr[keep_mask]
        sizes_arr = sizes_arr[keep_mask]
        shapes_arr = shapes_arr[keep_mask]
        conf_arr = conf_arr[keep_mask]
        ids_arr = ids_arr[keep_mask]
        obb_arr = obb_arr[keep_mask]
        if heading_arr is not None:
            heading_arr = heading_arr[keep_mask]
            directed_arr = directed_arr[keep_mask]

        meas_list = [meas_arr[i] for i in range(len(meas_arr))]
        sizes_list = sizes_arr.tolist()
        shapes_list = [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
        conf_list = conf_arr.tolist()
        ids_list = [int(v) for v in ids_arr.tolist()]
        obb_list = [obb_arr[i] for i in range(len(obb_arr))]
        heading_list = heading_arr.tolist() if heading_arr is not None else None
        directed_list = directed_arr.tolist() if directed_arr is not None else None

        if len(meas_list) > 1:
            if heading_list is None:
                (
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    ids_list,
                ) = self._filter_overlapping_detections(
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    iou_threshold,
                    detection_ids=ids_list,
                )
            else:
                (
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    ids_list,
                    heading_list,
                    directed_list,
                ) = self._filter_overlapping_detections(
                    meas_list,
                    sizes_list,
                    shapes_list,
                    conf_list,
                    obb_list,
                    iou_threshold,
                    detection_ids=ids_list,
                    heading_hints=heading_list,
                    directed_mask=directed_list,
                )

        if len(meas_list) > max_targets:
            idxs = np.argsort(sizes_list)[::-1][:max_targets]
            meas_list = [meas_list[i] for i in idxs]
            sizes_list = [sizes_list[i] for i in idxs]
            shapes_list = [shapes_list[i] for i in idxs]
            conf_list = [conf_list[i] for i in idxs]
            obb_list = [obb_list[i] for i in idxs]
            ids_list = [ids_list[i] for i in idxs]
            if heading_list is not None:
                heading_list = [heading_list[i] for i in idxs]
                directed_list = [directed_list[i] for i in idxs]

        if heading_list is None:
            return meas_list, sizes_list, shapes_list, conf_list, obb_list, ids_list
        return (
            meas_list,
            sizes_list,
            shapes_list,
            conf_list,
            obb_list,
            ids_list,
            heading_list,
            directed_list,
        )

    def detect_objects(
        self: object, frame: object, frame_count: object, return_raw: bool = False
    ) -> object:
        """
        Detects objects in a frame using YOLO OBB.

        Args:
            frame: Input frame (grayscale or BGR)
            frame_count: Current frame number for logging

        Returns:
            Default mode:
                meas, sizes, shapes, yolo_results, confidences
            If return_raw=True:
                raw_meas, raw_sizes, raw_shapes, yolo_results, raw_confidences,
                raw_obb_corners, raw_heading_hints, raw_directed_mask
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            if return_raw:
                return [], [], [], None, [], [], [], []
            return [], [], [], None, []

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)  # None means all classes
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        try:
            if self.obb_mode == "sequential":
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    yolo_results,
                ) = self._run_sequential_raw_detection(
                    frame,
                    target_classes=target_classes,
                    raw_conf_floor=raw_conf_floor,
                    max_det=max_det,
                )
            else:
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    yolo_results,
                ) = self._run_direct_raw_detection(
                    frame,
                    target_classes=target_classes,
                    raw_conf_floor=raw_conf_floor,
                    max_det=max_det,
                )
        except Exception as e:
            logger.error(f"YOLO inference failed on frame {frame_count}: {e}")
            if return_raw:
                return [], [], [], None, [], [], [], []
            return [], [], [], None, []

        if not raw_meas:
            raw_heading_hints, raw_directed_mask = [], []
            if return_raw:
                return (
                    [],
                    [],
                    [],
                    yolo_results,
                    [],
                    [],
                    raw_heading_hints,
                    raw_directed_mask,
                )
            return [], [], [], yolo_results, []

        raw_heading_hints, raw_directed_mask = self._compute_headtail_hints(
            frame, raw_obb_corners
        )

        if return_raw:
            if yolo_results is not None:
                pass
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                yolo_results,
                raw_confidences,
                raw_obb_corners,
                raw_heading_hints,
                raw_directed_mask,
            )

        (
            meas,
            sizes,
            shapes,
            confidences,
            obb_corners_list,
            _,
            filtered_heading_hints,
            filtered_directed_mask,
        ) = self.filter_raw_detections(
            raw_meas,
            raw_sizes,
            raw_shapes,
            raw_confidences,
            raw_obb_corners,
            roi_mask=None,
            detection_ids=None,
            heading_hints=raw_heading_hints,
            directed_mask=raw_directed_mask,
        )

        if meas:
            logger.debug(f"Frame {frame_count}: YOLO detected {len(meas)} objects")

        # Return filtered OBB corners alongside other data
        # Store in results object for access by individual dataset generator
        if yolo_results is not None:
            pass

        return meas, sizes, shapes, yolo_results, confidences

    def detect_objects_batched(
        self: object,
        frames: object,
        start_frame_idx: object,
        progress_callback: object = None,
        return_raw: bool = False,
    ) -> object:
        """
        Detect objects in a batch of frames using YOLO OBB.

        Args:
            frames: List of frames (numpy arrays)
            start_frame_idx: Starting frame index for this batch
            progress_callback: Optional callback(current, total, message) for progress updates

        Returns:
            List of tuples per frame:
              - return_raw=False: (meas, sizes, shapes, confidences, obb_corners)
              - return_raw=True:  (
                    raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners,
                    raw_heading_hints, raw_directed_mask
                )
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return [([], [], [], [], []) for _ in frames]

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        # Sequential mode and head-tail orientation require per-frame processing
        # because each frame has variable crop counts and optional classification.
        if self.obb_mode == "sequential" or self.headtail_model is not None:
            batch_detections = []
            for idx, frame in enumerate(frames):
                frame_count = int(start_frame_idx) + int(idx)
                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    _results,
                    raw_confidences,
                    raw_obb_corners,
                    raw_heading_hints,
                    raw_directed_mask,
                ) = self.detect_objects(frame, frame_count, return_raw=True)

                if return_raw:
                    batch_detections.append(
                        (
                            raw_meas,
                            raw_sizes,
                            raw_shapes,
                            raw_confidences,
                            raw_obb_corners,
                            raw_heading_hints,
                            raw_directed_mask,
                        )
                    )
                else:
                    (
                        meas,
                        sizes,
                        shapes,
                        confidences,
                        obb_corners_list,
                        _,
                        _heading_hints,
                        _directed_mask,
                    ) = self.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confidences,
                        raw_obb_corners,
                        roi_mask=None,
                        detection_ids=None,
                        heading_hints=raw_heading_hints,
                        directed_mask=raw_directed_mask,
                    )
                    batch_detections.append(
                        (meas, sizes, shapes, confidences, obb_corners_list)
                    )

                if progress_callback and (idx + 1) % 10 == 0:
                    progress_callback(
                        idx + 1,
                        len(frames),
                        f"Processing batch frame {idx + 1}/{len(frames)}",
                    )

            return batch_detections

        # Handle fixed-batch runtimes (TensorRT and static-batch ONNX).
        # These require exact batch size, so we:
        # 1. Chunk larger batches into runtime batch-sized pieces
        # 2. Pad the final chunk if smaller than runtime batch size
        actual_frame_count = len(frames)
        fixed_batch_size = None
        fixed_backend = None
        if self.use_tensorrt and hasattr(self, "tensorrt_batch_size"):
            fixed_batch_size = max(1, int(self.tensorrt_batch_size))
            fixed_backend = "TensorRT"
        elif self.use_onnx:
            fixed_batch_size = max(1, int(getattr(self, "onnx_batch_size", 1)))
            fixed_backend = "ONNX"

        if fixed_batch_size is not None:
            all_results = []
            # obb_predict_device is None when the model was placed via .to(device).
            # Always fall back to self.device so the explicit device= argument is always
            # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
            obb_predict_device = (
                getattr(self, "obb_predict_device", None) or self.device
            )

            # Process in fixed-size chunks
            for chunk_start in range(0, actual_frame_count, fixed_batch_size):
                chunk_end = min(chunk_start + fixed_batch_size, actual_frame_count)
                chunk_frames = frames[chunk_start:chunk_end]
                chunk_size = len(chunk_frames)

                # Pad chunk if smaller than fixed batch size
                if chunk_size < fixed_batch_size:
                    padding_needed = fixed_batch_size - chunk_size
                    dummy_frame = chunk_frames[0]
                    chunk_frames = list(chunk_frames) + [dummy_frame] * padding_needed
                    logger.debug(
                        f"Padded final chunk from {chunk_size} to {fixed_batch_size} for {fixed_backend}"
                    )

                # Run inference on this chunk
                # Use custom polygon-based IOU filtering OR YOLO's built-in NMS based on user preference
                try:
                    predict_kwargs = dict(
                        source=chunk_frames,
                        conf=raw_conf_floor,
                        iou=1.0,  # Always use custom OBB IOU filtering after inference
                        classes=target_classes,
                        max_det=max_det,
                        verbose=False,
                    )
                    if obb_predict_device is not None:
                        predict_kwargs["device"] = obb_predict_device
                    if self.use_onnx and self.onnx_imgsz:
                        predict_kwargs["imgsz"] = int(self.onnx_imgsz)
                    chunk_results = self.model.predict(**predict_kwargs)
                    # Only keep results for actual frames (not padding)
                    all_results.extend(chunk_results[:chunk_size])
                except Exception as e:
                    logger.error(f"YOLO batched inference failed on chunk: {e}")
                    # Return empty results for this chunk
                    all_results.extend([None] * chunk_size)

            results_batch = all_results
        else:
            # Standard PyTorch inference - no chunking needed
            # Use custom polygon-based IOU filtering OR YOLO's built-in NMS based on user preference
            try:
                # obb_predict_device is None when the model was placed via .to(device).
                # Always fall back to self.device so the explicit device= argument is always
                # passed to predict(), preventing Ultralytics from auto-selecting a wrong device.
                obb_predict_device = (
                    getattr(self, "obb_predict_device", None) or self.device
                )
                predict_kwargs = dict(
                    source=frames,
                    conf=raw_conf_floor,
                    iou=1.0,  # Always use custom OBB IOU filtering after inference
                    classes=target_classes,
                    max_det=max_det,
                    verbose=False,
                )
                if obb_predict_device is not None:
                    predict_kwargs["device"] = obb_predict_device
                if self.use_onnx and self.onnx_imgsz:
                    predict_kwargs["imgsz"] = int(self.onnx_imgsz)
                results_batch = self.model.predict(**predict_kwargs)
            except Exception as e:
                logger.error(f"YOLO batched inference failed: {e}")
                if not self.use_onnx:
                    return [([], [], [], [], []) for _ in frames]

                # Some ONNX exports are static with batch dimension fixed at 1.
                # Fall back to per-frame ONNX inference instead of aborting tracking.
                logger.warning(
                    "ONNX batched inference unavailable, falling back to per-frame ONNX inference."
                )
                results_batch = []
                for idx, frame in enumerate(frames):
                    try:
                        single_kwargs = dict(
                            source=frame,
                            conf=raw_conf_floor,
                            iou=1.0,
                            classes=target_classes,
                            max_det=max_det,
                            verbose=False,
                        )
                        if obb_predict_device is not None:
                            single_kwargs["device"] = obb_predict_device
                        if self.onnx_imgsz:
                            single_kwargs["imgsz"] = int(self.onnx_imgsz)
                        single_results = self.model.predict(**single_kwargs)
                        results_batch.append(
                            single_results[0] if len(single_results) > 0 else None
                        )
                    except Exception as frame_err:
                        logger.error(
                            "YOLO ONNX single-frame fallback failed at batch frame %d: %s",
                            start_frame_idx + idx,
                            frame_err,
                        )
                        results_batch.append(None)

        # Process each result
        batch_detections = []
        for idx in range(actual_frame_count):
            results = results_batch[idx]
            # frame_count = start_frame_idx + idx

            # Handle failed inference (None result from error)
            if results is None:
                batch_detections.append(([], [], [], [], []))
                continue

            if results.obb is None or len(results.obb) == 0:
                batch_detections.append(([], [], [], [], []))
                continue

            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
                self._extract_raw_detections(results.obb)
            )
            raw_heading_hints = [float("nan")] * len(raw_meas)
            raw_directed_mask = [0] * len(raw_meas)

            if return_raw:
                batch_detections.append(
                    (
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confidences,
                        raw_obb_corners,
                        raw_heading_hints,
                        raw_directed_mask,
                    )
                )
            else:
                (
                    meas,
                    sizes,
                    shapes,
                    confidences,
                    obb_corners_list,
                    _,
                    _heading_hints,
                    _directed_mask,
                ) = self.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=None,
                    detection_ids=None,
                    heading_hints=raw_heading_hints,
                    directed_mask=raw_directed_mask,
                )
                batch_detections.append(
                    (meas, sizes, shapes, confidences, obb_corners_list)
                )

            if progress_callback and (idx + 1) % 10 == 0:
                progress_callback(
                    idx + 1,
                    actual_frame_count,
                    f"Processing batch frame {idx + 1}/{actual_frame_count}",
                )

        return batch_detections

    def apply_conservative_split(self: object, fg_mask: object) -> object:
        """
        Placeholder method for compatibility with ObjectDetector interface.
        YOLO doesn't use foreground masks, so this is a no-op.
        """
        return fg_mask


def create_detector(params: object) -> object:
    """
    Factory function to create the appropriate detector based on configuration.

    Args:
        params: Configuration dictionary

    Returns:
        ObjectDetector or YOLOOBBDetector instance
    """
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")

    if detection_method == "yolo_obb":
        logger.info("Creating YOLO OBB detector")
        return YOLOOBBDetector(params)
    else:
        logger.info("Creating background subtraction detector")
        return ObjectDetector(params)


class DetectionFilter:
    """
    Lightweight post-hoc filter for cached raw YOLO detections.

    Contains only confidence thresholding and OBB IOU NMS — the exact same logic
    used by YOLOOBBDetector.filter_raw_detections — with no model loading.  Safe
    to instantiate cheaply inside inner optimizer loops.

    Usage::

        filt = DetectionFilter(params)
        meas, sizes, shapes, confs, corners, *_ = filt.filter_raw_detections(
            raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners
        )
    """

    def __init__(self, params):
        self.params = params

    # Assign the pure-logic methods from YOLOOBBDetector by reference.
    # None of them access YOLO-model state; they only read self.params and call
    # each other, so they work identically when bound to DetectionFilter instances.
    _compute_obb_iou = YOLOOBBDetector._compute_obb_iou
    _compute_obb_iou_batch = YOLOOBBDetector._compute_obb_iou_batch
    _filter_overlapping_detections = YOLOOBBDetector._filter_overlapping_detections
    filter_raw_detections = YOLOOBBDetector.filter_raw_detections
