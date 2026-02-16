"""
Object detection utilities for multi-object tracking.
Supports both background subtraction and YOLO OBB detection methods.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        self.device = self._detect_device()
        self.use_tensorrt = False
        self.use_onnx = False
        self.tensorrt_model_path = None
        self.onnx_model_path = None
        self.onnx_imgsz = None
        self.onnx_batch_size = 1
        self.tensorrt_batch_size = 1
        self._shapely_warning_shown = False  # Track if we've warned about shapely
        self._load_model()

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
            token = str(self.params.get("YOLO_MODEL_PATH", ""))
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
                    self.tensorrt_model_path = str(engine_path)
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
                self.tensorrt_model_path = str(engine_path)
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

        model_path_str = self.params.get("YOLO_MODEL_PATH", "yolov8n-obb.pt")
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
                return

        if (
            enable_tensorrt
            and TENSORRT_AVAILABLE
            and self.device.startswith("cuda")
            and local_model_file
        ):
            self._try_load_tensorrt_model(model_path_str)
            if self.use_tensorrt:
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

        try:
            from shapely import prepare
            from shapely.geometry import Polygon
            from shapely.validation import make_valid

            # Create and prepare reference polygon once
            poly1 = Polygon(corners1)
            if not poly1.is_valid:
                poly1 = make_valid(poly1)
            prepare(poly1)  # Optimize for multiple intersection checks

            area1 = poly1.area
            ious = np.zeros(len(indices))

            # Batch process all comparisons
            for i, idx in enumerate(indices):
                poly2 = Polygon(corners_list[idx])
                if not poly2.is_valid:
                    poly2 = make_valid(poly2)

                if not poly1.intersects(poly2):
                    ious[i] = 0.0
                    continue

                intersection = poly1.intersection(poly2).area
                union = area1 + poly2.area - intersection

                if union > 0:
                    ious[i] = intersection / union

            return ious

        except ImportError:
            # Fallback to individual calls
            return np.array(
                [self._compute_obb_iou(corners1, corners_list[idx]) for idx in indices]
            )

    def _compute_obb_iou(self, corners1, corners2):
        """
        Compute IOU between two oriented bounding boxes efficiently.

        Args:
            corners1: (4, 2) array of corner points for first OBB
            corners2: (4, 2) array of corner points for second OBB

        Returns:
            IOU value (0-1)
        """
        try:
            from shapely.geometry import Polygon
            from shapely.validation import make_valid

            # Create polygons from corners
            poly1 = Polygon(corners1)
            poly2 = Polygon(corners2)

            # Validate polygons (handle self-intersecting cases)
            if not poly1.is_valid:
                poly1 = make_valid(poly1)
            if not poly2.is_valid:
                poly2 = make_valid(poly2)

            # Quick rejection test
            if not poly1.intersects(poly2):
                return 0.0

            # Calculate intersection and union
            intersection = poly1.intersection(poly2).area
            union = poly1.area + poly2.area - intersection

            if union <= 0:
                return 0.0

            return intersection / union

        except ImportError:
            # Show warning once about using approximate IOU
            if not self._shapely_warning_shown:
                logger.info(
                    "Shapely not available - using axis-aligned bounding box IOU approximation. "
                    "For more accurate OBB filtering, install shapely: pip install shapely"
                )
                self._shapely_warning_shown = True

            # Fallback to axis-aligned bounding box IOU (less accurate but fast)
            # This is an approximation when shapely is not available
            x1_min, y1_min = corners1.min(axis=0)
            x1_max, y1_max = corners1.max(axis=0)
            x2_min, y2_min = corners2.min(axis=0)
            x2_max, y2_max = corners2.max(axis=0)

            # Calculate intersection of axis-aligned boxes
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0

            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = area1 + area2 - inter_area

            if union_area <= 0:
                return 0.0

            return inter_area / union_area

    def _filter_overlapping_detections(
        self,
        meas,
        sizes,
        shapes,
        confidences,
        obb_corners_list,
        iou_threshold,
        detection_ids=None,
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
            if detection_ids is None:
                return meas, sizes, shapes, confidences, obb_corners_list
            return meas, sizes, shapes, confidences, obb_corners_list, detection_ids

        n_detections = len(meas)

        # Convert inputs to numpy arrays for vectorized operations
        confidences_arr = np.array(confidences)

        # Pre-compute axis-aligned bounding boxes (fully vectorized)
        corners_array = np.array(obb_corners_list)  # (n, 4, 2)
        bbox_mins = corners_array.min(axis=1)  # (n, 2)
        bbox_maxs = corners_array.max(axis=1)  # (n, 2)
        bbox_areas = (bbox_maxs[:, 0] - bbox_mins[:, 0]) * (
            bbox_maxs[:, 1] - bbox_mins[:, 1]
        )

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
            curr_area = bbox_areas[current_idx]

            # Get remaining candidates
            remaining_indices = sorted_indices[idx + 1 :]
            rem_mins = bbox_mins[remaining_indices]
            rem_maxs = bbox_maxs[remaining_indices]
            rem_areas = bbox_areas[remaining_indices]

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
                # Calculate axis-aligned IOU (fast approximation)
                inter_areas = inter_wh[overlaps, 0] * inter_wh[overlaps, 1]
                union_areas = curr_area + rem_areas[overlaps] - inter_areas
                approx_ious = inter_areas / np.maximum(union_areas, 1e-6)

                # Initially keep all overlapping boxes, then selectively suppress high-IOU ones
                # This fixes the bug where low-IOU boxes were incorrectly removed
                overlapping_indices = np.where(overlaps)[0]
                keep_remaining[overlapping_indices] = True

                # Identify which need precise polygon IOU check
                # Only check if approx IOU is within 0.2 of threshold
                need_precise = approx_ious >= (iou_threshold - 0.2)

                if need_precise.any():
                    # Batch compute precise IOUs for candidates
                    overlapping_local = np.where(overlaps)[0]
                    precise_check_local = overlapping_local[need_precise]
                    precise_check_global = remaining_indices[precise_check_local]

                    # Batch IOU computation
                    precise_ious = self._compute_obb_iou_batch(
                        obb_corners_list[current_idx],
                        obb_corners_list,
                        precise_check_global,
                    )

                    # Mark overlapping detections for removal
                    suppress = precise_ious >= iou_threshold
                    keep_remaining[precise_check_local] = ~suppress

                # For boxes where approx IOU already exceeds threshold, remove them
                low_precision_suppress = (~need_precise) & (
                    approx_ious >= iou_threshold
                )
                if low_precision_suppress.any():
                    low_precision_indices = np.where(overlaps)[0][~need_precise][
                        low_precision_suppress
                    ]
                    keep_remaining[low_precision_indices] = False

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
        if detection_ids is not None:
            detection_ids = [detection_ids[i] for i in keep_indices]
            return meas, sizes, shapes, confidences, obb_corners_list, detection_ids

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
    ):
        """
        Apply vectorized confidence/size/ROI filtering, then custom OBB IOU suppression.
        This is shared by live detection and cached-raw detection paths.
        """
        if not meas:
            return [], [], [], [], [], []

        conf_threshold = float(self.params.get("YOLO_CONFIDENCE_THRESHOLD", 0.25))
        iou_threshold = float(self.params.get("YOLO_IOU_THRESHOLD", 0.7))
        max_targets = max(1, int(self.params.get("MAX_TARGETS", 8)))

        meas_arr = np.ascontiguousarray(np.asarray(meas, dtype=np.float32))
        sizes_arr = np.ascontiguousarray(np.asarray(sizes, dtype=np.float32))
        shapes_arr = np.ascontiguousarray(np.asarray(shapes, dtype=np.float32))
        conf_arr = np.ascontiguousarray(np.asarray(confidences, dtype=np.float32))

        if detection_ids is None:
            ids_arr = np.arange(len(meas_arr), dtype=np.float64)
        else:
            ids_arr = np.ascontiguousarray(np.asarray(detection_ids, dtype=np.float64))

        n = min(
            len(meas_arr), len(sizes_arr), len(shapes_arr), len(conf_arr), len(ids_arr)
        )
        if obb_corners_list:
            n = min(n, len(obb_corners_list))
        if n == 0:
            return [], [], [], [], [], []

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
            return [], [], [], [], [], []

        meas_arr = meas_arr[keep_mask]
        sizes_arr = sizes_arr[keep_mask]
        shapes_arr = shapes_arr[keep_mask]
        conf_arr = conf_arr[keep_mask]
        ids_arr = ids_arr[keep_mask]
        obb_arr = obb_arr[keep_mask]

        meas_list = [meas_arr[i] for i in range(len(meas_arr))]
        sizes_list = sizes_arr.tolist()
        shapes_list = [tuple(shapes_arr[i]) for i in range(len(shapes_arr))]
        conf_list = conf_arr.tolist()
        ids_list = ids_arr.tolist()
        obb_list = [obb_arr[i] for i in range(len(obb_arr))]

        if len(meas_list) > 1:
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

        if len(meas_list) > max_targets:
            idxs = np.argsort(sizes_list)[::-1][:max_targets]
            meas_list = [meas_list[i] for i in idxs]
            sizes_list = [sizes_list[i] for i in idxs]
            shapes_list = [shapes_list[i] for i in idxs]
            conf_list = [conf_list[i] for i in idxs]
            obb_list = [obb_list[i] for i in idxs]
            ids_list = [ids_list[i] for i in idxs]

        return meas_list, sizes_list, shapes_list, conf_list, obb_list, ids_list

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
                raw_meas, raw_sizes, raw_shapes, yolo_results, raw_confidences, raw_obb_corners
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            if return_raw:
                return [], [], [], None, [], []
            return [], [], [], None, []

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)  # None means all classes
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

        # Run inference on the configured device
        try:
            fixed_batch = 1
            if self.use_tensorrt and int(getattr(self, "tensorrt_batch_size", 1)) > 1:
                fixed_batch = int(self.tensorrt_batch_size)
            elif self.use_onnx and int(getattr(self, "onnx_batch_size", 1)) > 1:
                fixed_batch = int(self.onnx_batch_size)

            if fixed_batch > 1:
                # Static-batch runtimes require exact batch dimension.
                source_input = [frame] * fixed_batch
            else:
                source_input = frame
            predict_kwargs = dict(
                source=source_input,
                conf=raw_conf_floor,
                iou=1.0,  # Always use custom OBB IOU filtering after inference
                classes=target_classes,
                max_det=max_det,
                device=self.device,
                verbose=False,
            )
            if self.use_onnx and self.onnx_imgsz:
                predict_kwargs["imgsz"] = int(self.onnx_imgsz)
            results = self.model.predict(**predict_kwargs)
            if fixed_batch > 1:
                results = results[:1]
        except Exception as e:
            logger.error(f"YOLO inference failed on frame {frame_count}: {e}")
            if return_raw:
                return [], [], [], None, [], []
            return [], [], [], None, []

        if len(results) == 0 or results[0].obb is None or len(results[0].obb) == 0:
            if return_raw:
                return [], [], [], results[0] if len(results) > 0 else None, [], []
            return [], [], [], results[0] if len(results) > 0 else None, []

        raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners = (
            self._extract_raw_detections(results[0].obb)
        )

        if return_raw:
            results[0]._raw_obb_corners = raw_obb_corners
            return (
                raw_meas,
                raw_sizes,
                raw_shapes,
                results[0],
                raw_confidences,
                raw_obb_corners,
            )

        meas, sizes, shapes, confidences, obb_corners_list, _ = (
            self.filter_raw_detections(
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confidences,
                raw_obb_corners,
                roi_mask=None,
                detection_ids=None,
            )
        )

        if meas:
            logger.debug(f"Frame {frame_count}: YOLO detected {len(meas)} objects")

        # Return filtered OBB corners alongside other data
        # Store in results object for access by individual dataset generator
        results[0]._filtered_obb_corners = obb_corners_list

        return meas, sizes, shapes, results[0], confidences

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
              - return_raw=True:  (raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners)
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return [([], [], [], [], []) for _ in frames]

        p = self.params
        target_classes = p.get("YOLO_TARGET_CLASSES", None)
        raw_conf_floor = max(1e-4, float(p.get("RAW_YOLO_CONFIDENCE_FLOOR", 1e-3)))
        max_det = self._raw_detection_cap()

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
                        device=self.device,
                        verbose=False,
                    )
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
                predict_kwargs = dict(
                    source=frames,
                    conf=raw_conf_floor,
                    iou=1.0,  # Always use custom OBB IOU filtering after inference
                    classes=target_classes,
                    max_det=max_det,
                    device=self.device,
                    verbose=False,
                )
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
                            device=self.device,
                            verbose=False,
                        )
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

            if return_raw:
                batch_detections.append(
                    (raw_meas, raw_sizes, raw_shapes, raw_confidences, raw_obb_corners)
                )
            else:
                (
                    meas,
                    sizes,
                    shapes,
                    confidences,
                    obb_corners_list,
                    _,
                ) = self.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confidences,
                    raw_obb_corners,
                    roi_mask=None,
                    detection_ids=None,
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
