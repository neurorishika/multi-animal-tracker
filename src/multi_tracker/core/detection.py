"""
Object detection utilities for multi-object tracking.
Supports both background subtraction and YOLO OBB detection methods.
"""

import numpy as np
import cv2
import logging
from pathlib import Path

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

    def apply_conservative_split(self, fg_mask):
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

    def detect_objects(self, fg_mask, frame_count):
        """Detects and measures objects from the final foreground mask."""
        p = self.params
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        N = p["MAX_TARGETS"]
        max_allowed_contours = N * p.get("MAX_CONTOUR_MULTIPLIER", 20)

        if len(cnts) > max_allowed_contours:
            logger.debug(
                f"Frame {frame_count}: Too many contours ({len(cnts)}), skipping."
            )
            return [], [], []

        meas, sizes, shapes = [], [], []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p["MIN_CONTOUR_AREA"] or len(c) < 5:
                continue

            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)

            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                ang = (ang + 90) % 180

            meas.append(np.array([cx, cy, np.deg2rad(ang)], np.float32))
            sizes.append(area)
            shapes.append((np.pi * (ax1 / 2) * (ax2 / 2), ax1 / ax2 if ax2 > 0 else 0))

        if meas and p.get("ENABLE_SIZE_FILTERING", False):
            min_size = p.get("MIN_OBJECT_SIZE", 0)
            max_size = p.get("MAX_OBJECT_SIZE", float("inf"))

            original_count = len(meas)
            filtered = [
                (m, s, sh)
                for m, s, sh in zip(meas, sizes, shapes)
                if min_size <= s <= max_size
            ]

            if filtered:
                meas, sizes, shapes = zip(*filtered)
                meas, sizes, shapes = list(meas), list(sizes), list(shapes)
            else:
                meas, sizes, shapes = [], [], []

            if len(meas) != original_count:
                logger.debug(
                    f"Size filtering: {original_count} -> {len(meas)} detections"
                )

        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]

        return meas, sizes, shapes


class YOLOOBBDetector:
    """
    Detects objects using a pretrained YOLO OBB (Oriented Bounding Box) model.
    Compatible interface with ObjectDetector for seamless integration.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.device = self._detect_device()
        self._load_model()

    def _detect_device(self):
        """Detect and configure the optimal device for inference."""
        import torch
        
        # Check user preference
        device_preference = self.params.get("YOLO_DEVICE", "auto")
        
        if device_preference != "auto":
            logger.info(f"Using user-specified device: {device_preference}")
            return device_preference
        
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info(f"CUDA GPU detected, using {device}")
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon GPU
            logger.info("Apple Metal Performance Shaders (MPS) detected, using mps")
        else:
            device = "cpu"
            logger.info("No GPU detected, using CPU")
        
        return device

    def _load_model(self):
        """Load the YOLO OBB model."""
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

        # For pretrained model names (yolo26s-obb.pt, etc.), pass directly to YOLO
        # These will be auto-downloaded by ultralytics
        if model_path_str.startswith(("yolov8", "yolov11", "yolo26")):
            try:
                self.model = YOLO(model_path_str)
                # Move model to the appropriate device
                self.model.to(self.device)
                logger.info(f"YOLO OBB model loaded successfully: {model_path_str} on device: {self.device}")
                return
            except Exception as e:
                logger.error(f"Failed to load YOLO model '{model_path_str}': {e}")
                raise

        # For custom model paths, resolve and validate
        model_path = Path(model_path_str).expanduser().resolve()

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
            logger.info(f"YOLO OBB model loaded successfully from {model_path} on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

    def detect_objects(self, frame, frame_count):
        """
        Detects objects in a frame using YOLO OBB.

        Args:
            frame: Input frame (grayscale or BGR)
            frame_count: Current frame number for logging

        Returns:
            meas: List of measurements [cx, cy, angle] in radians
            sizes: List of detection areas
            shapes: List of (area, aspect_ratio) tuples
        """
        if self.model is None:
            logger.error("YOLO model not initialized")
            return [], [], []

        p = self.params
        conf_threshold = p.get("YOLO_CONFIDENCE_THRESHOLD", 0.25)
        iou_threshold = p.get("YOLO_IOU_THRESHOLD", 0.7)
        target_classes = p.get("YOLO_TARGET_CLASSES", None)  # None means all classes
        max_det = p.get("MAX_TARGETS", 8) * p.get("MAX_CONTOUR_MULTIPLIER", 20)

        # Run inference on the configured device
        try:
            results = self.model.predict(
                frame,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=target_classes,
                max_det=max_det,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"YOLO inference failed on frame {frame_count}: {e}")
            return [], [], []

        if len(results) == 0 or results[0].obb is None or len(results[0].obb) == 0:
            return [], [], []

        # Extract OBB detections
        obb_data = results[0].obb

        meas, sizes, shapes = [], [], []

        for i in range(len(obb_data)):
            # Get OBB parameters
            # obb_data.xyxyxyxy gives the 4 corner points
            # obb_data.xywhr gives [center_x, center_y, width, height, rotation]
            xywhr = obb_data.xywhr[i].cpu().numpy()
            cx, cy, w, h, angle_rad = xywhr

            # Convert angle to match our convention (0-180 degrees, then to radians)
            # YOLO OBB angles are typically in radians, ranging from -pi/2 to pi/2
            angle_deg = np.rad2deg(angle_rad) % 180

            # Ensure major axis is first (ax1 >= ax2)
            if w < h:
                w, h = h, w
                angle_deg = (angle_deg + 90) % 180

            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Apply size filtering if enabled
            if p.get("ENABLE_SIZE_FILTERING", False):
                min_size = p.get("MIN_OBJECT_SIZE", 0)
                max_size = p.get("MAX_OBJECT_SIZE", float("inf"))
                if not (min_size <= area <= max_size):
                    continue

            # Store measurement in format [cx, cy, angle_rad]
            meas.append(np.array([cx, cy, np.deg2rad(angle_deg)], dtype=np.float32))
            sizes.append(float(area))
            shapes.append(
                (np.pi * (w / 2) * (h / 2), aspect_ratio)
            )  # Ellipse area approximation

        # Keep only top N detections by size
        N = p["MAX_TARGETS"]
        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            sizes = [sizes[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]

        if meas:
            logger.debug(f"Frame {frame_count}: YOLO detected {len(meas)} objects")

        return meas, sizes, shapes

    def apply_conservative_split(self, fg_mask):
        """
        Placeholder method for compatibility with ObjectDetector interface.
        YOLO doesn't use foreground masks, so this is a no-op.
        """
        return fg_mask


def create_detector(params):
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
