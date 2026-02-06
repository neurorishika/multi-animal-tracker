"""
Individual-level analysis for identity classification and pose tracking.
Extracts regions around detections for downstream processing.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import json

from multi_tracker.utils.image_processing import compute_median_color_from_frame

logger = logging.getLogger(__name__)


class IdentityProcessor:
    """
    Base class for individual identity classification.
    Processes cropped regions around detections to assign identities.
    """

    def __init__(self, params):
        self.params = params
        self.method = params.get("IDENTITY_METHOD", "none")
        self.enabled = params.get("ENABLE_IDENTITY_ANALYSIS", False)

        # Crop parameters
        self.crop_size_multiplier = params.get("IDENTITY_CROP_SIZE_MULTIPLIER", 3.0)
        self.crop_min_size = params.get("IDENTITY_CROP_MIN_SIZE", 64)
        self.crop_max_size = params.get("IDENTITY_CROP_MAX_SIZE", 256)

        logger.info(
            f"Identity processor initialized: method={self.method}, enabled={self.enabled}"
        )

    def extract_crop(
        self, frame, cx, cy, body_size, theta=None, padding_multiplier=None
    ):
        """
        Extract a crop around a detection.

        Args:
            frame: Input frame (BGR)
            cx, cy: Center coordinates
            body_size: Reference body size for scaling crop
            theta: Orientation angle (for aligned crops)
            padding_multiplier: Override default crop size multiplier

        Returns:
            crop: Cropped image region
            crop_info: Dict with crop metadata (bbox, transform, etc.)
        """
        if padding_multiplier is None:
            padding_multiplier = self.crop_size_multiplier

        # Calculate crop size based on body size
        crop_size = int(body_size * padding_multiplier)
        crop_size = max(self.crop_min_size, min(crop_size, self.crop_max_size))

        # Make crop size even for easier processing
        if crop_size % 2 != 0:
            crop_size += 1

        half_size = crop_size // 2

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Calculate crop bounds with boundary checks
        x1 = max(0, int(cx - half_size))
        y1 = max(0, int(cy - half_size))
        x2 = min(w, int(cx + half_size))
        y2 = min(h, int(cy + half_size))

        # Extract crop
        crop = frame[y1:y2, x1:x2].copy()

        # Pad if crop hit frame boundary
        actual_h, actual_w = crop.shape[:2]
        if actual_h < crop_size or actual_w < crop_size:
            # Pad to desired size
            pad_top = (crop_size - actual_h) // 2
            pad_bottom = crop_size - actual_h - pad_top
            pad_left = (crop_size - actual_w) // 2
            pad_right = crop_size - actual_w - pad_left
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        crop_info = {
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "size": crop_size,
            "theta": theta,
            "padded": (actual_h < crop_size or actual_w < crop_size),
        }

        return crop, crop_info

    def process_frame(self, frame, detections, frame_id):
        """
        Process all detections in a frame to assign identities.

        Args:
            frame: Input frame (BGR)
            detections: List of detection dicts with keys: cx, cy, theta, body_size, track_id
            frame_id: Frame number

        Returns:
            identities: List of identity labels (one per detection)
            identity_confidences: List of confidence scores
            crops: List of extracted crop images (for visualization/debugging)
        """
        if not self.enabled or not detections:
            return [None] * len(detections), [0.0] * len(detections), []

        identities = []
        confidences = []
        crops = []

        for detection in detections:
            cx = detection["cx"]
            cy = detection["cy"]
            theta = detection.get("theta", 0)
            body_size = detection.get(
                "body_size", self.params.get("REFERENCE_BODY_SIZE", 20.0)
            )

            # Extract crop
            crop, crop_info = self.extract_crop(frame, cx, cy, body_size, theta)
            crops.append(crop)

            # Process crop to get identity (dummy for now)
            identity, confidence = self._classify_identity(crop, crop_info, detection)
            identities.append(identity)
            confidences.append(confidence)

        return identities, confidences, crops

    def _classify_identity(self, crop, crop_info, detection):
        """
        Classify identity from crop. Override this method for specific implementations.

        Args:
            crop: Cropped image
            crop_info: Crop metadata
            detection: Original detection data

        Returns:
            identity: Identity label (str or int)
            confidence: Classification confidence (0-1)
        """
        # Dummy implementation - returns placeholder
        if self.method == "none":
            return None, 0.0
        elif self.method == "color_tags":
            return self._dummy_color_tag_classifier(crop)
        elif self.method == "apriltags":
            return self._dummy_apriltag_classifier(crop)
        else:
            return None, 0.0

    def _dummy_color_tag_classifier(self, crop):
        """Placeholder for color tag classification."""
        # TODO: Implement YOLO-based color tag detection
        return f"color_tag_{np.random.randint(0, 10)}", 0.5

    def _dummy_apriltag_classifier(self, crop):
        """Placeholder for AprilTag detection."""
        # TODO: Implement AprilTag detection
        return f"apriltag_{np.random.randint(0, 10)}", 0.5


class IndividualDatasetGenerator:
    """
    Real-time individual dataset generator that runs during tracking.

    Exports OBB-masked crops for each detection, where only the detected
    animal's OBB region is visible and the rest is masked. This provides
    clean, isolated training data for individual-level analysis.

    Key features:
    - Runs in parallel with tracking (called per-frame)
    - Uses actual OBB polygon to mask out other animals
    - Saves crops with track/trajectory ID labels
    - Uses already-filtered detections (ROI + size filtering done by tracking)
    """

    def __init__(self, params, output_dir, video_name, dataset_name=None):
        """
        Initialize the individual dataset generator.

        Args:
            params: Parameter dictionary
            output_dir: Base directory for saving crops
            video_name: Name of the source video (for organizing output)
            dataset_name: Optional custom name for the dataset
        """
        self.params = params
        self.enabled = params.get("ENABLE_INDIVIDUAL_DATASET", False)

        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else None
        self.video_name = video_name
        self.dataset_name = dataset_name or params.get(
            "INDIVIDUAL_DATASET_NAME", "individual_dataset"
        )

        # Crop parameters - only padding (crop size is determined by OBB)
        self.padding_fraction = params.get("INDIVIDUAL_CROP_PADDING", 0.1)
        # Background color as BGR tuple (default: black)
        bg_color = params.get("INDIVIDUAL_BACKGROUND_COLOR", (0, 0, 0))
        if isinstance(bg_color, (list, tuple)) and len(bg_color) == 3:
            self.background_color = tuple(bg_color)
        else:
            self.background_color = (0, 0, 0)

        # Save interval (use detections as-is, just control frequency)
        self.save_every_n_frames = params.get("INDIVIDUAL_SAVE_INTERVAL", 1)

        # Output format
        self.output_format = params.get("INDIVIDUAL_OUTPUT_FORMAT", "png")  # png or jpg
        self.jpg_quality = params.get("INDIVIDUAL_JPG_QUALITY", 95)

        # Statistics
        self.total_saved = 0
        self.crops_dir = None
        self.metadata = []

        # Initialize output directory
        if self.enabled and self.output_dir:
            self._setup_output_directory()

        logger.info(
            f"Individual dataset generator initialized: enabled={self.enabled}, "
            f"output_dir={self.output_dir}"
        )

    def set_background_color(self, color):
        """
        Set the background color for masked crops.

        Args:
            color: Tuple of (B, G, R) values (0-255) or single value for grayscale
        """
        if isinstance(color, (list, tuple)) and len(color) == 3:
            self.background_color = tuple(color)
        else:
            raise ValueError(
                f"Background color must be a tuple of 3 values (BGR), got {color}"
            )

    @staticmethod
    def compute_median_color_from_frame(frame):
        """
        Compute the median color (BGR) from a frame.

        DEPRECATED: Use compute_median_color_from_frame from image_processing utils instead.
        This method is kept for backward compatibility.

        Args:
            frame: Input frame (BGR, shape: H x W x 3)

        Returns:
            Tuple of (B, G, R) median values
        """
        return compute_median_color_from_frame(frame)

    @staticmethod
    def ellipse_to_obb_corners(cx, cy, major_axis, minor_axis, theta):
        """
        Convert ellipse parameters to OBB corner points.

        The OBB is the oriented bounding box that exactly fits the ellipse,
        which is a rotated rectangle with dimensions (major_axis × minor_axis).

        Args:
            cx, cy: Center coordinates of the ellipse
            major_axis: Full length of the major axis (not semi-axis)
            minor_axis: Full length of the minor axis (not semi-axis)
            theta: Rotation angle in radians (orientation of major axis)

        Returns:
            corners: numpy array of shape (4, 2) with corner coordinates
        """
        # Half-lengths of the ellipse axes
        a = major_axis / 2.0  # Semi-major axis
        b = minor_axis / 2.0  # Semi-minor axis

        # Corner offsets in local (unrotated) coordinate system
        # The rectangle has width=major_axis, height=minor_axis
        local_corners = np.array(
            [
                [a, b],  # Top-right
                [-a, b],  # Top-left
                [-a, -b],  # Bottom-left
                [a, -b],  # Bottom-right
            ]
        )

        # Rotation matrix
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate corners and translate to center
        rotated_corners = local_corners @ rotation.T
        corners = rotated_corners + np.array([cx, cy])

        return corners.astype(np.float32)

    def _setup_output_directory(self):
        """Create output directory structure."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_folder_name = f"{self.dataset_name}_{timestamp}"
        self.crops_dir = self.output_dir / dataset_folder_name / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file path
        self.metadata_path = self.crops_dir.parent / "metadata.json"

        logger.info(f"Individual dataset output directory: {self.crops_dir}")

    def process_frame(
        self,
        frame,
        frame_id,
        meas,
        obb_corners=None,
        ellipse_params=None,
        confidences=None,
        track_ids=None,
        trajectory_ids=None,
        coord_scale_factor=1.0,
        detection_ids=None,
    ):
        """
        Process a frame and save masked crops for each detection.

        Supports both YOLO OBB detections and ellipse detections from
        background subtraction. For ellipses, OBB corners are computed
        from the ellipse parameters.

        Detections passed here are already filtered by ROI and size filtering
        in the tracking pipeline - no additional filtering needed.

        Args:
            frame: Input frame (BGR) - should be ORIGINAL resolution
            frame_id: Current frame number
            meas: List of measurements [cx, cy, theta, ...] for each detection (in detection resolution)
            obb_corners: List of OBB corner arrays (4 points each) for YOLO detections (in detection resolution)
            ellipse_params: List of ellipse params [major_axis, minor_axis] for BG sub detections (in detection resolution)
                           (center and theta are taken from meas)
            confidences: Optional list of confidence scores
            track_ids: Optional list of track IDs for each detection
            trajectory_ids: Optional list of trajectory IDs for each detection
            coord_scale_factor: Scale factor to convert detection coords to original resolution (1/resize_factor)
            detection_ids: Optional list of unique Detection IDs for each detection

        Returns:
            num_saved: Number of crops saved from this frame
        """
        if not self.enabled or self.crops_dir is None:
            return 0

        if not meas:
            return 0

        # Must have either obb_corners or ellipse_params
        if not obb_corners and not ellipse_params:
            return 0

        # Check save interval
        if frame_id % self.save_every_n_frames != 0:
            return 0

        num_saved = 0
        h, w = frame.shape[:2]

        # Determine detection source type
        use_obb = obb_corners is not None and len(obb_corners) > 0
        use_ellipse = ellipse_params is not None and len(ellipse_params) > 0

        for i in range(len(meas)):
            # Get detection info and scale to original resolution
            m = meas[i]
            cx = float(m[0]) * coord_scale_factor
            cy = float(m[1]) * coord_scale_factor
            theta = float(m[2]) if len(m) > 2 else 0.0

            # Get OBB corners - either directly or computed from ellipse
            # Scale corners/params to match original resolution
            if use_obb and i < len(obb_corners):
                # Scale OBB corners to original resolution
                corners = (
                    np.asarray(obb_corners[i], dtype=np.float32) * coord_scale_factor
                )  # Shape: (4, 2)
                source_type = "yolo_obb"
            elif use_ellipse and i < len(ellipse_params):
                # Get ellipse parameters and scale to original resolution
                ep = ellipse_params[i]
                major_axis = float(ep[0]) * coord_scale_factor
                minor_axis = float(ep[1]) * coord_scale_factor
                corners = self.ellipse_to_obb_corners(
                    cx, cy, major_axis, minor_axis, theta
                )
                source_type = "ellipse"
            else:
                # Skip if no geometry data available for this detection
                continue

            # Get confidence
            conf = confidences[i] if confidences and i < len(confidences) else 1.0

            # Extract and save masked crop
            crop, crop_info = self._extract_obb_masked_crop(frame, corners, h, w)

            if crop is not None:
                # Build metadata for this crop
                track_id = track_ids[i] if track_ids and i < len(track_ids) else -1
                traj_id = (
                    trajectory_ids[i]
                    if trajectory_ids and i < len(trajectory_ids)
                    else -1
                )
                det_id = (
                    detection_ids[i]
                    if detection_ids and i < len(detection_ids)
                    else None
                )

                crop_metadata = {
                    "frame_id": int(frame_id),
                    "detection_idx": i,
                    "track_id": int(track_id),
                    "trajectory_id": int(traj_id),
                    "detection_id": int(det_id) if det_id is not None else None,
                    "confidence": float(conf),
                    "center": [cx, cy],
                    "theta": theta,
                    "crop_size": crop_info["crop_size"],
                    "obb_corners": corners.tolist(),
                    "source_type": source_type,
                }

                # Save crop
                saved = self._save_crop(
                    crop, frame_id, i, track_id, crop_metadata, detection_id=det_id
                )
                if saved:
                    num_saved += 1

        return num_saved

    def _extract_obb_masked_crop(self, frame, corners, frame_h, frame_w):
        """
        Extract a crop with only the OBB region visible.

        Expands OBB corners outward from centroid by padding_fraction to
        capture more of the animal in case detection clips edges.

        Args:
            frame: Input frame (BGR)
            corners: OBB corner points (4, 2)
            frame_h, frame_w: Frame dimensions

        Returns:
            crop: Masked crop image
            crop_info: Crop metadata dict
        """
        # Expand OBB corners outward from centroid
        # This preserves the OBB shape while making it larger
        centroid = corners.mean(axis=0)  # Center of the OBB

        # Expand each corner away from centroid by padding_fraction
        expanded_corners = corners.copy()
        for i in range(4):
            direction = corners[i] - centroid
            expanded_corners[i] = centroid + direction * (1.0 + self.padding_fraction)

        # Clip expanded corners to frame bounds
        expanded_corners[:, 0] = np.clip(expanded_corners[:, 0], 0, frame_w - 1)
        expanded_corners[:, 1] = np.clip(expanded_corners[:, 1], 0, frame_h - 1)

        # Calculate bounding box of the expanded OBB (axis-aligned)
        x_min = max(0, int(expanded_corners[:, 0].min()))
        x_max = min(frame_w, int(expanded_corners[:, 0].max()) + 1)
        y_min = max(0, int(expanded_corners[:, 1].min()))
        y_max = min(frame_h, int(expanded_corners[:, 1].max()) + 1)

        crop_w = x_max - x_min
        crop_h = y_max - y_min

        # Skip if crop would be empty
        if crop_w <= 0 or crop_h <= 0:
            return None, None

        # Extract crop region
        crop = frame[y_min:y_max, x_min:x_max].copy()

        # Create mask for expanded OBB region (shift corners to crop coordinates)
        shifted_corners = expanded_corners.copy()
        shifted_corners[:, 0] -= x_min
        shifted_corners[:, 1] -= y_min

        # Create OBB polygon mask
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_corners.astype(np.int32)], 255)

        # Apply mask - keep OBB region, fill rest with background color
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_crop = cv2.bitwise_and(crop, mask_3ch)

        # Fill background with the specified color
        background = np.full_like(crop, self.background_color, dtype=np.uint8)
        mask_inv = cv2.bitwise_not(mask_3ch)
        background = cv2.bitwise_and(background, mask_inv)
        masked_crop = cv2.add(masked_crop, background)

        crop_info = {
            "crop_size": (crop_w, crop_h),
            "crop_bbox": (x_min, y_min, x_max, y_max),
            "obb_corners_local": shifted_corners.tolist(),
            "expansion_factor": 1.0 + self.padding_fraction,
        }

        return masked_crop, crop_info

    def _save_crop(
        self, crop, frame_id, det_idx, track_id, metadata, detection_id=None
    ):
        """
        Save a crop to disk.

        Args:
            crop: Image to save
            frame_id: Frame number
            det_idx: Detection index within frame
            track_id: Track ID
            metadata: Metadata dict for this crop
            detection_id: Unique Detection ID (optional)

        Returns:
            bool: True if saved successfully
        """
        try:
            # Generate filename
            if detection_id is not None:
                # Use DetectionID if available (preferred/unique)
                filename = f"did{int(detection_id)}.{self.output_format}"
            elif track_id >= 0:
                filename = f"f{frame_id:06d}_t{track_id:04d}_d{det_idx:02d}.{self.output_format}"
            else:
                filename = f"f{frame_id:06d}_d{det_idx:02d}.{self.output_format}"

            filepath = self.crops_dir / filename

            # Save image
            if self.output_format == "jpg":
                cv2.imwrite(
                    str(filepath), crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpg_quality]
                )
            else:
                cv2.imwrite(str(filepath), crop)

            # Store metadata
            metadata["filename"] = filename
            self.metadata.append(metadata)
            self.total_saved += 1

            return True

        except Exception as e:
            logger.warning(f"Failed to save crop: {e}")
            return False

    def finalize(self):
        """
        Finalize the dataset and save metadata.
        Called when tracking completes.

        Returns:
            str: Path to the dataset directory, or None if not enabled
        """
        if not self.enabled or self.crops_dir is None:
            return None

        # Save metadata JSON
        dataset_info = {
            "video_name": self.video_name,
            "total_crops": self.total_saved,
            "parameters": {
                "padding_fraction": self.padding_fraction,
                "save_interval": self.save_every_n_frames,
                "output_format": self.output_format,
            },
            "crops": self.metadata,
        }

        try:
            with open(self.metadata_path, "w") as f:
                json.dump(dataset_info, f, indent=2)

            logger.info(
                f"Individual dataset complete: {self.total_saved} crops saved to "
                f"{self.crops_dir.parent}"
            )

            # Create README
            readme_path = self.crops_dir.parent / "README.md"
            with open(readme_path, "w") as f:
                f.write(f"# Individual Detection Dataset\n\n")
                f.write(f"Generated from: {self.video_name}\n\n")
                f.write(f"## Contents\n\n")
                f.write(f"- **{self.total_saved} OBB-masked crops**\n")
                f.write(f"- Each crop contains only the detected animal (OBB region)\n")
                f.write(f"- Background outside OBB is masked to black\n")
                f.write(
                    f"- Detections are pre-filtered by ROI and size in tracking\n\n"
                )
                f.write(f"## File Naming\n\n")
                f.write(f"- `fXXXXXX_tYYYY_dZZ.{self.output_format}`\n")
                f.write(f"  - `fXXXXXX`: Frame number\n")
                f.write(f"  - `tYYYY`: Track ID\n")
                f.write(f"  - `dZZ`: Detection index within frame\n\n")
                f.write(f"## Usage\n\n")
                f.write(f"These crops can be used for:\n")
                f.write(f"- Training individual identity classifiers\n")
                f.write(f"- Pose estimation model training\n")
                f.write(f"- Behavior classification\n")

            return str(self.crops_dir.parent)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return None
