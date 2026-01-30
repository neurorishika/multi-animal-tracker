"""
Individual-level analysis for identity classification and pose tracking.
Extracts regions around detections for downstream processing.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
import json

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

    def __init__(self, params, output_dir, video_name):
        """
        Initialize the individual dataset generator.

        Args:
            params: Parameter dictionary
            output_dir: Base directory for saving crops
            video_name: Name of the source video (for organizing output)
        """
        self.params = params
        self.enabled = params.get("ENABLE_INDIVIDUAL_DATASET", False)

        # Output configuration
        self.output_dir = Path(output_dir) if output_dir else None
        self.video_name = video_name

        # Crop parameters - only padding (crop size is determined by OBB)
        self.padding_fraction = params.get("INDIVIDUAL_CROP_PADDING", 0.1)
        self.mask_fill_value = params.get("INDIVIDUAL_MASK_FILL", 0)  # Black fill

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

    def _setup_output_directory(self):
        """Create output directory structure."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"{self.video_name}_individual_{timestamp}"
        self.crops_dir = self.output_dir / dataset_name / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)

        # Create metadata file path
        self.metadata_path = self.crops_dir.parent / "metadata.json"

        logger.info(f"Individual dataset output directory: {self.crops_dir}")

    def process_frame(
        self,
        frame,
        frame_id,
        meas,
        obb_corners,
        confidences=None,
        track_ids=None,
        trajectory_ids=None,
    ):
        """
        Process a frame and save OBB-masked crops for each detection.

        Detections passed here are already filtered by ROI and size filtering
        in the tracking pipeline - no additional filtering needed.

        Args:
            frame: Input frame (BGR)
            frame_id: Current frame number
            meas: List of measurements [cx, cy, theta] for each detection
            obb_corners: List of OBB corner arrays (4 points each) for each detection
            confidences: Optional list of confidence scores
            track_ids: Optional list of track IDs for each detection
            trajectory_ids: Optional list of trajectory IDs for each detection

        Returns:
            num_saved: Number of crops saved from this frame
        """
        if not self.enabled or self.crops_dir is None:
            return 0

        if not meas or not obb_corners:
            return 0

        # Check save interval
        if frame_id % self.save_every_n_frames != 0:
            return 0

        num_saved = 0
        h, w = frame.shape[:2]

        for i in range(len(meas)):
            # Get detection info
            m = meas[i]
            cx, cy = float(m[0]), float(m[1])
            corners = obb_corners[i]  # Shape: (4, 2)

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

                crop_metadata = {
                    "frame_id": int(frame_id),
                    "detection_idx": i,
                    "track_id": int(track_id),
                    "trajectory_id": int(traj_id),
                    "confidence": float(conf),
                    "center": [cx, cy],
                    "crop_size": crop_info["crop_size"],
                    "obb_corners": corners.tolist(),
                }

                # Save crop
                saved = self._save_crop(crop, frame_id, i, track_id, crop_metadata)
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

        # Apply mask - keep OBB region, fill rest with mask_fill_value
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        masked_crop = cv2.bitwise_and(crop, mask_3ch)

        # Fill background if not black
        if self.mask_fill_value != 0:
            background = np.full_like(crop, self.mask_fill_value)
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

    def _save_crop(self, crop, frame_id, det_idx, track_id, metadata):
        """
        Save a crop to disk.

        Args:
            crop: Image to save
            frame_id: Frame number
            det_idx: Detection index within frame
            track_id: Track ID
            metadata: Metadata dict for this crop

        Returns:
            bool: True if saved successfully
        """
        try:
            # Generate filename
            if track_id >= 0:
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


class PoseTrackingExporter:
    """
    Export trajectory data for post-hoc pose tracking training.
    Creates videos and metadata for each trajectory.
    """

    def __init__(self, params, progress_callback=None):
        self.params = params
        self.enabled = params.get("ENABLE_POSE_EXPORT", False)
        self.crop_size_multiplier = params.get("POSE_CROP_SIZE_MULTIPLIER", 4.0)
        self.min_trajectory_length = params.get("POSE_MIN_TRAJECTORY_LENGTH", 30)
        self.export_fps = params.get("POSE_EXPORT_FPS", 30)
        self.progress_callback = progress_callback

        logger.info(f"Pose tracking exporter initialized: enabled={self.enabled}")

    def export_trajectories(self, video_path, csv_path, output_dir, dataset_name):
        """
        Export trajectory videos and metadata for pose tracking.
        Optimized to read video only once and cache needed frames.

        Args:
            video_path: Path to source video
            csv_path: Path to tracking CSV
            output_dir: Directory to save dataset
            dataset_name: Name for the dataset

        Returns:
            export_path: Path to exported dataset
        """
        import pandas as pd
        from datetime import datetime
        from collections import defaultdict

        if not self.enabled:
            logger.info("Pose tracking export is disabled")
            return None

        logger.info(f"Starting optimized pose tracking dataset export from {csv_path}")

        # Read tracking data
        df = pd.read_csv(csv_path)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_with_timestamp = f"{dataset_name}_pose_{timestamp}"
        export_dir = Path(output_dir) / dataset_name_with_timestamp
        export_dir.mkdir(parents=True, exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Filter trajectories by length
        trajectory_ids = df["TrajectoryID"].unique()
        valid_trajectory_ids = [
            traj_id
            for traj_id in trajectory_ids
            if len(df[df["TrajectoryID"] == traj_id]) >= self.min_trajectory_length
        ]
        total_trajectories = len(valid_trajectory_ids)

        if total_trajectories == 0:
            logger.warning("No valid trajectories to export")
            cap.release()
            return None

        logger.info(f"Exporting {total_trajectories} trajectories")

        # Step 1: Build frame-to-trajectories mapping (which trajectories need which frames)
        if self.progress_callback:
            self.progress_callback(0, "Building frame index...")

        frame_to_trajs = defaultdict(list)  # frame_id -> [(traj_id, traj_idx), ...]
        traj_data_dict = {}  # traj_id -> DataFrame

        for traj_id in valid_trajectory_ids:
            traj_data = df[df["TrajectoryID"] == traj_id].sort_values("FrameID")
            traj_data_dict[traj_id] = traj_data

            for idx, row in traj_data.iterrows():
                frame_id = int(row["FrameID"])
                traj_idx = traj_data.index.get_loc(idx)
                frame_to_trajs[frame_id].append((traj_id, traj_idx))

        unique_frames = sorted(frame_to_trajs.keys())
        logger.info(f"Need to read {len(unique_frames)} unique frames from video")

        # Step 2: Read video ONCE and extract all needed frames
        if self.progress_callback:
            self.progress_callback(
                5, f"Reading {len(unique_frames)} frames from video..."
            )

        frame_cache = {}  # frame_id -> frame (BGR image)
        current_frame_idx = 0
        frames_read = 0

        for frame_id in unique_frames:
            # Skip frames we don't need using grab (fast, no decode)
            while current_frame_idx < frame_id:
                ret = cap.grab()
                if not ret:
                    # Video ended prematurely, can't read any more frames
                    logger.warning(
                        f"Video ended at frame {current_frame_idx}, cannot read frame {frame_id}"
                    )
                    break
                current_frame_idx += 1

            # Read the frame we need using read() (grab + retrieve)
            if current_frame_idx == frame_id:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    frame_cache[frame_id] = frame.copy()
                    frames_read += 1
                else:
                    # Frame read failed - log it (shouldn't happen often)
                    if logger.isEnabledFor(logging.DEBUG):
                        if not ret:
                            logger.debug(f"Failed to read frame {frame_id}")
                        elif frame is None or frame.size == 0:
                            logger.debug(f"Frame {frame_id} is empty/invalid")
                current_frame_idx += 1
            else:
                # We couldn't reach this frame (video ended early)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Could not reach frame {frame_id} (current position: {current_frame_idx})"
                    )

            # Progress update every 100 frames
            if frames_read % 100 == 0 and self.progress_callback:
                read_pct = int((frames_read / len(unique_frames)) * 30)  # 5-35%
                self.progress_callback(
                    5 + read_pct, f"Read {frames_read}/{len(unique_frames)} frames..."
                )

        cap.release()
        logger.info(f"Successfully read {frames_read} frames into cache")
        if frames_read < len(unique_frames):
            logger.warning(
                f"Only read {frames_read}/{len(unique_frames)} frames - some frames may be missing"
            )

        # Step 3: Generate trajectory videos from cached frames
        if self.progress_callback:
            self.progress_callback(35, "Generating trajectory videos...")

        exported_count = 0
        metadata = {
            "dataset_name": dataset_name_with_timestamp,
            "source_video": str(video_path),
            "source_csv": str(csv_path),
            "fps": fps,
            "trajectories": [],
        }

        for idx, traj_id in enumerate(valid_trajectory_ids):
            traj_data = traj_data_dict[traj_id]

            # Export this trajectory using cached frames
            traj_info = self._export_single_trajectory_from_cache(
                frame_cache, traj_data, traj_id, export_dir, frame_width, frame_height
            )

            if traj_info:
                metadata["trajectories"].append(traj_info)
                exported_count += 1

            # Report progress (35-95%)
            if self.progress_callback:
                progress_pct = 35 + int(((idx + 1) / total_trajectories) * 60)
                self.progress_callback(
                    progress_pct,
                    f"Exported trajectory {idx + 1}/{total_trajectories} (ID: {traj_id})",
                )

        # Clear frame cache to free memory
        frame_cache.clear()

        # Save metadata
        if self.progress_callback:
            self.progress_callback(95, "Saving metadata...")

        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create README
        readme_path = export_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(f"# {dataset_name_with_timestamp}\n\n")
            f.write("Pose tracking dataset generated from multi-animal tracking.\n\n")
            f.write(f"## Contents\n\n")
            f.write(f"- **{exported_count} trajectory videos** (one per animal)\n")
            f.write(
                f"- **metadata.json**: Trajectory information and frame mappings\n\n"
            )
            f.write("## Next Steps\n\n")
            f.write(
                "1. Use pose tracking tools (DeepLabCut, SLEAP, etc.) to label keypoints\n"
            )
            f.write("2. Train pose estimation models on cropped trajectory videos\n")
            f.write(
                "3. Apply trained models back to full videos for complete analysis\n"
            )

        if self.progress_callback:
            self.progress_callback(100, "Export complete!")

        logger.info(f"Exported {exported_count} trajectories to {export_dir}")
        return str(export_dir)

    def _export_single_trajectory_from_cache(
        self, frame_cache, traj_data, traj_id, output_dir, frame_width, frame_height
    ):
        """Export a single trajectory as a video file using pre-loaded frames from cache."""
        # Get trajectory bounds
        frames = traj_data["FrameID"].values
        xs = traj_data["X"].values
        ys = traj_data["Y"].values

        # Skip if all NaN
        valid_mask = ~(np.isnan(xs) | np.isnan(ys))
        if not valid_mask.any():
            logger.warning(f"Trajectory {traj_id}: All coordinates are NaN, skipping")
            return None

        # Check coordinate ranges (debug mode only)
        if logger.isEnabledFor(logging.DEBUG):
            valid_xs = xs[valid_mask]
            valid_ys = ys[valid_mask]
            logger.debug(
                f"Trajectory {traj_id}: X range=[{valid_xs.min():.1f}, {valid_xs.max():.1f}], "
                f"Y range=[{valid_ys.min():.1f}, {valid_ys.max():.1f}], "
                f"Frame size=({frame_width}, {frame_height})"
            )

            # Warn if coordinates are outside frame bounds
            if (
                valid_xs.min() < 0
                or valid_xs.max() >= frame_width
                or valid_ys.min() < 0
                or valid_ys.max() >= frame_height
            ):
                logger.warning(
                    f"Trajectory {traj_id}: Some coordinates are outside frame bounds! "
                    f"This suggests a coordinate scaling mismatch."
                )

        # Calculate crop size
        body_size = self.params.get("REFERENCE_BODY_SIZE", 20.0)
        crop_size = int(body_size * self.crop_size_multiplier)
        crop_size = max(64, min(crop_size, 256))
        if crop_size % 2 != 0:
            crop_size += 1

        # Create video writer with better codec
        video_filename = f"trajectory_{traj_id:04d}.mp4"
        video_path = output_dir / video_filename

        # Use x264 codec if available, otherwise fall back to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
        out = cv2.VideoWriter(
            str(video_path), fourcc, self.export_fps, (crop_size, crop_size)
        )

        # Fallback to mp4v if avc1 fails
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                str(video_path), fourcc, self.export_fps, (crop_size, crop_size)
            )

        frame_info = []
        half_size = crop_size // 2

        for i, frame_id in enumerate(frames):
            frame_id = int(frame_id)
            cx, cy = xs[i], ys[i]

            # Handle NaN (occluded frames)
            if np.isnan(cx) or np.isnan(cy):
                # Write blank frame
                blank = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                out.write(blank)
                frame_info.append(
                    {"frame_id": frame_id, "x": None, "y": None, "occluded": True}
                )
            else:
                # Get frame from cache
                frame = frame_cache.get(frame_id)
                if frame is None:
                    # Frame not in cache (shouldn't happen), write blank
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Trajectory {traj_id}, frame {frame_id} not in cache"
                        )
                    blank = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    out.write(blank)
                    frame_info.append(
                        {"frame_id": frame_id, "x": None, "y": None, "occluded": True}
                    )
                    continue

                # Extract crop
                x1 = max(0, int(cx - half_size))
                y1 = max(0, int(cy - half_size))
                x2 = min(frame_width, int(cx + half_size))
                y2 = min(frame_height, int(cy + half_size))

                crop = frame[y1:y2, x1:x2].copy()

                # Check if crop is valid
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Trajectory {traj_id}, frame {frame_id}: Empty crop at ({cx:.1f}, {cy:.1f}), "
                            f"bounds=[{x1}:{x2}, {y1}:{y2}], frame_size=({frame_width}, {frame_height})"
                        )
                    blank = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                    out.write(blank)
                    frame_info.append(
                        {
                            "frame_id": frame_id,
                            "x": float(cx),
                            "y": float(cy),
                            "occluded": True,
                        }
                    )
                    continue

                # Pad if needed
                actual_h, actual_w = crop.shape[:2]
                if actual_h < crop_size or actual_w < crop_size:
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

                out.write(crop)
                frame_info.append(
                    {
                        "frame_id": frame_id,
                        "x": float(cx),
                        "y": float(cy),
                        "occluded": False,
                    }
                )

        out.release()

        return {
            "trajectory_id": int(traj_id),
            "video_file": video_filename,
            "num_frames": len(frames),
            "frames": frame_info,
        }

    def _export_single_trajectory(
        self, cap, traj_data, traj_id, output_dir, frame_width, frame_height
    ):
        """
        DEPRECATED: Legacy method using slow frame seeking.
        Kept for compatibility but should not be used.
        Use _export_single_trajectory_from_cache instead.
        """
        # Get trajectory bounds
        frames = traj_data["FrameID"].values
        xs = traj_data["X"].values
        ys = traj_data["Y"].values

        # Skip if all NaN
        valid_mask = ~(np.isnan(xs) | np.isnan(ys))
        if not valid_mask.any():
            return None

        # Calculate crop size
        body_size = self.params.get("REFERENCE_BODY_SIZE", 20.0)
        crop_size = int(body_size * self.crop_size_multiplier)
        crop_size = max(64, min(crop_size, 256))
        if crop_size % 2 != 0:
            crop_size += 1

        # Create video writer
        video_filename = f"trajectory_{traj_id:04d}.mp4"
        video_path = output_dir / video_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(video_path), fourcc, self.export_fps, (crop_size, crop_size)
        )

        frame_info = []

        for i, frame_id in enumerate(frames):
            # Read frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            # Get position
            cx, cy = xs[i], ys[i]

            # Handle NaN (occluded frames)
            if np.isnan(cx) or np.isnan(cy):
                # Write blank frame
                blank = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                out.write(blank)
                frame_info.append(
                    {"frame_id": int(frame_id), "x": None, "y": None, "occluded": True}
                )
            else:
                # Extract crop
                half_size = crop_size // 2
                x1 = max(0, int(cx - half_size))
                y1 = max(0, int(cy - half_size))
                x2 = min(frame_width, int(cx + half_size))
                y2 = min(frame_height, int(cy + half_size))

                crop = frame[y1:y2, x1:x2].copy()

                # Pad if needed
                actual_h, actual_w = crop.shape[:2]
                if actual_h < crop_size or actual_w < crop_size:
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

                out.write(crop)
                frame_info.append(
                    {
                        "frame_id": int(frame_id),
                        "x": float(cx),
                        "y": float(cy),
                        "occluded": False,
                    }
                )

        out.release()

        return {
            "trajectory_id": int(traj_id),
            "video_file": video_filename,
            "num_frames": len(frames),
            "frames": frame_info,
        }
