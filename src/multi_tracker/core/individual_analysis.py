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


class PoseTrackingExporter:
    """
    Export trajectory data for post-hoc pose tracking training.
    Creates videos and metadata for each trajectory.
    """

    def __init__(self, params):
        self.params = params
        self.enabled = params.get("ENABLE_POSE_EXPORT", False)
        self.crop_size_multiplier = params.get("POSE_CROP_SIZE_MULTIPLIER", 4.0)
        self.min_trajectory_length = params.get("POSE_MIN_TRAJECTORY_LENGTH", 30)
        self.export_fps = params.get("POSE_EXPORT_FPS", 30)

        logger.info(f"Pose tracking exporter initialized: enabled={self.enabled}")

    def export_trajectories(self, video_path, csv_path, output_dir, dataset_name):
        """
        Export trajectory videos and metadata for pose tracking.

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

        if not self.enabled:
            logger.info("Pose tracking export is disabled")
            return None

        logger.info(f"Starting pose tracking dataset export from {csv_path}")

        # Read tracking data
        df = pd.read_csv(csv_path)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name_with_timestamp = f"{dataset_name}_pose_{timestamp}"
        export_dir = Path(output_dir) / dataset_name_with_timestamp
        export_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process each trajectory
        trajectory_ids = df["TrajectoryID"].unique()
        exported_count = 0
        metadata = {
            "dataset_name": dataset_name_with_timestamp,
            "source_video": str(video_path),
            "source_csv": str(csv_path),
            "fps": fps,
            "trajectories": [],
        }

        for traj_id in trajectory_ids:
            traj_data = df[df["TrajectoryID"] == traj_id].sort_values("FrameID")

            # Skip short trajectories
            if len(traj_data) < self.min_trajectory_length:
                continue

            # Export this trajectory
            traj_info = self._export_single_trajectory(
                cap, traj_data, traj_id, export_dir, frame_width, frame_height
            )

            if traj_info:
                metadata["trajectories"].append(traj_info)
                exported_count += 1

        cap.release()

        # Save metadata
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

        logger.info(f"Exported {exported_count} trajectories to {export_dir}")
        return str(export_dir)

    def _export_single_trajectory(
        self, cap, traj_data, traj_id, output_dir, frame_width, frame_height
    ):
        """Export a single trajectory as a video file."""
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
