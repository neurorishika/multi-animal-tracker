"""
Dataset generation utilities for active learning.
Identifies challenging frames and exports them for annotation.
"""

import os
import cv2
import json
import zipfile
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FrameQualityScorer:
    """
    Scores frames based on various quality metrics to identify
    challenging frames that would benefit from additional training data.
    """

    def __init__(self, params):
        self.params = params
        self.frame_scores = defaultdict(lambda: {"score": 0.0, "metrics": {}})
        self.max_targets = params.get("MAX_TARGETS", 4)
        self.conf_threshold = params.get("DATASET_CONF_THRESHOLD", 0.5)

        # Enabled metrics
        self.use_confidence = params.get("METRIC_LOW_CONFIDENCE", True)
        self.use_count_mismatch = params.get("METRIC_COUNT_MISMATCH", True)
        self.use_assignment_cost = params.get("METRIC_HIGH_ASSIGNMENT_COST", True)
        self.use_track_loss = params.get("METRIC_TRACK_LOSS", True)
        self.use_uncertainty = params.get("METRIC_HIGH_UNCERTAINTY", False)

    def score_frame(self, frame_id, detection_data=None, tracking_data=None):
        """
        Score a single frame based on enabled quality metrics.

        Args:
            frame_id: Frame number
            detection_data: Dict with detection information
                - confidences: List of detection confidence scores
                - count: Number of detections
            tracking_data: Dict with tracking information
                - assignment_costs: List of assignment costs
                - lost_tracks: Number of lost tracks
                - uncertainties: List of position uncertainties

        Returns:
            score: Higher score = more problematic frame (0.0 to 1.0+)
        """
        score = 0.0
        metrics = {}

        if detection_data is None:
            detection_data = {}
        if tracking_data is None:
            tracking_data = {}

        # Metric 1: Low detection confidence
        if self.use_confidence and "confidences" in detection_data:
            confidences = detection_data["confidences"]
            if confidences:
                # Filter out NaN values (from background subtraction)
                valid_confs = [c for c in confidences if not np.isnan(c)]
                if valid_confs:
                    min_conf = min(valid_confs)
                    avg_conf = np.mean(valid_confs)

                    # Score based on how far below threshold
                    if min_conf < self.conf_threshold:
                        conf_score = (
                            self.conf_threshold - min_conf
                        ) / self.conf_threshold
                        score += conf_score * 0.4  # 40% weight
                        metrics["low_confidence"] = {
                            "min": min_conf,
                            "avg": avg_conf,
                            "score": conf_score,
                        }

        # Metric 2: Detection count mismatch
        if self.use_count_mismatch and "count" in detection_data:
            det_count = detection_data["count"]
            if det_count != self.max_targets:
                # More severe for under-detection than over-detection
                if det_count < self.max_targets:
                    count_score = (self.max_targets - det_count) / self.max_targets
                    score += count_score * 0.3  # 30% weight
                else:
                    count_score = (
                        min((det_count - self.max_targets) / self.max_targets, 1.0)
                        * 0.5
                    )
                    score += count_score * 0.15  # 15% weight for over-detection

                metrics["count_mismatch"] = {
                    "expected": self.max_targets,
                    "actual": det_count,
                    "score": (
                        count_score
                        if det_count < self.max_targets
                        else count_score * 0.5
                    ),
                }

        # Metric 3: High assignment cost
        if self.use_assignment_cost and "assignment_costs" in tracking_data:
            costs = tracking_data["assignment_costs"]
            if costs:
                avg_cost = np.mean(costs)
                max_cost = max(costs)

                # Normalize costs (typical good matches < 10, bad matches > 50)
                cost_score = min(avg_cost / 50.0, 1.0)
                score += cost_score * 0.15  # 15% weight

                metrics["high_assignment_cost"] = {
                    "avg": avg_cost,
                    "max": max_cost,
                    "score": cost_score,
                }

        # Metric 4: Track losses
        if self.use_track_loss and "lost_tracks" in tracking_data:
            lost_count = tracking_data["lost_tracks"]
            if lost_count > 0:
                loss_score = min(lost_count / self.max_targets, 1.0)
                score += loss_score * 0.1  # 10% weight

                metrics["track_loss"] = {"count": lost_count, "score": loss_score}

        # Metric 5: High position uncertainty
        if self.use_uncertainty and "uncertainties" in tracking_data:
            uncertainties = tracking_data["uncertainties"]
            if uncertainties:
                avg_uncertainty = np.mean(uncertainties)
                # Normalize (typical uncertainty < 10, high > 50)
                unc_score = min(avg_uncertainty / 50.0, 1.0)
                score += unc_score * 0.05  # 5% weight

                metrics["high_uncertainty"] = {
                    "avg": avg_uncertainty,
                    "score": unc_score,
                }

        # Store the score
        self.frame_scores[frame_id] = {"score": score, "metrics": metrics}

        return score

    def get_worst_frames(self, max_frames, diversity_window=30, probabilistic=True):
        """
        Select the worst N frames with visual diversity constraint.

        Args:
            max_frames: Maximum number of frames to select
            diversity_window: Minimum frame separation to ensure diversity
            probabilistic: If True, use rank-based probabilistic sampling.
                          If False, use greedy selection (worst frames first).

        Returns:
            selected_frames: List of frame IDs sorted by score (worst first)
        """
        if not self.frame_scores:
            return []

        # Sort frames by score (descending)
        sorted_frames = sorted(
            self.frame_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )

        if probabilistic:
            # Rank-based weighted sampling for better variety
            # Higher rank (worse frame) = higher probability
            candidates = sorted_frames.copy()
            selected = []

            while len(selected) < max_frames and candidates:
                # Calculate rank-based weights for remaining candidates
                # weight = 1 / (rank + 1), normalized
                weights = np.array([1.0 / (i + 1) for i in range(len(candidates))])
                weights = weights / weights.sum()  # Normalize to probabilities

                # Sample one frame
                idx = np.random.choice(len(candidates), p=weights)
                frame_id, data = candidates[idx]

                # Check diversity constraint
                if all(abs(frame_id - sel) >= diversity_window for sel in selected):
                    selected.append(frame_id)

                # Remove this candidate (and nearby frames to speed up)
                # Remove frames within diversity_window to avoid checking them repeatedly
                candidates = [
                    (fid, fdata)
                    for fid, fdata in candidates
                    if abs(fid - frame_id) >= diversity_window
                ]

            logger.info(
                f"Probabilistically selected {len(selected)} frames out of {len(sorted_frames)} "
                f"with diversity window of {diversity_window} frames"
            )
        else:
            # Greedy selection: take worst frames first
            selected = []
            for frame_id, data in sorted_frames:
                if len(selected) >= max_frames:
                    break

                # Check if this frame is far enough from already selected frames
                if all(abs(frame_id - sel) >= diversity_window for sel in selected):
                    selected.append(frame_id)

            logger.info(
                f"Greedily selected {len(selected)} frames out of {len(sorted_frames)} "
                f"with diversity window of {diversity_window} frames"
            )

        return selected

    def get_frame_metadata(self, frame_id):
        """Get stored metadata for a specific frame."""
        return self.frame_scores.get(frame_id, {"score": 0.0, "metrics": {}})


def export_dataset(
    video_path,
    csv_path,
    frame_ids,
    output_dir,
    dataset_name,
    class_name,
    params,
    include_context=True,
    yolo_results_dict=None,
):
    """
    Export selected frames and annotations as a training dataset.

    Args:
        video_path: Path to source video
        csv_path: Path to tracking CSV (for reading annotations)
        frame_ids: List of frame IDs to export
        output_dir: Directory to save dataset
        dataset_name: Name for the dataset
        class_name: Name of the object class (for classes.txt file)
        params: Parameters dict (for accessing RESIZE_FACTOR and REFERENCE_BODY_SIZE)
        include_context: Include ±1 frames around each selected frame
        yolo_results_dict: Optional dict of {frame_id: yolo_detections} for YOLO format export

    Returns:
        zip_path: Path to created zip file
    """
    import pandas as pd
    from datetime import datetime
    import shutil
    from ..core.detection import create_detector

    logger.info(f"Starting dataset export for {len(frame_ids)} frames")

    # Add timestamp to dataset name to avoid overwriting
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name_with_timestamp = f"{dataset_name}_{timestamp}"

    # Create output directory structure
    dataset_dir = Path(output_dir) / dataset_name_with_timestamp
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Initialize YOLO detector to get actual dimensions
    detector = None
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")
    if detection_method == "yolo_obb":
        try:
            detector = create_detector(params)
            logger.info("YOLO detector initialized for dimension extraction")
        except Exception as e:
            logger.warning(
                f"Could not initialize YOLO detector: {e}. Using reference size approximation."
            )
            detector = None

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read tracking CSV for annotations
    df = pd.read_csv(csv_path)

    # Expand frame list with context if requested
    frames_to_export = set()
    for frame_id in frame_ids:
        frames_to_export.add(frame_id)
        if include_context:
            if frame_id > 0:
                frames_to_export.add(frame_id - 1)
            if frame_id < total_frames - 1:
                frames_to_export.add(frame_id + 1)

    frames_to_export = sorted(frames_to_export)
    logger.info(f"Exporting {len(frames_to_export)} frames (including context)")

    # Export frames and create annotations
    exported_count = 0
    metadata = {
        "dataset_name": dataset_name,
        "source_video": str(video_path),
        "source_csv": str(csv_path),
        "total_frames": len(frames_to_export),
        "image_width": frame_width,
        "image_height": frame_height,
        "frames": [],
    }

    # Determine batch size for YOLO processing
    batch_size = 1  # Default to single frame processing
    if (
        detector is not None
        and hasattr(detector, "use_tensorrt")
        and detector.use_tensorrt
    ):
        if hasattr(detector, "tensorrt_batch_size"):
            batch_size = detector.tensorrt_batch_size
            logger.info(f"Using TensorRT batch processing with batch size {batch_size}")

    # Process frames in batches
    frame_batches = []
    for i in range(0, len(frames_to_export), batch_size):
        frame_batches.append(frames_to_export[i : i + batch_size])

    for batch_idx, batch_frame_ids in enumerate(frame_batches):
        # Read all frames in this batch
        batch_frames = []
        batch_frames_original = []
        valid_batch_indices = []
        first_frame_shape = None

        for idx, frame_id in enumerate(batch_frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                batch_frames_original.append((frame_id, frame))

                # Apply resize if needed for detection
                resize_factor = params.get("RESIZE_FACTOR", 1.0)
                if resize_factor != 1.0 and resize_factor > 0:
                    h, w = frame.shape[:2]
                    new_w = int(w * resize_factor)
                    new_h = int(h * resize_factor)

                    # Validate dimensions
                    if new_w > 0 and new_h > 0:
                        frame_for_detection = cv2.resize(frame, (new_w, new_h))
                    else:
                        logger.warning(
                            f"Invalid resize dimensions for frame {frame_id}, using original"
                        )
                        frame_for_detection = frame
                else:
                    frame_for_detection = frame

                # Validate frame shape
                if frame_for_detection.size == 0:
                    logger.warning(f"Frame {frame_id} has zero size, skipping")
                    continue

                # Ensure all frames have the same shape for batching
                if first_frame_shape is None:
                    first_frame_shape = frame_for_detection.shape
                elif frame_for_detection.shape != first_frame_shape:
                    # Resize to match first frame dimensions
                    h, w = first_frame_shape[:2]
                    frame_for_detection = cv2.resize(frame_for_detection, (w, h))
                    logger.debug(
                        f"Resized frame {frame_id} to match batch dimensions: {first_frame_shape}"
                    )

                batch_frames.append(frame_for_detection)
                valid_batch_indices.append(idx)
            else:
                logger.warning(f"Could not read frame {frame_id}, skipping")

        if not batch_frames:
            continue

        # Run YOLO detection on batch
        batch_yolo_detections = [{}] * len(
            batch_frames
        )  # List of {(cx, cy): (w, h, theta)} dicts

        if detector is not None and batch_frames:
            try:
                iou_threshold = params.get("YOLO_IOU_THRESHOLD", 0.7)
                target_classes = params.get("YOLO_TARGET_CLASSES", None)
                max_det = params.get("MAX_TARGETS", 8) * params.get(
                    "MAX_CONTOUR_MULTIPLIER", 20
                )

                try:
                    # Run batched prediction using detector method (handles TensorRT properly)
                    # Pass frames as list, not stacked array - YOLO handles preprocessing
                    results = detector.model.predict(
                        batch_frames,  # List of frames, not stacked array
                        conf=0.1,  # Lower confidence for dataset generation
                        iou=iou_threshold,
                        classes=target_classes,
                        max_det=max_det,
                        device=detector.device,
                        verbose=False,
                    )

                    # Process results for each frame in the batch
                    for result_idx in range(len(batch_frames)):
                        yolo_results = results[result_idx]
                        yolo_detections = {}

                        # Extract actual dimensions from YOLO results
                        if (
                            yolo_results is not None
                            and hasattr(yolo_results, "obb")
                            and yolo_results.obb is not None
                        ):
                            obb_data = yolo_results.obb
                            for i in range(len(obb_data)):
                                # Get center and dimensions from YOLO
                                xywhr = obb_data.xywhr[i].cpu().numpy()
                                cx_det, cy_det, w_det, h_det, angle_rad = xywhr

                                # Scale back to original frame coordinates
                                resize_factor = params.get("RESIZE_FACTOR", 1.0)
                                scale_back = 1.0 / resize_factor
                                cx_det = cx_det * scale_back
                                cy_det = cy_det * scale_back
                                w_det = w_det * scale_back
                                h_det = h_det * scale_back

                                # Ensure major axis is first (matching detection.py logic)
                                if w_det < h_det:
                                    w_det, h_det = h_det, w_det
                                    angle_rad = (np.rad2deg(angle_rad) + 90) % 180
                                    angle_rad = np.deg2rad(angle_rad)

                                # Store detection keyed by position
                                yolo_detections[(cx_det, cy_det)] = (
                                    w_det,
                                    h_det,
                                    angle_rad,
                                )

                        batch_yolo_detections[result_idx] = yolo_detections

                except Exception as e:
                    logger.warning(
                        f"Batched YOLO prediction failed: {e}, falling back to single-frame processing"
                    )
                    # Fallback to single-frame processing for this batch
                    for frame_idx, frame_for_detection in enumerate(batch_frames):
                        try:
                            original_conf = params.get(
                                "YOLO_CONFIDENCE_THRESHOLD", 0.25
                            )
                            params["YOLO_CONFIDENCE_THRESHOLD"] = 0.1
                            try:
                                meas, sizes, shapes, yolo_results, confidences = (
                                    detector.detect_objects(
                                        frame_for_detection,
                                        batch_frame_ids[valid_batch_indices[frame_idx]],
                                    )
                                )
                            finally:
                                params["YOLO_CONFIDENCE_THRESHOLD"] = original_conf

                            # Extract detections from results
                            yolo_detections = {}
                            if (
                                yolo_results is not None
                                and hasattr(yolo_results, "obb")
                                and yolo_results.obb is not None
                            ):
                                obb_data = yolo_results.obb
                                for i in range(len(obb_data)):
                                    xywhr = obb_data.xywhr[i].cpu().numpy()
                                    cx_det, cy_det, w_det, h_det, angle_rad = xywhr
                                    resize_factor = params.get("RESIZE_FACTOR", 1.0)
                                    scale_back = 1.0 / resize_factor
                                    cx_det = cx_det * scale_back
                                    cy_det = cy_det * scale_back
                                    w_det = w_det * scale_back
                                    h_det = h_det * scale_back
                                    if w_det < h_det:
                                        w_det, h_det = h_det, w_det
                                        angle_rad = (np.rad2deg(angle_rad) + 90) % 180
                                        angle_rad = np.deg2rad(angle_rad)
                                    yolo_detections[(cx_det, cy_det)] = (
                                        w_det,
                                        h_det,
                                        angle_rad,
                                    )
                            batch_yolo_detections[frame_idx] = yolo_detections
                        except Exception as inner_e:
                            logger.warning(
                                f"Single-frame fallback also failed for frame {batch_frame_ids[valid_batch_indices[frame_idx]]}: {inner_e}"
                            )
                            batch_yolo_detections[frame_idx] = {}

            except Exception as e:
                logger.error(f"YOLO detection failed for batch {batch_idx}: {e}")
                batch_yolo_detections = [{}] * len(batch_frames)

        # Now process each frame with its detections
        for frame_idx, (frame_id, frame) in enumerate(batch_frames_original):
            yolo_detections = batch_yolo_detections[frame_idx]

            logger.debug(
                f"Frame {frame_id}: Found {len(yolo_detections)} YOLO detections at conf=0.1"
            )

            # Save image
            image_filename = f"frame_{frame_id:06d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), frame)

            # Create YOLO format annotation
            label_filename = f"frame_{frame_id:06d}.txt"
            label_path = labels_dir / label_filename

            # Get detections for this frame from CSV
            frame_detections = df[df["FrameID"] == frame_id]

            # Get resize factor to scale CSV coordinates back to original space
            resize_factor = params.get("RESIZE_FACTOR", 1.0)
            scale_back = 1.0 / resize_factor

            annotations = []
            with open(label_path, "w") as f:
                for _, detection in frame_detections.iterrows():
                    # Skip if position is NaN (occluded)
                    if pd.isna(detection["X"]) or pd.isna(detection["Y"]):
                        continue

                    # CSV coordinates are in resized frame space, scale back to original
                    cx = detection["X"] * scale_back
                    cy = detection["Y"] * scale_back
                    theta = detection["Theta"]

                    # Try to find matching YOLO detection for this tracked object
                    w, h = None, None
                    if yolo_detections:
                        # Find closest YOLO detection to this tracked position
                        # (both are now in original frame space)
                        min_dist = float("inf")
                        matched_detection = None
                        for (cx_det, cy_det), (
                            w_det,
                            h_det,
                            theta_det,
                        ) in yolo_detections.items():
                            dist = np.sqrt((cx - cx_det) ** 2 + (cy - cy_det) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                matched_detection = (w_det, h_det, theta_det)

                        # Use YOLO dimensions if match is close enough (within 50 pixels in original space)
                        if min_dist < 50 and matched_detection is not None:
                            w, h, _ = (
                                matched_detection  # Use YOLO dimensions, keep tracked theta
                            )
                            logger.debug(
                                f"Frame {frame_id}: Matched tracking to YOLO detection (dist={min_dist:.1f})"
                            )

                    # Fallback to reference size if no YOLO match
                    if w is None or h is None:
                        # REFERENCE_BODY_SIZE is in original frame space, use it directly
                        ref_size = params.get("REFERENCE_BODY_SIZE", 20.0)
                        w = ref_size * 2.2  # Width (major axis)
                        h = ref_size * 0.8  # Height (minor axis)
                        logger.debug(
                            f"Frame {frame_id}: Using reference size approximation"
                        )

                    # Normalize to [0, 1]
                    cx_norm = cx / frame_width
                    cy_norm = cy / frame_height
                    w_norm = w / frame_width
                    h_norm = h / frame_height

                    # YOLO OBB format for x-AnyLabeling: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    # Calculate 4 corner points from center, size, and rotation
                    cos_theta = np.cos(theta)
                    sin_theta = np.sin(theta)

                    # Half dimensions
                    hw = w / 2.0
                    hh = h / 2.0

                    # Corner points in local coordinates (centered at origin)
                    corners_local = np.array(
                        [
                            [-hw, -hh],  # top-left
                            [hw, -hh],  # top-right
                            [hw, hh],  # bottom-right
                            [-hw, hh],  # bottom-left
                        ]
                    )

                    # Rotate corners
                    rotation_matrix = np.array(
                        [[cos_theta, -sin_theta], [sin_theta, cos_theta]]
                    )
                    corners_rotated = corners_local @ rotation_matrix.T

                    # Translate to center position
                    corners = corners_rotated + np.array([cx, cy])

                    # Normalize corner points to [0, 1]
                    corners_norm = corners.copy()
                    corners_norm[:, 0] /= frame_width
                    corners_norm[:, 1] /= frame_height

                    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    obb_line = f"0 {corners_norm[0,0]:.6f} {corners_norm[0,1]:.6f} {corners_norm[1,0]:.6f} {corners_norm[1,1]:.6f} {corners_norm[2,0]:.6f} {corners_norm[2,1]:.6f} {corners_norm[3,0]:.6f} {corners_norm[3,1]:.6f}\n"
                    f.write(obb_line)

                    annotations.append(
                        {
                            "track_id": int(detection["TrackID"]),
                            "x": float(cx),  # Now in original frame space
                            "y": float(cy),  # Now in original frame space
                            "theta": float(theta),
                            "state": detection.get("State", "unknown"),
                        }
                    )

            metadata["frames"].append(
                {
                    "frame_id": int(frame_id),
                    "image_file": image_filename,
                    "label_file": label_filename,
                    "annotations": annotations,
                }
            )

            exported_count += 1

    cap.release()

    # Save classes.txt file
    classes_path = dataset_dir / "classes.txt"
    with open(classes_path, "w") as f:
        f.write(f"{class_name}\n")
    logger.info(f"Created classes.txt with class: {class_name}")

    # Save metadata
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create README
    readme_path = dataset_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# {dataset_name}\n\n")
        f.write("This dataset was automatically generated for active learning.\n\n")
        f.write("## Contents\n\n")
        f.write(f"- **images/**: {exported_count} exported frames\n")
        f.write(f"- **labels/**: YOLO OBB format annotations (initial, needs review)\n")
        f.write(f"- **classes.txt**: Object class definition ({class_name})\n")
        f.write(f"- **metadata.json**: Detailed frame and annotation metadata\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. **Review and correct annotations** using x-AnyLabeling:\n")
        f.write("   - Install: `pip install x-anylabeling`\n")
        f.write("   - Load the images/ folder and review/correct the annotations\n")
        f.write("   - x-AnyLabeling supports YOLO OBB format natively\n\n")
        f.write("2. **Train improved YOLO model**:\n")
        f.write("   - Combine this dataset with your existing training data\n")
        f.write("   - Use YOLO training scripts with the corrected annotations\n")
        f.write("   - Update your model in the tracker configuration\n\n")
        f.write("3. **Iterate**:\n")
        f.write("   - Run tracking with the new model\n")
        f.write("   - Generate another dataset if needed\n")
        f.write("   - Repeat until performance is satisfactory\n")

    # Create zip file
    zip_path = Path(output_dir) / f"{dataset_name_with_timestamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(dataset_dir.parent)
                zipf.write(file_path, arcname)

    # Delete the temporary folder after creating the zip
    shutil.rmtree(dataset_dir)
    logger.info(f"Cleaned up temporary directory: {dataset_dir}")

    logger.info(f"Dataset exported successfully to {zip_path}")
    logger.info(f"Exported {exported_count} frames with annotations")

    return str(zip_path)
