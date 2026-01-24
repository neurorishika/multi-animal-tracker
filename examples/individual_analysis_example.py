#!/usr/bin/env python3
"""
Example script demonstrating individual-level analysis features.

This shows how to:
1. Extract crops around detections for identity classification
2. Export trajectory videos for pose tracking training
"""

import cv2
import numpy as np
from multi_tracker.core.individual_analysis import (
    IdentityProcessor,
    PoseTrackingExporter,
)

# Example 1: Identity Classification
print("Example 1: Identity Classification")
print("=" * 50)

# Create identity processor
identity_params = {
    "ENABLE_IDENTITY_ANALYSIS": True,
    "IDENTITY_METHOD": "color_tags",  # or "apriltags" or "none"
    "IDENTITY_CROP_SIZE_MULTIPLIER": 3.0,
    "IDENTITY_CROP_MIN_SIZE": 64,
    "IDENTITY_CROP_MAX_SIZE": 256,
    "REFERENCE_BODY_SIZE": 20.0,
}

processor = IdentityProcessor(identity_params)

# Simulate frame with detections
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Simulate detections (from tracking)
detections = [
    {"cx": 100, "cy": 100, "theta": 0, "body_size": 20, "track_id": 1},
    {"cx": 200, "cy": 150, "theta": 45, "body_size": 20, "track_id": 2},
    {"cx": 300, "cy": 200, "theta": 90, "body_size": 20, "track_id": 3},
]

# Process frame to get identities
identities, confidences, crops = processor.process_frame(frame, detections, frame_id=0)

print(f"Processed {len(detections)} detections")
for i, (det, identity, conf) in enumerate(zip(detections, identities, confidences)):
    print(f"  Track {det['track_id']}: Identity={identity}, Confidence={conf:.2f}")
    print(f"    Crop shape: {crops[i].shape}")

print()

# Example 2: Pose Tracking Export
print("Example 2: Pose Tracking Export")
print("=" * 50)

pose_params = {
    "ENABLE_POSE_EXPORT": True,
    "POSE_CROP_SIZE_MULTIPLIER": 4.0,
    "POSE_MIN_TRAJECTORY_LENGTH": 30,
    "POSE_EXPORT_FPS": 30,
    "REFERENCE_BODY_SIZE": 20.0,
}

exporter = PoseTrackingExporter(pose_params)

# These would be your actual paths after tracking
video_path = "path/to/your/video.mp4"
csv_path = "path/to/tracking_output.csv"
output_dir = "path/to/pose_datasets"
dataset_name = "my_pose_dataset"

print(f"Pose exporter configured:")
print(f"  Crop multiplier: {pose_params['POSE_CROP_SIZE_MULTIPLIER']}x")
print(f"  Min trajectory length: {pose_params['POSE_MIN_TRAJECTORY_LENGTH']} frames")
print(f"  Export FPS: {pose_params['POSE_EXPORT_FPS']}")
print()
print("To export pose dataset after tracking:")
print(f"  exporter.export_trajectories(")
print(f"      '{video_path}',")
print(f"      '{csv_path}',")
print(f"      '{output_dir}',")
print(f"      '{dataset_name}'")
print(f"  )")
print()
print("This will create:")
print("  - trajectory_0001.mp4, trajectory_0002.mp4, ... (one per animal)")
print("  - metadata.json (frame mappings and trajectory info)")
print("  - README.md (usage instructions)")

# Example 3: Custom Identity Classifier
print()
print("Example 3: Custom Identity Classifier")
print("=" * 50)
print(
    """
To implement your own identity classifier:

1. Edit src/multi_tracker/core/individual_analysis.py
2. Override the _classify_identity method in IdentityProcessor
3. Example:

    def _classify_identity(self, crop, crop_info, detection):
        # Your custom logic here
        # e.g., color analysis, barcode reading, etc.
        
        # Example: Simple color-based classification
        avg_color = np.mean(crop, axis=(0, 1))
        if avg_color[2] > avg_color[0]:  # More red than blue
            return "red_tagged", 0.9
        elif avg_color[0] > avg_color[2]:  # More blue than red
            return "blue_tagged", 0.9
        else:
            return "unknown", 0.3

4. Select "Custom" in the GUI's Individual Analysis tab
"""
)

print("\n✓ Example script complete!")
print("\nNext steps:")
print("1. Configure individual analysis in the GUI (Individual Analysis tab)")
print("2. Run tracking with identity analysis enabled")
print("3. After tracking, export pose datasets using the button")
print("4. Annotate pose videos in DeepLabCut/SLEAP")
print("5. Train pose models and apply to full videos")
