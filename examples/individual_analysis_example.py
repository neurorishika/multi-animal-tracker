#!/usr/bin/env python3
"""
Example script demonstrating individual-level analysis features.

This shows how to:
1. Extract crops around detections for identity classification
2. Use real-time individual dataset generation during tracking
"""

import cv2
import numpy as np
from multi_tracker.core.individual_analysis import (
    IdentityProcessor,
    IndividualDatasetGenerator,
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

# Example 2: Real-time Individual Dataset Generation
print("Example 2: Real-time Individual Dataset Generation")
print("=" * 50)

dataset_params = {
    "ENABLE_INDIVIDUAL_DATASET": True,
    "INDIVIDUAL_DATASET_OUTPUT_DIR": "/path/to/output",
    "INDIVIDUAL_OUTPUT_FORMAT": "png",
    "INDIVIDUAL_SAVE_INTERVAL": 1,  # Save every frame
    "INDIVIDUAL_CROP_PADDING": 0.1,  # 10% padding around OBB
}

# Initialize generator
generator = IndividualDatasetGenerator(
    params=dataset_params,
    video_path="/path/to/video.mp4",
    output_dir=dataset_params["INDIVIDUAL_DATASET_OUTPUT_DIR"],
)

print(f"Individual dataset generator configured:")
print(f"  Output format: {dataset_params['INDIVIDUAL_OUTPUT_FORMAT']}")
print(f"  Save interval: every {dataset_params['INDIVIDUAL_SAVE_INTERVAL']} frame(s)")
print(f"  Padding: {dataset_params['INDIVIDUAL_CROP_PADDING'] * 100}%")
print()
print("During tracking, call generator.process_frame() for each frame with:")
print("  - frame: The current video frame")
print("  - frame_id: Frame number")
print("  - detections: List of filtered detections")
print("  - track_ids: Assigned track IDs")
print("  - obb_corners: OBB corner coordinates (4 corners x 2 coords)")
print()
print("This generates OBB-masked crops in real-time during forward tracking.")

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
print("2. Enable 'Real-time Individual Dataset Generation' for YOLO tracking")
print("3. Run tracking - crops are saved automatically during forward pass")
print("4. Use the OBB-masked crops for training identity classifiers or pose models")
