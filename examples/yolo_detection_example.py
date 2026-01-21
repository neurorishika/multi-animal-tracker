#!/usr/bin/env python3
"""
Example script demonstrating how to use YOLO OBB detection
in the Multi-Animal Tracker.

This script shows both programmatic and configuration-based approaches.
"""

import json
from pathlib import Path


def create_yolo_config(output_path="yolo_tracking_config.json"):
    """
    Create a sample configuration file for YOLO-based tracking.

    Args:
        output_path: Path where to save the configuration
    """
    config = {
        # Detection method selection
        "detection_method": "yolo_obb",
        # YOLO-specific parameters
        "yolo_model_path": "yolov8s-obb.pt",  # Small model, good balance
        "yolo_confidence_threshold": 0.25,  # Confidence threshold
        "yolo_iou_threshold": 0.7,  # NMS IoU threshold
        "yolo_target_classes": None,  # None = detect all classes
        # Video input/output
        "file_path": "path/to/your/video.mp4",
        "csv_path": "path/to/output.csv",
        # Tracking parameters
        "max_targets": 8,
        "max_dist_thresh": 50,
        "resize_factor": 0.8,  # Reduce for faster processing
        # Size filtering (optional)
        "enable_size_filtering": True,
        "min_object_size": 50,
        "max_object_size": 500,
        # Kalman filter parameters
        "kalman_noise": 0.03,
        "kalman_meas_noise": 0.13,
        # Track management
        "lost_threshold_frames": 20,
        "min_respawn_distance": 500,
        # Post-processing
        "enable_postprocessing": True,
        "min_trajectory_length": 10,
        # Visualization
        "show_trajectories": True,
        "show_labels": True,
        "show_orientation": True,
        "traj_history": 60,
        # Note: Background subtraction parameters are ignored when using YOLO
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"YOLO configuration saved to {output_path}")
    print("\nTo use this configuration:")
    print("1. Update 'file_path' and 'csv_path' with your actual paths")
    print("2. Run: multianimaltracker")
    print("3. Load this config file in the GUI")
    print("\nOr load it programmatically in your script")


def create_custom_model_config(model_path, output_path="custom_yolo_config.json"):
    """
    Create a configuration for a custom-trained YOLO model.

    Args:
        model_path: Path to your custom YOLO model
        output_path: Path where to save the configuration
    """
    config = {
        "detection_method": "yolo_obb",
        "yolo_model_path": str(model_path),
        "yolo_confidence_threshold": 0.35,  # Adjust based on your model
        "yolo_iou_threshold": 0.7,
        "yolo_target_classes": [0],  # Adjust based on your classes
        "file_path": "path/to/your/video.mp4",
        "csv_path": "path/to/output.csv",
        "max_targets": 8,
        "resize_factor": 1.0,  # Full resolution for custom models
        # Adjust other parameters as needed
        "enable_size_filtering": False,  # Model should handle this
        "kalman_noise": 0.03,
        "kalman_meas_noise": 0.13,
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Custom model configuration saved to {output_path}")


def show_yolo_model_options():
    """Display available YOLO model options."""
    print("Available YOLOv8 OBB Models:")
    print("=" * 50)
    print("\nSpeed vs Accuracy Trade-off:")
    print("  yolov8n-obb.pt  - Nano   (Fastest, lowest accuracy)")
    print("  yolov8s-obb.pt  - Small  (Good balance)")
    print("  yolov8m-obb.pt  - Medium (Better accuracy)")
    print("  yolov8l-obb.pt  - Large  (High accuracy)")
    print("  yolov8x-obb.pt  - Extra  (Best accuracy, slowest)")
    print("\nYOLOv11 OBB Models (Latest):")
    print("  yolov11n-obb.pt - Nano")
    print("  yolov11s-obb.pt - Small")
    print("  yolov11m-obb.pt - Medium")
    print("  yolov11l-obb.pt - Large")
    print("  yolov11x-obb.pt - Extra")
    print("\nRecommendation:")
    print("  - For real-time: yolov8n-obb.pt or yolov8s-obb.pt")
    print("  - For accuracy:  yolov8m-obb.pt or yolov8l-obb.pt")
    print("  - Custom animals: Train your own model")


def compare_detection_methods():
    """Show comparison between detection methods."""
    print("\nDetection Method Comparison:")
    print("=" * 70)
    print("\nBackground Subtraction:")
    print("  Pros:")
    print("    ✓ Fast (CPU-friendly)")
    print("    ✓ No training data needed")
    print("    ✓ Good for high-contrast setups")
    print("  Cons:")
    print("    ✗ Sensitive to lighting")
    print("    ✗ Fails with stationary animals")
    print("    ✗ Requires static background")

    print("\nYOLO OBB Detection:")
    print("  Pros:")
    print("    ✓ Detects stationary animals")
    print("    ✓ Robust to lighting changes")
    print("    ✓ Oriented bounding boxes")
    print("    ✓ Can use custom-trained models")
    print("  Cons:")
    print("    ✗ Requires GPU for speed")
    print("    ✗ Slower on CPU")
    print("    ✗ May need custom training")
    print("    ✗ Larger memory footprint")


if __name__ == "__main__":
    print("YOLO OBB Detection for Multi-Animal Tracker")
    print("=" * 70)

    # Show available models
    show_yolo_model_options()

    # Show method comparison
    compare_detection_methods()

    print("\n" + "=" * 70)
    print("Creating example configurations...")
    print()

    # Create sample configurations
    create_yolo_config("example_yolo_config.json")

    print("\nExample: Custom Model Configuration")
    print("-" * 70)
    custom_model = "/path/to/my_trained_model.pt"
    create_custom_model_config(custom_model, "example_custom_yolo_config.json")

    print("\n" + "=" * 70)
    print("Quick Start:")
    print("1. Install ultralytics: pip install ultralytics")
    print("2. Edit example_yolo_config.json with your video path")
    print("3. Run: multianimaltracker")
    print("4. Load the config and start tracking!")
    print("\nFor more details, see docs/yolo_detection_guide.md")
