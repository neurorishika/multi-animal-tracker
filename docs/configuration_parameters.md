# Configuration Parameters Reference

This document provides a comprehensive reference of all configurable parameters in the Multi-Animal Tracker application. All parameters are saved to and loaded from JSON configuration files (e.g., `video_name_config.json`).

---

## Table of Contents

1. [Setup Tab](#setup-tab)
2. [Detection Tab](#detection-tab)
3. [Tracking Tab](#tracking-tab)
4. [Processing Tab](#processing-tab)
5. [Visuals Tab](#visuals-tab)
6. [Dataset Generation Tab](#dataset-generation-tab)
7. [Individual Analysis Tab](#individual-analysis-tab)
8. [Other Parameters](#other-parameters)

---

## Setup Tab

### File Management

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Input Video Path | `file_path` | string | `""` | Path to the input video file |
| CSV Output Path | `csv_path` | string | `""` | Path for trajectory CSV output |
| Video Output Enabled | `video_output_enabled` | bool | `false` | Enable video output with visualization |
| Video Output Path | `video_output_path` | string | `""` | Path for visualization video output |

### Time Reference

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Acquisition Frame Rate (FPS) | `fps` | float | `30.0` | 1.0-240.0 | Acquisition frame rate in frames per second (may differ from video file framerate). Time-dependent parameters (velocities, durations) scale with this value. |

### System Performance

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Processing Resize Factor | `resize_factor` | float | `1.0` | 0.1-1.0 | Downscale video frames for faster processing. 1.0 = full resolution, 0.5 = half resolution (4× faster). |
| Save Confidence Metrics | `save_confidence_metrics` | bool | `true` | - | Save detection, assignment, and position uncertainty metrics to CSV. Adds ~10-20% processing time. |
| Visualization-Free Mode | `visualization_free_mode` | bool | `false` | - | Skip all frame visualization for maximum speed (2-4× speedup). |

---

## Detection Tab

### Spatial Reference

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Reference Body Size | `reference_body_size` | float | `20.0` | 1.0-500.0 | Reference animal body diameter in pixels (at resize=1.0). All distance/size parameters are scaled relative to this value. Should be set before configuring tracking parameters. |

### Detection Strategy

| Parameter | Config Key | Type | Default | Options | Description |
|-----------|------------|------|---------|---------|-------------|
| Detection Method | `detection_method` | string | `"background_subtraction"` | `"background_subtraction"`, `"yolo_obb"` | Method for detecting animals in each frame |

### Size Filtering

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Enable Size Filtering | `enable_size_filtering` | bool | `false` | - | Filter detected objects by area to remove noise and artifacts |
| Min Object Size | `min_object_size_multiplier` | float | `0.3` | 0.1-5.0 | Minimum object area as multiple of reference body area |
| Max Object Size | `max_object_size_multiplier` | float | `3.0` | 0.5-10.0 | Maximum object area as multiple of reference body area |

### Image Enhancement (Pre-processing)

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Brightness | `brightness` | int | `0` | -255 to 255 | Adjust overall image brightness. Positive = lighter, Negative = darker. |
| Contrast | `contrast` | float | `1.0` | 0.0-3.0 | Adjust image contrast. 1.0 = original, >1.0 = more contrast |
| Gamma | `gamma` | float | `1.0` | 0.1-3.0 | Gamma correction. 1.0 = original, >1.0 = brighter mid-tones |
| Dark on Light Background | `dark_on_light_background` | bool | `true` | - | Check if animals are darker than background |

### Background Subtraction Parameters

#### Background Model

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Priming Frames | `background_prime_frames` | int | `10` | 0-5000 | Number of initial frames to build background model |
| Adaptive Background | `enable_adaptive_background` | bool | `true` | - | Continuously update background model during tracking |
| Background Learning Rate | `background_learning_rate` | float | `0.001` | 0.0001-0.1 | How quickly background adapts to changes |
| Subtraction Threshold | `subtraction_threshold` | int | `50` | 0-255 | Pixel intensity difference to detect foreground |

#### Lighting Stabilization

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Enable Stabilization | `enable_lighting_stabilization` | bool | `true` | - | Compensate for gradual lighting changes over time |
| Smooth Factor | `lighting_smooth_factor` | float | `0.95` | 0.8-0.999 | Temporal smoothing factor for lighting correction |
| Median Window | `lighting_median_window` | int | `5` | 3-15 | Median filter window size (odd number) |

#### Morphology & Noise

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Main Kernel Size | `morph_kernel_size` | int | `5` | 1-25 | Morphological operation kernel size (odd number) |
| Min Contour Area | `min_contour_area` | int | `50` | 0-100000 | Minimum contour area in pixels² to keep |
| Max Contour Multiplier | `max_contour_multiplier` | int | `20` | 5-100 | Maximum contour area as multiplier of minimum |

#### Advanced Separation

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Conservative Splitting | `enable_conservative_split` | bool | `true` | - | Use erosion to separate touching animals |
| Conservative Kernel Size | `conservative_kernel_size` | int | `3` | 1-15 | Erosion kernel size (odd number) |
| Conservative Erode Iterations | `conservative_erode_iterations` | int | `1` | 1-10 | Number of erosion iterations |
| Merge Area Threshold | `merge_area_threshold` | int | `1000` | 100-10000 | Maximum area (px²) of small blobs to merge |
| Additional Dilation | `enable_additional_dilation` | bool | `false` | - | Use dilation to reconnect thin parts |
| Dilation Kernel Size | `dilation_kernel_size` | int | `3` | 1-15 | Dilation kernel size (odd number) |
| Dilation Iterations | `dilation_iterations` | int | `2` | 1-10 | Number of dilation iterations |

### YOLO Configuration

| Parameter | Config Key | Type | Default | Range/Options | Description |
|-----------|------------|------|---------|---------------|-------------|
| YOLO Model Path | `yolo_model_path` | string | `"yolo26s-obb.pt"` | - | Path to YOLO model file |
| YOLO Confidence | `yolo_confidence_threshold` | float | `0.25` | 0.01-1.0 | Minimum confidence score for detections |
| YOLO IOU Threshold | `yolo_iou_threshold` | float | `0.7` | 0.01-1.0 | IOU threshold for non-max suppression |
| YOLO Target Classes | `yolo_target_classes` | list/null | `null` | - | Comma-separated class IDs to detect (null for all) |
| YOLO Device | `yolo_device` | string | `"auto"` | `"auto"`, `"cpu"`, `"cuda:0"`, `"mps"` | Hardware device for YOLO inference |

---

## Tracking Tab

### Core Tracking Parameters

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Max Targets | `max_targets` | int | `4` | 1-200 | Maximum number of animals to track simultaneously |
| Max Assignment Distance | `max_assignment_distance_multiplier` | float | `1.5` | 0.1-20.0 | Maximum distance for track-to-detection assignment (×body size) |
| Recovery Search Distance | `recovery_search_distance_multiplier` | float | `0.5` | 0.1-10.0 | Search radius for recovering lost tracks (×body size) |
| Enable Backward Tracking | `enable_backward_tracking` | bool | `true` | - | Run tracking in reverse after forward pass |

### Kalman Filter (Motion Model)

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Process Noise | `kalman_process_noise` | float | `0.03` | 0.0-1.0 | Process noise covariance for motion prediction |
| Measurement Noise | `kalman_measurement_noise` | float | `0.1` | 0.0-1.0 | Measurement noise covariance |
| Velocity Damping | `kalman_velocity_damping` | float | `0.95` | 0.5-0.99 | Velocity damping coefficient |

#### Age-Dependent Damping

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Maturity Age | `kalman_maturity_age` | int | `5` | 1-30 | Frames for track to reach maturity |
| Initial Velocity Retention | `kalman_initial_velocity_retention` | float | `0.2` | 0.0-1.0 | Velocity retention for new tracks |
| Max Velocity Multiplier | `kalman_max_velocity_multiplier` | float | `2.0` | 0.5-10.0 | Maximum velocity constraint (×body size) |

#### Anisotropic Process Noise

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Longitudinal Multiplier | `kalman_longitudinal_noise_multiplier` | float | `5.0` | 0.1-20.0 | Forward/longitudinal noise multiplier |
| Lateral Multiplier | `kalman_lateral_noise_multiplier` | float | `0.1` | 0.01-5.0 | Sideways/lateral noise multiplier |

### Cost Function Weights

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Position Weight | `weight_position` | float | `1.0` | 0.0-10.0 | Weight for position distance in assignment cost |
| Orientation Weight | `weight_orientation` | float | `1.0` | 0.0-10.0 | Weight for orientation difference |
| Area Weight | `weight_area` | float | `0.001` | 0.0-1.0 | Weight for area difference |
| Aspect Ratio Weight | `weight_aspect_ratio` | float | `0.1` | 0.0-10.0 | Weight for aspect ratio difference |
| Use Mahalanobis Distance | `use_mahalanobis_distance` | bool | `true` | - | Use Mahalanobis instead of Euclidean distance |

### Assignment Algorithm

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Enable Greedy Assignment | `enable_greedy_assignment` | bool | `false` | Use greedy approximation (faster for N>100) instead of Hungarian |
| Enable Spatial Optimization | `enable_spatial_optimization` | bool | `false` | Use KD-tree to reduce comparisons for large N |

### Orientation & Lifecycle

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Velocity Threshold | `velocity_threshold` | float | `5.0` | 0.1-100.0 | Velocity threshold (body-sizes/second) to classify as 'moving' |
| Instant Flip | `enable_instant_flip` | bool | `true` | - | Allow instant 180° orientation flip when moving quickly |
| Max Orient Delta (Stopped) | `max_orientation_delta_stopped` | float | `30.0` | 1-180 | Maximum orientation change (degrees) when stationary |

### Track Lifecycle

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Lost Frames Threshold | `lost_frames_threshold` | int | `10` | 1-100 | Frames without detection before track termination |
| Min Respawn Distance | `min_respawn_distance_multiplier` | float | `2.5` | 0.0-20.0 | Minimum distance from existing tracks to spawn new track (×body size) |

### Initialization Stability

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Min Detections to Start | `min_detections_to_start` | int | `1` | 1-50 | Minimum consecutive detections before starting a new track |
| Min Detect Frames | `min_detect_frames` | int | `10` | 1-500 | Minimum total detection frames to keep a track |
| Min Tracking Frames | `min_track_frames` | int | `10` | 1-500 | Minimum tracking frames to keep a track |

---

## Processing Tab

### Trajectory Post-Processing

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Enable Post-processing | `enable_postprocessing` | bool | `true` | - | Automatically clean trajectories |
| Min Trajectory Length | `min_trajectory_length` | int | `10` | 1-1000 | Remove trajectories shorter than this (frames) |
| Max Velocity Break | `max_velocity_break` | float | `50.0` | 1.0-500.0 | Maximum velocity (body-sizes/second) before breaking trajectory |
| Max Distance Break | `max_distance_break_multiplier` | float | `15.0` | 1.0-50.0 | Maximum distance jump before breaking trajectory (×body size) |
| Max Occlusion Gap | `max_occlusion_gap` | int | `30` | 0-200 | Maximum consecutive occluded frames before splitting trajectory |

### Interpolation

| Parameter | Config Key | Type | Default | Options | Description |
|-----------|------------|------|---------|---------|-------------|
| Interpolation Method | `interpolation_method` | string | `"None"` | `"None"`, `"Linear"`, `"Cubic"`, `"Spline"` | Method for filling gaps in trajectories |
| Max Interpolation Gap | `interpolation_max_gap` | int | `10` | 1-100 | Maximum gap size to interpolate (frames) |

### Cleanup

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Auto-cleanup Temp Files | `cleanup_temp_files` | bool | `true` | Automatically delete temporary files after successful tracking |

### Real-Time Analytics

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Enable Histograms | `enable_histograms` | bool | `false` | - | Collect real-time statistics during tracking |
| Histogram History | `histogram_history_frames` | int | `300` | 50-5000 | Number of frames for rolling statistics |

---

## Visuals Tab

### Common Overlays

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Show Track Markers | `show_track_markers` | bool | `true` | Draw circles around tracked animals |
| Show Orientation Lines | `show_orientation_lines` | bool | `true` | Draw lines showing heading direction |
| Show Trajectory Trails | `show_trajectory_trails` | bool | `true` | Draw recent path history for each track |
| Show ID Labels | `show_id_labels` | bool | `true` | Display unique track IDs on each animal |
| Show State Text | `show_state_text` | bool | `true` | Display tracking state (ACTIVE, PREDICTED, etc.) |
| Show Kalman Uncertainty | `show_kalman_uncertainty` | bool | `false` | Draw ellipses showing Kalman filter position uncertainty |

### Background Subtraction Overlays

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Show Foreground Mask | `show_foreground_mask` | bool | `true` | Display the foreground detection mask |
| Show Background Model | `show_background_model` | bool | `true` | Display the learned background model |

### YOLO Overlays

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Show YOLO OBB | `show_yolo_obb` | bool | `false` | Show oriented bounding boxes from YOLO detection |

### Display Settings

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Trail History | `trajectory_history_seconds` | int | `5` | 1-60 | Length of trajectory trails to display (seconds) |

### Debug

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Debug Logging | `debug_logging` | bool | `false` | Enable verbose debug logging |

---

## Dataset Generation Tab

### Active Learning Dataset Generation

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Enable Dataset Generation | `enable_dataset_generation` | bool | `false` | Enable automatic generation of training dataset |

### Dataset Configuration

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Dataset Name | `dataset_name` | string | `""` | Name for the dataset (used for folder/zip naming) |
| Class Name | `dataset_class_name` | string | `"object"` | Name of the object class being tracked |
| Output Directory | `dataset_output_dir` | string | `""` | Directory where the dataset will be saved |

### Frame Selection Criteria

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Max Frames to Export | `dataset_max_frames` | int | `100` | 10-1000 | Maximum number of frames to export |
| Confidence Threshold | `dataset_confidence_threshold` | float | `0.5` | 0.0-1.0 | Flag frames where YOLO confidence is below this |
| Diversity Window | `dataset_diversity_window` | int | `30` | 10-500 | Minimum frame separation for visual diversity |
| Include Context Frames | `dataset_include_context` | bool | `true` | Export ±1 frames around selected frames |
| Probabilistic Sampling | `dataset_probabilistic_sampling` | bool | `true` | Use rank-based probabilistic sampling |

### Quality Metrics

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Low Confidence | `metric_low_confidence` | bool | `true` | Flag frames with low YOLO confidence |
| Count Mismatch | `metric_count_mismatch` | bool | `true` | Flag frames with detection count mismatch |
| High Assignment Cost | `metric_high_assignment_cost` | bool | `true` | Flag frames with high tracker assignment cost |
| Track Loss | `metric_track_loss` | bool | `true` | Flag frames with frequent track losses |
| High Uncertainty | `metric_high_uncertainty` | bool | `false` | Flag frames with high Kalman uncertainty |

---

## Individual Analysis Tab

### Identity Classification

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| Enable Identity Analysis | `enable_identity_analysis` | bool | `false` | Enable real-time identity classification |
| Identity Method | `identity_method` | string | `"none_disabled"` | Classification method: `"none_disabled"`, `"color_tags_yolo"`, `"apriltags"`, `"custom"` |

### Crop Parameters

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Crop Size Multiplier | `identity_crop_size_multiplier` | float | `3.0` | 1.0-10.0 | Crop size = body_size × multiplier |
| Crop Min Size | `identity_crop_min_size` | int | `64` | 32-512 | Minimum crop size in pixels |
| Crop Max Size | `identity_crop_max_size` | int | `256` | 64-1024 | Maximum crop size in pixels |

### Color Tags (YOLO)

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Color Tag Model Path | `color_tag_model_path` | string | `""` | - | Path to color tag YOLO model |
| Color Tag Confidence | `color_tag_confidence` | float | `0.5` | 0.01-1.0 | Minimum confidence for color tag detection |

### AprilTags

| Parameter | Config Key | Type | Default | Options | Description |
|-----------|------------|------|---------|---------|-------------|
| AprilTag Family | `apriltag_family` | string | `"tag36h11"` | `"tag36h11"`, `"tag25h9"`, `"tag16h5"`, `"tagCircle21h7"`, `"tagStandard41h12"` | AprilTag family to detect |
| AprilTag Decimate | `apriltag_decimate` | float | `1.0` | 1.0-4.0 | Decimation factor for faster detection |

---

## Other Parameters

### Display

| Parameter | Config Key | Type | Default | Range | Description |
|-----------|------------|------|---------|-------|-------------|
| Zoom Factor | `zoom_factor` | float | `1.0` | 0.1-5.0 | Display zoom level |

### Region of Interest (ROI)

| Parameter | Config Key | Type | Default | Description |
|-----------|------------|------|---------|-------------|
| ROI Shapes | `roi_shapes` | list | `[]` | List of ROI shape definitions. Each shape is a dict with `type` ("circle" or "polygon"), `params`, and `mode` ("include" or "exclude") |

---

## Configuration File Format

Configuration files are stored as JSON with the following structure:

```json
{
  "file_path": "/path/to/video.mp4",
  "csv_path": "/path/to/output.csv",
  "fps": 30.0,
  "reference_body_size": 20.0,
  "detection_method": "background_subtraction",
  "max_targets": 4,
  ...
  "roi_shapes": [
    {
      "type": "circle",
      "params": {"center": [500, 500], "radius": 400},
      "mode": "include"
    }
  ]
}
```

### Auto-Save Behavior

- When a video is loaded, the tracker automatically looks for a config file named `{video_name}_config.json` in the same directory
- When saving, the default location is the same video directory with the video name prefix
- This allows portable tracking settings that travel with the video file

---

## Parameter Categories Summary

| Category | Parameter Count | Description |
|----------|-----------------|-------------|
| Setup | 9 | File paths, FPS, body size, performance |
| Detection | 31 | Detection method, size filtering, image processing, background subtraction, lighting, morphology, YOLO |
| Tracking | 27 | Targets, distances, Kalman filter, weights, assignment, orientation, lifecycle |
| Processing | 10 | Post-processing, interpolation, histograms |
| Visuals | 11 | Overlays, display settings, debug |
| Dataset Generation | 20 | Active learning, pose export |
| Individual Analysis | 9 | Identity classification, crop settings |
| Other | 2 | Zoom, ROI shapes |
| **Total** | **119** | All configurable parameters |

---

## Version History

- **v1.0** (January 2026): Initial comprehensive parameter documentation
