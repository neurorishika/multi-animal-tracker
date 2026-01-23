# Confidence Tracking System

## Overview

The multi-animal tracker includes optional comprehensive confidence metrics for post-hoc quality control and data filtering. Three types of confidence scores can be calculated and saved for each tracked detection:

1. **Detection Confidence** (0-1 or NaN): Quality of the object detection itself
2. **Assignment Confidence** (0-1): Quality of the track-to-detection matching
3. **Position Uncertainty**: Kalman filter state uncertainty estimate

**Performance Note:** Confidence tracking adds approximately 5-10% overhead to processing time. You can disable it in the "System Performance" section of the GUI for maximum speed.

## Detection Confidence

Detection confidence quantifies how confident the detector is that a detected object is a valid animal.

### Background Subtraction Method

For traditional background subtraction, detection confidence is set to **NaN** (not a number). This is because:

- Quality metrics are highly context-specific (lighting conditions, camera quality, species morphology)
- Simple geometric measures (solidity, circularity, aspect ratio) don't reliably predict detection quality
- Performance cost of computing detailed shape metrics is not justified

If you need quality filtering for background subtraction, use assignment confidence and position uncertainty instead, or manually review your data.

### YOLO Detection Method

For YOLO-based oriented bounding box detection, the native confidence scores from the neural network are used directly. These represent the model's certainty that the detection contains an animal.

## Assignment Confidence

Assignment confidence quantifies how well a detection matches its assigned track. It's computed using a sigmoid transformation of the assignment cost:

```
confidence = 1.0 / (1.0 + cost / scale)
```

Where:
- **cost**: The cost value from the Hungarian assignment algorithm
- **scale**: Half of `MAX_DISTANCE_THRESHOLD` parameter

Properties:
- Perfect match (cost = 0) → confidence = 1.0
- High cost matches → confidence approaches 0.0
- Provides smooth transition between good and poor matches

## Position Uncertainty

Position uncertainty is derived from the Kalman filter's error covariance matrix. Specifically, it's the trace (sum of diagonal elements) of the 2×2 position covariance submatrix:

```
uncertainty = cov[0,0] + cov[1,1]
```

This represents the sum of variances in x and y position estimates. Higher values indicate:
- Less confident position estimates
- More prediction uncertainty
- Tracks that haven't been updated recently

## CSV Output

All confidence metrics are saved in the trajectory CSV file with the following columns:

| Column | Description | Range | Notes |
|--------|-------------|-------|-------|
| DetectionConfidence | Quality of detection | 0-1 or NaN | NaN for background subtraction, 0 for unmatched tracks |
| AssignmentConfidence | Quality of track assignment | 0-1 | 0 for unmatched/predicted tracks |
| PositionUncertainty | Kalman position uncertainty | 0+ | Higher = more uncertain |

## Usage for Quality Control

### Filtering Low-Quality Detections

```python
import pandas as pd

# Load trajectory data
df = pd.read_csv('trajectories.csv')

# Filter for high-quality tracks (works best with YOLO detection)
# For background subtraction, focus on assignment confidence and uncertainty
high_quality = df[
    (df['AssignmentConfidence'] > 0.8) &
    (df['PositionUncertainty'] < 10.0)
]

# If using YOLO, also filter by detection confidence
if not df['DetectionConfidence'].isna().all():
    high_quality = high_quality[high_quality['DetectionConfidence'] > 0.7]
```

### Identifying Problematic Frames

```python
# Find frames with many low-confidence assignments
# (works for both detection methods)
problematic_frames = df.groupby('Frame').agg({
    'AssignmentConfidence': 'mean'
}).query('AssignmentConfidence < 0.5')
```

### Track Quality Assessment

```python
# Calculate per-track quality scores
track_quality = df.groupby('TrajectoryID').agg({
    'AssignmentConfidence': ['mean', 'std'],
    'PositionUncertainty': ['mean', 'max']
})

# Identify unreliable tracks
unreliable_tracks = track_quality[
    (track_quality[('AssignmentConfidence', 'mean')] < 0.7)
]
```

## Implementation Details

### Code Locations

- **Detection confidence calculation**:
  - [detection.py](../src/multi_tracker/core/detection.py): `ObjectDetector.detect_objects()` (lines 50-110)
  - [detection.py](../src/multi_tracker/core/detection.py): `YOLOOBBDetector.detect_objects()` (lines 200-300)

- **Assignment confidence calculation**:
  - [assignment.py](../src/multi_tracker/core/assignment.py): `compute_assignment_confidence()` (lines 360-396)

- **Position uncertainty extraction**:
  - [kalman_filters.py](../src/multi_tracker/core/kalman_filters.py): `get_position_uncertainties()` (lines 55-73)

- **CSV integration**:
  - [main_window.py](../src/multi_tracker/gui/main_window.py): CSV header definition (lines 3220-3231)
  - [tracking_worker.py](../src/multi_tracker/core/tracking_worker.py): Confidence tracking through pipeline

### Special Cases

- **Unmatched tracks** (occluded/lost): Detection and assignment confidence set to 0.0
- **Predicted positions**: When Kalman filter predicts without measurement update, detection/assignment confidence = 0.0
- **Track initialization**: First frame confidence may be lower until filter stabilizes

## Recommendations

### Detection Confidence Thresholds (YOLO only)
- **Excellent**: > 0.8
- **Good**: 0.6 - 0.8
- **Fair**: 0.4 - 0.6
- **Poor**: < 0.4

*Note: For background subtraction, DetectionConfidence will be NaN.*

### Assignment Confidence Thresholds (All methods)
- **Excellent**: > 0.9
- **Good**: 0.7 - 0.9
- **Fair**: 0.5 - 0.7
- **Poor**: < 0.5

### Position Uncertainty Thresholds
- **Low**: < 5 pixels
- **Medium**: 5 - 20 pixels
- **High**: > 20 pixels

*Note: These thresholds are guidelines and should be adjusted based on your specific tracking scenario, video quality, and animal density.*
