# Individual-Level Analysis Guide

This guide covers the individual-level analysis features for identity classification and pose tracking in the Multi-Animal-Tracker.

## Overview

Individual-level analysis extends the tracker with two key capabilities:

1. **Real-time Identity Classification** - Assign identities to tracked animals using color tags, AprilTags, or custom methods
2. **Post-hoc Pose Tracking Export** - Generate cropped videos for each trajectory for downstream pose estimation

## Identity Classification

### Concept

During tracking, the system extracts a crop around each detected animal and processes it to assign an identity. This enables:

- Tracking specific individuals across the video
- Analyzing individual behavior patterns
- Training identity-aware models

### Methods

#### None (Disabled)
No identity classification is performed. Use this if you only need anonymous tracking.

#### Color Tags (YOLO)
Detect colored markers on animals using a trained YOLO model.

**Requirements:**
- Trained YOLO model for color tag detection (`.pt` file)
- Animals with visible color markers

**Configuration:**
- **Model File**: Path to YOLO model trained on color tags
- **Confidence**: Minimum detection confidence (0.01-1.0)

**Use Cases:**
- Ants/bees with paint marks
- Mice with ear tags
- Fish with colored implants

#### AprilTags
Detect fiducial markers (AprilTags) attached to animals.

**Requirements:**
- Animals with attached AprilTag markers
- Tags large enough to be visible in crops

**Configuration:**
- **Tag Family**: AprilTag family (tag36h11, tag25h9, etc.)
- **Decimate**: Speed/accuracy trade-off (1.0 = accurate, higher = faster)

**Use Cases:**
- Larger animals that can carry tags
- Controlled lab environments
- High-accuracy requirements

#### Custom
Implement your own identity classifier.

**Implementation:**
1. Edit `src/multi_tracker/core/identity/analysis.py`
2. Override `_classify_identity` in `IdentityProcessor`
3. Process the crop and return identity + confidence

**Example:**
```python
def _classify_identity(self, crop, crop_info, detection):
    # Your custom logic
    avg_color = np.mean(crop, axis=(0, 1))
    
    if avg_color[2] > 150:  # Red dominant
        return "red_individual", 0.9
    elif avg_color[0] > 150:  # Blue dominant
        return "blue_individual", 0.9
    else:
        return "unknown", 0.3
```

### Crop Parameters

**Size Multiplier** (1.0 - 10.0)
- Crop size = body_size × multiplier
- **3.0** (default): Good balance for most cases
- **2.0**: Tight crop focusing on animal
- **5.0**: Include more context/background

**Min Size** (32 - 512 px)
- Minimum crop dimension in pixels
- **64** (default): Prevents too-small crops
- Increase if your classifier needs more resolution

**Max Size** (64 - 1024 px)
- Maximum crop dimension in pixels
- **256** (default): Good for most models
- Increase for high-resolution classifiers

### Workflow

1. **Configure Method**
   - Open GUI → Individual Analysis tab
   - Enable "Enable Individual-Level Analysis"
   - Select identity method
   - Configure method-specific parameters

2. **Adjust Crop Parameters**
   - Set size multiplier based on your animals
   - Adjust min/max sizes if needed

3. **Run Tracking**
   - Start tracking as normal
   - Identity processing happens in real-time
   - Identities are logged (future: saved to CSV)

## Pose Tracking Export

### Concept

After tracking, export cropped videos centered on each individual trajectory. These videos are ideal for:

- Training pose estimation models (DeepLabCut, SLEAP, Anipose, etc.)
- Manual keypoint annotation
- Individual behavior analysis

### Output Format

For each trajectory longer than the minimum length:

```
pose_dataset_20260124_143052/
├── trajectory_0001.mp4    # Video for trajectory 1
├── trajectory_0002.mp4    # Video for trajectory 2
├── ...
├── metadata.json          # Frame mappings and trajectory info
└── README.md              # Usage instructions
```

**metadata.json structure:**
```json
{
  "dataset_name": "pose_dataset_20260124_143052",
  "source_video": "/path/to/original.mp4",
  "source_csv": "/path/to/tracking.csv",
  "fps": 30.0,
  "trajectories": [
    {
      "trajectory_id": 1,
      "video_file": "trajectory_0001.mp4",
      "num_frames": 150,
      "frames": [
        {"frame_id": 0, "x": 100.5, "y": 200.3, "occluded": false},
        {"frame_id": 1, "x": 102.1, "y": 201.8, "occluded": false},
        ...
      ]
    },
    ...
  ]
}
```

### Configuration

**Output Directory**
- Where to save pose datasets
- Separate directory from tracking outputs recommended

**Dataset Name**
- Base name for the dataset
- Timestamp is appended automatically
- Example: `ant_colony_001` → `ant_colony_001_pose_20260124_143052`

**Crop Multiplier** (1.0 - 10.0)
- Similar to identity crop multiplier
- **4.0** (default): Includes full body + margin for keypoints
- Use larger values if animals have long appendages

**Min Length** (5 - 1000 frames)
- Minimum trajectory length to export
- **30** (default): Enough for meaningful training data
- Short tracks are often low-quality or spurious

**Export FPS** (1 - 120)
- Frame rate for exported videos
- **30** (default): Good for most pose tools
- Match your original video FPS for 1:1 correspondence

### Workflow

1. **Complete Tracking**
   - Run full tracking (forward + backward + merge)
   - Ensure CSV output is generated

2. **Configure Export**
   - Open GUI → Individual Analysis tab
   - Enable "Enable Pose Tracking Export"
   - Select output directory
   - Set dataset name and parameters

3. **Export Dataset**
   - Click "Export Pose Dataset" button
   - Wait for processing (logged to console)
   - Check output directory for trajectory videos

4. **Use in Pose Tools**
   - **DeepLabCut**: Import videos, label frames, train model
   - **SLEAP**: Create project, add videos, annotate, train
   - **Anipose**: Set up multi-view project with trajectory videos

### Best Practices

**For Identity Classification:**
- Test different crop sizes on representative frames
- Ensure markers are visible in crops (use Test Detection)
- Higher confidence thresholds = fewer false positives
- Custom classifiers allow maximum flexibility

**For Pose Tracking:**
- Use larger crop multiplier (4.0+) to capture full body
- Set min length to filter out junk trajectories
- Export FPS should match or be lower than source video
- Use descriptive dataset names for organization

**General Tips:**
- Preview frame helps visualize crop sizes
- Enable identity analysis during tracking to collect data
- Pose export is post-hoc (no performance impact on tracking)
- Clean up old datasets to save disk space

## Integration with Other Tools

### DeepLabCut
```bash
# After exporting pose dataset
deeplabcut.create_new_project('MyProject', 'YourName', 
                               ['/path/to/trajectory_0001.mp4'])
deeplabcut.extract_frames(config_path)
deeplabcut.label_frames(config_path)
deeplabcut.create_training_dataset(config_path)
deeplabcut.train_network(config_path)
```

### SLEAP
```bash
# Import trajectory videos into SLEAP
sleap-label /path/to/pose_dataset_*/trajectory_*.mp4
# Annotate keypoints, train model
# Apply model back to full videos
```

### Custom Analysis
```python
import json
import cv2

# Load metadata
with open('metadata.json') as f:
    meta = json.load(f)

# Process each trajectory
for traj in meta['trajectories']:
    video_path = traj['video_file']
    frames = traj['frames']
    
    # Your analysis here
    cap = cv2.VideoCapture(video_path)
    # ...
```

## Troubleshooting

**Identity classification not working:**
- Check model path is correct
- Verify crop size is appropriate
- Test on individual frames first
- Check logger output for errors

**No crops being generated:**
- Ensure "Enable Individual-Level Analysis" is checked
- Verify body size parameter is set correctly
- Check min/max crop sizes aren't too restrictive

**Pose export button disabled:**
- Complete tracking first (need CSV file)
- Enable pose export checkbox
- Select output directory

**Exported videos are blank:**
- Check trajectory data has valid X, Y coordinates
- Verify source video path is correct
- Look for NaN values in tracking CSV

**Performance issues:**
- Identity analysis adds overhead during tracking
- Use simpler methods (e.g., color tags vs. deep learning)
- Reduce crop size if possible
- Pose export is post-hoc (no tracking slowdown)

## API Reference

### IdentityProcessor

```python
class IdentityProcessor:
    def __init__(self, params):
        """Initialize identity processor with parameters."""
        
    def extract_crop(self, frame, cx, cy, body_size, theta=None):
        """Extract crop around detection.
        
        Returns:
            crop: Cropped image (BGR)
            crop_info: Metadata dict
        """
        
    def process_frame(self, frame, detections, frame_id):
        """Process all detections in frame.
        
        Returns:
            identities: List of identity labels
            confidences: List of confidence scores
            crops: List of crop images
        """
```

### IndividualDatasetGenerator

```python
class IndividualDatasetGenerator:
    def __init__(self, params, video_path, output_dir):
        """Initialize real-time individual dataset generator."""
        
    def process_frame(self, frame, frame_id, detections, track_ids, obb_corners):
        """Process frame and save OBB-masked crops.
        
        Called during forward tracking for each frame with detections.
        Generates crops with only the detected animal visible (OBB mask applied).
        """
        
    def finalize(self):
        """Finalize dataset and save metadata.
        
        Returns:
            output_path: Path to the generated dataset
        """
```

## Future Enhancements

Planned features:
- [ ] Save identities to CSV during tracking
- [ ] Visualize identities in overlay
- [ ] Real-time pose estimation (integrate SLEAP)
- [ ] Multi-camera pose tracking support
- [ ] Automatic keypoint propagation
- [ ] Identity re-identification across occlusions

## Examples

See `examples/individual_analysis_example.py` for code examples.
