# YOLO OBB Detection Guide

This guide explains how to use YOLO OBB (Oriented Bounding Box) detection as an alternative to background subtraction for animal detection in the Multi-Animal Tracker.

## Overview

The Multi-Animal Tracker now supports two detection methods:
1. **Background Subtraction** (default): Uses adaptive background modeling to detect moving objects
2. **YOLO OBB**: Uses a pretrained YOLO model with oriented bounding boxes for object detection

**NEW**: YOLO detection is now fully integrated into the GUI! You can easily switch between detection methods and configure YOLO parameters directly in the interface.

## Installation

To use YOLO detection, you need to install the `ultralytics` package:

### Using Conda (Recommended)

If you're using the conda environment, ultralytics is already included in the `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate multi-animal-tracker
```

### Manual Installation

If you need to install ultralytics separately:

```bash
pip install ultralytics
```

This will also install PyTorch and other required dependencies.

## Using YOLO Detection in the GUI

### Quick Start

1. **Launch the application:**
   ```bash
   multianimaltracker
   ```

2. **Select YOLO detection:**
   - In the "Detection Method" section, select **"YOLO OBB"** from the dropdown
   - The YOLO parameters panel will appear

3. **Choose a model:**
   - Select from pretrained models (YOLOv8 or YOLOv11)
   - **Recommended for beginners:** "yolov8s-obb.pt (Small - Balanced)"
   - Or select "Custom Model..." to use your own trained model

4. **Adjust confidence (optional):**
   - Default: 0.25 (good starting point)
   - Lower values: More detections
   - Higher values: Fewer, more confident detections

5. **Start tracking:**
   - Select your video file
   - Choose CSV output path
   - Click "Start Tracking"

### Available Models in GUI

The GUI provides easy access to YOLO26 (latest - January 2026) and YOLO11 models:

**YOLO26 OBB Models (Latest - Recommended):**
- yolo26n-obb.pt (Nano - Fastest, 43% faster CPU)
- yolo26s-obb.pt (Small - Balanced) ⭐ **Recommended**
- yolo26m-obb.pt (Medium)
- yolo26l-obb.pt (Large)
- yolo26x-obb.pt (Extra Large)

**YOLO11 OBB Models:**
- yolov11n-obb.pt (YOLO11 Nano)
- yolov11s-obb.pt (YOLO11 Small)
- yolov11m-obb.pt (YOLO11 Medium)
- yolov11l-obb.pt (YOLO11 Large)
- yolov11x-obb.pt (YOLO11 Extra Large)

**Custom Model:**
- Select "Custom Model..." to browse for your own trained model file

**Auto-Download:** All pretrained models are automatically downloaded on first use (6-50MB depending on model size). Downloaded models are cached locally for future use.

### GUI Parameters

**Detection Method:** Choose between Background Subtraction or YOLO OBB

**YOLO Model:** Select from dropdown or use custom model

**Confidence Threshold:** Minimum confidence for detections (0.01 - 1.0)

**IOU Threshold:** Overlap threshold for NMS (0.01 - 1.0)

**Target Classes:** Optional comma-separated COCO class IDs (e.g., "15,16,17" for cat, dog, horse)

## Configuration File Method

For advanced users or batch processing, you can still configure YOLO detection via JSON:

### Basic Setup

To enable YOLO detection, modify your `tracking_config.json`:

```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "yolov8s-obb.pt",
  "yolo_confidence_threshold": 0.25,
  "yolo_iou_threshold": 0.7,
  "yolo_target_classes": null,
  ...
}
```

## Configuration

#### `detection_method`
- **Type**: string
- **Options**: `"background_subtraction"` or `"yolo_obb"`
- **Default**: `"background_subtraction"`
- **Description**: Selects which detection method to use

#### `yolo_model_path`
- **Type**: string
- **Default**: `"yolov8n-obb.pt"`
- **Description**: Path to the YOLO OBB model file
  - Can be a local path: `"/path/to/custom_model.pt"`
  - Can be a pretrained model name: `"yolov8n-obb.pt"`, `"yolov8s-obb.pt"`, `"yolov8m-obb.pt"`, etc.
  - On first use, pretrained models will be automatically downloaded

#### `yolo_confidence_threshold`
- **Type**: float (0.0 - 1.0)
- **Default**: `0.25`
- **Description**: Minimum confidence score for detections
  - Lower values: More detections, but potentially more false positives
  - Higher values: Fewer detections, but higher quality

#### `yolo_iou_threshold`
- **Type**: float (0.0 - 1.0)
- **Default**: `0.7`
- **Description**: Intersection-over-Union threshold for Non-Maximum Suppression (NMS)
  - Lower values: More aggressive suppression of overlapping boxes
  - Higher values: Allow more overlapping detections

#### `yolo_target_classes`
- **Type**: list of integers or null
- **Default**: `null`
- **Description**: Specific COCO classes to detect (null = all classes)
  - Example: `[0]` for person only
  - Example: `[15, 16, 17]` for bird, cat, dog
  - See [COCO classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) for class IDs

## Available YOLO Models

### YOLOv8 OBB Models
- `yolov8n-obb.pt`: Nano - Fastest, lowest accuracy
- `yolov8s-obb.pt`: Small - Good balance
- `yolov8m-obb.pt`: Medium - Better accuracy
- `yolov8l-obb.pt`: Large - High accuracy
- `yolov8x-obb.pt`: Extra Large - Highest accuracy, slowest

### YOLOv11 OBB Models (Latest)
- `yolov11n-obb.pt`: Nano
- `yolov11s-obb.pt`: Small
- `yolov11m-obb.pt`: Medium
- `yolov11l-obb.pt`: Large
- `yolov11x-obb.pt`: Extra Large

**Recommendation**: Start with `yolov8n-obb.pt` or `yolov8s-obb.pt` for real-time tracking. Use larger models if accuracy is more important than speed.

## Using Custom YOLO Models

You can train a custom YOLO OBB model on your specific animals:

### 1. Prepare Your Dataset

Create a dataset in YOLO OBB format with oriented bounding box annotations.

### 2. Train Your Model

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolov8n-obb.pt')

# Train on your custom dataset
results = model.train(
    data='path/to/your/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 3. Use Your Custom Model

Update the config to use your trained model:

```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "/path/to/your/best.pt",
  "yolo_confidence_threshold": 0.4,
  ...
}
```

## Comparison: YOLO vs Background Subtraction

### Background Subtraction
**Pros:**
- No GPU required
- Works well with stationary camera and moving animals
- No training data needed
- Fast processing

**Cons:**
- Sensitive to lighting changes
- Struggles with stationary animals
- Requires good contrast between animals and background
- Can fail with camera movement

### YOLO OBB
**Pros:**
- Detects stationary and moving animals
- Robust to lighting changes
- Can work with camera movement (if trained appropriately)
- Provides oriented bounding boxes (better for elongated animals)

**Cons:**
- Requires GPU for optimal performance
- May need custom training for specific animals
- Slower than background subtraction on CPU
- Requires more memory

## Performance Tips

### For Real-Time Processing

1. Use a smaller model: `yolov8n-obb.pt`
2. Reduce input resolution: `"resize_factor": 0.5`
3. Increase confidence threshold: `"yolo_confidence_threshold": 0.4`
4. Use GPU if available

### For Maximum Accuracy

1. Use a larger model: `yolov8l-obb.pt` or train custom
2. Lower confidence threshold: `"yolo_confidence_threshold": 0.15`
3. Use full resolution: `"resize_factor": 1.0`
4. Filter by specific classes if applicable

## Troubleshooting

### "ultralytics package not found"
Install ultralytics: `pip install ultralytics`

### Slow detection speed
- Use a smaller model (e.g., `yolov8n-obb.pt`)
- Reduce frame resolution with `resize_factor`
- Ensure GPU is available and CUDA is installed

### Poor detection quality
- Try a larger model
- Lower the confidence threshold
- Train a custom model on your specific animals
- Check that target_classes includes the objects you want to detect

### Model download fails
- Check internet connection
- Manually download model from [Ultralytics releases](https://github.com/ultralytics/assets/releases)
- Place in a local directory and update `yolo_model_path`

## Example Configurations

### General Animals (COCO classes)
```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "yolov8s-obb.pt",
  "yolo_confidence_threshold": 0.25,
  "yolo_iou_threshold": 0.7,
  "yolo_target_classes": null
}
```

### Birds Only
```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "yolov8m-obb.pt",
  "yolo_confidence_threshold": 0.3,
  "yolo_iou_threshold": 0.6,
  "yolo_target_classes": [14]
}
```

### Custom Trained Model
```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "/path/to/my_custom_model.pt",
  "yolo_confidence_threshold": 0.35,
  "yolo_iou_threshold": 0.7,
  "yolo_target_classes": [0]
}
```

## GPU Support

YOLO detection will automatically use GPU if available. To check:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

For Mac with Apple Silicon:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

## Further Resources

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLOv8 OBB Documentation](https://docs.ultralytics.com/tasks/obb/)
- [YOLO Training Guide](https://docs.ultralytics.com/modes/train/)
- [Model Zoo](https://github.com/ultralytics/ultralytics)
