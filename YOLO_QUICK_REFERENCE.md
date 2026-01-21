# YOLO Detection Quick Reference

## GUI Method (Recommended)

### Enable YOLO in 3 Steps:

1. **Launch:** `multianimaltracker`
2. **Select:** Choose "YOLO OBB" from Detection Method dropdown
3. **Pick Model:** Select a YOLO model (yolov8s-obb.pt recommended)

That's it! The YOLO parameters panel will appear with sensible defaults.

### Quick Model Selection

Use the dropdown in the GUI:
- **For real-time:** yolo26n-obb.pt (43% faster CPU than previous models!)
- **For balance:** yolo26s-obb.pt or yolov11s-obb.pt ⭐ **Recommended**
- **For accuracy:** yolo26m-obb.pt or yolo26l-obb.pt
- **For custom animals:** Select "Custom Model..." and browse to your .pt file

**Note:** Models auto-download on first use (~6-50MB). Cached locally after download.

### Fine-Tuning in GUI

- **More detections?** Lower the Confidence Threshold (try 0.15)
- **Fewer false positives?** Raise the Confidence Threshold (try 0.40)
- **Specific animals only?** Enter COCO class IDs in Target Classes (e.g., "15,16" for cats and dogs)

## Configuration File Method

Edit `tracking_config.json`:
```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "yolov8s-obb.pt",
  "yolo_confidence_threshold": 0.25
}
```

## Install Dependencies

```bash
pip install ultralytics
```

## Model Selection

| Model | Speed | Accuracy | CPU Performance | Use Case |
|-------|-------|----------|-----------------|----------|
| yolo26n-obb.pt | ⚡⚡⚡⚡⚡ | ⭐⭐ | 43% faster | Real-time, edge devices |
| yolo26s-obb.pt | ⚡⚡⚡⚡ | ⭐⭐⭐ | Excellent | Balanced (recommended) |
| yolo26m-obb.pt | ⚡⚡⚡ | ⭐⭐⭐⭐ | Good | Better accuracy |
| yolo26l-obb.pt | ⚡⚡ | ⭐⭐⭐⭐⭐ | Moderate | High accuracy |
| yolov11s-obb.pt | ⚡⚡⚡⚡ | ⭐⭐⭐ | Good | YOLO11 alternative |

## Key Parameters

```json
{
  "detection_method": "yolo_obb",           // or "background_subtraction"
  "yolo_model_path": "yolov8s-obb.pt",     // model file
  "yolo_confidence_threshold": 0.25,        // 0.0-1.0, lower=more detections
  "yolo_iou_threshold": 0.7,                // 0.0-1.0, for NMS
  "yolo_target_classes": null,              // null=all, [0]=person, [15,16,17]=animals
  "resize_factor": 0.8                      // reduce for speed
}
```

## Common COCO Classes

| Class ID | Animal |
|----------|--------|
| 14 | Bird |
| 15 | Cat |
| 16 | Dog |
| 17 | Horse |
| 18 | Sheep |
| 19 | Cow |

## Switch Between Methods

### Background Subtraction (Default)
```json
{"detection_method": "background_subtraction"}
```

### YOLO Detection
```json
{"detection_method": "yolo_obb"}
```

## Performance Tips

### Faster Processing
- Use smaller model: `yolov8n-obb.pt`
- Reduce resolution: `"resize_factor": 0.5`
- Increase confidence: `"yolo_confidence_threshold": 0.4`

### Better Accuracy
- Use larger model: `yolov8l-obb.pt`
- Full resolution: `"resize_factor": 1.0`
- Lower confidence: `"yolo_confidence_threshold": 0.15`

## Custom Model

```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "/path/to/your/custom_model.pt",
  "yolo_confidence_threshold": 0.35
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "ultralytics not found" | `pip install ultralytics` |
| Slow detection | Use smaller model or reduce resize_factor |
| Too many detections | Increase confidence_threshold |
| Missing detections | Decrease confidence_threshold |
| Model download fails | Download manually from GitHub |

## Example Workflow

1. **Install**: `pip install ultralytics`
2. **Configure**: Set `"detection_method": "yolo_obb"`
3. **Select Model**: Choose appropriate YOLO model
4. **Adjust Confidence**: Fine-tune threshold
5. **Run**: `multianimaltracker`

## More Information

- Full Guide: `docs/yolo_detection_guide.md`
- Example Script: `examples/yolo_detection_example.py`
- README: `README.md`
