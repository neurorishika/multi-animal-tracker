# YOLO Detection GUI Guide

## Visual Walkthrough

### 1. Detection Method Selection

When you open the Multi-Animal-Tracker, you'll see the **Detection Method** section near the top of the control panel:

```
┌─────────────────────────────────────┐
│  Detection Method                   │
├─────────────────────────────────────┤
│  Detection Method: [Background Sub▼]│
│                    [YOLO OBB      ] │
└─────────────────────────────────────┘
```

### 2. Selecting YOLO OBB

Click the dropdown and select "YOLO OBB":

```
Detection Method: [YOLO OBB ▼]
```

### 3. YOLO Parameters Panel Appears

Once YOLO is selected, the YOLO Detection Parameters group will appear:

```
┌─────────────────────────────────────────────────────────┐
│  YOLO Detection Parameters                              │
├─────────────────────────────────────────────────────────┤
│  YOLO Model:          [yolo26s-obb.pt (YOLO26 Small...▼]│
│                       [yolo26n-obb.pt (YOLO26 Nano...) ]│
│                       [yolo26m-obb.pt (YOLO26 Med...) ]│
│                       [yolov11s-obb.pt (YOLO11 Sm...) ]│
│                       [Custom Model...              ]   │
│                                                         │
│  Confidence Threshold: [0.25  ]                        │
│  IOU Threshold:        [0.70  ]                        │
│  Target Classes:       [                            ]  │
│                        (comma-separated, e.g. 15,16)   │
└─────────────────────────────────────────────────────────┘
```

**YOLO26 Highlights:**
- 🚀 **43% faster CPU inference** than previous models
- 🎯 **End-to-end inference** - no post-processing needed
- 📦 **Auto-downloads** on first use
- 💾 **Cached locally** after download

### 4. Selecting a Custom Model

If you choose "Custom Model..." from the dropdown, a file browser button appears:

```
YOLO Model: [Custom Model... ▼]

Custom Model Path: [/path/to/my_model.pt        ] [Browse...]
```

Click "Browse..." to select your custom .pt file.

### 5. Setting Target Classes

To detect only specific animals, enter COCO class IDs in the Target Classes field:

**Example - Detect only cats and dogs:**
```
Target Classes: [15,16]
```

**Example - Detect only birds:**
```
Target Classes: [14]
```

**Leave empty to detect all classes:**
```
Target Classes: [                    ]
```

### 6. Common COCO Class IDs

| Class ID | Animal |
|----------|--------|
| 14 | bird |
| 15 | cat |
| 16 | dog |
| 17 | horse |
| 18 | sheep |
| 19 | cow |
| 20 | elephant |
| 21 | bear |
| 22 | zebra |
| 23 | giraffe |

## Quick Start Examples

### Example 1: Fast Real-Time Tracking (YOLO26)

```
Detection Method:      YOLO OBB
YOLO Model:           yolo26n-obb.pt (YOLO26 Nano - Fastest)
Confidence Threshold: 0.30
IOU Threshold:        0.70
Target Classes:       (empty - all classes)
```

**Why YOLO26n:** 43% faster CPU inference, perfect for real-time edge deployment

### Example 2: High Accuracy Tracking (YOLO26)

```
Detection Method:      YOLO OBB
YOLO Model:           yolo26l-obb.pt (YOLO26 Large)
Confidence Threshold: 0.20
IOU Threshold:        0.60
Target Classes:       (empty - all classes)
```

**Why YOLO26l:** Best accuracy with end-to-end inference

### Example 3: Custom Animal Model

```
Detection Method:      YOLO OBB
YOLO Model:           Custom Model...
Custom Model Path:    /path/to/worm_detector.pt
Confidence Threshold: 0.35
IOU Threshold:        0.70
Target Classes:       0 (your custom class)
```

### Example 4: Detect Only Birds

```
Detection Method:      YOLO OBB
YOLO Model:           yolov8s-obb.pt (Small - Balanced)
Confidence Threshold: 0.25
IOU Threshold:        0.70
Target Classes:       14
```

## Tips for Best Results

### Confidence Threshold Tuning

**Too Many False Detections?**
- Increase confidence threshold (try 0.35 or 0.40)
- Example: Background objects being detected as animals

**Missing Detections?**
- Decrease confidence threshold (try 0.15 or 0.20)
- Example: Small or partially occluded animals not detected

### Model Selection Guide

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| Real-time tracking on CPU | yolo26n-obb.pt | 43% faster CPU, optimized for edge |
| Balanced speed/accuracy | yolo26s-obb.pt | Best all-around YOLO26 performance |
| Offline analysis, GPU available | yolo26l-obb.pt | High accuracy, end-to-end |
| Latest YOLO11 features | yolov11s-obb.pt | YOLO11 alternative |
| Specific animals (not in COCO) | Custom Model | Train on your data |

### Performance Optimization

**For Faster Processing:**
1. Use a smaller model (yolov8n-obb.pt)
2. Increase confidence threshold
3. Set Resize Factor to 0.5 or 0.75 in Core Tracking settings
4. Use GPU if available

**For Better Accuracy:**
1. Use a larger model (yolov8m-obb.pt or larger)
2. Decrease confidence threshold slightly
3. Set Resize Factor to 1.0 (full resolution)
4. Ensure good lighting in your videos

## Switching Between Methods

You can easily switch between Background Subtraction and YOLO OBB:

1. **Stop any running tracking** (click Stop if tracking is active)
2. **Change Detection Method** dropdown
3. **Adjust parameters** for the selected method
4. **Start tracking** again

The application will use the appropriate detection method for your new session.

## Saving and Loading Configurations

Your detection method selection and YOLO parameters are automatically saved when you:
- Close the application (saves to `tracking_config.json`)
- Manually click "Save Config" (if implemented in your version)

When you reopen the application, your last-used settings will be restored, including:
- Detection method (Background Subtraction or YOLO OBB)
- Selected YOLO model
- Confidence and IOU thresholds
- Target classes

## Troubleshooting

### "YOLO Parameters panel doesn't appear"
- Make sure you selected "YOLO OBB" from the Detection Method dropdown
- Try restarting the application

### "Model download is slow"
- First-time model download can take a few minutes depending on internet speed
- The model is cached locally after first download
- Try using a smaller model (yolov8n-obb.pt) for faster initial setup

### "Not detecting my animals"
- Try lowering the confidence threshold
- Check if your animals match COCO classes (or use custom model)
- Ensure video quality is good (not too dark, blurry, or low resolution)

### "Too many false detections"
- Increase confidence threshold
- Use target classes to filter specific animals
- Try a larger, more accurate model

## Advanced: Training Custom Models

If you need to detect animals not in the COCO dataset:

1. **Prepare your dataset** with oriented bounding box annotations
2. **Train using Ultralytics:**
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8s-obb.pt')
   model.train(data='your_data.yaml', epochs=100)
   ```
3. **Use in GUI:**
   - Select "Custom Model..." from dropdown
   - Browse to your `best.pt` file
   - Set appropriate confidence threshold

See the [YOLO Detection Guide](yolo_detection_guide.md) for complete training instructions.

## Summary

The GUI makes YOLO detection easy:
1. ✅ Select "YOLO OBB" from dropdown
2. ✅ Choose a model (or use custom)
3. ✅ Optionally adjust confidence
4. ✅ Start tracking!

No need to edit JSON files manually - everything is accessible through the intuitive interface!
