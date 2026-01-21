# YOLO26 Integration Update

## What's New

The Multi-Animal Tracker now features **YOLO26** - the latest YOLO model released in January 2026! YOLO26 brings significant performance improvements and is now the default recommended model in the GUI.

## YOLO26 Highlights

### 🚀 Performance Improvements
- **43% faster CPU inference** compared to previous YOLO versions
- Optimized specifically for edge computing and resource-constrained devices
- Real-time performance even without GPU acceleration

### 🎯 Architecture Innovations
- **End-to-End Inference:** No NMS (Non-Maximum Suppression) post-processing needed
- **DFL Removal:** Simplified architecture for better hardware compatibility
- **MuSGD Optimizer:** Hybrid SGD + Muon optimizer for stable, faster training
- **ProgLoss + STAL:** Improved loss functions for better small-object detection

### 📦 Deployment Benefits
- Simplified export process
- Broader hardware support (edge devices, IoT, embedded systems)
- Reduced latency and memory footprint
- Easier integration into production systems

## Available YOLO26 Models

All YOLO26 models support **Oriented Bounding Boxes (OBB)** for tracking:

| Model | Speed | Accuracy | mAP@50-95 | Params | FLOPs | Best For |
|-------|-------|----------|-----------|--------|-------|----------|
| yolo26n-obb.pt | ⚡⚡⚡⚡⚡ | ⭐⭐ | 40.9 | 2.4M | 5.4G | Real-time edge |
| yolo26s-obb.pt | ⚡⚡⚡⚡ | ⭐⭐⭐ | 48.6 | 9.5M | 20.7G | Balanced ⭐ |
| yolo26m-obb.pt | ⚡⚡⚡ | ⭐⭐⭐⭐ | 53.1 | 20.4M | 68.2G | High accuracy |
| yolo26l-obb.pt | ⚡⚡ | ⭐⭐⭐⭐⭐ | 55.0 | 24.8M | 86.4G | Best accuracy |
| yolo26x-obb.pt | ⚡ | ⭐⭐⭐⭐⭐ | 57.5 | 55.7M | 193.9G | Maximum accuracy |

## Auto-Download Feature

All YOLO26 and YOLO11 models automatically download on first use:

1. **Select model** from GUI dropdown
2. **First use:** Model downloads automatically (6-50MB)
3. **Cached locally:** No re-download needed for future use
4. **Progress indication:** Download progress shown in terminal

**Download locations:**
- **macOS/Linux:** `~/.cache/ultralytics/`
- **Windows:** `%USERPROFILE%\.cache\ultralytics\`

## Changes Made

### GUI Updates
**File:** `src/multi_tracker/gui/main_window.py`

**Old Model List (YOLOv8):**
```python
"yolov8n-obb.pt (Nano - Fastest)",
"yolov8s-obb.pt (Small - Balanced)",
...
```

**New Model List (YOLO26 + YOLO11):**
```python
"yolo26n-obb.pt (YOLO26 Nano - Fastest, 43% faster CPU)",
"yolo26s-obb.pt (YOLO26 Small - Balanced)",
"yolo26m-obb.pt (YOLO26 Medium)",
"yolo26l-obb.pt (YOLO26 Large)",
"yolo26x-obb.pt (YOLO26 Extra Large)",
"yolov11n-obb.pt (YOLO11 Nano)",
"yolov11s-obb.pt (YOLO11 Small)",
...
```

**Default changed from:** `yolov8s-obb.pt` → `yolo26s-obb.pt`

### Configuration Updates
**File:** `tracking_config.json`
```json
{
  "yolo_model_path": "yolo26s-obb.pt"  // Updated from yolov8n-obb.pt
}
```

### Documentation Updates
Updated files:
- ✅ `README.md` - Added YOLO26 info and features
- ✅ `docs/yolo_detection_guide.md` - Updated model list and recommendations
- ✅ `docs/yolo_gui_guide.md` - Updated examples with YOLO26
- ✅ `YOLO_QUICK_REFERENCE.md` - Updated quick start with YOLO26
- ✅ `YOLO_GUI_INTEGRATION.md` - Updated model availability

## Usage Examples

### Quick Start with YOLO26

```bash
multianimaltracker
```

Then in GUI:
1. Select **"YOLO OBB"** from Detection Method
2. Choose **"yolo26s-obb.pt"** (default, recommended)
3. Click **"Start Tracking"**

First-time download will occur automatically (~9.5MB for yolo26s).

### Python API

```python
from ultralytics import YOLO

# Load YOLO26 model (auto-downloads on first use)
model = YOLO("yolo26s-obb.pt")

# Run inference
results = model("video.mp4")
```

### Configuration File

```json
{
  "detection_method": "yolo_obb",
  "yolo_model_path": "yolo26s-obb.pt",
  "yolo_confidence_threshold": 0.25,
  "yolo_iou_threshold": 0.7
}
```

## Performance Comparison

### YOLO26 vs YOLOv8 (OBB, COCO Dataset)

| Metric | YOLOv8s | YOLO26s | Improvement |
|--------|---------|---------|-------------|
| mAP@50-95 | 44.3 | 48.6 | +9.7% |
| CPU Inference | Baseline | 43% faster | +43% |
| Params | 11.0M | 9.5M | -13.6% |
| Model Size | ~22MB | ~19MB | -13.6% |

### YOLO26 vs YOLO11

| Metric | YOLO11s | YOLO26s | Notes |
|--------|---------|---------|-------|
| mAP@50-95 | 47.9 | 48.6 | YOLO26 slightly better |
| CPU Speed | Fast | 43% faster | YOLO26 optimized for CPU |
| Architecture | Traditional | End-to-end | YOLO26 no NMS |
| Edge Deploy | Good | Excellent | YOLO26 simplified |

## Why YOLO26?

### Advantages over YOLOv8
- ✅ 43% faster on CPU
- ✅ Better small object detection
- ✅ Simplified architecture
- ✅ Better edge deployment
- ✅ End-to-end inference

### Advantages over YOLO11
- ✅ Even faster CPU performance
- ✅ Latest optimization techniques
- ✅ Improved training stability
- ✅ Better convergence

### When to Use YOLO11 Instead
- ⚠️ If you need GPU-optimized performance specifically
- ⚠️ If you have existing YOLO11 trained models
- ⚠️ For comparison/benchmarking purposes

## Compatibility

### Ultralytics Version
YOLO26 requires:
```bash
pip install ultralytics>=8.4.0
```

**Check your version:**
```python
import ultralytics
print(ultralytics.__version__)
```

### Backward Compatibility
- ✅ All existing tracking code works unchanged
- ✅ Configuration files from older versions compatible
- ✅ Can still use YOLO11 models if preferred
- ✅ Custom trained models work as before

## Migration Guide

### From YOLOv8 to YOLO26

**Old:**
```json
{
  "yolo_model_path": "yolov8s-obb.pt"
}
```

**New:**
```json
{
  "yolo_model_path": "yolo26s-obb.pt"
}
```

### GUI Migration
1. Open Multi-Animal Tracker
2. Select "YOLO OBB" detection method
3. Choose YOLO26 model from dropdown (auto-selected by default)
4. No other changes needed!

### Custom Model Training
If training custom models, use YOLO26 as base:

```python
from ultralytics import YOLO

# Load YOLO26 pretrained
model = YOLO("yolo26s-obb.pt")

# Train on your data
model.train(
    data="your_data.yaml",
    epochs=100,
    imgsz=640
)
```

## Troubleshooting

### Download Issues

**Problem:** Model fails to download
**Solution:** 
```bash
# Manual download
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt
# Place in: ~/.cache/ultralytics/
```

### Version Conflicts

**Problem:** YOLO26 models not available
**Solution:**
```bash
# Update ultralytics
pip install --upgrade ultralytics
```

### Performance Not Improved

**Problem:** Not seeing 43% CPU speedup
**Solution:**
- Ensure using CPU inference (not GPU)
- Update to latest ultralytics version
- Check system resources (no thermal throttling)

## Future Enhancements

Planned improvements:
1. **YOLO26 Segmentation** - Add instance segmentation support
2. **YOLO26 Pose** - Add pose estimation models
3. **Model Comparison Tool** - Compare YOLO26 vs YOLO11 performance
4. **Quantized Models** - INT8 quantization for even faster inference
5. **Multi-Model Ensemble** - Combine multiple YOLO26 models

## Resources

### Documentation
- [YOLO26 Official Docs](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [Model Zoo](https://github.com/ultralytics/assets/releases)

### Citation
```bibtex
@software{yolo26_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO26},
  version = {26.0.0},
  year = {2026},
  url = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```

## Summary

YOLO26 is now the recommended model for the Multi-Animal Tracker:

| Feature | Status |
|---------|--------|
| GUI Integration | ✅ Complete |
| Auto-Download | ✅ Enabled |
| Default Model | ✅ yolo26s-obb.pt |
| Documentation | ✅ Updated |
| Backward Compatible | ✅ Yes |
| Performance | ✅ 43% faster CPU |

**Recommendation:** Use `yolo26s-obb.pt` for the best balance of speed and accuracy!

---

**Updated:** January 21, 2026  
**YOLO26 Release:** January 14, 2026  
**Integration Version:** 1.1.0
