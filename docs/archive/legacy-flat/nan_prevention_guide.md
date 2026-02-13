# NaN Prevention Guide for Pose Model Training

## Problem

NaN (Not a Number) values during validation can occur due to several numerical instability issues:

1. **Very small bounding boxes** - Bboxes with near-zero width or height cause division by zero in loss calculations
2. **Non-finite coordinates** - NaN or infinity values in keypoint/bbox coordinates
3. **Invalid coordinate ranges** - Values outside the expected [0, 1] normalized range
4. **Edge cases** - Keypoints at exact image boundaries or degenerate cases

## Fixes Applied

### 1. Enhanced Label Validation (`pose_label_extensions.py`)

#### Minimum Bounding Box Size
- Added `MIN_BBOX_DIM = 0.001` (approximately 1 pixel at 640px resolution)
- Prevents training on tiny/degenerate bboxes that cause numerical instability

#### NaN/Inf Detection
- Added explicit checks using `np.isfinite()` for all numeric values
- Warning logs when non-finite values are detected
- Invalid labels are rejected before training

#### Coordinate Validation
- Added epsilon tolerance (`1e-8`) for floating-point comparisons
- Validates that coordinates are within reasonable bounds
- Checks both bbox and keypoint coordinates independently

### 2. Improved Save Functions (`pose_label_extensions.py` & `pose_label.py`)

#### Enhanced `_clamp01()` Function
```python
def _clamp01(x: float) -> float:
    if not np.isfinite(x):
        logger.warning(f"Non-finite value clamped to 0: {x}")
        return 0.0
    return max(0.0, min(1.0, x))
```
- Detects and handles NaN/inf values before they enter training data
- Logs warnings so you can find the source

#### Safe `save_yolo_pose_label()`
- Validates bbox coordinates before normalization
- Ensures minimum bbox dimensions in normalized space
- Checks for non-finite values at each step
- Uses `max(img_w, 1)` to prevent division by zero

### 3. Robust `compute_bbox_from_kpts()`

- Added `MIN_BBOX_SIZE = 2.0` pixels to ensure reasonable bbox sizes
- Validates keypoint coordinates are finite
- Expands tiny bboxes to minimum size
- Logs warnings for problematic cases

### 4. Validation Tool (`validate_labels.py`)

A command-line tool to check labels before training:

```bash
python src/multi_tracker/gui/validate_labels.py \
    path/to/labels \
    --kpt-count 13 \
    --verbose
```

**Features:**
- Checks all label files for NaN/inf values
- Validates bbox dimensions and coordinate ranges
- Ensures visibility values are valid (0, 1, or 2)
- Reports detailed issues for each problematic label
- Summary statistics

## How to Use

### Before Training

1. **Validate your dataset:**
```bash
# Navigate to your project directory
cd /path/to/your/project

# Run validation
python src/multi_tracker/gui/validate_labels.py \
    labels/ \
    --kpt-count <number_of_keypoints> \
    --verbose
```

2. **Check the output:**
   - ✅ All valid → Safe to train
   - ⚠️  Issues found → Review and fix problematic labels

### Common Issues and Fixes

#### Issue: "Bbox width/height too small"
**Cause:** Keypoints are too close together or only one visible keypoint

**Solutions:**
- Label more keypoints for that frame
- Increase `bbox_pad_frac` in project settings
- Skip frames where the animal is too small to annotate accurately

#### Issue: "Non-finite bbox value"
**Cause:** Corrupted label file or invalid calculations

**Solutions:**
- Delete the label file and re-annotate
- Check for manual edits to `.txt` files
- Ensure image dimensions are valid

#### Issue: "Keypoint out of range"
**Cause:** Keypoint placed outside image boundaries

**Solutions:**
- Re-annotate the frame
- Ensure you're clicking within image bounds
- Check for coordinate scaling issues

#### Issue: "No visible keypoints"
**Cause:** All keypoints marked as missing (v=0)

**Solutions:**
- Mark at least some keypoints as visible (v=2)
- Skip frames where no keypoints are visible
- Use occluded (v=1) for partially visible keypoints

### Best Practices

1. **Label Quality:**
   - Always have at least 2-3 visible keypoints per frame
   - Avoid labeling frames where the subject is barely visible
   - Use appropriate visibility flags: 0=missing, 1=occluded, 2=visible

2. **Project Settings:**
   - Use reasonable `bbox_pad_frac` (default 0.1-0.2)
   - Set minimum image size requirements
   - Test with a small dataset first

3. **Training Parameters:**
   - Start with conservative augmentation values
   - Use auto-batch reduction for OOM protection
   - Monitor training logs for early warnings

4. **Regular Validation:**
   - Run validation after each labeling session
   - Check validation metrics during training
   - Watch for sudden NaN appearances in logs

## Training Parameters to Avoid NaN

### Safe Defaults
```python
batch_size = 16  # Or use auto-batch
imgsz = 640
patience = 10
epochs = 50

# Conservative augmentations
hsv_h = 0.01     # Small hue jitter
hsv_s = 0.2      # Moderate saturation
hsv_v = 0.1      # Small brightness
degrees = 5.0    # Conservative rotation
translate = 0.05 # Small translation
scale = 0.2      # Moderate scaling
fliplr = 0.2     # 20% flip probability
```

### Red Flags During Training

Watch for these in training logs:
- `nan` or `inf` in loss values
- Validation loss suddenly jumping to NaN
- Extremely large gradients
- Warnings about non-finite values

### If NaN Occurs During Training

1. **Stop training immediately** - Continuing wastes time and resources

2. **Run validation tool:**
```bash
python src/multi_tracker/gui/validate_labels.py \
    dataset/labels/train/ \
    --kpt-count <N>
```

3. **Check recent logs** for warnings about specific files

4. **Fix problematic labels:**
   - Re-annotate frames with issues
   - Remove badly labeled frames
   - Check for very small subjects

5. **Reduce learning rate** (if using custom training):
   - YOLO usually auto-adjusts, but manual reduction may help

6. **Restart training** after fixing labels

## Monitoring During Training

### What to Watch

1. **Loss curves:**
   - Should decrease smoothly
   - Validation loss should track training loss
   - Sudden spikes may indicate problematic samples

2. **Metrics:**
   - PCK (Percentage of Correct Keypoints)
   - OKS (Object Keypoint Similarity)
   - Bbox metrics

3. **Logs:**
   - Warning messages about labels
   - Memory usage (OOM can sometimes manifest as NaN)
   - Epoch completion without errors

### When to Intervene

- **Loss becomes NaN** → Stop and validate dataset
- **Loss oscillates wildly** → Reduce learning rate or augmentation
- **No improvement after many epochs** → Check dataset quality
- **OOM errors** → Reduce batch size (auto-batch should handle this)

## Summary

The fixes prevent NaN values by:
1. ✅ Validating all numeric values for NaN/inf
2. ✅ Enforcing minimum bbox dimensions
3. ✅ Checking coordinate ranges
4. ✅ Providing detailed warnings and logs
5. ✅ Offering validation tools to catch issues early

**Always validate your dataset before training!** This saves time and prevents wasted GPU hours.
