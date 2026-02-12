# PoseKit Labeler - New Features Summary

## 1. Crash-Safe Recovery ✓

**Implementation:** `CrashSafeWriter` class in `pose_label_extensions.py`

- Uses atomic file writes with temporary files
- Writes to `.tmp` file in same directory, then atomic rename
- Prevents data loss if application crashes during save
- Integrated into `save_yolo_pose_label()` function

**Usage:**
- Automatic - all label saves now use crash-safe writing
- No user action required

## 2. Versioned Label Backups ✓

**Implementation:** `LabelVersioning` class in `pose_label_extensions.py`

- Maintains rolling backups in `labels_history/` directory
- Default: keeps last 5 versions per frame
- Backup format: `{stem}.v001.txt`, `{stem}.v002.txt`, etc.
- Automatically creates backup before overwriting

**Features:**
- `backup_label()`: Create versioned backup
- `restore_label()`: Restore from backup
- `list_versions()`: List all available versions
- Automatic rotation (removes oldest when limit exceeded)

**Usage:**
- Automatic backup on every save
- Can be disabled by passing `create_backup=False` to `save_yolo_pose_label()`

## 3. Per-Frame Metadata/Tags ✓

**Implementation:** 
- `FrameMetadata` dataclass for storing metadata
- `MetadataManager` class for managing metadata
- `FrameMetadataDialog` for UI interaction

**Features:**
- **Tags:** Pre-defined tags for common issues:
  - `occluded` - Frame has occlusions
  - `weird_posture` - Unusual pose
  - `motion_blur` - Motion artifacts
  - `poor_lighting` - Lighting issues
  - `partial_view` - Incomplete view
  - `unclear` - General quality issues
  
- **Notes:** Free-form text notes per frame
- **Cluster ID:** Automatically stored when clustering is performed
- Persistent storage in JSON format

**Data Structure:**
```python
{
    "image_path": str,
    "tags": List[str],
    "notes": str,
    "cluster_id": Optional[int]
}
```

**Integration Points (to be added to main GUI):**
- Right-click frame → "Edit Metadata"
- Menu: Tools → Frame Metadata
- Shortcut: Ctrl+M
- Filter frames by tag in frame list

## 4. Cluster-Stratified Dataset Splitting ✓

**Implementation:** Functions in `pose_label_extensions.py` and `DatasetSplitDialog`

### Train/Val/Test Split
- `cluster_stratified_split()` function
- Preserves cluster distribution across splits
- Configurable train/val/test fractions
- Guarantees minimum samples per cluster per split
- Small clusters (<3 frames) go entirely to train

**Parameters:**
- `train_frac`: Fraction for training (default: 0.7)
- `val_frac`: Fraction for validation (default: 0.15)
- `test_frac`: Fraction for test (default: 0.15)
- `min_per_cluster`: Minimum samples per cluster per split (default: 1)
- `seed`: Random seed for reproducibility

### K-Fold Cross-Validation
- `cluster_kfold_split()` function
- Creates K stratified folds
- Each fold preserves cluster distribution
- Useful for hyperparameter tuning

**Parameters:**
- `n_folds`: Number of folds (default: 5)
- `seed`: Random seed for reproducibility

### Output Files
Generated in `{project}/splits/` directory:

**Train/Val/Test:**
- `{split_name}_train.txt` - List of training images
- `{split_name}_val.txt` - List of validation images
- `{split_name}_test.txt` - List of test images
- `{split_name}_summary.json` - Split statistics

**K-Fold:**
- `{split_name}_fold1_train.txt`, `{split_name}_fold1_val.txt`
- `{split_name}_fold2_train.txt`, `{split_name}_fold2_val.txt`
- ... (one pair per fold)
- Summary JSON for each fold

## Integration into Main GUI

### Menu Structure (to be added):

```
File
├── ...
└── Export Dataset Splits... (Ctrl+Shift+E)

Tools
├── ...
├── Frame Metadata... (Ctrl+M)
└── Dataset Split Generator...
```

### Context Menu (frame list):
- Edit Metadata...
- Filter by Tag →
  - Occluded
  - Weird Posture
  - Motion Blur
  - Poor Lighting
  - Partial View
  - Unclear
- Clear All Tags

### Status Bar Indicators:
- Show current frame tags as small badges
- Show if frame is in a cluster

## Usage Workflow

### 1. Label your data normally

### 2. (Optional) Use Smart Select with clustering
- This assigns cluster IDs to frames
- Cluster info is stored in metadata

### 3. (Optional) Tag problematic frames
- Open frame metadata dialog
- Check relevant tags
- Add notes if needed

### 4. Generate dataset splits
- Open Dataset Split Generator
- Choose mode (Train/Val/Test or K-Fold)
- Configure parameters
- Click "Generate Split"
- Files are created in `{project}/splits/`

### 5. Use splits for training
```python
with open('splits/split_train.txt') as f:
    train_images = [line.strip() for line in f]
```

## Technical Notes

### Metadata Storage
- Location: `{project}/metadata.json`
- Format: JSON with per-frame entries
- Automatically saved after each change

### Backup Storage
- Location: `{project}/labels_history/`
- Format: `{frame_stem}.v{version:03d}.txt`
- Max versions configurable (default: 5)

### Cluster Integration
- Cluster IDs from Smart Select automatically stored
- Used for stratified splitting
- Visible in metadata dialog

### Safety Features
- All saves are atomic (crash-safe)
- Backups created before overwrites
- Temp files cleaned up automatically
- Cross-platform compatible

## Future Enhancements

1. **Metadata Export:**
   - Export metadata to CSV
   - Filter and export by tags

2. **Advanced Filtering:**
   - Show only frames with specific tags
   - Combine multiple tag filters
   - Search notes

3. **Statistics:**
   - Tag distribution histograms
   - Cluster size visualization
   - Split balance verification

4. **Batch Operations:**
   - Bulk tag assignment
   - Tag propagation (apply to similar frames)
   - Metadata templates

5. **Integration with Training:**
   - Auto-generate YAML config for YOLO
   - Include split paths
   - Export metadata alongside datasets
