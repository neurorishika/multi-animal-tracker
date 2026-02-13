# Model Archive

This directory serves as the local model archive for the Multi-Animal-Tracker.

## Purpose

Storing your models here enables:

- **Cross-device compatibility**: Presets can reference models using relative paths
- **Organization**: Keep all models organized by type and species
- **Portability**: Share presets with colleagues who have the same models

## Directory Structure

Models are organized by **model type**, then by **species/organism**:

```
models/
├── YOLO-obb/              # Oriented Bounding Box detection
│   ├── flies/
│   │   ├── drosophila-melanogaster.pt
│   │   └── tagged-flies.pt
│   ├── fish/
│   │   └── zebrafish.pt
│   ├── mice/
│   │   └── mouse-obb.pt
│   └── yolo26s-obb.pt     # Generic model
│
├── YOLO-classify/         # Classification models
│   ├── flies/
│   │   └── color-tags.pt
│   └── fish/
│       └── species-classifier.pt
│
├── YOLO-pose/             # Pose estimation
│   └── flies/
│       └── fly-pose.pt
│
├── SLEAP/                 # SLEAP models
│   └── ...
│
└── ViTPose/               # ViTPose models
    └── ...
```

## Usage

### Adding Models

1. **Automatic**: When selecting a model outside this directory, the application will offer to copy it here
2. **Manual**: Copy your model files into the appropriate type/species subdirectory

### Model References

- **Presets**: Models in this archive are saved with relative paths (e.g., `flies/drosophila.pt`)
- **Configs**: Models can be anywhere; paths are preserved for video-specific configs
- **Resolution**: The application automatically finds models here when loading presets

## Recommended Organization

### For YOLO OBB Detection Models

Place in `YOLO-obb/<species>/`:
- `YOLO-obb/flies/` - Drosophila and other fly species
- `YOLO-obb/fish/` - Zebrafish, medaka, etc.
- `YOLO-obb/mice/` - Mouse models
- `YOLO-obb/` (root) - Generic/multi-species models

### For Individual ID Models

Place in `YOLO-classify/<species>/`:
- Classification models for color tags
- Individual identification models

### For Pose Estimation

Place in appropriate type folder:
- `YOLO-pose/<species>/` - YOLO-based pose models
- `SLEAP/<species>/` - SLEAP models
- `ViTPose/<species>/` - ViTPose models

## Example: Fly Tracking Setup

```
models/YOLO-obb/flies/
├── drosophila-melanogaster.pt    # Main detection model
└── tagged-drosophila.pt          # For tagged individuals

models/YOLO-classify/flies/
└── color-tags-classifier.pt      # Individual ID via color tags
```

Preset would reference these as:
- Detection: `flies/drosophila-melanogaster.pt`
- Classification: `flies/color-tags-classifier.pt`

## Notes

- **Subdirectories**: Fully supported - entire relative path is preserved
- **Cross-device**: Models outside this archive use absolute paths (not portable)
- **Validation**: Missing models trigger warnings when loading presets
- **Auto-creation**: Directories are created automatically when needed
