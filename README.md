# Multi-Animal Tracker

A real-time multi-animal tracking system with support for both background subtraction and YOLO OBB detection methods. Designed for behavioral analysis in controlled arenas with circular ROIs.

## Features

- **Dual Detection Methods**:
  - Background Subtraction: Fast, CPU-friendly detection for moving animals
  - YOLO OBB: Deep learning-based detection with oriented bounding boxes
- **Real-time Tracking**: Kalman filter-based tracking with Hungarian algorithm assignment
- **Circular ROI Support**: Optimized for circular arenas
- **Interactive GUI**: Live visualization with PySide2
- **Trajectory Analysis**: Full trajectory recording and post-processing
- **CSV Export**: Automatic data logging for downstream analysis
- **Histogram Visualization**: Real-time monitoring of velocity, orientation, size, and assignment cost

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker

# Create and activate conda environment
conda env create -f environment.yml
conda activate multi-animal-tracker

# Install the package
pip install -e .
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n multi-animal-tracker python=3.11
conda activate multi-animal-tracker

# Install core dependencies
conda install -c conda-forge opencv numpy scipy matplotlib pyside2

# For YOLO detection (optional)
conda install pytorch torchvision -c pytorch
pip install ultralytics

# Install the package
pip install -e .
```

## Quick Start

### Launch the GUI

```bash
multianimaltracker
# or
mat
```

### Basic Usage

1. **Load a video**: Click "Browse" to select your video file
2. **Set output path**: Choose where to save tracking data (CSV)
3. **Select detection method**: 
   - Choose "Background Subtraction" (default) or "YOLO OBB" from the Detection Method dropdown
   - If using YOLO, select your preferred model (yolov8s-obb.pt recommended for balance of speed/accuracy)
4. **Configure parameters**: Adjust tracking parameters as needed
5. **Start tracking**: Click "Start Tracking"

### Using YOLO Detection (NEW!)

YOLO detection is now fully integrated into the GUI with **YOLO26** (latest - January 2026) and YOLO11 models:

1. In the "Detection Method" section, select **"YOLO OBB"**
2. Choose a model from the dropdown:
   - **yolo26s-obb.pt** - Latest YOLO26, 43% faster CPU inference ⭐ **Recommended**
   - **yolo26n-obb.pt** - Fastest for edge devices
   - **yolov11s-obb.pt** - YOLO11 alternative
   - **Custom Model...** - Use your own trained model
3. Optionally adjust confidence threshold and other parameters
4. Start tracking!

**Auto-Download:** Pretrained models are automatically downloaded on first use (6-50MB). Cached locally for future use.

The first time you use a pretrained model, it will be automatically downloaded (~6-50MB depending on model size).

## Detection Methods

### Background Subtraction (Default)

Best for:
- Stationary camera setups
- High contrast between animals and background
- Moving animals against static background
- Real-time processing on CPU

**Configuration**: See `tracking_config.json` for background subtraction parameters.

### YOLO OBB Detection

Best for:
- Detecting stationary animals
- Varying lighting conditions
- Complex backgrounds
- When custom animal models are available

**Configuration**: Set `"detection_method": "yolo_obb"` in your config file.

For detailed YOLO setup and usage, see [YOLO Detection Guide](docs/yolo_detection_guide.md).

## Configuration

All tracking parameters are controlled via `tracking_config.json`:

```json
{
  "detection_method": "background_subtraction",  // or "yolo_obb"
  "max_targets": 8,
  "yolo_model_path": "yolov8n-obb.pt",  // for YOLO detection
  "yolo_confidence_threshold": 0.25,
  "threshold_value": 9,
  "min_contour_area": 50,
  ...
}
```

### Key Parameters

- `detection_method`: Choose `"background_subtraction"` or `"yolo_obb"`
- `max_targets`: Maximum number of animals to track
- `yolo_model_path`: Path to YOLO model (for YOLO detection)
- `yolo_confidence_threshold`: Minimum confidence for YOLO detections
- `resize_factor`: Scale video for faster processing
- `kalman_noise`, `kalman_meas_noise`: Kalman filter tuning

## Documentation

- [User Guide](docs/user_guide.md) - Comprehensive usage instructions
- [YOLO Detection Guide](docs/yolo_detection_guide.md) - YOLO setup and configuration
- [API Reference](docs/api_reference.md) - Developer documentation
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Project Structure

```
multi-animal-tracker/
├── src/
│   └── multi_tracker/
│       ├── main.py              # Application entry point
│       ├── core/
│       │   ├── detection.py     # Detection methods (BG subtraction + YOLO)
│       │   ├── background_models.py
│       │   ├── tracking_worker.py
│       │   ├── kalman_filters.py
│       │   ├── assignment.py
│       │   └── post_processing.py
│       ├── gui/
│       │   ├── main_window.py
│       │   └── histogram_widgets.py
│       └── utils/
│           ├── video_io.py
│           ├── image_processing.py
│           ├── geometry.py
│           └── csv_writer.py
├── docs/
│   ├── yolo_detection_guide.md
│   ├── user_guide.md
│   └── api_reference.md
├── tracking_config.json         # Default configuration
├── environment.yml              # Conda environment
└── README.md
```

## Requirements

### Core Dependencies
- Python >= 3.11
- OpenCV
- NumPy
- SciPy
- Matplotlib
- PySide2

### Optional (for YOLO detection)
- PyTorch
- ultralytics

## Output Format

Tracking data is saved to CSV with the following columns:

- `track_id`: Track identifier (0 to max_targets-1)
- `trajectory_id`: Unique trajectory ID (handles track respawning)
- `local_count`: Frame count within this trajectory
- `x`, `y`: Position coordinates
- `orientation`: Angle in radians
- `frame`: Global frame number
- `state`: Track state (active, occluded, lost)

## Examples

### Using Background Subtraction
```json
{
  "detection_method": "background_subtraction",
  "max_targets": 8,
  "threshold_value": 9,
  "brightness": 0.0,
  "contrast": 0.6,
  "gamma": 3.0
}
```

### Using YOLO Detection
```json
{
  "detection_method": "yolo_obb",
  "max_targets": 8,
  "yolo_model_path": "yolov8s-obb.pt",
  "yolo_confidence_threshold": 0.3,
  "resize_factor": 0.8
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - See LICENSE file for details

## Citation

If you use this software in your research, please cite:

```
@software{multi_animal_tracker,
  author = {Mohanta, Rishika},
  title = {Multi-Animal Tracker},
  year = {2025},
  url = {https://github.com/neurorishika/multi-animal-tracker}
}
```

## Acknowledgments

- Built for the MBL Neurobiology course
- YOLO OBB detection powered by [Ultralytics](https://github.com/ultralytics/ultralytics)
- GUI framework using PySide2

## Support

For questions, issues, or feature requests, please open an issue on GitHub or contact neurorishika@gmail.com.
