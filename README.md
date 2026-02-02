# Multi-Animal Tracker

A real-time multi-animal tracking system with support for both background subtraction and YOLO OBB detection methods. Designed for behavioral analysis in controlled arenas with circular ROIs.

## Features

- **Dual Detection Methods**:
  - Background Subtraction: Fast, CPU-friendly detection for moving animals
  - YOLO OBB: Deep learning-based detection with oriented bounding boxes
  - **Intelligent Batched YOLO**: Automatic GPU batch size optimization for 2-5× faster detection
- **Real-time Tracking**: Kalman filter-based tracking with Hungarian algorithm assignment
- **Bidirectional Tracking**: Forward and backward passes with trajectory merging for improved accuracy
- **Memory-Efficient Detection Caching**: Reuses forward detections in backward pass - no RAM-intensive video reversal needed
- **Circular ROI Support**: Optimized for circular arenas
- **Interactive GUI**: Live visualization with PySide2
- **Trajectory Analysis**: Full trajectory recording and post-processing
- **CSV Export**: Automatic data logging for downstream analysis
- **Histogram Visualization**: Real-time monitoring of velocity, orientation, size, and assignment cost

## Installation

### Quick Install (Recommended)

We use a two-step installation for maximum speed: **mamba** for conda packages, **uv** for pip packages.

```bash
# Clone the repository
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker

# Step 1: Create conda environment (use mamba for 10-100x faster install)
mamba env create -f environment.yml
conda activate multi-animal-tracker-base

# Step 2: Install pip packages (uv is 10-100x faster than pip)
uv pip install -v -r requirements.txt
```

### GPU-Accelerated Install (NVIDIA GPUs)

For 8-30x faster background subtraction and 2-5x faster YOLO inference:

```bash
mamba env create -f environment-gpu.yml
conda activate multi-animal-tracker-gpu
uv pip install -v -r requirements-gpu.txt
```

> **Note**: Edit `requirements-gpu.txt` to match your CUDA version (11.x, 12.x, or 13.x)

### Minimal Install

For lightweight deployments with minimal dependencies:

```bash
mamba env create -f environment-minimal.yml
conda activate multi-animal-tracker-minimal
uv pip install -v -r requirements-minimal.txt
```

📖 **See [ENVIRONMENTS.md](ENVIRONMENTS.md) for detailed installation options, troubleshooting, and platform-specific notes.**

### Installing Mamba (if not already installed)

```bash
conda install -c conda-forge mamba
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

**GPU Batching (Full Tracking Only):** When running full tracking (not preview) with YOLO on a GPU, the system automatically:
- Detects your device (CUDA or Apple Silicon MPS)
- Estimates optimal batch size based on available memory
- Runs detection in two phases: (1) batched detection → cache, (2) tracking + visualization
- Provides **2-5× speedup** on GPU with zero configuration

**Batch Settings (in Detection tab):**
- **Enable Batched Detection**: Turn GPU batching on/off
- **Batch Size Mode**: "Auto" (recommended) or "Manual"
- **Manual Batch Size**: Override auto-detection (1-64 frames)

**Advanced Configuration** (`advanced_config.json` in package directory):
- `mps_memory_fraction`: 0.3 (30% of unified memory for Apple Silicon)
- `cuda_memory_fraction`: 0.7 (70% of VRAM for NVIDIA GPUs)

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

## Bidirectional Tracking with Detection Caching

The tracker supports bidirectional tracking to improve trajectory accuracy by running two passes:

### How It Works

1. **Forward Pass**: 
   - Tracks animals from start to end
   - Caches all detection data to disk (~10 MB per 10,000 frames)
   - Outputs `*_forward.csv`

2. **Backward Pass**:
   - Loads cached detections from forward pass
   - Runs tracking algorithm in reverse chronological order
   - **No video reading** - operates purely on cached detection data
   - **No visualization** - maximizes speed by skipping frame processing entirely
   - Detection computation is skipped entirely (uses cache)
   - Outputs `*_backward.csv`

3. **Trajectory Merging**:
   - Intelligently merges forward and backward trajectories
   - Resolves ID switches and occlusions
   - Outputs final `*_merged.csv`

### Benefits

- **Extremely Fast Backward Pass**: No frame reading/seeking, no detection, no visualization - pure tracking
- **Memory Efficient**: No RAM-intensive FFmpeg video reversal required
- **Consistent**: Same detections in both passes ensure reproducibility
- **Robust**: Bidirectional tracking resolves ambiguities and improves accuracy
- **Typical speedup**: Backward pass is ~50-70% faster than forward pass

### Configuration

Enable/disable in GUI: Check "Run Backward Tracking after Forward" in Tracking tab, or in config:

```json
{
  "enable_backward_tracking": true
}
```

**Temporary Files**: Detection cache (`*_detection_cache.npz`) is automatically cleaned up after merging if auto-cleanup is enabled.

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
- [GPU Utils Developer Guide](docs/gpu_utils_developer_guide.md) - GPU acceleration and device detection
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
│           ├── csv_writer.py
│           └── gpu_utils.py     # GPU/acceleration detection
├── docs/
│   ├── yolo_detection_guide.md
│   ├── gpu_utils_developer_guide.md
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
