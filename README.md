# Multi-Animal Tracker

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey)](https://github.com/neurorishika/multi-animal-tracker)

A high-performance, production-ready multi-animal tracking system for behavioral neuroscience research. Supports both classical background subtraction and state-of-the-art YOLO OBB deep learning detection methods, with GPU acceleration for real-time processing of high-resolution video.

## 🎯 Key Features

### Detection Methods
- **Background Subtraction**: CPU-optimized for moving animals on static backgrounds (8-30× faster with GPU)
- **YOLO OBB Detection**: Deep learning with oriented bounding boxes for stationary/moving animals
  - Latest YOLO26 models (Jan 2026): 43% faster CPU inference vs YOLO11
  - TensorRT optimization: 2-5× inference speedup on NVIDIA GPUs
  - Automatic batch size optimization based on available GPU memory
  - Custom model support for specialized applications

### Tracking & Analysis
- **Kalman Filter Tracking**: Predictive state estimation with Hungarian algorithm assignment
- **Bidirectional Tracking**: Forward/backward passes with intelligent trajectory merging
- **Memory-Efficient Caching**: Detection results cached to disk (~10 MB/10k frames), enabling:
  - Ultra-fast backward pass (50-70% faster - no video I/O, no detection)
  - Reproducible results across passes
  - No RAM-intensive video reversal required
- **Real-time Performance**: Live preview mode with configurable frame skipping
- **Post-processing Pipeline**: Automatic trajectory cleaning, gap filling, and validation

### User Interface
- **Interactive GUI**: Qt6-based interface with:
  - Live video preview with overlay visualization
  - ROI drawing tools (circular arenas, exclusion zones)
  - Real-time histograms (velocity, orientation, size, cost)
  - Parameter tuning without restart
  - Session logging for reproducibility
- **Batch Processing**: Command-line interface for headless operation
- **Cross-platform**: Linux, macOS, Windows support

### Data Export
- **CSV Format**: Industry-standard output with:
  - Track ID, position (x, y), orientation
  - Trajectory ID (handles track respawning)
  - State flags (active, occluded, lost)
  - Optional: detection confidence, assignment cost, uncertainty metrics
- **Individual Animal Analysis**: Extract per-animal datasets for downstream analysis
- **Trajectory Validation**: Automated quality metrics and outlier detection

## 📦 Installation

### Prerequisites

- **Conda/Mamba**: Package manager ([install instructions](https://mamba.readthedocs.io/en/latest/installation.html))
- **CUDA Toolkit** (GPU only): NVIDIA drivers + CUDA 11.x/12.x/13.x ([download](https://developer.nvidia.com/cuda-downloads))
- **Hardware Requirements**:
  - Minimum: 8 GB RAM, dual-core CPU
  - Recommended: 16+ GB RAM, quad-core CPU, NVIDIA GPU (8+ GB VRAM)

### Quick Start (Recommended)

We use a **two-step installation** optimized for speed: mamba for conda packages (10-100× faster than conda), uv for pip packages (10-100× faster than pip). Total install time: ~3 minutes.

```bash
# Clone repository
git clone https://github.com/neurorishika/multi-animal-tracker.git
cd multi-animal-tracker

# Install mamba (if not already installed)
conda install -c conda-forge mamba

# Step 1: Create conda environment with mamba
mamba env create -f environment.yml

# Step 2: Activate and install pip packages with uv
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt

# Verify installation
python -c "from multi_tracker.main import main; print('✓ Installation successful!')"
```

### GPU-Accelerated Installation (NVIDIA Only)

For 8-30× faster background subtraction (CuPy) and 2-5× faster YOLO inference (TensorRT):

```bash
# Step 1: Create GPU environment
mamba env create -f environment-gpu.yml
conda activate multi-animal-tracker-gpu

# Step 2: Install GPU packages (edit requirements-gpu.txt first!)
# Important: Match PyTorch/CuPy to your CUDA version (check with: nvidia-smi)
# Edit requirements-gpu.txt and uncomment appropriate CUDA version lines
uv pip install -v -r requirements-gpu.txt

# Test GPU availability
python -c "from multi_tracker.utils.gpu_utils import log_device_info; log_device_info()"
```

**CUDA Version Configuration**:
1. Check your CUDA version: `nvidia-smi` (top right corner)
2. Edit `requirements-gpu.txt`:
   - CUDA 13.x: Uncomment `cu130` lines (default)
   - CUDA 12.x: Uncomment `cu126` or `cu128` lines
   - CUDA 11.x: Uncomment `cu118` lines

### Minimal Installation

For production deployments with minimal dependencies (~2 GB vs ~5 GB):

```bash
mamba env create -f environment-minimal.yml
conda activate multi-animal-tracker-minimal
uv pip install -v -r requirements-minimal.txt
```

### Using Makefile (Alternative)

```bash
# View all available commands
make help

# Standard environment
make setup              # Step 1: Create env
conda activate multi-animal-tracker-base
make install            # Step 2: Install packages

# GPU environment
make setup-gpu
conda activate multi-animal-tracker-gpu
make install-gpu
```

### Troubleshooting Installation

**Problem: Slow conda solve**
```bash
# Use libmamba solver (faster)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

**Problem: Import errors after install**
```bash
# Reinstall package in editable mode
pip install -e . --no-deps --force-reinstall
```

**Problem: GPU not detected**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version  # Should match requirements-gpu.txt

# Reinstall GPU packages
uv pip install -v -r requirements-gpu.txt --force-reinstall
```

📖 **See [ENVIRONMENTS.md](ENVIRONMENTS.md) for detailed installation options, platform-specific notes, and advanced configuration.**

## 🚀 Quick Start Guide

### 1. Launch the Application

```bash
# Activate environment
conda activate multi-animal-tracker-base  # or -gpu, or -minimal

# Launch GUI
multianimaltracker
# Or shortcut
mat
```

### 2. First-Time Setup Workflow

#### A. Load Video and Configure Output
1. Click **"Browse Video"** → Select your video file (MP4, AVI, MOV supported)
2. Click **"Browse CSV Output"** → Choose save location for tracking data
3. *(Optional)* **"Browse Video Output"** → Save annotated video with overlays

#### B. Choose Detection Method

**For Moving Animals (Recommended for beginners)**:
1. Select **"Background Subtraction"** in Detection Method dropdown
2. Adjust threshold slider while watching preview (higher = more sensitive)
3. Fine-tune using Advanced tab:
   - Brightness/Contrast/Gamma for lighting correction
   - Morphological kernel size for noise reduction
   - Enable size filtering to exclude artifacts

**For Stationary or Complex Scenarios**:
1. Select **"YOLO OBB"** in Detection Method dropdown
2. Choose model:
   - **yolo26s-obb.pt**: Best balance (recommended) ⭐
   - **yolo26n-obb.pt**: Fastest, lower accuracy
   - **Custom Model**: Browse to your trained .pt file
3. Adjust confidence threshold (0.25-0.35 typical range)
4. *(GPU users)* Enable TensorRT for 2-5× speedup (see GPU section)

#### C. Define ROI (Region of Interest)
1. Navigate to **ROI tab**
2. Click **"Enable ROI"**
3. Draw circular arena:
   - Click center point
   - Click edge to set radius
4. *(Optional)* Add exclusion zones for equipment/obstacles
5. Preview ROI overlay in video window

#### D. Configure Tracking Parameters
1. **Max Targets**: Set to expected number of animals
2. **Assignment Distance**: Max pixels animals can move between frames
   - Formula: `max_speed_pixels_per_second / fps`
   - Example: 100 px/s at 30 fps = 3.3 pixels
3. **Post-processing** (optional but recommended):
   - Enables trajectory smoothing and gap filling
   - Min trajectory length: Filters short spurious tracks
   - Max velocity break: Detects impossible movements

#### E. Test and Run
1. Click **"Test Detection"** to verify settings on current frame
2. Review detection overlay (green bounding boxes)
3. If satisfactory, click **"Start Tracking"**
4. Monitor progress bar and real-time histograms
5. *(Optional)* Enable backward tracking for improved accuracy

### 3. Understanding the Output

**CSV File Structure**:
```csv
TrackID,TrajectoryID,Index,X,Y,Theta,FrameID,State
0,0,0,512.3,384.7,1.234,0,active
0,0,1,515.1,385.2,1.241,1,active
1,1,0,256.8,512.1,0.456,0,active
...
```

- **TrackID**: Persistent animal ID (0 to max_targets-1)
- **TrajectoryID**: Unique ID per trajectory (handles track breaks/respawns)
- **Index**: Frame counter within this trajectory
- **X, Y**: Position in pixels (origin: top-left)
- **Theta**: Orientation in radians (0 = right, π/2 = down)
- **FrameID**: Global frame number
- **State**: `active`, `occluded`, `lost`, `tentative`

**With Confidence Tracking Enabled**:
```csv
TrackID,...,State,DetectionConfidence,AssignmentConfidence,PositionUncertainty
0,...,active,0.95,0.87,2.3
```

### 4. Common Workflows

#### Workflow 1: Basic Tracking (Background Subtraction)
```bash
mat  # Launch GUI
# 1. Load video.mp4
# 2. Select Background Subtraction
# 3. Adjust threshold slider (preview mode)
# 4. Set max targets = 8
# 5. Start Tracking
# 6. Output: video_output.csv
```

#### Workflow 2: YOLO with GPU Acceleration
```bash
conda activate multi-animal-tracker-gpu
mat
# 1. Load high-res video (4K supported)
# 2. Select YOLO OBB → yolo26s-obb.pt
# 3. Enable TensorRT (first run: builds engine, ~2-5 min)
# 4. Set TensorRT batch size = Auto
# 5. Start Tracking
# 6. Enjoy 2-5× speedup on batched inference
```

#### Workflow 3: Bidirectional Tracking (Best Accuracy)
```bash
mat
# 1. Load video + configure detection
# 2. Enable "Run Backward Tracking" in Tracking tab
# 3. Start Tracking
# 4. System runs: Forward → Backward → Merge
# 5. Outputs:
#    - video_forward.csv
#    - video_backward.csv
#    - video_merged.csv (use this one!)
#    - video_detection_cache.npz (auto-deleted)
```

#### Workflow 4: Batch Processing (Command Line)
```python
# batch_process.py
from multi_tracker.core.tracking_worker import TrackingWorker
import cv2

videos = ["exp1.mp4", "exp2.mp4", "exp3.mp4"]
config = {
    "DETECTION_METHOD": "yolo_obb",
    "YOLO_MODEL_PATH": "yolo26s-obb.pt",
    "MAX_TARGETS": 6,
    "ENABLE_GPU_BACKGROUND": True,
    "ENABLE_TENSORRT": True,
}

for video in videos:
    cap = cv2.VideoCapture(video)
    worker = TrackingWorker(
        video, 
        csv_writer_thread=...,
        params=config
    )
    worker.run()
    print(f"✓ Processed {video}")
```

## 🎮 Detection Methods in Detail

### Background Subtraction

**When to Use**:
- ✅ Moving animals on static background
- ✅ High contrast between animals and arena
- ✅ Controlled lighting conditions
- ✅ Real-time CPU processing required
- ✅ Circular arenas (optimized)

**How It Works**:
1. Builds background model (lightest pixel or adaptive)
2. Subtracts current frame from background
3. Thresholds difference to create foreground mask
4. Morphological operations (open/close) remove noise
5. Contour detection finds animal blobs
6. Ellipse fitting estimates position and orientation

**Performance**:
- **CPU**: 30-60 FPS on 1080p video (Intel i7)
- **GPU** (CuPy): 200-500 FPS on 1080p video (RTX 3080)
- **Memory**: ~50 MB baseline + video frame buffer

**Configuration Parameters**:
```json
{
  "DETECTION_METHOD": "background_subtraction",
  "THRESHOLD_VALUE": 9,              // Sensitivity (higher = more sensitive)
  "MORPH_KERNEL_SIZE": 3,            // Noise reduction (odd number, 3-11)
  "MIN_CONTOUR_AREA": 50,            // Min pixels to be considered animal
  "ENABLE_SIZE_FILTERING": true,      // Filter by area range
  "MIN_OBJECT_SIZE": 100,             // Min area (mm² if calibrated)
  "MAX_OBJECT_SIZE": 500,             // Max area (mm² if calibrated)
  "DARK_ON_LIGHT_BACKGROUND": true,   // Animal darker than background
  "BRIGHTNESS": 0.0,                  // Adjust brightness (-1 to 1)
  "CONTRAST": 0.6,                    // Adjust contrast (0 to 2)
  "GAMMA": 3.0,                       // Gamma correction (0.1 to 5)
  "ENABLE_ADAPTIVE_BACKGROUND": true, // Adapt to gradual lighting changes
  "BACKGROUND_LEARNING_RATE": 0.001   // How fast background adapts
}
```

**Tips**:
- Start with threshold = 9, adjust up/down while watching preview
- Use gamma correction (2-4) for fluorescent lighting
- Enable adaptive background for long recordings (>10 min)
- Set size filtering to exclude debris/reflections

---

### YOLO OBB Detection

**When to Use**:
- ✅ Stationary or slow-moving animals
- ✅ Variable lighting conditions
- ✅ Complex backgrounds (bedding, objects)
- ✅ Overlapping/touching animals
- ✅ Custom animal species (train your own model)

**How It Works**:
1. Preprocesses frame to 1024×1024 (YOLO input size)
2. Runs inference on PyTorch model (or TensorRT engine)
3. Non-maximum suppression filters overlapping detections
4. Oriented bounding boxes provide position + orientation
5. Confidence scores enable quality filtering

**Performance**:
- **CPU** (PyTorch): 3-8 FPS (yolo26n), 1-4 FPS (yolo26s)
- **GPU** (PyTorch): 20-50 FPS on RTX 3080
- **GPU** (TensorRT): 50-150 FPS on RTX 3080 (2-5× speedup)
- **Memory**: ~500 MB model weights + inference buffer

**Configuration Parameters**:
```json
{
  "DETECTION_METHOD": "yolo_obb",
  "YOLO_MODEL_PATH": "yolo26s-obb.pt",   // Model file or name
  "YOLO_CONFIDENCE_THRESHOLD": 0.25,      // Min confidence (0-1)
  "YOLO_IOU_THRESHOLD": 0.7,              // NMS overlap threshold
  "YOLO_TARGET_CLASSES": null,            // null = all classes, or [0, 2, 5]
  "YOLO_DEVICE": "auto",                  // "auto", "cuda:0", "mps", "cpu"
  "ENABLE_TENSORRT": true,                // GPU only, 2-5× speedup
  "TENSORRT_MAX_BATCH_SIZE": 16,          // Static batch size for TensorRT
  "ADVANCED_CONFIG": {
    "enable_yolo_batching": true,         // Batch process full video
    "yolo_batch_size_mode": "auto",       // "auto" or "manual"
    "yolo_manual_batch_size": 32,         // Override auto batch size
    "cuda_memory_fraction": 0.7            // % of VRAM to use
  }
}
```

**Available Models** (auto-downloaded on first use):
| Model | Size | Speed | Accuracy | Recommended For |
|-------|------|-------|----------|-----------------|
| yolo26n-obb.pt | 8 MB | Fastest | Lower | Edge devices, real-time preview |
| yolo26s-obb.pt | 27 MB | Fast | Good | **Most users** ⭐ |
| yolo26m-obb.pt | 55 MB | Medium | Better | High accuracy needed |
| yolo11n-obb.pt | 6 MB | Fast | Lower | YOLO11 baseline |
| yolo11s-obb.pt | 22 MB | Medium | Good | YOLO11 alternative |

**TensorRT Optimization** (NVIDIA GPUs only):
1. First run: Builds optimized engine (~2-5 minutes)
2. Engine cached to `~/.cache/multi_tracker/tensorrt/`
3. Filename includes batch size: `yolo26s_batch16.engine`
4. Subsequent runs: Instant load, 2-5× faster inference
5. Different batch sizes create separate cached engines

**Tips**:
- Use yolo26s-obb.pt for best balance of speed/accuracy
- Lower confidence threshold (0.2-0.25) for dim lighting
- TensorRT batch size: Start with Auto, reduce if OOM errors
- Train custom models on [Ultralytics](https://docs.ultralytics.com/) for species-specific detection

---

### Comparison Table

| Feature | Background Subtraction | YOLO OBB |
|---------|------------------------|----------|
| **Stationary animals** | ❌ No | ✅ Yes |
| **CPU speed** | ⚡ Very fast (30-60 FPS) | ⏱️ Moderate (3-8 FPS) |
| **GPU acceleration** | ✅ Yes (CuPy, 8-30×) | ✅ Yes (TensorRT, 2-5×) |
| **Lighting sensitivity** | ⚠️ High | ✅ Low |
| **Complex backgrounds** | ❌ Poor | ✅ Good |
| **Overlapping animals** | ❌ Struggles | ✅ Handles well |
| **Setup complexity** | ⚡ Simple (threshold tuning) | ⏱️ Moderate (model selection) |
| **Memory usage** | ✅ Low (50 MB) | ⚠️ High (500 MB) |
| **Custom animals** | ❌ Not applicable | ✅ Train custom model |

**Hybrid Approach**: Use YOLO for detection quality, background subtraction for real-time preview

## ⚡ GPU Acceleration Guide

### Overview

GPU acceleration provides dramatic speedups for computationally intensive operations:

| Operation | CPU | GPU (CUDA) | Speedup |
|-----------|-----|------------|---------|
| Background subtraction | 30 FPS | 250 FPS | 8-30× |
| YOLO inference (batch) | 5 FPS | 80 FPS | 2-5× (TensorRT) |
| Morphological ops | 50 FPS | 500 FPS | 10× |

### CuPy (Background Subtraction Acceleration)

**Automatic**: Enabled when GPU environment installed
- Accelerates morphological operations (erosion, dilation)
- Accelerates background model updates
- Falls back to CPU if compilation errors occur

**Requirements**:
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit matching CuPy version
- 2+ GB VRAM

**Verification**:
```python
from multi_tracker.utils.gpu_utils import log_device_info
log_device_info()
# Output:
# ✓ CUDA (CuPy): Available
#   Devices: 1
# ✓ CUDA (PyTorch): Available
#   Device: NVIDIA RTX 3080
```

**Troubleshooting**:
```
# CuPy compilation errors (CUDA 13 with older CuPy)
uv pip install --pre -f https://pip.cupy.dev/pre cupy-cuda13x --upgrade

# Memory errors
# Reduce resolution or disable GPU background:
"ENABLE_GPU_BACKGROUND": false
```

### TensorRT (YOLO Inference Acceleration)

**Manual**: Enable in GUI or config
- Converts PyTorch model to optimized TensorRT engine
- 2-5× faster inference on NVIDIA GPUs
- Supports static batch sizes for maximum throughput

**First-Time Setup**:
1. Enable TensorRT checkbox in YOLO tab
2. Set max batch size (16 recommended, 8 if OOM)
3. Start tracking
4. **Wait 2-5 minutes** for engine build (one-time)
5. Engine cached to `~/.cache/multi_tracker/tensorrt/`

**Batch Size Optimization**:
```json
{
  "ENABLE_TENSORRT": true,
  "TENSORRT_MAX_BATCH_SIZE": 16,  // Start here, reduce if OOM
  "ADVANCED_CONFIG": {
    "enable_yolo_batching": true,
    "yolo_batch_size_mode": "auto"  // Automatically uses TensorRT batch size
  }
}
```

**How It Works**:
1. **Forward Pass** (Phase 1): Batched YOLO detection
   - Processes video in chunks of `TENSORRT_MAX_BATCH_SIZE`
   - Larger batches padded to exact size (e.g., 44 frames → 3×16 + 1×12→16)
   - Results cached to disk
2. **Forward Pass** (Phase 2): Tracking + visualization
   - Loads cached detections
   - No re-inference needed

**Performance Example** (4K video, 750 frames, RTX 6000 Ada):
- PyTorch (no batching): 750 frames / 5 FPS = 150 seconds
- PyTorch (batch=44): 750 frames / 20 FPS = 37 seconds
- **TensorRT (batch=16)**: 750 frames / 80 FPS = **9 seconds** ⚡

**Troubleshooting**:
```
# Engine build fails (OOM)
# Reduce batch size: 16 → 8 → 4

# "input size not equal to max model size"
# Batch optimizer didn't clamp correctly - fixed in latest version
# Workaround: Set manual batch size matching TensorRT

# Different CUDA version
# Edit requirements-gpu.txt to match your CUDA (nvidia-smi)
```

### Apple Silicon (M1/M2/M3) - MPS

**Partial Support**:
- ✅ YOLO inference (PyTorch MPS backend)
- ❌ CuPy operations (CUDA only)
- ❌ TensorRT (NVIDIA only)

**Performance**: ~2-3× faster than CPU for YOLO, same as CPU for background subtraction

**Enable MPS**:
```json
{
  "YOLO_DEVICE": "mps"  // Automatic if CUDA unavailable
}
```

## 🔄 Bidirectional Tracking

Improve trajectory accuracy by running tracking in both temporal directions and intelligently merging results.

### How It Works

```
Forward Pass          Backward Pass         Merging
─────────────        ─────────────         ─────────
Frame 1 →            ← Frame 1000         Best of both
Frame 2 →            ← Frame 999          + conflict
Frame 3 →            ← Frame 998          resolution
   ...                    ...
Frame 1000 →         ← Frame 1

   ↓                      ↓                    ↓
*_forward.csv      *_backward.csv       *_merged.csv
                                         (USE THIS!)
```

### Three Phases

**Phase 1: Forward Pass**
- Tracks animals from start to end
- Saves detections to cache: `*_detection_cache.npz` (~10 MB / 10k frames)
- Outputs: `*_forward.csv`
- Time: 100% (baseline)

**Phase 2: Backward Pass** (50-70% faster!)
- Loads cached detections (no video I/O, no detection compute)
- Runs tracking in reverse chronological order
- No visualization (maximizes speed)
- Outputs: `*_backward.csv`
- Time: ~30-50% of forward pass ⚡

**Phase 3: Trajectory Merging**
- Aligns forward and backward trajectories
- Resolves ID switches and occlusions
- Fills gaps using bidirectional context
- Outputs: `*_merged.csv` (final result)
- Time: ~5% of forward pass

**Total Time**: ~135-155% of single-pass tracking for significantly improved accuracy

### Benefits

✅ **Resolves ID Switches**: Forward/backward agreement confirms identities  
✅ **Fills Occlusions**: Uses temporal context from both directions  
✅ **Improves Accuracy**: Typical 15-30% reduction in tracking errors  
✅ **Memory Efficient**: No RAM-intensive video reversal (FFmpeg approach)  
✅ **Reproducible**: Same detections in both passes  
✅ **Fast**: Backward pass reuses cached detections

### Configuration

**GUI**: Enable checkbox in Tracking tab → "Run Backward Tracking after Forward"

**Config File**:
```json
{
  "enable_backward_tracking": true,
  "auto_cleanup_cache": true  // Delete detection_cache.npz after merging
}
```

### When to Use

**✅ Highly Recommended For**:
- Long recordings (>5 minutes)
- Frequent occlusions
- High animal density (>4 animals)
- Critical experiments requiring maximum accuracy

**❓ Optional For**:
- Short recordings (<1 minute)
- Sparse tracking (1-2 animals)
- Preliminary analysis / pilot data

**❌ Not Needed For**:
- Real-time preview
- Test runs
- When speed >> accuracy

---

## Configuration

## ⚙️ Configuration Reference

### Configuration Files

1. **GUI Config** (`*_config.json`): Auto-saved next to video
   - Stores all GUI parameters
   - Auto-loaded when reopening same video
   - Shareable across experiments

2. **Default Config** (`tracking_config.json`): Package defaults
   - Located in package installation directory
   - Loaded when no video-specific config exists
   - Edit for global defaults

### Core Parameters

```json
{
  // Detection
  "DETECTION_METHOD": "background_subtraction",  // or "yolo_obb"
  "MAX_TARGETS": 8,                              // Max animals to track
  
  // Background Subtraction
  "THRESHOLD_VALUE": 9,                          // Sensitivity (1-50)
  "MORPH_KERNEL_SIZE": 3,                        // Noise reduction (3-11, odd)
  "MIN_CONTOUR_AREA": 50,                        // Min detection size (pixels²)
  "ENABLE_SIZE_FILTERING": true,
  "MIN_OBJECT_SIZE": 100,                        // Physical size (mm²)
  "MAX_OBJECT_SIZE": 500,
  "DARK_ON_LIGHT_BACKGROUND": true,              // Animal appearance
  
  // YOLO OBB
  "YOLO_MODEL_PATH": "yolo26s-obb.pt",
  "YOLO_CONFIDENCE_THRESHOLD": 0.25,             // 0.2-0.35 typical
  "YOLO_IOU_THRESHOLD": 0.7,                     // NMS overlap (0.5-0.8)
  "YOLO_DEVICE": "auto",                         // "cuda:0", "mps", "cpu"
  
  // Tracking
  "MAX_DISTANCE_THRESHOLD": 50,                  // Max pixels/frame movement
  "MIN_DETECTIONS_TO_START": 3,                  // Frames to confirm new track
  "MIN_TRAJECTORY_LENGTH": 10,                   // Post-process: min frames
  "MAX_OCCLUSION_GAP": 30,                       // Max frames to interpolate
  "CONTINUITY_THRESHOLD": 80,                    // Gap filling distance
  "MIN_RESPAWN_DISTANCE": 100,                   // Min distance for new track
  
  // Kalman Filter
  "KALMAN_PROCESS_NOISE": 4.0,                   // Motion model uncertainty
  "KALMAN_MEASUREMENT_NOISE": 1.0,               // Detection uncertainty
  
  // Performance
  "RESIZE_FACTOR": 1.0,                          // Scale video (0.5 = 50%)
  "FPS": 30,                                     // Video frame rate
  "ENABLE_GPU_BACKGROUND": true,                 // CuPy acceleration
  "ENABLE_TENSORRT": false,                      // TensorRT optimization
  
  // Output
  "ENABLE_POSTPROCESSING": true,                 // Trajectory cleaning
  "SAVE_CONFIDENCE_SCORES": false,               // Extra columns in CSV
  "ENABLE_BACKWARD_TRACKING": false              // Bidirectional mode
}
```

### Advanced Configuration

Edit `advanced_config` section in config file:

```json
{
  "ADVANCED_CONFIG": {
    // YOLO Batching
    "enable_yolo_batching": true,
    "yolo_batch_size_mode": "auto",              // or "manual"
    "yolo_manual_batch_size": 32,
    
    // GPU Memory Management
    "cuda_memory_fraction": 0.7,                 // NVIDIA: 70% of VRAM
    "mps_memory_fraction": 0.3,                  // Apple: 30% of unified memory
    
    // Background Model
    "enable_adaptive_background": true,
    "background_learning_rate": 0.001,           // Adaptation speed
    
    // Visualization
    "visualization_free_mode": false,            // Disable all viz for speed
    "enable_frame_prefetching": true             // Async frame loading
  }
}
```

### Calibration (Physical Units)

To convert pixels to mm/cm:

```json
{
  "PIXELS_PER_MM": 2.5,  // Measure known distance in video
  
  // Now use physical units:
  "MIN_OBJECT_SIZE": 100,         // 100 mm²
  "MAX_OBJECT_SIZE": 500,         // 500 mm²
  "MAX_DISTANCE_THRESHOLD": 20    // 20 mm/frame
}
```

**How to Calibrate**:
1. Place object of known size in arena (e.g., 10 mm marker)
2. Measure in pixels using video (e.g., 25 pixels)
3. Calculate: `PIXELS_PER_MM = 25 px / 10 mm = 2.5`

---

## 📚 Documentation

### User Guides
- **[User Guide](docs/user_guide.md)**: Comprehensive usage instructions with screenshots
- **[YOLO Detection Guide](docs/yolo_detection_guide.md)**: YOLO setup, model selection, custom training
- **[YOLO GUI Guide](docs/yolo_gui_guide.md)**: Using YOLO through the graphical interface
- **[Individual Analysis Guide](docs/individual_analysis_guide.md)**: Extract per-animal data for downstream analysis
- **[Confidence Tracking](docs/confidence_tracking.md)**: Using detection/assignment confidence scores
- **[Configuration Parameters](docs/configuration_parameters.md)**: Complete parameter reference
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions

### Developer Documentation
- **[API Reference](docs/api_reference.md)**: Python API for programmatic usage
- **[GPU Utils Developer Guide](docs/gpu_utils_developer_guide.md)**: GPU acceleration internals
- **[Installation Guide](docs/installation.md)**: Detailed installation for all platforms

### Quick Links
- **Environment Setup**: See [ENVIRONMENTS.md](ENVIRONMENTS.md) for installation variants
- **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for version history
- **License**: [MIT License](LICENSE)

---

## 🏗️ Project Structure

```
multi-animal-tracker/
├── src/
│   └── multi_tracker/
│       ├── main.py                    # Entry point, GUI launch
│       ├── core/                      # Core tracking algorithms
│       │   ├── detection.py           # Detection methods (BG + YOLO)
│       │   ├── background_models.py   # Background modeling
│       │   ├── tracking_worker.py     # Main tracking loop
│       │   ├── kalman_filters.py      # Predictive filtering
│       │   ├── assignment.py          # Hungarian algorithm
│       │   ├── post_processing.py     # Trajectory cleaning
│       │   └── individual_analysis.py # Per-animal data extraction
│       ├── gui/                       # Qt6 interface
│       │   ├── main_window.py         # Main application window
│       │   └── histogram_widgets.py   # Real-time visualization
│       └── utils/                     # Utilities
│           ├── gpu_utils.py           # GPU/acceleration detection
│           ├── batch_optimizer.py     # Auto batch sizing
│           ├── detection_cache.py     # Detection caching for backward pass
│           ├── frame_prefetcher.py    # Async frame loading
│           ├── csv_writer.py          # Async CSV output
│           ├── image_processing.py    # Frame preprocessing
│           ├── geometry.py            # ROI and spatial ops
│           └── dataset_generation.py  # YOLO dataset export
├── docs/                              # Documentation
│   ├── user_guide.md
│   ├── yolo_detection_guide.md
│   ├── gpu_utils_developer_guide.md
│   ├── api_reference.md
│   ├── confidence_tracking.md
│   ├── individual_analysis_guide.md
│   └── troubleshooting.md
├── environment.yml                    # Conda: standard
├── environment-gpu.yml                # Conda: GPU accelerated
├── environment-minimal.yml            # Conda: lightweight
├── requirements.txt                   # Pip: standard packages
├── requirements-gpu.txt               # Pip: GPU packages
├── requirements-minimal.txt           # Pip: minimal packages
├── ENVIRONMENTS.md                    # Installation guide
├── README.md                          # This file
├── CHANGELOG.md                       # Version history
├── LICENSE                            # MIT license
├── makefile                           # Build automation
└── setup.py                           # Package configuration
```

---

## 🎓 Example Use Cases

### 1. Fruit Fly Social Behavior
```python
config = {
    "DETECTION_METHOD": "background_subtraction",
    "MAX_TARGETS": 10,
    "THRESHOLD_VALUE": 12,
    "MIN_OBJECT_SIZE": 50,    # mm²
    "MAX_OBJECT_SIZE": 200,
    "PIXELS_PER_MM": 3.2,
    "FPS": 30
}
# Typical throughput: 60 FPS on CPU, 400 FPS on GPU
```

### 2. Zebrafish Larva Tracking (High Resolution)
```python
config = {
    "DETECTION_METHOD": "yolo_obb",
    "YOLO_MODEL_PATH": "yolo26s-obb.pt",
    "ENABLE_TENSORRT": True,
    "TENSORRT_MAX_BATCH_SIZE": 16,
    "MAX_TARGETS": 20,
    "RESIZE_FACTOR": 0.75,   # 4K → 3K for speed
    "ENABLE_BACKWARD_TRACKING": True
}
# Typical throughput: 80 FPS (TensorRT batch), 15 FPS (PyTorch single)
```

### 3. Mouse Open Field Test
```python
config = {
    "DETECTION_METHOD": "background_subtraction",
    "MAX_TARGETS": 1,
    "ENABLE_ADAPTIVE_BACKGROUND": True,  # Long recording
    "BACKGROUND_LEARNING_RATE": 0.001,
    "ENABLE_POSTPROCESSING": True,
    "MIN_TRAJECTORY_LENGTH": 30,         # Filter artifacts
    "ENABLE_BACKWARD_TRACKING": True
}
```

### 4. Ant Colony (Many Animals)
```python
config = {
    "DETECTION_METHOD": "yolo_obb",
    "YOLO_MODEL_PATH": "custom_ant_model.pt",  # Trained on ants
    "MAX_TARGETS": 50,
    "YOLO_CONFIDENCE_THRESHOLD": 0.20,  # Lower for small animals
    "MAX_DISTANCE_THRESHOLD": 30,       # Slow movers
    "MAX_OCCLUSION_GAP": 60,            # Many overlaps
    "ENABLE_GPU_BACKGROUND": True
}
```

---

## 🚦 Performance Benchmarks

### Test System
- CPU: Intel i7-12700K (12 cores)
- GPU: NVIDIA RTX 3080 (10 GB VRAM)
- RAM: 32 GB DDR5
- Video: 1920×1080, 30 FPS, 10,000 frames

### Results (Frames Per Second)

| Configuration | Detection FPS | Tracking FPS | Total FPS |
|--------------|---------------|--------------|-----------|
| BG Sub (CPU) | 55 | 300 | **45** |
| BG Sub (GPU/CuPy) | 420 | 300 | **180** |
| YOLO (CPU, yolo26s) | 4 | 300 | **4** |
| YOLO (GPU, PyTorch) | 35 | 300 | **30** |
| YOLO (GPU, TensorRT) | 110 | 300 | **85** |

**Speedup Factors**:
- CuPy vs CPU: 4× (background subtraction)
- TensorRT vs PyTorch: 3.1× (YOLO inference)
- GPU Total vs CPU Total: 11× (BG), 21× (YOLO)

### Memory Usage

| Configuration | RAM | VRAM | Disk Cache |
|--------------|-----|------|------------|
| BG Sub (CPU) | 200 MB | - | - |
| BG Sub (GPU) | 250 MB | 500 MB | - |
| YOLO (CPU) | 1.2 GB | - | 100 MB |
| YOLO (GPU, batch) | 1.5 GB | 3.2 GB | 100 MB |

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/multi-animal-tracker.git
cd multi-animal-tracker

# Create dev environment
mamba env create -f environment.yml
conda activate multi-animal-tracker-base
uv pip install -v -r requirements.txt

# Install development tools
pip install pytest black flake8

# Run tests
pytest

# Format code
black src/
```

### Code Style
- Follow PEP 8
- Use type hints where applicable
- Document all public functions
- Add tests for new features

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Rishika Mohanta

---

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

---

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@software{multi_animal_tracker_2026,
  title = {Multi-Animal Tracker: High-Performance Video Tracking with GPU Acceleration},
  author = {Mohanta, Rishika and Chemtob, Yohann},
  year = {2026},
  url = {https://github.com/rutalab/multi-animal-tracker},
  version = {1.0.0}
}
```

### Research Using This Software
- *Your paper here!* - Submit a PR to add your publication

---

## 🙏 Acknowledgments

This project builds on excellent open-source tools:

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**: State-of-the-art object detection
- **[OpenCV](https://opencv.org/)**: Computer vision primitives
- **[CuPy](https://cupy.dev/)**: GPU-accelerated NumPy
- **[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)**: Inference optimization
- **[Qt](https://www.qt.io/)**: Cross-platform GUI framework

Special thanks to:
- The SciPy community for the Hungarian algorithm implementation
- NVIDIA for CUDA toolkit and optimization guides
- All contributors and users providing feedback

---

## ❓ FAQ

### General Questions

**Q: How many animals can I track?**  
A: No theoretical limit, but tested up to 50. Set `MAX_TARGETS` to expected maximum. Performance scales roughly linearly.

**Q: What video formats are supported?**  
A: Any format OpenCV supports: MP4, AVI, MOV, MKV, etc. For best performance, use H.264/H.265 encoded MP4.

**Q: Can I track through the entire video even if animals enter/leave?**  
A: Yes! Enable `ENABLE_BACKWARD_TRACKING` for three-phase tracking (forward → reverse → merge) to handle late arrivals.

**Q: How do I extract individual animal data?**  
A: See [Individual Analysis Guide](docs/individual_analysis_guide.md). Use the individual analysis feature to segment trajectories by track ID.

### Detection & Tracking

**Q: Background subtraction vs YOLO - which should I use?**  
A: 
- **Background subtraction**: Clean backgrounds, high contrast, need speed (400 FPS GPU)
- **YOLO**: Complex backgrounds, occlusions, need orientation, willing to train custom model (85 FPS with TensorRT)

**Q: My background subtraction isn't detecting animals properly**  
A: Common fixes:
1. Adjust `THRESHOLD_VALUE` (5–15 typical)
2. Tune brightness/contrast/gamma
3. Check `MIN_OBJECT_SIZE` / `MAX_OBJECT_SIZE`
4. Enable `ENABLE_ADAPTIVE_BACKGROUND` for lighting changes
5. Use "Tune Detection" mode to visualize thresholds

**Q: YOLO is too slow, how do I speed it up?**  
A:
1. Enable TensorRT: ~3× speedup (85 vs 30 FPS)
2. Use smaller model: yolo26n instead of yolo26m
3. Reduce `RESIZE_FACTOR`: 0.5–0.75
4. Increase `TENSORRT_MAX_BATCH_SIZE`: 16–32 if VRAM allows

**Q: Can I train a custom YOLO model?**  
A: Yes! See [YOLO Detection Guide](docs/yolo_detection_guide.md) for dataset generation and training instructions. The GUI can export labeled frames.

### GPU Acceleration

**Q: How do I know if GPU acceleration is working?**  
A: Check the log on startup:
```
[INFO] GPU Device: NVIDIA GeForce RTX 3080
[INFO] CuPy available: Yes
[INFO] TensorRT available: Yes
```

**Q: I have a GPU but CuPy isn't working**  
A: Likely CUDA version mismatch. Install GPU environment:
```bash
mamba env create -f environment-gpu.yml
conda activate multi-animal-tracker-gpu
uv pip install -v --pre -f https://pip.cupy.dev/pre -r requirements-gpu.txt
```

**Q: TensorRT isn't working or gives errors**  
A:
1. Check CUDA/cuDNN compatibility: TensorRT 10.x needs CUDA 12–13
2. Try rebuilding cache: Delete `.tensorrt_cache/` folder
3. Reduce `TENSORRT_MAX_BATCH_SIZE` if VRAM limited
4. See [Troubleshooting](docs/troubleshooting.md)

**Q: Can I use AMD GPUs or Apple Silicon?**  
A: 
- **AMD**: Experimental via ROCm (untested)
- **Apple M1/M2/M3**: Use MPS backend (PyTorch auto-detects, ~2× speedup)

### Output & Data

**Q: What units are X and Y coordinates in?**  
A: Pixels by default. Set `PIXELS_PER_MM` to convert to mm in post-processing.

**Q: What does `trajectory_id` mean vs `track_id`?**  
A:
- `track_id`: Slot number (0 to MAX_TARGETS-1), can be reused
- `trajectory_id`: Unique ID for each continuous trajectory, never reused

**Q: How do I handle lost tracks?**  
A: Enable post-processing options:
- `ENABLE_POSTPROCESSING`: Interpolate gaps
- `MAX_INTERPOLATION_GAP`: Maximum frames to interpolate
- `ENABLE_BACKWARD_TRACKING`: Fill gaps from reverse pass

**Q: Can I export individual animal videos?**  
A: Not built-in yet, but you can use the CSV + OpenCV to crop. See [Individual Analysis Guide](docs/individual_analysis_guide.md) for examples.

### Performance & Optimization

**Q: Tracking is slower than real-time, how do I speed it up?**  
A:
1. Reduce `RESIZE_FACTOR`: 0.5–0.8
2. Enable GPU: See [GPU Acceleration Guide](#-gpu-acceleration-cupy--tensorrt--mps)
3. Disable preview: Runs headless at maximum speed
4. Use background subtraction instead of YOLO
5. Close other applications to free RAM/VRAM

**Q: I'm running out of memory**  
A:
1. Lower `RESIZE_FACTOR`
2. Reduce `TENSORRT_MAX_BATCH_SIZE`
3. Disable backward tracking
4. Process video in chunks
5. Upgrade RAM/VRAM if possible

**Q: How long does processing take?**  
A: Rule of thumb:
- CPU BG subtraction: ~0.5–2× real-time
- GPU BG subtraction: ~6–12× real-time
- CPU YOLO: ~0.1× real-time (10× slower)
- GPU YOLO (TensorRT): ~2–3× real-time

### Troubleshooting

**Q: GUI won't launch / crashes on startup**  
A:
1. Check Qt installation: `python -c "from PySide6 import QtWidgets"`
2. Try minimal environment if conflicts exist
3. Check logs for missing dependencies
4. See [Troubleshooting Guide](docs/troubleshooting.md)

**Q: Bidirectional tracking gives different results than forward-only**  
A: Expected! Bidirectional tracking:
- Fills gaps from reverse pass
- Merges fragmented trajectories
- May change track IDs for better continuity
- See "Differences from Forward-Only" in [Configuration Reference](#configuration-reference)

**Q: Where can I get help?**  
A:
1. Check [Troubleshooting Guide](docs/troubleshooting.md)
2. Search [GitHub Issues](https://github.com/rutalab/multi-animal-tracker/issues)
3. Open new issue with:
   - Full error message/traceback
   - Configuration JSON
   - System info (GPU, CUDA version, etc.)
   - Minimal example to reproduce
4. Email: neurorishika@gmail.com

---

## 🔗 Links

- **Documentation**: [docs/](docs/)
- **GitHub**: https://github.com/rutalab/multi-animal-tracker
- **Issues**: https://github.com/rutalab/multi-animal-tracker/issues
- **Releases**: https://github.com/rutalab/multi-animal-tracker/releases
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

**Made with ❤️ for the behavioral neuroscience community**

