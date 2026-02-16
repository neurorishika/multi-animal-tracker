# Metal-Accelerated FAISS for macOS

ClassKit can use Metal-accelerated FAISS (`metalfaiss`) on Apple Silicon Macs for much faster clustering without the segfault issues of standard FAISS.

## Why Metal FAISS?

- **Fast**: GPU-accelerated clustering using Apple's Metal API
- **Stable**: No segfault issues like standard FAISS on macOS
- **Native**: Optimized for Apple Silicon (M1/M2/M3)

## Installation

### Option 1: Using pip (Recommended)

```bash
# Activate your environment
conda activate multi-animal-tracker-mps

# Install MLX and metalfaiss
pip install mlx
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/MLXPorts/Faiss-mlx.git
cd Faiss-mlx/python

# Install
pip install -e .
```

## Verification

Test that it works:

```python
import mlx.core as mx
from metalfaiss.clustering import AnyClustering, ClusteringParameters
import numpy as np

# Create test data
data = np.random.randn(1000, 128).astype(np.float32).tolist()

# Test clustering
params = ClusteringParameters(max_iterations=20)
kmeans = AnyClustering.new(d=128, k=10, parameters=params)
kmeans.train(data)
centers = kmeans.centroids()
print("✓ Metal FAISS working!")
print(f"Centroids shape: {centers.shape}")
```

## Fallback Behavior

ClassKit will automatically detect and use the best available clustering backend:

1. **macOS**:
   - Try `metalfaiss` (Metal-accelerated) ← **Best option**
   - Fallback to `scikit-learn` (CPU-based, slower but stable)
   - Never use standard FAISS (causes segfaults)

2. **Linux/Windows**:
   - Use standard `faiss-cpu` or `faiss-gpu`
   - Fallback to `scikit-learn` if FAISS unavailable

## Performance Comparison

For clustering 10,000 vectors into 500 clusters:

| Backend | Time | Notes |
|---------|------|-------|
| metalfaiss (Metal) | ~2s | Fast, GPU-accelerated |
| sklearn MiniBatch | ~15s | Slower, CPU-based |
| faiss-cpu (macOS) | ❌ | Segmentation fault |

## Troubleshooting

### Import Error: No module named 'mlx'

```bash
pip install mlx mlx-nn
```

### Import Error: No module named 'metalfaiss'

```bash
# Install from local Faiss-mlx directory
cd /path/to/Faiss-mlx/python
pip install -e .
```

### Still getting segfaults?

The app will automatically fallback to sklearn. You can force sklearn by setting:

```python
# In ClusterDialog, GPU checkbox will show "CPU (sklearn)" on macOS
```

## API Reference

The `metalfaiss` package provides:

- `metalfaiss.clustering.AnyClustering` - Main clustering class
- `metalfaiss.clustering.ClusteringParameters` - Configuration dataclass
- `metalfaiss.clustering.kmeans_clustering()` - Simple function interface

### Example Usage

```python
from metalfaiss.clustering import AnyClustering, ClusteringParameters
import numpy as np

# Prepare data (must be List[List[float]])
data = np.random.randn(1000, 128).astype(np.float32)
data_list = data.tolist()

# Configure clustering
params = ClusteringParameters(
    max_iterations=20,
    tolerance=1e-4,
    spherical=False,
    seed=42
)

# Create and train
kmeans = AnyClustering.new(d=128, k=100, parameters=params)
kmeans.train(data_list)

# Get results
centroids = kmeans.centroids()  # Returns mlx.array
centroids_np = np.array(centroids)  # Convert to numpy if needed

# Compute assignments manually
import mlx.core as mx
embeddings_mlx = mx.array(data)
centroids_mlx = centroids
dists = mx.sum(mx.square(mx.subtract(embeddings_mlx[:, None], centroids_mlx[None])), axis=2)
assignments = np.array(mx.argmin(dists, axis=1))

print(f"Created {len(np.unique(assignments))} clusters from {len(data)} points")
```

## References

- metalfaiss (Faiss-mlx): https://github.com/MLXPorts/Faiss-mlx
- MLX: https://github.com/ml-explore/mlx
- Standard FAISS: https://github.com/facebookresearch/faiss
