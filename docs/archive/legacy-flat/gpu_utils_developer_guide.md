# GPU Utils Developer Guide

## Quick Start

The `multi_tracker.utils.gpu_utils` module provides centralized GPU and acceleration detection for the entire codebase.

### Basic Usage

#### Check if GPU is available
```python
from multi_tracker.utils import GPU_AVAILABLE

if GPU_AVAILABLE:
    print("Using GPU acceleration!")
else:
    print("Using CPU")
```

#### Check specific GPU types
```python
from multi_tracker.utils import CUDA_AVAILABLE, MPS_AVAILABLE

if CUDA_AVAILABLE:
    print("NVIDIA CUDA GPU available")
elif MPS_AVAILABLE:
    print("Apple Silicon MPS available")
```

#### Get detailed device information
```python
from multi_tracker.utils import get_device_info

info = get_device_info()
print(f"CUDA: {info['cuda_available']}")
print(f"MPS: {info['mps_available']}")
print(f"CUDA Devices: {info['cuda_device_count']}")
print(f"PyTorch Version: {info['torch_version']}")
```

## Available Flags

All flags are Boolean (`True`/`False`) and automatically detected on module import:

| Flag | Description | Use Case |
|------|-------------|----------|
| `CUDA_AVAILABLE` | CuPy available for NVIDIA GPUs | Array operations with CuPy |
| `MPS_AVAILABLE` | PyTorch MPS for Apple Silicon | PyTorch operations on M1/M2/M3 |
| `TORCH_CUDA_AVAILABLE` | PyTorch with CUDA support | PyTorch operations on NVIDIA |
| `CUPY_AVAILABLE` | Alias for `CUDA_AVAILABLE` | Alternative name |
| `TORCH_AVAILABLE` | PyTorch available (any backend) | Any PyTorch operations |
| `NUMBA_AVAILABLE` | Numba JIT for CPU acceleration | CPU-based JIT compilation |
| `GPU_AVAILABLE` | Any GPU (CUDA or MPS) | Generic GPU check |
| `ANY_ACCELERATION` | Any acceleration (GPU or Numba) | Performance optimization check |

## Available Utility Functions

### `get_device_info() -> dict`

Returns comprehensive device information:

```python
{
    'cuda_available': bool,
    'cuda_device_count': int,
    'cupy_version': str or None,
    'mps_available': bool,
    'torch_available': bool,
    'torch_cuda_available': bool,
    'torch_version': str or None,
    'numba_available': bool,
    'numba_version': str or None,
    'gpu_available': bool,
    'any_acceleration': bool
}
```

**Example:**
```python
from multi_tracker.utils.gpu_utils import get_device_info

info = get_device_info()
if info['cuda_available']:
    print(f"Found {info['cuda_device_count']} CUDA device(s)")
    print(f"CuPy version: {info['cupy_version']}")
```

### `log_device_info() -> None`

Pretty-prints device availability to console:

```python
from multi_tracker.utils.gpu_utils import log_device_info

log_device_info()
```

**Output example:**
```
=== GPU & Acceleration Status ===
CUDA (NVIDIA):     ✓ Available (2 devices, CuPy 12.3.0)
MPS (Apple):       ✗ Not Available
Numba (CPU JIT):   ✓ Available (0.58.1)
Overall:           GPU acceleration enabled (CUDA)
================================
```

### `get_optimal_device(enable_gpu=True, prefer_cuda=True) -> str`

Auto-selects best available device for PyTorch operations:

```python
from multi_tracker.utils.gpu_utils import get_optimal_device

device = get_optimal_device(enable_gpu=True, prefer_cuda=True)
# Returns: "cuda:0", "mps", or "cpu"

# Use with PyTorch
import torch
model = model.to(device)
```

**Parameters:**
- `enable_gpu` (bool): Whether to use GPU if available (default: True)
- `prefer_cuda` (bool): Prefer CUDA over MPS if both available (default: True)

**Selection hierarchy:**
1. If `enable_gpu=False`: Returns "cpu"
2. If CUDA available and `prefer_cuda=True`: Returns "cuda:0"
3. If MPS available: Returns "mps"
4. If CUDA available and `prefer_cuda=False`: Returns "cuda:0"
5. Otherwise: Returns "cpu"

## Module References

The `gpu_utils` module also exports the actual modules/functions, pre-imported with fallbacks:

| Export | Type | Description | Fallback if unavailable |
|--------|------|-------------|------------------------|
| `cp` | module | CuPy (NumPy-compatible GPU arrays) | `None` |
| `cupy_ndimage` | module | CuPy ndimage (like scipy.ndimage) | `None` |
| `torch` | module | PyTorch | `None` |
| `F` | module | PyTorch functional API | `None` |
| `njit` | decorator | Numba JIT compiler | Dummy decorator (no-op) |
| `prange` | function | Numba parallel range | Regular `range` |

### Safe Usage Pattern

```python
from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, cp

if CUDA_AVAILABLE and cp is not None:
    # Use CuPy
    array = cp.array([1, 2, 3])
    result = cp.sum(array)
else:
    # Fall back to NumPy
    import numpy as np
    array = np.array([1, 2, 3])
    result = np.sum(array)
```

### Using Numba with Fallback

```python
from multi_tracker.utils.gpu_utils import njit, prange

@njit(parallel=True)
def fast_function(n):
    total = 0
    for i in prange(n):  # Parallel if Numba available, serial otherwise
        total += i
    return total

result = fast_function(1000000)
```

Note: If Numba is not available, `njit` becomes a no-op decorator and `prange` is just `range`.

## Common Patterns

### Pattern 1: GPU-Aware Array Operations

```python
from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, MPS_AVAILABLE, cp, torch, F
import numpy as np

def process_image(image: np.ndarray) -> np.ndarray:
    """Process image using best available acceleration."""

    if CUDA_AVAILABLE and cp is not None:
        # Use CuPy for NVIDIA GPU
        img_gpu = cp.asarray(image)
        result = cp.asnumpy(some_cupy_operation(img_gpu))
        return result

    elif MPS_AVAILABLE and torch is not None:
        # Use PyTorch MPS for Apple Silicon
        img_tensor = torch.from_numpy(image).to('mps')
        result = some_torch_operation(img_tensor).cpu().numpy()
        return result

    else:
        # CPU fallback
        return some_numpy_operation(image)
```

### Pattern 2: Conditional Import

```python
from multi_tracker.utils.gpu_utils import GPU_AVAILABLE

if GPU_AVAILABLE:
    from .gpu_optimized_module import GpuProcessor as Processor
else:
    from .cpu_module import CpuProcessor as Processor

# Use Processor regardless of GPU availability
processor = Processor()
```

### Pattern 3: Dynamic Model Device Selection

```python
from multi_tracker.utils.gpu_utils import get_optimal_device
import torch

class MyModel(torch.nn.Module):
    def __init__(self, enable_gpu=True):
        super().__init__()
        self.device = get_optimal_device(enable_gpu=enable_gpu)

        # Build model
        self.conv1 = torch.nn.Conv2d(3, 64, 3)
        # ... more layers

        # Move to device
        self.to(self.device)

    def forward(self, x):
        # Input automatically on correct device
        return self.conv1(x)
```

### Pattern 4: Logging Device Info at Startup

```python
from multi_tracker.utils.gpu_utils import log_device_info
import logging

def main():
    logging.info("Initializing Multi-Animal-Tracker")
    log_device_info()  # Show user their GPU capabilities

    # Continue with application
    # ...
```

## Integration Examples

### Example 1: Background Subtraction Module

Before (duplicated detection):
```python
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    import torch
    MPS_AVAILABLE = torch.backends.mps.is_available()
except ImportError:
    MPS_AVAILABLE = False
```

After (centralized):
```python
from ..utils.gpu_utils import (
    CUDA_AVAILABLE,
    MPS_AVAILABLE,
    cp,
    torch,
    F
)
```

### Example 2: YOLO Device Selection

Before (manual checks):
```python
import torch

def _detect_device(self):
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

After (using gpu_utils):
```python
from ..utils.gpu_utils import TORCH_CUDA_AVAILABLE, MPS_AVAILABLE

def _detect_device(self):
    if TORCH_CUDA_AVAILABLE:
        return "cuda:0"
    elif MPS_AVAILABLE:
        return "mps"
    else:
        return "cpu"
```

Or even simpler:
```python
from ..utils.gpu_utils import get_optimal_device

def _detect_device(self):
    return get_optimal_device(enable_gpu=True)
```

### Example 3: GUI Device Dropdown

Before (hardcoded):
```python
device_options = ["auto", "cpu", "cuda:0", "mps"]
```

After (dynamic):
```python
from ..utils.gpu_utils import TORCH_CUDA_AVAILABLE, MPS_AVAILABLE

device_options = ["auto", "cpu"]
if TORCH_CUDA_AVAILABLE:
    device_options.append("cuda:0")
if MPS_AVAILABLE:
    device_options.append("mps")
```

## Testing GPU Utils

### Test on Different Systems

**NVIDIA System:**
```bash
python -c "from multi_tracker.utils import get_device_info, log_device_info; log_device_info()"
# Expected: CUDA available, device count > 0
```

**Apple Silicon:**
```bash
python -c "from multi_tracker.utils import get_device_info, log_device_info; log_device_info()"
# Expected: MPS available
```

**CPU-Only:**
```bash
python -c "from multi_tracker.utils import get_device_info, log_device_info; log_device_info()"
# Expected: Only Numba available (if installed)
```

### Unit Test Example

```python
import unittest
from multi_tracker.utils.gpu_utils import (
    get_device_info,
    get_optimal_device,
    GPU_AVAILABLE,
    CUDA_AVAILABLE,
    MPS_AVAILABLE
)

class TestGPUUtils(unittest.TestCase):

    def test_device_info_structure(self):
        """Test that device info returns expected keys."""
        info = get_device_info()

        required_keys = [
            'cuda_available', 'mps_available', 'torch_available',
            'numba_available', 'gpu_available', 'any_acceleration'
        ]

        for key in required_keys:
            self.assertIn(key, info)
            self.assertIsInstance(info[key], bool)

    def test_optimal_device_returns_string(self):
        """Test that optimal device returns a valid string."""
        device = get_optimal_device()
        self.assertIsInstance(device, str)
        self.assertIn(device, ["cuda:0", "mps", "cpu"])

    def test_gpu_flag_consistency(self):
        """Test that GPU_AVAILABLE matches CUDA or MPS."""
        self.assertEqual(GPU_AVAILABLE, CUDA_AVAILABLE or MPS_AVAILABLE)
```

## Best Practices

### DO ✓

1. **Import from `gpu_utils`** - Use centralized detection
   ```python
   from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE
   ```

2. **Check flags before using modules** - Ensure module is available
   ```python
   if CUDA_AVAILABLE and cp is not None:
       use_cupy()
   ```

3. **Use `get_optimal_device()`** - Let the system choose
   ```python
   device = get_optimal_device()
   ```

4. **Provide CPU fallbacks** - Always have a fallback path
   ```python
   if GPU_AVAILABLE:
       fast_path()
   else:
       cpu_path()
   ```

### DON'T ✗

1. **Don't duplicate detection** - Never re-implement GPU checks
   ```python
   # BAD - duplicates detection
   try:
       import torch
       has_cuda = torch.cuda.is_available()
   except:
       has_cuda = False
   ```

2. **Don't import modules directly** - Use gpu_utils exports
   ```python
   # BAD - bypasses centralized detection
   import cupy as cp

   # GOOD - uses centralized detection
   from multi_tracker.utils.gpu_utils import cp, CUDA_AVAILABLE
   ```

3. **Don't assume GPU is available** - Always check flags
   ```python
   # BAD - assumes CUDA
   import cupy as cp
   array = cp.array([1, 2, 3])  # Crashes if no GPU

   # GOOD - checks first
   from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, cp
   if CUDA_AVAILABLE:
       array = cp.array([1, 2, 3])
   ```

4. **Don't hardcode device strings** - Use dynamic detection
   ```python
   # BAD - hardcoded
   device_options = ["cuda:0", "mps", "cpu"]

   # GOOD - dynamic
   from multi_tracker.utils.gpu_utils import TORCH_CUDA_AVAILABLE, MPS_AVAILABLE
   device_options = ["cpu"]
   if TORCH_CUDA_AVAILABLE:
       device_options.insert(0, "cuda:0")
   if MPS_AVAILABLE:
       device_options.insert(0, "mps")
   ```

## Adding New GPU Backends

To add support for a new GPU backend (e.g., AMD ROCm, Intel OneAPI):

1. **Edit `gpu_utils.py`** - Add detection logic
   ```python
   # AMD ROCm detection
   try:
       import torch
       ROCM_AVAILABLE = torch.version.hip is not None
   except (ImportError, AttributeError):
       ROCM_AVAILABLE = False
   ```

2. **Export the flag** - Add to `__all__`
   ```python
   __all__ = [
       # ... existing flags
       'ROCM_AVAILABLE',
   ]
   ```

3. **Update `get_device_info()`** - Include in device info
   ```python
   def get_device_info():
       return {
           # ... existing fields
           'rocm_available': ROCM_AVAILABLE,
       }
   ```

4. **Update `get_optimal_device()`** - Add to selection logic
   ```python
   def get_optimal_device(...):
       # ... existing logic
       if ROCM_AVAILABLE:
           return "rocm:0"
   ```

5. **That's it!** All modules importing from `gpu_utils` automatically get ROCm support.

## Troubleshooting

### "Module 'cp' is None"

**Problem:** Trying to use CuPy when not available
```python
from multi_tracker.utils.gpu_utils import cp
array = cp.array([1, 2, 3])  # AttributeError: 'NoneType' has no attribute 'array'
```

**Solution:** Always check availability first
```python
from multi_tracker.utils.gpu_utils import CUDA_AVAILABLE, cp

if CUDA_AVAILABLE and cp is not None:
    array = cp.array([1, 2, 3])
else:
    import numpy as np
    array = np.array([1, 2, 3])
```

### "GPU shows available but operations fail"

**Problem:** Flag is True but operations crash

**Possible causes:**
1. Out of GPU memory - reduce batch size
2. Driver mismatch - update GPU drivers
3. Module version mismatch - reinstall packages

**Debug:**
```python
from multi_tracker.utils.gpu_utils import get_device_info
info = get_device_info()
print(f"CUDA devices: {info['cuda_device_count']}")
print(f"CuPy version: {info['cupy_version']}")
print(f"PyTorch version: {info['torch_version']}")
```

### "Want to force CPU mode for testing"

**Solution:** Use `get_optimal_device(enable_gpu=False)`
```python
device = get_optimal_device(enable_gpu=False)  # Always returns "cpu"
```

## Summary

The `gpu_utils` module provides:
- ✓ **Centralized GPU detection** - One source of truth
- ✓ **Multi-platform support** - NVIDIA, Apple, CPU
- ✓ **Safe fallbacks** - Graceful degradation
- ✓ **Easy integration** - Simple imports
- ✓ **Rich information** - Version numbers, device counts
- ✓ **Future-proof** - Easy to extend

Always prefer using `gpu_utils` over manual GPU detection to maintain code consistency and reliability.
