# CUDA Dependency Updates Tracker

This page tracks workarounds and pinned dependencies that depend on external packages improving their CUDA 13 support. As package maintainers release updated versions with better CUDA compatibility, the items below should be revisited and updated.

## Current Workarounds

### FAISS GPU Wheel for CUDA 13

**Status**: ⚠️ Requires Action

**Current Workaround**: Use `faiss-cpu` in `requirements-cuda13.txt`

**File**: `requirements-cuda13.txt`

**Issue**:

- FAISS GPU wheels are unavailable for CUDA 13 + Python 3.13 due to limited build coverage from Meta
- CPU variant is a functional fallback but lacks GPU acceleration for vector similarity search

**Action Required When Fixed**:

1. Monitor [FAISS releases](https://github.com/facebookresearch/faiss/releases) for CUDA 13 + Python 3.13 wheel availability
2. Test `faiss-gpu` with CUDA 13.x in CI environment
3. Replace `faiss-cpu` with `faiss-gpu` in `requirements-cuda13.txt`
4. Update documentation in `docs/getting-started/installation.md` and `docs/getting-started/environments.md` to remove the fallback note

**Related Issue**: Meta/FAISS issue tracker for CUDA 13 wheel builds

---

### ONNX Runtime GPU Version Pinning

**Status**: 📌 Version Pinned

**Current Pin**: `onnxruntime-gpu==1.24.1` in `requirements-cuda.txt`

**File**: `requirements-cuda.txt`

**Reason for Pinning**:

- Version 1.24.1 is known to work with CUDA 12 user-space library linkage (libcublasLt.so.12, etc.) across both CUDA 12.x and 13.x environments
- Later versions may have changed their CUDA 12 binary compatibility or introduced stricter version requirements

**Action Required When Fixed**:

1. Monitor [ONNX Runtime releases](https://github.com/onnx/onnx-runtime/releases) for CUDA 12–13 compatibility improvements
2. Test newer versions (1.25.x, 1.26.x+) with:
   - CUDA 12.x environments
   - CUDA 13.x environments (with conda CUDA 12 runtime libs for linkage)
   - CPU provider fallback behavior
3. If newer versions offer better compatibility or performance, update to `onnxruntime-gpu>=1.24.1,<2.0` (or specific newer pin)
4. Update CI/CD testing to verify linkage across versions
5. Document the upgrade path in the changelog

**Related Issue**: Check ONNX Runtime issues for "CUDA 13" and "CUDA compatibility"

---

### CuPy Prerelease on CUDA 13

**Status**: 🔧 Prerelease Required

**Current Configuration**: Uses `--pre` flag and `https://pip.cupy.dev/pre` in `requirements-cuda13.txt`

**File**: `requirements-cuda13.txt`

**Reason for Prerelease**:

- CUDA 13 stable `cupy-cuda13x` wheels may not be available from the main PyPI index
- Prerelease wheels from CuPy's development server ensure CUDA 13 support

**Action Required When Fixed**:

1. Monitor [CuPy releases](https://pypi.org/project/cupy-cuda13x/) for CUDA 13 stable wheel availability
2. When stable wheels are published:
   - Remove `--pre` flag from `requirements-cuda13.txt`
   - Remove custom index URL `https://pip.cupy.dev/pre`
   - Update to pinned stable version (e.g., `cupy-cuda13x==X.Y.Z`)
3. Test in CI with stable index to confirm compatibility
4. Update `docs/getting-started/installation.md` to note that CUDA 13 installation is now fully stable

**Related Issue**: Monitor [CuPy GitHub Releases](https://github.com/cupy/cupy/releases) for CUDA 13 stable tag

---

### TensorRT Version-Specific Pins

**Status**: ✅ Versioned, May Improve

**Current Configuration**: Separate `tensorrt-cu12`, `tensorrt-cu13` (unversioned) in version-specific files

**Files**: `requirements-cuda12.txt`, `requirements-cuda13.txt`

**Considerations**:

- If NVIDIA publishes unified TensorRT CPU wheels with better version compatibility, we may simplify to a single tensorrt package
- CUDA 13 support may improve with new releases; consider stricter pinning if compatibility issues emerge

**Action Required When Fixed**:

1. Monitor [NVIDIA TensorRT releases](https://github.com/NVIDIA/TensorRT/releases) for unified CUDA version support
2. If NVIDIA publishes CPU/GPU-agnostic wheels:
   - Move TensorRT to `requirements-cuda.txt` (shared base)
   - Remove version-specific TensorRT packages
   - Simplify requirements structure
3. If CUDA 13 TensorRT stability improves, consider pinning to a specific version range

**Related Issue**: NVIDIA TensorRT issue tracker

---

### PyTorch Index URLs

**Status**: 📦 Version-Specific, Monitor

**Current Configuration**:

- CUDA 12: `--extra-index-url https://download.pytorch.org/whl/cu128`
- CUDA 13: `--extra-index-url https://download.pytorch.org/whl/cu130`

**Files**: `requirements-cuda12.txt`, `requirements-cuda13.txt`

**Considerations**:

- PyTorch uses a custom index URL distribution strategy; check for future changes to their build/distribution infrastructure
- CUDA 13 might be integrated into the main PyPI wheels in future major versions, eliminating the need for custom indexes

**Action Required When Fixed**:

1. Monitor [PyTorch installation docs](https://pytorch.org/get-started/locally/) for index URL changes
2. If PyTorch integrates CUDA variants into standard PyPI:
   - Remove custom index URLs from both CUDA requirement files
   - Simplify to pinned PyTorch versions (e.g., `torch>=2.1.0`)
3. Update CI to verify installation without custom indexes

---

## Monitoring Checklist

When updating dependencies:

- [ ] Check FAISS releases for CUDA 13 wheel availability
- [ ] Review ONNX Runtime release notes and test compatibility
- [ ] Monitor CuPy stable CUDA 13 wheel status
- [ ] Evaluate TensorRT unified wheel roadmap
- [ ] Check PyTorch for index URL or distribution strategy changes
- [ ] Run full CI/CD suite with new dependencies
- [ ] Update this document with resolved items

## Summary Table

| Package | Current Fix | File | Priority | Check Frequency |
| --- | --- | --- | --- | --- |
| FAISS | `faiss-cpu` fallback | requirements-cuda13.txt | High | Monthly |
| ONNX Runtime | Version pin 1.24.1 | requirements-cuda.txt | High | Quarterly |
| CuPy | Prerelease CUDA 13 | requirements-cuda13.txt | Medium | Monthly |
| TensorRT | Version-specific import | requirements-cudaX.txt | Medium | Quarterly |
| PyTorch | Custom index URLs | requirements-cudaX.txt | Low | Quarterly |

## How to Report Resolved Items

When any of these items is resolved:

1. Update the relevant requirement file
2. Update this document, marking the item as ✅ **Resolved**
3. Update related documentation (installation.md, environments.md)
4. Add a note to `CHANGELOG.md` under the new release
5. Remove the resolved item from the monitoring checklist
