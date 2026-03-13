# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Environment Setup

Platform-specific conda environments are used. Choose one:

```bash
# Create environment (pick your platform)
make setup            # CPU / NumPy+Numba
make setup-mps        # Apple Silicon (M1/M2/M3/M4)
make setup-cuda       # NVIDIA GPU (CUDA 12 or 13)
make setup-rocm       # AMD GPU (ROCm 6.0+)

# After activating the environment, install runtime packages
make install          # or install-mps / install-cuda / install-rocm

# Install dev/audit tools
make install-dev

# Install docs tools
make docs-install
```

Environment names: `multi-animal-tracker`, `multi-animal-tracker-mps`, `multi-animal-tracker-cuda`, `multi-animal-tracker-rocm`.

## Running Tests

```bash
make pytest                # Run all tests
python -m pytest tests/test_<name>.py         # Run a single test file
python -m pytest tests/test_<name>.py::test_fn  # Run a single test function
make test-cov              # Tests + terminal coverage
make test-cov-html         # Tests + HTML coverage (htmlcov/index.html)
```

pytest is configured in `pyproject.toml`. Test files are in `tests/`. Benchmarks are excluded by default (`-m "not benchmark"`).

## Launching the Applications

```bash
mat                    # Multi-Animal Tracker GUI
posekit-labeler        # PoseKit pose-labeling GUI
datasieve              # DataSieve tool
classkit-labeler       # ClassKit labeler
```

## Code Quality

```bash
make format            # autopep8 → black → isort (do this before committing)
make format-check      # Check formatting without changing files
make lint              # Moderate-severity flake8 (recommended gate)
make lint-fix          # Auto-fix with ruff then reformat
make lint-strict       # Strict flake8
make lint-report       # Side-by-side issue counts at all three levels
```

**Pre-PR checklist** (from `docs/developer-guide/contributing.md`):
```bash
make format
make lint
make docs-check        # docs-build + quality metrics + terminology check
```

For large PRs, also run `make audit` (dead-code + dep-graph + type-check + coverage).

## Documentation

```bash
make docs-serve        # Live preview at http://127.0.0.1:8000
make docs-build        # Strict mkdocs build
make docs-check        # Build + quality metrics + terminology check
make techref-build     # Build LaTeX technical reference
```

Terminology rule: use `posekit-labeler` (not `posekit-labeller`) and `multi_tracker.posekit` (not legacy names).

## Code Health Tools

```bash
make dead-code         # Find unused code (vulture, ≥80% confidence)
make dead-code-whitelist  # Generate vulture_whitelist.py for false-positive review
make dep-graph         # Visual SVG dependency graph → multi_tracker.svg
make dep-graph-text    # Text module map via pyreverse
make type-check        # mypy static type checking (lenient mode)
make audit             # Full sweep: all of the above + coverage
```

See `to_fix.md` for known dead-code findings and the rationale for false-positive exclusions (Qt dynamic dispatch, PyTorch `forward`, `classkit/` stubs, etc.).

---

## Architecture

### System Layers

| Layer | Package | Role |
|---|---|---|
| Launcher / CLI | `multi_tracker.app` | Entry points and bootstrap |
| MAT GUI | `multi_tracker.gui` | Tracking GUI, dialogs, widgets |
| Core | `multi_tracker.core` | Detection, Kalman filter, assignment, post-processing, identity |
| Data | `multi_tracker.data` | CSV export, detection cache, dataset generation/merge |
| PoseKit | `multi_tracker.posekit` | Pose-labeling application (separate UI surface, shared env) |
| ClassKit | `multi_tracker.classkit` | Classification/embedding toolkit (active development) |
| Training | `multi_tracker.training` | Dataset builders, training runner, registry, service |
| Tools | `multi_tracker.tools` | Standalone tools (datasieve, etc.) |
| Utils | `multi_tracker.utils` | GPU detection, geometry, image processing, batching, prefetch |

**Key boundary rules:**
- Core tracking must not depend on GUI widget internals.
- Data layer must be reusable from both GUI and scripts.
- PoseKit is a separate app surface with its own workflow.

### MAT Tracking Pipeline

1. Video frames read (optionally prefetched via `utils`)
2. Detector generates measurements (`core/detectors/engine.py`)
3. Kalman filter predicts and update track state (`core/filters/kalman.py`)
4. Hungarian or greedy assignment (`core/assigners/hungarian.py`)
5. TrackingWorker emits frame/status/metrics to GUI (`core/tracking/worker.py`)
6. Trajectories written to CSV (`data/csv_writer.py`)
7. Optional backward pass reuses detection cache (`.npz`)
8. Post-processing resolves, relinks, and interpolates (`core/post/processing.py`)

Detection cache (`.npz`) enables backward tracking, reproducible reruns, and pose precompute. Backward mode refuses to run without a valid cache.

### Kalman State Vector

```
state = [x, y, theta, vx, vy]   # position, heading, velocity
measurement = [x, y, theta]
```

Process noise is anisotropic (longitudinal vs. lateral). Young tracks have attenuated velocity (`KALMAN_MATURITY_AGE`, `KALMAN_INITIAL_VELOCITY_RETENTION`).

### Identity / Runtime System

All compute-heavy methods use a single `compute_runtime` setting. Runtime support logic is centralized in:

- `src/multi_tracker/core/runtime/compute_runtime.py`
- `src/multi_tracker/utils/gpu_utils.py`

Canonical runtimes: `cpu`, `mps`, `cuda`, `rocm`, `onnx_cpu`, `onnx_cuda`, `onnx_rocm`, `tensorrt`.

When adding a new model/method: define a pipeline key, add capability rules in `_pipeline_supports_runtime()`, add runtime translation, wire UI intersection gating. See `docs/developer-guide/runtime-integration.md` for the full checklist.

### Extension Points

- **New detector**: implement `detect_objects`-compatible output in `core/detectors/engine.py`
- **New identity method**: extend `core/identity/analysis.py`; preserve the crop extraction metadata and output contract
- **New runtime pipeline**: follow the checklist in `docs/developer-guide/runtime-integration.md`

### Key Source Files for Auditing Behavior

- `src/multi_tracker/core/tracking/worker.py` — top-level orchestrator
- `src/multi_tracker/core/filters/kalman.py`
- `src/multi_tracker/core/assigners/hungarian.py`
- `src/multi_tracker/core/post/processing.py`
- `src/multi_tracker/core/identity/runtime_api.py`
- `src/multi_tracker/core/runtime/compute_runtime.py`

### `legacy/` Policy

Superseded code is moved to `legacy/` for one release cycle before deletion. `legacy/` is excluded from tests, mypy, and formatters. Never import from `legacy/` in `src/` or `tests/`.

### PoseKit Pipeline

1. Image set + project metadata loaded
2. Annotation state edited in UI (`posekit/ui/`)
3. Labels persisted to YOLO pose format
4. Optional model-assisted inference (YOLO pose, SLEAP) and split-generation steps

PoseKit inference uses the same `compute_runtime` system and the same ONNX/TensorRT artifact auto-management pattern as MAT.
