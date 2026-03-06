# Gemini CLI Project Context: multi-animal-tracker

This file provides project-specific context and instructions for Gemini CLI when working on the `multi-animal-tracker` repository.

## Project Overview
A high-performance Python toolkit for multi-animal tracking and pose estimation. It features a GUI for tracking (MAT), a pose-labeling tool (PoseKit), and advanced classification utilities (ClassKit).

- **Core Tech Stack:** Python (3.11-3.13), PySide6 (Qt6), PyTorch, Ultralytics (YOLO), OpenCV, Numba (CPU acceleration), Faiss (vector search).
- **Package Name:** `multi_tracker` (located in `src/`).

## Architecture & Module Map

| Layer | Package | Role |
|---|---|---|
| **Launcher** | `multi_tracker.app` | Entry points (`mat`, `multi-animal-tracker`). |
| **MAT GUI** | `multi_tracker.gui` | Tracking dashboard, widgets, and dialogs. |
| **Core** | `multi_tracker.core` | Tracking logic: `detectors`, `filters` (Kalman), `assigners` (Hungarian), `post` (processing). |
| **Data** | `multi_tracker.data` | I/O, CSV export, detection cache (`.npz`). |
| **PoseKit** | `multi_tracker.posekit` | Pose labeling sub-application (`posekit-labeler`). |
| **ClassKit** | `multi_tracker.classkit` | Classification/embedding toolkit (Active development). |
| **Training** | `multi_tracker.training` | Model training runners and registry. |
| **Tools** | `multi_tracker.tools` | Standalone utilities like `datasieve`. |
| **Utils** | `multi_tracker.utils` | GPU/Runtime detection, geometry, image processing. |

### Tracking Pipeline Flow
1. **Source:** Video frames read (via `utils/frame_prefetcher.py`).
2. **Detection:** `core/detectors/engine.py` generates measurements.
3. **Estimation:** `core/filters/kalman.py` predicts/updates states.
4. **Association:** `core/assigners/hungarian.py` matches detections to tracks.
5. **Orchestration:** `core/tracking/worker.py` (TrackingWorker) manages the loop.
6. **Export:** `data/csv_writer.py` persists results.

## Development Workflow

### 1. Environment Setup
The project uses platform-specific conda environments. Always use `mamba` (preferred) or `conda` with `uv` for fast installs.

```bash
# Setup (Pick one)
make setup            # CPU (NumPy + Numba)
make setup-mps        # Apple Silicon (M1/M2/M3/M4)
make setup-cuda       # NVIDIA GPU (defaults to CUDA 13)
make setup-rocm       # AMD GPU (ROCm 6.0+)

# Install (After activating env)
make install          # or install-mps / install-cuda / install-rocm
make install-dev      # Mandatory for formatting/linting/audit
```

### 2. Common CLI Commands
- **MAT GUI:** `mat`
- **PoseKit Labeler:** `posekit-labeler`
- **DataSieve:** `datasieve`
- **ClassKit:** `classkit-labeler`

### 3. Testing & Validation
- **Run all tests:** `make pytest`
- **Coverage (Terminal):** `make test-cov`
- **Coverage (HTML):** `make test-cov-html` -> `htmlcov/index.html`
- **Benchmarks:** Excluded by default. Run with `pytest -m benchmark`.

### 4. Code Quality & Standards
Strict adherence to formatting and linting is required for CI success.

- **Formatting:** `make format` (autopep8 -> black -> isort).
- **Linting:** `make lint` (Moderate - recommended gate).
- **Strict Linting:** `make lint-strict`.
- **Type Checking:** `make type-check` (mypy).
- **Full Audit:** `make audit` (Dead code + dep-graph + types + coverage).

## Engineering Standards & Rules

### Core Principles
- **GUI/Core Decoupling:** Tracking logic (`core/`) must never import from or depend on `gui/` widgets.
- **Compute Runtime:** Use the `compute_runtime` system (`core/runtime/compute_runtime.py`) for all compute-heavy operations (supports `cpu`, `cuda`, `mps`, `rocm`, `onnx`).
- **Terminology:** Use `labeler` (one 'l'), never `labeller`. Specifically: `posekit-labeler`.
- **Detection Cache:** Backward tracking relies on `.npz` detection caches. Do not break the cache schema.
- **Legacy Policy:** Superseded code goes to `legacy/` for one release cycle before deletion. Never import from `legacy/`.

### Coding Style
- **Black:** 88 character line limit.
- **Imports:** Sorted by `isort` (Standard -> 3rd Party -> Local).
- **Type Hints:** Required for all new public API methods in `core/` and `data/`.

## Key Files for Reference
- `pyproject.toml`: Dependencies and entry points.
- `makefile`: All dev task commands.
- `src/multi_tracker/core/tracking/worker.py`: The "brain" of the tracking loop.
- `to_fix.md`: Known technical debt and false-positive audit findings.
- `docs/developer-guide/runtime-integration.md`: Checklist for adding new models/backends.
