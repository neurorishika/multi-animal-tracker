# Source Reorganization Design

**Date:** 2026-03-30
**Status:** Approved
**Scope:** Structural reorganization + boundary cleanup (option B)

## Problem

The `multi_tracker/` package contains 5 independent applications (MAT, PoseKit,
ClassKit, Afterhours, DataSieve) but MAT's code is "ambient" — its `gui/`,
`app/` live at the package root alongside satellite apps, making it unclear what
belongs to MAT vs. what is shared infrastructure. DataSieve is buried under a
generic `tools/` namespace. Two reverse dependency violations exist where `core/`
imports from satellite app packages.

## Target Directory Structure

```
multi_tracker/
  # ── App layer (top) ──────────────────────────────────
  mat/                        # MAT main tracker app
    app/                      #   launcher, bootstrap  (moved from app/)
    gui/                      #   main_window, dialogs, widgets  (moved from gui/)
  posekit/                    # PoseKit app  (stays)
    core/
    inference/
    ui/
  classkit/                   # ClassKit app  (stays)
    ...
  afterhours/                 # Afterhours app  (stays)
    core/
    gui/
  datasieve/                  # DataSieve app  (promoted from tools/data_sieve/)
    gui.py

  # ── Integrations layer ──────────────────────────────
  integrations/               # External tool bridges
    sleap/                    #   formats, launcher, PoseInferenceService
    xanylabeling/             #   X-AnyLabeling CLI adapter  (moved from integrations/)

  # ── Core layer ──────────────────────────────────────
  core/                       # Shared tracking pipeline
    assigners/
    background/
    canonicalization/
    detectors/
    filters/
    identity/
      pose/backends/          #   SLEAP + YOLO runtime inference
    post/
    tracking/

  # ── Shared infrastructure (bottom) ──────────────────
  runtime/                    # Compute runtime  (promoted from core/runtime/)
  data/                       # CSV, caches  (stays)
  training/                   # Model training  (stays)
  utils/                      # Geometry, GPU, image processing  (stays)
```

## Dependency Rules

```
apps  (mat, posekit, classkit, afterhours, datasieve)
  |
  v
integrations/
  |
  v
core/
  |
  v
runtime/  data/  training/  utils/   (shared infrastructure)
```

- **Apps** may import from any layer below.
- **Integrations** may import from core + shared infrastructure. Never from apps.
- **Core** may import from shared infrastructure. Never from integrations or apps.
- **Shared infrastructure** packages may not import from each other except
  `utils/` which is a leaf dependency available to all.
- **No cross-imports between apps.** Each app is independent.

## Reverse Dependency Fixes

### Fix 1: Density computation  (`core` -> `afterhours`)

**Current:** `core/tracking/worker.py` imports from
`afterhours.core.confidence_density` at 3 call sites (lines 858, 953, 972).

**Fix:** Move `confidence_density.py` (or the relevant functions:
`compute_density_map_from_cache`, `export_diagnostic_video`, `save_regions`,
`load_regions`) from `afterhours/core/` into `core/tracking/density.py`.
Afterhours then imports from `multi_tracker.core.tracking.density`.

**Rationale:** Density map computation directly influences assignment behavior
during forward and backward tracking passes. It is core tracking pipeline logic.

### Fix 2: SLEAP inference service  (`core` -> `posekit`)

**Current:** `core/identity/pose/backends/sleap.py:302` imports
`PoseInferenceService` from `posekit.inference.service`.

**Fix:** Move `PoseInferenceService` into `integrations/sleap/service.py`.
Both `core/identity/pose/backends/sleap.py` and `posekit/` import from
`multi_tracker.integrations.sleap.service`.

**Rationale:** `PoseInferenceService` manages SLEAP as an external tool
(launching, communicating with the SLEAP process). This is integration logic,
not pose estimation pipeline logic. The SLEAP *backend* in `core/` calls the
service but shouldn't own its implementation.

## Moves Summary

| Current location | New location | Reason |
|---|---|---|
| `app/` | `mat/app/` | Make MAT explicit |
| `gui/` | `mat/gui/` | Make MAT explicit |
| `tools/data_sieve/` | `datasieve/` | Promote to peer app |
| `tools/` | *(deleted)* | Empty after datasieve promotion |
| `core/runtime/` | `runtime/` | Shared by multiple apps |
| `integrations/xanylabeling_cli.py` | `integrations/xanylabeling/cli.py` | Organized sub-package |
| `afterhours/core/confidence_density.py` | `core/tracking/density.py` | Fix reverse dependency |
| `posekit/inference/service.py` | `integrations/sleap/service.py` | Fix reverse dependency |

## Entry Point Updates

All `[project.scripts]` in `pyproject.toml` need path updates:

| Entry point | Current target | New target |
|---|---|---|
| `mat` | `multi_tracker.app.launcher:main` | `multi_tracker.mat.app.launcher:main` |
| `datasieve` | `multi_tracker.tools.data_sieve.gui:main` | `multi_tracker.datasieve.gui:main` |
| `posekit-labeler` | *(unchanged)* | *(unchanged)* |
| `classkit-labeler` | *(unchanged)* | *(unchanged)* |
| `mat-afterhours` | *(unchanged)* | *(unchanged)* |

## Import Update Scope

Every `from multi_tracker.app`, `from multi_tracker.gui`,
`from multi_tracker.tools`, `from multi_tracker.core.runtime`, and the 4
reverse-dependency import sites need updating. A bulk find-and-replace pass
across `src/` and `tests/` will cover this.

## What Does NOT Move

- `posekit/` — already self-contained
- `classkit/` — already self-contained
- `afterhours/` — already self-contained (only loses `confidence_density.py`)
- `core/` — stays at root (minus `runtime/`)
- `data/`, `training/`, `utils/` — stay at root

## Testing Strategy

1. Run full test suite after each logical move to catch broken imports early.
2. Verify all 5 CLI entry points launch successfully.
3. Run `make lint` to catch any stale imports.
