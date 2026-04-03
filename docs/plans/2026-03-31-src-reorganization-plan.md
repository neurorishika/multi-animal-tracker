# Source Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `src/hydra_suite/` so MAT is an explicit app, DataSieve is a peer app, `runtime/` is promoted to shared infrastructure, external integrations are centralized, and all reverse dependency violations are eliminated.

**Architecture:** Move MAT's ambient `app/` and `gui/` under `mat/`, promote `tools/data_sieve/` to `datasieve/`, promote `core/runtime/` to `runtime/`, restructure `integrations/` with SLEAP sub-package, and fix 2 reverse dependencies (core→afterhours, core→posekit). Dependency flow: apps → integrations → core → shared infra.

**Tech Stack:** Python, git mv, pyproject.toml entry points

**Design doc:** `docs/plans/2026-03-30-src-reorganization-design.md`

---

## File Structure Overview

### Directories to create
- `src/hydra_suite/mat/` (new app namespace)
- `src/hydra_suite/mat/app/` (moved from `app/`)
- `src/hydra_suite/mat/gui/` (moved from `gui/`)
- `src/hydra_suite/mat/gui/dialogs/` (moved from `gui/dialogs/`)
- `src/hydra_suite/mat/gui/widgets/` (moved from `gui/widgets/`)
- `src/hydra_suite/datasieve/` (moved from `tools/data_sieve/`)
- `src/hydra_suite/runtime/` (moved from `core/runtime/`)
- `src/hydra_suite/integrations/sleap/` (new)
- `src/hydra_suite/integrations/xanylabeling/` (new)

### Directories to delete (after moves)
- `src/hydra_suite/app/`
- `src/hydra_suite/gui/`
- `src/hydra_suite/tools/`
- `src/hydra_suite/core/runtime/`

---

### Task 1: Create a working branch

**Files:**
- None

- [ ] **Step 1: Create and switch to reorganization branch**

```bash
git checkout -b refactor/src-reorganization
```

- [ ] **Step 2: Verify clean starting point**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
```

Expected: All tests pass (or note any pre-existing failures).

- [ ] **Step 3: Commit any uncommitted work if needed**

If there are uncommitted changes from prior work, commit them first so the reorganization starts from a clean state.

---

### Task 2: Move `app/` and `gui/` under `mat/`

This is the largest move — MAT's ambient code gets its own namespace.

**Files:**
- Move: `src/hydra_suite/app/` → `src/hydra_suite/mat/app/`
- Move: `src/hydra_suite/gui/` → `src/hydra_suite/mat/gui/`
- Create: `src/hydra_suite/mat/__init__.py`
- Modify: `src/hydra_suite/__init__.py`
- Modify: `src/hydra_suite/mat/gui/main_window.py` (internal import)
- Modify: `tests/test_main_window_config_persistence.py`
- Modify: `tests/test_preview_background_cache.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Create `mat/` package and move directories**

```bash
cd src/hydra_suite
mkdir -p mat
touch mat/__init__.py
git mv app mat/app
git mv gui mat/gui
```

- [ ] **Step 2: Update `src/hydra_suite/__init__.py`**

Replace the imports at the bottom of the file:

```python
from .mat.app.launcher import main, parse_arguments, setup_logging

try:
    from .core.tracking.worker import TrackingWorker
    from .mat.gui.main_window import MainWindow

    __all__ = [
        "main",
        "parse_arguments",
        "setup_logging",
        "TrackingWorker",
        "MainWindow",
    ]
except ImportError:
    __all__ = ["main", "parse_arguments", "setup_logging"]
```

- [ ] **Step 3: Update internal import in `mat/gui/main_window.py`**

Find and replace:

```python
# Old
from hydra_suite.gui.dialogs import CNNIdentityImportDialog
# New
from hydra_suite.tracker.gui.dialogs import CNNIdentityImportDialog
```

- [ ] **Step 4: Update `pyproject.toml` entry points**

```toml
# Old
hydra-suite = "hydra_suite.app.launcher:main"
mat = "hydra_suite.app.launcher:main"

# New
hydra-suite = "hydra_suite.tracker.app.launcher:main"
mat = "hydra_suite.tracker.app.launcher:main"
```

- [ ] **Step 5: Update test imports**

In `tests/test_main_window_config_persistence.py`:
```python
# Old
from hydra_suite.gui.main_window import MainWindow
# New
from hydra_suite.tracker.gui.main_window import MainWindow
```

In `tests/test_preview_background_cache.py`:
```python
# Old
from hydra_suite.gui import main_window
# New
from hydra_suite.tracker.gui import main_window
```

- [ ] **Step 6: Search for any remaining `hydra_suite.gui` or `hydra_suite.app` references**

```bash
cd /path/to/repo
grep -rn "hydra_suite\.gui\b" src/ tests/ --include="*.py" | grep -v "__pycache__"
grep -rn "hydra_suite\.app\b" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references found. Also check for string references (e.g., in mock patches or config strings).

- [ ] **Step 7: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move app/ and gui/ under mat/ namespace"
```

---

### Task 3: Promote `tools/data_sieve/` to `datasieve/`

**Files:**
- Move: `src/hydra_suite/tools/data_sieve/` → `src/hydra_suite/datasieve/`
- Delete: `src/hydra_suite/tools/` (after move)
- Modify: `src/hydra_suite/datasieve/gui.py` (if internal imports exist)
- Modify: `src/hydra_suite/datasieve/main.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Move data_sieve and remove tools/**

```bash
cd src/hydra_suite
git mv tools/data_sieve datasieve
git rm -r tools/
```

If `git rm -r tools/` fails because of `run_data_sieve.py`, remove it explicitly:

```bash
git rm tools/run_data_sieve.py
git rm tools/__init__.py
```

Then remove the empty directory if needed.

- [ ] **Step 2: Update internal imports in `datasieve/main.py`**

```python
# Old
from hydra_suite.tools.data_sieve.gui import main
# New
from hydra_suite.datasieve.gui import main
```

- [ ] **Step 3: Update `pyproject.toml` entry points**

```toml
# Old
datasieve = "hydra_suite.tools.data_sieve.gui:main"
sieve = "hydra_suite.tools.data_sieve.gui:main"

# New
datasieve = "hydra_suite.datasieve.gui:main"
sieve = "hydra_suite.datasieve.gui:main"
```

- [ ] **Step 4: Search for any remaining `hydra_suite.tools` references**

```bash
grep -rn "hydra_suite\.tools" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references found.

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: promote datasieve to peer app namespace"
```

---

### Task 4: Promote `core/runtime/` to root-level `runtime/`

**Files:**
- Move: `src/hydra_suite/core/runtime/` → `src/hydra_suite/runtime/`
- Modify: `src/hydra_suite/core/identity/pose/api.py`
- Modify: `src/hydra_suite/posekit/ui/runtimes.py` (imports `core.runtime`)
- Modify: `tests/test_compute_runtime.py`
- Modify: `tools/benchmark_models.py`

- [ ] **Step 1: Move runtime/ out of core/**

```bash
cd src/hydra_suite
git mv core/runtime runtime
```

- [ ] **Step 2: Update import in `core/identity/pose/api.py`**

```python
# Old
from hydra_suite.core.runtime.compute_runtime import derive_pose_runtime_settings
# New
from hydra_suite.runtime.compute_runtime import derive_pose_runtime_settings
```

- [ ] **Step 3: Update import in `posekit/ui/runtimes.py`**

```python
# Old
from hydra_suite.core.runtime.compute_runtime import ...
# New
from hydra_suite.runtime.compute_runtime import ...
```

Note: Verify the exact import line first — the explore agent found this file imports from `core.runtime`.

- [ ] **Step 4: Update `tests/test_compute_runtime.py`**

```python
# Old
return importlib.import_module("hydra_suite.core.runtime.compute_runtime")
# New
return importlib.import_module("hydra_suite.runtime.compute_runtime")
```

- [ ] **Step 5: Update `tools/benchmark_models.py`**

```python
# Old
from hydra_suite.core.runtime.compute_runtime import (
# New
from hydra_suite.runtime.compute_runtime import (
```

- [ ] **Step 6: Search for any remaining `hydra_suite.core.runtime` references**

```bash
grep -rn "hydra_suite\.core\.runtime" src/ tests/ tools/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references found.

- [ ] **Step 7: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: promote runtime/ to shared infrastructure"
```

---

### Task 5: Fix reverse dependency — move density computation from afterhours to core

**Files:**
- Move: `src/hydra_suite/afterhours/core/confidence_density.py` → `src/hydra_suite/core/tracking/density.py`
- Modify: `src/hydra_suite/core/tracking/worker.py` (3 import sites: lines ~858, ~953, ~972)
- Modify: `src/hydra_suite/afterhours/core/event_scorer.py`
- Modify: `src/hydra_suite/afterhours/gui/main_window.py`
- Modify: `tests/test_density_aware_assignment.py`
- Modify: `tests/test_confidence_density_video.py`
- Modify: `tests/test_confidence_density.py`

- [ ] **Step 1: Move confidence_density.py to core/tracking/density.py**

```bash
cd src/hydra_suite
git mv afterhours/core/confidence_density.py core/tracking/density.py
```

- [ ] **Step 2: Update imports in `core/tracking/worker.py`**

There are 3 import sites. Replace all occurrences:

```python
# Old (appears at ~3 locations)
from hydra_suite.afterhours.core.confidence_density import (
# New
from hydra_suite.core.tracking.density import (
```

- [ ] **Step 3: Update import in `afterhours/core/event_scorer.py`**

```python
# Old
from hydra_suite.afterhours.core.confidence_density import DensityRegion
# New
from hydra_suite.core.tracking.density import DensityRegion
```

- [ ] **Step 4: Update import in `afterhours/gui/main_window.py`**

```python
# Old
from hydra_suite.afterhours.core.confidence_density import load_regions
# New
from hydra_suite.core.tracking.density import load_regions
```

- [ ] **Step 5: Update test imports**

In `tests/test_density_aware_assignment.py`:
```python
# Old
from hydra_suite.afterhours.core.confidence_density import DensityRegion
# New
from hydra_suite.core.tracking.density import DensityRegion
```

In `tests/test_confidence_density_video.py`:
```python
# Old
from hydra_suite.afterhours.core.confidence_density import (
# New
from hydra_suite.core.tracking.density import (
```

In `tests/test_confidence_density.py`:
```python
# Old
from hydra_suite.afterhours.core.confidence_density import (
# New
from hydra_suite.core.tracking.density import (
```

- [ ] **Step 6: Search for any remaining `afterhours.core.confidence_density` references**

```bash
grep -rn "afterhours\.core\.confidence_density\|afterhours/core/confidence_density" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references found.

- [ ] **Step 7: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: move density computation from afterhours to core/tracking"
```

---

### Task 6: Fix reverse dependency — move SLEAP inference service to integrations

**Files:**
- Create: `src/hydra_suite/integrations/sleap/__init__.py`
- Move: `src/hydra_suite/posekit/inference/service.py` → `src/hydra_suite/integrations/sleap/service.py`
- Modify: `src/hydra_suite/core/identity/pose/backends/sleap.py`
- Modify: `src/hydra_suite/posekit/ui/workers.py`
- Modify: `src/hydra_suite/posekit/ui/main_window.py`
- Modify: `src/hydra_suite/posekit/ui/dialogs/utils.py`
- Modify: `tests/test_runtime_api_sleap_export.py`

- [ ] **Step 1: Create `integrations/sleap/` package and move service**

```bash
cd src/hydra_suite
mkdir -p integrations/sleap
touch integrations/sleap/__init__.py
git mv posekit/inference/service.py integrations/sleap/service.py
```

- [ ] **Step 2: Update import in `core/identity/pose/backends/sleap.py`**

```python
# Old
from hydra_suite.posekit.inference.service import PoseInferenceService
# New
from hydra_suite.integrations.sleap.service import PoseInferenceService
```

Also check for the legacy fallback import on the next line and update or remove if appropriate:
```python
# Old fallback
from hydra_suite.posekit_old.pose_inference import PoseInferenceService
```

- [ ] **Step 3: Update import in `posekit/ui/workers.py`**

```python
# Old
from hydra_suite.posekit.inference.service import PoseInferenceService
# New
from hydra_suite.integrations.sleap.service import PoseInferenceService
```

- [ ] **Step 4: Update import in `posekit/ui/main_window.py`**

```python
# Old
from hydra_suite.posekit.inference.service import PoseInferenceService
# New
from hydra_suite.integrations.sleap.service import PoseInferenceService
```

- [ ] **Step 5: Update import in `posekit/ui/dialogs/utils.py`**

```python
# Old
from hydra_suite.posekit.inference.service import PoseInferenceService
# New
from hydra_suite.integrations.sleap.service import PoseInferenceService
```

- [ ] **Step 6: Update mock path in `tests/test_runtime_api_sleap_export.py`**

```python
# Old
"hydra_suite.posekit.inference.service"
# New
"hydra_suite.integrations.sleap.service"
```

- [ ] **Step 7: Search for any remaining `posekit.inference.service` references**

```bash
grep -rn "posekit\.inference\.service\|posekit/inference/service" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references found.

- [ ] **Step 8: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: move SLEAP inference service to integrations/"
```

---

### Task 7: Restructure `integrations/` — move xanylabeling into sub-package

**Files:**
- Create: `src/hydra_suite/integrations/xanylabeling/__init__.py`
- Move: `src/hydra_suite/integrations/xanylabeling_cli.py` → `src/hydra_suite/integrations/xanylabeling/cli.py`
- Modify: `src/hydra_suite/integrations/__init__.py`

- [ ] **Step 1: Create xanylabeling sub-package and move file**

```bash
cd src/hydra_suite
mkdir -p integrations/xanylabeling
touch integrations/xanylabeling/__init__.py
git mv integrations/xanylabeling_cli.py integrations/xanylabeling/cli.py
```

- [ ] **Step 2: Update `integrations/__init__.py`**

```python
# Old
from .xanylabeling_cli import HARD_CODED_CMD, convert_project

__all__ = ["HARD_CODED_CMD", "convert_project"]

# New
from .xanylabeling.cli import HARD_CODED_CMD, convert_project

__all__ = ["HARD_CODED_CMD", "convert_project"]
```

- [ ] **Step 3: Search for any direct references to `xanylabeling_cli`**

```bash
grep -rn "xanylabeling_cli" src/ tests/ --include="*.py" | grep -v "__pycache__"
```

Fix any remaining references. Most code should import via `hydra_suite.integrations` (the `__init__.py` re-exports), but verify.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -10
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: restructure integrations/ with xanylabeling sub-package"
```

---

### Task 8: Clean up and final verification

**Files:**
- Modify: `docs/plans/2026-03-30-src-reorganization-design.md` (mark complete)
- Possibly modify: `CLAUDE.md` (update architecture table)
- Possibly modify: docs referencing old paths

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass.

- [ ] **Step 2: Run linter**

```bash
make lint
```

Expected: No new lint errors.

- [ ] **Step 3: Verify all 5 CLI entry points resolve**

```bash
python -c "from hydra_suite.tracker.app.launcher import main; print('mat OK')"
python -c "from hydra_suite.posekit.ui.main import main; print('posekit OK')"
python -c "from hydra_suite.datasieve.gui import main; print('datasieve OK')"
python -c "from hydra_suite.classkit.app import main; print('classkit OK')"
python -c "from hydra_suite.afterhours.app import main; print('afterhours OK')"
```

Expected: All 5 print OK.

- [ ] **Step 4: Verify no remaining import violations**

```bash
# core/ should not import from app-layer packages
grep -rn "from hydra_suite\.\(mat\|posekit\|classkit\|afterhours\|datasieve\)\." src/hydra_suite/core/ --include="*.py" | grep -v "__pycache__"

# integrations/ should not import from app-layer packages
grep -rn "from hydra_suite\.\(mat\|posekit\|classkit\|afterhours\|datasieve\)\." src/hydra_suite/integrations/ --include="*.py" | grep -v "__pycache__"

# shared infra should not import upward
grep -rn "from hydra_suite\.\(core\|mat\|posekit\|classkit\|afterhours\|datasieve\|integrations\)\." src/hydra_suite/runtime/ src/hydra_suite/utils/ src/hydra_suite/data/ src/hydra_suite/training/ --include="*.py" | grep -v "__pycache__"
```

Expected: No matches for any of the above.

- [ ] **Step 5: Update `CLAUDE.md` architecture table**

Update the "System Layers" table to reflect new paths:

```markdown
| Layer | Package | Role |
|---|---|---|
| MAT App | `hydra_suite.tracker` | MAT launcher, GUI, dialogs, widgets |
| PoseKit | `hydra_suite.posekit` | Pose-labeling application |
| ClassKit | `hydra_suite.classkit` | Classification/embedding toolkit |
| Afterhours | `hydra_suite.afterhours` | Interactive proofreading |
| DataSieve | `hydra_suite.datasieve` | Data sieve tool |
| Integrations | `hydra_suite.integrations` | External tool bridges (SLEAP, X-AnyLabeling) |
| Core | `hydra_suite.core` | Detection, Kalman filter, assignment, post-processing, identity |
| Runtime | `hydra_suite.runtime` | Compute runtime selection and GPU utilities |
| Data | `hydra_suite.data` | CSV export, detection cache, dataset generation/merge |
| Training | `hydra_suite.training` | Dataset builders, training runner, registry, service |
| Utils | `hydra_suite.utils` | Geometry, image processing, batching, prefetch |
```

Also update file path references throughout CLAUDE.md (e.g., `core/runtime/compute_runtime.py` → `runtime/compute_runtime.py`, `app.launcher` → `mat.app.launcher`).

- [ ] **Step 6: Update API docs references if they exist**

Check `docs/reference/api-posekit.md` for references to `hydra_suite.posekit.inference.service` and update to `hydra_suite.integrations.sleap.service`.

```bash
grep -rn "posekit\.inference" docs/ | grep -v "__pycache__"
grep -rn "hydra_suite\.app\.\|hydra_suite\.gui\.\|hydra_suite\.tools\.\|hydra_suite\.core\.runtime" docs/ | grep -v "__pycache__"
```

Fix any stale path references found.

- [ ] **Step 7: Commit final cleanup**

```bash
git add -A
git commit -m "docs: update architecture docs for src reorganization"
```

---

## Task Dependency Order

```
Task 1 (branch)
  → Task 2 (mat/)
  → Task 3 (datasieve/)
  → Task 4 (runtime/)
  → Task 5 (density fix)
  → Task 6 (SLEAP service fix)
  → Task 7 (xanylabeling restructure)
  → Task 8 (cleanup + verification)
```

Tasks 2-4 are independent moves (could be parallelized but sequential is safer for clean git history). Tasks 5-6 are the dependency fixes. Task 7 is a minor restructure. Task 8 is final verification.
