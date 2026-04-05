# Slice 2 — Config/Schema Unification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each of the five remaining kits (trackerkit, posekit, refinekit, detectkit, filterkit) a typed config schema following the classkit pattern, so GUI state lives in a dataclass rather than scattered widget attributes on `MainWindow`.

**Architecture:** Each kit gets `<kit>/config/schemas.py` with a root `<KitName>Config` dataclass. `MainWindow.__init__` initializes `self.config = <KitName>Config()`. Workers receive the config dataclass — not a reference to `MainWindow`. The classkit `config/schemas.py` is the reference implementation.

**Tech Stack:** Python `dataclasses`, `pathlib.Path`, pytest

**Reference:** `src/hydra_suite/classkit/config/schemas.py` — read this before starting each task.

**Important:** This plan covers schema creation and `MainWindow` wiring. It does NOT migrate every widget attribute in one pass — that would be too large. The pattern is:
1. Create the schema with the fields most likely held in `MainWindow.__init__`.
2. Wire `self.config` in `__init__`.
3. Update the workers in that kit to accept config instead of `self` (the `MainWindow`).
Any widget attribute that turns out to be truly ephemeral UI state (selection highlight, open tab index, etc.) should stay on `MainWindow` — only *session-meaningful* state belongs in the config.

---

### Task 1: trackerkit config schema

**Files:**
- Create: `src/hydra_suite/trackerkit/config/__init__.py`
- Create: `src/hydra_suite/trackerkit/config/schemas.py`
- Create: `tests/test_trackerkit_config.py`
- Modify: `src/hydra_suite/trackerkit/gui/main_window.py`

- [ ] **Step 1: Read the existing MainWindow `__init__` to identify session state**

Open `src/hydra_suite/trackerkit/gui/main_window.py` and scan `__init__` for all `self.X = <default>` assignments where `X` is not a Qt widget, layout, signal, or purely ephemeral UI variable. These are the config fields.

Common patterns to look for:
- File paths: `self.video_path`, `self.output_dir`, `self.model_path`
- Counts/numeric parameters: `self.n_animals`, `self.fps`, `self.scale`
- Runtime selection: `self.compute_runtime`
- Feature flags: `self.use_backward_pass`, `self.use_identity`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_trackerkit_config.py
import pytest
from pathlib import Path


def test_tracker_config_defaults():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    cfg = TrackerConfig()
    assert isinstance(cfg.video_path, str)
    assert isinstance(cfg.n_animals, int)
    assert cfg.n_animals >= 1


def test_tracker_config_round_trip():
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    cfg = TrackerConfig(video_path="/tmp/test.mp4", n_animals=3)
    d = cfg.to_dict()
    restored = TrackerConfig.from_dict(d)
    assert restored.video_path == "/tmp/test.mp4"
    assert restored.n_animals == 3


def test_tracker_config_path_fields_are_str():
    """Path fields store str, not Path objects, for JSON compatibility."""
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    cfg = TrackerConfig(video_path="/some/path.mp4")
    d = cfg.to_dict()
    assert isinstance(d["video_path"], str)
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_trackerkit_config.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 4: Create `src/hydra_suite/trackerkit/config/__init__.py`**

```python
# empty
```

- [ ] **Step 5: Create `src/hydra_suite/trackerkit/config/schemas.py`**

Fill in the fields you identified in step 1. Minimum viable schema:

```python
# src/hydra_suite/trackerkit/config/schemas.py
"""Runtime configuration schema for the MAT tracker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """All session-meaningful state for the MAT tracking application.

    Fields reflect what used to live as ``self.X`` attributes on
    ``MainWindow.__init__``.  Ephemeral UI state (widget selection,
    open tab index, etc.) stays on ``MainWindow``.
    """

    # --- Input/Output ---
    video_path: str = ""
    output_dir: str = ""

    # --- Model ---
    model_path: str = ""
    compute_runtime: str = "cpu"

    # --- Tracking parameters ---
    n_animals: int = 1
    use_backward_pass: bool = False
    use_identity: bool = False

    # Add further fields discovered in MainWindow.__init__ here.

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "video_path": self.video_path,
            "output_dir": self.output_dir,
            "model_path": self.model_path,
            "compute_runtime": self.compute_runtime,
            "n_animals": self.n_animals,
            "use_backward_pass": self.use_backward_pass,
            "use_identity": self.use_identity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrackerConfig":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls(
            video_path=data.get("video_path", ""),
            output_dir=data.get("output_dir", ""),
            model_path=data.get("model_path", ""),
            compute_runtime=data.get("compute_runtime", "cpu"),
            n_animals=data.get("n_animals", 1),
            use_backward_pass=data.get("use_backward_pass", False),
            use_identity=data.get("use_identity", False),
        )
```

Expand the field list based on your step 1 scan.

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_trackerkit_config.py -v
```

Expected: 3 tests pass.

- [ ] **Step 7: Wire `self.config` in `MainWindow.__init__`**

In `src/hydra_suite/trackerkit/gui/main_window.py`, add at the top of the imports:

```python
from hydra_suite.trackerkit.config.schemas import TrackerConfig
```

In `MainWindow.__init__`, add early (before any widget setup):

```python
self.config = TrackerConfig()
```

Then replace each scattered `self.video_path = ""`, `self.n_animals = 1`, etc. with reads/writes of `self.config.video_path`, `self.config.n_animals`, etc.

Do this field by field — search for each attribute name and update the site. Run the app after each field to verify nothing broke.

- [ ] **Step 8: Update workers in trackerkit to accept config**

For each worker in `main_window.py` that previously received `self` (the `MainWindow`) or a grab-bag of parameters, update its `__init__` to accept `TrackerConfig`:

```python
# Before:
class MergeWorker(BaseWorker):
    def __init__(self, main_window, ...):
        super().__init__()
        self.main_window = main_window

# After:
class MergeWorker(BaseWorker):
    def __init__(self, config: TrackerConfig, ...):
        super().__init__()
        self.config = config
```

Update `execute()` to read from `self.config` instead of `self.main_window.some_attribute`.

- [ ] **Step 9: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 10: Commit**

```bash
git add src/hydra_suite/trackerkit/config/ tests/test_trackerkit_config.py src/hydra_suite/trackerkit/gui/main_window.py
git commit -m "feat: add TrackerConfig schema and wire to trackerkit MainWindow"
```

---

### Task 2: detectkit config schema

**Files:**
- Create: `src/hydra_suite/detectkit/config/__init__.py`
- Create: `src/hydra_suite/detectkit/config/schemas.py`
- Create: `tests/test_detectkit_config.py`
- Modify: `src/hydra_suite/detectkit/gui/main_window.py` (or equivalent entry point)

- [ ] **Step 1: Read detectkit MainWindow `__init__` to identify fields**

Open `src/hydra_suite/detectkit/gui/main_window.py` and scan for session-meaningful `self.X` attributes (same criteria as Task 1 step 1).

- [ ] **Step 2: Write the failing test**

```python
# tests/test_detectkit_config.py
def test_detectkit_config_defaults():
    from hydra_suite.detectkit.config.schemas import DetectKitConfig
    cfg = DetectKitConfig()
    assert isinstance(cfg.model_path, str)
    assert isinstance(cfg.compute_runtime, str)


def test_detectkit_config_round_trip():
    from hydra_suite.detectkit.config.schemas import DetectKitConfig
    cfg = DetectKitConfig(model_path="/tmp/model.pt", compute_runtime="cuda")
    restored = DetectKitConfig.from_dict(cfg.to_dict())
    assert restored.model_path == "/tmp/model.pt"
    assert restored.compute_runtime == "cuda"
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_detectkit_config.py -v
```

- [ ] **Step 4: Create `src/hydra_suite/detectkit/config/__init__.py`**

```python
# empty
```

- [ ] **Step 5: Create `src/hydra_suite/detectkit/config/schemas.py`**

```python
# src/hydra_suite/detectkit/config/schemas.py
"""Runtime configuration schema for DetectKit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectKitConfig:
    """Session-meaningful state for the DetectKit labeling application."""

    # --- Model ---
    model_path: str = ""
    compute_runtime: str = "cpu"

    # --- Dataset / output ---
    dataset_dir: str = ""
    output_dir: str = ""

    # Add further fields discovered in MainWindow.__init__ here.

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "compute_runtime": self.compute_runtime,
            "dataset_dir": self.dataset_dir,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectKitConfig":
        return cls(
            model_path=data.get("model_path", ""),
            compute_runtime=data.get("compute_runtime", "cpu"),
            dataset_dir=data.get("dataset_dir", ""),
            output_dir=data.get("output_dir", ""),
        )
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_detectkit_config.py -v
```

- [ ] **Step 7: Wire `self.config` in detectkit MainWindow**

Same pattern as Task 1 step 7: add import, `self.config = DetectKitConfig()` early in `__init__`, replace scattered attributes.

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/detectkit/config/ tests/test_detectkit_config.py src/hydra_suite/detectkit/gui/
git commit -m "feat: add DetectKitConfig schema and wire to detectkit MainWindow"
```

---

### Task 3: posekit config schema

**Files:**
- Create: `src/hydra_suite/posekit/config/__init__.py`
- Create: `src/hydra_suite/posekit/config/schemas.py`
- Create: `tests/test_posekit_config.py`
- Modify: `src/hydra_suite/posekit/gui/main_window.py`

- [ ] **Step 1: Read posekit MainWindow `__init__` to identify fields**

Open `src/hydra_suite/posekit/gui/main_window.py` and scan for session-meaningful `self.X` attributes.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_posekit_config.py
def test_posekit_config_defaults():
    from hydra_suite.posekit.config.schemas import PoseKitConfig
    cfg = PoseKitConfig()
    assert isinstance(cfg.project_dir, str)
    assert isinstance(cfg.compute_runtime, str)


def test_posekit_config_round_trip():
    from hydra_suite.posekit.config.schemas import PoseKitConfig
    cfg = PoseKitConfig(project_dir="/tmp/project", compute_runtime="mps")
    restored = PoseKitConfig.from_dict(cfg.to_dict())
    assert restored.project_dir == "/tmp/project"
    assert restored.compute_runtime == "mps"
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_posekit_config.py -v
```

- [ ] **Step 4: Create `src/hydra_suite/posekit/config/__init__.py`**

```python
# empty
```

- [ ] **Step 5: Create `src/hydra_suite/posekit/config/schemas.py`**

```python
# src/hydra_suite/posekit/config/schemas.py
"""Runtime configuration schema for PoseKit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PoseKitConfig:
    """Session-meaningful state for the PoseKit labeling application."""

    # --- Project ---
    project_dir: str = ""
    skeleton_name: str = ""

    # --- Inference ---
    model_path: str = ""
    compute_runtime: str = "cpu"

    # Add further fields discovered in MainWindow.__init__ here.

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "skeleton_name": self.skeleton_name,
            "model_path": self.model_path,
            "compute_runtime": self.compute_runtime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseKitConfig":
        return cls(
            project_dir=data.get("project_dir", ""),
            skeleton_name=data.get("skeleton_name", ""),
            model_path=data.get("model_path", ""),
            compute_runtime=data.get("compute_runtime", "cpu"),
        )
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_posekit_config.py -v
```

- [ ] **Step 7: Wire `self.config` in posekit MainWindow**

Same pattern: import, `self.config = PoseKitConfig()`, replace scattered attributes.

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/posekit/config/ tests/test_posekit_config.py src/hydra_suite/posekit/gui/main_window.py
git commit -m "feat: add PoseKitConfig schema and wire to posekit MainWindow"
```

---

### Task 4: refinekit config schema

**Files:**
- Create: `src/hydra_suite/refinekit/config/__init__.py`
- Create: `src/hydra_suite/refinekit/config/schemas.py`
- Create: `tests/test_refinekit_config.py`
- Modify: `src/hydra_suite/refinekit/gui/main_window.py`

- [ ] **Step 1: Read refinekit MainWindow `__init__` to identify fields**

- [ ] **Step 2: Write the failing test**

```python
# tests/test_refinekit_config.py
def test_refinekit_config_defaults():
    from hydra_suite.refinekit.config.schemas import RefineKitConfig
    cfg = RefineKitConfig()
    assert isinstance(cfg.project_dir, str)


def test_refinekit_config_round_trip():
    from hydra_suite.refinekit.config.schemas import RefineKitConfig
    cfg = RefineKitConfig(project_dir="/tmp/project")
    restored = RefineKitConfig.from_dict(cfg.to_dict())
    assert restored.project_dir == "/tmp/project"
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_refinekit_config.py -v
```

- [ ] **Step 4: Create config package**

```python
# src/hydra_suite/refinekit/config/__init__.py
# empty
```

- [ ] **Step 5: Create `src/hydra_suite/refinekit/config/schemas.py`**

```python
# src/hydra_suite/refinekit/config/schemas.py
"""Runtime configuration schema for RefineKit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RefineKitConfig:
    """Session-meaningful state for the RefineKit proofreading application."""

    # --- Project ---
    project_dir: str = ""
    csv_path: str = ""
    video_path: str = ""

    # Add further fields discovered in MainWindow.__init__ here.

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "csv_path": self.csv_path,
            "video_path": self.video_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefineKitConfig":
        return cls(
            project_dir=data.get("project_dir", ""),
            csv_path=data.get("csv_path", ""),
            video_path=data.get("video_path", ""),
        )
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_refinekit_config.py -v
```

- [ ] **Step 7: Wire `self.config` in refinekit MainWindow**

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/refinekit/config/ tests/test_refinekit_config.py src/hydra_suite/refinekit/gui/main_window.py
git commit -m "feat: add RefineKitConfig schema and wire to refinekit MainWindow"
```

---

### Task 5: filterkit config schema

**Files:**
- Create: `src/hydra_suite/filterkit/config/__init__.py`
- Create: `src/hydra_suite/filterkit/config/schemas.py`
- Create: `tests/test_filterkit_config.py`
- Modify: `src/hydra_suite/filterkit/gui/main_window.py`

- [ ] **Step 1: Read filterkit MainWindow `__init__` to identify fields**

- [ ] **Step 2: Write the failing test**

```python
# tests/test_filterkit_config.py
def test_filterkit_config_defaults():
    from hydra_suite.filterkit.config.schemas import FilterKitConfig
    cfg = FilterKitConfig()
    assert isinstance(cfg.input_csv, str)
    assert isinstance(cfg.output_dir, str)


def test_filterkit_config_round_trip():
    from hydra_suite.filterkit.config.schemas import FilterKitConfig
    cfg = FilterKitConfig(input_csv="/tmp/data.csv", output_dir="/tmp/out")
    restored = FilterKitConfig.from_dict(cfg.to_dict())
    assert restored.input_csv == "/tmp/data.csv"
    assert restored.output_dir == "/tmp/out"
```

- [ ] **Step 3: Run to confirm failure**

```bash
python -m pytest tests/test_filterkit_config.py -v
```

- [ ] **Step 4: Create config package**

```python
# src/hydra_suite/filterkit/config/__init__.py
# empty
```

- [ ] **Step 5: Create `src/hydra_suite/filterkit/config/schemas.py`**

```python
# src/hydra_suite/filterkit/config/schemas.py
"""Runtime configuration schema for FilterKit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FilterKitConfig:
    """Session-meaningful state for the FilterKit sieve application."""

    input_csv: str = ""
    output_dir: str = ""

    # Add further fields discovered in MainWindow.__init__ here.

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_csv": self.input_csv,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FilterKitConfig":
        return cls(
            input_csv=data.get("input_csv", ""),
            output_dir=data.get("output_dir", ""),
        )
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_filterkit_config.py -v
```

- [ ] **Step 7: Wire `self.config` in filterkit MainWindow**

- [ ] **Step 8: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 9: Commit**

```bash
git add src/hydra_suite/filterkit/config/ tests/test_filterkit_config.py src/hydra_suite/filterkit/gui/main_window.py
git commit -m "feat: add FilterKitConfig schema and wire to filterkit MainWindow"
```

---

### Task 6: Verify and finalize

- [ ] **Step 1: Confirm all kits have a config package**

```bash
ls src/hydra_suite/*/config/schemas.py
```

Expected: 6 files (classkit + 5 new ones).

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short -m "not benchmark"
```

- [ ] **Step 3: Commit completion marker**

```bash
git commit --allow-empty -m "chore: Slice 2 (config schema unification) complete"
```
