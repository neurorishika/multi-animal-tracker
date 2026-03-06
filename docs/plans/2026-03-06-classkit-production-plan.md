# ClassKit Production Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring ClassKit to production-ready status as the sole animal labeling and classification training tool, with multi-factor labeling, preset workflows, full training surface (replacing MAT's classification training), and model publishing to the shared `models/` registry.

**Architecture:** Extend existing ClassKit data model with `LabelingScheme`/`Factor` dataclasses; add a `FactorStepperWidget` for sequential labeling; replace the simple `TrainingDialog` with a full `ClassKitTrainingDialog` (live log, multi-mode); extend `training/contracts.py` and `model_publish.py` with four new classify roles; strip classification from MAT's `TrainYoloDialog`.

**Tech Stack:** Python 3.11+, PyQt6, PyTorch, Ultralytics YOLO, dataclasses, pytest

---

## Reference Files (read these before touching anything)

- `src/multi_tracker/classkit/config/schemas.py` — data model to extend
- `src/multi_tracker/classkit/presets.py` — new file
- `src/multi_tracker/training/contracts.py` — `TrainingRole` enum to extend
- `src/multi_tracker/training/model_publish.py` — `_repo_dir_for_role` and `publish_trained_model` to extend
- `src/multi_tracker/training/runner.py` — `_iter_classify_samples` to generalize; add `_train_tiny_classify`
- `src/multi_tracker/classkit/gui/widgets/factor_stepper.py` — new widget
- `src/multi_tracker/classkit/gui/dialogs.py` — `NewProjectDialog` + replace `TrainingDialog`
- `src/multi_tracker/classkit/jobs/task_workers.py` — add `ClassKitTrainingWorker`
- `src/multi_tracker/classkit/gui/mainwindow.py` — wire stepper + new training dialog
- `src/multi_tracker/gui/dialogs/train_yolo_dialog.py` — strip classification roles

---

## Task 1: Data Model — `LabelingScheme` and `Factor`

**Files:**
- Modify: `src/multi_tracker/classkit/config/schemas.py`
- Create: `tests/test_classkit_scheme.py`

**Step 1: Write the failing tests**

```python
# tests/test_classkit_scheme.py
from multi_tracker.classkit.config.schemas import Factor, LabelingScheme, ProjectConfig
from pathlib import Path


def test_factor_has_name_and_labels():
    f = Factor(name="tag_1", labels=["red", "blue", "green"])
    assert f.name == "tag_1"
    assert f.labels == ["red", "blue", "green"]
    assert f.shortcut_keys == []


def test_scheme_single_factor():
    scheme = LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=["young", "old"])],
        training_modes=["flat_tiny", "flat_yolo"],
    )
    assert len(scheme.factors) == 1
    assert scheme.total_classes == 2


def test_scheme_two_factor_cartesian():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue", "green"]),
            Factor(name="tag_2", labels=["red", "blue", "green"]),
        ],
        training_modes=["flat_yolo", "multihead_yolo"],
    )
    assert scheme.total_classes == 9


def test_scheme_composite_label_round_trip():
    scheme = LabelingScheme(
        name="color2",
        factors=[
            Factor(name="tag_1", labels=["red", "blue"]),
            Factor(name="tag_2", labels=["green", "yellow"]),
        ],
        training_modes=["flat_yolo"],
    )
    composite = scheme.encode_label(["red", "green"])
    assert composite == "red|green"
    decoded = scheme.decode_label(composite)
    assert decoded == ["red", "green"]


def test_project_config_accepts_scheme():
    scheme = LabelingScheme(
        name="test",
        factors=[Factor(name="f", labels=["a", "b"])],
        training_modes=["flat_tiny"],
    )
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"), scheme=scheme)
    assert cfg.scheme is not None


def test_project_config_scheme_defaults_none():
    cfg = ProjectConfig(name="proj", classes=[], root_dir=Path("/tmp"))
    assert cfg.scheme is None
```

**Step 2: Run tests to verify they fail**

```bash
cd "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker"
python -m pytest tests/test_classkit_scheme.py -v
```

Expected: `FAILED` — `Factor`, `LabelingScheme` don't exist yet; `ProjectConfig` has no `scheme` field.

**Step 3: Implement — append to `schemas.py` and update `ProjectConfig`**

Add after the existing imports at the top of `schemas.py`:

```python
from typing import Optional
```

Append to `schemas.py` after `ClassKitConfig`:

```python
@dataclass
class Factor:
    name: str
    labels: List[str]
    shortcut_keys: List[str] = field(default_factory=list)


@dataclass
class LabelingScheme:
    name: str
    factors: List[Factor]
    training_modes: List[str]
    description: str = ""

    @property
    def total_classes(self) -> int:
        result = 1
        for f in self.factors:
            result *= len(f.labels)
        return result

    def encode_label(self, factor_values: List[str]) -> str:
        """Encode a list of per-factor values to a composite label string."""
        return "|".join(factor_values)

    def decode_label(self, composite: str) -> List[str]:
        """Decode a composite label string back to per-factor values."""
        return composite.split("|")
```

In `ProjectConfig`, add the new optional field:

```python
@dataclass
class ProjectConfig:
    name: str
    classes: List[str]
    root_dir: Path
    description: str = ""
    scheme: Optional["LabelingScheme"] = None
```

Also add `Optional` to the existing `from typing import Dict, List` import.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_scheme.py -v
```

Expected: all 6 tests `PASSED`.

**Step 5: Commit**

```bash
git add src/multi_tracker/classkit/config/schemas.py tests/test_classkit_scheme.py
git commit -m "feat(classkit): add Factor and LabelingScheme dataclasses with composite label encoding"
```

---

## Task 2: Preset Factory Functions

**Files:**
- Create: `src/multi_tracker/classkit/presets.py`
- Modify: `tests/test_classkit_scheme.py` (extend)

**Step 1: Write failing tests — append to `tests/test_classkit_scheme.py`**

```python
from multi_tracker.classkit.presets import (
    head_tail_preset,
    color_tag_preset,
    age_preset,
)


def test_head_tail_preset():
    scheme = head_tail_preset()
    assert scheme.name == "head_tail"
    assert len(scheme.factors) == 1
    assert set(scheme.factors[0].labels) == {"left", "right", "up", "down"}
    assert scheme.total_classes == 4


def test_color_tag_preset_1factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=1, colors=colors)
    assert scheme.total_classes == 5
    assert len(scheme.factors) == 1


def test_color_tag_preset_2factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=2, colors=colors)
    assert scheme.total_classes == 25
    assert len(scheme.factors) == 2
    assert scheme.factors[0].name == "tag_1"
    assert scheme.factors[1].name == "tag_2"


def test_color_tag_preset_3factor():
    colors = ["red", "blue", "green", "yellow", "white"]
    scheme = color_tag_preset(n_factors=3, colors=colors)
    assert scheme.total_classes == 125


def test_age_preset_default():
    scheme = age_preset()
    assert scheme.total_classes == 2
    assert "young" in scheme.factors[0].labels
    assert "old" in scheme.factors[0].labels


def test_age_preset_extra_classes():
    scheme = age_preset(extra_classes=["juvenile"])
    assert scheme.total_classes == 3
    assert "juvenile" in scheme.factors[0].labels


def test_color_tag_preset_custom_colors():
    scheme = color_tag_preset(n_factors=2, colors=["a", "b", "c"])
    assert scheme.total_classes == 9
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_scheme.py -k "preset" -v
```

Expected: `FAILED` — `multi_tracker.classkit.presets` module does not exist.

**Step 3: Create `src/multi_tracker/classkit/presets.py`**

```python
"""Preset LabelingScheme factory functions for common animal classification tasks."""

from __future__ import annotations

from .config.schemas import Factor, LabelingScheme


def head_tail_preset() -> LabelingScheme:
    """Single-factor, 4-class head/tail direction classifier."""
    return LabelingScheme(
        name="head_tail",
        factors=[
            Factor(
                name="direction",
                labels=["left", "right", "up", "down"],
                shortcut_keys=["a", "d", "w", "s"],
            )
        ],
        training_modes=["flat_tiny", "flat_yolo"],
        description="Head/tail orientation: left, right, up, down",
    )


def color_tag_preset(n_factors: int, colors: list[str]) -> LabelingScheme:
    """Multi-factor ordered color-tag classifier.

    Args:
        n_factors: Number of ordered color tag positions (1, 2, or 3 typical).
        colors: Ordered list of color label names for each factor.
    """
    if n_factors < 1:
        raise ValueError("n_factors must be >= 1")
    if not colors:
        raise ValueError("colors must be non-empty")

    factors = [
        Factor(name=f"tag_{i + 1}", labels=list(colors))
        for i in range(n_factors)
    ]
    modes = ["flat_yolo", "multihead_yolo"]
    if n_factors == 1:
        modes = ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]

    total = len(colors) ** n_factors
    return LabelingScheme(
        name=f"color_tags_{n_factors}factor",
        factors=factors,
        training_modes=modes,
        description=f"{n_factors}-factor color tag: {len(colors)} colors each → {total} composites",
    )


def age_preset(extra_classes: list[str] | None = None) -> LabelingScheme:
    """Single-factor age classifier (young/old), extensible."""
    labels = ["young", "old"] + list(extra_classes or [])
    return LabelingScheme(
        name="age",
        factors=[Factor(name="age", labels=labels)],
        training_modes=["flat_tiny", "flat_yolo"],
        description="Age classification: " + ", ".join(labels),
    )
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_scheme.py -v
```

Expected: all tests `PASSED`.

**Step 5: Commit**

```bash
git add src/multi_tracker/classkit/presets.py tests/test_classkit_scheme.py
git commit -m "feat(classkit): add preset factory functions for head-tail, color-tag, and age schemes"
```

---

## Task 3: New `TrainingRole` Values and `model_publish` Extension

**Files:**
- Modify: `src/multi_tracker/training/contracts.py`
- Modify: `src/multi_tracker/training/model_publish.py`
- Create: `tests/test_classkit_publish.py`

**Step 1: Write failing tests**

```python
# tests/test_classkit_publish.py
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from multi_tracker.training.contracts import TrainingRole
from multi_tracker.training.model_publish import _repo_dir_for_role


def test_new_roles_exist():
    assert TrainingRole.CLASSIFY_FLAT_YOLO.value == "classify_flat_yolo"
    assert TrainingRole.CLASSIFY_FLAT_TINY.value == "classify_flat_tiny"
    assert TrainingRole.CLASSIFY_MULTIHEAD_YOLO.value == "classify_multihead_yolo"
    assert TrainingRole.CLASSIFY_MULTIHEAD_TINY.value == "classify_multihead_tiny"


def test_repo_dir_flat_yolo(tmp_path):
    with patch("multi_tracker.training.model_publish.get_models_root", return_value=tmp_path):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_FLAT_YOLO, scheme_name="color2")
    assert d == tmp_path / "YOLO-classify" / "color2"
    assert d.exists()


def test_repo_dir_flat_tiny(tmp_path):
    with patch("multi_tracker.training.model_publish.get_models_root", return_value=tmp_path):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_FLAT_TINY, scheme_name="age")
    assert d == tmp_path / "tiny-classify" / "age"


def test_repo_dir_multihead_yolo(tmp_path):
    with patch("multi_tracker.training.model_publish.get_models_root", return_value=tmp_path):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_MULTIHEAD_YOLO, scheme_name="color2")
    assert d == tmp_path / "YOLO-classify" / "multihead" / "color2"


def test_repo_dir_multihead_tiny(tmp_path):
    with patch("multi_tracker.training.model_publish.get_models_root", return_value=tmp_path):
        d = _repo_dir_for_role(TrainingRole.CLASSIFY_MULTIHEAD_TINY, scheme_name="color2")
    assert d == tmp_path / "tiny-classify" / "multihead" / "color2"


def test_existing_headtail_roles_unchanged(tmp_path):
    """Backwards compat: existing head-tail roles still map correctly."""
    with patch("multi_tracker.training.model_publish.get_models_root", return_value=tmp_path):
        d = _repo_dir_for_role(TrainingRole.HEADTAIL_YOLO)
    assert d == tmp_path / "YOLO-classify" / "orientation"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_publish.py -v
```

Expected: `FAILED` — new role values don't exist; `_repo_dir_for_role` doesn't accept `scheme_name`.

**Step 3: Add new roles to `contracts.py`**

In `TrainingRole` enum, append after `HEADTAIL_TINY`:

```python
# ClassKit classification roles
CLASSIFY_FLAT_YOLO = "classify_flat_yolo"
CLASSIFY_FLAT_TINY = "classify_flat_tiny"
CLASSIFY_MULTIHEAD_YOLO = "classify_multihead_yolo"
CLASSIFY_MULTIHEAD_TINY = "classify_multihead_tiny"
```

**Step 4: Update `_repo_dir_for_role` in `model_publish.py`**

Change signature and body:

```python
def _repo_dir_for_role(role: TrainingRole, scheme_name: str = "orientation") -> Path:
    root = get_models_root()
    if role == TrainingRole.SEQ_DETECT:
        out = root / "YOLO-detect"
    elif role == TrainingRole.SEQ_CROP_OBB:
        out = root / "YOLO-obb" / "cropped"
    elif role in (TrainingRole.HEADTAIL_TINY, TrainingRole.HEADTAIL_YOLO):
        out = root / "YOLO-classify" / "orientation"
    elif role == TrainingRole.CLASSIFY_FLAT_YOLO:
        out = root / "YOLO-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_FLAT_TINY:
        out = root / "tiny-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_YOLO:
        out = root / "YOLO-classify" / "multihead" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_TINY:
        out = root / "tiny-classify" / "multihead" / scheme_name
    else:
        out = root / "YOLO-obb"
    out.mkdir(parents=True, exist_ok=True)
    return out
```

Update all internal callers of `_repo_dir_for_role` in the same file to pass `scheme_name=""` to keep existing behavior (the `orientation` default covers head-tail).

Also update `publish_trained_model` signature to accept optional `scheme_name` and `factor_index`/`factor_name` kwargs, and write them into `metadata`:

```python
def publish_trained_model(
    *,
    role: TrainingRole,
    artifact_path: str,
    size: str,
    species: str,
    model_info: str,
    trained_from_run_id: str,
    dataset_fingerprint: str,
    base_model: str,
    scheme_name: str = "",
    factor_index: int | None = None,
    factor_name: str | None = None,
) -> tuple[str, str]:
```

In the metadata dict, add:
```python
"scheme_name": str(scheme_name or ""),
"factor_index": factor_index,
"factor_name": str(factor_name or "") if factor_name else None,
```

Pass `scheme_name` through to `_repo_dir_for_role`:
```python
repo_dir = _repo_dir_for_role(role, scheme_name=scheme_name or "orientation")
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_publish.py -v
```

Expected: all 6 tests `PASSED`.

**Step 6: Run full test suite to check no regressions**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all existing tests still pass.

**Step 7: Commit**

```bash
git add src/multi_tracker/training/contracts.py src/multi_tracker/training/model_publish.py tests/test_classkit_publish.py
git commit -m "feat(training): add CLASSIFY_* roles and extend publish/repo-dir for ClassKit models"
```

---

## Task 4: Generalize `_iter_classify_samples` and Add `_train_tiny_classify`

**Files:**
- Modify: `src/multi_tracker/training/runner.py`
- Create: `tests/test_classkit_tiny_train.py`

**Step 1: Write failing tests**

```python
# tests/test_classkit_tiny_train.py
import os
import tempfile
from pathlib import Path

import pytest


def _make_dummy_dataset(root: Path, classes: list[str], n_per_class: int = 4):
    """Create minimal image-folder dataset with tiny solid-color PNG images."""
    import numpy as np
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    for split in ("train", "val"):
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n_per_class):
                img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
                cv2.imwrite(str(d / f"img_{i}.png"), img)


def test_iter_classify_samples_general(tmp_path):
    from multi_tracker.training.runner import _iter_classify_samples

    _make_dummy_dataset(tmp_path, ["cat", "dog", "bird"])
    samples = list(_iter_classify_samples(tmp_path, "train"))
    assert len(samples) == 12  # 3 classes × 4 images
    labels = {s[1] for s in samples}
    assert labels == {0, 1, 2}


def test_iter_classify_samples_returns_sorted_class_order(tmp_path):
    from multi_tracker.training.runner import _iter_classify_samples

    _make_dummy_dataset(tmp_path, ["zebra", "apple", "mango"])
    samples = list(_iter_classify_samples(tmp_path, "train"))
    # Classes sorted alphabetically: apple=0, mango=1, zebra=2
    class_dir_names = sorted(["apple", "mango", "zebra"])
    label_map = {name: i for i, name in enumerate(class_dir_names)}
    for img_path, label in samples:
        expected = label_map[img_path.parent.name]
        assert label == expected


def test_train_tiny_classify_runs(tmp_path):
    """Smoke test: tiny N-class CNN trains without error on minimal data."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from multi_tracker.training.contracts import (
        TrainingRole, TrainingRunSpec, TinyHeadTailParams,
        TrainingHyperParams, AugmentationProfile, PublishPolicy
    )
    from multi_tracker.training.runner import run_training

    _make_dummy_dataset(tmp_path / "dataset", ["a", "b", "c"], n_per_class=4)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_FLAT_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="",
        hyperparams=TrainingHyperParams(),
        tiny_params=TinyHeadTailParams(epochs=2, batch=4),
    )

    result = run_training(spec, tmp_path / "run")
    assert result["success"] is True
    assert Path(result["artifact_path"]).exists()
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_tiny_train.py -v
```

Expected: `FAILED` — `_iter_classify_samples` hardcodes two specific class names; `CLASSIFY_FLAT_TINY` not handled in `run_training`.

**Step 3: Generalize `_iter_classify_samples` in `runner.py`**

Replace the existing function body:

```python
def _iter_classify_samples(dataset_dir: Path, split: str):
    """Yield (image_path, class_index) for all images in a classify split.

    Class index is assigned by sorted alphabetical order of class folder names.
    This works for any class structure, including composite label folders.
    """
    split_dir = dataset_dir / split
    if not split_dir.exists():
        return
    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    for cls_idx, cls_dir in enumerate(class_dirs):
        for img in sorted(cls_dir.rglob("*")):
            if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                yield img, cls_idx
```

**Step 4: Add `_train_tiny_classify` to `runner.py`**

Add this function after `_train_tiny_headtail`. It is the general N-class version:

```python
def _train_tiny_classify(
    spec: TrainingRunSpec,
    run_dir: Path,
    log_cb: LogCallback | None = None,
    progress_cb: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict:
    """Train a tiny N-class CNN classifier from an image-folder dataset."""
    try:
        import cv2
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:
        raise RuntimeError(
            f"Tiny classify training requires torch/cv2: {exc}"
        ) from exc

    dataset_dir = Path(spec.derived_dataset_dir).expanduser().resolve()
    device = _pick_torch_device(spec.device)
    _safe_log(log_cb, f"Tiny classify device: {device}")

    train_samples = list(_iter_classify_samples(dataset_dir, "train"))
    val_samples = list(_iter_classify_samples(dataset_dir, "val"))
    if len(train_samples) < 2:
        raise RuntimeError("Tiny classify training requires at least 2 train samples.")

    # Infer num_classes from train split directory count
    train_dir = dataset_dir / "train"
    num_classes = len(sorted(d for d in train_dir.iterdir() if d.is_dir()))
    if num_classes < 2:
        raise RuntimeError("Need at least 2 classes in train split.")

    input_w = int(spec.tiny_params.input_width)
    input_h = int(spec.tiny_params.input_height)

    class TinyDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[1] != input_w or img.shape[0] != input_h:
                img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            y = torch.tensor(label, dtype=torch.long)
            return x, y

    class TinyClassifier(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_classes))

        def forward(self, x):
            return self.classifier(self.features(x))

    model = TinyClassifier(num_classes).to(device)
    train_loader = DataLoader(TinyDataset(train_samples), batch_size=max(1, int(spec.tiny_params.batch)), shuffle=True, num_workers=0)
    val_loader = DataLoader(TinyDataset(val_samples), batch_size=max(1, int(spec.tiny_params.batch)), shuffle=False, num_workers=0) if val_samples else None

    opt = torch.optim.AdamW(model.parameters(), lr=float(spec.tiny_params.lr), weight_decay=float(spec.tiny_params.weight_decay))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    epochs = max(1, int(spec.tiny_params.epochs))
    history = []

    for epoch in range(epochs):
        if should_cancel and should_cancel():
            raise RuntimeError("Canceled")

        model.train()
        train_loss, train_n = 0.0, 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            loss = criterion(model(xs), ys)
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * len(ys)
            train_n += len(ys)

        val_acc = 0.0
        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.inference_mode():
                for xs, ys in val_loader:
                    xs, ys = xs.to(device), ys.to(device)
                    preds = model(xs).argmax(dim=1)
                    correct += int((preds == ys).sum().item())
                    total += len(ys)
            val_acc = correct / max(1, total)

        mean_loss = train_loss / max(1, train_n)
        history.append({"epoch": epoch + 1, "train_loss": mean_loss, "val_acc": val_acc})

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        _safe_log(log_cb, f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} val_acc={val_acc:.4f}")
        if progress_cb:
            progress_cb(epoch + 1, epochs)

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = weights_dir / "best.pth"

    import json as _json
    torch.save({"model_state_dict": model.state_dict(), "input_size": [input_w, input_h], "num_classes": num_classes, "best_val_acc": float(best_val_acc), "history": history}, out_ckpt)

    metrics_path = run_dir / "tiny_metrics.json"
    metrics_path.write_text(_json.dumps({"best_val_acc": float(best_val_acc), "history": history}, indent=2), encoding="utf-8")

    return {"success": True, "artifact_path": str(out_ckpt), "metrics_path": str(metrics_path), "command": ["tiny_classify_inprocess"], "task": "tiny_classify"}
```

**Step 5: Wire new roles into `run_training`**

In `run_training`, add a branch before the Ultralytics subprocess block:

```python
if spec.role in (TrainingRole.CLASSIFY_FLAT_TINY, TrainingRole.CLASSIFY_MULTIHEAD_TINY):
    return _train_tiny_classify(
        spec, run_dir, log_cb=log_cb, progress_cb=progress_cb, should_cancel=should_cancel
    )
```

Also add `CLASSIFY_FLAT_YOLO` and `CLASSIFY_MULTIHEAD_YOLO` to `_ultralytics_task_for_role`:

```python
if role in (TrainingRole.CLASSIFY_FLAT_YOLO, TrainingRole.CLASSIFY_MULTIHEAD_YOLO):
    return "classify"
```

**Step 6: Run tests**

```bash
python -m pytest tests/test_classkit_tiny_train.py tests/test_classkit_publish.py -v
```

Expected: all tests `PASSED`.

**Step 7: Commit**

```bash
git add src/multi_tracker/training/runner.py tests/test_classkit_tiny_train.py
git commit -m "feat(training): generalize _iter_classify_samples and add N-class _train_tiny_classify"
```

---

## Task 5: `FactorStepperWidget`

**Files:**
- Create: `src/multi_tracker/classkit/gui/widgets/factor_stepper.py`
- Create: `tests/test_classkit_stepper.py`

**Step 1: Write failing tests (pure logic, no Qt display)**

```python
# tests/test_classkit_stepper.py
"""Tests for FactorStepper state machine — no Qt event loop needed."""
import pytest
from multi_tracker.classkit.config.schemas import Factor, LabelingScheme


def make_scheme(factor_labels: list[list[str]]) -> LabelingScheme:
    return LabelingScheme(
        name="test",
        factors=[Factor(name=f"f{i}", labels=labels) for i, labels in enumerate(factor_labels)],
        training_modes=["flat_tiny"],
    )


def test_stepper_advances_on_pick():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    assert state.current_factor_index == 0
    assert not state.is_complete

    state.pick("a")
    assert state.current_factor_index == 1
    assert not state.is_complete

    state.pick("x")
    assert state.is_complete
    assert state.composite_label == "a|x"


def test_stepper_back():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    state.pick("a")
    state.back()
    assert state.current_factor_index == 0
    assert state.picks == []


def test_stepper_reset():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    state.pick("a")
    state.pick("x")
    assert state.is_complete
    state.reset()
    assert state.current_factor_index == 0
    assert not state.is_complete
    assert state.picks == []


def test_stepper_single_factor_complete_immediately():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["young", "old"]])
    state = StepperState(scheme)
    state.pick("young")
    assert state.is_complete
    assert state.composite_label == "young"


def test_stepper_invalid_pick_raises():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"]])
    state = StepperState(scheme)
    with pytest.raises(ValueError, match="not in labels"):
        state.pick("invalid")


def test_stepper_back_at_start_does_nothing():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"]])
    state = StepperState(scheme)
    state.back()  # should not raise
    assert state.current_factor_index == 0
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_stepper.py -v
```

Expected: `FAILED` — `StepperState` does not exist yet.

**Step 3: Create `src/multi_tracker/classkit/gui/widgets/factor_stepper.py`**

```python
"""Factor stepper widget for multi-factor compositional labeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ....classkit.config.schemas import LabelingScheme


# ---------------------------------------------------------------------------
# Pure state machine — no Qt dependencies, fully testable
# ---------------------------------------------------------------------------


class StepperState:
    """Tracks progress through a multi-factor labeling sequence."""

    def __init__(self, scheme: LabelingScheme) -> None:
        self._scheme = scheme
        self.picks: list[str] = []

    @property
    def current_factor_index(self) -> int:
        return len(self.picks)

    @property
    def is_complete(self) -> bool:
        return len(self.picks) == len(self._scheme.factors)

    @property
    def composite_label(self) -> str:
        if not self.is_complete:
            raise RuntimeError("Stepper is not complete — call pick() for all factors first.")
        return self._scheme.encode_label(self.picks)

    def pick(self, label: str) -> None:
        """Record a label choice for the current factor and advance."""
        if self.is_complete:
            raise RuntimeError("All factors already picked. Call reset() first.")
        idx = self.current_factor_index
        allowed = self._scheme.factors[idx].labels
        if label not in allowed:
            raise ValueError(f"'{label}' not in labels {allowed} for factor '{self._scheme.factors[idx].name}'")
        self.picks.append(label)

    def back(self) -> None:
        """Undo the last factor pick. No-op if at the start."""
        if self.picks:
            self.picks.pop()

    def reset(self) -> None:
        """Reset to the first factor."""
        self.picks.clear()

    @property
    def current_factor(self):
        if self.is_complete:
            return None
        return self._scheme.factors[self.current_factor_index]

    @property
    def total_factors(self) -> int:
        return len(self._scheme.factors)


# ---------------------------------------------------------------------------
# Qt widget — wraps StepperState with a visual button row
# ---------------------------------------------------------------------------


def _build_qt_widget(scheme: LabelingScheme):  # pragma: no cover
    """Build and return a FactorStepperWidget (Qt import deferred)."""
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import (
        QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
    )

    class FactorStepperWidget(QWidget):
        """Sequential per-factor label picker.

        Signals:
            label_committed(str): Emitted with the composite label when all
                factors are picked.
            skipped(): Emitted when the user clicks Skip.
        """

        label_committed = pyqtSignal(str)
        skipped = pyqtSignal()

        def __init__(self, scheme: LabelingScheme, parent=None):
            super().__init__(parent)
            self._state = StepperState(scheme)
            self._scheme = scheme
            self._buttons: list[QPushButton] = []
            self._shortcut_map: dict[str, str] = {}
            self._build_ui()
            self._refresh()

        # ------------------------------------------------------------------
        def _build_ui(self):
            self._outer = QVBoxLayout(self)
            self._outer.setContentsMargins(0, 0, 0, 0)
            self._outer.setSpacing(6)

            self._header = QLabel()
            self._header.setStyleSheet("color: #cccccc; font-weight: bold;")
            self._outer.addWidget(self._header)

            self._btn_row = QHBoxLayout()
            self._outer.addLayout(self._btn_row)

            nav_row = QHBoxLayout()
            self._back_btn = QPushButton("< Back")
            self._back_btn.setFixedWidth(80)
            self._back_btn.clicked.connect(self._on_back)
            self._skip_btn = QPushButton("Skip >")
            self._skip_btn.setFixedWidth(80)
            self._skip_btn.clicked.connect(self.skipped.emit)
            nav_row.addWidget(self._back_btn)
            nav_row.addStretch()
            nav_row.addWidget(self._skip_btn)
            self._outer.addLayout(nav_row)

        def _refresh(self):
            # Clear existing buttons
            for btn in self._buttons:
                self._btn_row.removeWidget(btn)
                btn.deleteLater()
            self._buttons.clear()
            self._shortcut_map.clear()

            if self._state.is_complete:
                self._header.setText("All factors assigned.")
                self._back_btn.setEnabled(True)
                return

            factor = self._state.current_factor
            idx = self._state.current_factor_index
            self._header.setText(
                f"Factor {idx + 1} of {self._state.total_factors}: <b>{factor.name}</b>"
            )

            for i, label in enumerate(factor.labels):
                btn = QPushButton(label.capitalize())
                btn.setStyleSheet(
                    "QPushButton { background: #2d2d2d; color: #e0e0e0; border: 1px solid #555; "
                    "border-radius: 4px; padding: 6px 12px; } "
                    "QPushButton:hover { background: #007acc; }"
                )
                btn.clicked.connect(lambda checked, lbl=label: self._on_pick(lbl))
                self._btn_row.addWidget(btn)
                self._buttons.append(btn)

                key = factor.shortcut_keys[i] if i < len(factor.shortcut_keys) else None
                if key:
                    self._shortcut_map[key] = label

            self._back_btn.setEnabled(self._state.current_factor_index > 0)

        def _on_pick(self, label: str):
            self._state.pick(label)
            self._refresh()
            if self._state.is_complete:
                self.label_committed.emit(self._state.composite_label)
                self._state.reset()
                self._refresh()

        def _on_back(self):
            self._state.back()
            self._refresh()

        def reset(self):
            self._state.reset()
            self._refresh()

        def handle_key(self, key: str) -> bool:
            """Call from parent keyPressEvent. Returns True if key was consumed."""
            label = self._shortcut_map.get(key.lower())
            if label:
                self._on_pick(label)
                return True
            return False

    return FactorStepperWidget


def get_factor_stepper_class():  # pragma: no cover
    """Lazy factory — import only when Qt is available."""
    return _build_qt_widget


# Expose for direct import when Qt is available
try:
    FactorStepperWidget = _build_qt_widget(None)  # type: ignore[arg-type]
except Exception:
    FactorStepperWidget = None  # type: ignore[assignment,misc]
```

> **Note:** The `FactorStepperWidget` Qt class is built lazily. Tests only use `StepperState` which has no Qt dependency.

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_stepper.py -v
```

Expected: all 6 tests `PASSED`.

**Step 5: Commit**

```bash
git add src/multi_tracker/classkit/gui/widgets/factor_stepper.py tests/test_classkit_stepper.py
git commit -m "feat(classkit): add StepperState and FactorStepperWidget for multi-factor sequential labeling"
```

---

## Task 6: `NewProjectDialog` Preset Selector

**Files:**
- Modify: `src/multi_tracker/classkit/gui/dialogs.py`

This is a UI-only change with no pure-logic testable unit. Verify by launching the app.

**Step 1: Read the current `NewProjectDialog.__init__`**

```bash
# In your editor, open src/multi_tracker/classkit/gui/dialogs.py
# Locate NewProjectDialog.__init__ — it currently has: name, location, description fields
```

**Step 2: Add preset selector to `NewProjectDialog`**

In `NewProjectDialog.__init__`, after the existing form rows, add:

```python
# --- Preset selector ---
from PyQt6.QtWidgets import QComboBox, QStackedWidget

self.preset_combo = QComboBox()
self.preset_combo.addItem("None (free-form labels)", "none")
self.preset_combo.addItem("Head / Tail direction (4 classes)", "head_tail")
self.preset_combo.addItem("Color tags — 1 factor (5 colors)", "color_tag_1")
self.preset_combo.addItem("Color tags — 2 factors (25 composites)", "color_tag_2")
self.preset_combo.addItem("Color tags — 3 factors (125 composites)", "color_tag_3")
self.preset_combo.addItem("Age (young / old)", "age")
self.preset_combo.addItem("Custom...", "custom")
form.addRow("<b>Labeling Preset:</b>", self.preset_combo)

self._scheme_info = QLabel("")
self._scheme_info.setWordWrap(True)
self._scheme_info.setStyleSheet("color: #888888; font-size: 11px;")
form.addRow("", self._scheme_info)

self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
self._on_preset_changed()
```

Add the handler method to `NewProjectDialog`:

```python
def _on_preset_changed(self):
    key = self.preset_combo.currentData()
    from ...classkit.presets import head_tail_preset, color_tag_preset, age_preset
    DEFAULT_COLORS = ["red", "blue", "green", "yellow", "white"]
    info_map = {
        "none": "Free-form: define class labels manually after creating the project.",
        "head_tail": "4 classes: left, right, up, down.",
        "color_tag_1": f"5 classes: {', '.join(DEFAULT_COLORS)}.",
        "color_tag_2": "25 composite classes (tag_1 × tag_2).",
        "color_tag_3": "125 composite classes (tag_1 × tag_2 × tag_3).",
        "age": "2 classes: young, old.",
        "custom": "Define your own factors after project creation.",
    }
    self._scheme_info.setText(info_map.get(key, ""))
```

Update `get_project_info` to return the chosen scheme:

```python
def get_project_info(self):
    from ...classkit.presets import head_tail_preset, color_tag_preset, age_preset
    DEFAULT_COLORS = ["red", "blue", "green", "yellow", "white"]
    key = self.preset_combo.currentData()
    scheme_map = {
        "head_tail": head_tail_preset(),
        "color_tag_1": color_tag_preset(1, DEFAULT_COLORS),
        "color_tag_2": color_tag_preset(2, DEFAULT_COLORS),
        "color_tag_3": color_tag_preset(3, DEFAULT_COLORS),
        "age": age_preset(),
    }
    return {
        "name": self.name_edit.text().strip(),
        "location": self.location_edit.text().strip(),
        "description": self.description_edit.text().strip(),
        "scheme": scheme_map.get(key),  # None for "none" and "custom"
    }
```

**Step 3: Verify manually**

```bash
classkit-labeler
```

Open New Project dialog — verify the preset dropdown appears, info text updates on selection.

**Step 4: Commit**

```bash
git add src/multi_tracker/classkit/gui/dialogs.py
git commit -m "feat(classkit): add labeling preset selector to NewProjectDialog"
```

---

## Task 7: `ClassKitTrainingWorker` and `ClassKitTrainingDialog`

**Files:**
- Modify: `src/multi_tracker/classkit/jobs/task_workers.py`
- Modify: `src/multi_tracker/classkit/gui/dialogs.py`

### 7a — Training Worker

**Step 1: Add `ClassKitTrainingWorker` to `task_workers.py`**

Append after the existing `TrainingWorker` class:

```python
class ClassKitTrainingWorker(QRunnable):
    """Worker for ClassKit classification training (flat or multi-head)."""

    def __init__(
        self,
        *,
        role,           # TrainingRole
        spec,           # TrainingRunSpec (or list of specs for multi-head)
        run_dir: str,
        multi_head: bool = False,
    ):
        super().__init__()
        self.role = role
        self.spec = spec                  # single spec (flat) or list (multi-head)
        self.run_dir = run_dir
        self.multi_head = multi_head
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        from pathlib import Path
        from ...training.runner import run_training

        try:
            self.signals.started.emit()
            results = []
            specs = self.spec if self.multi_head else [self.spec]

            for i, spec in enumerate(specs):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return
                factor_run_dir = Path(self.run_dir) / (f"factor_{i}" if self.multi_head else "run")

                def log_cb(msg, _i=i):
                    prefix = f"[factor {_i}] " if self.multi_head else ""
                    self.signals.progress.emit(0, f"{prefix}{msg}")

                def progress_cb(current, total, _i=i):
                    # For multi-head: report per-factor progress as fraction of total
                    if self.multi_head:
                        overall = int((_i * 100 + current * 100 // max(1, total)) / len(specs))
                    else:
                        overall = current * 100 // max(1, total)
                    self.signals.progress.emit(overall, "")

                result = run_training(
                    spec,
                    factor_run_dir,
                    log_cb=log_cb,
                    progress_cb=progress_cb,
                    should_cancel=lambda: self._canceled,
                )
                results.append(result)
                if not result.get("success"):
                    self.signals.error.emit(f"Training failed for spec {i}")
                    return

            self.signals.success.emit(results)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
```

> Note: `TaskSignals.progress` currently emits `(int, str)` — verify the existing signature in `task_workers.py` and adjust if needed. If it only emits `int`, add an overloaded signal or use a single-int signal and route log lines separately.

**Step 2: Replace `TrainingDialog` with `ClassKitTrainingDialog` in `dialogs.py`**

The existing `TrainingDialog` class is replaced. Keep the name `TrainingDialog` as an alias for backwards compatibility:

```python
class ClassKitTrainingDialog(QDialog):
    """Full training dialog for ClassKit: flat or multi-head, tiny CNN or YOLO-classify."""

    def __init__(self, scheme=None, parent=None):
        super().__init__(parent)
        self._scheme = scheme
        self._result = None
        self.setWindowTitle("Train Classifier")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header
        total = self._scheme.total_classes if self._scheme else "?"
        name = self._scheme.name if self._scheme else "free-form"
        header = QLabel(f"<h2 style='color:#ffffff'>Train Classifier</h2>"
                        f"<p style='color:#888'>Scheme: <b>{name}</b> &nbsp;|&nbsp; "
                        f"Total classes: <b>{total}</b></p>")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(8)

        # Training mode
        self.mode_combo = QComboBox()
        self._populate_modes()
        form.addRow("<b>Training Mode:</b>", self.mode_combo)

        # Device
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU", "cpu")
        try:
            from ...utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE
            if TORCH_CUDA_AVAILABLE:
                self.device_combo.addItem("CUDA GPU", "cuda")
            if MPS_AVAILABLE:
                self.device_combo.addItem("Apple Silicon (MPS)", "mps")
                self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        except Exception:
            pass
        form.addRow("<b>Device:</b>", self.device_combo)

        # Base model (YOLO only)
        self.base_model_combo = QComboBox()
        for m in ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt"]:
            self.base_model_combo.addItem(m, m)
        form.addRow("<b>Base Model (YOLO):</b>", self.base_model_combo)

        # Hyperparams
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(50)
        form.addRow("<b>Epochs:</b>", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 512)
        self.batch_spin.setValue(32)
        form.addRow("<b>Batch Size:</b>", self.batch_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(0.001)
        form.addRow("<b>Learning Rate:</b>", self.lr_spin)

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0.0, 0.5)
        self.val_fraction_spin.setSingleStep(0.05)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.2)
        form.addRow("<b>Val Fraction:</b>", self.val_fraction_spin)

        layout.addLayout(form)

        # Log panel
        from PyQt6.QtWidgets import QPlainTextEdit
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(180)
        self.log_view.setStyleSheet("background:#111; color:#ccc; font-family:monospace; font-size:11px;")
        layout.addWidget(self.log_view)

        # Progress bar
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        self.publish_btn = QPushButton("Publish to models/")
        self.publish_btn.setEnabled(False)
        self.publish_btn.clicked.connect(self._publish)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.publish_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed()

    def _populate_modes(self):
        self.mode_combo.clear()
        if self._scheme is None:
            # No scheme: only embedding-head modes (legacy fallback)
            self.mode_combo.addItem("Flat – Tiny CNN", "flat_tiny")
            self.mode_combo.addItem("Flat – YOLO-classify", "flat_yolo")
            return
        available = self._scheme.training_modes
        labels = {
            "flat_tiny": "Flat – Tiny CNN",
            "flat_yolo": "Flat – YOLO-classify",
            "multihead_tiny": "Multi-head – Tiny CNN (one model per factor)",
            "multihead_yolo": "Multi-head – YOLO-classify (one model per factor)",
        }
        for key in ["flat_tiny", "flat_yolo", "multihead_tiny", "multihead_yolo"]:
            if key in available:
                self.mode_combo.addItem(labels[key], key)

    def _on_mode_changed(self):
        mode = self.mode_combo.currentData() or ""
        is_yolo = "yolo" in mode
        self.base_model_combo.setVisible(is_yolo)

    def _append_log(self, msg: str):
        if msg:
            self.log_view.appendPlainText(msg)
            self.log_view.ensureCursorVisible()

    def _start(self):
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.publish_btn.setEnabled(False)
        self.log_view.clear()
        self.progress_bar.setValue(0)
        # Build spec(s) and dispatch worker — wired in MainWindow.train_classifier
        self.accept()  # placeholder: real wiring done in Task 8

    def _cancel(self):
        if hasattr(self, "_worker") and self._worker:
            self._worker.cancel()

    def _publish(self):
        # Publish logic wired in Task 8 after training completes
        pass

    def get_settings(self) -> dict:
        return {
            "mode": self.mode_combo.currentData(),
            "device": self.device_combo.currentData(),
            "base_model": self.base_model_combo.currentData(),
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "lr": self.lr_spin.value(),
            "val_fraction": self.val_fraction_spin.value(),
        }


# Backwards-compat alias
TrainingDialog = ClassKitTrainingDialog
```

**Step 3: Verify dialog opens**

```bash
classkit-labeler
# Open an existing project, click Train Classifier
# Verify dialog shows training mode selector, log panel, buttons
```

**Step 4: Commit**

```bash
git add src/multi_tracker/classkit/jobs/task_workers.py src/multi_tracker/classkit/gui/dialogs.py
git commit -m "feat(classkit): add ClassKitTrainingWorker and ClassKitTrainingDialog with live log and multi-mode support"
```

---

## Task 8: Wire Stepper and Training Into `MainWindow`

**Files:**
- Modify: `src/multi_tracker/classkit/gui/mainwindow.py`

This task has two parts: (A) wiring the stepper into the label panel, and (B) fully wiring the training dialog with dataset export, worker dispatch, and publish.

### 8A — Wire `FactorStepperWidget`

**Step 1: Find where label buttons are built**

In `mainwindow.py`, locate `rebuild_label_buttons` (around line search: `rebuild_label_buttons`). This method currently builds flat `QPushButton`s for each class.

**Step 2: Extend `rebuild_label_buttons` to insert the stepper when a scheme has >1 factor**

```python
def rebuild_label_buttons(self):
    # Clear existing widgets from the label button container
    # (find the container widget — typically self._label_button_container or similar)
    ...
    scheme = getattr(self.config.project, "scheme", None) if self.config else None

    if scheme is not None and len(scheme.factors) > 1:
        from ..gui.widgets.factor_stepper import _build_qt_widget
        StepperClass = _build_qt_widget(scheme)
        self._stepper = StepperClass(scheme, parent=self)
        self._stepper.label_committed.connect(self._on_stepper_label_committed)
        self._stepper.skipped.connect(self.on_next_image)
        # add self._stepper to the label panel layout
        ...
    else:
        self._stepper = None
        # existing flat button logic unchanged
        ...
```

**Step 3: Add `_on_stepper_label_committed`**

```python
def _on_stepper_label_committed(self, composite_label: str):
    """Called when the stepper finishes all factors for the current image."""
    self._set_label_for_index(self._nav_index, composite_label)
    self.on_next_image()
```

**Step 4: Extend `setup_label_shortcuts` for multi-factor mode**

When `self._stepper` is not None, route key events to `self._stepper.handle_key(key)` in `eventFilter`. The stepper handles its own rebinding internally.

### 8B — Wire Training Dialog End-to-End

**Step 1: Rewrite `train_classifier` in `mainwindow.py`**

Replace the current body with:

```python
def train_classifier(self):
    """Open ClassKitTrainingDialog, export dataset, run training, offer publish."""
    self._flush_pending_label_updates(force=True)

    if not self.image_labels or not any(self.image_labels):
        QMessageBox.warning(self, "No Labels", "Label some images before training.")
        return

    scheme = getattr(self.config.project, "scheme", None) if self.config else None

    from .dialogs import ClassKitTrainingDialog
    dialog = ClassKitTrainingDialog(scheme=scheme, parent=self)

    # Connect live log and progress
    from ..jobs.task_workers import ClassKitTrainingWorker
    import tempfile, os
    from pathlib import Path

    def _do_train():
        settings = dialog.get_settings()
        mode = settings["mode"]

        # 1. Export dataset
        from ..export.ultralytics_classify import export_ultralytics_classify
        export_dir = Path(self.project_path) / ".classkit_train_export"
        export_dir.mkdir(parents=True, exist_ok=True)

        labeled = [(p, l) for p, l in zip(self.image_paths, self.image_labels) if l]
        if not labeled:
            QMessageBox.warning(dialog, "No Labels", "No labeled images found.")
            return

        images = [Path(p) for p, _ in labeled]
        labels_str = [l for _, l in labeled]

        if scheme is not None:
            # Map composite string labels to integer indices (sorted for determinism)
            unique = sorted(set(labels_str))
            label_map = {s: i for i, s in enumerate(unique)}
            int_labels = [label_map[l] for l in labels_str]
            class_names = {i: s for s, i in label_map.items()}
        else:
            unique = sorted(set(labels_str))
            label_map = {s: i for i, s in enumerate(unique)}
            int_labels = [label_map[l] for l in labels_str]
            class_names = {i: s for s, i in label_map.items()}

        val_frac = settings["val_fraction"]
        n_val = max(1, int(len(images) * val_frac)) if val_frac > 0 else 0
        train_images = images[n_val:]
        train_labels = int_labels[n_val:]
        val_images = images[:n_val]
        val_labels = int_labels[:n_val]

        export_ultralytics_classify(
            export_dir, train_images, train_labels,
            val_images, val_labels, class_names=class_names,
        )

        # 2. Build spec(s)
        from ...training.contracts import (
            TrainingRole, TrainingRunSpec, TrainingHyperParams,
            TinyHeadTailParams, AugmentationProfile, PublishPolicy,
        )
        multi_head = mode.startswith("multihead")
        is_yolo = "yolo" in mode

        role_map = {
            "flat_tiny": TrainingRole.CLASSIFY_FLAT_TINY,
            "flat_yolo": TrainingRole.CLASSIFY_FLAT_YOLO,
            "multihead_tiny": TrainingRole.CLASSIFY_MULTIHEAD_TINY,
            "multihead_yolo": TrainingRole.CLASSIFY_MULTIHEAD_YOLO,
        }
        role = role_map[mode]
        run_dir = Path(self.project_path) / ".classkit_runs" / mode

        if multi_head and scheme is not None:
            # Build one spec per factor — each gets a single-factor dataset
            specs = []
            for fi, factor in enumerate(scheme.factors):
                factor_export = export_dir.parent / f".classkit_train_export_f{fi}"
                _export_single_factor(
                    factor_export, images, labels_str, fi, scheme, val_frac
                )
                specs.append(TrainingRunSpec(
                    role=role,
                    source_datasets=[],
                    derived_dataset_dir=str(factor_export),
                    base_model=settings["base_model"] if is_yolo else "",
                    hyperparams=TrainingHyperParams(
                        epochs=settings["epochs"], batch=settings["batch"], lr0=settings["lr"]
                    ),
                    tiny_params=TinyHeadTailParams(
                        epochs=settings["epochs"], batch=settings["batch"], lr=settings["lr"]
                    ),
                    device=settings["device"],
                ))
            worker = ClassKitTrainingWorker(
                role=role, spec=specs, run_dir=str(run_dir), multi_head=True
            )
        else:
            spec = TrainingRunSpec(
                role=role,
                source_datasets=[],
                derived_dataset_dir=str(export_dir),
                base_model=settings["base_model"] if is_yolo else "",
                hyperparams=TrainingHyperParams(
                    epochs=settings["epochs"], batch=settings["batch"], lr0=settings["lr"]
                ),
                tiny_params=TinyHeadTailParams(
                    epochs=settings["epochs"], batch=settings["batch"], lr=settings["lr"]
                ),
                device=settings["device"],
            )
            worker = ClassKitTrainingWorker(
                role=role, spec=spec, run_dir=str(run_dir), multi_head=False
            )

        dialog._worker = worker
        worker.signals.progress.connect(lambda pct, msg: (
            dialog.progress_bar.setValue(pct),
            dialog._append_log(msg) if msg else None,
        ))
        worker.signals.success.connect(lambda results: (
            dialog.publish_btn.setEnabled(True),
            dialog._append_log("Training complete."),
            setattr(dialog, "_train_results", results),
        ))
        worker.signals.error.connect(lambda err: dialog._append_log(f"ERROR: {err}"))
        worker.signals.finished.connect(lambda: (
            dialog.start_btn.setEnabled(True),
            dialog.cancel_btn.setEnabled(False),
        ))
        self.threadpool.start(worker)

    dialog.start_btn.clicked.disconnect()
    dialog.start_btn.clicked.connect(_do_train)
    dialog.exec()
```

Add a helper method `_export_single_factor`:

```python
def _export_single_factor(self, export_dir, images, labels_str, factor_idx, scheme, val_frac):
    """Export a dataset with only factor_idx's labels (for multi-head training)."""
    from pathlib import Path
    from ..export.ultralytics_classify import export_ultralytics_classify

    factor = scheme.factors[factor_idx]
    factor_labels = [scheme.decode_label(l)[factor_idx] for l in labels_str]
    unique = sorted(set(factor_labels))
    label_map = {s: i for i, s in enumerate(unique)}
    int_labels = [label_map[l] for l in factor_labels]
    class_names = {i: s for s, i in label_map.items()}

    n_val = max(1, int(len(images) * val_frac)) if val_frac > 0 else 0
    export_ultralytics_classify(
        export_dir, images[n_val:], int_labels[n_val:],
        images[:n_val], int_labels[:n_val], class_names=class_names,
    )
```

Wire `_publish` in `ClassKitTrainingDialog` to call `publish_trained_model` for each result in `dialog._train_results`.

**Step 2: Verify end-to-end manually**

```bash
classkit-labeler
# Load a project with labeled images
# Click Train → choose Flat Tiny → Start Training
# Verify log fills, progress bar moves, Publish activates on completion
# Click Publish — verify models/ directory is populated
```

**Step 3: Run test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

**Step 4: Commit**

```bash
git add src/multi_tracker/classkit/gui/mainwindow.py src/multi_tracker/classkit/gui/dialogs.py
git commit -m "feat(classkit): wire FactorStepperWidget and ClassKitTrainingDialog into MainWindow"
```

---

## Task 9: MAT Migration — Strip Classification from `TrainYoloDialog`

**Files:**
- Modify: `src/multi_tracker/gui/dialogs/train_yolo_dialog.py`
- Modify: `src/multi_tracker/training/contracts.py` (deprecation comment)

**Step 1: Remove classification-related UI from `TrainYoloDialog`**

In `_build_roles_group`:
- Remove checkboxes / rows for `HEADTAIL_TINY` and `HEADTAIL_YOLO`
- Add a `QLabel` info banner in their place:

```python
info = QLabel(
    "<b>Classification training has moved to ClassKit.</b><br>"
    "Use <code>classkit-labeler</code> to label data and train classifiers."
)
info.setWordWrap(True)
info.setStyleSheet("padding: 8px; background: #1a2a1a; border-left: 3px solid #4caf50; color: #aaa;")
roles_layout.addWidget(info)
```

**Step 2: Remove head-tail override UI**

Remove the following methods and their UI connections from `TrainYoloDialog`:
- `_choose_headtail_override`
- `_set_headtail_source`
- Any `headtail_override` widgets from `_build_sources_group` or `_build_config_group`

**Step 3: Remove classify branches from `_base_model_for_role` and `_build_role_datasets`**

In `_base_model_for_role`, remove the `HEADTAIL_YOLO` / `HEADTAIL_TINY` branches.
In `_build_role_datasets`, remove the headtail dataset build branch.
In `_selected_roles`, remove `HEADTAIL_YOLO` / `HEADTAIL_TINY` from the list.

**Step 4: Mark roles deprecated in `contracts.py`**

```python
class TrainingRole(str, Enum):
    """Canonical training roles supported by MAT."""
    OBB_DIRECT = "obb_direct"
    SEQ_DETECT = "seq_detect"
    SEQ_CROP_OBB = "seq_crop_obb"
    # Deprecated: classification training has moved to ClassKit (classkit-labeler)
    HEADTAIL_YOLO = "headtail_yolo"   # kept for registry backwards-compat only
    HEADTAIL_TINY = "headtail_tiny"   # kept for registry backwards-compat only
    # ClassKit classification roles
    CLASSIFY_FLAT_YOLO = "classify_flat_yolo"
    CLASSIFY_FLAT_TINY = "classify_flat_tiny"
    CLASSIFY_MULTIHEAD_YOLO = "classify_multihead_yolo"
    CLASSIFY_MULTIHEAD_TINY = "classify_multihead_tiny"
```

**Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

**Step 6: Launch MAT and verify**

```bash
mat
# Open training dialog (Train → YOLO)
# Verify no head-tail checkboxes appear
# Verify info banner shows "Classification training has moved to ClassKit"
```

**Step 7: Commit**

```bash
git add src/multi_tracker/gui/dialogs/train_yolo_dialog.py src/multi_tracker/training/contracts.py
git commit -m "feat(mat): migrate classification training to ClassKit; remove head-tail roles from TrainYoloDialog"
```

---

## Final Verification

```bash
# Run full test suite
python -m pytest tests/ -v --tb=short

# Check formatting
make format-check

# Check lint
make lint

# Launch ClassKit and walk through a full workflow:
# 1. New project → pick "Color tags — 2 factors" preset
# 2. Ingest images, compute embeddings
# 3. Label ~20 images using the stepper (pick tag_1, then tag_2 per image)
# 4. Train → Flat Tiny → Start → Publish
# 5. Verify models/tiny-classify/color_tags_2factor/ exists with .pth file
# 6. Verify model_registry.json contains new entry

classkit-labeler
```

---

## Task Summary

| # | What | Key files | Test |
|---|---|---|---|
| 1 | Data model: `Factor`, `LabelingScheme` | `schemas.py` | `test_classkit_scheme.py` |
| 2 | Preset factories | `presets.py` | `test_classkit_scheme.py` |
| 3 | New `TrainingRole` values + `model_publish` extension | `contracts.py`, `model_publish.py` | `test_classkit_publish.py` |
| 4 | Generalize `_iter_classify_samples` + `_train_tiny_classify` | `runner.py` | `test_classkit_tiny_train.py` |
| 5 | `StepperState` + `FactorStepperWidget` | `factor_stepper.py` | `test_classkit_stepper.py` |
| 6 | `NewProjectDialog` preset selector | `dialogs.py` | manual |
| 7 | `ClassKitTrainingDialog` + `ClassKitTrainingWorker` | `dialogs.py`, `task_workers.py` | manual |
| 8 | Wire stepper + training into `MainWindow` | `mainwindow.py` | manual + full suite |
| 9 | Strip classification from MAT `TrainYoloDialog` | `train_yolo_dialog.py`, `contracts.py` | manual + full suite |
