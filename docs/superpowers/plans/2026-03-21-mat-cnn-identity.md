# MAT CNN Identity — ClassKit Model Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "CNN Classifier" identity method to MAT that imports ClassKit-trained models, runs a precompute phase on OBB crops, and adjusts Hungarian assignment costs using match bonus / mismatch penalty.

**Architecture:** A new `core/identity/cnn_identity.py` module owns `CNNIdentityConfig`, `ClassPrediction`, `CNNIdentityCache`, `CNNIdentityBackend`, and `TrackCNNHistory`; `worker.py` adds a precompute phase following the AprilTag pattern; `hungarian.py` extends its cost loop with CNN identity bonus/penalty; `main_window.py` renames "Color Tags (YOLO)" to "CNN Classifier" and adds the settings panel and import handler.

**Prerequisite:** The `src/multi_tracker/training/torchvision_model.py` module created in Spec A (ClassKit Extended Training) must exist before this plan's `CNNIdentityBackend` can load torchvision `.pth` checkpoints. Implement Spec A first, or add a `try/except ImportError` guard around the torchvision import path in `_load_pth` if implementing in parallel.

**Config key casing convention:** `configs/default.json` uses lowercase keys (e.g. `cnn_classifier_model_path`) for UI persistence. The params dict `p` passed to the worker and assigner uses UPPERCASE keys (e.g. `CNN_CLASSIFIER_MODEL_PATH`) — set explicitly in `main_window.py`'s params-building block (line ~14233). There is no automatic casing conversion. Follow the same pattern as `TAG_MATCH_BONUS` / `COLOR_TAG_MODEL_PATH` — add uppercase keys to the params dict explicitly.

**Tech Stack:** Python, PyTorch, torchvision, onnxruntime, numpy, PySide6, ultralytics (for YOLO .pt inference), existing MAT infrastructure.

**File Map:**

| File | Action | Role |
|---|---|---|
| `src/multi_tracker/core/identity/cnn_identity.py` | Create | Config, prediction dataclasses, cache, backend, history |
| `src/multi_tracker/core/tracking/worker.py` | Modify | Add CNN identity precompute phase |
| `src/multi_tracker/core/assigners/hungarian.py` | Modify | Add CNN identity match bonus / mismatch penalty |
| `src/multi_tracker/gui/main_window.py` | Modify | Rename identity method, settings panel, import handler |
| `src/multi_tracker/gui/dialogs/cnn_identity_import_dialog.py` | Create | `CNNIdentityImportDialog` |
| `src/multi_tracker/gui/dialogs/__init__.py` | Modify | Export `CNNIdentityImportDialog` |
| `configs/default.json` | Modify | Add 8 new `cnn_classifier_*` config keys |
| `tests/test_mat_cnn_identity.py` | Create | Full test suite |

---

## Task 1 — `cnn_identity.py`: config, prediction, cache, backend

**Files:**
- Create: `src/multi_tracker/core/identity/cnn_identity.py`
- Create (start): `tests/test_mat_cnn_identity.py`

### Step 1.1 — Write failing tests

- [ ] Create `tests/test_mat_cnn_identity.py` with the following content:

```python
"""Tests for MAT CNN identity method."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CNNIdentityConfig tests
# ---------------------------------------------------------------------------

def test_cnn_identity_config_defaults():
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig
    cfg = CNNIdentityConfig()
    assert cfg.model_path == ""
    assert cfg.confidence == 0.5
    assert cfg.label == ""
    assert cfg.batch_size == 64
    assert cfg.crop_padding == 0.1
    assert cfg.match_bonus == 20.0
    assert cfg.mismatch_penalty == 50.0
    assert cfg.window == 10


def test_cnn_identity_config_custom():
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig
    cfg = CNNIdentityConfig(model_path="/tmp/model.pth", confidence=0.8, window=5)
    assert cfg.model_path == "/tmp/model.pth"
    assert cfg.confidence == 0.8
    assert cfg.window == 5


# ---------------------------------------------------------------------------
# ClassPrediction tests
# ---------------------------------------------------------------------------

def test_class_prediction_fields():
    from multi_tracker.core.identity.cnn_identity import ClassPrediction
    p = ClassPrediction(class_name="tag_3", confidence=0.92, det_index=0)
    assert p.class_name == "tag_3"
    assert p.confidence == pytest.approx(0.92)
    assert p.det_index == 0


def test_class_prediction_none_class_name():
    from multi_tracker.core.identity.cnn_identity import ClassPrediction
    p = ClassPrediction(class_name=None, confidence=0.3, det_index=2)
    assert p.class_name is None


# ---------------------------------------------------------------------------
# CNNIdentityCache round-trip tests
# ---------------------------------------------------------------------------

def test_cnn_identity_cache_roundtrip(tmp_path):
    from multi_tracker.core.identity.cnn_identity import CNNIdentityCache, ClassPrediction
    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    preds = [
        ClassPrediction(class_name="tag_0", confidence=0.9, det_index=0),
        ClassPrediction(class_name=None, confidence=0.3, det_index=1),
    ]
    cache.save(5, preds)
    loaded = cache.load(5)
    assert len(loaded) == 2
    assert loaded[0].class_name == "tag_0"
    assert loaded[0].confidence == pytest.approx(0.9)
    assert loaded[0].det_index == 0
    assert loaded[1].class_name is None
    assert loaded[1].det_index == 1


def test_cnn_identity_cache_exists(tmp_path):
    from multi_tracker.core.identity.cnn_identity import CNNIdentityCache, ClassPrediction
    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    assert not cache.exists()
    cache.save(0, [ClassPrediction(class_name="tag_0", confidence=0.9, det_index=0)])
    assert cache.exists()


def test_cnn_identity_cache_empty_frame(tmp_path):
    from multi_tracker.core.identity.cnn_identity import CNNIdentityCache
    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    cache.save(10, [])
    loaded = cache.load(10)
    assert loaded == []


def test_cnn_identity_cache_missing_frame_returns_empty(tmp_path):
    from multi_tracker.core.identity.cnn_identity import CNNIdentityCache, ClassPrediction
    cache_path = tmp_path / "cnn_identity.npz"
    cache = CNNIdentityCache(str(cache_path))
    cache.save(0, [ClassPrediction(class_name="tag_0", confidence=0.9, det_index=0)])
    loaded = cache.load(99)  # frame 99 not saved
    assert loaded == []


# ---------------------------------------------------------------------------
# CNNIdentityBackend (mocked) tests
# ---------------------------------------------------------------------------

def test_backend_predict_batch_cardinality():
    """predict_batch() must return exactly one ClassPrediction per input crop."""
    import numpy as np
    from unittest.mock import patch
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig, CNNIdentityBackend

    cfg = CNNIdentityConfig(model_path="/tmp/m.pth", confidence=0.5)
    crops = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    backend = CNNIdentityBackend(cfg, model_path="/tmp/m.pth", compute_runtime="cpu")

    # Mock _ensure_loaded and _infer_fn together to avoid touching disk
    fixed_logits = np.array([[1.0, 2.0, 0.5]] * 3, dtype=np.float32)

    def fake_ensure_loaded():
        backend._loaded = True
        backend._class_names = ["tag_0", "tag_1", "no_tag"]
        backend._input_size = (64, 64)
        backend._infer_fn = lambda batch_np: fixed_logits

    with patch.object(backend, "_ensure_loaded", fake_ensure_loaded):
        results = backend.predict_batch(crops)

    assert len(results) == len(crops)


def test_backend_below_confidence_returns_none_class():
    """Predictions below confidence threshold return class_name=None."""
    from multi_tracker.core.identity.cnn_identity import CNNIdentityConfig, ClassPrediction

    # A ClassPrediction with confidence below threshold should have class_name=None
    # This tests the contract, not the implementation (backend internals tested separately)
    pred = ClassPrediction(class_name=None, confidence=0.3, det_index=0)
    assert pred.class_name is None


# ---------------------------------------------------------------------------
# Checkpoint metadata extraction tests (for _handle_add_new_cnn_identity_model)
# ---------------------------------------------------------------------------

def test_pth_checkpoint_metadata_extraction(tmp_path):
    """Verify that .pth checkpoint fields are correctly extracted during import."""
    import torch
    from pathlib import Path

    ckpt = {
        "arch": "resnet18",
        "class_names": ["tag_0", "tag_1", "no_tag"],
        "factor_names": [],
        "input_size": (224, 224),
        "num_classes": 3,
        "model_state_dict": {},
        "best_val_acc": 0.95,
        "history": {},
        "trainable_layers": 0,
        "backbone_lr_scale": 0.1,
    }
    ckpt_path = tmp_path / "model.pth"
    torch.save(ckpt, str(ckpt_path))

    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert loaded["arch"] == "resnet18"
    assert loaded["class_names"] == ["tag_0", "tag_1", "no_tag"]
    assert loaded["num_classes"] == 3
    assert list(loaded["input_size"]) == [224, 224]


def test_registry_entry_format_after_import(tmp_path):
    """Registry entry for a CNN identity model has all required fields."""
    import json
    from datetime import datetime

    entry = {
        "arch": "convnext_tiny",
        "num_classes": 11,
        "class_names": [f"tag_{i}" for i in range(10)] + ["no_tag"],
        "factor_names": [],
        "input_size": [224, 224],
        "species": "ant",
        "classification_label": "apriltag",
        "added_at": datetime.now().isoformat(),
        "task_family": "classify",
        "usage_role": "cnn_identity",
    }
    registry_path = tmp_path / "model_registry.json"
    registry = {"classification/identity/test.pth": entry}
    registry_path.write_text(json.dumps(registry))

    loaded = json.loads(registry_path.read_text())
    loaded_entry = loaded["classification/identity/test.pth"]
    required = {"arch", "num_classes", "class_names", "factor_names", "input_size",
                "species", "classification_label", "added_at", "task_family", "usage_role"}
    assert required.issubset(set(loaded_entry.keys()))
    assert loaded_entry["usage_role"] == "cnn_identity"
    assert loaded_entry["num_classes"] == 11
```

### Step 1.2 — Run tests to verify they fail

- [ ] Run:
```bash
cd "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker"
conda run -n multi-animal-tracker-mps python -m pytest tests/test_mat_cnn_identity.py -v 2>&1 | head -20
```
Expected: `ImportError` — `cnn_identity` module does not exist.

### Step 1.3 — Create `src/multi_tracker/core/identity/cnn_identity.py`

- [ ] Create the file at `src/multi_tracker/core/identity/cnn_identity.py`:

```python
"""CNN identity backend for MAT: config, predictions, cache, and inference backend.

Pure Python — no Qt dependency.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CNNIdentityConfig:
    """Configuration for CNN Classifier identity method."""
    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    crop_padding: float = 0.1
    match_bonus: float = 20.0
    mismatch_penalty: float = 50.0
    window: int = 10


@dataclass
class ClassPrediction:
    """Per-detection CNN classification result for one frame."""
    class_name: str | None   # None if below confidence threshold
    confidence: float
    det_index: int           # which detection slot in the frame


# ---------------------------------------------------------------------------
# CNNIdentityCache
# ---------------------------------------------------------------------------

_SENTINEL_NONE = "__NONE__"  # stored in npz when class_name is None


class CNNIdentityCache:
    """Persistent .npz cache of per-frame CNN identity predictions.

    Data is written lazily — call ``save()`` for each frame during precompute,
    then ``load()`` during the tracking loop.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        # In-memory dict, flushed to disk on every save()
        self._data: dict[str, Any] = {}
        if self._path.exists():
            raw = np.load(str(self._path), allow_pickle=True)
            self._data = dict(raw)

    def exists(self) -> bool:
        return self._path.exists()

    def save(self, frame_idx: int, predictions: list[ClassPrediction]) -> None:
        """Persist predictions for *frame_idx*. Overwrites existing data."""
        if not predictions:
            self._data[f"f{frame_idx}_det"] = np.array([], dtype=np.int32)
            self._data[f"f{frame_idx}_cls"] = np.array([], dtype=object)
            self._data[f"f{frame_idx}_conf"] = np.array([], dtype=np.float32)
        else:
            det_arr = np.array([p.det_index for p in predictions], dtype=np.int32)
            cls_arr = np.array(
                [p.class_name if p.class_name is not None else _SENTINEL_NONE for p in predictions],
                dtype=object,
            )
            conf_arr = np.array([p.confidence for p in predictions], dtype=np.float32)
            self._data[f"f{frame_idx}_det"] = det_arr
            self._data[f"f{frame_idx}_cls"] = cls_arr
            self._data[f"f{frame_idx}_conf"] = conf_arr
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(self._path), **self._data)

    def load(self, frame_idx: int) -> list[ClassPrediction]:
        """Return saved predictions for *frame_idx*, or [] if not found."""
        key_det = f"f{frame_idx}_det"
        if key_det not in self._data:
            return []
        det_arr = self._data[key_det]
        cls_arr = self._data[f"f{frame_idx}_cls"]
        conf_arr = self._data[f"f{frame_idx}_conf"]
        results = []
        for i in range(len(det_arr)):
            raw_cls = str(cls_arr[i])
            class_name = None if raw_cls == _SENTINEL_NONE else raw_cls
            results.append(
                ClassPrediction(
                    class_name=class_name,
                    confidence=float(conf_arr[i]),
                    det_index=int(det_arr[i]),
                )
            )
        return results


# ---------------------------------------------------------------------------
# CNNIdentityBackend
# ---------------------------------------------------------------------------

class CNNIdentityBackend:
    """Wraps model loading and batch inference for CNN identity classification.

    Supports:
    - .pth checkpoints (TinyClassifier or torchvision — detected via 'arch' field)
    - YOLO .pt checkpoints (ultralytics)
    Runtime selection via compute_runtime (cpu/mps/cuda). ONNX artifacts are
    derived lazily from .pth and cached alongside the source file.
    """

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str,
        compute_runtime: str = "cpu",
    ) -> None:
        self._config = config
        self._model_path = str(model_path)
        self._compute_runtime = str(compute_runtime or "cpu")
        self._model = None
        self._class_names: list[str] = []
        self._input_size: tuple[int, int] = (224, 224)
        self._arch: str = "tinyclassifier"
        self._is_yolo: bool = self._model_path.endswith(".pt")
        self._loaded: bool = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        use_onnx = self._compute_runtime in (
            "onnx_cpu", "onnx_cuda", "onnx_rocm", "tensorrt"
        )
        device = self._torch_device(self._compute_runtime)

        if self._is_yolo:
            self._load_yolo(use_onnx, device)
        else:
            self._load_pth(use_onnx, device)
        self._loaded = True

    def _torch_device(self, rt: str) -> str:
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"
        return "cpu"

    def _load_pth(self, use_onnx: bool, device: str) -> None:
        import torch
        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        self._class_names = ckpt.get("class_names", [])
        raw_size = ckpt.get("input_size", (224, 224))
        self._input_size = tuple(raw_size) if isinstance(raw_size, (list, tuple)) else (raw_size, raw_size)
        self._arch = ckpt.get("arch", "tinyclassifier")

        if use_onnx:
            onnx_path = self._derive_onnx(ckpt, device)
            import onnxruntime as ort
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "cuda" in self._compute_runtime
                else ["CPUExecutionProvider"]
            )
            self._model = ort.InferenceSession(onnx_path, providers=providers)
            self._infer_fn = self._infer_onnx
        else:
            if self._arch == "tinyclassifier":
                from multi_tracker.training.tiny_model import load_tiny_classifier
                self._model = load_tiny_classifier(self._model_path, device=device)
            else:
                # Requires Spec A (ClassKit Extended Training) to be implemented first
                from multi_tracker.training.torchvision_model import load_torchvision_classifier
                self._model, _ = load_torchvision_classifier(self._model_path, device=device)
            self._infer_fn = lambda batch_np, dev=device: self._infer_torch(batch_np, dev)

    def _load_yolo(self, use_onnx: bool, device: str) -> None:
        from ultralytics import YOLO
        yolo = YOLO(self._model_path)
        names = yolo.names
        self._class_names = [names[i] for i in sorted(names.keys())]
        self._input_size = (224, 224)
        self._arch = "yolo"
        if use_onnx:
            onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
            if not os.path.exists(onnx_path):
                yolo.export(format="onnx", imgsz=224)
            import onnxruntime as ort
            self._model = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            self._infer_fn = self._infer_onnx
        else:
            self._model = yolo
            self._infer_fn = self._infer_yolo

    def _derive_onnx(self, ckpt: dict, device: str) -> str:
        """Lazy-derive ONNX from .pth. Returns path to .onnx file."""
        onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
        if os.path.exists(onnx_path):
            return onnx_path
        if self._arch == "tinyclassifier":
            from multi_tracker.training.tiny_model import load_tiny_classifier
            import torch
            model = load_tiny_classifier(self._model_path, device="cpu")
            h, w = self._input_size
            dummy = torch.zeros(1, 3, h, w)
            torch.onnx.export(
                model, dummy, onnx_path,
                opset_version=17, input_names=["input"], output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        else:
            from multi_tracker.training.torchvision_model import (
                load_torchvision_classifier, export_torchvision_to_onnx
            )
            model, loaded_ckpt = load_torchvision_classifier(self._model_path, device="cpu")
            export_torchvision_to_onnx(model, loaded_ckpt, onnx_path)
        return onnx_path

    def _infer_torch(self, batch_np: np.ndarray, device: str) -> np.ndarray:
        import torch
        t = torch.tensor(batch_np).to(device)
        with torch.no_grad():
            logits = self._model(t).cpu().numpy()
        return logits

    def _infer_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        input_name = self._model.get_inputs()[0].name
        return self._model.run(None, {input_name: batch_np.astype(np.float32)})[0]

    def _infer_yolo(self, batch_np: np.ndarray) -> np.ndarray:
        # YOLO classify expects list of numpy arrays or paths; use batch inference
        results = self._model(list(batch_np), verbose=False)
        probs = np.array([r.probs.data.cpu().numpy() for r in results])
        # Return as logits (log of probs to be consistent with softmax path below)
        return np.log(np.clip(probs, 1e-9, 1.0))

    def _preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """Resize and normalize crops to the model's expected input format."""
        import cv2
        from torchvision import transforms
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        h, w = self._input_size
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        from PIL import Image
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(np.zeros((3, h, w), dtype=np.float32))
                continue
            img_bgr = cv2.resize(crop, (w, h))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensors.append(tf(pil_img).numpy())
        return np.stack(tensors).astype(np.float32)

    def predict_batch(self, crops: list[np.ndarray]) -> list[ClassPrediction]:
        """Run inference on *crops*. Returns one ClassPrediction per crop."""
        if not crops:
            return []
        self._ensure_loaded()
        batch_np = self._preprocess(crops)
        logits = self._infer_fn(batch_np)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for i, prob in enumerate(probs):
            best_idx = int(np.argmax(prob))
            best_conf = float(prob[best_idx])
            if best_conf >= self._config.confidence and self._class_names:
                class_name = self._class_names[best_idx]
            else:
                class_name = None
            results.append(ClassPrediction(
                class_name=class_name,
                confidence=best_conf,
                det_index=i,
            ))
        return results

    def close(self) -> None:
        self._model = None
        self._loaded = False
```

### Step 1.4 — Run tests

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_mat_cnn_identity.py -v 2>&1 | tail -20
```
Expected: all tests PASS. Note: `test_backend_predict_batch_cardinality` requires care — the mock must match the internal `_infer_fn` pattern. If it fails due to internal attribute mocking, adjust the test to use the `predict_batch` contract with a fully mocked backend.

### Step 1.5 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/core/identity/cnn_identity.py tests/test_mat_cnn_identity.py
git commit -m "$(cat <<'EOF'
feat(mat): add CNNIdentityConfig, ClassPrediction, CNNIdentityCache, CNNIdentityBackend

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2 — `TrackCNNHistory` + Hungarian cost adjustment

**Files:**
- Modify: `src/multi_tracker/core/identity/cnn_identity.py` (append `TrackCNNHistory`)
- Modify: `src/multi_tracker/core/assigners/hungarian.py`
- Test: `tests/test_mat_cnn_identity.py` (append)

### Step 2.1 — Write failing tests

- [ ] Append to `tests/test_mat_cnn_identity.py`:

```python
# ---------------------------------------------------------------------------
# TrackCNNHistory tests
# ---------------------------------------------------------------------------

def test_track_cnn_history_majority_vote():
    """3 out of 5 frames predict the same class → that class is the identity."""
    from multi_tracker.core.identity.cnn_identity import TrackCNNHistory
    hist = TrackCNNHistory(n_tracks=1, window=10)
    hist.record(0, 1, "tag_3")
    hist.record(0, 2, "tag_3")
    hist.record(0, 3, "tag_3")
    hist.record(0, 4, "tag_0")
    hist.record(0, 5, "tag_1")
    assert hist.majority_class(0) == "tag_3"


def test_track_cnn_history_no_observations_returns_none():
    from multi_tracker.core.identity.cnn_identity import TrackCNNHistory
    hist = TrackCNNHistory(n_tracks=2, window=10)
    assert hist.majority_class(0) is None
    assert hist.majority_class(1) is None


def test_track_cnn_history_tied_returns_none():
    """Exact tie in majority vote → no clear identity."""
    from multi_tracker.core.identity.cnn_identity import TrackCNNHistory
    hist = TrackCNNHistory(n_tracks=1, window=10)
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_1")
    # Tied: no majority → returns None
    assert hist.majority_class(0) is None


def test_track_cnn_history_window_drops_old():
    """Observations outside the window are not counted."""
    from multi_tracker.core.identity.cnn_identity import TrackCNNHistory
    hist = TrackCNNHistory(n_tracks=1, window=3)
    # Old observations (frames 0-2): tag_0 wins
    hist.record(0, 0, "tag_0")
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_0")
    # New observations (frames 3-5): tag_1 wins; frame 0-2 drop out
    hist.record(0, 3, "tag_1")
    hist.record(0, 4, "tag_1")
    hist.record(0, 5, "tag_1")
    assert hist.majority_class(0) == "tag_1"


def test_track_cnn_history_build_list():
    from multi_tracker.core.identity.cnn_identity import TrackCNNHistory
    hist = TrackCNNHistory(n_tracks=3, window=10)
    hist.record(0, 1, "tag_0")
    hist.record(0, 2, "tag_0")
    hist.record(2, 1, "no_tag")
    identity_list = hist.build_track_identity_list(3)
    assert identity_list[0] == "tag_0"
    assert identity_list[1] is None
    assert identity_list[2] == "no_tag"


# ---------------------------------------------------------------------------
# Hungarian cost adjustment tests
# ---------------------------------------------------------------------------

def test_hungarian_cnn_match_bonus_applied():
    """When detection class == track identity, cost decreases by match_bonus."""
    from multi_tracker.core.identity.cnn_identity import _apply_cnn_identity_cost

    cost = 50.0
    adjusted = _apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_3",
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0 - 20.0)


def test_hungarian_cnn_mismatch_penalty_applied():
    """When detection class != track identity, cost increases by mismatch_penalty."""
    from multi_tracker.core.identity.cnn_identity import _apply_cnn_identity_cost

    cost = 50.0
    adjusted = _apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_0",
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0 + 50.0)


def test_hungarian_cnn_no_adjustment_when_det_none():
    """No cost adjustment when det_class is None (low confidence)."""
    from multi_tracker.core.identity.cnn_identity import _apply_cnn_identity_cost

    cost = 50.0
    adjusted = _apply_cnn_identity_cost(
        cost=cost,
        det_class=None,
        track_identity="tag_3",
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0)


def test_hungarian_cnn_no_adjustment_when_track_identity_none():
    """No cost adjustment when track identity is None (unassigned)."""
    from multi_tracker.core.identity.cnn_identity import _apply_cnn_identity_cost

    cost = 50.0
    adjusted = _apply_cnn_identity_cost(
        cost=cost,
        det_class="tag_3",
        track_identity=None,
        match_bonus=20.0,
        mismatch_penalty=50.0,
    )
    assert adjusted == pytest.approx(50.0)
```

### Step 2.2 — Run tests to verify they fail

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_mat_cnn_identity.py -k "cnn_history or cnn_match or cnn_mismatch or cnn_no_adj" -v 2>&1 | head -20
```
Expected: `ImportError` — `TrackCNNHistory` and `_apply_cnn_identity_cost` not yet defined.

### Step 2.3 — Add `TrackCNNHistory` and `_apply_cnn_identity_cost` to `cnn_identity.py`

- [ ] Append to `src/multi_tracker/core/identity/cnn_identity.py`:

```python
# ---------------------------------------------------------------------------
# TrackCNNHistory
# ---------------------------------------------------------------------------

from collections import Counter as _Counter


class TrackCNNHistory:
    """Maintains a sliding-window CNN identity history for every track slot.

    Follows the same pattern as :class:`TrackTagHistory` in tag_features.py.
    The majority class across the window is the track's assigned identity.
    Returns ``None`` when no majority exists (empty window or exact tie).
    """

    def __init__(self, n_tracks: int, window: int = 10) -> None:
        self._window = max(1, window)
        # _history[track_idx] is a list of (frame_idx, class_name) pairs
        self._history: list[list[tuple[int, str]]] = [[] for _ in range(n_tracks)]

    @property
    def n_tracks(self) -> int:
        return len(self._history)

    def resize(self, n_tracks: int) -> None:
        """Grow (never shrink) the history to accommodate *n_tracks*."""
        while len(self._history) < n_tracks:
            self._history.append([])

    def record(self, track_idx: int, frame_idx: int, class_name: str) -> None:
        """Record a confident CNN prediction for *track_idx* on *frame_idx*."""
        if track_idx >= len(self._history):
            self.resize(track_idx + 1)
        self._history[track_idx].append((frame_idx, class_name))
        # Trim entries older than window
        cutoff = frame_idx - self._window
        hist = self._history[track_idx]
        while hist and hist[0][0] < cutoff:
            hist.pop(0)

    def majority_class(self, track_idx: int) -> str | None:
        """Return majority-vote class for *track_idx*, or None if no clear winner."""
        if track_idx >= len(self._history):
            return None
        hist = self._history[track_idx]
        if not hist:
            return None
        counts = _Counter(cls for _, cls in hist)
        if len(counts) == 0:
            return None
        top_two = counts.most_common(2)
        if len(top_two) == 2 and top_two[0][1] == top_two[1][1]:
            return None  # exact tie → no majority
        return str(top_two[0][0])

    def clear_track(self, track_idx: int) -> None:
        """Clear history for a track slot (e.g., after identity loss)."""
        if track_idx < len(self._history):
            self._history[track_idx].clear()

    def build_track_identity_list(self, n_tracks: int) -> list[str | None]:
        """Return list of length *n_tracks*: majority class per slot (or None).

        This is what goes into ``association_data["track_cnn_identities"]``.
        """
        self.resize(n_tracks)
        return [self.majority_class(i) for i in range(n_tracks)]


# ---------------------------------------------------------------------------
# Hungarian cost helper
# ---------------------------------------------------------------------------

def _apply_cnn_identity_cost(
    cost: float,
    det_class: str | None,
    track_identity: str | None,
    match_bonus: float,
    mismatch_penalty: float,
) -> float:
    """Apply CNN identity match bonus / mismatch penalty to a cost value.

    Returns cost unchanged when either side is uncertain (None).
    """
    if det_class is None or track_identity is None:
        return cost
    if det_class == track_identity:
        return cost - match_bonus
    return cost + mismatch_penalty
```

### Step 2.4 — Add CNN identity cost block to `hungarian.py`

- [ ] Open `src/multi_tracker/core/assigners/hungarian.py`. In `_compute_advanced_cost_matrix`, the inner cost loop ends at line 398 with:

```python
                cost[track_idx, det_idx] = motion_core_cost
```

Immediately before that line (after the AprilTag bonus/penalty block that ends at line 398), add:

```python
                # --- CNN Classifier identity bonus / penalty ---
                _det_cnn_classes = association_data.get("detection_cnn_classes", [])
                _track_cnn_identities = association_data.get("track_cnn_identities", [])
                _cnn_match_bonus = float(p.get("CNN_CLASSIFIER_MATCH_BONUS", 20.0))
                _cnn_mismatch_penalty = float(p.get("CNN_CLASSIFIER_MISMATCH_PENALTY", 50.0))
                _det_cls = _det_cnn_classes[det_idx] if det_idx < len(_det_cnn_classes) else None
                _track_cls = (
                    _track_cnn_identities[track_idx]
                    if track_idx < len(_track_cnn_identities)
                    else None
                )
                if _det_cls is not None and _track_cls is not None:
                    if _det_cls == _track_cls:
                        motion_core_cost -= _cnn_match_bonus
                    else:
                        motion_core_cost += _cnn_mismatch_penalty
```

Note: the `_det_cnn_classes` and `_track_cnn_identities` lookups read outside the candidate loop. Move them before the `for track_idx, det_indices in candidates.items():` loop (line 326) to avoid repeated dict lookups — mirror the AprilTag pattern which reads `_det_tag_ids` and `_track_tag_ids` once before the loop (lines 319–321 area).

The correct placement is: move the `_det_cnn_classes`, `_track_cnn_identities`, `_cnn_match_bonus`, `_cnn_mismatch_penalty` assignments to just after the AprilTag equivalents (line 322 area), and reference them in the inner loop body:

```python
        # --- AprilTag identity cost config --- (existing, line ~319)
        _det_tag_ids = association_data.get("detection_tag_ids", [])
        _track_tag_ids = association_data.get("track_last_tag_ids", [])
        _tag_match_bonus = float(p.get("TAG_MATCH_BONUS", 20.0))
        _tag_mismatch_penalty = float(p.get("TAG_MISMATCH_PENALTY", 50.0))
        _NO_TAG = -1
        # --- CNN Classifier identity cost config --- (new)
        _det_cnn_classes = association_data.get("detection_cnn_classes", [])
        _track_cnn_identities = association_data.get("track_cnn_identities", [])
        _cnn_match_bonus = float(p.get("CNN_CLASSIFIER_MATCH_BONUS", 20.0))
        _cnn_mismatch_penalty = float(p.get("CNN_CLASSIFIER_MISMATCH_PENALTY", 50.0))
```

And in the inner loop (after the AprilTag block):
```python
                # --- CNN identity bonus / penalty ---
                _det_cls = _det_cnn_classes[det_idx] if det_idx < len(_det_cnn_classes) else None
                _track_cls = (
                    _track_cnn_identities[track_idx]
                    if track_idx < len(_track_cnn_identities)
                    else None
                )
                if _det_cls is not None and _track_cls is not None:
                    if _det_cls == _track_cls:
                        motion_core_cost -= _cnn_match_bonus
                    else:
                        motion_core_cost += _cnn_mismatch_penalty
```

### Step 2.5 — Run tests

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_mat_cnn_identity.py -v 2>&1 | tail -20
```
Expected: all tests PASS.

### Step 2.6 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/core/identity/cnn_identity.py src/multi_tracker/core/assigners/hungarian.py tests/test_mat_cnn_identity.py
git commit -m "$(cat <<'EOF'
feat(mat): add TrackCNNHistory, cost helpers, CNN identity adjustment in hungarian.py

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3 — CNN identity precompute phase in `worker.py`

**Files:**
- Modify: `src/multi_tracker/core/tracking/worker.py`

No unit tests for the worker precompute phase (consistent with the AprilTag precompute pattern — no unit tests exist for `_run_apriltag_precompute`). Verification via import + grep for the new methods.

### Step 3.1 — Add precompute helpers to `worker.py`

- [ ] Open `src/multi_tracker/core/tracking/worker.py`. After the `_should_precompute_apriltag_data` method (line 647) and before `_build_tag_cache_path`, add three new methods:

```python
    # ------------------------------------------------------------------
    # CNN Identity precompute
    # ------------------------------------------------------------------

    def _should_precompute_cnn_identity_data(self, params, detection_method):
        """Return True when the CNN identity precompute phase should run."""
        identity_method = str(params.get("IDENTITY_METHOD", "none_disabled")).lower()
        return bool(
            not self.backward_mode
            and not self.preview_mode
            and detection_method == "yolo_obb"
            and identity_method == "cnn_classifier"
        )

    def _build_cnn_identity_cache_path(self, start_frame, end_frame):
        """Derive the CNN identity cache path from the detection cache path."""
        if not self.detection_cache_path:
            return None
        from pathlib import Path as _P

        base = _P(self.detection_cache_path)
        return str(base.with_name(
            base.stem + f"_cnn_identity_{start_frame}_{end_frame}.npz"
        ))

    def _run_cnn_identity_precompute(
        self, detection_cache, params, cap, start_frame, end_frame
    ):
        """Run CNN identity classification on all frames and cache predictions.

        Follows the same lifecycle as _run_apriltag_precompute():
        - Reads OBB crops from detection_cache frame by frame
        - Calls CNNIdentityBackend.predict_batch() in batches
        - Writes predictions to CNNIdentityCache
        - Emits progress_signal per frame
        - Returns cache path or None on failure
        """
        from multi_tracker.core.identity.cnn_identity import (
            CNNIdentityBackend,
            CNNIdentityCache,
            CNNIdentityConfig,
        )
        from pathlib import Path as _P
        import os

        model_path = str(params.get("CNN_CLASSIFIER_MODEL_PATH", ""))
        if not model_path or not os.path.exists(model_path):
            logger.warning("CNN identity precompute: model_path not found: %s", model_path)
            return None

        cache_path = self._build_cnn_identity_cache_path(start_frame, end_frame)
        if cache_path is None:
            return None

        cfg = CNNIdentityConfig(
            model_path=model_path,
            confidence=float(params.get("CNN_CLASSIFIER_CONFIDENCE", 0.5)),
            batch_size=int(params.get("CNN_CLASSIFIER_BATCH_SIZE", 64)),
            crop_padding=float(params.get("CNN_CLASSIFIER_CROP_PADDING", 0.1)),
        )
        compute_runtime = str(params.get("COMPUTE_RUNTIME", "cpu"))
        backend = CNNIdentityBackend(cfg, model_path=model_path, compute_runtime=compute_runtime)
        cache = CNNIdentityCache(cache_path)

        total_frames = end_frame - start_frame
        self.progress_signal.emit(0, "CNN identity precompute: starting...")

        # Mirrors the AprilTag precompute loop in _run_apriltag_precompute (lines ~700-815)
        import cv2
        resize_f = float(params.get("RESIZE_FACTOR", 1.0))
        scale = 1.0 / resize_f if resize_f > 0 else 1.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_offset in range(total_frames):
            frame_idx = start_frame + frame_offset

            if self._stop_requested:
                logger.info("CNN identity precompute cancelled.")
                backend.close()
                return None

            # Read video frame
            ret, frame = cap.read()
            if not ret or frame is None:
                cache.save(frame_idx, [])
                continue

            # Get cached detections
            try:
                (_meas, _sizes, _shapes, _confs, obb_corners,
                 detection_ids, _hh, _dm) = detection_cache.get_frame(frame_idx)
            except Exception:
                cache.save(frame_idx, [])
                continue

            if not obb_corners:
                cache.save(frame_idx, [])
                continue

            # Scale OBB corners from detection resolution to original frame resolution
            scaled_corners = [
                np.asarray(c, dtype=np.float32) * scale for c in obb_corners
            ]
            det_idx_list = (
                list(range(len(obb_corners)))
                if not detection_ids
                else [int(d) for d in detection_ids]
            )

            # Extract crops using the same infrastructure as the AprilTag precompute
            crops = []
            crop_det_indices = []
            for _di, _corners in enumerate(scaled_corners):
                _crop, _offset = self._extract_expanded_obb_crop(
                    frame, _corners, cfg.crop_padding
                )
                if _crop is None:
                    continue
                crops.append(_crop)
                crop_det_indices.append(det_idx_list[_di])

            if crops:
                predictions = backend.predict_batch(crops)
                # Re-assign det_index to match the detection slot
                for pred, det_idx in zip(predictions, crop_det_indices):
                    pred.det_index = det_idx
            else:
                predictions = []

            cache.save(frame_idx, predictions)

            pct = int((frame_offset + 1) * 100 / max(total_frames, 1))
            if frame_offset % 50 == 0 or frame_offset == total_frames - 1:
                self.progress_signal.emit(
                    pct, f"CNN identity precompute: frame {frame_offset + 1}/{total_frames}"
                )

        backend.close()
        self.progress_signal.emit(100, "CNN identity precompute: complete.")
        logger.info("CNN identity cache written: %s", cache_path)
        return cache_path
```

### Step 3.2 — Wire precompute into the tracking loop

- [ ] In `worker.py`, find where the AprilTag precompute is triggered (around line 1651):

```python
        apriltag_precompute_enabled = self._should_precompute_apriltag_data(p, detection_method)
```

After the entire AprilTag precompute block (lines 1651–1671), add the CNN identity precompute:

```python
        # CNN Identity precompute — mirrors AprilTag precompute pattern (lines 1651-1671)
        cnn_identity_cache_path = None
        cnn_identity_precompute_enabled = self._should_precompute_cnn_identity_data(
            p, detection_method
        )
        if cnn_identity_precompute_enabled and detection_cache is not None and use_cached_detections:
            try:
                cnn_identity_cache_path = self._run_cnn_identity_precompute(
                    detection_cache, p, cap, start_frame, end_frame
                )
            except Exception as _cnn_pre_exc:
                logger.warning(
                    "CNN identity precompute failed (tracking continues without it): %s",
                    _cnn_pre_exc,
                )
                self.warning_signal.emit(
                    f"CNN identity precompute failed: {_cnn_pre_exc}"
                )
```

- [ ] After the tag observation cache loading block (line ~1685), open the CNN identity cache for reading:

```python
        # Open CNN identity cache for reading during tracking loop.
        cnn_identity_cache = None
        cnn_track_history = None
        if cnn_identity_cache_path and os.path.exists(cnn_identity_cache_path):
            from multi_tracker.core.identity.cnn_identity import CNNIdentityCache, TrackCNNHistory
            cnn_identity_cache = CNNIdentityCache(cnn_identity_cache_path)
            cnn_track_history = TrackCNNHistory(
                N, window=int(p.get("CNN_CLASSIFIER_WINDOW", 10))
            )
            logger.info("CNN identity cache loaded: %s", cnn_identity_cache_path)
```

- [ ] In the per-frame tracking loop, after the `track_tag_history.build_track_tag_id_list(N)` call (line ~2219), add CNN identity data to `association_data`:

```python
                # CNN identity data for assigner
                if cnn_identity_cache is not None and cnn_track_history is not None:
                    frame_cnn_preds = cnn_identity_cache.load(actual_frame_index)
                    _det_cnn_classes = [None] * len(detections_for_frame)
                    for pred in frame_cnn_preds:
                        if pred.det_index < len(_det_cnn_classes):
                            _det_cnn_classes[pred.det_index] = pred.class_name
                    association_data["detection_cnn_classes"] = _det_cnn_classes
                    association_data["track_cnn_identities"] = (
                        cnn_track_history.build_track_identity_list(N)
                    )
```

- [ ] After the assignment step (where tag history is updated, line ~2388), update the CNN history:

```python
                # Update CNN track history after assignment
                if cnn_track_history is not None and cnn_identity_cache is not None:
                    cnn_track_history.resize(N)
                    frame_cnn_preds = cnn_identity_cache.load(actual_frame_index)
                    _pred_by_det = {p.det_index: p for p in frame_cnn_preds}
                    for r, c in zip(rows, cols):
                        pred = _pred_by_det.get(c)
                        if pred is not None and pred.class_name is not None:
                            cnn_track_history.record(r, actual_frame_index, pred.class_name)
```

Note: The assignment result arrays in `worker.py` are `rows` and `cols` (confirmed at lines 2393–2394, e.g. `for r, c in zip(rows, cols):`). The code snippet uses these names correctly.

### Step 3.3 — Verify import

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.core.tracking.worker import TrackingWorker; print('OK')"
```
Expected: prints OK.

### Step 3.4 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/core/tracking/worker.py
git commit -m "$(cat <<'EOF'
feat(mat): add CNN identity precompute phase to tracking worker

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4 — Config keys + identity method rename in `main_window.py`

**Files:**
- Modify: `configs/default.json`
- Modify: `src/multi_tracker/gui/main_window.py`

### Step 4.1 — Add config keys to `default.json`

- [ ] Open `configs/default.json`. After the `color_tag_model_path` line (line 124), add the 8 new CNN classifier keys:

```json
  "color_tag_confidence": 0.5,
  "color_tag_model_path": "",
  "cnn_classifier_model_path": "",
  "cnn_classifier_confidence": 0.5,
  "cnn_classifier_label": "",
  "cnn_classifier_batch_size": 64,
  "cnn_classifier_crop_padding": 0.1,
  "cnn_classifier_match_bonus": 20.0,
  "cnn_classifier_mismatch_penalty": 50.0,
  "cnn_classifier_window": 10,
```

Note: `color_tag_model_path` and `color_tag_confidence` are kept as backward-compat aliases. The existing code reads them by uppercase key `COLOR_TAG_MODEL_PATH` — check whether `main_window.py` uppercases the keys before building the params dict `p`. Verify by grepping for `COLOR_TAG` vs `color_tag` in `main_window.py`.

### Step 4.2 — Rename identity method in `main_window.py`

- [ ] Open `src/multi_tracker/gui/main_window.py`. Find and update the identity method combo label:
  - Display text: `"Color Tags (YOLO)"` → `"CNN Classifier"` (in the `addItem` calls for the identity method combo)

- [ ] In `_apply_config_to_ui()` at line ~15055, find the `method_map` dict:

```python
method_map = {
    "none_disabled": 0,
    "color_tags_yolo": 1,
    "apriltags": 2,
    "custom": 0,
}
```

Update to include `"cnn_classifier"` at the same index (1 = CNN Classifier slot), and keep `"color_tags_yolo"` for backward compat:

```python
method_map = {
    "none_disabled": 0,
    "color_tags_yolo": 1,   # backward compat for old saved configs
    "cnn_classifier": 1,    # new key
    "apriltags": 2,
    "custom": 0,
}
```

- [ ] Add the backward compat identity method migration IMMEDIATELY BEFORE `method_map.get(identity_method, 0)` is called (i.e., after `identity_method = str(get_cfg(...)).lower()...` but before the lookup):

```python
# Backward compat: rename color_tags_yolo → cnn_classifier on load
if identity_method == "color_tags_yolo":
    identity_method = "cnn_classifier"
```

- [ ] In the params dict building path (where `COLOR_TAG_MODEL_PATH` is written, line ~14234), add both backward compat aliases so the new widgets populate correctly when loading an old config:

```python
# Backward compat: map old color_tag keys to new cnn_classifier keys
if "COLOR_TAG_MODEL_PATH" in p and "CNN_CLASSIFIER_MODEL_PATH" not in p:
    p["CNN_CLASSIFIER_MODEL_PATH"] = p["COLOR_TAG_MODEL_PATH"]
if "COLOR_TAG_CONFIDENCE" in p and "CNN_CLASSIFIER_CONFIDENCE" not in p:
    p["CNN_CLASSIFIER_CONFIDENCE"] = p["COLOR_TAG_CONFIDENCE"]
```

This prevents silent data loss when loading an old config that has `color_tag_confidence` saved but no `cnn_classifier_confidence` key.

### Step 4.3 — Show/hide CNN settings panel

- [ ] Find the existing logic that shows/hides the color tag settings panel when the identity method combo changes. This is likely in `_on_identity_method_changed()` or similar. Update it to show the CNN classifier panel when `"cnn_classifier"` is selected (replacing the `"color_tags_yolo"` condition).

### Step 4.4 — Verify import

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.gui.main_window import MainWindow; print('OK')"
```
Expected: prints OK.

### Step 4.5 — Commit

- [ ] Run:
```bash
git add configs/default.json src/multi_tracker/gui/main_window.py
git commit -m "$(cat <<'EOF'
feat(mat): add CNN_CLASSIFIER_* config keys and rename Color Tags to CNN Classifier

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5 — CNN settings panel UI + model import handler

**Files:**
- Modify: `src/multi_tracker/gui/main_window.py`

### Step 5.1 — Add CNN classifier settings panel widgets

- [ ] In `main_window.py`, find the existing color tag settings panel (around lines 6066–6085). Replace or expand the existing color tag controls with the full CNN classifier settings panel:

```python
# CNN Classifier identity settings panel
# Model combo (existing models from registry + sentinel)
self.combo_cnn_identity_model = QComboBox()
self.combo_cnn_identity_model.addItem("— select model —", "")
self.combo_cnn_identity_model.addItem("＋ Add New Model…", "__add_new__")
self.combo_cnn_identity_model.activated.connect(self._on_cnn_identity_model_selected)

# Verification block (read-only labels)
self.lbl_cnn_arch = QLabel("—")
self.lbl_cnn_num_classes = QLabel("—")
self.lbl_cnn_class_names = QLabel("—")
self.lbl_cnn_input_size = QLabel("—")
self.lbl_cnn_label = QLabel("—")

# Inference and cost settings
self.spin_cnn_confidence = QDoubleSpinBox()
self.spin_cnn_confidence.setRange(0.0, 1.0)
self.spin_cnn_confidence.setSingleStep(0.05)
self.spin_cnn_confidence.setValue(0.5)

self.spin_cnn_match_bonus = QDoubleSpinBox()
self.spin_cnn_match_bonus.setRange(0.0, 200.0)
self.spin_cnn_match_bonus.setSingleStep(5.0)
self.spin_cnn_match_bonus.setValue(20.0)

self.spin_cnn_mismatch_penalty = QDoubleSpinBox()
self.spin_cnn_mismatch_penalty.setRange(0.0, 200.0)
self.spin_cnn_mismatch_penalty.setSingleStep(5.0)
self.spin_cnn_mismatch_penalty.setValue(50.0)

self.spin_cnn_window = QSpinBox()
self.spin_cnn_window.setRange(1, 100)
self.spin_cnn_window.setValue(10)

self.spin_cnn_crop_padding = QDoubleSpinBox()
self.spin_cnn_crop_padding.setRange(0.0, 1.0)
self.spin_cnn_crop_padding.setSingleStep(0.05)
self.spin_cnn_crop_padding.setValue(0.1)
```

Layout: wrap all these in a `QGroupBox("CNN Classifier")` or similar container. The group box visibility is toggled in `_on_identity_method_changed()` (already updated in Task 4).

### Step 5.2 — Populate combo from model registry on startup

- [ ] Add `_refresh_cnn_identity_model_combo()` method. Note: `get_models_root_directory()` is a **module-level function** in `main_window.py` (line ~1626), not an instance attribute — use it directly (no `self.`). You can also use `get_yolo_model_registry_path()` (line ~1787) as a shortcut for `os.path.join(get_models_root_directory(), "model_registry.json")`.

```python
def _refresh_cnn_identity_model_combo(self) -> None:
    """Populate the CNN identity model combo from model_registry.json."""
    import json
    registry_path = os.path.join(get_models_root_directory(), "model_registry.json")
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    self.combo_cnn_identity_model.blockSignals(True)
    current_path = self.combo_cnn_identity_model.currentData()
    self.combo_cnn_identity_model.clear()
    self.combo_cnn_identity_model.addItem("— select model —", "")
    for rel_path, meta in registry.items():
        if meta.get("usage_role") != "cnn_identity":
            continue
        arch = meta.get("arch", "?")
        label = meta.get("classification_label", "")
        species = meta.get("species", "")
        n_cls = meta.get("num_classes", "?")
        display = f"{arch} | {n_cls} cls"
        if species:
            display += f" | {species}"
        if label:
            display += f" | {label}"
        self.combo_cnn_identity_model.addItem(display, rel_path)
    self.combo_cnn_identity_model.addItem("＋ Add New Model…", "__add_new__")

    # Restore selection if still present
    idx = self.combo_cnn_identity_model.findData(current_path)
    if idx >= 0:
        self.combo_cnn_identity_model.setCurrentIndex(idx)
    self.combo_cnn_identity_model.blockSignals(False)
```

Call `_refresh_cnn_identity_model_combo()` during startup (same place as YOLO model combos are refreshed).

### Step 5.3 — Add `_on_cnn_identity_model_selected()` slot

```python
def _on_cnn_identity_model_selected(self, index: int) -> None:
    """Handle combo activation — sentinel triggers import dialog."""
    rel_path = self.combo_cnn_identity_model.itemData(index)
    if rel_path == "__add_new__":
        self._handle_add_new_cnn_identity_model()
        return
    # Update verification block from registry
    self._update_cnn_identity_verification_panel(rel_path)
```

### Step 5.4 — Add `_update_cnn_identity_verification_panel()` helper

```python
def _update_cnn_identity_verification_panel(self, rel_path: str) -> None:
    """Populate the read-only verification labels from the registry entry."""
    import json
    registry_path = os.path.join(get_models_root_directory(), "model_registry.json")
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except Exception:
        return
    meta = registry.get(rel_path, {})
    self.lbl_cnn_arch.setText(str(meta.get("arch", "—")))
    self.lbl_cnn_num_classes.setText(str(meta.get("num_classes", "—")))
    class_names = meta.get("class_names", [])
    preview = ", ".join(class_names[:12])
    if len(class_names) > 12:
        preview += f", … ({len(class_names)} total)"
    self.lbl_cnn_class_names.setText(preview or "—")
    raw_size = meta.get("input_size", "—")
    self.lbl_cnn_input_size.setText(str(raw_size))
    self.lbl_cnn_label.setText(str(meta.get("classification_label", "—")))
```

### Step 5.5 — Add `_handle_add_new_cnn_identity_model()` import handler

```python
def _handle_add_new_cnn_identity_model(self) -> None:
    """Import a ClassKit-trained .pth or YOLO .pt model for CNN identity."""
    import json
    import shutil
    from datetime import datetime

    # Restore previous selection on cancel
    self.combo_cnn_identity_model.blockSignals(True)
    prev_idx = self.combo_cnn_identity_model.findData(
        self.combo_cnn_identity_model.currentData()
    )
    self.combo_cnn_identity_model.blockSignals(False)

    src_path, _ = QFileDialog.getOpenFileName(
        self,
        "Import ClassKit Model for CNN Identity",
        os.path.join(get_models_root_directory(), "classification", "identity"),
        "ClassKit Model Files (*.pth *.pt);;All Files (*)",
    )
    if not src_path:
        self.combo_cnn_identity_model.setCurrentIndex(max(prev_idx, 0))
        return

    # Read checkpoint metadata
    meta = {}
    try:
        if src_path.endswith(".pth"):
            import torch
            ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
            meta["arch"] = ckpt.get("arch", "tinyclassifier")
            meta["class_names"] = ckpt.get("class_names", [])
            meta["factor_names"] = ckpt.get("factor_names", [])
            raw_size = ckpt.get("input_size", (224, 224))
            meta["input_size"] = list(raw_size) if isinstance(raw_size, (list, tuple)) else [raw_size, raw_size]
            meta["num_classes"] = ckpt.get("num_classes", len(meta["class_names"]))
        else:  # .pt (YOLO)
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO(src_path)
            names = yolo.names
            meta["arch"] = "yolo"
            meta["class_names"] = [names[i] for i in sorted(names.keys())]
            meta["factor_names"] = []
            meta["input_size"] = [224, 224]
            meta["num_classes"] = len(meta["class_names"])
    except Exception as exc:
        QMessageBox.critical(self, "Import Error", f"Could not read checkpoint metadata:\n{exc}")
        self.combo_cnn_identity_model.setCurrentIndex(max(prev_idx, 0))
        return

    # Show metadata import dialog
    from multi_tracker.gui.dialogs import CNNIdentityImportDialog
    dlg = CNNIdentityImportDialog(meta, parent=self)
    if dlg.exec() != QDialog.Accepted:
        self.combo_cnn_identity_model.setCurrentIndex(max(prev_idx, 0))
        return

    species = dlg.species()
    classification_label = dlg.classification_label()

    # Copy to models/classification/identity/
    dest_dir = os.path.join(get_models_root_directory(), "classification", "identity")
    os.makedirs(dest_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ext = Path(src_path).suffix
    filename = f"{timestamp}_{meta['arch']}_{species}_{classification_label}{ext}"
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(src_path, dest_path)

    # Register in model_registry.json
    rel_path = os.path.relpath(dest_path, get_models_root_directory())
    registry_path = os.path.join(get_models_root_directory(), "model_registry.json")
    try:
        with open(registry_path) as f:
            registry = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        registry = {}

    registry[rel_path] = {
        "arch": meta["arch"],
        "num_classes": meta["num_classes"],
        "class_names": meta["class_names"],
        "factor_names": meta["factor_names"],
        "input_size": meta["input_size"],
        "species": species,
        "classification_label": classification_label,
        "added_at": datetime.now().isoformat(),
        "task_family": "classify",
        "usage_role": "cnn_identity",
    }
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Refresh combo and select the new model
    self._refresh_cnn_identity_model_combo()
    idx = self.combo_cnn_identity_model.findData(rel_path)
    if idx >= 0:
        self.combo_cnn_identity_model.setCurrentIndex(idx)
        self._update_cnn_identity_verification_panel(rel_path)
```

### Step 5.6 — Add `CNNIdentityImportDialog` to the `gui/dialogs/` package

Note: `src/multi_tracker/gui/dialogs/` is a **package directory**, not a single file. It contains `__init__.py`, `parameter_helper.py`, and `train_yolo_dialog.py`.

- [ ] Create `src/multi_tracker/gui/dialogs/cnn_identity_import_dialog.py` with the dialog class.
- [ ] Export it from `src/multi_tracker/gui/dialogs/__init__.py` by adding:
  ```python
  from .cnn_identity_import_dialog import CNNIdentityImportDialog
  ```

Dialog class to write in the new file:

```python
class CNNIdentityImportDialog(QDialog):
    """Pre-filled dialog for user to verify and annotate CNN identity model import."""

    def __init__(self, meta: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import CNN Identity Model")
        layout = QFormLayout(self)

        # Read-only metadata display
        layout.addRow("Architecture:", QLabel(str(meta.get("arch", "—"))))
        layout.addRow("Num classes:", QLabel(str(meta.get("num_classes", "—"))))
        class_preview = ", ".join(meta.get("class_names", [])[:8])
        if len(meta.get("class_names", [])) > 8:
            class_preview += f" … ({len(meta['class_names'])} total)"
        layout.addRow("Classes:", QLabel(class_preview or "—"))
        layout.addRow("Input size:", QLabel(str(meta.get("input_size", "—"))))

        # Editable fields
        self._species_edit = QLineEdit()
        self._species_edit.setPlaceholderText("e.g. ant")
        layout.addRow("Species:", self._species_edit)

        self._label_edit = QLineEdit()
        self._label_edit.setPlaceholderText("e.g. apriltag, colortag (optional)")
        layout.addRow("Classification label:", self._label_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def species(self) -> str:
        return self._species_edit.text().strip() or "unknown"

    def classification_label(self) -> str:
        return self._label_edit.text().strip()
```

### Step 5.7 — Wire config read/write for the new controls

- [ ] In `main_window.py`, find where the config dict is built for tracking (the large `params = {...}` dict). Add the CNN classifier keys, reading from the new spinboxes and combo:

```python
"CNN_CLASSIFIER_MODEL_PATH": os.path.join(
    get_models_root_directory(), self.combo_cnn_identity_model.currentData() or ""
) if self.combo_cnn_identity_model.currentData() not in (None, "", "__add_new__") else "",
"CNN_CLASSIFIER_CONFIDENCE": self.spin_cnn_confidence.value(),
"CNN_CLASSIFIER_MATCH_BONUS": self.spin_cnn_match_bonus.value(),
"CNN_CLASSIFIER_MISMATCH_PENALTY": self.spin_cnn_mismatch_penalty.value(),
"CNN_CLASSIFIER_WINDOW": self.spin_cnn_window.value(),
"CNN_CLASSIFIER_BATCH_SIZE": 64,  # not user-configurable via UI; internal default
"CNN_CLASSIFIER_CROP_PADDING": self.spin_cnn_crop_padding.value(),
"CNN_CLASSIFIER_LABEL": "",  # cosmetic only; read from registry entry if needed
```

- [ ] In the config-to-UI loading path (`_apply_config_to_ui()` or equivalent), populate the CNN controls from the saved config.

### Step 5.8 — Verify import

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.gui.main_window import MainWindow; print('OK')"
conda run -n multi-animal-tracker-mps python -c "from multi_tracker.gui.dialogs import CNNIdentityImportDialog; print('OK')"
```
Expected: both print OK.

### Step 5.9 — Run full test suite

- [ ] Run:
```bash
conda run -n multi-animal-tracker-mps python -m pytest tests/test_mat_cnn_identity.py -v 2>&1 | tail -15
```
Expected: all tests PASS.

### Step 5.10 — Commit

- [ ] Run:
```bash
git add src/multi_tracker/gui/main_window.py \
        src/multi_tracker/gui/dialogs/cnn_identity_import_dialog.py \
        src/multi_tracker/gui/dialogs/__init__.py
git commit -m "$(cat <<'EOF'
feat(mat): add CNN Classifier settings panel, import handler, registry integration

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
