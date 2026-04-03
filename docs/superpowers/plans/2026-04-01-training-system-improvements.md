# Training System Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the MAT Training Center from "functional" to "production-quality" with augmentation controls, post-training model testing, resume-from-checkpoint, run history browsing, live metrics, reproducible config export, auto-batch tuning, background training, and stratified dataset splitting.

**Architecture:** Nine independent features, each adding to either the training dialog UI (`src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`), the training backend (`src/hydra_suite/training/`), or both. Features are ordered so each builds cleanly on the prior state. All UI work is PySide6/Qt — no matplotlib.

**Tech Stack:** Python 3.10+, PySide6, ultralytics CLI, PyTorch, numpy, cv2

---

## File Map

| Feature | Files Created | Files Modified |
|---------|--------------|----------------|
| 1. Augmentation controls | `tests/test_training_augmentation.py` | `train_yolo_dialog.py`, `runner.py` |
| 2. Quick test | `src/hydra_suite/mat/gui/dialogs/model_test_dialog.py`, `tests/test_model_test_dialog.py` | `train_yolo_dialog.py` |
| 3. Resume from checkpoint | `tests/test_training_resume.py` | `train_yolo_dialog.py`, `runner.py`, `contracts.py` |
| 4. Run history viewer | `src/hydra_suite/mat/gui/dialogs/run_history_dialog.py`, `tests/test_run_history.py` | `train_yolo_dialog.py` |
| 5. Live loss plot | `src/hydra_suite/mat/gui/widgets/loss_plot_widget.py`, `tests/test_loss_plot_widget.py` | `train_yolo_dialog.py` |
| 6. Export training config | `tests/test_training_config_export.py` | `train_yolo_dialog.py` |
| 7. Auto-batch and multi-GPU | `tests/test_training_batch_gpu.py` | `train_yolo_dialog.py`, `runner.py` |
| 8. Background training | `tests/test_training_detach.py` | `train_yolo_dialog.py`, `runner.py` |
| 9. Stratified splitting | `tests/test_stratified_split.py` | `dataset_builders.py`, `dataset_inspector.py` |

All paths relative to repo root. `train_yolo_dialog.py` = `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`.

---

## Task 1: Augmentation Controls for OBB Roles

**Why:** The `AugmentationProfile` dataclass already exists in `contracts.py` with `flipud`, `fliplr`, `rotate`, `brightness`, `contrast`, and an `args` dict for ultralytics-native params. But the training dialog creates `TrainingRunSpec` without setting any augmentation profile, so defaults silently apply (`fliplr=0.5`, rest at 0). Users cannot control augmentation or even see what is active. For animals with left/right asymmetry, `fliplr=0.5` can be harmful.

**Files:**
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`
- Modify: `src/hydra_suite/training/runner.py`
- Test: `tests/test_training_augmentation.py`

### Step-by-step

- [ ] **Step 1: Write failing test for augmentation args in ultralytics command**

```python
# tests/test_training_augmentation.py
from __future__ import annotations

from hydra_suite.training.contracts import (
    AugmentationProfile,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_augmentation_args_passed_to_command():
    """Augmentation args dict entries appear as CLI flags."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=True,
            args={"flipud": 0.3, "mosaic": 0.0, "mixup": 0.1},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud=0.3" in cmd_str
    assert "mosaic=0.0" in cmd_str
    assert "mixup=0.1" in cmd_str


def test_augmentation_disabled_skips_args():
    """When enabled=False, no augmentation args are emitted."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        augmentation_profile=AugmentationProfile(
            enabled=False,
            args={"flipud": 0.3},
        ),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "flipud" not in cmd_str
```

- [ ] **Step 2: Run tests to confirm they pass** (the runner already supports `augmentation_profile.args` at `runner.py:119-121`)

Run: `python -m pytest tests/test_training_augmentation.py -v`
Expected: PASS (both tests -- the backend already works, we just need UI)

- [ ] **Step 3: Add augmentation group to training dialog**

In `train_yolo_dialog.py`, add a new method `_build_augmentation_group` and call it from `_build_ui` between config and run groups. Add these widgets:

```python
def _build_augmentation_group(self):
    gb = QGroupBox("Augmentation (Ultralytics)")
    gb.setCheckable(True)
    gb.setChecked(True)
    v = QVBoxLayout(gb)

    form = QFormLayout()

    self.spin_aug_fliplr = QDoubleSpinBox()
    self.spin_aug_fliplr.setRange(0.0, 1.0)
    self.spin_aug_fliplr.setSingleStep(0.05)
    self.spin_aug_fliplr.setValue(0.5)
    self.spin_aug_fliplr.setToolTip(
        "Horizontal flip probability. Set 0.0 for animals with "
        "left/right asymmetry (e.g. color markings on one side)."
    )
    form.addRow("fliplr", self.spin_aug_fliplr)

    self.spin_aug_flipud = QDoubleSpinBox()
    self.spin_aug_flipud.setRange(0.0, 1.0)
    self.spin_aug_flipud.setSingleStep(0.05)
    self.spin_aug_flipud.setValue(0.0)
    form.addRow("flipud", self.spin_aug_flipud)

    self.spin_aug_degrees = QDoubleSpinBox()
    self.spin_aug_degrees.setRange(0.0, 180.0)
    self.spin_aug_degrees.setSingleStep(5.0)
    self.spin_aug_degrees.setValue(0.0)
    self.spin_aug_degrees.setToolTip("Random rotation range in degrees.")
    form.addRow("degrees", self.spin_aug_degrees)

    self.spin_aug_mosaic = QDoubleSpinBox()
    self.spin_aug_mosaic.setRange(0.0, 1.0)
    self.spin_aug_mosaic.setSingleStep(0.1)
    self.spin_aug_mosaic.setValue(1.0)
    self.spin_aug_mosaic.setToolTip(
        "Mosaic augmentation probability. Default 1.0 (Ultralytics default)."
    )
    form.addRow("mosaic", self.spin_aug_mosaic)

    self.spin_aug_mixup = QDoubleSpinBox()
    self.spin_aug_mixup.setRange(0.0, 1.0)
    self.spin_aug_mixup.setSingleStep(0.05)
    self.spin_aug_mixup.setValue(0.0)
    form.addRow("mixup", self.spin_aug_mixup)

    self.spin_aug_hsv_h = QDoubleSpinBox()
    self.spin_aug_hsv_h.setRange(0.0, 1.0)
    self.spin_aug_hsv_h.setSingleStep(0.005)
    self.spin_aug_hsv_h.setDecimals(3)
    self.spin_aug_hsv_h.setValue(0.015)
    form.addRow("hsv_h", self.spin_aug_hsv_h)

    self.spin_aug_hsv_s = QDoubleSpinBox()
    self.spin_aug_hsv_s.setRange(0.0, 1.0)
    self.spin_aug_hsv_s.setSingleStep(0.05)
    self.spin_aug_hsv_s.setValue(0.7)
    form.addRow("hsv_s", self.spin_aug_hsv_s)

    self.spin_aug_hsv_v = QDoubleSpinBox()
    self.spin_aug_hsv_v.setRange(0.0, 1.0)
    self.spin_aug_hsv_v.setSingleStep(0.05)
    self.spin_aug_hsv_v.setValue(0.4)
    form.addRow("hsv_v", self.spin_aug_hsv_v)

    v.addLayout(form)

    note = QLabel(
        "<i>These are passed directly to Ultralytics. "
        "Defaults match Ultralytics v8 defaults. "
        "Set fliplr=0 for asymmetric animals.</i>"
    )
    note.setWordWrap(True)
    v.addWidget(note)

    self.aug_group = gb
    return gb
```

- [ ] **Step 4: Wire augmentation group into `_build_ui`**

In `_build_ui`, after the `_build_config_group()` call, add:

```python
layout.addWidget(self._build_augmentation_group())
```

- [ ] **Step 5: Build `AugmentationProfile` from UI in `_start_training`**

In `_start_training`, where `TrainingRunSpec` is created, add augmentation profile construction:

```python
aug_args = {}
if self.aug_group.isChecked():
    aug_args = {
        "fliplr": self.spin_aug_fliplr.value(),
        "flipud": self.spin_aug_flipud.value(),
        "degrees": self.spin_aug_degrees.value(),
        "mosaic": self.spin_aug_mosaic.value(),
        "mixup": self.spin_aug_mixup.value(),
        "hsv_h": self.spin_aug_hsv_h.value(),
        "hsv_s": self.spin_aug_hsv_s.value(),
        "hsv_v": self.spin_aug_hsv_v.value(),
    }

# Inside the TrainingRunSpec constructor, add:
augmentation_profile=AugmentationProfile(
    enabled=self.aug_group.isChecked(),
    args=aug_args,
),
```

Also add `AugmentationProfile` to the imports from `hydra_suite.training`.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_training_augmentation.py tests/test_training_framework.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add tests/test_training_augmentation.py src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py
git commit -m "feat(training): add augmentation controls to Training Center dialog"
```

---

## Task 2: Quick Test -- Run Model on Sample Images

**Why:** After training completes, users have no way to visually verify the model before deploying it. They must close the dialog, switch model paths, and start a tracking run. A "Quick Test" button that runs the trained model on a few dataset images and shows annotated results catches bad models immediately.

**Files:**
- Create: `src/hydra_suite/mat/gui/dialogs/model_test_dialog.py`
- Create: `tests/test_model_test_dialog.py`
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`

### Step-by-step

- [ ] **Step 1: Write test for model test dialog data flow**

```python
# tests/test_model_test_dialog.py
"""Tests for ModelTestDialog parameter construction."""
from __future__ import annotations

from hydra_suite.tracker.gui.dialogs.model_test_dialog import build_test_params


def test_build_test_params_direct_mode():
    params = build_test_params(
        model_path="/models/best.pt",
        role="obb_direct",
        device="cpu",
        imgsz=640,
    )
    assert params["YOLO_OBB_MODE"] == "direct"
    assert params["YOLO_OBB_DIRECT_MODEL_PATH"] == "/models/best.pt"
    assert params["YOLO_MODEL_PATH"] == "/models/best.pt"
    assert params["YOLO_DEVICE"] == "cpu"


def test_build_test_params_seq_crop_obb():
    params = build_test_params(
        model_path="/models/crop.pt",
        role="seq_crop_obb",
        device="cpu",
        imgsz=160,
        crop_pad_ratio=0.15,
        min_crop_size_px=64,
        enforce_square=True,
    )
    assert params["YOLO_OBB_MODE"] == "sequential"
    assert params["YOLO_CROP_OBB_MODEL_PATH"] == "/models/crop.pt"
    assert params["YOLO_SEQ_STAGE2_IMGSZ"] == 160
    assert params["YOLO_SEQ_CROP_PAD_RATIO"] == 0.15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model_test_dialog.py -v`
Expected: FAIL -- `model_test_dialog` module does not exist yet

- [ ] **Step 3: Create model_test_dialog.py with parameter builder**

```python
# src/hydra_suite/mat/gui/dialogs/model_test_dialog.py
"""Quick model test dialog -- run a trained model on sample images."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def build_test_params(
    model_path: str,
    role: str,
    device: str = "cpu",
    imgsz: int = 640,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
    detect_model_path: str = "",
) -> dict:
    """Build a YOLO detector params dict for quick testing a trained model."""
    is_sequential = role in ("seq_detect", "seq_crop_obb")
    params = {
        "DETECTION_METHOD": 1,
        "YOLO_OBB_MODE": "sequential" if is_sequential else "direct",
        "YOLO_OBB_DIRECT_MODEL_PATH": model_path if not is_sequential else "",
        "YOLO_MODEL_PATH": model_path,
        "YOLO_DETECT_MODEL_PATH": detect_model_path if is_sequential else "",
        "YOLO_CROP_OBB_MODEL_PATH": model_path if role == "seq_crop_obb" else "",
        "YOLO_HEADTAIL_MODEL_PATH": "",
        "YOLO_DEVICE": device,
        "YOLO_CONFIDENCE_THRESHOLD": 0.25,
        "YOLO_IOU_THRESHOLD": 0.45,
        "RAW_YOLO_CONFIDENCE_FLOOR": 0.01,
        "YOLO_TARGET_CLASSES": None,
        "MAX_TARGETS": 100,
        "ENABLE_TENSORRT": False,
        "ENABLE_ONNX_RUNTIME": False,
        "YOLO_SEQ_DETECT_CONF_THRESHOLD": 0.25,
        "YOLO_SEQ_CROP_PAD_RATIO": crop_pad_ratio,
        "YOLO_SEQ_MIN_CROP_SIZE_PX": min_crop_size_px,
        "YOLO_SEQ_ENFORCE_SQUARE_CROP": enforce_square,
        "YOLO_SEQ_STAGE2_IMGSZ": imgsz if role == "seq_crop_obb" else 0,
    }
    return params


class _TestWorker(QThread):
    """Run inference on sample images in a background thread."""

    result_signal = Signal(list)  # list of (annotated_bgr, n_detections)
    error_signal = Signal(str)

    def __init__(self, params: dict, image_paths: list[str]):
        super().__init__()
        self.params = params
        self.image_paths = image_paths

    def run(self):
        try:
            from hydra_suite.core.detectors import YOLOOBBDetector

            detector = YOLOOBBDetector(dict(self.params))
            results = []
            for i, img_path in enumerate(self.image_paths):
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                raw = detector.detect_objects(frame, frame_count=i, return_raw=True)
                # raw = (meas, sizes, shapes, yolo_results, confs, corners, ...)
                meas = raw[0] if raw else []
                corners = raw[5] if len(raw) > 5 else []
                annotated = frame.copy()
                for corner_set in corners:
                    pts = np.array(corner_set, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
                results.append((annotated, len(meas)))
            self.result_signal.emit(results)
        except Exception as exc:
            self.error_signal.emit(str(exc))


class ModelTestDialog(QDialog):
    """Show quick inference results from a trained model on sample images."""

    def __init__(
        self,
        parent=None,
        model_path: str = "",
        role: str = "obb_direct",
        dataset_dir: str = "",
        device: str = "cpu",
        imgsz: int = 640,
        crop_pad_ratio: float = 0.15,
        min_crop_size_px: int = 64,
        enforce_square: bool = True,
        detect_model_path: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Quick Test -- " + Path(model_path).name)
        self.resize(900, 500)
        self.model_path = model_path
        self.role = role
        self.dataset_dir = dataset_dir
        self.device = device
        self.imgsz = imgsz
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_size_px = min_crop_size_px
        self.enforce_square = enforce_square
        self.detect_model_path = detect_model_path
        self.worker = None

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Running inference on sample images...")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        layout.addWidget(self.progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_layout = QHBoxLayout(self.results_widget)
        scroll.setWidget(self.results_widget)
        layout.addWidget(scroll)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)

        self._run_test()

    def _collect_sample_images(self, n: int = 6) -> list[str]:
        """Pick sample images from the dataset directory."""
        ds = Path(self.dataset_dir)
        candidates = []
        for split_dir in (ds / "images" / "val", ds / "images" / "train"):
            if split_dir.exists():
                for f in split_dir.rglob("*"):
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                        candidates.append(str(f))
        if not candidates:
            # Fallback: any images under dataset dir.
            for f in ds.rglob("*"):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    candidates.append(str(f))
        rng = random.Random(42)
        rng.shuffle(candidates)
        return candidates[:n]

    def _run_test(self):
        images = self._collect_sample_images()
        if not images:
            self.status_label.setText("No sample images found in dataset directory.")
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
            return

        params = build_test_params(
            model_path=self.model_path,
            role=self.role,
            device=self.device,
            imgsz=self.imgsz,
            crop_pad_ratio=self.crop_pad_ratio,
            min_crop_size_px=self.min_crop_size_px,
            enforce_square=self.enforce_square,
            detect_model_path=self.detect_model_path,
        )
        self.worker = _TestWorker(params, images)
        self.worker.result_signal.connect(self._on_results)
        self.worker.error_signal.connect(self._on_error)
        self.worker.start()

    def _on_results(self, results: list):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        total_det = sum(n for _, n in results)
        self.status_label.setText(
            "Done -- "
            + str(len(results))
            + " images, "
            + str(total_det)
            + " total detections. Green polygons = OBB detections."
        )
        display_h = 300
        for annotated_bgr, n_det in results:
            rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaledToHeight(
                display_h, Qt.SmoothTransformation
            )
            lbl = QLabel()
            lbl.setPixmap(pixmap)
            lbl.setToolTip(str(n_det) + " detections")
            lbl.setStyleSheet("border: 1px solid #555; margin: 2px;")
            self.results_layout.addWidget(lbl)

    def _on_error(self, msg: str):
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.status_label.setText("Error: " + msg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model_test_dialog.py -v`
Expected: PASS

- [ ] **Step 5: Add "Quick Test" button to training dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add after `self.btn_stop`:
```python
self.btn_quick_test = QPushButton("Quick Test...")
self.btn_quick_test.setEnabled(False)
self.btn_quick_test.setToolTip(
    "Run the last trained model on sample images to visually verify detections."
)
row.addWidget(self.btn_quick_test)
self.btn_quick_test.clicked.connect(self._quick_test)
```

Add instance variable in `__init__`:
```python
self._last_training_results: list[dict] = []
```

In `_on_training_done`, store results and enable button:
```python
self._last_training_results = results
succeeded = [r for r in results if r.get("success")]
self.btn_quick_test.setEnabled(bool(succeeded))
```

Add the handler:
```python
def _quick_test(self):
    from hydra_suite.tracker.gui.dialogs.model_test_dialog import ModelTestDialog

    succeeded = [r for r in self._last_training_results if r.get("success")]
    if not succeeded:
        QMessageBox.warning(self, "No Model", "No successful training run to test.")
        return
    # Test the first successful role (usually obb_direct).
    result = succeeded[0]
    role = str(result.get("role", "obb_direct"))
    model_path = str(result.get("artifact_path") or result.get("published_model_path", ""))
    if not model_path:
        QMessageBox.warning(self, "No Artifact", "No model artifact found.")
        return
    ds_dir = self.role_dataset_dirs.get(role, "")
    detect_model_path = ""
    if role == "seq_crop_obb":
        # Need the detect model too for sequential mode.
        for r2 in self._last_training_results:
            if r2.get("role") == "seq_detect" and r2.get("success"):
                detect_model_path = str(
                    r2.get("artifact_path") or r2.get("published_model_path", "")
                )
                break

    dlg = ModelTestDialog(
        parent=self,
        model_path=model_path,
        role=role,
        dataset_dir=ds_dir,
        device=self.combo_device.currentText().strip() or "cpu",
        imgsz=self._imgsz_for_role(
            TrainingRole(role) if role in [r.value for r in TrainingRole] else TrainingRole.OBB_DIRECT
        ),
        crop_pad_ratio=self.spin_crop_pad.value(),
        min_crop_size_px=self.spin_crop_min_px.value(),
        enforce_square=self.chk_crop_square.isChecked(),
        detect_model_path=detect_model_path,
    )
    dlg.exec()
```

- [ ] **Step 6: Run all training tests**

Run: `python -m pytest tests/test_model_test_dialog.py tests/test_training_framework.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/mat/gui/dialogs/model_test_dialog.py tests/test_model_test_dialog.py src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py
git commit -m "feat(training): add Quick Test dialog for post-training model verification"
```

---

## Task 3: Resume from Checkpoint

**Why:** If training crashes, early-stops too aggressively, or a user wants to continue training for more epochs, they currently have to start from scratch. Ultralytics supports `resume=True` which loads `last.pt` and continues.

**Files:**
- Modify: `src/hydra_suite/training/runner.py`
- Modify: `src/hydra_suite/training/contracts.py`
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`
- Test: `tests/test_training_resume.py`

### Step-by-step

- [ ] **Step 1: Write failing test**

```python
# tests/test_training_resume.py
from __future__ import annotations

from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_resume_flag_in_command():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="/tmp/runs/last.pt",
        hyperparams=TrainingHyperParams(epochs=200, imgsz=640),
        resume_from="/tmp/runs/last.pt",
    )
    cmd = build_ultralytics_command(spec, "/tmp/run2")
    cmd_str = " ".join(cmd)
    assert "resume=True" in cmd_str
    # When resuming, model should point to last.pt, epochs to new target.
    assert "model=/tmp/runs/last.pt" in cmd_str


def test_no_resume_flag_by_default():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=100, imgsz=640),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "resume" not in cmd_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_resume.py -v`
Expected: FAIL -- `resume_from` field does not exist on `TrainingRunSpec`

- [ ] **Step 3: Add `resume_from` field to contracts**

In `src/hydra_suite/training/contracts.py`, add to `TrainingRunSpec`:
```python
resume_from: str = ""  # Path to last.pt checkpoint to resume from
```

- [ ] **Step 4: Handle `resume_from` in runner**

In `src/hydra_suite/training/runner.py`, in `build_ultralytics_command`, after the `cache` check:
```python
if spec.resume_from:
    args.append("resume=True")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_training_resume.py -v`
Expected: PASS

- [ ] **Step 6: Add "Resume Last Run" button to dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add after `self.btn_stop`:
```python
self.btn_resume = QPushButton("Resume Last Run")
self.btn_resume.setEnabled(False)
self.btn_resume.setToolTip(
    "Resume training from the last.pt checkpoint of the most recent run."
)
row.addWidget(self.btn_resume)
self.btn_resume.clicked.connect(self._resume_training)
```

Store last run dir in `_on_training_done`:
```python
for r in results:
    run_dir = ""
    artifact = r.get("artifact_path", "")
    if artifact:
        _wdir = Path(artifact).parent
        if _wdir.name == "weights":
            run_dir = str(_wdir.parent)
    r["_run_dir"] = run_dir
self.btn_resume.setEnabled(any(
    r.get("_run_dir") and Path(r["_run_dir"]).joinpath("weights", "last.pt").exists()
    for r in results
))
```

Add resume handler:
```python
def _resume_training(self):
    """Resume training from the last checkpoint."""
    for r in reversed(self._last_training_results):
        run_dir = r.get("_run_dir", "")
        last_pt = Path(run_dir) / "weights" / "last.pt" if run_dir else None
        if last_pt and last_pt.exists():
            role_str = r.get("role", "obb_direct")
            try:
                role = TrainingRole(role_str)
            except ValueError:
                continue
            self._append_log("Resuming " + role.value + " from " + str(last_pt))
            source_obb = self._collect_sources()
            ds = self.role_dataset_dirs.get(role.value, "")
            if not ds:
                QMessageBox.warning(self, "No Dataset", "No dataset for " + role.value + ".")
                return
            spec = TrainingRunSpec(
                role=role,
                source_datasets=source_obb,
                derived_dataset_dir=ds,
                base_model=str(last_pt),
                hyperparams=TrainingHyperParams(
                    epochs=self.spin_epochs.value(),
                    imgsz=self._imgsz_for_role(role),
                    batch=self.spin_batch.value(),
                    lr0=self.spin_lr0.value(),
                    patience=self.spin_patience.value(),
                    workers=self.spin_workers.value(),
                    cache=self.chk_cache.isChecked(),
                ),
                device=self.combo_device.currentText().strip() or "auto",
                seed=self.spin_seed.value(),
                resume_from=str(last_pt),
                publish_policy=PublishPolicy(
                    auto_import=self.chk_auto_import.isChecked(),
                    auto_select=self.chk_auto_select.isChecked(),
                ),
            )
            entry = {
                "role": role,
                "spec": spec,
                "publish_meta": self._publish_meta_for_role(
                    role, self._base_model_for_role(role)
                ),
            }
            self.worker = RoleTrainingWorker(self.orchestrator, [entry])
            self.worker.log_signal.connect(self._append_log)
            self.worker.role_started.connect(self._on_role_started)
            self.worker.role_finished.connect(self._on_role_finished)
            self.worker.progress_signal.connect(self._on_role_progress)
            self.worker.done_signal.connect(self._on_training_done)
            self.btn_train.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.progress.setValue(0)
            self.worker.start()
            return
    QMessageBox.warning(self, "No Checkpoint", "No last.pt checkpoint found to resume.")
```

- [ ] **Step 7: Run all tests and commit**

Run: `python -m pytest tests/test_training_resume.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/training/contracts.py src/hydra_suite/training/runner.py src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py tests/test_training_resume.py
git commit -m "feat(training): add resume-from-checkpoint support"
```

---

## Task 4: Training Run History Viewer

**Why:** The registry at `registry.json` tracks every training run with full specs, timestamps, and results, but there is no UI to browse it. Users cannot compare runs or recall what settings produced a good model.

**Files:**
- Create: `src/hydra_suite/mat/gui/dialogs/run_history_dialog.py`
- Create: `tests/test_run_history.py`
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`

### Step-by-step

- [ ] **Step 1: Write test for run history data loading**

```python
# tests/test_run_history.py
from __future__ import annotations

import json
from pathlib import Path

from hydra_suite.tracker.gui.dialogs.run_history_dialog import load_run_history


def test_load_run_history_empty(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    reg_path.write_text('{"runs": []}', encoding="utf-8")
    runs = load_run_history(str(reg_path))
    assert runs == []


def test_load_run_history_with_entries(tmp_path: Path):
    reg_path = tmp_path / "registry.json"
    data = {
        "runs": [
            {
                "run_id": "20260401-120000_obb_direct_abc12345",
                "started_at": "2026-04-01T12:00:00",
                "finished_at": "2026-04-01T12:30:00",
                "status": "completed",
                "role": "obb_direct",
                "spec": {
                    "hyperparams": {"epochs": 100, "imgsz": 640, "batch": 16},
                    "base_model": "yolo26s-obb.pt",
                },
            },
            {
                "run_id": "20260401-130000_seq_detect_def67890",
                "started_at": "2026-04-01T13:00:00",
                "finished_at": "",
                "status": "failed",
                "role": "seq_detect",
                "spec": {
                    "hyperparams": {"epochs": 50, "imgsz": 640, "batch": 8},
                    "base_model": "yolo26s.pt",
                },
            },
        ]
    }
    reg_path.write_text(json.dumps(data), encoding="utf-8")
    runs = load_run_history(str(reg_path))
    assert len(runs) == 2
    assert runs[0]["status"] == "completed"
    assert runs[1]["status"] == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_history.py -v`
Expected: FAIL -- module does not exist

- [ ] **Step 3: Create run_history_dialog.py**

```python
# src/hydra_suite/mat/gui/dialogs/run_history_dialog.py
"""Training run history viewer dialog."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
)

logger = logging.getLogger(__name__)


def load_run_history(registry_path: str) -> list[dict[str, Any]]:
    """Load run records from a registry JSON file."""
    path = Path(registry_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        return []
    return runs


class RunHistoryDialog(QDialog):
    """Browse past training runs with specs and outcomes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Run History")
        self.resize(1000, 600)

        from hydra_suite.training.registry import get_registry_path

        self.registry_path = str(get_registry_path())
        self.runs = load_run_history(self.registry_path)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Registry: " + self.registry_path))

        self.table = QTableWidget(len(self.runs), 6)
        self.table.setHorizontalHeaderLabels(
            ["Run ID", "Role", "Status", "Started", "Base Model", "Epochs"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)

        for i, run in enumerate(reversed(self.runs)):
            spec = run.get("spec", {})
            hp = spec.get("hyperparams", {})
            self.table.setItem(i, 0, QTableWidgetItem(str(run.get("run_id", ""))))
            self.table.setItem(i, 1, QTableWidgetItem(str(run.get("role", ""))))

            status_item = QTableWidgetItem(str(run.get("status", "")))
            status = str(run.get("status", ""))
            if status == "completed":
                status_item.setForeground(Qt.green)
            elif status == "failed":
                status_item.setForeground(Qt.red)
            elif status == "canceled":
                status_item.setForeground(Qt.yellow)
            self.table.setItem(i, 2, status_item)

            self.table.setItem(i, 3, QTableWidgetItem(str(run.get("started_at", ""))))
            self.table.setItem(i, 4, QTableWidgetItem(str(spec.get("base_model", ""))))
            self.table.setItem(i, 5, QTableWidgetItem(str(hp.get("epochs", ""))))

        self.table.currentCellChanged.connect(self._on_row_changed)
        layout.addWidget(self.table)

        self.detail_view = QTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setMaximumHeight(200)
        self.detail_view.setPlaceholderText("Select a run to see full details.")
        layout.addWidget(self.detail_view)

        btn_row = QHBoxLayout()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def _on_row_changed(self, row: int, _col: int, _prev_row: int, _prev_col: int):
        if row < 0 or row >= len(self.runs):
            return
        run = list(reversed(self.runs))[row]
        self.detail_view.setPlainText(json.dumps(run, indent=2, default=str))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_history.py -v`
Expected: PASS

- [ ] **Step 5: Add "Run History..." button to training dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add:
```python
self.btn_history = QPushButton("Run History...")
row.addWidget(self.btn_history)
self.btn_history.clicked.connect(self._show_history)
```

Add handler:
```python
def _show_history(self):
    from hydra_suite.tracker.gui.dialogs.run_history_dialog import RunHistoryDialog
    dlg = RunHistoryDialog(parent=self)
    dlg.exec()
```

- [ ] **Step 6: Run all tests and commit**

Run: `python -m pytest tests/test_run_history.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/mat/gui/dialogs/run_history_dialog.py tests/test_run_history.py src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py
git commit -m "feat(training): add run history viewer dialog"
```

---

## Task 5: Live Loss/Metrics Plot

**Why:** Training logs stream text but users cannot see if loss is converging or diverging without squinting at numbers. A simple live plot widget that parses ultralytics output and draws loss/mAP curves gives immediate feedback on training health.

**Files:**
- Create: `src/hydra_suite/mat/gui/widgets/loss_plot_widget.py`
- Create: `tests/test_loss_plot_widget.py`
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`

### Step-by-step

- [ ] **Step 1: Write test for metrics parsing**

```python
# tests/test_loss_plot_widget.py
from __future__ import annotations

from hydra_suite.tracker.gui.widgets.loss_plot_widget import parse_ultralytics_log_line


def test_parse_epoch_line():
    line = "      1/100      0.987      1.234      0.567        40       640"
    result = parse_ultralytics_log_line(line)
    assert result is not None
    assert result["epoch"] == 1
    assert result["total_epochs"] == 100
    assert abs(result["box_loss"] - 0.987) < 0.01


def test_parse_non_epoch_line():
    line = "Ultralytics YOLO v8.3.0 - training started"
    result = parse_ultralytics_log_line(line)
    assert result is None


def test_parse_val_metrics_line():
    # Val summary lines do not match the epoch pattern.
    line = "                 all        120        200      0.912      0.887      0.902      0.678"
    result = parse_ultralytics_log_line(line)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loss_plot_widget.py -v`
Expected: FAIL -- module does not exist

- [ ] **Step 3: Create loss_plot_widget.py**

This widget uses only Qt QPainter -- no matplotlib dependency. It draws a simple line chart of parsed loss values.

```python
# src/hydra_suite/mat/gui/widgets/loss_plot_widget.py
"""Live loss plot widget using Qt painting -- no matplotlib required."""
from __future__ import annotations

import re
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

# Ultralytics training output: "  epoch/total  box_loss  cls_loss  dfl_loss  instances  imgsz"
_EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+"    # epoch/total
    r"([\d.]+)\s+"            # box_loss
    r"([\d.]+)\s+"            # cls_loss
    r"([\d.]+)"               # dfl_loss
)


def parse_ultralytics_log_line(line: str) -> dict[str, Any] | None:
    """Parse a single ultralytics training log line into metrics dict."""
    m = _EPOCH_RE.match(line.strip())
    if not m:
        return None
    return {
        "epoch": int(m.group(1)),
        "total_epochs": int(m.group(2)),
        "box_loss": float(m.group(3)),
        "cls_loss": float(m.group(4)),
        "dfl_loss": float(m.group(5)),
    }


class LossPlotWidget(QWidget):
    """Minimal live loss chart drawn with QPainter."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.series: dict[str, list[float]] = {
            "box_loss": [],
            "cls_loss": [],
            "dfl_loss": [],
        }
        self.colors = {
            "box_loss": QColor(66, 133, 244),   # blue
            "cls_loss": QColor(234, 67, 53),     # red
            "dfl_loss": QColor(251, 188, 4),     # amber
        }

    def add_metrics(self, metrics: dict[str, Any]):
        """Append parsed metrics point and trigger repaint."""
        for key in self.series:
            val = metrics.get(key)
            if val is not None:
                self.series[key].append(float(val))
        self.update()

    def clear(self):
        for key in self.series:
            self.series[key].clear()
        self.update()

    def ingest_log_line(self, line: str):
        """Try parsing a log line; if it is a metrics line, add it."""
        parsed = parse_ultralytics_log_line(line)
        if parsed:
            self.add_metrics(parsed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        w = self.width()
        h = self.height()
        margin = 40

        # Find global y range across all series.
        all_vals = []
        for vals in self.series.values():
            all_vals.extend(vals)
        if not all_vals:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "No training data yet")
            painter.end()
            return

        y_min = min(all_vals) * 0.9
        y_max = max(all_vals) * 1.1
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0
        max_n = max(len(v) for v in self.series.values())

        def to_px(i: int, val: float) -> tuple[int, int]:
            x = margin + int((w - 2 * margin) * i / max(1, max_n - 1))
            y = h - margin - int((h - 2 * margin) * (val - y_min) / (y_max - y_min))
            return x, y

        # Draw axes.
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        painter.drawLine(margin, margin, margin, h - margin)

        # Axis labels.
        painter.setPen(QColor(160, 160, 160))
        painter.drawText(5, margin + 5, "{:.2f}".format(y_max))
        painter.drawText(5, h - margin, "{:.2f}".format(y_min))
        painter.drawText(margin, h - 5, "0")
        painter.drawText(w - margin - 20, h - 5, str(max_n))

        # Draw each series.
        for name, vals in self.series.items():
            if len(vals) < 2:
                continue
            pen = QPen(self.colors.get(name, QColor(200, 200, 200)), 2)
            painter.setPen(pen)
            for i in range(1, len(vals)):
                x1, y1 = to_px(i - 1, vals[i - 1])
                x2, y2 = to_px(i, vals[i])
                painter.drawLine(x1, y1, x2, y2)

        # Legend.
        lx = margin + 10
        ly = margin + 5
        for name, color in self.colors.items():
            if not self.series[name]:
                continue
            painter.setPen(QPen(color, 2))
            painter.drawLine(lx, ly, lx + 20, ly)
            painter.setPen(QColor(200, 200, 200))
            last_val = self.series[name][-1]
            painter.drawText(lx + 25, ly + 4, name + ": " + "{:.3f}".format(last_val))
            ly += 15

        painter.end()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loss_plot_widget.py -v`
Expected: PASS

- [ ] **Step 5: Integrate loss plot into training dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add the widget between progress bar and log view:
```python
from hydra_suite.tracker.gui.widgets.loss_plot_widget import LossPlotWidget

self.loss_plot = LossPlotWidget()
self.loss_plot.setMinimumHeight(180)
v.addWidget(self.loss_plot)
```

In `_append_log`, add line ingestion:
```python
def _append_log(self, text: str):
    self.log_view.append(str(text))
    if hasattr(self, "loss_plot"):
        self.loss_plot.ingest_log_line(str(text))
```

In `_start_training`, clear the plot before starting:
```python
if hasattr(self, "loss_plot"):
    self.loss_plot.clear()
```

- [ ] **Step 6: Run all tests and commit**

Run: `python -m pytest tests/test_loss_plot_widget.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/mat/gui/widgets/loss_plot_widget.py tests/test_loss_plot_widget.py src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py
git commit -m "feat(training): add live loss plot to Training Center"
```

---

## Task 6: Export Reproducible Training Config

**Why:** Users need to share or archive training setups. The registry stores spec snapshots, but they are not easily portable. A "Save Config" button that exports a complete, re-importable JSON of all dialog settings enables reproducibility and collaboration.

**Files:**
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`
- Test: `tests/test_training_config_export.py`

### Step-by-step

- [ ] **Step 1: Write test for config format**

```python
# tests/test_training_config_export.py
from __future__ import annotations

import json
from pathlib import Path


def test_training_config_roundtrip(tmp_path: Path):
    """A saved training config JSON should be valid and contain all key fields."""
    config = {
        "version": 1,
        "class_name": "ant",
        "roles": ["obb_direct", "seq_detect", "seq_crop_obb"],
        "sources": [
            {"source_type": "obb", "path": "/data/obb_ds_1"},
            {"source_type": "obb", "path": "/data/obb_ds_2"},
        ],
        "hyperparams": {
            "epochs": 100,
            "batch": 16,
            "lr0": 0.01,
            "patience": 30,
            "workers": 8,
            "cache": False,
        },
        "imgsz": {
            "obb_direct": 640,
            "seq_detect": 640,
            "seq_crop_obb": 160,
        },
        "split": {"train": 0.8, "val": 0.2},
        "seed": 42,
        "dedup": True,
        "crop_derivation": {
            "pad_ratio": 0.15,
            "min_crop_size_px": 64,
            "enforce_square": True,
        },
        "base_models": {
            "obb_direct": "yolo26s-obb.pt",
            "seq_detect": "yolo26s.pt",
            "seq_crop_obb": "yolo26s-obb.pt",
        },
        "augmentation": {
            "enabled": True,
            "fliplr": 0.5,
            "flipud": 0.0,
            "degrees": 0.0,
            "mosaic": 1.0,
            "mixup": 0.0,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
        },
        "device": "auto",
    }

    out = tmp_path / "training_config.json"
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["version"] == 1
    assert loaded["class_name"] == "ant"
    assert len(loaded["sources"]) == 2
    assert loaded["imgsz"]["seq_crop_obb"] == 160
    assert loaded["augmentation"]["fliplr"] == 0.5
```

- [ ] **Step 2: Run test to verify it passes** (this is a data format test)

Run: `python -m pytest tests/test_training_config_export.py -v`
Expected: PASS

- [ ] **Step 3: Add Save/Load Config buttons and handlers to dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add:
```python
self.btn_save_config = QPushButton("Save Config...")
self.btn_load_config = QPushButton("Load Config...")
row.addWidget(self.btn_save_config)
row.addWidget(self.btn_load_config)
self.btn_save_config.clicked.connect(self._save_training_config)
self.btn_load_config.clicked.connect(self._load_training_config)
```

Add save handler (`_save_training_config`) that collects all widget values into a versioned JSON dict with keys: version, class_name, roles, sources, hyperparams, imgsz, split, seed, dedup, crop_derivation, base_models, augmentation, device, publish. Writes via `QFileDialog.getSaveFileName` + `json.dumps`.

Add load handler (`_load_training_config`) that reads a JSON file via `QFileDialog.getOpenFileName`, then sets each widget value from the loaded dict. Uses safe `.get()` access for each key so partial configs still load successfully.

Both handlers should log via `self._append_log()`.

- [ ] **Step 4: Run all tests and commit**

Run: `python -m pytest tests/test_training_config_export.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py tests/test_training_config_export.py
git commit -m "feat(training): add save/load training config for reproducibility"
```

---

## Task 7: Auto-Batch and Multi-GPU

**Why:** Users with powerful hardware cannot easily use auto batch sizing (`batch=-1` in Ultralytics) or multi-GPU training (`device=0,1`). The dialog only has a fixed batch spinner and single-device combo.

**Files:**
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`
- Modify: `src/hydra_suite/training/runner.py`
- Test: `tests/test_training_batch_gpu.py`

### Step-by-step

- [ ] **Step 1: Write test**

```python
# tests/test_training_batch_gpu.py
from __future__ import annotations

from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_auto_batch_flag():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640, batch=-1),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "batch=-1" in cmd_str


def test_multi_gpu_device():
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=10, imgsz=640),
        device="0,1",
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    cmd_str = " ".join(cmd)
    assert "device=0,1" in cmd_str
```

- [ ] **Step 2: Run test to verify behavior**

Run: `python -m pytest tests/test_training_batch_gpu.py -v`
Expected: Both PASS (the runner already passes batch and device as-is)

- [ ] **Step 3: Add auto-batch checkbox and editable device combo to dialog**

In `train_yolo_dialog.py`, modify batch spinner area to add an "Auto" checkbox:
```python
self.chk_auto_batch = QCheckBox("Auto")
self.chk_auto_batch.setToolTip(
    "Let Ultralytics auto-detect optimal batch size (batch=-1). "
    "Overrides manual batch setting."
)
self.chk_auto_batch.toggled.connect(
    lambda checked: self.spin_batch.setEnabled(not checked)
)
```

Make the device combo editable so users can type `0,1`:
```python
self.combo_device.setEditable(True)
self.combo_device.setToolTip(
    "Select compute device. For multi-GPU, type a comma-separated list "
    "like '0,1' in the editable combo box."
)
```

Wire auto-batch into spec creation in `_start_training`:
```python
batch_val = -1 if self.chk_auto_batch.isChecked() else self.spin_batch.value()
```

- [ ] **Step 4: Run all tests and commit**

Run: `python -m pytest tests/test_training_batch_gpu.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py tests/test_training_batch_gpu.py
git commit -m "feat(training): add auto-batch sizing and multi-GPU device support"
```

---

## Task 8: Background / Detached Training

**Why:** Long training runs (hours) block the Training Center dialog. Users cannot track while training. A "Detach" option launches training as a standalone subprocess that writes results to a log file, so the dialog can close.

**Files:**
- Modify: `src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py`
- Modify: `src/hydra_suite/training/runner.py`
- Test: `tests/test_training_detach.py`

### Step-by-step

- [ ] **Step 1: Write test for detached command building**

```python
# tests/test_training_detach.py
from __future__ import annotations

from hydra_suite.training.contracts import (
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from hydra_suite.training.runner import build_ultralytics_command


def test_detached_command_is_valid():
    """Detached training reuses the same command -- it is just launched differently."""
    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[],
        derived_dataset_dir="/tmp/ds",
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(epochs=100, imgsz=640),
    )
    cmd = build_ultralytics_command(spec, "/tmp/run")
    assert len(cmd) > 0
    assert "train" in cmd
    assert any("epochs=100" in arg for arg in cmd)
```

- [ ] **Step 2: Run test to confirm it passes**

Run: `python -m pytest tests/test_training_detach.py -v`
Expected: PASS

- [ ] **Step 3: Add "Start Detached" button to training dialog**

In `train_yolo_dialog.py`, in `_build_run_group`, add:
```python
self.btn_detach = QPushButton("Start Detached")
self.btn_detach.setToolTip(
    "Launch training as a background process. You can close this dialog "
    "and continue tracking. Check Run History for results."
)
row.addWidget(self.btn_detach)
self.btn_detach.clicked.connect(self._start_detached)
```

Add handler `_start_detached` that:
1. Validates roles and datasets (same as `_start_training`)
2. Builds `TrainingRunSpec` for each role
3. Calls `build_ultralytics_command(spec, run_dir)` to get the command
4. Opens a log file at `run_dir / "detached_output.log"`
5. Launches via `subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)`
6. Shows a QMessageBox with PID and log path for each launched role

- [ ] **Step 4: Run all tests and commit**

Run: `python -m pytest tests/test_training_detach.py tests/test_training_framework.py -v`
Expected: All PASS

```bash
git add src/hydra_suite/mat/gui/dialogs/train_yolo_dialog.py tests/test_training_detach.py
git commit -m "feat(training): add detached background training mode"
```

---

## Task 9: Stratified Dataset Splitting

**Why:** Current splitting is random shuffle -- small datasets may have all examples of a rare class end up in one split. Stratified splitting ensures proportional class representation in train/val/test.

**Files:**
- Modify: `src/hydra_suite/training/dataset_builders.py`
- Modify: `src/hydra_suite/training/dataset_inspector.py`
- Test: `tests/test_stratified_split.py`

### Step-by-step

- [ ] **Step 1: Write failing test**

```python
# tests/test_stratified_split.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hydra_suite.training.dataset_inspector import (
    DatasetItem,
    stratified_split_items,
)


def _write_label(path: Path, class_ids: list[int]):
    """Write OBB labels with given class IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cid in class_ids:
        lines.append(str(cid) + " 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stratified_split_preserves_class_proportions(tmp_path: Path):
    """Each split should have roughly proportional class representation."""
    items = []
    for i in range(20):
        img_path = tmp_path / ("img_" + str(i) + ".jpg")
        lbl_path = tmp_path / ("img_" + str(i) + ".txt")
        img = np.full((80, 120, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        cls_id = 0 if i < 14 else 1
        _write_label(lbl_path, [cls_id])
        items.append(DatasetItem(
            image_path=str(img_path), label_path=str(lbl_path), split="all"
        ))

    result = stratified_split_items(items, split_cfg=(0.7, 0.3, 0.0), seed=42)
    train_items = result["train"]
    val_items = result["val"]

    def count_classes(split_items):
        counts = {0: 0, 1: 0}
        for item in split_items:
            for line in Path(item.label_path).read_text().strip().splitlines():
                cid = int(float(line.split()[0]))
                counts[cid] = counts.get(cid, 0) + 1
        return counts

    train_counts = count_classes(train_items)
    val_counts = count_classes(val_items)

    # Class 1 should appear in both splits (not all in one).
    assert val_counts.get(1, 0) >= 1, "Class 1 should appear in val split"
    assert train_counts.get(1, 0) >= 1, "Class 1 should appear in train split"


def test_stratified_split_handles_single_class():
    """Single-class datasets should still split normally."""
    items = [
        DatasetItem(image_path="/img_" + str(i) + ".jpg", label_path="/lbl_" + str(i) + ".txt", split="all")
        for i in range(10)
    ]
    # No label files exist, so stratification falls back to random.
    result = stratified_split_items(items, split_cfg=(0.7, 0.3, 0.0), seed=42)
    assert len(result["train"]) + len(result["val"]) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_stratified_split.py -v`
Expected: FAIL -- `stratified_split_items` does not exist

- [ ] **Step 3: Add `stratified_split_items` to dataset_inspector.py**

Add at the end of `src/hydra_suite/training/dataset_inspector.py`:

```python
def _read_class_ids_from_label(label_path: str) -> set[int]:
    """Extract class IDs from an OBB/detect label file."""
    path = Path(label_path)
    if not path.exists():
        return set()
    ids: set[int] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                ids.add(int(float(parts[0])))
    except Exception:
        pass
    return ids


def stratified_split_items(
    items: list[DatasetItem],
    split_cfg: tuple[float, float, float],
    seed: int,
) -> dict[str, list[DatasetItem]]:
    """Split items with stratification by class distribution.

    Falls back to random shuffle if labels are unreadable or single-class.
    """
    import random

    rng = random.Random(int(seed))
    train_r, val_r, test_r = split_cfg
    total = max(1e-8, float(train_r) + float(val_r) + float(test_r))
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    # Group items by their dominant class (first class ID in label).
    class_buckets: dict[int, list[DatasetItem]] = {}
    unclassified: list[DatasetItem] = []
    for item in items:
        ids = _read_class_ids_from_label(item.label_path)
        if ids:
            dominant = min(ids)  # deterministic tie-break
            class_buckets.setdefault(dominant, []).append(item)
        else:
            unclassified.append(item)

    # If single class or no labels, fall back to simple shuffle.
    if len(class_buckets) <= 1 and not unclassified:
        all_items = list(items)
        rng.shuffle(all_items)
        n = len(all_items)
        n_train = max(1, int(round(n * train_r))) if n >= 2 else n
        n_val = max(1, min(n - n_train, int(round(n * val_r)))) if n >= 2 else 0
        train = all_items[:n_train]
        val = all_items[n_train:n_train + n_val]
        test = all_items[n_train + n_val:]
    else:
        train, val, test = [], [], []
        for _cls_id, bucket in sorted(class_buckets.items()):
            rng.shuffle(bucket)
            n = len(bucket)
            n_train = max(1, int(round(n * train_r))) if n >= 2 else n
            n_val = max(0, min(n - n_train, int(round(n * val_r))))
            if n >= 2 and n_val == 0:
                n_val = 1
                n_train = min(n_train, n - n_val)
            train.extend(bucket[:n_train])
            val.extend(bucket[n_train:n_train + n_val])
            test.extend(bucket[n_train + n_val:])
        rng.shuffle(unclassified)
        n_u = len(unclassified)
        n_u_train = int(round(n_u * train_r))
        n_u_val = int(round(n_u * val_r))
        train.extend(unclassified[:n_u_train])
        val.extend(unclassified[n_u_train:n_u_train + n_u_val])
        test.extend(unclassified[n_u_train + n_u_val:])
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

    for it in train:
        it.split = "train"
    for it in val:
        it.split = "val"
    for it in test:
        it.split = "test"

    return {"train": train, "val": val, "test": test}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_stratified_split.py -v`
Expected: PASS

- [ ] **Step 5: Wire stratified splitting into merge_obb_sources**

In `src/hydra_suite/training/dataset_builders.py`, add to imports:
```python
from .dataset_inspector import (
    DatasetInspection,
    inspect_obb_or_detect_dataset,
    split_items_for_training,
    stratified_split_items,
)
```

Then in `merge_obb_sources`, where `split_items_for_training` is called for unsplit sources, replace with:
```python
split_map = stratified_split_items(
    list(insp.splits.get("all", [])),
    split_cfg=(train_r, val_r, test_r),
    seed=seed,
)
```

This is a drop-in replacement -- `stratified_split_items` returns the same `dict[str, list[DatasetItem]]` structure.

- [ ] **Step 6: Run all training tests**

Run: `python -m pytest tests/test_stratified_split.py tests/test_training_framework.py -v`
Expected: All PASS (existing tests still pass since stratified splitting is backward-compatible with single-class datasets)

- [ ] **Step 7: Commit**

```bash
git add src/hydra_suite/training/dataset_inspector.py src/hydra_suite/training/dataset_builders.py tests/test_stratified_split.py
git commit -m "feat(training): add stratified dataset splitting for class-balanced train/val"
```

---

## Dependency Order

Tasks are independent and can be implemented in any order. However, this sequence minimizes merge conflicts:

1. **Task 9** (stratified splitting) -- backend only, no UI overlap
2. **Task 1** (augmentation controls) -- adds UI group, referenced by Task 6
3. **Task 5** (live loss plot) -- adds widget, modifies `_append_log`
4. **Task 3** (resume) -- adds button to run group
5. **Task 7** (auto-batch/multi-GPU) -- modifies batch widget
6. **Task 8** (detached training) -- adds button to run group, references auto-batch
7. **Task 4** (run history) -- new dialog, one button addition
8. **Task 2** (quick test) -- new dialog, depends on training results
9. **Task 6** (config export) -- captures full dialog state, should be last since it references all widgets including augmentation

Tasks 1-3 and 9 are the highest value. Tasks 4-6 are medium. Tasks 7-8 are nice-to-have.
