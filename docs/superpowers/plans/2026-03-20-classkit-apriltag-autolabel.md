# ClassKit AprilTag Auto-Labeler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Auto-label AprilTags" feature to ClassKit that runs MAT's AprilTag detector across multiple preprocessing profiles on each cropped image, auto-populates confident labels, and creates the labeling scheme automatically from user-configured tag family and max tag ID.

**Architecture:** A pure-Python `classkit/autolabel/apriltag.py` module holds all detection logic (no Qt); a `AprilTagAutoLabelWorker` in `task_workers.py` runs it in the background; `AprilTagAutoLabelDialog` in `dialogs.py` handles configuration; `mainwindow.py` wires the menu action, scheme creation, and worker lifecycle together.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, MAT's `AprilTagDetector`/`AprilTagConfig` from `multi_tracker.core.identity.apriltag_detector`, PySide6 (`QRunnable`, `QDialog`), SQLite via `ClassKitDB`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/multi_tracker/classkit/autolabel/__init__.py` | Create | Package exports |
| `src/multi_tracker/classkit/autolabel/apriltag.py` | Create | Core auto-labeler: profiles, `LabelResult`, `AprilTagAutoLabeler`, `autolabel_images()` |
| `src/multi_tracker/classkit/presets.py` | Modify | Add `apriltag_preset()` |
| `src/multi_tracker/classkit/store/db.py` | Modify | Add `update_labels_with_confidence_batch()` and `clear_all_labels()` |
| `src/multi_tracker/classkit/jobs/task_workers.py` | Modify | Add `AprilTagAutoLabelWorker` |
| `src/multi_tracker/classkit/gui/dialogs.py` | Modify | Add `AprilTagAutoLabelDialog` |
| `src/multi_tracker/classkit/gui/mainwindow.py` | Modify | Add menu action, scheme write, label clear, worker hookup |
| `tests/test_classkit_apriltag_autolabel.py` | Create | All unit tests (pure logic only, no Qt) |

---

## Task 1: `apriltag_preset()` in `presets.py`

**Files:**
- Modify: `src/multi_tracker/classkit/presets.py`
- Test: `tests/test_classkit_apriltag_autolabel.py` (create file here)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classkit_apriltag_autolabel.py`:

```python
"""Tests for ClassKit AprilTag auto-labeler."""
from __future__ import annotations

import pytest
from multi_tracker.classkit.presets import apriltag_preset


def test_apriltag_preset_labels():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    labels = scheme.factors[0].labels
    assert labels[:10] == [f"tag_{i}" for i in range(10)]
    assert labels[-1] == "no_tag"
    assert len(labels) == 11  # tag_0..tag_9 + no_tag


def test_apriltag_preset_total_classes():
    scheme = apriltag_preset("tag36h11", max_tag_id=4)
    assert scheme.total_classes == 6  # tag_0..tag_4 + no_tag


def test_apriltag_preset_single_factor():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert len(scheme.factors) == 1


def test_apriltag_preset_factor_name_matches_family():
    scheme = apriltag_preset("tag25h9", max_tag_id=5)
    assert scheme.factors[0].name == "tag25h9"


def test_apriltag_preset_name():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.name == "apriltag_tag36h11"


def test_apriltag_preset_flat_training_modes_only():
    scheme = apriltag_preset("tag36h11", max_tag_id=9)
    assert scheme.training_modes == ["flat_tiny", "flat_yolo"]


def test_apriltag_preset_max_tag_id_zero():
    scheme = apriltag_preset("tag36h11", max_tag_id=0)
    assert scheme.factors[0].labels == ["tag_0", "no_tag"]
    assert scheme.total_classes == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker"
python -m pytest tests/test_classkit_apriltag_autolabel.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'apriltag_preset'`

- [ ] **Step 3: Implement `apriltag_preset()`**

Add to the end of `src/multi_tracker/classkit/presets.py`:

```python
def apriltag_preset(family: str, max_tag_id: int) -> LabelingScheme:
    """Single-factor scheme for AprilTag ID classification.

    Args:
        family: AprilTag family string (e.g. 'tag36h11').
        max_tag_id: Highest tag ID to include. Labels will be tag_0..tag_N + no_tag.
    """
    labels = [f"tag_{i}" for i in range(max_tag_id + 1)] + ["no_tag"]
    return LabelingScheme(
        name=f"apriltag_{family}",
        factors=[Factor(name=family, labels=labels)],
        training_modes=["flat_tiny", "flat_yolo"],
        description=f"AprilTag {family} classifier: tag_0..tag_{max_tag_id} + no_tag",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_labels tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_total_classes tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_single_factor tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_factor_name_matches_family tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_name tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_flat_training_modes_only tests/test_classkit_apriltag_autolabel.py::test_apriltag_preset_max_tag_id_zero -v
```

Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/multi_tracker/classkit/presets.py tests/test_classkit_apriltag_autolabel.py
git commit -m "feat(classkit): add apriltag_preset() to presets"
```

---

## Task 2: `ClassKitDB` additions

**Files:**
- Modify: `src/multi_tracker/classkit/store/db.py`
- Test: `tests/test_classkit_apriltag_autolabel.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classkit_apriltag_autolabel.py`:

```python
import sqlite3
from pathlib import Path
from multi_tracker.classkit.store.db import ClassKitDB


def _make_db(tmp_path: Path) -> ClassKitDB:
    """Create a DB with two test images."""
    db = ClassKitDB(tmp_path / "test.db")
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/a.jpg", "aaa"),
        )
        conn.execute(
            "INSERT INTO images (file_path, file_hash) VALUES (?, ?)",
            ("/img/b.jpg", "bbb"),
        )
        conn.commit()
    return db


def test_update_labels_with_confidence_batch_writes_label_and_confidence(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch({
        "/img/a.jpg": ("tag_3", 0.8),
        "/img/b.jpg": ("no_tag", 1.0),
    })
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT file_path, label, confidence FROM images ORDER BY file_path"
        ).fetchall()
    assert rows[0] == ("/img/a.jpg", "tag_3", 0.8)
    assert rows[1] == ("/img/b.jpg", "no_tag", 1.0)


def test_update_labels_with_confidence_batch_empty_is_noop(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch({})  # must not raise
    labels = db.get_all_labels()
    assert all(lbl is None for lbl in labels)


def test_clear_all_labels_sets_null(tmp_path):
    db = _make_db(tmp_path)
    db.update_labels_with_confidence_batch({
        "/img/a.jpg": ("tag_3", 0.8),
    })
    db.clear_all_labels()
    with sqlite3.connect(db.db_path) as conn:
        rows = conn.execute(
            "SELECT label, confidence FROM images"
        ).fetchall()
    assert all(row[0] is None and row[1] is None for row in rows)


def test_clear_all_labels_on_empty_db_is_noop(tmp_path):
    db = ClassKitDB(tmp_path / "empty.db")
    db.clear_all_labels()  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py::test_update_labels_with_confidence_batch_writes_label_and_confidence tests/test_classkit_apriltag_autolabel.py::test_clear_all_labels_sets_null -v 2>&1 | head -20
```

Expected: `AttributeError: 'ClassKitDB' object has no attribute 'update_labels_with_confidence_batch'`

- [ ] **Step 3: Implement the two DB methods**

Add after `update_labels_batch()` (around line 206) in `src/multi_tracker/classkit/store/db.py`:

```python
def update_labels_with_confidence_batch(
    self, updates: Dict[str, Tuple[str, float]]
) -> None:
    """Write label and confidence for multiple images in one transaction.

    Args:
        updates: mapping of {file_path: (label, confidence)}
    """
    if not updates:
        return
    rows = [(label, confidence, path) for path, (label, confidence) in updates.items()]
    with sqlite3.connect(self.db_path) as conn:
        conn.executemany(
            "UPDATE images SET label = ?, confidence = ? WHERE file_path = ?",
            rows,
        )
        conn.commit()

def clear_all_labels(self) -> None:
    """Set label=NULL and confidence=NULL for all images."""
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("UPDATE images SET label = NULL, confidence = NULL")
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py::test_update_labels_with_confidence_batch_writes_label_and_confidence tests/test_classkit_apriltag_autolabel.py::test_update_labels_with_confidence_batch_empty_is_noop tests/test_classkit_apriltag_autolabel.py::test_clear_all_labels_sets_null tests/test_classkit_apriltag_autolabel.py::test_clear_all_labels_on_empty_db_is_noop -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/multi_tracker/classkit/store/db.py tests/test_classkit_apriltag_autolabel.py
git commit -m "feat(classkit): add update_labels_with_confidence_batch and clear_all_labels to ClassKitDB"
```

---

## Task 3: Core autolabel module

**Files:**
- Create: `src/multi_tracker/classkit/autolabel/__init__.py`
- Create: `src/multi_tracker/classkit/autolabel/apriltag.py`
- Test: `tests/test_classkit_apriltag_autolabel.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_classkit_apriltag_autolabel.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a solid gray BGR uint8 image for testing."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_observation(tag_id: int, hamming: int = 0):
    """Create a mock TagObservation."""
    obs = MagicMock()
    obs.tag_id = tag_id
    obs.hamming = hamming
    return obs


def _make_config(unsharp_kernel_size=(5, 5), unsharp_sigma=1.0, unsharp_amount=1.5):
    from multi_tracker.core.identity.apriltag_detector import AprilTagConfig
    return AprilTagConfig(
        unsharp_kernel_size=unsharp_kernel_size,
        unsharp_sigma=unsharp_sigma,
        unsharp_amount=unsharp_amount,
    )


# ---------------------------------------------------------------------------
# Preprocessing profile tests
# ---------------------------------------------------------------------------

def test_profile_raw_returns_same_array():
    from multi_tracker.classkit.autolabel.apriltag import _profile_raw
    img = _bgr_image()
    out = _profile_raw(img)
    np.testing.assert_array_equal(out, img)


def test_profile_raw_is_3channel_bgr():
    from multi_tracker.classkit.autolabel.apriltag import _profile_raw
    out = _profile_raw(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_clahe_is_3channel_bgr():
    from multi_tracker.classkit.autolabel.apriltag import _profile_clahe
    out = _profile_clahe(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_clahe_preserves_shape():
    from multi_tracker.classkit.autolabel.apriltag import _profile_clahe
    img = _bgr_image(32, 48)
    out = _profile_clahe(img)
    assert out.shape == img.shape


def test_profile_gamma_boost_is_3channel_bgr():
    from multi_tracker.classkit.autolabel.apriltag import _profile_gamma_boost
    out = _profile_gamma_boost(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_profile_gamma_boost_brightens_dark_image():
    from multi_tracker.classkit.autolabel.apriltag import _profile_gamma_boost
    dark = np.full((8, 8, 3), 50, dtype=np.uint8)
    bright = _profile_gamma_boost(dark)
    assert int(bright.mean()) > int(dark.mean())


def test_profile_gamma_darken_darkens_bright_image():
    from multi_tracker.classkit.autolabel.apriltag import _profile_gamma_darken
    bright = np.full((8, 8, 3), 200, dtype=np.uint8)
    dark = _profile_gamma_darken(bright)
    assert int(dark.mean()) < int(bright.mean())


def test_profile_gamma_darken_is_3channel_bgr():
    from multi_tracker.classkit.autolabel.apriltag import _profile_gamma_darken
    out = _profile_gamma_darken(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_unsharp_strong_is_3channel_bgr():
    from multi_tracker.classkit.autolabel.apriltag import _make_unsharp_strong
    profile = _make_unsharp_strong(_make_config())
    out = profile(_bgr_image())
    assert out.ndim == 3 and out.shape[2] == 3 and out.dtype == np.uint8


def test_unsharp_strong_preserves_shape():
    from multi_tracker.classkit.autolabel.apriltag import _make_unsharp_strong
    profile = _make_unsharp_strong(_make_config())
    img = _bgr_image(32, 48)
    assert profile(img).shape == img.shape


# ---------------------------------------------------------------------------
# AprilTagAutoLabeler tests (AprilTagDetector is mocked)
# ---------------------------------------------------------------------------

@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_detector_called_with_internal_config_no_preprocessing(MockDetector):
    """Internal config must have unsharp_amount=0.0 and contrast_factor=1.0."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler
    config = _make_config()
    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    labeler = AprilTagAutoLabeler(config, confidence_threshold=0.6)
    # Check that the config passed to the detector has preprocessing disabled
    call_args = MockDetector.call_args
    internal_cfg = call_args[0][0]
    assert internal_cfg.unsharp_amount == 0.0
    assert internal_cfg.contrast_factor == 1.0


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_detector_wrapped_with_crops_and_offsets(MockDetector):
    """Each profile call must wrap the image as crops=[img], offsets_xy=[(0,0)]."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler
    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    labeler.label_image(_bgr_image())

    for call in mock_inst.detect_in_crops.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args
        crops = kwargs.get("crops", args[0] if args else None)
        offsets = kwargs.get("offsets_xy", args[1] if len(args) > 1 else None)
        assert crops is not None and len(crops) == 1
        assert offsets is not None and offsets == [(0, 0)]


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_no_tag_when_all_profiles_return_empty(MockDetector):
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler
    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label == "no_tag"
    assert result.confidence == 1.0
    assert result.detected_tag_id is None
    assert result.all_no_tag is True


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_no_tag_when_all_profiles_ambiguous(MockDetector):
    """Ambiguous (tied hamming) profiles also produce no_tag with confidence 1.0."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler
    mock_inst = MockDetector.return_value
    # Two observations with same hamming → AMBIGUOUS → None per profile
    mock_inst.detect_in_crops.return_value = [
        _make_observation(tag_id=1, hamming=0),
        _make_observation(tag_id=2, hamming=0),
    ]

    labeler = AprilTagAutoLabeler(_make_config(), 0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label == "no_tag"
    assert result.confidence == 1.0


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_confidence_fraction_correct(MockDetector):
    """3 of 5 profiles agree on tag_3 → confidence = 3/5 = 0.6."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        # Profiles 1,2,3 return tag_3; profiles 4,5 return no tag
        if call_count[0] <= 3:
            return [_make_observation(tag_id=3, hamming=0)]
        return []

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    assert result.label == "tag_3"
    assert abs(result.confidence - 0.6) < 1e-9
    assert result.detected_tag_id == 3


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_below_threshold_returns_none_label(MockDetector):
    """1 of 5 profiles agree → confidence=0.2, below threshold=0.6 → label=None."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [_make_observation(tag_id=5, hamming=0)]
        return []

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.6)
    result = labeler.label_image(_bgr_image())

    assert result.label is None
    assert abs(result.confidence - 0.2) < 1e-9


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_multi_tag_tie_discards_profile(MockDetector):
    """Two observations with same hamming → AMBIGUOUS → profile counts as no detection."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        # First profile: ambiguous (two tags same hamming)
        if call_count[0] == 1:
            return [
                _make_observation(tag_id=1, hamming=0),
                _make_observation(tag_id=2, hamming=0),
            ]
        # Remaining 4 profiles: clear tag_7
        return [_make_observation(tag_id=7, hamming=0)]

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    # 4 out of 5 profiles detected tag_7 → confidence=0.8
    assert result.label == "tag_7"
    assert abs(result.confidence - 0.8) < 1e-9


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_multi_tag_clear_winner_used(MockDetector):
    """Profile with two observations, one has lower hamming → that tag_id is used."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = [
        _make_observation(tag_id=1, hamming=1),  # higher hamming → loser
        _make_observation(tag_id=2, hamming=0),  # lower hamming → winner
    ]

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.1)
    result = labeler.label_image(_bgr_image())

    # All 5 profiles return tag_2 → confidence=1.0
    assert result.detected_tag_id == 2
    assert result.label == "tag_2"


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_majority_vote_wins(MockDetector):
    """3 profiles return tag_1, 2 profiles return tag_2 → tag_1 wins."""
    from multi_tracker.classkit.autolabel.apriltag import AprilTagAutoLabeler

    mock_inst = MockDetector.return_value
    call_count = [0]

    def side_effect(crops, offsets_xy, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 3:
            return [_make_observation(tag_id=1, hamming=0)]
        return [_make_observation(tag_id=2, hamming=0)]

    mock_inst.detect_in_crops.side_effect = side_effect

    labeler = AprilTagAutoLabeler(_make_config(), confidence_threshold=0.5)
    result = labeler.label_image(_bgr_image())

    assert result.detected_tag_id == 1
    assert result.label == "tag_1"


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_autolabel_images_empty_list(MockDetector):
    from multi_tracker.classkit.autolabel.apriltag import autolabel_images
    results = autolabel_images([], _make_config(), threshold=0.6)
    assert results == []
    MockDetector.assert_not_called()


@patch("multi_tracker.classkit.autolabel.apriltag.AprilTagDetector")
def test_autolabel_images_returns_one_result_per_path(MockDetector, tmp_path):
    from multi_tracker.classkit.autolabel.apriltag import autolabel_images
    import cv2 as _cv2
    mock_inst = MockDetector.return_value
    mock_inst.detect_in_crops.return_value = []

    # Create two real image files
    for name in ("a.jpg", "b.jpg"):
        _cv2.imwrite(str(tmp_path / name), _bgr_image())

    results = autolabel_images(
        [tmp_path / "a.jpg", tmp_path / "b.jpg"], _make_config(), threshold=0.6
    )
    assert len(results) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py::test_no_tag_when_all_profiles_return_empty tests/test_classkit_apriltag_autolabel.py::test_confidence_fraction_correct -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'multi_tracker.classkit.autolabel'`

- [ ] **Step 3: Create the package init**

Create `src/multi_tracker/classkit/autolabel/__init__.py`:

```python
"""ClassKit auto-labeling tools."""
from .apriltag import AprilTagAutoLabeler, LabelResult, autolabel_images

__all__ = ["AprilTagAutoLabeler", "LabelResult", "autolabel_images"]
```

- [ ] **Step 4: Create the core module**

Create `src/multi_tracker/classkit/autolabel/apriltag.py`:

```python
"""AprilTag auto-labeler for ClassKit.

Runs MAT's AprilTagDetector across multiple preprocessing profiles on each
cropped image. Confidence is the fraction of profiles agreeing on the same
tag ID. Only labels above the confidence threshold are committed.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from ...core.identity.apriltag_detector import AprilTagConfig, AprilTagDetector

# ---------------------------------------------------------------------------
# Preprocessing profiles
# All must return 3-channel BGR uint8 — required because AprilTagDetector's
# _detect_composite always runs cv2.cvtColor(composite, COLOR_BGR2GRAY).
# ---------------------------------------------------------------------------


def _profile_raw(image: np.ndarray) -> np.ndarray:
    """Pass image through unchanged."""
    return image


def _profile_clahe(image: np.ndarray) -> np.ndarray:
    """CLAHE on grayscale, returned as BGR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _profile_gamma_boost(image: np.ndarray) -> np.ndarray:
    """Gamma γ=0.5: brightens dark images."""
    lut = np.array([int(((i / 255.0) ** 0.5) * 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def _profile_gamma_darken(image: np.ndarray) -> np.ndarray:
    """Gamma γ=2.0: darkens overexposed images."""
    lut = np.array([int(((i / 255.0) ** 2.0) * 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, lut)


def _make_unsharp_strong(
    config: AprilTagConfig,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return an unsharp-mask profile using 2× the user-configured amount."""
    ks = config.unsharp_kernel_size  # Tuple[int, int] e.g. (5, 5)
    sigma = config.unsharp_sigma
    amount = config.unsharp_amount * 2.0

    def _profile(image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, ks, sigma)
        return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)

    return _profile


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class LabelResult:
    """Outcome of auto-labeling one image."""

    label: Optional[str]
    """'tag_N', 'no_tag', or None (uncertain — leave unlabeled)."""

    confidence: float
    """count_of_majority_tag / N_total_profiles."""

    n_profiles_run: int
    """Always equal to len(PREPROCESSING_PROFILES) = 5."""

    detected_tag_id: Optional[int]
    """Raw integer majority tag ID, or None."""

    all_no_tag: bool = False
    """True when every profile returned no observations (or all ambiguous)."""


# ---------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------

_N_PROFILES = 5  # raw, clahe, gamma_boost, gamma_darken, unsharp_strong


class AprilTagAutoLabeler:
    """Detect AprilTags in a single crop using multiple preprocessing profiles.

    The internal ``AprilTagDetector`` is constructed with its own preprocessing
    disabled (``unsharp_amount=0.0``, ``contrast_factor=1.0``) so that each
    profile fully controls what the detector sees.
    """

    def __init__(self, config: AprilTagConfig, confidence_threshold: float = 0.6):
        self.config = config
        self.threshold = confidence_threshold
        # Disable detector's built-in preprocessing so profiles are in full control
        internal_config = dataclasses.replace(
            config, unsharp_amount=0.0, contrast_factor=1.0
        )
        self._detector = AprilTagDetector(internal_config)
        self._profiles: List[Callable[[np.ndarray], np.ndarray]] = [
            _profile_raw,
            _profile_clahe,
            _profile_gamma_boost,
            _profile_gamma_darken,
            _make_unsharp_strong(config),
        ]

    def _run_profile(
        self, image: np.ndarray, profile_fn: Callable[[np.ndarray], np.ndarray]
    ) -> Optional[int]:
        """Apply one profile and run detection. Returns tag_id or None."""
        preprocessed = profile_fn(image)
        observations = self._detector.detect_in_crops(
            crops=[preprocessed], offsets_xy=[(0, 0)]
        )
        if not observations:
            return None  # NO_TAG
        if len(observations) == 1:
            return observations[0].tag_id
        # Multiple observations: pick lowest-hamming; if tied → AMBIGUOUS → None
        min_hamming = min(o.hamming for o in observations)
        winners = [o for o in observations if o.hamming == min_hamming]
        if len(winners) == 1:
            return winners[0].tag_id
        return None  # AMBIGUOUS

    def label_image(self, image: np.ndarray) -> LabelResult:
        """Label one BGR crop image."""
        n = len(self._profiles)
        per_profile: List[Optional[int]] = [
            self._run_profile(image, pf) for pf in self._profiles
        ]
        detected = [tag_id for tag_id in per_profile if tag_id is not None]

        if not detected:
            return LabelResult(
                label="no_tag",
                confidence=1.0,
                n_profiles_run=n,
                detected_tag_id=None,
                all_no_tag=True,
            )

        counts = Counter(detected)
        majority_id, majority_count = counts.most_common(1)[0]
        confidence = majority_count / n

        if confidence >= self.threshold:
            return LabelResult(
                label=f"tag_{majority_id}",
                confidence=confidence,
                n_profiles_run=n,
                detected_tag_id=majority_id,
            )
        return LabelResult(
            label=None,
            confidence=confidence,
            n_profiles_run=n,
            detected_tag_id=majority_id,
        )

    def close(self) -> None:
        """Release detector resources."""
        self._detector.close()


# ---------------------------------------------------------------------------
# Batch entrypoint
# ---------------------------------------------------------------------------


def autolabel_images(
    image_paths: List[Path],
    config: AprilTagConfig,
    threshold: float,
) -> List[LabelResult]:
    """Auto-label a list of image paths.

    Returns one ``LabelResult`` per path, in the same order. Returns an empty
    list when ``image_paths`` is empty. Images that cannot be read (bad path or
    corrupt file) produce ``LabelResult(label=None, confidence=0.0, ...)``.
    """
    if not image_paths:
        return []

    labeler = AprilTagAutoLabeler(config, threshold)
    results: List[LabelResult] = []
    try:
        for path in image_paths:
            image = cv2.imread(str(path))
            if image is None:
                results.append(
                    LabelResult(
                        label=None,
                        confidence=0.0,
                        n_profiles_run=_N_PROFILES,
                        detected_tag_id=None,
                    )
                )
                continue
            results.append(labeler.label_image(image))
    finally:
        labeler.close()
    return results
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py -k "profile or no_tag or confidence or threshold or ambiguous or multi_tag or majority or autolabel_images" -v
```

Expected: all profile tests and labeler logic tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/multi_tracker/classkit/autolabel/ tests/test_classkit_apriltag_autolabel.py
git commit -m "feat(classkit): add AprilTag auto-labeler core module"
```

---

## Task 4: Background worker `AprilTagAutoLabelWorker`

**Files:**
- Modify: `src/multi_tracker/classkit/jobs/task_workers.py`

No Qt unit tests needed for the worker itself (consistent with existing worker pattern in this codebase — see `ExportWorker` tests which only test error paths, not the full Qt run loop). The worker is exercised via the GUI in Task 6.

- [ ] **Step 1: Add the worker**

Append to the end of `src/multi_tracker/classkit/jobs/task_workers.py`:

```python
class AprilTagAutoLabelWorker(QRunnable):
    """Background worker that runs AprilTag auto-labeling on a list of images.

    The caller (mainwindow) is responsible for:
    - pre-filtering image_paths to unlabeled images only
    - writing the labeling scheme before starting the worker
    - connecting signals to update the UI

    This worker uses ``db`` only for writing labels — it never reads from it.
    """

    def __init__(
        self,
        image_paths: List[Path],
        config,  # AprilTagConfig — imported lazily to avoid hard dep at module load
        threshold: float,
        db,      # ClassKitDB — imported lazily
    ):
        super().__init__()
        self.setAutoDelete(False)  # prevent Qt from freeing C++ side before Python GC
        self.image_paths = image_paths
        self.config = config
        self.threshold = threshold
        self.db = db
        self.signals = TaskSignals()
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    @Slot()
    def run(self) -> None:
        from ..autolabel.apriltag import autolabel_images

        try:
            self.signals.started.emit()
            total = len(self.image_paths)
            if total == 0:
                self.signals.success.emit(
                    {"n_labeled": 0, "n_skipped": 0, "n_no_tag": 0}
                )
                return

            self.signals.progress.emit(0, f"Auto-labeling {total:,} images...")

            n_labeled = 0
            n_skipped = 0
            n_no_tag = 0
            batch_size = 100

            for batch_start in range(0, total, batch_size):
                if self._canceled:
                    self.signals.error.emit("Canceled by user")
                    return

                batch = self.image_paths[batch_start : batch_start + batch_size]
                results = autolabel_images(batch, self.config, self.threshold)

                updates: Dict[str, tuple] = {}
                for path, result in zip(batch, results):
                    if result.label is not None:
                        updates[str(path)] = (result.label, result.confidence)
                        if result.label == "no_tag":
                            n_no_tag += 1
                        else:
                            n_labeled += 1
                    else:
                        n_skipped += 1

                if updates:
                    self.db.update_labels_with_confidence_batch(updates)

                done = min(batch_start + batch_size, total)
                pct = int(done * 100 / total)
                self.signals.progress.emit(
                    pct,
                    f"Labeled {n_labeled + n_no_tag}, skipped {n_skipped} of {total}",
                )

            self.signals.success.emit(
                {"n_labeled": n_labeled, "n_skipped": n_skipped, "n_no_tag": n_no_tag}
            )

        except Exception as exc:
            traceback.print_exc()
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()
```

- [ ] **Step 2: Verify the existing test suite still passes**

```bash
python -m pytest tests/test_classkit_export_worker.py tests/test_classkit_apriltag_autolabel.py -v
```

Expected: all existing tests PASS, new file tests still PASS

- [ ] **Step 3: Commit**

```bash
git add src/multi_tracker/classkit/jobs/task_workers.py
git commit -m "feat(classkit): add AprilTagAutoLabelWorker background worker"
```

---

## Task 5: `AprilTagAutoLabelDialog`

**Files:**
- Modify: `src/multi_tracker/classkit/gui/dialogs.py`

No unit tests for the dialog (Qt widget test would require a running QApplication — consistent with how all other ClassKit dialogs are handled).

- [ ] **Step 1: Add imports and the dialog class**

Open `src/multi_tracker/classkit/gui/dialogs.py`. After the existing imports block (around line 38), add `QSlider` to the existing PySide6 imports list. The current import is:

```python
from PySide6.QtWidgets import (
    ...
    QWidget,
)
```

Add `QSlider` to that block. Then scroll to the end of the file and append the dialog class:

```python
APRILTAG_FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tag16h5",
    "tagCircle21h7",
    "tagCircle49h12",
    "tagCustom48h12",
    "tagStandard41h12",
    "tagStandard52h13",
]


class AprilTagAutoLabelDialog(QDialog):
    """Configure and launch AprilTag auto-labeling for a ClassKit project.

    On accept, call ``get_config()`` and ``get_threshold()`` to retrieve the
    user's settings. The dialog does NOT run the labeling itself — that is the
    caller's responsibility.
    """

    def __init__(self, image_paths=None, parent=None):
        """
        Args:
            image_paths: Optional list of image paths used for the preview button.
        """
        super().__init__(parent)
        self._image_paths = image_paths or []
        self.setWindowTitle("Auto-label AprilTags")
        self.setMinimumWidth(480)
        self.setStyleSheet(_DARK_STYLE)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ── AprilTag detection parameters ──────────────────────────────
        det_group = QGroupBox("AprilTag Detection Parameters")
        det_form = QFormLayout(det_group)
        det_form.setSpacing(8)

        self._family_combo = QComboBox()
        self._family_combo.addItems(APRILTAG_FAMILIES)
        self._family_combo.setCurrentText("tag36h11")
        self._family_combo.currentTextChanged.connect(self._update_scheme_preview)
        det_form.addRow("Tag family:", self._family_combo)

        self._max_tag_id_spin = QSpinBox()
        self._max_tag_id_spin.setRange(0, 999)
        self._max_tag_id_spin.setValue(9)
        self._max_tag_id_spin.valueChanged.connect(self._update_scheme_preview)
        det_form.addRow("Max tag ID:", self._max_tag_id_spin)

        self._max_hamming_spin = QSpinBox()
        self._max_hamming_spin.setRange(0, 3)
        self._max_hamming_spin.setValue(1)
        det_form.addRow("Max hamming:", self._max_hamming_spin)

        self._decimate_spin = QDoubleSpinBox()
        self._decimate_spin.setRange(1.0, 8.0)
        self._decimate_spin.setSingleStep(0.5)
        self._decimate_spin.setValue(2.0)
        det_form.addRow("Decimate:", self._decimate_spin)

        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(0.0, 5.0)
        self._blur_spin.setSingleStep(0.1)
        self._blur_spin.setValue(0.8)
        det_form.addRow("Blur:", self._blur_spin)

        self._contrast_spin = QDoubleSpinBox()
        self._contrast_spin.setRange(0.5, 5.0)
        self._contrast_spin.setSingleStep(0.1)
        self._contrast_spin.setValue(1.5)
        det_form.addRow("Contrast factor (unsharp_strong profile):", self._contrast_spin)

        self._unsharp_amount_spin = QDoubleSpinBox()
        self._unsharp_amount_spin.setRange(0.0, 10.0)
        self._unsharp_amount_spin.setSingleStep(0.1)
        self._unsharp_amount_spin.setValue(1.0)
        det_form.addRow("Unsharp amount:", self._unsharp_amount_spin)

        self._unsharp_sigma_spin = QDoubleSpinBox()
        self._unsharp_sigma_spin.setRange(0.1, 10.0)
        self._unsharp_sigma_spin.setSingleStep(0.1)
        self._unsharp_sigma_spin.setValue(1.0)
        det_form.addRow("Unsharp sigma:", self._unsharp_sigma_spin)

        self._unsharp_kernel_spin = QSpinBox()
        self._unsharp_kernel_spin.setRange(1, 31)
        self._unsharp_kernel_spin.setSingleStep(2)  # keep odd
        self._unsharp_kernel_spin.setValue(5)
        det_form.addRow("Unsharp kernel size (n → (n,n)):", self._unsharp_kernel_spin)

        root.addWidget(det_group)

        # ── Labeling parameters ────────────────────────────────────────
        label_group = QGroupBox("Labeling Parameters")
        label_form = QFormLayout(label_group)
        label_form.setSpacing(8)

        thresh_row = QHBoxLayout()
        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(0, 100)
        self._thresh_slider.setValue(60)
        self._thresh_label = QLabel("0.60")
        self._thresh_slider.valueChanged.connect(
            lambda v: self._thresh_label.setText(f"{v / 100:.2f}")
        )
        thresh_row.addWidget(self._thresh_slider)
        thresh_row.addWidget(self._thresh_label)
        label_form.addRow("Confidence threshold:", thresh_row)

        self._preview_btn = QPushButton("Preview (10 images)")
        self._preview_btn.setEnabled(bool(self._image_paths))
        self._preview_btn.clicked.connect(self._run_preview)
        label_form.addRow(self._preview_btn)

        self._preview_result_label = QLabel("")
        self._preview_result_label.setWordWrap(True)
        label_form.addRow(self._preview_result_label)

        root.addWidget(label_group)

        # ── Scheme preview ─────────────────────────────────────────────
        scheme_group = QGroupBox("Labeling Scheme Preview")
        scheme_layout = QVBoxLayout(scheme_group)
        self._scheme_preview = QLabel()
        self._scheme_preview.setWordWrap(True)
        scheme_layout.addWidget(self._scheme_preview)
        root.addWidget(scheme_group)

        self._update_scheme_preview()

        # ── Buttons ────────────────────────────────────────────────────
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _update_scheme_preview(self):
        family = self._family_combo.currentText()
        max_id = self._max_tag_id_spin.value()
        labels = [f"tag_{i}" for i in range(max_id + 1)] + ["no_tag"]
        self._scheme_preview.setText(
            f"<b>Scheme:</b> apriltag_{family}<br>"
            f"<b>Labels ({len(labels)}):</b> "
            + ", ".join(labels[:12])
            + ("…" if len(labels) > 12 else "")
        )

    def _run_preview(self):
        import random

        sample = random.sample(
            self._image_paths, min(10, len(self._image_paths))
        )
        from ..autolabel.apriltag import autolabel_images

        results = autolabel_images(sample, self.get_config(), self.get_threshold())
        n_tagged = sum(1 for r in results if r.label and r.label != "no_tag")
        n_no_tag = sum(1 for r in results if r.label == "no_tag")
        n_skip = sum(1 for r in results if r.label is None)
        self._preview_result_label.setText(
            f"Preview ({len(results)} images): "
            f"{n_tagged} tagged, {n_no_tag} no_tag, {n_skip} uncertain/skipped"
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_config(self):
        """Return an AprilTagConfig built from the dialog's current settings."""
        from ...core.identity.apriltag_detector import AprilTagConfig

        n = self._unsharp_kernel_spin.value()
        return AprilTagConfig(
            family=self._family_combo.currentText(),
            max_hamming=self._max_hamming_spin.value(),
            decimate=self._decimate_spin.value(),
            blur=self._blur_spin.value(),
            contrast_factor=self._contrast_spin.value(),
            unsharp_amount=self._unsharp_amount_spin.value(),
            unsharp_sigma=self._unsharp_sigma_spin.value(),
            unsharp_kernel_size=(n, n),
            max_tag_id=self._max_tag_id_spin.value(),
        )

    def get_threshold(self) -> float:
        """Return the confidence threshold (0.0–1.0)."""
        return self._thresh_slider.value() / 100.0
```

- [ ] **Step 2: Add `QSlider` to the imports block**

Find the PySide6 imports near the top of `dialogs.py` (around line 12) and add `QSlider` to the `QWidget` line region. The existing block ends with `QWidget,`. Add `QSlider` before `QWidget`:

```python
    QSlider,
    QWidget,
```

- [ ] **Step 3: Verify the file is importable**

```bash
python -c "from multi_tracker.classkit.gui.dialogs import AprilTagAutoLabelDialog; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/multi_tracker/classkit/gui/dialogs.py
git commit -m "feat(classkit): add AprilTagAutoLabelDialog"
```

---

## Task 6: Main window integration

**Files:**
- Modify: `src/multi_tracker/classkit/gui/mainwindow.py`

- [ ] **Step 1: Find where to add the menu action**

Search for the Edit menu or toolbar action group in `mainwindow.py`:

```bash
grep -n "menubar\|addAction\|QMenu\|toolbar" "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/classkit/gui/mainwindow.py" | head -30
```

Note the line numbers of the menu/toolbar setup.

- [ ] **Step 2: Add the menu action**

Find where the Edit menu or the main menu bar is constructed in `mainwindow.py`. Add the new action there. Example — if the file has a section like:

```python
edit_menu = menubar.addMenu("Edit")
edit_menu.addAction(some_action)
```

Add after the last `edit_menu.addAction(...)`:

```python
self._autolabel_apriltag_action = edit_menu.addAction("Auto-label AprilTags…")
self._autolabel_apriltag_action.setEnabled(False)
self._autolabel_apriltag_action.triggered.connect(self._run_apriltag_autolabel)
```

Also find where the project is loaded/unloaded (the place that calls `self.project_path = ...`) and add a line to enable/disable the action:

```python
self._autolabel_apriltag_action.setEnabled(self.project_path is not None)
```

This is likely near the existing `self._update_labeling_progress_indicator()` calls.

- [ ] **Step 3: Add the `_run_apriltag_autolabel` method**

Find the method group near the end of `mainwindow.py` (look for other `def _run_*` or `def open_*` methods). Add:

```python
def _run_apriltag_autolabel(self) -> None:
    """Open the AprilTag auto-label dialog and start the background worker."""
    import json
    from pathlib import Path

    from ..autolabel.apriltag import autolabel_images  # noqa: F401 (verify importable)
    from ..gui.dialogs import AprilTagAutoLabelDialog
    from ..jobs.task_workers import AprilTagAutoLabelWorker
    from ..presets import apriltag_preset

    if self.project_path is None:
        return

    # Collect all image paths (dialog needs them for preview)
    all_paths = [Path(p) for p in self.db.get_all_image_paths()]

    dlg = AprilTagAutoLabelDialog(image_paths=all_paths, parent=self)
    if dlg.exec() != AprilTagAutoLabelDialog.DialogCode.Accepted:
        return

    config = dlg.get_config()
    threshold = dlg.get_threshold()
    max_tag_id = config.max_tag_id

    # Warn if existing scheme will be replaced
    scheme_path = self.project_path / "scheme.json"
    if scheme_path.exists():
        from PySide6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Replace Existing Scheme?",
            "A labeling scheme already exists.\n\n"
            "Replacing it will ERASE all existing labels.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

    # 1. Write the new scheme to disk
    scheme = apriltag_preset(config.family, max_tag_id)
    with open(scheme_path, "w") as f:
        json.dump(scheme.to_dict(), f, indent=2)

    # 2. Clear all existing labels (prevents stale labels under new scheme)
    self.db.clear_all_labels()

    # 3. Update self.classes and rebuild label buttons
    self.classes = scheme.factors[0].labels
    self.rebuild_label_buttons()  # public method — no underscore

    # 4. Collect unlabeled image paths
    labels = self.db.get_all_labels()
    unlabeled = [p for p, lbl in zip(all_paths, labels) if lbl is None]

    if not unlabeled:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Auto-label", "No unlabeled images to process.")
        return

    # 5. Start worker
    worker = AprilTagAutoLabelWorker(
        image_paths=unlabeled,
        config=config,
        threshold=threshold,
        db=self.db,
    )

    def _on_progress(pct: int, msg: str) -> None:
        self.statusBar().showMessage(f"AprilTag auto-label: {msg} ({pct}%)")

    def _on_success(result: dict) -> None:
        n_tag = result.get("n_labeled", 0)
        n_no = result.get("n_no_tag", 0)
        n_skip = result.get("n_skipped", 0)
        self.statusBar().showMessage(
            f"Auto-label complete: {n_tag} tagged, {n_no} no_tag, {n_skip} uncertain"
        )
        self._update_labeling_progress_indicator()

    def _on_error(msg: str) -> None:
        self.statusBar().showMessage(f"Auto-label error: {msg}")

    worker.signals.progress.connect(_on_progress)
    worker.signals.success.connect(_on_success)
    worker.signals.error.connect(_on_error)
    self._threadpool_start(worker)
```

- [ ] **Step 4: Confirm `rebuild_label_buttons` exists**

The code in Step 3 calls `self.rebuild_label_buttons()`. Verify this public method exists at line 2041 of `mainwindow.py`:

```bash
grep -n "def rebuild_label_buttons" "/Users/neurorishika/Projects/MBL/Module 4/multi-animal-tracker/src/multi_tracker/classkit/gui/mainwindow.py"
```

Expected: `2041:    def rebuild_label_buttons(self):`

No action needed — the method name in Step 3 is already correct.

- [ ] **Step 5: Verify the app starts without errors**

```bash
python -c "from multi_tracker.classkit.app import main; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py tests/test_classkit_scheme.py tests/test_classkit_db_model_cache.py -v
```

Expected: all PASSED

- [ ] **Step 7: Commit**

```bash
git add src/multi_tracker/classkit/gui/mainwindow.py
git commit -m "feat(classkit): wire Auto-label AprilTags menu action in mainwindow"
```

---

## Task 7: Final integration check

- [ ] **Step 1: Run full test suite including apriltag tests**

```bash
python -m pytest tests/test_classkit_apriltag_autolabel.py -v
```

Expected: all tests PASS

- [ ] **Step 2: Run broader classkit test suite to confirm no regressions**

```bash
python -m pytest tests/test_classkit_scheme.py tests/test_classkit_db_model_cache.py tests/test_classkit_export_worker.py tests/test_classkit_tiny_train.py -v
```

Expected: all PASSED

- [ ] **Step 3: Final commit**

```bash
git add -A
git status  # verify only expected files are staged
git commit -m "feat(classkit): AprilTag auto-labeler — full integration"
```
