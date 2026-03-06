# tests/test_classkit_tiny_train.py
from pathlib import Path

import pytest


def _make_dummy_dataset(root: Path, classes: list[str], n_per_class: int = 4):
    """Create minimal image-folder dataset with tiny solid-color images."""
    import numpy as np

    cv2 = pytest.importorskip("cv2")

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


def test_iter_classify_samples_empty_split(tmp_path):
    from multi_tracker.training.runner import _iter_classify_samples

    # Non-existent split should yield nothing, not raise
    samples = list(_iter_classify_samples(tmp_path, "train"))
    assert samples == []


def test_train_tiny_classify_runs(tmp_path):
    """Smoke test: tiny N-class CNN trains without error on minimal data."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from multi_tracker.training.contracts import (
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
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
    assert result["task"] == "tiny_classify"


def test_train_tiny_classify_multihead_role(tmp_path):
    """CLASSIFY_MULTIHEAD_TINY uses same in-process trainer."""
    pytest.importorskip("torch")
    pytest.importorskip("cv2")

    from multi_tracker.training.contracts import (
        TinyHeadTailParams,
        TrainingHyperParams,
        TrainingRole,
        TrainingRunSpec,
    )
    from multi_tracker.training.runner import run_training

    _make_dummy_dataset(tmp_path / "dataset", ["x", "y"], n_per_class=4)

    spec = TrainingRunSpec(
        role=TrainingRole.CLASSIFY_MULTIHEAD_TINY,
        source_datasets=[],
        derived_dataset_dir=str(tmp_path / "dataset"),
        base_model="",
        hyperparams=TrainingHyperParams(),
        tiny_params=TinyHeadTailParams(epochs=2, batch=4),
    )

    result = run_training(spec, tmp_path / "run")
    assert result["success"] is True
