from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from multi_tracker.training.contracts import (
    SourceDataset,
    SplitConfig,
    TrainingHyperParams,
    TrainingRole,
    TrainingRunSpec,
)
from multi_tracker.training.dataset_builders import (
    derive_crop_obb_dataset_from_obb,
    derive_detect_dataset_from_obb,
    merge_obb_sources,
)
from multi_tracker.training.dataset_inspector import inspect_obb_or_detect_dataset
from multi_tracker.training.model_publish import publish_trained_model
from multi_tracker.training.registry import (
    create_run_record,
    dataset_fingerprint,
    finalize_run_record,
    load_registry,
    new_run_id,
)
from multi_tracker.training.runner import build_ultralytics_command


def _write_image(path: Path, value: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((80, 120, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _write_obb_label(path: Path, cls_id: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    # normalized square OBB
    path.write_text(f"{cls_id} 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n", encoding="utf-8")


def _make_unsplit_source(root: Path, name: str, img_values: list[int]):
    src = root / name
    for i, val in enumerate(img_values):
        _write_image(src / "images" / f"img_{i}.jpg", value=val)
        _write_obb_label(src / "labels" / f"img_{i}.txt")
    return src


def _make_split_obb_dataset(root: Path):
    ds = root / "obb_ds"
    for split, value in (("train", 100), ("val", 140), ("test", 180)):
        _write_image(ds / "images" / split / f"{split}_0.jpg", value=value)
        _write_obb_label(ds / "labels" / split / f"{split}_0.txt")
    (ds / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {ds}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: ant",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return ds


def test_inspector_supports_yaml_and_txt_list(tmp_path: Path):
    root = tmp_path / "list_ds"
    _write_image(root / "images" / "train" / "a.jpg", value=111)
    _write_obb_label(root / "labels" / "train" / "a.txt")
    (root / "train.txt").write_text("images/train/a.jpg\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train.txt",
                "val: train.txt",
                "names:",
                "  0: ant",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    inspection = inspect_obb_or_detect_dataset(root)
    assert "train" in inspection.splits
    assert len(inspection.splits["train"]) == 1
    item = inspection.splits["train"][0]
    assert Path(item.label_path).name == "a.txt"


def test_inspector_txt_list_honors_custom_labels_root(tmp_path: Path):
    root = tmp_path / "list_custom_labels"
    _write_image(root / "images" / "train" / "a.jpg", value=111)
    _write_obb_label(root / "ann" / "train" / "a.txt")
    (root / "train.txt").write_text("images/train/a.jpg\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train.txt",
                "val: train.txt",
                "labels: ann",
                "names:",
                "  0: ant",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    inspection = inspect_obb_or_detect_dataset(root)
    item = inspection.splits["train"][0]
    assert (
        Path(item.label_path).resolve() == (root / "ann" / "train" / "a.txt").resolve()
    )


def test_merge_obb_sources_dedup_and_counts(tmp_path: Path):
    src1 = _make_unsplit_source(tmp_path, "src1", [50, 60])
    src2 = _make_unsplit_source(tmp_path, "src2", [60, 200])

    res = merge_obb_sources(
        sources=[
            SourceDataset(path=str(src1), name="src1"),
            SourceDataset(path=str(src2), name="src2"),
        ],
        output_root=tmp_path / "out",
        class_name="ant",
        split_cfg=SplitConfig(train=0.8, val=0.2, test=0.0),
        seed=7,
        dedup=True,
    )
    assert Path(res.dataset_dir).exists()
    assert res.stats["counts"]["train"] > 0
    assert res.stats["counts"]["val"] > 0
    assert res.stats["duplicates_skipped"] >= 1


def test_merge_preserves_test_split_without_dropping(tmp_path: Path):
    src = _make_split_obb_dataset(tmp_path)
    res = merge_obb_sources(
        sources=[SourceDataset(path=str(src), name="src")],
        output_root=tmp_path / "out",
        class_name="ant",
        split_cfg=SplitConfig(train=0.8, val=0.2, test=0.0),
        seed=3,
        dedup=False,
    )
    assert res.stats["counts"]["train"] == 1
    assert res.stats["counts"]["val"] == 1
    assert res.stats["counts"]["test"] == 1
    assert list((Path(res.dataset_dir) / "labels" / "test").glob("*.txt"))
    yaml_text = (Path(res.dataset_dir) / "dataset.yaml").read_text(encoding="utf-8")
    assert "test: images/test" in yaml_text


def test_derive_detect_and_crop_from_obb(tmp_path: Path):
    src = _make_split_obb_dataset(tmp_path)

    det = derive_detect_dataset_from_obb(src, tmp_path / "derived", class_name="ant")
    det_train_labels = list((Path(det.dataset_dir) / "labels" / "train").glob("*.txt"))
    assert det_train_labels
    det_test_labels = list((Path(det.dataset_dir) / "labels" / "test").glob("*.txt"))
    assert det_test_labels
    line = det_train_labels[0].read_text(encoding="utf-8").strip().split()
    assert len(line) == 5

    crop = derive_crop_obb_dataset_from_obb(
        src,
        tmp_path / "derived",
        class_name="ant",
        pad_ratio=0.15,
        min_crop_size_px=32,
        enforce_square=True,
    )
    crop_train_labels = list(
        (Path(crop.dataset_dir) / "labels" / "train").glob("*.txt")
    )
    assert crop_train_labels
    crop_test_labels = list((Path(crop.dataset_dir) / "labels" / "test").glob("*.txt"))
    assert crop_test_labels
    vals = crop_train_labels[0].read_text(encoding="utf-8").strip().split()
    assert len(vals) == 9
    coords = [float(v) for v in vals[1:]]
    assert all(0.0 <= c <= 1.0 for c in coords)


def test_runner_command_per_role(tmp_path: Path):
    spec = TrainingRunSpec(
        role=TrainingRole.SEQ_DETECT,
        source_datasets=[SourceDataset(path="/tmp/src")],
        derived_dataset_dir=str(tmp_path / "ds"),
        base_model="yolo26s.pt",
        hyperparams=TrainingHyperParams(
            epochs=10, imgsz=640, batch=8, lr0=0.01, patience=5, workers=2
        ),
        device="cpu",
    )
    cmd = build_ultralytics_command(spec, tmp_path / "runs" / "abc")
    joined = " ".join(cmd)
    assert "detect" in joined
    assert "train" in joined
    assert "data=" in joined


def test_runner_fallback_uses_ultralytics_module(tmp_path: Path, monkeypatch):
    import multi_tracker.training.runner as runner

    monkeypatch.setattr(runner.shutil, "which", lambda _cmd: None)
    spec = TrainingRunSpec(
        role=TrainingRole.SEQ_DETECT,
        source_datasets=[SourceDataset(path="/tmp/src")],
        derived_dataset_dir=str(tmp_path / "ds"),
        base_model="yolo26s.pt",
        hyperparams=TrainingHyperParams(),
        device="cpu",
    )
    cmd = runner.build_ultralytics_command(spec, tmp_path / "runs" / "abc")
    assert cmd[:3] == [sys.executable, "-m", "ultralytics"]


def test_registry_and_publish_lineage(tmp_path: Path, monkeypatch):
    import multi_tracker.training.model_publish as pub
    import multi_tracker.training.registry as reg

    monkeypatch.setattr(reg, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(pub, "_project_root", lambda: tmp_path)

    run_id = new_run_id("obb_direct")
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset.yaml").write_text(
        "train: images/train\nval: images/val\n", encoding="utf-8"
    )
    fp = dataset_fingerprint(ds_dir)

    spec = TrainingRunSpec(
        role=TrainingRole.OBB_DIRECT,
        source_datasets=[SourceDataset(path=str(ds_dir))],
        derived_dataset_dir=str(ds_dir),
        base_model="yolo26s-obb.pt",
        hyperparams=TrainingHyperParams(),
        device="cpu",
    )
    create_run_record(
        spec,
        run_id=run_id,
        run_dir=tmp_path / "training" / "runs" / run_id,
        dataset_fp=fp,
    )

    artifact = tmp_path / "best.pt"
    artifact.write_text("dummy", encoding="utf-8")
    key, dst = publish_trained_model(
        role=TrainingRole.OBB_DIRECT,
        artifact_path=str(artifact),
        size="26s",
        species="ant",
        model_info="new",
        trained_from_run_id=run_id,
        dataset_fingerprint=fp,
        base_model="yolo26s-obb.pt",
    )
    assert key
    assert Path(dst).exists()

    finalize_run_record(
        run_id,
        status="completed",
        command=["yolo", "obb", "train"],
        artifact_paths=[dst],
        published_model_path=dst,
        published_registry_entry=key,
    )

    reg_data = load_registry()
    assert reg_data["runs"]
    rec = reg_data["runs"][0]
    assert rec["status"] == "completed"
    assert rec["published_registry_entry"] == key

    model_reg = json.loads(
        (tmp_path / "models" / "model_registry.json").read_text(encoding="utf-8")
    )
    assert key in model_reg
    assert model_reg[key]["trained_from_run_id"] == run_id
