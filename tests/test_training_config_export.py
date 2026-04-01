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
