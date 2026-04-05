from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from tests.helpers.module_loader import SRC_ROOT


def _ns_pkg(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _exec_src_module(relative_path: str, module_name: str, stubs=None):
    module_path = SRC_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sentinel = object()
    originals = {}
    stubs = stubs or {}

    try:
        for name, stub in stubs.items():
            originals[name] = sys.modules.get(name, sentinel)
            sys.modules[name] = stub
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in originals.items():
            if old is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _load_validation_modules():
    contracts = _exec_src_module(
        "hydra_suite/training/contracts.py",
        "hydra_suite.training.contracts",
    )
    dataset_inspector = _exec_src_module(
        "hydra_suite/training/dataset_inspector.py",
        "hydra_suite.training.dataset_inspector",
    )
    stubs = {
        "hydra_suite": _ns_pkg("hydra_suite"),
        "hydra_suite.training": _ns_pkg("hydra_suite.training"),
        "hydra_suite.training.contracts": contracts,
        "hydra_suite.training.dataset_inspector": dataset_inspector,
    }
    validation = _exec_src_module(
        "hydra_suite/training/validation.py",
        "hydra_suite.training.validation_under_test",
        stubs=stubs,
    )
    return validation, contracts, dataset_inspector


def _write_label(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(line + "\n", encoding="utf-8")


def test_validate_obb_dataset_reports_split_counts_and_missing_labels(
    tmp_path: Path,
) -> None:
    validation, _contracts, inspector = _load_validation_modules()
    image_path = tmp_path / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"jpg")

    inspection = inspector.DatasetInspection(
        root_dir=str(tmp_path),
        splits={
            "train": [],
            "val": [],
            "test": [
                inspector.DatasetItem(
                    image_path=str(image_path),
                    label_path=str(tmp_path / "labels" / "sample.txt"),
                    split="test",
                )
            ],
        },
    )

    report = validation.validate_obb_dataset(inspection)
    codes = {issue.code for issue in report.issues}

    assert report.valid is False
    assert {"empty_train", "empty_val", "missing_label"}.issubset(codes)
    assert report.stats["missing_labels"] == 1


def test_validate_obb_dataset_reports_range_and_multi_class_errors(
    tmp_path: Path,
) -> None:
    validation, _contracts, inspector = _load_validation_modules()

    train_img = tmp_path / "images" / "train.jpg"
    val_img = tmp_path / "images" / "val.jpg"
    train_img.parent.mkdir(parents=True, exist_ok=True)
    train_img.write_bytes(b"train")
    val_img.write_bytes(b"val")

    train_lbl = tmp_path / "labels" / "train.txt"
    val_lbl = tmp_path / "labels" / "val.txt"
    _write_label(train_lbl, "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2")
    _write_label(val_lbl, "1 1.5 0.1 0.2 0.1 0.2 0.2 0.1 0.2")

    inspection = inspector.DatasetInspection(
        root_dir=str(tmp_path),
        splits={
            "train": [
                inspector.DatasetItem(
                    image_path=str(train_img),
                    label_path=str(train_lbl),
                    split="train",
                )
            ],
            "val": [
                inspector.DatasetItem(
                    image_path=str(val_img),
                    label_path=str(val_lbl),
                    split="val",
                )
            ],
        },
    )

    report = validation.validate_obb_dataset(inspection)
    codes = {issue.code for issue in report.issues}

    assert report.valid is False
    assert "coord_out_of_range" in codes
    assert "multi_class_source" in codes
    assert report.stats["class_ids"] == [0, 1]


def test_format_validation_report_includes_paths_and_status() -> None:
    validation, contracts, _inspector = _load_validation_modules()
    report = contracts.ValidationReport(
        valid=False,
        issues=[
            contracts.ValidationIssue(
                severity="error",
                code="missing_label",
                message="Missing label for split 'train'.",
                path="/tmp/example.txt",
            )
        ],
        stats={"missing_labels": 1},
    )

    text = validation.format_validation_report(report)

    assert "Validation: FAIL" in text
    assert "missing_label" in text
    assert "/tmp/example.txt" in text
