"""Dataset inspection and layout discovery for MAT training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(slots=True)
class DatasetItem:
    """Single image/label pair in a split."""

    image_path: str
    label_path: str
    split: str


@dataclass(slots=True)
class DatasetInspection:
    """Inspection result for OBB/detect-style datasets."""

    root_dir: str
    splits: dict[str, list[DatasetItem]] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ClassifyInspection:
    """Inspection result for classify-style datasets."""

    root_dir: str
    splits: dict[str, dict[str, list[str]]] = field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep branch
        raise RuntimeError("PyYAML is required to parse dataset.yaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid dataset.yaml structure: {path}")
    return data


def _resolve_data_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    p = Path(str(value).strip())
    if not p:
        return None
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _find_label_for_image(image_path: Path, labels_root: Path) -> Path:
    # Preferred: preserve relative path from images_root into labels_root.
    rel = None
    if "images" in image_path.parts:
        idx = image_path.parts.index("images")
        rel = Path(*image_path.parts[idx + 1 :])
    if rel is not None and rel.parts:
        cand = (labels_root / rel).with_suffix(".txt")
        if cand.exists():
            return cand

    # Fallback: same basename under labels root (recursive search).
    stem = image_path.stem
    matches = list(labels_root.rglob(f"{stem}.txt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Deterministic tie-break
        return sorted(matches)[0]

    # Final fallback: sibling txt
    return image_path.with_suffix(".txt")


def _collect_dir_split(
    images_dir: Path, labels_dir: Path, split: str
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    if not images_dir.exists():
        return items
    for image_path in sorted(images_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = _find_label_for_image(image_path, labels_dir)
        items.append(
            DatasetItem(
                image_path=str(image_path.resolve()),
                label_path=str(label_path.resolve()),
                split=split,
            )
        )
    return items


def _infer_label_path_from_image(
    root: Path, image_path: Path, labels_root: Path | None = None
) -> Path:
    labels_root = (labels_root or (root / "labels")).resolve()
    image_posix = image_path.as_posix()
    if "/images/" in image_posix:
        image_parts = image_path.parts
        idx = image_parts.index("images")
        rel = Path(*image_parts[idx + 1 :])
        return (labels_root / rel).with_suffix(".txt")
    # Fallback to sibling labels folder
    return (labels_root / image_path.name).with_suffix(".txt")


def _collect_list_split(
    root: Path, list_path: Path, split: str, labels_root: Path | None = None
) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    if not list_path.exists():
        return items
    lines = [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines()]
    for ln in lines:
        if not ln:
            continue
        p = Path(ln)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = _infer_label_path_from_image(root, p, labels_root=labels_root)
        if not lbl.is_absolute():
            lbl = (root / lbl).resolve()
        items.append(DatasetItem(image_path=str(p), label_path=str(lbl), split=split))
    return items


def _extract_class_names(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names", {})
    out: dict[int, str] = {}
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
    elif isinstance(names, list):
        for i, v in enumerate(names):
            out[i] = str(v)
    return out


def inspect_obb_or_detect_dataset(root_dir: str | Path) -> DatasetInspection:
    """Inspect a YOLO OBB/detect dataset and return resolved split items."""

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Dataset root not found: {root}")

    inspection = DatasetInspection(root_dir=str(root))
    yaml_path = root / "dataset.yaml"

    if yaml_path.exists():
        data = _read_yaml(yaml_path)
        inspection.class_names = _extract_class_names(data)

        for split in ("train", "val", "test"):
            split_ref = data.get(split)
            if split_ref is None:
                continue
            split_path = _resolve_data_path(root, split_ref)
            if split_path is None:
                continue
            labels_dir = _resolve_data_path(root, data.get("labels"))
            if labels_dir is None:
                labels_dir = root / "labels"

            if split_path.suffix.lower() == ".txt":
                items = _collect_list_split(
                    root, split_path, split=split, labels_root=labels_dir
                )
            else:
                # If split_path is images/<split>, labels should be labels/<split>
                if split_path.is_dir() and split_path.name in {"train", "val", "test"}:
                    split_labels = labels_dir / split_path.name
                    if split_labels.exists():
                        labels_dir = split_labels
                items = _collect_dir_split(split_path, labels_dir, split=split)
            if items:
                inspection.splits[split] = items

        if inspection.splits:
            inspection.metadata["source"] = "dataset.yaml"
            return inspection

    # Standard directory layouts fallback
    images_root = root / "images"
    labels_root = root / "labels"
    if images_root.exists() and labels_root.exists():
        split_items: dict[str, list[DatasetItem]] = {}
        split_found = False
        for split in ("train", "val", "test"):
            img_dir = images_root / split
            lbl_dir = (
                labels_root / split if (labels_root / split).exists() else labels_root
            )
            if img_dir.exists():
                split_items[split] = _collect_dir_split(img_dir, lbl_dir, split)
                split_found = True
        if split_found:
            inspection.splits = split_items
            inspection.metadata["source"] = "images/labels split"
            return inspection

        # Unsplit dataset (images + labels roots)
        inspection.splits = {
            "all": _collect_dir_split(images_root, labels_root, split="all"),
        }
        inspection.metadata["source"] = "images/labels unsplit"
        return inspection

    raise RuntimeError(f"No valid OBB/detect dataset layout found in {root}")


def inspect_classify_dataset(root_dir: str | Path) -> ClassifyInspection:
    """Inspect a classify dataset with split/class/image directory layout."""

    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise RuntimeError(f"Classify dataset root not found: {root}")

    splits: dict[str, dict[str, list[str]]] = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        classes: dict[str, list[str]] = {}
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            images = [
                str(p.resolve())
                for p in sorted(class_dir.rglob("*"))
                if p.suffix.lower() in IMAGE_EXTS
            ]
            if images:
                classes[class_dir.name] = images
        if classes:
            splits[split] = classes

    if not splits:
        raise RuntimeError(f"No valid classify split structure found in {root}")

    return ClassifyInspection(root_dir=str(root), splits=splits)


def split_items_for_training(
    inspection: DatasetInspection, split_cfg: tuple[float, float, float], seed: int
) -> dict[str, list[DatasetItem]]:
    """Normalize to train/val/test using provided ratios when source is unsplit."""

    import random

    if "all" not in inspection.splits:
        out = {
            "train": list(inspection.splits.get("train", [])),
            "val": list(inspection.splits.get("val", [])),
            "test": list(inspection.splits.get("test", [])),
        }
        return out

    items = list(inspection.splits.get("all", []))
    rng = random.Random(int(seed))
    rng.shuffle(items)

    train_r, val_r, test_r = split_cfg
    total = max(1e-8, float(train_r) + float(val_r) + float(test_r))
    train_r, val_r, test_r = train_r / total, val_r / total, test_r / total

    n = len(items)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))

    # Guardrails for non-empty train/val when feasible
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
        n_val = max(1, min(n - n_train, n_val))

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]

    for it in train:
        it.split = "train"
    for it in val:
        it.split = "val"
    for it in test:
        it.split = "test"

    return {"train": train, "val": val, "test": test}
