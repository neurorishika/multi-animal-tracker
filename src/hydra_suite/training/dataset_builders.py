"""Dataset builders for MAT multi-role training."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .contracts import DatasetBuildResult, SourceDataset, SplitConfig, TrainingRole
from .dataset_inspector import (
    DatasetInspection,
    inspect_obb_or_detect_dataset,
    split_items_for_training,
    stratified_split_items,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_split_cfg(cfg: SplitConfig) -> tuple[float, float, float]:
    train = float(cfg.train)
    val = float(cfg.val)
    test = float(cfg.test)
    if train < 0 or val < 0 or test < 0:
        raise RuntimeError("Split ratios must be non-negative.")
    total = train + val + test
    if total <= 0:
        raise RuntimeError("Split ratios sum to zero.")
    return train / total, val / total, test / total


def _write_dataset_yaml(
    root_dir: Path, class_name: str, include_test: bool = False
) -> Path:
    data = {
        "path": str(root_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {0: class_name},
    }
    if include_test:
        data["test"] = "images/test"

    out = root_dir / "dataset.yaml"
    try:
        import yaml  # type: ignore

        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except Exception:
        lines = [
            f"path: {root_dir.resolve()}",
            "train: images/train",
            "val: images/val",
        ]
        if include_test:
            lines.append("test: images/test")
        lines += ["names:", f"  0: {class_name}"]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _write_manifest(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_name(text: str) -> str:
    return (
        "".join(
            ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text or "")
        ).strip("_")
        or "src"
    )


def _parse_obb_label_lines(lbl_path: Path) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    for ln in lbl_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) != 9:
            raise RuntimeError(f"Invalid OBB label format in {lbl_path}: {ln}")
        cls_id = int(float(parts[0]))
        vals = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(4, 2)
        out.append((cls_id, vals))
    return out


def merge_obb_sources(
    sources: list[SourceDataset],
    output_root: str | Path,
    class_name: str,
    split_cfg: SplitConfig,
    seed: int = 42,
    dedup: bool = True,
    remap_single_class: bool = True,
) -> DatasetBuildResult:
    """Merge OBB sources into one canonical dataset (non-destructive)."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / f"combined_obb_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_tuple = _normalize_split_cfg(split_cfg)

    rng = random.Random(int(seed))
    seen_hashes: set[str] = set()
    split_counts = {"train": 0, "val": 0, "test": 0}
    source_items: dict[str, int] = defaultdict(int)
    duplicate_skipped = 0

    for src in sources:
        src_name = _safe_name(src.name or Path(src.path).name)
        inspection: DatasetInspection = inspect_obb_or_detect_dataset(src.path)
        # Use stratified splitting to ensure proportional class representation
        if "all" in inspection.splits:
            split_items = stratified_split_items(
                list(inspection.splits["all"]), split_tuple, seed=seed
            )
        else:
            split_items = split_items_for_training(inspection, split_tuple, seed=seed)

        # If source already has train/val we keep their order deterministic but shuffled per-source.
        for split in ("train", "val", "test"):
            items = list(split_items.get(split, []))
            rng.shuffle(items)
            for idx, item in enumerate(items):
                img = Path(item.image_path)
                lbl = Path(item.label_path)
                if dedup:
                    file_hash = _hash_file(img)
                    if file_hash in seen_hashes:
                        duplicate_skipped += 1
                        continue
                    seen_hashes.add(file_hash)

                stem = f"{src_name}_{img.stem}_{idx:06d}"
                img_dst = out_dir / "images" / split / f"{stem}{img.suffix.lower()}"
                lbl_dst = out_dir / "labels" / split / f"{stem}.txt"

                shutil.copy2(img, img_dst)
                if remap_single_class:
                    lines = []
                    for cls_id, poly in _parse_obb_label_lines(lbl):
                        _ = cls_id
                        coords = " ".join(f"{float(v):.6f}" for v in poly.reshape(-1))
                        lines.append(f"0 {coords}")
                    lbl_dst.write_text(
                        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                    )
                else:
                    shutil.copy2(lbl, lbl_dst)

                split_counts[split] += 1
                source_items[src_name] += 1

    if split_counts["train"] <= 0 or split_counts["val"] <= 0:
        raise RuntimeError(
            "Merged dataset has empty split(s). Ensure source datasets provide enough labeled frames."
        )

    include_test = split_counts["test"] > 0
    _write_dataset_yaml(out_dir, class_name=class_name, include_test=include_test)
    manifest = {
        "type": "merged_obb",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [asdict(s) for s in sources],
        "split_cfg": asdict(split_cfg),
        "seed": int(seed),
        "dedup": bool(dedup),
        "remap_single_class": bool(remap_single_class),
        "counts": split_counts,
        "source_items": dict(source_items),
        "duplicates_skipped": int(duplicate_skipped),
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    return DatasetBuildResult(
        dataset_dir=str(out_dir),
        stats=manifest,
        manifest_path=str(manifest_path),
    )


def _convert_obb_to_aabb(poly: np.ndarray) -> tuple[float, float, float, float]:
    xs = poly[:, 0]
    ys = poly[:, 1]
    x1 = float(np.clip(np.min(xs), 0.0, 1.0))
    y1 = float(np.clip(np.min(ys), 0.0, 1.0))
    x2 = float(np.clip(np.max(xs), 0.0, 1.0))
    y2 = float(np.clip(np.max(ys), 0.0, 1.0))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    return cx, cy, w, h


def derive_detect_dataset_from_obb(
    obb_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str,
) -> DatasetBuildResult:
    """Create YOLO-detect dataset by converting OBB polygons to AABB labels."""

    src = Path(obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_detect_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                continue
            rel = img_path.relative_to(src_img)
            lbl_path = (src_lbl / rel).with_suffix(".txt")
            if not lbl_path.exists():
                lbl_path = src_lbl / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            detections = _parse_obb_label_lines(lbl_path)
            out_lines = []
            for cls_id, poly in detections:
                _ = cls_id
                cx, cy, bw, bh = _convert_obb_to_aabb(poly)
                out_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if not out_lines:
                continue

            dst_img = out_dir / "images" / split / img_path.name
            dst_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
            counter = 1
            while dst_img.exists() or dst_lbl.exists():
                dst_img = (
                    out_dir
                    / "images"
                    / split
                    / f"{img_path.stem}_{counter}{img_path.suffix.lower()}"
                )
                dst_lbl = out_dir / "labels" / split / f"{img_path.stem}_{counter}.txt"
                counter += 1

            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
            counts[split] += 1
            counts["objects"] += len(out_lines)

    include_test = counts["test"] > 0
    _write_dataset_yaml(out_dir, class_name=class_name, include_test=include_test)
    manifest = {
        "type": "derived_detect",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def _clip_crop(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int] | None:
    xi1 = int(max(0, math.floor(x1)))
    yi1 = int(max(0, math.floor(y1)))
    xi2 = int(min(w, math.ceil(x2)))
    yi2 = int(min(h, math.ceil(y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return xi1, yi1, xi2, yi2


def derive_crop_obb_dataset_from_obb(
    obb_dataset_dir: str | Path,
    output_root: str | Path,
    class_name: str,
    pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
) -> DatasetBuildResult:
    """Create crop-domain OBB dataset for sequential stage-2 training."""

    src = Path(obb_dataset_dir).expanduser().resolve()
    out_root = Path(output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / f"derived_crop_obb_{_timestamp()}"

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0, "objects": 0}

    for split in ("train", "val", "test"):
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        if not src_img.exists():
            continue
        for img_path in sorted(src_img.rglob("*")):
            if img_path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
            }:
                continue
            rel = img_path.relative_to(src_img)
            lbl_path = (src_lbl / rel).with_suffix(".txt")
            if not lbl_path.exists():
                lbl_path = src_lbl / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]

            detections = _parse_obb_label_lines(lbl_path)
            for obj_idx, (_cls_id, poly_norm) in enumerate(detections):
                poly_px = np.zeros((4, 2), dtype=np.float32)
                poly_px[:, 0] = poly_norm[:, 0] * float(w)
                poly_px[:, 1] = poly_norm[:, 1] * float(h)

                x1 = float(np.min(poly_px[:, 0]))
                y1 = float(np.min(poly_px[:, 1]))
                x2 = float(np.max(poly_px[:, 0]))
                y2 = float(np.max(poly_px[:, 1]))

                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                cx = x1 + bw * 0.5
                cy = y1 + bh * 0.5

                crop_w = max(
                    float(min_crop_size_px), bw * (1.0 + 2.0 * max(0.0, pad_ratio))
                )
                crop_h = max(
                    float(min_crop_size_px), bh * (1.0 + 2.0 * max(0.0, pad_ratio))
                )
                if enforce_square:
                    side = max(crop_w, crop_h)
                    crop_w = side
                    crop_h = side

                c = _clip_crop(
                    cx - crop_w * 0.5,
                    cy - crop_h * 0.5,
                    cx + crop_w * 0.5,
                    cy + crop_h * 0.5,
                    w,
                    h,
                )
                if c is None:
                    continue
                xi1, yi1, xi2, yi2 = c
                crop = img[yi1:yi2, xi1:xi2]
                if crop is None or crop.size == 0:
                    continue

                poly_crop = poly_px.copy()
                poly_crop[:, 0] -= float(xi1)
                poly_crop[:, 1] -= float(yi1)
                ch, cw = crop.shape[:2]
                if ch <= 0 or cw <= 0:
                    continue

                poly_crop[:, 0] = np.clip(poly_crop[:, 0] / float(cw), 0.0, 1.0)
                poly_crop[:, 1] = np.clip(poly_crop[:, 1] / float(ch), 0.0, 1.0)

                stem = f"{img_path.stem}__obj{obj_idx:03d}"
                dst_img = out_dir / "images" / split / f"{stem}.jpg"
                dst_lbl = out_dir / "labels" / split / f"{stem}.txt"
                counter = 1
                while dst_img.exists() or dst_lbl.exists():
                    dst_img = out_dir / "images" / split / f"{stem}_{counter}.jpg"
                    dst_lbl = out_dir / "labels" / split / f"{stem}_{counter}.txt"
                    counter += 1

                cv2.imwrite(str(dst_img), crop)
                coords = " ".join(f"{float(v):.6f}" for v in poly_crop.reshape(-1))
                dst_lbl.write_text(f"0 {coords}\n", encoding="utf-8")
                counts[split] += 1
                counts["objects"] += 1

    include_test = counts["test"] > 0
    _write_dataset_yaml(out_dir, class_name=class_name, include_test=include_test)
    manifest = {
        "type": "derived_crop_obb",
        "source": str(src),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pad_ratio": float(pad_ratio),
        "min_crop_size_px": int(min_crop_size_px),
        "enforce_square": bool(enforce_square),
        "counts": counts,
    }
    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)
    return DatasetBuildResult(
        str(out_dir), stats=manifest, manifest_path=str(manifest_path)
    )


def prepare_role_dataset(
    role: TrainingRole,
    merged_obb_dataset_dir: str,
    role_output_root: str | Path,
    class_name: str,
    *,
    crop_pad_ratio: float = 0.15,
    min_crop_size_px: int = 64,
    enforce_square: bool = True,
) -> DatasetBuildResult:
    """Prepare role-specific dataset from merged OBB source."""

    out_root = Path(role_output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if role == TrainingRole.OBB_DIRECT:
        manifest_path = Path(merged_obb_dataset_dir) / "manifest.json"
        return DatasetBuildResult(
            dataset_dir=str(Path(merged_obb_dataset_dir).resolve()),
            stats={"type": "passthrough_obb"},
            manifest_path=str(manifest_path) if manifest_path.exists() else "",
        )
    if role == TrainingRole.SEQ_DETECT:
        return derive_detect_dataset_from_obb(
            merged_obb_dataset_dir, out_root, class_name
        )
    if role == TrainingRole.SEQ_CROP_OBB:
        return derive_crop_obb_dataset_from_obb(
            merged_obb_dataset_dir,
            out_root,
            class_name,
            pad_ratio=crop_pad_ratio,
            min_crop_size_px=min_crop_size_px,
            enforce_square=enforce_square,
        )

    raise RuntimeError(f"Unsupported training role for dataset preparation: {role}")
