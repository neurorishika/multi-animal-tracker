"""
Export datasets in Ultralytics classify format.

Format structure:
dataset/
    train/
        class_0/
        class_1/
    val/
        class_0/
        class_1/
    data.yaml (metadata file)
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm


def export_ultralytics_classify(
    output_path: Path,
    train_images: List[Path],
    train_labels: List[int],
    val_images: List[Path],
    val_labels: List[int],
    test_images: Optional[List[Path]] = None,
    test_labels: Optional[List[int]] = None,
    class_names: Optional[Dict[int, str]] = None,
    copy: bool = True,
) -> Path:
    """
    Export dataset in Ultralytics classify format.

    Args:
        output_path: Root output directory
        train_images: Training image paths
        train_labels: Training labels
        val_images: Validation image paths
        val_labels: Validation labels
        test_images: Test image paths (optional)
        test_labels: Test labels (optional)
        class_names: Mapping from label ID to class name
        copy: Copy files (True) or symlink (False)

    Returns:
        Path to dataset root
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine unique classes
    all_labels = set(train_labels) | set(val_labels)
    if test_labels:
        all_labels |= set(test_labels)

    # Build class names list
    if class_names is None:
        class_names = {label: f"class_{label}" for label in all_labels}

    names_list = [class_names[i] for i in sorted(all_labels)]
    num_classes = len(names_list)

    # Export train/val/test splits
    _export_split(output_path, "train", train_images, train_labels, class_names, copy)
    _export_split(output_path, "val", val_images, val_labels, class_names, copy)

    if test_images and test_labels:
        _export_split(output_path, "test", test_images, test_labels, class_names, copy)

    # Create data.yaml
    data_yaml = {
        "path": str(output_path.absolute()),
        "train": "train",
        "val": "val",
        "nc": num_classes,
        "names": names_list,
    }

    if test_images and test_labels:
        data_yaml["test"] = "test"

    with open(output_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    return output_path


def _export_split(
    root: Path,
    split: str,
    images: List[Path],
    labels: List[int],
    class_names: Dict[int, str],
    copy: bool,
):
    """Export a single split (train/val/test)."""
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Group by label
    label_to_images: Dict[int, List[Path]] = {}
    for img, label in zip(images, labels):
        if label not in label_to_images:
            label_to_images[label] = []
        label_to_images[label].append(img)

    # Create class directories
    for label, imgs in tqdm(
        label_to_images.items(), desc=f"Exporting {split}", unit="class"
    ):
        class_name = class_names[label]
        class_dir = split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            dest_path = class_dir / img_path.name

            # Handle collisions
            counter = 1
            while dest_path.exists():
                dest_path = class_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1

            if copy:
                shutil.copy2(img_path, dest_path)
            else:
                dest_path.symlink_to(img_path.resolve())
