"""
Export labeled datasets to CSV/Parquet format.

Format:
    image_path, label, split, class_name
    /path/to/img1.jpg, 0, train, dog
    /path/to/img2.jpg, 1, val, cat
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional


def export_to_csv(
    output_path: Path,
    image_paths: List[Path],
    labels: List[int],
    class_names: Optional[Dict[int, str]] = None,
    splits: Optional[List[str]] = None,
    include_header: bool = True,
):
    """
    Export dataset to CSV format.

    Args:
        output_path: Output CSV file path
        image_paths: List of image paths
        labels: List of labels
        class_names: Optional mapping from label ID to class name
        splits: Optional list of split names (same length as image_paths)
        include_header: Include CSV header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        if include_header:
            writer.writerow(["image_path", "label", "split", "class_name"])

        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            class_name = (
                class_names.get(label, f"class_{label}")
                if class_names
                else f"class_{label}"
            )
            split = splits[i] if splits and i < len(splits) else ""

            writer.writerow([str(img_path), label, split, class_name])


def export_to_parquet(
    output_path: Path,
    image_paths: List[Path],
    labels: List[int],
    class_names: Optional[Dict[int, str]] = None,
    splits: Optional[List[str]] = None,
):
    """
    Export dataset to Parquet format (requires pyarrow or fastparquet).

    Args:
        output_path: Output Parquet file path
        image_paths: List of image paths
        labels: List of labels
        class_names: Optional mapping from label ID to class name
        splits: Optional list of split names
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for Parquet export")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dataframe
    data = {
        "image_path": [str(p) for p in image_paths],
        "label": labels,
    }

    if class_names:
        data["class_name"] = [
            class_names.get(label, f"class_{label}") for label in labels
        ]

    if splits:
        data["split"] = splits

    df = pd.DataFrame(data)

    # Write to parquet
    df.to_parquet(output_path, index=False)
