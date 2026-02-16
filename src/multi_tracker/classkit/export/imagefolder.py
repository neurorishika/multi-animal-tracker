import os
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def export_to_imagefolder(
    dataset_root: Path,
    images: List[Tuple[Path, str, str]],  # (source_path, label, split)
    copy: bool = True,
):
    """
    Export dataset to PyTorch ImageFolder structure.

    Args:
        dataset_root: The root directory for the export.
        images: List of tuples (source_path, label, split).
                split should be 'train', 'val', or 'test'.
        copy: If True, copies files. If False, symlinks (ln -s).
    """
    dataset_root = Path(dataset_root)
    if dataset_root.exists():
        # Safety check: avoid deleting something unintended
        # shutil.rmtree(dataset_root)
        pass

    dataset_root.mkdir(parents=True, exist_ok=True)

    for src_path, label, split in tqdm(images, desc="Exporting ImageFolder"):
        if not label:
            continue

        dest_dir = dataset_root / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / src_path.name

        # Handle duplicates? Simple overwrite for now
        if copy:
            shutil.copy2(src_path, dest_path)
        else:
            if not dest_path.exists():
                os.symlink(src_path.absolute(), dest_path)

    print(f"Exported ImageFolder to {dataset_root}")
