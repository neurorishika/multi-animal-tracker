"""Shared source-folder validation helpers for ClassKit dialogs."""

from __future__ import annotations

from pathlib import Path

CLASSKIT_IMAGES_SUBDIR = "images"
CLASSKIT_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_classkit_images(folder: Path) -> int:
    """Return the number of supported image files directly inside *folder*."""
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in CLASSKIT_IMAGE_EXTS
    )


def has_classkit_images(folder: Path) -> bool:
    """Return whether *folder* contains any supported image files."""
    return count_classkit_images(folder) > 0


def resolve_classkit_images_dir(dataset_root: Path) -> Path:
    """Return the required images directory for a selected ClassKit source."""
    images_dir = dataset_root / CLASSKIT_IMAGES_SUBDIR
    if not images_dir.is_dir():
        raise ValueError(
            "Selected source folder must contain an images/ subdirectory.\n\n"
            f"{dataset_root}"
        )
    if not has_classkit_images(images_dir):
        raise ValueError(
            f"No images were found in:\n{images_dir}\n\n"
            "Please select a source folder whose images/ subdirectory contains "
            ".jpg / .jpeg / .png files."
        )
    return images_dir
