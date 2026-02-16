import hashlib
from pathlib import Path
from typing import Generator, List, Union


def compute_image_hash(path: Path) -> str:
    """Compute MD5 hash of an image file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def scan_images(
    root: Union[str, Path], extensions: List[str] = None
) -> Generator[Path, None, None]:
    """recursively yield image paths from root directory."""
    root = Path(root)
    if not root.exists():
        return

    if extensions is None:
        extensions = [".jpg", ".png", ".jpeg"]

    # Standardize extensions to lowercase for checking
    valid_exts = {e.lower() for e in extensions}

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in valid_exts:
            yield path


class IngestWorker:
    """Worker for scanning and ingesting images into database."""

    def __init__(self, db):
        """
        Args:
            db: ClassKitDB instance
        """
        self.db = db

    def scan_folder(self, folder: Path, extensions: List[str] = None) -> List[Path]:
        """Scan a folder for images.

        Args:
            folder: Path to folder to scan
            extensions: List of valid extensions (default: ['.jpg', '.png', '.jpeg'])

        Returns:
            List of image paths
        """
        if extensions is None:
            extensions = [".jpg", ".png", ".jpeg"]

        images = list(scan_images(folder, extensions))
        return images

    def ingest(self, image_paths: List[Path], compute_hashes: bool = True):
        """Ingest images into database.

        Args:
            image_paths: List of image paths
            compute_hashes: Whether to compute MD5 hashes
        """
        if compute_hashes:
            hashes = [compute_image_hash(p) for p in image_paths]
        else:
            hashes = None

        self.db.add_images(image_paths, hashes)
