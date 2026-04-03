"""Path fingerprinting and artifact caching for pose runtime backends."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def path_fingerprint_token(path_str: str) -> str:
    """Generate fingerprint token for path (file or directory)."""
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        return f"{p}|missing"
    if p.is_file():
        stat = p.stat()
        return f"{p}|f|{stat.st_mtime_ns}|{stat.st_size}"

    # Directory fingerprint: root stat + key files + shallow recursive fallback.
    parts: List[str] = []
    try:
        stat = p.stat()
        parts.append(f"{p}|d|{stat.st_mtime_ns}")
    except OSError:
        parts.append(f"{p}|d|unknown")

    key_names = {
        "best.ckpt",
        "training_config.yaml",
        "training_config.json",
        "export_metadata.json",
        "metadata.json",
    }
    file_count = 0
    for child in sorted(p.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(p)
        suffix = child.suffix.lower()
        if (
            child.name in key_names
            or suffix
            in {".pt", ".ckpt", ".onnx", ".engine", ".trt", ".json", ".yaml", ".yml"}
            or file_count < 64
        ):
            try:
                st = child.stat()
                parts.append(f"{rel}|{st.st_mtime_ns}|{st.st_size}")
            except OSError:
                parts.append(f"{rel}|unknown")
            file_count += 1
        if file_count >= 256:
            break
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:24]


def artifact_meta_path(path: Path) -> Path:
    """Get metadata file path for artifact."""
    p = path.expanduser().resolve()
    if p.exists() and p.is_dir():
        return p / ".runtime_meta.json"
    if p.suffix:
        return p.with_suffix(f"{p.suffix}.runtime_meta.json")
    return p / ".runtime_meta.json"


def artifact_meta_matches(path: Path, signature: str) -> bool:
    """Check if artifact metadata matches expected signature."""
    p = path.expanduser().resolve()
    if not p.exists():
        return False
    meta = artifact_meta_path(p)
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(data.get("signature", "")) == str(signature)


def write_artifact_meta(path: Path, signature: str) -> None:
    """Write artifact metadata with signature."""
    meta = artifact_meta_path(path.expanduser().resolve())
    payload = {"signature": str(signature)}
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
