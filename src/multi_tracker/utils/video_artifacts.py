from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable


def _video_stem(video_path: str | os.PathLike[str]) -> str:
    stem = Path(video_path).stem.strip()
    return stem or "video"


def _normalize_base_dir(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def candidate_artifact_base_dirs(
    video_path: str | os.PathLike[str],
    preferred_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()
    raw_candidates: list[str | os.PathLike[str] | None] = [
        Path(video_path).expanduser().parent,
        *(preferred_base_dirs or []),
        tempfile.gettempdir(),
    ]
    for raw in raw_candidates:
        candidate = _normalize_base_dir(raw)
        if candidate is None:
            continue
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
    return candidates


def choose_writable_artifact_base_dir(
    video_path: str | os.PathLike[str],
    preferred_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path:
    for candidate in candidate_artifact_base_dirs(video_path, preferred_base_dirs):
        if candidate.exists() and candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return Path(tempfile.gettempdir())


def build_video_cache_dir(
    video_path: str | os.PathLike[str],
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    cache_dir = base_dir / f"{_video_stem(video_path)}_caches"
    if create:
        cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def build_video_log_dir(
    video_path: str | os.PathLike[str],
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    log_dir = base_dir / f"{_video_stem(video_path)}_logs"
    if create:
        log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def build_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    return cache_dir / f"{_video_stem(video_path)}_detection_cache_{model_id}.npz"


def build_legacy_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
) -> Path:
    base_dir = (
        _normalize_base_dir(artifact_base_dir) or Path(video_path).expanduser().parent
    )
    return base_dir / f"{_video_stem(video_path)}_detection_cache_{model_id}.npz"


def find_existing_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_id: str,
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
) -> Path | None:
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)

    for base_dir in base_dirs:
        current = build_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dir=base_dir,
        )
        if current.exists():
            return current

    for base_dir in base_dirs:
        legacy = build_legacy_detection_cache_path(
            video_path,
            model_id,
            artifact_base_dir=base_dir,
        )
        if legacy.exists():
            return legacy
    return None


def build_optimizer_detection_cache_path(
    video_path: str | os.PathLike[str],
    model_name: str,
    resize_percent: int,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    cache_dir = build_video_cache_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    stem = _video_stem(video_path)
    return cache_dir / f"{stem}_{model_name}_r{int(resize_percent)}_opt_cache.npz"


def build_individual_properties_cache_path(
    video_path: str | os.PathLike[str],
    properties_id: str,
    start_frame: int,
    end_frame: int,
    detection_cache_path: str | os.PathLike[str] | None = None,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    if detection_cache_path:
        base_dir = Path(detection_cache_path).expanduser().parent
        if create_dir:
            base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = build_video_cache_dir(
            video_path,
            artifact_base_dir=artifact_base_dir,
            create=create_dir,
        )
    stem = _video_stem(video_path)
    return base_dir / (
        f"{stem}_individual_properties_{properties_id}_"
        f"{int(start_frame)}_{int(end_frame)}.npz"
    )


def build_autotune_state_path(
    detection_cache_path: str | os.PathLike[str],
) -> Path:
    return Path(detection_cache_path).expanduser().with_suffix(".autotune_state.json")


def build_tracking_session_log_path(
    video_path: str | os.PathLike[str],
    timestamp: str,
    artifact_base_dir: str | os.PathLike[str] | None = None,
    create_dir: bool = False,
) -> Path:
    log_dir = build_video_log_dir(
        video_path,
        artifact_base_dir=artifact_base_dir,
        create=create_dir,
    )
    return log_dir / f"{_video_stem(video_path)}_tracking_{timestamp}.log"


def iter_detection_cache_candidates(
    video_path: str | os.PathLike[str],
    artifact_base_dirs: Iterable[str | os.PathLike[str] | None] | None = None,
    include_legacy: bool = True,
) -> list[Path]:
    stem = _video_stem(video_path)
    pattern = f"{stem}*_cache*.npz"
    found: dict[str, Path] = {}
    base_dirs = artifact_base_dirs or candidate_artifact_base_dirs(video_path)

    for base_dir in base_dirs:
        normalized_base_dir = _normalize_base_dir(base_dir)
        if normalized_base_dir is None:
            continue

        cache_dir = build_video_cache_dir(
            video_path, artifact_base_dir=normalized_base_dir
        )
        if cache_dir.exists():
            for path in cache_dir.glob(pattern):
                found[str(path.resolve())] = path

        if include_legacy and normalized_base_dir.exists():
            for path in normalized_base_dir.glob(pattern):
                found[str(path.resolve())] = path

    return sorted(
        found.values(),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )


__all__ = [
    "build_autotune_state_path",
    "build_detection_cache_path",
    "build_individual_properties_cache_path",
    "build_optimizer_detection_cache_path",
    "build_tracking_session_log_path",
    "build_video_cache_dir",
    "build_video_log_dir",
    "candidate_artifact_base_dirs",
    "choose_writable_artifact_base_dir",
    "find_existing_detection_cache_path",
    "iter_detection_cache_candidates",
]
