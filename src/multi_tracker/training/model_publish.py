"""Model publishing into MAT model repositories with metadata lineage."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import TrainingRole


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_models_root() -> Path:
    root = _project_root() / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_dir_for_role(role: TrainingRole, scheme_name: str = "orientation") -> Path:
    root = get_models_root()
    if role == TrainingRole.SEQ_DETECT:
        out = root / "YOLO-detect"
    elif role == TrainingRole.SEQ_CROP_OBB:
        out = root / "YOLO-obb" / "cropped"
    elif role in (TrainingRole.HEADTAIL_TINY, TrainingRole.HEADTAIL_YOLO):
        out = root / "YOLO-classify" / "orientation"
    elif role == TrainingRole.CLASSIFY_FLAT_YOLO:
        out = root / "YOLO-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_FLAT_TINY:
        out = root / "tiny-classify" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_YOLO:
        out = root / "YOLO-classify" / "multihead" / scheme_name
    elif role == TrainingRole.CLASSIFY_MULTIHEAD_TINY:
        out = root / "tiny-classify" / "multihead" / scheme_name
    else:
        out = root / "YOLO-obb"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _task_usage_for_role(role: TrainingRole) -> tuple[str, str]:
    if role == TrainingRole.OBB_DIRECT:
        return "obb", "obb_direct"
    if role == TrainingRole.SEQ_DETECT:
        return "detect", "seq_detect"
    if role == TrainingRole.SEQ_CROP_OBB:
        return "obb", "seq_crop_obb"
    if role in (TrainingRole.CLASSIFY_FLAT_YOLO, TrainingRole.CLASSIFY_MULTIHEAD_YOLO):
        return "classify", "classify_yolo"
    if role in (TrainingRole.CLASSIFY_FLAT_TINY, TrainingRole.CLASSIFY_MULTIHEAD_TINY):
        return "classify", "classify_tiny"
    return "classify", "headtail"


def _registry_path() -> Path:
    return get_models_root() / "model_registry.json"


def load_model_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_model_registry(registry: dict[str, Any]) -> None:
    _registry_path().write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _registry_key_for_model(model_path: Path) -> str:
    """Generate registry key compatible with existing main_window behavior."""

    model_path = model_path.resolve()
    models_root = get_models_root().resolve()
    yolo_obb_root = (models_root / "YOLO-obb").resolve()

    try:
        rel_obb = model_path.relative_to(yolo_obb_root)
        return rel_obb.as_posix()
    except Exception:
        pass

    try:
        rel_root = model_path.relative_to(models_root)
        return rel_root.as_posix()
    except Exception:
        return str(model_path)


def publish_trained_model(
    *,
    role: TrainingRole,
    artifact_path: str,
    size: str,
    species: str,
    model_info: str,
    trained_from_run_id: str,
    dataset_fingerprint: str,
    base_model: str,
    scheme_name: str = "",
    factor_index: int | None = None,
    factor_name: str | None = None,
) -> tuple[str, str]:
    """Copy trained artifact into repository and register metadata.

    Returns:
        (registry_key, absolute_model_path)
    """

    src = Path(artifact_path).expanduser().resolve()
    if not src.exists():
        raise RuntimeError(f"Trained artifact not found: {src}")

    repo_dir = _repo_dir_for_role(role, scheme_name=scheme_name or "orientation")
    task_family, usage_role = _task_usage_for_role(role)

    now = datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    added_at = now.isoformat(timespec="seconds")

    safe_species = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(species or "species")
        ).strip("_")
        or "species"
    )
    safe_info = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(model_info or "model")
        ).strip("_")
        or "model"
    )
    safe_size = (
        "".join(
            c if c.isalnum() or c in "-_" else "_" for c in str(size or "unknown")
        ).strip("_")
        or "unknown"
    )

    ext = src.suffix.lower() or ".pt"
    base_name = f"{stamp}_{safe_size}_{safe_species}_{safe_info}"
    dst = repo_dir / f"{base_name}{ext}"
    counter = 1
    while dst.exists():
        dst = repo_dir / f"{base_name}_{counter}{ext}"
        counter += 1

    shutil.copy2(src, dst)

    key = _registry_key_for_model(dst)
    metadata = {
        "size": safe_size,
        "species": safe_species,
        "model_info": safe_info,
        "added_at": added_at,
        "source_path": str(src),
        "stored_filename": dst.name,
        "task_family": task_family,
        "usage_role": usage_role,
        "trained_from_run_id": str(trained_from_run_id or ""),
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "base_model": str(base_model or ""),
        "scheme_name": str(scheme_name or ""),
        "factor_index": factor_index,
        "factor_name": str(factor_name) if factor_name is not None else None,
    }

    reg = load_model_registry()
    reg[key] = metadata
    save_model_registry(reg)

    return key, str(dst)
