"""Local training run registry for MAT."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .contracts import TrainingRunSpec


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _use_project_root_override() -> bool:
    return getattr(_project_root, "__module__", __name__) != __name__


def get_runs_root() -> Path:
    if _use_project_root_override():
        root = _project_root() / "training" / "runs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    from hydra_suite.paths import get_training_runs_dir

    return get_training_runs_dir()


def get_registry_path() -> Path:
    return get_runs_root() / "registry.json"


def load_registry() -> dict[str, Any]:
    path = get_registry_path()
    if not path.exists():
        return {"runs": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": []}
    if not isinstance(data, dict):
        return {"runs": []}
    runs = data.get("runs")
    if not isinstance(runs, list):
        data["runs"] = []
    return data


def save_registry(registry: dict[str, Any]) -> None:
    path = get_registry_path()
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def dataset_fingerprint(dataset_dir: str | Path) -> str:
    """Compute stable dataset fingerprint from metadata and file listing."""

    root = Path(dataset_dir).expanduser().resolve()
    h = hashlib.sha256()
    h.update(str(root).encode("utf-8"))

    manifest = root / "manifest.json"
    if manifest.exists():
        h.update(manifest.read_bytes())

    yaml_path = root / "dataset.yaml"
    if yaml_path.exists():
        h.update(yaml_path.read_bytes())

    # Include deterministic listing with file sizes and mtime_ns.
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        h.update(rel.encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))

    return h.hexdigest()


def new_run_id(role: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tail = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return f"{stamp}_{role}_{tail}"


def create_run_record(
    spec: TrainingRunSpec,
    run_id: str,
    run_dir: str | Path,
    dataset_fp: str,
    parent_run_id: str = "",
) -> dict[str, Any]:
    """Create and persist initial run record."""

    now = datetime.now().isoformat(timespec="seconds")
    rec = {
        "run_id": run_id,
        "started_at": now,
        "finished_at": "",
        "status": "running",
        "role": spec.role.value,
        "dataset_fingerprint": dataset_fp,
        "command": [],
        "metrics_paths": [],
        "artifact_paths": [],
        "published_model_path": "",
        "published_registry_entry": "",
        "parent_run_id": parent_run_id,
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "spec": spec.to_dict(),
    }
    reg = load_registry()
    reg.setdefault("runs", []).append(rec)
    save_registry(reg)
    return rec


def update_run_record(run_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    reg = load_registry()
    for rec in reg.get("runs", []):
        if rec.get("run_id") == run_id:
            rec.update(patch)
            save_registry(reg)
            return rec
    return None


def finalize_run_record(
    run_id: str,
    *,
    status: str,
    command: list[str] | None = None,
    metrics_paths: list[str] | None = None,
    artifact_paths: list[str] | None = None,
    published_model_path: str = "",
    published_registry_entry: str = "",
    error_message: str = "",
) -> dict[str, Any] | None:
    """Finalize a run record with terminal status."""

    patch = {
        "status": status,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "command": list(command or []),
        "metrics_paths": list(metrics_paths or []),
        "artifact_paths": list(artifact_paths or []),
        "published_model_path": published_model_path,
        "published_registry_entry": published_registry_entry,
    }
    if error_message:
        patch["error_message"] = error_message
    return update_run_record(run_id, patch)
