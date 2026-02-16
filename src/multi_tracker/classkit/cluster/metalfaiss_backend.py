"""MetalFaiss backend detection and loading helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _remove_shadow_paths() -> bool:
    """Remove known local MetalFaiss source paths that can shadow site-packages."""
    removed = False
    original = list(sys.path)
    filtered = []
    for entry in original:
        normalized = entry.replace("\\", "/")
        if (
            normalized.endswith("/Faiss-mlx/python")
            or "/Faiss-mlx/python" in normalized
        ):
            removed = True
            continue
        filtered.append(entry)

    if removed:
        sys.path[:] = filtered
    return removed


def _clear_metalfaiss_modules() -> None:
    """Clear cached metalfaiss modules so imports can be retried with updated sys.path."""
    to_delete = [
        name
        for name in list(sys.modules)
        if name == "metalfaiss" or name.startswith("metalfaiss.")
    ]
    for name in to_delete:
        del sys.modules[name]


def _ensure_local_faiss_mlx_on_path() -> bool:
    """Add local Faiss-mlx/python path if present and not already on sys.path."""
    spec = importlib.util.find_spec("metalfaiss")
    if spec is not None:
        return False

    repo_root = Path(__file__).resolve().parents[4]
    candidate = repo_root / "Faiss-mlx" / "python"
    if not candidate.exists():
        return False

    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
        return True
    return False


def _install_local_index_pointer_alias() -> bool:
    """Install compatibility alias for local MetalFaiss layouts.

    Some local Faiss-mlx trees place the module at
    `metalfaiss.index.index_pointer` while clustering imports
    `metalfaiss.index_pointer`.
    """
    if "metalfaiss.index_pointer" in sys.modules:
        return True

    try:
        from metalfaiss.index import index_pointer as idx_ptr_mod
    except Exception:
        return False

    sys.modules["metalfaiss.index_pointer"] = idx_ptr_mod
    return True


def _import_clustering_symbols() -> Tuple[Any, Any]:
    """Import clustering symbols with a local-layout compatibility retry."""
    try:
        from metalfaiss.clustering import AnyClustering, ClusteringParameters

        return AnyClustering, ClusteringParameters
    except ModuleNotFoundError as exc:
        if "metalfaiss.index_pointer" not in str(exc):
            raise
        if not _install_local_index_pointer_alias():
            raise
        from metalfaiss.clustering import AnyClustering, ClusteringParameters

        return AnyClustering, ClusteringParameters


def load_metalfaiss_backend() -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Load MLX + MetalFaiss clustering symbols with one retry against path shadowing."""
    local_path_added = _ensure_local_faiss_mlx_on_path()
    import mlx.core as mx

    shadow_path_removed = False
    try:
        AnyClustering, ClusteringParameters = _import_clustering_symbols()
    except Exception:
        shadow_path_removed = _remove_shadow_paths()
        if not shadow_path_removed:
            raise
        _clear_metalfaiss_modules()
        AnyClustering, ClusteringParameters = _import_clustering_symbols()

    spec = importlib.util.find_spec("metalfaiss")
    origin = getattr(spec, "origin", None)
    info = {
        "origin": origin,
        "shadow_path_removed": shadow_path_removed,
        "local_path_added": local_path_added,
    }
    return mx, AnyClustering, ClusteringParameters, info


def probe_metalfaiss_backend() -> Dict[str, Any]:
    """Probe if MetalFaiss backend is importable in the current Python runtime."""
    local_path_added = _ensure_local_faiss_mlx_on_path()
    spec = importlib.util.find_spec("metalfaiss")
    installed = spec is not None
    origin = getattr(spec, "origin", None) if spec else None
    result: Dict[str, Any] = {
        "installed": installed,
        "ready": False,
        "origin": origin,
        "error": None,
        "remediation": None,
        "shadow_path_removed": False,
        "local_path_added": local_path_added,
        "likely_local_shadow": bool(
            origin and "Faiss-mlx/python/metalfaiss" in str(origin).replace("\\", "/")
        ),
    }

    if not installed:
        return result

    try:
        _, _, _, info = load_metalfaiss_backend()
        result["ready"] = True
        result["shadow_path_removed"] = info.get("shadow_path_removed", False)
        result["local_path_added"] = info.get("local_path_added", local_path_added)
        if info.get("origin"):
            result["origin"] = info["origin"]
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        result["error"] = err
        if "metalfaiss.index_pointer" in str(exc):
            result["remediation"] = (
                "Local MetalFaiss layout mismatch: clustering expects metalfaiss.index_pointer. "
                "ClassKit now attempts an automatic compatibility alias to metalfaiss.index.index_pointer."
            )

    return result
