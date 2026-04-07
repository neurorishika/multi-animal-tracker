"""ANN/clustering backend probe helpers for ClassKit.
No FAISS, MLX, or other    platfoRm-specific dependencies required.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict


def probe_clustering_backend() -> Dict[str, Any]:
    """Probe available ANN / clustering backends.

    Returns a dict describing which libraries are importable and which to use.
    """
    result: Dict[str, Any] = {
        "hnswlib": importlib.util.find_spec("hnswlib") is not None,
        "usearch": importlib.util.find_spec("usearch") is not None,
        "annoy": importlib.util.find_spec("annoy") is not None,
        "hdbscan_available": False,
        "sklearn_version": None,
    }

    try:
        import sklearn  # noqa: F401

        result["sklearn_version"] = sklearn.__version__
        from sklearn.cluster import HDBSCAN  # noqa: F401

        result["hdbscan_available"] = True
    except (ImportError, AttributeError):
        pass

    if result["hnswlib"]:
        result["best_ann"] = "hnswlib"
    elif result["usearch"]:
        result["best_ann"] = "usearch"
    elif result["annoy"]:
        result["best_ann"] = "annoy"
    else:
        result["best_ann"] = "numpy"

    return result
