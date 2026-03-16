"""ANN/clustering backend probe helpers for ClassKit.

Replaces the old metalfaiss_backend.py.  No FAISS, MLX, or
platform-specific dependencies required.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict


def probe_ann_backend() -> Dict[str, Any]:
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


# ── Legacy shim ────────────────────────────────────────────────────────────────
# Any code that still calls probe_metalfaiss_backend() gets a safe
# "not installed" response instead of an ImportError.


def probe_metalfaiss_backend() -> (
    Dict[str, Any]
):  # noqa: DC02  (legacy shim; callers get a safe "not installed" response)
    """Legacy shim — metalfaiss removed; use probe_ann_backend() instead."""
    return {
        "installed": False,
        "ready": False,
        "origin": None,
        "error": "metalfaiss removed; backend is now hnswlib/sklearn",
        "remediation": None,
        "shadow_path_removed": False,
        "local_path_added": False,
        "likely_local_shadow": False,
    }
