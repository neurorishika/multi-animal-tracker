"""
sklearn-based clustering for visual modes (overclustering).

Replaces the old FAISSClusterer with a pure-sklearn implementation that
works identically on macOS (ARM/x86), Linux, and Windows with no
platform-specific dependencies.

Method options
--------------
``"minibatch"``  (default)
    MiniBatchKMeans — fast, suitable for 10 k–500 k vectors.
``"kmeans"``
    Exact KMeans — higher quality, slower; auto-selected for N < 5000.
``"hdbscan"``
    HDBSCAN density clustering — discovers K automatically; requires
    scikit-learn ≥ 1.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class ClusterStats:
    """Per-cluster statistics."""

    cluster_id: int
    size: int
    density: float  # mean kNN distance within cluster
    purity: Optional[float] = None  # fraction of dominant label
    disagreement: Optional[float] = None  # prediction ≠ label rate


class SKLearnClusterer:
    """Cross-platform k-means / HDBSCAN clusterer backed by scikit-learn."""

    def __init__(
        self,
        n_clusters: int = 500,
        method: str = "minibatch",
        niter: int = 50,
        verbose: bool = False,
    ):
        self.n_clusters = n_clusters
        self.method = method
        self.niter = niter
        self.verbose = verbose
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_assignments: Optional[np.ndarray] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster *embeddings* and return (N,) integer cluster-ID array."""
        emb = np.ascontiguousarray(embeddings.astype(np.float32))
        N = len(emb)

        if self.method == "hdbscan":
            return self._fit_hdbscan(emb)

        k = min(self.n_clusters, N)
        if k < 2:
            self.cluster_assignments = np.zeros(N, dtype=np.int32)
            self.cluster_centers = emb[:1].copy()
            return self.cluster_assignments

        if self.method == "minibatch" or N > 5000:
            from sklearn.cluster import MiniBatchKMeans

            batch_size = min(max(k * 3, 1024), N)
            clf = MiniBatchKMeans(
                n_clusters=k,
                batch_size=batch_size,
                max_iter=self.niter,
                n_init=5,
                random_state=42,
                verbose=int(self.verbose),
            )
        else:
            from sklearn.cluster import KMeans

            clf = KMeans(
                n_clusters=k,
                max_iter=self.niter,
                n_init=5,
                random_state=42,
                verbose=int(self.verbose),
            )

        self.cluster_assignments = clf.fit_predict(emb)
        self.cluster_centers = clf.cluster_centers_
        return self.cluster_assignments

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign new embeddings to nearest cluster centre."""
        if self.cluster_centers is None:
            raise RuntimeError("Must call fit() first")
        emb = embeddings.astype(np.float32)
        dists = np.linalg.norm(emb[:, None] - self.cluster_centers[None], axis=2)
        return dists.argmin(axis=1).astype(np.int32)

    def compute_cluster_stats(
        self,
        embeddings: np.ndarray,
        assignments: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        knn_k: int = 10,
    ) -> Dict[int, ClusterStats]:
        """Compute per-cluster density, purity, and disagreement stats."""
        if assignments is None:
            assignments = self.cluster_assignments
        if assignments is None:
            raise RuntimeError("No cluster assignments available")

        emb = embeddings.astype(np.float32)
        n_clusters = int(assignments.max()) + 1
        knn_fn = _build_knn_fn(emb, knn_k)

        stats: Dict[int, ClusterStats] = {}
        for cid in range(n_clusters):
            mask = assignments == cid
            size = int(mask.sum())
            if size == 0:
                continue

            density = float(knn_fn(emb[mask]).mean())

            purity = None
            if labels is not None:
                cl = labels[mask]
                cl = cl[cl >= 0]
                if len(cl) > 0:
                    _, counts = np.unique(cl, return_counts=True)
                    purity = float(counts.max() / len(cl))

            disagreement = None
            if labels is not None and predictions is not None:
                cl = labels[mask]
                cp = predictions[mask]
                valid = cl >= 0
                if valid.sum() > 0:
                    disagreement = float((cl[valid] != cp[valid]).mean())

            stats[cid] = ClusterStats(
                cluster_id=cid,
                size=size,
                density=density,
                purity=purity,
                disagreement=disagreement,
            )
        return stats

    # ── private ───────────────────────────────────────────────────────────────

    def _fit_hdbscan(self, emb: np.ndarray) -> np.ndarray:
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            raise ImportError(
                "HDBSCAN requires scikit-learn ≥ 1.3. "
                "Upgrade with: pip install -U scikit-learn"
            )

        min_cs = max(5, len(emb) // max(self.n_clusters, 1))
        clf = HDBSCAN(min_cluster_size=min_cs)
        labels = clf.fit_predict(emb)

        n_found = int(labels.max()) + 1
        if n_found < 1:
            # All noise — fall back to minibatch
            self.method = "minibatch"
            return self.fit(emb)

        # Compute centres from labelled members
        centers = np.array([emb[labels == i].mean(axis=0) for i in range(n_found)])

        # Re-assign noise points (label -1) to nearest centre
        noise = labels == -1
        if noise.any():
            d = np.linalg.norm(emb[noise][:, None] - centers[None], axis=2)
            labels[noise] = d.argmin(axis=1)

        self.cluster_assignments = labels
        self.cluster_centers = centers
        return labels


def _build_knn_fn(embeddings: np.ndarray, k: int):
    """Return a callable(queries) → (N,) mean-kNN-distance array.

    Tries hnswlib first; falls back to brute-force numpy.
    """
    k_actual = min(k + 1, len(embeddings))  # +1 to exclude self match

    try:
        import hnswlib

        dim = embeddings.shape[1]
        idx = hnswlib.Index(space="l2", dim=dim)
        idx.init_index(max_elements=len(embeddings), ef_construction=100, M=16)
        idx.add_items(embeddings)
        idx.set_ef(max(k_actual, 50))

        def _hn(q):
            _, dists = idx.knn_query(q, k=k_actual)
            return dists[:, 1:].mean(axis=1)  # skip self (dist ~0)

        return _hn
    except ImportError:
        pass

    def _np(q):
        d = np.linalg.norm(q[:, None] - embeddings[None], axis=2)
        d.sort(axis=1)
        return d[:, 1:k_actual].mean(axis=1)

    return _np


# Backward-compat alias so any code that does
# `from .clusterer import FAISSClusterer` keeps working.
FAISSClusterer = SKLearnClusterer
