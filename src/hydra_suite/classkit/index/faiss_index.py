"""KNN index backed by hnswlib → usearch → numpy brute-force.

Replaces the old FAISS-based KnnIndex with a pure-wheel stack that
installs on every platform without compilation.  Public API
(add / search / save / load) is identical to the previous version.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np


def _best_backend() -> str:
    for name in ("hnswlib", "usearch"):
        if importlib.util.find_spec(name) is not None:
            return name
    return "numpy"


class KnnIndex:
    """ANN index: hnswlib → usearch → numpy brute-force."""

    def __init__(self, d: int, metric: str = "l2"):
        self.d = d
        self.metric = metric  # "l2" | "ip"
        self._backend = _best_backend()
        # numpy fallback storage
        self._np_data: np.ndarray = np.empty((0, d), dtype=np.float32)
        self._index = None
        self._count = 0  # items added (used as ID counter)

    # ── add ──────────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(
                f"Expected embeddings shape (N, {self.d}), got {embeddings.shape}"
            )
        emb = np.ascontiguousarray(embeddings.astype(np.float32))

        if self._backend == "hnswlib":
            self._add_hnswlib(emb)
        elif self._backend == "usearch":
            self._add_usearch(emb)
        else:
            self._np_data = (
                np.concatenate([self._np_data, emb], axis=0)
                if len(self._np_data) > 0
                else emb.copy()
            )
        self._count += len(emb)

    def _add_hnswlib(self, emb: np.ndarray):
        import hnswlib

        space = "l2" if self.metric in ("l2", "euclidean") else "ip"
        if self._index is None:
            idx = hnswlib.Index(space=space, dim=self.d)
            idx.init_index(
                max_elements=max(len(emb), 1),
                ef_construction=200,
                M=16,
            )
            idx.set_ef(50)
            self._index = idx
        ids = np.arange(self._count, self._count + len(emb))
        self._index.add_items(emb, ids)

    def _add_usearch(self, emb: np.ndarray):
        from usearch.index import Index, MetricKind

        metric = (
            MetricKind.L2sq if self.metric in ("l2", "euclidean") else MetricKind.IP
        )
        if self._index is None:
            self._index = Index(ndim=self.d, metric=metric)
        ids = np.arange(self._count, self._count + len(emb))
        self._index.add(ids, emb)

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) each of shape (N, k)."""
        q = np.ascontiguousarray(queries.astype(np.float32))

        if self._backend == "hnswlib" and self._index is not None:
            return self._search_hnswlib(q, k)
        if self._backend == "usearch" and self._index is not None:
            return self._search_usearch(q, k)
        return self._search_numpy(q, k)

    def _search_hnswlib(self, q, k):
        self._index.set_ef(max(k, 50))
        labels, dists = self._index.knn_query(q, k=min(k, self._count))
        return dists, labels

    def _search_usearch(self, q, k):
        results = self._index.search(q, min(k, self._count))
        return results.distances, results.keys

    def _search_numpy(self, q, k):
        if len(self._np_data) == 0:
            return (
                np.zeros((len(q), k), dtype=np.float32),
                np.zeros((len(q), k), dtype=np.int64),
            )
        dists = np.linalg.norm(q[:, None] - self._np_data[None], axis=2)
        idx = np.argsort(dists, axis=1)[:, :k]
        return np.take_along_axis(dists, idx, axis=1), idx

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        """Save index vectors to a .npz file (``path + '.knn.npz'``)."""
        path = Path(path)
        data = self._np_data if self._backend == "numpy" else self._reconstruct_data()
        np.savez(
            str(path) + ".knn",
            data=data,
            d=np.array([self.d]),
            metric=np.array([self.metric]),
        )

    def load(self, path: Path):
        """Load from file saved by save()."""
        path = Path(path)
        npz = np.load(str(path) + ".knn.npz", allow_pickle=True)
        self.d = int(npz["d"][0])
        self.metric = str(npz["metric"][0])
        self._backend = _best_backend()
        self._np_data = np.empty((0, self.d), dtype=np.float32)
        self._index = None
        self._count = 0
        data = npz["data"]
        if len(data) > 0:
            self.add(data)

    def _reconstruct_data(self) -> np.ndarray:
        """Reconstruct raw vectors from ANN index (for save)."""
        if self._backend == "hnswlib" and self._index is not None:
            try:
                return np.array(
                    [self._index.get_items([i])[0] for i in range(self._count)],
                    dtype=np.float32,
                )
            except Exception:
                pass
        return np.empty((0, self.d), dtype=np.float32)
