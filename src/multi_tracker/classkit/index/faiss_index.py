try:
    import faiss
except ImportError:
    faiss = None

from pathlib import Path
from typing import Tuple

import numpy as np


class KnnIndex:
    """Wrapper around FAISS index."""

    def __init__(self, d: int, metric: str = "l2"):
        if faiss is None:
            raise ImportError(
                "faiss is not installed. Please install faiss-cpu or faiss-gpu."
            )

        self.d = d
        if metric == "l2":
            self.index = faiss.IndexFlatL2(d)
        elif metric == "ip":
            self.index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unknown metric {metric}")

    def add(self, embeddings: np.ndarray):
        """Add vectors to index."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.d:
            raise ValueError(
                f"Expected embeddings shape (N, {self.d}), got {embeddings.shape}"
            )

        # FAISS expects float32
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors.
        Returns: (distances, indices)
        """
        queries = queries.astype(np.float32)
        return self.index.search(queries, k)

    def save(self, path: Path):
        faiss.write_index(self.index, str(path))

    def load(self, path: Path):
        self.index = faiss.read_index(str(path))
        self.d = self.index.d
