"""
FAISS-based clustering for visual modes (overclustering).

Key principle: Clusters represent visual modes, NOT classes.
We use many clusters (hundreds to thousands) to capture fine-grained structure.
"""

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ClusterStats:
    """Per-cluster statistics."""

    cluster_id: int
    size: int
    density: float  # mean kNN distance within cluster
    purity: Optional[float] = None  # if labels exist
    disagreement: Optional[float] = None  # prediction disagreement rate


class FAISSClusterer:
    """
    K-means clustering using FAISS for efficiency.

    Supports GPU acceleration if available.
    """

    def __init__(self, n_clusters: int = 500, niter: int = 50, verbose: bool = False):
        """
        Args:
            n_clusters: Number of clusters (default: overcluster)
            niter: Number of k-means iterations
            verbose: Print clustering progress
        """
        if faiss is None:
            raise ImportError("faiss is required for clustering")

        self.n_clusters = n_clusters
        self.niter = niter
        self.verbose = verbose
        self.kmeans = None
        self.cluster_centers = None
        self.cluster_assignments = None

    def fit(self, embeddings: np.ndarray, gpu: bool = False) -> np.ndarray:
        """
        Cluster embeddings using k-means.

        Args:
            embeddings: (N, D) array of embeddings
            gpu: Use GPU if available

        Returns:
            cluster_assignments: (N,) array of cluster IDs
        """
        N, D = embeddings.shape

        if N < self.n_clusters:
            raise ValueError(f"Not enough samples ({N}) for {self.n_clusters} clusters")

        # Ensure float32 and C-contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Initialize k-means
        self.kmeans = faiss.Kmeans(
            D,
            self.n_clusters,
            niter=self.niter,
            verbose=self.verbose,
            gpu=gpu,
            spherical=False,  # Euclidean distance
        )

        # Fit
        self.kmeans.train(embeddings)

        # Get cluster centers
        self.cluster_centers = self.kmeans.centroids  # (K, D)

        # Assign all points
        _, self.cluster_assignments = self.kmeans.index.search(embeddings, 1)
        self.cluster_assignments = self.cluster_assignments.ravel()

        return self.cluster_assignments

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Assign embeddings to nearest cluster.

        Args:
            embeddings: (N, D) array

        Returns:
            cluster_ids: (N,) array
        """
        if self.kmeans is None:
            raise RuntimeError("Must call fit() first")

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        _, assignments = self.kmeans.index.search(embeddings, 1)
        return assignments.ravel()

    def compute_cluster_stats(
        self,
        embeddings: np.ndarray,
        assignments: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
        knn_k: int = 10,
    ) -> Dict[int, ClusterStats]:
        """
        Compute statistics for each cluster.

        Args:
            embeddings: (N, D) embeddings
            assignments: (N,) cluster IDs (if None, use stored)
            labels: (N,) ground-truth labels (optional)
            predictions: (N,) model predictions (optional)
            knn_k: K for density estimation

        Returns:
            Dictionary mapping cluster_id -> ClusterStats
        """
        if assignments is None:
            assignments = self.cluster_assignments
        if assignments is None:
            raise RuntimeError("No cluster assignments available")

        stats = {}

        # Build kNN index for density
        embeddings_f32 = np.ascontiguousarray(embeddings.astype(np.float32))
        D = embeddings.shape[1]
        index = faiss.IndexFlatL2(D)
        index.add(embeddings_f32)

        for cluster_id in range(self.n_clusters):
            mask = assignments == cluster_id
            cluster_size = mask.sum()

            if cluster_size == 0:
                continue  # Empty cluster

            # Density: mean kNN distance within cluster
            cluster_embs = embeddings_f32[mask]
            k_search = min(knn_k + 1, cluster_size)  # +1 to exclude self
            distances, _ = index.search(cluster_embs, k_search)
            # Exclude self (distance 0)
            mean_density = distances[:, 1:].mean()

            # Purity: fraction of most common label (if labels exist)
            purity = None
            if labels is not None:
                cluster_labels = labels[mask]
                cluster_labels = cluster_labels[
                    cluster_labels >= 0
                ]  # exclude unlabeled
                if len(cluster_labels) > 0:
                    unique, counts = np.unique(cluster_labels, return_counts=True)
                    purity = counts.max() / len(cluster_labels)

            # Disagreement: fraction where prediction != label
            disagreement = None
            if labels is not None and predictions is not None:
                cluster_labels = labels[mask]
                cluster_preds = predictions[mask]
                valid = cluster_labels >= 0
                if valid.sum() > 0:
                    disagreement = (
                        cluster_labels[valid] != cluster_preds[valid]
                    ).mean()

            stats[cluster_id] = ClusterStats(
                cluster_id=cluster_id,
                size=cluster_size,
                density=float(mean_density),
                purity=purity,
                disagreement=disagreement,
            )

        return stats
