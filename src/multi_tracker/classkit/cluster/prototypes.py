"""
Compute cluster prototypes (medoids) for visualization and exploration.

A prototype is the most "representative" sample from each cluster,
typically the sample closest to the cluster centroid.
"""

try:
    import numpy as np
except ImportError:
    np = None

from pathlib import Path
from typing import Dict, List


class PrototypeSelector:
    """
    Select representative samples (prototypes/medoids) from each cluster.
    """

    def __init__(self, method: str = "medoid"):
        """
        Args:
            method: 'medoid' (closest to center) or 'diverse' (k diverse samples)
        """
        self.method = method

    def select_prototypes(
        self,
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        cluster_centers: np.ndarray,
        image_paths: List[Path],
        k: int = 5,
    ) -> Dict[int, List[Path]]:
        """
        Select k prototypes per cluster.

        Args:
            embeddings: (N, D) embeddings
            cluster_assignments: (N,) cluster IDs
            cluster_centers: (K, D) cluster centroids
            image_paths: List of N image paths
            k: Number of prototypes per cluster

        Returns:
            Dictionary: cluster_id -> [prototype_path1, ...]
        """
        n_clusters = len(cluster_centers)
        prototypes = {}

        for cluster_id in range(n_clusters):
            mask = cluster_assignments == cluster_id
            if mask.sum() == 0:
                continue

            cluster_embs = embeddings[mask]
            cluster_paths = [image_paths[i] for i in np.where(mask)[0]]

            if self.method == "medoid":
                # Find closest to centroid
                center = cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_embs - center, axis=1)
                top_k_indices = np.argsort(distances)[:k]
                prototypes[cluster_id] = [cluster_paths[i] for i in top_k_indices]

            elif self.method == "diverse":
                # Greedy farthest-first selection
                selected = []
                selected_embs = []

                # Start with medoid
                center = cluster_centers[cluster_id]
                distances = np.linalg.norm(cluster_embs - center, axis=1)
                first_idx = np.argmin(distances)
                selected.append(first_idx)
                selected_embs.append(cluster_embs[first_idx])

                # Greedily add farthest from selected set
                for _ in range(k - 1):
                    if len(selected) >= len(cluster_embs):
                        break

                    # Compute min distance to selected set
                    min_dists = np.full(len(cluster_embs), np.inf)
                    for sel_emb in selected_embs:
                        dists = np.linalg.norm(cluster_embs - sel_emb, axis=1)
                        min_dists = np.minimum(min_dists, dists)

                    # Mask already selected
                    min_dists[selected] = -np.inf

                    # Add farthest
                    next_idx = np.argmax(min_dists)
                    selected.append(next_idx)
                    selected_embs.append(cluster_embs[next_idx])

                prototypes[cluster_id] = [cluster_paths[i] for i in selected]

        return prototypes

    def select_single_medoid(
        self,
        embeddings: np.ndarray,
        cluster_assignments: np.ndarray,
        cluster_centers: np.ndarray,
        image_paths: List[Path],
    ) -> Dict[int, Path]:
        """
        Select single best representative per cluster.

        Args:
            embeddings: (N, D)
            cluster_assignments: (N,)
            cluster_centers: (K, D)
            image_paths: List[Path]

        Returns:
            Dictionary: cluster_id -> medoid_path
        """
        n_clusters = len(cluster_centers)
        medoids = {}

        for cluster_id in range(n_clusters):
            mask = cluster_assignments == cluster_id
            if mask.sum() == 0:
                continue

            cluster_embs = embeddings[mask]
            cluster_paths = [image_paths[i] for i in np.where(mask)[0]]

            center = cluster_centers[cluster_id]
            distances = np.linalg.norm(cluster_embs - center, axis=1)
            medoid_idx = np.argmin(distances)

            medoids[cluster_id] = cluster_paths[medoid_idx]

        return medoids
