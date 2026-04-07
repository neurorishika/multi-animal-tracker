"""
Density estimation utilities for active learning.

Used for representativeness and cluster-based selection.
Backed by hnswlib (if available) with a numpy brute-force fallback.
"""

from __future__ import annotations

import importlib.util
from typing import Optional, Tuple, Union

import numpy as np


def compute_knn_density(
    embeddings: np.ndarray,
    k: int = 10,
    return_indices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute density proxy as mean distance to k nearest neighbors.

    Lower distance = higher density (more "typical" samples).
    Higher distance = lower density (more "unusual" samples).

    Args:
        embeddings: (N, D) embeddings
        k: Number of neighbors
        return_indices: If True, also return neighbor indices

    Returns:
        densities: (N,) mean kNN distances
        indices (optional): (N, k) neighbor indices
    """
    N, D = embeddings.shape
    emb = np.ascontiguousarray(embeddings.astype(np.float32))
    k_search = min(k + 1, N)  # +1 to exclude self

    if importlib.util.find_spec("hnswlib") is not None:
        import hnswlib

        idx = hnswlib.Index(space="l2", dim=D)
        idx.init_index(max_elements=N, ef_construction=100, M=16)
        idx.add_items(emb)
        idx.set_ef(max(k_search, 50))
        indices, distances = idx.knn_query(emb, k=k_search)
        # hnswlib returns (labels, distances); exclude self (col 0)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        # numpy brute-force
        dists_all = np.linalg.norm(emb[:, None] - emb[None], axis=2)
        order = np.argsort(dists_all, axis=1)
        indices = order[:, 1:k_search]
        distances = np.take_along_axis(dists_all, indices, axis=1)

    densities = distances.mean(axis=1)

    if return_indices:
        return densities, indices
    return densities


def compute_cluster_densities(
    embeddings: np.ndarray, cluster_assignments: np.ndarray, k: int = 10
) -> np.ndarray:
    """
    Compute per-cluster average density.

    Args:
        embeddings: (N, D)
        cluster_assignments: (N,)
        k: kNN k

    Returns:
        cluster_densities: (K,) mean density per cluster
    """
    point_densities = compute_knn_density(embeddings, k=k)

    n_clusters = cluster_assignments.max() + 1
    cluster_densities = np.zeros(n_clusters)

    for cluster_id in range(n_clusters):
        mask = cluster_assignments == cluster_id
        if mask.sum() > 0:
            cluster_densities[cluster_id] = point_densities[mask].mean()

    return cluster_densities


def select_diverse_samples(
    embeddings: np.ndarray, n_samples: int, seed: Optional[int] = None
) -> np.ndarray:
    """
    Greedy farthest-first selection (k-center greedy).

    Args:
        embeddings: (N, D) embeddings
        n_samples: Number of samples to select
        seed: Random seed for first sample

    Returns:
        selected_indices: (n_samples,) indices
    """
    N = len(embeddings)
    if n_samples >= N:
        return np.arange(N)

    if seed is not None:
        np.random.seed(seed)

    # Start with random sample
    selected = [np.random.randint(N)]
    selected_embs = [embeddings[selected[0]]]

    # Greedily add farthest from selected set
    for _ in range(n_samples - 1):
        # Compute min distance to selected set
        min_dists = np.full(N, np.inf)
        for sel_emb in selected_embs:
            dists = np.linalg.norm(embeddings - sel_emb, axis=1)
            min_dists = np.minimum(min_dists, dists)

        # Mask already selected
        min_dists[selected] = -np.inf

        # Add farthest
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)
        selected_embs.append(embeddings[next_idx])

    return np.array(selected)
