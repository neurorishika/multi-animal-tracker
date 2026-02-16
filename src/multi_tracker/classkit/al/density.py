"""
Density estimation utilities for active learning.

Used for representativeness and cluster-based selection.
"""

try:
    import faiss
    import numpy as np
except ImportError:
    np = None
    faiss = None

from typing import Optional


def compute_knn_density(
    embeddings: np.ndarray, k: int = 10, return_indices: bool = False
) -> np.ndarray:
    """
    Compute density proxy as mean distance to k nearest neighbors.

    Lower distance = higher density (more "typical" samples).
    Higher distance = lower density (more "unusual" samples).

    Args:
        embeddings: (N, D) embeddings
        k: Number of neighbors
        return_indices: If True, also return neighbor indices

    Returns:
        densities: (N,) mean kNN distances
        (optional) indices: (N, k) neighbor indices
    """
    if faiss is None:
        raise ImportError("faiss required for density computation")

    N, D = embeddings.shape
    embeddings_f32 = np.ascontiguousarray(embeddings.astype(np.float32))

    # Build index
    index = faiss.IndexFlatL2(D)
    index.add(embeddings_f32)

    # Query k+1 to exclude self
    k_search = min(k + 1, N)
    distances, indices = index.search(embeddings_f32, k_search)

    # Exclude self (first neighbor, distance ~0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Mean distance as density proxy
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
