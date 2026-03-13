from warnings import warn

import numpy as np

try:
    import umap
except ImportError:
    umap = None


class UMAPReducer:
    """
    Wrapper for UMAP dimensionality reduction.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "cosine",
    ):
        """
        Initialize UMAP reducer.

        Args:
            n_neighbors: Size of local neighborhood (default: 15)
            min_dist: Minimum distance between points in embedding (default: 0.1)
            n_components: Number of dimensions to reduce to (default: 2)
            metric: Distance metric (default: "cosine" for deep features)
        """
        if umap is None:
            raise ImportError(
                "umap-learn not installed. Please install: pip install umap-learn"
            )

        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP and transform embeddings to lower dimensions.

        Args:
            embeddings: Input embeddings (N, D)

        Returns:
            Reduced embeddings (N, n_components)
        """
        # Adjust n_neighbors if needed
        n_neighbors = self.n_neighbors
        if len(embeddings) < n_neighbors:
            warn(
                f"n_neighbors ({n_neighbors}) > n_samples ({len(embeddings)}). "
                f"Reducing n_neighbors to {max(2, len(embeddings) - 1)}"
            )
            n_neighbors = max(2, len(embeddings) - 1)

        # Create and fit UMAP
        self.reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
            verbose=False,
        )

        # Transform
        return self.reducer.fit_transform(embeddings)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using fitted UMAP.

        Args:
            embeddings: Input embeddings (N, D)

        Returns:
            Reduced embeddings (N, n_components)
        """
        if self.reducer is None:
            raise RuntimeError("Must call fit_transform() before transform()")

        return self.reducer.transform(embeddings)


def compute_umap_viz(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce embeddings to 2D for visualization.
    Shape: (N, n_components)
    """
    reducer = UMAPReducer(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components
    )
    return reducer.fit_transform(embeddings)
