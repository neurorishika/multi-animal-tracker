from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class EmbedderBase(ABC):
    """
    Abstract base class for image embedding models.
    Implementations should wrap different backends (clip, vit, etc).
    """

    @abstractmethod
    def load_model(self):
        """Prepare backend (load weights)."""
        pass

    @abstractmethod
    def embed(self, image_paths: List[Path]) -> np.ndarray:
        """
        Embed a list of images to a numpy array (N, D).
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension D."""
        pass
