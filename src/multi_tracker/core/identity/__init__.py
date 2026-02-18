"""Identity and individual-level analysis."""

from .analysis import IdentityProcessor, IndividualDatasetGenerator
from .appearance_cache import (
    AppearanceEmbeddingCache,
    build_appearance_cache_path,
    compute_appearance_embedding_id,
    compute_appearance_extractor_hash,
)
from .runtime_api import (
    AppearanceInferenceBackend,
    AppearanceResult,
    AppearanceRuntimeConfig,
    create_appearance_backend_from_config,
)

__all__ = [
    "IdentityProcessor",
    "IndividualDatasetGenerator",
    "AppearanceResult",
    "AppearanceRuntimeConfig",
    "AppearanceInferenceBackend",
    "create_appearance_backend_from_config",
    "AppearanceEmbeddingCache",
    "compute_appearance_extractor_hash",
    "compute_appearance_embedding_id",
    "build_appearance_cache_path",
]
