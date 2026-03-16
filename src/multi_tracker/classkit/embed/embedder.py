import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List

import numpy as np
from tqdm import tqdm

try:
    import timm
    import torch
    from PIL import Image
except ImportError:
    torch = None
    timm = None
    Image = None

from ...utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE
from .embedder_base import EmbedderBase


class ModelLoadError(RuntimeError):
    """Raised when pretrained timm weights cannot be loaded."""


_REMOTE_WEIGHT_ERROR_TOKENS = (
    "network is unreachable",
    "temporary failure in name resolution",
    "name or service not known",
    "connection error",
    "max retries exceeded",
    "offline mode",
    "client has been closed",
    "local cache",
    "disk cache",
    "localentrynotfounderror",
)


@contextmanager
def _temporary_env_var(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _exception_chain_text(error: BaseException) -> str:
    parts = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def _looks_like_remote_weight_error(error: BaseException) -> bool:
    text = _exception_chain_text(error)
    return any(token in text for token in _REMOTE_WEIGHT_ERROR_TOKENS)


def resolve_embedder_device(device: str) -> str:
    """Normalize requested device against current runtime availability."""
    resolved = str(device or "cpu").strip().lower()
    if resolved == "cuda" and not TORCH_CUDA_AVAILABLE:
        if MPS_AVAILABLE:
            return "mps"
        return "cpu"
    return resolved


class TimmEmbedder(EmbedderBase):
    def __init__(
        self, model_name: str = "vit_base_patch14_dinov2.lvd142m", device: str = "cuda"
    ):
        if torch is None or timm is None:
            raise ImportError("torch and timm are required for TimmEmbedder")

        self.model_name = model_name
        self.device = resolve_embedder_device(device)
        self.model = None
        self.transform = None
        self._dim = None

    def load_model(self):
        print(f"Loading model: {self.model_name} on {self.device}")

        try:
            self.model = self._create_model(local_files_only=False)
        except Exception as exc:
            if not _looks_like_remote_weight_error(exc):
                raise
            try:
                self.model = self._create_model(local_files_only=True)
            except Exception as offline_exc:
                raise ModelLoadError(
                    self._format_pretrained_load_error(offline_exc)
                ) from offline_exc

        self._configure_loaded_model()

    def _create_model(self, *, local_files_only: bool):
        if local_files_only:
            with _temporary_env_var("HF_HUB_OFFLINE", "1"):
                return timm.create_model(
                    self.model_name, pretrained=True, num_classes=0
                )
        return timm.create_model(self.model_name, pretrained=True, num_classes=0)

    def _configure_loaded_model(self) -> None:
        self.model.to(self.device)
        self.model.eval()

        # Get data config for transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # Infer dimension
        input_size = data_config.get("input_size", (3, 224, 224))
        # input_size is typically (C, H, W)
        self.input_size = (input_size[1], input_size[2])
        H, W = self.input_size

        with torch.no_grad():
            dummy = torch.randn(1, 3, H, W).to(self.device)
            out = self.model(dummy)
            self._dim = out.shape[1]

    def _format_pretrained_load_error(self, error: BaseException) -> str:
        detail = str(error).strip() or type(error).__name__
        return (
            f"Could not load pretrained weights for '{self.model_name}'. "
            "ClassKit could not reach Hugging Face and no cached local copy was available. "
            "Connect to the internet once to download this model, or choose a model whose weights are already cached locally."
            f"\n\nDevice: {self.device}\nDetails: {detail}"
        )

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self.load_model()
        return self._dim

    def embed(
        self, image_paths: List[Path], batch_size: int = 32, preprocess_fn=None
    ) -> np.ndarray:
        if self.model is None:
            self.load_model()

        embeddings = []

        # Simple dataset/loader
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Embedding"):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    if preprocess_fn is not None:
                        img = preprocess_fn(Path(p), img)
                    batch_tensors.append(self.transform(img))
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                    # Skip failed images to avoid shape mismatch
                    continue

            if not batch_tensors:
                continue

            batch_input = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch_input)
                # Normalize descriptors? Often good for retrieval
                # features = torch.nn.functional.normalize(features, p=2, dim=1)
                embeddings.append(features.cpu().numpy())

        if not embeddings:
            return np.zeros((0, self.dimension))

        return np.concatenate(embeddings, axis=0)
