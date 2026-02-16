try:
    import timm
    import torch
    from PIL import Image
except ImportError:
    torch = None
    timm = None
    Image = None

from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from ...utils.gpu_utils import MPS_AVAILABLE, TORCH_CUDA_AVAILABLE
from .embedder_base import EmbedderBase

DEFAULT_MODELS = [
    # DINO v2 models (recommended for general image understanding)
    "vit_base_patch14_dinov2.lvd142m",  # ~86M params, excellent quality
    "vit_small_patch14_dinov2.lvd142m",  # ~22M params, good balance
    "vit_large_patch14_dinov2.lvd142m",  # ~304M params, best quality
    "vit_giant_patch14_dinov2.lvd142m",  # ~1.1B params, huge model
    # CLIP models (text+image understanding, good for semantic search)
    "vit_base_patch32_clip_224.openai",  # ~88M params, good balance
    "vit_bigG_14_clip_224.laion400M_e32",  # ~2B params, best quality
    "convnext_base_w_clip.laion2b_s29B_b131k_ft_in1k",  # ~88M params, efficient
    # ResNet models (faster, smaller)
    "resnet50.a1_in1k",  # ~25M params, fast
    "resnet18.a1_in1k",  # ~11M params, very fast
    "resnet101.a1_in1k",  # ~44M params, more capacity
    # EfficientNet models (efficient, good accuracy/speed tradeoff)
    "efficientnet_b0.ra_in1k",  # ~5M params, very efficient
    "efficientnet_b3.ra2_in1k",  # ~12M params, good balance
    "efficientnet_b5.sw_in12k_ft_in1k",  # ~30M params, high quality
    # MobileNet models (fastest, smallest)
    "mobilenetv3_small_100.lamb_in1k",  # ~2M params, mobile
    "mobilenetv3_large_100.ra_in1k",  # ~5M params, mobile
    # ConvNeXt models (modern convnets)
    "convnext_tiny.fb_in1k",  # ~28M params, modern
    "convnext_base.fb_in1k",  # ~88M params, powerful
]


class TimmEmbedder(EmbedderBase):
    def __init__(
        self, model_name: str = "vit_base_patch14_dinov2.lvd142m", device: str = "cuda"
    ):
        if torch is None or timm is None:
            raise ImportError("torch and timm are required for TimmEmbedder")

        # Resolve device request against availability flags from gpu_utils
        if device == "cuda" and not TORCH_CUDA_AVAILABLE:
            if MPS_AVAILABLE:
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.model = None
        self.transform = None
        self._dim = None

    def load_model(self):
        print(f"Loading model: {self.model_name} on {self.device}")

        # Load model with correct number of classes for feature extraction
        self.model = timm.create_model(
            self.model_name, pretrained=True, num_classes=0
        )  # num_classes=0 for embeddings
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

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self.load_model()
        return self._dim

    def embed(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
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
