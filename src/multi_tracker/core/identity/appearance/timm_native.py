from __future__ import annotations

import logging
from typing import List, Sequence

import cv2
import numpy as np

from multi_tracker.core.identity.runtime_types import (
    AppearanceResult,
    AppearanceRuntimeConfig,
)
from multi_tracker.core.identity.runtime_utils import _resolve_device

from .utils import _empty_appearance_result


class TimmNativeBackend:
    """TIMM-based appearance embedding backend using native PyTorch."""

    def __init__(self, config: AppearanceRuntimeConfig):
        self.config = config
        self._embedder = None
        self._dimension = None
        self._device = None
        self._transform = None
        self._logger = logging.getLogger(f"{__name__}.TimmNativeBackend")

        # Import dependencies
        try:
            import timm
            import torch
            from PIL import Image

            self._timm = timm
            self._torch = torch
            self._Image = Image
        except ImportError as e:
            raise ImportError(
                "timm, torch, and PIL are required for TimmNativeBackend"
            ) from e

        # Resolve device using global runtime logic
        self._device = _resolve_device(config.device, "timm")
        self._logger.info(f"TIMM appearance backend will use device: {self._device}")

    @property
    def output_dimension(self) -> int:
        if self._dimension is None:
            self.warmup()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def warmup(self) -> None:
        """Load TIMM model and initialize transforms."""
        if self._embedder is not None:
            return

        self._logger.info(f"Loading TIMM model: {self.config.model_name}")

        try:
            # Load model
            model = self._timm.create_model(
                self.config.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head to get embeddings
            )
            model = model.eval().to(self._device)
            self._embedder = model

            # Get data config and create transform
            data_config = self._timm.data.resolve_model_data_config(model)
            self._transform = self._timm.data.create_transform(**data_config)

            # Get embedding dimension
            with self._torch.no_grad():
                dummy_input = self._torch.randn(1, 3, 224, 224).to(self._device)
                dummy_output = self._embedder(dummy_input)
                self._dimension = dummy_output.shape[1]

            self._logger.info(
                f"TIMM model loaded. Embedding dimension: {self._dimension}"
            )

        except Exception as e:
            self._logger.error(f"Failed to load TIMM model: {e}")
            raise

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess a crop for TIMM inference."""
        if crop is None or crop.size == 0:
            raise ValueError("Empty crop")

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Optional CLAHE enhancement
        if self.config.use_clahe:
            lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            crop_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Resize if needed
        h, w = crop_rgb.shape[:2]
        max_side = self.config.max_image_side
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            crop_rgb = cv2.resize(
                crop_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        return crop_rgb

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[AppearanceResult]:
        """Run inference on a batch of crops."""
        if self._embedder is None:
            self.warmup()

        if not crops:
            return []

        results = []
        batch_size = self.config.batch_size

        for batch_start in range(0, len(crops), batch_size):
            batch_crops = crops[batch_start : batch_start + batch_size]
            batch_tensors = []

            for crop in batch_crops:
                try:
                    # Preprocess crop
                    processed = self._preprocess_crop(crop)

                    # Convert to PIL Image and apply TIMM transforms
                    pil_img = self._Image.fromarray(processed)
                    tensor = self._transform(pil_img)
                    batch_tensors.append(tensor)

                except Exception as e:
                    self._logger.warning(f"Failed to preprocess crop: {e}")
                    batch_tensors.append(None)

            # Filter out failed preprocessed crops
            valid_indices = [i for i, t in enumerate(batch_tensors) if t is not None]
            valid_tensors = [batch_tensors[i] for i in valid_indices]

            if not valid_tensors:
                # All crops failed preprocessing
                results.extend(
                    [
                        _empty_appearance_result(
                            self.config.model_name, self._dimension
                        )
                        for _ in batch_crops
                    ]
                )
                continue

            # Run inference
            batch_input = self._torch.stack(valid_tensors).to(self._device)

            with self._torch.no_grad():
                features = self._embedder(batch_input)

                # Optional L2 normalization
                if self.config.normalize_embeddings:
                    features = self._torch.nn.functional.normalize(features, p=2, dim=1)

                embeddings = features.cpu().numpy()

            # Map embeddings back to original batch positions
            embedding_idx = 0
            for i, tensor in enumerate(batch_tensors):
                if tensor is None:
                    results.append(
                        _empty_appearance_result(
                            self.config.model_name, self._dimension
                        )
                    )
                else:
                    results.append(
                        AppearanceResult(
                            embedding=embeddings[embedding_idx],
                            dimension=self._dimension,
                            model_name=self.config.model_name,
                        )
                    )
                    embedding_idx += 1

        return results

    def close(self) -> None:
        """Release backend resources."""
        if self._embedder is not None:
            del self._embedder
            self._embedder = None

            # Clear CUDA cache if using GPU
            if "cuda" in str(self._device):
                if hasattr(self._torch.cuda, "empty_cache"):
                    self._torch.cuda.empty_cache()

        self._logger.info("TIMM appearance backend closed")
