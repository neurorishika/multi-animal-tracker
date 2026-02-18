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

from .utils import _auto_export_timm_model, _empty_appearance_result


class TimmOnnxBackend:
    """ONNX Runtime backend for TIMM appearance embeddings."""

    def __init__(self, config: AppearanceRuntimeConfig):
        self.config = config
        self._session = None
        self._dimension = None
        self._input_shape = None
        self._logger = logging.getLogger(f"{__name__}.TimmOnnxBackend")

        # Import dependencies
        try:
            import onnxruntime as ort
            import timm
            from PIL import Image

            self._ort = ort
            self._Image = Image
            self._timm = timm
        except ImportError as e:
            raise ImportError(
                "onnxruntime, PIL, and timm required for TimmOnnxBackend"
            ) from e

        # Resolve device using global runtime logic
        device_request = _resolve_device(config.device, "timm")

        # Map to ONNX providers
        if "cuda" in device_request:
            self._providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self._providers = ["CPUExecutionProvider"]

        self._logger.info(f"TIMM ONNX backend will use providers: {self._providers}")

    @property
    def output_dimension(self) -> int:
        if self._dimension is None:
            self.warmup()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def warmup(self) -> None:
        """Load ONNX model and initialize session."""
        if self._session is not None:
            return

        self._logger.info(f"Loading TIMM ONNX model: {self.config.model_name}")

        # Export model if needed
        exported_path, dimension = _auto_export_timm_model(
            self.config, "onnx", runtime_device=self.config.device
        )
        self._dimension = dimension

        # Create ONNX session
        sess_options = self._ort.SessionOptions()
        sess_options.graph_optimization_level = (
            self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self._session = self._ort.InferenceSession(
            exported_path,
            sess_options=sess_options,
            providers=self._providers,
        )

        # Get input shape
        input_meta = self._session.get_inputs()[0]
        self._input_shape = input_meta.shape

        self._logger.info(
            f"TIMM ONNX model loaded. Embedding dimension: {self._dimension}, "
            f"input shape: {self._input_shape}"
        )

        # Load TIMM transforms for preprocessing
        model_name = self.config.model_name
        if model_name.startswith("timm/"):
            model_name = model_name[5:]

        import timm

        dummy_model = timm.create_model(model_name, pretrained=False, num_classes=0)
        data_config = timm.data.resolve_model_data_config(dummy_model)
        self._transform = timm.data.create_transform(**data_config)
        del dummy_model

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
        if self._session is None:
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
                    batch_tensors.append(tensor.numpy())

                except Exception as e:
                    self._logger.warning(f"Failed to preprocess crop: {e}")
                    batch_tensors.append(None)

            # Filter out failed crops
            valid_indices = [i for i, t in enumerate(batch_tensors) if t is not None]
            valid_tensors = [batch_tensors[i] for i in valid_indices]

            if not valid_tensors:
                results.extend(
                    [
                        _empty_appearance_result(
                            self.config.model_name, self._dimension
                        )
                        for _ in batch_crops
                    ]
                )
                continue

            # Run ONNX inference
            batch_input = np.stack(valid_tensors, axis=0).astype(np.float32)

            outputs = self._session.run(None, {"input": batch_input})
            embeddings = outputs[0]

            # Optional L2 normalization
            if self.config.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)  # Avoid division by zero
                embeddings = embeddings / norms

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
        if self._session is not None:
            del self._session
            self._session = None
        self._logger.info("TIMM ONNX backend closed")
