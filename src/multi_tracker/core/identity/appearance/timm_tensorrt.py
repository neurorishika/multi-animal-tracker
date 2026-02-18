from __future__ import annotations

import logging
from typing import List, Sequence

import cv2
import numpy as np

from multi_tracker.core.identity.runtime_types import (
    AppearanceResult,
    AppearanceRuntimeConfig,
)

from .utils import _auto_export_timm_model, _empty_appearance_result


class TimmTensorRTBackend:
    """TensorRT backend for TIMM appearance embeddings."""

    def __init__(self, config: AppearanceRuntimeConfig):
        self.config = config
        self._engine = None
        self._context = None
        self._dimension = None
        self._input_shape = None
        self._logger = logging.getLogger(f"{__name__}.TimmTensorRTBackend")

        # Import dependencies
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
            import timm
            from PIL import Image

            self._trt = trt
            self._cuda = cuda
            self._Image = Image
            self._timm = timm
        except ImportError as e:
            raise ImportError(
                "tensorrt, pycuda, PIL, and timm required for TimmTensorRTBackend"
            ) from e

        self._logger.info("TIMM TensorRT backend will use CUDA")
        self._stream = None
        self._bindings = []
        self._inputs = []
        self._outputs = []

    @property
    def output_dimension(self) -> int:
        if self._dimension is None:
            self.warmup()
        return self._dimension

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def warmup(self) -> None:
        """Load TensorRT engine and initialize context."""
        if self._engine is not None:
            return

        self._logger.info(f"Loading TIMM TensorRT engine: {self.config.model_name}")

        # Export model if needed
        exported_path, dimension = _auto_export_timm_model(
            self.config, "tensorrt", runtime_device="cuda"
        )
        self._dimension = dimension

        # Load TensorRT engine
        TRT_LOGGER = self._trt.Logger(self._trt.Logger.WARNING)
        runtime = self._trt.Runtime(TRT_LOGGER)

        with open(exported_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {exported_path}")

        self._context = self._engine.create_execution_context()
        self._stream = self._cuda.Stream()

        # Allocate buffers
        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            dtype = self._engine.get_tensor_dtype(tensor_name)
            shape = self._engine.get_tensor_shape(tensor_name)

            # Convert TRT dtype to numpy
            if dtype == self._trt.DataType.FLOAT:
                np_dtype = np.float32
            else:
                np_dtype = np.float32

            size = int(np.prod(shape))
            host_mem = self._cuda.pagelocked_empty(size, np_dtype)
            device_mem = self._cuda.mem_alloc(host_mem.nbytes)

            self._bindings.append(int(device_mem))

            if (
                self._engine.get_tensor_mode(tensor_name)
                == self._trt.TensorIOMode.INPUT
            ):
                self._inputs.append(
                    {"host": host_mem, "device": device_mem, "shape": shape}
                )
                self._input_shape = shape
            else:
                self._outputs.append(
                    {"host": host_mem, "device": device_mem, "shape": shape}
                )

        self._logger.info(
            f"TIMM TensorRT engine loaded. Embedding dimension: {self._dimension}, "
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
        if self._engine is None:
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

            # Run TensorRT inference
            batch_input = np.stack(valid_tensors, axis=0).astype(np.float32)

            # Copy input to device
            np.copyto(self._inputs[0]["host"][: batch_input.size], batch_input.ravel())
            self._cuda.memcpy_htod_async(
                self._inputs[0]["device"], self._inputs[0]["host"], self._stream
            )

            # Run inference
            self._context.execute_async_v2(
                bindings=self._bindings, stream_handle=self._stream.handle
            )

            # Copy output from device
            self._cuda.memcpy_dtoh_async(
                self._outputs[0]["host"], self._outputs[0]["device"], self._stream
            )
            self._stream.synchronize()

            # Reshape output
            output_size = len(valid_tensors) * self._dimension
            embeddings = self._outputs[0]["host"][:output_size].reshape(
                len(valid_tensors), self._dimension
            )

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
        if self._context is not None:
            del self._context
            self._context = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        if self._stream is not None:
            del self._stream
            self._stream = None
        self._bindings = []
