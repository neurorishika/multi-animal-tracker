"""CNN identity backend for MAT: config, predictions, cache, and inference backend.

Pure Python — no Qt dependency.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CNNIdentityConfig:
    """Configuration for CNN Classifier identity method."""

    model_path: str = ""
    confidence: float = 0.5
    label: str = ""
    batch_size: int = 64
    crop_padding: float = 0.1
    match_bonus: float = 20.0
    mismatch_penalty: float = 50.0
    window: int = 10


@dataclass
class ClassPrediction:
    """Per-detection CNN classification result for one frame."""

    class_name: str | None  # None if below confidence threshold
    confidence: float
    det_index: int  # which detection slot in the frame


# ---------------------------------------------------------------------------
# CNNIdentityCache
# ---------------------------------------------------------------------------

_SENTINEL_NONE = "__NONE__"  # stored in npz when class_name is None


class CNNIdentityCache:
    """Persistent .npz cache of per-frame CNN identity predictions.

    Data is accumulated in memory via ``save()`` and written to disk in a
    single compressed write via ``flush()``.  Call ``load()`` during the
    tracking loop to retrieve per-frame predictions.
    """

    def __init__(self, cache_path: str | Path) -> None:
        self._path = Path(cache_path)
        self._data: dict[str, Any] = {}
        if self._path.exists():
            raw = np.load(str(self._path), allow_pickle=True)
            self._data = dict(raw)

    def exists(self) -> bool:
        return self._path.exists() or bool(self._data)

    def save(self, frame_idx: int, predictions: list[ClassPrediction]) -> None:
        """Update in-memory cache for *frame_idx*. Call flush() when done."""
        if not predictions:
            self._data[f"f{frame_idx}_det"] = np.array([], dtype=np.int32)
            self._data[f"f{frame_idx}_cls"] = np.array([], dtype=object)
            self._data[f"f{frame_idx}_conf"] = np.array([], dtype=np.float32)
        else:
            det_arr = np.array([p.det_index for p in predictions], dtype=np.int32)
            cls_arr = np.array(
                [
                    p.class_name if p.class_name is not None else _SENTINEL_NONE
                    for p in predictions
                ],
                dtype=object,
            )
            conf_arr = np.array([p.confidence for p in predictions], dtype=np.float32)
            self._data[f"f{frame_idx}_det"] = det_arr
            self._data[f"f{frame_idx}_cls"] = cls_arr
            self._data[f"f{frame_idx}_conf"] = conf_arr

    def flush(self) -> None:
        """Write all in-memory predictions to disk."""
        if not self._data:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(self._path), **self._data)

    def load(self, frame_idx: int) -> list[ClassPrediction]:
        """Return saved predictions for *frame_idx*, or [] if not found."""
        key_det = f"f{frame_idx}_det"
        if key_det not in self._data:
            return []
        det_arr = self._data[key_det]
        cls_arr = self._data[f"f{frame_idx}_cls"]
        conf_arr = self._data[f"f{frame_idx}_conf"]
        results = []
        for i in range(len(det_arr)):
            raw_cls = str(cls_arr[i])
            class_name = None if raw_cls == _SENTINEL_NONE else raw_cls
            results.append(
                ClassPrediction(
                    class_name=class_name,
                    confidence=float(conf_arr[i]),
                    det_index=int(det_arr[i]),
                )
            )
        return results


# ---------------------------------------------------------------------------
# CNNIdentityBackend
# ---------------------------------------------------------------------------


class CNNIdentityBackend:
    """Wraps model loading and batch inference for CNN identity classification.

    Supports:
    - .pth checkpoints (TinyClassifier or torchvision — detected via 'arch' field)
    - YOLO .pt checkpoints (ultralytics)
    Runtime selection via compute_runtime (cpu/mps/cuda). ONNX artifacts are
    derived lazily from .pth and cached alongside the source file.
    """

    def __init__(
        self,
        config: CNNIdentityConfig,
        model_path: str,
        compute_runtime: str = "cpu",
    ) -> None:
        self._config = config
        self._model_path = str(model_path)
        self._compute_runtime = str(compute_runtime or "cpu")
        self._model = None
        self._class_names: list[str] = []
        self._input_size: tuple[int, int] = (224, 224)
        self._arch: str = "tinyclassifier"
        self._is_yolo: bool = self._model_path.endswith(".pt")
        self._loaded: bool = False
        self._infer_fn = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        use_onnx = self._compute_runtime in (
            "onnx_cpu",
            "onnx_cuda",
            "onnx_rocm",
            "tensorrt",
        )
        device = self._torch_device(self._compute_runtime)

        if self._is_yolo:
            self._load_yolo(use_onnx, device)
        else:
            self._load_pth(use_onnx, device)
        self._loaded = True

    def _torch_device(self, rt: str) -> str:
        if rt in ("cuda", "onnx_cuda", "tensorrt"):
            return "cuda"
        if rt == "mps":
            return "mps"
        if rt in ("rocm", "onnx_rocm"):
            return "cuda"
        return "cpu"

    def _load_pth(self, use_onnx: bool, device: str) -> None:
        import torch

        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
        self._class_names = ckpt.get("class_names", [])
        raw_size = ckpt.get("input_size", (224, 224))
        self._input_size = (
            tuple(raw_size)
            if isinstance(raw_size, (list, tuple))
            else (raw_size, raw_size)
        )
        self._arch = ckpt.get("arch", "tinyclassifier")

        if use_onnx:
            onnx_path = self._derive_onnx(ckpt, device)
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "cuda" in self._compute_runtime
                else ["CPUExecutionProvider"]
            )
            self._model = ort.InferenceSession(onnx_path, providers=providers)
            self._infer_fn = self._infer_onnx
        else:
            if self._arch == "tinyclassifier":
                from multi_tracker.training.tiny_model import load_tiny_classifier

                self._model = load_tiny_classifier(self._model_path, device=device)
            else:
                # Requires Spec A (ClassKit Extended Training) to be implemented first
                from multi_tracker.training.torchvision_model import (
                    load_torchvision_classifier,
                )

                self._model, _ = load_torchvision_classifier(
                    self._model_path, device=device
                )
            self._infer_fn = lambda batch_np, dev=device: self._infer_torch(
                batch_np, dev
            )

    def _load_yolo(self, use_onnx: bool, device: str) -> None:
        from ultralytics import YOLO

        yolo = YOLO(self._model_path)
        names = yolo.names
        self._class_names = [names[i] for i in sorted(names.keys())]
        self._input_size = (224, 224)
        self._arch = "yolo"
        if use_onnx:
            onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
            if not os.path.exists(onnx_path):
                yolo.export(format="onnx", imgsz=224)
            import onnxruntime as ort

            self._model = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )
            self._infer_fn = self._infer_onnx
        else:
            self._model = yolo
            self._infer_fn = self._infer_yolo

    def _derive_onnx(self, ckpt: dict, device: str) -> str:
        """Lazy-derive ONNX from .pth. Returns path to .onnx file."""
        onnx_path = str(Path(self._model_path).with_suffix(".onnx"))
        if os.path.exists(onnx_path):
            return onnx_path
        if self._arch == "tinyclassifier":
            import torch

            from multi_tracker.training.tiny_model import load_tiny_classifier

            model = load_tiny_classifier(self._model_path, device="cpu")
            h, w = self._input_size
            dummy = torch.zeros(1, 3, h, w)
            torch.onnx.export(
                model,
                dummy,
                onnx_path,
                opset_version=17,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            )
        else:
            from multi_tracker.training.torchvision_model import (
                export_torchvision_to_onnx,
                load_torchvision_classifier,
            )

            model, loaded_ckpt = load_torchvision_classifier(
                self._model_path, device="cpu"
            )
            export_torchvision_to_onnx(model, loaded_ckpt, onnx_path)
        return onnx_path

    def _infer_torch(self, batch_np: np.ndarray, device: str) -> np.ndarray:
        import torch

        t = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            logits = self._model(t).cpu().numpy()
        return logits

    def _infer_onnx(self, batch_np: np.ndarray) -> np.ndarray:
        input_name = self._model.get_inputs()[0].name
        return self._model.run(None, {input_name: batch_np.astype(np.float32)})[0]

    def _infer_yolo(self, crops: list[np.ndarray]) -> np.ndarray:
        # YOLO classify expects list of numpy arrays in HWC uint8 format
        results = self._model(crops, verbose=False)
        probs = np.array([r.probs.data.cpu().numpy() for r in results])
        return np.log(np.clip(probs, 1e-9, 1.0))

    def _preprocess(self, crops: list[np.ndarray]) -> np.ndarray:
        """Resize and normalize crops to the model's expected input format."""
        import cv2
        from PIL import Image
        from torchvision import transforms

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        h, w = self._input_size
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                tensors.append(np.zeros((3, h, w), dtype=np.float32))
                continue
            img_bgr = cv2.resize(crop, (w, h))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensors.append(tf(pil_img).numpy())
        return np.stack(tensors).astype(np.float32)

    def predict_batch(self, crops: list[np.ndarray]) -> list[ClassPrediction]:
        """Run inference on *crops*. Returns one ClassPrediction per crop."""
        if not crops:
            return []
        self._ensure_loaded()
        # YOLO native inference does its own preprocessing — pass raw crops
        if self._is_yolo and self._compute_runtime not in (
            "onnx_cpu",
            "onnx_cuda",
            "onnx_rocm",
            "tensorrt",
        ):
            logits = self._infer_fn(crops)
        else:
            batch_np = self._preprocess(crops)
            logits = self._infer_fn(batch_np)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        results = []
        for i, prob in enumerate(probs):
            best_idx = int(np.argmax(prob))
            best_conf = float(prob[best_idx])
            if best_conf >= self._config.confidence and self._class_names:
                class_name = self._class_names[best_idx]
            else:
                class_name = None
            results.append(
                ClassPrediction(
                    class_name=class_name,
                    confidence=best_conf,
                    det_index=i,
                )
            )
        return results

    def close(self) -> None:
        self._model = None
        self._infer_fn = None
        self._loaded = False
