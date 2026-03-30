"""Standalone head-tail direction analyzer.

Extracts the head-tail model loading and inference logic from
``core.detectors.engine.YOLOOBBDetector`` so that it can be used
outside of the detection pipeline (e.g. during interpolation).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Canonical class sets (shared with engine.py)
_HEADTAIL_DIRECTIONAL_CLASS_SET = frozenset({"left", "right"})
_HEADTAIL_FIVE_CLASS_SET = frozenset({"left", "right", "up", "down", "unknown"})

# Label aliases
_LABEL_ALIASES: Dict[str, str] = {
    "left": "left",
    "head_left": "left",
    "right": "right",
    "head_right": "right",
    "up": "up",
    "head_up": "up",
    "down": "down",
    "head_down": "down",
    "unknown": "unknown",
    "head_unknown": "unknown",
}


class HeadTailAnalyzer:
    """Run head-tail direction classification without a full YOLO detector."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        reference_aspect_ratio: float = 2.0,
        canonical_margin: float = 1.3,
        predict_device: Optional[str] = None,
    ) -> None:
        self._device = device
        self._conf_threshold = conf_threshold
        self._ref_ar = max(1.0, reference_aspect_ratio)
        self._canonical_margin = canonical_margin
        self._padding_fraction = max(0.0, canonical_margin - 1.0)
        self._predict_device = predict_device

        # Populated by _load_model
        self._backend: str = "none"  # "tiny", "classkit_tiny", "yolo"
        self._model = None
        self._class_names: Optional[List[str]] = None
        self._input_size: Optional[Tuple[int, int]] = None

        self._load_model(model_path)

    @classmethod
    def from_components(
        cls,
        model,
        backend: str,
        class_names: Optional[List[str]],
        input_size: Optional[Tuple[int, int]],
        device: str = "cpu",
        conf_threshold: float = 0.5,
        reference_aspect_ratio: float = 2.0,
        canonical_margin: float = 1.3,
        predict_device: Optional[str] = None,
    ) -> "HeadTailAnalyzer":
        """Create from a pre-loaded model, skipping file-based loading."""
        obj = cls.__new__(cls)
        obj._device = device
        obj._conf_threshold = conf_threshold
        obj._ref_ar = max(1.0, reference_aspect_ratio)
        obj._canonical_margin = canonical_margin
        obj._padding_fraction = max(0.0, canonical_margin - 1.0)
        obj._predict_device = predict_device
        obj._model = model
        obj._backend = backend
        obj._class_names = class_names
        obj._input_size = input_size
        return obj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._backend != "none"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    @property
    def input_size(self) -> Optional[Tuple[int, int]]:
        return self._input_size

    @property
    def model(self):
        return self._model

    def analyze_crops(
        self,
        frames: List[np.ndarray],
        per_frame_obb_corners: List[List[np.ndarray]],
    ) -> List[List[Tuple[float, float, int]]]:
        """Run head-tail analysis on multiple frames.

        Args:
            frames: BGR video frames.
            per_frame_obb_corners: For each frame, a list of (4,2) OBB corners.

        Returns:
            Per-frame list of ``(heading_radians, confidence, directed_flag)``
            tuples.  ``heading_radians`` is ``nan`` when direction is ambiguous.
            ``directed_flag`` is 1 when classifier was confident, 0 otherwise.
        """
        if not self.is_available:
            return [
                [(float("nan"), 0.0, 0)] * len(corners)
                for corners in per_frame_obb_corners
            ]

        # Phase 1: collect canonical crops across all frames
        all_crops: List[np.ndarray] = []
        all_meta: List[Tuple[int, int, float, np.ndarray]] = []
        for fi, (frame, corners_list) in enumerate(zip(frames, per_frame_obb_corners)):
            for di, corners in enumerate(corners_list):
                result = self._canonicalize_obb(frame, corners)
                if result is None:
                    continue
                crop, axis_theta, M_align = result
                all_crops.append(crop)
                all_meta.append((fi, di, float(axis_theta), M_align))

        # Pre-allocate results
        results: List[List[Tuple[float, float, int]]] = [
            [(float("nan"), 0.0, 0)] * len(c) for c in per_frame_obb_corners
        ]

        if not all_crops:
            return results

        # Phase 2: single GPU inference pass
        cls_results = self._predict(all_crops)
        if cls_results is None or len(cls_results) == 0:
            return results

        # Phase 3: scatter results
        TWO_PI = 2.0 * np.pi
        if self._backend == "tiny":
            probs = np.asarray(cls_results, dtype=np.float32).reshape(-1)
            n_eval = min(len(all_meta), len(probs))
            for j in range(n_eval):
                fi, di, axis_theta, _ = all_meta[j]
                p_right = float(probs[j])
                conf = max(p_right, 1.0 - p_right)
                if conf < self._conf_threshold:
                    results[fi][di] = (float("nan"), float(conf), 0)
                    continue
                theta = axis_theta if p_right >= 0.5 else (axis_theta + np.pi)
                results[fi][di] = (float(theta % TWO_PI), float(conf), 1)

        elif self._backend == "classkit_tiny":
            n_eval = min(len(all_meta), len(cls_results))
            for j in range(n_eval):
                fi, di, axis_theta, _ = all_meta[j]
                try:
                    direction, conf = cls_results[j]
                except Exception:
                    continue
                if direction not in {"left", "right"}:
                    results[fi][di] = (float("nan"), float(conf), 0)
                    continue
                if float(conf) < self._conf_threshold:
                    results[fi][di] = (float("nan"), float(conf), 0)
                    continue
                theta = axis_theta if direction == "right" else (axis_theta + np.pi)
                results[fi][di] = (float(theta % TWO_PI), float(conf), 1)

        else:  # yolo backend
            n_eval = min(len(all_meta), len(cls_results))
            for j in range(n_eval):
                fi, di, axis_theta, _ = all_meta[j]
                try:
                    result = cls_results[j]
                    if result is None:
                        continue
                    probs_obj = getattr(result, "probs", None)
                    if probs_obj is None:
                        continue
                    top1 = int(getattr(probs_obj, "top1", -1))
                    top1_conf = float(getattr(probs_obj, "top1conf", 0.0))
                    if top1 < 0 or top1_conf < self._conf_threshold:
                        results[fi][di] = (float("nan"), top1_conf, 0)
                        continue
                    label = self._label_from_top1(top1)
                    direction = self._class_to_direction(label, cls_idx=top1)
                    if direction is None:
                        results[fi][di] = (float("nan"), top1_conf, 0)
                        continue
                    theta = axis_theta if direction == "right" else (axis_theta + np.pi)
                    results[fi][di] = (float(theta % TWO_PI), top1_conf, 1)
                except Exception:
                    continue

        return results

    def close(self) -> None:
        self._model = None
        self._backend = "none"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path_str: str) -> None:
        if not model_path_str:
            return
        # Try tiny/classkit_tiny first
        tiny_result = self._try_load_tiny(model_path_str)
        if tiny_result is not None:
            model, class_names, input_size = tiny_result
            self._input_size = input_size
            if class_names is not None:
                self._class_names = self._validate_class_names(class_names)
                self._backend = "classkit_tiny"
            else:
                self._backend = "tiny"
            self._model = model
            logger.info(
                "HeadTailAnalyzer: loaded %s model from %s",
                self._backend,
                model_path_str,
            )
            return

        # Try YOLO classify model
        try:
            from ultralytics import YOLO

            model = YOLO(model_path_str, task="classify")
            model_names = getattr(model, "names", None)
            if model_names is None:
                model_names = getattr(getattr(model, "model", None), "names", None)
            self._class_names = self._validate_class_names(model_names)
            self._backend = "yolo"
            self._model = model
            logger.info(
                "HeadTailAnalyzer: loaded YOLO classify model from %s",
                model_path_str,
            )
        except Exception as exc:
            logger.warning("HeadTailAnalyzer: failed to load model: %s", exc)

    def _try_load_tiny(self, model_path_str: str):
        import torch

        model_path = Path(model_path_str).expanduser().resolve()
        if not model_path.exists():
            return None
        if model_path.suffix.lower() not in {".pth", ".pt"}:
            return None

        try:
            checkpoint = torch.load(
                str(model_path), map_location="cpu", weights_only=False
            )
        except Exception:
            return None

        state_dict = None
        input_size = (128, 64)
        class_names = None

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint.get("model_state_dict")
            maybe_size = checkpoint.get("input_size")
            if isinstance(maybe_size, (list, tuple)) and len(maybe_size) == 2:
                input_size = (int(maybe_size[0]), int(maybe_size[1]))
            raw_names = checkpoint.get("class_names")
            if isinstance(raw_names, (list, tuple)) and raw_names:
                class_names = [str(n) for n in raw_names]
        elif isinstance(checkpoint, (dict, OrderedDict)):
            state_dict = checkpoint
        else:
            return None

        if not isinstance(state_dict, (dict, OrderedDict)):
            return None
        keys = list(state_dict.keys())
        if not keys or not any(str(k).startswith("features.") for k in keys):
            return None

        linear_keys = sorted(
            [k for k in keys if k.startswith("classifier.") and k.endswith(".weight")],
            key=lambda k: int(k.split(".")[1]),
        )
        if not linear_keys:
            return None
        n_out = int(state_dict[linear_keys[-1]].shape[0])

        if n_out == 1:
            model = self._build_tiny_classifier(input_size=input_size)
            model.load_state_dict(state_dict, strict=True)
        else:
            try:
                from multi_tracker.training.tiny_model import rebuild_from_checkpoint

                model = rebuild_from_checkpoint({"model_state_dict": state_dict})
            except Exception as exc:
                logger.warning("Failed to load ClassKit tiny head-tail: %s", exc)
                return None

        import torch

        device = torch.device(self._device)
        model.to(device)
        model.eval()
        return model, class_names, input_size

    @staticmethod
    def _build_tiny_classifier(input_size=(128, 64)):
        import torch.nn as nn

        class _TinyHeadClassifier(nn.Module):
            def __init__(self, input_size=(128, 64)):
                super().__init__()
                self.input_size = tuple(input_size)
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return _TinyHeadClassifier(input_size=input_size)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(self, source_crops: List[np.ndarray]):
        if self._model is None or not source_crops:
            return []

        if self._backend == "tiny":
            import torch

            batch = self._crops_to_tensor(source_crops, self._input_size)
            device = torch.device(self._device)
            batch = batch.to(device)
            with torch.inference_mode():
                logits = self._model(batch)
                probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            return probs

        if self._backend == "classkit_tiny":
            import torch
            import torch.nn.functional as F

            batch = self._crops_to_tensor(source_crops, self._input_size)
            device = torch.device(self._device)
            batch = batch.to(device)
            with torch.inference_mode():
                logits = self._model(batch)
                softmax = F.softmax(logits, dim=1)
                top1_conf, top1_idx = softmax.max(dim=1)
                top1_conf = top1_conf.detach().cpu().numpy()
                top1_idx = top1_idx.detach().cpu().numpy()

            classified = []
            for cls_idx, conf in zip(top1_idx, top1_conf):
                label = self._label_from_top1(int(cls_idx))
                direction = self._class_to_direction(label, cls_idx=int(cls_idx))
                classified.append((direction, float(conf)))
            return classified

        # YOLO backend
        try:
            kwargs = dict(source=source_crops, conf=0.0, verbose=False)
            if self._predict_device is not None:
                kwargs["device"] = self._predict_device
            return self._model.predict(**kwargs)
        except Exception:
            outputs = []
            for crop in source_crops:
                try:
                    kw = dict(source=crop, conf=0.0, verbose=False)
                    if self._predict_device is not None:
                        kw["device"] = self._predict_device
                    one = self._model.predict(**kw)
                    outputs.append(one[0] if one else None)
                except Exception:
                    outputs.append(None)
            return outputs

    @staticmethod
    def _crops_to_tensor(source_crops, target_hw=None):
        import torch

        tensors = []
        for crop in source_crops:
            c = np.asarray(crop)
            if c.ndim == 2:
                c = np.stack([c, c, c], axis=-1)
            if c.ndim == 3 and c.shape[2] == 3:
                c = c[:, :, ::-1].copy()  # BGR → RGB
            if target_hw is not None:
                w, h = int(target_hw[0]), int(target_hw[1])
                if c.shape[1] != w or c.shape[0] != h:
                    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
            tensors.append(t)
        return torch.stack(tensors, dim=0)

    # ------------------------------------------------------------------
    # Canonical crop extraction
    # ------------------------------------------------------------------

    def _canonicalize_obb(self, frame, corners):
        from multi_tracker.core.tracking.canonical_crop import (
            compute_alignment_affine,
            extract_canonical_crop,
        )

        if self._input_size is not None and len(self._input_size) == 2:
            out_w, out_h = max(8, int(self._input_size[0])), max(
                8, int(self._input_size[1])
            )
        else:
            out_w = 128
            out_h = max(8, int(round(128 / self._ref_ar)))
            out_h = out_h + (out_h % 2)

        try:
            M_align, axis_theta = compute_alignment_affine(
                corners, out_w, out_h, self._padding_fraction
            )
        except ValueError:
            return None

        crop = extract_canonical_crop(frame, M_align, out_w, out_h)
        if crop is None or crop.size == 0:
            return None
        return crop, axis_theta, M_align

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------

    def _label_from_top1(self, cls_idx: int) -> str:
        names = self._class_names
        if names is None:
            return ""
        if isinstance(names, dict):
            return str(names.get(int(cls_idx), "")).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= int(cls_idx) < len(names):
            return str(names[int(cls_idx)]).strip().lower()
        return ""

    def _class_to_direction(self, label: str, cls_idx=None) -> Optional[str]:
        text = _LABEL_ALIASES.get(
            str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
        )
        if text == "left":
            return "left"
        if text == "right":
            return "right"
        if text in {"up", "down", "unknown"}:
            return None
        # Fallback for unnamed binary classifiers
        names = self._class_names
        if names is not None:
            ordered = (
                [str(v) for v in names]
                if isinstance(names, (list, tuple))
                else [str(v) for _, v in sorted(names.items())]
            )
            if len(ordered) == 2 and cls_idx is not None:
                return "right" if int(cls_idx) == 1 else "left"
        return None

    @staticmethod
    def _validate_class_names(
        class_names, *, strict: bool = False, source: str = "model"
    ) -> List[str]:
        if class_names is None:
            if strict:
                raise ValueError(f"{source} is missing class names.")
            return []
        if isinstance(class_names, dict):
            try:
                ordered = [
                    str(v)
                    for _, v in sorted(class_names.items(), key=lambda kv: int(kv[0]))
                ]
            except Exception:
                ordered = [str(v) for v in class_names.values()]
        elif isinstance(class_names, (list, tuple)):
            ordered = [str(n) for n in class_names]
        else:
            if strict:
                raise ValueError(
                    f"Unexpected class_names type in {source}: {type(class_names)}"
                )
            return []

        normalized = []
        for raw in ordered:
            token = _LABEL_ALIASES.get(
                raw.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if token is None:
                if strict:
                    raise ValueError(
                        f"Unsupported head-tail class label {raw!r} in {source}. "
                        "Expected exactly left/right or up/down/left/right/unknown."
                    )
                return ordered  # can't normalize; return raw
            normalized.append(token)

        if strict:
            normalized_set = frozenset(normalized)
            if len(normalized_set) != len(normalized):
                raise ValueError(
                    f"Duplicate or aliased head-tail labels in {source}: {ordered}."
                )
            if normalized_set not in (
                _HEADTAIL_DIRECTIONAL_CLASS_SET,
                _HEADTAIL_FIVE_CLASS_SET,
            ):
                raise ValueError(
                    f"Unsupported head-tail class schema in {source}: {ordered}. "
                    "Expected exactly left/right or up/down/left/right/unknown."
                )

        return normalized
