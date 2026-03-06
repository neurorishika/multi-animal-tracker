"""Shared canonicalization helpers for MAT individual-dataset exports."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@lru_cache(maxsize=128)
def _find_metadata_path(image_path_str: str) -> str:
    image_path = Path(image_path_str).expanduser().resolve()
    if image_path.parent.name == "images":
        candidate = image_path.parent.parent / "metadata.json"
        if candidate.exists():
            return str(candidate.resolve())
    for parent in image_path.parents:
        candidate = parent / "metadata.json"
        if candidate.exists():
            return str(candidate.resolve())
    return ""


@lru_cache(maxsize=32)
def _load_metadata_index(metadata_path_str: str) -> dict[str, Any]:
    metadata_path = Path(metadata_path_str).expanduser().resolve()
    root = metadata_path.parent
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    index: dict[str, list[dict[str, Any]]] = {}
    for frame in data.get("frames", []):
        image_file = str(frame.get("image_file", "")).strip()
        annotations = frame.get("annotations", [])
        if not image_file or not isinstance(annotations, list):
            continue
        keys = {
            image_file,
            Path(image_file).name,
            str((root / "images" / image_file).resolve()),
            str(Path("images") / image_file),
        }
        for key in keys:
            index[key] = annotations
    for image_entry in data.get("images", data.get("crops", [])):
        image_file = str(image_entry.get("filename", "")).strip()
        if not image_file:
            continue
        annotations = [image_entry]
        keys = {
            image_file,
            Path(image_file).name,
            str((root / "images" / image_file).resolve()),
            str(Path("images") / image_file),
        }
        for key in keys:
            index[key] = annotations
    return {"root": str(root), "index": index}


def _resolve_annotations_for_image(
    image_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    metadata_path_str = _find_metadata_path(str(image_path))
    if not metadata_path_str:
        return "", []
    payload = _load_metadata_index(metadata_path_str)
    index = payload.get("index", {})
    keys = [str(image_path.resolve()), image_path.name]
    root = Path(str(payload.get("root", ""))) if payload.get("root") else None
    if root is not None:
        try:
            keys.append(str(image_path.resolve().relative_to(root)))
        except Exception:
            pass
        try:
            keys.append(
                str(Path("images") / image_path.resolve().relative_to(root / "images"))
            )
        except Exception:
            pass
    for key in keys:
        anns = index.get(key)
        if isinstance(anns, list):
            return metadata_path_str, anns
    return metadata_path_str, []


class MatMetadataCanonicalizer:
    """Canonicalize images using MAT individual-dataset metadata when possible."""

    def __init__(self, enabled: bool = False, margin: float = 1.3):
        self.enabled = bool(enabled)
        self.margin = float(margin)
        self.applied_count = 0
        self.skipped_count = 0
        self.skip_reasons: dict[str, int] = {}

    def _record_skip(self, reason: str) -> None:
        self.skipped_count += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied_count": int(self.applied_count),
            "skipped_count": int(self.skipped_count),
            "skip_reasons": dict(self.skip_reasons),
            "margin": float(self.margin),
        }

    def __call__(self, image_path: Path, img_pil):
        if not self.enabled:
            return img_pil

        metadata_path, annotations = _resolve_annotations_for_image(
            Path(image_path).expanduser().resolve()
        )
        if not metadata_path:
            self._record_skip("missing_metadata")
            return img_pil
        if len(annotations) != 1:
            self._record_skip("ambiguous_annotation_count")
            return img_pil

        ann = annotations[0]
        canonical = ann.get("canonicalization", {})
        center = canonical.get("center_px")
        size = canonical.get("size_px")
        angle = canonical.get("angle_rad", ann.get("theta"))
        if (
            not isinstance(center, (list, tuple))
            or len(center) != 2
            or not isinstance(size, (list, tuple))
            or len(size) != 2
            or angle is None
        ):
            self._record_skip("missing_canonicalization_fields")
            return img_pil

        cx = float(center[0])
        cy = float(center[1])
        box_w = max(1.0, float(size[0]))
        box_h = max(1.0, float(size[1]))
        angle = float(angle)

        arr = np.asarray(img_pil.convert("RGB"))
        if arr.ndim != 3 or arr.shape[2] != 3:
            self._record_skip("invalid_image")
            return img_pil

        crop_w = max(8.0, box_w * self.margin)
        crop_h = max(8.0, box_h * self.margin)
        out_w = max(8, int(round(crop_w)))
        out_h = max(8, int(round(crop_h)))

        cos_a = float(math.cos(angle))
        sin_a = float(math.sin(angle))
        half_w = crop_w * 0.5
        half_h = crop_h * 0.5
        src_pts = np.array(
            [
                [
                    cx - half_w * cos_a + half_h * sin_a,
                    cy - half_w * sin_a - half_h * cos_a,
                ],
                [
                    cx + half_w * cos_a + half_h * sin_a,
                    cy + half_w * sin_a - half_h * cos_a,
                ],
                [
                    cx - half_w * cos_a - half_h * sin_a,
                    cy - half_w * sin_a + half_h * cos_a,
                ],
            ],
            dtype=np.float32,
        )
        dst_pts = np.array(
            [[0, 0], [out_w - 1, 0], [0, out_h - 1]],
            dtype=np.float32,
        )
        try:
            affine = cv2.getAffineTransform(src_pts, dst_pts)
            warped = cv2.warpAffine(
                arr,
                affine,
                (out_w, out_h),
                flags=getattr(cv2, "INTER_LINEAR", getattr(cv2, "INTER_AREA", 1)),
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            self._record_skip("warp_failed")
            return img_pil

        if warped is None or warped.size == 0:
            self._record_skip("empty_warp")
            return img_pil

        self.applied_count += 1
        try:
            from PIL import Image

            return Image.fromarray(warped)
        except Exception:
            self._record_skip("pil_conversion_failed")
            return img_pil


def get_canon_transform(image_path: "str | Path", margin: float = 1.3) -> "dict | None":
    """Compute the canonicalization affine transforms for an image without applying them.

    Returns a ``dict`` with:

    * ``"affine"`` – 2×3 :class:`numpy.ndarray` mapping **original → canonical** pixel coords.
    * ``"inv_affine"`` – 2×3 :class:`numpy.ndarray` mapping **canonical → original** pixel coords.
    * ``"canon_w"``, ``"canon_h"`` – output (canonicalized) image dimensions.

    Returns ``None`` if metadata is unavailable or the annotation count is ambiguous.
    Use this to convert saved keypoint coordinates between original and canonical spaces.
    """
    metadata_path, annotations = _resolve_annotations_for_image(
        Path(image_path).expanduser().resolve()
    )
    if not metadata_path or len(annotations) != 1:
        return None

    ann = annotations[0]
    canonical = ann.get("canonicalization", {})
    center = canonical.get("center_px")
    size = canonical.get("size_px")
    angle = canonical.get("angle_rad", ann.get("theta"))
    if (
        not isinstance(center, (list, tuple))
        or len(center) != 2
        or not isinstance(size, (list, tuple))
        or len(size) != 2
        or angle is None
    ):
        return None

    cx = float(center[0])
    cy = float(center[1])
    box_w = max(1.0, float(size[0]))
    box_h = max(1.0, float(size[1]))
    angle = float(angle)

    crop_w = max(8.0, box_w * margin)
    crop_h = max(8.0, box_h * margin)
    out_w = max(8, int(round(crop_w)))
    out_h = max(8, int(round(crop_h)))

    cos_a = float(math.cos(angle))
    sin_a = float(math.sin(angle))
    half_w = crop_w * 0.5
    half_h = crop_h * 0.5
    src_pts = np.array(
        [
            [
                cx - half_w * cos_a + half_h * sin_a,
                cy - half_w * sin_a - half_h * cos_a,
            ],
            [
                cx + half_w * cos_a + half_h * sin_a,
                cy + half_w * sin_a - half_h * cos_a,
            ],
            [
                cx - half_w * cos_a - half_h * sin_a,
                cy - half_w * sin_a + half_h * cos_a,
            ],
        ],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[0, 0], [out_w - 1, 0], [0, out_h - 1]],
        dtype=np.float32,
    )
    try:
        affine = cv2.getAffineTransform(src_pts, dst_pts)
        inv_affine = cv2.invertAffineTransform(affine)
    except Exception:
        return None

    return {
        "affine": affine,
        "inv_affine": inv_affine,
        "canon_w": out_w,
        "canon_h": out_h,
    }
