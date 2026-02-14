"""
Unified runtime adapters for per-detection individual feature inference.

This module presents a common outward interface for different pose backends
(currently YOLO and SLEAP) so tracking-side code can be shared.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseResult:
    """Canonical pose output for one detection crop."""

    def __init__(
        self,
        keypoints: Optional[np.ndarray],
        mean_conf: float,
        valid_fraction: float,
        num_valid: int,
        num_keypoints: int,
    ):
        self.keypoints = keypoints
        self.mean_conf = mean_conf
        self.valid_fraction = valid_fraction
        self.num_valid = num_valid
        self.num_keypoints = num_keypoints


def _empty_pose_result() -> PoseResult:
    return PoseResult(
        keypoints=None,
        mean_conf=0.0,
        valid_fraction=0.0,
        num_valid=0,
        num_keypoints=0,
    )


def _summarize_keypoints(
    keypoints: Optional[np.ndarray], min_valid_conf: float
) -> PoseResult:
    if keypoints is None:
        return _empty_pose_result()
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return _empty_pose_result()
    valid_mask = arr[:, 2] >= float(min_valid_conf)
    return PoseResult(
        keypoints=arr,
        mean_conf=float(np.nanmean(arr[:, 2])),
        valid_fraction=float(np.mean(valid_mask)),
        num_valid=int(np.sum(valid_mask)),
        num_keypoints=int(len(arr)),
    )


class BasePoseBackend:
    """Shared pose backend interface for runtime inference on detection crops."""

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class YoloPoseBackend(BasePoseBackend):
    """YOLO pose runtime adapter."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        min_valid_conf: float = 0.2,
        ignore_keypoints: Optional[Sequence[Any]] = None,
        keypoint_names: Optional[Sequence[str]] = None,
    ):
        from ultralytics import YOLO

        self.model_path = str(Path(model_path).expanduser().resolve())
        self.device = device
        self.min_valid_conf = float(min_valid_conf)
        self.ignore_keypoints = list(ignore_keypoints or [])
        self.keypoint_names = [str(v) for v in (keypoint_names or [])]
        self.output_keypoint_names = _filter_keypoint_names(
            self.keypoint_names, self.ignore_keypoints
        )
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []
        results = self.model.predict(
            source=list(crops),
            conf=1e-4,
            iou=0.7,
            max_det=1,
            verbose=False,
            device=self.device,
        )

        outputs: List[PoseResult] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            if keypoints is None:
                outputs.append(_empty_pose_result())
                continue

            try:
                xy = keypoints.xy
                xy = xy.cpu().numpy() if hasattr(xy, "cpu") else np.asarray(xy)
                conf = getattr(keypoints, "conf", None)
                if conf is not None:
                    conf = (
                        conf.cpu().numpy() if hasattr(conf, "cpu") else np.asarray(conf)
                    )
            except Exception:
                outputs.append(_empty_pose_result())
                continue

            if xy.ndim == 2:
                xy = xy[None, :, :]
            if conf is not None and conf.ndim == 1:
                conf = conf[None, :]

            if xy.size == 0:
                outputs.append(_empty_pose_result())
                continue

            if conf is None:
                conf = np.zeros((xy.shape[0], xy.shape[1]), dtype=np.float32)
            mean_per_instance = np.nanmean(conf, axis=1)
            best_idx = (
                int(np.nanargmax(mean_per_instance)) if len(mean_per_instance) else 0
            )
            pred_xy = np.asarray(xy[best_idx], dtype=np.float32)
            pred_conf = np.asarray(conf[best_idx], dtype=np.float32)
            if pred_xy.ndim != 2 or pred_xy.shape[1] != 2:
                outputs.append(_empty_pose_result())
                continue

            kpts = np.column_stack((pred_xy, pred_conf)).astype(np.float32)
            kpts = _apply_ignore_keypoints(
                kpts,
                self.ignore_keypoints,
                self.keypoint_names,
            )
            outputs.append(_summarize_keypoints(kpts, self.min_valid_conf))

        return outputs


class SleapPoseBackend(BasePoseBackend):
    """SLEAP runtime adapter via PoseInferenceService."""

    def __init__(
        self,
        model_dir: str,
        out_root: str,
        keypoint_names: Sequence[str],
        min_valid_conf: float = 0.2,
        sleap_env: str = "sleap",
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        skeleton_edges: Optional[Sequence[Sequence[int]]] = None,
        ignore_keypoints: Optional[Sequence[Any]] = None,
    ):
        from multi_tracker.posekit.pose_inference import PoseInferenceService

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.keypoint_names = list(keypoint_names)
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_env = str(sleap_env)
        self.sleap_device = str(sleap_device)
        self.sleap_batch = int(sleap_batch)
        self.sleap_max_instances = 1
        self.ignore_keypoints = list(ignore_keypoints or [])
        self.output_keypoint_names = _filter_keypoint_names(
            self.keypoint_names, self.ignore_keypoints
        )
        self.skeleton_edges = (
            [tuple(int(v) for v in e[:2]) for e in skeleton_edges]
            if skeleton_edges
            else []
        )
        self._infer = PoseInferenceService(
            self.out_root, self.keypoint_names, self.skeleton_edges
        )
        self._tmp_root = (
            self.out_root / "posekit" / "tmp" / f"mat_runtime_{uuid.uuid4().hex}"
        )
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        if not crops:
            return []

        paths: List[Path] = []
        for i, crop in enumerate(crops):
            p = self._tmp_root / f"crop_{i:06d}.png"
            ok = cv2.imwrite(str(p), crop)
            if not ok:
                paths.append(Path("__invalid__"))
            else:
                paths.append(p)

        valid_paths = [p for p in paths if p.exists()]
        preds: Dict[str, List[Any]] = {}
        if valid_paths:
            pred_map, err = self._infer.predict(
                model_path=self.model_dir,
                image_paths=valid_paths,
                device="auto",
                imgsz=640,
                conf=1e-4,
                batch=max(1, self.sleap_batch),
                backend="sleap",
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=max(1, self.sleap_batch),
                sleap_max_instances=max(1, self.sleap_max_instances),
            )
            if pred_map is None:
                raise RuntimeError(err or "SLEAP inference failed.")
            preds = pred_map

        outputs: List[PoseResult] = []
        for p in paths:
            if not p.exists():
                outputs.append(_empty_pose_result())
                continue
            pred = preds.get(str(p))
            if pred is None:
                pred = preds.get(str(p.resolve()))
            if not pred:
                outputs.append(_empty_pose_result())
                continue
            arr = np.asarray(pred, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 3:
                outputs.append(_empty_pose_result())
                continue
            arr = _apply_ignore_keypoints(
                arr,
                self.ignore_keypoints,
                self.keypoint_names,
            )
            outputs.append(_summarize_keypoints(arr, self.min_valid_conf))

        return outputs

    def close(self) -> None:
        if self._tmp_root.exists():
            try:
                for p in self._tmp_root.glob("*.png"):
                    p.unlink(missing_ok=True)
                self._tmp_root.rmdir()
            except Exception:
                pass


def _parse_ignore_keypoints(raw: Any) -> List[Any]:
    if raw is None:
        return []
    if isinstance(raw, str):
        out: List[Any] = []
        for token in raw.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                out.append(int(t))
            except ValueError:
                out.append(t)
        return out
    if isinstance(raw, (list, tuple)):
        out = []
        for v in raw:
            try:
                out.append(int(v))
            except Exception:
                out.append(str(v))
        return out
    return []


def _load_skeleton_from_json(path_str: str):
    if not path_str:
        return [], []
    p = Path(path_str).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise RuntimeError(f"Skeleton file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))

    names_raw = data.get("keypoint_names", data.get("keypoints", []))
    edges_raw = data.get("skeleton_edges", data.get("edges", []))

    names = [str(v).strip() for v in names_raw if str(v).strip()]
    edges = []
    for edge in edges_raw:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        try:
            edges.append((int(edge[0]), int(edge[1])))
        except Exception:
            continue
    return names, edges


def _apply_ignore_keypoints(
    keypoints: np.ndarray,
    ignore_keypoints: Sequence[Any],
    keypoint_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    if keypoints is None or len(keypoints) == 0:
        return keypoints
    ignore_spec = _parse_ignore_keypoints(ignore_keypoints)
    if not ignore_spec:
        return keypoints

    k = int(len(keypoints))
    ignore_idxs = set()
    for v in ignore_spec:
        if isinstance(v, int):
            if 0 <= v < k:
                ignore_idxs.add(v)
            continue
        if isinstance(v, str) and keypoint_names:
            try:
                idx = [str(n) for n in keypoint_names].index(v)
                if 0 <= idx < k:
                    ignore_idxs.add(idx)
            except ValueError:
                continue
    if not ignore_idxs:
        return keypoints
    keep = [i for i in range(k) if i not in ignore_idxs]
    if not keep:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(keypoints[keep], dtype=np.float32)


def _filter_keypoint_names(
    keypoint_names: Sequence[str], ignore_keypoints: Sequence[Any]
) -> List[str]:
    names = [str(n) for n in (keypoint_names or [])]
    if not names:
        return []
    ignore_spec = _parse_ignore_keypoints(ignore_keypoints)
    if not ignore_spec:
        return names

    ignore_idxs = set()
    for token in ignore_spec:
        if isinstance(token, int):
            if 0 <= token < len(names):
                ignore_idxs.add(token)
            continue
        if isinstance(token, str):
            try:
                idx = names.index(token)
                ignore_idxs.add(idx)
            except ValueError:
                continue
    if not ignore_idxs:
        return names
    return [name for idx, name in enumerate(names) if idx not in ignore_idxs]


def create_pose_backend(params: Dict[str, Any], out_root: str) -> BasePoseBackend:
    """Factory for unified pose backend adapters."""
    backend = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
    model_path = str(params.get("POSE_MODEL_DIR", "")).strip()
    min_valid_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
    ignore_keypoints = _parse_ignore_keypoints(params.get("POSE_IGNORE_KEYPOINTS", []))
    skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
    skeleton_names, skeleton_edges = _load_skeleton_from_json(skeleton_file)
    if not model_path:
        raise RuntimeError("Pose model path is empty.")

    if backend == "yolo":
        device = str(params.get("YOLO_DEVICE", "cpu"))
        return YoloPoseBackend(
            model_path=model_path,
            device=device,
            min_valid_conf=min_valid_conf,
            ignore_keypoints=ignore_keypoints,
            keypoint_names=skeleton_names,
        )

    if backend == "sleap":
        if not skeleton_names:
            raise RuntimeError(
                "SLEAP pose backend requires a skeleton JSON with keypoint_names."
            )
        return SleapPoseBackend(
            model_dir=model_path,
            out_root=out_root,
            keypoint_names=skeleton_names,
            min_valid_conf=min_valid_conf,
            sleap_env=str(params.get("POSE_SLEAP_ENV", "sleap")),
            sleap_device=str(params.get("POSE_SLEAP_DEVICE", "auto")),
            sleap_batch=int(params.get("POSE_SLEAP_BATCH", 4)),
            skeleton_edges=skeleton_edges,
            ignore_keypoints=ignore_keypoints,
        )

    raise RuntimeError(f"Unsupported pose backend: {backend}")
