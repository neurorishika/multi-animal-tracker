"""
Compatibility layer for pose runtime backends.

Historically this module implemented concrete YOLO/SLEAP runtime classes.
It now delegates to shared runtime modules while preserving the public API used
by MAT workers and tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from .runtime_api import (
        PoseResult,
        SleapServiceBackend,
        YoloNativeBackend,
        build_runtime_config,
    )
    from .runtime_manager import InferenceRuntimeManager
except Exception:
    # Support direct module loading in tests.
    from multi_tracker.core.identity.runtime_api import (  # type: ignore
        PoseResult,
        SleapServiceBackend,
        YoloNativeBackend,
        build_runtime_config,
    )
    from multi_tracker.core.identity.runtime_manager import (  # type: ignore
        InferenceRuntimeManager,
    )


class BasePoseBackend:
    """Shared pose backend interface for runtime inference on detection crops."""

    output_keypoint_names: List[str]

    def warmup(self) -> None:
        return None

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def resolved_artifact_path(self) -> Optional[str]:
        return None


class _BackendAdapter(BasePoseBackend):
    """Adapter from runtime_api backend protocol to legacy feature_runtime API."""

    def __init__(self, backend, manager: Optional[InferenceRuntimeManager] = None):
        self._backend = backend
        self._manager = manager
        self.output_keypoint_names = list(
            getattr(self._backend, "output_keypoint_names", []) or []
        )
        # Backward-compatible alias used by existing tests/callers.
        self.keypoint_names = list(self.output_keypoint_names)

    @classmethod
    def from_runtime(cls, backend, manager: Optional[InferenceRuntimeManager] = None):
        obj = cls.__new__(cls)
        _BackendAdapter.__init__(obj, backend=backend, manager=manager)
        return obj

    def warmup(self) -> None:
        if self._manager is not None:
            self._manager.warmup()
        else:
            self._backend.warmup()

    def predict_crops(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
        return self._backend.predict_batch(crops)

    def close(self) -> None:
        if self._manager is not None:
            self._manager.close()
        else:
            self._backend.close()

    def resolved_artifact_path(self) -> Optional[str]:
        candidates = [
            getattr(self._backend, "exported_model_path", None),
            getattr(self._backend, "model_path", None),
        ]
        for value in candidates:
            if not value:
                continue
            try:
                return str(value)
            except Exception:
                continue
        return None


class YoloPoseBackend(_BackendAdapter):
    """YOLO pose runtime adapter (legacy constructor preserved)."""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        min_valid_conf: float = 0.2,
        keypoint_names: Optional[Sequence[str]] = None,
    ):
        backend = YoloNativeBackend(
            model_path=model_path,
            device=device,
            min_valid_conf=min_valid_conf,
            keypoint_names=keypoint_names,
        )
        super().__init__(backend, manager=None)


class SleapPoseBackend(_BackendAdapter):
    """SLEAP runtime adapter via PoseInferenceService (legacy constructor preserved)."""

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
    ):
        backend = SleapServiceBackend(
            model_dir=model_dir,
            out_root=out_root,
            keypoint_names=keypoint_names,
            min_valid_conf=min_valid_conf,
            sleap_env=sleap_env,
            sleap_device=sleap_device,
            sleap_batch=sleap_batch,
            sleap_max_instances=1,
            skeleton_edges=skeleton_edges,
        )
        super().__init__(backend, manager=None)


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
    """
    Factory for unified pose backend adapters.

    This now routes through runtime_manager/runtime_api so all callers share the
    same selection and lifecycle behavior.
    """
    manager = InferenceRuntimeManager(session_name="pose_runtime")
    config = build_runtime_config(params, out_root=out_root)
    backend = manager.create_backend(config)
    if str(config.backend_family).lower() == "sleap":
        adapter = SleapPoseBackend.from_runtime(backend, manager=manager)
    else:
        adapter = YoloPoseBackend.from_runtime(backend, manager=manager)
    adapter.warmup()
    return adapter
