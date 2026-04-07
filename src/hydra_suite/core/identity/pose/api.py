"""
Shared pose inference runtime API for MAT + PoseKit.

Centralizes backend selection and runtime behavior while keeping
the calling surface small and stable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from hydra_suite.core.identity.pose.backends.sleap import (
    SleapServiceBackend,
    auto_export_sleap_model,
)
from hydra_suite.core.identity.pose.backends.sleap_utils import (
    derive_sleap_export_input_hw,
)
from hydra_suite.core.identity.pose.backends.yolo import (
    YoloNativeBackend,
    auto_export_yolo_model,
)
from hydra_suite.core.identity.pose.types import PoseInferenceBackend, PoseRuntimeConfig
from hydra_suite.core.identity.pose.utils import (
    load_skeleton_from_json,
    normalize_runtime_flavor,
    parse_runtime_request,
)
from hydra_suite.runtime.compute_runtime import derive_pose_runtime_settings

logger = logging.getLogger(__name__)


def _norm_hw(value: Any) -> Optional[int]:
    """Normalize a height/width value: align to 32, clamp to [64, 1024]."""
    try:
        v = int(value)
    except Exception:
        return None
    if v <= 0:
        return None
    v = int(((v + 31) // 32) * 32)
    v = max(64, min(1024, v))
    return v


def _resolve_device_and_batch(
    params: Dict[str, Any],
    backend_family: str,
    derived_device: str,
) -> Tuple[str, int]:
    """Resolve the device string and batch size from params and derived settings."""
    if backend_family == "sleap":
        device = (
            derived_device
            or str(params.get("POSE_SLEAP_DEVICE", "auto")).strip()
            or "auto"
        )
        batch_size = int(params.get("POSE_SLEAP_BATCH", 4))
    else:
        device = derived_device or (
            str(params.get("YOLO_DEVICE", params.get("POSE_DEVICE", "auto"))).strip()
            or "auto"
        )
        batch_size = int(
            params.get("POSE_YOLO_BATCH", params.get("POSE_BATCH_SIZE", 4))
        )
    return device, batch_size


def _resolve_export_hw(
    params: Dict[str, Any],
    backend_family: str,
    model_path: str,
) -> Optional[Tuple[int, int]]:
    """Resolve the SLEAP export input height/width from params and model introspection."""
    derived_model_hw: Optional[Tuple[int, int]] = None
    if backend_family == "sleap" and model_path:
        try:
            derived_model_hw = derive_sleap_export_input_hw(model_path)
        except Exception:
            derived_model_hw = None

    export_h = _norm_hw(params.get("POSE_SLEAP_EXPORT_INPUT_HEIGHT", 0))
    export_w = _norm_hw(params.get("POSE_SLEAP_EXPORT_INPUT_WIDTH", 0))
    if export_h is None and export_w is None and derived_model_hw is not None:
        export_h, export_w = int(derived_model_hw[0]), int(derived_model_hw[1])
    if export_h is None and export_w is None:
        crop_hint = _norm_hw(params.get("IDENTITY_CROP_MAX_SIZE", 0))
        if crop_hint is not None:
            export_h = crop_hint
            export_w = crop_hint
    if export_h is None and export_w is not None:
        export_h = export_w
    if export_w is None and export_h is not None:
        export_w = export_h
    return (int(export_h), int(export_w)) if (export_h and export_w) else None


def build_runtime_config(
    params: Dict[str, Any],
    out_root: str,
    keypoint_names_override: Optional[Sequence[str]] = None,
    skeleton_edges_override: Optional[Sequence[Sequence[int]]] = None,
) -> PoseRuntimeConfig:
    """Construct a PoseRuntimeConfig from a tracking params dict.

    Resolves backend family, runtime flavor, device, batch size, skeleton, and
    all model-path fields, applying compute_runtime-derived overrides where appropriate.
    """
    backend_family = str(params.get("POSE_MODEL_TYPE", "yolo")).strip().lower()
    runtime_flavor = str(params.get("POSE_RUNTIME_FLAVOR", "auto")).strip().lower()
    model_path = str(params.get("POSE_MODEL_DIR", "")).strip()
    exported_model_path = str(params.get("POSE_EXPORTED_MODEL_PATH", "")).strip()
    compute_runtime = str(
        params.get("COMPUTE_RUNTIME", params.get("compute_runtime", ""))
    ).strip()

    skeleton_file = str(params.get("POSE_SKELETON_FILE", "")).strip()
    skeleton_names, skeleton_edges = load_skeleton_from_json(skeleton_file)

    if keypoint_names_override:
        skeleton_names = [str(v) for v in keypoint_names_override]
    if skeleton_edges_override:
        skeleton_edges = [
            (int(edge[0]), int(edge[1]))
            for edge in skeleton_edges_override
            if isinstance(edge, (list, tuple)) and len(edge) >= 2
        ]

    derived_device = ""
    if compute_runtime:
        derived = derive_pose_runtime_settings(compute_runtime, backend_family)
        runtime_flavor = (
            str(derived.get("pose_runtime_flavor", runtime_flavor)).strip().lower()
        )
        derived_device = str(derived.get("pose_sleap_device", "auto")).strip() or "auto"

    device, batch_size = _resolve_device_and_batch(
        params,
        backend_family,
        derived_device,
    )
    export_hw = _resolve_export_hw(params, backend_family, model_path)

    return PoseRuntimeConfig(
        backend_family=backend_family,
        runtime_flavor=runtime_flavor,
        device=device,
        batch_size=max(1, batch_size),
        model_path=model_path,
        exported_model_path=exported_model_path,
        out_root=str(out_root),
        min_valid_conf=float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2)),
        yolo_conf=float(params.get("POSE_YOLO_CONF", 1e-4)),
        yolo_iou=float(params.get("POSE_YOLO_IOU", 0.7)),
        yolo_max_det=int(params.get("POSE_YOLO_MAX_DET", 1)),
        yolo_batch=max(1, int(batch_size)),
        sleap_env=str(params.get("POSE_SLEAP_ENV", "sleap")).strip() or "sleap",
        sleap_device=derived_device or str(params.get("POSE_SLEAP_DEVICE", "auto")),
        sleap_batch=int(params.get("POSE_SLEAP_BATCH", 4)),
        sleap_max_instances=int(params.get("POSE_SLEAP_MAX_INSTANCES", 1)),
        sleap_export_input_hw=export_hw,
        sleap_experimental_features=bool(
            params.get("POSE_SLEAP_EXPERIMENTAL_FEATURES", False)
        ),
        keypoint_names=skeleton_names,
        skeleton_edges=skeleton_edges,
    )


def create_pose_backend_from_config(config: PoseRuntimeConfig) -> PoseInferenceBackend:
    backend_family = str(config.backend_family or "yolo").strip().lower()
    requested_runtime = str(config.runtime_flavor or "auto").strip().lower()
    parsed_runtime, parsed_device = parse_runtime_request(requested_runtime)
    runtime_flavor = normalize_runtime_flavor(backend_family, requested_runtime)
    effective_device = (
        parsed_device
        if parsed_device
        else (
            str(config.sleap_device or "auto")
            if backend_family == "sleap"
            else str(config.device or "auto")
        )
    )
    if parsed_runtime == "auto":
        logger.info(
            "Pose runtime auto-selected for %s backend: %s",
            backend_family,
            runtime_flavor,
        )

    if backend_family == "yolo":
        model_candidate = str(config.model_path).strip()
        model_candidate_path = (
            Path(model_candidate).expanduser().resolve() if model_candidate else None
        )
        if model_candidate_path is not None and model_candidate_path.exists():
            if model_candidate_path.is_dir():
                raise RuntimeError(
                    "POSE_MODEL_TYPE is set to 'yolo' but POSE_MODEL_DIR points to a "
                    "directory. This looks like a SLEAP model directory. "
                    "Set POSE_MODEL_TYPE='sleap' for this model, or select a YOLO model "
                    "file (.pt/.onnx/.engine)."
                )
            valid_yolo_suffixes = {".pt", ".onnx", ".engine", ".trt"}
            if model_candidate_path.suffix.lower() not in valid_yolo_suffixes:
                raise RuntimeError(
                    "Unsupported YOLO model path for pose inference: "
                    f"{model_candidate_path}. Expected one of: "
                    ".pt, .onnx, .engine, .trt"
                )
        if runtime_flavor in ("onnx", "tensorrt"):
            try:
                model_candidate = auto_export_yolo_model(
                    config, runtime_flavor, runtime_device=effective_device
                )
            except Exception as exc:
                logger.warning(
                    "YOLO %s runtime initialization failed (%s). Falling back to native runtime.",
                    runtime_flavor,
                    exc,
                )
                runtime_flavor = "native"
                model_candidate = str(config.model_path).strip()
        if not model_candidate:
            raise RuntimeError("Pose model path is empty.")
        return YoloNativeBackend(
            model_path=model_candidate,
            device=effective_device,
            min_valid_conf=config.min_valid_conf,
            keypoint_names=config.keypoint_names,
            conf=config.yolo_conf,
            iou=config.yolo_iou,
            max_det=config.yolo_max_det,
            batch_size=config.yolo_batch,
        )

    if backend_family == "sleap":
        if not config.keypoint_names:
            raise RuntimeError(
                "SLEAP backend requires keypoint_names (from skeleton JSON or override)."
            )

        if (
            runtime_flavor in ("onnx", "tensorrt")
            and not config.sleap_experimental_features
        ):
            logger.warning(
                "SLEAP %s runtime is experimental and disabled. Reverting to native runtime.",
                runtime_flavor,
            )
            runtime_flavor = "native"

        exported_candidate = ""
        if runtime_flavor in ("onnx", "tensorrt"):
            try:
                exported_candidate = auto_export_sleap_model(config, runtime_flavor)
                logger.info(
                    "SLEAP model exported for %s runtime: %s",
                    runtime_flavor,
                    exported_candidate,
                )
            except Exception as exc:
                logger.warning(
                    "SLEAP exported runtime (%s) initialization failed: %s. "
                    "Falling back to SLEAP service backend with native runtime.",
                    runtime_flavor,
                    exc,
                )
                runtime_flavor = "native"
                exported_candidate = ""

        service_backend = SleapServiceBackend(
            model_dir=config.model_path,
            out_root=config.out_root,
            keypoint_names=config.keypoint_names,
            min_valid_conf=config.min_valid_conf,
            sleap_env=config.sleap_env,
            sleap_device=effective_device,
            sleap_batch=config.sleap_batch,
            sleap_max_instances=max(1, int(config.sleap_max_instances)),
            skeleton_edges=config.skeleton_edges,
            runtime_flavor=runtime_flavor,
            exported_model_path=exported_candidate,
            export_input_hw=config.sleap_export_input_hw,
        )
        return service_backend

    raise RuntimeError(f"Unsupported pose backend family: {backend_family}")
