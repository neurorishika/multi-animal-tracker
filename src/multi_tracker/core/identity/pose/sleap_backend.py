"""
SLEAP pose backend implementation.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from multi_tracker.core.identity.runtime_types import PoseResult, PoseRuntimeConfig
from multi_tracker.core.identity.runtime_utils import (
    _artifact_meta_matches,
    _empty_pose_result,
    _looks_like_sleap_export_path,
    _path_fingerprint_token,
    _run_cli_command,
    _summarize_keypoints,
    _write_artifact_meta,
)

logger = logging.getLogger(__name__)


def _attempt_sleap_python_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    batch_size: int,
    max_instances: int,
) -> Tuple[bool, str]:
    try:
        import importlib
        import inspect
    except Exception as exc:
        return False, str(exc)

    runtime = str(runtime_flavor).strip().lower()
    module_candidates = ["sleap_nn.export.exporters", "sleap_nn.export"]
    func_candidates = ["export_model", "export"]
    if runtime == "onnx":
        func_candidates.insert(0, "export_to_onnx")
    elif runtime == "tensorrt":
        func_candidates.insert(0, "export_to_tensorrt")

    last_err = "No compatible SLEAP Python export API found."
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            last_err = str(exc)
            continue
        for fn_name in func_candidates:
            fn = getattr(mod, fn_name, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                kwargs: Dict[str, Any] = {}
                if "model_dir" in params:
                    kwargs["model_dir"] = str(model_dir)
                elif "model_path" in params:
                    kwargs["model_path"] = str(model_dir)
                elif "trained_model_path" in params:
                    kwargs["trained_model_path"] = str(model_dir)
                if "output_dir" in params:
                    kwargs["output_dir"] = str(export_dir)
                elif "export_dir" in params:
                    kwargs["export_dir"] = str(export_dir)
                elif "save_dir" in params:
                    kwargs["save_dir"] = str(export_dir)
                if "runtime" in params:
                    kwargs["runtime"] = runtime
                if "model_type" in params:
                    kwargs["model_type"] = runtime
                if "format" in params:
                    kwargs["format"] = runtime
                if "batch_size" in params:
                    kwargs["batch_size"] = int(max(1, batch_size))
                if "max_instances" in params:
                    kwargs["max_instances"] = int(max(1, max_instances))
                if kwargs:
                    fn(**kwargs)
                else:
                    fn(str(model_dir), str(export_dir))
                if _looks_like_sleap_export_path(str(export_dir), runtime):
                    return True, ""
            except Exception as exc:
                last_err = str(exc)
                continue
    return False, last_err


def _attempt_sleap_cli_export(
    model_dir: Path,
    export_dir: Path,
    runtime_flavor: str,
    sleap_env: str,
    input_hw: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
) -> Tuple[bool, str]:
    runtime = str(runtime_flavor).strip().lower()
    sleap_env = str(sleap_env or "").strip()
    batch_size = int(max(1, int(batch_size)))
    runtime_tokens = [runtime]
    if runtime == "tensorrt":
        runtime_tokens.append("trt")

    size_candidates: List[Tuple[int, int]] = []
    if (
        isinstance(input_hw, (tuple, list))
        and len(input_hw) >= 2
        and int(input_hw[0]) > 0
        and int(input_hw[1]) > 0
    ):
        size_candidates.append((int(input_hw[0]), int(input_hw[1])))
    # Fallback sizes that are generally safe for encoder/decoder stride alignment.
    size_candidates.extend([(224, 224), (256, 256)])
    # Deduplicate while preserving order.
    deduped_sizes: List[Tuple[int, int]] = []
    for s in size_candidates:
        if s not in deduped_sizes:
            deduped_sizes.append(s)
    size_candidates = deduped_sizes

    command_variants: List[List[str]] = []
    for token in runtime_tokens:
        profile_variants = [
            ["--batch-size", str(batch_size)],
            ["--batch", str(batch_size)],
            [],
        ]
        base_variants = [
            [
                "sleap-nn",
                "export",
                str(model_dir),
                "--output",
                str(export_dir),
                "--format",
                token,
            ],
            [
                "sleap-nn",
                "export",
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                str(model_dir),
                "--output",
                str(export_dir),
                "--format",
                token,
            ],
            [
                "python",
                "-m",
                "sleap_nn.export.cli",
                "--output",
                str(export_dir),
                "--format",
                token,
                str(model_dir),
            ],
        ]
        for h, w in size_candidates:
            for base in base_variants:
                for prof in profile_variants:
                    command_variants.append(
                        [
                            *base,
                            "--input-height",
                            str(int(h)),
                            "--input-width",
                            str(int(w)),
                            *prof,
                        ]
                    )
        # Keep no-size variants as last resort for CLI versions/models that infer size.
        for base in base_variants:
            for prof in profile_variants:
                command_variants.append([*base, *prof])

    if shutil.which("conda") and sleap_env:
        conda_wrapped = []
        for cmd in command_variants:
            conda_wrapped.append(["conda", "run", "-n", sleap_env, *cmd])
        # When a SLEAP env is explicitly selected, keep export execution in that env.
        # Do not fall back to the MAT process env.
        command_variants = conda_wrapped

    last_err = "No SLEAP export CLI command succeeded."
    for cmd in command_variants:
        ok, err = _run_cli_command(cmd)
        if ok and _looks_like_sleap_export_path(str(export_dir), runtime):
            return True, ""
        if err:
            last_err = err
    return False, last_err


def _auto_export_sleap_model(config: PoseRuntimeConfig, runtime_flavor: str) -> str:
    runtime = str(runtime_flavor or "native").strip().lower()
    if runtime not in {"onnx", "tensorrt"}:
        raise RuntimeError(f"Unsupported SLEAP auto-export runtime: {runtime}")

    model_path = Path(str(config.model_path or "")).expanduser().resolve()
    if _looks_like_sleap_export_path(str(model_path), runtime):
        return str(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise RuntimeError(
            f"SLEAP model path does not exist or is not a directory: {model_path}"
        )

    input_hw = (
        tuple(int(v) for v in config.sleap_export_input_hw)
        if config.sleap_export_input_hw is not None
        else None
    )
    sig_blob = (
        f"{_path_fingerprint_token(str(model_path))}|runtime={runtime}|"
        f"batch={int(config.sleap_batch)}|max_instances={int(config.sleap_max_instances)}|"
        f"input_hw={input_hw}"
    ).encode("utf-8")
    sig = hashlib.sha1(sig_blob).hexdigest()[:16]
    export_dir = model_path.parent / f"{model_path.name}.{runtime}"
    if _looks_like_sleap_export_path(
        str(export_dir), runtime
    ) and _artifact_meta_matches(export_dir, sig):
        return str(export_dir.resolve())
    if export_dir.exists():
        shutil.rmtree(export_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting SLEAP model for %s runtime: %s -> %s",
        runtime,
        model_path,
        export_dir,
    )
    if input_hw is not None:
        logger.info(
            "SLEAP export input size hint: %dx%d",
            int(input_hw[0]),
            int(input_hw[1]),
        )
    sleap_env = str(config.sleap_env or "").strip()
    # Preferred/export-stable path: run export through selected SLEAP env.
    ok, err = _attempt_sleap_cli_export(
        model_dir=model_path,
        export_dir=export_dir,
        runtime_flavor=runtime,
        sleap_env=sleap_env,
        input_hw=input_hw,
        batch_size=int(max(1, config.sleap_batch)),
    )
    if not ok and not sleap_env:
        # Dev fallback when no env is configured.
        ok, err = _attempt_sleap_python_export(
            model_dir=model_path,
            export_dir=export_dir,
            runtime_flavor=runtime,
            batch_size=int(max(1, config.sleap_batch)),
            max_instances=int(max(1, config.sleap_max_instances)),
        )
    if not ok or not _looks_like_sleap_export_path(str(export_dir), runtime):
        raise RuntimeError(f"SLEAP auto-export failed for runtime '{runtime}'. {err}")
    _write_artifact_meta(export_dir, sig)
    return str(export_dir.resolve())


class SleapServiceBackend:
    """SLEAP runtime adapter via PoseInferenceService HTTP service."""

    def __init__(
        self,
        model_dir: str,
        out_root: str,
        keypoint_names: Sequence[str],
        min_valid_conf: float = 0.2,
        sleap_env: str = "sleap",
        sleap_device: str = "auto",
        sleap_batch: int = 4,
        sleap_max_instances: int = 1,
        skeleton_edges: Optional[Sequence[Sequence[int]]] = None,
        runtime_flavor: str = "native",
        exported_model_path: str = "",
        export_input_hw: Optional[Tuple[int, int]] = None,
    ):
        try:
            from multi_tracker.posekit.inference.service import PoseInferenceService
        except ImportError:
            from multi_tracker.posekit_old.pose_inference import PoseInferenceService

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.out_root = Path(out_root).expanduser().resolve()
        self.output_keypoint_names = [str(v) for v in keypoint_names]
        self.min_valid_conf = float(min_valid_conf)
        self.sleap_env = str(sleap_env or "sleap").strip() or "sleap"
        self.sleap_device = str(sleap_device or "auto")
        self.sleap_batch = max(1, int(sleap_batch))
        self.sleap_max_instances = max(1, int(sleap_max_instances))
        self.runtime_flavor = str(runtime_flavor or "native").strip().lower()
        self.exported_model_path = str(exported_model_path or "").strip()
        self.export_input_hw = (
            (int(export_input_hw[0]), int(export_input_hw[1]))
            if isinstance(export_input_hw, (tuple, list))
            and len(export_input_hw) >= 2
            and int(export_input_hw[0]) > 0
            and int(export_input_hw[1]) > 0
            else None
        )
        self.skeleton_edges = (
            [tuple(int(v) for v in e[:2]) for e in (skeleton_edges or [])]
            if skeleton_edges
            else []
        )
        self._infer = PoseInferenceService(
            self.out_root, self.output_keypoint_names, self.skeleton_edges
        )
        self._service_started_here = False
        self._tmp_root = (
            self.out_root / "posekit" / "tmp" / f"runtime_{uuid.uuid4().hex}"
        )
        self._tmp_root.mkdir(parents=True, exist_ok=True)

    def warmup(self) -> None:
        # All SLEAP runtime flavors now execute through the persistent service.
        try:
            was_running = self._infer.sleap_service_running()
            ok, err, _ = self._infer.start_sleap_service(self.sleap_env, self.out_root)
            if not ok:
                raise RuntimeError(err or "Failed to start SLEAP service.")
            self._service_started_here = (
                not was_running
            ) and self._infer.sleap_service_running()
        except Exception as exc:
            logger.warning("SLEAP service warmup failed: %s", exc)

    def predict_batch(self, crops: Sequence[np.ndarray]) -> List[PoseResult]:
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
                batch=self.sleap_batch,
                backend="sleap",
                sleap_env=self.sleap_env,
                sleap_device=self.sleap_device,
                sleap_batch=self.sleap_batch,
                sleap_max_instances=self.sleap_max_instances,
                sleap_runtime_flavor=self.runtime_flavor,
                sleap_exported_model_path=self.exported_model_path,
                sleap_export_input_hw=self.export_input_hw,
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
            outputs.append(_summarize_keypoints(arr, self.min_valid_conf))

        return outputs

    def close(self) -> None:
        if self._service_started_here:
            try:
                self._infer.shutdown_sleap_service()
            except Exception:
                logger.debug(
                    "Failed to stop SLEAP service from backend close.", exc_info=True
                )
            self._service_started_here = False
        if self._tmp_root.exists():
            try:
                for p in self._tmp_root.glob("*.png"):
                    p.unlink(missing_ok=True)
                self._tmp_root.rmdir()
            except Exception:
                pass
