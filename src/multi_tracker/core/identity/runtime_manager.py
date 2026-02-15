"""
Run-scoped lifecycle manager for pose inference backends.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .runtime_api import (
    PoseInferenceBackend,
    PoseRuntimeConfig,
    build_runtime_config,
    create_pose_backend_from_config,
)

logger = logging.getLogger(__name__)


@dataclass
class RuntimeMetrics:
    startup_ms: float = 0.0
    warmup_ms: float = 0.0
    closed_ms: float = 0.0


class InferenceRuntimeManager:
    """Owns lifecycle and metrics for one pose inference runtime session."""

    def __init__(self, session_name: str = "pose_runtime"):
        self.session_name = str(session_name or "pose_runtime")
        self.backend: Optional[PoseInferenceBackend] = None
        self.config: Optional[PoseRuntimeConfig] = None
        self.metrics = RuntimeMetrics()
        self._sleap_service_started = False

    def __enter__(self) -> "InferenceRuntimeManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def create_backend(self, config: PoseRuntimeConfig) -> PoseInferenceBackend:
        """Create backend for config and perform run-scoped pre-start logic."""
        self.config = config
        t0 = time.perf_counter()

        # Run-scoped SLEAP service ownership for MAT flows.
        if str(config.backend_family).lower() == "sleap" and str(
            config.runtime_flavor
        ).lower() in {"native"}:
            self._start_sleap_service_if_needed(config)

        self.backend = create_pose_backend_from_config(config)
        self.metrics.startup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Pose runtime backend created (%s/%s) in %.2f ms",
            config.backend_family,
            config.runtime_flavor,
            self.metrics.startup_ms,
        )
        return self.backend

    def create_backend_from_params(
        self,
        params,
        out_root: str,
        keypoint_names_override=None,
        skeleton_edges_override=None,
    ) -> PoseInferenceBackend:
        cfg = build_runtime_config(
            params,
            out_root=out_root,
            keypoint_names_override=keypoint_names_override,
            skeleton_edges_override=skeleton_edges_override,
        )
        return self.create_backend(cfg)

    def warmup(self) -> None:
        if self.backend is None:
            return
        t0 = time.perf_counter()
        self.backend.warmup()
        self.metrics.warmup_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("Pose runtime warmup finished in %.2f ms", self.metrics.warmup_ms)

    def close(self) -> None:
        t0 = time.perf_counter()
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                logger.debug("Pose backend close failed.", exc_info=True)
            finally:
                self.backend = None
        if self._sleap_service_started:
            try:
                from multi_tracker.posekit.pose_inference import PoseInferenceService

                PoseInferenceService.shutdown_sleap_service()
                logger.info("Stopped SLEAP service for session: %s", self.session_name)
            except Exception:
                logger.debug("Failed to stop SLEAP service.", exc_info=True)
            self._sleap_service_started = False
        self.metrics.closed_ms = (time.perf_counter() - t0) * 1000.0

    def _start_sleap_service_if_needed(self, config: PoseRuntimeConfig) -> None:
        env = str(config.sleap_env or "").strip()
        if not env:
            return
        try:
            from multi_tracker.posekit.pose_inference import PoseInferenceService

            if PoseInferenceService.sleap_service_running():
                return
            ok, err, log_path = PoseInferenceService.start_sleap_service(
                env_name=env, out_root=Path(config.out_root)
            )
            if ok:
                self._sleap_service_started = True
                logger.info(
                    "Started SLEAP service (%s) for session %s%s",
                    env,
                    self.session_name,
                    f" [log: {log_path}]" if log_path else "",
                )
            else:
                logger.warning(
                    "Failed to pre-start SLEAP service for session %s: %s",
                    self.session_name,
                    err or "unknown error",
                )
        except Exception:
            logger.debug("SLEAP pre-start skipped.", exc_info=True)
