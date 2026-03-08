"""
Tracking Parameter Optimizer and Previewer.
Enhanced with Dynamic Bayesian Optimization.
"""

import logging
import math
from collections import deque
from typing import Any, Dict

import cv2
import numpy as np
import optuna
from PySide6.QtCore import QThread, Signal

from multi_tracker.core.assigners.hungarian import TrackAssigner
from multi_tracker.core.detectors.engine import DetectionFilter
from multi_tracker.core.filters.kalman import KalmanFilterManager
from multi_tracker.core.tracking.pose_features import (
    build_pose_detection_keypoint_map as _pf_build_keypoint_map,
)
from multi_tracker.core.tracking.pose_features import (  # noqa: F401 (re-exported via parity path)
    collapse_obb_axis_theta as _pf_collapse_obb_axis,
)
from multi_tracker.core.tracking.pose_features import (
    compute_detection_pose_features as _pf_compute_det_features,
)
from multi_tracker.core.tracking.pose_features import (
    load_pose_context_from_params as _pf_load_pose_context,
)
from multi_tracker.core.tracking.pose_features import (
    normalize_theta as _pf_normalize_theta,
)
from multi_tracker.core.tracking.pose_features import (
    resolve_tracking_theta as _pf_resolve_tracking_theta,
)
from multi_tracker.core.tracking.pose_features import (
    select_directed_heading as _pf_select_directed_heading,
)
from multi_tracker.data.detection_cache import DetectionCache

logger = logging.getLogger(__name__)

# Canonical search-space bounds — module-level so they can be imported externally.
_PARAM_RANGES: Dict[str, tuple] = {
    "YOLO_CONFIDENCE_THRESHOLD": ("float", 0.05, 0.8),
    "YOLO_IOU_THRESHOLD": ("float", 0.1, 0.9),
    "MAX_DISTANCE_MULTIPLIER": ("float", 0.5, 3.0),
    "KALMAN_NOISE_COVARIANCE": ("log_float", 0.001, 0.2),
    "KALMAN_MEASUREMENT_NOISE_COVARIANCE": ("log_float", 0.01, 1.0),
    "W_POSITION": ("float", 0.1, 5.0),
    "W_ORIENTATION": ("float", 0.0, 5.0),
    "W_AREA": ("float", 0.0, 2.0),
    "W_ASPECT": ("float", 0.0, 2.0),
    "KALMAN_DAMPING": ("float", 0.70, 0.999),
    "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": ("float", 1.0, 20.0),
    "KALMAN_INITIAL_VELOCITY_RETENTION": ("float", 0.0, 1.0),
    "KALMAN_YOUNG_GATE_MULTIPLIER": ("float", 1.0, 4.0),
    "LOST_THRESHOLD_FRAMES": ("int", 2, 25),
    "KALMAN_MATURITY_AGE": ("int", 1, 20),
}


class OptimizationResult:
    def __init__(
        self,
        params: Dict[str, Any],
        score: float,
        trial_number: int,
        sub_scores: Dict[str, float] | None = None,
    ):
        self.params = params
        self.score = score
        self.trial_number = trial_number
        self.sub_scores: Dict[str, float] = sub_scores or {}


class TrackingOptimizer(QThread):
    """
    Runs Bayesian optimization on a video slice with different parameter sets.
    Requires a pre-populated DetectionCache for speed.
    """

    progress_signal = Signal(int, str)
    result_signal = Signal(list)  # List of OptimizationResult
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        base_params: Dict[str, Any],
        tuning_config: Dict[str, bool],
        n_trials: int = 50,
        n_seeds: int = 3,
        on_plateau: str = "restart",
        sampler_type: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.base_params = base_params
        self.tuning_config = tuning_config
        self.n_trials = n_trials
        self.n_seeds = max(1, n_seeds)
        self.on_plateau = on_plateau  # "restart" | "stop"
        self.sampler_type = sampler_type  # "auto" | "gp" | "tpe"
        self._stop_requested = False
        self.cache = None

    # Alias to module-level constant so existing internal references keep working.
    _PARAM_RANGES = _PARAM_RANGES  # type: ignore[assignment]

    def _build_sampler(self, n_active: int):
        """
        Construct the Optuna sampler based on ``self.sampler_type``.

        "auto"  — OptunaHub AutoSampler: uses GPSampler for early trials then
                  falls back to TPE.  Best overall choice.
        "gp"    — GPSampler with Matérn-2.5 + ARD + log-EI.  Fastest convergence
                  for budgets ≤ 500 trials; needs scipy + torch.
        "tpe"   — Multivariate TPE.  Robust fallback that needs no extra deps.
        """
        stype = self.sampler_type

        if stype == "auto":
            try:
                import optunahub  # type: ignore[import-untyped]

                return optunahub.load_module("samplers/auto_sampler").AutoSampler()
            except Exception as e:
                logger.warning(
                    "AutoSampler unavailable (%s), falling back to GPSampler.", e
                )
                stype = "gp"  # fall through to GP

        if stype == "gp":
            try:
                qmc = optuna.samplers.QMCSampler(
                    qmc_type="sobol",
                    seed=42,
                    independent_sampler=optuna.samplers.RandomSampler(seed=42),
                )
                return optuna.samplers.GPSampler(
                    seed=42,
                    n_startup_trials=max(10, n_active),
                    deterministic_objective=True,  # tracking is fully deterministic
                    independent_sampler=qmc,
                )
            except Exception as e:
                logger.warning(
                    "GPSampler unavailable (%s), falling back to multivariate TPE.", e
                )
                # fall through to tpe

        # "tpe" (default / fallback)
        return optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=max(20, n_active * 2),
            seed=42,
        )

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self.cache = DetectionCache(self.detection_cache_path, mode="r")
            if not self.cache.is_compatible():
                self.progress_signal.emit(0, "Error: Incompatible detection cache.")
                self.cache.close()
                return
        except Exception as e:
            self.progress_signal.emit(0, f"Error loading cache: {str(e)}")
            return

        # Validate that the cache covers the requested frame range before wasting time
        if not self.cache.covers_frame_range(self.start_frame, self.end_frame):
            missing = self.cache.get_missing_frames(self.start_frame, self.end_frame)
            msg = f"Error: Cache does not cover frames {self.start_frame}-{self.end_frame}. Missing: {missing}"
            self.progress_signal.emit(0, msg)
            self.cache.close()
            return

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_active = sum(1 for v in self.tuning_config.values() if v)

        sampler = self._build_sampler(n_active)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        results = []

        # Pre-calculated scaled body size for pixel conversions
        ref_size = self.base_params.get("REFERENCE_BODY_SIZE", 20.0)
        resize_f = self.base_params.get("RESIZE_FACTOR", 1.0)
        scaled_body_size = ref_size * resize_f

        # Pre-load pose data once so _run_tracking_loop never touches the NPZ.
        # Opening and reading the pose cache per-trial (and per-frame) is the
        # dominant cost when pose is enabled; the data never changes between trials.
        self._pose_run_context = (
            None,
            [],
            [],
            [],
            False,
        )  # (cache, ant, post, ign, enabled)
        self._pose_frame_cache: dict = {}  # {frame_idx: {det_id: keypoints}}
        _pose_ctx = _pf_load_pose_context(self.base_params)
        if (
            _pose_ctx[0] is not None and _pose_ctx[4]
        ):  # cache opened and direction enabled
            _tmp_cache, _ant, _post, _ign, _enabled = _pose_ctx
            self._pose_run_context = (None, _ant, _post, _ign, _enabled)
            for _fi in range(self.start_frame, self.end_frame + 1):
                self._pose_frame_cache[_fi] = _pf_build_keypoint_map(_tmp_cache, _fi)
            try:
                _tmp_cache.close()
            except Exception:
                pass
            logger.info(
                "Optimizer: pre-loaded pose keypoints for %d frames into memory.",
                len(self._pose_frame_cache),
            )
        else:
            # Pose disabled or cache absent — release any opened file immediately.
            if _pose_ctx[0] is not None:
                try:
                    _pose_ctx[0].close()
                except Exception:
                    pass

        # Seed the first trial with the user's current production settings so the
        # surrogate model has an immediate reference point and explores outward from it.
        # Values are clamped to each parameter's valid range so that base_params with
        # zero or out-of-range entries (e.g. KALMAN_NOISE_COVARIANCE = 0.0) never
        # produce an invalid enqueued trial — especially critical for log-scale params
        # where the value must be strictly positive.
        _DEFAULTS = {
            "YOLO_CONFIDENCE_THRESHOLD": 0.25,
            "YOLO_IOU_THRESHOLD": 0.7,
            "MAX_DISTANCE_MULTIPLIER": 3.0,
            "KALMAN_NOISE_COVARIANCE": 0.03,
            "KALMAN_MEASUREMENT_NOISE_COVARIANCE": 0.1,
            "W_POSITION": 1.0,
            "W_ORIENTATION": 0.5,
            "W_AREA": 0.2,
            "W_ASPECT": 0.2,
            "KALMAN_DAMPING": 0.95,
            "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER": 5.0,
            "KALMAN_INITIAL_VELOCITY_RETENTION": 0.2,
            "KALMAN_YOUNG_GATE_MULTIPLIER": 1.5,
            "LOST_THRESHOLD_FRAMES": 10,
            "KALMAN_MATURITY_AGE": 5,
        }
        seed_params: Dict[str, Any] = {}
        for key, (ptype, low, high) in self._PARAM_RANGES.items():
            if not self.tuning_config.get(key):
                continue
            raw = self.base_params.get(key, _DEFAULTS.get(key, low))
            if ptype == "int":
                seed_params[key] = int(np.clip(int(raw), low, high))
            else:
                # For log_float, clamp to [low, high] so value is strictly positive.
                seed_params[key] = float(np.clip(float(raw), low, high))
        if seed_params:
            study.enqueue_trial(seed_params)

        # Additional diversity seeds: random points spread across the full search space.
        # A deterministic RNG (seed=42) makes seeds reproducible.  Each extra seed
        # gives TPE a different region to model from, reducing local-optima risk.
        _rng_seeds = np.random.default_rng(42)

        def _perturb_near_base(rng, scale: float) -> dict:
            """Sample a parameter point perturbed around base_params.

            Works in the natural space for each parameter type:
            - log_float: Normal(log(base), scale*(log(high)-log(low))), then exp.
            - float    : Normal(base, scale*(high-low)), clipped to [low, high].
            - int      : same as float, then rounded and clipped.

            ``scale`` is the fraction of the total range used as σ.  A scale of 0.1
            stays close to the current settings; 0.3 allows more exploration.
            """
            pt: Dict[str, Any] = {}
            for key, (ptype, low, high) in self._PARAM_RANGES.items():
                if not self.tuning_config.get(key):
                    continue
                base_val = self.base_params.get(key)
                if ptype == "log_float":
                    log_low, log_high = np.log(low), np.log(high)
                    log_center = (
                        np.log(float(base_val))
                        if base_val is not None
                        else (log_low + log_high) / 2.0
                    )
                    sigma = scale * (log_high - log_low)
                    pt[key] = float(
                        np.exp(
                            np.clip(rng.normal(log_center, sigma), log_low, log_high)
                        )
                    )
                elif ptype == "float":
                    center = (
                        float(base_val) if base_val is not None else (low + high) / 2.0
                    )
                    sigma = scale * (high - low)
                    pt[key] = float(np.clip(rng.normal(center, sigma), low, high))
                else:  # int
                    center = (
                        float(base_val) if base_val is not None else (low + high) / 2.0
                    )
                    sigma = scale * (high - low)
                    pt[key] = int(np.clip(round(rng.normal(center, sigma)), low, high))
            return pt

        def _random_from_ranges(rng) -> dict:
            """Uniform-random point across the full search space (used for plateau restarts)."""
            pt: Dict[str, Any] = {}
            for key, (ptype, low, high) in self._PARAM_RANGES.items():
                if not self.tuning_config.get(key):
                    continue
                if ptype == "log_float":
                    pt[key] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                elif ptype == "float":
                    pt[key] = float(rng.uniform(low, high))
                else:  # int
                    pt[key] = int(rng.integers(low, high + 1))
            return pt

        # Diversity seeds fan out from base_params: tight for early seeds, wider for
        # later ones.  σ grows linearly from 10 % to 30 % of each parameter's range
        # so the initial probes respect domain knowledge while still giving the
        # surrogate model spread.
        n_extra = self.n_seeds - 1
        for i in range(n_extra):
            scale = 0.10 + 0.20 * (i / max(1, n_extra - 1)) if n_extra > 1 else 0.15
            extra_seed = _perturb_near_base(_rng_seeds, scale)
            if extra_seed:
                study.enqueue_trial(extra_seed)

        # A separate unseeded RNG for plateau restarts so each restart is genuinely
        # different (not deterministically replaying the diversity seeds).
        _rng_restart = np.random.default_rng()

        # Plateau detection: patience counted in consecutive non-improving trials.
        _PLATEAU_PATIENCE = max(15, self.n_trials // 5)
        _no_improve_count = 0
        _best_score_seen = float("inf")

        def objective(trial):
            nonlocal _no_improve_count, _best_score_seen
            if self._stop_requested:
                raise optuna.TrialPruned()

            trial_params = {}

            # Suggest only selected parameters
            if self.tuning_config.get("YOLO_CONFIDENCE_THRESHOLD"):
                trial_params["YOLO_CONFIDENCE_THRESHOLD"] = trial.suggest_float(
                    "YOLO_CONFIDENCE_THRESHOLD", 0.05, 0.8
                )

            # The detection cache stores low-confidence, pre-NMS raw detections.
            # Both YOLO_CONFIDENCE_THRESHOLD and YOLO_IOU_THRESHOLD are applied
            # post-hoc via DetectionFilter, so both can be meaningfully tuned.
            if self.tuning_config.get("YOLO_IOU_THRESHOLD"):
                trial_params["YOLO_IOU_THRESHOLD"] = trial.suggest_float(
                    "YOLO_IOU_THRESHOLD", 0.1, 0.9
                )

            if self.tuning_config.get("MAX_DISTANCE_MULTIPLIER"):
                trial_params["MAX_DISTANCE_MULTIPLIER"] = trial.suggest_float(
                    "MAX_DISTANCE_MULTIPLIER", 0.5, 3.0
                )
                trial_params["MAX_DISTANCE_THRESHOLD"] = (
                    trial_params["MAX_DISTANCE_MULTIPLIER"] * scaled_body_size
                )

            if self.tuning_config.get("KALMAN_NOISE_COVARIANCE"):
                trial_params["KALMAN_NOISE_COVARIANCE"] = trial.suggest_float(
                    "KALMAN_NOISE_COVARIANCE", 0.001, 0.2, log=True
                )

            if self.tuning_config.get("KALMAN_MEASUREMENT_NOISE_COVARIANCE"):
                trial_params["KALMAN_MEASUREMENT_NOISE_COVARIANCE"] = (
                    trial.suggest_float(
                        "KALMAN_MEASUREMENT_NOISE_COVARIANCE", 0.01, 1.0, log=True
                    )
                )

            if self.tuning_config.get("W_POSITION"):
                trial_params["W_POSITION"] = trial.suggest_float("W_POSITION", 0.1, 5.0)

            if self.tuning_config.get("W_ORIENTATION"):
                trial_params["W_ORIENTATION"] = trial.suggest_float(
                    "W_ORIENTATION", 0.0, 5.0
                )

            if self.tuning_config.get("W_AREA"):
                trial_params["W_AREA"] = trial.suggest_float("W_AREA", 0.0, 2.0)

            if self.tuning_config.get("W_ASPECT"):
                trial_params["W_ASPECT"] = trial.suggest_float("W_ASPECT", 0.0, 2.0)

            if self.tuning_config.get("KALMAN_DAMPING"):
                trial_params["KALMAN_DAMPING"] = trial.suggest_float(
                    "KALMAN_DAMPING", 0.70, 0.999
                )

            if self.tuning_config.get("KALMAN_LONGITUDINAL_NOISE_MULTIPLIER"):
                trial_params["KALMAN_LONGITUDINAL_NOISE_MULTIPLIER"] = (
                    trial.suggest_float(
                        "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER", 1.0, 20.0
                    )
                )

            if self.tuning_config.get("KALMAN_INITIAL_VELOCITY_RETENTION"):
                trial_params["KALMAN_INITIAL_VELOCITY_RETENTION"] = trial.suggest_float(
                    "KALMAN_INITIAL_VELOCITY_RETENTION", 0.0, 1.0
                )

            if self.tuning_config.get("KALMAN_YOUNG_GATE_MULTIPLIER"):
                trial_params["KALMAN_YOUNG_GATE_MULTIPLIER"] = trial.suggest_float(
                    "KALMAN_YOUNG_GATE_MULTIPLIER", 1.0, 4.0
                )

            if self.tuning_config.get("LOST_THRESHOLD_FRAMES"):
                trial_params["LOST_THRESHOLD_FRAMES"] = trial.suggest_int(
                    "LOST_THRESHOLD_FRAMES", 2, 25
                )

            if self.tuning_config.get("KALMAN_MATURITY_AGE"):
                trial_params["KALMAN_MATURITY_AGE"] = trial.suggest_int(
                    "KALMAN_MATURITY_AGE", 1, 20
                )

            # Merge into full param set
            current_params = self.base_params.copy()
            current_params.update(trial_params)

            # Run tracking simulation and compute composite score
            composite, sub_scores, _ = self._run_tracking_loop(current_params)
            score = composite

            results.append(
                OptimizationResult(trial_params, score, trial.number, sub_scores)
            )
            pct = int(((trial.number + 1) / self.n_trials) * 100)
            self.progress_signal.emit(
                pct, f"Trial {trial.number + 1}/{self.n_trials} (Score: {score:.3f})"
            )

            # Plateau detection
            if score < _best_score_seen:
                _best_score_seen = score
                _no_improve_count = 0
            else:
                _no_improve_count += 1
                if _no_improve_count >= _PLATEAU_PATIENCE:
                    if self.on_plateau == "restart":
                        # Inject a random restart so the study continues exploring
                        # instead of grinding in the same neighbourhood.
                        restart_pt = _random_from_ranges(_rng_restart)
                        if restart_pt:
                            study.enqueue_trial(restart_pt)
                        _no_improve_count = 0
                    else:
                        self._stop_requested = True

            return score

        try:
            study.optimize(objective, n_trials=self.n_trials)
        except Exception as e:
            logger.error(f"Optimization trial failed: {e}")

        results.sort(key=lambda x: x.score)
        self.cache.close()
        self._pose_frame_cache = None  # free memory after optimization
        try:
            self.result_signal.emit(results)
            self.finished_signal.emit()
        except RuntimeError as e:
            # Dialog was closed before the thread finished; ignore orphaned signals.
            logger.warning("Optimizer signal emission skipped (dialog closed?): %s", e)

    def _run_tracking_loop(self, params: Dict[str, Any], reverse: bool = False):
        """
        Core tracking simulation shared by quality scoring and the FB consistency metric.

        Returns
        -------
        composite : float
            Lower-is-better composite score balancing four orthogonal objectives:
            coverage (are all animals tracked?), assignment quality (how confident
            are matches?), fragmentation (are trajectories long and unbroken?), and
            occlusion rate (how often are animals transiently missed?). A spread
            penalty prevents gaming any single dimension.
        frame_positions : Dict[int, np.ndarray]
            Maps frame_idx -> (N, 2) float32 array of KF-estimated positions.
            Rows for lost tracks contain NaN.
        """
        det_filter = DetectionFilter(params)
        kf_manager = KalmanFilterManager(params["MAX_TARGETS"], params)
        assigner = TrackAssigner(params)
        _roi_mask = params.get("ROI_MASK", None)

        # --- Pose context: use pre-loaded in-memory data when available ---
        # _pose_run_context is populated once in run() before study.optimize();
        # if it is not set (e.g. called standalone), fall back to live load.
        if hasattr(self, "_pose_run_context") and self._pose_run_context is not None:
            _, _pose_anterior, _pose_posterior, _pose_ignore, _pose_enabled = (
                self._pose_run_context
            )
            _pose_frame_data = getattr(self, "_pose_frame_cache", {})
        else:
            (
                _tmp_pose_cache,
                _pose_anterior,
                _pose_posterior,
                _pose_ignore,
                _pose_enabled,
            ) = _pf_load_pose_context(params)
            _pose_frame_data = {}
            if _tmp_pose_cache is not None:
                for _fi in range(self.start_frame, self.end_frame + 1):
                    _pose_frame_data[_fi] = _pf_build_keypoint_map(_tmp_pose_cache, _fi)
                try:
                    _tmp_pose_cache.close()
                except Exception:
                    pass
        _pose_min_conf = float(params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
        track_pose_prototypes: list = [None] * params["MAX_TARGETS"]

        # ── Composite objective accumulators (one per scored dimension) ──────────
        # Each dimension is normalised to [0, 1]; lower = better.
        _coverage_sum = 0.0  # sum of (active_tracks / N) per frame
        _occlusion_sum = 0.0  # sum of (occluded_tracks / N) per frame
        _assign_cost_sum = 0.0  # sum of per-assignment body-size-normalised costs
        _assign_count = 0  # total matched pairs seen
        _suspicious_assign_count = 0  # assignments at > 2 × body_size (far-field)
        _det_count_sum = 0  # total post-filter detections across frames
        # ─────────────────────────────────────────────────────────────────────────

        N = params["MAX_TARGETS"]
        _max_continuity = [0] * N  # longest unbroken tracking run per track slot
        # Start all tracks as "lost" so Phase-3 initialises them from the first
        # frame's detections.  "active" + zero KF state gives bad initial costs.
        track_states = ["lost"] * N
        tracking_continuity = [0] * N

        # ── Parameter-derived constants (constant across the loop) ────────────
        _body_size = max(
            params.get("REFERENCE_BODY_SIZE", 20.0) * params.get("RESIZE_FACTOR", 1.0),
            5.0,
        )
        _n_pairs = max(N * (N - 1) // 2, 1)  # max pair count for crowding normalisation

        # ── User-configurable scoring weights ─────────────────────────────────
        # Read from params so the Scoring Weights UI can control them per-session.
        # Negative values are clamped to zero; all six are normalised to sum=1
        # at runtime so the user can use any convenient scale (e.g. 0–100).
        # A weight of 0 fully disables that objective AND excludes it from the
        # spread-penalty calculation — critical for crowding on ant datasets
        # where physical crowding is normal biology, not a tracking failure.
        _w_cov = max(float(params.get("SCORE_WEIGHT_COVERAGE", 0.25)), 0.0)
        _w_asn = max(float(params.get("SCORE_WEIGHT_ASSIGNMENT", 0.15)), 0.0)
        _w_frg = max(float(params.get("SCORE_WEIGHT_FRAGMENTATION", 0.20)), 0.0)
        _w_occ = max(float(params.get("SCORE_WEIGHT_OCCLUSION", 0.10)), 0.0)
        _w_vel = max(float(params.get("SCORE_WEIGHT_VELOCITY", 0.20)), 0.0)
        _w_crd = max(float(params.get("SCORE_WEIGHT_CROWDING", 0.10)), 0.0)
        _w_sum = _w_cov + _w_asn + _w_frg + _w_occ + _w_vel + _w_crd
        if _w_sum < 1e-9:  # all-zero guard: default to equal weights minus crowding
            _w_cov = _w_asn = _w_frg = _w_occ = _w_vel = 0.2
            _w_crd = 0.0
            _w_sum = 1.0
        _w_cov /= _w_sum
        _w_asn /= _w_sum
        _w_frg /= _w_sum
        _w_occ /= _w_sum
        _w_vel /= _w_sum
        _w_crd /= _w_sum

        # ── New per-frame accumulators ────────────────────────────────────────
        _step_norms: list = []  # body-size-normalised step distances (for percentile)
        _direction_reversals: list = (
            []
        )  # 0/1 markers: was this step a heading reversal?
        _crowding_sum = 0.0  # sum of normalised per-frame crowding violation
        _crowding_frames = 0  # frames counted for crowding normalisation
        _prev_positions: Dict[int, np.ndarray] = {}  # slot → last active position
        _prev_vecs: Dict[int, np.ndarray] = (
            {}
        )  # slot → last significant displacement vector
        missed_frames = [0] * N
        trajectory_ids = list(range(N))
        next_trajectory_id = N
        last_shape_info = [None] * N
        # Last committed theta per slot; used for OBB axis-flip disambiguation.
        orientation_last: list = [None] * N
        # Per-track EMA of inter-frame step size — mirrors worker.py track_avg_step.
        # Used by the advanced cost matrix to scale the motion gate adaptively.
        track_avg_step = np.zeros(N, dtype=np.float32)
        lost_threshold = params.get("LOST_THRESHOLD_FRAMES", 5)
        n_frames = self.end_frame - self.start_frame + 1

        frame_order = (
            range(self.end_frame, self.start_frame - 1, -1)
            if reverse
            else range(self.start_frame, self.end_frame + 1)
        )
        frame_positions: Dict[int, np.ndarray] = {}

        for f_idx in frame_order:
            if self._stop_requested:
                break

            (
                raw_meas,
                raw_sizes,
                raw_shapes,
                raw_confs,
                raw_obb,
                raw_det_ids,
                raw_heading_hints,
                raw_directed_mask,
            ) = self.cache.get_frame(f_idx)
            # Pass heading hints when present so directed-OBB data flows through
            # filter/NMS, exactly mirroring the TrackingWorker cached-path.
            if raw_heading_hints:
                filtered = det_filter.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    roi_mask=_roi_mask,
                    detection_ids=raw_det_ids,
                    heading_hints=raw_heading_hints,
                    directed_mask=raw_directed_mask,
                )
                (
                    meas,
                    _,
                    shapes,
                    _confs,
                    _obb_out,
                    detection_ids,
                    _headtail_hints,
                    _headtail_directed,
                ) = filtered
            else:
                filtered = det_filter.filter_raw_detections(
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    roi_mask=_roi_mask,
                    detection_ids=raw_det_ids,
                )
                meas, _, shapes, _confs, _obb_out, detection_ids = filtered
                _headtail_hints, _headtail_directed = [], []

            # --- Per-frame pose features (from in-memory pre-loaded dict) ---
            _det_pose_kpts: list = [None] * len(meas)
            _det_pose_vis = np.zeros(len(meas), dtype=np.float32)
            _det_pose_headings: list = [None] * len(meas)
            if _pose_enabled and meas and detection_ids:
                _frame_kpt_map = _pose_frame_data.get(f_idx, {})
                _det_pose_kpts, _det_pose_vis, _det_pose_headings = (
                    _pf_compute_det_features(
                        [int(d) for d in detection_ids],
                        _frame_kpt_map,
                        _pose_anterior,
                        _pose_posterior,
                        _pose_ignore,
                        _pose_min_conf,
                        return_headings=True,
                    )
                )
            _association_data: dict = {
                "detection_pose_keypoints": _det_pose_kpts,
                "detection_pose_visibility": _det_pose_vis,
                "track_pose_prototypes": track_pose_prototypes,
                "track_avg_step": track_avg_step.copy(),
            }
            # Build per-detection directed mask, mirroring worker.py heading logic.
            detection_directed_mask = np.zeros(len(meas), dtype=np.uint8)
            if len(meas) > 0:
                _pose_overrides_ht = bool(params.get("POSE_OVERRIDES_HEADTAIL", True))
                for _di in range(len(meas)):
                    _ph = (
                        _det_pose_headings[_di]
                        if _di < len(_det_pose_headings)
                        else None
                    )
                    _pd = _ph is not None and math.isfinite(float(_ph))
                    _hth = (
                        float(_headtail_hints[_di])
                        if _headtail_hints and _di < len(_headtail_hints)
                        else math.nan
                    )
                    _htd = (
                        bool(_headtail_directed[_di])
                        if _headtail_directed and _di < len(_headtail_directed)
                        else False
                    )
                    _, _is_dir = _pf_select_directed_heading(
                        float(_ph) if _pd else math.nan,
                        _pd,
                        _hth,
                        _htd,
                        _pose_overrides_ht,
                    )
                    if _is_dir:
                        detection_directed_mask[_di] = 1

            if meas:
                # Advance KF only when detections exist — mirrors worker.py where
                # get_predictions() (which calls predict()) is gated on
                # `detection_initialized and meas`.
                kf_manager.predict()

                # ── Cost matrix + Hungarian assignment ──────────────────────────
                cost, _spatial_candidates = assigner.compute_cost_matrix(
                    N,
                    meas,
                    kf_manager.X,
                    shapes,
                    kf_manager,
                    last_shape_info,
                    meas_ori_directed=(
                        detection_directed_mask
                        if len(detection_directed_mask) == len(meas)
                        else None
                    ),
                    association_data=_association_data,
                )
                matched_r, matched_c, free_dets, next_trajectory_id, _ = (
                    assigner.assign_tracks(
                        cost,
                        N,
                        len(meas),
                        meas,
                        track_states,
                        tracking_continuity,
                        kf_manager,
                        trajectory_ids,
                        next_trajectory_id,
                        _spatial_candidates,
                    )
                )
                respawned_matches = {r for r in matched_r if track_states[r] == "lost"}

                # Snapshot predicted positions BEFORE correct() for innovation
                # scoring (how far the KF prediction was from the measurement).
                _pre_correct: Dict[int, np.ndarray] = {
                    r: kf_manager.X[r, :2].copy() for r in matched_r
                }

                # For Phase-3 respawned tracks (lost→matched), init state first.
                _feature_alpha = float(params.get("TRACK_FEATURE_EMA_ALPHA", 0.85))
                _high_conf_thresh = float(
                    params.get("ASSOCIATION_HIGH_CONFIDENCE_THRESHOLD", 0.7)
                )
                for r, c in zip(matched_r, matched_c):
                    m = np.asarray(meas[c], dtype=np.float32)
                    _pose_d = (
                        bool(detection_directed_mask[c])
                        if c < len(detection_directed_mask)
                        else False
                    )
                    # Match worker.py: pass KF-predicted theta as fallback for
                    # OBB axis-flip disambiguation when orientation_last is None.
                    theta_cor = _pf_resolve_tracking_theta(
                        r,
                        m[2],
                        _pose_d,
                        orientation_last,
                        fallback_theta=(
                            float(kf_manager.X[r, 2]) if r < len(kf_manager.X) else None
                        ),
                    )
                    m_cor = np.array([m[0], m[1], theta_cor], dtype=np.float32)
                    if r in respawned_matches:
                        _prev_positions.pop(r, None)
                        _prev_vecs.pop(r, None)
                        track_avg_step[r] = 0.0
                        kf_manager.initialize_filter(
                            r,
                            np.array(
                                [m_cor[0], m_cor[1], theta_cor, 0.0, 0.0],
                                dtype=np.float32,
                            ),
                        )
                    kf_manager.correct(r, m_cor)
                    curr = kf_manager.X[r, :2].copy()
                    orientation_last[r] = _pf_normalize_theta(float(kf_manager.X[r, 2]))
                    # Update track_avg_step EMA — mirrors worker.py speed tracking
                    # so the advanced cost matrix gets accurate motion gate values.
                    _det_conf = float(_confs[c]) if _confs and c < len(_confs) else 0.0
                    if r in _prev_positions and _det_conf >= _high_conf_thresh:
                        _step = float(np.linalg.norm(curr - _prev_positions[r]))
                        track_avg_step[r] = (
                            _feature_alpha * float(track_avg_step[r])
                            + (1.0 - _feature_alpha) * _step
                        )

                # Innovation cost: skip freshly-respawned "lost" tracks (no prior).
                for r, c in zip(matched_r, matched_c):
                    if track_states[r] == "lost":
                        continue
                    pixel_dist = float(
                        np.linalg.norm(
                            np.asarray(meas[c][:2], dtype=np.float32) - _pre_correct[r]
                        )
                    )
                    # Normalise by 2 × reference body size — a fixed physical scale.
                    # Using MAX_DISTANCE_THRESHOLD here would be maladaptive: the
                    # optimizer could inflate MAX_DISTANCE_MULTIPLIER to make long-
                    # range (bad) assignments appear cheap. A match one body-length
                    # away scores ~0.5; anything beyond 2 body-lengths is capped 1.0.
                    _assign_cost_sum += min(
                        pixel_dist / max(2.0 * _body_size, 1e-6), 1.0
                    )
                    # Only flag the assignment as suspicious for MATURE tracks.
                    # Young tracks (continuity < KALMAN_MATURITY_AGE) legitimately
                    # use a wider gate because their KF predictions are unreliable;
                    # penalising them here would push KALMAN_YOUNG_GATE_MULTIPLIER
                    # toward 1.0, defeating its purpose.
                    _is_young = tracking_continuity[r] < params.get(
                        "KALMAN_MATURITY_AGE", 5
                    )
                    if pixel_dist > 2.0 * _body_size and not _is_young:
                        _suspicious_assign_count += 1
                    _assign_count += 1

                # Keep track pose prototypes current for next frame's association.
                for r, c in zip(matched_r, matched_c):
                    proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
                    if proto is not None:
                        track_pose_prototypes[r] = np.asarray(
                            proto, dtype=np.float32
                        ).copy()

                matched_r_set = set(matched_r)
                for r in matched_r:
                    missed_frames[r] = 0
                    track_states[r] = "active"
                    tracking_continuity[r] += 1
                for r in range(N):
                    if r not in matched_r_set and track_states[r] != "lost":
                        missed_frames[r] += 1
                        if missed_frames[r] >= lost_threshold:
                            track_states[r] = "lost"
                            tracking_continuity[r] = 0
                        else:
                            track_states[r] = "occluded"
                for r, c in zip(matched_r, matched_c):
                    last_shape_info[r] = shapes[c]

                # ── Free detections → respawn lost tracks ────────────────────────
                # assigner.assign_tracks() Phase-3 uses a distance gate that fails
                # when lost tracks are at (0, 0) on the first frame (or after a long
                # gap).  This explicit loop mirrors worker.py lines 2324-2391, which
                # has no distance check and simply claims the first available lost slot.
                for d_idx in free_dets:
                    for track_idx in range(N):
                        if track_states[track_idx] == "lost":
                            _pose_d_f = (
                                bool(detection_directed_mask[d_idx])
                                if d_idx < len(detection_directed_mask)
                                else False
                            )
                            theta_init = _pf_resolve_tracking_theta(
                                track_idx,
                                float(meas[d_idx][2]),
                                _pose_d_f,
                                orientation_last,
                                fallback_theta=(
                                    float(kf_manager.X[track_idx, 2])
                                    if track_idx < len(kf_manager.X)
                                    else None
                                ),
                            )
                            kf_manager.initialize_filter(
                                track_idx,
                                np.array(
                                    [
                                        meas[d_idx][0],
                                        meas[d_idx][1],
                                        theta_init,
                                        0.0,
                                        0.0,
                                    ],
                                    dtype=np.float32,
                                ),
                            )
                            track_states[track_idx] = "active"
                            missed_frames[track_idx] = 0
                            tracking_continuity[track_idx] = 0
                            trajectory_ids[track_idx] = next_trajectory_id
                            next_trajectory_id += 1
                            orientation_last[track_idx] = theta_init
                            last_shape_info[track_idx] = (
                                shapes[d_idx] if d_idx < len(shapes) else None
                            )
                            track_pose_prototypes[track_idx] = (
                                np.asarray(
                                    _det_pose_kpts[d_idx], dtype=np.float32
                                ).copy()
                                if (
                                    d_idx < len(_det_pose_kpts)
                                    and _det_pose_kpts[d_idx] is not None
                                )
                                else None
                            )
                            track_avg_step[track_idx] = 0.0
                            break
            else:
                for r in range(N):
                    if track_states[r] != "lost":
                        missed_frames[r] += 1
                        if missed_frames[r] >= lost_threshold:
                            track_states[r] = "lost"
                            tracking_continuity[r] = 0
                        else:
                            track_states[r] = "occluded"

            # Per-frame coverage accounting (after state management)
            for r in range(N):
                if (
                    track_states[r] == "active"
                    and tracking_continuity[r] > _max_continuity[r]
                ):
                    _max_continuity[r] = tracking_continuity[r]
            _coverage_sum += sum(1 for r in range(N) if track_states[r] == "active") / N
            _occlusion_sum += (
                sum(1 for r in range(N) if track_states[r] == "occluded") / N
            )
            _det_count_sum += len(meas)

            # ── Inter-frame velocity + direction-reversal (ID-swap penalty) ─────────
            # Two complementary signals are tracked:
            #
            #  1. Displacement magnitude (step_norms): catches "teleportation" swaps
            #     where the two animals were far apart at the crossing moment.
            #
            #  2. Direction reversals: a track whose heading flips >107° in one
            #     frame (cos < -0.3) almost certainly suffered an ID swap.  Physical
            #     animals have inertia — they do not reverse course in a single frame.
            #     This fires on both far-apart AND tight-crossing swaps, making the
            #     combined sub-score much more specific to identity failures.
            #
            # Only steps above 0.2 body-lengths are considered for direction so that
            # stationary animals (random Kalman drift) don't pollute the signal.
            _MOVE_MIN = 0.2 * _body_size
            for r in range(N):
                if track_states[r] != "lost":
                    curr = kf_manager.X[r, :2].copy()
                    if r in _prev_positions:
                        step_vec = curr - _prev_positions[r]
                        step = float(np.linalg.norm(step_vec))
                        _step_norms.append(step / max(_body_size, 1e-6))
                        if step > _MOVE_MIN and r in _prev_vecs:
                            prev_v = _prev_vecs[r]
                            prev_norm = float(np.linalg.norm(prev_v))
                            if prev_norm > _MOVE_MIN:
                                cos_a = float(
                                    np.dot(step_vec, prev_v) / (step * prev_norm)
                                )
                                _direction_reversals.append(
                                    1.0 if cos_a < -0.3 else 0.0
                                )
                        if step > _MOVE_MIN:
                            _prev_vecs[r] = step_vec
                    _prev_positions[r] = curr
                else:
                    _prev_positions.pop(r, None)  # reset; don't bridge across lost gap
                    _prev_vecs.pop(r, None)

            # ── Crowding / false-merge proximity penalty ──────────────────────
            # Any pair of active tracks within REFERENCE_BODY_SIZE pixels likely
            # represents a false merge (one animal, two track slots) or an
            # imminent identity swap.  We normalise the per-frame violation by the
            # maximum possible pair count so the sub-score stays in [0, 1].
            _crowding_frames += 1
            active_slots = [r for r in range(N) if track_states[r] == "active"]
            if len(active_slots) >= 2:
                frame_crowd = 0.0
                for _ci in range(len(active_slots)):
                    for _cj in range(_ci + 1, len(active_slots)):
                        _d = float(
                            np.linalg.norm(
                                kf_manager.X[active_slots[_ci], :2]
                                - kf_manager.X[active_slots[_cj], :2]
                            )
                        )
                        if _d < _body_size:
                            frame_crowd += 1.0 - _d / max(_body_size, 1e-6)
                _crowding_sum += frame_crowd / _n_pairs

            # Record estimated positions (NaN = lost track)
            pos = np.full((N, 2), np.nan, dtype=np.float32)
            for r in range(N):
                if track_states[r] != "lost":
                    pos[r] = kf_manager.X[r, :2]
            frame_positions[f_idx] = pos

        # ── Composite multi-objective score ──────────────────────────────────────
        # Four orthogonal sub-scores, each in [0, 1] (0 = perfect):
        #
        #   coverage_cost  – fraction of time any animal is permanently lost
        #   assign_cost    – mean normalised match cost (tracking uncertainty)
        #   frag_cost      – how intermittent tracks are vs. longest unbroken run
        #   occlusion_cost – mean fraction of animals in transient occlusion
        #
        # The spread penalty (std of sub-scores × 0.3) punishes configurations
        # that ace one dimension at the expense of another, driving the optimiser
        # toward balanced solutions rather than degenerate extremes.
        #
        # Anti-exploitation design principles applied to each sub-score:
        #
        #  coverage    – includes a detection-density penalty so that lowering
        #                YOLO_CONFIDENCE_THRESHOLD past the point of collecting
        #                genuine detections (mean > 1.2×N) does not help.
        #
        #  assignment  – residuals are normalised by 2×body_size, not by the
        #                tunable MAX_DISTANCE_THRESHOLD.  Inflating the gate cannot
        #                shrink the score.
        #
        #  fragmentation – blends raw continuity (70%) with the suspicious-assign
        #                rate (30%) so that greedy far-field matching that keeps
        #                tracks alive does not yield a low fragmentation score.
        _cov_frac = _coverage_sum / max(n_frames, 1)
        # Penalise mean detection count well above N: indicates the confidence
        # threshold is too permissive and is flooding the assigner with noise.
        _mean_dets = _det_count_sum / max(n_frames, 1)
        _det_excess = min(
            max(_mean_dets / max(N, 1.0) - 1.2, 0.0) / max(N * 0.5, 1.0), 1.0
        )
        coverage_cost = 0.80 * (1.0 - _cov_frac) + 0.20 * _det_excess
        assign_cost = _assign_cost_sum / _assign_count if _assign_count > 0 else 1.0
        _cont_frac = min(sum(_max_continuity) / max(N, 1) / max(n_frames, 1), 1.0)
        _suspicious_rate = _suspicious_assign_count / max(_assign_count, 1)
        # Fragmentation: 70% raw continuity + 30% suspicious-assignment rate.
        # Long runs built on plausible close matches score well; runs sustained
        # only by wide-gate greedy matching score poorly.
        frag_cost = 0.70 * (1.0 - _cont_frac) + 0.30 * _suspicious_rate
        occlusion_cost = _occlusion_sum / max(n_frames, 1)
        # Velocity sub-score: two-component blend.
        #
        #  Magnitude component (40%): median + 95th-percentile of displacement
        #  magnitude, body-size normalised.  Catches far-apart teleportation swaps.
        #    3 body-lengths/frame mean  → score 1.0
        #    8 body-lengths  p95        → score 1.0
        #
        #  Direction-reversal component (60%): fraction of eligible step-pairs
        #  where the heading reversed >107° (cos < -0.3).  Catches tight-crossing
        #  swaps where the magnitude spike is small.  Physical animals rarely
        #  reverse course in a single frame; any such event is likely a swap.
        if _step_norms:
            _steps_arr = np.asarray(_step_norms, dtype=np.float32)
            _vel_median = float(np.median(_steps_arr))
            _vel_p95 = float(np.percentile(_steps_arr, 95))
            magnitude_cost = min(
                0.5 * min(_vel_median / 3.0, 1.0) + 0.5 * min(_vel_p95 / 8.0, 1.0),
                1.0,
            )
        else:
            magnitude_cost = 1.0
        direction_cost = (
            float(np.mean(_direction_reversals)) if _direction_reversals else 0.0
        )
        velocity_cost = 0.4 * magnitude_cost + 0.6 * direction_cost
        crowding_cost = _crowding_sum / max(_crowding_frames, 1)

        sub_scores = {
            "coverage": coverage_cost,
            "assignment": assign_cost,
            "fragmentation": frag_cost,
            "occlusion": occlusion_cost,
            "velocity": velocity_cost,
            "crowding": crowding_cost,
        }
        # Weights sum to 1.0:
        #   coverage(0.25) + assignment(0.15) + fragmentation(0.20)
        #   + occlusion(0.10) + velocity(0.20) + crowding(0.10) = 1.00
        # velocity and crowding are weighted heavily because they directly
        # target false merges and inter-frame jumps — the most damaging failure
        # modes in practice.
        sub_scores_arr = np.array(
            [
                coverage_cost,
                assign_cost,
                frag_cost,
                occlusion_cost,
                velocity_cost,
                crowding_cost,
            ],
            dtype=np.float64,
        )
        weights = np.array(
            [_w_cov, _w_asn, _w_frg, _w_occ, _w_vel, _w_crd], dtype=np.float64
        )
        composite = float(np.dot(weights, sub_scores_arr))
        # Spread penalty: only consider actively-weighted dimensions so that a
        # zero-weighted term (e.g. crowding disabled for ants) does not inflate
        # the variance and bias the optimiser away from otherwise good solutions.
        active_mask = weights > 1e-9
        if active_mask.sum() > 1:
            composite += 0.3 * float(np.std(sub_scores_arr[active_mask]))

        return composite, sub_scores, frame_positions

    def _run_tracking_pass_cached(self, params: Dict[str, Any]) -> float:
        """Thin wrapper retained for backwards compatibility."""
        score, _, _ = self._run_tracking_loop(params)
        return score


class DetectionCacheBuilderWorker(QThread):
    """
    Phase-1-only worker: runs YOLO detection on a frame range and writes a
    DetectionCache.  Does NOT run Kalman tracking, pose precompute, CSV
    writing, interpolation, or any other production-pipeline stage.

    Used by the Parameter Helper to build a minimal detection cache so the
    Bayesian optimizer can run without triggering the full tracking pipeline.
    """

    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)  # (success, cache_path)

    def __init__(
        self,
        video_path: str,
        cache_path: str,
        params: Dict[str, Any],
        start_frame: int,
        end_frame: int,
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_path = cache_path
        self.params = params.copy()
        self.start_frame = start_frame
        self.end_frame = end_frame
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        import time
        from collections import deque

        from multi_tracker.core.detectors.engine import create_detector
        from multi_tracker.utils.batch_optimizer import BatchOptimizer

        # --- Load detector (YOLO model) ---
        try:
            detector = create_detector(self.params)
        except Exception as e:
            logger.error("DetectionCacheBuilder: could not create detector: %s", e)
            self.finished_signal.emit(False, "")
            return

        cap = cv2.VideoCapture(self.video_path)
        cache = None
        try:
            if not cap.isOpened():
                logger.error(
                    "DetectionCacheBuilder: could not open video: %s", self.video_path
                )
                self.finished_signal.emit(False, "")
                return

            resize_f = self.params.get("RESIZE_FACTOR", 1.0)
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_f)
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_f)

            advanced = self.params.get("ADVANCED_CONFIG", {}).copy()
            advanced["enable_tensorrt"] = self.params.get("ENABLE_TENSORRT", False)
            advanced["tensorrt_max_batch_size"] = self.params.get(
                "TENSORRT_MAX_BATCH_SIZE", 16
            )
            batch_size = BatchOptimizer(advanced).estimate_batch_size(
                fw, fh, self.params.get("YOLO_MODEL_PATH", "")
            )

            cache = DetectionCache(
                self.cache_path,
                mode="w",
                start_frame=self.start_frame,
                end_frame=self.end_frame,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            total_frames = self.end_frame - self.start_frame + 1
            rel_idx = 0  # 0-based position relative to start_frame
            batch_times: deque = deque(maxlen=30)

            while rel_idx < total_frames and not self._stop_requested:
                batch_start = rel_idx
                batch_frames = []
                while len(batch_frames) < batch_size and rel_idx < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if resize_f != 1.0:
                        frame = cv2.resize(
                            frame,
                            (0, 0),
                            fx=resize_f,
                            fy=resize_f,
                            interpolation=cv2.INTER_AREA,
                        )
                    batch_frames.append(frame)
                    rel_idx += 1

                if not batch_frames:
                    break

                bt0 = time.time()
                results = detector.detect_objects_batched(
                    batch_frames, batch_start, None, return_raw=True
                )
                batch_times.append(time.time() - bt0)

                for local_i, (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_hints,
                    raw_directed,
                ) in enumerate(results):
                    actual_f = self.start_frame + batch_start + local_i
                    det_ids = [actual_f * 10000 + i for i in range(len(raw_meas))]
                    cache.add_frame(
                        actual_f,
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        det_ids,
                        raw_hints,
                        raw_directed,
                    )

                pct = int(rel_idx * 100 / total_frames)
                avg = sum(batch_times) / len(batch_times) if batch_times else 0
                fps = (len(batch_frames) / avg) if avg > 0 else 0
                eta_s = int((total_frames - rel_idx) / fps) if fps > 0 else -1
                eta_str = f"  ETA {eta_s}s" if eta_s >= 0 else ""
                self.progress_signal.emit(
                    pct, f"Building detection cache: {pct}%{eta_str}"
                )

            if self._stop_requested:
                self.progress_signal.emit(0, "Cancelled.")
                self.finished_signal.emit(False, "")
                return

            cache.save()
            cache.close()
            cache = None
            logger.info("DetectionCacheBuilder: cache saved to %s", self.cache_path)
            self.finished_signal.emit(True, self.cache_path)

        except Exception:
            logger.exception("DetectionCacheBuilder error")
            self.finished_signal.emit(False, "")
        finally:
            cap.release()
            if cache is not None:
                try:
                    cache.close()
                except Exception:
                    pass


class TrackingPreviewWorker(QThread):
    """
    Emits visualization frames for previewing optimization results.
    """

    frame_signal = Signal(np.ndarray)
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        detection_cache_path: str,
        start_frame: int,
        end_frame: int,
        params: Dict[str, Any],
        parent=None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.params = params
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        cache = DetectionCache(self.detection_cache_path, mode="r")
        try:
            if not cap.isOpened():
                logger.error("PreviewWorker: could not open video: %s", self.video_path)
                return
            if not cache.is_compatible():
                logger.error("PreviewWorker: incompatible detection cache.")
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            kf_manager = KalmanFilterManager(self.params["MAX_TARGETS"], self.params)
            assigner = TrackAssigner(self.params)
            det_filter = DetectionFilter(self.params)
            _roi_mask = self.params.get("ROI_MASK", None)

            N = self.params["MAX_TARGETS"]

            # --- Pose context ---
            (
                _pose_cache,
                _pose_anterior,
                _pose_posterior,
                _pose_ignore,
                _pose_enabled,
            ) = _pf_load_pose_context(self.params)
            _pose_min_conf = float(self.params.get("POSE_MIN_KPT_CONF_VALID", 0.2))
            _pose_kpt_map: dict = {}
            _pose_kpt_map_frame = None
            track_pose_prototypes: list = [None] * N
            # Start all tracks as "lost" so Phase-3 bootstraps each slot from the
            # first frame's real detections via initialize_filter.  Starting as
            # "active" with a zero-initialised KF state causes every track to sit
            # at (0, 0) = top-left corner until it happens to match a detection.
            track_states, tracking_continuity = ["lost"] * N, [0] * N
            missed_frames = [0] * N
            trajectory_ids, next_trajectory_id = list(range(N)), N
            # Last committed theta per slot for OBB axis-flip disambiguation.
            orientation_last: list = [None] * N
            last_shape_info = [None] * N
            lost_threshold = self.params.get("LOST_THRESHOLD_FRAMES", 5)

            resize_f = self.params.get("RESIZE_FACTOR", 1.0)

            # Per-track color: use TRAJECTORY_COLORS from params (same seed as main window)
            traj_colors = self.params.get("TRAJECTORY_COLORS", [])
            if not traj_colors:
                np.random.seed(42)
                traj_colors = [
                    tuple(int(c) for c in row)
                    for row in np.random.randint(0, 255, (max(N, 32), 3))
                ]

            # Trail history: deque of (x, y) per track slot (slot index, not ID)
            # FPS is stored as "FPS" by get_parameters_dict; "VIDEO_FPS" is the
            # legacy preview-only key — check both so old param dicts still work.
            _fps = self.params.get("FPS") or self.params.get("VIDEO_FPS", 25)
            _TRAIL_LEN = int(
                self.params.get("TRAJECTORY_HISTORY_SECONDS", 5) * max(_fps, 1)
            )
            trail: list[deque] = [deque(maxlen=max(_TRAIL_LEN, 10)) for _ in range(N)]

            show_circles = self.params.get("SHOW_CIRCLES", True)
            show_orientation = self.params.get("SHOW_ORIENTATION", True)
            show_trails = self.params.get("SHOW_TRAJECTORIES", True)
            show_labels = self.params.get("SHOW_LABELS", True)

            for f_idx in range(self.start_frame, self.end_frame + 1):
                if self._stop_requested:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                (
                    raw_meas,
                    raw_sizes,
                    raw_shapes,
                    raw_confs,
                    raw_obb,
                    raw_det_ids,
                    raw_heading_hints,
                    raw_directed_mask,
                ) = cache.get_frame(f_idx)
                if raw_heading_hints:
                    filtered = det_filter.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        roi_mask=_roi_mask,
                        detection_ids=raw_det_ids,
                        heading_hints=raw_heading_hints,
                        directed_mask=raw_directed_mask,
                    )
                    (
                        meas,
                        _,
                        shapes,
                        _confs,
                        _obb_out,
                        detection_ids,
                        _headtail_hints,
                        _headtail_directed,
                    ) = filtered
                else:
                    filtered = det_filter.filter_raw_detections(
                        raw_meas,
                        raw_sizes,
                        raw_shapes,
                        raw_confs,
                        raw_obb,
                        roi_mask=_roi_mask,
                        detection_ids=raw_det_ids,
                    )
                    meas, _, shapes, _confs, _obb_out, detection_ids = filtered
                    _headtail_hints, _headtail_directed = [], []

                kf_manager.predict()

                # --- Per-frame pose features ---
                _det_pose_kpts: list = [None] * len(meas)
                _det_pose_vis = np.zeros(len(meas), dtype=np.float32)
                _det_pose_headings: list = [None] * len(meas)
                if _pose_enabled and meas and detection_ids:
                    if _pose_kpt_map_frame != f_idx:
                        _pose_kpt_map = _pf_build_keypoint_map(_pose_cache, f_idx)
                        _pose_kpt_map_frame = f_idx
                    _det_pose_kpts, _det_pose_vis, _det_pose_headings = (
                        _pf_compute_det_features(
                            [int(d) for d in detection_ids],
                            _pose_kpt_map,
                            _pose_anterior,
                            _pose_posterior,
                            _pose_ignore,
                            _pose_min_conf,
                            return_headings=True,
                        )
                    )
                _association_data: dict = {
                    "detection_pose_keypoints": _det_pose_kpts,
                    "detection_pose_visibility": _det_pose_vis,
                    "track_pose_prototypes": track_pose_prototypes,
                    "track_avg_step": np.zeros(N, dtype=np.float32),
                }
                # Build per-detection directed mask, mirroring worker.py.
                detection_directed_mask = np.zeros(len(meas), dtype=np.uint8)
                if len(meas) > 0:
                    _pose_overrides_ht = bool(
                        self.params.get("POSE_OVERRIDES_HEADTAIL", True)
                    )
                    for _di in range(len(meas)):
                        _ph = (
                            _det_pose_headings[_di]
                            if _di < len(_det_pose_headings)
                            else None
                        )
                        _pd = _ph is not None and math.isfinite(float(_ph))
                        _hth = (
                            float(_headtail_hints[_di])
                            if _headtail_hints and _di < len(_headtail_hints)
                            else math.nan
                        )
                        _htd = (
                            bool(_headtail_directed[_di])
                            if _headtail_directed and _di < len(_headtail_directed)
                            else False
                        )
                        _, _is_dir = _pf_select_directed_heading(
                            float(_ph) if _pd else math.nan,
                            _pd,
                            _hth,
                            _htd,
                            _pose_overrides_ht,
                        )
                        if _is_dir:
                            detection_directed_mask[_di] = 1

                if meas:
                    cost, _ = assigner.compute_cost_matrix(
                        N,
                        meas,
                        kf_manager.X,
                        shapes,
                        kf_manager,
                        last_shape_info,
                        meas_ori_directed=(
                            detection_directed_mask
                            if len(detection_directed_mask) == len(meas)
                            else None
                        ),
                        association_data=_association_data,
                    )
                    matched_r, matched_c, free_dets, next_trajectory_id, _ = (
                        assigner.assign_tracks(
                            cost,
                            N,
                            len(meas),
                            meas,
                            track_states,
                            tracking_continuity,
                            kf_manager,
                            trajectory_ids,
                            next_trajectory_id,
                        )
                    )
                    # For Phase-3 respawned tracks (lost→matched), initialize before correct()
                    for r, c in zip(matched_r, matched_c):
                        m = np.asarray(meas[c], dtype=np.float32)
                        # Disambiguate OBB axis-flip before committing to KF.
                        _pose_d = (
                            bool(detection_directed_mask[c])
                            if c < len(detection_directed_mask)
                            else False
                        )
                        theta_cor = _pf_resolve_tracking_theta(
                            r, m[2], _pose_d, orientation_last
                        )
                        m_cor = np.array([m[0], m[1], theta_cor], dtype=np.float32)
                        if track_states[r] == "lost":
                            trail[r].clear()
                            kf_manager.initialize_filter(
                                r,
                                np.array(
                                    [m_cor[0], m_cor[1], theta_cor, 0.0, 0.0],
                                    dtype=np.float32,
                                ),
                            )
                        kf_manager.correct(r, m_cor)
                        orientation_last[r] = _pf_normalize_theta(
                            float(kf_manager.X[r, 2])
                        )

                    # Keep per-track pose prototypes current.
                    for r, c in zip(matched_r, matched_c):
                        proto = _det_pose_kpts[c] if c < len(_det_pose_kpts) else None
                        if proto is not None:
                            track_pose_prototypes[r] = np.asarray(
                                proto, dtype=np.float32
                            ).copy()

                    # Initialize lost tracks from unmatched detections — mirrors worker.py
                    newly_initialized: set = set()
                    existing_matched = set(matched_r)
                    for d_idx in free_dets:
                        for r in range(N):
                            if (
                                r not in existing_matched | newly_initialized
                                and track_states[r] == "lost"
                            ):
                                m = np.asarray(meas[d_idx], dtype=np.float32)
                                _pose_d = (
                                    bool(detection_directed_mask[d_idx])
                                    if d_idx < len(detection_directed_mask)
                                    else False
                                )
                                theta_cor = _pf_resolve_tracking_theta(
                                    r, m[2], _pose_d, orientation_last
                                )
                                kf_manager.initialize_filter(
                                    r,
                                    np.array(
                                        [m[0], m[1], theta_cor, 0.0, 0.0],
                                        dtype=np.float32,
                                    ),
                                )
                                trail[r].clear()
                                orientation_last[r] = _pf_normalize_theta(theta_cor)
                                track_states[r] = "active"
                                missed_frames[r] = 0
                                tracking_continuity[r] = 0
                                trajectory_ids[r] = next_trajectory_id
                                next_trajectory_id += 1
                                newly_initialized.add(r)
                                # Seed pose prototype from unmatched detection.
                                proto = (
                                    _det_pose_kpts[d_idx]
                                    if d_idx < len(_det_pose_kpts)
                                    else None
                                )
                                if proto is not None:
                                    track_pose_prototypes[r] = np.asarray(
                                        proto, dtype=np.float32
                                    ).copy()
                                break
                else:
                    matched_r, matched_c, newly_initialized = [], [], set()

                # --- State management (mirrors worker.py logic) ---
                matched_r_set = set(matched_r) | newly_initialized
                for r in matched_r:
                    missed_frames[r] = 0
                    track_states[r] = "active"
                    tracking_continuity[r] += 1
                for r in range(N):
                    if r not in matched_r_set and track_states[r] != "lost":
                        missed_frames[r] += 1
                        if missed_frames[r] >= lost_threshold:
                            track_states[r] = "lost"
                            tracking_continuity[r] = 0
                        else:
                            track_states[r] = "occluded"
                for r, c in zip(matched_r, matched_c):
                    last_shape_info[r] = shapes[c]

                # Update trails
                for r in range(N):
                    if track_states[r] != "lost":
                        x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
                        if math.isfinite(x) and math.isfinite(y):
                            trail[r].append((int(x), int(y)))
                    else:
                        trail[r].clear()
                # --------------------------------------------------

                display = cv2.resize(frame, (0, 0), fx=resize_f, fy=resize_f)

                for r in range(N):
                    if track_states[r] == "lost":
                        continue
                    # Color is keyed by trajectory_id, not slot index, so the same
                    # animal keeps its color even if it moves to a different slot.
                    col = traj_colors[trajectory_ids[r] % len(traj_colors)]
                    # Dim occluded tracks
                    draw_col = (
                        tuple(int(c * 0.5) for c in col)
                        if track_states[r] == "occluded"
                        else col
                    )

                    x, y = float(kf_manager.X[r, 0]), float(kf_manager.X[r, 1])
                    theta = float(kf_manager.X[r, 2])
                    if not (math.isfinite(x) and math.isfinite(y)):
                        continue

                    pt = (int(x), int(y))

                    # Trail
                    if show_trails and len(trail[r]) > 1:
                        pts = np.array(list(trail[r]), dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(
                            display, [pts], isClosed=False, color=draw_col, thickness=2
                        )

                    # Circle
                    if show_circles:
                        cv2.circle(display, pt, 7, draw_col, -1)

                    # Orientation arrow
                    if show_orientation:
                        ex = int(x + 18 * math.cos(theta))
                        ey = int(y + 18 * math.sin(theta))
                        cv2.arrowedLine(
                            display, pt, (ex, ey), draw_col, 2, tipLength=0.4
                        )

                    # Label: track ID + state
                    if show_labels:
                        state_tag = (
                            ""
                            if track_states[r] == "active"
                            else f" ({track_states[r]})"
                        )
                        cv2.putText(
                            display,
                            f"T{trajectory_ids[r]}{state_tag}",
                            (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            draw_col,
                            1,
                            cv2.LINE_AA,
                        )

                self.frame_signal.emit(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                self.msleep(20)  # Fast preview; msleep is interruptible by stop()
        except Exception:
            logger.exception("PreviewWorker encountered an error.")
        finally:
            cap.release()
            cache.close()
            if _pose_cache is not None:
                try:
                    _pose_cache.close()
                except Exception:
                    pass
            self.finished_signal.emit()
