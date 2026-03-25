"""Optuna-based optimizer for background-subtraction detection parameters.

Samples frames from the video, primes the background model once, then
runs Bayesian optimisation over threshold / morphology / split parameters
with three self-contained objectives that avoid the circular dependency
between detection params and REFERENCE_BODY_SIZE:

1. **Count accuracy** – fraction of sampled frames where the number of
   detections equals ``MAX_TARGETS``.
2. **Size consistency** – within each frame that has the correct count,
   how uniform are the detection areas? (1 − coefficient of variation).
3. **Size stability** – across frames, how stable is the median detection
   area? (1 − CoV of per-frame medians).

None of these require REFERENCE_BODY_SIZE as input, so the optimizer
can run *before* body-size calibration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore[assignment]

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter ranges  (mirrors _PARAM_RANGES in tracking/optimizer.py)
# ---------------------------------------------------------------------------

_BG_PARAM_RANGES: Dict[str, tuple] = {
    "THRESHOLD_VALUE": (5, 80, "int"),
    "MORPH_KERNEL_SIZE": (3, 15, "odd"),
    "MIN_CONTOUR_AREA": (10, 500, "int"),
    "ENABLE_ADDITIONAL_DILATION": (False, True, "bool"),
    "DILATION_KERNEL_SIZE": (3, 11, "odd"),
    "DILATION_ITERATIONS": (1, 5, "int"),
    "ENABLE_CONSERVATIVE_SPLIT": (False, True, "bool"),
    "CONSERVATIVE_KERNEL_SIZE": (1, 11, "int"),
    "CONSERVATIVE_ERODE_ITER": (1, 5, "int"),
}

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BgOptimizationResult:
    """One completed trial from the BG optimiser."""

    params: Dict[str, Any]
    score: float
    trial_number: int
    sub_scores: Dict[str, float] = field(default_factory=dict)
    median_area: float = 0.0  # median detection area in pixels²


# ---------------------------------------------------------------------------
# Optimizer thread
# ---------------------------------------------------------------------------


class BgSubtractionOptimizer(QThread):
    """QThread that runs an Optuna study to tune detection parameters."""

    progress_signal = Signal(int, str)  # (percentage, message)
    result_signal = Signal(list)  # list[BgOptimizationResult]
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        base_params: Dict[str, Any],
        tuning_config: Dict[str, bool],
        scoring_weights: Dict[str, float] | None = None,
        n_trials: int = 50,
        n_sample_frames: int = 30,
        sampler_type: str = "tpe",
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = dict(base_params)
        self.tuning_config = tuning_config
        self.scoring_weights = scoring_weights or {
            "SCORE_WEIGHT_COUNT": 0.50,
            "SCORE_WEIGHT_CONSISTENCY": 0.30,
            "SCORE_WEIGHT_STABILITY": 0.20,
        }
        self.n_trials = n_trials
        self.n_sample_frames = n_sample_frames
        self.sampler_type = sampler_type
        self._stop_requested = False

    # ------------------------------------------------------------------
    def _build_sampler(self, n_active: int):
        """Construct the Optuna sampler based on *sampler_type*.

        "auto" — OptunaHub AutoSampler (GP early, TPE later).
        "gp"   — GPSampler (Matérn-2.5, ARD, log-EI).
        "tpe"  — Multivariate TPE (robust fallback).
        """
        stype = self.sampler_type

        if stype == "auto":
            try:
                import optunahub  # type: ignore[import-untyped]

                return optunahub.load_module(
                    "samplers/auto_sampler",
                ).AutoSampler()
            except Exception as e:
                logger.warning(
                    "AutoSampler unavailable (%s), falling back to GP.",
                    e,
                )
                stype = "gp"

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
                    deterministic_objective=True,
                    independent_sampler=qmc,
                )
            except Exception as e:
                logger.warning(
                    "GPSampler unavailable (%s), falling back to TPE.",
                    e,
                )

        # "tpe" (default / fallback)
        return optuna.samplers.TPESampler(
            multivariate=True,
            n_startup_trials=max(20, n_active * 2),
            seed=42,
        )

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901  (complex but self-contained)
        try:
            self._run_optimization()
        except Exception as e:
            logger.exception("BG-subtraction optimization failed")
            self.progress_signal.emit(0, f"Error: {e}")
        finally:
            self.finished_signal.emit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_optimization(self) -> None:
        if optuna is None:
            self.progress_signal.emit(0, "Error: optuna is not installed")
            return

        from ...utils.image_processing import apply_image_adjustments
        from ..background.model import BackgroundModel
        from .engine import ObjectDetector

        params = self.base_params

        # --- 1. open video -------------------------------------------------
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.progress_signal.emit(0, "Error: cannot open video")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = params.get("START_FRAME", 0)
        end = min(params.get("END_FRAME", total_frames - 1), total_frames - 1)
        prime_n = params.get("BACKGROUND_PRIME_FRAMES", 30)

        # --- 2. prime background model -------------------------------------
        self.progress_signal.emit(0, "Priming background model …")
        bg_model = BackgroundModel(params)
        bg_model.prime_background(cap)

        if bg_model.lightest_background is None:
            cap.release()
            self.progress_signal.emit(0, "Error: background priming failed")
            return

        bg_float = bg_model.lightest_background  # float32
        bg_u8 = cv2.convertScaleAbs(bg_float)

        # --- 3. choose sample frames --------------------------------------
        first_valid = start + prime_n
        if first_valid >= end:
            first_valid = start
        n_available = end - first_valid + 1
        n_sample = min(self.n_sample_frames, max(n_available, 1))
        sample_indices = np.linspace(first_valid, end, n_sample, dtype=int)

        resize_f = params.get("RESIZE_FACTOR", 1.0)
        dark_on_light = params.get("DARK_ON_LIGHT_BACKGROUND", True)

        # --- 4. pre-compute diff images ------------------------------------
        self.progress_signal.emit(5, "Reading sample frames …")
        diffs: List[np.ndarray] = []
        grays: List[np.ndarray] = []

        for idx in sample_indices:
            if self._stop_requested:
                cap.release()
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            if resize_f < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_f,
                    fy=resize_f,
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(
                gray,
                params.get("BRIGHTNESS", 0),
                params.get("CONTRAST", 1.0),
                params.get("GAMMA", 1.0),
                use_gpu=False,
            )
            if dark_on_light:
                diff = cv2.subtract(bg_u8, gray)
            else:
                diff = cv2.subtract(gray, bg_u8)
            grays.append(gray)
            diffs.append(diff)

        cap.release()

        if not diffs:
            self.progress_signal.emit(0, "Error: no frames could be read")
            return

        # --- 5. scoring setup -----------------------------------------------
        max_targets = params.get("MAX_TARGETS", 5)

        # Pre-resize ROI mask
        roi_mask = params.get("ROI_MASK")
        if roi_mask is not None and resize_f < 1.0:
            roi_mask = cv2.resize(
                roi_mask,
                (grays[0].shape[1], grays[0].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # --- 6. run Optuna -------------------------------------------------
        self.progress_signal.emit(10, "Starting optimisation \u2026")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_active = sum(1 for v in self.tuning_config.values() if v)
        n_frames = len(diffs)

        # Pruning: report intermediate count scores and let Optuna kill
        # obviously-bad trials early (e.g. wrong threshold → 0 detections).
        # Trials report every PRUNE_INTERVAL frames; the pruner kicks in
        # after n_warmup_steps reports and n_startup_trials completed trials.
        _PRUNE_INTERVAL = max(1, n_frames // 6)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1,
            interval_steps=1,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=self._build_sampler(n_active),
            pruner=pruner,
        )
        results: List[BgOptimizationResult] = []

        # Normalise scoring weights
        sw = self.scoring_weights
        w_cnt_r = sw.get("SCORE_WEIGHT_COUNT", 0.50)
        w_con_r = sw.get("SCORE_WEIGHT_CONSISTENCY", 0.30)
        w_stb_r = sw.get("SCORE_WEIGHT_STABILITY", 0.20)
        w_sum = w_cnt_r + w_con_r + w_stb_r
        if w_sum < 1e-9:
            w_cnt, w_con, w_stb = 1 / 3, 1 / 3, 1 / 3
        else:
            w_cnt = w_cnt_r / w_sum
            w_con = w_con_r / w_sum
            w_stb = w_stb_r / w_sum

        # Filter tuning_config to only enabled params
        tune = {k: v for k, v in self.tuning_config.items() if v}

        def objective(trial):  # noqa: C901
            if self._stop_requested:
                raise optuna.TrialPruned()

            # --- suggest or inherit each parameter -------------------------
            trial_params: Dict[str, Any] = {}

            # THRESHOLD_VALUE
            if "THRESHOLD_VALUE" in tune:
                trial_params["THRESHOLD_VALUE"] = trial.suggest_int(
                    "THRESHOLD_VALUE",
                    5,
                    80,
                )
            else:
                trial_params["THRESHOLD_VALUE"] = params["THRESHOLD_VALUE"]
            threshold = trial_params["THRESHOLD_VALUE"]

            # MORPH_KERNEL_SIZE (odd)
            if "MORPH_KERNEL_SIZE" in tune:
                morph_half = trial.suggest_int("MORPH_KERNEL_HALF", 1, 7)
                trial_params["MORPH_KERNEL_SIZE"] = morph_half * 2 + 1
            else:
                trial_params["MORPH_KERNEL_SIZE"] = params["MORPH_KERNEL_SIZE"]
            morph_k = trial_params["MORPH_KERNEL_SIZE"]

            # MIN_CONTOUR_AREA
            if "MIN_CONTOUR_AREA" in tune:
                trial_params["MIN_CONTOUR_AREA"] = trial.suggest_int(
                    "MIN_CONTOUR_AREA",
                    10,
                    500,
                )
            else:
                trial_params["MIN_CONTOUR_AREA"] = params["MIN_CONTOUR_AREA"]

            # ENABLE_ADDITIONAL_DILATION group
            if "ENABLE_ADDITIONAL_DILATION" in tune:
                enable_dil = trial.suggest_categorical(
                    "ENABLE_ADDITIONAL_DILATION",
                    [True, False],
                )
            else:
                enable_dil = params.get("ENABLE_ADDITIONAL_DILATION", False)
            trial_params["ENABLE_ADDITIONAL_DILATION"] = enable_dil

            if enable_dil:
                if "DILATION_KERNEL_SIZE" in tune:
                    dil_half = trial.suggest_int("DILATION_KERNEL_HALF", 1, 5)
                    trial_params["DILATION_KERNEL_SIZE"] = dil_half * 2 + 1
                else:
                    trial_params["DILATION_KERNEL_SIZE"] = params.get(
                        "DILATION_KERNEL_SIZE",
                        3,
                    )
                if "DILATION_ITERATIONS" in tune:
                    trial_params["DILATION_ITERATIONS"] = trial.suggest_int(
                        "DILATION_ITERATIONS",
                        1,
                        5,
                    )
                else:
                    trial_params["DILATION_ITERATIONS"] = params.get(
                        "DILATION_ITERATIONS",
                        1,
                    )
            else:
                trial_params["DILATION_KERNEL_SIZE"] = params.get(
                    "DILATION_KERNEL_SIZE",
                    3,
                )
                trial_params["DILATION_ITERATIONS"] = params.get(
                    "DILATION_ITERATIONS",
                    1,
                )
            dil_k = trial_params["DILATION_KERNEL_SIZE"]
            dil_iter = trial_params["DILATION_ITERATIONS"]

            # ENABLE_CONSERVATIVE_SPLIT group
            if "ENABLE_CONSERVATIVE_SPLIT" in tune:
                enable_split = trial.suggest_categorical(
                    "ENABLE_CONSERVATIVE_SPLIT",
                    [True, False],
                )
            else:
                enable_split = params.get("ENABLE_CONSERVATIVE_SPLIT", False)
            trial_params["ENABLE_CONSERVATIVE_SPLIT"] = enable_split

            if enable_split:
                if "CONSERVATIVE_KERNEL_SIZE" in tune:
                    trial_params["CONSERVATIVE_KERNEL_SIZE"] = trial.suggest_int(
                        "CONSERVATIVE_KERNEL_SIZE",
                        1,
                        11,
                    )
                else:
                    trial_params["CONSERVATIVE_KERNEL_SIZE"] = params.get(
                        "CONSERVATIVE_KERNEL_SIZE",
                        3,
                    )
                if "CONSERVATIVE_ERODE_ITER" in tune:
                    trial_params["CONSERVATIVE_ERODE_ITER"] = trial.suggest_int(
                        "CONSERVATIVE_ERODE_ITER",
                        1,
                        5,
                    )
                else:
                    trial_params["CONSERVATIVE_ERODE_ITER"] = params.get(
                        "CONSERVATIVE_ERODE_ITER",
                        1,
                    )
            else:
                trial_params["CONSERVATIVE_KERNEL_SIZE"] = params.get(
                    "CONSERVATIVE_KERNEL_SIZE",
                    3,
                )
                trial_params["CONSERVATIVE_ERODE_ITER"] = params.get(
                    "CONSERVATIVE_ERODE_ITER",
                    1,
                )

            # --- build per-trial detector ----------------------------------
            det_params = dict(params)
            det_params.update(trial_params)
            detector = ObjectDetector(det_params)

            # Pre-build structuring elements once per trial (not per frame)
            morph_ker = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (morph_k, morph_k),
            )
            dil_ker = (
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_k, dil_k))
                if enable_dil
                else None
            )

            count_scores: List[float] = []
            consistency_scores: List[float] = []
            frame_medians: List[float] = []
            prune_step = 0

            for fi, diff in enumerate(diffs):
                if self._stop_requested:
                    raise optuna.TrialPruned()

                # threshold
                _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                # morph open + close
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_ker)
                fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, morph_ker)
                # additional dilation
                if dil_ker is not None:
                    fg = cv2.dilate(fg, dil_ker, iterations=dil_iter)
                # ROI
                if roi_mask is not None:
                    fg = cv2.bitwise_and(fg, fg, mask=roi_mask)
                # conservative split
                if enable_split:
                    fg = detector.apply_conservative_split(
                        fg,
                        grays[fi],
                        bg_u8,
                    )
                # detect
                meas, sizes, _shapes, _yolo, _conf = detector.detect_objects(
                    fg,
                    fi,
                )

                # --- count sub-score ----------------------------------------
                n_det = len(meas)
                if max_targets > 0:
                    err = abs(n_det - max_targets) / max_targets
                    count_scores.append(max(0.0, 1.0 - err))

                # --- size consistency sub-score -----------------------------
                if sizes and n_det == max_targets and n_det >= 2:
                    areas = np.array(sizes, dtype=float)
                    mean_a = areas.mean()
                    if mean_a > 1e-6:
                        cov = areas.std() / mean_a
                        consistency_scores.append(max(0.0, 1.0 - cov))
                    else:
                        consistency_scores.append(0.0)
                    frame_medians.append(float(np.median(areas)))
                elif sizes:
                    frame_medians.append(float(np.median(sizes)))

                # --- intermediate pruning -----------------------------------
                # Report running count score periodically so the pruner can
                # kill obviously-bad trials early (e.g. wrong threshold →
                # zero detections on every frame).
                if (fi + 1) % _PRUNE_INTERVAL == 0 or fi == n_frames - 1:
                    running = float(np.mean(count_scores)) if count_scores else 0.0
                    trial.report(running, step=prune_step)
                    prune_step += 1
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            # --- aggregate per-trial scores --------------------------------
            s_count = float(np.mean(count_scores)) if count_scores else 0.0
            s_consistency = (
                float(np.mean(consistency_scores)) if consistency_scores else 0.0
            )

            # Size stability: 1 - CoV of per-frame median areas
            if len(frame_medians) >= 2:
                med_arr = np.array(frame_medians)
                med_mean = med_arr.mean()
                if med_mean > 1e-6:
                    s_stability = max(0.0, 1.0 - med_arr.std() / med_mean)
                else:
                    s_stability = 0.0
            elif frame_medians:
                s_stability = 1.0  # single frame — no variation
            else:
                s_stability = 0.0

            score = w_cnt * s_count + w_con * s_consistency + w_stb * s_stability

            trial_median_area = (
                float(np.median(frame_medians)) if frame_medians else 0.0
            )

            results.append(
                BgOptimizationResult(
                    params=trial_params,
                    score=score,
                    trial_number=trial.number,
                    sub_scores={
                        "count": s_count,
                        "consistency": s_consistency,
                        "stability": s_stability,
                    },
                    median_area=trial_median_area,
                )
            )

            pct = int(10 + 85 * (trial.number + 1) / self.n_trials)
            self.progress_signal.emit(
                pct,
                f"Trial {trial.number + 1}/{self.n_trials} \u2014 "
                f"score {score:.3f}  "
                f"(cnt {s_count:.2f}, con {s_consistency:.2f}, "
                f"stb {s_stability:.2f})",
            )
            return score

        study.optimize(objective, n_trials=self.n_trials)

        # Store bg_u8 so the preview worker can reuse it
        self._cached_bg_u8 = bg_u8

        results.sort(key=lambda r: -r.score)
        self.progress_signal.emit(100, "Optimisation complete!")
        self.result_signal.emit(results)


# ---------------------------------------------------------------------------
# Detection preview worker
# ---------------------------------------------------------------------------


class BgDetectionPreviewWorker(QThread):
    """Generate annotated preview frames for a given detection parameter set.

    Reads the same sample frames as the optimiser, runs the full BG-sub
    detection pipeline for the chosen parameters, and emits (index, RGB)
    pairs so the dialog can display them.
    """

    # (frame_index_in_sample_list, rgb_numpy_array)
    frame_signal = Signal(int, object)
    finished_signal = Signal()

    def __init__(
        self,
        video_path: str,
        base_params: Dict[str, Any],
        trial_params: Dict[str, Any],
        n_sample_frames: int = 30,
        bg_u8: Optional[np.ndarray] = None,
        parent: Optional[Any] = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.base_params = dict(base_params)
        self.trial_params = dict(trial_params)
        self.n_sample_frames = n_sample_frames
        self._cached_bg_u8 = bg_u8
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._generate_previews()
        except Exception:
            logger.exception("BG detection preview failed")
        finally:
            self.finished_signal.emit()

    def _generate_previews(self) -> None:
        from ...utils.image_processing import apply_image_adjustments
        from .engine import ObjectDetector

        params = self.base_params
        det_params = dict(params)
        det_params.update(self.trial_params)

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start = params.get("START_FRAME", 0)
        end = min(params.get("END_FRAME", total_frames - 1), total_frames - 1)
        prime_n = params.get("BACKGROUND_PRIME_FRAMES", 30)

        # Reuse cached background if available; otherwise prime from scratch
        if self._cached_bg_u8 is not None:
            bg_u8 = self._cached_bg_u8
        else:
            from ..background.model import BackgroundModel

            bg_model = BackgroundModel(params)
            bg_model.prime_background(cap)
            if bg_model.lightest_background is None:
                cap.release()
                return
            bg_u8 = cv2.convertScaleAbs(bg_model.lightest_background)

        first_valid = start + prime_n
        if first_valid >= end:
            first_valid = start
        n_available = end - first_valid + 1
        n_sample = min(self.n_sample_frames, max(n_available, 1))
        sample_indices = np.linspace(first_valid, end, n_sample, dtype=int)

        resize_f = params.get("RESIZE_FACTOR", 1.0)
        dark_on_light = params.get("DARK_ON_LIGHT_BACKGROUND", True)

        threshold = det_params.get("THRESHOLD_VALUE", 25)
        morph_k = det_params.get("MORPH_KERNEL_SIZE", 5)
        enable_dil = det_params.get("ENABLE_ADDITIONAL_DILATION", False)
        dil_k = det_params.get("DILATION_KERNEL_SIZE", 3)
        dil_iter = det_params.get("DILATION_ITERATIONS", 1)
        enable_split = det_params.get("ENABLE_CONSERVATIVE_SPLIT", False)

        morph_ker = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_k, morph_k),
        )
        detector = ObjectDetector(det_params)

        roi_mask = params.get("ROI_MASK")

        for fi, idx in enumerate(sample_indices):
            if self._stop_requested:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            if resize_f < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_f,
                    fy=resize_f,
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(
                gray,
                params.get("BRIGHTNESS", 0),
                params.get("CONTRAST", 1.0),
                params.get("GAMMA", 1.0),
                use_gpu=False,
            )
            if dark_on_light:
                diff = cv2.subtract(bg_u8, gray)
            else:
                diff = cv2.subtract(gray, bg_u8)

            if roi_mask is not None and resize_f < 1.0:
                roi_r = cv2.resize(
                    roi_mask,
                    (gray.shape[1], gray.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                roi_r = roi_mask

            # detection pipeline
            _, fg = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_ker)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, morph_ker)
            if enable_dil:
                dk = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (dil_k, dil_k),
                )
                fg = cv2.dilate(fg, dk, iterations=dil_iter)
            if roi_r is not None:
                fg = cv2.bitwise_and(fg, fg, mask=roi_r)
            if enable_split:
                fg = detector.apply_conservative_split(fg, gray, bg_u8)

            meas, sizes, shapes, _yolo, _conf = detector.detect_objects(fg, fi)

            # draw detections on frame
            display = frame.copy()
            for j, (m, sz) in enumerate(zip(meas, sizes)):
                cx, cy = int(m[0]), int(m[1])
                radius = max(int((sz / 3.14159) ** 0.5), 3)
                cv2.circle(display, (cx, cy), radius, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    str(j + 1),
                    (cx + radius + 2, cy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            # count overlay
            max_t = params.get("MAX_TARGETS", 5)
            n_det = len(meas)
            clr = (
                (0, 255, 0)
                if n_det == max_t
                else ((0, 200, 255) if n_det < max_t else (0, 0, 255))
            )
            cv2.putText(
                display,
                f"Frame {int(idx)}  |  {n_det}/{max_t} detections",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                clr,
                1,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.frame_signal.emit(fi, rgb)

        cap.release()
