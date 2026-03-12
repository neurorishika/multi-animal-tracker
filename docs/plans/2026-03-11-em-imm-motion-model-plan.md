# EM Noise Estimation + IMM Filter — Implementation Plan

**Date:** 2026-03-11
**Branch:** `mat-pose-integration`
**Author:** Claude Code

---

## Motivation

The current Kalman filter uses a **fixed constant-velocity model** with hand-tuned (or Optuna-searched) noise parameters. Two complementary improvements, both **fully optional** (disabled by default):

1. **EM Noise Estimation** — After an initial forward pass, compute maximum-likelihood Q and R matrices from innovation residuals. Replaces 4 Optuna-tunable Kalman noise parameters with data-derived values. When enabled, a **second forward pass** runs with the calibrated parameters before the backward pass, so both full passes benefit from the improved noise model.

2. **Interacting Multiple Model (IMM)** — Run 2–3 parallel Kalman filters (stationary, constant-velocity, maneuvering) and blend their outputs based on observation likelihood. Handles animals that stop, cruise, and turn without a single compromise Q. EM estimation automatically calibrates per-mode Q matrices when both features are enabled together.

**Runtime cost:** Negligible. Kalman operations are 5×5 matrices (~0.2ms/frame for 20 animals with 3 IMM models vs ~0.05ms for single filter). Detection inference dominates at 50–200ms/frame. The second forward pass re-uses the detection cache (no GPU inference), so it costs only Kalman + assignment time (~5–20ms/frame).

**Why a second forward pass matters:** The forward pass is where identity is first committed. Better noise parameters improve Mahalanobis gating, which directly reduces ID swaps during occlusions. The backward pass cannot undo committed swaps — it can only add coverage — so re-running the forward pass with EM-calibrated params is essential for maximum benefit.

---

## Architecture Overview

### Tracking pass structure

**EM+IMM disabled (default — unchanged behavior):**
```
Forward pass  →  Backward pass  →  Post-processing
```

**EM enabled (standard or IMM mode):**
```
Forward pass (initial, default params)
        │
        ▼
EM Noise Estimation  (fit Q, R from innovations)
        │
        ▼
Second Forward pass  (with EM-calibrated params — cheap, uses detection cache)
        │
        ▼
Backward pass  (with EM-calibrated params)
        │
        ▼
Post-processing  (unchanged)
```

### Module relationships

```
                    ┌─────────────────────────┐
                    │   KalmanFilterManager    │  (existing, refactored)
                    │                         │
                    │  mode: "standard" | "imm"│
                    │                         │
                    │  ┌───────────────────┐  │
                    │  │  IMMFilterBank     │  │  (new, Phase 2)
                    │  │  ├─ mode 0: still  │  │
                    │  │  ├─ mode 1: cruise  │  │
                    │  │  └─ mode 2: maneuver│  │
                    │  └───────────────────┘  │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼──────┐  ┌────▼─────┐  ┌──────▼──────────┐
    │ EMEstimator    │  │ worker   │  │ optimizer        │
    │ (new, Phase 1) │  │ .py      │  │ .py (trimmed)    │
    │                │  │          │  │                  │
    │ fit(trajs) →   │  │ 3-pass   │  │ removes 4 Kalman │
    │   Q_hat, R_hat │  │ pipeline │  │ noise params     │
    └────────────────┘  └──────────┘  └──────────────────┘
```

---

## Phase 0 — Trajectory Segmentation Utility

**Goal:** Classify trajectory segments by motion regime. Used by both EM (to fit per-regime Q) and IMM (to calibrate transition probabilities).

### New file: `src/multi_tracker/core/filters/motion_segments.py`

```python
@dataclass
class MotionSegment:
    track_id: int
    start_frame: int
    end_frame: int
    regime: str  # "stationary" | "cruising" | "maneuvering"
    mean_speed: float
    mean_angular_velocity: float

def segment_trajectory(
    xs: np.ndarray, ys: np.ndarray, thetas: np.ndarray,
    body_size: float, fps: float,
    stationary_threshold: float = 0.1,   # body-lengths/frame
    maneuver_angular_threshold: float = 0.3,  # rad/frame
    min_segment_length: int = 5,
) -> List[MotionSegment]: ...

def segment_all_trajectories(
    df: pd.DataFrame, body_size: float, fps: float, **kwargs
) -> List[MotionSegment]: ...

def compute_regime_statistics(
    segments: List[MotionSegment]
) -> Dict[str, Dict[str, float]]:
    """Return per-regime stats: fraction of time, mean speed, mean duration."""
    ...
```

**Segmentation algorithm:**
1. Compute per-frame speed = `sqrt(dx^2 + dy^2)` and angular velocity = `|dtheta|`
2. Classify each frame: `stationary` if speed < threshold, `maneuvering` if angular_vel > threshold, else `cruising`
3. Run-length encode into contiguous segments
4. Merge short segments (< `min_segment_length`) into neighbors

**Tests:** `tests/test_motion_segments.py` — ~8 tests
- Stationary track classified correctly
- Straight-line track classified as cruising
- Sharp-turn track classified as maneuvering
- Mixed trajectory produces multiple segments
- Short segments merged into neighbors
- Empty/single-frame edge cases

---

## Phase 1 — EM Noise Estimation

**Goal:** Compute maximum-likelihood Q and R from tracking residuals, optionally per motion regime.

### New file: `src/multi_tracker/core/filters/em_noise_estimator.py`

```python
@dataclass
class NoiseEstimate:
    """Result of EM noise estimation."""
    Q_position: np.ndarray     # 2x2 (x,y position process noise)
    Q_velocity: np.ndarray     # 2x2 (vx,vy velocity process noise) — anisotropic
    Q_theta: float             # scalar theta process noise
    R: np.ndarray              # 3x3 measurement noise
    n_samples: int             # number of innovation samples used
    regime: str                # "all" | "stationary" | "cruising" | "maneuvering"

class EMNoiseEstimator:
    """Estimate Kalman noise matrices from trajectory data.

    Uses a two-pass approach:
    1. Run tracking with current (possibly default) Q, R
    2. Collect innovation sequences: y_t = z_t - H @ x_{t|t-1}
    3. Compute ML estimates: R_hat = (1/T) sum(y_t @ y_t^T) - H @ P_{t|t-1} @ H^T
                             Q_hat from smoothed state residuals via EM
    """

    def __init__(
        self,
        body_size: float,
        fps: float,
        min_samples: int = 50,
        per_regime: bool = True,
    ): ...

    def collect_innovations(
        self,
        track_idx: int,
        measurement: np.ndarray,      # [x, y, theta]
        predicted_state: np.ndarray,   # [x, y, theta, vx, vy]
        innovation_covariance: np.ndarray,  # S = H @ P @ H^T + R (3x3)
        frame_idx: int,
    ) -> None:
        """Called during tracking to accumulate innovation data."""
        ...

    def fit(
        self,
        segments: Optional[List[MotionSegment]] = None,
    ) -> Dict[str, NoiseEstimate]:
        """Compute ML noise estimates.

        Returns dict keyed by regime name ("all", "stationary", etc.).
        If per_regime=False or segments=None, returns {"all": estimate}.
        """
        ...

    def fit_from_trajectories(
        self,
        df: pd.DataFrame,
        kalman_params: Dict[str, Any],
    ) -> Dict[str, NoiseEstimate]:
        """Convenience: replay tracking on df, collect innovations, fit.

        This is the offline post-hoc path: given completed trajectories
        and the params that produced them, re-run the Kalman filter to
        extract innovations and estimate noise.
        """
        ...
```

### EM Algorithm Details

**Innovation-based R estimation (Mehra 1970):**
```
R_hat = (1/T) * sum_{t=1}^{T} [ y_t @ y_t^T ] - H @ P_bar @ H^T
```
where `P_bar` is the mean predicted covariance. This gives a direct ML estimate.

**Q estimation via state-space EM (Shumway & Stoffer):**
```
E-step:  Run Kalman smoother (forward filter + backward smoother)
M-step:  Q_hat = (1/T) * sum_{t=1}^{T} [ (x_t^s - F @ x_{t-1}^s) @ (...)^T + P_t^s - F @ P_{t-1,t}^s ]
```
where `x^s` = smoothed states, `P^s` = smoothed covariances, `P_{t-1,t}^s` = cross-covariance.

**Simplification for our use case:**
Since we have completed trajectories (measurements at every frame), we can use a simpler approach:
1. Compute velocity residuals: `v_residual_t = (x_{t+1} - x_t) - v_predicted_t`
2. Fit Q_velocity from `Cov(v_residual)` in body-aligned coordinates
3. This naturally captures the anisotropy (longitudinal vs lateral) from data

**Per-regime fitting:**
- Use `MotionSegment` labels to partition innovations by regime
- Fit separate `NoiseEstimate` per regime
- Each IMM model gets its regime-specific Q

### Integration point: `worker.py`

After the **initial forward pass** completes, optionally run EM then re-run the forward pass:

```python
# In TrackingWorker, after initial forward pass:
if params.get("EM_NOISE_ESTIMATION_ENABLED", False):
    estimator = EMNoiseEstimator(body_size, fps)
    segments = segment_all_trajectories(forward_df, body_size, fps)
    estimates = estimator.fit_from_trajectories(forward_df, params)
    # Update params for second forward + backward passes
    if "cruising" in estimates:
        params["KALMAN_NOISE_COVARIANCE"] = estimates["cruising"].Q_theta
        params["KALMAN_MEASUREMENT_NOISE_COVARIANCE"] = float(np.mean(np.diag(estimates["cruising"].R)))
        # ... etc
    # Re-init KalmanFilterManager with EM-calibrated params
    self.kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)
    # Run second forward pass (uses detection cache — no GPU inference)
    self.status_signal.emit("Re-running forward pass with EM-calibrated parameters...")
    forward_df = self._run_forward_pass(use_cache=True)
    # Backward pass and post-processing then use EM params automatically
```

### New config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `em_noise_estimation_enabled` | bool | false | Enable EM noise estimation after initial forward pass |
| `em_rerun_forward_pass` | bool | true | Re-run forward pass with EM-calibrated params (strongly recommended; fast due to detection cache) |
| `em_min_samples` | int | 50 | Minimum innovation samples per regime |
| `em_per_regime` | bool | true | Fit separate noise per motion regime |

### Tests: `tests/test_em_noise_estimator.py` — ~12 tests

- `test_collect_innovations_accumulates` — verify data is stored
- `test_fit_returns_noise_estimate` — basic output shape
- `test_r_estimate_matches_known_noise` — simulate with known R, verify recovery
- `test_q_estimate_matches_known_noise` — simulate with known Q, verify recovery
- `test_per_regime_produces_different_estimates` — stationary vs cruising Q differ
- `test_min_samples_guard` — returns None / falls back if insufficient data
- `test_fit_from_trajectories_end_to_end` — replay on a DataFrame
- `test_anisotropy_recovered` — longitudinal > lateral noise recovered from data
- `test_single_track_works` — no crash with just one trajectory
- `test_empty_dataframe_returns_none` — graceful handling
- `test_stationary_regime_has_low_q` — Q_velocity near zero for still animals
- `test_maneuvering_regime_has_high_q` — Q_velocity large for turning animals

---

## Phase 2 — IMM Filter

**Goal:** Replace the single Kalman filter with an Interacting Multiple Model filter that blends 2–3 motion models.

### New file: `src/multi_tracker/core/filters/imm_filter.py`

```python
@dataclass
class IMMMode:
    """One mode (motion model) within the IMM bank."""
    name: str                    # "stationary" | "cruising" | "maneuvering"
    F: np.ndarray               # 5x5 transition matrix
    Q: np.ndarray               # 5x5 process noise (from EM or defaults)
    damping: float              # velocity damping for this mode

class IMMFilterBank:
    """Interacting Multiple Model filter for a single track.

    Runs N parallel Kalman filters, each with a different motion model.
    At each timestep:
    1. Mix: blend prior states using transition probabilities
    2. Predict: each mode predicts independently
    3. Update: each mode updates with measurement, compute likelihood
    4. Combine: weighted combination of mode estimates
    """

    def __init__(
        self,
        modes: List[IMMMode],
        H: np.ndarray,                    # 3x5 measurement matrix (shared)
        R: np.ndarray,                    # 3x3 measurement noise (shared or per-mode)
        transition_matrix: np.ndarray,    # NxN Markov transition probs
        initial_mode_probs: np.ndarray,   # N-vector
    ): ...

    def predict(self, theta: float) -> np.ndarray:
        """IMM predict step. Returns combined predicted state (5,)."""
        # 1. Compute mixing probabilities
        # 2. Mix state estimates and covariances across modes
        # 3. Each mode predicts with its own F, Q (rotated by theta)
        # 4. Return combined prediction
        ...

    def correct(
        self, measurement: np.ndarray, max_velocity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """IMM update step. Returns (combined_state, combined_covariance).

        Also updates mode probabilities based on measurement likelihood.
        """
        # 1. Each mode computes innovation and likelihood
        # 2. Update mode probabilities: mu_j = (likelihood_j * c_j) / sum(...)
        # 3. Combine state and covariance
        ...

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Current mode probability vector (for diagnostics/UI)."""
        ...

    @property
    def active_mode(self) -> int:
        """Index of the most probable mode."""
        ...

    @property
    def state(self) -> np.ndarray:
        """Combined (blended) state estimate."""
        ...

    @property
    def covariance(self) -> np.ndarray:
        """Combined (blended) covariance."""
        ...

    def reset(self, initial_state: np.ndarray) -> None:
        """Reset all modes to the same initial state."""
        ...
```

### Default IMM Configuration (3 modes)

| Mode | F (velocity) | Q scaling | Damping | Description |
|------|-------------|-----------|---------|-------------|
| **Stationary** | vx,vy zeroed | 0.01x base | 0.0 | Animal is still |
| **Cruising** | standard CV | 1.0x base (from EM) | 0.95 | Normal locomotion |
| **Maneuvering** | standard CV | 5.0x base | 0.80 | Sharp turns, acceleration |

**Default transition matrix** (calibrated from trajectory segments or EM):
```
        To:  Still  Cruise  Maneuver
From:
  Still    [ 0.90   0.08    0.02  ]
  Cruise   [ 0.05   0.85    0.10  ]
  Maneuver [ 0.05   0.30    0.65  ]
```

These defaults encode: animals mostly stay in their current regime; maneuvering is transient (quickly returns to cruising); stationary is sticky.

### Refactoring `KalmanFilterManager`

The existing `KalmanFilterManager` gains a `mode` parameter:

```python
class KalmanFilterManager:
    def __init__(self, n_tracks: int, params: Dict[str, Any]):
        self._mode = params.get("KALMAN_FILTER_MODE", "standard")  # "standard" | "imm"

        if self._mode == "imm":
            self._imm_banks: List[IMMFilterBank] = []
            modes = self._build_imm_modes(params)
            trans_mat = self._build_transition_matrix(params)
            for _ in range(n_tracks):
                self._imm_banks.append(IMMFilterBank(modes, H, R, trans_mat, ...))
        else:
            # existing single-filter initialization (unchanged)
            ...
```

**Key design constraint:** The `predict()` and `correct()` public API stays identical. The caller (worker.py, optimizer.py) does not need to know whether IMM is active. This is critical for backward compatibility.

Additional public methods:
```python
def get_mode_probabilities(self) -> Optional[np.ndarray]:
    """Return (n_tracks, n_modes) array of IMM mode probs, or None if standard mode."""
    ...

def get_active_modes(self) -> Optional[np.ndarray]:
    """Return (n_tracks,) array of most-probable mode index, or None."""
    ...
```

### New config keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `kalman_filter_mode` | str | "standard" | "standard" or "imm" |
| `imm_n_modes` | int | 3 | Number of IMM modes (2 or 3) |
| `imm_stationary_q_scale` | float | 0.01 | Q multiplier for stationary mode |
| `imm_maneuvering_q_scale` | float | 5.0 | Q multiplier for maneuvering mode |
| `imm_stationary_damping` | float | 0.0 | Velocity damping for stationary mode |
| `imm_maneuvering_damping` | float | 0.80 | Velocity damping for maneuvering mode |
| `imm_transition_stay_prob` | float | 0.85 | Diagonal of transition matrix (self-transition) |

### Tests: `tests/test_imm_filter.py` — ~15 tests

- `test_imm_filter_bank_init` — correct number of modes, shapes
- `test_imm_predict_returns_state` — output shape (5,)
- `test_imm_correct_updates_probs` — mode probs change after measurement
- `test_stationary_object_favors_still_mode` — still measurements drive mode 0 probability up
- `test_cruising_object_favors_cruise_mode` — steady velocity drives mode 1
- `test_maneuvering_object_favors_maneuver_mode` — sharp turns drive mode 2
- `test_mode_transition_detection` — stop→move transition shifts mode probs
- `test_combined_state_is_weighted_blend` — verify mixing formula
- `test_reset_equalizes_mode_probs` — reset returns to initial probs
- `test_imm_matches_standard_for_single_mode` — 1-mode IMM = standard Kalman
- `test_kfm_imm_mode_predict_correct` — KalmanFilterManager with mode="imm" works
- `test_kfm_standard_unchanged` — mode="standard" behavior identical to current
- `test_kfm_get_mode_probabilities` — returns (n_tracks, n_modes) in IMM mode
- `test_kfm_get_mode_probabilities_none_standard` — returns None in standard mode
- `test_imm_covariance_positive_definite` — no numerical instability after 100 steps

---

## Phase 3 — EM + IMM Integration

**Goal:** Wire EM estimation into the IMM initialization so each mode gets data-driven noise parameters.

### Modified flow in `worker.py`

```
Initial Forward pass  (uses existing default/configured params)
        │
        ▼
EM Noise Estimation
  ├── Segment trajectories by motion regime
  ├── Fit per-regime Q, R
  └── Output: Dict[regime, NoiseEstimate]
        │
        ▼
Build IMM modes from EM estimates  (if kalman_filter_mode == "imm")
  ├── Stationary mode: Q from "stationary" estimate (or scaled default)
  ├── Cruising mode: Q from "cruising" estimate
  ├── Maneuvering mode: Q from "maneuvering" estimate
  └── Transition matrix from segment transition counts
        │
        ▼
Second Forward pass  (re-runs Kalman+assignment; detection cache → fast)
  └── Uses EM-calibrated params / IMM filter for both direction and backward
        │
        ▼
Backward pass  (with EM-calibrated params or IMM filter)
        │
        ▼
Post-processing  (unchanged)
```

**Note on `em_rerun_forward_pass`:** If the user explicitly sets this to `false`, the second forward pass is skipped and only the backward pass benefits from EM. This is the only case where Option B degrades to Option A. The default is `true`.

### New function in `em_noise_estimator.py`

```python
def build_imm_params_from_em(
    estimates: Dict[str, NoiseEstimate],
    segments: List[MotionSegment],
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert EM estimates + segment statistics into IMM config params.

    Returns a dict of param overrides that can be merged into the
    tracking params dict before initializing KalmanFilterManager.
    """
    # 1. Set per-mode Q scales from EM estimates
    # 2. Compute transition matrix from segment transitions
    # 3. Set per-mode damping from regime mean speeds
    # 4. Return param dict with IMM keys populated
    ...
```

### Transition matrix calibration

Count regime transitions in the segment list:
```
T[i,j] = count(segment_k.regime == i AND segment_{k+1}.regime == j) / count(segment_k.regime == i)
```
Smooth with Laplace prior (add 1 to all counts) to avoid zero probabilities.

### Tests: `tests/test_em_imm_integration.py` — ~8 tests

- `test_build_imm_params_from_em_keys` — output has all IMM config keys
- `test_transition_matrix_from_segments` — verify row-stochastic matrix
- `test_em_to_imm_end_to_end` — full pipeline: trajectories → EM → IMM params → tracking
- `test_imm_with_em_improves_over_default` — IMM+EM produces lower afterhours suspicion scores
- `test_em_fallback_when_insufficient_data` — graceful degradation to defaults
- `test_second_forward_pass_uses_em_calibrated_params` — verify second forward uses updated KFM
- `test_backward_pass_uses_em_calibrated_imm` — verify backward pass gets updated params
- `test_em_rerun_false_skips_second_forward` — when flag is false, only backward benefits

---

## Phase 4 — Optimizer Trimming

**Goal:** Remove the 4 Kalman noise parameters from Optuna search space when EM is enabled. Keep them available when EM is disabled (backward compatibility).

### Modified: `optimizer.py`

**Changes to `_PARAM_RANGES`:**

The module-level dict stays unchanged (it defines the full possible space). The trimming happens at trial suggestion time.

```python
# Parameters that EM estimation replaces:
_EM_HANDLED_PARAMS = {
    "KALMAN_NOISE_COVARIANCE",
    "KALMAN_MEASUREMENT_NOISE_COVARIANCE",
    "KALMAN_LONGITUDINAL_NOISE_MULTIPLIER",
    "KALMAN_DAMPING",
}

# In objective():
for key in _EM_HANDLED_PARAMS:
    if params.get("EM_NOISE_ESTIMATION_ENABLED", False):
        # Skip suggesting these — they're computed by EM
        tuning_config[key] = False
```

**Changes to `_run_tracking_loop`:**

When EM is enabled, the loop runs a mini-EM pass on the trial's frame range to compute noise params before the scoring loop. This ensures each Optuna trial still gets optimal noise for its non-noise parameter configuration.

### Modified: `parameter_helper.py`

**Kalman tab changes:**

When EM is enabled, the 4 noise checkboxes are:
- Grayed out (disabled)
- Show tooltip: "Handled by EM Noise Estimation"
- Unchecked automatically

```python
# In Kalman tab setup:
self.cb_em_enabled = QCheckBox("Use EM Noise Estimation (auto-calibrate)")
self.cb_em_enabled.setToolTip(
    "Estimate process and measurement noise from trajectory data.\n"
    "Replaces manual tuning of noise covariances."
)
self.cb_em_enabled.toggled.connect(self._on_em_toggled)

def _on_em_toggled(self, checked: bool):
    for cb in [self.cb_kalman_p, self.cb_kalman_m, self.cb_kalman_damp, self.cb_kalman_long_noise]:
        cb.setEnabled(not checked)
        if checked:
            cb.setChecked(False)
```

### Tests: update `tests/test_swap_scorer.py` (existing tests should still pass)

No new test file needed — this is config/UI wiring. Existing optimizer tests (if any) verify parameter flow.

---

## Phase 5 — UI Additions

### 5a. Main Window — Kalman/IMM Section

**Location:** Tracking parameters area in `main_window.py`, near existing Kalman spinboxes.

**New widgets:**

```
┌─ Motion Model ─────────────────────────────────────┐
│                                                     │
│  Filter mode:  [Standard ▼]  ← combo: standard/imm │
│                                                     │
│  ☑ EM Noise Estimation (auto-calibrate Q, R)        │
│    ☑ Re-run forward pass after calibration           │
│      (fast; uses detection cache — strongly advised) │
│    Min samples per regime: [50    ]                  │
│    ☑ Fit per motion regime                           │
│                                                     │
│  ── IMM Settings (visible when mode=imm) ──         │
│  Number of modes:     [3  ▼]  ← combo: 2 or 3      │
│  Stationary Q scale:  [0.01  ]                      │
│  Maneuvering Q scale: [5.0   ]                      │
│  Stationary damping:  [0.0   ]                      │
│  Maneuvering damping: [0.80  ]                      │
│  Self-transition prob: [0.85  ]                      │
│                                                     │
│  ── Standard Kalman (grayed when EM enabled) ──     │
│  Process noise:       [0.0300]  ← existing          │
│  Measurement noise:   [0.1000]  ← existing          │
│  Velocity damping:    [0.950 ]  ← existing          │
│  Longitudinal noise:  [5.0   ]  ← existing          │
│  Lateral noise:       [0.1   ]  ← existing          │
│                                                     │
│  ── Always active ──                                │
│  Maturity age:        [5     ]  ← existing          │
│  Initial vel retention:[0.20 ]  ← existing          │
│  Max velocity:        [2.0   ]  ← existing          │
└─────────────────────────────────────────────────────┘
```

**Visibility logic:**
- IMM settings: visible only when `combo_filter_mode == "imm"`
- Standard Kalman noise spinboxes: enabled only when EM checkbox is unchecked
- Maturity age, initial velocity retention, max velocity: always visible (not replaced by EM)

### 5b. Post-Tracking EM Results Display

After the forward pass completes (when EM is enabled), show a brief summary in the status area:

```
EM Noise Estimation complete:
  Cruising:    Q_long=0.042  Q_lat=0.008  R=0.091  (1847 samples)
  Stationary:  Q_long=0.003  Q_lat=0.002  R=0.085  (423 samples)
  Maneuvering: Q_long=0.187  Q_lat=0.031  R=0.098  (312 samples)
  Transition matrix calibrated from 89 regime transitions.
```

This is logged via the existing `status_signal` mechanism in `TrackingWorker`.

### 5c. IMM Mode Visualization (optional diagnostic)

In the tracking video overlay, when IMM is active, color-code each track's bounding box by active mode:
- Blue = stationary
- Green = cruising
- Orange = maneuvering

This reuses the existing track color rendering path with a mode-dependent color override. Controlled by a new checkbox:

```python
self.chk_show_imm_modes = QCheckBox("Color tracks by motion mode")
self.chk_show_imm_modes.setVisible(False)  # shown only when IMM enabled
```

### 5d. Config Save/Load

**New keys in `save_config()` / `_load_config_from_file()`:**

```python
# In save_config:
config["kalman_filter_mode"] = self.combo_filter_mode.currentText().lower()
config["em_noise_estimation_enabled"] = self.chk_em_enabled.isChecked()
config["em_rerun_forward_pass"] = self.chk_em_rerun_forward.isChecked()
config["em_min_samples"] = self.spin_em_min_samples.value()
config["em_per_regime"] = self.chk_em_per_regime.isChecked()
config["imm_n_modes"] = self.combo_imm_n_modes.currentIndex() + 2  # 2 or 3
config["imm_stationary_q_scale"] = self.spin_imm_stationary_q.value()
config["imm_maneuvering_q_scale"] = self.spin_imm_maneuvering_q.value()
config["imm_stationary_damping"] = self.spin_imm_stationary_damping.value()
config["imm_maneuvering_damping"] = self.spin_imm_maneuvering_damping.value()
config["imm_transition_stay_prob"] = self.spin_imm_stay_prob.value()
config["imm_show_mode_colors"] = self.chk_show_imm_modes.isChecked()
```

**In `_load_config_from_file()`:**

```python
# Motion model section
if "kalman_filter_mode" in config:
    idx = self.combo_filter_mode.findText(config["kalman_filter_mode"], Qt.MatchFixedString)
    if idx >= 0:
        self.combo_filter_mode.setCurrentIndex(idx)
if "em_noise_estimation_enabled" in config:
    self.chk_em_enabled.setChecked(config["em_noise_estimation_enabled"])
if "em_rerun_forward_pass" in config:
    self.chk_em_rerun_forward.setChecked(config["em_rerun_forward_pass"])
# ... etc for all new keys
```

### 5e. `default.json` additions

```json
{
  "kalman_filter_mode": "standard",
  "em_noise_estimation_enabled": false,
  "em_rerun_forward_pass": true,
  "em_min_samples": 50,
  "em_per_regime": true,
  "imm_n_modes": 3,
  "imm_stationary_q_scale": 0.01,
  "imm_maneuvering_q_scale": 5.0,
  "imm_stationary_damping": 0.0,
  "imm_maneuvering_damping": 0.80,
  "imm_transition_stay_prob": 0.85,
  "imm_show_mode_colors": false
}
```

---

## Phase 6 — Worker Integration

### Modified: `worker.py`

**Full three-pass pipeline:**

```python
# ── Step 1: Initial forward pass (existing code, unchanged) ────────────────
forward_df = self._run_forward_pass()   # existing forward tracking loop

# ── Step 2: EM Noise Estimation (optional) ─────────────────────────────────
_em_enabled = bool(p.get("EM_NOISE_ESTIMATION_ENABLED", False))
_em_rerun   = bool(p.get("EM_RERUN_FORWARD_PASS", True))
_imm_mode   = p.get("KALMAN_FILTER_MODE", "standard") == "imm"

if _em_enabled:
    self.status_signal.emit("Running EM noise estimation...")
    from multi_tracker.core.filters.em_noise_estimator import EMNoiseEstimator, build_imm_params_from_em
    from multi_tracker.core.filters.motion_segments import segment_all_trajectories

    segments = segment_all_trajectories(forward_df, body_size, fps)
    estimator = EMNoiseEstimator(
        body_size, fps,
        min_samples=int(p.get("EM_MIN_SAMPLES", 50)),
        per_regime=bool(p.get("EM_PER_REGIME", True)),
    )
    estimates = estimator.fit_from_trajectories(forward_df, p)

    # Log estimated noise to status area
    for regime, est in estimates.items():
        self.status_signal.emit(
            f"  EM [{regime}]: Q_long={est.Q_velocity[0,0]:.4f}  "
            f"Q_lat={est.Q_velocity[1,1]:.4f}  "
            f"R={float(np.mean(np.diag(est.R))):.4f}  "
            f"({est.n_samples} samples)"
        )

    # Apply EM results to params dict
    if _imm_mode:
        imm_overrides = build_imm_params_from_em(estimates, segments, p)
        p.update(imm_overrides)
    else:
        ref = estimates.get("cruising") or estimates.get("all")
        if ref is not None:
            p["KALMAN_NOISE_COVARIANCE"] = float(ref.Q_theta)
            p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"] = float(np.mean(np.diag(ref.R)))
            q_long = ref.Q_velocity[0, 0]
            q_lat  = ref.Q_velocity[1, 1]
            if q_lat > 0:
                p["KALMAN_ANISOTROPY_RATIO"] = q_long / q_lat

    # Re-init KalmanFilterManager with calibrated params
    self.kf_manager = KalmanFilterManager(p["MAX_TARGETS"], p)

    # ── Step 3: Second forward pass (fast — detection cache, no GPU) ────────
    if _em_rerun:
        self.status_signal.emit(
            "Re-running forward pass with EM-calibrated parameters..."
        )
        forward_df = self._run_forward_pass(use_cache=True)

# ── Step 4: Backward pass (existing code) ──────────────────────────────────
# Uses the EM-calibrated p and self.kf_manager automatically.
backward_df = self._run_backward_pass(use_cache=True)

# ── Step 5: Post-processing (unchanged) ────────────────────────────────────
```

**Progress reporting:** The status bar shows three pass labels when EM is active:
- "Forward pass (1/3)..." → "EM noise estimation..." → "Forward pass (2/3)..." → "Backward pass (3/3)..."

compared to the standard two-label flow:
- "Forward pass (1/2)..." → "Backward pass (2/2)..."

The progress bar denominator adjusts accordingly so the user always sees a sensible percentage.

### Parameter flow: `get_current_tracking_parameters()`

The main window's `get_current_tracking_parameters()` method reads all spinbox values and builds the params dict. New keys are added:

```python
params["KALMAN_FILTER_MODE"] = self.combo_filter_mode.currentText().lower()
params["EM_NOISE_ESTIMATION_ENABLED"] = self.chk_em_enabled.isChecked()
params["EM_RERUN_FORWARD_PASS"] = self.chk_em_rerun_forward.isChecked()
params["EM_MIN_SAMPLES"] = self.spin_em_min_samples.value()
params["EM_PER_REGIME"] = self.chk_em_per_regime.isChecked()
params["IMM_N_MODES"] = self.combo_imm_n_modes.currentIndex() + 2
params["IMM_STATIONARY_Q_SCALE"] = self.spin_imm_stationary_q.value()
params["IMM_MANEUVERING_Q_SCALE"] = self.spin_imm_maneuvering_q.value()
params["IMM_STATIONARY_DAMPING"] = self.spin_imm_stationary_damping.value()
params["IMM_MANEUVERING_DAMPING"] = self.spin_imm_maneuvering_damping.value()
params["IMM_TRANSITION_STAY_PROB"] = self.spin_imm_stay_prob.value()
```

---

## Phase 7 — CSV Output Columns (Optional)

When IMM is active, add per-row mode information to the output CSV:

| Column | Type | Description |
|--------|------|-------------|
| `IMMActiveMode` | str | Most probable mode name ("stationary", "cruising", "maneuvering") |
| `IMMModeProbStationary` | float | Mode 0 probability |
| `IMMModeProbCruising` | float | Mode 1 probability |
| `IMMModeProbManeuvering` | float | Mode 2 probability (if 3-mode) |

These columns are only present when `kalman_filter_mode == "imm"`.

**Integration:** Add to `csv_writer.py` header and row emission, gated on the IMM flag.

---

## Phase 8 — Forward-Pass Terminal State Sidecar

**Goal:** Cache the Kalman state vector and covariance of each track at its last active
forward-pass frame, saving the result as a lightweight `.npz` sidecar. This allows
the afterhours merge wizard (Phase 1 of the two-phase proofreading pipeline) to use
proper Kalman-predicted positions instead of re-deriving a simplified constant-velocity
extrapolation from CSV columns alone.

### Why "forward-pass only"

The merge wizard asks: *"where would this dying track go next?"*
That is answered by the **forward-pass** terminal state — the filter's best estimate of
position and velocity at the moment the track vanished.

The backward-pass terminal state answers the opposite question ("where did this track
come from?") and is not useful here. Caching it would waste storage and invite incorrect
usage.

### What is saved

For each track that ends **before the last video frame** (i.e., every truly dying track,
not just truncated-at-end tracks), record at the terminal active frame:

```python
@dataclass
class TrackTerminalState:
    track_id: int           # final TrajectoryID (from CSV), NOT Kalman slot index
    frame: int              # last active frame (forward pass)
    X: np.ndarray           # shape (5,) — [x, y, θ, vx, vy]
    P: np.ndarray           # shape (5, 5) — posterior covariance
    imm_mode_probs: Optional[np.ndarray]  # shape (n_modes,) if IMM active, else None
    imm_active_mode: Optional[int]        # argmax of mode_probs, or None
```

Sidecar filename: `{video_stem}_kalman_terminal_states.npz`
Stored as: one compressed array with `track_ids`, `frames`, `X_states` (N×5),
`P_states` (N×5×5), and optionally `imm_mode_probs` (N×n_modes).

### Why P matters more than X

`X` gives the expected position, but `P` gives the *uncertainty ellipse*. A fast-moving
animal (high covariance in the velocity components) projects a wide ellipse in the
forward direction — a candidate 20px away along the heading but 0px sideways is very
likely the same animal, while a candidate 20px perpendicular is far less likely.

Replace the hard pixel threshold in merge scoring with a **Mahalanobis distance**:

```
d_maha = sqrt((p_target - p_predicted)^T @ Σ_pred^{-1} @ (p_target - p_predicted))
```

where `Σ_pred = P_pred[0:2, 0:2]` (the 2D positional submatrix of the predicted
covariance after N steps). A threshold of `d_maha < 3.0` corresponds to a 3-sigma
gate irrespective of track speed or uncertainty.

### Proper N-step Kalman extrapolation

With the cached terminal state, the N-step prediction is exact (within the linear
Gaussian model):

```python
def kalman_extrapolate(X: np.ndarray, P: np.ndarray, F: np.ndarray,
                       Q_built: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Exact N-step forward extrapolation from terminal Kalman state."""
    for _ in range(n_steps):
        X = F @ X
        P = F @ P @ F.T + Q_built
    return X, P  # predicted state and covariance after n_steps frames
```

`Q_built` is the per-frame process noise (already computed in `KalmanFilterManager.__init__`
— reuse its `Q_base`, `q_long`, `q_lat` and the heading at death for the anisotropic term).

When IMM is active, use the mode probabilities to select the right `F` and `Q`:
- If `imm_active_mode == STATIONARY`: use stationary F (velocity zeroed)
- If `imm_active_mode == MANEUVERING`: use maneuvering Q (5× scale)
- Otherwise: use standard cruising params

### Worker integration

**Location:** End of the forward pass loop in `worker.py`, at the frame where a track
slot is de-activated (track dies). This happens when the assignment step leaves a slot
unmatched for more than the allowed number of frames — the exact point where the final
corrected state `X[slot]` / `P[slot]` reflects the last observation-informed estimate.

```python
# In TrackingWorker, at track-death bookkeeping:
if slot_just_died and self._save_terminal_states:
    trajectory_id = self._slot_to_trajectory_id[slot]
    self._terminal_states[trajectory_id] = TrackTerminalState(
        track_id=trajectory_id,
        frame=current_frame,
        X=self.kf_manager.X[slot].copy(),
        P=self.kf_manager.P[slot].copy(),
        imm_mode_probs=self.kf_manager.get_mode_probabilities_for(slot),
        imm_active_mode=self.kf_manager.get_active_mode_for(slot),
    )

# After forward pass loop, before backward pass:
if self._save_terminal_states and self._terminal_states:
    _save_terminal_states_npz(self._terminal_states, sidecar_path)
    self.status_signal.emit(
        f"Saved Kalman terminal states for {len(self._terminal_states)} tracks "
        f"→ {sidecar_path.name}"
    )
```

### Afterhours integration

In `merge_candidates.py`, replace the `predict_position()` simple CV function:

```python
def load_terminal_states(sidecar_path: Path) -> Optional[Dict[int, TrackTerminalState]]:
    """Load terminal states sidecar if available, return None if not found."""
    ...

def kalman_predicted_position(
    state: TrackTerminalState,
    n_steps: int,
    kalman_params: Dict[str, Any],
) -> Tuple[Tuple[float, float], np.ndarray]:
    """Return (predicted_xy, P_pred_2x2) using proper N-step Kalman extrapolation."""
    ...

def mahalanobis_dist_2d(point: Tuple[float, float],
                        mean: Tuple[float, float],
                        cov2x2: np.ndarray) -> float:
    """2D Mahalanobis distance; falls back to Euclidean if cov is degenerate."""
    ...
```

**Graceful degradation:** If the sidecar is absent (old runs, runs without the new
worker code, or backward-pass-only rescoring), `merge_candidates.py` falls back to
the simple constant-velocity `predict_position()` with a hard pixel threshold — the
Mahalanobis upgrade is additive, not required.

### File naming and discovery

Sidecar path derivation mirrors the detection cache pattern:

```python
def build_kalman_terminal_states_path(csv_path: Path) -> Path:
    # strips _proofread suffix if present, so it always references the original run
    stem = csv_path.stem.replace("_proofread", "")
    return csv_path.parent / f"{stem}_kalman_terminal_states.npz"
```

Afterhours discovers it via the same CSV-adjacent scheme used for `_density_regions.json`.

### Storage cost

`N_dying_tracks × (5 + 25) float32 = N × 120 bytes`.
For 1,000 dying tracks: 120 KB. Negligible.

### Tests: `tests/test_kalman_terminal_states.py` — ~6 tests

- `test_save_load_roundtrip` — serialize and reload, verify arrays match
- `test_kalman_extrapolate_zero_steps` — returns input state unchanged
- `test_kalman_extrapolate_constant_velocity` — linear motion, verify analytical result
- `test_mahalanobis_vs_euclidean` — isotropic P → Mahalanobis = Euclidean / sigma
- `test_mahalanobis_anisotropy` — elongated P penalizes off-axis candidates correctly
- `test_graceful_fallback_missing_sidecar` — `load_terminal_states()` returns None, scorer uses CV fallback

---

## Implementation Order

| Phase | Module | Files | Est. Tests |
|-------|--------|-------|-----------|
| **0** | Motion segments | `filters/motion_segments.py` | 8 |
| **1** | EM estimator | `filters/em_noise_estimator.py` | 12 |
| **2** | IMM filter | `filters/imm_filter.py`, `filters/kalman.py` | 15 |
| **3** | EM+IMM integration | `em_noise_estimator.py`, `worker.py` | 8 |
| **4** | Optimizer trimming | `optimizer.py`, `parameter_helper.py` | 0 (existing tests) |
| **5** | UI + config | `main_window.py`, `default.json` | 0 (manual QA) |
| **6** | Worker wiring (3-pass pipeline) | `worker.py` | 0 (covered by Phase 3) |
| **7** | CSV output | `csv_writer.py` | 2 |
| **8** | Terminal state sidecar | `worker.py`, `afterhours/core/merge_candidates.py` | 6 |
| **Total** | | | **~51 new tests** |

**Dependencies:** Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4/5/6/7/8 (parallel)
Phase 8 can be built independently of Phases 1–3 — it has no EM/IMM dependency.
The afterhours consumer of Phase 8 is the merge wizard from the fragment merge proofreading plan.

---

## Backward Compatibility

- `kalman_filter_mode: "standard"` + `em_noise_estimation_enabled: false` = **identical behavior to current code — single forward pass, single backward pass**
- All new config keys have defaults that preserve current behavior
- Existing saved configs load without error (missing keys use defaults)
- Optuna optimizer works unchanged when EM is disabled
- The `predict()` / `correct()` API of `KalmanFilterManager` is unchanged
- The second forward pass only runs when `em_noise_estimation_enabled: true` AND `em_rerun_forward_pass: true` (both default to false/true respectively, so enabling EM automatically opts in to the second pass)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| EM overfits to noisy initial tracks | Regularize: blend EM estimate with prior (default values) using `alpha * EM + (1-alpha) * default`, alpha = min(1, n_samples/200) |
| IMM numerical instability | Joseph-form updates in all modes; covariance floor per mode; mode probability floor at 0.001 |
| Insufficient data for per-regime EM | Fall back to single "all" estimate if any regime has < min_samples |
| Performance regression in optimizer | EM mini-pass in optimizer adds ~0.1s per trial (negligible vs tracking simulation) |
| Second forward pass confusion | Progress bar labels update to "Forward pass (1/3)", "(2/3)", "Backward (3/3)" when EM active — user always sees context |
| Second forward pass wasted if EM has insufficient data | EM falls back to prior and skips second forward, emitting a status warning |
| Config migration | All new keys are optional with sensible defaults |
