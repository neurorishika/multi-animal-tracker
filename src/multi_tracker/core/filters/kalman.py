"""
SOTA Biologically-Constrained Vectorized Kalman Filter.
Features:
1. Anisotropic Process Noise (Longitudinal vs. Lateral uncertainty)
2. Velocity Damping (Friction) for stop-and-go behavior
3. Joseph-Form Numerical Stability
4. Circular Angle Wrap-around
"""

import numpy as np
from multi_tracker.utils.gpu_utils import NUMBA_AVAILABLE, njit


# --- Numba Kernels (Optimized for Large N) ---
@njit(cache=True, fastmath=True)
def _predict_kernel(X, P, F, Q_base, q_long, q_lat):
    """
    Predicts next state and rotates process noise to align with animal heading.
    """
    for i in range(len(X)):
        # 1. Rotate Process Noise based on current orientation
        theta = X[i, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Calculate rotated velocity noise components
        # (Rotates a diagonal [q_long, q_lat] matrix by theta)
        r11 = cos_t**2 * q_long + sin_t**2 * q_lat
        r12 = cos_t * sin_t * (q_long - q_lat)
        r22 = sin_t**2 * q_long + cos_t**2 * q_lat

        # Apply specific noise to this animal
        Qi = Q_base.copy()
        Qi[3, 3] = r11
        Qi[3, 4] = r12
        Qi[4, 3] = r12
        Qi[4, 4] = r22

        # 2. State Prediction: X = F @ X
        X[i] = F @ X[i]

        # 3. Covariance Prediction: P = FPF^T + Qi
        P[i] = F @ P[i] @ F.T + Qi

        # 4. SOTA Stability: Variance Floor
        # Prevents the filter from 'collapsing' during long pauses
        for j in range(5):
            if P[i, j, j] < 0.1:
                P[i, j, j] = 0.1

    return X, P


@njit(cache=True, fastmath=True)
def _correct_kernel(X, P, H, R, I, track_idx, measurement, max_velocity):
    """
    Corrects state using Joseph Form for stability and circular angle logic.
    """
    x = X[track_idx].reshape(5, 1)
    p = P[track_idx]
    z = measurement.reshape(3, 1)

    # Innovation
    y = z - (H @ x)

    # --- Circular Angle Wrap ---
    # Ensures a turn from 359 deg to 1 deg is seen as 2 deg, not -358 deg
    if y[2, 0] > np.pi:
        y[2, 0] -= 2 * np.pi
    elif y[2, 0] < -np.pi:
        y[2, 0] += 2 * np.pi

    # Innovation Covariance
    S = (H @ p @ H.T) + R

    # Kalman Gain
    K = p @ H.T @ np.linalg.inv(S)

    # Update State
    X[track_idx] = (x + (K @ y)).flatten()

    # Apply velocity constraint after correction
    vx, vy = X[track_idx, 3], X[track_idx, 4]
    speed = np.sqrt(vx**2 + vy**2)
    if speed > max_velocity:
        scale = max_velocity / speed
        X[track_idx, 3] *= scale
        X[track_idx, 4] *= scale

    # Joseph Form Covariance Update: P = (I-KH)P(I-KH)^T + KRK^T
    # Guarantees P remains positive-definite
    IKH = I - (K @ H)
    P[track_idx] = (IKH @ p @ IKH.T) + (K @ R @ K.T)

    return X, P


@njit(cache=True, fastmath=True)
def _get_mahal_kernel(P, H, R):
    """Batch calculation of Inverse Innovation Covariance for Assigner."""
    num_targets = len(P)
    S_inv = np.zeros((num_targets, 3, 3), dtype=np.float32)
    for i in range(num_targets):
        S = H @ P[i] @ H.T + R
        S_inv[i] = np.linalg.inv(S)
    return S_inv


class KalmanFilterManager:
    """
    Manages a batch of biologically-constrained Kalman Filters.
    """

    def __init__(self, num_targets, params):
        self.num_targets = num_targets
        self.params = params
        self.dim_s = 5  # [x, y, theta, vx, vy]
        self.dim_m = 3  # [x, y, theta]

        # Track ages (number of updates since initialization)
        self.track_ages = np.zeros(num_targets, dtype=np.int32)
        self.age_threshold = params.get(
            "KALMAN_MATURITY_AGE", 5
        )  # Frames to reach full dynamics
        self.initial_velocity_retention = params.get(
            "KALMAN_INITIAL_VELOCITY_RETENTION", 0.2
        )

        # Maximum velocity constraint (pixels/frame, as multiplier of body size)
        reference_body_size = params.get("REFERENCE_BODY_SIZE", 20.0)
        max_velocity_multiplier = params.get("KALMAN_MAX_VELOCITY_MULTIPLIER", 2.0)
        self.max_velocity = max_velocity_multiplier * reference_body_size

        # 1. Initialize State (N, 5)
        self.X = np.zeros((self.num_targets, self.dim_s), dtype=np.float32)

        # 2. Initialize Covariance (N, 5, 5)
        # Moderate uncertainty allows the filter to adapt quickly to initial motion
        self.init_P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0]).astype(np.float32)
        self.P = np.stack([self.init_P.copy() for _ in range(num_targets)])

        # 3. Process Noise Parameters
        q_sigma = params.get("KALMAN_NOISE_COVARIANCE", 0.03)
        # SOTA: High noise forward, low noise sideways (anisotropic)
        q_long_multiplier = params.get("KALMAN_LONGITUDINAL_NOISE_MULTIPLIER", 5.0)
        q_lat_multiplier = params.get("KALMAN_LATERAL_NOISE_MULTIPLIER", 0.1)
        self.q_long = q_sigma * q_long_multiplier
        self.q_lat = q_sigma * q_lat_multiplier

        # Base jitter for position and theta
        self.Q_base = np.diag([q_sigma, q_sigma, q_sigma, 0.0, 0.0]).astype(np.float32)

        # 4. Measurement Noise
        r_val = params.get("KALMAN_MEASUREMENT_NOISE_COVARIANCE", 0.1)
        self.R = np.eye(self.dim_m, dtype=np.float32) * r_val

        # 5. Transition Matrix F with Friction (Damping)
        # Prevents overshoot when an animal stops suddenly
        damp = params.get("KALMAN_DAMPING", 0.95)
        self.F = np.array(
            [
                [1, 0, 0, damp, 0],
                [0, 1, 0, 0, damp],
                [0, 0, 1, 0, 0],
                [0, 0, 0, damp, 0],
                [0, 0, 0, 0, damp],
            ],
            dtype=np.float32,
        )

        # 6. Measurement Matrix H
        self.H = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32
        )

        self.I = np.eye(self.dim_s, dtype=np.float32)

    def initialize_filter(self, track_idx, initial_state):
        """Reset a specific track's state and uncertainty."""
        self.X[track_idx] = initial_state.flatten()
        self.P[track_idx] = self.init_P.copy()
        self.track_ages[track_idx] = 0  # Reset age counter

    def predict(self):
        """Batch predict all N targets with age-dependent velocity damping."""
        # Standard batch prediction
        if NUMBA_AVAILABLE:
            self.X, self.P = _predict_kernel(
                self.X, self.P, self.F, self.Q_base, self.q_long, self.q_lat
            )
        else:
            # Basic NumPy Fallback (Standard isotropic prediction)
            for i in range(self.num_targets):
                theta = self.X[i, 2]
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                r11 = cos_t**2 * self.q_long + sin_t**2 * self.q_lat
                r12 = cos_t * sin_t * (self.q_long - self.q_lat)
                r22 = sin_t**2 * self.q_long + cos_t**2 * self.q_lat

                Qi = self.Q_base.copy()
                Qi[3, 3] = r11
                Qi[3, 4] = r12
                Qi[4, 3] = r12
                Qi[4, 4] = r22

                self.X[i] = self.F @ self.X[i]
                self.P[i] = self.F @ self.P[i] @ self.F.T + Qi

                # Variance floor
                for j in range(5):
                    if self.P[i, j, j] < 0.1:
                        self.P[i, j, j] = 0.1

        # Apply age-dependent velocity damping AFTER prediction
        # Young tracks have their velocity heavily damped toward zero
        for i in range(self.num_targets):
            age = self.track_ages[i]
            if age < self.age_threshold:
                # Calculate age-dependent damping factor
                # age=0: use initial_velocity_retention (default 0.2 = keep 20% of velocity)
                # age=threshold: use 1.0 (keep 100% of velocity)
                age_ratio = age / self.age_threshold
                velocity_retention = (
                    self.initial_velocity_retention
                    + (1.0 - self.initial_velocity_retention) * age_ratio
                )

                # Damp velocity estimates
                self.X[i, 3] *= velocity_retention  # vx
                self.X[i, 4] *= velocity_retention  # vy

            # Apply maximum velocity constraint
            # This prevents unrealistic predictions during occlusions or poor measurements
            vx, vy = self.X[i, 3], self.X[i, 4]
            speed = np.sqrt(vx**2 + vy**2)
            if speed > self.max_velocity:
                # Scale down velocity while preserving direction
                scale = self.max_velocity / speed
                self.X[i, 3] *= scale
                self.X[i, 4] *= scale

        return self.X[:, :3].copy()

    def get_predictions(self):
        return self.predict()

    def correct(self, track_idx, measurement):
        """Update a track with new detection data."""
        if NUMBA_AVAILABLE:
            self.X, self.P = _correct_kernel(
                self.X,
                self.P,
                self.H,
                self.R,
                self.I,
                track_idx,
                measurement,
                self.max_velocity,
            )
        else:
            # Manual fallback with Theta-Wrap logic
            z = measurement.reshape(3, 1)
            x = self.X[track_idx].reshape(5, 1)
            p = self.P[track_idx]
            y = z - (self.H @ x)
            if y[2, 0] > np.pi:
                y[2, 0] -= 2 * np.pi
            elif y[2, 0] < -np.pi:
                y[2, 0] += 2 * np.pi
            S = self.H @ p @ self.H.T + self.R
            K = p @ self.H.T @ np.linalg.inv(S)
            self.X[track_idx] = (x + (K @ y)).flatten()
            IKH = self.I - (K @ self.H)
            self.P[track_idx] = IKH @ p @ IKH.T + (K @ self.R @ K.T)

            # Apply velocity constraint after correction
            # This prevents the filter from learning unrealistic velocities from noisy measurements
            vx, vy = self.X[track_idx, 3], self.X[track_idx, 4]
            speed = np.sqrt(vx**2 + vy**2)
            if speed > self.max_velocity:
                scale = self.max_velocity / speed
                self.X[track_idx, 3] *= scale
                self.X[track_idx, 4] *= scale

        # Increment track age after successful update
        self.track_ages[track_idx] += 1

    def get_mahalanobis_matrices(self):
        """Calculates batch S_inv for optimized Mahalanobis assignment."""
        if NUMBA_AVAILABLE:
            return _get_mahal_kernel(self.P, self.H, self.R)
        else:
            S = self.H @ self.P @ self.H.T + self.R
            return np.linalg.inv(S)

    def get_position_uncertainties(self):
        """Returns position variance trace for track quality monitoring."""
        return np.trace(self.P[:, :2, :2], axis1=1, axis2=2).tolist()
