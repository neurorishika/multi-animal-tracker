import numpy as np

# --- Numba Kernels ---
try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


@njit(cache=True, fastmath=True)
def _predict_kernel(X, P, F, Q):
    for i in range(len(X)):
        # Predict State: X = F @ X
        X[i] = F @ X[i]
        # Predict Covariance: P = FPF^T + Q
        P[i] = F @ P[i] @ F.T + Q
    return X, P


@njit(cache=True, fastmath=True)
def _correct_kernel(X, P, H, R, I, track_idx, measurement):
    x = X[track_idx].reshape(5, 1)
    p = P[track_idx]
    z = measurement.reshape(3, 1)

    # 1. Innovation (Error)
    y = z - (H @ x)

    # --- CRITICAL FIX: Circular Angle Wrap ---
    # This is the ONLY SOTA fix needed for theta stability.
    # It prevents 'spinning' when crossing 0 <-> 2pi
    if y[2, 0] > np.pi:
        y[2, 0] -= 2 * np.pi
    elif y[2, 0] < -np.pi:
        y[2, 0] += 2 * np.pi

    # 2. Innovation Covariance
    S = (H @ p @ H.T) + R

    # 3. Kalman Gain
    K = p @ H.T @ np.linalg.inv(S)

    # 4. Update State
    X[track_idx] = (x + (K @ y)).flatten()

    # 5. Joseph Form Covariance Update (The most stable version)
    IKH = I - (K @ H)
    P[track_idx] = (IKH @ p @ IKH.T) + (K @ R @ K.T)

    return X, P


class KalmanFilterManager:
    def __init__(self, num_targets, params):
        self.num_targets = num_targets
        self.params = params
        self.dim_s = 5
        self.dim_m = 3

        self.X = np.zeros((num_targets, self.dim_s), dtype=np.float32)

        # --- MATCHING OPENCV DEFAULTS ---
        # OpenCV's errorCovPost defaults to Identity.
        # Don't use 1000.0; it destroys the Mahalanobis gating.
        self.P = np.stack(
            [np.eye(self.dim_s, dtype=np.float32) for _ in range(num_targets)]
        )

        # Noise parameters from your UI
        q_val = params.get("KALMAN_NOISE_COVARIANCE", 0.03)
        r_val = params.get("KALMAN_MEASUREMENT_NOISE_COVARIANCE", 0.1)

        self.Q = np.eye(self.dim_s, dtype=np.float32) * q_val
        self.R = np.eye(self.dim_m, dtype=np.float32) * r_val

        self.F = np.array(
            [
                [1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.H = np.array(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.float32
        )

        self.I = np.eye(self.dim_s, dtype=np.float32)

    def initialize_filter(self, track_idx, initial_state):
        """Warm start: reset state and uncertainty."""
        self.X[track_idx] = initial_state.flatten()
        # Reset P to Identity (OpenCV default) to restore discriminative power
        self.P[track_idx] = np.eye(self.dim_s, dtype=np.float32)

    def predict(self):
        if NUMBA_AVAILABLE:
            self.X, self.P = _predict_kernel(self.X, self.P, self.F, self.Q)
        else:
            # Batch NumPy fallback
            self.X = (self.F @ self.X.T).T
            # Batched P update: P = FPF^T + Q
            self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X[:, :3].copy()

    def get_predictions(self):
        return self.predict()

    def correct(self, track_idx, measurement):
        if NUMBA_AVAILABLE:
            self.X, self.P = _correct_kernel(
                self.X, self.P, self.H, self.R, self.I, track_idx, measurement
            )
        else:
            # Standard NumPy fallback with Theta-Wrap
            z = measurement.reshape(3, 1)
            x = self.X[track_idx].reshape(5, 1)
            p = self.P[track_idx]
            y = z - (self.H @ x)
            # Wrap theta
            if y[2, 0] > np.pi:
                y[2, 0] -= 2 * np.pi
            elif y[2, 0] < -np.pi:
                y[2, 0] += 2 * np.pi

            S = self.H @ p @ self.H.T + self.R
            K = p @ self.H.T @ np.linalg.inv(S)
            self.X[track_idx] = (x + (K @ y)).flatten()
            IKH = self.I - (K @ self.H)
            self.P[track_idx] = IKH @ p @ IKH.T + (K @ self.R @ K.T)

    def get_mahalanobis_matrices(self):
        """Pre-calculates S_inv for the Assigner."""
        # S = HPH^T + R
        S = self.H @ self.P @ self.H.T + self.R
        return np.linalg.inv(S)

    def get_position_uncertainties(self):
        return np.trace(self.P[:, :2, :2], axis1=1, axis2=2).tolist()
