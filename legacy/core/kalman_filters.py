"""
Kalman filter utilities for multi-object tracking.
Functionally identical to the original implementation's KF logic.
"""

import cv2
import numpy as np


class KalmanFilterManager:
    """Manages Kalman filters for multi-object tracking."""

    def __init__(self, num_targets, params):
        self.num_targets = num_targets
        self.params = params
        self.filters = self._init_kalman_filters()

    def _init_kalman_filters(self):
        """Creates and configures Kalman filters for all targets."""
        p = self.params
        kfs = []
        for _ in range(self.num_targets):
            kf = cv2.KalmanFilter(5, 3)
            kf.measurementMatrix = np.array(
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], np.float32
            )
            kf.transitionMatrix = np.array(
                [
                    [1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ],
                np.float32,
            )
            kf.processNoiseCov = (
                np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
            )
            kf.measurementNoiseCov = (
                np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            )
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            kfs.append(kf)
        return kfs

    def initialize_filter(self, track_idx, initial_state):
        """Initializes a specific Kalman filter."""
        if 0 <= track_idx < len(self.filters):
            kf = self.filters[track_idx]
            kf.statePre = initial_state.copy()
            kf.statePost = initial_state.copy()

    def get_predictions(self):
        """Get predictions from all Kalman filters."""
        preds = [kf.predict()[:3].flatten() for kf in self.filters]
        return np.array(preds, np.float32)

    def get_position_uncertainties(self):
        """Get position uncertainty (covariance trace) for all filters.

        Returns:
            List of uncertainty values (one per track)
            Higher values = more uncertain position estimate
        """
        uncertainties = []
        for kf in self.filters:
            # Extract position covariance (x, y) from error covariance matrix
            pos_cov = kf.errorCovPost[:2, :2]
            # Use trace (sum of variances) as uncertainty measure
            uncertainty = np.trace(pos_cov)
            uncertainties.append(float(uncertainty))
        return uncertainties
