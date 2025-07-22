"""
Kalman filter utilities for multi-object tracking.
Functionally identical to the original implementation's KF logic.
"""
import numpy as np
import cv2

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
            kf.measurementMatrix = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]], np.float32)
            kf.transitionMatrix = np.array([[1,0,0,1,0], [0,1,0,0,1], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], np.float32)
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * p["KALMAN_NOISE_COVARIANCE"]
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * p["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
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