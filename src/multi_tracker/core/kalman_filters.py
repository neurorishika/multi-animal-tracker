"""
Kalman filter utilities for multi-object tracking.

This module provides Kalman filter initialization and management for tracking
object positions and orientations over time.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class KalmanFilterManager:
    """
    Manages Kalman filters for multi-object tracking.
    
    Each target gets its own Kalman filter for state estimation and prediction.
    The state vector contains [x, y, theta, vx, vy] where:
    - x, y: position coordinates
    - theta: orientation angle  
    - vx, vy: velocity components
    """
    
    def __init__(self, num_targets, params):
        """
        Initialize Kalman filters for tracking.
        
        Args:
            num_targets (int): Number of targets to track
            params (dict): Tracking parameters
        """
        self.num_targets = num_targets
        self.params = params
        self.filters = self._create_kalman_filters()
        
    def _create_kalman_filters(self):
        """
        Create and configure Kalman filters for all targets.
        
        Returns:
            list: List of configured cv2.KalmanFilter objects
        """
        filters = []
        
        for _ in range(self.num_targets):
            # Create Kalman filter with 5D state vector and 3D measurement vector
            kf = cv2.KalmanFilter(5, 3)
            
            # Measurement matrix: maps state [x,y,theta,vx,vy] to observation [x,y,theta]
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0, 0],  # x position
                [0, 1, 0, 0, 0],  # y position  
                [0, 0, 1, 0, 0]   # orientation
            ], np.float32)
            
            # Transition matrix: constant velocity model
            kf.transitionMatrix = np.array([
                [1, 0, 0, 1, 0],  # x(t+1) = x(t) + vx(t)
                [0, 1, 0, 0, 1],  # y(t+1) = y(t) + vy(t)
                [0, 0, 1, 0, 0],  # theta(t+1) = theta(t) (constant orientation)
                [0, 0, 0, 1, 0],  # vx(t+1) = vx(t) (constant velocity)
                [0, 0, 0, 0, 1]   # vy(t+1) = vy(t) (constant velocity)
            ], np.float32)
            
            # Process noise: models uncertainty in motion model
            kf.processNoiseCov = np.eye(5, dtype=np.float32) * self.params["KALMAN_NOISE_COVARIANCE"]
            
            # Measurement noise: models uncertainty in observations
            kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * self.params["KALMAN_MEASUREMENT_NOISE_COVARIANCE"]
            
            # Initial error covariance
            kf.errorCovPre = np.eye(5, dtype=np.float32)
            filters.append(kf)
            
        return filters
    
    def correct_filter(self, track_idx, measurement):
        """
        Update a specific Kalman filter with a measurement.
        
        Args:
            track_idx (int): Index of the track to update
            measurement (np.ndarray): Measurement vector [x, y, theta]
        """
        if 0 <= track_idx < len(self.filters):
            self.filters[track_idx].correct(measurement.reshape(3, 1))
    
    def initialize_filter(self, track_idx, initial_state):
        """
        Initialize a specific Kalman filter with an initial state.
        
        Args:
            track_idx (int): Index of the track to initialize
            initial_state (np.ndarray): Initial state vector [x, y, theta, vx, vy]
        """
        if 0 <= track_idx < len(self.filters):
            kf = self.filters[track_idx]
            kf.statePre = initial_state.copy()
            kf.statePost = initial_state.copy()
    
    def get_filter_state(self, track_idx):
        """
        Get the current state of a specific Kalman filter.
        
        Args:
            track_idx (int): Index of the track
            
        Returns:
            np.ndarray or None: Current state vector or None if invalid index
        """
        if 0 <= track_idx < len(self.filters):
            return self.filters[track_idx].statePost
        return None
    
    def get_predictions(self):
        """
        Get predictions from all Kalman filters.
        
        Returns:
            np.ndarray: Array of predictions with shape (N, 3) [x, y, theta]
        """
        preds = [kf.predict()[:3].flatten() for kf in self.filters]
        return np.array(preds, np.float32)
    
    def get_position_covariance(self, track_idx):
        """
        Get the position covariance matrix for a specific track.
        
        Args:
            track_idx (int): Index of the track
            
        Returns:
            np.ndarray: 2x2 position covariance matrix
        """
        if 0 <= track_idx < len(self.filters):
            return self.filters[track_idx].errorCovPre[:2, :2]
        return np.eye(2, dtype=np.float32)
