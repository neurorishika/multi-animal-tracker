"""
Utility functions for image processing in multi-animal tracking.
"""

import cv2
import numpy as np
from collections import deque

def apply_image_adjustments(gray, brightness, contrast, gamma):
    """
    Apply brightness, contrast, and gamma corrections to grayscale image.
    
    This preprocessing step enhances image quality for better foreground detection
    by adjusting luminance characteristics to account for varying lighting conditions.
    
    Args:
        gray (np.ndarray): Input grayscale image
        brightness (float): Brightness adjustment (-255 to +255)
        contrast (float): Contrast multiplier (0.0 to 3.0+)
        gamma (float): Gamma correction factor (0.1 to 3.0+)
        
    Returns:
        np.ndarray: Adjusted grayscale image
        
    Note:
        - Brightness: Additive adjustment (linear shift)
        - Contrast: Multiplicative adjustment (scaling)
        - Gamma: Power-law transformation for non-linear luminance correction
    """
    # Apply brightness and contrast using OpenCV's optimized function
    adj = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    
    # Apply gamma correction if significantly different from 1.0
    if abs(gamma-1.0) > 1e-3:
        # Create lookup table for gamma correction
        lut = np.array([
            np.clip(((i/255.0)**(1.0/gamma))*255.0, 0, 255) 
            for i in range(256)
        ], np.uint8)
        adj = cv2.LUT(adj, lut)
    return adj

def stabilize_lighting(frame, reference_intensity, current_intensity_history, alpha=0.95, roi_mask=None, median_window=5, lighting_state=None):
    """
    Stabilize lighting conditions by normalizing frame intensity to a reference level.
    
    This function compensates for gradual lighting changes by:
    1. Computing frame's global intensity statistics (within ROI if provided)
    2. Comparing to reference intensity established during background priming
    3. Applying smooth intensity correction to maintain consistent illumination
    4. Using rolling history with median filtering to suppress high-frequency noise
    
    Args:
        frame (np.ndarray): Input grayscale frame
        reference_intensity (float): Target intensity level from background priming
        current_intensity_history (deque): Rolling history of recent frame intensities
        alpha (float): Smoothing factor for intensity adaptation (0.9-0.99)
        roi_mask (np.ndarray, optional): Binary mask defining region of interest
        median_window (int): Window size for median filtering (3-15)
        lighting_state (dict, optional): Dictionary to store smoothing state
        
    Returns:
        tuple: (stabilized_frame, updated_intensity_history, current_mean_intensity)
    """
    if reference_intensity is None:
        return frame, current_intensity_history, np.mean(frame)
    
    # Initialize lighting state if not provided
    if lighting_state is None:
        lighting_state = {}
    
    # Calculate current frame's mean intensity (excluding extreme outliers)
    # Use only ROI pixels if mask is provided
    if roi_mask is not None:
        roi_pixels = frame[roi_mask > 0]
        if len(roi_pixels) < 100:  # Fallback if ROI too small
            frame_flat = frame.flatten()
        else:
            frame_flat = roi_pixels
    else:
        frame_flat = frame.flatten()
    
    # Use percentile-based mean to reduce animal influence
    p10, p25, p75, p90 = np.percentile(frame_flat, [10, 25, 75, 90])
    
    # More robust mean calculation excluding outliers (animals, shadows, bright spots)
    # Use tighter percentile range to reduce noise influence
    mask = (frame_flat >= p25) & (frame_flat <= p75)
    if np.sum(mask) > frame_flat.size * 0.1:  # Ensure enough pixels for reliable estimate
        current_mean = np.mean(frame_flat[mask])
    else:
        # Fallback to wider range if too few pixels in tight range
        mask = (frame_flat >= p10) & (frame_flat <= p90)
        current_mean = np.mean(frame_flat[mask]) if np.sum(mask) > 0 else np.mean(frame_flat)
    
    # Update intensity history for smoothing
    current_intensity_history.append(current_mean)
    
    # Apply median filtering to suppress high-frequency fluctuations
    if len(current_intensity_history) >= median_window:
        # Use median of recent history to suppress spikes
        recent_values = list(current_intensity_history)[-median_window:]
        median_intensity = np.median(recent_values)
        
        # Use exponential moving average with median-filtered input
        if len(current_intensity_history) > 1:
            # Get the last smoothed value (or initialize with median)
            if 'smoothed_value' in lighting_state:
                smoothed_intensity = lighting_state['smoothed_value']
            else:
                smoothed_intensity = median_intensity
            
            # Apply smoothing with median-filtered input
            smoothed_intensity = alpha * smoothed_intensity + (1 - alpha) * median_intensity
            lighting_state['smoothed_value'] = smoothed_intensity
        else:
            smoothed_intensity = median_intensity
            lighting_state['smoothed_value'] = smoothed_intensity
    else:
        # Not enough history yet, use simple averaging
        smoothed_intensity = np.mean(list(current_intensity_history))
        lighting_state['smoothed_value'] = smoothed_intensity
    
    # Calculate correction factor to match reference intensity
    if smoothed_intensity > 0:
        correction_factor = reference_intensity / smoothed_intensity
        # Limit correction to prevent over-compensation and reduce sudden jumps
        correction_factor = np.clip(correction_factor, 0.7, 1.4)  # Tighter bounds for stability
        
        # Apply additional smoothing to correction factor itself
        if 'last_correction' in lighting_state:
            correction_smooth_alpha = 0.8  # Smooth correction factor changes
            correction_factor = (correction_smooth_alpha * lighting_state['last_correction'] + 
                               (1 - correction_smooth_alpha) * correction_factor)
        lighting_state['last_correction'] = correction_factor
    else:
        correction_factor = 1.0
    
    # Apply lighting correction
    stabilized = cv2.convertScaleAbs(frame, alpha=correction_factor, beta=0)
    
    return stabilized, current_intensity_history, current_mean
