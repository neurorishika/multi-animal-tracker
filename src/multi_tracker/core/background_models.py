"""
Background modeling utilities for multi-object tracking.

This module provides background subtraction methods including the "lightest pixel"
approach and adaptive background modeling for handling lighting changes.
"""

import numpy as np
import cv2
import logging
import random
from collections import deque

from ..utils.image_processing import apply_image_adjustments

logger = logging.getLogger(__name__)


class BackgroundModel:
    """
    Manages background models for foreground detection in tracking.
    
    Supports multiple background modeling approaches:
    - Lightest pixel method: maintains maximum intensity at each pixel
    - Adaptive background: smoothly adapts to lighting changes
    - Lighting stabilization: compensates for illumination variations
    """
    
    def __init__(self, params):
        """
        Initialize background model.
        
        Args:
            params (dict): Tracking parameters
        """
        self.params = params
        self.lightest_background = None
        self.adaptive_background = None
        self.reference_intensity = None
        
    def prime_background(self, cap, roi_mask=None, resize_factor=1.0):
        """
        Initialize background model by sampling random frames from the video.
        
        This method samples random frames and builds a background model by taking the
        maximum intensity at each pixel. It also establishes a robust reference
        intensity for lighting stabilization by analyzing the pixel distribution.
        
        Args:
            cap (cv2.VideoCapture): Video capture object
            roi_mask (np.ndarray, optional): ROI mask to limit processing area
            resize_factor (float): Resize factor for performance optimization
        """
        logger.info("Initializing lightest-pixel background model...")
        
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or self.params["BACKGROUND_PRIME_FRAMES"] < 1:
            return
            
        # Limit sampling to available frames
        count = min(self.params["BACKGROUND_PRIME_FRAMES"], total)
        br, ct, gm = self.params["BRIGHTNESS"], self.params["CONTRAST"], self.params["GAMMA"]
        
        # Sample random frame indices to avoid temporal bias
        idxs = random.sample(range(total), count)
        bg_temp = None
        intensity_samples = []
        
        logger.info(f"Sampling {len(idxs)} random frames for background model...")
        
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: 
                continue
                
            # Apply same preprocessing as main loop
            if resize_factor < 1.0:
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor, 
                                 interpolation=cv2.INTER_AREA)
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, br, ct, gm)
            
            # Apply ROI mask if provided
            if roi_mask is not None:
                if resize_factor != 1.0:
                    roi_resized = cv2.resize(roi_mask, (gray.shape[1], gray.shape[0]), 
                                           interpolation=cv2.INTER_NEAREST)
                else:
                    roi_resized = roi_mask
                gray = cv2.bitwise_and(gray, gray, mask=roi_resized)
            else:
                roi_resized = None

            # Collect intensity statistics for reference (robust mean excluding outliers)
            if roi_resized is not None:
                # Calculate statistics only from ROI pixels
                roi_pixels = gray[roi_resized > 0]
                if len(roi_pixels) > 100:  # Ensure enough pixels
                    p25, p75 = np.percentile(roi_pixels, [25, 75])
                    mask = (roi_pixels >= p25) & (roi_pixels <= p75)
                    if np.sum(mask) > 0:
                        intensity_samples.append(np.mean(roi_pixels[mask]))
            else:
                # Use entire frame if no ROI
                frame_flat = gray.flatten()
                p25, p75 = np.percentile(frame_flat, [25, 75])
                mask = (frame_flat >= p25) & (frame_flat <= p75)
                if np.sum(mask) > 0:
                    intensity_samples.append(np.mean(frame_flat[mask]))
            
            # Build maximum intensity background model
            if bg_temp is None:
                bg_temp = gray.astype(np.float32)
            else:
                bg_temp = np.maximum(bg_temp, gray.astype(np.float32))
        
        # Store the background model
        if bg_temp is not None:
            self.lightest_background = bg_temp
            self.adaptive_background = bg_temp.copy()
            logger.info(f"Background model initialized with {len(intensity_samples)} samples")
        else:
            logger.error("Failed to initialize background model - no valid frames found")
            return
            
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Establish reference intensity from sampled frames
        if intensity_samples:
            self.reference_intensity = np.median(intensity_samples)
            logger.info(f"Reference intensity established: {self.reference_intensity:.1f}")
        else:
            # Fallback: use background model statistics
            if roi_mask is not None:
                roi_resized = cv2.resize(roi_mask, (bg_temp.shape[1], bg_temp.shape[0]), 
                                       interpolation=cv2.INTER_NEAREST) if resize_factor != 1.0 else roi_mask
                roi_bg_pixels = bg_temp[roi_resized > 0]
                self.reference_intensity = np.mean(roi_bg_pixels) if len(roi_bg_pixels) > 0 else np.mean(bg_temp)
            else:
                self.reference_intensity = np.mean(bg_temp)
            logger.warning(f"Using fallback reference intensity: {self.reference_intensity:.1f}")
    
    def update_background(self, frame, roi_mask=None, tracking_stabilized=False):
        """
        Update background models with new frame.
        
        Args:
            frame (np.ndarray): Current grayscale frame
            roi_mask (np.ndarray, optional): ROI mask to limit updates
            tracking_stabilized (bool): Whether tracking system is stable
            
        Returns:
            np.ndarray: Background image for subtraction
        """
        # Initialize background on first frame
        if self.lightest_background is None:
            self.lightest_background = frame.astype(np.float32)
            self.adaptive_background = frame.astype(np.float32)
            return frame
        
        # Update lightest pixel background
        if roi_mask is not None:
            roi_mask_bool = roi_mask > 0
            self.lightest_background[roi_mask_bool] = np.maximum(
                self.lightest_background[roi_mask_bool],
                frame.astype(np.float32)[roi_mask_bool]
            )
        else:
            self.lightest_background = np.maximum(self.lightest_background, frame.astype(np.float32))
        
        # Update adaptive background for lighting changes
        if self.params.get("ENABLE_ADAPTIVE_BACKGROUND", True) and self.adaptive_background is not None:
            learning_rate = self.params.get("BACKGROUND_LEARNING_RATE", 0.001)
            if roi_mask is not None:
                roi_mask_bool = roi_mask > 0
                self.adaptive_background[roi_mask_bool] = (
                    (1 - learning_rate) * self.adaptive_background[roi_mask_bool] +
                    learning_rate * frame.astype(np.float32)[roi_mask_bool]
                )
            else:
                self.adaptive_background = ((1 - learning_rate) * self.adaptive_background +
                                          learning_rate * frame.astype(np.float32))
            
            # Choose background based on tracking stability
            if tracking_stabilized:
                return cv2.convertScaleAbs(self.adaptive_background)
            else:
                return cv2.convertScaleAbs(self.lightest_background)
        else:
            return cv2.convertScaleAbs(self.lightest_background)
    
    def generate_foreground_mask(self, frame, background):
        """
        Generate foreground mask using background subtraction.
        
        Args:
            frame (np.ndarray): Current grayscale frame
            background (np.ndarray): Background image
            
        Returns:
            np.ndarray: Binary foreground mask
        """
        dark_on_light = self.params.get("DARK_ON_LIGHT_BACKGROUND", True)
        
        # Choose subtraction direction based on contrast
        if dark_on_light:
            diff = cv2.subtract(background, frame)
        else:
            diff = cv2.subtract(frame, background)
        
        # Apply binary thresholding
        _, fg_mask = cv2.threshold(diff, self.params["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)
        
        return fg_mask
    
    def apply_morphological_operations(self, mask):
        """
        Clean up foreground mask using morphological operations.
        
        Args:
            mask (np.ndarray): Binary foreground mask
            
        Returns:
            np.ndarray: Cleaned foreground mask
        """
        # Basic morphological operations
        kernel_size = self.params["MORPH_KERNEL_SIZE"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Remove noise and fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional dilation for thin animals if enabled
        if self.params.get("ENABLE_ADDITIONAL_DILATION", False):
            dil_size = self.params.get("DILATION_KERNEL_SIZE", 3)
            dil_iterations = self.params.get("DILATION_ITERATIONS", 2)
            dil_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_size, dil_size))
            mask = cv2.dilate(mask, dil_kernel, iterations=dil_iterations)
        
        return mask