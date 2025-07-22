"""
Object detection utilities for multi-object tracking.

This module provides object detection from foreground masks, including contour
analysis, ellipse fitting, and object splitting for merged detections.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Detects objects in foreground masks and extracts measurements.
    
    Handles contour detection, ellipse fitting, size filtering, and
    conservative splitting of merged objects.
    """
    
    def __init__(self, params):
        """
        Initialize object detector.
        
        Args:
            params (dict): Detection parameters
        """
        self.params = params
        
    def detect_objects(self, foreground_mask, frame_count=0):
        """
        Detect objects in foreground mask and extract measurements.
        
        Args:
            foreground_mask (np.ndarray): Binary foreground mask
            frame_count (int): Current frame number for logging
            
        Returns:
            tuple: (measurements, sizes, shapes) where:
                - measurements: list of [x, y, theta] arrays
                - sizes: list of contour areas
                - shapes: list of (ellipse_area, aspect_ratio) tuples
        """
        # Find contours in mask
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Frame quality check - reject frames with too many contours
        max_targets = self.params["MAX_TARGETS"]
        max_contour_multiplier = self.params.get("MAX_CONTOUR_MULTIPLIER", 20)
        max_allowed_contours = max_targets * max_contour_multiplier
        
        if len(contours) > max_allowed_contours:
            logger.debug(f"Frame {frame_count}: Too many contours ({len(contours)} > {max_allowed_contours}), treating as no detections")
            return [], [], []
        
        measurements, sizes, shapes = [], [], []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small noise and invalid contours
            if area < self.params["MIN_CONTOUR_AREA"] or len(contour) < 5:
                continue
            
            # Fit ellipse to contour for position and orientation
            try:
                (cx, cy), (ax1, ax2), angle = cv2.fitEllipse(contour)
            except cv2.error:
                continue  # Skip invalid ellipses
            
            # Ensure major axis is ax1, adjust angle accordingly
            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                angle = (angle + 90) % 180
            
            # Convert angle to radians
            angle_rad = np.deg2rad(angle)
            
            # Store measurement and shape information
            measurements.append(np.array([cx, cy, angle_rad], np.float32))
            sizes.append(area)
            
            # Calculate ellipse area and aspect ratio
            ellipse_area = np.pi * (ax1/2) * (ax2/2)
            aspect_ratio = ax1 / ax2
            shapes.append((ellipse_area, aspect_ratio))
        
        # Apply size-based filtering if enabled
        if measurements and self.params.get("ENABLE_SIZE_FILTERING", False):
            measurements, sizes, shapes = self._filter_by_size(measurements, sizes, shapes)
        
        # Keep only the N largest detections
        max_targets = self.params["MAX_TARGETS"]
        if len(measurements) > max_targets:
            indices = np.argsort(sizes)[::-1][:max_targets]  # Sort by size, take largest N
            measurements = [measurements[i] for i in indices]
            shapes = [shapes[i] for i in indices]
        
        return measurements, sizes, shapes
    
    def _filter_by_size(self, measurements, sizes, shapes):
        """
        Filter detections by size range.
        
        Args:
            measurements (list): List of measurement arrays
            sizes (list): List of contour areas
            shapes (list): List of shape tuples
            
        Returns:
            tuple: Filtered (measurements, sizes, shapes)
        """
        min_size = self.params.get("MIN_OBJECT_SIZE", 0)
        max_size = self.params.get("MAX_OBJECT_SIZE", float('inf'))
        original_count = len(measurements)
        
        filtered_measurements, filtered_sizes, filtered_shapes = [], [], []
        
        for i, size in enumerate(sizes):
            if min_size <= size <= max_size:
                filtered_measurements.append(measurements[i])
                filtered_sizes.append(sizes[i])
                filtered_shapes.append(shapes[i])
        
        if len(filtered_measurements) != original_count:
            logger.debug(f"Size filtering: {original_count} detections -> {len(filtered_measurements)} detections (range: {min_size}-{max_size})")
        
        return filtered_measurements, filtered_sizes, filtered_shapes
    
    def apply_conservative_split(self, foreground_mask, detection_initialized=False):
        """
        Apply conservative object splitting to separate merged objects.
        
        Args:
            foreground_mask (np.ndarray): Binary foreground mask
            detection_initialized (bool): Whether detection system is initialized
            
        Returns:
            np.ndarray: Mask with split objects
        """
        if not detection_initialized or not self.params.get("ENABLE_CONSERVATIVE_SPLIT", True):
            return foreground_mask
        
        # Find contours to identify suspicious regions
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_targets = self.params["MAX_TARGETS"]
        
        # Identify suspicious contours (too large or wrong count)
        suspicious_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if (area > self.params["MERGE_AREA_THRESHOLD"] or
                sum(1 for c in contours if cv2.contourArea(c) > 0) < max_targets):
                suspicious_regions.append(cv2.boundingRect(contour))
        
        # Apply splitting to suspicious regions
        result_mask = foreground_mask.copy()
        for bx, by, bw, bh in suspicious_regions:
            sub_region = foreground_mask[by:by+bh, bx:bx+bw]
            split_region = self._local_conservative_split(sub_region, self.params)
            result_mask[by:by+bh, bx:bx+bw] = split_region
        
        return result_mask
    
    def _local_conservative_split(self, sub, p):
        """
        Apply conservative morphological operations to split merged objects.
        
        When multiple objects are detected as a single large contour (due to
        proximity or lighting), this method attempts to split them using
        erosion followed by opening operations.
        
        Args:
            sub (np.ndarray): Binary mask region to process
            p (dict): Parameters with CONSERVATIVE_KERNEL_SIZE and CONSERVATIVE_ERODE_ITER
            
        Returns:
            np.ndarray: Processed binary mask with potential object separation
        """
        k = p["CONSERVATIVE_KERNEL_SIZE"]
        it = p["CONSERVATIVE_ERODE_ITER"]
        
        # Create elliptical structuring element for natural object shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        
        # Erode to separate touching objects
        out = cv2.erode(sub, kernel, iterations=it)
        
        # Open to remove noise and smooth boundaries
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
