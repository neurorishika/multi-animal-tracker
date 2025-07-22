"""
Object detection utilities for multi-object tracking.
Functionally identical to the original implementation's detection logic.
"""
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Detects objects in foreground masks and extracts measurements.
    """
    def __init__(self, params):
        self.params = params

    def _local_conservative_split(self, sub):
        """Applies conservative morphological operations to split merged objects."""
        k = self.params["CONSERVATIVE_KERNEL_SIZE"]
        it = self.params["CONSERVATIVE_ERODE_ITER"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.erode(sub, kernel, iterations=it)
        return cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    def apply_conservative_split(self, fg_mask):
        """Attempts to split merged objects in the foreground mask."""
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        N = self.params["MAX_TARGETS"]
        
        suspicious = [
            cv2.boundingRect(c) for c in cnts
            if cv2.contourArea(c) > self.params["MERGE_AREA_THRESHOLD"]
            or sum(1 for cc in cnts if cv2.contourArea(cc) > 0) < N
        ]
        
        for bx, by, bw, bh in suspicious:
            sub = fg_mask[by:by+bh, bx:bx+bw]
            fg_mask[by:by+bh, bx:bx+bw] = self._local_conservative_split(sub)
            
        return fg_mask

    def detect_objects(self, fg_mask, frame_count):
        """Detects and measures objects from the final foreground mask."""
        p = self.params
        cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        N = p["MAX_TARGETS"]
        max_allowed_contours = N * p.get("MAX_CONTOUR_MULTIPLIER", 20)
        
        if len(cnts) > max_allowed_contours:
            logger.debug(f"Frame {frame_count}: Too many contours ({len(cnts)}), skipping.")
            return [], [], []

        meas, sizes, shapes = [], [], []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p["MIN_CONTOUR_AREA"] or len(c) < 5:
                continue

            (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(c)
            
            if ax1 < ax2:
                ax1, ax2 = ax2, ax1
                ang = (ang + 90) % 180
            
            meas.append(np.array([cx, cy, np.deg2rad(ang)], np.float32))
            sizes.append(area)
            shapes.append((np.pi * (ax1/2) * (ax2/2), ax1/ax2 if ax2 > 0 else 0))

        if meas and p.get("ENABLE_SIZE_FILTERING", False):
            min_size = p.get("MIN_OBJECT_SIZE", 0)
            max_size = p.get("MAX_OBJECT_SIZE", float('inf'))
            
            original_count = len(meas)
            filtered = [(m, s, sh) for m, s, sh in zip(meas, sizes, shapes) if min_size <= s <= max_size]
            
            if filtered:
                meas, sizes, shapes = zip(*filtered)
                meas, sizes, shapes = list(meas), list(sizes), list(shapes)
            else:
                meas, sizes, shapes = [], [], []

            if len(meas) != original_count:
                logger.debug(f"Size filtering: {original_count} -> {len(meas)} detections")

        if len(meas) > N:
            idxs = np.argsort(sizes)[::-1][:N]
            meas = [meas[i] for i in idxs]
            shapes = [shapes[i] for i in idxs]

        return meas, sizes, shapes