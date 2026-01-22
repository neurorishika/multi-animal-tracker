"""
Background modeling utilities for multi-object tracking.
Functionally identical to the original implementation's background logic.
"""

import numpy as np
import cv2
import logging
import random
from ..utils.image_processing import apply_image_adjustments

logger = logging.getLogger(__name__)


class BackgroundModel:
    """
    Manages background models for foreground detection in tracking.
    """

    def __init__(self, params):
        self.params = params
        self.lightest_background = None
        self.adaptive_background = None
        self.reference_intensity = None

    def prime_background(self, cap):
        """
        Initialize background model using "lightest pixel" method with lighting reference.
        This is an exact port of the original `prime_lightest_background` method.
        """
        p = self.params
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or p["BACKGROUND_PRIME_FRAMES"] < 1:
            return

        count = min(p["BACKGROUND_PRIME_FRAMES"], total)
        br, ct, gm = p["BRIGHTNESS"], p["CONTRAST"], p["GAMMA"]
        ROI_mask = p.get("ROI_MASK", None)
        resize_f = p.get("RESIZE_FACTOR", 1.0)

        idxs = random.sample(range(total), count)
        bg_temp = None
        intensity_samples = []

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if resize_f < 1.0:
                frame = cv2.resize(
                    frame,
                    (0, 0),
                    fx=resize_f,
                    fy=resize_f,
                    interpolation=cv2.INTER_AREA,
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = apply_image_adjustments(gray, br, ct, gm)

            roi_resized = None
            if ROI_mask is not None:
                roi_resized = (
                    cv2.resize(
                        ROI_mask,
                        (gray.shape[1], gray.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    if resize_f != 1.0
                    else ROI_mask
                )

            if ROI_mask is not None and roi_resized is not None:
                roi_pixels = gray[roi_resized > 0]
                if len(roi_pixels) > 100:
                    p25, p75 = np.percentile(roi_pixels, [25, 75])
                    mask = (roi_pixels >= p25) & (roi_pixels <= p75)
                    if np.sum(mask) > 0:
                        intensity_samples.append(np.mean(roi_pixels[mask]))
            else:
                frame_flat = gray.flatten()
                p25, p75 = np.percentile(frame_flat, [25, 75])
                mask = (frame_flat >= p25) & (frame_flat <= p75)
                if np.sum(mask) > 0:
                    intensity_samples.append(np.mean(frame_flat[mask]))

            if bg_temp is None:
                bg_temp = gray.astype(np.float32)
            else:
                bg_temp = np.maximum(bg_temp, gray.astype(np.float32))

        if bg_temp is not None:
            self.lightest_background = bg_temp
            self.adaptive_background = bg_temp.copy()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if intensity_samples:
                self.reference_intensity = np.median(intensity_samples)
                logger.info(
                    f"Reference intensity established: {self.reference_intensity:.1f}"
                )
            else:  # Fallback
                if ROI_mask is not None:
                    roi_resized = (
                        cv2.resize(
                            ROI_mask,
                            (bg_temp.shape[1], bg_temp.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        if resize_f != 1.0
                        else ROI_mask
                    )
                    roi_bg_pixels = bg_temp[roi_resized > 0]
                    self.reference_intensity = (
                        np.mean(roi_bg_pixels)
                        if len(roi_bg_pixels) > 0
                        else np.mean(bg_temp)
                    )
                else:
                    self.reference_intensity = np.mean(bg_temp)
                logger.info(
                    f"Fallback reference intensity: {self.reference_intensity:.1f}"
                )

    def update_and_get_background(self, gray, roi_mask, tracking_stabilized):
        """Updates background models and returns the one for subtraction."""
        p = self.params
        if self.lightest_background is None:
            self.lightest_background = gray.astype(np.float32)
            self.adaptive_background = gray.astype(np.float32)
            return None  # Indicates first frame

        # Update full-frame background (ROI masking happens during detection)
        self.lightest_background = np.maximum(
            self.lightest_background, gray.astype(np.float32)
        )

        if (
            p.get("ENABLE_ADAPTIVE_BACKGROUND", True)
            and self.adaptive_background is not None
        ):
            learning_rate = p.get("BACKGROUND_LEARNING_RATE", 0.001)
            self.adaptive_background = (
                1 - learning_rate
            ) * self.adaptive_background + learning_rate * gray.astype(np.float32)

        if tracking_stabilized:
            return cv2.convertScaleAbs(self.adaptive_background)
        else:
            return cv2.convertScaleAbs(self.lightest_background)

    def generate_foreground_mask(self, gray, background):
        """Generates the foreground mask from the gray frame and background."""
        p = self.params
        dark_on_light = p.get("DARK_ON_LIGHT_BACKGROUND", True)

        if dark_on_light:
            diff = cv2.subtract(background, gray)
        else:
            diff = cv2.subtract(gray, background)

        _, fg_mask = cv2.threshold(diff, p["THRESHOLD_VALUE"], 255, cv2.THRESH_BINARY)

        ksz = p["MORPH_KERNEL_SIZE"]
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, ker)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, ker)

        if p.get("ENABLE_ADDITIONAL_DILATION", False):
            dil_ksz = p.get("DILATION_KERNEL_SIZE", 3)
            dil_iter = p.get("DILATION_ITERATIONS", 2)
            dil_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dil_ksz, dil_ksz)
            )
            fg_mask = cv2.dilate(fg_mask, dil_kernel, iterations=dil_iter)

        return fg_mask
