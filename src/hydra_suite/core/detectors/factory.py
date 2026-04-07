"""Factory function for creating detector instances."""

import logging

logger = logging.getLogger(__name__)


def create_detector(params: object) -> object:
    """
    Factory function to create the appropriate detector based on configuration.

    Args:
        params: Configuration dictionary

    Returns:
        ObjectDetector or YOLOOBBDetector instance
    """
    detection_method = params.get("DETECTION_METHOD", "background_subtraction")

    if detection_method == "yolo_obb":
        from .yolo_detector import YOLOOBBDetector

        logger.info("Creating YOLO OBB detector")
        return YOLOOBBDetector(params)
    else:
        from .bg_detector import ObjectDetector

        logger.info("Creating background subtraction detector")
        return ObjectDetector(params)
