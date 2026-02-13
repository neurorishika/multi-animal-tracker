"""
Tests for ObjectDetector class (background subtraction detection).

Tests cover:
- Basic object detection from foreground masks
- Conservative split for merged objects
- Size filtering
- Detection count limiting
- Measurement extraction (position, angle, size)
"""

import cv2
import numpy as np

from multi_tracker.core.detectors.engine import ObjectDetector


def create_test_foreground_mask(num_objects=2, width=320, height=240, object_size=30):
    """Create a synthetic foreground mask with circular objects."""
    mask = np.zeros((height, width), dtype=np.uint8)

    if num_objects == 1:
        # Single object in center
        cv2.circle(mask, (width // 2, height // 2), object_size, 255, -1)
    elif num_objects == 2:
        # Two objects
        cv2.circle(mask, (width // 3, height // 2), object_size, 255, -1)
        cv2.circle(mask, (2 * width // 3, height // 2), object_size, 255, -1)
    elif num_objects >= 3:
        # Multiple objects in a grid
        positions = [
            (width // 4, height // 3),
            (3 * width // 4, height // 3),
            (width // 4, 2 * height // 3),
            (3 * width // 4, 2 * height // 3),
        ]
        for i, (x, y) in enumerate(positions[:num_objects]):
            cv2.circle(mask, (x, y), object_size, 255, -1)

    return mask


class TestObjectDetector:
    """Test suite for ObjectDetector class."""

    def test_initialization(self):
        """Test detector initialization with basic parameters."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "MERGE_AREA_THRESHOLD": 1000,
        }

        detector = ObjectDetector(params)

        assert detector.params == params

    def test_detect_single_object(self):
        """Test detection of a single object."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(num_objects=1, object_size=25)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should detect one object
        assert len(meas) == 1
        assert len(sizes) == 1
        assert len(shapes) == 1
        assert len(confidences) == 1
        assert yolo_results is None

        # Check measurement format [cx, cy, angle]
        assert meas[0].shape == (3,)
        assert 0 <= meas[0][0] <= 320  # cx
        assert 0 <= meas[0][1] <= 240  # cy

        # Check that size is reasonable
        assert sizes[0] > 100  # Should be larger than min area

    def test_detect_multiple_objects(self):
        """Test detection of multiple objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(num_objects=3, object_size=25)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should detect three objects
        assert len(meas) == 3
        assert len(sizes) == 3
        assert len(shapes) == 3
        assert len(confidences) == 3

    def test_min_contour_area_filtering(self):
        """Test that small contours are filtered out."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 500,  # High threshold
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(
            num_objects=2, object_size=10
        )  # Small objects

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Small objects should be filtered
        assert len(meas) == 0

    def test_max_targets_limiting(self):
        """Test that detections are limited to MAX_TARGETS."""
        params = {
            "MAX_TARGETS": 2,  # Limit to 2
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(
            num_objects=4, object_size=25
        )  # Create 4 objects

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Should be limited to MAX_TARGETS
        assert len(meas) <= 2

    def test_size_filtering_enabled(self):
        """Test size-based filtering of detections."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 50,
            "ENABLE_SIZE_FILTERING": True,
            "MIN_OBJECT_SIZE": 1500,  # Minimum size threshold
            "MAX_OBJECT_SIZE": 3000,  # Maximum size threshold
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)

        # Create mask with objects of different sizes
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (80, 120), 15, 255, -1)  # Small (area ~700)
        cv2.circle(mask, (160, 120), 25, 255, -1)  # Medium (area ~2000)
        cv2.circle(mask, (240, 120), 40, 255, -1)  # Large (area ~5000)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            mask, frame_count=0
        )

        # Only medium object should pass filtering
        assert len(meas) == 1
        assert 1500 <= sizes[0] <= 3000

    def test_size_filtering_disabled(self):
        """Test that all objects are detected when size filtering is disabled."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "ENABLE_SIZE_FILTERING": False,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(num_objects=3, object_size=25)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        assert len(meas) == 3

    def test_empty_mask(self):
        """Test detection on empty mask."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = np.zeros((240, 320), dtype=np.uint8)  # Empty mask

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        assert len(meas) == 0
        assert len(sizes) == 0
        assert len(shapes) == 0
        assert len(confidences) == 0

    def test_too_many_contours(self):
        """Test handling of excessive contours (noise)."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 1,  # Very low to allow noise
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)

        # Create very noisy mask with many small contours
        mask = np.random.randint(0, 2, (240, 320), dtype=np.uint8) * 255

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            mask, frame_count=0
        )

        # Should handle gracefully (may return empty or limited results)
        assert isinstance(meas, list)
        assert isinstance(sizes, list)

    def test_shape_information(self):
        """Test that shape information (area, aspect ratio) is computed."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(num_objects=2, object_size=25)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Check shapes format [(area, aspect_ratio), ...]
        for shape in shapes:
            assert len(shape) == 2
            area, aspect_ratio = shape
            assert area > 0
            assert aspect_ratio > 0

    def test_confidence_values(self):
        """Test that confidence values are returned (NaN for background subtraction)."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)
        fg_mask = create_test_foreground_mask(num_objects=2, object_size=25)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            fg_mask, frame_count=0
        )

        # Confidences should be NaN for background subtraction
        assert len(confidences) == len(meas)
        for conf in confidences:
            assert np.isnan(conf)

    def test_conservative_split_basic(self):
        """Test conservative split on a simple mask."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "MERGE_AREA_THRESHOLD": 1000,
        }

        detector = ObjectDetector(params)

        # Create a large blob that might be merged objects
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(mask, (100, 80), (220, 160), 255, -1)  # Large rectangle

        result_mask = detector.apply_conservative_split(mask)

        # Should return a valid mask
        assert result_mask.shape == mask.shape
        assert result_mask.dtype == np.uint8

    def test_conservative_split_preserves_small_objects(self):
        """Test that conservative split doesn't over-erode small objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "CONSERVATIVE_KERNEL_SIZE": 3,
            "CONSERVATIVE_ERODE_ITER": 1,
            "MERGE_AREA_THRESHOLD": 2000,  # High threshold
        }

        detector = ObjectDetector(params)

        # Create small objects that shouldn't be affected
        mask = create_test_foreground_mask(num_objects=2, object_size=20)
        original_white_pixels = np.sum(mask > 0)

        result_mask = detector.apply_conservative_split(mask)
        result_white_pixels = np.sum(result_mask > 0)

        # Should preserve most pixels for small objects below threshold
        assert result_white_pixels >= original_white_pixels * 0.5

    def test_measurement_coordinates(self):
        """Test that measurements contain valid coordinates."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)

        # Create object at known position
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(mask, (160, 120), 25, 255, -1)  # Center of image

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            mask, frame_count=0
        )

        assert len(meas) == 1
        cx, cy, angle = meas[0]

        # Should be near center
        assert 140 <= cx <= 180
        assert 100 <= cy <= 140
        # Angle should be in radians
        assert -np.pi <= angle <= np.pi

    def test_ellipse_fitting(self):
        """Test that ellipse fitting works for elongated objects."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)

        # Create elongated ellipse
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.ellipse(mask, (160, 120), (40, 20), 0, 0, 360, 255, -1)

        meas, sizes, shapes, yolo_results, confidences = detector.detect_objects(
            mask, frame_count=0
        )

        assert len(meas) == 1
        assert len(shapes) == 1

        area, aspect_ratio = shapes[0]
        # Elongated ellipse should have aspect ratio > 1
        assert aspect_ratio > 1.5

    def test_different_frame_sizes(self):
        """Test detection on different frame sizes."""
        params = {
            "MAX_TARGETS": 4,
            "MIN_CONTOUR_AREA": 100,
            "MAX_CONTOUR_MULTIPLIER": 20,
        }

        detector = ObjectDetector(params)

        sizes = [(240, 320), (480, 640), (120, 160)]

        for height, width in sizes:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (width // 2, height // 2), 25, 255, -1)

            meas, det_sizes, shapes, yolo_results, confidences = (
                detector.detect_objects(mask, frame_count=0)
            )

            assert len(meas) == 1
            # Coordinates should be within frame
            assert 0 <= meas[0][0] <= width
            assert 0 <= meas[0][1] <= height
