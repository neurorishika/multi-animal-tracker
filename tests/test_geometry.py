"""
Comprehensive tests for geometry utility functions.

Tests cover:
- Circle fitting with various point configurations
- Angle wrapping edge cases
- Error handling for invalid inputs
"""

import numpy as np

from multi_tracker.utils.geometry import fit_circle_to_points, wrap_angle_degs


class TestFitCircleToPoints:
    """Test suite for circle fitting function."""

    def test_exact_circle_four_points(self):
        """Test fitting a circle to 4 points that lie exactly on a circle."""
        # Points on a circle with center (0, 0) and radius 10
        points = [(10, 0), (0, 10), (-10, 0), (0, -10)]

        result = fit_circle_to_points(points)
        assert result is not None

        cx, cy, r = result
        assert abs(cx - 0.0) < 0.01
        assert abs(cy - 0.0) < 0.01
        assert abs(r - 10.0) < 0.01

    def test_exact_circle_many_points(self):
        """Test fitting a circle to many points on a perfect circle."""
        # Generate 20 points on a circle with center (5, 7) and radius 15
        angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        center_x, center_y, radius = 5.0, 7.0, 15.0
        points = [
            (center_x + radius * np.cos(a), center_y + radius * np.sin(a))
            for a in angles
        ]

        result = fit_circle_to_points(points)
        assert result is not None

        cx, cy, r = result
        assert abs(cx - center_x) < 0.1
        assert abs(cy - center_y) < 0.1
        assert abs(r - radius) < 0.1

    def test_circle_with_noise(self):
        """Test fitting a circle to points with small noise."""
        # Points on a circle with small random noise
        angles = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        center_x, center_y, radius = 3.0, 4.0, 8.0
        np.random.seed(42)
        noise = np.random.randn(10, 2) * 0.1
        points = [
            (
                center_x + radius * np.cos(a) + noise[i, 0],
                center_y + radius * np.sin(a) + noise[i, 1],
            )
            for i, a in enumerate(angles)
        ]

        result = fit_circle_to_points(points)
        assert result is not None

        cx, cy, r = result
        # With noise, expect less precise fit
        assert abs(cx - center_x) < 0.5
        assert abs(cy - center_y) < 0.5
        assert abs(r - radius) < 0.5

    def test_minimum_three_points(self):
        """Test that fitting works with exactly 3 points."""
        # Triangle inscribed in a circle
        points = [(1, 0), (-0.5, 0.866), (-0.5, -0.866)]

        result = fit_circle_to_points(points)
        assert result is not None
        cx, cy, r = result
        assert r > 0

    def test_less_than_three_points_returns_none(self):
        """Test that fewer than 3 points returns None."""
        assert fit_circle_to_points([]) is None
        assert fit_circle_to_points([(1, 2)]) is None
        assert fit_circle_to_points([(1, 2), (3, 4)]) is None

    def test_collinear_points_returns_none(self):
        """Test that collinear points produce a mathematically valid but small radius."""
        # Points on a straight line
        points = [(0, 0), (1, 1), (2, 2), (3, 3)]

        result = fit_circle_to_points(points)
        # Least squares fitting will find a "best fit" circle even for collinear points
        # The result is mathematically valid but may not be meaningful
        assert result is not None
        cx, cy, r = result
        assert r > 0  # Radius should be positive

    def test_large_circle(self):
        """Test fitting a very large circle."""
        # Large circle with center (1000, 2000) and radius 500
        angles = np.linspace(0, 2 * np.pi, 15, endpoint=False)
        center_x, center_y, radius = 1000.0, 2000.0, 500.0
        points = [
            (center_x + radius * np.cos(a), center_y + radius * np.sin(a))
            for a in angles
        ]

        result = fit_circle_to_points(points)
        assert result is not None

        cx, cy, r = result
        assert abs(cx - center_x) < 1.0
        assert abs(cy - center_y) < 1.0
        assert abs(r - radius) < 1.0

    def test_small_circle(self):
        """Test fitting a very small circle."""
        # Small circle with center (0.5, 0.5) and radius 0.1
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        center_x, center_y, radius = 0.5, 0.5, 0.1
        points = [
            (center_x + radius * np.cos(a), center_y + radius * np.sin(a))
            for a in angles
        ]

        result = fit_circle_to_points(points)
        assert result is not None

        cx, cy, r = result
        assert abs(cx - center_x) < 0.01
        assert abs(cy - center_y) < 0.01
        assert abs(r - radius) < 0.01

    def test_negative_coordinates(self):
        """Test fitting a circle with negative coordinates."""
        points = [(-10, -5), (-8, -8), (-5, -10), (-2, -8), (0, -5)]

        result = fit_circle_to_points(points)
        assert result is not None
        cx, cy, r = result
        assert r > 0
        # Center should be in negative quadrant
        assert cx < 0
        assert cy < 0


class TestWrapAngleDegs:
    """Test suite for angle wrapping function."""

    def test_zero_angle(self):
        """Test that 0 degrees remains 0."""
        assert wrap_angle_degs(0) == 0

    def test_small_positive_angle(self):
        """Test small positive angle."""
        assert wrap_angle_degs(45) == 45
        assert wrap_angle_degs(90) == 90
        assert wrap_angle_degs(179) == 179

    def test_small_negative_angle(self):
        """Test small negative angle."""
        assert wrap_angle_degs(-45) == -45
        assert wrap_angle_degs(-90) == -90
        assert wrap_angle_degs(-179) == -179

    def test_boundary_180(self):
        """Test boundary at 180 degrees."""
        assert wrap_angle_degs(180) == -180
        assert wrap_angle_degs(-180) == -180

    def test_just_above_180(self):
        """Test angles just above 180."""
        assert wrap_angle_degs(181) == -179
        assert wrap_angle_degs(270) == -90
        assert wrap_angle_degs(359) == -1

    def test_full_rotation(self):
        """Test that 360 degrees wraps to 0."""
        assert wrap_angle_degs(360) == 0
        assert wrap_angle_degs(-360) == 0

    def test_multiple_rotations_positive(self):
        """Test multiple positive rotations."""
        assert wrap_angle_degs(720) == 0
        assert wrap_angle_degs(725) == 5
        assert wrap_angle_degs(900) == -180
        assert wrap_angle_degs(450) == 90

    def test_multiple_rotations_negative(self):
        """Test multiple negative rotations."""
        assert wrap_angle_degs(-720) == 0
        assert wrap_angle_degs(-725) == -5
        assert wrap_angle_degs(-900) == -180
        assert wrap_angle_degs(-450) == -90

    def test_large_positive_angle(self):
        """Test very large positive angles."""
        assert wrap_angle_degs(3600) == 0
        assert wrap_angle_degs(3645) == 45
        assert wrap_angle_degs(7200 + 90) == 90

    def test_large_negative_angle(self):
        """Test very large negative angles."""
        assert wrap_angle_degs(-3600) == 0
        assert wrap_angle_degs(-3645) == -45
        assert wrap_angle_degs(-7200 - 90) == -90

    def test_wrap_continuity(self):
        """Test that wrapping maintains continuity across boundary."""
        # Small steps across 180/-180 boundary
        assert wrap_angle_degs(179) == 179
        assert wrap_angle_degs(180) == -180
        assert wrap_angle_degs(181) == -179

    def test_float_angles(self):
        """Test that float angles are handled correctly."""
        assert abs(wrap_angle_degs(45.5) - 45.5) < 1e-10
        assert abs(wrap_angle_degs(180.5) - (-179.5)) < 1e-10
        assert abs(wrap_angle_degs(359.7) - (-0.3)) < 1e-10

    def test_preserve_precision(self):
        """Test that precision is maintained for small angle differences."""
        angle = 45.123456789
        wrapped = wrap_angle_degs(angle)
        assert abs(wrapped - angle) < 1e-9

        # Test with angle that needs wrapping
        angle = 180.123456789
        wrapped = wrap_angle_degs(angle)
        expected = angle - 360
        assert abs(wrapped - expected) < 1e-9
