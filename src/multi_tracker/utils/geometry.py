"""
Utility functions for geometry operations in multi-animal tracking.
"""

import numpy as np


def fit_circle_to_points(points: object) -> object:
    """
    Fit a circle to a set of points using least squares optimization.

    Uses algebraic circle fitting method for robust estimation from 3+ points.

    Args:
        points (list): List of (x, y) coordinate tuples

    Returns:
        tuple: (center_x, center_y, radius) or None if fitting fails
    """
    if len(points) < 3:
        return None

    try:
        # Convert to numpy array
        points = np.array(points, dtype=np.float64)
        x, y = points[:, 0], points[:, 1]

        # Set up system of equations for circle fitting
        # Circle equation: (x-a)² + (y-b)² = r²
        # Expanded: x² + y² - 2ax - 2by + (a² + b² - r²) = 0
        # Linear form: x² + y² = 2ax + 2by - (a² + b² - r²)

        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x**2 + y**2

        # Solve using least squares
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        center_x, center_y, c = params

        # Calculate radius from fitted parameters
        radius = np.sqrt(center_x**2 + center_y**2 + c)

        # Validate the result
        if radius > 0 and not np.isnan(radius):
            return (float(center_x), float(center_y), float(radius))
        else:
            return None

    except (np.linalg.LinAlgError, ValueError):
        return None


def wrap_angle_degs(deg: float) -> float:
    """
    Normalize angle to [-180, 180] degree range.

    This function is crucial for orientation tracking to ensure smooth
    angle transitions and prevent discontinuities at the 0/360 boundary.

    Args:
        deg (float): Input angle in degrees

    Returns:
        float: Normalized angle in range [-180, 180] degrees
    """
    deg %= 360
    return deg - 360 if deg >= 180 else deg
