"""
Real-time histogram widgets for parameter visualization.

This module provides matplotlib-based histogram widgets that update in real-time
to show distributions of tracking parameters like velocity, size, orientation, etc.
"""

from collections import deque

import matplotlib
import numpy as np
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

matplotlib.use("QtAgg")  # Qt6-compatible backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RealtimeHistogramWidget(QWidget):
    """
    Widget for displaying real-time histograms of detection parameters.

    This widget creates matplotlib histograms that update in real-time to show
    distributions of tracking parameters like velocity, size, orientation, etc.
    over a rolling window of recent frames.
    """

    def __init__(self, title="Parameter Histogram", bins=30, history_frames=300):
        """
        Initialize the histogram widget.

        Args:
            title (str): Title for the histogram display
            bins (int): Number of histogram bins
            history_frames (int): Number of recent frames to include in histogram
        """
        super().__init__()

        self.title = title
        self.bins = bins
        self.history_frames = history_frames
        self.data_history = deque(maxlen=history_frames)

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.axis = self.figure.add_subplot(111)

        # Set up layout
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Initialize empty plot
        self.axis.set_title(self.title)
        self.axis.set_xlabel("Value")
        self.axis.set_ylabel("Frequency")
        self.canvas.draw()

    def update_data(self: object, new_data: object) -> object:
        """
        Update histogram with new data points.

        Args:
            new_data (list): New data points to add to histogram
        """
        if not new_data:
            return

        # Add new data to history
        self.data_history.extend(new_data)

        # Update histogram
        if len(self.data_history) > 0:
            self.axis.clear()
            self.axis.hist(
                list(self.data_history), bins=self.bins, alpha=0.7, edgecolor="black"
            )
            self.axis.set_title(f"{self.title} (n={len(self.data_history)})")
            self.axis.set_xlabel("Value")
            self.axis.set_ylabel("Frequency")
            self.canvas.draw()

    def clear_data(self: object) -> object:
        """Clear all historical data."""
        self.data_history.clear()
        self.axis.clear()
        self.axis.set_title(self.title)
        self.axis.set_xlabel("Value")
        self.axis.set_ylabel("Frequency")
        self.canvas.draw()


class HistogramPanel(QWidget):
    """
    Panel containing multiple real-time histograms for different tracking parameters.
    """

    def __init__(self, history_frames=300):
        """
        Initialize the histogram panel.

        Args:
            history_frames (int): Number of recent frames to include in histograms
        """
        super().__init__()

        self.history_frames = history_frames

        # Create individual histogram widgets
        self.velocity_histogram = RealtimeHistogramWidget(
            "Velocity (px/frame)", bins=30, history_frames=history_frames
        )
        self.size_histogram = RealtimeHistogramWidget(
            "Object Size (px²)", bins=30, history_frames=history_frames
        )
        self.orientation_histogram = RealtimeHistogramWidget(
            "Orientation (radians)", bins=30, history_frames=history_frames
        )
        self.assignment_cost_histogram = RealtimeHistogramWidget(
            "Assignment Cost", bins=30, history_frames=history_frames
        )

        # Set up layout with 2x2 grid
        layout = QVBoxLayout()

        # Top row
        top_row = QHBoxLayout()
        top_row.addWidget(self.velocity_histogram)
        top_row.addWidget(self.size_histogram)

        # Bottom row
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.orientation_histogram)
        bottom_row.addWidget(self.assignment_cost_histogram)

        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self.setLayout(layout)

    def set_history_frames(self: object, history_frames: object) -> object:
        """
        Update the history window size for all histograms.

        Args:
            history_frames (int): New history window size
        """
        self.history_frames = history_frames

        # Update each histogram's history size
        self.velocity_histogram.data_history = deque(
            self.velocity_histogram.data_history, maxlen=history_frames
        )
        self.size_histogram.data_history = deque(
            self.size_histogram.data_history, maxlen=history_frames
        )
        self.orientation_histogram.data_history = deque(
            self.orientation_histogram.data_history, maxlen=history_frames
        )
        self.assignment_cost_histogram.data_history = deque(
            self.assignment_cost_histogram.data_history, maxlen=history_frames
        )

    def update_velocity_data(self: object, velocities: object) -> object:
        """Update velocity histogram with new data."""
        self.velocity_histogram.update_data(velocities)

    def update_size_data(self: object, sizes: object) -> object:
        """Update size histogram with new data."""
        self.size_histogram.update_data(sizes)

    def update_orientation_data(self: object, orientations: object) -> object:
        """Update orientation histogram with new data."""
        # Convert orientations to degrees for better readability
        orientations_deg = [np.degrees(o) for o in orientations]
        self.orientation_histogram.update_data(orientations_deg)

    def update_assignment_cost_data(self: object, costs: object) -> object:
        """Update assignment cost histogram with new data."""
        self.assignment_cost_histogram.update_data(costs)

    def clear_all_data(self: object) -> object:
        """Clear all histogram data."""
        self.velocity_histogram.clear_data()
        self.size_histogram.clear_data()
        self.orientation_histogram.clear_data()
        self.assignment_cost_histogram.clear_data()
