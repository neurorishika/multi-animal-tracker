"""
GUI module for Multi-Animal Tracker.

This module contains the graphical user interface components including the main window,
histogram widgets, and other UI elements.
"""

from .main_window import MainWindow
from .widgets.histograms import RealtimeHistogramWidget, HistogramPanel

__all__ = ["MainWindow", "RealtimeHistogramWidget", "HistogramPanel"]
