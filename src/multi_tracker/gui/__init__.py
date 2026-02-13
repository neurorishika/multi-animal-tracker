"""
GUI module for Multi-Animal-Tracker.

This module contains the graphical user interface components including the main window,
histogram widgets, and other UI elements.
"""

try:
    from .main_window import MainWindow
    from .widgets.histograms import RealtimeHistogramWidget, HistogramPanel
except (
    Exception
):  # pragma: no cover - allows lightweight metadata imports without GUI deps
    MainWindow = None
    RealtimeHistogramWidget = None
    HistogramPanel = None

__all__ = ["MainWindow", "RealtimeHistogramWidget", "HistogramPanel"]
