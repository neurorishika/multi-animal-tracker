"""
GUI module for Multi-Animal-Tracker.

This module contains the graphical user interface components including the main window.
"""

try:
    from .main_window import MainWindow
except (
    Exception
):  # pragma: no cover - allows lightweight metadata imports without GUI deps
    MainWindow = None

__all__ = ["MainWindow"]
