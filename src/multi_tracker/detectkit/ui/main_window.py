"""DetectKit main window -- placeholder for Task 4."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DetectKit")
        self.setCentralWidget(QLabel("DetectKit -- coming soon"))
