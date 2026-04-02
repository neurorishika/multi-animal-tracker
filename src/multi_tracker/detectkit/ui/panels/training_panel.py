"""Training panel stub -- implemented in Task 6."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class TrainingPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Training panel -- loading..."))

    def set_project(self, proj, main_window):
        pass

    def collect_state(self, proj):
        pass
