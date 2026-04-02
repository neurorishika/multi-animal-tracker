"""Evaluation panel stub -- implemented in Task 7."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class EvaluationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Evaluation -- loading..."))

    def set_project(self, proj, main_window):
        pass

    def collect_state(self, proj):
        pass
