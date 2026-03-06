"""Factor stepper widget for multi-factor compositional labeling."""

from __future__ import annotations

from multi_tracker.classkit.config.schemas import LabelingScheme

# ---------------------------------------------------------------------------
# Pure state machine — no Qt dependencies, fully testable
# ---------------------------------------------------------------------------


class StepperState:
    """Tracks progress through a multi-factor labeling sequence."""

    def __init__(self, scheme: LabelingScheme) -> None:
        self._scheme = scheme
        self.picks: list[str] = []

    @property
    def current_factor_index(self) -> int:
        return len(self.picks)

    @property
    def is_complete(self) -> bool:
        return len(self.picks) == len(self._scheme.factors)

    @property
    def composite_label(self) -> str:
        if not self.is_complete:
            raise RuntimeError(
                "Stepper is not complete — call pick() for all factors first."
            )
        return self._scheme.encode_label(self.picks)

    def pick(self, label: str) -> None:
        """Record a label choice for the current factor and advance."""
        if self.is_complete:
            raise RuntimeError("All factors already picked. Call reset() first.")
        idx = self.current_factor_index
        allowed = self._scheme.factors[idx].labels
        if label not in allowed:
            raise ValueError(
                f"'{label}' not in labels {allowed} for factor '{self._scheme.factors[idx].name}'"
            )
        self.picks.append(label)

    def back(self) -> None:
        """Undo the last factor pick. No-op if at the start."""
        if self.picks:
            self.picks.pop()

    def reset(self) -> None:
        """Reset to the first factor."""
        self.picks.clear()

    @property
    def current_factor(self):
        if self.is_complete:
            return None
        return self._scheme.factors[self.current_factor_index]

    @property
    def total_factors(self) -> int:
        return len(self._scheme.factors)


# ---------------------------------------------------------------------------
# Qt widget — built lazily to avoid hard Qt import at module level
# ---------------------------------------------------------------------------


def _build_qt_widget(scheme: LabelingScheme):  # pragma: no cover
    """Build and return a FactorStepperWidget class (deferred Qt import)."""
    from PyQt6.QtCore import pyqtSignal
    from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    class FactorStepperWidget(QWidget):
        """Sequential per-factor label picker.

        Signals:
            label_committed(str): Emitted with the composite label when all
                factors are picked.
            skipped: Emitted when the user clicks Skip.
        """

        label_committed = pyqtSignal(str)
        skipped = pyqtSignal()

        def __init__(self, scheme: LabelingScheme, parent=None):
            super().__init__(parent)
            self._state = StepperState(scheme)
            self._scheme = scheme
            self._buttons: list[QPushButton] = []
            self._shortcut_map: dict[str, str] = {}
            self._build_ui()
            self._refresh()

        def _build_ui(self):
            self._outer = QVBoxLayout(self)
            self._outer.setContentsMargins(0, 0, 0, 0)
            self._outer.setSpacing(6)

            self._header = QLabel()
            self._header.setStyleSheet("color: #cccccc; font-weight: bold;")
            self._outer.addWidget(self._header)

            self._btn_row = QHBoxLayout()
            self._outer.addLayout(self._btn_row)

            nav_row = QHBoxLayout()
            self._back_btn = QPushButton("< Back")
            self._back_btn.setFixedWidth(80)
            self._back_btn.clicked.connect(self._on_back)
            self._skip_btn = QPushButton("Skip >")
            self._skip_btn.setFixedWidth(80)
            self._skip_btn.clicked.connect(self.skipped.emit)
            nav_row.addWidget(self._back_btn)
            nav_row.addStretch()
            nav_row.addWidget(self._skip_btn)
            self._outer.addLayout(nav_row)

        def _refresh(self):
            for btn in self._buttons:
                self._btn_row.removeWidget(btn)
                btn.deleteLater()
            self._buttons.clear()
            self._shortcut_map.clear()

            if self._state.is_complete:
                self._header.setText("All factors assigned.")
                self._back_btn.setEnabled(True)
                return

            factor = self._state.current_factor
            idx = self._state.current_factor_index
            self._header.setText(
                f"Factor {idx + 1} of {self._state.total_factors}: <b>{factor.name}</b>"
            )

            for i, label in enumerate(factor.labels):
                btn = QPushButton(label.capitalize())
                btn.setStyleSheet(
                    "QPushButton { background: #2d2d2d; color: #e0e0e0; border: 1px solid #555; "
                    "border-radius: 4px; padding: 6px 12px; } "
                    "QPushButton:hover { background: #007acc; }"
                )
                btn.clicked.connect(lambda checked, lbl=label: self._on_pick(lbl))
                self._btn_row.addWidget(btn)
                self._buttons.append(btn)

                key = factor.shortcut_keys[i] if i < len(factor.shortcut_keys) else None
                if key:
                    self._shortcut_map[key] = label

            self._back_btn.setEnabled(self._state.current_factor_index > 0)

        def _on_pick(self, label: str):
            self._state.pick(label)
            self._refresh()
            if self._state.is_complete:
                composite = self._state.composite_label
                self._state.reset()
                self._refresh()
                self.label_committed.emit(composite)

        def _on_back(self):
            self._state.back()
            self._refresh()

        def reset(self):
            self._state.reset()
            self._refresh()

        def handle_key(self, key: str) -> bool:
            """Call from parent keyPressEvent. Returns True if key was consumed."""
            label = self._shortcut_map.get(key.lower())
            if label:
                self._on_pick(label)
                return True
            return False

    return FactorStepperWidget
