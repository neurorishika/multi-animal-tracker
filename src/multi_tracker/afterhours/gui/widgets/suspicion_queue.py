"""Suspicion queue widget for MAT-afterhours.

Displays a scrollable list of :class:`SuspicionEvent` cards organised
by tiered score thresholds.  Each click on a card emits
:pyqtSignal:`event_selected` so the main window can seek the video and
offer a review dialog.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.event_types import SuspicionEvent

# ---------------------------------------------------------------------------
# _EventCard
# ---------------------------------------------------------------------------


class _EventCard(QFrame):
    """Visual card representing a single :class:`SuspicionEvent`."""

    clicked = Signal(object)

    def __init__(self, event: SuspicionEvent, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.swap_event = event
        self._resolved = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(resolved=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Single row: flag · score · tracks · frames
        row = QHBoxLayout()

        flag = QLabel("\u26a0")
        flag.setStyleSheet("font-size: 12px; color: #f48771;")
        flag.setToolTip(event.event_type.value)
        row.addWidget(flag)

        score_label = QLabel(f"{event.score:.2f}")
        score_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #f48771;")
        row.addWidget(score_label)

        track_text = ", ".join(f"T{t}" for t in event.involved_tracks)
        frame_text = f"  frames {event.frame_range[0]}\u2013{event.frame_range[1]}"
        info_label = QLabel(track_text + frame_text)
        info_label.setStyleSheet("font-size: 11px; color: #cccccc;")
        row.addWidget(info_label)
        row.addStretch()
        layout.addLayout(row)

    # ------------------------------------------------------------------

    def mark_resolved(self) -> None:
        """Visually mark this card as resolved."""
        self._resolved = True
        self._apply_style(resolved=True)

    def _apply_style(self, resolved: bool) -> None:
        if resolved:
            self.setStyleSheet(
                "QFrame { background-color: #1a2e1a; border: 1px solid #2a5a2a; "
                "border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: #2d2410; border: 1px solid #6b4f0a; "
                "border-radius: 4px; }"
            )

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit(self.swap_event)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# SuspicionQueueWidget
# ---------------------------------------------------------------------------


class SuspicionQueueWidget(QWidget):
    """Scrollable flat list of suspicion event cards, sorted by score."""

    event_selected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._events: List[SuspicionEvent] = []
        self._cards: List[_EventCard] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Suspicion Queue")
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px; color: #9cdcfe;"
        )
        outer.addWidget(header)

        # Indeterminate progress bar shown while scoring runs in background.
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFormat("Scoring\u2026")
        self._progress.setFixedHeight(16)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        # Event count label shown after scoring completes.
        self._count_lbl = QLabel("")
        self._count_lbl.setStyleSheet(
            "color: #6a9955; font-size: 11px; padding: 2px 4px;"
        )
        self._count_lbl.setVisible(False)
        outer.addWidget(self._count_lbl)

        # Scroll area holding the cards.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_content = QWidget()
        self._card_layout = QVBoxLayout(self._scroll_content)
        self._card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._card_layout.setSpacing(4)
        self._card_layout.setContentsMargins(4, 4, 4, 4)
        self._scroll.setWidget(self._scroll_content)
        outer.addWidget(self._scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_scoring_progress(self) -> None:
        """Show the indeterminate progress bar while scoring runs."""
        self.clear()
        self._progress.setVisible(True)
        self._count_lbl.setVisible(False)

    def hide_scoring_progress(self) -> None:
        """Hide the progress bar once scoring is complete."""
        self._progress.setVisible(False)

    def populate(self, events: List[SuspicionEvent]) -> None:
        """Replace the card list with *events* (already sorted by score)."""
        self.clear()
        self._events = list(events)
        for ev in self._events:
            card = _EventCard(ev)
            card.clicked.connect(self._on_card_clicked)
            self._card_layout.addWidget(card)
            self._cards.append(card)

        n = len(self._events)
        if n:
            self._count_lbl.setText(f"{n} event{'s' if n != 1 else ''} found")
        else:
            self._count_lbl.setText("No suspicious events found \u2714")
        self._count_lbl.setVisible(True)

    def clear(self) -> None:
        """Remove all cards."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._events.clear()

    def mark_resolved(self, event: SuspicionEvent) -> None:
        """Find the card matching *event* and mark it resolved."""
        for card in self._cards:
            if (
                card.swap_event.involved_tracks == event.involved_tracks
                and card.swap_event.frame_peak == event.frame_peak
            ):
                card.mark_resolved()
                break

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_card_clicked(self, event: SuspicionEvent) -> None:
        self.event_selected.emit(event)
