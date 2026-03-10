"""Suspicion queue widget for MAT-afterhours.

Displays a scrollable list of :class:`SwapSuspicionEvent` cards organised
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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.swap_scorer import SwapSuspicionEvent

# Tiered score thresholds — each "Show more" click reveals events down to the
# next threshold.
_TIER_THRESHOLDS: List[float] = [0.6, 0.4, 0.25, 0.15, 0.0]


# ---------------------------------------------------------------------------
# _EventCard
# ---------------------------------------------------------------------------


class _EventCard(QFrame):
    """Visual card representing a single :class:`SwapSuspicionEvent`."""

    clicked = Signal(object)

    def __init__(self, event: SwapSuspicionEvent, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.event = event
        self._resolved = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(resolved=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Top row: score badge + signal codes
        top = QHBoxLayout()
        score_label = QLabel(f"{event.score:.2f}")
        score_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #d32f2f;")
        top.addWidget(score_label)

        signals_text = "+".join(event.signals) if event.signals else "—"
        signals_label = QLabel(signals_text)
        signals_label.setStyleSheet("color: #555; font-size: 12px;")
        top.addWidget(signals_label)
        top.addStretch()
        layout.addLayout(top)

        # Middle row: track IDs + frame range
        track_text = f"Tracks {event.track_a}"
        if event.track_b is not None:
            track_text += f" & {event.track_b}"
        frame_text = f"  frames {event.frame_range[0]}–{event.frame_range[1]}"
        mid_label = QLabel(track_text + frame_text)
        mid_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(mid_label)

        # Bottom row: region label
        region_text = event.region_label
        if event.region_boundary:
            region_text += " (boundary)"
        region_label = QLabel(region_text)
        region_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(region_label)

    # ------------------------------------------------------------------

    def mark_resolved(self) -> None:
        """Visually mark this card as resolved."""
        self._resolved = True
        self._apply_style(resolved=True)

    def _apply_style(self, resolved: bool) -> None:
        if resolved:
            self.setStyleSheet(
                "QFrame { background-color: #e8f5e9; border: 1px solid #a5d6a7; "
                "border-radius: 4px; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: #fff3e0; border: 1px solid #ffe0b2; "
                "border-radius: 4px; }"
            )

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit(self.event)
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# SuspicionQueueWidget
# ---------------------------------------------------------------------------


class SuspicionQueueWidget(QWidget):
    """Scrollable, tiered list of swap suspicion event cards."""

    event_selected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._events: List[SwapSuspicionEvent] = []
        self._cards: List[_EventCard] = []
        self._current_tier: int = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Suspicion Queue")
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        outer.addWidget(header)

        # Scroll area
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

        # Show more button
        self._show_more_btn = QPushButton("Show more")
        self._show_more_btn.clicked.connect(self._on_show_more)
        self._show_more_btn.setVisible(False)
        outer.addWidget(self._show_more_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, events: List[SwapSuspicionEvent]) -> None:
        """Load events and render the first tier."""
        self.clear()
        self._events = list(events)
        self._current_tier = 0
        self._render_tier()

    def clear(self) -> None:
        """Remove all cards."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._events.clear()
        self._current_tier = 0
        self._show_more_btn.setVisible(False)

    def mark_resolved(self, event: SwapSuspicionEvent) -> None:
        """Find the card matching *event* and mark it resolved."""
        for card in self._cards:
            if (
                card.event.track_a == event.track_a
                and card.event.track_b == event.track_b
                and card.event.frame_peak == event.frame_peak
            ):
                card.mark_resolved()
                break

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render_tier(self) -> None:
        """Render cards for the current tier threshold and update button."""
        if self._current_tier >= len(_TIER_THRESHOLDS):
            self._show_more_btn.setVisible(False)
            return

        threshold = _TIER_THRESHOLDS[self._current_tier]
        # Next threshold (for filtering already shown)
        prev_threshold = (
            _TIER_THRESHOLDS[self._current_tier - 1]
            if self._current_tier > 0
            else float("inf")
        )

        for ev in self._events:
            # Show events in range (threshold, prev_threshold]
            if ev.score >= threshold and (
                self._current_tier == 0 or ev.score < prev_threshold
            ):
                card = _EventCard(ev)
                card.clicked.connect(self._on_card_clicked)
                self._card_layout.addWidget(card)
                self._cards.append(card)

        # Show "Show more" if there are more tiers
        has_more = self._current_tier + 1 < len(_TIER_THRESHOLDS)
        # Check if there are actually events in lower tiers
        if has_more:
            next_threshold = _TIER_THRESHOLDS[self._current_tier + 1]
            has_lower = any(
                ev.score >= next_threshold and ev.score < threshold
                for ev in self._events
            )
            self._show_more_btn.setVisible(has_lower)
        else:
            self._show_more_btn.setVisible(False)

    def _on_show_more(self) -> None:
        self._current_tier += 1
        self._render_tier()

    def _on_card_clicked(self, event: SwapSuspicionEvent) -> None:
        self.event_selected.emit(event)
