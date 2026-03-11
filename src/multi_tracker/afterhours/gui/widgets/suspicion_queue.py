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
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.event_types import (
    EVENT_TYPE_COLOR,
    EVENT_TYPE_LABEL,
    EventType,
    SuspicionEvent,
)

# Tiered score thresholds — each "Show more" click reveals events down to the
# next threshold.
_TIER_THRESHOLDS: List[float] = [0.6, 0.4, 0.25, 0.15, 0.0]


# ---------------------------------------------------------------------------
# _EventCard
# ---------------------------------------------------------------------------


class _EventCard(QFrame):
    """Visual card representing a single :class:`SuspicionEvent`."""

    clicked = Signal(object)
    reclassified = Signal(object, object)  # (event, new_EventType)

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

        # Top row: type badge + score + signal codes
        top = QHBoxLayout()

        # Event type badge
        color = EVENT_TYPE_COLOR.get(event.event_type, "#cccccc")
        label_text = EVENT_TYPE_LABEL.get(event.event_type, "?")
        self._badge = QLabel(label_text)
        self._badge.setStyleSheet(
            f"background-color: {color}; color: #1e1e1e; font-weight: bold; "
            f"padding: 1px 6px; border-radius: 3px; font-size: 10px;"
        )
        top.addWidget(self._badge)

        score_label = QLabel(f"{event.score:.2f}")
        score_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #f48771;")
        top.addWidget(score_label)

        signals_text = "+".join(event.signals) if event.signals else "—"
        signals_label = QLabel(signals_text)
        signals_label.setStyleSheet("color: #9cdcfe; font-size: 12px;")
        top.addWidget(signals_label)
        top.addStretch()
        layout.addLayout(top)

        # Middle row: track IDs + frame range
        track_text = "Tracks " + ", ".join(str(t) for t in event.involved_tracks)
        frame_text = f"  frames {event.frame_range[0]}–{event.frame_range[1]}"
        mid_label = QLabel(track_text + frame_text)
        mid_label.setStyleSheet("font-size: 11px; color: #cccccc;")
        layout.addWidget(mid_label)

        # Bottom row: reclassify combo
        reclass_row = QHBoxLayout()
        reclass_row.addWidget(QLabel("Type:"))
        self._reclass_combo = QComboBox()
        for et in EventType:
            self._reclass_combo.addItem(EVENT_TYPE_LABEL[et], et)
        self._reclass_combo.setCurrentIndex(list(EventType).index(event.event_type))
        self._reclass_combo.currentIndexChanged.connect(self._on_reclassify)
        reclass_row.addWidget(self._reclass_combo, stretch=1)
        layout.addLayout(reclass_row)

    # ------------------------------------------------------------------

    def mark_resolved(self) -> None:
        """Visually mark this card as resolved."""
        self._resolved = True
        self._apply_style(resolved=True)

    def _on_reclassify(self, index: int) -> None:
        new_type = self._reclass_combo.itemData(index)
        if new_type is None or new_type == self.swap_event.event_type:
            return
        old_event = self.swap_event
        self.swap_event = SuspicionEvent(
            event_type=new_type,
            involved_tracks=old_event.involved_tracks,
            frame_peak=old_event.frame_peak,
            frame_range=old_event.frame_range,
            score=old_event.score,
            signals=old_event.signals,
            region_label=old_event.region_label,
            region_boundary=old_event.region_boundary,
        )
        # Update badge
        color = EVENT_TYPE_COLOR.get(new_type, "#cccccc")
        label_text = EVENT_TYPE_LABEL.get(new_type, "?")
        self._badge.setText(label_text)
        self._badge.setStyleSheet(
            f"background-color: {color}; color: #1e1e1e; font-weight: bold; "
            f"padding: 1px 6px; border-radius: 3px; font-size: 10px;"
        )
        self.reclassified.emit(old_event, new_type)

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
    """Scrollable, tiered list of suspicion event cards."""

    event_selected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._events: List[SuspicionEvent] = []
        self._cards: List[_EventCard] = []
        self._current_tier: int = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Suspicion Queue")
        header.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 4px; color: #9cdcfe;"
        )
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

    def populate(self, events: List[SuspicionEvent]) -> None:
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

    def _on_card_clicked(self, event: SuspicionEvent) -> None:
        self.event_selected.emit(event)
