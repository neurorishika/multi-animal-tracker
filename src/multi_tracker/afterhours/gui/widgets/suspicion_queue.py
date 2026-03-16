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
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.event_types import EventType, SuspicionEvent

# ---------------------------------------------------------------------------
# Event-type display helpers
# ---------------------------------------------------------------------------

# Short label shown inline on every card
_EVENT_ABBREV = {
    EventType.SWAP: ("SWAP", "#4fc3f7"),  # cyan
    EventType.FLICKER: ("FLKR", "#f48fb1"),  # pink
    EventType.FRAGMENTATION: ("FRAG", "#ffb74d"),  # orange
    EventType.ABSORPTION: ("ABS", "#fff176"),  # yellow
    EventType.PHANTOM: ("PHNT", "#ce93d8"),  # purple
    EventType.MULTI_SHUFFLE: ("SHUF", "#80cbc4"),  # teal
    EventType.MANUAL: ("EDIT", "#9e9e9e"),  # grey
}


def _score_tier(score: float) -> str:
    """Return a CSS border-left color string for a given score."""
    if score >= 0.70:
        return "#e05000"  # high urgency — orange-red
    if score >= 0.40:
        return "#c8a000"  # medium urgency — amber
    return "#4a6a9a"  # low urgency — steel blue


# ---------------------------------------------------------------------------
# _EventCard
# ---------------------------------------------------------------------------


class _EventCard(QFrame):
    """Visual card representing a single :class:`SuspicionEvent`."""

    clicked = Signal(object)

    def __init__(self, event: SuspicionEvent, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.swap_event = event
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(resolved=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Determine score tier colour and event abbreviation
        abbrev, type_color = _EVENT_ABBREV.get(
            event.event_type, (event.event_type.value.upper()[:4], "#9e9e9e")
        )
        tier_color = _score_tier(event.score)

        # Single row: type badge · score · tracks · frames
        row = QHBoxLayout()

        type_badge = QLabel(abbrev)
        type_badge.setStyleSheet(
            f"font-size: 10px; font-weight: bold; color: {type_color}; "
            f"border: 1px solid {type_color}; border-radius: 3px; "
            f"padding: 1px 4px;"
        )
        type_badge.setToolTip(event.event_type.value)
        row.addWidget(type_badge)

        score_label = QLabel(f"{event.score:.2f}")
        score_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {tier_color};"
        )
        row.addWidget(score_label)

        track_text = ", ".join(f"T{t}" for t in event.involved_tracks)
        frame_text = f"  frames {event.frame_range[0]}\u2013{event.frame_range[1]}"
        info_label = QLabel(track_text + frame_text)
        info_label.setStyleSheet("font-size: 11px; color: #cccccc;")
        row.addWidget(info_label)
        row.addStretch()
        layout.addLayout(row)

        self._tier_color = tier_color

    # ------------------------------------------------------------------

    def mark_resolved(self) -> None:
        """Visually mark this card as resolved."""
        self._apply_style(resolved=True)

    def _apply_style(self, resolved: bool) -> None:
        if resolved:
            self.setStyleSheet(
                "QFrame { background-color: #1a2e1a; border: 1px solid #2a5a2a; "
                "border-left: 3px solid #2a5a2a; border-radius: 4px; }"
            )
        else:
            tier = getattr(self, "_tier_color", "#6b4f0a")
            self.setStyleSheet(
                f"QFrame {{ background-color: #1e1e1e; "
                f"border: 1px solid #3e3e42; "
                f"border-left: 3px solid {tier}; "
                f"border-radius: 4px; }}"
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
    rescore_all_requested = Signal()
    merge_wizard_requested = Signal()

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

        # Rescore All button — visible after edits so the user can trigger
        # a full rescore when they are ready.
        self._rescore_btn = QPushButton("Rescore All")
        self._rescore_btn.setToolTip(
            "Run a full rescore of the entire video.\n"
            "Use this after finishing a batch of edits."
        )
        self._rescore_btn.setVisible(False)
        self._rescore_btn.clicked.connect(self.rescore_all_requested.emit)
        outer.addWidget(self._rescore_btn)

        # Merge Wizard button — lets user re-run the fragment merge wizard.
        self._merge_wizard_btn = QPushButton("Merge Wizard")
        self._merge_wizard_btn.setToolTip(
            "Open the fragment merge wizard to stitch broken tracks."
        )
        self._merge_wizard_btn.setVisible(False)
        self._merge_wizard_btn.clicked.connect(self.merge_wizard_requested.emit)
        outer.addWidget(self._merge_wizard_btn)

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
        self._count_lbl.setVisible(False)

    def mark_resolved(self, event: SuspicionEvent) -> None:
        """Find the card matching *event* and mark it resolved."""
        for card in self._cards:
            if (
                card.swap_event.involved_tracks == event.involved_tracks
                and card.swap_event.frame_peak == event.frame_peak
            ):
                card.mark_resolved()
                break

    def remove_events_for_tracks(
        self,
        tracks: List[int],
        frame_range: Optional[tuple] = None,
    ) -> None:
        """Remove cards whose involved tracks overlap *tracks*.

        If *frame_range* ``(start, end)`` is given, only events whose peak
        falls within that window are removed.
        """
        track_set = set(tracks)
        to_remove: List[_EventCard] = []
        for card in self._cards:
            ev = card.swap_event
            if not (set(ev.involved_tracks) & track_set):
                continue
            if frame_range is not None:
                f_start, f_end = frame_range
                if not (f_start <= ev.frame_peak <= f_end):
                    continue
            to_remove.append(card)

        for card in to_remove:
            self._cards.remove(card)
            self._events.remove(card.swap_event)
            card.setParent(None)
            card.deleteLater()

        self._update_count_label()

    def add_events(self, events: List[SuspicionEvent]) -> None:
        """Insert new events into the queue, maintaining descending score order."""
        for ev in events:
            self._events.append(ev)

        # Re-sort all events by score descending
        self._events.sort(key=lambda e: e.score, reverse=True)

        # Rebuild the card layout in sorted order
        for card in self._cards:
            self._card_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

        for ev in self._events:
            card = _EventCard(ev)
            card.clicked.connect(self._on_card_clicked)
            self._card_layout.addWidget(card)
            self._cards.append(card)

        self._update_count_label()

    def show_rescore_button(self, visible: bool = True) -> None:
        """Show or hide the 'Rescore All' button."""
        self._rescore_btn.setVisible(visible)

    def show_merge_wizard_button(self, visible: bool = True) -> None:
        """Show or hide the 'Merge Wizard' button."""
        self._merge_wizard_btn.setVisible(visible)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_count_label(self) -> None:
        n = len(self._events)
        if n:
            self._count_lbl.setText(f"{n} event{'s' if n != 1 else ''} found")
        else:
            self._count_lbl.setText("No suspicious events found \u2714")
        self._count_lbl.setVisible(True)

    def _on_card_clicked(self, event: SuspicionEvent) -> None:
        self.event_selected.emit(event)
