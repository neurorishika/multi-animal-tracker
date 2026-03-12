"""Event type taxonomy for MAT-afterhours.

Defines the :class:`EventType` enum and the generalised
:class:`SuspicionEvent` dataclass.  All detectors produce
``SuspicionEvent`` instances; the old ``SwapSuspicionEvent`` is
superseded by this module.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class EventType(enum.Enum):
    """Classification of a suspicious tracking event."""

    SWAP = "swap"
    """Pairwise identity swap — two tracks exchange IDs."""

    FLICKER = "flicker"
    """Swap + immediate swap-back within a short window."""

    FRAGMENTATION = "fragmentation"
    """One animal's trajectory is split across multiple IDs."""

    ABSORPTION = "absorption"
    """Two tracks merge into one (missed detection on one animal)."""

    PHANTOM = "phantom"
    """Short-lived noise track — likely a false-positive detection."""

    MULTI_SHUFFLE = "multi_shuffle"
    """3+ way identity shuffle in a crowded interaction."""

    MANUAL = "manual"
    """User-initiated manual region review (not auto-detected)."""


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

#: Human-readable label for each type.
EVENT_TYPE_LABEL: Dict[EventType, str] = {
    EventType.SWAP: "Swap",
    EventType.FLICKER: "Flicker",
    EventType.FRAGMENTATION: "Fragment",
    EventType.ABSORPTION: "Absorption",
    EventType.PHANTOM: "Phantom",
    EventType.MULTI_SHUFFLE: "Multi-Shuffle",
    EventType.MANUAL: "Manual",
}

#: Badge colour for each type (dark-theme-safe hex).
EVENT_TYPE_COLOR: Dict[EventType, str] = {
    EventType.SWAP: "#f48771",  # warm red-orange
    EventType.FLICKER: "#dcdcaa",  # muted yellow
    EventType.FRAGMENTATION: "#9cdcfe",  # light blue
    EventType.ABSORPTION: "#c586c0",  # purple
    EventType.PHANTOM: "#6a9955",  # dim green
    EventType.MULTI_SHUFFLE: "#ce9178",  # peach
    EventType.MANUAL: "#808080",  # grey
}

#: Default suggested human-readable action for each type.
EVENT_TYPE_SUGGESTED_ACTION: Dict[EventType, str] = {
    EventType.SWAP: "Swap IDs at frame",
    EventType.FLICKER: "Erase flicker (undo both swaps)",
    EventType.FRAGMENTATION: "Merge fragments into one ID",
    EventType.ABSORPTION: "Split + reassign at separation",
    EventType.PHANTOM: "Delete phantom track",
    EventType.MULTI_SHUFFLE: "Reassign ID chain",
    EventType.MANUAL: "Review and correct manually",
}


# ---------------------------------------------------------------------------
# SuspicionEvent
# ---------------------------------------------------------------------------


@dataclass
class SuspicionEvent:
    """A generalised suspicious tracking event.

    Replaces the old ``SwapSuspicionEvent`` with richer metadata.

    Attributes
    ----------
    event_type:
        Automatic classification (can be overridden by the user).
    involved_tracks:
        All trajectory IDs that participate in this event.
    frame_peak:
        Frame index where the suspicion is strongest.
    frame_range:
        Inclusive ``(start, end)`` frame range of the event.
    score:
        Combined suspicion score in ``[0, 1]``.
    signals:
        List of signal codes that fired (e.g. ``["Cr", "Hd"]``).
    region_label:
        Density region label at the peak location, or ``"open_field"``.
    region_boundary:
        Whether the peak falls on a region temporal boundary.
    suggested_action:
        Human-readable hint for the default resolution.
    """

    event_type: EventType
    involved_tracks: List[int]
    frame_peak: int
    frame_range: Tuple[int, int]
    score: float
    signals: List[str] = field(default_factory=list)
    region_label: str = "open_field"
    region_boundary: bool = False
    suggested_action: str = ""

    # ----- Convenience properties -----

    @property
    def track_a(self) -> int:
        """First involved track (always present)."""
        return self.involved_tracks[0]

    @property
    def track_b(self) -> Optional[int]:
        """Second involved track, or ``None`` for single-track events."""
        return self.involved_tracks[1] if len(self.involved_tracks) > 1 else None

    def __post_init__(self) -> None:
        if not self.suggested_action:
            self.suggested_action = EVENT_TYPE_SUGGESTED_ACTION.get(self.event_type, "")
