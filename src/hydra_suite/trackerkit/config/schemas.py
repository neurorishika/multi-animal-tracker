"""Runtime configuration schema for the MAT tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrackerConfig:
    """Session-meaningful state for the MAT tracking application.

    Only persistent, user-configurable fields live here.
    Ephemeral runtime state (ROI masks, playback position, session
    counters, etc.) stays on MainWindow.
    """

    # --- Input ---
    current_video_path: str = ""
    batch_videos: list = field(default_factory=list)

    # --- ROI ---
    roi_shapes: list = field(default_factory=list)
    roi_current_mode: str = "circle"  # 'circle' or 'polygon'
    roi_current_zone_type: str = "include"  # 'include' or 'exclude'

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "current_video_path": self.current_video_path,
            "batch_videos": list(self.batch_videos),
            "roi_shapes": list(self.roi_shapes),
            "roi_current_mode": self.roi_current_mode,
            "roi_current_zone_type": self.roi_current_zone_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackerConfig:
        """Deserialize from a dict produced by ``to_dict``."""
        return cls(
            current_video_path=data.get("current_video_path", ""),
            batch_videos=list(data.get("batch_videos", [])),
            roi_shapes=list(data.get("roi_shapes", [])),
            roi_current_mode=data.get("roi_current_mode", "circle"),
            roi_current_zone_type=data.get("roi_current_zone_type", "include"),
        )
