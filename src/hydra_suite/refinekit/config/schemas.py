"""Runtime configuration schema for RefineKit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RefineKitConfig:
    """Session-meaningful state for the RefineKit proofreading application."""

    last_video_path: str = ""
    sessions: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dict with ``last_video_path`` and ``sessions``."""
        return {
            "last_video_path": self.last_video_path,
            "sessions": list(self.sessions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefineKitConfig":
        """Reconstruct a config instance from a plain dict, applying defaults for missing keys."""
        return cls(
            last_video_path=data.get("last_video_path", ""),
            sessions=list(data.get("sessions", [])),
        )
