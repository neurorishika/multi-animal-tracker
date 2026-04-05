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
        return {
            "last_video_path": self.last_video_path,
            "sessions": list(self.sessions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefineKitConfig":
        return cls(
            last_video_path=data.get("last_video_path", ""),
            sessions=list(data.get("sessions", [])),
        )
