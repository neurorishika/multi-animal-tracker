"""Runtime configuration schema for DetectKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectKitConfig:
    """Session-meaningful state for the DetectKit labeling application."""

    last_project_path: str = ""
    compute_runtime: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a plain dictionary suitable for JSON persistence."""
        return {
            "last_project_path": self.last_project_path,
            "compute_runtime": self.compute_runtime,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectKitConfig:
        """Deserialize a ``DetectKitConfig`` from a plain dictionary, using defaults for missing keys."""
        return cls(
            last_project_path=data.get("last_project_path", ""),
            compute_runtime=data.get("compute_runtime", "cpu"),
        )
