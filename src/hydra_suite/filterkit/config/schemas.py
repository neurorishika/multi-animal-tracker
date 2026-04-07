"""Runtime configuration schema for FilterKit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FilterKitConfig:
    """User-configurable preferences for the FilterKit sieve application."""

    dataset_path: str = ""
    images_per_page: int = 200
    removed_per_page: int = 150

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "images_per_page": self.images_per_page,
            "removed_per_page": self.removed_per_page,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterKitConfig:
        return cls(
            dataset_path=data.get("dataset_path", ""),
            images_per_page=data.get("images_per_page", 200),
            removed_per_page=data.get("removed_per_page", 150),
        )
