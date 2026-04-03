"""Persistent recent-items store for hydra-suite applications."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_RECENT = 20


class RecentItemsStore:
    """JSON-backed recent-items list for a specific app.

    Storage location respects ``HYDRA_DATA_DIR`` via ``hydra_suite.paths``.
    """

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name

    def _json_path(self) -> Path:
        from hydra_suite.paths import _user_data_dir

        return _user_data_dir() / self._app_name / "recents.json"

    def load(self) -> list[str]:
        """Return recent items, most-recent first."""
        p = self._json_path()
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            logger.debug("Failed to read recents for %s", self._app_name, exc_info=True)
        return []

    def add(self, path: str) -> None:
        """Add *path* to the top, de-duplicating and trimming to max."""
        items = self.load()
        items = [x for x in items if x != path]
        items.insert(0, path)
        self._save(items)

    def remove(self, path: str) -> None:
        """Remove *path* from the list."""
        items = self.load()
        items = [x for x in items if x != path]
        self._save(items)

    def clear(self) -> None:
        """Remove all recent items."""
        self._save([])

    def _save(self, items: list[str]) -> None:
        p = self._json_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(items[:_MAX_RECENT], indent=2), encoding="utf-8")
