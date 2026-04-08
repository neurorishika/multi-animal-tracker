"""Persistent recent-items store for hydra-suite applications."""

from __future__ import annotations

import json
import logging
import tempfile
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
        from hydra_suite.paths import get_app_data_dir

        return get_app_data_dir(self._app_name) / "recents.json"

    @staticmethod
    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    @classmethod
    def _is_transient_test_path(cls, item: str) -> bool:
        raw = str(item).strip()
        if not raw:
            return True

        try:
            path = Path(raw).expanduser().resolve(strict=False)
        except Exception:
            path = Path(raw).expanduser()

        parts = [part.lower() for part in path.parts]
        if any(
            part.startswith("pytest-of-") or part.startswith("pytest-")
            for part in parts
        ):
            return True

        try:
            temp_root = Path(tempfile.gettempdir()).resolve()
        except Exception:
            temp_root = None

        if temp_root is not None and cls._is_relative_to(path, temp_root):
            name = path.name.lower()
            if name.startswith("test_") or name.endswith("_0"):
                return True

        return False

    @classmethod
    def _sanitize_items(cls, items: list[object]) -> list[str]:
        cleaned: list[str] = []
        seen = set()
        for item in items:
            text = str(item).strip()
            if not text or cls._is_transient_test_path(text) or text in seen:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned[:_MAX_RECENT]

    def load(self) -> list[str]:
        """Return recent items, most-recent first."""
        p = self._json_path()
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items = self._sanitize_items(data)
                if items != [str(x).strip() for x in data]:
                    self._save(items)
                return items
        except Exception:
            logger.debug("Failed to read recents for %s", self._app_name, exc_info=True)
        return []

    def add(self, path: str) -> None:
        """Add *path* to the top, de-duplicating and trimming to max."""
        path = str(path).strip()
        if self._is_transient_test_path(path):
            return
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
        p.write_text(
            json.dumps(self._sanitize_items(items), indent=2),
            encoding="utf-8",
        )
