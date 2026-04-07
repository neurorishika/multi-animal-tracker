"""Hydra-aware file picker wrappers.

Drop-in replacements for the ``QFileDialog`` static methods that automatically
inject HYDRA platform directories as sidebar bookmarks and apply consistent
non-native dialog styling on Linux.

Usage
-----
In any module that uses ``QFileDialog``, add **one line** after the PySide6
import block::

    from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog  # noqa: F811

All existing ``QFileDialog.get*`` call sites continue to work unchanged.
The sidebar will show quick-access shortcuts to every HYDRA platform directory
plus the standard OS shortcuts (Home, Desktop, Documents, Downloads).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QFileDialog, QWidget

# ---------------------------------------------------------------------------
# Sidebar URL builder
# ---------------------------------------------------------------------------


def _sidebar_urls() -> list[QUrl]:
    """Build the sidebar bookmark list with all HYDRA dirs + OS standard locations."""
    from hydra_suite.paths import (
        get_config_dir,
        get_data_dir,
        get_models_dir,
        get_presets_dir,
        get_projects_dir,
        get_training_runs_dir,
    )

    candidates: list[Path] = []

    # HYDRA dirs first (most relevant)
    for getter in (
        get_projects_dir,
        get_data_dir,
        get_models_dir,
        get_training_runs_dir,
        get_config_dir,
        get_presets_dir,
    ):
        try:
            candidates.append(getter())
        except Exception:
            pass

    # Standard OS locations
    for rel in ("", "Desktop", "Documents", "Downloads"):
        candidates.append(Path.home() / rel if rel else Path.home())

    seen: set[str] = set()
    urls: list[QUrl] = []
    for p in candidates:
        try:
            key = str(Path(p).resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            urls.append(QUrl.fromLocalFile(str(p)))
    return urls


def _configure(dlg: QFileDialog) -> None:
    """Apply sidebar bookmarks and Linux non-native flag."""
    dlg.setSidebarUrls(_sidebar_urls())
    if sys.platform not in ("darwin", "win32"):
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)


# ---------------------------------------------------------------------------
# Drop-in subclass
# ---------------------------------------------------------------------------


class HydraFileDialog(QFileDialog):
    """Drop-in for ``QFileDialog`` with HYDRA sidebar shortcuts on all platforms.

    The class overrides all four static picker methods so that any module that
    imports::

        from hydra_suite.utils.file_dialogs import HydraFileDialog as QFileDialog

    gets HYDRA sidebar bookmarks automatically without changing a single call
    site.  All enum/flag attributes (``ShowDirsOnly``, ``FileMode``, etc.) are
    inherited from ``QFileDialog``.
    """

    # ------------------------------------------------------------------
    # getExistingDirectory
    # ------------------------------------------------------------------

    @classmethod
    def getExistingDirectory(  # type: ignore[override]
        cls,
        parent: QWidget | None = None,
        caption: str = "",
        dir: str = "",
        options: QFileDialog.Options | None = None,
    ) -> str:
        dlg = QFileDialog(parent, caption, dir)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        if options is not None:
            dlg.setOptions(options)
        _configure(dlg)
        if dlg.exec():
            files = dlg.selectedFiles()
            return files[0] if files else ""
        return ""

    # ------------------------------------------------------------------
    # getOpenFileName
    # ------------------------------------------------------------------

    @classmethod
    def getOpenFileName(  # type: ignore[override]
        cls,
        parent: QWidget | None = None,
        caption: str = "",
        dir: str = "",
        filter: str = "",
        initialFilter: str = "",
        options: QFileDialog.Options | None = None,
    ) -> tuple[str, str]:
        dlg = QFileDialog(parent, caption, dir)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
        if filter:
            if ";;" in filter:
                dlg.setNameFilters(filter.split(";;"))
            else:
                dlg.setNameFilter(filter)
        if initialFilter:
            dlg.selectNameFilter(initialFilter)
        if options is not None:
            dlg.setOptions(options)
        _configure(dlg)
        if dlg.exec():
            files = dlg.selectedFiles()
            return (files[0] if files else ""), dlg.selectedNameFilter()
        return "", ""

    # ------------------------------------------------------------------
    # getOpenFileNames
    # ------------------------------------------------------------------

    @classmethod
    def getOpenFileNames(  # type: ignore[override]
        cls,
        parent: QWidget | None = None,
        caption: str = "",
        dir: str = "",
        filter: str = "",
        initialFilter: str = "",
        options: QFileDialog.Options | None = None,
    ) -> tuple[list[str], str]:
        dlg = QFileDialog(parent, caption, dir)
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        if filter:
            if ";;" in filter:
                dlg.setNameFilters(filter.split(";;"))
            else:
                dlg.setNameFilter(filter)
        if initialFilter:
            dlg.selectNameFilter(initialFilter)
        if options is not None:
            dlg.setOptions(options)
        _configure(dlg)
        if dlg.exec():
            return dlg.selectedFiles(), dlg.selectedNameFilter()
        return [], ""

    # ------------------------------------------------------------------
    # getSaveFileName
    # ------------------------------------------------------------------

    @classmethod
    def getSaveFileName(  # type: ignore[override]
        cls,
        parent: QWidget | None = None,
        caption: str = "",
        dir: str = "",
        filter: str = "",
        initialFilter: str = "",
        options: QFileDialog.Options | None = None,
    ) -> tuple[str, str]:
        dlg = QFileDialog(parent, caption, dir)
        dlg.setFileMode(QFileDialog.FileMode.AnyFile)
        dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if filter:
            if ";;" in filter:
                dlg.setNameFilters(filter.split(";;"))
            else:
                dlg.setNameFilter(filter)
        if initialFilter:
            dlg.selectNameFilter(initialFilter)
        if options is not None:
            dlg.setOptions(options)
        _configure(dlg)
        if dlg.exec():
            files = dlg.selectedFiles()
            return (files[0] if files else ""), dlg.selectedNameFilter()
        return "", ""
