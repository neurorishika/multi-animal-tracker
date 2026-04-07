"""Verify DetectKit package structure and imports."""

from __future__ import annotations


def test_detectkit_package_imports():
    from hydra_suite.detectkit.gui.constants import (
        DEFAULT_PROJECT_FILENAME,
        IMG_EXTS,
        OBB_LABEL_FIELDS,
    )

    assert OBB_LABEL_FIELDS == 9
    assert ".jpg" in IMG_EXTS
    assert DEFAULT_PROJECT_FILENAME == "detectkit_project.json"
