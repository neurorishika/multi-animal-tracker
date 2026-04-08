"""Tests for FilterKit launcher entry points."""

from __future__ import annotations

from hydra_suite.filterkit import __main__ as package_main
from hydra_suite.filterkit.app import main as app_main
from hydra_suite.filterkit.gui import __main__ as gui_module_main
from hydra_suite.filterkit.gui import main as gui_main
from hydra_suite.launcher.app import APP_CATALOG


def test_filterkit_launcher_catalog_uses_app_entrypoint() -> None:
    filterkit_entry = next(
        item["entry"] for item in APP_CATALOG if item["name"] == "FilterKit"
    )

    assert filterkit_entry == "hydra_suite.filterkit.app:main"


def test_filterkit_legacy_gui_entrypoints_delegate_to_app_main() -> None:
    assert gui_main is not None
    assert package_main.main is app_main
    assert gui_module_main.main is app_main
