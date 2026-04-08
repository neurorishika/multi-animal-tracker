"""FilterKit GUI package."""

from .main_window import FilterKitWindow


def main():
    """Backward-compatible launcher for older gui-based entry targets."""
    from hydra_suite.filterkit.app import main as app_main

    return app_main()


__all__ = ["FilterKitWindow", "main"]
