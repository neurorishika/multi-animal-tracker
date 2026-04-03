"""DetectKit application entry point."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication


def main() -> None:
    """Launch the DetectKit application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("DetectKit")
    app.setApplicationDisplayName("DetectKit")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("detectkit")

    try:
        from multi_tracker.paths import get_brand_qicon

        icon = get_brand_qicon("detectkit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    from multi_tracker.detectkit.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1600, 1000)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
