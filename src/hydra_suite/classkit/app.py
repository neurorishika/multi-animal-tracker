"""ClassKit application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def main():
    """Launch the ClassKit application, showing the main window maximized."""
    app = QApplication(sys.argv)
    app.setApplicationName("ClassKit")
    app.setApplicationDisplayName("ClassKit")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("classkit")

    try:
        from hydra_suite.paths import get_brand_qicon

        icon = get_brand_qicon("classkit.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    window = MainWindow()
    try:
        window.setWindowIcon(app.windowIcon())
    except Exception:
        pass
    window.resize(1600, 1000)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
