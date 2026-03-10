import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("MATAfterhours")
    app.setApplicationDisplayName("MAT Afterhours")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("mat-afterhours")

    try:
        icon_path = (
            Path(__file__).resolve().parents[2]
            / "brand"
            / "multianimaltrackerafterhours.svg"
        )
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass

    from .gui.main_window import MainWindow

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
