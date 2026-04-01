import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def main():
    ap = argparse.ArgumentParser(
        description="MAT Afterhours — Interactive Identity Proofreading"
    )
    ap.add_argument("video", nargs="?", default=None, help="Path to video file to open")
    args, qt_args = ap.parse_known_args()

    app = QApplication(qt_args)
    app.setApplicationName("MATAfterhours")
    app.setApplicationDisplayName("MAT Afterhours")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("NeuroRishika")
    app.setDesktopFileName("mat-afterhours")

    try:
        from multi_tracker.paths import get_brand_qicon

        icon = get_brand_qicon("multianimaltrackerafterhours.svg")
        if icon and not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    from .gui.main_window import MainWindow

    window = MainWindow()

    if args.video:
        video = str(Path(args.video).resolve())
        window._sessions = [video]
        window._session_idx = 0
        window._open_current_session()

    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
