import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .gui.mainwindow import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ClassKitLabeler")
    app.setApplicationDisplayName("ClassKit Labeler")
    window = MainWindow()
    window.resize(1600, 1000)
    window.showMaximized()
    QTimer.singleShot(0, window.show_startup_overlay)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
