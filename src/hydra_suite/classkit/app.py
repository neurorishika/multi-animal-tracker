import sys

from PySide6.QtWidgets import QApplication

from .gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ClassKitLabeler")
    app.setApplicationDisplayName("ClassKit Labeler")
    window = MainWindow()
    window.resize(1600, 1000)
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
