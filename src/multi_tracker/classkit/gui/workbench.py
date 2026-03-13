from PySide6.QtCore import Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .widgets.image_viewer import ImageCanvas


class LabelWorkbench(QWidget):
    """
    Main labeling interface.
    Left: Large Image Canvas
    Right/Bottom: Class buttons and controls.
    """

    # Signals to communicate with MainWindow
    next_image = Signal()
    prev_image = Signal()
    assign_label = Signal(str)

    def __init__(self, classes=None):
        super().__init__()
        self.classes = classes or ["class_1", "class_2"]

        self.layout = QHBoxLayout(self)

        # 1. Image Viewer (Center/Left)
        self.viewer = ImageCanvas()
        self.layout.addWidget(self.viewer, 3)

        # 2. Controls (Right Sidebar)
        self.sidebar = QWidget()
        self.sidebar.setFixedWidth(250)
        self.side_layout = QVBoxLayout(self.sidebar)

        # Class Buttons Grid
        self.class_group = QWidget()
        self.class_grid = QGridLayout(self.class_group)

        self.setup_class_buttons()
        self.side_layout.addWidget(QLabel("<b>Classes (1-9)</b>"))
        self.side_layout.addWidget(self.class_group)

        # Navigation
        self.nav_group = QWidget()
        self.nav_layout = QHBoxLayout(self.nav_group)

        self.btn_prev = QPushButton("← Prev (A)")
        self.btn_next = QPushButton("Next (D) →")

        self.btn_prev.clicked.connect(self.prev_image.emit)
        self.btn_next.clicked.connect(self.next_image.emit)

        self.nav_layout.addWidget(self.btn_prev)
        self.nav_layout.addWidget(self.btn_next)

        self.side_layout.addStretch()
        self.side_layout.addWidget(self.nav_group)

        self.layout.addWidget(self.sidebar, 1)

        self.setup_shortcuts()

    def setup_class_buttons(self):
        # Clear existing
        for i in reversed(range(self.class_grid.count())):
            self.class_grid.itemAt(i).widget().setParent(None)

        for i, cls_name in enumerate(self.classes):
            btn = QPushButton(f"{i + 1}: {cls_name}")
            btn.setStyleSheet("text-align: left; padding: 5px;")
            # Capture variable in lambda
            btn.clicked.connect(
                lambda checked=False, c=cls_name: self.assign_label.emit(c)
            )

            row = i // 2
            col = i % 2
            self.class_grid.addWidget(btn, row, col)

    def setup_shortcuts(self):
        # Number keys 1-9
        for i in range(1, 10):
            if i <= len(self.classes):
                cls_name = self.classes[i - 1]
                QShortcut(QKeySequence(f"{i}"), self).activated.connect(
                    lambda c=cls_name: self.assign_label.emit(c)
                )

        # Navigation
        QShortcut(QKeySequence("A"), self).activated.connect(self.prev_image.emit)
        QShortcut(QKeySequence("D"), self).activated.connect(self.next_image.emit)

    def load_image(self, path):
        self.viewer.set_image(path)

    def set_classes(self, classes):
        self.classes = classes
        self.setup_class_buttons()
        self.setup_shortcuts()
