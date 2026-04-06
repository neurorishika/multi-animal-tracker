"""CurrentPageStackedWidget — stacked widget whose size hint tracks the active page."""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QStackedWidget


class CurrentPageStackedWidget(QStackedWidget):
    """A stacked widget whose size hint tracks the active page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def sizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            hint = current.sizeHint()
            if hint.isValid():
                return hint
        return super().sizeHint()

    def minimumSizeHint(self) -> QSize:
        current = self.currentWidget()
        if current is not None:
            hint = current.minimumSizeHint()
            if hint.isValid():
                return hint
        return super().minimumSizeHint()
