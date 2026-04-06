"""ImmediateTooltipButton — tool button that shows tooltip instantly on hover."""

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QToolButton, QToolTip


class ImmediateTooltipButton(QToolButton):
    """Tool button that shows its tooltip immediately on hover, focus, or click."""

    def _show_tooltip_now(self) -> None:
        tooltip = self.toolTip().strip()
        if not tooltip:
            return
        anchor = self.mapToGlobal(QPoint(self.width() + 6, self.height() // 2))
        QToolTip.showText(anchor, tooltip, self, self.rect(), 30000)

    def enterEvent(self, event):
        super().enterEvent(event)
        self._show_tooltip_now()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._show_tooltip_now()

    def mousePressEvent(self, event):
        self._show_tooltip_now()
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        super().leaveEvent(event)
