"""CompactHelpLabel — inline help text widget that attaches to group box titles."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOptionGroupBox,
    QWidget,
)

from hydra_suite.trackerkit.gui.widgets.collapsible import CollapsibleGroupBox
from hydra_suite.trackerkit.gui.widgets.tooltip_button import ImmediateTooltipButton


class CompactHelpLabel(QWidget):
    """Compact inline help affordance that keeps full guidance in an explicit icon."""

    def __init__(
        self, text: str = "", parent=None, attach_to_title: bool = True
    ) -> None:
        super().__init__(parent)
        self._text = ""
        self._attach_to_title = attach_to_title
        self._attached_host = None
        self._title_button = None
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._icon_button = ImmediateTooltipButton()
        self._icon_button.setText("?")
        self._icon_button.setCursor(Qt.PointingHandCursor)
        self._icon_button.setAutoRaise(True)
        self._icon_button.setFixedSize(16, 16)
        self._icon_button.setStyleSheet(
            "QToolButton {"
            " background-color: #1f3b53; color: #8fd3ff;"
            " border: 1px solid #325978; border-radius: 8px;"
            " font-size: 10px; font-weight: 700; padding: 0px;"
            "}"
            "QToolButton:hover { background-color: #255174; border-color: #4fc1ff; }"
        )
        layout.addWidget(self._icon_button, 0, Qt.AlignLeft | Qt.AlignTop)

        self.setText(text)

    def setText(self, text: str) -> None:
        self._text = text or ""
        for widget in (self, self._icon_button):
            widget.setToolTip(self._text)
        if isinstance(self._attached_host, CollapsibleGroupBox):
            self._attached_host.setHelpToolTip(self._text)
        elif self._title_button is not None:
            self._title_button.setToolTip(self._text)

    def text(self) -> str:
        return self._text

    def setWordWrap(self, enabled: bool) -> None:
        return None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._attach_to_title and self._attached_host is None:
            QTimer.singleShot(0, self._maybe_attach_to_title)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._attached_host and self._title_button is not None:
            self._position_groupbox_help_button()
        return super().eventFilter(watched, event)

    def _maybe_attach_to_title(self) -> None:
        if not self._attach_to_title or self._attached_host is not None:
            return
        host = self._find_title_host()
        if host is None:
            return

        self._attached_host = host
        if isinstance(host, CollapsibleGroupBox):
            host.setHelpToolTip(self._text)
        else:
            button = getattr(host, "_title_help_button", None)
            if button is None:
                button = ImmediateTooltipButton(host)
                button.setText("?")
                button.setAutoRaise(True)
                button.setCursor(Qt.PointingHandCursor)
                button.setFixedSize(16, 16)
                button.setStyleSheet(
                    "QToolButton {"
                    " background-color: #1f3b53; color: #8fd3ff;"
                    " border: 1px solid #325978; border-radius: 8px;"
                    " font-size: 10px; font-weight: 700; padding: 0px;"
                    "}"
                    "QToolButton:hover { background-color: #255174; border-color: #4fc1ff; }"
                )
                host._title_help_button = button
                host.installEventFilter(self)
            self._title_button = button
            self._title_button.setToolTip(self._text)
            self._title_button.show()
            self._position_groupbox_help_button()

        self.hide()
        self.setFixedSize(0, 0)

    def _find_title_host(self):
        if not self._is_first_widget_in_parent_layout():
            return None
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, (QGroupBox, CollapsibleGroupBox)):
                return parent
            parent = parent.parentWidget()
        return None

    def _is_first_widget_in_parent_layout(self) -> bool:
        parent = self.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None:
            return False
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is not None:
                return widget is self
        return False

    def _position_groupbox_help_button(self) -> None:
        if self._title_button is None or self._attached_host is None:
            return
        host = self._attached_host
        option = QStyleOptionGroupBox()
        host.initStyleOption(option)
        label_rect = host.style().subControlRect(
            QStyle.CC_GroupBox,
            option,
            QStyle.SC_GroupBoxLabel,
            host,
        )
        check_rect = host.style().subControlRect(
            QStyle.CC_GroupBox,
            option,
            QStyle.SC_GroupBoxCheckBox,
            host,
        )
        title_right = max(label_rect.right(), check_rect.right())
        x = min(host.width() - self._title_button.width() - 10, title_right + 6)
        x = max(18, x)
        y = max(2, label_rect.center().y() - (self._title_button.height() // 2))
        self._title_button.move(x, y)
