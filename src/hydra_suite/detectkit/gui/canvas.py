"""Read-only OBB canvas viewer for DetectKit."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from .constants import CANVAS_BG_COLOR, DEFAULT_OBB_FONT_SIZE, DEFAULT_OBB_LINE_WIDTH

logger = logging.getLogger(__name__)

# 8-colour palette for class IDs (cycled via modulo)
_PALETTE = [
    QColor(0, 255, 0),  # green
    QColor(255, 80, 80),  # red
    QColor(80, 180, 255),  # blue
    QColor(255, 200, 0),  # yellow
    QColor(200, 80, 255),  # purple
    QColor(0, 220, 200),  # cyan
    QColor(255, 140, 0),  # orange
    QColor(180, 220, 80),  # lime
]


class OBBCanvas(QGraphicsView):
    """Read-only image viewer with oriented-bounding-box overlays."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Rendering
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(CANVAS_BG_COLOR)))

        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # State
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._obb_items: list = []
        self._label_items: list = []
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 30.0
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image_path: str) -> bool:
        """Load an image from *image_path* via OpenCV."""
        bgr = cv2.imread(image_path)
        if bgr is None:
            logger.warning("Failed to read image: %s", image_path)
            return False
        return self.set_image_array(bgr)

    def set_image_array(self, bgr: np.ndarray) -> bool:
        """Display a BGR numpy array on the canvas."""
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            bytes_per_line = w * 3
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            if self._pix_item is None:
                self._pix_item = QGraphicsPixmapItem(pixmap)
                self._scene.addItem(self._pix_item)
            else:
                self._pix_item.setPixmap(pixmap)

            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.fit_in_view()
            return True
        except Exception:
            logger.warning("Failed to set image array", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Detection overlays
    # ------------------------------------------------------------------

    def set_detections(
        self,
        detections: list[dict],
        class_name: str = "object",
    ) -> None:
        """Draw OBB polygons for *detections* on the canvas.

        Each detection dict must have ``class_id`` (int) and
        ``polygon_px`` (list of four ``(x, y)`` tuples).
        """
        self.clear_detections()

        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)

        for det in detections:
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue

            colour = _PALETTE[class_id % len(_PALETTE)]

            # Build polygon
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))  # close polygon

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            poly_item = self._scene.addPolygon(
                qpoly, pen, QBrush(Qt.BrushStyle.NoBrush)
            )
            self._obb_items.append(poly_item)

            # Label at first corner
            label_text = f"{class_name}:{class_id}"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            self._scene.addItem(txt_item)
            self._label_items.append(txt_item)

    def clear_detections(self) -> None:
        """Remove all OBB polygon and label items from the scene."""
        for item in self._obb_items:
            self._scene.removeItem(item)
        for item in self._label_items:
            self._scene.removeItem(item)
        self._obb_items.clear()
        self._label_items.clear()

    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._obb_items.clear()
        self._label_items.clear()

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def fit_in_view(self) -> None:
        """Fit the current pixmap in the viewport, keeping aspect ratio."""
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom in/out on scroll."""
        factor = 1.15
        if event.angleDelta().y() > 0:
            zoom_factor = factor
        else:
            zoom_factor = 1.0 / factor

        new_zoom = self._zoom * zoom_factor
        if new_zoom < self._min_zoom or new_zoom > self._max_zoom:
            return

        self._zoom = new_zoom
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Start panning on MiddleButton or Ctrl+LeftButton."""
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        """Adjust scrollbars while panning."""
        if self._panning and self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        """End panning."""
        if self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-fit image when the widget is resized."""
        super().resizeEvent(event)
        if self._pix_item is not None:
            self.fit_in_view()
