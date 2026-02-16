from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class ImageCanvas(QGraphicsView):
    """
    Styled image viewer based on PoseKit's PoseCanvas.
    Supports zooming, panning, and displaying an image.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # PoseKit dark theme
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.setStyleSheet("QGraphicsView { background-color: #121212; border: none; }")

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 20.0
        self._pixmap_cache = {}
        self._cache_order = []
        self._cache_limit = 96
        self._last_path = None
        self._last_dimensions = None

    def _cache_get(self, key: str):
        if key not in self._pixmap_cache:
            return None
        return self._pixmap_cache[key]

    def _cache_put(self, key: str, pixmap: QPixmap):
        if key in self._pixmap_cache:
            return
        self._pixmap_cache[key] = pixmap
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_limit:
            old_key = self._cache_order.pop(0)
            self._pixmap_cache.pop(old_key, None)

    def set_image(self, img_path: str):
        """Load an image from path and display it."""
        if not img_path:
            return

        if (
            self._last_path == img_path
            and self.pix_item.pixmap()
            and not self.pix_item.pixmap().isNull()
        ):
            return

        cached = self._cache_get(img_path)
        if cached is not None:
            self.pix_item.setPixmap(cached)
            self.scene.setSceneRect(QRectF(cached.rect()))
            dims = (cached.width(), cached.height())
            if self._last_dimensions != dims:
                self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                self._last_dimensions = dims
            self._last_path = img_path
            return

        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            return

        self._cache_put(img_path, pixmap)
        self.pix_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

        dims = (pixmap.width(), pixmap.height())
        if self._last_dimensions != dims:
            self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            self._last_dimensions = dims
        self._last_path = img_path

    def wheelEvent(self, event: QWheelEvent):
        """Zoom logic from PoseKit."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in = delta > 0
        factor = 1.15 if zoom_in else 1 / 1.15

        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        if self._min_zoom <= new_scale <= self._max_zoom:
            self.scale(factor, factor)
