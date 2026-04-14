"""Read-only OBB canvas viewer for DetectKit."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Scene
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # State
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        # GT layer (ground-truth, solid lines)
        self._gt_obb_items: list = []
        self._gt_label_items: list = []
        self._gt_class_ids: list[int] = []
        # Prediction layer (model output, dashed lines)
        self._pred_obb_items: list = []
        self._pred_label_items: list = []
        self._pred_class_ids: list[int] = []
        # Visibility state
        self._show_gt: bool = True
        self._show_pred: bool = True
        self._visible_class_ids: set[int] = set()
        # Backward-compat aliases (views of GT layer)
        self._obb_items = self._gt_obb_items
        self._label_items = self._gt_label_items
        self._zoom: float = 1.0
        self._min_zoom: float = 0.1
        self._max_zoom: float = 4.0
        self._panning: bool = False
        self._pan_start: Optional[QPointF] = None
        self._fit_mode: bool = True

        for target in (self, self.viewport()):
            target.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
            target.grabGesture(Qt.PinchGesture)
            target.installEventFilter(self)

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

    def _build_class_lookup(
        self, class_names: list[str] | dict[int, str] | None
    ) -> dict[int, str]:
        if isinstance(class_names, dict):
            return {int(k): str(v) for k, v in class_names.items()}
        return {idx: str(n) for idx, n in enumerate(class_names or ["object"])}

    def _draw_detections(
        self,
        detections: list[dict],
        obb_items: list,
        label_items: list,
        class_ids: list,
        class_names: list[str] | dict[int, str] | None,
        line_style: "Qt.PenStyle",
        show_confidence: bool = False,
    ) -> None:
        """Render *detections* into the given item lists."""
        font = QFont()
        font.setPixelSize(DEFAULT_OBB_FONT_SIZE)
        lookup = self._build_class_lookup(class_names)

        for det in detections:
            class_id: int = det.get("class_id", 0)
            polygon_px = det.get("polygon_px", [])
            if len(polygon_px) < 3:
                continue
            confidence = det.get("confidence", None)

            colour = _PALETTE[class_id % len(_PALETTE)]
            qpoly = QPolygonF()
            for x, y in polygon_px:
                qpoly.append(QPointF(x, y))
            qpoly.append(QPointF(*polygon_px[0]))

            pen = QPen(colour, DEFAULT_OBB_LINE_WIDTH)
            pen.setCosmetic(True)
            pen.setStyle(line_style)
            poly_item = self._scene.addPolygon(
                qpoly, pen, QBrush(Qt.BrushStyle.NoBrush)
            )
            obb_items.append(poly_item)
            class_ids.append(class_id)

            label_name = lookup.get(class_id, f"class_{class_id}")
            if show_confidence and confidence is not None:
                label_text = f"{label_name} ({confidence:.2f})"
            else:
                label_text = f"{label_name} ({class_id})"
            txt_item = QGraphicsTextItem(label_text)
            txt_item.setFont(font)
            txt_item.setDefaultTextColor(colour)
            txt_item.setPos(QPointF(*polygon_px[0]))
            self._scene.addItem(txt_item)
            label_items.append(txt_item)

    def _apply_visibility(self) -> None:
        """Show/hide items based on visibility flags and class filter."""

        def _set_layer(obb_items, label_items, class_ids, layer_visible):
            for obb, lbl, cid in zip(obb_items, label_items, class_ids):
                visible = layer_visible and (
                    not self._visible_class_ids or cid in self._visible_class_ids
                )
                obb.setVisible(visible)
                lbl.setVisible(visible)

        _set_layer(
            self._gt_obb_items, self._gt_label_items, self._gt_class_ids, self._show_gt
        )
        _set_layer(
            self._pred_obb_items,
            self._pred_label_items,
            self._pred_class_ids,
            self._show_pred,
        )

    def set_gt_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
        *,
        append: bool = False,
    ) -> None:
        """Draw ground-truth OBB polygons (solid lines)."""
        if not append:
            self.clear_gt_detections()
        self._draw_detections(
            detections,
            self._gt_obb_items,
            self._gt_label_items,
            self._gt_class_ids,
            class_names,
            Qt.PenStyle.SolidLine,
            show_confidence=False,
        )
        self._apply_visibility()

    def set_pred_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
    ) -> None:
        """Draw model-prediction OBB polygons (dashed lines)."""
        self.clear_pred_detections()
        self._draw_detections(
            detections,
            self._pred_obb_items,
            self._pred_label_items,
            self._pred_class_ids,
            class_names,
            Qt.PenStyle.DashLine,
            show_confidence=True,
        )
        self._apply_visibility()

    def set_overlay_visibility(self, show_gt: bool, show_pred: bool) -> None:
        """Toggle GT and prediction layer visibility."""
        self._show_gt = show_gt
        self._show_pred = show_pred
        self._apply_visibility()

    def set_class_filter(self, visible_class_ids: set[int]) -> None:
        """Show only the given class IDs (empty set = show all)."""
        self._visible_class_ids = set(visible_class_ids)
        self._apply_visibility()

    # Backward-compat aliases
    def set_detections(
        self,
        detections: list[dict],
        class_names: list[str] | dict[int, str] | None = None,
    ) -> None:
        """Backward-compatible alias: set GT detections."""
        self.set_gt_detections(detections, class_names)

    def clear_detections(self) -> None:
        """Backward-compatible alias: clear GT layer."""
        self.clear_gt_detections()

    def clear_gt_detections(self) -> None:
        """Remove all GT polygon and label items from the scene."""
        for item in self._gt_obb_items:
            self._scene.removeItem(item)
        for item in self._gt_label_items:
            self._scene.removeItem(item)
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()

    def clear_pred_detections(self) -> None:
        """Remove all prediction polygon and label items from the scene."""
        for item in self._pred_obb_items:
            self._scene.removeItem(item)
        for item in self._pred_label_items:
            self._scene.removeItem(item)
        self._pred_obb_items.clear()
        self._pred_label_items.clear()
        self._pred_class_ids.clear()

    def clear_all(self) -> None:
        """Remove everything from the scene."""
        self._scene.clear()
        self._pix_item = None
        self._gt_obb_items.clear()
        self._gt_label_items.clear()
        self._gt_class_ids.clear()
        self._pred_obb_items.clear()
        self._pred_label_items.clear()
        self._pred_class_ids.clear()
        self._zoom = 1.0
        self._fit_mode = True
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------------
    # View helpers
    # ------------------------------------------------------------------

    def _current_zoom(self) -> float:
        return float(self.transform().m11())

    def _set_zoom(self, new_zoom: float) -> bool:
        bounded_zoom = max(self._min_zoom, min(self._max_zoom, float(new_zoom)))
        current_zoom = self._current_zoom()
        if current_zoom <= 0:
            return False
        if abs(bounded_zoom - current_zoom) < 1e-6:
            self._zoom = bounded_zoom
            return False
        self.scale(bounded_zoom / current_zoom, bounded_zoom / current_zoom)
        self._zoom = bounded_zoom
        self._fit_mode = False
        return True

    def _step_zoom(self, delta: int) -> bool:
        if delta == 0:
            return False
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        return self._set_zoom(self._current_zoom() * factor)

    def fit_in_view(self) -> None:
        """Fit the current pixmap in the viewport, keeping aspect ratio."""
        if self._pix_item is not None:
            self.resetTransform()
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = max(
                self._min_zoom,
                min(self._max_zoom, self._current_zoom()),
            )
            if abs(self._zoom - self._current_zoom()) > 1e-6:
                self.resetTransform()
                self.scale(self._zoom, self._zoom)
                self.centerOn(self._pix_item)
            self._fit_mode = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def _handle_native_gesture(self, event) -> bool:
        gesture_type = event.gestureType()
        zoom_gesture = getattr(Qt, "ZoomNativeGesture", None)
        begin_gesture = getattr(Qt, "BeginNativeGesture", None)
        end_gesture = getattr(Qt, "EndNativeGesture", None)
        if gesture_type in (begin_gesture, end_gesture):
            event.accept()
            return True
        if gesture_type != zoom_gesture:
            return False

        scale_delta = float(event.value())
        if abs(scale_delta) < 1e-6:
            event.accept()
            return True

        current_zoom = self._current_zoom()
        scaled_zoom = current_zoom * max(0.2, 1.0 + scale_delta)
        if int(round(scaled_zoom * 100)) == int(round(current_zoom * 100)):
            scaled_zoom = current_zoom + (0.01 if scale_delta > 0 else -0.01)
        self._set_zoom(scaled_zoom)
        event.accept()
        return True

    def _handle_pinch_gesture(self, event) -> bool:
        pinch = event.gesture(Qt.PinchGesture)
        if pinch is None:
            return False
        if pinch.state() == Qt.GestureUpdated:
            zoom_delta = int((pinch.scaleFactor() - 1.0) * 60)
            if zoom_delta != 0:
                self._set_zoom(self._current_zoom() + (zoom_delta / 100.0))
        event.accept()
        return True

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def eventFilter(self, watched, event):
        if watched in (self, self.viewport()):
            if event.type() == QEvent.NativeGesture and self._handle_native_gesture(
                event
            ):
                return True
            if event.type() == QEvent.Gesture and self._handle_pinch_gesture(event):
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        """Zoom with Ctrl+wheel; otherwise allow normal viewport scrolling."""
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._step_zoom(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Start panning on left or middle drag."""
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
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
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        """Fit the image to the viewport on double-click."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_in_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Support keyboard zoom, fit, and panning."""
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self._set_zoom(self._current_zoom() + 0.1)
            event.accept()
            return
        if key == Qt.Key_Minus:
            self._set_zoom(self._current_zoom() - 0.1)
            event.accept()
            return
        if key in (Qt.Key_0, Qt.Key_F):
            self.fit_in_view()
            event.accept()
            return

        pan_step = 48
        if key == Qt.Key_Left:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Right:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() + pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Up:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - pan_step
            )
            event.accept()
            return
        if key == Qt.Key_Down:
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() + pan_step
            )
            event.accept()
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep fit-to-screen active across resizes when fit mode is enabled."""
        super().resizeEvent(event)
        if self._pix_item is not None and self._fit_mode:
            self.fit_in_view()
