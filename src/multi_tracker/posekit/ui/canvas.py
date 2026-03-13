from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QRect, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPainter,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

from .constants import DEFAULT_EDGE_WIDTH, DEFAULT_KPT_RADIUS, DEFAULT_LABEL_FONT_SIZE
from .models import Keypoint
from .utils import get_keypoint_palette


class PoseCanvas(QGraphicsView):
    """Interactive graphics canvas for keypoint editing and prediction overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))
        self.setStyleSheet("QGraphicsView { background-color: #121212; border: none; }")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pix_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)

        self.kpt_items: List[Optional[QGraphicsEllipseItem]] = []
        self.kpt_labels: List[Optional[QGraphicsTextItem]] = []
        self.edge_items: List[QGraphicsLineItem] = []
        self.pred_items: List[Optional[QGraphicsEllipseItem]] = []
        self.pred_edge_items: List[Optional[QGraphicsLineItem]] = []
        self.pred_labels: List[Optional[QGraphicsTextItem]] = []

        self._img_w = 1
        self._img_h = 1
        self._current_kpt = 0
        self._kpt_radius = DEFAULT_KPT_RADIUS
        self._label_font_size = DEFAULT_LABEL_FONT_SIZE
        self._palette = get_keypoint_palette()
        self._kpt_opacity = 1.0
        self._edge_opacity = 0.7
        self._edge_width = DEFAULT_EDGE_WIDTH
        self._zoom_factor = 1.0
        self._min_zoom = 0.3
        self._max_zoom = 30.0

        self._on_place = None
        self._on_move = None
        self._on_select = None
        self._dragging_kpt = None
        self._drag_start_pos = None
        self._dragging_pred = None
        self._pred_edges: List[Tuple[int, int]] = []

    def _ellipse_item_at(self, scene_pos) -> Optional[QGraphicsEllipseItem]:
        for it in self.scene.items(scene_pos):
            if isinstance(it, QGraphicsEllipseItem):
                return it
        return None

    def set_callbacks(
        self: object, on_place: object, on_move: object, on_select: object = None
    ) -> None:
        """Register callbacks for placement, drag movement, and optional selection."""
        self._on_place = on_place
        self._on_move = on_move
        self._on_select = on_select

    def set_current_keypoint(self, idx: int) -> None:
        """Set the active keypoint index used for click placement."""
        self._current_kpt = idx

    def set_kpt_radius(self, r: float) -> None:
        """Update rendered keypoint radius."""
        self._kpt_radius = max(0.5, float(r))

    def set_label_font_size(self, size: int) -> None:
        """Update keypoint label font size."""
        self._label_font_size = max(4, int(size))

    def set_kpt_opacity(self, opacity: float) -> None:
        """Update keypoint alpha channel used for overlay drawing."""
        self._kpt_opacity = max(0.0, min(1.0, float(opacity)))

    def set_edge_opacity(self, opacity: float) -> None:
        """Update skeleton edge alpha channel used for overlay drawing."""
        self._edge_opacity = max(0.0, min(1.0, float(opacity)))

    def set_edge_width(self, width: float) -> None:
        """Update skeleton edge width used for annotation and prediction edges."""
        self._edge_width = max(0.5, float(width))

    def fit_to_view(self: object) -> object:
        """Fit the image to the view."""
        if self.pix_item.pixmap().isNull():
            return
        self.fitInView(self.pix_item, Qt.KeepAspectRatio)
        # Update zoom factor
        self._zoom_factor = self.transform().m11()

    def set_image(self, img_bgr: np.ndarray) -> None:
        """Load a BGR image into the canvas and fit it to the current viewport."""
        h, w = img_bgr.shape[:2]
        self._img_w, self._img_h = w, h
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.pix_item.setPixmap(QPixmap.fromImage(qimg))
        self.scene.setSceneRect(QRectF(0, 0, w, h))
        self.resetTransform()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def rebuild_overlays(
        self,
        kpts: List[Keypoint],
        kpt_names: List[str],
        edges: List[Tuple[int, int]],
        pred_kpts: Optional[List[Keypoint]] = None,
        pred_confs: Optional[List[float]] = None,
        show_pred_conf: bool = False,
    ) -> None:
        """Rebuild all keypoint/skeleton and optional prediction overlay items."""
        for item in (
            self.kpt_items
            + self.kpt_labels
            + self.edge_items
            + self.pred_items
            + self.pred_edge_items
            + self.pred_labels
        ):
            if item is not None:
                self.scene.removeItem(item)
        self.kpt_items = [None] * len(kpts)
        self.kpt_labels = [None] * len(kpts)
        self.edge_items = [None] * len(edges)
        self.pred_items = [None] * (len(pred_kpts) if pred_kpts else 0)
        self.pred_edge_items = [None] * len(edges)
        self.pred_labels = [None] * (len(pred_kpts) if pred_kpts else 0)

        # Create edges with opacity
        edge_alpha = int(255 * self._edge_opacity)
        for ei, (a, b) in enumerate(edges):
            pen = QPen(QColor(100, 100, 100, edge_alpha), self._edge_width)
            edge = QGraphicsLineItem()
            edge.setPen(pen)
            edge.setZValue(3)
            self.scene.addItem(edge)
            self.edge_items[ei] = edge

        # Create keypoints and labels
        label_positions = []  # Track label positions for collision avoidance
        for i, kp in enumerate(kpts):
            if kp.v == 0:
                continue

            color = self._palette[i % len(self._palette)]
            base = color
            alpha = int(255 * self._kpt_opacity)

            # Different appearance for occluded keypoints (v=1)
            if kp.v == 1:
                alpha = int(alpha * 0.6)

            color = QColor(base.red(), base.green(), base.blue(), alpha)

            r = self._kpt_radius
            circ = QGraphicsEllipseItem(kp.x - r, kp.y - r, 2 * r, 2 * r)
            pen_alpha = int(200 * self._kpt_opacity)

            # Use different border style for occluded keypoints
            if kp.v == 1:
                # Dashed border for occluded keypoints
                pen = QPen(QColor(255, 100, 0, pen_alpha), 2.0)
                pen.setStyle(Qt.DashLine)
            else:
                # Solid border for visible keypoints
                pen = QPen(QColor(0, 0, 0, pen_alpha), 1.5)

            circ.setPen(pen)
            circ.setBrush(color)
            circ.setZValue(4)
            # Movement is handled manually to keep edges/labels in sync.
            circ.setData(0, i)
            self.scene.addItem(circ)
            self.kpt_items[i] = circ

            nm = kpt_names[i] if i < len(kpt_names) else "kp"
            lab = QGraphicsTextItem(f"{i}:{nm}")
            label_alpha = int(220 * self._kpt_opacity)
            lab.setDefaultTextColor(
                QColor(base.red(), base.green(), base.blue(), label_alpha)
            )
            lab.setFont(QFont("Arial", self._label_font_size))
            lab.setAcceptedMouseButtons(Qt.NoButton)
            lab.setFlag(QGraphicsTextItem.ItemIsSelectable, False)

            # Smart label positioning with collision avoidance
            dx, dy = self._find_best_label_offset(kp.x, kp.y, label_positions)
            lab.setPos(kp.x + dx, kp.y + dy)
            lab.setZValue(5)

            # Track this label's bounding box
            label_rect = lab.boundingRect()
            label_positions.append(
                (kp.x + dx, kp.y + dy, label_rect.width(), label_rect.height())
            )

            self.scene.addItem(lab)
            self.kpt_labels[i] = lab

        # Prediction overlays (distinct style)
        self._pred_edges = list(edges)
        if pred_kpts:
            pred_edge_alpha = int(160 * self._edge_opacity)
            for ei, (a, b) in enumerate(edges):
                if (
                    a < len(pred_kpts)
                    and b < len(pred_kpts)
                    and pred_kpts[a].v > 0
                    and pred_kpts[b].v > 0
                ):
                    pred_w = max(0.75, self._edge_width * 0.75)
                    pen = QPen(QColor(0, 220, 255, pred_edge_alpha), pred_w)
                    pen.setStyle(Qt.DashLine)
                    edge = QGraphicsLineItem(
                        pred_kpts[a].x,
                        pred_kpts[a].y,
                        pred_kpts[b].x,
                        pred_kpts[b].y,
                    )
                    edge.setPen(pen)
                    edge.setZValue(1)
                    self.scene.addItem(edge)
                    self.pred_edge_items[ei] = edge

            pred_alpha = int(140 * self._kpt_opacity)
            for i, kp in enumerate(pred_kpts):
                if kp.v == 0:
                    continue
                r = max(2.0, self._kpt_radius - 1.5)
                circ = QGraphicsEllipseItem(kp.x - r, kp.y - r, 2 * r, 2 * r)
                pen = QPen(QColor(0, 220, 255, pred_alpha), 1.5)
                pen.setStyle(Qt.DashLine)
                circ.setPen(pen)
                circ.setBrush(Qt.NoBrush)
                circ.setZValue(2)
                circ.setData(0, i)
                circ.setData(1, "pred")
                self.scene.addItem(circ)
                self.pred_items[i] = circ
                if show_pred_conf and pred_confs and i < len(pred_confs):
                    conf = pred_confs[i]
                    lab = QGraphicsTextItem(f"{conf:.3f}")
                    lab.setDefaultTextColor(QColor(0, 220, 255, 200))
                    lab.setFont(
                        QFont("Arial", max(4, int(self._label_font_size * 0.8)))
                    )
                    lab.setPos(kp.x + r + 2, kp.y - r - 2)
                    lab.setZValue(2)
                    lab.setAcceptedMouseButtons(Qt.NoButton)
                    self.scene.addItem(lab)
                    self.pred_labels[i] = lab

        self._update_edges_from_points(edges)

    def _update_edges_from_points(self, edges: List[Tuple[int, int]]):
        for ei, (a, b) in enumerate(edges):
            if ei >= len(self.edge_items):
                break
            pa = self.kpt_items[a] if 0 <= a < len(self.kpt_items) else None
            pb = self.kpt_items[b] if 0 <= b < len(self.kpt_items) else None
            if pa is None or pb is None:
                self.edge_items[ei].setVisible(False)
                continue
            self.edge_items[ei].setVisible(True)
            ra = pa.rect()
            rb = pb.rect()
            ax = ra.x() + ra.width() / 2
            ay = ra.y() + ra.height() / 2
            bx = rb.x() + rb.width() / 2
            by = rb.y() + rb.height() / 2
            self.edge_items[ei].setLine(ax, ay, bx, by)

    def _find_best_label_offset(
        self,
        x: float,
        y: float,
        existing_labels: List[Tuple[float, float, float, float]],
    ) -> Tuple[float, float]:
        """Find the best offset for a label to avoid collisions."""
        r = self._kpt_radius
        # Try multiple candidate positions in order of preference
        candidates = [
            (r + 8, -(r + 14)),  # top-right
            (r + 8, r + 4),  # bottom-right
            (-(r + 60), -(r + 14)),  # top-left
            (-(r + 60), r + 4),  # bottom-left
            (-(r + 26), -(r + 28)),  # far top-left
            (r + 22, -(r + 28)),  # far top-right
            (-(r + 26), r + 18),  # far bottom-left
            (r + 22, r + 18),  # far bottom-right
        ]

        # Approximate label size (will be refined when created)
        label_w = 50
        label_h = 16

        for dx, dy in candidates:
            label_x = x + dx
            label_y = y + dy

            # Check for collision with existing labels
            collision = False
            for ex, ey, ew, eh in existing_labels:
                # Check if rectangles overlap
                if not (
                    label_x + label_w < ex
                    or label_x > ex + ew
                    or label_y + label_h < ey
                    or label_y > ey + eh
                ):
                    collision = True
                    break

            if not collision:
                return (dx, dy)

        # If all positions collide, return the first one anyway
        return candidates[0]

    def wheelEvent(self: object, event: object) -> None:
        """Zoom in/out around cursor while clamping to sane zoom bounds."""
        # Use smaller zoom increments for smoother control
        factor = 1.01 if event.angleDelta().y() > 0 else 1 / 1.01
        new_zoom = self._zoom_factor * factor

        # Clamp zoom to reasonable bounds
        if new_zoom < self._min_zoom:
            new_zoom = self._min_zoom
            factor = new_zoom / self._zoom_factor
        elif new_zoom > self._max_zoom:
            new_zoom = self._max_zoom
            factor = new_zoom / self._zoom_factor

        self.scale(factor, factor)
        self._zoom_factor = new_zoom
        event.accept()

    def mousePressEvent(self: object, event: object) -> None:
        """Handle placement, selection, visibility toggle, and drag-start behavior."""
        if event.button() == Qt.RightButton:
            pos = self.mapToScene(event.position().toPoint())
            item = self._ellipse_item_at(pos)
            if item is not None:
                idx = int(item.data(0))
                is_pred = item.data(1) == "pred"
                if not is_pred:
                    parent = self.parent()
                    if parent and not hasattr(parent, "mode"):
                        parent = self.window()
                    if parent and hasattr(parent, "_toggle_kpt_visibility"):
                        parent._toggle_kpt_visibility(idx)
                        return
            if self._on_place:
                # Right-click marks keypoint as occluded (v=1)
                self._on_place(self._current_kpt, float(pos.x()), float(pos.y()), 1)
            return

        pos = self.mapToScene(event.position().toPoint())
        item = self._ellipse_item_at(pos)
        if item is not None:
            # Clicked on existing keypoint
            idx = int(item.data(0))
            is_pred = item.data(1) == "pred"

            # Get parent's mode
            parent = self.parent()
            # If parent is splitter (default Qt behavior when added to splitter), try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            mode = getattr(parent, "mode", "frame") if parent else "frame"

            if is_pred:
                # Start dragging prediction (adopt on release)
                self._dragging_pred = idx
                self._drag_start_pos = self.mapToScene(event.position().toPoint())
                return super().mousePressEvent(event)

            if mode == "keypoint":
                # In keypoint mode: NEVER allow dragging - treat click as placing next unlabeled keypoint
                ann = getattr(parent, "_ann", None) if parent else None
                if ann and hasattr(ann, "kpts"):
                    # Find next unlabeled keypoint
                    next_unlabeled = None
                    for i, kp in enumerate(ann.kpts):
                        if kp.v == 0:
                            next_unlabeled = i
                            break

                    if next_unlabeled is not None:
                        # Update canvas internal state FIRST
                        self._current_kpt = next_unlabeled

                        # Get position
                        pos = self.mapToScene(event.position().toPoint())

                        # Update parent state
                        if parent:
                            parent.current_kpt = next_unlabeled
                            parent.kpt_list.setCurrentRow(next_unlabeled)
                            # Update canvas visual state
                            parent.canvas.set_current_keypoint(next_unlabeled)
                            # Force UI update
                            QApplication.processEvents()

                        # Now place the keypoint with the updated state
                        if self._on_place:
                            self._on_place(
                                next_unlabeled,
                                float(pos.x()),
                                float(pos.y()),
                                2,
                            )
                    # If all labeled, ignore click
                return super().mousePressEvent(event)
            else:
                # In frame mode: start dragging existing keypoint
                if self._on_select:
                    self._on_select(idx)
                self._dragging_kpt = idx
                self._drag_start_pos = self.mapToScene(event.position().toPoint())
                return super().mousePressEvent(event)

        if event.button() == Qt.LeftButton:
            # Clicking on empty space - always place next unlabeled keypoint in both modes
            parent = self.parent()
            # If parent is splitter, try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            ann = getattr(parent, "_ann", None) if parent else None

            if ann and hasattr(ann, "kpts"):
                # Check if all keypoints are already labeled
                all_labeled = all(kp.v > 0 for kp in ann.kpts)
                if all_labeled:
                    # Ignore click - all keypoints exist
                    super().mousePressEvent(event)
                    return

                # Find next unlabeled keypoint
                next_unlabeled = None
                for i, kp in enumerate(ann.kpts):
                    if kp.v == 0:
                        next_unlabeled = i
                        break

                if next_unlabeled is not None:
                    # Update canvas internal state FIRST
                    self._current_kpt = next_unlabeled

                    # Update parent state
                    if parent:
                        parent.current_kpt = next_unlabeled
                        parent.kpt_list.setCurrentRow(next_unlabeled)
                        # Update canvas visual state
                        parent.canvas.set_current_keypoint(next_unlabeled)
                        # Force UI update
                        QApplication.processEvents()

                    # Now place the keypoint with the updated state
                    if self._on_place:
                        self._on_place(
                            next_unlabeled,
                            float(pos.x()),
                            float(pos.y()),
                            2,
                        )
            else:
                # No annotation data, place normally
                if self._on_place:
                    self._on_place(
                        self._current_kpt,
                        float(pos.x()),
                        float(pos.y()),
                        2,
                    )
        super().mousePressEvent(event)

    def mouseMoveEvent(self: object, event: object) -> None:
        """Update dragged keypoint/prediction position during pointer motion."""
        if self._dragging_pred is not None:
            pos = self.mapToScene(event.position().toPoint())
            pred_item = (
                self.pred_items[self._dragging_pred]
                if self._dragging_pred < len(self.pred_items)
                else None
            )
            if pred_item is not None:
                r = pred_item.rect().width() / 2.0
                pred_item.setRect(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
                self._update_pred_edges_from_items()
            super().mouseMoveEvent(event)
            return
        if self._dragging_kpt is not None and self._drag_start_pos is not None:
            # Never allow drag in keypoint mode
            parent = self.parent()
            # If parent is splitter, try window()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()

            mode = getattr(parent, "mode", "frame") if parent else "frame"
            if mode == "keypoint":
                # Keypoint mode: no dragging allowed
                super().mouseMoveEvent(event)
                return
            # Update keypoint position during drag (frame mode only)
            pos = self.mapToScene(event.position().toPoint())
            if self._on_move:
                self._on_move(self._dragging_kpt, float(pos.x()), float(pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self: object, event: object) -> None:
        """Commit drag interactions and optional prediction adoption on release."""
        if self._dragging_pred is not None:
            idx = self._dragging_pred
            self._dragging_pred = None
            pos = self.mapToScene(event.position().toPoint())
            parent = self.parent()
            if parent and not hasattr(parent, "mode"):
                parent = self.window()
            if parent and hasattr(parent, "_adopt_prediction_kpt"):
                parent._adopt_prediction_kpt(idx, float(pos.x()), float(pos.y()))
            super().mouseReleaseEvent(event)
            return
        if self._dragging_kpt is not None:
            # Complete drag operation
            self._dragging_kpt = None
            self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _update_pred_edges_from_items(self):
        if not self.pred_edge_items or not self.pred_items:
            return
        for ei, edge in enumerate(self.pred_edge_items):
            if edge is None:
                continue
            if ei >= len(self._pred_edges):
                continue
            a, b = self._pred_edges[ei]
            if a >= len(self.pred_items) or b >= len(self.pred_items):
                continue
            a_item = self.pred_items[a]
            b_item = self.pred_items[b]
            if a_item is None or b_item is None:
                continue
            a_pos = a_item.rect().center()
            b_pos = b_item.rect().center()
            edge.setLine(a_pos.x(), a_pos.y(), b_pos.x(), b_pos.y())


class FrameListDelegate(QStyledItemDelegate):
    """Render frame name with right-aligned prediction confidence."""

    CONF_ROLE = Qt.UserRole + 2
    KP_COUNT_ROLE = Qt.UserRole + 3
    CLUSTER_ROLE = Qt.UserRole + 4

    def paint(self: object, painter: object, option: object, index: object) -> None:
        """Render frame row with elided name plus confidence and label counters."""
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        text = index.data(Qt.DisplayRole) or ""
        conf_val = index.data(self.CONF_ROLE)
        conf_text = "" if conf_val in (None, "") else f"{float(conf_val):.4f}"
        kpt_val = index.data(self.KP_COUNT_ROLE)
        kpt_text = "" if kpt_val in (None, "") else f"k{int(kpt_val)}"
        cluster_val = index.data(self.CLUSTER_ROLE)
        if cluster_val in (None, "", -1):
            cluster_text = ""
        else:
            try:
                cluster_text = f"c{int(cluster_val)}"
            except Exception:
                cluster_text = ""

        fg = index.data(Qt.ForegroundRole)
        if isinstance(fg, QBrush):
            base_color = fg.color()
        elif isinstance(fg, QColor):
            base_color = fg
        else:
            base_color = opt.palette.color(QPalette.Text)

        if opt.state & QStyle.State_Selected:
            text_color = opt.palette.color(QPalette.HighlightedText)
            conf_color = opt.palette.color(QPalette.HighlightedText)
        else:
            text_color = base_color
            conf_color = QColor(140, 140, 140)

        r = opt.rect.adjusted(6, 0, -6, 0)
        conf_w = 54 if conf_text else 0
        kpt_w = 40 if kpt_text else 0
        cluster_w = 40 if cluster_text else 0
        available = max(0, r.width())
        total = conf_w + kpt_w + cluster_w
        if available < conf_w:
            conf_w = available
            kpt_w = 0
            cluster_w = 0
        elif available < conf_w + kpt_w:
            cluster_w = 0
        elif available < total:
            cluster_w = 0
        right_total = conf_w + kpt_w + cluster_w
        name_rect = QRect(
            r.left(), r.top(), max(0, r.width() - right_total), r.height()
        )
        conf_rect = QRect(
            max(r.left(), r.right() - conf_w + 1), r.top(), max(0, conf_w), r.height()
        )
        kpt_rect = QRect(
            max(r.left(), conf_rect.left() - kpt_w), r.top(), max(0, kpt_w), r.height()
        )
        cluster_rect = QRect(
            max(r.left(), kpt_rect.left() - cluster_w),
            r.top(),
            max(0, cluster_w),
            r.height(),
        )

        painter.save()
        painter.setPen(text_color)
        if name_rect.width() > 0:
            fm = painter.fontMetrics()
            elided = fm.elidedText(text, Qt.ElideRight, max(0, name_rect.width()))
            painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, elided)
        if cluster_text and cluster_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            cluster_elided = fm.elidedText(
                cluster_text, Qt.ElideLeft, cluster_rect.width()
            )
            painter.drawText(
                cluster_rect, Qt.AlignVCenter | Qt.AlignRight, cluster_elided
            )
        if kpt_text and kpt_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            kpt_elided = fm.elidedText(kpt_text, Qt.ElideLeft, kpt_rect.width())
            painter.drawText(kpt_rect, Qt.AlignVCenter | Qt.AlignRight, kpt_elided)
        if conf_text and conf_rect.width() > 0:
            painter.setPen(conf_color)
            fm = painter.fontMetrics()
            conf_elided = fm.elidedText(conf_text, Qt.ElideLeft, conf_rect.width())
            painter.drawText(conf_rect, Qt.AlignVCenter | Qt.AlignRight, conf_elided)
        painter.restore()
