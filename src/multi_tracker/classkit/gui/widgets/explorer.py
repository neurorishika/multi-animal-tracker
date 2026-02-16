import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
)


class ExplorerView(QGraphicsView):
    """
    Scatter plot visualization of UMAP embeddings.
    Allows selection and exploration.
    """

    point_clicked = Signal(int)  # Emits index of clicked point
    point_hovered = Signal(int)  # Emits index when hovering over a point
    empty_double_clicked = Signal()  # Emits when empty background is double-clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        # Dark theme background
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setStyleSheet("QGraphicsView { border: none; }")

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.points = []  # Keep track of items
        self.data_indices = []  # Map item index to data index
        self.selected_index = None
        self.candidate_indices = set()
        self.round_labeled_indices = set()
        self.labeling_mode = False
        self.interactive_indices = set()
        self._last_hover_idx = None
        self._has_fitted_view = False
        self._coords = None
        self._labels = None
        self._confidences = None
        self._base_colors = []
        self._base_radii = []
        self._point_centers = []
        self._zoom_redraw_limit = 4000

        # Interaction state
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 50.0

    def _radius_scale(self) -> float:
        view_scale = self.transform().m11()
        if view_scale <= 0:
            view_scale = 1.0
        zoom_gain = min(0.45, max(-0.35, (view_scale - 1.0) * 0.10))
        return 1.0 - zoom_gain

    def _apply_item_style(
        self, idx: int, item: QGraphicsEllipseItem, radius_scale: float
    ):
        """Apply pen/z-order style for a point based on selection and candidate state."""
        base_color = (
            self._base_colors[idx]
            if idx < len(self._base_colors)
            else QColor(100, 100, 255)
        )
        base_radius = (
            self._base_radii[idx]
            if idx < len(self._base_radii)
            else (3.0 * radius_scale)
        )
        center = item.rect().center()

        if self.selected_index is not None and idx == self.selected_index:
            selected_radius = max(base_radius * 1.55, 10.5 * radius_scale)
            item.setRect(
                center.x() - selected_radius,
                center.y() - selected_radius,
                selected_radius * 2,
                selected_radius * 2,
            )
            item.setBrush(QBrush(QColor(255, 245, 120)))
            item.setPen(QPen(QColor(255, 40, 40), max(2.2, 2.8 * radius_scale)))
            item.setZValue(8)
        elif idx in self.candidate_indices:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))
            item.setPen(QPen(QColor(255, 220, 120), max(0.8, 1.0 * radius_scale)))
            item.setZValue(3)
        elif idx in self.round_labeled_indices:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))
            item.setPen(QPen(QColor(200, 200, 200), max(0.6, 0.8 * radius_scale)))
            item.setZValue(2)
        else:
            item.setRect(
                center.x() - base_radius,
                center.y() - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            item.setBrush(QBrush(base_color))
            item.setPen(QPen(Qt.NoPen))
            item.setZValue(1)

    def set_selected_index(self, selected_index: int | None) -> bool:
        """Update selected point styling without rebuilding the full scene.

        Returns True when fast-path update was applied, False when caller should fall back
        to a full redraw.
        """
        if not self.points:
            self.selected_index = selected_index
            return False

        previous = self.selected_index
        if previous == selected_index:
            return True

        self.selected_index = selected_index
        if self.labeling_mode:
            self.interactive_indices = set(self.candidate_indices)
            if self.selected_index is not None:
                self.interactive_indices.add(self.selected_index)

        radius_scale = self._radius_scale()

        for idx in (previous, selected_index):
            if idx is None:
                continue
            if idx < 0 or idx >= len(self.points):
                continue
            item = self.points[idx]
            self._apply_item_style(idx, item, radius_scale)

        return True

    def update_state(
        self,
        labels: list = None,
        candidate_indices: list = None,
        round_labeled_indices: list = None,
        selected_index: int = None,
        labeling_mode: bool = False,
    ) -> bool:
        """Update point styling/labels without rebuilding scene geometry.

        Returns False when caller should fall back to full set_data.
        """
        if not self.points or self._coords is None:
            return False

        point_count = len(self.points)
        if len(self._coords) != point_count:
            return False

        self._labels = labels
        self.selected_index = selected_index
        self.candidate_indices = set(candidate_indices or [])
        self.round_labeled_indices = set(round_labeled_indices or [])
        self.labeling_mode = labeling_mode
        if self.labeling_mode:
            self.interactive_indices = set(self.candidate_indices)
            if self.selected_index is not None:
                self.interactive_indices.add(self.selected_index)
        else:
            self.interactive_indices = set(range(point_count))

        radius_scale = self._radius_scale()
        self._base_colors = []
        self._base_radii = []

        for i, item in enumerate(self.points):
            color = QColor(100, 100, 255)
            if (
                labels is not None
                and len(labels) > i
                and labels[i] is not None
                and labels[i] != ""
            ):
                import hashlib

                h = int(hashlib.md5(str(labels[i]).encode()).hexdigest(), 16)
                r = (h & 0xFF0000) >> 16
                g = (h & 0x00FF00) >> 8
                b = h & 0x0000FF
                color = QColor(r, g, b)

            if (
                self.labeling_mode
                and i not in self.candidate_indices
                and i not in self.round_labeled_indices
            ):
                color = QColor(90, 90, 90)

            base_radius = 3.0 * radius_scale
            if i in self.candidate_indices:
                base_radius = 6.0 * radius_scale
            if i in self.round_labeled_indices:
                base_radius = 5.0 * radius_scale

            self._base_colors.append(color)
            self._base_radii.append(base_radius)

            if i < len(self._point_centers):
                cx, cy = self._point_centers[i]
                item.setRect(
                    cx - base_radius,
                    cy - base_radius,
                    base_radius * 2,
                    base_radius * 2,
                )
            item.setBrush(QBrush(color))
            self._apply_item_style(i, item, radius_scale)

        return True

    def set_data(
        self,
        coords: np.ndarray,
        labels: list = None,
        confidences: list = None,
        candidate_indices: list = None,
        round_labeled_indices: list = None,
        selected_index: int = None,
        labeling_mode: bool = False,
        preserve_view: bool = True,
    ):
        """
        Populate scatter plot.
        coords: (N, 2) normalized or raw UMAP coordinates.
        labels: check for coloring.
        """
        self._coords = coords
        self._labels = labels
        self._confidences = confidences

        self.scene.clear()
        self.points = []
        self._base_colors = []
        self._base_radii = []
        self._point_centers = []
        self.data_indices = range(len(coords))
        self.selected_index = selected_index
        self.candidate_indices = set(candidate_indices or [])
        self.round_labeled_indices = set(round_labeled_indices or [])
        self.labeling_mode = labeling_mode
        if self.labeling_mode:
            self.interactive_indices = set(self.candidate_indices)
            if self.selected_index is not None:
                self.interactive_indices.add(self.selected_index)
        else:
            self.interactive_indices = set(range(len(coords)))
        self._last_hover_idx = None

        if len(coords) == 0:
            return

        # Normalize coordinates to fit in a reasonable view box (e.g. 0-1000)
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        span = max_vals - min_vals
        if span[0] == 0:
            span[0] = 1
        if span[1] == 0:
            span[1] = 1

        norm_coords = (coords - min_vals) / span * 1000.0

        # Draw points
        default_radius = 3.0
        radius_scale = self._radius_scale()

        for i, (x, y) in enumerate(norm_coords):
            # Color logic
            color = QColor(100, 100, 255)  # Default blue
            if (
                labels is not None
                and len(labels) > i
                and labels[i] is not None
                and labels[i] != ""
            ):
                # Distinct color for labeled (e.g. green)
                # Simple hash for consistency color from label string
                import hashlib

                h = int(hashlib.md5(str(labels[i]).encode()).hexdigest(), 16)
                r = (h & 0xFF0000) >> 16
                g = (h & 0x00FF00) >> 8
                b = h & 0x0000FF
                color = QColor(r, g, b)

            if (
                self.labeling_mode
                and i not in self.candidate_indices
                and i not in self.round_labeled_indices
            ):
                color = QColor(90, 90, 90)

            base_radius = default_radius * radius_scale
            if i in self.candidate_indices:
                base_radius = 6.0 * radius_scale
            if i in self.round_labeled_indices:
                base_radius = 5.0 * radius_scale

            item = QGraphicsEllipseItem(
                x - base_radius,
                y - base_radius,
                base_radius * 2,
                base_radius * 2,
            )
            self._point_centers.append((x, y))
            self._base_colors.append(color)
            self._base_radii.append(base_radius)
            self._apply_item_style(i, item, radius_scale)
            item.setFlag(QGraphicsItem.ItemIsSelectable)
            # Store index in data field or subclass
            item.setData(0, i)

            self.scene.addItem(item)
            self.points.append(item)

        if not preserve_view or not self._has_fitted_view:
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._has_fitted_view = True

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if not self.labeling_mode:
                return
            item = self.itemAt(event.pos())
            if isinstance(item, QGraphicsEllipseItem):
                idx = item.data(0)
                if idx in self.interactive_indices:
                    self.point_clicked.emit(idx)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        item = self.itemAt(event.pos())
        if isinstance(item, QGraphicsEllipseItem):
            idx = item.data(0)
            if idx in self.interactive_indices and idx != self._last_hover_idx:
                self._last_hover_idx = idx
                self.point_hovered.emit(idx)
        else:
            self._last_hover_idx = None

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if not isinstance(item, QGraphicsEllipseItem):
                self.empty_double_clicked.emit()

    def wheelEvent(self, event: QWheelEvent):
        """Zoom logic."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        zoom_in = delta > 0
        factor = 1.15 if zoom_in else 1 / 1.15

        self.scale(factor, factor)

        # Re-render to adjust size tiers with zoom level while preserving view.
        # For large datasets, avoid full scene rebuild on every wheel event.
        if self._coords is not None and len(self._coords) <= self._zoom_redraw_limit:
            self.set_data(
                self._coords,
                self._labels,
                self._confidences,
                candidate_indices=list(self.candidate_indices),
                round_labeled_indices=list(self.round_labeled_indices),
                selected_index=self.selected_index,
                labeling_mode=self.labeling_mode,
                preserve_view=True,
            )
