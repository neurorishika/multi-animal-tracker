from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtGui = pytest.importorskip("PySide6.QtGui")
Qt = QtCore.Qt
QApplication = QtWidgets.QApplication
QGraphicsEllipseItem = QtWidgets.QGraphicsEllipseItem
QColor = QtGui.QColor

ExplorerView = pytest.importorskip(
    "hydra_suite.classkit.gui.widgets.explorer"
).ExplorerView


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def test_explorer_set_data_does_not_duplicate_scene_items(qapp) -> None:
    view = ExplorerView()
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.25]], dtype=np.float32)

    view.set_data(coords, labels=["a", "b", "c"])
    ellipse_items = [
        item for item in view.scene.items() if isinstance(item, QGraphicsEllipseItem)
    ]

    assert len(view.points) == 3
    assert len(ellipse_items) == 3
    assert view.update_state(labels=["c", "b", "a"]) is True

    view.set_data(coords, labels=["a", "b", "c"])
    ellipse_items = [
        item for item in view.scene.items() if isinstance(item, QGraphicsEllipseItem)
    ]

    assert len(view.points) == 3
    assert len(ellipse_items) == 3


def test_labeling_mode_without_candidate_batch_keeps_all_points_interactive(
    qapp,
) -> None:
    view = ExplorerView()
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    view.set_data(
        coords,
        labels=["alpha", None],
        candidate_indices=[],
        round_labeled_indices=[],
        labeling_mode=True,
    )

    assert view.labeling_mode is True
    assert view.interactive_indices == {0, 1}
    assert view.points[0].brush().color() != QColor(95, 95, 95)
    assert view.points[1].brush().color() == QColor(95, 95, 95)
    assert view.points[1].zValue() < view.points[0].zValue()


def test_uncertainty_outlines_only_render_in_prediction_mode(qapp) -> None:
    view = ExplorerView()
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = ["alpha", "beta"]
    confidences = [0.2, 0.9]

    view.set_data(coords, labels=labels, confidences=confidences)
    assert view.points[0].pen().style() == Qt.PenStyle.NoPen

    view.set_data(
        coords,
        labels=labels,
        confidences=confidences,
        prediction_mode=True,
    )
    assert view.points[0].pen().style() != Qt.PenStyle.NoPen
    assert view.points[0].pen().color() == QColor(255, 255, 255)

    view.set_data(
        coords,
        labels=labels,
        confidences=confidences,
        labeling_mode=True,
        prediction_mode=False,
    )
    assert view.points[0].pen().style() == Qt.PenStyle.NoPen
