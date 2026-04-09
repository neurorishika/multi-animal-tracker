from __future__ import annotations

import gc
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtGui = pytest.importorskip("PySide6.QtGui")
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


def test_explorer_labeling_batch_keeps_labeled_points_colored_and_interactive(
    qapp,
) -> None:
    view = ExplorerView()
    coords = np.array(
        [[0.0, 0.0], [1.0, 1.0], [0.5, 0.25], [0.75, 0.75]], dtype=np.float32
    )

    view.set_data(
        coords,
        labels=["ant", "bee", None, "ant"],
        candidate_indices=[0],
        round_labeled_indices=[1],
        labeling_mode=True,
    )

    outside_unlabeled = view.points[2].brush().color()
    outside_labeled = view.points[3].brush().color()

    assert outside_unlabeled == QColor(90, 90, 90)
    assert outside_labeled != QColor(90, 90, 90)
    assert view.interactive_indices == {0, 1}
