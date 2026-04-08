from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QApplication = QtWidgets.QApplication

ImageCanvas = pytest.importorskip(
    "hydra_suite.classkit.gui.widgets.image_viewer"
).ImageCanvas


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


def test_image_canvas_clears_last_path_when_clahe_state_changes(
    qapp, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 1] = 180
    assert cv2.imwrite(str(image_path), image)

    canvas = ImageCanvas()
    canvas.set_image(str(image_path))

    assert canvas._last_path == str(image_path)

    canvas.use_clahe = True

    assert canvas._last_path is None
    assert canvas._pixmap_cache == {}


def test_image_canvas_clears_last_path_when_clahe_params_change(
    qapp, tmp_path: Path
) -> None:
    image_path = tmp_path / "sample.png"
    image = np.full((12, 12, 3), 90, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    canvas = ImageCanvas()
    canvas.set_image(str(image_path))

    assert canvas._last_path == str(image_path)

    canvas.set_clahe_params(3.0, (10, 10))

    assert canvas._last_path is None
    assert canvas._pixmap_cache == {}
