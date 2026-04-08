"""Tests for OBB label parsing (canvas drawing tested manually)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from hydra_suite.detectkit.gui.canvas import OBBCanvas  # noqa: E402
from hydra_suite.detectkit.gui.utils import parse_obb_label  # noqa: E402


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_parse_obb_label(tmp_path: Path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 0
    assert len(dets[0]["polygon_px"]) == 4
    assert abs(dets[0]["polygon_px"][0][0] - 10.0) < 0.1
    assert abs(dets[0]["polygon_px"][0][1] - 20.0) < 0.1


def test_parse_obb_label_empty(tmp_path: Path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert dets == []


def test_parse_obb_label_invalid_line(tmp_path: Path):
    lbl = tmp_path / "bad.txt"
    lbl.write_text("0 0.1 0.2\n0 0.1 0.2 0.9 0.2 0.9 0.8 0.1 0.8\n", encoding="utf-8")
    dets = parse_obb_label(lbl, img_w=100, img_h=100)
    assert len(dets) == 1


def test_canvas_uses_class_name_lookup_for_labels(qapp):
    canvas = OBBCanvas()
    canvas.set_detections(
        [
            {
                "class_id": 1,
                "polygon_px": [(10.0, 10.0), (40.0, 10.0), (40.0, 30.0), (10.0, 30.0)],
            }
        ],
        class_names=["ant", "bee"],
    )

    assert len(canvas._label_items) == 1
    assert canvas._label_items[0].toPlainText() == "bee (1)"


class _StubWheelEvent:
    def __init__(self, delta: int = 120) -> None:
        self._delta = delta
        self.accepted = False

    def modifiers(self):
        return Qt.ControlModifier

    def angleDelta(self):
        from PySide6.QtCore import QPoint

        return QPoint(0, self._delta)

    def accept(self) -> None:
        self.accepted = True


def test_canvas_auto_fits_loaded_image_and_clamps_zoom_controls(qapp):
    canvas = OBBCanvas()
    canvas.resize(640, 480)
    canvas.show()
    qapp.processEvents()

    img = np.zeros((1800, 2400, 3), dtype=np.uint8)
    assert canvas.set_image_array(img) is True
    qapp.processEvents()

    assert canvas._zoom < 1.0

    canvas._set_zoom(10.0)
    assert canvas._zoom == canvas._max_zoom

    canvas.fit_in_view()
    assert canvas._fit_mode is True


def test_canvas_ctrl_wheel_zoom_changes_zoom(qapp):
    canvas = OBBCanvas()
    canvas.resize(640, 480)
    canvas.show()
    qapp.processEvents()

    img = np.zeros((800, 1200, 3), dtype=np.uint8)
    assert canvas.set_image_array(img) is True
    qapp.processEvents()

    before = canvas._zoom
    event = _StubWheelEvent(120)
    canvas.wheelEvent(event)

    assert event.accepted is True
    assert canvas._zoom > before
