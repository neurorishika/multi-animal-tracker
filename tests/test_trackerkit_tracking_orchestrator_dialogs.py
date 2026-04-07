from __future__ import annotations

from types import SimpleNamespace

from hydra_suite.trackerkit.gui.orchestrators import tracking as tracking_module
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


def _make_orchestrator() -> tuple[TrackingOrchestrator, object]:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: "video.mp4"))
    )
    orchestrator = TrackingOrchestrator(main_window=main_window, config=object(), panels=panels)
    return orchestrator, main_window


def test_show_gpu_info_uses_main_window_parent(monkeypatch) -> None:
    orchestrator, main_window = _make_orchestrator()
    captured: dict[str, object] = {}

    class FakeMessageBox:
        Information = object()

        def __init__(self, parent=None):
            captured["parent"] = parent

        def setWindowTitle(self, title):
            captured["title"] = title

        def setTextFormat(self, text_format):
            captured["text_format"] = text_format

        def setText(self, text):
            captured["text"] = text

        def setIcon(self, icon):
            captured["icon"] = icon

        def exec(self):
            captured["executed"] = True

    monkeypatch.setattr(
        tracking_module,
        "QMessageBox",
        FakeMessageBox,
    )
    monkeypatch.setattr(
        "hydra_suite.utils.gpu_utils.get_device_info",
        lambda: {
            "cuda_available": False,
            "mps_available": True,
            "numba_available": True,
            "tensorrt_available": False,
            "torch_available": False,
        },
    )

    orchestrator.show_gpu_info()

    assert captured["parent"] is main_window
    assert captured["title"] == "GPU & Acceleration Info"
    assert captured["executed"] is True


def test_on_tracking_warning_uses_main_window_parent(monkeypatch) -> None:
    orchestrator, main_window = _make_orchestrator()
    orchestrator._mw = SimpleNamespace(_stop_all_requested=False)
    captured: dict[str, object] = {}

    def fake_information(parent, title, message):
        captured["parent"] = parent
        captured["title"] = title
        captured["message"] = message

    monkeypatch.setattr(tracking_module.QMessageBox, "information", fake_information)

    orchestrator._mw = SimpleNamespace(_stop_all_requested=False)
    main_window = orchestrator._mw
    orchestrator.on_tracking_warning("Heads up", "Check this")

    assert captured == {
        "parent": main_window,
        "title": "Heads up",
        "message": "Check this",
    }