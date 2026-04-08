from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hydra_suite.trackerkit.gui.orchestrators import config as config_module
from hydra_suite.trackerkit.gui.orchestrators import tracking as tracking_module
from hydra_suite.trackerkit.gui.orchestrators.config import ConfigOrchestrator
from hydra_suite.trackerkit.gui.orchestrators.tracking import TrackingOrchestrator


def _make_orchestrator() -> tuple[TrackingOrchestrator, object]:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(file_line=SimpleNamespace(text=lambda: "video.mp4"))
    )
    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
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


def test_load_config_uses_main_window_parent(monkeypatch) -> None:
    main_window = object()
    panels = SimpleNamespace(setup=SimpleNamespace(config_status_label=None))
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    def fake_get_open_file_name(parent, title, directory, file_filter):
        captured["parent"] = parent
        captured["title"] = title
        captured["directory"] = directory
        captured["file_filter"] = file_filter
        return "", ""

    monkeypatch.setattr(
        config_module.QFileDialog,
        "getOpenFileName",
        fake_get_open_file_name,
    )

    orchestrator.load_config()

    assert captured == {
        "parent": main_window,
        "title": "Load Configuration",
        "directory": "",
        "file_filter": "JSON Files (*.json)",
    }


def test_open_parameter_helper_uses_main_window_parent(monkeypatch) -> None:
    main_window = object()
    panels = SimpleNamespace(
        setup=SimpleNamespace(
            file_line=SimpleNamespace(text=lambda: "video.mp4"),
            spin_start_frame=SimpleNamespace(value=lambda: 0),
            spin_end_frame=SimpleNamespace(value=lambda: 100),
        )
    )
    orchestrator = ConfigOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )
    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(
            self,
            video_path: str,
            cache_path: str,
            start_frame: int,
            end_frame: int,
            params: dict[str, object],
            parent=None,
        ) -> None:
            captured.update(
                {
                    "video_path": video_path,
                    "cache_path": cache_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "params": params,
                    "parent": parent,
                }
            )

        def exec(self) -> int:
            return config_module.QDialog.Rejected

    monkeypatch.setattr(
        "hydra_suite.trackerkit.gui.dialogs.parameter_helper.ParameterHelperDialog",
        FakeDialog,
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "get_parameters_dict",
        lambda self: {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
    )
    monkeypatch.setattr(
        ConfigOrchestrator,
        "_find_or_plan_optimizer_cache_path",
        lambda self, video_path, params, start_frame, end_frame: (
            "/tmp/cache.npz",
            True,
        ),
    )
    monkeypatch.setattr(config_module.os.path, "exists", lambda path: True)

    orchestrator._open_parameter_helper()

    assert captured == {
        "video_path": "video.mp4",
        "cache_path": "/tmp/cache.npz",
        "start_frame": 0,
        "end_frame": 100,
        "params": {"YOLO_CONFIDENCE_THRESHOLD": 0.5},
        "parent": main_window,
    }


def test_start_tracking_on_video_restores_csv_and_worker_imports(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeSignal:
        def connect(self, _callback) -> None:
            return None

    class FakeProgress:
        def setVisible(self, _visible: bool) -> None:
            return None

        def setValue(self, _value: int) -> None:
            return None

        def setText(self, _text: str) -> None:
            return None

    class FakeCSVWriterThread:
        def __init__(self, path: str, header=None) -> None:
            captured["csv_path"] = path
            captured["csv_header"] = list(header or [])

        def start(self) -> None:
            captured["csv_started"] = True

    class FakeTrackingWorker:
        def __init__(self, *args, **kwargs) -> None:
            captured["worker_args"] = args
            captured["worker_kwargs"] = kwargs
            self.frame_signal = FakeSignal()
            self.finished_signal = FakeSignal()
            self.progress_signal = FakeSignal()
            self.stats_signal = FakeSignal()
            self.warning_signal = FakeSignal()
            self.pose_exported_model_resolved_signal = FakeSignal()

        def set_parameters(self, params) -> None:
            captured["params"] = dict(params)

        def start(self) -> None:
            captured["worker_started"] = True

        def isRunning(self) -> bool:
            return False

        def update_parameters(self, _params) -> None:
            return None

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    csv_path = tmp_path / "tracks.csv"

    setup_panel = SimpleNamespace(
        csv_line=SimpleNamespace(text=lambda: str(csv_path)),
        check_save_confidence=SimpleNamespace(isChecked=lambda: False),
        chk_use_cached_detections=SimpleNamespace(isChecked=lambda: False),
        file_line=SimpleNamespace(text=lambda: str(video_path)),
    )
    tracking_panel = SimpleNamespace(
        chk_enable_backward=SimpleNamespace(isChecked=lambda: False)
    )
    panels = SimpleNamespace(setup=setup_panel, tracking=tracking_panel)

    main_window = SimpleNamespace(
        tracking_worker=None,
        _stop_all_requested=False,
        _pending_finish_after_interp=False,
        _session_result_dataset=None,
        _dataset_was_started=False,
        _show_summary_on_dataset_done=False,
        _session_wall_start=None,
        _session_final_csv_path=None,
        _session_fps_list=[],
        _session_frames_processed=0,
        is_playing=False,
        _tracking_first_frame=False,
        csv_writer_thread=None,
        current_detection_cache_path=None,
        parameters_changed=FakeSignal(),
        progress_bar=FakeProgress(),
        progress_label=FakeProgress(),
        _selected_identity_method=lambda: "",
        get_parameters_dict=lambda: {"DETECTION_METHOD": "background_subtraction"},
        _prepare_tracking_display=lambda: captured.setdefault("prepared", True),
        _apply_ui_state=lambda state: captured.setdefault("ui_state", state),
        _stop_playback=lambda: None,
    )

    orchestrator = TrackingOrchestrator(
        main_window=main_window,
        config=object(),
        panels=panels,
    )

    monkeypatch.setattr(
        "hydra_suite.data.csv_writer.CSVWriterThread",
        FakeCSVWriterThread,
    )
    monkeypatch.setattr(
        "hydra_suite.core.tracking.TrackingWorker",
        FakeTrackingWorker,
    )
    monkeypatch.setattr(
        tracking_module,
        "candidate_artifact_base_dirs",
        lambda _video_path, preferred_base_dirs=None: [tmp_path],
    )
    monkeypatch.setattr(
        tracking_module,
        "choose_writable_artifact_base_dir",
        lambda _video_path, preferred_base_dirs=None: tmp_path,
    )
    monkeypatch.setattr(
        tracking_module,
        "find_existing_detection_cache_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tracking_module,
        "build_detection_cache_path",
        lambda *_args, **_kwargs: tmp_path / "cache.npz",
    )

    orchestrator.start_tracking_on_video(str(video_path), backward_mode=False)

    assert captured["csv_path"] == str(csv_path)
    assert captured["csv_started"] is True
    assert captured["worker_started"] is True
    assert main_window.csv_writer_thread is not None
    assert main_window.tracking_worker is not None
    assert captured["worker_kwargs"]["detection_cache_path"] == str(
        tmp_path / "cache.npz"
    )
    assert captured["worker_kwargs"]["preview_mode"] is False
