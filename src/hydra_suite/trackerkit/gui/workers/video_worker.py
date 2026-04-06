"""OrientedTrackVideoWorker — per-track orientation-corrected video export."""

from PySide6.QtCore import Signal

from hydra_suite.core.identity.dataset.oriented_video import OrientedTrackVideoExporter
from hydra_suite.widgets.workers import BaseWorker


class OrientedTrackVideoWorker(BaseWorker):
    """Worker thread for exporting orientation-fixed per-track videos."""

    progress_signal = Signal(int, str)
    finished_signal = Signal(dict)
    error_signal = Signal(str)

    def __init__(
        self,
        final_csv_path,
        dataset_dir,
        video_path,
        detection_cache_path,
        interpolated_roi_npz_path,
        fps,
        padding_fraction,
        background_color,
        suppress_foreign_obb,
    ):
        super().__init__()
        self.final_csv_path = final_csv_path
        self.dataset_dir = dataset_dir
        self.video_path = video_path
        self.detection_cache_path = detection_cache_path
        self.interpolated_roi_npz_path = interpolated_roi_npz_path
        self.fps = fps
        self.padding_fraction = padding_fraction
        self.background_color = background_color
        self.suppress_foreign_obb = suppress_foreign_obb
        self._stop_requested = False

    def stop(self):
        """Request cooperative cancellation."""
        self._stop_requested = True

    def _should_stop(self) -> bool:
        return bool(self._stop_requested or self.isInterruptionRequested())

    def execute(self):
        """Track and orient detected animals in video frames."""
        try:
            exporter = OrientedTrackVideoExporter(
                self.dataset_dir,
                self.final_csv_path,
                video_path=self.video_path,
                detection_cache_path=self.detection_cache_path,
                interpolated_roi_npz_path=self.interpolated_roi_npz_path,
                fps=self.fps,
                padding_fraction=self.padding_fraction,
                background_color=self.background_color,
                suppress_foreign_obb=self.suppress_foreign_obb,
            )
            result = exporter.export(
                progress_callback=self.progress_signal.emit,
                should_stop=self._should_stop,
            )
            if not self._should_stop():
                self.finished_signal.emit(result.to_dict())
        except Exception as exc:
            if not self._should_stop():
                self.error_signal.emit(str(exc))
