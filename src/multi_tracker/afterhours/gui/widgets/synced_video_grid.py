"""Synchronized multi-panel video grid for merge comparison.

Provides N ``InteractiveCanvas`` panels that share a single transport
(play/pause/seek/frame-step).  Zoom and pan are mirrored across all
panels.

Used by the Fragment Merge Wizard to show side-by-side hypothesis
comparisons.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.gui.widgets.interactive_canvas import InteractiveCanvas

logger = logging.getLogger(__name__)

_PLAYBACK_FPS = 15


# ---------------------------------------------------------------------------
# Background frame loader for multiple crops
# ---------------------------------------------------------------------------


class _MultiCropLoader(QThread):
    """Load cropped frames for multiple panels from a single video.

    Each panel has its own crop box.  Frames are decoded once and each
    crop is extracted.
    """

    progress = Signal(int, int)
    finished = Signal()

    def __init__(
        self,
        video_path: str,
        crop_boxes: List[Tuple[int, int, int, int]],
        frame_start: int,
        frame_end: int,
        parent=None,
    ):
        super().__init__(parent)
        self._path = video_path
        self._crops = crop_boxes
        self._start = frame_start
        self._end = frame_end
        # panel_idx → {frame_idx: BGR ndarray}
        self.frames: Dict[int, Dict[int, np.ndarray]] = {
            i: {} for i in range(len(crop_boxes))
        }

    def run(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self.finished.emit()
            return
        total = self._end - self._start + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)
        for i in range(total):
            if self.isInterruptionRequested():
                break
            ret, frame = cap.read()
            if not ret:
                break
            idx = self._start + i
            h, w = frame.shape[:2]
            for pi, (x1, y1, x2, y2) in enumerate(self._crops):
                crop = frame[y1 : min(y2, h), x1 : min(x2, w)].copy()
                self.frames[pi][idx] = crop
            self.progress.emit(i + 1, total)
        cap.release()
        self.finished.emit()


# ---------------------------------------------------------------------------
# SyncedVideoGrid
# ---------------------------------------------------------------------------


class SyncedVideoGrid(QWidget):
    """Grid of InteractiveCanvas panels with shared playback controls.

    Signals
    -------
    frame_changed(int)
        Emitted when the current frame position changes.
    """

    frame_changed = Signal(int)

    def __init__(self, n_panels: int = 3, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._n_panels = n_panels
        self._panels: List[InteractiveCanvas] = []
        self._overlay_fns: List[Optional[Callable]] = [None] * n_panels
        self._frames: Dict[int, Dict[int, np.ndarray]] = {}  # panel → frame → BGR
        self._frame_start = 0
        self._frame_end = 0
        self._current_frame = 0
        self._loader: Optional[_MultiCropLoader] = None
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / _PLAYBACK_FPS))
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playing = False
        self._auto_play = False
        self._n_active_panels = 0
        # (video_path, crop_boxes_tuple, frame_start, frame_end) → _MultiCropLoader
        self._prefetch_cache: Dict[tuple, _MultiCropLoader] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Panel row
        panel_row = QHBoxLayout()
        panel_row.setSpacing(4)
        for _ in range(self._n_panels):
            canvas = InteractiveCanvas()
            canvas.zoom_changed.connect(self._mirror_zoom)
            canvas.setVisible(False)
            panel_row.addWidget(canvas, stretch=1)
            self._panels.append(canvas)
        root.addLayout(panel_row, stretch=1)

        # Loading indicator — large centered overlay
        self._loading_label = QLabel("\u23f3  Loading frames\u2026")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet(
            "color: #f0c040; font-size: 28px; font-weight: bold; padding: 20px; "
            "background: rgba(30, 30, 30, 200); border-radius: 12px;"
        )
        self._loading_label.setVisible(False)
        root.addWidget(self._loading_label)

        # Transport controls
        transport = QHBoxLayout()
        transport.setContentsMargins(4, 0, 4, 0)
        transport.setSpacing(6)

        self._btn_prev = QPushButton("|◄")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.setToolTip("Previous frame (Ctrl+←)")
        self._btn_prev.clicked.connect(self.step_back)
        transport.addWidget(self._btn_prev)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(32)
        self._btn_play.setToolTip("Play / Pause (Space)")
        self._btn_play.clicked.connect(self.toggle_play)
        transport.addWidget(self._btn_play)

        self._btn_next = QPushButton("►|")
        self._btn_next.setFixedWidth(32)
        self._btn_next.setToolTip("Next frame (Ctrl+→)")
        self._btn_next.clicked.connect(self.step_forward)
        transport.addWidget(self._btn_next)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setToolTip("Drag to seek within this clip")
        self._slider.valueChanged.connect(self._on_slider)
        transport.addWidget(self._slider, stretch=1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(100)
        transport.addWidget(self._frame_label)

        root.addLayout(transport)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def configure(
        self,
        video_path: str,
        crop_boxes: List[Tuple[int, int, int, int]],
        frame_start: int,
        frame_end: int,
        overlay_fns: Optional[List[Optional[Callable]]] = None,
        auto_play: bool = True,
    ) -> None:
        """Configure and start loading frames for each panel.

        Parameters
        ----------
        crop_boxes:
            One crop box per visible panel ``(x1, y1, x2, y2)``.
        overlay_fns:
            Optional per-panel callables ``fn(bgr_crop, frame_idx) -> bgr_crop``
            that draw overlays (trajectories, labels, etc.).
        auto_play:
            If ``True``, start playback automatically once frames have loaded.
        """
        self.stop()
        self._auto_play = auto_play
        n = len(crop_boxes)
        self._frame_start = frame_start
        self._frame_end = frame_end
        self._current_frame = frame_start

        # Store active count; panels are hidden until load completes
        self._n_active_panels = n
        for panel in self._panels:
            panel.setVisible(False)

        self._overlay_fns = [None] * self._n_panels
        if overlay_fns:
            for i, fn in enumerate(overlay_fns):
                if i < self._n_panels:
                    self._overlay_fns[i] = fn

        self._slider.setRange(frame_start, frame_end)
        self._slider.setValue(frame_start)
        self._update_frame_label()

        # Check prefetch cache for an already-started loader
        cache_key = (video_path, tuple(crop_boxes), frame_start, frame_end)
        cached = self._prefetch_cache.pop(cache_key, None)
        if cached is not None:
            if cached.isFinished():
                # Frames already decoded — skip the loading spinner entirely
                self._frames = cached.frames
                self._on_load_finished()
                return
            else:
                # Loader still running — hook into its finished signal and
                # show a lightweight spinner while we wait
                self._loading_label.setVisible(True)
                self._btn_play.setEnabled(False)
                self._btn_prev.setEnabled(False)
                self._btn_next.setEnabled(False)
                self._slider.setEnabled(False)
                if self._loader is not None and self._loader.isRunning():
                    self._loader.requestInterruption()
                    self._loader.wait(2000)
                self._loader = cached
                self._frames = cached.frames
                cached.finished.connect(self._on_load_finished)
                return

        # Show loading indicator and disable transport
        self._loading_label.setVisible(True)
        self._btn_play.setEnabled(False)
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._slider.setEnabled(False)

        # Start loader
        if self._loader is not None and self._loader.isRunning():
            self._loader.requestInterruption()
            self._loader.wait(2000)

        self._loader = _MultiCropLoader(
            video_path, crop_boxes, frame_start, frame_end, self
        )
        self._loader.finished.connect(self._on_load_finished)
        self._frames = self._loader.frames
        self._loader.start()

    def prefetch(
        self,
        video_path: str,
        crop_boxes: List[Tuple[int, int, int, int]],
        frame_start: int,
        frame_end: int,
    ) -> None:
        """Pre-load frames for a future ``configure()`` call in the background.

        If ``configure()`` is later called with the same arguments, it will
        reuse the already-decoded frames rather than starting a fresh load,
        eliminating (or greatly reducing) the loading spinner.
        """
        key = (video_path, tuple(crop_boxes), frame_start, frame_end)
        if key in self._prefetch_cache:
            return  # already loading or loaded
        loader = _MultiCropLoader(video_path, crop_boxes, frame_start, frame_end)
        self._prefetch_cache[key] = loader
        loader.start()

    def seek(self, frame: int) -> None:
        """Seek to a specific frame."""
        frame = max(self._frame_start, min(self._frame_end, frame))
        self._current_frame = frame
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._render_current()
        self._update_frame_label()
        self.frame_changed.emit(frame)

    def stop(self) -> None:
        """Stop playback."""
        self._playing = False
        self._play_timer.stop()
        self._btn_play.setText("▶")

    def toggle_play(self) -> None:
        """Toggle play/pause."""
        if self._playing:
            self.stop()
        else:
            self._playing = True
            self._btn_play.setText("❚❚")
            self._play_timer.start()

    def panel(self, idx: int) -> InteractiveCanvas:
        return self._panels[idx]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider(self, value: int) -> None:
        self._current_frame = value
        self._render_current()
        self._update_frame_label()
        self.frame_changed.emit(value)

    def step_forward(self) -> None:
        """Step one frame forward."""
        if self._current_frame < self._frame_end:
            self.seek(self._current_frame + 1)

    def step_back(self) -> None:
        """Step one frame backward."""
        if self._current_frame > self._frame_start:
            self.seek(self._current_frame - 1)

    def _on_play_tick(self) -> None:
        if self._current_frame >= self._frame_end:
            self.seek(self._frame_start)  # loop
        else:
            self.seek(self._current_frame + 1)

    def _mirror_zoom(self, factor: float) -> None:
        """Mirror zoom change across all visible panels."""
        for panel in self._panels:
            if panel.isVisible() and abs(panel.zoom - factor) > 0.01:
                panel.set_zoom(factor)

    def _render_current(self) -> None:
        """Render current frame to all visible panels."""
        for i, panel in enumerate(self._panels):
            if not panel.isVisible():
                continue
            panel_frames = self._frames.get(i, {})
            bgr = panel_frames.get(self._current_frame)
            if bgr is None:
                continue
            frame = bgr.copy()
            if self._overlay_fns[i] is not None:
                frame = self._overlay_fns[i](frame, self._current_frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            panel.set_pixmap(QPixmap.fromImage(qimg))

    def _update_frame_label(self) -> None:
        pos = self._current_frame - self._frame_start
        total = max(1, self._frame_end - self._frame_start)
        self._frame_label.setText(f"{pos} / {total}")

    def _on_load_finished(self) -> None:
        """Show panels, fit to view, re-enable transport, and optionally auto-play."""
        # Reveal panels now that frames are available
        for i, panel in enumerate(self._panels):
            panel.setVisible(i < self._n_active_panels)

        # Hide loading indicator and re-enable transport
        self._loading_label.setVisible(False)
        self._btn_play.setEnabled(True)
        self._btn_prev.setEnabled(True)
        self._btn_next.setEnabled(True)
        self._slider.setEnabled(True)

        # Reset to the beginning of the clip
        self.seek(self._frame_start)

        # Fit each visible panel to its content
        for panel in self._panels:
            if panel.isVisible():
                panel.fit()

        # Auto-play if requested
        if self._auto_play:
            self.toggle_play()
        logger.debug(
            "SyncedVideoGrid loaded frames %d–%d",
            self._frame_start,
            self._frame_end,
        )

    def cleanup(self) -> None:
        """Stop loader and release memory."""
        self.stop()
        if self._loader is not None and self._loader.isRunning():
            self._loader.requestInterruption()
            self._loader.wait(2000)
        for loader in self._prefetch_cache.values():
            if loader.isRunning():
                loader.requestInterruption()
                loader.wait(500)
        self._prefetch_cache.clear()
        self._frames.clear()
