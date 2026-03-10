"""Main window for MAT-afterhours.

Provides a tabbed interface for reviewing and correcting identity swaps
detected in multi-animal tracking trajectories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from multi_tracker.afterhours.core.confidence_density import load_regions
from multi_tracker.afterhours.core.correction_writer import CorrectionWriter
from multi_tracker.afterhours.core.swap_scorer import SwapScorer, SwapSuspicionEvent
from multi_tracker.afterhours.gui.widgets.suspicion_queue import SuspicionQueueWidget
from multi_tracker.afterhours.gui.widgets.timeline_panel import TimelinePanelWidget
from multi_tracker.afterhours.gui.widgets.video_player import VideoPlayerWidget

logger = logging.getLogger(__name__)

_VIDEO_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*)"


class MainWindow(QMainWindow):
    """MAT-afterhours main window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MAT Afterhours")

        self._sessions: List[str] = []
        self._session_idx: int = -1
        self._video_path: Optional[str] = None
        self._csv_path: Optional[Path] = None
        self._writer: Optional[CorrectionWriter] = None
        self._df: Optional[pd.DataFrame] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # --- Top bar: session navigator ---
        nav_bar = QHBoxLayout()

        self._btn_prev = QPushButton("<")
        self._btn_prev.setFixedWidth(30)
        self._btn_prev.clicked.connect(self._prev_session)
        nav_bar.addWidget(self._btn_prev)

        self._session_label = QLabel("No session loaded")
        self._session_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_bar.addWidget(self._session_label, stretch=1)

        self._btn_next = QPushButton(">")
        self._btn_next.setFixedWidth(30)
        self._btn_next.clicked.connect(self._next_session)
        nav_bar.addWidget(self._btn_next)

        self._btn_load = QPushButton("Load...")
        self._btn_load.clicked.connect(self._load_session)
        nav_bar.addWidget(self._btn_load)

        root.addLayout(nav_bar)

        # --- Tab bar ---
        self._tabs = QTabWidget()

        # Swap Review tab
        swap_tab = QWidget()
        swap_layout = QHBoxLayout(swap_tab)
        swap_layout.setContentsMargins(0, 0, 0, 0)

        hsplitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: suspicion queue
        self._queue = SuspicionQueueWidget()
        self._queue.event_selected.connect(self._on_event_selected)
        self._queue.setMinimumWidth(260)
        self._queue.setMaximumWidth(400)
        hsplitter.addWidget(self._queue)

        # Right: video + timeline in vertical splitter
        vsplitter = QSplitter(Qt.Orientation.Vertical)
        self._player = VideoPlayerWidget()
        vsplitter.addWidget(self._player)

        self._timeline = TimelinePanelWidget()
        self._timeline.split_requested.connect(self._on_manual_split)
        self._timeline.setMaximumHeight(200)
        vsplitter.addWidget(self._timeline)

        vsplitter.setStretchFactor(0, 3)
        vsplitter.setStretchFactor(1, 1)
        hsplitter.addWidget(vsplitter)

        hsplitter.setStretchFactor(0, 0)
        hsplitter.setStretchFactor(1, 1)
        swap_layout.addWidget(hsplitter)

        self._tabs.addTab(swap_tab, "Swap Review")

        # Placeholder tabs
        self._tabs.addTab(QLabel("Merge review — coming soon"), "Merge Review")
        self._tabs.addTab(QLabel("Manual edit — coming soon"), "Manual Edit")

        root.addWidget(self._tabs)
        self._update_nav_state()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_session(self) -> None:
        """Ask user to load a single video or a text list of videos."""
        choice = QMessageBox.question(
            self,
            "Load session",
            "Load a single video file?\n\n"
            'Click "Yes" for a single video, "No" for a .txt list of videos.',
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._load_single_video()
        elif choice == QMessageBox.StandardButton.No:
            self._load_video_list()

    def _load_single_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", _VIDEO_FILTER)
        if path:
            self._sessions = [path]
            self._session_idx = 0
            self._open_current_session()

    def _load_video_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video list", "", "Text files (*.txt);;All files (*)"
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

        if not lines:
            QMessageBox.warning(self, "Empty list", "No video paths found.")
            return

        self._sessions = lines
        self._session_idx = 0
        self._open_current_session()

    def _prev_session(self) -> None:
        if self._session_idx > 0:
            self._session_idx -= 1
            self._open_current_session()

    def _next_session(self) -> None:
        if self._session_idx < len(self._sessions) - 1:
            self._session_idx += 1
            self._open_current_session()

    def _update_nav_state(self) -> None:
        has = len(self._sessions) > 0
        self._btn_prev.setEnabled(has and self._session_idx > 0)
        self._btn_next.setEnabled(has and self._session_idx < len(self._sessions) - 1)
        if has:
            name = Path(self._sessions[self._session_idx]).stem
            self._session_label.setText(
                f"{name} ({self._session_idx + 1}/{len(self._sessions)})"
            )
        else:
            self._session_label.setText("No session loaded")

    def _open_current_session(self) -> None:
        """Open the video, discover CSV, create writer, load data, score."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        video_path = self._sessions[self._session_idx]
        self._video_path = video_path

        csv_path = self._discover_csv(video_path)
        if csv_path is None:
            QMessageBox.warning(
                self,
                "CSV not found",
                f"Could not find a tracking CSV for:\n{video_path}",
            )
            self._update_nav_state()
            return

        self._csv_path = csv_path
        self._writer = CorrectionWriter(csv_path)
        self._writer.open()
        self._df = self._writer.df

        self._player.load_video(video_path)
        self._player.load_trajectories(self._df)
        self._timeline.load_trajectories(self._df)

        self._run_scorer()
        self._update_nav_state()
        logger.info(
            "Opened session %d: %s  CSV=%s",
            self._session_idx,
            video_path,
            csv_path,
        )

    @staticmethod
    def _discover_csv(video_path: str) -> Optional[Path]:
        """Find matching CSV for a video by trying common suffixes."""
        vp = Path(video_path)
        stem = vp.stem
        parent = vp.parent

        for suffix in ("_with_pose.csv", "_tracked.csv", ".csv"):
            candidate = parent / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

        return None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _run_scorer(self) -> None:
        """Load regions (if available) and run the swap scorer."""
        if self._df is None or self._video_path is None:
            return

        regions = []
        vp = Path(self._video_path)
        regions_path = vp.parent / f"{vp.stem}_density_regions.json"
        if regions_path.exists():
            try:
                regions = load_regions(regions_path)
            except Exception:
                logger.warning("Failed to load density regions from %s", regions_path)

        scorer = SwapScorer(regions=regions)
        events = scorer.score(self._df)

        self._queue.populate(events)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event_selected(self, event: SwapSuspicionEvent) -> None:
        """Seek video, highlight tracks, show review dialogs."""
        self._player.seek_to(event.frame_peak)
        highlight = [event.track_a]
        if event.track_b is not None:
            highlight.append(event.track_b)
        self._player.highlight_tracks(highlight)
        self._timeline.highlight_event(event)

        self._show_review_dialogs(event)

    def _show_review_dialogs(self, event: SwapSuspicionEvent) -> None:
        """Open FramePickerDialog then IdentityAssignmentDialog."""
        if self._video_path is None or self._df is None or event.track_b is None:
            return

        # Lazy import to avoid circular dependencies.
        from multi_tracker.afterhours.gui.dialogs.frame_picker import FramePickerDialog

        picker = FramePickerDialog(
            video_path=self._video_path,
            df=self._df,
            track_a=event.track_a,
            track_b=event.track_b,
            frame_range=event.frame_range,
            parent=self,
        )
        if picker.exec() != FramePickerDialog.DialogCode.Accepted:
            return

        split_frame = picker.selected_frame()

        from multi_tracker.afterhours.gui.dialogs.identity_assignment import (
            IdentityAssignmentDialog,
        )

        assigner = IdentityAssignmentDialog(
            video_path=self._video_path,
            df=self._df,
            track_a=event.track_a,
            track_b=event.track_b,
            split_frame=split_frame,
            parent=self,
        )
        if assigner.exec() != IdentityAssignmentDialog.DialogCode.Accepted:
            return

        swap = assigner.should_swap()

        # Apply correction
        if self._writer is not None:
            self._writer.apply_correction(
                track_a=event.track_a,
                track_b=event.track_b,
                split_frame=split_frame,
                swap_post=swap,
            )
            self._df = self._writer.df
            self._player.load_trajectories(self._df)
            self._timeline.load_trajectories(self._df)
            self._queue.mark_resolved(event)

    def _on_manual_split(self, track_id: int, frame: int) -> None:
        """Handle a manual split request from the timeline."""
        logger.info("Manual split requested: track %d at frame %d", track_id, frame)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        super().closeEvent(event)
