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
from multi_tracker.afterhours.core.event_scorer import EventScorer
from multi_tracker.afterhours.core.event_types import EventType, SuspicionEvent
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
        self.apply_stylesheet()
        self.statusBar().showMessage("MAT Afterhours — ready", 4000)

    # ------------------------------------------------------------------
    # Stylesheet
    # ------------------------------------------------------------------

    def apply_stylesheet(self) -> None:
        """Apply the MAT dark theme to the entire window (matches MAT / PoseKit / ClassKit)."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, Arial, sans-serif;
                font-size: 11px;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px;
                font-weight: 600;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                background-color: #1e1e1e;
                color: #9cdcfe;
                border-radius: 3px;
            }
            QListWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
                outline: none;
            }
            QListWidget::item {
                padding: 6px 10px;
                border-radius: 3px;
                margin: 1px 0px;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background-color: #2a2d2e;
            }
            QPushButton {
                background-color: #0e639c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3e3e42;
                color: #777777;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QComboBox:hover { border-color: #0e639c; }
            QComboBox:focus { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #252526;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
                selection-color: #ffffff;
                outline: none;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 8px;
                min-height: 22px;
            }
            QLineEdit:hover { border-color: #0e639c; }
            QLineEdit:focus { border-color: #007acc; }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px 4px 4px 8px;
                min-height: 22px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border-color: #0e639c; }
            QSpinBox:focus, QDoubleSpinBox:focus { border-color: #007acc; }
            QCheckBox { color: #cccccc; spacing: 8px; }
            QCheckBox::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border-color: #007acc;
            }
            QRadioButton { color: #cccccc; spacing: 8px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background-color: #3c3c3c;
            }
            QRadioButton::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QToolBar {
                background-color: #252526;
                border-bottom: 1px solid #3e3e42;
                spacing: 6px;
                padding: 4px 6px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 6px 10px;
                color: #cccccc;
            }
            QToolButton:hover { background-color: #2a2d2e; }
            QToolButton:pressed { background-color: #094771; }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                border-radius: 0px;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-bottom: none;
                padding: 6px 16px;
                min-width: 80px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                color: #ffffff;
                border-top: 2px solid #007acc;
            }
            QTabBar::tab:hover:!selected { background-color: #2a2d2e; }
            QStatusBar {
                background-color: #007acc;
                color: #ffffff;
                border-top: 1px solid #0098ff;
                font-weight: 500;
                font-size: 12px;
            }
            QStatusBar QLabel {
                background-color: transparent;
                color: #ffffff;
                padding: 0px 4px;
            }
            QMenuBar {
                background-color: #252526;
                color: #cccccc;
                border-bottom: 1px solid #3e3e42;
                padding: 2px;
            }
            QMenuBar::item { padding: 5px 10px; background-color: transparent; border-radius: 3px; }
            QMenuBar::item:selected { background-color: #2a2d2e; }
            QMenuBar::item:pressed { background-color: #094771; }
            QMenu {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 4px;
            }
            QMenu::item { padding: 6px 20px 6px 12px; border-radius: 3px; }
            QMenu::item:selected { background-color: #094771; color: #ffffff; }
            QMenu::separator { height: 1px; background-color: #3e3e42; margin: 4px 8px; }
            QSplitter::handle { background-color: #3e3e42; }
            QSplitter::handle:hover { background-color: #007acc; }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical {
                background-color: #252526;
                width: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover { background-color: #007acc; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar:horizontal {
                background-color: #252526;
                height: 10px;
                border-radius: 5px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 5px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover { background-color: #007acc; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                text-align: center;
                background-color: #252526;
                color: #cccccc;
                font-size: 11px;
            }
            QProgressBar::chunk { background-color: #0e639c; border-radius: 3px; }
            QSlider::groove:horizontal {
                height: 4px;
                background-color: #3e3e42;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background-color: #007acc;
                border: none;
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QSlider::handle:horizontal:hover { background-color: #1177bb; }
            QSlider::sub-page:horizontal { background-color: #007acc; border-radius: 2px; }
            QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #3e3e42; }
        """)

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
        for suffix in ("tracking_final_with_pose.csv", "tracking_final.csv"):
            candidate = parent / f"{stem}_{suffix}"
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

        scorer = EventScorer(regions=regions)
        events = scorer.score_all(self._df)

        self._queue.populate(events)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _on_event_selected(self, event: SuspicionEvent) -> None:
        """Seek video, highlight tracks, show resolution dialog."""
        self._player.seek_to(event.frame_peak)
        self._player.highlight_tracks(event.involved_tracks)
        self._timeline.highlight_event(event)

        self._show_resolution_dialog(event)

    def _show_resolution_dialog(self, event: SuspicionEvent) -> None:
        """Open the unified ResolutionDialog and apply the chosen correction."""
        if self._video_path is None or self._df is None:
            return

        from multi_tracker.afterhours.gui.dialogs.resolution_dialog import (
            ResolutionDialog,
        )

        dlg = ResolutionDialog(
            video_path=self._video_path,
            df=self._df,
            event=event,
            parent=self,
        )
        if dlg.exec() != ResolutionDialog.DialogCode.Accepted:
            return

        action = dlg.selected_action()
        etype = dlg.effective_event_type()
        split_frame = dlg.selected_frame()
        resolved_event = dlg.event

        if action == "Skip (no correction)":
            return

        if self._writer is None:
            return

        if etype == EventType.SWAP:
            swap_post = action == "Swap IDs at split frame"
            if event.track_b is not None:
                self._writer.apply_correction(
                    track_a=event.track_a,
                    track_b=event.track_b,
                    split_frame=split_frame,
                    swap_post=swap_post,
                )
        elif etype == EventType.FLICKER:
            if event.track_b is not None:
                self._writer.apply_erase_flicker(
                    track_a=event.track_a,
                    track_b=event.track_b,
                    frame_start=event.frame_range[0],
                    frame_end=event.frame_range[1],
                )
        elif etype == EventType.FRAGMENTATION:
            self._writer.apply_merge(event.involved_tracks)
        elif etype == EventType.ABSORPTION:
            swap_post = action == "Split + swap at re-appearance"
            if event.track_b is not None:
                self._writer.apply_correction(
                    track_a=event.track_a,
                    track_b=event.track_b,
                    split_frame=split_frame,
                    swap_post=swap_post,
                )
        elif etype == EventType.PHANTOM:
            frame_range = event.frame_range if "range only" in action else None
            self._writer.apply_delete(event.track_a, frame_range=frame_range)
        elif etype == EventType.MULTI_SHUFFLE:
            # For now treat as pairwise swap on first two tracks
            if event.track_b is not None:
                swap_post = "Swap" in action
                self._writer.apply_correction(
                    track_a=event.track_a,
                    track_b=event.track_b,
                    split_frame=split_frame,
                    swap_post=swap_post,
                )

        # Refresh state
        self._df = self._writer.df
        self._player.load_trajectories(self._df)
        self._timeline.load_trajectories(self._df)
        self._queue.mark_resolved(resolved_event)

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
