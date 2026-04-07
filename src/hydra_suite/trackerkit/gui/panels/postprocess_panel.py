"""PostProcessPanel — trajectory cleaning, relinking, and interpolation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class PostProcessPanel(QWidget):
    """Trajectory post-processing: cleaning, velocity breaks, and interpolation."""

    config_changed: Signal = Signal(object)

    def __init__(
        self,
        main_window: "MainWindow",
        config: TrackerConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._main_window = main_window
        self._config = config
        self._layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self) -> None:
        """Populate the panel layout."""
        layout = self._layout
        layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)
        self._main_window._set_compact_scroll_layout(vbox)

        # Post-Processing
        g_pp = QGroupBox("How should tracks be cleaned after tracking?")
        self._main_window._set_compact_section_widget(g_pp)
        vl_pp = QVBoxLayout(g_pp)
        vl_pp.addWidget(
            self._main_window._create_help_label(
                "Clean trajectories after tracking by removing outliers and splitting at identity swaps. "
                "Velocity/distance breaks detect unrealistic jumps that indicate ID switching."
            )
        )
        f_pp = QFormLayout(None)
        f_pp.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.enable_postprocessing = QCheckBox("Auto-clean trajectories")
        self.enable_postprocessing.setChecked(True)
        self.enable_postprocessing.setToolTip(
            "Automatically clean trajectories by removing outliers and fragments.\n"
            "Uses velocity and distance thresholds to detect anomalies.\n"
            "Recommended: Enable for cleaner data output."
        )
        self.enable_postprocessing.stateChanged.connect(self._on_cleaning_toggled)
        f_pp.addRow(self.enable_postprocessing)

        self.spin_min_trajectory_length = QDoubleSpinBox()
        self.spin_min_trajectory_length.setRange(0.01, 60.0)
        self.spin_min_trajectory_length.setSingleStep(0.1)
        self.spin_min_trajectory_length.setDecimals(2)
        self.spin_min_trajectory_length.setValue(0.33)
        self.spin_min_trajectory_length.setToolTip(
            "Remove trajectories shorter than this (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Filters out brief false detections and transient tracks.\n"
            "Recommended: 0.15-1.0 s depending on video length."
        )
        self.lbl_min_trajectory_length = QLabel("Minimum trajectory length (seconds)")
        f_pp.addRow(self.lbl_min_trajectory_length, self.spin_min_trajectory_length)

        self.spin_max_velocity_break = QDoubleSpinBox()
        self.spin_max_velocity_break.setRange(1.0, 500.0)
        self.spin_max_velocity_break.setSingleStep(5.0)
        self.spin_max_velocity_break.setDecimals(1)
        self.spin_max_velocity_break.setValue(50.0)
        self.spin_max_velocity_break.setToolTip(
            "Maximum velocity (body-sizes/second) before breaking trajectory.\n"
            "Splits tracks at unrealistic speed jumps (likely identity swaps).\n"
            "Independent of frame rate - automatically scaled by FPS.\n"
            "Recommended: 30-100 body-sizes/s for typical animal motion."
        )
        self.lbl_max_velocity_break = QLabel(
            "Break trajectory above speed (body lengths/sec)"
        )
        f_pp.addRow(self.lbl_max_velocity_break, self.spin_max_velocity_break)

        self.spin_max_occlusion_gap = QDoubleSpinBox()
        self.spin_max_occlusion_gap.setRange(0.0, 10.0)
        self.spin_max_occlusion_gap.setSingleStep(0.1)
        self.spin_max_occlusion_gap.setDecimals(2)
        self.spin_max_occlusion_gap.setValue(1.0)
        self.spin_max_occlusion_gap.setToolTip(
            "Maximum occlusion duration before splitting trajectory (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Prevents unreliable interpolation across long gaps.\n"
            "Set to 0 to disable occlusion-based splitting.\n"
            "Recommended: 0.5-2.0 s for typical tracking scenarios."
        )
        self.lbl_max_occlusion_gap = QLabel("Maximum occlusion gap (seconds)")
        f_pp.addRow(self.lbl_max_occlusion_gap, self.spin_max_occlusion_gap)

        self.chk_enable_tracklet_relinking = QCheckBox(
            "Relink fragments after pose interpolation"
        )
        self.chk_enable_tracklet_relinking.setChecked(False)
        self.chk_enable_tracklet_relinking.setToolTip(
            "\u26a0 USE WITH CAUTION \u2014 disabled by default.\n"
            "\n"
            "Reconnect short trajectory fragments after pose/interpolation completes.\n"
            "In dense multi-animal scenes this can cause identity swaps by incorrectly\n"
            "merging fragments from different animals into one trajectory.\n"
            "\n"
            "Bidirectional tracking (forward + backward pass) already handles most\n"
            "occlusion recovery. Enable relinking only if you see fragmented trajectories\n"
            "that bidirectional tracking could not repair, and verify results carefully."
        )
        self.lbl_enable_tracklet_relinking = QLabel("Enable relinking")
        f_pp.addRow(
            self.lbl_enable_tracklet_relinking, self.chk_enable_tracklet_relinking
        )

        self.spin_relink_pose_max_distance = QDoubleSpinBox()
        self.spin_relink_pose_max_distance.setRange(0.05, 5.0)
        self.spin_relink_pose_max_distance.setSingleStep(0.05)
        self.spin_relink_pose_max_distance.setDecimals(2)
        self.spin_relink_pose_max_distance.setValue(0.45)
        self.spin_relink_pose_max_distance.setToolTip(
            "Maximum normalized same-keypoint pose distance allowed when relinking fragments.\n"
            "Lower values are stricter and reduce risky merges."
        )
        self.lbl_relink_pose_max_distance = QLabel("Max normalized pose distance")
        f_pp.addRow(
            self.lbl_relink_pose_max_distance, self.spin_relink_pose_max_distance
        )

        self.spin_pose_export_min_valid_fraction = QDoubleSpinBox()
        self.spin_pose_export_min_valid_fraction.setRange(0.0, 1.0)
        self.spin_pose_export_min_valid_fraction.setSingleStep(0.05)
        self.spin_pose_export_min_valid_fraction.setDecimals(2)
        self.spin_pose_export_min_valid_fraction.setValue(0.5)
        self.spin_pose_export_min_valid_fraction.setToolTip(
            "Minimum fraction of valid keypoints required for a pose row to be accepted.\n"
            "Rows below this threshold have all confidence values zeroed but are kept\n"
            "in the output with quality flags. Only active when pose extraction is enabled.\n"
            "Range: 0.0\u20131.0. Recommended: 0.3\u20130.6."
        )
        self.lbl_pose_export_min_valid_fraction = QLabel("Min valid keypoint fraction")
        f_pp.addRow(
            self.lbl_pose_export_min_valid_fraction,
            self.spin_pose_export_min_valid_fraction,
        )

        self.spin_pose_export_min_valid_keypoints = QSpinBox()
        self.spin_pose_export_min_valid_keypoints.setRange(1, 50)
        self.spin_pose_export_min_valid_keypoints.setValue(3)
        self.spin_pose_export_min_valid_keypoints.setToolTip(
            "Minimum absolute count of valid keypoints required for a pose row to be accepted.\n"
            "Both this threshold and the fraction threshold must be satisfied."
        )
        self.lbl_pose_export_min_valid_keypoints = QLabel("Min valid keypoints (count)")
        f_pp.addRow(
            self.lbl_pose_export_min_valid_keypoints,
            self.spin_pose_export_min_valid_keypoints,
        )

        self.spin_relink_min_pose_quality = QDoubleSpinBox()
        self.spin_relink_min_pose_quality.setRange(0.0, 1.0)
        self.spin_relink_min_pose_quality.setSingleStep(0.05)
        self.spin_relink_min_pose_quality.setDecimals(2)
        self.spin_relink_min_pose_quality.setValue(0.6)
        self.spin_relink_min_pose_quality.setToolTip(
            "Minimum pose quality score required at both endpoints of a candidate\n"
            "fragment pair before pose distance contributes to relinking score.\n"
            "Below this threshold, relinking uses motion-only scoring.\n"
            "Higher values (0.6+) are strongly recommended \u2014 motion-only relinking\n"
            "in dense scenes is likely to merge fragments from different animals."
        )
        self.lbl_relink_min_pose_quality = QLabel("Min pose quality for relinking")
        f_pp.addRow(
            self.lbl_relink_min_pose_quality,
            self.spin_relink_min_pose_quality,
        )

        self.spin_pose_postproc_max_gap = QSpinBox()
        self.spin_pose_postproc_max_gap.setRange(0, 50)
        self.spin_pose_postproc_max_gap.setValue(5)
        self.spin_pose_postproc_max_gap.setToolTip(
            "Maximum consecutive missing frames to gap-fill via linear interpolation\n"
            "during pose temporal post-processing. Set to 0 to disable gap-filling."
        )
        self.lbl_pose_postproc_max_gap = QLabel("Pose postproc max gap (frames)")
        f_pp.addRow(
            self.lbl_pose_postproc_max_gap,
            self.spin_pose_postproc_max_gap,
        )

        self.spin_pose_temporal_outlier_zscore = QDoubleSpinBox()
        self.spin_pose_temporal_outlier_zscore.setRange(0.0, 20.0)
        self.spin_pose_temporal_outlier_zscore.setSingleStep(0.5)
        self.spin_pose_temporal_outlier_zscore.setDecimals(1)
        self.spin_pose_temporal_outlier_zscore.setValue(3.0)
        self.spin_pose_temporal_outlier_zscore.setToolTip(
            "Rolling z-score threshold for temporal keypoint outlier suppression.\n"
            "Keypoint positions deviating beyond this threshold from their local\n"
            "rolling mean are zeroed. Set to 0.0 to disable. Recommended: 2.5\u20135.0."
        )
        self.lbl_pose_temporal_outlier_zscore = QLabel("Pose temporal outlier z-score")
        f_pp.addRow(
            self.lbl_pose_temporal_outlier_zscore,
            self.spin_pose_temporal_outlier_zscore,
        )

        # Z-score based velocity breaking
        self.spin_max_velocity_zscore = QDoubleSpinBox()
        self.spin_max_velocity_zscore.setRange(0.0, 10.0)
        self.spin_max_velocity_zscore.setSingleStep(0.5)
        self.spin_max_velocity_zscore.setDecimals(1)
        self.spin_max_velocity_zscore.setValue(0.0)  # 0 = disabled
        self.spin_max_velocity_zscore.setToolTip(
            "Z-score threshold for velocity-based trajectory breaking (0 = disabled).\n"
            "Detects sudden, statistically anomalous velocity changes that often\n"
            "indicate identity swaps or tracking errors.\n\n"
            "Safeguards prevent false breaks when animals transition from rest to movement:\n"
            "\u2022 Only triggers on substantial velocities (>2 px/frame)\n"
            "\u2022 Uses regularized statistics to handle low-variability periods\n"
            "\u2022 Filters out stationary noise from baseline calculations\n\n"
            "Recommended: 3.0-5.0 for sensitive detection, 0 to disable."
        )
        self.lbl_max_velocity_zscore = QLabel("Velocity z-score threshold")
        f_pp.addRow(self.lbl_max_velocity_zscore, self.spin_max_velocity_zscore)

        self.spin_velocity_zscore_window = QDoubleSpinBox()
        self.spin_velocity_zscore_window.setRange(0.1, 5.0)
        self.spin_velocity_zscore_window.setSingleStep(0.1)
        self.spin_velocity_zscore_window.setDecimals(2)
        self.spin_velocity_zscore_window.setValue(0.33)
        self.spin_velocity_zscore_window.setToolTip(
            "Time window for z-score velocity calculation (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Larger windows = more stable statistics but less responsive.\n"
            "Smaller windows = more sensitive but may be noisy.\n"
            "Recommended: 0.3-0.7 s."
        )
        self.lbl_velocity_zscore_window = QLabel("Z-score window (seconds)")
        f_pp.addRow(self.lbl_velocity_zscore_window, self.spin_velocity_zscore_window)

        self.spin_velocity_zscore_min_vel = QDoubleSpinBox()
        self.spin_velocity_zscore_min_vel.setRange(0.1, 50.0)
        self.spin_velocity_zscore_min_vel.setSingleStep(0.5)
        self.spin_velocity_zscore_min_vel.setDecimals(1)
        self.spin_velocity_zscore_min_vel.setValue(2.0)
        self.spin_velocity_zscore_min_vel.setToolTip(
            "Minimum velocity for z-score breaking (body-sizes/second).\n"
            "Prevents false breaks when animal starts moving from stationary state.\n"
            "Z-score analysis only triggers when velocity exceeds this threshold.\n"
            "Automatically scaled by body size and frame rate.\n"
            "Recommended: 1.0-3.0 body-sizes/s depending on animal locomotion speed."
        )
        self.lbl_velocity_zscore_min_vel = QLabel(
            "Minimum speed for z-score check (body lengths/sec)"
        )
        f_pp.addRow(self.lbl_velocity_zscore_min_vel, self.spin_velocity_zscore_min_vel)

        # Interpolation settings
        self.combo_interpolation_method = QComboBox()
        self.combo_interpolation_method.addItems(["None", "Linear", "Cubic", "Spline"])
        self.combo_interpolation_method.setCurrentText("None")
        self.combo_interpolation_method.setToolTip(
            "Interpolation method for filling gaps in trajectories:\n"
            "\u2022 None: No interpolation (keep NaN values)\n"
            "\u2022 Linear: Simple linear interpolation\n"
            "\u2022 Cubic: Smooth cubic spline interpolation\n"
            "\u2022 Spline: Smoothing spline with automatic smoothing\n"
            "Applied to X, Y positions and heading (circular interpolation)."
        )
        self.lbl_interpolation_method = QLabel(
            "Which interpolation method should be used?"
        )
        f_pp.addRow(self.lbl_interpolation_method, self.combo_interpolation_method)

        self.spin_interpolation_max_gap = QDoubleSpinBox()
        self.spin_interpolation_max_gap.setRange(0.01, 10.0)
        self.spin_interpolation_max_gap.setSingleStep(0.1)
        self.spin_interpolation_max_gap.setDecimals(2)
        self.spin_interpolation_max_gap.setValue(0.33)
        self.spin_interpolation_max_gap.setToolTip(
            "Maximum gap duration to interpolate (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Gaps larger than this will remain as NaN.\n"
            "Prevents interpolation across large occlusions.\n"
            "Recommended: 0.15-0.50 s."
        )
        self.lbl_interpolation_max_gap = QLabel("Maximum interpolation gap (seconds)")
        f_pp.addRow(self.lbl_interpolation_max_gap, self.spin_interpolation_max_gap)

        self.spin_heading_flip_max_burst = QSpinBox()
        self.spin_heading_flip_max_burst.setRange(1, 50)
        self.spin_heading_flip_max_burst.setValue(5)
        self.spin_heading_flip_max_burst.setToolTip(
            "Maximum length (frames) of an isolated heading-flip burst that\n"
            "post-processing will correct. Contiguous runs of ~180\u00b0 flips\n"
            "shorter than this are assumed to be classifier errors and are\n"
            "reverted. Longer runs are kept as genuine orientation changes.\n"
            "Increase if brief real flips are being suppressed; decrease if\n"
            "extended flip artefacts survive. Recommended: 3\u201310."
        )
        self.lbl_heading_flip_max_burst = QLabel(
            "Max heading-flip burst to correct (frames)"
        )
        f_pp.addRow(self.lbl_heading_flip_max_burst, self.spin_heading_flip_max_burst)

        # Trajectory Merging Settings (Conservative Strategy)
        self.spin_merge_overlap_multiplier = QDoubleSpinBox()
        self.spin_merge_overlap_multiplier.setRange(0.1, 10.0)
        self.spin_merge_overlap_multiplier.setSingleStep(0.1)
        self.spin_merge_overlap_multiplier.setDecimals(2)
        self.spin_merge_overlap_multiplier.setValue(0.5)
        self.spin_merge_overlap_multiplier.setToolTip(
            "Agreement distance for merging forward/backward trajectories (\u00d7body size).\n"
            "Frames where both trajectories are within this distance are considered 'agreeing'.\n"
            "Disagreeing frames cause trajectory splits for conservative identity handling.\n"
            "Recommended: 0.3-0.7\u00d7 body size."
        )
        self.lbl_merge_overlap_multiplier = QLabel(
            "Merge agreement distance (body lengths)"
        )
        f_pp.addRow(
            self.lbl_merge_overlap_multiplier, self.spin_merge_overlap_multiplier
        )

        self.spin_min_overlap_frames = QSpinBox()
        self.spin_min_overlap_frames.setRange(1, 100)
        self.spin_min_overlap_frames.setValue(5)
        self.spin_min_overlap_frames.setToolTip(
            "Minimum agreeing frames required to consider trajectories as merge candidates.\n"
            "Forward/backward trajectory pairs need at least this many frames within\n"
            "the agreement distance to be merged. Higher = more conservative.\n"
            "Recommended: 5-15 frames."
        )
        self.lbl_min_overlap_frames = QLabel("Minimum overlap frames")
        f_pp.addRow(self.lbl_min_overlap_frames, self.spin_min_overlap_frames)

        # Cleanup option
        self.chk_cleanup_temp_files = QCheckBox("Auto-cleanup temporary files")
        self.chk_cleanup_temp_files.setChecked(True)
        self.chk_cleanup_temp_files.setToolTip(
            "Automatically delete temporary files after successful tracking:\n"
            "\u2022 Intermediate CSV files (*_forward.csv, *_backward.csv)\n"
            "\u2022 Pose inference cache (posekit/ directory)\n"
            "Keeps only final merged/processed output files."
        )
        f_pp.addRow("", self.chk_cleanup_temp_files)

        vl_pp.addLayout(f_pp)
        vbox.addWidget(g_pp)

        # Video Export (from post-processed trajectories)
        g_video = QGroupBox("What export video should be created?")
        self._main_window._set_compact_section_widget(g_video)
        vl_video = QVBoxLayout(g_video)
        vl_video.addWidget(
            self._main_window._create_help_label(
                "Generate annotated video from final post-processed trajectories. "
                "Video is created AFTER merging and interpolation, showing clean tracks with stable IDs. "
                "This is recommended over real-time video output during tracking."
            )
        )
        f_video = QFormLayout(None)
        f_video.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.check_video_output = QCheckBox("Export trajectory video")
        self.check_video_output.setChecked(False)
        self.check_video_output.toggled.connect(self._on_video_output_toggled)
        self.check_video_output.setToolTip(
            "Generate annotated video showing post-processed trajectories.\n"
            "Video is created from merged/interpolated tracks, not raw tracking.\n"
            "Shows clean, stable trajectories with final IDs.\n"
            "Recommended for publication and visualization."
        )
        f_video.addRow("", self.check_video_output)

        self.btn_video_out = QPushButton("Select Video Output...")
        self.btn_video_out.clicked.connect(self.select_video_output)
        self.btn_video_out.setEnabled(False)
        self.video_out_line = QLineEdit()
        self.video_out_line.setPlaceholderText("Path for annotated video output")
        self.video_out_line.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.video_out_line.setEnabled(False)
        self.lbl_video_path = QLabel("")
        f_video.addRow(self.lbl_video_path, self.btn_video_out)
        f_video.addRow("", self.video_out_line)

        # Video Visualization Settings
        f_video.addRow(QLabel(""))  # Spacer
        self.lbl_video_viz_settings = QLabel("<b>Visualization Settings</b>")
        f_video.addRow(self.lbl_video_viz_settings)

        self.check_show_labels = QCheckBox("Show Track IDs")
        self.check_show_labels.setChecked(True)
        self.check_show_labels.setToolTip(
            "Display trajectory ID labels next to each tracked animal."
        )
        f_video.addRow("", self.check_show_labels)

        self.check_show_orientation = QCheckBox("Show Orientation Arrows")
        self.check_show_orientation.setChecked(True)
        self.check_show_orientation.setToolTip(
            "Display arrows indicating heading direction."
        )
        f_video.addRow("", self.check_show_orientation)

        self.check_show_trails = QCheckBox("Show Trajectory Trails")
        self.check_show_trails.setChecked(False)
        self.check_show_trails.setToolTip(
            "Display past trajectory path as a fading trail."
        )
        f_video.addRow("", self.check_show_trails)

        self.spin_trail_duration = QDoubleSpinBox()
        self.spin_trail_duration.setRange(0.1, 10.0)
        self.spin_trail_duration.setSingleStep(0.5)
        self.spin_trail_duration.setDecimals(1)
        self.spin_trail_duration.setValue(1.0)
        self.spin_trail_duration.setToolTip(
            "Duration of trail history in seconds (0.1-10.0).\n"
            "Longer trails show more movement history.\n"
            "Automatically converted to frames using video FPS."
        )
        self.lbl_trail_duration = QLabel("Trail duration (seconds)")
        f_video.addRow(self.lbl_trail_duration, self.spin_trail_duration)

        self.spin_marker_size = QDoubleSpinBox()
        self.spin_marker_size.setRange(0.1, 300.0)
        self.spin_marker_size.setSingleStep(0.1)
        self.spin_marker_size.setDecimals(1)
        self.spin_marker_size.setValue(0.3)
        self.spin_marker_size.setToolTip(
            "Size of position marker (0.1-5.0 \u00d7 body size).\n"
            "Scaled by reference body size for consistency."
        )
        self.lbl_marker_size = QLabel("Marker size (body lengths)")
        f_video.addRow(self.lbl_marker_size, self.spin_marker_size)

        self.spin_text_scale = QDoubleSpinBox()
        self.spin_text_scale.setRange(0.3, 3.0)
        self.spin_text_scale.setSingleStep(0.1)
        self.spin_text_scale.setDecimals(1)
        self.spin_text_scale.setValue(0.5)
        self.spin_text_scale.setToolTip(
            "Scale factor for ID labels (0.3-3.0).\n" "Larger values = bigger text."
        )
        self.lbl_text_scale = QLabel("Text scale")
        f_video.addRow(self.lbl_text_scale, self.spin_text_scale)

        self.spin_arrow_length = QDoubleSpinBox()
        self.spin_arrow_length.setRange(0.5, 10.0)
        self.spin_arrow_length.setSingleStep(0.5)
        self.spin_arrow_length.setDecimals(1)
        self.spin_arrow_length.setValue(0.7)
        self.spin_arrow_length.setToolTip(
            "Length of orientation arrow (0.5-10.0 \u00d7 body size).\n"
            "Scaled by reference body size."
        )
        self.lbl_arrow_length = QLabel("Arrow length (body lengths)")
        f_video.addRow(self.lbl_arrow_length, self.spin_arrow_length)

        f_video.addRow(QLabel(""))  # Spacer
        self.lbl_video_pose_settings = QLabel("<b>Pose Overlay Settings</b>")
        f_video.addRow(self.lbl_video_pose_settings)

        self.check_video_show_pose = QCheckBox("Show Pose Keypoints/Skeleton")
        self.check_video_show_pose.setChecked(
            bool(self._main_window.advanced_config.get("video_show_pose", True))
        )
        self.check_video_show_pose.setToolTip(
            "Overlay pose keypoints/skeleton in exported video.\n"
            "Requires pose inference to be enabled in Analyze Individuals."
        )
        self.check_video_show_pose.toggled.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        f_video.addRow("", self.check_video_show_pose)

        self.combo_video_pose_color_mode = QComboBox()
        self.combo_video_pose_color_mode.addItems(["Track Color", "Fixed Color"])
        color_mode = str(
            self._main_window.advanced_config.get("video_pose_color_mode", "track")
        ).strip()
        self.combo_video_pose_color_mode.setCurrentIndex(
            0 if color_mode == "track" else 1
        )
        self.combo_video_pose_color_mode.setToolTip(
            "Pose color source for video overlay."
        )
        self.combo_video_pose_color_mode.currentIndexChanged.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        self.lbl_video_pose_color_mode = QLabel("Pose color mode")
        f_video.addRow(self.lbl_video_pose_color_mode, self.combo_video_pose_color_mode)

        pose_color_row = QHBoxLayout()
        self.btn_video_pose_color = QPushButton()
        self.btn_video_pose_color.setMaximumWidth(60)
        self.btn_video_pose_color.setMinimumHeight(28)
        self.btn_video_pose_color.clicked.connect(self._select_video_pose_color)
        self.lbl_video_pose_color = QLabel("")
        pose_color_row.addWidget(self.btn_video_pose_color)
        pose_color_row.addWidget(self.lbl_video_pose_color)
        pose_color_cfg = self._main_window.advanced_config.get(
            "video_pose_color", [255, 255, 255]
        )
        if isinstance(pose_color_cfg, (list, tuple)) and len(pose_color_cfg) == 3:
            self._video_pose_color = tuple(
                int(max(0, min(255, float(v)))) for v in pose_color_cfg
            )
        else:
            self._video_pose_color = (255, 255, 255)
        # Update button appearance inline (panel not yet assigned on main_window at this point)
        b, g, r = self._video_pose_color
        self.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self.lbl_video_pose_color.setText(f"{self._video_pose_color}")
        self.lbl_video_pose_color_label = QLabel("Fixed pose color (BGR)")
        f_video.addRow(self.lbl_video_pose_color_label, pose_color_row)

        self.spin_video_pose_point_radius = QSpinBox()
        self.spin_video_pose_point_radius.setRange(1, 20)
        self.spin_video_pose_point_radius.setValue(
            int(self._main_window.advanced_config.get("video_pose_point_radius", 3))
        )
        self.spin_video_pose_point_radius.setToolTip(
            "Radius of rendered pose keypoints in pixels."
        )
        self.lbl_video_pose_point_radius = QLabel("Pose keypoint radius (px)")
        f_video.addRow(
            self.lbl_video_pose_point_radius, self.spin_video_pose_point_radius
        )

        self.spin_video_pose_point_thickness = QSpinBox()
        self.spin_video_pose_point_thickness.setRange(-1, 10)
        self.spin_video_pose_point_thickness.setValue(
            int(self._main_window.advanced_config.get("video_pose_point_thickness", -1))
        )
        self.spin_video_pose_point_thickness.setToolTip(
            "Keypoint circle thickness (-1 fills circles)."
        )
        self.lbl_video_pose_point_thickness = QLabel("Pose keypoint thickness")
        f_video.addRow(
            self.lbl_video_pose_point_thickness, self.spin_video_pose_point_thickness
        )

        self.spin_video_pose_line_thickness = QSpinBox()
        self.spin_video_pose_line_thickness.setRange(1, 12)
        self.spin_video_pose_line_thickness.setValue(
            int(self._main_window.advanced_config.get("video_pose_line_thickness", 2))
        )
        self.spin_video_pose_line_thickness.setToolTip(
            "Skeleton edge line thickness in pixels."
        )
        self.lbl_video_pose_line_thickness = QLabel("Pose skeleton thickness (px)")
        f_video.addRow(
            self.lbl_video_pose_line_thickness, self.spin_video_pose_line_thickness
        )

        self.lbl_video_pose_disabled_hint = self._main_window._create_help_label(
            "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings.",
            attach_to_title=False,
        )
        f_video.addRow("", self.lbl_video_pose_disabled_hint)

        vl_video.addLayout(f_video)
        vbox.addWidget(g_video)

        # Set initial visibility for video export widgets (hidden since checkbox starts unchecked)
        self.btn_video_out.setVisible(False)
        self.video_out_line.setVisible(False)
        self.lbl_video_path.setVisible(False)
        self.lbl_video_viz_settings.setVisible(False)
        self.check_show_labels.setVisible(False)
        self.check_show_orientation.setVisible(False)
        self.check_show_trails.setVisible(False)
        self.spin_trail_duration.setVisible(False)
        self.lbl_trail_duration.setVisible(False)
        self.spin_marker_size.setVisible(False)
        self.lbl_marker_size.setVisible(False)
        self.spin_text_scale.setVisible(False)
        self.lbl_text_scale.setVisible(False)
        self.spin_arrow_length.setVisible(False)
        self.lbl_arrow_length.setVisible(False)
        self.lbl_video_pose_settings.setVisible(False)
        self.check_video_show_pose.setVisible(False)
        self.lbl_video_pose_color_mode.setVisible(False)
        self.combo_video_pose_color_mode.setVisible(False)
        self.lbl_video_pose_color_label.setVisible(False)
        self.btn_video_pose_color.setVisible(False)
        self.lbl_video_pose_color.setVisible(False)
        self.lbl_video_pose_point_radius.setVisible(False)
        self.spin_video_pose_point_radius.setVisible(False)
        self.lbl_video_pose_point_thickness.setVisible(False)
        self.spin_video_pose_point_thickness.setVisible(False)
        self.lbl_video_pose_line_thickness.setVisible(False)
        self.spin_video_pose_line_thickness.setVisible(False)
        self.lbl_video_pose_disabled_hint.setVisible(False)

        # RefineKit launch button
        g_refinekit = QGroupBox("Interactive Proofreading")
        self._main_window._set_compact_section_widget(g_refinekit)
        vl_refinekit = QVBoxLayout(g_refinekit)
        vl_refinekit.addWidget(
            self._main_window._create_help_label(
                "Open completed tracking results in RefineKit for "
                "interactive identity proofreading and swap correction."
            )
        )
        self._btn_open_refinekit = QPushButton("Open in RefineKit")
        self._btn_open_refinekit.setToolTip(
            "Open completed tracking results in RefineKit for "
            "interactive proofreading"
        )
        self._btn_open_refinekit.setEnabled(False)
        self._btn_open_refinekit.clicked.connect(self._main_window._open_refinekit)
        vl_refinekit.addWidget(self._btn_open_refinekit)
        vbox.addWidget(g_refinekit)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config

    # =========================================================================
    # HANDLER METHODS (moved from MainWindow)
    # =========================================================================

    def _on_cleaning_toggled(self, state):
        """Enable/disable trajectory cleaning controls based on checkbox."""
        enabled = self.enable_postprocessing.isChecked()

        # Hide/show all cleaning parameter widgets
        self.spin_min_trajectory_length.setVisible(enabled)
        self.lbl_min_trajectory_length.setVisible(enabled)
        self.spin_max_velocity_break.setVisible(enabled)
        self.lbl_max_velocity_break.setVisible(enabled)
        self.spin_max_occlusion_gap.setVisible(enabled)
        self.lbl_max_occlusion_gap.setVisible(enabled)
        self.chk_enable_tracklet_relinking.setVisible(enabled)
        self.lbl_enable_tracklet_relinking.setVisible(enabled)
        self.spin_relink_pose_max_distance.setVisible(enabled)
        self.lbl_relink_pose_max_distance.setVisible(enabled)
        self.spin_max_velocity_zscore.setVisible(enabled)
        self.lbl_max_velocity_zscore.setVisible(enabled)
        self.spin_velocity_zscore_window.setVisible(enabled)
        self.lbl_velocity_zscore_window.setVisible(enabled)
        self.spin_velocity_zscore_min_vel.setVisible(enabled)
        self.lbl_velocity_zscore_min_vel.setVisible(enabled)
        self.combo_interpolation_method.setVisible(enabled)
        self.lbl_interpolation_method.setVisible(enabled)
        self.spin_interpolation_max_gap.setVisible(enabled)
        self.lbl_interpolation_max_gap.setVisible(enabled)
        self.spin_heading_flip_max_burst.setVisible(enabled)
        self.lbl_heading_flip_max_burst.setVisible(enabled)
        self.spin_merge_overlap_multiplier.setVisible(enabled)
        self.lbl_merge_overlap_multiplier.setVisible(enabled)
        self.spin_min_overlap_frames.setVisible(enabled)
        self.lbl_min_overlap_frames.setVisible(enabled)
        self.chk_cleanup_temp_files.setVisible(enabled)

        # Also control enable state
        self.spin_min_trajectory_length.setEnabled(enabled)
        self.spin_max_velocity_break.setEnabled(enabled)
        self.spin_max_occlusion_gap.setEnabled(enabled)
        self.chk_enable_tracklet_relinking.setEnabled(enabled)
        self.spin_relink_pose_max_distance.setEnabled(enabled)
        self.spin_max_velocity_zscore.setEnabled(enabled)
        self.spin_velocity_zscore_window.setEnabled(enabled)
        self.spin_velocity_zscore_min_vel.setEnabled(enabled)
        self.combo_interpolation_method.setEnabled(enabled)
        self.spin_interpolation_max_gap.setEnabled(enabled)
        self.spin_heading_flip_max_burst.setEnabled(enabled)
        self.spin_merge_overlap_multiplier.setEnabled(enabled)
        self.spin_min_overlap_frames.setEnabled(enabled)
        self.chk_cleanup_temp_files.setEnabled(enabled)

        # Pose quality widgets — visible only when post-processing AND pose export are active
        pose_enabled = enabled and self._main_window._is_pose_export_enabled()
        self.spin_pose_export_min_valid_fraction.setVisible(pose_enabled)
        self.lbl_pose_export_min_valid_fraction.setVisible(pose_enabled)
        self.spin_pose_export_min_valid_keypoints.setVisible(pose_enabled)
        self.lbl_pose_export_min_valid_keypoints.setVisible(pose_enabled)
        self.spin_relink_min_pose_quality.setVisible(pose_enabled)
        self.lbl_relink_min_pose_quality.setVisible(pose_enabled)
        self.spin_pose_postproc_max_gap.setVisible(pose_enabled)
        self.lbl_pose_postproc_max_gap.setVisible(pose_enabled)
        self.spin_pose_temporal_outlier_zscore.setVisible(pose_enabled)
        self.lbl_pose_temporal_outlier_zscore.setVisible(pose_enabled)

        self.spin_pose_export_min_valid_fraction.setEnabled(pose_enabled)
        self.spin_pose_export_min_valid_keypoints.setEnabled(pose_enabled)
        self.spin_relink_min_pose_quality.setEnabled(pose_enabled)
        self.spin_pose_postproc_max_gap.setEnabled(pose_enabled)
        self.spin_pose_temporal_outlier_zscore.setEnabled(pose_enabled)

    def _on_video_output_toggled(self, checked):
        """Enable/disable video output controls."""
        # Hide/show all video output widgets
        self.btn_video_out.setVisible(checked)
        self.video_out_line.setVisible(checked)
        self.lbl_video_path.setVisible(checked)
        self.lbl_video_viz_settings.setVisible(checked)
        self.check_show_labels.setVisible(checked)
        self.check_show_orientation.setVisible(checked)
        self.check_show_trails.setVisible(checked)
        self.spin_trail_duration.setVisible(checked)
        self.lbl_trail_duration.setVisible(checked)
        self.spin_marker_size.setVisible(checked)
        self.lbl_marker_size.setVisible(checked)
        self.spin_text_scale.setVisible(checked)
        self.lbl_text_scale.setVisible(checked)
        self.spin_arrow_length.setVisible(checked)
        self.lbl_arrow_length.setVisible(checked)
        self.lbl_video_pose_settings.setVisible(checked)
        self.check_video_show_pose.setVisible(checked)
        self.lbl_video_pose_color_mode.setVisible(checked)
        self.combo_video_pose_color_mode.setVisible(checked)
        self.lbl_video_pose_color_label.setVisible(checked)
        self.btn_video_pose_color.setVisible(checked)
        self.lbl_video_pose_color.setVisible(checked)
        self.lbl_video_pose_point_radius.setVisible(checked)
        self.spin_video_pose_point_radius.setVisible(checked)
        self.lbl_video_pose_point_thickness.setVisible(checked)
        self.spin_video_pose_point_thickness.setVisible(checked)
        self.lbl_video_pose_line_thickness.setVisible(checked)
        self.spin_video_pose_line_thickness.setVisible(checked)
        self.lbl_video_pose_disabled_hint.setVisible(checked)

        # Also control enable state
        self.btn_video_out.setEnabled(checked)
        self.video_out_line.setEnabled(checked)
        self._main_window._sync_video_pose_overlay_controls()

    def select_video_output(self) -> None:
        """select_video_output method documentation."""
        fp, _ = QFileDialog.getSaveFileName(
            self, "Select Video Output", "", "Video Files (*.mp4 *.avi)"
        )
        if fp:
            self.video_out_line.setText(fp)

    def _select_video_pose_color(self):
        """Open color picker for fixed pose overlay color (BGR)."""
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QColorDialog

        b, g, r = self._video_pose_color
        initial_color = QColor(r, g, b)
        color = QColorDialog.getColor(initial_color, self, "Choose Pose Overlay Color")
        if color.isValid():
            self._video_pose_color = (
                color.blue(),
                color.green(),
                color.red(),
            )
            self._update_video_pose_color_button()

    def _update_video_pose_color_button(self):
        """Update fixed pose-color preview button and text label."""
        b, g, r = self._video_pose_color
        self.btn_video_pose_color.setStyleSheet(
            f"background-color: rgb({r}, {g}, {b}); "
            f"border: 1px solid #333; border-radius: 2px;"
        )
        self.lbl_video_pose_color.setText(f"{self._video_pose_color}")
