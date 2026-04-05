"""DatasetPanel — active learning dataset generation controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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


class DatasetPanel(QWidget):
    """Active learning dataset generation: frame selection, export, and controls."""

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

        # Add scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        form = QVBoxLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(8)
        self._main_window._set_compact_scroll_layout(form)

        # ============================================================
        # Active Learning Dataset Section
        # ============================================================
        self.g_active_learning = QGroupBox(
            "Do you want to generate a detection dataset?"
        )
        self._main_window._set_compact_section_widget(self.g_active_learning)
        vl_active = QVBoxLayout(self.g_active_learning)
        vl_active.addWidget(
            self._main_window._create_help_label(
                "Automatically identify challenging frames during tracking and export them for annotation.\n\n"
                "Workflow: Run tracking → Review/correct in X-AnyLabeling → Train improved YOLO model"
            )
        )

        # Enable checkbox
        self.chk_enable_dataset_gen = QCheckBox(
            "Enable Dataset Generation for Active Learning"
        )
        self.chk_enable_dataset_gen.setChecked(False)
        self.chk_enable_dataset_gen.toggled.connect(
            self._main_window._on_dataset_generation_toggled
        )
        vl_active.addWidget(self.chk_enable_dataset_gen)

        # Content container for all configuration options
        self.active_learning_content = QWidget()
        vl_content = QVBoxLayout(self.active_learning_content)

        # Dataset configuration
        self.g_dataset_config = QGroupBox("How should the dataset be configured?")
        self._main_window._set_compact_section_widget(self.g_dataset_config)
        f_config = QFormLayout(self.g_dataset_config)
        f_config.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Class name
        self.line_dataset_class_name = QLineEdit()
        self.line_dataset_class_name.setPlaceholderText("e.g., ant")
        self.line_dataset_class_name.setText("object")
        self.line_dataset_class_name.setToolTip(
            "Name of the object class being tracked.\n"
            "This will be used in the classes.txt file for YOLO training.\n"
            "Examples: ant, bee, mouse, fish, etc."
        )
        f_config.addRow("Class label", self.line_dataset_class_name)

        vl_content.addWidget(self.g_dataset_config)

        # Frame selection parameters
        self.g_frame_selection = QGroupBox("How should frames be selected?")
        self._main_window._set_compact_section_widget(self.g_frame_selection)
        f_selection = QFormLayout(self.g_frame_selection)
        f_selection.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Number of frames to export
        self.spin_dataset_max_frames = QSpinBox()
        self.spin_dataset_max_frames.setRange(10, 1000)
        self.spin_dataset_max_frames.setValue(100)
        self.spin_dataset_max_frames.setToolTip(
            "Maximum number of frames to export (10-1000).\n"
            "Higher values provide more training data but increase annotation time.\n"
            "Recommended: 50-200 frames for initial improvement."
        )
        f_selection.addRow("Maximum frames to export", self.spin_dataset_max_frames)

        # Frame quality scoring threshold (used DURING TRACKING to identify problematic frames)
        self.spin_dataset_conf_threshold = QDoubleSpinBox()
        self.spin_dataset_conf_threshold.setRange(0.0, 1.0)
        self.spin_dataset_conf_threshold.setSingleStep(0.05)
        self.spin_dataset_conf_threshold.setDecimals(2)
        self.spin_dataset_conf_threshold.setValue(0.5)
        self.spin_dataset_conf_threshold.setToolTip(
            "[FRAME SELECTION] Confidence threshold for identifying problematic frames DURING TRACKING.\n\n"
            "Frames with detections below this confidence are scored as 'challenging'\n"
            "and prioritized for export to improve training data.\n\n"
            "• Lower (0.3-0.4): Only flag very uncertain detections\n"
            "• Higher (0.5-0.7): Flag moderately uncertain detections too\n\n"
            "Recommended: 0.5 (default) - captures frames that need model improvement"
        )
        f_selection.addRow("Quality threshold", self.spin_dataset_conf_threshold)

        # Add help label explaining advanced options
        advanced_help = self._main_window._create_help_label(
            "Note: YOLO detection sensitivity for export (confidence=0.05, IOU=0.5) can be "
            "customized in advanced_config.json. These are separate from tracking parameters and "
            "optimized for annotation (detect everything, manual review corrects errors).",
            attach_to_title=False,
        )
        f_selection.addRow(advanced_help)

        # Visual diversity window
        self.spin_dataset_diversity_window = QSpinBox()
        self.spin_dataset_diversity_window.setRange(10, 500)
        self.spin_dataset_diversity_window.setValue(30)
        self.spin_dataset_diversity_window.setToolTip(
            "Minimum frame separation for visual diversity (10-500 frames).\n"
            "Prevents selecting too many consecutive similar frames.\n"
            "Higher = more spread out frames, more visual variety.\n"
            "Recommended: 20-50 frames (depends on video frame rate)."
        )
        f_selection.addRow(
            "Diversity window (frames)",
            self.spin_dataset_diversity_window,
        )

        # Include context frames
        self.chk_dataset_include_context = QCheckBox(
            "Include neighboring frames (+/-1)"
        )
        self.chk_dataset_include_context.setChecked(True)
        self.chk_dataset_include_context.setToolTip(
            "Export the frame before and after each selected frame.\n"
            "Provides temporal context which can improve annotation quality.\n"
            "Increases dataset size by 3x."
        )
        self.chk_dataset_probabilistic = QCheckBox("Probabilistic Sampling")
        self.chk_dataset_probabilistic.setChecked(True)
        self.chk_dataset_probabilistic.setToolTip(
            "Use rank-based probabilistic sampling instead of greedy selection.\n"
            "Probabilistic: Higher quality scores = higher probability (more variety).\n"
            "Greedy: Always select absolute worst frames first (may be too extreme).\n"
            "Recommended: Enabled for better training data diversity."
        )
        _sel_chk_row = QHBoxLayout()
        _sel_chk_row.addWidget(self.chk_dataset_include_context)
        _sel_chk_row.addWidget(self.chk_dataset_probabilistic)
        f_selection.addRow(_sel_chk_row)

        vl_content.addWidget(self.g_frame_selection)

        # Quality metrics
        self.g_quality_metrics = QGroupBox("Which quality checks should be applied?")
        self._main_window._set_compact_section_widget(self.g_quality_metrics)
        v_metrics = QVBoxLayout(self.g_quality_metrics)

        self.chk_metric_low_confidence = QCheckBox("Flag low detection confidence")
        self.chk_metric_low_confidence.setChecked(True)
        self.chk_metric_low_confidence.setToolTip(
            "Flag frames where YOLO confidence is below threshold."
        )
        self.chk_metric_count_mismatch = QCheckBox("Flag detection count mismatch")
        self.chk_metric_count_mismatch.setChecked(True)
        self.chk_metric_count_mismatch.setToolTip(
            "Flag frames where detected count doesn't match expected number of animals."
        )
        self.chk_metric_high_assignment_cost = QCheckBox(
            "Flag uncertain track assignment"
        )
        self.chk_metric_high_assignment_cost.setChecked(True)
        self.chk_metric_high_assignment_cost.setToolTip(
            "Flag frames where tracker struggles to match detections to tracks."
        )
        self.chk_metric_track_loss = QCheckBox("Flag frequent track loss")
        self.chk_metric_track_loss.setChecked(True)
        self.chk_metric_track_loss.setToolTip(
            "Flag frames where tracks are frequently lost."
        )
        self.chk_metric_high_uncertainty = QCheckBox("Flag high position uncertainty")
        self.chk_metric_high_uncertainty.setChecked(False)
        self.chk_metric_high_uncertainty.setToolTip(
            "Flag frames where Kalman filter is very uncertain about positions."
        )
        _m_row1 = QHBoxLayout()
        _m_row1.addWidget(self.chk_metric_low_confidence)
        _m_row1.addWidget(self.chk_metric_count_mismatch)
        _m_row2 = QHBoxLayout()
        _m_row2.addWidget(self.chk_metric_high_assignment_cost)
        _m_row2.addWidget(self.chk_metric_track_loss)
        v_metrics.addLayout(_m_row1)
        v_metrics.addLayout(_m_row2)
        v_metrics.addWidget(self.chk_metric_high_uncertainty)

        vl_content.addWidget(self.g_quality_metrics)

        # X-AnyLabeling Integration
        self.g_xanylabeling = QGroupBox("How should X-AnyLabeling be integrated?")
        self._main_window._set_compact_section_widget(self.g_xanylabeling)
        vl_xany = QVBoxLayout(self.g_xanylabeling)

        # Conda environment selection
        h_env = QHBoxLayout()
        h_env.addWidget(QLabel("Conda environment"))
        self.combo_xanylabeling_env = QComboBox()
        self.combo_xanylabeling_env.setToolTip(
            "Select a conda environment with X-AnyLabeling installed.\n"
            "Environment names should start with 'x-anylabeling-' to be detected."
        )
        self.combo_xanylabeling_env.currentTextChanged.connect(
            self._main_window._on_xanylabeling_env_changed
        )
        h_env.addWidget(self.combo_xanylabeling_env, 1)
        self.btn_refresh_envs = QPushButton("🔄")
        self.btn_refresh_envs.setMaximumWidth(40)
        self.btn_refresh_envs.setToolTip("Refresh conda environments list")
        self.btn_refresh_envs.clicked.connect(
            self._main_window._refresh_xanylabeling_envs
        )
        h_env.addWidget(self.btn_refresh_envs)
        vl_xany.addLayout(h_env)

        # Open in X-AnyLabeling button
        self.btn_open_xanylabeling = QPushButton(
            "Open Active Learning Dataset in X-AnyLabeling"
        )
        self.btn_open_xanylabeling.setToolTip(
            "Browse for a dataset directory and open it in X-AnyLabeling.\n"
            "Directory must contain: classes.txt, images/, and labels/"
        )
        self.btn_open_xanylabeling.clicked.connect(
            self._main_window._open_in_xanylabeling
        )
        self.btn_open_xanylabeling.setEnabled(False)
        vl_xany.addWidget(self.btn_open_xanylabeling)

        # X-AnyLabeling integration is now separate from Active Learning content

        # Add content to main group box
        vl_active.addWidget(self.active_learning_content)

        # Add main group box to form
        form.addWidget(self.g_active_learning)

        # ============================================================
        # X-AnyLabeling Integration (separate section)
        # ============================================================
        form.addWidget(self.g_xanylabeling)

        # Initially hide content (checkbox starts unchecked)
        self.active_learning_content.setVisible(False)

        # ============================================================
        # Individual Dataset Generator Section (Real-time OBB crops)
        # ============================================================
        self.g_individual_dataset = QGroupBox(
            "Should individual crops be collected in real time?"
        )
        self._main_window._set_compact_section_widget(self.g_individual_dataset)
        vl_ind_dataset = QVBoxLayout(self.g_individual_dataset)
        vl_ind_dataset.addWidget(
            self._main_window._create_help_label(
                "Persist individual-analysis crop images to disk.\n\n"
                "• This save option depends on Analyze Individuals pipeline being enabled\n"
                "• Crops contain only the detected animal (OBB-masked)\n"
                "• Intended for downstream labeling/training workflows\n\n"
                "Note: Available only in YOLO OBB mode."
            )
        )

        self.chk_enable_individual_dataset = QCheckBox(
            "Save Individual Analysis Images to Disk"
        )
        self.chk_enable_individual_dataset.toggled.connect(
            self._main_window._on_individual_dataset_toggled
        )
        vl_ind_dataset.addWidget(self.chk_enable_individual_dataset)

        self.chk_generate_individual_track_videos = QCheckBox(
            "Generate orientation-fixed videos for final tracks after cleanup"
        )
        self.chk_generate_individual_track_videos.setChecked(False)
        self.chk_generate_individual_track_videos.setToolTip(
            "After final cleaning completes, export one orientation-fixed video per\n"
            "final TrajectoryID by streaming the source video and using the detection\n"
            "cache plus interpolated ROI cache. Independent from saved crop files."
        )
        vl_ind_dataset.addWidget(self.chk_generate_individual_track_videos)

        # Output Configuration
        self.ind_output_group = QGroupBox(
            "Where should individual-analysis outputs go?"
        )
        ind_output_layout = QFormLayout(self.ind_output_group)
        ind_output_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Output format
        self.combo_individual_format = QComboBox()
        self.combo_individual_format.addItems(["PNG", "JPEG"])
        self.combo_individual_format.setCurrentText("PNG")
        self.combo_individual_format.setToolTip(
            "PNG: Lossless, larger files\nJPEG: Smaller files, slight quality loss"
        )
        # Save interval
        self.spin_individual_interval = QSpinBox()
        self.spin_individual_interval.setRange(1, 100)
        self.spin_individual_interval.setValue(1)
        self.spin_individual_interval.setSingleStep(1)
        self.spin_individual_interval.setToolTip(
            "Save crops every N frames.\n"
            "1 = every frame, 10 = every 10th frame, etc."
        )
        _ind_fmt_row = QHBoxLayout()
        _ind_fmt_row.addWidget(QLabel("Format"))
        _ind_fmt_row.addWidget(self.combo_individual_format)
        _ind_fmt_row.addWidget(QLabel("Save every N frames"))
        _ind_fmt_row.addWidget(self.spin_individual_interval)
        ind_output_layout.addRow(_ind_fmt_row)

        vl_ind_dataset.addWidget(
            self._main_window._create_help_label(
                "Interpolation, padding, and crop background settings are configured in:\n"
                "Analyze Individuals -> Individual Analysis Pipeline Settings"
            )
        )

        vl_ind_dataset.addWidget(self.ind_output_group)

        self.chk_suppress_foreign_obb_dataset = QCheckBox(
            "Suppress foreign animal regions in saved crops"
        )
        self.chk_suppress_foreign_obb_dataset.setChecked(False)
        self.chk_suppress_foreign_obb_dataset.setToolTip(
            "Fill overlapping animals' OBB areas with the background color before\n"
            "saving each individual crop image.\n"
            "\n"
            "Prevents other animals from appearing in a detection's crop, which\n"
            "can confuse downstream labeling or training.\n"
            "Only applies to YOLO OBB detections (no effect in background-subtraction mode)."
        )
        vl_ind_dataset.addWidget(self.chk_suppress_foreign_obb_dataset)

        # Info label about filtering
        self.lbl_individual_info = self._main_window._create_help_label(
            "Note: Crops use detections already filtered by ROI and size settings.\n"
            "No additional filtering parameters needed.",
            attach_to_title=False,
        )
        vl_ind_dataset.addWidget(self.lbl_individual_info)

        form.addWidget(self.g_individual_dataset)

        # ============================================================
        # Pose Label UI Integration (Top-level section)
        # ============================================================
        self.g_pose_label = QGroupBox("Do you want to launch PoseKit labeler?")
        self._main_window._set_compact_section_widget(self.g_pose_label)
        vl_pose = QVBoxLayout(self.g_pose_label)
        vl_pose.addWidget(
            self._main_window._create_help_label(
                "Open an individual dataset folder in PoseKit Labeler for keypoint annotation.\n"
                "Select a dataset root that contains an `images/` directory."
            )
        )

        # Open in Pose Label button
        self.btn_open_pose_label = QPushButton(
            "Open Individual Dataset in PoseKit Labeler"
        )
        self.btn_open_pose_label.setToolTip(
            "Browse for a dataset directory and open it in PoseKit Labeler.\n"
            "Directory must contain: images/\n"
            "PoseKit project data will be stored in: posekit_project/"
        )
        self.btn_open_pose_label.clicked.connect(self._main_window._open_pose_label_ui)
        self.btn_open_pose_label.setEnabled(True)
        vl_pose.addWidget(self.btn_open_pose_label)

        form.addWidget(self.g_pose_label)

        # Initially hide individual dataset widgets (checkbox starts unchecked)
        self.g_individual_dataset.setVisible(False)
        self.ind_output_group.setVisible(False)
        self.lbl_individual_info.setVisible(False)
        self.chk_generate_individual_track_videos.setVisible(False)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
