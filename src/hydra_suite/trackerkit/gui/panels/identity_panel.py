"""IdentityPanel — identity classification, pose analysis, and keypoint config."""

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
    QListWidget,
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


class IdentityPanel(QWidget):
    """CNN/appearance identity assignment and pose backend configuration."""

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
        self._layout.setContentsMargins(0, 0, 0, 0)
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
        form = QVBoxLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(8)
        self._main_window._set_compact_scroll_layout(form)

        self.lbl_individual_yolo_only_notice = self._main_window._create_help_label(
            "Individual analysis requires YOLO OBB mode.\n"
            "Switch detection method to YOLO OBB to enable this pipeline.",
            attach_to_title=False,
        )
        self.lbl_individual_yolo_only_notice.setVisible(False)
        form.addWidget(self.lbl_individual_yolo_only_notice)

        # Identity Classification Section
        self.g_identity = QGroupBox("Enable Identity Classification")
        self.g_identity.setCheckable(True)
        self.g_identity.setChecked(False)
        self.g_identity.toggled.connect(self._main_window._on_identity_analysis_toggled)
        self._main_window._set_compact_section_widget(self.g_identity)
        vl_identity = QVBoxLayout(self.g_identity)
        self.identity_content = QWidget()
        self.identity_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        identity_content_layout = QVBoxLayout(self.identity_content)
        identity_content_layout.setContentsMargins(0, 0, 0, 0)
        identity_content_layout.setSpacing(8)
        self.lbl_identity_help = self._main_window._create_help_label(
            "Classify individual identity during tracking. Extracts crops around each detection "
            "and processes them with the selected method."
        )
        identity_content_layout.addWidget(self.lbl_identity_help)

        # Hidden legacy widgets (referenced by model-import dialog)
        self.line_color_tag_model = QLineEdit()
        self.line_color_tag_model.setPlaceholderText("path/to/color_tag_model.pt")
        self.line_color_tag_model.setVisible(False)
        self.spin_color_tag_conf = QDoubleSpinBox()
        self.spin_color_tag_conf.setRange(0.01, 1.0)
        self.spin_color_tag_conf.setValue(0.5)
        self.spin_color_tag_conf.setSingleStep(0.05)
        self.spin_color_tag_conf.setToolTip(
            "Minimum confidence for color tag detection"
        )
        self.spin_color_tag_conf.setVisible(False)

        # --- Shared identity cost controls ---
        fl_identity_cost = QFormLayout()
        fl_identity_cost.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.spin_identity_match_bonus = QDoubleSpinBox()
        self.spin_identity_match_bonus.setRange(0.0, 200.0)
        self.spin_identity_match_bonus.setSingleStep(5.0)
        self.spin_identity_match_bonus.setValue(20.0)
        self.spin_identity_match_bonus.setToolTip(
            "Cost bonus (subtracted) when an identity observation matches the track.\n"
            "Divided equally across all active identity sources (AprilTags + each CNN)."
        )
        fl_identity_cost.addRow("Identity match bonus", self.spin_identity_match_bonus)
        self.spin_identity_mismatch_penalty = QDoubleSpinBox()
        self.spin_identity_mismatch_penalty.setRange(0.0, 500.0)
        self.spin_identity_mismatch_penalty.setSingleStep(5.0)
        self.spin_identity_mismatch_penalty.setValue(50.0)
        self.spin_identity_mismatch_penalty.setToolTip(
            "Cost penalty (added) when an identity observation conflicts with the track.\n"
            "Divided equally across all active identity sources (AprilTags + each CNN)."
        )
        fl_identity_cost.addRow(
            "Identity mismatch penalty", self.spin_identity_mismatch_penalty
        )
        identity_content_layout.addLayout(fl_identity_cost)

        # --- AprilTags group ---
        self.g_apriltags = QGroupBox("AprilTags")
        self.g_apriltags.setCheckable(True)
        self.g_apriltags.setChecked(False)
        self.g_apriltags.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        vl_apriltags_outer = QVBoxLayout(self.g_apriltags)
        vl_apriltags_outer.setContentsMargins(0, 0, 0, 0)
        self.apriltag_settings_widget = QWidget()
        self.apriltag_settings_widget.setVisible(False)
        fl_apriltags = QFormLayout(self.apriltag_settings_widget)
        fl_apriltags.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.combo_apriltag_family = QComboBox()
        self.combo_apriltag_family.addItems(self._main_window._get_apriltag_families())
        self.combo_apriltag_family.setToolTip(
            "AprilTag family to detect.\n"
            "The list is populated from your installed apriltag library."
        )
        fl_apriltags.addRow("AprilTag family", self.combo_apriltag_family)
        self.spin_apriltag_decimate = QDoubleSpinBox()
        self.spin_apriltag_decimate.setRange(1.0, 4.0)
        self.spin_apriltag_decimate.setValue(1.0)
        self.spin_apriltag_decimate.setSingleStep(0.5)
        self.spin_apriltag_decimate.setToolTip(
            "Decimation factor for faster detection (higher = faster but less accurate)"
        )
        fl_apriltags.addRow("AprilTag downsampling", self.spin_apriltag_decimate)
        vl_apriltags_outer.addWidget(self.apriltag_settings_widget)
        self.g_apriltags.toggled.connect(self.apriltag_settings_widget.setVisible)
        identity_content_layout.addWidget(self.g_apriltags)

        # --- CNN Classifiers group ---
        self.g_cnn_classifiers = QGroupBox("CNN Classifiers")
        self._main_window._set_compact_section_widget(self.g_cnn_classifiers)
        vl_cnn = QVBoxLayout(self.g_cnn_classifiers)
        vl_cnn.setSpacing(4)
        self.btn_add_cnn_classifier = QPushButton("\uff0b Add CNN Classifier")
        self.btn_add_cnn_classifier.clicked.connect(
            self._main_window._add_cnn_classifier_row
        )
        vl_cnn.addWidget(self.btn_add_cnn_classifier)
        self.cnn_scroll_area = QScrollArea()
        self.cnn_scroll_area.setWidgetResizable(True)
        self.cnn_scroll_area.setFrameShape(QFrame.NoFrame)
        self.cnn_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.cnn_scroll_area.setMinimumHeight(0)
        self.cnn_scroll_area.setMaximumHeight(200)
        self.cnn_scroll_area.setVisible(False)
        cnn_scroll_widget = QWidget()
        cnn_scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.cnn_rows_layout = QVBoxLayout(cnn_scroll_widget)
        self.cnn_rows_layout.setSpacing(6)
        self.cnn_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.cnn_scroll_area.setWidget(cnn_scroll_widget)
        vl_cnn.addWidget(self.cnn_scroll_area)
        identity_content_layout.addWidget(self.g_cnn_classifiers)

        # Hidden sentinel combo for model import dialog (reused by CNNClassifierRow)
        self.combo_cnn_identity_model = QComboBox()
        self.combo_cnn_identity_model.setVisible(False)
        # Note: combo will be populated post-construction via _refresh_cnn_identity_model_combo
        # Hidden verification labels (may be queried by model import dialog)
        self.lbl_cnn_arch = QLabel("—")
        self.lbl_cnn_num_classes = QLabel("—")
        self.lbl_cnn_class_names = QLabel("—")
        self.lbl_cnn_input_size = QLabel("—")
        self.lbl_cnn_label = QLabel("—")
        self.spin_cnn_confidence = QDoubleSpinBox()
        self.spin_cnn_confidence.setRange(0.0, 1.0)
        self.spin_cnn_confidence.setSingleStep(0.05)
        self.spin_cnn_confidence.setValue(0.5)
        self.spin_cnn_window = QSpinBox()
        self.spin_cnn_window.setRange(1, 100)
        self.spin_cnn_window.setValue(10)
        vl_identity.addWidget(self.identity_content)

        self.g_individual_pipeline_common = QGroupBox(
            "Individual Analysis Pipeline Settings"
        )
        self._main_window._set_compact_section_widget(self.g_individual_pipeline_common)
        fl_common = QFormLayout(self.g_individual_pipeline_common)

        self.chk_individual_interpolate = QCheckBox(
            "Interpolate Occluded Frames After Tracking"
        )
        self.chk_individual_interpolate.setChecked(True)
        self.chk_individual_interpolate.setToolTip(
            "After tracking completes, fill occluded gaps by interpolating center/size/angle\n"
            "and generate additional masked crops. Interpolated crops are prefixed with 'interp_'."
        )
        fl_common.addRow("Interpolate occluded frames", self.chk_individual_interpolate)

        self.spin_individual_padding = QDoubleSpinBox()
        self.spin_individual_padding.setRange(0.0, 0.5)
        self.spin_individual_padding.setValue(0.1)
        self.spin_individual_padding.setSingleStep(0.05)
        self.spin_individual_padding.setDecimals(2)
        self.spin_individual_padding.setToolTip(
            "Padding around OBB bounding box as fraction of size.\n"
            "0.1 = 10% padding on each side.\n"
            "Applies to all precompute phases: pose, AprilTag, and CNN identity."
        )
        fl_common.addRow(
            "Crop padding fraction (all phases)", self.spin_individual_padding
        )

        bg_color_layout = QHBoxLayout()
        self.btn_background_color = QPushButton()
        self.btn_background_color.setMaximumWidth(60)
        self.btn_background_color.setMinimumHeight(30)
        self.btn_background_color.setToolTip("Click to choose background color")
        self.btn_background_color.clicked.connect(
            self._main_window._select_individual_background_color
        )
        self._background_color = (0, 0, 0)  # BGR
        bg_color_layout.addWidget(self.btn_background_color)

        self.btn_median_color = QPushButton("Use Median from Frame")
        self.btn_median_color.setToolTip(
            "Compute median color from the preview frame and use as background"
        )
        self.btn_median_color.clicked.connect(
            self._main_window._compute_median_background_color
        )
        bg_color_layout.addWidget(self.btn_median_color)

        self.lbl_background_color = QLabel("(0, 0, 0)")
        self.lbl_background_color.setToolTip("Current background color in BGR format")
        bg_color_layout.addWidget(self.lbl_background_color)
        # Note: background color button will be updated post-construction
        fl_common.addRow("Crop background color", bg_color_layout)

        self.chk_suppress_foreign_obb = QCheckBox("Suppress foreign animal regions")
        self.chk_suppress_foreign_obb.setChecked(True)
        self.chk_suppress_foreign_obb.setToolTip(
            "Fill overlapping animals' OBB areas with the background color before\n"
            "pose inference, and zero out any keypoints that fall inside another\n"
            "animal's bounding box after inference.\n"
            "\n"
            "Prevents pose contamination when animals are close or overlapping.\n"
            "Disable only if you observe incorrect masking of valid keypoints."
        )
        fl_common.addRow(
            "Pose contamination suppression", self.chk_suppress_foreign_obb
        )

        # Head-tail orientation classifier
        self.combo_yolo_headtail_model_type = QComboBox()
        self.combo_yolo_headtail_model_type.addItems(["YOLO", "tiny"])
        self.combo_yolo_headtail_model_type.setFixedHeight(30)
        self.combo_yolo_headtail_model_type.setToolTip(
            "Architecture family of the head-tail classifier.\n"
            "YOLO → models/classification/orientation/YOLO/\n"
            "tiny → models/classification/orientation/tiny/"
        )
        self.combo_yolo_headtail_model_type.currentIndexChanged.connect(
            self._main_window._on_headtail_model_type_changed
        )

        self.combo_yolo_headtail_model = QComboBox()
        # Note: combo_yolo_headtail_model will be populated post-construction
        self.combo_yolo_headtail_model.activated.connect(
            self._main_window.on_yolo_headtail_model_changed
        )
        self.combo_yolo_headtail_model.setFixedHeight(30)
        self.combo_yolo_headtail_model.setToolTip(
            "Optional classifier to resolve head vs. tail orientation along the OBB major axis.\n"
            "Runs during tracking and post-hoc individual analysis."
        )

        self.headtail_model_row_widget = QWidget()
        _headtail_row = QHBoxLayout(self.headtail_model_row_widget)
        _headtail_row.setContentsMargins(0, 0, 0, 0)
        _headtail_row.setSpacing(4)
        _headtail_row.addWidget(self.combo_yolo_headtail_model_type, 0)
        _headtail_row.addWidget(self.combo_yolo_headtail_model, 1)
        fl_common.addRow("Head-tail model", self.headtail_model_row_widget)

        self.spin_yolo_headtail_conf = QDoubleSpinBox()
        self.spin_yolo_headtail_conf.setRange(0.0, 1.0)
        self.spin_yolo_headtail_conf.setSingleStep(0.01)
        self.spin_yolo_headtail_conf.setValue(0.50)
        self.spin_yolo_headtail_conf.setFixedHeight(30)
        self.spin_yolo_headtail_conf.setToolTip(
            "Minimum classifier confidence for a head-tail assignment to be accepted (0–1).\n"
            "Lower = more assignments accepted; higher = fewer but more reliable."
        )
        fl_common.addRow("Head-tail min confidence", self.spin_yolo_headtail_conf)

        self.chk_pose_overrides_headtail = QCheckBox(
            "Pose orientation overrides head-tail"
        )
        self.chk_pose_overrides_headtail.setChecked(True)
        self.chk_pose_overrides_headtail.setToolTip(
            "When enabled, valid pose heading takes precedence over head-tail heading."
        )
        fl_common.addRow("", self.chk_pose_overrides_headtail)

        form.addWidget(self.g_individual_pipeline_common)
        form.addWidget(self.g_identity)

        self.g_pose_runtime = QGroupBox("Enable Pose Extraction")
        self.g_pose_runtime.setCheckable(True)
        self.g_pose_runtime.setChecked(False)
        self.g_pose_runtime.toggled.connect(self._main_window._on_pose_analysis_toggled)
        self.g_pose_runtime.toggled.connect(
            self._main_window._sync_video_pose_overlay_controls
        )
        self.g_pose_runtime.toggled.connect(
            self._main_window._on_runtime_context_changed
        )
        # Cross-panel connection wired in MainWindow.init_ui() after both panels exist
        self.chk_enable_pose_extractor = self.g_pose_runtime
        self._main_window._set_compact_section_widget(self.g_pose_runtime)
        vl_pose = QVBoxLayout(self.g_pose_runtime)
        self.pose_runtime_content = QWidget()
        self.pose_runtime_content.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Maximum
        )
        pose_runtime_content_layout = QVBoxLayout(self.pose_runtime_content)
        pose_runtime_content_layout.setContentsMargins(0, 0, 0, 0)
        pose_runtime_content_layout.setSpacing(8)
        self.lbl_pose_runtime_help = self._main_window._create_help_label(
            "Minimum runtime pose settings used by the individual-analysis pipeline."
        )
        pose_runtime_content_layout.addWidget(self.lbl_pose_runtime_help)
        fl_pose = QFormLayout(None)
        fl_pose.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.form_pose_runtime = fl_pose

        self.combo_pose_model_type = QComboBox()
        self.combo_pose_model_type.addItems(["YOLO", "SLEAP", "ViTPose"])
        self.combo_pose_model_type.setToolTip("Pose backend for individual analysis.")
        self.combo_pose_model_type.currentIndexChanged.connect(
            self._main_window._sync_pose_backend_ui
        )
        fl_pose.addRow("Pose model type", self.combo_pose_model_type)

        self.combo_pose_runtime_flavor = QComboBox()
        self.combo_pose_runtime_flavor.setToolTip(
            "Pose runtime implementation.\n"
            "Auto/Native uses default backend runtime.\n"
            "ONNX/TensorRT artifacts are exported and reused automatically."
        )
        # Note: combo_pose_runtime_flavor will be populated post-construction
        self.combo_pose_runtime_flavor.currentIndexChanged.connect(
            self._main_window._sync_pose_backend_ui
        )
        fl_pose.addRow("Pose runtime", self.combo_pose_runtime_flavor)
        self._main_window._set_form_row_visible(
            fl_pose, self.combo_pose_runtime_flavor, False
        )

        self.combo_pose_model = QComboBox()
        self.combo_pose_model.setToolTip(
            "Pose model for individual analysis.\n"
            "Choose an imported model or select '＋ Add New Model…' to browse and import."
        )
        self.combo_pose_model.currentIndexChanged.connect(
            self._main_window.on_pose_model_changed
        )
        # Note: combo_pose_model will be populated post-construction
        fl_pose.addRow("Pose model", self.combo_pose_model)

        self.spin_pose_min_kpt_conf_valid = QDoubleSpinBox()
        self.spin_pose_min_kpt_conf_valid.setRange(0.0, 1.0)
        self.spin_pose_min_kpt_conf_valid.setSingleStep(0.05)
        self.spin_pose_min_kpt_conf_valid.setDecimals(2)
        self.spin_pose_min_kpt_conf_valid.setValue(0.2)
        self.spin_pose_min_kpt_conf_valid.setToolTip(
            "Minimum per-keypoint confidence to consider a keypoint valid."
        )
        fl_pose.addRow("Min keypoint confidence", self.spin_pose_min_kpt_conf_valid)

        self.spin_pose_batch = QSpinBox()
        self.spin_pose_batch.setRange(1, 256)
        self.spin_pose_batch.setValue(
            int(self._main_window.advanced_config.get("pose_batch_size", 4))
        )
        self.spin_pose_batch.setToolTip(
            "Shared batch size for pose inference across YOLO and SLEAP backends."
        )
        fl_pose.addRow("Pose batch size", self.spin_pose_batch)

        h_pose_skeleton = QHBoxLayout()
        self.line_pose_skeleton_file = QLineEdit()
        self.line_pose_skeleton_file.setPlaceholderText(
            "Select skeleton JSON (keypoint_names + skeleton_edges)..."
        )
        default_skeleton = str(
            self._main_window.advanced_config.get("pose_skeleton_file", "")
        ).strip()
        if not default_skeleton:
            from hydra_suite.paths import get_skeleton_dir

            candidate = get_skeleton_dir() / "ooceraea_biroi.json"
            if candidate.exists():
                default_skeleton = str(candidate)
        if default_skeleton:
            self.line_pose_skeleton_file.setText(default_skeleton)
        self.btn_browse_pose_skeleton_file = QPushButton("Browse...")
        self.btn_browse_pose_skeleton_file.clicked.connect(
            self._main_window._select_pose_skeleton_file
        )
        self.line_pose_skeleton_file.textChanged.connect(
            self._main_window._refresh_pose_direction_keypoint_lists
        )
        h_pose_skeleton.addWidget(self.line_pose_skeleton_file)
        h_pose_skeleton.addWidget(self.btn_browse_pose_skeleton_file)
        fl_pose.addRow("Skeleton file", h_pose_skeleton)

        self.list_pose_ignore_keypoints = QListWidget()
        self.list_pose_ignore_keypoints.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_ignore_keypoints.setMinimumHeight(96)
        self.list_pose_ignore_keypoints.setMaximumHeight(120)
        self.list_pose_ignore_keypoints.setToolTip(
            "Select keypoints to ignore in pose export and orientation logic."
        )

        self.list_pose_direction_anterior = QListWidget()
        self.list_pose_direction_anterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_anterior.setMinimumHeight(96)
        self.list_pose_direction_anterior.setMaximumHeight(120)
        self.list_pose_direction_anterior.setToolTip(
            "Select anterior keypoints from skeleton keypoint list."
        )

        self.list_pose_direction_posterior = QListWidget()
        self.list_pose_direction_posterior.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.list_pose_direction_posterior.setMinimumHeight(96)
        self.list_pose_direction_posterior.setMaximumHeight(120)
        self.list_pose_direction_posterior.setToolTip(
            "Select posterior keypoints from skeleton keypoint list."
        )

        pose_kpt_groups_widget = QWidget()
        pose_kpt_groups_layout = QHBoxLayout(pose_kpt_groups_widget)
        pose_kpt_groups_layout.setContentsMargins(0, 0, 0, 0)
        pose_kpt_groups_layout.setSpacing(8)

        ignore_col = QVBoxLayout()
        ignore_label = QLabel("Ignore")
        ignore_label.setStyleSheet("font-weight: bold;")
        ignore_col.addWidget(ignore_label)
        ignore_col.addWidget(self.list_pose_ignore_keypoints)
        pose_kpt_groups_layout.addLayout(ignore_col, 1)

        anterior_col = QVBoxLayout()
        anterior_label = QLabel("Anterior")
        anterior_label.setStyleSheet("font-weight: bold;")
        anterior_col.addWidget(anterior_label)
        anterior_col.addWidget(self.list_pose_direction_anterior)
        pose_kpt_groups_layout.addLayout(anterior_col, 1)

        posterior_col = QVBoxLayout()
        posterior_label = QLabel("Posterior")
        posterior_label.setStyleSheet("font-weight: bold;")
        posterior_col.addWidget(posterior_label)
        posterior_col.addWidget(self.list_pose_direction_posterior)
        pose_kpt_groups_layout.addLayout(posterior_col, 1)

        fl_pose.addRow("Keypoint groups", pose_kpt_groups_widget)
        self.list_pose_ignore_keypoints.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("ignore")
        )
        self.list_pose_direction_anterior.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("anterior")
        )
        self.list_pose_direction_posterior.itemSelectionChanged.connect(
            lambda: self._main_window._on_pose_keypoint_group_changed("posterior")
        )

        h_sleap_env = QHBoxLayout()
        self.combo_pose_sleap_env = QComboBox()
        self.combo_pose_sleap_env.setToolTip(
            "Conda environment name must start with 'sleap'."
        )
        h_sleap_env.addWidget(self.combo_pose_sleap_env, 1)
        self.btn_refresh_pose_sleap_envs = QPushButton("Refresh")
        self.btn_refresh_pose_sleap_envs.setToolTip("Refresh SLEAP conda envs list")
        self.btn_refresh_pose_sleap_envs.clicked.connect(
            self._main_window._refresh_pose_sleap_envs
        )
        h_sleap_env.addWidget(self.btn_refresh_pose_sleap_envs)
        self.pose_sleap_env_row_widget = QWidget()
        self.pose_sleap_env_row_widget.setLayout(h_sleap_env)
        fl_pose.addRow("SLEAP env", self.pose_sleap_env_row_widget)

        self.chk_sleap_experimental_features = QCheckBox("Allow experimental features")
        self.chk_sleap_experimental_features.setChecked(False)
        self.chk_sleap_experimental_features.setToolTip(
            "Enable experimental SLEAP features (ONNX/TensorRT runtimes).\n"
            "When disabled, ONNX/TensorRT runtime selections will revert to native runtime.\n"
            "Experimental features may have stability or accuracy issues."
        )
        self.chk_sleap_experimental_features.stateChanged.connect(
            self._main_window._on_sleap_experimental_toggled
        )
        self.pose_sleap_experimental_row_widget = QWidget()
        sleap_exp_layout = QHBoxLayout()
        sleap_exp_layout.setContentsMargins(0, 0, 0, 0)
        sleap_exp_layout.addWidget(self.chk_sleap_experimental_features)
        self.pose_sleap_experimental_row_widget.setLayout(sleap_exp_layout)
        fl_pose.addRow("", self.pose_sleap_experimental_row_widget)

        pose_runtime_content_layout.addLayout(fl_pose)
        vl_pose.addWidget(self.pose_runtime_content)
        form.addWidget(self.g_pose_runtime)

        # Note: pose_sleap_envs and pose_runtime_flavor will be populated post-construction
        self._main_window._set_form_row_visible(
            fl_pose, self.pose_sleap_experimental_row_widget, False
        )

        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Initially disable all controls (sync will show/hide based on state)
        self.g_identity.setVisible(False)
        self.g_identity.setEnabled(False)
        self.g_individual_pipeline_common.setVisible(False)
        self.g_individual_pipeline_common.setEnabled(False)
        self.g_pose_runtime.setVisible(False)
        self.g_pose_runtime.setEnabled(False)
        # Note: _refresh_pose_direction_keypoint_lists and _sync_pose_backend_ui
        # are called post-construction from main_window

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
