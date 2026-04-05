"""DetectionPanel — detection method, image preprocessing, and model config."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hydra_suite.trackerkit.config.schemas import TrackerConfig
from hydra_suite.utils.gpu_utils import (
    MPS_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

if TYPE_CHECKING:
    from hydra_suite.trackerkit.gui.main_window import MainWindow


class DetectionPanel(QWidget):
    """Detection method selector, image-processing pipeline, and YOLO config."""

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
        from hydra_suite.trackerkit.gui.main_window import (
            AccordionContainer,
            CollapsibleGroupBox,
        )

        layout = self._layout
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)
        self._main_window._set_compact_scroll_layout(vbox)

        # ============================================================
        # 1. Detection Method Selector
        # ============================================================
        g_method = QGroupBox("Detection")
        self._main_window._set_compact_section_widget(g_method)
        l_method_outer = QVBoxLayout(g_method)
        l_method_outer.setSpacing(6)
        f_method = QFormLayout(None)
        f_method.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_method.setHorizontalSpacing(10)
        f_method.setVerticalSpacing(8)
        method_help = self._main_window._create_help_label(
            "Choose how to detect animals in each frame. Background subtraction models the static background and finds moving objects. YOLO uses deep learning to detect animals directly.",
            attach_to_title=False,
        )
        self.combo_detection_method = QComboBox()
        self.combo_detection_method.addItems(["Background Subtraction", "YOLO OBB"])
        self.combo_detection_method.setFixedHeight(30)
        self.combo_detection_method.currentIndexChanged.connect(
            self._main_window._on_detection_method_changed_ui
        )
        method_row = QHBoxLayout()
        method_row.setSpacing(6)
        method_row.addWidget(self.combo_detection_method, 1)
        method_row.addWidget(method_help, 0, Qt.AlignVCenter)
        f_method.addRow("Method", method_row)

        # Legacy device selection (hidden; derived from canonical runtime).
        self.combo_device = QComboBox()
        device_options = ["auto", "cpu"]
        device_tooltip_parts = [
            "Select compute device for detection:",
            "  • auto - Automatically select best available device",
            "  • cpu - CPU-only mode",
        ]

        if TORCH_CUDA_AVAILABLE:
            device_options.append("cuda:0")
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • cuda:0 - NVIDIA GPU (not available)")

        if MPS_AVAILABLE:
            device_options.append("mps")
            device_tooltip_parts.append("  • mps - Apple Silicon GPU ✓ Available")
        else:
            device_tooltip_parts.append("  • mps - Apple Silicon GPU (not available)")

        device_tooltip_parts.append(
            "\nApplies to both YOLO and Background Subtraction GPU acceleration."
        )

        self.combo_device.addItems(device_options)
        self.combo_device.setToolTip("\n".join(device_tooltip_parts))
        f_method.addRow("Which compute device should run detection?", self.combo_device)
        device_label = f_method.labelForField(self.combo_device)
        if device_label is not None:
            device_label.setVisible(False)
        self.combo_device.setVisible(False)

        l_method_outer.addLayout(f_method)
        vbox.addWidget(g_method)

        # Stacked Widget for Method Specific Params
        self.stack_detection = QStackedWidget()
        self.stack_detection.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        # --- Page 0: Background Subtraction Params ---
        page_bg = QWidget()
        page_bg.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        l_bg = QVBoxLayout(page_bg)
        l_bg.setContentsMargins(0, 0, 0, 0)
        l_bg.setSpacing(6)

        # Create accordion for BG subtraction settings
        self.bg_accordion = AccordionContainer()

        # Image Enhancement (Pre-processing)
        self.g_img = CollapsibleGroupBox("Image")
        self.bg_accordion.addCollapsible(self.g_img)
        vl_img = QVBoxLayout()
        vl_img.addWidget(
            self._main_window._create_help_label(
                "Adjust image properties before detection to improve contrast between animals and background. "
                "Start with default values and adjust only if animals are hard to distinguish."
            )
        )

        # Brightness slider
        bright_layout = QVBoxLayout()
        bright_label_row = QHBoxLayout()
        bright_label_row.addWidget(QLabel("Brightness"))
        self.label_brightness_val = QLabel("0")
        self.label_brightness_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        bright_label_row.addWidget(self.label_brightness_val)
        bright_label_row.addSpacing(6)
        bright_layout.addLayout(bright_label_row)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-255, 255)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTickPosition(QSlider.TicksBelow)
        self.slider_brightness.setTickInterval(50)
        self.slider_brightness.valueChanged.connect(
            self._main_window._on_brightness_changed
        )
        self.slider_brightness.setToolTip(
            "Adjust overall image brightness.\n"
            "Positive = lighter, Negative = darker.\n"
            "Use to improve contrast between animals and background."
        )
        bright_layout.addWidget(self.slider_brightness)
        vl_img.addLayout(bright_layout)

        # Contrast slider
        contrast_layout = QVBoxLayout()
        contrast_label_row = QHBoxLayout()
        contrast_label_row.addWidget(QLabel("Contrast"))
        self.label_contrast_val = QLabel("1.0")
        self.label_contrast_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        contrast_label_row.addWidget(self.label_contrast_val)
        contrast_label_row.addSpacing(6)
        contrast_layout.addLayout(contrast_label_row)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(0, 300)  # 0.0 to 3.0, scaled by 100
        self.slider_contrast.setValue(100)  # 1.0
        self.slider_contrast.setTickPosition(QSlider.TicksBelow)
        self.slider_contrast.setTickInterval(50)
        self.slider_contrast.valueChanged.connect(
            self._main_window._on_contrast_changed
        )
        self.slider_contrast.setToolTip(
            "Adjust image contrast (difference between light and dark).\n"
            "1.0 = original, >1.0 = more contrast, <1.0 = less contrast.\n"
            "Increase to make animals stand out from background."
        )
        contrast_layout.addWidget(self.slider_contrast)
        vl_img.addLayout(contrast_layout)

        # Gamma slider
        gamma_layout = QVBoxLayout()
        gamma_label_row = QHBoxLayout()
        gamma_label_row.addWidget(QLabel("Gamma"))
        self.label_gamma_val = QLabel("1.0")
        self.label_gamma_val.setStyleSheet("color: #4fc1ff; font-weight: bold;")
        gamma_label_row.addWidget(self.label_gamma_val)
        gamma_label_row.addSpacing(6)
        gamma_layout.addLayout(gamma_label_row)

        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 300)  # 0.1 to 3.0, scaled by 100
        self.slider_gamma.setValue(100)  # 1.0
        self.slider_gamma.setTickPosition(QSlider.TicksBelow)
        self.slider_gamma.setTickInterval(50)
        self.slider_gamma.valueChanged.connect(self._main_window._on_gamma_changed)
        self.slider_gamma.setToolTip(
            "Adjust gamma correction (mid-tone brightness).\n"
            "1.0 = original, >1.0 = brighter mid-tones, <1.0 = darker mid-tones.\n"
            "Use to enhance detail in shadowed or bright areas."
        )
        gamma_layout.addWidget(self.slider_gamma)
        vl_img.addLayout(gamma_layout)

        # Dark on light checkbox
        self.chk_dark_on_light = QCheckBox("Animals are darker than background")
        self.chk_dark_on_light.setChecked(True)
        self.chk_dark_on_light.setToolTip(
            "Check if animals are darker than background (most common).\n"
            "Uncheck if animals are lighter than background.\n"
            "This inverts the foreground detection."
        )
        vl_img.addWidget(self.chk_dark_on_light)
        self.g_img.setContentLayout(vl_img)
        l_bg.addWidget(self.g_img)
        self._main_window._remember_collapsible_state(
            "detection.brightness_contrast_gamma", self.g_img
        )

        # Background Model
        g_bg_model = CollapsibleGroupBox("Background")
        self.bg_accordion.addCollapsible(g_bg_model)
        vl_bg_model = QVBoxLayout()
        vl_bg_model.addWidget(
            self._main_window._create_help_label(
                "Build a model of the static background. Priming frames establish initial model, learning rate "
                "controls adaptation speed, threshold sets sensitivity. Lower threshold = more sensitive detection."
            )
        )
        f_bg = QFormLayout(None)
        f_bg.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_bg.setHorizontalSpacing(10)
        f_bg.setVerticalSpacing(8)
        self.spin_bg_prime = QDoubleSpinBox()
        self.spin_bg_prime.setRange(0.0, 120.0)
        self.spin_bg_prime.setSingleStep(0.5)
        self.spin_bg_prime.setDecimals(2)
        self.spin_bg_prime.setValue(0.33)
        self.spin_bg_prime.setFixedHeight(30)
        self.spin_bg_prime.setToolTip(
            "Time to build background model (seconds).\n"
            "Converted to frames using the acquisition frame rate.\n"
            "Recommended: 0.3-3.0 s.\n"
            "Use more if background varies or animals are present initially."
        )
        f_bg.addRow("Startup time (seconds)", self.spin_bg_prime)

        self.chk_adaptive_bg = QCheckBox("Continuously update background model")
        self.chk_adaptive_bg.setChecked(True)
        self.chk_adaptive_bg.setToolTip(
            "Continuously update background model during tracking.\n"
            "Recommended: Enable for videos with changing lighting.\n"
            "Disable for static background to improve performance."
        )
        f_bg.addRow(self.chk_adaptive_bg)

        self.spin_bg_learning = QDoubleSpinBox()
        self.spin_bg_learning.setRange(0.0001, 0.1)
        self.spin_bg_learning.setDecimals(4)
        self.spin_bg_learning.setValue(0.001)
        self.spin_bg_learning.setFixedHeight(30)
        self.spin_bg_learning.setToolTip(
            "How quickly background adapts to changes (0.0001-0.1).\n"
            "Lower = slower adaptation (stable, good for mostly static background).\n"
            "Higher = faster adaptation (use for variable lighting/shadows)."
        )
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(0, 255)
        self.spin_threshold.setValue(50)
        self.spin_threshold.setFixedHeight(30)
        self.spin_threshold.setToolTip(
            "Pixel intensity difference to detect foreground (0-255).\n"
            "Lower = more sensitive (detects subtle animals, more noise).\n"
            "Higher = less sensitive (cleaner, may miss animals).\n"
            "Recommended: 30-70 depending on contrast."
        )
        _bg_rate_row = QHBoxLayout()
        _bg_rate_row.addWidget(QLabel("Learn rate:"))
        _bg_rate_row.addWidget(self.spin_bg_learning, 1)
        _bg_rate_row.addSpacing(8)
        _bg_rate_row.addWidget(QLabel("Threshold:"))
        _bg_rate_row.addWidget(self.spin_threshold, 1)
        f_bg.addRow(_bg_rate_row)
        vl_bg_model.addLayout(f_bg)
        g_bg_model.setContentLayout(vl_bg_model)
        l_bg.addWidget(g_bg_model)
        self._main_window._remember_collapsible_state(
            "detection.background_estimation", g_bg_model
        )

        # Lighting Stab
        g_light = CollapsibleGroupBox("Lighting")
        self.bg_accordion.addCollapsible(g_light)
        vl_light = QVBoxLayout()
        vl_light.addWidget(
            self._main_window._create_help_label(
                "Compensate for gradual lighting changes (clouds, time of day). Smoothing factor controls "
                "adaptation speed - higher = slower/more stable. Enable for outdoor or variable-light videos."
            )
        )
        f_light = QFormLayout(None)
        f_light.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_light.setHorizontalSpacing(10)
        f_light.setVerticalSpacing(8)
        self.chk_lighting_stab = QCheckBox("Enable Stabilization")
        self.chk_lighting_stab.setChecked(True)
        self.chk_lighting_stab.setToolTip(
            "Compensate for gradual lighting changes over time.\n"
            "Recommended: Enable for videos with variable lighting.\n"
            "Disable for consistent illumination to improve speed."
        )
        f_light.addRow(self.chk_lighting_stab)

        self.spin_lighting_smooth = QDoubleSpinBox()
        self.spin_lighting_smooth.setRange(0.8, 0.999)
        self.spin_lighting_smooth.setValue(0.95)
        self.spin_lighting_smooth.setFixedHeight(30)
        self.spin_lighting_smooth.setToolTip(
            "Temporal smoothing factor for lighting correction (0.8-0.999).\n"
            "Higher = smoother, slower adaptation to lighting changes.\n"
            "Lower = faster response to sudden lighting shifts.\n"
            "Recommended: 0.9-0.98"
        )
        self.spin_lighting_median = QSpinBox()
        self.spin_lighting_median.setRange(3, 15)
        self.spin_lighting_median.setSingleStep(2)
        self.spin_lighting_median.setValue(5)
        self.spin_lighting_median.setFixedHeight(30)
        self.spin_lighting_median.setToolTip(
            "Median filter window size (odd number, 3-15).\n"
            "Larger window = smoother lighting estimate, slower response.\n"
            "Smaller window = faster response, less smoothing.\n"
            "Recommended: 5-9"
        )
        _light_row = QHBoxLayout()
        _light_row.addWidget(QLabel("Smoothing:"))
        _light_row.addWidget(self.spin_lighting_smooth, 1)
        _light_row.addSpacing(8)
        _light_row.addWidget(QLabel("Median (frames):"))
        _light_row.addWidget(self.spin_lighting_median, 1)
        f_light.addRow(_light_row)
        vl_light.addLayout(f_light)
        g_light.setContentLayout(vl_light)
        l_bg.addWidget(g_light)
        self._main_window._remember_collapsible_state(
            "detection.scene_lighting", g_light
        )

        # Morphology (Standard)
        g_morph = CollapsibleGroupBox("Morphology")
        self.bg_accordion.addCollapsible(g_morph)
        vl_morph = QVBoxLayout()
        vl_morph.addWidget(
            self._main_window._create_help_label(
                "Clean up detected blobs using morphological operations. Closing fills small holes, opening removes "
                "small noise. Larger kernels = stronger effect but may distort shape. Use odd numbers only."
            )
        )
        f_morph = QFormLayout(None)
        f_morph.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_morph.setHorizontalSpacing(10)
        f_morph.setVerticalSpacing(8)
        self.spin_morph_size = QSpinBox()
        self.spin_morph_size.setRange(1, 25)
        self.spin_morph_size.setSingleStep(2)
        self.spin_morph_size.setValue(5)
        self.spin_morph_size.setFixedHeight(30)
        self.spin_morph_size.setToolTip(
            "Morphological operation kernel size (odd number, 1-25).\n"
            "Larger = more aggressive noise removal, may merge nearby animals.\n"
            "Smaller = preserves detail, may leave noise.\n"
            "Recommended: 3-7 for typical tracking scenarios."
        )
        f_morph.addRow("Kernel size", self.spin_morph_size)

        self.spin_min_contour = QSpinBox()
        self.spin_min_contour.setRange(0, 100000)
        self.spin_min_contour.setValue(50)
        self.spin_min_contour.setFixedHeight(30)
        self.spin_min_contour.setToolTip(
            "Minimum contour area in pixels² to keep.\n"
            "Filters out small noise blobs after morphology.\n"
            "Recommended: 20-100 depending on animal size and zoom.\n"
            "Note: Similar to min object size but in absolute pixels."
        )
        self.spin_max_contour_multiplier = QSpinBox()
        self.spin_max_contour_multiplier.setRange(5, 100)
        self.spin_max_contour_multiplier.setValue(20)
        self.spin_max_contour_multiplier.setFixedHeight(30)
        self.spin_max_contour_multiplier.setToolTip(
            "Maximum contour area as multiplier of minimum (5-100).\n"
            "Max area = min_contour × this multiplier.\n"
            "Filters out very large blobs (clusters, shadows, artifacts).\n"
            "Recommended: 10-30"
        )
        _contour_row = QHBoxLayout()
        _contour_row.addWidget(QLabel("Min area (px²):"))
        _contour_row.addWidget(self.spin_min_contour, 1)
        _contour_row.addSpacing(8)
        _contour_row.addWidget(QLabel("Max multiplier:"))
        _contour_row.addWidget(self.spin_max_contour_multiplier, 1)
        f_morph.addRow(_contour_row)
        vl_morph.addLayout(f_morph)
        g_morph.setContentLayout(vl_morph)
        l_bg.addWidget(g_morph)
        self._main_window._remember_collapsible_state(
            "detection.noise_removal", g_morph
        )

        # Morphology (Advanced/Splitting)
        g_split = CollapsibleGroupBox("Split Touching")
        self.bg_accordion.addCollapsible(g_split)
        vl_split = QVBoxLayout()
        vl_split.addWidget(
            self._main_window._create_help_label(
                "Split touching animals using body-size-aware erosion only in locally crowded regions. "
                "Enable only if animals frequently touch."
            )
        )
        f_split = QFormLayout(None)
        f_split.setHorizontalSpacing(10)
        f_split.setVerticalSpacing(8)
        self.chk_conservative_split = QCheckBox("Use conservative split")
        self.chk_conservative_split.setChecked(True)
        self.chk_conservative_split.setToolTip(
            "Locally raise the detection threshold inside suspected merged\n"
            "blobs to separate touching animals at their weakest connection\n"
            "point while preserving body shape."
        )
        f_split.addRow(self.chk_conservative_split)

        h_split = QHBoxLayout()
        self.spin_conservative_kernel = QSpinBox()
        self.spin_conservative_kernel.setRange(1, 15)
        self.spin_conservative_kernel.setSingleStep(2)
        self.spin_conservative_kernel.setValue(3)
        self.spin_conservative_kernel.setFixedHeight(30)
        self.spin_conservative_kernel.setToolTip(
            "Gaussian blur kernel applied to the local difference\n"
            "image before re-thresholding (odd number, 1-15).\n"
            "Larger = smoother split boundaries.\n"
            "1 = no smoothing. Recommended: 3-5"
        )
        self.spin_conservative_erode = QSpinBox()
        self.spin_conservative_erode.setRange(1, 10)
        self.spin_conservative_erode.setValue(1)
        self.spin_conservative_erode.setFixedHeight(30)
        self.spin_conservative_erode.setToolTip(
            "Threshold boost steps (1-10).\n"
            "Each step pulls the split threshold 25%% closer to\n"
            "nearby local peaks inside suspected merged blobs.\n"
            "Higher = more aggressive local separation.\n"
            "Recommended: 1-3"
        )
        h_split.addWidget(QLabel("Blur kernel"))
        h_split.addWidget(self.spin_conservative_kernel)
        h_split.addWidget(QLabel("Boost steps"))
        h_split.addWidget(self.spin_conservative_erode)
        f_split.addRow(h_split)

        self.chk_additional_dilation = QCheckBox("Reconnect thin parts (dilation)")
        self.chk_additional_dilation.setToolTip(
            "Use dilation to reconnect thin parts (e.g., legs, antennae).\n"
            "Recommended: Enable if animals have thin appendages.\n"
            "Disable to maintain accurate body shape."
        )
        f_split.addRow(self.chk_additional_dilation)

        h_dil = QHBoxLayout()
        self.spin_dilation_kernel_size = QSpinBox()
        self.spin_dilation_kernel_size.setRange(1, 15)
        self.spin_dilation_kernel_size.setSingleStep(2)
        self.spin_dilation_kernel_size.setValue(3)
        self.spin_dilation_kernel_size.setFixedHeight(30)
        self.spin_dilation_kernel_size.setToolTip(
            "Dilation kernel size (odd number, 1-15).\n"
            "Larger = thicker reconnection.\n"
            "Recommended: 3-5"
        )
        self.spin_dilation_iterations = QSpinBox()
        self.spin_dilation_iterations.setRange(1, 10)
        self.spin_dilation_iterations.setValue(2)
        self.spin_dilation_iterations.setFixedHeight(30)
        self.spin_dilation_iterations.setToolTip(
            "Number of dilation iterations (1-10).\n"
            "More iterations = thicker result.\n"
            "Recommended: 1-3"
        )
        h_dil.addWidget(QLabel("Kernel size"))
        h_dil.addWidget(self.spin_dilation_kernel_size)
        h_dil.addWidget(QLabel("Iterations"))
        h_dil.addWidget(self.spin_dilation_iterations)
        f_split.addRow(h_dil)
        vl_split.addLayout(f_split)
        g_split.setContentLayout(vl_split)

        l_bg.addWidget(g_split)
        self._main_window._remember_collapsible_state(
            "detection.split_touching", g_split
        )

        # --- Auto-Tune Detection Parameters button ---
        self.btn_bg_autotune = QPushButton("Auto-Tune Detection Parameters…")
        self.btn_bg_autotune.setToolTip(
            "Use Optuna to search for the best threshold, morphology,\n"
            "and conservative-split settings for your video."
        )
        self.btn_bg_autotune.clicked.connect(
            self._main_window._open_bg_parameter_helper
        )
        l_bg.addWidget(self.btn_bg_autotune)

        # --- Page 1: YOLO Params ---
        page_yolo = QWidget()
        page_yolo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        l_yolo = QVBoxLayout(page_yolo)
        l_yolo.setContentsMargins(0, 0, 0, 0)
        l_yolo.setSpacing(6)

        self.yolo_group = QGroupBox("YOLO")
        self._main_window._set_compact_section_widget(self.yolo_group)
        f_yolo = QFormLayout(self.yolo_group)
        f_yolo.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_yolo.setHorizontalSpacing(10)
        f_yolo.setVerticalSpacing(8)
        self.yolo_group.layout().setContentsMargins(9, 10, 9, 9)
        self.yolo_group.layout().setSpacing(8)
        self.yolo_group.layout().addWidget(
            self._main_window._create_help_label(
                "YOLO uses a trained neural network to detect animals. Choose your model file and adjust thresholds to balance recall and false positives."
            )
        )

        self.combo_yolo_obb_mode = QComboBox()
        self.combo_yolo_obb_mode.addItems(["Direct", "Sequential (Faster)"])
        self.combo_yolo_obb_mode.setFixedHeight(30)
        self.combo_yolo_obb_mode.currentIndexChanged.connect(
            self._main_window._on_yolo_mode_changed
        )
        self.combo_yolo_obb_mode.setToolTip(
            "Direct: run OBB on full frame.\n"
            "Sequential: run detect model first, crop detections, then run OBB on crops."
        )
        f_yolo.addRow("YOLO OBB mode", self.combo_yolo_obb_mode)

        self.lbl_obb_mode_warning = QLabel()
        self.lbl_obb_mode_warning.setWordWrap(True)
        self.lbl_obb_mode_warning.setStyleSheet(
            "color: #f0ad4e; font-style: italic; padding: 2px 0px;"
        )
        self.lbl_obb_mode_warning.setVisible(False)
        f_yolo.addRow("", self.lbl_obb_mode_warning)

        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.activated.connect(self._main_window.on_yolo_model_changed)
        self.combo_yolo_model.setFixedHeight(30)
        self.combo_yolo_model.setToolTip("Direct-mode YOLO OBB model.")
        f_yolo.addRow("Direct OBB model", self.combo_yolo_model)

        self.combo_yolo_detect_model = QComboBox()
        self.combo_yolo_detect_model.activated.connect(
            self._main_window.on_yolo_detect_model_changed
        )
        self.combo_yolo_detect_model.setFixedHeight(30)
        self.combo_yolo_detect_model.setToolTip(
            "Sequential stage-1 model (axis-aligned detect)."
        )
        f_yolo.addRow("Seq detect model", self.combo_yolo_detect_model)

        self.combo_yolo_crop_obb_model = QComboBox()
        self.combo_yolo_crop_obb_model.activated.connect(
            self._main_window.on_yolo_crop_obb_model_changed
        )
        self.combo_yolo_crop_obb_model.setFixedHeight(30)
        self.combo_yolo_crop_obb_model.setToolTip(
            "Sequential stage-2 OBB model trained on cropped detections."
        )
        f_yolo.addRow("Seq crop OBB model", self.combo_yolo_crop_obb_model)

        self.yolo_seq_advanced = CollapsibleGroupBox(
            "Sequential Advanced Settings", initially_expanded=False
        )
        self.yolo_seq_advanced_content = QWidget()
        f_seq_adv = QFormLayout(self.yolo_seq_advanced_content)
        self.spin_yolo_seq_crop_pad = QDoubleSpinBox()
        self.spin_yolo_seq_crop_pad.setRange(0.0, 1.0)
        self.spin_yolo_seq_crop_pad.setSingleStep(0.01)
        self.spin_yolo_seq_crop_pad.setValue(0.15)
        self.spin_yolo_seq_crop_pad.setFixedHeight(30)
        f_seq_adv.addRow("Crop pad ratio", self.spin_yolo_seq_crop_pad)
        self.spin_yolo_seq_min_crop_px = QSpinBox()
        self.spin_yolo_seq_min_crop_px.setRange(8, 1024)
        self.spin_yolo_seq_min_crop_px.setValue(64)
        self.spin_yolo_seq_min_crop_px.setFixedHeight(30)
        f_seq_adv.addRow("Min crop size (px)", self.spin_yolo_seq_min_crop_px)
        self.chk_yolo_seq_square_crop = QCheckBox("Enforce square crop")
        self.chk_yolo_seq_square_crop.setChecked(True)
        f_seq_adv.addRow("", self.chk_yolo_seq_square_crop)
        self.spin_yolo_seq_detect_conf = QDoubleSpinBox()
        self.spin_yolo_seq_detect_conf.setRange(0.01, 1.0)
        self.spin_yolo_seq_detect_conf.setSingleStep(0.01)
        self.spin_yolo_seq_detect_conf.setValue(0.25)
        self.spin_yolo_seq_detect_conf.setFixedHeight(30)
        self.spin_yolo_seq_detect_conf.setToolTip(
            "Minimum confidence for the stage-1 detection model (sequential mode only).\n"
            "Lower = more crops sent to stage-2 (higher recall, slower).\n"
            "Higher = fewer crops (faster, may miss occluded animals).\n"
            "Recommended: 0.1–0.3"
        )
        f_seq_adv.addRow("Stage-1 detect conf", self.spin_yolo_seq_detect_conf)
        self.spin_yolo_seq_stage2_imgsz = QSpinBox()
        self.spin_yolo_seq_stage2_imgsz.setRange(0, 2048)
        self.spin_yolo_seq_stage2_imgsz.setValue(160)
        self.spin_yolo_seq_stage2_imgsz.setFixedHeight(30)
        self.spin_yolo_seq_stage2_imgsz.setToolTip(
            "Crop OBB stage input size in pixels. Set 0 to disable pre-resize."
        )
        f_seq_adv.addRow("Stage-2 imgsz (px)", self.spin_yolo_seq_stage2_imgsz)
        self.chk_yolo_seq_stage2_pow2_pad = QCheckBox(
            "Pad stage-2 batch to power-of-two"
        )
        self.chk_yolo_seq_stage2_pow2_pad.setChecked(False)
        self.chk_yolo_seq_stage2_pow2_pad.setToolTip(
            "Reduces dynamic batch-size variants in sequential stage-2 inference."
        )
        f_seq_adv.addRow("", self.chk_yolo_seq_stage2_pow2_pad)
        self.yolo_seq_advanced.setContentLayout(f_seq_adv)
        f_yolo.addRow(self.yolo_seq_advanced)

        self.spin_yolo_confidence = QDoubleSpinBox()
        self.spin_yolo_confidence.setRange(0.01, 1.0)
        self.spin_yolo_confidence.setValue(0.25)
        self.spin_yolo_confidence.setFixedHeight(30)
        self.spin_yolo_confidence.setToolTip(
            "Minimum confidence score for YOLO detections (0.01-1.0).\n"
            "Lower = more detections (more false positives).\n"
            "Higher = fewer detections (may miss animals).\n"
            "Recommended: 0.2-0.4"
        )
        self.spin_yolo_iou = QDoubleSpinBox()
        self.spin_yolo_iou.setRange(0.01, 1.0)
        self.spin_yolo_iou.setValue(0.7)
        self.spin_yolo_iou.setFixedHeight(30)
        self.spin_yolo_iou.setToolTip(
            "Intersection-over-Union threshold for non-max suppression (0.01-1.0).\n"
            "Lower = more aggressive duplicate removal.\n"
            "Higher = keep more overlapping detections.\n"
            "Recommended: 0.5-0.8"
        )
        _yolo_thresh_row = QHBoxLayout()
        _yolo_thresh_row.addWidget(QLabel("Confidence:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_confidence, 1)
        _yolo_thresh_row.addSpacing(8)
        _yolo_thresh_row.addWidget(QLabel("IOU:"))
        _yolo_thresh_row.addWidget(self.spin_yolo_iou, 1)
        f_yolo.addRow(_yolo_thresh_row)

        self.chk_use_custom_obb_iou = QCheckBox("Use custom OBB overlap filtering")
        self.chk_use_custom_obb_iou.setChecked(True)
        self.chk_use_custom_obb_iou.setEnabled(False)
        self.chk_use_custom_obb_iou.setToolTip(
            "Custom polygon-based OBB IOU filtering is always enabled.\n"
            "This improves overlap handling consistency across cached and live detections."
        )
        self.chk_use_custom_obb_iou.setVisible(False)

        self.line_yolo_classes = QLineEdit()
        self.line_yolo_classes.setFixedHeight(30)
        self.line_yolo_classes.setPlaceholderText("e.g. 15, 16 (Empty for all)")
        self.line_yolo_classes.setToolTip(
            "Comma-separated class IDs to detect (leave empty for all classes).\n"
            "Example: '0,1,2' to detect only classes 0, 1, and 2.\n"
            "Refer to your model's class definitions."
        )
        f_yolo.addRow("Classes (optional)", self.line_yolo_classes)
        self._main_window._on_yolo_mode_changed(self.combo_yolo_obb_mode.currentIndex())

        l_yolo.addWidget(self.yolo_group)

        # ============================================================
        # YOLO Inference Acceleration (TensorRT + Batching)
        # ============================================================
        self.g_gpu_accel = QGroupBox("Inference Acceleration")
        self._main_window._set_compact_section_widget(self.g_gpu_accel)
        vl_gpu = QVBoxLayout(self.g_gpu_accel)
        vl_gpu.setSpacing(6)
        vl_gpu.addWidget(
            self._main_window._create_help_label(
                "Control YOLO throughput features. Use batching for faster full-run detection, and TensorRT when NVIDIA export/runtime support is available."
            )
        )

        # TensorRT Optimization
        self.chk_enable_tensorrt = QCheckBox("TensorRT engine")
        self.chk_enable_tensorrt.setChecked(False)
        self.chk_enable_tensorrt.setEnabled(TENSORRT_AVAILABLE)

        tensorrt_tooltip = (
            "Enable NVIDIA TensorRT for 2-5× faster YOLO inference.\n"
            "Requires NVIDIA GPU with CUDA.\n"
            "First run will export model (1-5 min), then cached for future use.\n"
            "Uses FP16 precision for maximum speed.\n"
        )
        if TENSORRT_AVAILABLE:
            tensorrt_tooltip += "\n✓ TensorRT is available on this system"
        else:
            tensorrt_tooltip += (
                "\n✗ TensorRT not available (requires NVIDIA GPU + tensorrt package)"
            )

        self.chk_enable_tensorrt.setToolTip(tensorrt_tooltip)
        self.chk_enable_tensorrt.stateChanged.connect(
            self._main_window._on_tensorrt_toggled
        )

        self.spin_tensorrt_batch = QSpinBox()
        self.spin_tensorrt_batch.setRange(1, 64)
        self.spin_tensorrt_batch.setValue(
            self._main_window.advanced_config.get("tensorrt_max_batch_size", 16)
        )
        self.spin_tensorrt_batch.setFixedHeight(30)
        self.spin_tensorrt_batch.setToolTip(
            "Maximum batch size for TensorRT engine.\n"
            "Higher = potentially faster, Lower = more stable.\n"
            "Reduce if build fails (try 8, 4, or 1).\n"
            "Typical: 16-32 (high-end), 8-16 (mid-range), 1-8 (low VRAM)"
        )
        self.lbl_tensorrt_batch = QLabel("TensorRT max batch")
        self.lbl_tensorrt_batch.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        self.chk_enable_yolo_batching = QCheckBox("GPU batching")
        self.chk_enable_yolo_batching.setChecked(
            self._main_window.advanced_config.get("enable_yolo_batching", True)
        )
        self.chk_enable_yolo_batching.setToolTip(
            "Process frames in batches on GPU for 2-5× faster detection.\n"
            "Only works in full tracking mode (not preview)."
        )
        self.chk_enable_yolo_batching.stateChanged.connect(
            self._main_window._on_yolo_batching_toggled
        )

        self.combo_yolo_batch_mode = QComboBox()
        self.combo_yolo_batch_mode.addItems(["Auto", "Manual"])
        self.combo_yolo_batch_mode.setFixedHeight(30)
        self.combo_yolo_batch_mode.setToolTip(
            "Auto: Automatically estimate batch size based on GPU memory.\n"
            "Manual: Specify a fixed batch size."
        )
        self.combo_yolo_batch_mode.currentIndexChanged.connect(
            self._main_window._on_yolo_batch_mode_changed
        )
        self.lbl_yolo_batch_mode = QLabel("Batch mode")
        self.lbl_yolo_batch_mode.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        self.spin_yolo_batch_size = QSpinBox()
        self.spin_yolo_batch_size.setRange(1, 64)
        self.spin_yolo_batch_size.setValue(
            self._main_window.advanced_config.get("yolo_manual_batch_size", 16)
        )
        self.spin_yolo_batch_size.setFixedHeight(30)
        self.spin_yolo_batch_size.setToolTip(
            "Manual batch size (only used when mode is Manual).\n"
            "Larger = faster but uses more GPU memory.\n"
            "Typical values: 8-32 depending on GPU."
        )
        self.spin_yolo_batch_size.valueChanged.connect(
            self._main_window._on_yolo_manual_batch_size_changed
        )
        self.lbl_yolo_batch_size = QLabel("Manual batch")
        self.lbl_yolo_batch_size.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: #bdbdbd;"
        )

        accel_toggle_grid = QGridLayout()
        accel_toggle_grid.setContentsMargins(0, 0, 0, 0)
        accel_toggle_grid.setHorizontalSpacing(12)
        accel_toggle_grid.setVerticalSpacing(6)
        accel_toggle_grid.addWidget(self.chk_enable_yolo_batching, 0, 0)
        accel_toggle_grid.addWidget(self.chk_enable_tensorrt, 0, 1)
        accel_toggle_grid.setColumnStretch(0, 1)
        accel_toggle_grid.setColumnStretch(1, 1)
        vl_gpu.addLayout(accel_toggle_grid)

        accel_controls_grid = QGridLayout()
        accel_controls_grid.setContentsMargins(0, 0, 0, 0)
        accel_controls_grid.setHorizontalSpacing(12)
        accel_controls_grid.setVerticalSpacing(4)
        accel_controls_grid.addWidget(self.lbl_yolo_batch_mode, 0, 0)
        accel_controls_grid.addWidget(self.lbl_yolo_batch_size, 0, 1)
        accel_controls_grid.addWidget(self.lbl_tensorrt_batch, 0, 2)
        accel_controls_grid.addWidget(self.combo_yolo_batch_mode, 1, 0)
        accel_controls_grid.addWidget(self.spin_yolo_batch_size, 1, 1)
        accel_controls_grid.addWidget(self.spin_tensorrt_batch, 1, 2)
        accel_controls_grid.setColumnStretch(0, 1)
        accel_controls_grid.setColumnStretch(1, 1)
        accel_controls_grid.setColumnStretch(2, 1)
        vl_gpu.addLayout(accel_controls_grid)
        l_yolo.addWidget(self.g_gpu_accel)

        # Set initial visibility for TensorRT widgets
        self.chk_enable_tensorrt.setVisible(False)
        self.spin_tensorrt_batch.setVisible(False)
        self.lbl_tensorrt_batch.setVisible(False)

        # Set initial visibility for batching widgets
        initial_batching_enabled = self.chk_enable_yolo_batching.isChecked()
        self.combo_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_mode.setVisible(initial_batching_enabled)
        self.spin_yolo_batch_size.setVisible(initial_batching_enabled)
        self.lbl_yolo_batch_size.setVisible(initial_batching_enabled)
        self.combo_yolo_batch_mode.setEnabled(initial_batching_enabled)
        self.spin_yolo_batch_size.setEnabled(False)  # Auto mode by default
        # Add pages to stack
        self.stack_detection.addWidget(page_bg)
        self.stack_detection.addWidget(page_yolo)

        vbox.addWidget(self.stack_detection)

        # ============================================================
        # Detection Overlays (moved from Visuals tab)
        # ============================================================
        # Background Subtraction specific overlays
        self.g_overlays_bg = QGroupBox("Background Diagnostics")
        self._main_window._set_compact_section_widget(self.g_overlays_bg)
        v_ov_bg = QVBoxLayout(self.g_overlays_bg)
        v_ov_bg.addWidget(
            self._main_window._create_help_label(
                "Debug background subtraction by viewing the foreground mask (detected movement) "
                "and background model (learned static image)."
            )
        )

        self.chk_show_fg = QCheckBox("Show Foreground Mask")
        self.chk_show_fg.setChecked(True)
        self.chk_show_bg = QCheckBox("Show Background Model")
        self.chk_show_bg.setChecked(True)
        _bg_ov_row = QHBoxLayout()
        _bg_ov_row.addWidget(self.chk_show_fg)
        _bg_ov_row.addWidget(self.chk_show_bg)
        v_ov_bg.addLayout(_bg_ov_row)

        vbox.addWidget(self.g_overlays_bg)

        # YOLO specific overlays
        self.g_overlays_yolo = QGroupBox("YOLO Diagnostics")
        self._main_window._set_compact_section_widget(self.g_overlays_yolo)
        v_ov_yolo = QVBoxLayout(self.g_overlays_yolo)
        v_ov_yolo.addWidget(
            self._main_window._create_help_label(
                "Show oriented bounding boxes from YOLO detection. Useful for debugging detection quality "
                "and verifying model performance."
            )
        )

        self.chk_show_yolo_obb = QCheckBox("Show YOLO OBB Detection Boxes")
        self.chk_show_yolo_obb.setChecked(False)
        v_ov_yolo.addWidget(self.chk_show_yolo_obb)

        vbox.addWidget(self.g_overlays_yolo)

        # Initially show/hide based on detection method (will be set properly by combo)
        self.g_overlays_bg.setVisible(True)
        self.g_overlays_yolo.setVisible(False)

        # ============================================================
        # Reference Scale (size + aspect ratio)
        # ============================================================
        g_ref_scale = QGroupBox("Reference Scale")
        self._main_window._set_compact_section_widget(g_ref_scale)
        vl_ref_scale = QVBoxLayout(g_ref_scale)
        vl_ref_scale.addWidget(
            self._main_window._create_help_label(
                "Define the spatial scale for tracking. These reference values make all distance, "
                "size, and shape parameters portable across videos and species. Set them BEFORE "
                "configuring tracking parameters. Use 'Test Detection' then the Auto-Set buttons "
                "to have values estimated automatically from a sample frame."
            )
        )
        fl_ref = QFormLayout(None)
        fl_ref.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        fl_ref.setHorizontalSpacing(10)
        fl_ref.setVerticalSpacing(8)

        self.spin_reference_body_size = QDoubleSpinBox()
        self.spin_reference_body_size.setRange(1.0, 500.0)
        self.spin_reference_body_size.setSingleStep(1.0)
        self.spin_reference_body_size.setValue(20.0)
        self.spin_reference_body_size.setDecimals(2)
        self.spin_reference_body_size.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.spin_reference_body_size.setFixedHeight(30)
        self.spin_reference_body_size.setToolTip(
            "Reference animal body diameter in pixels (at resize=1.0).\n"
            "All distance/size parameters are scaled relative to this value."
        )
        self.spin_reference_body_size.valueChanged.connect(
            self._main_window._update_body_size_info
        )
        fl_ref.addRow("Reference body size (px)", self.spin_reference_body_size)

        self.label_body_size_info = QLabel()
        self.label_body_size_info.setStyleSheet(
            "color: #6a6a6a; font-size: 10px; font-style: italic;"
        )
        fl_ref.addRow("", self.label_body_size_info)

        self.spin_reference_aspect_ratio = QDoubleSpinBox()
        self.spin_reference_aspect_ratio.setRange(1.0, 20.0)
        self.spin_reference_aspect_ratio.setSingleStep(0.1)
        self.spin_reference_aspect_ratio.setDecimals(2)
        self.spin_reference_aspect_ratio.setValue(2.0)
        self.spin_reference_aspect_ratio.setFixedHeight(30)
        self.spin_reference_aspect_ratio.setToolTip(
            "Species-typical major/minor axis ratio.\n"
            "Used for adaptive canonical crop dimensions and aspect ratio filtering.\n"
            "Click 'Auto-Set Aspect Ratio' to detect from sample frames."
        )
        fl_ref.addRow("Reference aspect ratio", self.spin_reference_aspect_ratio)

        vl_ref_scale.addLayout(fl_ref)

        self.label_detection_stats = QLabel(
            "No detection data yet.\nRun 'Test Detection' to estimate sizes."
        )
        self.label_detection_stats.setStyleSheet(
            "color: #9a9a9a; font-size: 11px; padding: 8px; "
            "background-color: #252526; border-radius: 4px;"
        )
        self.label_detection_stats.setWordWrap(True)
        vl_ref_scale.addWidget(self.label_detection_stats)

        btn_layout = QHBoxLayout()
        self.btn_auto_set_body_size = QPushButton("Auto-Set Body Size from Median")
        self.btn_auto_set_body_size.clicked.connect(
            self._main_window._auto_set_body_size_from_detection
        )
        self.btn_auto_set_body_size.setEnabled(False)
        self.btn_auto_set_body_size.setToolTip(
            "Automatically set reference body size to the median detected diameter"
        )
        btn_layout.addWidget(self.btn_auto_set_body_size)

        self.btn_auto_set_aspect_ratio = QPushButton("Auto-Set Aspect Ratio")
        self.btn_auto_set_aspect_ratio.clicked.connect(
            self._main_window._auto_set_aspect_ratio_from_detection
        )
        self.btn_auto_set_aspect_ratio.setEnabled(False)
        self.btn_auto_set_aspect_ratio.setToolTip(
            "Set reference aspect ratio from the median detected major/minor ratio"
        )
        btn_layout.addWidget(self.btn_auto_set_aspect_ratio)
        vl_ref_scale.addLayout(btn_layout)

        vbox.addWidget(g_ref_scale)

        # ============================================================
        # Detection Filters (size + aspect ratio ranges)
        # ============================================================
        g_filters = QGroupBox("Detection Filters")
        self._main_window._set_compact_section_widget(g_filters)
        vl_filters = QVBoxLayout(g_filters)
        vl_filters.addWidget(
            self._main_window._create_help_label(
                "Filter detections by size and aspect ratio relative to the reference values above. "
                "Enabling these removes noise, debris, and erroneous clusters before tracking."
            )
        )
        f_filters = QFormLayout(None)
        f_filters.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        f_filters.setHorizontalSpacing(10)
        f_filters.setVerticalSpacing(8)

        self.chk_size_filtering = QCheckBox("Filter detections by size")
        self.chk_size_filtering.setToolTip(
            "Filter detected objects by area to remove noise and artifacts.\n"
            "Recommended: Enable for cleaner tracking."
        )
        f_filters.addRow(self.chk_size_filtering)

        h_sf = QHBoxLayout()
        self.spin_min_object_size = QDoubleSpinBox()
        self.spin_min_object_size.setRange(0.1, 5.0)
        self.spin_min_object_size.setSingleStep(0.1)
        self.spin_min_object_size.setDecimals(2)
        self.spin_min_object_size.setValue(0.3)
        self.spin_min_object_size.setFixedHeight(30)
        self.spin_min_object_size.setToolTip(
            "Minimum object area as multiple of reference body area.\n"
            "Filters out small noise/artifacts.\n"
            "Recommended: 0.2-0.5× (allows partial occlusion)"
        )
        self.spin_max_object_size = QDoubleSpinBox()
        self.spin_max_object_size.setRange(0.5, 10.0)
        self.spin_max_object_size.setSingleStep(0.1)
        self.spin_max_object_size.setDecimals(2)
        self.spin_max_object_size.setValue(3.0)
        self.spin_max_object_size.setFixedHeight(30)
        self.spin_max_object_size.setToolTip(
            "Maximum object area as multiple of reference body area.\n"
            "Filters out large clusters or artifacts.\n"
            "Recommended: 2-4× (handles overlapping animals)"
        )
        h_sf.addWidget(QLabel("Min size (body lengths)"))
        h_sf.addWidget(self.spin_min_object_size)
        h_sf.addWidget(QLabel("Max size (body lengths)"))
        h_sf.addWidget(self.spin_max_object_size)
        f_filters.addRow(h_sf)

        self.chk_enable_aspect_ratio_filtering = QCheckBox(
            "Filter detections by aspect ratio"
        )
        self.chk_enable_aspect_ratio_filtering.setChecked(False)
        self.chk_enable_aspect_ratio_filtering.setToolTip(
            "Reject detections with aspect ratios outside the expected range.\n"
            "Helps filter scratches, debris, and other non-animal detections."
        )
        f_filters.addRow(self.chk_enable_aspect_ratio_filtering)

        h_ar_mult = QHBoxLayout()
        self.spin_min_ar_multiplier = QDoubleSpinBox()
        self.spin_min_ar_multiplier.setRange(0.1, 1.0)
        self.spin_min_ar_multiplier.setSingleStep(0.05)
        self.spin_min_ar_multiplier.setDecimals(2)
        self.spin_min_ar_multiplier.setValue(0.5)
        self.spin_min_ar_multiplier.setFixedHeight(30)
        self.spin_min_ar_multiplier.setToolTip(
            "Minimum aspect ratio = reference × this multiplier.\n"
            "Detections more compact than this are rejected."
        )
        self.spin_max_ar_multiplier = QDoubleSpinBox()
        self.spin_max_ar_multiplier.setRange(1.0, 10.0)
        self.spin_max_ar_multiplier.setSingleStep(0.1)
        self.spin_max_ar_multiplier.setDecimals(2)
        self.spin_max_ar_multiplier.setValue(2.0)
        self.spin_max_ar_multiplier.setFixedHeight(30)
        self.spin_max_ar_multiplier.setToolTip(
            "Maximum aspect ratio = reference × this multiplier.\n"
            "Detections more elongated than this are rejected."
        )
        h_ar_mult.addWidget(QLabel("Min multiplier"))
        h_ar_mult.addWidget(self.spin_min_ar_multiplier)
        h_ar_mult.addWidget(QLabel("Max multiplier"))
        h_ar_mult.addWidget(self.spin_max_ar_multiplier)
        f_filters.addRow(h_ar_mult)

        vl_filters.addLayout(f_filters)
        vbox.addWidget(g_filters)

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def apply_config(self, config: TrackerConfig) -> None:
        """Update panel widgets to reflect a new config object."""
        self._config = config
