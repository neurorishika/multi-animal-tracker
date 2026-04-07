"""SessionOrchestrator — logging, progress, UI state machine."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hydra_suite.trackerkit.config.schemas import TrackerConfig
    from hydra_suite.trackerkit.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Manages session logging, progress display, and UI state transitions."""

    def __init__(self, main_window: "MainWindow", config: "TrackerConfig", panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels

    # =========================================================================
    # UI STATE MACHINE
    # =========================================================================

    def _set_ui_controls_enabled(self, enabled: bool):
        if enabled:
            if self._mw.current_video_path:
                self._apply_ui_state("idle")
            else:
                self._apply_ui_state("no_video")
            return

        # Disabled state - choose mode based on tracking/preview status
        if self._mw.tracking_worker and self._mw.tracking_worker.isRunning():
            if self._mw.btn_preview.isChecked():
                self._apply_ui_state("preview")
            else:
                self._apply_ui_state("tracking")
        else:
            self._apply_ui_state("locked")

    def _collect_preview_controls(self):
        return [
            self._mw.btn_test_detection,
            self._panels.setup.slider_timeline,
            self._panels.setup.btn_first_frame,
            self._panels.setup.btn_prev_frame,
            self._panels.setup.btn_play_pause,
            self._panels.setup.btn_next_frame,
            self._panels.setup.btn_last_frame,
            self._panels.setup.btn_random_seek,
            self._panels.setup.combo_playback_speed,
            self._panels.setup.spin_start_frame,
            self._panels.setup.spin_end_frame,
            self._panels.setup.btn_set_start_current,
            self._panels.setup.btn_set_end_current,
            self._panels.setup.btn_reset_range,
        ]

    def _set_interactive_widgets_enabled(
        self,
        enabled: bool,
        allowlist=None,
        blocklist=None,
        remember_state: bool = True,
    ):
        from PySide6.QtWidgets import (
            QAbstractButton,
            QComboBox,
            QDoubleSpinBox,
            QLineEdit,
            QSlider,
            QSpinBox,
        )

        allow = set(allowlist or [])
        block = set(blocklist or [])
        interactive_types = (
            QAbstractButton,
            QLineEdit,
            QComboBox,
            QSpinBox,
            QDoubleSpinBox,
            QSlider,
        )
        # Exclude welcome-page widgets — they are managed by WelcomePage itself
        welcome = getattr(self._mw, "_welcome_page", None)
        widgets = []
        for widget_type in interactive_types:
            for w in self._mw.findChildren(widget_type):
                if welcome is not None and welcome.isAncestorOf(w):
                    continue
                widgets.append(w)

        if enabled and remember_state and self._mw._saved_widget_enabled_states:
            for widget in widgets:
                if widget in block:
                    widget.setEnabled(False)
                elif widget in allow:
                    widget.setEnabled(True)
                elif widget in self._mw._saved_widget_enabled_states:
                    widget.setEnabled(self._mw._saved_widget_enabled_states[widget])
            self._mw._saved_widget_enabled_states = {}
            return

        if not enabled and remember_state:
            for widget in widgets:
                if widget in block or widget in allow:
                    continue
                self._mw._saved_widget_enabled_states[widget] = widget.isEnabled()

        for widget in widgets:
            if widget in block:
                widget.setEnabled(False)
            elif widget in allow:
                widget.setEnabled(True)
            else:
                widget.setEnabled(enabled)

    def _set_video_interaction_enabled(self, enabled: bool):
        self._mw._video_interactions_enabled = enabled
        self._mw.slider_zoom.setEnabled(enabled)
        # Keep the viewport enabled so placeholder/logo rendering is not dimmed
        # by disabled-widget styling (notably on macOS).
        self._mw.scroll.setEnabled(True)
        if not enabled:
            self._mw.video_label.unsetCursor()

    def _sync_contextual_controls(self):
        # ROI
        self._mw.btn_finish_roi.setEnabled(self._mw.roi_selection_active)
        self._mw.btn_undo_roi.setEnabled(len(self._mw.roi_shapes) > 0)
        self._mw.btn_clear_roi.setEnabled(
            len(self._mw.roi_shapes) > 0 or self._mw.roi_selection_active
        )

        # Crop video only if ROI exists and video loaded
        if hasattr(self._mw, "btn_crop_video"):
            self._mw.btn_crop_video.setEnabled(
                bool(self._mw.roi_shapes) and bool(self._mw.current_video_path)
            )

    def _apply_ui_state(self, state: str):
        if state == "no_video":
            extra_allowed = [
                self._mw._dataset_panel.combo_xanylabeling_env,
                self._mw._dataset_panel.btn_refresh_envs,
                self._mw._dataset_panel.btn_open_xanylabeling,
                self._mw._dataset_panel.btn_open_pose_label,
            ]
            self._set_interactive_widgets_enabled(
                False,
                allowlist=[
                    self._panels.setup.btn_file,
                    self._panels.setup.btn_load_config,
                ]
                + extra_allowed,
                remember_state=False,
            )
            self._mw.btn_start.setEnabled(False)
            self._mw.btn_preview.setEnabled(False)
            if hasattr(self._mw, "_tracking_panel"):
                self._mw._tracking_panel.btn_param_helper.setEnabled(False)
            self._set_video_interaction_enabled(False)
            self._panels.setup.g_video_player.setVisible(False)
            self._show_video_logo_placeholder()
            return

        if state == "idle":
            self._set_interactive_widgets_enabled(True)
            self._mw.btn_start.setEnabled(True)
            self._mw.btn_preview.setEnabled(True)
            if hasattr(self._mw, "_tracking_panel"):
                self._mw._tracking_panel.btn_param_helper.setEnabled(True)
            self._set_video_interaction_enabled(True)
            self._sync_contextual_controls()
            return

        if state == "tracking":
            allow = [self._mw.btn_start]
            if self._mw._is_visualization_enabled():
                allow.append(self._mw.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self._mw.btn_start.setEnabled(True)
            self._set_video_interaction_enabled(self._mw._is_visualization_enabled())
            return

        if state == "preview":
            allow = [self._mw.btn_preview] + list(self._mw._preview_controls)
            if self._mw._is_visualization_enabled():
                allow.append(self._mw.slider_zoom)
            self._set_interactive_widgets_enabled(False, allowlist=allow)
            self._mw.btn_preview.setEnabled(True)
            self._set_video_interaction_enabled(self._mw._is_visualization_enabled())
            return

        # Locked (non-tracking) state: disable all interactive widgets
        if state == "locked":
            self._set_interactive_widgets_enabled(False)
            self._set_video_interaction_enabled(False)
            return

    def _prepare_tracking_display(self):
        """Clear any stale frame before tracking starts."""
        self._mw.video_label.clear()
        if self._mw._is_visualization_enabled():
            self._mw.video_label.setText("")
            self._mw.video_label.setStyleSheet("color: #6a6a6a; font-size: 16px;")
        else:
            self._mw.video_label.setText(
                "Visualization Disabled\n\n"
                "Maximum speed processing mode active.\n"
                "Real-time stats displayed below."
            )
            self._mw.video_label.setStyleSheet("color: #9a9a9a; font-size: 14px;")

    def _show_video_logo_placeholder(self):
        """Show HYDRA logo in the video panel when no video is loaded."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainter, QPixmap
        from PySide6.QtSvg import QSvgRenderer

        try:
            from PySide6.QtCore import QByteArray

            from hydra_suite.paths import get_brand_icon_bytes

            logo_data = get_brand_icon_bytes("trackerkit.svg")
            vw = max(640, self._mw.scroll.viewport().width())
            vh = max(420, self._mw.scroll.viewport().height())
            canvas = QPixmap(vw, vh)
            canvas.fill(QColor(0, 0, 0, 0))

            renderer = (
                QSvgRenderer(QByteArray(logo_data)) if logo_data else QSvgRenderer()
            )
            if renderer.isValid():
                view_box = renderer.viewBoxF()
                if view_box.isEmpty():
                    default_size = renderer.defaultSize()
                    view_box = QRectF(
                        0,
                        0,
                        max(1, default_size.width()),
                        max(1, default_size.height()),
                    )

                # Preserve source aspect ratio and size it prominently.
                max_w = max(1, int(vw * 0.9))
                max_h = max(1, int(vh * 0.8))
                scale = min(max_w / view_box.width(), max_h / view_box.height())
                logo_w = max(1, int(view_box.width() * scale))
                logo_h = max(1, int(view_box.height() * scale))
                x = (vw - logo_w) // 2
                y = (vh - logo_h) // 2

                painter = QPainter(canvas)
                painter.setRenderHint(QPainter.Antialiasing, True)
                painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
                renderer.render(painter, QRectF(x, y, logo_w, logo_h))
                painter.end()
                self._mw.video_label.setPixmap(canvas)
                self._mw.video_label.setText("")
                return
        except Exception:
            pass
        self._mw.video_label.setPixmap(QPixmap())
        self._mw.video_label.setText("HYDRA\n\nLoad a video to begin...")

    # =========================================================================
    # PROGRESS VISIBILITY
    # =========================================================================

    def _is_worker_running(self, worker):
        """Safely check whether a worker thread-like object is running."""
        if worker is None:
            return False
        try:
            return bool(worker.isRunning())
        except Exception:
            return False

    def _has_active_progress_task(self) -> bool:
        """Return True if any async task that owns progress UI is still active."""
        return any(
            [
                self._is_worker_running(self._mw.tracking_worker),
                self._is_worker_running(getattr(self._mw, "merge_worker", None)),
                self._is_worker_running(self._mw.dataset_worker),
                self._is_worker_running(self._mw.interp_worker),
                self._is_worker_running(self._mw.oriented_video_worker),
            ]
        )

    def _refresh_progress_visibility(self):
        """Keep progress UI visible while any async tracking task is still running."""
        has_active_task = self._has_active_progress_task()
        self._mw.progress_bar.setVisible(has_active_task)
        self._mw.progress_label.setVisible(has_active_task)

    # =========================================================================
    # SESSION LOGGING
    # =========================================================================

    def _setup_session_logging(self, video_path, backward_mode=False):
        """Set up comprehensive logging for the entire tracking session."""
        from datetime import datetime
        from pathlib import Path

        from hydra_suite.utils.video_artifacts import (
            build_tracking_session_log_path,
            choose_writable_artifact_base_dir,
        )

        # Close existing session log if any
        self._cleanup_session_logging()

        # Only set up logging if not already set up
        if self._mw.session_log_handler is not None:
            logger.info("=" * 80)
            logger.info("Session log already active, continuing...")
            logger.info("=" * 80)
            return

        # Create a session log in the video's dedicated log directory.
        video_path = Path(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_dir = (
            os.path.dirname(self._panels.setup.csv_line.text())
            if self._panels.setup.csv_line.text()
            else ""
        )
        artifact_base_dir = choose_writable_artifact_base_dir(
            video_path,
            preferred_base_dirs=[csv_dir],
        )
        log_path = build_tracking_session_log_path(
            video_path,
            timestamp,
            artifact_base_dir=artifact_base_dir,
            create_dir=True,
        )

        # Create file handler for session
        self._mw.session_log_handler = logging.FileHandler(log_path, mode="w")
        self._mw.session_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self._mw.session_log_handler.setFormatter(formatter)

        # Add to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(self._mw.session_log_handler)

        logger.info("=" * 80)
        logger.info("TRACKING SESSION STARTED")
        logger.info(f"Session log: {log_path}")
        logger.info(f"Video: {video_path}")
        logger.info("=" * 80)

    def _cleanup_session_logging(self):
        """Remove session log handler from root logger."""
        if self._mw.session_log_handler:
            logger.info("=" * 80)
            logger.info("Tracking session completed")
            logger.info("=" * 80)

            root_logger = logging.getLogger()
            root_logger.removeHandler(self._mw.session_log_handler)
            self._mw.session_log_handler.close()
            self._mw.session_log_handler = None

    # =========================================================================
    # TEMPORARY FILES
    # =========================================================================

    def _cleanup_temporary_files(self):
        """Remove temporary files if cleanup is enabled."""
        if not self._mw._postprocess_panel.chk_cleanup_temp_files.isChecked():
            logger.info("Temporary file cleanup disabled, keeping intermediate files.")
            return

        if not self._mw.temporary_files:
            logger.info("No temporary files to clean up.")
            return

        cleaned = []
        failed = []
        for temp_file in self._mw.temporary_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    cleaned.append(os.path.basename(temp_file))
                    logger.info(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    failed.append(os.path.basename(temp_file))
                    logger.warning(f"Failed to remove temporary file {temp_file}: {e}")

        # Clear the list after cleanup attempt
        self._mw.temporary_files.clear()

        # Also clean up posekit directories if they exist
        params = self._mw.get_parameters_dict()
        output_dir = str(params.get("INDIVIDUAL_DATASET_OUTPUT_DIR", "")).strip()
        if output_dir and os.path.exists(output_dir):
            posekit_dir = os.path.join(output_dir, "posekit")
            if os.path.exists(posekit_dir) and os.path.isdir(posekit_dir):
                try:
                    import shutil

                    shutil.rmtree(posekit_dir)
                    logger.info(f"Removed posekit directory: {posekit_dir}")
                    cleaned.append("posekit/")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove posekit directory {posekit_dir}: {e}"
                    )
                    failed.append("posekit/")

        if cleaned:
            logger.info(
                f"Cleaned up {len(cleaned)} temporary file(s): {', '.join(cleaned)}"
            )
        if failed:
            logger.warning(
                f"Failed to clean {len(failed)} file(s): {', '.join(failed)}"
            )

    # =========================================================================
    # WIDGET SETUP HELPERS
    # =========================================================================

    def _disable_spinbox_wheel_events(self):
        """Disable wheel events on all spinboxes to prevent accidental value changes."""
        from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

        # Find all QSpinBox and QDoubleSpinBox widgets
        spinboxes = self._mw.findChildren(QSpinBox) + self._mw.findChildren(QDoubleSpinBox)
        for spinbox in spinboxes:
            spinbox.wheelEvent = lambda event: None

    def _connect_parameter_signals(self):
        from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QSpinBox

        widgets_to_connect = (
            self._mw.findChildren(QSpinBox)
            + self._mw.findChildren(QDoubleSpinBox)
            + self._mw.findChildren(QCheckBox)
        )
        for widget in widgets_to_connect:
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._mw._on_parameter_changed)
            elif hasattr(widget, "stateChanged"):
                widget.stateChanged.connect(self._mw._on_parameter_changed)

    # =========================================================================
    # UI SETTINGS PERSISTENCE
    # =========================================================================

    def _load_ui_settings(self) -> dict:
        """Load persistent HYDRA UI settings."""
        import json

        path = self._mw._get_ui_settings_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _queue_ui_state_save(self) -> None:
        """Debounce HYDRA UI settings writes while the user resizes or switches tabs."""
        if hasattr(self._mw, "_ui_state_save_timer"):
            self._mw._ui_state_save_timer.start()

    def _remember_collapsible_state(self, key: str, collapsible) -> None:
        """Restore and track expanded state for a collapsible section."""
        self._mw._collapsible_state_widgets[key] = collapsible
        saved = self._mw._ui_settings.get("collapsed_sections", {}).get(key)
        if isinstance(saved, bool):
            collapsible.setExpanded(saved)
        collapsible.toggled.connect(
            lambda _expanded, _key=key: self._queue_ui_state_save()
        )

    def _restore_ui_state(self) -> None:
        """Apply persisted HYDRA UI layout preferences after construction."""
        settings = self._mw._ui_settings or {}

        detection_index = settings.get("detection_method_index")
        if isinstance(detection_index, int) and hasattr(self._mw, "_detection_panel"):
            self._mw._detection_panel.combo_detection_method.setCurrentIndex(
                max(
                    0,
                    min(
                        detection_index,
                        self._mw._detection_panel.combo_detection_method.count() - 1,
                    ),
                )
            )

        tab_index = settings.get("active_tab_index")
        if isinstance(tab_index, int) and hasattr(self._mw, "tabs"):
            tab_index = max(0, min(tab_index, self._mw.tabs.count() - 1))
            if self._mw.tabs.isTabEnabled(tab_index):
                self._mw.tabs.setCurrentIndex(tab_index)

        splitter_sizes = settings.get("splitter_sizes")
        if (
            isinstance(splitter_sizes, list)
            and len(splitter_sizes) == 2
            and all(isinstance(size, int) and size > 0 for size in splitter_sizes)
            and hasattr(self._mw, "splitter")
        ):
            self._mw.splitter.setSizes(splitter_sizes)

    def _save_ui_settings(self) -> None:
        """Persist HYDRA UI layout preferences without touching tracking configs."""
        import json

        if not hasattr(self._mw, "tabs") or not hasattr(self._mw, "splitter"):
            return

        collapsed_sections = {
            key: widget.isExpanded()
            for key, widget in self._mw._collapsible_state_widgets.items()
        }
        settings = {
            "active_tab_index": int(self._mw.tabs.currentIndex()),
            "splitter_sizes": [int(size) for size in self._mw.splitter.sizes()],
            "detection_method_index": (
                int(self._mw._detection_panel.combo_detection_method.currentIndex())
                if hasattr(self._mw, "_detection_panel")
                else 0
            ),
            "collapsed_sections": collapsed_sections,
        }

        path = self._mw._get_ui_settings_path()
        try:
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
            self._mw._ui_settings = settings
        except Exception:
            logger.debug("Failed to save HYDRA UI settings", exc_info=True)

    # =========================================================================
    # CONTEXTUAL/SYNC UI HELPERS
    # =========================================================================

    def _sync_batch_list_ui(self):
        """Refresh the batch list widget with markers for the keystone."""
        from PySide6.QtWidgets import QListWidgetItem

        self._panels.setup.list_batch_videos.clear()
        current_fp = (
            os.path.normpath(self._panels.setup.file_line.text().strip())
            if self._panels.setup.file_line.text().strip()
            else ""
        )

        for i, fp in enumerate(self._mw.batch_videos):
            norm_fp = os.path.normpath(fp)
            if i == 0:
                item_text = f"⭐ KEYSTONE: {fp}"
            else:
                item_text = fp

            if norm_fp == current_fp:
                item_text = f"▶ CURRENT: {item_text}"

            item = QListWidgetItem(item_text)
            item.setToolTip(fp)

            if norm_fp == current_fp:
                font = item.font()
                font.setBold(True)
                item.setFont(font)

            self._panels.setup.list_batch_videos.addItem(item)

            if norm_fp == current_fp:
                self._panels.setup.list_batch_videos.setCurrentItem(item)

    def _sync_video_pose_overlay_controls(self, *_args):
        """Gate pose video overlay controls based on pose inference enable state."""
        has_controls = hasattr(self._mw, "check_video_show_pose") and hasattr(
            self._mw, "combo_video_pose_color_mode"
        )
        if not has_controls:
            return

        video_visible = bool(
            hasattr(self._mw, "_postprocess_panel")
            and self._mw._postprocess_panel.check_video_output.isChecked()
        )
        pose_enabled = self._is_pose_inference_enabled()
        enabled = bool(video_visible and pose_enabled)

        self._mw._postprocess_panel.check_video_show_pose.setEnabled(enabled)
        show_pose = bool(
            enabled and self._mw._postprocess_panel.check_video_show_pose.isChecked()
        )
        fixed_color_mode = (
            self._mw._postprocess_panel.combo_video_pose_color_mode.currentIndex() == 1
        )

        # Show detailed controls only when pose overlay is on.
        self._mw._postprocess_panel.lbl_video_pose_color_mode.setVisible(show_pose)
        self._mw._postprocess_panel.combo_video_pose_color_mode.setVisible(show_pose)
        self._mw._postprocess_panel.lbl_video_pose_point_radius.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_radius.setVisible(show_pose)
        self._mw._postprocess_panel.lbl_video_pose_point_thickness.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_thickness.setVisible(show_pose)
        self._mw._postprocess_panel.lbl_video_pose_line_thickness.setVisible(show_pose)
        self._mw._postprocess_panel.spin_video_pose_line_thickness.setVisible(show_pose)

        show_fixed_color = bool(show_pose and fixed_color_mode)
        self._mw._postprocess_panel.lbl_video_pose_color_label.setVisible(show_fixed_color)
        self._mw._postprocess_panel.btn_video_pose_color.setVisible(show_fixed_color)
        self._mw._postprocess_panel.lbl_video_pose_color.setVisible(show_fixed_color)

        self._mw._postprocess_panel.combo_video_pose_color_mode.setEnabled(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_radius.setEnabled(show_pose)
        self._mw._postprocess_panel.spin_video_pose_point_thickness.setEnabled(show_pose)
        self._mw._postprocess_panel.spin_video_pose_line_thickness.setEnabled(show_pose)
        self._mw._postprocess_panel.btn_video_pose_color.setEnabled(show_fixed_color)

        self._mw._postprocess_panel.lbl_video_pose_disabled_hint.setVisible(video_visible)
        if enabled:
            self._mw._postprocess_panel.lbl_video_pose_disabled_hint.setText(
                "Pose overlay will use keypoints from pose-augmented tracking output."
            )
        else:
            self._mw._postprocess_panel.lbl_video_pose_disabled_hint.setText(
                "Enable Pose Extraction in Analyze Individuals to edit pose overlay settings."
            )

    def _sync_pose_backend_ui(self):
        """Show/hide backend-specific pose controls."""
        if not hasattr(self._mw, "_identity_panel"):
            return
        backend = (
            self._mw._identity_panel.combo_pose_model_type.currentText().strip().lower()
        )
        self._mw._populate_pose_runtime_flavor_options(backend=backend)
        is_sleap = backend == "sleap"
        if hasattr(self._mw, "_identity_panel") and hasattr(
            self._mw._identity_panel, "pose_sleap_env_row_widget"
        ):
            self._mw._set_form_row_visible(
                self._mw._identity_panel.form_pose_runtime,
                self._mw._identity_panel.pose_sleap_env_row_widget,
                is_sleap,
            )
        if hasattr(self._mw, "_identity_panel") and hasattr(
            self._mw._identity_panel, "pose_sleap_experimental_row_widget"
        ):
            self._mw._set_form_row_visible(
                self._mw._identity_panel.form_pose_runtime,
                self._mw._identity_panel.pose_sleap_experimental_row_widget,
                is_sleap,
            )
        # Refresh pose model combo to show models for the selected backend.
        self._mw._refresh_pose_model_combo(
            preferred_model_path=self._mw._pose_model_path_for_backend(backend)
        )
        self._mw._on_runtime_context_changed()

    def _update_obb_mode_warning(self) -> None:
        """Show a performance hint when device/mode is a suboptimal combination."""
        if not hasattr(self._mw, "_detection_panel"):
            return
        runtime = (
            self._mw._selected_compute_runtime() if hasattr(self._mw, "_setup_panel") else ""
        )
        sequential = (
            hasattr(self._mw, "_detection_panel")
            and self._mw._detection_panel.combo_yolo_obb_mode.currentIndex() == 1
        )
        is_mps = "mps" in runtime.lower()
        is_cuda = "cuda" in runtime.lower()
        if is_mps and sequential:
            msg = (
                "⚠ Sequential mode is significantly slower on Apple Silicon (MPS). "
                "Direct mode is recommended for MPS — it runs ~4× faster."
            )
        elif is_cuda and not sequential:
            msg = (
                "⚠ Sequential mode is typically faster on CUDA GPUs. "
                "Consider switching to Sequential for better throughput."
            )
        else:
            msg = ""
        self._mw._detection_panel.lbl_obb_mode_warning.setText(msg)
        self._mw._detection_panel.lbl_obb_mode_warning.setVisible(bool(msg))

    def _update_range_info(self):
        """Update the frame range info label."""
        start = self._panels.setup.spin_start_frame.value()
        end = self._panels.setup.spin_end_frame.value()
        num_frames = end - start + 1

        fps = self._panels.setup.spin_fps.value()
        duration_sec = num_frames / fps if fps > 0 else 0

        self._panels.setup.lbl_range_info.setText(
            f"Tracking {num_frames} frames ({duration_sec:.2f} seconds)"
        )

    # =========================================================================
    # PIPELINE STATE QUERIES
    # =========================================================================

    def _is_pose_inference_enabled(self) -> bool:
        """Return whether pose inference is actively enabled for the run."""
        return bool(
            self._is_individual_pipeline_enabled()
            and hasattr(self._mw, "_identity_panel")
            and self._mw._identity_panel.chk_enable_pose_extractor.isChecked()
        )

    def _is_individual_pipeline_enabled(self) -> bool:
        """Return effective runtime state for individual analysis pipeline."""
        return self._mw._is_yolo_detection_mode()

    def _is_individual_image_save_enabled(self) -> bool:
        """Return effective runtime state for saving individual crops."""
        if not hasattr(self._mw, "_dataset_panel"):
            return False
        return bool(
            self._mw._dataset_panel.chk_enable_individual_dataset.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _should_generate_oriented_track_videos(self) -> bool:
        """Return True when final per-track oriented videos should be exported."""
        if not hasattr(self._mw, "_dataset_panel"):
            return False
        return bool(
            self._mw._dataset_panel.chk_generate_individual_track_videos.isChecked()
            and self._is_individual_pipeline_enabled()
        )

    def _should_run_interpolated_postpass(self) -> bool:
        """
        Return True when interpolated post-pass should run.

        We run this pass when interpolation is enabled and either:
        - individual crop saving is enabled, or
        - pose export is enabled (to fill occluded-frame pose rows in final CSV), or
        - oriented track video export is enabled (to cache interpolated ROI geometry).
        """
        if not hasattr(self._mw, "_identity_panel"):
            return False
        if not self._mw._identity_panel.chk_individual_interpolate.isChecked():
            return False
        if not self._is_individual_pipeline_enabled():
            return False
        return bool(
            self._is_individual_image_save_enabled()
            or self._mw._is_pose_export_enabled()
            or self._should_generate_oriented_track_videos()
        )
