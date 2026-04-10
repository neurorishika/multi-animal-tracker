from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QPixmap = QtGui.QPixmap
QApplication = QtWidgets.QApplication

classkit_config_path = pytest.importorskip(
    "hydra_suite.classkit.gui.project"
).classkit_config_path
main_window_module = pytest.importorskip("hydra_suite.classkit.gui.main_window")
MainWindow = main_window_module.MainWindow


@pytest.fixture()
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture(autouse=True)
def cleanup_qt_widgets(qapp):
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
    gc.collect()


def test_reset_analysis_view_restores_explore_embedding_mode(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window._show_model_umap = True
    window._show_model_pca = True
    window.btn_umap_model.setEnabled(True)
    window.btn_pca_model.setEnabled(True)
    window.btn_umap_embedding.setChecked(False)
    window.btn_umap_model.setChecked(True)
    window.btn_pca_model.setChecked(True)

    window._reset_analysis_view()

    assert window.explorer_mode == "explore"
    assert window._show_model_umap is False
    assert window._show_model_pca is False
    assert window.btn_umap_embedding.isChecked() is True
    assert window.btn_umap_model.isChecked() is False
    assert window.btn_pca_model.isChecked() is False


def test_outline_control_only_visible_in_labeling_mode(qapp) -> None:
    window = MainWindow()

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True

    window.set_explorer_mode("labeling")
    assert window.outline_threshold_label.isHidden() is False
    assert window.outline_threshold_spin.isHidden() is False

    window.set_explorer_mode("explore")
    assert window.outline_threshold_label.isHidden() is True
    assert window.outline_threshold_spin.isHidden() is True


def test_review_buttons_live_in_left_review_panel(qapp) -> None:
    window = MainWindow()

    def is_descendant(widget, ancestor) -> bool:
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = current.parentWidget()
        return False

    assert is_descendant(window.review_selected_btn, window.context_panel) is True
    assert is_descendant(window.review_reject_btn, window.context_panel) is True
    assert (
        is_descendant(window.review_clear_unverified_btn, window.context_panel) is True
    )

    assert is_descendant(window.review_selected_btn, window.preview_panel) is False
    assert is_descendant(window.review_reject_btn, window.preview_panel) is False
    assert (
        is_descendant(window.review_clear_unverified_btn, window.preview_panel) is False
    )


def test_empty_review_mode_reverts_combo_selection(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append(args[1:3]) or QtWidgets.QMessageBox.Ok,
    )

    window = MainWindow()
    review_idx = window.view_mode_combo.findData("review")

    window.view_mode_combo.setCurrentIndex(review_idx)

    assert window.explorer_mode == "explore"
    assert window.view_mode_combo.currentData() == "explore"
    assert shown == [
        ("Review Queue Empty", "There are no unverified machine labels to review yet.")
    ]


def test_assign_label_can_continue_in_infinite_labeling_mode(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
        Path("/tmp/d.png"),
    ]
    window.image_labels = [None, None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.5, 10.2],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1])
    window.set_explorer_mode("labeling")
    window.selected_point_index = 0
    window.candidate_indices = []

    monkeypatch.setattr(window, "_prompt_after_label_set_complete", lambda: "infinite")

    window.assign_label_to_selected("alpha")

    assert window.image_labels[0] == "alpha"
    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {2, 3}
    assert window.candidate_indices == [window.selected_point_index]
    assert window.round_labeled_indices == []
    assert int(window.cluster_assignments[window.selected_point_index]) == 1


def test_manual_click_in_labeling_mode_browses_database_with_prev_next(qapp) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(6)]
    window.image_labels = [None] * 6
    window.candidate_indices = [1, 4]
    window.set_explorer_mode("labeling")

    window.on_explorer_point_clicked(2)
    window.on_next_image()

    assert window.selected_point_index == 3

    window._command_busy = False
    window._command_block_until = 0.0
    window.on_prev_image()

    assert window.selected_point_index == 2


def test_clicking_different_point_replaces_labeling_selection(qapp) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(5)]
    window.image_labels = [None] * 5
    window.candidate_indices = [1, 3]
    window.set_explorer_mode("labeling")

    window.on_explorer_point_clicked(1)
    window.on_explorer_point_clicked(4)

    assert window.selected_point_index == 4
    assert window._labeling_navigation_scope == "database"


def test_infinite_labeling_reuses_cached_distance_scores(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
        Path("/tmp/d.png"),
    ]
    window.image_labels = ["seed", None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [8.0, 8.0],
            [8.5, 8.2],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1])

    assert window._start_infinite_labeling_mode() is True
    first_choice = window.selected_point_index
    assert first_choice in {2, 3}

    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full distance recomputation should not run again")
        ),
    )

    window.assign_label_to_selected("beta")

    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {1, 2, 3}
    assert window.selected_point_index != first_choice


def test_infinite_labeling_restores_persisted_cache_before_recomputing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_infinite_label_cache(self, embedding_cache_id=None):
            assert embedding_cache_id == 41
            return {
                "distance_cache": np.array([0.0, 0.25, 81.0], dtype=np.float32),
                "cluster_counts": np.array([1, 0], dtype=np.int32),
                "owner_cache": np.array([0, 0, 0], dtype=np.int32),
                "embedding_cache_id": 41,
                "cluster_cache_id": 9,
                "label_signature": window._labeled_index_signature([0]),
            }

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png"), Path("/tmp/c.png")]
    window.image_labels = ["seed", None, None]
    window.embeddings = np.array([[0.0, 0.0], [0.5, 0.0], [9.0, 9.0]], dtype=np.float32)
    window.cluster_assignments = np.array([0, 0, 1], dtype=np.int32)
    window._current_embedding_cache_id = 41
    window._current_cluster_cache_id = 9

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("distance cache should restore from project storage")
        ),
    )

    restored = window._ensure_infinite_labeling_cache()

    assert restored is not None
    assert np.array_equal(
        restored,
        np.array([0.0, 0.25, 81.0], dtype=np.float32),
    )
    assert np.array_equal(
        window._infinite_label_cluster_counts,
        np.array([1, 0], dtype=np.int32),
    )
    assert np.array_equal(
        window._infinite_label_owner_cache,
        np.array([0, 0, 0], dtype=np.int32),
    )


def test_deleting_label_keeps_infinite_cache_hot_for_next_label(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_single_shot = main_window_module.QTimer.singleShot

    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(
            lambda msec, fn: fn() if int(msec) == 0 else original_single_shot(msec, fn)
        ),
    )

    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(5)]
    window.image_labels = ["alpha", "beta", None, None, None]
    window.embeddings = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.0],
            [7.0, 7.0],
            [7.4, 7.1],
            [0.8, 0.1],
        ],
        dtype=np.float32,
    )
    window.cluster_assignments = np.array([0, 0, 1, 1, 0], dtype=np.int32)
    window.set_explorer_mode("labeling")
    window._labeling_flow_mode = "infinite"
    window.selected_point_index = 2
    window.candidate_indices = [2]

    assert window._ensure_infinite_labeling_cache() is not None
    assert window._infinite_label_owner_cache is not None

    window._set_label_for_index(1, None)

    assert window._infinite_label_distance_cache is not None
    assert window._infinite_label_owner_cache is not None

    monkeypatch.setattr(
        window,
        "_min_distance_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("full cache rebuild should not run after delete")
        ),
    )
    monkeypatch.setattr(
        window,
        "_min_distance_and_owner_to_reference_scores",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("owner-aware full cache rebuild should not run after delete")
        ),
    )

    window.assign_label_to_selected("gamma")

    assert window.image_labels[2] == "gamma"
    assert window._labeling_flow_mode == "infinite"
    assert window.selected_point_index in {3, 4}


def test_after_training_inference_skips_invalid_auto_model_umap(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    critical_calls = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda _msec, fn: fn()),
    )

    class Dialog:
        def __init__(self) -> None:
            self.logs = []

        def append_log(self, message: str) -> None:
            self.logs.append(message)

    window = MainWindow()
    window._model_probs = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    dialog = Dialog()

    window._after_training_inference(dialog)

    assert critical_calls == []
    assert dialog.logs == [
        "Inference complete — Metrics tab updated.",
        "Auto-computing model-space UMAP...",
        "Model-space UMAP skipped: Need at least 2 prediction columns to compute model-space UMAP.",
    ]


def test_after_training_inference_logs_model_umap_worker_error_without_modal(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    critical_calls = []

    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: critical_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        main_window_module.QTimer,
        "singleShot",
        staticmethod(lambda _msec, fn: fn()),
    )

    task_workers = pytest.importorskip("hydra_suite.classkit.jobs.task_workers")

    class DummySignal:
        def __init__(self) -> None:
            self._callbacks = []

        def connect(self, callback) -> None:
            self._callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in list(self._callbacks):
                callback(*args)

    class DummySignals:
        def __init__(self) -> None:
            self.success = DummySignal()
            self.error = DummySignal()
            self.progress = DummySignal()
            self.finished = DummySignal()

    class FailingWorker:
        def __init__(self, *args, **kwargs) -> None:
            self.signals = DummySignals()

    class Dialog:
        def __init__(self) -> None:
            self.logs = []

        def append_log(self, message: str) -> None:
            self.logs.append(message)

    monkeypatch.setattr(task_workers, "LogitsUMAPWorker", FailingWorker)

    window = MainWindow()
    window._model_probs = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]],
        dtype=np.float32,
    )
    window._threadpool_start = lambda worker: (
        worker.signals.error.emit("boom"),
        worker.signals.finished.emit(),
    )
    dialog = Dialog()

    window._after_training_inference(dialog)

    assert critical_calls == []
    assert dialog.logs[-1] == "Model-space UMAP failed: boom"


def test_empty_hover_clears_preview_without_active_label_selection(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = None
    window.hover_locked = False
    window.last_preview_index = 5
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    window.on_explorer_empty_hover()

    assert window.last_preview_index is None
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "Hover a point to preview the source image" in window.preview_info.text()


def test_empty_hover_clears_preview_in_explore_mode_and_cancels_pending_hover(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window.set_explorer_mode("explore")
    window.selected_point_index = None
    window.hover_locked = False

    image_path = tmp_path / "preview.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_path)) is True

    window.image_paths = [image_path]
    window.request_preview_for_index(0, source="hover")

    window.on_explorer_empty_hover()
    window._flush_preview_update()

    assert window.last_preview_index is None
    assert window._pending_preview_index is None
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "Hover a point to preview the source image" in window.preview_info.text()


def test_empty_hover_keeps_selected_preview_with_locked_selection(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 3
    window.hover_locked = True
    window.last_preview_index = 3
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    window.on_explorer_empty_hover()

    assert window._pending_preview_index == 3
    assert window._pending_preview_source == "selection"
    assert window.last_preview_index == 3
    assert window.preview_canvas.pix_item.pixmap().isNull() is False


def test_hovering_other_points_does_not_replace_locked_labeling_preview(qapp) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 2
    window.hover_locked = True
    window.candidate_indices = [1, 2, 3]

    requested = []
    window.request_preview_for_index = lambda index, source="hover": requested.append(
        (index, source)
    )

    window.on_explorer_point_hovered(1)

    assert requested == []


def test_empty_background_double_click_clears_labeling_pool_but_stays_in_labeling(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(4)]
    window.image_labels = [None] * 4
    window.set_explorer_mode("labeling")
    window.selected_point_index = 2
    window.hover_locked = True
    window.candidate_indices = [1, 2]
    window.round_labeled_indices = [1]
    window._labeling_flow_mode = "infinite"
    window._labeling_navigation_scope = "database"

    window.on_explorer_background_double_click()

    assert window.explorer_mode == "labeling"
    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.candidate_indices == []
    assert window.round_labeled_indices == []
    assert window._labeling_flow_mode == "batch"
    assert window._labeling_navigation_scope == "pool"


def test_empty_background_double_click_keeps_labeling_mode_without_active_set(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path(f"/tmp/{idx}.png") for idx in range(3)]
    window.image_labels = [None] * 3
    window.set_explorer_mode("labeling")
    window.selected_point_index = 1
    window.hover_locked = True

    window.on_explorer_background_double_click()

    assert window.explorer_mode == "labeling"
    assert window.selected_point_index is None
    assert window.hover_locked is False


def test_leaving_labeling_mode_clears_selection_and_restores_hover_preview(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window.set_explorer_mode("labeling")
    window.selected_point_index = 0
    window.hover_locked = True
    window.last_preview_index = 0
    window.preview_canvas.set_pixmap(QPixmap(10, 10))

    image_a = tmp_path / "labeling_a.png"
    image_b = tmp_path / "labeling_b.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_a)) is True
    assert pixmap.save(str(image_b)) is True
    window.image_paths = [image_a, image_b]

    window.set_explorer_mode("explore")

    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.preview_canvas.pix_item.pixmap().isNull() is True
    assert "selection lock is disabled in this mode" in window.selection_info.text()

    window.on_explorer_point_hovered(1)
    window._flush_preview_update()

    assert window.last_preview_index == 1


def test_leaving_review_mode_clears_selection_and_restores_hover_preview(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()
    window._review_candidate_indices = [0]
    window.selected_point_index = 0
    window.hover_locked = True
    window.last_preview_index = 0
    window.preview_canvas.set_pixmap(QPixmap(10, 10))
    window.set_explorer_mode("review")

    image_a = tmp_path / "review_a.png"
    image_b = tmp_path / "review_b.png"
    pixmap = QPixmap(10, 10)
    assert pixmap.save(str(image_a)) is True
    assert pixmap.save(str(image_b)) is True
    window.image_paths = [image_a, image_b]

    window.set_explorer_mode("explore")

    assert window.selected_point_index is None
    assert window.hover_locked is False
    assert window.preview_canvas.pix_item.pixmap().isNull() is True

    window.on_explorer_point_hovered(1)
    window._flush_preview_update()

    assert window.last_preview_index == 1


def test_training_settings_persist_in_project_config(qapp, tmp_path: Path) -> None:
    window = MainWindow()
    window.project_path = tmp_path
    window._last_training_settings = {
        "mode": "flat_custom",
        "custom_input_size": 192,
        "tiny_width": 160,
        "tiny_height": 96,
        "device": "cpu",
    }

    window._save_last_training_settings()

    config = json.loads(classkit_config_path(tmp_path).read_text())
    assert config["last_training_settings"]["custom_input_size"] == 192

    restored = MainWindow()
    restored._apply_project_config(config)

    assert restored._last_training_settings["mode"] == "flat_custom"
    assert restored._last_training_settings["tiny_width"] == 160


def test_default_training_settings_use_average_image_dimensions(
    qapp, tmp_path: Path
) -> None:
    from PIL import Image

    image_a = tmp_path / "img_a.png"
    image_b = tmp_path / "img_b.png"
    Image.new("RGB", (104, 60), color=(32, 64, 96)).save(image_a)
    Image.new("RGB", (136, 68), color=(96, 64, 32)).save(image_b)

    window = MainWindow()
    window.image_paths = [image_a, image_b]

    defaults = window._default_training_settings_from_project()

    assert defaults["tiny_width"] == 128
    assert defaults["tiny_height"] == 64
    assert defaults["custom_input_size"] == 96


def test_make_training_spec_uses_selected_initial_model_path(
    qapp, tmp_path: Path
) -> None:
    window = MainWindow()

    yolo_start = tmp_path / "previous_yolo.pt"
    yolo_start.write_text("weights", encoding="utf-8")
    yolo_spec = window._make_training_spec(
        {
            "base_model": "yolo26n-cls.pt",
            "initial_model_path": str(yolo_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_yolo"),
        "flat_yolo",
        True,
        tmp_path / "export_yolo",
    )

    assert yolo_spec.base_model == str(yolo_start)
    assert yolo_spec.resume_from == ""

    custom_start = tmp_path / "previous_custom.pth"
    custom_start.write_text("weights", encoding="utf-8")
    custom_spec = window._make_training_spec(
        {
            "custom_backbone": "resnet18",
            "initial_model_path": str(custom_start),
            "epochs": 5,
            "batch": 8,
            "lr": 0.001,
            "patience": 2,
        },
        window._training_role_for_mode("flat_custom"),
        "flat_custom",
        False,
        tmp_path / "export_custom",
    )

    assert custom_spec.base_model == ""
    assert custom_spec.resume_from == str(custom_start)


def test_autoload_cached_analysis_restores_state_without_prompt(qapp) -> None:
    class FakeDB:
        def get_most_recent_embeddings(self):
            return np.array([[1.0, 2.0], [3.0, 4.0]]), {"id": 17}

        def get_most_recent_cluster_cache(self, embedding_cache_id=None):
            assert embedding_cache_id == 17
            return {"assignments": np.array([0, 1]), "n_clusters": 2}

        def get_most_recent_umap_cache(self, embedding_cache_id=None, kind="embedding"):
            assert embedding_cache_id == 17
            assert kind == "embedding"
            return {
                "coords": np.array([[0.1, 0.2], [0.3, 0.4]]),
                "n_neighbors": 21,
                "min_dist": 0.05,
            }

        def get_most_recent_candidate_cache(self):
            return {"candidate_indices": [0, 1]}

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["kept-label", None]
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    window._autoload_cached_analysis(FakeDB())

    assert window._current_embedding_cache_id == 17
    assert np.array_equal(window.embeddings, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.array_equal(window.cluster_assignments, np.array([0, 1]))
    assert np.array_equal(window.umap_coords, np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert window.last_umap_params == {"n_neighbors": 21, "min_dist": 0.05}
    assert window.candidate_indices == [0, 1]
    assert window.round_labeled_indices == [0]
    assert window.explorer_mode == "labeling"


def test_restore_cached_candidate_batch_marks_existing_labels(qapp) -> None:
    window = MainWindow()
    window.image_paths = [
        Path("/tmp/a.png"),
        Path("/tmp/b.png"),
        Path("/tmp/c.png"),
    ]
    window.image_labels = ["alpha", None, "beta"]

    window._restore_cached_candidate_batch([0, 1, 2, 1, -1, 99, "2"])

    assert window.candidate_indices == [0, 1, 2]
    assert window.round_labeled_indices == [0, 2]


def test_autoload_model_from_db_prefers_cached_predictions_without_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")
    cached_predictions = {
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "class_names": ["a", "b"],
        "active_model_mode": "tiny",
    }

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_prediction_cache(self):
            return cached_predictions

        def get_most_recent_model_cache(self):
            return None

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    applied = {}

    def fake_apply(payload, db):
        applied["payload"] = payload
        applied["db"] = db

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(window, "_apply_cached_predictions", fake_apply)

    window._autoload_model_from_db()

    assert applied["payload"] is cached_predictions
    assert isinstance(applied["db"], FakeDB)


def test_autoload_model_from_db_prefers_newer_model_over_stale_predictions(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")
    cached_predictions = {
        "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
        "class_names": ["a", "b"],
        "active_model_mode": "tiny",
        "timestamp": "2026-04-08 09:00:00",
    }
    recent_model = {
        "id": 42,
        "mode": "tiny",
        "artifact_paths": [str(tmp_path / "latest_model.pt")],
        "timestamp": "2026-04-08 10:00:00",
    }

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def get_most_recent_prediction_cache(self):
            return cached_predictions

        def get_most_recent_model_cache(self):
            return recent_model

    window = MainWindow()
    window.db_path = tmp_path / "classkit.db"

    scheduled = {}
    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QTimer.singleShot",
        lambda _ms, callback: callback(),
    )
    monkeypatch.setattr(
        window,
        "_apply_cached_predictions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale cached predictions should not be used")
        ),
    )

    def fake_load(entry, on_success=None):
        scheduled["entry"] = entry
        scheduled["has_callback"] = callable(on_success)

    monkeypatch.setattr(window, "_load_model_from_cache_entry", fake_load)

    window._autoload_model_from_db()

    assert scheduled["entry"] is recent_model
    assert scheduled["has_callback"] is True


def test_autoload_yolo_classifier_runs_inference_without_prompt(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    yolo_path = models_dir / "yolo_classifier_latest.pt"
    yolo_path.write_text("weights", encoding="utf-8")

    class FakeDB:
        def get_most_recent_prediction_cache(self):
            return None

    window = MainWindow()
    window.project_path = tmp_path
    window._ask_yes_no = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("autoload should not prompt")
    )

    launched = {}

    monkeypatch.setattr(
        "hydra_suite.classkit.gui.main_window.QTimer.singleShot",
        lambda _ms, callback: callback(),
    )
    monkeypatch.setattr(
        window,
        "_run_yolo_inference",
        lambda path, on_success=None: launched.setdefault("path", path),
    )

    scheduled = window._autoload_yolo_classifier(FakeDB())

    assert scheduled is True
    assert window._yolo_model_path == yolo_path
    assert window._active_model_mode == "yolo"
    assert launched["path"] == yolo_path


def test_apply_cached_predictions_switches_to_prediction_view_when_no_batch(
    qapp,
) -> None:
    class FakeDB:
        def get_most_recent_umap_cache(self, kind="model"):
            return None

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    window._apply_cached_predictions(
        {
            "probs": np.array([[0.8, 0.2], [0.3, 0.7]]),
            "class_names": ["a", "b"],
            "active_model_mode": "tiny",
        },
        FakeDB(),
    )

    assert window.explorer_mode == "predictions"


def test_apply_cached_predictions_restores_metrics_without_switching_tabs(qapp) -> None:
    class FakeDB:
        def get_most_recent_umap_cache(self, kind="model"):
            return None

        def get_most_recent_model_cache(self):
            return {"best_val_acc": 0.875}

    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["class_1", "class_2"]
    window.classes = ["class_1", "class_2"]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    current_tab = window.tabs.currentWidget()

    window._apply_cached_predictions(
        {
            "probs": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "class_names": ["class_1", "class_2"],
            "active_model_mode": "tiny",
        },
        FakeDB(),
    )

    assert "Classification Metrics" in window.metrics_view.toPlainText()
    assert "0.875" in window.metrics_validation_label.text()
    assert window.tabs.currentWidget() is current_tab


def test_metrics_display_includes_heldout_validation_summary(qapp) -> None:
    metrics_module = pytest.importorskip("hydra_suite.classkit.core.train.metrics")

    window = MainWindow()
    metrics = metrics_module.compute_metrics(
        np.array([0, 1]),
        np.array([0, 1]),
        class_names=["class_1", "class_2"],
    )
    window._set_heldout_validation_summary(
        window._validation_summary_from_results([{"best_val_acc": 0.875}])
    )

    window._update_metrics_display(metrics, activate_metrics_tab=False)

    assert "0.875" in window.metrics_validation_label.text()
    assert "0.875" in window.metrics_view.toPlainText()


def test_load_project_data_clears_stale_model_state_before_restore(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_module = pytest.importorskip("hydra_suite.classkit.core.store.db")

    class FakeDB:
        def __init__(self, _path):
            self.path = _path

        def migrate_paths_to_resolved(self):
            return 0

        def get_all_image_paths(self):
            return [str(tmp_path / "a.png"), str(tmp_path / "b.png")]

        def get_all_labels(self):
            return ["class_1", "class_2"]

        def get_most_recent_embeddings(self):
            return None

        def get_most_recent_cluster_cache(self, embedding_cache_id=None):
            return None

        def get_most_recent_umap_cache(self, embedding_cache_id=None, kind="embedding"):
            return None

        def get_most_recent_candidate_cache(self):
            return None

        def get_most_recent_prediction_cache(self):
            return {
                "probs": np.array([[0.9, 0.1], [0.2, 0.8]]),
                "class_names": ["class_1", "class_2"],
                "active_model_mode": "tiny",
            }

        def get_most_recent_model_cache(self):
            return None

    window = MainWindow()
    window.project_path = tmp_path
    window.db_path = tmp_path / "classkit.db"
    window._model_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    window.metrics_view.setPlainText("stale metrics")

    monkeypatch.setattr(db_module, "ClassKitDB", FakeDB)
    monkeypatch.setattr(window, "_load_project_config", lambda: {})

    window.load_project_data()

    assert "Classification Metrics" in window.metrics_view.toPlainText()
    assert np.allclose(window._model_probs, np.array([[0.9, 0.1], [0.2, 0.8]]))


def test_activate_saved_labels_view_switches_to_labeling_when_labels_exist(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["alpha", None]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])
    window.explorer_mode = "explore"

    window._activate_saved_labels_view_if_available()

    assert window.explorer_mode == "labeling"


def test_activate_saved_labels_view_does_not_override_predictions_or_batches(
    qapp,
) -> None:
    window = MainWindow()
    window.image_paths = [Path("/tmp/a.png"), Path("/tmp/b.png")]
    window.image_labels = ["alpha", None]
    window.umap_coords = np.array([[0.0, 0.0], [1.0, 1.0]])

    window._model_probs = np.array([[0.8, 0.2], [0.3, 0.7]])
    window.explorer_mode = "predictions"
    window._activate_saved_labels_view_if_available()
    assert window.explorer_mode == "predictions"

    window._model_probs = None
    window.candidate_indices = [1]
    window.explorer_mode = "explore"
    window._activate_saved_labels_view_if_available()
    assert window.explorer_mode == "explore"


def test_open_recent_project_flushes_pending_label_updates_before_switch(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "classkit.db").write_text("db", encoding="utf-8")

    window = MainWindow()
    calls = []

    monkeypatch.setattr(
        window,
        "_flush_pending_label_updates",
        lambda force=False: calls.append(("flush", force)),
    )
    monkeypatch.setattr(
        window,
        "load_project_data",
        lambda: calls.append(("load", str(window.project_path))),
    )
    monkeypatch.setattr(
        window,
        "update_context_panel",
        lambda: calls.append(("context", None)),
    )

    window._open_recent_project(str(project_dir))

    assert calls[0] == ("flush", True)
    assert calls[1] == ("load", str(project_dir))
