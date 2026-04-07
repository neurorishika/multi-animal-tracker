# TrackerKit Monolith Phase 2 — Design Spec

**Date:** 2026-04-05
**Author:** Rishika Mohanta
**Status:** Approved
**Predecessor:** `2026-04-04-panel-extraction-sprint2-design.md` (Phase 1 of Slice 4)

---

## Goal

Complete the decomposition of `trackerkit/gui/main_window.py` into a thin coordinator (≤600 lines). Phase 1 (Sprint 2) extracted UI construction into 6 panel classes, reducing `main_window.py` from 19,975 → 16,221 lines. Phase 2 extracts:

1. **5 embedded worker classes** → `trackerkit/gui/workers/`
2. **4 utility widget classes** → `trackerkit/gui/widgets/`
3. **4 dialogs** → migrate to `BaseDialog`; 1 embedded worker → `BaseWorker`
4. **~200 panel-scoped handler methods** → into their respective panels
5. **~230 cross-panel orchestration methods** → 3 new orchestrator classes

Target: `main_window.py` ≤ 600 lines, all 841+ tests passing.

---

## Starting State (2026-04-05)

- `main_window.py`: 16,221 lines, 11 classes, ~430 methods on `MainWindow`
- 6 panels populated with UI construction only (no handlers yet)
- 5 existing trackerkit dialogs; 1 already uses `BaseDialog` (`CNNIdentityImportDialog`)
- `BaseWorker` and `BaseDialog` exist in `hydra_suite/widgets/`
- All 5 embedded workers already inherit `BaseWorker`

---

## Target File Layout

```
trackerkit/gui/
    __init__.py
    main_window.py              # thin coordinator, ≤600 lines
    validate_labels.py          # unchanged
    panels/                     # populated in Sprint 2 — now gains handlers
        __init__.py
        setup_panel.py          # + setup/video/ROI handlers (~35 methods)
        detection_panel.py      # + detection/preview handlers (~45 methods)
        tracking_panel.py       # + tracking start/stop handlers (~20 methods)
        postprocess_panel.py    # + post-process/export handlers (~25 methods)
        dataset_panel.py        # + dataset/xanylabeling handlers (~20 methods)
        identity_panel.py       # + identity/CNN/pose handlers (~55 methods)
    orchestrators/              # NEW
        __init__.py
        tracking.py             # TrackingOrchestrator: run→merge→export→finalize
        config.py               # ConfigOrchestrator: load/save/presets/ROI/video setup
        session.py              # SessionOrchestrator: logging/progress/UI state machine
    workers/                    # NEW — moved from main_window.py
        __init__.py
        merge_worker.py         # MergeWorker + _write_csv_artifact + _write_roi_npz
        crops_worker.py         # InterpolatedCropsWorker
        video_worker.py         # OrientedTrackVideoWorker
        dataset_worker.py       # DatasetGenerationWorker
        preview_worker.py       # PreviewDetectionWorker + preview background helpers
    widgets/                    # NEW — moved from main_window.py
        __init__.py
        collapsible.py          # CollapsibleGroupBox + AccordionContainer
        help_label.py           # CompactHelpLabel
        tooltip_button.py       # ImmediateTooltipButton
    dialogs/                    # exists — now fully on BaseDialog
        __init__.py
        bg_parameter_helper.py  # QDialog → BaseDialog
        parameter_helper.py     # QDialog → BaseDialog
        model_test_dialog.py    # QDialog → BaseDialog; _TestWorker → BaseWorker
        run_history_dialog.py   # QDialog → BaseDialog
        train_yolo_dialog.py    # QDialog → BaseDialog (RoleTrainingWorker already BaseWorker)
        cnn_identity_import_dialog.py  # already done ✓
```

`main_window.py` retains after the sprint:
- `__init__` — instantiates panels, orchestrators, connects signals
- `init_ui` — assembles tab layout, welcome page
- `_make_welcome_page` / `_show_workspace`
- `CurrentPageStackedWidget` (or moves to `widgets/`)
- `_connect_signals` — pure signal wiring (~30 lines)
- UI utility wrappers that are genuinely cross-cutting (~10 methods max)

---

## Architecture

### Orchestrator Pattern

Each orchestrator is a plain Python class (not a `QObject`). `MainWindow` constructs all three in `__init__` and passes each a reference to `self` (for message boxes and status bar), the `TrackerConfig`, and the panel bundle.

```python
class TrackingOrchestrator:
    def __init__(self, main_window, config: TrackerConfig, panels) -> None:
        self._mw = main_window
        self._config = config
        self._panels = panels
```

Orchestrators do not hold their own Qt signals. They call panel methods directly for UI feedback (e.g., `self._panels.tracking.set_progress(50)`) and emit via the main window's status bar for global feedback.

### Panel Signal Convention

Panels define Qt signals for user-initiated actions. `MainWindow._connect_signals()` wires these to orchestrators:

```python
# In MainWindow._connect_signals():
self._detection_panel.preview_requested.connect(self._tracking_orch.start_preview_on_video)
self._setup_panel.video_selected.connect(self._config_orch.setup_video_file)
self._tracking_panel.start_requested.connect(self._tracking_orch.start_tracking)
```

Orchestrators call panel update methods for feedback:

```python
# In TrackingOrchestrator:
self._panels.tracking.set_progress(value)
self._panels.tracking.display_frame(rgb)
```

### Worker Module-Level Helpers

Module-level functions in `main_window.py` that serve a single worker move into that worker's file:

| Functions | Target file |
|---|---|
| `_clear_preview_background_cache`, `_hash_preview_roi_mask`, `_preview_background_cache_key`, `_build_preview_background_params`, `_get_cached_preview_background_state`, `_store_preview_background_state`, `_build_preview_background_model`, `_normalize_preview_model_names`, `_preview_class_label`, `_preview_label_anchor`, `_draw_preview_label_stack`, `_draw_preview_pose_points`, `_run_preview_detection_job` | `workers/preview_worker.py` |
| `_write_csv_artifact`, `_write_roi_npz` | `workers/merge_worker.py` |

Model registry functions (`get_video_config_path`, `get_models_directory`, `load_yolo_model_registry`, `save_yolo_model_registry`, `register_yolo_model`, etc.) move to `config.py` orchestrator or a new `trackerkit/registry.py` module if they are referenced outside `main_window.py`.

---

## Per-Component Extraction Scope

### Step 1 — Workers (mechanical)

Move 5 worker classes and their companion module-level functions verbatim into `workers/`. Update imports in `main_window.py`. No logic change.

| Worker | Source lines (approx.) | Target |
|---|---|---|
| `MergeWorker` + helpers | 306–543 | `workers/merge_worker.py` |
| `InterpolatedCropsWorker` | 544–1684 | `workers/crops_worker.py` |
| `OrientedTrackVideoWorker` | 1685–1747 | `workers/video_worker.py` |
| `DatasetGenerationWorker` | 1748–1912 | `workers/dataset_worker.py` |
| `PreviewDetectionWorker` + helpers | 140–305, 1913–2829 | `workers/preview_worker.py` |

Expected `main_window.py` after Step 1: ≤ 14,500 lines.

### Step 2 — Widget Classes (mechanical)

Move 4 classes verbatim. Update imports in `panels/*.py` (which currently import from `main_window`).

| Class | Target |
|---|---|
| `CollapsibleGroupBox` + `AccordionContainer` | `widgets/collapsible.py` |
| `CompactHelpLabel` | `widgets/help_label.py` |
| `ImmediateTooltipButton` | `widgets/tooltip_button.py` |

All panel files currently import these from `main_window`. Update to `from hydra_suite.trackerkit.gui.widgets.collapsible import CollapsibleGroupBox, AccordionContainer`, etc.

Expected `main_window.py` after Step 2: ≤ 13,900 lines.

### Step 3 — Dialog BaseDialog Migration

Migrate 4 dialogs to `BaseDialog`. Each dialog:
1. Changes `class FooDialog(QDialog)` → `class FooDialog(BaseDialog)`
2. Removes manual `setWindowTitle`, `setModal`, `QVBoxLayout` root setup, and button box boilerplate
3. Calls `self.add_content(widget)` instead of `self._root_layout.addWidget(widget)`
4. Migrates `_TestWorker(QThread)` in `model_test_dialog.py` to `BaseWorker`

No line count target for `main_window.py` (dialogs are already in separate files).

### Step 4 — Panel Handler Migration

Handlers move in risk-first order (smallest panel first). After each panel:
- Run import smoke test
- Run full test suite
- Resolve any failures before starting next panel

| Step | Panel | Methods moving in | Representative handlers |
|---|---|---|---|
| 4a | `DatasetPanel` | ~20 | `_on_dataset_generation_toggled`, `_on_xanylabeling_env_changed`, `_open_in_xanylabeling`, `_open_pose_label_ui` |
| 4b | `SetupPanel` | ~35 | `_open_recent_video`, `_add_videos_to_batch`, `_clear_batch`, `_detect_fps_from_current_video`, `_refresh_xanylabeling_envs` |
| 4c | `PostProcessPanel` | ~25 | `_on_cleaning_toggled`, `_on_video_output_toggled`, `_select_video_pose_color`, `select_video_output` |
| 4d | `TrackingPanel` | ~20 | `_on_confidence_density_map_toggled`, `_on_visualization_mode_changed`, `_on_parameter_changed`, `_draw_roi_overlay` |
| 4e | `IdentityPanel` | ~55 | `_on_identity_method_changed`, `_add_cnn_classifier_row`, `_remove_cnn_classifier_row`, `_refresh_cnn_identity_model_combo`, `_on_pose_analysis_toggled`, `_select_color_tag_model` |
| 4f | `DetectionPanel` | ~45 | `_on_brightness_changed`, `_on_yolo_mode_changed`, `_test_detection_on_preview`, `_on_preview_detection_finished`, `_update_detection_stats` |

Expected `main_window.py` after Step 4: ≤ 5,000 lines.

### Step 5 — Orchestrator Assembly

The remaining MainWindow methods are partitioned into 3 orchestrators:

**`TrackingOrchestrator`** (~80 methods):
`start_tracking`, `start_backward_tracking`, `start_preview_on_video`, `stop_tracking`, `start_full`, `_request_qthread_stop`, `_stop_csv_writer`, `on_tracking_finished`, `_finish_tracking_session`, `_finalize_tracking_session_ui`, `merge_and_save_trajectories`, `on_merge_finished`, `on_merge_error`, `on_merge_progress`, `_generate_video_from_trajectories`, `_generate_interpolated_individual_crops`, `_start_pending_oriented_track_video_export`, `_on_oriented_track_videos_finished`, `save_trajectories_to_csv`, `on_new_frame`, `on_stats_update`, `on_progress_update`, `on_tracking_warning`, and helpers.

**`ConfigOrchestrator`** (~80 methods):
`load_config`, `_load_config_from_file`, `save_config`, `_resolve_config_save_path`, `_atomic_json_write`, `get_parameters_dict`, `_populate_preset_combo`, `_populate_compute_runtime_options`, `_load_selected_preset`, `_save_custom_preset`, `_load_default_preset_on_startup`, `_load_advanced_config`, `_save_advanced_config`, `_setup_video_file`, `crop_video_to_roi`, `_calculate_roi_bounding_box`, `_estimate_roi_efficiency`, `_update_roi_optimization_info`, `_invalidate_roi_cache`, `_find_or_plan_optimizer_cache_path`, `_build_optimizer_detection_cache`, `_apply_optimized_params`, and helpers.

**`SessionOrchestrator`** (~50 methods):
`_apply_ui_state`, `_set_ui_controls_enabled`, `_collect_preview_controls`, `_set_interactive_widgets_enabled`, `_set_video_interaction_enabled`, `_sync_contextual_controls`, `_refresh_progress_visibility`, `_is_worker_running`, `_has_active_progress_task`, `_prepare_tracking_display`, `_show_video_logo_placeholder`, `_is_visualization_enabled`, `_setup_session_logging`, `_cleanup_session_logging`, `_cleanup_thread_reference`, `_cleanup_temporary_files`, `_show_session_summary`, `_disable_spinbox_wheel_events`, `_connect_parameter_signals`, `_load_ui_settings`, `_save_ui_settings`, `_restore_ui_state`, `_queue_ui_state_save`, `_remember_collapsible_state`, and helpers.

Expected `main_window.py` after Step 5: ≤ 1,000 lines.

### Step 6 — MainWindow Thinning

Delete remaining dead code (duplicate helper stubs, forwarding methods). Confirm `main_window.py` ≤ 600 lines.

---

## Testing Strategy

### Verification Gate (after every step)

```bash
# Import smoke test
conda run -n hydra-mps python -c \
  "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"

# Full suite
conda run -n hydra-mps python -m pytest tests/ -x -q \
  --ignore=tests/test_confidence_density.py \
  --ignore=tests/test_correction_writer.py \
  --ignore=tests/test_tag_identity.py
```

No step begins until both pass.

### Test File Updates

`test_main_window_config_persistence.py` calls `window.load_config(...)` and `window.save_config(...)`. After Step 5, update to `window._config_orch.load_config(...)` etc.

### New Smoke Tests (`tests/test_trackerkit_orchestrators_smoke.py`)

```python
def test_tracking_orchestrator_constructed(main_window):
    assert main_window._tracking_orch is not None

def test_config_orchestrator_constructed(main_window):
    assert main_window._config_orch is not None

def test_session_orchestrator_constructed(main_window):
    assert main_window._session_orch is not None
```

---

## Completion Criteria

| Step | Milestone | `main_window.py` target | Tests |
|---|---|---|---|
| 1 | 5 workers in `workers/` | ≤ 14,500 lines | 841+ pass |
| 2 | 4 widget classes in `widgets/` | ≤ 13,900 lines | 841+ pass |
| 3 | 4 dialogs on `BaseDialog`, `_TestWorker` on `BaseWorker` | — | 841+ pass |
| 4a–4f | All 6 panels have their handlers | ≤ 5,000 lines | 841+ pass |
| 5 | 3 orchestrators assembled | ≤ 1,000 lines | 841+ pass |
| 6 | MainWindow thinned | ≤ 600 lines | 841+ pass |

Final state:
- [ ] `main_window.py` ≤ 600 lines
- [ ] No worker class defined in `main_window.py`
- [ ] No widget utility class defined in `main_window.py`
- [ ] All dialogs inherit `BaseDialog`; all embedded workers inherit `BaseWorker`
- [ ] All panel handler methods live in their panel files
- [ ] Cross-panel logic lives in one of the 3 orchestrator classes
- [ ] 841+ tests collected, all passing
- [ ] Import smoke test passes after every step

---

## Out of Scope

- Making panel widget attributes private (underscore prefix) — deferred to Phase 3
- Removing `_main_window` back-reference from panels — deferred to Phase 3
- Moving `CollapsibleGroupBox` etc. to `hydra_suite/widgets/` (shared layer) — trackerkit-local for now
- Refactoring any other kit
- New features of any kind
- Changes to public CLI entry points

---

## Files Created or Modified

**New directories:**
- `src/hydra_suite/trackerkit/gui/workers/`
- `src/hydra_suite/trackerkit/gui/widgets/`
- `src/hydra_suite/trackerkit/gui/orchestrators/`

**New files:**
- `src/hydra_suite/trackerkit/gui/workers/__init__.py`
- `src/hydra_suite/trackerkit/gui/workers/merge_worker.py`
- `src/hydra_suite/trackerkit/gui/workers/crops_worker.py`
- `src/hydra_suite/trackerkit/gui/workers/video_worker.py`
- `src/hydra_suite/trackerkit/gui/workers/dataset_worker.py`
- `src/hydra_suite/trackerkit/gui/workers/preview_worker.py`
- `src/hydra_suite/trackerkit/gui/widgets/__init__.py`
- `src/hydra_suite/trackerkit/gui/widgets/collapsible.py`
- `src/hydra_suite/trackerkit/gui/widgets/help_label.py`
- `src/hydra_suite/trackerkit/gui/widgets/tooltip_button.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/__init__.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/tracking.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/config.py`
- `src/hydra_suite/trackerkit/gui/orchestrators/session.py`
- `tests/test_trackerkit_orchestrators_smoke.py`

**Modified:**
- `src/hydra_suite/trackerkit/gui/main_window.py` — workers/widgets/handlers removed; orchestrators instantiated; signal wiring consolidated
- `src/hydra_suite/trackerkit/gui/panels/*.py` — handlers added; imports updated to local `widgets/`
- `src/hydra_suite/trackerkit/gui/dialogs/*.py` — migrated to `BaseDialog`
- `tests/test_main_window_config_persistence.py` — config method refs updated to orchestrator-qualified paths
