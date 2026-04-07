# Panel Extraction Sprint 2 — Design Spec

**Date:** 2026-04-04
**Author:** Rishika Mohanta
**Status:** Approved
**Predecessor:** `2026-04-04-codebase-simplification-design.md` (Slice 4, Phase 1)

---

## Goal

Extract the 6 `setup_*_ui()` methods from `trackerkit/gui/main_window.py` into the 6 stub panel classes that were scaffolded in Sprint 1. After this sprint, `main_window.py` shrinks from 19,975 lines to approximately 16,000 lines. The panels package becomes real — each panel owns its widget tree and exposes every widget as a public attribute. MainWindow business logic continues to access panel widgets via `self._panel_name.widget_name` rather than the old `self.widget_name`.

This is **Phase 1 of Slice 4** from the parent simplification spec. No business logic moves in this sprint. No signal handler methods move. Only the UI construction code migrates.

---

## Measured Starting State

From static analysis of `main_window.py` (19,975 lines):

| Panel class | MainWindow method | Source lines | Public widgets assigned | External refs |
|---|---|---|---|---|
| `SetupPanel` | `setup_setup_ui` | 4341–5037 (697 lines) | 58 | ~247 |
| `DetectionPanel` | `setup_detection_ui` | 5038–6051 (1,014 lines) | 73 | ~293 |
| `TrackingPanel` | `setup_tracking_ui` | 6052–6846 (795 lines) | 48 | ~158 |
| `PostProcessPanel` | `setup_data_ui` | 6847–7447 (601 lines) | 67 | ~230 |
| `DatasetPanel` | `setup_dataset_ui` | 7448–7827 (380 lines) | 30 | ~87 |
| `IdentityPanel` | `setup_individual_analysis_ui` | 7828–8324 (497 lines) | 51 | ~188 |
| **Total** | | **3,984 lines** | **327 widgets** | **~1,203 refs** |

Key observations from analysis:
- No widget name is shared across panels (zero name collisions).
- No setup method references a widget owned by another setup method (panels are build-time independent).
- Each setup method references `self.tab_X` exactly once — the very first line (`QVBoxLayout(self.tab_X)`) — to attach its layout to the tab container.
- Signal connections inside setup methods all point to `self.handler_name` on MainWindow. There are 33, 16, 2, 7, 6, and 21 connections respectively across the six panels.
- Helper methods called during build (`_create_help_label`, `_set_compact_scroll_layout`, `_set_compact_section_widget`, `_remember_collapsible_state`, `_populate_preset_combo`, etc.) live in MainWindow and must remain callable during panel construction.
- `CollapsibleGroupBox`, `AccordionContainer`, and `CompactHelpLabel` are defined in `main_window.py` above the `MainWindow` class (lines 3190, 3345, 3365). They are used in the setup methods and must be importable from `main_window.py` during panel construction.
- `DetectionPanel` uses module-level flags `TORCH_CUDA_AVAILABLE`, `MPS_AVAILABLE`, `TENSORRT_AVAILABLE` imported at the top of `main_window.py`.
- Tests in `test_main_window_config_persistence.py` access widgets by `window.widget_name`. After extraction those references must become `window._panel_name.widget_name`, which means the test file also needs updating per panel.

---

## Why Option C (Phased Extraction) Is the Right Tradeoff

Three options exist for this extraction:

**Option A — Panels set attributes on MainWindow.** Each `_build_ui()` receives `main_window` and does `main_window.widget_name = QWidget(...)`. This keeps the 1,203 external references in MainWindow unchanged but is an anti-pattern: panels mutate a foreign object's namespace, which defeats the purpose of encapsulation and makes it impossible to test panels independently.

**Option B — Full ownership migration in one pass.** Panels own widgets as private attributes, and all 1,203 references in MainWindow's 16,000-line business logic section are updated to `self._panel.widget_name` in the same commit. This is architecturally correct but represents a ~1,600-line diff in business logic code that has no regression coverage. One missed reference causes a silent `AttributeError` at runtime. The risk of introducing regressions without detection is unacceptably high.

**Option C — Phased.** Panels own widgets as *public* attributes (no underscore prefix). Phase 1 (this sprint) moves only the UI construction code. After each panel extraction, every `self.widget_name` reference in the remaining MainWindow code is mechanically rewritten to `self._panel_name.widget_name`. The attribute still exists, it is just accessed through the panel. The existing test suite runs after each panel, providing regression signal before moving on to the next. Phase 2 (a future sprint) moves business logic methods into panels and converts public widget attributes to private ones.

Option C is the right tradeoff because:
1. It is mechanically safe — each individual panel extraction is an independent, reviewable, testable change.
2. It preserves test coverage signal throughout the sprint.
3. It does not require the 16,000-line business logic section to be understood or modified beyond mechanical search-and-replace.
4. It leaves a clean seam for Phase 2.

---

## Architecture

### Panel Constructor Signature

All six panels follow this constructor pattern (update the Sprint 1 stubs to match):

```python
def __init__(self, main_window: "MainWindow", config: TrackerConfig,
             parent: QWidget | None = None) -> None:
    super().__init__(parent)
    self._main_window = main_window
    self._config = config
    self._build_ui()
```

`main_window` is passed so that signal connections (`.connect(self._main_window.some_handler)`) work identically to the original code. This is not a long-term dependency — Phase 2 will move the handlers into the panels and remove `_main_window`.

### Widget Attribute Visibility

All widgets created in `_build_ui()` are assigned as **public** attributes (`self.widget_name = ...`, no underscore). This is intentional: MainWindow must still reach them via `self._panel.widget_name` throughout Sprint 2. Private attributes (`self._widget_name`) happen in Phase 2 when the accessing business logic moves into the panel.

### Helper Method Access

The six panels will call MainWindow helper methods during `_build_ui()`:
- `self._main_window._create_help_label(...)`
- `self._main_window._set_compact_scroll_layout(...)`
- `self._main_window._set_compact_section_widget(...)`
- `self._main_window._remember_collapsible_state(...)`

These helpers are utility methods with no side effects on MainWindow state. They remain on MainWindow for now.

### Custom Widget Classes

`CollapsibleGroupBox`, `AccordionContainer`, and `CompactHelpLabel` are defined at module level in `main_window.py`. Panels import them directly:

```python
from hydra_suite.trackerkit.gui.main_window import (
    CollapsibleGroupBox,
    AccordionContainer,
    CompactHelpLabel,
)
```

These will move to `trackerkit/gui/widgets/` in a later sprint.

### Module-Level Flags (DetectionPanel Only)

`DetectionPanel._build_ui()` uses `TORCH_CUDA_AVAILABLE`, `MPS_AVAILABLE`, and `TENSORRT_AVAILABLE`. Import them in `detection_panel.py`:

```python
from hydra_suite.utils.gpu_utils import (
    MPS_AVAILABLE,
    TENSORRT_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)
```

### How MainWindow Changes

In `init_ui()`, the six blocks that currently do:

```python
self.tab_setup = QWidget()
self.setup_setup_ui()
self.tabs.addTab(self.tab_setup, "Get Started")
```

become:

```python
self._setup_panel = SetupPanel(main_window=self, config=self.config)
self.tabs.addTab(self._setup_panel, "Get Started")
```

The `tab_X` attributes disappear (they are no longer needed — the panel widget is the tab). The `setup_*_ui()` method bodies are deleted from `main_window.py`.

All remaining `self.widget_name` references throughout `main_window.py` are rewritten to `self._panel_name.widget_name` after each extraction.

---

## Per-Panel Extraction Scope

Extraction order follows risk-first (smallest/simplest panels first to validate the pattern).

### Step 1 — DatasetPanel (smallest: 380 lines, 30 widgets)

**Source:** `setup_dataset_ui` (lines 7448–7827)
**Target:** `trackerkit/gui/panels/dataset_panel.py`
**Panel attribute on MainWindow:** `self._dataset_panel`
**Tab label:** `"Build Dataset"`

Body of `setup_dataset_ui` moves verbatim into `DatasetPanel._build_ui()`. The first line `layout = QVBoxLayout(self.tab_dataset)` becomes `layout = self._layout`. After move: delete `setup_dataset_ui` from `main_window.py`, replace the 3-line tab block with 2-line panel instantiation, rewrite ~87 external references from `self.widget` to `self._dataset_panel.widget`, remove `self.tab_dataset` attribute.

`combo_xanylabeling_env` is referenced in `test_main_window_config_persistence.py` — update to `window._dataset_panel.combo_xanylabeling_env`.

**Expected MainWindow line count after Step 1:** ≤ 19,600 lines.

### Step 2 — SetupPanel (697 lines, 58 widgets)

**Source:** `setup_setup_ui` (lines 4341–5037)
**Target:** `trackerkit/gui/panels/setup_panel.py`
**Panel attribute on MainWindow:** `self._setup_panel`
**Tab label:** `"Get Started"`

Signal connections reference handlers including `_load_selected_preset`, `_save_custom_preset`, `_on_preset_selection_changed`, `_add_videos_to_batch`, `_clear_batch`, `_detect_fps_from_current_video`. All connected as `self._main_window.handler_name`.

The `_populate_preset_combo()` and `_populate_compute_runtime_options()` calls at the end of `setup_setup_ui` (lines 5035–5036) become `self._main_window._populate_preset_combo()` etc., or are called from `MainWindow.__init__` after panel construction.

**Expected MainWindow line count after Step 2:** ≤ 18,900 lines.

### Step 3 — TrackingPanel (795 lines, 48 widgets)

**Source:** `setup_tracking_ui` (lines 6052–6846)
**Target:** `trackerkit/gui/panels/tracking_panel.py`
**Panel attribute on MainWindow:** `self._tracking_panel`
**Tab label:** `"Track Movement"`

Only 2 signal connections (to `_open_parameter_helper` and `_on_confidence_density_map_toggled`). This is the simplest connection surface of the larger panels.

`chk_enable_confidence_density_map` and `g_density` are referenced in `test_main_window_config_persistence.py` — update to `window._tracking_panel.*`.

**Expected MainWindow line count after Step 3:** ≤ 18,100 lines.

### Step 4 — PostProcessPanel (601 lines, 67 widgets)

**Source:** `setup_data_ui` (lines 6847–7447)
**Target:** `trackerkit/gui/panels/postprocess_panel.py`
**Panel attribute on MainWindow:** `self._postprocess_panel`
**Tab label:** `"Clean Results"`

Signal handlers: `_on_cleaning_toggled`, `_on_video_output_toggled`, `_open_refinekit`, `_select_video_pose_color`, `select_video_output`.

**Expected MainWindow line count after Step 4:** ≤ 17,500 lines.

### Step 5 — IdentityPanel (497 lines, 51 widgets)

**Source:** `setup_individual_analysis_ui` (lines 7828–8324)
**Target:** `trackerkit/gui/panels/identity_panel.py`
**Panel attribute on MainWindow:** `self._identity_panel`
**Tab label:** `"Analyze Individuals"`

Largest signal surface (21 connections). Handlers include `_add_cnn_classifier_row`, `_on_identity_analysis_toggled`, `_on_pose_analysis_toggled`, `_on_runtime_context_changed`, `_refresh_pose_sleap_envs`.

`combo_yolo_headtail_model` and `combo_yolo_headtail_model_type` are referenced in `test_main_window_config_persistence.py` — update to `window._identity_panel.*`.

The `_sync_individual_analysis_mode_ui()` call that currently follows `self.setup_individual_analysis_ui()` in `init_ui` must remain as `self._sync_individual_analysis_mode_ui()` in MainWindow after panel instantiation.

**Expected MainWindow line count after Step 5:** ≤ 17,000 lines.

### Step 6 — DetectionPanel (1,014 lines, 73 widgets) — largest, saved for last

**Source:** `setup_detection_ui` (lines 5038–6051)
**Target:** `trackerkit/gui/panels/detection_panel.py`
**Panel attribute on MainWindow:** `self._detection_panel`
**Tab label:** `"Find Animals"`

Additional imports: `TORCH_CUDA_AVAILABLE`, `MPS_AVAILABLE`, `TENSORRT_AVAILABLE` from `hydra_suite.utils.gpu_utils`.

Signal handlers: `_on_brightness_changed`, `_on_contrast_changed`, `_on_gamma_changed`, `_on_tensorrt_toggled`, `_on_yolo_mode_changed`, `_open_bg_parameter_helper`, `_update_body_size_info`, `on_yolo_model_changed`, `_refresh_yolo_detect_model_combo`, `_refresh_yolo_crop_obb_model_combo`, `_refresh_yolo_model_combo`.

**Expected MainWindow line count after Step 6:** ≤ 16,050 lines.

---

## Widget Reference Update Strategy

After each panel's `_build_ui()` is populated and the `setup_*_ui()` method is deleted from `main_window.py`, all remaining `self.widget_name` references in `main_window.py` must be updated to `self._panel_name.widget_name`.

### Mechanical Process (Per Panel)

1. Collect the full set of public widget names assigned in the extracted panel (use `grep -n "^\s*self\.[a-z_]* =" panel_file.py | sed 's/.*self\.\([a-z_]*\) = .*/\1/' | sort -u`).

2. For each name, apply a whole-word substitution in `main_window.py` only:
   `self.widget_name` → `self._panel_name.widget_name`

   Use whole-word matching to avoid false positives (e.g., `self.spin_fps` must not match `self.spin_fps_target`).

3. Do NOT substitute inside the panel file itself — panel code uses `self.widget_name` where `self` is the panel.

4. Handle `self.tab_X` removal: after extraction, delete the 3-line tab creation block from `init_ui`.

### Verification After Each Panel

```bash
# Import smoke test — catches missing attributes at module load
conda run -n hydra-mps python -c "from hydra_suite.trackerkit.gui.main_window import MainWindow; print('ok')"

# Full test suite
conda run -n hydra-mps python -m pytest tests/ -x -q --ignore=tests/test_confidence_density.py --ignore=tests/test_correction_writer.py --ignore=tests/test_tag_identity.py
```

A successful import confirms no broken cross-module references. Any new test failure must be resolved before starting the next panel.

---

## Test Strategy

### Regression Baseline

Before starting, confirm the baseline: 841 tests collected, 1 deselected (benchmark), with 3 collection errors in pre-existing files (`test_correction_writer.py`, `test_tag_identity.py`, `test_confidence_density.py`).

### Required Test Updates Per Panel

| Panel | Test file | Widget refs to update |
|---|---|---|
| `DatasetPanel` | `test_main_window_config_persistence.py` | `window.combo_xanylabeling_env` |
| `TrackingPanel` | `test_main_window_config_persistence.py` | `window.chk_enable_confidence_density_map`, `window.g_density` |
| `IdentityPanel` | `test_main_window_config_persistence.py` | `window.combo_yolo_headtail_model`, `window.combo_yolo_headtail_model_type` |

Test updates happen in the same commit as the panel extraction.

### Optional: Panel Smoke Tests

Add `tests/test_trackerkit_panels_smoke.py` with one test per panel:

```python
def test_setup_panel_builds_without_error(qapp, main_window):
    assert main_window._setup_panel is not None
    assert hasattr(main_window._setup_panel, "combo_presets")
```

---

## Completion Criteria

| Step | Panel | Method deleted | MainWindow target | Tests pass |
|---|---|---|---|---|
| 1 | `DatasetPanel` | `setup_dataset_ui` | ≤ 19,600 lines | 841+ collected |
| 2 | `SetupPanel` | `setup_setup_ui` | ≤ 18,900 lines | 841+ collected |
| 3 | `TrackingPanel` | `setup_tracking_ui` | ≤ 18,100 lines | 841+ collected |
| 4 | `PostProcessPanel` | `setup_data_ui` | ≤ 17,500 lines | 841+ collected |
| 5 | `IdentityPanel` | `setup_individual_analysis_ui` | ≤ 17,000 lines | 841+ collected |
| 6 | `DetectionPanel` | `setup_detection_ui` | ≤ 16,050 lines | 841+ collected |

Final state:
- [ ] All 6 `setup_*_ui()` methods deleted from `main_window.py`
- [ ] All 6 `tab_X = QWidget()` attributes deleted from `init_ui()`
- [ ] All 6 panels are instantiated in `init_ui()` and added directly to `self.tabs`
- [ ] All 327 public widget attributes are accessible as `self._panel_name.widget_name` from MainWindow
- [ ] `main_window.py` is under 16,100 lines
- [ ] 841+ tests collected, all passing
- [ ] `test_main_window_config_persistence.py` references updated to panel-qualified widget paths

---

## Out of Scope for This Sprint

- Moving any business logic method (event handlers, workflows) from MainWindow into panels
- Making panel widget attributes private (underscore prefix)
- Moving `CollapsibleGroupBox`, `AccordionContainer`, `CompactHelpLabel` out of `main_window.py`
- Moving helper methods (`_create_help_label`, `_set_compact_*`) out of MainWindow
- Removing `_main_window` back-reference from panels (deferred to Phase 2)
- Any changes to workers, dialogs, config schemas, or other kits

---

## Files Created or Modified

**Modified:**
- `src/hydra_suite/trackerkit/gui/main_window.py` — 6 setup methods deleted, `init_ui()` updated, ~1,203 widget references updated
- `tests/test_main_window_config_persistence.py` — 3–5 widget references updated to panel-qualified paths

**Populated (bodies added to Sprint 1 stubs):**
- `src/hydra_suite/trackerkit/gui/panels/setup_panel.py`
- `src/hydra_suite/trackerkit/gui/panels/detection_panel.py`
- `src/hydra_suite/trackerkit/gui/panels/tracking_panel.py`
- `src/hydra_suite/trackerkit/gui/panels/postprocess_panel.py`
- `src/hydra_suite/trackerkit/gui/panels/dataset_panel.py`
- `src/hydra_suite/trackerkit/gui/panels/identity_panel.py`

**Created (optional):**
- `tests/test_trackerkit_panels_smoke.py`
