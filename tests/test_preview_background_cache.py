from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from multi_tracker.core.background import model as background_model_module
from multi_tracker.gui import main_window


class _FakeVideoCapture:
    open_count = 0
    release_count = 0

    def __init__(self, _path: str):
        type(self).open_count += 1

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        type(self).release_count += 1


class _FakeBackgroundModel:
    prime_calls = 0

    def __init__(self, params: dict):
        self.params = params
        self.lightest_background = None
        self.adaptive_background = None
        self.reference_intensity = None

    def prime_background(self, _cap) -> None:
        type(self).prime_calls += 1
        value = float(type(self).prime_calls)
        self.lightest_background = np.full((3, 3), value, dtype=np.float32)
        self.adaptive_background = np.full((3, 3), value + 10.0, dtype=np.float32)
        self.reference_intensity = value + 20.0


@pytest.fixture(autouse=True)
def _reset_preview_background_cache(monkeypatch: pytest.MonkeyPatch):
    main_window._clear_preview_background_cache()
    _FakeVideoCapture.open_count = 0
    _FakeVideoCapture.release_count = 0
    _FakeBackgroundModel.prime_calls = 0
    monkeypatch.setattr(main_window.cv2, "VideoCapture", _FakeVideoCapture)
    monkeypatch.setattr(
        background_model_module, "BackgroundModel", _FakeBackgroundModel
    )
    yield
    main_window._clear_preview_background_cache()


def _make_context(**overrides) -> dict:
    context = {
        "video_path": "/tmp/preview-cache-test.mp4",
        "bg_prime_frames": 30,
        "brightness": 5,
        "contrast": 1.15,
        "gamma": 0.95,
        "roi_mask": np.array([[0, 255], [255, 255]], dtype=np.uint8),
        "resize_factor": 0.5,
        "dark_on_light": False,
        "threshold_value": 20,
        "morph_kernel_size": 3,
        "enable_additional_dilation": False,
        "dilation_kernel_size": 3,
        "dilation_iterations": 1,
        "enable_conservative_split": True,
        "conservative_kernel_size": 3,
        "conservative_erode_iterations": 1,
        "max_targets": 5,
        "min_contour": 50,
        "max_contour_multiplier": 20,
    }
    context.update(overrides)
    return context


def test_preview_background_cache_reuses_primed_state_for_non_priming_changes() -> None:
    context = _make_context()

    first_model, first_params = main_window._build_preview_background_model(context)

    assert _FakeBackgroundModel.prime_calls == 1
    assert _FakeVideoCapture.open_count == 1
    assert _FakeVideoCapture.release_count == 1
    assert first_params["THRESHOLD_VALUE"] == 20

    first_model.lightest_background[:, :] = 99.0
    first_model.adaptive_background[:, :] = 199.0

    reused_context = _make_context(threshold_value=42, morph_kernel_size=7)
    reused_model, reused_params = main_window._build_preview_background_model(
        reused_context
    )

    assert _FakeBackgroundModel.prime_calls == 1
    assert _FakeVideoCapture.open_count == 1
    assert _FakeVideoCapture.release_count == 1
    assert reused_params["THRESHOLD_VALUE"] == 42
    assert reused_params["MORPH_KERNEL_SIZE"] == 7
    assert np.all(reused_model.lightest_background == 1.0)
    assert np.all(reused_model.adaptive_background == 11.0)


def test_preview_background_cache_invalidates_when_priming_settings_change() -> None:
    main_window._build_preview_background_model(_make_context())
    main_window._build_preview_background_model(_make_context(brightness=9))

    assert _FakeBackgroundModel.prime_calls == 2
    assert _FakeVideoCapture.open_count == 2
    assert _FakeVideoCapture.release_count == 2
