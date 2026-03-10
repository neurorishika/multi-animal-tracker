"""Tests for the density-aware assignment helper function."""

import pytest

from multi_tracker.core.tracking.worker import get_assignment_params_for_region


def test_open_field_returns_default_params():
    """Detections in open field use standard assignment distance."""
    params = get_assignment_params_for_region(
        region_label="open_field",
        base_max_distance=50.0,
        conservative_factor=0.5,
    )
    assert params["max_distance"] == pytest.approx(50.0)


def test_flagged_region_tightens_distance():
    """Detections in a flagged region get reduced max assignment distance."""
    params = get_assignment_params_for_region(
        region_label="region-1",
        base_max_distance=50.0,
        conservative_factor=0.5,
    )
    assert params["max_distance"] < 50.0
    assert params["max_distance"] == pytest.approx(25.0)


def test_conservative_factor_zero_means_no_change():
    """conservative_factor=0 disables density-aware adjustment."""
    params = get_assignment_params_for_region(
        region_label="region-1",
        base_max_distance=50.0,
        conservative_factor=0.0,
    )
    assert params["max_distance"] == pytest.approx(50.0)


def test_any_region_label_triggers_tightening():
    """Any non-open_field label causes tightening."""
    params = get_assignment_params_for_region(
        region_label="region-42",
        base_max_distance=100.0,
        conservative_factor=0.3,
    )
    assert params["max_distance"] == pytest.approx(70.0)
