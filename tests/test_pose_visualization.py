from __future__ import annotations

from tests.helpers.module_loader import load_src_module

mod = load_src_module(
    "hydra_suite/utils/pose_visualization.py",
    "pose_visualization_under_test",
)


def test_zero_confidence_is_never_renderable_even_at_zero_threshold() -> None:
    assert not mod.is_renderable_pose_keypoint(10.0, 20.0, 0.0, 0.0)


def test_nan_confidence_is_never_renderable() -> None:
    assert not mod.is_renderable_pose_keypoint(10.0, 20.0, float("nan"), 0.0)


def test_non_finite_coordinates_are_not_renderable() -> None:
    assert not mod.is_renderable_pose_keypoint(float("inf"), 20.0, 0.8, 0.2)
    assert not mod.is_renderable_pose_keypoint(10.0, float("nan"), 0.8, 0.2)


def test_positive_confidence_must_meet_threshold() -> None:
    assert mod.is_renderable_pose_keypoint(10.0, 20.0, 0.8, 0.2)
    assert not mod.is_renderable_pose_keypoint(10.0, 20.0, 0.19, 0.2)


def test_render_min_conf_is_sanitized() -> None:
    assert mod.normalize_pose_render_min_conf(float("nan")) == 0.2
    assert mod.normalize_pose_render_min_conf(-1.0) == 0.0
