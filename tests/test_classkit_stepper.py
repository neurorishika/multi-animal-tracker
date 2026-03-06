"""Tests for FactorStepper state machine — no Qt event loop needed."""

import pytest

from multi_tracker.classkit.config.schemas import Factor, LabelingScheme


def make_scheme(factor_labels: list[list[str]]) -> LabelingScheme:
    return LabelingScheme(
        name="test",
        factors=[
            Factor(name=f"f{i}", labels=labels)
            for i, labels in enumerate(factor_labels)
        ],
        training_modes=["flat_tiny"],
    )


def test_stepper_advances_on_pick():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    assert state.current_factor_index == 0
    assert not state.is_complete

    state.pick("a")
    assert state.current_factor_index == 1
    assert not state.is_complete

    state.pick("x")
    assert state.is_complete
    assert state.composite_label == "a|x"


def test_stepper_back():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    state.pick("a")
    state.back()
    assert state.current_factor_index == 0
    assert state.picks == []


def test_stepper_reset():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    state.pick("a")
    state.pick("x")
    assert state.is_complete
    state.reset()
    assert state.current_factor_index == 0
    assert not state.is_complete
    assert state.picks == []


def test_stepper_single_factor_complete_immediately():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["young", "old"]])
    state = StepperState(scheme)
    state.pick("young")
    assert state.is_complete
    assert state.composite_label == "young"


def test_stepper_invalid_pick_raises():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"]])
    state = StepperState(scheme)
    with pytest.raises(ValueError, match="not in labels"):
        state.pick("invalid")


def test_stepper_back_at_start_does_nothing():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"]])
    state = StepperState(scheme)
    state.back()  # should not raise
    assert state.current_factor_index == 0


def test_stepper_composite_label_raises_when_incomplete():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"], ["x", "y"]])
    state = StepperState(scheme)
    state.pick("a")
    with pytest.raises(RuntimeError, match="not complete"):
        _ = state.composite_label


def test_stepper_pick_when_complete_raises():
    from multi_tracker.classkit.gui.widgets.factor_stepper import StepperState

    scheme = make_scheme([["a", "b"]])
    state = StepperState(scheme)
    state.pick("a")
    with pytest.raises(RuntimeError, match="already picked"):
        state.pick("a")
