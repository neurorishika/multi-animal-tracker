from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers.postproc_runner import (
    assert_equivalent,
    available_cases,
    load_interpolate_fixture,
    load_resolve_fixture,
    run_interpolate,
    run_resolve_trajectories,
)

RESOLVE_CASES = available_cases("resolve")
INTERPOLATE_CASES = available_cases("interpolate")
STRICT_TOL = {"abs": 1e-9, "rel": 1e-9}


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_resolve_equivalence(case_name: str) -> None:
    forward, backward, params, expected = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    assert_equivalent(actual, expected, tol=STRICT_TOL)


@pytest.mark.parametrize("case_name", INTERPOLATE_CASES)
def test_interpolate_equivalence(case_name: str) -> None:
    input_df, params, expected = load_interpolate_fixture(case_name)
    actual = run_interpolate(input_df, params)
    assert_equivalent(actual, expected, tol=STRICT_TOL)


def test_empty_input_resolve() -> None:
    actual = run_resolve_trajectories(
        {"forward": pd.DataFrame(), "backward": pd.DataFrame()},
        {
            "AGREEMENT_DISTANCE": 15.0,
            "MIN_OVERLAP_FRAMES": 2,
            "MIN_TRAJECTORY_LENGTH": 2,
        },
    )
    assert actual.empty


def test_stability_repeated_runs() -> None:
    case_name = "heavy_overlap_many_to_many"
    forward, backward, params, _ = load_resolve_fixture(case_name)
    baseline = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    for _ in range(5):
        actual = run_resolve_trajectories(
            {"forward": forward, "backward": backward}, params
        )
        assert_equivalent(actual, baseline, tol=STRICT_TOL)


def test_subset_forward_only_fixture_regression() -> None:
    forward, backward, params, expected = load_resolve_fixture("subset_forward_only")
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    assert_equivalent(actual, expected, tol=STRICT_TOL)


def test_subset_bidirectional_fixture_regression() -> None:
    forward, backward, params, expected = load_resolve_fixture("subset_bidirectional")
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    assert_equivalent(actual, expected, tol=STRICT_TOL)
