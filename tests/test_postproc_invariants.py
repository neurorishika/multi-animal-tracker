from __future__ import annotations

import pandas as pd
import pytest

from tests.helpers.postproc_runner import (
    available_cases,
    load_resolve_fixture,
    normalize_output,
    run_resolve_trajectories,
)

RESOLVE_CASES = available_cases("resolve")


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_frame_bounds_invariant(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    if actual.empty:
        return

    input_df = pd.concat([forward, backward], ignore_index=True, sort=False)
    if input_df.empty:
        return

    min_frame = int(input_df["FrameID"].min())
    max_frame = int(input_df["FrameID"].max())
    assert int(actual["FrameID"].min()) >= min_frame
    assert int(actual["FrameID"].max()) <= max_frame


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_no_duplicate_trajectory_frame_pairs(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    if actual.empty:
        return

    dup_mask = actual.duplicated(subset=["TrajectoryID", "FrameID"], keep=False)
    assert not dup_mask.any(), (
        f"Found duplicate (TrajectoryID, FrameID) rows in case {case_name}: "
        f"{actual.loc[dup_mask, ['TrajectoryID', 'FrameID']].to_dict(orient='records')}"
    )


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_row_and_trajectory_counts_match_baseline(case_name: str) -> None:
    forward, backward, params, expected = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    actual_n = normalize_output(actual)
    expected_n = normalize_output(expected)

    assert len(actual_n) == len(expected_n)
    assert actual_n["TrajectoryID"].nunique(dropna=True) == expected_n[
        "TrajectoryID"
    ].nunique(dropna=True)


@pytest.mark.parametrize("case_name", RESOLVE_CASES)
def test_normalization_is_deterministic(case_name: str) -> None:
    forward, backward, params, _ = load_resolve_fixture(case_name)
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    first = normalize_output(actual)
    second = normalize_output(actual)
    assert first.equals(second)
