from __future__ import annotations

import numpy as np
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


def test_merge_theta_avoids_orthogonal_artifact_for_pi_ambiguous_pairs() -> None:
    forward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 10.0,
                "Y": 10.0,
                "Theta": 0.0,
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 12.0,
                "Y": 10.0,
                "Theta": 0.0,
                "State": "active",
            },
        ]
    )
    backward = pd.DataFrame(
        [
            {
                "TrajectoryID": 0,
                "FrameID": 0,
                "X": 10.1,
                "Y": 10.0,
                "Theta": float(np.pi),
                "State": "active",
            },
            {
                "TrajectoryID": 0,
                "FrameID": 1,
                "X": 12.1,
                "Y": 10.0,
                "Theta": float(np.pi),
                "State": "active",
            },
        ]
    )
    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 1,
        "MIN_TRAJECTORY_LENGTH": 1,
    }

    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )
    assert not actual.empty

    theta = actual["Theta"].dropna().to_numpy(dtype=float)
    assert theta.size > 0

    # Treat theta as axis-valued for this invariant: result should align to 0/pi,
    # not collapse to orthogonal pi/2.
    dist_to_axis = np.minimum(
        np.abs(np.arctan2(np.sin(theta), np.cos(theta))),
        np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))),
    )
    assert float(np.max(dist_to_axis)) < 1e-6


def test_different_animals_crossing_are_not_merged_into_one_trajectory() -> None:
    """
    Two animals crossing paths should remain as separate trajectories after merging.

    Bug: _merge_overlapping_agreeing_trajectories only required `agreeing >= min_overlap`
    (count), not a ratio. Two different animals that are briefly close (e.g. at a
    crossing for 5+ frames) would be merged, producing a single trajectory that jumps
    between both animals' positions.
    """
    n = 100
    # Animal A moves right: x = 1..100, y = 0
    fwd_a = pd.DataFrame(
        {
            "TrajectoryID": 0,
            "FrameID": list(range(1, n + 1)),
            "X": [float(f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    bwd_a = fwd_a.copy()  # forward and backward agree perfectly for animal A

    # Animal B moves left: x = 100..1, y = 0
    fwd_b = pd.DataFrame(
        {
            "TrajectoryID": 1,
            "FrameID": list(range(1, n + 1)),
            "X": [float(n + 1 - f) for f in range(1, n + 1)],
            "Y": [0.0] * n,
            "Theta": [0.0] * n,
            "State": ["active"] * n,
        }
    )
    bwd_b = fwd_b.copy()

    forward = pd.concat([fwd_a, fwd_b], ignore_index=True)
    backward = pd.concat([bwd_a, bwd_b], ignore_index=True)

    params = {
        "AGREEMENT_DISTANCE": 5.0,
        "MIN_OVERLAP_FRAMES": 5,
        "MIN_TRAJECTORY_LENGTH": 5,
    }
    actual = run_resolve_trajectories(
        {"forward": forward, "backward": backward}, params
    )

    assert not actual.empty
    n_traj = actual["TrajectoryID"].nunique()
    assert n_traj == 2, (
        f"Expected 2 trajectories (one per animal), got {n_traj}. "
        "Different animals crossing paths should not be merged into one."
    )

    # Each trajectory should span all frames without large position jumps
    for traj_id, group in actual.groupby("TrajectoryID"):
        group = group.sort_values("FrameID")
        x = group["X"].dropna().to_numpy(dtype=float)
        if len(x) < 2:
            continue
        max_jump = float(np.max(np.abs(np.diff(x))))
        assert max_jump <= 5.0, (
            f"Trajectory {traj_id} has a jump of {max_jump:.1f} px — "
            "indicates two animals' data were incorrectly merged."
        )
