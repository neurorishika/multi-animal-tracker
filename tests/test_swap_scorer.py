# tests/test_swap_scorer.py
import numpy as np
import pandas as pd
import pytest

from multi_tracker.afterhours.core.swap_scorer import SwapScorer, SwapSuspicionEvent


def _make_df(track_id, frames, xs, ys, thetas=None, pose_quality=None):
    """Build a minimal trajectory DataFrame."""
    n = len(frames)
    d = {
        "TrajectoryID": track_id,
        "FrameID": frames,
        "X": xs,
        "Y": ys,
        "Theta": thetas if thetas is not None else [0.0] * n,
        "State": ["active"] * n,
    }
    if pose_quality is not None:
        d["PoseQualityScore"] = pose_quality
    return pd.DataFrame(d)


def test_swap_suspicion_event_fields():
    """SwapSuspicionEvent has required fields and sensible defaults."""
    ev = SwapSuspicionEvent(
        event_type="swap",
        involved_tracks=[1, 2],
        frame_peak=100,
        frame_range=(90, 110),
        score=0.75,
        signals=["Cr", "Hd"],
        region_label="open_field",
        region_boundary=False,
    )
    assert ev.score == pytest.approx(0.75)
    assert "Cr" in ev.signals
    assert ev.track_a == 1
    assert ev.track_b == 2


def test_scorer_returns_list():
    """SwapScorer.score returns a list (possibly empty) of events."""
    df = pd.concat(
        [
            _make_df(1, list(range(10)), list(range(10)), [0.0] * 10),
            _make_df(2, list(range(10)), [50.0] * 10, [0.0] * 10),
        ]
    )
    scorer = SwapScorer(regions=[])
    events = scorer.score(df)
    assert isinstance(events, list)


def test_scorer_ranks_by_score_descending():
    """Events are returned sorted highest score first."""
    # Two tracks that cross in the middle — should trigger a swap event
    frames = list(range(30))
    xs_a = list(range(30))  # moves right
    xs_b = list(range(29, -1, -1))  # moves left — they cross at frame 15
    df = pd.concat(
        [
            _make_df(1, frames, xs_a, [50.0] * 30),
            _make_df(2, frames, xs_b, [50.0] * 30),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.0)
    events = scorer.score(df)
    if len(events) >= 2:
        scores = [e.score for e in events]
        assert scores == sorted(scores, reverse=True)


def test_scorer_detects_crossing():
    """Two tracks that exchange X positions should be flagged."""
    frames = list(range(40))
    xs_a = [float(i) for i in range(40)]  # moves right: 0->39
    xs_b = [float(39 - i) for i in range(40)]  # moves left: 39->0
    df = pd.concat(
        [
            _make_df(1, frames, xs_a, [50.0] * 40),
            _make_df(2, frames, xs_b, [50.0] * 40),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.0, approach_distance=60.0)
    events = scorer.score(df)
    assert len(events) >= 1
    assert "Cr" in events[0].signals


def test_scorer_detects_proximity():
    """Two tracks that come close then separate should fire Pr signal."""
    frames = list(range(30))
    # Track A stays at x=50. Track B approaches from right, closest at frame 15 (dist=20), retreats
    xs_a = [50.0] * 30
    xs_b = [
        70.0 + abs(15 - i) * 8.0 for i in range(30)
    ]  # closest at frame 15: x=70, dist=20
    df = pd.concat(
        [
            _make_df(1, frames, xs_a, [50.0] * 30),
            _make_df(2, frames, xs_b, [50.0] * 30),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.0, approach_distance=40.0)
    events = scorer.score(df)
    assert len(events) >= 1
    # Should have Pr since they get close but don't cross
    assert any("Pr" in e.signals for e in events)


def test_scorer_filters_below_threshold():
    """Events below min_score are excluded from output."""
    df = pd.concat(
        [
            _make_df(1, list(range(10)), list(range(10)), [0.0] * 10),
            _make_df(2, list(range(10)), [500.0] * 10, [0.0] * 10),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.99)
    events = scorer.score(df)
    assert all(e.score >= 0.99 for e in events)


def test_scorer_heading_discontinuity():
    """A sudden heading reversal fires Hd signal."""
    frames = list(range(20))
    # Smooth heading then sudden 180 degree reversal at frame 10
    thetas = [0.0] * 10 + [np.pi] * 10
    # Two tracks close enough to interact
    df = pd.concat(
        [
            _make_df(
                1, frames, [50.0 + i for i in range(20)], [50.0] * 20, thetas=thetas
            ),
            _make_df(2, frames, [50.0 + i for i in range(20)], [52.0] * 20),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.0, approach_distance=20.0)
    events = scorer.score(df)
    assert any("Hd" in e.signals for e in events)


def test_scorer_no_swap_events_for_distant_tracks():
    """Tracks that are always far apart produce no SWAP events."""
    df = pd.concat(
        [
            _make_df(1, list(range(10)), [0.0] * 10, [0.0] * 10),
            _make_df(2, list(range(10)), [1000.0] * 10, [1000.0] * 10),
        ]
    )
    scorer = SwapScorer(regions=[], min_score=0.0, approach_distance=50.0)
    events = scorer.score(df)
    swap_events = [e for e in events if e.event_type.value == "swap"]
    assert len(swap_events) == 0
